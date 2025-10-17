# ======================================================================
# Optimizer comparison: Muon vs AdamW vs PolarGrad
#  - Runs the same depth & dim sweeps for each optimizer
#  - Records: val loss vs steps, avg step time, peak memory
#  - Produces plots + a JSON log
# ======================================================================
import os, sys, time, json, math, itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt

from train_gpt import (
    GPT, Muon, PolarGrad,            # <-- uses your implementations in train_gpt.py
    distributed_data_generator, get_window_size_blocks,
    get_lr, Hyperparameters, norm, F
)

# ------------------------------ DDP setup ------------------------------
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
master_process = (rank == 0)

def print0(s, console=False):
    if master_process:
        if console: print(s)
        os.makedirs("logs", exist_ok=True)
        with open("logs/optimizer_compare_log.txt", "a") as f:
            print(s, file=f)

# ------------------------ GPT wrapper for sweeps -----------------------
class GPTForSweep(GPT):
    """Make forward robust to arbitrary depth (nL). Keeps your U-net + value-embed scheme."""
    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor, sliding_window_num_blocks: torch.Tensor):
        assert input_seq.ndim == 1
        nL = len(self.blocks)

        # Value embeddings: 3 front / 3 tail, else None for shallow nets
        ve_all = [ve(input_seq) for ve in self.value_embeds]
        if nL >= 6:
            front = [ve_all[0], ve_all[1], ve_all[2]]
            mid   = [None] * (nL - 6)
            tail  = [ve_all[0], ve_all[1], ve_all[2]]
            ve = (front + mid + tail)[:nL]
        else:
            ve = [None] * nL

        # Block masks: long every 4th and last
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm if (i % 4 == 0 or i == nL - 1) else short_bm for i in range(nL)]

        x = x0 = norm(self.embed(input_seq)[None])

        # U-net scalars sized to nL
        skip_connections = []
        skip_weights = self.scalars[:(nL // 2)]
        lambdas     = self.scalars[1 * nL : 3 * nL].view(-1, 2)
        sa_lambdas  = self.scalars[3 * nL : 5 * nL].view(-1, 2)

        n = nL // 2
        for i in range(nL):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        return loss

# ---------------------------- Base hyperparams -------------------------
args = Hyperparameters()
# Short/fair defaults for comparisons (tweak if needed)
args.num_iterations  = 300          # same as your Muon sweep
args.val_loss_every  = 100
args.save_checkpoint = False
args.cooldown_frac   = 0.5
args.val_tokens      = 1048576      # 1M validation tokens
args.train_seq_len   = 8192
args.val_seq_len     = 8192

# ------------------------------- Sweeps --------------------------------
depth_values = [4, 8, 12, 16]          # vary layers (fixed dim)
dim_values   = [384, 768, 1024, 1536]  # vary model_dim (fixed depth)

RUN_DEPTH_SWEEP = True
RUN_DIM_SWEEP   = True

FIXED_DIM   = 768   # for depth sweep
FIXED_DEPTH = 12    # for dim sweep

# -------------------------- Optimizer configs --------------------------
OPTIMIZERS = {
    "muon": {
        "two_d_lr": 0.06,
        "momentum": 0.95,
        "adamw_lr": 0.008,                # for scalars/embed/head (like your main)
        "adamw_betas": (0.8, 0.95),
        "adamw_eps": 1e-10,
        "adamw_wd": 0.0,
    },
    "adamw": {
        "lr": 0.0015,                     # conservative, stable across depths
        "betas": (0.9, 0.98),
        "eps": 1e-8,
        "wd": 0.01,                       # a bit of wd for AdamW baseline
    },
    "polargrad": {
        "two_d_lr": 0.06,
        "momentum": 0.95,
        "polar_iters": 5,
        "adamw_lr": 0.008,
        "adamw_betas": (0.8, 0.95),
        "adamw_eps": 1e-10,
        "adamw_wd": 0.0,
    },
}

# -------------------------- Utility / recording ------------------------
def to_mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)

def materialize_zero_grads(optim: torch.optim.Optimizer):
    """Ensure every param owned by the optimizer has a grad tensor."""
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

# Store all results here
all_results = []  # list of dict rows

# --------------------------- Single experiment -------------------------
def run_experiment(optimizer_name: str, num_layers: int, model_dim: int):
    cfg = OPTIMIZERS[optimizer_name]
    tag = f"{optimizer_name}|L={num_layers}|D={model_dim}"
    print0(f"\n=== Running {tag} ===", console=True)

    # Build model (heads = dim/128 keeps head_dim≈128)
    assert model_dim % 128 == 0, "model_dim must be a multiple of 128."
    model = GPTForSweep(
        vocab_size=50257,
        num_layers=num_layers,
        num_heads=(model_dim // 128),
        model_dim=model_dim,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
    ).cuda()
    # NOTE: to avoid Dynamo overhead/fragility during quick sweeps, skip torch.compile

    # Make embeddings bf16 like your main
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    # Sync params across ranks
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    # Param groups
    hidden_2d = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2]
    scalars   = [p for p in model.parameters() if p.ndim < 2]
    head_w    = [model.lm_head.weight]
    embeds    = [p for n, p in model.named_parameters() if "embed" in n]
    gates     = [p for n, p in model.named_parameters() if "gate" in n]
    two_d_all = hidden_2d + gates

    # Build optimizer(s)
    optimizers = []
    optimizer_labels = []  # to know which one to warmup for muon/polargrad

    if optimizer_name == "muon":
        # AdamW for non-2D bits, Muon for 2D + gates
        opt1 = torch.optim.AdamW(
            scalars + head_w + embeds,
            lr=cfg["adamw_lr"], betas=cfg["adamw_betas"], eps=cfg["adamw_eps"], weight_decay=cfg["adamw_wd"]
        )
        opt2 = Muon(two_d_all, lr=cfg["two_d_lr"], momentum=cfg["momentum"], weight_decay=0.0)
        optimizers = [opt1, opt2]
        optimizer_labels = ["adamw_misc", "muon_main"]

    elif optimizer_name == "adamw":
        # AdamW for ALL params (strong baseline)
        opt = torch.optim.AdamW(
            list(model.parameters()),
            lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"], weight_decay=cfg["wd"],
        )
        optimizers = [opt]
        optimizer_labels = ["adamw_all"]

    elif optimizer_name == "polargrad":
        # AdamW for non-2D bits, PolarGrad for 2D + gates
        opt1 = torch.optim.AdamW(
            scalars + head_w + embeds,
            lr=cfg["adamw_lr"], betas=cfg["adamw_betas"], eps=cfg["adamw_eps"], weight_decay=cfg["adamw_wd"]
        )
        opt2 = PolarGrad(
            two_d_all, lr=cfg["two_d_lr"], momentum=cfg["momentum"], weight_decay=0.0,
            custom_sizing=(dist.get_world_size()==8), polar_iters=cfg["polar_iters"]
        )
        optimizers = [opt1, opt2]
        optimizer_labels = ["adamw_misc", "polargrad_main"]

    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    # Stash initial LR for schedule
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    # Data
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=False)
    val_loader   = distributed_data_generator(args.val_files,   world_size * args.val_seq_len,   align_to_bos=False)

    # Reset CUDA peak mem stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    training_time_ms = 0

    # Record per-step val losses to plot curves later
    val_curve = []  # (step, val_loss)

    # Train short run
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        # ---- validation ----
        if last_step or (args.val_loss_every and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            model.eval()
            val_batch_size = world_size * args.val_seq_len
            val_steps = args.val_tokens // val_batch_size
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step))
            val_loss /= val_steps
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            model.train()

            if master_process:
                print0(f"[{optimizer_name}] L={num_layers} D={model_dim} "
                       f"step={step} val_loss={val_loss:.4f} "
                       f"time={training_time_ms/1000:.1f}s", console=True)
                val_curve.append((int(step), float(val_loss)))

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # ---- training ----
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()

        # LR schedule
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * get_lr(step)

        # Momentum warmup for Muon/PolarGrad second optimizer only
        if "muon_main" in optimizer_labels or "polargrad_main" in optimizer_labels:
            for label, opt in zip(optimizer_labels, optimizers):
                if label.endswith("_main"):  # muon_main or polargrad_main
                    frac = min(step / 300.0, 1.0)
                    for g in opt.param_groups:
                        g["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Ensure grads exist for Muon/PolarGrad param sets
        for label, opt in zip(optimizer_labels, optimizers):
            if label.endswith("_main"):
                materialize_zero_grads(opt)

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    # Metrics
    peak_alloc_mb = to_mb(torch.cuda.max_memory_allocated())
    avg_step_ms = training_time_ms / max(1, args.num_iterations)

    # Log row(s)
    if master_process:
        # final val loss = last of val_curve
        final_val = val_curve[-1][1] if val_curve else float("nan")
        all_results.append({
            "kind": "depth" if model_dim == FIXED_DIM else "dim",
            "optimizer": optimizer_name,
            "layers": int(num_layers),
            "dim": int(model_dim),
            "final_val_loss": float(final_val),
            "avg_step_ms": float(avg_step_ms),
            "peak_alloc_mb": float(peak_alloc_mb),
            "val_curve": val_curve,   # for plotting val loss vs steps
        })

    # cleanup
    del model, optimizers, train_loader, val_loader
    torch.cuda.empty_cache()

# ------------------------------- Run sweeps ------------------------------
OPT_LIST = ["muon", "adamw", "polargrad"]

if RUN_DEPTH_SWEEP:
    for depth in depth_values:
        for opt_name in OPT_LIST:
            run_experiment(opt_name, depth, FIXED_DIM)

if RUN_DIM_SWEEP:
    for dim in dim_values:
        for opt_name in OPT_LIST:
            run_experiment(opt_name, FIXED_DEPTH, dim)

# -------------------------- Save & plot results --------------------------
if master_process:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    out_json = f"logs/optimizer_compare_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Comparison complete! Results saved to {out_json}")

    # ---- Plots ----
    # A) For the fixed-depth DIM sweep and fixed-dim DEPTH sweep, plot final val losses (per optimizer)
    import numpy as np
    def collect(kind_key, sweep_vals, x_key, fixed_key, fixed_val):
        # returns dict: {optimizer: (x_list, y_list)} for final_val_loss
        out = {}
        for opt_name in OPT_LIST:
            xs, ys = [], []
            for v in sweep_vals:
                rows = [r for r in all_results
                        if r["kind"] == kind_key and r["optimizer"] == opt_name
                        and r[x_key] == v and r[fixed_key] == fixed_val]
                if not rows: continue
                # take the row (there should be exactly one)
                xs.append(v)
                ys.append(rows[0]["final_val_loss"])
            out[opt_name] = (xs, ys)
        return out

    # Depth sweep (x=layers, fixed dim)
    depth_data = collect("depth", depth_values, "layers", "dim", FIXED_DIM)
    # Dim sweep (x=dim, fixed layers)
    dim_data   = collect("dim", dim_values, "dim", "layers", FIXED_DEPTH)

    plt.figure(figsize=(12,4))
    # Left: final val loss vs depth
    plt.subplot(1,2,1)
    for opt_name, (xs, ys) in depth_data.items():
        if xs:
            plt.plot(xs, ys, marker="o", label=opt_name)
    plt.xlabel("Layers")
    plt.ylabel("Final val loss")
    plt.title(f"Final val loss vs depth (dim={FIXED_DIM})")
    plt.legend()

    # Right: final val loss vs dim
    plt.subplot(1,2,2)
    for opt_name, (xs, ys) in dim_data.items():
        if xs:
            plt.plot(xs, ys, marker="o", label=opt_name)
    plt.xlabel("Model dim")
    plt.ylabel("Final val loss")
    plt.title(f"Final val loss vs dim (layers={FIXED_DEPTH})")
    plt.legend()
    plt.tight_layout()
    out_png_a = f"logs/optimizer_compare_val_{timestamp}.png"
    plt.savefig(out_png_a, dpi=150)
    print(f"📈 Plot (final val loss) saved to {out_png_a}")

    # B) For *one* canonical config (depth=FIXED_DEPTH, dim=FIXED_DIM), plot val loss vs step curves
    plt.figure(figsize=(6,4))
    for opt_name in OPT_LIST:
        rows = [r for r in all_results
                if r["layers"] == FIXED_DEPTH and r["dim"] == FIXED_DIM and r["optimizer"] == opt_name]
        if rows and rows[0]["val_curve"]:
            steps, vals = zip(*rows[0]["val_curve"])
            plt.plot(steps, vals, marker="o", label=opt_name)
    plt.xlabel("Step")
    plt.ylabel("Val loss")
    plt.title(f"Val loss vs steps @ L={FIXED_DEPTH}, D={FIXED_DIM}")
    plt.legend()
    plt.tight_layout()
    out_png_b = f"logs/optimizer_compare_curves_{timestamp}.png"
    plt.savefig(out_png_b, dpi=150)
    print(f"📈 Plot (val loss vs steps) saved to {out_png_b}")

    # C) Bars: avg step time + peak memory for that same canonical config
    rows_cfg = [r for r in all_results if r["layers"] == FIXED_DEPTH and r["dim"] == FIXED_DIM]
    if rows_cfg:
        labels = [r["optimizer"] for r in rows_cfg]
        avg_ms = [r["avg_step_ms"] for r in rows_cfg]
        mem_mb = [r["peak_alloc_mb"] for r in rows_cfg]

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.bar(labels, avg_ms)
        plt.ylabel("Avg step time (ms)")
        plt.title(f"Avg step time @ L={FIXED_DEPTH}, D={FIXED_DIM}")

        plt.subplot(1,2,2)
        plt.bar(labels, mem_mb)
        plt.ylabel("Peak memory (MB)")
        plt.title(f"Peak memory @ L={FIXED_DEPTH}, D={FIXED_DIM}")

        plt.tight_layout()
        out_png_c = f"logs/optimizer_compare_speed_mem_{timestamp}.png"
        plt.savefig(out_png_c, dpi=150)
        print(f"⚡️ Plots (speed/memory) saved to {out_png_c}")

# Always shutdown DDP
dist.destroy_process_group()
