# ======================================================================
# Muon scaling experiments: varying depth or embedding dimension
# ======================================================================
import os, sys, time, json, itertools, math
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from datetime import datetime

from train_gpt import (
    GPT, Muon,  # we won't use DistAdam in the sweep (keep it simple)
    distributed_data_generator, get_window_size_blocks,
    get_lr, Hyperparameters, norm, F, BlockMask
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
        with open("logs/muon_depth_dim_sweep_log.txt", "a") as f:
            print(s, file=f)

# ------------------------ GPT wrapper for sweeps -----------------------
class GPTForSweep(GPT):
    """Make forward robust to arbitrary depth (nL)."""
    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor, sliding_window_num_blocks: torch.Tensor):
        assert input_seq.ndim == 1
        nL = len(self.blocks)

        # Value embeddings schedule
        ve_all = [ve(input_seq) for ve in self.value_embeds]
        if nL >= 6:
            front = [ve_all[0], ve_all[1], ve_all[2]]
            mid = [None] * (nL - 6)
            tail = [ve_all[0], ve_all[1], ve_all[2]]
            ve = (front + mid + tail)[:nL]
        else:
            ve = [None] * nL

        # Block masks: simple pattern, long every 4th layer (and last)
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm if (i % 4 == 0 or i == nL - 1) else short_bm for i in range(nL)]

        x = x0 = norm(self.embed(input_seq)[None])

        # U-net like skip structure driven by scalars
        skip_connections = []
        skip_weights = self.scalars[:(nL // 2)]
        lambdas = self.scalars[1 * nL : 3 * nL].view(-1, 2)
        sa_lambdas = self.scalars[3 * nL : 5 * nL].view(-1, 2)

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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# ---------------------------- Base hyperparams -------------------------
args = Hyperparameters()
# Shorter/fair sweep defaults (tweak if you like)
args.num_iterations = 300
args.val_loss_every = 100
args.save_checkpoint = False
args.cooldown_frac = 0.5
args.val_tokens = 1048576          # 1M tokens
args.train_seq_len = 8192
args.val_seq_len = 8192

# ------------------------------- Sweeps --------------------------------
depth_values = [4, 8, 12, 16]          # vary layers (fixed dim)
dim_values   = [384, 768, 1024, 1536]  # vary model_dim (fixed depth)

RUN_DEPTH_SWEEP = True
RUN_DIM_SWEEP   = True

FIXED_DIM   = 768
FIXED_DEPTH = 12

sweep_results = []

# -------------------------- Helper: zero-grads --------------------------
def materialize_zero_grads(optim: torch.optim.Optimizer):
    """Ensure every param the optimizer owns has a (possibly zero) grad tensor.
       Muon expects grads for its 2D params; if a layer is skipped/zero-init,
       some grads can be None."""
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

# --------------------------- Single experiment --------------------------
def run_muon_experiment(num_layers, model_dim):
    print0(f"=== Running Muon experiment: layers={num_layers}, dim={model_dim} ===", console=True)

    # when building the model in the sweep
    assert model_dim % 128 == 0, "model_dim must be a multiple of 128"
    num_heads = model_dim // 128  # NOT max(4, ...)
    model = GPTForSweep(
        vocab_size=50257,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max(args.train_seq_len, args.val_seq_len)
    ).cuda()
    # Keep it simple: skip torch.compile during sweep to avoid Dynamo overhead on tiny runs
    # model = torch.compile(model, dynamic=False)

    # embeddings in bf16 (like your main run)
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    # sync params across ranks
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    # Params
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params   = [model.lm_head.weight]
    embed_params  = [p for n, p in model.named_parameters() if "embed" in n]
    gate_params   = [p for n, p in model.named_parameters() if "gate" in n]

    # Optimizers (simple): AdamW for scalar/embed/head; Muon for 2D hidden + gates
    optimizer1 = torch.optim.AdamW(
        scalar_params + head_params + embed_params,
        lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0
    )
    optimizer2 = Muon(
        hidden_matrix_params + gate_params,
        lr=0.06, momentum=0.95, weight_decay=0.0
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    # Data
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=False)
    val_loader   = distributed_data_generator(args.val_files,   world_size * args.val_seq_len,   align_to_bos=False)

    # Train short run
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

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

            print0(f"[DepthDim] layers={num_layers}, dim={model_dim} step={step} "
                   f"val_loss={val_loss:.4f} time={training_time_ms/1000:.1f}s", console=True)
            if master_process:
                sweep_results.append({
                    "kind": "depth" if model_dim == FIXED_DIM else "dim",
                    "layers": int(num_layers),
                    "dim": int(model_dim),
                    "step": int(step),
                    "val_loss": float(val_loss),
                    "time_s": float(training_time_ms / 1000.0),
                })
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # ---- training ----
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()

        # LR schedule + Muon momentum warmup
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * get_lr(step)
        for g in optimizer2.param_groups:
            frac = min(step / 300, 1.0)
            g["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # Ensure grads exist for Muon
        materialize_zero_grads(optimizer2)

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    # cleanup
    del model, optimizer1, optimizer2, optimizers, train_loader, val_loader
    torch.cuda.empty_cache()

# ------------------------------- Run sweeps ------------------------------
if RUN_DEPTH_SWEEP:
    for depth in depth_values:
        run_muon_experiment(depth, FIXED_DIM)

if RUN_DIM_SWEEP:
    for dim in dim_values:
        run_muon_experiment(FIXED_DEPTH, dim)

# -------------------------- Save & plot results --------------------------
if master_process:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    out_json = f"logs/muon_depth_dim_sweep_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n✅ Sweep complete! Results saved to {out_json}")

    # Build simple plots: best (last) val_loss per config
    by_depth = {}
    by_dim = {}
    for r in sweep_results:
        key = (r["layers"], r["dim"])
        if r["kind"] == "depth" and r["dim"] == FIXED_DIM:
            by_depth[r["layers"]] = r["val_loss"]
        if r["kind"] == "dim" and r["layers"] == FIXED_DEPTH:
            by_dim[r["dim"]] = r["val_loss"]

    depths = sorted(by_depth.keys())
    dims   = sorted(by_dim.keys())
    depth_losses = [by_depth[d] for d in depths]
    dim_losses   = [by_dim[d] for d in dims]

    plt.figure(figsize=(10,4))
    # Left: depth sweep
    plt.subplot(1,2,1)
    plt.plot(depths, depth_losses, marker="o")
    plt.xlabel("Layers")
    plt.ylabel("Val loss")
    plt.title(f"Muon: val loss vs depth (dim={FIXED_DIM})")
    # Right: dim sweep
    plt.subplot(1,2,2)
    plt.plot(dims, dim_losses, marker="o")
    plt.xlabel("Model dim")
    plt.ylabel("Val loss")
    plt.title(f"Muon: val loss vs dim (layers={FIXED_DEPTH})")
    plt.tight_layout()
    out_png = f"logs/muon_depth_dim_sweep_{timestamp}.png"
    plt.savefig(out_png, dpi=150)
    print(f"📈 Plot saved to {out_png}")

# Always shutdown DDP
dist.destroy_process_group()
