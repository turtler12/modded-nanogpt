"""
train_gpt_benchmark.py
======================
A/B/C/D optimizer benchmark on GPT-2 124M, derived from train_gpt_simple.py.

Configs (select with --config A|B|C|D):
  A) Stock NS-12: muon_update with 12 iterations of the quintic (36 mm/param).
                  This is what train_gpt_simple.py actually runs, named
                  zeropower_via_newtonschulz5 despite being 12 iterations.
  B) Krylov-only: krylov_polar(num_steps=2, block_size=min_dim) for every
                  Muon-eligible param. No warm-start, no X_prev, no drift monitor.
                  ~3 mm/param (square) or ~4 mm/param (rect).
  C) WarmMuon:    warm-start where aspect ≤ 2, Krylov elsewhere, drift monitor.
                  Same FLOP budget as B but numerically closer to true polar.
  D) NS-2:        Same as A but 2 quintic iterations (6 mm/param). Cheapest NS.
                  Critical control: if D ≈ A, "fewer iterations" is the win.

Usage (single GPU):
    torchrun --nproc_per_node=1 train_gpt_benchmark.py --config B --seed 0

Usage (multi-GPU, e.g. 8):
    torchrun --nproc_per_node=8 train_gpt_benchmark.py --config B --seed 0

Step-timer sanity check (runs one step per config, reports optimizer wall-clock):
    torchrun --nproc_per_node=1 train_gpt_benchmark.py --step-timer

Results are written to logs/<run_id>.txt and printed to stdout.
"""

import os
import sys
import uuid
import time
import math
import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


# ── Parse args before anything dist-related ────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config",     default="A",  choices=["A","B","C","D"],
                    help="Optimizer config to run")
parser.add_argument("--seed",       type=int, default=0,
                    help="Random seed (use 0,1,2 for 3-seed runs)")
parser.add_argument("--steps",      type=int, default=3350,
                    help="Training steps (default 3350, speedrun standard)")
parser.add_argument("--step-timer", action="store_true",
                    help="Run one-step wall-clock timer for all 4 configs and exit")
parser.add_argument("--target-loss", type=float, default=3.28,
                    help="Val-loss target for steps-to-target metric")
# absorb torchrun's trailing integer if present
args, _extra = parser.parse_known_args()


# ── Distributed setup ──────────────────────────────────────────────────────

device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
assert 8 % dist.get_world_size() == 0
rank       = dist.get_rank()
world_size = dist.get_world_size()


# ── Logging ────────────────────────────────────────────────────────────────

if rank == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/benchmark_{args.config}_seed{args.seed}_{uuid.uuid4().hex[:8]}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if rank == 0:
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

print0(f"config={args.config} seed={args.seed} world_size={world_size}")
print0(f"Running PyTorch {torch.__version__} on {torch.cuda.get_device_name(device)}")


# ── Data ───────────────────────────────────────────────────────────────────

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520 and header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

def distributed_data_generator(pattern, batch_size, seq_len=1024):
    files = sorted(Path.cwd().glob(pattern))
    assert len(files) > 0, f"No files found for pattern: {pattern}"
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_bs:][:local_bs + 1]
        inputs  = buf[:-1].to(device=device, dtype=torch.int32,  non_blocking=True)
        targets = buf[1:].to(device=device,  dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


# ── Architecture (verbatim from train_gpt_simple.py) ──────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        angular_freq = (1/1024)**torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))
    def forward(self, x_BTHD):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim  = head_dim
        hdim = self.num_heads * head_dim
        self.q = Linear(dim, hdim); self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim); self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
    def forward(self, x):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
            scale=0.12, is_causal=True).transpose(1,2)
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = Linear(dim, 4*dim); self.proj = Linear(4*dim, dim)
    def forward(self, x):
        return self.proj(self.fc(x).relu().square())

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn  = CausalSelfAttention(dim); self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim);             self.norm2 = RMSNorm(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj   = Linear(model_dim, vocab_size)
        self.norm1  = RMSNorm(model_dim); self.norm2 = RMSNorm(model_dim)
    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs).float())
        for block in self.blocks: x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


# ── Orthogonalizers ────────────────────────────────────────────────────────

# ---- Config A & D: Newton-Schulz ----

def _ns_polar(G: Tensor, n_iters: int) -> Tensor:
    """Newton-Schulz quintic, n_iters iterations. Stock is n_iters=12."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2,-1), keepdim=True) + 1e-7)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(n_iters):
        A = X @ X.mT
        B = b*A + c*(A @ A)
        X = a*X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def ns12_polar(G: Tensor) -> Tensor:
    return _ns_polar(G, 12)

@torch.compile
def ns2_polar(G: Tensor) -> Tensor:
    return _ns_polar(G, 2)


# ---- Config B: Krylov-only ----

def krylov_polar(M: Tensor, num_steps: int = 2, block_size: int | None = None) -> Tensor:
    """
    Block-Krylov polar decomposition. ~3 matmuls for square, ~4 for rectangular.
    Identical to warmuon.krylov_polar — reproduced here to avoid import side-effects.
    """
    assert M.ndim == 2
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M
    n, m = Mt.shape
    bs = min(block_size or m, n)
    g = torch.Generator(device=Mt.device)
    g.manual_seed(0)
    if n < m:
        rand_init = torch.randn(n, bs, generator=g, dtype=Mt.dtype, device=Mt.device)
        Omega = Mt.T @ rand_init
    else:
        Omega = torch.randn(m, bs, generator=g, dtype=Mt.dtype, device=Mt.device)
    Q, _ = torch.linalg.qr(Omega)
    Q_blocks = [Q]
    for _ in range(num_steps - 1):
        Y = Mt @ Q_blocks[-1]
        Z = Mt.T @ Y
        for Qb in Q_blocks:
            Z = Z - Qb @ (Qb.T @ Z)
        Z, _ = torch.linalg.qr(Z)
        Q_blocks.append(Z)
    Q = torch.cat(Q_blocks, dim=1)[:, :n]
    MQ = Mt @ Q
    U_mq, _, Vh_mq = torch.linalg.svd(MQ, full_matrices=False)
    X = U_mq @ Vh_mq @ Q.T
    return X.T if transposed else X

def krylov_update(G: Tensor) -> Tensor:
    """Krylov polar, normalized like NS (scale by max(1, rows/cols)^0.5 is in the optimizer)."""
    M = G.float()
    n, m = M.shape
    return krylov_polar(M, num_steps=2, block_size=min(n, m)).to(G.dtype)


# ---- Config C: WarmMuon ----
# Imported from warmuon.py; WarmMuon.step() handles everything including
# the aspect-ratio dispatch and X_prev threading.
from warmuon import WarmMuon


# ── Optimizer wrappers (A, B, D share Muon structure) ─────────────────────

def _nesterov_momentum(grad: Tensor, buf: Tensor, mu: float = 0.95) -> Tensor:
    buf.lerp_(grad, 1.0 - mu)
    return grad.lerp_(buf, mu)  # nesterov

@torch.compile
def _nesterov_momentum_compiled(grad: Tensor, buf: Tensor, mu: float = 0.95) -> Tensor:
    buf.lerp_(grad, 1.0 - mu)
    return grad.lerp_(buf, mu)


class MuonVariant(torch.optim.Optimizer):
    """
    General Muon-style optimizer. Swap orthogonalizer via `ortho_fn`.
    Mirrors the distributed all-gather round-robin from train_gpt_simple.py.
    """

    def __init__(self, params, lr=0.035, weight_decay=0.025, mu=0.95,
                 ortho_fn=None, name="MuonVariant"):
        assert isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.numel(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.ortho_fn = ortho_fn  # (grad_f32) -> polar_f32
        self.name     = name
        # per-step timing accumulators
        self._opt_time_ms = 0.0
        self._step_count  = 0

    @torch.no_grad()
    def step(self):
        t0 = time.perf_counter()
        for group in self.param_groups:
            mu = group["mu"]; lr = group["lr"]; wd = group["weight_decay"]
            params = group["params"]
            pad = [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            params_padded = params + pad
            for base_i in range(0, len(params_padded), world_size):
                local_i = base_i + rank
                if local_i < len(params):
                    p = params[local_i]
                    assert p.grad is not None
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    buf     = state["momentum"]
                    g_f32   = p.grad.float()
                    update_g = _nesterov_momentum_compiled(g_f32, buf, mu)
                    # reshape to 2D
                    orig_shape = update_g.shape
                    M = update_g.view(-1, orig_shape[-1]) if update_g.ndim > 2 else update_g
                    # orthogonalize
                    X = self.ortho_fn(M)
                    # scale: max(1, rows/cols)^0.5
                    n_r, n_c = M.shape
                    X = X * max(1.0, n_r / n_c) ** 0.5
                    X = X.reshape(orig_shape).to(p.dtype)
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(X, alpha=-lr)
                if world_size > 1:
                    dist.all_gather(params_padded[base_i:base_i+world_size],
                                    params_padded[base_i+rank])
        torch.cuda.synchronize()
        self._opt_time_ms += (time.perf_counter() - t0) * 1000
        self._step_count  += 1

    def avg_opt_time_ms(self):
        return self._opt_time_ms / self._step_count if self._step_count else 0.0


# ── FLOP accounting ────────────────────────────────────────────────────────

def optimizer_flops_per_step(config: str, muon_params: list,
                              warmuon_refresh_rate: float = 0.0) -> float:
    """Theoretical matmul FLOPs per optimizer step for each config."""
    # matmul cost formula: n_mm * 2 * min(r,c) * min(r,c) * max(r,c)
    # (all matmuls on a param of shape (r,c) are dominated by that size)
    def mm_flops(r, c, n_mm):
        n = min(r, c); m = max(r, c)
        return n_mm * 2 * n * n * m

    total = 0.0
    for p in muon_params:
        r, c = p.shape[-2], p.shape[-1]
        aspect = max(r, c) / min(r, c)
        if config == "A":
            total += mm_flops(r, c, 36)   # 12 iters × 3 mm
        elif config == "D":
            total += mm_flops(r, c, 6)    # 2 iters × 3 mm
        elif config == "B":
            mm = 4 if aspect > 2.0 else 3
            total += mm_flops(r, c, mm)
        elif config == "C":
            if aspect > 2.0:
                total += mm_flops(r, c, 4)    # always krylov rect
            else:
                # amortized: (1-r)*3 + r*3 = 3 regardless; same as B
                total += mm_flops(r, c, 3)
    return total


# ── Model init helper ──────────────────────────────────────────────────────

def init_model_weights(model, seed):
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:          w.zero_()
            elif "embed" in name:       w.normal_()
            else:                       w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):     w.zero_()
        elif name.endswith("gains"):    w.normal_(mean=1, std=0)
        else: raise ValueError(f"Uninitialized: {name}")


# ── Step-timer sanity check ────────────────────────────────────────────────

def run_step_timer():
    """
    Time one optimizer step per config on a real model + random batch.
    Reports: optimizer time, total step time, opt_frac, theoretical FLOPs.
    """
    print0("=" * 70, console=True)
    print0("Step-timer sanity check — 10 warmup + 20 timed steps per config",
           console=True)
    print0("=" * 70, console=True)

    VOCAB = 50304; NLAYERS = 12; DIM = 768
    BS = 8; SEQ = 1024

    model = GPT(vocab_size=VOCAB, num_layers=NLAYERS, model_dim=DIM).cuda()
    model.compile(dynamic=False)
    init_model_weights(model, seed=0)

    muon_eligible = [p for p in model.blocks.parameters() if p.ndim >= 2]

    results = {}

    configs_to_time = [
        ("A", lambda M: ns12_polar(M).float()),
        ("D", lambda M: ns2_polar(M).float()),
        ("B", lambda M: krylov_update(M)),
        ("C", None),   # WarmMuon handled separately
    ]

    for cfg, ortho_fn in configs_to_time:
        init_model_weights(model, seed=0)

        if cfg == "C":
            opt_muon = WarmMuon(
                [p for p in model.blocks.parameters() if p.ndim >= 2],
                lr=0.035, weight_decay=0.025, mu=0.95,
                cold_steps=10, drift_threshold=0.1,
            )
        else:
            opt_muon = MuonVariant(
                [p for p in model.blocks.parameters() if p.ndim >= 2],
                lr=0.035, weight_decay=0.025, mu=0.95,
                ortho_fn=ortho_fn, name=f"Config-{cfg}",
            )

        opt_adam = AdamW(
            [dict(params=[model.embed.weight], lr=0.3),
             dict(params=[model.proj.weight],  lr=1/320),
             dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
            betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True
        )

        N_WARMUP = 10; N_TIME = 20
        step_times = []; opt_times = []

        for step in range(N_WARMUP + N_TIME):
            inputs  = torch.randint(0, VOCAB, (BS, SEQ), device=device, dtype=torch.int32)
            targets = torch.randint(0, VOCAB, (BS, SEQ), device=device, dtype=torch.int64)

            torch.cuda.synchronize()
            t_step = time.perf_counter()

            model(inputs, targets).backward()
            if world_size > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)

            torch.cuda.synchronize()
            t_opt = time.perf_counter()

            if cfg == "C":
                opt_muon.step()
            else:
                opt_muon.step()
            opt_adam.step()

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            model.zero_grad(set_to_none=True)

            if step >= N_WARMUP:
                step_times.append((t_end - t_step) * 1000)
                opt_times.append((t_end - t_opt) * 1000)

        avg_step = sum(step_times) / len(step_times)
        avg_opt  = sum(opt_times)  / len(opt_times)
        opt_frac = avg_opt / avg_step
        gflops   = optimizer_flops_per_step(cfg, muon_eligible) / 1e9

        results[cfg] = dict(step_ms=avg_step, opt_ms=avg_opt,
                            opt_frac=opt_frac, gflops=gflops)

        print0(f"Config {cfg}: step={avg_step:.1f}ms  opt={avg_opt:.1f}ms  "
               f"opt_frac={opt_frac:.1%}  FLOPs={gflops:.1f} GFLOPs/step",
               console=True)

    # Decision logic
    print0("", console=True)
    ref_step = results["A"]["step_ms"]
    print0(f"Relative step times vs A:", console=True)
    for cfg in ["B", "C", "D"]:
        r = results[cfg]
        delta = (r["step_ms"] - ref_step) / ref_step
        print0(f"  {cfg}: {r['step_ms']:.1f}ms  ({delta:+.1%} vs A)", console=True)

    max_delta = max(abs((results[c]["step_ms"] - ref_step)/ref_step)
                    for c in ["B","C","D"])
    if max_delta < 0.05:
        print0("\n⚠  Wall-clock change across all configs < 5%.", console=True)
        print0("   Orthogonalization is NOT the bottleneck at this model size.", console=True)
        print0("   Training benchmark will show FLOP reduction without wall-clock gain.", console=True)
        print0("   Consider: larger model, single-GPU (no all-reduce hiding opt cost),", console=True)
        print0("   or profiling to find actual bottleneck.", console=True)
    else:
        print0(f"\n✓  Wall-clock spread = {max_delta:.1%} — optimizer cost is visible.", console=True)

    print0("=" * 70, console=True)
    dist.destroy_process_group()


# ── Full training run ──────────────────────────────────────────────────────

def run_training():
    config = args.config
    seed   = args.seed
    train_steps  = args.steps
    target_loss  = args.target_loss

    VOCAB = 50304; NLAYERS = 12; DIM = 768
    batch_size = 8 * 64 * 1024
    mbs = 64

    print0(f"{'='*70}")
    print0(f"Config {config} | seed {seed} | {train_steps} steps | "
           f"target_loss {target_loss}")
    print0(f"{'='*70}")

    # ---- data ----
    val_tokens = 20 * 524288
    val_inputs, val_targets = next(distributed_data_generator(
        "data/fineweb10B/fineweb_val_*.bin", val_tokens))

    # ---- model ----
    model = GPT(vocab_size=VOCAB, num_layers=NLAYERS, model_dim=DIM).cuda()
    model.compile(dynamic=False)
    init_model_weights(model, seed=seed)

    # Broadcast initial weights from rank 0 so all ranks start identically
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    muon_eligible = [p for p in model.blocks.parameters() if p.ndim >= 2]

    # ---- optimizer ----
    adam_params = [
        dict(params=[model.embed.weight], lr=0.3),
        dict(params=[model.proj.weight],  lr=1/320),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
    ]
    optimizer1 = AdamW(adam_params, betas=(0.8, 0.95), eps=1e-10,
                       weight_decay=0, fused=True)

    if config == "A":
        ortho_fn = lambda M: ns12_polar(M).float()
        optimizer2 = MuonVariant(muon_eligible, lr=0.035, weight_decay=0.025,
                                 ortho_fn=ortho_fn, name="NS-12")
        opt_label = "NS-12 (12 quintic iters, 36 mm/param)"
    elif config == "B":
        ortho_fn = lambda M: krylov_update(M)
        optimizer2 = MuonVariant(muon_eligible, lr=0.035, weight_decay=0.025,
                                 ortho_fn=ortho_fn, name="Krylov-only")
        opt_label = "Krylov-only (2 steps, ~3-4 mm/param)"
    elif config == "C":
        optimizer2 = WarmMuon(muon_eligible, lr=0.035, weight_decay=0.025,
                              mu=0.95, cold_steps=10, drift_threshold=0.1)
        opt_label = "WarmMuon (warm-start aspect≤2, Krylov rect, drift monitor)"
    elif config == "D":
        ortho_fn = lambda M: ns2_polar(M).float()
        optimizer2 = MuonVariant(muon_eligible, lr=0.035, weight_decay=0.025,
                                 ortho_fn=ortho_fn, name="NS-2")
        opt_label = "NS-2 (2 quintic iters, 6 mm/param)"

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    print0(f"Optimizer: {opt_label}", console=True)
    flops_gflops = optimizer_flops_per_step(config, muon_eligible) / 1e9
    print0(f"Theoretical optimizer FLOPs: {flops_gflops:.1f} GFLOPs/step", console=True)

    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    # ---- training loop ----
    train_loader = distributed_data_generator(
        "data/fineweb10B/fineweb_train_*.bin", batch_size)

    training_time   = 0.0
    last_val_step   = 0
    steps_to_target = None   # first step where val_loss ≤ target_loss
    all_val_losses  = []

    # per-step optimizer timing
    total_opt_ms  = 0.0
    total_step_ms = 0.0
    timed_steps   = 0

    dist.barrier()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):

        # ---- validation ----
        val_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_freq == 0:
            dist.barrier()
            elapsed = time.perf_counter() - t0
            step_avg = elapsed / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += elapsed

            model.eval()
            val_loss = torch.tensor(0.0, device=device)
            with torch.no_grad():
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs],
                                      val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss_f = float(val_loss / val_tokens)
            all_val_losses.append((step, val_loss_f))

            if steps_to_target is None and val_loss_f <= target_loss:
                steps_to_target = step

            opt_frac_str = ""
            if timed_steps > 0:
                avg_step_ms = total_step_ms / timed_steps
                avg_opt_ms  = total_opt_ms  / timed_steps
                opt_frac_str = f" opt_frac:{avg_opt_ms/avg_step_ms:.1%}"

            print0(f"step:{step}/{train_steps} val_loss:{val_loss_f:.5f} "
                   f"train_time:{training_time:.3f}s step_avg:{1000*step_avg:.2f}ms"
                   + opt_frac_str, console=True)
            model.train()
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # ---- training step ----
        inputs, targets = next(train_loader)
        assert len(inputs) % mbs == 0

        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        t_opt_start = time.perf_counter()

        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t_step_end = time.perf_counter()

        total_opt_ms  += (t_step_end  - t_opt_start)  * 1000
        total_step_ms += (t_step_end  - t_step_start) * 1000
        timed_steps   += 1

        approx_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_time:.3f}s "
               f"step_avg:{1000*approx_time/(step+1):.2f}ms",
               console=False)

    # ---- final summary ----
    final_val = all_val_losses[-1][1] if all_val_losses else float("nan")
    avg_step_ms = total_step_ms / timed_steps if timed_steps else float("nan")
    avg_opt_ms  = total_opt_ms  / timed_steps if timed_steps else float("nan")
    opt_frac    = avg_opt_ms / avg_step_ms

    # WarmMuon-specific stats
    wm_stats = ""
    if config == "C" and hasattr(optimizer2, "optimizer_stats"):
        s = optimizer2.optimizer_stats()
        wm_stats = (f" | warmuon_refresh={s['warmuon/refresh_rate']:.2%}"
                    f" avg_drift={s['warmuon/avg_drift']:.4f}")

    print0("", console=True)
    print0("=" * 70, console=True)
    print0(f"RESULT config={config} seed={seed}", console=True)
    print0(f"  optimizer:       {opt_label}", console=True)
    print0(f"  steps_to_{target_loss}: {steps_to_target if steps_to_target else '>'+str(train_steps)}",
           console=True)
    print0(f"  final_val_loss:  {final_val:.5f}", console=True)
    print0(f"  total_wall_time: {training_time:.1f}s", console=True)
    print0(f"  avg_step_ms:     {avg_step_ms:.2f}ms", console=True)
    print0(f"  avg_opt_ms:      {avg_opt_ms:.2f}ms  ({opt_frac:.1%} of step)",
           console=True)
    print0(f"  opt_GFLOPs/step: {flops_gflops:.1f}" + wm_stats, console=True)
    print0("=" * 70, console=True)

    dist.destroy_process_group()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if args.step_timer:
        run_step_timer()
    else:
        run_training()
