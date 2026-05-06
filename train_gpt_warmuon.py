"""
train_gpt_warmuon.py
====================
Benchmark: WarmMuon vs Muon (NS-5 baseline) vs WarmMuonColdOnly ablation.

Usage:
    torchrun --nproc_per_node=N train_gpt_warmuon.py --optimizer [muon|warmuon|coldonly]

Descends from records/track_3_optimization/train_gpt_simple.py.
Adds:
  - --optimizer flag selects the Muon-side optimizer
  - FLOP accounting for the orthogonalization step
  - Per-step timing split: model vs optimizer
  - WarmMuon diagnostic logging (drift, refresh_rate)
  - Results table printed at end of each run

FLOP accounting methodology:
  Each matrix multiply of shape (n, k) @ (k, m) costs 2*n*k*m FLOPs.
  For a parameter of shape (n, m) with k = min(n, m):
    NS-5:        5 iterations × 3 matmuls × 2*n*k*k + 2*n*k*k (final) = 15 * (2nk²)
    Krylov-2:    2*(num_steps-1) * 2*k*m*k + 1 * 2*n*k*k = 2*2k²m + 2nk² ≈ 3 * (2nk²)
    Warm (2mm):  2 * 2*n*m*m  (Y=X_prev.T@M, X=X_prev@polar_Y)
  We track cumulative orthogonalization FLOPs across all steps and parameters.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import argparse
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

from warmuon import WarmMuon, WarmMuonColdOnly, krylov_polar


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

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
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))
    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu().square())

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#     Muon baseline (NS-5)             #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#         FLOP accounting              #
########################################

def ns5_flops_per_param(n: int, m: int) -> int:
    """
    NS-5 orthogonalization FLOPs for a matrix of shape (n, m).
    Each of 5 iterations: 3 matmuls. For tall (n>m): works on (m,n).
    Square matmul (m,m)@(m,n) costs 2*m*m*n.
    Total: 5 * 3 * 2 * min(n,m)^2 * max(n,m)
    """
    a, b = min(n, m), max(n, m)
    return 5 * 3 * 2 * a * a * b

def warmmuon_flops_per_param(n: int, m: int, is_cold: bool, is_refresh: bool) -> int:
    """
    WarmMuon FLOPs per orthogonalization for a (n, m) matrix.
    - Cold / refresh: krylov_polar with num_steps=2, block_size=min(n,m)
      Cost: ~3 large matmuls of shape (min,min)×(min,max)
    - Warm:  2 matmuls of shape (n,m) each ≈ 2 * 2*n*m*min(n,m)
      Plus drift check: 1 matmul (min,max)×(max,min) = 2*min²*max
    """
    a, b = min(n, m), max(n, m)
    if is_cold or is_refresh:
        # krylov: step 0 init (1 matmul) + (num_steps-1)*2 main matmuls + 1 final
        return (1 + (2-1)*2 + 1) * 2 * a * a * b   # = 4 * 2a²b
    else:
        warm_mm = 2 * (2 * n * m * a)    # Y and X = X_prev @ polar_Y
        drift_mm = 2 * a * a * b         # drift check: X.T @ X
        return warm_mm + drift_mm


########################################
#              Setup                   #
########################################

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=["muon", "warmuon", "coldonly"],
                    default="warmuon",
                    help="Which optimizer to use for the Muon-side params")
args, _ = parser.parse_known_args()

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
assert 8 % dist.get_world_size() == 0

if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/warmuon_{args.optimizer}_{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

print0(f"optimizer={args.optimizer}")
print0(code)
print0("=" * 100)
print0(f"Running PyTorch {torch.version.__version__} on {torch.cuda.get_device_name(device)}"
       f" world_size={dist.get_world_size()}")
print0("=" * 100)


########################################
#         Training                     #
########################################

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

num_trials = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1

# Collect all Muon-eligible params (2D, from transformer blocks)
muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]

# Precompute per-param shapes for FLOP accounting
param_shapes = {id(p): (p.shape[0], p.shape[-1]) for p in muon_params}

for trial in range(num_trials):
    train_steps = 3350

    # Initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # Adam for non-Muon params
    optimizer1 = AdamW(
        [dict(params=[model.embed.weight], lr=0.3),
         dict(params=[model.proj.weight], lr=1/320),
         dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
        betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True,
    )

    # Muon-side optimizer
    if args.optimizer == "muon":
        optimizer2 = Muon(muon_params, lr=0.035, weight_decay=0.025)
    elif args.optimizer == "warmuon":
        optimizer2 = WarmMuon(muon_params, lr=0.035, weight_decay=0.025, mu=0.95,
                              cold_steps=10, drift_threshold=0.01, krylov_steps=2)
    elif args.optimizer == "coldonly":
        optimizer2 = WarmMuonColdOnly(muon_params, lr=0.035, weight_decay=0.025, mu=0.95,
                                      cold_steps=0, drift_threshold=0.01, krylov_steps=2)

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    # FLOP / timing accumulators
    total_opt_flops = 0
    total_model_time = 0.0
    total_opt_time = 0.0
    training_time = 0.0
    steps_to_target: int | None = None   # steps to reach val_loss <= 3.28
    TARGET_LOSS = 3.28

    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()

    for step in range(train_steps + 1):
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            dist.barrier()
            time_since_last = time.perf_counter() - t0
            step_avg = time_since_last / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss = (val_loss / val_tokens).item()

            if steps_to_target is None and val_loss <= TARGET_LOSS:
                steps_to_target = step

            opt_frac = total_opt_time / (total_model_time + total_opt_time + 1e-9)
            print0(
                f"step:{step}/{train_steps} val_loss:{val_loss:.5f} "
                f"train_time:{training_time:.3f}s step_avg:{1000*step_avg:.2f}ms "
                f"opt_frac:{opt_frac:.1%} opt_gflops:{total_opt_flops/1e9:.2f}",
                console=True,
            )
            model.train()
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --- Training step ---
        inputs, targets = next(train_loader)
        t_model_start = time.perf_counter()
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        total_model_time += time.perf_counter() - t_model_start

        set_hparams(step)

        t_opt_start = time.perf_counter()
        optimizer1.step()
        optimizer2.step()
        torch.cuda.synchronize()
        total_opt_time += time.perf_counter() - t_opt_start

        model.zero_grad(set_to_none=True)

        # FLOP accounting for optimizer2
        if args.optimizer == "muon":
            for p in muon_params:
                n, m = param_shapes[id(p)]
                total_opt_flops += ns5_flops_per_param(n, m)
        else:
            stats = optimizer2.optimizer_stats()
            refresh_rate = stats["warmuon/refresh_rate"]
            cold_active  = stats["warmuon/cold_active"]
            for p in muon_params:
                n, m = param_shapes[id(p)]
                # Approximate: assume refresh_rate fraction were refreshed
                is_cold = cold_active
                is_refresh = (not cold_active) and (refresh_rate > 0)
                total_opt_flops += warmmuon_flops_per_param(n, m, is_cold, is_refresh)

            # Log WarmMuon diagnostics every 100 steps
            if step % 100 == 0:
                print0(
                    f"  warmuon step={step}: avg_drift={stats['warmuon/avg_drift']:.4f} "
                    f"max_drift={stats['warmuon/max_drift']:.4f} "
                    f"refresh_rate={stats['warmuon/refresh_rate']:.2%} "
                    f"cold_active={stats['warmuon/cold_active']}"
                )

        approx_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_time:.3f}s"
               f" step_avg:{1000*approx_time/(step+1):.2f}ms",
               console=True, log=False)

    # --- End-of-trial summary ---
    final_loss = val_loss
    print0("\n" + "=" * 70, console=True)
    print0(f"RESULTS  optimizer={args.optimizer}  trial={trial+1}/{num_trials}", console=True)
    print0(f"  steps_to_{TARGET_LOSS}: {steps_to_target if steps_to_target is not None else 'NOT_REACHED'}", console=True)
    print0(f"  final_val_loss:         {final_loss:.5f}", console=True)
    print0(f"  total_wall_time:        {training_time:.1f}s", console=True)
    print0(f"  model_time:             {total_model_time:.1f}s  ({100*total_model_time/(total_model_time+total_opt_time+1e-9):.0f}%)", console=True)
    print0(f"  opt_time:               {total_opt_time:.1f}s  ({100*total_opt_time/(total_model_time+total_opt_time+1e-9):.0f}%)", console=True)
    print0(f"  opt_GFLOPs:             {total_opt_flops/1e9:.1f}", console=True)
    print0("=" * 70, console=True)

dist.destroy_process_group()
