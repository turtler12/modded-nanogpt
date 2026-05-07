"""
train_gpt_krylov.py

ContraMuon SOTA script with Krylov polar replacing NS-12.
All hyperparameters, architecture, schedule identical to contra_muon_sota.
Only change: zeropower_via_newtonschulz5 → krylov_ns_polar (GPU-native, compilable).

Runs on 1–8 GPUs. With 1 GPU: 8 gradient accumulation steps (same total tokens/step
as the 8-GPU SOTA run: 8 * 64 * 1024 = 524,288 tokens).

Run:
    torchrun --nproc_per_node=1 train_gpt_krylov.py
    torchrun --nproc_per_node=8 train_gpt_krylov.py
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

# ── ContraMuon hyperparameters (unchanged from SOTA) ─────────────────────────
CONTRA_MUON      = 0.4
MU               = 0.95
MUON_LR          = 0.0375
MUON_WEIGHT_DECAY = 0.025
TARGET_UW        = 0.35
SEED             = 0

# ── Krylov sub-problem NS iters ───────────────────────────────────────────────
NS_SUB_ITERS = 5


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
        assert nbytes == 2 * num_tokens
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs  = buf[:-1].to(device="cuda", dtype=torch.int32,  non_blocking=True)
        targets = buf[1:].to(device="cuda",  dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

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
        self.head_dim  = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim); self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim); self.proj = Linear(hdim, dim)
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
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim); self.proj = Linear(4 * dim, dim)
    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu().square())

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim); self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim);            self.norm2 = RMSNorm(dim)
    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj   = Linear(model_dim, vocab_size)
        self.norm1  = RMSNorm(model_dim); self.norm2 = RMSNorm(model_dim)
    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks: x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#     GPU-native Krylov polar          #
########################################

def _ns_square(X: Tensor, n_iters: int) -> Tensor:
    """NS polar for square matrices (no tall-branch needed in sub-problems)."""
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(n_iters):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X

@torch.compile
def krylov_ns_polar(G: Tensor) -> Tensor:
    """
    GPU-native Krylov polar: replaces NS-12 with 3 large matmuls + 3×NS_SUB_ITERS
    small (n×n) matmuls. Fully compilable, bfloat16 throughout.

    Cost vs NS-12:
      NS-12: 36 mm/param (12 iters × 3 matmuls, all on full shape)
      Krylov: ~3 large + 45 small n×n — ≈3 equivalent large matmuls/param
    """
    transposed = G.size(-2) > G.size(-1)
    Mt = G.mT if transposed else G          # (n, m), n ≤ m
    Mt = Mt.bfloat16()
    n, m = Mt.size(-2), Mt.size(-1)

    torch.manual_seed(0)
    if n < m:
        rand_init = torch.randn(n, n, dtype=Mt.dtype, device=Mt.device)
        Omega = Mt.mT @ rand_init           # (m, n) — row-space init for rectangular
    else:
        Omega = torch.randn(m, n, dtype=Mt.dtype, device=Mt.device)

    Q  = _ns_square(Omega.mT, NS_SUB_ITERS).mT   # (m, n)
    Y  = Mt @ Q                                    # (n, n)
    Z  = Mt.mT @ Y                                 # (m, n)
    Z  = Z - Q @ (Q.mT @ Z)                       # reorthogonalize
    Q2 = _ns_square(Z.mT, NS_SUB_ITERS).mT        # (m, n)

    Q_full  = torch.cat([Q, Q2], dim=-1)[:, :n]   # (m, n)
    MQ      = Mt @ Q_full                          # (n, n)
    polar_MQ = _ns_square(MQ, NS_SUB_ITERS)       # (n, n)
    X = polar_MQ @ Q_full.mT                      # (n, m)
    return X.mT if transposed else X


########################################
#            Optimizer                 #
########################################

def scale_to_unit_operator_norm(G: Tensor, eps: float = 1e-10) -> Tensor:
    X = G.float()
    v = torch.ones(X.size(-1), dtype=X.dtype, device=X.device)
    v = v / torch.clamp(v.norm(), min=eps)
    for _ in range(5):
        u = X @ v;  u = u / torch.clamp(u.norm(), min=eps)
        v = X.mT @ u; v = v / torch.clamp(v.norm(), min=eps)
    return G / torch.clamp((X @ v).norm(), min=eps).to(G.dtype)

@torch.compile
def muon_update(grad, momentum, second_moment, mu=0.95, beta2=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    normalized_grad = scale_to_unit_operator_norm(update.clone())

    # ── Krylov polar instead of NS-12 ──────────────────────────────────────
    update = krylov_ns_polar(update).float()
    opower_frobenius_norm = update.norm()

    # ContraMuon subtraction
    update = update - CONTRA_MUON / 2 * normalized_grad
    update = update * opower_frobenius_norm / torch.clamp(update.norm(), min=1e-10)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5

    # Per-row (or per-col) variance EMA — NorMuon-lite
    if update.size(-2) >= update.size(-1):
        per_row_var = (update * update).mean(dim=-1, keepdim=True)
    else:
        per_row_var = (update * update).mean(dim=-2, keepdim=True)
    second_moment.lerp_(per_row_var.float(), 1 - beta2)
    vnorm = update.norm()
    update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
    update = update * (vnorm / update.norm().clamp_min(1e-10))
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu))

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
                        if p.size(-2) >= p.size(-1):
                            state["second_moment"] = torch.zeros((*p.shape[:-1], 1),
                                dtype=torch.float32, device=p.device)
                        else:
                            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),
                                dtype=torch.float32, device=p.device)
                    update = muon_update(p.grad, state["momentum"], state["second_moment"],
                                        mu=group["mu"])
                    # u/w-floor
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    scale = torch.where(u_fro / p_fro < TARGET_UW,
                                        TARGET_UW * p_fro / u_fro,
                                        torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size],
                                params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
torch.manual_seed(SEED)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
assert 8 % dist.get_world_size() == 0

if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console: print(s)
        if log:
            with open(logfile, "a") as f: print(s, file=f)

print0(code)
print0("=" * 100)
print0(f"train_gpt_krylov.py — ContraMuon + Krylov polar (NS_SUB_ITERS={NS_SUB_ITERS})")
print0(f"Running PyTorch {torch.version.__version__} on {torch.cuda.get_device_name(device)}"
       f" world_size={dist.get_world_size()}")
print0(f"CONTRA_MUON={CONTRA_MUON}  MU={MU}  LR={MUON_LR}  WD={MUON_WEIGHT_DECAY}  TARGET_UW={TARGET_UW}")
print0("=" * 100)

val_tokens  = 20 * 524288
batch_size  = 8 * 64 * 1024   # total tokens/step — identical to 8-GPU SOTA
mbs         = 64               # microbatch sequences per forward/backward
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator(
    "data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

########################################
#       Init & Optim Hyperparams       #
########################################

train_steps = 3225
val_regular_interval = 125
extra_val_steps = {3150, 3175, 3200, 3225}

for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight],  lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                  lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY, mu=MU)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta

########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)

training_time = 0
dist.barrier()
t0 = time.perf_counter()

for step in range(train_steps + 1):

    should_validate = (
        step == train_steps
        or (step > 0 and step % val_regular_interval == 0)
        or step in extra_val_steps
        or step == 0
    )
    if should_validate:
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               f" step_avg:{1000*training_time/max(step,1):.2f}ms", console=True)
        model.train()
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    inputs, targets = next(train_loader)
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite train loss at step {step} mb {i}: {loss.item()}")
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx:.3f}s"
           f" step_avg:{1000*approx/(step+1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
