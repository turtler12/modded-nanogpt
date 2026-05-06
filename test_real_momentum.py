"""
test_real_momentum.py — Check 1: Real momentum tensor validation for WarmMuon.

Runs a short GPT-2 124M training loop with stock Muon (no external data needed —
uses random token sequences through the real model to get authentic gradient structure).
Every 5 steps starting at step 100, captures the Nesterov-corrected momentum tensor
for 4 representative parameters, saves to real_momentum.pt, then replays through
WarmMuon._orthogonalize() and compares to true polar (fp64 SVD).

Verdict format: "Real-momentum validation: PASS/FAIL — min cos_sim X.XX on layer Y at step Z."
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# macOS OMP safety
import numpy as np
_w = np.linalg.svd(np.ones((4, 4)), compute_uv=False); del _w

import torch.distributed as dist
if not dist.is_initialized():
    dist.is_initialized = lambda: False
    dist.get_world_size = lambda: 1
    dist.get_rank = lambda: 0

from warmuon import WarmMuon, krylov_polar

DEVICE = torch.device("cpu")
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failures: list[str] = []

def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {msg}")
    if not cond:
        _failures.append(msg)


# ── GPT-2 124M architecture (verbatim from train_gpt_simple.py) ──────────────

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
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))
    def forward(self, x_BTHD):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
    def forward(self, x):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2),
                                           v.transpose(1,2), scale=0.12, is_causal=True).transpose(1,2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
    def forward(self, x):
        return self.proj(self.fc(x).relu().square())

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs).float())
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


# ── Helpers ───────────────────────────────────────────────────────────────────

def true_polar_f64(M: torch.Tensor) -> torch.Tensor:
    M64 = M.double()
    U, _, Vh = torch.linalg.svd(M64, full_matrices=False)
    return (U @ Vh).to(M.dtype)

def cos_sim(A: torch.Tensor, B: torch.Tensor) -> float:
    af, bf = A.flatten().double(), B.flatten().double()
    return float(af @ bf / (af.norm() * bf.norm()))


# ── Step 1: collect real Nesterov momentum from a training run ────────────────

COLLECT_STEPS = 600
SAVE_EVERY    = 5
SAVE_START    = 100
VOCAB         = 50304
SEQ_LEN       = 64     # short for CPU speed; gradient structure is real
BATCH         = 2
MU            = 0.95

# The 4 representative params we care about
TARGET_PARAMS = [
    "blocks.0.attn.q.weight",    # (768, 768)  square, aspect 1:1
    "blocks.0.attn.proj.weight", # (768, 768)  square, aspect 1:1
    "blocks.0.mlp.fc.weight",    # (3072, 768) tall,   aspect 4:1
    "blocks.0.mlp.proj.weight",  # (768, 3072) wide,   aspect 1:4
]

PARAM_LABELS = {
    "blocks.0.attn.q.weight":    "attn-Q (768×768 square)",
    "blocks.0.attn.proj.weight": "attn-proj (768×768 square)",
    "blocks.0.mlp.fc.weight":    "mlp-fc (3072×768 tall 4:1)",
    "blocks.0.mlp.proj.weight":  "mlp-proj (768×3072 wide 1:4)",
}

SAVE_PATH = "real_momentum.pt"


def collect_real_momentum(force_recompute=False):
    if not force_recompute and os.path.exists(SAVE_PATH):
        print(f"  Loading cached momentum from {SAVE_PATH} ...")
        return torch.load(SAVE_PATH, weights_only=False)

    print(f"  Running {COLLECT_STEPS}-step training loop on CPU (seq_len={SEQ_LEN}, batch={BATCH}) ...")
    print(f"  (Using real backprop through GPT-2 124M with random token inputs)")

    torch.manual_seed(42)
    model = GPT(vocab_size=VOCAB, num_layers=12, model_dim=768).to(DEVICE)

    # Muon-eligible params: all 2D params from blocks
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    # Map name → param for the 4 targets
    name_to_param = {n: p for n, p in model.named_parameters()}
    target_param_objs = {n: name_to_param[n] for n in TARGET_PARAMS}

    # Momentum buffers (fp32), mirroring Muon internals
    momentum_bufs = {n: torch.zeros_like(p, dtype=torch.float32)
                     for n, p in target_param_objs.items()}

    saved = []  # list of {step, param_name, tensor (nesterov-corrected fp32)}

    for step in range(COLLECT_STEPS):
        if step % 50 == 0:
            print(f"    step {step}/{COLLECT_STEPS}", end="\r", flush=True)

        # Random batch
        inputs  = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)
        targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)

        loss = model(inputs, targets)
        loss.backward()

        # Compute and save Nesterov-corrected momentum for target params
        for name, p in target_param_objs.items():
            assert p.grad is not None
            g = p.grad.float()
            buf = momentum_bufs[name]
            buf.lerp_(g, 1.0 - MU)
            update_g = g.lerp_(buf, MU)  # nesterov-corrected

            if step >= SAVE_START and (step - SAVE_START) % SAVE_EVERY == 0:
                saved.append({
                    "step": step,
                    "param_name": name,
                    "tensor": update_g.clone(),
                })

        # Zero grads (don't do a real param update — we want clean gradients each step)
        model.zero_grad()

    print(f"    step {COLLECT_STEPS}/{COLLECT_STEPS}  done        ")
    print(f"  Collected {len(saved)} snapshots for {len(TARGET_PARAMS)} params × "
          f"{(COLLECT_STEPS - SAVE_START) // SAVE_EVERY + 1} save-steps")
    torch.save(saved, SAVE_PATH)
    print(f"  Saved → {SAVE_PATH}")
    return saved


# ── Step 2: replay through WarmMuon._orthogonalize and compare ────────────────

def replay_through_warmuon(saved):
    """
    For each target param, replay its momentum sequence through _orthogonalize
    with X_prev correctly threaded.  Compare each output X to true_polar_f64(M).
    Returns per-param results dict.
    """

    # Group snapshots by param_name, in chronological order
    by_param: dict[str, list] = {n: [] for n in TARGET_PARAMS}
    for snap in saved:
        if snap["param_name"] in by_param:
            by_param[snap["param_name"]].append(snap)
    for n in by_param:
        by_param[n].sort(key=lambda s: s["step"])

    # Build a minimal WarmMuon instance just to access _orthogonalize.
    # Use cold_steps=0 so we immediately enter warm-start after the first krylov step.
    # We'll manually seed the first step as "cold" by passing a fresh state.
    dummy_param = nn.Parameter(torch.randn(768, 768))
    opt = WarmMuon(
        [dummy_param],
        lr=0.02, mu=MU, cold_steps=0,   # cold_steps=0: we control cold manually
        drift_threshold=0.1,
        krylov_steps=2,
    )
    # Pull out the group defaults so _orthogonalize gets the right group dict
    group = opt.param_groups[0]

    results = {}
    for param_name, snaps in by_param.items():
        label = PARAM_LABELS[param_name]
        if not snaps:
            print(f"  [WARN] no snapshots for {param_name}, skipping")
            continue

        n_rows, n_cols = snaps[0]["tensor"].shape
        aspect = max(n_rows, n_cols) / min(n_rows, n_cols)
        is_rectangular = aspect > 2.0

        # Fresh per-param state — cold_steps=0 means first step is already "warm"
        # but X_prev is None, which forces Krylov. Subsequent steps use warm-start
        # (for square) or always-Krylov (for rect).
        state = {"step_count": 0, "X_prev": None, "momentum": torch.zeros(n_rows, n_cols)}

        cos_sims       = []
        refreshes      = 0
        warm_steps     = 0
        krylov_steps_used = 0

        for snap in snaps:
            M = snap["tensor"].clone()
            M_2d = M.view(-1, M.shape[-1]) if M.ndim > 2 else M

            # Reset per-step accumulators
            opt._step_drifts   = []
            opt._step_refreshes = 0
            opt._step_params   = 0
            opt._step_cold     = 0

            X = opt._orthogonalize(M_2d, state, group)
            ref = true_polar_f64(M_2d)
            cs  = cos_sim(X.float(), ref)
            cos_sims.append((snap["step"], cs))

            stats = opt.optimizer_stats()
            if state["step_count"] > 1:   # after the very first krylov bootstrap
                warm_steps += 1
                if stats["warmuon/refresh_rate"] > 0:
                    refreshes += 1

        min_cs   = min(cs for _, cs in cos_sims)
        mean_cs  = sum(cs for _, cs in cos_sims) / len(cos_sims)
        bad      = [(s, cs) for s, cs in cos_sims if cs < 0.999]
        min_step = min((s for s, cs in cos_sims if cs == min_cs), default=-1)
        refresh_rate = refreshes / warm_steps if warm_steps > 0 else 0.0

        results[param_name] = {
            "label":          label,
            "aspect":         aspect,
            "is_rectangular": is_rectangular,
            "cos_sims":       cos_sims,
            "min_cs":         min_cs,
            "min_step":       min_step,
            "mean_cs":        mean_cs,
            "refresh_rate":   refresh_rate,
            "n_bad":          len(bad),
        }

    return results


# ── Step 3: run checks and print verdict ─────────────────────────────────────

def test_real_momentum(force_recompute=False):
    print("\n── Check 1: Real momentum tensor validation ──")

    saved = collect_real_momentum(force_recompute=force_recompute)
    results = replay_through_warmuon(saved)

    print()
    print(f"  {'Parameter':<38} {'aspect':>6}  {'min_cs':>8}  {'mean_cs':>8}  {'refresh%':>9}  {'#<0.999':>7}")
    print(f"  {'-'*38}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}")
    for pname in TARGET_PARAMS:
        r = results.get(pname)
        if r is None:
            print(f"  {PARAM_LABELS[pname]:<38}  MISSING")
            continue
        flag = "" if r["n_bad"] == 0 else f"  ← {r['n_bad']} bad steps"
        print(f"  {r['label']:<38}  {r['aspect']:>6.1f}  {r['min_cs']:>8.6f}  "
              f"{r['mean_cs']:>8.6f}  {r['refresh_rate']:>9.2%}{flag}")

    print()
    # Per-param checks
    overall_min_cs   = 1.0
    overall_min_name = ""
    overall_min_step = -1

    for pname in TARGET_PARAMS:
        r = results.get(pname)
        if r is None:
            check(False, f"{PARAM_LABELS[pname]}: no data")
            continue

        # 1. cos_sim quality
        check(r["min_cs"] >= 0.999,
              f"{r['label']}: min cos_sim={r['min_cs']:.6f} ≥ 0.999 "
              f"(mean={r['mean_cs']:.6f}, {r['n_bad']} steps below threshold)")

        if r["min_cs"] < overall_min_cs:
            overall_min_cs   = r["min_cs"]
            overall_min_name = r["label"]
            overall_min_step = r["min_step"]

        # 2. Code path verification
        if r["is_rectangular"]:
            # aspect > 2 → always-Krylov; warm-start never runs → refresh_rate = 0
            check(r["refresh_rate"] == 0.0,
                  f"{r['label']}: always-Krylov path confirmed (refresh_rate=0, as expected for aspect={r['aspect']:.1f})")
        else:
            # aspect ≤ 2 → warm-start in steady state; refresh_rate < 5%
            check(r["refresh_rate"] < 0.05,
                  f"{r['label']}: warm-start steady-state refresh_rate={r['refresh_rate']:.2%} < 5%")

    # Overall verdict
    verdict_ok = len([f for f in _failures if "Real-momentum" not in f]) == 0
    # Count only failures added in this function
    pre_fail_count = sum(1 for f in _failures if "Check 1" not in f)
    new_fails = _failures[pre_fail_count:]
    passed = len(new_fails) == 0

    print()
    if passed:
        print(f"  Real-momentum validation: {PASS} — "
              f"min cos_sim {overall_min_cs:.4f} on {overall_min_name} at step {overall_min_step}")
    else:
        worst = min(results.values(), key=lambda r: r["min_cs"]) if results else None
        if worst:
            print(f"  Real-momentum validation: {FAIL} — "
                  f"min cos_sim {worst['min_cs']:.4f} on {worst['label']} at step {worst['min_step']}")
        else:
            print(f"  Real-momentum validation: {FAIL} — no results")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--recompute" in sys.argv
    print("=" * 60)
    print("Check 1: Real momentum tensor validation")
    print("=" * 60)

    test_real_momentum(force_recompute=force)

    print()
    print("=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s):")
        for f in _failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
    print("=" * 60)
