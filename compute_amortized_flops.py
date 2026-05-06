"""
compute_amortized_flops.py — Check 2: Honest matmul FLOP accounting for WarmMuon.

Enumerates all parameters in GPT-2 124M (12 layers, d_model=768, d_ff=3072),
classifies each by optimizer path, and computes matmul FLOPs per optimizer step
for both NS-5 (baseline) and WarmMuon.

Usage:
    python compute_amortized_flops.py [--refresh-rate RATE]

    --refresh-rate: steady-state refresh rate from Check 1 (default: 0.05)
                    affects amortized cost of square warm-start params.
"""

import sys
import math

# ── Architecture constants ────────────────────────────────────────────────────

N_LAYERS    = 12
D_MODEL     = 768
D_FF        = 4 * D_MODEL   # 3072
N_HEADS     = D_MODEL // 128  # 6
HEAD_DIM    = 128
VOCAB       = 50304

# ── Matmul FLOP formula ───────────────────────────────────────────────────────
# One matmul (n, k) × (k, m) costs 2*n*k*m FLOPs.
# For a param of shape (n_rows, n_cols), the orthogonalizer works on (n, m)
# where n = min(n_rows, n_cols), m = max(n_rows, n_cols).
# (tall params are transposed internally)

def matmul_flops(n_rows, n_cols, n_matmuls):
    """Total FLOPs for n_matmuls on a param of shape (n_rows, n_cols)."""
    n = min(n_rows, n_cols)
    m = max(n_rows, n_cols)
    # Each "large" matmul is roughly (n, n) × (n, m) or (m, n) × (n, m)
    # NS-5: operates on the short side, cost ≈ 2*n*n*m per matmul iteration
    # (15 matmuls total: each iter does A=X@X.T (n²m), B ops (n³), X update (n²m) → ~3n²m)
    # Standard formula used in the literature: cost = n_matmuls * 2 * n * n * m
    return n_matmuls * 2 * n * n * m


# ── NS-5 baseline matmul count ────────────────────────────────────────────────
# zeropower_via_newtonschulz5 in train_gpt_simple.py does 12 iterations
# of the quintic with a=2, b=-1.5, c=0.5:
#   A = X @ X.mT          → 1 matmul (n×m @ m×n = n×n)
#   B = b*A + c*A@A        → 1 matmul (A@A = n×n @ n×n = n×n, but A is square so n²·n)
#   X = a*X + B@X          → 1 matmul (B@X = n×n @ n×m = n×m)
#   Total per iter: 3 matmuls, all of size dominated by n²m
# 12 iterations × 3 = 36 matmuls... but that overcounts because A@A uses
# the square n×n matrix. In standard Muon literature the "15 matmuls" count
# refers to 5 quintic iterations × 3 matmuls. We use the train_gpt_simple.py
# actual iteration count (12) but report relative to NS-5 convention.

NS5_ITERS    = 12   # actual iters in train_gpt_simple.py
NS5_MM_PER_ITER = 3
NS5_MATMULS  = NS5_ITERS * NS5_MM_PER_ITER   # 36

# WarmMuon paths:
# - Krylov-2 (always used for rectangular aspect>2):
#     For tall M (n×m, n>m, so Mt=M.T is wide→transposed): n=min, m=max
#     Init (rectangular): 1 matmul (Mt.T @ randn = m×n @ n×bs)
#     Iter (num_steps-1=1): Mt@Q (n×n matmul) + Mt.T@Y (n×n matmul) = 2 matmuls
#     Final: MQ = Mt@Q (n×n matmul) + SVD(n×n) + U@Vh@Q.T
#     Total large matmuls ≈ 1 (init) + 2 (iter) + 1 (MQ) = 4 for rectangular
#     For square (non-rectangular): 0 init + 2 (iter) + 1 (MQ) = 3 matmuls
KRYLOV_MM_SQUARE = 3
KRYLOV_MM_RECT   = 4   # +1 for row-space init matmul

# - Warm-start (square, aspect≤2, steady state):
#     Y = X_prev.T @ M      → 1 matmul (n×n @ n×m = n×m, but X_prev is n×n → n²)
#     SVD(Y) is n×n → cheap
#     X = X_prev @ polar_Y  → 1 matmul (n×n @ n×n)
#     Drift check: X.T @ X  → 1 matmul (n×n)
#     Total: 3 matmuls per warm step (2 warm + 1 drift)
# On refresh (rate r): krylov_mm_square matmuls instead of 3
# Amortized = (1-r)*3 + r*krylov_mm_square
WARM_MM_PER_STEP = 3   # Y, X, drift check


def warmmuon_mm_per_step(n_rows, n_cols, refresh_rate=0.0):
    """Amortized matmuls per step for WarmMuon."""
    aspect = max(n_rows, n_cols) / min(n_rows, n_cols)
    if aspect > 2.0:
        return float(KRYLOV_MM_RECT)
    else:
        return (1.0 - refresh_rate) * WARM_MM_PER_STEP + refresh_rate * KRYLOV_MM_SQUARE


# ── Parameter enumeration ─────────────────────────────────────────────────────

def enumerate_params():
    """
    Returns list of (param_name, n_rows, n_cols, optimizer) for GPT-2 124M.
    optimizer: 'muon' or 'adam'
    """
    params = []

    # Embedding (Adam)
    params.append(("embed.weight",       VOCAB,    D_MODEL, "adam"))
    params.append(("proj.weight",        VOCAB,    D_MODEL, "adam"))  # lm_head
    # norm gains are 1D → Adam

    for i in range(N_LAYERS):
        prefix = f"blocks.{i}"

        # Attention: q, k, v (D_MODEL × D_MODEL), proj (D_MODEL × D_MODEL)
        params.append((f"{prefix}.attn.q.weight",    D_MODEL, D_MODEL, "muon"))
        params.append((f"{prefix}.attn.k.weight",    D_MODEL, D_MODEL, "muon"))
        params.append((f"{prefix}.attn.v.weight",    D_MODEL, D_MODEL, "muon"))
        params.append((f"{prefix}.attn.proj.weight", D_MODEL, D_MODEL, "muon"))

        # MLP: fc (D_FF × D_MODEL = 3072×768), proj (D_MODEL × D_FF = 768×3072)
        params.append((f"{prefix}.mlp.fc.weight",    D_FF,    D_MODEL, "muon"))
        params.append((f"{prefix}.mlp.proj.weight",  D_MODEL, D_FF,    "muon"))

        # Biases, norm gains → 1D, Adam (not listed here, not relevant for FLOP count)

    return params


# ── Main accounting ───────────────────────────────────────────────────────────

def compute_and_print(refresh_rate_square: float = 0.05):
    params = enumerate_params()
    muon_params  = [(n, r, c) for n, r, c, opt in params if opt == "muon"]
    adam_params  = [(n, r, c) for n, r, c, opt in params if opt == "adam"]

    total_params = sum(r * c for _, r, c, _ in params)
    muon_total   = sum(r * c for _, r, c in muon_params)

    # Split Muon params by aspect ratio
    square_params = [(n, r, c) for n, r, c in muon_params
                     if max(r, c) / min(r, c) <= 2.0]
    rect_params   = [(n, r, c) for n, r, c in muon_params
                     if max(r, c) / min(r, c) >  2.0]

    # ── NS-5 baseline FLOPs ──
    ns5_flops_square = sum(matmul_flops(r, c, NS5_MATMULS) for _, r, c in square_params)
    ns5_flops_rect   = sum(matmul_flops(r, c, NS5_MATMULS) for _, r, c in rect_params)
    ns5_total        = ns5_flops_square + ns5_flops_rect

    # ── WarmMuon FLOPs ──
    wm_flops_square = sum(
        matmul_flops(r, c, warmmuon_mm_per_step(r, c, refresh_rate=refresh_rate_square))
        for _, r, c in square_params
    )
    wm_flops_rect   = sum(
        matmul_flops(r, c, warmmuon_mm_per_step(r, c, refresh_rate=0.0))
        for _, r, c in rect_params
    )
    wm_total = wm_flops_square + wm_flops_rect

    speedup = ns5_total / wm_total
    # Effective matmul count: scale WarmMuon total back to "equivalent NS-5 matmuls"
    effective_mm = wm_total / (ns5_total / NS5_MATMULS)

    rect_frac = wm_flops_rect / wm_total

    # ── Print ──
    SEP = "─" * 62
    print(SEP)
    print(f"Architecture: GPT-2 124M  ({N_LAYERS} layers, d={D_MODEL}, d_ff={D_FF})")
    print(SEP)
    print(f"Total parameters:        {total_params/1e6:6.1f}M")
    print(f"Muon-eligible (2D blocks): {muon_total/1e6:5.1f}M  "
          f"({100*muon_total/total_params:.0f}% of total params)")
    print()
    print(f"  Square (aspect ≤ 2): {len(square_params):3d} params")
    for name, r, c in square_params[:4]:
        mm = warmmuon_mm_per_step(r, c, refresh_rate=refresh_rate_square)
        print(f"    {name:<40}  ({r}×{c})  {mm:.2f} mm/step")
    if len(square_params) > 4:
        print(f"    ... and {len(square_params)-4} more (same shape)")
    print(f"    → WarmMuon:  {wm_flops_square/1e9:.3f} GFLOPs/step  "
          f"[refresh_rate={refresh_rate_square:.0%}, amortized {warmmuon_mm_per_step(D_MODEL,D_MODEL,refresh_rate_square):.2f} mm/param]")
    print(f"    → NS-5:      {ns5_flops_square/1e9:.3f} GFLOPs/step  [{NS5_MATMULS} mm/param]")
    print()
    print(f"  Rectangular (aspect > 2): {len(rect_params):3d} params")
    for name, r, c in rect_params[:4]:
        mm = warmmuon_mm_per_step(r, c)
        print(f"    {name:<40}  ({r}×{c})  {mm:.2f} mm/step  [Krylov-2]")
    if len(rect_params) > 4:
        print(f"    ... and {len(rect_params)-4} more (same shape)")
    print(f"    → WarmMuon:  {wm_flops_rect/1e9:.3f} GFLOPs/step  [{KRYLOV_MM_RECT} mm/param]")
    print(f"    → NS-5:      {ns5_flops_rect/1e9:.3f} GFLOPs/step  [{NS5_MATMULS} mm/param]")
    print()
    print(SEP)
    print(f"NS-5 total:     {ns5_total/1e9:7.3f} GFLOPs/step")
    print(f"WarmMuon total: {wm_total/1e9:7.3f} GFLOPs/step")
    print(f"Speedup:        {ns5_total/1e9:.3f} / {wm_total/1e9:.3f} = {speedup:.2f}×")
    print(f"Effective matmul count: {wm_total/(ns5_total/NS5_MATMULS):.1f} mm/step "
          f"(vs NS-5 baseline of {NS5_MATMULS})")
    print(SEP)
    print()
    print(f"FLOP breakdown within WarmMuon:")
    print(f"  Rectangular Krylov: {100*rect_frac:.1f}% of optimizer FLOPs")
    print(f"  Square warm-start:  {100*(1-rect_frac):.1f}% of optimizer FLOPs")
    print()
    if rect_frac > 0.80:
        print("  ⚠  Rectangular params dominate (>80% of optimizer FLOPs).")
        print("     Warm-start contributes a small slice of total speedup.")
        print("     WarmMuon-ColdOnly may match WarmMuon in practice.")
        print("     Story: 'Krylov beats NS-5' rather than 'warm-start beats NS-5'.")
    else:
        print(f"  Square warm-start contributes {100*(1-rect_frac):.0f}% of optimizer FLOPs,")
        print(f"  so warm-start speedup is the primary driver.")

    print()
    print(SEP)
    # Note: for square matrices, warm-start (2mm + 1 drift) = Krylov-square (3mm),
    # so ColdOnly and WarmMuon have identical FLOP counts. The speedup is purely
    # from count reduction (3 vs 36), not warm-start vs cold distinction.
    # The warm-start's advantage is latency (avoids repeated matmuls on the same
    # matrix in tight loops) and numerical quality, not raw FLOP count.
    print(f"Note on square params: warm-start (2mm + 1 drift check) = Krylov (3mm).")
    print(f"FLOP counts for WarmMuon and ColdOnly are identical — speedup is purely")
    print(f"count reduction (3 vs {NS5_MATMULS} matmuls), not warm-start vs cold distinction.")
    print()
    # What matters: how does refresh_rate affect total?
    wm_no_refresh_sq = sum(matmul_flops(r, c, warmmuon_mm_per_step(r, c, refresh_rate=0.0))
                           for _, r, c in square_params)
    wm_full_refresh_sq = sum(matmul_flops(r, c, KRYLOV_MM_SQUARE)
                             for _, r, c in square_params)
    # These should be identical since WARM_MM_PER_STEP == KRYLOV_MM_SQUARE == 3
    assert abs(wm_no_refresh_sq - wm_full_refresh_sq) < 1e6, \
        f"Expected identical: {wm_no_refresh_sq} vs {wm_full_refresh_sq}"
    print(f"Sensitivity to NS-5 iteration count:")
    for ns_mm in [15, 20, 30, 36]:
        ns_flops_alt = sum(matmul_flops(r, c, ns_mm) for _, r, c in muon_params)
        print(f"  NS-{ns_mm//3 if ns_mm % 3 == 0 else ns_mm} ({ns_mm} mm/param):  "
              f"{ns_flops_alt/1e9:.3f} GFLOPs/step  →  {ns_flops_alt/wm_total:.2f}× speedup over WarmMuon")
    print(SEP)


if __name__ == "__main__":
    # Parse optional --refresh-rate flag
    refresh_rate = 0.05
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--refresh-rate" and i + 2 <= len(sys.argv) - 1:
            refresh_rate = float(sys.argv[i + 2])

    print()
    compute_and_print(refresh_rate_square=refresh_rate)
