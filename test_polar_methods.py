"""
Numerical test harness for orthogonalization methods used in Muon-style optimizers.

Goal: measure how close each method gets to the true polar decomposition
polar(M) = U V^T, on matrices with spectra similar to those seen during transformer
training. Compare against 3-NS and 5-NS as reference points.

Methods compared:
  - Newton-Schulz (1, 2, 3, 5 iterations) - baselines
  - Chebyshev polynomial via Clenshaw recurrence (numerically stable)
  - Chebyshev polynomial via Paterson-Stockmeyer (fewer matmuls)
  - Warm-started polar (uses polar of previous matrix in a sequence)
  - Block-Krylov polar (row-space-initialized for rectangular matrices)

Metrics:
  - Frobenius distance to true polar (relative)
  - Orthogonality defect ||X^T X - I||_F
  - Max |sigma_i(X) - 1|  (singular-value deviation from 1)
  - Cosine sim of vec(X) with vec(polar(M)) - the optimizer-relevant one
"""

import json
import math
import os
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

# Pre-warm numpy's BLAS/LAPACK thread pool before torch initializes its own,
# to avoid a LAPACK/OMP double-initialization segfault on macOS.
_np_warmup = np.linalg.svd(np.ones((4, 4)), compute_uv=False)
del _np_warmup

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Matrix generators - synthesize matrices with realistic spectra
# ---------------------------------------------------------------------------

def gen_with_spectrum(n, m, sigmas, seed=None):
    """Random matrix with prescribed singular value spectrum."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    A = torch.randn(n, m, generator=g)
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    sigmas = torch.tensor(sigmas, dtype=torch.float64)
    return U @ torch.diag(sigmas) @ Vh


def power_law_sigmas(k, alpha):
    s = np.array([(i + 1) ** (-alpha) for i in range(k)], dtype=np.float64)
    return s / s[0]  # normalize so sigma_max = 1


def exp_sigmas(k, rate):
    s = np.exp(-rate * np.arange(k) / k)
    return s


_iid_sigmas_cache = {}

def iid_sigmas(k, seed=0):
    key = (k, seed)
    if key not in _iid_sigmas_cache:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((k * 2, k)) / np.sqrt(k * 2)
        _iid_sigmas_cache[key] = np.linalg.svd(A, compute_uv=False)
    return _iid_sigmas_cache[key]


def gen_momentum_sequence(n, m, n_steps, beta=0.95, alpha=0.5, seed=0):
    """Simulate Muon momentum trajectory: M_t = beta*M_{t-1} + (1-beta)*G_t.
       Each G_t has structured power-law spectrum + random orthogonal frame."""
    rng = np.random.default_rng(seed)
    k = min(n, m)
    sigmas = power_law_sigmas(k, alpha)
    M = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
    seq = []
    for t in range(n_steps):
        G = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
        M = beta * M + (1 - beta) * G
        seq.append(M.clone())
    return seq


# ---------------------------------------------------------------------------
# Reference: true polar via SVD
# ---------------------------------------------------------------------------

def true_polar(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


# ---------------------------------------------------------------------------
# Method 1: Newton-Schulz (the modded-nanogpt quintic)
# ---------------------------------------------------------------------------

def newton_schulz(M, num_iters):
    """Standard quintic NS. 3 matmuls per iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.clone()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def ns_matmuls(num_iters):
    return 3 * num_iters


# ---------------------------------------------------------------------------
# Method 2a: Chebyshev polynomial via Clenshaw recurrence
#
# Evaluates p(A)M where p approximates x^{-1/2} on [a, b].
# Uses the Clenshaw recurrence in the Chebyshev basis — numerically stable
# because we never convert to the monomial basis.
#
# Matmul cost: 1 (A = MM^T) + (d-1) (Clenshaw loop, skipping two trivial
# initial steps) + 1 (final T@b1) + 1 (result @ M) = d+2 matmuls.
# ---------------------------------------------------------------------------

def _cheby_spectrum_bounds(Mt, sigma_min_frac=0.05):
    """Estimate [s_min^2, s_max^2] for the row space of Mt."""
    s = torch.linalg.svdvals(Mt)
    s_max = s[0].item()
    # clip small singular values so the approximation interval isn't too wide
    s_min = max(s[-1].item(), sigma_min_frac * s_max)
    return s_min ** 2, s_max ** 2


def cheby_clenshaw_polar(M, degree, sigma_min_frac=0.05):
    """
    X = p(MM^T) M, where p is the degree-d Chebyshev approximant to x^{-1/2}
    on [a, b], evaluated via the Clenshaw recurrence.

    Matmul count: d + 2  (1 for A=MM^T, d-1 loop steps + 1 final T@b1, 1 for result@M)
    """
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M

    a, b = _cheby_spectrum_bounds(Mt, sigma_min_frac)
    n_nodes = max(4 * (degree + 1), 300)
    t_nodes = np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)
    x_nodes = (b - a) / 2 * t_nodes + (a + b) / 2
    y_nodes = 1.0 / np.sqrt(x_nodes)
    cheb = Chebyshev.fit(t_nodes, y_nodes, degree)  # fit in [-1,1] directly
    c = cheb.coef  # c[0..degree], Chebyshev coefficients

    # A = Mt @ Mt^T, then T = (2A - (a+b)I) / (b-a)  [no matmul — free op]
    A = Mt @ Mt.T                            # matmul 1
    n = A.shape[0]
    ab_mid = float(a + b)
    ab_range = float(b - a)
    I = torch.eye(n, dtype=A.dtype, device=A.device)

    # Clenshaw recurrence: b_{d+1} = 0, b_d = c[d]*I (trivial, skip matmul)
    # For k = d-1 down to 1: b_k = 2*T@b_{k+1} - b_{k+2} + c[k]*I
    # Then result = T@b_1 - b_2 + c[0]*I
    # T@X = (2*A@X - ab_mid*X) / ab_range  — 1 matmul per step

    def T_at(X_mat):
        return (2.0 * (A @ X_mat) - ab_mid * X_mat) / ab_range

    if degree == 0:
        poly_I = float(c[0]) * I
        X = poly_I @ Mt
        return X.T if transposed else X

    # b_{d+1} = 0, b_d = c[d] * I
    b_next2 = None                      # b_{k+2}, starts as 0 (implicit)
    b_next1 = float(c[degree]) * I      # b_{d}, trivial (no matmul)

    for k in range(degree - 1, 0, -1):
        # b_k = 2*T@b_{k+1} - b_{k+2} + c[k]*I
        Tb = T_at(b_next1)              # 1 matmul per iteration
        b_k = 2.0 * Tb + float(c[k]) * I
        if b_next2 is not None:
            b_k = b_k - b_next2
        b_next2 = b_next1
        b_next1 = b_k

    # final: result = T@b_1 - b_2 + c[0]*I
    result = T_at(b_next1) + float(c[0]) * I   # 1 matmul
    if b_next2 is not None:
        result = result - b_next2

    X = result @ Mt                             # matmul: result @ M
    return X.T if transposed else X


def clenshaw_matmuls(degree):
    """1 (A) + max(0, degree-1) (loop) + 1 (final T@b1) + 1 (result@M)."""
    if degree == 0:
        return 1
    return degree + 2


# ---------------------------------------------------------------------------
# Method 2b: Chebyshev polynomial via Paterson-Stockmeyer in Chebyshev basis
#
# Builds T_1..T_s via 3-term recurrence, then evaluates degree-d poly in
# chunks of size s using those basis matrices.
# Matmul cost: 1 (A) + (s-1) (basis T_2..T_s) + 1 (stride T_{s+1}) +
#              ceil((d+1)/(s+1)) - 1 (chunked Horner) + 1 (result@M)
# ---------------------------------------------------------------------------

def cheby_ps_polar(M, degree, sigma_min_frac=0.05):
    """
    Paterson-Stockmeyer evaluation of the Chebyshev approximant to x^{-1/2}.
    """
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M

    a, b = _cheby_spectrum_bounds(Mt, sigma_min_frac)
    n_nodes = max(4 * (degree + 1), 300)
    t_nodes = np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)
    x_nodes = (b - a) / 2 * t_nodes + (a + b) / 2
    y_nodes = 1.0 / np.sqrt(x_nodes)
    cheb = Chebyshev.fit(t_nodes, y_nodes, degree)
    c = list(cheb.coef)

    A = Mt @ Mt.T
    n = A.shape[0]
    ab_mid = float(a + b)
    ab_range = float(b - a)
    I = torch.eye(n, dtype=A.dtype, device=A.device)

    def T_at(X_mat):
        return (2.0 * (A @ X_mat) - ab_mid * X_mat) / ab_range

    if degree == 0:
        return (float(c[0]) * I @ Mt).T if transposed else float(c[0]) * I @ Mt

    # Optimal chunk size
    s = max(1, int(round(math.sqrt(degree + 1))))

    # Build Chebyshev basis T_0=I, T_1=T(A), T_2=2T*T_1-T_0, ..., T_s
    T_basis = [None] * (s + 1)
    T_basis[0] = I
    T_basis[1] = T_at(I)           # 1 matmul (T*I = T)
    for i in range(2, s + 1):
        T_basis[i] = 2.0 * T_at(T_basis[i - 1]) - T_basis[i - 2]  # 1 matmul each

    # Stride matrix: T_{s+1} used as Horner stepping block
    T_stride = 2.0 * T_at(T_basis[s]) - T_basis[s - 1]  # 1 matmul

    # Pad coefficients so len is a multiple of (s+1)
    chunk = s + 1
    n_chunks = math.ceil((degree + 1) / chunk)
    c_padded = c + [0.0] * (n_chunks * chunk - len(c))

    # Horner over chunks (highest chunk first): result = T_stride @ result + block
    result = None
    for i in range(n_chunks - 1, -1, -1):
        seg = c_padded[i * chunk: (i + 1) * chunk]
        block = sum(float(seg[j]) * T_basis[j] for j in range(chunk))
        if result is None:
            result = block
        else:
            result = T_stride @ result + block   # 1 matmul per Horner step

    X = result @ Mt
    return X.T if transposed else X


def ps_cheby_matmuls(degree):
    """1 (A) + s (basis T_1..T_s) + 1 (T_stride) + (n_chunks-1) (Horner) + 1 (result@M)."""
    if degree == 0:
        return 1
    s = max(1, int(round(math.sqrt(degree + 1))))
    chunk = s + 1
    n_chunks = math.ceil((degree + 1) / chunk)
    return 1 + s + 1 + (n_chunks - 1) + 1


# ---------------------------------------------------------------------------
# Method 3: Warm-started polar (uses previous polar as starting point)
#
# Fix: NS refinement uses symmetric quintic coefficients (15/8, -10/8, 3/8)
# whose fixed point is x=1, so they work correctly on near-unitary matrices
# without any pre-scaling. The modded-nanogpt coefficients (a,b,c = 3.4445,
# -4.7750, 2.0315) map x=1 to 0.701 — badly wrong for refinement.
# ---------------------------------------------------------------------------

def warmstart_polar(M, X_prev, num_ns_refine=1):
    """Project M onto X_prev's basis, polar the small projection, refine."""
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M
    Xt = X_prev.T if transposed else X_prev
    Y = Xt.T @ Mt
    U_y, _, Vh_y = torch.linalg.svd(Y, full_matrices=False)
    polar_Y = U_y @ Vh_y
    X = Xt @ polar_Y
    # Symmetric quintic: fixed point at sigma=1, suitable for near-unitary input.
    # f(x) = (15x - 10x^3 + 3x^5) / 8, f(1) = (15-10+3)/8 = 1 exactly.
    for _ in range(num_ns_refine):
        A = X @ X.T
        B = (-10.0 / 8.0) * A + (3.0 / 8.0) * (A @ A)
        X = (15.0 / 8.0) * X + B @ X
    return X.T if transposed else X


def warmstart_matmuls(num_ns_refine=1):
    """1 for X_prev^T M, 1 for X_prev * polar_Y, 3 per refine step."""
    return 2 + 3 * num_ns_refine


# ---------------------------------------------------------------------------
# Method 4: Block Krylov on M^T M, initialized in row space of M
#
# Fix for rectangular matrices: instead of a random starting block (which
# lives mostly in the null space of M^T M when M is wide/tall), we initialize
# with Omega = Mt^T @ randn(n, block_size), which lands exactly in row(Mt).
# This costs 1 extra matmul but makes Kry-3x256 hit machine precision on
# rectangular matrices instead of frob_rel ~0.78.
# ---------------------------------------------------------------------------

def krylov_polar(M, num_steps=3, block_size=64, seed=0):
    """Block Krylov subspace of M^T M, project polar onto subspace."""
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M  # ensure Mt is tall (n >= m... wait, see below)
    # After transposing: Mt is the matrix whose row space we want to span.
    # For square: Mt = M (or M.T if tall). For rectangular: Mt = M (wide → tall after T).
    # Convention: Mt has shape (n, m) with n <= m (Mt is wide or square).
    # The Krylov subspace lives in R^m (column space of Mt^T).
    n, m = Mt.shape
    rectangular = (n < m)
    block_size_eff = min(block_size, n if rectangular else m)

    g = torch.Generator()
    g.manual_seed(seed)

    if rectangular:
        # Initialize in row space of Mt: Omega = Mt^T @ randn(n, bs)
        rand_init = torch.randn(n, block_size_eff, generator=g, dtype=Mt.dtype)
        Omega = Mt.T @ rand_init            # m x block_size_eff  [+1 matmul]
    else:
        Omega = torch.randn(m, block_size_eff, generator=g, dtype=Mt.dtype)

    Q, _ = torch.linalg.qr(Omega)
    Q_blocks = [Q]
    for _ in range(num_steps - 1):
        Y = Mt @ Q_blocks[-1]
        Z = Mt.T @ Y
        for Qb in Q_blocks:
            Z = Z - Qb @ (Qb.T @ Z)
        Z, _ = torch.linalg.qr(Z)
        Q_blocks.append(Z)
    Q = torch.cat(Q_blocks, dim=1)
    MQ = Mt @ Q
    U_mq, _, Vh_mq = torch.linalg.svd(MQ, full_matrices=False)
    X = U_mq @ Vh_mq @ Q.T
    return X.T if transposed else X


def krylov_matmuls(num_steps, rectangular=False):
    """2 large matmuls per Krylov step (M*B and M^T*B), +1 for final MQ.
    +1 if rectangular (row-space initialization)."""
    base = 2 * (num_steps - 1) + 1
    return base + (1 if rectangular else 0)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def frob_rel(X, target):
    return float((X - target).norm() / target.norm())


def ortho_defect(X):
    if X.shape[0] >= X.shape[1]:
        G = X.T @ X
    else:
        G = X @ X.T
    I = torch.eye(G.shape[0], dtype=G.dtype)
    return float((G - I).norm())


def sigma_dev(X):
    s = torch.linalg.svdvals(X)
    return float((s - 1).abs().max())


def cos_sim(X, target):
    return float(X.flatten() @ target.flatten() / (X.norm() * target.norm()))


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def evaluate(M, target, method_name, X, matmul_count):
    return {
        'method': method_name,
        'matmuls': matmul_count,
        'frob_rel': frob_rel(X, target),
        'ortho_def': ortho_defect(X),
        'sigma_dev': sigma_dev(X),
        'cos_sim': cos_sim(X, target),
    }


def run_one_matrix(M, X_prev=None, label=''):
    target = true_polar(M)
    results = []
    rectangular = (min(M.shape) < max(M.shape))

    # NS baselines
    for n_iter in [1, 2, 3, 5]:
        X = newton_schulz(M, n_iter)
        results.append(evaluate(M, target, f'NS-{n_iter}', X, ns_matmuls(n_iter)))

    # Chebyshev Clenshaw
    for d in [3, 5, 7, 9, 12, 16]:
        try:
            X = cheby_clenshaw_polar(M, d)
            results.append(evaluate(M, target, f'Cheby-d{d}', X, clenshaw_matmuls(d)))
        except Exception as e:
            print(f'  Cheby-d{d} failed: {e}')

    # Chebyshev PS
    for d in [7, 9, 12, 16]:
        try:
            X = cheby_ps_polar(M, d)
            results.append(evaluate(M, target, f'ChebyPS-d{d}', X, ps_cheby_matmuls(d)))
        except Exception as e:
            print(f'  ChebyPS-d{d} failed: {e}')

    # Warm-start (only when previous available)
    if X_prev is not None:
        for r in [0, 1, 2]:
            X = warmstart_polar(M, X_prev, num_ns_refine=r)
            results.append(evaluate(M, target, f'Warm-r{r}', X, warmstart_matmuls(r)))

    # Krylov
    k = min(M.shape)
    for ns in [2, 3, 4]:
        for bs in [32, 64, 128, 256]:
            # For rectangular, block_size_eff = min(bs, n) where n = min(shape)
            bs_eff = min(bs, k)
            if bs_eff * ns > k:
                continue
            X = krylov_polar(M, num_steps=ns, block_size=bs)
            results.append(evaluate(M, target, f'Kry-{ns}x{bs}',
                                    X, krylov_matmuls(ns, rectangular)))
    return results


def aggregate(rows):
    """Average over multiple trials with the same method label."""
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        bucket[r['method']].append(r)
    out = []
    for name, items in bucket.items():
        avg = {'method': name, 'matmuls': items[0]['matmuls']}
        for k in ['frob_rel', 'ortho_def', 'sigma_dev', 'cos_sim']:
            avg[k] = float(np.mean([it[k] for it in items]))
        out.append(avg)
    out.sort(key=lambda x: x['matmuls'])
    return out


def print_table(rows, title):
    print(f'\n=== {title} ===')
    print(f'{"method":<16} {"matmuls":>7}  {"frob_rel":>10}  {"ortho_def":>10}  {"sigma_dev":>10}  {"cos_sim":>10}')
    print('-' * 82)
    for r in rows:
        print(f'{r["method"]:<16} {r["matmuls"]:>7}  '
              f'{r["frob_rel"]:>10.3e}  {r["ortho_def"]:>10.3e}  '
              f'{r["sigma_dev"]:>10.3e}  {r["cos_sim"]:>10.6f}')


# ---------------------------------------------------------------------------
# Trajectory consistency test
#
# Simulates what actually happens inside a Muon optimizer over 200 steps:
#   W_0 = 0
#   X_t = method(M_t, X_{t-1})   [warm-start methods carry their own output]
#   W_t = W_{t-1} - lr * X_t
#
# Reference uses true_polar at each step. Warm-start methods receive their
# own previous OUTPUT as X_prev (not the true polar), so errors compound
# exactly as they would in a real training run.
# ---------------------------------------------------------------------------

def _make_methods(n, m):
    """
    Returns list of (name, matmuls, callable).
    callable signature: (M, X_prev) -> X_new
    X_prev is None for stateless methods.
    """
    rectangular = (n != m)
    methods = []

    # NS baselines
    for n_iter in [3, 5]:
        def _ns(M, _prev, ni=n_iter):
            return newton_schulz(M, ni)
        methods.append((f'NS-{n_iter}', ns_matmuls(n_iter), _ns))

    # Chebyshev Clenshaw
    for d in [5, 7, 9, 12]:
        def _cheby(M, _prev, deg=d):
            return cheby_clenshaw_polar(M, deg)
        methods.append((f'Cheby-d{d}', clenshaw_matmuls(d), _cheby))

    # Krylov variants that fit in budget
    k = min(n, m)
    for ns in [2, 3]:
        for bs in [64, 128, 256]:
            bs_eff = min(bs, k)
            if bs_eff * ns > k:
                continue
            def _kry(M, _prev, nsteps=ns, bsize=bs, rect=rectangular):
                return krylov_polar(M, num_steps=nsteps, block_size=bsize)
            methods.append((f'Kry-{ns}x{bs}', krylov_matmuls(ns, rectangular), _kry))

    # Warm-start variants (stateful — carry own X_prev)
    for r in [0, 1, 2]:
        def _warm(M, X_prev, refine=r):
            if X_prev is None:
                return true_polar(M)  # bootstrap first step with true polar
            return warmstart_polar(M, X_prev, num_ns_refine=refine)
        methods.append((f'Warm-r{r}', warmstart_matmuls(r), _warm))

    return methods


def run_trajectory_test(seq, lr=0.01):
    """
    Run trajectory simulation on a pre-generated momentum sequence.
    seq: list of T tensors, each shape (n, m).
    Returns list of dicts sorted by final_cos_sim descending.
    """
    T = len(seq)
    n, m = seq[0].shape
    methods = _make_methods(n, m)

    # Reference trajectory using true_polar
    W_ref = torch.zeros(n, m, dtype=torch.float64)
    polars_ref = []
    for t in range(T):
        P = true_polar(seq[t])
        polars_ref.append(P)
        W_ref = W_ref - lr * P

    results = []
    for name, matmuls, method_fn in methods:
        W = torch.zeros(n, m, dtype=torch.float64)
        X_prev = None
        step_cos_sims = []
        step_frob_rels = []
        step_sigma_devs = []
        W_snapshots = []   # for max_rel_err computation

        for t in range(T):
            X = method_fn(seq[t], X_prev)
            X_prev = X  # stateful: warm-start gets its own output next step
            W = W - lr * X
            step_cos_sims.append(cos_sim(X, polars_ref[t]))
            step_frob_rels.append(frob_rel(X, polars_ref[t]))
            step_sigma_devs.append(sigma_dev(X))
            W_snapshots.append(W.clone())

        # Build reference W snapshots
        W_ref_running = torch.zeros(n, m, dtype=torch.float64)
        W_ref_snapshots = []
        for t in range(T):
            W_ref_running = W_ref_running - lr * polars_ref[t]
            W_ref_snapshots.append(W_ref_running.clone())

        max_rel_err = max(
            float((W_snapshots[t] - W_ref_snapshots[t]).norm() /
                  (W_ref_snapshots[t].norm() + 1e-12))
            for t in range(T)
        )
        final_cs = cos_sim(W_snapshots[-1], W_ref_snapshots[-1])

        results.append({
            'method': name,
            'matmuls': matmuls,
            'final_cos_sim': final_cs,
            'max_rel_err': max_rel_err,
            'mean_step_cos_sim': float(np.mean(step_cos_sims)),
            'mean_step_frob_rel': float(np.mean(step_frob_rels)),
            'mean_step_sigma_dev': float(np.mean(step_sigma_devs)),
        })

    results.sort(key=lambda x: -x['final_cos_sim'])
    return results


def print_trajectory_table(rows, title):
    print(f'\n=== {title} ===')
    hdr = f'{"method":<16} {"matmuls":>7}  {"final_cos_sim":>14}  {"max_rel_err":>12}  {"mean_step_cs":>13}  {"mean_frob":>10}  {"mean_sigma_dev":>14}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["method"]:<16} {r["matmuls"]:>7}  '
              f'{r["final_cos_sim"]:>14.8f}  '
              f'{r["max_rel_err"]:>12.3e}  '
              f'{r["mean_step_cos_sim"]:>13.8f}  '
              f'{r["mean_step_frob_rel"]:>10.3e}  '
              f'{r["mean_step_sigma_dev"]:>14.3e}')


# ---------------------------------------------------------------------------
# Hook for real Muon momentum tensors
# ---------------------------------------------------------------------------

def run_trajectory_test_from_file(path, lr=0.01):
    """
    Load a list of momentum tensors from path (saved via torch.save) and run
    the trajectory test on them.

    Expected format: torch.save(list_of_tensors, path)
    All tensors must have the same shape (n, m) and dtype float32 or float64.
    """
    seq = torch.load(path, map_location='cpu')
    seq = [t.to(dtype=torch.float64) for t in seq]
    results = run_trajectory_test(seq, lr=lr)
    print_trajectory_table(results, f'Trajectory test: {os.path.basename(path)}')
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_vs_matmuls(traj_results, save_path='accuracy_vs_matmuls.png'):
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_names = {'NS-3', 'NS-5'}
    warm_names = {r['method'] for r in traj_results if r['method'].startswith('Warm')}
    krylov_names = {r['method'] for r in traj_results if r['method'].startswith('Kry')}
    cheby_names = {r['method'] for r in traj_results
                   if r['method'].startswith('Cheby') or r['method'].startswith('ChebyPS')}

    for r in traj_results:
        name = r['method']
        x = r['matmuls']
        y = r['final_cos_sim']

        if name in baseline_names:
            ax.scatter(x, y, color='red', s=120, zorder=5, marker='D')
            ax.annotate(name, (x, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=8, color='red', fontweight='bold')
        elif name in warm_names:
            ax.scatter(x, y, color='green', s=80, zorder=4, marker='^')
            ax.annotate(name, (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=7, color='green')
        elif name in krylov_names:
            ax.scatter(x, y, color='purple', s=80, zorder=4, marker='s')
            ax.annotate(name, (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=7, color='purple')
        elif name in cheby_names:
            ax.scatter(x, y, color='steelblue', s=80, zorder=4, marker='o')
            ax.annotate(name, (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=7, color='steelblue')
        else:
            ax.scatter(x, y, color='gray', s=60, zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Matmul count', fontsize=12)
    ax.set_ylabel('Final trajectory cos_sim (W_T vs reference)', fontsize=12)
    ax.set_title('Accuracy vs Matmul Budget — Muon polar approximators\n(200-step momentum trajectory, 768×768)', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=9, label='NS baselines'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Chebyshev (Clenshaw/PS)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='Krylov'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Warm-start'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'\nPlot saved to {save_path}')


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_cheby_fix():
    """
    Verify Chebyshev Clenshaw is numerically stable and beats NS-3 on well-
    conditioned spectra. For ill-conditioned spectra (powerlaw-0.7, exp-3),
    no fixed-degree polynomial can hit cos_sim>0.99 — check that we at least
    beat the old monomial evaluation (which had catastrophic cancellation).
    """
    print('\n--- Verifying Chebyshev fix (Cheby-d7 cos_sim on all spectra) ---')
    # Thresholds: well-conditioned spectra should clear 0.99;
    # ill-conditioned (cond ~400) are physically limited at degree 7.
    thresholds = {
        'iid':          0.99,
        'powerlaw-0.3': 0.99,
        'powerlaw-0.7': 0.75,   # cond~400, degree-7 poly cannot do better
        'exp-3':        0.96,   # moderate ill-conditioning
    }
    spectra = {
        'iid':          lambda k: iid_sigmas(k, seed=42),
        'powerlaw-0.3': lambda k: power_law_sigmas(k, 0.3),
        'powerlaw-0.7': lambda k: power_law_sigmas(k, 0.7),
        'exp-3':        lambda k: exp_sigmas(k, 3.0),
    }
    all_pass = True
    for spec_name, spec_fn in spectra.items():
        k = 768
        sigmas = spec_fn(k)
        M = gen_with_spectrum(768, 768, sigmas, seed=42)
        target = true_polar(M)
        X = cheby_clenshaw_polar(M, degree=7)
        cs = cos_sim(X, target)
        thr = thresholds[spec_name]
        status = 'PASS' if cs > thr else 'FAIL'
        if cs <= thr:
            all_pass = False
        print(f'  {spec_name:<16}: cos_sim={cs:.6f}  (threshold={thr})  [{status}]')
    return all_pass


def verify_krylov_rect(threshold=1e-6):
    """Spot-check that Kry-3x256 hits near machine precision on rectangular shapes."""
    print('\n--- Verifying Krylov fix (Kry-3x256 frob_rel on rectangular matrices) ---')
    all_pass = True
    for shape in [(768, 3072), (3072, 768)]:
        k = min(shape)
        sigmas = iid_sigmas(k, seed=42)
        M = gen_with_spectrum(*shape, sigmas, seed=7)
        target = true_polar(M)
        X = krylov_polar(M, num_steps=3, block_size=256)
        fr = frob_rel(X, target)
        status = 'PASS' if fr < threshold else 'FAIL'
        if fr >= threshold:
            all_pass = False
        print(f'  {shape[0]}x{shape[1]}: frob_rel={fr:.3e}  [{status}]')
    return all_pass


def verify_warmstart_fix():
    """Spot-check that Warm-r1 is now better than Warm-r0 (frob_rel lower)."""
    print('\n--- Verifying warm-start refinement fix (Warm-r1 < Warm-r0 frob_rel) ---')
    seq = gen_momentum_sequence(768, 768, n_steps=5, beta=0.95, alpha=0.5, seed=999)
    X_prev = true_polar(seq[0])
    # Use a step where error has accumulated — not step 1 (too easy)
    M = seq[3]
    target = true_polar(M)
    X0 = warmstart_polar(M, X_prev, num_ns_refine=0)
    X1 = warmstart_polar(M, X_prev, num_ns_refine=1)
    X2 = warmstart_polar(M, X_prev, num_ns_refine=2)
    fr0 = frob_rel(X0, target)
    fr1 = frob_rel(X1, target)
    fr2 = frob_rel(X2, target)
    print(f'  Warm-r0 frob_rel: {fr0:.3e}')
    print(f'  Warm-r1 frob_rel: {fr1:.3e}  [{"PASS" if fr1 < fr0 else "FAIL — not better"}]')
    print(f'  Warm-r2 frob_rel: {fr2:.3e}  [{"PASS" if fr2 < fr1 else "FAIL — not better"}]')
    return fr1 < fr0


# ---------------------------------------------------------------------------
# Stress trajectory tests
#
# Tractable on CPU: 256×256 matrices, 4 essential methods, shorter sequences.
# Saves partial JSON after each scenario so earlier results survive a stall.
#
# Focused method set:
#   Warm-r0 (2mm), Kry-2x128 (3mm, spans full space for 256×256),
#   NS-5 (15mm), Cheby-d9 (11mm)
# ---------------------------------------------------------------------------

def _stress_methods():
    """
    Returns list of (name, matmuls, fn) for the focused stress-test method set.
    fn signature: (M, X_prev) -> X_new  (X_prev is None for stateless methods)
    Warm-start bootstraps step 0 with true_polar if X_prev is None.

    Sized for 256×256 matrices on CPU (~minutes not hours):
      Kry-2x128: num_steps=2, block_size=128 → 2×128=256 columns = full space.
    """
    def _ns5(M, _prev):  return newton_schulz(M, 5)
    def _cd9(M, _prev):  return cheby_clenshaw_polar(M, 9)
    def _kry(M, _prev):  return krylov_polar(M, num_steps=2, block_size=128)
    def _w0(M, X_prev):
        return true_polar(M) if X_prev is None else warmstart_polar(M, X_prev, num_ns_refine=0)
    return [
        ('Warm-r0',   warmstart_matmuls(0),  _w0),
        ('Kry-2x128', krylov_matmuls(2),     _kry),
        ('NS-5',      ns_matmuls(5),          _ns5),
        ('Cheby-d9',  clenshaw_matmuls(9),    _cd9),
    ]


def _run_stress_scenario(seq, title, lr=0.01, checkpoint_every=None):
    """
    Core stress-test runner. Returns per-method dicts with:
      final_cos_sim, first_below_999 (step, 1-indexed), first_below_99,
      max_ortho_defect, max_rel_err, and (if checkpoint_every) step_cos_sims list.

    Uses incremental max_rel_err tracking — no W-snapshot accumulation.
    Per-method errors are caught and logged so one failure doesn't abort the run.
    """
    T = len(seq)
    n, m = seq[0].shape
    methods = _stress_methods()

    print(f'  precomputing {T} reference polars...', flush=True)
    polars_ref = [true_polar(seq[t]) for t in range(T)]

    # Precompute reference W norms (scalar per step) and final W for cos_sim.
    W_ref = torch.zeros(n, m, dtype=torch.float64)
    W_ref_norms = []
    for t in range(T):
        W_ref = W_ref - lr * polars_ref[t]
        W_ref_norms.append(float(W_ref.norm()))
    W_ref_final = W_ref  # alias — not mutated after this point

    results = []
    for name, matmuls, method_fn in methods:
        print(f'  {name}...', flush=True)
        try:
            W    = torch.zeros(n, m, dtype=torch.float64)
            diff = torch.zeros(n, m, dtype=torch.float64)
            X_prev = None
            first_below_999 = 999999
            first_below_99  = 999999
            max_ortho   = 0.0
            max_rel_err = 0.0
            step_cs_list = [] if checkpoint_every is not None else None

            for t in range(T):
                if t % 50 == 0 or t == T - 1:
                    print(f'    step {t+1}/{T}', end='\r', flush=True)
                X = method_fn(seq[t], X_prev)
                X_prev = X
                W    = W    - lr * X
                diff = diff - lr * (X - polars_ref[t])

                cs = cos_sim(X, polars_ref[t])
                od = ortho_defect(X)
                if od > max_ortho:
                    max_ortho = od
                if cs < 0.999 and first_below_999 == 999999:
                    first_below_999 = t + 1
                if cs < 0.99 and first_below_99 == 999999:
                    first_below_99 = t + 1
                rel_err = float(diff.norm()) / (W_ref_norms[t] + 1e-12)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                if checkpoint_every is not None and (t + 1) % checkpoint_every == 0:
                    step_cs_list.append((t + 1, cs))

            print(f'    step {T}/{T}  done', flush=True)
            final_cs = cos_sim(W, W_ref_final)

            r = {
                'method':           name,
                'matmuls':          matmuls,
                'final_cos_sim':    final_cs,
                'max_rel_err':      max_rel_err,
                'first_below_999':  first_below_999,
                'first_below_99':   first_below_99,
                'max_ortho_defect': max_ortho,
            }
            if step_cs_list is not None:
                r['step_cos_sims'] = step_cs_list
        except Exception as exc:
            print(f'    ERROR: {exc}', flush=True)
            r = {
                'method':           name,
                'matmuls':          matmuls,
                'error':            str(exc),
                'final_cos_sim':    float('nan'),
                'max_rel_err':      float('nan'),
                'first_below_999':  999999,
                'first_below_99':   999999,
                'max_ortho_defect': float('nan'),
            }
            if checkpoint_every is not None:
                r['step_cos_sims'] = []

        results.append(r)

    results.sort(key=lambda x: x['final_cos_sim'] if not (x['final_cos_sim'] != x['final_cos_sim']) else -1,
                 reverse=True)
    return results


def _print_stress_table(rows, title):
    print(f'\n=== {title} ===')
    hdr = (f'{"method":<12} {"matmuls":>7}  {"final_cs":>10}  '
           f'{"<0.999@step":>11}  {"<0.99@step":>10}  '
           f'{"max_rel_err":>11}  {"max_ortho":>10}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        if 'error' in r:
            print(f'{r["method"]:<12} {r["matmuls"]:>7}  ERROR: {r["error"]}')
            continue
        fb999 = str(r['first_below_999']) if r['first_below_999'] < 999999 else 'never'
        fb99  = str(r['first_below_99'])  if r['first_below_99']  < 999999 else 'never'
        print(f'{r["method"]:<12} {r["matmuls"]:>7}  '
              f'{r["final_cos_sim"]:>10.6f}  '
              f'{fb999:>11}  {fb99:>10}  '
              f'{r["max_rel_err"]:>11.3e}  '
              f'{r["max_ortho_defect"]:>10.3e}')


# ---------------------------------------------------------------------------
# Scenario 1: β-schedule warmup (β: 0.5 → 0.95 over first 100 steps)
# ---------------------------------------------------------------------------

def _gen_beta_schedule_seq(n, m, n_steps=500, seed=0):
    """
    Momentum sequence where β ramps from 0.5 to 0.95 over the first 100 steps,
    then stays at 0.95. Mimics optimizer warmup — early steps are near-iid
    (fast-changing), late steps are slow-drifting.
    """
    rng = np.random.default_rng(seed)
    k = min(n, m)
    sigmas = power_law_sigmas(k, alpha=0.5)
    M = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
    seq = []
    for t in range(n_steps):
        # Linear ramp: β(t) = 0.5 + 0.45 * min(t/100, 1)
        beta_t = 0.5 + 0.45 * min(t / 100.0, 1.0)
        G = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
        M = beta_t * M + (1 - beta_t) * G
        seq.append(M.clone())
    return seq


# ---------------------------------------------------------------------------
# Scenario 2: Spectrum shifts every 200 steps
# ---------------------------------------------------------------------------

def _gen_spectrum_shift_seq(n, m, n_steps=800, beta=0.95, seed=0):
    """
    Every 200 steps, the noise spectrum cycles through:
      powerlaw-0.3 → powerlaw-0.7 → exp-3 → iid → powerlaw-0.3 → ...
    Tests whether warm-start handles abrupt changes in matrix structure.
    800 steps = exactly 4 phases of 200 each.
    """
    rng = np.random.default_rng(seed)
    k = min(n, m)
    spec_cycle = [
        power_law_sigmas(k, 0.3),
        power_law_sigmas(k, 0.7),
        exp_sigmas(k, 3.0),
        iid_sigmas(k, seed=42),
    ]

    current_spec = spec_cycle[0]
    M = gen_with_spectrum(n, m, current_spec, seed=int(rng.integers(2**31)))
    seq = []
    for t in range(n_steps):
        phase = (t // 200) % len(spec_cycle)
        current_spec = spec_cycle[phase]
        G = gen_with_spectrum(n, m, current_spec, seed=int(rng.integers(2**31)))
        M = beta * M + (1 - beta) * G
        seq.append(M.clone())
    return seq


# ---------------------------------------------------------------------------
# Scenario 3: Gradient spikes at steps 150, 400, 750
# ---------------------------------------------------------------------------

def _gen_spike_seq(n, m, n_steps=500, beta=0.95, spike_scale=10.0,
                   spike_steps=None, seed=0):
    """
    Normal momentum sequence but with 10× scale gradient injections at
    specified steps. Tests recovery from outlier gradients.
    """
    if spike_steps is None:
        spike_steps = {100, 250, 400}
    rng = np.random.default_rng(seed)
    k = min(n, m)
    sigmas = power_law_sigmas(k, alpha=0.5)
    M = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
    seq = []
    for t in range(n_steps):
        scale = spike_scale if (t + 1) in spike_steps else 1.0
        G = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
        G = scale * G
        M = beta * M + (1 - beta) * G
        seq.append(M.clone())
    return seq


# ---------------------------------------------------------------------------
# Scenario 4: Long horizon — 5000 steps, powerlaw-0.5, β=0.95
# ---------------------------------------------------------------------------

def _gen_long_horizon_seq(n, m, n_steps=2000, beta=0.95, alpha=0.5, seed=0):
    """Pure powerlaw-0.5, β=0.95 for n_steps. Same as gen_momentum_sequence."""
    return gen_momentum_sequence(n, m, n_steps=n_steps, beta=beta,
                                 alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Main stress test runner
# ---------------------------------------------------------------------------

def run_stress_trajectory_test(n=256, m=256, lr=0.01, seed=0,
                                partial_json='stress_results.json'):
    """
    Run all four stress scenarios on 256×256 matrices (CPU-tractable).
    Saves partial JSON after each scenario completes.
    Returns dict of all results.

    Scenario lengths:
      1. β-schedule:      500 steps
      2. spectrum-shifts: 800 steps (4×200)
      3. gradient-spikes: 500 steps
      4. long-horizon:   2000 steps (checkpoint every 50)
    """
    results = {}

    def _save_partial():
        try:
            with open(partial_json, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f'  [warn] could not save partial JSON: {e}', flush=True)

    # ---- Scenario 1: β-schedule ----
    print('\n... stress scenario 1/4: β-schedule (β: 0.5→0.95 over 100 steps, 500 steps)',
          flush=True)
    try:
        seq1 = _gen_beta_schedule_seq(n, m, n_steps=500, seed=seed)
        r1 = _run_stress_scenario(seq1, 'beta-schedule', lr=lr)
        results['stress_beta_schedule'] = r1
        _print_stress_table(r1, 'Stress: β-schedule warmup (500 steps, 256×256)')
    except Exception as e:
        print(f'  scenario 1 FAILED: {e}', flush=True)
        results['stress_beta_schedule'] = {'error': str(e)}
    _save_partial()

    # ---- Scenario 2: Spectrum shifts ----
    print('\n... stress scenario 2/4: spectrum shifts every 200 steps (800 steps)',
          flush=True)
    try:
        seq2 = _gen_spectrum_shift_seq(n, m, n_steps=800, beta=0.95, seed=seed)
        r2 = _run_stress_scenario(seq2, 'spectrum-shift', lr=lr)
        results['stress_spectrum_shift'] = r2
        _print_stress_table(r2, 'Stress: spectrum shifts every 200 steps (800 steps, 256×256)')
    except Exception as e:
        print(f'  scenario 2 FAILED: {e}', flush=True)
        results['stress_spectrum_shift'] = {'error': str(e)}
    _save_partial()

    # ---- Scenario 3: Gradient spikes ----
    print('\n... stress scenario 3/4: gradient spikes 10× at steps 100, 250, 400 (500 steps)',
          flush=True)
    try:
        seq3 = _gen_spike_seq(n, m, n_steps=500, beta=0.95,
                              spike_scale=10.0, spike_steps={100, 250, 400}, seed=seed)
        r3 = _run_stress_scenario(seq3, 'gradient-spikes', lr=lr)
        results['stress_gradient_spikes'] = r3
        _print_stress_table(r3, 'Stress: gradient spikes 10× at steps 100, 250, 400 (500 steps, 256×256)')
    except Exception as e:
        print(f'  scenario 3 FAILED: {e}', flush=True)
        results['stress_gradient_spikes'] = {'error': str(e)}
    _save_partial()

    # ---- Scenario 4: Long horizon (2000 steps, cos_sim tracked every 50) ----
    print('\n... stress scenario 4/4: long horizon 2000 steps (this takes the most time)',
          flush=True)
    try:
        seq4 = _gen_long_horizon_seq(n, m, n_steps=2000, beta=0.95, alpha=0.5, seed=seed)
        r4 = _run_stress_scenario(seq4, 'long-horizon', lr=lr, checkpoint_every=50)
        results['stress_long_horizon'] = r4
        _print_stress_table(r4, 'Stress: long horizon (2000 steps, powerlaw-0.5, β=0.95, 256×256)')
        _plot_long_horizon(r4, n_steps=2000, save_path='stress_trajectories.png')
    except Exception as e:
        print(f'  scenario 4 FAILED: {e}', flush=True)
        results['stress_long_horizon'] = {'error': str(e)}
    _save_partial()

    # ---- Cross-scenario summary table ----
    _print_stress_summary(results)

    return results


def _plot_long_horizon(rows, n_steps=2000, save_path='stress_trajectories.png'):
    """Per-step cos_sim curves for the long-horizon scenario."""
    colors = {
        'Warm-r0':   ('green',     '-',  2.0),
        'Kry-2x128': ('purple',    '-',  2.0),
        'NS-5':      ('darkred',   '-',  1.5),
        'Cheby-d9':  ('steelblue', '-',  1.5),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for r in rows:
        name = r['method']
        if 'step_cos_sims' not in r or not r['step_cos_sims']:
            continue
        steps = [s for s, _ in r['step_cos_sims']]
        vals  = [v for _, v in r['step_cos_sims']]
        col, ls, lw = colors.get(name, ('gray', '-', 1.0))
        ax.plot(steps, vals, label=f'{name} ({r["matmuls"]}mm)',
                color=col, linestyle=ls, linewidth=lw)

    ax.axhline(0.999, color='black', linestyle=':', linewidth=0.8, alpha=0.6, label='cos_sim=0.999')
    ax.axhline(0.99,  color='black', linestyle='--', linewidth=0.8, alpha=0.4, label='cos_sim=0.99')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Per-step cos_sim(X_t, polar(M_t))', fontsize=12)
    ax.set_title(f'Long-horizon stress test: per-step polar accuracy\n'
                 f'({n_steps} steps, powerlaw-0.5, β=0.95, 256×256, warm-start uses own output)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.01))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'\nStress plot saved to {save_path}')


def _print_stress_summary(results):
    """
    Cross-scenario summary: one row per method, columns = scenarios.
    Shows (final_cos_sim | <0.999@step | <0.99@step) per scenario.
    """
    scenario_keys = [
        ('stress_beta_schedule',   'β-sched(500)'),
        ('stress_spectrum_shift',  'spec-shift(800)'),
        ('stress_gradient_spikes', 'spikes(500)'),
        ('stress_long_horizon',    'long-horiz(2000)'),
    ]

    # Collect all method names that appeared in any scenario
    method_order = ['Warm-r0', 'Kry-2x128', 'NS-5', 'Cheby-d9']

    def _fmt(r):
        if r is None or 'error' in r:
            return 'ERR'
        fb999 = str(r['first_below_999']) if r['first_below_999'] < 999999 else '—'
        fb99  = str(r['first_below_99'])  if r['first_below_99']  < 999999 else '—'
        return f'{r["final_cos_sim"]:.4f}|{fb999:>5}|{fb99:>5}'

    # Build lookup: scenario_key -> method_name -> row
    lookup = {}
    for sk, _ in scenario_keys:
        lookup[sk] = {}
        rows = results.get(sk, [])
        if isinstance(rows, list):
            for row in rows:
                lookup[sk][row['method']] = row

    col_w = 22
    print('\n\n=== STRESS SUMMARY: final_cs | <0.999@step | <0.99@step ===')
    header = f'{"method":<12}' + ''.join(f'  {sl:<{col_w}}' for _, sl in scenario_keys)
    print(header)
    print('-' * len(header))
    for meth in method_order:
        row_str = f'{meth:<12}'
        for sk, _ in scenario_keys:
            r = lookup[sk].get(meth)
            row_str += f'  {_fmt(r):<{col_w}}'
        print(row_str)


# ---------------------------------------------------------------------------
# Refresh-rate sweep for warm-start
#
# Warm-r0 + periodic Kry-2x128 refresh every K steps.
# At each refresh step we pay 3 matmuls (Kry-2x128) instead of 2 (Warm-r0).
# Amortized cost per step = 2 + 1/K  (approaches 2 as K→∞, equals 3 at K=1).
# ---------------------------------------------------------------------------

def run_trajectory_with_refresh(n=256, m=256, n_steps=2000,
                                 beta=0.95, alpha=0.5, lr=0.01,
                                 K_values=None, seed=0):
    """
    Long-horizon trajectory (same sequence as stress scenario 4) but Warm-r0
    periodically refreshes X_prev with Kry-2x128 every K steps.

    K=1   → pure Kry-2x128 (3 matmuls/step)
    K=inf → pure Warm-r0    (2 matmuls/step)

    Reports: K, amortized matmuls/step, final_cos_sim, max_rel_err.
    Uses incremental max_rel_err — no W-snapshot accumulation.
    """
    if K_values is None:
        K_values = [25, 100, 500, 999999]  # 999999 ≈ ∞ (never refresh after bootstrap)

    seq = _gen_long_horizon_seq(n, m, n_steps=n_steps, beta=beta,
                                alpha=alpha, seed=seed)

    # Precompute reference polars once
    print(f'  precomputing {n_steps} reference polars...', flush=True)
    polars_ref = [true_polar(seq[t]) for t in range(n_steps)]

    # Precompute reference W norms (scalar) for incremental max_rel_err
    W_ref = torch.zeros(n, m, dtype=torch.float64)
    W_ref_norms = []
    for t in range(n_steps):
        W_ref = W_ref - lr * polars_ref[t]
        W_ref_norms.append(float(W_ref.norm()))
    W_ref_final = W_ref

    rows = []
    for K in K_values:
        label = str(K) if K < 999999 else '∞'
        print(f'  K={label}...', flush=True)
        W    = torch.zeros(n, m, dtype=torch.float64)
        diff = torch.zeros(n, m, dtype=torch.float64)
        X_prev = None
        total_matmuls = 0
        max_rel_err = 0.0

        for t in range(n_steps):
            if t % 100 == 0 or t == n_steps - 1:
                print(f'    step {t+1}/{n_steps}', end='\r', flush=True)
            # Refresh on step 0 (bootstrap) and every K steps thereafter
            if X_prev is None or (t % K == 0):
                X = krylov_polar(seq[t], num_steps=2, block_size=128)
                total_matmuls += krylov_matmuls(2)   # 3
            else:
                X = warmstart_polar(seq[t], X_prev, num_ns_refine=0)
                total_matmuls += warmstart_matmuls(0)  # 2
            X_prev = X
            W    = W    - lr * X
            diff = diff - lr * (X - polars_ref[t])

            rel_err = float(diff.norm()) / (W_ref_norms[t] + 1e-12)
            if rel_err > max_rel_err:
                max_rel_err = rel_err

        print(f'    step {n_steps}/{n_steps}  done', flush=True)
        final_cs = cos_sim(W, W_ref_final)
        amortized = total_matmuls / n_steps

        rows.append({
            'K':              label,
            'amortized_mm':   amortized,
            'final_cos_sim':  final_cs,
            'max_rel_err':    max_rel_err,
            'total_matmuls':  total_matmuls,
        })

    _print_refresh_table(rows, n_steps)
    return rows


def _print_refresh_table(rows, n_steps):
    print(f'\n=== Warm-r0 + periodic Kry-2x128 refresh ({n_steps} steps, 256×256) ===')
    hdr = (f'{"K":>6}  {"amort_mm/step":>13}  '
           f'{"final_cos_sim":>14}  {"max_rel_err":>12}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["K"]:>6}  {r["amortized_mm"]:>13.3f}  '
              f'{r["final_cos_sim"]:>14.8f}  {r["max_rel_err"]:>12.3e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SHAPES = [(768, 768), (768, 3072), (3072, 768)]
    SPECTRA = {
        'iid':           lambda k: iid_sigmas(k, seed=42),
        'powerlaw-0.3':  lambda k: power_law_sigmas(k, 0.3),
        'powerlaw-0.7':  lambda k: power_law_sigmas(k, 0.7),
        'exp-3':         lambda k: exp_sigmas(k, 3.0),
    }
    N_TRIALS = 3

    # ---- Verification checks ----
    ok_cheby = verify_cheby_fix()
    ok_kry   = verify_krylov_rect()
    ok_warm  = verify_warmstart_fix()
    if not (ok_cheby and ok_kry and ok_warm):
        print('\nWARNING: one or more verification checks failed — inspect above.')

    all_results = {}

    # ---- Per-matrix accuracy tables (existing tests, now with Cheby) ----
    for shape in SHAPES:
        for spec_name, spec_fn in SPECTRA.items():
            label = f'{shape[0]}x{shape[1]}, {spec_name}'
            print(f'\n... running {label}')
            rows = []
            for trial in range(N_TRIALS):
                k = min(shape)
                sigmas = spec_fn(k)
                M = gen_with_spectrum(*shape, sigmas, seed=1000 * trial + 7)
                rows.extend(run_one_matrix(M))
            agg = aggregate(rows)
            all_results[label] = agg
            print_table(agg, label)

    # ---- Warm-start test on short momentum sequence (existing test) ----
    print('\n... running momentum-sequence test (warm-start applicable, 8 steps)')
    seq_rows = []
    for trial in range(N_TRIALS):
        seq = gen_momentum_sequence(768, 768, n_steps=8, beta=0.95, alpha=0.5,
                                    seed=2000 * trial + 11)
        X_prev = true_polar(seq[0])
        for t in range(1, len(seq)):
            seq_rows.extend(run_one_matrix(seq[t], X_prev=X_prev))
            X_prev = true_polar(seq[t])
    seq_agg = aggregate(seq_rows)
    all_results['momentum_seq_8step'] = seq_agg
    print_table(seq_agg, 'Momentum sequence (768x768, beta=0.95, 8 steps)')

    # ---- Trajectory consistency test (200 steps, error compounding) ----
    print('\n... running trajectory test (200 steps, warm-start uses own output)')
    seq200 = gen_momentum_sequence(768, 768, n_steps=200, beta=0.95, alpha=0.5, seed=12345)
    traj_results = run_trajectory_test(seq200, lr=0.01)
    all_results['trajectory_200step'] = traj_results
    print_trajectory_table(traj_results, 'Trajectory test (768x768, 200 steps, beta=0.95)')

    # ---- Stress trajectory tests (256×256, CPU-tractable) ----
    stress_results = run_stress_trajectory_test(n=256, m=256, lr=0.01, seed=0)
    all_results.update(stress_results)

    # ---- Refresh-rate sweep (256×256, 2000 steps) ----
    refresh_results = run_trajectory_with_refresh(
        n=256, m=256, n_steps=2000, beta=0.95, alpha=0.5, lr=0.01,
        K_values=[25, 100, 500, 999999], seed=0,
    )
    all_results['refresh_sweep'] = refresh_results

    # ---- Save results ----
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nResults saved to results.json')

    # ---- Plots ----
    plot_accuracy_vs_matmuls(traj_results)


if __name__ == '__main__':
    main()
