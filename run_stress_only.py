"""Run only the stress tests and refresh sweep.
Uses float32 for speed (stress tests measure qualitative behavior, not numerical precision).
Tracks max_rel_err incrementally to avoid O(T*n*m) snapshot storage.
"""
import json, math, os
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

_np_warmup = np.linalg.svd(np.ones((4, 4)), compute_uv=False)
del _np_warmup

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DTYPE = torch.float32   # float32: ~4x faster SVDs, same qualitative results


# ---- Inline all helpers to avoid import issues ----

def gen_with_spectrum(n, m, sigmas, seed=None):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    A = torch.randn(n, m, generator=g, dtype=DTYPE)
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    sigmas_t = torch.tensor(sigmas, dtype=DTYPE)
    return U @ torch.diag(sigmas_t) @ Vh


def power_law_sigmas(k, alpha):
    s = np.array([(i + 1) ** (-alpha) for i in range(k)], dtype=np.float32)
    return s / s[0]


def exp_sigmas(k, rate):
    return np.exp(-rate * np.arange(k) / k).astype(np.float32)


_iid_cache = {}
def iid_sigmas(k, seed=0):
    key = (k, seed)
    if key not in _iid_cache:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((k * 2, k)).astype(np.float32) / np.sqrt(k * 2)
        _iid_cache[key] = np.linalg.svd(A, compute_uv=False).astype(np.float32)
    return _iid_cache[key]


def gen_momentum_sequence(n, m, n_steps, beta=0.95, alpha=0.5, seed=0):
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


def true_polar(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def newton_schulz(M, num_iters):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.clone()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed: X = X.T
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed: X = X.T
    return X


def _cheby_spectrum_bounds(Mt, sigma_min_frac=0.05):
    s = torch.linalg.svdvals(Mt)
    s_max = s[0].item()
    s_min = max(s[-1].item(), sigma_min_frac * s_max)
    return float(s_min ** 2), float(s_max ** 2)


def cheby_clenshaw_polar(M, degree, sigma_min_frac=0.05):
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
        X = float(c[0]) * I @ Mt
        return X.T if transposed else X
    b_next2 = None
    b_next1 = float(c[degree]) * I
    for k in range(degree - 1, 0, -1):
        Tb = T_at(b_next1)
        b_k = 2.0 * Tb + float(c[k]) * I
        if b_next2 is not None:
            b_k = b_k - b_next2
        b_next2 = b_next1
        b_next1 = b_k
    result = T_at(b_next1) + float(c[0]) * I
    if b_next2 is not None:
        result = result - b_next2
    X = result @ Mt
    return X.T if transposed else X


def warmstart_polar(M, X_prev, num_ns_refine=1):
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M
    Xt = X_prev.T if transposed else X_prev
    Y = Xt.T @ Mt
    U_y, _, Vh_y = torch.linalg.svd(Y, full_matrices=False)
    polar_Y = U_y @ Vh_y
    X = Xt @ polar_Y
    for _ in range(num_ns_refine):
        A = X @ X.T
        B = (-10.0 / 8.0) * A + (3.0 / 8.0) * (A @ A)
        X = (15.0 / 8.0) * X + B @ X
    return X.T if transposed else X


def krylov_polar(M, num_steps=3, block_size=64, seed=0):
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M
    n, m = Mt.shape
    rectangular = (n < m)
    block_size_eff = min(block_size, n if rectangular else m)
    g = torch.Generator()
    g.manual_seed(seed)
    if rectangular:
        rand_init = torch.randn(n, block_size_eff, generator=g, dtype=Mt.dtype)
        Omega = Mt.T @ rand_init
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


def cos_sim(X, target):
    return float(X.flatten() @ target.flatten() / (X.norm() * target.norm()))


def ortho_defect(X):
    if X.shape[0] >= X.shape[1]:
        G = X.T @ X
    else:
        G = X @ X.T
    I = torch.eye(G.shape[0], dtype=G.dtype)
    return float((G - I).norm())


def ns_matmuls(n): return 3 * n
def clenshaw_matmuls(d): return 1 if d == 0 else d + 2
def krylov_matmuls(ns, rect=False): return 2 * (ns - 1) + 1 + (1 if rect else 0)
def warmstart_matmuls(r=1): return 2 + 3 * r


# ---- Sequence generators ----

def _gen_beta_schedule_seq(n, m, n_steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = min(n, m)
    sigmas = power_law_sigmas(k, alpha=0.5)
    M = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
    seq = []
    for t in range(n_steps):
        beta_t = 0.5 + 0.45 * min(t / 100.0, 1.0)
        G = gen_with_spectrum(n, m, sigmas, seed=int(rng.integers(2**31)))
        M = beta_t * M + (1 - beta_t) * G
        seq.append(M.clone())
    return seq


def _gen_spectrum_shift_seq(n, m, n_steps=1000, beta=0.95, seed=0):
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


def _gen_spike_seq(n, m, n_steps=1000, beta=0.95, spike_scale=10.0, spike_steps=None, seed=0):
    if spike_steps is None:
        spike_steps = {150, 400, 750}
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


def _gen_long_horizon_seq(n, m, n_steps=5000, beta=0.95, alpha=0.5, seed=0):
    return gen_momentum_sequence(n, m, n_steps=n_steps, beta=beta, alpha=alpha, seed=seed)


# ---- Core stress runner (incremental max_rel_err) ----

def _stress_methods():
    def _ns3(M, _p):  return newton_schulz(M, 3)
    def _ns5(M, _p):  return newton_schulz(M, 5)
    def _cd9(M, _p):  return cheby_clenshaw_polar(M, 9)
    def _kry(M, _p):  return krylov_polar(M, num_steps=3, block_size=64)
    def _w0(M, Xp):
        return true_polar(M) if Xp is None else warmstart_polar(M, Xp, num_ns_refine=0)
    def _w1(M, Xp):
        return true_polar(M) if Xp is None else warmstart_polar(M, Xp, num_ns_refine=1)
    return [
        ('Warm-r0',    warmstart_matmuls(0), _w0),
        ('Warm-r1',    warmstart_matmuls(1), _w1),
        ('Kry-3x64',   krylov_matmuls(3),    _kry),
        ('NS-3',       ns_matmuls(3),         _ns3),
        ('NS-5',       ns_matmuls(5),         _ns5),
        ('Cheby-d9',   clenshaw_matmuls(9),   _cd9),
    ]


def _run_stress_scenario(seq, lr=0.01, checkpoint_every=None):
    T = len(seq)
    n, m = seq[0].shape
    methods = _stress_methods()

    print(f'  precomputing {T} reference polars...', flush=True)
    polars_ref = [true_polar(seq[t]) for t in range(T)]

    # Precompute reference W norms (scalar per step) instead of storing tensors
    print(f'  precomputing reference W norms...', flush=True)
    W_ref = torch.zeros(n, m, dtype=DTYPE)
    W_ref_norms = []
    for t in range(T):
        W_ref = W_ref - lr * polars_ref[t]
        W_ref_norms.append(float(W_ref.norm()))
    W_ref_final = W_ref.clone()

    results = []
    for idx, (name, matmuls, method_fn) in enumerate(methods):
        print(f'  [{idx+1}/{len(methods)}] {name}...', flush=True)
        diff = torch.zeros(n, m, dtype=DTYPE)
        W_t  = torch.zeros(n, m, dtype=DTYPE)
        X_prev = None
        first_below_999 = 999999
        first_below_99  = 999999
        max_ortho = 0.0
        max_rel_err = 0.0
        step_cs_list = [] if checkpoint_every is not None else None

        for t in range(T):
            X = method_fn(seq[t], X_prev)
            X_prev = X
            W_t  = W_t  - lr * X
            diff = diff - lr * (X - polars_ref[t])

            cs = cos_sim(X, polars_ref[t])
            od = ortho_defect(X)
            if od > max_ortho: max_ortho = od
            if cs < 0.999 and first_below_999 == 999999: first_below_999 = t + 1
            if cs < 0.99  and first_below_99  == 999999: first_below_99  = t + 1

            rel_err = float(diff.norm()) / (W_ref_norms[t] + 1e-12)
            if rel_err > max_rel_err: max_rel_err = rel_err

            if checkpoint_every is not None and (t + 1) % checkpoint_every == 0:
                step_cs_list.append((t + 1, cs))

        final_cs = cos_sim(W_t, W_ref_final)
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
        results.append(r)

    results.sort(key=lambda x: -x['final_cos_sim'])
    return results


def _print_stress_table(rows, title):
    print(f'\n=== {title} ===')
    hdr = (f'{"method":<12} {"matmuls":>7}  {"final_cs":>10}  '
           f'{"<0.999@step":>11}  {"<0.99@step":>10}  '
           f'{"max_rel_err":>11}  {"max_ortho":>10}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        fb999 = str(r['first_below_999']) if r['first_below_999'] < 999999 else 'never'
        fb99  = str(r['first_below_99'])  if r['first_below_99']  < 999999 else 'never'
        print(f'{r["method"]:<12} {r["matmuls"]:>7}  '
              f'{r["final_cos_sim"]:>10.6f}  '
              f'{fb999:>11}  {fb99:>10}  '
              f'{r["max_rel_err"]:>11.3e}  '
              f'{r["max_ortho_defect"]:>10.3e}')


def _plot_long_horizon(rows, save_path='stress_trajectories.png'):
    colors = {
        'Warm-r0':   ('green',     '-',  2.0),
        'Warm-r1':   ('limegreen', '--', 1.5),
        'Kry-3x64':  ('purple',    '-',  2.0),
        'NS-3':      ('red',       '--', 1.5),
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
    ax.set_title('Long-horizon stress test: per-step polar accuracy\n'
                 '(5000 steps, powerlaw-0.5, β=0.95, 768×768, warm-start uses own output)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'\nStress plot saved to {save_path}')


def run_trajectory_with_refresh(n=768, m=768, n_steps=5000,
                                 beta=0.95, alpha=0.5, lr=0.01,
                                 K_values=None, seed=0, polars_ref=None):
    if K_values is None:
        K_values = [10, 50, 100, 500, 999999]

    seq = _gen_long_horizon_seq(n, m, n_steps=n_steps, beta=beta, alpha=alpha, seed=seed)

    if polars_ref is None:
        print(f'  precomputing {n_steps} reference polars for refresh sweep...', flush=True)
        polars_ref = [true_polar(seq[t]) for t in range(n_steps)]

    print(f'  precomputing reference W norms for refresh sweep...', flush=True)
    W_ref = torch.zeros(n, m, dtype=DTYPE)
    W_ref_norms = []
    for t in range(n_steps):
        W_ref = W_ref - lr * polars_ref[t]
        W_ref_norms.append(float(W_ref.norm()))
    W_ref_final = W_ref.clone()

    rows = []
    for K in K_values:
        label = str(K) if K < 999999 else '∞'
        print(f'  K={label}...', flush=True)
        W    = torch.zeros(n, m, dtype=DTYPE)
        diff = torch.zeros(n, m, dtype=DTYPE)
        X_prev = None
        total_matmuls = 0
        max_rel_err = 0.0

        for t in range(n_steps):
            if X_prev is None or (t % K == 0):
                X = krylov_polar(seq[t], num_steps=3, block_size=64)
                total_matmuls += krylov_matmuls(3)
            else:
                X = warmstart_polar(seq[t], X_prev, num_ns_refine=0)
                total_matmuls += warmstart_matmuls(0)
            X_prev = X
            W    = W    - lr * X
            diff = diff - lr * (X - polars_ref[t])
            rel_err = float(diff.norm()) / (W_ref_norms[t] + 1e-12)
            if rel_err > max_rel_err: max_rel_err = rel_err

        final_cs = cos_sim(W, W_ref_final)
        amortized = total_matmuls / n_steps
        rows.append({
            'K':             label,
            'amortized_mm':  amortized,
            'final_cos_sim': final_cs,
            'max_rel_err':   max_rel_err,
            'total_matmuls': total_matmuls,
        })

    print(f'\n=== Warm-r0 + periodic Kry-3x256 refresh ({n_steps} steps) ===')
    hdr = f'{"K":>6}  {"amort_mm/step":>13}  {"final_cos_sim":>14}  {"max_rel_err":>12}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["K"]:>6}  {r["amortized_mm"]:>13.3f}  '
              f'{r["final_cos_sim"]:>14.8f}  {r["max_rel_err"]:>12.3e}')
    return rows


def main():
    n, m, lr, seed = 256, 256, 0.01, 0
    all_results = {}

    print('\n... stress scenario 1/4: β-schedule (β: 0.5→0.95 over 100 steps, 1000 steps)', flush=True)
    seq1 = _gen_beta_schedule_seq(n, m, n_steps=1000, seed=seed)
    r1 = _run_stress_scenario(seq1, lr=lr)
    all_results['stress_beta_schedule'] = r1
    _print_stress_table(r1, 'Stress: β-schedule warmup (1000 steps)')

    print('\n... stress scenario 2/4: spectrum shifts every 200 steps (1000 steps)', flush=True)
    seq2 = _gen_spectrum_shift_seq(n, m, n_steps=1000, beta=0.95, seed=seed)
    r2 = _run_stress_scenario(seq2, lr=lr)
    all_results['stress_spectrum_shift'] = r2
    _print_stress_table(r2, 'Stress: spectrum shifts every 200 steps (1000 steps)')

    print('\n... stress scenario 3/4: gradient spikes 10× at steps 150, 400, 750 (1000 steps)', flush=True)
    seq3 = _gen_spike_seq(n, m, n_steps=1000, beta=0.95,
                          spike_scale=10.0, spike_steps={150, 400, 750}, seed=seed)
    r3 = _run_stress_scenario(seq3, lr=lr)
    all_results['stress_gradient_spikes'] = r3
    _print_stress_table(r3, 'Stress: gradient spikes 10× at steps 150, 400, 750 (1000 steps)')

    print('\n... stress scenario 4/4: long horizon 5000 steps', flush=True)
    seq4 = _gen_long_horizon_seq(n, m, n_steps=5000, beta=0.95, alpha=0.5, seed=seed)
    r4 = _run_stress_scenario(seq4, lr=lr, checkpoint_every=100)
    all_results['stress_long_horizon'] = r4
    _print_stress_table(r4, 'Stress: long horizon (5000 steps, powerlaw-0.5, β=0.95)')
    _plot_long_horizon(r4, save_path='stress_trajectories.png')

    print('\n... refresh sweep (5000 steps, reusing seq4 polars_ref)', flush=True)
    # Reuse seq4's reference polars (same seed=0, same gen_long_horizon_seq → same sequence)
    polars_ref4 = [true_polar(seq4[t]) for t in range(len(seq4))]
    refresh = run_trajectory_with_refresh(
        n=n, m=m, n_steps=5000, beta=0.95, alpha=0.5, lr=lr,
        K_values=[10, 50, 100, 500, 999999], seed=seed,
        polars_ref=polars_ref4,
    )
    all_results['refresh_sweep'] = refresh

    with open('stress_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nResults saved to stress_results.json')


if __name__ == '__main__':
    main()
