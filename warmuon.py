"""
WarmMuon optimizer
==================
Drop-in replacement for the Muon class in train_gpt_simple.py.

Core idea: warm-started polar decomposition.
If X_prev = polar(M_{t-1}) and M_t ≈ M_{t-1} (β=0.95 momentum), then:
    Y      = X_prev.T @ M_t          # matmul 1
    polar_Y = svd_polar(Y)            # small m×m SVD, microseconds
    X_t    = X_prev @ polar_Y         # matmul 2
This is exact when M_t is in col(X_prev), i.e. when momentum is smooth.

Guardrails:
  - Cold-start: first `cold_steps` steps use krylov_polar (~3 matmuls, exact)
  - Drift monitor: cheap ||X.T X - I|| check; triggers krylov_polar refresh
  - All refresh events are counted per-parameter for diagnostics

Memory cost:
  - Extra state vs Muon: X_prev (bf16, same shape as param) + step_count (int)
  - Extra state vs NorMuon: same as above (NorMuon already has second-moment v)
"""

import math
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Krylov polar — the exact fallback (~3 matmuls for square, exact to float eps)
# ---------------------------------------------------------------------------

def krylov_polar(M: Tensor, num_steps: int = 2, block_size: int | None = None) -> Tensor:
    """
    Block-Krylov polar decomposition. Exact for square M when
    block_size * num_steps >= min(M.shape).

    For tall M (rows >= cols): works directly.
    For wide M: transposes, runs, transposes back.

    Matmul cost: 2*(num_steps-1) large matmuls + 1 final MQ matmul
                 + 1 extra for row-space init on rectangular M
                 = ~3 for square num_steps=2 block_size=min(shape)

    Args:
        M:          2-D tensor (n, m), any dtype
        num_steps:  Krylov iterations (2 is sufficient for block_size=min(shape))
        block_size: columns per block; defaults to min(M.shape)
    """
    assert M.ndim == 2
    transposed = M.shape[0] > M.shape[1]
    Mt = M.T if transposed else M          # Mt is short/square: shape (n, m), n <= m

    n, m = Mt.shape
    bs = min(block_size or m, n)           # effective block size

    g = torch.Generator(device=Mt.device)
    g.manual_seed(0)                        # deterministic; state resets per call

    rectangular = (n < m)
    if rectangular:
        # Initialize in row space to avoid null-space issues
        rand_init = torch.randn(n, bs, generator=g, dtype=Mt.dtype, device=Mt.device)
        Omega = Mt.T @ rand_init           # (m, bs)  +1 matmul
    else:
        Omega = torch.randn(m, bs, generator=g, dtype=Mt.dtype, device=Mt.device)

    Q, _ = torch.linalg.qr(Omega)
    Q_blocks = [Q]

    for _ in range(num_steps - 1):
        Y = Mt @ Q_blocks[-1]              # (n, bs)  large matmul
        Z = Mt.T @ Y                       # (m, bs)  large matmul
        # Reorthogonalize against all previous blocks
        for Qb in Q_blocks:
            Z = Z - Qb @ (Qb.T @ Z)
        Z, _ = torch.linalg.qr(Z)
        Q_blocks.append(Z)

    Q = torch.cat(Q_blocks, dim=1)         # (m, k*bs)
    Q = Q[:, :n]                           # cap at n cols — Q has rank ≤ n anyway
    MQ = Mt @ Q                            # (n, n)  large matmul
    U_mq, _, Vh_mq = torch.linalg.svd(MQ, full_matrices=False)
    X = U_mq @ Vh_mq @ Q.T                # (n, m)

    return X.T if transposed else X


# ---------------------------------------------------------------------------
# WarmMuon
# ---------------------------------------------------------------------------

class WarmMuon(torch.optim.Optimizer):
    """
    Momentum Orthogonalized by Warm-started polar decomposition.

    Compatible with the Muon API: same constructor, same step() signature.
    Replaces Newton-Schulz (15 matmuls/step) with a 2-matmul warm-start,
    falling back to krylov_polar (~3 matmuls) during cold-start and drift events.

    Extra hyperparameters vs Muon:
        cold_steps      (int,   10)    — use krylov on first N steps per param
        drift_threshold (float, 0.01)  — ||X.T X - I||_F above which we refresh
        krylov_steps    (int,   2)     — Krylov iterations for fallback
        nesterov        (bool,  True)  — Nesterov-corrected momentum

    Logging (accumulated each step, reset each step):
        optimizer_stats() → dict with avg_drift, max_drift, refresh_rate, cold_active
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        cold_steps: int = 10,
        drift_threshold: float = 0.1,
        krylov_steps: int = 2,
        nesterov: bool = True,
    ):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            cold_steps=cold_steps,
            drift_threshold=drift_threshold,
            krylov_steps=krylov_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

        # Per-step diagnostic accumulators (reset each step call)
        self._step_drifts: list[float] = []
        self._step_refreshes: int = 0
        self._step_params: int = 0
        self._step_cold: int = 0

        # Log memory cost at construction
        total_param_bytes = sum(p.numel() * p.element_size()
                                for group in self.param_groups for p in group["params"])
        x_prev_bytes = sum(p.numel() * 2         # bf16 = 2 bytes
                           for group in self.param_groups for p in group["params"])
        print(
            f"WarmMuon: {len([p for g in self.param_groups for p in g['params']])} params, "
            f"param storage {total_param_bytes/1e9:.3f} GB, "
            f"X_prev overhead {x_prev_bytes/1e9:.3f} GB (bf16), "
            f"total optimizer state ≈ {(total_param_bytes*4 + x_prev_bytes)/1e9:.3f} GB "
            f"(momentum fp32 + X_prev bf16)"
        )

    # ------------------------------------------------------------------
    # Internal: warm-start orthogonalization for one matrix
    # ------------------------------------------------------------------

    def _orthogonalize(self, M: Tensor, state: dict, group: dict) -> Tensor:
        """
        Returns X ≈ polar(M), updating state['X_prev'].

        Uses warm-start (2 matmuls) in steady state, krylov_polar as fallback.
        """
        cold_steps     = group["cold_steps"]
        drift_threshold = group["drift_threshold"]
        krylov_steps   = group["krylov_steps"]

        step_count = state.get("step_count", 0)
        X_prev_bf16 = state.get("X_prev", None)

        # Warm-start is exact only when M stays within col(X_prev).
        # For tall/wide matrices the column space of X_prev is low-dimensional
        # relative to the ambient space, so warm-start degrades as M drifts.
        # Restrict warm-start to near-square matrices (aspect ratio ≤ 2).
        n_rows, n_cols = M.shape
        aspect = max(n_rows, n_cols) / max(min(n_rows, n_cols), 1)
        use_krylov = (step_count < cold_steps) or (X_prev_bf16 is None) or (aspect > 2.0)

        if not use_krylov:
            # Warm-start path: 2 large matmuls + one small (m×m) SVD
            X_prev = X_prev_bf16.to(M.dtype)
            # Ensure X_prev has same orientation as we'll produce
            Y = X_prev.T @ M                                         # matmul 1
            U_y, _, Vh_y = torch.linalg.svd(Y, full_matrices=False)  # small SVD
            polar_Y = U_y @ Vh_y
            X = X_prev @ polar_Y                                     # matmul 2

            # Drift check: ||X.T X - I||_F  (one matmul on the small side)
            m = min(X.shape)
            if X.shape[0] >= X.shape[1]:
                G = X.T @ X   # (m, m)
            else:
                G = X @ X.T   # (m, m)
            I = torch.eye(m, device=G.device, dtype=G.dtype)
            drift = float((G - I).norm())

            self._step_drifts.append(drift)

            if drift > drift_threshold:
                use_krylov = True          # fall through to refresh below
                self._step_refreshes += 1
            else:
                state["X_prev"] = X.to(torch.bfloat16)
                state["step_count"] = step_count + 1
                return X

        # Cold-start or drift refresh: exact Krylov polar
        if step_count < cold_steps:
            self._step_cold += 1
        bs = min(M.shape)
        X = krylov_polar(M, num_steps=krylov_steps, block_size=bs)

        if X_prev_bf16 is None:
            # First-ever drift check value (will be near 0)
            self._step_drifts.append(0.0)

        state["X_prev"] = X.to(torch.bfloat16)
        state["step_count"] = step_count + 1
        return X

    # ------------------------------------------------------------------
    # Public: diagnostic summary for logging
    # ------------------------------------------------------------------

    def optimizer_stats(self) -> dict:
        """
        Returns per-step diagnostic dict. Call after step(), before next step.
        Keys: avg_drift, max_drift, refresh_rate, cold_active, n_params
        """
        n = self._step_params
        drifts = self._step_drifts
        return {
            "warmuon/avg_drift":    float(sum(drifts) / len(drifts)) if drifts else 0.0,
            "warmuon/max_drift":    float(max(drifts)) if drifts else 0.0,
            "warmuon/refresh_rate": self._step_refreshes / n if n > 0 else 0.0,
            "warmuon/cold_active":  self._step_cold > 0,
            "warmuon/n_params":     n,
        }

    # ------------------------------------------------------------------
    # step() — distributed-compatible, mirrors Muon's all-gather pattern
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self):
        # Reset per-step accumulators
        self._step_drifts = []
        self._step_refreshes = 0
        self._step_params = 0
        self._step_cold = 0

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        for group in self.param_groups:
            mu        = group["mu"]
            lr        = group["lr"]
            wd        = group["weight_decay"]
            nesterov  = group["nesterov"]

            params = group["params"]
            # Pad to a multiple of world_size for the distributed round-robin
            pad = [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            params_padded = params + pad

            for base_i in range(0, len(params_padded), world_size):
                # Each rank owns one param in this round-robin slice
                local_idx = base_i + rank
                if local_idx < len(params):
                    p = params[local_idx]
                    g = p.grad
                    assert g is not None, f"WarmMuon: param at index {local_idx} has no grad"

                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                        state["X_prev"] = None
                        state["step_count"] = 0

                    # Nesterov momentum (fp32)
                    buf = state["momentum"]
                    g_f32 = g.float()
                    buf.lerp_(g_f32, 1.0 - mu)
                    update_g = g_f32.lerp_(buf, mu) if nesterov else buf.clone()

                    # Reshape to 2D for orthogonalization (handles batched params)
                    orig_shape = update_g.shape
                    M = update_g.view(-1, orig_shape[-1]) if update_g.ndim > 2 else update_g

                    # Orthogonalize
                    self._step_params += 1
                    X = self._orthogonalize(M, state, group)

                    # Scale: match Muon's max(1, n/m)^0.5 normalization
                    n_rows, n_cols = M.shape
                    scale = max(1.0, n_rows / n_cols) ** 0.5
                    X = X * scale

                    X = X.reshape(orig_shape).to(p.dtype)

                    # Weight decay + update
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(X, alpha=-lr)

                # Synchronize this round-robin slice across ranks via all-gather
                if world_size > 1:
                    dist.all_gather(
                        params_padded[base_i:base_i + world_size],
                        params_padded[base_i + rank],
                    )


# ---------------------------------------------------------------------------
# Cold-only ablation: same as WarmMuon but always uses krylov_polar
# (no warm-start, no X_prev). Used to isolate krylov vs warm-start contribution.
# ---------------------------------------------------------------------------

class WarmMuonColdOnly(WarmMuon):
    """
    Ablation: identical to WarmMuon but orthogonalization is always Krylov.
    Isolates the contribution of the warm-start from the Krylov fallback.
    """

    def _orthogonalize(self, M: Tensor, state: dict, group: dict) -> Tensor:
        krylov_steps = group["krylov_steps"]
        bs = min(M.shape)
        X = krylov_polar(M, num_steps=krylov_steps, block_size=bs)
        self._step_drifts.append(0.0)
        state["X_prev"] = X.to(torch.bfloat16)
        state["step_count"] = state.get("step_count", 0) + 1
        return X
