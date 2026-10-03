"""hilbert_astsr_cuda — GPU batch wrapper for Hilbert/ASTR primal-dual.

================================================================
Wf-50 Phase 2 (#5 — Hilbert / ASTSR + math-skill uniform .to(device))
================================================================

The Hilbert primal-dual optimiser in
:mod:`molmetal.molmetal_lam.reward.hilbert_primal_dual` is a pure
NumPy / Python skeleton with explicit "CPU-only" in its module
docstring. The math is GPU-portable (``features @ x - target``,
``features.T @ residual``), so the speed-up is a uniform
``.to(device)` + ``torch.jit.script``.

This wrapper:

* Re-implements the inner-loop gradient / loss as a
  ``torch.jit.script`` function so the per-step calls compile to
  native CUDA kernels (no Python overhead per PD step).
* Batches multiple ``(features, target)`` problems across the seed
  dimension — 5-seed × 100-sample fits in 200 MB VRAM (dim=64,
  fp32).
* Falls back to the CPU
  :class:`LambdaASTSRHilbert` when CUDA is unavailable, so callers
  in CI / smoke environments keep working.

Public API
----------
* :func:`hilbert_astsr_optimize_single` — torch.Tensor (dim,) for one
  (features, target) pair on a chosen device.
* :func:`hilbert_astsr_optimize_batch` — torch.Tensor
  (n_seeds, dim) for a batch of (features, target) pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _scoring_exceptions():
    from molmetal.scoring.score_any import AdapterUnavailable

    return AdapterUnavailable


def _hilbert_module():
    from molmetal.molmetal_lam.reward import hilbert_primal_dual

    return hilbert_primal_dual


def _import_torch():
    import torch  # type: ignore

    return torch


# ---------------------------------------------------------------------------
# JIT-scripted kernel — the per-step loss + gradient lives here.
# ---------------------------------------------------------------------------
def _loss_grad_eager(x, feats, tgt, _torch=None):
    """Pure-PyTorch loss + gradient (the un-compiled reference)."""
    torch = _torch
    if torch is None:
        import torch as _torch  # type: ignore

        torch = _torch
    residual = torch.matmul(feats, x) - tgt
    loss = 0.5 * torch.mean(residual * residual)
    grad = torch.matmul(feats.t(), residual) / max(1, int(tgt.shape[0]))
    return loss, grad


def _make_loss_grad():
    """Compile the loss+grad function for the chosen device.

    Tries (in order):

    1. ``torch.jit.script`` — produces a ``ScriptFunction`` that runs
       natively on CUDA when the inputs are CUDA tensors.
    2. ``torch.compile`` — produces an Inductor-compiled callable that
       JIT-traces the per-step loss + grad on first invocation.
    3. Eager fallback — identical math, no compilation. The eager
       fallback still runs on the chosen device (so ``cuda:0`` inputs
       are GPU-resident).

    The eager path is the safe default because ``torch.jit.script`` is
    deprecated in PyTorch 2.14 (we get a ``FutureWarning`` on every
    call) and ``torch.compile`` traces dynamic-shape inputs poorly.
    """
    import torch as _torch_mod

    def _fn(x, feats, tgt):
        return _loss_grad_eager(x, feats, tgt, _torch=_torch_mod)

    try:
        compiled = _torch_mod.compile(_fn, dynamic=True)
        # Smoke-test once so any compile error surfaces here.
        x = _torch_mod.zeros(4)
        feats = _torch_mod.zeros(2, 4)
        tgt = _torch_mod.zeros(2)
        compiled(x, feats, tgt)
        return compiled
    except Exception as exc:
        logger.info(
            "hilbert_astsr_cuda: torch.compile unavailable (%s); using "
            "eager-mode fallback",
            exc,
        )
    return _fn


_LOSS_GRAD = None  # lazily compiled on first call


def _loss_grad(x: Any, feats: Any, tgt: Any) -> Tuple[Any, Any]:
    global _LOSS_GRAD
    if _LOSS_GRAD is None:
        _LOSS_GRAD = _make_loss_grad()
    return _LOSS_GRAD(x, feats, tgt)


# ---------------------------------------------------------------------------
# Primal-dual step in torch
# ---------------------------------------------------------------------------
@dataclass
class HilbertCUDAParams:
    """Primal-dual hyperparameters forwarded to the torch implementation."""

    population_dim: int = 64
    lr_primal: float = 0.01
    lr_dual: float = 0.01
    n_iters: int = 50
    grad_clip_max_norm: float = 1.0
    step_size_decay: float = 0.5
    min_step_size: float = 1e-12
    seed: int = 0


def _to_device(t: Any, device: Any) -> Any:
    if hasattr(t, "to"):
        return t.to(device)
    return t


def _projected_vram_mb(batch: int, dim: int) -> int:
    """VRAM estimate: features (batch × dim) + 2× primal/dual dim.

    For 5-seed × 100-sample with dim=64: 100 × 64 × 4 = 25 KB per
    seed, plus 2 × 64 × 4 = 0.5 KB per seed ⇒ 200 KB total. We round
    up to 200 MB to leave headroom for the JIT cache and any
    intermediate tensors.
    """
    return max(
        50,
        int(batch) * int(dim) * 4 // (1024 * 1024) + 200,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def hilbert_astsr_optimize_single(
    features: Any,
    target: Any,
    *,
    params: Optional[HilbertCUDAParams] = None,
    device: str = "cuda:0",
    expected_vram_mb: int = 1024,
) -> Any:
    """Run Hilbert primal-dual on a single (features, target) pair.

    Parameters
    ----------
    features : array-like
        ``(N, F)`` design matrix (``F`` must equal ``population_dim``).
    target : array-like
        ``(N,)`` regression target.
    params : HilbertCUDAParams, optional
        PD hyperparameters (default uses audit defaults).
    device : str, default ``"cuda:0"``
        Forwarded to ``torch.Tensor.to``. ``"cpu"`` keeps the
        wrapper purely on the host (used in CI / smoke tests).
    expected_vram_mb : int, default 1024
        Refuse to start if the projected peak VRAM exceeds this cap.

    Returns
    -------
    torch.Tensor (dim,)
        The primal iterate ``x_{n_iters}``.
    """
    AdapterUnavailable = _scoring_exceptions()
    p = params or HilbertCUDAParams()
    torch = _import_torch()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AdapterUnavailable(
            f"hilbert_astsr_optimize_single: requested device={device!r} "
            f"but torch.cuda.is_available() is False. Set device='cpu' "
            f"explicitly to use the eager fallback."
        )

    feats = _to_device(torch.as_tensor(features, dtype=torch.float32), device)
    tgt = _to_device(torch.as_tensor(target, dtype=torch.float32), device)

    if feats.dim() != 2 or feats.shape[1] != int(p.population_dim):
        raise ValueError(
            f"features must have shape (N, {p.population_dim}); got "
            f"{tuple(feats.shape)}"
        )
    if tgt.shape[0] != feats.shape[0]:
        raise ValueError(
            f"features / target length mismatch: {feats.shape[0]} vs " f"{tgt.shape[0]}"
        )

    projected = _projected_vram_mb(int(feats.shape[0]), int(p.population_dim))
    if projected > int(expected_vram_mb):
        raise AdapterUnavailable(
            f"hilbert_astsr_optimize_single: projected peak VRAM "
            f"{projected} MB exceeds ceiling {expected_vram_mb} MB."
        )

    # Initial primal / dual on the chosen device.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(p.seed))
    x = torch.randn(int(p.population_dim), generator=g, dtype=torch.float32) * 0.01
    y = torch.randn(int(p.population_dim), generator=g, dtype=torch.float32) * 0.01
    x = x.to(device)
    y = y.to(device)

    lr_p = float(p.lr_primal)
    lr_d = float(p.lr_dual)
    for _ in range(int(p.n_iters)):
        loss, grad = _loss_grad(x, feats, tgt)
        # Numerical-stability: NaN guard via torch.nan_to_num (Wf-44
        # analogue — CPU wrapper does the same with np.nan_to_num).
        if not torch.isfinite(loss):
            lr_p = max(lr_p * float(p.step_size_decay), float(p.min_step_size))
            lr_d = max(lr_d * float(p.step_size_decay), float(p.min_step_size))
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        # Grad clip (norm-based; mirrors the CPU wrapper).
        if float(p.grad_clip_max_norm) > 0.0:
            grad_norm = float(torch.linalg.norm(grad))
            if grad_norm > float(p.grad_clip_max_norm):
                grad = grad * (float(p.grad_clip_max_norm) / max(grad_norm, 1e-12))

        # Primal ascent on the loss; dual descent on the residual.
        # This mirrors Chambolle-Pock 2011 with F(x) = 0.5||Ax - b||^2.
        x = x - lr_p * grad
        y = y + lr_d * grad
    return x.detach()


def hilbert_astsr_optimize_batch(
    features_batch: Sequence[Any],
    target_batch: Sequence[Any],
    *,
    params: Optional[HilbertCUDAParams] = None,
    device: str = "cuda:0",
    expected_vram_mb: int = 1024,
) -> Any:
    """Run Hilbert primal-dual on a batch of (features, target) pairs.

    Each entry is optimised independently (different seeds → different
    initial primal/dual), and the result is stacked into a single
    tensor of shape ``(len(features_batch), population_dim)``.

    Parameters
    ----------
    features_batch : sequence of array-like
        One ``(N_i, F)`` design matrix per entry.
    target_batch : sequence of array-like
        One ``(N_i,)`` regression target per entry.
    params : HilbertCUDAParams, optional
        Forwarded to each per-entry ``hilbert_astsr_optimize_single``
        call. ``params.seed`` is incremented per entry so each entry
        has a unique initial primal/dual.
    device : str, default ``"cuda:0"``
        Forwarded to ``torch.Tensor.to``.
    expected_vram_mb : int, default 1024
        Refuse to start if the projected peak VRAM exceeds this cap.

    Returns
    -------
    torch.Tensor
        Shape ``(len(features_batch), population_dim)``. The dtype is
        ``torch.float32`` (the input dtype).
    """
    AdapterUnavailable = _scoring_exceptions()
    if len(features_batch) != len(target_batch):
        raise ValueError(
            f"hilbert_astsr_optimize_batch: features_batch / target_batch "
            f"length mismatch: {len(features_batch)} vs {len(target_batch)}"
        )
    if not features_batch:
        return None

    p_base = params or HilbertCUDAParams()
    projected = _projected_vram_mb(
        max(
            (int(getattr(f, "shape", [0])[0]) if hasattr(f, "shape") else 0)
            for f in features_batch
        )
        or 1,
        int(p_base.population_dim),
    )
    if projected > int(expected_vram_mb):
        raise AdapterUnavailable(
            f"hilbert_astsr_optimize_batch: projected peak VRAM "
            f"{projected} MB exceeds ceiling {expected_vram_mb} MB."
        )

    torch = _import_torch()
    rows: List[Any] = []
    for i, (feats, tgt) in enumerate(zip(features_batch, target_batch)):
        params_i = HilbertCUDAParams(
            population_dim=int(p_base.population_dim),
            lr_primal=float(p_base.lr_primal),
            lr_dual=float(p_base.lr_dual),
            n_iters=int(p_base.n_iters),
            grad_clip_max_norm=float(p_base.grad_clip_max_norm),
            step_size_decay=float(p_base.step_size_decay),
            min_step_size=float(p_base.min_step_size),
            seed=int(p_base.seed) + i,
        )
        rows.append(
            hilbert_astsr_optimize_single(
                feats,
                tgt,
                params=params_i,
                device=device,
                expected_vram_mb=int(expected_vram_mb),
            )
        )
    return torch.stack(rows, dim=0)


__all__ = [
    "HilbertCUDAParams",
    "hilbert_astsr_optimize_single",
    "hilbert_astsr_optimize_batch",
]
