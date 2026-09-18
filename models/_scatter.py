"""Scatter-sum primitive used by the model layer.

The model code (``models.encoder``, ``models.velocity_net``, ...) must
not depend on any specific GPU backend.  This module owns that choice:
at import time it tries to import the Triton-backed
:func:`triton_kernels.equivariant_ops.aggregate_vectors`; if that fails
(or is disabled via the env var ``MOLFLOW_DISABLE_TRITON``) it falls
back to a pure-PyTorch implementation based on
``torch.zeros(...).scatter_add_``.

The public API exposed to model code is a single function
:func:`scatter_sum` with the signature::

    scatter_sum(src, dst, features, n_atoms, mask=None, out=None)
        -> torch.Tensor

This signature is intentionally minimal — it matches how the model
layer already thinks about the operation ("sum per-edge features from
``src`` to ``dst`` over ``n_atoms``") — and keeps the Triton-only
``edge_index`` packing details internal to this module.

The :data:`scatter_backend` constant is exported so a developer (or a
test) can see at a glance whether the high-performance path is in use.

Backward pass
-------------
The Triton kernel itself is forward-only (``tl.atomic_add`` has no
autograd).  Without an explicit backward the output tensor would carry
``requires_grad=False`` and training would silently fail: the encoder
would still learn (because ``atom_embed`` provides a gradient path),
but the EGNN's equivariance would be lost — every message-passing
step would be a no-op from autograd's point of view.

To avoid this we wrap the forward in :class:`_ScatterSum` (a
``torch.autograd.Function``) whose backward is a plain
``torch.scatter_add_``.  The cost is one extra dispatch per call, in
return for which the entire ``models.encoder`` and
``models.velocity_net`` see a fully differentiable scatter-sum.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _try_load_triton_aggregate():
    """Return :func:`aggregate_vectors` from the Triton package, or ``None``.

    The Triton path is the preferred one (it is what the
    ``triton_kernels`` package is for); the fallback only exists so that
    the model layer can still run on environments where the Triton
    extension / kernel is unavailable (CI without GPU, AMD ROCm build
    issues, etc.).
    """
    if os.environ.get("MOLFLOW_DISABLE_TRITON", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        from triton_kernels.equivariant_ops import aggregate_vectors  # type: ignore
    except Exception:
        return None
    return aggregate_vectors


_TRITON_AGGREGATE = _try_load_triton_aggregate()

# Detect RDNA3 (gfx1100/gfx1101) at import time.  Per the triton-rocm
# integration report, ``torch.index_add_`` beats the Triton
# ``aggregate_vectors`` kernel by 3.0-3.5x on RDNA3 at every shape the
# Mol-Metal stack actually exercises, so we default the dispatch to the
# torch path on that architecture.  The check is done lazily because
# torch.cuda may be unavailable in CPU-only CI.
def _detect_rdna3() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    # gfx1100 / gfx1101 / gfx1102 all report (11, 0) under ROCm's PTX
    # translation layer.  We keep the check loose so future RDNA3 SKUs
    # are picked up automatically.
    return cap[0] == 11 and cap[1] in (0, 1, 2)


_IS_RDNA3: bool = _detect_rdna3()

# Shape threshold for "use Triton": beyond this (n_atoms * rbf_dim) the
# kernel becomes competitive with ``torch.index_add_`` on CDNA and is
# worth paying the autotune cost.  The number is the proxy for the
# (8192, 2048) region from molmetal/reports/triton_integration_r0.md.
_SCATTER_TRITON_FLOPS_THRESHOLD: int = 1_000_000

# Module-level override: lets ``bench_rocm_throughput.py --scatter-backend``
# (and the test suite) force a path without mutating the env var.
_BACKEND_OVERRIDE: Optional[str] = None


def set_scatter_backend(name: Optional[str]) -> None:
    """Force the backend to ``'torch'``, ``'triton'``, or back to ``None`` (auto).

    Thread the override through the lru_cache by clearing it.
    """
    global _BACKEND_OVERRIDE
    if name is None:
        _BACKEND_OVERRIDE = None
    elif name in ("torch", "triton"):
        _BACKEND_OVERRIDE = name
    else:
        raise ValueError(
            f"scatter backend must be 'torch', 'triton', or None; got {name!r}"
        )
    _cached_pick_backend.cache_clear()


def _pick_backend(device_cap: tuple, n_atoms: int, rbf_dim: int) -> str:
    """Pick the scatter backend for a given shape.

    Returns ``'triton'`` or ``'torch'``.  The result is memoised per
    ``(device_cap, n_atoms, rbf_dim)`` so we don't pay the cost of the
    decision on every call.
    """
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    if _TRITON_AGGREGATE is None:
        return "torch"
    size = n_atoms * rbf_dim
    is_rdna3 = device_cap[0] == 11 and device_cap[1] in (0, 1, 2)
    if is_rdna3:
        # RDNA3: torch wins everywhere except the very-large region
        # where the gap narrows.  Default to torch.
        return "triton" if size >= _SCATTER_TRITON_FLOPS_THRESHOLD else "torch"
    # CDNA / other: Triton wins except for very small shapes where the
    # kernel-launch + autotune cost dominates.
    return "torch" if size < 1024 else "triton"


@functools.lru_cache(maxsize=128)
def _cached_pick_backend(device_cap: tuple, n_atoms: int, rbf_dim: int) -> str:
    return _pick_backend(device_cap, n_atoms, rbf_dim)


def _backend_for(n_atoms: int, rbf_dim: int) -> str:
    """Memoised backend decision that reads the current GPU capability."""
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            cap_t = (int(cap[0]), int(cap[1]))
        except Exception:
            cap_t = (0, 0)
    else:
        cap_t = (0, 0)
    return _cached_pick_backend(cap_t, int(n_atoms), int(rbf_dim))


def _current_backend_string() -> str:
    """The ``scatter_backend``-shaped value exposed to callers/tests.

    Picks a representative shape (the report's (8192, 2048)) so the
    reported value matches what most calls will see on the current GPU.
    """
    return _backend_for(n_atoms=8192, rbf_dim=2048)


scatter_backend: str = _current_backend_string()


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class _ScatterSum(torch.autograd.Function):
    """Forward: Triton (or torch fallback) scatter-sum.

    Backward: ``torch.scatter_add_`` on the (already-aggregated)
    output gradient.  This mirrors how ``torch_scatter.scatter_add``
    defines its backward and keeps the autograd graph intact for the
    caller.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        src: torch.Tensor,
        dst: torch.Tensor,
        features: torch.Tensor,
        n_atoms: int,
        mask: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
        batch_size: Optional[int],
    ) -> torch.Tensor:
        ctx.save_for_backward(dst, mask)
        ctx.batch_size = batch_size
        ctx.n_atoms = n_atoms
        return _scatter_sum_impl(
            src, dst, features, n_atoms, mask, out, batch_size,
        )

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        dst, mask = ctx.saved_tensors
        bs = ctx.batch_size if (ctx.batch_size is not None and ctx.batch_size > 1) else 1
        grad_features = _scatter_backward(grad_out, dst, mask, bs, ctx.n_atoms)
        # src is unused in the math, so its gradient is None.
        return None, None, grad_features, None, None, None, None


def _scatter_backward(
    grad_out: torch.Tensor,
    dst: torch.Tensor,
    mask: Optional[torch.Tensor],
    batch_size: int,
    n_atoms: int,
) -> torch.Tensor:
    """Compute ``d features / d loss`` from ``d out / d loss``.

    The forward rule is ``out[atom] += features[edge]`` (each edge
    contributes to exactly one atom), so the backward is just
    ``grad_features[edge] = grad_out[dst[edge]]`` — a plain ``gather``
    over ``dst``.  Padding edges (``mask == 0``) get zero gradient so
    they don't accidentally move the live weights.

    For the batched path we offset ``dst`` per-graph so the indices
    land in the correct slice of ``grad_out`` (which is laid out as
    ``(batch_size, n_atoms, F)`` by the caller).
    """
    dst_long = dst.long()
    if batch_size > 1:
        device = dst.device
        offsets = (
            torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1) * n_atoms
        )
        dst_offset = (dst_long.view(batch_size, -1) + offsets).reshape(-1)
        # ``grad_out`` has shape ``(B, N, F)``; flatten its leading
        # two dims so we can index by ``dst_offset`` (which lives in
        # ``[0, B*N)``).
        grad_features = grad_out.reshape(batch_size * n_atoms, -1)[dst_offset]
    else:
        grad_features = grad_out[dst_long]
    if mask is not None:
        grad_features = grad_features * mask.unsqueeze(-1).to(grad_features.dtype)
    return grad_features


def _scatter_sum_impl(
    src: torch.Tensor,
    dst: torch.Tensor,
    features: torch.Tensor,
    n_atoms: int,
    mask: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    batch_size: Optional[int],
) -> torch.Tensor:
    """Inner forward that does the actual work — no autograd wrapping."""
    if features.dim() != 2:
        raise ValueError(
            f"`features` must have shape (E, F); got {tuple(features.shape)}"
        )

    # Backend selection — keyed on (n_atoms, rbf_dim) so the dispatch
    # table is small and the per-call overhead is one dict lookup
    # (memoised by ``_backend_for``).
    backend = _backend_for(n_atoms=n_atoms, rbf_dim=features.shape[1])
    use_triton = backend == "triton" and _TRITON_AGGREGATE is not None

    bs = batch_size if (batch_size is not None and batch_size > 1) else 1
    if bs > 1:
        device = dst.device
        offsets = (
            torch.arange(bs, device=device, dtype=dst.dtype).unsqueeze(1) * n_atoms
        )
        dst_offset = (dst.view(bs, -1) + offsets).reshape(-1)
        if out is None:
            out_buf = torch.zeros(
                (bs * n_atoms, features.shape[1]),
                dtype=features.dtype,
                device=features.device,
            )
        else:
            if out.shape != (bs * n_atoms, features.shape[1]):
                raise ValueError(
                    f"`out` shape {tuple(out.shape)} does not match "
                    f"(batch_size*n_atoms, F) = "
                    f"({bs * n_atoms}, {features.shape[1]})."
                )
            out_buf = out
        if use_triton:
            edge_index = torch.stack([src, dst_offset], dim=0)
            _TRITON_AGGREGATE(
                edge_index=edge_index,
                edge_features=features,
                n_atoms=bs * n_atoms,
                edge_mask=mask,
                out=out_buf,
            )
        else:
            if mask is not None:
                masked = features * mask.unsqueeze(-1).to(features.dtype)
            else:
                masked = features
            out_buf.scatter_add_(
                0,
                dst_offset.unsqueeze(-1).expand(-1, features.shape[1]),
                masked,
            )
        return out_buf.view(bs, n_atoms, features.shape[1])

    # Single-graph path (preserves the pre-decoupling API exactly).
    if use_triton:
        edge_index = torch.stack([src, dst], dim=0)
        return _TRITON_AGGREGATE(
            edge_index=edge_index,
            edge_features=features,
            n_atoms=n_atoms,
            edge_mask=mask,
            out=out,
        )

    # Pure-PyTorch fallback.
    if out is None:
        out = torch.zeros(
            (n_atoms, features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
    if mask is not None:
        features = features * mask.unsqueeze(-1).to(features.dtype)
    out = out.scatter_add(0, dst.unsqueeze(-1).expand(-1, features.shape[1]), features)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scatter_sum(
    src: torch.Tensor,
    dst: torch.Tensor,
    features: torch.Tensor,
    n_atoms: int,
    mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter-sum per-edge features onto their destination atom.

    Parameters
    ----------
    src : ``(E,)`` int tensor
        Source atom indices (currently unused on the math side — kept
        in the signature for symmetry with the Triton kernel's
        ``edge_index`` and so callers don't have to pack/unpack).
    dst : ``(E,)`` int tensor
        Destination atom indices.  Each feature row ``features[i]``
        is added to ``out[dst[i]]``.
    features : ``(E, F)`` tensor
        Per-edge feature vectors to be summed.
    n_atoms : int
        Number of destination atoms per graph (rows in the output).
    mask : optional ``(E,)`` bool / uint8 tensor
        Real-edge mask.  Padded edges contribute zero.  When ``None``
        every edge is treated as real.
    out : optional ``(N*batch_size, F)`` tensor
        Output buffer.  Allocated (zero-initialised) when ``None``.
    batch_size : optional int
        When set, ``src``/``dst``/``features``/``mask`` are flattened
        across this many graphs and the per-graph destination indices
        are offset by ``i * n_atoms``.  The output is then reshaped
        to ``(batch_size, n_atoms, F)``.

    Returns
    -------
    When ``batch_size`` is ``None``/1: ``(n_atoms, F)`` tensor.
    When ``batch_size`` is set: ``(batch_size, n_atoms, F)`` tensor.

    Notes
    -----
    Gradients flow through ``features`` (and through ``out`` when the
    caller supplies it).  The backward is a plain
    ``torch.scatter_add_`` — fast, correct, and standard.
    """
    return _ScatterSum.apply(
        src, dst, features, n_atoms, mask, out, batch_size,
    )