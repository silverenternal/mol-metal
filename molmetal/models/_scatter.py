"""Scatter-sum primitive used by the model layer (molmetal shim).

The canonical implementation lives at the project-root
``models/_scatter.py`` (so it can be reused across the molmetal and
non-molmetal model trees).  This shim re-exports the same public API
under the ``molmetal.models._scatter`` namespace so D-MPNN, EGNN and
the training scripts can import it without reaching outside the
``molmetal`` tree.

The :func:`scatter_sum` exposed here dispatches to the Triton
``aggregate_vectors`` kernel when running on a GPU with the kernel
available, and falls back to ``torch.scatter_add_`` otherwise.
Gradients flow through ``features`` automatically: the forward call is
wrapped in :class:`torch.autograd.Function` whose backward is a plain
``index_add`` over ``dst`` (one line; mirrors ``torch_scatter``'s
behaviour).

Public API
-----------
- :func:`scatter_sum`  - canonical scatter-sum (Triton or torch)
- :data:`scatter_backend` - ``"triton"`` or ``"torch"`` for diagnostics
"""

from __future__ import annotations

from typing import Optional

# Re-export the canonical module's symbols so callers get identical
# behaviour whether they import via the molmetal or the top-level path.
from models._scatter import (  # type: ignore  # noqa: E402
    _TRITON_AGGREGATE,
    scatter_backend,
    scatter_sum,
    _ScatterSum,  # noqa: F401  - exposed for testing
)


# Convenience wrapper that mirrors the legacy ``_scatter_sum(values,
# index, dim=0, dim_size=None)`` signature used throughout D-MPNN, EGNN
# and the training scripts.  The underlying :func:`scatter_sum` requires
# a ``src`` argument (kept on the signature for symmetry with the
# Triton kernel's ``edge_index``); here we synthesise one from the
# destination indices.  The ``src`` tensor never enters the math.
def scatter_sum_legacy(
    values: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> "torch.Tensor":
    """Legacy ``_scatter_sum`` signature backed by the Triton kernel.

    Mirrors the four pure-PyTorch implementations that used to live in
    ``molmetal/models/dmpnn.py``, ``molmetal/adapters/egnn_rocm.py`` and
    the two training scripts.  The first dimension of ``values`` is
    treated as the scatter axis (``dim`` is currently ignored because
    only ``dim=0`` is supported by every caller in the tree).

    The output is allocated (zero-initialised) when ``dim_size`` is
    ``None``; otherwise it uses the supplied size.

    Gradients flow through ``values`` automatically.
    """
    import torch

    if dim != 0:
        raise NotImplementedError("Only dim=0 scatter is supported here")
    if values.dim() != 2:
        raise ValueError(
            f"`values` must have shape (E, F); got {tuple(values.shape)}"
        )
    n = int(dim_size) if dim_size is not None else (int(index.max()) + 1)
    # ``src`` is unused in the math; we still need to pass a tensor of
    # the right shape so :func:`scatter_sum` can build the internal
    # ``edge_index`` without a special-case.
    src = torch.zeros_like(index)
    return scatter_sum(
        src=src,
        dst=index,
        features=values,
        n_atoms=n,
        mask=None,
        out=None,
        batch_size=None,
    )


__all__ = [
    "scatter_sum",
    "scatter_sum_legacy",
    "scatter_backend",
    "_TRITON_AGGREGATE",
]