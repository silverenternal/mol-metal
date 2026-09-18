"""Fused softmax kernel for MolFlow-Triton.

This module cherry-picks the "fused softmax over a row" kernel from the
upstream Triton tutorial ``02-fused-softmax.py`` and adapts it to the
MolFlow-Triton constraints:

- **No TMA / WGMMA / cluster launch / warp specialization.** Plain
  pointer-based ``tl.load`` / ``tl.store`` with ``num_warps`` /
  ``num_stages`` as the only tuning axes (``TODO/environment.md`` §4, §7).
- **No ``waves_per_eu``.** It is CDNA-only and ignored on gfx1101.
- **Import convention.** We import :mod:`triton` only inside this file
  (the leaf kernel module); callers do ``from triton_kernels import
  fused_softmax``.
- **No driver introspection.** The upstream tutorial walks through
  ``driver.active.utils.get_device_properties(...)`` to pick an
  occupancy-aware grid.  We keep that logic out of the leaf module;
  callers that want a persistent-grid launch should use the lower-level
  :func:`_softmax_kernel` directly.
- **CPU fallback.** :func:`softmax_last_dim` falls back to
  :func:`torch.softmax` when the input device is not CUDA/HIP, so
  tests can run on a CPU-only machine.

Public API
----------

- :func:`softmax_last_dim` - ``softmax(x, dim=-1)`` with a Triton
  fused kernel when a GPU is available and the reduction fits in a
  single block, with an autograd :class:`torch.autograd.Function`
  wrapping the kernel so the operation is fully differentiable.

The kernel implements the canonical 3-pass softmax within a single
Triton program (one row per program instance):

    1. Read the row, subtract its max (numerical stability).
    2. Exponentiate and sum to obtain the denominator.
    3. Divide and write the result.

This is the textbook "row fits in SRAM" win; for tall-thin matrices
where each row is small but there are many rows, the kernel is bound
by the same MN traffic as the naive version but avoids four
intermediate tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


# ---------------------------------------------------------------------------
# Host-side guard against CPU tensors reaching a Triton kernel
# ---------------------------------------------------------------------------
# Mirrors :func:`triton_kernels.equivariant_ops._check_cuda_pointers`.
# ``softmax_last_dim`` falls back to ``torch.softmax`` on CPU, so this
# only fires when the caller bypassed the wrapper and invoked
# ``_softmax_kernel`` directly (e.g. a test).
def _check_cuda_pointers(arg_names: dict[str, object]) -> None:
    cpu_args = [
        name
        for name, value in arg_names.items()
        if isinstance(value, torch.Tensor) and not value.is_cuda
    ]
    if cpu_args:
        raise RuntimeError(
            "Triton kernel received CPU tensor(s); the caller must call "
            "`.to(device)` (typically `cuda:0` / `hip:0`) before invoking "
            "the fused-softmax kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
# The launch grid is one program instance per row.  Each program holds an
# entire row in registers and does all three softmax passes within that
# single block.  The autotune key is ``["N"]`` because the row width
# alone determines the in-block reduction width and therefore the optimal
# ``BLOCK_SIZE``-related choices (num_warps, num_stages).  Row count only
# affects parallelism, not the per-program work.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N"])
@triton.jit
def _softmax_kernel(
    out_ptr,           # *fp   [M, N]
    in_ptr,            # *fp   [M, N]
    in_row_stride,     # i32
    out_row_stride,    # i32
    n_rows,            # i32
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * in_row_stride
    col_off = tl.arange(0, BLOCK_SIZE)
    in_ptrs = in_ptr + row_start + col_off
    mask = col_off < n_cols

    # Load the row into registers, padding with -inf so it never
    # participates in the max / sum reductions.
    row = tl.load(in_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Pass 1+2 fused: subtract the max, exponentiate, sum.
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    # Pass 3: write the result back.
    out_start = row_idx * out_row_stride
    out_ptrs = out_ptr + out_start + col_off
    tl.store(out_ptrs, softmax_out.to(in_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    out = 1
    while out < n:
        out *= 2
    return out


def _move_axis_to_end(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return ``(x_perm, original_shape)`` where ``dim`` is the last axis.

    The returned ``x_perm`` has the same data as ``x`` but with axis
    ``dim`` moved to the end (and a ``.contiguous()`` copy so the
    strides match the new layout).
    """
    if dim < 0:
        dim = x.dim() + dim
    if not (0 <= dim < x.dim()):
        raise ValueError(
            f"`dim` out of range for {x.dim()}-D tensor: {dim}"
        )
    if dim == x.dim() - 1:
        return x.contiguous(), tuple(x.shape)
    perm = [d for d in range(x.dim()) if d != dim] + [dim]
    return x.permute(perm).contiguous(), tuple(x.shape)


def _move_axis_from_end(x: torch.Tensor, original_shape: tuple[int, ...], dim: int) -> torch.Tensor:
    """Inverse of :func:`_move_axis_to_end`.

    ``x`` here is a 2-D tensor of shape
    ``(prod(leading), original_shape[dim])``; we undo the permute from
    :func:`_move_axis_to_end` and reshape back to ``original_shape``.
    """
    if dim < 0:
        dim = len(original_shape) + dim
    # Build the shape of the permuted view (which is what we have
    # *before* flattening).
    leading_shape = (
        list(original_shape[:dim]) + list(original_shape[dim + 1 :])
    )
    permuted_shape = tuple(leading_shape) + (original_shape[dim],)
    x_perm = x.reshape(permuted_shape)
    if dim == len(original_shape) - 1:
        return x_perm.reshape(original_shape)
    # The forward perm is ``P = [non-dim axes..., dim]``.  The inverse
    # perm has ``P_inv[P[i]] = i``.  Build it explicitly.
    ndim = len(original_shape)
    forward_perm = [d for d in range(ndim) if d != dim] + [dim]
    inv_perm = [0] * ndim
    for new_axis, src_axis in enumerate(forward_perm):
        inv_perm[src_axis] = new_axis
    return x_perm.permute(inv_perm).contiguous().reshape(original_shape)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------
class _FusedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        if dim < 0:
            dim = x.dim() + dim
        # Move the softmax axis to the end so the kernel can treat the
        # input as a 2-D (n_rows, n_cols) matrix.
        x_perm, original_shape = _move_axis_to_end(x, dim)
        n_cols = x_perm.shape[-1]
        x_2d = x_perm.reshape(-1, n_cols)
        n_rows = x_2d.shape[0]
        block_size = _next_pow2(n_cols)
        if block_size > 4096:
            # Fall back to PyTorch softmax for very wide rows that do
            # not fit in our in-block reduction budget.
            return torch.softmax(x, dim=dim)
        y_2d = torch.empty_like(x_2d)
        # Defensive check before the autotune probe.
        _check_cuda_pointers({"in": x_2d, "out": y_2d})
        grid = (n_rows,)
        _softmax_kernel[grid](
            y_2d,
            x_2d,
            x_2d.stride(0),
            y_2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(y_2d)
        ctx.original_shape = original_shape
        ctx.dim = dim
        # y_2d has softmax over axis=1 (the last axis of the 2-D
        # view); this is what ``_softmax_backward_data`` expects.
        ctx.dim_2d = 1
        return _move_axis_from_end(y_2d, original_shape, dim)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (y_2d,) = ctx.saved_tensors
        original_shape = ctx.original_shape
        dim = ctx.dim
        dim_2d = ctx.dim_2d

        # PyTorch's softmax_backward computes
        # ``grad_x = y * (grad_out - sum_c grad_out_c * y_c)`` exactly.
        # We rely on the native backward rather than re-implement it
        # - the goal of the fused kernel is the forward pass, not the
        # backward.
        grad_perm, _ = _move_axis_to_end(grad_out, dim)
        n_cols = grad_perm.shape[-1]
        grad_out_2d = grad_perm.reshape(-1, n_cols)
        grad_x_2d = torch._softmax_backward_data(
            grad_out_2d,
            y_2d,
            dim_2d,
            y_2d.dtype,
        )
        # grad_x_2d has shape (n_rows, n_cols); reshape back through
        # the permuted view then back to original_shape.
        grad_x = _move_axis_from_end(grad_x_2d, original_shape, dim)
        return grad_x, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def softmax_last_dim(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in replacement for :func:`torch.softmax` with a Triton kernel.

    Parameters
    ----------
    x : tensor
        Any shape; the reduction is over ``dim``.
    dim : int, default -1
        Axis along which to compute the softmax.

    Returns
    -------
    Tensor with the same shape and dtype as ``x``.  On CPU or when
    ``dim`` does not fit a single block the function falls back to
    :func:`torch.softmax`.
    """
    if not x.is_cuda:
        return torch.softmax(x, dim=dim)
    if x.dim() == 0:
        return torch.softmax(x, dim=dim)
    return _FusedSoftmax.apply(x, dim)


__all__ = ["softmax_last_dim", "_softmax_kernel"]