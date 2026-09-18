"""Fused cross-entropy (log-softmax + NLL) kernel for MolFlow-Triton.

This module cherry-picks the "online softmax + fused NLL" pattern from
FlagGems (``molmetal/references/flag_gems_repo/src/flag_gems/fused/
cross_entropy_loss.py``) and adapts it to the MolFlow-Triton
constraints:

- **No TMA / WGMMA / cluster launch / warp specialization.** Plain
  pointer-based ``tl.load`` / ``tl.store`` with ``num_warps`` /
  ``num_stages`` as the only tuning axes (``TODO/environment.md`` §4, §7).
- **No ``waves_per_eu``.** It is CDNA-only and ignored on gfx1101.
- **Import convention.** We import :mod:`triton` only inside this file
  (the leaf kernel module); callers do ``from triton_kernels import
  fused_cross_entropy``.
- **CPU fallback.** :func:`fused_cross_entropy` falls back to
  :func:`torch.nn.functional.cross_entropy` when the input device is
  not CUDA/HIP, so tests can run on a CPU-only machine.
- **Scope.** FlagGems ships three flavours (indices / probability /
  indices+smoothing) and a separate backward kernel for each.  We
  implement only the indices flavour (the dominant one for
  classification) and we fold the backward into the same kernel
  pattern via the standard "log-softmax + NLL" dual formula:

        forward:  loss_i = log(sum_c exp(z_ic)) - z_i[t_i]
        backward: d z_ic = softmax(z_i)_c - 1{c == t_i}

  This is the textbook online-softmax reduction (one pass per row)
  with the target-logit gathered at the end.  Numerically equivalent
  to the three-pass log_softmax + NLL; numerically stable via the
  subtract-the-running-max trick.

Public API
----------

- :func:`fused_cross_entropy` - ``F.cross_entropy`` with a Triton
  fused kernel when a GPU is available; :func:`fused_log_softmax_nll`
  is the lower-level entry point used internally.

Both functions support the standard PyTorch ``reduction`` argument
(``"mean"``, ``"sum"``, ``"none"``).
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
# ``fused_cross_entropy`` falls back to ``F.cross_entropy`` on CPU, so
# this only fires when the wrapper is bypassed.
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
            "the fused-cross-entropy kernel.  CPU-typed argument(s): "
            f"{cpu_args}."
        )


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
# One program instance per row.  Each program reads a single row of
# logits of width ``N_CLASSES``, computes the log-softmax denominator
# in a single pass via the standard "subtract running max" trick
# (numerically stable), then gathers the target logit and writes the
# per-row loss (mean reduction is done in PyTorch via a final sum +
# divide).
#
# We expose two ``constexpr`` block sizes for the column axis:
# ``BLOCK_NC`` is the in-block width of the logits row; ``BLOCK_TGT``
# is always 1 (we load a single target index per row).  The autotune
# key is ``["N_CLASSES"]`` because the column axis drives the
# reduction width and therefore the optimal ``num_warps`` /
# ``num_stages`` choice.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_CLASSES"])
@triton.jit
def _cross_entropy_kernel(
    logits_ptr,        # *fp    [n_rows, N_CLASSES]
    targets_ptr,       # *i32   [n_rows]
    out_ptr,           # *fp    [n_rows]
    n_rows,            # i32
    N_CLASSES: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    row_off = row * N_CLASSES
    col_off = tl.arange(0, BLOCK_NC)
    mask = col_off < N_CLASSES

    # Load the row of logits in FP32 for stable reductions.
    logits = tl.load(
        logits_ptr + row_off + col_off, mask=mask, other=-float("inf")
    ).to(tl.float32)

    # Pass 1: running max (numerical stability).
    row_max = tl.max(logits, axis=0)
    # Pass 2: shifted exp + sum -> log-sum-exp.
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    denom = tl.sum(exp_shifted, axis=0)
    log_z = tl.log(denom) + row_max

    # Gather the target logit.  We re-load with mask protection so
    # out-of-range class ids (or ``IGNORE_INDEX`` rows) yield ``0``,
    # which keeps the final per-row loss at ``0`` instead of NaN.
    target = tl.load(targets_ptr + row)
    in_range = (target >= 0) & (target < N_CLASSES) & (target != IGNORE_INDEX)
    target_logit = tl.load(
        logits_ptr + row_off + target,
        mask=in_range,
        other=0.0,
    ).to(tl.float32)

    # NLL loss = log-sum-exp - z_target; masked rows get 0.
    loss = log_z - target_logit
    loss = tl.where(in_range, loss, 0.0)

    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    out = 1
    while out < n:
        out *= 2
    return out


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------
# The backward of cross-entropy is ``softmax(logits) - one_hot(targets)``
# (for the indices flavour, no label smoothing).  We compute softmax
# inline (re-using the same "subtract max" pattern, no need to save
# the full forward pass) so the saved tensors are just (logits,
# targets) which is already what we have.  Output gradient is the
# outer reduction factor (1.0 for sum / 1/N for mean).
class _FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, reduction: str) -> torch.Tensor:
        # The kernel works on 2-D (n_rows, n_classes) views.  Flatten
        # leading dims if needed.
        if logits.dim() < 2:
            # Defer 1-D logit support to F.cross_entropy (rare and
            # usually indicates a bug at the call site).
            return torch.nn.functional.cross_entropy(
                logits, targets, ignore_index=ignore_index, reduction=reduction
            )

        original_shape = logits.shape
        n_classes = original_shape[-1]
        logits_2d = logits.reshape(-1, n_classes).contiguous()
        n_rows = logits_2d.shape[0]
        targets_flat = targets.reshape(-1).contiguous().to(torch.int32)
        if targets_flat.shape[0] != n_rows:
            raise ValueError(
                f"`targets` must have the same number of elements as the "
                f"leading dims of `logits` (got {targets_flat.shape[0]} vs "
                f"{n_rows})."
            )

        block_nc = _next_pow2(n_classes)
        if block_nc > 4096:
            # Wide-row fallback to keep the in-block reduction feasible.
            return torch.nn.functional.cross_entropy(
                logits, targets, ignore_index=ignore_index, reduction=reduction
            )

        # Per-row losses.
        losses = torch.empty(n_rows, dtype=logits.dtype, device=logits.device)
        # Defensive check before the autotune probe.
        _check_cuda_pointers({"logits": logits_2d, "targets": targets_flat, "out": losses})
        grid = (n_rows,)
        _cross_entropy_kernel[grid](
            logits_2d,
            targets_flat,
            losses,
            n_rows,
            N_CLASSES=n_classes,
            IGNORE_INDEX=ignore_index,
            BLOCK_NC=block_nc,
        )

        # Reshape losses back to logits.shape[:-1].
        per_sample = losses.reshape(original_shape[:-1])

        if reduction == "none":
            ctx.save_for_backward(logits, targets)
            ctx.n_classes = n_classes
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.logits_shape = original_shape
            ctx.scale = None
            return per_sample

        if reduction == "sum":
            total = per_sample.sum()
            ctx.save_for_backward(logits, targets)
            ctx.n_classes = n_classes
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.logits_shape = original_shape
            ctx.scale = None  # sum backward is constant 1
            return total

        # mean: divide by the number of un-ignored rows.
        # We count ``targets != ignore_index`` exactly the same way
        # F.cross_entropy does, falling back to PyTorch's reduction
        # if the count is zero (matches PyTorch's NaN semantics).
        valid = (targets_flat != ignore_index).sum()
        if valid.item() == 0:
            total = per_sample.sum()
            ctx.save_for_backward(logits, targets)
            ctx.n_classes = n_classes
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.logits_shape = original_shape
            ctx.scale = None
            # Return NaN to match F.cross_entropy behaviour on empty
            # reduction.
            return total / float("nan")
        total = per_sample.sum() / valid.to(logits.dtype)
        ctx.save_for_backward(logits, targets)
        ctx.n_classes = n_classes
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.logits_shape = original_shape
        ctx.scale = 1.0 / valid.to(logits.dtype)
        return total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, targets = ctx.saved_tensors
        n_classes = ctx.n_classes
        ignore_index = ctx.ignore_index
        original_shape = ctx.logits_shape

        # softmax(logits, dim=-1) - one_hot(targets).
        # We compute softmax ourselves so the autograd.Function does
        # not need to save the forward softmax result.
        logits_f = logits.detach().to(torch.float32)
        m = logits_f.max(dim=-1, keepdim=True).values
        exp_ = torch.exp(logits_f - m)
        softmax = exp_ / exp_.sum(dim=-1, keepdim=True)

        # One-hot targets, respecting ignore_index.
        targets_flat = targets.reshape(-1)
        valid = (targets_flat != ignore_index).unsqueeze(-1)
        targets_safe = torch.where(valid.squeeze(-1), targets_flat, torch.zeros_like(targets_flat))
        one_hot = torch.nn.functional.one_hot(targets_safe, num_classes=n_classes).to(softmax.dtype)
        one_hot = one_hot.reshape(*original_shape[:-1], n_classes)

        grad_logits = (softmax - one_hot) * valid.to(softmax.dtype)

        # Apply reduction scaling.
        if ctx.reduction == "mean":
            scale = ctx.scale if ctx.scale is not None else torch.tensor(1.0, device=logits.device, dtype=grad_logits.dtype)
            grad_logits = grad_logits * scale * grad_out
        elif ctx.reduction == "sum":
            grad_logits = grad_logits * grad_out
        else:
            # "none" - grad_out is per-sample; broadcast.
            grad_out_b = grad_out.unsqueeze(-1).to(grad_logits.dtype)
            grad_logits = grad_logits * grad_out_b

        return grad_logits.reshape(original_shape).to(logits.dtype), None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_log_softmax_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return the per-sample negative-log-likelihood loss.

    Equivalent to ``-log_softmax(logits)[..., targets]`` but computed
    in a single fused kernel pass.  No reduction is applied.
    """
    if not logits.is_cuda:
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=ignore_index, reduction="none"
        )
    return _FusedCrossEntropy.apply(logits, targets, ignore_index, "none")


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Drop-in replacement for :func:`F.cross_entropy` with a Triton kernel.

    Parameters
    ----------
    logits : tensor with shape ``(*, num_classes)``
    targets : tensor with shape ``(*)`` and integer dtype
    ignore_index : int, default -100
        Targets equal to this value contribute zero loss / gradient.
    reduction : {"mean", "sum", "none"}, default "mean"

    Returns
    -------
    Scalar (``mean`` or ``sum``) or per-sample tensor (``none``)
    with the same dtype as ``logits``.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none'; got {reduction!r}")
    if not logits.is_cuda:
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=ignore_index, reduction=reduction
        )
    return _FusedCrossEntropy.apply(logits, targets, ignore_index, reduction)


__all__ = ["fused_cross_entropy", "fused_log_softmax_nll"]