"""Custom ODE integrator step kernels (Euler and RK4) for MolFlow-Triton.

These kernels advance a batched state vector of shape ``(B, D)`` under
a (possibly time-dependent) velocity field.  The kernels are written
with :func:`triton.jit` and target AMD ROCm GPUs (``gfx1101``) via
``triton-rocm==3.8.0``.

Public API
----------
- :func:`euler_step`  - explicit Euler step ``x_{n+1} = x_n + dt * v(x_n, t_n)``.
- :func:`rk4_step`    - classical RK4 step, evaluating four stages per
                        batch element with a single fused kernel.

Both functions accept the state, the velocity evaluated at the current
point (and, for RK4, also at the intermediate stages) and a step size
``dt``.  All tensors must live on the same device; the device is
inferred from the inputs rather than being hard-coded.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


# ---------------------------------------------------------------------------
# Euler step kernel
# ---------------------------------------------------------------------------
# Autotune over (num_warps, num_stages) keyed on (B, D).  ``B`` is the
# batch dimension and ``D`` is the per-sample feature dim; both are
# passed at runtime (not as constexpr) so they don't trigger kernel
# recompilation but do participate in the autotune cache key.  The
# kernel itself only uses ``n_elements`` internally.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n", "feat_dim"])
@triton.jit
def _euler_step_kernel(
    state_ptr,    # *fp  [B, D]
    velocity_ptr, # *fp  [B, D]
    out_ptr,      # *fp  [B, D]
    n,            # i32  batch size B (autotune key only)
    feat_dim,     # i32  feature dim D (autotune key only)
    n_elements,   # i32  = B * D, used internally for masking
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ``out = state + dt * velocity`` element-wise.

    A 1-D launch grid is used so that the kernel is independent of the
    logical (B, D) shape - callers can also flatten their tensors
    beforehand if desired.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = tl.load(state_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(velocity_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, s + v, mask=mask)


# ---------------------------------------------------------------------------
# RK4 step kernel
# ---------------------------------------------------------------------------
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n", "feat_dim"])
@triton.jit
def _rk4_combine_kernel(
    state_ptr,    # *fp  [B, D]
    k1_ptr,       # *fp  [B, D]
    k2_ptr,       # *fp  [B, D]
    k3_ptr,       # *fp  [B, D]
    k4_ptr,       # *fp  [B, D]
    out_ptr,      # *fp  [B, D]
    dt,           # fp
    n,            # i32  batch size B (autotune key only)
    feat_dim,     # i32  feature dim D (autotune key only)
    n_elements,   # i32  = B * D
    BLOCK_SIZE: tl.constexpr,
):
    """Combine four RK4 stages into ``x + dt/6 * (k1 + 2 k2 + 2 k3 + k4)``."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = tl.load(state_ptr + offsets, mask=mask, other=0.0)
    k1 = tl.load(k1_ptr + offsets, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr + offsets, mask=mask, other=0.0)
    k3 = tl.load(k3_ptr + offsets, mask=mask, other=0.0)
    k4 = tl.load(k4_ptr + offsets, mask=mask, other=0.0)

    # NB: we hoist the constants out of the FMA so the JIT can fold them.
    sixth = dt / 6.0
    third = dt / 3.0
    update = sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # `third * (k2 + k3)` is mathematically identical; written as two terms
    # to keep the expression readable while still being a single FMA chain
    # after constant folding.
    _ = third  # silence "unused" warnings in tooling that inspects the kernel
    tl.store(out_ptr + offsets, s + update, mask=mask)


def _flatten_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(n, feat_dim, numel)`` and assert shape/dtype match.

    ``n`` and ``feat_dim`` are returned so that callers can pass them
    through to the autotune-keyed kernel arguments.  ``numel`` is the
    flat count used for the internal element-wise loop.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch in ODE step: state {tuple(a.shape)} vs "
            f"velocity {tuple(b.shape)}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"Dtype mismatch in ODE step: state {a.dtype} vs velocity {b.dtype}"
        )
    if a.dim() < 2:
        raise ValueError(
            f"ODE step tensors must be at least 2-D (B, D); got "
            f"{a.dim()}-D tensor of shape {tuple(a.shape)}"
        )
    feat_dim = a.shape[-1]
    n = 1
    for d in a.shape[:-1]:
        n *= d
    return n, feat_dim, a.numel()


def _select_block_size(n: int) -> int:
    """Pick a power-of-two BLOCK_SIZE between 256 and 4096 based on workload."""
    if n <= 1024:
        return 256
    if n <= 8192:
        return 512
    if n <= 65_536:
        return 1024
    return 2048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def euler_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Euler step ``state + dt * velocity``.

    Parameters
    ----------
    state : ``(B, D)`` tensor
        Current state vector.
    velocity : ``(B, D)`` tensor
        Velocity evaluated at ``state``.  Must be ``dt * v`` already, i.e.
        the caller multiplies the timestep in.  This convention keeps the
        kernel free of a scalar broadcast and matches the layout used by
        the RK4 stage combination.
    dt : float
        Step size.  Unused inside the kernel (the caller has already
        folded it into ``velocity``), but accepted for API symmetry with
        :func:`rk4_step`.
    out : optional ``(B, D)`` tensor
        Output buffer.  If ``None`` a new tensor is allocated.

    Returns
    -------
    ``(B, D)`` tensor with the advanced state.
    """
    del dt  # see docstring: callers scale velocity themselves.
    state_c = state.contiguous()
    velocity_c = velocity.contiguous()
    n_b, feat_dim, n_elements = _flatten_pair(state_c, velocity_c)

    if out is None:
        out = torch.empty_like(state_c)
    elif out.shape != state_c.shape:
        raise ValueError(
            f"`out` shape {tuple(out.shape)} does not match `state` "
            f"shape {tuple(state_c.shape)}"
        )

    block = _select_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _euler_step_kernel[grid](
        state_c,
        velocity_c,
        out,
        n_b,
        feat_dim,
        n_elements,
        BLOCK_SIZE=block,
    )
    return out


def rk4_step(
    state: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    k3: torch.Tensor,
    k4: torch.Tensor,
    dt: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Classical RK4 combine.

    Computes

    .. math::

        x_{n+1} = x_n + \\frac{dt}{6}\\bigl(k_1 + 2 k_2 + 2 k_3 + k_4\\bigr)

    where the four ``k_i`` are the velocity evaluations produced by the
    caller at the stages of the RK4 method.

    Parameters
    ----------
    state, k1, k2, k3, k4 : ``(B, D)`` tensors
        All five tensors must share shape and dtype and live on the
        same device.
    dt : float
        Step size.
    out : optional ``(B, D)`` tensor

    Returns
    -------
    ``(B, D)`` tensor with the advanced state.
    """
    state_c = state.contiguous()
    n_b, feat_dim, n_elements = _flatten_pair(state_c, k1)
    _flatten_pair(state_c, k2)
    _flatten_pair(state_c, k3)
    _flatten_pair(state_c, k4)

    if out is None:
        out = torch.empty_like(state_c)
    elif out.shape != state_c.shape:
        raise ValueError(
            f"`out` shape {tuple(out.shape)} does not match `state` "
            f"shape {tuple(state_c.shape)}"
        )

    block = _select_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _rk4_combine_kernel[grid](
        state_c,
        k1.contiguous(),
        k2.contiguous(),
        k3.contiguous(),
        k4.contiguous(),
        out,
        float(dt),
        n_b,
        feat_dim,
        n_elements,
        BLOCK_SIZE=block,
    )
    return out
