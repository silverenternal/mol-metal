"""SE(3)-equivariant primitives for MolFlow-Triton.

Two operations are exposed:

- :func:`aggregate_vectors` - scatter-sum of per-edge vector features
  onto their target atom, supporting a variable number of neighbours
  per atom via an explicit mask.
- :func:`rotation_from_axis_angle` - rotation matrix built from an
  axis-angle representation, used to apply or compose SE(3)
  transformations.

Both kernels target AMD ROCm GPUs (``gfx1101``) via
``triton-rocm==3.8.0`` and avoid features that 3.8 does not support
(for example ``tl.dot`` with integer operands or FP8 dtypes).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .autotune import AUTOTUNE_CONFIGS


# ---------------------------------------------------------------------------
# Host-side guard against CPU tensors reaching a Triton kernel
# ---------------------------------------------------------------------------
# The Triton / HIP backend rejects CPU tensors with the cryptic
# ``ValueError: Pointer argument (at 8) cannot be accessed from Triton
# (cpu tensor?)`` -- raised during the autotune *probe* (the first
# invocation tries every config before any user code runs).  This makes
# it look like a Triton bug rather than a caller-side ``.to(device)``
# mistake.  We therefore validate every pointer-typed kernel argument
# on the host before any ``_kernel[grid](...)`` call.
def _check_cuda_pointers(arg_names: dict[str, torch.Tensor]) -> None:
    """Raise a clear :class:`RuntimeError` if any tensor arg lives on CPU.

    ``arg_names`` maps a human-readable argument name to the actual
    tensor that will be passed by pointer to the kernel.  Non-tensor
    entries (scalars, ``None``, etc.) are skipped via ``hasattr``
    checks so callers can safely include optional buffers.
    """
    cpu_args = [
        name
        for name, value in arg_names.items()
        if isinstance(value, torch.Tensor) and not value.is_cuda
    ]
    if cpu_args:
        raise RuntimeError(
            "Triton kernel received CPU tensor(s); the caller must call "
            "`.to(device)` (typically `cuda:0` / `hip:0`) before invoking "
            "scatter_sum / aggregate_vectors.  CPU-typed argument(s): "
            f"{cpu_args}.  Hint: build `mask`, `edge_index` and "
            "`edge_features` on the same CUDA device as the EGNN hidden "
            "state (or pass `device='cuda'` to the upstream helper)."
        )


# ---------------------------------------------------------------------------
# Aggregate (scatter-sum) over neighbours
# ---------------------------------------------------------------------------
# Autotune over (num_warps, num_stages) keyed on the problem size.
# ``FEAT_BLOCK`` is a constexpr on the kernel itself (it controls the
# inner reduction width), so we cannot include it as a tuning axis —
# the autotuner just picks the best warp / pipeline combination for
# each (n_edges, feat_dim) shape.
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_edges", "feat_dim"], reset_to_zero=["out_ptr"])
@triton.jit
def _aggregate_vectors_kernel(
    edge_src_ptr,      # *i32  [E]   - global source atom index per edge
    edge_dst_ptr,      # *i32  [E]   - global destination atom index per edge
    edge_mask_ptr,     # *i1   [E]   - 1 if edge is real, 0 if padded
    edge_feat_ptr,     # *fp   [E, F] - per-edge vector features
    out_ptr,           # *fp   [N, F] - aggregated per-atom output
    n_edges,           # i32
    n_atoms,           # i32
    feat_dim,          # i32
    FEAT_BLOCK: tl.constexpr,
):
    """One program instance per edge; for each feature dim, conditionally
    atomically add the per-edge feature to the destination atom.
    """
    eid = tl.program_id(axis=0)
    if eid >= n_edges:
        return

    valid = tl.load(edge_mask_ptr + eid)
    if valid == 0:
        return

    dst = tl.load(edge_dst_ptr + eid)

    feat_offsets = tl.arange(0, FEAT_BLOCK)
    feat_mask = feat_offsets < feat_dim

    feat = tl.load(
        edge_feat_ptr + eid * feat_dim + feat_offsets,
        mask=feat_mask,
        other=0.0,
    )
    tl.atomic_add(
        out_ptr + dst * feat_dim + feat_offsets,
        feat,
        mask=feat_mask,
    )


# ---------------------------------------------------------------------------
# Rotation matrix from axis-angle
# ---------------------------------------------------------------------------
@triton.jit
def _rotation_from_axis_angle_kernel(
    axis_angle_ptr,    # *fp   [B, 3]
    out_ptr,           # *fp   [B, 3, 3]
    batch_size,        # i32
    BLOCK_BATCH: tl.constexpr,
):
    """Compute a 3x3 rotation matrix from an axis-angle vector.

    Rodrigues' formula is used:

    .. math::

        R = I + \\sin(\\theta) K + (1 - \\cos(\\theta)) K^2

    where ``K`` is the skew-symmetric matrix of the (normalised) axis
    and ``\\theta = \\|axis_angle\\|``.  Each program instance handles
    a single batch element, writing a flat 3x3 matrix in row-major
    order.
    """
    pid = tl.program_id(axis=0)
    bid = pid
    if bid >= batch_size:
        return

    ax = tl.load(axis_angle_ptr + bid * 3 + 0)
    ay = tl.load(axis_angle_ptr + bid * 3 + 1)
    az = tl.load(axis_angle_ptr + bid * 3 + 2)

    theta = tl.sqrt(ax * ax + ay * ay + az * az)
    # Protect against zero-norm; treat as identity.
    eps = 1e-12
    safe_theta = tl.maximum(theta, eps)
    inv_theta = 1.0 / safe_theta

    nx = ax * inv_theta
    ny = ay * inv_theta
    nz = az * inv_theta

    sin_t = tl.sin(theta)
    cos_t = tl.cos(theta)
    one_minus_cos = 1.0 - cos_t

    # Skew-symmetric K and K^2 terms, expanded symbolically.
    # Row 0
    r00 = cos_t + nx * nx * one_minus_cos
    r01 = nx * ny * one_minus_cos - nz * sin_t
    r02 = nx * nz * one_minus_cos + ny * sin_t
    # Row 1
    r10 = ny * nx * one_minus_cos + nz * sin_t
    r11 = cos_t + ny * ny * one_minus_cos
    r12 = ny * nz * one_minus_cos - nx * sin_t
    # Row 2
    r20 = nz * nx * one_minus_cos - ny * sin_t
    r21 = nz * ny * one_minus_cos + nx * sin_t
    r22 = cos_t + nz * nz * one_minus_cos

    base = out_ptr + bid * 9
    tl.store(base + 0, r00)
    tl.store(base + 1, r01)
    tl.store(base + 2, r02)
    tl.store(base + 3, r10)
    tl.store(base + 4, r11)
    tl.store(base + 5, r12)
    tl.store(base + 6, r20)
    tl.store(base + 7, r21)
    tl.store(base + 8, r22)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def aggregate_vectors(
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    n_atoms: int,
    edge_mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter-sum per-edge vector features onto their destination atom.

    Parameters
    ----------
    edge_index : ``(2, E)`` int64 tensor
        ``edge_index[0]`` are source atom indices, ``edge_index[1]`` are
        destination atom indices.
    edge_features : ``(E, F)`` tensor
        Per-edge feature vectors to be aggregated.  ``F`` is the
        feature dimension (e.g. 3 for 3-D vector fields).
    n_atoms : int
        Number of atoms (rows in the output).
    edge_mask : optional ``(E,)`` uint8/bool tensor
        Real-edge mask.  Padded entries are skipped.  If ``None`` all
        edges are treated as real.
    out : optional ``(N, F)`` tensor
        Output buffer.  Allocated (zero-initialised) when ``None``.

    Returns
    -------
    ``(N, F)`` tensor with the per-atom aggregated vectors.
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"`edge_index` must have shape (2, E); got {tuple(edge_index.shape)}"
        )
    if edge_features.dim() != 2:
        raise ValueError(
            "`edge_features` must have shape (E, F); got "
            f"{tuple(edge_features.shape)}"
        )

    edge_src = edge_index[0].contiguous().to(torch.int32)
    edge_dst = edge_index[1].contiguous().to(torch.int32)
    feat = edge_features.contiguous()
    n_edges = edge_src.shape[0]
    feat_dim = feat.shape[1]

    if edge_mask is None:
        edge_mask = torch.ones(n_edges, dtype=torch.uint8, device=edge_src.device)
    else:
        edge_mask = edge_mask.contiguous().to(torch.uint8)

    if out is None:
        out = torch.zeros(
            (n_atoms, feat_dim),
            dtype=feat.dtype,
            device=feat.device,
        )
    else:
        if out.shape != (n_atoms, feat_dim):
            raise ValueError(
                f"`out` shape {tuple(out.shape)} does not match "
                f"(n_atoms, feat_dim) = ({n_atoms}, {feat_dim})"
            )
        if out.dtype != feat.dtype:
            raise ValueError(
                f"`out` dtype {out.dtype} does not match `edge_features` "
                f"dtype {feat.dtype}"
            )

    feat_block = 1
    while feat_block < feat_dim:
        feat_block *= 2
    if feat_block > 1024:
        raise ValueError(
            f"Feature dimension {feat_dim} is too large for the single-pass "
            f"scatter kernel; please chunk the call."
        )

    # Reject CPU tensors BEFORE the autotune probe runs (otherwise the
    # AMD backend raises the cryptic "Pointer argument ... cannot be
    # accessed from Triton (cpu tensor?)" during the very first
    # invocation, hiding the real bug: the caller forgot `.to(device)`).
    _check_cuda_pointers(
        {
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_mask": edge_mask,
            "edge_feat": feat,
            "out": out,
        }
    )

    grid = (n_edges,)
    _aggregate_vectors_kernel[grid](
        edge_src,
        edge_dst,
        edge_mask,
        feat,
        out,
        n_edges,
        n_atoms,
        feat_dim,
        FEAT_BLOCK=feat_block,
    )
    return out


def rotation_from_axis_angle(
    axis_angle: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a 3x3 rotation matrix from an axis-angle vector.

    Parameters
    ----------
    axis_angle : ``(B, 3)`` tensor
        Axis-angle vectors.  The rotation angle ``theta`` is the L2
        norm of each vector; the axis is the vector divided by
        ``theta``.  Zero vectors produce the identity matrix.
    out : optional ``(B, 3, 3)`` tensor

    Returns
    -------
    ``(B, 3, 3)`` tensor of rotation matrices.
    """
    if axis_angle.dim() != 2 or axis_angle.shape[1] != 3:
        raise ValueError(
            f"`axis_angle` must have shape (B, 3); got {tuple(axis_angle.shape)}"
        )
    aa = axis_angle.contiguous()
    batch_size = aa.shape[0]

    if out is None:
        out = torch.empty(
            (batch_size, 3, 3),
            dtype=aa.dtype,
            device=aa.device,
        )
    else:
        if out.shape != (batch_size, 3, 3):
            raise ValueError(
                f"`out` shape {tuple(out.shape)} does not match "
                f"(B, 3, 3) = ({batch_size}, 3, 3)"
            )

    grid = (batch_size,)
    _check_cuda_pointers({"axis_angle": aa, "out": out})
    _rotation_from_axis_angle_kernel[grid](
        aa,
        out,
        batch_size,
        BLOCK_BATCH=1,
        num_warps=1,
    )
    return out
