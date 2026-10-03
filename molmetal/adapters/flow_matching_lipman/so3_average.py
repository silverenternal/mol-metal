"""Wf-380-A1 §2.4 — SO(3)-Averaged Flow Matching SCAFFOLD.

Reference: Cao, N., Schebek, M., Reynaud, M., Bekkers, E., Vilaça, R.,
Bredell, G., Schönlieb, C.-B., Kobyzev, I. (2025).  *SO(3)-Averaged
Flow Matching for 3D Molecular Generation.*  ICML 2025.
arXiv:2507.09785.

The SO(3)-Averaged FM objective integrates out the rotation alignment
in the loss itself instead of relying on data-augmentation:

    L_SO3 = (1/K) · Σ_R  ‖v_θ(R · x_t) − R · target‖²

where R is sampled uniformly from SO(3) (the special-orthogonal group).
For a velocity field that is equivariant by construction this loss is
identical to the single-sample FM loss; for a non-equivariant field it
provides a soft regulariser.

SCAFFOLD ONLY — the canonical implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.so3_averaged_fm` (376 LOC,
Wf-340-A4).  This module exposes the small, importable reference
``so3_average_loss``.

Honest framing: SCAFFOLD-only; no GPU launches; lift claim
(+5-10% on m15 RMSD) is PROJECTED per Wf-340-A4 + Cao 2025 ICML.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


def random_so3_matrix(
    n: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample ``n`` random rotation matrices uniformly from SO(3).

    Uses the standard QR-based construction (Stewart 1980, "The
    Efficient Generation of Random Normal Matrices"):

        A ~ N(0, 1)   (3, 3)
        Q, R = QR(A)
        Q · diag(sign(diag(R)))

    Yields a uniform distribution on O(3) (with reflection).  By the
    Stewart theorem we get a reflection with probability 1/8, which we
    flip back to ensure SO(3)-membership.  Returns (n, 3, 3).

    Parameters
    ----------
    n : int
        Number of rotation matrices to sample.
    generator : optional torch.Generator for reproducibility.

    Returns
    -------
    R : (n, 3, 3) — rotation matrices with det(R) = +1.
    """
    a = torch.randn(n, 3, 3, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    # Flip signs via diag(R) to enforce uniform SO(3).
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    # In case any sign is zero (rare), default to +1.
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    # Any reflection has det = -1 — flip one column to recover SO(3).
    needs_flip = det < 0
    q[needs_flip, :, 0] = -q[needs_flip, :, 0]
    return q.to(torch.float32)


def rotate_coords(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply rotation ``R`` to coordinate tensor ``x``.

    Parameters
    ----------
    x : (..., 3)
        Coordinates to rotate.
    R : (3, 3)
        Rotation matrix.

    Returns
    -------
    Rotated coordinates with the same leading shape as ``x``.
    """
    return x @ R.t()


def so3_average_loss(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    target: torch.Tensor,
    n_rotations: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """SO(3)-Averaged FM loss (Cao et al. 2025, Eq. 4).

    Parameters
    ----------
    pred_fn : callable
        Velocity-field ``pred_fn(x_rot) -> (B, ..., 3)`` returning the
        raw velocity prediction for the rotated state.
    x_t : (B, ..., 3)
        The flow-interpolated state.
    target : (B, ..., 3)
        The conditional velocity ``x_1 − x_0``.
    n_rotations : int
        Number of Monte-Carlo rotations K.  Default 4 per Cao 2025 §3.2.
    generator : optional torch.Generator for reproducibility.

    Returns
    -------
    Scalar tensor — the SO(3)-averaged MSE loss.
    """
    R = random_so3_matrix(n_rotations, generator=generator).to(x_t.device)
    # Broadcast the rotation across the batch (apply the same rotation
    # to every sample) — this matches the per-sample augmentation in
    # the original paper.  Caller may pre-augment per-sample if needed.
    losses = []
    for k in range(n_rotations):
        x_rot = rotate_coords(x_t, R[k])
        target_rot = rotate_coords(target, R[k])
        pred_rot = pred_fn(x_rot)
        losses.append(((pred_rot - target_rot) ** 2).mean())
    return torch.stack(losses).mean()


__all__ = ["random_so3_matrix", "rotate_coords", "so3_average_loss"]
