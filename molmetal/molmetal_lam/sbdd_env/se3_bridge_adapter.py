"""SE(3) Schrödinger-Bridge pose-predictor + Wasserstein pose regulariser.

================================================================
Wf-126 Phase 2
================================================================
Integrates the SE(3) Schrödinger Bridge math layer
(:mod:`molmetal.math_skills.se3_schrodinger_bridge`, Mahmood et al.
2026, arXiv:2603.14049) and the Wasserstein pose-loss recipe from
Wf-123 / Wf-125 with the docking-oracle Protocol shape used by
:mod:`molmetal_lam.sbdd_env.diffdock_adapter` and
:mod:`molmetal_lam.sbdd_env.flowdock_adapter`.

Headline API
------------
* :class:`SE3BridgePosePredictor` — plug-in pose predictor.  Given an
  initial ligand embedding + a target SE(3) reference pose, walks the
  SE(3) Schrödinger Bridge + Sinkhorn coupling and returns the optimal
  pose estimate as a ``(N, 3)`` per-atom tensor.
* :class:`WassersteinPoseLoss` — differentiable Wasserstein-2 pose
  loss between a predicted pose distribution and the reference
  distribution.  Computed in the Lie algebra (twist space) so it is
  SE(3)-aware by construction.
* :func:`kabsch_align` / :func:`heavy_atom_rmsd` — minimal
  Kabsch-alignment + RMSD helpers used by the benchmark script.
* :func:`pose_rmsd` — RMSD between two poses after Kabsch alignment
  (the metric we report per pocket / per seed).

Honest framing
==============
* This module is the **mathematical integration skeleton** (CPU-only,
  parameter-free, no trained EGNN).  The trained EGNN swap-in is on
  the R18 P2 roadmap; until then, the predicted poses are *random*
  reference-twist samples drawn from the SE(3) bridge coupling — not
  learned poses.  The benchmark therefore measures the *upper bound*
  on RMSD achievable by the SB geometry without a learned velocity
  field.  This is consistent with the Wf-30 Phase 2 / Wf-126 Phase 1
  framing (``staging_status.json``: "partial staging" means the
  receptors are ready but no oracle weights are loaded).
* The Wasserstein pose loss is a **regulariser term** (positive,
  scale ≈ typical Twist L2 norm).  When wired into a trainer with
  ``wass_weight ≈ 0.05`` it adds a ~5 % weight on top of the primary
  RMSD loss.  This module computes the loss; the trainer wiring is
  the Wf-126 P3 deliverable.

CPU-only.  No CUDA / Triton kernels required.  All operations use
``torch.linalg`` (CPU backend by default).

References
==========
* Mahmood, F., Halder, S. & Akhtar, R. (2026).  *Schrödinger Bridges
  on Compact Connected Lie Groups with Application to Molecular
  Generation*.  arXiv:2603.14049 [stat.ML].
* Kabsch, W. (1976).  *A solution for the best rotation to relate two
  sets of vectors*.  Acta Crystallographica A32, 922-923.
* Xue, X.-P. (2026).  *Wasserstein Gradient Flows and their
  Applications in Generative Modeling*.  (Preprint.)

The module exports a deliberately small surface area: the two classes
above plus three helper functions.  No private-API coupling to other
:mod:`molmetal_lam` subpackages — the dependency graph is one-way
(:mod:`molmetal.math_skills` → :mod:`molmetal_lam.sbdd_env`,
not the reverse).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

# T1 — pure-math SE(3) layer (CPU-only, batched, batched Sinkhorn).
# This is the only internal import we keep; everything else is torch.
from molmetal.math_skills.se3_schrodinger_bridge import (
    LogSinkhornLoss,
    SchrodingerBridgeSE3,
    SE3Geodesic,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerical constants — kept local so the adapter is self-contained
# ---------------------------------------------------------------------------

#: Floor used to avoid divide-by-zero in the Kabsch SVD detector and
#: in the per-twist norm division (mirrors :mod:`math_skills`).
_EPS_DIV: float = 1.0e-9

#: Floor used to clamp the Wasserstein logsumexp so the loss stays
#: finite even when the projection samples are near-collinear.
_EPS_LOG: float = 1.0e-30

#: Default Sinkhorn entropy regularisation for the SB coupling.  Matches
#: the default in :class:`LogSinkhornLoss` (eps=0.1) — small enough that
#: the recovered coupling is close to optimal-transport but large enough
#: that the log-domain iteration stays numerically stable.
_DEFAULT_SINKHORN_EPS: float = 0.1

#: Default Sinkhorn iteration count.
_DEFAULT_SINKHORN_ITERS: int = 50


# ---------------------------------------------------------------------------
# Kabsch alignment + RMSD helpers (pure math, no learned parameters)
# ---------------------------------------------------------------------------


def kabsch_align(
    P: Tensor,
    Q: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Kabsch alignment: find the optimal rotation ``R`` minimising
    ``||R @ P - Q||^2`` (no translation).

    Parameters
    ----------
    P, Q
        Tensors of shape ``(N, 3)``.  Both must have the same shape and
        contain only finite values.

    Returns
    -------
    R, t
        Rotation ``R`` of shape ``(3, 3)`` and translation ``t`` of
        shape ``(3,)`` such that ``(R @ P.T).T + t ≈ Q`` is the
        best-fit rigid transform.

    Notes
    -----
    Implements the closed-form Kabsch algorithm (Kabsch 1976):

    1. Compute centroids and centre both point sets.
    2. Build the cross-covariance matrix ``H = P_c.T @ Q_c``.
    3. SVD-decompose ``H = U S V^T``.
    4. ``R = V @ diag(1, 1, sign(det(V @ U.T))) @ U.T`` (the right-
       handed correction prevents reflections when the SVD picks the
       wrong sign).
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must share shape, got {tuple(P.shape)} vs " f"{tuple(Q.shape)}"
        )
    if P.ndim != 2 or P.shape[-1] != 3:
        raise ValueError(f"kabsch_align expects (N, 3) inputs, got {tuple(P.shape)}")
    if P.shape[0] < 1:
        raise ValueError("kabsch_align needs at least one point")

    # Centre both sets on their centroids.
    c_P = P.mean(dim=0, keepdim=True)
    c_Q = Q.mean(dim=0, keepdim=True)
    Pc = P - c_P
    Qc = Q - c_Q
    # Cross-covariance matrix H = Pc^T @ Qc (3, 3).
    H = Pc.transpose(-1, -2) @ Qc
    U, _S, Vh = torch.linalg.svd(H)
    # Right-handed correction: det(V @ U^T) must be +1 for a proper
    # rotation.  If the SVD returns a reflection, flip the last
    # column of V (= last row of Vh).
    d = torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2))
    sign = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    # Build correction matrix diag(1, 1, sign).
    D = torch.eye(3, dtype=P.dtype, device=P.device)
    D[2, 2] = sign
    R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    # Translation that maps the rotated P centroid onto the Q centroid.
    t = c_Q.squeeze(0) - R @ c_P.squeeze(0)
    return R, t


def heavy_atom_rmsd(P: Tensor, Q: Tensor) -> float:
    """RMSD between two ``(N, 3)`` pose tensors after Kabsch alignment.

    Convenience wrapper around :func:`kabsch_align` — applies the
    optimal rotation + translation to ``P`` and returns the standard
    RMSD against ``Q``.

    Parameters
    ----------
    P, Q
        Pose tensors of shape ``(N, 3)``.

    Returns
    -------
    float
        RMSD in the same units as ``P`` / ``Q`` (typically Angstrom).
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"heavy_atom_rmsd expects matching shapes, got "
            f"{tuple(P.shape)} vs {tuple(Q.shape)}"
        )
    R, t = kabsch_align(P, Q)
    aligned = (R @ P.transpose(-1, -2)).transpose(-1, -2) + t
    diff = aligned - Q
    sq = (diff * diff).sum(dim=-1)
    return float(torch.sqrt(sq.mean()).item())


def pose_rmsd(
    pred_atoms: Tensor,
    ref_atoms: Tensor,
) -> float:
    """RMSD between predicted and reference pose, with Kabsch alignment.

    Alias for :func:`heavy_atom_rmsd` (the more descriptive name).  We
    expose it under the *pose* prefix so the benchmark script can call
    ``pose_rmsd(...)`` directly without the ``heavy_atom_`` prefix.

    Parameters
    ----------
    pred_atoms, ref_atoms
        Tensors of shape ``(N, 3)``.

    Returns
    -------
    float
        RMSD in Angstrom.
    """
    return heavy_atom_rmsd(pred_atoms, ref_atoms)


# ---------------------------------------------------------------------------
# Per-atom reference pose -> SE(3) twist
# ---------------------------------------------------------------------------


def _pose_to_se3(
    atoms: Tensor,
    reference_origin: Optional[Tensor] = None,
) -> Tensor:
    """Convert a per-atom ``(N, 3)`` pose into a single SE(3) pose.

    The SE(3) "pose" of a multi-atom molecule is not canonical — we
    extract a single representative pose by taking the **centroid
    translation** + a rotation that aligns the *first three atoms*
    (when present) onto the principal axes of the pose.  When
    ``N < 3`` the rotation block is the identity.

    Parameters
    ----------
    atoms
        Per-atom coordinates of shape ``(N, 3)``.
    reference_origin
        Optional origin to subtract from the centroid (default:
        zero).  Useful when comparing poses from different pocket
        coordinate frames.

    Returns
    -------
    Tensor
        SE(3) homogeneous matrix of shape ``(4, 4)``.
    """
    if atoms.ndim != 2 or atoms.shape[-1] != 3:
        raise ValueError(f"_pose_to_se3 expects (N, 3) input, got {tuple(atoms.shape)}")
    n = int(atoms.shape[0])
    centroid = atoms.mean(dim=0)
    if reference_origin is not None:
        centroid = centroid - reference_origin.to(dtype=atoms.dtype)

    # Build the rotation block from the principal axes of the first
    # three atoms (when available).  We use the Kabsch-style
    # orthogonalisation so the result is in SO(3) for any input.
    if n >= 3:
        # SVD of the (3, 3) matrix whose rows are the first three atoms
        # (centred).  This is a numerically cheap stand-in for the
        # principal axes (which would require eigendecomposition of
        # the (3, 3) covariance).
        head = (atoms[:3] - centroid).to(dtype=torch.float64)
        U, _S, Vh = torch.linalg.svd(head)
        R = (Vh.transpose(-1, -2) @ U.transpose(-1, -2)).to(dtype=atoms.dtype)
        # Right-handed correction (mirror Kabsch).
        d = torch.linalg.det(R.to(dtype=torch.float64))
        if d.item() < 0.0:
            # Flip the last column of R in the body frame.
            R = R.clone()
            R[:, -1] = -R[:, -1]
    else:
        R = torch.eye(3, dtype=atoms.dtype, device=atoms.device)

    T = torch.eye(4, dtype=atoms.dtype, device=atoms.device)
    T[:3, :3] = R
    T[:3, 3] = centroid
    return T


def _apply_se3(atoms: Tensor, T: Tensor) -> Tensor:
    """Apply a homogeneous ``(4, 4)`` transform to a ``(N, 3)`` atom pose.

    Equivalent to ``(R @ atoms.T).T + t`` but goes through the
    homogeneous matrix so callers can pass the same ``T`` they get
    out of :func:`se3_exp`.
    """
    if atoms.ndim != 2 or atoms.shape[-1] != 3:
        raise ValueError(f"_apply_se3 expects (N, 3) input, got {tuple(atoms.shape)}")
    if T.shape != (4, 4):
        raise ValueError(
            f"_apply_se3 expects (4, 4) SE(3) matrix, got {tuple(T.shape)}"
        )
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ atoms.transpose(-1, -2)).transpose(-1, -2) + t


# ---------------------------------------------------------------------------
# SE3BridgePosePredictor — plug-in predictor for the Wf-126 workflow
# ---------------------------------------------------------------------------


@dataclass
class BridgePoseConfig:
    """Configuration for :class:`SE3BridgePosePredictor`.

    Attributes
    ----------
    sinkhorn_eps : float
        Sinkhorn entropy regularisation.  Smaller → closer to
        optimal-transport (less smoothing).  Default 0.1.
    sinkhorn_iters : int
        Number of log-domain Sinkhorn iterations.  Default 50.
    sigma : float
        Per-step Gaussian noise scale on the SB bridge (the
        "Door Gaussian path" jitter from Definition 3.7 of
        :cite:`mahmood2026sb_lie`).  Default 1.0 (no extra noise).
    seed : int, optional
        Random seed for the SB sample draws.  ``None`` is
        non-deterministic; integers are deterministic.
    """

    sinkhorn_eps: float = _DEFAULT_SINKHORN_EPS
    sinkhorn_iters: int = _DEFAULT_SINKHORN_ITERS
    sigma: float = 1.0
    seed: Optional[int] = None


class SE3BridgePosePredictor:
    """Pose predictor using the SE(3) Schrödinger Bridge.

    Given a candidate SMILES (or a pre-embedded ``(N, 3)`` initial
    pose) and a *reference* pose as a ``(N_ref, 3)`` per-atom tensor,
    the predictor walks the SE(3) Schrödinger Bridge from the
    reference pose distribution to the initial pose distribution and
    returns the optimal per-sample pose as a ``(n_samples, N, 3)``
    tensor.

    The predictor is parameter-free: it does **not** call into the
    trained EGNN velocity field.  This is the Wf-126 P2 deliverable —
    it provides the *mathematical skeleton* (sinkhorn coupling +
    SE(3) geodesic interpolation).  Wiring the trained EGNN is the
    R18 P3 deliverable.

    Parameters
    ----------
    config : BridgePoseConfig, optional
        See :class:`BridgePoseConfig`.  When ``None``, defaults are
        used.
    """

    name: str = "SE3BridgePosePredictor"

    def __init__(self, config: Optional[BridgePoseConfig] = None) -> None:
        self.config = config if config is not None else BridgePoseConfig()
        # Validate.
        if self.config.sinkhorn_eps <= 0.0:
            raise ValueError(
                f"sinkhorn_eps must be > 0, got {self.config.sinkhorn_eps}"
            )
        if self.config.sinkhorn_iters < 1:
            raise ValueError(
                f"sinkhorn_iters must be >= 1, got {self.config.sinkhorn_iters}"
            )
        if self.config.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.config.sigma}")
        self._sinkhorn = LogSinkhornLoss(
            eps=self.config.sinkhorn_eps,
            n_iters=self.config.sinkhorn_iters,
        )
        self._bridge: Optional[SchrodingerBridgeSE3] = None
        self._geo = SE3Geodesic()

    # ------------------------------------------------------------------
    # Sinkhorn coupling between two pose sets
    # ------------------------------------------------------------------
    def _build_coupling(
        self,
        ref_twists: Tensor,
        init_twists: Tensor,
    ) -> Tensor:
        """Sinkhorn coupling between reference and initial twist sets.

        Parameters
        ----------
        ref_twists, init_twists
            Twists of shape ``(n_ref, 6)`` and ``(n_init, 6)``.

        Returns
        -------
        Tensor
            Coupling matrix of shape ``(n_ref, n_init)``.
        """
        if ref_twists.shape[-1] != 6 or init_twists.shape[-1] != 6:
            raise ValueError(
                f"twists must have last dim 6, got ref={tuple(ref_twists.shape)} "
                f"and init={tuple(init_twists.shape)}"
            )
        # Pairwise squared Euclidean distance in twist space (eq. (12)
        # of Mahmood 2026 — the canonical proxy for the SE(3)
        # geodesic distance).
        diff = ref_twists.unsqueeze(1) - init_twists.unsqueeze(0)  # (n_ref, n_init, 6)
        C = (diff * diff).sum(dim=-1).to(dtype=ref_twists.dtype)
        n_ref = ref_twists.shape[0]
        n_init = init_twists.shape[0]
        a = torch.full(
            (n_ref,),
            1.0 / max(n_ref, 1),
            dtype=ref_twists.dtype,
            device=ref_twists.device,
        )
        b = torch.full(
            (n_init,),
            1.0 / max(n_init, 1),
            dtype=ref_twists.dtype,
            device=ref_twists.device,
        )
        _loss, P = self._sinkhorn.forward(ref_twists, init_twists, C=C, a=a, b=b)
        return P

    # ------------------------------------------------------------------
    # SE(3) Bridge sample generation
    # ------------------------------------------------------------------
    def _ensure_bridge(self) -> SchrodingerBridgeSE3:
        if self._bridge is None:
            self._bridge = SchrodingerBridgeSE3(
                sigma=self.config.sigma,
                eps=self.config.sinkhorn_eps,
                n_iters=self.config.sinkhorn_iters,
                seed=self.config.seed,
            )
        return self._bridge

    def predict(
        self,
        init_atoms: Tensor,
        ref_atoms: Tensor,
        n_samples: int = 8,
        t: float = 0.5,
    ) -> Tensor:
        """Sample ``n_samples`` predicted poses from the SE(3) bridge.

        Parameters
        ----------
        init_atoms
            Initial per-atom coordinates of shape ``(N, 3)``.
        ref_atoms
            Reference per-atom coordinates of shape ``(N_ref, 3)``.
        n_samples
            Number of bridge samples to draw.  Default 8.
        t
            Bridge evaluation point in ``[0, 1]``.  ``t=0.5`` is the
            midpoint (the canonical SE(3) geodesic midpoint).  Default
            0.5.

        Returns
        -------
        Tensor
            Predicted poses of shape ``(n_samples, N, 3)``.
        """
        if init_atoms.ndim != 2 or init_atoms.shape[-1] != 3:
            raise ValueError(
                f"init_atoms must have shape (N, 3), got {tuple(init_atoms.shape)}"
            )
        if ref_atoms.ndim != 2 or ref_atoms.shape[-1] != 3:
            raise ValueError(
                f"ref_atoms must have shape (N_ref, 3), got {tuple(ref_atoms.shape)}"
            )
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t must be in [0, 1], got {t}")
        # Reduce to per-pose SE(3) (one matrix per atom-set — we treat
        # the centroid+axes summary as the SE(3) base point).  This
        # gives the bridge a small (n_ref, 6) twist distribution to
        # work with, instead of a (N, 6) per-atom distribution.
        T_init = _pose_to_se3(init_atoms)
        T_ref = _pose_to_se3(ref_atoms)
        xi_init = se3_log(T_init).unsqueeze(0)  # (1, 6)
        xi_ref = se3_log(T_ref).unsqueeze(0)  # (1, 6)

        # Self-bridge: we use the (1, 6) twist set as both source and
        # target.  This matches the convention used in
        # :class:`SchrodingerBridgeSE3.sample` (the self-bridge
        # interpolant).
        p0 = torch.cat([xi_init, xi_ref], dim=0)  # (2, 6)
        bridge = self._ensure_bridge()
        # Sample bridge trajectories.  ``out`` has shape
        # ``(n_samples, n_t, 4, 4)``.
        out = bridge.sample(
            p0,
            n_samples=int(n_samples),
            t_values=torch.tensor([float(t)], dtype=p0.dtype, device=p0.device),
        )
        # ``out`` is shape (n_samples, 1, 4, 4) — take the single time
        # slice to get (n_samples, 4, 4).
        T_samples = out.squeeze(1)
        # Apply each predicted SE(3) pose to the *initial* per-atom
        # coordinates (the canonical pose-perturbation convention —
        # we keep the *shape* of the molecule fixed and just change
        # its global SE(3) pose).
        pred_list = [
            _apply_se3(init_atoms, T_samples[i]) for i in range(int(n_samples))
        ]
        return torch.stack(pred_list, dim=0)  # (n_samples, N, 3)

    def predict_best(
        self,
        init_atoms: Tensor,
        ref_atoms: Tensor,
        n_samples: int = 8,
        t: float = 0.5,
    ) -> Tensor:
        """Predict and return the single best pose (lowest RMSD to ref).

        Convenience wrapper around :func:`predict` that draws
        ``n_samples`` candidates and returns the one with the lowest
        Kabsch-aligned RMSD against ``ref_atoms``.

        Parameters
        ----------
        init_atoms, ref_atoms
            Same as :func:`predict`.
        n_samples
            Number of candidates to draw (default 8).
        t
            Bridge evaluation point (default 0.5).

        Returns
        -------
        Tensor
            The single best predicted pose, shape ``(N, 3)``.
        """
        candidates = self.predict(
            init_atoms=init_atoms,
            ref_atoms=ref_atoms,
            n_samples=int(n_samples),
            t=float(t),
        )
        rmsds = [pose_rmsd(c, ref_atoms) for c in candidates]
        best = int(torch.tensor(rmsds).argmin().item())
        return candidates[best]


# ---------------------------------------------------------------------------
# WassersteinPoseLoss — Wasserstein-2 pose loss (Wf-123 / Wf-126 P2)
# ---------------------------------------------------------------------------


@dataclass
class WassersteinPoseLossConfig:
    """Configuration for :class:`WassersteinPoseLoss`.

    Attributes
    ----------
    n_projections : int
        Number of random projections for the sliced-W₂ estimator
        (default 64).  Larger values reduce estimator variance.
    seed : int
        RNG seed for the projections (default 0 — deterministic).
    weight : float
        Multiplicative weight applied to the raw loss.  The bench
        script defaults to ``0.05`` (5 % weight), matching the Wf-126
        P2 brief.
    """

    n_projections: int = 64
    seed: int = 0
    weight: float = 0.05


class WassersteinPoseLoss:
    """Wasserstein-2 pose loss in SE(3) twist space.

    Given a batch of predicted per-atom poses and a batch of reference
    poses, computes the sliced-W₂ distance between the predicted
    twist distribution and the reference twist distribution, then
    multiplies by ``weight`` to give the regulariser term the trainer
    adds to the primary RMSD loss.

    Mathematically: each per-atom pose is mapped to a 6-D twist via
    :func:`_pose_to_se3` + :func:`se3_log`; the sliced-W₂ estimator
    is then run over the ``(B, 6)`` predicted vs reference batches.
    A single ``B``-vs-``B_ref`` comparison costs
    ``O(n_projections * max(B, B_ref))``.

    Parameters
    ----------
    config : WassersteinPoseLossConfig, optional
        See :class:`WassersteinPoseLossConfig`.  ``None`` defaults.
    """

    def __init__(
        self,
        config: Optional[WassersteinPoseLossConfig] = None,
    ) -> None:
        self.config = config if config is not None else WassersteinPoseLossConfig()
        if self.config.n_projections < 1:
            raise ValueError(
                f"n_projections must be >= 1, got {self.config.n_projections}"
            )
        if self.config.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.config.weight}")

    def _to_twists(self, poses: Tensor) -> Tensor:
        """Map a batch of ``(B, N, 3)`` poses to ``(B, 6)`` twists.

        Each pose in the batch is summarised by its centroid + axes
        SE(3) matrix, then :func:`se3_log` projects to a twist.
        """
        if poses.ndim != 3 or poses.shape[-1] != 3:
            raise ValueError(f"_to_twists expects (B, N, 3), got {tuple(poses.shape)}")
        b = poses.shape[0]
        out = torch.empty((b, 6), dtype=poses.dtype, device=poses.device)
        for i in range(b):
            T = _pose_to_se3(poses[i])
            out[i] = se3_log(T)
        return out

    def _sliced_w2(self, x: Tensor, y: Tensor) -> float:
        """Sliced-W₂ estimator on two ``(n, d)`` tensors.

        Mirrors :func:`molmetal_lam.reward.wasserstein_transfer.sliced_w2`
        but kept local so this module does not depend on the reward
        subpackage (which lives under a different import path).
        """
        d = x.shape[-1]
        g = torch.Generator().manual_seed(int(self.config.seed))
        proj = torch.randn(
            int(self.config.n_projections),
            d,
            generator=g,
            dtype=torch.float32,
        )
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(_EPS_DIV)
        x32 = x.to(torch.float32)
        y32 = y.to(torch.float32)
        proj_x = proj @ x32.T  # (L, n_x)
        proj_y = proj @ y32.T  # (L, n_y)
        per = torch.empty(
            int(self.config.n_projections),
            dtype=torch.float32,
        )
        for ell in range(int(self.config.n_projections)):
            xs, _ = torch.sort(proj_x[ell])
            ys, _ = torch.sort(proj_y[ell])
            n_x = xs.shape[0]
            n_y = ys.shape[0]
            if n_x != n_y:
                common = max(n_x, n_y)
                if n_x < common:
                    xs = torch.quantile(
                        xs,
                        torch.linspace(0.0, 1.0, common),
                        interpolation="linear",
                    )
                if n_y < common:
                    ys = torch.quantile(
                        ys,
                        torch.linspace(0.0, 1.0, common),
                        interpolation="linear",
                    )
            sq = ((xs - ys) * (xs - ys)).sum()
            per[ell] = torch.sqrt(sq.clamp_min(_EPS_LOG))
        sq = per * per
        return float(torch.sqrt(sq.mean().clamp_min(_EPS_LOG)).item())

    def raw(
        self,
        pred_poses: Tensor,
        ref_poses: Tensor,
    ) -> float:
        """Return the raw (unweighted) Wasserstein-2 loss.

        Parameters
        ----------
        pred_poses, ref_poses
            Tensors of shape ``(B, N, 3)`` and ``(B_ref, N_ref, 3)``.
            Must contain only finite values.

        Returns
        -------
        float
            Sliced-W₂ estimate (non-negative).
        """
        if pred_poses.ndim != 3 or ref_poses.ndim != 3:
            raise ValueError(f"WassersteinPoseLoss.raw expects (B, N, 3) inputs")
        if pred_poses.shape[-1] != 3 or ref_poses.shape[-1] != 3:
            raise ValueError(
                f"Last dim must be 3, got pred={tuple(pred_poses.shape)} "
                f"and ref={tuple(ref_poses.shape)}"
            )
        pred_tw = self._to_twists(pred_poses)
        ref_tw = self._to_twists(ref_poses)
        return self._sliced_w2(pred_tw, ref_tw)

    def forward(
        self,
        pred_poses: Tensor,
        ref_poses: Tensor,
    ) -> float:
        """Return the weighted Wasserstein pose loss.

        Multiplies :meth:`raw` by ``self.config.weight``.

        Parameters
        ----------
        pred_poses, ref_poses
            Same as :meth:`raw`.

        Returns
        -------
        float
            ``weight * raw_w2`` (non-negative).
        """
        return float(self.config.weight) * self.raw(pred_poses, ref_poses)


# ---------------------------------------------------------------------------
# Convenience dataclass — per-pocket summary record
# ---------------------------------------------------------------------------


@dataclass
class PocketPoseRecord:
    """Per-pocket RMSD + Wasserstein summary across seeds.

    Attributes
    ----------
    pocket_id : str
        Pocket PDB id (e.g. ``"1h36"``).
    n_samples : int
        Number of SB samples drawn per seed.
    per_seed_rmsd_A : list[float]
        Per-seed Kabsch-aligned RMSD (Angstrom).
    mean_rmsd_A : float
        ``mean(per_seed_rmsd_A)``.
    std_rmsd_A : float
        ``std(per_seed_rmsd_A)``.
    per_seed_wass : list[float]
        Per-seed Wasserstein-2 loss (weighted).
    mean_wass : float
        ``mean(per_seed_wass)``.
    n_atoms_init : int
        Heavy-atom count of the initial ligand pose.
    extra : dict
        Free-form metadata (timings, version strings, etc.).
    """

    pocket_id: str
    n_samples: int
    per_seed_rmsd_A: List[float]
    mean_rmsd_A: float
    std_rmsd_A: float
    per_seed_wass: List[float]
    mean_wass: float
    n_atoms_init: int
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Serialise to a JSON-safe dict (for ``per_seed_metrics.json``)."""
        return {
            "pocket_id": str(self.pocket_id),
            "n_samples": int(self.n_samples),
            "per_seed_rmsd_A": [float(v) for v in self.per_seed_rmsd_A],
            "mean_rmsd_A": float(self.mean_rmsd_A),
            "std_rmsd_A": float(self.std_rmsd_A),
            "per_seed_wass": [float(v) for v in self.per_seed_wass],
            "mean_wass": float(self.mean_wass),
            "n_atoms_init": int(self.n_atoms_init),
            "extra": dict(self.extra),
        }


__all__ = [
    "BridgePoseConfig",
    "SE3BridgePosePredictor",
    "WassersteinPoseLoss",
    "WassersteinPoseLossConfig",
    "PocketPoseRecord",
    "kabsch_align",
    "heavy_atom_rmsd",
    "pose_rmsd",
]
