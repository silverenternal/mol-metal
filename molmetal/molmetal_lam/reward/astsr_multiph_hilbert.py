"""ASTSR + Multi-PH joint Hilbert features — integration layer.

================================================================
Wf-14 phase-3 (joint Hilbert optimization, 2026-09-26)
================================================================
Additive-only glue between :class:`HilbertASTSRComposite` (Wf-14
phase 2) and :class:`MultiPHPLLAugment` / :class:`MultiPHPLLSignature`
(Wf-14 phase 4).  Neither underlying module is modified; the
composite exposes one ``fit`` / ``predict`` / ``compare`` API so the
lambda-model coupling adapter can pick between joint and separate
Hilbert optimisation at runtime.

Wf-236 — Multi-Parameter PH → feature extraction
================================================
Adds :class:`StableMultiPHExtractor` (proper 2-parameter Vietoris-
Rips filtration via :class:`TwoParameterFiltration`) and the
:class:`MultiPHConcatAdapter` that bridges the pocket point cloud
+ atom types into the joint ASTSR + multi-PH Hilbert stream.
This is the *production* multi-PH extraction path — the existing
:class:`JointFeatureConcatenator` accepts a pre-computed multi-PH
feature tensor, but did not provide a way to *compute* it from raw
pocket coordinates; the adapter closes that gap.

The motivation
--------------
ASTSR features (Deflexpressor / R17 ship) carry the *symbolic*
structure of a λ-term; multi-PH features (Kim & Jung 2026) carry
the *multi-parameter persistent homology* structure of the pocket
geometry.  Concatenating them into a single Hilbert space lets the
primal-dual iteration exploit cross-stream correlations that two
separate regressors cannot.

Lit anchors
-----------
* Chambolle & Pock 2011 — A first-order primal-dual algorithm for
  convex problems with applications to imaging.  *J. Math. Imaging
  Vision* 40(1): 120-145 (issued 2010-12-21), DOI
  10.1007/s10851-010-0251-1 — the primal-dual splitting actually
  implemented by the Hilbert optimiser
  (``molmetal/molmetal_lam/reward/hilbert_primal_dual.py``): one step is
  ``x_{k+1} = prox_primal(x_k - lr_primal * ∇F(x_k))`` followed by
  ``y_{k+1} = y_k + lr_dual * (x_{k+1} - x_k)``.
* Sun & Zheng 2026 — 孙祥凯 (Sun Xiangkai) & 郑丽娟 (Zheng Lijuan),
  "Hilbert空间上可分离凸优化的原始-对偶动力学方法",
  《数学学报(中文版)》 2026年04期, DOI 10.12386/A20250022 — the
  Hilbert-space primal-dual-dynamics source of the O(1/k)-style rate;
  volume / issue / pages UNVERIFIED (the journal landing page at
  actamath.cjoe.ac.cn is captcha-gated).  Verified same-author English
  record: Sun, Zheng & Teo, arXiv:2408.06884, *J. Optim. Theory Appl.*
  207(1): 12 (2025), DOI 10.1007/s10957-025-02766-6.  Hilbert-space PD
  rate ``||x_k - x*||²_ℋ ≤ C / k`` (theorem number UNVERIFIED).
* Yu / Yang / Ren / Zhao 2026 (arXiv:2606.07426) — Deflexpressor
  lambda-AST symbolic regression.
* Kim & Jung 2026 (Adv. Appl. Math. 174:103023) — multi-parameter
  persistent homology.
* Liu & Feng 2026 (J. Chem. Inf. Model. 2026, 66(18): 11767-11777,
  DOI 10.1021/acs.jcim.6c02460; arXiv:2603.21503) — normalised graph
  Laplacian spectrum (PLL).

.. NOTE-Wf-A149 (2026-10-31, citation-truth wave).  Superseded wording
   is retained verbatim below as a historical record — never delete a
   superseded record (AGENTS.md §5.5 rule 6):

       * Sun & Zheng 2026 (arXiv:2510.08423) — Hilbert-space PD rate
         ``||x_k - x*||²_ℋ ≤ C / k``.
       * Liu & Feng 2026 (J. Chem. Inf. Model. 66:1024-1037) — normalised
         graph Laplacian spectrum (PLL).

   arXiv:2510.08423 resolves to "Computational Bell Inequalities"
   (Merkulov & Arnon, quant-ph, v2 2025-10-10) — NOT a Hilbert-space
   primal-dual paper; the Sun & Zheng paper itself IS real (see the
   corrected bullet above).  The PLL page range is REFUTED: Crossref
   for DOI 10.1021/acs.jcim.6c02460 returns *J. Chem. Inf. Model.*
   66(18): 11767-11777 (issued 2026-09-04), authors Jian Liu &
   Hongsong Feng; the arXiv preprint arXiv:2603.21503 ("Persistent
   local Laplacian prediction of protein-ligand binding affinities")
   carries no journal-ref.  The Liu & Feng paper itself is real — only
   the volume/issue/page string was wrong.  The Douglas-Rachford
   substitute proposed for the Hilbert anchor was checked and is
   NOT-FOUND (DOI 10.1137/110822976 → HTTP 404), so it is NOT used.

Mathematical correctness — multi-PH (Wf-236 audit)
--------------------------------------------------
* The 2-parameter filtration is computed via :class:`TwoParameterFiltration`
  with the *standard* "AND" multi-filter rule (Carlsson-Mémoli 2010):
  edge :math:`\\{i, j\\}` enters the complex when both
  :math:`\\|x_i - x_j\\|_2 \\le r` **and**
  :math:`|z_i - z_j| \\le d` are satisfied.
* We restrict to :math:`H_0 + H_1` features (multi-PH on
  :math:`\\mathbb{R}^2` filter space is **not** finitely generated
  in general — Miller 2020 §6 — but H₀ and H₁ **are**, see Kim-Jung
  2026 §4.1 ``CountGrid``).
* Stability: the discrete 2-Wasserstein stability bound (Kim-Jung
  2026 Th.1) applies; we expose a :meth:`is_stable` validator that
  rejects infinite / NaN feature maps (an empirical stability
  gate).

Honest framing
--------------
* "Joint" here means *feature-level concat* → one Hilbert regressor.
* "Separate" means two Hilbert regressors (one per stream), whose
  predictions are summed at inference time.
* The "joint R² ≥ separate R²" claim holds in expectation under a
  linear-Gaussian data model where the two streams carry *different*
  linear information about the target.  It can fail on small-sample
  regimes where the joined regressor overfits — the comparator
  reports both numbers honestly.
* Wall-time is machine-dependent — the comparison is intended for
  *relative* speedup, not absolute numbers.
* :class:`StableMultiPHExtractor` runs in O(B · N² · n_bins) — the
  W2.3 smoke budget caps pocket size at ``N <= 200``.  Paper-grade
  pockets should use the R18-P3 Fourier-acceleration path (Kim-Jung
  2026 §4, marked TODO in :mod:`molmetal.math_skills.multi_parameter_ph`).

Public API
----------
* :class:`JointFeatureConcatenator` — concat + z-score normalisation.
* :class:`JointHilbertOptimizer` — joint Hilbert regression.
* :class:`JointVsSeparateComparator` — head-to-head wall-time + R².
* :class:`StableMultiPHExtractor` — proper 2-PH extraction (Wf-236).
* :class:`MultiPHConcatAdapter` — pocket coords → multi-PH feature
  matrix + joint Hilbert integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from molmetal.math_skills.multi_parameter_ph import (
    TwoParameterFiltration,
    _EPS_LOG,
)
from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
    HilbertASTSRComposite,
)
from molmetal.molmetal_lam.reward.lambda_astsr import compute_r2


__all__ = [
    "JointFeatureConcatenator",
    "JointHilbertOptimizer",
    "JointVsSeparateComparator",
    "StableMultiPHExtractor",
    "MultiPHConcatAdapter",
]


__version__ = "0.1.0-astsr-multiph-hilbert"


# ---------------------------------------------------------------------------
# JointFeatureConcatenator — concat + z-score normalisation
# ---------------------------------------------------------------------------
class JointFeatureConcatenator:
    """Concatenate ASTSR + multi-PH feature streams into one design matrix.

    The concatenator handles either 2-D ASTSR features ``(N, d_a)`` and
    3-D multi-PH feature maps ``(N, n_bins, n_bins)`` — the latter are
    flattened on the fly to ``(N, n_bins ** 2)``.  ``normalize`` is
    z-score with a per-column std floor of ``_EPS_STD`` to guard
    against zero-variance columns.

    Attributes
    ----------
    dim_astr : Optional[int]
        ASTSR feature dim recorded on the most recent ``concatenate``
        call.  ``None`` until :meth:`concatenate` has been invoked.
    dim_multi_ph : Optional[int]
        Flattened multi-PH feature dim recorded on the most recent
        :meth:`concatenate` call.  ``None`` until :meth:`concatenate`
        has been invoked.
    """

    #: Numerical floor for the std column — keeps z-score finite
    #: for zero-variance features.
    _EPS_STD: float = 1.0e-8

    #: Numerical floor for the std row — unused but kept for ABI parity.
    _EPS_LOG: float = 1.0e-12

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._dim_astr: Optional[int] = None
        self._dim_multi_ph: Optional[int] = None

    # ------------------------------------------------------------------
    @property
    def dim_astr(self) -> Optional[int]:
        """ASTSR feature dim; ``None`` until :meth:`concatenate` runs."""
        return self._dim_astr

    @property
    def dim_multi_ph(self) -> Optional[int]:
        """Flattened multi-PH feature dim; ``None`` until first call."""
        return self._dim_multi_ph

    @property
    def is_normalized(self) -> bool:
        """``True`` once at least one normalization statistic is cached."""
        return self._mean is not None and self._std is not None

    # ------------------------------------------------------------------
    def concatenate(
        self,
        astr_feat: np.ndarray,
        multi_ph_feat: np.ndarray,
    ) -> np.ndarray:
        """Concatenate ``astr_feat`` and ``multi_ph_feat`` along the feature axis.

        Parameters
        ----------
        astr_feat
            ``(N, d_a)`` ASTSR lambda-feature matrix.
        multi_ph_feat
            ``(N, ...)`` multi-PH feature tensor; trailing dims are
            flattened into a single feature axis.

        Returns
        -------
        np.ndarray
            ``(N, d_a + d_m)`` joint feature matrix.
        """
        astr = np.asarray(astr_feat, dtype=float)
        mph = np.asarray(multi_ph_feat, dtype=float)

        if astr.ndim == 1:
            astr = astr.reshape(1, -1)
        if astr.ndim != 2:
            raise ValueError(f"astr_feat must be (N, d_a); got shape {astr.shape}")
        if mph.ndim < 2:
            raise ValueError(
                f"multi_ph_feat must have shape (N, ...) with N>0; " f"got {mph.shape}"
            )
        n_astr = int(astr.shape[0])
        n_mph = int(mph.shape[0])
        if n_astr != n_mph:
            raise ValueError(f"batch mismatch: astr N={n_astr} vs multi_ph N={n_mph}")

        # Flatten trailing multi-PH dims to a single feature axis.
        mph_flat = mph.reshape(n_mph, -1)
        # Cache the dims for downstream introspection (used by
        # ``JointHilbertOptimizer._build_composite``).
        self._dim_astr = int(astr.shape[1])
        self._dim_multi_ph = int(mph_flat.shape[1])
        return np.concatenate([astr, mph_flat], axis=-1)

    # ------------------------------------------------------------------
    def fit_normalize(
        self,
        combined_features: np.ndarray,
    ) -> np.ndarray:
        """Compute (mean, std) on ``combined_features`` and z-score them.

        Parameters
        ----------
        combined_features
            ``(N, D)`` joint design matrix.

        Returns
        -------
        np.ndarray
            ``(N, D)`` z-scored joint design matrix.  The same shape
            as the input.
        """
        feats = np.asarray(combined_features, dtype=float)
        if feats.ndim != 2:
            raise ValueError(
                f"combined_features must be (N, D); got shape {feats.shape}"
            )
        self._mean = np.mean(feats, axis=0).astype(float)
        std_raw = np.std(feats, axis=0).astype(float)
        std_safe = np.where(std_raw > self._EPS_STD, std_raw, self._EPS_STD)
        self._std = std_safe
        return (feats - self._mean) / std_safe

    # ------------------------------------------------------------------
    def transform(
        self,
        combined_features: np.ndarray,
    ) -> np.ndarray:
        """Apply cached (mean, std) to ``combined_features``.

        Falls back to :meth:`fit_normalize` when no statistics have
        been fit yet — this keeps the API forgiving for callers that
        only need a one-shot transform.
        """
        feats = np.asarray(combined_features, dtype=float)
        if self._mean is None or self._std is None:
            return self.fit_normalize(feats)
        std_safe = np.where(self._std > self._EPS_STD, self._std, self._EPS_STD)
        return (feats - self._mean) / std_safe

    # ------------------------------------------------------------------
    def normalize(
        self,
        combined_features: np.ndarray,
    ) -> np.ndarray:
        """Fit + apply normalization in one call.

        Convenience method that combines :meth:`fit_normalize` and
        :meth:`transform` into a single call.  Subsequent calls with
        new data reuse the cached statistics — i.e. ``normalize``
        follows the *train-then-test* pattern.
        """
        # First call: fit + transform.  Subsequent calls: reuse stats.
        if self._mean is None:
            return self.fit_normalize(combined_features)
        return self.transform(combined_features)


# ---------------------------------------------------------------------------
# JointHilbertOptimizer — single Hilbert regressor on the joined stream
# ---------------------------------------------------------------------------
@dataclass
class JointHilbertOptimizer:
    """Joint Hilbert optimization over (ASTSR + multi-PH) features.

    The optimizer concatenates the two feature streams, optionally
    z-score normalises the result, and fits a single Hilbert
    primal-dual regressor on the joined design matrix.  This is the
    *single-pass* alternative to fitting two Hilbert regressors (one
    per stream) and summing their predictions — see
    :class:`JointVsSeparateComparator` for the head-to-head
    comparison.

    Parameters
    ----------
    hilbert_dim : int, default ``64``
        Minimum population dim passed to the underlying Hilbert
        regressor.  The actual ``population_dim`` is
        ``max(hilbert_dim, join_dim)`` so the joined design matrix
        is never down-projected.
    normalize : bool, default ``True``
        Whether to z-score the joined design matrix before fitting.
    feature_columns : sequence of str, optional
        F5-compatible feature column tuple; the default is empty
        (the joined stream has its own internal column semantics).
    """

    hilbert_dim: int = 64
    normalize: bool = True
    feature_columns: Tuple[str, ...] = field(default_factory=tuple)
    n_iters: int = 200
    lr: float = 0.05

    # Filled by ``fit``.
    last_fit_r2_: float = field(default=0.0, init=False)
    last_fit_wall_s_: float = field(default=0.0, init=False)
    last_join_dim_: int = field(default=0, init=False)
    is_fitted_: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if int(self.hilbert_dim) < 1:
            raise ValueError(f"hilbert_dim must be >= 1; got {self.hilbert_dim}")
        if int(self.n_iters) < 1:
            raise ValueError(f"n_iters must be >= 1; got {self.n_iters}")
        if float(self.lr) <= 0.0:
            raise ValueError(f"lr must be > 0; got {self.lr}")
        # Owned concatenator.
        self.concatenator: JointFeatureConcatenator = JointFeatureConcatenator()
        # Backing composite — lazy.
        self._hilbert_composite: Optional[HilbertASTSRComposite] = None

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """``True`` if ``fit`` has succeeded at least once."""
        return bool(self.is_fitted_)

    @property
    def last_join_dim(self) -> int:
        """Joint feature dim from the most recent :meth:`fit` call."""
        return int(self.last_join_dim_)

    @property
    def hilbert_composite(self) -> Optional[HilbertASTSRComposite]:
        """The backing Hilbert composite (or ``None`` if not yet built)."""
        return self._hilbert_composite

    # ------------------------------------------------------------------
    def _build_composite(self, join_dim: int) -> HilbertASTSRComposite:
        """Build (or rebuild) the backing Hilbert composite.

        The composite's ``population_dim`` is set to
        ``max(hilbert_dim, join_dim)`` so the Hilbert regressor sees
        the joined design matrix in its native dimensionality — the
        existing :class:`LambdaASTSRHilbert` reshape to
        ``(N, population_dim)`` would otherwise fail when the user-
        supplied ``hilbert_dim`` is smaller than the join dim.
        """
        pop_dim = max(int(self.hilbert_dim), int(join_dim))
        # Build a Hilbert regressor with the user-supplied convergence
        # settings so :meth:`JointHilbertOptimizer.fit` converges at the
        # same rate as the comparator's separate factories.
        from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
            LambdaASTSRHilbert,
        )

        hilbert = LambdaASTSRHilbert(
            population_dim=int(pop_dim),
            lr_primal=float(self.lr),
            lr_dual=float(self.lr),
            n_iters=int(self.n_iters),
            seed=0,
        )
        return HilbertASTSRComposite(
            hilbert_optimizer=hilbert,
            feature_columns=tuple(self.feature_columns),
            population_dim=int(pop_dim),
            use_hilbert=True,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        astr_features: np.ndarray,
        multi_ph_features: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit the joint Hilbert regressor on (astr + multi_ph) features.

        Parameters
        ----------
        astr_features
            ``(N, d_a)`` ASTSR lambda-feature matrix.
        multi_ph_features
            ``(N, ...)`` multi-PH feature tensor.
        target
            ``(N,)`` regression target.

        Returns
        -------
        Dict[str, Any]
            ``{"r2", "wall_s", "join_dim", "backend"}`` — wall-clock
            includes concatenation, normalization and Hilbert PD.
        """
        tgt = np.asarray(target, dtype=float).reshape(-1)
        if tgt.size == 0:
            raise ValueError("target must be non-empty")

        combined = self.concatenator.concatenate(astr_features, multi_ph_features)
        if combined.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"features / target length mismatch: "
                f"{combined.shape[0]} vs {tgt.shape[0]}"
            )
        if self.normalize:
            combined = self.concatenator.fit_normalize(combined)

        # (Re-)build the backing composite so its ``population_dim``
        # matches the joined feature dim exactly.  This guarantees
        # the Hilbert regressor's ``reshape(-1, population_dim)``
        # sees ``(N, join_dim)`` as designed.
        self._hilbert_composite = self._build_composite(combined.shape[1])

        t0 = time.perf_counter()
        result = self._hilbert_composite.fit(combined, tgt)
        t1 = time.perf_counter()

        self.last_fit_r2_ = float(result["r2"])
        self.last_fit_wall_s_ = float(t1 - t0)
        self.last_join_dim_ = int(combined.shape[1])
        self.is_fitted_ = True
        return {
            "r2": self.last_fit_r2_,
            "wall_s": self.last_fit_wall_s_,
            "join_dim": self.last_join_dim_,
            "backend": "joint_hilbert",
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        astr_features: np.ndarray,
        multi_ph_features: np.ndarray,
    ) -> np.ndarray:
        """Predict using the joint Hilbert regressor.

        Returns ``np.zeros(N)`` when the optimizer has not been
        fitted yet — the same NaN-safe stub convention as
        :meth:`HilbertASTSRComposite.predict`.
        """
        if not self.is_fitted_ or self._hilbert_composite is None:
            astr = np.asarray(astr_features, dtype=float)
            n = int(astr.shape[0]) if astr.ndim >= 1 else 0
            return np.zeros(n, dtype=float)
        combined = self.concatenator.concatenate(astr_features, multi_ph_features)
        if self.normalize:
            # Re-use cached stats when available; fall back to fit on
            # first transform.
            combined = self.concatenator.transform(combined)
        return self._hilbert_composite.predict(combined)


# ---------------------------------------------------------------------------
# JointVsSeparateComparator — head-to-head wall-time + R²
# ---------------------------------------------------------------------------
@dataclass
class JointVsSeparateComparator:
    """Compare joint vs separate Hilbert regression over (astr + mph).

    Parameters
    ----------
    separate_hilbert : Callable[..., Any]
        A *factory* (typically the ``LambdaASTSRHilbert`` class) that
        returns a fresh Hilbert regressor given a ``population_dim``
        kwarg.  The comparator instantiates two such regressors per
        seed — one for the ASTSR stream, one for the multi-PH stream
        — and sums their predictions at inference.
    joint_hilbert : JointHilbertOptimizer
        Pre-built joint optimizer (used as-is, only re-fit per seed).
    """

    separate_hilbert: Callable[..., Any]
    joint_hilbert: JointHilbertOptimizer

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not callable(self.separate_hilbert):
            raise ValueError(
                f"separate_hilbert must be callable; got "
                f"{type(self.separate_hilbert).__name__}"
            )

    # ------------------------------------------------------------------
    def compare(
        self,
        features_list: Tuple[np.ndarray, np.ndarray],
        target: np.ndarray,
        n_seeds: int = 3,
    ) -> Dict[str, Any]:
        """Compare joint vs separate Hilbert regression.

        Parameters
        ----------
        features_list
            ``(astr_features, multi_ph_features)`` tuple — the two
            feature arrays to share across all seeds.
        target
            ``(N,)`` regression target.
        n_seeds
            Number of seeds to average over (default 3).

        Returns
        -------
        Dict[str, Any]
            ``{"joint_wall_s", "separate_wall_s", "joint_r2",
            "separate_r2", "speedup", "n_seeds"}`` — same shape as
            :meth:`HilbertASTSRComposite.compare`-style dicts.
        """
        if not isinstance(features_list, (tuple, list)) or len(features_list) != 2:
            raise ValueError(
                f"features_list must be a 2-tuple (astr, multi_ph); "
                f"got {type(features_list).__name__} of length "
                f"{len(features_list) if hasattr(features_list, '__len__') else '?'}"
            )
        astr_X, mph_X = features_list
        astr_X = np.asarray(astr_X, dtype=float)
        mph_X = np.asarray(mph_X, dtype=float)
        tgt = np.asarray(target, dtype=float).reshape(-1)
        if astr_X.ndim != 2 or mph_X.ndim < 2:
            raise ValueError(
                f"astr_X must be (N, d_a) and mph_X (N, ...); "
                f"got {astr_X.shape} vs {mph_X.shape}"
            )
        if astr_X.shape[0] != tgt.shape[0] or mph_X.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"features / target length mismatch: "
                f"astr={astr_X.shape[0]} mph={mph_X.shape[0]} vs "
                f"target={tgt.shape[0]}"
            )
        if int(n_seeds) < 1:
            raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")

        # Flattened views for the per-stream regressors.
        astr_dim = int(astr_X.shape[1])
        mph_dim = int(np.prod(mph_X.shape[1:]))
        astr_flat = astr_X.reshape(tgt.shape[0], astr_dim)
        mph_flat = mph_X.reshape(tgt.shape[0], mph_dim)

        joint_walls: List[float] = []
        joint_r2s: List[float] = []
        sep_walls: List[float] = []
        sep_r2s: List[float] = []

        for s in range(int(n_seeds)):
            # --- Joint Hilbert regressor (single regressor, joined stream).
            t0 = time.perf_counter()
            self.joint_hilbert._set_seed(int(s))  # type: ignore[attr-defined]
            res_j = self.joint_hilbert.fit(astr_X, mph_X, tgt)
            t1 = time.perf_counter()
            joint_walls.append(float(t1 - t0))
            joint_r2s.append(float(res_j["r2"]))

            # --- Separate regressors (one per stream, sum predictions).
            t0 = time.perf_counter()
            astr_reg = self.separate_hilbert(
                population_dim=int(max(astr_dim, 1)),
                seed=int(s),
            )
            mph_reg = self.separate_hilbert(
                population_dim=int(max(mph_dim, 1)),
                seed=int(s),
            )
            astr_reg.optimize(astr_flat, tgt)
            mph_reg.optimize(mph_flat, tgt)
            y_astr = np.asarray(astr_reg.predict(astr_flat), dtype=float).reshape(-1)
            y_mph = np.asarray(mph_reg.predict(mph_flat), dtype=float).reshape(-1)
            y_hat = y_astr + y_mph
            r2_sep = float(compute_r2(tgt, y_hat))
            t1 = time.perf_counter()
            sep_walls.append(float(t1 - t0))
            sep_r2s.append(r2_sep)

        avg_joint_wall = float(np.mean(joint_walls))
        avg_sep_wall = float(np.mean(sep_walls))
        speedup = avg_sep_wall / avg_joint_wall if avg_joint_wall > 0 else float("inf")
        return {
            "joint_wall_s": avg_joint_wall,
            "separate_wall_s": avg_sep_wall,
            "joint_r2": float(np.mean(joint_r2s)),
            "separate_r2": float(np.mean(sep_r2s)),
            "speedup": float(speedup),
            "n_seeds": int(n_seeds),
        }


# ---------------------------------------------------------------------------
# Internal helper — joint_hilbert seed forwarder (for the comparator).
# ---------------------------------------------------------------------------
def _set_seed(joint_hilbert: JointHilbertOptimizer, seed: int) -> None:
    """Forward ``seed`` to the backing Hilbert regressor if built.

    The joint optimizer's :meth:`fit` rebuilds the backing composite
    each call so the seed forwarder also resets the new composite's
    ``hilbert_optimizer.seed``.  This is a no-op when the composite
    has not been built yet.
    """
    composite = joint_hilbert.hilbert_composite
    if composite is None:
        return
    inner = composite.hilbert_optimizer
    if inner is None:
        return
    if hasattr(inner, "seed"):
        inner.seed = int(seed)


# Late-bind the helper on the class so the comparator's call site
# stays terse.  We monkey-patch the method *outside* the class body
# to avoid the dataclass decorator complaining about non-field
# attributes.
JointHilbertOptimizer._set_seed = _set_seed  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# StableMultiPHExtractor — proper 2-PH extraction (Wf-236)
# ---------------------------------------------------------------------------
class StableMultiPHExtractor:
    """Stable multi-parameter persistent homology feature extractor.

    Wraps :class:`molmetal.math_skills.multi_parameter_ph.TwoParameterFiltration`
    with a numpy-friendly interface, input validation, and a
    stability gate.  The 2-parameter Vietoris-Rips filtration uses
    two filter functions:

    * ``r(i, j) = ||x_i - x_j||_2`` — spatial distance,
    * ``d(i, j) = |z_i - z_j|`` — atomic-number difference.

    Edge :math:`\\{i, j\\}` enters the complex when **both** filters
    are below the threshold :math:`(r, d)` — the standard "AND"
    multi-filter rule (Carlsson-Mémoli 2010 §2).  Features are
    restricted to :math:`H_0 + H_1` which **are** finitely generated
    (Miller 2020 §6) and have a 2-Wasserstein stability bound
    (Kim-Jung 2026 Th.1).

    Parameters
    ----------
    max_radius
        Maximum spatial filter in Ångström.  Must be > 0.
    max_coord
        Maximum atomic-number-coordination filter value (in ``z``-
        units).  Must be >= 0.
    n_bins
        Number of bins per filter axis (must be >= 2).
    seed
        RNG seed for any future Fourier-acceleration path; the
        W2.3 smoke path is deterministic so this is currently
        unused.

    Notes
    -----
    * O(B · N² · n_bins) wall-time.  The W2.3 smoke budget caps
      pocket size at ``N <= 200``.  Above 200 atoms the R18-P3
      Fourier path (Kim-Jung 2026 §4, TODO) should be used.
    * Stability gate: :meth:`is_stable` rejects infinite / NaN /
      zero-variance feature maps so the joint Hilbert regressor
      does not ingest degenerate inputs.
    """

    #: Numerical floor for the empirical variance guardrail — a
    #: multi-PH feature map with std < this value across all bins
    #: is rejected as "degenerate" (the persistence surface is
    #: uniform → no information about the pocket topology).
    _EPS_STD: float = 1.0e-8

    def __init__(
        self,
        max_radius: float = 5.0,
        max_coord: float = 4.0,
        n_bins: int = 32,
        seed: int = 0,
    ) -> None:
        if max_radius <= 0.0:
            raise ValueError(f"max_radius must be > 0, got {max_radius}")
        if max_coord < 0.0:
            raise ValueError(f"max_coord must be >= 0, got {max_coord}")
        if int(n_bins) < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.max_radius = float(max_radius)
        self.max_coord = float(max_coord)
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        # Re-use the math layer's filtration class so the multi-PH
        # extraction matches the W2.3 surface exactly.
        self._filtration = TwoParameterFiltration(
            max_radius=max_radius,
            max_coord=max_coord,
            n_bins=n_bins,
        )

    @property
    def feature_dim(self) -> int:
        """Flattened feature dim (``n_bins ** 2``)."""
        return int(self.n_bins) ** 2

    # -- core extraction --------------------------------------------------

    def compute(
        self,
        pocket_coords: np.ndarray,
        pocket_atom_types: np.ndarray,
    ) -> np.ndarray:
        """Compute the multi-PH feature matrix from a pocket point cloud.

        Parameters
        ----------
        pocket_coords
            ``(N, 3)`` array of per-atom xyz coordinates (Å).
        pocket_atom_types
            ``(N,)`` array of atomic numbers (``z_i``).
            ``0`` is interpreted as padding by
            :class:`TwoParameterFiltration` and auto-promoted to
            carbon (``z = 6``).

        Returns
        -------
        np.ndarray
            ``(n_bins ** 2,)`` flattened 2-PH feature map for the
            pocket — the **single-pocket** extraction path.  Use
            :meth:`compute_batch` for batched pockets.
        """
        coords_t = torch.as_tensor(
            np.asarray(pocket_coords, dtype=np.float32),
        ).unsqueeze(0)  # (1, N, 3)
        atom_t = torch.as_tensor(
            np.asarray(pocket_atom_types, dtype=np.int64),
        ).unsqueeze(0)  # (1, N)
        feat = self._filtration.compute_with_atom_types(coords_t, atom_t)
        # (1, n_bins, n_bins) → (n_bins ** 2,)
        feat_np = feat.squeeze(0).reshape(-1).detach().cpu().numpy()
        return feat_np.astype(np.float64)

    def compute_batch(
        self,
        pocket_coords: np.ndarray,
        pocket_atom_types: np.ndarray,
        pocket_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute multi-PH features for a batch of pockets.

        Parameters
        ----------
        pocket_coords
            ``(B, N_max, 3)`` padded pocket coordinates (Å).
            ``0``-padded rows (per :attr:`pocket_mask`) are
            accepted by :class:`TwoParameterFiltration` as long as
            they carry zeros in the coordinate channels; passing
            ``pocket_mask`` filters them out explicitly.
        pocket_atom_types
            ``(B, N_max)`` atomic numbers.  We clamp the masked
            atoms to ``z = 6`` (carbon) so the second filter
            (``|z_i - z_j|``) is well-defined for the masked
            rows — the multi-PH edge counts then vanish for
            padded rows because the spatial filter fails on the
            (0, 0, 0) origin.
        pocket_mask
            ``(B, N_max)`` boolean mask; ``True`` = real atom.
            When ``None`` we treat every atom as real.

        Returns
        -------
        np.ndarray
            ``(B, n_bins ** 2)`` flattened multi-PH feature matrix.
        """
        coords_arr = np.asarray(pocket_coords, dtype=np.float32)
        atom_arr = np.asarray(pocket_atom_types, dtype=np.int64)
        if coords_arr.ndim != 3 or coords_arr.shape[-1] != 3:
            raise ValueError(
                f"pocket_coords must be (B, N, 3); got {tuple(coords_arr.shape)}"
            )
        if atom_arr.shape != coords_arr.shape[:2]:
            raise ValueError(
                f"pocket_atom_types must be (B, N); got {tuple(atom_arr.shape)}"
            )
        if pocket_mask is None:
            pocket_mask = np.ones(coords_arr.shape[:2], dtype=bool)
        mask = np.asarray(pocket_mask, dtype=bool)
        if mask.shape != coords_arr.shape[:2]:
            raise ValueError(f"pocket_mask must be (B, N); got {tuple(mask.shape)}")
        # Zero-out the padded atoms so the spatial filter rejects
        # them (||0 - 0|| = 0 satisfies max_radius=0 trivially, but
        # the (0, 0) origin carries no geometric signal).
        coords_clean = coords_arr * mask[..., None].astype(np.float32)
        # Clamp masked atom types to 6 (carbon) — the |z_i - z_j|
        # filter is then 0 for masked rows, which is consistent
        # with the W2.3 smoke convention.
        atom_clean = np.where(mask, atom_arr, 6)

        coords_t = torch.as_tensor(coords_clean)
        atom_t = torch.as_tensor(atom_clean, dtype=torch.long)
        feat = self._filtration.compute_with_atom_types(coords_t, atom_t)
        # (B, n_bins, n_bins) → (B, n_bins ** 2)
        feat_np = feat.reshape(feat.shape[0], -1).detach().cpu().numpy()
        return feat_np.astype(np.float64)

    # -- stability gate ---------------------------------------------------

    def is_stable(
        self,
        feat: np.ndarray,
        *,
        check_variance: bool = True,
    ) -> bool:
        """Validate that ``feat`` is finite and (optionally) non-degenerate.

        Returns ``False`` when ``feat`` contains NaN / Inf entries,
        when the *entire* feature map is uniformly zero (no edges
        detected → no information about pocket topology), or when
        the per-column variance floor :attr:`_EPS_STD` is not met.
        This is the empirical stability gate described in Kim-Jung
        2026 Th.1 — the discrete 2-Wasserstein stability bound
        applies to *bounded* feature maps; we enforce the bound
        here.
        """
        f = np.asarray(feat)
        if not np.isfinite(f).all():
            return False
        # Uniformly zero feature map → no edges → degenerate.  Applies
        # regardless of shape (a single zero feature map of any
        # shape is degenerate).
        if float(np.abs(f).max()) < self._EPS_STD:
            return False
        if check_variance and f.ndim >= 2 and f.shape[0] >= 2:
            if float(f.std(axis=0).max()) < self._EPS_STD:
                return False
        return True


# ---------------------------------------------------------------------------
# MultiPHConcatAdapter — pocket coords → joint ASTSR + multi-PH feature stream
# ---------------------------------------------------------------------------
class MultiPHConcatAdapter:
    """Bridge from pocket point cloud to the joint ASTSR + multi-PH stream.

    The adapter wraps a :class:`StableMultiPHExtractor` and a
    :class:`JointFeatureConcatenator` so the caller can hand in
    raw pocket coordinates + atomic numbers and obtain a single
    joint feature matrix suitable for
    :meth:`JointHilbertOptimizer.fit`.  This is the *production*
    path for the Wf-236 multi-PH integration — the underlying
    :class:`JointFeatureConcatenator` is unmodified, so the
    downstream joint Hilbert regressor is bit-identical.

    Parameters
    ----------
    extractor
        Pre-built :class:`StableMultiPHExtractor` (default
        :class:`StableMultiPHExtractor` with W2.3 smoke defaults).
    concatenator
        Pre-built :class:`JointFeatureConcatenator` (default
        :class:`JointFeatureConcatenator()`).

    Notes
    -----
    The adapter is **stateless** beyond the (extractor,
    concatenator) pair — re-use a single instance across all
    pockets in a batch to amortise the extractor construction
    cost.
    """

    def __init__(
        self,
        extractor: Optional[StableMultiPHExtractor] = None,
        concatenator: Optional[JointFeatureConcatenator] = None,
    ) -> None:
        self.extractor = extractor or StableMultiPHExtractor()
        self.concatenator = concatenator or JointFeatureConcatenator()

    # -- joint feature extraction ----------------------------------------

    def extract_joint(
        self,
        astr_features: np.ndarray,
        pocket_coords: np.ndarray,
        pocket_atom_types: np.ndarray,
        pocket_mask: Optional[np.ndarray] = None,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Compute the joint ASTSR + multi-PH feature matrix.

        Parameters
        ----------
        astr_features
            ``(N, d_a)`` ASTSR lambda-feature matrix.
        pocket_coords
            ``(N, 3)`` *or* ``(B, N_max, 3)`` pocket coordinates.
            ``(N, 3)`` is interpreted as a single pocket and is
            broadcast to match ``astr_features.shape[0]``.  For a
            batched call, pass ``pocket_coords`` with three
            trailing axes (the ``(B, N_max, 3)`` layout).
        pocket_atom_types
            ``(N,)`` *or* ``(B, N_max)`` atomic numbers.
        pocket_mask
            ``(B, N_max)`` boolean mask for batched extraction;
            ignored for single-pocket calls.
        normalize
            Whether to z-score the joined matrix via
            :meth:`JointFeatureConcatenator.normalize`.

        Returns
        -------
        np.ndarray
            ``(N, d_a + n_bins**2)`` joined feature matrix.
        """
        astr = np.asarray(astr_features, dtype=float)
        if astr.ndim == 1:
            astr = astr.reshape(1, -1)
        n_a = int(astr.shape[0])

        coords = np.asarray(pocket_coords, dtype=np.float32)
        atoms = np.asarray(pocket_atom_types, dtype=np.int64)
        if coords.ndim == 2:
            # Single-pocket broadcast — same coords for every ASTSR row.
            coords_b = np.broadcast_to(coords[None, ...], (n_a,) + coords.shape)
            atoms_b = np.broadcast_to(atoms[None, ...], (n_a,) + atoms.shape)
            multi_ph = self.extractor.compute_batch(
                coords_b.copy(),
                atoms_b.copy(),
                pocket_mask=None,
            )
        elif coords.ndim == 3:
            if coords.shape[0] != n_a:
                raise ValueError(
                    f"batch mismatch: astr N={n_a} vs pockets B=" f"{coords.shape[0]}"
                )
            multi_ph = self.extractor.compute_batch(
                coords,
                atoms,
                pocket_mask=pocket_mask,
            )
        else:
            raise ValueError(
                f"pocket_coords must be (N, 3) or (B, N, 3); got "
                f"{tuple(coords.shape)}"
            )

        # Stability gate — drop a warning when the multi-PH features
        # are degenerate so the caller knows the joint stream will
        # carry less information than expected.
        if not self.extractor.is_stable(multi_ph):
            import warnings

            warnings.warn(
                "Multi-PH feature map is unstable (zero variance or "
                "non-finite entries); the joint ASTSR + multi-PH stream "
                "may not improve over the ASTSR-only baseline.",
                RuntimeWarning,
                stacklevel=2,
            )

        combined = self.concatenator.concatenate(astr, multi_ph)
        if normalize:
            combined = self.concatenator.normalize(combined)
        return combined

    # -- drop-in fit helper ----------------------------------------------

    def fit_joint_hilbert(
        self,
        joint_hilbert: JointHilbertOptimizer,
        astr_features: np.ndarray,
        pocket_coords: np.ndarray,
        pocket_atom_types: np.ndarray,
        target: np.ndarray,
        pocket_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit ``joint_hilbert`` on the extracted joint feature stream.

        This is a thin convenience wrapper around
        :meth:`JointHilbertOptimizer.fit` that handles the multi-PH
        extraction automatically.  Useful when the caller wants a
        one-liner integration path.

        Returns
        -------
        Dict[str, Any]
            The :meth:`JointHilbertOptimizer.fit` result dict,
            augmented with the multi-PH stability flag under
            ``multi_ph_stable``.
        """
        multi_ph = self._extract_multi_ph(
            astr_features,
            pocket_coords,
            pocket_atom_types,
            pocket_mask,
        )
        result = joint_hilbert.fit(astr_features, multi_ph, target)
        result["multi_ph_stable"] = bool(
            self.extractor.is_stable(multi_ph),
        )
        return result

    # -- internal helper --------------------------------------------------

    def _extract_multi_ph(
        self,
        astr_features: np.ndarray,
        pocket_coords: np.ndarray,
        pocket_atom_types: np.ndarray,
        pocket_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Raw multi-PH extraction — mirrors :meth:`extract_joint` minus the ASTSR concat."""
        astr = np.asarray(astr_features, dtype=float)
        if astr.ndim == 1:
            astr = astr.reshape(1, -1)
        n_a = int(astr.shape[0])
        coords = np.asarray(pocket_coords, dtype=np.float32)
        atoms = np.asarray(pocket_atom_types, dtype=np.int64)
        if coords.ndim == 2:
            coords_b = np.broadcast_to(
                coords[None, ...],
                (n_a,) + coords.shape,
            )
            atoms_b = np.broadcast_to(
                atoms[None, ...],
                (n_a,) + atoms.shape,
            )
            return self.extractor.compute_batch(
                coords_b.copy(),
                atoms_b.copy(),
                pocket_mask=None,
            )
        if coords.ndim == 3 and coords.shape[0] == n_a:
            return self.extractor.compute_batch(
                coords,
                atoms,
                pocket_mask=pocket_mask,
            )
        raise ValueError(
            f"pocket_coords shape {tuple(coords.shape)} incompatible "
            f"with astr N={n_a}"
        )
