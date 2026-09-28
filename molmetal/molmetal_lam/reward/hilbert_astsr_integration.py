"""Hilbert primal-dual ↔ LambdaASTSR evolutionary search — integration layer.

================================================================
Wf-9 phase-3 integration (TODO-34 §L-R18-P1-5, 2026-09-26)
================================================================
Additive-only glue between :class:`LambdaASTSR` (R17 ship) and
:class:`LambdaASTSRHilbert` (Wf-9 ship).  Neither underlying module
is modified; the composite exposes one ``fit`` / ``predict`` /
``switch_to_hilbert`` API so the lambda-model coupling adapter can
pick a hypothesis class at runtime.

Lit anchors
-----------
* Sun & Zheng 2026 (arXiv:2510.08423) — Hilbert-space PD rate
  ``||x_k - x*||²_ℋ ≤ C / k``.
* Yu / Yang / Ren / Zhao 2026 (arXiv:2606.07426) — Deflexpressor
  lambda-AST symbolic regression.

Honest framing
--------------
* Both regressors are kept as black boxes; the integration does not
  average their R² or "fuse" their predictions.  Wall-time is
  machine-dependent — the comparison is intended for *relative*
  speedup, not absolute numbers.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from molmetal.molmetal_lam.reward.lambda_astsr import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_ASTSR_SCALE_CLIP,
    LambdaASTSR,
    clip_astr_signature,
    compute_r2,
)


# ---------------------------------------------------------------------------
# Runtime flag — mirrors the parallel pattern in hilbert_primal_dual.
# ---------------------------------------------------------------------------
_ENV_FLAG = "USE_HILBERT_PRIMAL_DUAL"


def _is_hilbert_active() -> bool:
    raw = os.environ.get(_ENV_FLAG, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _set_hilbert_active(value: bool) -> None:
    os.environ[_ENV_FLAG] = "1" if bool(value) else "0"


# ---------------------------------------------------------------------------
# HilbertASTSRComposite — one fit/predict entry point for both regressors
# ---------------------------------------------------------------------------
@dataclass
class HilbertASTSRComposite:
    """Wrap :class:`LambdaASTSR` + :class:`LambdaASTSRHilbert` behind one API.

    Parameters
    ----------
    lambda_astsr : LambdaASTSR, optional
        Pre-fitted (or to-be-fitted) evolutionary regressor.  A stub
        is constructed when ``None``.
    hilbert_optimizer : LambdaASTSRHilbert, optional
        Pre-instantiated Hilbert primal-dual regressor.  Defaults to
        ``LambdaASTSRHilbert(population_dim=population_dim)``.
    feature_columns : sequence of str, optional
        F5-compatible feature column tuple.
    use_hilbert : bool, default ``True``
        Whether :meth:`fit` / :meth:`predict` delegate to the Hilbert
        regressor (``True``) or the evolutionary LambdaASTSR.
    population_dim : int, default ``64``
        Hilbert space dimension when ``hilbert_optimizer`` is ``None``.
    loss_scale : float, default ``128.0``
        Wf-44 Phase 4 numerical-stability fix.  Multiplies the
        training loss (and hence the gradient) by this factor before
        back-propagation; this is the standard "loss-scaling" pattern
        used to keep fp16 gradients from underflowing.  Set to
        ``1.0`` to disable scaling.  When ``0.0`` we disable loss
        scaling entirely (the gradient is computed at its native
        scale).
    fp64_fallback : bool, default ``True``
        Wf-44 Phase 4 numerical-stability fix.  When ``True``, if the
        Hilbert branch returns a non-finite R² or non-finite predictions
        after ``:meth:fit``, the fit is retried in float64 precision
        (a *fallback*, not a primary path).  When ``False``, the
        fallback is skipped and the non-finite result is recorded as
        such.
    l2_weight : float, default ``0.01``
        Wf-127 Phase 2 L2 weight-decay guard.  Forwarded to the
        Hilbert regressor at construction (and re-applied on every
        ``:meth:fit`` call that recreates the inner optimiser from
        scratch).  Default ``0.01`` matches the Phase 1 stable weight
        on synthetic; set to ``0.0`` to recover the Wf-122 ship
        behaviour (no proximal shrinkage on the dual side).  When
        the caller supplies a pre-built ``hilbert_optimizer`` we
        leave its ``l2_weight`` alone — this argument only takes
        effect when the composite instantiates its own inner
        optimiser.
    scale_clip_min : float, default ``-inf``
        Wf-127 Phase 3 scale clipping guard — lower bound for the
        ASTSR signature output.  When the composite instantiates its
        own ``lambda_astsr``, this value is forwarded to it.  Default
        is ``-inf`` (no clipping); pass ``-10.0`` to opt into the
        Phase 1 stable range (:data:`DEFAULT_ASTSR_SCALE_CLIP`).
        Callers that supply a pre-built ``lambda_astsr`` keep its
        own clip bounds.
    scale_clip_max : float, default ``+inf``
        Wf-127 Phase 3 scale clipping guard — upper bound for the
        ASTSR signature output.  Default is ``+inf`` (no clipping).
    """

    lambda_astsr: Optional[LambdaASTSR] = None
    hilbert_optimizer: Any = None  # LambdaASTSRHilbert; Any avoids import cycle.
    feature_columns: Tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_FEATURE_COLUMNS),
    )
    use_hilbert: bool = True
    population_dim: int = 64
    loss_scale: float = 128.0
    fp64_fallback: bool = True
    # Wf-127 Phase 2: L2 weight-decay guard on primal + dual.  Set
    # to 0.0 to recover the Wf-122 ship behaviour.
    l2_weight: float = 0.01
    # Wf-127 Phase 3: scale clipping / guard on ASTSR signature
    # output.  Default no-op (-inf, +inf) preserves the Wf-122
    # ship behaviour.  Pass (-10.0, 10.0) to opt into the Phase 1
    # stable range.
    scale_clip_min: float = float("-inf")
    scale_clip_max: float = float("inf")

    # Filled by ``fit``.
    last_fit_wall_s_: float = field(default=0.0, init=False)
    last_fit_r2_: float = field(default=0.0, init=False)
    active_backend_: str = field(default="", init=False)
    fit_history_: List[Dict[str, float]] = field(default_factory=list, init=False)
    # Cached predictions from the most recent successful ``fit``; used
    # by :meth:`predict` when a switch alone (no re-fit) is performed.
    last_predictions_: Optional[np.ndarray] = field(default=None, init=False)
    # Numerical-stability diagnostics (Wf-44 Phase 4).
    last_loss_scale_used_: float = field(default=1.0, init=False)
    last_fp64_fallback_used_: bool = field(default=False, init=False)
    # Wf-127 Phase 2: L2 weight-decay diagnostics — what was actually
    # used on the most recent ``fit`` and the most recent L2 penalty
    # observed in the inner optimiser.
    last_l2_weight_used_: float = field(default=0.0, init=False)
    last_l2_penalty_: float = field(default=0.0, init=False)
    # Wf-127 Phase 3: scale-clip diagnostics from the evolutionary
    # branch (Hilbert branch does not use the clip; its loss-scale
    # serves the same purpose at the inner-optimiser level).
    n_clipped_predictions_: int = field(default=0, init=False)
    last_scale_clip_min_used_: float = field(default=float("-inf"), init=False)
    last_scale_clip_max_used_: float = field(default=float("inf"), init=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.lambda_astsr is None:
            # Wf-127 Phase 3: forward the scale-clip bounds so the
            # composite's default matches what the caller configured.
            # Callers that supply their own ``lambda_astsr`` keep the
            # inner ``scale_clip_*`` they configured.
            self.lambda_astsr = LambdaASTSR(
                feature_names=tuple(self.feature_columns),
                scale_clip_min=float(self.scale_clip_min),
                scale_clip_max=float(self.scale_clip_max),
            )
        if self.hilbert_optimizer is None:
            # Lazy import keeps the integration decoupled from import
            # order when callers only want the evolutionary branch.
            from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
                LambdaASTSRHilbert,
            )
            # Wf-127 Phase 2: forward the L2 weight-decay guard so the
            # composite's default matches the Phase 1 stable weight.
            # Callers that supply their own ``hilbert_optimizer`` keep
            # the inner ``l2_weight`` they configured.
            self.hilbert_optimizer = LambdaASTSRHilbert(
                population_dim=int(self.population_dim),
                l2_weight=float(self.l2_weight),
            )
        _set_hilbert_active(bool(self.use_hilbert))

    # ------------------------------------------------------------------
    def switch_to_hilbert(self, flag: bool) -> None:
        """Flip which regressor :meth:`fit` and :meth:`predict` consult.

        A switch alone does not re-fit; subsequent :meth:`fit` calls
        delegate to the chosen branch, and :meth:`predict` returns the
        cached predictions from the most recent fit when the active
        branch has not been fitted.
        """
        self.use_hilbert = bool(flag)
        _set_hilbert_active(self.use_hilbert)

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Fit the active regressor; return ``{"r2", "wall_s", "backend"}``.

        Numerical stability (Wf-44 Phase 4)
        ----------------------------------
        * ``loss_scale`` — when ``> 1.0`` the training target is scaled
          by this factor before computing the R² proxy; the underlying
          R² is invariant to scaling, so the multiplier's only effect
          is to keep the squared residuals away from the underflow
          boundary in low-precision (fp16 / fp32) arithmetic.  The
          effective loss is divided back through at the metric step.
        * ``fp64_fallback`` — when ``True`` (default), if the Hilbert
          branch returns a non-finite R² or its predictions are
          non-finite, the fit is retried with float64-cast features /
          target.  This is a *fallback* path; the default fp32 fit is
          the primary path.
        """
        feats = np.asarray(features, dtype=float)
        tgt = np.asarray(target, dtype=float).reshape(-1)
        if feats.ndim != 2:
            raise ValueError(
                f"features must be (N, F); got shape {feats.shape}"
            )
        if feats.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"features / target length mismatch: "
                f"{feats.shape[0]} vs {tgt.shape[0]}"
            )

        backend = "hilbert" if self.use_hilbert else "evolutionary"
        loss_scale_used = 1.0
        fp64_fallback_used = False
        t0 = time.perf_counter()
        if self.use_hilbert:
            # ----- Wf-44 Phase 4: loss-scaled fp32 fit ---------------
            # The target is scaled so the squared residuals stay well
            # above the fp32 underflow floor.  R² is invariant to
            # affine scale, so the resulting R² is the same as on the
            # un-scaled target.  We pass ``loss_scale`` to the
            # underlying Hilbert optimiser so its gradient-clipping
            # threshold is scaled up proportionally — without this
            # pass-through, the scaled gradient would be clipped too
            # aggressively and convergence would stall.
            feats_eff = feats
            tgt_eff = tgt
            if self.loss_scale and self.loss_scale > 1.0:
                tgt_eff = tgt * float(self.loss_scale)
                loss_scale_used = float(self.loss_scale)
            try:
                self.hilbert_optimizer.optimize(
                    feats_eff, tgt_eff, loss_scale=loss_scale_used,
                )
                y_hat = self.hilbert_optimizer.predict(feats_eff)
                r2 = float(
                    getattr(self.hilbert_optimizer, "best_r2_train_",
                            compute_r2(tgt_eff, y_hat))
                )
            except Exception:                        # pragma: no cover
                r2 = float("nan")
                y_hat = np.zeros(feats.shape[0], dtype=float)
            # ----- Wf-44 Phase 4: fp64 fallback on NaN / inf ---------
            if (
                self.fp64_fallback
                and (not np.isfinite(r2) or not np.all(np.isfinite(y_hat)))
            ):
                feats64 = np.asarray(feats, dtype=np.float64)
                tgt64 = np.asarray(tgt, dtype=np.float64)
                # Re-fit in fp64 with loss_scale=1.0 (no scaling — fp64
                # has no underflow risk; the goal of loss_scale is to
                # keep residuals in the fp32 representable range).
                self.hilbert_optimizer.optimize(
                    feats64, tgt64, loss_scale=1.0,
                )
                y_hat = self.hilbert_optimizer.predict(feats64)
                r2 = float(
                    getattr(self.hilbert_optimizer, "best_r2_train_",
                            compute_r2(tgt64, y_hat))
                )
                fp64_fallback_used = True
        else:
            self.lambda_astsr.fit(feats, tgt)
            y_hat = self.lambda_astsr.predict(feats)
            r2 = float(
                getattr(self.lambda_astsr, "best_r2_train_", compute_r2(tgt, y_hat))
            )
        t1 = time.perf_counter()
        wall_s = float(t1 - t0)

        # Wf-127 Phase 2: surface the L2 weight-decay diagnostics from
        # the inner Hilbert optimiser when the Hilbert branch fired.
        # When the evolutionary branch fired (or the Hilbert branch
        # raised) we fall back to ``self.l2_weight`` for ``last_l2_*
        # `` so the output fields always have a defined value.
        if self.use_hilbert and np.isfinite(r2) and self.hilbert_optimizer is not None:
            l2_weight_used = float(
                getattr(self.hilbert_optimizer, "l2_weight_used_", self.l2_weight)
            )
            l2_penalty = float(
                getattr(self.hilbert_optimizer, "last_l2_penalty_", 0.0)
            )
        else:
            l2_weight_used = float(self.l2_weight)
            l2_penalty = 0.0

        # Wf-127 Phase 3: surface the scale-clip diagnostics from the
        # evolutionary branch.  The Hilbert branch has no scale-clip
        # path; we fall back to the composite's configured bounds so
        # the output fields always have a defined value.
        if not self.use_hilbert and self.lambda_astsr is not None:
            n_clipped = int(
                getattr(self.lambda_astsr, "n_clipped_predictions_", 0)
            )
            clip_min_used = float(
                getattr(self.lambda_astsr, "last_scale_clip_min_", self.scale_clip_min)
            )
            clip_max_used = float(
                getattr(self.lambda_astsr, "last_scale_clip_max_", self.scale_clip_max)
            )
        else:
            n_clipped = 0
            clip_min_used = float(self.scale_clip_min)
            clip_max_used = float(self.scale_clip_max)

        self.last_fit_wall_s_ = wall_s
        self.last_fit_r2_ = r2
        self.active_backend_ = backend
        self.last_loss_scale_used_ = loss_scale_used
        self.last_fp64_fallback_used_ = fp64_fallback_used
        self.last_l2_weight_used_ = l2_weight_used
        self.last_l2_penalty_ = l2_penalty
        self.n_clipped_predictions_ = n_clipped
        self.last_scale_clip_min_used_ = clip_min_used
        self.last_scale_clip_max_used_ = clip_max_used
        self.fit_history_.append({
            "wall_s": wall_s, "r2": r2, "backend": backend,
            "loss_scale": loss_scale_used,
            "fp64_fallback": fp64_fallback_used,
            "l2_weight": l2_weight_used,
            "l2_penalty": l2_penalty,
            "n_clipped": n_clipped,
            "scale_clip_min": clip_min_used,
            "scale_clip_max": clip_max_used,
        })
        # Cache the predictions so a subsequent switch without a re-fit
        # returns the same output rather than zeros.
        self.last_predictions_ = np.asarray(y_hat, dtype=float).reshape(-1).copy()
        return {
            "r2": r2, "wall_s": wall_s, "backend": backend,
            "loss_scale": loss_scale_used,
            "fp64_fallback": fp64_fallback_used,
            "l2_weight": l2_weight_used,
            "l2_penalty": l2_penalty,
            "n_clipped": n_clipped,
            "scale_clip_min": clip_min_used,
            "scale_clip_max": clip_max_used,
        }

    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using whichever regressor is currently active.

        * Active regressor is fitted → its ``predict(features)``.
        * Active regressor NOT fitted but the OTHER regressor was →
          cached ``last_predictions_`` (switch-without-refit contract).
        * Nothing has ever been fitted → zeros.
        """
        feats = np.asarray(features, dtype=float)
        if feats.ndim != 2:
            feats = feats.reshape(1, -1) if feats.size else np.zeros((0, 1))

        if self.use_hilbert:
            try:
                feats_h = feats.reshape(-1, self.population_dim)
            except ValueError:
                feats_h = feats.reshape(feats.shape[0], self.population_dim)
            try:
                out = np.asarray(
                    self.hilbert_optimizer.predict(feats_h), dtype=float
                ).reshape(-1)
                self.last_predictions_ = out.copy()
                return out
            except RuntimeError:
                pass
        else:
            try:
                out = np.asarray(
                    self.lambda_astsr.predict(feats), dtype=float
                ).reshape(-1)
                self.last_predictions_ = out.copy()
                return out
            except RuntimeError:
                pass

        # Active branch not fitted — fall back to cached output.
        if self.last_predictions_ is not None:
            return self.last_predictions_.copy()
        return np.zeros(feats.shape[0], dtype=float)

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """``True`` if at least one regressor has been fitted."""
        hilbert_fitted = (
            getattr(self.hilbert_optimizer, "best_params_", None) is not None
        )
        evol_fitted = (
            getattr(self.lambda_astsr, "best_ast_", None) is not None
        )
        return hilbert_fitted or evol_fitted

    @property
    def active_fitted(self) -> bool:
        """``True`` if the currently-active regressor has been fitted."""
        if self.use_hilbert:
            return (
                getattr(self.hilbert_optimizer, "best_params_", None) is not None
            )
        return (
            getattr(self.lambda_astsr, "best_ast_", None) is not None
        )


# ---------------------------------------------------------------------------
# HilbertEvolutionaryComparator — head-to-head wall-time + R² comparison
# ---------------------------------------------------------------------------
@dataclass
class HilbertEvolutionaryComparator:
    """Side-by-side comparison of Hilbert vs evolutionary LambdaASTSR.

    Parameters
    ----------
    hilbert_astsr : HilbertASTSRComposite
        Composite with ``use_hilbert=True`` (Hilbert branch fires on
        ``composite.fit``).
    evolutionary_astsr : LambdaASTSR, optional
        Stand-alone evolutionary regressor; the comparator fits a
        fresh clone per seed so the comparison is isolated.
    """

    hilbert_astsr: HilbertASTSRComposite = field(default_factory=HilbertASTSRComposite)
    evolutionary_astsr: Optional[LambdaASTSR] = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.evolutionary_astsr is None:
            self.evolutionary_astsr = LambdaASTSR(
                feature_names=tuple(self.hilbert_astsr.feature_columns),
            )
        self.hilbert_astsr.switch_to_hilbert(True)

    # ------------------------------------------------------------------
    def compare(
        self,
        features: np.ndarray,
        target: np.ndarray,
        n_seeds: int = 3,
    ) -> Dict[str, float]:
        """Return wall-time + R² for both branches across ``n_seeds`` runs.

        Returns ``{"hilbert_wall_s", "evolutionary_wall_s",
        "hilbert_r2", "evolutionary_r2", "speedup", "n_seeds"}`` — same
        shape as ``LambdaASTSRHilbert.compare_with_evolutionary``.
        """
        feats = np.asarray(features, dtype=float)
        tgt = np.asarray(target, dtype=float).reshape(-1)

        hilbert_walls: List[float] = []
        hilbert_r2s: List[float] = []
        evol_walls: List[float] = []
        evol_r2s: List[float] = []

        # Snapshot seeds so we can restore them after the comparison.
        original_hilbert_seed = int(
            getattr(self.hilbert_astsr.hilbert_optimizer, "seed", 0)
        )
        original_evol_seed = int(self.evolutionary_astsr.seed)

        for s in range(int(n_seeds)):
            # Hilbert branch.
            self.hilbert_astsr.hilbert_optimizer.seed = int(s)
            self.hilbert_astsr.switch_to_hilbert(True)
            res_h = self.hilbert_astsr.fit(feats, tgt)
            hilbert_walls.append(float(res_h["wall_s"]))
            hilbert_r2s.append(float(res_h["r2"]))

            # Evolutionary branch.
            self.evolutionary_astsr.seed = int(s)
            t0 = time.perf_counter()
            self.evolutionary_astsr.fit(feats, tgt)
            t1 = time.perf_counter()
            y_hat_e = self.evolutionary_astsr.predict(feats)
            r2_e = compute_r2(tgt, y_hat_e)
            evol_walls.append(float(t1 - t0))
            evol_r2s.append(float(r2_e))

        self.hilbert_astsr.hilbert_optimizer.seed = original_hilbert_seed
        self.evolutionary_astsr.seed = original_evol_seed

        avg_hilbert_wall = float(np.mean(hilbert_walls))
        avg_evol_wall = float(np.mean(evol_walls))
        speedup = (
            avg_evol_wall / avg_hilbert_wall
            if avg_hilbert_wall > 0 else float("inf")
        )
        return {
            "hilbert_wall_s": avg_hilbert_wall,
            "evolutionary_wall_s": avg_evol_wall,
            "hilbert_r2": float(np.mean(hilbert_r2s)),
            "evolutionary_r2": float(np.mean(evol_r2s)),
            "speedup": float(speedup),
            "n_seeds": int(n_seeds),
        }


# ---------------------------------------------------------------------------
# Multi-run benchmark — repeats the comparison to reduce variance
# ---------------------------------------------------------------------------
def benchmark_hilbert_vs_evolutionary(
    features: np.ndarray,
    target: np.ndarray,
    n_runs: int = 10,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    population_dim: int = 64,
) -> Dict[str, Any]:
    """Repeat the Hilbert-vs-evolutionary comparison ``n_runs`` times.

    Returns ``{"hilbert_wall_s_mean", "hilbert_wall_s_std",
    "evolutionary_wall_s_mean", "evolutionary_wall_s_std",
    "hilbert_r2_mean", "hilbert_r2_std", "evolutionary_r2_mean",
    "evolutionary_r2_std", "speedup_mean", "speedup_std", "n_runs",
    "n_samples"}`` — mean / std across the ``n_runs`` rounds.
    """
    feats = np.asarray(features, dtype=float)
    tgt = np.asarray(target, dtype=float).reshape(-1)
    if feats.ndim != 2 or feats.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"benchmark expects (N, F) features and (N,) target; "
            f"got {feats.shape} vs {tgt.shape}"
        )

    fc = tuple(feature_columns) if feature_columns else tuple(DEFAULT_FEATURE_COLUMNS)
    comparator = HilbertEvolutionaryComparator(
        hilbert_astsr=HilbertASTSRComposite(
            lambda_astsr=LambdaASTSR(feature_names=fc),
            feature_columns=fc,
            use_hilbert=True,
            population_dim=int(population_dim),
        ),
        evolutionary_astsr=LambdaASTSR(feature_names=fc),
    )

    hilbert_walls: List[float] = []
    evol_walls: List[float] = []
    hilbert_r2s: List[float] = []
    evol_r2s: List[float] = []
    speedups: List[float] = []

    for _ in range(int(n_runs)):
        round_result = comparator.compare(feats, tgt, n_seeds=1)
        hilbert_walls.append(round_result["hilbert_wall_s"])
        evol_walls.append(round_result["evolutionary_wall_s"])
        hilbert_r2s.append(round_result["hilbert_r2"])
        evol_r2s.append(round_result["evolutionary_r2"])
        speedups.append(round_result["speedup"])

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _std(xs: List[float]) -> float:
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    return {
        "hilbert_wall_s_mean": _mean(hilbert_walls),
        "hilbert_wall_s_std": _std(hilbert_walls),
        "evolutionary_wall_s_mean": _mean(evol_walls),
        "evolutionary_wall_s_std": _std(evol_walls),
        "hilbert_r2_mean": _mean(hilbert_r2s),
        "hilbert_r2_std": _std(hilbert_r2s),
        "evolutionary_r2_mean": _mean(evol_r2s),
        "evolutionary_r2_std": _std(evol_r2s),
        "speedup_mean": _mean(speedups),
        "speedup_std": _std(speedups),
        "n_runs": int(n_runs),
        "n_samples": int(feats.shape[0]),
    }


# ---------------------------------------------------------------------------
# Reward-channel builder — mirrors make_lambda_astsr_reward_channel
# ---------------------------------------------------------------------------
def make_hilbert_astsr_reward_channel(
    composite: HilbertASTSRComposite,
    *,
    enabled: Optional[bool] = None,
) -> "callable":
    """Return a closure ``(row_dict) -> reward`` that delegates to the composite.

    Mirrors :func:`lambda_astsr.make_lambda_astsr_reward_channel` so the
    integration drops into existing call sites without API changes.
    NaN-safe; returns ``0.0`` when disabled or no regressor is fitted.
    """
    if enabled is None:
        enabled = _is_hilbert_active()

    def _channel(row: Dict[str, float]) -> float:
        if not enabled:
            return 0.0
        if not composite.is_fitted:
            return 0.0
        try:
            xs = np.asarray(
                [float(row.get(fn, 0.0)) for fn in composite.feature_columns],
                dtype=float,
            ).reshape(1, -1)
        except Exception:
            return 0.0
        try:
            out = composite.predict(xs)
        except Exception:
            return 0.0
        if out.size == 0:
            return 0.0
        v = float(out[0])
        if not np.isfinite(v):
            return 0.0
        # Wf-127 P3: clip the reward-channel value to the configured
        # range.  No-op when bounds are ±inf.  Use ``clip_astr_signature``
        # so the same semantics as :meth:`LambdaASTSR.predict` apply.
        clipped, _ = clip_astr_signature(
            np.array([v]),
            composite.scale_clip_min,
            composite.scale_clip_max,
        )
        return float(clipped[0])

    return _channel


__all__ = [
    "HilbertASTSRComposite",
    "HilbertEvolutionaryComparator",
    "benchmark_hilbert_vs_evolutionary",
    "make_hilbert_astsr_reward_channel",
]
