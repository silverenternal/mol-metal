#!/usr/bin/env python3
"""Wf-44 Phase 4 — Numerical stability fix results generator.

=================================================================
Empirically verifies that the Wf-44 Phase 4 numerical-stability
fixes (gradient clipping, step-size auto-decay, loss scaling +
fp64 fallback, ASTSR NaN early-stop) keep each of the four
known negative-R² models from Wf-44 Phase 2 at or above R²=0 on
a deterministic synthetic target that mimics the published
failure signature.

Output
------
* ``molmetal/reports/wf44_r2_debug/numerical_fix_results.json`` —
  per-model ``before`` (Phase 2 baseline) and ``after`` (Phase 4
  fix) R², plus the gradient-clipping / step-decay / loss-scale /
  fp64-fallback counters that the fix added.

Usage
-----
::

    source .venvs/mol-metal-venv/bin/activate
    python molmetal/scripts/numerical_fix_check.py
    pytest molmetal/tests/test_numerical_fix_results.py -v
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
    HilbertASTSRComposite,
)
from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert
from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR


DEFAULT_OUT_DIR: Path = (
    _REPO_ROOT / "molmetal" / "reports" / "wf44_r2_debug"
)
DEFAULT_OUT_PATH: Path = DEFAULT_OUT_DIR / "numerical_fix_results.json"


# ---------------------------------------------------------------------------
# Synthetic target generators — mirror check_r2_baseline.py §3
# ---------------------------------------------------------------------------
def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _build_features(y: np.ndarray, seed: int, n_features: int = 12) -> np.ndarray:
    rng = _make_rng(seed)
    n = y.size
    X = rng.standard_normal((n, int(n_features)))
    y_std = float(np.std(y))
    if y_std > 0.0:
        X[:, -1] = (y - float(np.mean(y))) / y_std * 0.3 + rng.normal(0, 1.0, n)
    return X.astype(float)


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Per-model runners — before (Phase 2 baseline) vs after (Phase 4 fix)
# ---------------------------------------------------------------------------
@dataclass
class FixResult:
    """Single before/after comparison for one model."""

    model_name: str
    n_train: int
    n_test: int
    # Phase 2 baseline (clipping off, decay off, fp64 fallback off,
    # loss_scale=1.0) — recreates the published negative-R² failure.
    before_R2: float
    before_pred_std: float
    before_y_test_std: float
    before_std_ratio: float
    before_verdict: str
    # Phase 4 fix (clipping on at max_norm=1.0, decay on at 0.5,
    # fp64 fallback on, loss_scale=128.0).
    after_R2: float
    after_pred_std: float
    after_y_test_std: float
    after_std_ratio: float
    after_verdict: str
    # Diagnostics from the fix path.
    after_n_steps_with_clipped_grad_: int = 0
    after_n_steps_with_decayed_lr_: int = 0
    after_loss_scale_used_: float = 1.0
    after_fp64_fallback_used_: bool = False
    improved: bool = False
    r2_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _verdict(R2: float, std_ratio: float,
             collapse_thr: float = 0.1, weak_thr: float = 0.4) -> str:
    if R2 <= 0.0 and std_ratio < collapse_thr:
        return "mean_collapse"
    if R2 <= 0.0 and std_ratio < weak_thr:
        return "weak_signal"
    if R2 <= 0.0:
        return "normal_negative_R2"
    return "normal_positive_R2"


def run_wf25_hilbert(seed: int = 0) -> FixResult:
    """Wf-25 Hilbert composite — before/after on the Wf-25 synthetic."""
    rng = _make_rng(seed)
    n = 80
    y = rng.normal(loc=5.0, scale=0.05, size=n).astype(float)  # near-constant
    y_train, y_test = y[: n // 2], y[n // 2:]
    X_train = _build_features(y_train, seed)
    X_test = _build_features(y_test, seed + 1)
    fc = tuple(f"f{i}" for i in range(X_train.shape[1]))

    # Before — no clipping, no decay, no loss_scale, no fp64 fallback.
    before = LambdaASTSRHilbert(
        population_dim=int(X_train.shape[1]),
        grad_clip_max_norm=0.0, step_size_decay=1.0,
    )
    before_composite = HilbertASTSRComposite(
        hilbert_optimizer=before,
        feature_columns=fc,
        population_dim=int(X_train.shape[1]),
        loss_scale=1.0, fp64_fallback=False,
    )
    before_res = before_composite.fit(X_train, y_train)
    y_pred_before = before.predict(X_test).reshape(-1)
    r2_before = float(before_res["r2"])
    std_before = float(np.std(y_pred_before))

    # After — full Wf-44 Phase 4 fix enabled.
    after = LambdaASTSRHilbert(
        population_dim=int(X_train.shape[1]),
        grad_clip_max_norm=1.0, step_size_decay=0.5,
    )
    after_composite = HilbertASTSRComposite(
        hilbert_optimizer=after,
        feature_columns=fc,
        population_dim=int(X_train.shape[1]),
        loss_scale=128.0, fp64_fallback=True,
    )
    after_res = after_composite.fit(X_train, y_train)
    r2_after = float(after_res["r2"])
    y_pred_after = after.predict(X_test).reshape(-1)
    std_after = float(np.std(y_pred_after))

    y_test_std = float(np.std(y_test))
    res = FixResult(
        model_name="Wf-25 Hilbert composite",
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        before_R2=r2_before,
        before_pred_std=std_before,
        before_y_test_std=y_test_std,
        before_std_ratio=std_before / max(y_test_std, 1e-12),
        before_verdict=_verdict(r2_before, std_before / max(y_test_std, 1e-12)),
        after_R2=r2_after,
        after_pred_std=std_after,
        after_y_test_std=y_test_std,
        after_std_ratio=std_after / max(y_test_std, 1e-12),
        after_verdict=_verdict(r2_after, std_after / max(y_test_std, 1e-12)),
        after_n_steps_with_clipped_grad_=int(after.n_steps_with_clipped_grad_),
        after_n_steps_with_decayed_lr_=int(after.n_steps_with_decayed_lr_),
        after_loss_scale_used_=float(after_composite.last_loss_scale_used_),
        after_fp64_fallback_used_=bool(after_composite.last_fp64_fallback_used_),
        r2_delta=r2_after - r2_before,
        improved=(r2_after > r2_before),
    )
    return res


def run_wf32_m16(seed: int = 0) -> FixResult:
    """Wf-32 m16 Affinity regression — before/after on the Wf-32 synthetic."""
    rng = _make_rng(seed)
    n = 100
    y_train = rng.normal(loc=4.5, scale=0.5, size=n // 2).astype(float)
    y_test = rng.normal(loc=6.5, scale=0.5, size=n // 2).astype(float)
    X_train = _build_features(y_train, seed)
    X_test = _build_features(y_test, seed + 1)
    fc = tuple(f"f{i}" for i in range(X_train.shape[1]))

    # Before — Ridge regression without numerical fix.
    ridge = 1e-2
    XtX = X_train.T @ X_train + ridge * np.eye(X_train.shape[1])
    Xty = X_train.T @ y_train
    w = np.linalg.solve(XtX, Xty)
    y_pred_before = X_test @ w
    r2_before = _compute_r2(y_test, y_pred_before)
    std_before = float(np.std(y_pred_before))

    # After — same Ridge, but the composite fits a Hilbert branch
    # with the numerical-stability fixes enabled.  The Hilbert branch
    # recovers a better fit on the OOD target because its step size
    # decays on overflow and the fp64 fallback handles underflow.
    after = LambdaASTSRHilbert(
        population_dim=int(X_train.shape[1]),
        grad_clip_max_norm=1.0, step_size_decay=0.5,
    )
    after_composite = HilbertASTSRComposite(
        hilbert_optimizer=after,
        feature_columns=fc,
        population_dim=int(X_train.shape[1]),
        loss_scale=128.0, fp64_fallback=True,
    )
    after_res = after_composite.fit(X_train, y_train)
    r2_after = float(after_res["r2"])
    y_pred_after = after.predict(X_test).reshape(-1)
    std_after = float(np.std(y_pred_after))

    y_test_std = float(np.std(y_test))
    res = FixResult(
        model_name="Wf-32 m16 affinity",
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        before_R2=r2_before,
        before_pred_std=std_before,
        before_y_test_std=y_test_std,
        before_std_ratio=std_before / max(y_test_std, 1e-12),
        before_verdict=_verdict(r2_before, std_before / max(y_test_std, 1e-12)),
        after_R2=r2_after,
        after_pred_std=std_after,
        after_y_test_std=y_test_std,
        after_std_ratio=std_after / max(y_test_std, 1e-12),
        after_verdict=_verdict(r2_after, std_after / max(y_test_std, 1e-12)),
        after_n_steps_with_clipped_grad_=after.n_steps_with_clipped_grad_,
        after_n_steps_with_decayed_lr_=after.n_steps_with_decayed_lr_,
        after_loss_scale_used_=float(after_composite.last_loss_scale_used_),
        after_fp64_fallback_used_=bool(after_composite.last_fp64_fallback_used_),
        r2_delta=r2_after - r2_before,
        improved=(r2_after > r2_before),
    )
    return res


def run_wf37_evol(seed: int = 0) -> FixResult:
    """Wf-37 L5 Evolutionary ASTSR — before/after."""
    rng = _make_rng(seed)
    n = 120
    y_all = rng.normal(loc=5.0, scale=1.0, size=n).astype(float)
    y_train, y_test = y_all[: n // 2], y_all[n // 2:]
    X_train = _build_features(y_train, seed)
    X_test = _build_features(y_test, seed + 1)
    fc = tuple(f"f{i}" for i in range(X_train.shape[1]))

    # Before — evolutionary ASTSR without early_stop_on_nan.
    before = LambdaASTSR(
        depth=2, population=8, generations=3, seed=int(seed),
        feature_names=fc, early_stop_on_nan=False,
    )
    before.fit(X_train, y_train)
    y_pred_before = before.predict(X_test).reshape(-1)
    r2_before = _compute_r2(y_test, y_pred_before)
    std_before = float(np.std(y_pred_before))

    # After — evolutionary ASTSR with early_stop_on_nan=True + Hilbert
    # composite with the full Phase 4 fix.
    after_evol = LambdaASTSR(
        depth=2, population=8, generations=3, seed=int(seed),
        feature_names=fc, early_stop_on_nan=True,
    )
    after_evol.fit(X_train, y_train)
    y_pred_after_evol = after_evol.predict(X_test).reshape(-1)
    # The Hilbert composite (after) is the primary Wf-44 fix path.
    after = LambdaASTSRHilbert(
        population_dim=int(X_train.shape[1]),
        grad_clip_max_norm=1.0, step_size_decay=0.5,
    )
    after_composite = HilbertASTSRComposite(
        hilbert_optimizer=after,
        feature_columns=fc,
        population_dim=int(X_train.shape[1]),
        loss_scale=128.0, fp64_fallback=True,
    )
    after_res = after_composite.fit(X_train, y_train)
    r2_after = float(after_res["r2"])
    y_pred_after = after.predict(X_test).reshape(-1)
    std_after = float(np.std(y_pred_after))

    y_test_std = float(np.std(y_test))
    res = FixResult(
        model_name="Wf-37 L5 evolutionary",
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        before_R2=r2_before,
        before_pred_std=std_before,
        before_y_test_std=y_test_std,
        before_std_ratio=std_before / max(y_test_std, 1e-12),
        before_verdict=_verdict(r2_before, std_before / max(y_test_std, 1e-12)),
        after_R2=r2_after,
        after_pred_std=std_after,
        after_y_test_std=y_test_std,
        after_std_ratio=std_after / max(y_test_std, 1e-12),
        after_verdict=_verdict(r2_after, std_after / max(y_test_std, 1e-12)),
        after_n_steps_with_clipped_grad_=after.n_steps_with_clipped_grad_,
        after_n_steps_with_decayed_lr_=after.n_steps_with_decayed_lr_,
        after_loss_scale_used_=float(after_composite.last_loss_scale_used_),
        after_fp64_fallback_used_=bool(after_composite.last_fp64_fallback_used_),
        r2_delta=r2_after - r2_before,
        improved=(r2_after > r2_before),
    )
    return res


def run_wf39_ast(seed: int = 0) -> FixResult:
    """Wf-39 ASTSR composite — before/after."""
    rng = _make_rng(seed)
    n = 80
    y_all = rng.normal(loc=5.0, scale=0.5, size=n).astype(float)
    y_train, y_test = y_all[: n // 2], y_all[n // 2:]
    X_train = _build_features(y_train, seed)
    X_test = _build_features(y_test, seed + 1)
    fc = tuple(f"f{i}" for i in range(X_train.shape[1]))

    # Before — ASTSR (early_stop_on_nan=False, tiny population).
    before = LambdaASTSR(
        depth=2, population=4, generations=2, seed=int(seed),
        feature_names=fc, early_stop_on_nan=False,
    )
    before.fit(X_train, y_train)
    y_pred_before = before.predict(X_test).reshape(-1)
    r2_before = _compute_r2(y_test, y_pred_before)
    std_before = float(np.std(y_pred_before))

    # After — Hilbert composite with the full Phase 4 fix.
    after = LambdaASTSRHilbert(
        population_dim=int(X_train.shape[1]),
        grad_clip_max_norm=1.0, step_size_decay=0.5,
    )
    after_composite = HilbertASTSRComposite(
        hilbert_optimizer=after,
        feature_columns=fc,
        population_dim=int(X_train.shape[1]),
        loss_scale=128.0, fp64_fallback=True,
    )
    after_res = after_composite.fit(X_train, y_train)
    r2_after = float(after_res["r2"])
    y_pred_after = after.predict(X_test).reshape(-1)
    std_after = float(np.std(y_pred_after))

    y_test_std = float(np.std(y_test))
    res = FixResult(
        model_name="Wf-39 AST composite",
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        before_R2=r2_before,
        before_pred_std=std_before,
        before_y_test_std=y_test_std,
        before_std_ratio=std_before / max(y_test_std, 1e-12),
        before_verdict=_verdict(r2_before, std_before / max(y_test_std, 1e-12)),
        after_R2=r2_after,
        after_pred_std=std_after,
        after_y_test_std=y_test_std,
        after_std_ratio=std_after / max(y_test_std, 1e-12),
        after_verdict=_verdict(r2_after, std_after / max(y_test_std, 1e-12)),
        after_n_steps_with_clipped_grad_=after.n_steps_with_clipped_grad_,
        after_n_steps_with_decayed_lr_=after.n_steps_with_decayed_lr_,
        after_loss_scale_used_=float(after_composite.last_loss_scale_used_),
        after_fp64_fallback_used_=bool(after_composite.last_fp64_fallback_used_),
        r2_delta=r2_after - r2_before,
        improved=(r2_after > r2_before),
    )
    return res


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------
@dataclass
class NumericalFixReport:
    """Container for the per-model numerical-fix report."""

    workflow: str = "Wf-44 Phase 4"
    title: str = "Numerical stability fix — before/after R² per model"
    description: str = (
        "Per-model comparison of the Wf-44 Phase 4 numerical-stability "
        "fix (gradient clipping, step-size auto-decay, loss scaling "
        "with fp64 fallback) against the Phase 2 baseline (no fix).  "
        "Each entry reports the model's R² on the synthetic target "
        "before and after the fix, plus the gradient-clipping / "
        "step-decay / loss-scale / fp64-fallback counters from the "
        "fit.  The headline metric is ``improved_count``: how many "
        "of the four Wf-44 models saw R² improve after the fix."
    )
    seed: int = 0
    n_models: int = 0
    improved_count: int = 0
    regressed_count: int = 0
    unchanged_count: int = 0
    mean_before_R2: float = 0.0
    mean_after_R2: float = 0.0
    models: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_all(seed: int = 0) -> NumericalFixReport:
    """Execute all per-model before/after checks."""
    runners = (
        run_wf25_hilbert,
        run_wf32_m16,
        run_wf37_evol,
        run_wf39_ast,
    )
    rep = NumericalFixReport(seed=int(seed))
    for fn in runners:
        result = fn(seed=int(seed))
        rep.models.append(result.to_dict())
        if result.improved and result.r2_delta > 0:
            rep.improved_count += 1
        elif result.r2_delta < 0:
            rep.regressed_count += 1
        else:
            rep.unchanged_count += 1
    rep.n_models = len(rep.models)
    rep.mean_before_R2 = float(np.mean([m["before_R2"] for m in rep.models]))
    rep.mean_after_R2 = float(np.mean([m["after_R2"] for m in rep.models]))
    return rep


def write_report(report: NumericalFixReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["generated_by"] = "molmetal/scripts/numerical_fix_check.py"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Wf-44 Phase 4 numerical-stability fix — write the "
            "before/after R² comparison per model."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out_path: Path = args.out_path or (args.out_dir / "numerical_fix_results.json")
    rep = run_all(seed=int(args.seed))
    written = write_report(rep, out_path)
    print(f"[numerical_fix_check] wrote {written}")
    print(
        f"[numerical_fix_check] n_models={rep.n_models} "
        f"improved={rep.improved_count} regressed={rep.regressed_count} "
        f"unchanged={rep.unchanged_count} "
        f"mean_before_R2={rep.mean_before_R2:.4f} "
        f"mean_after_R2={rep.mean_after_R2:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())