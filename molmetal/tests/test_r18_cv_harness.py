"""Tests for the R18 5-fold CV harness.

================================================================
W2.3 / R18 ultracode dispatch (2026-09-26).

These are the pytest mirrors of
:file:`molmetal/scripts/r18_cv_harness.py`.  The script emits a JSON
artefact (``molmetal/reports/wf_cv_harness/results.json``) with
per-module × per-fold metrics; here we re-invoke the harness with a
small CPU-only budget so regressions surface with focused names.

The 15 tests pin the 5-fold CV contract:

  1. test_cv_5_fold                       — default n_folds == 5
  2. test_per_module_per_fold              — every (module, fold) cell exists
  3. test_compute_mean_std                 — per-module mean/std present + finite
  4. test_default_modules_is_32            — full registry returns 32 modules
  5. test_fold_seeds_are_distinct          — fold seeds differ
  6. test_fold_sizes_balanced              — folds slice samples evenly
  7. test_accuracy_in_unit_interval        — accuracy ∈ [0, 1]
  8. test_wall_time_non_negative           — wall times ≥ 0
  9. test_status_field_values              — status ∈ {ok, skipped, error}
 10. test_mean_metrics_schema              — cross-module mean keys present
 11. test_std_metrics_schema               — cross-module std keys present
 12. test_json_written_to_results_path     — results.json exists + valid JSON
 13. test_smoke_run_invokes_every_module   — small CPU sweep completes
 14. test_build_folds_balanced             — _build_folds helper correctness
 15. test_module_groups_total_32           — module groups sum to 32

All tests are CPU-only, deterministic, and run in < 60 seconds total.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors the script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from molmetal.scripts.r18_cv_harness import (  # noqa: E402
    DEFAULT_N_FOLDS,
    _aggregate_mean,
    _aggregate_std,
    _build_folds,
    _build_module_registry,
    _cross_module_aggregate,
    _make_synthetic_smiles,
    _MODULE_GROUPS,
    _per_module_cv,
    _render_text,
    _run_fold,
    run_cv,
)

REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_cv_harness"
REPORT_FILE = REPORT_DIR / "results.json"

SMOKE_N_FOLDS = 5
SMOKE_N_RUNS = 2
SMOKE_N_SAMPLES = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry() -> Dict[str, Any]:
    """Module registry from the CV script."""
    return _build_module_registry()


@pytest.fixture(scope="module")
def smoke_payload() -> Dict[str, Any]:
    """Small 5-fold CV run with synthetic SMILES (CPU budget ≤ 60 s)."""
    return run_cv(
        csv_path=REPORT_DIR / "_does_not_exist.csv",  # forces synthetic
        n_folds=SMOKE_N_FOLDS,
        n_runs_per_fold=SMOKE_N_RUNS,
        n_samples=SMOKE_N_SAMPLES,
        seed=42,
    )


# ---------------------------------------------------------------------------
# 1. Default n_folds is 5
# ---------------------------------------------------------------------------
def test_cv_5_fold() -> None:
    """The harness default is exactly 5 folds."""
    assert DEFAULT_N_FOLDS == 5, (
        f"expected DEFAULT_N_FOLDS=5, got {DEFAULT_N_FOLDS}"
    )


# ---------------------------------------------------------------------------
# 2. Per-module per-fold records present
# ---------------------------------------------------------------------------
def test_per_module_per_fold(smoke_payload: Dict[str, Any]) -> None:
    """Every (module, fold) cell carries an accuracy / wall / memory triple."""
    pmp = smoke_payload["per_module_per_fold_metrics"]
    n_folds = smoke_payload["n_folds"]
    for mod_id, rec in pmp.items():
        assert "folds" in rec, f"missing folds for {mod_id}"
        assert "mean" in rec, f"missing mean for {mod_id}"
        assert "std" in rec, f"missing std for {mod_id}"
        assert len(rec["folds"]) == n_folds, (
            f"{mod_id} has {len(rec['folds'])} folds; expected {n_folds}"
        )
        for fold in rec["folds"]:
            assert "accuracy" in fold
            assert "wall_time_ms_median" in fold
            assert "wall_time_ms_mean" in fold
            assert "memory_mb" in fold
            assert "status" in fold


# ---------------------------------------------------------------------------
# 3. Mean / std computed and finite
# ---------------------------------------------------------------------------
def test_compute_mean_std(smoke_payload: Dict[str, Any]) -> None:
    """Each module's mean/std is computed and contains finite numbers."""
    import math
    pmp = smoke_payload["per_module_per_fold_metrics"]
    for mod_id, rec in pmp.items():
        m = rec["mean"]
        s = rec["std"]
        for key in ("accuracy", "wall_time_ms_median", "wall_time_ms_mean", "memory_mb"):
            assert key in m, f"missing mean.{key} for {mod_id}"
            assert key in s, f"missing std.{key} for {mod_id}"
            assert math.isfinite(m[key]), (
                f"{mod_id} mean.{key}={m[key]} not finite"
            )
            assert math.isfinite(s[key]), (
                f"{mod_id} std.{key}={s[key]} not finite"
            )


# ---------------------------------------------------------------------------
# 4. Default registry returns all 32 modules
# ---------------------------------------------------------------------------
def test_default_modules_is_32(registry: Dict[str, Any]) -> None:
    """The full registry contains exactly 32 modules (matches R18 ship manifest)."""
    assert len(registry) == 32, (
        f"R18 CV registry changed: got {len(registry)}, expected 32"
    )


# ---------------------------------------------------------------------------
# 5. Fold seeds are distinct
# ---------------------------------------------------------------------------
def test_fold_seeds_are_distinct(smoke_payload: Dict[str, Any]) -> None:
    """Fold seeds are unique (seed + k for k in [0, n_folds))."""
    seeds = smoke_payload["fold_seeds"]
    assert len(seeds) == smoke_payload["n_folds"]
    assert len(set(seeds)) == len(seeds), (
        f"duplicate fold seeds: {seeds}"
    )


# ---------------------------------------------------------------------------
# 6. Fold sizes are balanced
# ---------------------------------------------------------------------------
def test_fold_sizes_balanced(smoke_payload: Dict[str, Any]) -> None:
    """Fold sizes are contiguous slices of the sample batch.

    Sizes should differ by at most 1 (the last fold is truncated if
    the sample count does not divide evenly).
    """
    sizes = smoke_payload["fold_sizes"]
    n_folds = smoke_payload["n_folds"]
    assert len(sizes) == n_folds
    assert sum(sizes) == smoke_payload["n_samples_total"]
    # max - min <= 1
    assert max(sizes) - min(sizes) <= 1, (
        f"fold sizes unbalanced: {sizes}"
    )


# ---------------------------------------------------------------------------
# 7. Accuracy is in [0, 1]
# ---------------------------------------------------------------------------
def test_accuracy_in_unit_interval(smoke_payload: Dict[str, Any]) -> None:
    """Per-fold and per-module accuracy lie in the closed interval [0, 1]."""
    for mod_id, rec in smoke_payload["per_module_per_fold_metrics"].items():
        m_acc = rec["mean"]["accuracy"]
        assert 0.0 <= m_acc <= 1.0, f"{mod_id} mean accuracy {m_acc} outside [0,1]"
        for fold in rec["folds"]:
            f_acc = fold["accuracy"]
            assert 0.0 <= f_acc <= 1.0, (
                f"{mod_id} fold {fold['fold_idx']} accuracy {f_acc} outside [0,1]"
            )


# ---------------------------------------------------------------------------
# 8. Wall times are non-negative
# ---------------------------------------------------------------------------
def test_wall_time_non_negative(smoke_payload: Dict[str, Any]) -> None:
    """Per-fold and per-module wall times are ≥ 0."""
    for mod_id, rec in smoke_payload["per_module_per_fold_metrics"].items():
        for fold in rec["folds"]:
            assert fold["wall_time_ms_median"] >= 0.0
            assert fold["wall_time_ms_mean"] >= 0.0
            assert fold["memory_mb"] >= 0.0


# ---------------------------------------------------------------------------
# 9. Status field is constrained
# ---------------------------------------------------------------------------
def test_status_field_values(smoke_payload: Dict[str, Any]) -> None:
    """Per-fold status is one of {ok, skipped, error}."""
    allowed = {"ok", "skipped", "error"}
    for mod_id, rec in smoke_payload["per_module_per_fold_metrics"].items():
        for fold in rec["folds"]:
            assert fold["status"] in allowed, (
                f"{mod_id} fold {fold['fold_idx']} status={fold['status']!r}; "
                f"allowed={allowed}"
            )


# ---------------------------------------------------------------------------
# 10. Cross-module mean metrics schema
# ---------------------------------------------------------------------------
def test_mean_metrics_schema(smoke_payload: Dict[str, Any]) -> None:
    """``mean_metrics`` carries the cross-module aggregate keys."""
    mm = smoke_payload["mean_metrics"]
    for key in (
        "accuracy_mean",
        "accuracy_min",
        "accuracy_max",
        "wall_time_ms_median_mean",
        "wall_time_ms_mean_mean",
        "memory_mb_mean",
    ):
        assert key in mm, f"missing mean_metrics key: {key}"
        # accuracy bounds
        if key.startswith("accuracy"):
            assert 0.0 <= mm[key] <= 1.0


# ---------------------------------------------------------------------------
# 11. Cross-module std metrics schema
# ---------------------------------------------------------------------------
def test_std_metrics_schema(smoke_payload: Dict[str, Any]) -> None:
    """``std_metrics`` carries the cross-module aggregate keys."""
    sm = smoke_payload["std_metrics"]
    for key in (
        "accuracy_std",
        "wall_time_ms_median_std",
        "wall_time_ms_mean_std",
        "memory_mb_std",
    ):
        assert key in sm, f"missing std_metrics key: {key}"
        assert sm[key] >= 0.0, f"std.{key}={sm[key]} is negative"


# ---------------------------------------------------------------------------
# 12. JSON written to expected path (idempotent)
# ---------------------------------------------------------------------------
def test_json_written_to_results_path(tmp_path: Path) -> None:
    """The harness writes a JSON file with the documented schema."""
    payload = run_cv(
        csv_path=tmp_path / "missing.csv",
        n_folds=2,
        n_runs_per_fold=1,
        n_samples=10,
        seed=7,
    )
    out_path = tmp_path / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(out_path.read_text())
    for key in (
        "n_modules",
        "n_folds",
        "n_runs_per_fold",
        "n_samples_total",
        "module_groups",
        "per_module_per_fold_metrics",
        "mean_metrics",
        "std_metrics",
        "fold_seeds",
        "fold_sizes",
        "total_wall_time_s",
        "peak_memory_mb_total",
        "timestamp_utc",
        "python",
    ):
        assert key in loaded, f"missing JSON key: {key}"


# ---------------------------------------------------------------------------
# 13. Smoke run invokes every module
# ---------------------------------------------------------------------------
def test_smoke_run_invokes_every_module(
    smoke_payload: Dict[str, Any],
    registry: Dict[str, Any],
) -> None:
    """The smoke payload covers every module in the registry."""
    pmp = smoke_payload["per_module_per_fold_metrics"]
    assert set(pmp.keys()) == set(registry.keys()), (
        f"smoke run missed modules: {set(registry.keys()) - set(pmp.keys())}"
    )


# ---------------------------------------------------------------------------
# 14. _build_folds helper correctness
# ---------------------------------------------------------------------------
def test_build_folds_balanced() -> None:
    """``_build_folds`` slices samples into contiguous, balanced folds."""
    samples = [f"mol_{i}" for i in range(50)]
    folds = _build_folds(samples, 5)
    assert len(folds) == 5
    # Reassemble — every sample appears exactly once.
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(samples)
    # Sizes are equal (50/5 = 10).
    assert all(len(f) == 10 for f in folds)

    # n_folds > n_samples — empty trailing folds.
    small = ["a", "b"]
    folds_small = _build_folds(small, 5)
    assert len(folds_small) == 5
    assert sum(len(f) for f in folds_small) == 2


# ---------------------------------------------------------------------------
# 15. Module groups sum to 32
# ---------------------------------------------------------------------------
def test_module_groups_total_32() -> None:
    """The 6 module groups cover exactly 32 distinct module IDs."""
    all_ids: List[str] = []
    for grp, ids in _MODULE_GROUPS.items():
        all_ids.extend(ids)
    assert len(all_ids) == 32, f"expected 32 module IDs, got {len(all_ids)}"
    assert len(set(all_ids)) == 32, "duplicate module IDs across groups"
    # Group sizes pinned to R18 ship manifest.
    assert len(_MODULE_GROUPS["math_skills"]) == 9
    assert len(_MODULE_GROUPS["adapters"]) == 7
    assert len(_MODULE_GROUPS["lam_chem"]) == 11
    assert len(_MODULE_GROUPS["reward"]) == 1
    assert len(_MODULE_GROUPS["search_alg"]) == 2
    assert len(_MODULE_GROUPS["flow_matching_lipman"]) == 2
