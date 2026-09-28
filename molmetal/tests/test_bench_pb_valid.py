"""Tests for Wf-32 Phase 3 — 10-pocket × 5-candidate PoseBusters sweep.

These tests verify the integrity of the Wf-32 Phase-3 PB-Valid outputs
that ``molmetal/scripts/bench_pb_valid.py`` and
``molmetal/scripts/verify_bench_pb_valid.py`` produce:

* the 50-evaluation per-method metrics JSON,
* the slim per-pocket summary JSON,
* the headline ``overall_pb_pass_rate`` field,
* the constants / dataclasses / candidate loading helpers.

The tests are read-only / opt-in.  They skip when the Wf-32 Phase-3
outputs are missing (so CI without the prior sweep is not broken).

Test count target: ~17 distinct checks.

1.  ``test_module_imports`` — bench_pb_valid imports cleanly.
2.  ``test_default_pockets_count_10`` — DEFAULT_POCKETS has 10 entries.
3.  ``test_default_pb_modes`` — DEFAULT_PB_MODES = ('mol', 'dock').
4.  ``test_default_trajectory_methods`` — DEFAULT_TRAJECTORY_METHODS has 4 entries.
5.  ``test_n_pb_checks_dock_is_26`` — N_PB_CHECKS_DOCK == 26.
6.  ``test_n_pb_checks_mol_is_14`` — N_PB_CHECKS_MOL == 14.
7.  ``test_default_candidates_load`` — _demo_5_metaldrugs.txt yields 5 candidates.
8.  ``test_select_candidates_caps`` — select_candidates(..., n=2) returns 2.
9.  ``test_select_candidates_full`` — select_candidates(..., n=0) returns all.
10. ``test_load_pocket_bundle_existing`` — staged pocket resolves.
11. ``test_load_pocket_bundle_missing`` — missing pocket reports pdbqt_exists=False.
12. ``test_per_check_counts_for_passed_mol_report`` — synthetic report parsing.
13. ``test_per_check_counts_handles_missing_checks`` — missing fields tolerated.
14. ``test_coerce_status_passed`` — _coerce_status maps True → 'passed'.
15. ``test_coerce_status_skipped`` — skipped=True → 'skipped'/None.
16. ``test_aggregate_results_empty`` — aggregate_results returns empty on empty input.
17. ``test_aggregate_results_synthetic`` — 3-cell synthetic aggregation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
METRICS = PROJECT_ROOT / "metrics" / "by_metric"

# Wf-32 Phase-3 outputs (default paths; may not exist on a fresh checkout).
WF32_METRICS = METRICS / "wf32_pb_valid.json"
WF32_SUMMARY = METRICS / "wf32_pb_valid_summary.json"

WF32_POCKETS = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "1g9v",
    "2hnx",
    "4rn0",
    "1fmc",
)
N_POCKETS = 10
N_CANDIDATES = 5
N_PB_MODES = 2  # mol + dock
N_CELLS = N_POCKETS * N_CANDIDATES * N_PB_MODES  # 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"WF-32 Phase-3 output missing: {path}")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """``bench_pb_valid`` imports without error."""
    import sys
    p = str(SCRIPTS)
    if p not in sys.path:
        sys.path.insert(0, p)
    mod = __import__("bench_pb_valid")
    assert hasattr(mod, "run_sweep")
    assert hasattr(mod, "run_pb_one")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# 2-5. Constants
# ---------------------------------------------------------------------------
def test_default_pockets_count_10() -> None:
    """DEFAULT_POCKETS must list exactly 10 WF-30 pockets."""
    from bench_pb_valid import DEFAULT_POCKETS  # type: ignore
    assert len(DEFAULT_POCKETS) == N_POCKETS
    assert set(DEFAULT_POCKETS) == set(WF32_POCKETS)


def test_default_pb_modes() -> None:
    """DEFAULT_PB_MODES must be ('mol', 'dock')."""
    from bench_pb_valid import DEFAULT_PB_MODES  # type: ignore
    assert DEFAULT_PB_MODES == ("mol", "dock")


def test_default_trajectory_methods() -> None:
    """DEFAULT_TRAJECTORY_METHODS includes the 4 Wf-30 trajectory tags."""
    from bench_pb_valid import DEFAULT_TRAJECTORY_METHODS  # type: ignore
    assert set(DEFAULT_TRAJECTORY_METHODS) == {
        "se3sb", "transport", "multiph", "baseline",
    }


def test_n_pb_checks_dock_is_26() -> None:
    """N_PB_CHECKS_DOCK == 26 (14 chemistry + 12 protein-aware)."""
    from bench_pb_valid import (  # type: ignore
        N_PB_CHECKS_DOCK,
        N_PB_CHECKS_MOL,
        N_PB_CHECKS_TOTAL,
    )
    assert N_PB_CHECKS_DOCK == 26
    assert N_PB_CHECKS_MOL == 14
    assert N_PB_CHECKS_TOTAL == 26
    assert N_PB_CHECKS_DOCK == N_PB_CHECKS_MOL + 12


def test_default_candidates_load() -> None:
    """The default _demo_5_metaldrugs.txt must yield 5 candidates."""
    from bench_pb_valid import (  # type: ignore
        DEFAULT_CANDIDATES_TXT,
        load_default_candidates,
    )
    cands = load_default_candidates(DEFAULT_CANDIDATES_TXT)
    assert len(cands) >= N_CANDIDATES
    for c in cands[:N_CANDIDATES]:
        assert c.smiles, f"empty SMILES for candidate {c.name}"
        assert c.name, "candidate must have a non-empty name"


# ---------------------------------------------------------------------------
# 6-7. Candidate selection + pocket bundle
# ---------------------------------------------------------------------------
def test_select_candidates_caps() -> None:
    """select_candidates(..., n=2) returns the first 2 candidates."""
    from bench_pb_valid import (  # type: ignore
        DEFAULT_CANDIDATES_TXT,
        Candidate,
        load_default_candidates,
        select_candidates,
    )
    pool = load_default_candidates(DEFAULT_CANDIDATES_TXT)
    picked = select_candidates(pool, n=2)
    assert len(picked) == 2
    assert picked[0].name == pool[0].name
    assert picked[1].name == pool[1].name


def test_select_candidates_full() -> None:
    """select_candidates(..., n=0) returns the full pool."""
    from bench_pb_valid import (  # type: ignore
        DEFAULT_CANDIDATES_TXT,
        load_default_candidates,
        select_candidates,
    )
    pool = load_default_candidates(DEFAULT_CANDIDATES_TXT)
    picked = select_candidates(pool, n=0)
    assert len(picked) == len(pool)


def test_load_pocket_bundle_existing() -> None:
    """A staged pocket resolves to a PocketBundle with pdbqt_exists=True."""
    from bench_pb_valid import (  # type: ignore
        WF30_POCKET_DIR,
        load_pocket_bundle,
    )
    bundle = load_pocket_bundle("830c", WF30_POCKET_DIR)
    assert bundle.pocket_id == "830c"
    assert bundle.pdbqt_path.name == "receptor.pdbqt"
    assert bundle.pdbqt_exists is True


def test_load_pocket_bundle_missing() -> None:
    """A non-staged pocket reports pdbqt_exists=False (no exception)."""
    from bench_pb_valid import (  # type: ignore
        WF30_POCKET_DIR,
        load_pocket_bundle,
    )
    bundle = load_pocket_bundle("__no_such_pocket__", WF30_POCKET_DIR)
    assert bundle.pdbqt_exists is False


# ---------------------------------------------------------------------------
# 8-9. Per-check report parsing + status coercion
# ---------------------------------------------------------------------------
def test_per_check_counts_for_passed_mol_report() -> None:
    """_per_check_counts reads n_checks/n_passed/n_failed correctly."""
    from bench_pb_valid import _per_check_counts  # type: ignore

    report = {
        "checks": {"sanitization": True, "bond_lengths": True,
                   "internal_steric_clash": False},
        "failures": ["internal_steric_clash"],
        "missing_checks": (),
        "config": "mol",
    }
    n_total, n_passed, n_failed, n_skipped, failed, missing = (
        _per_check_counts(report)
    )
    assert n_total == 3
    assert n_passed == 2
    assert n_failed == 1
    assert n_skipped == 0
    assert failed == ("internal_steric_clash",)
    assert missing == ()


def test_per_check_counts_handles_missing_checks() -> None:
    """_per_check_counts tolerates missing-checks entries."""
    from bench_pb_valid import _per_check_counts  # type: ignore

    report = {
        "checks": {"sanitization": True, "bond_lengths": None},
        "failures": [],
        "missing_checks": ("cofactor_distance",),
        "config": "dock",
    }
    n_total, n_passed, n_failed, n_skipped, failed, missing = (
        _per_check_counts(report)
    )
    assert n_total == 3
    assert n_passed == 1
    assert n_failed == 0
    assert n_skipped == 2
    assert "cofactor_distance" in missing


def test_coerce_status_passed() -> None:
    """_coerce_status: True → ('passed', True)."""
    from bench_pb_valid import _coerce_status  # type: ignore
    status, pb_valid = _coerce_status({"pb_valid": True, "skipped": False})
    assert status == "passed"
    assert pb_valid is True


def test_coerce_status_skipped() -> None:
    """_coerce_status: skipped=True → ('skipped', None)."""
    from bench_pb_valid import _coerce_status  # type: ignore
    status, pb_valid = _coerce_status({"skipped": True, "pb_valid": None})
    assert status == "skipped"
    assert pb_valid is None


# ---------------------------------------------------------------------------
# 10. Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_results_empty() -> None:
    """aggregate_results returns ({}, {}) on empty input."""
    from bench_pb_valid import aggregate_results  # type: ignore
    per_pocket, aggregate = aggregate_results([])
    assert per_pocket == {}
    assert aggregate == {}


def test_aggregate_results_synthetic() -> None:
    """aggregate_results reduces 4 PBResults into per-pocket-per-method dict."""
    from bench_pb_valid import (  # type: ignore
        PBResult,
        aggregate_results,
    )
    results = [
        PBResult(pocket_id="830c", candidate="CCO", candidate_name="c0",
                 method="mol", trajectory_method="se3sb",
                 status="passed", pb_valid=True,
                 n_checks=14, n_passed=14, n_failed=0, n_skipped=0,
                 pass_rate=1.0, failed_checks=(), missing_checks=(),
                 elapsed_s=0.1),
        PBResult(pocket_id="830c", candidate="CCN", candidate_name="c1",
                 method="mol", trajectory_method="se3sb",
                 status="failed", pb_valid=False,
                 n_checks=14, n_passed=12, n_failed=2, n_skipped=1,
                 pass_rate=12 / 14, failed_checks=("bond_lengths",),
                 missing_checks=(), elapsed_s=0.1),
        PBResult(pocket_id="830c", candidate="CCO", candidate_name="c0",
                 method="dock", trajectory_method="se3sb",
                 status="skipped", pb_valid=None,
                 n_checks=0, n_passed=0, n_failed=0, n_skipped=0,
                 pass_rate=0.0, failed_checks=(), missing_checks=(),
                 elapsed_s=0.0, note="receptor missing"),
    ]
    per_pocket, aggregate = aggregate_results(results)
    assert "830c" in per_pocket
    assert "mol" in per_pocket["830c"]
    assert "dock" in per_pocket["830c"]
    block_mol = per_pocket["830c"]["mol"]
    assert block_mol["n"] == 2
    assert math.isclose(block_mol["mean_pass_rate"], (1.0 + 12 / 14) / 2, rel_tol=1e-6)
    block_dock = per_pocket["830c"]["dock"]
    assert block_dock["n"] == 1
    assert block_dock["n_skipped"] == 1


# ---------------------------------------------------------------------------
# 11-13. Optional — sweep outputs (skip when missing)
# ---------------------------------------------------------------------------
def test_wf32_metrics_exists() -> None:
    """wf32_pb_valid.json must exist and parse as JSON when present."""
    _skip_if_missing(WF32_METRICS)
    payload = _load(WF32_METRICS)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf32_pb_valid"
    assert payload.get("n_pockets") == N_POCKETS
    assert payload.get("n_candidates") == N_CANDIDATES
    assert payload.get("n_pb_modes") == N_PB_MODES


def test_wf32_summary_overall_finite() -> None:
    """wf32_pb_valid_summary.json must record finite overall rates or None."""
    _skip_if_missing(WF32_SUMMARY)
    payload = _load(WF32_SUMMARY)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf32_pb_valid_summary"
    overall = payload.get("overall_pb_pass_rate", {})
    assert "mol" in overall
    assert "dock" in overall
    for m, v in overall.items():
        assert v is None or (isinstance(v, (int, float)) and math.isfinite(v)), (
            f"{m}: non-finite overall_pb_pass_rate={v!r}"
        )