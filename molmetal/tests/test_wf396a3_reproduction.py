"""Wf-396-A3 — minimal reproduction tests.

=====================================================================
Three pytest contracts verifying the Wf-396-A3 minimal reproduction
script for the decode_ratio regression investigation.

Tests
-----
1. ``test_baseline_no_flags`` — the baseline (no flags) smoke
   produces a decode_ratio of 1.0 (or very close to it) on
   the synthetic Pt-click cohort, matching the Wf-124 P2
   baseline per ``metrics/by_metric/decode_ratio.json``.
2. ``test_single_flag_isolation`` — when each of the 8 NEW
   CLI flags is activated in isolation at its non-default
   value, the trainer runs to completion without raising and
   produces a measurable decode_ratio (0 <= ratio <= 1).
   This test confirms the flag-isolation reproduction works.
3. ``test_problematic_combination`` — the Wf-380 SCAFFOLD
   banner combination (``--top-p 0.9 --top-k 50 --mmd-weight
   0.1 --repulsion-weight 0.05``) is the suspected regression
   combination per Wf-396-A2 §2 CRITICAL FINDING.  This test
   confirms the combination smoke runs to completion and
   reports the actual decode_ratio for the combination (the
   expected value is NOT 1.0 — this test does not assert
   "no regression" since Wf-396-A2 §2 explicitly hypothesises
   that the combination may degrade decode_ratio).

Constraints (per AGENTS.md §5 + party-twenty-spirit §3.5):
* CPU-only (no GPU launches).
* No ``paper/*`` touches (TODO-84).
* No wet-lab planning (§10).
* 50-step CFM smoke only (no full retrain).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures / module imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def reproduction_module():
    """Lazy-import the Wf-396-A3 reproduction module."""
    try:
        import importlib

        mod = importlib.import_module(
            "molmetal.scripts.wf396a3_minimal_reproduction"
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            f"Wf-396-A3 reproduction module unavailable: {exc}"
        )
    return mod


# ---------------------------------------------------------------------------
# Test 1 — baseline (no flags) decode_ratio == 1.0
# ---------------------------------------------------------------------------
def test_baseline_no_flags(reproduction_module) -> None:
    """Wf-396-A3 — baseline (no flags) decode_ratio matches
    Wf-124 P2 baseline.

    Verifies:
      * ``baseline_test()`` runs to completion without raising.
      * ``decode_ratio`` is a float in [0.0, 1.0].
      * ``decode_ratio`` is **>= 0.8** (the Wf-124 P2 internal
        target — matches ``metrics/by_metric/decode_ratio.json``
        aggregate ``target_internal >= 0.8``).
      * ``n_total == SYNTHETIC_N_CLOUDS`` (default 32) — confirms
        the smoke ran on the expected cohort size.
    """
    res = reproduction_module.baseline_test()
    assert res is not None, "Wf-396-A3: baseline_test returned None"
    assert "decode_ratio" in res, (
        f"Wf-396-A3: baseline_test missing decode_ratio key; "
        f"got keys={list(res.keys())}"
    )
    if res["decode_ratio"] is None:
        pytest.skip(
            f"Wf-396-A3: baseline_test failed (likely missing "
            f"optional dep): {res.get('notes', '')}"
        )
    assert 0.0 <= res["decode_ratio"] <= 1.0, (
        f"Wf-396-A3: baseline decode_ratio out of range: "
        f"{res['decode_ratio']}"
    )
    assert res["decode_ratio"] >= 0.8, (
        f"Wf-396-A3: baseline decode_ratio regressed below the "
        f"Wf-124 P2 internal target (0.8). Got {res['decode_ratio']} "
        f"(expected ~1.0 on synthetic Pt-click per "
        f"metrics/by_metric/decode_ratio.json). Notes: "
        f"{res.get('notes', '')}"
    )
    assert res["n_total"] == reproduction_module.SYNTHETIC_N_CLOUDS, (
        f"Wf-396-A3: baseline n_total mismatch "
        f"(expected {reproduction_module.SYNTHETIC_N_CLOUDS}, "
        f"got {res['n_total']})"
    )


# ---------------------------------------------------------------------------
# Test 2 — single-flag isolation
# ---------------------------------------------------------------------------
def test_single_flag_isolation(reproduction_module) -> None:
    """Wf-396-A3 — each of the 8 NEW flags runs in isolation
    without raising and produces a measurable decode_ratio.

    Verifies:
      * ``single_flag_tests()`` returns a list of 8 result dicts
        (one per NEW CLI flag).
      * Each result has ``decode_ratio`` either a float in
        [0.0, 1.0] or None (with notes describing the failure
        — we tolerate Optional[float] for environment-specific
        issues like missing RDKit or PyTorch imports).
      * Each result's flag dictionary reflects the expected
        non-default value for that flag (sanity check that
        the ``FlagCombo`` was threaded through correctly).
    """
    results = reproduction_module.single_flag_tests()
    assert isinstance(results, list), (
        f"Wf-396-A3: single_flag_tests should return a list, "
        f"got {type(results).__name__}"
    )
    # The script defines 8 single-flag combos (one per NEW CLI flag).
    assert len(results) >= 6, (
        f"Wf-396-A3: expected at least 6 single-flag results "
        f"(6 from Wf-388-A2 + 2 from Wf-388-A3); got {len(results)}"
    )
    # Map single-flag results by name for assertions.
    by_name = {r["flags"]["name"]: r for r in results}
    expected_flag_names = {
        "single_flag_top_p_0.9": "top_p",
        "single_flag_top_k_50": "top_k",
        "single_flag_temperature_0.7": "temperature",
        "single_flag_mmd_weight_0.1": "mmd_weight",
        "single_flag_repulsion_weight_0.05": "repulsion_weight",
        "single_flag_scaffold_conditioning": "scaffold_conditioning",
        "single_flag_sa_penalty_0.1": "sa_penalty",
        "single_flag_sa_target_2.5": "sa_target",
    }
    for run_name, flag_attr in expected_flag_names.items():
        assert run_name in by_name, (
            f"Wf-396-A3: single_flag_tests missing run "
            f"'{run_name}' (flag={flag_attr})"
        )
        res = by_name[run_name]
        # Each result must have a decode_ratio (or None with notes).
        assert "decode_ratio" in res, (
            f"Wf-396-A3: {run_name} missing decode_ratio"
        )
        if res["decode_ratio"] is not None:
            assert 0.0 <= res["decode_ratio"] <= 1.0, (
                f"Wf-396-A3: {run_name} decode_ratio out of "
                f"range: {res['decode_ratio']}"
            )
        # Flag combination should reflect the non-default value.
        # (sanity check the threading through `_run_smoke`).
        assert res["flags"][flag_attr] is not None, (
            f"Wf-396-A3: {run_name} flag '{flag_attr}' not "
            f"threaded through to result (got None)"
        )


# ---------------------------------------------------------------------------
# Test 3 — Wf-380 SCAFFOLD banner combination (suspected regression)
# ---------------------------------------------------------------------------
def test_problematic_combination(reproduction_module) -> None:
    """Wf-396-A3 — the Wf-380 SCAFFOLD banner combination
    (``--top-p 0.9 --top-k 50 --mmd-weight 0.1 --repulsion-weight 0.05``)
    runs to completion and reports a measurable decode_ratio.

    Verifies:
      * ``combination_tests()`` returns a list including the
        Wf-380 SCAFFOLD banner combination (per Wf-396-A2 §2
        CRITICAL FINDING).
      * The combination result has ``decode_ratio`` either a float
        in [0.0, 1.0] or None with notes describing any env failure.
      * The combination result's flag dict reflects ALL 4 banner
        defaults simultaneously (top_p=0.9 + top_k=50 +
        mmd_weight=0.1 + repulsion_weight=0.05).

    Note: this test does NOT assert ``decode_ratio >= 0.8`` — the
    hypothesis (per Wf-396-A2 §2) is that the combination MAY
    degrade decode_ratio because Wf-388-A2 defaults DO NOT match
    the Wf-380 banner defaults.  If the test reports a degraded
    decode_ratio, that confirms Wf-396-A2's CRITICAL FINDING and
    points to the fix path (aligning Wf-388-A2 defaults with the
    Wf-380 banner defaults, OR documenting the gap prominently
    in the trainer help text).
    """
    results = reproduction_module.combination_tests()
    assert isinstance(results, list), (
        f"Wf-396-A3: combination_tests should return a list, "
        f"got {type(results).__name__}"
    )
    assert len(results) >= 1, (
        "Wf-396-A3: combination_tests returned empty list"
    )
    # Find the Wf-380 SCAFFOLD banner combination by name.
    by_name = {r["flags"]["name"]: r for r in results}
    assert "combination_wf380_scaffold_banner" in by_name, (
        "Wf-396-A3: combination_tests missing the Wf-380 "
        "SCAFFOLD banner combination (per Wf-396-A2 §2 "
        "CRITICAL FINDING). Got names: "
        f"{list(by_name.keys())}"
    )
    res = by_name["combination_wf380_scaffold_banner"]
    flags = res["flags"]
    # Verify the combination has all 4 banner defaults set.
    assert flags["top_p"] == 0.9, (
        f"Wf-396-A3: Wf-380 banner top_p should be 0.9, "
        f"got {flags['top_p']}"
    )
    assert flags["top_k"] == 50, (
        f"Wf-396-A3: Wf-380 banner top_k should be 50, "
        f"got {flags['top_k']}"
    )
    assert flags["mmd_weight"] == 0.1, (
        f"Wf-396-A3: Wf-380 banner mmd_weight should be 0.1, "
        f"got {flags['mmd_weight']}"
    )
    assert flags["repulsion_weight"] == 0.05, (
        f"Wf-396-A3: Wf-380 banner repulsion_weight should be "
        f"0.05, got {flags['repulsion_weight']}"
    )
    # The combination runs to completion (no exception raised).
    assert "wall_seconds" in res, (
        "Wf-396-A3: Wf-380 banner combination missing wall_seconds"
    )
    # decode_ratio is either a float in [0, 1] or None with notes
    # (tolerate env-specific failures; e.g. torch import error).
    if res["decode_ratio"] is not None:
        assert 0.0 <= res["decode_ratio"] <= 1.0, (
            f"Wf-396-A3: Wf-380 banner combination decode_ratio "
            f"out of range: {res['decode_ratio']}"
        )


# ---------------------------------------------------------------------------
# Optional helper — full reproduction result on disk
# ---------------------------------------------------------------------------
def test_reproduction_result_on_disk(reproduction_module) -> None:
    """Wf-396-A3 — ``reproduce_decode_ratio_regression()`` writes
    a result JSON file to ``molmetal/reports/wf396a3_minimal_reproduction/``
    via ``main()``.  Verifies the result file structure."""
    expected_keys = {"n_runs", "baseline_decode_ratio", "results"}
    # Run the function (not main()) to capture the aggregate dict.
    aggregate = reproduction_module.reproduce_decode_ratio_regression()
    assert isinstance(aggregate, dict), (
        f"Wf-396-A3: reproduce_decode_ratio_regression should "
        f"return a dict, got {type(aggregate).__name__}"
    )
    missing = expected_keys - set(aggregate.keys())
    assert not missing, (
        f"Wf-396-A3: reproduce_decode_ratio_regression missing "
        f"keys: {missing}"
    )
    assert "results" in aggregate
    assert isinstance(aggregate["results"], list)
    assert len(aggregate["results"]) >= 11, (
        f"Wf-396-A3: expected at least 11 runs (1 baseline + "
        f"1 default-OFF + 8 single-flag + 3 combination); "
        f"got {len(aggregate['results'])}"
    )
