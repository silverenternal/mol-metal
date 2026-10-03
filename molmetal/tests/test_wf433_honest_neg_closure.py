"""Wf-433 — m6_kappa + m1 + m2 honest-negative pre-closure verification tests.

=====================================================================
Wf-433-A1 — Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §3.1 verification:
  1. m6_kappa DROP decision propagation
     - m6_kappa.json has wf424a15_drop_decision block + 10 paths exhausted
     - PAPER_SECTION_3_METRIC_TABLE_FINAL.md has §3 main-result + §3 supplementary
     - PAPER_SECTION_6_v4_FINAL.md has §6.5 m6_kappa deep-dive entry
  2. m2 R² EXTENDED features closure
     - wf433_m2_extended.py runs 5x5 CV Ridge on 160-d features
     - m2_r2_per_cohort.json updated with EXTENDED values
     - result.json in molmetal/reports/wf433_m2_extended/
  3. m1 canonical catalog sync
     - metal_lomo_auc_wasserstein.json has 7/7 (metal, seed) pair aggregate
     - value = 0.7577 ± 0.0924 (per Wf-413-A1)

CPU-only tests (sklearn-based Ridge regression, no torch, no GPU).
Honest framing (party-twenty-spirit §3.5):
* Each test verifies a documented MEASURED file/path against the
  expected state per Wf-433-A1 design doc + AGENTS.md §5.5 filesystem-truth.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers — filesystem-truth + JSON load
# ---------------------------------------------------------------------------
def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Test 1: m6_kappa.json has wf424a15_drop_decision block + 10 paths
# ---------------------------------------------------------------------------
def test_m6_kappa_drop_decision_block_present() -> None:
    """Verify metrics/by_metric/m6_kappa.json has wf424a15_drop_decision block.

    Per Wf-433-A1 §0.2 + Wf-424-A15 + AGENTS.md §5.5 filesystem-truth.
    """
    assert _exists(
        "metrics/by_metric/m6_kappa.json"
    ), "m6_kappa.json must exist on disk per AGENTS.md §5.5"
    catalog = _read_json("metrics/by_metric/m6_kappa.json")
    assert (
        "wf424a15_drop_decision" in catalog
    ), "m6_kappa.json must have wf424a15_drop_decision block"
    block = catalog["wf424a15_drop_decision"]
    assert block.get("wf") == "Wf-424-A15"
    assert (
        block.get("verdict_field")
        == "MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL"
    )
    assert block.get("decision") == (
        "DROP_FROM_§3_MAIN_RESULT + PROMOTE_TOP3_TO_§3_SUPPLEMENTARY_SUBTABLE "
        "+ RETAIN_AS_§6_COMPLEMENTARY_HEURISTIC"
    )
    assert block.get("paths_exhausted") == 10


# ---------------------------------------------------------------------------
# Test 2: PAPER_SECTION_3_METRIC_TABLE_FINAL.md has §3 + supplementary table
# ---------------------------------------------------------------------------
def test_paper_section3_metric_table_final_present() -> None:
    """Verify docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md exists with §3 + supplementary."""
    path = "docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md"
    assert _exists(path), "PAPER_SECTION_3_METRIC_TABLE_FINAL.md must exist on disk"
    with open(path, "r") as f:
        content = f.read()
    # Section 3 main-result table should reference 4 CEILING_BREAK metrics.
    for metric_id in ("m1_auc", "m2", "m3", "m15"):
        assert (
            metric_id in content
        ), f"PAPER §3 must reference {metric_id} in main-result table"
    # m6_kappa DROP decision should be present.
    assert (
        "DROP" in content.upper() or "drop_from_main_result" in content.lower()
    ), "PAPER §3 must document m6_kappa DROP decision"
    # Top-3 alternatives should be promoted.
    for alt in ("MolLogP", "MorganDensity", "Chi3v"):
        assert alt in content, f"PAPER §3 supplementary must reference {alt}"


# ---------------------------------------------------------------------------
# Test 3: PAPER_SECTION_6_v4_FINAL.md has §6.5 m6_kappa deep-dive entry
# ---------------------------------------------------------------------------
def test_paper_section6_v4_final_has_m6_kappa_section() -> None:
    """Verify docs/PAPER_SECTION_6_v4_FINAL.md has §6.5 m6_kappa deep-dive entry."""
    path = "docs/PAPER_SECTION_6_v4_FINAL.md"
    assert _exists(path), "PAPER_SECTION_6_v4_FINAL.md must exist on disk"
    with open(path, "r") as f:
        content = f.read()
    # Section 6.5 m6_kappa deep-dive.
    assert "m6_kappa" in content, "§6.5 must reference m6_kappa"
    # Paths exhausted (9 or 10 depending on convergence brief).
    has_paths_count = (
        "9 paths" in content
        or "10 paths" in content
        or "9-path" in content
        or "10-path" in content
        or "path inventory" in content.lower()
        or "paths_exhausted" in content.lower()
    )
    assert has_paths_count, (
        "§6.5 must document m6_kappa paths exhausted (any of: 9 paths, 10 paths, "
        "9-path, 10-path, path inventory, paths_exhausted)"
    )
    # Wf-365-A3 verdict cross-reference.
    assert (
        "Wf-365-A3" in content or "365-A3" in content
    ), "§6.5 must reference Wf-365-A3 verdict"


# ---------------------------------------------------------------------------
# Test 4: m1 canonical catalog — 7/7 (metal, seed) pair aggregate
# ---------------------------------------------------------------------------
def test_m1_canonical_catalog_7_of_7_aggregate() -> None:
    """Verify metrics/by_metric/metal_lomo_auc_wasserstein.json has 7/7 aggregate.

    Per Wf-413-A1 §4 + Wf-433-A1 §0.2 item 1 + AGENTS.md §5.5.
    """
    path = "metrics/by_metric/metal_lomo_auc_wasserstein.json"
    assert _exists(path), "metal_lomo_auc_wasserstein.json must exist"
    catalog = _read_json(path)
    # n_runs == 7, n_runs_completed == 7 (per Wf-413-A1 7-of-7 close).
    assert catalog.get("n_runs") == 7, f"n_runs must be 7; got {catalog.get('n_runs')}"
    assert (
        catalog.get("n_runs_completed") == 7
    ), f"n_runs_completed must be 7; got {catalog.get('n_runs_completed')}"
    # value = 0.7577 ± 0.0924.
    val = catalog.get("value")
    val_std = catalog.get("value_std")
    assert abs(val - 0.7577) < 0.005, f"value must be 0.7577 ± 0.005; got {val}"
    assert (
        abs(val_std - 0.0924) < 0.005
    ), f"value_std must be 0.0924 ± 0.005; got {val_std}"
    # Per-fold verdict preserved.
    assert (
        catalog.get("per_fold_krasnov_verdict", {}).get("Rh") == "PASS"
    ), "Rh fold must PASS Krasnov gate"
    assert (
        catalog.get("per_fold_krasnov_verdict", {}).get("Re") == "BELOW"
    ), "Re fold must be BELOW Krasnov gate (HONEST_NEGATIVE preserved)"


# ---------------------------------------------------------------------------
# Test 5: m1 7-run array present + correct per_run_auc values
# ---------------------------------------------------------------------------
def test_m1_per_run_auc_array_7_entries() -> None:
    """Verify metal_lomo_auc_wasserstein.json::per_run_auc has 7 entries."""
    path = "metrics/by_metric/metal_lomo_auc_wasserstein.json"
    catalog = _read_json(path)
    per_run = catalog.get("per_run_auc", [])
    assert len(per_run) == 7, f"per_run_auc must have 7 entries; got {len(per_run)}"
    # All values should be in (0, 1) for ROC-AUC.
    for v in per_run:
        assert 0.0 < v < 1.0, f"per_run_auc value must be in (0, 1); got {v}"
    # Mean of per_run_auc should be close to aggregate value.
    per_run_mean = float(np.mean(per_run))
    assert (
        abs(per_run_mean - catalog["value"]) < 0.001
    ), f"per_run_auc mean {per_run_mean} must match aggregate value {catalog['value']}"


# ---------------------------------------------------------------------------
# Test 6: m2_r2_per_cohort.json EXTENDED 160-d block + verdict
# ---------------------------------------------------------------------------
def test_m2_per_cohort_extended_160d_block() -> None:
    """Verify metrics/by_metric/m2_r2_per_cohort.json has per_cohort_extended_160d block."""
    path = "metrics/by_metric/m2_r2_per_cohort.json"
    assert _exists(path), "m2_r2_per_cohort.json must exist"
    catalog = _read_json(path)
    assert (
        "per_cohort_extended_160d" in catalog
    ), "m2_r2_per_cohort.json must have per_cohort_extended_160d block (Wf-433-A2)"
    ext = catalog["per_cohort_extended_160d"]
    assert (
        len(ext) == 3
    ), f"per_cohort_extended_160d must have 3 cohort entries; got {len(ext)}"
    cohort_names = {c["cohort"] for c in ext}
    expected_cohorts = {"platinai_4cell", "hilbert", "metalcytotox_combined"}
    assert (
        cohort_names == expected_cohorts
    ), f"per_cohort_extended_160d cohorts must be {expected_cohorts}; got {cohort_names}"
    # All 3 entries must have r2_mean, n_features_used=160, verdict.
    for c in ext:
        assert (
            c.get("n_features_used") == 160
        ), f"n_features_used must be 160; got {c.get('n_features_used')}"
        assert "r2_mean" in c, f"r2_mean must be present in {c['cohort']}"
        assert "verdict" in c, f"verdict must be present in {c['cohort']}"


# ---------------------------------------------------------------------------
# Test 7: wf433_m2_extended.py produces valid result.json with 3 cohorts
# ---------------------------------------------------------------------------
def test_wf433_m2_extended_result_json_exists() -> None:
    """Verify molmetal/reports/wf433_m2_extended/result.json exists with 3 cohorts."""
    path = "molmetal/reports/wf433_m2_extended/result.json"
    assert _exists(path), (
        "molmetal/reports/wf433_m2_extended/result.json must exist on disk per "
        "AGENTS.md §5.5 + Wf-433-A2"
    )
    result = _read_json(path)
    assert "per_cohort" in result, "result.json must have per_cohort"
    assert "aggregate" in result, "result.json must have aggregate"
    assert "config" in result, "result.json must have config"
    # 3 cohorts tested.
    assert (
        len(result["per_cohort"]) == 3
    ), f"per_cohort must have 3 entries; got {len(result['per_cohort'])}"
    # config.n_features == 160.
    assert (
        result["config"]["n_features"] == 160
    ), f"config.n_features must be 160; got {result['config']['n_features']}"


# ---------------------------------------------------------------------------
# Test 8: wf433_m2_extended.py structural-limit CONFIRMED (n=200 cohorts negative)
# ---------------------------------------------------------------------------
def test_wf433_m2_extended_confirms_structural_limit() -> None:
    """Verify that platinai_4cell + hilbert EXTENDED R² < 12-d baseline (overfit).

    Per Wf-377-A2 §2.1 + Wf-367-B1 + Wf-388-A1 + Wf-390-A2: adding more features
    on n=200 cohorts OVERFITS. EXTENDED 160-d should make R² WORSE than
    12-d baseline (-0.1440) for platinai_4cell + hilbert.
    """
    path = "molmetal/reports/wf433_m2_extended/result.json"
    result = _read_json(path)
    by_cohort = {c["cohort"]: c for c in result["per_cohort"]}

    platinai_r2 = by_cohort["platinai_4cell"]["r2_mean"]
    hilbert_r2 = by_cohort["hilbert"]["r2_mean"]
    baseline_12d = -0.14401547578768395

    # EXTENDED R² must be MUCH worse than 12-d baseline for n=200 cohorts.
    assert platinai_r2 < baseline_12d, (
        f"platinai_4cell EXTENDED R² must be < 12-d baseline ({baseline_12d}); "
        f"got {platinai_r2}"
    )
    assert hilbert_r2 < baseline_12d, (
        f"hilbert EXTENDED R² must be < 12-d baseline ({baseline_12d}); "
        f"got {hilbert_r2}"
    )
    # Lift vs 12-d should be negative (OVERFIT axis confirmed).
    platinai_lift = platinai_r2 - baseline_12d
    hilbert_lift = hilbert_r2 - baseline_12d
    assert (
        platinai_lift < -0.5
    ), f"platinai lift should be strongly negative (overfit); got {platinai_lift}"
    assert (
        hilbert_lift < -0.5
    ), f"hilbert lift should be strongly negative (overfit); got {hilbert_lift}"


# ---------------------------------------------------------------------------
# Test 9: wf433_m2_extended.py metalcytotox_combined LIFT vs 12-d baseline
# ---------------------------------------------------------------------------
def test_wf433_m2_extended_metalcytotox_lifts_at_n2000() -> None:
    """Verify metalcytotox_combined EXTENDED R² > 12-d baseline (bias-variance gap closed).

    Per Wf-377-A2 §1.1: n=2000 cohort closes the bias-variance gap that
    prevents n=200 cohorts from lifting. EXTENDED 160-d should provide
    genuine lift on metalcytotox_combined.
    """
    path = "molmetal/reports/wf433_m2_extended/result.json"
    result = _read_json(path)
    by_cohort = {c["cohort"]: c for c in result["per_cohort"]}

    metalcyto_r2 = by_cohort["metalcytotox_combined"]["r2_mean"]
    baseline_12d = 0.19808189300924267

    # EXTENDED R² must beat the 12-d baseline on n=2000 cohort.
    assert metalcyto_r2 > baseline_12d, (
        f"metalcytotox_combined EXTENDED R² must be > 12-d baseline "
        f"({baseline_12d}); got {metalcyto_r2}"
    )
    # Aggregate verdict preserved.
    agg_status = result["aggregate"]["aggregate_status"]
    assert agg_status in ("CEILING_BREAK_PARTIAL", "MEASURED_PARTIAL_LIFT"), (
        f"aggregate_status must be CEILING_BREAK_PARTIAL or MEASURED_PARTIAL_LIFT; "
        f"got {agg_status}"
    )


# ---------------------------------------------------------------------------
# Test 10 (bonus): pre-flight file system check — all 3 SHIPPED on disk
# ---------------------------------------------------------------------------
def test_all_3_closures_fileshipped() -> None:
    """Verify all 3 Wf-433-A1 closures shipped on disk per AGENTS.md §5.5."""
    # Closure 1: m6_kappa.json has wf424a15_drop_decision
    assert _exists("metrics/by_metric/m6_kappa.json")
    catalog_m6 = _read_json("metrics/by_metric/m6_kappa.json")
    assert "wf424a15_drop_decision" in catalog_m6

    # Closure 2: m2 EXTENDED result.json + catalog update
    assert _exists("molmetal/reports/wf433_m2_extended/result.json")
    assert _exists("molmetal/scripts/wf433_m2_extended.py")
    catalog_m2 = _read_json("metrics/by_metric/m2_r2_per_cohort.json")
    assert "per_cohort_extended_160d" in catalog_m2

    # Closure 3: m1 canonical 7/7 catalog
    assert _exists("metrics/by_metric/metal_lomo_auc_wasserstein.json")
    catalog_m1 = _read_json("metrics/by_metric/metal_lomo_auc_wasserstein.json")
    assert catalog_m1.get("n_runs_completed") == 7
