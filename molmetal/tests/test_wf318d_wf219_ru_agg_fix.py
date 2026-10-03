"""Wf-318D — Wf-219 Ru aggregation fix tests.

================================================================
Wf-318D (Wf-219 follow-up): wire Ru MEASURED per-seed AUCs (Wf-310A G5
n=2/3 seed overlap) into the Wf-219 sweep_summary.json aggregation +
recompute the 5-metal aggregate honestly per party-twenty-spirit §3.5.

This file carries 8 tests:
  * 3 errors: per_metal.Ru field correctness
  * 3 errors: agg_5metal_recompute correctness
  * 2 errors: honest_framing_note preservation

Per `molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py` v1.1.0.

MEASURED anchors (Wf-310A G5 retrain, Wf-219 §3.5 honest framing):
  * Ru_seed0_auc = 0.7716 (MEASURED via Wf-310A G5 multi-task D-MPNN)
  * Ru_seed1_auc = 0.8005 (MEASURED via Wf-310A G5 multi-task D-MPNN)
  * Ru_seed2 = missing pending retry (Wf-310A pending)
  * Ru_mean_auc = 0.7860 (n=2/3) MEASURED
  * Ru_std = 0.0145 MEASURED
  * Epochs = [43, 32, 36] (original Wf-219 timeout counts, preserved for audit)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SWEEP = (
    REPO_ROOT / "molmetal" / "reports" / "wf219_krasnov_aligned" / "sweep_summary.json"
)
FIXED_SWEEP = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf318d_wf219_ru_agg"
    / "sweep_summary_ru_fixed.json"
)


@pytest.fixture(scope="module")
def original_sweep() -> dict:
    with ORIGINAL_SWEEP.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fixed_sweep() -> dict:
    assert FIXED_SWEEP.exists(), f"Run agg_fix.py first to generate {FIXED_SWEEP}"
    with FIXED_SWEEP.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Group 1: per_metal.Ru field correctness (3 tests)
# ---------------------------------------------------------------------------


def test_per_metal_Ru_mean_auc_is_g5_measured(fixed_sweep: dict) -> None:
    """Ru.mean_auc must equal Wf-310A G5 MEASURED 0.7860 (n=2/3)."""
    ru = fixed_sweep["per_metal_summary"]["Ru"]
    assert ru["mean_auc"] == pytest.approx(
        0.7860418335758276, abs=1e-6
    ), f"Ru.mean_auc must be 0.7860 (Wf-310A G5 MEASURED), got {ru['mean_auc']}"


def test_per_metal_Ru_std_and_n_seeds_ok(fixed_sweep: dict) -> None:
    """Ru.std = 0.0145 MEASURED; n_seeds_ok = 2 (n=2/3 from Wf-310A)."""
    ru = fixed_sweep["per_metal_summary"]["Ru"]
    assert ru["std"] == pytest.approx(
        0.014488075689580993, abs=1e-6
    ), f"Ru.std must be 0.0145 (Wf-310A G5 MEASURED), got {ru['std']}"
    assert ru["n_seeds_ok"] == 2, (
        f"Ru.n_seeds_ok must be 2 (Wf-310A G5 n=2/3, seed2 missing pending retry), "
        f"got {ru['n_seeds_ok']}"
    )


def test_per_metal_Ru_epochs_completed_preserved(fixed_sweep: dict) -> None:
    """Ru.epochs_completed = [43, 32, 36] preserved from original Wf-219 timeouts."""
    ru = fixed_sweep["per_metal_summary"]["Ru"]
    assert ru["epochs_completed"] == [43, 32, 36], (
        f"Ru.epochs_completed must be [43, 32, 36] (Wf-219 original timeouts, audit), "
        f"got {ru['epochs_completed']}"
    )


# ---------------------------------------------------------------------------
# Group 2: agg_5metal_recompute correctness (3 tests)
# ---------------------------------------------------------------------------


def test_agg_5metal_includes_ru_not_dropped(fixed_sweep: dict) -> None:
    """5-metal aggregate must include Ru (no longer 4-metal-only)."""
    agg = fixed_sweep["agg_mean_auc_5metal"]
    # 5-metal mean: (Re 0.8027 + Os 0.7884 + Rh 0.7525 + Ru 0.7860 + Ir 0.7586) / 5
    expected = (
        0.8026819923371648
        + 0.7884199134199135
        + 0.7524509803921569
        + 0.7860418335758276
        + 0.758592132505176
    ) / 5.0
    assert agg == pytest.approx(expected, abs=1e-4), (
        f"agg_mean_auc_5metal must equal 5-metal mean including Ru, "
        f"got {agg} (expected ~{expected:.4f})"
    )
    # Sanity: original 4-metal agg was 0.7755; with Ru=0.7860 (slightly higher than
    # 4-metal mean) the 5-metal moves UP, not down. Honest framing per §3.5.
    assert agg > 0.7755, (
        f"5-metal agg ({agg:.4f}) must be > 0.7755 (Ru MEASURED 0.7860 > 4-metal mean 0.7555-ish). "
        f"Task spec's expected 0.7477 contradicts the actual MEASURED Ru value."
    )


def test_agg_5metal_not_equals_original_4metal(
    fixed_sweep: dict, original_sweep: dict
) -> None:
    """5-metal agg must DIFFER from original 4-metal-only agg (proves Ru wired in)."""
    new_agg = fixed_sweep["agg_mean_auc_5metal"]
    old_agg = original_sweep["agg_mean_auc_5metal"]
    assert (
        new_agg != pytest.approx(old_agg, abs=1e-6)
    ), f"5-metal agg ({new_agg:.4f}) must differ from 4-metal-only old agg ({old_agg:.4f})"


def test_agg_computation_note_documents_change(fixed_sweep: dict) -> None:
    """agg_computation_note must explicitly mention Ru MEASURED + delta vs old."""
    note = fixed_sweep.get("agg_computation_note", "")
    assert (
        "Ru" in note and "MEASURED" in note
    ), f"agg_computation_note must document Ru MEASURED + n=2/3 + delta, got: {note}"
    assert (
        "0.7755" in note or "4-metal" in note
    ), f"agg_computation_note must reference old 4-metal agg (0.7755), got: {note}"


# ---------------------------------------------------------------------------
# Group 3: honest_framing_note preservation (2 tests)
# ---------------------------------------------------------------------------


def test_Ru_honest_framing_note_present(fixed_sweep: dict) -> None:
    """Ru block must carry honest_framing_note explaining partial-convergence AUCs."""
    ru = fixed_sweep["per_metal_summary"]["Ru"]
    note = ru.get("honest_framing_note", "")
    assert (
        "MEASURED" in note
    ), f"Ru.honest_framing_note must declare MEASURED status, got: {note}"
    assert (
        "43/32/36" in note or "partial-convergence" in note
    ), f"Ru.honest_framing_note must explain timeout + partial-convergence, got: {note}"


def test_top_level_verdict_updated_to_partial_5metal(fixed_sweep: dict) -> None:
    """Top-level verdict must reflect 5-metal partial (Ru MEASURED via G5 n=2/3)."""
    verdict = fixed_sweep.get("verdict", "")
    assert (
        "PARTIAL" in verdict
    ), f"verdict must be PARTIAL (honest framing, Ru still n=2/3 not full 3/3), got: {verdict}"
    assert (
        "G5" in verdict or "MEASURED" in verdict or "n=2/3" in verdict
    ), f"verdict must reference G5 MEASURED n=2/3, got: {verdict}"
    # Must NOT claim "missing" (Ru is no longer missing — it has MEASURED G5 data).
    assert (
        "missing" not in verdict.lower()
    ), f"verdict must NOT say 'missing' (Ru MEASURED via G5), got: {verdict}"
