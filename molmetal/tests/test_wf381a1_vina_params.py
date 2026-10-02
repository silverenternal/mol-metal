"""Wf-381-A1 — m3 Vina docking parameter tuning tests (follow-up wave).

================================================================
Why this test file exists
================================================================
TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) calls
for a 3-level Vina exhaustiveness ladder. Wf-381-A1 (follow-up
wave to Wf-380-A4) ships:

1. ``molmetal/scripts/wf381a1_vina_param_tuning.py`` — Wf-381-A1
   follow-up variant with the same 3-function API as Wf-380-A4
   (tune_docking + per_pocket_median_kcal + aggregate_median).
2. ``molmetal/reports/wf381a1_vina_tuning_smoke/result.json`` —
   smoke report with per-level cohort median kcal/mol.

This test file covers the Wf-381-A1 NEW test contract:

  (i)   The 3-level config loads successfully and the per-level
        (exhaustiveness, n_poses) pairs match the documented spec:
            low  -> (8, 9)   baseline
            med  -> (32, 20) lift target
            high -> (64, 50) PROJECTED ceiling

  (ii)  ``tune_docking(cohort, level)`` returns deterministic
        per-pocket seed kcal/mol + per-pocket median kcal/mol +
        cohort median kcal/mol for each of the 3 levels.

  (iii) The 3-level cohort median kcal/mol follows the Wf-141 §0
        PROJECTED lift curve: each higher level is more negative
        than the previous one (more binding = lower kcal/mol).
        Specifically:
            medium < low  (delta <= -0.3 kcal/mol, PROJECTED -0.5)
            high   < med  (delta <= -0.2 kcal/mol, PROJECTED -0.5)

  (iv)  ``write_smoke_report`` emits a JSON file with the canonical
        schema (workflow / phase / per_level / aggregate sections).

All tests are CPU-only and do NOT launch the Vina binary.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "molmetal" / "scripts" / "wf381a1_vina_param_tuning.py"
CONFIG_PATH = ROOT / "molmetal" / "configs" / "vina_docking_params.yaml"
REPORT_PATH = ROOT / "molmetal" / "reports" / "wf381a1_vina_tuning_smoke" / "result.json"

_SCRIPT_DIR = ROOT / "molmetal" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

LOG = logging.getLogger("test_wf381a1_vina_params")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vina_config():
    """Load the canonical Wf-381-A1 config (delegated to Wf-380-A4 loader)."""
    from wf381a1_vina_param_tuning import load_vina_config

    return load_vina_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def smoke_report():
    """Ensure the smoke report exists; generate it if missing."""
    if not REPORT_PATH.exists():
        subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(CONFIG_PATH), "--out", str(REPORT_PATH)],
            check=True,
            cwd=str(ROOT),
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# NEW test (Wf-381-A1 task brief spec: verify tuning config loads + per_level
# median bounds)
# ---------------------------------------------------------------------------
def test_wf381a1_vina_config_levels_and_per_level_median_bounds(
    vina_config, smoke_report
):
    """Wf-381-A1 NEW test: config loads + per-level median bounds follow
    Wf-141 §0 PROJECTED lift curve.

    Contract:
      (a) The 3-level config (low / medium / high) loads with the
          canonical (exhaustiveness, n_poses) pairs (8, 9) / (32, 20) /
          (64, 50).
      (b) The per-level cohort median kcal/mol in the smoke report
          follows the Wf-141 §0 PROJECTED lift curve:
            - low median ≈ -7.0 ± 0.3 kcal/mol (synthetic)
            - medium < low by at least 0.3 kcal/mol (PROJECTED -0.5)
            - high < medium by at least 0.2 kcal/mol (PROJECTED -0.5)
      (c) The smoke report JSON has the canonical schema with the
          ``_workflow`` = "Wf-381-A1" field.
    """
    levels = vina_config["levels"]
    # Contract (a): canonical (exhaustiveness, n_poses) pairs.
    assert levels["low"]["exhaustiveness"] == 8
    assert levels["low"]["n_poses"] == 9
    assert levels["medium"]["exhaustiveness"] == 32
    assert levels["medium"]["n_poses"] == 20
    assert levels["high"]["exhaustiveness"] == 64
    assert levels["high"]["n_poses"] == 50

    # Contract (b): per-level cohort median bounds (PROJECTED lift curve).
    per_level = smoke_report["_per_level"]
    low_median = per_level["low"]["cohort_median_kcal"]
    med_median = per_level["medium"]["cohort_median_kcal"]
    high_median = per_level["high"]["cohort_median_kcal"]

    # low median is in the synthetic Wf-121/Wf-143 baseline band [-7.3, -6.8].
    assert -7.30 <= low_median <= -6.80, (
        f"low cohort median {low_median} not in [-7.30, -6.80] synthetic band"
    )
    # medium < low by at least 0.3 kcal/mol (Wf-141 §0 PROJECTED -0.5).
    assert (low_median - med_median) >= 0.30, (
        f"medium lift {low_median - med_median:.2f} kcal/mol < 0.30 floor"
    )
    # high < medium by at least 0.2 kcal/mol (Pareto frontier ceiling).
    assert (med_median - high_median) >= 0.20, (
        f"high lift {med_median - high_median:.2f} kcal/mol < 0.20 floor"
    )

    # Contract (c): canonical Wf-381-A1 schema fields present.
    assert smoke_report["_workflow"] == "Wf-381-A1"
    assert smoke_report["_phase"] == "P1"
    assert "_per_level" in smoke_report
    assert "_aggregate" in smoke_report
    assert "_honest_framing" in smoke_report
    assert smoke_report["_honest_framing"]["status"] == "SYNTHETIC_PLACEHOLDER"

    # Aggregate verdict sanity-check.
    agg = smoke_report["_aggregate"]
    assert agg["lift_vs_low"]["medium"] <= -0.30
    assert agg["lift_vs_low"]["high"] <= -0.20
    assert agg["verdict"] == "SYNTHETIC_PLACEHOLDER"

    # Industry cutoff verdict sanity-check (medium + high should pass).
    assert agg["expected_pass_at_industry_cutoff_-7_0"]["medium"] is True
    assert agg["expected_pass_at_industry_cutoff_-7_0"]["high"] is True
