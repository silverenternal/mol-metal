"""Test Wf-244-D: click chemistry yield R² pipeline acceptance gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PY = "/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT = "/home/hugo/codes/mol-metal/molmetal/scripts/wf244d_click_yield_r2.py"
REPORT = Path(
    "/home/hugo/codes/mol-metal/molmetal/reports/wf244d_click_yield/R2_report.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    """Re-run the script once per test module and read the canonical report."""
    subprocess.run([PY, SCRIPT], check=True, capture_output=True)
    return json.loads(REPORT.read_text())


def test_g1_csv_loaded(report: dict) -> None:
    assert report["G1_csv_loaded"] is True


def test_g2_n_entries(report: dict) -> None:
    assert report["G2_n_entries"] == 60


def test_g3_features_in_target_band(report: dict) -> None:
    n = report["G3_features_extracted"]
    assert 25 <= n <= 35, f"features={n} outside [25, 35]"


def test_g4_ridge_runs(report: dict) -> None:
    assert report["G4_ridge_runs"] is True
    assert report["G4_n_folds"] == 5


def test_g5_mean_r2_is_finite(report: dict) -> None:
    mean = report["G5_mean_r2"]
    assert isinstance(mean, float)
    assert mean == mean  # not NaN
    # Honest framing: R² may be low / negative, but must be a real number.
    assert mean > -10.0 and mean < 1.5


def test_g6_per_reaction_r2_shape(report: dict) -> None:
    per = report["G6_per_reaction_r2"]
    assert isinstance(per, dict)
    # Each reaction should have a numeric R² (or None if degenerate / N<3)
    assert len(per) >= 1
    for r, v in per.items():
        assert v is None or isinstance(v, float)


def test_g7_top_features_present(report: dict) -> None:
    top = report["G7_top_features"]
    assert isinstance(top, list)
    assert len(top) == 3
    assert all(isinstance(s, str) for s in top)
    # Features must be a subset of the namespaced names
    feats = set(report["G3_feature_names"])
    assert all(s in feats for s in top)


def test_g8_report_written(report: dict) -> None:
    assert report["G8_report_written"] is True
    assert REPORT.exists()
    assert REPORT.stat().st_size > 500


def test_honest_framing_note_present(report: dict) -> None:
    note = report["honest_framing_note"]
    assert isinstance(note, str) and len(note) > 20
