"""STR-1 tests — the seed ensemble must stack the right rows and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molmetal.scripts.msrs_seed_ensemble import (
    _diff,
    _load_oof,
    _reference,
    _summary,
)


def _write_oof(path: Path, pred_offset: float = 0.0, y_shift: float = 0.0) -> None:
    n = 6
    payload = {
        "y": np.linspace(0.0, 1.0, n) + y_shift,
        "doi": np.array([f"10.1/{i}" for i in range(n)]),
        "metal": np.array(["Ru", "Ir", "Ru", "Ir", "Ru", "Ir"]),
        "cell_line_key": np.array(["A549"] * n),
        "ligand_canonical": np.array([f"lig{i}" for i in range(n)]),
        "time_h": np.array([24.0] * n),
    }
    for protocol in ("ligand", "doi", "scaffold", "year"):
        payload[f"{protocol}__fold"] = np.arange(n) % 5
        payload[f"{protocol}__ligand+both"] = np.linspace(0.0, 1.0, n) + pred_offset
    np.savez_compressed(path, **payload)


def test_load_oof_stacks_predictions_per_seed(tmp_path: Path):
    _write_oof(tmp_path / "oof_seed0.npz", pred_offset=0.0)
    _write_oof(tmp_path / "oof_seed1.npz", pred_offset=0.2)
    data = _load_oof(sorted(tmp_path.glob("oof_seed*.npz")))
    assert data["ligand__preds"].shape == (6, 2)
    assert data["seeds"] == ["0", "1"]
    assert data["ligand__preds"][0, 1] - data["ligand__preds"][0, 0] == pytest.approx(0.2)


def test_load_oof_rejects_a_different_label_vector(tmp_path: Path):
    _write_oof(tmp_path / "oof_seed0.npz")
    _write_oof(tmp_path / "oof_seed1.npz", y_shift=1.0)
    with pytest.raises(SystemExit):
        _load_oof(sorted(tmp_path.glob("oof_seed*.npz")))


def test_load_oof_rejects_a_different_fold_vector(tmp_path: Path):
    _write_oof(tmp_path / "oof_seed0.npz")
    # Rewrite seed 1 with shifted folds.
    payload = dict(np.load(tmp_path / "oof_seed0.npz"))
    payload["doi__fold"] = (payload["doi__fold"] + 1) % 5
    np.savez_compressed(tmp_path / "oof_seed1.npz", **payload)
    with pytest.raises(SystemExit):
        _load_oof(sorted(tmp_path.glob("oof_seed*.npz")))


def test_load_oof_requires_at_least_one_file(tmp_path: Path):
    with pytest.raises(SystemExit):
        _load_oof([])


def test_summary_reports_mean_sd_min_max():
    rows = [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]
    out = _summary(rows, ["x"])["x"]
    assert out["mean"] == 3.0
    assert out["sd"] == pytest.approx(2.0)
    assert (out["min"], out["max"]) == (1.0, 5.0)


def test_summary_skips_missing_values():
    rows = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
    out = _summary(rows, ["x"])["x"]
    assert out["mean"] == 2.0


def test_diff_returns_none_when_either_side_is_missing():
    assert _diff(None, 1.0) is None
    assert _diff(1.0, None) is None
    assert _diff(1.25, 1.0) == 0.25


def _write_metric(dirpath: Path, name: str, payload: dict) -> None:
    (dirpath / name).write_text(json.dumps(payload))


def test_reference_reads_the_published_single_seed_numbers(tmp_path: Path):
    protocols = {p: {} for p in ("ligand", "doi", "scaffold", "year")}
    gen = {"splits": {p: {"arms": {"ligand+both": {"r2": 0.5}}} for p in protocols}}
    delta = {
        "protocols": {
            p: {"thresholds": {"0.2091": {"sign_accuracy": 0.6, "mae_delta": 0.5}}}
            for p in protocols
        }
    }
    cal = {
        "protocols": {
            p: {"variants": {"global": {"coverage": 0.9, "median_width": 2.0}}}
            for p in protocols
        }
    }
    dec = {
        "protocols": {
            p: {"full_coverage": {"model": {"mean_regret": 0.11}}} for p in protocols
        }
    }
    _write_metric(tmp_path, "msrs_generalization.json", gen)
    _write_metric(tmp_path, "msrs_delta_metrics.json", delta)
    _write_metric(tmp_path, "msrs_calibration.json", cal)
    _write_metric(tmp_path, "msrs_decision_curve.json", dec)

    ref = _reference(tmp_path)
    assert ref["doi"]["r2"] == 0.5
    assert ref["doi"]["sign_at_gate"] == 0.6
    assert ref["doi"]["calibration_median_width"] == 2.0
    assert ref["doi"]["decision_regret_full"] == 0.11


def test_reference_reports_missing_artifacts_instead_of_crashing(tmp_path: Path):
    assert _reference(tmp_path)["status"] == "MISSING"
