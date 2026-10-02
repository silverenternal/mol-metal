"""Wf-432-A2 m2 R² real-cohort extension tests.

CPU-only tests; no GPU; no paper/*; no wet-lab; no sudo.
Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule + §5.6 atomic write protocol.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/wf432_m2_r2_real_cohort_extension.py"


def _write_fake_cohort_csv(tmp_dir: Path, n: int = 30) -> Path:
    """Write a small synthetic cohort CSV with n smiles + random pIC50 targets."""
    import random
    rng = random.Random(42)
    p = tmp_dir / "cohort.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "target"])
        # Some realistic-looking metallo-drug SMILES with random pIC50 between 4.0 and 9.0
        smiles_pool = [
            "CC(=O)O[Pt](N)(N)Cl",
            "[Pt](N)(N)(Cl)Cl",
            "[Ru](CC)(CO)(CO)CO",
            "CN(C)C[Ir]CC(=O)O",
            "[Rh](CC)(N)N",
            "O=C[Cu]CC(=O)O",
            "[Co](N)(N)(Cl)Cl",
            "[Pd](Cl)Cl",
            "[Os](N)(N)(Cl)Cl",
            "[Fe](N)(N)(CO)CO",
        ]
        for i in range(n):
            sm = smiles_pool[i % len(smiles_pool)]
            target = round(rng.uniform(4.0, 9.0), 3)
            writer.writerow([sm, target])
    return p


def test_cli_help_exits_zero() -> None:
    """wf432_m2_r2_real_cohort_extension.py --help exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "Wf-432-A2" in result.stdout


def test_ridge_5x5_cv_returns_dict() -> None:
    """Smoke: pipeline runs end-to-end on a synthetic 30-row cohort."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cohort = _write_fake_cohort_csv(tmp, n=30)
        out_path = tmp / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--cohort-csv", str(cohort),
             "--cohort-name", "fake_cohort",
             "--out", str(out_path),
             "--n-seeds", "2",
             "--n-folds", "3",
             "--verbose"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"pipeline failed: stderr={result.stderr} stdout={result.stdout}"
        data = json.loads(out_path.read_text())
        assert "r2_mean" in data
        assert "r2_std" in data
        assert "n_splits_effective" in data
        assert data["n_splits_effective"] >= 1  # at least 1 split evaluated
        # tier should be n_lt_300 for n=30
        assert data["tier"] == "n_lt_300"
        assert data["n_features_used"] == 12
        # Allow nan (signal too weak) but the run completed
        assert isinstance(data["r2_mean"], (float, int))


def test_missing_cohort_csv_exits_nonzero() -> None:
    """Missing cohort CSV should exit non-zero."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--cohort-csv", "/tmp/nonexistent.csv",
         "--cohort-name", "fake"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0, "expected non-zero exit for missing CSV"


def test_invalid_csv_columns_exit_nonzero() -> None:
    """CSV missing 'smiles' or 'target' columns should exit non-zero."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        bad_csv = tmp / "bad.csv"
        bad_csv.write_text("foo,bar\n1,2\n")
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--cohort-csv", str(bad_csv),
             "--cohort-name", "fake"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0, "expected non-zero exit for invalid CSV"


def test_tier_assignment_promote_correct() -> None:
    """n=200 → n_lt_300 tier with n_dims=12."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cohort = _write_fake_cohort_csv(tmp, n=200)
        out_path = tmp / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--cohort-csv", str(cohort),
             "--cohort-name", "fake_200",
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, f"failed: {result.stderr}"
        data = json.loads(out_path.read_text())
        assert data["tier"] == "n_lt_300"
        assert data["n_features_used"] == 12
        assert data["alpha_used"] == 1.0


if __name__ == "__main__":
    test_cli_help_exits_zero()
    test_ridge_5x5_cv_returns_dict()
    test_missing_cohort_csv_exits_nonzero()
    test_invalid_csv_columns_exit_nonzero()
    test_tier_assignment_promote_correct()
    print("OK: 5/5 Wf-432-A2 m2 R² extension tests PASS")