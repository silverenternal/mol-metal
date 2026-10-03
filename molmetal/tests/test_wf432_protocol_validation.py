"""Wf-432-A1 Protocol Validation tests.

Standalone CPU-only tests; no GPU; no paper/*; no wet-lab; no sudo.
Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule + §5.6 atomic write protocol.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/wf432_protocol_validation.py"


def _write_trainer_results(tmp_dir: Path, value: float, value_field: str = "value") -> Path:
    """Write a fake trainer-results JSON for CLI test."""
    p = tmp_dir / "trainer.json"
    p.write_text(json.dumps({value_field: value, "metric": "fake_metric"}))
    return p


def test_cli_help_exits_zero() -> None:
    """wf432_protocol_validation.py --help exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "Wf-432-A1" in result.stdout


def test_list_baselines_returns_3() -> None:
    """list-baselines subcommand returns 3 baseline templates."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "list-baselines"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"list-baselines failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert set(data.keys()) == {"Krasnov", "Rusanov", "DrugOOD"}


def test_validate_krasnov_pass() -> None:
    """Validate against Krasnov baseline: value=0.83 should PASS."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = _write_trainer_results(tmp, 0.83)
        out_path = tmp / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "validate",
             "--trainer-results", str(trainer),
             "--baseline", "Krasnov",
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out_path.read_text())
        assert data["verdict"] == "PASS", f"expected PASS, got {data['verdict']}"
        assert abs(data["gap"] - 0.02) < 1e-9


def test_validate_krasnov_fail() -> None:
    """Validate against Krasnov baseline: value=0.50 should FAIL."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = _write_trainer_results(tmp, 0.50)
        out_path = tmp / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "validate",
             "--trainer-results", str(trainer),
             "--baseline", "Krasnov",
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out_path.read_text())
        assert data["verdict"] == "FAIL", f"expected FAIL, got {data['verdict']}"


def test_validate_rusanov_per_cohort() -> None:
    """Validate per-cohort dict against Rusanov baseline."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer_path = tmp / "trainer.json"
        trainer_path.write_text(json.dumps({
            "per_cohort": {"mcF7": 0.90, "a549": 0.85, "a2780": 0.80}
        }))
        out_path = tmp / "out.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "validate",
             "--trainer-results", str(trainer_path),
             "--baseline", "Rusanov",
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"validate failed: {result.stderr}"
        data = json.loads(out_path.read_text())
        # per-cohort verdicts live in verdict_by_cohort, not verdict
        assert data["verdict_by_cohort"]["mcF7"] == "PASS"
        assert data["verdict_by_cohort"]["a549"] == "PASS"
        assert data["verdict_by_cohort"]["a2780"] == "FAIL"


def test_validate_unknown_baseline_exits_nonzero() -> None:
    """validate --baseline with unknown value should fail."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        trainer = _write_trainer_results(tmp, 0.83)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "validate",
             "--trainer-results", str(trainer),
             "--baseline", "Unknown2027",
             "--out", str(tmp / "out.json")],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0, "expected non-zero exit for unknown baseline"


if __name__ == "__main__":
    test_cli_help_exits_zero()
    test_list_baselines_returns_3()
    test_validate_krasnov_pass()
    test_validate_krasnov_fail()
    test_validate_rusanov_per_cohort()
    test_validate_unknown_baseline_exits_nonzero()
    print("OK: 6/6 Wf-432-A1 protocol validation tests PASS")
