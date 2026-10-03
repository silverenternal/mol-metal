"""Tests for the ``diagnose_metric.py`` auto-diagnostic script (D1).

These tests exercise the script as a black box: spawn it via
``subprocess`` and assert that the output matches the expected
contract documented in
``molmetal/scripts/diagnose_metric.py``.

Run with::

    cd /home/hugo/codes/mol-metal
    pytest molmetal/tests/test_diagnose_metric.py -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "diagnose_metric.py"


def _run(args, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``diagnose_metric.py`` with the given args from REPO_ROOT."""
    assert SCRIPT.is_file(), f"diagnose_metric.py missing at {SCRIPT}"
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + list(args),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Script availability / parsing smoke
# ---------------------------------------------------------------------------


def test_script_is_executable_and_lists_metrics():
    """`--list` enumerates the metric IDs covered by §4 of the mapping."""
    if not shutil.which("pytest"):
        pytest.skip("pytest is not on PATH; cannot run the inner diagnostics")
    proc = _run(["--list"])
    assert (
        proc.returncode == 0
    ), f"--list exited with {proc.returncode}; stderr: {proc.stderr}"
    assert "m14" in proc.stdout, f"--list output missing m14 entry: {proc.stdout!r}"
    assert "Metrics with failure-mode entries" in proc.stdout


def test_m14_diagnosis_prints_required_sections():
    """`--metric m14` (human output) prints the diagnostic header + per-row checks."""
    proc = _run(["--metric", "m14"])
    out = proc.stdout
    assert "Metric m14" in out, f"missing 'Metric m14' header in:\n{out}"
    assert "M15" in out, f"missing M15 mechanism check in:\n{out}"
    # The output must include a PASS / FAIL / MISS line per mechanism row.
    assert any(
        token in out for token in ("PASS", "FAIL", "MISS")
    ), f"no per-mechanism status line in output:\n{out}"


def test_m14_diagnosis_json_contract():
    """`--metric m14 --json` returns structured data with the expected keys."""
    proc = _run(["--metric", "m14", "--json"])
    assert proc.returncode in (
        0,
        1,
    ), f"--json exited with {proc.returncode}; stderr: {proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["metric"] == "m14", payload
    assert "diagnostics" in payload, payload
    assert isinstance(payload["diagnostics"], list), payload
    assert len(payload["diagnostics"]) > 0, "expected at least one mechanism entry"
    # Every diagnostic entry must have mechanism + status keys.
    for entry in payload["diagnostics"]:
        assert "mechanism" in entry, entry
        assert "status" in entry, entry
        assert entry["status"] in {"PASS", "FAIL", "MISS", "SKIP"}, entry


def test_invalid_metric_reports_error_and_exits_nonzero():
    """A bogus metric ID returns exit code 1 and a helpful stderr message."""
    proc = _run(["--metric", "m999"])
    assert (
        proc.returncode != 0
    ), f"bogus metric should exit non-zero; got rc={proc.returncode}"
    assert (
        "No failure-mode row" in proc.stderr or "m999" in proc.stderr
    ), f"missing error context in stderr: {proc.stderr!r}"


def test_malformed_metric_rejected():
    """A metric ID that does not match m\\d+ is rejected by argparse."""
    proc = _run(["--metric", "not_a_metric"])
    assert proc.returncode != 0
    # argparse exits with code 2 and prints usage to stderr.
    assert (
        "usage" in proc.stderr.lower() or "invalid" in proc.stderr.lower()
    ), proc.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
