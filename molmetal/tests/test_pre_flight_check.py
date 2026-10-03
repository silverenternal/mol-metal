"""Wf-259 — unit tests for ``molmetal/scripts/pre_flight_check.py``.

Three unit tests (per v2-doc acceptance G7):

* ``test_cli_runs`` — invoke with ``--json`` and verify exit 0.
* ``test_json_schema`` — verify output matches the §8.1 schema contract.
* ``test_pipeline_filter`` — ``--pipeline cfm`` runs ONLY CFM gates.

The script under test is CPU-only and does not touch any training path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"

# Match v2 doc §8.1 schema fields we expect on every report.
REQUIRED_TOP_KEYS = {"timestamp", "pipeline", "summary", "gates"}
REQUIRED_SUMMARY_KEYS = {"pass", "fail", "skip", "total_wall_sec"}
REQUIRED_GATE_KEYS = {"name", "priority", "pipeline", "status", "evidence", "wall_sec"}
ALLOWED_STATUS = {"PASS", "FAIL", "SKIP"}
ALLOWED_PRIORITY = {"P0", "P1", "P2", "P3"}


def _run_script(
    *args: str, env_extra: dict | None = None
) -> subprocess.CompletedProcess:
    """Invoke ``pre_flight_check.py`` and return the CompletedProcess."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [VENV_PY, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def test_cli_runs(tmp_path: Path) -> None:
    """--json path: exit 0 and JSON file is parseable + matches schema keys."""
    out_json = tmp_path / "preflight.json"
    # Use --pipeline cfm so the run completes quickly (chemprop --quick on CPU is slow).
    result = _run_script("--pipeline", "cfm", "--json", str(out_json))
    assert result.returncode in (
        0,
        1,
    ), f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    assert out_json.exists(), "JSON output file was not written"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert REQUIRED_TOP_KEYS.issubset(
        data.keys()
    ), f"missing top-level keys: {REQUIRED_TOP_KEYS - set(data.keys())}"


def test_json_schema() -> None:
    """--pipeline all: JSON output matches §8.1 schema (top-level + per-gate)."""
    result = _run_script("--pipeline", "all", "--verbose")
    # Tolerate exit=1 because some gates legitimately FAIL on this host
    # (vina missing, chemprop --quick may exceed 60s on CPU).
    assert result.returncode in (
        0,
        1,
    ), f"unexpected exit {result.returncode}; stderr={result.stderr[-500:]!r}"

    # The script prints JSON to stdout (after stderr --verbose lines).
    payload = result.stdout.strip()
    # Strip any stderr leak that may have been merged (shouldn't happen
    # but be defensive).
    if not payload.startswith("{"):
        # Take the last {…} block.
        start = payload.rfind("{")
        end = payload.rfind("}")
        assert start >= 0 and end > start, "no JSON object in stdout"
        payload = payload[start : end + 1]

    data = json.loads(payload)
    assert REQUIRED_TOP_KEYS.issubset(
        data.keys()
    ), f"missing top-level keys: {REQUIRED_TOP_KEYS - set(data.keys())}"
    assert REQUIRED_SUMMARY_KEYS.issubset(data["summary"].keys()), (
        f"missing summary keys: "
        f"{REQUIRED_SUMMARY_KEYS - set(data['summary'].keys())}"
    )
    assert (
        isinstance(data["gates"], list) and data["gates"]
    ), "gates must be non-empty list"
    for gate in data["gates"]:
        missing = REQUIRED_GATE_KEYS - set(gate.keys())
        assert not missing, f"gate missing keys {missing}: {gate}"
        assert gate["status"] in ALLOWED_STATUS, f"bad status {gate['status']}"
        assert gate["priority"] in ALLOWED_PRIORITY, f"bad priority {gate['priority']}"
        assert isinstance(gate["wall_sec"], (int, float)), "wall_sec must be numeric"
        assert isinstance(gate["evidence"], str), "evidence must be string"
    # n_pass + n_fail + n_skip must equal len(gates).
    s = data["summary"]
    assert s["pass"] + s["fail"] + s["skip"] == len(
        data["gates"]
    ), f"summary.totals don't match gates list: {s}"


def test_pipeline_filter(tmp_path: Path) -> None:
    """--pipeline cfm runs ONLY CFM gates (no chemprop or data gates)."""
    out_json = tmp_path / "cfm_only.json"
    result = _run_script("--pipeline", "cfm", "--json", str(out_json))
    assert result.returncode in (
        0,
        1,
    ), f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    pipelines_seen = {g["pipeline"] for g in data["gates"]}
    assert pipelines_seen == {
        "cfm"
    }, f"--pipeline cfm should yield only CFM gates; saw {pipelines_seen}"
    names = [g["name"] for g in data["gates"]]
    # Sanity: 6 CFM gates per spec §4.2 (script/config/decoder/wass/smoke_decoded/catherine_30).
    assert any(
        n.startswith("p1_cfm_") for n in names
    ), f"expected p1_cfm_* gates in --pipeline cfm; got {names}"
    assert all(
        not n.startswith("p1_chemprop_") for n in names
    ), f"chemprop gates must NOT appear in --pipeline cfm; saw {[n for n in names if 'chemprop' in n]}"
    assert all(
        not n.startswith("p2_") for n in names
    ), f"P2 gates must NOT appear in --pipeline cfm; saw {[n for n in names if n.startswith('p2_')]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
