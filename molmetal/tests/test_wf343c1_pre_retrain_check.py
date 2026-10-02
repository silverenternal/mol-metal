"""Wf-343c-I1 — unit tests for ``molmetal/scripts/wf343c1_pre_retrain_check.py``.

Eight unit tests (per v1-doc acceptance):

* ``test_cli_runs`` — invoke with ``--json`` and verify exit code.
* ``test_json_schema`` — output matches the schema contract (top-level +
  per-gate keys; status in {PASS, FAIL, SKIP}; priority in {P1, P2}).
* ``test_pipeline_filter`` — ``--pipeline code`` runs ONLY code gates.
* ``test_summary_counts_match_gates`` — pass+fail+skip == len(gates).
* ``test_gate_count_consistency`` — total expected gate counts match the
  documented EXPECTED_GATE_COUNTS dict (the audit surface for
  per-pipeline gate-count drift).
* ``test_post_retrain_skip_is_honest_negative`` — §5.11-5.14 gates and
  BindingDB gate return SKIP (not FAIL) per honest framing.
* ``test_parent_doc_reference_present`` — JSON ``parent_doc_reference``
  field points to the parent checklist doc.
* ``test_gate_result_helpers_unit`` — direct unit tests of the
  ``GateResult`` dataclass + ``_grep_file`` / ``_line_count`` helpers
  (no subprocess; pure-Python).

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
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf343c1_pre_retrain_check.py"
VENV_PY = "/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"

# Match the v1 doc schema fields we expect on every report.
REQUIRED_TOP_KEYS = {
    "timestamp",
    "schema_version",
    "pipeline",
    "summary",
    "gates",
    "parent_doc_reference",
}
REQUIRED_SUMMARY_KEYS = {
    "pass",
    "fail",
    "skip",
    "total",
    "total_wall_sec",
}
REQUIRED_GATE_KEYS = {
    "name",
    "priority",
    "pipeline",
    "status",
    "evidence",
    "wall_sec",
}
ALLOWED_STATUS = {"PASS", "FAIL", "SKIP"}
ALLOWED_PRIORITY = {"P0", "P1", "P2", "P3"}


def _run_script(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke ``wf343c1_pre_retrain_check.py`` and return the CompletedProcess."""
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


# ---------------------------------------------------------------------------
# Test 1 — CLI runs and produces JSON file
# ---------------------------------------------------------------------------


def test_cli_runs(tmp_path: Path) -> None:
    """--json path: exit 0/1 and JSON file is parseable + matches schema keys."""
    out_json = tmp_path / "pre_retrain.json"
    # Use --pipeline code so the run completes quickly (no subprocess help).
    result = _run_script("--pipeline", "code", "--json", str(out_json))
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    assert out_json.exists(), "JSON output file was not written"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert REQUIRED_TOP_KEYS.issubset(data.keys()), (
        f"missing top-level keys: {REQUIRED_TOP_KEYS - set(data.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 2 — JSON schema (top-level + per-gate keys)
# ---------------------------------------------------------------------------


def test_json_schema() -> None:
    """--pipeline all: JSON output matches the v1 schema (top-level + per-gate)."""
    result = _run_script("--pipeline", "all")
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr[-500:]!r}"
    )

    payload = result.stdout.strip()
    if not payload.startswith("{"):
        # Defensive: take the last {…} block (in case stderr leaked).
        start = payload.rfind("{")
        end = payload.rfind("}")
        assert start >= 0 and end > start, "no JSON object in stdout"
        payload = payload[start : end + 1]

    data = json.loads(payload)
    assert REQUIRED_TOP_KEYS.issubset(data.keys()), (
        f"missing top-level keys: {REQUIRED_TOP_KEYS - set(data.keys())}"
    )
    assert REQUIRED_SUMMARY_KEYS.issubset(data["summary"].keys()), (
        f"missing summary keys: "
        f"{REQUIRED_SUMMARY_KEYS - set(data['summary'].keys())}"
    )
    assert isinstance(data["gates"], list) and data["gates"], (
        "gates must be non-empty list"
    )
    for gate in data["gates"]:
        missing = REQUIRED_GATE_KEYS - set(gate.keys())
        assert not missing, f"gate missing keys {missing}: {gate}"
        assert gate["status"] in ALLOWED_STATUS, (
            f"bad status {gate['status']}"
        )
        assert gate["priority"] in ALLOWED_PRIORITY, (
            f"bad priority {gate['priority']}"
        )
        assert isinstance(gate["wall_sec"], (int, float)), (
            "wall_sec must be numeric"
        )
        assert isinstance(gate["evidence"], str), "evidence must be string"
    # n_pass + n_fail + n_skip must equal len(gates).
    s = data["summary"]
    assert s["pass"] + s["fail"] + s["skip"] == len(data["gates"]), (
        f"summary.totals don't match gates list: {s}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Pipeline filter (--pipeline code runs only code gates)
# ---------------------------------------------------------------------------


def test_pipeline_filter(tmp_path: Path) -> None:
    """--pipeline code runs ONLY code gates (no data / ablation / etc.)."""
    out_json = tmp_path / "code_only.json"
    result = _run_script("--pipeline", "code", "--json", str(out_json))
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr!r}"
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    pipelines_seen = {g["pipeline"] for g in data["gates"]}
    assert pipelines_seen == {"code"}, (
        f"--pipeline code should yield only code gates; saw {pipelines_seen}"
    )
    names = [g["name"] for g in data["gates"]]
    # Sanity: code pipeline emits 10 §1.1.x gates per spec.
    assert any(n.startswith("code_1_1_") for n in names), (
        f"expected code_1_1_* gates in --pipeline code; got {names}"
    )
    assert all(not n.startswith("data_2_1_") for n in names), (
        f"data gates must NOT appear in --pipeline code; "
        f"saw {[n for n in names if n.startswith('data_')]}"
    )
    assert all(not n.startswith("ablation_") for n in names), (
        f"ablation gates must NOT appear in --pipeline code; "
        f"saw {[n for n in names if n.startswith('ablation_')]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Summary counts match gates list
# ---------------------------------------------------------------------------


def test_summary_counts_match_gates() -> None:
    """pass+fail+skip == len(gates) invariant holds for the --pipeline all run."""
    result = _run_script("--pipeline", "all")
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr[-500:]!r}"
    )
    payload = result.stdout.strip()
    if not payload.startswith("{"):
        start = payload.rfind("{")
        end = payload.rfind("}")
        payload = payload[start : end + 1]
    data = json.loads(payload)
    s = data["summary"]
    n = len(data["gates"])
    assert s["pass"] + s["fail"] + s["skip"] == n, (
        f"summary mismatch: pass={s['pass']} fail={s['fail']} "
        f"skip={s['skip']} sum={s['pass'] + s['fail'] + s['skip']} "
        f"vs gates={n}"
    )
    assert s["total"] == n, f"summary.total ({s['total']}) != gates ({n})"


# ---------------------------------------------------------------------------
# Test 5 — Gate-count consistency (per-pipeline documented EXPECTED_GATE_COUNTS)
# ---------------------------------------------------------------------------


def test_gate_count_consistency() -> None:
    """The total expected gate count (across all 6 sub-pipelines) must match
    the documented EXPECTED_GATE_COUNTS so per-pipeline drift is auditable.

    We import the script as a module via ``python -c`` to read the
    EXPECTED_GATE_COUNTS dict without invoking the full CLI (which would
    take ~30s on --pipeline all).
    """
    result = subprocess.run(
        [
            VENV_PY, "-c",
            "import sys; sys.path.insert(0, '/home/hugo/codes/mol-metal');"
            "from molmetal.scripts.wf343c1_pre_retrain_check import "
            "EXPECTED_GATE_COUNTS; "
            "import json; print(json.dumps(EXPECTED_GATE_COUNTS))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"could not import EXPECTED_GATE_COUNTS: {result.stderr!r}"
    )
    counts = json.loads(result.stdout.strip())
    expected_total = sum(
        v for v in counts.values() if isinstance(v, int)
    )
    assert expected_total == 26, (
        f"expected total gate count across all 6 pipelines = 26, "
        f"got {expected_total} (per-pipeline: {counts})"
    )
    # Per-pipeline spot-checks (catch silent drift).
    assert counts["code_level"] == 10
    assert counts["ablation_flags"] == 3
    assert counts["data_level"] == 7
    assert counts["test_level"] == 1
    assert counts["post_retrain"] == 4
    assert counts["honest_framing"] == 1


# ---------------------------------------------------------------------------
# Test 6 — Post-retrain gates and BindingDB return SKIP (honest-negative)
# ---------------------------------------------------------------------------


def test_post_retrain_skip_is_honest_negative() -> None:
    """§5.11-5.14 + BindingDB gates return SKIP (not FAIL) per honest framing.

    Per the parent doc §5 sign-off convention, post-retrain gates are
    filled in after the retrain wave completes; reporting them as SKIP
    (not FAIL) preserves the honest framing.  The BindingDB gate is a
    SKIP too: per §2.1.6, BindingDB is external and NOT used in retrain
    (data fact, not blocker).
    """
    result = _run_script("--pipeline", "all")
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr[-500:]!r}"
    )
    payload = result.stdout.strip()
    if not payload.startswith("{"):
        start = payload.rfind("{")
        end = payload.rfind("}")
        payload = payload[start : end + 1]
    data = json.loads(payload)

    by_name = {g["name"]: g for g in data["gates"]}

    # All 4 post-retrain §5.11-5.14 gates must be SKIP.
    for gname in (
        "post_retrain_5_11_per_seed_std",
        "post_retrain_5_12_os_auc_non_degenerate",
        "post_retrain_5_13_ru_w2_epoch_avg",
        "post_retrain_5_14_no_nan_auc",
    ):
        assert gname in by_name, f"{gname} not in gates: {list(by_name)}"
        assert by_name[gname]["status"] == "SKIP", (
            f"{gname} must be SKIP (post-retrain), got "
            f"{by_name[gname]['status']}"
        )

    # BindingDB §2.1.6 gate must also be SKIP (honest-negative data fact).
    bindingdb_gate = "data_2_1_6_bindingdb_external_honest_negative"
    assert bindingdb_gate in by_name, (
        f"{bindingdb_gate} not in gates: {list(by_name)}"
    )
    assert by_name[bindingdb_gate]["status"] == "SKIP", (
        f"{bindingdb_gate} must be SKIP (data fact, not blocker), "
        f"got {by_name[bindingdb_gate]['status']}"
    )


# ---------------------------------------------------------------------------
# Test 7 — JSON parent_doc_reference points to the parent checklist doc
# ---------------------------------------------------------------------------


def test_parent_doc_reference_present() -> None:
    """JSON ``parent_doc_reference`` field must point to the parent doc."""
    result = _run_script("--pipeline", "honest_framing")
    assert result.returncode in (0, 1), (
        f"unexpected exit {result.returncode}; stderr={result.stderr[-500:]!r}"
    )
    payload = result.stdout.strip()
    if not payload.startswith("{"):
        start = payload.rfind("{")
        end = payload.rfind("}")
        payload = payload[start : end + 1]
    data = json.loads(payload)
    assert "parent_doc_reference" in data, (
        "JSON missing parent_doc_reference field"
    )
    ref = data["parent_doc_reference"]
    # Reference must include the canonical parent doc filename.
    assert "pre_retrain_checklist" in ref, (
        f"parent_doc_reference should name the parent doc; got {ref!r}"
    )
    # Reference must include the gate-family names so reviewers can
    # trace each gate back to its §-level section.
    for family in ("§1.1", "§1.3", "§2.1", "§3.1", "§5.11", "§8.15"):
        assert family in ref, (
            f"parent_doc_reference missing family marker {family}; got {ref!r}"
        )


# ---------------------------------------------------------------------------
# Test 8 — Direct unit tests of GateResult + helpers (no subprocess)
# ---------------------------------------------------------------------------


def test_gate_result_helpers_unit() -> None:
    """Direct unit tests of the ``GateResult`` dataclass + helpers.

    Catches regressions in the dataclass shape, the ``to_dict``
    serialisation, and the ``_grep_file`` / ``_line_count`` helpers
    without invoking the full CLI.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from molmetal.scripts.wf343c1_pre_retrain_check import (
        GateResult,
        _grep_file,
        _line_count,
        gate_grep,
        gate_file_exists,
        gate_min_lines,
    )

    # GateResult shape + to_dict keys.
    gr = GateResult(
        name="unit_test_gate",
        priority="P1",
        pipeline="unit",
        status="PASS",
        evidence="unit evidence",
        wall_sec=0.123,
    )
    d = gr.to_dict()
    assert d == {
        "name": "unit_test_gate",
        "priority": "P1",
        "pipeline": "unit",
        "status": "PASS",
        "evidence": "unit evidence",
        "wall_sec": 0.123,
    }

    # gate_grep PASS path on a known pattern in train_chemprop_v2_v3.py.
    chemprop_path = (
        REPO_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"
    )
    res_pass = gate_grep(
        "unit_grep_pass", "P1", "unit",
        "indices = train_idx.copy()", chemprop_path,
    )
    assert res_pass.status == "PASS"
    assert res_pass.wall_sec >= 0.0

    # gate_grep FAIL path on a non-existent pattern.
    res_fail = gate_grep(
        "unit_grep_fail", "P1", "unit",
        "definitely_not_present_zzz_xyz", chemprop_path,
    )
    assert res_fail.status == "FAIL"

    # _grep_file returns False for missing files.
    assert _grep_file("anything", Path("/nonexistent/path/xyz")) is False

    # _line_count returns -1 for missing files.
    assert _line_count(Path("/nonexistent/path/xyz")) == -1
    # _line_count on the actual chemprop driver is positive.
    assert _line_count(chemprop_path) > 2000

    # gate_file_exists PASS on a known existing file.
    res_fe_pass = gate_file_exists(
        "unit_fe_pass", "P1", "unit", chemprop_path,
    )
    assert res_fe_pass.status == "PASS"
    # gate_file_exists FAIL on a missing file.
    res_fe_fail = gate_file_exists(
        "unit_fe_fail", "P1", "unit", Path("/nonexistent/zzz"),
    )
    assert res_fe_fail.status == "FAIL"

    # gate_min_lines PASS / FAIL paths.
    res_ml_pass = gate_min_lines(
        "unit_ml_pass", "P1", "unit", chemprop_path, min_lines=100,
    )
    assert res_ml_pass.status == "PASS"
    res_ml_fail = gate_min_lines(
        "unit_ml_fail", "P1", "unit", chemprop_path, min_lines=10**9,
    )
    assert res_ml_fail.status == "FAIL"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
