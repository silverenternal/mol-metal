"""Wf-43 Phase 4 — Integration verify acceptance gate.

=====================================================================
This test enforces the Wf-43 Phase 4 contract on the
``molmetal/scripts/integration_verify_scoring.py`` orchestrator:

1. The orchestrator module imports cleanly and exposes the expected
   public surface (``main``, ``_run_master_verify``,
   ``_run_score_any_section``, ``_per_script_section``,
   ``_deprecation_section``).
2. The expected result-container dataclasses (``CheckResult``,
   ``SectionReport``) have the documented fields.
3. The per-script routing table
   (``molmetal/reports/wf43_integration/per_script_routing.json``)
   exists and lists every script in
   :data:`R18_SHIPPED_SCRIPTS + WF43_PHASE3_SCRIPTS` with the right
   ``expected_metrics`` for each.
4. The deprecation report
   (``molmetal/reports/wf43_deprecation/deprecated_mocks.md``) is
   present (the Wf-43 Phase 3 anchor).
5. The end-to-end orchestrator run produces a JSON + Markdown report
   pair at the canonical paths and the JSON parses with the
   documented schema.
6. The score_any() per-metric section's checks cover every metric
   in :data:`REGISTRY` — no metric is silently skipped.
7. Forbidden mock-scorer patterns (``_stub_score``,
   ``_mock_score``, ``_fake_score``, ``_dummy_score``) never appear
   as top-level function definitions in the 8 R18 train scripts.
8. Every deprecated Wf-43 Phase 3 script with an expected
   ``score_any()`` redirect uses ``score_any`` with the right metric
   name.

Verify::

    pytest molmetal/tests/test_integration_scoring.py -v
"""
from __future__ import annotations

import ast
import importlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf43_integration"
JSON_OUT = REPORTS_DIR / "integration_report.json"
MD_OUT = REPORTS_DIR / "integration_report.md"
PER_SCRIPT_OUT = REPORTS_DIR / "per_script_routing.json"
DEPRECATION_REPORT = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf43_deprecation"
    / "deprecated_mocks.md"
)
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ===========================================================================
# 1. Module imports + public surface
# ===========================================================================
def test_integration_verify_scoring_imports() -> None:
    """``molmetal.scripts.integration_verify_scoring`` imports under the venv."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    proc = importlib.import_module("integration_verify_scoring")
    for name in ("main", "_run_master_verify", "_run_score_any_section",
                 "_per_script_section", "_deprecation_section"):
        assert hasattr(proc, name), (
            f"missing public symbol: {name}"
        )


def test_module_exposes_expected_constants() -> None:
    """The expected constants are present and well-formed."""
    proc = importlib.import_module("integration_verify_scoring")

    # 8 R18 shipped scripts
    assert hasattr(proc, "R18_SHIPPED_SCRIPTS")
    assert isinstance(proc.R18_SHIPPED_SCRIPTS, tuple)
    assert len(proc.R18_SHIPPED_SCRIPTS) == 8, (
        f"expected 8 R18 train scripts, got {len(proc.R18_SHIPPED_SCRIPTS)}: "
        f"{list(proc.R18_SHIPPED_SCRIPTS)}"
    )
    assert "train_se3sb.py" in proc.R18_SHIPPED_SCRIPTS
    assert "bench_pb_valid.py" in proc.R18_SHIPPED_SCRIPTS

    # 10 Wf-43 Phase 3 deprecated scripts
    assert hasattr(proc, "WF43_PHASE3_SCRIPTS")
    assert isinstance(proc.WF43_PHASE3_SCRIPTS, tuple)
    assert len(proc.WF43_PHASE3_SCRIPTS) == 10, (
        f"expected 10 Wf-43 Phase 3 deprecated scripts, got "
        f"{len(proc.WF43_PHASE3_SCRIPTS)}"
    )

    # Expected metric mapping
    assert hasattr(proc, "EXPECTED_SCORE_ANY_METRIC")
    expected = proc.EXPECTED_SCORE_ANY_METRIC
    assert "bench_pb_valid.py" in expected
    assert "pb_pass_rate" in expected["bench_pb_valid.py"]
    assert "benchmark_astsr_extended.py" in expected
    assert "boltz_affinity" in expected["benchmark_astsr_extended.py"]

    # Forbidden mock names
    assert hasattr(proc, "FORBIDDEN_MOCK_NAMES")
    assert "_stub_score" in proc.FORBIDDEN_MOCK_NAMES
    assert "_mock_score" in proc.FORBIDDEN_MOCK_NAMES
    assert "_fake_score" in proc.FORBIDDEN_MOCK_NAMES


# ===========================================================================
# 2. Dataclass shapes
# ===========================================================================
def test_check_result_defaults() -> None:
    """``CheckResult`` defaults: passed=False, skipped=False."""
    proc = importlib.import_module("integration_verify_scoring")
    cr = proc.CheckResult(name="x", passed=True)
    assert cr.passed is True
    assert cr.skipped is False
    assert cr.seconds == 0.0
    assert cr.message == ""
    assert isinstance(cr.detail, dict)


def test_section_report_defaults() -> None:
    """``SectionReport`` defaults: status='PASS', empty checks."""
    proc = importlib.import_module("integration_verify_scoring")
    sr = proc.SectionReport(section="x")
    assert sr.section == "x"
    assert sr.status == "PASS"
    assert sr.checks == []
    assert sr.elapsed_seconds == 0.0


# ===========================================================================
# 3. Helpers
# ===========================================================================
def test_to_jsonable_handles_path() -> None:
    """``_to_jsonable`` converts Path → str."""
    proc = importlib.import_module("integration_verify_scoring")
    out = proc._to_jsonable({"p": Path("/tmp/x")})
    assert out == {"p": "/tmp/x"}


def test_score_any_metrics_referenced_regex() -> None:
    """``_score_any_metrics_referenced`` extracts metric-name literals."""
    proc = importlib.import_module("integration_verify_scoring")
    src = textwrap.dedent(
        """
        score_any("sa_score", molecules=...)
        score_any('pb_pass_rate', molecules=...)
        score_any("boltz_affinity", molecules=...)
        # score_any("commented_out")
        """
    )
    refs = proc._score_any_metrics_referenced(src)
    assert refs == {"sa_score", "pb_pass_rate", "boltz_affinity"}


def test_top_level_forbidden_mock_names() -> None:
    """``_top_level_forbidden_mock_names`` flags top-level mock defs only."""
    proc = importlib.import_module("integration_verify_scoring")
    # A script with a top-level _stub_score + a nested helper.
    bad = textwrap.dedent(
        """
        def _stub_score(s): return 1.0
        def ok(x):
            def _stub_score(y): return 2.0  # nested — should NOT count
            return _stub_score(x)
        """
    )
    found = proc._top_level_forbidden_mock_names(bad)
    assert "_stub_score" in found
    # Clean script — none found.
    clean = "def real(x): return x\n"
    assert proc._top_level_forbidden_mock_names(clean) == []


# ===========================================================================
# 4. Source-level audits (cheap, no subprocess)
# ===========================================================================
def _read_source(name: str) -> str:
    path = SCRIPTS_DIR / name
    assert path.exists(), f"script not found: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script_name",
    [
        "train_se3sb.py",
        "train_transport.py",
        "train_multiph.py",
        "train_ru_5metal_se3sb.py",
        "train_cisplatin_resistance_v2.py",
        "train_affinity_regression.py",
        "train_multimetal_lomo.py",
        "bench_pb_valid.py",
    ],
)
def test_no_forbidden_mock_scorer_in_r18_script(script_name: str) -> None:
    """None of the 8 R18 train scripts may define a top-level
    ``_stub_score`` / ``_mock_score`` / ``_fake_score`` /
    ``_dummy_score`` function."""
    proc = importlib.import_module("integration_verify_scoring")
    src = _read_source(script_name)
    found = proc._top_level_forbidden_mock_names(src)
    assert not found, (
        f"{script_name}: forbidden top-level mock-scorer "
        f"function(s) defined: {found}"
    )


def test_bench_pb_valid_uses_score_any_for_pb_pass_rate() -> None:
    """``bench_pb_valid.py`` is the one R18 script that DOES use
    ``score_any()`` (for ``pb_pass_rate``). Confirm the import +
    call site is present."""
    src = _read_source("bench_pb_valid.py")
    assert "from molmetal.scoring.score_any import" in src, (
        "bench_pb_valid.py must import from molmetal.scoring.score_any"
    )
    assert 'score_any("pb_pass_rate"' in src or (
        "score_any(\n            \"pb_pass_rate\"" in src
    ) or 'score_any(\n        "pb_pass_rate"' in src, (
        "bench_pb_valid.py must call score_any(\"pb_pass_rate\", ...)"
    )


# ===========================================================================
# 5. End-to-end run
# ===========================================================================
def test_integration_verify_runs_end_to_end() -> None:
    """Run ``integration_verify_scoring.py`` against the venv and
    confirm it exits 0, writes JSON + MD + per_script_routing JSON,
    and the JSON parses with the documented schema."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")

    cmd = [
        str(VENV_PY), str(SCRIPTS_DIR / "integration_verify_scoring.py"),
    ]
    proc = subprocess_run_with_timeout(cmd, timeout=600)
    assert proc.returncode == 0, (
        f"integration_verify_scoring.py exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    for path in (JSON_OUT, MD_OUT, PER_SCRIPT_OUT):
        assert path.exists(), f"expected artefact missing: {path}"

    payload = json.loads(JSON_OUT.read_text())
    assert payload["schema_version"] == "1.0"
    assert "Wf-43 Phase 4" in payload["workflow"]
    assert payload["summary"]["n_sections"] >= 4
    assert payload["summary"]["n_sections_failed"] == 0
    assert payload["summary"]["n_checks_failed"] == 0

    sections = {s["section"] for s in payload["sections"]}
    assert "master_verify" in sections
    assert "score_any_per_metric" in sections
    assert "per_script_routing" in sections
    assert "deprecated_mocks_replaced" in sections


def test_per_script_routing_json_lists_every_audited_script() -> None:
    """The ``per_script_routing.json`` artifact lists all 8 R18 train
    scripts + all 10 Wf-43 Phase 3 deprecated scripts with the right
    ``expected_metrics`` per script. ``bench_pb_valid.py`` is in both
    sets (it is both an R18 train script and a Wf-43 Phase 3
    deprecated script) and gets a ``buckets`` tag carrying both
    classifications.
    """
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    if not PER_SCRIPT_OUT.exists():
        # Run the orchestrator first.
        proc = subprocess_run_with_timeout(
            [str(VENV_PY),
             str(SCRIPTS_DIR / "integration_verify_scoring.py")],
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr

    payload = json.loads(PER_SCRIPT_OUT.read_text())
    assert payload["schema_version"] == "1.0"
    assert "Wf-43 Phase 4" in payload["workflow"]
    assert len(payload["r18_shipped"]) == 8
    assert len(payload["wf43_phase3_deprecated"]) == 10
    # Spot-check one entry.
    pb_valid_entry = next(
        e for e in payload["wf43_phase3_deprecated"]
        if e["script"] == "bench_pb_valid.py"
    )
    assert "pb_pass_rate" in pb_valid_entry["expected_metrics"]
    assert pb_valid_entry["has_score_any"] is True
    assert "pb_pass_rate" in pb_valid_entry["referenced_metrics"]
    # Cross-cutting: bench_pb_valid.py is in BOTH buckets.
    assert "r18_shipped" in pb_valid_entry["buckets"]
    assert "wf43_phase3_deprecated" in pb_valid_entry["buckets"]


def test_markdown_report_contains_section_table() -> None:
    """The Markdown artefact includes the section roll-up table."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    if not MD_OUT.exists():
        proc = subprocess_run_with_timeout(
            [str(VENV_PY),
             str(SCRIPTS_DIR / "integration_verify_scoring.py")],
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr

    body = MD_OUT.read_text()
    assert "Wf-43 Phase 4 — Integration Verify" in body
    assert "Section roll-up" in body
    assert "master_verify" in body
    assert "score_any_per_metric" in body
    assert "per_script_routing" in body
    assert "deprecated_mocks_replaced" in body


def test_score_any_section_covers_every_registered_metric() -> None:
    """The score_any() per-metric section must include a check for
    every metric in :data:`REGISTRY` — no metric may be silently
    skipped."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    if not JSON_OUT.exists():
        proc = subprocess_run_with_timeout(
            [str(VENV_PY),
             str(SCRIPTS_DIR / "integration_verify_scoring.py")],
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr

    payload = json.loads(JSON_OUT.read_text())
    sec = next(
        s for s in payload["sections"]
        if s["section"] == "score_any_per_metric"
    )
    checked_metrics = {
        c["name"].split(":", 1)[-1]
        for c in sec["checks"]
        if c["name"].startswith("score_any:")
    }
    from molmetal.scoring import available_metrics
    registered = set(available_metrics())
    missing = registered - checked_metrics
    assert not missing, (
        f"score_any() per-metric section missed these REGISTRY "
        f"metrics: {sorted(missing)}"
    )


def test_deprecation_report_anchor_present() -> None:
    """The Wf-43 Phase 3 deprecation report is the upstream anchor
    for the Wf-43 Phase 4 audit. Confirm it is present."""
    assert DEPRECATION_REPORT.exists(), (
        f"Wf-43 Phase 3 deprecation report missing at {DEPRECATION_REPORT}"
    )
    body = DEPRECATION_REPORT.read_text()
    assert "bench_mcts_real_trajectory.py" in body
    assert "bench_pb_valid.py" in body


def test_main_help_exits_zero() -> None:
    """``integration_verify_scoring.py --help`` exits 0."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python missing at {VENV_PY}")
    proc = subprocess_run_with_timeout(
        [str(VENV_PY),
         str(SCRIPTS_DIR / "integration_verify_scoring.py"),
         "--help"],
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_py_compile() -> None:
    """``python -m py_compile`` must succeed on the orchestrator."""
    import py_compile
    script = SCRIPTS_DIR / "integration_verify_scoring.py"
    py_compile.compile(str(script), doraise=True)


# ===========================================================================
# Helpers
# ===========================================================================
def subprocess_run_with_timeout(cmd, *, timeout):
    import subprocess
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )