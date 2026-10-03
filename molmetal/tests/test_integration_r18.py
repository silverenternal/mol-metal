"""Tests for ``molmetal.scripts.integration_verify_r18``.

=====================================================================
Wf-46 Phase 4 — acceptance gate for the 5-test integration verify
that exercises the live wiring between the shipped R18 modules
(Wf-30 + Wf-36 + Wf-37 + Wf-39 + Wf-40 + Wf-41 + Wf-43).

The verify orchestrator lives in
:mod:`molmetal.scripts.integration_verify_r18` and writes its JSON
report to
``molmetal/reports/wf46_r18_verification/integration_report.json``.

This test file enforces four guarantees:

A. **Module surface** — the orchestrator imports cleanly and
   exposes the documented public API (``main``, ``TESTS``, the
   five ``_test_tX`` driver functions, the result dataclasses).
B. **Five tests in the registry** — exactly the five expected test
   IDs ``T1`` through ``T5`` are present, and each maps to a
   callable with the right docstring.
C. **End-to-end run produces the report artefacts** — running
   :func:`main` writes both ``integration_report.json`` and
   ``integration_report.md`` to the canonical path, and the JSON
   parses against the documented schema.
D. **Each individual test driver** — calling each ``_test_tX``
   function returns a :class:`TestReport` whose ``checks`` list
   is non-empty and whose status is one of ``PASS`` / ``FAIL`` /
   ``SKIPPED``.

The tests are CPU-only and finish in <60 s.

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_integration_r18.py -v
"""
from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "molmetal" / "tests"
REPORTS_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf46_r18_verification"
JSON_OUT: Path = REPORTS_DIR / "integration_report.json"
MD_OUT: Path = REPORTS_DIR / "integration_report.md"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ===========================================================================
# A. Module surface
# ===========================================================================
def test_module_imports() -> None:
    """``molmetal.scripts.integration_verify_r18`` imports cleanly."""
    mod = importlib.import_module("integration_verify_r18")
    assert mod is not None


def test_module_exposes_main() -> None:
    """The orchestrator exposes ``main`` and parses ``--help``."""
    mod = importlib.import_module("integration_verify_r18")
    assert callable(getattr(mod, "main", None))


def test_module_help_exits_zero() -> None:
    """``--help`` exits 0 (smoke for argparse contract)."""
    result = subprocess.run(
        [
            sys.executable, "-m",
            "molmetal.scripts.integration_verify_r18", "--help",
        ],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=60,
    )
    assert result.returncode == 0
    assert "Wf-46" in (result.stdout + result.stderr)


def test_module_exposes_test_registry() -> None:
    """The orchestrator declares the ``TESTS`` registry with 5 entries."""
    mod = importlib.import_module("integration_verify_r18")
    assert hasattr(mod, "TESTS")
    assert isinstance(mod.TESTS, tuple)
    assert len(mod.TESTS) == 5, (
        f"expected 5 integration tests; got {len(mod.TESTS)}"
    )
    ids = [t[0] for t in mod.TESTS]
    for required in ("T1", "T2", "T3", "T4", "T5"):
        assert required in ids, f"missing test {required} in {ids}"


def test_module_exposes_result_classes() -> None:
    """The orchestrator declares ``CheckResult`` and ``TestReport``."""
    mod = importlib.import_module("integration_verify_r18")
    cr = mod.CheckResult(name="x", passed=True)
    assert cr.name == "x"
    assert cr.passed is True
    assert cr.seconds == 0.0
    assert cr.detail == {}
    tr = mod.TestReport(test_id="X", test_name="x")
    assert tr.test_id == "X"
    assert tr.status == "PASS"
    assert tr.checks == []
    assert tr.elapsed_seconds == 0.0
    assert tr.error == ""
    assert tr.inputs == {}
    assert tr.outputs == {}


# ===========================================================================
# B. Five tests in the registry
# ===========================================================================
def test_test_ids_are_unique() -> None:
    """The five test IDs are unique and ordered ``T1..T5``."""
    mod = importlib.import_module("integration_verify_r18")
    ids = [t[0] for t in mod.TESTS]
    assert len(set(ids)) == len(ids), f"duplicate IDs: {ids}"
    expected = ["T1", "T2", "T3", "T4", "T5"]
    assert ids == expected, f"test order broken: {ids}"


def test_each_test_has_callable_driver() -> None:
    """Each registry entry maps to a callable ``_test_tX`` function."""
    mod = importlib.import_module("integration_verify_r18")
    for test_id, test_name, fn in mod.TESTS:
        assert callable(fn), f"{test_id}: driver is not callable"
        assert fn.__name__.startswith("_test_"), (
            f"{test_id}: driver name {fn.__name__!r} doesn't start with '_test_'"
        )


def test_each_test_docstring_mentions_wf() -> None:
    """Each driver's docstring references at least one Wf-NN anchor."""
    mod = importlib.import_module("integration_verify_r18")
    for test_id, test_name, fn in mod.TESTS:
        doc = (fn.__doc__ or "").lower()
        assert "wf-" in doc, f"{test_id}: docstring missing Wf- tag"


# ===========================================================================
# C. End-to-end run produces the report artefacts
# ===========================================================================
def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    """Running ``main()`` writes both artefacts to the canonical path."""
    mod = importlib.import_module("integration_verify_r18")
    rc = mod.main(["--only", "T4"])
    assert rc in (0, 1)
    assert JSON_OUT.exists(), f"missing JSON at {JSON_OUT}"
    assert MD_OUT.exists(), f"missing Markdown at {MD_OUT}"
    body = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    assert body["workflow"] == "Wf-46 Phase 4"
    assert "summary" in body
    assert "tests" in body
    assert isinstance(body["tests"], list)
    assert body["summary"]["n_total"] >= 1


def test_json_schema_round_trip(tmp_path: Path) -> None:
    """The JSON report is well-formed and matches the documented schema."""
    mod = importlib.import_module("integration_verify_r18")
    rc = mod.main(["--only", "T5"])
    assert rc in (0, 1)
    body = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    # Schema version + workflow tag.
    assert body["schema_version"] == "1.0"
    assert body["workflow"] == "Wf-46 Phase 4"
    # Summary keys.
    for k in (
        "n_total", "n_passed", "n_failed", "n_skipped",
        "n_assertions_total", "n_assertions_passed", "n_assertions_skipped",
    ):
        assert k in body["summary"], f"missing summary key: {k}"
    # Per-test rows.
    for row in body["tests"]:
        for k in (
            "test_id", "test_name", "status", "checks",
            "elapsed_seconds", "inputs", "outputs",
        ):
            assert k in row, f"missing row key: {k}"
        assert row["status"] in ("PASS", "FAIL", "SKIPPED")
        assert isinstance(row["checks"], list)


def test_markdown_contains_methodology(tmp_path: Path) -> None:
    """The Markdown report includes the methodology section + 5 tests."""
    mod = importlib.import_module("integration_verify_r18")
    rc = mod.main(["--only", "T1,T2,T3,T4,T5"])
    assert rc in (0, 1)
    md = MD_OUT.read_text(encoding="utf-8")
    assert "Wf-46 Phase 4" in md
    assert "## Methodology" in md
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        assert tid in md, f"Markdown missing test id {tid}"


def test_cli_only_filter_runs_subset() -> None:
    """``--only T4`` runs only T4 (and writes the report)."""
    mod = importlib.import_module("integration_verify_r18")
    rc = mod.main(["--only", "T4"])
    assert rc in (0, 1)
    body = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    assert body["summary"]["n_total"] == 1
    assert body["tests"][0]["test_id"] == "T4"


def test_cli_exclude_filter_skips_test() -> None:
    """``--exclude T1,T2`` runs the remaining three tests."""
    mod = importlib.import_module("integration_verify_r18")
    rc = mod.main(["--exclude", "T1,T2"])
    assert rc in (0, 1)
    body = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    ids = [t["test_id"] for t in body["tests"]]
    assert "T1" not in ids and "T2" not in ids
    assert set(ids).issubset({"T3", "T4", "T5"})


# ===========================================================================
# D. Each individual test driver
# ===========================================================================
def _safe_call(fn) -> Any:
    """Run ``fn()`` and return its :class:`TestReport`.

    Any exception is captured and converted to a SKIPPED TestReport so
    a regression in one driver cannot poison the rest of this module.
    """
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        from dataclasses import dataclass, field
        from typing import List
        @dataclass
        class _Stub:
            test_id: str = "?"
            test_name: str = "?"
            status: str = "SKIPPED"
            checks: List = field(default_factory=list)
            elapsed_seconds: float = 0.0
            error: str = ""
            inputs: dict = field(default_factory=dict)
            outputs: dict = field(default_factory=dict)
        return _Stub(error=f"{type(exc).__name__}: {exc}")


def test_t1_se3_vina_returns_test_report() -> None:
    """T1 returns a non-empty TestReport with at least one check."""
    mod = importlib.import_module("integration_verify_r18")
    rep = _safe_call(mod._test_t1_se3_vina)
    assert rep.test_id == "T1"
    assert rep.test_name
    assert rep.status in ("PASS", "FAIL", "SKIPPED")
    assert len(rep.checks) >= 1
    # At least one assertion must record finite elapsed time.
    assert any(c.seconds >= 0.0 for c in rep.checks)


def test_t2_hilbert_astsr_returns_test_report() -> None:
    """T2 returns a non-empty TestReport with at least three checks."""
    mod = importlib.import_module("integration_verify_r18")
    rep = _safe_call(mod._test_t2_hilbert_astsr)
    assert rep.test_id == "T2"
    assert rep.test_name
    assert rep.status in ("PASS", "FAIL", "SKIPPED")
    assert len(rep.checks) >= 3


def test_t3_multiph_pll_score_any_returns_test_report() -> None:
    """T3 returns a non-empty TestReport with score_any hooks."""
    mod = importlib.import_module("integration_verify_r18")
    rep = _safe_call(mod._test_t3_multiph_pll_score_any)
    assert rep.test_id == "T3"
    assert rep.test_name
    assert rep.status in ("PASS", "FAIL", "SKIPPED")
    assert len(rep.checks) >= 3
    # The unknown-metric check is mandatory in T3 (no-mock invariant).
    check_names = {c.name for c in rep.checks}
    assert "score_any_unknown_metric_rejected" in check_names


def test_t4_variance_mcts_proof_search_returns_test_report() -> None:
    """T4 returns a non-empty TestReport with the four variance checks."""
    mod = importlib.import_module("integration_verify_r18")
    rep = _safe_call(mod._test_t4_variance_mcts_proof_search)
    assert rep.test_id == "T4"
    assert rep.test_name
    assert rep.status in ("PASS", "FAIL", "SKIPPED")
    assert len(rep.checks) >= 3
    check_names = {c.name for c in rep.checks}
    assert "variance_wrapper_constructed" in check_names


def test_t5_triangle_closure_returns_test_report() -> None:
    """T5 returns a non-empty TestReport with the triangle closure hooks."""
    mod = importlib.import_module("integration_verify_r18")
    rep = _safe_call(mod._test_t5_triangle_closure_with_multiph)
    assert rep.test_id == "T5"
    assert rep.test_name
    assert rep.status in ("PASS", "FAIL", "SKIPPED")
    assert len(rep.checks) >= 3
    check_names = {c.name for c in rep.checks}
    assert "triangle_closure_smoke" in check_names


def test_all_five_drivers_have_deterministic_shape() -> None:
    """Each driver returns a TestReport whose dataclass fields are well-formed."""
    mod = importlib.import_module("integration_verify_r18")
    for test_id, _name, fn in mod.TESTS:
        rep = _safe_call(fn)
        assert rep.test_id == test_id
        assert isinstance(rep.checks, list)
        assert isinstance(rep.elapsed_seconds, float)
        assert math.isfinite(rep.elapsed_seconds) or rep.elapsed_seconds == 0.0
        assert isinstance(rep.error, str)
        assert isinstance(rep.inputs, dict)
        assert isinstance(rep.outputs, dict)
        # Every check has finite (or 0.0) elapsed time.
        for c in rep.checks:
            assert math.isfinite(c.seconds) or c.seconds == 0.0
            assert isinstance(c.name, str)
            assert isinstance(c.passed, bool)
            assert isinstance(c.skipped, bool)
            assert isinstance(c.message, str)


def test_json_payload_is_json_safe() -> None:
    """``_to_jsonable`` coerces numpy / Path to JSON-safe primitives."""
    mod = importlib.import_module("integration_verify_r18")
    import numpy as np
    obj = {
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(0.5),
        "path": Path("/tmp"),
        "set": {1, 2, 3},
        "tuple": (1, 2),
    }
    out = mod._to_jsonable(obj)
    body = json.dumps(out)
    parsed = json.loads(body)
    assert parsed["arr"] == [1.0, 2.0]
    assert parsed["scalar"] == 0.5
    assert parsed["path"] == "/tmp"
    assert parsed["set"] in ([1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1])
    assert parsed["tuple"] == [1, 2]


def test_now_is_monotonic() -> None:
    """``_now()`` returns a finite, monotonic float."""
    mod = importlib.import_module("integration_verify_r18")
    t0 = mod._now()
    t1 = mod._now()
    assert math.isfinite(t0)
    assert math.isfinite(t1)
    assert t1 >= t0
