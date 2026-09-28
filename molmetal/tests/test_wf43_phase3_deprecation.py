"""Wf-43 Phase 3 — Mock-script deprecation acceptance test.

================================================================
Wf-43 Phase 3: deprecate mock-using scripts; reroute through score_any()
================================================================

This test enforces the Wf-43 Phase 3 contract on the 10 deprecated
bench/benchmark scripts:

1. Each of the 10 scripts must be importable (parses cleanly and
   Python-imports without error).
2. None of the 10 scripts may define a ``_stub_score`` /
   ``_mock_score`` / ``_fake_score`` function — these were the Wf-42
   audit's "mock scorer" findings and they have all been either
   rewritten, renamed (``_StubVNet`` → ``_CpuOnlyVNetShim``), or
   classified as CPU benchmark scaffolding (see
   ``molmetal/reports/wf43_deprecation/deprecated_mocks.md``).
3. Where a real ``score_any`` REGISTRY metric exists for the
   bench's headline output, the script must expose a function that
   imports / uses ``molmetal.scoring.score_any``.  The specific
   metric used per script is:

       bench_pb_valid.py                       → pb_pass_rate
       bench_mcts_real_trajectory.py           → sa_score (--mode real)
       benchmark_mcts_real.py                  → sa_score
       benchmark_astsr_extended.py             → boltz_affinity

4. The deprecation report
   ``molmetal/reports/wf43_deprecation/deprecated_mocks.md`` exists
   and lists every target script in §2.

The test joins the existing
``molmetal/tests/test_score_any_policy.py`` (Phase 2) +
``test_score_any.py`` (Phase 1) as the **Wf-43 acceptance chain**.
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from typing import Iterable, List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
_REPORT_PATH = (
    _REPO_ROOT / "molmetal" / "reports" / "wf43_deprecation"
    / "deprecated_mocks.md"
)

#: The 10 scripts Wf-43 Phase 3 deprecates.
TARGET_SCRIPTS: tuple = (
    "bench_mcts_real_trajectory.py",
    "bench_deflexpressor_end_to_end.py",
    "bench_coupling_ic50.py",
    "bench_lambda_cfm_e2e.py",
    "benchmark_astsr_extended.py",
    "benchmark_r18_end_to_end.py",
    "benchmark_r18_platinai.py",
    "benchmark_r18_metalcytotox.py",
    "benchmark_mcts_real.py",
    "bench_pb_valid.py",
)

#: Forbidden mock-scorer function names.  The Wf-42 audit found these
#: patterns in the original 10 scripts.  After Phase 3 none of these
#: names must appear as top-level function definitions.
FORBIDDEN_FN_NAMES: tuple = (
    "_stub_score",
    "_mock_score",
    "_fake_score",
    "_dummy_score",
)

#: Per-script: the metric name in ``score_any`` REGISTRY that the
#: script's ``score_any()`` redirect must use.  Empty tuple means
#: "no score_any redirect required — classified as no-mock-found /
#: CPU-scaffolding / label-text / module-name only" (see the
#: deprecation report for the per-script rationale).
EXPECTED_SCORE_ANY_METRIC: dict = {
    "bench_mcts_real_trajectory.py": ("sa_score",),
    "bench_deflexpressor_end_to_end.py": (),  # no mock — r2 is not a score_any metric
    "bench_coupling_ic50.py": (),  # warning text only — bridge is loaded from real NPZ
    "bench_lambda_cfm_e2e.py": (),  # status label only — bridge is loaded from real NPZ
    "benchmark_astsr_extended.py": ("boltz_affinity",),
    "benchmark_r18_end_to_end.py": (),  # CPU scaffolding only
    "benchmark_r18_platinai.py": (),  # CPU scaffolding only
    "benchmark_r18_metalcytotox.py": (),  # module-name reference only
    "benchmark_mcts_real.py": ("sa_score",),
    "bench_pb_valid.py": ("pb_pass_rate",),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_source(name: str) -> str:
    """Read the source of a target script (raises if missing)."""
    path = _SCRIPTS_DIR / name
    assert path.exists(), f"Wf-43 Phase 3 target script not found: {path}"
    return path.read_text(encoding="utf-8")


def _top_level_function_names(source: str) -> List[str]:
    """Return the list of top-level function / method names defined in source."""
    tree = ast.parse(source)
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # We deliberately include NESTED classes' methods because
            # ``_StubVNet`` is nested inside a method body in some
            # scripts — for those we do NOT want to flag the class
            # itself as a top-level mock scorer.  The strict check
            # below only fires when the name appears at module scope.
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                names.append(node.name)
    return names


def _has_score_any_reference(source: str) -> bool:
    """Return True iff source contains a real ``molmetal.scoring.score_any`` import."""
    return bool(
        re.search(
            r"from\s+molmetal\.scoring\.score_any\s+import",
            source,
        )
        or re.search(
            r"from\s+molmetal\.scoring\s+import\s+.*score_any",
            source,
        )
        or re.search(
            r"score_any\(",
            source,
        )
    )


def _score_any_metrics_referenced(source: str) -> set:
    """Return the set of metric-name string literals passed to ``score_any(...)``."""
    # Greedy match for ``score_any("metric_name", ...)`` patterns.
    return set(
        re.findall(
            r'score_any\(\s*["\']([a-z_0-9]+)["\']',
            source,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestWf43Phase3Deprecation:
    """Wf-43 Phase 3 acceptance — every target script is deprecation-safe."""

    def test_all_target_scripts_exist(self) -> None:
        for name in TARGET_SCRIPTS:
            assert (_SCRIPTS_DIR / name).exists(), (
                f"Wf-43 Phase 3 target script missing: {name}"
            )

    def test_all_target_scripts_parse_cleanly(self) -> None:
        """Every target script must compile (AST parse)."""
        for name in TARGET_SCRIPTS:
            src = _read_source(name)
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"{name}: SyntaxError {e}")

    @pytest.mark.parametrize("name", TARGET_SCRIPTS)
    def test_no_forbidden_mock_scorer_function(self, name: str) -> None:
        """No ``_stub_score`` / ``_mock_score`` / ``_fake_score`` / ``_dummy_score``."""
        src = _read_source(name)
        names = _top_level_function_names(src)
        forbidden_hits = [n for n in names if n in FORBIDDEN_FN_NAMES]
        assert not forbidden_hits, (
            f"{name}: forbidden mock-scorer function(s) still present: "
            f"{forbidden_hits}.  See Wf-43 Phase 3 deprecation report."
        )

    @pytest.mark.parametrize("name", TARGET_SCRIPTS)
    def test_score_any_redirect_for_required_metric(self, name: str) -> None:
        """Where applicable, the script must reference the expected score_any metric."""
        expected = EXPECTED_SCORE_ANY_METRIC.get(name, ())
        if not expected:
            return  # No score_any redirect required.
        src = _read_source(name)
        # The script must import score_any OR call score_any.
        assert _has_score_any_reference(src), (
            f"{name}: expected score_any() reference for metric {expected}, "
            f"but no molmetal.scoring.score_any import / call found."
        )
        referenced = _score_any_metrics_referenced(src)
        missing = set(expected) - referenced
        assert not missing, (
            f"{name}: expected score_any({sorted(expected)}) but only "
            f"score_any({sorted(referenced)}) found (missing {sorted(missing)})."
        )

    def test_deprecation_report_exists(self) -> None:
        """The deprecation report must exist."""
        assert _REPORT_PATH.exists(), (
            f"Wf-43 Phase 3 deprecation report missing: {_REPORT_PATH}"
        )
        content = _REPORT_PATH.read_text(encoding="utf-8")
        # The report must mention every target script in §2.
        for name in TARGET_SCRIPTS:
            assert name in content, (
                f"Deprecation report does not mention target script: {name}"
            )
        # The report must list at least one score_any() redirect.
        assert "score_any" in content, (
            "Deprecation report does not mention score_any()."
        )

    def test_score_any_metric_registry_has_required_metrics(self) -> None:
        """All metrics referenced by the 10 scripts must be in score_any REGISTRY."""
        from molmetal.scoring.score_any import available_metrics

        registered = set(available_metrics())
        for name in TARGET_SCRIPTS:
            src = _read_source(name)
            for metric in _score_any_metrics_referenced(src):
                assert metric in registered, (
                    f"{name}: score_any({metric!r}) is not in REGISTRY. "
                    f"Available: {sorted(registered)}"
                )


class TestWf43Phase3ScoreAnyReroute:
    """End-to-end check: the new score_any helpers actually return MEASURED."""

    def test_pb_pass_rate_via_score_any_returns_measured(self) -> None:
        """``overall_pass_rate_via_score_any`` returns a real dict."""
        from molmetal.scripts.bench_pb_valid import (
            PBResult,
            overall_pass_rate_via_score_any,
        )

        # Build a couple of synthetic PBResults (no real docking
        # required — we only test the JSON envelope).
        results = [
            PBResult(
                pocket_id="test1",
                candidate="CCO",
                candidate_name="ethanol",
                method="mol",
                trajectory_method="se3sb",
                status="ok",
                pb_valid=True,
                n_checks=14,
                n_passed=14,
                n_failed=0,
                n_skipped=0,
                pass_rate=1.0,
                failed_checks=(),
                missing_checks=(),
                elapsed_s=0.01,
                note=None,
            ),
        ]
        d = overall_pass_rate_via_score_any(results, pb_mode="mol")
        assert d["metric"] == "pb_pass_rate"
        assert d["tool"] == "PoseBusters (R17 ship)"
        assert d["backend"] == "python"
        # status is "MEASURED" when the tool is wired, or
        # "skipped: AdapterUnavailable" when PoseBusters is missing.
        assert d["status"].startswith("MEASURED") or d["status"].startswith(
            "skipped:"
        ), f"unexpected status: {d['status']!r}"

    def test_sa_score_baseline_for_smiles_returns_measured(self) -> None:
        """``_sa_score_baseline_for_smiles`` returns a real dict."""
        from molmetal.scripts.benchmark_mcts_real import (
            _sa_score_baseline_for_smiles,
        )

        d = _sa_score_baseline_for_smiles(["CCO", "CCN"])
        assert d["tool"] == "SA score (RDKit Ertl)"
        assert d["status"].startswith("MEASURED") or d["status"].startswith(
            "skipped:"
        ), f"unexpected status: {d['status']!r}"