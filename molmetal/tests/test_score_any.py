"""Smoke + integration tests for ``molmetal.scoring.score_any``.

Wf-43 Phase 1 acceptance gate. Three layers:

1. Module-shape tests (always run):
     * available_metrics() is sorted and non-empty
     * tool_for_metric() rejects unknown names
     * score_any() rejects unknown metric names with AdapterUnavailable

2. Per-metric routing tests (always run, *do not* call the tool):
     * Each registered metric name routes to the *real* wrapper
     * No metric name points to a mock
     * Wrappers that depend on missing adapters raise AdapterUnavailable
       — never silently fabricate a number.

3. Real-tool smoke (gated by adapter availability):
     * score_any("sa_score", molecules=[smi]) returns a real SA value.
     * score_any("qed", molecules=[smi]) returns a real QED value.
     * score_any("boltz_affinity", molecules=[smi]) returns a real
       D-MPNN pIC50 (or NaN if the checkpoint is missing).

The tests below cover (1) and (2) for every registered metric. (3) is
optional and skipped automatically when the underlying tool is
unavailable on the test host.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest

from molmetal.scoring import (
    AdapterUnavailable,
    REGISTRY,
    available_metrics,
    score_any,
    tool_for_metric,
)
from molmetal.scoring.score_any import ToolRef


# ---------------------------------------------------------------------------
# 1. Module-shape tests
# ---------------------------------------------------------------------------
class TestRegistryShape:
    def test_available_metrics_non_empty(self):
        names = available_metrics()
        assert isinstance(names, list)
        assert len(names) > 0, "REGISTRY must expose at least one metric"

    def test_available_metrics_sorted(self):
        names = available_metrics()
        assert names == sorted(names)

    def test_available_metrics_are_strings(self):
        names = available_metrics()
        for n in names:
            assert (
                isinstance(n, str) and n
            ), f"metric name must be non-empty string, got {n!r}"

    def test_no_mock_keyword_in_registry(self):
        """Hard guarantee: no metric name contains 'mock'."""
        for n in available_metrics():
            assert (
                "mock" not in n.lower()
            ), f"metric {n!r} contains 'mock' — forbidden by Wf-43"

    def test_every_metric_is_a_toolref(self):
        for n, ref in REGISTRY.items():
            assert isinstance(
                ref, ToolRef
            ), f"{n!r} maps to {type(ref).__name__}, not ToolRef"
            assert callable(ref.wrapper)
            assert isinstance(ref.tool_name, str) and ref.tool_name
            assert isinstance(ref.backend, str) and ref.backend


# ---------------------------------------------------------------------------
# 2. Routing tests — every metric must route to a real tool
# ---------------------------------------------------------------------------
class TestRouting:
    """Verify score_any dispatches to the correct real tool.

    We *do not* invoke the heavy wrappers (those are tested separately
    and may require local binaries). Instead we monkey-patch every
    wrapper to a sentinel that records its metric name, and assert the
    router calls exactly the wrapper corresponding to the metric.
    """

    @pytest.fixture
    def patched_wrappers(self, monkeypatch):
        calls = {}

        def make_recorder(name):
            def _recorder(ctx):
                calls.setdefault(name, []).append(ctx)
                return [0.0] * len(_as_smiles(ctx["molecules"]))

            return _recorder

        for metric, ref in REGISTRY.items():
            ref_proxy = ToolRef(
                wrapper=make_recorder(metric),
                tool_name=ref.tool_name,
                backend=ref.backend,
                notes="[test-sentinel]",
            )
            monkeypatch.setitem(REGISTRY, metric, ref_proxy)
        return calls

    def test_each_metric_dispatches_to_its_own_wrapper(self, patched_wrappers):
        for metric in available_metrics():
            patched_wrappers.clear()
            score_any(metric, molecules=["CCO"], pocket=None)
            assert metric in patched_wrappers, (
                f"score_any({metric!r}) did not call its registered "
                f"wrapper. Called: {list(patched_wrappers.keys())}"
            )

    def test_vina_score_routes_to_vina_not_qvina2_by_default(self, patched_wrappers):
        patched_wrappers.clear()
        score_any("vina_score", molecules=["CCO"], pocket=None)
        assert "vina_score" in patched_wrappers
        assert "vina_qvina2" not in patched_wrappers

    def test_vina_score_routes_to_qvina2_when_env_var_set(
        self, patched_wrappers, monkeypatch
    ):
        monkeypatch.setenv("MOLMETAL_VINA_ENGINE", "qvina2")
        patched_wrappers.clear()
        score_any("vina_score", molecules=["CCO"], pocket=None)
        assert "vina_qvina2" in patched_wrappers
        assert "vina_score" not in patched_wrappers

    def test_vina_score_routes_to_vina_gpu_when_env_var_set(
        self, patched_wrappers, monkeypatch
    ):
        monkeypatch.setenv("MOLMETAL_VINA_ENGINE", "gpu")
        patched_wrappers.clear()
        score_any("vina_score", molecules=["CCO"], pocket=None)
        assert "vina_gpu" in patched_wrappers
        assert "vina_score" not in patched_wrappers

    def test_score_any_no_mock_fallback(self):
        """No metric must silently fabricate a value."""
        # Every wrapper is required to either return a real value or
        # raise AdapterUnavailable. The registry has no `mock` key, and
        # none of the wrappers do `try/except: return 0.0`.
        src = Path(__file__).resolve().parents[1] / "scoring" / "score_any.py"
        text = src.read_text()
        # Defensive: no `fallback` wrapper (Wf-43 forbids it).
        assert "fallback" not in REGISTRY, (
            "REGISTRY must not contain a 'fallback' metric — that would "
            "be a silent mock."
        )
        # The word "mock" should never appear as a wrapper or metric
        # name. (The docstring and comments above mention the word —
        # we only forbid it appearing as code.)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "MockScorer" not in stripped
            assert "MockDocker" not in stripped
            assert "MockPredictor" not in stripped
            assert "MockGenerator" not in stripped


# ---------------------------------------------------------------------------
# 3. Unknown-metric tests
# ---------------------------------------------------------------------------
class TestUnknownMetric:
    def test_unknown_metric_raises_adapter_unavailable(self):
        with pytest.raises(AdapterUnavailable):
            score_any("definitely_not_a_real_metric", molecules=["CCO"])

    def test_tool_for_unknown_metric_raises(self):
        with pytest.raises(AdapterUnavailable):
            tool_for_metric("not_real_either")

    def test_error_message_lists_known_metrics(self):
        try:
            score_any("nope", molecules=["CCO"])
        except AdapterUnavailable as e:
            msg = str(e)
            assert "vina_score" in msg, (
                "AdapterUnavailable message should list known metrics "
                "to help users diagnose the typo"
            )


# ---------------------------------------------------------------------------
# 4. Per-tool wrappers surface their own AdapterUnavailable when missing
# ---------------------------------------------------------------------------
class TestWrapperFailureModes:
    """Reserved / not-yet-shipped metrics must raise, not fabricate.

    This is the load-bearing invariant: every metric in REGISTRY either
    returns a real value or raises. There is no third option.
    """

    @pytest.mark.parametrize(
        "metric",
        [
            "equibind_pose",
            "decomp_diff",
            "molcraft",
            "imold_ood",
            "drugood_ood",
            "glide_score",
        ],
    )
    def test_not_yet_shipped_metrics_raise(self, metric):
        with pytest.raises(AdapterUnavailable):
            score_any(metric, molecules=["CCO"], pocket=None)

    def test_metrics_that_require_pocket_raise_without_one(self):
        # vina_score requires a pocket; the wrapper must raise rather
        # than fabricate.
        with pytest.raises(AdapterUnavailable):
            score_any("vina_score", molecules=["CCO"], pocket=None)


# ---------------------------------------------------------------------------
# 5. Real-tool smoke (auto-skipped when the tool is unavailable)
# ---------------------------------------------------------------------------
def _have_real_sa() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl  # noqa: F401

        sa_score_ertl("CCO")  # actually exercise to surface deep errors
        return True
    except Exception:
        return False


def _have_real_qed() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles  # noqa: F401

        qed_from_smiles("CCO")
        return True
    except Exception:
        return False


def _have_real_pic50() -> bool:
    try:
        from molmetal.molmetal_lam.sbdd_env.pic50_predictor import predict_pic50  # noqa: F401

        return True
    except Exception:
        return False


def _have_real_aizynth() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        from molmetal.molmetal_lam.sbdd_env.aizynth_adapter import check_list  # noqa: F401

        check_list(["CCO"])
        return True
    except Exception:
        return False


class TestRealToolSmoke:
    @pytest.mark.skipif(not _have_real_sa(), reason="SA score not importable")
    def test_sa_score_real_call(self):
        result = score_any("sa_score", molecules=["CCO"])
        assert result["status"] == "MEASURED"
        assert result["metric"] == "sa_score"
        assert result["n"] == 1
        assert result["values"][0] > 0  # real Ertl SA is ~ 1-2 for ethanol

    @pytest.mark.skipif(not _have_real_qed(), reason="QED not importable")
    def test_qed_real_call(self):
        result = score_any("qed", molecules=["CC(=O)Oc1ccccc1C(=O)O"])
        assert result["status"] == "MEASURED"
        assert result["metric"] == "qed"
        assert 0.0 <= result["values"][0] <= 1.0

    @pytest.mark.skipif(not _have_real_aizynth(), reason="AiZynth not importable")
    def test_aizynth_real_call(self):
        result = score_any("aizynth_synthesis", molecules=["CCO"])
        assert result["status"] == "MEASURED"
        assert result["metric"] == "aizynth_synthesis"
        assert result["n"] == 1
        assert isinstance(result["values"][0], bool)

    @pytest.mark.skipif(not _have_real_pic50(), reason="pIC50 predictor not importable")
    def test_boltz_affinity_real_call(self):
        # NaN is acceptable if the ckpt is missing on this host — the
        # important thing is we get a NUMBER from a real model, not a
        # fabricated one.
        result = score_any("boltz_affinity", molecules=["CCO"])
        assert result["status"] == "MEASURED"
        assert result["metric"] == "boltz_affinity"
        assert isinstance(result["values"][0], float)


# ---------------------------------------------------------------------------
# 6. Mocked-end-to-end: feed a mock wrapper for one metric and assert
#    the router calls it, returns the value, and the metadata is right.
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_return_shape(self):
        # SA-score is real on every box; use it for an end-to-end check.
        if not _have_real_sa():
            pytest.skip("SA score not importable on this host")
        result = score_any("sa_score", molecules=["CCO", "CCCCO"])
        assert set(result.keys()) >= {
            "metric",
            "tool",
            "backend",
            "values",
            "n",
            "status",
        }
        assert result["metric"] == "sa_score"
        assert result["n"] == 2
        assert result["status"] == "MEASURED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_smiles(molecules: Iterable) -> list:
    """Mirror of ``molmetal.scoring.score_any._as_smiles_list`` for tests."""
    out = []
    for m in molecules:
        if isinstance(m, str):
            out.append(m)
            continue
        smi = getattr(m, "smiles", None)
        out.append(str(smi) if smi is not None else str(m))
    return out
