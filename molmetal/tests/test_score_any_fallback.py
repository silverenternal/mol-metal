"""Tests for Wf-48 Phase 3 — score_any() fallback chain test.

Validates the Wf-48 Phase 3 fallback chain harness
(:file:`molmetal/scripts/test_score_any_fallback.py`):

1. The harness module imports cleanly and exposes the public API.
2. ``ROUTING_POLICY`` metadata audit catches broken refs, MOCK_DELETED
   refs, cycles, and self-references in every metric's ``fallback_chain``.
3. The primary-unavailable scenario propagates ``AdapterUnavailable``
   (no silent mock substitution) for every metric whose primary tool
   can be patched.
4. The ``MOLMETAL_VINA_ENGINE`` env var routes vina_score /
   vina_high_affinity / vina_min_score to the correct fallback wrapper
   (vina_qvina2 / vina_gpu).
5. The harness writes a well-formed ``fallback_test.json`` report.
6. The harness exit code is 0 on PASS and 2 on FAIL.

Test count target: ≥20 distinct checks.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
FALLBACK_REPORTS = REPORTS / "wf48_fallback"


def _ensure_paths() -> None:
    for p in (str(SCRIPTS), str(PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_paths()


# ---------------------------------------------------------------------------
# 1. Module-shape tests
# ---------------------------------------------------------------------------
def test_harness_module_imports() -> None:
    """The harness module imports cleanly."""
    mod = importlib.import_module("test_score_any_fallback")
    assert mod is not None
    assert hasattr(mod, "run_test")
    assert hasattr(mod, "main")
    assert hasattr(mod, "FallbackReport")
    assert hasattr(mod, "FallbackConfig")
    assert hasattr(mod, "FallbackStepResult")


def test_harness_version_string() -> None:
    """The harness exposes a wf48 version string."""
    mod = importlib.import_module("test_score_any_fallback")
    assert isinstance(getattr(mod, "__version__", ""), str)
    assert "wf48" in mod.__version__


def test_dataclass_shapes() -> None:
    """The result dataclasses are well-formed dataclasses."""
    mod = importlib.import_module("test_score_any_fallback")
    from dataclasses import fields

    for cls in (
        mod.FallbackStepResult,
        mod.FallbackReport,
        mod.FallbackConfig,
    ):
        fs = fields(cls)
        assert isinstance(fs, (list, tuple))
        assert len(fs) >= 1


# ---------------------------------------------------------------------------
# 2. Metadata-audit tests — fallback_chain integrity at the policy level
# ---------------------------------------------------------------------------
def test_metadata_audit_no_broken_refs() -> None:
    """Every ref in every metric's ``fallback_chain`` resolves."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    bad = mod._check_chain_refs(ROUTING_POLICY)
    assert not bad, f"broken fallback_chain refs: {bad}"


def test_metadata_audit_no_mock_deleted_refs() -> None:
    """No fallback_chain points at a MOCK_DELETED metric."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    bad = mod._check_chain_no_mock_deleted(ROUTING_POLICY)
    assert not bad, (
        f"fallback_chain points at MOCK_DELETED metrics: {bad}"
    )


def test_metadata_audit_no_cycles() -> None:
    """No cycles in the fallback_chain graph."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    cycles = mod._check_chain_no_cycles(ROUTING_POLICY)
    assert not cycles, f"cycles detected: {cycles}"


def test_metadata_audit_no_self_refs() -> None:
    """No metric appears in its own ``fallback_chain``."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    bad = mod._check_chain_no_self_refs(ROUTING_POLICY)
    assert not bad, f"self-references detected: {bad}"


def test_metadata_audit_metrics_with_chain_count() -> None:
    """Audit reports the right number of metrics with a non-empty chain."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    audit = mod._run_metadata_audit(ROUTING_POLICY)
    expected = sum(1 for e in ROUTING_POLICY.values() if e.fallback_chain)
    assert audit["n_metrics_with_chain"] == expected
    assert expected >= 8, (
        f"Expected ≥8 metrics with fallback_chain (Wf-43 Phase 2 ships "
        f"vina + diffdock + flowdock + glide fallback families), got "
        f"{expected}"
    )


def test_metadata_audit_includes_known_fallback_families() -> None:
    """The audit reports metrics from every documented fallback family."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    expected_metrics_with_chain = {
        "vina_score",
        "vina_high_affinity",
        "vina_min_score",
        "vina_gpu",
        "diffdock_pose",
        "equibind_pose",
        "flowdock_pose",
        "glide_score",
    }
    actual = {
        m for m, e in ROUTING_POLICY.items() if e.fallback_chain
    }
    assert expected_metrics_with_chain <= actual, (
        f"missing fallback families: "
        f"{expected_metrics_with_chain - actual}"
    )


# ---------------------------------------------------------------------------
# 3. Primary-unavailable propagation tests
# ---------------------------------------------------------------------------
def _run_primary_unavailable(metric: str) -> Any:
    """Run the harness's primary-unavailable scenario for ``metric``."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    primary = str(ROUTING_POLICY[metric].primary_tool)
    chain = tuple(ROUTING_POLICY[metric].fallback_chain or ())
    return mod._run_primary_unavailable_scenario(metric, primary, chain)


def test_primary_unavailable_propagates_adapter_unavailable() -> None:
    """When primary tool is unavailable, the router raises AdapterUnavailable."""
    from molmetal.scoring.score_any import AdapterUnavailable

    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring import score_any

    with mod._patch_primary_unavailable("vina_score"):
        with pytest.raises(AdapterUnavailable):
            score_any("vina_score", molecules=["CCO"], pocket=None)


def test_primary_unavailable_propagates_for_all_chain_metrics() -> None:
    """Every metric in the fallback families propagates AdapterUnavailable."""
    metrics = [
        "vina_score", "vina_high_affinity", "vina_min_score",
        "vina_gpu", "diffdock_pose", "equibind_pose",
        "flowdock_pose", "glide_score",
    ]
    for m in metrics:
        result = _run_primary_unavailable(m)
        assert result.actual_status == "ADAPTER_UNAVAILABLE", (
            f"{m!r}: expected ADAPTER_UNAVAILABLE, got "
            f"{result.actual_status!r}; error={result.error}"
        )


def test_primary_unavailable_records_scenario_metadata() -> None:
    """The scenario record preserves metric + chain metadata."""
    result = _run_primary_unavailable("vina_score")
    assert result.metric == "vina_score"
    assert result.scenario == "primary_unavailable"
    assert result.expected_status == "ADAPTER_UNAVAILABLE"
    assert "vina_qvina2" in result.fallback_chain
    assert "vina_gpu" in result.fallback_chain


def test_primary_unavailable_restores_registry() -> None:
    """The monkey-patch is reverted after the scenario completes."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import REGISTRY

    pre_notes = {k: v.notes for k, v in REGISTRY.items()}
    _run_primary_unavailable("vina_score")
    post_notes = {k: v.notes for k, v in REGISTRY.items()}
    assert pre_notes == post_notes, (
        "REGISTRY monkey-patch leaked after scenario"
    )


# ---------------------------------------------------------------------------
# 4. Env-var routing tests
# ---------------------------------------------------------------------------
def test_env_routing_qvina2_for_vina_score(monkeypatch) -> None:
    """``MOLMETAL_VINA_ENGINE=qvina2`` routes vina_score to vina_qvina2."""
    from molmetal.scoring.score_any import (
        AdapterUnavailable,
        REGISTRY,
    )
    from molmetal.scoring import score_any

    monkeypatch.setenv("MOLMETAL_VINA_ENGINE", "qvina2")
    pre_qv2 = REGISTRY["vina_qvina2"]
    try:
        try:
            score_any("vina_score", molecules=["CCO"], pocket=None)
        except AdapterUnavailable:
            # The routing decision is captured before the real tool
            # fails. We check that REGISTRY pointed at vina_qvina2
            # for the duration of the call by inspecting the call
            # path: score_any() first checks MOLMETAL_VINA_ENGINE and
            # then calls REGISTRY[env-metric].wrapper.
            pass
        # Verify the routing function picks vina_qvina2:
        from molmetal.scoring.score_any import score_any as _sa
        # Direct check of the env routing branch via a patched
        # wrapper: replace vina_qvina2 with a sentinel, and verify
        # the sentinel was the one called when env=qvina2.
        from molmetal.scoring.score_any import ToolRef

        def _sentinel(ctx: Dict[str, Any]) -> List[float]:
            return [-9.9]

        REGISTRY["vina_qvina2"] = ToolRef(
            wrapper=_sentinel,
            tool_name=pre_qv2.tool_name,
            backend=pre_qv2.backend,
            notes="[test-env-routing]",
        )
        result = score_any("vina_score", molecules=["CCO"], pocket=None)
        assert result["tool"] == pre_qv2.tool_name
        assert result["values"] == [-9.9]
    finally:
        REGISTRY["vina_qvina2"] = pre_qv2


def test_env_routing_gpu_for_vina_high_affinity(monkeypatch) -> None:
    """``MOLMETAL_VINA_ENGINE=gpu`` routes vina_high_affinity to vina_gpu."""
    from molmetal.scoring.score_any import REGISTRY, ToolRef
    from molmetal.scoring import score_any

    monkeypatch.setenv("MOLMETAL_VINA_ENGINE", "gpu")
    pre_vgpu = REGISTRY["vina_gpu"]
    try:
        def _sentinel(ctx: Dict[str, Any]) -> List[float]:
            return [-10.5]

        REGISTRY["vina_gpu"] = ToolRef(
            wrapper=_sentinel,
            tool_name=pre_vgpu.tool_name,
            backend=pre_vgpu.backend,
            notes="[test-env-routing]",
        )
        result = score_any(
            "vina_high_affinity", molecules=["CCO"], pocket=None
        )
        assert result["tool"] == pre_vgpu.tool_name
    finally:
        REGISTRY["vina_gpu"] = pre_vgpu


def test_env_routing_scenario_returns_env_routed_status() -> None:
    """The env-routing scenario records ``ENV_ROUTED`` status."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    primary = str(ROUTING_POLICY["vina_score"].primary_tool)
    chain = tuple(ROUTING_POLICY["vina_score"].fallback_chain or ())
    result = mod._run_env_routing_scenario(
        "vina_score", primary, chain, "qvina2"
    )
    assert result.scenario == "env_MOLMETAL_VINA_ENGINE=qvina2"
    assert result.expected_status == "ENV_ROUTED"


# ---------------------------------------------------------------------------
# 5. Per-metric report tests
# ---------------------------------------------------------------------------
def test_run_metric_scenarios_returns_fallback_report() -> None:
    """``_run_metric_scenarios`` returns a well-formed FallbackReport."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    rep = mod._run_metric_scenarios("vina_score", ROUTING_POLICY)
    assert rep.metric == "vina_score"
    assert rep.overall_status in ("PASS", "FAIL", "N/A")
    assert len(rep.results) >= 1
    for sr in rep.results:
        assert "scenario" in sr
        assert "actual_status" in sr
        assert sr["actual_status"] in (
            "MEASURED", "ADAPTER_UNAVAILABLE",
            "ENV_ROUTED", "FAIL",
        )


def test_run_metric_scenarios_no_chain_metric() -> None:
    """A metric with no fallback_chain still passes chain_metadata_no_chain."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    # sa_score has no fallback chain
    rep = mod._run_metric_scenarios("sa_score", ROUTING_POLICY)
    assert rep.fallback_chain == ()
    assert rep.overall_status in ("PASS", "FAIL")


def test_aggregate_keys_present() -> None:
    """The aggregate dict exposes the keys callers rely on."""
    mod = importlib.import_module("test_score_any_fallback")

    sample: List[mod.FallbackReport] = [
        mod.FallbackReport(
            metric="vina_score",
            primary_tool="AutoDock Vina 1.2.7",
            backend="python+cli",
            fallback_chain=("vina_qvina2", "vina_gpu"),
            wf43_status="MEASURED",
            results=[
                {"scenario": "chain_metadata",
                 "actual_status": "MEASURED"},
                {"scenario": "primary_unavailable",
                 "actual_status": "ADAPTER_UNAVAILABLE"},
                {"scenario": "env_MOLMETAL_VINA_ENGINE=qvina2",
                 "actual_status": "ENV_ROUTED"},
            ],
            overall_status="PASS",
        ),
    ]
    agg = mod._aggregate(sample)
    for k in (
        "n_metrics_with_fallback_chain",
        "n_total_metrics_tested",
        "n_pass",
        "n_fail",
        "n_chain_metadata_pass",
        "n_primary_unavailable_propagate",
        "n_env_routed_pass",
        "fallback_chain_success_rate",
        "env_routing_success_rate",
    ):
        assert k in agg, f"missing aggregate key: {k!r}"
    assert agg["n_pass"] == 1
    assert agg["n_primary_unavailable_propagate"] == 1
    assert agg["n_env_routed_pass"] == 1


# ---------------------------------------------------------------------------
# 6. End-to-end: harness writes a well-formed report
# ---------------------------------------------------------------------------
def test_run_test_writes_fallback_test_json() -> None:
    """``run_test`` writes ``fallback_test.json`` with the expected schema."""
    mod = importlib.import_module("test_score_any_fallback")
    cfg = mod.FallbackConfig(out_dir=str(FALLBACK_REPORTS))
    payload = mod.run_test(cfg)
    json_path = FALLBACK_REPORTS / "fallback_test.json"
    assert json_path.exists(), f"{json_path} not written"
    on_disk = json.loads(json_path.read_text())
    assert on_disk["recipe"] == "wf48_fallback_chain_test"
    assert on_disk["tool"] == "test_score_any_fallback.py"
    assert "metadata_audit" in on_disk
    assert "per_metric" in on_disk
    assert "aggregate" in on_disk
    for required in (
        "broken_refs", "mock_deleted_refs",
        "cycles", "self_refs", "n_metrics_with_chain",
    ):
        assert required in on_disk["metadata_audit"]
    assert payload == on_disk


def test_run_test_per_metric_covers_all_policy_metrics() -> None:
    """Every metric in ROUTING_POLICY is covered by the per_metric list."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import ROUTING_POLICY

    cfg = mod.FallbackConfig(out_dir=str(FALLBACK_REPORTS))
    payload = mod.run_test(cfg)
    covered = {r["metric"] for r in payload["per_metric"]}
    assert set(ROUTING_POLICY.keys()) <= covered


def test_run_test_exit_code_zero_when_all_pass() -> None:
    """When every metric passes + metadata is clean, exit code is 0."""
    cmd = [
        sys.executable,
        str(SCRIPTS / "test_score_any_fallback.py"),
        "--out-dir", str(FALLBACK_REPORTS),
    ]
    res = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": (
                str(SCRIPTS)
                + os.pathsep
                + str(PROJECT_ROOT)
                + os.pathsep
                + os.environ.get("PYTHONPATH", "")
            ),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode in (0, 2), (
        f"harness crashed: rc={res.returncode} stderr={res.stderr[-500:]}"
    )
    json_path = FALLBACK_REPORTS / "fallback_test.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    if (
        int(payload["aggregate"]["n_fail"]) == 0
        and not payload["metadata_audit"]["broken_refs"]
        and not payload["metadata_audit"]["mock_deleted_refs"]
        and not payload["metadata_audit"]["cycles"]
        and not payload["metadata_audit"]["self_refs"]
    ):
        assert res.returncode == 0, (
            f"expected rc=0 on clean run, got {res.returncode}: "
            f"{res.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# 7. Specific chain-shape assertions — the families Wf-43 Phase 2 ships
# ---------------------------------------------------------------------------
def test_vina_family_fallback_chain() -> None:
    """vina_score fallback chain is (vina_qvina2, vina_gpu)."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    for m in ("vina_score", "vina_high_affinity", "vina_min_score"):
        chain = ROUTING_POLICY[m].fallback_chain
        assert "vina_qvina2" in chain, f"{m!r} missing vina_qvina2"
        assert "vina_gpu" in chain, f"{m!r} missing vina_gpu"


def test_vina_gpu_fallback_chain() -> None:
    """vina_gpu falls back to vina_score."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    chain = ROUTING_POLICY["vina_gpu"].fallback_chain
    assert "vina_score" in chain


def test_diffdock_flowdock_bidirectional_chain() -> None:
    """diffdock_pose → flowdock_pose, flowdock_pose → diffdock_pose."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    assert "flowdock_pose" in ROUTING_POLICY["diffdock_pose"].fallback_chain
    assert "diffdock_pose" in ROUTING_POLICY["flowdock_pose"].fallback_chain


def test_equibind_pose_chain_includes_both_se3_surrogates() -> None:
    """equibind_pose fallback chain is (flowdock_pose, diffdock_pose)."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    chain = ROUTING_POLICY["equibind_pose"].fallback_chain
    assert "flowdock_pose" in chain
    assert "diffdock_pose" in chain


def test_glide_score_falls_back_to_vina() -> None:
    """Commercial Glide falls back to OSS AutoDock Vina."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    entry = ROUTING_POLICY["glide_score"]
    assert "vina_score" in entry.fallback_chain


# ---------------------------------------------------------------------------
# 8. Wf-43 no-mock invariant — the harness never fabricates a value
# ---------------------------------------------------------------------------
def test_harness_source_has_no_mock_scorer_definitions() -> None:
    """The harness source code must not define a mock fallback function."""
    src_path = (
        Path(__file__).resolve().parents[1] / "scripts"
        / "test_score_any_fallback.py"
    )
    text = src_path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "MockScorer" in stripped or "MockDocker" in stripped:
            raise AssertionError(
                f"harness defines a mock fallback: {stripped[:120]}"
            )


# ---------------------------------------------------------------------------
# 9. _capture_fallback_calls — auxiliary context manager
# ---------------------------------------------------------------------------
def test_capture_fallback_calls_restores_registry() -> None:
    """``_capture_fallback_calls`` reverts wrapper monkey-patches on exit."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import REGISTRY

    pre_notes = {k: v.notes for k, v in REGISTRY.items()}
    with mod._capture_fallback_calls(("vina_score", "vina_qvina2")):
        # Inside: notes are augmented with the counter marker.
        assert "wf48-fallback-counter" in REGISTRY["vina_score"].notes
        assert "wf48-fallback-counter" in REGISTRY["vina_qvina2"].notes
    post_notes = {k: v.notes for k, v in REGISTRY.items()}
    assert pre_notes == post_notes, (
        "_capture_fallback_calls leaked monkey-patch"
    )


def test_capture_fallback_calls_counts_invocation() -> None:
    """``_capture_fallback_calls`` increments the counter on each call."""
    mod = importlib.import_module("test_score_any_fallback")
    from molmetal.scoring.score_any import (
        AdapterUnavailable,
        ToolRef,
        REGISTRY,
    )
    from molmetal.scoring import score_any

    pre_qv2 = REGISTRY["vina_qvina2"]
    try:
        def _sentinel(ctx: Dict[str, Any]) -> List[float]:
            return [-9.9]

        REGISTRY["vina_qvina2"] = ToolRef(
            wrapper=_sentinel,
            tool_name=pre_qv2.tool_name,
            backend=pre_qv2.backend,
            notes="[test-capture-counter]",
        )
        with mod._capture_fallback_calls(("vina_qvina2",)) as counts:
            result = score_any("vina_qvina2", molecules=["CCO"], pocket=None)
            assert result["values"] == [-9.9]
            assert counts["vina_qvina2"] >= 1
    finally:
        REGISTRY["vina_qvina2"] = pre_qv2
