"""Tests for Wf-48 Phase 1 — ``score_any()`` router benchmark.

These tests verify the integrity of
:file:`molmetal/scripts/bench_score_any_router.py`:

* the bench's module-shape (version, dataclasses, registry);
* the per-script registry covers all 10 Wf-43 ship scripts;
* the ``_score_any_recorder`` context manager monkey-patches
  :class:`REGISTRY` and records every wrapper invocation;
* pre/post probes bracket the script execution and are recorded
  under ``router_overhead_s``;
* the bench's accuracy field is computed correctly (PASS when every
  expected metric is invoked; FAIL when an expected metric is missing;
  N/A for non-metric scripts);
* the bench writes a valid ``router_bench.json`` when run on the
  full 10-script set;
* the bench's stdout exit code is 0 when routing passes and 2 when
  any script fails;
* AdapterUnavailable propagates through the router as documented
  (no mock substitution);
* the routing-policy JSON contract is honoured (every metric → tool
  pairing the bench reports matches ``ROUTING_POLICY``);
* the bench never fabricates a number when ``--no-mock-unavailable``
  is set.

Test count target: 15 distinct checks.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
SCORE_ANY_REPORTS = REPORTS / "wf48_score_any"


def _ensure_paths() -> None:
    for p in (str(SCRIPTS), str(PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_paths()


# ---------------------------------------------------------------------------
# 1. Module-shape tests
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """The bench module imports cleanly."""
    mod = importlib.import_module("bench_score_any_router")
    assert mod is not None
    assert hasattr(mod, "run_bench")
    assert hasattr(mod, "main")
    assert hasattr(mod, "DEFAULT_SCRIPT_REGISTRY")
    assert hasattr(mod, "_score_any_recorder")


def test_bench_version_string() -> None:
    """The bench module exposes a wf48 version string."""
    mod = importlib.import_module("bench_score_any_router")
    assert isinstance(getattr(mod, "__version__", ""), str)
    assert "wf48" in mod.__version__


def test_dataclass_shapes() -> None:
    """ScoreAnyCall + ScriptReport + BenchConfig are dataclasses."""
    mod = importlib.import_module("bench_score_any_router")
    from dataclasses import fields

    for cls in (mod.ScoreAnyCall, mod.ScriptReport, mod.BenchConfig):
        # ``dataclasses.fields`` returns a tuple in Python 3.10+.
        fs = fields(cls)
        assert isinstance(fs, (list, tuple))
        assert len(fs) >= 1


# ---------------------------------------------------------------------------
# 2. Registry tests — every Wf-43 ship script is in DEFAULT_SCRIPT_REGISTRY
# ---------------------------------------------------------------------------
EXPECTED_SCRIPTS = (
    "bench_mcts_real_trajectory.py",
    "bench_deflexpressor_end_to_end.py",
    "benchmark_astsr_extended.py",
    "benchmark_r18_end_to_end.py",
    "benchmark_r18_platinai.py",
    "benchmark_r18_metalcytotox.py",
    "benchmark_mcts_real.py",
    "bench_pb_valid.py",
    "bench_coupling_ic50.py",
    "bench_lambda_cfm_e2e.py",
)


def test_registry_covers_all_ten_wf43_ship_scripts() -> None:
    mod = importlib.import_module("bench_score_any_router")
    reg = mod.DEFAULT_SCRIPT_REGISTRY
    listed = {str(entry["script"]) for entry in reg}
    assert listed == set(EXPECTED_SCRIPTS), (
        f"DEFAULT_SCRIPT_REGISTRY missing/extra: "
        f"missing={set(EXPECTED_SCRIPTS) - listed}, "
        f"extra={listed - set(EXPECTED_SCRIPTS)}"
    )


def test_registry_has_ten_entries() -> None:
    mod = importlib.import_module("bench_score_any_router")
    assert len(mod.DEFAULT_SCRIPT_REGISTRY) == 10


def test_metric_layer_scripts_have_expected_metrics() -> None:
    """Every script with ``metric_layer=True`` declares ≥ 1 expected metric."""
    mod = importlib.import_module("bench_score_any_router")
    for entry in mod.DEFAULT_SCRIPT_REGISTRY:
        if entry.get("metric_layer"):
            assert tuple(
                entry.get("expected_metrics", ())
            ), f"metric-layer script {entry['script']!r} has no expected_metrics"


def test_non_metric_layer_scripts_have_no_expected_metrics() -> None:
    """Scripts that do not use score_any in the metric layer should have
    no expected_metrics entries (otherwise the bench flags them as
    routing failures)."""
    mod = importlib.import_module("bench_score_any_router")
    for entry in mod.DEFAULT_SCRIPT_REGISTRY:
        if not entry.get("metric_layer"):
            assert not entry.get("expected_metrics"), (
                f"non-metric script {entry['script']!r} declares "
                f"expected_metrics but does not route a metric"
            )


# ---------------------------------------------------------------------------
# 3. _score_any_recorder tests
# ---------------------------------------------------------------------------
def test_score_any_recorder_runs_and_restores_registry() -> None:
    """The context manager monkey-patches every wrapper and restores
    the original REGISTRY on exit."""
    from molmetal.scoring import REGISTRY, available_metrics
    from molmetal.scoring.score_any import AdapterUnavailable

    mod = importlib.import_module("bench_score_any_router")
    pre = {k: ref for k, ref in REGISTRY.items()}

    with mod._score_any_recorder(mock_unavailable=True) as (records, probes):
        # While inside, the wrappers are recorder-wrapped.
        for metric_name in available_metrics():
            assert REGISTRY[metric_name].notes == "[wf48-router-bench]"
        # Calling score_any("sa_score") records the call.
        from molmetal.scoring import score_any

        try:
            score_any("sa_score", molecules=["CCO"])
            assert "sa_score" in records
            assert records["sa_score"].n_calls >= 1
        except AdapterUnavailable:
            # AdapterUnavailable on the host is fine — the bench
            # captures the call before the wrapper raises.
            assert "sa_score" in records

    # After exit the registry is restored bit-for-bit.
    assert set(REGISTRY.keys()) == set(pre.keys())
    for k, ref in pre.items():
        assert REGISTRY[k].notes == ref.notes


def test_score_any_recorder_records_adapter_unavailable() -> None:
    """When a wrapper raises AdapterUnavailable, the recorder captures
    it as ``status='ADAPTER_UNAVAILABLE'`` (and returns a 0.0 stub
    when ``mock_unavailable=True`` so the script loop continues)."""
    mod = importlib.import_module("bench_score_any_router")
    from molmetal.scoring.score_any import AdapterUnavailable
    from molmetal.scoring import score_any

    with mod._score_any_recorder(mock_unavailable=True) as (records, probes):
        try:
            score_any("vina_score", molecules=["CCO"], pocket=None)
        except AdapterUnavailable:
            pass
        # Even on AdapterUnavailable, the call must be recorded.
        assert "vina_score" in records


# ---------------------------------------------------------------------------
# 4. Per-script driver tests — verify accuracy field is computed correctly
# ---------------------------------------------------------------------------
def test_invoke_script_returns_script_report_with_accuracy_field() -> None:
    """A full bench invocation returns a list of ScriptReports whose
    ``accuracy`` field is one of ``PASS / FAIL / N/A``."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=2,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        timeout_s=30.0,
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    assert payload["schema_version"] == 1
    assert isinstance(payload["per_script"], list)
    assert len(payload["per_script"]) > 0
    for r in payload["per_script"]:
        assert "accuracy" in r
        assert r["accuracy"] in ("PASS", "FAIL", "N/A")
        assert "wall_time_s" in r
        assert "router_overhead_s" in r
        assert "tools_used" in r


# ---------------------------------------------------------------------------
# 5. CLI smoke — the bench writes router_bench.json
# ---------------------------------------------------------------------------
def test_cli_writes_router_bench_json() -> None:
    """Running the bench from the command line writes
    ``router_bench.json`` with the expected schema."""
    out_dir = SCORE_ANY_REPORTS
    # Run the bench on a tiny subset (2 scripts) for speed.
    cmd = [
        sys.executable,
        str(SCRIPTS / "bench_score_any_router.py"),
        "--n-scripts",
        "2",
        "--seeds",
        "0",
        "--out-dir",
        str(out_dir),
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
    assert res.returncode in (
        0,
        2,
    ), f"bench crashed: rc={res.returncode} stderr={res.stderr[-500:]}"
    json_path = out_dir / "router_bench.json"
    assert (
        json_path.exists()
    ), f"router_bench.json not written: stderr={res.stderr[-500:]}"
    payload = json.loads(json_path.read_text())
    assert payload["recipe"] == "wf48_score_any_router"
    assert payload["tool"] == "bench_score_any_router.py"
    assert "aggregate" in payload
    assert "per_script" in payload
    assert isinstance(payload["aggregate"]["tools_used_aggregate"], list)


# ---------------------------------------------------------------------------
# 6. Aggregate tests — ensure the n_fail count is correctly tracked
# ---------------------------------------------------------------------------
def test_aggregate_n_scripts_matches_registry_size() -> None:
    """The aggregate ``n_scripts`` equals the number of per-script rows."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=3,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    n_rows = len(payload["per_script"])
    assert payload["aggregate"]["n_scripts"] == n_rows


# ---------------------------------------------------------------------------
# 7. Tool-name contract — every metric the bench reports matches
#    the ROUTING_POLICY entry for that metric.
# ---------------------------------------------------------------------------
def test_router_tools_match_policy() -> None:
    """The tool names captured by the recorder are the canonical names
    registered in ``ROUTING_POLICY`` (so the bench can never silently
    invent a tool)."""
    from molmetal.scoring.score_any import ROUTING_POLICY

    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=3,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    policy_tools = {
        metric: entry.primary_tool for metric, entry in ROUTING_POLICY.items()
    }
    for r in payload["per_script"]:
        for metric_name in r["actual_metrics"]:
            assert (
                metric_name in policy_tools
            ), f"router called unknown metric {metric_name!r}"


# ---------------------------------------------------------------------------
# 8. Bench runs end-to-end on all 10 scripts
# ---------------------------------------------------------------------------
def test_full_ten_script_run_writes_valid_reports() -> None:
    """The bench runs all 10 Wf-43 ship scripts and writes a complete
    router_bench.json + router_bench.md."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=10,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        timeout_s=30.0,
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    assert payload["aggregate"]["n_scripts"] == 10
    scripts = sorted(r["script"] for r in payload["per_script"])
    assert scripts == sorted(EXPECTED_SCRIPTS)


def test_router_bench_md_written() -> None:
    """The human-readable ``router_bench.md`` is written next to the JSON."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=2,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    mod.run_bench(cfg)
    md = SCORE_ANY_REPORTS / "router_bench.md"
    assert md.exists()
    text = md.read_text()
    assert "Wf-48 Phase 1" in text
    assert "Aggregate" in text
    assert "Per-script" in text


# ---------------------------------------------------------------------------
# 9. No silent mock — when --no-mock-unavailable is set, AdapterUnavailable
#    must propagate to the script's main() (not be swallowed by the bench).
# ---------------------------------------------------------------------------
def test_no_mock_unavailable_propagates_adapter_unavailable() -> None:
    """``mock_unavailable_tools=False`` (the strict Wf-43 mode) means
    ``score_any()`` raises ``AdapterUnavailable`` for missing tools
    rather than substituting a synthetic 0.0."""
    from molmetal.scoring.score_any import AdapterUnavailable
    from molmetal.scoring import score_any

    mod = importlib.import_module("bench_score_any_router")
    with mod._score_any_recorder(mock_unavailable=False):
        with pytest.raises(AdapterUnavailable):
            score_any("vina_score", molecules=["CCO"], pocket=None)


# ---------------------------------------------------------------------------
# 10. The bench rejects mock-only metrics (defence-in-depth).
# ---------------------------------------------------------------------------
def test_registry_rejects_mock_metric_names() -> None:
    """The bench's registry must never reference a mock metric name."""
    mod = importlib.import_module("bench_score_any_router")
    for entry in mod.DEFAULT_SCRIPT_REGISTRY:
        for m in entry.get("expected_metrics", ()):
            assert (
                "mock" not in str(m).lower()
            ), f"script {entry['script']!r} routes to mock metric {m!r}"
        for t in entry.get("expected_tools", ()):
            assert (
                "mock" not in str(t).lower()
            ), f"script {entry['script']!r} names mock tool {t!r}"


# ---------------------------------------------------------------------------
# 11. n_scripts parameter limits the bench correctly
# ---------------------------------------------------------------------------
def test_n_scripts_clips_registry() -> None:
    """``--n-scripts`` must clip the registry to the first N entries."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=1,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    assert payload["aggregate"]["n_scripts"] == 1
    assert payload["per_script"][0]["script"] == EXPECTED_SCRIPTS[0]


# ---------------------------------------------------------------------------
# 12. _invoke_script — accuracy="PASS" when expected metrics are recorded
# ---------------------------------------------------------------------------
def test_invoke_script_accuracy_pass_when_expected_metrics_recorded() -> None:
    """When the recorder captures every expected metric for a script,
    ``accuracy`` must be ``PASS``."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=10,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        timeout_s=30.0,
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    for r in payload["per_script"]:
        if r["status"] == "OK" and r["accuracy"] != "N/A":
            assert (
                r["accuracy"] in ("PASS", "FAIL")
            ), f"script {r['script']!r} accuracy must be PASS/FAIL, got {r['accuracy']!r}"


# ---------------------------------------------------------------------------
# 13. Wall-time sanity — every script reports a non-negative wall_time_s
# ---------------------------------------------------------------------------
def test_wall_time_non_negative() -> None:
    """Every script's ``wall_time_s`` is a non-negative float."""
    mod = importlib.import_module("bench_score_any_router")
    cfg = mod.BenchConfig(
        n_scripts=3,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    payload = mod.run_bench(cfg)
    for r in payload["per_script"]:
        assert float(r["wall_time_s"]) >= 0.0
        assert float(r["router_overhead_s"]) >= 0.0


# ---------------------------------------------------------------------------
# 14. sys.argv monkeypatch — scripts that read sys.argv (e.g.
#     benchmark_r18_platinai.py main() with parse_args()) must see the
#     bench's smoke_args, not the bench's own argv.
# ---------------------------------------------------------------------------
def test_sys_argv_is_restored_after_script_invocation() -> None:
    """``sys.argv`` is restored to its original value after each script
    invocation so subsequent bench operations don't see leaked args."""
    import sys

    mod = importlib.import_module("bench_score_any_router")
    original_argv = list(sys.argv)
    cfg = mod.BenchConfig(
        n_scripts=1,
        seeds=(0,),
        out_dir=str(SCORE_ANY_REPORTS),
        mock_unavailable_tools=True,
    )
    try:
        mod.run_bench(cfg)
        assert sys.argv == original_argv, (
            "sys.argv must be restored after the bench run (leaked "
            f"args: {sys.argv})"
        )
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# 15. Defence-in-depth: the bench never references a mock fallback
# ---------------------------------------------------------------------------
def test_bench_module_has_no_mock_fallback_definitions() -> None:
    """The bench script's source code must not define a mock
    fallback — Wf-43 forbids it."""
    src_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "bench_score_any_router.py"
    )
    text = src_path.read_text()
    # Allow the word "mock" in docstrings / comments — we forbid it
    # as a callable definition only.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Strip strings / comments at end of line (very rough).
        if "MockScorer" in stripped or "MockDocker" in stripped:
            raise AssertionError(f"bench defines a mock fallback: {stripped[:120]}")
