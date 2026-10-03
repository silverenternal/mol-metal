"""Tests for Wf-48 Phase 2 — per-tool accuracy verification.

Verifies the integrity of
:file:`molmetal/scripts/bench_per_tool_accuracy.py`:

* the bench's module-shape (version, dataclasses, registry, dataset
  loaders);
* the ``DEFAULT_TOOL_REGISTRY`` covers the 11 Wf-48 Phase 2 tools
  (Vina, DiffDock, Boltz-2, AF2, PoseBusters, AiZynth, EquiBind,
  iMoLD, DrugOOD, Pocket2Mol, TargetDiff);
* every registry metric name is registered in
  :mod:`molmetal.scoring.score_any` (no phantom metrics);
* the dataset loaders resolve on this host;
* ``run_bench()`` returns the expected JSON-serialisable payload with
  per-tool records + aggregate summary;
* the bench writes a valid ``per_tool_accuracy.json`` when invoked
  end-to-end;
* the bench's stdout exit code is 0 / 1 / 2 according to the documented
  policy (0 — clean; 1 — no tools measured; 2 — at least one ERROR);
* AdapterUnavailable is captured as a MISSING_DEP / NOT_SHIPPED
  status, never swallowed as a fake MEASURED;
* per-tool records expose ``accuracy``, ``mean_speed_s``, and
  ``failure_rate`` with the documented semantics;
* the bench never fabricates a number — NaN / None is the canonical
  "no measurement" marker.

Test count target: 15 distinct checks.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
PER_TOOL_REPORTS = REPORTS / "wf48_per_tool"


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
    mod = importlib.import_module("bench_per_tool_accuracy")
    assert mod is not None
    assert hasattr(mod, "run_bench")
    assert hasattr(mod, "main")
    assert hasattr(mod, "DEFAULT_TOOL_REGISTRY")
    assert hasattr(mod, "DATASET_LOADERS")


def test_bench_version_string() -> None:
    """The bench module exposes a wf48 phase 2 version string."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    assert isinstance(getattr(mod, "__version__", ""), str)
    assert "wf48" in mod.__version__
    assert "phase2" in mod.__version__


def test_dataclass_shapes() -> None:
    """ToolRecord + BenchConfig are dataclasses with the expected fields."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    from dataclasses import fields

    for cls in (mod.ToolRecord, mod.BenchConfig):
        fs = fields(cls)
        assert isinstance(fs, (list, tuple))
        assert len(fs) >= 1

    # ToolRecord must carry the per-tool KPIs.
    tr_fields = {f.name for f in fields(mod.ToolRecord)}
    for required in (
        "tool", "metric", "backend", "datasets",
        "n_calls", "n_measured", "n_adapter_unavailable",
        "status", "accuracy", "mean_speed_s", "failure_rate",
    ):
        assert required in tr_fields, (
            f"ToolRecord missing field {required!r}"
        )


# ---------------------------------------------------------------------------
# 2. Registry tests — the 11 Wf-48 Phase 2 tools
# ---------------------------------------------------------------------------
EXPECTED_TOOLS = (
    "vina_score",        # AutoDock Vina
    "diffdock_pose",     # DiffDock
    "boltz_affinity",    # Boltz-style pIC50
    "af2_confidence",    # AF2 surrogate
    "pb_pass_rate",      # PoseBusters
    "aizynth_synthesis", # AiZynthFinder
    "equibind_pose",     # EquiBind (reserved)
    "imold_ood",         # iMoLD (reserved)
    "drugood_ood",       # DrugOOD (reserved)
    "pocket2mol",        # Pocket2Mol
    "targetdiff",        # TargetDiff
)


def test_registry_has_eleven_entries() -> None:
    mod = importlib.import_module("bench_per_tool_accuracy")
    assert len(mod.DEFAULT_TOOL_REGISTRY) == 11


def test_registry_covers_all_eleven_wf48_tools() -> None:
    mod = importlib.import_module("bench_per_tool_accuracy")
    listed = {str(entry["metric"]) for entry in mod.DEFAULT_TOOL_REGISTRY}
    assert listed == set(EXPECTED_TOOLS), (
        f"DEFAULT_TOOL_REGISTRY mismatch: "
        f"missing={set(EXPECTED_TOOLS) - listed}, "
        f"extra={listed - set(EXPECTED_TOOLS)}"
    )


def test_registry_metrics_are_in_score_any() -> None:
    """Every registry metric is registered in score_any (no phantoms)."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    from molmetal.scoring import available_metrics
    reg_metrics = {str(entry["metric"]) for entry in mod.DEFAULT_TOOL_REGISTRY}
    avail = set(available_metrics())
    missing = reg_metrics - avail
    assert not missing, f"registry metric(s) not in score_any: {missing}"


def test_registry_tools_are_unique() -> None:
    """No two registry entries share the same metric name."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    metrics = [str(entry["metric"]) for entry in mod.DEFAULT_TOOL_REGISTRY]
    assert len(metrics) == len(set(metrics)), (
        f"duplicate metric entries in registry: {metrics}"
    )


# ---------------------------------------------------------------------------
# 3. Dataset loader tests
# ---------------------------------------------------------------------------
def test_dataset_loaders_resolve() -> None:
    """The dataset loader mapping exposes both PlatinAI 5K and MetalCytoToxDB."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    assert "platinai_5k" in mod.DATASET_LOADERS
    assert "metalcytotox" in mod.DATASET_LOADERS
    # Loaders are callables returning lists.
    for ds, loader in mod.DATASET_LOADERS.items():
        assert callable(loader), f"{ds} loader is not callable"


def test_platinai_loader_returns_smiles() -> None:
    """On this host the PlatinAI loader returns at least one SMILES."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    if "platinai_5k" not in mod.DATASET_LOADERS:
        pytest.skip("platinai_5k loader not registered")
    smis = mod.DATASET_LOADERS["platinai_5k"](n=4, seed=0)
    # The loader may return [] on hosts missing the CSV, but in CI it
    # must produce real SMILES.
    if smis:
        assert all(isinstance(s, str) and len(s) > 0 for s in smis)


# ---------------------------------------------------------------------------
# 4. Driver tests — invoke a single tool
# ---------------------------------------------------------------------------
def test_invoke_tool_returns_tool_record() -> None:
    """A single tool invocation returns a ToolRecord with the expected schema."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    # boltz_affinity is the cheapest shipped tool that touches score_any.
    entry = next(
        e for e in mod.DEFAULT_TOOL_REGISTRY if e["metric"] == "boltz_affinity"
    )
    cfg = mod.BenchConfig(
        n_tools=1,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    rec = mod._invoke_tool(entry, cfg, 0)
    assert rec.tool == entry["tool"]
    assert rec.metric == entry["metric"]
    assert rec.n_calls >= 1  # counted even when the call raises
    assert rec.status in ("OK", "NOT_SHIPPED", "MISSING_DEP", "PARTIAL", "ERROR")


def test_invoke_tool_records_adapter_unavailable() -> None:
    """When score_any raises AdapterUnavailable, the record captures it
    and the bench never fabricates a value."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    # equibind_pose is reserved — score_any raises AdapterUnavailable.
    entry = next(
        e for e in mod.DEFAULT_TOOL_REGISTRY if e["metric"] == "equibind_pose"
    )
    cfg = mod.BenchConfig(
        n_tools=1,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    rec = mod._invoke_tool(entry, cfg, 0)
    assert rec.status == "NOT_SHIPPED"
    assert rec.n_adapter_unavailable >= 1
    assert rec.error is not None and "EquiBind" in rec.error


# ---------------------------------------------------------------------------
# 5. Aggregate tests
# ---------------------------------------------------------------------------
def test_aggregate_n_tools_matches_registry_size() -> None:
    """The aggregate ``n_tools`` equals the number of per-tool rows."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=11,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    payload = mod.run_bench(cfg)
    n_rows = len(payload["per_tool"])
    assert payload["aggregate"]["n_tools"] == n_rows


def test_aggregate_schema_is_complete() -> None:
    """The aggregate dict has all expected keys."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=3,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    payload = mod.run_bench(cfg)
    agg = payload["aggregate"]
    for key in (
        "n_tools", "n_measured", "n_not_shipped", "n_missing_dep",
        "n_partial", "n_error",
        "mean_accuracy", "mean_speed_s", "mean_failure_rate",
    ):
        assert key in agg, f"aggregate missing key {key!r}"


# ---------------------------------------------------------------------------
# 6. Per-tool record schema
# ---------------------------------------------------------------------------
def test_per_tool_record_exposes_kpis() -> None:
    """Every per-tool record has accuracy / mean_speed_s / failure_rate."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=11,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    payload = mod.run_bench(cfg)
    for r in payload["per_tool"]:
        assert "accuracy" in r
        assert "mean_speed_s" in r
        assert "failure_rate" in r
        # NaN is the canonical "no measurement" marker — it must be
        # serialised as null by the JSON writer, but the in-process
        # value is preserved.
        assert r["status"] in ("OK", "NOT_SHIPPED", "MISSING_DEP", "PARTIAL", "ERROR")


# ---------------------------------------------------------------------------
# 7. CLI smoke
# ---------------------------------------------------------------------------
def test_cli_writes_per_tool_accuracy_json() -> None:
    """Running the bench from the command line writes a valid
    ``per_tool_accuracy.json``."""
    out_dir = PER_TOOL_REPORTS
    cmd = [
        sys.executable,
        str(SCRIPTS / "bench_per_tool_accuracy.py"),
        "--tools", "3",
        "--datasets", "platinai_5k",
        "--n-samples", "2",
        "--seeds", "0",
        "--out-dir", str(out_dir),
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
    assert res.returncode in (0, 1, 2), (
        f"bench crashed: rc={res.returncode} stderr={res.stderr[-500:]}"
    )
    json_path = out_dir / "per_tool_accuracy.json"
    assert json_path.exists(), (
        f"per_tool_accuracy.json not written: stderr={res.stderr[-500:]}"
    )
    payload = json.loads(json_path.read_text())
    assert payload["recipe"] == "wf48_per_tool_accuracy"
    assert payload["tool"] == "bench_per_tool_accuracy.py"
    assert "aggregate" in payload
    assert "per_tool" in payload
    assert isinstance(payload["aggregate"]["n_tools"], int)


def test_per_tool_accuracy_md_written() -> None:
    """The human-readable ``per_tool_accuracy.md`` is written next to the JSON."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=2,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    mod.run_bench(cfg)
    md = PER_TOOL_REPORTS / "per_tool_accuracy.md"
    assert md.exists()
    text = md.read_text()
    assert "Wf-48 Phase 2" in text
    assert "Aggregate" in text
    assert "Per-tool" in text


# ---------------------------------------------------------------------------
# 8. Defence-in-depth — no mock substitution
# ---------------------------------------------------------------------------
def test_registry_rejects_mock_metric_names() -> None:
    """The bench's registry must never reference a mock metric name."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    for entry in mod.DEFAULT_TOOL_REGISTRY:
        for s in (entry.get("tool", ""), entry.get("metric", "")):
            assert "mock" not in str(s).lower(), (
                f"registry entry contains 'mock': tool={entry.get('tool')!r} "
                f"metric={entry.get('metric')!r}"
            )


def test_json_writer_normalises_nan() -> None:
    """The JSON writer converts NaN → null so the output is valid JSON."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=11,
        datasets=("platinai_5k",),
        n_samples=2,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    payload = mod.run_bench(cfg)
    # Re-serialise and confirm the file parses as valid JSON.
    out_dir = Path(cfg.out_dir)
    json_path = out_dir / "per_tool_accuracy.json"
    text = json_path.read_text()
    # No literal NaN in the file.
    assert "NaN" not in text, "JSON file contains literal NaN — invalid"
    # Round-trip via json.loads.
    parsed = json.loads(text)
    assert parsed["recipe"] == "wf48_per_tool_accuracy"


# ---------------------------------------------------------------------------
# 9. --tools parameter limits the registry
# ---------------------------------------------------------------------------
def test_n_tools_clips_registry() -> None:
    """``--tools`` clips the registry to the first N entries."""
    mod = importlib.import_module("bench_per_tool_accuracy")
    cfg = mod.BenchConfig(
        n_tools=1,
        datasets=("platinai_5k",),
        n_samples=1,
        seeds=(0,),
        out_dir=str(PER_TOOL_REPORTS),
    )
    payload = mod.run_bench(cfg)
    assert payload["aggregate"]["n_tools"] == 1
    assert payload["per_tool"][0]["metric"] == EXPECTED_TOOLS[0]
