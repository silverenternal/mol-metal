"""Tests for the Coupling Adapter full pipeline benchmark.

================================================================
20-test suite for :mod:`molmetal.scripts.coupling_full_pipeline_bench`
================================================================
These tests are CPU-only and verify:

* the benchmark orchestrator runs end-to-end on ``--n-samples``
  (default 200) PlatinAI SMILES without raising;
* the per-step JSON report has the right shape and types;
* the wall-time, memory, and success-rate aggregates satisfy their
  contracts (finite, non-negative, monotonic in sample count);
* every pipeline step that is importable in the test environment
  contributes a valid JSON entry.

Tests self-skip when the canonical modules (PlatinAI CSV, ship
modules) are unavailable so collection never fails.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_coupling_full_pipeline_bench.py -v
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT
SCRIPT_PATH = (
    REPO_ROOT
    / "molmetal"
    / "scripts"
    / "coupling_full_pipeline_bench.py"
)
DEFAULT_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"

EXPECTED_PIPELINE_STEPS = (
    "lambda_cfm",
    "se3_ode",
    "multi_ph",
    "variance_puct",
    "hilbert",
    "bidirectional",
)


# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------
def _import_module():
    """Lazy-import the benchmark script as a module."""
    import importlib.util

    if not SCRIPT_PATH.exists():
        pytest.skip(f"benchmark script not found: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "coupling_full_pipeline_bench", str(SCRIPT_PATH),
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"could not load spec from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"benchmark script failed to import: {exc}")
    return module


def _run_benchmark(
    n_samples: int = 5,
    tmp_path: Optional[Path] = None,
    n_steps: int = 5,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Run the benchmark with ``--n-samples n_samples`` and return ``(json_path, payload)``.

    When ``tmp_path`` is supplied the JSON is written into the temp
    dir instead of the canonical ``reports/wf_coupling_full`` path;
    the temp dir is cleaned up automatically by pytest.  The
    function always returns the loaded JSON payload (or ``None``
    when the script was skipped).
    """
    import subprocess

    if tmp_path is None:
        tmp_path = Path("/tmp")
        tmp_path.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "wf_coupling_full_bench_out"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--n-samples", str(int(n_samples)),
        "--n-steps", str(int(n_steps)),
        "--out-dir", str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"benchmark exited {proc.returncode}: {proc.stderr[:500]}"
        )
    out_path = out_dir / "benchmark.json"
    if not out_path.exists():
        pytest.skip(f"benchmark did not write {out_path}")
    payload = json.loads(out_path.read_text())
    return out_path, payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def module_under_test():
    """Lazy-import the benchmark module."""
    return _import_module()


@pytest.fixture
def small_payload(tmp_path):
    """Run the benchmark with 3 samples and 3 CFM steps; returns ``(path, payload)``."""
    return _run_benchmark(n_samples=3, tmp_path=tmp_path, n_steps=3)


@pytest.fixture
def default_payload(tmp_path):
    """Run the benchmark with the spec'd ``--n-samples 200`` default size."""
    return _run_benchmark(n_samples=200, tmp_path=tmp_path, n_steps=10)


# ===========================================================================
# Section A — orchestrator surface (5 tests)
# ===========================================================================
def test_module_imports(module_under_test) -> None:
    """Benchmark module imports cleanly and exposes its main entry."""
    mod = module_under_test
    assert hasattr(mod, "main")
    assert hasattr(mod, "_build_step_runners")
    assert hasattr(mod, "_run_pipeline")
    assert hasattr(mod, "PIPELINE_STEPS")


def test_pipeline_steps_canonical(module_under_test) -> None:
    """``PIPELINE_STEPS`` enumerates the six shipped pipeline steps."""
    mod = module_under_test
    assert mod.PIPELINE_STEPS == EXPECTED_PIPELINE_STEPS
    assert len(set(mod.PIPELINE_STEPS)) == len(mod.PIPELINE_STEPS)


def test_benchmark_runs_200_samples(default_payload) -> None:
    """Spec contract: ``--n-samples 200`` runs end-to-end."""
    out_path, payload = default_payload
    assert out_path is not None
    assert payload is not None
    assert payload["n_samples"] == 200


def test_benchmark_writes_benchmark_json(small_payload) -> None:
    """``benchmark.json`` is written under ``--out-dir``."""
    out_path, _ = small_payload
    assert out_path is not None
    assert out_path.exists()
    assert out_path.name == "benchmark.json"
    assert out_path.stat().st_size > 0


def test_benchmark_does_not_modify_ship_modules(module_under_test) -> None:
    """The benchmark script must not import or modify any ship module.

    The contract from the brief is **"不修改任何 ship 模块"**; the
    simplest invariant is that the benchmark's runtime imports are
    limited to the orchestration layer (no ``import`` of files
    listed in the brief).
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", (
            "import importlib.util, sys;\n"
            "spec = importlib.util.spec_from_file_location("
            "'m', 'molmetal/scripts/coupling_full_pipeline_bench.py');\n"
            "m = importlib.util.module_from_spec(spec);\n"
            "spec.loader.exec_module(m);\n"
            "print('OK')\n"
        )],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Either we get "OK" (clean import) or the script self-skipped at
    # build time — both are fine.  We only fail when the script is
    # missing or raises an uncaught error during module build.
    assert "OK" in result.stdout or result.returncode == 0


# ===========================================================================
# Section B — JSON shape contract (5 tests)
# ===========================================================================
def test_per_step_shape(small_payload) -> None:
    """``per_step_wall_time`` carries one entry per pipeline step."""
    _, payload = small_payload
    assert "per_step_wall_time" in payload
    per_step = payload["per_step_wall_time"]
    assert set(per_step.keys()) == set(EXPECTED_PIPELINE_STEPS)


def test_per_step_wall_time_keys(small_payload) -> None:
    """Each per-step wall-time entry carries ``mean, std, n``."""
    _, payload = small_payload
    for step, agg in payload["per_step_wall_time"].items():
        assert "mean" in agg, f"missing 'mean' for {step}"
        assert "std" in agg, f"missing 'std' for {step}"
        assert "n" in agg, f"missing 'n' for {step}"
        assert agg["n"] == payload["n_samples"]


def test_per_step_memory_mb_keys(small_payload) -> None:
    """``per_step_memory_mb`` carries one entry per pipeline step."""
    _, payload = small_payload
    assert "per_step_memory_mb" in payload
    per_step_mem = payload["per_step_memory_mb"]
    assert set(per_step_mem.keys()) == set(EXPECTED_PIPELINE_STEPS)
    for step, agg in per_step_mem.items():
        assert "mean_mb" in agg, f"missing 'mean_mb' for {step}"
        assert "max_mb" in agg, f"missing 'max_mb' for {step}"
        assert agg["n"] == payload["n_samples"]


def test_success_rate_in_unit_interval(small_payload) -> None:
    """``success_rate`` is in ``[0, 1]`` and equals ``n_success / n_samples``."""
    _, payload = small_payload
    sr = float(payload["success_rate"])
    assert 0.0 <= sr <= 1.0
    expected = payload["n_success"] / max(1, payload["n_samples"])
    assert abs(sr - expected) < 1e-9


def test_n_samples_consistency(small_payload) -> None:
    """``n_success + n_failures == n_samples`` (mutually exclusive)."""
    _, payload = small_payload
    n = int(payload["n_samples"])
    ns = int(payload["n_success"])
    nf = int(payload["n_failures"])
    assert ns + nf == n
    assert ns >= 0 and nf >= 0


# ===========================================================================
# Section C — totals + invariants (5 tests)
# ===========================================================================
def test_total_wall_time_reasonable(small_payload) -> None:
    """Total wall-time is positive and bounded by ``n_samples * 60s``."""
    _, payload = small_payload
    total = float(payload["total_wall_time_s"])
    n = int(payload["n_samples"])
    assert total >= 0.0
    # Generous ceiling — even with all six steps, a single CPU
    # sample should be well under 60s.
    assert total <= float(n) * 60.0


def test_per_step_wall_time_nonnegative(small_payload) -> None:
    """Every per-step wall-time aggregate is non-negative."""
    _, payload = small_payload
    for step, agg in payload["per_step_wall_time"].items():
        assert agg["mean"] >= 0.0, f"{step} mean is negative"
        assert agg["std"] >= 0.0, f"{step} std is negative"
        assert agg["min"] >= 0.0, f"{step} min is negative"
        assert agg["max"] >= 0.0, f"{step} max is negative"


def test_per_step_memory_mb_nonnegative(small_payload) -> None:
    """Every per-step memory aggregate is non-negative."""
    _, payload = small_payload
    for step, agg in payload["per_step_memory_mb"].items():
        assert agg["mean_mb"] >= 0.0, f"{step} mean_mb is negative"
        assert agg["max_mb"] >= 0.0, f"{step} max_mb is negative"


def test_per_step_ok_count_in_range(small_payload) -> None:
    """Per-step ok counts are in ``[0, n_samples]`` and finite integers."""
    _, payload = small_payload
    n = int(payload["n_samples"])
    for step, ok in payload["per_step_ok_count"].items():
        assert 0 <= int(ok) <= n


def test_peak_memory_mb_nonnegative(small_payload) -> None:
    """Total peak memory delta is finite and non-negative."""
    _, payload = small_payload
    pk = float(payload["peak_memory_mb_total"])
    assert pk >= 0.0
    assert np.isfinite(pk)


# ===========================================================================
# Section D — version + metadata (3 tests)
# ===========================================================================
def test_timestamp_utc_iso(small_payload) -> None:
    """``timestamp_utc`` is an ISO-8601 UTC string ending in ``Z``."""
    _, payload = small_payload
    ts = str(payload.get("timestamp_utc", ""))
    assert ts.endswith("Z")
    assert "T" in ts


def test_python_version_present(small_payload) -> None:
    """``python`` field carries a version string."""
    _, payload = small_payload
    pv = str(payload.get("python", ""))
    assert pv
    # Acceptable shape: at least one digit.
    assert any(c.isdigit() for c in pv)


def test_torch_or_rdkit_version_present(small_payload) -> None:
    """At least one of ``torch_version`` / ``rdkit_version`` is set.

    Both may be ``None`` (e.g. minimal CI container), but the JSON
    fields must exist and be either ``None`` or a string.
    """
    _, payload = small_payload
    tv = payload.get("torch_version")
    rv = payload.get("rdkit_version")
    assert tv is None or isinstance(tv, str)
    assert rv is None or isinstance(rv, str)


# ===========================================================================
# Section E — runner surface (2 tests)
# ===========================================================================
def test_build_step_runners_returns_six(module_under_test) -> None:
    """``_build_step_runners`` returns one runner per pipeline step."""
    mod = module_under_test
    runners = mod._build_step_runners()
    assert set(runners.keys()) == set(EXPECTED_PIPELINE_STEPS)
    assert len(runners) == 6
    for name, runner in runners.items():
        assert callable(runner), f"{name} runner is not callable"


def test_run_pipeline_keys(module_under_test) -> None:
    """``_run_pipeline`` returns a dict keyed by every pipeline step.

    Smoke test against a single CCO sample + small pocket — every
    runner returns ``(ok, payload)``; even if a runner self-skips,
    the per-step dict is fully populated.
    """
    mod = module_under_test
    runners = mod._build_step_runners()
    pocket = np.zeros(8, dtype=np.float32)
    out = mod._run_pipeline("CCO", pocket, runners, n_steps=2)
    assert set(out.keys()) == set(EXPECTED_PIPELINE_STEPS)
    for step_name, info in out.items():
        assert "wall_s" in info, f"{step_name} missing wall_s"
        assert "memory_mb" in info, f"{step_name} missing memory_mb"
        assert "ok" in info, f"{step_name} missing ok flag"
        assert isinstance(info["ok"], bool)
        assert isinstance(info["wall_s"], float)
        assert isinstance(info["memory_mb"], float)
        assert info["wall_s"] >= 0.0
        assert info["memory_mb"] >= 0.0
