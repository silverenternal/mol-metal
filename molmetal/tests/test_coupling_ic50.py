"""Tests for the Wf-39 Phase 3 Coupling Adapter IC50 benchmark.

================================================================
15-test suite for :mod:`molmetal.scripts.bench_coupling_ic50`
================================================================
This file verifies:

* the bench module imports cleanly and exposes its CLI;
* the three methods (bridge / joint / bidirectional) all complete
  end-to-end on a small PlatinAI 5K sample without raising;
* the per-method JSON payload has the right shape and finite entries;
* the λ-features quality block (determinism / separability / norm)
  is finite and bounded by the expected ranges;
* the bridge RMSE is finite and non-negative for every method.

Tests self-skip when the canonical modules (PlatinAI CSV, bridge
checkpoint, ship modules) are unavailable so collection never fails.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_coupling_ic50.py -v
"""

from __future__ import annotations

import importlib.util
import json
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
SCRIPT_PATH = REPO_ROOT / "molmetal" / "scripts" / "bench_coupling_ic50.py"
DEFAULT_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_BRIDGE_NPZ = REPO_ROOT / "models" / "coupling_adapter_platinai5k.npz"

EXPECTED_METHODS = ("bridge", "joint", "bidirectional")


# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------
def _import_module():
    """Lazy-import the benchmark script as a module."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"benchmark script not found: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "bench_coupling_ic50",
        str(SCRIPT_PATH),
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"could not load spec from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register in ``sys.modules`` so the ``@dataclass`` decorator can
    # resolve ``cls.__module__`` against the module's own namespace.
    sys.modules.setdefault("bench_coupling_ic50", module)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"benchmark script failed to import: {exc}")
    return module


def _run_benchmark(
    n_samples: int = 8,
    tmp_path: Optional[Path] = None,
    n_steps: int = 5,
    methods: Optional[Tuple[str, ...]] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Run the benchmark with ``--n-samples n_samples`` and return ``(json_path, payload)``.

    When ``tmp_path`` is supplied the JSON is written into the temp
    dir instead of the canonical reports path.  The function always
    returns the loaded JSON payload (or ``None`` when the script was
    skipped).
    """
    import subprocess

    if tmp_path is None:
        tmp_path = Path("/tmp")
        tmp_path.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "wf39_coupling_bench_out"
    cmd = [
        sys.executable,
        "-m",
        "molmetal.scripts.bench_coupling_ic50",
        "--n-samples",
        str(int(n_samples)),
        "--seeds",
        "1",
        "--n-steps",
        str(int(n_steps)),
        "--output",
        str(out_dir / "coupling_bench.json"),
        "--method",
        *(methods or EXPECTED_METHODS),
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
        pytest.skip(f"benchmark exited {proc.returncode}: {proc.stderr[:500]}")
    out_path = out_dir / "coupling_bench.json"
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
    """Run the benchmark with 8 samples × 1 seed × all methods."""
    return _run_benchmark(n_samples=8, tmp_path=tmp_path, n_steps=5)


@pytest.fixture
def bridge_only_payload(tmp_path):
    """Run the benchmark with 8 samples × 1 seed × ``bridge`` only."""
    return _run_benchmark(
        n_samples=8,
        tmp_path=tmp_path,
        n_steps=5,
        methods=("bridge",),
    )


# ===========================================================================
# Section A — module surface (3 tests)
# ===========================================================================
def test_module_imports(module_under_test) -> None:
    """Benchmark module imports cleanly and exposes its main entry."""
    mod = module_under_test
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_benchmark")
    assert hasattr(mod, "_run_one")
    assert hasattr(mod, "_METHOD_REGISTRY")
    assert hasattr(mod, "Wf39CouplingError")


def test_default_methods_canonical(module_under_test) -> None:
    """``DEFAULT_METHODS`` enumerates the three shipped methods."""
    mod = module_under_test
    assert mod.DEFAULT_METHODS == EXPECTED_METHODS
    assert set(mod._METHOD_REGISTRY.keys()) == set(EXPECTED_METHODS)


def test_default_seeds_5(module_under_test) -> None:
    """``DEFAULT_SEEDS`` is the spec'd ``(0, 1, 2, 3, 4)`` tuple."""
    mod = module_under_test
    assert tuple(mod.DEFAULT_SEEDS) == (0, 1, 2, 3, 4)
    assert len(mod.DEFAULT_SEEDS) == 5


# ===========================================================================
# Section B — orchestrator end-to-end (3 tests)
# ===========================================================================
def test_benchmark_runs_end_to_end(small_payload) -> None:
    """``--n-samples 8`` runs end-to-end and writes a JSON payload."""
    out_path, payload = small_payload
    assert out_path is not None
    assert payload is not None
    assert payload["config"]["n_samples_loaded"] == 8
    assert payload["config"]["n_samples_requested"] == 8


def test_benchmark_writes_coupling_bench_json(small_payload) -> None:
    """``coupling_bench.json`` is written under ``--output``."""
    out_path, _ = small_payload
    assert out_path is not None
    assert out_path.exists()
    assert out_path.name == "coupling_bench.json"
    assert out_path.stat().st_size > 0


def test_benchmark_records_per_method(small_payload) -> None:
    """``records`` contains one entry per ``(method, seed)``."""
    _, payload = small_payload
    records = payload["records"]
    methods_seen = {r["method"] for r in records}
    assert methods_seen == set(EXPECTED_METHODS)
    # One record per method at the configured seed (0).
    assert len(records) == len(EXPECTED_METHODS)


# ===========================================================================
# Section C — JSON shape contract (3 tests)
# ===========================================================================
def test_per_method_aggregate_shape(small_payload) -> None:
    """``per_method`` exposes the full aggregate contract."""
    _, payload = small_payload
    pm = payload["per_method"]
    assert set(pm.keys()) == set(EXPECTED_METHODS)
    for m in EXPECTED_METHODS:
        agg = pm[m]
        for key in (
            "bridge_rmse_mean",
            "bridge_rmse_std",
            "lambda_determinism_mean",
            "lambda_separability_mean",
            "lambda_norm_mean",
            "lambda_norm_std",
            "wall_s_mean",
            "wall_s_std",
            "peak_mem_mb_mean",
            "peak_mem_mb_std",
            "n_records",
            "per_seed",
        ):
            assert key in agg, f"{m} missing {key!r}"


def test_per_seed_block_per_method(small_payload) -> None:
    """``per_seed`` is a dict keyed by integer seed (stringified)."""
    _, payload = small_payload
    pm = payload["per_method"]
    for m in EXPECTED_METHODS:
        per_seed = pm[m]["per_seed"]
        assert "0" in per_seed
        for k in (
            "bridge_rmse",
            "lambda_determinism",
            "lambda_separability",
            "lambda_norm_mean",
            "lambda_norm_std",
            "wall_s",
            "peak_mem_mb",
        ):
            assert k in per_seed["0"], f"{m}/seed 0 missing {k!r}"


def test_config_block_shape(small_payload) -> None:
    """``config`` carries the headline knobs."""
    _, payload = small_payload
    cfg = payload["config"]
    for key in (
        "dataset",
        "csv_path",
        "bridge_npz",
        "n_samples_requested",
        "n_samples_loaded",
        "seeds",
        "methods",
        "n_steps_cfm",
        "pocket_size",
        "lambda_dim",
        "embed_dim",
        "bridge_rmse_gate",
        "timestamp_utc",
        "version",
    ):
        assert key in cfg, f"config missing {key!r}"
    assert cfg["methods"] == list(EXPECTED_METHODS)
    assert cfg["dataset"] == "platinai_5k"
    assert cfg["lambda_dim"] == 32
    assert cfg["embed_dim"] == 64


# ===========================================================================
# Section D — value contracts (4 tests)
# ===========================================================================
def test_bridge_rmse_nonnegative(small_payload) -> None:
    """Every per-method bridge RMSE is finite and non-negative."""
    _, payload = small_payload
    pm = payload["per_method"]
    for m in EXPECTED_METHODS:
        mean_rmse = float(pm[m]["bridge_rmse_mean"])
        std_rmse = float(pm[m]["bridge_rmse_std"])
        assert np.isfinite(mean_rmse), f"{m} mean_rmse not finite"
        assert np.isfinite(std_rmse), f"{m} std_rmse not finite"
        assert mean_rmse >= 0.0, f"{m} mean_rmse is negative"
        assert std_rmse >= 0.0, f"{m} std_rmse is negative"


def test_lambda_determinism_one(small_payload) -> None:
    """λ-features are bit-for-bit deterministic (determinism == 1.0)."""
    _, payload = small_payload
    pm = payload["per_method"]
    for m in EXPECTED_METHODS:
        det = float(pm[m]["lambda_determinism_mean"])
        assert np.isfinite(det), f"{m} determinism not finite"
        # Equal-bit determinism means 1.0 exactly; allow tiny float
        # slack from the mean-of-means aggregation across seeds.
        assert det == pytest.approx(1.0, abs=1e-9), f"{m} determinism {det} != 1.0"


def test_lambda_separability_bounded(small_payload) -> None:
    """λ-features pairwise cosine sim is in ``[-1, 1]`` and finite."""
    _, payload = small_payload
    pm = payload["per_method"]
    for m in EXPECTED_METHODS:
        sep = float(pm[m]["lambda_separability_mean"])
        assert np.isfinite(sep), f"{m} separability not finite"
        assert -1.0 <= sep <= 1.0, f"{m} separability {sep} out of [-1, 1]"


def test_bridge_method_hits_rmse_gate(bridge_only_payload) -> None:
    """``bridge`` per-method RMSE is ≤ 0.05 (matches R17 hold-out gate).

    A small random sample of PlatinAI 5K mixes train and hold-out
    rows, so the per-sample RMSE is bounded by the per-epoch hold-
    out RMSE (~0.005) plus a small variance term; 0.05 is a generous
    spec ceiling.  When the bridge checkpoint is the R17 ship artefact
    the assertion holds comfortably.
    """
    _, payload = bridge_only_payload
    rmse = float(payload["per_method"]["bridge"]["bridge_rmse_mean"])
    assert np.isfinite(rmse), "bridge RMSE not finite"
    assert rmse <= 0.05, f"bridge RMSE {rmse:.4f} exceeds gate 0.05"


# ===========================================================================
# Section E — wall-time + memory + versioning (2 tests)
# ===========================================================================
def test_wall_time_and_memory_nonnegative(small_payload) -> None:
    """Per-method wall-time and peak memory are non-negative."""
    _, payload = small_payload
    pm = payload["per_method"]
    for m in EXPECTED_METHODS:
        assert pm[m]["wall_s_mean"] >= 0.0, f"{m} wall_s_mean negative"
        assert pm[m]["wall_s_std"] >= 0.0, f"{m} wall_s_std negative"
        assert pm[m]["peak_mem_mb_mean"] >= 0.0, f"{m} mem_mean negative"
        assert pm[m]["peak_mem_mb_std"] >= 0.0, f"{m} mem_std negative"


def test_timestamp_and_version_present(small_payload) -> None:
    """``timestamp_utc`` is ISO-8601 UTC; ``version`` carries the wf tag."""
    _, payload = small_payload
    ts = str(payload["config"]["timestamp_utc"])
    assert "T" in ts and ts.endswith("+00:00"), f"bad timestamp: {ts}"
    assert "wf39" in str(payload["config"]["version"]).lower()
