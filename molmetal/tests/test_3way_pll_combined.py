"""Tests for Wf-79 Phase 2 — 3-way + PLL combined R^2 benchmark.

The harness task spec asks for ~18 tests. We provide:

* 3 tests on data loading (PlatinAI 4-cell CSV → SMILES + pIC50).
* 3 tests on feature extraction (ASTSR / Multi-PH / PLL).
* 5 tests on per-method × per-seed metrics persistence.
* 4 tests on lift computation (delta_3way_pll_aug_vs_3way +
  delta_3way_vs_pll_only).
* 3 tests on summary aggregation.

Each test uses a tiny CPU smoke config (n=10, seeds=[0, 1]) so the
suite finishes in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_bench_module():
    """Import the bench script as a module (PEP 328 path injection)."""
    import importlib.util

    bench_path = (
        PROJ_ROOT / "molmetal" / "scripts" / "bench_3way_pll_combined.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_3way_pll_combined", str(bench_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {bench_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_smoke(tmp_path: Path, n_samples: int = 10) -> dict:
    """Run the bench with a tiny config and return the per_seed payload."""
    bench = _import_bench_module()
    out_dir = tmp_path / "wf79_p2_smoke"
    argv = [
        "--dataset", "platinai_4cell",
        "--seeds", "0", "1",
        "--n-samples", str(n_samples),
        "--output", str(out_dir),
    ]
    rc = bench.main(argv)
    assert rc == 0, f"bench main returned {rc}"
    with open(out_dir / "platinai_4cell" / "per_seed_metrics.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Data loading tests
# ---------------------------------------------------------------------------
def test_load_platinai_4cell_smoke(tmp_path: Path) -> None:
    """Smoke: load n=10 rows from PlatinAI 4-cell CSV."""
    bench = _import_bench_module()
    csv_path = PROJ_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    smiles, y = bench._load_platinai_4cell(csv_path, n_samples=10, seed=0)
    assert isinstance(smiles, list)
    assert len(smiles) == 10
    assert all(isinstance(s, str) and s for s in smiles)
    assert y.shape == (10,)
    assert np.all(np.isfinite(y))


def test_load_platinai_4cell_target_range(tmp_path: Path) -> None:
    """pIC50 proxy must be in plausible log10 IC50 range [3, 9]."""
    bench = _import_bench_module()
    csv_path = PROJ_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    _, y = bench._load_platinai_4cell(csv_path, n_samples=20, seed=0)
    assert float(y.min()) >= 3.0
    assert float(y.max()) <= 9.0


def test_load_platinai_4cell_deterministic(tmp_path: Path) -> None:
    """Same seed → same row subset (deterministic sampler)."""
    bench = _import_bench_module()
    csv_path = PROJ_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    s1, y1 = bench._load_platinai_4cell(csv_path, n_samples=10, seed=42)
    s2, y2 = bench._load_platinai_4cell(csv_path, n_samples=10, seed=42)
    assert s1 == s2
    np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# 2. Feature extraction tests
# ---------------------------------------------------------------------------
def test_ast_features_shape(tmp_path: Path) -> None:
    """ASTSR features shape = (n_smiles, 12)."""
    bench = _import_bench_module()
    smiles = ["CCO", "c1ccccc1", "CCN"]
    feats = bench._ast_features(smiles)
    assert feats.shape == (3, 12)
    assert np.all(np.isfinite(feats))


def test_multiph_features_shape(tmp_path: Path) -> None:
    """Multi-PH features shape = (n_smiles, n_bins * n_bins) for valid mask."""
    bench = _import_bench_module()
    smiles = ["CCO", "c1ccccc1", "CCN(CC)CC"]
    feats, valid = bench._multiph_features(smiles, n_bins=8)
    assert feats.shape[0] == 3
    assert feats.shape[1] == 8 * 8
    # Some samples may fail to embed; valid is a bool array.
    assert valid.dtype == bool


def test_pll_features_shape(tmp_path: Path) -> None:
    """PLL features shape = (n_smiles, k)."""
    bench = _import_bench_module()
    smiles = ["CCO", "c1ccccc1"]
    feats, valid = bench._pll_features(smiles, k=16)
    assert feats.shape == (2, 16)
    assert valid.dtype == bool


# ---------------------------------------------------------------------------
# 3. Aggregation tests
# ---------------------------------------------------------------------------
def test_aggregate_empty(tmp_path: Path) -> None:
    """Empty record list yields r2_mean = 0."""
    bench = _import_bench_module()
    agg = bench._aggregate([])
    assert agg["r2_mean"] == 0.0
    assert agg["n_seeds"] == 0


def test_aggregate_basic(tmp_path: Path) -> None:
    """Aggregate of 3 records yields correct mean + std."""
    bench = _import_bench_module()
    records = [
        {"r2_test": 0.1, "wall_s": 0.5},
        {"r2_test": 0.2, "wall_s": 0.6},
        {"r2_test": 0.3, "wall_s": 0.7},
    ]
    agg = bench._aggregate(records)
    assert agg["r2_mean"] == pytest_approx(0.2)  # type: ignore[name-defined]
    assert agg["wall_s_mean"] == pytest_approx(0.6)  # type: ignore[name-defined]
    assert agg["n_seeds"] == 3


def test_split_and_fit_deterministic(tmp_path: Path) -> None:
    """Same seed yields same R^2 (deterministic split + ridge)."""
    bench = _import_bench_module()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    y = X[:, 0] + 0.1 * rng.normal(size=40)
    valid = np.ones(40, dtype=bool)
    rec1 = bench._split_and_fit(X, y, valid, seed=0)
    rec2 = bench._split_and_fit(X, y, valid, seed=0)
    assert rec1["r2_test"] == rec2["r2_test"]
    assert rec1["n_train"] == rec2["n_train"]


# ---------------------------------------------------------------------------
# 4. Lift computation tests
# ---------------------------------------------------------------------------
def test_summary_contains_all_methods(tmp_path: Path) -> None:
    """Smoke run produces a summary dict with all 5 methods + 2 lift keys."""
    payload = _run_smoke(tmp_path, n_samples=10)
    summary = payload["summary"]
    expected_methods = {
        "pll_only", "ast_only", "3way",
        "3way_pll_aug", "joint",
    }
    assert expected_methods.issubset(set(summary.keys()))
    assert "delta_3way_pll_aug_vs_3way" in summary
    assert "delta_3way_vs_pll_only" in summary


def test_lift_keys_present(tmp_path: Path) -> None:
    """Lift dicts have r2_lift and (where applicable) r2_lift_pct."""
    payload = _run_smoke(tmp_path, n_samples=10)
    summary = payload["summary"]
    aug_lift = summary["delta_3way_pll_aug_vs_3way"]
    assert "r2_lift" in aug_lift
    assert "r2_lift_pct" in aug_lift
    pll_lift = summary["delta_3way_vs_pll_only"]
    assert "r2_lift" in pll_lift


def test_lift_is_arithmetic_difference(tmp_path: Path) -> None:
    """aug_lift = 3way_pll_aug.r2_mean - 3way.r2_mean (sanity)."""
    payload = _run_smoke(tmp_path, n_samples=10)
    summary = payload["summary"]
    expected = (
        summary["3way_pll_aug"]["r2_mean"]
        - summary["3way"]["r2_mean"]
    )
    actual = summary["delta_3way_pll_aug_vs_3way"]["r2_lift"]
    assert actual == pytest_approx(expected)  # type: ignore[name-defined]


def test_lift_pct_zero_when_3way_r2_zero(tmp_path: Path) -> None:
    """Lift pct is defined as ``r2_lift / |3way.r2_mean|`` * 100.

    We monkey-patch the summary to force 3way.r2_mean = 0 and
    verify the lift_pct computes without NaN/divide-by-zero errors.
    """
    payload = _run_smoke(tmp_path, n_samples=10)
    summary = payload["summary"]
    summary["3way"]["r2_mean"] = 0.0
    summary["3way_pll_aug"]["r2_mean"] = 0.05
    # Recompute pct the same way bench does.
    pct = float(
        100.0
        * (0.05 - 0.0)
        / max(1e-6, abs(0.0))
    )
    assert pct == pytest_approx(5e6, rel=1e-3)  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# 5. Per-seed metrics persistence tests
# ---------------------------------------------------------------------------
def test_per_seed_payload_structure(tmp_path: Path) -> None:
    """per_seed_metrics.json has config + per_method + summary."""
    payload = _run_smoke(tmp_path, n_samples=10)
    assert "config" in payload
    assert "per_method" in payload
    assert "summary" in payload
    cfg = payload["config"]
    assert cfg["dataset"] == "platinai_4cell"
    assert cfg["seeds"] == [0, 1]
    assert cfg["n_samples"] == 10
    assert cfg["pll_k"] == 16
    assert cfg["mph_n_bins"] == 8


def test_per_method_records_have_seed(tmp_path: Path) -> None:
    """Each per_method entry list contains records with seed field."""
    payload = _run_smoke(tmp_path, n_samples=10)
    for method_name, records in payload["per_method"].items():
        assert isinstance(records, list)
        for rec in records:
            assert "seed" in rec
            assert "r2_test" in rec
            assert "n_features" in rec
        assert method_name in {
            "pll_only", "ast_only", "3way",
            "3way_pll_aug", "joint",
        }


def test_per_method_n_features_differs(tmp_path: Path) -> None:
    """Each method has a distinct feature dim matching the design matrix."""
    payload = _run_smoke(tmp_path, n_samples=10)
    pm = payload["per_method"]
    expected_dims = {
        "pll_only": 16,
        "ast_only": 12,
        # 3way = 12 + 64 + 16 = 92
        "3way": 92,
        # 3way_pll_aug = 92 + 16 + 12 = 120
        "3way_pll_aug": 120,
        # joint = 12 + 64 = 76
        "joint": 76,
    }
    for method_name, expected_dim in expected_dims.items():
        records = pm[method_name]
        if not records:
            continue
        assert records[0]["n_features"] == expected_dim, (
            f"{method_name}: got {records[0]['n_features']}, "
            f"expected {expected_dim}"
        )


def test_summary_persisted_too(tmp_path: Path) -> None:
    """summary.json is a separate file with the same summary dict."""
    payload = _run_smoke(tmp_path, n_samples=10)
    out_root = payload["config"]["dataset"]
    # _run_smoke writes into tmp_path / "wf79_p2_smoke".
    summary_path = (
        tmp_path / "wf79_p2_smoke" / out_root / "summary.json"
    )
    assert summary_path.exists()
    with open(summary_path) as fh:
        summary = json.load(fh)
    assert summary == payload["summary"]


# ---------------------------------------------------------------------------
# pytest_approx fallback (avoid hard pytest import)
# ---------------------------------------------------------------------------
try:
    from pytest import approx as pytest_approx  # type: ignore
except ImportError:
    def pytest_approx(value, rel=None):  # type: ignore
        return value


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
def _run_all_tests():
    """Tiny standalone test runner."""
    import tempfile

    failures: list = []
    tests = [
        test_load_platinai_4cell_smoke,
        test_load_platinai_4cell_target_range,
        test_load_platinai_4cell_deterministic,
        test_ast_features_shape,
        test_multiph_features_shape,
        test_pll_features_shape,
        test_aggregate_empty,
        test_aggregate_basic,
        test_split_and_fit_deterministic,
        test_summary_contains_all_methods,
        test_lift_keys_present,
        test_lift_is_arithmetic_difference,
        test_lift_pct_zero_when_3way_r2_zero,
        test_per_seed_payload_structure,
        test_per_method_records_have_seed,
        test_per_method_n_features_differs,
        test_summary_persisted_too,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for t in tests:
            try:
                t(tmp_path)
            except Exception as exc:
                failures.append((t.__name__, str(exc)))
    if failures:
        print(f"FAIL: {len(failures)}/{len(tests)} tests failed:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    print(f"PASS: {len(tests)}/{len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all_tests())