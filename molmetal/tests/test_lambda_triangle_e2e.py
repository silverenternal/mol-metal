"""Tests for the Wf-39 L4 triangle closure end-to-end bench.

================================================================
CPU-only contract tests for ``bench_lambda_triangle_end_to_end``
================================================================
Validates the Wf-39 L4 deliverable — the multi-seed PlatinAI-5K
end-to-end bench that extends Wf-37 L8 with REAL metallodrug data,
multi-seed support, and per-stage success-rate reporting.

Test inventory (16 tests)
-------------------------
1. **Constants & defaults**                  (3 tests)
2. **Feature extraction**                    (3 tests)
3. **Data loading & sampling**               (2 tests)
4. **Standardisation helper**                (2 tests)
5. **Single-seed bench**                     (3 tests)
6. **Multi-seed build_report**               (2 tests)
7. **Smoke / IO helpers**                    (3 tests)

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_lambda_triangle_e2e.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from molmetal.scripts import bench_lambda_triangle_end_to_end as bte


# ---------------------------------------------------------------------------
# 1. Constants & defaults
# ---------------------------------------------------------------------------
def test_default_constants_are_well_formed() -> None:
    """Default constants are positive ints / non-empty tuples."""
    assert isinstance(bte.DEFAULT_N_TRAJECTORIES, int)
    assert bte.DEFAULT_N_TRAJECTORIES > 0
    assert isinstance(bte.DEFAULT_N_ROUNDS, int)
    assert bte.DEFAULT_N_ROUNDS > 0
    assert isinstance(bte.DEFAULT_SEEDS, tuple)
    assert len(bte.DEFAULT_SEEDS) >= 1
    assert all(isinstance(s, int) for s in bte.DEFAULT_SEEDS)
    assert len(bte.METAL_LIST) >= 2
    assert all(isinstance(m, str) for m in bte.METAL_LIST)
    assert len(bte.FEATURE_NAMES) == 5


def test_metal_index_mapping_is_canonical() -> None:
    """Each metal in :data:`METAL_LIST` maps to its canonical index."""
    for i, m in enumerate(bte.METAL_LIST):
        assert bte.METAL_TO_IDX[m] == i
    # Unknown metals map to -1 (sentinel).
    assert bte.METAL_TO_IDX.get("Xx", -1) == -1


def test_feature_names_match_extractor_output() -> None:
    """The 5-tuple output of :func:`_smiles_features` matches :data:`FEATURE_NAMES` length."""
    out = bte._smiles_features("CCO", "Pt")
    assert isinstance(out, tuple)
    assert len(out) == len(bte.FEATURE_NAMES)
    assert all(isinstance(v, float) for v in out)


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------
def test_smiles_features_simple_organic() -> None:
    """A simple organic SMILES yields finite 5-D features."""
    smiles = "CCO"  # ethanol
    feats = bte._smiles_features(smiles, "Pt")
    sl, n_heavy, n_branch, n_charge, m_idx = feats
    assert sl == float(len(smiles))
    # Heavy atoms: C, C, O = 3 alphabetic chars.
    assert n_heavy == 3.0
    assert n_branch == 0.0
    assert n_charge == 0.0
    assert m_idx == float(bte.METAL_TO_IDX["Pt"])


def test_smiles_features_with_charges_and_branches() -> None:
    """Charged metallodrug SMILES yields non-zero branch + charge counts."""
    smiles = "CC12CCC([CH-]3->[Pd+2](<-[Cl-]))C2(C)C"
    feats = bte._smiles_features(smiles, "Pd")
    sl, n_heavy, n_branch, n_charge, m_idx = feats
    assert sl == float(len(smiles))
    assert n_branch > 0.0  # Multiple "(" and ")"
    assert n_charge > 0.0  # "+2", "-"
    assert m_idx == float(bte.METAL_TO_IDX["Pd"])


def test_smiles_features_handles_empty_and_garbage() -> None:
    """Empty / non-string / garbage inputs map to safe defaults."""
    # Empty string.
    feats = bte._smiles_features("", "Pt")
    assert feats == (0.0, 0.0, 0.0, 0.0, float(bte.METAL_TO_IDX["Pt"]))
    # None metal — falls back to -1 sentinel.
    feats = bte._smiles_features("CCO", "")
    assert feats[-1] == -1.0


# ---------------------------------------------------------------------------
# 3. Data loading & sampling
# ---------------------------------------------------------------------------
def test_load_platinai_5k_real(tmp_path: Path) -> None:
    """The real PlatinAI 5K CSV loads > 1000 valid rows."""
    csv_path = Path(bte._REPO_ROOT) / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"PlatinAI 5K CSV not found at {csv_path}")
    rows = bte._load_platinai_5k(csv_path)
    assert len(rows) > 1000
    # Every loaded row has a non-empty SMILES.
    assert all(r["smiles"].strip() for r in rows)
    # Every loaded row has a metal field.
    assert all("metal" in r for r in rows)


def test_sample_rows_is_deterministic_and_bounded() -> None:
    """``_sample_rows`` is reproducible and bounded by the input length."""
    rows = [{"smiles": "CCO", "metal": "Pt"}] * 50
    s1 = bte._sample_rows(rows, n=10, seed=0)
    s2 = bte._sample_rows(rows, n=10, seed=0)
    assert [r["smiles"] for r in s1] == [r["smiles"] for r in s2]
    # Different seed → likely different selection (with n=10 < 50).
    s3 = bte._sample_rows(rows, n=10, seed=1)
    # Both runs are subsets of the input.
    assert len(s1) == 10
    assert len(s3) == 10
    # n=0 yields empty list.
    assert bte._sample_rows(rows, n=0, seed=0) == []


# ---------------------------------------------------------------------------
# 4. Standardisation helper
# ---------------------------------------------------------------------------
def test_standardize_centers_and_scales() -> None:
    """Standardised columns have ~zero mean and ~unit std."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)) * 100.0 + 5.0
    X_std, mean, std = bte._standardize(X)
    assert np.allclose(np.mean(X_std, axis=0), 0.0, atol=1e-9)
    assert np.allclose(np.std(X_std, axis=0), 1.0, atol=1e-9)
    # Stored mean / std reproduce the transform.
    X_std_2 = (X - mean) / std
    assert np.allclose(X_std, X_std_2)


def test_standardize_handles_zero_variance_column() -> None:
    """A constant column does not divide by zero (we clamp std to 1.0)."""
    X = np.ones((10, 3), dtype=float)
    X[:, 1] = 5.0  # constant column
    X[:, 2] = np.arange(10, dtype=float)
    X_std, _, _ = bte._standardize(X)
    assert np.all(np.isfinite(X_std))
    # Constant column becomes zero-mean after transform.
    assert np.allclose(X_std[:, 1], 0.0)


# ---------------------------------------------------------------------------
# 5. Single-seed bench
# ---------------------------------------------------------------------------
def test_build_design_matrix_shape_and_finite() -> None:
    """``_build_design_matrix`` returns finite ``(n, 5)`` X and finite y."""
    fake_rows = [
        {"smiles": "CCO", "metal": "Pt"},
        {"smiles": "CC12CCC(...)C2(C)C", "metal": "Pd"},
        {"smiles": "CCN1c2ccccc2", "metal": "Ru"},
    ] * 5  # 15 rows
    X, y, w = bte._build_design_matrix(fake_rows, seed=0)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert w.shape == (5,)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(w))


def test_run_one_seed_returns_finite_metrics() -> None:
    """A single-seed run produces finite per-stage metrics."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, 5)) * 10.0 + 5.0
    y = rng.standard_normal(20)
    res = bte._run_one_seed(
        seed=0,
        n_trajectories=20,
        n_rounds=2,
        n_arms=4,
        hilbert_iters=10,
        X_seed=X,
        y_seed=y,
    )
    assert res["seed"] == 0
    assert res["n_trajectories"] == 20
    assert res["n_rounds"] == 2
    assert len(res["per_round"]) == 2
    # Per-stage wall times are non-negative finite.
    assert math.isfinite(res["mean_astsr_wall_s"])
    assert res["mean_astsr_wall_s"] >= 0.0
    assert math.isfinite(res["mean_hilbert_wall_s"])
    assert res["mean_hilbert_wall_s"] >= 0.0
    assert math.isfinite(res["mean_mcts_wall_s"])
    assert res["mean_mcts_wall_s"] >= 0.0
    # Success rates are in [0, 1].
    assert 0.0 <= res["astsr_success_rate"] <= 1.0
    assert 0.0 <= res["hilbert_success_rate"] <= 1.0
    assert 0.0 <= res["mcts_success_rate"] <= 1.0
    # R² metrics are finite floats.
    assert math.isfinite(res["mean_hilbert_r2"])
    assert math.isfinite(res["mean_evolutionary_r2"])
    assert math.isfinite(res["mean_variance_reduction"])


def test_run_one_seed_deterministic_for_same_inputs() -> None:
    """Identical inputs produce identical outputs."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((20, 5)) * 5.0 + 5.0
    y = rng.standard_normal(20)
    r1 = bte._run_one_seed(
        seed=42, n_trajectories=20, n_rounds=2, n_arms=4,
        hilbert_iters=10, X_seed=X, y_seed=y,
    )
    r2 = bte._run_one_seed(
        seed=42, n_trajectories=20, n_rounds=2, n_arms=4,
        hilbert_iters=10, X_seed=X, y_seed=y,
    )
    assert r1["mean_hilbert_r2"] == r2["mean_hilbert_r2"]
    assert r1["mean_evolutionary_r2"] == r2["mean_evolutionary_r2"]
    assert r1["mean_variance_reduction"] == r2["mean_variance_reduction"]
    assert r1["astsr_success_rate"] == r2["astsr_success_rate"]


# ---------------------------------------------------------------------------
# 6. Multi-seed build_report
# ---------------------------------------------------------------------------
def test_build_report_with_real_platinai_csv() -> None:
    """End-to-end build_report on the real PlatinAI 5K CSV."""
    csv_path = Path(bte._REPO_ROOT) / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"PlatinAI 5K CSV not found at {csv_path}")
    report = bte.build_report(
        platinai_csv=str(csv_path),
        n_trajectories=20,
        n_rounds=2,
        seeds=(0, 1, 2),
        n_arms=4,
        hilbert_iters=10,
    )
    assert report["n_trajectories"] == 20
    assert report["n_rounds"] == 2
    assert report["seeds"] == [0, 1, 2]
    assert len(report["per_seed"]) == 3
    summary = report["summary"]
    assert summary["n_seeds"] == 3
    assert summary["platinai_rows_loaded"] > 1000
    # Summary keys all present and finite.
    for key in (
        "mean_astsr_wall_s", "mean_hilbert_wall_s", "mean_mcts_wall_s",
        "std_astsr_wall_s", "std_hilbert_wall_s", "std_mcts_wall_s",
        "astsr_success_rate", "hilbert_success_rate", "mcts_success_rate",
        "mean_hilbert_r2", "std_hilbert_r2",
        "mean_evolutionary_r2", "std_evolutionary_r2",
        "mean_variance_reduction", "std_variance_reduction",
    ):
        assert key in summary, key
        assert math.isfinite(summary[key]), (key, summary[key])
    # Success rates live in [0, 1].
    for stage in ("astsr", "hilbert", "mcts"):
        assert 0.0 <= summary[f"{stage}_success_rate"] <= 1.0, stage


def test_build_report_with_synthetic_csv(tmp_path: Path) -> None:
    """``build_report`` works against an arbitrary small synthetic CSV."""
    csv_path = tmp_path / "tiny_plat.csv"
    lines = ["id,smiles,ic50_nm,selectivity,metal"]
    for _ in range(8):
        lines.append("a,CCO,,,Pt")
        lines.append("b,CC12CCC(...)C2(C)C,,,Pd")
        lines.append("c,CCN1c2ccccc2,,,Ru")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = bte.build_report(
        platinai_csv=str(csv_path),
        n_trajectories=5,
        n_rounds=2,
        seeds=(0, 1),
        n_arms=3,
        hilbert_iters=5,
    )
    assert report["summary"]["n_seeds"] == 2
    assert report["summary"]["platinai_rows_loaded"] == 24
    assert len(report["per_seed"]) == 2


# ---------------------------------------------------------------------------
# 7. Smoke / IO helpers
# ---------------------------------------------------------------------------
def test_write_report_emits_all_three_files(tmp_path: Path) -> None:
    """``write_report`` emits ``triangle_bench.json``, ``per_seed.csv``,
    and ``triangle_bench.md``."""
    csv_path = Path(bte._REPO_ROOT) / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"PlatinAI 5K CSV not found at {csv_path}")
    report = bte.build_report(
        platinai_csv=str(csv_path),
        n_trajectories=10,
        n_rounds=2,
        seeds=(0, 1),
        n_arms=4,
        hilbert_iters=8,
    )
    json_path, csv_path_out, md_path = bte.write_report(
        out_dir=str(tmp_path), report=report,
        also_markdown=True, also_csv=True,
    )
    assert json_path.exists()
    assert json_path.name == "triangle_bench.json"
    assert csv_path_out is not None and csv_path_out.exists()
    assert csv_path_out.name == "per_seed.csv"
    assert md_path is not None and md_path.exists()
    assert md_path.name == "triangle_bench.md"
    # JSON round-trip.
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["seeds"] == [0, 1]
    assert loaded["summary"]["n_seeds"] == 2
    # Markdown mentions the stage names.
    body = md_path.read_text(encoding="utf-8")
    assert "Wf-39 L4" in body
    assert "success rate" in body.lower()
    assert "ASTSR" in body and "Hilbert" in body and "MCTS" in body


def test_smoke_run_returns_true_on_happy_path() -> None:
    """``smoke_run`` returns ``True`` against the real PlatinAI 5K CSV."""
    csv_path = Path(bte._REPO_ROOT) / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"PlatinAI 5K CSV not found at {csv_path}")
    assert bte.smoke_run(platinai_csv=str(csv_path)) is True


def test_cli_smoke_exits_zero() -> None:
    """``--smoke`` exits with code 0 on the happy path."""
    import subprocess
    csv_path = Path(bte._REPO_ROOT) / "molmetal" / "data" / "platinai_5k.csv"
    if not csv_path.exists():
        pytest.skip(f"PlatinAI 5K CSV not found at {csv_path}")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(bte._REPO_ROOT) / "molmetal" / "scripts"
                 / "bench_lambda_triangle_end_to_end.py"),
            "--smoke",
            "--platinai-csv", str(csv_path),
        ],
        capture_output=True, text=True,
        cwd=str(bte._REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PASS" in result.stdout


def test_cli_help_exits_zero() -> None:
    """``--help`` exits with code 0 and mentions the bench."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(Path(bte._REPO_ROOT) / "molmetal" / "scripts"
                 / "bench_lambda_triangle_end_to_end.py"),
            "--help",
        ],
        capture_output=True, text=True,
        cwd=str(bte._REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0
    combined = (result.stdout + result.stderr).lower()
    assert "triangle" in combined
    assert "platinai" in combined
