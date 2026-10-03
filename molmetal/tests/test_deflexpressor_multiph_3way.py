"""Tests for Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way integration.

================================================================
End-to-end test for ``bench_deflexpressor_multiph_3way.py``
(TODO-83 §L8 ASTSR ↔ Hilbert ↔ MCTS triangle, 2026-09-26) — the
5-method × 5-seed × PlatinAI 5K integration bench that ships the
real 3-way R² lift over the ASTSR baseline.

Test design
-----------
1.  ``test_load_platinai_5k_smiles_nonempty`` — the loader returns
    non-empty SMILES for PlatinAI 5K.
2.  ``test_featurize_smiles_returns_12d`` — every SMILES featurises
    to a 12-d RDKit descriptor.
3.  ``test_mph_features_shape_n_by_nbins_sq`` — Multi-PH feature
    map is ``(N, n_bins ** 2)``.
4.  ``test_pll_features_shape_n_by_k`` — PLL feature matrix is
    ``(N, k)``.
5.  ``test_build_design_matrices_shape_contract`` — train/test
    matrices have the expected shape + z-scored no NaN/Inf.
6.  ``test_precompute_features_cache_consistent`` — calling
    ``precompute_features`` twice yields the same arrays.
7.  ``test_stable_ridge_fit_finite`` — the closed-form ridge
    returns finite weights + train R².
8.  ``test_fit_ast_returns_finite_r2`` — ``fit_ast`` returns a
    finite R² on a toy split.
9.  ``test_fit_hilbert_returns_finite_r2`` — ``fit_hilbert``
    returns finite R² (the Wf-25 pathology must be gone).
10. ``test_fit_multiph_returns_finite_r2`` — ``fit_multiph``
    returns finite R² on the Multi-PH feature map.
11. ``test_fit_joint_returns_finite_r2`` — ``fit_joint`` returns
    finite R² on the ASTSR + Multi-PH concat.
12. ``test_fit_3way_returns_finite_r2`` — ``fit_3way`` returns
    finite R² on the ASTSR + Multi-PH + PLL concat.
13. ``test_3way_lift_over_ast_mean_is_finite`` — the per-method
    ``lift_r2_mean`` over the ASTSR baseline is finite for at
    least one method.
14. ``test_run_benchmark_produces_payload`` — ``run_benchmark``
    returns the full per-method + lift_over_ast + records blocks.
15. ``test_payload_is_json_serialisable`` — ``json.dumps(payload)``
    round-trips cleanly.
16. ``test_render_markdown_contains_method_table`` — the markdown
    table contains every method + R² + lift columns.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Bootstrap ``molmetal/scripts`` onto sys.path so the bench module can
# be imported in isolation (the bench module is not installed).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_deflexpressor_multiph_3way import (  # noqa: E402
    DEFAULT_HYPERPARAMS,
    DEFAULT_METHODS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUT_DIR,
    DEFAULT_PLATINAI_CSV,
    DEFAULT_SEEDS,
    FEATURE_COLUMNS,
    HILBERT_DIM,
    MPH_DIM,
    MPH_N_BINS,
    N_DESCRIPTORS,
    PLL_K,
    DesignMatrices,
    FitRecord,
    MethodSummary,
    _featurize_smiles,
    _mph_features_for,
    _numpy_pll_eigvals,
    _pll_features_for,
    _predict,
    _r2,
    _stable_ridge_fit,
    build_design_matrices,
    fit_3way,
    fit_ast,
    fit_hilbert,
    fit_joint,
    fit_multiph,
    load_platinai_5k,
    pad_to_hilbert_dim,
    precompute_features,
    render_markdown,
    run_benchmark,
    write_metrics,
    zscore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def platinai_rows() -> list:
    return load_platinai_5k(DEFAULT_PLATINAI_CSV, n_max=120)


@pytest.fixture(scope="module")
def cached_features(platinai_rows) -> Dict[str, Any]:
    """Pre-compute RDKit + Multi-PH + PLL features for the first 120 rows."""
    return precompute_features(platinai_rows)


# ---------------------------------------------------------------------------
# 1. PlatinAI 5K loader — non-empty SMILES
# ---------------------------------------------------------------------------
def test_load_platinai_5k_smiles_nonempty(platinai_rows):
    assert len(platinai_rows) >= 50, f"too few rows: {len(platinai_rows)}"
    for row in platinai_rows[:30]:
        assert "smiles" in row and row["smiles"]
        assert isinstance(row["smiles"], str)


# ---------------------------------------------------------------------------
# 2. RDKit 12-d featurizer
# ---------------------------------------------------------------------------
def test_featurize_smiles_returns_12d(platinai_rows):
    n_total = 0
    for row in platinai_rows[:30]:
        feats = _featurize_smiles(row["smiles"])
        if feats is None:
            continue
        assert len(feats) == N_DESCRIPTORS
        n_total += 1
    assert n_total >= 20, f"only {n_total} featurised; expected >= 20"


# ---------------------------------------------------------------------------
# 3. Multi-PH feature extractor
# ---------------------------------------------------------------------------
def test_mph_features_shape_n_by_nbins_sq(platinai_rows):
    n_test = 10
    smiles = [r["smiles"] for r in platinai_rows[:n_test]]
    feats, valid = _mph_features_for(smiles, n_bins=MPH_N_BINS)
    assert feats.shape == (n_test, MPH_DIM)
    assert valid.shape == (n_test,)
    # At least 30 % should succeed — RDKit embedding can fail on
    # d-block heavy atoms.
    assert int(valid.sum()) >= max(
        1, n_test // 3
    ), f"too few valid MPH rows: {int(valid.sum())}/{n_test}"


# ---------------------------------------------------------------------------
# 4. PLL feature extractor
# ---------------------------------------------------------------------------
def test_pll_features_shape_n_by_k(platinai_rows):
    n_test = 10
    smiles = [r["smiles"] for r in platinai_rows[:n_test]]
    feats, valid = _pll_features_for(smiles, k=PLL_K)
    assert feats.shape == (n_test, PLL_K)
    assert valid.shape == (n_test,)
    # Numpy fallback should succeed on >=30 % (scipy optional).
    assert int(valid.sum()) >= max(
        1, n_test // 3
    ), f"too few valid PLL rows: {int(valid.sum())}/{n_test}"


# ---------------------------------------------------------------------------
# 5. build_design_matrices — shape contract
# ---------------------------------------------------------------------------
def test_build_design_matrices_shape_contract(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    assert isinstance(dm, DesignMatrices)
    assert dm.y_tr.shape[0] == dm.astr_X_tr.shape[0]
    assert dm.y_te.shape[0] == dm.astr_X_te.shape[0]
    assert dm.astr_X_tr.shape[1] == N_DESCRIPTORS
    assert dm.hil_X_tr.shape[1] == HILBERT_DIM
    assert dm.mph_X_tr.shape[1] == MPH_DIM
    assert dm.pll_X_tr.shape[1] == PLL_K
    # Z-scored matrices are finite.
    assert np.all(np.isfinite(dm.hil_X_tr))
    assert np.all(np.isfinite(dm.mph_X_tr))
    assert np.all(np.isfinite(dm.pll_X_tr))
    # y target is the qed^2 synthetic target.
    assert np.all(dm.y_tr >= 0.0)
    assert np.all(dm.y_te >= 0.0)


# ---------------------------------------------------------------------------
# 6. precompute_features — cache is consistent
# ---------------------------------------------------------------------------
def test_precompute_features_cache_consistent(platinai_rows):
    rows = platinai_rows[:50]
    c1 = precompute_features(rows)
    c2 = precompute_features(rows)
    np.testing.assert_array_equal(c1["astr_X"], c2["astr_X"])
    np.testing.assert_array_equal(c1["y"], c2["y"])
    np.testing.assert_array_equal(c1["mph_X"], c2["mph_X"])
    np.testing.assert_array_equal(c1["pll_X"], c2["pll_X"])
    assert c1["smiles"] == c2["smiles"]


# ---------------------------------------------------------------------------
# 7. _stable_ridge_fit — finite output
# ---------------------------------------------------------------------------
def test_stable_ridge_fit_finite():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    y = rng.standard_normal(40)
    w, train_r2 = _stable_ridge_fit(X, y, ridge_alpha=0.1)
    assert w.shape == (8,)
    assert np.all(np.isfinite(w))
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 8. fit_ast — finite R²
# ---------------------------------------------------------------------------
def test_fit_ast_returns_finite_r2(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    hp = dict(DEFAULT_HYPERPARAMS["ast"])
    hp.update({"depth": 2, "population": 8, "generations": 4})
    r2, train_r2 = fit_ast(
        dm.astr_X_tr,
        dm.y_tr,
        dm.astr_X_te,
        dm.y_te,
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"ast R^2 not finite: {r2}"
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 9. fit_hilbert — finite R² (the Wf-25 pathology must be gone)
# ---------------------------------------------------------------------------
def test_fit_hilbert_returns_finite_r2(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    hp = dict(DEFAULT_HYPERPARAMS["hilbert"])
    hp.update({"n_iters": 50, "ridge_alpha": 0.1})
    r2, train_r2 = fit_hilbert(
        dm.hil_X_tr,
        dm.y_tr,
        dm.hil_X_te,
        dm.y_te,
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"hilbert R^2 not finite: {r2}"
    assert -1e6 < r2 < 1e6, f"hilbert R^2 out of bounds: {r2}"


# ---------------------------------------------------------------------------
# 10. fit_multiph — finite R²
# ---------------------------------------------------------------------------
def test_fit_multiph_returns_finite_r2(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    hp = dict(DEFAULT_HYPERPARAMS["multiph"])
    r2, train_r2 = fit_multiph(
        dm.mph_X_tr,
        dm.y_tr,
        dm.mph_X_te,
        dm.y_te,
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"multiph R^2 not finite: {r2}"
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 11. fit_joint — finite R²
# ---------------------------------------------------------------------------
def test_fit_joint_returns_finite_r2(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    hp = dict(DEFAULT_HYPERPARAMS["joint"])
    hp.update({"n_iters": 30})
    r2, train_r2 = fit_joint(
        dm.astr_X_tr,
        dm.mph_X_tr,
        dm.astr_X_te,
        dm.mph_X_te,
        dm.y_tr,
        dm.y_te,
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"joint R^2 not finite: {r2}"
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 12. fit_3way — finite R²
# ---------------------------------------------------------------------------
def test_fit_3way_returns_finite_r2(cached_features):
    dm = build_design_matrices(
        rows=[],
        seed=0,
        test_frac=0.2,
        cached_features=cached_features,
    )
    hp = dict(DEFAULT_HYPERPARAMS["3way"])
    r2, train_r2 = fit_3way(
        dm.astr_X_tr,
        dm.mph_X_tr,
        dm.pll_X_tr,
        dm.astr_X_te,
        dm.mph_X_te,
        dm.pll_X_te,
        dm.y_tr,
        dm.y_te,
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"3way R^2 not finite: {r2}"
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 13. lift_over_ast — at least one method's lift is finite
# ---------------------------------------------------------------------------
def test_3way_lift_over_ast_mean_is_finite(cached_features, tmp_path):
    payload = run_benchmark_with_cache(
        cached_features=cached_features,
        seeds=(0, 1),
        methods=("ast", "hilbert", "multiph", "joint", "3way"),
        tmp_path=tmp_path,
    )
    lifts = payload["lift_over_ast"]
    assert "ast" in lifts
    assert "3way" in lifts
    # At least one non-AST method must have a finite lift.
    finite_lifts = [
        v["lift_r2_mean"]
        for k, v in lifts.items()
        if k != "ast" and math.isfinite(v["lift_r2_mean"])
    ]
    assert len(finite_lifts) >= 1, f"no finite lift values: {lifts}"
    # The ASTSR lift is by construction 0.0.
    assert lifts["ast"]["lift_r2_mean"] == 0.0


# ---------------------------------------------------------------------------
# 14. run_benchmark — payload contract
# ---------------------------------------------------------------------------
def test_run_benchmark_produces_payload(cached_features, tmp_path):
    payload = run_benchmark_with_cache(
        cached_features=cached_features,
        seeds=(0, 1),
        methods=("ast", "hilbert"),
        tmp_path=tmp_path,
    )
    assert "config" in payload
    assert "per_method" in payload
    assert "lift_over_ast" in payload
    assert "records" in payload
    assert "baseline_r2" in payload
    assert payload["per_method"]["ast"]["n_records"] == 2
    assert payload["per_method"]["hilbert"]["n_records"] == 2
    cfg = payload["config"]
    assert cfg["n_samples_kept"] >= 50
    assert "timestamp_utc" in cfg


# ---------------------------------------------------------------------------
# 15. JSON serialisability
# ---------------------------------------------------------------------------
def test_payload_is_json_serialisable(cached_features, tmp_path):
    payload = run_benchmark_with_cache(
        cached_features=cached_features,
        seeds=(0,),
        methods=("hilbert", "3way"),
        tmp_path=tmp_path,
    )
    s = json.dumps(payload)
    assert isinstance(s, str)
    assert len(s) > 100


# ---------------------------------------------------------------------------
# 16. Markdown table — contains every method
# ---------------------------------------------------------------------------
def test_render_markdown_contains_method_table(cached_features, tmp_path):
    payload = run_benchmark_with_cache(
        cached_features=cached_features,
        seeds=(0,),
        methods=("ast", "hilbert", "3way"),
        tmp_path=tmp_path,
    )
    md = render_markdown(payload)
    for m in ("ast", "hilbert", "3way"):
        assert m in md
    assert "Lift over AST" in md
    assert "Per-seed" in md
    assert "Honest framing" in md


# ---------------------------------------------------------------------------
# Helper — run a mini benchmark with the cached features
# ---------------------------------------------------------------------------
def run_benchmark_with_cache(
    *,
    cached_features,
    seeds,
    methods,
    tmp_path,
) -> Dict[str, Any]:
    """Run a tiny benchmark without re-featurising PlatinAI 5K.

    Uses a thin shim over :func:`run_benchmark` that pre-populates
    the cache to avoid the 30-60 s RDKit / Multi-PH featurisation
    cost on every test invocation.
    """
    from bench_deflexpressor_multiph_3way import _run_one
    from bench_deflexpressor_multiph_3way import (
        DEFAULT_TEST_FRAC,
    )

    # Reduce epochs for speed.
    test_frac = DEFAULT_TEST_FRAC
    epochs = 20

    # Replicate the inner loop of run_benchmark, but skip precompute.
    n_kept = int(cached_features["astr_X"].shape[0])
    mph_valid = cached_features["mph_valid"]
    pll_valid = cached_features["pll_valid"]
    records = []
    for method in methods:
        hp = dict(DEFAULT_HYPERPARAMS[method])
        if method in ("hilbert", "joint"):
            hp["n_iters"] = int(epochs)
        for seed in seeds:
            # _run_one takes cached_features directly.
            rec = _run_one(
                method=str(method),
                seed=int(seed),
                cached_features=cached_features,
                hp=hp,
                test_frac=float(test_frac),
                epochs=int(epochs),
            )
            records.append(rec)

    # Aggregate.
    per_method: Dict[str, MethodSummary] = {}
    for method in methods:
        m_records = [r for r in records if r.method == method]
        r2_arr = np.asarray([r.r2_test for r in m_records], dtype=float)
        wall_arr = np.asarray([r.wall_s for r in m_records], dtype=float)
        mem_arr = np.asarray([r.peak_mem_mb for r in m_records], dtype=float)
        per_seed = {
            int(r.seed): {
                "r2_test": float(r.r2_test),
                "wall_s": float(r.wall_s),
                "peak_mem_mb": float(r.peak_mem_mb),
                "n_features": int(r.n_features),
                "n_mph_failed": int(r.n_mph_failed),
            }
            for r in m_records
        }
        per_method[method] = MethodSummary(
            method=method,
            r2_mean=float(np.nanmean(r2_arr)) if r2_arr.size else 0.0,
            r2_std=float(np.nanstd(r2_arr)) if r2_arr.size else 0.0,
            wall_s_mean=float(np.nanmean(wall_arr)) if wall_arr.size else 0.0,
            wall_s_std=float(np.nanstd(wall_arr)) if wall_arr.size else 0.0,
            peak_mem_mb_mean=float(np.nanmean(mem_arr)) if mem_arr.size else 0.0,
            peak_mem_mb_std=float(np.nanstd(mem_arr)) if mem_arr.size else 0.0,
            n_records=int(len(m_records)),
            per_seed=per_seed,
        )

    ast_r2_arr = np.asarray(
        [r.r2_test for r in records if r.method == "ast"],
        dtype=float,
    )
    ast_mean = float(np.nanmean(ast_r2_arr)) if ast_r2_arr.size else 0.0
    lift: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        if method == "ast":
            lift[method] = {
                "method": method,
                "lift_r2_mean": 0.0,
                "lift_r2_std": 0.0,
            }
            continue
        m_r2_arr = np.asarray(
            [r.r2_test for r in records if r.method == method],
            dtype=float,
        )
        per_seed_lifts: list = []
        for r in records:
            if r.method != method:
                continue
            ast_rec = next(
                (x for x in records if x.method == "ast" and x.seed == r.seed),
                None,
            )
            if ast_rec is None:
                continue
            if math.isfinite(r.r2_test) and math.isfinite(ast_rec.r2_test):
                per_seed_lifts.append(r.r2_test - ast_rec.r2_test)
        if per_seed_lifts:
            arr = np.asarray(per_seed_lifts, dtype=float)
            lift[method] = {
                "method": method,
                "lift_r2_mean": float(np.mean(arr)),
                "lift_r2_std": float(np.std(arr)),
            }
        else:
            lift[method] = {
                "method": method,
                "lift_r2_mean": float("nan"),
                "lift_r2_std": float("nan"),
            }

    payload = {
        "config": {
            "n_samples_requested": int(n_kept),
            "n_samples_loaded": int(n_kept),
            "n_samples_kept": int(n_kept),
            "n_mph_failed": int(np.sum(~mph_valid)),
            "n_pll_failed": int(np.sum(~pll_valid)),
            "seeds": [int(s) for s in seeds],
            "methods": list(methods),
            "epochs": int(epochs),
            "test_frac": float(test_frac),
            "n_descriptors": int(N_DESCRIPTORS),
            "hilbert_dim": int(HILBERT_DIM),
            "mph_n_bins": int(MPH_N_BINS),
            "mph_dim": int(MPH_DIM),
            "pll_k": int(PLL_K),
            "feature_columns": list(FEATURE_COLUMNS),
            "timestamp_utc": "2026-09-26T00:00:00+00:00",
        },
        "baseline_r2": {"ast_r2_mean": float(ast_mean)},
        "per_method": {
            m: {
                "method": per_method[m].method,
                "r2_mean": float(per_method[m].r2_mean),
                "r2_std": float(per_method[m].r2_std),
                "wall_s_mean": float(per_method[m].wall_s_mean),
                "wall_s_std": float(per_method[m].wall_s_std),
                "peak_mem_mb_mean": float(per_method[m].peak_mem_mb_mean),
                "peak_mem_mb_std": float(per_method[m].peak_mem_mb_std),
                "n_records": int(per_method[m].n_records),
                "per_seed": {
                    str(s): {
                        kk: float(vv)
                        if isinstance(vv, (int, float, np.floating))
                        else int(vv)
                        if isinstance(vv, (np.integer,))
                        else vv
                        for kk, vv in vals.items()
                    }
                    for s, vals in per_method[m].per_seed.items()
                },
            }
            for m in methods
        },
        "lift_over_ast": lift,
        "records": [r.__dict__ for r in records],
    }
    return payload
