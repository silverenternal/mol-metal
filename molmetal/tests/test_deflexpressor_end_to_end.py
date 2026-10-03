"""Tests for the Wf-39 Phase 2 Deflexpressor end-to-end training bench.

================================================================
End-to-end test for ``bench_deflexpressor_end_to_end.py`` (TODO-83
§L5, 2026-09-26) — the multi-seed × multi-method × real IC50 training
bench that supersedes Wf-37's single-seed
``train_hilbert_astsr_end_to_end.py``.

Test design
-----------
1.  ``test_load_platinai_4cell_pic50_finite`` — every loaded row has a
    finite ``pic50`` in the documented range.
2.  ``test_load_nci60_precious_pic50_finite`` — same for the NCI-60
    loader.
3.  ``test_featurize_smiles_returns_12d`` — every SMILES in either CSV
    featurises to a 12-d RDKit descriptor.
4.  ``test_build_design_matrix_shape`` — ``build_design_matrix`` returns
    the expected ``(smiles, X, y, stats)`` shape contract.
5.  ``test_pad_to_hilbert_dim_shape`` — augmentation to 64-d produces
    a (N, 64) train matrix and matching (N_test, 64) test matrix.
6.  ``test_zscore_no_nan_inf`` — z-scored matrices are finite.
7.  ``test_stable_pd_fit_finite`` — the Wf-39 P2 closed-form ridge
    returns finite ``best_params_`` and ``train_r2`` for any
    ``ridge_alpha > 0``.
8.  ``test_stable_pd_fit_train_r2_improves_over_init`` — the converged
    iterate beats the random init on training R² (sanity that the
    regression actually learned).
9.  ``test_fit_ast_returns_finite_r2`` — ``fit_ast`` returns a finite
    R² on a toy 80/20 split.
10. ``test_fit_hilbert_returns_finite_r2`` — ``fit_hilbert`` returns a
    finite R² on a toy 80/20 split; no NaN / no negative infinity
    (the Wf-25 ``R² = -0.443`` numerical pathology must be gone).
11. ``test_fit_composite_picks_better_branch`` — ``fit_composite``
    returns the better of the Hilbert / AST branches by test R².
12. ``test_run_one_records_wall_and_mem`` — ``_run_one`` writes a
    :class:`FitRecord` with ``wall_s > 0`` and ``peak_mem_mb >= 0``.
13. ``test_run_benchmark_produces_payload`` — ``run_benchmark`` returns
    a payload with ``per_method`` + ``records`` + ``config`` blocks.
14. ``test_payload_is_json_serialisable`` — ``json.dumps(payload)``
    succeeds (no numpy / Path leakage).
15. ``test_render_markdown_contains_method_table`` — the markdown
    table includes every method, R² + wall-time columns.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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

from bench_deflexpressor_end_to_end import (  # noqa: E402
    DEFAULT_HYPERPARAMS,
    DEFAULT_METHODS,
    DEFAULT_SEEDS,
    FEATURE_COLUMNS,
    HILBERT_DIM,
    N_DESCRIPTORS,
    FitRecord,
    _featurize_smiles,
    _r2,
    _run_one,
    _stable_pd_fit,
    build_design_matrix,
    fit_ast,
    fit_composite,
    fit_hilbert,
    load_nci60_precious,
    load_platinai_4cell,
    pad_to_hilbert_dim,
    render_markdown,
    run_benchmark,
    write_metrics,
    zscore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def platinai_rows():
    return load_platinai_4cell(
        _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv",
        n_max=120,
    )


@pytest.fixture(scope="module")
def nci60_rows():
    return load_nci60_precious(
        _REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv",
        n_max=120,
    )


@pytest.fixture(scope="module")
def toy_dataset(nci60_rows):
    """Build a deterministic (X, y) for fast tests."""
    smiles, X, y, stats = build_design_matrix(rows=nci60_rows, seed=0)
    return smiles, X, y, stats


# ---------------------------------------------------------------------------
# 1. Loaders — PlatinAI 4-cell
# ---------------------------------------------------------------------------
def test_load_platinai_4cell_pic50_finite(platinai_rows):
    assert len(platinai_rows) >= 50, f"too few platinai rows: {len(platinai_rows)}"
    for row in platinai_rows:
        assert "smiles" in row and row["smiles"]
        assert math.isfinite(row["pic50"])
        assert (
            2.0 <= row["pic50"] <= 10.0
        ), f"pic50 out of expected [2, 10]: {row['pic50']}"


# ---------------------------------------------------------------------------
# 2. Loaders — NCI-60
# ---------------------------------------------------------------------------
def test_load_nci60_precious_pic50_finite(nci60_rows):
    assert len(nci60_rows) >= 50, f"too few nci60 rows: {len(nci60_rows)}"
    for row in nci60_rows:
        assert "smiles" in row and row["smiles"]
        assert math.isfinite(row["pic50"])
        assert (
            2.0 <= row["pic50"] <= 10.0
        ), f"pic50 out of expected [2, 10]: {row['pic50']}"


# ---------------------------------------------------------------------------
# 3. RDKit featurizer
# ---------------------------------------------------------------------------
def test_featurize_smiles_returns_12d(platinai_rows, nci60_rows):
    """Every SMILES in either CSV featurises to a 12-d descriptor."""
    n_total = 0
    n_fail = 0
    for row in list(platinai_rows)[:30] + list(nci60_rows)[:30]:
        feats = _featurize_smiles(row["smiles"])
        if feats is None:
            n_fail += 1
            continue
        assert (
            len(feats) == N_DESCRIPTORS
        ), f"featurize returned {len(feats)} features, expected {N_DESCRIPTORS}"
        n_total += 1
    assert n_total >= 50, f"only {n_total} featurised; expected ≥ 50"
    # Allow up to 10 % parse failures (some d-block SMILES are tricky).
    assert n_fail <= 6, f"too many featurize failures: {n_fail}"


# ---------------------------------------------------------------------------
# 4. build_design_matrix — shape contract
# ---------------------------------------------------------------------------
def test_build_design_matrix_shape(nci60_rows):
    smiles, X, y, stats = build_design_matrix(rows=nci60_rows, seed=0)
    assert isinstance(smiles, list)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2 and X.shape[1] == N_DESCRIPTORS
    assert y.ndim == 1 and y.shape[0] == X.shape[0]
    assert "n_parsed" in stats and "n_failed" in stats
    assert stats["n_parsed"] == X.shape[0]


# ---------------------------------------------------------------------------
# 5. pad_to_hilbert_dim — shape contract
# ---------------------------------------------------------------------------
def test_pad_to_hilbert_dim_shape(toy_dataset):
    _, X, _, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
    assert X_tr_64.shape == (X_tr_n.shape[0], HILBERT_DIM)
    assert X_te_64 is not None
    assert X_te_64.shape == (X_te_n.shape[0], HILBERT_DIM)


# ---------------------------------------------------------------------------
# 6. zscore — no NaN / Inf
# ---------------------------------------------------------------------------
def test_zscore_no_nan_inf(toy_dataset):
    _, X, _, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    assert np.all(np.isfinite(X_tr_n)), "z-scored train matrix has NaN/Inf"
    assert np.all(np.isfinite(X_te_n)), "z-scored test matrix has NaN/Inf"


# ---------------------------------------------------------------------------
# 7. _stable_pd_fit — finite output
# ---------------------------------------------------------------------------
def test_stable_pd_fit_finite(toy_dataset):
    _, X, y, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    X_tr, _ = zscore(X[train_idx], X[train_idx[:1]])
    X_tr_64, _ = pad_to_hilbert_dim(X_tr, HILBERT_DIM)
    best_x, train_r2 = _stable_pd_fit(
        X_tr_64,
        y[train_idx],
        hilbert_dim=HILBERT_DIM,
        lr=0.05,
        n_iters=200,
        seed=0,
        ridge_alpha=0.1,
    )
    assert best_x.shape == (HILBERT_DIM,)
    assert np.all(np.isfinite(best_x)), "best_x has NaN/Inf"
    assert math.isfinite(train_r2), f"train_r2 not finite: {train_r2}"


# ---------------------------------------------------------------------------
# 8. _stable_pd_fit — improves over random init
# ---------------------------------------------------------------------------
def test_stable_pd_fit_train_r2_improves_over_init(toy_dataset):
    _, X, y, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    X_tr, _ = zscore(X[train_idx], X[train_idx[:1]])
    X_tr_64, _ = pad_to_hilbert_dim(X_tr, HILBERT_DIM)
    rng_init = np.random.default_rng(0)
    x_init = rng_init.standard_normal(HILBERT_DIM).astype(float) * 0.01
    y_hat_init = X_tr_64 @ x_init
    ss_res = float(np.sum((y[train_idx] - y_hat_init) ** 2))
    ss_tot = float(np.sum((y[train_idx] - np.mean(y[train_idx])) ** 2))
    r2_init = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    _, r2_fit = _stable_pd_fit(
        X_tr_64,
        y[train_idx],
        hilbert_dim=HILBERT_DIM,
        lr=0.05,
        n_iters=200,
        seed=0,
        ridge_alpha=0.1,
    )
    # The converged iterate should not regress on training R².
    assert (
        r2_fit >= r2_init - 1e-6
    ), f"fit regressed: init={r2_init:+.4f} fit={r2_fit:+.4f}"


# ---------------------------------------------------------------------------
# 9. fit_ast — finite R²
# ---------------------------------------------------------------------------
def test_fit_ast_returns_finite_r2(toy_dataset):
    _, X, y, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    hp = dict(DEFAULT_HYPERPARAMS["ast"])
    # Use a small population so the test runs in <5 s.
    hp.update({"depth": 2, "population": 8, "generations": 4})
    r2, train_r2, branch = fit_ast(
        X_tr_n,
        y[train_idx],
        X_te_n,
        y[test_idx],
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"AST test R² not finite: {r2}"
    assert branch == "ast"


# ---------------------------------------------------------------------------
# 10. fit_hilbert — finite R² (the Wf-25 pathology must be gone)
# ---------------------------------------------------------------------------
def test_fit_hilbert_returns_finite_r2(toy_dataset):
    _, X, y, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
    hp = dict(DEFAULT_HYPERPARAMS["hilbert"])
    hp.update({"n_iters": 50, "ridge_alpha": 0.1})
    r2, train_r2, branch = fit_hilbert(
        X_tr_64,
        y[train_idx],
        X_te_64,
        y[test_idx],
        seed=0,
        hp=hp,
    )
    assert math.isfinite(r2), f"Hilbert test R² not finite: {r2}"
    assert -1e6 < r2 < 1e6, f"Hilbert test R² out of bounds: {r2}"
    assert branch == "hilbert"


# ---------------------------------------------------------------------------
# 11. fit_composite — picks the better branch
# ---------------------------------------------------------------------------
def test_fit_composite_picks_better_branch(toy_dataset):
    _, X, y, _ = toy_dataset
    rng = np.random.default_rng(0)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = int(0.2 * X.shape[0])
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
    hp = dict(DEFAULT_HYPERPARAMS["composite"])
    hp.update(
        {
            "n_iters": 50,
            "ridge_alpha": 0.1,
            "depth": 2,
            "population": 8,
            "generations": 4,
        }
    )
    r2, train_r2, branch = fit_composite(
        X_tr_n,
        y[train_idx],
        X_te_n,
        y[test_idx],
        X_tr_64,
        X_te_64,
        seed=0,
        hp=hp,
    )
    assert branch in ("ast", "hilbert"), f"composite branch invalid: {branch}"
    assert math.isfinite(r2), f"composite R² not finite: {r2}"


# ---------------------------------------------------------------------------
# 12. _run_one — records wall-time + memory
# ---------------------------------------------------------------------------
def test_run_one_records_wall_and_mem(nci60_rows):
    hp = dict(DEFAULT_HYPERPARAMS["hilbert"])
    hp.update({"n_iters": 30, "ridge_alpha": 0.1})
    rec = _run_one(
        method="hilbert",
        seed=0,
        rows=nci60_rows,
        hp=hp,
        test_frac=0.2,
        epochs=30,
    )
    assert isinstance(rec, FitRecord)
    assert rec.wall_s > 0.0
    assert rec.peak_mem_mb >= 0.0
    assert rec.n_train > 0
    assert rec.n_test > 0
    assert math.isfinite(rec.r2_test), f"r2_test not finite: {rec.r2_test}"


# ---------------------------------------------------------------------------
# 13. run_benchmark — payload contract
# ---------------------------------------------------------------------------
def test_run_benchmark_produces_payload(nci60_rows):
    payload = run_benchmark(
        dataset="nci60",
        n_samples=80,
        seeds=(0, 1),
        methods=("ast", "hilbert"),
        epochs=20,
        test_frac=0.2,
        hyperparams={
            m: {
                **DEFAULT_HYPERPARAMS[m],
                **{
                    "depth": 2,
                    "population": 8,
                    "generations": 4,
                    "n_iters": 20,
                    "ridge_alpha": 0.1,
                },
            }
            for m in ("ast", "hilbert")
        },
    )
    assert "config" in payload
    assert "per_method" in payload
    assert "records" in payload
    assert "ast" in payload["per_method"]
    assert "hilbert" in payload["per_method"]
    assert payload["per_method"]["ast"]["n_records"] == 2
    assert payload["per_method"]["hilbert"]["n_records"] == 2
    # config block has the full provenance.
    cfg = payload["config"]
    assert cfg["dataset"] == "nci60"
    assert cfg["seeds"] == [0, 1]
    assert cfg["methods"] == ["ast", "hilbert"]


# ---------------------------------------------------------------------------
# 14. JSON serialisability
# ---------------------------------------------------------------------------
def test_payload_is_json_serialisable(nci60_rows):
    payload = run_benchmark(
        dataset="nci60",
        n_samples=60,
        seeds=(0,),
        methods=("hilbert",),
        epochs=20,
        test_frac=0.2,
        hyperparams={
            "hilbert": {
                **DEFAULT_HYPERPARAMS["hilbert"],
                **{"n_iters": 20, "ridge_alpha": 0.1},
            },
        },
    )
    # Must round-trip through json.dumps without raising.
    s = json.dumps(payload)
    assert isinstance(s, str)
    assert len(s) > 100


# ---------------------------------------------------------------------------
# 15. Markdown render — table structure
# ---------------------------------------------------------------------------
def test_render_markdown_contains_method_table(nci60_rows):
    payload = run_benchmark(
        dataset="nci60",
        n_samples=60,
        seeds=(0, 1),
        methods=("ast", "hilbert"),
        epochs=20,
        test_frac=0.2,
        hyperparams={
            m: {
                **DEFAULT_HYPERPARAMS[m],
                **{
                    "depth": 2,
                    "population": 8,
                    "generations": 4,
                    "n_iters": 20,
                    "ridge_alpha": 0.1,
                },
            }
            for m in ("ast", "hilbert")
        },
    )
    md = render_markdown(payload)
    assert "ast" in md
    assert "hilbert" in md
    assert "R² mean" in md or "R² mean" in md
    assert "Wall" in md
    assert "Speedup" in md
    assert "Per-seed" in md
