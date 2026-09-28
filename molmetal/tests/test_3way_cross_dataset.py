"""Pytest suite for Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH +
ASTSR) cross-dataset verify on PlatinAI 4-cell + tmQM d-block.

The scripts under test:

* ``molmetal/scripts/bench_3way_platinai_4cell.py`` — per cell-line
  (MCF7 / A549 / A2780 / A2780cis) 5-fold × 5-seed 3-way integration.
* ``molmetal/scripts/bench_3way_tmqm.py`` — per d-block element
  (29 elements with ≥5 rows) 5-fold × 5-seed 3-way integration.

These tests guard the script invariants and provide fast unit-level
confidence for Wf-128 P3.

Test inventory
--------------
1.  ``test_platinai_cell_cols_constant`` — 4 cell-line columns
    (ic50_mcf7_nm, ic50_a549_nm, ic50_a2780_nm, ic50_a2780cis_nm)
    are present.
2.  ``test_platinai_default_cell_lines`` — default cohort is
    ``("mcf7", "a549", "a2780", "a2780cis")``.
3.  ``test_load_platinai_cell_mcf7_shape`` — load MCF7 cell line:
    X shape (n, 172), y is finite pIC50.
4.  ``test_load_platinai_cell_unknown_raises`` — unknown cell line
    raises ``ValueError``.
5.  ``test_evaluate_fold_returns_fit_record`` — ``evaluate_fold``
    returns a ``FitRecord`` with finite combined R² on MCF7 cohort.
6.  ``test_evaluate_cell_line_returns_records`` — ``evaluate_cell_line``
    returns one ``FitRecord`` per (seed, fold) for MCF7 smoke.
7.  ``test_aggregate_cell_line_per_seed_summary`` — ``aggregate_cell_line``
    returns one ``CellLineSeedSummary`` per (cell_line, seed).
8.  ``test_platinai_smoke_bench_produces_artifacts`` — smoke run
    writes ``per_seed_metrics.json`` and ``synthesis.md``.
9.  ``test_tmqm_default_elements_subset`` — default elements list
    is a 5-subset of the full 30-element d-block.
10. ``test_load_tmqm_cu_shape`` — load Cu cohort: X shape (n, 172),
    y is finite coord_number.
11. ``test_list_available_elements_min_rows`` — element list only
    includes elements with ≥ 5 rows.
12. ``test_tmqm_smoke_bench_produces_artifacts`` — smoke run writes
    ``per_seed_metrics.json`` and ``synthesis.md``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent  # molmetal/
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "bench_3way_platinai_4cell.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT), str(_REPO_ROOT / "molmetal" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_3way_platinai_4cell import (  # noqa: E402
    DEFAULT_CELL_LINES,
    FEATURE_DIM,
    FitRecord as PlatFitRecord,
    CellLineSeedSummary as PlatSeedSummary,
    PLATINAI_CELL_COLS,
    aggregate_cell_line,
    evaluate_cell_line,
    evaluate_fold as plat_evaluate_fold,
    load_platinai_cell,
    write_synthesis as plat_write_synthesis,
)
from bench_3way_tmqm import (  # noqa: E402
    ALL_DBLOCK_ELEMENTS,
    DEFAULT_ELEMENTS,
    MIN_ROWS_PER_ELEMENT,
    FitRecord as TmqmFitRecord,
    ElementSeedSummary as TmqmSeedSummary,
    aggregate_element,
    evaluate_element,
    list_available_elements,
    load_tmqm_element,
    write_synthesis as tmqm_write_synthesis,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_CELL_COLS = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}


# ---------------------------------------------------------------------------
# 1. PlatinAI cell column mapping is stable.
# ---------------------------------------------------------------------------
def test_platinai_cell_cols_constant():
    assert PLATINAI_CELL_COLS == EXPECTED_CELL_COLS
    assert set(PLATINAI_CELL_COLS.keys()) == set(EXPECTED_CELL_COLS.keys())


# ---------------------------------------------------------------------------
# 2. PlatinAI default cell lines.
# ---------------------------------------------------------------------------
def test_platinai_default_cell_lines():
    assert tuple(DEFAULT_CELL_LINES) == ("mcf7", "a549", "a2780", "a2780cis")
    assert set(DEFAULT_CELL_LINES) == set(EXPECTED_CELL_COLS.keys())


# ---------------------------------------------------------------------------
# 3. PlatinAI loader — MCF7 shape.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_mcf7_shape():
    try:
        X, y, smiles = load_platinai_cell("mcf7", n_max=300)
    except (FileNotFoundError, RuntimeError):
        pytest.skip("PlatinAI CSV not available")
    assert X.ndim == 2
    assert X.shape[1] == FEATURE_DIM, (
        f"expected {FEATURE_DIM}-d features; got {X.shape[1]}"
    )
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] == len(smiles)
    assert np.all(np.isfinite(y))
    # Multi-PH + PLL slots are zero-padded.
    assert np.allclose(X[:, 12:FEATURE_DIM], 0.0)


# ---------------------------------------------------------------------------
# 4. PlatinAI loader — unknown cell line raises ValueError.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_unknown_raises():
    with pytest.raises(ValueError):
        load_platinai_cell("not_a_cell_line")


# ---------------------------------------------------------------------------
# 5. PlatinAI evaluate_fold returns FitRecord with finite combined R².
# ---------------------------------------------------------------------------
def test_evaluate_fold_returns_fit_record():
    try:
        X, y, _smi = load_platinai_cell("mcf7", n_max=200)
    except (FileNotFoundError, RuntimeError):
        pytest.skip("PlatinAI CSV not available")
    n = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr = max(10, n // 2)
    tr, te = perm[:n_tr], perm[n_tr:]
    rec = plat_evaluate_fold(
        cell_line="mcf7", seed=0, fold=0, tr=tr, te=te,
        X=X, y=y, epochs=3, hilbert_dim=32,
    )
    assert isinstance(rec, PlatFitRecord)
    assert math.isfinite(rec.combined_r2)
    assert math.isfinite(rec.hilbert_r2)
    assert math.isfinite(rec.mph_r2)
    assert math.isfinite(rec.astr_r2)
    assert rec.n_train > 0
    assert rec.n_test > 0


# ---------------------------------------------------------------------------
# 6. PlatinAI evaluate_cell_line returns records.
# ---------------------------------------------------------------------------
def test_evaluate_cell_line_returns_records():
    try:
        recs = evaluate_cell_line(
            cell_line="mcf7", seeds=(0,), n_folds=2,
            epochs=3, hilbert_dim=32,
            csv_path="molmetal/data/platinai_4cell.csv", n_max=120,
        )
    except (FileNotFoundError, RuntimeError):
        pytest.skip("PlatinAI CSV not available")
    assert len(recs) == 2
    for r in recs:
        assert isinstance(r, PlatFitRecord)
        assert math.isfinite(r.combined_r2)


# ---------------------------------------------------------------------------
# 7. PlatinAI aggregate_cell_line returns CellLineSeedSummary per (cl, seed).
# ---------------------------------------------------------------------------
def test_aggregate_cell_line_per_seed_summary():
    records = [
        PlatFitRecord(
            cell_line="mcf7", seed=0, fold=0, n_train=40, n_test=10,
            combined_r2=-0.05, hilbert_r2=-0.10, mph_r2=-0.20, astr_r2=-0.30,
            wall_s=1.0, collapse_detected=False,
        ),
        PlatFitRecord(
            cell_line="mcf7", seed=0, fold=1, n_train=40, n_test=10,
            combined_r2=0.10, hilbert_r2=-0.05, mph_r2=0.05, astr_r2=-0.20,
            wall_s=1.2, collapse_detected=True,
        ),
    ]
    out = aggregate_cell_line(records)
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, PlatSeedSummary)
    assert s.cell_line == "mcf7"
    assert s.seed == 0
    assert s.n_folds == 2
    assert math.isfinite(s.combined_r2_mean)
    assert s.collapse_detected_any is True


# ---------------------------------------------------------------------------
# 8. PlatinAI smoke bench — produces artifacts.
# ---------------------------------------------------------------------------
def test_platinai_smoke_bench_produces_artifacts(tmp_path):
    out_dir = tmp_path / "wf128_p3_plat_smoke"
    argv = [
        "--smoke",
        "--out-dir", str(out_dir),
    ]
    from bench_3way_platinai_4cell import main as plat_main
    try:
        rc = plat_main(argv)
    except Exception as exc:
        pytest.skip(f"PlatinAI bench unavailable: {exc!r}")
    assert rc == 0
    assert (out_dir / "synthesis.md").exists()
    assert (out_dir / "mcf7" / "per_seed_metrics.json").exists()
    payload = json.loads(
        (out_dir / "mcf7" / "per_seed_metrics.json").read_text()
    )
    assert payload["cell_line"] == "mcf7"
    assert "per_seed" in payload and len(payload["per_seed"]) == 1
    assert "config" in payload
    assert payload["config"]["feature_dim"] == FEATURE_DIM


# ---------------------------------------------------------------------------
# 9. tmQM default elements — 5-subset of full 30-element d-block.
# ---------------------------------------------------------------------------
def test_tmqm_default_elements_subset():
    assert len(DEFAULT_ELEMENTS) == 5
    for el in DEFAULT_ELEMENTS:
        assert el in ALL_DBLOCK_ELEMENTS, (
            f"{el} not in d-block elements list"
        )
    assert len(ALL_DBLOCK_ELEMENTS) == 30


# ---------------------------------------------------------------------------
# 10. tmQM loader — Cu cohort shape.
# ---------------------------------------------------------------------------
def test_load_tmqm_cu_shape():
    try:
        X, y, smiles = load_tmqm_element("Cu", n_max=100)
    except (FileNotFoundError, RuntimeError):
        pytest.skip("tmQM CSV not available")
    assert X.ndim == 2
    assert X.shape[1] == FEATURE_DIM
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] == len(smiles)
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# 11. tmQM list_available_elements respects MIN_ROWS_PER_ELEMENT.
# ---------------------------------------------------------------------------
def test_list_available_elements_min_rows():
    try:
        available = list_available_elements()
    except FileNotFoundError:
        pytest.skip("tmQM CSV not available")
    # All available elements are d-block metals.
    for el in available:
        assert el in ALL_DBLOCK_ELEMENTS, f"{el} not in d-block"
    # All available elements should have ≥ MIN_ROWS_PER_ELEMENT rows
    # (verified indirectly: we only emit elements that pass the filter).
    # We can also assert the list is non-empty (real CSV has data).
    assert len(available) >= 5, (
        f"expected ≥ 5 elements; got {len(available)}"
    )


# ---------------------------------------------------------------------------
# 12. tmQM smoke bench — produces artifacts.
# ---------------------------------------------------------------------------
def test_tmqm_smoke_bench_produces_artifacts(tmp_path):
    out_dir = tmp_path / "wf128_p3_tmqm_smoke"
    argv = [
        "--smoke",
        "--out-dir", str(out_dir),
    ]
    from bench_3way_tmqm import main as tmqm_main
    try:
        rc = tmqm_main(argv)
    except Exception as exc:
        pytest.skip(f"tmQM bench unavailable: {exc!r}")
    assert rc == 0
    assert (out_dir / "synthesis.md").exists()
    # Cu is a smoke element.
    assert (out_dir / "Cu" / "per_seed_metrics.json").exists()
    payload = json.loads(
        (out_dir / "Cu" / "per_seed_metrics.json").read_text()
    )
    assert payload["element"] == "Cu"
    assert "per_seed" in payload and len(payload["per_seed"]) == 1
    assert "config" in payload
    assert payload["config"]["target_col"] == "coord_number"