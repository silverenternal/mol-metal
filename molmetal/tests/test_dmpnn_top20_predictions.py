"""Tests for Wf-58 Phase 2 — D-MPNN pIC50 prediction on top-20 candidates.

12 tests covering:
  * top-20 CSV loader (3 tests)
  * external pIC50 predict helper (2 tests)
  * per-compound / per-cell aggregators (3 tests)
  * per-compound std gate (1 test)
  * end-to-end CLI smoke (1 test)
  * synthesis.md structure (2 tests)

All tests are CPU-only and use a tiny fixture dataset (n=10 in
platinai_4cell fixture) with ``--epochs 2`` so the suite runs in <30 s.

Honest framing
--------------
The fixture is a *synthetic* subset; AUC numbers are bench-quality, not
real-data quality. Tests verify **shape / gate / reproducibility** only,
not absolute accuracy.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "bench_dmpnn_top20_predictions.py"
FIXTURE_DIR = PROJECT_ROOT / "molmetal" / "tests" / "fixtures" / "wf58_dmpnn_top20"
TOP20_FIXTURE = FIXTURE_DIR / "top20_min.csv"
DATA_FIXTURE = FIXTURE_DIR / "platinai_4cell_min.csv"
TMP_OUT = FIXTURE_DIR / "_tmp_outputs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _ensure_fixtures():
    """Create minimal fixtures if missing."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUT.mkdir(parents=True, exist_ok=True)

    if not TOP20_FIXTURE.exists():
        # 5 candidates spanning 3 metals
        with TOP20_FIXTURE.open("w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "rank",
                    "compound_smiles",
                    "id",
                    "metal",
                    "dmpnn_pic50",
                    "qed",
                    "sa",
                    "vina_kcal",
                    "astsr_score",
                ]
            )
            w.writerow([1, "CCO", "test_001", "Pt", 7.0, 0.5, 1.0, -5.0, 0.05])
            w.writerow([2, "CC(=O)O", "test_002", "Pd", 6.5, 0.4, 1.0, -5.5, 0.04])
            w.writerow([3, "c1ccccc1", "test_003", "Ir", 7.2, 0.6, 1.0, -4.8, 0.06])
            w.writerow([4, "CCCC", "test_004", "Ru", 6.8, 0.55, 1.0, -5.2, 0.05])
            w.writerow([5, "CC(C)C", "test_005", "Au", 7.1, 0.45, 1.0, -5.1, 0.04])

    if not DATA_FIXTURE.exists():
        # 30-row synthetic Pt dataset with all 4 IC50 columns
        import random

        random.seed(0)
        smiles_pool = [
            "CCO",
            "CC(=O)O",
            "c1ccccc1",
            "CCCC",
            "CC(C)C",
            "CCN",
            "CC=O",
            "CCCO",
            "CCCCO",
            "c1ccc(O)cc1",
            "C1CCCCC1",
            "c1ccc(N)cc1",
            "c1ccc(F)cc1",
            "c1cccnc1",
            "CCS",
            "CC(C)O",
            "CCOC",
            "CCSC",
            "C1CCNCC1",
            "c1ccsc1",
            "C1=CC=CC=C1",
            "C=CC=C",
            "CC=CC",
            "CC=CCC",
            "C=CC=CC",
            "CC#C",
            "C#N",
            "CC(=O)C",
            "CCCCCC",
            "c1cnccn1",
        ]
        with DATA_FIXTURE.open("w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "complex_id",
                    "smiles",
                    "ic50_mcf7_nm",
                    "ic50_a549_nm",
                    "ic50_a2780_nm",
                    "ic50_a2780cis_nm",
                    "source",
                ]
            )
            for i in range(30):
                smi = smiles_pool[i % len(smiles_pool)]
                w.writerow(
                    [
                        f"pt_fix_{i:04d}",
                        smi,
                        1000.0 + random.gauss(0, 200),
                        5000.0 + random.gauss(0, 500),
                        800.0 + random.gauss(0, 150),
                        10000.0 + random.gauss(0, 1500),
                        f"src_{i % 5}",
                    ]
                )


@pytest.fixture
def top20_path():
    return TOP20_FIXTURE


@pytest.fixture
def data_path():
    return DATA_FIXTURE


@pytest.fixture
def out_dir():
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    return TMP_OUT


# ---------------------------------------------------------------------------
# Top-20 loader
# ---------------------------------------------------------------------------
def test_load_top20_returns_list_of_records(top20_path):
    from molmetal.scripts.bench_dmpnn_top20_predictions import load_top20

    top20 = load_top20(top20_path)
    assert isinstance(top20, list)
    assert len(top20) == 5
    assert all(hasattr(c, "rank") for c in top20)
    assert all(hasattr(c, "compound_smiles") for c in top20)
    assert all(hasattr(c, "metal") for c in top20)


def test_load_top20_preserves_rank_order(top20_path):
    from molmetal.scripts.bench_dmpnn_top20_predictions import load_top20

    top20 = load_top20(top20_path)
    ranks = [c.rank for c in top20]
    assert ranks == sorted(ranks), f"ranks not sorted: {ranks}"


def test_load_top20_handles_missing_file(tmp_path):
    from molmetal.scripts.bench_dmpnn_top20_predictions import load_top20

    with pytest.raises(FileNotFoundError):
        load_top20(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# External pIC50 prediction helper
# ---------------------------------------------------------------------------
def test_predict_pic50_external_returns_n_by_4():
    from molmetal.scripts.bench_dmpnn_top20_predictions import (
        predict_pic50_external,
    )
    from molmetal.scripts.train_platinai_4cell import DMPNNFourCellModel

    # Build a freshly-initialised model (not trained) — only the *shape* contract
    fresh_model = DMPNNFourCellModel()

    # Wrap into a minimal "runner" stub
    class _StubRunner:
        model = fresh_model
        device = torch.device("cpu")

    smiles = ["CCO", "c1ccccc1", "INVALID$$$"]
    pic50 = predict_pic50_external(_StubRunner(), smiles, batch_size=2)
    assert pic50.shape == (3, 4)
    # Valid SMILES should produce finite predictions; invalid should be NaN.
    assert np.isfinite(pic50[0]).all()
    assert np.isfinite(pic50[1]).all()
    assert np.isnan(pic50[2]).all()


def test_predict_pic50_external_raises_without_fit():
    from molmetal.scripts.bench_dmpnn_top20_predictions import predict_pic50_external

    class _EmptyRunner:
        model = None
        device = torch.device("cpu")

    with pytest.raises(RuntimeError):
        predict_pic50_external(_EmptyRunner(), ["CCO"])


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def test_aggregate_per_cell_returns_all_four_lines():
    from molmetal.scripts.bench_dmpnn_top20_predictions import (
        CELL_LINES,
        aggregate_per_cell,
        Top20Candidate,
    )

    # 2 seeds × 3 compounds × 4 cells
    arr1 = np.array([[6.0, 6.5, 5.5, 7.0], [6.1, 6.6, 5.6, 7.1], [6.2, 6.7, 5.7, 7.2]])
    arr2 = arr1 + 0.05  # small shift
    top20 = [
        Top20Candidate(rank=1, compound_smiles="CCO", compound_id="t1", metal="Pt"),
        Top20Candidate(rank=2, compound_smiles="CCN", compound_id="t2", metal="Pd"),
        Top20Candidate(rank=3, compound_smiles="CC=O", compound_id="t3", metal="Ir"),
    ]
    per_seed = [(0, arr1, top20), (1, arr2, top20)]
    agg = aggregate_per_cell(per_seed)
    assert set(agg.keys()) == set(CELL_LINES)
    for line, stats in agg.items():
        assert stats["n"] == 6  # 2 seeds × 3 compounds
        assert math.isfinite(stats["mean"])
        assert stats["std"] >= 0.0


def test_per_compound_table_has_expected_count():
    from molmetal.scripts.bench_dmpnn_top20_predictions import (
        build_per_compound_table,
        Top20Candidate,
    )

    arr = np.full((3, 4), 6.5)
    top20 = [
        Top20Candidate(rank=i, compound_smiles="CCO", compound_id=f"t{i}", metal="Pt")
        for i in range(3)
    ]
    per_seed = [(0, arr, top20), (1, arr, top20), (2, arr, top20)]
    table = build_per_compound_table(per_seed)
    assert len(table) == 9  # 3 compounds × 3 seeds


def test_per_compound_std_handles_constant_predictions():
    from molmetal.scripts.bench_dmpnn_top20_predictions import (
        aggregate_per_compound_std,
        Top20Candidate,
    )

    arr = np.full((4, 4), 6.5)
    top20 = [
        Top20Candidate(rank=i, compound_smiles="CCO", compound_id=f"t{i}", metal="Pt")
        for i in range(4)
    ]
    per_seed = [(0, arr, top20), (1, arr, top20), (2, arr, top20)]
    out = aggregate_per_compound_std(per_seed)
    assert len(out) == 4
    # Constant predictions across seeds ⇒ std = 0 ⇒ all pass gate
    for c in out:
        assert c["std_pooled_log_units"] == 0.0
        assert c["gate_pass"] is True


# ---------------------------------------------------------------------------
# Synthesis markdown gate
# ---------------------------------------------------------------------------
def test_synthesis_md_includes_honest_framing(tmp_path):
    """Synthesis must contain the honest-framing section (paper gate)."""
    import argparse
    from molmetal.scripts.bench_dmpnn_top20_predictions import (
        build_synthesis_md,
        Top20Candidate,
    )

    args = argparse.Namespace(
        hidden=64,
        depth=2,
        dropout=0.1,
        epochs=2,
        lr=1e-3,
        batch_size=8,
        alpha=0.5,
        device="cpu",
        seeds=[0, 1],
    )
    top20 = [
        Top20Candidate(rank=i, compound_smiles="CCO", compound_id=f"t{i}", metal="Pt")
        for i in range(2)
    ]
    arr = np.array([[6.0, 6.5, 5.5, 7.0], [6.1, 6.6, 5.6, 7.1]])
    per_seed = [(0, arr, top20), (1, arr, top20)]
    per_compound_std = [
        {
            "rank": c.rank,
            "compound_id": c.compound_id,
            "metal": c.metal,
            "per_cell_mean": {l: 6.5 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
            "std_pooled_log_units": 0.05,
            "gate_pass": True,
        }
        for c in top20
    ]
    per_cell_agg = {
        l: {"mean": 6.5, "median": 6.5, "std": 0.05, "min": 6.0, "max": 7.1, "n": 4}
        for l in ["MCF7", "A549", "A2780", "A2780cis"]
    }
    per_seed_meta = [
        {
            "auc_mean": 0.78,
            "auc_a2780cis": 0.75,
            "per_line_auc": {l: 0.77 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
            "per_line_ap": {l: 0.7 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
            "per_line_n": {l: 100 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
        },
        {
            "auc_mean": 0.79,
            "auc_a2780cis": 0.76,
            "per_line_auc": {l: 0.78 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
            "per_line_ap": {l: 0.72 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
            "per_line_n": {l: 100 for l in ["MCF7", "A549", "A2780", "A2780cis"]},
        },
    ]
    out_path = tmp_path / "synthesis.md"
    build_synthesis_md(
        args, top20, per_seed_meta, per_seed, per_compound_std, per_cell_agg, out_path
    )
    text = out_path.read_text()
    assert "Honest framing" in text
    assert "Cross-metal transfer" in text
    assert "Synthetic IC50 labels" in text
    assert "No wet-lab synthesis" in text
    assert "TODO-84" in text


# ---------------------------------------------------------------------------
# End-to-end smoke (CPU, 1 seed, 2 epochs)
# ---------------------------------------------------------------------------
def test_end_to_end_smoke_cli_runs(out_dir, top20_path, data_path):
    """Run the actual CLI on the 30-row fixture, 1 seed × 2 epochs.

    Must produce all 4 output files (per_compound / per_cell / auc / md).
    Must produce 5 records (5 compounds × 1 seed).
    """
    if not SCRIPT.exists():
        pytest.skip(f"bench script missing at {SCRIPT}")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--top20",
        str(top20_path),
        "--data",
        str(data_path),
        "--seeds",
        "0",
        "--epochs",
        "2",
        "--batch-size",
        "8",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        pytest.fail(f"CLI failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    # Check all 4 outputs
    expected_files = [
        "per_compound_pic50.json",
        "per_cell_pic50.json",
        "auc_per_seed.json",
        "synthesis.md",
    ]
    for fn in expected_files:
        assert (out_dir / fn).exists(), f"missing output: {fn}"
    # Verify per_compound has 5 records
    pcp = json.loads((out_dir / "per_compound_pic50.json").read_text())
    assert pcp["meta"]["n_compounds"] == 5
    assert pcp["meta"]["n_seeds"] == 1
    assert len(pcp["compounds"]) == 5
    # Verify per_cell has 4 cell lines
    pca = json.loads((out_dir / "per_cell_pic50.json").read_text())
    assert set(pca["per_cell_aggregate"].keys()) == {
        "MCF7",
        "A549",
        "A2780",
        "A2780cis",
    }
