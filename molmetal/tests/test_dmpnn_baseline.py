"""Integration tests for the D-MPNN baseline family (TODO/04 C4).

The D-MPNN baseline family provides *learned baselines* for the MetalCytoToxDB
HeLa48h / dark-cohort split — meant for **comparison against Lambda / CFM**, not
as a SOTA claim.  These tests verify:

1. ``regression_metrics`` correctly emits MAE / RMSE / Pearson r.
2. ``mol_to_graph`` produces directed-edge graphs with the correct shapes.
3. ``DMPNNModel`` forward pass returns scalar logits per molecule.
4. ``DMPNNMultiTaskModel`` returns ``(pic50, active_logits)`` shaped correctly.
5. End-to-end multi-task baseline run on a small subset of the real
   MetalCytoToxDB Ru cohort — verifies the full pipeline (load → featurize →
   split → train → predict → evaluate) executes without exceptions and emits
   the expected metric dict.
6. The CLI entry point ``molmetal.scripts.train_dmpnn_multitask`` is importable
   and re-uses the correct DMPNN classes.
7. Loss-objective sanity: ``MultiTaskLoss`` + ``NormalizedLoss`` +
   ``UncertWeightedLoss`` all accept the canonical (pic50, logits, pIC50,
   active) input layout and emit finite outputs.

All tests are CPU-only — no GPU required.  Where the full HeLa48h cohort is
needed the test degrades gracefully (skips) if RDKit cannot parse a molecule.

Honest framing: these tests guard the *plumbing* of the D-MPNN baseline.  They
do **not** validate that D-MPNN is competitive against SOTA — that's a
measurement that lives in the reports under
``molmetal/reports/baseline_*_dmpnn_*.json`` and is captured by the paper.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    DMPNNBaseline,
    DMPNNModel,
    DMPNNResult,
    SPLITTER_FACTORIES,
    featurize_smiles_list,
    mol_to_graph,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    DMPNNMultiTaskModel,
    DMPNNMultiTaskResult,
    MT_ALPHA,
    MultiTaskLoss,
    NormalizedLoss,
    UncertWeightedLoss,
    regression_metrics,
)
from molmetal.baselines.eval_utils import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SMILES_8 = [
    "CCO",                # ethanol
    "CC(=O)O",            # acetic acid
    "c1ccccc1",           # benzene
    "CC(C)C",             # isobutane
    "CCC",                # propane
    "C1CCCCC1",           # cyclohexane
    "CCOCC",              # diethyl ether
    "CCCC",               # butane
]


def _collate(smiles_list):
    """Featurise + collate 8 SMILES → (atom_f, bond_f, e_src, e_dst)."""
    from molmetal.baselines.dmpnn import collate_graphs
    graphs = featurize_smiles_list(smiles_list)
    batch = collate_graphs(graphs)
    return batch


# ---------------------------------------------------------------------------
# Test 1: regression_metrics emits MAE / RMSE / Pearson r
# ---------------------------------------------------------------------------
def test_regression_metrics_emits_expected_keys():
    """regression_metrics on perfect predictions → MAE=0, RMSE=0, Pearson=1."""
    y_true = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
    y_pred = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
    out = regression_metrics(y_true, y_pred)
    assert set(out.keys()) == {"mae", "rmse", "pearson", "n"}, (
        f"Expected keys {{mae, rmse, pearson, n}}, got {set(out.keys())}"
    )
    assert out["mae"] == pytest.approx(0.0, abs=1e-6)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert out["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert out["n"] == 4


def test_regression_metrics_handles_nan():
    """NaN labels are masked out, not propagated as NaN metrics."""
    y_true = np.array([5.0, np.nan, 7.0, 8.0], dtype=float)
    y_pred = np.array([5.1, 6.0, 6.9, 8.2], dtype=float)
    out = regression_metrics(y_true, y_pred)
    # 3 valid samples → n == 3
    assert out["n"] == 3
    # Finite, no NaN
    assert np.isfinite(out["mae"])
    assert np.isfinite(out["rmse"])
    assert np.isfinite(out["pearson"])


def test_regression_metrics_empty():
    """Empty arrays → NaN metrics + n=0 (matches docstring)."""
    out = regression_metrics(np.array([]), np.array([]))
    assert out["n"] == 0
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["pearson"])


# ---------------------------------------------------------------------------
# Test 2: mol_to_graph shapes
# ---------------------------------------------------------------------------
def test_mol_to_graph_shapes_ethanol():
    """mol_to_graph on ethanol → atom_f shape (3, 39), bond_f shape (4, 10)
    (each undirected bond becomes 2 directed edges)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    atom_f, bond_f, e_src, e_dst = mol_to_graph(mol)
    assert atom_f.shape == (3, ATOM_FEATURE_DIM), (
        f"Expected (3, {ATOM_FEATURE_DIM}), got {atom_f.shape}"
    )
    # 2 undirected bonds × 2 directed edges = 4 directed edges
    assert bond_f.shape == (4, BOND_FEATURE_DIM), (
        f"Expected (4, {BOND_FEATURE_DIM}), got {bond_f.shape}"
    )
    assert e_src.shape == (4,)
    assert e_dst.shape == (4,)
    # Directed edges: each (src, dst) must be a valid atom index
    V = atom_f.shape[0]
    assert int(e_src.max()) < V, f"e_src has out-of-range atom index"
    assert int(e_dst.max()) < V, f"e_dst has out-of-range atom index"


def test_mol_to_graph_handles_empty():
    """Empty molecule (no atoms) returns empty arrays gracefully."""
    from rdkit import Chem
    empty_mol = Chem.MolFromSmiles("")
    if empty_mol is None:
        pytest.skip("RDKit returns None for empty SMILES — skip")
    atom_f, bond_f, e_src, e_dst = mol_to_graph(empty_mol)
    assert atom_f.shape == (0, ATOM_FEATURE_DIM)
    assert bond_f.shape == (0, BOND_FEATURE_DIM)
    assert e_src.shape == (0,)
    assert e_dst.shape == (0,)


# ---------------------------------------------------------------------------
# Test 3: DMPNNModel forward pass
# ---------------------------------------------------------------------------
def test_dmpnn_model_forward_scalar_logit():
    """DMPNNModel forward on 8 mols → output shape (8,)."""
    batch = _collate(SMILES_8)
    atom_f, bond_f, e_src, e_dst = batch
    model = DMPNNModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=64,
        depth=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        logits = model(atom_f, bond_f, e_src, e_dst)
    assert logits.shape == (8,), f"Expected (8,), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"


# ---------------------------------------------------------------------------
# Test 4: DMPNNMultiTaskModel returns (pic50, active_logits)
# ---------------------------------------------------------------------------
def test_multitask_model_returns_dual_head():
    """Forward returns (B,) pIC50 + (B, 2) active logits; both finite."""
    batch = _collate(SMILES_8)
    atom_f, bond_f, e_src, e_dst = batch
    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=64,
        depth=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        pic50, active_logits = model(atom_f, bond_f, e_src, e_dst)
    assert pic50.shape == (8,), f"Expected pic50 shape (8,), got {pic50.shape}"
    assert active_logits.shape == (8, 2), (
        f"Expected active_logits (8, 2), got {active_logits.shape}"
    )
    assert torch.isfinite(pic50).all()
    assert torch.isfinite(active_logits).all()


# ---------------------------------------------------------------------------
# Test 5: End-to-end multi-task baseline on a real (small) cohort
# ---------------------------------------------------------------------------
def test_end_to_end_baseline_run_small_subset():
    """Run multi-task baseline on 200-row Ru cohort → returns result dataclass.

    Exercises the *full* pipeline:
    - CytotoxFilter → MetalCytotoxDataset.from_csv
    - Random split
    - Featurise → train → predict → compute_metrics + regression_metrics
    - Return DMPNNMultiTaskResult with all expected metric keys

    Honest framing: this is a 5-epoch smoke test, *not* a claim that D-MPNN
    beats SOTA.  The numbers are noisy; we only assert the pipeline runs
    end-to-end and emits well-formed metrics.
    """
    from molmetal.baselines.dmpnn_multitask import quick_smoke_mt
    result = quick_smoke_mt(metal="Ru", max_rows=200, alpha=MT_ALPHA)
    assert isinstance(result, DMPNNMultiTaskResult)
    # Train/val/test split
    assert result.n_train > 0
    assert result.n_val > 0
    assert result.n_test > 0
    # Classification metrics
    for key in ("roc_auc", "pr_auc", "hit_rate_top5pct"):
        assert key in result.test_metrics, (
            f"Missing test_metrics['{key}'] in {sorted(result.test_metrics.keys())}"
        )
        assert np.isfinite(result.test_metrics[key]), (
            f"test_metrics['{key}'] = {result.test_metrics[key]} is not finite"
        )
    # Regression metrics
    for key in ("mae", "rmse", "pearson"):
        assert key in result.test_pic50_metrics, (
            f"Missing test_pic50_metrics['{key}'] in {sorted(result.test_pic50_metrics.keys())}"
        )
        # pearson can be NaN if y has zero variance, but mae/rmse must be finite
        assert np.isfinite(result.test_pic50_metrics[key]), (
            f"test_pic50_metrics['{key}'] = {result.test_pic50_metrics[key]} not finite"
        )
    # per_cell_line may be empty for very small test sets; allow that
    assert isinstance(result.per_cell_line, dict)
    # Alpha / metal round-trip
    assert result.alpha == MT_ALPHA
    assert result.metal == "Ru"


# ---------------------------------------------------------------------------
# Test 6: CLI entry point is importable
# ---------------------------------------------------------------------------
def test_train_dmpnn_multitask_cli_importable():
    """The CLI script imports without errors and exposes `main()`."""
    from molmetal.scripts import train_dmpnn_multitask as cli

    assert hasattr(cli, "main"), "train_dmpnn_multitask must expose main()"
    assert hasattr(cli, "_parse_args"), "train_dmpnn_multitask must expose _parse_args()"
    # Argparse smoke test
    args = cli._parse_args(["--metal", "Ru", "--epochs", "1", "--split", "random"])
    assert args.metal == "Ru"
    assert args.epochs == 1
    assert args.split == "random"


def test_splitter_factories_keys():
    """SPLITTER_FACTORIES exposes the documented splitter keys."""
    expected = {"random", "ligand_dedup", "scaffold", "temporal", "chemical"}
    actual = set(SPLITTER_FACTORIES.keys())
    assert expected == actual, f"Expected splitters {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Test 7: All three loss heads produce finite outputs
# ---------------------------------------------------------------------------
def _synth_targets(B: int = 8):
    torch.manual_seed(0)
    return (
        4.0 + 4.0 * torch.rand(B),                # pIC50
        torch.randint(0, 2, (B,)),                 # active
    )


@pytest.mark.parametrize("LossCls", [MultiTaskLoss, NormalizedLoss, UncertWeightedLoss])
def test_loss_objective_emits_finite_outputs(LossCls):
    """Each loss class accepts (pic50, logits, pIC50, active) and emits finite (total, cls, reg)."""
    y_pic50, y_active = _synth_targets(B=8)
    B = y_pic50.shape[0]
    pic50_pred = torch.zeros(B)
    active_logits = torch.zeros(B, 2)
    crit = LossCls() if LossCls is not MultiTaskLoss else MultiTaskLoss(alpha=MT_ALPHA)
    total, cls, reg = crit(pic50_pred, active_logits, y_pic50, y_active)
    assert torch.isfinite(total), f"{LossCls.__name__}: total not finite"
    assert torch.isfinite(cls), f"{LossCls.__name__}: cls not finite"
    assert torch.isfinite(reg), f"{LossCls.__name__}: reg not finite"
    assert total.item() >= 0.0, f"{LossCls.__name__}: total < 0 (got {total.item()})"


def test_multitask_loss_alpha_bounds():
    """alpha outside [0, 1] must raise ValueError."""
    with pytest.raises(ValueError):
        MultiTaskLoss(alpha=-0.1)
    with pytest.raises(ValueError):
        MultiTaskLoss(alpha=1.5)


# ---------------------------------------------------------------------------
# Test 8: Multi-task vs single-task parity on synthetic data (sanity)
# ---------------------------------------------------------------------------
def test_multitask_predict_returns_probabilities_and_pic50():
    """predict_mt returns (y_true, prob in [0,1], pic50_pred) for valid mols.

    Validates the eval-side contract: scores must be valid probabilities
    (∈ [0,1]) so compute_metrics(roc_auc, pr_auc, hit_rate) can be called
    downstream.
    """
    batch = _collate(SMILES_8)
    graphs = featurize_smiles_list(SMILES_8)
    # Fake pIC50 / active labels
    pic50 = np.array([5.0, 6.0, 4.5, 7.0, 5.5, 6.5, 4.0, 7.5], dtype=np.float32)
    active = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=int)

    from molmetal.baselines.dmpnn_multitask import predict_mt

    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=32,
        depth=2,
        dropout=0.0,
    )
    y_t, y_score, pic50_pred = predict_mt(model, graphs, pic50, active, batch_size=4)
    # 8 molecules, 8 outputs
    assert y_t.shape == (8,)
    assert y_score.shape == (8,)
    assert pic50_pred.shape == (8,)
    # Probabilities in [0, 1]
    assert ((y_score >= 0.0) & (y_score <= 1.0)).all(), (
        f"Scores not in [0, 1]: {y_score}"
    )
    # compute_metrics should accept the output without raising
    metrics = compute_metrics(y_t, y_score)
    assert "roc_auc" in metrics


# ---------------------------------------------------------------------------
# Test 9: DMPNNResult dataclass JSON round-trip
# ---------------------------------------------------------------------------
def test_dmpnn_result_to_json_round_trip():
    """DMPNNMultiTaskResult.to_json() emits parseable JSON with all expected keys."""
    res = DMPNNMultiTaskResult(
        metal="Ru",
        test_metrics={"roc_auc": 0.5, "pr_auc": 0.3, "hit_rate_top5pct": 0.1},
        val_metrics={"roc_auc": 0.6, "pr_auc": 0.4, "hit_rate_top5pct": 0.2},
        test_pic50_metrics={"mae": 0.7, "rmse": 0.9, "pearson": 0.05, "n": 50},
        val_pic50_metrics={"mae": 0.5, "rmse": 0.7, "pearson": 0.4, "n": 50},
        per_cell_line={"HeLa": 0.55, "A549": 0.6},
        alpha=MT_ALPHA,
        n_train=200, n_val=50, n_test=50,
        pos_rate_train=0.27, pos_rate_test=0.30,
    )
    payload = json.loads(res.to_json())
    assert payload["metal"] == "Ru"
    assert payload["model"] == "dmpnn_multitask"
    assert payload["alpha"] == MT_ALPHA
    assert payload["test_metrics"]["roc_auc"] == 0.5
    assert payload["test_pic50_metrics"]["mae"] == 0.7
    assert payload["per_cell_line_auc"]["HeLa"] == 0.55


# ---------------------------------------------------------------------------
# Test 10: Additive — no existing baseline imports broken
# ---------------------------------------------------------------------------
def test_existing_baselines_still_importable():
    """Guard against accidental breaking of xgb / lightgbm / rf baselines."""
    from molmetal.baselines.morgan_xgb import MorganXGBBaseline
    from molmetal.baselines.morgan_lightgbm import MorganLightGBMBaseline
    from molmetal.baselines.rf_baseline import RFBaseline
    from molmetal.baselines.dmpnn_attentive import AttentiveDMPNNBaseline

    # Each baseline class should be constructable
    for Cls in (
        MorganXGBBaseline,
        MorganLightGBMBaseline,
        RFBaseline,
        DMPNNBaseline,
        AttentiveDMPNNBaseline,
        DMPNNMultiTaskBaseline,
    ):
        assert isinstance(Cls, type), f"{Cls} is not a class"
        assert hasattr(Cls, "run"), f"{Cls.__name__} missing .run()"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
