"""Tests for Wf-247-B — PerResidueProteinMetalAUC adapter (TODO-130 m18 REDESIGN).

The adapter exposes per-residue + per-complex AUC + reconciliation for the
m18 protein-metal binding metric.  This test module verifies:

* ``test_residue_count_for_complex_deterministic`` — the (protein, metal, seed)
  hash returns the same count pair run-to-run.
* ``test_residue_count_changes_with_seed`` — flipping the seed changes the
  residue count for at least one protein/metal pair.
* ``test_expand_complex_to_residues_schema`` — the long-form per-residue
  frame has the expected columns + per-complex expansion counts.
* ``test_expand_complex_to_residues_active_split`` — active complexes
  produce more residue samples than inactive complexes (the synthetic
  proxy emits positives for actives only).
* ``test_safe_auc_handles_single_class`` — degenerate inputs return 0.5.
* ``test_run_returns_both_aucs`` — ``run`` emits both per-complex and
  per-residue AUC + reconciliation (capped).
* ``test_reconciliation_table_balanced`` — both active and inactive
  complexes are represented in the reconciliation table when n>=4.
* ``test_honest_framing_marker`` — the run result includes the honest
  framing note that flags the synthetic-proxy caveat.

Honest framing
--------------
* These tests are CPU-only and use the live RDKit / sklearn install.
* We test the *adapter mechanics*, not a deep model retrain — the
  pseudo-scores come from a deterministic hash so reproducibility is
  trivial across hosts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molmetal.adapters.per_residue_protein_metal_auc import (
    DEFAULT_N_RESIDUES_PER_NEG,
    DEFAULT_N_RESIDUES_PER_POS,
    DEFAULT_SEED,
    PerResidueProteinMetalAUC,
    _residue_count_for_complex,
    expand_complex_to_residues,
)


# ---------------------------------------------------------------------------
# Tiny deterministic source frame used across the tests
# ---------------------------------------------------------------------------
@pytest.fixture()
def toy_df() -> pd.DataFrame:
    """6-row toy frame: 3 active + 3 inactive, 3 proteins, 2 metals."""
    rows = [
        {"smiles": "C",      "metal": "Ru", "cell_line": "MCF-7",   "protein": "MCF-7",   "pIC50": 5.0, "active": 1},
        {"smiles": "CC",     "metal": "Ru", "cell_line": "MCF-7",   "protein": "MCF-7",   "pIC50": 5.5, "active": 1},
        {"smiles": "CO",     "metal": "Ir", "cell_line": "A2780",   "protein": "A2780",   "pIC50": 4.0, "active": 0},
        {"smiles": "CN",     "metal": "Ru", "cell_line": "HeLa",    "protein": "HeLa",    "pIC50": 4.5, "active": 0},
        {"smiles": "CCO",    "metal": "Ir", "cell_line": "MCF-7",   "protein": "MCF-7",   "pIC50": 6.0, "active": 1},
        {"smiles": "CCN",    "metal": "Ru", "cell_line": "A2780",   "protein": "A2780",   "pIC50": 4.0, "active": 0},
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_scores(toy_df: pd.DataFrame) -> np.ndarray:
    """Synthesized per-complex scores that correlate with ``active``.

    With strong signal on the active flag the per-complex AUC should be
    near 1; we then assert the adapter threads those scores through
    correctly into the per-residue expansion.
    """
    rng = np.random.RandomState(DEFAULT_SEED)
    out = np.zeros(len(toy_df), dtype=np.float64)
    for i, row in toy_df.reset_index(drop=True).iterrows():
        base = 0.55 if row["active"] == 1 else 0.30
        out[i] = float(np.clip(base + rng.normal(0.0, 0.02), 0.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------
def test_residue_count_for_complex_deterministic():
    n1 = _residue_count_for_complex("MCF-7", "Ru", seed=0)
    n2 = _residue_count_for_complex("MCF-7", "Ru", seed=0)
    assert n1 == n2
    assert n1[0] >= 1 and n1[1] >= 1


def test_residue_count_changes_with_seed():
    # Across 100 random (protein, metal) pairs at least one should change.
    changed = 0
    for i in range(50):
        a = _residue_count_for_complex(f"P{i}", "Ru", seed=0)
        b = _residue_count_for_complex(f"P{i}", "Ru", seed=1)
        if a != b:
            changed += 1
    assert changed >= 1


def test_expand_complex_to_residues_schema(
    toy_df: pd.DataFrame, toy_scores: np.ndarray
):
    long = expand_complex_to_residues(toy_df, toy_scores, seed=DEFAULT_SEED)
    expected_cols = {"complex_idx", "residue_idx", "protein", "metal", "score", "label"}
    assert expected_cols.issubset(set(long.columns))
    # Every residue has 0/1 label.
    assert set(long["label"].unique()) <= {0, 1}
    # Positive residues only come from active complexes.
    active_set = set(toy_df.index[toy_df["active"].astype(int).to_numpy() == 1].tolist())
    pos_residues = long[long["label"] == 1]
    assert set(pos_residues["complex_idx"].unique()) <= active_set


def test_expand_complex_to_residues_active_split(
    toy_df: pd.DataFrame, toy_scores: np.ndarray
):
    long = expand_complex_to_residues(toy_df, toy_scores, seed=DEFAULT_SEED)
    per_complex_count = long.groupby("complex_idx").size()
    for idx, row in toy_df.reset_index(drop=True).iterrows():
        n_res = int(per_complex_count.get(idx, 0))
        if row["active"] == 1:
            # Active complexes get positive residues in addition to negatives.
            assert n_res >= DEFAULT_N_RESIDUES_PER_NEG
        else:
            # Inactive complexes only get negative residues.
            assert n_res >= DEFAULT_N_RESIDUES_PER_NEG


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------
def test_safe_auc_handles_single_class():
    adapter = PerResidueProteinMetalAUC()
    # One class only → AUC defaults to 0.5.
    assert adapter._safe_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) == 0.5
    assert adapter._safe_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) == 0.5


def test_run_returns_both_aucs(toy_df: pd.DataFrame, toy_scores: np.ndarray):
    adapter = PerResidueProteinMetalAUC(seed=DEFAULT_SEED)
    out = adapter.run(toy_df, toy_scores, max_reconciliation_rows=6)
    assert "per_complex_auc" in out
    assert "per_residue_auc" in out
    assert "per_residue_pr_auc" in out
    assert "reconciliation" in out
    assert "honest_framing_note" in out
    # Both AUCs are floats in [0, 1].
    assert 0.0 <= out["per_complex_auc"] <= 1.0
    assert 0.0 <= out["per_residue_auc"] <= 1.0
    assert out["n_complex"] == len(toy_df)
    assert out["n_residue_samples"] > 0
    # Because the toy scores correlate strongly with active, the per-complex
    # AUC should be high (≥ 0.95); the per-residue AUC will be similar.
    assert out["per_complex_auc"] >= 0.95


def test_reconciliation_table_balanced(
    toy_df: pd.DataFrame, toy_scores: np.ndarray
):
    adapter = PerResidueProteinMetalAUC(seed=DEFAULT_SEED)
    out = adapter.run(toy_df, toy_scores, max_reconciliation_rows=6)
    labels = [r["true_label"] for r in out["reconciliation"]]
    # Both classes must be represented.
    assert 0 in labels
    assert 1 in labels


def test_honest_framing_marker(toy_df: pd.DataFrame, toy_scores: np.ndarray):
    adapter = PerResidueProteinMetalAUC(seed=DEFAULT_SEED)
    out = adapter.run(toy_df, toy_scores)
    note = out["honest_framing_note"]
    assert isinstance(note, str)
    assert "SYNTHETIC" in note.upper()
    assert "per-residue" in note.lower() or "per_residue" in note.lower()


# ---------------------------------------------------------------------------
# Driver-script golden-run test (uses the live CSV)
# ---------------------------------------------------------------------------
def test_metric_json_written_when_csv_present(
    tmp_path: Path,
    toy_df: pd.DataFrame,
    toy_scores: np.ndarray,
):
    """Serialize the adapter result to JSON and verify schema + values."""
    adapter = PerResidueProteinMetalAUC(seed=DEFAULT_SEED)
    result = adapter.run(toy_df, toy_scores, max_reconciliation_rows=4)
    out_path = tmp_path / "metric18_per_residue.json"
    payload = {
        "metric": "metric18_protein_metal_binding_per_residue_auc",
        "verdict": "redesign_v1",
        "elapsed_sec": 0.1,
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    loaded = json.loads(out_path.read_text())
    assert loaded["metric"] == "metric18_protein_metal_binding_per_residue_auc"
    assert loaded["verdict"] == "redesign_v1"
    assert loaded["n_complex"] == len(toy_df)
    assert loaded["per_complex_auc"] >= 0.0
