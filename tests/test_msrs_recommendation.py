"""Tests for the MSRS-6 recommendation utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts import msrs_recommendation as R


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def test_perfect_ranking_has_zero_regret():
    m = R.cohort_metrics([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert m["regret"] == 0.0
    assert m["top1_correct"] is True
    assert m["ndcg"] == pytest.approx(1.0)
    assert m["hit@0.1"] is True


def test_reversed_ranking_regret_equals_gap():
    m = R.cohort_metrics([1.0, 2.0], [2.1, 0.5])
    assert m["regret"] == pytest.approx(1.0)
    assert m["top1_correct"] is False
    assert m["hit@0.1"] is False
    assert m["hit@0.2091"] is False
    assert m["hit@0.5"] is False
    assert 0.0 < m["ndcg"] < 1.0


def test_hit_threshold_is_inclusive():
    # Correct pick, so regret is zero regardless of the gap.
    m = R.cohort_metrics([0.0, 0.1], [0.0, 1.0])
    assert m["hit@0.1"] is True
    # Wrong pick: regret is the gap, and the threshold is inclusive.
    m = R.cohort_metrics([0.0, 0.1], [1.0, 0.0])
    assert m["regret"] == pytest.approx(0.1)
    assert m["hit@0.1"] is True
    m = R.cohort_metrics([0.0, 0.11], [1.0, 0.0])
    assert m["hit@0.1"] is False


def test_ndcg_discounts_a_swapped_pair():
    # k=3, chosen metal is the true best, but ranks 2 and 3 are swapped.
    y = [0.0, 0.2091, 0.4182]
    pred = [0.0, 0.4182, 0.2091]
    m = R.cohort_metrics(y, pred)
    assert m["regret"] == 0.0
    assert m["ndcg"] < 1.0
    # gain(0)=1, gain(1)=0.5, gain(2)=0.25; dcg = 1 + 0.25/log2(3) + 0.5/log2(4)
    expected = (1 + 0.25 / np.log2(3) + 0.5 / np.log2(4)) / (1 + 0.5 / np.log2(3) + 0.25 / np.log2(4))
    assert m["ndcg"] == pytest.approx(expected, rel=1e-6)


def test_ndcg_uses_min_of_k_and_cohort_size():
    m2 = R.cohort_metrics([0.0, 0.2091], [0.0, 0.2091])
    assert m2["k"] == 2 and m2["ndcg"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def _frame(rows):
    return pd.DataFrame(rows)


def test_build_cohorts_pools_replicates_by_median():
    frame = _frame(
        [
            {"cohort": 1, "metal": "Ru", "y": 1.0, "doi": "d", "fold": 0, "n_compounds": 3, "additive": 1.1},
            {"cohort": 1, "metal": "Ru", "y": 2.0, "doi": "d", "fold": 0, "n_compounds": 3, "additive": 1.3},
            {"cohort": 1, "metal": "Ir", "y": 0.0, "doi": "d", "fold": 0, "n_compounds": 3, "additive": 0.2},
        ]
    )
    out = R.build_cohorts(frame, ("additive",))
    assert len(out) == 2
    ru = out[out["metal"] == "Ru"].iloc[0]
    assert ru["y"] == pytest.approx(1.5)
    assert ru["additive"] == pytest.approx(1.2)
    assert ru["n_rows"] == 2
    assert ru["n_compounds"] == 3


def test_build_cohorts_drops_singletons():
    frame = _frame(
        [
            {"cohort": 1, "metal": "Ru", "y": 1.0, "doi": "d", "fold": 0, "n_compounds": 1, "additive": 1.0},
            {"cohort": 1, "metal": "Ir", "y": 0.0, "doi": "d", "fold": 0, "n_compounds": 1, "additive": 0.0},
            {"cohort": 2, "metal": "Ru", "y": 1.0, "doi": "d", "fold": 1, "n_compounds": 1, "additive": 1.0},
        ]
    )
    out = R.build_cohorts(frame, ("additive",))
    assert set(out["cohort"]) == {1}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_fold_baseline_uses_training_fold_only():
    metals = np.array(["Ru", "Ir", "Ru", "Ir"])
    cells = np.array(["A", "A", "A", "A"])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    fold_id = np.array([0, 0, 1, 1])
    pred_global, pred_cell, notes = R.fold_baseline_scores(metals, cells, y, fold_id)
    # fold 0 is predicted from fold 1 (means 2 and 3), and vice versa
    assert pred_global.tolist() == pytest.approx([2.0, 3.0, 0.0, 1.0])
    assert pred_cell.tolist() == pytest.approx([2.0, 3.0, 0.0, 1.0])
    assert notes["rows_with_unseen_metal_in_train_fold"] == 0


def test_fold_baseline_falls_back_for_unseen_metal():
    metals = np.array(["Ru", "Ir", "Ru", "Re"])
    cells = np.array(["A", "A", "A", "A"])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    fold_id = np.array([0, 0, 1, 1])
    pred_global, _, notes = R.fold_baseline_scores(metals, cells, y, fold_id)
    # Re appears only in fold 1, so fold 0's Re score falls back to fold 1's mean.
    assert pred_global[1] == pytest.approx(2.5)
    # Ir is symmetric: absent from fold 1, so fold 1's Ir falls back to fold 0's mean.
    assert pred_global[3] == pytest.approx(0.5)
    assert notes["rows_with_unseen_metal_in_train_fold"] == 2


def test_fold_assignments_never_split_a_group():
    groups = np.array([f"g{i}" for i in range(50)])
    fold_id = R.fold_assignments(groups)
    for g in np.unique(groups):
        assert len(set(fold_id[groups == g])) == 1


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------
def _records(deltas, dois=None, model="additive", baseline="global_best"):
    """deltas: baseline regret - model regret, one entry per cohort."""
    dois = dois or [f"doi{i}" for i in range(len(deltas))]
    rows = []
    for i, d in enumerate(deltas):
        rows.append({"cohort": i, "scorer": baseline, "regret": d, "chosen": 0, "doi": dois[i]})
        rows.append({"cohort": i, "scorer": model, "regret": 0.0, "chosen": 0, "doi": dois[i]})
    return pd.DataFrame(rows)


def test_paired_test_detects_a_consistent_gain():
    rec = _records([0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52, 0.48])
    out = R.paired_regret_test(rec, "additive")
    assert out["mean_regret_delta"] == pytest.approx(0.5, abs=1e-6)
    assert out["p_one_sided_cohort"] < 0.001
    assert out["direction"] == "model better"


def test_paired_test_is_flat_when_models_agree():
    rec = _records([0.0, 0.0, 0.0, 0.0])
    out = R.paired_regret_test(rec, "additive")
    assert out["mean_regret_delta"] == 0.0
    assert out["p_one_sided_cohort"] == 1.0
    assert out["direction"] == "no difference"


def test_doi_clustering_kills_a_paper_level_illusion():
    # Two papers, 50 cohorts each. Within a paper the deltas alternate around a
    # constant, so the per-cohort t-test screams while the paper-level means are
    # identical. This is the exact failure mode clustering is meant to expose.
    deltas, dois = [], []
    for i in range(50):
        deltas.append(0.4 if i % 2 == 0 else 0.6)
        dois.append("A")
    for i in range(50):
        deltas.append(0.45 if i % 2 == 0 else 0.55)
        dois.append("B")
    rec = _records(deltas, dois)
    naive = R.paired_regret_test(rec, "additive")
    clustered = R.doi_clustered_test(rec, "additive")
    assert naive["p_one_sided_cohort"] < 1e-6
    assert clustered["n_dois"] == 2
    assert clustered["p_one_sided_doi"] > 0.5


def test_disagreement_decomposition_splits_by_pick():
    rows = [
        {"cohort": 0, "scorer": "global_best", "regret": 0.5, "chosen": 0, "doi": "d"},
        {"cohort": 0, "scorer": "additive", "regret": 0.0, "chosen": 1, "doi": "d"},
        {"cohort": 1, "scorer": "global_best", "regret": 0.0, "chosen": 0, "doi": "d"},
        {"cohort": 1, "scorer": "additive", "regret": 0.0, "chosen": 0, "doi": "d"},
    ]
    out = R.disagreement_decomposition(pd.DataFrame(rows), "additive")
    assert out["agreement_rate"] == pytest.approx(0.5)
    assert out["n_disagree"] == 1
    assert out["net_gain_when_disagree"] == pytest.approx(0.5)
    assert out["model_wins_disagreements"] == 1


# ---------------------------------------------------------------------------
# Full five-metal ranking
# ---------------------------------------------------------------------------
def test_full_five_counts_unverifiable_picks():
    records = pd.DataFrame(
        [
            {"cohort": 0, "metal": "Ru", "y": 1.0},
            {"cohort": 0, "metal": "Ir", "y": 2.0},
        ]
    )
    cf = pd.DataFrame(
        [{"cohort": 0, "metal": m} for m in R.METALS]
    )
    pred = np.array([0.0 if m == "Ru" else 9.0 for m in R.METALS])
    out = R.full_five_report(records, cf, pred)
    assert out["top1_is_observed_rate"] == 1.0
    assert out["mean_regret_verifiable"] == 0.0
    assert out["top1_metal_distribution"] == {"Ru": 1}

    pred_os = np.array([0.0 if m == "Os" else 9.0 for m in R.METALS])
    out = R.full_five_report(records, cf, pred_os)
    assert out["top1_is_observed_rate"] == 0.0
    assert out["n_verifiable"] == 0
    assert out["mean_regret_verifiable"] is None


# ---------------------------------------------------------------------------
# Structural invariance
# ---------------------------------------------------------------------------
def _cf_frame(n_cohorts: int = 2) -> pd.DataFrame:
    rows = []
    for c in range(n_cohorts):
        for m in R.METALS:
            rows.append({"cohort": c, "metal": m, "fold": 0})
    return pd.DataFrame(rows)


def test_additive_invariance_detects_a_constant_order():
    cf = _cf_frame()
    # Same score for a metal regardless of cohort: one order everywhere.
    pred = np.array([float(R.METALS.index(m)) for _ in range(2) for m in R.METALS])
    out = R.additive_ranking_invariance(cf, pred)
    assert out["constant_within_fold"] is True
    assert out["by_fold"]["0"]["n_distinct_orders"] == 1
    assert out["by_fold"]["0"]["modal_order"] == list(R.METALS)


def test_additive_invariance_flags_a_ligand_dependent_order():
    cf = _cf_frame()
    pred = []
    for c in range(2):
        for m in R.METALS:
            # Cohort 0 prefers Ru, cohort 1 prefers Ir.
            pred.append(0.0 if m == ("Ru" if c == 0 else "Ir") else 9.0)
    out = R.additive_ranking_invariance(cf, np.array(pred))
    assert out["constant_within_fold"] is False
    assert out["by_fold"]["0"]["n_distinct_orders"] == 2


def test_global_metal_order_uses_training_folds():
    df = pd.DataFrame({"metal": ["Ru", "Ir", "Ru", "Ir"]})
    y = np.array([0.0, 1.0, 2.0, 3.0])
    fold_id = np.array([0, 0, 1, 1])
    out = R.global_metal_order(df, y, fold_id)
    assert out["ascending"] == ["Ru", "Ir"]
    assert out["mean_y_by_metal"] == {"Ru": 1.0, "Ir": 2.0}


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def test_acceptance_requires_both_direction_and_significance():
    assert R.acceptance_decision({"mean_regret_delta": 0.2, "holm_p_across_primary_tests": 0.01})[0] is True
    assert R.acceptance_decision({"mean_regret_delta": 0.2, "holm_p_across_primary_tests": 0.2})[0] is False
    assert R.acceptance_decision({"mean_regret_delta": -0.2, "holm_p_across_primary_tests": 0.001})[0] is False
    assert R.acceptance_decision({"mean_regret_delta": None})[0] is False


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def _synthetic(n_ligands: int = 20) -> pd.DataFrame:
    smiles = [
        "CCO",
        "c1ccccc1",
        "CCN",
        "c1ccncc1",
        "CC(=O)O",
        "CC(C)O",
        "c1ccc(O)cc1",
        "CCCl",
        "c1ccsc1",
        "CC(C)(C)O",
        "c1ccc(N)cc1",
        "CCC(=O)O",
        "c1ccoc1",
        "COc1ccccc1",
        "CCBr",
        "c1ccc(F)cc1",
        "CCS",
        "c1cc[nH]c1",
        "CC(C)C",
        "c1ccc(C)cc1",
    ]
    rows = []
    for i in range(n_ligands):
        for metal, offset in (("Ru", 0.0), ("Ir", 0.6)):
            rows.append(
                {
                    "endpoint": "potency",
                    "value_log10": float(i % 5) * 0.1 + offset,
                    "ligand_main_fragment": smiles[i % len(smiles)],
                    "ligand_canonical": f"L{i:03d}",
                    "metal": metal,
                    "cell_line_key": "A549",
                    "doi": "10.0000/test",
                    "time_h": 48.0,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_end_to_end_structure():
    df = _synthetic()
    assert df["ligand_main_fragment"].nunique() == 20

    descriptors = pd.DataFrame(
        {
            "metal_symbol": list(R.METALS),
            "ie1_kj": [710.2, 880.0, 719.7, 840.0, 760.0],
            "ie2_kj": [1620.0, 1600.0, 1740.0, 1600.0, 1260.0],
            "pauling_en": [2.20, 2.20, 2.28, 2.20, 1.90],
        }
    )
    out = R.evaluate(df, "potency", descriptors, "strict", verbose=False)
    assert out["status"] == "OK"
    assert out["n_cohorts"] == 20
    assert out["cohort_size_distribution"] == {"2": 20}
    assert out["scores"]["oracle"]["mean_regret"] == 0.0
    assert out["scores"]["oracle"]["top1_accuracy"] == 1.0
    assert out["scores"]["additive"]["n_cohorts"] == 20
    assert out["full_five_metal"]["n_cohorts"] == 20
    assert out["full_five_metal"]["top1_is_observed_rate"] is not None
    # One DOI only, so the clustered test reports its degenerate branch.
    assert out["doi_clustered_vs_global_best"]["additive"]["n_dois"] == 1
