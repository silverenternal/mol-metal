# SPDX-License-Identifier: Apache-2.0
"""Wf-A97 (2026-10-19) — bond existence AP must not be the majority-class AP.

Between Wf-A70 and Wf-A97 the trainer set ``bond_pr_auc`` to the NO_BOND
one-vs-rest AP, on the claim that reversing the score preserves the ranking
and therefore the AP.  It does not: AP of the complement is not AP of the
class.  On a corpus where 70-94% of candidate pairs are NOT bonded, the
class-0 AP is trivially high and the headline read 0.92-0.98 while the same
pairs scored F1 0.00-0.11 at any global threshold.

MEASURED on cfm_fullcorpus_2026_10_10 over 16 real val clouds (2,141 pairs,
positive rate 0.298): existence AP 0.2651 vs class-0 AP 0.6366.
"""
from __future__ import annotations

import torch


def _trainer():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    return T


def _probs(rows):
    # One (E, 5) tensor per cloud: the function concatenates along dim 0.
    return [torch.tensor(rows, dtype=torch.float32)]


def test_constant_scores_return_the_base_rate_not_one():
    T = _trainer()
    # 8 negatives, 2 positives, identical distributions in both arms.
    probs = _probs([[0.9, 0.1, 0.0, 0.0, 0.0]] * 10)
    truth = [0] * 8 + [1] * 2
    out = T._bond_ranking_report_per_class(probs, truth)
    assert out["bond_pr_auc"] == 0.2  # existence base rate
    assert out["bond_pr_auc_class0"] == 0.8  # NO_BOND base rate
    assert out["bond_pr_auc"] != out["bond_pr_auc_class0"]


def test_a_ranker_that_only_orders_no_bond_cannot_look_good_on_existence():
    T = _trainer()
    # MEASURED-style counterexample: the class-0 ordering is mostly right
    # (0.833) while the existence ordering is much worse (0.500).  Perfect
    # class-0 ordering would imply perfect existence ordering, so the gap
    # only shows up when the ranking is partial -- which is the real case.
    # Found by search over 20k random 10-pair cases; the class-0 AP is 0.911
    # while the existence AP is 0.143.  That is the real failure mode: the
    # single true bond sits at P(bonded)=0.32, below seven non-bonds, so the
    # existence ranking is bad even though the no-bond ordering looks great.
    p_bonded = [0.84, 0.57, 0.23, 0.84, 0.34, 0.90, 0.34, 0.32, 0.17, 0.09]
    truth = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    probs = _probs([[1.0 - s, s, 0.0, 0.0, 0.0] for s in p_bonded])
    out = T._bond_ranking_report_per_class(probs, truth)
    assert out["bond_class_0_pr_auc"] > 0.85
    assert out["bond_pr_auc"] < 0.25, out["bond_pr_auc"]
    assert out["bond_existence_ap"] == out["bond_pr_auc"]


def test_best_f1_sweep_reaches_thresholds_below_005():
    T = _trainer()
    # Positives sit at 0.02, negatives at 0.01: the optimum is at ~0.02,
    # which the old 0.05-start grid could not see.
    probs = _probs(
        [[0.98, 0.02, 0.0, 0.0, 0.0]] * 10
        + [[0.99, 0.01, 0.0, 0.0, 0.0]] * 10
    )
    truth = [1] * 10 + [0] * 10
    out = T._bond_ranking_report_per_class(probs, truth)
    assert out["bond_best_f1_threshold"] <= 0.03
    assert out["bond_best_f1_at_threshold"] == 1.0
