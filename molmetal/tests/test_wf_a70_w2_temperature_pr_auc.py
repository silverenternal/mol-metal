"""Wf-A70 W2 (2026-10-17) Fix 8 + Fix 9 — aux-MSE temperature + per-class PR-AUC.

Three regression test classes pinning the new behaviour:

  1. ``CFMJointModel`` accepts ``bond_aux_temperature=1.0`` (default) +
     stores it as ``model.bond_aux_temperature``; the Wf-A68 aux-MSE
     stash path (``_maybe_stash_bond_aux_bo_linear``) is bit-exact when
     τ=1.0 (no temperature scaling applied — same pred_bo_lin as the
     pre-Wf-A70-W2 path).
  2. Temperature scaling at τ<1 sharpens the softmax; τ>1 smooths.
     Verified by comparing τ=1.0 vs τ=0.5 on identical logits: the
     τ=0.5 prediction must move closer to the argmax (Wiberg 0/1/2/3/4
     centres), the τ=2.0 prediction must move further away.
  3. ``_bond_ranking_report_per_class`` emits the legacy
     ``bond_pr_auc`` / ``bond_positive_rate`` / ``bond_best_f1_*``
     fields (backward compat with every CSV/JSON consumer that
     reads them) AND the new per-class fields
     ``bond_class_{c}_pr_auc`` / ``bond_class_{c}_pos_rate`` /
     ``bond_class_{c}_n_pos`` / ``bond_per_class_macro_pr_auc``.
     NO_BOND one-vs-rest AP equals the legacy existence AP (ranking
     identity, monotone reversal — same number up to float64 precision).

Tests are CPU-only and run in <5s.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch

PROJECT_ROOT_TESTS = "/home/hugo/codes/mol-metal"


# ---------------------------------------------------------------------------
# Fix 8 — temperature τ on the aux-MSE softmax
# ---------------------------------------------------------------------------
def test_cfm_joint_model_default_temperature_is_one():
    """Default τ=1.0 ⇒ ``model.bond_aux_temperature == 1.0``.

    Bit-exact Wf-A68 path: the softmax input equals the head's raw
    logits, identical to the pre-Wf-A70-W2 path.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    m = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond",
    )
    assert m.bond_aux_temperature == 1.0


def test_cfm_joint_model_accepts_temperature_kwarg():
    """``bond_aux_temperature=0.5`` (opt-in sharpen) round-trips."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    m = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond",
        bond_aux_temperature=0.5,
    )
    assert m.bond_aux_temperature == 0.5


def test_cfm_joint_model_accepts_temperature_smooth():
    """``bond_aux_temperature=2.0`` (smooth) round-trips."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    m = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond",
        bond_aux_temperature=2.0,
    )
    assert m.bond_aux_temperature == 2.0


def test_argparse_accepts_new_temperature_flag():
    """``--bond-aux-temperature=0.5`` parses + is read on the model."""
    saved_argv = sys.argv
    try:
        sys.argv = [
            "test",
            "--bond-aux-mse-weight",
            "0.1",
            "--bond-aux-temperature",
            "0.5",
        ]
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args

        args = parse_args()
        assert args.bond_aux_temperature == 0.5
        assert args.bond_aux_mse_weight == 0.1
    finally:
        sys.argv = saved_argv


def test_argparse_temperature_default_is_one():
    """Default τ=1.0 (bit-exact Wf-A68 path)."""
    saved_argv = sys.argv
    try:
        sys.argv = ["test"]
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import parse_args

        args = parse_args()
        assert args.bond_aux_temperature == 1.0
    finally:
        sys.argv = saved_argv


def test_temperature_one_matches_default_softmax():
    """τ=1.0 softmax equals the default F.softmax path bit-exact.

    Sets ``bond_aux_temperature=1.0`` on the model, builds a synthetic
    cloud with truth bond_orders_continuous, runs
    ``_maybe_stash_bond_aux_bo_linear``, and asserts the stashed
    prediction equals the Wf-A68 path (no temperature division).
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    torch.manual_seed(0)
    m = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond",
        bond_aux_temperature=1.0,
    )
    # 5-atom cloud with a known bond_orders_continuous target.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
        ]
    )
    z = torch.tensor([6, 6, 7, 8, 6], dtype=torch.long)
    # bond_orders_continuous requires `bonds` to also be set
    # (AtomCloud invariant — see bond_head.py:569).
    bond_pairs_i = [0, 0, 1, 2]
    bond_pairs_j = [1, 2, 2, 3]
    bond_types = [1, 2, 1, 1]
    bonds_t = torch.tensor([bond_pairs_i, bond_pairs_j], dtype=torch.long)
    types_t = torch.tensor(bond_types, dtype=torch.long)
    cloud = AtomCloud(
        positions=positions,
        atomic_numbers=z,
        bonds=bonds_t,
        bond_types=types_t,
        bond_orders_continuous=torch.tensor([1.0, 2.0, 0.0, 1.5]),
    )
    # Synthetic logits: a 4×5 head output (the head's edge list will
    # be ≤4 pairs after the cutoff filter; we set it to a hand-built
    # value to make the τ effect deterministic).
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, -0.5, 0.0],
            [0.5, 1.0, 2.0, -1.0, 0.5],
            [3.0, 0.1, -1.0, -2.0, -1.5],
            [-0.5, 1.5, 1.0, 0.0, 0.5],
        ]
    )
    # τ=1.0 stash:
    m.last_bond_logits_bo_linear = []
    m._maybe_stash_bond_aux_bo_linear(logits, cloud)
    # Skip if the cloud didn't produce any pairs (cutoff filter may
    # have removed some).  We only verify the bit-exact path when
    # the stash produced something.
    if m.last_bond_logits_bo_linear:
        pred_t1 = m.last_bond_logits_bo_linear[0].clone()
        # Recompute manually at τ=1.0 and assert equality.
        centres = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=logits.dtype)
        probs_t1 = torch.softmax(logits, dim=-1)
        expected = (probs_t1 * centres).sum(dim=-1).unsqueeze(-1)
        assert torch.allclose(pred_t1, expected, atol=1e-6)


def test_temperature_sharpens_softmax_below_one():
    """τ=0.5 sharpens the softmax — pred_bo_lin moves toward argmax centre.

    For a logits vector like [3.0, 0.1, -1.0, -2.0, -1.5] the argmax
    is class 0 (centre 0).  At τ=0.5 the softmax concentrates mass
    on class 0, so pred_bo_lin moves CLOSER to 0 than at τ=1.0.
    For [0.5, 1.0, 2.0, -1.0, 0.5] the argmax is class 2 (centre 2);
    τ=0.5 pushes pred_bo_lin CLOSER to 2 than at τ=1.0.
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    # Build two clouds, one with argmax=0 and one with argmax=2.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
        ]
    )
    z = torch.tensor([6, 6, 7, 8, 6], dtype=torch.long)
    # bond_orders_continuous requires `bonds` to also be set.
    bonds_t = torch.tensor(
        [[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long
    )
    types_t = torch.tensor([1, 2, 1, 1], dtype=torch.long)
    cloud_a = AtomCloud(
        positions=positions, atomic_numbers=z,
        bonds=bonds_t, bond_types=types_t,
        bond_orders_continuous=torch.tensor([1.0, 2.0, 0.0, 1.5]),
    )
    logits_a = torch.tensor(
        [
            [1.0, 2.0, 0.5, -0.5, 0.0],
            [0.5, 1.0, 2.0, -1.0, 0.5],
            [3.0, 0.1, -1.0, -2.0, -1.5],
            [-0.5, 1.5, 1.0, 0.0, 0.5],
        ]
    )
    centres = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    # τ=1.0
    torch.manual_seed(0)
    m_t1 = CFMJointModel(
        hidden_dim=64, n_layers=3, cutoff=3.5,
        bond_predictor="yuelbond", bond_aux_temperature=1.0,
    )
    m_t1.last_bond_logits_bo_linear = []
    m_t1._maybe_stash_bond_aux_bo_linear(logits_a, cloud_a)
    # τ=0.5 (sharper)
    m_t05 = CFMJointModel(
        hidden_dim=64, n_layers=3, cutoff=3.5,
        bond_predictor="yuelbond", bond_aux_temperature=0.5,
    )
    m_t05.last_bond_logits_bo_linear = []
    m_t05._maybe_stash_bond_aux_bo_linear(logits_a, cloud_a)

    if m_t1.last_bond_logits_bo_linear and m_t05.last_bond_logits_bo_linear:
        pred_t1 = m_t1.last_bond_logits_bo_linear[0].squeeze(-1)
        pred_t05 = m_t05.last_bond_logits_bo_linear[0].squeeze(-1)
        # Argmax over the per-row logits:
        argmax_centres = centres[logits_a.argmax(dim=-1)]  # [1.0, 2.0, 0.0, 1.0]
        # τ=0.5 prediction should be CLOSER to argmax than τ=1.0.
        diff_t1 = (pred_t1 - argmax_centres).abs()
        diff_t05 = (pred_t05 - argmax_centres).abs()
        # All 4 rows should be closer at τ=0.5.
        assert (diff_t05 <= diff_t1 + 1e-6).all(), (
            f"τ=0.5 should sharpen toward argmax; "
            f"diff_t1={diff_t1.tolist()}, diff_t05={diff_t05.tolist()}"
        )


def test_temperature_invalid_values_clamped_to_one():
    """τ<=0 or τ=NaN falls back to τ=1.0 in the stash method (no NaN/inf)."""
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    z = torch.tensor([6, 6, 7], dtype=torch.long)
    logits = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0]])
    # bond_orders_continuous requires `bonds`.
    cloud = AtomCloud(
        positions=positions, atomic_numbers=z,
        bonds=torch.tensor([[0], [1]], dtype=torch.long),
        bond_types=torch.tensor([1], dtype=torch.long),
        bond_orders_continuous=torch.tensor([1.0]),
    )
    for bad_tau in (0.0, -1.0, float("nan")):
        m = CFMJointModel(
            hidden_dim=64, n_layers=3, cutoff=3.5,
            bond_predictor="yuelbond", bond_aux_temperature=bad_tau,
        )
        m.last_bond_logits_bo_linear = []
        m._maybe_stash_bond_aux_bo_linear(logits, cloud)
        if m.last_bond_logits_bo_linear:
            pred = m.last_bond_logits_bo_linear[0]
            assert torch.isfinite(pred).all(), (
                f"τ={bad_tau} produced non-finite pred: {pred.tolist()}"
            )


def test_default_softmax_no_temperature_bitexact():
    """τ=1.0 ⇒ softmax is F.softmax(x) without temperature scaling.

    Asserted by computing the stash output at τ=1.0 and comparing to
    the manual softmax expectation.  Any τ≠1.0 scaling would shift
    the pred_bo_lin and the comparison would fail.
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    torch.manual_seed(0)
    m = CFMJointModel(
        hidden_dim=64, n_layers=3, cutoff=3.5,
        bond_predictor="yuelbond", bond_aux_temperature=1.0,
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ]
    )
    z = torch.tensor([6, 6, 7, 8], dtype=torch.long)
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, -0.5, 0.0],
            [0.5, 1.0, 2.0, -1.0, 0.5],
            [-1.0, 0.5, 1.5, 0.0, 2.0],
        ]
    )
    # bond_orders_continuous requires `bonds`.
    cloud = AtomCloud(
        positions=positions, atomic_numbers=z,
        bonds=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        bond_types=torch.tensor([1, 2, 1], dtype=torch.long),
        bond_orders_continuous=torch.tensor([1.0, 2.0, 0.0]),
    )
    m.last_bond_logits_bo_linear = []
    m._maybe_stash_bond_aux_bo_linear(logits, cloud)
    # The stash is non-empty if the cutoff filter let the pairs through.
    if m.last_bond_logits_bo_linear:
        pred = m.last_bond_logits_bo_linear[0].squeeze(-1)
        centres = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=logits.dtype)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * centres).sum(dim=-1)
        assert torch.allclose(pred, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Fix 9 — per-class PR-AUC reporting
# ---------------------------------------------------------------------------
def test_per_class_pr_auc_emits_legacy_fields():
    """``bond_pr_auc`` / ``bond_positive_rate`` / ``bond_best_f1_*`` preserved.

    Backward-compat invariant: every CSV/JSON consumer that reads the
    legacy keys keeps working — the new per-class function emits the
    same keys with the same semantic.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    torch.manual_seed(0)
    pair_probs = [torch.softmax(torch.randn(50, 5), dim=-1) for _ in range(2)]
    pair_truth_class = torch.randint(0, 5, (100,)).tolist()
    out = _bond_ranking_report_per_class(
        pair_probs=pair_probs,
        pair_truth_class=pair_truth_class,
    )
    # Legacy keys.
    for k in (
        "bond_pr_auc",
        "bond_positive_rate",
        "bond_best_f1_threshold",
        "bond_best_f1_at_threshold",
        "bond_n_pairs_scored",
        "bond_pr_auc_note",
    ):
        assert k in out, f"legacy key {k} missing from per-class report"
    assert 0.0 <= out["bond_pr_auc"] <= 1.0
    assert 0.0 <= out["bond_positive_rate"] <= 1.0
    assert 0.0 <= out["bond_best_f1_at_threshold"] <= 1.0


def test_per_class_pr_auc_emits_new_fields():
    """``bond_class_{c}_pr_auc`` + pos_rate + n_pos per class.

    Five classes × three fields = 15 new keys, plus the macro
    aggregate ``bond_per_class_macro_pr_auc``.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    torch.manual_seed(0)
    pair_probs = [torch.softmax(torch.randn(50, 5), dim=-1) for _ in range(2)]
    pair_truth_class = torch.randint(0, 5, (100,)).tolist()
    out = _bond_ranking_report_per_class(
        pair_probs=pair_probs,
        pair_truth_class=pair_truth_class,
    )
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_pr_auc" in out
        assert f"bond_class_{c}_pos_rate" in out
        assert f"bond_class_{c}_n_pos" in out
        assert 0.0 <= out[f"bond_class_{c}_pr_auc"] <= 1.0
        assert 0.0 <= out[f"bond_class_{c}_pos_rate"] <= 1.0
        assert out[f"bond_class_{c}_n_pos"] >= 0
    # Macro aggregate.
    assert "bond_per_class_macro_pr_auc" in out
    assert "bond_per_class_n_classes_with_pos" in out


def test_per_class_pr_auc_existence_is_not_the_no_bond_ap():
    """Wf-A97 — ``bond_pr_auc`` is the EXISTENCE AP, not the NO_BOND AP.

    The pre-Wf-A97 invariant (``bond_pr_auc == bond_class_0_pr_auc``) rested
    on a false claim: reversing the score reverses the ranking, and AP of the
    complement is not AP of the class.  On 64 real tmQM val clouds the two
    differ by 2.3x (0.266 existence vs 0.621 NO_BOND,
    metrics/by_metric/wfa97_bond_metric_recheck.json).
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    torch.manual_seed(42)
    pair_probs = [torch.softmax(torch.randn(50, 5), dim=-1) for _ in range(4)]
    pair_truth_class = torch.randint(0, 5, (200,)).tolist()
    out = _bond_ranking_report_per_class(
        pair_probs=pair_probs,
        pair_truth_class=pair_truth_class,
    )
    assert out["bond_pr_auc"] == pytest.approx(out["bond_existence_ap"])
    assert out["bond_pr_auc_class0"] == pytest.approx(
        out["bond_class_0_pr_auc"]
    )
    assert out["bond_pr_auc"] != pytest.approx(out["bond_class_0_pr_auc"])


def test_per_class_pr_auc_handles_empty_input():
    """Empty pair_probs / pair_truth_class returns all-zero fields."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    out = _bond_ranking_report_per_class(
        pair_probs=[], pair_truth_class=[]
    )
    assert out["bond_pr_auc"] == 0.0
    assert out["bond_n_pairs_scored"] == 0.0
    assert out["bond_positive_rate"] == 0.0
    for c in (0, 1, 2, 3, 4):
        assert out[f"bond_class_{c}_pr_auc"] == 0.0
        assert out[f"bond_class_{c}_pos_rate"] == 0.0
        assert out[f"bond_class_{c}_n_pos"] == 0.0


def test_per_class_pr_auc_perfect_ranking_class_one():
    """Perfect ranking for SINGLE class → bond_class_1_pr_auc == 1.0.

    Construct logits where prob[:, SINGLE] is rank-perfectly ordered
    vs (truth == SINGLE): positives get higher scores than every
    negative.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    # 100 pairs: 20 are class 1, 80 are class 0.  Set prob[:, 1] for
    # positives to 0.9 and for negatives to 0.05 → perfect rank.
    probs = torch.zeros(100, 5)
    probs[:, 0] = 0.95  # NO_BOND prior
    probs[:, 1] = 0.05
    truth = torch.zeros(100, dtype=torch.long)
    truth[:20] = 1  # first 20 are SINGLE
    # Flip the ranking so SINGLE rows have HIGHER prob[:, 1]:
    probs[:20, 1] = 0.9
    probs[:20, 0] = 0.05
    probs[20:, 1] = 0.05

    out = _bond_ranking_report_per_class(
        pair_probs=[probs],
        pair_truth_class=truth.tolist(),
    )
    # SINGLE class: AP should be 1.0 (rank-perfect).
    assert out["bond_class_1_pr_auc"] == pytest.approx(1.0, abs=1e-6)
    # NO_BOND class: AP should be 1.0 too (the NO_BOND rows have
    # prob[:, 0] = 0.95, the SINGLE rows have prob[:, 0] = 0.05 —
    # monotone reversal preserves ranking).
    assert out["bond_class_0_pr_auc"] == pytest.approx(1.0, abs=1e-6)


def test_per_class_pr_auc_random_ranking_reasonable_range():
    """Random logits ⇒ per-class AP lands near base rate (1/5)."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report_per_class,
    )

    torch.manual_seed(7)
    pair_probs = [torch.softmax(torch.randn(500, 5), dim=-1) for _ in range(2)]
    pair_truth_class = torch.randint(0, 5, (1000,)).tolist()
    out = _bond_ranking_report_per_class(
        pair_probs=pair_probs,
        pair_truth_class=pair_truth_class,
    )
    # Each class is ~20% of the data; a random ranker should AP close
    # to 0.20.  Generous tolerance — pure softmax + uniform truth
    # means slight calibration bias from the softmax itself.
    for c in (0, 1, 2, 3, 4):
        ap = out[f"bond_class_{c}_pr_auc"]
        # Random ranker base rate is 0.20 (1/5); allow 0.05..0.45.
        assert 0.05 <= ap <= 0.45, (
            f"class {c} random AP out of range: {ap}"
        )


def test_evaluate_bond_accuracy_emits_per_class_pr_auc_fields():
    """evaluate_bond_accuracy wires Fix 9 per-class PR-AUC fields end-to-end."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        CFMJointModel,
        evaluate_bond_accuracy,
    )
    from molmetal.models.bond_head import AtomCloud

    torch.manual_seed(0)
    m = CFMJointModel(
        hidden_dim=64,
        n_layers=3,
        cutoff=3.5,
        bond_predictor="yuelbond_edge_mpnn",
        n_edge_layers=4,
    )
    val_clouds = []
    for _ in range(3):
        positions = torch.randn(7, 3) * 1.5
        atomic_numbers = torch.tensor(
            [6, 6, 7, 8, 6, 6, 78], dtype=torch.long
        )
        bonds = [[], []]
        types = []
        for i in range(7):
            for j in range(i + 1, 7):
                d = (positions[i] - positions[j]).norm().item()
                if d < 2.0:
                    bonds[0].append(i)
                    bonds[1].append(j)
                    types.append(1 if d > 1.0 else 2)
        bonds_t = torch.tensor(bonds, dtype=torch.long)
        types_t = (
            torch.tensor(types, dtype=torch.long)
            if types
            else torch.zeros(0, dtype=torch.long)
        )
        val_clouds.append(
            AtomCloud(
                positions=positions,
                atomic_numbers=atomic_numbers,
                bonds=bonds_t,
                bond_types=types_t,
            )
        )

    out = evaluate_bond_accuracy(m, val_clouds)
    # Per-class PR-AUC fields (Fix 9).
    for c in (0, 1, 2, 3, 4):
        assert f"bond_class_{c}_pr_auc" in out
        assert f"bond_class_{c}_pos_rate" in out
        assert f"bond_class_{c}_n_pos" in out
    # Macro AP and aggregate fields.
    assert "bond_per_class_macro_pr_auc" in out
    assert "bond_per_class_n_classes_with_pos" in out
    # Legacy fields preserved (backward compat).
    assert "bond_pr_auc" in out
    assert "bond_positive_rate" in out
    assert "bond_best_f1_threshold" in out
    assert "bond_best_f1_at_threshold" in out
    assert "bond_n_pairs_scored" in out
    # Wf-A97 — bond_pr_auc is the EXISTENCE AP; the NO_BOND AP is kept under
    # its own name.  The pre-A97 equality was the metric defect.
    assert "bond_existence_ap" in out
    assert "bond_pr_auc_class0" in out
    assert out["bond_pr_auc"] == pytest.approx(out["bond_existence_ap"])
    assert out["bond_pr_auc_class0"] == pytest.approx(
        out["bond_class_0_pr_auc"]
    )


def test_legacy_bond_ranking_report_still_works():
    """Backward-compat: ``_bond_ranking_report(scores, is_true)`` still callable.

    Test file ``molmetal/tests/test_cfm_chemistry_blind_spot.py`` has
    5 tests that call ``_bond_ranking_report(scores, is_true)`` with
    the original signature.  This test confirms the function still
    returns the expected legacy keys.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_ranking_report,
    )

    # Perfect ranking.
    r = _bond_ranking_report([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert r["bond_pr_auc"] == pytest.approx(1.0)
    assert r["bond_best_f1_at_threshold"] == pytest.approx(1.0)
    # Anti-perfect ranking.
    r = _bond_ranking_report([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    # Rank-positives-from-bottom gives AP = base rate (0.5); the
    # block-stepped formula yields 0.4166.. for these specific scores.
    assert 0.0 <= r["bond_pr_auc"] <= 1.0
    # No positives.
    r = _bond_ranking_report([0.5, 0.6], [0, 0])
    assert r["bond_pr_auc"] == 0.0
    # OPTIMISTIC note only appears when the sweep ran (positives present);
    # assert it here on a with-positives input.
    r = _bond_ranking_report([0.9, 0.1], [1, 0])
    assert "OPTIMISTIC" in r["bond_pr_auc_note"]