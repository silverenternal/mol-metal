# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 — the radii-scaled bond candidacy (A3) and the valence-deficit
promotion (A4).

TWO OPT-IN DECODER RULES, ONE LOAD-BEARING INVARIANT
------------------------------------------------------
Both rules are **default-off**, and the invariant this file exists to pin is
that **the default decode is bit-identical to the pre-Wf-A155 decode**.

**A3 — the radii-scaled candidacy.**  The candidate-edge rule in
``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`` is a **FLAT 3.5 Å**
cutoff (``mask = d_raw < self.cutoff``) — a declared deviation from the
published YuelBond 3 Å (``docs/HONEST_NEGATIVE_CATALOG.md`` §6.257).
EF-TALFM (**arXiv 2609.08333 App. C.2.1**, Algorithm 3) instead uses a
PAIR-DEPENDENT, radii-scaled threshold::

    tau_ij  <-  lambda_pi * (r_cov(z_i) + r_cov(z_j))

A3 adds that rule as an **opt-in alternative** (``radii_scaled_candidacy=True``).
**The 3.5 Å flat cutoff stays the default.**

**A4 — the valence-deficit promotion.**  The decoder's forced-completion
stage (``_force_connectivity``) LOWERS a bond ORDER to free valence budget
so a bridge fits; it never deletes an edge.  A4 adds the mirror rule: when
an atom is **under-valent** after the current bond assignment, PROMOTE an
eligible candidate bond to raise it.  It is also default-off
(``valence_deficit_promotion=False``).

Both stages charge their valence through the decoder's OWN model
(``_endpoint_valence_cost``), so **a metal bond does not cost the ligand any
valence** (Wf-A113 T79) — a promotion that would raise a ligand's valence
through a metal bond cannot do so.

WHAT IS *NOT* COMPARABLE
------------------------
EF-TALFM reports **Connected 92.98 %** over **10,000 molecules** (App. A.2),
and the frontier band is 92.7-94 %.  Our figure is **8.51 %**, and
``docs/HONEST_NEGATIVE_CATALOG.md`` §6.264 §6 measures FOUR independent
breaks — different OBJECT (ours is a whole metal-ligand complex as one
connected fragment; theirs is an organic molecule's validity or a per-BOND
recovery F1), different DENOMINATOR (ours is over ALL samples drawn,
unparseable against; theirs is over generated molecules only), different
CHEMISTRY (ours carries a metal in 100 % of pool rows; every paper cohort is
all-organic), and different BOND RULE + RAW GEOMETRY (ours is the flat 3.5 Å
cutoff, theirs is the element-aware ``tau_ij``, and ours is RAW).  Turning
A3 on therefore changes WHICH pairs the decoder may bond; it does **not**
make our number comparable to theirs.

CPU-only.  No GPU, no checkpoint, no corpus, no network, no decode of a
published pool.  Every case here is a synthetic cloud of at most eight
atoms.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from molmetal.models.bond_head import BOND_SINGLE, AtomCloud
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
    YuelBondDecodeResult,
    _replay_remaining_budget,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    DEFAULT_RADII_SCALED_LAMBDA,
    YuelBondDecoderHead,
    YuelBondEdgeMPNNHead,
    YuelBondEdgeMPNNHeadResidual,
    YuelBondResult,
    compare_bond_candidacy,
    pair_collapse_mask,
    radii_scaled_candidacy_mask,
    radii_scaled_factor,
    radii_scaled_threshold,
    rcov_for_z,
)

#: The repository root — the parent of the ``molmetal`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#: The Cordero (2008) single-bond covalent radii, written out LITERALLY here
#: so the threshold test is a comparison against HAND-COMPUTED numbers and
#: not against the module's own table (which would be circular).
#:
#: MEASURED on 2026-11-01 from ``_CORDERO_RCOV`` in
#: ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py``.
HAND_RCOV: Dict[int, float] = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    78: 1.36,  # Pt
}


def _head(**kwargs) -> YuelBondDecoderHead:
    """A deterministic small head: same seed -> same weights."""
    torch.manual_seed(1234)
    return YuelBondDecoderHead(hidden_dim=16, n_layers=2, **kwargs)


def _decoder(**kwargs) -> YuelBondBondDecoder:
    """A deterministic YuelBond bond decoder."""
    torch.manual_seed(1234)
    return YuelBondBondDecoder(head=_head(), **kwargs)


def _cloud(positions: List[List[float]], zs: List[int]) -> AtomCloud:
    return AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )


def _cloud_hh() -> AtomCloud:
    """Two hydrogens 2.5 Å apart.

    FLAT: 2.5 < 3.5 -> SELECTED.  SCALED (lambda = 1):
    ``pi * (0.31 + 0.31) = 1.9478`` -> 2.5 > 1.9478 -> REJECTED.
    This is the direction in which the flat rule is MORE PERMISSIVE.
    """
    return _cloud([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], [1, 1])


def _cloud_cc() -> AtomCloud:
    """Two carbons 4.0 Å apart.

    FLAT: 4.0 >= 3.5 -> REJECTED.  SCALED (lambda_pi = 1):
    ``1.0 * (0.76 + 0.76) = 1.52`` -> 4.0 > 1.52 -> ALSO REJECTED.

    At ``lambda_pi = 1`` the radii-scaled rule is STRICTLY STRICTER than the
    flat cutoff for every pair the decoder can emit (the largest pair-sum
    in the on-disk Cordero table is 2.78 Å), so this pair is the direction in
    which the flat rule is MORE PERMISSIVE.  It becomes the other direction
    only for ``lambda_pi > 3.5 / 1.52 = 2.3026``.
    """
    return _cloud([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [6, 6])


def _cloud_ccc() -> AtomCloud:
    """Three carbons in a row, 1.2 Å apart.  Atom 0 is the stranded one."""
    return _cloud(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]],
        [6, 6, 6],
    )


def _cloud_pt_nn() -> AtomCloud:
    """Pt at the origin with two nitrogens at 2.05 Å, one on each side."""
    return _cloud(
        [
            [0.0, 0.0, 0.0],
            [2.05, 0.0, 0.0],
            [-2.05, 0.0, 0.0],
        ],
        [78, 7, 7],
    )


def _synthetic_result(
    edge_index: List[Tuple[int, int]],
    logits: List[List[float]],
) -> YuelBondResult:
    """A head result with hand-chosen logits.

    ``logits[k]`` is the 5-class logit row for the pair
    ``edge_index[k]``: class 0 is NO_BOND, class 1 SINGLE, 2 DOUBLE,
    3 TRIPLE, 4 AROMATIC.
    """
    return YuelBondResult(
        pair_features=None,
        logits=torch.tensor(logits, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
        decode_succeeded=True,
    )


def _old_candidate_pairs(
    cloud: AtomCloud,
    head: YuelBondDecoderHead,
) -> Tuple[List[int], List[int]]:
    """The PRE-Wf-A155 candidate selection, recomputed INDEPENDENTLY.

    This is the reference the default path is measured against.  The
    expression ``d_raw < self.cutoff`` is copied verbatim from the
    pre-Wf-A155 source; the clash veto is applied with the module's own
    public :func:`pair_collapse_mask`, because that veto is NOT what
    Wf-A155 changed.
    """
    n = int(cloud.atomic_numbers.shape[0])
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    d_raw = (cloud.positions[j_idx] - cloud.positions[i_idx]).norm(dim=-1)
    mask = d_raw < head.cutoff
    if head.clash_reject:
        collapsed = pair_collapse_mask(
            cloud.atomic_numbers, i_idx, j_idx, d_raw, head.clash_ratio
        )
        mask = mask & ~collapsed
    keep = mask.nonzero(as_tuple=True)[0]
    return i_idx[keep].tolist(), j_idx[keep].tolist()


def _assigned_valences(
    dec: YuelBondBondDecoder,
    cloud: AtomCloud,
    bond_orders: List[Tuple[int, int, int]],
) -> Tuple[List[float], List[int]]:
    """``(assigned valence, bond count)`` per atom, in the decoder's model.

    The assigned valence is the CAP minus the remaining budget, replayed
    through the decoder's own :func:`_replay_remaining_budget` — the same
    accounting the selector uses.
    """
    z_list = [int(z) for z in cloud.atomic_numbers.tolist()]
    caps = [dec._cap_for_z(z) for z in z_list]
    remaining, _arom = _replay_remaining_budget(
        bond_orders, z_list, caps, dec._is_metal_z
    )
    n_bonds = [0] * len(z_list)
    for i, j, _order in bond_orders:
        n_bonds[i] += 1
        n_bonds[j] += 1
    return [caps[a] - remaining[a] for a in range(len(z_list))], n_bonds


# ===========================================================================
# 1. A3 — the radii-scaled threshold IS the EF-TALFM formula
# ===========================================================================


def test_the_factor_is_lambda_pi():
    """``radii_scaled_factor`` is exactly ``lambda_pi`` — the IDENTITY.

    MEASURED: the identity holds for the default and for a non-default
    ``lambda``.  The factor used to be ``lambda * math.pi``, which was a
    MISREADING of the paper's subscript — the ``pi`` in EF-TALFM is a
    subscript denoting the reconstruction policy, not a multiplicative
    constant.  Independently re-derived from the paper's own source in
    ``molmetal/reports/wfa160_bond_threshold_independent_verification/``
    (verdict ``THE-PI-IS-A-SUBSCRIPT``, e-print sha256
    ``e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224``).
    """
    assert radii_scaled_factor() == 1.0
    assert radii_scaled_factor() == DEFAULT_RADII_SCALED_LAMBDA
    assert radii_scaled_factor(2.0) == 2.0
    assert radii_scaled_factor(0.5) == 0.5


def test_the_paper_lambda_is_exposed_as_a_parameter_not_hardcoded():
    """The paper's own ``lambda_pi`` is NOT recoverable from arXiv
    2609.08333, Appendix C.2.1, Algorithm 3 (the formula is stated without
    a number and ``\\lambda_\\pi`` occurs exactly once in the whole
    source), so it is a PARAMETER whose default is the neutral identity
    reading — and it is never presented as the paper's value."""
    assert DEFAULT_RADII_SCALED_LAMBDA == 1.0
    # The default factor is therefore exactly 1.0, i.e. lambda_pi = 1.
    assert radii_scaled_factor(DEFAULT_RADII_SCALED_LAMBDA) == 1.0
    # The parameter is honoured end-to-end.
    assert radii_scaled_factor(3.0) == 3.0


@pytest.mark.parametrize(
    "z_i,z_j",
    [
        (6, 6),
        (1, 1),
        (6, 7),
        (7, 8),
        (16, 17),
        (35, 6),
        (78, 7),
        (78, 78),
    ],
)
def test_the_threshold_equals_the_hand_computed_ef_talfm_value(z_i, z_j):
    """``tau_ij = lambda_pi * (r_cov(z_i) + r_cov(z_j))``.

    The expected value is computed from the HAND-WRITTEN radii above, so
    this is a comparison against hand-computed numbers and not against the
    module's own table.
    """
    expected = DEFAULT_RADII_SCALED_LAMBDA * (
        HAND_RCOV[z_i] + HAND_RCOV[z_j]
    )
    got = radii_scaled_threshold(z_i, z_j)
    assert got == pytest.approx(expected, rel=1e-12), (z_i, z_j, got, expected)
    # The module's own lookup agrees with the hand-written table.
    assert rcov_for_z(z_i) == pytest.approx(HAND_RCOV[z_i])
    assert rcov_for_z(z_j) == pytest.approx(HAND_RCOV[z_j])


def test_the_threshold_honours_a_non_default_lambda():
    """The factor scales the whole threshold linearly."""
    for lam in (0.5, 1.0, 2.0, 3.7):
        factor = radii_scaled_factor(lam)
        expected = factor * (HAND_RCOV[6] + HAND_RCOV[7])
        got = radii_scaled_threshold(6, 7, factor=factor)
        assert got == pytest.approx(expected, rel=1e-12), lam


def test_the_mask_selects_exactly_the_pairs_below_tau():
    """``radii_scaled_candidacy_mask`` is ``d_raw < tau_ij``, per pair.

    MEASURED on synthetic (distance, z_i, z_j) triples.
    """
    z = torch.tensor([6, 6, 1, 1, 78, 7], dtype=torch.long)
    i_idx = torch.tensor([0, 0, 2, 0, 4], dtype=torch.long)
    j_idx = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    d_raw = torch.tensor([4.0, 3.4, 2.5, 2.05, 3.0], dtype=torch.float32)
    mask = radii_scaled_candidacy_mask(z, i_idx, j_idx, d_raw)
    factor = radii_scaled_factor()
    for k in range(len(i_idx)):
        tau = factor * (
            HAND_RCOV[int(z[int(i_idx[k])])] + HAND_RCOV[int(z[int(j_idx[k])])]
        )
        expected = bool(float(d_raw[k]) < tau)
        assert bool(mask[k]) is expected, (k, float(d_raw[k]), tau)


def test_the_mask_uses_the_same_radii_table_as_the_steric_veto():
    """ONE radii table, not two.

    The steric-collapse veto (``pair_collapse_mask``) and the candidacy
    rule must never drift onto different radii, so both must read
    ``_CORDERO_RCOV``.  This is checked by construction: the veto's
    predicate is ``d < ratio * (r_i + r_j)`` and the candidacy rule's is
    ``d < factor * (r_i + r_j)`` — the SAME radius sum.
    """
    z = torch.tensor([6, 17], dtype=torch.long)
    i_idx = torch.tensor([0], dtype=torch.long)
    j_idx = torch.tensor([1], dtype=torch.long)
    d_raw = torch.tensor([2.5], dtype=torch.float32)
    veto = pair_collapse_mask(z, i_idx, j_idx, d_raw, ratio=0.7)
    cand = radii_scaled_candidacy_mask(z, i_idx, j_idx, d_raw)
    radius_sum = HAND_RCOV[6] + HAND_RCOV[17]
    assert bool(veto[0]) is (2.5 < 0.7 * radius_sum)
    assert bool(cand[0]) is (2.5 < radius_sum)


def test_the_scalar_and_tensor_radius_paths_agree_everywhere():
    """``rcov_for_z`` (scalar) and ``_rcov_per_atom`` (tensor) must return the
    SAME radius for every element, including the clamp boundaries.

    The candidacy mask applies the TENSOR path while the read-only
    comparison utility reports the SCALAR threshold; if the two could
    disagree, the utility would quote a threshold the mask never applied.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import _rcov_per_atom

    probe = [-5, -1, 0, 1, 6, 8, 35, 78, 91, 92, 93, 118, 200]
    tensor = _rcov_per_atom(torch.tensor(probe, dtype=torch.long))
    for k, z in enumerate(probe):
        assert rcov_for_z(z) == pytest.approx(float(tensor[k])), (z, z)


# ===========================================================================
# 2. A3 — the flat rule and the radii-scaled rule DISAGREE, both ways
# ===========================================================================


def test_the_flat_rule_selects_a_pair_the_radii_scaled_rule_rejects():
    """H–H at 2.5 Å: the flat 3.5 Å cutoff selects it, the radii-scaled
    rule (``tau = 1.0 * 0.62 = 0.62``) rejects it."""
    rows = compare_bond_candidacy([2.5], [1], [1])
    assert len(rows) == 1
    row = rows[0]
    assert row["flat_threshold"] == 3.5
    assert row["flat_selected"] is True
    assert row["scaled_threshold"] == pytest.approx(0.31 + 0.31, rel=1e-12)
    assert row["scaled_selected"] is False
    assert row["differs"] is True


def test_the_radii_scaled_rule_is_stricter_than_the_flat_cutoff_at_lambda_one():
    """At ``lambda_pi = 1`` the radii-scaled rule is STRICTLY STRICTER than
    the flat 3.5 Å cutoff for EVERY pair the decoder can emit.

    MEASURED: the largest pair-sum in the on-disk Cordero table is
    **2.78 Å** (I–I, z=53), so ``tau_ij <= 2.78 < 3.5`` for every pair and
    the scaled rule can never admit a pair the flat cutoff rejects.

    This is a DIRECT CONSEQUENCE of the Wf-A160 correction.  The previous
    ``lambda * math.pi`` factor gave ``tau_max = 3.1416 * 2.78 = 8.73`` Å,
    i.e. the "opt-in paper rule" was really "3.14x the flat cutoff" in
    disguise — a rule that would have admitted essentially every
    covalent-contact pair and therefore UNDERSTATED fragmentation in the
    very chain whose headline is "the CFM emits fragments".
    """
    rows = compare_bond_candidacy([4.0], [6], [6])
    row = rows[0]
    assert row["flat_selected"] is False
    assert row["scaled_threshold"] == pytest.approx(0.76 + 0.76, rel=1e-12)
    assert row["scaled_selected"] is False
    assert row["differs"] is False


def test_the_radii_scaled_rule_is_more_permissive_only_via_lambda():
    """The other direction of the disagreement requires ``lambda_pi`` big
    enough that ``lambda_pi * (r_i + r_j) > 3.5``.

    MEASURED: for the worst pair in the table (2.78 Å) that needs
    ``lambda_pi > 3.5 / 2.78 = 1.2590``; for C–C (1.52 Å) it needs
    ``lambda_pi > 2.3026``, and 3.0 is used here.
    """
    rows = compare_bond_candidacy([4.0], [6], [6], factor=3.0)
    row = rows[0]
    assert row["flat_selected"] is False
    assert row["scaled_threshold"] == pytest.approx(
        3.0 * (0.76 + 0.76), rel=1e-12
    )
    assert row["scaled_selected"] is True
    assert row["differs"] is True


def test_the_comparison_utility_agrees_with_the_head_on_both_clouds():
    """The READ-ONLY utility and the head's own candidate list must agree.

    This is what makes the utility a measurement instrument rather than a
    second, divergent implementation of the rule.
    """
    # H–H at 2.5 A: the head's default (flat) rule keeps the pair.
    head = _head()
    logits, (i_idx, j_idx) = head(_cloud_hh())
    assert i_idx.tolist() == [0] and j_idx.tolist() == [1]
    # ... and the utility says the scaled rule would drop it.
    assert compare_bond_candidacy([2.5], [1], [1])[0]["scaled_selected"] is False

    # C–C at 4.0 A: the head's default (flat) rule drops the pair.
    logits2, (i2, j2) = head(_cloud_cc())
    assert i2.numel() == 0 and j2.numel() == 0
    # ... and at lambda_pi = 1 the scaled rule drops it too, because the
    # radii-scaled threshold is at most 2.78 A -- strictly below the flat
    # 3.5 A cutoff for every pair (see the test above).
    row_cc = compare_bond_candidacy([4.0], [6], [6])[0]
    assert row_cc["scaled_selected"] is False
    assert row_cc["scaled_threshold"] == pytest.approx(0.76 + 0.76, rel=1e-12)


def test_the_comparison_utility_reports_every_field_and_mutates_nothing():
    """The utility is READ-ONLY: it returns plain dicts and touches no
    state."""
    rows = compare_bond_candidacy([2.5, 4.0], [1, 6], [1, 6])
    assert len(rows) == 2
    for k, row in enumerate(rows):
        assert isinstance(row, dict)
        assert set(row) == {
            "index",
            "distance",
            "z_i",
            "z_j",
            "rcov_i",
            "rcov_j",
            "flat_threshold",
            "flat_selected",
            "scaled_threshold",
            "scaled_selected",
            "differs",
        }, sorted(row)
        assert row["index"] == k
        assert isinstance(row["flat_selected"], bool)
        assert isinstance(row["scaled_selected"], bool)
        assert isinstance(row["differs"], bool)
        assert row["differs"] is (row["flat_selected"] != row["scaled_selected"])


# ===========================================================================
# 3. THE LOAD-BEARING REGRESSION — the DEFAULT path is bit-identical
# ===========================================================================


def test_the_default_flag_is_off_on_every_head_class():
    """Every head class that grew the A3 switch must default it to OFF.

    A head class that silently defaulted the flag ON would change the
    candidate set of every decode that uses it.
    """
    torch.manual_seed(1234)
    for head in (
        YuelBondDecoderHead(hidden_dim=8, n_layers=1),
        YuelBondEdgeMPNNHead(hidden_dim=8, n_edge_layers=1),
        YuelBondEdgeMPNNHeadResidual(hidden_dim=8, n_edge_layers=1),
    ):
        assert head.radii_scaled_candidacy is False, type(head).__name__
        assert head.radii_scaled_lambda == DEFAULT_RADII_SCALED_LAMBDA


def test_the_default_candidate_mask_is_exactly_the_pre_wfa155_expression():
    """THE LOAD-BEARING ASSERTION.

    The head's DEFAULT candidate list must be EXACTLY the pair set the
    pre-Wf-A155 source produced: ``d_raw < self.cutoff`` (the flat 3.5 Å
    rule) minus the steric-collapse veto.  The reference is recomputed
    here from the cloud alone, so a change to the default branch — a
    different comparison, a different constant, an off-by-one, or a
    silent switch to the radii rule — fails this test.
    """
    head = _head()
    clouds = [
        _cloud_hh(),
        _cloud_cc(),
        _cloud_ccc(),
        _cloud_pt_nn(),
        _cloud(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
            [6, 8, 6, 6],
        ),
    ]
    for cloud in clouds:
        _logits, (i_idx, j_idx) = head(cloud)
        exp_i, exp_j = _old_candidate_pairs(cloud, head)
        assert i_idx.tolist() == exp_i, (cloud.atomic_numbers.tolist(), i_idx.tolist(), exp_i)
        assert j_idx.tolist() == exp_j, (cloud.atomic_numbers.tolist(), j_idx.tolist(), exp_j)


def test_the_default_path_never_calls_the_new_rule(monkeypatch):
    """The default branch must not even EVALUATE the radii-scaled rule.

    A default path that called the new function and then ignored its
    result would still pass a mask-value test, and would still be
    bit-identical today — but it would not be if the function ever grew a
    side effect.  This pins the branch itself.
    """
    import molmetal.molmetal_lam.lam_chem.yuelbond_decoder as yb

    def _explode(*_args, **_kwargs):
        raise AssertionError(
            "the radii-scaled rule was evaluated on the DEFAULT path"
        )

    monkeypatch.setattr(yb, "radii_scaled_candidacy_mask", _explode)
    head = _head()
    _logits, (i_idx, j_idx) = head(_cloud_ccc())
    assert i_idx.numel() >= 1, "the cloud has no candidate pair at all"


def test_the_load_bearing_mask_assertion_would_fail_if_the_default_switched():
    """THE REGRESSION WITNESS for the mask assertion above.

    A test that compares the head's candidate list against a reference
    computed from the same source is only worth anything if the reference
    would DISAGREE when the implementation changes.  So: point the head
    at the radii-scaled rule and show that its candidate list no longer
    matches the flat-rule reference — i.e. the load-bearing assertion
    would fail if the DEFAULT branch were ever switched.
    """
    cloud = _cloud_hh()
    head_flat = _head()
    _l, (i_flat, j_flat) = head_flat(cloud)
    exp_i, exp_j = _old_candidate_pairs(cloud, head_flat)
    assert i_flat.tolist() == exp_i and j_flat.tolist() == exp_j

    head_scaled = _head(radii_scaled_candidacy=True)
    _l2, (i_scaled, _j_scaled) = head_scaled(cloud)
    assert i_scaled.numel() == 0, "the scaled rule kept the H-H pair"
    assert i_scaled.tolist() != exp_i, (
        "the witness is vacuous: the scaled rule produced the same "
        "candidate list as the flat rule"
    )


def test_the_opt_in_rule_changes_the_candidate_set_and_only_that():
    """With the flag ON, the candidate set follows ``tau_ij`` — and the
    DIFFERENCE is exactly the disagreement the comparison utility
    reports."""
    # H-H at 2.5 A: flat keeps it, scaled drops it.
    head_on = _head(radii_scaled_candidacy=True)
    _l_on, (i_on, j_on) = head_on(_cloud_hh())
    assert i_on.numel() == 0 and j_on.numel() == 0
    head_off = _head()
    _l_off, (i_off, j_off) = head_off(_cloud_hh())
    assert i_off.tolist() == [0] and j_off.tolist() == [1]

    # C-C at 4.0 A: at lambda_pi = 1 the scaled rule is STRICTLY STRICTER
    # than the flat 3.5 A cutoff (the largest pair-sum in the whole
    # Cordero table is 2.78 A), so BOTH rules drop the pair.
    head_on2 = _head(radii_scaled_candidacy=True)
    _l2_on, (i2_on, j2_on) = head_on2(_cloud_cc())
    assert i2_on.numel() == 0 and j2_on.numel() == 0
    head_off2 = _head()
    _l2_off, (i2_off, j2_off) = head_off2(_cloud_cc())
    assert i2_off.numel() == 0 and j2_off.numel() == 0

    # With lambda_pi = 3.0 the scaled rule KEEPS the C-C pair at 4.0 A
    # (tau = 3.0 * 1.52 = 4.56 A > 4.0 A) while the flat cutoff drops it --
    # and the comparison utility reports exactly that disagreement.  This
    # is the direction in which the flat rule is LESS PERMISSIVE, and it
    # is reachable only through lambda_pi (the paper's own value is not
    # recoverable -- see the module docstring).
    head_on3 = _head(radii_scaled_candidacy=True, radii_scaled_lambda=3.0)
    _l3_on, (i3_on, j3_on) = head_on3(_cloud_cc())
    assert i3_on.tolist() == [0] and j3_on.tolist() == [1]
    row_cc = compare_bond_candidacy([4.0], [6], [6], factor=3.0)[0]
    assert row_cc["flat_selected"] is False
    assert row_cc["scaled_selected"] is True
    assert row_cc["differs"] is True


def test_the_default_decode_is_unchanged_by_the_presence_of_the_new_code(
    monkeypatch,
):
    """A decode on the default path must produce the SAME molecule and the
    SAME counters whether or not the opt-in rule is reachable.

    The check replaces the new rule with a function that RAISES: a decode
    that still succeeds is a decode that never touched it.
    """
    import molmetal.molmetal_lam.lam_chem.yuelbond_decoder as yb

    def _explode(*_args, **_kwargs):
        raise AssertionError(
            "the radii-scaled rule was evaluated on the DEFAULT path"
        )

    for cloud in (_cloud_ccc(), _cloud_pt_nn()):
        base = _decoder().decode(cloud)
        monkeypatch.setattr(yb, "radii_scaled_candidacy_mask", _explode)
        dec = _decoder()
        res = dec.decode(cloud)
        monkeypatch.undo()
        assert res.decoded.smiles == base.decoded.smiles, (cloud.atomic_numbers.tolist(), res.decoded.smiles, base.decoded.smiles)
        assert res.bonds_accepted == base.bonds_accepted
        assert res.bonds_added_connectivity == base.bonds_added_connectivity
        assert res.connectivity_forced_bonds == base.connectivity_forced_bonds
        assert res.bonds_added_valence_deficit == 0


# ===========================================================================
# 4. A4 — the valence-deficit promotion
# ===========================================================================


def test_the_valence_deficit_stage_is_off_by_default():
    dec = _decoder()
    assert dec.valence_deficit_promotion is False
    assert dec.valence_deficit_min_logit == -2.0
    assert dec.valence_deficit_max_promotions_per_atom == 1


def test_the_default_decode_never_calls_the_valence_deficit_stage(monkeypatch):
    """The stage must be unreachable on the default path."""
    dec = _decoder()

    def _explode(*_args, **_kwargs):
        raise AssertionError(
            "the valence-deficit promotion ran on the DEFAULT path"
        )

    monkeypatch.setattr(dec, "_promote_valence_deficit_bonds", _explode)
    res = dec.decode(_cloud_ccc())
    assert res.bonds_added_valence_deficit == 0


def test_the_valence_deficit_promotion_raises_a_stranded_carbon_to_its_target():
    """THE A4 BEHAVIOURAL ASSERTION.

    Three carbons.  The head keeps the (1, 2) bond and declines the
    (0, 1) one, so atom 0 is STRANDED — attached to nothing, which under
    RDKit's implicit-hydrogen rule would make it a second fragment
    (methane).  With the stage ON, the promotion attaches atom 0 to its
    closest eligible candidate.

    Asserted on the DECODER'S OWN valence model: atom 0's assigned
    valence goes 0.0 -> 1.0 and its deficit goes 4.0 -> 0.0.
    """
    cloud = _cloud_ccc()
    edge_index = [(0, 1), (1, 2)]
    # (0, 1): NO_BOND wins, but the bond-class logits (-1.0) still clear
    # the -2.0 deference floor.  (1, 2): BOND_SINGLE wins.
    logits = [
        [5.0, -1.0, -1.0, -1.0, -1.0],
        [-1.0, 5.0, -1.0, -1.0, -1.0],
    ]
    yres = _synthetic_result(edge_index, logits)

    dec = _decoder()
    bond_orders: List[Tuple[int, int, int]] = []
    dative: List[Tuple[int, int]] = []
    selected, _d, _kept, _over = dec._select_bonds(cloud, yres)
    bond_orders.extend(selected)
    assert selected == [(1, 2, BOND_SINGLE)], selected

    before_assigned, before_nbonds = _assigned_valences(dec, cloud, bond_orders)
    assert before_assigned == [0.0, 1.0, 1.0], before_assigned
    assert before_nbonds == [0, 1, 1], before_nbonds
    deficit_before = dec._valence_deficit(6, before_assigned[0], dec._cap_for_z(6), 0)
    assert deficit_before == 4.0, deficit_before

    n_promoted = dec._promote_valence_deficit_bonds(
        cloud, yres, bond_orders, dative
    )
    assert n_promoted == 1, n_promoted
    assert (0, 1, BOND_SINGLE) in bond_orders, bond_orders

    after_assigned, after_nbonds = _assigned_valences(dec, cloud, bond_orders)
    assert after_assigned[0] == 1.0, after_assigned
    assert after_nbonds[0] == 1, after_nbonds
    deficit_after = dec._valence_deficit(
        6, after_assigned[0], dec._cap_for_z(6), after_nbonds[0]
    )
    assert deficit_after == 0.0, deficit_after


def test_the_valence_deficit_promotion_is_reported_on_the_result():
    """The count reaches the RESULT object, and the stage's work is
    visible in the decode's own accounting."""
    cloud = _cloud_ccc()
    yres = _synthetic_result(
        [(0, 1), (1, 2)],
        [
            [5.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0, -1.0, -1.0],
        ],
    )
    dec = _decoder()
    dec.valence_deficit_promotion = True
    dec._decoder.featurise = lambda _c, _y=yres: _y  # type: ignore[assignment]
    res = dec.decode(cloud)
    assert isinstance(res, YuelBondDecodeResult)
    assert res.bonds_added_valence_deficit == 1, res.bonds_added_valence_deficit
    assert res.bonds_accepted == 2, res.bonds_accepted
    # The molecule is one connected fragment and RDKit accepts it.
    assert res.decoded.sanitized is True
    assert res.decoded.smiles == "CCC", res.decoded.smiles


def test_the_promotion_never_deletes_an_edge_or_lowers_a_bond_order():
    """The stage only ADDS.  A bond the selector kept must survive it
    with the SAME order."""
    cloud = _cloud_ccc()
    yres = _synthetic_result(
        [(0, 1), (1, 2)],
        [
            [5.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0, -1.0, -1.0],
        ],
    )
    dec = _decoder()
    bond_orders, dative, _kept, _over = dec._select_bonds(cloud, yres)
    before = list(bond_orders)
    dec._promote_valence_deficit_bonds(cloud, yres, bond_orders, dative)
    for bond in before:
        assert bond in bond_orders, (bond, bond_orders)
    assert len(bond_orders) == len(before) + 1


def test_a_metal_bond_does_not_raise_the_ligand_valence():
    """THE Wf-A113 T79 MIRROR.

    A bond to a metal costs the ligand endpoint NOTHING (the electron pair
    is the ligand's lone pair), so a promotion that goes through a metal
    bond must raise the METAL's coordination count and leave the LIGAND's
    valence exactly where it was.
    """
    cloud = _cloud_pt_nn()
    # (0, 1) Pt-N1: the head declines it.  (0, 2) Pt-N2: the head keeps it.
    yres = _synthetic_result(
        [(0, 1), (0, 2)],
        [
            [5.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0, -1.0, -1.0],
        ],
    )
    dec = _decoder()
    bond_orders, dative, _kept, _over = dec._select_bonds(cloud, yres)
    assert bond_orders == [(0, 2, BOND_SINGLE)], bond_orders

    before_assigned, _before_nbonds = _assigned_valences(dec, cloud, bond_orders)
    assert before_assigned[1] == 0.0, before_assigned  # N1: the ligand
    pt_cap = dec._cap_for_z(78)
    assert before_assigned[0] == 1.0, before_assigned  # Pt: one bond

    n_promoted = dec._promote_valence_deficit_bonds(
        cloud, yres, bond_orders, dative
    )
    assert n_promoted == 1, n_promoted
    assert (0, 1, BOND_SINGLE) in bond_orders, bond_orders

    after_assigned, _after_nbonds = _assigned_valences(dec, cloud, bond_orders)
    # The METAL's coordination count rose by one ...
    assert after_assigned[0] == before_assigned[0] + 1.0, after_assigned
    assert dec._cap_for_z(78) == pt_cap
    # ... and the LIGAND's valence did NOT move at all: the metal bond
    # cost it nothing (Wf-A113 T79).
    assert after_assigned[1] == 0.0, after_assigned


def test_a_non_metal_promotion_raises_both_endpoints():
    """The contrast case: a CARBON-CARBON promotion raises BOTH endpoints,
    because neither endpoint is a metal."""
    cloud = _cloud_ccc()
    yres = _synthetic_result(
        [(0, 1), (1, 2)],
        [
            [5.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0, -1.0, -1.0],
        ],
    )
    dec = _decoder()
    bond_orders, dative, _kept, _over = dec._select_bonds(cloud, yres)
    before_assigned, _ = _assigned_valences(dec, cloud, bond_orders)
    dec._promote_valence_deficit_bonds(cloud, yres, bond_orders, dative)
    after_assigned, _ = _assigned_valences(dec, cloud, bond_orders)
    assert after_assigned[0] == before_assigned[0] + 1.0, (before_assigned, after_assigned)
    assert after_assigned[1] == before_assigned[1] + 1.0, (before_assigned, after_assigned)


# ===========================================================================
# 5. THE NO-CUDA CHECK
# ===========================================================================

_NO_CUDA_PROBE = r"""
import os

assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA is not masked"

import torch

assert not torch.cuda.is_initialized(), "CUDA was initialised at import"

from molmetal.models.bond_head import AtomCloud
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondBondDecoder
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoderHead,
    compare_bond_candidacy,
)

assert not torch.cuda.is_initialized(), "CUDA was initialised at import"

torch.manual_seed(0)
dec = YuelBondBondDecoder(head=YuelBondDecoderHead(hidden_dim=8, n_layers=1))
cloud = AtomCloud(
    positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32),
    atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
)
res = dec.decode(cloud)
assert not torch.cuda.is_initialized(), "CUDA was initialised by the decode"

compare_bond_candidacy([2.5], [1], [1])
assert not torch.cuda.is_initialized(), "CUDA was initialised by the utility"

print("OK")
"""


def test_importing_and_decoding_does_not_touch_cuda():
    """The two modules import and decode with CUDA masked out, and the
    process never initialises a CUDA context.

    Run in a SUBPROCESS so the check cannot be satisfied by the state of
    the test session's own interpreter.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_CUDA_PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"the no-CUDA probe failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout, proc.stdout


def test_this_session_never_initialised_cuda():
    """The same invariant, asserted in-process: importing and using the
    two modules must not initialise a CUDA context."""
    assert not torch.cuda.is_initialized(), (
        "a CUDA context was initialised during this test session"
    )
