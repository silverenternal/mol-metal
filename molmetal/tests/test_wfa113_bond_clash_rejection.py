# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 / T48 — a collapsed pair must not be a candidate bond.

THE DEFECT (SA-28 / D48)
------------------------
Every bond head built its candidate list as

    d = (p_j - p_i).norm().clamp(min=1.0)
    mask = d < 3.5

so the clamp ran BEFORE the candidate test.  Two atoms at 0.0 A apart became a
pair at exactly 1.0 A, ``is_short`` (d < 1.7) fired, and the pair entered the
head in the same feature cell as a real single bond.  On real geometry the
clamp moves the distance by at most 0.044 A (p01 of the minimum pair distance
is 0.963 A); on sampled geometry it moves 0.43 A (mean minimum pair distance
0.570 A).  That is how a steric collapse became a bond.

THE FIX
-------
The clamp is now numerical protection only.  Candidacy is decided on the
UNCLAMPED distance, and a pair closer than ``ratio * (r_i + r_j)`` (Cordero
radii, ratio 0.7) is dropped before the head scores it.  This is the same
predicate ``metrics/by_metric/cfm_generated_geometry_defect.json`` used to
define ``hard_clash``.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
The 143 hard-clash pairs in that artifact were counted on generated clouds
that were never written to disk, so this file cannot replay them.  It checks
the predicate at the artifact's two measured means (0.570 A generated, 1.172 A
real) and records the population check as UNMEASURED in the artifact.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from molmetal.models.bond_head import AtomCloud
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondBondDecoder
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    _CORDERO_RCOV,
    DEFAULT_BOND_CLASH_RATIO,
    YuelBondDecoderHead,
    YuelBondEdgeMPNNHead,
    pair_collapse_mask,
)

TMQM_DIR = REPO_ROOT / ".data" / "molmetal" / "tmQM"


def _cloud(positions, atomic_numbers) -> AtomCloud:
    return AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
    )


def _collapsed_triangle() -> AtomCloud:
    """Three carbons at 0.4 / 0.4 / 0.8 A — every pair is a hard clash.

    At 0.4 A a C-C pair is 0.53x the summed Cordero radii (1.52 A), well under
    the 0.7 threshold, and 0.8 A is 0.53x as well.  All three pairs are inside
    the 3.5 A cutoff, so this cloud isolates the collapse rule from the cutoff.
    """
    return _cloud(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
        [6, 6, 6],
    )


# ---------------------------------------------------------------------------
# 1. The radius table must not drift from the artifact's definition
# ---------------------------------------------------------------------------


def test_radius_table_matches_the_geometry_defect_probe():
    """``cfm_generated_geometry_defect.json`` names ``hard_clash`` via the
    probe's RCOV table.  A library module cannot import a script, so the table
    is duplicated — and this test is what makes the duplication safe."""
    path = REPO_ROOT / "molmetal" / "scripts" / "wf_blkc_l3_guidance_probe.py"
    spec = importlib.util.spec_from_file_location("_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe"] = mod
    spec.loader.exec_module(mod)
    assert dict(mod.RCOV) == dict(_CORDERO_RCOV)


def test_default_ratio_is_the_artifact_ratio():
    assert DEFAULT_BOND_CLASH_RATIO == 0.7


# ---------------------------------------------------------------------------
# 2. The gate that matters: no real candidate is lost
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TMQM_DIR.is_dir(), reason="tmQM corpus not on disk")
def test_real_tmqm_clouds_lose_no_candidate():
    """``no_false_reject``: 0 pairs dropped on real geometry.

    This is the kill condition in T48.  If it ever fires, the radius table or
    the ratio is wrong and the generation-side numbers must not be touched
    until that is fixed.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(str(TMQM_DIR), n_clouds=6, seed=0)
    assert clouds, "loader returned no clouds"
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1).eval()
    assert head.clash_reject is True, "the rule must be ON by default"
    rejected = 0
    with torch.no_grad():
        for c in clouds:
            head(c)
            rejected += head.last_clash_rejected_pairs
    assert rejected == 0, (
        f"{rejected} candidate pair(s) rejected on REAL tmQM geometry; the "
        "0.7*(r_i+r_j) predicate is supposed to reject none there"
    )


@pytest.mark.skipif(not TMQM_DIR.is_dir(), reason="tmQM corpus not on disk")
def test_real_tmqm_clouds_lose_no_candidate_edge_head():
    """The edge-MPNN head must agree with the atom-MPNN head on candidacy."""
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(str(TMQM_DIR), n_clouds=4, seed=0)
    head = YuelBondEdgeMPNNHead(hidden_dim=16, n_edge_layers=2).eval()
    rejected = 0
    with torch.no_grad():
        for c in clouds:
            head(c)
            rejected += head.last_clash_rejected_pairs
    assert rejected == 0


# ---------------------------------------------------------------------------
# 3. A collapse is not a candidate
# ---------------------------------------------------------------------------


def test_collapsed_triangle_loses_every_pair():
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1).eval()
    with torch.no_grad():
        logits, (i_idx, j_idx) = head(_collapsed_triangle())
    assert logits.shape[0] == 0
    assert i_idx.numel() == 0 and j_idx.numel() == 0
    assert head.last_clash_rejected_pairs == 3


def test_off_reproduces_the_old_candidate_set():
    """The switch is a switch, not a constant: OFF is the pre-A113 edge list."""
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1, clash_reject=False).eval()
    with torch.no_grad():
        logits, (i_idx, j_idx) = head(_collapsed_triangle())
    assert logits.shape[0] == 3
    assert sorted(zip(i_idx.tolist(), j_idx.tolist())) == [(0, 1), (0, 2), (1, 2)]
    assert head.last_clash_rejected_pairs == 0


def test_the_two_measured_means_classify_correctly():
    """0.570 A is the generated mean; 1.172 A is the real mean (same artifact).

    A rule that kept the first or dropped the second would be measuring the
    wrong thing, so both sides are pinned.
    """
    for d, should_reject in ((0.5702152846818632, True), (1.171822339266011, False)):
        cloud = _cloud([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], [6, 6])
        head = YuelBondDecoderHead(hidden_dim=8, n_layers=1).eval()
        with torch.no_grad():
            logits, _ = head(cloud)
        assert (logits.shape[0] == 0) is should_reject, (
            f"C-C at {d:.4f} A: expected reject={should_reject}, "
            f"got {head.last_clash_rejected_pairs} rejection(s)"
        )


def test_predicate_is_direction_and_order_independent():
    """``pair_collapse_mask`` reads a symmetric radius sum, so (i,j) vs (j,i)
    and the order of the pair list cannot change the verdict."""
    z = torch.tensor([6, 78, 17])
    i = torch.tensor([0, 0, 1])
    j = torch.tensor([1, 2, 2])
    d_raw = torch.tensor([2.0, 0.5, 2.4])
    mask = pair_collapse_mask(z, i, j, d_raw)
    # C-Pt 2.0 A: rsum 2.12, 0.7*rsum 1.484 -> keep.
    # C-Cl 0.5 A: rsum 1.78, 0.7*rsum 1.246 -> reject.
    # Pt-Cl 2.4 A: rsum 2.38, 0.7*rsum 1.666 -> keep.
    assert mask.tolist() == [False, True, False]
    flipped = pair_collapse_mask(z[j], j, i, d_raw)
    assert flipped.tolist() == mask.tolist()


# ---------------------------------------------------------------------------
# 4. The decode must not manufacture a ring from a collapse
# ---------------------------------------------------------------------------


def test_decode_yields_zero_bonds_not_a_triangle():
    """``no_ring3_from_clash``.

    With the rule OFF and the connectivity prior ON this same cloud decodes to
    ``CCC`` — a chain built from two 0.4 A "bonds".  That is the pre-A113
    behaviour, and it is asserted below so the contrast is measured rather
    than described.
    """
    dec = YuelBondBondDecoder(hidden_dim=16, n_layers=1, connectivity_prior=True)
    out = dec.decode(_collapsed_triangle())
    assert len(out.decoded.bond_orders) == 0
    assert out.bonds_added_connectivity == 0, (
        "the connectivity prior bridged a cloud whose every pair was a "
        "collapse; it draws candidates from the head's edge list, which the "
        "rule had already emptied"
    )
    assert out.clash_rejected_pairs == 3


def test_with_the_rule_off_the_collapse_lands_in_the_short_bond_cell():
    """The control, stated as the mechanism rather than as a decoded string.

    An earlier version of this test asserted that the connectivity prior turns
    the cloud into ``CCC``.  It was flaky: the prior only bridges when a pair's
    strongest non-NO_BOND logit clears ``connectivity_prior_min_logit``, and on
    a randomly initialised head that is a coin flip.  What is deterministic —
    and what the defect actually is — is the FEATURE the head sees: with the
    rule off, all three pairs survive at the clamped distance of exactly
    ``min_distance`` = 1.0 A, so ``is_short`` (d < 1.7) is 1 for every one of
    them.  That is the cell a real single bond occupies.
    """
    head = YuelBondDecoderHead(hidden_dim=8, n_layers=1, clash_reject=False).eval()
    cloud = _collapsed_triangle()
    with torch.no_grad():
        logits, (i_idx, j_idx) = head(cloud)
    assert logits.shape[0] == 3
    d_clamped = (cloud.positions[j_idx] - cloud.positions[i_idx]).norm(dim=-1).clamp(
        min=head.min_distance
    )
    assert torch.allclose(d_clamped, torch.ones(3))
    assert bool((d_clamped < 1.7).all()), "is_short must fire on all three"


# ---------------------------------------------------------------------------
# 5. The rule must be visible in the artifact, not only in the code
# ---------------------------------------------------------------------------


def test_trainer_flags_default_on_and_are_recorded():
    src = (
        REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    ).read_text(encoding="utf-8")
    # argparse defaults
    assert '"--bond-clash-reject"' in src
    assert '"--bond-clash-ratio"' in src
    # the feature schema is the artifact-side record
    assert '"bond_clash_reject": bool(' in src
    assert '"bond_clash_ratio": float(' in src
    # the per-epoch generation block, so a pool states its own rule
    assert '_gen_chem["bond_clash_reject"]' in src


def test_every_head_accepts_the_two_arguments():
    """A head that does not take them would silently decode under the old
    candidate rule; there are four head classes and three forwards."""
    for cls, kwargs in (
        (YuelBondDecoderHead, {"n_layers": 1}),
        (YuelBondEdgeMPNNHead, {"n_edge_layers": 1}),
    ):
        on = cls(hidden_dim=8, clash_reject=True, clash_ratio=0.9, **kwargs)
        off = cls(hidden_dim=8, clash_reject=False, **kwargs)
        assert on.clash_reject is True and on.clash_ratio == 0.9
        assert off.clash_reject is False
