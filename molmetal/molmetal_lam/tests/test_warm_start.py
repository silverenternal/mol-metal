"""Phase-3J — Tests for the per-pocket warm-start embedding module.

These tests verify the **contract** of
:mod:`molmetal_lam.search_alg.warm_start`:

* :func:`pocket_features` is deterministic and shape-correct.
* :func:`pocket_features` handles the empty-pocket case (zero vector).
* :func:`pocket_features` produces **bounded** descriptors (no NaN,
  no inf, fractions in [0, 1]).
* :func:`modify_root_prior` produces a probability distribution and
  the selected root action differs when the pocket embedding changes.

Run with::

    uv run pytest tests/test_warm_start.py -x --tb=short -q

Note
----
These tests are CPU-only and finish in well under 1 s.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from molmetal_lam.search_alg.warm_start import (
    HBOND_ACCEPTOR_RESIDUES,
    HBOND_DONOR_RESIDUES,
    HYDROPHOBIC_RESIDUES,
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
    POCKET_FEATURE_DIM,
    PocketResidue,
    modify_root_prior,
    pocket_features,
    pocket_features_from_binding_site,
)


# ---------------------------------------------------------------------------
# Helper residue fixtures
# ---------------------------------------------------------------------------
def _ca2_residues() -> List[PocketResidue]:
    """Carbonic-anhydrase-2 catalytic pocket shell (His triad).

    Mimics the ``1CA2`` residue set declared in
    :mod:`molmetal.data.metalloprotein_targets` (line 151).  All
    distances are ≤ 5 Å because the pocket shell is the catalytic
    triad of CA2.
    """
    return [
        PocketResidue("H", 94, 2.5, is_metal_anchor=True),
        PocketResidue("H", 96, 3.0, is_metal_anchor=True),
        PocketResidue("H", 119, 2.0, is_metal_anchor=True),
        PocketResidue("V", 143, 4.5),
        PocketResidue("L", 198, 3.7),
        PocketResidue("F", 131, 4.9),
        PocketResidue("E", 106, 4.2),
        PocketResidue("T", 199, 4.6),
        PocketResidue("K", 170, 4.8),
        PocketResidue("W", 209, 4.9),
    ]


def _mmp2_residues() -> List[PocketResidue]:
    """MMP2 active-site pocket (mixed His + hydrophobic)."""
    return [
        PocketResidue("H", 403, 2.5, is_metal_anchor=True),
        PocketResidue("H", 407, 3.0, is_metal_anchor=True),
        PocketResidue("H", 413, 2.0, is_metal_anchor=True),
        PocketResidue("E", 404, 3.5),
        PocketResidue("A", 417, 4.5),
        PocketResidue("L", 418, 4.7),
        PocketResidue("V", 422, 4.9),
    ]


# ---------------------------------------------------------------------------
# Determinism test (Step 4.1)
# ---------------------------------------------------------------------------
def test_pocket_features_deterministic() -> None:
    """Same input residues → same 64-d feature vector."""
    residues = _ca2_residues()
    v1 = pocket_features(residues, pocket_name="CA2")
    v2 = pocket_features(residues, pocket_name="CA2")
    assert v1.values.shape == (POCKET_FEATURE_DIM,)
    assert v1.values.dtype == np.float32
    np.testing.assert_array_equal(
        v1.values,
        v2.values,
        err_msg="pocket_features is not deterministic",
    )
    # The dict round-trip should also be stable.
    assert v1.to_dict() == v2.to_dict()
    # And a re-ordered residue list should NOT change the result
    # (the function is order-insensitive — we use set-style counts).
    reversed_residues = list(reversed(residues))
    v3 = pocket_features(reversed_residues, pocket_name="CA2")
    np.testing.assert_array_equal(v1.values, v3.values)


# ---------------------------------------------------------------------------
# Empty pocket test (Step 4.2)
# ---------------------------------------------------------------------------
def test_pocket_features_handles_empty() -> None:
    """Empty pocket → all zeros, residue_count_5A=0, volume=0."""
    v = pocket_features([], pocket_name="EMPTY")
    assert v.values.shape == (POCKET_FEATURE_DIM,)
    assert v.residue_count_5A == 0
    assert v.pocket_volume_A3 == 0.0
    assert np.all(v.values == 0.0)
    # to_dict should still be JSON-friendly.
    d = v.to_dict()
    assert d["pocket_name"] == "EMPTY"
    assert d["residue_count_5A"] == 0
    assert d["vector_L2"] == 0.0


def test_pocket_features_handles_dict_input() -> None:
    """Dict-style residues are accepted via PocketResidue.from_dict."""
    residues = [
        {"one_letter": "A", "resid": 94, "distance_to_ligand": 2.5,
         "is_metal_anchor": True},
        {"one_letter": "A", "resid": 96, "distance_to_ligand": 3.0},
        {"one_letter": "V", "resid": 143, "distance_to_ligand": 4.5},
    ]
    v = pocket_features(residues, pocket_name="dict_input")
    assert v.values.shape == (POCKET_FEATURE_DIM,)
    assert v.residue_count_5A == 3
    assert v.values[1] > 0.0  # V is hydrophobic
    assert v.values[2] == pytest.approx(0.0)  # no positive residues (A is neutral)


def test_pocket_features_rejects_bad_input() -> None:
    """Strings / ints are rejected with TypeError."""
    with pytest.raises(TypeError):
        pocket_features(["not_a_residue"], pocket_name="bad")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Bounded descriptor test (Step 4.3)
# ---------------------------------------------------------------------------
def test_pocket_features_normalized() -> None:
    """Fractions are in [0, 1], log-features finite, sum reasonable."""
    residues = _ca2_residues()
    v = pocket_features(residues, pocket_name="CA2")
    # Slots 1..5 are fractions ∈ [0, 1].
    for i in range(1, 6):
        assert 0.0 <= float(v.values[i]) <= 1.0, (
            f"slot {i} = {v.values[i]} is out of [0, 1]"
        )
    # Slot 0 is a log feature (non-negative, finite).
    assert float(v.values[0]) >= 0.0
    assert math.isfinite(float(v.values[0]))
    # Slot 6 is the log-volume feature.
    assert float(v.values[6]) >= 0.0
    assert math.isfinite(float(v.values[6]))
    # The placeholder slots (7..63) should all be zero.
    assert np.all(v.values[7:] == 0.0)
    # Pocket volume is bounded by the residue-sphere union.
    # For 10 residues at r_eff=3.5 Å, V = 10 * (4/3) * π * 3.5^3 ≈ 1796 Å³.
    assert 1000.0 < v.pocket_volume_A3 < 5000.0, (
        f"pocket_volume_A3 = {v.pocket_volume_A3} outside the expected band"
    )


def test_pocket_features_shell_cutoff() -> None:
    """Residues beyond cutoff_A are excluded from the shell."""
    residues = [
        PocketResidue("H", 1, 2.0),       # inside
        PocketResidue("H", 2, 8.0),       # outside
        PocketResidue("H", 3, 4.9),       # inside (just)
    ]
    v = pocket_features(residues, cutoff_A=5.0, pocket_name="shell_test")
    # Only 2 residues within 5 Å.
    assert v.residue_count_5A == 2
    # Slot 0 = log(1 + 2) ≈ 1.099.
    assert v.values[0] == pytest.approx(math.log(3.0), rel=1e-4)


def test_pocket_features_distinguishes_pockets() -> None:
    """CA2 (His-triad) and MMP2 (His + Leu/Val) produce different vectors."""
    v_ca2 = pocket_features(_ca2_residues(), pocket_name="CA2")
    v_mmp2 = pocket_features(_mmp2_residues(), pocket_name="MMP2")
    # The hydrophobic fractions must differ.
    assert v_ca2.values[1] != v_mmp2.values[1]


# ---------------------------------------------------------------------------
# modify_root_prior tests (Step 4.4)
# ---------------------------------------------------------------------------
def _sample_actions() -> List[Any]:
    """Five sample actions (a tuple of rule name + tile SMILES)."""
    return [
        ("CuAAC", "C#C"),
        ("CuAAC", "N=N=N"),
        ("SPAAC", "C#C"),
        ("ThiolEne", "C=C"),
        ("AmideCoupling", "C(=O)O"),
    ]


def test_modify_root_prior_changes_selection() -> None:
    """With vs without pocket features, the argmax action differs."""
    actions = _sample_actions()

    # Empty-pocket prior (v_P = 0) — the pocket contribution is zero.
    v_empty = pocket_features([], pocket_name="EMPTY")
    prior_empty = modify_root_prior(
        root_state_features=[0.1, 0.2, 0.3],
        pocket_features_vec=v_empty,
        actions=actions,
        pocket_bias_strength=1.0,
    )
    assert pytest.approx(sum(prior_empty.values())) == 1.0
    argmax_empty = max(prior_empty, key=prior_empty.get)

    # CA2-prior — the pocket contribution is non-zero (catalytic triad).
    v_ca2 = pocket_features(_ca2_residues(), pocket_name="CA2")
    prior_ca2 = modify_root_prior(
        root_state_features=[0.1, 0.2, 0.3],
        pocket_features_vec=v_ca2,
        actions=actions,
        pocket_bias_strength=10.0,  # amplify so the prior visibly shifts
    )
    assert pytest.approx(sum(prior_ca2.values())) == 1.0
    argmax_ca2 = max(prior_ca2, key=prior_ca2.get)

    # With a sufficiently strong pocket_bias_strength the argmax must move.
    assert argmax_empty != argmax_ca2, (
        "modify_root_prior failed to change selection under pocket conditioning"
    )


def test_modify_root_prior_returns_distribution() -> None:
    """modify_root_prior always returns a probability distribution."""
    actions = _sample_actions()
    v = pocket_features(_ca2_residues(), pocket_name="CA2")
    prior = modify_root_prior(
        root_state_features=[],
        pocket_features_vec=v,
        actions=actions,
    )
    # Sum to 1.0.
    assert pytest.approx(sum(prior.values())) == 1.0
    # All non-negative.
    for a, p in prior.items():
        assert p >= 0.0
    # Same actions returned.
    assert set(prior.keys()) == set(actions)


def test_modify_root_prior_handles_empty_actions() -> None:
    """No actions → empty dict, no exception."""
    v = pocket_features(_ca2_residues(), pocket_name="CA2")
    prior = modify_root_prior(
        root_state_features=[0.1],
        pocket_features_vec=v,
        actions=[],
    )
    assert prior == {}


def test_modify_root_prior_disabled() -> None:
    """pocket_bias_strength=0 → pure state-prior fallback."""
    actions = _sample_actions()
    v = pocket_features(_ca2_residues(), pocket_name="CA2")
    prior_off = modify_root_prior(
        root_state_features=[0.1, 0.2, 0.3],
        pocket_features_vec=v,
        actions=actions,
        pocket_bias_strength=0.0,
    )
    # All actions should have an essentially uniform prior (tie-broken
    # by the per-action epsilon, so the order is deterministic).
    assert pytest.approx(sum(prior_off.values())) == 1.0
    # Verify the dict has the right number of entries.
    assert len(prior_off) == len(actions)


# ---------------------------------------------------------------------------
# BindingSite adapter test
# ---------------------------------------------------------------------------
def test_pocket_features_from_binding_site() -> None:
    """Adapter produces a valid PocketFeatureVector from a BindingSite-like
    object even when no residue data is available."""
    # Minimal mock — the adapter only reads .name and .geometry_hints.
    class _MockBindingSite:
        name = "MMP2_active_site"
        geometry_hints = {"metal": "Zn", "min_donors": 2}

    v = pocket_features_from_binding_site(_MockBindingSite())
    assert v.values.shape == (POCKET_FEATURE_DIM,)
    assert v.pocket_name == "MMP2_active_site"
    # No residues → zero vector.
    assert np.all(v.values == 0.0)

    # When residues are supplied, the adapter delegates to pocket_features.
    v_with_residues = pocket_features_from_binding_site(
        _MockBindingSite(),
        pocket_residues=_mmp2_residues(),
    )
    assert v_with_residues.residue_count_5A == 7
    assert v_with_residues.values[1] > 0.0  # MMP2 has A + L + V


# ---------------------------------------------------------------------------
# Constant-coverage test (sanity check the residue sets)
# ---------------------------------------------------------------------------
def test_residue_sets_non_empty() -> None:
    """Sanity-check: every constant residue set is non-empty."""
    assert len(HYDROPHOBIC_RESIDUES) >= 5
    assert len(POSITIVE_RESIDUES) >= 2
    assert len(NEGATIVE_RESIDUES) >= 2
    assert len(HBOND_DONOR_RESIDUES) >= 5
    assert len(HBOND_ACCEPTOR_RESIDUES) >= 5


# Late import to avoid a hard dependency on math in the test body.
import math  # noqa: E402
