"""Tests for molmetal/molmetal_lam/sbdd_env/sheaf_section.py.

Adapted from multiflow's contact_sheaf.py (P0-236L) for Mol-Metal's
cross-molecule similarity axis (TODO-143 §2.5 Sprint B2).

Test plan (16 checks)
=====================

MoleculeSection dataclass (1-4)
-------------------------------
1.  ``test_molecule_section_dataclass_creation`` — frozen=True is
    honoured; equality on field values.
2.  ``test_molecule_section_default_weight`` — default weight is 1.0.
3.  ``test_molecule_section_frozen_rejects_mutation`` — assigning to a
    field raises ``FrozenInstanceError`` (Dataclass convention).
4.  ``test_molecule_section_to_payload_json_safe`` — payload has
    only JSON-safe types (list of lists, sorted, float weight).

Payload roundtrip (5-6)
-----------------------
5.  ``test_section_from_payload_roundtrip`` — ``to_payload`` →
    ``section_from_payload`` → identical section.
6.  ``test_section_from_payload_uses_section_id_alias`` — the helper
    accepts both ``name`` and ``section_id`` keys (Multiflow compat).

overlap_residual (7-13)
-----------------------
7.  ``test_overlap_residual_identical_sections_zero`` — identical
    inputs yield residual 0.
8.  ``test_overlap_residual_disjoint_atoms_one`` — disjoint atoms →
    atom_disagreement 1.0 and headline residual at least 0.5.
9.  ``test_overlap_residual_disjoint_fragments_one`` — disjoint
    fragments → fragment_disagreement 1.0.
10. ``test_overlap_residual_uses_jaccard_bonds`` — bonds
    overlap / union = Jaccard similarity; ``1 - Jaccard`` is the
    headline.
11. ``test_overlap_residual_partial_atoms_in_range`` — partial atom
    overlap → atom_disagreement strictly in (0, 1).
12. ``test_overlap_residual_keys_present`` — all expected keys are in
    the result dict.
13. ``test_overlap_residual_similarity_invert`` — ``1 - residual`` is
    a valid similarity scalar in [0, 1].

molecule_atlas_payload (14-16)
------------------------------
14. ``test_molecule_atlas_payload_empty_sections`` — empty input
    returns zeroed payload with ``global_section_exists`` True.
15. ``test_molecule_atlas_payload_identical_sections_glues`` — two
    identical sections glue (no failed overlaps).
16. ``test_molecule_atlas_payload_disjoint_sections_no_overlap`` —
    disjoint sections skip the pair (overlap_pair_count = 0).
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from molmetal.molmetal_lam.sbdd_env.sheaf_section import (
    MoleculeSection,
    molecule_atlas_payload,
    overlap_residual,
    section_from_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pt_cisplatin_section() -> MoleculeSection:
    """Minimal Pt/N/H/Cl molecule section (atoms 0-5)."""
    return MoleculeSection(
        name="pt_cisplatin",
        atoms=frozenset({0, 1, 2, 3, 4, 5}),
        fragments=frozenset({"amine", "chloride"}),
        bonds=frozenset({(0, 1), (1, 2), (3, 4), (4, 5)}),
        weight=2.0,
    )


def _au_oxaliplatin_section() -> MoleculeSection:
    """A second metal-domain section sharing only the ``amine`` fragment."""
    return MoleculeSection(
        name="au_oxaliplatin",
        atoms=frozenset({10, 11, 12, 13, 14, 15}),
        fragments=frozenset({"amine", "aromatic_ring"}),
        bonds=frozenset({(10, 11), (11, 12), (13, 14)}),
        weight=1.0,
    )


# ---------------------------------------------------------------------------
# 1-4. MoleculeSection dataclass
# ---------------------------------------------------------------------------
def test_molecule_section_dataclass_creation() -> None:
    """MoleculeSection is constructible and value-equal."""
    a = _pt_cisplatin_section()
    b = _pt_cisplatin_section()
    assert a == b
    assert a.name == "pt_cisplatin"
    assert 0 in a.atoms
    assert "amine" in a.fragments
    assert (0, 1) in a.bonds


def test_molecule_section_default_weight() -> None:
    """Default weight is 1.0 when not supplied."""
    section = MoleculeSection(
        name="x",
        atoms=frozenset({1}),
        fragments=frozenset(),
        bonds=frozenset(),
    )
    assert section.weight == 1.0


def test_molecule_section_frozen_rejects_mutation() -> None:
    """``frozen=True`` blocks attribute assignment."""
    section = _pt_cisplatin_section()
    with pytest.raises(FrozenInstanceError):
        section.name = "mutated"  # type: ignore[misc]


def test_molecule_section_to_payload_json_safe() -> None:
    """``to_payload`` is JSON-serializable with sorted, JSON-safe types."""
    payload = _pt_cisplatin_section().to_payload()
    # Round-trip through json to assert no exotic types.
    text = json.dumps(payload)
    assert isinstance(text, str)
    # Sorted invariant for determinism.
    assert payload["atoms"] == sorted(payload["atoms"])
    assert payload["fragments"] == sorted(payload["fragments"])
    assert payload["bonds"] == sorted(payload["bonds"])
    assert isinstance(payload["weight"], float)


# ---------------------------------------------------------------------------
# 5-6. Payload roundtrip
# ---------------------------------------------------------------------------
def test_section_from_payload_roundtrip() -> None:
    """``to_payload`` → ``section_from_payload`` returns an equal section."""
    section = _pt_cisplatin_section()
    payload = section.to_payload()
    rebuilt = section_from_payload(payload)
    assert rebuilt == section


def test_section_from_payload_uses_section_id_alias() -> None:
    """``section_id`` is accepted as a ``name`` alias (Multiflow compat)."""
    payload = {
        "section_id": "legacy_pt",
        "atoms": [0, 1, 2],
        "fragments": ["amine"],
        "bonds": [[0, 1], [1, 2]],
        "weight": 0.5,
    }
    section = section_from_payload(payload)
    assert section.name == "legacy_pt"
    assert section.atoms == frozenset({0, 1, 2})
    assert section.weight == 0.5


# ---------------------------------------------------------------------------
# 7-13. overlap_residual
# ---------------------------------------------------------------------------
def test_overlap_residual_identical_sections_zero() -> None:
    """Identical sections give residual 0.0 on every axis."""
    a = _pt_cisplatin_section()
    b = _pt_cisplatin_section()
    res = overlap_residual(a, b)
    assert res["bond_disagreement"] == 0.0
    assert res["atom_disagreement"] == 0.0
    assert res["fragment_disagreement"] == 0.0
    assert res["overlap_residual"] == 0.0


def test_overlap_residual_disjoint_atoms_one() -> None:
    """Disjoint atoms → atom_disagreement == 1.0."""
    a = _pt_cisplatin_section()
    b = MoleculeSection(
        name="far",
        atoms=frozenset({100, 101}),
        fragments=frozenset(),
        bonds=frozenset(),
    )
    res = overlap_residual(a, b)
    assert res["atom_disagreement"] == 1.0
    # Headline residual takes max(bond, 0.5*atom, 0.5*fragment);
    # with disjoint atoms and no fragments this is at least 0.5.
    assert res["overlap_residual"] >= 0.5


def test_overlap_residual_disjoint_fragments_one() -> None:
    """Disjoint fragments → fragment_disagreement == 1.0.

    Empty bond sets on both sides so the bond axis doesn't dominate;
    that isolates the fragment axis for the headline residual.
    """
    a = MoleculeSection(
        name="amide",
        atoms=frozenset({1, 2, 9}),
        fragments=frozenset({"amide"}),
        bonds=frozenset(),
    )
    b = MoleculeSection(
        name="aromatic",
        atoms=frozenset({3, 4, 9}),
        fragments=frozenset({"aromatic_ring"}),
        bonds=frozenset(),
    )
    res = overlap_residual(a, b)
    # Shared atom {9} → atom_disagreement = 1 - 1/5 = 0.8.
    assert 0.0 < res["atom_disagreement"] < 1.0
    # Shared fragment set is empty → fragment_disagreement = 1.0.
    assert res["fragment_disagreement"] == 1.0
    # Empty bond union on both sides → bond_disagreement == 0.0
    # (the Jaccard convention treats the empty cover as fully
    # agreeing, identical to the Multiflow formula).
    assert res["bond_disagreement"] == 0.0
    # Headline residual = max(0.0, 0.5·0.8, 0.5·1.0) = 0.5.
    assert res["overlap_residual"] == pytest.approx(0.5)
    # Cross-check: a fragment-only atlas entry shows the disagreement
    # axis is wired up (vs. silently returning 0).
    assert res["shared_fragment_count"] == 0


def test_overlap_residual_uses_jaccard_bonds() -> None:
    """bond_disagreement = 1 - |L ∩ R| / |L ∪ R| (Jaccard distance)."""
    left = MoleculeSection(
        name="L",
        atoms=frozenset({0, 1, 2}),
        fragments=frozenset(),
        bonds=frozenset({(0, 1), (1, 2), (0, 2)}),
    )
    right = MoleculeSection(
        name="R",
        atoms=frozenset({0, 1, 2}),
        fragments=frozenset(),
        bonds=frozenset({(0, 1), (1, 2)}),
    )
    # Overlap = 2, Union = 3 → Jaccard sim 2/3 → disagreement 1/3.
    res = overlap_residual(left, right)
    assert res["bond_overlap_count"] == 2
    assert res["bond_union_count"] == 3
    assert res["bond_disagreement"] == pytest.approx(1.0 - 2.0 / 3.0)


def test_overlap_residual_partial_atoms_in_range() -> None:
    """Partial atom overlap → atom_disagreement strictly in (0, 1)."""
    left = MoleculeSection(
        name="L",
        atoms=frozenset({1, 2, 3, 4}),
        fragments=frozenset(),
        bonds=frozenset(),
    )
    right = MoleculeSection(
        name="R",
        atoms=frozenset({3, 4, 5, 6, 7}),
        fragments=frozenset(),
        bonds=frozenset(),
    )
    res = overlap_residual(left, right)
    # Shared {3,4}, union {1..7} → atom_disagreement = 1 - 2/7.
    assert 0.0 < res["atom_disagreement"] < 1.0
    assert res["shared_atom_count"] == 2
    # Headline residual caps at 0.5·atom_disagreement here.
    assert res["overlap_residual"] == pytest.approx(0.5 * res["atom_disagreement"])


def test_overlap_residual_keys_present() -> None:
    """Result dict contains every documented key."""
    res = overlap_residual(_pt_cisplatin_section(), _au_oxaliplatin_section())
    expected = {
        "left_section",
        "right_section",
        "shared_atom_count",
        "shared_fragment_count",
        "bond_overlap_count",
        "bond_union_count",
        "bond_disagreement",
        "atom_disagreement",
        "fragment_disagreement",
        "overlap_residual",
    }
    assert expected.issubset(res.keys())


def test_overlap_residual_similarity_invert() -> None:
    """``1 - residual`` is a valid similarity scalar in [0, 1]."""
    for left, right in [
        (_pt_cisplatin_section(), _pt_cisplatin_section()),
        (_pt_cisplatin_section(), _au_oxaliplatin_section()),
        (
            MoleculeSection(
                name="a", atoms=frozenset({1}), fragments=frozenset(), bonds=frozenset()
            ),
            MoleculeSection(
                name="b", atoms=frozenset({2}), fragments=frozenset(), bonds=frozenset()
            ),
        ),
    ]:
        residual = overlap_residual(left, right)["overlap_residual"]
        similarity = 1.0 - residual
        assert 0.0 <= similarity <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 14-16. molecule_atlas_payload
# ---------------------------------------------------------------------------
def test_molecule_atlas_payload_empty_sections() -> None:
    """Empty atlas → zeroed payload, ``global_section_exists`` True."""
    payload = molecule_atlas_payload([], atlas_id="empty")
    assert payload["section_count"] == 0
    assert payload["overlap_pair_count"] == 0
    assert payload["max_overlap_residual"] == 0.0
    assert payload["mean_overlap_residual"] == 0.0
    assert payload["failed_overlap_count"] == 0
    assert payload["global_section_exists"] is True
    assert payload["atlas_id"] == "empty"


def test_molecule_atlas_payload_identical_sections_glues() -> None:
    """Two identical sections glue cleanly (no failed overlaps)."""
    a = _pt_cisplatin_section()
    payload = molecule_atlas_payload([a, a], atlas_id="duplicate")
    assert payload["section_count"] == 2
    # One pair, residual 0.0 → no failures.
    assert payload["overlap_pair_count"] == 1
    assert payload["max_overlap_residual"] == 0.0
    assert payload["failed_overlap_count"] == 0
    assert payload["global_section_exists"] is True


def test_molecule_atlas_payload_disjoint_sections_no_overlap() -> None:
    """Disjoint sections (no shared atoms or fragments) skip pair entirely."""
    a = MoleculeSection(
        name="A",
        atoms=frozenset({0, 1}),
        fragments=frozenset({"amide"}),
        bonds=frozenset({(0, 1)}),
    )
    b = MoleculeSection(
        name="B",
        atoms=frozenset({10, 11}),
        fragments=frozenset({"aromatic_ring"}),
        bonds=frozenset({(10, 11)}),
    )
    payload = molecule_atlas_payload([a, b], atlas_id="disjoint")
    assert payload["overlap_pair_count"] == 0
    assert payload["max_overlap_residual"] == 0.0
    assert payload["mean_overlap_residual"] == 0.0
    assert payload["global_section_exists"] is True
