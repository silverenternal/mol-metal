"""Tests for the layer-1/2/3 metric instrumentation.

One test per metric (17 total):

* L1 (5): ARITY_HIT_RATE, METAL_GEOMETRY_OK, SANITY_PASS_RATE,
          FALLBACK_ATOM_RATIO, PRIMITIVE_GEOMETRY_TAG_OK.
* L2 (6): BOND_KIND_DISTRIBUTION, DATIVE_FRACTION,
          FREE_SITES_AFTER_ASSEMBLE, BOND_VALIDITY_RATE,
          BUILDER_EXCEPTION_RATE, AROMATIC_RING_SIZE_OK.
* L3 (6): IS_CLOSED_RATE, IS_BETA_NORMAL_FORM_RATE, REDEX_HIT_RATE,
          REDEX_REDUCTION_RATE, NF_TERM, ALPHA_EQUIV_COLLISIONS.
          (ATOM_BOND_RATIO is exercised by the same probes that hit
          IS_CLOSED_RATE.)

Each test invokes the corresponding layer entry point and asserts
that the expected metric dict is appended to the layer's history.
"""

from __future__ import annotations

import pytest

from molmetal_lam.atoms import combinators as l1
from molmetal_lam.bonds import application as l2
from molmetal_lam.molecules import closed_term as l3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_all_histories() -> None:
    """Clear every layer's history before each test."""
    l1.reset_history()
    l2.reset_history()
    l3.reset_history()


def _has_metric(history, name: str) -> dict | None:
    for entry in history:
        if entry.get("metric") == name:
            return entry
    return None


# ---------------------------------------------------------------------------
# Layer 1 — atoms/combinators.py  (5 tests)
# ---------------------------------------------------------------------------

def test_l1_arity_hit_rate_recorded() -> None:
    """ARITY_HIT_RATE is appended when from_smiles parses a SMILES."""
    l1.from_smiles("CCO")
    entry = _has_metric(l1.get_history(), "ARITY_HIT_RATE")
    assert entry is not None
    assert 0.0 <= entry["value"] <= 1.0


def test_l1_metal_geometry_ok_recorded() -> None:
    """METAL_GEOMETRY_OK is appended on sanity_check()."""
    l1.sanity_check()
    entry = _has_metric(l1.get_history(), "METAL_GEOMETRY_OK")
    assert entry is not None
    assert 0.0 <= entry["value"] <= 1.0


def test_l1_sanity_pass_rate_recorded() -> None:
    """SANITY_PASS_RATE is appended on sanity_check()."""
    l1.sanity_check()
    entry = _has_metric(l1.get_history(), "SANITY_PASS_RATE")
    assert entry is not None
    assert entry["n_checks"] == 3
    assert entry["n_passed"] == 3


def test_l1_fallback_atom_ratio_recorded() -> None:
    """FALLBACK_ATOM_RATIO is appended when from_smiles hits a fallback."""
    l1.from_smiles("CCO")  # all in PRIMITIVE_ATOMS
    entry = _has_metric(l1.get_history(), "FALLBACK_ATOM_RATIO")
    assert entry is not None
    assert entry["value"] == 0.0


def test_l1_primitive_geometry_tag_ok_recorded() -> None:
    """PRIMITIVE_GEOMETRY_TAG_OK is appended on sanity_check()."""
    l1.sanity_check()
    entry = _has_metric(l1.get_history(), "PRIMITIVE_GEOMETRY_TAG_OK")
    assert entry is not None
    assert entry["n_primitive"] == 10
    # All 10 primitives carry a non-empty geometry tag.
    assert entry["value"] == 1.0


# ---------------------------------------------------------------------------
# Layer 2 — bonds/application.py  (6 tests)
# ---------------------------------------------------------------------------

def test_l2_bond_kind_distribution_recorded() -> None:
    """BOND_KIND_DISTRIBUTION is appended by assemble()."""
    l2.assemble([])
    entry = _has_metric(l2.get_history(), "BOND_KIND_DISTRIBUTION")
    assert entry is not None
    assert isinstance(entry["value"], dict)


def test_l2_dative_fraction_recorded() -> None:
    """DATIVE_FRACTION: cisplatin's 4/4 dative bonds."""
    mol = l2.cisplatin()
    entry = _has_metric(l2.get_history(), "DATIVE_FRACTION")
    assert entry is not None
    assert entry["n_dative"] == 4
    assert entry["value"] == 1.0
    assert mol["is_closed"] is True


def test_l2_free_sites_after_assemble_recorded() -> None:
    """FREE_SITES_AFTER_ASSEMBLE exposes open_sites + is_closed."""
    l2.cisplatin()
    entry = _has_metric(l2.get_history(), "FREE_SITES_AFTER_ASSEMBLE")
    assert entry is not None
    assert entry["value"]["open_sites"] == 0
    assert entry["value"]["is_closed"] is True


def test_l2_bond_validity_rate_recorded() -> None:
    """BOND_VALIDITY_RATE is appended with the assemble() validity flag."""
    l2.cisplatin()
    entry = _has_metric(l2.get_history(), "BOND_VALIDITY_RATE")
    assert entry is not None
    assert entry["value"] == 1.0


def test_l2_builder_exception_rate_recorded() -> None:
    """BUILDER_EXCEPTION_RATE is appended via builder_exception_snapshot()."""
    # Trigger a few factory calls — none should raise.
    from molmetal_lam.atoms.combinators import METAL_ATOMS, make_ligand
    pt = l2.distinct(METAL_ATOMS["Pt_II"])
    nh3 = l2.distinct(make_ligand("NH3"))
    l2.safe_factory(l2.Bond.dative, pt, nh3)
    snap = l2.builder_exception_snapshot()
    entry = _has_metric(l2.get_history(), "BUILDER_EXCEPTION_RATE")
    assert entry is not None
    assert snap["calls"] >= 1
    assert entry["value"] >= 0.0


def test_l2_aromatic_ring_size_ok_recorded() -> None:
    """AROMATIC_RING_SIZE_OK is appended by assemble()."""
    l2.assemble([])
    entry = _has_metric(l2.get_history(), "AROMATIC_RING_SIZE_OK")
    assert entry is not None
    assert 0.0 <= entry["value"] <= 1.0


# ---------------------------------------------------------------------------
# Layer 3 — molecules/closed_term.py  (6 tests)
# ---------------------------------------------------------------------------

def test_l3_is_closed_rate_recorded() -> None:
    """IS_CLOSED_RATE: water (O + 2 implicit H) is closed."""
    term = l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    entry = _has_metric(l3.get_history(), "IS_CLOSED_RATE")
    assert entry is not None
    assert entry["value"] == 1.0
    assert term.is_closed is True


def test_l3_is_beta_normal_form_rate_recorded() -> None:
    """IS_BETA_NORMAL_FORM_RATE: water is in β-NF (closed + no redex)."""
    l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    entry = _has_metric(l3.get_history(), "IS_BETA_NORMAL_FORM_RATE")
    assert entry is not None
    assert entry["value"] == 1.0


def test_l3_redex_hit_rate_recorded() -> None:
    """REDEX_HIT_RATE: water has no redex."""
    l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    entry = _has_metric(l3.get_history(), "REDEX_HIT_RATE")
    assert entry is not None
    assert entry["value"] == 0.0


def test_l3_redex_reduction_rate_recorded() -> None:
    """REDEX_REDUCTION_RATE: reduce_once on a no-redex term is a no-op."""
    term = l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    l3.reset_history()  # only record the reduce_once metric
    term.reduce_once()
    entry = _has_metric(l3.get_history(), "REDEX_REDUCTION_RATE")
    # Either a redex was found (changed=True) or none (changed=False).
    # The metric must always be recorded.
    assert entry is not None
    assert entry["value"] in (0.0, 1.0)


def test_l3_nf_term_recorded() -> None:
    """NF_TERM: β-NF boolean emitted after reduce_once()."""
    term = l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    l3.reset_history()
    term.reduce_once()
    entry = _has_metric(l3.get_history(), "NF_TERM")
    assert entry is not None
    assert isinstance(entry["value"], bool)


def test_l3_alpha_equiv_collisions_recorded() -> None:
    """ALPHA_EQUIV_COLLISIONS: canonical-SMILES round-trip is stable."""
    l3.MoleculeClosedTerm.from_smiles("O", embed_3d=False)
    entry = _has_metric(l3.get_history(), "ALPHA_EQUIV_COLLISIONS")
    assert entry is not None
    # Water round-trips canonically with zero collisions.
    assert entry["value"] == 0.0
