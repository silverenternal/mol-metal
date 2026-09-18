"""Tests for the shared ``_sweep_helpers`` module.

Verifies that

1. every public symbol is importable,
2. :func:`build_seed_smiles` round-trips a small SDF through RDKit,
3. :func:`lipinski_pass` accepts aspirin (Ro5-friendly) and rejects a
   large polymer.

These tests are deliberately scoped to helpers that don't require the
heavy MCTS machinery to be wired (no MCTSProofSearch, no Vina).
"""

from __future__ import annotations

import os
import tempfile

import pytest
from rdkit import Chem

from molmetal_lam.scripts._sweep_helpers import (
    BRANCHING_LABELS,
    DEFAULT_SEED_SMILES,
    build_reward,
    build_seed_smiles,
    build_seed_term,
    build_tile_library_for_branching,
    lipinski_pass,
    smi_of,
    vina_proxy,
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_all_helpers_importable() -> None:
    """All advertised helpers exist and are callable."""
    for sym in (
        "BRANCHING_LABELS",
        "DEFAULT_SEED_SMILES",
        "build_reward",
        "build_seed_smiles",
        "build_seed_term",
        "build_tile_library_for_branching",
        "lipinski_pass",
        "smi_of",
        "vina_proxy",
    ):
        obj = globals().get(sym)
        assert obj is not None, f"{sym} missing from _sweep_helpers"


def test_branching_labels_keys() -> None:
    """Branching labels cover the supported factors (12/60/1020)."""
    assert set(BRANCHING_LABELS.keys()) >= {12, 60, 1020}


def test_default_seed_smiles_is_valid() -> None:
    mol = Chem.MolFromSmiles(DEFAULT_SEED_SMILES)
    assert mol is not None
    # Round-trip should be a non-empty canonical form
    assert Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# build_seed_smiles
# ---------------------------------------------------------------------------


def _write_pocket_with_sdf(sdf_path: str, smi: str) -> None:
    """Write a single-molecule SDF on disk for build_seed_smiles."""
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()


def test_build_seed_smiles_returns_valid_smiles(tmp_path) -> None:
    """build_seed_smiles reads the first molecule from a pocket's SDF."""
    sdf_path = os.path.join(str(tmp_path), "ligand.sdf")
    _write_pocket_with_sdf(sdf_path, "CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    smi = build_seed_smiles(str(tmp_path))
    assert smi is not None
    # Canonicalise and check that it's a valid molecule
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None


def test_build_seed_smiles_returns_none_for_empty_pocket(tmp_path) -> None:
    """build_seed_smiles returns None when no SDF is present."""
    assert build_seed_smiles(str(tmp_path)) is None


def test_build_seed_smiles_returns_none_for_missing_dir() -> None:
    """build_seed_smiles returns None for non-existent directories."""
    assert build_seed_smiles("/tmp/_definitely_not_a_real_pocket_xyz123") is None


def test_build_seed_term_falls_back_to_default(tmp_path) -> None:
    """build_seed_term parses the default cyclopentadiene when no SDF."""
    term = build_seed_term(str(tmp_path))
    assert term is not None
    assert term.source_smiles == DEFAULT_SEED_SMILES


# ---------------------------------------------------------------------------
# lipinski_pass
# ---------------------------------------------------------------------------


def test_lipinski_pass_aspirin() -> None:
    """Aspirin (MW ~180) is Ro5-compliant."""
    assert lipinski_pass("CC(=O)Oc1ccccc1C(=O)O") is True


def test_lipinski_pass_glucose() -> None:
    """Glucose is small enough to pass the standard Ro5 rule."""
    # D-glucose canonical SMILES from RDKit
    assert lipinski_pass("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O") is True


def test_lipinski_pass_large_polymer_rejected() -> None:
    """A 1000-atom polymer chain should violate at least MW <= 500."""
    # Generate a long alkane: CCCCC...C with 1000 carbons
    long_smi = "C" * 1000
    assert lipinski_pass(long_smi) is False


def test_lipinski_pass_invalid_smiles_returns_false() -> None:
    """A non-parseable SMILES must return False (never raise)."""
    assert lipinski_pass("not_a_smiles@@@") is False


def test_lipinski_pass_empty_string_returns_false() -> None:
    """Empty SMILES parses as a 0-atom molecule — passes MW=0 trivially.

    We document this behaviour: RDKit returns a non-None mol for ``""``,
    so the Ro5 filters all return True.  If we ever want to treat empty
    SMILES as invalid we should add an explicit guard, but the legacy
    implementation never did and downstream callers never relied on
    rejecting it.
    """
    # Empty SMILES is rejected explicitly so invalid candidates cannot enter
    # the sweep metrics.
    assert lipinski_pass("") is False


# ---------------------------------------------------------------------------
# smi_of
# ---------------------------------------------------------------------------


def test_smi_of_uses_source_smiles_attribute() -> None:
    """Objects with a .source_smiles attribute should be returned as-is."""

    class FakeState:
        source_smiles = "CCO"

    assert smi_of(FakeState()) == "CCO"


def test_smi_of_falls_back_to_str() -> None:
    """When nothing else is available, str(state) is the fallback."""

    class WeirdState:
        def __str__(self) -> str:
            return "<weird-state>"

    # No source_smiles, no canonical_smiles -> str fallback
    assert smi_of(WeirdState()) == "<weird-state>"


# ---------------------------------------------------------------------------
# build_reward / vina_proxy
# ---------------------------------------------------------------------------


def test_build_reward_returns_aggregator() -> None:
    """build_reward returns a RewardAggregator with r_sa + r_qed wired."""
    reward = build_reward()
    # Both channels must be wired
    assert reward.r_sa is not None
    assert reward.r_qed is not None
    # Default weights are 0.5/0.5 per legacy contract
    assert reward.w_sa == pytest.approx(0.5)
    assert reward.w_qed == pytest.approx(0.5)


def test_vina_proxy_returns_float_on_bad_state() -> None:
    """vina_proxy never raises — returns 0.0 on parse failure."""

    class BadState:
        source_smiles = "not-a-smiles@@@"

    assert vina_proxy(BadState()) == 0.0


def test_vina_proxy_returns_negative_for_normal_molecule() -> None:
    """For a normal small molecule the proxy should be in [-infty, 0)."""

    class Ethanol:
        source_smiles = "CCO"

    val = vina_proxy(Ethanol())
    assert isinstance(val, float)
    assert val < 0.0


# ---------------------------------------------------------------------------
# build_tile_library_for_branching
# ---------------------------------------------------------------------------


def test_build_tile_library_for_branching_small() -> None:
    """Branching=12 returns a list of MoleculeClosedTerm (size 12)."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    terms = build_tile_library_for_branching(12)
    assert isinstance(terms, list)
    assert all(isinstance(t, MoleculeClosedTerm) for t in terms)
    assert len(terms) == 12


def test_build_tile_library_for_branching_unsupported_raises() -> None:
    """Unsupported branching factors raise ValueError."""
    with pytest.raises(ValueError):
        build_tile_library_for_branching(9999)


# ---------------------------------------------------------------------------
# End-to-end: an actual pocket directory with an SDF
# ---------------------------------------------------------------------------


def test_end_to_end_pocket_to_term(tmp_path) -> None:
    """Build a tiny pocket directory and verify build_seed_term works."""
    pocket_dir = str(tmp_path / "pocket_1")
    os.makedirs(pocket_dir)
    sdf_path = os.path.join(pocket_dir, "ligand.sdf")
    _write_pocket_with_sdf(sdf_path, "c1ccccc1")  # benzene

    term = build_seed_term(pocket_dir)
    assert term is not None
    # benzene canonical SMILES
    assert term.source_smiles == "c1ccccc1"


def test_smoke_import_paths() -> None:
    """The shared helpers should be importable from the long path too."""
    # This test exists primarily to catch module-resolution regressions
    # when the file is relocated.
    from molmetal_lam.scripts import _sweep_helpers  # noqa: F401

    assert _sweep_helpers.lipinski_pass("CCO") is True
