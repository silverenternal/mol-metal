"""Tests for the expanded MLC atom table (30 transition metals + 4 halogens).

These tests validate that ``molmetal_lam.atoms.combinators`` has been
extended to cover the full d-block (30 transition metals) on top of
the original 6 hand-curated ones (Pt_II, Ru_II, Zn_II, Ir_III,
Cu_II, Au_III).  They are the contract tests for the R17 ultracode
Track-B (Lambda calculus) wire-up — cf. ``TODO/pending/
32_ultracode_trackB_lambda_calculus.md`` §B1.

Required expanded state
-----------------------
* :data:`PRIMITIVE_ATOMS`     includes entries for all 30 d-block
  transition metals (by element symbol).
* :data:`METAL_SYMBOLS`       is a collection containing all 30
  d-block symbols (Sc..Hg per :data:`TRANSITION_METALS`).
* Every expanded entry has ``arity == valence + lone_pairs`` and a
  geometry drawn from
  ``{square_planar, octahedral, tetrahedral, linear, trigonal_planar}``.
* Cisplatin / carboplatin / oxaliplatin still parse via RDKit
  ``from_smiles`` and the metal centre is correctly tagged as Pt_II.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from molmetal_lam.atoms.combinators import (
    PRIMITIVE_ATOMS,
    from_smiles,
)

try:
    from molmetal_lam.atoms.combinators import METAL_SYMBOLS  # noqa: F401
except ImportError:
    METAL_SYMBOLS = None  # type: ignore[assignment]
try:
    from molmetal_lam.atoms.combinators import METAL_ATOMS  # noqa: F401
except ImportError:
    METAL_ATOMS = None  # type: ignore[assignment]
from molmetal_lam.lam_chem.platinai_dataset import TRANSITION_METALS


def _metal_symbol_set() -> set:
    """Return ``METAL_SYMBOLS`` as a ``set`` even when it's a tuple/frozenset.

    Returns an empty ``set`` when ``METAL_SYMBOLS`` has not been added
    to :mod:`molmetal_lam.atoms.combinators` yet — the test then
    fails individually rather than blocking collection.
    """
    if METAL_SYMBOLS is None:
        return set()
    if isinstance(METAL_SYMBOLS, (set, frozenset)):
        return set(METAL_SYMBOLS)
    if isinstance(METAL_SYMBOLS, Iterable):
        return set(METAL_SYMBOLS)
    return {METAL_SYMBOLS}


VALID_GEOMETRIES = frozenset(
    {
        "square_planar",
        "octahedral",
        "tetrahedral",
        "linear",
        "trigonal_planar",
    }
)


def test_30_transition_metals_in_primitive_atoms() -> None:
    """All 30 d-block transition metals are present in PRIMITIVE_ATOMS."""
    missing = [m for m in TRANSITION_METALS if m not in PRIMITIVE_ATOMS]
    assert missing == [], (
        f"PRIMITIVE_ATOMS missing {len(missing)} transition metals: "
        f"{missing[:5]}... (expected 30 total per platinai_dataset.TM_SET)"
    )


def test_halogens_present() -> None:
    """All 4 halogens (F/Cl/Br/I) are present in PRIMITIVE_ATOMS."""
    for h in ("F", "Cl", "Br", "I"):
        assert h in PRIMITIVE_ATOMS, f"halogen {h!r} missing from PRIMITIVE_ATOMS"


def test_metal_geometries_are_canonical() -> None:
    """Each transition metal has a valid d-block geometry."""
    bad = [
        (m, PRIMITIVE_ATOMS[m].geometry)
        for m in TRANSITION_METALS
        if m in PRIMITIVE_ATOMS and PRIMITIVE_ATOMS[m].geometry not in VALID_GEOMETRIES
    ]
    assert bad == [], f"metals with non-canonical geometry: {bad[:5]}"


def test_arity_equals_valence_plus_lone_pairs() -> None:
    """arity == valence + lone_pairs for every PRIMITIVE_ATOMS entry."""
    off = [
        (sym, a.valence, a.lone_pairs, a.arity)
        for sym, a in PRIMITIVE_ATOMS.items()
        if a.arity != a.valence + a.lone_pairs
    ]
    assert off == [], f"PRIMITIVE_ATOMS entries with wrong arity: {off[:5]}"


def test_cisplatin_still_parses() -> None:
    """Cisplatin SMILES parses; metal centre is Pt_II with arity 4."""
    smi = "[NH3][Pt](Cl)(Cl)[NH3]"
    atoms = from_smiles(smi)
    pt_atoms = [a for a in atoms if a.is_metal]
    assert len(pt_atoms) == 1
    pt = pt_atoms[0]
    assert pt.symbol == "Pt_II"
    assert pt.arity == 4


def test_carboplatin_and_oxaliplatin_parse() -> None:
    """Carboplatin and oxaliplatin parse; metal is Pt_II with arity 4."""
    cases = {
        "carboplatin": "N.N.[Pt]12(OC(=O)CCC1)OC(=O)CCC2",
        "oxaliplatin": "N.N.O[Pt]([O-])([NH3])([NH3])OC(=O)[C@@H](N)CCC(=O)O",
    }
    for name, smi in cases.items():
        atoms = from_smiles(smi)
        metals = [a for a in atoms if a.is_metal]
        assert len(metals) == 1, f"{name}: expected 1 metal, got {len(metals)}"
        assert metals[0].symbol == "Pt_II", f"{name}: metal={metals[0].symbol}"
        assert metals[0].arity == 4, f"{name}: arity={metals[0].arity}"


def test_metal_symbols_includes_all_30() -> None:
    """METAL_SYMBOLS contains all 30 d-block transition metal symbols."""
    sym_set = _metal_symbol_set()
    missing = [m for m in TRANSITION_METALS if m not in sym_set]
    assert missing == [], (
        f"METAL_SYMBOLS missing {len(missing)} transition metals: " f"{missing[:5]}..."
    )


# ---------------------------------------------------------------------------
# R17 UC-B1 — METAL_ATOMS extension tests
# ---------------------------------------------------------------------------

#: METAL_ATOMS keys with oxidation-state suffix that we expect to find.
EXPECTED_METAL_ATOMS_KEYS: frozenset = frozenset(
    {
        # Original 6 (R16 contract; preserve semantics)
        "Pt_II",
        "Ru_II",
        "Zn_II",
        "Ir_III",
        "Cu_II",
        "Au_III",
        # 3d (10) — Pt_II/Zn_II/Cu_II already above
        "Sc_III",
        "Ti_IV",
        "V_V",
        "Cr_III",
        "Mn_II",
        "Fe_II",
        "Co_II",
        "Ni_II",
        # 4d (10) — Ru_II already above
        "Y_III",
        "Zr_IV",
        "Nb_V",
        "Mo_VI",
        "Tc_VII",
        "Rh_III",
        "Pd_II",
        "Ag_I",
        "Cd_II",
        # 5d + La (8) — Pt_II/Ir_III/Au_III already above
        "La_III",
        "Hf_IV",
        "Ta_V",
        "W_VI",
        "Re_VII",
        "Os_III",
        "Hg_II",
    }
)


def test_metal_atoms_has_at_least_30_entries() -> None:
    """METAL_ATOMS dict has >= 30 entries (R17 UC-B1 contract)."""
    if METAL_ATOMS is None:
        pytest.fail("METAL_ATOMS not exported from molmetal_lam.atoms.combinators")
    assert (
        len(METAL_ATOMS) >= 30
    ), f"METAL_ATOMS has only {len(METAL_ATOMS)} entries (expected >=30)"


def test_metal_atoms_keys_match_expected_set() -> None:
    """All 30 expected METAL_ATOMS keys (with oxidation suffix) are present."""
    if METAL_ATOMS is None:
        pytest.fail("METAL_ATOMS not exported from molmetal_lam.atoms.combinators")
    missing = sorted(EXPECTED_METAL_ATOMS_KEYS - set(METAL_ATOMS.keys()))
    assert missing == [], f"METAL_ATOMS missing {len(missing)} expected keys: {missing}"


@pytest.mark.parametrize("symbol", sorted(EXPECTED_METAL_ATOMS_KEYS))
def test_metal_atoms_arity_matches_coordination_number(symbol: str) -> None:
    """For each METAL_ATOMS entry: arity = valence + lone_pairs (contract)."""
    if METAL_ATOMS is None:
        pytest.fail("METAL_ATOMS not exported from molmetal_lam.atoms.combinators")
    atom = METAL_ATOMS[symbol]
    assert atom.arity == atom.valence + atom.lone_pairs, (
        f"{symbol}: arity={atom.arity}, valence={atom.valence}, "
        f"lone_pairs={atom.lone_pairs} (arity != v + lp)"
    )
    assert atom.is_metal is True, f"{symbol}: is_metal=False (should be True)"
    assert (
        atom.geometry in VALID_GEOMETRIES
    ), f"{symbol}: geometry={atom.geometry!r} not in {VALID_GEOMETRIES}"


@pytest.mark.parametrize(
    "symbol,expected_arity",
    [
        # Original 6 (preserve R16 semantics)
        ("Pt_II", 4),
        ("Ru_II", 6),
        ("Zn_II", 4),
        ("Ir_III", 6),
        ("Cu_II", 4),
        ("Au_III", 4),
        # 3d
        ("Sc_III", 6),
        ("Ti_IV", 6),
        ("V_V", 6),
        ("Cr_III", 6),
        ("Mn_II", 6),
        ("Fe_II", 6),
        ("Co_II", 6),
        ("Ni_II", 6),
        # 4d
        ("Y_III", 6),
        ("Zr_IV", 6),
        ("Nb_V", 6),
        ("Mo_VI", 6),
        ("Tc_VII", 6),
        ("Rh_III", 4),
        ("Pd_II", 4),
        ("Ag_I", 2),
        ("Cd_II", 4),
        # 5d + La
        ("La_III", 6),
        ("Hf_IV", 6),
        ("Ta_V", 6),
        ("W_VI", 6),
        ("Re_VII", 6),
        ("Os_III", 6),
        ("Hg_II", 2),
    ],
)
def test_metal_atoms_arity_table(symbol: str, expected_arity: int) -> None:
    """Each METAL_ATOMS entry's arity matches its canonical coordination number."""
    if METAL_ATOMS is None:
        pytest.fail("METAL_ATOMS not exported from molmetal_lam.atoms.combinators")
    assert (
        METAL_ATOMS[symbol].arity == expected_arity
    ), f"{symbol}: arity={METAL_ATOMS[symbol].arity} != expected {expected_arity}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
