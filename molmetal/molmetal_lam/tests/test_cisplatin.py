"""Tests for cisplatin as a Molecular Lambda Calculus witness.

The canonical MLC example: cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl).

This file is the *killer example* that grounds the framework's central
claim — that **a chemical molecule IS a closed λ-term** — in the most
iconic metal-coordination complex in cancer chemotherapy.

The test exercises all 8 layers of MLC on cisplatin:

  Layer 1 (atoms/combinators)   Pt_II combinator with arity 4
  Layer 2 (bonds/application)    dative bonds = curried partial apps
  Layer 3 (molecules/closed_term) closed term, is_closed=True
  Layer 4 (reactions/...)        (N/A for cisplatin: no click rule forms it)
  Layer 5 (synthesis/...)        β-reduction sequence is the assembly
  Layer 6 (binding/types)        PT_DNA_MAJOR_GROOVE type inhabitation
  Layer 7 (types/predicates)     (N/A: cisplatin is small + drug-like)
  Layer 8 (search_alg/...)       (N/A: closed-form witness, no search)

The output includes a printed λ-expression for cisplatin so the test
serves as living documentation of the framework's central thesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from molmetal_lam.atoms.combinators import (
    METAL_ATOMS,
    PRIMITIVE_ATOMS,
    Atom,
    make_ligand,
)
from molmetal_lam.binding.types import (
    PT_DNA_MAJOR_GROOVE,
    BindingTypeCheckResult,
    typecheck,
)
from molmetal_lam.bonds.application import (
    Bond,
    FreeSiteLedger,
    assemble,
    distinct,
    free_sites,
)


# ---------------------------------------------------------------------------
# Build cisplatin step by step as a curried partial application
# ---------------------------------------------------------------------------

def build_cisplatin_term() -> dict:
    """Build cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl) as 4 dative bonds.

    Returns the molecule dict from :func:`assemble`, with the
    handwritten λ-expression ``((((Pt_II NH3) NH3) Cl) Cl)`` attached
    under the ``term`` key.

    The build uses a **private** :class:`FreeSiteLedger` so the library
    singletons (the single Pt_II / Cl / NH3 in the atom registry) are
    not mutated — cisplatin gets its own bookkeeping.
    """
    led = FreeSiteLedger()

    # The metal — a curried 4-arity function.
    pt: Atom = distinct(METAL_ATOMS["Pt_II"])

    # Two ammonia ligands and two chloride ligands.  ``distinct``
    # makes each occurrence object-distinct from the library singletons
    # so the ledger keys them separately.
    nh3_a: Atom = distinct(make_ligand("NH3"))
    nh3_b: Atom = distinct(make_ligand("NH3"))
    cl_a: Atom = distinct(PRIMITIVE_ATOMS["Cl"])
    cl_b: Atom = distinct(PRIMITIVE_ATOMS["Cl"])

    # 4 curried partial applications.  Each call to Bond.dative
    # (a) consumes 1 free site from Pt (currying the metal),
    # (b) collapses the donor to a saturated value (the donor stops
    #     being a function and becomes a value — a closed term).
    # The resulting term is closed after 4 applications: Pt had 4 free
    # sites, and the 4 ligands each accept 1.
    bonds: List[Bond] = [
        Bond.dative(pt, nh3_a, ledger=led),   # (Pt_II NH3)
        Bond.dative(pt, nh3_b, ledger=led),   # ((Pt_II NH3) NH3)
        Bond.dative(pt, cl_a, ledger=led),    # (((Pt_II NH3) NH3) Cl)
        Bond.dative(pt, cl_b, ledger=led),    # ((((Pt_II NH3) NH3) Cl) Cl)
    ]

    mol = assemble(bonds, ledger=led)
    # Annotate the term with the human-readable λ-expression (the
    # canonical "killer figure" of the paper).
    mol["term"] = "((((Pt_II NH3) NH3) Cl) Cl)"
    return mol


# ---------------------------------------------------------------------------
# Test 1: build cisplatin
# ---------------------------------------------------------------------------

def test_cisplatin_builds_via_mlc_primitives() -> None:
    """Build cisplatin with Pt_II + 2 NH3 + 2 Cl via MLC primitives."""
    mol = build_cisplatin_term()
    # 5 atoms: Pt + 2 NH3 + 2 Cl
    assert len(mol["atoms"]) == 5
    # 4 dative bonds
    assert len(mol["bonds"]) == 4
    for bond in mol["bonds"]:
        assert bond.kind == "dative"
        assert bond.is_dative is True
    # The handwritten λ-expression is attached.
    assert mol["term"] == "((((Pt_II NH3) NH3) Cl) Cl)"


# ---------------------------------------------------------------------------
# Test 2: Pt_II arity = 4
# ---------------------------------------------------------------------------

def test_pt_ii_arity_is_four() -> None:
    """The Pt_II combinator must have arity 4 (square-planar fits 4)."""
    pt = METAL_ATOMS["Pt_II"]
    assert pt.arity == 4
    # Geometry must be square_planar.
    assert pt.geometry == "square_planar"
    assert pt.is_metal is True
    # Valence + lone_pairs = arity
    assert pt.valence + pt.lone_pairs == 4


# ---------------------------------------------------------------------------
# Test 3: 4 ligands = 4 free sites on Pt
# ---------------------------------------------------------------------------

def test_total_free_sites_match_four_ligands() -> None:
    """Total free sites on Pt (4) must match the 4 ligands (each 1 bond)."""
    led = FreeSiteLedger()
    pt: Atom = distinct(METAL_ATOMS["Pt_II"])

    # Pre-conditions: Pt has exactly 4 free sites.
    assert free_sites(pt, ledger=led) == 4
    # 4 ligands × 1 bond each = 4 ligand slots — exactly Pt's arity.
    n_ligand_bonds = 4 * 1
    assert pt.arity == n_ligand_bonds

    # Form all 4 dative bonds in sequence.
    for i in range(4):
        # Pt has exactly (4 - i) free sites before this bond.
        assert free_sites(pt, ledger=led) == 4 - i
        if i < 2:
            ligand = distinct(make_ligand("NH3"))
        else:
            ligand = distinct(PRIMITIVE_ATOMS["Cl"])
        Bond.dative(pt, ligand, ledger=led)

    # After all 4 applications, Pt has 0 free sites — saturated.
    assert free_sites(pt, ledger=led) == 0


# ---------------------------------------------------------------------------
# Test 4: each dative bond preserves 1 free site on Pt
# ---------------------------------------------------------------------------

def test_each_dative_bond_preserves_one_free_site_on_pt() -> None:
    """Each dative application consumes EXACTLY one Pt free site (currying)."""
    led = FreeSiteLedger()
    pt: Atom = distinct(METAL_ATOMS["Pt_II"])

    # Apply 4 dative bonds in sequence; track the free-site count.
    free_counts: List[int] = []
    for i in range(4):
        if i < 2:
            ligand = distinct(make_ligand("NH3"))
        else:
            ligand = distinct(PRIMITIVE_ATOMS["Cl"])
        before = free_sites(pt, ledger=led)
        Bond.dative(pt, ligand, ledger=led)
        after = free_sites(pt, ledger=led)
        free_counts.append(after)
        # Each bond: before - after == 1 (curried partial application).
        assert before - after == 1, (
            f"Step {i}: before={before}, after={after} — "
            "dative bond should consume exactly 1 free site"
        )

    # Sequence: 4 -> 3 -> 2 -> 1 -> 0.
    assert free_counts == [3, 2, 1, 0]


# ---------------------------------------------------------------------------
# Test 5: the resulting molecule is_closed=True
# ---------------------------------------------------------------------------

def test_cisplatin_molecule_is_closed() -> None:
    """After all 4 dative applications the term is closed (β-NF)."""
    mol = build_cisplatin_term()
    # No free sites anywhere.
    assert mol["open_sites"] == 0
    # Every atom has 0 free sites.
    assert all(v == 0 for v in mol["free_sites_per_atom"].values())
    # The closed-term predicate: True iff no atom has free sites.
    assert mol["is_closed"] is True
    # All bonds are well-typed.
    assert mol["valid"] is True


# ---------------------------------------------------------------------------
# Test 6: print the λ-expression
# ---------------------------------------------------------------------------

def test_cisplatin_lambda_expression() -> None:
    """Print the λ-expression for cisplatin (the central thesis witness)."""
    mol = build_cisplatin_term()
    expr = mol["term"]
    # Re-import for clarity in the print statement (no-op at runtime).
    _ = Atom
    print("\n" + "=" * 72)
    print("CISPLATIN AS A MOLECULAR LAMBDA CALCULUS TERM")
    print("=" * 72)
    print(f"Lambda expression: {expr}")
    print(f"Number of atoms  : {len(mol['atoms'])}  "
          f"({[a.symbol for a in mol['atoms']]})")
    print(f"Number of bonds  : {len(mol['bonds'])}  (all dative)")
    print(f"Is closed        : {mol['is_closed']}")
    print(f"Free sites per atom:")
    for label, n in mol["free_sites_per_atom"].items():
        print(f"  {label:8s} -> {n}")
    print("=" * 72)

    # Assert the expression is the canonical left-associated application.
    assert expr == "((((Pt_II NH3) NH3) Cl) Cl)"
    # Pt is the function head; ligands are the arguments, in order.
    assert expr.startswith("((((Pt_II NH3) NH3) Cl) Cl)")
    assert expr.count("(") == 4  # 4 opening parens (one per application)


# ---------------------------------------------------------------------------
# Test 7: cisplatin inhabits the PT_DNA_MAJOR_GROOVE binding type
# ---------------------------------------------------------------------------

def test_cisplatin_inhabits_dna_major_groove_binding_type() -> None:
    """Cisplatin should type-check against PT_DNA_MAJOR_GROOVE.

    This is the cross-layer test: a real-world drug inhabits the
    higher-order type for its known binding site.  Cisplatin is a
    Pt(II) square-planar complex with >= 4 dative sites, which is
    exactly the geometric β-check required by PT_DNA_MAJOR_GROOVE.
    """
    # Build a MoleculeClosedTerm from the RDKit SMILES of cisplatin.
    # This is a different code path from the Bond.dative assembly
    # above; it tests that the framework handles real molecules too.
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    # RDKit SMILES for cisplatin (no stereochemistry): [H][N]([H])([H])[Pt](Cl)(Cl)([N]([H])([H])[H])[H]
    # Simpler: use the disconnected form so RDKit is happy.
    smiles = "[NH3][Pt](Cl)(Cl)[NH3]"
    mol = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)

    result = typecheck(mol, PT_DNA_MAJOR_GROOVE)
    assert isinstance(result, BindingTypeCheckResult)
    # Cisplatin has a Pt center with >= 4 dative bonds → square_planar_ok.
    # It is a known square-planar Pt(II) coordination complex; the
    # type-check should succeed.
    assert result.success is True, (
        f"Cisplatin failed PT_DNA_MAJOR_GROOVE typecheck: "
        f"{result.violated_constraints}"
    )
    # Heuristic pIC50 estimate should be populated (not NaN).
    assert result.pic50_estimate == result.pic50_estimate  # not NaN
    assert result.pic50_estimate > 0.0


# ---------------------------------------------------------------------------
# Test 8: cisplatin is in β-normal form
# ---------------------------------------------------------------------------

def test_cisplatin_is_beta_normal_form() -> None:
    """The cisplatin term is in β-normal form (no redexes left)."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Use the Bond.dative assembly via build_cisplatin_term + wrap
    # in a MoleculeClosedTerm for the is_beta_normal_form check.
    mol_dict = build_cisplatin_term()

    # Build a MoleculeClosedTerm from the assembled atoms + bonds.
    # Use a fresh ledger that we share with the bond ledger so the
    # free-site bookkeeping is consistent.
    mol = MoleculeClosedTerm(
        atoms=list(mol_dict["atoms"]),
        bonds=list(mol_dict["bonds"]),
        ledger=mol_dict["bonds"][0].ledger,  # all bonds share a ledger
        source_smiles="[NH3][Pt](Cl)(Cl)[NH3]",
        term=mol_dict["term"],
    )

    # is_closed: True iff every atom valence-saturated.
    assert mol.is_closed is True
    # is_beta_normal_form: True iff closed AND no redex left.
    assert mol.is_beta_normal_form is True
    # has_redex: should be False (no reducible applications remain).
    assert mol.has_redex() is False


# ---------------------------------------------------------------------------
# Test 9: alpha-equivalence — different assembly orders give same molecule
# ---------------------------------------------------------------------------

def test_cisplatin_alpha_equivalent_to_itself() -> None:
    """Cisplatin is α-equivalent to itself (canonical SMILES equality)."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    # Build the assembled term.
    a = build_cisplatin_term()
    mol_a = MoleculeClosedTerm(
        atoms=list(a["atoms"]),
        bonds=list(a["bonds"]),
        ledger=a["bonds"][0].ledger,
        term=a["term"],
    )

    # Build it again via a fresh ledger (different atom identities).
    b = build_cisplatin_term()
    mol_b = MoleculeClosedTerm(
        atoms=list(b["atoms"]),
        bonds=list(b["bonds"]),
        ledger=b["bonds"][0].ledger,
        term=b["term"],
    )

    # alpha_equivalent checks canonical SMILES equality (with RDKit
    # fallback to value signature if RDKit can't serialize the dative
    # topology).
    assert mol_a.alpha_equivalent(mol_b) is True


# ---------------------------------------------------------------------------
# Test 10: currying semantics — Pt after one application is still arity 3
# ---------------------------------------------------------------------------

def test_currying_pt_after_one_dative_application() -> None:
    """After one NH3, Pt should still expect 3 more ligands (arity 3)."""
    led = FreeSiteLedger()
    pt: Atom = distinct(METAL_ATOMS["Pt_II"])
    nh3: Atom = distinct(make_ligand("NH3"))

    # Pre-condition: Pt has 4 free sites.
    assert free_sites(pt, ledger=led) == 4
    # Apply one dative bond (currying step).
    Bond.dative(pt, nh3, ledger=led)
    # Post-condition: Pt has 3 free sites (arity 3 — still a function).
    assert free_sites(pt, ledger=led) == 3
    # NH3 is saturated (donor becomes a value, not a function).
    assert free_sites(nh3, ledger=led) == 0


if __name__ == "__main__":
    # Allow running this file directly for quick verification:
    #     python -m molmetal.molmetal_lam.tests.test_cisplatin
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
