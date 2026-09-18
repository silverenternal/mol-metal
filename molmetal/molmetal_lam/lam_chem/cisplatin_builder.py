"""Programmatic cisplatin construction — the canonical MLC witness.

This module builds cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl) as a closed
:class:`~molmetal_lam.molecules.closed_term.MoleculeClosedTerm`, using
explicit curried partial applications in a fixed order so the canonical
SMILES is reproducible.

The heavy-atom composition is::

    [Pt, N, N, Cl, Cl]      # NH3 contributes the N atom (H's are implicit)

Pt_II is a square-planar combinator of arity 4, so the four ligands are
applied one at a time (left-associated currying).  After the four
dative applications the metal is saturated and every donor is collapsed
to a value, leaving a closed lambda-term in beta-normal form.

Public API
----------
``build_cisplatin()``            build the canonical cisplatin term
"""

from __future__ import annotations

from typing import List

from molmetal_lam.atoms.combinators import (
    METAL_ATOMS,
    PRIMITIVE_ATOMS,
    make_ligand,
)
from molmetal_lam.bonds.application import (
    Bond,
    FreeSiteLedger,
    assemble,
    distinct,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

from molmetal_lam.lam_chem.well_formedness import (
    assert_well_formed,
    check_beta_normal_form,
)


# The canonical left-associated currying order for cisplatin.
# The intermediate terms after each step have arities 3, 2, 1, 0.
_CISPLATIN_LIGAND_SYMBOLS: List[str] = ["NH3", "NH3", "Cl", "Cl"]


def build_cisplatin() -> MoleculeClosedTerm:
    """Build Pt_II(NH3)(NH3)(Cl)(Cl) as a MoleculeClosedTerm.

    Returns
    -------
    MoleculeClosedTerm
        A closed lambda-term whose heavy atoms are [Pt, N, N, Cl, Cl]
        (the three implicit H atoms of each NH3 are folded into the
        nitrogen's valence — they do not appear as separate atom
        entries).  Every bond is dative; the metal has arity 4 and is
        exactly saturated after the four applications.

    The build sequence is fixed so the canonical SMILES output is
    deterministic across runs.  The lambda-term expression attached
    to ``term`` is the textbook::

        ((((Pt_II NH3) NH3) Cl) Cl)

    See Also
    --------
    :func:`molmetal_lam.bonds.application.cisplatin` — the same molecule
        returned as a plain ``dict`` (without MoleculeClosedTerm wrap).
    """
    # Private ledger so library singletons are untouched.
    ledger = FreeSiteLedger()

    # The Pt(II) centre — a curried 4-arity function.  ``distinct``
    # makes a value-equal but object-distinct copy so the ledger keys
    # this occurrence independently of any other.
    pt = distinct(METAL_ATOMS["Pt_II"])

    # Four ligand occurrences — distinct copies of the library singletons
    # so each occurrence carries its own free-site bookkeeping.
    ligands = [distinct(make_ligand(sym)) for sym in _CISPLATIN_LIGAND_SYMBOLS]
    nh3_a, nh3_b, cl_a, cl_b = ligands

    # Four curried partial applications (dative bonds).  Each call
    # consumes 1 free site from the metal and saturates the donor.
    # The order is fixed: NH3, NH3, Cl, Cl — giving the canonical
    # lambda expression ((((Pt_II NH3) NH3) Cl) Cl).
    bonds: List[Bond] = [
        Bond.dative(pt, nh3_a, ledger=ledger),   # (Pt_II NH3)
        Bond.dative(pt, nh3_b, ledger=ledger),   # ((Pt_II NH3) NH3)
        Bond.dative(pt, cl_a, ledger=ledger),    # (((Pt_II NH3) NH3) Cl)
        Bond.dative(pt, cl_b, ledger=ledger),    # ((((Pt_II NH3) NH3) Cl) Cl)
    ]

    # Lift the assembled bond list into a MoleculeClosedTerm so the
    # well-formedness / canonical-SMILES API is available.
    atoms = [pt] + ligands
    term = MoleculeClosedTerm(
        atoms=atoms,
        bonds=bonds,
        ledger=ledger,
        valence_used={i: 0 for i in range(len(atoms))},
        source_smiles="[NH3][Pt](Cl)(Cl)[NH3]",
        term="((((Pt_II NH3) NH3) Cl) Cl)",
    )

    # The full assemble() side-effect: emit L2 metrics.
    assemble(bonds, ledger=ledger)

    # Enforce the MLC well-formedness contract — this raises if any of
    # the three conditions (closed_term / arity_conservation /
    # beta_normal_form) fails.  It is exactly the "chemistry = lambda-
    # calculus" thesis made operational in code.
    assert_well_formed(term)
    # Double-belt: explicit β-NF check.
    assert check_beta_normal_form(term), (
        "build_cisplatin produced a term that is not in β-normal form"
    )
    return term


__all__ = ["build_cisplatin"]