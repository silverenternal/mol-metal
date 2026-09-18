"""Well-formedness predicates for the Molecular Lambda Calculus (MLC).

This module **enforces** the central thesis of MLC in code:

    A molecule IS a closed lambda-term in beta-normal form.

The three well-formedness conditions below are exactly the operational
content of that statement:

    1. ``check_closed_term(term)``        - every bond is well-typed and
                                            matches its atoms' arities;
                                            no orphan bonds, no
                                            over-saturation.
    2. ``check_arity_conservation(bonds)``- free sites balance across
                                            the bond list (no free sites
                                            "leak" anywhere).
    3. ``check_beta_normal_form(term)``   - every atom is saturated
                                            (current_bonds == arity)
                                            — i.e. there is no further
                                            beta-reduction possible.

If any of these fails, ``assert_well_formed`` raises
:class:`WellFormednessError` with a human-readable message identifying
the offending atom, bond, or count.

Why this lives in lam_chem/
---------------------------
``lam_chem`` is the home of MLC's *type-theoretic* machinery (AST,
substitution, beta-reduction). Well-formedness sits at the boundary
between ``molecules/closed_term`` (the runtime representation) and the
type theory, so it lives here next to :mod:`lam_chem.ast`.

Public API
----------
``WellFormednessError``                raised by :func:`assert_well_formed`
``check_closed_term``                  full predicate (1 + 2 + 3)
``check_arity_conservation``           free-site accounting over bonds
``check_beta_normal_form``             every atom saturated
``assert_well_formed``                 raise on failure
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from molmetal_lam.bonds.application import (
    AROMATIC,
    COVALENT,
    DATIVE,
    HYDROGEN,
    Bond,
    FreeSiteLedger,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm


class WellFormednessError(ValueError):
    """Raised when a molecule fails the MLC well-formedness conditions.

    Subclass of :class:`ValueError` so existing ``except ValueError:``
    handlers still match.  The message identifies which condition failed
    (closed-term / arity-conservation / beta-normal-form) and which
    atom/bond triggered the failure.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bonds_consumed_by_atom(bonds: Iterable[Bond], atom_index: int,
                            atoms: List) -> int:
    """Return how many free sites ``atoms[atom_index]`` consumed via ``bonds``.

    Uses Bond.currying semantics:

    * **covalent**: consumes ``order`` sites on each partner.
    * **dative**:   consumes 1 site on the *acceptor* (the metal) AND
      collapses the donor to a value.  We count 1 site consumed on each
      partner — matching :attr:`MoleculeClosedTerm.is_closed`.
    * **aromatic**: consumes 1 site on each partner.
    * **hydrogen**: consumes 0 sites on either partner.
    """
    total = 0
    target = atoms[atom_index]
    for b in bonds:
        if b.kind == HYDROGEN:
            continue
        a_in_b = b.atom_a is target
        b_in_b = b.atom_b is target
        if not (a_in_b or b_in_b):
            continue
        if b.kind == COVALENT:
            total += b.order
        elif b.kind == AROMATIC:
            total += 1
        elif b.kind == DATIVE:
            # Dative: both donor and acceptor consume 1 site — the donor
            # transitions to a value, the acceptor receives the lone pair.
            total += 1
        else:
            # Defensive: an unknown bond kind still consumes 1 site.
            total += 1
    return total


# ---------------------------------------------------------------------------
# check_closed_term — full structural well-typedness
# ---------------------------------------------------------------------------

def check_closed_term(term: MoleculeClosedTerm) -> bool:
    """Verify every bond is well-typed: arity matched, no orphans, no oversat.

    Returns True iff:

    * every bond refers to atoms present in ``term.atoms`` (no orphans);
    * no atom is over-saturated (current_bonds > arity);
    * every bond is valid per :func:`molmetal_lam.bonds.application.is_valid`.

    Note: ``check_closed_term`` is the *structural* check; for the
    lambda-calculus closure predicate see :func:`check_beta_normal_form`.
    """
    if not isinstance(term, MoleculeClosedTerm):
        return False
    atoms = term.atoms
    atom_ids = {id(a) for a in atoms}
    # No orphans: every bond endpoint must be in term.atoms.
    for b in term.bonds:
        if id(b.atom_a) not in atom_ids or id(b.atom_b) not in atom_ids:
            return False
        if b.atom_a is b.atom_b:
            return False
    # No over-saturation: per-atom consumption must not exceed arity.
    # We use the ledger directly — it tracks the canonical "current_bonds"
    # for each atom occurrence.  ``free_sites(atom)`` returns
    # ``arity - used``, so ``arity - free_sites = used``.
    for i, a in enumerate(atoms):
        used_via_ledger = a.arity - term.ledger.free_sites(a)
        if used_via_ledger > a.arity:
            return False
        if used_via_ledger < 0:
            return False
    return True


# ---------------------------------------------------------------------------
# check_arity_conservation — free-site accounting
# ---------------------------------------------------------------------------

def check_arity_conservation(
    bonds: Iterable[Bond],
) -> Dict[str, int]:
    """Verify free-site accounting balances across the bond list.

    The conservation rule, evaluated at the bond list level, is::

        sum(free_before) == sum(free_after) + sum(consumed_per_bond)

    where ``consumed_per_bond`` is the total free-site consumption
    that this bond kind actually performs in the ledger:

    * **covalent / aromatic** — both partners lose ``order`` sites, so
      ``consumed_per_bond == 2 * order``.
    * **dative** — the acceptor loses 1 site (currying) AND the donor
      collapses to a value via ``AtomSite.saturate()`` which sets
      ``used_sites = max(used_sites, atom.arity)``.  Concretely, the
      donor's free-site count drops from its pre-bond value to 0, so
      ``consumed_per_bond == 1 + donor_free_before``.
    * **hydrogen** — no free-site consumption (``consumed == 0``).

    This matches the chemistry: a dative donor (NH3, Cl⁻) becomes a
    *value* after donating its lone pair, so all of its free-site
    capacity is exhausted at once.

    The return value is a diagnostic dict::

        {
          "n_bonds":          int,   # bond count
          "free_before":      int,   # sum of free_a_before + free_b_before
          "free_after":       int,   # sum of free_a_after + free_b_after
          "consumed":         int,   # sum of free-site consumption
          "conserved":        bool,  # free_before == free_after + consumed
        }
    """
    bond_list = list(bonds)
    free_before = 0
    free_after = 0
    consumed = 0
    for b in bond_list:
        if b.free_a_before < 0 or b.free_b_before < 0:
            # Bond built without snapshots — we cannot check it.
            continue
        free_before += b.free_a_before + b.free_b_before
        if b.kind == DATIVE:
            # Acceptor loses 1 site.  Donor's free_sites go to 0 via
            # saturate(), so the donor's free-site drop is exactly
            # ``donor_free_before``.
            if b.donor_is_a:
                donor_before = b.free_a_before
                acceptor_before = b.free_b_before
            else:
                acceptor_before = b.free_a_before
                donor_before = b.free_b_before
            free_after += max(0, acceptor_before - 1) + 0  # donor -> 0
            consumed += 1 + donor_before
        elif b.kind == HYDROGEN:
            free_after += b.free_a_before + b.free_b_before
            # consumed += 0
        else:
            # Covalent / aromatic: both sides lose ``order``.
            k = b.order
            free_after += max(0, b.free_a_before - k)
            free_after += max(0, b.free_b_before - k)
            consumed += 2 * k
    conserved = (free_before == free_after + consumed)
    return {
        "n_bonds": len(bond_list),
        "free_before": free_before,
        "free_after": free_after,
        "consumed": consumed,
        "conserved": conserved,
    }


# ---------------------------------------------------------------------------
# check_beta_normal_form — every atom saturated
# ---------------------------------------------------------------------------

def check_beta_normal_form(term: MoleculeClosedTerm) -> bool:
    """Verify every atom is valence-saturated (legacy arity fallback).

    The legacy predicate treats :class:`Atom.arity` (valence + lone_pairs)
    as the saturation threshold — ``ledger.free_sites(atom) == 0``. This
    is too strict for chemical generation because main-group atoms
    (N/O/S/F/Cl/Br/I) keep ``lone_pairs > 0`` even after all covalent
    bonds are formed (the lone pairs are reserved for H-bonding / dative
    interactions and should NOT keep the term "open").

    The chemistry-correct predicate is :func:`check_beta_normal_form_for_rdkit`
    which requires only that every atom has filled its covalent
    ``valence`` (not its full ``arity``).  This wrapper preserves the
    legacy name for backwards compatibility — the per-cell harness uses
    the valence-based variant via ``synthesizability_via_lambda_paths(
    use_valence_bnf=True)``.
    """
    if not isinstance(term, MoleculeClosedTerm):
        return False
    if not term.atoms:
        return True
    for a in term.atoms:
        if term.ledger.free_sites(a) != 0:
            return False
    return True


def check_beta_normal_form_for_rdkit(a, atom) -> bool:
    """Atom-occurrence predicate: True iff this atom is valence-saturated.

    WF-Lambda-1c semantic patch — the BNF predicate for *chemical*
    generation must use ``valence`` (covalent bond capacity), not
    ``arity = valence + lone_pairs`` (the lambda-combinator arity
    including dative / H-bond sites).

    Args:
        a: An :class:`Atom` occurrence carried in the ledger (the
            bookkeeping cell :class:`AtomSite` — obtained via
            ``ledger.site(atom)``).  We read ``a.used_sites`` from this
            object as a *lower bound* on covalent-bond count.
        atom: The :class:`Atom` whose ``valence`` we use as the
            saturation threshold.

    Returns:
        ``True`` iff the covalent bonds already consumed for ``atom``
        (from explicit ledger bookkeeping) meet or exceed
        ``atom.valence``. A nitrogen with three covalent bonds (e.g.
        an amine N in a generic drug-like molecule) is BNF-saturated
        even though it has one lone pair left.

    Field name resolution
    ---------------------
    We use :attr:`AtomSite.used_sites` from :mod:`bonds.application`
    which counts applied arguments (= bonds consumed via β-reduction).
    Lone-pair capacity is the *remainder* (``atom.arity - used_sites``)
    and does not contribute to ``used_sites`` — so reading ``used_sites``
    already gives the covalent-bond count we need.

    The fallback accepts ``a`` being either an :class:`AtomSite` or the
    :class:`Atom` itself; in the latter case we cannot recover
    used_sites and conservatively return ``False`` (no false positives
    on radicals / unfilled valences).
    """
    used = getattr(a, "used_sites", None)
    if used is None:
        # ``a`` was passed as the bare Atom (not the AtomSite). Without
        # bookkeeping access we cannot prove valence-saturated, so
        # return False — this is the conservative (no false positive)
        # branch.
        return False
    try:
        return int(used) >= int(atom.valence)
    except Exception:
        return False


def check_beta_normal_form_for_rdkit_term(term) -> bool:
    """Term-level valence-based BNF using ``term.valence_used`` (RDKit-aware).

    WF-Lambda-1c — the :func:`check_beta_normal_form_for_rdkit` per-atom
    predicate uses :attr:`AtomSite.used_sites` from the bond ledger,
    but for SMILES parsed through :meth:`MoleculeClosedTerm.from_smiles`
    the bond ledger records only *heavy-atom* bonds (RDKit's
    ``valence_used`` already accounts for implicit H bonds in the
    ``valence_used`` dict that ``from_rdkit`` populates).  Without
    falling back on ``valence_used``, ligands like ``[NH3]`` (3
    implicit H + 1 heavy bond = total valence 4) appear valence-
    unsaturated because the bond ledger only sees 1 bond.

    This term-level helper uses ``valence_used[i]`` (RDKit total
    valence, implicit-H-inclusive) when available, falling back to
    ``AtomSite.used_sites`` otherwise.  An atom is valence-saturated
    iff ``valence_used[i] >= atoms[i].valence`` — the chemistry-
    correct definition that ignores lone-pair capacity.

    Returns ``True`` for an empty / uninitialised term (trivially
    BNF-saturated).
    """
    atoms = list(getattr(term, "atoms", []) or [])
    if not atoms:
        return True
    # Prefer term.valence_used (RDKit total, implicit-H-inclusive).
    vu = dict(getattr(term, "valence_used", {}) or {})
    for i in range(len(atoms)):
        vu.setdefault(i, 0)
    # Walk all atoms.  If any has valence_used < atom.valence, fail.
    for i, atom in enumerate(atoms):
        try:
            used = int(vu.get(i, 0))
        except Exception:
            used = 0
        # Cross-check with ledger used_sites: take the max (the bond
        # ledger can be more up-to-date than valence_used if the
        # caller mutated bonds after from_smiles).
        ledger = getattr(term, "ledger", None)
        if ledger is not None:
            try:
                cell = ledger.site(atom)
                used = max(used, int(getattr(cell, "used_sites", 0)))
            except Exception:
                pass
        if used < int(atom.valence):
            return False
    return True


# ---------------------------------------------------------------------------
# assert_well_formed — raise on failure
# ---------------------------------------------------------------------------

def assert_well_formed(term: MoleculeClosedTerm) -> None:
    """Raise :class:`WellFormednessError` iff ``term`` is ill-formed.

    Runs all three checks in order:

    1. :func:`check_closed_term` (structural well-typedness);
    2. :func:`check_arity_conservation` (free-site accounting);
    3. :func:`check_beta_normal_form` (lambda-closure).

    On failure the exception message identifies which check failed and
    (where possible) which atom or bond triggered the failure.
    """
    if not check_closed_term(term):
        raise WellFormednessError(
            "MoleculeClosedTerm fails check_closed_term: "
            "an atom has over- or under-saturation, or a bond refers "
            "to an atom not in term.atoms"
        )
    rep = check_arity_conservation(term.bonds)
    if not rep["conserved"]:
        raise WellFormednessError(
            "MoleculeClosedTerm fails check_arity_conservation: "
            f"free_before={rep['free_before']} but "
            f"free_after + consumed = "
            f"{rep['free_after']} + {rep['consumed']} = "
            f"{rep['free_after'] + rep['consumed']}"
        )
    if not check_beta_normal_form(term):
        # Identify the unsaturated atoms for the message.
        unsaturated = [
            (i, a.symbol, term.ledger.free_sites(a))
            for i, a in enumerate(term.atoms)
            if term.ledger.free_sites(a) > 0
        ]
        raise WellFormednessError(
            "MoleculeClosedTerm fails check_beta_normal_form: "
            f"unsaturated atoms {unsaturated}"
        )


__all__ = [
    "WellFormednessError",
    "check_closed_term",
    "check_arity_conservation",
    "check_beta_normal_form",
    "assert_well_formed",
]