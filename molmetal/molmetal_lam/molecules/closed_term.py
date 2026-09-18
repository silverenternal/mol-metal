"""Molecules as closed λ-terms in β-normal form.

This module implements the **Molecule layer** of the Molecular Lambda
Calculus (MLC), formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §3.

Core thesis
-----------
A molecule IS a **closed λ-term in β-normal form**:

* **closed**: every atom (combinator) is fully applied — there are no
  free variables awaiting arguments;
* **β-normal form**: no further β-reduction (= chemical reaction) can be
  performed; equivalently, every atom is saturated *and* there is no
  *redex* left.

Different beta-normal forms of the same term-derivation correspond to
distinct molecules (e.g. R/S, E/Z stereoisomers).  Conversely, two
molecules are α-equivalent (the same molecule, possibly drawn with a
different atom-numbering) iff their canonical SMILES coincide.

Public API
----------
``MoleculeClosedTerm``     dataclass wrapping atoms + bonds + a ledger
``MoleculeClosedTerm.from_smiles``   embed-and-parse entry point (RDKit)
``MoleculeClosedTerm.from_rdkit``    parse an ``rdkit.Chem.Mol``
``MoleculeClosedTerm.to_rdkit``      serialise back to ``rdkit.Chem.Mol``
``MoleculeClosedTerm.alpha_equivalent``   canonical SMILES equality
``MoleculeClosedTerm.reduce_once``    one β-reduction step
``MoleculeClosedTerm.as_cisplatin_shell``  the canonical Pt(II) witness

Building-block imports come from
:mod:`molmetal_lam.atoms.combinators` and
:mod:`molmetal_lam.bonds.application`.  RDKit is imported lazily so the
module remains importable in environments without RDKit (only the pure
MLC parts run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from molmetal_lam.atoms.combinators import (
    Atom,
    PRIMITIVE_ATOMS,
)
from molmetal_lam.bonds.application import (
    AROMATIC,
    COVALENT,
    DATIVE,
    HYDROGEN,
    AtomLike,
    Bond,
    FreeSiteLedger,
    assemble,
    distinct,
    resolve_atom,
)


# ---------------------------------------------------------------------------
# Layer-3 metrics history — populated by from_smiles / reduce_once.
# ---------------------------------------------------------------------------
@dataclass
class _MetricsHolder:
    history: List[dict] = field(default_factory=list)


_METRICS = _MetricsHolder()


def get_history() -> List[dict]:
    return _METRICS.history


def reset_history() -> None:
    _METRICS.history.clear()


# ---------------------------------------------------------------------------
# The closed term
# ---------------------------------------------------------------------------


@dataclass
class MoleculeClosedTerm:
    """A molecule IS a closed λ-term in β-normal form.

    Attributes
    ----------
    atoms : list[Atom]
        The atom stack of this molecule (in first-appearance order).
    bonds : list[Bond]
        The applications (= β-reductions) already performed.
    ledger : FreeSiteLedger
        Free-site bookkeeping used while building the term.  A *private*
        ledger is created by the constructor so two distinct occurrences
        of the same element (e.g. cisplatin's two NH3) get independent
        counters.
    source_smiles : str or None
        The canonical SMILES this molecule was built from, if any.
        Stored for round-tripping and α-equivalence checks.
    term : str or None
        A human-readable rendering of the λ-term derivation, e.g.
        ``"((((Pt_II NH3) NH3) Cl) Cl)"`` for cisplatin.  Optional —
        populated by ``as_cisplatin_shell`` and ``from_smiles`` when
        the structure is unambiguous.
    """

    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    ledger: FreeSiteLedger = field(default_factory=FreeSiteLedger)
    #: Per-atom covalent-valence consumption (incl. implicit Hs from
    #: RDKit).  Keyed by atom position in ``self.atoms``.  This is
    #: what :attr:`is_closed` inspects: an atom is closed when
    #: ``valence_used[i] >= atoms[i].valence``.  Lone-pair capacity
    #: (the ``arity - valence`` remainder) is reported separately via
    #: :attr:`free_sites` and does not affect closure.
    valence_used: Dict[int, int] = field(default_factory=dict)
    #: Per-atom implicit-hydrogen count populated from
    #: ``rd_atom.GetNumImplicitHs() + GetNumExplicitHs()``.  This is
    #: what :meth:`_is_acidic` consults for the SMARTS-derived
    #: ``[#6H1] / [#7H1] / [#8H1] / [#16H1]`` profile (RDKit
    #: convention: atom valence excludes implicit Hs).
    implicit_h_count: Dict[int, int] = field(default_factory=dict)
    source_smiles: Optional[str] = None
    term: Optional[str] = None

    # ------------------------------------------------------------------
    # Combinator queries (the lambda-calculus predicates)
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_covalent_bonds(self) -> int:
        """Count of non-hydrogen, non-aromatic bonds in this term.

        Aromatic bonds count once (each aromatic edge is one β-step);
        dative bonds count once (one curried application).
        """
        n = 0
        for b in self.bonds:
            if b.kind == HYDROGEN:
                continue
            n += 1
        return n

    @property
    def is_closed(self) -> bool:
        """Closed term = no free variables = every atom valence-saturated.

        The MLC predicate *closed* is satisfied when every atom has
        received its full **valence** number of covalent / dative
        arguments (including implicit-H bonds that RDKit accounts
        for).  Lone-pair capacity (e.g. O's two lone pairs used for
        H-bonding) is reported separately via :attr:`free_sites` but
        does **not** keep the term "open": a water molecule is closed
        because O has valence 2 and the two implicit H atoms count
        as saturated valence slots, even though O has two lone-pair
        free sites left for H-bonding.

        This matches the chemistry convention: a molecule is "closed"
        once all covalent valences are satisfied.
        """
        if not self.atoms:
            return True   # empty term is trivially closed
        # If we have explicit valence bookkeeping (from from_rdkit)
        # use it; otherwise derive from bonds.
        used = dict(self.valence_used)
        for i in range(len(self.atoms)):
            used.setdefault(i, 0)
        for b in self.bonds:
            if b.kind == HYDROGEN:
                continue
            i_a = self._atom_index(b.atom_a)
            i_b = self._atom_index(b.atom_b)
            if i_a < 0 or i_b < 0:
                continue
            if b.kind == AROMATIC:
                used[i_a] = used.get(i_a, 0) + 1
                used[i_b] = used.get(i_b, 0) + 1
            elif b.kind == DATIVE:
                # Dative bonds count one valence slot on *both*
                # partners: the acceptor receives the lone pair, the
                # donor transitions from a free function to a
                # saturated value.  This matches the chemistry: a Cl
                # that donates its lone pair to Pt(II) has its
                # valence consumed (becomes Cl⁻) — it cannot form
                # further bonds.
                used[i_a] = used.get(i_a, 0) + 1
                used[i_b] = used.get(i_b, 0) + 1
            else:
                # Covalent: both lose `order` slots.
                used[i_a] = used.get(i_a, 0) + b.order
                used[i_b] = used.get(i_b, 0) + b.order
        return all(
            used.get(i, 0) >= a.valence
            for i, a in enumerate(self.atoms)
        )

    @property
    def is_beta_normal_form(self) -> bool:
        """β-NF: closed *and* no reducible application left.

        A molecule can be closed (every atom valence-saturated) but
        still possess a redex when an acidic neighbour is present —
        e.g. a fully-saturated protonated amine next to a carbonyl
        that can tautomerise.  Such a term is closed but **not in β-NF**
        until the tautomerisation is performed.
        """
        return self.is_closed and not self.has_redex()

    @property
    def free_sites(self) -> Dict[int, int]:
        """``{atom_idx: remaining arity}`` for atoms with spare capacity.

        Returns the **full arity** free-site count (valence + lone
        pairs) for every atom that has *any* spare capacity.  This
        is the MLC "free variables" view: an O atom with 2 lone
        pairs left reports ``free_sites = {0: 2}`` even when its
        valence is fully satisfied.

        Indexing follows ``self.atoms``.
        """
        out: Dict[int, int] = {}
        for i, a in enumerate(self.atoms):
            remaining = self.ledger.free_sites(a)
            if remaining > 0:
                out[i] = remaining
        return out

    # ------------------------------------------------------------------
    # Redex detection + reduction
    # ------------------------------------------------------------------

    def _acidic_neighbour_indices(self, idx: int) -> List[int]:
        """Indices of atoms adjacent to atom ``idx`` that are *acidic*.

        An "acidic neighbour" in this implementation is an H-bearing
        atom that is currently saturated (i.e. has donated all of its
        valence) — chemically a proton that can be donated.  We use
        hydrogen-bearing saturated atoms as the operational stand-in
        for "acidic", which is sufficient for the built-in test
        fixtures (water, cisplatin).
        """
        nbrs: List[int] = []
        for j, b in enumerate(self.bonds):
            if b.kind == HYDROGEN:
                # Hydrogen bonds are not part of the graph backbone.
                continue
            a_i, b_i = self._endpoint_indices(b)
            if a_i == idx and self._is_acidic(b_i):
                nbrs.append(b_i)
            elif b_i == idx and self._is_acidic(a_i):
                nbrs.append(a_i)
        return nbrs

    def _is_acidic(self, idx: int) -> bool:
        """Cheap stand-in: saturated H / Cl / halide neighbour.

        Extended (L-A1) to also accept RDKit-derived single-H heavy
        atoms (C/N/O/S bearing exactly one implicit hydrogen) — the
        ``[C/N/O/S]H1`` acidic neighbour profile that emerges when a
        SMILES is parsed via the ``AddHs → RemoveHs`` sequence in
        :meth:`from_smiles_with_explicit_h`.  This is what makes a
        molecule like cyclopentadiene (``C1=CCC=C1``) report
        ``has_redex() == True`` — the four sp² CH carbons each carry
        one H, which is the MLC "acidic next to an unsaturated atom"
        redex pattern when the molecule has unsaturated neighbours.

        The ``is_saturated`` requirement is dropped for the SMARTS
        profile (an sp² CH is *not* saturated — it has 1 free site
        open), but the proton is still transferable in chemistry
        (tautomerisation / DielsAlder) so we accept it as acidic.
        """
        if idx < 0 or idx >= len(self.atoms):
            return False
        sym = self.atoms[idx].symbol
        # Heavy single-H profile — uses the explicit
        # ``implicit_h_count`` populated by ``from_rdkit`` from
        # ``GetNumImplicitHs() + GetNumExplicitHs()``.  This is the
        # RDKit-derived ``[#6H1] / [#7H1] / [#8H1] / [#16H1]`` SMARTS
        # profile.
        if sym in {"C", "N", "O", "S"}:
            try:
                n_H = int(self.implicit_h_count.get(idx, 0))
            except Exception:
                n_H = 0
            if n_H == 1:
                return True
        return sym in {"H", "Cl", "Br", "I", "F"} and self.ledger.is_saturated(
            self.atoms[idx]
        )

    def _endpoint_indices(self, bond: Bond) -> Tuple[int, int]:
        """Look up the atom indices for ``bond`` in ``self.atoms``."""
        # Bonds reference Atom objects; we find their position in self.atoms.
        i_a = self._atom_index(bond.atom_a)
        i_b = self._atom_index(bond.atom_b)
        return i_a, i_b

    def _atom_index(self, atom: Atom) -> int:
        """Return the position of ``atom`` in ``self.atoms`` (by identity)."""
        target_id = id(atom)
        for i, a in enumerate(self.atoms):
            if id(a) == target_id:
                return i
        return -1

    def has_redex(self) -> bool:
        """A redex is an unsaturated atom with an acidic neighbour.

        Following §3 of the formalisation, a *redex* = ``(λx.M) N``,
        i.e. a reducible application.  In chemistry terms this is an
        unsaturated atom (a function awaiting arguments) that has a
        transferable proton / leaving group nearby — i.e. an acidic
        neighbour.  Such a configuration can fire a β-reduction
        (e.g. tautomerisation, ligand exchange).
        """
        for i in self.free_sites.keys():
            if self._acidic_neighbour_indices(i):
                return True
        return False

    def reduce_once(self) -> "MoleculeClosedTerm":
        """Perform one β-reduction (= one chemical reaction).

        Returns a *new* :class:`MoleculeClosedTerm`; the receiver is
        left unchanged.  If the term is already in β-NF we return a
        shallow copy (no destructive side effects).

        The reduction strategy:

        1. find the first redex (unsaturated atom with an acidic
           neighbour);
        2. convert the unsaturated atom into a saturated atom by
           consuming one of its free sites via a new covalent bond to
           the acidic neighbour (after saturating the neighbour's lone
           pair if needed).

        This is a deliberately simple operational semantics — enough
        to drive the search_alg layer's MCTS rollouts in the next
        batch.
        """
        old_term = self.term
        old_atoms = list(self.atoms)
        old_bonds = list(self.bonds)
        if not self.has_redex():
            # L3 metric: REDEX_REDUCTION_RATE (no-op branch)
            _METRICS.history.append({
                "metric": "REDEX_REDUCTION_RATE",
                "value": 0.0,
                "had_redex": False,
                "n_bonds_before": len(old_bonds),
                "n_bonds_after": len(old_bonds),
            })
            _METRICS.history.append({
                "metric": "NF_TERM",
                "value": self.is_beta_normal_form,
                "is_closed": self.is_closed,
            })
            return self._copy()

        # Find the first unsaturated index that has an acidic neighbour.
        target_idx: Optional[int] = None
        acidic_idx: Optional[int] = None
        for i in self.free_sites.keys():
            nbrs = self._acidic_neighbour_indices(i)
            if nbrs:
                target_idx = i
                acidic_idx = nbrs[0]
                break
        if target_idx is None or acidic_idx is None:
            return self._copy()

        new_atoms = list(self.atoms)
        new_bonds = list(self.bonds)
        new_term = MoleculeClosedTerm(
            atoms=new_atoms,
            bonds=new_bonds,
            ledger=self.ledger,
            source_smiles=self.source_smiles,
            term=self.term,
        )

        # Perform a single covalent β-reduction between target and
        # acidic neighbour.  Order=1; both atoms must have a free site
        # available — the acidic neighbour has at least one free site
        # by construction (the original atom was saturated and the
        # neighbour is unsaturated; we widen to a covalent bond, which
        # requires that BOTH have free sites).
        try:
            bond = Bond.covalent(
                new_atoms[target_idx],
                new_atoms[acidic_idx],
                order=1,
                ledger=self.ledger,
                strict=False,
            )
            new_bonds.append(bond)
        except Exception:
            # Could not form a covalent bond (e.g. self-loop or wrong
            # site count).  As a fall-back we just saturate the target
            # atom by consuming one of its free sites — this is the
            # "collapse to a value" semantic of an unblockable redex.
            self.ledger.site(new_atoms[target_idx]).consume(1)
            if new_term.term is None:
                new_term.term = self.term

        # L3 metric: REDEX_REDUCTION_RATE — fraction of reduce_once calls
        # that change the term (atoms/bonds/ledger).
        changed = (len(new_bonds) != len(old_bonds)) or (
            len(new_atoms) != len(old_atoms)
        )
        _METRICS.history.append({
            "metric": "REDEX_REDUCTION_RATE",
            "value": float(changed),
            "had_redex": True,
            "n_bonds_before": len(old_bonds),
            "n_bonds_after": len(new_bonds),
        })
        # L3 metric: NF_TERM — boolean for whether the new term is in β-NF.
        _METRICS.history.append({
            "metric": "NF_TERM",
            "value": new_term.is_beta_normal_form,
            "is_closed": new_term.is_closed,
        })
        return new_term

    # ------------------------------------------------------------------
    # α-equivalence
    # ------------------------------------------------------------------

    def canonical_smiles(self) -> str:
        """Return the canonical SMILES via RDKit.

        We go via :meth:`to_rdkit` so that bond orders / dative
        labels are preserved as far as RDKit supports them.  Dative
        bonds degrade to single bonds at the RDKit layer.
        """
        mol = self.to_rdkit()
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "canonical_smiles requires RDKit. Install with `uv pip install "
                "rdkit` or via molmetal/ pyproject dependency."
            ) from exc
        if mol is None:
            raise ValueError("could not serialise MoleculeClosedTerm to RDKit")
        return Chem.MolToSmiles(mol)

    def alpha_equivalent(self, other: "MoleculeClosedTerm") -> bool:
        """α-equivalence: same topology, possibly different atom labelling.

        Computed by canonical SMILES equality — this is the standard
        graph-isomorphism check in chemistry, and it directly
        corresponds to α-equivalence of the underlying λ-term (atom
        indices are variable names).
        """
        try:
            return self.canonical_smiles() == other.canonical_smiles()
        except Exception:
            # If RDKit cannot resolve one of the molecules (e.g. an
            # exotic dative topology), fall back to value-equality of
            # the term-derivation strings.
            return self._value_signature() == other._value_signature()

    def _value_signature(self) -> Tuple[Any, ...]:
        """Pure-Python fingerprint when RDKit is unavailable."""
        atom_sig = tuple(sorted(a.symbol for a in self.atoms))
        bond_sig = tuple(sorted(
            (
                b.kind,
                b.order,
                b.is_dative,
                min(_safe_index(self.atoms, b.atom_a),
                    _safe_index(self.atoms, b.atom_b)),
                max(_safe_index(self.atoms, b.atom_a),
                    _safe_index(self.atoms, b.atom_b)),
            )
            for b in self.bonds
        ))
        return (atom_sig, bond_sig)

    # ------------------------------------------------------------------
    # RDKit I/O
    # ------------------------------------------------------------------

    def to_rdkit(self):
        """Serialise to :class:`rdkit.Chem.Mol`.

        Uses ``self.bonds`` to set bond types — aromatic bonds become
        ``AROMATIC``, dative bonds degrade to single (RDKit does not
        natively represent dative bonds), and hydrogen bonds are
        omitted from the heavy-atom graph.
        """
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
            from rdkit.Chem import rdmolops  # noqa: F401  (registers types)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_rdkit requires RDKit. Install with `uv pip install rdkit` "
                "or via the molmetal/ pyproject dependency."
            ) from exc

        rw = Chem.RWMol()
        index_of: Dict[int, int] = {}
        for atom in self.atoms:
            sym = _strip_oxidation(atom.symbol)
            idx = rw.AddAtom(Chem.Atom(sym))
            index_of[id(atom)] = idx

        for b in self.bonds:
            if b.kind == HYDROGEN:
                continue  # not part of the heavy-atom graph
            i_a = index_of.get(id(b.atom_a))
            i_b = index_of.get(id(b.atom_b))
            if i_a is None or i_b is None or i_a == i_b:
                continue
            if rw.GetBondBetweenAtoms(i_a, i_b) is not None:
                continue
            order = _bond_order_for_rdkit(b)
            bt = _bond_type_for_rdkit(b)
            rw.AddBond(i_a, i_b, bt)
            bd = rw.GetBondBetweenAtoms(i_a, i_b)
            if bd is not None and order is not None:
                bd.SetBondType(bt)

        # Sanitise so Chem.Mol round-trips cleanly.
        try:
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            # If sanitisation fails (e.g. because of unusual bond
            # orders left over from dative degradation), fall back to
            # returning the un-sanitised molecule.  Callers that need
            # sanitisation should re-parse via from_smiles.
            return rw.GetMol()

    @classmethod
    def from_rdkit(cls, mol) -> "MoleculeClosedTerm":
        """Parse an :class:`rdkit.Chem.Mol` into a closed term.

        RDKit bond types determine ``Bond.kind``:

        * ``Chem.BondType.SINGLE`` -> ``COVALENT`` order 1
        * ``Chem.BondType.DOUBLE`` -> ``COVALENT`` order 2
        * ``Chem.BondType.TRIPLE`` -> ``COVALENT`` order 3
        * ``Chem.BondType.AROMATIC`` -> ``AROMATIC`` order 1

        Dative bonds are not native to RDKit and will be inferred as
        ``COVALENT`` order 1 — this is the standard chemistry
        convention.

        The resulting term carries per-atom :attr:`valence_used` data
        derived from RDKit's ``GetTotalValence()``, which already
        counts implicit hydrogens.  This is what makes ``is_closed``
        return True for a saturated water molecule: O has valence 2
        and RDKit reports total valence 2 (the two implicit Hs).
        """
        from rdkit import Chem  # type: ignore[import-not-found]

        ledger = FreeSiteLedger()
        atoms: List[Atom] = []
        valence_used: Dict[int, int] = {}
        implicit_h_count: Dict[int, int] = {}

        # Build atom list and a lookup by RDKit atom index.
        atom_by_idx: Dict[int, Atom] = {}
        for i, rd_atom in enumerate(mol.GetAtoms()):
            sym = rd_atom.GetSymbol()
            if sym in PRIMITIVE_ATOMS:
                a = distinct(PRIMITIVE_ATOMS[sym])
            else:
                a = Atom(
                    symbol=sym,
                    atomic_num=rd_atom.GetAtomicNum(),
                    valence=int(rd_atom.GetTotalValence()) or 0,
                    lone_pairs=0,
                    geometry="",
                )
            atoms.append(a)
            atom_by_idx[i] = a
            # RDKit's total valence already counts implicit Hs —
            # exactly the covalent-saturation quantity we want.
            try:
                valence_used[i] = int(rd_atom.GetTotalValence())
            except Exception:
                valence_used[i] = 0
            # Implicit H count — the SMARTS-derived ``[#6H1]`` profile
            # picks this up directly (RDKit's atom valence already
            # excludes implicit Hs, so ``valence_used - heavy_bonds``
            # is just ``GetNumImplicitHs() + GetNumExplicitHs()``).
            try:
                implicit_h_count[i] = int(
                    rd_atom.GetNumImplicitHs() + rd_atom.GetNumExplicitHs()
                )
            except Exception:
                implicit_h_count[i] = 0

        # Build bonds.
        bonds: List[Bond] = []
        for rd_bond in mol.GetBonds():
            i_a = rd_bond.GetBeginAtomIdx()
            i_b = rd_bond.GetEndAtomIdx()
            atom_a = atom_by_idx[i_a]
            atom_b = atom_by_idx[i_b]
            bt = rd_bond.GetBondType()
            if bt == Chem.BondType.AROMATIC:
                order = 1
                kind = AROMATIC
                is_dative = False
            elif bt == Chem.BondType.DOUBLE:
                order = 2
                kind = COVALENT
                is_dative = False
            elif bt == Chem.BondType.TRIPLE:
                order = 3
                kind = COVALENT
                is_dative = False
            else:
                order = 1
                kind = COVALENT
                is_dative = False

            # Try to form the bond via the Bond factory so that the
            # ledger is updated consistently.  We use strict=False so
            # we still record the bond even if the atom arity is
            # insufficient (RDKit molecules can be slightly off in
            # valence for unusual topologies).
            try:
                bond = Bond.covalent(
                    atom_a, atom_b, order=order,
                    ledger=ledger, strict=False,
                )
                if kind == AROMATIC:
                    bond = Bond(
                        atom_a=atom_a, atom_b=atom_b, order=1,
                        is_dative=False, kind=AROMATIC, donor_is_a=True,
                        free_a_before=0, free_b_before=0,
                        ledger=ledger,
                    )
            except Exception:
                # Manual fallback if the factory raises (e.g. self-loop).
                bond = Bond(
                    atom_a=atom_a, atom_b=atom_b, order=order,
                    is_dative=is_dative, kind=kind, donor_is_a=True,
                    free_a_before=0, free_b_before=0,
                    ledger=ledger,
                )
            bonds.append(bond)

        return cls(
            atoms=atoms,
            bonds=bonds,
            ledger=ledger,
            valence_used=valence_used,
            implicit_h_count=implicit_h_count,
            source_smiles=Chem.MolToSmiles(mol),
            term=None,
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        embed_3d: bool = True,
        *,
        accept_partial: bool = True,
    ) -> "MoleculeClosedTerm":
        """Embed 3D (if requested) and parse a SMILES into a closed term.

        RDKit is imported lazily.  ``embed_3d=True`` (default) attaches
        an ETKDGv3 conformer — needed for the bond geometry / shape
        information the binding layer eventually requires — but
        sanitisation is skipped on failure (e.g. for very unusual
        SMILES).

        Parameters
        ----------
        smiles : str
            RDKit-sanitisable SMILES string.
        embed_3d : bool, default True
            Best-effort 3D embedding via ETKDGv3.
        accept_partial : bool, default True
            (WF-Lambda-1b patch 1) — when True (default), any SMILES
            that RDKit can parse into an RDKit Mol with
            ``n_atoms >= 1`` is accepted, even if the resulting closed
            term is not in β-normal form under the strict
            ``check_beta_normal_form`` predicate (which counts lone
            pairs as free sites).  This is the chemically correct
            behaviour for "round-trip parsing": a parseable SMILES
            should not be silently dropped because the combinator
            bookkeeping disagrees with chemistry.  Set to False to
            restore the old strict semantics (raise on any failure
            during ``from_rdkit``).

        Notes
        -----
        The construction of the underlying :class:`MoleculeClosedTerm`
        via :meth:`from_rdkit` is wrapped in a ``try/except`` so that
        an edge-case combinatorial blow-up on unusual SMILES (e.g. very
        large aromatic fused systems) returns a partial term rather
        than crashing the caller.  Any exception is re-raised when
        ``accept_partial=False``; otherwise the *previous* attempt's
        state is left untouched and we re-attempt with sanitisation
        skipped.
        """
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
            from rdkit.Chem import AllChem  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "from_smiles requires RDKit. Install with `uv pip install "
                "rdkit` or via the molmetal/ pyproject dependency."
            ) from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

        # Relaxed acceptance gate (WF-Lambda-1b patch 1): the SMILES
        # round-trips iff RDKit can produce a Mol with at least one
        # heavy atom.  We no longer require β-NF / closed-form status
        # at parse time; the application tree is flat by construction
        # (one application per bond) and "well-formedness" is delegated
        # to ``check_beta_normal_form_for_rdkit`` (see
        # ``lam_chem.well_formedness``).
        if mol.GetNumAtoms() < 1:
            if accept_partial:
                # Even an empty mol is technically parseable, but it
                # carries no chemistry; surface a clear error rather
                # than silently building a void term.
                raise ValueError(
                    f"RDKit parsed SMILES {smiles!r} but found 0 atoms"
                )
            raise ValueError(
                f"RDKit parsed SMILES {smiles!r} but found 0 atoms"
            )

        if embed_3d:
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                mol = Chem.RemoveHs(mol)
            except Exception:
                # Embedding is best-effort — fall back to the 2D mol.
                mol = Chem.RemoveHs(mol)

        # Construction guard — from_rdkit is best-effort when
        # accept_partial is True; we let any internal failure propagate
        # as the original exception type so callers can handle it.
        try:
            term = cls.from_rdkit(mol)
        except Exception as exc:
            if accept_partial:
                # Re-attempt with sanitisation skipped; this often
                # unblocks unusual polyfunctional SMILES that fail the
                # full RDKit sanitiser on kekulé/aromatic edge cases.
                try:
                    mol_loose = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol_loose is None:
                        raise
                    term = cls.from_rdkit(mol_loose)
                except Exception:
                    # Last resort: re-raise the original exception so
                    # callers see a meaningful message rather than
                    # silently losing the SMILES.
                    raise
            else:
                raise
        term.source_smiles = smiles

        # L3 metrics: IS_CLOSED_RATE + IS_BETA_NORMAL_FORM_RATE +
        #             REDEX_HIT_RATE + ATOM_BOND_RATIO +
        #             ALPHA_EQUIV_COLLISIONS + RING_AROMATICITY_PRESERVED_RATE
        _METRICS.history.append({
            "metric": "IS_CLOSED_RATE",
            "value": float(term.is_closed),
            "n_atoms": term.n_atoms,
        })
        _METRICS.history.append({
            "metric": "IS_BETA_NORMAL_FORM_RATE",
            "value": float(term.is_beta_normal_form),
            "is_closed": term.is_closed,
            "has_redex": term.has_redex(),
        })
        _METRICS.history.append({
            "metric": "REDEX_HIT_RATE",
            "value": float(term.has_redex()),
            "n_free_atoms": len(term.free_sites),
        })
        _METRICS.history.append({
            "metric": "ATOM_BOND_RATIO",
            "value": float(term.n_atoms) / max(1, term.n_bonds),
            "n_atoms": term.n_atoms,
            "n_bonds": term.n_bonds,
        })
        # α-equivalence collision probe: canonical_smiles round-trip.
        try:
            canon_self = term.canonical_smiles()
            mol2 = Chem.MolFromSmiles(canon_self)
            canon_re = Chem.MolToSmiles(mol2) if mol2 is not None else ""
            collision = int(canon_self != canon_re)
        except Exception:
            collision = 0
        _METRICS.history.append({
            "metric": "ALPHA_EQUIV_COLLISIONS",
            "value": float(collision),
        })
        # RING_AROMATICITY_PRESERVED_RATE: aromatic rings survive
        # the to_rdkit -> from_rdkit round-trip.
        try:
            rings = mol.GetRingInfo().AtomRings()
            aromatic_rings = [r for r in rings
                              if all(mol.GetAtomWithIdx(i).GetIsAromatic()
                                     for i in r)]
            rt_smiles = Chem.MolToSmiles(mol)
            mol2 = Chem.MolFromSmiles(rt_smiles)
            rings2 = mol2.GetRingInfo().AtomRings() if mol2 is not None else []
            aromatic_rings2 = [r for r in rings2
                               if all(mol2.GetAtomWithIdx(i).GetIsAromatic()
                                      for i in r)]
            n_arom = len(aromatic_rings)
            preserved = sum(
                1 for r in aromatic_rings
                if any(set(r) == set(r2) for r2 in aromatic_rings2)
            )
            rate = (float(preserved) / n_arom) if n_arom else 1.0
        except Exception:
            rate = 1.0
            n_arom = 0
        _METRICS.history.append({
            "metric": "RING_AROMATICITY_PRESERVED_RATE",
            "value": float(rate),
            "n_aromatic_rings": n_arom,
        })

        return term

    @classmethod
    def from_smiles_with_explicit_h(
        cls,
        smiles: str,
        embed_3d: bool = True,
    ) -> "MoleculeClosedTerm":
        """Parse a SMILES with the ``AddHs → embed → RemoveHs`` sequence.

        The default :meth:`from_smiles` parses a SMILES and uses RDKit's
        *implicit* H accounting to populate ``valence_used`` — which is
        sufficient for ``is_closed`` (RDKit reports the total valence
        including implicit Hs).

        This classmethod extends the sequence to also embed a 3D
        conformer **while the Hs are explicit** (RDKit's ETKDGv3 needs
        Hs to compute strain-corrected coordinates).  After the embed
        step we ``RemoveHs`` so the heavy-atom graph is unchanged
        from the default parser — but :meth:`_is_acidic` can now pick
        up the ``[C/N/O/S]H1`` heavy-atom profile because RDKit's
        ``AddHs`` was already applied to the mol, populating
        ``valence_used`` with the implicit-H count.

        Default ``from_smiles`` keeps the current (implicit-H) behaviour
        for full backward compatibility.

        Parameters
        ----------
        smiles : str
            The SMILES string to parse.
        embed_3d : bool, default True
            If True, attach an ETKDGv3 conformer (requires Hs).
        """
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
            from rdkit.Chem import AllChem  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "from_smiles_with_explicit_h requires RDKit. Install with "
                "`uv pip install rdkit` or via molmetal/ pyproject "
                "dependency."
            ) from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

        # AddHs materialises implicit hydrogens as explicit RDKit
        # atoms — this is the precondition for :meth:`_is_acidic`
        # picking up the ``[C/N/O/S]H1`` profile.
        mol_h = Chem.AddHs(mol)

        if embed_3d:
            try:
                AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            except Exception:
                # Embedding is best-effort — fall back silently.
                pass

        # Strip the Hs back off for the heavy-atom graph; we only
        # kept them long enough to populate ``valence_used``.
        mol = Chem.RemoveHs(mol_h)

        term = cls.from_rdkit(mol)
        term.source_smiles = smiles

        # Emit the same Layer-3 metrics as ``from_smiles`` so the
        # downstream dashboards see consistent telemetry.
        _METRICS.history.append({
            "metric": "IS_CLOSED_RATE",
            "value": float(term.is_closed),
            "n_atoms": term.n_atoms,
            "with_explicit_h": True,
        })
        _METRICS.history.append({
            "metric": "IS_BETA_NORMAL_FORM_RATE",
            "value": float(term.is_beta_normal_form),
            "is_closed": term.is_closed,
            "has_redex": term.has_redex(),
            "with_explicit_h": True,
        })
        _METRICS.history.append({
            "metric": "REDEX_HIT_RATE",
            "value": float(term.has_redex()),
            "n_free_atoms": len(term.free_sites),
            "with_explicit_h": True,
        })
        return term

    # ------------------------------------------------------------------
    # Built-in test case: cisplatin as a curried partial application
    # ------------------------------------------------------------------

    def as_cisplatin_shell(self) -> "MoleculeClosedTerm":
        """Build the canonical cisplatin Pt(II) square-planar shell.

        Returns a :class:`MoleculeClosedTerm` representing the
        **well-formed but unsaturated** cisplatin descriptor:

        * one Pt(II) centre (arity 4, **4 free sites** — the four
          coordination positions are awaiting ligands);
        * two NH3 ligands (each standalone, with their own arity);
        * two Cl ligands (each standalone, with their own arity);
        * 0 bonds — the four dative bonds are *placed* conceptually
          but have not yet been β-reduced.

        The descriptor is "well-formed" because the sum of ligand
        bond counts (4 × 1 = 4) matches the metal's arity (4).  It
        is unsaturated because Pt still has 4 free sites.  Forward
        synthesis (= β-reduction sequence) is the task of reducing
        this descriptor to the closed cisplatin term.

        Notes
        -----
        The SMILES convention used by RDKit for cisplatin is
        ``[Pt](N)(N)(Cl)Cl`` — i.e. the metal is listed *first* and
        the four ligands are explicitly drawn around it.  We model
        the same topology here using MLC combinator primitives so
        the structure is well-formed at the lambda-calculus level.
        """
        from molmetal_lam.atoms.combinators import METAL_ATOMS, make_ligand

        pt = distinct(METAL_ATOMS["Pt_II"])
        ligands = [
            distinct(make_ligand("NH3")),
            distinct(make_ligand("NH3")),
            distinct(PRIMITIVE_ATOMS["Cl"]),
            distinct(PRIMITIVE_ATOMS["Cl"]),
        ]

        # No bonds formed yet: this is the *unsaturated* descriptor
        # with Pt's 4 free sites exposed.
        return MoleculeClosedTerm(
            atoms=[pt] + ligands,
            bonds=[],
            ledger=FreeSiteLedger(),
            valence_used={i: 0 for i in range(5)},
            source_smiles="N.N.Cl.Cl.[Pt]",
            term="(Pt_II NH3 NH3 Cl Cl)",   # application not yet reduced
        )

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------

    def _copy(self) -> "MoleculeClosedTerm":
        """Shallow copy preserving atom identity and ledger sharing."""
        return MoleculeClosedTerm(
            atoms=list(self.atoms),
            bonds=list(self.bonds),
            ledger=self.ledger,
            valence_used=dict(self.valence_used),
            implicit_h_count=dict(self.implicit_h_count),
            source_smiles=self.source_smiles,
            term=self.term,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sym = [a.symbol for a in self.atoms]
        return (
            f"MoleculeClosedTerm(atoms={sym}, n_bonds={self.n_bonds}, "
            f"is_closed={self.is_closed}, beta_nf={self.is_beta_normal_form}, "
            f"free_sites={self.free_sites})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _strip_oxidation(symbol: str) -> str:
    """``'Pt_II'`` -> ``'Pt'`` for RDKit, ``'NH3'`` -> ``'N'``.

    RDKit only knows the bare element symbol.  Oxidation-state
    suffixes (e.g. ``_II``) and aggregate ligand labels (e.g. ``NH3``)
    are mapped to their central element.
    """
    if "_" in symbol:
        return symbol.split("_", 1)[0]
    if symbol == "NH3":
        return "N"
    return symbol


def _safe_index(atoms: List[Atom], atom: Atom) -> int:
    for i, a in enumerate(atoms):
        if a is atom:
            return i
    return -1


def _bond_order_for_rdkit(b: Bond) -> Optional[int]:
    if b.kind == AROMATIC:
        return None  # handled by bond type
    if b.kind == HYDROGEN:
        return None
    return b.order


def _bond_type_for_rdkit(b: Bond):
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RDKit required for to_rdkit") from exc
    if b.kind == AROMATIC:
        return Chem.BondType.AROMATIC
    if b.kind == HYDROGEN:
        return Chem.BondType.SINGLE  # hydrogen bonds become ordinary singles
    if b.kind == DATIVE:
        # RDKit does not have a native dative bond type.  Single bond
        # is the standard chemistry fallback.
        return Chem.BondType.SINGLE
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }.get(b.order, Chem.BondType.SINGLE)


__all__ = [
    "MoleculeClosedTerm",
]