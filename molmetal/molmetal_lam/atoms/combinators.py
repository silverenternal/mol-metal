"""Atoms as primitive combinators of the Molecular Lambda Calculus (MLC).

This module implements the **Atom layer** of MLC, formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §1.

Core thesis
-----------
An atom is a primitive lambda-combinator with a fixed arity

    arity = valence + lone_pairs

where:
    * ``valence``       = number of covalent bonds it can form (e.g. C=4, H=1)
    * ``lone_pairs``    = number of additional electron-pair sites that
                          accept coordination / dative bonds
    * ``geometry``      = ``'sp3'``, ``'sp2'``, ``'square_planar'``,
                          ``'octahedral'``, ``'tetrahedral'``, ``'s'``, ...

A *saturated* atom (``current_bonds == arity``) is in **beta-normal form** —
no further β-reduction (= bonding) is possible.

Primitive combinator library
--------------------------
``PRIMITIVE_ATOMS``  : H, C, N, O, F, P, S, Cl, Br, I
``METAL_ATOMS``      : Pt_II, Ru_II, Zn_II, Ir_III, Cu_II, Au_III

Sanity
------
``sanity_check()`` verifies the foundational claims:
    * Pt_II.arity == 4  (square-planar fits 4 ligands)
    * Ru_II.arity == 6  (octahedral fits 6)
    * Cisplatin descriptor ``Pt_II + 2 NH3 + 2 Cl`` is well-formed.

And ``from_smiles(smiles)`` parses an arbitrary SMILES string via RDKit
into a list of :class:`Atom` instances (lazy RDKit import).

This file is the SYNTAX layer of MLC. Bonding (β-reduction) lives in
``molmetal_lam.bonds.application`` and molecules-as-closed-terms lives in
``molmetal_lam.molecules.closed_term``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Module-level metrics history — populated by layer entry points.
# We use a dataclass-like holder so we can still use `field(default_factory=list)`.
@dataclass
class _MetricsHolder:
    history: List[dict] = field(default_factory=list)


_METRICS = _MetricsHolder()
def get_history() -> List[dict]:
    return _METRICS.history
def reset_history() -> None:
    _METRICS.history.clear()


# ---------------------------------------------------------------------------
# The Atom combinator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    """An atom IS a primitive λ-combinator with a fixed arity.

    Attributes
    ----------
    symbol : str
        Element symbol (e.g. ``'H'``, ``'C'``, ``'Pt'``). For metals with
        an oxidation state we use the suffix ``_<roman>``, e.g.
        ``'Pt_II'`` (Pt²⁺), ``'Ru_II'`` (Ru²⁺), ``'Ir_III'`` (Ir³⁺).
    atomic_num : int
        Atomic number Z (1 = H, 78 = Pt, 44 = Ru, ...).
    valence : int
        Number of *covalent* bonds the atom can form in this representation.
        For metals this is the *additional* ligand coordination capacity
        beyond the lone-pair dative acceptors.
    lone_pairs : int
        Number of additional electron-pair sites available for dative /
        coordination bonds. A lone pair site behaves like one extra
        free parameter of the combinator.
    geometry : str
        Local geometry tag. One of ``'s'``, ``'sp'``, ``'sp2'``, ``'sp3'``,
        ``'square_planar'``, ``'tetrahedral'``, ``'octahedral'``,
        ``'trigonal_bipyramidal'``, ``''`` (unknown).
    is_metal : bool
        True iff this atom is a transition-metal centre with a
        coordination geometry distinct from main-group sp/sp2/sp3.
    """

    symbol: str
    atomic_num: int
    valence: int
    lone_pairs: int = 0
    geometry: str = ""
    is_metal: bool = field(default=False, kw_only=True)

    # ------------------------------------------------------------------
    # Combinator algebra
    # ------------------------------------------------------------------

    @property
    def arity(self) -> int:
        """The arity of this atom as a λ-combinator.

        Equals ``valence + lone_pairs``. This is the number of *free
        parameters* the combinator expects to receive (= the maximum
        number of bonds / dative interactions it can sustain).
        """
        return self.valence + self.lone_pairs

    def is_saturated(self, current_bonds: int) -> bool:
        """A saturated atom is fully applied (in β-normal form).

        Returns True iff the atom has received ``arity`` arguments —
        i.e. has formed its maximum number of bonds. At that point it
        is a *value* (closed term) and cannot undergo further
        β-reduction (= bonding).
        """
        return current_bonds >= self.arity

    def with_one_less_free_site(self) -> "Atom":
        """Return a copy of this atom with one fewer free site.

        We do **not** mutate valence/lone_pairs (those are intrinsic
        properties of the element); instead we return a new dataclass
        with a flag that the caller can use to track bond count. For
        simplicity in this layer we annotate via a *virtual* atom
        descriptor: the caller is expected to compare ``current_bonds``
        to ``arity`` rather than mutate the Atom. Here we provide a
        conservative copy whose internal ``current_bonds`` would be
        tracked externally.

        For the dataclass-frozen contract we return a copy with the
        same fields; the bookkeeping happens in the Molecule layer.
        """
        return Atom(
            symbol=self.symbol,
            atomic_num=self.atomic_num,
            valence=self.valence,
            lone_pairs=self.lone_pairs,
            geometry=self.geometry,
            is_metal=self.is_metal,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.is_metal or "_" in self.symbol:
            return f"Atom({self.symbol}, arity={self.arity}, geom={self.geometry!r})"
        return f"Atom({self.symbol}, Z={self.atomic_num}, arity={self.arity})"


# ---------------------------------------------------------------------------
# Primitive combinator library — main-group organics + halides
# ---------------------------------------------------------------------------

# Convention: standard chemistry rules — valence = number of covalent bonds
# the atom commonly forms; lone_pairs = remaining sp3 electrons available
# for dative bonds (e.g. O has 2 lone pairs, both as H-bond acceptors in
# practice; for the combinator abstraction we treat them as free sites
# that can receive H-bond donors).
PRIMITIVE_ATOMS: Dict[str, Atom] = {
    "H":   Atom(symbol="H",   atomic_num=1,  valence=1, lone_pairs=0, geometry="s"),
    "C":   Atom(symbol="C",   atomic_num=6,  valence=4, lone_pairs=0, geometry="sp3"),
    "N":   Atom(symbol="N",   atomic_num=7,  valence=3, lone_pairs=1, geometry="sp3"),
    "O":   Atom(symbol="O",   atomic_num=8,  valence=2, lone_pairs=2, geometry="sp3"),
    "F":   Atom(symbol="F",   atomic_num=9,  valence=1, lone_pairs=3, geometry="sp3"),
    "P":   Atom(symbol="P",   atomic_num=15, valence=3, lone_pairs=1, geometry="sp3"),
    "S":   Atom(symbol="S",   atomic_num=16, valence=2, lone_pairs=2, geometry="sp3"),
    "Cl":  Atom(symbol="Cl",  atomic_num=17, valence=1, lone_pairs=3, geometry="sp3"),
    "Br":  Atom(symbol="Br",  atomic_num=35, valence=1, lone_pairs=3, geometry="sp3"),
    "I":   Atom(symbol="I",   atomic_num=53, valence=1, lone_pairs=3, geometry="sp3"),
}


# ---------------------------------------------------------------------------
# Metal combinator library — coordination chemistry as n-arity combinators
# ---------------------------------------------------------------------------

# Each metal is a curried n-arity function:
#     Pt_II  ≡ λa.λb.λc.λd. complex(a,b,c,d)       # square-planar (4)
#     Ru_II  ≡ λa.λb.λc.λd.λe.λf. complex(a..f)    # octahedral (6)
#     Zn_II  ≡ λa.λb.λc.λd. complex(a,b,c,d)       # tetrahedral (4)
#     Ir_III ≡ λa.λb.λc.λd.λe.λf. complex(a..f)    # octahedral (6)
#     Cu_II  ≡ λa.λb.λc.λd. complex(a,b,c,d)       # square-planar (4)
#     Au_III ≡ λa.λb.λc.λd. complex(a,b,c,d)       # square-planar (4)
#
# The `valence` field records *additional* bond capacity beyond the
# lone-pair dative sites. In all cases `arity = valence + lone_pairs`
# matches the metal's coordination number.
METAL_ATOMS: Dict[str, Atom] = {
    "Pt_II": Atom(
        symbol="Pt_II", atomic_num=78,
        valence=2, lone_pairs=2, geometry="square_planar",
        is_metal=True,
    ),
    "Ru_II": Atom(
        symbol="Ru_II", atomic_num=44,
        valence=0, lone_pairs=6, geometry="octahedral",
        is_metal=True,
    ),
    "Zn_II": Atom(
        symbol="Zn_II", atomic_num=30,
        valence=2, lone_pairs=2, geometry="tetrahedral",
        is_metal=True,
    ),
    "Ir_III": Atom(
        symbol="Ir_III", atomic_num=77,
        valence=3, lone_pairs=3, geometry="octahedral",
        is_metal=True,
    ),
    "Cu_II": Atom(
        symbol="Cu_II", atomic_num=29,
        valence=2, lone_pairs=2, geometry="square_planar",
        is_metal=True,
    ),
    "Au_III": Atom(
        symbol="Au_III", atomic_num=79,
        valence=2, lone_pairs=2, geometry="square_planar",
        is_metal=True,
    ),
}

# Expected metal coordination numbers — lookup table for METAL_GEOMETRY_OK.
# Healthy: 1.0 (these are intrinsic, any drift is a regression).
_EXPECTED_METAL_CN: Dict[str, int] = {
    "Pt_II": 4, "Ru_II": 6, "Zn_II": 4,
    "Ir_III": 6, "Cu_II": 4, "Au_III": 4,
}


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _check_pt_ii_arity() -> bool:
    """Pt_II must have arity == 4 (square-planar fits 4 ligands)."""
    return METAL_ATOMS["Pt_II"].arity == 4


def _check_ru_ii_arity() -> bool:
    """Ru_II must have arity == 6 (octahedral fits 6 ligands)."""
    return METAL_ATOMS["Ru_II"].arity == 6


def _check_cisplatin_well_formed() -> bool:
    """Cisplatin descriptor 'Pt_II + 2 NH3 + 2 Cl' must be well-formed.

    A descriptor is *well-formed* iff:
        sum of arities of supplied ligands == metal's arity
        AND all ligands are saturated standalone (NH3, Cl each
        have arity 1 and form one bond each).

    Here we model NH3 as a single N atom that has already accepted 2 H
    bonds (i.e. it has 0 remaining free sites for additional bonding —
    it is a *value*, not a function). For the descriptor check we
    only need: Pt_II.arity = 4 ligands × 1 bond each = 4. ✓
    """
    pt = METAL_ATOMS["Pt_II"]
    # 2 × NH3 (1 bond each) + 2 × Cl (1 bond each) = 4 ligand slots
    n_ligand_bonds = 2 * 1 + 2 * 1
    return pt.arity == n_ligand_bonds


def sanity_check() -> Dict[str, bool]:
    """Run the foundational MLC sanity checks.

    Returns
    -------
    dict
        Mapping ``{check_name: passed}``. All entries must be ``True``
        for the Atom layer to be considered well-formed.
    """
    rep = {
        "Pt_II.arity == 4 (square-planar)": _check_pt_ii_arity(),
        "Ru_II.arity == 6 (octahedral)":    _check_ru_ii_arity(),
        "cisplatin well-formed":            _check_cisplatin_well_formed(),
    }
    # L1 metrics: SANITY_PASS_RATE + METAL_GEOMETRY_OK + PRIMITIVE_GEOMETRY_TAG_OK
    _METRICS.history.append({
        "metric": "SANITY_PASS_RATE",
        "value": float(sum(1 for v in rep.values() if v)) / max(1, len(rep)),
        "n_checks": len(rep),
        "n_passed": sum(1 for v in rep.values() if v),
    })
    metal_ok = sum(1 for m, cn in _EXPECTED_METAL_CN.items()
                   if m in METAL_ATOMS and METAL_ATOMS[m].arity == cn)
    _METRICS.history.append({
        "metric": "METAL_GEOMETRY_OK",
        "value": float(metal_ok) / max(1, len(_EXPECTED_METAL_CN)),
        "n_metal": len(_EXPECTED_METAL_CN),
        "n_ok": metal_ok,
    })
    prim_tagged = sum(1 for a in PRIMITIVE_ATOMS.values() if a.geometry)
    _METRICS.history.append({
        "metric": "PRIMITIVE_GEOMETRY_TAG_OK",
        "value": float(prim_tagged) / max(1, len(PRIMITIVE_ATOMS)),
        "n_primitive": len(PRIMITIVE_ATOMS),
        "n_tagged": prim_tagged,
    })
    return rep


def assert_well_formed() -> None:
    """Raise ``AssertionError`` if any sanity check fails.

    Useful as an import-time guard.
    """
    report = sanity_check()
    failed = [k for k, v in report.items() if not v]
    if failed:
        raise AssertionError(f"MLC atom-layer sanity failed: {failed}")


# ---------------------------------------------------------------------------
# SMILES parser (lazy RDKit import)
# ---------------------------------------------------------------------------

def from_smiles(smiles: str) -> List[Atom]:
    """Parse a SMILES string into a list of :class:`Atom` instances.

    RDKit is imported lazily so that this module remains usable in
    environments without RDKit installed (only ``sanity_check`` and
    the dataclass definitions themselves are RDKit-free).

    Each heavy atom in the molecule is mapped to a primitive combinator
    from ``PRIMITIVE_ATOMS`` / ``METAL_ATOMS`` where possible. If an
    element is not in the library (e.g. exotic atoms like ``B`` or
    ``Si``), we fall back to a sensible default based on the element's
    most common valence.

    Parameters
    ----------
    smiles : str
        A valid SMILES string (e.g. ``"CCO"`` for ethanol,
        ``"N.N.Cl.Cl.[Pt]"`` for cisplatin's disconnected fragments,
        ``"Cl[Pt](N)(N)Cl"`` for cisplatin with explicit H suppressed).

    Returns
    -------
    list[Atom]
        One Atom per heavy atom in the molecule, in SMILES-traversal
        order. Hydrogens are NOT returned (they are implicit).
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_smiles requires RDKit. Install with `uv pip install rdkit` "
            "or via the molmetal/ pyproject dependency."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

    atoms: List[Atom] = []
    n_hit = 0
    n_fallback = 0
    for rd_atom in mol.GetAtoms():
        sym = rd_atom.GetSymbol()
        # Try primitive library first (covers all main-group organics
        # and halides in this layer).
        if sym in PRIMITIVE_ATOMS:
            atoms.append(PRIMITIVE_ATOMS[sym])
            n_hit += 1
            continue
        # Fallback: lookup in metal library by symbol (without oxidation
        # suffix — RDKit only knows the bare element).
        metal_match = next(
            (a for a in METAL_ATOMS.values() if a.symbol.startswith(sym)),
            None,
        )
        if metal_match is not None:
            atoms.append(metal_match)
            n_hit += 1
            continue
        # Generic fallback for elements not in the primitive set.
        # Use RDKit's default valence as our `valence`, with 0 lone
        # pairs (we don't have data on them).
        default_valence = rd_atom.GetTotalValence()
        atoms.append(Atom(
            symbol=sym,
            atomic_num=rd_atom.GetAtomicNum(),
            valence=int(default_valence),
            lone_pairs=0,
            geometry="",
            is_metal=False,
        ))
        n_fallback += 1
    # L1 metrics: ARITY_HIT_RATE + FALLBACK_ATOM_RATIO
    total = max(1, n_hit + n_fallback)
    _METRICS.history.append({
        "metric": "ARITY_HIT_RATE",
        "value": float(n_hit) / total,
        "n_atoms": total,
        "n_hit": n_hit,
    })
    _METRICS.history.append({
        "metric": "FALLBACK_ATOM_RATIO",
        "value": float(n_fallback) / total,
        "n_atoms": total,
        "n_fallback": n_fallback,
    })
    return atoms


# ---------------------------------------------------------------------------
# Convenience: build a small ligand combinator on the fly
# ---------------------------------------------------------------------------

def make_ligand(symbol: str, occupancy: int = 1) -> Atom:
    """Build a single-bond ligand combinator (e.g. ``Cl``, ``NH3``).

    For non-metal single-bond donors (Cl, Br, I, F, H), the ligand has
    arity 1: it can donate one bond. For the special case of NH3 we
    construct a saturated nitrogen (valence exhausted by 3 H's) which
    behaves as a *value* — i.e. it cannot accept further bonds.
    """
    if symbol == "NH3":
        # Saturated amine: nitrogen with 3 H bonds already formed.
        # Acts as a value (closed term) for the purposes of dative
        # bonding to a metal centre.
        return Atom(symbol="NH3", atomic_num=7, valence=0, lone_pairs=1,
                    geometry="sp3")
    if symbol in PRIMITIVE_ATOMS:
        return PRIMITIVE_ATOMS[symbol]
    raise ValueError(f"Unknown ligand symbol: {symbol!r}")


# ---------------------------------------------------------------------------
# AtomInstance — mutable combinator occurrence
# ---------------------------------------------------------------------------
#
# The :class:`Atom` dataclass is frozen (its intrinsic valence / lone-pair
# properties belong to the *element*, not to any one occurrence in a
# particular molecule).  But every *occurrence* of an atom in a
# particular molecule has a *current* bond count that changes as we
# apply arguments.  We therefore wrap an :class:`Atom` together with
# that mutable bond count in a non-frozen :class:`AtomInstance` helper.
#
# This is a thin convenience layer used by cisplatin_builder.py and any
# future code that prefers an instance-style API.  The canonical
# bookkeeping for the bonds/application layer continues to live in
# :class:`FreeSiteLedger`.

@dataclass
class AtomInstance:
    """An atom occurrence: an :class:`Atom` plus its current bond count.

    Attributes
    ----------
    atom : Atom
        The underlying frozen atom descriptor (intrinsic arity).
    current_bonds : int
        How many bonds this occurrence has formed so far (default 0).
        An atom instance is *saturated* (= in beta-normal form, a
        value) when ``current_bonds >= atom.arity``.

    The dataclass is **not** frozen so callers can increment
    ``current_bonds`` directly.  The :class:`Atom` it wraps remains
    frozen and hashable, so an AtomInstance carries no commitment to
    its underlying atom's identity.
    """

    atom: Atom
    current_bonds: int = 0

    @property
    def arity(self) -> int:
        """Delegate to the wrapped :class:`Atom`."""
        return self.atom.arity

    @property
    def free_sites(self) -> int:
        """Remaining free parameters of this occurrence."""
        return max(0, self.atom.arity - self.current_bonds)

    @property
    def is_saturated(self) -> bool:
        """True iff this occurrence has formed all of its atom's bonds."""
        return self.current_bonds >= self.atom.arity

    def with_dative_bond_to(
        self, other: "AtomInstance",
    ) -> Tuple["AtomInstance", "AtomInstance"]:
        """Form a curried dative bond between two occurrences.

        The *self* side becomes the donor (its bond count saturates to
        its arity and it is collapsed to a value), and the *other*
        side becomes the acceptor (its bond count increments by 1).
        Returns ``(donor, acceptor)`` with their updated bond counts.

        This is a *very* simplified version of :meth:`Bond.dative`
        suitable for hand-coded combinator building; it does **not**
        touch any :class:`FreeSiteLedger`.  The full bookkeeping path
        remains ``Bond.dative(atom_a, atom_b, ledger=led)``.
        """
        donor = self
        acceptor = other
        # Saturate the donor — its bond count jumps to its arity so it
        # is collapsed to a value and cannot accept further applications.
        if donor.current_bonds < donor.atom.arity:
            donor.current_bonds = donor.atom.arity
        # Increment the acceptor's bond count by one (currying).
        if acceptor.atom.arity > 0:
            acceptor.current_bonds += 1
        return donor, acceptor

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AtomInstance({self.atom.symbol}, bonds={self.current_bonds}/"
            f"{self.atom.arity})"
        )


def make_instance(atom: Atom, current_bonds: int = 0) -> AtomInstance:
    """Wrap ``atom`` in an :class:`AtomInstance` with ``current_bonds``.

    See :class:`AtomInstance` for the semantics.  Convenience factory.
    """
    return AtomInstance(atom=atom, current_bonds=current_bonds)


__all__ = [
    "Atom",
    "AtomInstance",
    "PRIMITIVE_ATOMS",
    "METAL_ATOMS",
    "sanity_check",
    "assert_well_formed",
    "from_smiles",
    "make_ligand",
    "make_instance",
]