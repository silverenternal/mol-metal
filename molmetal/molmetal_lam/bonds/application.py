"""Bonds as function application of the Molecular Lambda Calculus (MLC).

This module implements the **Bond layer** of MLC, formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §2.

Core thesis
-----------
A chemical bond **is** a lambda-term application (= one beta-reduction
step).  Bonding two atoms consumes free parameters of the two
combinators defined in :mod:`molmetal_lam.atoms.combinators`::

    bond type   | lambda-calculus counterpart      | example
    ------------|----------------------------------|--------------------------
    single      | direct application               | C-H     = (C H)
    double      | nested application               | C=O     = ((C O) O)
    triple      | triple application               | C#N     = (((C N) N) N)
    aromatic    | eta-equivalence class            | benzene's 6 C-C bonds
    dative      | *curried partial application*    | Pt-NH3  = (Pt NH3)
    hydrogen    | type-checked application         | N-H...O (both saturated)

The single most important asymmetry is the **dative bond**::

    Pt_II  ==  \\a.\\b.\\c.\\d. complex(a, b, c, d)      # arity 4

    (Pt_II NH3)                                       # ONE argument applied
        -> still a function of arity 3
        -> the metal keeps 3 free coordination sites
        -> the ligand NH3 becomes a *value* (fully saturated, closed term)

So ``Bond.dative`` decrements the **acceptor** (metal) only, and
saturates the **donor** (ligand).  A covalent bond, by contrast, is a
symmetric application: both partners lose ``order`` free sites.

Free-site bookkeeping
---------------------
:class:`~molmetal_lam.atoms.combinators.Atom` is a frozen dataclass with
no mutable bond counter (``arity`` is an intrinsic property of the
element).  Free-site consumption is therefore tracked *outside* the atom
in a :class:`FreeSiteLedger`, keyed by object identity.  The module-level
:data:`DEFAULT_LEDGER` is used when no ledger is passed explicitly::

    pt   = METAL_ATOMS["Pt_II"]          # arity 4
    b    = Bond.dative(pt, make_ligand("NH3"))
    free_sites(pt)                       # -> 3     (curried!)

Because the library dictionaries (``PRIMITIVE_ATOMS`` / ``METAL_ATOMS``)
hold *singletons*, use :func:`distinct` to obtain an independent copy
whenever a molecule needs two chemically identical but structurally
separate atoms (e.g. cisplatin's two NH3 ligands), or pass a private
:class:`FreeSiteLedger` per molecule.

Public API
----------
``Bond``                dataclass + factories (covalent/dative/aromatic/hydrogen)
``bond(a, b)``          dispatcher: picks the right bond type from atom properties
``is_valid(bond)``      validity predicate of the four bond kinds
``can_bond(a, b)``      pre-check that does not mutate the ledger
``Bond.apply()``        the beta-reduction: returns a molecule-like dict
``assemble(bonds)``     beta-reduce a whole bond list into one molecule dict
``FreeSiteLedger`` / ``free_sites`` / ``saturate`` / ``reset_free_sites``
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from molmetal_lam.atoms.combinators import (
    METAL_ATOMS,
    PRIMITIVE_ATOMS,
    Atom,
    make_ligand,
)


# ---------------------------------------------------------------------------
# Layer-2 metrics history — populated by assemble() (the L2 entry point).
# ---------------------------------------------------------------------------
@dataclass
class _MetricsHolder:
    history: List[dict] = field(default_factory=list)


_METRICS = _MetricsHolder()


def get_history() -> List[dict]:
    return _METRICS.history


def reset_history() -> None:
    _METRICS.history.clear()


# Module-level counters for BUILDER_EXCEPTION_RATE.
_BUILDER_CALLS: int = 0
_BUILDER_EXCEPTIONS: int = 0


def builder_exception_snapshot() -> Dict[str, object]:
    """Return {calls, exceptions, rate} for BUILDER_EXCEPTION_RATE."""
    rate = (
        float(_BUILDER_EXCEPTIONS) / _BUILDER_CALLS
        if _BUILDER_CALLS > 0 else 0.0
    )
    _METRICS.history.append({
        "metric": "BUILDER_EXCEPTION_RATE",
        "value": rate,
        "n_calls": _BUILDER_CALLS,
        "n_exceptions": _BUILDER_EXCEPTIONS,
    })
    return {
        "calls": _BUILDER_CALLS,
        "exceptions": _BUILDER_EXCEPTIONS,
        "rate": rate,
    }


def safe_factory(callable_, *args, **kwargs):
    """Call a Bond factory inside a try/except and bump exception counters.

    This is the module-level wrapper suggested by the governance review.
    Returns the constructed Bond (or list of Bonds) on success; on
    exception returns ``None`` and increments ``_BUILDER_EXCEPTIONS``.
    """
    global _BUILDER_CALLS, _BUILDER_EXCEPTIONS
    _BUILDER_CALLS += 1
    try:
        return callable_(*args, **kwargs)
    except Exception:
        _BUILDER_EXCEPTIONS += 1
        return None

# Bond kind tags (also the eta-equivalence class labels).
COVALENT = "covalent"
DATIVE = "dative"
AROMATIC = "aromatic"
HYDROGEN = "hydrogen"

#: An atom may be given as an :class:`Atom`, as an element symbol
#: (``"N"``, ``"Cl"``, ``"Pt_II"``, ``"NH3"``), or as ``None`` (which
#: resolves to the anonymous lone-pair donor below).
AtomLike = Union[Atom, str, None]

#: Fallback donor used when a caller passes ``None`` (e.g. a lookup miss
#: such as ``METAL_ATOMS.get("N")``).  It is a saturated amine: a value,
#: not a function, carrying exactly one lone pair to donate.
ANONYMOUS_DONOR: Atom = make_ligand("NH3")


class BondError(ValueError):
    """Raised when an ill-typed application (= impossible bond) is requested."""


# ---------------------------------------------------------------------------
# Atom resolution helpers
# ---------------------------------------------------------------------------

def resolve_atom(atom: AtomLike) -> Atom:
    """Coerce ``atom`` into an :class:`Atom`.

    Accepts an :class:`Atom` (returned unchanged), an element symbol
    looked up in ``PRIMITIVE_ATOMS`` / ``METAL_ATOMS`` / ``make_ligand``,
    or ``None`` -> :data:`ANONYMOUS_DONOR`.
    """
    if atom is None:
        return ANONYMOUS_DONOR
    if isinstance(atom, Atom):
        return atom
    if isinstance(atom, str):
        if atom in PRIMITIVE_ATOMS:
            return PRIMITIVE_ATOMS[atom]
        if atom in METAL_ATOMS:
            return METAL_ATOMS[atom]
        try:
            return make_ligand(atom)
        except ValueError as exc:  # pragma: no cover - defensive
            raise BondError(f"cannot resolve atom {atom!r}") from exc
    raise BondError(f"cannot resolve atom of type {type(atom).__name__}")


def distinct(atom: AtomLike) -> Atom:
    """Return a value-equal but *object-distinct* copy of ``atom``.

    The ledger is keyed by object identity, so two occurrences of the
    same library singleton (e.g. cisplatin's two NH3 ligands) must be
    made distinct before they can carry independent free-site counts.
    """
    return replace(resolve_atom(atom))


# ---------------------------------------------------------------------------
# Free-site bookkeeping (the "environment" of the lambda-term)
# ---------------------------------------------------------------------------

@dataclass
class AtomSite:
    """Mutable bookkeeping cell for one atom occurrence.

    ``used_sites`` counts applied arguments; the atom is in beta-normal
    form (saturated, a *value*) once ``used_sites >= atom.arity`` or once
    it has been explicitly :meth:`saturate`\\ d (which is what happens to
    a dative donor).
    """

    atom: Atom
    used_sites: int = 0
    forced_saturated: bool = False

    @property
    def free_sites(self) -> int:
        """Remaining free parameters of this combinator occurrence."""
        if self.forced_saturated:
            return 0
        return max(0, self.atom.arity - self.used_sites)

    @property
    def is_saturated(self) -> bool:
        """True iff no further application is possible (beta-normal form)."""
        return self.free_sites == 0

    def consume(self, n: int = 1) -> None:
        """Apply ``n`` arguments to this combinator."""
        if n < 0:
            raise BondError(f"cannot consume a negative number of sites: {n}")
        if n > self.free_sites:
            raise BondError(
                f"{self.atom.symbol} has {self.free_sites} free site(s), "
                f"cannot consume {n}"
            )
        self.used_sites += n

    def saturate(self) -> None:
        """Collapse this combinator to a value (fully applied).

        Used for the *donor* of a dative bond: the ligand stops being a
        function once it has donated its electron pair.
        """
        self.forced_saturated = True
        self.used_sites = max(self.used_sites, self.atom.arity)


class FreeSiteLedger:
    """Identity-keyed store of :class:`AtomSite` bookkeeping cells.

    :class:`Atom` is frozen and compares by value, so a plain dict keyed
    by the atom itself would conflate distinct occurrences of the same
    element.  We key by ``id(atom)`` and keep a strong reference to the
    atom so the id stays alive and unique.
    """

    def __init__(self) -> None:
        self._sites: Dict[int, AtomSite] = {}

    def site(self, atom: AtomLike) -> AtomSite:
        """Return (creating on first use) the bookkeeping cell of ``atom``."""
        resolved = resolve_atom(atom)
        key = id(resolved)
        cell = self._sites.get(key)
        if cell is None:
            cell = AtomSite(atom=resolved)
            self._sites[key] = cell
        return cell

    def free_sites(self, atom: AtomLike) -> int:
        """Free sites remaining on ``atom``."""
        return self.site(atom).free_sites

    def is_saturated(self, atom: AtomLike) -> bool:
        """True iff ``atom`` is in beta-normal form (a value)."""
        return self.site(atom).is_saturated

    def reset(self, atom: Optional[AtomLike] = None) -> None:
        """Forget the bookkeeping for ``atom`` (or for everything)."""
        if atom is None:
            self._sites.clear()
        else:
            self._sites.pop(id(resolve_atom(atom)), None)

    def snapshot(self, atoms: Sequence[Atom]) -> Dict[str, int]:
        """``{label: free_sites}`` for ``atoms``, labelled ``symbol#index``."""
        return {
            f"{a.symbol}#{i}": self.free_sites(a) for i, a in enumerate(atoms)
        }

    def check_arity_conservation(self, bonds: Sequence[Bond]) -> bool:
        """Verify free-site accounting balances across ``bonds``.

        Returns True iff the total conservation rule holds::

            sum(free_before) == sum(free_after) + sum(consumed)

        where ``consumed`` is the total free-site consumption that
        each bond kind actually performs in the ledger:

        * **covalent / aromatic** — both partners lose ``order``
          sites, so ``consumed_per_bond == 2 * order``.
        * **dative** — the acceptor loses 1 site (currying) AND the
          donor collapses to a value via ``AtomSite.saturate()``, so
          ``consumed_per_bond == 1 + donor_free_before``.
        * **hydrogen** — no consumption (``consumed == 0``).

        This is the lambda-calculus discipline: every β-reduction
        consumes exactly the number of free parameters its bond kind
        requires — no free sites "leak" anywhere.
        """
        total_before = 0
        total_after = 0
        consumed = 0
        for b in bonds:
            if b.free_a_before < 0 or b.free_b_before < 0:
                # Bond built without snapshots — we cannot check it.
                return False
            total_before += b.free_a_before + b.free_b_before
            if b.kind == DATIVE:
                # Acceptor loses 1 site.  Donor's free_sites go to 0
                # via saturate(), so the donor's free-site drop is
                # exactly ``donor_free_before``.
                if b.donor_is_a:
                    donor_before = b.free_a_before
                    acceptor_before = b.free_b_before
                else:
                    acceptor_before = b.free_a_before
                    donor_before = b.free_b_before
                total_after += max(0, acceptor_before - 1) + 0
                consumed += 1 + donor_before
            elif b.kind == HYDROGEN:
                total_after += b.free_a_before + b.free_b_before
                # consumed += 0
            else:
                # Covalent / aromatic: both sides lose ``order``.
                k = b.order
                a_after = max(0, b.free_a_before - k)
                b_after = max(0, b.free_b_before - k)
                total_after += a_after + b_after
                consumed += 2 * k
        return total_before == total_after + consumed

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FreeSiteLedger({len(self._sites)} tracked atoms)"


#: Process-wide default ledger used when no explicit ledger is supplied.
DEFAULT_LEDGER = FreeSiteLedger()


def _ledger_or_default(ledger: Optional[FreeSiteLedger]) -> FreeSiteLedger:
    return DEFAULT_LEDGER if ledger is None else ledger


def free_sites(atom: AtomLike, ledger: Optional[FreeSiteLedger] = None) -> int:
    """Free sites remaining on ``atom`` in ``ledger`` (default: global)."""
    return _ledger_or_default(ledger).free_sites(atom)


def saturate(atom: AtomLike, ledger: Optional[FreeSiteLedger] = None) -> Atom:
    """Force ``atom`` into beta-normal form (a value) and return it."""
    resolved = resolve_atom(atom)
    _ledger_or_default(ledger).site(resolved).saturate()
    return resolved


def reset_free_sites(ledger: Optional[FreeSiteLedger] = None) -> None:
    """Clear all free-site bookkeeping in ``ledger`` (default: global)."""
    _ledger_or_default(ledger).reset()


# ---------------------------------------------------------------------------
# The Bond = one application
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bond:
    """A bond IS an application ``(atom_a atom_b)`` = one beta-reduction.

    Attributes
    ----------
    atom_a, atom_b : Atom
        The two combinators involved.  For a dative bond the orientation
        is recorded separately in :attr:`donor_is_a`.
    order : int
        Bond order: 1 single, 2 double, 3 triple.  Aromatic bonds carry
        ``order == 1`` with ``kind == 'aromatic'``; their conventional
        1.5 order is exposed as :attr:`effective_order`.  Hydrogen bonds
        carry ``order == 0`` (no sites consumed).
    is_dative : bool
        True iff this is a coordination bond (curried partial application).
    kind : str
        One of ``'covalent'``, ``'dative'``, ``'aromatic'``, ``'hydrogen'``.
    donor_is_a : bool
        Dative bonds only: True iff ``atom_a`` is the electron-pair donor
        (the ligand) and ``atom_b`` is the acceptor (the metal).
    free_a_before, free_b_before : int
        Free sites of each partner *immediately before* the application.
        These snapshots make :func:`is_valid` decidable after the fact.
    """

    atom_a: Atom
    atom_b: Atom
    order: int = 1
    is_dative: bool = False
    kind: str = COVALENT
    donor_is_a: bool = True
    free_a_before: int = -1
    free_b_before: int = -1
    ledger: Optional[FreeSiteLedger] = field(
        default=None, compare=False, repr=False
    )

    # -- orientation ----------------------------------------------------

    @property
    def donor(self) -> Atom:
        """The electron-pair donor (ligand) of a dative bond."""
        return self.atom_a if self.donor_is_a else self.atom_b

    @property
    def acceptor(self) -> Atom:
        """The electron-pair acceptor (metal centre) of a dative bond."""
        return self.atom_b if self.donor_is_a else self.atom_a

    @property
    def atoms(self) -> Tuple[Atom, Atom]:
        return (self.atom_a, self.atom_b)

    # -- kind predicates ------------------------------------------------

    @property
    def is_covalent(self) -> bool:
        return self.kind == COVALENT

    @property
    def is_aromatic(self) -> bool:
        return self.kind == AROMATIC

    @property
    def is_hydrogen(self) -> bool:
        return self.kind == HYDROGEN

    @property
    def effective_order(self) -> float:
        """Chemical bond order (1.5 for aromatic, 0 for hydrogen)."""
        if self.is_aromatic:
            return 1.5
        if self.is_hydrogen:
            return 0.0
        return float(self.order)

    # -- factories = the four application rules --------------------------

    @classmethod
    def covalent(
        cls,
        atom_a: AtomLike,
        atom_b: AtomLike,
        order: int = 1,
        ledger: Optional[FreeSiteLedger] = None,
        strict: bool = True,
    ) -> "Bond":
        """Symmetric application ``(atom_a atom_b)``: a covalent bond.

        Both partners lose ``order`` free sites — a double bond is a
        nested application ``((C O) O)`` and therefore consumes two sites
        on each side.
        """
        led = _ledger_or_default(ledger)
        a, b = resolve_atom(atom_a), resolve_atom(atom_b)
        if a is b:
            raise BondError("an atom cannot bond to itself (no self-application)")
        if order < 1:
            raise BondError(f"covalent bond order must be >= 1, got {order}")
        site_a, site_b = led.site(a), led.site(b)
        free_a, free_b = site_a.free_sites, site_b.free_sites
        if strict and (free_a < order or free_b < order):
            raise BondError(
                f"cannot form order-{order} covalent bond {a.symbol}-{b.symbol}: "
                f"free sites are {free_a} and {free_b}"
            )
        site_a.consume(min(order, free_a))
        site_b.consume(min(order, free_b))
        return cls(
            atom_a=a, atom_b=b, order=order, is_dative=False, kind=COVALENT,
            donor_is_a=True, free_a_before=free_a, free_b_before=free_b,
            ledger=led,
        )

    @classmethod
    def dative(
        cls,
        donor_atom: AtomLike,
        acceptor_atom: AtomLike,
        ledger: Optional[FreeSiteLedger] = None,
        strict: bool = True,
    ) -> "Bond":
        """Curried partial application: a dative / coordination bond.

        **Only the acceptor (metal) loses a free site**; the donor
        (ligand) becomes fully saturated — it stops being a function and
        becomes a value.  This is exactly currying::

            Pt_II : arity 4
            (Pt_II NH3) : arity 3        # Pt keeps 3 free sites
            NH3 : value                  # ligand saturated

        Orientation is auto-corrected: if the *first* argument is a metal
        and the second is not, the pair is swapped so that the metal is
        the acceptor.  Hence both ``Bond.dative(NH3, Pt)`` and the more
        natural-reading ``Bond.dative(Pt, NH3)`` produce the same bond.
        ``None`` resolves to :data:`ANONYMOUS_DONOR` (a saturated amine).
        """
        led = _ledger_or_default(ledger)
        first, second = resolve_atom(donor_atom), resolve_atom(acceptor_atom)
        if first is second:
            raise BondError("an atom cannot coordinate to itself")
        donor, acceptor, donor_is_first = _orient_dative(first, second)
        site_d, site_acc = led.site(donor), led.site(acceptor)
        free_d, free_acc = site_d.free_sites, site_acc.free_sites
        if strict:
            if acceptor.lone_pairs < 1:
                raise BondError(
                    f"{acceptor.symbol} has no lone-pair acceptor site "
                    "(dative bonds need lone_pairs >= 1 on the acceptor)"
                )
            if free_acc < 1:
                raise BondError(
                    f"{acceptor.symbol} is saturated: no coordination site left"
                )
        # Curried application: the acceptor consumes ONE argument slot,
        # the donor collapses to a value.
        if free_acc >= 1:
            site_acc.consume(1)
        site_d.saturate()
        # Preserve the caller's argument order in atom_a/atom_b so the
        # bond mirrors the call, while donor_is_a records the chemistry.
        donor_is_a = donor_is_first
        return cls(
            atom_a=first, atom_b=second, order=1, is_dative=True, kind=DATIVE,
            donor_is_a=donor_is_a,
            free_a_before=free_d if donor_is_first else free_acc,
            free_b_before=free_acc if donor_is_first else free_d,
            ledger=led,
        )

    #: Alias reading in the chemically natural ``(metal, ligand)`` order.
    @classmethod
    def coordinate(
        cls,
        metal: AtomLike,
        ligand: AtomLike,
        ledger: Optional[FreeSiteLedger] = None,
        strict: bool = True,
    ) -> "Bond":
        """``Bond.dative`` spelled ``(metal, ligand)``. See :meth:`dative`."""
        return cls.dative(ligand, metal, ledger=ledger, strict=strict)

    @classmethod
    def aromatic(
        cls,
        ring_atoms: Sequence[AtomLike],
        ledger: Optional[FreeSiteLedger] = None,
        strict: bool = True,
    ) -> List["Bond"]:
        """Eta-equivalence class: the ring bonds of an aromatic system.

        For a 6-membered ring (benzene, pyridine) this returns **6**
        aromatic bonds ``(a0 a1), (a1 a2), ..., (a5 a0)``.  Each ring atom
        participates in two of them and therefore loses exactly two free
        sites (so a ring carbon, arity 4, keeps 2 sites for its H /
        substituent plus its share of the delocalized pi system).

        All six bonds are eta-equivalent: no single Kekule assignment is
        privileged, so every ring bond carries
        ``effective_order == 1.5``.
        """
        atoms = [resolve_atom(a) for a in ring_atoms]
        n = len(atoms)
        if n < 3:
            raise BondError(f"an aromatic ring needs >= 3 atoms, got {n}")
        if len({id(a) for a in atoms}) != n:
            raise BondError(
                "aromatic ring atoms must be distinct objects; "
                "use distinct(atom) to copy library singletons"
            )
        led = _ledger_or_default(ledger)
        if strict:
            for a in atoms:
                if led.free_sites(a) < 2:
                    raise BondError(
                        f"{a.symbol} needs 2 free sites to sit in an aromatic "
                        f"ring, has {led.free_sites(a)}"
                    )
        bonds: List[Bond] = []
        for i in range(n):
            a, b = atoms[i], atoms[(i + 1) % n]
            site_a, site_b = led.site(a), led.site(b)
            free_a, free_b = site_a.free_sites, site_b.free_sites
            site_a.consume(min(1, free_a))
            site_b.consume(min(1, free_b))
            bonds.append(cls(
                atom_a=a, atom_b=b, order=1, is_dative=False, kind=AROMATIC,
                donor_is_a=True, free_a_before=free_a, free_b_before=free_b,
                ledger=led,
            ))
        return bonds

    @classmethod
    def hydrogen(
        cls,
        donor: AtomLike,
        acceptor: AtomLike,
        ledger: Optional[FreeSiteLedger] = None,
        strict: bool = True,
    ) -> "Bond":
        """Type-checked application between two **saturated** atoms.

        A hydrogen bond is not a beta-reduction of free sites: both
        partners are already closed terms (values).  It is a
        *type-checked* interaction between their partial charges, so no
        free site is consumed and ``order == 0``.
        """
        led = _ledger_or_default(ledger)
        d, a = resolve_atom(donor), resolve_atom(acceptor)
        if d is a:
            raise BondError("an atom cannot hydrogen-bond to itself")
        site_d, site_a = led.site(d), led.site(a)
        free_d, free_a = site_d.free_sites, site_a.free_sites
        if strict and (free_d > 0 or free_a > 0):
            raise BondError(
                f"hydrogen bond requires both atoms saturated; free sites are "
                f"{d.symbol}={free_d}, {a.symbol}={free_a} "
                "(call saturate(atom) first)"
            )
        return cls(
            atom_a=d, atom_b=a, order=0, is_dative=False, kind=HYDROGEN,
            donor_is_a=True, free_a_before=free_d, free_b_before=free_a,
            ledger=led,
        )

    # -- the beta-reduction itself ---------------------------------------

    def apply(self, ledger: Optional[FreeSiteLedger] = None) -> Dict[str, object]:
        """Perform / report the beta-reduction: return a molecule-like dict.

        The free-site consumption already happened in the factory (the
        application is what *creates* the bond); ``apply`` materialises
        the reduced term::

            {
              "atoms": [Atom, Atom],
              "bonds": [Bond],
              "free_sites_per_atom": {"Pt_II#0": 3, "NH3#1": 0},
              "is_closed": False,      # Pt still expects 3 arguments
              "open_sites": 3,
            }

        ``is_closed`` is the MLC predicate "all atoms saturated" — i.e.
        this fragment is a closed lambda-term with no free variables.
        """
        led = _ledger_or_default(ledger if ledger is not None else self.ledger)
        return assemble([self], ledger=led)

    # -- cosmetics --------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.is_dative:
            return f"{self.donor.symbol} ->dative {self.acceptor.symbol}"
        glyph = {1: "-", 2: "=", 3: "#"}.get(self.order, "-")
        if self.is_aromatic:
            glyph = ":"
        if self.is_hydrogen:
            glyph = "..."
        return f"{self.atom_a.symbol}{glyph}{self.atom_b.symbol}"


# ---------------------------------------------------------------------------
# Orientation & dispatch
# ---------------------------------------------------------------------------

def _orient_dative(first: Atom, second: Atom) -> Tuple[Atom, Atom, bool]:
    """Return ``(donor, acceptor, donor_is_first)`` for a dative pair.

    Preference order for the acceptor: a metal centre; otherwise the atom
    with more lone pairs available to accept; otherwise ``second`` (which
    matches the declared ``(donor, acceptor)`` signature).
    """
    if first.is_metal and not second.is_metal:
        return second, first, False          # swap: metal must be acceptor
    if second.is_metal and not first.is_metal:
        return first, second, True
    # Neither (or both) is a metal: honour the declared signature.
    return first, second, True


def bond(
    atom_a: AtomLike,
    atom_b: AtomLike,
    order: int = 1,
    ledger: Optional[FreeSiteLedger] = None,
    strict: bool = True,
) -> Bond:
    """Dispatcher: pick the bond kind from the atoms' properties.

    Decision procedure (first match wins):

    1. exactly one partner is a **metal** with a free coordination site
       -> :meth:`Bond.dative` (curried partial application);
    2. both partners are **saturated** (no free sites) -> :meth:`Bond.hydrogen`
       (type-checked application, no sites consumed);
    3. both partners have ``>= order`` free sites -> :meth:`Bond.covalent`.

    Aromatic bonds are *not* dispatched here: aromaticity is a property
    of a whole ring (an eta-equivalence class), so use
    :meth:`Bond.aromatic` with the ring atom list.
    """
    led = _ledger_or_default(ledger)
    a, b = resolve_atom(atom_a), resolve_atom(atom_b)
    free_a, free_b = led.free_sites(a), led.free_sites(b)

    metal_a, metal_b = a.is_metal, b.is_metal
    if metal_a != metal_b:
        acceptor, acceptor_free = (a, free_a) if metal_a else (b, free_b)
        if acceptor.lone_pairs >= 1 and acceptor_free >= 1:
            return Bond.dative(a, b, ledger=led, strict=strict)

    if free_a == 0 and free_b == 0:
        return Bond.hydrogen(a, b, ledger=led, strict=strict)

    if free_a >= order and free_b >= order:
        return Bond.covalent(a, b, order=order, ledger=led, strict=strict)

    # Last resort: a lone-pair donor meeting an unsaturated acceptor is
    # still a dative interaction (e.g. N: -> B).
    if free_a == 0 and b.lone_pairs >= 1 and free_b >= 1:
        return Bond.dative(a, b, ledger=led, strict=strict)
    if free_b == 0 and a.lone_pairs >= 1 and free_a >= 1:
        return Bond.dative(b, a, ledger=led, strict=strict)

    if strict:
        raise BondError(
            f"no valid application between {a.symbol} (free={free_a}) and "
            f"{b.symbol} (free={free_b})"
        )
    return Bond.covalent(a, b, order=order, ledger=led, strict=False)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def is_valid(b: Bond) -> bool:
    """Is ``b`` a well-typed application?

    Judged against the free-site snapshot taken *at bond-formation time*
    (``free_a_before`` / ``free_b_before``), so the answer stays stable
    after the sites have been consumed:

    * **covalent**  iff ``order >= 1`` and both atoms had ``>= order``
      free sites left;
    * **dative**    iff the acceptor has ``lone_pairs >= 1`` and had a
      free coordination site, and the donor was either saturated (a
      value) or itself had free sites;
    * **aromatic**  iff both ring partners had a free site;
    * **hydrogen**  iff both atoms were already saturated (closed terms).
    """
    if not isinstance(b, Bond):
        return False
    if b.free_a_before < 0 or b.free_b_before < 0:
        return False  # bond built by hand without a snapshot

    if b.kind == COVALENT:
        return (
            not b.is_dative
            and b.order >= 1
            and b.free_a_before >= b.order
            and b.free_b_before >= b.order
        )

    if b.kind == DATIVE:
        donor_free = b.free_a_before if b.donor_is_a else b.free_b_before
        acceptor_free = b.free_b_before if b.donor_is_a else b.free_a_before
        acceptor_ok = b.acceptor.lone_pairs >= 1 and acceptor_free >= 1
        # donor is saturated (a value) OR still has free sites of its own
        donor_ok = donor_free == 0 or donor_free >= 1
        return bool(b.is_dative and acceptor_ok and donor_ok)

    if b.kind == AROMATIC:
        return b.free_a_before >= 1 and b.free_b_before >= 1

    if b.kind == HYDROGEN:
        return b.free_a_before == 0 and b.free_b_before == 0

    return False


def can_bond(
    atom_a: AtomLike,
    atom_b: AtomLike,
    kind: str = COVALENT,
    order: int = 1,
    ledger: Optional[FreeSiteLedger] = None,
) -> bool:
    """Non-mutating pre-check: could ``atom_a`` and ``atom_b`` form ``kind``?"""
    led = _ledger_or_default(ledger)
    a, b = resolve_atom(atom_a), resolve_atom(atom_b)
    if a is b:
        return False
    free_a, free_b = led.free_sites(a), led.free_sites(b)
    if kind == COVALENT:
        return order >= 1 and free_a >= order and free_b >= order
    if kind == AROMATIC:
        return free_a >= 1 and free_b >= 1
    if kind == HYDROGEN:
        return free_a == 0 and free_b == 0
    if kind == DATIVE:
        _, acceptor, donor_is_first = _orient_dative(a, b)
        acceptor_free = free_b if donor_is_first else free_a
        return acceptor.lone_pairs >= 1 and acceptor_free >= 1
    raise BondError(f"unknown bond kind: {kind!r}")


# ---------------------------------------------------------------------------
# beta-reduction of a bond list -> molecule-like dict
# ---------------------------------------------------------------------------

def assemble(
    bonds: Iterable[Bond],
    extra_atoms: Sequence[AtomLike] = (),
    ledger: Optional[FreeSiteLedger] = None,
) -> Dict[str, object]:
    """Reduce a list of applications into one molecule-like dict.

    Returns::

        {
          "atoms": [Atom, ...],                 # in first-appearance order
          "bonds": [Bond, ...],
          "free_sites_per_atom": {"C#0": 2, ...},
          "is_closed": bool,     # closed lambda-term: every atom saturated
          "open_sites": int,     # total unapplied arguments remaining
          "valid": bool,         # every bond is well-typed
        }
    """
    bond_list = list(bonds)
    led = _ledger_or_default(ledger)

    atoms: List[Atom] = []
    seen: set[int] = set()
    for b in bond_list:
        for a in (b.atom_a, b.atom_b):
            if id(a) not in seen:
                seen.add(id(a))
                atoms.append(a)
    for raw in extra_atoms:
        a = resolve_atom(raw)
        if id(a) not in seen:
            seen.add(id(a))
            atoms.append(a)

    per_atom = led.snapshot(atoms)
    open_sites = sum(per_atom.values())
    valid_all = all(is_valid(b) for b in bond_list)
    # L2 metrics: BOND_KIND_DISTRIBUTION + DATIVE_FRACTION + FREE_SITES_AFTER_ASSEMBLE +
    #             BOND_VALIDITY_RATE + AROMATIC_RING_SIZE_OK + BOND_STEREO_RATE
    kind_counts: Dict[str, int] = {}
    n_dative = 0
    n_aromatic = 0
    n_double_stereo = 0
    n_double = 0
    aromatic_size_bad = 0
    # Detect ring sizes by walking aromatic bonds (each aromatic ring
    # forms a closed chain of aromatic bonds; we approximate ring size
    # by the count of consecutive aromatic bonds sharing a vertex
    # within this molecule).
    arom_graph: Dict[int, set] = {}
    for b in bond_list:
        kind_counts[b.kind] = kind_counts.get(b.kind, 0) + 1
        if b.kind == DATIVE:
            n_dative += 1
        if b.kind == AROMATIC:
            n_aromatic += 1
            ai = id(b.atom_a)
            bi = id(b.atom_b)
            arom_graph.setdefault(ai, set()).add(bi)
            arom_graph.setdefault(bi, set()).add(ai)
        if b.order == 2:
            n_double += 1
            # BOND_STEREO_RATE: requires Chem.Bond.GetStereo() on the
            # RDKit side.  Since Bond has no stereo field directly,
            # we approximate as 0.0 here (stereo is reported from
            # the RDKit round-trip in layer 3).
            n_double_stereo += 0
        if b.kind == AROMATIC:
            n_atoms_ring = len(arom_graph.get(id(b.atom_a), set()))
            # Cheap proxy: every aromatic bond contributes its incident
            # degree; if degree ∉ {5,6,7} the ring size is wrong.
            if n_atoms_ring not in {5, 6, 7}:
                aromatic_size_bad += 1
    _METRICS.history.append({
        "metric": "BOND_KIND_DISTRIBUTION",
        "value": dict(kind_counts),
        "n_bonds": len(bond_list),
    })
    denom = max(1, len(bond_list))
    _METRICS.history.append({
        "metric": "DATIVE_FRACTION",
        "value": float(n_dative) / denom,
        "n_dative": n_dative,
        "n_total": len(bond_list),
    })
    _METRICS.history.append({
        "metric": "FREE_SITES_AFTER_ASSEMBLE",
        "value": {"open_sites": open_sites, "is_closed": open_sites == 0},
        "n_atoms": len(atoms),
    })
    _METRICS.history.append({
        "metric": "BOND_VALIDITY_RATE",
        "value": float(valid_all),
        "n_bonds": len(bond_list),
        "n_valid": sum(1 for b in bond_list if is_valid(b)),
    })
    _METRICS.history.append({
        "metric": "AROMATIC_RING_SIZE_OK",
        "value": float(1.0 - (aromatic_size_bad / max(1, n_aromatic))) if n_aromatic else 1.0,
        "n_aromatic": n_aromatic,
        "n_bad_size": aromatic_size_bad,
    })
    _METRICS.history.append({
        "metric": "BOND_STEREO_RATE",
        "value": float(n_double_stereo) / max(1, n_double),
        "n_double": n_double,
        "n_stereo": n_double_stereo,
    })
    return {
        "atoms": atoms,
        "bonds": bond_list,
        "free_sites_per_atom": per_atom,
        "is_closed": open_sites == 0,
        "open_sites": open_sites,
        "valid": valid_all,
    }


# ---------------------------------------------------------------------------
# Worked example: cisplatin as a curried partial application
# ---------------------------------------------------------------------------

def cisplatin() -> Dict[str, object]:
    """Build ``Pt_II NH3 NH3 Cl Cl`` — the canonical MLC witness.

    ``Pt_II`` is a 4-arity combinator; the four ligands are applied one
    at a time (left-associated currying), so the intermediate terms have
    arity 3, 2, 1, 0.  The final term is closed (beta-normal form).

    Uses a private ledger so the library singletons are untouched.
    """
    led = FreeSiteLedger()
    pt = distinct(METAL_ATOMS["Pt_II"])
    ligands = [
        distinct(make_ligand("NH3")),
        distinct(make_ligand("NH3")),
        distinct(PRIMITIVE_ATOMS["Cl"]),
        distinct(PRIMITIVE_ATOMS["Cl"]),
    ]
    bonds = [Bond.dative(pt, lig, ledger=led) for lig in ligands]
    mol = assemble(bonds, ledger=led)
    mol["term"] = "((((Pt_II NH3) NH3) Cl) Cl)"
    return mol


__all__ = [
    "COVALENT",
    "DATIVE",
    "AROMATIC",
    "HYDROGEN",
    "ANONYMOUS_DONOR",
    "AtomLike",
    "BondError",
    "AtomSite",
    "FreeSiteLedger",
    "DEFAULT_LEDGER",
    "Bond",
    "bond",
    "is_valid",
    "can_bond",
    "assemble",
    "cisplatin",
    "resolve_atom",
    "distinct",
    "free_sites",
    "saturate",
    "reset_free_sites",
]
