"""Type predicates — the TYPE LAYER (Layer 7) of the Molecular Lambda Calculus.

================================================================
Curry-Howard for chemistry
================================================================
    type            = proposition  (e.g. "Lipinski", "binds MMP2")
    term            = proof        (a molecule, i.e. a closed lambda-term)
    inhabitation    = synthesizable drug candidate

An ADMET filter is therefore literally a *type predicate*: the molecule
either inhabits the type (well-typed, satisfies the predicate) or it
does not (ill-typed, rejected). Drug design = inhabitation search.

================================================================
Why a dataclass + a separate predicate function?
================================================================
We keep the predicate itself a plain function (so it can be pickled,
serialised, and used directly by other layers such as `search_alg/`),
and wrap it in a `TypePredicate` dataclass that carries the
human-readable name + a description of what *ill-typed* means. This
matches the spec in TODO/13_lambda_clickchem/molecular_lambda_calculus.md
section 7 (ADMET layer).

================================================================
Laziness
================================================================
RDKit is a heavy import. It is imported lazily inside each predicate
function so that this module can be loaded cheaply even in environments
where RDKit is unavailable (e.g. CI runners that only test predicates
on pre-computed descriptor dicts).

================================================================
Standard predicates
================================================================
- LIPINSKI  : Lipinski's Rule of Five (Ro5)
- VEBER     : Veber (oral bioavailability)
- EGAN      : Egan (QED-based drug-likeness, QED >= 0.5)
- REOS      : REOS filters (synthetic accessibility + drug-likeness)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

try:
    from molmetal_lam.bonds.application import HYDROGEN  # noqa: F401
except ImportError:  # pragma: no cover - defensive
    HYDROGEN = "hydrogen"


# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypePredicate:
    """A property constraint = a type predicate in the Molecular Lambda Calculus.

    In Curry-Howard, a TypePredicate is the *proposition* side of the
    isomorphism. Calling ``predicate(mol)`` is asking "does ``mol``
    inhabit this type?". Returning ``True`` means the molecule is a
    *proof term* for that proposition; ``False`` means *ill-typed*.

    Attributes
    ----------
    name:
        Human-readable identifier (e.g. ``"Lipinski"``). Used in logs,
        in MCTS node annotations, and in any paper figure that prints
        the inhabited type signature of a generated molecule.
    predicate_fn:
        A pure callable ``(mol: Mol | Any) -> bool``. We accept any
        object that exposes ``mol.GetNumAtoms()`` (i.e. an RDKit
        ``Mol``) so this layer stays decoupled from our own Molecule
        type and can be reused in any RDKit-based pipeline.
    description:
        Free-text description of what *ill-typed* means — i.e. the
        consequence of failing this predicate. Shown to chemists and
        printed in ablations so it is clear why a candidate was
        rejected.
    """

    name: str
    predicate_fn: Callable[[Any], bool]
    description: str = ""

    def __call__(self, mol: Any) -> bool:
        """Apply the predicate (= inhabit the type, or fail)."""
        return bool(self.predicate_fn(mol))

    def ill_typed_reason(self, mol: Any) -> str:
        """Return a human-readable explanation of why ``mol`` is ill-typed.

        Useful for the MCTS logger and the closed-loop diagnostics.
        Returns ``"<name>: satisfied"`` when the predicate passes.
        """
        if self(mol):
            return f"{self.name}: satisfied"
        return f"{self.name}: FAILED — {self.description}"


# ---------------------------------------------------------------------------
# Lazy RDKit descriptor helpers
# ---------------------------------------------------------------------------


def _require_rdkit():
    """Lazily import RDKit and return the modules we depend on.

    Doing this inside the predicate functions keeps module import cheap
    and lets us emit a *single*, clear error if RDKit is missing
    instead of a stack trace at import time.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors, Lipinski  # type: ignore
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "molmetal_lam.types.predicates requires RDKit "
            "(pip install rdkit) — predicate functions need RDKit "
            "Descriptors to evaluate molecular properties."
        ) from e
    return Chem, Descriptors, Lipinski


def _descriptors(mol: Any) -> dict:
    """Compute the standard ADMET descriptors for ``mol``.

    Returns a dict with keys:
        mw        : molecular weight (g/mol)
        logp      : Wildman-Crippen LogP
        hbd       : number of H-bond donors
        hba       : number of H-bond acceptors
        tpsa      : topological polar surface area (Å^2)
        rotb      : number of rotatable bonds
        qed       : QED (Bickerton 2012), in [0, 1]

    ``mol`` may be:
        * an RDKit ``Mol`` directly
        * an object exposing ``.rdkit_mol`` attribute
        * a :class:`molmetal_lam.molecules.closed_term.MoleculeClosedTerm`
          (we serialise via ``to_rdkit()``)

    RDKit is imported lazily so this function is cheap unless called.
    """
    Chem, Descriptors, Lipinski = _require_rdkit()

    rdkit_mol = _resolve_rdkit_mol(mol)
    if rdkit_mol is None:
        _L7_STATE.rdkit_descriptor_miss += 1  # L7.4: resolver miss (upstream of _descriptors)
        raise ValueError(
            "predicates: molecule has no RDKit Mol (mol is None or "
            "could not be serialised via to_rdkit())"
        )

    _t0 = time.perf_counter()
    out = {
        "mw":   float(Descriptors.MolWt(rdkit_mol)),
        "logp": float(Descriptors.MolLogP(rdkit_mol)),
        "hbd":  int(Lipinski.NumHDonors(rdkit_mol)),
        "hba":  int(Lipinski.NumHAcceptors(rdkit_mol)),
        "tpsa": float(Descriptors.TPSA(rdkit_mol)),
        "rotb": int(Lipinski.NumRotatableBonds(rdkit_mol)),
        "qed":  float(Descriptors.qed(rdkit_mol)),
    }
    _L7_STATE.descriptor_compute_ms.append((time.perf_counter() - _t0) * 1000.0)
    return out


def _resolve_rdkit_mol(mol: Any):
    """Best-effort resolution of any ligand-like input to an RDKit Mol.

    Handles:
        * ``None``            -> ``None``
        * RDKit ``Mol``        -> unchanged
        * object with ``.rdkit_mol`` -> follow attribute
        * :class:`MoleculeClosedTerm` -> serialise via ``to_rdkit()``
    """
    if mol is None:
        return None
    if hasattr(mol, "HasSubstructMatch") and hasattr(mol, "GetNumAtoms"):
        return mol
    if hasattr(mol, "rdkit_mol"):
        resolved = getattr(mol, "rdkit_mol")
        if resolved is not None:
            return resolved
    # MLC native type: serialise via the closed-term layer.
    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # noqa: F401
    except ImportError:  # pragma: no cover
        return None
    if isinstance(mol, MoleculeClosedTerm):
        try:
            return mol.to_rdkit()
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Standard ADMET predicates
# ---------------------------------------------------------------------------


def _lipinski_predicate(mol: Any) -> bool:
    """Lipinski's Rule of Five (Ro5): MW<=500, logP<=5, HBD<=5, HBA<=10.

    Ill-typed consequences (failing Ro5):
        Poor oral bioavailability — the molecule will likely struggle
        to cross cell membranes by passive diffusion and is rejected
        from the standard "drug-like" chemical space used by almost
        every oral-drug discovery campaign.
    """
    _t0 = time.perf_counter()
    d = _descriptors(mol)
    ok = (d["mw"] <= 500.0 and d["logp"] <= 5.0 and d["hbd"] <= 5 and d["hba"] <= 10)
    _L7_STATE.per_predicate_calls[LIPINSKI.name] += 1
    _L7_STATE.per_predicate_pass[LIPINSKI.name] += int(ok)
    _L7_STATE.per_predicate_time_ms[LIPINSKI.name].append((time.perf_counter() - _t0) * 1000.0)
    return ok


def _veber_predicate(mol: Any) -> bool:
    """Veber filter: TPSA<=140 Å² AND rotatable bonds <=10.

    Ill-typed consequences:
        Poor oral bioavailability due to either too much polar surface
        (TPSA>140 — bad membrane permeability) or excessive flexibility
        (RotB>10 — poor target binding because the molecule cannot
        lock into a single bioactive conformation).
    """
    _t0 = time.perf_counter()
    d = _descriptors(mol)
    ok = d["tpsa"] <= 140.0 and d["rotb"] <= 10
    _L7_STATE.per_predicate_calls[VEBER.name] += 1
    _L7_STATE.per_predicate_pass[VEBER.name] += int(ok)
    _L7_STATE.per_predicate_time_ms[VEBER.name].append((time.perf_counter() - _t0) * 1000.0)
    return ok


def _egan_predicate(mol: Any) -> bool:
    """Egan filter: QED >= 0.5.

    QED (Quantitative Estimate of Drug-likeness, Bickerton 2012) is a
    multivariate descriptor in [0,1]. Values >=0.5 are considered
    "drug-like" by Egan's original threshold.

    Ill-typed consequences (QED < 0.5):
        The molecule falls outside the multivariate drug-like region
        defined by Egan — typically because of an unusual combination
        of MW, logP, HBD/HBA, TPSA, rotatable bonds, aromatic rings, or
        alerts. Even if individual rules pass, QED summarises them into
        a single number and QED<0.5 is a strong signal that this is
        *not* a typical oral-drug-like compound.
    """
    _t0 = time.perf_counter()
    d = _descriptors(mol)
    ok = d["qed"] >= 0.5
    _L7_STATE.per_predicate_calls[EGAN.name] += 1
    _L7_STATE.per_predicate_pass[EGAN.name] += int(ok)
    _L7_STATE.per_predicate_time_ms[EGAN.name].append((time.perf_counter() - _t0) * 1000.0)
    return ok


def _metal_geometry_predicate(mol: Any) -> bool:
    """METAL_GEOMETRY_OK — every metal atom has its expected coordination number.

    This is the *lambda-calculus* layer of metal chemistry: each metal
    combinator (Pt_II, Ru_II, Zn_II, Ir_III, Cu_II, Au_III) has a fixed
    coordination number (the metal's ``arity``).  A molecule that
    *inhabits* the ``metal_geometry_ok`` type has every metal atom
    forming exactly that number of bonds — not more, not fewer.

    Implementation
    --------------
    We count each metal atom's coordination from the heavy-atom bonds
    in :class:`MoleculeClosedTerm` (or, as a fallback, the RDKit Mol's
    bond list).  The expected CN lookup table is the same one the
    Atom-layer :func:`sanity_check` consults — keeping the geometry
    definition in *one* canonical place.

    A non-metal atom never participates in this check; it does not
    contribute to (or against) ``metal_geometry_ok``.

    Ill-typed consequences
    ----------------------
    A molecule that fails ``metal_geometry_ok`` has a metal centre
    whose actual coordination is different from its preferred
    geometry — e.g. a 5-coordinate Pt(II) (kinetically unstable, not
    the canonical cisplatin-like square-planar complex) or a
    4-coordinate octahedral Ru(II) (vacant coordination sites that
    will be filled by solvent).  Such species are usually excluded
    from structure-based design because the binding pose and the
    pharmacological geometry are both ill-defined.
    """
    _t0 = time.perf_counter()
    # Try MoleculeClosedTerm first — it carries the bond list directly.
    if hasattr(mol, "bonds") and hasattr(mol, "atoms"):
        atoms = list(mol.atoms)
        bonds = list(mol.bonds)
        cn = _coord_numbers_from_bonds(atoms, bonds)
    else:
        # Fallback: resolve via RDKit.  We treat every heavy-atom bond
        # as contributing 1 to coordination — which is the standard
        # chemistry convention for coordination-count bookkeeping.
        rdkit_mol = _resolve_rdkit_mol(mol)
        if rdkit_mol is None:
            ok = False
            _L7_STATE.per_predicate_calls[METAL_GEOMETRY_OK.name] += 1
            _L7_STATE.per_predicate_pass[METAL_GEOMETRY_OK.name] += int(ok)
            _L7_STATE.per_predicate_time_ms[METAL_GEOMETRY_OK.name].append(
                (time.perf_counter() - _t0) * 1000.0
            )
            return ok
        cn = _coord_numbers_from_rdkit(rdkit_mol)

    # Pull the expected CN lookup from the Atom layer (single source of truth).
    try:
        from molmetal_lam.atoms.combinators import _EXPECTED_METAL_CN
    except ImportError:  # pragma: no cover - defensive
        ok = False
        _L7_STATE.per_predicate_calls[METAL_GEOMETRY_OK.name] += 1
        _L7_STATE.per_predicate_pass[METAL_GEOMETRY_OK.name] += int(ok)
        _L7_STATE.per_predicate_time_ms[METAL_GEOMETRY_OK.name].append(
            (time.perf_counter() - _t0) * 1000.0
        )
        return ok

    ok = True
    for sym, expected_cn in _EXPECTED_METAL_CN.items():
        actual = cn.get(sym, 0)
        if actual != 0 and actual != expected_cn:
            ok = False
            break
    _L7_STATE.per_predicate_calls[METAL_GEOMETRY_OK.name] += 1
    _L7_STATE.per_predicate_pass[METAL_GEOMETRY_OK.name] += int(ok)
    _L7_STATE.per_predicate_time_ms[METAL_GEOMETRY_OK.name].append(
        (time.perf_counter() - _t0) * 1000.0
    )
    return ok


def _coord_numbers_from_bonds(atoms: List, bonds: List) -> Dict[str, int]:
    """Count coordination numbers from a MoleculeClosedTerm bond list.

    Returns ``{symbol: count_of_bonds}`` for every metal atom (non-
    metal atoms are ignored).  Each non-hydrogen bond contributes 1 to
    the coordination of each of its two partners.
    """
    cn: Dict[str, int] = defaultdict(int)
    seen: Dict[int, int] = {id(a): i for i, a in enumerate(atoms)}
    metal_syms = set(_metal_symbols())
    for b in bonds:
        if getattr(b, "kind", None) == HYDROGEN:
            continue
        a = b.atom_a
        bb = b.atom_b
        if id(a) in seen:
            sym_a = atoms[seen[id(a)]].symbol
            if sym_a in metal_syms:
                cn[sym_a] += 1
        if id(bb) in seen:
            sym_b = atoms[seen[id(bb)]].symbol
            if sym_b in metal_syms:
                cn[sym_b] += 1
    return dict(cn)


def _coord_numbers_from_rdkit(rdkit_mol) -> Dict[str, int]:
    """Count coordination numbers from an RDKit Mol.

    Returns ``{metal_symbol: count_of_bonds}``.  RDKit bond orders do
    not change coordination count (double bond = one coordination
    slot, chemically), so every heavy-atom bond contributes exactly 1.
    """
    cn: Dict[str, int] = defaultdict(int)
    metal_syms = set(_metal_symbols())
    for atom in rdkit_mol.GetAtoms():
        sym = atom.GetSymbol()
        # RDKit gives only the bare element; for "Pt_II" we need to
        # check the *suffix* via the oxidation state isn't available
        # natively.  We map each metal atom to its first matching
        # canonical symbol — Pt_II -> Pt, etc.
        for canonical, bare in _METAL_BARE_TO_CANONICAL.items():
            if sym == bare:
                cn[canonical] = cn.get(canonical, 0) + len(atom.GetNeighbors())
                break
    return dict(cn)


_METAL_BARE_TO_CANONICAL: Dict[str, str] = {
    "Pt": "Pt_II",
    "Ru": "Ru_II",
    "Zn": "Zn_II",
    "Ir": "Ir_III",
    "Cu": "Cu_II",
    "Au": "Au_III",
}


def _metal_symbols() -> List[str]:
    """Return the canonical MLC metal symbols (sourced from atoms layer)."""
    try:
        from molmetal_lam.atoms.combinators import _EXPECTED_METAL_CN
        return list(_EXPECTED_METAL_CN.keys())
    except ImportError:  # pragma: no cover - defensive
        return ["Pt_II", "Ru_II", "Zn_II", "Ir_III", "Cu_II", "Au_III"]


def _reos_predicate(mol: Any) -> bool:
    """REOS filter (Walters & Namchuk 2003): a stricter, combined filter
    aimed at synthetic accessibility + drug-likeness:
        200 <= MW <= 500
        -2 <= logP <= 5
        HBA <= 10
        HBD <= 5
        TPSA <= 140
        RotB <= 10

    Ill-typed consequences (failing REOS):
        The molecule is either too small to be a meaningful drug
        (MW<200 — likely a metabolite / fragment) or falls outside the
        synthetic accessibility window (too large, too polar, too
        flexible). REOS is meant to reject both the "trivial" and
        "obviously undoable" extremes.
    """
    _t0 = time.perf_counter()
    d = _descriptors(mol)
    ok = (
        200.0 <= d["mw"] <= 500.0
        and -2.0 <= d["logp"] <= 5.0
        and d["hba"] <= 10
        and d["hbd"] <= 5
        and d["tpsa"] <= 140.0
        and d["rotb"] <= 10
    )
    _L7_STATE.per_predicate_calls[REOS.name] += 1
    _L7_STATE.per_predicate_pass[REOS.name] += int(ok)
    _L7_STATE.per_predicate_time_ms[REOS.name].append((time.perf_counter() - _t0) * 1000.0)
    return ok


# Module-level singletons. These are the type predicates that the rest
# of the framework (search_alg/, synthesis/, binding/, ...) refers to.
# They are *propositions*; a generated molecule either inhabits them
# or it does not.

LIPINSKI: TypePredicate = TypePredicate(
    name="Lipinski",
    predicate_fn=_lipinski_predicate,
    description=(
        "Fails Ro5: MW>500, logP>5, HBD>5, or HBA>10. "
        "Ill-typed => poor predicted oral bioavailability."
    ),
)

VEBER: TypePredicate = TypePredicate(
    name="Veber",
    predicate_fn=_veber_predicate,
    description=(
        "Fails Veber: TPSA>140 or rotatable bonds>10. "
        "Ill-typed => poor oral bioavailability or excessive flexibility."
    ),
)

EGAN: TypePredicate = TypePredicate(
    name="Egan",
    predicate_fn=_egan_predicate,
    description=(
        "Fails Egan: QED<0.5. "
        "Ill-typed => outside the multivariate drug-like region "
        "(Bickerton 2012)."
    ),
)

REOS: TypePredicate = TypePredicate(
    name="REOS",
    predicate_fn=_reos_predicate,
    description=(
        "Fails REOS: MW outside [200,500], logP outside [-2,5], "
        "HBA>10, HBD>5, TPSA>140, or RotB>10. "
        "Ill-typed => outside synthetic-accessibility window."
    ),
)


METAL_GEOMETRY_OK: TypePredicate = TypePredicate(
    name="metal_geometry_ok",
    predicate_fn=_metal_geometry_predicate,
    description=(
        "Fails metal geometry: a metal atom (Pt_II/Ru_II/Zn_II/Ir_III/"
        "Cu_II/Au_III) has a coordination number that does not match "
        "its expected geometry (4/6/4/6/4/4). "
        "Ill-typed => ill-defined metal geometry; pharmacophore "
        "shape and binding pose are unreliable."
    ),
)


# Standard "well-typed drug candidate" set. In the paper this is the
# type signature printed in the ablations: any molecule that inhabits
# all four of these is called DrugCandidate.
ALL_ADMET: List[TypePredicate] = [LIPINSKI, VEBER, EGAN, REOS]


# Metal-specific predicates (the geometry type layer). These are
# separate from ALL_ADMET because they are only relevant when the
# molecule contains a metal centre — but they are exposed at module
# level so callers can opt-in to the "metal-drug well-typed" check.
METAL_PREDICATES: List[TypePredicate] = [METAL_GEOMETRY_OK]


# ---------------------------------------------------------------------------
# L7 governance instrumentation (1-line hooks per predicate)
# ---------------------------------------------------------------------------


@dataclass
class _L7Metrics:
    """Counters + histograms for Layer 7 governance metrics.

    Populated by tiny hooks inside the predicate functions and
    :func:`_descriptors`. Snapshot via :func:`l7_metrics`.
    """

    per_predicate_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_predicate_pass: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    descriptor_compute_ms: List[float] = field(default_factory=list)
    rdkit_descriptor_miss: int = 0
    well_typed_calls: int = 0
    well_typed_true: int = 0
    ill_typed_first_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_predicate_time_ms: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


_L7_STATE = _L7Metrics()


# ---------------------------------------------------------------------------
# Inhabitation: DrugCandidate = list of TypePredicates
# ---------------------------------------------------------------------------


# A "DrugCandidate" is, by Curry-Howard, the conjunction of its type
# predicates. We make it a List[TypePredicate] rather than a custom
# class because (a) lists are hashable by content in JSON dumps and
# (b) the framework code in search_alg/ and synthesis/ already iterates
# over plain lists.
DrugCandidate = List[TypePredicate]


def well_typed(molecule: Any, predicates: List[TypePredicate]) -> bool:
    """Return ``True`` iff ``molecule`` inhabits every type in ``predicates``.

    This is the type-checker of the MLC. In Curry-Howard language:
        well_typed(M, [LIPINSKI, VEBER, EGAN]) == True
    means "``M`` is a proof of the proposition
           (Lipinski ∧ Veber ∧ Egan)".
    """
    ok = all(p(molecule) for p in predicates)
    _L7_STATE.well_typed_calls += 1
    _L7_STATE.well_typed_true += int(ok)
    return ok


def ill_typed_reasons(molecule: Any, predicates: List[TypePredicate]) -> List[str]:
    """Return a list of failure messages, one per predicate that fails.

    Useful for MCTS diagnostics, ablation logs, and the closed-loop
    dashboard. Empty list => the molecule is fully well-typed.
    """
    reasons = [p.ill_typed_reason(molecule) for p in predicates if not p(molecule)]
    if reasons:
        _L7_STATE.ill_typed_first_reason[reasons[0].split(":", 1)[0]] += 1  # L7.3: first-failure tally
    return reasons


# ---------------------------------------------------------------------------
# BindingType placeholder (Layer 6 / binding layer stub)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BindingType:
    """Placeholder for the BINDING type of a target (Layer 6 / binding/).

    A BindingType is a higher-order type in the MLC: "ligand must satisfy
    these property AND geometric constraints to inhabit this target".
    Today this is only the *property* side (a list of TypePredicates
    that the ligand must satisfy). The full binding layer — including
    geometric docking, pose scoring, and η-invariant affinity heads —
    is implemented in ``molmetal_lam/binding/`` (Layer 6).

    In the type-rule notation of molecular_lambda_calculus.md §11.2:

        Γ ⊢ M : σ       σ ≤ BindingType(T)
        ──────────────────────────────
              Γ ⊢ M : "binds T"

    i.e. if the ligand passes all the TypePredicates of the binding
    type AND (later) η-canonicalises into the pocket, it inhabits the
    BindingType.

    Attributes
    ----------
    name:
        Target identifier (e.g. ``"MMP2_active_site"``).
    predicates:
        The list of TypePredicates (e.g. ADMET + target-specific
        property constraints) that the ligand must satisfy to inhabit
        this binding type.
    description:
        Free-text description of the binding site (shape constraints,
        pharmacophore, etc.). Kept here as a placeholder; the full
        geometric description lives in the binding/ package.
    """

    name: str
    predicates: List[TypePredicate]
    description: str = ""

    def typecheck(self, molecule: Any) -> bool:
        """Return True iff ``molecule`` inhabits this binding type's
        property predicates.

        Geometric docking is *not* included yet — this is the property
        side of the inhabitation check only. The geometric side will
        come from the binding/ layer (Layer 6).
        """
        return well_typed(molecule, self.predicates)

    def ill_typed_reasons(self, molecule: Any) -> List[str]:
        return ill_typed_reasons(molecule, self.predicates)


__all__ = [
    "TypePredicate",
    "DrugCandidate",
    "ALL_ADMET",
    "METAL_PREDICATES",
    "LIPINSKI",
    "VEBER",
    "EGAN",
    "REOS",
    "METAL_GEOMETRY_OK",
    "well_typed",
    "ill_typed_reasons",
    "BindingType",
    "l7_metrics",
    "reset_l7_metrics",
]


def reset_l7_metrics() -> None:
    """Zero out the L7 counters (useful for tests / re-measurement)."""
    global _L7_STATE
    _L7_STATE = _L7Metrics()


def l7_metrics() -> Dict[str, Any]:
    """Snapshot of Layer 7 governance counters.

    Keys
    ----
    per_predicate_calls / per_predicate_pass : dict[str, int]
        L7.1 — pass-rate per predicate (LIPINSKI/VEBER/EGAN/REOS).
    descriptor_compute_ms : list[float]
        L7.2 — per-call wall time of ``_descriptors``.
    ill_typed_first_reason : dict[str, int]
        L7.3 — first-failure reason histogram.
    rdkit_descriptor_miss : int
        L7.4 — upstream resolver miss counter.
    well_typed_calls / well_typed_true : int
        L7.5 — fraction of ``well_typed`` calls returning True.
    per_predicate_time_ms : dict[str, list[float]]
        L7-NEW — per-predicate wall-time histogram.
    """
    # Materialise defaultdicts into plain dicts for deterministic output.
    calls = dict(_L7_STATE.per_predicate_calls)
    pass_ = dict(_L7_STATE.per_predicate_pass)
    reasons = dict(_L7_STATE.ill_typed_first_reason)
    times = {k: list(v) for k, v in _L7_STATE.per_predicate_time_ms.items()}
    pass_rate = {
        name: (pass_.get(name, 0) / calls[name]) if calls.get(name, 0) else 0.0
        for name in calls
    }
    return {
        "per_predicate_calls": calls,
        "per_predicate_pass": pass_,
        "per_predicate_pass_rate": pass_rate,
        "descriptor_compute_ms": list(_L7_STATE.descriptor_compute_ms),
        "ill_typed_first_reason": reasons,
        "rdkit_descriptor_miss": int(_L7_STATE.rdkit_descriptor_miss),
        "well_typed_calls": int(_L7_STATE.well_typed_calls),
        "well_typed_true": int(_L7_STATE.well_typed_true),
        "well_typed_fraction": (
            _L7_STATE.well_typed_true / _L7_STATE.well_typed_calls
            if _L7_STATE.well_typed_calls
            else 0.0
        ),
        "per_predicate_time_ms": times,
    }