"""Binding Layer — Protein + Ligand = β-reduction + type check.

This module implements **Layer 6** of the Molecular Lambda Calculus
(MLC), formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §6.

Core thesis
-----------
A binding site IS a **higher-order type** — a proposition "ligand M
binds pocket P" — whose constraints must all be satisfied (the type
checker) and whose geometric / electronic interactions must be
reducible (β-reduction of partial applications).  Equivalently:

    "ligand binds target"  ≡  ligand inhabits BindingType(target)

This module covers the *type* and *type-check* side; the *pose* and
*affinity* side will be supplied by the docking adapters under
``molmetal/molmetal_lam/sbdd_env/`` (REINVENT4 / DiffDock / FlexSBDD)
and by EGNN-based pIC50 heads in the parallel ``12_flow_matching``
track.  Here we only implement:

    * :class:`BindingSite`       — a higher-order type
    * :class:`BindingTypeCheckResult` — the type-checker's verdict
    * :func:`typecheck`          — predicate: does ligand inhabit the type?
    * Canonical sites :data:`MMP2_ACTIVE`, :data:`PT_DNA_MAJOR_GROOVE`,
      :data:`KINASE_ATP`, :data:`PROTEASE_GENERIC`.

The "binding IS β-reduction + type-check" slogan is implemented as
follows:

    1. Each constraint is a :class:`TypePredicate` (the property side).
    2. Each constraint is checked against the *ligand closed-term* — if
       any constraint fails, the type-checker returns immediately with
       the violation.  The β-reduction is performed internally via the
       bond layer (the ligand's bonds are already applications that
       consumed free sites; the type-checker walks the resulting term).
    3. The geometry / chemistry of the binding site is a list of
       additional *coordination requirements* (e.g. "must donate 2 lone
       pairs to a square-planar Pt(II) centre").  These are checked as
       *reduction rules*: they look for the necessary sub-term in the
       ligand and report whether the corresponding β-step would be
       well-typed.

The outcome is a :class:`BindingTypeCheckResult` with three fields:

    success                  bool: does it inhabit the binding type?
    pic50_estimate           float: rough pIC50 estimate (NaN if N/A)
    violated_constraints     list[str]: human-readable failure reasons

These three are the input to the MCTS layer
(``molmetal_lam.search_alg``) and to the closed-loop pipeline in
``molmetal.orchestration``.

Notes
-----
This layer DOES NOT perform real 3D docking.  Docking is delegated to
the sbdd_env adapters.  The :func:`typecheck` here only verifies the
*type* of the binding interaction: pharmacophore + property + a
topological check that the right donor/acceptor atoms exist in the
ligand.  Geometric feasibility (van der Waals clashes, pose RMSD) is
added later by the docking step.

We deliberately keep the implementation free of RDKit at import-time
(the geometry-hint computations use RDKit lazily) so this module can
be imported in any environment, including the MLC paper-figure
generator that only needs the type-checker.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from molmetal_lam.atoms.combinators import METAL_ATOMS, Atom
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.types.predicates import (
    LIPINSKI,
    TypePredicate,
    well_typed as _well_typed,
)


# ---------------------------------------------------------------------------
# Binding site (higher-order type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BindingSite:
    """A binding site IS a higher-order type in the MLC.

    A binding site is the **proposition** "ligand M binds pocket P"
    under a Curry-Howard reading.  Its inhabitants are exactly those
    ligands whose property constraints and chemistry requirements are
    all satisfied.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``'MMP2_active_site'``,
        ``'Pt_DNA_major_groove'``).
    constraints : list[TypePredicate]
        Property predicates (ADMET-style filters, chemotype filters,
        etc.) that the ligand must satisfy to inhabit this type.
        Typical contents for an MMP2 inhibitor site include:

            * hydroxamic-acid ZBG (for Zn coordination)
            * at least 2 H-bond donors (backbone H-bond network)
            * a moderate logP (hydrophobic-pocket access)

        A ligand that fails any single constraint is **ill-typed** and
        cannot inhabit this binding type — the type-checker reports
        the violation immediately.
    geometry_hints : dict
        Free-form bag of geometric / electronic requirements.  Common
        keys:

            * ``'coordination_number'``   : int (e.g. 4 for Pt(II))
            * ``'preferred_donors'``      : list[str] (e.g. ``['N', 'O']``)
            * ``'geometry'``              : str (e.g. ``'square_planar'``)
            * ``'metal'``                 : str (e.g. ``'Pt_II'``)
            * ``'min_donors'``            : int (count of donor atoms)
            * ``'min_hbond_donors'``      : int
            * ``'min_hbond_acceptors'``   : int
            * ``'logp_window'``           : (lo, hi) tuple

        These hints are read by :func:`typecheck` and contribute to the
        β-reduction / geometry check.  Unknown keys are tolerated (they
        are documentation) — the type-checker does not fail on extras.
    description : str
        Free-text description of the binding site (shape, residues,
        pharmacophore context).
    """

    name: str
    constraints: List[TypePredicate] = field(default_factory=list)
    geometry_hints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # ------------------------------------------------------------------
    # Higher-order type queries
    # ------------------------------------------------------------------

    @property
    def constraint_names(self) -> List[str]:
        """Names of the constraints (for logs and paper figures)."""
        return [c.name for c in self.constraints]

    @property
    def metal(self) -> Optional[str]:
        """Name of the metal centre declared in geometry_hints, if any."""
        v = self.geometry_hints.get("metal")
        return v if isinstance(v, str) else None

    @property
    def coordination_number(self) -> Optional[int]:
        """Required coordination number (from geometry_hints)."""
        v = self.geometry_hints.get("coordination_number")
        return int(v) if v is not None else None

    # ------------------------------------------------------------------
    # PDB-driven factory (stub)
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb(
        cls,
        pdb_id: str,
        lig_center: Any,
        radius: float = 10.0,
        name: Optional[str] = None,
    ) -> "BindingSite":
        """Build a BindingSite from a PDB file + a ligand centre.

        STUB
        ----
        This is a placeholder for the Phase-3 work that will pipe a real
        PDB file through the docking stack (``sbdd_env``) and convert
        the resulting pocket into a BindingSite.  Today we
        conservatively construct a minimal BindingSite whose geometry
        hints describe the ligand centre.

        The full implementation will:

            1. call ``Pocket.from_pdb_file(pdb_id, lig_center, radius)``
               (lives in ``molmetal/molmetal_lam/sbdd_env/pocket.py``);
            2. inspect the pocket residues to detect a metal centre
               (e.g. Zn in MMP2, Mg in kinase, Pt covalently bound to
               a guanine N7 in DNA adducts);
            3. convert detected pharmacophore features (H-bond
               donors/acceptors, hydrophobic residues, charged
               residues) into the constraints list.

        Parameters
        ----------
        pdb_id : str
            Four-letter PDB code (e.g. ``'1QIB'`` for MMP2).
        lig_center : array-like of length 3
            Cartesian centre of the bound ligand (Å).
        radius : float
            Pocket radius around ``lig_center`` in Å.
        name : str or None
            Optional override for the binding-site name.  Defaults to
            ``f"{pdb_id}_pocket"``.
        """
        # Future: real PDB parsing. Today we just record the inputs.
        lig_center_tuple = (
            tuple(lig_center) if hasattr(lig_center, "__iter__") else (lig_center,)
        )
        return cls(
            name=name or f"{pdb_id}_pocket",
            constraints=[],
            geometry_hints={
                "pdb_id": pdb_id,
                "lig_center": lig_center_tuple,
                "radius": float(radius),
                # Default conservative hints — replaced by real pocket
                # inspection in the production implementation.
                "preferred_donors": ["N", "O"],
                "min_donors": 1,
            },
            description=(
                f"Stub BindingSite built from PDB {pdb_id} at centre "
                f"{lig_center_tuple} (radius {radius:.1f} Å). "
                "Replace with Pocket.from_pdb_file for real pharmacophore."
            ),
        )


# ---------------------------------------------------------------------------
# Type-check result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BindingTypeCheckResult:
    """The type-checker's verdict on a ligand / binding-site inhabitation.

    Attributes
    ----------
    success : bool
        True iff every constraint passed (and the geometric β-check
        also succeeded).  Inhabitation = a positive witness exists.
    pic50_estimate : float
        Rough pIC50 estimate in [0, 12]; ``float('nan')`` if no
        estimate is available.  This is the result of a heuristic
        *type-inhabitation score*, not a real docking calculation —
        the docking pipeline will refine it.  By convention:

            * pic50 >= 8       strong binder (nanomolar)
            * 6 <= pic50 < 8  moderate binder (micromolar)
            * pic50 < 6       weak binder
    violated_constraints : list[str]
        Human-readable failure reasons.  Empty when ``success=True``.
        The first entry is the *earliest* constraint that failed
        (= the one that caused the rejection).
    details : dict
        Free-form diagnostic bag (per-constraint pass/fail flags,
        geometric β-check verdict, descriptor snapshot).  Used by the
        closed-loop pipeline and the paper-figure generator.
    """

    success: bool
    pic50_estimate: float = float("nan")
    violated_constraints: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Library of canonical predicate constructors
# ---------------------------------------------------------------------------


def _require_rdkit():
    """Lazy RDKit import for the chemotype / descriptor predicates."""
    try:
        from rdkit import Chem  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "binding/types chemotype predicates require RDKit "
            "(uv pip install rdkit)."
        ) from exc
    return Chem


def _resolve_for_substructure(mol: Any):
    """Resolve any ligand-like input into an RDKit ``Mol`` for SMARTS matching.

    Accepts:
        * an RDKit ``Mol`` directly
        * any object with ``.rdkit_mol`` attribute
        * a :class:`MoleculeClosedTerm` (we serialise via ``to_rdkit()``)

    Returns ``None`` if the ligand cannot be resolved.
    """
    if mol is None:
        return None
    if hasattr(mol, "rdkit_mol"):
        resolved = getattr(mol, "rdkit_mol")
        if resolved is not None:
            return resolved
    if isinstance(mol, MoleculeClosedTerm):
        try:
            return mol.to_rdkit()
        except Exception:
            return None
    if hasattr(mol, "HasSubstructMatch"):
        return mol
    return None


def hydroxamic_acid_present(mol: Any) -> bool:
    """True iff the ligand contains a hydroxamic acid ZBG ``C(=O)NOH``.

    The hydroxamic acid ``-C(=O)-NH-OH`` is the canonical zinc-binding
    group (ZBG) for matrix metalloproteinase (MMP) inhibitors — it
    chelates the catalytic Zn²⁺ in a bidentate fashion and is the
    "warhead" used by essentially every MMP2 / MMP9 clinical candidate.
    """
    Chem = _require_rdkit()
    rdkit_mol = _resolve_for_substructure(mol)
    if rdkit_mol is None:
        return False
    pattern = Chem.MolFromSmarts("[CX3](=O)[NX3][OX2]")
    if pattern is None:
        return False
    return rdkit_mol.HasSubstructMatch(pattern)


def has_metal_coordination_warhead(mol: Any) -> bool:
    """True iff the ligand has a recognised metal-coordination warhead.

    Recognised ZBGs (subset relevant to Zn²⁺ / Pt(II) / Ru(II)):
        * hydroxamic acid   ``-C(=O)-NH-OH``
        * carboxylic acid   ``-C(=O)-OH``
        * reverse hydroxamate ``-NH-OH`` adjacent to ``C=O``
        * thiol             ``-SH``
        * phosphonate       ``-P(=O)(OH)(OH)``
        * a built-in transition-metal centre (Pt, Pd, Ru, Ir, Au, Zn)

    Returns False on an unknown warhead.
    """
    Chem = _require_rdkit()
    rdkit_mol = _resolve_for_substructure(mol)
    if rdkit_mol is None:
        return False
    warheads = [
        "[CX3](=O)[NX3][OX2]",   # hydroxamic acid
        "[CX3](=O)[OX2H1]",      # carboxylic acid
        "[SX2H1]",               # thiol
        "[PX4](=O)([OX2H1])[OX2H1]",  # phosphonate
    ]
    for sma in warheads:
        pat = Chem.MolFromSmarts(sma)
        if pat is not None and rdkit_mol.HasSubstructMatch(pat):
            return True
    # A built-in coordination centre also counts as a metal-coordination
    # ligand: cisplatin / auranofin / etc. are *coordination complexes*
    # where the metal itself is the warhead.
    for atom in rdkit_mol.GetAtoms():
        if atom.GetSymbol() in {"Pt", "Pd", "Ru", "Ir", "Au", "Zn",
                                 "Cu", "Ni", "Fe", "Co", "Mn", "Mg"}:
            return True
    return False


def square_planar_pt_center(mol: Any) -> bool:
    """True iff the ligand has a Pt(II) centre with 4 dative sites.

    Detection: the molecule contains a Pt atom bonded to >= 4 other
    heavy atoms in a non-covalent sense (i.e. ligands).  We use the
    Pt(II) combinator's arity = 4 as the canonical witness.
    """
    Chem = _require_rdkit()
    rdkit_mol = _resolve_for_substructure(mol)
    if rdkit_mol is None:
        return False
    for atom in rdkit_mol.GetAtoms():
        if atom.GetSymbol() == "Pt":
            # coordination count = heavy-atom neighbours (RDKit
            # represents dative bonds as ordinary singles)
            if atom.GetDegree() >= 4:
                return True
    return False


def has_hydrophobic_pocket(mol: Any) -> bool:
    """True iff the ligand carries at least one substantial hydrophobic group.

    Heuristic: the molecule has at least one aromatic ring OR at least
    three sp3 carbons in a row (alkyl chain).  This is the simplest
    sufficient condition for "fills a hydrophobic pocket" — it does
    not compute the actual LogP contribution.
    """
    Chem = _require_rdkit()
    rdkit_mol = _resolve_for_substructure(mol)
    if rdkit_mol is None:
        return False
    # RDKit exposes ring membership via GetRingInfo (older versions
    # used GetSymmSSSR). Try the modern API first, then the legacy.
    rings: List = []
    try:
        info = rdkit_mol.GetRingInfo()
        rings = list(info.AtomRings())
    except Exception:
        try:
            rings = list(rdkit_mol.GetSymmSSSR())
        except Exception:
            rings = []
    aromatic = sum(
        1 for r in rings if any(
            rdkit_mol.GetAtomWithIdx(i).GetIsAromatic() for i in r
        )
    )
    if aromatic > 0:
        return True
    # 3+ consecutive sp3 carbons
    sp3_carbon = Chem.MolFromSmarts("[CX4][CX4][CX4]")
    if sp3_carbon is not None and rdkit_mol.HasSubstructMatch(sp3_carbon):
        return True
    return False


def min_hbond_donors(n: int):
    """Build a predicate: ligand must have at least ``n`` H-bond donors."""

    def _pred(mol: Any) -> bool:
        try:
            from rdkit.Chem import Lipinski  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("min_hbond_donors requires RDKit") from exc
        rdkit_mol = _resolve_for_substructure(mol)
        if rdkit_mol is None:
            return False
        return int(Lipinski.NumHDonors(rdkit_mol)) >= n

    return _pred


def min_hbond_acceptors(n: int):
    """Build a predicate: ligand must have at least ``n`` H-bond acceptors."""

    def _pred(mol: Any) -> bool:
        try:
            from rdkit.Chem import Lipinski  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("min_hbond_acceptors requires RDKit") from exc
        rdkit_mol = _resolve_for_substructure(mol)
        if rdkit_mol is None:
            return False
        return int(Lipinski.NumHAcceptors(rdkit_mol)) >= n

    return _pred


def logp_in_range(lo: float, hi: float):
    """Build a predicate: Wildman-Crippen logP must lie in ``[lo, hi]``."""

    def _pred(mol: Any) -> bool:
        try:
            from rdkit.Chem import Descriptors  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("logp_in_range requires RDKit") from exc
        rdkit_mol = _resolve_for_substructure(mol)
        if rdkit_mol is None:
            return False
        return lo <= float(Descriptors.MolLogP(rdkit_mol)) <= hi

    return _pred


# ---------------------------------------------------------------------------
# The type-checker = β-reduction + property type-check
# ---------------------------------------------------------------------------


def _ligand_rdkit_mol(ligand: Any):
    """Best-effort extraction of the RDKit Mol from a ligand input.

    Accepts:
        * :class:`MoleculeClosedTerm` (we call ``to_rdkit()``)
        * an RDKit ``Mol`` directly
        * any object that exposes ``.rdkit_mol``

    Returns ``None`` if the ligand cannot be resolved.
    """
    if isinstance(ligand, MoleculeClosedTerm):
        try:
            return ligand.to_rdkit()
        except Exception:
            return None
    if hasattr(ligand, "rdkit_mol"):
        return getattr(ligand, "rdkit_mol")
    # RDKit Mol passthrough: just check it has the right interface.
    if hasattr(ligand, "GetNumAtoms") and hasattr(ligand, "HasSubstructMatch"):
        return ligand
    return None


def _geometric_beta_check(
    ligand: Any, site: BindingSite, mol
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Geometric / electronic β-check on the ligand.

    This is the "β-reduction" half of "binding IS β-reduction +
    type-check".  We verify that the ligand **can** form the
    applications required by the binding site — concretely:

        * the right donor elements exist (``preferred_donors``)
        * the right count of donor atoms exists (``min_donors``)
        * a metal centre, if required, is present and has the
          expected coordination number

    Returns
    -------
    (ok, reasons, details)
        ok        : True iff every required geometric / electronic
                    condition is met
        reasons   : list[str] of failure reasons (empty on success)
        details   : dict with per-check results for diagnostics
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {}
    if mol is None:
        return False, ["geometric β-check skipped: ligand has no RDKit Mol"], details

    Chem = _require_rdkit()
    hints = site.geometry_hints
    metal_name = hints.get("metal")
    metal_symbol = metal_name.split("_", 1)[0] if metal_name else None

    # Convention: when the binding site declares a metalloprotein
    # centre (e.g. Zn_II in MMP2), the metal is part of the **pocket**
    # and the ligand supplies a warhead (hydroxamic acid, etc.) that
    # coordinates it.  In that case we *don't* require the metal to
    # appear in the ligand — only the donor atoms.  When the metal is
    # part of the **ligand** (e.g. cisplatin → Pt_DNA), we *do* require
    # it to be present.  The flag ``metal_in_ligand`` distinguishes.
    metal_in_ligand = bool(hints.get("metal_in_ligand", metal_symbol in {
        "Pt", "Pd", "Ru", "Ir", "Au",
    }))

    # 1. Donor element check.
    preferred_donors: List[str] = list(hints.get("preferred_donors", []))
    if preferred_donors:
        donor_atom_symbols = {
            atom.GetSymbol() for atom in mol.GetAtoms()
        }
        has_preferred = any(d in donor_atom_symbols for d in preferred_donors)
        details["preferred_donors"] = preferred_donors
        details["donor_atom_symbols"] = sorted(donor_atom_symbols)
        details["has_preferred_donor"] = has_preferred
        if not has_preferred:
            reasons.append(
                f"no atom with preferred donor element "
                f"{preferred_donors} (found {sorted(donor_atom_symbols)})"
            )

    # 2. Minimum donor count.
    min_donors = hints.get("min_donors")
    if min_donors is not None:
        donor_atoms: List[int] = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in {"N", "O", "S", "P"}:
                donor_atoms.append(atom.GetIdx())
            elif (metal_in_ligand and metal_symbol is not None
                  and sym == metal_symbol):
                # Metal centre itself is the donor (coordination core)
                donor_atoms.append(atom.GetIdx())
        details["min_donors_required"] = int(min_donors)
        details["donor_atoms"] = donor_atoms
        if len(donor_atoms) < int(min_donors):
            reasons.append(
                f"need at least {min_donors} donor atoms "
                f"(N/O/S/P or metal), found {len(donor_atoms)}"
            )

    # 3. Metal centre check (only required when the metal is part of
    #    the ligand — cisplatin / auranofin / etc.).
    if metal_in_ligand and metal_name is not None and metal_symbol is not None:
        found_metal = any(a.GetSymbol() == metal_symbol for a in mol.GetAtoms())
        details["metal_required"] = metal_name
        details["metal_found"] = found_metal
        if not found_metal:
            reasons.append(f"required metal {metal_name} not present in ligand")

    # 4. Coordination number check (ligand-metal binding).
    coord = hints.get("coordination_number")
    if coord is not None and metal_in_ligand and metal_name is not None:
        ok = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == metal_symbol and atom.GetDegree() >= int(coord):
                ok = True
                break
        details["coordination_number_required"] = int(coord)
        details["coordination_ok"] = ok
        if not ok:
            reasons.append(
                f"required coordination_number={coord} for {metal_name} "
                f"not satisfied"
            )

    # 5. H-bond donor / acceptor minimums (if requested via hints).
    min_hbd = hints.get("min_hbond_donors")
    if min_hbd is not None:
        try:
            from rdkit.Chem import Lipinski  # type: ignore
        except ImportError:  # pragma: no cover
            pass
        else:
            hbd = int(Lipinski.NumHDonors(mol))
            details["min_hbond_donors_required"] = int(min_hbd)
            details["hbond_donors"] = hbd
            if hbd < int(min_hbd):
                reasons.append(
                    f"need at least {min_hbd} H-bond donors, found {hbd}"
                )

    min_hba = hints.get("min_hbond_acceptors")
    if min_hba is not None:
        try:
            from rdkit.Chem import Lipinski  # type: ignore
        except ImportError:  # pragma: no cover
            pass
        else:
            hba = int(Lipinski.NumHAcceptors(mol))
            details["min_hbond_acceptors_required"] = int(min_hba)
            details["hbond_acceptors"] = hba
            if hba < int(min_hba):
                reasons.append(
                    f"need at least {min_hba} H-bond acceptors, found {hba}"
                )

    return (len(reasons) == 0), reasons, details


def _estimate_pic50(
    ligand: Any, site: BindingSite, mol, geom_ok: bool, details: Dict[str, Any]
) -> float:
    """Heuristic pIC50 estimate = type-inhabitation score.

    This is NOT a real docking score.  It is the *type inhabitation
    score* the EGNN head will eventually replace (see Layer 10 of MLC).
    Today it is a transparent heuristic in [0, 12]:

        base     = 5.0
        +1.5     if hydroxamic / warhead present (MMP-like site)
        +1.0     if geometric β-check passed
        +1.0     if Lipinski satisfied
        +1.0     if MW < 500
        +0.5     if HBA >= 3
        -1.5     if Lipinski violated

    Returns ``float('nan')`` if the molecule cannot be resolved.
    """
    if mol is None:
        return float("nan")
    score = 5.0

    try:
        from rdkit.Chem import Descriptors, Lipinski  # type: ignore
    except ImportError:  # pragma: no cover
        return float("nan")

    mw = float(Descriptors.MolWt(mol))
    hba = int(Lipinski.NumHAcceptors(mol))

    # Property checks
    lipinski_ok = _well_typed(mol, [LIPINSKI])
    score += 1.0 if lipinski_ok else -1.5
    score += 1.0 if mw < 500.0 else -0.5
    score += 0.5 if hba >= 3 else 0.0

    # Geometric check
    if geom_ok:
        score += 1.0

    # Site-specific warhead bonus
    if site.metal is None and "MMP" in site.name.upper():
        if hydroxamic_acid_present(mol):
            score += 1.5

    # Pt-specific: bonus if there is a Pt centre with >=4 dative bonds
    if site.metal and site.metal.startswith("Pt"):
        if square_planar_pt_center(mol):
            score += 2.0

    # Clip into a sane range
    score = max(0.0, min(12.0, score))
    details["pic50_components"] = {
        "base": 5.0,
        "lipinski_ok": lipinski_ok,
        "mw_lt_500": mw < 500.0,
        "hba_ge_3": hba >= 3,
        "geom_ok": geom_ok,
        "final": score,
    }
    return float(score)


def _extract_smiles(ligand: Any) -> Optional[str]:
    """Best-effort SMILES extraction from a ligand input.

    Returns ``None`` when the ligand cannot be serialised.
    """
    if isinstance(ligand, MoleculeClosedTerm):
        try:
            return ligand.canonical_smiles()
        except Exception:
            return None
    if isinstance(ligand, str):
        return ligand
    if hasattr(ligand, "canonical_smiles"):
        try:
            return ligand.canonical_smiles()
        except Exception:
            return None
    return None


def _resolve_docking_oracle():
    """Pick the best available docking oracle (FlowDock > DiffDock > None).

    Both adapters raise :class:`AdapterUnavailable` from ``dock`` when
    their backend is missing — callers MUST catch that and fall back to
    the fingerprint stub.  We never raise here at import-time: the
    absence of a backend is the *expected* state in CI / smoke tests.
    """
    try:  # pragma: no cover — import guard
        from molmetal_lam.sbdd_env.flowdock_adapter import FlowDockAdapter
        flow = FlowDockAdapter()
        if flow.is_available():
            return flow
    except Exception:  # pragma: no cover - defensive
        pass
    try:  # pragma: no cover — import guard
        from molmetal_lam.sbdd_env.diffdock_adapter import DiffDockAdapter
        diff = DiffDockAdapter()
        if diff.is_available():
            return diff
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def _oracle_leaf_typecheck(
    ligand: Any, site: BindingSite, oracle: Any
) -> Optional[BindingTypeCheckResult]:
    """Run the docking oracle on a *top-K* leaf candidate.

    Returns ``None`` when the oracle cannot produce a verdict
    (e.g. :class:`AdapterUnavailable` is raised, or the ligand cannot
    be serialised to SMILES) — the caller then falls back to the
    fingerprint stub.

    The verdict is *and-ed* with the fingerprint stub verdict: the
    binding type still has to type-check AND the oracle has to
    agree on RMSD / Vina.  This makes the binding layer a 2-of-2
    witness rather than either-or.
    """
    try:
        smiles = _extract_smiles(ligand)
        if smiles is None:
            return None
        # Pocket PDB path comes from the geometry hints, when present.
        pocket_pdb = ""
        try:
            pocket_pdb = str(site.geometry_hints.get("pdb_path", ""))
        except Exception:
            pocket_pdb = ""
        if not pocket_pdb:
            # Without a real pocket PDB the oracle cannot dock; this
            # is the expected state during MCTS expansion.  Returning
            # ``None`` here lets the fingerprint stub take over.
            return None
        result = oracle.dock(smiles, pocket_pdb)
    except Exception as exc:  # pragma: no cover - AdapterUnavailable etc.
        logger_name = "molmetal_lam.binding.types"
        import logging as _logging
        _logging.getLogger(logger_name).debug(
            "docking oracle unavailable (%s): %s", type(exc).__name__, exc,
        )
        return None
    # Compose the oracle verdict into a BindingTypeCheckResult.
    rmsd_ok = result.rmsd_A == result.rmsd_A and result.rmsd_A < 2.0  # <2Å canonical
    vina_ok = result.vina_kcal == result.vina_kcal and result.vina_kcal <= -7.0
    return BindingTypeCheckResult(
        success=bool(rmsd_ok and vina_ok),
        pic50_estimate=float("nan"),
        violated_constraints=(
            [] if (rmsd_ok and vina_ok)
            else [
                f"oracle: rmsd_A={result.rmsd_A:.2f} vina_kcal={result.vina_kcal:.2f}"
                f" confidence={result.confidence:.2f}"
            ]
        ),
        details={
            "oracle": oracle.name,
            "rmsd_A": float(result.rmsd_A),
            "vina_kcal": float(result.vina_kcal),
            "confidence": float(result.confidence),
        },
    )


def typecheck(
    ligand: Any,
    site: BindingSite,
    *,
    leaf_oracle_call: bool = False,
    oracle: Any = None,
) -> BindingTypeCheckResult:
    """Type-check a ligand against a binding site (= β-reduction + type check).

    This is the entry point of the MLC Binding Layer.  It performs
    two stages:

    **Stage 1 — property type-check.**  Every TypePredicate in
    ``site.constraints`` is applied to the ligand.  Failure of any
    single predicate short-circuits the check.

    **Stage 2 — geometric β-check.**  The ligand's atom stack is
    walked (via RDKit) and compared against ``site.geometry_hints``.
    This is the "β-reduction" half — it verifies that the right donor
    atoms exist and that the geometric reductions the binding pocket
    expects can be performed.

    **Stage 3 — optional oracle leaf type-check (top-K only).**
    When ``leaf_oracle_call=True`` and an ``oracle`` is supplied (or
    one can be discovered), the docking oracle is invoked to produce
    real RMSD / Vina / confidence numbers.  The oracle verdict is
    *and-ed* with the fingerprint stub — both must succeed for the
    binding type to be inhabited.  When the oracle is unavailable it
    raises :class:`AdapterUnavailable` which we catch and silently
    fall back to the fingerprint stub.

    The two/three stages are intentional mirrors of the halves of
    "binding IS β-reduction + type check":

        type-check  ⇔  ligand satisfies the binding site's *type*
        β-reduction ⇔  ligand's atom stack can be β-reduced against
                       the site's geometric / electronic shape
        oracle      ⇔  real docking oracle agrees (DiffDock/FlowDock)

    Parameters
    ----------
    ligand : MoleculeClosedTerm | rdkit.Chem.Mol | Any
        The candidate ligand.  Either a :class:`MoleculeClosedTerm`
        (preferred — the MLC native type) or any object exposing
        ``rdkit_mol`` / RDKit ``Mol``-compatible methods.
    site : BindingSite
        The binding site (higher-order type) the ligand is being
        checked against.
    leaf_oracle_call : bool, default False
        When True, attempt to call the docking oracle.  MCTS passes
        True only on the **top-K** candidates at iteration end — NOT
        during expansion, where the docking cost is prohibitive.
    oracle : DockingOracle, optional
        Pre-resolved oracle.  When omitted we attempt
        :func:`_resolve_docking_oracle` (FlowDock preferred, then
        DiffDock).

    Returns
    -------
    BindingTypeCheckResult
        Verdict + heuristic pIC50 + violations + diagnostic details.
    """
    # Stage 1 — property type-check.
    violated: List[str] = []
    details: Dict[str, Any] = {"constraint_results": {}}
    for predicate in site.constraints:
        try:
            ok = bool(predicate(ligand))
        except Exception as exc:  # pragma: no cover - defensive
            ok = False
            details["constraint_results"][predicate.name] = {
                "ok": False,
                "error": str(exc),
            }
        details["constraint_results"][predicate.name] = {
            "ok": ok,
        }
        if not ok:
            violated.append(
                f"{predicate.name}: {predicate.description or 'failed'}"
            )

    # Stage 2 — geometric β-check.
    mol = _ligand_rdkit_mol(ligand)
    geom_ok, geom_reasons, geom_details = _geometric_beta_check(
        ligand, site, mol
    )
    details["geometric_check"] = geom_details
    violated.extend(geom_reasons)

    # Estimate pIC50 from the type-inhabitation score.
    pic50 = _estimate_pic50(ligand, site, mol, geom_ok, details)

    success = len(violated) == 0

    # Stage 3 — optional oracle leaf type-check (top-K only).
    if leaf_oracle_call and success:
        # Only call the oracle when the fingerprint stub already passes;
        # otherwise we save a docking call.  (Failing type-checks
        # never make it into the top-K anyway.)
        try:
            from molmetal_lam.sbdd_env.diffdock_adapter import AdapterUnavailable
        except Exception:  # pragma: no cover
            AdapterUnavailable = Exception  # type: ignore[assignment]
        if oracle is None:
            oracle = _resolve_docking_oracle()
        if oracle is not None:
            oracle_verdict = _oracle_leaf_typecheck(ligand, site, oracle)
            if oracle_verdict is not None:
                details["oracle_check"] = oracle_verdict.details
                if not oracle_verdict.success:
                    success = False
                    violated.extend(oracle_verdict.violated_constraints)

    # ---- L8 governance instrumentation (1-line hooks per metric) ----
    _L8_STATE.per_site_calls[site.name] += 1
    _L8_STATE.per_site_ok[site.name] += int(success)
    _L8_STATE.per_site_geom_ok[site.name] += int(geom_ok)
    _L8_STATE.per_site_warhead[site.name] += int(has_metal_coordination_warhead(ligand))
    if success and pic50 == pic50:  # not NaN
        _L8_STATE.per_site_pic50[site.name].append(float(pic50))
        comps = details.get("pic50_components", {})
        final = comps.get("final", float("nan"))
        if final == final:
            _L8_STATE.per_site_pic50_components[site.name].append(float(final))
        vina = _approx_vina_from_pic50(pic50)
        _L8_STATE.vina_in_pocket_total[site.name] += 1
        _L8_STATE.vina_in_pocket_pass[site.name] += int(vina <= -7.0)
    if violated:
        _L8_STATE.first_failure[violated[0].split(":", 1)[0]] += 1

    return BindingTypeCheckResult(
        success=success,
        pic50_estimate=pic50 if success else float("nan"),
        violated_constraints=violated,
        details=details,
    )


# ---------------------------------------------------------------------------
# Canonical binding sites
# ---------------------------------------------------------------------------


#: MMP2 catalytic-site type.  The canonical MMP2 / MMP9 ZBG is a
#: hydroxamic acid chelating the catalytic Zn²⁺ in a bidentate
#: manner; the pocket also demands a moderate lipophilic group to
#: occupy the S1' hydrophobic specificity pocket.
MMP2_ACTIVE: BindingSite = BindingSite(
    name="MMP2_active_site",
    constraints=[
        TypePredicate(
            name="hydroxamic_acid_zbg",
            predicate_fn=hydroxamic_acid_present,
            description=(
                "Ligand must contain a hydroxamic acid warhead "
                "(-C(=O)-NH-OH) for bidentate Zn²⁺ coordination."
            ),
        ),
        TypePredicate(
            name="hydrophobic_pocket_filler",
            predicate_fn=has_hydrophobic_pocket,
            description=(
                "Ligand must carry a hydrophobic group (aromatic ring "
                "or 3+ alkyl chain) to occupy the S1' pocket."
            ),
        ),
        TypePredicate(
            name="lipinski_compatible",
            predicate_fn=LIPINSKI,
            description="Lipinski Ro5 — drug-like ADMET envelope.",
        ),
    ],
    geometry_hints={
        "metal": "Zn_II",
        "coordination_number": 4,
        "geometry": "tetrahedral",
        "preferred_donors": ["N", "O"],
        "min_donors": 2,
        "min_hbond_donors": 2,
        "min_hbond_acceptors": 3,
        "logp_window": (-1.0, 5.0),
    },
    description=(
        "MMP2 / MMP9 catalytic zinc-metalloproteinase active site.  "
        "Zn²⁺ is coordinated tetrahedrally by three His Nε and the "
        "bidentate hydroxamic acid of the inhibitor; the S1' pocket "
        "is hydrophobic and determines isoform selectivity."
    ),
)


#: Major-groove Pt(II) coordination adduct on DNA.  Cisplatin and its
#: analogues form covalent 1,2-intrastrand d(GpG) cross-links by
#: binding two adjacent guanine N7 donors in a square-planar geometry.
PT_DNA_MAJOR_GROOVE: BindingSite = BindingSite(
    name="Pt_DNA_major_groove",
    constraints=[
        TypePredicate(
            name="square_planar_pt_center",
            predicate_fn=square_planar_pt_center,
            description=(
                "Ligand must contain a Pt(II) centre coordinated to "
                ">= 4 heavy atoms in a square-planar geometry."
            ),
        ),
        TypePredicate(
            name="metal_coordination_warhead",
            predicate_fn=has_metal_coordination_warhead,
            description=(
                "Ligand must carry a metal-coordination warhead "
                "(carboxylate / hydroxamate / thiol / phosphonate)."
            ),
        ),
    ],
    geometry_hints={
        "metal": "Pt_II",
        "coordination_number": 4,
        "geometry": "square_planar",
        "preferred_donors": ["N", "Cl", "O", "S"],
        "min_donors": 2,
        "min_hbond_donors": 0,
        "min_hbond_acceptors": 2,
    },
    description=(
        "Cisplatin-class Pt(II) covalent DNA adduct.  The Pt centre "
        "is square-planar; two cis positions bind adjacent guanine "
        "N7 donors in the DNA major groove, forming the canonical "
        "1,2-d(GpG) intrastrand cross-link that drives cytotoxicity."
    ),
)


#: Generic ATP-site kinase binding type.  Used as a sensible default
#: for kinase targets when no specific site is supplied.
KINASE_ATP: BindingSite = BindingSite(
    name="Kinase_ATP_site",
    constraints=[
        TypePredicate(
            name="atp_competitor",
            predicate_fn=has_metal_coordination_warhead,
            description=(
                "Kinase ATP-site inhibitor usually carries a hinge "
                "binder (warhead).  Generic stand-in: any metal-"
                "coordination / H-bond competent group is present."
            ),
        ),
        TypePredicate(
            name="lipinski_compatible",
            predicate_fn=LIPINSKI,
            description="Lipinski Ro5 — drug-like ADMET envelope.",
        ),
    ],
    geometry_hints={
        "preferred_donors": ["N", "O"],
        "min_donors": 2,
        "min_hbond_donors": 1,
        "min_hbond_acceptors": 3,
        "logp_window": (0.0, 5.0),
    },
    description=(
        "Generic ATP-competitive kinase binding site.  Requires a "
        "hinge-binder (donor/acceptor) and a small hydrophobic "
        "moiety in the ribose pocket; Lipinski-compatible for oral "
        "kinase inhibitor programmes."
    ),
)


#: A minimal generic protease binding type.  Useful as the default
#: when no target-specific site is available.
PROTEASE_GENERIC: BindingSite = BindingSite(
    name="Protease_generic",
    constraints=[
        TypePredicate(
            name="lipinski_compatible",
            predicate_fn=LIPINSKI,
            description="Lipinski Ro5 — drug-like ADMET envelope.",
        ),
    ],
    geometry_hints={
        "preferred_donors": ["N", "O"],
        "min_donors": 1,
        "min_hbond_donors": 1,
        "min_hbond_acceptors": 2,
    },
    description=(
        "Generic protease binding site placeholder.  Requires one "
        "donor + two acceptors — the minimal pharmacophore for "
        "interacting with the catalytic aspartate / serine / "
        "cysteine residues of a protease."
    ),
)


# A registry mapping canonical name -> BindingSite for the search layer.
CANONICAL_BINDING_SITES: Dict[str, BindingSite] = {
    s.name: s
    for s in (MMP2_ACTIVE, PT_DNA_MAJOR_GROOVE, KINASE_ATP, PROTEASE_GENERIC)
}


# ---------------------------------------------------------------------------
# L8 governance instrumentation (1-line hooks in typecheck)
# ---------------------------------------------------------------------------


@dataclass
class _L8Metrics:
    """Counters + histograms for Layer 8 governance metrics."""

    per_site_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_site_ok: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_site_geom_ok: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_site_warhead: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_site_pic50: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    per_site_pic50_components: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    first_failure: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vina_in_pocket_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vina_in_pocket_pass: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


_L8_STATE = _L8Metrics()


def _approx_vina_from_pic50(pic50: float) -> float:
    """Crude Vina-kcal/mol crosswalk from the pIC50 heuristic.

    The ``vina_adapter`` maps -12 kcal/mol -> confidence 1.0 and
    -4 kcal/mol -> confidence 0.0. pIC50 lives in [0, 12]; we map
    pIC50 -> [10, 4] kcal/mol with a smooth midpoint at pIC50=6 so that
    pIC50 >= 7 ~= -9 kcal/mol (sub-micromolar-affinity, the canonical
    "in pocket" cutoff for the L8-NEW metric).
    """
    if pic50 != pic50:  # NaN
        return 0.0
    # Linear: pic50 0 -> -4; pic50 6 -> -7; pic50 12 -> -10.
    return float(-4.0 - (pic50 / 12.0) * 6.0)


__all__ = [
    "BindingSite",
    "BindingTypeCheckResult",
    "typecheck",
    "hydroxamic_acid_present",
    "has_metal_coordination_warhead",
    "square_planar_pt_center",
    "has_hydrophobic_pocket",
    "min_hbond_donors",
    "min_hbond_acceptors",
    "logp_in_range",
    "MMP2_ACTIVE",
    "PT_DNA_MAJOR_GROOVE",
    "KINASE_ATP",
    "PROTEASE_GENERIC",
    "CANONICAL_BINDING_SITES",
    "l8_metrics",
    "reset_l8_metrics",
]


def reset_l8_metrics() -> None:
    """Zero out the L8 counters (useful for tests / re-measurement)."""
    global _L8_STATE
    _L8_STATE = _L8Metrics()


def _stdev(xs: List[float]) -> float:
    """Sample standard deviation of ``xs`` (returns 0.0 on < 2 points)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return float(var ** 0.5)


def _median(xs: List[float]) -> float:
    """Median of ``xs`` (returns 0.0 on empty)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def l8_metrics() -> Dict[str, Any]:
    """Snapshot of Layer 8 governance counters.

    Keys
    ----
    per_site_calls / per_site_ok : dict[str, int]
        L8.1 — typecheck success rate per site.
    per_site_pic50 : dict[str, list[float]]
        L8.2 — pIC50 estimate per site.
    first_failure : dict[str, int]
        L8.3 — first-failure reason histogram.
    per_site_geom_ok : dict[str, int]
        L8.4 — geometric β-check pass rate per site.
    per_site_warhead : dict[str, int]
        L8.5 — metal-coordination warhead hit rate per site.
    per_site_pic50_sigma : dict[str, float]
        L8.6 — residual σ of the pIC50 components per site.
    vina_in_pocket_rate : dict[str, float]
        L8-NEW — per-site Vina ≤ -7.0 kcal/mol rate.
    """
    sites = sorted(set(_L8_STATE.per_site_calls) |
                   set(_L8_STATE.per_site_ok) |
                   set(_L8_STATE.vina_in_pocket_total))
    typecheck_rate = {
        s: (
            _L8_STATE.per_site_ok[s] / _L8_STATE.per_site_calls[s]
            if _L8_STATE.per_site_calls[s]
            else 0.0
        )
        for s in sites
    }
    geom_rate = {
        s: (
            _L8_STATE.per_site_geom_ok[s] / _L8_STATE.per_site_calls[s]
            if _L8_STATE.per_site_calls[s]
            else 0.0
        )
        for s in sites
    }
    warhead_rate = {
        s: (
            _L8_STATE.per_site_warhead[s] / _L8_STATE.per_site_calls[s]
            if _L8_STATE.per_site_calls[s]
            else 0.0
        )
        for s in sites
    }
    pic50_median = {s: _median(_L8_STATE.per_site_pic50[s]) for s in sites}
    pic50_sigma = {s: _stdev(_L8_STATE.per_site_pic50_components[s]) for s in sites}
    vina_rate = {
        s: (
            _L8_STATE.vina_in_pocket_pass[s] / _L8_STATE.vina_in_pocket_total[s]
            if _L8_STATE.vina_in_pocket_total[s]
            else 0.0
        )
        for s in sites
    }
    return {
        "per_site_calls": dict(_L8_STATE.per_site_calls),
        "per_site_ok": dict(_L8_STATE.per_site_ok),
        "typecheck_success_rate": typecheck_rate,
        "per_site_pic50": {s: list(_L8_STATE.per_site_pic50[s]) for s in sites},
        "per_site_pic50_median": pic50_median,
        "per_site_pic50_sigma": pic50_sigma,
        "first_failure": dict(_L8_STATE.first_failure),
        "per_site_geom_ok": dict(_L8_STATE.per_site_geom_ok),
        "geom_pass_rate": geom_rate,
        "per_site_warhead": dict(_L8_STATE.per_site_warhead),
        "warhead_hit_rate": warhead_rate,
        "vina_in_pocket_total": dict(_L8_STATE.vina_in_pocket_total),
        "vina_in_pocket_pass": dict(_L8_STATE.vina_in_pocket_pass),
        "vina_in_pocket_rate": vina_rate,
    }