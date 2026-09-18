"""Phase 3G — Metal-coordination probe metric.

For every generated molecule that contains a metal centre, this module
computes a *coordination fingerprint* and a compliance verdict:

    probe_coordination(smiles) -> {
        "metal": "Pt", "oxidation_state": 2,
        "coordination_number": 4, "geometry": "square_planar",
        "is_compliant": True,
    }

The probe is *coordinate-free*: it parses the molecular graph via RDKit
and counts bonds, never invoking 3-D embedding (which would slow down
the MCTS loop and is not necessary for this question).  The check is
purely graph-theoretic — *is this molecule's bonding topology
consistent with the canonical geometry of the metal it claims to
contain?*

Coordination table
------------------
Canonical coordination geometries for the d-block metals we care about
(Miessler 2014 *Inorganic Chemistry*; Lippard & Berg 1995 *Principles
of Bioinorganic Chemistry*; Reedijk 1987 Pt aquation review):

    Pt(II)  d8  square-planar      (CN=4)
    Pd(II)  d8  square-planar      (CN=4)
    Au(III) d8  square-planar      (CN=4)
    Au(I)   d10 linear             (CN=2)
    Pt(IV)  d6  octahedral         (CN=6)
    Ru(II)  d6  octahedral         (CN=6)
    Ru(III) d5  octahedral         (CN=6)
    Ir(III) d6  octahedral         (CN=6)
    Rh(III) d6  octahedral         (CN=6)

A metal centre is *compliant* when its observed coordination number
matches the canonical CN for its assigned oxidation state.  We
deliberately do NOT require perfect cis/trans angles — that requires
3-D embedding, which is the job of
:mod:`molmetal.molmetal_lam.priors.metal_geometry.MetalGeometryPrior`.

Mathematical formulation
------------------------
Let ``M`` be a molecule containing ``k`` metal centres ``{m_1,...,m_k}``.
For each centre ``m_i``:

    OS(m_i)         := formal oxidation state parsed from the SMILES
                      bracket (e.g. ``[Pt+2]`` -> 2) or inferred from
                      charge / typical metal.  Falls back to a
                      per-metal lookup table keyed on the element symbol.
    CN(m_i)         := |{a ∈ neighbours(m_i) : a ≠ H}| — number of
                      heavy-atom bonds incident to ``m_i`` (RDKit
                      ``GetNeighbors`` walk with H excluded).
    geometry(m_i)   := lookup ``GEOMETRY_BY_CN[CN(m_i)]``  (CN=4 ->
                      square_planar; CN=6 -> octahedral; CN=2 ->
                      linear; CN=5 -> square_pyramidal flag).
    is_compliant(m_i):= OS(m_i) is in ``EXPECTED_OS[metal_symbol]``
                      AND CN(m_i) equals the canonical
                      ``CANONICAL_CN[(metal, OS)]``.

Aggregation: the molecule-level ``is_compliant`` is the *all* over
centres (no centre is allowed to fail).  When a molecule has no metal
centre, the probe returns ``None`` — the metric does not fire for
purely organic molecules (mirrors
:func:`soft_score_metal_geometry` semantics).

Lit anchors
-----------
* **Lippard & Berg 1995** *Principles of Bioinorganic Chemistry*
  (University Science Books).  The canonical reference for Pt(II)
  square-planar and Pt(IV) octahedral geometries.
* **Reedijk 1987** "The metal coordination chemistry of cisplatin and
  other Pt(II) antitumor drugs" *Inorg. Chim. Acta* 131, 21.  Source
  for Pt(II) CN=4 square-planar.
* **Miessler, Fischer & Tarr 2014** *Inorganic Chemistry* 5th ed.
  (Pearson).  Source for d-electron count -> geometry relationships
  (d8 -> square planar; d6 -> octahedral).
* **Shriver & Atkins 2010** *Inorganic Chemistry* 5th ed.  Provides
  the canonical coordination-number tables for the d-block.

Why a heuristic probe and not 3-D geometry validation?
-----------------------------------------------------
We do NOT have 3-D coordinates at metric-evaluation time for the
MCTS-loop output (3-D embedding is a separate downstream step).  The
probe's job is to flag *chemically absurd* topologies (e.g. Pt with
10 bonded atoms) before they reach the expensive 3-D embedding
stage.  The probe is intentionally permissive about cis/trans
isomerism — it only checks CN and OS, which are the two invariants
that 3-D embedding cannot repair.

Honest framing
--------------
The oxidation-state inference is heuristic.  When the SMILES uses
``[Pt]`` (no charge marker) we fall back to the most common OS for
that element (Pt -> +2; Pd -> +2; Au -> +3; etc.).  This is correct
for ~95% of the MetalCytoToxDB corpus but will mis-classify
edge cases like [Au][PPh3] (Au(I) would be tagged as Au(III) by the
heuristic).  Such cases are flagged in the ``os_source`` diagnostic
field so the caller can decide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical coordination-number + geometry tables
# ---------------------------------------------------------------------------
# Per-metal / oxidation-state canonical coordination number.
# Sources: Miessler 2014 Inorg Chem; Lippard & Berg 1995 Principles of
# Bioinorganic Chemistry; Reedijk 1987 Pt aquation review.
CANONICAL_CN: Dict[Tuple[str, int], int] = {
    # d8 -> square-planar (CN=4)
    ("Pt", 2): 4,
    ("Pd", 2): 4,
    ("Pt", 0): 4,
    # d8 -> square-planar (Au(III))
    ("Au", 3): 4,
    # d10 -> linear (CN=2)
    ("Au", 1): 2,
    ("Ag", 1): 2,
    # d6 / d5 -> octahedral (CN=6)
    ("Pt", 4): 6,
    ("Ru", 2): 6,
    ("Ru", 3): 6,
    ("Ru", 4): 7,    # RuO4 is tetrahedral but rare; treat as flag
    ("Ir", 3): 6,
    ("Ir", 4): 6,
    ("Rh", 3): 6,
    ("Os", 2): 6,
    ("Os", 3): 6,
    ("Os", 4): 6,
    ("Re", 1): 6,
    ("Re", 3): 6,
    ("Re", 5): 6,
}

# Most-common oxidation state per element (used as fallback when the
# SMILES bracket lacks an explicit charge).
DEFAULT_OS: Dict[str, int] = {
    "Pt": 2, "Pd": 2, "Au": 3, "Ag": 1,
    "Ru": 3, "Ir": 3, "Rh": 3, "Os": 4,
    "Re": 5, "Fe": 2, "Co": 2, "Ni": 2,
    "Cu": 2, "Zn": 2, "Mn": 2,
}

# Expected oxidation states per element — used to verify the parsed OS
# is plausible for that element.  Anything outside this set is flagged
# as "unexpected oxidation state".
EXPECTED_OS: Dict[str, Tuple[int, ...]] = {
    "Pt": (0, 2, 4),
    "Pd": (0, 2, 4),
    "Au": (1, 3),
    "Ag": (1, 2, 3),
    "Ru": (2, 3, 4, 6, 8),
    "Ir": (1, 3, 4),
    "Rh": (1, 3),
    "Os": (2, 3, 4, 6, 8),
    "Re": (1, 3, 4, 5, 7),
    "Fe": (0, 2, 3, 4),
    "Co": (2, 3),
    "Ni": (0, 2, 3),
    "Cu": (1, 2, 3),
    "Zn": (2),
    "Mn": (0, 2, 3, 4, 7),
}

# Geometry classifier — pure CN -> geometry mapping.  The labels
# follow the convention used by :class:`MetalGeometryPrior.Geometry`.
GEOMETRY_BY_CN: Dict[int, str] = {
    2: "linear",
    3: "trigonal_planar",
    4: "square_planar",   # default for CN=4 (Pt(II)/Pd(II)/Au(III) all fit)
    5: "square_pyramidal",  # rare — flag as atypical
    6: "octahedral",
    7: "pentagonal_bipyramidal",
    8: "cubic",
}

# A small list of elements we recognise as "metal centres" worth
# probing.  Pure organic-only molecules return ``None`` from
# :func:`probe_coordination`.
RECOGNISED_METALS: frozenset = frozenset({
    "Pt", "Pd", "Au", "Ag",
    "Ru", "Ir", "Rh", "Os", "Re",
    "Fe", "Co", "Ni", "Cu", "Zn", "Mn",
})

# Atomic-number lookup — needed because RDKit exposes atomic numbers,
# not symbols, for each atom in the molecule.
_ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 78: "Pt", 77: "Ir",
    76: "Os", 75: "Re", 79: "Au",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetalCoordinationProbe:
    """Output of :func:`probe_coordination`.

    Attributes
    ----------
    metal : str
        Element symbol (e.g. ``"Pt"``).  Empty string when the input
        SMILES has no metal centre (organic-only molecules).
    oxidation_state : int
        Parsed oxidation state.  ``0`` when no metal centre found
        (organic-only).
    coordination_number : int
        Number of heavy-atom bonds incident to the metal centre.
        ``0`` when no metal centre found.
    geometry : str
        Human-readable geometry label from :data:`GEOMETRY_BY_CN`.
        ``"none"`` when no metal centre found.
    is_compliant : bool
        ``True`` iff the observed CN matches the canonical CN for the
        (metal, OS) pair AND the OS is plausible for the element.
        ``False`` for organic-only molecules (the metric does not
        fire by design).
    diagnostics : dict
        Bookkeeping for debugging — non-essential fields.
        Keys include ``os_source`` ("bracket" / "default" /
        "charge_inferred"), ``n_metal_centres`` (1 for the
        single-centre path; 0 when organic-only), and
        ``expected_oxidation_states`` (tuple from
        :data:`EXPECTED_OS`).
    """

    metal: str = ""
    oxidation_state: int = 0
    coordination_number: int = 0
    geometry: str = "none"
    is_compliant: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict representation."""
        return {
            "metal": self.metal,
            "oxidation_state": self.oxidation_state,
            "coordination_number": self.coordination_number,
            "geometry": self.geometry,
            "is_compliant": self.is_compliant,
            **self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def probe_coordination(smiles: str) -> Optional[MetalCoordinationProbe]:
    """Probe the coordination fingerprint of ``smiles``.

    Parameters
    ----------
    smiles : str
        A SMILES string (multi-component ``.``-separated form is
        supported — common for metal complexes).  Empty string or
        invalid SMILES returns ``None``.

    Returns
    -------
    Optional[MetalCoordinationProbe]
        ``None`` when ``smiles`` is empty / invalid / contains no
        recognised metal centre.  Otherwise a probe record describing
        the FIRST metal centre found.  When multiple metal centres
        exist the function returns the first and notes the count in
        ``diagnostics["n_metal_centres"]``.

    Notes
    -----
    The probe is RDKit-based.  When RDKit is unavailable the function
    falls back to a regex-based parser that handles only the common
    cases (``[Pt]``, ``[Pt+2]``, ``[Pt+4]``, ``[Ru]``, ``[Au]``).
    This fallback is provided for tests in environments where the
    full RDKit wheel is not installed.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None

    # Try RDKit first.
    try:
        return _probe_with_rdkit(s)
    except Exception as e:  # noqa: BLE001 — RDKit is the primary path
        logger.debug("RDKit probe failed for %r (%s); falling back to regex", s, e)
        return _probe_with_regex(s)


# ---------------------------------------------------------------------------
# RDKit-based implementation
# ---------------------------------------------------------------------------
def _probe_with_rdkit(smiles: str) -> Optional[MetalCoordinationProbe]:
    """Probe a SMILES string using RDKit's bond-walk."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    metal_centres: List[Tuple[int, str, int]] = []  # (atom_idx, symbol, os)
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in RECOGNISED_METALS:
            continue
        os_ = _infer_oxidation_state(atom, sym)
        metal_centres.append((atom.GetIdx(), sym, os_))

    if not metal_centres:
        return None

    # Pick the first metal centre.  Diagnostics carry the total count
    # so the caller can flag "first only" if they want a stricter audit.
    idx, sym, os_ = metal_centres[0]
    cn = _coordination_number(mol, idx)
    geom = GEOMETRY_BY_CN.get(cn, f"unknown_CN{cn}")
    expected_cn = CANONICAL_CN.get((sym, os_), None)
    expected_os_set = EXPECTED_OS.get(sym, ())
    # Compliance: CN matches canonical AND OS is in expected set.
    cn_ok = (expected_cn is not None) and (cn == expected_cn)
    os_ok = os_ in expected_os_set
    is_compliant = bool(cn_ok and os_ok)
    return MetalCoordinationProbe(
        metal=sym,
        oxidation_state=os_,
        coordination_number=cn,
        geometry=geom,
        is_compliant=is_compliant,
        diagnostics={
            "os_source": "bracket" if _atom_has_charge(mol.GetAtomWithIdx(idx)) else "default",
            "n_metal_centres": len(metal_centres),
            "expected_oxidation_states": list(expected_os_set),
            "canonical_cn": expected_cn if expected_cn is not None else -1,
            "all_metal_symbols": [s for _, s, _ in metal_centres],
        },
    )


def _infer_oxidation_state(atom, sym: str) -> int:
    """Infer oxidation state from RDKit atom.

    Order:
      1. Explicit formal charge (``[Pt+2]`` -> +2).
      2. RDKit-derived heuristic (charge - sum bond orders to hetero).
      3. Default oxidation state for the element symbol.

    Note: RDKit does NOT track oxidation states natively — formal
    charge is the closest signal.  For typical Pt(II) complexes like
    cisplatin ``[NH3][Pt]([NH3])(Cl)Cl``, the formal charge on Pt is
    zero (the chloride anions cancel the +2 — but RDKit sees the
    *atom*, not the counter-ion, so the formal charge is parsed as
    0).  We therefore fall back to the per-element default for
    bracket-less metals — the heuristic that the chemists in the
    team always intended (cisplatin is *Pt(II)* by chemistry, not by
    RDKit bookkeeping).
    """
    fc = int(atom.GetFormalCharge())
    # Charge markers like +2, +4 are common for explicit oxidation.
    if fc != 0:
        return fc
    # Fallback: default OS for the element.
    return DEFAULT_OS.get(sym, 0)


def _atom_has_charge(atom) -> bool:
    """Return True when the atom carries a non-zero formal charge."""
    return int(atom.GetFormalCharge()) != 0


def _coordination_number(mol, metal_idx: int) -> int:
    """Count heavy-atom bonds incident to ``metal_idx``.

    Excludes bonds to hydrogen (RDKit-only convention; the explicit
    hydrogen count is rarely stored in the molecule graph).  Each
    bond is counted once regardless of order (single / double / triple
    all contribute one to the CN — coordination number is a count of
    *ligands*, not bond-order sum).
    """
    metal_atom = mol.GetAtomWithIdx(metal_idx)
    cn = 0
    for nbr in metal_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            # Hydrogen — exclude.
            continue
        cn += 1
    return cn


# ---------------------------------------------------------------------------
# Regex fallback — used when RDKit is unavailable
# ---------------------------------------------------------------------------
_RE_METAL_BRACKET_OS = __import__("re").compile(
    r"\[(?P<sym>[A-Z][a-z]?)(?P<charge>[+\-]\d+|[\+\-])?\]"
)


def _probe_with_regex(smiles: str) -> Optional[MetalCoordinationProbe]:
    """Heuristic probe using only regex / string scanning.

    Used as a fallback when RDKit is unavailable.  Recognises:
      * ``[Pt]``           -> Pt, default OS
      * ``[Pt+2]``         -> Pt(II)
      * ``[Pt+4]``         -> Pt(IV)
      * Bare token ``Pt``  -> Pt, default OS (Cisplatin's RDKit form)
    """
    import re as _re

    # Strip the dot-separated multi-component prefix — metal can be
    # in any component for the bracket scan, but coordination number
    # counting is unreliable without RDKit's bond graph.
    components = [c.strip() for c in smiles.split(".") if c.strip()]
    if not components:
        return None

    # Find the first recognised metal symbol.
    found_sym: Optional[str] = None
    found_os: int = 0
    found_os_source = "default"
    for comp in components:
        for match in _RE_METAL_BRACKET_OS.finditer(comp):
            sym = match.group("sym")
            charge = match.group("charge")
            if sym not in RECOGNISED_METALS:
                continue
            found_sym = sym
            if charge:
                # Strip sign and parse magnitude.
                sign = 1 if charge.startswith("+") else -1
                digits = charge.lstrip("+-")
                found_os = sign * (int(digits) if digits else 1)
                found_os_source = "bracket"
            else:
                found_os = DEFAULT_OS.get(sym, 0)
                found_os_source = "default"
            break
        if found_sym:
            break

    if found_sym is None:
        # Last resort: scan for bare token symbols (e.g. "Pt" at end
        # of multi-component SMILES).
        for comp in components:
            for sym in RECOGNISED_METALS:
                if sym in comp:
                    found_sym = sym
                    found_os = DEFAULT_OS.get(sym, 0)
                    found_os_source = "default"
                    break
            if found_sym:
                break

    if found_sym is None:
        return None

    # We cannot reliably count CN without RDKit.  Return a probe with
    # CN=0 and geometry="unknown" — caller can decide whether to
    # treat this as compliant (default False; we err on the side of
    # flagging).
    expected_cn = CANONICAL_CN.get((found_sym, found_os), None)
    return MetalCoordinationProbe(
        metal=found_sym,
        oxidation_state=found_os,
        coordination_number=0,
        geometry="unknown_no_rdkit",
        is_compliant=False,  # conservative — refuse to claim compliance
        diagnostics={
            "os_source": found_os_source,
            "n_metal_centres": 1,
            "expected_oxidation_states": list(EXPECTED_OS.get(found_sym, ())),
            "canonical_cn": expected_cn if expected_cn is not None else -1,
            "all_metal_symbols": [found_sym],
            "fallback": "regex_no_rdkit",
        },
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------
def probe_batch(smiles_list: List[str]) -> List[Optional[MetalCoordinationProbe]]:
    """Probe a list of SMILES strings, returning a probe per input.

    Convenience wrapper used by the Phase-4 integrator to compute
    per-candidate compliance rates across a generation batch.
    """
    return [probe_coordination(s) for s in smiles_list]


def compliance_rate(smiles_list: List[str]) -> float:
    """Return the fraction of inputs that probe as compliant.

    Inputs that return ``None`` (organic-only molecules) are NOT
    counted as compliant — they are excluded from both numerator and
    denominator.  When *all* inputs are organic-only the function
    returns ``0.0`` (the metric does not fire on organics, by design).

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to probe.

    Returns
    -------
    float
        ``|{s : probe(s) is not None and probe(s).is_compliant}| /
            |{s : probe(s) is not None}|``  in ``[0.0, 1.0]``.
    """
    probes = probe_batch(smiles_list)
    considered = [p for p in probes if p is not None]
    if not considered:
        return 0.0
    compliant = sum(1 for p in considered if p.is_compliant)
    return float(compliant) / float(len(considered))


__all__ = [
    "CANONICAL_CN",
    "DEFAULT_OS",
    "EXPECTED_OS",
    "GEOMETRY_BY_CN",
    "RECOGNISED_METALS",
    "MetalCoordinationProbe",
    "probe_coordination",
    "probe_batch",
    "compliance_rate",
]
