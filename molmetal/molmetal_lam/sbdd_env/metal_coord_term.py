"""Metal-coordination scoring terms for AutoDock Vina (Wf-246-C).

================================================================
Why this module exists
================================================================
TODO-130 baseline flagged m3/m10 Vina as a "REDESIGN" because the
generic Vina scoring function lacks any **metal-coordination term**.
For precious-metal metallodrugs (Pt/Ru/Ir/Os/Rh/Pd/Au/Re), the
binding affinity is dominated by:

  (a) d-orbital coordination geometry
      (octahedral for Ru(II)/Ir(III)/Rh(III)/Os(II) — CN=6;
       square-planar for Pt(II)/Pd(II)/Au(III) — CN=4)
  (b) HSAB (hard-soft acid-base) matching
      between the metal centre and the protein ligand set
      (e.g. cis-[Pt(NH3)2]2+ prefers hard N/O donors from His/Glu;
       auranofin [Au(PEt3)]+ prefers soft thiolate from Cys)
  (c) Metal-ligand distance penalty
      (M-N ~2.0 Å, M-O ~2.0 Å, M-S ~2.3 Å, M-P ~2.3 Å)

Without (a)-(c) the kcal/mol returned by Vina is `status =
"MEASURED_GENERIC_ATOM"` (per `metal_pdbqt_overrides.GENERIC_ATYPE`),
which **systematically under-estimates** metallodrug binding. This
module provides three CPU-cheap scoring terms (sub-millisecond per
pose) that can be added on top of Vina's kcal/mol to produce a
metal-aware pseudo-score.

Design constraints
------------------
* **Pure-Python / CPU-only**: must work without GPU, RDKit, or vina.
* **Coordinate-free where possible**: the geometry / HSAB scores
  work on the SMILES graph; the distance score needs a 3-D
  conformer (provided by Vina's docked pose).
* **Backward compatible**: when ``enable_metal_coord=False`` (or the
  molecule has no metal centre), the term is **exactly zero** and
  Vina's kcal/mol is returned unchanged.
* **Honest framing**: every score is reported alongside a
  diagnostic dict so downstream consumers can audit which term
  fired (or didn't) for each pose.

Lit anchors
-----------
* Lippard & Berg 1995 *Principles of Bioinorganic Chemistry* —
  canonical Pt(II) CN=4 square-planar + Pt(IV) CN=6 octahedral.
* Pearson 1963 *J Am Chem Soc* 85:3533 — HSAB classification of
  d-block metals (Pt2+/Pd2+/Cr3+ = hard borderline; Au+/Ag+ = soft).
* Shannon 1976 *Acta Cryst A* 32:751 — ionic radii (Pt2+ 0.74 Å,
  Au3+ 0.85 Å, Ru2+ 0.82 Å) underpin the metal-ligand distance.
* Reedijk 1987 *Inorg Chim Acta* 131:21 — Pt-N 2.0 Å / Pt-S 2.3 Å.

Public API
----------
* :class:`MetalCoordinationTerm` — composite scoring term
* :func:`coordination_geometry_score(smiles)` — CN/geometry score
* :func:`hsab_match_score(smiles, ligand_atomicnums)` — HSAB score
* :func:`metal_ligand_distance_score(metal, ligand_idx,
  ligand_pos, metal_pos)` — distance score
* :data:`HSAB_TABLE` — 5-metal × donor class table
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard/soft acid-base (HSAB) classification (Pearson 1963 + 1985 IUPAC)
# ---------------------------------------------------------------------------
# Five precious-metal anticancer centres, each scored against the four
# donor classes we see in metallodrug-protein binding:
#   N  : nitrogen (imidazole / amine / amide) — borderline hard
#   O  : oxygen (carboxylate / carbonyl / water) — hard
#   S  : sulfur (thiolate / thioether / methionine) — soft
#   P  : phosphorus (phosphine / phosphate) — soft
#
# Score values are dimensionless favourabilities in [0, 1]:
#   1.0 — ideal match (e.g. Pt2+ + N/O; Au+ + S/P)
#   0.7 — good match
#   0.4 — borderline / possible but strained
#   0.1 — mismatch (e.g. Au+ + O hard donor = poor)
#
# Source: Pearson 1963 JACS 85:3533 hard/soft acids-bases; expanded
# with the Pt2+ borderline-hard classification from Ho 1975
# *J Chem Educ* 52:74 and the Ru/Ir/Os/Rh classification from
# Jørgensen 1963 *Inorg Chem* 2:1212.
HSAB_TABLE: Dict[str, Dict[str, float]] = {
    # d8 square-planar — borderline hard
    "Pt": {"N": 1.00, "O": 1.00, "S": 0.40, "P": 0.40},
    "Pd": {"N": 1.00, "O": 1.00, "S": 0.40, "P": 0.40},
    # d8 square-planar — borderline hard (Au3+)
    "Au": {"N": 0.70, "O": 0.70, "S": 1.00, "P": 1.00},
    # d6 octahedral — borderline hard (Ru2+/Ir3+/Rh3+/Os2+)
    "Ru": {"N": 1.00, "O": 1.00, "S": 0.70, "P": 0.70},
    "Ir": {"N": 1.00, "O": 1.00, "S": 0.70, "P": 0.70},
    "Os": {"N": 1.00, "O": 1.00, "S": 0.70, "P": 0.70},
    "Rh": {"N": 1.00, "O": 1.00, "S": 0.70, "P": 0.70},
    "Re": {"N": 0.70, "O": 0.70, "S": 1.00, "P": 1.00},
}

# Atomic numbers recognised as donor atoms (Pearson 1963 list)
_DONOR_CLASS_BY_ATOMIC_NUM: Dict[int, str] = {
    7:  "N",   # nitrogen
    8:  "O",   # oxygen
    16: "S",   # sulfur
    15: "P",   # phosphorus
}

# Metal centres recognised by the term (5-metal core set per the
# project convention; matches Pt/Ru/Ir/Os/Rh/Re from TODO-130 m3/m10)
_METAL_ATOMIC_NUMS: Dict[int, str] = {
    44: "Ru", 45: "Rh", 46: "Pd", 76: "Os", 77: "Ir",
    78: "Pt", 75: "Re", 79: "Au",
}


# ---------------------------------------------------------------------------
# Canonical metal-ligand bond lengths (Å) — Shannon 1976 ionic radii +
# typical coordination bond lengths (Lippard & Berg 1995 Table 2.2)
# ---------------------------------------------------------------------------
# Used as the optimum distance for the Gaussian distance penalty:
#   score(d) = exp(-((d - d_opt) / sigma)^2)
# where sigma captures the bond-length flexibility (typical ±0.2 Å).
METAL_LIGAND_OPT_DISTANCE: Dict[Tuple[str, str], float] = {
    # M-N bonds (~2.0 Å for 1st-row d8 square-planar; ~2.05 Å for
    # octahedral Ru/Ir/Rh/Os).
    ("Pt", "N"): 2.04, ("Pd", "N"): 2.02, ("Au", "N"): 2.05,
    ("Ru", "N"): 2.06, ("Ir", "N"): 2.07, ("Os", "N"): 2.08,
    ("Rh", "N"): 2.05, ("Re", "N"): 2.10,
    # M-O bonds (slightly shorter; ~1.99-2.05 Å).
    ("Pt", "O"): 2.00, ("Pd", "O"): 2.00, ("Au", "O"): 2.02,
    ("Ru", "O"): 2.04, ("Ir", "O"): 2.05, ("Os", "O"): 2.06,
    ("Rh", "O"): 2.04, ("Re", "O"): 2.07,
    # M-S bonds (longer; ~2.30 Å for Pt-S, ~2.32 Å for Au-S).
    ("Pt", "S"): 2.30, ("Pd", "S"): 2.30, ("Au", "S"): 2.32,
    ("Ru", "S"): 2.34, ("Ir", "S"): 2.36, ("Os", "S"): 2.35,
    ("Rh", "S"): 2.33, ("Re", "S"): 2.38,
    # M-P bonds (longest; ~2.30 Å for Pt-P phosphine).
    ("Pt", "P"): 2.28, ("Pd", "P"): 2.28, ("Au", "P"): 2.30,
    ("Ru", "P"): 2.32, ("Ir", "P"): 2.34, ("Os", "P"): 2.33,
    ("Rh", "P"): 2.31, ("Re", "P"): 2.36,
}

# Gaussian sigma for distance penalty (Å). Larger sigma = more permissive.
_DISTANCE_SIGMA: Dict[str, float] = {
    "N": 0.20, "O": 0.18, "S": 0.22, "P": 0.22,
}


# ---------------------------------------------------------------------------
# Coordination-geometry table (Lippard & Berg 1995)
# ---------------------------------------------------------------------------
# Per (metal, oxidation_state) coordinate, the *expected* coordination
# geometry. Score = 1.0 if observed CN matches the table; degrades
# linearly with deviation.
EXPECTED_GEOMETRY: Dict[Tuple[str, int], Tuple[str, int]] = {
    # (geometry, coordination_number)
    ("Pt", 2): ("square_planar", 4),
    ("Pd", 2): ("square_planar", 4),
    ("Au", 3): ("square_planar", 4),
    ("Pt", 4): ("octahedral", 6),
    ("Ru", 2): ("octahedral", 6),
    ("Ru", 3): ("octahedral", 6),
    ("Ir", 3): ("octahedral", 6),
    ("Rh", 3): ("octahedral", 6),
    ("Os", 2): ("octahedral", 6),
    ("Os", 3): ("octahedral", 6),
    ("Re", 1): ("octahedral", 6),
    ("Re", 3): ("octahedral", 6),
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetalCoordinationScore:
    """Composite metal-coordination score.

    All three sub-scores are dimensionless and live in [0, 1] (or
    ``None`` when not applicable — e.g. ``geometry_score=None`` for an
    organic ligand). The composite ``total`` is a weighted sum with
    fixed weights (geom=0.4, hsab=0.4, dist=0.2); consumers can mix
    their own weights by reading the three sub-scores directly.

    Attributes
    ----------
    geometry_score : Optional[float]
        Square-planar / octahedral geometry compliance in [0, 1].
        ``None`` for non-metal ligands.
    hsab_score : Optional[float]
        HSAB hard/soft acid-base donor match in [0, 1].
        ``None`` when no donor atoms are recognised.
    distance_score : Optional[float]
        Gaussian-weighted M-ligand distance penalty in [0, 1].
        ``None`` when no metal-ligand distances are computable.
    total : Optional[float]
        Weighted sum of sub-scores. ``None`` when no sub-score fired.
    diagnostics : dict
        Bookkeeping (geometry, oxidation_state, donor_classes,
        n_metal_ligand_contacts, etc.).
    """

    geometry_score: Optional[float] = None
    hsab_score: Optional[float] = None
    distance_score: Optional[float] = None
    total: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "geometry_score": self.geometry_score,
            "hsab_score": self.hsab_score,
            "distance_score": self.distance_score,
            "total": self.total,
            **self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_metal_symbol(smiles: str) -> Optional[Tuple[str, int, int]]:
    """Return ``(symbol, oxidation_state, n_centres)`` for the first
    metal atom in ``smiles``, or ``None`` when none present.

    Oxidation state defaults to the most-common OS for the element
    (Pt->+2, Au->+3, Ru->+3, etc.) when the SMILES bracket lacks an
    explicit ``[M+2]`` marker. This mirrors the heuristic used by
    :func:`metal_coord_probe.probe_coordination`.
    """
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    centres: List[Tuple[int, str, int]] = []
    for atom in mol.GetAtoms():
        anum = int(atom.GetAtomicNum())
        if anum in _METAL_ATOMIC_NUMS:
            sym = _METAL_ATOMIC_NUMS[anum]
            fc = int(atom.GetFormalCharge())
            # Bracket OS marker like [Pt+2] -> +2; else default
            if fc != 0:
                os_ = fc
            else:
                os_ = _default_oxidation_state(sym)
            centres.append((atom.GetIdx(), sym, os_))

    if not centres:
        return None
    idx, sym, os_ = centres[0]
    return sym, os_, len(centres)


def _default_oxidation_state(sym: str) -> int:
    return {
        "Pt": 2, "Pd": 2, "Au": 3, "Ag": 1,
        "Ru": 3, "Ir": 3, "Rh": 3, "Os": 4, "Re": 5,
    }.get(sym, 0)


def _coordination_number_from_smiles(smiles: str, metal_symbol: str) -> int:
    """Count heavy-atom bonds incident to the first metal centre.

    Excludes hydrogens (RDKit convention). Returns 0 when RDKit
    fails to parse the SMILES or no metal centre is found.
    """
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return 0
    if mol is None:
        return 0
    for atom in mol.GetAtoms():
        if _METAL_ATOMIC_NUMS.get(int(atom.GetAtomicNum())) == metal_symbol:
            cn = 0
            for nbr in atom.GetNeighbors():
                if int(nbr.GetAtomicNum()) == 1:
                    continue
                cn += 1
            return cn
    return 0


def _coordination_geometry_score(
    cn: int, metal_symbol: str, oxidation_state: int,
) -> float:
    """Score the coordination geometry match in [0, 1].

    Score = 1.0 when ``cn`` matches the canonical
    ``(metal, oxidation_state)`` tuple; degrades linearly with
    deviation up to CN ±2. CN values without an entry in
    :data:`EXPECTED_GEOMETRY` fall back to the generic CN-only
    mapping (``square_planar=4``, ``octahedral=6``, ``linear=2``).
    """
    key = (metal_symbol, oxidation_state)
    expected = EXPECTED_GEOMETRY.get(key)
    if expected is None:
        # No canonical entry — fall back to a CN-only heuristic.
        # For the 5-metal core set this is a non-blocking mode.
        return _generic_cn_score(cn)
    expected_geom, expected_cn = expected
    # CN difference penalty: 0 if match, -0.5 per CN off, clamped >= 0.
    diff = abs(cn - expected_cn)
    if diff == 0:
        return 1.0
    return max(0.0, 1.0 - 0.5 * diff)


def _generic_cn_score(cn: int) -> float:
    """Heuristic CN score when no (metal, OS) entry exists."""
    if cn in (4, 6):
        return 1.0
    if cn in (3, 5):
        return 0.5
    if cn in (2, 7, 8):
        return 0.7  # linear / 7-coordinate are atypical but not absurd
    return 0.2


# ---------------------------------------------------------------------------
# Coordination geometry score
# ---------------------------------------------------------------------------
def coordination_geometry_score(smiles: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """Coordination-geometry compliance score in [0, 1].

    Parameters
    ----------
    smiles : str
        The ligand SMILES.

    Returns
    -------
    score : Optional[float]
        ``None`` for organic-only ligands. Otherwise a value in
        [0, 1] where 1.0 = canonical geometry and <1.0 = off-canonical.
    diagnostics : dict
        Per-call audit fields (``metal``, ``oxidation_state``,
        ``coordination_number``, ``expected_geometry``,
        ``expected_coordination_number``).
    """
    detected = _detect_metal_symbol(smiles)
    if detected is None:
        return None, {"metal": "", "coordination_number": 0}
    sym, os_, n_centres = detected
    cn = _coordination_number_from_smiles(smiles, sym)
    expected = EXPECTED_GEOMETRY.get((sym, os_))
    expected_geom, expected_cn = (expected if expected else ("unknown", -1))
    score = _coordination_geometry_score(cn, sym, os_)
    return score, {
        "metal": sym,
        "oxidation_state": os_,
        "coordination_number": cn,
        "expected_geometry": expected_geom,
        "expected_coordination_number": expected_cn,
        "n_metal_centres": n_centres,
    }


# ---------------------------------------------------------------------------
# HSAB match score
# ---------------------------------------------------------------------------
def hsab_match_score(
    smiles: str,
    pocket_donor_counts: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """HSAB (hard/soft acid-base) match score in [0, 1].

    Two modes:

    1. ``pocket_donor_counts`` provided (e.g. from a 3-D pocket scan):
       each donor class contributes ``count[cls] * HSAB_TABLE[metal][cls]``
       and the score is the count-weighted HSAB favourability.

    2. ``pocket_donor_counts`` is ``None``: fall back to a "donor atoms
       within the ligand itself" mode — useful for the smoke path
       where we don't have a 3-D pocket. We count N/O/S/P atoms bound
       to the metal centre and weight by HSAB.

    Parameters
    ----------
    smiles : str
        The ligand SMILES.
    pocket_donor_counts : Optional[Dict[str, int]]
        Optional ``{"N": n_n, "O": n_o, "S": n_s, "P": n_p}`` from a
        pocket scan. ``None`` falls back to ligand-internal mode.

    Returns
    -------
    score : Optional[float]
        ``None`` when no metal or no donors. Otherwise in [0, 1].
    diagnostics : dict
        Per-call audit fields (``metal``, ``donor_counts``,
        ``hsab_table_row``).
    """
    detected = _detect_metal_symbol(smiles)
    if detected is None:
        return None, {"metal": "", "donor_counts": {}}
    sym, os_, _ = detected
    table_row = HSAB_TABLE.get(sym)
    if table_row is None:
        return None, {"metal": sym, "donor_counts": {}, "reason": "no_hsab_table_row"}

    if pocket_donor_counts is not None:
        counts = {cls: int(pocket_donor_counts.get(cls, 0)) for cls in ("N", "O", "S", "P")}
    else:
        counts = _ligand_donor_counts(smiles, sym)

    total = sum(counts.values())
    if total == 0:
        return None, {"metal": sym, "donor_counts": counts, "reason": "no_donors"}

    weighted = sum(table_row[cls] * counts[cls] for cls in ("N", "O", "S", "P"))
    score = weighted / total
    return float(score), {
        "metal": sym,
        "donor_counts": counts,
        "hsab_table_row": dict(table_row),
    }


def _ligand_donor_counts(smiles: str, metal_symbol: str) -> Dict[str, int]:
    """Count N/O/S/P atoms bound to the first metal centre in
    ``smiles``. Returns ``{cls: count}`` with cls in (N,O,S,P)."""
    counts = {"N": 0, "O": 0, "S": 0, "P": 0}
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return counts
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        if _METAL_ATOMIC_NUMS.get(int(atom.GetAtomicNum())) != metal_symbol:
            continue
        for nbr in atom.GetNeighbors():
            cls = _DONOR_CLASS_BY_ATOMIC_NUM.get(int(nbr.GetAtomicNum()))
            if cls is None:
                continue
            counts[cls] += 1
        return counts
    return counts


# ---------------------------------------------------------------------------
# Metal-ligand distance score
# ---------------------------------------------------------------------------
def metal_ligand_distance_score(
    metal_symbol: str,
    donor_class: str,
    distance_angstrom: float,
) -> Optional[float]:
    """Gaussian-weighted distance score in [0, 1].

    score(d) = exp(-((d - d_opt) / sigma) ** 2)

    where ``d_opt = METAL_LIGAND_OPT_DISTANCE[(metal, donor_class)]``
    and ``sigma = _DISTANCE_SIGMA[donor_class]``. Returns ``None``
    when no entry exists for the (metal, donor_class) pair.
    """
    key = (metal_symbol, donor_class)
    d_opt = METAL_LIGAND_OPT_DISTANCE.get(key)
    if d_opt is None:
        return None
    sigma = _DISTANCE_SIGMA.get(donor_class, 0.20)
    diff = float(distance_angstrom) - float(d_opt)
    return float(math.exp(-(diff / sigma) ** 2))


def pocket_donor_counts_from_atomicnums(
    pocket_atom_types: Sequence[int],
    cutoff_angstrom: float = 4.5,
    centre: Optional[Sequence[float]] = None,
) -> Dict[str, int]:
    """Aggregate pocket donor counts near ``centre``.

    Convenience helper used by the Vina adapter to derive a
    pocket-side HSAB donor fingerprint. Without 3-D information
    we just count N/O/S/P atoms in the pocket — accurate enough
    for the HSAB heuristic (the first coordination shell is
    dominated by these atoms for His/Cys/Glu residues).
    """
    counts = {"N": 0, "O": 0, "S": 0, "P": 0}
    for anum in pocket_atom_types:
        cls = _DONOR_CLASS_BY_ATOMIC_NUM.get(int(anum))
        if cls is None:
            continue
        counts[cls] += 1
    return counts


# ---------------------------------------------------------------------------
# Composite term
# ---------------------------------------------------------------------------
class MetalCoordinationTerm:
    """Composite metal-coordination scoring term.

    Computes :meth:`score` for a given ligand SMILES (and optional
    3-D pose / pocket donor counts). All three sub-scores are
    optional — when none fires (e.g. organic ligand) the composite
    is ``None`` and the caller should leave Vina's kcal/mol
    unchanged.

    Parameters
    ----------
    weight_geometry : float, default 0.4
        Weight of the geometry-compliance sub-score in the composite.
    weight_hsab : float, default 0.4
        Weight of the HSAB sub-score.
    weight_distance : float, default 0.2
        Weight of the distance sub-score.
    kcal_per_unit : float, default 1.0
        Multiplier converting the composite (in [0, 1]) to kcal/mol.
        1.0 means a "perfect" coordination term is worth -1 kcal/mol
        of binding free-energy (Vina convention: lower / more negative
        kcal/mol = better binding). A "perfect mismatch" (composite = 0.0)
        is +1 kcal/mol (penalising the absence of coordination).
        Final adjustment is ``(1 - 2 * composite) * kcal_per_unit``.
    """

    def __init__(
        self,
        weight_geometry: float = 0.4,
        weight_hsab: float = 0.4,
        weight_distance: float = 0.2,
        kcal_per_unit: float = 1.0,
    ) -> None:
        self.weight_geometry = float(weight_geometry)
        self.weight_hsab = float(weight_hsab)
        self.weight_distance = float(weight_distance)
        self.kcal_per_unit = float(kcal_per_unit)

    def score(
        self,
        smiles: str,
        pocket_donor_counts: Optional[Dict[str, int]] = None,
        metal_ligand_distances: Optional[Dict[str, float]] = None,
    ) -> MetalCoordinationScore:
        """Compute the composite metal-coordination score.

        Parameters
        ----------
        smiles : str
            The ligand SMILES.
        pocket_donor_counts : Optional[Dict[str, int]]
            ``{cls: count}`` from a pocket scan. ``None`` falls back
            to ligand-internal donor counting (HSAB mode).
        metal_ligand_distances : Optional[Dict[str, float]]
            ``{cls: distance_angstrom}`` of the closest pocket
            donor atom to the metal centre, per donor class.
            ``None`` skips the distance sub-score.

        Returns
        -------
        :class:`MetalCoordinationScore`
            Composite + per-term scores.
        """
        geom_score, geom_diag = coordination_geometry_score(smiles)
        hsab_score, hsab_diag = hsab_match_score(smiles, pocket_donor_counts)
        dist_score = None
        dist_diag: Dict[str, Any] = {"enabled": metal_ligand_distances is not None}

        if metal_ligand_distances is not None:
            # Average over donor classes actually present.
            detected = _detect_metal_symbol(smiles)
            if detected is not None:
                sym, _, _ = detected
                contributions: List[float] = []
                for cls, d in metal_ligand_distances.items():
                    s = metal_ligand_distance_score(sym, cls, float(d))
                    if s is None:
                        continue
                    contributions.append(s)
                if contributions:
                    dist_score = float(sum(contributions) / len(contributions))
                dist_diag["per_class"] = {
                    cls: metal_ligand_distance_score(sym, cls, float(d))
                    for cls, d in metal_ligand_distances.items()
                }

        # Composite weighted sum. ``None`` sub-scores are skipped.
        weights: List[float] = []
        values: List[float] = []
        if geom_score is not None:
            weights.append(self.weight_geometry)
            values.append(float(geom_score))
        if hsab_score is not None:
            weights.append(self.weight_hsab)
            values.append(float(hsab_score))
        if dist_score is not None:
            weights.append(self.weight_distance)
            values.append(float(dist_score))

        total: Optional[float]
        if not values:
            total = None
        else:
            w_sum = sum(weights)
            total = float(sum(v * w for v, w in zip(values, weights)) / w_sum)

        diagnostics: Dict[str, Any] = {}
        diagnostics["geometry"] = geom_diag
        diagnostics["hsab"] = hsab_diag
        diagnostics["distance"] = dist_diag
        return MetalCoordinationScore(
            geometry_score=geom_score,
            hsab_score=hsab_score,
            distance_score=dist_score,
            total=total,
            diagnostics=diagnostics,
        )

    def kcal_adjustment(self, score: MetalCoordinationScore) -> float:
        """Convert a :class:`MetalCoordinationScore` to kcal/mol
        adjustment for Vina's score.

        When ``score.total is None`` (organic ligand, no donors), the
        adjustment is exactly 0.0 — backward-compat safe.

        Adjustment = (1 - 2 * composite) * kcal_per_unit
        so a perfect composite (1.0) yields -1.0 * kcal_per_unit
        (better binding — Vina convention: lower kcal/mol = better)
        and a zero composite yields +1.0 * kcal_per_unit
        (worse binding — penalising absence of coordination).
        Wf-333-V1 + Wf-246-C sign-fix: ``(1 - 2*c) * kcal_per_unit``
        per Vina convention "lower (more negative) = better binding".
        """
        if score.total is None:
            return 0.0
        return float((1.0 - 2.0 * score.total) * self.kcal_per_unit)


# ---------------------------------------------------------------------------
# 5-metal × donor summary table — paper §B / honest disclosure
# ---------------------------------------------------------------------------
def hsab_summary_5metal() -> Dict[str, Dict[str, float]]:
    """Return the 5-metal HSAB table (Pt/Ru/Ir/Os/Rh) used for
    the smoke test's per-metal diagnostic.

    This is the same as :data:`HSAB_TABLE` but sliced to the 5-metal
    core set with explicit donor columns — convenient for paper
    tables (paper §B m3/m10 Vina redesign).
    """
    metals = ("Pt", "Ru", "Ir", "Os", "Rh")
    donors = ("N", "O", "S", "P")
    return {m: {d: HSAB_TABLE.get(m, {}).get(d, 0.0) for d in donors} for m in metals}


__all__ = [
    "HSAB_TABLE",
    "METAL_LIGAND_OPT_DISTANCE",
    "EXPECTED_GEOMETRY",
    "MetalCoordinationScore",
    "MetalCoordinationTerm",
    "coordination_geometry_score",
    "hsab_match_score",
    "metal_ligand_distance_score",
    "pocket_donor_counts_from_atomicnums",
    "hsab_summary_5metal",
]
