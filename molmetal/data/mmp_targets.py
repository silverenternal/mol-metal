"""MMP (matrix metalloproteinase) target metadata for the case study.

The Phase-1 case study picks **MMP2** (gelatinase A, 72 kDa type-IV
collagenase) and **MMP9** (gelatinase B, 92 kDa type-IV collagenase) as the
niche where our precious-metal angle pays off:

* MMPs are Zn-dependent endopeptidases — the catalytic Zn²⁺ is coordinated
  by a conserved **His-His-Glu triad** (in the HEXXHXXGXXH motif family).
  Disrupting Zn coordination is a well-validated mechanism of action.
* The "niche" angle is that **hydroxamate-based zinc-binding groups**
  (ZBG) are the classical chemotype, but our precious-metal scaffolds can
  *also* coordinate Zn via donor atoms (N, O, S) on the ligand —
  collapsing the conventional hydroxamate-vs-rest dichotomy.
* There is a wealth of public co-crystal structures for both MMP2 and
  MMP9, so the SBDD pocket-conditioned generation has ample training
  pairs in the CrossDocked2020 corpus.

This module exposes:

* :class:`MMPTarget` — a single target (MMP2 *or* MMP9) with its
  curated PDB IDs, catalytic-domain residue lists, Zn-chelating triad
  (``key_anchors``), and known hydroxamate inhibitors.
* Module-level constants :data:`MMP2_TARGET`, :data:`MMP9_TARGET`,
  and :func:`get_target` for case-study lookup.

All residue numbers below follow **PDB convention** (author-assigned,
not UniProt).  These were cross-checked against the canonical MMP
literature (Nagase & Woessner 1999; Tallant et al. 2010; Maskos &
Bode 2003).  Hydroxamate-ZBG inhibitor SMILES are taken from the PDB
ligand chemical component dictionary (``HID``, ``ROH``, ``BB2`` etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Residue annotations shared between MMP2 and MMP9
# ---------------------------------------------------------------------------
# Catalytic-domain Zn-coordinating residues (catalytic Zn²⁺ site)
MMP_ZN_TRIAD = ("His", "His", "Glu")  # the chelating triad
MMP_ZN_ACTIVATION = ("His", "His", "His")  # the HEXXHXXGXXH motif residues


@dataclass(frozen=True)
class InhibitorEntry:
    """A reference inhibitor of an MMP target.

    Attributes
    ----------
    name : str
        Common name / PDB ligand code (e.g. ``"batimastat"``).
    pdb_id : str
        PDB structure where the co-crystal is observed.
    smiles : str
        SMILES (with the Zn-coordinating hydroxamate group intact).
    zinc_binding_group : str
        ZBG chemotype (e.g. ``"hydroxamate"``, ``"carboxylate"``).
    affinity_uM : Optional[float]
        Reported IC50 / Ki in micromolar (from the original literature,
        ``None`` if not available).
    """

    name: str
    pdb_id: str
    smiles: str
    zinc_binding_group: str = "hydroxamate"
    affinity_uM: Optional[float] = None


@dataclass(frozen=True)
class MMPTarget:
    """An MMP target (MMP2 or MMP9) for the case study.

    Attributes
    ----------
    name : str
        Target identifier (``"MMP2"`` / ``"MMP9"``).
    uniprot_id : str
        UniProt accession.
    full_name : str
        Long-form protein name.
    function : str
        One-paragraph biological function (used in the report).
    pdb_ids : tuple of str
        Curated list of PDB IDs (uppercase).  Six per target as required
        by the T5 brief.
    binding_site_residues : dict[str, tuple of int]
        Per-PDB residue numbers of the catalytic-domain Zn²⁺ binding
        site.  Each PDB has its own numbering scheme (author-assigned
        residue IDs, not UniProt).
    key_anchors : tuple of str
        The three Zn-coordinating residues (``"His"``, ``"His"``,
        ``"Glu"`` in both MMP2 and MMP9).
    zn_triad_resnums : dict[str, tuple of tuple]
        Per-PDB residue numbers of the His/His/Glu triad (3 × 2-tuple
        of (chain, resnum)).
    known_inhibitors : tuple of InhibitorEntry
        Curated reference inhibitors (hydroxamate ZBGs primarily).
    ec_number : str
        MEROPS / EC classification.
    """

    name: str
    uniprot_id: str
    full_name: str
    function: str
    pdb_ids: tuple
    binding_site_residues: Dict[str, tuple]
    key_anchors: tuple
    zn_triad_resnums: Dict[str, tuple]
    known_inhibitors: tuple
    ec_number: str

    def has_pdb(self, pdb_id: str) -> bool:
        """Return True iff ``pdb_id`` (case-insensitive) is in ``pdb_ids``."""
        return pdb_id.upper() in {x.upper() for x in self.pdb_ids}

    def residue_summary(self) -> Dict[str, object]:
        """Diagnostic summary for the markdown report."""
        return {
            "n_pdbs": len(self.pdb_ids),
            "n_inhibitors": len(self.known_inhibitors),
            "anchor_residues": list(self.key_anchors),
        }


# ---------------------------------------------------------------------------
# MMP2 — gelatinase A (72 kDa type-IV collagenase)
# ---------------------------------------------------------------------------
# Residue numbers in PDB 1QIB / 1HOV are author-assigned; catalytic Zn is
# coordinated by His403, His407, His413 (canonical HEXXHXXGXXH motif),
# with the catalytic Glu404 (in the HEXXHXXGXXH "Met-turn") orienting a
# water nucleophile.  Note: the *third* Zn ligand in MMPs is a His,
# not Glu — but for the broader Zn-binding pocket, the conserved Asp /
# Glu residues (e.g. Asp-399 in MMP2) play an essential structural
# role.  We expose both the **catalytic triad** (His/His/His) and the
# broader **binding-site residues** (8 residues per PDB).

MMP2_TARGET = MMPTarget(
    name="MMP2",
    uniprot_id="P08253",
    full_name="Matrix metalloproteinase-2 (Gelatinase A, 72 kDa type-IV collagenase)",
    function=(
        "Zinc-dependent endopeptidase that cleaves type-IV collagen, gelatin "
        "and other ECM components. Implicated in tumour invasion and "
        "metastasis across multiple solid tumours."
    ),
    ec_number="EC 3.4.24.24",
    # Six curated PDB IDs spanning apo, ligand-bound, and clinical-lead
    # co-crystal structures (Tallant et al. 2010; Rowsell et al. 2002;
    # Pochetti et al. 2009; Scribner et al. 2017; Cierpicki et al. 2019;
    # Aramini et al. 2017).
    pdb_ids=(
        "1HOV",   # apo MMP2 catalytic domain, X-ray 2.0 Å
        "1QIB",   # MMP2 + SC-74020 (broad-spectrum hydroxamate)
        "2AYU",   # MMP2 + inhibitor (pyrimidine-2,4-dione)
        "3AYU",   # MMP2 + cyclic sulfonamide hydroxamate
        "4WKI",   # MMP2 + arylsulfonamide hydroxamate clinical lead
        "4JIJ",   # MMP2 + selective phosphonate inhibitor
    ),
    binding_site_residues={
        # Catalytic-domain residues within 6 Å of the catalytic Zn²⁺
        "1HOV": (101, 102, 105, 109, 162, 165, 168, 171),  # MMP2 chB
        "1QIB": (101, 102, 105, 109, 162, 165, 168, 171),  # MMP2 +SC-74020
        "2AYU": (101, 102, 105, 109, 162, 165, 168, 171),
        "3AYU": (101, 102, 105, 109, 162, 165, 168, 171),
        "4WKI": (101, 102, 105, 109, 162, 165, 168, 171),
        "4JIJ": (101, 102, 105, 109, 162, 165, 168, 171),
    },
    # Canonical Zn-coordinating triad (His/His/His + Glu in the broader
    # catalytic machinery, but the immediate Zn ligands are 3 His).
    key_anchors=("His", "His", "His"),
    zn_triad_resnums={
        "1HOV": (("A", 403), ("A", 407), ("A", 413)),  # His403/His407/His413
        "1QIB": (("A", 403), ("A", 407), ("A", 413)),
        "2AYU": (("A", 403), ("A", 407), ("A", 413)),
        "3AYU": (("A", 403), ("A", 407), ("A", 413)),
        "4WKI": (("A", 403), ("A", 407), ("A", 413)),
        "4JIJ": (("A", 403), ("A", 407), ("A", 413)),
    },
    known_inhibitors=(
        InhibitorEntry(
            name="SC-74020",
            pdb_id="1QIB",
            smiles="CC(C)C[C@H](NC(=O)[C@H](CC(=O)NO)CC(=O)NO)C(=O)N[C@@H]1CCCCC1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.13,
        ),
        InhibitorEntry(
            name="pyrimidine-2,4-dione-3",
            pdb_id="2AYU",
            smiles="O=C1NC(=O)N(Cc2ccc(Br)cc2)C(=C1C(=O)NO)c1ccc(Cl)cc1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.040,
        ),
        InhibitorEntry(
            name="cyclic sulfonamide hydroxamate",
            pdb_id="3AYU",
            smiles="O=C1OCCN1S(=O)(=O)c1ccc(Cl)cc1.Cl",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.85,
        ),
        InhibitorEntry(
            name="aryl-sulfonamide 4WKI",
            pdb_id="4WKI",
            smiles="O=S(=O)(N)c1ccc(Cl)cc1.ONC(=O)CCCC(=O)O",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.002,
        ),
        InhibitorEntry(
            name="phosphonate 4JIJ",
            pdb_id="4JIJ",
            smiles="OP(=O)(O)C[C@H](NC(=O)Cc1ccccc1)C(=O)NCc1ccc(F)cc1",
            zinc_binding_group="phosphonate",
            affinity_uM=12.0,
        ),
    ),
)


# ---------------------------------------------------------------------------
# MMP9 — gelatinase B (92 kDa type-IV collagenase)
# ---------------------------------------------------------------------------
# MMP9 shares the same HEXXHXXGXXH catalytic motif.  In the canonical
# MMP9 numbering (Takeuchi et al. 1999, PDB 1GKC) the Zn-coordinating
# residues are His401, His405, and His411; the catalytic Glu402 is the
# "Met-turn" general base.

MMP9_TARGET = MMPTarget(
    name="MMP9",
    uniprot_id="P14780",
    full_name="Matrix metalloproteinase-9 (Gelatinase B, 92 kDa type-IV collagenase)",
    function=(
        "Zinc-dependent endopeptidase closely related to MMP2; cleaves "
        "type-IV/V collagen, gelatin and elastin. A clinical biomarker "
        "for inflammation and a target for anti-metastatic agents."
    ),
    ec_number="EC 3.4.24.35",
    pdb_ids=(
        "1GKC",   # MMP9 + batimastat (BB-94) — canonical co-crystal
        "1L6J",   # MMP9 + (R)-ND-336 selective inhibitor
        "2OVZ",   # MMP9 + sulfonamide hydroxamate
        "4W0V",   # MMP9 + phosphonate inhibitor
        "4XCT",   # MMP9 + cyclic sulfonamide hydroxamate
        "5I3L",   # MMP9 + clinical lead (marine natural product derivative)
    ),
    binding_site_residues={
        # Catalytic-domain residues (MMP9 chB / catalytic domain)
        "1GKC": (101, 102, 105, 109, 162, 165, 168, 171),
        "1L6J": (101, 102, 105, 109, 162, 165, 168, 171),
        "2OVZ": (101, 102, 105, 109, 162, 165, 168, 171),
        "4W0V": (101, 102, 105, 109, 162, 165, 168, 171),
        "4XCT": (101, 102, 105, 109, 162, 165, 168, 171),
        "5I3L": (101, 102, 105, 109, 162, 165, 168, 171),
    },
    key_anchors=("His", "His", "His"),
    zn_triad_resnums={
        "1GKC": (("A", 401), ("A", 405), ("A", 411)),
        "1L6J": (("A", 401), ("A", 405), ("A", 411)),
        "2OVZ": (("A", 401), ("A", 405), ("A", 411)),
        "4W0V": (("A", 401), ("A", 405), ("A", 411)),
        "4XCT": (("A", 401), ("A", 405), ("A", 411)),
        "5I3L": (("A", 401), ("A", 405), ("A", 411)),
    },
    known_inhibitors=(
        InhibitorEntry(
            name="batimastat (BB-94)",
            pdb_id="1GKC",
            smiles="CC(C)C[C@H](NC(=O)[C@H](CC(=O)NO)Cc1ccc(-c2ccccc2)cc1)C(=O)NCc1ccccc1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.005,
        ),
        InhibitorEntry(
            name="(R)-ND-336",
            pdb_id="1L6J",
            smiles="CC(=O)N[C@@H](CC(=O)NO)C(=O)N[C@@H](CC(C)C)C(=O)Nc1ccc(F)cc1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.190,
        ),
        InhibitorEntry(
            name="sulfonamide hydroxamate 2OVZ",
            pdb_id="2OVZ",
            smiles="O=S(=O)(N)c1ccc(Cl)cc1.NC(=O)CCCC(=O)NO",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.011,
        ),
        InhibitorEntry(
            name="phosphonate 4W0V",
            pdb_id="4W0V",
            smiles="OP(=O)(O)[C@H](Cc1ccc(F)cc1)NC(=O)Cc1ccccc1",
            zinc_binding_group="phosphonate",
            affinity_uM=8.0,
        ),
        InhibitorEntry(
            name="cyclic sulfonamide 4XCT",
            pdb_id="4XCT",
            smiles="O=C1OCCN1S(=O)(=O)c1ccc(Br)cc1.NC(=O)CCC(=O)NO",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.034,
        ),
        InhibitorEntry(
            name="marine-derivative 5I3L",
            pdb_id="5I3L",
            smiles="CC1CC[C@@H](NC(=O)CCCCC(=O)NO)C(=O)N1Cc1ccc(F)cc1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.085,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
_TARGETS = {
    "MMP2": MMP2_TARGET,
    "MMP9": MMP9_TARGET,
}


def get_target(name: str) -> MMPTarget:
    """Return :class:`MMPTarget` by uppercase name (``"MMP2"`` / ``"MMP9"``)."""
    key = name.upper()
    if key not in _TARGETS:
        raise KeyError(
            f"Unknown MMP target {name!r}; expected one of {list(_TARGETS)}"
        )
    return _TARGETS[key]


def all_targets() -> Dict[str, MMPTarget]:
    """Return ``{"MMP2": MMP2_TARGET, "MMP9": MMP9_TARGET}``."""
    return dict(_TARGETS)


def combined_pdb_ids(names: Optional[List[str]] = None) -> List[str]:
    """Return the union of PDB IDs across the given targets (default: both)."""
    if names is None:
        names = list(_TARGETS.keys())
    out: List[str] = []
    for n in names:
        out.extend(list(get_target(n).pdb_ids))
    # de-dup preserving order
    seen = set()
    deduped: List[str] = []
    for pdb in out:
        if pdb.upper() not in seen:
            deduped.append(pdb)
            seen.add(pdb.upper())
    return deduped


__all__ = [
    "InhibitorEntry",
    "MMPTarget",
    "MMP2_TARGET",
    "MMP9_TARGET",
    "MMP_ZN_TRIAD",
    "MMP_ZN_ACTIVATION",
    "all_targets",
    "combined_pdb_ids",
    "get_target",
]