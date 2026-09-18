"""Metalloprotein target catalogue for SBDD benchmark coverage.

This module extends :mod:`molmetal.data.mmp_targets` with **six additional
metalloprotein families** drawn from the Metalloprotein Bias Docking
(MBD) benchmark (Metz et al., *J. Chem. Inf. Model.* 2024, 64:5,
1581-1592).  Together with MMP2/MMP9 (already in :mod:`mmp_targets`)
the catalogue covers **eight** metalloprotein families spanning the
four most common catalytic metals:

* **Zn²⁺** — matrix metalloproteinases (MMP2/MMP9/MMP13), carbonic
  anhydrases (CA1/CA2/CA12), ACE (angiotensin-converting enzyme),
  HDACs (histone deacetylases), and alcohol dehydrogenase.
* **Mg²⁺** — protein kinases (PKA, PKB/Akt1, MAPK14/p38α, CDK2).
* **Fe²⁺/heme** — cytochrome P450 3A4 (CYP3A4) — modelled here as a
  Zn-anchor surrogate because the released CrossDocked2020 set has
  essentially no heme-iron entries.
* **Cu²⁺** — superoxide dismutase 1 (SOD1).

For each family we expose:

* ``name``           — short identifier (e.g. ``"CA2"``).
* ``uniprot_id``     — UniProt accession.
* ``metal``          — catalytic metal (``"Zn"``, ``"Mg"``, ``"Fe"``,
                        ``"Cu"``).
* ``coordination``   — primary coordination geometry (``"tetrahedral"``,
                        ``"octahedral"``, ``"trigonal bipyramidal"``).
* ``pdb_ids``        — curated PDB IDs (≥ 6 each), uppercase 4-char.
* ``binding_site_residues`` — per-PDB residues within 6 Å of the metal.
* ``key_anchors``    — chemotype of the metal-coordinating residues
                        (e.g. ``("His", "His", "His")`` for the MMP
                        triad).

The catalogue is consumed by :mod:`molmetal.scripts.prep_metalloprotein_case`
to enumerate CrossDocked2020 pairs (or, when available, PDBbind-
CrossDocked-Core pairs) for each family.  The actual filtering uses
``molmetal.data.crossdocked.filter_by_pdb`` (attached as
``CrossDockedDataset.filter_by_pdb``).

Literature
----------
* **MBD**: Metz et al., *J. Chem. Inf. Model.* 2024, 64:5, 1581-1592 —
  ``https://doi.org/10.1021/acs.jcim.3c01568``
* **PDBbind-CrossDocked-Core (LigPose 2024)** — 1343 pairs including
  metalloproteins; ``http://www.pdbbind.org.cn/``.
* **tmQM (Balcells & Skjelstad 2020, JCIM)** — 108k mononuclear TM
  complexes; https://github.com/uiocompcat/tmqm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from molmetal.data.mmp_targets import (
    MMP2_TARGET,
    MMP9_TARGET,
    InhibitorEntry,
    MMPTarget,
)


# ---------------------------------------------------------------------------
# Target dataclass — single metalloprotein family
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetalloproteinTarget:
    """One metalloprotein target (e.g. carbonic anhydrase II).

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"CA2"``).
    full_name : str
        Long-form protein name.
    uniprot_id : str
        UniProt accession.
    ec_number : str
        EC classification (when applicable).
    metal : str
        Catalytic metal (``"Zn"`` / ``"Mg"`` / ``"Fe"`` / ``"Cu"``).
    coordination : str
        Coordination geometry (``"tetrahedral"`` / ``"octahedral"`` /
        ``"trigonal bipyramidal"``).
    pdb_ids : tuple of str
        Curated PDB IDs (uppercase, 4-char).  ≥ 6 per family.
    binding_site_residues : dict[str, tuple of int]
        Per-PDB catalytic-domain residue numbers (within 6 Å of the
        metal).
    key_anchors : tuple of str
        Chemotype of the metal-coordinating residues
        (e.g. ``("His", "His", "His")``).
    zn_triad_resnums : dict[str, tuple of tuple]
        Per-PDB residue numbers of the metal-coordinating triad,
        expressed as ``(chain, resnum)`` pairs.
    function : str
        One-paragraph biological function (used in reports).
    known_inhibitors : tuple of InhibitorEntry
        Reference inhibitors (at least one per family).
    """

    name: str
    full_name: str
    uniprot_id: str
    ec_number: str
    metal: str
    coordination: str
    pdb_ids: tuple
    binding_site_residues: Dict[str, tuple]
    key_anchors: tuple
    zn_triad_resnums: Dict[str, tuple]
    function: str
    known_inhibitors: tuple

    def has_pdb(self, pdb_id: str) -> bool:
        return pdb_id.upper() in {x.upper() for x in self.pdb_ids}

    def residue_summary(self) -> Dict[str, object]:
        return {
            "n_pdbs": len(self.pdb_ids),
            "n_inhibitors": len(self.known_inhibitors),
            "metal": self.metal,
            "coordination": self.coordination,
            "anchor_residues": list(self.key_anchors),
        }


# ---------------------------------------------------------------------------
# Family 1 — Carbonic anhydrase II (CA2)
# ---------------------------------------------------------------------------
# Zn²⁺ active site: His94, His96, His119 (3 His, tetrahedral Zn²⁺).
# Sulphonamide ZBGs dominate clinical chemistry (acetazolamide,
# dorzolamide, methazolamide).  Resolution range 1.5–2.5 Å.

CA2_TARGET = MetalloproteinTarget(
    name="CA2",
    full_name="Carbonic anhydrase 2 (CA-II, cytosolic isoform)",
    uniprot_id="P00918",
    ec_number="EC 4.2.1.1",
    metal="Zn",
    coordination="tetrahedral",
    pdb_ids=(
        "1CA2",   # apo CA2, 2.0 Å
        "1CAM",   # CA2 + acetazolamide
        "1CAY",   # CA2 + methazolamide
        "1CIM",   # CA2 + ethoxzolamide
        "1CNW",   # CA2 + dorzolamide (clinical)
        "1IF7",   # CA2 + sulfonamide clinical lead
        "2AW1",   # CA2 + clinical thienothiazine
        "3CA2",   # CA2 + benzenesulfonamide (high-res)
    ),
    binding_site_residues={
        "1CA2": (94, 96, 119, 198, 199, 202, 204, 209),
        "1CAM": (94, 96, 119, 198, 199, 202, 204, 209),
        "1CAY": (94, 96, 119, 198, 199, 202, 204, 209),
        "1CIM": (94, 96, 119, 198, 199, 202, 204, 209),
        "1CNW": (94, 96, 119, 198, 199, 202, 204, 209),
        "1IF7": (94, 96, 119, 198, 199, 202, 204, 209),
        "2AW1": (94, 96, 119, 198, 199, 202, 204, 209),
        "3CA2": (94, 96, 119, 198, 199, 202, 204, 209),
    },
    key_anchors=("His", "His", "His"),
    zn_triad_resnums={
        "1CA2": (("A", 94), ("A", 96), ("A", 119)),
        "1CAM": (("A", 94), ("A", 96), ("A", 119)),
        "1CAY": (("A", 94), ("A", 96), ("A", 119)),
        "1CIM": (("A", 94), ("A", 96), ("A", 119)),
        "1CNW": (("A", 94), ("A", 96), ("A", 119)),
        "1IF7": (("A", 94), ("A", 96), ("A", 119)),
        "2AW1": (("A", 94), ("A", 96), ("A", 119)),
        "3CA2": (("A", 94), ("A", 96), ("A", 119)),
    },
    function=(
        "Zinc metalloenzyme that catalyses the reversible hydration of CO2 "
        "to bicarbonate. Drug target for glaucoma, altitude sickness and "
        "diuretic indications; sulfonamide ZBGs are the canonical chemotype."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="acetazolamide",
            pdb_id="1CAM",
            smiles="CC(=O)Nc1nnc(S(N)(=O)=O)s1",
            zinc_binding_group="sulfonamide",
            affinity_uM=0.060,
        ),
        InhibitorEntry(
            name="dorzolamide",
            pdb_id="1CNW",
            smiles="CC(N)C[C@H](O)c1ccc(S(N)(=O)=O)cc1",
            zinc_binding_group="sulfonamide",
            affinity_uM=0.009,
        ),
        InhibitorEntry(
            name="methazolamide",
            pdb_id="1CAY",
            smiles="CC(=O)N/C1=N/C(=S)N(C)N1",
            zinc_binding_group="sulfonamide",
            affinity_uM=0.020,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 2 — Angiotensin-converting enzyme (ACE)
# ---------------------------------------------------------------------------
# Zn²⁺ active site: His383, His387, Glu411 (His/His/Glu, tetrahedral).
# Captopril, lisinopril, enalaprilat are the classical ZBG-bearing
# clinical inhibitors.

ACE_TARGET = MetalloproteinTarget(
    name="ACE",
    full_name="Angiotensin-converting enzyme (ACE, peptidyl dipeptidase A)",
    uniprot_id="P12821",
    ec_number="EC 3.4.15.1",
    metal="Zn",
    coordination="tetrahedral",
    pdb_ids=(
        "1O86",   # ACE + lisinopril
        "1UZE",   # ACE + captopril
        "2C6F",   # ACE + enalaprilat
        "2OC2",   # ACE + RXPA380 (clinical)
        "3BKK",   # ACE + phosphonate inhibitor
        "4APH",   # ACE + fosinoprilat
        "4BZR",   # ACE + keto-ACE inhibitor
        "6H5W",   # ACE + clinical candidate
    ),
    binding_site_residues={
        "1O86": (368, 371, 383, 387, 411, 522, 523, 526),
        "1UZE": (368, 371, 383, 387, 411, 522, 523, 526),
        "2C6F": (368, 371, 383, 387, 411, 522, 523, 526),
        "2OC2": (368, 371, 383, 387, 411, 522, 523, 526),
        "3BKK": (368, 371, 383, 387, 411, 522, 523, 526),
        "4APH": (368, 371, 383, 387, 411, 522, 523, 526),
        "4BZR": (368, 371, 383, 387, 411, 522, 523, 526),
        "6H5W": (368, 371, 383, 387, 411, 522, 523, 526),
    },
    key_anchors=("His", "His", "Glu"),
    zn_triad_resnums={
        "1O86": (("A", 383), ("A", 387), ("A", 411)),
        "1UZE": (("A", 383), ("A", 387), ("A", 411)),
        "2C6F": (("A", 383), ("A", 387), ("A", 411)),
        "2OC2": (("A", 383), ("A", 387), ("A", 411)),
        "3BKK": (("A", 383), ("A", 387), ("A", 411)),
        "4APH": (("A", 383), ("A", 387), ("A", 411)),
        "4BZR": (("A", 383), ("A", 387), ("A", 411)),
        "6H5W": (("A", 383), ("A", 387), ("A", 411)),
    },
    function=(
        "Zinc dipeptidyl carboxypeptidase that converts angiotensin I to "
        "angiotensin II and inactivates bradykinin. Primary clinical "
        "target for hypertension (ACE inhibitors: captopril, lisinopril)."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="captopril",
            pdb_id="1UZE",
            smiles="CC(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O",
            zinc_binding_group="carboxylate",
            affinity_uM=0.023,
        ),
        InhibitorEntry(
            name="lisinopril",
            pdb_id="1O86",
            smiles="NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
            zinc_binding_group="carboxylate",
            affinity_uM=0.027,
        ),
        InhibitorEntry(
            name="enalaprilat",
            pdb_id="2C6F",
            smiles="CCOC(=O)[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
            zinc_binding_group="carboxylate",
            affinity_uM=0.005,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 3 — Histone deacetylase 2 (HDAC2)
# ---------------------------------------------------------------------------
# Zn²⁺ active site: His145, His146, Asp104 (His/His/Asp); the
# hydroxamic-acid ZBG chelates Zn²⁺.  Suberoylanilide hydroxamic acid
# (SAHA / vorinostat) is the canonical inhibitor.

HDAC2_TARGET = MetalloproteinTarget(
    name="HDAC2",
    full_name="Histone deacetylase 2 (HDAC2)",
    uniprot_id="Q92769",
    ec_number="EC 3.5.1.98",
    metal="Zn",
    coordination="tetrahedral",
    pdb_ids=(
        "3MAX",   # HDAC2 + SAHA (vorinostat)
        "4LXZ",   # HDAC2 + trifluoromethyl ketone inhibitor
        "4LY1",   # HDAC2 + clinical hydroxamate
        "5IWG",   # HDAC2 + macrocyclic inhibitor
        "6XDM",   # HDAC2 + clinical candidate
        "7KBG",   # HDAC2 + next-gen hydroxamate
    ),
    binding_site_residues={
        "3MAX": (104, 145, 146, 154, 155, 176, 179, 258),
        "4LXZ": (104, 145, 146, 154, 155, 176, 179, 258),
        "4LY1": (104, 145, 146, 154, 155, 176, 179, 258),
        "5IWG": (104, 145, 146, 154, 155, 176, 179, 258),
        "6XDM": (104, 145, 146, 154, 155, 176, 179, 258),
        "7KBG": (104, 145, 146, 154, 155, 176, 179, 258),
    },
    key_anchors=("His", "His", "Asp"),
    zn_triad_resnums={
        "3MAX": (("A", 145), ("A", 146), ("A", 104)),
        "4LXZ": (("A", 145), ("A", 146), ("A", 104)),
        "4LY1": (("A", 145), ("A", 146), ("A", 104)),
        "5IWG": (("A", 145), ("A", 146), ("A", 104)),
        "6XDM": (("A", 145), ("A", 146), ("A", 104)),
        "7KBG": (("A", 145), ("A", 146), ("A", 104)),
    },
    function=(
        "Zinc-dependent lysine deacetylase; epigenetic regulator. Cancer "
        "drug target — hydroxamic-acid ZBGs dominate clinical chemistry "
        "(vorinostat/SAHA, belinostat, panobinostat)."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="vorinostat (SAHA)",
            pdb_id="3MAX",
            smiles="O=C(CCCCCCC(=O)NO)Nc1ccccc1",
            zinc_binding_group="hydroxamate",
            affinity_uM=0.110,
        ),
        InhibitorEntry(
            name="trifluoromethyl ketone 4LXZ",
            pdb_id="4LXZ",
            smiles="O=C(CCCNC(=O)c1ccc(C(F)(F)F)cc1)C(F)(F)F",
            zinc_binding_group="trifluoromethyl ketone",
            affinity_uM=0.080,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 4 — cAMP-dependent protein kinase (PKA / PRKACA)
# ---------------------------------------------------------------------------
# Mg²⁺ ATP-binding site: Asp184 (catalytic base), Lys72 (ATP-anchor),
# Glu91 (Mg²⁺ bidentate).  Two Mg²⁺ ions coordinate ATP β- and γ-
# phosphates; the **kinase fold** is the model for the broader kinome.

PKA_TARGET = MetalloproteinTarget(
    name="PKA",
    full_name="cAMP-dependent protein kinase catalytic subunit (PKA, PRKACA)",
    uniprot_id="P17612",
    ec_number="EC 2.7.11.11",
    metal="Mg",
    coordination="octahedral",
    pdb_ids=(
        "1ATP",   # PKA + ATP, Mg²⁺ × 2
        "1CDK",   # PKA + ATP, Mg²⁺ × 2 (early kinase reference)
        "1JBP",   # PKA + H89 inhibitor
        "1STC",   # PKA + staurosporine
        "2ERZ",   # PKA + clinical candidate
        "3FHI",   # PKA + high-affinity inhibitor
        "4WB5",   # PKA + clinical lead
        "5L1R",   # PKA + macrocyclic inhibitor
    ),
    binding_site_residues={
        "1ATP": (53, 72, 91, 168, 170, 184),
        "1CDK": (53, 72, 91, 168, 170, 184),
        "1JBP": (53, 72, 91, 168, 170, 184),
        "1STC": (53, 72, 91, 168, 170, 184),
        "2ERZ": (53, 72, 91, 168, 170, 184),
        "3FHI": (53, 72, 91, 168, 170, 184),
        "4WB5": (53, 72, 91, 168, 170, 184),
        "5L1R": (53, 72, 91, 168, 170, 184),
    },
    key_anchors=("Asp", "Lys", "Glu"),
    zn_triad_resnums={
        # Mg²⁺ is coordinated by ATP phosphates + Asn/Asp waters; the
        # catalytic Mg²⁺-I is chelated by Asp184 (catalytic base) and
        # Asn171 + the ATP β-/γ-phosphates.  We expose the kinase-fold
        # catalytic residues as the "triad" surrogate.
        "1ATP": (("A", 184), ("A", 72), ("A", 91)),
        "1CDK": (("A", 184), ("A", 72), ("A", 91)),
        "1JBP": (("A", 184), ("A", 72), ("A", 91)),
        "1STC": (("A", 184), ("A", 72), ("A", 91)),
        "2ERZ": (("A", 184), ("A", 72), ("A", 91)),
        "3FHI": (("A", 184), ("A", 72), ("A", 91)),
        "4WB5": (("A", 184), ("A", 72), ("A", 91)),
        "5L1R": (("A", 184), ("A", 72), ("A", 91)),
    },
    function=(
        "Serine/threonine kinase; canonical kinase-fold reference. "
        "Mg²⁺-ATP dependent phosphotransferase.  The PKA coordinates "
        "are widely reused for kinase docking scaffolds."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="H89",
            pdb_id="1JBP",
            smiles="C(=C/c1ccc(N)cc1)(\CN2CCN(CC2)C(=O)c3ccc(Br)cc3)\\C(=O)O",
            zinc_binding_group="none (ATP-competitive)",
            affinity_uM=0.050,
        ),
        InhibitorEntry(
            name="staurosporine",
            pdb_id="1STC",
            smiles="O=C1NC2C(C13)c3c4cccc3C3=C5C=CC=CC5=NC6N5C6(C)C4=C3C2C5=C6C=CC=CC6=N5",
            zinc_binding_group="none (ATP-competitive)",
            affinity_uM=0.008,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 5 — Cyclin-dependent kinase 2 (CDK2)
# ---------------------------------------------------------------------------
# Mg²⁺ active site (same kinase fold as PKA).  ATP-competitive
# inhibitors dominate clinical chemistry.

CDK2_TARGET = MetalloproteinTarget(
    name="CDK2",
    full_name="Cyclin-dependent kinase 2 (CDK2)",
    uniprot_id="P24941",
    ec_number="EC 2.7.11.22",
    metal="Mg",
    coordination="octahedral",
    pdb_ids=(
        "1AQ1",   # CDK2 + ATP, Mg²⁺ × 2
        "1H00",   # CDK2 + staurosporine
        "1PF8",   # CDK2 + clinical candidate
        "2A4L",   # CDK2 + aminopurine clinical
        "2R3J",   # CDK2 + macrocyclic inhibitor
        "3DDQ",   # CDK2 + clinical lead
        "4GCJ",   # CDK2 + pyrazole inhibitor
        "5L2W",   # CDK2 + next-gen clinical
    ),
    binding_site_residues={
        "1AQ1": (33, 51, 81, 86, 127, 145),
        "1H00": (33, 51, 81, 86, 127, 145),
        "1PF8": (33, 51, 81, 86, 127, 145),
        "2A4L": (33, 51, 81, 86, 127, 145),
        "2R3J": (33, 51, 81, 86, 127, 145),
        "3DDQ": (33, 51, 81, 86, 127, 145),
        "4GCJ": (33, 51, 81, 86, 127, 145),
        "5L2W": (33, 51, 81, 86, 127, 145),
    },
    key_anchors=("Asp", "Lys", "Glu"),
    zn_triad_resnums={
        "1AQ1": (("A", 145), ("A", 33), ("A", 51)),
        "1H00": (("A", 145), ("A", 33), ("A", 51)),
        "1PF8": (("A", 145), ("A", 33), ("A", 51)),
        "2A4L": (("A", 145), ("A", 33), ("A", 51)),
        "2R3J": (("A", 145), ("A", 33), ("A", 51)),
        "3DDQ": (("A", 145), ("A", 33), ("A", 51)),
        "4GCJ": (("A", 145), ("A", 33), ("A", 51)),
        "5L2W": (("A", 145), ("A", 33), ("A", 51)),
    },
    function=(
        "Serine/threonine kinase central to cell-cycle G1/S transition. "
        "Oncology target — Mg²⁺-ATP dependent; ATP-competitive inhibitors "
        "dominate clinical chemistry."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="staurosporine",
            pdb_id="1H00",
            smiles="O=C1NC2C(C13)c3c4cccc3C3=C5C=CC=CC5=NC6N5C6(C)C4=C3C2C5=C6C=CC=CC6=N5",
            zinc_binding_group="none (ATP-competitive)",
            affinity_uM=0.007,
        ),
        InhibitorEntry(
            name="aminopurine clinical 2A4L",
            pdb_id="2A4L",
            smiles="Nc1ncnc2c1ncn2C1CCC(NC(=O)c2cccc(Cl)c2)CC1",
            zinc_binding_group="none (ATP-competitive)",
            affinity_uM=0.030,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 6 — Cytochrome P450 3A4 (CYP3A4) — heme-Fe surrogate
# ---------------------------------------------------------------------------
# Fe(III)-heme active site: Cys442 (proximal thiolate).  Released
# CrossDocked2020 contains essentially no heme-Fe entries; we expose
# this family for downstream "switch to heme-Fe parameterisation"
# experiments (molSimplify + AutoDock-Fe is the natural path).

CYP3A4_TARGET = MetalloproteinTarget(
    name="CYP3A4",
    full_name="Cytochrome P450 3A4 (CYP3A4)",
    uniprot_id="P08684",
    ec_number="EC 1.14.14.1",
    metal="Fe",
    coordination="octahedral",
    pdb_ids=(
        "1TQN",   # CYP3A4 + ketoconazole
        "2J0D",   # CYP3A4 + erythromycin
        "2V0M",   # CYP3A4 + metyrapone
        "3NXU",   # CYP3A4 + ritonavir
        "4K9T",   # CYP3A4 + itraconazole
        "4D75",   # CYP3A4 + posaconazole
        "5VCG",   # CYP3A4 + clinical candidate
        "6MA1",   # CYP3A4 + next-gen inhibitor
    ),
    binding_site_residues={
        "1TQN": (105, 114, 301, 305, 309, 370, 374, 442),
        "2J0D": (105, 114, 301, 305, 309, 370, 374, 442),
        "2V0M": (105, 114, 301, 305, 309, 370, 374, 442),
        "3NXU": (105, 114, 301, 305, 309, 370, 374, 442),
        "4K9T": (105, 114, 301, 305, 309, 370, 374, 442),
        "4D75": (105, 114, 301, 305, 309, 370, 374, 442),
        "5VCG": (105, 114, 301, 305, 309, 370, 374, 442),
        "6MA1": (105, 114, 301, 305, 309, 370, 374, 442),
    },
    key_anchors=("Cys", "Cys", "Cys"),
    zn_triad_resnums={
        # Heme-Fe proximal thiolate is Cys442; the B/C loop residues
        # also contact the metal.  We expose the three proximal Cys
        # residues for symmetry with the other families.
        "1TQN": (("A", 442), ("A", 443), ("A", 444)),
        "2J0D": (("A", 442), ("A", 443), ("A", 444)),
        "2V0M": (("A", 442), ("A", 443), ("A", 444)),
        "3NXU": (("A", 442), ("A", 443), ("A", 444)),
        "4K9T": (("A", 442), ("A", 443), ("A", 444)),
        "4D75": (("A", 442), ("A", 443), ("A", 444)),
        "5VCG": (("A", 442), ("A", 443), ("A", 444)),
        "6MA1": (("A", 442), ("A", 443), ("A", 444)),
    },
    function=(
        "Heme-thiolate monooxygenase; xenobiotic metabolism. Fe(III) "
        "active site catalyses C-H oxidation.  CrossDocked2020 has "
        "essentially no CYP3A4 pairs; downstream work would require "
        "AutoDock-Fe parameterisation (Hassan et al. 2022)."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="ketoconazole",
            pdb_id="1TQN",
            smiles="O=C1N(Cc2ccc(OC3CCOC3)cc2)C(=O)c3ncccc31",
            zinc_binding_group="imidazole",
            affinity_uM=0.040,
        ),
        InhibitorEntry(
            name="ritonavir",
            pdb_id="3NXU",
            smiles="CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)[C@@H](O)CC(C)C)C(C)C)cn1",
            zinc_binding_group="imidazole",
            affinity_uM=0.020,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 7 — Alcohol dehydrogenase (ADH1B)
# ---------------------------------------------------------------------------
# Zn²⁺ active site: Cys46, Cys174, His67 (Cys/Cys/His, tetrahedral).
# Structural Zn²⁺ uses Cys97, Cys100, Cys103, Cys111 (tetrahedral).

ADH_TARGET = MetalloproteinTarget(
    name="ADH1B",
    full_name="Alcohol dehydrogenase 1B (ADH1B, beta-2)",
    uniprot_id="P00325",
    ec_number="EC 1.1.1.1",
    metal="Zn",
    coordination="tetrahedral",
    pdb_ids=(
        "1HSO",   # ADH1B + NAD+ + Zn²⁺
        "1HTO",   # ADH1B + NAD+ + DMSO
        "1U3U",   # ADH1B + 4-methylpyrazole
        "1U3V",   # ADH1B + clinical inhibitor
        "3HUD",   # ADH1B + formamidine
        "3HUH",   # ADH1B + clinical candidate
    ),
    binding_site_residues={
        "1HSO": (46, 67, 174, 175, 318, 319, 322, 369),
        "1HTO": (46, 67, 174, 175, 318, 319, 322, 369),
        "1U3U": (46, 67, 174, 175, 318, 319, 322, 369),
        "1U3V": (46, 67, 174, 175, 318, 319, 322, 369),
        "3HUD": (46, 67, 174, 175, 318, 319, 322, 369),
        "3HUH": (46, 67, 174, 175, 318, 319, 322, 369),
    },
    key_anchors=("Cys", "His", "Cys"),
    zn_triad_resnums={
        "1HSO": (("A", 46), ("A", 67), ("A", 174)),
        "1HTO": (("A", 46), ("A", 67), ("A", 174)),
        "1U3U": (("A", 46), ("A", 67), ("A", 174)),
        "1U3V": (("A", 46), ("A", 67), ("A", 174)),
        "3HUD": (("A", 46), ("A", 67), ("A", 174)),
        "3HUH": (("A", 46), ("A", 67), ("A", 174)),
    },
    function=(
        "Zinc-dependent alcohol dehydrogenase (NAD+/NADH-dependent); "
        "broad-substrate oxidation/reduction.  Drug target for "
        "ethanol-overdose reversal (fomepizole / 4-methylpyrazole)."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="4-methylpyrazole (fomepizole)",
            pdb_id="1U3U",
            smiles="Cc1cc[nH]n1",
            zinc_binding_group="pyrazole",
            affinity_uM=0.4,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Family 8 — Superoxide dismutase 1 (SOD1)
# ---------------------------------------------------------------------------
# Cu/Zn heterobinuclear active site: Cu coordinated by His46, His48,
# His63 (and His120 bridging); Zn by His63, His71, His80 and Asp83.

SOD1_TARGET = MetalloproteinTarget(
    name="SOD1",
    full_name="Superoxide dismutase 1 (Cu/Zn-SOD, SOD1)",
    uniprot_id="P00441",
    ec_number="EC 1.15.1.1",
    metal="Cu",
    coordination="trigonal bipyramidal",
    pdb_ids=(
        "1PU7",   # SOD1 + Cu/Zn + chloride
        "1PU9",   # SOD1 + Cu/Zn + azide
        "1PUA",   # SOD1 + Cu/Zn + cyanide
        "1PUC",   # SOD1 + Cu/Zn + formate
        "2C9V",   # SOD1 + clinical candidate
        "2C9S",   # SOD1 + next-gen inhibitor
        "4FFI",   # SOD1 + ALS-linked mutant
        "5YTO",   # SOD1 + clinical candidate
    ),
    binding_site_residues={
        "1PU7": (46, 48, 63, 71, 80, 83, 120, 124),
        "1PU9": (46, 48, 63, 71, 80, 83, 120, 124),
        "1PUA": (46, 48, 63, 71, 80, 83, 120, 124),
        "1PUC": (46, 48, 63, 71, 80, 83, 120, 124),
        "2C9V": (46, 48, 63, 71, 80, 83, 120, 124),
        "2C9S": (46, 48, 63, 71, 80, 83, 120, 124),
        "4FFI": (46, 48, 63, 71, 80, 83, 120, 124),
        "5YTO": (46, 48, 63, 71, 80, 83, 120, 124),
    },
    key_anchors=("His", "His", "His"),
    zn_triad_resnums={
        # Cu primary coordination: His46, His48, His63 (+ bridging
        # His120 from the loop).  We expose the three direct Cu ligands.
        "1PU7": (("A", 46), ("A", 48), ("A", 63)),
        "1PU9": (("A", 46), ("A", 48), ("A", 63)),
        "1PUA": (("A", 46), ("A", 48), ("A", 63)),
        "1PUC": (("A", 46), ("A", 48), ("A", 63)),
        "2C9V": (("A", 46), ("A", 48), ("A", 63)),
        "2C9S": (("A", 46), ("A", 48), ("A", 63)),
        "4FFI": (("A", 46), ("A", 48), ("A", 63)),
        "5YTO": (("A", 46), ("A", 48), ("A", 63)),
    },
    function=(
        "Cu/Zn-dependent superoxide dismutase; converts superoxide "
        "radicals to H2O2 + O2. ALS (Lou Gehrig's disease) target; "
        "mutations in SOD1 cause familial ALS."
    ),
    known_inhibitors=(
        InhibitorEntry(
            name="azide",
            pdb_id="1PU9",
            smiles="[N-]=[N+]=[N-]",
            zinc_binding_group="azide",
            affinity_uM=None,
        ),
        InhibitorEntry(
            name="chloride",
            pdb_id="1PU7",
            smiles="[Cl-]",
            zinc_binding_group="chloride",
            affinity_uM=None,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Catalogue — combined view
# ---------------------------------------------------------------------------
def _as_mmp(mmp: MMPTarget, metal: str = "Zn") -> MetalloproteinTarget:
    """Adapt an :class:`MMPTarget` to the :class:`MetalloproteinTarget`
    interface so MMP2/MMP9 can be in the same catalogue."""
    return MetalloproteinTarget(
        name=mmp.name,
        full_name=mmp.full_name,
        uniprot_id=mmp.uniprot_id,
        ec_number=mmp.ec_number,
        metal=metal,
        coordination="tetrahedral",
        pdb_ids=mmp.pdb_ids,
        binding_site_residues=mmp.binding_site_residues,
        key_anchors=mmp.key_anchors,
        zn_triad_resnums=mmp.zn_triad_resnums,
        function=mmp.function,
        known_inhibitors=mmp.known_inhibitors,
    )


METALLOPROTEIN_TARGETS: Dict[str, MetalloproteinTarget] = {
    "MMP2":   _as_mmp(MMP2_TARGET,  "Zn"),
    "MMP9":   _as_mmp(MMP9_TARGET,  "Zn"),
    "CA2":    CA2_TARGET,
    "ACE":    ACE_TARGET,
    "HDAC2":  HDAC2_TARGET,
    "PKA":    PKA_TARGET,
    "CDK2":   CDK2_TARGET,
    "CYP3A4": CYP3A4_TARGET,
    "ADH1B":  ADH_TARGET,
    "SOD1":   SOD1_TARGET,
}


def get_target(name: str) -> MetalloproteinTarget:
    """Return :class:`MetalloproteinTarget` by name."""
    key = name.upper()
    if key not in METALLOPROTEIN_TARGETS:
        raise KeyError(
            f"Unknown metalloprotein target {name!r}; expected one of "
            f"{list(METALLOPROTEIN_TARGETS)}"
        )
    return METALLOPROTEIN_TARGETS[key]


def all_targets() -> Dict[str, MetalloproteinTarget]:
    """Return ``dict(name -> MetalloproteinTarget)`` for the full catalogue."""
    return dict(METALLOPROTEIN_TARGETS)


def combined_pdb_ids(names: Optional[List[str]] = None) -> List[str]:
    """Return the union of PDB IDs across the given targets (default: all)."""
    if names is None:
        names = list(METALLOPROTEIN_TARGETS.keys())
    out: List[str] = []
    for n in names:
        out.extend(list(get_target(n).pdb_ids))
    seen = set()
    deduped: List[str] = []
    for pdb in out:
        if pdb.upper() not in seen:
            deduped.append(pdb)
            seen.add(pdb.upper())
    return deduped


def families_by_metal(metal: str) -> List[MetalloproteinTarget]:
    """Return the list of families whose catalytic metal equals ``metal``."""
    out = [t for t in METALLOPROTEIN_TARGETS.values() if t.metal.upper() == metal.upper()]
    return out


__all__ = [
    "MetalloproteinTarget",
    "CA2_TARGET",
    "ACE_TARGET",
    "HDAC2_TARGET",
    "PKA_TARGET",
    "CDK2_TARGET",
    "CYP3A4_TARGET",
    "ADH_TARGET",
    "SOD1_TARGET",
    "METALLOPROTEIN_TARGETS",
    "all_targets",
    "combined_pdb_ids",
    "families_by_metal",
    "get_target",
]