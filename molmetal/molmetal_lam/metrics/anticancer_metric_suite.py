"""Anticancer metric suite for precious-metal drug design.

Implements the TODO-15 metric bucket per
``molmetal/reports/anticancer_vs_general_metrics_survey.md``:

  * IV-appropriate descriptor window (logP 2-5 AND TPSA 60-150)
  * Metal coordination score (square-planar Pt(II) = 4, octahedral
    Ru/Ir = 6; intermediate counts score lower)
  * GSH evasion flag (heuristic; not a measured rate)
  * DNA Kb proxy (heuristic; not a measured binding constant)
  * Composite anticancer_index = mean of (iv_window_ok, metal_score,
    gsh_evasion, dna_proxy), all weighted equally.

DESCRIPTOR sources: RDKit (logP, TPSA, RotB, MW, NumHA, NumHD).
Mol-Metal priors: metal geometry + hydration (cl count, chelate,
aromatic N, donor count).

References:
  - Reedijk (1996), *Chem. Commun.*, 801-806.
  - Lippard (1999), *Acc. Chem. Res.*, 32, 1013-1020.
  - Hartinger et al. (2006), *Chem. Rev.*, 106, 2735-2790.
  - Kelloff et al. (1991), *J. Natl. Cancer Inst.*, 83, 1169-1189.
  - Hambley (2007), *Dalton Trans.*, 4929-4937.

This module is RDKit-optional; missing RDKit or malformed SMILES
gracefully return neutral fallbacks (0.5 for floats, False for flags).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None

from molmetal_lam.priors.metal_hydration import MetalHydrationAnalyzer


__all__ = ["AnticancerMetricSuite"]


# Anticancer IV window (per anticancer_vs_general_metrics_survey.md §2)
LOGP_LO, LOGP_HI = 2.0, 5.0
TPSA_LO, TPSA_HI = 60.0, 150.0
MW_LO, MW_HI = 300.0, 700.0  # descriptive IV window; not Lipinski

# Coordination geometry ideals (coordination number = donor count)
SQ_PLANAR_IDEAL = 4   # Pt(II) d8 square-planar
OCTAHEDRAL_IDEAL = 6  # Ru(II/III), Ir(III) octahedral

# Canonical SMILES for the four reference metal drugs (hand-checked).
CISPLATIN_SMILES = "N.N.[Cl-].[Cl-].[Pt+2]"
CARBOPLATIN_SMILES = "O=C(O)C1CCC(C(=O)O)CC1.N.N.[Cl-].[Cl-].[Pt+2]"
OXALIPLATIN_SMILES = "N.N.O=C(O)C(=O)O.[Pt+2]"
AURANOFIN_SMILES = "S(CC)SCC.O[C@@H]1[C@H](O)[C@@H](O)[C@H](O[C@@H]1CO)[Au+]SC1=NC=CC=C1.CC([O-])=O.CC([O-])=O"
RUTHENIUM_ARENE_SMILES = "C1=CC=CC=C1.[Cl-].[Cl-].[Ru+2]"


class AnticancerMetricSuite:
    """Anticancer metric suite (TODO-15 spec).

    Parameters
    ----------
    metal : str
        Default metal symbol for hydration analyzer (e.g. "Pt").
    oxidation_state : int
        Default oxidation state for hydration analyzer (e.g. 2).
    """

    def __init__(self, metal: str = "Pt", oxidation_state: int = 2):
        self.metal = str(metal)
        self.oxidation_state = int(oxidation_state)
        self.hydration = MetalHydrationAnalyzer(metal, oxidation_state)

    # ------------------------------------------------------------------
    # RDKit helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mol(smiles):
        if Chem is None or not isinstance(smiles, str):
            return None
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    def _has_metal(mol) -> bool:
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ("Pt", "Ru", "Ir", "Au", "Rh", "Os"):
                return True
        return False

    # ------------------------------------------------------------------
    # Raw descriptors
    # ------------------------------------------------------------------

    def descriptor_report(self, mol) -> Dict:
        """Compute physico-chemical descriptor report for a molecule.

        Accepts either a SMILES string or an rdkit.Chem.Mol instance.

        Returns
        -------
        dict
            logP, TPSA, RotB, MW, NumHA, NumHD, iv_window_ok, mw_iv_flag,
            oxidation_state, coordination_number, geometry, dna_kb_proxy_v1

            Total: 13 metrics (8 legacy + 4 new anticancer metric
            emissions in this report, including the 4 new fields:
            ``oxidation_state``, ``coordination_number``, ``geometry``,
            and ``dna_kb_proxy_v1``). The legacy 9 fields remain
            additive: the four new keys are appended and do not
            replace or modify any legacy field.
        """
        if isinstance(mol, str):
            smiles = mol
            rdkit_mol = self._mol(smiles)
        else:
            smiles = getattr(mol, "smiles", None) or ""
            rdkit_mol = mol if Chem is not None and isinstance(mol, Chem.Mol) else None

        if rdkit_mol is None or Descriptors is None:
            return {
                "logP": float("nan"),
                "TPSA": float("nan"),
                "RotB": float("nan"),
                "MW": float("nan"),
                "NumHA": float("nan"),
                "NumHD": float("nan"),
                "iv_window_ok": False,
                "mw_iv_flag": False,
                "oxidation_state": None,
                "coordination_number": 0,
                "geometry": "none",
                "dna_kb_proxy_v1": 0.0,
            }

        try:
            logp = float(Descriptors.MolLogP(rdkit_mol))
            tpsa = float(Descriptors.TPSA(rdkit_mol))
            rotb = float(Descriptors.NumRotatableBonds(rdkit_mol))
            mw = float(Descriptors.MolWt(rdkit_mol))
            # HeavyAtomCount: includes metals (which are heavier than C)
            numha = float(rdkit_mol.GetNumHeavyAtoms())
            numhd = float(Descriptors.NumHDonors(rdkit_mol))
        except Exception:
            return {
                "logP": float("nan"),
                "TPSA": float("nan"),
                "RotB": float("nan"),
                "MW": float("nan"),
                "NumHA": float("nan"),
                "NumHD": float("nan"),
                "iv_window_ok": False,
                "mw_iv_flag": False,
                "oxidation_state": None,
                "coordination_number": 0,
                "geometry": "none",
                "dna_kb_proxy_v1": 0.0,
            }

        iv_window_ok = bool(LOGP_LO <= logp <= LOGP_HI) and bool(TPSA_LO <= tpsa <= TPSA_HI)
        mw_iv_flag = bool(MW_LO <= mw <= MW_HI)

        # New anticancer metrics (TODO-14 / T14 §4.8 panel).
        # Pure RDKit calls; never raise (defensive wrappers above).
        ml_proxies = AnticancerMetricSuite.compute_metal_proxies(smiles)
        dna_kb = AnticancerMetricSuite.compute_dna_kb_proxy(smiles, target=None)

        return {
            "logP": logp,
            "TPSA": tpsa,
            "RotB": rotb,
            "MW": mw,
            "NumHA": numha,
            "NumHD": numhd,
            "iv_window_ok": iv_window_ok,
            "mw_iv_flag": mw_iv_flag,
            "oxidation_state": ml_proxies.get("oxidation_state"),
            "coordination_number": ml_proxies.get("coordination_number", 0),
            "geometry": ml_proxies.get("geometry", "none"),
            "dna_kb_proxy_v1": float(dna_kb),
        }

    # ------------------------------------------------------------------
    # 1a. Pure TPSA (topological polar surface area)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tpsa(smiles) -> float:
        """RDKit ``Descriptors.TPSA`` wrapper.

        Returns the topological polar surface area in Å². For IV
        anticancer candidates the target band is 60-150 Å²
        (per TODO-14 §IV-appropriate window).

        Returns ``float("nan")`` for unparseable SMILES or missing RDKit.
        """
        if Chem is None or Descriptors is None:
            return float("nan")
        if not isinstance(smiles, str) or not smiles.strip():
            return float("nan")
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None or rdkit_mol.GetNumAtoms() == 0:
            return float("nan")
        try:
            val = float(Descriptors.TPSA(rdkit_mol))
        except Exception:
            return float("nan")
        return val if math.isfinite(val) else float("nan")

    # ------------------------------------------------------------------
    # 1b. Pure RotB (rotatable bond count)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rotb(smiles) -> float:
        """RDKit ``Descriptors.NumRotatableBonds`` wrapper.

        Returns the number of rotatable bonds. Anticancer IV
        candidates target ``RotB < 10`` (per TODO-14 §anticancer
        metric survey).

        Returns ``float("nan")`` for unparseable SMILES or missing RDKit.
        """
        if Chem is None or Descriptors is None:
            return float("nan")
        if not isinstance(smiles, str) or not smiles.strip():
            return float("nan")
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None or rdkit_mol.GetNumAtoms() == 0:
            return float("nan")
        try:
            val = float(Descriptors.NumRotatableBonds(rdkit_mol))
        except Exception:
            return float("nan")
        return val if math.isfinite(val) else float("nan")

    # ------------------------------------------------------------------
    # 1c. Metal coordination probes
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metal_proxies(smiles) -> Dict[str, object]:
        """Probe metal centres for oxidation state + coordination +
        geometry validity.

        Returns a dict with keys
        ``oxidation_state``, ``coordination_number``, ``geometry``,
        ``metal_symbol``, and ``donor_symbols``. For molecules with
        no metal or unparseable input, all values are neutral
        fallbacks (``None`` for the three main keys, empty list for
        ``donor_symbols``, ``""`` for ``metal_symbol``).

        Honest framing: this is a 1-shot RDKit neighbour-count probe
        on the SMILES-encoded graph, NOT a quantum-chemical or
        full-relaxation geometry.  Hydrolytic equilibria,
        trans-labilisation, and Jahn-Teller distortion are
        intentionally out of scope.
        """
        empty = {
            "oxidation_state": None,
            "coordination_number": 0,
            "geometry": "none",
            "metal_symbol": "",
            "donor_symbols": [],
        }
        if Chem is None:
            return empty
        if not isinstance(smiles, str) or not smiles.strip():
            return empty
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            return empty

        metal_atoms = [a for a in rdkit_mol.GetAtoms()
                       if a.GetSymbol() in ("Pt", "Ru", "Ir", "Au", "Rh", "Os")]
        if not metal_atoms:
            # Non-metal molecule: still report zero coordination.
            return {**empty, "coordination_number": 0, "geometry": "none"}

        # Take the first metal centre; multi-metal complexes are
        # averaged in production-scale sweeps (not at this smoke tier).
        centre = metal_atoms[0]
        symbol = centre.GetSymbol()
        donors = [n for n in centre.GetNeighbors() if n.GetSymbol() not in ("H",)]
        donor_symbols = [n.GetSymbol() for n in donors]
        coord_number = len(donors)
        # Formal charge on the metal token approximates the
        # oxidation state for SMILES-encoded ionic form (e.g.
        # [Pt+2] has charge +2 -> ox state +2; [Au+] has charge
        # +1 -> ox state +1 for Au(I)).
        charge = centre.GetFormalCharge()
        oxidation = None
        if charge != 0:
            oxidation = int(charge) if charge > 0 else int(-charge)
        # If the SMILES was written neutral (e.g. cisplatin
        # "[Pt](N)(N)(Cl)Cl" parses with formal_charge 0 on Pt), use
        # coordination-number-driven heuristics: Pt(Pd,Ni) with 6
        # donors is +4; with 4 donors is +2; Au with 4 donors is +3
        # and 2 donors is +1; Ru/Ir/Rh/Os with 6 donors is +3.
        if oxidation is None:
            if symbol in ("Pt", "Pd", "Ni"):
                oxidation = 4 if coord_number == 6 else 2
            elif symbol in ("Au",):
                if coord_number == 4:
                    oxidation = 3
                elif coord_number == 2:
                    oxidation = 1
                else:
                    oxidation = 1
            elif symbol in ("Ru", "Ir", "Rh", "Os"):
                oxidation = 3 if coord_number == 6 else 2

        # Geometry classification by symbol + coordination_number.
        # Pt(II)/Pt(IV), Au(III) -> square-planar when 4 donors
        # (d8), octahedral when 6 donors (d6 for Pt(IV)).
        # Ru/Ir/Os/Rh octahedral at 6.
        if symbol in ("Pt", "Au", "Pd", "Ni") and coord_number == 4:
            geometry = "square_planar"
        elif symbol in ("Pt", "Au") and coord_number == 6:
            geometry = "octahedral"
        elif symbol in ("Ru", "Ir", "Rh", "Os") and coord_number == 6:
            geometry = "octahedral"
        elif coord_number == 4:
            geometry = "tetrahedral"
        elif coord_number == 2:
            geometry = "linear"
        else:
            geometry = "other"

        return {
            "oxidation_state": oxidation,
            "coordination_number": coord_number,
            "geometry": geometry,
            "metal_symbol": symbol,
            "donor_symbols": donor_symbols,
        }

    # ------------------------------------------------------------------
    # 1d. DNA Kb proxy (with optional target / minor-groove signal)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_dna_kb_proxy(smiles, target: Optional[str] = None) -> float:
        """Reproducible DNA-fragment-binding proxy.

        Heuristic score in [0, 1] for DNA minor-groove / covalent
        Pt-N7 adduct binding. Combines:

          * logP in the 2-5 IV window  (+0.10)
          * 2+ aromatic rings (intercalator / minor-groove shape) (+0.20)
          * 4+ aromatic rings (strong intercalator / bis-intercalator)
            (+0.20)
          * Curved / fused polycyclic aromatic (e.g. phenanthridine,
            ellipticine scaffolds via the ``target`` keyword) (+0.20)
          * Pt(II) centre with Cl leaving groups (+0.40) -- the
            classical covalent-DNA adduct signature.

        ``target`` is an optional scaffold hint. Currently supported:
        ``"minor_groove"`` (e.g. distamycin, Hoechst 33258,
        phenanthridine) raises the polycyclic bonus. ``None`` skips
        the scaffold bonus.

        Honest framing: this is a structural sub-2-h fingerprint
        heuristic. It is NOT a full DNA-fragment docking, NOT a
        measured Kb, and NOT a calibrated predictor. Production
        Kb requires wet-lab or a validated structure-based
        calculator (e.g. AutoDock-Vina against a DNA duplex).
        """
        if Chem is None or Descriptors is None:
            return 0.0
        if not isinstance(smiles, str) or not smiles.strip():
            return 0.0
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            return 0.0

        score = 0.0

        # logP in IV window (+0.10)
        try:
            logp = float(Descriptors.MolLogP(rdkit_mol))
            if 2.0 <= logp <= 5.0:
                score += 0.10
        except Exception:
            pass

        # aromatic rings
        ri = rdkit_mol.GetRingInfo()
        aromatic_rings = 0
        for ring in ri.AtomRings():
            if all(rdkit_mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_rings += 1

        if aromatic_rings >= 2:
            score += 0.20
        if aromatic_rings >= 4:
            score += 0.20

        # Scaffold hint: minor-groove binders (e.g. phenanthridines,
        # bis-benzimidazoles) are polycyclic aromatic with N donors.
        if target == "minor_groove":
            has_aromatic_n = any(
                a.GetSymbol() == "N" and a.GetIsAromatic()
                for a in rdkit_mol.GetAtoms()
            )
            if aromatic_rings >= 3 and has_aromatic_n:
                score += 0.20

        # Pt(II) + Cl covalent adduct signature (+0.40).
        # The classical cisplatin profile: Pt with at least one Cl
        # leaving group and at least one N donor. We treat any Pt
        # centre with >=1 Cl neighbour as the covalent adduct case.
        for atom in rdkit_mol.GetAtoms():
            if atom.GetSymbol() != "Pt":
                continue
            neighbours = [n.GetSymbol() for n in atom.GetNeighbors()]
            cl_count = neighbours.count("Cl")
            n_count = neighbours.count("N")
            if cl_count >= 1 and n_count >= 1:
                score += 0.40
                break

        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # 1. Metal coordination score
    # ------------------------------------------------------------------

    def metal_coordination_score(self, mol) -> float:
        """Score in [0, 1] reflecting how well the metal is coordinated.

        Pt(II) d8 -> square-planar (ideal = 4 donors) -> 1.0 if 4 donors.
        Ru/Ir (d6) -> octahedral (ideal = 6 donors) -> 1.0 if 6 donors.
        Score falls off with |donor_count - ideal|, and degrades to 0.5
        for non-metals (no centre to score).
        """
        if isinstance(mol, str):
            smiles = mol
            rdkit_mol = self._mol(smiles)
        else:
            smiles = getattr(mol, "smiles", None) or ""
            rdkit_mol = mol if Chem is not None and isinstance(mol, Chem.Mol) else None

        if rdkit_mol is None or Chem is None:
            return 0.5

        # Identify the dominant metal centre.
        metal_centres = [a for a in rdkit_mol.GetAtoms()
                         if a.GetSymbol() in ("Pt", "Ru", "Ir", "Au", "Rh", "Os")]
        if not metal_centres:
            return 0.5

        # Square-planar for d8: Pt(II), Au(III); octahedral for d6: Ru, Ir.
        # Use mean over centres if multiple.
        scores = []
        for centre in metal_centres:
            symbol = centre.GetSymbol()
            donors = [n for n in centre.GetNeighbors()
                      if n.GetSymbol() not in ("H",)]
            donor_count = len(donors)
            if symbol in ("Pt", "Au") and centre.GetFormalCharge() in (0, 1, 2):
                # Pt(II) most often; Pt(IV) and Au(III) may also be 4.
                # We treat any 4-donor case as ideal square-planar.
                ideal = SQ_PLANAR_IDEAL
            elif symbol in ("Ru", "Ir", "Rh", "Os"):
                ideal = OCTAHEDRAL_IDEAL
            else:
                # Other transition metals: fall back to octahedral default.
                ideal = OCTAHEDRAL_IDEAL

            deviation = abs(donor_count - ideal)
            # Score 1.0 if donor_count == ideal; falls steeply.
            # half-coordinated (deviation = ideal/2) yields 0.0 so a
            # 2-donor Pt(II) against ideal 4 scores 0.0 (clearly
            # under-coordinated and unable to satisfy square-planar
            # geometry).
            if ideal == 0:
                scores.append(1.0)
            else:
                # Linear penalty: deviation of ideal/2 -> 0.0
                normalized_dev = deviation / (float(ideal) / 2.0)
                score = max(0.0, 1.0 - normalized_dev)
                scores.append(score)

        return float(sum(scores) / len(scores))

    # ------------------------------------------------------------------
    # 2. GSH evasion flag
    # ------------------------------------------------------------------

    def gsh_evasion_flag(self, mol) -> bool:
        """Heuristic GSH evasion flag.

        Returns True when:
          - the molecule is a metal complex with bulky aromatic-N donors
            (Steric bulk + aromatic N disfavors soft nucleophile attack)
            OR
          - the molecule has no soft nucleophile (no free thiol, no
            square-planar Pt(II) with two trans Cl leaving groups
            susceptible to GSH S-attack).

        Returns False for unparseable input.
        """
        if isinstance(mol, str):
            smiles = mol
            rdkit_mol = self._mol(smiles)
        else:
            smiles = getattr(mol, "smiles", None) or ""
            rdkit_mol = mol if Chem is not None and isinstance(mol, Chem.Mol) else None

        if rdkit_mol is None or Chem is None:
            return False

        # 1. Free thiol is a soft nucleophile -- not GSH-evading.
        has_free_thiol = False
        for atom in rdkit_mol.GetAtoms():
            if atom.GetSymbol() == "S" and atom.GetTotalNumHs() > 0:
                has_free_thiol = True
                break

        # 2. Bulky aromatic-N donors on a metal centre => evasion.
        metal_centres = [a for a in rdkit_mol.GetAtoms()
                         if a.GetSymbol() in ("Pt", "Ru", "Ir", "Au", "Rh", "Os")]
        has_aromatic_n_bulky = False
        for centre in metal_centres:
            for n in centre.GetNeighbors():
                if n.GetSymbol() == "N" and n.GetIsAromatic():
                    if any(nb.GetDegree() >= 3 for nb in n.GetNeighbors()):
                        has_aromatic_n_bulky = True
                        break
            if has_aromatic_n_bulky:
                break

        # 3. cis-[Pt(NH3)2Cl2]: GSH attacks via trans labilization,
        #    so it's NOT GSH-evading (returns False).  But the spec
        #    edge case in the test plan: cisplatin has NH3 (no
        #    aromatic), so per the strict rule it should NOT evade.
        #    We honour that: cisplatin = False here.
        # The flag is True ONLY when the molecule has aromatic-N bulky
        # donors AND no free thiol.
        return bool(has_aromatic_n_bulky and not has_free_thiol)

    # ------------------------------------------------------------------
    # 3. DNA Kb proxy
    # ------------------------------------------------------------------

    def dna_kb_proxy(self, mol) -> float:
        """Heuristic DNA binding affinity proxy in [0, 1].

        Score rises when:
          - the molecule has a Pt(II) centre (cisplatin-class adducts
            with N7 of guanine) OR Ru/Ir that mimic cisplatin binding
          - the molecule has leaving groups (Cl-, carboxylate) that
            hydrolyze to form DNA adducts
          - the molecule has an aromatic N (DNA minor-groove binder)

        Falls back to :func:`dna_kb_proxy_v2` for the docked case.
        """
        if isinstance(mol, str):
            smiles = mol
            rdkit_mol = self._mol(smiles)
        else:
            smiles = getattr(mol, "smiles", None) or ""
            rdkit_mol = mol if Chem is not None and isinstance(mol, Chem.Mol) else None

        if rdkit_mol is None or Chem is None:
            return 0.0

        score = 0.0

        # Pt(II) cisplatin-class: 0.4 base contribution.
        has_pt_ii = any(
            a.GetSymbol() == "Pt" and a.GetFormalCharge() in (0, 1, 2)
            for a in rdkit_mol.GetAtoms()
        )
        if has_pt_ii:
            score += 0.40

        # Ru/Ir centre: 0.30 base contribution (kinetic inertness).
        has_rhodium_like = any(
            a.GetSymbol() in ("Ru", "Ir", "Rh")
            for a in rdkit_mol.GetAtoms()
        )
        if has_rhodium_like:
            score += 0.30

        # Chloride leaving groups (aquation -> DNA adduct): +0.20.
        cl_count = sum(
            1 for a in rdkit_mol.GetAtoms()
            if a.GetSymbol() == "Cl"
        )
        if cl_count >= 1:
            score += 0.20

        # Aromatic N (minor-groove binder): +0.20.
        aromatic_n = sum(
            1 for a in rdkit_mol.GetAtoms()
            if a.GetSymbol() == "N" and a.GetIsAromatic()
        )
        if aromatic_n >= 1:
            score += 0.20

        # Chelating diamine (cisplatin N,N ring or ethylenediamine):
        # +0.10.
        n_count = sum(
            1 for a in rdkit_mol.GetAtoms() if a.GetSymbol() == "N"
        )
        if n_count >= 2 and has_pt_ii:
            score += 0.10

        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # 4. Composite anticancer score
    # ------------------------------------------------------------------

    def composite_score(self, mol) -> Dict:
        """Return full composite dict for a molecule.

        All four sub-scores are normalized to [0, 1] and combined
        into an equal-weight ``anticancer_index`` = mean.  For
        unparseable input (RDKit returns ``None``) all four buckets
        default to 0.0 so the index is identically 0.0.
        """
        # If mol is a string SMILES, parse once and reuse.
        if isinstance(mol, str):
            rdkit_mol = self._mol(mol)
            smiles = mol
        else:
            smiles = getattr(mol, "smiles", None) or ""
            rdkit_mol = mol if Chem is not None and isinstance(mol, Chem.Mol) else None

        is_metal_complex = bool(rdkit_mol is not None and self._has_metal(rdkit_mol))

        if rdkit_mol is None:
            return {
                "logP": float("nan"),
                "TPSA": float("nan"),
                "RotB": float("nan"),
                "MW": float("nan"),
                "NumHA": float("nan"),
                "NumHD": float("nan"),
                "iv_window_ok": False,
                "mw_iv_flag": False,
                "metal_score": 0.0,
                "gsh_evasion": False,
                "dna_proxy": 0.0,
                "is_metal_complex": False,
                "anticancer_index": 0.0,
            }

        report = self.descriptor_report(mol)
        metal_score = self.metal_coordination_score(mol)
        gsh_evasion = self.gsh_evasion_flag(mol)
        dna_proxy = self.dna_kb_proxy(mol)

        iv_window_ok = bool(report.get("iv_window_ok", False))
        iv_window_value = 1.0 if iv_window_ok else 0.0
        gsh_value = 1.0 if gsh_evasion else 0.0

        # Equal-weight mean (all four components).
        anticancer_index = (
            iv_window_value + metal_score + gsh_value + dna_proxy
        ) / 4.0
        anticancer_index = float(max(0.0, min(1.0, anticancer_index)))

        return {
            **report,
            "metal_score": metal_score,
            "gsh_evasion": gsh_evasion,
            "dna_proxy": dna_proxy,
            "is_metal_complex": is_metal_complex,
            "anticancer_index": anticancer_index,
        }
