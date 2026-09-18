"""Metal anticancer drug-likeness metrics (Kellogg 1991; Veber 2002).

The hERG heuristic follows Fermini & Coyne, *J. Med. Chem.* 2018, 61:3599.
Missing RDKit or malformed SMILES gracefully return 0.5.
"""
from __future__ import annotations
from typing import Dict
import math

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None

from .metal_hydration import MetalHydrationAnalyzer

_ANTICANCER_COUNTERS: Dict[str, int] = {k: 0 for k in (
    "logp_calls", "tpsa_calls", "rotb_calls", "herg_calls",
    "composite_calls", "in_range_hits", "out_of_range_hits")}

class AnticancerMetricSuite:
    def __init__(self, metal="Pt", oxidation_state=2, logp_range=(2.0, 5.0), tpsa_range=(60.0, 150.0)):
        self.logp_range = tuple(map(float, logp_range)); self.tpsa_range = tuple(map(float, tpsa_range))
        self.hydration = MetalHydrationAnalyzer(metal, oxidation_state)

    def _mol(self, smiles):
        if Chem is None or not isinstance(smiles, str): return None
        try: return Chem.MolFromSmiles(smiles)
        except Exception: return None

    def _band(self, value, bounds, *, logp=False):
        lo, hi = bounds
        if lo <= value <= hi:
            _ANTICANCER_COUNTERS["in_range_hits"] += 1; return 1.0
        _ANTICANCER_COUNTERS["out_of_range_hits"] += 1
        if value < lo:
            return max(0.0, value - (lo - 1.0)) if logp else max(0.0, value / lo)
        return max(0.0, 1.0 - (value - hi) / (1.0 if logp else hi))

    def logp_anticancer(self, smiles: str) -> float:
        _ANTICANCER_COUNTERS["logp_calls"] += 1; mol = self._mol(smiles)
        if mol is None or Descriptors is None: return 0.5
        try: return float(self._band(Descriptors.MolLogP(mol), self.logp_range, logp=True))
        except Exception: return 0.5

    def tpsa_iv(self, smiles: str) -> float:
        _ANTICANCER_COUNTERS["tpsa_calls"] += 1; mol = self._mol(smiles)
        if mol is None or Descriptors is None: return 0.5
        try: return float(self._band(Descriptors.TPSA(mol), self.tpsa_range))
        except Exception: return 0.5

    def rotatable_bonds(self, smiles: str) -> float:
        _ANTICANCER_COUNTERS["rotb_calls"] += 1; mol = self._mol(smiles)
        if mol is None or Descriptors is None: return 0.5
        try:
            n = Descriptors.NumRotatableBonds(mol); return 1.0 if n <= 10 else (0.0 if n > 15 else (15-n)/5.0)
        except Exception: return 0.5

    def _raw_descriptors(self, smiles: str) -> dict[str, float]:
        """Parse once for reporting; incomplete measurements invalidate the row.

        This stricter report path intentionally does not change the historical
        neutral fallback used by reward desirability functions.
        """
        invalid = {name: float("nan") for name in ("logP", "TPSA", "RotB", "MW")}
        if not isinstance(smiles, str) or not smiles.strip() or Descriptors is None:
            return invalid
        mol = self._mol(smiles)
        if mol is None or mol.GetNumAtoms() == 0 or any(
            atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()
        ):
            return invalid
        try:
            values = {
                "logP": float(Descriptors.MolLogP(mol)),
                "TPSA": float(Descriptors.TPSA(mol)),
                "RotB": float(Descriptors.NumRotatableBonds(mol)),
                "MW": float(Descriptors.MolWt(mol)),
            }
        except Exception:
            return invalid
        return values if all(math.isfinite(v) for v in values.values()) else invalid

    def molecular_weight(self, smiles: str) -> float:
        """Return raw MW, or NaN for invalid/incomplete descriptor reports."""
        return self._raw_descriptors(smiles)["MW"]

    def adjusted_flags(self, smiles: str) -> dict[str, bool]:
        """MW <=500 Lipinski flag and inclusive 300–700 Da descriptive range.

        The metal range is descriptive; it is not a validated drug-likeness
        cutoff and is not folded into the composite reward.
        """
        report = self.descriptor_report(smiles)
        return {
            "mw_lipinski": report["MW_lipinski"],
            "mw_metal_adjusted": report["MW_metal_adjusted"],
        }

    def descriptor_report(self, smiles: str) -> dict[str, float | bool]:
        """Raw logP/TPSA/RotB/MW with inclusive bands, strict RotB <10.

        MW_metal_adjusted names TODO14's descriptive 300–700 Da range.
        Empty, wildcard, unparseable or nonfinite rows have NaN descriptors
        and all flags false, including descriptor_valid.
        """
        values = self._raw_descriptors(smiles)
        valid = all(math.isfinite(v) for v in values.values())
        return {
            **values,
            "descriptor_valid": valid,
            "logP_in_range": bool(valid and self.logp_range[0] <= values["logP"] <= self.logp_range[1]),
            "TPSA_in_range": bool(valid and self.tpsa_range[0] <= values["TPSA"] <= self.tpsa_range[1]),
            "RotB_in_range": bool(valid and values["RotB"] < 10.0),
            "MW_lipinski": bool(valid and values["MW"] <= 500.0),
            "MW_metal_adjusted": bool(valid and 300.0 <= values["MW"] <= 700.0),
        }

    def herg_proxy(self, smiles: str) -> float:
        """Heuristic cardiotoxicity score in [0, 1]; HIGHER = SAFER (less hERG risk).

        Honest framing
        --------------
        This is a *descriptor-based proxy* — NOT a wet-lab hERG assay. It is
        intended for ranking generated scaffolds during MCTS reward shaping.
        Do not interpret a passing score as regulatory evidence of safety.

        References
        ----------
        Aronov, A. M. *Predictive in silico modeling for hERG channel blockers.*
        **J. Med. Chem.** 2005, 48, 1289-1300.  doi:10.1021/jm049371n
        The classical Aronov 2005 hERG model emphasises (a) a protonatable
        basic nitrogen centre and (b) lipophilicity as the dominant features,
        with aromatic-ring count and molecular size as secondary contributors.

        Veber, D. F.; Johnson, S. R.; Cheng, H.-Y.; Smith, B. R.; Ward, K. W.;
        Kopple, K. D. *Molecular properties that influence the oral
        bioavailability of drug candidates.* **J. Med. Chem.** 2002, 45,
        2615-2623.  doi:10.1021/jm020017n
        Veber 2002 motivates the "rotatable bonds ≤ 10" and the
        "TPSA / polar-surface" criteria used elsewhere in this suite; here
        the Veber-style molecular-size envelope (MW < 500) is reused as a
        *partial* safeguard (above ~500 Da, hERG promiscuity climbs sharply).

        Feature weights (sum clipped at 1.0; this function returns 1 - risk):

        | Feature                       | Weight  | Rationale                       |
        |-------------------------------|---------|---------------------------------|
        | tertiary basic N count        |  0.70   | Aronov 2005 primary feature     |
        | max(0, logP - 3.5)            |  0.15   | Aronov 2005 secondary feature   |
        | max(0, aromatic_rings - 2)    |  0.15   | accommodates channel pore       |
        | max(0, (MW - 400) / 100) * 0.1|  0.10   | Veber 2002 size envelope        |

        Tertiary basic N definition: nitrogen (Z=7), formal charge == 0,
        and total H count == 0 (i.e., a fully-substituted amine, the
        canonical hERG-pharmacophore signature).
        """
        _ANTICANCER_COUNTERS["herg_calls"] += 1; mol = self._mol(smiles)
        if mol is None or Descriptors is None or mol.GetNumAtoms() == 0: return 0.5
        try:
            basic_n = sum(
                a.GetAtomicNum() == 7
                and a.GetFormalCharge() == 0
                and a.GetTotalNumHs() == 0
                and a.GetAtomicNum() != 0  # exclude wildcard *
                for a in mol.GetAtoms()
            )
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            try:
                from rdkit.Chem import rdMolDescriptors as _rdmd
                aromatic_rings = _rdmd.CalcNumAromaticRings(mol)
            except Exception:
                aromatic_rings = 0

            risk = (
                basic_n * 0.70
                + max(0.0, logp - 3.5) * 0.15
                + max(0.0, aromatic_rings - 2) * 0.15
                + max(0.0, (mw - 400.0) / 100.0) * 0.10
            )
            return float(max(0.0, min(1.0, 1.0 - min(1.0, risk))))
        except Exception: return 0.5

    def composite_score(self, smiles: str) -> float:
        _ANTICANCER_COUNTERS["composite_calls"] += 1
        gsh = getattr(self.hydration, "gsh_resistance_flag", getattr(self.hydration, "gs_resistance_flag", lambda s: False))
        vals = [self.logp_anticancer(smiles), self.tpsa_iv(smiles), self.rotatable_bonds(smiles), self.herg_proxy(smiles), self.hydration.aquation_rate_proxy(smiles), 1.0 if gsh(smiles) else 0.0]
        return float(max(0.0, min(1.0, sum(v*w for v,w in zip(vals, (0.20,0.15,0.10,0.20,0.20,0.15))))))

    def l5_metrics(self): return dict(_ANTICANCER_COUNTERS)

__all__ = ["AnticancerMetricSuite", "_ANTICANCER_COUNTERS"]
