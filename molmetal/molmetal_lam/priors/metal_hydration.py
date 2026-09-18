"""Heuristic hydration and aquation priors for medicinal metal complexes.

The qualitative defaults are informed by Reedijk (1996), Lippard (1999),
and Hartinger (2006).  These are screening proxies, not measured rate
constants; RDKit is used only for lightweight ligand-structure features.
"""
from __future__ import annotations

from typing import Dict

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

__all__ = ["MetalHydrationAnalyzer"]


class MetalHydrationAnalyzer:
    """Estimate aquation, inertness, GSH resistance and reduction class."""

    DEFAULT_PRIORS: Dict[str, Dict[int, float]] = {
        "Pt": {2: 0.72, 4: 0.25},
        "Ru": {2: 0.22, 3: 0.38},
        "Ir": {3: 0.18},
    }
    DEFAULT_D_ELECTRONS = {"Pt": 8, "Ru": 6, "Ir": 6}
    METAL_GROUP = {"Pt": 10, "Ru": 8, "Ir": 9}

    def __init__(self, metal: str = "Pt", oxidation_state: int = 2, default_pH: float = 7.4):
        self.metal = str(metal).title()
        self.oxidation_state = int(oxidation_state)
        self.default_pH = float(default_pH)
        self._base = self.DEFAULT_PRIORS.get(self.metal, {}).get(self.oxidation_state, 0.5)

    def _mol(self, smiles):
        if Chem is None or not isinstance(smiles, str):
            return None, None
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return None, None
        if mol is None:
            try:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is not None:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                mol = None
        if mol is None:
            return None, None
        metals = [a for a in mol.GetAtoms() if a.GetSymbol() == self.metal]
        return mol, (metals[0] if metals else None)

    def _features(self, smiles):
        mol, metal = self._mol(smiles)
        if mol is None or metal is None:
            return None
        neigh = list(metal.GetNeighbors())
        cl = sum(a.GetSymbol() == "Cl" for a in neigh)
        sulfur = sum(a.GetSymbol() == "S" for a in neigh)
        n = sum(a.GetSymbol() == "N" for a in neigh)
        aromatic_n = sum(a.GetSymbol() == "N" and a.GetIsAromatic() for a in neigh)
        # RDKit may reject aromatic-to-metal kekulization; retain useful
        # ligand-level signals from the parsed molecule in that case.
        if not neigh:
            neigh = [a for a in mol.GetAtoms() if a.GetIdx() != metal.GetIdx()]
            cl = sum(a.GetSymbol() == "Cl" for a in neigh)
            sulfur = sum(a.GetSymbol() == "S" for a in neigh)
            n = sum(a.GetSymbol() == "N" for a in neigh)
            aromatic_n = sum(a.GetSymbol() == "N" and a.GetIsAromatic() for a in neigh)
        # Rings containing a metal-neighbour donor indicate chelation.
        chelate = 0
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if metal.GetIdx() in ring and sum(i in ring for i in [a.GetIdx() for a in neigh]) >= 2:
                chelate += 1
        # More generally, N/O/S ring donors around a metal are chelating motifs.
        if chelate == 0 and len(neigh) >= 2:
            chelate = max(0, len(neigh) - 2) // 2
        pi_acceptor = any(a.GetSymbol() in ("C", "P", "S") and a.GetIsAromatic() for a in neigh)
        bulky = any(a.GetDegree() >= 3 for a in neigh)
        if not bulky:
            # Steric bulk is often on the carbons alpha to an aromatic
            # N-donor (substituted pyridines/bipyridines), rather than on N.
            for a in neigh:
                if a.GetIsAromatic() and a.GetSymbol() == "N":
                    bulky = any(nb.GetDegree() >= 3 for nb in a.GetNeighbors())
                    if bulky:
                        break
        if aromatic_n == 0:
            aromatic_n = sum(a.GetSymbol() == "N" and a.GetIsAromatic() for a in mol.GetAtoms())
        if not bulky:
            bulky = any(a.GetIsAromatic() and a.GetDegree() >= 3 for a in mol.GetAtoms())
        return dict(cl=cl, sulfur=sulfur, n=n, aromatic_n=aromatic_n,
                    chelate=chelate, pi_acceptor=pi_acceptor, bulky=bulky,
                    d=self.DEFAULT_D_ELECTRONS.get(self.metal, 5))

    def aquation_rate_proxy(self, smiles: str) -> float:
        f = self._features(smiles)
        if f is None:
            return 0.5
        value = self._base + 0.10 * f["cl"] - 0.10 * f["chelate"] - 0.06 * f["aromatic_n"] + 0.04 * f["sulfur"]
        if f["d"] <= 6:
            value -= 0.12
        return max(0.0, min(1.0, value))

    def kinetic_inertness_score(self, smiles: str) -> float:
        f = self._features(smiles)
        if f is None:
            return 0.5
        value = 0.25 + 0.07 * f["d"] + 0.14 * f["chelate"] + (0.10 if f["pi_acceptor"] else 0)
        if f["d"] == 6:
            value += 0.12
        return max(0.0, min(1.0, value))

    def gs_resistance_flag(self, smiles: str) -> bool:
        f = self._features(smiles)
        if f is None:
            return False
        return bool(f["bulky"] and f["d"] >= 6 and f["aromatic_n"] >= 1)

    def reduction_potential_class(self, smiles: str) -> str:
        f = self._features(smiles)
        if f is None:
            return "moderately_reduced"
        if self.oxidation_state >= 4 or (self.metal == "Ru" and self.oxidation_state >= 3):
            score = 0
        else:
            score = 1
        score += min(1, f["chelate"] + (1 if f["aromatic_n"] else 0))
        return "easily_reduced" if score == 0 else ("moderately_reduced" if score == 1 else "hard_to_reduce")

    def trans_effect_indicator(self, smiles: str) -> dict:
        """Estimate trans-effect donor strength and the likely labile site.

        The ranking follows the ligand series discussed by Appleton et al.,
        *Coordination Chemistry Reviews* **166**, 313–359 (1997).
        Invalid input or a molecule without this analyzer's metal returns a
        neutral fallback indicator.
        """
        fallback = {"strongest_trans_donor": None,
                    "trans_labilization_score": 0.5,
                    "labile_position_index": None}
        mol, metal = self._mol(smiles)
        if mol is None or metal is None:
            return fallback

        strengths = {"CN-": 1.0, "CO": 0.95, "PR3": 0.9, "C2H4": 0.85,
                     "NO2-": 0.8, "NCS-": 0.7, "Cl-": 0.4,
                     "NH3": 0.3, "H2O": 0.25}

        def donor(atom):
            symbol = atom.GetSymbol()
            # Inspect bonds from the metal-bound atom for characteristic motifs.
            if symbol == "Cl":
                return "Cl-"
            if symbol == "O":
                return "CO" if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetSymbol() == "C"
                                    for b in atom.GetBonds()) else "H2O"
            if symbol == "P":
                return "PR3"
            if symbol == "N":
                if any(b.GetBondTypeAsDouble() >= 2.5 and b.GetOtherAtom(atom).GetSymbol() == "C" for b in atom.GetBonds()):
                    return "CN-"
                if any(b.GetOtherAtom(atom).GetSymbol() == "S" for b in atom.GetBonds()):
                    return "NCS-"
                if any(b.GetOtherAtom(atom).GetSymbol() == "O" for b in atom.GetBonds()):
                    return "NO2-"
                return "NH3"
            if symbol == "C":
                if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetSymbol() == "O" for b in atom.GetBonds()):
                    return "CO"
                if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetSymbol() == "C" for b in atom.GetBonds()):
                    return "C2H4"
                return "CH3-"
            return None

        ligands = [(a, donor(a)) for a in metal.GetNeighbors()]
        ligands = [(a, d) for a, d in ligands if d is not None]
        if not ligands:
            return fallback
        strongest_atom, strongest = max(ligands, key=lambda item: strengths[item[1]])
        score = strengths[strongest]
        labile = None
        # SMILES neighbour order is used as a deterministic proxy for square-planar
        # trans pairs (0↔2 and 1↔3) when Pt(II) has four donors.
        if self.metal == "Pt" and self.oxidation_state == 2 and len(ligands) == 4:
            pos = next(i for i, (a, _) in enumerate(ligands) if a.GetIdx() == strongest_atom.GetIdx())
            opposite = pos + 2 if pos < 2 else pos - 2
            labile = opposite
            score = max(score, strengths[ligands[opposite][1]])
        return {"strongest_trans_donor": strongest,
                "trans_labilization_score": float(max(0.0, min(1.0, score))),
                "labile_position_index": labile}
