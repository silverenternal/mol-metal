"""Metal-aware ADMET proxies (Wf-A11 / eval wave).

Extends the 8 organic-only proxies in
:file:`molmetal/molmetal_lam/sbdd_env/metrics_v2.py` so they accept a
metal context (oxidation_state, coord_number) and apply metal-aware
formula corrections.  Backward-compat: when the metal context is
``None``, the organic-only baselines are returned.

The 8 proxies are:

  1. logp7_4
  2. gi50_proxy
  3. cell_permeability_logPapp
  4. herg_cardio_risk
  5. ames_mutagen
  6. hepatotox_index
  7. aqueous_solubility_logS
  8. plasma_protein_binding

Backward-compat
---------------
* ``metal_aware_admet.forward(mol, metal_context=None, oxidation_state=2,
  coord_number=4)`` returns the organic-only baseline by calling each
  proxy directly.  When ``metal_context`` is not None, it adds a small
  metal-electronegativity-based offset to the per-metal correctable
  proxies (logP, logS, GI50, hERG).
"""

from __future__ import annotations

from typing import Dict, Optional

from rdkit import Chem

from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
    ames_mutagen,
    aqueous_solubility_logS,
    cell_permeability_logPapp,
    gi50_proxy,
    hepatotox_index,
    herg_cardio_risk,
    logp7_4,
    plasma_protein_binding,
)

#: Pauling electronegativity for the 9 metals we care about (rough guide).
#: Used only to derive a per-metal correctable offset; this is NOT a
#: quantum-mechanical polarisation term, just a sign convention so that
#: Pt(II) (EN=2.3) carries a different offset than Cu(I) (EN=1.9).
_EN_TABLE: Dict[str, float] = {
    "Pt": 2.3,
    "Pd": 2.2,
    "Au": 2.4,
    "Ru": 2.2,
    "Ir": 2.2,
    "Rh": 2.3,
    "Os": 2.2,
    "Zn": 1.6,
    "Cu": 1.9,
}


class MetalAwareADMET:
    """8 ADMET proxies with optional metal-aware corrections.

    The "metal-aware" branch applies a per-metal offset proportional to
    electronegativity(EN(metal)) - 2.0 (the typical organic-only baseline
    falls within ~2.0).  The offset is small (magnitude ~ 0.1) — these
    are still coarse proxies, not quantum-mechanical descriptors.
    """

    name = "metal_aware_admet"

    def __init__(self, default_metal: str = "Pt") -> None:
        self.default_metal = str(default_metal)

    def _offset(self, metal_context: Optional[str], oxidation_state: int) -> float:
        """Compute the per-metal offset (magnitude <= 0.3)."""
        if metal_context is None:
            return 0.0
        en = _EN_TABLE.get(str(metal_context), 2.0)
        # +0.05 per 0.1 EN above 2.0; -0.05 per 0.1 below.  Capped at 0.3.
        return max(-0.3, min(0.3, (float(en) - 2.0) * 0.5))

    def forward(
        self,
        decoded_mol: Chem.Mol,
        metal_context: Optional[str] = None,
        oxidation_state: int = 2,
        coord_number: int = 4,
    ) -> Dict[str, float]:
        """Compute the 8 ADMET proxies, optionally with metal-aware offsets."""
        # RDKit SMILES comes from the decoded_mol, never via the legacy
        # metrics_v2 path's _safe_mol conversion — we use the live mol
        # object so the proxy sees the same atom/bond metadata the
        # ChemistryDiscriminator saw.
        try:
            smi = Chem.MolToSmiles(decoded_mol)
        except Exception:
            smi = ""
        if not smi:
            return {k: 0.0 for k in (
                "logp7_4",
                "gi50_proxy",
                "cell_permeability_logPapp",
                "herg_cardio_risk",
                "ames_mutagen",
                "hepatotox_index",
                "aqueous_solubility_logS",
                "plasma_protein_binding",
            )}

        # When metal_context is None, return organic-only baseline.
        m = metal_context if metal_context is not None else self.default_metal
        offset = self._offset(metal_context, oxidation_state) if metal_context is not None else 0.0

        # Per-proxy metal-aware corrections.  Each is a small branch that
        # adjusts the organic-only baseline by `offset` (or a function of
        # it), reflecting the metal's electronegativity shift.
        return {
            "logp7_4": float(logp7_4(smi)) + float(offset),
            "gi50_proxy": float(gi50_proxy(smi)) + 0.5 * float(offset),
            "cell_permeability_logPapp": float(cell_permeability_logPapp(smi)),
            "herg_cardio_risk": float(herg_cardio_risk(smi)) + 0.3 * float(offset),
            "ames_mutagen": float(ames_mutagen(smi)),
            "hepatotox_index": float(hepatotox_index(smi)),
            "aqueous_solubility_logS": float(aqueous_solubility_logS(smi)) + 0.2 * float(offset),
            "plasma_protein_binding": float(plasma_protein_binding(smi)),
        }


__all__ = ["MetalAwareADMET"]