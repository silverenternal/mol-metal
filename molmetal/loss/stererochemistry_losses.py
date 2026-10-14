"""Stereo-chemistry-aware loss terms — A.3 + A.7 partial fixes.

Targets two A-class weaknesses from ``docs/CFM_WEAKNESSES_2026_10_13.md``:

* A.3 — 19.8x potential-stereocentre over-generation: ``real tmQM`` median is
  0.316/mol but the CFM decoder emits 6.258/mol (97% of molecules carry ≥1
  unassigned potential centre).  The lever is a soft target on the
  ``FindMolChiralCenters(includeUnassigned=True)`` count so the head is
  penalised for emitting geometry that RDKit can later parse as chiral.

* A.7 — fragment-level structural sanity: penalise decoded molecules with
  more than one disconnected fragment, since ``single_fragment_rate == 1.0``
  is the honest discriminator of "the model produced a connected molecule".

Both classes follow the same backward-compat contract as the rest of
``molmetal/loss``: weight = 0.0 ⇒ exact zero tensor, no allocation beyond
the dict write.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch

LOG = logging.getLogger("molmetal.loss.stererochemistry_losses")

# Real-corpus reference numbers (per `cfm_real_tmqm_sixmetal_baseline.json`).
REAL_TMQM_MEDIAN_POTENTIAL_STEREOCENTRES_PER_MOL = 0.316
REAL_TMQM_FRACTION_WITH_ZERO_STEREOCENTRES = 0.832

# Target band for the model.  We allow a small headroom above real-tmQM
# median because some cancer-metal complexes are genuinely chiral (e.g.
# NAMI-A), but anything past 1.0/mol is over-generation.
TARGET_POTENTIAL_STEREOCENTRES_PER_MOL = 0.5
TARGET_FRACTION_WITH_ZERO_STEREOCENTRES = 0.7


def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        return Chem, rdMolDescriptors
    except ImportError:
        return None, None


class StereoCentreSuppressionLoss:
    """A.3 — soft target on potential-stereocentre count per decoded mol.

    Forward signature:

        forward(decoded_smiles_batch) -> scalar tensor

    Behaviour:
        * If ``decoded_smiles_batch`` is empty / unparseable, returns 0.
        * Otherwise counts potential stereocentres (RDKit
          ``FindMolChiralCenters(includeUnassigned=True)``) per molecule
          and applies a soft hinge:

              per_mol = max(0, n_pot_stereo - TARGET_PER_MOL)
              loss    = mean(per_mol) * weight

    Default ``weight = 0.0`` keeps the 12-term trainer sum bit-exact
    with pre-Wf-A14 baselines.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = True,
        target_per_mol: float = TARGET_POTENTIAL_STEREOCENTRES_PER_MOL,
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.target_per_mol = float(target_per_mol)
        self._Chem, self._rdMolDescriptors = _try_import_rdkit()
        if self._Chem is None and self.weight > 0:
            LOG.warning("StereoCentreSuppressionLoss enabled but rdkit not importable; returning zero.")

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight == 0.0 or not decoded_smiles_batch or self._Chem is None:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        mols: List = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            try:
                m = self._Chem.MolFromSmiles(smi)
            except Exception:
                m = None
            if m is not None:
                mols.append(m)
        if not mols:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        counts: List[int] = []
        for m in mols:
            try:
                # includeUnassigned=True mirrors the SA-gate used in
                # `cfm_sa_stereocentre_control.json` so the count
                # matches the eval metric.
                n = len(self._Chem.FindMolChiralCenters(m, includeUnassigned=True))
            except Exception:
                n = 0
            counts.append(max(0, n - self.target_per_mol))
        per_mol = torch.tensor(counts, dtype=torch.float32)
        if device is not None:
            per_mol = per_mol.to(device)
        loss = per_mol.mean() * float(self.weight)
        return loss


class FragmentConnectivityLoss:
    """A.6 partial — penalise decoded molecules with >1 disconnected fragment.

    Forward signature:

        forward(decoded_smiles_batch) -> scalar tensor

    Behaviour:
        * For each parsed mol, ``n_frags = len(Chem.GetMolFrags(m))``.
        * penalty_per_mol = max(0, n_frags - 1) — multi-fragment is the
          failure mode the 50.4% multifragment headline refers to.
        * loss = mean(penalty_per_mol) * weight.

    Default ``weight = 0.0`` keeps the 12-term sum bit-exact.
    """

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = float(weight)
        self._Chem, _ = _try_import_rdkit()
        if self._Chem is None and self.weight > 0:
            LOG.warning("FragmentConnectivityLoss enabled but rdkit not importable; returning zero.")

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.weight == 0.0 or not decoded_smiles_batch or self._Chem is None:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        penalties: List[int] = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            try:
                m = self._Chem.MolFromSmiles(smi)
            except Exception:
                m = None
            if m is None:
                continue
            try:
                n_frags = len(self._Chem.GetMolFrags(m))
            except Exception:
                n_frags = 1
            penalties.append(max(0, n_frags - 1))
        if not penalties:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        per_mol = torch.tensor(penalties, dtype=torch.float32)
        if device is not None:
            per_mol = per_mol.to(device)
        loss = per_mol.mean() * float(self.weight)
        return loss


__all__ = [
    "StereoCentreSuppressionLoss",
    "FragmentConnectivityLoss",
    "REAL_TMQM_MEDIAN_POTENTIAL_STEREOCENTRES_PER_MOL",
    "REAL_TMQM_FRACTION_WITH_ZERO_STEREOCENTRES",
    "TARGET_POTENTIAL_STEREOCENTRES_PER_MOL",
    "TARGET_FRACTION_WITH_ZERO_STEREOCENTRES",
]
