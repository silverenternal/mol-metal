"""Molecular-property-target loss — A.4 fix.

Targets the A.4 headline from ``docs/CFM_WEAKNESSES_2026_10_13.md``:

    A.4 — generated molecules deviate from real-tmQM on every chemistry
    axis measured: ROTB 16.86/mol vs 5.18 (3.26x), aromatic-bond
    fraction 0.0012 vs 0.4546 (0.27%), logP 8.09 vs 4.15, heavy-atom
    count 37.15 vs 26.18 (+42%).

This module ships a per-property soft-target loss.  Each target is
``y_real`` (real-corpus median / mean per
``cfm_real_tmqm_sixmetal_baseline.json``) and the loss penalises the
absolute deviation between the decoded-batch mean of each property and
its target.  Weight defaults to 0.0 so the 12-term trainer sum stays
bit-exact pre-Wf-A14.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

LOG = logging.getLogger("molmetal.loss.property_losses")


def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
        return Chem, AllChem, Crippen, Descriptors, rdMolDescriptors
    except ImportError:
        return None, None, None, None, None


# Per-property targets — sourced from
# ``cfm_real_tmqm_sixmetal_baseline.json::arm_B.real_tmqm``.
#   ROTB  = NumRotatableBonds per molecule
#   AROM  = aromatic-bond fraction
#   LOGP  = Crippen MolLogP
#   HEAVY = heavy-atom count per molecule
TARGETS = {
    "ROTB_per_mol": 5.18,
    "aromatic_bond_fraction": 0.4546,
    "logP": 4.15,
    "heavy_atoms_per_mol": 26.18,
}

# Which properties the loss supervises.  Toggleable from yaml via
# ``loss.property_targets_enabled`` (default = all four).
DEFAULT_PROPS = ("ROTB_per_mol", "aromatic_bond_fraction", "logP", "heavy_atoms_per_mol")


class MolecularPropertyTargetLoss:
    """A.4 — penalise per-property deviation from real-tmQM on decoded batch.

    Forward signature:

        forward(decoded_smiles_batch) -> scalar tensor

    Behaviour:
        * For each parsed mol, compute per-mol ROTB / aromatic_bond_fraction
          / logP / heavy_atoms.
        * Average over the batch and form a vector of 4 deviations.
        * loss = (sum of |dev|) * weight.
    """

    def __init__(
        self,
        weight: float = 0.0,
        targets: Optional[dict] = None,
        enabled_props: Optional[Sequence[str]] = None,
    ) -> None:
        self.weight = float(weight)
        self.targets = dict(targets) if targets is not None else dict(TARGETS)
        self.enabled_props = tuple(enabled_props) if enabled_props is not None else DEFAULT_PROPS
        mods = _try_import_rdkit()
        self._Chem, self._AllChem, self._Crippen, self._Descriptors, self._rdMolDescriptors = mods
        if self._Chem is None and self.weight > 0:
            LOG.warning("MolecularPropertyTargetLoss enabled but rdkit not importable; returning zero.")

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.weight == 0.0 or not decoded_smiles_batch or self._Chem is None:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        mols = []
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
        # Per-property batch means.
        per_mol: dict = {k: [] for k in self.enabled_props}
        for m in mols:
            try:
                rotb = int(self._Descriptors.NumRotatableBonds(m))
            except Exception:
                rotb = 0
            try:
                n_arom = sum(1 for b in m.GetBonds() if b.GetIsAromatic())
                n_bonds = m.GetNumBonds()
                arom_frac = (n_arom / n_bonds) if n_bonds else 0.0
            except Exception:
                arom_frac = 0.0
            try:
                logp = float(self._Crippen.MolLogP(m))
            except Exception:
                logp = 0.0
            try:
                heavy = m.GetNumHeavyAtoms()
            except Exception:
                heavy = 0
            if "ROTB_per_mol" in per_mol:
                per_mol["ROTB_per_mol"].append(rotb)
            if "aromatic_bond_fraction" in per_mol:
                per_mol["aromatic_bond_fraction"].append(arom_frac)
            if "logP" in per_mol:
                per_mol["logP"].append(logp)
            if "heavy_atoms_per_mol" in per_mol:
                per_mol["heavy_atoms_per_mol"].append(heavy)
        deviations: list = []
        for k, vs in per_mol.items():
            if not vs:
                continue
            mean_v = sum(vs) / len(vs)
            target = self.targets.get(k)
            if target is None:
                continue
            deviations.append(abs(mean_v - target))
        if not deviations:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # Sum-of-absolute-deviations reduced to scalar.
        loss = sum(deviations) * float(self.weight)
        return torch.tensor(loss, device=device) if device is not None else torch.tensor(loss)


__all__ = [
    "MolecularPropertyTargetLoss",
    "TARGETS",
    "DEFAULT_PROPS",
]
