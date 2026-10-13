"""Single-fragment / molecular-integrity loss — A.6 primary fix.

Targets the A.6 headline from ``docs/CFM_WEAKNESSES_2026_10_13.md``:

    A.6 — 50.4% of decoded molecules are ≥2 fragments; 23% of
    molecules have the metal atom with zero bonds (``metal-coord-1
    rate = 1.0`` per ``molmetal/eval/discriminators/chemistry.py``).
    The ``assemble_term arithmetic_mean`` fix (molmetal/loss/aggregation_losses.py)
    closes the assemble-side path; this class closes the
    fragment-side path so the head is penalised for emitting multi-
    fragment geometry.

Forward signature:

    forward(decoded_smiles_batch) -> scalar tensor

Behaviour:
    * Per parsed mol: ``n_frags = len(Chem.GetMolFrags(m))``.
    * If ``n_frags == 1``: penalty = 0 (good).
    * Else: penalty = 1.0 (binary).
    * loss = mean(penalty) * weight — same arithmetic as the discriminator
      that ``molmetal/eval/discriminators/topology.py::TopologyDiscriminator``
      reports as ``single_fragment_rate``.

Default ``weight = 0.0`` keeps the 12-term trainer sum bit-exact.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch

LOG = logging.getLogger("molmetal.loss.fragment_losses")


def _try_import_rdkit():
    try:
        from rdkit import Chem
        return Chem
    except ImportError:
        return None


class SingleFragmentLoss:
    """A.6 — binary penalty on multi-fragment decoded molecules.

    Loss is the batch-mean of:

        per_mol = (n_frags(m) != 1)

    which is 0.0 when every decoded mol is connected, ~0.5 when half
    the batch is multi-fragment (the current shipped baseline), and 1.0
    when every decoded mol is disconnected.  The training-time signal
    is therefore a soft pressure on the bond head to keep every decoded
    graph connected.
    """

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = float(weight)
        self._Chem = _try_import_rdkit()
        if self._Chem is None and self.weight > 0:
            LOG.warning("SingleFragmentLoss enabled but rdkit not importable; returning zero.")

    def compute(
        self,
        decoded_smiles_batch: Optional[Sequence[str]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.weight == 0.0 or not decoded_smiles_batch or self._Chem is None:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        penalties: List[float] = []
        for smi in decoded_smiles_batch:
            if not isinstance(smi, str) or not smi:
                continue
            try:
                m = self._Chem.MolFromSmiles(smi)
            except Exception:
                m = None
            if m is None:
                # Unparseable SMILES — count as a fragment (penalty=1).
                # This matches the eval-time treatment of decode failures
                # (decode_ratio counts them as failures, not as successes).
                penalties.append(1.0)
                continue
            try:
                n_frags = len(self._Chem.GetMolFrags(m))
            except Exception:
                n_frags = 1
            penalties.append(0.0 if n_frags == 1 else 1.0)
        if not penalties:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        mean_pen = sum(penalties) / len(penalties)
        loss = mean_pen * float(self.weight)
        return torch.tensor(loss, device=device) if device is not None else torch.tensor(loss)


__all__ = ["SingleFragmentLoss"]
