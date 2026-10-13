"""Model-output alignment discriminator (Wf-A11 / eval wave).

Measures how well the bond-head logits + EGNN hidden state agree with
the actual decoded mol.  Useful for detecting cases where the
model's output is internally consistent (high logits on the right
class) but the decoded topology is wrong (e.g. ring closure failed).

Four metrics:

  * ``bond_logits_top1_correctness`` — fraction of pairs whose
    argmax logit equals the real bond class.
  * ``bond_logits_kl_divergence``   — KL(true || softmax(logits)).
  * ``egnn_h_out_norm``             — L2 norm of the EGNN's hidden
    state output (mean across atoms).
  * ``decoded_vs_real_alignment``   — fraction of generated bonds
    that exist in the real target's adjacency.

Backward-compat
---------------
* All four metrics are guarded against ``None`` tensors.  When any
  input is missing, the metric returns ``None`` for that key only.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from rdkit import Chem


class ModelOutputAlignment:
    """Model-output alignment audit over one eval sample."""

    name = "model_output_alignment"

    def forward(
        self,
        bond_head_logits: Optional[torch.Tensor],
        egnn_h_out: Optional[torch.Tensor],
        decoded_mol: Optional[Chem.Mol],
        real_target: Optional[Chem.Mol] = None,
    ) -> Optional[Dict[str, float]]:
        """Return the alignment audit dict for one eval sample."""
        result: Dict[str, float] = {}

        # (1) bond_logits_top1_correctness
        if bond_head_logits is not None and real_target is not None:
            try:
                pred = bond_head_logits.argmax(dim=-1).detach().cpu()
                # Real target bond classes — for now approximate from
                # real_target.GetBonds() bond-type numeric values.
                real_classes: list = []
                real_bonds = real_target.GetBonds()
                for b in real_bonds:
                    real_classes.append(int(b.GetBondType()))
                n = min(len(pred), len(real_classes))
                if n > 0:
                    correct = sum(
                        1 for i in range(n) if int(pred[i]) == int(real_classes[i])
                    )
                    result["bond_logits_top1_correctness"] = float(correct) / float(n)
            except Exception:
                pass

        # (2) bond_logits_kl_divergence
        if bond_head_logits is not None and real_target is not None:
            try:
                log_probs = torch.log_softmax(bond_head_logits, dim=-1).detach().cpu()
                # Approximate true distribution: uniform over real_classes.
                n_classes = int(bond_head_logits.shape[-1])
                real_classes = [int(b.GetBondType()) for b in real_target.GetBonds()]
                if real_classes:
                    p_true = torch.zeros_like(log_probs)
                    for c in real_classes:
                        if 0 <= c < n_classes:
                            p_true[:, c] += 1.0
                    p_true = p_true / p_true.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    kl = (
                        p_true
                        * (
                            torch.log(p_true.clamp(min=1e-9))
                            - log_probs
                        )
                    ).sum(dim=-1).mean().item()
                    result["bond_logits_kl_divergence"] = float(kl)
            except Exception:
                pass

        # (3) egnn_h_out_norm (per-atom L2)
        if egnn_h_out is not None:
            try:
                norms = torch.norm(egnn_h_out.detach().cpu(), dim=-1)
                result["egnn_h_out_norm"] = float(norms.mean().item())
            except Exception:
                pass

        # (4) decoded_vs_real_alignment (bond set intersection / union)
        if decoded_mol is not None and real_target is not None:
            try:
                decoded_bonds = set(
                    tuple(sorted([int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())]))
                    for b in decoded_mol.GetBonds()
                )
                real_bonds = set(
                    tuple(sorted([int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())]))
                    for b in real_target.GetBonds()
                )
                if decoded_bonds or real_bonds:
                    intersect = decoded_bonds & real_bonds
                    union = decoded_bonds | real_bonds
                    result["decoded_vs_real_alignment"] = (
                        float(len(intersect)) / float(len(union))
                        if union else 0.0
                    )
            except Exception:
                pass

        return result if result else None


__all__ = ["ModelOutputAlignment"]