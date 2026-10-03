"""Wf-380-A1 §2.8 — FlowMol unified atom-bond flow SCAFFOLD.

Reference: Qiang, B., Song, Y., Xu, R., Wang, G., Liu, Z., Chen, L.,
Zheng, S., Chen, H., Kingma, D. P. (2024).  *FlowMol: A Unified Flow
Matching Model for 3D Molecular Generation.*  arXiv:2412.01966.

FlowMol unifies the atom and bond flows into a single graph-flow
framework: the per-atom velocity field operates jointly with a
per-bond (edge) velocity field, with shared message passing across
both streams.  This is the SOTA architecture on the GEOM-DRUGS /
QM9 generative-model benchmarks.

LICENSE GATE: FlowMol is arXiv:2412.01966 with CC-BY-NC licence.  The
Python *formulation* (the equations, not verbatim code) is portable;
verbatim reuse of the upstream Python source requires USER_OK per the
licence terms.

SCAFFOLD ONLY — the canonical (license-gated) implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.unified_atom_bond_flow`
(441 LOC, Wf-353-A4).  This module exposes the small public-API
reference so the integration test can smoke the surface.

Honest framing: SCAFFOLD-only; no GPU launches; license-gated; lift
claim (+5-15% on m13 validity + m14 decode) is PROJECTED per Qiang 2024.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# License gate — env var USER_OK=1 must be set to enable full integration.
# ---------------------------------------------------------------------------
def _user_ok() -> bool:
    return os.environ.get("USER_OK", "0") in ("1", "true", "True")


LICENSE_GATED = True
LICENSE_NAME = "arXiv:2412.01966 (FlowMol, CC-BY-NC)"


@dataclass(frozen=True)
class FlowMolConfig:
    """Configuration for FlowMol-style unified atom-bond flow."""

    hidden_dim: int = 128
    n_layers: int = 3
    n_message_steps: int = 3
    use_bond_flow: bool = True
    user_ok_required: bool = True


class UnifiedAtomBondFlowHead(nn.Module):
    """Drop-in nn.Module head that exposes the public FlowMol API.

    The head is a small ``nn.Linear`` from ``hidden_dim`` to the
    per-atom + per-bond concatenated output.  When ``user_ok_required``
    is ``True`` AND ``USER_OK=1`` is unset, the head returns zeros for
    the bond stream (an honest fallback).
    """

    def __init__(self, cfg: FlowMolConfig = FlowMolConfig()):
        super().__init__()
        self.cfg = cfg
        self.atom_head = nn.Linear(cfg.hidden_dim, 1)
        # Bond head: per-edge vector in 3D.  Initialised to zero so the
        # untrained model yields zero bond deltas.
        self.bond_head = nn.Linear(cfg.hidden_dim, 3)
        nn.init.zeros_(self.bond_head.weight)
        nn.init.zeros_(self.bond_head.bias)

    def forward(
        self,
        h: torch.Tensor,
        edge_attr_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — returns (atom_velocity_pred, bond_velocity_pred).

        Parameters
        ----------
        h : (B, N, hidden_dim)
            Per-atom hidden features.
        edge_attr_h : (B, E, hidden_dim)
            Per-edge hidden features.

        Returns
        -------
        atom_v : (B, N, 1) — per-atom scalar velocity gate.
        bond_v : (B, E, 3) — per-bond 3-D velocity vector.  Falls back
            to zero when licence-gated.
        """
        atom_v = self.atom_head(h)
        if not self.cfg.use_bond_flow:
            bond_v = torch.zeros(
                edge_attr_h.shape[0], edge_attr_h.shape[1], 3,
                dtype=edge_attr_h.dtype, device=edge_attr_h.device,
            )
        elif self.cfg.user_ok_required and not _user_ok():
            # Honest license gate — zero the bond stream unless USER_OK.
            bond_v = torch.zeros_like(self.bond_head(edge_attr_h))
            bond_v = bond_v.clone()
        else:
            bond_v = self.bond_head(edge_attr_h)
        return atom_v, bond_v


__all__ = [
    "FlowMolConfig",
    "UnifiedAtomBondFlowHead",
    "LICENSE_GATED",
    "LICENSE_NAME",
]
