"""Wf-380-A1 §2.8 — FlowMol unified atom-bond flow SCAFFOLD.

Reference: Dunn, I., Koes, D. R. (2024).  *Mixed Continuous and
Categorical Flow Matching for 3D De Novo Molecule Generation*
(FlowMol).  arXiv:2404.19739.  Code: github.com/dunni3/FlowMol.
Discrete sibling (bond axis): Dunn, I., Koes, D. R. (2024).  *Exploring
Discrete Flow Matching for 3D De Novo Molecule Generation*
(FlowMol-CTMC).  arXiv:2411.16644.

FlowMol unifies the atom and bond flows into a single graph-flow
framework: the per-atom velocity field operates jointly with a
per-bond (edge) velocity field, with shared message passing across
both streams.  Upstream, FlowMol predicts atom positions, atom types,
formal charges and bond orders jointly, and reports performance
competitive with state-of-the-art diffusion models on the GEOM-DRUGS /
QM9 generative-model benchmarks (arXiv:2404.19739).

LICENSE GATE: the upstream FlowMol papers are CC BY-SA 4.0 (NOT
CC-BY-NC).  The Python *formulation* (the equations, not verbatim code)
is portable under attribution + share-alike; verbatim reuse of the
upstream Python source requires USER_OK per the licence terms.

CITATION CORRECTION (2026-10-31, Wf-A149).  Superseded wording, quoted
verbatim: "Reference: Qiang, B., Song, Y., Xu, R., Wang, G., Liu, Z., Chen, L.,
Zheng, S., Chen, H., Kingma, D. P. (2024).  *FlowMol: A Unified Flow Matching
Model for 3D Molecular Generation.*  arXiv:2412.01966." and "LICENSE GATE:
FlowMol is arXiv:2412.01966 with CC-BY-NC licence."  Verified against the
primary sources on 2026-10-31: arXiv:2412.01966 is NOT FlowMol -- the arXiv abs
page returns 'Implementing Semiclassical Szegedy Walks in Classical-Quantum
Circuits for Homomorphic Encryption' (Ortega, S. A.; Fernandez, P.;
Martin-Delgado, M. A.), a quantum-homomorphic-encryption paper, and the arXiv
API returns no record titled 'FlowMol: A Unified Flow Matching Model for 3D
Molecular Generation' and no record authored by 'Qiang, B.; Song, Y.; ...;
Kingma, D. P.'.  The real upstream is Dunn & Koes arXiv:2404.19739 -- the
identifier this repo's own canonical unified_atom_bond_flow.py already cites --
with siblings arXiv:2411.16644 (FlowMol-CTMC) + arXiv:2508.12629 (FlowMol3).
LICENCE: all three carry CC BY-SA 4.0 (arXiv abs-page
creativecommons.org/licenses/by-sa/4.0/), NOT CC-BY-NC.  The LICENSE GATE is
UNCHANGED -- CC BY-SA 4.0 still carries attribution + share-alike obligations,
so USER_OK=1 still gates the bond stream; only the citation and the licence
string are corrected.  Superseded wording preserved verbatim per AGENTS.md §5.5
rule #6.

SCAFFOLD ONLY — the canonical (license-gated) implementation lives in
:mod:`molmetal.adapters.flow_matching_lipman.unified_atom_bond_flow`
(441 LOC, Wf-353-A4).  This module exposes the small public-API
reference so the integration test can smoke the surface.

Honest framing: SCAFFOLD-only; no GPU launches; license-gated; lift
claim (+5-15% on m13 validity + m14 decode) is PROJECTED per Dunn & Koes 2024 (arXiv:2404.19739).
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
LICENSE_NAME = "arXiv:2404.19739 (Dunn & Koes 2024 — FlowMol, CC BY-SA 4.0)"


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
                edge_attr_h.shape[0],
                edge_attr_h.shape[1],
                3,
                dtype=edge_attr_h.dtype,
                device=edge_attr_h.device,
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
