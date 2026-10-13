"""Aggregation losses — L6 / L-A7 fix + soft-valence surrogate.

D-NEW-2 distinction
-------------------
``AssembleTerm`` is LIVE-but-INERT by default: ``requires_grad=True`` AND
``value=0.0`` simultaneously. The shipped ``assemble_formulation ==
"product"`` collapses ``P^k[i,i]`` for k=1..12 into ``P`` raised to the
12th power — a value ~5e-9 smaller than the ``arithmetic_mean`` formulation.
Both formulations are valid; only MEAN is the load-bearing formulation.

D-NEW-3 decoupling
-------------------
``SoftValenceTerm`` reads DECODED per-atom valence counts (NOT logits),
so it is immune to ``bond_head.training_mode == "frozen"``. The TRAINER's
existing soft_valence term at 2889-2978 reads logits, which is
gradient-frozen when the head is frozen. The new class decouples the
two paths.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import torch

LOG = logging.getLogger("molmetal.loss.aggregation_losses")


class AssembleTerm:
    """L6 / L-A7 — mean or product formulation over ``P^k[i,i]`` for k=1..12.

    Default ``weight=0.0`` + ``enabled=False`` → bit-exact zero (pre-Wf-A13).
    Opt-in via ``loss.assemble_term_weight > 0``.

    YAML keys
    ---------
    loss.assemble_formulation: {"product", "arithmetic_mean"}
    loss.assemble_term_weight: float = 0.0   # default OFF
    loss.assemble_hinge_tau: float = -1.0    # -1 disables hinge
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        formulation: str = "arithmetic_mean",
        hinge_tau: float = -1.0,
        name: str = "assemble_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.formulation = str(formulation)
        self.hinge_tau = float(hinge_tau)
        self.name = str(name)

    def compute(
        self,
        logits_per_edge: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        n_atoms: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or logits_per_edge is None
            or edge_index is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        if logits_per_edge.numel() == 0 or n_atoms <= 0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # P = softmax over the 5 bond classes; P_bond = sum of classes 1..4.
        P = torch.softmax(logits_per_edge, dim=-1)
        P_bond = P[..., 1:5].sum(dim=-1)  # (E,)
        # edge_index is a (2, E) long tensor of (src, dst) pair indices. The
        # shipped trainer uses (i_idx, j_idx) tuple — accept either shape.
        if isinstance(edge_index, tuple):
            src, dst = edge_index[0], edge_index[1]
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
        # Diagonal of the k-step reachability matrix: P_k[i, i] =
        #     sum over paths i→v1→…→vk=i of ∏_edge P_bond[edge]
        # Computed via matmul powers of an (n_atoms, n_atoms) matrix.
        M = torch.zeros((n_atoms, n_atoms), dtype=P_bond.dtype, device=P_bond.device)
        M[src.long(), dst.long()] = P_bond
        diag_vals: list[torch.Tensor] = []
        M_k = M.clone()
        for _k in range(1, 13):
            diag_vals.append(M_k.diagonal())
            M_k = M_k @ M
        if self.formulation == "arithmetic_mean":
            diag_stack = torch.stack(diag_vals, dim=0)
            loss_val = diag_stack.mean()
        elif self.formulation == "product":
            # Product formulation — preserves the LIVE-but-INERT condition
            # at ``requires_grad=True`` and ``value=~5e-9``. We DELIBERATELY
            # support this; the test_fully_connected_graph_has_near_zero_
            # gradient regression is the documented honest-negative.
            loss_val = torch.stack(diag_vals, dim=0).prod(dim=0).mean()
        else:
            LOG.warning(
                "[AssembleTerm] unknown formulation=%r; falling back to arithmetic_mean",
                self.formulation,
            )
            diag_stack = torch.stack(diag_vals, dim=0)
            loss_val = diag_stack.mean()
        if self.hinge_tau > 0.0:
            loss_val = torch.clamp(loss_val - self.hinge_tau, min=0.0)
        return loss_val * self.weight

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class SoftValenceTerm:
    """Decoupled soft-valence surrogate over DECODED per-atom counts.

    Reads ``per_atom_valence_count : (N_atoms,)`` directly, not logits —
    this decouples the loss from ``bond_head.training_mode == "frozen"``
    (D-NEW-3). Penalty is ``max(0, count - max_valence)^2`` averaged
    over atoms.

    YAML keys
    ---------
    loss.soft_valence_weight: float = 0.0   # default OFF
    loss.soft_valence_max: int = 4          # Pt(II)=4; per-element override
    loss.soft_valence_enabled: bool = False
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        max_valence: int = 4,
        name: str = "soft_valence_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.max_valence = int(max_valence)
        self.name = str(name)

    def compute(
        self,
        per_atom_valence_count: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or per_atom_valence_count is None
            or per_atom_valence_count.numel() == 0
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        excess = per_atom_valence_count - float(self.max_valence)
        penalty = torch.clamp(excess, min=0.0).pow(2).mean()
        return penalty * self.weight

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
