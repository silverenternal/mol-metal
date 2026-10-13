"""Regularization losses — 4 of L8's 8 DEAD terms, as honest placeholders.

These classes close the structural part of L8 (8/12 DEAD terms per
``cfm_loss_term_liveness.json``). Note: ``assemble_term`` is NOT in this
group — it is LIVE-but-INERT (``requires_grad=True`` + ``value=0``) per
D-NEW-2, a structurally DIFFERENT condition from "DEAD".

Every class returns ``torch.zeros(())`` when disabled. The shipped YAMLs
do not enable them either, so the 12-term sum is bit-exact identical to
pre-Wf-A13 on the default path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class EntropyTerm:
    """Maximise H(softmax(logits)) — bit-exact counterpart of inline 2999-3005.

    Sign is NEGATIVE so the optimiser maximises entropy. Disabled by
    default; opt-in via ``loss.entropy_reg_weight > 0``.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "entropy_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        logits: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or logits is None
            or logits.numel() == 0
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -((log_probs.exp() * (log_probs + 1e-9)).sum(dim=-1).mean())
        # NEGATIVE sign: optimiser minimises loss to maximise entropy.
        return -float(self.weight) * entropy

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class ConsistencyTerm:
    """Placeholder — returns ``0`` when disabled."""

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "consistency_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        # TODO: real consistency loss lands in a sibling wave.
        return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class SO3Term:
    """SO(3) augmentation consistency placeholder."""

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "so3_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class EDMTerm:
    """sigma-preconditioning placeholder (EDM, Karras 2022 §5)."""

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "edm_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.weight <= 0.0:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
