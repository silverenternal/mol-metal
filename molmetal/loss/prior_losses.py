"""Prior losses — standalone clash penalty.

Independent of the model-side ``_compute_clash_penalty`` so that the
YAML's ``loss.clash_penalty_weight_override`` can dial loss-side weight
separately from model-side.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


# Covalent radii (Cordero 2008, single-bond) for the cohort atoms.
_RCOV = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 0.99, 34: 1.20, 35: 1.14,
    53: 1.33, 78: 1.28,
}


class ClashPenalty:
    """Pairwise clash penalty on positions + atomic_numbers.

    ``penalty = exp(-(d - rcov_sum) / sigma)`` summed over pairs whose
    interatomic distance is BELOW ``rcov_sum``. The penalty is positive
    when atoms are too close; the model's minimisation pushes them apart.

    YAML keys
    ---------
    loss.clash_penalty_weight_override: float = 0.01
    loss.clash_penalty_sigma: float = 0.5
    """

    def __init__(
        self,
        weight: float = 0.01,
        enabled: bool = True,
        sigma: float = 0.5,
        name: str = "clash_penalty_loss",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.sigma = float(sigma)
        self.name = str(name)

    def compute(
        self,
        positions: Optional[torch.Tensor],
        atomic_numbers: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or positions is None
            or atomic_numbers is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        n = int(positions.shape[0])
        if n < 2:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        rcov = torch.tensor(
            [_RCOV.get(int(z), 0.77) for z in atomic_numbers.tolist()],
            dtype=positions.dtype,
            device=positions.device,
        )
        # Pairwise distance matrix (n, n).
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        d = diff.pow(2).sum(dim=-1).sqrt()
        # Pairwise rcov sum.
        rcov_sum = (rcov.unsqueeze(0) + rcov.unsqueeze(1)).triu(diagonal=1)
        mask = rcov_sum > 0
        # Penalty only where d < rcov_sum.
        in_range = (d < rcov_sum) & mask
        d_safe = torch.where(in_range, d, torch.ones_like(d))
        # exp(-(d - rcov_sum) / sigma) — positive when d < rcov_sum.
        penalty = torch.where(
            in_range,
            torch.exp(-(d_safe - rcov_sum) / float(self.sigma)),
            torch.zeros_like(d),
        )
        # Take only the upper triangle (avoid double-counting + diagonal).
        penalty = penalty.triu(diagonal=1)
        return penalty.sum() * self.weight

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor
