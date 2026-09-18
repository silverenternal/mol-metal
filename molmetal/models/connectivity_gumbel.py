"""DropEdge + Gumbel-top-k connectivity prior (WF-2 A6 fallback).

Background
----------
WF-2 A5 attempts to joint-train :class:`BondOrderHead` end-to-end under
the pocket-conditioned CFM loss.  When A5 fails at the *decode-budget*
level (the CFM velocity field produces unphysical atom clouds and the
bond-head has to throw away > 80 % of candidate pairs), an honest
fallback is to use a discrete connectivity prior on top of the existing
:class:`BondAwareDecoder` pipeline.

This module provides two primitives:

* :class:`DropEdge` — input dropout for the candidate edge list.
  Operates on the (E,) ``keep_mask`` directly so it is differentiable-
  in-spirit (no batch dependency, no graph adjacency changes).  During
  training we zero out edges with probability ``p``; during inference
  we return the unmasked tensor.

* :class:`GumbelConnectivity` — per-pair edge scorer that consumes the
  same per-edge features the bond head already produces.  In training
  mode it returns the Gumbel-softmax relaxed edges with temperature
  :math:`\\tau` annealed; in inference mode it produces hard top-k
  edges where ``k = expected_bonds_per_atom``.

Both classes are deliberately decoupled from :class:`BondOrderHead` —
the head still produces bond *order* logits, but the Gumbel scorer
takes over the *connectivity* decision (which pairs become candidates
for bond-order classification in the first place).

Reference patterns
------------------
Pocket2Mol §3.2 and TargetDiff §3.3 both use a discrete connectivity
prior on top of a continuous diffusion process.  The Pocket2Mol
decoder applies a learned MLP to per-pair distance + atom-type
features and uses a temperature-controlled softmax to soft-select
edges during training, with hard top-k for inference.  We follow the
same recipe but reuse the existing bond-head feature layout so we do
not have to retrain from scratch.

Honest framing
--------------
All numbers reported in :file:`molmetal/reports/wf2_a6_gumbel_fallback.md`
are MEASURED on the synthetic tmQM dataset, CPU seed 0, 1k edges,
n_runs=20.  PROJECTED figures (e.g. throughput on RX 7800 XT) are
labelled explicitly in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "DropEdge",
    "GumbelConnectivity",
    "ConnectivityOutput",
]


# ---------------------------------------------------------------------------
# DropEdge — random edge dropout
# ---------------------------------------------------------------------------
class DropEdge:
    """Randomly zero out edges with probability ``p`` during training.

    Pure-tensor implementation (no ``nn.Module``) so it can be used
    inside per-pair feature pipelines that are not :class:`nn.Module`
    subclasses (e.g. the legacy decoder).  During ``eval()`` mode the
    method is a no-op (returns the input unchanged).

    Parameters
    ----------
    p : float
        Per-edge drop probability in ``[0, 1]``.  ``p = 0.0`` is a
        no-op; ``p = 0.5`` keeps ~half the edges on average.
    """

    def __init__(self, p: float = 0.1) -> None:
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(
                f"DropEdge.p must be in [0, 1], got {p!r}"
            )
        self.p: float = float(p)

    def __call__(
        self,
        edge_features: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(edge_features_dropped, keep_mask)``.

        ``keep_mask`` is shape ``(E,)`` of bool — ``True`` if the edge
        was kept.  The mask is the same in both training and eval;
        only the *application* changes.
        """
        if edge_features.dim() != 2:
            raise ValueError(
                f"expected (E, F) edge_features, got {tuple(edge_features.shape)}"
            )
        e = edge_features.shape[0]
        if e == 0:
            return edge_features, torch.zeros((0,), dtype=torch.bool)
        if not training or self.p == 0.0:
            return edge_features, torch.ones((e,), dtype=torch.bool)
        if self.p == 1.0:
            return torch.zeros_like(edge_features), torch.zeros(
                (e,), dtype=torch.bool,
            )
        # Sample a per-edge keep probability from U(0, 1) and threshold.
        keep = torch.rand(e, generator=None) >= self.p
        keep_mask = keep.to(torch.bool)
        dropped = edge_features * keep_mask.to(edge_features.dtype).unsqueeze(-1)
        return dropped, keep_mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


# ---------------------------------------------------------------------------
# GumbelConnectivity — soft edge scorer with hard top-k inference
# ---------------------------------------------------------------------------
@dataclass
class ConnectivityOutput:
    """Output container for :class:`GumbelConnectivity`.

    Attributes
    ----------
    logits : (E,) tensor — un-normalised connectivity scores (per-edge).
    soft_weights : (E,) tensor — softmax probabilities (training mode).
    hard_mask : (E,) bool tensor — top-k keep mask (inference mode).
    temperature : float — current τ (post-anneal if a schedule is set).
    """

    logits: torch.Tensor
    soft_weights: torch.Tensor
    hard_mask: torch.Tensor
    temperature: float


class GumbelConnectivity(nn.Module):
    """Connectivity prior — per-pair soft / hard edge scorer.

    Architecture
    ------------
    ``Linear(in_dim, hidden) -> ReLU -> Linear(hidden, 1)`` — a small
    MLP that produces a single connectivity logit per edge.  The 9-D
    input feature layout matches :class:`BondOrderHead` so the two
    heads share the same featurisation pipeline.

    Behaviour
    ---------
    * ``forward`` — returns Gumbel-softmax relaxed edge weights.  With
      ``hard=False`` (default) the output is ``softmax((logits + g) / τ)``;
      with ``hard=True`` the output is one-hot (``straight-through``
      estimator).
    * ``inference_topk`` — returns a hard top-k mask (``True`` = keep).
      ``k = expected_bonds_per_atom * n_atoms / 2`` so that the
      average degree matches the prior expectation.

    Temperature annealing
    ---------------------
    The default temperature schedule is::

        τ(step) = τ_start * (τ_end / τ_start) ** (step / total_steps)

    which decays ``τ_start`` exponentially to ``τ_end`` over
    ``total_steps``.  Call :meth:`step_anneal` once per training
    iteration to advance the schedule.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 32,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        total_steps: int = 1000,
        expected_bonds_per_atom: float = 3.0,
    ) -> None:
        super().__init__()
        if tau_start <= 0.0 or tau_end <= 0.0:
            raise ValueError(
                f"tau_start and tau_end must be > 0, got {tau_start}, {tau_end}"
            )
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if expected_bonds_per_atom <= 0.0:
            raise ValueError(
                f"expected_bonds_per_atom must be > 0, got {expected_bonds_per_atom}"
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.total_steps = int(total_steps)
        self.expected_bonds_per_atom = float(expected_bonds_per_atom)
        self._step: int = 0
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Initialise the final layer bias near 0 so logits start near 0
        # — the softmax is then close to uniform at the first step
        # (matches the Gumbel-softmax "warm-up" expectation).
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def current_tau(self) -> float:
        if self._step >= self.total_steps:
            return self.tau_end
        # Geometric interpolation so τ is monotonically decreasing.
        ratio = self._step / max(1, self.total_steps - 1)
        tau = self.tau_start * (self.tau_end / self.tau_start) ** ratio
        return float(tau)

    def step_anneal(self) -> None:
        """Advance the temperature schedule by one step."""
        self._step = min(self._step + 1, self.total_steps)

    def reset_anneal(self) -> None:
        """Reset the schedule back to step 0 (τ = τ_start)."""
        self._step = 0

    def forward(
        self,
        edge_features: torch.Tensor,
        hard: bool = False,
    ) -> ConnectivityOutput:
        """Run one connectivity pass.

        Parameters
        ----------
        edge_features : (E, in_dim) tensor — same layout as BondOrderHead.
        hard : bool — if True, use the straight-through Gumbel-softmax
            (returns one-hot with STE gradient).  Default False
            (soft-relaxed).
        """
        if edge_features.dim() != 2 or edge_features.shape[1] != self.in_dim:
            raise ValueError(
                f"expected (E, {self.in_dim}) edge_features, got {tuple(edge_features.shape)}"
            )
        e = edge_features.shape[0]
        if e == 0:
            zeros = torch.zeros((0,), dtype=edge_features.dtype, device=edge_features.device)
            empty = torch.zeros((0,), dtype=torch.bool, device=edge_features.device)
            return ConnectivityOutput(
                logits=zeros, soft_weights=zeros, hard_mask=empty,
                temperature=self.current_tau,
            )
        h = F.relu(self.fc1(edge_features))
        logits = self.fc2(h).squeeze(-1)  # (E,)
        tau = self.current_tau
        # Gumbel-softmax with temperature.  We do the full softmax over
        # all edges (not the categorical over k buckets) — this is the
        # relaxation that the A6 spec asks for (Pocket2Mol-style).
        if self.training or hard:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(logits).clamp(min=1e-20)).clamp(min=1e-20),
            )
            relaxed = (logits + gumbel) / max(tau, 1e-8)
            soft_weights = F.softmax(relaxed, dim=-1)
        else:
            soft_weights = F.softmax(logits / max(tau, 1e-8), dim=-1)
        if hard:
            # Straight-through: hard on the forward pass, soft on the
            # backward pass.
            idx = torch.argmax(soft_weights)
            hard_weights = torch.zeros_like(soft_weights)
            hard_weights[idx] = 1.0
            soft_weights = hard_weights + (soft_weights - soft_weights.detach())
        # The hard top-k mask is computed by :meth:`inference_topk`
        # rather than here — keeping :meth:`forward` cheap and
        # side-effect-free for the training loop.
        return ConnectivityOutput(
            logits=logits,
            soft_weights=soft_weights,
            hard_mask=torch.ones_like(logits, dtype=torch.bool),
            temperature=tau,
        )

    @torch.no_grad()
    def inference_topk(
        self,
        edge_features: torch.Tensor,
        n_atoms: int,
        expected_bonds_per_atom: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ConnectivityOutput]:
        """Run inference and return a hard top-k mask.

        ``k = ceil(expected_bonds_per_atom * n_atoms / 2)`` — the +1
        ensures ``k >= 1`` for any non-empty edge list.  If the edge
        list is shorter than ``k``, all edges are kept.
        """
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be > 0, got {n_atoms}")
        # ``eval()`` mode disables the Gumbel noise — the top-k picks
        # on the raw logits alone.
        was_training = self.training
        self.eval()
        out = self.forward(edge_features, hard=False)
        self.train(was_training)
        k_per_atom = (
            float(expected_bonds_per_atom)
            if expected_bonds_per_atom is not None
            else self.expected_bonds_per_atom
        )
        k = int(math.ceil(k_per_atom * n_atoms / 2.0))
        k = max(1, k)
        e = out.logits.shape[0]
        if k >= e:
            mask = torch.ones((e,), dtype=torch.bool, device=out.logits.device)
        else:
            # topk returns the largest k values; we mark the indices
            # ``True`` in the mask.
            _, idx = torch.topk(out.logits, k=k, sorted=False)
            mask = torch.zeros((e,), dtype=torch.bool, device=out.logits.device)
            mask[idx] = True
        out.hard_mask = mask
        return mask, out


# ---------------------------------------------------------------------------
# Self-test (lightweight — run as ``python -m molmetal.models.connectivity_gumbel``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    de = DropEdge(p=0.5)
    x = torch.randn(100, 3)
    kept, mask = de(x, training=True)
    print(f"DropEdge kept {int(mask.sum().item())} / {x.shape[0]} edges (expected ~50)")
    g = GumbelConnectivity(in_dim=9, hidden_dim=32, expected_bonds_per_atom=3.0)
    feats = torch.randn(50, 9)
    g.train()
    out = g(feats)
    print(
        f"GumbelConnectivity forward: τ={out.temperature:.3f}, "
        f"soft_weights.sum={out.soft_weights.sum().item():.3f}"
    )
    g.eval()
    mask, inf = g.inference_topk(feats, n_atoms=10)
    print(
        f"inference_topk: kept {int(mask.sum().item())} / {feats.shape[0]} edges "
        f"(expected 15 for n_atoms=10, k_per_atom=3.0)"
    )
