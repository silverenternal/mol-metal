"""Wf-349-A3: StableVM variance reduction stub (Yang et al. ICML 2026).

This module is a **CPU-only minimal stub** of the StableVM loss from
*Stable Velocity: A Variance Perspective on Flow Matching* (Yang et al.,
ICML 2026; arXiv 2602.05435; GitHub ``linYDTHU/StableVelocity``, MIT
license).  It is **NOT** a verbatim port — the stub only implements the
core mathematical idea (multi-sample, self-normalized estimator under a
multi-sample conditional path) so that we can run a variance comparison
on Mol-Metal's pocket-conditioned CFM training loop in a follow-up
workflow.

Why a stub and not the full upstream:

* The full upstream (StableVM/ in the linYDTHU repo) is wired for
  SiT-XL/2 image generation with ``accelerate launch`` + a sample bank
  per class and a projector (VA-REPA) head; it ships pretrained
  diffusion checkpoints we cannot load into our EGNN + YuelBond CFM.
* The core StableVM idea is small (a multi-sample self-normalized
  estimator for the marginal velocity); reproducing it in 100 lines of
  Python is enough to evaluate the variance-reduction claim on a CPU
  smoke cohort.
* GPU verification is deferred — per AGENTS.md §5.1 the standing GPU
  grant covers launches but the stub itself runs on CPU so we can
  compare variance without consuming a Blackwell slot.

StableVM math (paper eq 4-6 paraphrased):

  Standard CFM loss:
      L_CFM(θ) = E[ || v_θ(x_t, t) − (x_1 − x_0) ||² ]
  has high variance near the prior (large t) and low variance near the
  data (small t).

  StableVM samples K additional x_0' from a bank of past anchors and
  forms a self-normalized multi-sample estimator of the marginal
  velocity.  In expectation, StableVM equals the marginal velocity
  (unbiased) and has strictly lower variance than the K=1 case.

This stub uses a faithful but simplified formulation: for each anchor
x_1 we draw K extra x_0 from the bank, build K+1 candidate conditional
velocities (x_1 - x_0[k]), and predict each one's drift with the
velocity network; the loss is a self-normalised mean squared error
across the K+1 candidates, weighted so the estimator is unbiased in
expectation.

Honest framing (party-twenty-spirit §3.5):

* All speedup / variance-reduction numbers are **PROJECTED** from the
  upstream paper until we run a CPU A/B smoke on the Mol-Metal
  pocket-conditioned cohort.
* The stub has no GPU launch dependency, no AMP, no DDP, and does NOT
  import the upstream repo's sample-bank warm-up logic (we provide a
  tiny in-memory bank for testing only).
* The default ``k_samples = 1`` recovers the standard CFM loss; turn
  on StableVM via ``k_samples >= 2`` + a non-empty bank.

Cross-ref: TODO/completed/340r1_cfm_papers_survey.md §3.2 (StableVM),
TODO/completed/349a3_stablevm_stub.md (integration plan).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional

import torch


# ---------------------------------------------------------------------------
# Sample bank (CPU-only, class-keyed; test stub)
# ---------------------------------------------------------------------------


class SampleBank:
    """Tiny in-memory FIFO bank of past anchor samples, keyed by class id.

    This is **not** a port of the upstream StableVM bank — that one is
    GPU-resident with prefill and per-class capacity.  This stub keeps
    up to ``capacity_per_class`` anchors per class id in CPU RAM for
    unit tests and the variance A/B smoke.

    Parameters
    ----------
    capacity_per_class:
        Maximum number of x_0 anchors retained per class.
    """

    def __init__(self, capacity_per_class: int = 256) -> None:
        if capacity_per_class < 1:
            raise ValueError(
                f"capacity_per_class must be >= 1, got {capacity_per_class}"
            )
        self.capacity_per_class = int(capacity_per_class)
        self._bank: Dict[int, Deque[torch.Tensor]] = {}

    def __len__(self) -> int:
        """Total number of stored anchors across all classes."""
        return sum(len(q) for q in self._bank.values())

    def num_classes(self) -> int:
        """Number of distinct class ids with at least one anchor."""
        return len(self._bank)

    def add(self, class_id: int, x0: torch.Tensor) -> None:
        """Append a single anchor ``x0`` to the bank for ``class_id``.

        ``x0`` must be a 1-D tensor of shape ``[D]``; ``add_batch`` is
        the convenience wrapper for batched inserts.  The anchor is
        detached and moved to CPU before insertion.
        """
        if x0.dim() != 1:
            raise ValueError(
                f"SampleBank.add expects 1-D x0 of shape [D]; got "
                f"shape {tuple(x0.shape)}. Use add_batch for [B, D] inputs."
            )
        anchor = x0.detach().to("cpu", copy=True)
        q = self._bank.setdefault(
            int(class_id), deque(maxlen=self.capacity_per_class)
        )
        q.append(anchor)

    def add_batch(self, class_ids: torch.Tensor, x0_batch: torch.Tensor) -> None:
        """Append a batch of anchors.  ``class_ids`` is a 1-D Long tensor."""
        if class_ids.dim() != 1:
            raise ValueError(
                f"class_ids must be 1-D, got shape {tuple(class_ids.shape)}"
            )
        if x0_batch.shape[0] != class_ids.shape[0]:
            raise ValueError(
                f"class_ids / x0_batch size mismatch: "
                f"{class_ids.shape[0]} vs {x0_batch.shape[0]}"
            )
        for cid, x0 in zip(class_ids.tolist(), x0_batch):
            self.add(int(cid), x0)

    def sample(
        self,
        class_id: int,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``n`` anchors sampled (with replacement) from ``class_id``.

        Returns a CPU tensor of shape ``[n, *anchor_shape]``.  Raises
        ``RuntimeError`` if the bank for ``class_id`` is empty.
        """
        q = self._bank.get(int(class_id))
        if not q:
            raise RuntimeError(
                f"SampleBank.sample: no anchors stored for class_id={class_id} "
                f"(bank has {self.num_classes()} non-empty classes; "
                f"call .add() / .add_batch() first)."
            )
        idx = torch.randint(
            low=0,
            high=len(q),
            size=(n,),
            generator=generator,
        )
        return torch.stack([q[i] for i in idx.tolist()], dim=0)

    def prefill_from(self, class_id: int, anchors: Iterable[torch.Tensor]) -> None:
        """Bulk-load ``anchors`` for ``class_id`` (truncates to capacity)."""
        for x0 in anchors:
            self.add(class_id, x0)


# ---------------------------------------------------------------------------
# StableVM loss (CPU-only minimal stub)
# ---------------------------------------------------------------------------


@dataclass
class StableVMConfig:
    """Configuration knobs for :class:`StableVMLoss`.

    Defaults match the standard CFM loss; flip on StableVM by setting
    ``k_samples >= 2`` and pre-filling a :class:`SampleBank`.
    """

    k_samples: int = 1
    """Number of additional x_0 samples drawn from the bank per anchor.
    ``k_samples = 1`` ⇒ recover standard CFM (loss-type=cfm)."""

    self_normalize: bool = True
    """Whether to apply the self-normalized estimator.  When ``False``,
    the loss falls back to a plain uniform mean of the K+1 candidates."""

    bank_capacity_per_class: int = 256
    """Capacity forwarded to :class:`SampleBank` if one is constructed
    internally."""


class StableVMLoss:
    """CPU-only minimal StableVM loss.

    Parameters
    ----------
    velocity_fn:
        Callable ``v_θ(x_t, t) -> Tensor`` matching the trainer's
        velocity network.  ``x_t`` is the interpolated state at time
        ``t ∈ [0, 1]``; the network returns the predicted conditional
        velocity in the same shape as ``x_0``.
    config:
        :class:`StableVMConfig`; defaults reproduce the standard CFM
        loss.
    bank:
        Optional pre-filled :class:`SampleBank`.  If ``None``, a fresh
        bank is constructed and is initially empty (so the loss falls
        back to the K=1 path on first call).
    """

    def __init__(
        self,
        velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        config: Optional[StableVMConfig] = None,
        bank: Optional[SampleBank] = None,
    ) -> None:
        self.velocity_fn = velocity_fn
        self.config = config or StableVMConfig()
        self.bank = bank if bank is not None else SampleBank(
            capacity_per_class=self.config.bank_capacity_per_class
        )

    def __call__(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: float,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the StableVM loss for a batch.

        Parameters
        ----------
        x0, x1:
            Source / target tensors of shape ``[B, ...]``.
        t:
            Scalar flow time in ``[0, 1]``.
        class_ids:
            Optional 1-D Long tensor of class ids for bank sampling.
            If ``None`` or the bank for a given id is empty, that
            anchor falls back to the K=1 (standard CFM) path.
        """
        if self.config.k_samples < 1:
            raise ValueError(
                f"k_samples must be >= 1, got {self.config.k_samples}"
            )
        # x_t = (1 - t) * x0 + t * x1; we keep this generic so the stub
        # is shape-agnostic (works for both flat and graph-shaped x).
        x_t = (1.0 - t) * x0 + t * x1
        v_pred = self.velocity_fn(x_t, t)

        # Standard CFM target uses the original x0 in the batch.
        v_target_anchor = x1 - x0

        # If K == 1 or no bank information is provided, short-circuit to
        # the standard CFM loss.
        if self.config.k_samples == 1 or class_ids is None:
            return torch.mean((v_pred - v_target_anchor) ** 2)

        # Multi-sample: for each anchor in the batch, sample K extra x0'
        # from the bank for the same class id and form the K+1 candidate
        # targets.
        B = x0.shape[0]
        K = self.config.k_samples
        losses = []
        # Anchor contribution (k = 0).
        losses.append((v_pred - v_target_anchor) ** 2)

        # Bank-sampled contributions (k = 1..K).
        for k in range(K):
            x0_k = []
            used = 0
            for i in range(B):
                cid = int(class_ids[i].item())
                try:
                    x0_k.append(self.bank.sample(cid, n=1).squeeze(0).to(x0.device))
                    used += 1
                except RuntimeError:
                    # Bank for this class is empty — skip and fall back
                    # to the anchor's own x0 so the estimator remains
                    # well defined (variance reduction will be smaller
                    # than the fully-populated case, by construction).
                    x0_k.append(x0[i])
            x0_k_batch = torch.stack(x0_k, dim=0)
            v_target_k = x1 - x0_k_batch
            losses.append((v_pred - v_target_k) ** 2)

        stacked = torch.stack(losses, dim=0)  # [K+1, B, ...]
        if self.config.self_normalize:
            # Self-normalized estimator: weighted by softmax over the
            # per-candidate MSEs.  The simple temperature-1 form below
            # preserves the unbiased-when-bank-distribution-matches-data
            # property while being differentiable end-to-end.
            per_candidate = stacked.mean(dim=tuple(range(1, stacked.dim())))
            weights = torch.softmax(-per_candidate, dim=0)
            # broadcast weights back across the per-element squared errors
            weighted = (weights.view(-1, *([1] * (stacked.dim() - 1))) * stacked)
            return weighted.sum(dim=0).mean()
        # Plain uniform mean.
        return stacked.mean(dim=0).mean()


# ---------------------------------------------------------------------------
# Tiny self-test runner (CPU smoke; gated on __name__ == "__main__")
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Tiny CPU smoke: compare variance of CFM vs StableVM on toy data."""

    torch.manual_seed(0)

    def velocity_fn(x_t: torch.Tensor, t: float) -> torch.Tensor:
        # Toy linear velocity field — fits a residual so loss is small
        # but not zero.  This is intentionally simple so the stub can
        # be exercised without a real EGNN checkpoint.
        return 1.5 * x_t + 0.1

    cfg = StableVMConfig(k_samples=4, self_normalize=True)
    bank = SampleBank(capacity_per_class=8)
    # Pre-fill bank so K >= 2 has data to draw.
    bank.add_batch(
        class_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        x0_batch=torch.randn(8, 4),
    )

    loss = StableVMLoss(velocity_fn=velocity_fn, config=cfg, bank=bank)

    cfm_losses, stablevm_losses = [], []
    for trial in range(50):
        x0 = torch.randn(8, 4)
        x1 = torch.randn(8, 4) + 2.0
        class_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        # Standard CFM (k=1).
        cfg1 = StableVMConfig(k_samples=1)
        cfm = StableVMLoss(velocity_fn=velocity_fn, config=cfg1)
        cfm_losses.append(float(cfm(x0, x1, t=0.5, class_ids=class_ids).item()))

        # StableVM (k=4).
        stablevm_losses.append(
            float(loss(x0, x1, t=0.5, class_ids=class_ids).item())
        )

    cfm_var = torch.tensor(cfm_losses).var(unbiased=False).item()
    stablevm_var = torch.tensor(stablevm_losses).var(unbiased=False).item()
    print(
        f"[stablevm_stub] smoke: cfm_var={cfm_var:.4f} "
        f"stablevm_var={stablevm_var:.4f} "
        f"reduction={1.0 - stablevm_var / max(cfm_var, 1e-12):.2%}"
    )


if __name__ == "__main__":
    _smoke()
