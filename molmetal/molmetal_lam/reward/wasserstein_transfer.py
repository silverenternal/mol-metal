"""Wasserstein transfer regularization for per-metal embedding distributions.

Wf-125 Phase 1 — M1 *application layer* (TODO-121 Wasserstein transfer).

This module is a thin **application layer** that builds on the canonical
M1 Wasserstein primitive shipped by Wf-123 P1 in
:mod:`molmetal.molmetal_lam.reward.wasserstein_loss`:

* :func:`wasserstein_loss.wasserstein_transfer_loss` is the canonical
  M1 primitive that computes the pairwise W₂ sum over the
  ``{metal: Tensor(n_k, d)}`` embedding dict.
* This module (:mod:`wasserstein_transfer`) wraps that primitive in
  per-metal Wf-125 diagnostic classes (:class:`WassersteinTransferLoss`,
  :class:`WassersteinTransferResult`, :class:`WassersteinPairResult`)
  plus a per-pair snapshot reporter that the Wf-125 P1 driver
  (:mod:`molmetal.scripts.wasserstein_transfer_rh`) consumes to
  populate ``synthetic_rh_cohort.json`` and ``wasserstein_decay.json``.

Background
==========
The Wasserstein-2 (W₂) gradient-flow framework (Xue 2026,
``molmetal.math_skills.wasserstein_gradient_flow``) gives us a
*canonical* regulariser for cross-metal generalisation: when we
project per-metal molecular embeddings into a shared latent space, we
want the per-metal marginal distributions to be **as close as possible
to one another** so the shared downstream head can transfer signal.

The Wf-125 P1 driver applies the canonical
:func:`wasserstein_loss.wasserstein_transfer_loss` to per-metal
embeddings derived from the MetalCytoToxDB 5-metal cohort and
measures the **Wasserstein distance decay curve** as synthetic Rh
compounds are interpolated into the real Rh cohort.  This module
provides the diagnostic classes that capture per-pair + per-metal
diagnostic snapshots for the Wf-125 P1 verification JSON.

Honest framing
==============
* The decay curve measures whether the synthetic Rh compounds pull
  the real Rh marginal *closer* to the other-metal marginals.  The
  measured signal is **+38% (W₂ increases)** — the synthetic cohort
  is a *diversity* augmentation (different chemotypes) rather than a
  *density* augmentation (closer-to-real samples).  See the Wf-125
  P1 synthesis.md §4-§5 for the full discussion.
* This module is the Wf-125-specific diagnostic layer; the *real*
  mathematical work lives in
  :func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`
  (the canonical M1 primitive shipped by Wf-123 P1).

CPU-only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-pair / per-metal diagnostic dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WassersteinPairResult:
    """Diagnostic snapshot for one (anchor, partner) metal pair.

    Attributes
    ----------
    anchor : str
        Source metal symbol (e.g. ``"Ru"``).
    partner : str
        Target metal symbol (e.g. ``"Rh"``).
    n_anchor : int
        Number of anchor samples used in the estimate.
    n_partner : int
        Number of partner samples used in the estimate.
    w2_distance : float
        W₂ distance between the two embedding distributions.
    embedding_dim : int
        Embedding dimension (``d``).
    """

    anchor: str
    partner: str
    n_anchor: int
    n_partner: int
    w2_distance: float
    embedding_dim: int


@dataclass
class WassersteinTransferResult:
    """Multi-metal W₂ diagnostic bundle.

    Attributes
    ----------
    per_metal_n : dict
        ``{metal: n_samples}`` for every metal in the input.
    per_pair : list[WassersteinPairResult]
        Per-pair :class:`WassersteinPairResult` for every
        :math:`k < k'` pair.
    total_w2 : float
        Sum of pairwise :class:`WassersteinPairResult.w2_distance`
        values (the quantity the trainer minimises).
    mean_w2 : float
        ``total_w2 / n_pairs``.
    max_w2 : float
        Max of pairwise :class:`WassersteinPairResult.w2_distance`.
    n_pairs : int
        Number of distinct metal pairs (``K(K-1)/2``).
    embedding_dim : int
        Embedding dimension (``d``).
    wall_time_s : float
        Wall-time of the full computation (seconds).
    """

    per_metal_n: Dict[str, int] = field(default_factory=dict)
    per_pair: List[WassersteinPairResult] = field(default_factory=list)
    total_w2: float = 0.0
    mean_w2: float = 0.0
    max_w2: float = 0.0
    n_pairs: int = 0
    embedding_dim: int = 0
    wall_time_s: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        """Serialise to a JSON-safe dict."""
        return {
            "per_metal_n": dict(self.per_metal_n),
            "per_pair": [
                {
                    "anchor": p.anchor,
                    "partner": p.partner,
                    "n_anchor": p.n_anchor,
                    "n_partner": p.n_partner,
                    "w2_distance": p.w2_distance,
                    "embedding_dim": p.embedding_dim,
                }
                for p in self.per_pair
            ],
            "total_w2": self.total_w2,
            "mean_w2": self.mean_w2,
            "max_w2": self.max_w2,
            "n_pairs": self.n_pairs,
            "embedding_dim": self.embedding_dim,
            "wall_time_s": self.wall_time_s,
        }


# ---------------------------------------------------------------------------
# WassersteinTransferLoss — diagnostic wrapper around the M1 primitive
# ---------------------------------------------------------------------------


class WassersteinTransferLoss:
    """Wasserstein transfer diagnostic wrapper (Wf-125 P1).

    Thin application-layer wrapper that invokes the canonical
    :func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_transfer_loss`
    primitive once per pair and assembles a
    :class:`WassersteinTransferResult` for the Wf-125 P1 verification
    JSON.

    The wrapper exists because the canonical M1 primitive returns the
    *sum* (a single scalar tensor); the Wf-125 driver wants the
    *per-pair* breakdown plus the per-metal sample counts (for the
    decay curve).

    Parameters
    ----------
    n_mc : int
        Number of random projection directions for the sliced-W₂
        estimator (default 64).
    seed : int
        RNG seed for the projections (default 0).
    reduction : str
        ``"sum"`` (default) or ``"mean"`` — affects the ``loss_value``
        API only; the per-pair diagnostics are unaffected.

    CPU-only.
    """

    _REDUCTIONS = ("sum", "mean")

    def __init__(
        self,
        n_mc: int = 64,
        seed: int = 0,
        reduction: str = "sum",
    ) -> None:
        if reduction not in self._REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._REDUCTIONS}, got " f"{reduction!r}"
            )
        if n_mc < 1:
            raise ValueError(f"n_mc must be >= 1, got {n_mc}")
        self.n_mc = int(n_mc)
        self.seed = int(seed)
        self.reduction = str(reduction)

    def compute(
        self,
        per_metal_embeddings: Mapping[str, Tensor],
    ) -> WassersteinTransferResult:
        """Compute the full multi-metal W₂ diagnostic bundle.

        Parameters
        ----------
        per_metal_embeddings : Mapping
            ``{metal: Tensor(n_k, d)}``.  Each metal must have at
            least one sample.  The dimension ``d`` must be consistent
            across metals (we coerce to ``float32`` and validate).

        Returns
        -------
        WassersteinTransferResult
            Per-pair results plus the aggregate ``total_w2``,
            ``mean_w2``, ``max_w2``, ``n_pairs``, ``wall_time_s``.
        """
        import time

        if not per_metal_embeddings:
            raise ValueError("per_metal_embeddings is empty")

        # Lazy-import the M1 primitive so the diagnostic wrapper can
        # be imported even when wasserstein_loss is mid-edit.
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_transfer_loss as _m1_transfer_loss,
        )

        # Validate + collect per-metal (n_k, d) tensors.
        cleaned: Dict[str, Tensor] = {}
        per_metal_n: Dict[str, int] = {}
        embedding_dim: Optional[int] = None
        for metal, emb in per_metal_embeddings.items():
            if not isinstance(emb, Tensor):
                raise TypeError(
                    f"per_metal_embeddings[{metal!r}] must be Tensor, "
                    f"got {type(emb).__name__}"
                )
            if emb.ndim != 2:
                raise ValueError(
                    f"per_metal_embeddings[{metal!r}] must be 2-D, "
                    f"got {tuple(emb.shape)}"
                )
            if embedding_dim is None:
                embedding_dim = int(emb.size(1))
                cleaned[metal] = emb.to(torch.float32)
            else:
                if emb.size(1) != embedding_dim:
                    raise ValueError(
                        f"per_metal_embeddings[{metal!r}] has dim="
                        f"{emb.size(1)}, expected {embedding_dim}"
                    )
                cleaned[metal] = emb.to(torch.float32)
            per_metal_n[metal] = int(cleaned[metal].size(0))
        if embedding_dim is None:
            embedding_dim = 0
        metals = sorted(cleaned.keys())
        if len(metals) < 2:
            raise ValueError(f"need at least 2 metals for W₂; got {metals}")

        t0 = time.perf_counter()
        per_pair: List[WassersteinPairResult] = []
        for i, anchor in enumerate(metals):
            for partner in metals[i + 1 :]:
                w2_tensor = _m1_transfer_loss(
                    {anchor: cleaned[anchor], partner: cleaned[partner]},
                    n_mc=self.n_mc,
                    seed=self.seed,
                )
                # The M1 primitive returns the *mean* over pairs; for a
                # 1-pair input the mean equals the sum, so we just
                # take the float value.
                w2 = float(w2_tensor.item())
                per_pair.append(
                    WassersteinPairResult(
                        anchor=anchor,
                        partner=partner,
                        n_anchor=per_metal_n[anchor],
                        n_partner=per_metal_n[partner],
                        w2_distance=w2,
                        embedding_dim=embedding_dim,
                    )
                )

        n_pairs = len(per_pair)
        if n_pairs == 0:
            total_w2 = 0.0
            mean_w2 = 0.0
            max_w2 = 0.0
        else:
            distances = [p.w2_distance for p in per_pair]
            total_w2 = float(sum(distances))
            mean_w2 = float(total_w2 / n_pairs)
            max_w2 = float(max(distances))
        wall = time.perf_counter() - t0

        return WassersteinTransferResult(
            per_metal_n=per_metal_n,
            per_pair=per_pair,
            total_w2=total_w2,
            mean_w2=mean_w2,
            max_w2=max_w2,
            n_pairs=n_pairs,
            embedding_dim=int(embedding_dim),
            wall_time_s=wall,
        )

    def loss_value(
        self,
        per_metal_embeddings: Mapping[str, Tensor],
    ) -> float:
        """Scalar loss value (sum or mean of pairwise W₂)."""
        result = self.compute(per_metal_embeddings)
        if self.reduction == "sum":
            return result.total_w2
        return result.mean_w2


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def pairwise_w2_matrix(
    per_metal_embeddings: Mapping[str, Tensor],
    n_mc: int = 64,
    seed: int = 0,
) -> Tuple[List[str], Tensor]:
    """Compute the full pairwise W₂ matrix.

    Returns
    -------
    metals : list[str]
        Sorted metal labels (rows / columns of the matrix).
    matrix : Tensor
        Square ``(K, K)`` symmetric matrix with zero diagonal and
        pairwise W₂ distances on the off-diagonal entries.  The
        matrix is symmetric by construction.
    """
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        wasserstein_transfer_loss as _m1_transfer_loss,
    )

    metals = sorted(per_metal_embeddings.keys())
    K = len(metals)
    matrix = torch.zeros(K, K, dtype=torch.float32)
    for i, m_a in enumerate(metals):
        for j, m_b in enumerate(metals):
            if j <= i:
                continue
            w2_t = _m1_transfer_loss(
                {m_a: per_metal_embeddings[m_a], m_b: per_metal_embeddings[m_b]},
                n_mc=int(n_mc),
                seed=int(seed),
            )
            w2 = float(w2_t.item())
            matrix[i, j] = w2
            matrix[j, i] = w2
    return metals, matrix


__all__ = [
    "WassersteinPairResult",
    "WassersteinTransferResult",
    "WassersteinTransferLoss",
    "pairwise_w2_matrix",
]
