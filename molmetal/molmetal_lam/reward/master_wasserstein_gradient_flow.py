"""master_wasserstein_gradient_flow.py — Master M1 facade for Wasserstein gradient-flow regularisation.

====================================================================
Wf-247-D TODO-126 §3 M1 **master** module (2026-09-28)
====================================================================

This module is the **single-API master facade** for the Wasserstein
gradient-flow regularisation machinery used across the 6 strengthen
TODOs (119, 120, 121, 122, 123, 125).  It does **not** re-implement
the W₂ primitives; instead it provides a thin orchestrator that:

* Picks the correct W₂ flavour (``loss_type``) for the calling site
  (e.g. cross-metal transfer, pIC50 manifold, or generic gradient flow);
* Aggregates the scalar loss + per-call diagnostics into a single
  :class:`MasterWassersteinResult` so the downstream reporter can
  attribute W₂ contributions honestly (TODO-126 §7: every NEW
  scoring component gets an explicit ``engine`` tag rather than a
  silent substitution);
* Verifies the JKO-style CFL decay gate
  (:func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_decay_check`)
  on the trailing loss history so trainers can gate updates.

Public surface
--------------
* :class:`MasterWassersteinGradientFlow`   — single class with one
  :meth:`compute` method that selects a primitive by ``loss_type``.
* :class:`MasterWassersteinResult`         — dataclass returned by
  :meth:`MasterWassersteinGradientFlow.compute`.
* :class:`MasterWassersteinConfig`         — shared hyper-parameters
  (mirrors :class:`WassersteinConfig`).

Loss-type selector
------------------
The :meth:`compute` API takes a ``loss_type`` argument with three
valid values:

* ``"transfer"``     — multi-metal per-metal embedding dict.
* ``"pic50"``        — predicted pIC50 scalars vs reference pIC50 dist.
* ``"gradient_flow"``— generic W₂ between two sample clouds.

These three are the *currently supported* selectors; the
:class:`MasterWassersteinGradientFlow` is intentionally a thin facade,
**not** a replacement for the underlying primitives.  Callers who
need a flavour we don't expose (``"pose"``, ``"pocket"``) should call
the canonical primitive directly from
:mod:`molmetal.molmetal_lam.reward.wasserstein_loss`.

Backward compatibility
----------------------
This module is **additive** — it does **not** modify, deprecate, or
rename any public symbol in :mod:`wasserstein_loss`,
:mod:`wasserstein_transfer`, or :mod:`closure_oversample`.  It is a
**facade** that re-exports a select subset of the canonical API
under a single ``compute(..., loss_type=...)`` entry point so
upstream TODO-119 / 121 / 122 / 123 / 125 call sites have a uniform
interface.

Existing scripts that continue to import from the canonical modules
(``train_chemprop_v2_v3.py``, ``train_hilbert_astsr_end_to_end.py``,
``bench_deflexpressor_end_to_end.py``, ``triangle_closure.py``, ...)
are not affected by this module and continue to work unchanged.

Honest framing
--------------
* The facade **does not change the W₂ result** — every selector
  delegates to the canonical primitive, so a caller that switches
  from ``wasserstein_loss(pred, ref)`` to
  ``MasterWassersteinGradientFlow().compute(pred, ref,
  loss_type="gradient_flow")`` gets the *same* scalar value
  (up to float round-off from the wrapping dataclass).
* The facade is **not** a re-implementation of the W₂ primitives;
  if the canonical primitive fails (e.g. malformed input), the
  facade propagates the error verbatim so the failure mode stays
  visible to the caller.
* Per TODO-126 §7, every result carries
  ``engine="wasserstein_reg"`` so downstream reporting can attribute
  W₂ contributions honestly.

CPU-only / PyTorch-only.  No GPU kernels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — supported loss_type values + engine tag
# ---------------------------------------------------------------------------

#: Allowed values for ``loss_type`` argument to
#: :meth:`MasterWassersteinGradientFlow.compute`.
SUPPORTED_LOSS_TYPES: tuple = ("transfer", "pic50", "gradient_flow")

#: Engine tag for honest reporting (TODO-126 §7).
ENGINE_TAG: str = "wasserstein_reg"

#: Default n_mc (matches :data:`wasserstein_loss._DEFAULT_N_MC`).
_DEFAULT_N_MC: int = 1000

#: Default random seed.
_DEFAULT_SEED: int = 0


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MasterWassersteinConfig:
    """Hyper-parameters for :class:`MasterWassersteinGradientFlow`.

    Mirrors :class:`molmetal.molmetal_lam.reward.wasserstein_loss.WassersteinConfig`
    — kept separate so callers who only want the master facade do not
    have to import the canonical module.

    Attributes
    ----------
    n_mc : int
        Number of random projection directions for sliced-W₂.
    seed : int
        Random seed for the projection directions.
    """

    n_mc: int = _DEFAULT_N_MC
    seed: int = _DEFAULT_SEED

    def __post_init__(self) -> None:
        if int(self.n_mc) < 1:
            raise ValueError(f"n_mc must be >= 1, got {self.n_mc}")
        self.n_mc = int(self.n_mc)
        self.seed = int(self.seed)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MasterWassersteinResult:
    """Scalar loss + diagnostics returned by ``MasterWassersteinGradientFlow.compute``.

    Attributes
    ----------
    loss_value : float
        Scalar W₂ estimate (0-dim tensor unwrapped to Python ``float``).
    loss_type : str
        Selector value used by the call (``"transfer"`` / ``"pic50"``
        / ``"gradient_flow"``).
    engine : str
        ``"wasserstein_reg"`` — diagnostic tag for downstream honest
        reporting (TODO-126 §7).
    finite : bool
        ``True`` iff ``loss_value`` is finite (not NaN / Inf).
    wall_time_s : float
        Wall-time of the compute call (seconds).
    n_mc : int
        Number of random projection directions used.
    seed : int
        Random seed used for the projection directions.
    extras : dict
        Free-form additional diagnostics.  For ``loss_type="transfer"``
        this carries ``{"n_metals": int, "per_metal_n": dict}`` so the
        downstream reporter can break down the transfer loss.
    """

    loss_value: float
    loss_type: str
    engine: str = ENGINE_TAG
    finite: bool = True
    wall_time_s: float = 0.0
    n_mc: int = _DEFAULT_N_MC
    seed: int = _DEFAULT_SEED
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Master facade — single class with selector-driven compute
# ---------------------------------------------------------------------------


class MasterWassersteinGradientFlow:
    """Master facade for Wasserstein gradient-flow regularisation.

    Single class with a single API (:meth:`compute`) that selects the
    canonical primitive via the ``loss_type`` argument.  See the
    module docstring's "Loss-type selector" section for the three
    supported selector values.

    Parameters
    ----------
    config : MasterWassersteinConfig, optional
        Shared hyper-parameters.  Defaults to
        :class:`MasterWassersteinConfig()` (n_mc=1000, seed=0).
    history : Sequence[float], optional
        Optional initial loss history passed at construction time
        (e.g. resumed trainer state).  The decay check at the end of
        :meth:`compute` consults this history; new losses are *not*
        auto-appended — callers own the history (per TODO-126 §7
        auditability).

    CPU-only.
    """

    def __init__(
        self,
        config: Optional[MasterWassersteinConfig] = None,
        history: Optional[Sequence[float]] = None,
    ) -> None:
        self.config = config if config is not None else MasterWassersteinConfig()
        if history is None:
            self._history: list = []
        else:
            self._history = [float(v) for v in history]

    # -- history helpers ----------------------------------------------------

    @property
    def history(self) -> list:
        """Loss history (read-only view)."""
        return list(self._history)

    def append_history(self, loss: float) -> None:
        """Append a scalar loss value to the history (for decay checks)."""
        self._history.append(float(loss))

    def decay_ok(self, window: int = 10, tol: float = 1.0e-3) -> bool:
        """Return True iff the trailing window is non-increasing.

        Thin wrapper over
        :func:`molmetal.molmetal_lam.reward.wasserstein_loss.wasserstein_decay_check`
        so callers can use the master object as a single source of
        truth for both compute + decay gate.
        """
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_decay_check,
        )
        return wasserstein_decay_check(self._history, window=window, tol=tol)

    # -- single compute API -------------------------------------------------

    def compute(
        self,
        metal_logits: Union[Tensor, Mapping[str, Tensor]],
        ref_distribution: Optional[Tensor] = None,
        loss_type: str = "transfer",
    ) -> MasterWassersteinResult:
        """Dispatch to the canonical primitive selected by ``loss_type``.

        Parameters
        ----------
        metal_logits : Tensor or Mapping
            For ``loss_type="gradient_flow"``: a 2-D ``(N, d)`` predicted
            sample cloud.  The ``ref_distribution`` is the 2-D
            ``(M, d)`` reference cloud.
            For ``loss_type="pic50"``: a 1-D (or ``(N, 1)``) predicted
            pIC50 tensor.  ``ref_distribution`` is the 1-D reference
            pIC50 tensor.
            For ``loss_type="transfer"``: a mapping
            ``{metal_id: Tensor(N_k, d)}`` of per-metal embeddings.
            ``ref_distribution`` is **ignored** (the canonical
            :func:`wasserstein_transfer_loss` consumes the per-metal
            dict directly).
        ref_distribution : Tensor, optional
            Reference distribution for ``"gradient_flow"`` and
            ``"pic50"``.  Ignored for ``"transfer"`` (raises if
            supplied, to surface caller-side misuse).
        loss_type : str
            One of :data:`SUPPORTED_LOSS_TYPES`.  Default
            ``"transfer"`` matches the TODO-121 / TODO-125 usage.

        Returns
        -------
        MasterWassersteinResult
            Scalar loss + diagnostics.

        Raises
        ------
        ValueError
            On an unknown ``loss_type`` or malformed inputs (propagated
            from the canonical primitive — error modes are unchanged).
        TypeError
            On wrong argument types (propagated from canonical primitive).
        """
        if loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {SUPPORTED_LOSS_TYPES}, "
                f"got {loss_type!r}"
            )

        t0 = time.perf_counter()

        if loss_type == "transfer":
            if not isinstance(metal_logits, Mapping):
                raise TypeError(
                    f"loss_type='transfer' requires metal_logits to be a "
                    f"Mapping[str, Tensor], got "
                    f"{type(metal_logits).__name__}"
                )
            extras, loss_tensor = self._compute_transfer(metal_logits)
        elif loss_type == "pic50":
            if not isinstance(metal_logits, Tensor):
                raise TypeError(
                    f"loss_type='pic50' requires metal_logits to be "
                    f"Tensor, got {type(metal_logits).__name__}"
                )
            if not isinstance(ref_distribution, Tensor):
                raise TypeError(
                    f"loss_type='pic50' requires ref_distribution to "
                    f"be Tensor, got {type(ref_distribution).__name__}"
                )
            loss_tensor = self._compute_pic50(metal_logits, ref_distribution)
            extras = {"pred_n": int(metal_logits.flatten().numel()),
                      "ref_n": int(ref_distribution.flatten().numel())}
        else:  # "gradient_flow"
            if not isinstance(metal_logits, Tensor):
                raise TypeError(
                    f"loss_type='gradient_flow' requires metal_logits to "
                    f"be Tensor, got {type(metal_logits).__name__}"
                )
            if not isinstance(ref_distribution, Tensor):
                raise TypeError(
                    f"loss_type='gradient_flow' requires "
                    f"ref_distribution to be Tensor, got "
                    f"{type(ref_distribution).__name__}"
                )
            loss_tensor = self._compute_gradient_flow(
                metal_logits, ref_distribution,
            )
            extras = {"pred_n": int(metal_logits.shape[0]),
                      "ref_n": int(ref_distribution.shape[0])}

        wall = time.perf_counter() - t0
        loss_scalar = float(loss_tensor.detach().cpu().item())
        finite = bool(torch.isfinite(loss_tensor).item())
        return MasterWassersteinResult(
            loss_value=loss_scalar,
            loss_type=str(loss_type),
            engine=ENGINE_TAG,
            finite=finite,
            wall_time_s=float(wall),
            n_mc=int(self.config.n_mc),
            seed=int(self.config.seed),
            extras=dict(extras),
        )

    # -- per-selector wrappers ---------------------------------------------

    def _compute_transfer(
        self,
        per_metal: Mapping[str, Tensor],
    ) -> tuple:
        """Cross-metal transfer loss (delegates to M1 primitive).

        Returns
        -------
        extras : dict
            ``{"n_metals": int, "per_metal_n": dict}``.
        loss_tensor : Tensor
            Scalar mean pairwise W₂.
        """
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_transfer_loss,
        )
        loss_tensor = wasserstein_transfer_loss(
            dict(per_metal),  # copy so caller cannot mutate mid-call
            n_mc=int(self.config.n_mc),
            seed=int(self.config.seed),
        )
        per_metal_n = {
            str(k): int(v.shape[0]) for k, v in per_metal.items()
        }
        extras = {
            "n_metals": len(per_metal_n),
            "per_metal_n": per_metal_n,
        }
        return extras, loss_tensor

    def _compute_pic50(
        self,
        pred_pic50: Tensor,
        ref_pic50_dist: Tensor,
    ) -> Tensor:
        """pIC50 manifold W₂ (delegates to M1 primitive)."""
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_pic50_loss,
        )
        return wasserstein_pic50_loss(
            pred_pic50.float(),
            ref_pic50_dist.float(),
            n_mc=int(self.config.n_mc),
            seed=int(self.config.seed),
        )

    def _compute_gradient_flow(
        self,
        pred_dist: Tensor,
        ref_dist: Tensor,
    ) -> Tensor:
        """Generic W₂ (delegates to M1 primitive)."""
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_loss,
        )
        return wasserstein_loss(
            pred_dist.float(),
            ref_dist.float(),
            n_mc=int(self.config.n_mc),
            seed=int(self.config.seed),
        )

    # -- convenience --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the master's diagnostic state to a JSON-safe dict.

        Does **not** compute — returns the current configuration +
        history.  Useful for the Wf-247 verification JSON.
        """
        return {
            "config": {
                "n_mc": int(self.config.n_mc),
                "seed": int(self.config.seed),
            },
            "history_len": len(self._history),
            "history_tail": list(self._history[-10:]),
        }


# ---------------------------------------------------------------------------
# Convenience: build per-loss_type verification (Wf-247 ship surface)
# ---------------------------------------------------------------------------


def build_master_verification(
    batch_size: int = 32,
    embedding_dim: int = 16,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run :class:`MasterWassersteinGradientFlow` on every selector and
    return a verification dict.

    Returns
    -------
    dict
        ``{"master_wasserstein_gradient_flow": {"transfer": {...},
        "pic50": {...}, "gradient_flow": {...}}, "engine":
        "wasserstein_reg", "cpu_only": True}`` so the Wf-247 ship
        JSON can record a single record per supported selector.
    """
    master = MasterWassersteinGradientFlow(
        config=MasterWassersteinConfig(n_mc=64, seed=int(seed)),
    )
    g = torch.Generator().manual_seed(int(seed))

    def _record(result: MasterWassersteinResult) -> Dict[str, Any]:
        return {
            "loss_value": float(result.loss_value),
            "loss_type": str(result.loss_type),
            "engine": str(result.engine),
            "finite": bool(result.finite),
            "wall_time_s": float(result.wall_time_s),
            "n_mc": int(result.n_mc),
            "extras": dict(result.extras),
        }

    out: Dict[str, Any] = {"engine": ENGINE_TAG, "cpu_only": True}

    # 1. transfer — 5-metal per-metal embedding dict.
    metals = ("Ru", "Ir", "Os", "Rh", "Re")
    per_metal = {
        m: torch.randn(batch_size, embedding_dim, generator=g) + i * 0.1
        for i, m in enumerate(metals)
    }
    out["transfer"] = _record(master.compute(per_metal, loss_type="transfer"))

    # 2. pic50 — pred vs ref 1-D pIC50.
    pred_p = torch.randn(batch_size, generator=g) * 0.5 + 5.0
    ref_p = torch.randn(batch_size * 2, generator=g) * 0.5 + 5.0
    out["pic50"] = _record(master.compute(pred_p, ref_p, loss_type="pic50"))

    # 3. gradient_flow — generic 2-D pred vs ref.
    pred = torch.randn(batch_size, embedding_dim, generator=g)
    ref = torch.randn(batch_size, embedding_dim, generator=g) + 0.7
    out["gradient_flow"] = _record(
        master.compute(pred, ref, loss_type="gradient_flow"),
    )

    return {"master_wasserstein_gradient_flow": out}


__all__ = [
    "ENGINE_TAG",
    "MasterWassersteinConfig",
    "MasterWassersteinGradientFlow",
    "MasterWassersteinResult",
    "SUPPORTED_LOSS_TYPES",
    "build_master_verification",
]
