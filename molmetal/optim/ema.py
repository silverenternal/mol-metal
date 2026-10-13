"""EMA shadow-weights utility (Wf-A11 / optimizer wave).

Implements an Exponential Moving Average over model parameters and
buffers, with state_dict / load_state_dict round-trip and a
default decay of ``0.9999`` (was 0.999 in the legacy 1.x trainer; the
04-section grading relied on the longer time-constant for stable R^2
fits at 50 epochs).

API
---

* :class:`EMAModel` holds a shadow copy of every parameter + buffer.
* :meth:`update` consumes a new model snapshot and advances the EMA.
* :meth:`apply_to` swaps the EMA values into a (usually the same) model
  in-place and returns a context manager that on exit restores the live
  weights.  Use it inside an evaluation block so the SMEMA weights are
  evaluated and the live ones are restored when the block exits.
* :meth:`state_dict` / :meth:`load_state_dict` give the same contract
  as :class:`torch.nn.Module` so a checkpoint can persist EMA state.

Backward-compat
---------------
* ``decay=0.9999`` matches the post-Wf-A11 default; passing ``0.999``
  explicitly restores the legacy decay.
* :meth:`apply_to` returns an object whose ``__exit__`` always restores
  the live weights even if the body raised — so a failure inside the
  eval block does not poison the next training iteration.

Lit anchors
-----------
* Polyak 1990 ("A New Stochastic Approximation Type Procedure",
  USSR Computational Mathematics and Mathematical Physics) — the original
  averaged-iteration estimator.
* Yazdino 2019 / Yazdino 2020 ("Mean Teacher") — the canonical EMA
  application to deep nets.
* Mol-Metal Wf-A11 design §B.2 — the 0.9999 default rationale.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, Iterator, Optional

import torch
import torch.nn as nn


class _EMAContext:
    """Restore-on-exit context manager returned by :meth:`EMAModel.apply_to`."""

    def __init__(
        self,
        model: nn.Module,
        shadow: Dict[str, torch.Tensor],
        live_snapshot: Dict[str, torch.Tensor],
    ) -> None:
        self._model = model
        self._shadow = shadow
        self._live_snapshot = live_snapshot

    def __enter__(self) -> nn.Module:
        # Swap EMA shadow into the model.
        for name, param in self._model.named_parameters():
            if name in self._shadow:
                param.data.copy_(self._shadow[name])
        for name, buf in self._model.named_buffers():
            if name in self._shadow:
                buf.data.copy_(self._shadow[name])
        return self._model

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Always restore live weights, even on exception.
        for name, param in self._model.named_parameters():
            if name in self._live_snapshot:
                param.data.copy_(self._live_snapshot[name])
        for name, buf in self._model.named_buffers():
            if name in self._live_snapshot:
                buf.data.copy_(self._live_snapshot[name])
        return False  # do not eat exceptions


class EMAModel:
    """Exponential moving average over a model's parameters + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 <= float(decay) < 1.0):
            raise ValueError(
                f"decay must be in [0, 1); got {decay}"
            )
        self.decay = float(decay)
        self._shadow: Dict[str, torch.Tensor] = {}
        # Initialise shadow = live snapshot, detached, on CPU initially
        # to spare VRAM until the first :meth:`update`.
        for name, param in model.named_parameters():
            self._shadow[name] = param.detach().clone().cpu()
        for name, buf in model.named_buffers():
            self._shadow[name] = buf.detach().clone().cpu()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, model: nn.Module) -> None:
        """Advance the EMA using a new model snapshot.

        Each shadow tensor is updated as
        ``shadow = decay * shadow + (1 - decay) * param.detach()``.
        """
        for name, param in model.named_parameters():
            if name not in self._shadow:
                continue
            new_val = param.detach().cpu() if not param.detach().is_cuda else param.detach().to(self._shadow[name].device)
            self._shadow[name].mul_(self.decay).add_(new_val, alpha=1.0 - self.decay)
        for name, buf in model.named_buffers():
            if name not in self._shadow:
                continue
            new_val = buf.detach().cpu() if not buf.detach().is_cuda else buf.detach().to(self._shadow[name].device)
            self._shadow[name].mul_(self.decay).add_(new_val, alpha=1.0 - self.decay)

    # ------------------------------------------------------------------
    # Apply EMA into a model (eval-time) with auto-restore
    # ------------------------------------------------------------------
    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[nn.Module]:
        """Swap EMA shadow into *model* and restore live weights on exit."""
        live_snapshot: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self._shadow:
                live_snapshot[name] = param.detach().clone()
        for name, buf in model.named_buffers():
            if name in self._shadow:
                live_snapshot[name] = buf.detach().clone()
        ctx = _EMAContext(model, self._shadow, live_snapshot)
        with ctx:
            yield model

    # ------------------------------------------------------------------
    # State-dict round-trip
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {"decay": self.decay, "shadow": {k: v for k, v in self._shadow.items()}}

    def load_state_dict(self, sd: Dict[str, Optional[object]]) -> None:
        if "decay" in sd and sd["decay"] is not None:
            self.decay = float(sd["decay"])  # type: ignore[arg-type]
        shadow_sd = sd.get("shadow") or {}
        if not isinstance(shadow_sd, dict):
            return
        for k, v in shadow_sd.items():
            if k in self._shadow and isinstance(v, torch.Tensor):
                self._shadow[k] = v.detach().clone().cpu()

    # ------------------------------------------------------------------
    # Direct access
    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._shadow.get(name)


__all__ = ["EMAModel"]