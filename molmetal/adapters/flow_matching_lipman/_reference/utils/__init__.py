"""Shim — legacy ``utils.ModelWrapper`` for upstream flow_matching samplers.

The legacy Lipman 2023 API exposed ``ModelWrapper`` as a thin callable
adapter that normalises the signature of a velocity net (some models
take ``(x, t, **extras)``, others take ``(x, t)``). The 2024-restructure
upstream ``flow_matching.sampler.FlowMatchingSampler`` takes a callable
directly, but the molmetal adapter still uses ``ModelWrapper``.

This shim provides a minimal ``ModelWrapper`` that handles both legacy
positional signatures.
"""
from __future__ import annotations

from typing import Callable, Optional


class ModelWrapper:
    """Thin callable wrapper that lets a velocity net be called as ``v(x, t)``.

    Handles both:
      - Legacy positional: ``model(x, t)`` -> v(x, t)
      - Modern keyword:    ``model(x, t=t, **extras)`` -> v(x, t)

    The wrapper preserves the underlying model's training/eval mode and
    parameter set; only the call signature is normalised.
    """

    def __init__(self, model: Callable):
        self.model = model
        try:
            import inspect

            self._sig = inspect.signature(model)
            self._has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in self._sig.parameters.values()
            )
            self._param_count = len(self._sig.parameters)
        except (TypeError, ValueError):
            self._sig = None
            self._has_var_keyword = False
            self._param_count = 2

    def __call__(self, x, t, **extras):
        if self._sig is None:
            return self.model(x, t)
        # 1. try 2-arg positional ``(x, t)``
        try:
            return self.model(x, t)
        except TypeError:
            pass
        # 2. try keyword-time (handles ``def forward(x, *, t)``)
        try:
            return self.model(x, t=t)
        except TypeError:
            pass
        # 3. try keyword-time + extras (handles **kwargs)
        try:
            return self.model(x, t=t, **extras)
        except TypeError:
            pass
        # 4. last resort: positional with extras merged
        return self.model(x, t, **extras)


__all__ = ["ModelWrapper"]
