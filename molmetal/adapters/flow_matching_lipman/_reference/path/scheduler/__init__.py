"""Shim — legacy ``path.scheduler.CondOTScheduler`` -> upstream ``SamplerConfig``.

The upstream facebookresearch/flow_matching library restructured in 2024:
the OT scheduler was folded into ``flow_matching.sampler_config.SamplerConfig``.
This shim preserves the legacy import path so the adapter can keep using
``CondOTScheduler``.
"""

from __future__ import annotations

from molmetal.adapters.flow_matching_lipman._reference.upstream_flow_matching.sampler_config import (  # noqa: E501
    SamplerConfig as _UpstreamSamplerConfig,
)


class CondOTScheduler:
    """Facade for the legacy CondOTScheduler (Lipman et al. 2023, §4.7).

    Maps onto upstream ``SamplerConfig`` with sensible defaults that match
    the OT-path semantics callers expect (n_steps=100, method='euler',
    t_start=0.0, t_end=1.0).
    """

    def __init__(self, n_steps: int = 100, method: str = "euler"):
        self.n_steps = int(n_steps)
        self.method = str(method)
        self._cfg = _UpstreamSamplerConfig(
            n_steps=self.n_steps, method=self.method, t_start=0.0, t_end=1.0
        )

    def __call__(self, t: "torch.Tensor") -> "torch.Tensor":
        """Return alpha_t (= t) and sigma_t (= 1 - t) for OT path."""
        import torch

        return torch.stack([t, 1.0 - t], dim=-1)

    @property
    def config(self) -> _UpstreamSamplerConfig:
        return self._cfg


__all__ = ["CondOTScheduler"]
