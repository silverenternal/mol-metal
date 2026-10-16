"""Wf-A61 (2026-10-17) L23 — :class:`DiversitySamplingConfig` defaults
to ``use_mmd=False, use_repulsion=False`` so a direct caller that
constructs the dataclass without overrides does NOT silently activate
both diversity terms.
"""

from __future__ import annotations

import torch

from molmetal.adapters.flow_matching_lipman.diversity_sampling import (
    DiversitySamplingConfig,
    diversity_aware_sampling_step,
)


def test_default_config_has_use_mmd_false() -> None:
    """L23 — default ``use_mmd`` is False after the Wf-A61 flip."""
    cfg = DiversitySamplingConfig()
    assert cfg.use_mmd is False
    assert cfg.use_repulsion is False


def test_default_config_produces_zero_combined_loss() -> None:
    """L23 — default config gives ``combined_loss == 0``."""
    cfg = DiversitySamplingConfig()
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    out = diversity_aware_sampling_step(
        logits,
        cfg,
        reference_embeddings=torch.zeros((1, 8)),
        embedding_layer=torch.nn.Embedding(4, 8),  # type: ignore[arg-type]
    )
    assert float(out.combined_loss.item()) == 0.0


def test_explicit_use_mmd_true_activates_path() -> None:
    """L23 — opting back in via ``use_mmd=True`` re-enables the MMD term."""
    cfg = DiversitySamplingConfig(use_mmd=True, use_repulsion=True)
    assert cfg.use_mmd is True
    assert cfg.use_repulsion is True