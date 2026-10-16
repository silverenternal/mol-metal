"""Wf-A61 (2026-10-17) L20+L21 — ``nucleus_sample`` rejects non-positive
temperature so a misconfigured caller fails fast instead of producing
inverted distributions (negative temperature) or silent argmax
(temperature == 0).
"""

from __future__ import annotations

import pytest
import torch

from molmetal.adapters.flow_matching_lipman.diversity_sampling import (
    nucleus_sample,
)


def test_nucleus_sample_zero_temperature_raises() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        nucleus_sample(logits, p=0.9, temperature=0.0)


def test_nucleus_sample_negative_temperature_raises() -> None:
    """L21 — negative temperature would invert the distribution silently."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        nucleus_sample(logits, p=0.9, temperature=-0.5)


def test_nucleus_sample_small_positive_temperature_runs() -> None:
    """L20 — temperature=1e-3 (near-greedy) runs and returns valid tokens."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    sampled = nucleus_sample(logits, p=0.9, temperature=1e-3)
    assert sampled.shape == (1,)
    assert int(sampled.item()) in {0, 1, 2, 3}


def test_nucleus_sample_top_p_zero_returns_argmax() -> None:
    """L20 doc — top_p=0 is the documented argmax path."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    sampled = nucleus_sample(logits, p=0.0, temperature=1.0)
    assert int(sampled.item()) == 2  # argmax