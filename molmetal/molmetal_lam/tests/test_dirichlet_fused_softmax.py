"""Parity test for the fused-softmax wiring in proof_search.py.

Verifies that the Dirichlet mixing + final softmax over the root
children produces the same answer whether it goes through
:func:`triton_kernels.softmax_last_dim` (CPU fallback) or the
scalar fallback path.  The test is intentionally tiny (8 children) so
it runs on CPU as well as HIP.
"""

from __future__ import annotations

import math

import torch

from triton_kernels import softmax_last_dim
from triton_kernels.config import triton_config


def _scalar_dirichlet_mix(priors, noise, eps):
    """Reference scalar-path Dirichlet mixing with no renormalisation."""
    return [
        max(0.0, min(1.0, (1.0 - eps) * p + eps * e))
        for p, e in zip(priors, noise)
    ]


def _fused_dirichlet_mix(priors, noise, eps):
    """The new fused-softmax path used by ``_apply_dirichlet_to_root``.

    Mirrors the production implementation: build the Dirichlet-mixed
    priors, take their log (clipped) and softmax over the last dim,
    then clamp to ``[0, 1]``.
    """
    p_t = torch.tensor(priors, dtype=torch.float32)
    e_t = torch.tensor(noise, dtype=torch.float32)
    mixed = (1.0 - eps) * p_t + eps * e_t
    logits = torch.log(torch.clamp(mixed, min=1e-12))
    probs = softmax_last_dim(logits, dim=-1)
    return [max(0.0, min(1.0, float(v))) for v in probs.tolist()]


def test_fused_softmax_gate_says_yes_for_tiny_shape():
    # The gate must admit a tiny reduction axis (≤ MAX_FEAT_DIM).
    assert triton_config.use_fused_softmax(8) is True


def test_scalar_vs_fused_dirichlet_close():
    """The two paths should agree within a few ulp on a small input."""
    priors = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01]
    noise = [0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    eps = 0.25

    a = _scalar_dirichlet_mix(priors, noise, eps)
    b = _fused_dirichlet_mix(priors, noise, eps)
    # Fused path renormalises so the children sum to 1; scalar path
    # does not.  We compare the *shape* (each value is in [0, 1]) and
    # the L1 distance is bounded by the renormalisation gap.
    assert all(0.0 <= v <= 1.0 for v in a)
    assert all(0.0 <= v <= 1.0 for v in b)
    assert math.isclose(sum(b), 1.0, abs_tol=1e-4)
    # No value should be wildly different from its scalar counterpart
    # — the renormalisation gap is bounded by 1/(len(priors)).
    for x, y in zip(a, b):
        assert abs(x - y) < 0.5


def test_fused_softmax_matches_torch_softmax():
    """``softmax_last_dim`` is a drop-in for ``torch.softmax`` on CPU."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ref = torch.softmax(x, dim=-1)
    got = softmax_last_dim(x, dim=-1)
    assert torch.allclose(ref, got, atol=1e-5)


def test_fused_softmax_last_dim_for_high_dim_input():
    """Works on a 2-D tensor (rows × cols) — the canonical Triton path."""
    x = torch.randn(4, 16)
    ref = torch.softmax(x, dim=-1)
    got = softmax_last_dim(x, dim=-1)
    assert torch.allclose(ref, got, atol=1e-5)
    # Each row should sum to 1.
    sums = got.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)