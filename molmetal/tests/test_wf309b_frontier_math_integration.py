"""Wf-309B — Frontier Math Integration wrapper smoke tests.

================================================================
4 smoke tests for the 4 STANDALONE frontier-math adapters composed
via :mod:`molmetal.adapters.frontier_math_integration`:

* ``test_multi_ph_forward`` — :class:`MultiPHPocketFeatures` builds
  + one dummy forward pass on a synthetic ``(B, P, 3)`` pocket cloud.
* ``test_se3_lipman_solver_init`` — :class:`SE3LipmanSolver` builds
  with the canonical W2.1 defaults (n_steps=50, method="midpoint",
  ortho_tol=1e-3).
* ``test_se3_sb_decoder_init`` — :class:`SE3SBDecoder` builds with
  the canonical W2.1 defaults (eps=0.1, n_iters=50,
  use_sinkhorn=True).
* ``test_transport_relaxation_init`` — :class:`RelaxedEndpointTerm`
  builds with the canonical W2.2 defaults (lambda_relax=1e-3,
  eps=0.1) and produces a non-negative scalar.

Honest framing
--------------
* No modification of :mod:`molmetal.adapters.flow_matching_lipman`.
* Each test exercises ONLY the factory surface from
  :class:`FrontierMathIntegration` — no direct adapter import.
* All 4 tests are CPU-only and run in <1 s on a single thread.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fmi():
    """Return a :class:`FrontierMathIntegration` with default config."""
    from molmetal.adapters.frontier_math_integration import FrontierMathIntegration
    return FrontierMathIntegration()


# ---------------------------------------------------------------------------
# 1. Multi-PH pocket features — factory + dummy forward pass
# ---------------------------------------------------------------------------
def test_multi_ph_forward(fmi):
    """``get_multi_ph_features`` builds a usable module + 1 forward pass."""
    from molmetal.adapters.frontier_math_integration import (
        FrontierMathIntegration,
    )
    assert isinstance(fmi, FrontierMathIntegration)
    features_mod = fmi.get_multi_ph_features()
    # Use the smallest W2.3 smoke config to keep the test <100 ms.
    features_mod = fmi.get_multi_ph_features(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    # Synthetic pocket cloud: (B=2, P=8, 3).
    torch.manual_seed(0)
    pocket = torch.randn(2, 8, 3)
    out = features_mod.compute(pocket)
    # Output shape = (B, n_bins, n_bins) = (2, 4, 4).
    assert out.shape == (2, 4, 4), f"unexpected feature map shape: {tuple(out.shape)}"
    # The feature map is a count of (r, d) pairs per pocket — must
    # be non-negative integers (cast to long for the assertion).
    assert (out >= 0).all(), "multi-PH feature map must be non-negative"


# ---------------------------------------------------------------------------
# 2. SE(3) Lipman solver — factory init
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_init(fmi):
    """``get_se3_lipman_solver`` returns a usable SE3LipmanSolver."""
    solver = fmi.get_se3_lipman_solver()
    # Canonical W2.1 / Integration 1 defaults.
    assert solver.n_steps == 50, f"n_steps should default to 50, got {solver.n_steps}"
    assert solver.method == "midpoint", f"method should default to 'midpoint', got {solver.method!r}"
    assert solver.ortho_tol == pytest.approx(1e-3), (
        f"ortho_tol should default to 1e-3, got {solver.ortho_tol}"
    )
    # ``se3_adapter`` is the wrapper, must be present (the default
    # fallback is a zero-velocity wrapper around a dummy EGNN).
    assert solver.se3_adapter is not None


# ---------------------------------------------------------------------------
# 3. SE(3) SB decoder — factory init
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_init(fmi):
    """``get_se3_sb_decoder`` returns a usable SE3SBDecoder."""
    decoder = fmi.get_se3_sb_decoder()
    # Canonical W2.1 defaults.
    assert decoder.eps == pytest.approx(0.1), (
        f"eps should default to 0.1, got {decoder.eps}"
    )
    assert decoder.n_iters == 50, f"n_iters should default to 50, got {decoder.n_iters}"
    assert decoder.use_sinkhorn is True, (
        f"use_sinkhorn should default to True, got {decoder.use_sinkhorn}"
    )


# ---------------------------------------------------------------------------
# 4. Transport relaxation — factory init + zero-lambda sanity
# ---------------------------------------------------------------------------
def test_transport_relaxation_init(fmi):
    """``get_transport_relaxation`` returns a usable RelaxedEndpointTerm."""
    term = fmi.get_transport_relaxation()
    # Canonical W2.2 defaults.
    assert term.lambda_relax == pytest.approx(1e-3), (
        f"lambda_relax should default to 1e-3, got {term.lambda_relax}"
    )
    assert term.eps == pytest.approx(0.1), f"eps should default to 0.1, got {term.eps}"
    # Forward on a synthetic (B=4, D=3) endpoint + equal energies
    # should produce exactly zero KL (Gibbs-minimum case).
    trajectory = torch.randn(4, 3)
    energies = torch.zeros(4)            # all-equal → q_T is uniform
    penalty = term.forward(trajectory, energies)
    assert penalty.dim() == 0, f"penalty must be scalar, got shape {tuple(penalty.shape)}"
    assert float(penalty.item()) == pytest.approx(0.0, abs=1e-6), (
        f"penalty on uniform q_T should be 0, got {float(penalty.item())}"
    )
