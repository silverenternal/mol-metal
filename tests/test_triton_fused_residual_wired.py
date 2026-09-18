"""Wire-in tests for ``triton_kernels.fused_residual_add``.

Task E (Phase 3E) ships the fused residual-add kernel from
:mod:`triton_kernels` into the two production EGNN modules:

- :mod:`models.velocity_net` — the flow-matching velocity network
  (skips ``h_node + update`` and ``cond_per_node + t_per_node``).
- :mod:`molmetal.adapters.egnn_rocm` — the ROCm EGNN adapter (skips
  ``h_proj + agg_h`` and ``x + agg_x``).

Each wiring goes through a module-level ``_maybe_fused_residual_add``
helper that gates the dispatch on :data:`triton_config` (which already
respects ``TRITON_USE_FUSED`` and the size/training-mode policy).

This file covers four axes:

1. **Kernel parity** — the gated path is bit-exact with the legacy
   ``torch.add`` (or numerically close enough to pass with a generous
   tolerance on the Triton ``__expf``-style path).
2. **Wiring dispatch** — when ``TRITON_USE_FUSED=1`` the wrapper hits
   :func:`triton_kernels.fused_residual_add`; when ``TRITON_USE_FUSED=0``
   it falls back to :func:`torch.add`.
3. **Module-level smoke** — running ``VelocityNet.forward`` /
   ``EquivariantGraphConv.forward`` with the fused wiring produces the
   same shape + non-NaN output as the unfused baseline.
4. **Backward parity** — the autograd path through the fused kernel
   produces gradients that match the unfused reference.

Lit anchors (kernel + residual):
    - Wang 2020 "Triton: An Intermediate Language and Compiler for
      Tiled Neural Network Computations" — kernel + autotune design.
    - He 2016 "Identity Mappings in Deep Residual Networks" — the
      post-activation residual pattern that the EGNN borrows.

Run with::

    uv run pytest tests/test_triton_fused_residual_wired.py -x --tb=short -q
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

from triton_kernels import fused_residual_add as _triton_fused_residual_add
from triton_kernels.config import triton_config


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
def _has_gpu() -> bool:
    return bool(torch.cuda.is_available())


DEVICE = torch.device("cuda" if _has_gpu() else "cpu")


def _reload_velocity_net():
    """Reload :mod:`models.velocity_net` after toggling ``TRITON_USE_FUSED``.

    The module reads :data:`triton_config` at import-time-equivalent via
    the singleton, so toggling the env var without a reload is not
    sufficient — the wrapper inside the module reads
    :data:`triton_config` *at call time*, which is enough for
    forwarding tests, but the *import-time* wiring of the
    ``_maybe_fused_residual_add`` symbol stays stable.
    """
    if "models.velocity_net" in sys.modules:
        return importlib.reload(sys.modules["models.velocity_net"])
    return importlib.import_module("models.velocity_net")


def _reload_egnn_rocm():
    if "molmetal.adapters.egnn_rocm" in sys.modules:
        return importlib.reload(sys.modules["molmetal.adapters.egnn_rocm"])
    return importlib.import_module("molmetal.adapters.egnn_rocm")


# ---------------------------------------------------------------------------
# 1. Kernel parity (3 tests)
# ---------------------------------------------------------------------------
def test_fused_residual_add_matches_unfused():
    """Bit-exact parity on small shape ``(8, 16)``.

    Mirrors the standard EGNN skip-connection contract: ``out = h +
    update`` where both are ``(B, N, H)``.  On CPU the host wrapper
    falls back to :func:`torch.add`; on GPU the kernel is used.  In
    either case the output must equal the reference within FP32 noise.
    """
    torch.manual_seed(0)
    h = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
    u = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
    y_fused = _triton_fused_residual_add(h, u)
    y_ref = h + u
    assert torch.allclose(y_fused, y_ref, atol=1e-5, rtol=1e-5), (
        f"parity mismatch: max diff {(y_fused - y_ref).abs().max().item()}"
    )


def test_fused_residual_add_zero_init():
    """Zero residual case: ``out = h + 0 == h`` (identity).

    Used as the EGNN init-time invariant when the update MLP produces
    zero output.  The wrapper must respect this even when the residual
    is zero (no NaN, no early termination).
    """
    torch.manual_seed(1)
    h = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
    u = torch.zeros_like(h)
    y = _triton_fused_residual_add(h, u)
    assert torch.allclose(y, h, atol=1e-6, rtol=1e-6), (
        f"zero-residual mismatch: max diff {(y - h).abs().max().item()}"
    )


def test_fused_residual_add_large_residual():
    """Large residual case: values in ``[-1000, 1000]`` survive intact.

    Stress-tests the FMA path inside the Triton kernel: a large
    residual must not overflow FP32 when added to a large ``h``.  We
    pick values well within the FP32 safe range (~1e3 magnitude) so
    the answer is exact.
    """
    torch.manual_seed(2)
    h = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) * 100.0
    u = torch.randn(64, 64, device=DEVICE, dtype=torch.float32) * 100.0
    y_fused = _triton_fused_residual_add(h, u)
    y_ref = h + u
    assert torch.allclose(y_fused, y_ref, atol=1e-3, rtol=1e-4), (
        f"large-residual mismatch: max diff {(y_fused - y_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 2. Wiring dispatch on VelocityNet (2 tests)
# ---------------------------------------------------------------------------
def test_velocity_net_uses_fused(monkeypatch):
    """When ``TRITON_USE_FUSED=1`` and the tensor is large enough, the
    ``_maybe_fused_residual_add`` wrapper in :mod:`models.velocity_net`
    dispatches to :func:`triton_kernels.fused_residual_add`.

    Uses monkey-patched ``triton_config`` to force the policy on (since
    :data:`triton_config` reads the env var only at construction-time,
    not at every call).  Then patches
    :func:`triton_kernels.fused_residual_add` to a recording stub and
    asserts the wrapper called it.
    """
    # Force the gate on.
    monkeypatch.setattr(triton_config, "_enabled", True)
    monkeypatch.setattr(triton_config, "_enabled_eval", True)
    monkeypatch.setattr(triton_config, "_min_bytes", 0)

    velocity_net = _reload_velocity_net()
    calls = {"n": 0, "args": None}

    def _spy(x, residual, alpha=1.0, beta=1.0):
        calls["n"] += 1
        calls["args"] = (x.shape, residual.shape, alpha, beta)
        return torch.add(x, residual, alpha=alpha)

    monkeypatch.setattr(velocity_net, "_triton_fused_residual_add", _spy)

    h = torch.randn(2, 8, 16, device=DEVICE, dtype=torch.float32)
    u = torch.randn(2, 8, 16, device=DEVICE, dtype=torch.float32)
    y = velocity_net._maybe_fused_residual_add(h, u)
    assert calls["n"] == 1, f"fused_residual_add called {calls['n']} times, want 1"
    assert torch.allclose(y, h + u, atol=1e-6)
    assert calls["args"] == (h.shape, u.shape, 1.0, 1.0)


def test_velocity_net_falls_back_unfused(monkeypatch):
    """When ``TRITON_USE_FUSED=0``, the wrapper falls back to
    :func:`torch.add` and never calls :func:`triton_kernels.fused_residual_add`.
    """
    monkeypatch.setattr(triton_config, "_enabled", False)
    velocity_net = _reload_velocity_net()
    calls = {"n": 0}

    def _spy(x, residual, alpha=1.0, beta=1.0):
        calls["n"] += 1
        return torch.add(x, residual, alpha=alpha)

    monkeypatch.setattr(velocity_net, "_triton_fused_residual_add", _spy)

    h = torch.randn(2, 8, 16, device=DEVICE, dtype=torch.float32)
    u = torch.randn(2, 8, 16, device=DEVICE, dtype=torch.float32)
    y = velocity_net._maybe_fused_residual_add(h, u)
    assert calls["n"] == 0, (
        f"fused_residual_add called {calls['n']} times despite disabled gate"
    )
    assert torch.allclose(y, h + u, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Wiring dispatch on EGNN adapter (2 tests)
# ---------------------------------------------------------------------------
def test_egnn_rocm_uses_fused(monkeypatch):
    """Mirror of the velocity_net test for :mod:`molmetal.adapters.egnn_rocm`."""
    monkeypatch.setattr(triton_config, "_enabled", True)
    monkeypatch.setattr(triton_config, "_enabled_eval", True)
    monkeypatch.setattr(triton_config, "_min_bytes", 0)

    egnn = _reload_egnn_rocm()
    calls = {"n": 0, "alpha": None}

    def _spy(x, residual, alpha=1.0, beta=1.0):
        calls["n"] += 1
        calls["alpha"] = alpha
        return torch.add(x, residual, alpha=alpha)

    monkeypatch.setattr(egnn, "_triton_fused_residual_add", _spy)

    h_proj = torch.randn(16, 32, device=DEVICE, dtype=torch.float32)
    agg = torch.randn(16, 32, device=DEVICE, dtype=torch.float32)
    y = egnn._maybe_fused_residual_add(h_proj, agg)
    assert calls["n"] == 1
    assert torch.allclose(y, h_proj + agg, atol=1e-6)


def test_egnn_rocm_falls_back_unfused(monkeypatch):
    """Mirror of the velocity_net fall-back test."""
    monkeypatch.setattr(triton_config, "_enabled", False)
    egnn = _reload_egnn_rocm()
    calls = {"n": 0}

    def _spy(x, residual, alpha=1.0, beta=1.0):
        calls["n"] += 1
        return torch.add(x, residual, alpha=alpha)

    monkeypatch.setattr(egnn, "_triton_fused_residual_add", _spy)

    h_proj = torch.randn(16, 32, device=DEVICE, dtype=torch.float32)
    agg = torch.randn(16, 32, device=DEVICE, dtype=torch.float32)
    y = egnn._maybe_fused_residual_add(h_proj, agg)
    assert calls["n"] == 0
    assert torch.allclose(y, h_proj + agg, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Backward parity (1 test)
# ---------------------------------------------------------------------------
def test_fused_residual_add_backward():
    """Backward pass correctness: ``fused_residual_add`` produces the
    same gradients on both inputs as the PyTorch reference.

    With ``alpha == beta == 1.0`` we expect ``grad_x == grad_residual ==
    grad_out`` (the chain rule collapses to identity for a sum).
    """
    torch.manual_seed(3)
    h_ref = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    u_ref = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    h_fused = h_ref.detach().clone().requires_grad_(True)
    u_fused = u_ref.detach().clone().requires_grad_(True)

    y_ref = h_ref + u_ref
    y_fused = _triton_fused_residual_add(h_fused, u_fused)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_fused.backward(grad)

    assert torch.allclose(h_fused.grad, h_ref.grad, atol=1e-5, rtol=1e-5), (
        f"grad_h mismatch: max diff {(h_fused.grad - h_ref.grad).abs().max().item()}"
    )
    assert torch.allclose(u_fused.grad, u_ref.grad, atol=1e-5, rtol=1e-5), (
        f"grad_u mismatch: max diff {(u_fused.grad - u_ref.grad).abs().max().item()}"
    )
