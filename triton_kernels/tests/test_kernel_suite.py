"""Smoke and cross implementation parity checks for exported kernels."""

from __future__ import annotations

import pytest
import torch

import triton_kernels as tk
from triton_kernels import backend


GPU_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")

# CPU parity remains useful for operators that provide a documented fallback.
# Accelerator tests use the shared ``gpu`` marker and are skipped on hosts
# without a CUDA device, allowing the suite to run during documentation
# builds too.


@pytest.fixture
def device() -> torch.device:
    """Device for the accelerator parity tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_shape() -> tuple[int, int]:
    """``(rows, feat_dim)`` small enough to keep the suite fast."""
    return (6, 8)


def _normalized_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """``max|got - ref|`` as a fraction of the reference's own magnitude.

    The absolute error of a GEMM grows with the size of its output, so
    a fixed ``atol`` either fails on large activations or passes
    anything on small ones.  Dividing by ``max|ref|`` makes the bound
    a property of the arithmetic rather than of the random draw.
    """
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def _dot_tolerance() -> float:
    """Accepted normalized error for a ``tl.dot`` kernel.

    ``tf32`` carries a 10-bit mantissa, so a K=64 reduction lands
    within about ``2**-10`` of the reference magnitude.  Measured on
    sm_120 (2026-10-23): 7.2e-4 for a 64x64x64 matmul, 8.1e-4 for the
    fused SiLU MLP.  ``ieee`` accumulates in full FP32 and holds ~1e-6.
    ``TRITON_DOT_PRECISION`` selects between them at import time.
    """
    return 1e-3 if backend.dot_precision() == "tf32" else 1e-5


@GPU_SKIP
def test_public_kernel_exports_are_callable() -> None:
    """All callable kernel exports remain importable from the package root."""
    metadata = {
        "AUTOTUNE_CONFIGS",
        "AUTOTUNE_CONFIGS_CUDA",
        "AUTOTUNE_CONFIGS_FAST",
        "AUTOTUNE_CONFIGS_ROCM",
        "backend",
        "triton_config",
    }
    for name in tk.__all__:
        if name in metadata:
            continue
        assert callable(getattr(tk, name)), f"export {name!r} is not callable"


@pytest.mark.gpu
def test_fused_silu_mlp_matches_manual(
    small_shape: tuple[int, int], device: torch.device
) -> None:
    b, d = small_shape
    h = 64
    x = torch.randn(b, d, device=device)
    # fused_silu_mlp uses transposed Linear weights: (input, hidden),
    # followed by (hidden, output).
    w1, w2 = torch.randn(d, h, device=device), torch.randn(h, d, device=device)
    b1, b2 = torch.randn(h, device=device), torch.randn(d, device=device)
    got = tk.fused_silu_mlp(x, w1, b1, w2, b2)
    ref = torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Linear(h, d)
    ).to(device)
    with torch.no_grad():
        ref[0].weight.copy_(w1.T)
        ref[0].bias.copy_(b1)
        ref[2].weight.copy_(w2.T)
        ref[2].bias.copy_(b2)
    # The fused operator applies SiLU between the two projections.
    err = _normalized_error(got, ref(x))
    assert err < _dot_tolerance(), f"normalized error {err:.2e} exceeds {_dot_tolerance():.0e}"


@pytest.mark.gpu
def test_fused_rms_norm_matches_torch(
    small_shape: tuple[int, int], device: torch.device
) -> None:
    b, d = small_shape
    x, weight = torch.randn(b, d, device=device), torch.randn(d, device=device)
    got = tk.fused_rms_norm(x, weight, eps=1e-5)
    ref = torch.nn.RMSNorm(d, eps=1e-5, device=device)
    with torch.no_grad():
        ref.weight.copy_(weight)
    assert torch.allclose(got, ref(x), atol=1e-4, rtol=1e-4)


def test_fused_softmax_matches_torch(small_shape: tuple[int, int]) -> None:
    x = torch.randn(*small_shape)
    assert torch.allclose(tk.softmax_last_dim(x), torch.softmax(x, dim=-1))


@pytest.mark.gpu
def test_matmul_matches_torch(device: torch.device) -> None:
    a, b = torch.randn(64, 64, device=device), torch.randn(64, 64, device=device)
    got, ref = tk.matmul_fn(a, b), torch.matmul(a, b)
    err = _normalized_error(got, ref)
    assert err < _dot_tolerance(), f"normalized error {err:.2e} exceeds {_dot_tolerance():.0e}"
