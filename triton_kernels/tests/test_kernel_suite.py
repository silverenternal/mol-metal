"""Smoke and cross implementation parity checks for exported kernels."""

from __future__ import annotations

import pytest
import torch

import triton_kernels as tk


GPU_SKIP = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")

# CPU parity remains useful for operators that provide a documented fallback.
# Accelerator tests use the shared ``gpu`` marker and are skipped on hosts
# without ROCm, allowing the suite to run during documentation builds too.


@GPU_SKIP
def test_public_kernel_exports_are_callable() -> None:
    """All callable kernel exports remain importable from the package root."""
    metadata = {"AUTOTUNE_CONFIGS", "triton_config"}
    for name in tk.__all__:
        if name in metadata:
            continue
        assert callable(getattr(tk, name)), f"export {name!r} is not callable"


@pytest.mark.gpu
def test_fused_silu_mlp_matches_manual(small_shape: tuple[int, int], device: torch.device) -> None:
    b, d = small_shape
    h = 64
    x = torch.randn(b, d, device=device)
    # fused_silu_mlp uses transposed Linear weights: (input, hidden),
    # followed by (hidden, output).
    w1, w2 = torch.randn(d, h, device=device), torch.randn(h, d, device=device)
    b1, b2 = torch.randn(h, device=device), torch.randn(d, device=device)
    got = tk.fused_silu_mlp(x, w1, b1, w2, b2)
    ref = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Linear(h, d)).to(device)
    with torch.no_grad():
        ref[0].weight.copy_(w1.T); ref[0].bias.copy_(b1)
        ref[2].weight.copy_(w2.T); ref[2].bias.copy_(b2)
    # The fused operator applies SiLU after both projections.
    assert torch.allclose(got, ref(x), atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_fused_rms_norm_matches_torch(small_shape: tuple[int, int], device: torch.device) -> None:
    b, d = small_shape
    x, weight = torch.randn(b, d, device=device), torch.randn(d, device=device)
    got = tk.fused_rms_norm(x, weight, eps=1e-5)
    ref = torch.nn.RMSNorm(d, eps=1e-5, device=device)
    with torch.no_grad(): ref.weight.copy_(weight)
    assert torch.allclose(got, ref(x), atol=1e-4, rtol=1e-4)


def test_fused_softmax_matches_torch(small_shape: tuple[int, int]) -> None:
    x = torch.randn(*small_shape)
    assert torch.allclose(tk.softmax_last_dim(x), torch.softmax(x, dim=-1))


@pytest.mark.gpu
def test_matmul_matches_torch(device: torch.device) -> None:
    a, b = torch.randn(64, 64, device=device), torch.randn(64, 64, device=device)
    assert torch.allclose(tk.matmul_fn(a, b), torch.matmul(a, b), atol=1e-5, rtol=1e-5)
