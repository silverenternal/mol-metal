"""Parity and gradient tests for the fused softmax / cross-entropy kernels.

These tests cover the two new Triton kernels in :mod:`triton_kernels`:

- :func:`triton_kernels.softmax_last_dim`
- :func:`triton_kernels.fused_cross_entropy`

Each kernel is checked against the corresponding PyTorch reference on
both CPU and (when available) the AMD ROCm device.  When no GPU is
present, the kernels transparently fall back to the PyTorch
implementation, so these tests act as a regression guard for the
dispatch path as well as for the kernel itself.

Run with::

    uv run pytest tests/test_fused_ops.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from triton_kernels import fused_cross_entropy, softmax_last_dim


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _has_gpu() -> bool:
    return bool(torch.cuda.is_available())


# Skip with a clear message rather than error if no GPU is present.
# The kernels fall back to the PyTorch reference on CPU, so the parity
# tests will still pass; we use the device to decide whether to verify
# the kernel itself or only the fallback path.
DEVICE = torch.device("cuda" if _has_gpu() else "cpu")


# ---------------------------------------------------------------------------
# Fused softmax tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(16, 32), (128, 64), (1823, 781), (8, 4, 64)])
def test_fused_softmax_parity(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
    y_fused = softmax_last_dim(x, dim=-1)
    y_ref = torch.softmax(x, dim=-1)
    # The fused kernel uses tl.exp (approx __expf on CUDA) so a
    # generous tolerance is required for parity with the exact
    # torch.softmax reference.
    assert torch.allclose(y_fused, y_ref, atol=1e-5, rtol=1e-5), (
        f"softmax mismatch for shape {shape}: max diff "
        f"{(y_fused - y_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("shape", [(16, 32), (128, 64), (8, 4, 64)])
def test_fused_softmax_grad(shape):
    torch.manual_seed(1)
    # Forward + backward check: ``softmax_last_dim`` must produce
    # the same gradient as the PyTorch reference within FP32 noise.
    x_ref = torch.randn(*shape, device=DEVICE, dtype=torch.float32, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    y_ref = torch.softmax(x_ref, dim=-1)
    y_fused = softmax_last_dim(x_fused, dim=-1)
    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_fused.backward(grad)
    assert torch.allclose(x_fused.grad, x_ref.grad, atol=1e-5, rtol=1e-5), (
        f"softmax grad mismatch for shape {shape}: max diff "
        f"{(x_fused.grad - x_ref.grad).abs().max().item()}"
    )


def test_fused_softmax_handles_non_last_dim():
    """``softmax_last_dim`` permutes non-trailing dims internally."""
    torch.manual_seed(2)
    x = torch.randn(4, 16, 8, device=DEVICE, dtype=torch.float32)
    y_fused = softmax_last_dim(x, dim=1)
    y_ref = torch.softmax(x, dim=1)
    assert torch.allclose(y_fused, y_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Fused cross-entropy tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch,n_classes",
    [(4, 8), (32, 64), (128, 256), (16, 1024)],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_fused_ce_parity(batch, n_classes, reduction):
    torch.manual_seed(batch + n_classes)
    logits = torch.randn(batch, n_classes, device=DEVICE, dtype=torch.float32)
    targets = torch.randint(0, n_classes, (batch,), device=DEVICE, dtype=torch.int64)
    out_fused = fused_cross_entropy(logits, targets, ignore_index=-100, reduction=reduction)
    out_ref = F.cross_entropy(logits, targets, ignore_index=-100, reduction=reduction)
    # Cross-entropy uses ``tl.exp`` / ``tl.log`` (approx); allow 1e-4.
    assert torch.allclose(out_fused, out_ref, atol=1e-4, rtol=1e-4), (
        f"CE mismatch for shape ({batch}, {n_classes}) reduction={reduction}: "
        f"max diff {(out_fused - out_ref).abs().max().item()}"
    )


def test_fused_ce_ignore_index():
    """``ignore_index`` should mask the loss contribution."""
    torch.manual_seed(3)
    batch, n_classes = 8, 16
    logits = torch.randn(batch, n_classes, device=DEVICE, dtype=torch.float32)
    targets = torch.randint(0, n_classes, (batch,), device=DEVICE, dtype=torch.int64)
    targets[0] = -100  # masked row
    out_fused = fused_cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    out_ref = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    assert torch.allclose(out_fused, out_ref, atol=1e-4, rtol=1e-4)
    # Masked row's loss is exactly zero.
    assert out_fused[0].item() == 0.0


def test_fused_ce_backward_mean():
    torch.manual_seed(4)
    batch, n_classes = 16, 32
    logits_ref = torch.randn(batch, n_classes, device=DEVICE, dtype=torch.float32, requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, n_classes, (batch,), device=DEVICE, dtype=torch.int64)
    F.cross_entropy(logits_ref, targets, reduction="mean").backward()
    fused_cross_entropy(logits_fused, targets, reduction="mean").backward()
    assert torch.allclose(logits_fused.grad, logits_ref.grad, atol=1e-4, rtol=1e-4), (
        f"CE mean backward mismatch: max diff "
        f"{(logits_fused.grad - logits_ref.grad).abs().max().item()}"
    )


def test_fused_ce_backward_with_ignore_index():
    torch.manual_seed(5)
    batch, n_classes = 16, 32
    logits_ref = torch.randn(batch, n_classes, device=DEVICE, dtype=torch.float32, requires_grad=True)
    logits_fused = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, n_classes, (batch,), device=DEVICE, dtype=torch.int64)
    targets[3] = -100
    F.cross_entropy(logits_ref, targets, ignore_index=-100, reduction="mean").backward()
    fused_cross_entropy(logits_fused, targets, ignore_index=-100, reduction="mean").backward()
    assert torch.allclose(logits_fused.grad, logits_ref.grad, atol=1e-4, rtol=1e-4), (
        f"CE ignore_index backward mismatch: max diff "
        f"{(logits_fused.grad - logits_ref.grad).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Sanity: report which device we are running on.  Useful when reading
# CI logs to know whether the kernels were actually executed.
# ---------------------------------------------------------------------------
def test_device_report(capsys):
    with capsys.disabled():
        print(f"\n[fused_ops tests] device = {DEVICE}, has_gpu = {_has_gpu()}")
    assert True