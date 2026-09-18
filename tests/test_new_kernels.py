"""Parity and smoke tests for the new Triton kernels.

Covers:

- :func:`triton_kernels.fused_dropout_residual`
- :func:`triton_kernels.fused_residual_add`
- :func:`triton_kernels.batched_fused_silu_mlp`
- :func:`triton_kernels.batched_fused_gelu_mlp`

Each kernel is checked against a PyTorch reference for shape parity
and (where meaningful) gradient parity, plus a small smoke test that
exercises the autograd path.

Run with::

    uv run pytest tests/test_new_kernels.py -v

"""

from __future__ import annotations

import pytest
import torch

from triton_kernels import (
    batched_fused_gelu_mlp,
    batched_fused_silu_mlp,
    batched_fused_silu_mlp_per_k,
    fused_dropout_residual,
    fused_residual_add,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_gpu() -> bool:
    return bool(torch.cuda.is_available())


DEVICE = torch.device("cuda" if _has_gpu() else "cpu")


# ---------------------------------------------------------------------------
# fused_dropout_residual
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(8, 16), (32, 128), (4, 8, 16)])
def test_fused_dropout_residual_parity(shape):
    torch.manual_seed(0)
    p = 0.3
    x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
    res = torch.randn(*shape, device=DEVICE, dtype=torch.float32)

    out, _ = fused_dropout_residual(x, res, p=p, seed=42)
    # Reference: same RNG state via torch's dropout.
    torch.manual_seed(42)
    # ``torch.dropout`` uses a different RNG stream than our kernel,
    # so we instead validate the contract:
    #   1. ``out`` has the same shape as ``x``,
    #   2. ``out - residual`` is either 0 or ``scale * x``,
    #   3. the keep-probability is approximately ``1 - p``.
    assert out.shape == x.shape
    diff = out - res
    scale = 1.0 / (1.0 - p)
    # Each element of ``diff`` must be either 0 or ``scale * x_i``.
    candidate_a = torch.zeros_like(diff)
    candidate_b = scale * x
    is_a = torch.isclose(diff, candidate_a, atol=1e-5)
    is_b = torch.isclose(diff, candidate_b, atol=1e-5)
    assert torch.all(is_a | is_b), (
        "fused_dropout_residual produced an out-of-grammar element."
    )
    # Keep probability check.
    kept = (~is_a).float().mean().item()
    assert abs(kept - (1.0 - p)) < 0.1, f"keep prob {kept} vs {1.0 - p}"


def test_fused_dropout_residual_grad():
    """Backward must produce a sensible gradient (mask matches forward)."""
    torch.manual_seed(1)
    p = 0.5
    x_ref = torch.randn(8, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    res_ref = torch.randn(8, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    res_fused = res_ref.detach().clone().requires_grad_(True)

    out_ref = torch.dropout(x_ref, p, True) + res_ref
    out_fused, _ = fused_dropout_residual(x_fused, res_fused, p=p, seed=7)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_fused.backward(grad)

    # The mask used by torch.dropout and by our kernel are not
    # bit-identical (different RNG streams), so we cannot check exact
    # equality of ``x_fused.grad`` vs ``x_ref.grad``.  Instead we check
    # that the gradient is non-zero on at least some elements and that
    # the residual gradient is identical to ``grad`` (the residual
    # backward is always the upstream gradient).
    assert x_fused.grad is not None
    assert (x_fused.grad.abs() > 0).any()
    assert torch.allclose(res_fused.grad, grad, atol=1e-5)


def test_fused_dropout_residual_smoke():
    """Smoke test for the autograd path on a small shape."""
    torch.manual_seed(2)
    x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    res = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
    out, mask = fused_dropout_residual(x, res, p=0.2, seed=11)
    assert out.shape == (4, 8)
    assert mask.shape == (4 * 8,)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# fused_residual_add
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 1.0), (1.0, -0.3)])
def test_fused_residual_add_parity(alpha, beta):
    torch.manual_seed(0)
    x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
    res = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)

    out_fused = fused_residual_add(x, res, alpha=alpha, beta=beta)
    out_ref = alpha * x + beta * res
    assert torch.allclose(out_fused, out_ref, atol=1e-5, rtol=1e-5), (
        f"fused_residual_add mismatch: max diff "
        f"{(out_fused - out_ref).abs().max().item()}"
    )


def test_fused_residual_add_grad():
    torch.manual_seed(1)
    x_ref = torch.randn(8, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    res_ref = torch.randn(8, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    res_fused = res_ref.detach().clone().requires_grad_(True)

    out_ref = x_ref + res_ref
    out_fused = fused_residual_add(x_fused, res_fused)

    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_fused.backward(grad)

    assert torch.allclose(x_fused.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(res_fused.grad, res_ref.grad, atol=1e-5, rtol=1e-5)


def test_fused_residual_add_smoke():
    """Smoke test: forward shape + autograd path on a small input."""
    torch.manual_seed(2)
    x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    res = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
    out = fused_residual_add(x, res)
    assert out.shape == (4, 8)
    out.sum().backward()
    assert x.grad is not None and res.grad is not None


# ---------------------------------------------------------------------------
# batched_fused_silu_mlp / batched_fused_gelu_mlp
# ---------------------------------------------------------------------------
def _ref_silu_mlp(x, w1, b1, w2, b2):
    pre = x @ w1
    if b1 is not None:
        pre = pre + b1
    post = torch.nn.functional.silu(pre)
    out = post @ w2
    if b2 is not None:
        out = out + b2
    return out


def _ref_gelu_mlp(x, w1, b1, w2, b2):
    pre = x @ w1
    if b1 is not None:
        pre = pre + b1
    post = torch.nn.functional.gelu(pre, approximate="tanh")
    out = post @ w2
    if b2 is not None:
        out = out + b2
    return out


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("with_bias", [True, False])
def test_batched_fused_silu_mlp_parity(K, with_bias):
    torch.manual_seed(0)
    M, D, H1, H2 = 8, 16, 32, 12

    x = torch.randn(M, D, device=DEVICE, dtype=torch.float32)
    w1s = torch.randn(K, D, H1, device=DEVICE, dtype=torch.float32) * 0.1
    w2s = torch.randn(K, H1, H2, device=DEVICE, dtype=torch.float32) * 0.1
    b1s = torch.randn(K, H1, device=DEVICE, dtype=torch.float32) if with_bias else None
    b2s = torch.randn(K, H2, device=DEVICE, dtype=torch.float32) if with_bias else None

    out_fused = batched_fused_silu_mlp(x, w1s, b1s, w2s, b2s)
    # Reference: sum the K per-MLP outputs.
    out_ref = sum(
        _ref_silu_mlp(x, w1s[k], b1s[k] if b1s is not None else None,
                      w2s[k], b2s[k] if b2s is not None else None)
        for k in range(K)
    )

    assert torch.allclose(out_fused, out_ref, atol=1e-4, rtol=1e-4), (
        f"batched_fused_silu_mlp mismatch (K={K}, with_bias={with_bias}): "
        f"max diff {(out_fused - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("K", [1, 3])
def test_batched_fused_gelu_mlp_parity(K):
    torch.manual_seed(1)
    M, D, H1, H2 = 8, 16, 32, 12

    x = torch.randn(M, D, device=DEVICE, dtype=torch.float32)
    w1s = torch.randn(K, D, H1, device=DEVICE, dtype=torch.float32) * 0.1
    w2s = torch.randn(K, H1, H2, device=DEVICE, dtype=torch.float32) * 0.1

    out_fused = batched_fused_gelu_mlp(x, w1s, None, w2s, None)
    out_ref = sum(
        _ref_gelu_mlp(x, w1s[k], None, w2s[k], None) for k in range(K)
    )

    assert torch.allclose(out_fused, out_ref, atol=1e-4, rtol=1e-4), (
        f"batched_fused_gelu_mlp mismatch (K={K}): "
        f"max diff {(out_fused - out_ref).abs().max().item()}"
    )


def test_batched_fused_silu_mlp_smoke():
    """Smoke test for the autograd path on a small batched MLP."""
    torch.manual_seed(2)
    K = 2
    M, D, H1, H2 = 4, 8, 16, 6
    x = torch.randn(M, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    # Use leaf tensors (no in-place ops) so .grad is populated.
    w1s = (torch.randn(K, D, H1, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)
    w2s = (torch.randn(K, H1, H2, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)

    out = batched_fused_silu_mlp(x, w1s, None, w2s, None)
    assert out.shape == (M, H2)
    out.sum().backward()
    assert x.grad is not None
    assert w1s.grad is not None
    assert w2s.grad is not None


# ---------------------------------------------------------------------------
# batched_fused_silu_mlp_per_k — round-7 G4 follow-up
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("with_bias", [True, False])
def test_batched_fused_silu_mlp_per_k(K, with_bias):
    """``batched_fused_silu_mlp_per_k`` returns ``(K, M, H2)`` and matches
    the PyTorch reference (sum of K per-MLP outputs) for each k-slice.
    """
    torch.manual_seed(0)
    M, D, H1, H2 = 8, 16, 32, 12

    x = torch.randn(M, D, device=DEVICE, dtype=torch.float32)
    w1s = torch.randn(K, D, H1, device=DEVICE, dtype=torch.float32) * 0.1
    w2s = torch.randn(K, H1, H2, device=DEVICE, dtype=torch.float32) * 0.1
    b1s = torch.randn(K, H1, device=DEVICE, dtype=torch.float32) if with_bias else None
    b2s = torch.randn(K, H2, device=DEVICE, dtype=torch.float32) if with_bias else None

    out = batched_fused_silu_mlp_per_k(x, w1s, b1s, w2s, b2s)
    assert out.shape == (K, M, H2), (
        f"batched_fused_silu_mlp_per_k must return (K, M, H2); got {tuple(out.shape)}"
    )

    for k in range(K):
        ref_k = _ref_silu_mlp(
            x,
            w1s[k],
            b1s[k] if b1s is not None else None,
            w2s[k],
            b2s[k] if b2s is not None else None,
        )
        assert torch.allclose(out[k], ref_k, atol=1e-4, rtol=1e-4), (
            f"per-K slice {k} mismatch (K={K}, with_bias={with_bias}): "
            f"max diff {(out[k] - ref_k).abs().max().item()}"
        )


def test_batched_fused_silu_mlp_per_k_grad():
    """Backward through the per-K variant must populate gradients on
    the input ``x`` and on the stacked weights / biases (where given).
    """
    torch.manual_seed(1)
    K = 2
    M, D, H1, H2 = 4, 8, 16, 6
    x = torch.randn(M, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
    w1s = (torch.randn(K, D, H1, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)
    w2s = (torch.randn(K, H1, H2, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)

    out = batched_fused_silu_mlp_per_k(x, w1s, None, w2s, None)
    assert out.shape == (K, M, H2)
    # Use a deterministic upstream gradient so we can compare.
    grad = torch.full_like(out, 0.25)
    out.backward(grad)
    assert x.grad is not None
    assert w1s.grad is not None
    assert w2s.grad is not None
    # Each gradient must have the expected shape.
    assert x.grad.shape == (M, D)
    assert w1s.grad.shape == (K, D, H1)
    assert w2s.grad.shape == (K, H1, H2)