import itertools

import pytest
import torch

from triton_kernels import fused_rmsnorm_residual


"""Coverage for the fused post residual RMSNorm operation.

The CPU cases validate numerical behavior and shape handling without requiring
ROCm hardware.  The GPU case exercises the differentiable path when a device
is available in the test environment.
"""


def torch_rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def test_parity_cpu():
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)
    weight = torch.randn(128)
    expected = torch_rmsnorm(x + residual) * weight
    actual = fused_rmsnorm_residual(x, residual, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_rejects_mismatched_residual():
    x = torch.randn(2, 8)
    residual = torch.randn(2, 7)
    weight = torch.ones(8)
    with pytest.raises(ValueError, match="same shape"):
        fused_rmsnorm_residual(x, residual, weight)


def test_rejects_bad_weight():
    x = torch.randn(2, 8)
    with pytest.raises(ValueError, match="weight must have shape"):
        fused_rmsnorm_residual(x, torch.randn_like(x), torch.ones(7))


@pytest.mark.parametrize("batch,dim", list(itertools.product([2, 8, 32, 128], [64, 128, 256, 512])))
@pytest.mark.parametrize("weight_kind", ["ones", "random"])
def test_random_shapes_cpu(batch, dim, weight_kind):
    torch.manual_seed(batch + dim)
    x = torch.randn(batch, dim)
    residual = torch.randn(batch, dim)
    weight = torch.ones(dim) if weight_kind == "ones" else torch.randn(dim)
    expected = torch_rmsnorm(x + residual) * weight
    actual = fused_rmsnorm_residual(x, residual, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="gradcheck needs CUDA/HIP")
def test_gradcheck_gpu():
    torch.manual_seed(1)
    x = torch.randn(2, 16, device="cuda", dtype=torch.float64, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(16, device="cuda", dtype=torch.float64, requires_grad=True)

    def fn(a, b, c):
        return fused_rmsnorm_residual(a, b, c)

    assert torch.autograd.gradcheck(
        fn,
        (x, residual, weight),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
