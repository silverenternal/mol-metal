import pytest
import torch
from triton_kernels import FusedGeluMLP


@pytest.mark.parametrize("shape", [(4, 64), (32, 128, 512), (8, 256, 1024)])
def test_shapes_and_dtype(shape):
    m = FusedGeluMLP(shape[-1], max(16, shape[-1] // 2), device="cpu")
    x = torch.randn(*shape, dtype=torch.float16)
    y = m(x)
    assert y.shape == shape[:-1] + (shape[-1],)
    assert y.dtype == x.dtype


def test_cpu_fallback_parity():
    torch.manual_seed(0)
    m = FusedGeluMLP(64, 128, device="cpu")
    x = torch.randn(4, 64, dtype=torch.float16)
    expected = torch.nn.functional.gelu(torch.nn.functional.linear(x, m.w1, m.b1)).matmul(m.w2.t()).add(m.b2)
    torch.testing.assert_close(m(x), expected, atol=1.5e-3, rtol=1.5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")
def test_gpu_parity():
    torch.manual_seed(0)
    m = FusedGeluMLP(64, 128, device="cuda")
    x = torch.randn(64, 128, 64, device="cuda", dtype=torch.float16)
    ref = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)).cuda().to(torch.float16)
    ref[0].weight.data.copy_(m.w1); ref[0].bias.data.copy_(m.b1)
    ref[2].weight.data.copy_(m.w2); ref[2].bias.data.copy_(m.b2)
    torch.testing.assert_close(m(x), ref(x), atol=1.5e-3, rtol=1.5e-3)
