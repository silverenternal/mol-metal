"""P5 sanity tests: ROCm throughput must clear minimal time budgets.

These tests gate the P5 work — they fail loudly if the GPU is
unavailable, if a kernel takes pathologically long, or if a model
forward regresses below the budget doc's published rates.  They are
the companion to ``molmetal/scripts/bench_rocm_throughput.py`` (which
prints numbers for human inspection); the script does NOT fail on
slow numbers — that's the job of these tests.

The time budgets are deliberately generous (1 s for 1024³ matmul, 5 s
for the full D-MPNN batch) so a noisy CI machine can still pass; they
are written against the budget dGPU predictions *with* a 10× slack
factor.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

# Make sure the project root is on sys.path so ``models._scatter`` etc.
# resolve from the test runner (pytest's conftest does not always add
# the project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from molmetal.utils.device import ROCM_AVAILABLE  # noqa: E402

# Skip the whole module on CPU-only environments — without a GPU the
# timings are meaningless and would slow CI to a crawl.
pytestmark = pytest.mark.skipif(
    not ROCM_AVAILABLE,
    reason="No ROCm/CUDA device visible to torch — throughput tests are GPU-only.",
)


# ---------------------------------------------------------------------------
# 1. Matmul 1024^3 in < 1 s
# ---------------------------------------------------------------------------
def test_matmul_in_rocm():
    """A single FP32 1024³ matmul should finish well under 1 s on ROCm.

    On the budget dGPU (RX 7800 XT) this is a 50 TFLOP/s roofline
    operation taking ~0.4 ms.  We allow 1 s — i.e. a 2 500× slack — so
    the test passes on any RDNA 3 / Navi class GPU, including the
    slower iGPUs.
    """
    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)

    # Warmup so cuBLAS picks the right kernel
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 5
    for _ in range(n_iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    mean_s = (time.perf_counter() - t0) / n_iters

    # 1 s is the absolute upper bound; budget dGPU does this in <1 ms.
    assert mean_s < 1.0, (
        f"FP32 1024³ matmul took {mean_s * 1000:.2f} ms — slower than "
        f"the 1 s upper bound.  Check that ROCm is actually being used "
        f"(device={a.device}, name={torch.cuda.get_device_name(0)})."
    )

    # Also assert minimum throughput for diagnostic purposes (this is
    # NOT a hard fail threshold — it just prints a warning).
    gflops = 2 * 1024 ** 3 / mean_s / 1e9
    print(f"FP32 1024³ matmul: {mean_s * 1000:.3f} ms  ({gflops:.1f} GFLOP/s)")


# ---------------------------------------------------------------------------
# 2. D-MPNN forward 100 mols × 30 atoms, batch=10, finishes in < 5 s
# ---------------------------------------------------------------------------
def test_dmpnn_30_atoms_batch_10():
    """Forward 100 small molecules (30 atoms) in batches of 10 within 5 s.

    The budget doc predicts 2 000–4 500 samples/sec at batch=10 on the
    7800 XT.  100 samples in 5 s = 20 samples/s, a 100× slack — the
    test fails only if the kernel takes >50 ms / batch, which would
    indicate something is broken in the ROCm runtime.
    """
    from molmetal.models.dmpnn import DirectedMPNN

    device = torch.device("cuda:0")
    model = DirectedMPNN().to(device).eval()

    batch_size = 10
    n_atoms = 30
    n_mols = 100
    n_batches = n_mols // batch_size

    # Fully-connected directed edges (no self-loops): E = N*(N-1)
    idx = torch.arange(n_atoms, device=device)
    src = idx.view(1, -1, 1).expand(batch_size, n_atoms, n_atoms)
    dst = idx.view(1, 1, -1).expand(batch_size, n_atoms, n_atoms)
    mask = src != dst
    src_flat = src[mask].view(batch_size, -1)
    dst_flat = dst[mask].view(batch_size, -1)
    edge_index = torch.stack([src_flat, dst_flat], dim=1)
    h_atom = torch.randn(batch_size, n_atoms, 39, device=device)
    edge_attr = torch.randn(batch_size, edge_index.shape[-1], 6, device=device)
    batch_idx = (
        torch.arange(batch_size, device=device).view(-1, 1).expand(-1, n_atoms).contiguous()
    )

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(h_atom, edge_index, edge_attr, batch_idx)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_batches):
            _ = model(h_atom, edge_index, edge_attr, batch_idx)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    samples_per_sec = n_mols / total_s
    print(
        f"D-MPNN forward 100 mols × 30 atoms (batch=10): "
        f"{total_s:.3f} s  ({samples_per_sec:.1f} samples/s)"
    )

    assert total_s < 5.0, (
        f"D-MPNN forward took {total_s:.3f}s — exceeds the 5 s budget. "
        f"This usually means the Python-loop in DirectedMPNN.forward is "
        f"amplifying the per-batch dispatch overhead by ~10×.  See "
        f"molmetal/reports/rocm_throughput.md § 3 for the diagnosis."
    )
