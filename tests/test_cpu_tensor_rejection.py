"""Regression tests for the host-side CPU-tensor guard.

These guard against the cryptic
``ValueError: Pointer argument (at 8) cannot be accessed from Triton
(cpu tensor?)`` that the AMD HIP backend used to raise when the
autotuner probed a Triton kernel with a still-CPU tensor.  The
message obscured the real cause -- the caller forgot ``.to(device)``.

After the fix (see TODO-T2 in ``molmetal/reports/triton_integration_r0.md``)
each Triton-backed wrapper validates its pointer arguments on the
*host* (before the autotune probe runs) and raises a
:class:`RuntimeError` that names the offending tensor.

We exercise three call sites:

  * :func:`triton_kernels.aggregate_vectors` -- the original TODO-T2
    surface; the autotuner used to crash with the cryptic message.
  * :func:`triton_kernels.softmax_last_dim` -- which already fell
    back to ``torch.softmax`` on CPU.  The new wrapper must continue
    to behave that way (no :class:`RuntimeError`).
  * :func:`triton_kernels.fused_cross_entropy` -- same fallback path.

The CUDA-parity smoke test uses the ``(8, 128)`` shape from the
TODO-T2 report and verifies the scatter-sum matches a hand-rolled
``index_add_`` reference.
"""

from __future__ import annotations

import pytest
import torch

from triton_kernels import aggregate_vectors, fused_cross_entropy, softmax_last_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_gpu() -> bool:
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# TODO-T2 -- the core regression
# ---------------------------------------------------------------------------
def test_aggregate_vectors_rejects_cpu_mask_with_clear_message():
    """A CPU ``edge_mask`` must surface a clear :class:`RuntimeError`.

    Before the fix the call died inside the autotune probe with the
    cryptic ``Pointer argument (at 8) cannot be accessed from Triton
    (cpu tensor?)`` message.  After the fix we get a message that
    points at the actual offender (``edge_mask``).
    """
    if _has_gpu():
        # Build most tensors on CUDA but leave `mask` on CPU to mimic
        # the regression (caller forgot to move it).
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device="cuda")
        edge_features = torch.randn(3, 4, device="cuda", dtype=torch.float32)
        edge_mask = torch.tensor([1, 1, 0], dtype=torch.uint8)  # CPU!

        with pytest.raises(RuntimeError) as excinfo:
            aggregate_vectors(edge_index, edge_features, n_atoms=3, edge_mask=edge_mask)

        msg = str(excinfo.value)
        # The message must NOT be the cryptic Triton backend message.
        assert "Pointer argument" not in msg
        assert "cannot be accessed from Triton" not in msg
        # The message MUST name the offending argument so the caller
        # can fix the bug without grepping stack traces.
        assert "edge_mask" in msg
        assert "CPU" in msg or ".to(device)" in msg
    else:
        pytest.skip("CUDA not available -- cannot exercise the regression path")


def test_aggregate_vectors_rejects_cpu_edge_index():
    """CPU ``edge_index`` must also surface a clear message."""
    if not _has_gpu():
        pytest.skip("CUDA not available -- cannot exercise the regression path")

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # CPU
    edge_features = torch.randn(3, 4, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError) as excinfo:
        aggregate_vectors(edge_index, edge_features, n_atoms=3)

    msg = str(excinfo.value)
    assert "Pointer argument" not in msg
    # Either of the edge_index slices (edge_src, edge_dst) should be named.
    assert ("edge_src" in msg) or ("edge_dst" in msg)


# ---------------------------------------------------------------------------
# CPU fallback paths -- must keep behaving correctly
# ---------------------------------------------------------------------------
def test_softmax_falls_back_on_cpu():
    """``softmax_last_dim`` must continue to fall back on CPU.

    The new guard lives INSIDE the autotuned kernel call, but the
    public wrapper already short-circuits to ``torch.softmax`` when
    ``x.is_cuda`` is False.  Verify that path still works.
    """
    x = torch.randn(8, 16, dtype=torch.float32)
    y = softmax_last_dim(x, dim=-1)
    assert torch.allclose(y, torch.softmax(x, dim=-1), atol=1e-6)


def test_fused_cross_entropy_falls_back_on_cpu():
    """``fused_cross_entropy`` must continue to fall back on CPU."""
    logits = torch.randn(8, 16, dtype=torch.float32)
    targets = torch.randint(0, 16, (8,), dtype=torch.long)
    loss = fused_cross_entropy(logits, targets, reduction="mean")
    ref = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(loss, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# GPU parity smoke -- TODO-T2 verification at (8, 128)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_gpu(), reason="CUDA not available")
def test_aggregate_vectors_cuda_parity_at_8_128():
    """End-to-end parity for the TODO-T2 regression at the (8, 128) shape.

    Builds the inputs on CUDA, runs the Triton kernel, and checks the
    scatter-sum against an ``index_add_`` reference.
    """
    torch.manual_seed(0)
    n_atoms = 8
    feat_dim = 128
    n_edges = 24

    dst = torch.randint(0, n_atoms, (n_edges,), dtype=torch.int64, device="cuda")
    src = torch.randint(0, n_atoms, (n_edges,), dtype=torch.int64, device="cuda")
    edge_index = torch.stack([src, dst], dim=0)
    edge_features = torch.randn(n_edges, feat_dim, device="cuda", dtype=torch.float32)
    edge_mask = torch.ones(n_edges, dtype=torch.uint8, device="cuda")

    out = aggregate_vectors(
        edge_index, edge_features, n_atoms=n_atoms, edge_mask=edge_mask
    )
    assert out.shape == (n_atoms, feat_dim)

    # Reference: scatter-add via index_add_.
    ref = torch.zeros(n_atoms, feat_dim, device="cuda", dtype=torch.float32)
    ref.index_add_(0, dst.long(), edge_features)

    assert torch.allclose(out, ref, atol=1e-5), (
        f"max diff = {(out - ref).abs().max().item()}"
    )


@pytest.mark.skipif(not _has_gpu(), reason="CUDA not available")
def test_aggregate_vectors_cuda_parity_with_padding():
    """Same as above but ``edge_mask`` has a few padding entries.

    This was the exact scenario in the TODO-T2 failing tests:
    ``edge_mask`` had CPU zeros on a default-CPU test, and the
    autotuner crashed.  With the fix we can pass a real mask on CUDA
    and verify padding rows are skipped.
    """
    torch.manual_seed(1)
    n_atoms = 8
    feat_dim = 128
    n_edges = 32

    dst = torch.randint(0, n_atoms, (n_edges,), dtype=torch.int64, device="cuda")
    src = torch.randint(0, n_atoms, (n_edges,), dtype=torch.int64, device="cuda")
    edge_index = torch.stack([src, dst], dim=0)
    edge_features = torch.randn(n_edges, feat_dim, device="cuda", dtype=torch.float32)
    edge_mask = torch.ones(n_edges, dtype=torch.uint8, device="cuda")
    # Mask out the last 8 edges as padding.
    edge_mask[-8:] = 0

    out = aggregate_vectors(
        edge_index, edge_features, n_atoms=n_atoms, edge_mask=edge_mask
    )
    assert out.shape == (n_atoms, feat_dim)

    # Reference: scatter-add over the masked rows only.
    valid = edge_mask.bool()
    ref = torch.zeros(n_atoms, feat_dim, device="cuda", dtype=torch.float32)
    ref.index_add_(0, dst[valid].long(), edge_features[valid])

    assert torch.allclose(out, ref, atol=1e-5), (
        f"max diff = {(out - ref).abs().max().item()}"
    )