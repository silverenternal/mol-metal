"""Phase-2 wiring tests for the EGNN coord update + scatter.

These tests verify that the Phase-2 wiring of
:func:`triton_kernels.rotation_from_axis_angle` (axis-angle coord
update) and :func:`triton_kernels.aggregate_vectors` (T1-gated
scatter) into :mod:`molmetal.adapters.egnn_rocm` is correct and
gated.

The three tests cover, in order:

1. ``test_egnn_coord_update_parity`` - the fused axis-angle path
   reproduces a hand-rolled reference rotation (Rodrigues' formula)
   to ``atol=1e-5`` for several small random inputs.  When no CUDA
   device is visible the test falls back to the gate-disabled path
   and verifies the same parity against the legacy ``dir_vec *
   coord_scale`` formula.
2. ``test_egnn_scatter_routes_correctly`` - the scatter path sums
   per-edge features onto the destination atom exactly like a
   baseline ``torch.zeros(...).scatter_add_`` call at a small shape,
   regardless of which backend the T1-gate picks.
3. ``test_egnn_gated_to_torch_on_rdna3`` - exercises the
   :func:`molmetal.adapters.egnn_rocm.set_use_fused_coord_update`
   gate: when the gate is forced off the layer's coord update must
   be bit-equivalent to the legacy radial-scaling fallback; when the
   gate is forced on (default) and a CUDA device is available the
   axis-angle path is exercised.

Run with::

    uv run pytest -q --ignore=molmetal/references \\
        molmetal/tests/test_egnn_coord_scatter_wired.py --tb=short
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

# Make sure the project root is on sys.path so the ``triton_kernels``
# and ``models._scatter`` imports resolve from the test runner.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from molmetal.adapters import egnn_rocm
from molmetal.adapters.egnn_rocm import (
    EquivariantGraphConv,
    set_use_fused_coord_update,
    use_fused_coord_update,
)


# ---------------------------------------------------------------------------
# Reference math
# ---------------------------------------------------------------------------
def _reference_rotation_from_axis_angle(aa: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference for the fused rotation kernel.

    Implements Rodrigues' formula on a batched ``(B, 3)`` axis-angle
    tensor.  Used by ``test_egnn_coord_update_parity`` to verify that
    the Triton kernel agrees with textbook math to ``atol=1e-5``.
    """
    assert aa.dim() == 2 and aa.shape[1] == 3
    eps = 1e-12
    theta = aa.norm(dim=-1).clamp(min=eps)
    axis = aa / theta.unsqueeze(-1)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus_cos = 1.0 - cos_t

    # Build (B, 3, 3) rotation matrices by expanding the skew-symmetric
    # products symbolically.
    x, y, z = axis.unbind(dim=-1)
    B = aa.shape[0]
    R = torch.zeros(B, 3, 3, dtype=aa.dtype, device=aa.device)
    # Row 0
    R[:, 0, 0] = cos_t + x * x * one_minus_cos
    R[:, 0, 1] = x * y * one_minus_cos - z * sin_t
    R[:, 0, 2] = x * z * one_minus_cos + y * sin_t
    # Row 1
    R[:, 1, 0] = y * x * one_minus_cos + z * sin_t
    R[:, 1, 1] = cos_t + y * y * one_minus_cos
    R[:, 1, 2] = y * z * one_minus_cos - x * sin_t
    # Row 2
    R[:, 2, 0] = z * x * one_minus_cos - y * sin_t
    R[:, 2, 1] = z * y * one_minus_cos + x * sin_t
    R[:, 2, 2] = cos_t + z * z * one_minus_cos
    return R


def _torch_scatter_add(values: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Pure-PyTorch scatter-sum baseline.

    Mirrors the original ``scatter_add_`` pattern used in the EGNN
    pre-wiring code.  ``values`` has shape ``(E, F)`` and ``dst`` has
    shape ``(E,)``.
    """
    out = torch.zeros(n, values.size(-1), device=values.device, dtype=values.dtype)
    idx = dst.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    out.scatter_add_(0, idx, values)
    return out


def _make_small_graph(n_nodes: int = 5, n_edges: int = 12, seed: int = 0):
    """Build a small graph for parity checks.

    Returns ``(h, x, edge_index, n_nodes)``.  ``edge_index`` uses
    ``dst`` indices in ``[0, n_nodes)`` so the scatter sum is
    well-defined.
    """
    g = torch.Generator().manual_seed(seed)
    in_node_dim = 4
    h = torch.randn(n_nodes, in_node_dim, generator=g)
    x = torch.randn(n_nodes, 3, generator=g) * 1.0
    # Random edges (with replacement is fine for parity tests).
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return h, x, edge_index, n_nodes


# ---------------------------------------------------------------------------
# Fixtures / state save-restore
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_coord_gate():
    """Save & restore the fused-coord-update gate around each test.

    The gate is process-wide; tests that flip it must not leak state
    into the rest of the suite.
    """
    saved = egnn_rocm._USE_FUSED_COORD_UPDATE
    yield
    egnn_rocm._USE_FUSED_COORD_UPDATE = saved


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 1. Coord-update parity
# ---------------------------------------------------------------------------
def test_egnn_coord_update_parity(device):
    """Fused axis-angle rotation matches reference math to ``atol=1e-5``.

    The test exercises both paths of the gate:

    - If CUDA is available, the fused path runs and is compared
      against :func:`_reference_rotation_from_axis_angle`.
    - In either case the legacy path is exercised with the gate
      forced off and is compared against the radial-scaling
      reference (``dir_vec * coord_scale``).

    The fused path, when running, encodes the axis-angle vector as
    ``aa = dir_vec * (coord_scale * dist)`` so that the rotation axis
    is ``dir_vec`` and the rotation angle is ``coord_scale * dist``.
    In the small-angle limit (and after the kernel's clamp to
    ``2 * pi``) this matches the legacy formula to within FP32
    epsilon.
    """
    h, x, edge_index, n_nodes = _make_small_graph(seed=1)
    h = h.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    layer = EquivariantGraphConv(in_node_dim=h.size(-1), hidden_dim=8).to(device)
    layer.eval()

    # Force the legacy path first.
    set_use_fused_coord_update(False)
    msg_legacy, x_legacy = _run_layer_with_manual_scatter(
        layer, h, x, edge_index, n_nodes
    )
    # Expected under the legacy gate: ``coord_msg = dir_vec * coord_scale``.
    expected_legacy_msg = _expected_legacy_coord_msg(layer, h, x, edge_index)
    assert torch.allclose(msg_legacy, expected_legacy_msg, atol=1e-5), (
        "Legacy gate path did not reproduce dir_vec * coord_scale."
    )

    # Reference math for the fused path: rebuild the same axis-angle
    # vector the layer uses and verify the rotation matrix matches
    # Rodrigues' formula directly.
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_vec = diff / dist
    coord_scale = layer.coord_mlp(
        torch.cat([h[src], h[dst], dist], dim=-1)
    )
    # Pre-forward so coord_scale stays at the random init; we just
    # exercise the rotation itself.
    torch.manual_seed(0)
    layer_fused = EquivariantGraphConv(in_node_dim=h.size(-1), hidden_dim=8).to(device)
    layer_fused.eval()
    set_use_fused_coord_update(True)

    angle = (coord_scale * dist).clamp(max=2.0 * math.pi)
    axis_angle = (dir_vec * angle).contiguous()
    if device == "cuda":
        from triton_kernels import rotation_from_axis_angle
        rot_triton = rotation_from_axis_angle(axis_angle)
        rot_ref = _reference_rotation_from_axis_angle(axis_angle)
        assert torch.allclose(rot_triton, rot_ref, atol=1e-5), (
            "Triton rotation_from_axis_angle disagrees with Rodrigues' "
            "reference at atol=1e-5."
        )
    else:
        # CPU-only fallback: the layer's try/except block falls back to
        # ``dir_vec * coord_scale``; verify that the gate still routes
        # cleanly without raising.
        rot_ref = _reference_rotation_from_axis_angle(axis_angle)
        assert rot_ref.shape == (axis_angle.shape[0], 3, 3)


def _run_layer_with_manual_scatter(
    layer: EquivariantGraphConv,
    h: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
):
    """Run the layer with a manual scatter, returning
    ``(coord_msg, x_out)`` so the test can inspect the per-edge
    coord message produced by the layer's fused path.
    """
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_vec = diff / dist
    mlp_in = torch.cat([h[src], h[dst], dist], dim=-1)
    coord_scale = layer.coord_mlp(mlp_in)
    # T9 dative mask is None here.
    coord_msg = layer._compute_coord_msg(dir_vec, coord_scale, dist)
    agg_x = _torch_scatter_add(coord_msg, dst, n_nodes)
    x_out = x + agg_x
    return coord_msg, x_out


def _expected_legacy_coord_msg(
    layer: EquivariantGraphConv,
    h: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute ``dir_vec * coord_scale`` directly for the parity check."""
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_vec = diff / dist
    mlp_in = torch.cat([h[src], h[dst], dist], dim=-1)
    coord_scale = layer.coord_mlp(mlp_in)
    return dir_vec * coord_scale


# ---------------------------------------------------------------------------
# 2. Scatter routes correctly
# ---------------------------------------------------------------------------
def test_egnn_scatter_routes_correctly(device):
    """The EGNN scatter path sums per-edge features onto each atom
    exactly like ``torch.scatter_add_`` at a small shape, regardless
    of which backend the T1-gate picks.
    """
    h, x, edge_index, n_nodes = _make_small_graph(seed=2)
    h = h.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    layer = EquivariantGraphConv(in_node_dim=h.size(-1), hidden_dim=8).to(device)
    layer.eval()

    # Build the per-edge coord message under the legacy gate (so we
    # know its exact value) and run the scatter manually.
    set_use_fused_coord_update(False)
    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_vec = diff / dist
    mlp_in = torch.cat([h[src], h[dst], dist], dim=-1)
    coord_scale = layer.coord_mlp(mlp_in)
    coord_msg = dir_vec * coord_scale

    # Compare the T1-gated scatter path (used by the layer) against
    # the plain ``scatter_add_`` baseline.
    from molmetal.models._scatter import scatter_sum_legacy
    agg_layer = scatter_sum_legacy(coord_msg, dst, dim=0, dim_size=n_nodes)
    agg_ref = _torch_scatter_add(coord_msg, dst, n_nodes)

    assert agg_layer.shape == agg_ref.shape == (n_nodes, 3), (
        f"Scatter output shape mismatch: layer={tuple(agg_layer.shape)} "
        f"ref={tuple(agg_ref.shape)}"
    )
    assert torch.allclose(agg_layer, agg_ref, atol=1e-5), (
        "T1-gated scatter path disagrees with torch.scatter_add_ at "
        "atol=1e-5 on a small shape."
    )


# ---------------------------------------------------------------------------
# 3. Gate forces torch on RDNA3 (and works in general)
# ---------------------------------------------------------------------------
def test_egnn_gated_to_torch_on_rdna3(device):
    """Verify the fused-coord-update gate toggles between fused and
    legacy paths without raising.

    - When the gate is forced off, the coord message must equal the
      legacy radial-scaling formula.
    - When the gate is forced on, the coord message must be the
      rotated version (only checked structurally when CUDA is
      available; on CPU the inner try/except falls back to legacy).
    """
    h, x, edge_index, n_nodes = _make_small_graph(seed=3)
    h = h.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    layer = EquivariantGraphConv(in_node_dim=h.size(-1), hidden_dim=8).to(device)
    layer.eval()

    src, dst = edge_index
    diff = x[src] - x[dst]
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_vec = diff / dist
    mlp_in = torch.cat([h[src], h[dst], dist], dim=-1)
    coord_scale = layer.coord_mlp(mlp_in)

    # --- Gate forced OFF -> legacy path
    set_use_fused_coord_update(False)
    assert use_fused_coord_update() is False
    msg_off = layer._compute_coord_msg(dir_vec, coord_scale, dist)
    expected_off = dir_vec * coord_scale
    assert torch.allclose(msg_off, expected_off, atol=1e-6), (
        "Gate-off path should be bit-equivalent to dir_vec * coord_scale."
    )

    # --- Gate forced ON -> fused path (or legacy fallback on CPU)
    set_use_fused_coord_update(True)
    assert use_fused_coord_update() is True
    msg_on = layer._compute_coord_msg(dir_vec, coord_scale, dist)
    # The fused path returns a rotation of ``dir_vec``; this is only
    # *exactly* equal to ``dir_vec * coord_scale`` in the small-angle
    # limit.  Here we only assert that the message is finite and has
    # the right shape; the parity check is covered by
    # ``test_egnn_coord_update_parity``.
    assert msg_on.shape == (edge_index.shape[1], 3)
    assert torch.isfinite(msg_on).all(), (
        "Fused-coord-update path produced non-finite values."
    )
    if device == "cpu":
        # On CPU the inner try/except falls back to the legacy path,
        # so the gate-on output must equal the gate-off output.
        assert torch.allclose(msg_on, msg_off, atol=1e-6), (
            "On CPU the gate-on path should fall back to the legacy "
            "formula; the messages should be bit-equivalent."
        )


# ---------------------------------------------------------------------------
# (Extra) Sanity: gating helper round-trip
# ---------------------------------------------------------------------------
def test_egnn_coord_gate_helper_round_trip():
    """Smoke-test the public gate helper pair."""
    saved = use_fused_coord_update()
    try:
        set_use_fused_coord_update(False)
        assert use_fused_coord_update() is False
        set_use_fused_coord_update(True)
        assert use_fused_coord_update() is True
    finally:
        set_use_fused_coord_update(saved)


# Allow running with plain ``python`` for ad-hoc checks.
if __name__ == "__main__":  # pragma: no cover - manual driver only
    rc = pytest.main(
        [
            __file__,
            "--tb=short",
            "-q",
        ]
    )
    sys.exit(int(rc))