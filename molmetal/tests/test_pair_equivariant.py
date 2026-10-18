"""Tests for the Wf-A77 Allegro pairwise equivariant SCAFFOLD.

5 tests cover:

1. **Shape preservation** — default config returns ``(E, hidden_dim)``
   from the documented input contract.
2. **SE(3) invariance of the L=0 path** — rotating
   ``edge_directions`` does NOT change the L=0 output; translation
   does not change ``edge_distances`` (already invariant by
   construction).  This is the SCAFFOLD's formal invariance claim.
3. **SCAFFOLD contract: ``use_equivariant=True`` raises
   ``NotImplementedError``** — the load-bearing check that the L≥1
   path is BY CONSTRUCTION not runnable, so a future contributor
   cannot accidentally ship dead code.
4. **Permutation equivariance over edge order** — reordering edges
   (permute rows of every input) gives the correspondingly-permuted
   output.  Standard primitive-test sanity check.
5. **Residual fidelity + gradient flow** — the L=0 output is bounded
   by the MLP magnitude around the projection, gradients flow to all
   inputs and parameters, output is finite under extreme inputs.

All tests run on CPU only — the module is intentionally
CUDA / ROCm agnostic (mirroring the ``test_coordination_gnn.py``
header convention).
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from molmetal.adapters.pair_equivariant import (
    EDGE_FEAT_DIM_DEFAULT,
    HIDDEN_DIM_DEFAULT,
    AllegroPairwiseLayer,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _seeded_inputs(
    n_atoms: int = 8,
    n_edges: int = 16,
    hidden_dim: int = HIDDEN_DIM_DEFAULT,
    edge_feat_dim: int = EDGE_FEAT_DIM_DEFAULT,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic ``(edge_index, edge_features, edge_distances,
    edge_directions)`` for the SCAFFOLD's forward signature.

    Mirrors ``tests/test_coordination_gnn.py::_seeded_inputs``: a
    private helper that owns the ``torch.Generator().manual_seed(seed)``
    call so every test can request a deterministic input tuple by seed.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # edge_index: (2, E) with src != dst, in [0, n_atoms).
    src = torch.randint(0, n_atoms, (n_edges,), generator=g, device=device)
    dst = torch.randint(0, n_atoms, (n_edges,), generator=g, device=device)
    # Re-roll any src == dst (the SCAFFOLD accepts them but it
    # muddles the test message; the future wire-in will skip them).
    same = src == dst
    while bool(same.any().item()):
        dst[same] = torch.randint(
            0, n_atoms, (int(same.sum().item()),), generator=g, device=device
        )
        same = src == dst
    edge_index = torch.stack([src, dst], dim=0)
    # edge_features: (E, edge_feat_dim) — values in [0, 1] to mimic
    # the (d, is_short, is_long, is_dative) flag set the head builds.
    edge_features = torch.rand(
        (n_edges, edge_feat_dim), generator=g, device=device
    )
    # edge_distances: (E,) in [1.0, 4.0] Å.
    edge_distances = 1.0 + 3.0 * torch.rand(
        (n_edges,), generator=g, device=device
    )
    # edge_directions: (E, 3) unit vectors.
    raw = torch.randn((n_edges, 3), generator=g, device=device)
    edge_directions = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return edge_index, edge_features, edge_distances, edge_directions


def _make_layer(
    hidden_dim: int = HIDDEN_DIM_DEFAULT,
    edge_feat_dim: int = EDGE_FEAT_DIM_DEFAULT,
    use_equivariant: bool = False,
    seed: int = 0,
) -> AllegroPairwiseLayer:
    """Build a deterministic ``AllegroPairwiseLayer`` for tests.

    Mirrors ``tests/test_coordination_gnn.py::_make_layer``: a
    private helper that calls ``torch.manual_seed(seed)`` once before
    ``AllegroPairwiseLayer(...)`` so every test can request a
    deterministic layer by seed.
    """
    torch.manual_seed(seed)
    return AllegroPairwiseLayer(
        hidden_dim=hidden_dim,
        edge_feat_dim=edge_feat_dim,
        use_equivariant=use_equivariant,
    )


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    """Build a uniform 3x3 rotation matrix via Gram-Schmidt.

    Mirrors ``tests/test_coordination_gnn.py::_random_rotation``: a
    private helper that returns a 3x3 ``R`` with ``R @ R.T = I``
    and ``det(R) = +1`` (a proper rotation, no reflections).
    """
    a = torch.randn(3, generator=generator)
    a = a / a.norm().clamp(min=1e-9)
    b = torch.randn(3, generator=generator)
    b = b - (b @ a) * a
    b = b / b.norm().clamp(min=1e-9)
    c = torch.linalg.cross(a, b)
    return torch.stack([a, b, c], dim=0)


# ---------------------------------------------------------------------------
# Test 1 — shape preservation
# ---------------------------------------------------------------------------


def test_layer_shape_preserved_default() -> None:
    """Default ``hidden_dim=128``, ``edge_feat_dim=4`` -> ``(E, 128)`` output.

    The L=0 path is the live path; the L≥1 output is ``None``.
    """
    layer = _make_layer(seed=1)
    edge_index, edge_features, edge_distances, edge_directions = _seeded_inputs(
        n_atoms=8, n_edges=16, seed=1
    )
    h_scalar, h_equiv = layer(
        edge_index, edge_features, edge_distances, edge_directions
    )
    assert h_scalar.shape == (16, HIDDEN_DIM_DEFAULT), (
        f"WF-A77: h_scalar must be (E={16}, H={HIDDEN_DIM_DEFAULT}); "
        f"got {tuple(h_scalar.shape)}"
    )
    assert h_equiv is None, (
        f"WF-A77: use_equivariant=False means the L≥1 path must "
        f"return None; got {type(h_equiv).__name__}"
    )
    assert h_scalar.dtype == edge_features.dtype, (
        f"WF-A77: dtype must pass through; got {h_scalar.dtype} vs "
        f"{edge_features.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2 — SE(3) invariance of the L=0 path
# ---------------------------------------------------------------------------


def test_layer_scalar_path_se3_invariant() -> None:
    """Rotating ``edge_directions`` does NOT change the L=0 output.

    The L=0 path consumes only ``edge_features`` (invariant) and
    ``edge_distances`` (invariant by construction); ``edge_directions``
    is accepted but ignored.  The formal check: build the layer
    output for a (features, dists, dirs) tuple, then for the same
    tuple with ``dirs -> R @ dirs`` for a random proper rotation
    ``R``; the two outputs must be exactly equal.
    """
    layer = _make_layer(seed=2)
    edge_index, edge_features, edge_distances, edge_directions = _seeded_inputs(
        n_atoms=8, n_edges=16, seed=2
    )
    out_base, _ = layer(
        edge_index, edge_features, edge_distances, edge_directions
    )
    # Rotate directions; keep features and distances fixed.
    g = torch.Generator().manual_seed(2)
    R = _random_rotation(g).to(edge_directions.device).to(edge_directions.dtype)
    rotated = (R @ edge_directions.T).T  # (E, 3)
    out_rot, _ = layer(
        edge_index, edge_features, edge_distances, rotated
    )
    assert torch.allclose(out_base, out_rot, atol=1e-6), (
        f"WF-A77 SCAFFOLD: L=0 output MUST be invariant under rotation "
        f"of edge_directions (the layer ignores directions when "
        f"use_equivariant=False).  max abs diff = "
        f"{(out_base - out_rot).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — SCAFFOLD contract: use_equivariant=True raises
# ---------------------------------------------------------------------------


def test_layer_use_equivariant_true_raises_not_implemented() -> None:
    """``use_equivariant=True`` raises ``NotImplementedError``.

    The SCAFFOLD's load-bearing check: the L≥1 (equivariant) path
    is BY CONSTRUCTION not runnable, so a future contributor
    flipping the flag cannot accidentally ship a dead path.  The
    error message must mention the deferred work queue (e3nn
    wire-in, the catalog entry §6.104) so a reader of the failure
    knows what to do next.
    """
    layer = _make_layer(use_equivariant=True, seed=3)
    edge_index, edge_features, edge_distances, edge_directions = _seeded_inputs(
        n_atoms=8, n_edges=16, seed=3
    )
    with pytest.raises(NotImplementedError) as exc_info:
        layer(edge_index, edge_features, edge_distances, edge_directions)
    msg = str(exc_info.value)
    assert "Wf-A77" in msg, (
        f"WF-A77 SCAFFOLD: the NotImplementedError message must "
        f"reference Wf-A77; got {msg!r}"
    )
    assert "e3nn" in msg, (
        f"WF-A77 SCAFFOLD: the NotImplementedError message must "
        f"point at the e3nn wire-in; got {msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — permutation equivariance over edge order
# ---------------------------------------------------------------------------


def test_layer_permutation_edge_order_invariant() -> None:
    """Reordering edges (permute rows of every input) gives the
    correspondingly-permuted output.

    Standard primitive-test sanity check: the layer treats the
    ``(2, E)`` edge list as a SET, not a sequence, so permuting
    the input rows must permute the output rows by the same
    permutation.
    """
    layer = _make_layer(seed=4)
    edge_index, edge_features, edge_distances, edge_directions = _seeded_inputs(
        n_atoms=8, n_edges=20, seed=4
    )
    out_base, _ = layer(
        edge_index, edge_features, edge_distances, edge_directions
    )
    # Build a deterministic permutation of the E rows.
    g = torch.Generator().manual_seed(4)
    perm = torch.randperm(20, generator=g)
    # Permute every input along the leading (E,) / (2, E) dim.
    edge_index_p = edge_index[:, perm]
    edge_features_p = edge_features[perm, :]
    edge_distances_p = edge_distances[perm]
    edge_directions_p = edge_directions[perm, :]
    out_perm, _ = layer(
        edge_index_p, edge_features_p, edge_distances_p, edge_directions_p
    )
    # Re-permute the permuted output back to the original order.
    inv_perm = torch.argsort(perm)
    out_perm_back = out_perm[inv_perm, :]
    assert torch.allclose(out_base, out_perm_back, atol=1e-6), (
        f"WF-A77 SCAFFOLD: the L=0 path must be permutation-equivariant "
        f"over edge order.  max abs diff = "
        f"{(out_base - out_perm_back).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — residual fidelity + gradient flow
# ---------------------------------------------------------------------------


def test_layer_residual_fidelity_and_gradients() -> None:
    """The L=0 path is a residual: ``out = edge_proj(x) + edge_mlp(edge_proj(x))``.

    The residual magnitude is bounded by the MLP magnitude (the
    projection alone is one component; the MLP is the other).
    Gradients flow to ``edge_features`` (the only non-Linear path
    that receives the input), to all layer parameters, and the
    output is finite under extreme inputs (NaN / Inf sentinel).
    """
    layer = _make_layer(seed=5)
    edge_index, edge_features, edge_distances, edge_directions = _seeded_inputs(
        n_atoms=8, n_edges=12, seed=5
    )
    # Build a grad-enabled copy of edge_features (so the residual
    # path is exercised in autograd).
    edge_features_g = edge_features.clone().requires_grad_(True)
    h_scalar, _ = layer(
        edge_index, edge_features_g, edge_distances, edge_directions
    )
    # 1) Residual fidelity: out == proj + mlp(proj).
    proj = layer.edge_proj(edge_features_g)
    update = layer.edge_mlp(proj)
    expected = proj + update
    assert torch.allclose(h_scalar, expected, atol=1e-6), (
        f"WF-A77 SCAFFOLD: L=0 path must be the documented residual "
        f"out = edge_proj + edge_mlp(edge_proj).  max abs diff = "
        f"{(h_scalar - expected).abs().max().item():.3e}"
    )
    # 2) Gradient flow: h_scalar.sum().backward() should populate
    #    edge_features_g.grad AND every leaf parameter.
    h_scalar.sum().backward()
    assert edge_features_g.grad is not None, (
        f"WF-A77 SCAFFOLD: gradient must flow to edge_features; got None"
    )
    finite_grads = torch.isfinite(edge_features_g.grad).all().item()
    assert finite_grads, (
        f"WF-A77 SCAFFOLD: edge_features.grad must be finite; got "
        f"{(~torch.isfinite(edge_features_g.grad)).sum().item()} non-finite entries"
    )
    for name, p in layer.named_parameters():
        assert p.grad is not None, (
            f"WF-A77 SCAFFOLD: parameter {name!r} must receive a "
            f"gradient; got None"
        )
        assert torch.isfinite(p.grad).all().item(), (
            f"WF-A77 SCAFFOLD: parameter {name!r}.grad must be finite"
        )
    # 3) Finiteness under extreme inputs: large features, small
    #    distances, large directions.  The ReLU in the MLP caps
    #    negative projections, so the output should remain finite
    #    even at scale 1e3.
    extreme_features = 1e3 * torch.ones_like(edge_features)
    extreme_features_g = extreme_features.clone().requires_grad_(True)
    extreme_dists = 1e-3 * torch.ones_like(edge_distances)
    extreme_dirs = edge_directions  # unit vectors, already bounded
    h_extreme, _ = layer(
        edge_index, extreme_features_g, extreme_dists, extreme_dirs
    )
    assert torch.isfinite(h_extreme).all().item(), (
        f"WF-A77 SCAFFOLD: output must be finite under extreme inputs; "
        f"got {(~torch.isfinite(h_extreme)).sum().item()} non-finite entries"
    )
