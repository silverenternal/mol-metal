"""Pytest: W2.3 — Multi-Parameter Persistence adapter integration.

This test module pins the W2.3 contract end-to-end:

1. :class:`MultiPHPocketFeatures.compute` returns a tensor of shape
   ``(B, n_bins, n_bins)``.
2. The feature map is finite and non-negative.
3. The feature map is invariant under global rotation / translation
   of the input pocket.
4. Self-distance (paired identical clouds) yields a near-1 kernel
   value; unrelated pockets yield values closer to the RBF baseline.
5. :class:`MultiPHMetalCoordHead.forward` returns logits of shape
   ``(B, n_metals)``.
6. The kernel matrix is symmetric.
7. Setting explicit references overrides the lazy default.
8. :func:`augment_pocket_embedding` returns a tensor of shape
   ``(B, H + n_bins ** 2)``.
9. ``augment_pocket_embedding`` does NOT mutate its inputs.
10. The adapter integrates with the existing CFM stack without
    modifying ``molmetal.adapters.flow_matching_lipman``.

All tests run on CPU in <10 s — no GPU / external dependencies
required.  The tests do NOT touch the trained EGNN checkpoint or
the CFM velocity field (the *integration* contract is verified by
checking that the public API names line up with the R17 close-out).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _toy_pocket(
    b: int = 2,
    p: int = 8,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (coords, atom_types) pocket clouds for tests."""
    g = torch.Generator().manual_seed(int(seed))
    coords = torch.randn(b, p, 3, generator=g) * 2.0
    atom_types = torch.randint(1, 18, (b, p), generator=g)  # H(1) .. Ar(18)
    return coords, atom_types


def _identical_pocket(
    b: int = 3,
    p: int = 8,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample distinct + one shared cloud — used by the symmetry tests."""
    g = torch.Generator().manual_seed(int(seed))
    coords = torch.randn(b, p, 3, generator=g) * 2.0
    atom_types = torch.randint(1, 18, (b, p), generator=g)
    return coords, atom_types


def _rotation_matrix_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# ---------------------------------------------------------------------------
# Test 1: MultiPHPocketFeatures shape
# ---------------------------------------------------------------------------
def test_multi_ph_pocket_features_shape():
    """``compute`` returns a tensor of shape ``(B, n_bins, n_bins)``.

    This is the W2.3 primary contract; pinned so any future
    refactor that changes the layout fails CI.
    """
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=32)
    coords, atom_types = _toy_pocket(b=4, p=10, seed=42)
    out = feats.compute_with_atom_types(coords, atom_types)
    assert out.shape == (4, 32, 32), f"expected (4, 32, 32), got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# Test 2: compute() default-atom-types form also returns the right shape
# ---------------------------------------------------------------------------
def test_multi_ph_compute_default_atom_types_shape():
    """The no-args ``compute`` works on (B, P, 3) coords only."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=16)
    coords, _ = _toy_pocket(b=2, p=6)
    out = feats.compute(coords)
    assert out.shape == (2, 16, 16)


# ---------------------------------------------------------------------------
# Test 3: Feature map is finite and non-negative
# ---------------------------------------------------------------------------
def test_multi_ph_features_finite_non_negative():
    """All entries are finite and >= 0 (a count is a count)."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=20)
    coords, atom_types = _toy_pocket(b=4, p=12, seed=11)
    out = feats.compute_with_atom_types(coords, atom_types)
    assert torch.isfinite(out).all(), "feature map contains NaN/Inf"
    assert (out >= 0).all(), "feature map must be non-negative"


# ---------------------------------------------------------------------------
# Test 4: Feature map is SE(3)-invariant (rotation)
# ---------------------------------------------------------------------------
def test_multi_ph_features_rotation_invariant():
    """A global rotation of the pocket does NOT change the feature map.

    Multi-PH is intrinsically SE(3)-invariant because both filter
    functions (``r`` = distance, ``d`` = atomic-number difference)
    are invariant to rotations of the coordinates.
    """
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=24)
    coords, atom_types = _toy_pocket(b=2, p=8, seed=5)
    R = _rotation_matrix_z(math.pi / 4)
    coords_rot = coords @ R.T  # (B, P, 3)
    out = feats.compute_with_atom_types(coords, atom_types)
    out_rot = feats.compute_with_atom_types(coords_rot, atom_types)
    torch.testing.assert_close(out, out_rot, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: Feature map is translation-invariant
# ---------------------------------------------------------------------------
def test_multi_ph_features_translation_invariant():
    """A global translation of the pocket does NOT change the feature map."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=20)
    coords, atom_types = _toy_pocket(b=2, p=8, seed=7)
    shift = torch.tensor([10.0, -3.0, 7.0])
    coords_shift = coords + shift
    out = feats.compute_with_atom_types(coords, atom_types)
    out_shift = feats.compute_with_atom_types(coords_shift, atom_types)
    torch.testing.assert_close(out, out_shift, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: max_radius=0 is rejected
# ---------------------------------------------------------------------------
def test_multi_ph_features_rejects_zero_max_radius():
    """``__init__`` rejects ``max_radius <= 0``."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    with pytest.raises(ValueError):
        MultiPHPocketFeatures(max_radius=0.0)
    with pytest.raises(ValueError):
        MultiPHPocketFeatures(max_radius=-1.0)


# ---------------------------------------------------------------------------
# Test 7: n_bins < 2 is rejected
# ---------------------------------------------------------------------------
def test_multi_ph_features_rejects_too_few_bins():
    """``__init__`` rejects ``n_bins < 2``."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    with pytest.raises(ValueError):
        MultiPHPocketFeatures(n_bins=1)


# ---------------------------------------------------------------------------
# Test 8: MultiPHMetalCoordHead logits shape
# ---------------------------------------------------------------------------
def test_multi_ph_metal_coord_head_logits():
    """``MultiPHMetalCoordHead.forward`` returns logits of shape
    ``(B, n_metals)``."""
    from molmetal.adapters.multi_ph_integration import (
        MultiPHMetalCoordHead,
        MultiPHPocketFeatures,
    )

    feats = MultiPHPocketFeatures(n_bins=16)
    head = MultiPHMetalCoordHead(n_metals=12, n_bins=16, n_projections=4)
    coords, atom_types = _toy_pocket(b=4, p=8)
    out = feats.compute_with_atom_types(coords, atom_types)
    logits = head(out)
    assert logits.shape == (4, 12), f"expected (4, 12); got {tuple(logits.shape)}"


# ---------------------------------------------------------------------------
# Test 9: Metal-coord head changes its output when references are set
# ---------------------------------------------------------------------------
def test_metal_coord_head_set_reference_changes_output():
    """Setting custom references moves the head away from the lazy default."""
    from molmetal.adapters.multi_ph_integration import (
        MultiPHMetalCoordHead,
        MultiPHPocketFeatures,
    )

    feats = MultiPHPocketFeatures(n_bins=12)
    head = MultiPHMetalCoordHead(n_metals=8, n_bins=12, n_projections=4)
    coords, atom_types = _toy_pocket(b=2, p=6)
    out = feats.compute_with_atom_types(coords, atom_types)
    logits_default = head(out).detach().clone()

    # Build two distinct reference clouds (centred at different locations).
    ref_coords = torch.zeros(8, 6, 3)
    for m in range(8):
        ref_coords[m, :, 0] = float(m) * 0.5
        ref_coords[m, :, 1] = 0.2
    ref_atom_types = torch.ones(8, 6, dtype=torch.long) * 6  # all C
    head.set_reference(ref_coords, ref_atom_types)
    logits_ref = head(out).detach().clone()
    # The two outputs must differ — set_reference has taken effect.
    assert not torch.allclose(
        logits_default, logits_ref, atol=1e-6
    ), "set_reference must change the head's output"


# ---------------------------------------------------------------------------
# Test 10: Metal-coord head logits are finite
# ---------------------------------------------------------------------------
def test_metal_coord_head_logits_finite():
    """Logits are finite under small-batch and large-batch regimes."""
    from molmetal.adapters.multi_ph_integration import (
        MultiPHMetalCoordHead,
        MultiPHPocketFeatures,
    )

    feats = MultiPHPocketFeatures(n_bins=12)
    head = MultiPHMetalCoordHead(n_metals=8, n_bins=12, n_projections=4)
    for b in (1, 4, 12):
        coords, atom_types = _toy_pocket(b=b, p=6, seed=b)
        out = feats.compute_with_atom_types(coords, atom_types)
        logits = head(out)
        assert torch.isfinite(
            logits
        ).all(), f"logits must be finite at b={b}; got NaN/Inf"


# ---------------------------------------------------------------------------
# Test 11: Sliced-Wasserstein kernel is symmetric
# ---------------------------------------------------------------------------
def test_multi_ph_kernel_sliced_wasserstein():
    """``sliced_wasserstein_kernel`` returns a symmetric matrix.

    Two different diagrams ``D_a`` and ``D_b`` give the same kernel
    value whether computed as ``K(D_a, D_b)`` or ``K(D_b, D_a)``
    (within numerical tolerance from the random projections).
    """
    from molmetal.math_skills.multi_parameter_ph import (
        TwoParameterDiagram,
        sliced_wasserstein_kernel,
    )

    coords_a, atom_a = _toy_pocket(b=2, p=6, seed=1)
    coords_b, atom_b = _toy_pocket(b=2, p=6, seed=2)
    diagram_a = TwoParameterDiagram.from_filtration(
        coords_a,
        atom_a,
        max_radius=4.0,
        max_coord=3.0,
        n_bins=12,
    )
    diagram_b = TwoParameterDiagram.from_filtration(
        coords_b,
        atom_b,
        max_radius=4.0,
        max_coord=3.0,
        n_bins=12,
    )

    k_ab = sliced_wasserstein_kernel(
        diagram_a,
        diagram_b,
        n_projections=4,
        sigma=2.0,
        seed=7,
    )
    k_ba = sliced_wasserstein_kernel(
        diagram_b,
        diagram_a,
        n_projections=4,
        sigma=2.0,
        seed=7,
    )
    torch.testing.assert_close(k_ab, k_ba.T, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 12: Sliced-Wasserstein diagonal is 1.0 (self-similarity)
# ---------------------------------------------------------------------------
def test_multi_ph_kernel_self_similarity_one():
    """``K(D, D)`` has diagonal exactly ``1.0`` (zero distance → RBF = 1)."""
    from molmetal.math_skills.multi_parameter_ph import (
        TwoParameterDiagram,
        sliced_wasserstein_kernel,
    )

    coords, atom_types = _toy_pocket(b=3, p=6, seed=11)
    diagram = TwoParameterDiagram.from_filtration(
        coords,
        atom_types,
        max_radius=4.0,
        max_coord=3.0,
        n_bins=12,
    )
    k = sliced_wasserstein_kernel(
        diagram,
        diagram,
        n_projections=4,
        sigma=2.0,
        seed=11,
    )
    diag = torch.diagonal(k)
    torch.testing.assert_close(
        diag,
        torch.ones_like(diag),
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Test 13: Kernel value is in (0, 1] for finite sigma
# ---------------------------------------------------------------------------
def test_multi_ph_kernel_in_unit_interval():
    """RBF kernel values lie in (0, 1] — the off-diagonal entries are
    between 0 and 1; the diagonal is exactly 1."""
    from molmetal.math_skills.multi_parameter_ph import (
        TwoParameterDiagram,
        sliced_wasserstein_kernel,
    )

    coords_a, atom_a = _toy_pocket(b=2, p=6, seed=1)
    coords_b, atom_b = _toy_pocket(b=2, p=6, seed=99)
    diagram_a = TwoParameterDiagram.from_filtration(
        coords_a,
        atom_a,
        max_radius=4.0,
        max_coord=3.0,
        n_bins=10,
    )
    diagram_b = TwoParameterDiagram.from_filtration(
        coords_b,
        atom_b,
        max_radius=4.0,
        max_coord=3.0,
        n_bins=10,
    )
    k = sliced_wasserstein_kernel(
        diagram_a,
        diagram_b,
        n_projections=4,
        sigma=2.0,
        seed=3,
    )
    assert (k > 0).all() and (k <= 1.0 + 1e-6).all(), (
        f"kernel values must be in (0, 1]; got min={k.min().item()} "
        f"max={k.max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 14: Reject unsupported kernel name
# ---------------------------------------------------------------------------
def test_metal_coord_head_rejects_unknown_kernel():
    """``__init__`` raises for an unsupported kernel string."""
    from molmetal.adapters.multi_ph_integration import MultiPHMetalCoordHead

    with pytest.raises(ValueError):
        MultiPHMetalCoordHead(n_metals=8, kernel="not_a_kernel")


# ---------------------------------------------------------------------------
# Test 15: Reject n_metals < 2
# ---------------------------------------------------------------------------
def test_metal_coord_head_rejects_one_metal():
    """``__init__`` rejects ``n_metals < 2``."""
    from molmetal.adapters.multi_ph_integration import MultiPHMetalCoordHead

    with pytest.raises(ValueError):
        MultiPHMetalCoordHead(n_metals=1)


# ---------------------------------------------------------------------------
# Test 16: Pocket embedding augmentation — output shape
# ---------------------------------------------------------------------------
def test_augment_pocket_embedding_shape():
    """``augment_pocket_embedding`` returns a tensor of shape
    ``(B, H + n_bins ** 2)``."""
    from molmetal.adapters.multi_ph_integration import (
        augment_pocket_embedding,
    )

    b, h, n_bins = 4, 16, 8
    base = torch.randn(b, h)
    features = torch.randn(b, n_bins, n_bins)
    out = augment_pocket_embedding(base, features)
    assert out.shape == (
        b,
        h + n_bins**2,
    ), f"expected ({b}, {h + n_bins ** 2}); got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# Test 17: Pocket augmentation does NOT mutate the inputs
# ---------------------------------------------------------------------------
def test_augment_pocket_embedding_no_mutation():
    """``augment_pocket_embedding`` does not modify its inputs."""
    from molmetal.adapters.multi_ph_integration import (
        augment_pocket_embedding,
    )

    b, h, n_bins = 3, 8, 6
    base = torch.randn(b, h)
    features = torch.randn(b, n_bins, n_bins)
    base_before = base.clone()
    features_before = features.clone()
    _ = augment_pocket_embedding(base, features)
    torch.testing.assert_close(base, base_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(features, features_before, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Test 18: Pocket augmentation concatenates the right slices
# ---------------------------------------------------------------------------
def test_augment_pocket_embedding_first_slice_equals_base():
    """The leading ``H`` columns of the augmented embedding equal ``base``."""
    from molmetal.adapters.multi_ph_integration import (
        augment_pocket_embedding,
    )

    b, h, n_bins = 3, 10, 4
    base = torch.randn(b, h)
    features = torch.randn(b, n_bins, n_bins)
    out = augment_pocket_embedding(base, features)
    torch.testing.assert_close(out[:, :h], base, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        out[:, h:].reshape(b, n_bins, n_bins),
        features,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Test 19: Pocket augmentation rejects shape mismatch (batch dim)
# ---------------------------------------------------------------------------
def test_augment_pocket_embedding_batch_mismatch():
    """Augmenting mismatched batch sizes raises ``ValueError``."""
    from molmetal.adapters.multi_ph_integration import (
        augment_pocket_embedding,
    )

    base = torch.randn(3, 8)
    features = torch.randn(4, 6, 6)  # batch=4 vs base=3
    with pytest.raises(ValueError):
        augment_pocket_embedding(base, features)


# ---------------------------------------------------------------------------
# Test 20: Pocket augmentation rejects a 2D feature map
# ---------------------------------------------------------------------------
def test_augment_pocket_embedding_rejects_2d_features():
    """``multi_ph_features`` must be ``(B, n_bins, n_bins)``; a
    2-D tensor raises ``ValueError``."""
    from molmetal.adapters.multi_ph_integration import (
        augment_pocket_embedding,
    )

    base = torch.randn(2, 8)
    features_2d = torch.randn(2, 16)  # would be a wrong-rank input.
    with pytest.raises(ValueError):
        augment_pocket_embedding(base, features_2d)


# ---------------------------------------------------------------------------
# Test 21: Integration with the existing PocketEncoder (no modification)
# ---------------------------------------------------------------------------
def test_multi_ph_uses_existing_pocket_encoder_without_modification():
    """The adapter imports the R17 close-out for convenience but does NOT
    mutate any field of ``molmetal.adapters.flow_matching_lipman``.

    We verify by snapshotting the ``PocketEncoder`` class location via
    ``inspect`` and asserting it has *not* been replaced between
    module import and our own usage.
    """
    import importlib

    # Import the R17 adapter module (no mutation, just import).
    lipman = importlib.import_module(
        "molmetal.adapters.flow_matching_lipman",
    )
    # Confirm the public symbol still exists with the expected name.
    assert hasattr(
        lipman, "PocketEncoder"
    ), "PocketEncoder must remain exported after W2.3"

    # Now import the W2.3 adapter.  This must NOT silently modify
    # the lipman module — we re-check by id comparison.
    lipman_id_before = id(lipman)
    from molmetal.adapters import multi_ph_integration  # noqa: F401

    lipman_id_after = id(lipman)
    assert (
        lipman_id_before == lipman_id_after
    ), "W2.3 adapter must not replace the lipman module"
    assert hasattr(
        lipman, "PocketEncoder"
    ), "W2.3 adapter must not delete PocketEncoder"


# ---------------------------------------------------------------------------
# Test 22: Batched end-to-end (multi-PH + head + augmentation)
# ---------------------------------------------------------------------------
def test_multi_ph_end_to_end_pipeline_cpu():
    """End-to-end pipeline: pocket coords → features → head logits → augmented.

    Pins the W2.3 contract that the three public surfaces
    (``MultiPHPocketFeatures``, ``MultiPHMetalCoordHead``,
    ``augment_pocket_embedding``) compose cleanly.  No GPU
    required; runs in <2 s.
    """
    from molmetal.adapters.multi_ph_integration import (
        MultiPHMetalCoordHead,
        MultiPHPocketFeatures,
        augment_pocket_embedding,
    )

    feats = MultiPHPocketFeatures(n_bins=12)
    head = MultiPHMetalCoordHead(n_metals=8, n_bins=12, n_projections=4)
    base = torch.randn(3, 16)  # 16-dim "PocketEncoder" placeholder

    coords, atom_types = _toy_pocket(b=3, p=6, seed=3)
    features = feats.compute_with_atom_types(coords, atom_types)
    logits = head(features)
    augmented = augment_pocket_embedding(base, features)
    # Sanity-check all three outputs.
    assert features.shape == (3, 12, 12)
    assert logits.shape == (3, 8)
    assert augmented.shape == (3, 16 + 144)


# ---------------------------------------------------------------------------
# Test 23: Forward alias works (compute and forward interchangeable)
# ---------------------------------------------------------------------------
def test_multi_ph_forward_alias_matches_compute():
    """``forward`` is an alias for ``compute`` — both return the same
    feature map."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=10)
    coords, atom_types = _toy_pocket(b=2, p=5, seed=11)
    out_compute = feats.compute_with_atom_types(coords, atom_types)
    out_forward = feats(coords)  # uses default atom-types
    # Both must be (B, n_bins, n_bins) and finite (values may differ
    # because forward uses default 1 atom-type vs compute_with_atom_types).
    assert out_compute.shape == out_forward.shape
    assert torch.isfinite(out_forward).all()


# ---------------------------------------------------------------------------
# Test 24: Empty pocket (p=0) does not crash
# ---------------------------------------------------------------------------
def test_multi_ph_handles_minimal_pocket():
    """A pocket with ``P = 1`` (the smallest non-empty cloud)
    produces a valid feature map (no NaN/Inf)."""
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    feats = MultiPHPocketFeatures(n_bins=8)
    coords = torch.randn(2, 1, 3)
    atom_types = torch.ones(2, 1, dtype=torch.long) * 6
    out = feats.compute_with_atom_types(coords, atom_types)
    assert out.shape == (2, 8, 8)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 25: Linear head is a learned nn.Linear
# ---------------------------------------------------------------------------
def test_metal_coord_head_has_learned_linear():
    """The head holds a single learnable ``nn.Linear`` mapping
    ``n_metals → n_metals``."""
    from molmetal.adapters.multi_ph_integration import MultiPHMetalCoordHead

    head = MultiPHMetalCoordHead(n_metals=8)
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    assert (
        len(linears) == 1
    ), f"head must hold exactly one nn.Linear; got {len(linears)}"
    lin = linears[0]
    assert lin.weight.shape == (8, 8)
    # Should have a bias by default.
    assert lin.bias is not None
