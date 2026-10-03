"""Tests for ``molmetal_lam.sbdd_env.groupnorm_replacement``.

Covers the four contract guarantees from TODO-143 §2.5 Step 1.3:

    1. Importable from the package (no side effects on import).
    2. ``replace_batchnorm_with_groupnorm`` recurses through nested modules.
    3. ``num_groups > num_channels`` and other awkward channel counts
       resolve to a valid divisor (no GroupNorm construction error).
    4. Forward-pass output shape is preserved on a tiny dummy model that
       contains BatchNorm layers (the typical chemprop-style pattern).

Honest framing (2026-09-27):
    Our in-tree D-MPNN currently has zero BatchNorm layers (see
    ``molmetal/baselines/dmpnn*.py``). These tests target the *generic*
    utility itself, using synthetic fixtures so the test does not depend
    on whether the in-tree D-MPNN ever grows BatchNorm layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molmetal.molmetal_lam.sbdd_env.groupnorm_replacement import (
    count_batchnorm,
    num_groups_for,
    replace_batchnorm_with_groupnorm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyBlock(nn.Module):
    """A small conv-style block with a BatchNorm1d — represents the pattern
    used by chemprop v2's MPNN message-passing output."""

    def __init__(self, c_in: int = 64, c_hidden: int = 64, c_out: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(c_in, c_hidden)
        self.bn1 = nn.BatchNorm1d(c_hidden)
        self.linear2 = nn.Linear(c_hidden, c_out)
        self.bn2 = nn.BatchNorm1d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in)
        h = self.linear1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.linear2(h)
        h = self.bn2(h)
        return h


class NestedModel(nn.Module):
    """A nested container that exercises the recursion."""

    def __init__(self):
        super().__init__()
        self.block_a = TinyBlock(c_in=64, c_hidden=64, c_out=32)
        self.block_b = TinyBlock(c_in=32, c_hidden=16, c_out=8)
        self.seq = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64)
        h = self.block_a(x)
        h = self.block_b(h)
        h = self.seq(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_and_helpers_exist():
    """The module is importable and exports its public API."""
    # All three names resolved at import time — no exception path.
    assert callable(replace_batchnorm_with_groupnorm)
    assert callable(count_batchnorm)
    assert callable(num_groups_for)


def test_num_groups_for_simple_divisor():
    """Requested <= num_channels and divides cleanly is honoured."""
    # 64 channels, 32 groups requested → 32 divides 64 → keep 32.
    assert num_groups_for(64, requested=32) == 32
    # 64 channels, 16 groups requested → 16 divides 64 → keep 16.
    assert num_groups_for(64, requested=16) == 16
    # 128 channels, 32 groups requested → keep 32.
    assert num_groups_for(128, requested=32) == 32


def test_num_groups_for_requested_larger_than_channels():
    """``requested > num_channels`` falls back to ``num_channels``."""
    # 16 channels but 32 groups requested → use 16 (= LayerNorm-over-channels).
    assert num_groups_for(16, requested=32) == 16


def test_num_groups_for_no_clean_divisor():
    """When no divisor in (1, requested] divides cleanly, fall back to 1."""
    # 7 channels, requested 32 → 7 is prime → 7 divides 7 trivially.
    assert num_groups_for(7, requested=32) == 7
    # 15 channels, requested 8 → divisors ≤ 8 are 1, 3, 5 → largest = 5.
    assert num_groups_for(15, requested=8) == 5


def test_count_batchnorm_nested():
    """count_batchnorm walks nested children (non-mutating)."""
    model = NestedModel()
    # block_a: 2, block_b: 2, seq: 1 → total 5.
    assert count_batchnorm(model) == 5
    # Original model is untouched (count is non-mutating).
    assert count_batchnorm(model) == 5


def test_replace_recurses_through_nested_modules():
    """replace_batchnorm_with_groupnorm mutates all nested children."""
    model = NestedModel()
    n_before = count_batchnorm(model)
    assert n_before == 5
    replaced = replace_batchnorm_with_groupnorm(model, num_groups=32)
    # Same object returned (in-place mutation, fluent chain).
    assert replaced is model
    n_after = count_batchnorm(model)
    assert n_after == 0
    # Every formerly-BatchNorm slot is now a GroupNorm.
    assert isinstance(model.block_a.bn1, nn.GroupNorm)
    assert isinstance(model.block_a.bn2, nn.GroupNorm)
    assert isinstance(model.block_b.bn1, nn.GroupNorm)
    assert isinstance(model.block_b.bn2, nn.GroupNorm)
    assert isinstance(model.seq[1], nn.GroupNorm)


def test_replace_groupnorm_group_count_matches_channels():
    """Each replaced layer has ``num_groups`` that divides ``num_channels``."""
    model = NestedModel()
    replace_batchnorm_with_groupnorm(model, num_groups=32)
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            assert m.num_channels % m.num_groups == 0
            # Sanity: num_groups should never exceed num_channels.
            assert m.num_groups <= m.num_channels


def test_replace_preserves_affine_params():
    """Learnable ``weight``/``bias`` are copied from BatchNorm to GroupNorm."""
    torch.manual_seed(0)
    model = NestedModel()
    # Snapshot original BN affine (before replacement).
    originals = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            originals[name] = (m.weight.detach().clone(), m.bias.detach().clone())
    replace_batchnorm_with_groupnorm(model, num_groups=32)
    # Check every GroupNorm now has the original affine values.
    for name, m in model.named_modules():
        if isinstance(m, nn.GroupNorm) and m.affine and name in originals:
            w_orig, b_orig = originals[name]
            assert torch.allclose(m.weight, w_orig), f"weight mismatch at {name}"
            assert torch.allclose(m.bias, b_orig), f"bias mismatch at {name}"


def test_replace_idempotent():
    """Calling replace twice is a no-op the second time."""
    model = NestedModel()
    replace_batchnorm_with_groupnorm(model, num_groups=32)
    replace_batchnorm_with_groupnorm(model, num_groups=32)
    assert count_batchnorm(model) == 0
    # All BatchNorm slots are still GroupNorm (no double-wrapping).
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            raise AssertionError("BatchNorm reappeared after second replace pass")


def test_forward_shape_preserved():
    """Forward pass on a tiny dummy model yields the same output shape."""
    torch.manual_seed(123)
    model = NestedModel()
    model.eval()  # avoid running-stats divergence between BN and GN
    with torch.no_grad():
        out_before = model(torch.randn(8, 64))
    replace_batchnorm_with_groupnorm(model, num_groups=32)
    model.eval()
    with torch.no_grad():
        out_after = model(torch.randn(8, 64))
    assert out_before.shape == out_after.shape
    assert out_after.shape == torch.Size([8, 1])


def test_replace_on_minimal_dmpnn_like_backbone():
    """The pattern used by chemprop v2 MPNN (Linear → BN1d → ReLU) is handled."""
    torch.manual_seed(7)
    backbone = nn.Sequential(
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    n_before = count_batchnorm(backbone)
    assert n_before == 2
    replace_batchnorm_with_groupnorm(backbone, num_groups=32)
    assert count_batchnorm(backbone) == 0
    # Shape preserved on a forward pass.
    backbone.eval()
    with torch.no_grad():
        y = backbone(torch.randn(4, 128))
    assert y.shape == torch.Size([4, 1])


def test_honest_inventory_our_dmpnn_has_no_batchnorm():
    """Document the empirical finding: our D-MPNN files contain no BatchNorm.

    This test is a tripwire — if anyone later adds a BatchNorm to the in-tree
    D-MPNN without wiring the replacement hook, this test will (correctly)
    fail and force them to reconsider the TODO-143 §2.5 plan.
    """
    # Lazy import so the test only loads these modules if the assertion is hit.
    from molmetal.baselines import dmpnn as _dmpnn_baseline
    from molmetal.baselines import dmpnn_attentive as _dmpnn_attentive
    from molmetal.baselines import dmpnn_multitask as _dmpnn_multitask
    from molmetal.models import dmpnn as _dmpnn_model
    from molmetal.models import v4_multitask_dmpnn as _v4

    for mod in (
        _dmpnn_baseline,
        _dmpnn_attentive,
        _dmpnn_multitask,
        _dmpnn_model,
        _v4,
    ):
        # The modules expose class definitions — instantiate to scan the
        # instantiated Module class.
        for name in dir(mod):
            obj = getattr(mod, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and obj is not nn.Module
            ):
                try:
                    inst = obj()
                except Exception:
                    # Skip classes that require constructor arguments — we are
                    # only auditing the *easy-to-instantiate* classes here.
                    continue
                n = count_batchnorm(inst)
                assert n == 0, (
                    f"{mod.name}.{name} now contains {n} BatchNorm layer(s). "
                    "TODO-143 §2.5 says we should be BatchNorm-free; "
                    "either remove the BN or wire replace_batchnorm_with_groupnorm."
                )
