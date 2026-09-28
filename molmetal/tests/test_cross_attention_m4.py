"""Pytest: M4 cross-attention pocket conditioning — multi-head + atom bias.

Twenty tests covering the M4 follow-up to the R17 P1.3 per-atom
single-head cross-attention (lines 1183-1204 of
``molmetal/adapters/flow_matching_lipman/__init__.py``):

 1.  test_multi_head_cross_attention_shape
 2.  test_multi_head_cross_attention_n_heads
 3.  test_multi_head_cross_attention_n_heads_invalid
 4.  test_multi_head_cross_attention_dropout
 5.  test_multi_head_cross_attention_dropout_invalid
 6.  test_multi_head_cross_attention_pocket_projection
 7.  test_multi_head_cross_attention_mask_handling
 8.  test_multi_head_cross_attention_residual_path
 9.  test_multi_head_cross_attention_gradient_flows
10.  test_pocket_aware_atom_bias_shape
11.  test_pocket_aware_atom_bias_zero_pocket
12.  test_pocket_aware_atom_bias_mask_handling
13.  test_pocket_aware_atom_bias_gate_init_zero
14.  test_pocket_aware_atom_bias_gradient_flows
15.  test_m4_composite_attention_with_all_three
16.  test_m4_composite_attention_with_only_multi
17.  test_m4_composite_attention_with_only_bias
18.  test_m4_composite_attention_with_only_single
19.  test_m4_composite_attention_invalid_inputs_raise
20.  test_module_level_constants

All tests run on CPU in <3 s — no GPU or external evaluators required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cross_attention_m4 import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_ATOMS,
    DEFAULT_N_HEADS,
    M4CompositeAttention,
    MultiHeadPocketCrossAttention,
    PocketAwareAtomBias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_atom_features(
    batch: int = 2,
    n_atoms: int = 6,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    seed: int = 0,
) -> torch.Tensor:
    """Random per-ligand-atom features."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_atoms, hidden_dim, generator=g)


def _make_pocket_features(
    batch: int = 2,
    n_pocket: int = 8,
    in_dim: int = DEFAULT_HIDDEN_DIM,
    seed: int = 1,
) -> torch.Tensor:
    """Random per-pocket-atom features."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_pocket, in_dim, generator=g)


def _make_pocket_mask(
    batch: int,
    n_pocket: int,
    valid_lens: list,
) -> torch.Tensor:
    """Build a (B, n_pocket) bool mask with the given per-sample lengths."""
    m = torch.zeros(batch, n_pocket, dtype=torch.bool)
    for b, ln in enumerate(valid_lens):
        ln = min(int(ln), n_pocket)
        m[b, :ln] = True
    return m


class _StubSingleHead(nn.Module):
    """A stub P1.3 single-head cross-attention for the composite tests.

    Implements the same forward contract as
    :class:`MultiHeadPocketCrossAttention` but uses a simple Linear
    projection (no attention) so we can test the M4 composite in
    isolation without pulling in the heavy R17 ``flow_matching_lipman``
    dependency.
    """

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        pocket_features: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Mean-pool the pocket features and broadcast back to ligand
        # atoms — a *very* rough proxy for cross-attention but
        # shape-contract compatible with the M4 composite.
        if mask is not None:
            mask_f = mask.bool().unsqueeze(-1).to(pocket_features.dtype)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (pocket_features * mask_f).sum(dim=1) / denom
        else:
            pooled = pocket_features.mean(dim=1)
        ctx = self.proj(pooled).unsqueeze(1).expand(
            atom_features.shape[0], atom_features.shape[1], self.hidden_dim,
        )
        return self.norm(atom_features + ctx)


# ---------------------------------------------------------------------------
# Test 1: MultiHeadPocketCrossAttention output shape
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_shape():
    """Forward returns (batch, n_atoms, hidden_dim) tensor."""
    torch.manual_seed(0)
    attn = MultiHeadPocketCrossAttention(hidden_dim=128, n_heads=8)
    atom = _make_atom_features(batch=2, n_atoms=6, hidden_dim=128)
    pocket = _make_pocket_features(batch=2, n_pocket=8, in_dim=128)
    out = attn(atom, pocket)
    assert out.shape == (2, 6, 128), (
        f"expected (2, 6, 128), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 2: default n_heads = 8 (vs R17 P1.3's 4)
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_n_heads():
    """Default n_heads is 8 (M4.1 spec)."""
    attn = MultiHeadPocketCrossAttention()
    assert attn.n_heads == 8, (
        f"n_heads default must be 8, got {attn.n_heads}"
    )
    assert attn.n_heads == DEFAULT_N_HEADS, (
        f"n_heads default must equal DEFAULT_N_HEADS={DEFAULT_N_HEADS}"
    )
    # head_dim must equal hidden_dim // n_heads.
    assert attn.head_dim == DEFAULT_HIDDEN_DIM // DEFAULT_N_HEADS, (
        f"head_dim ({attn.head_dim}) must equal "
        f"hidden_dim // n_heads ({DEFAULT_HIDDEN_DIM // DEFAULT_N_HEADS})"
    )


# ---------------------------------------------------------------------------
# Test 3: invalid n_heads raises (must divide hidden_dim)
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_n_heads_invalid():
    """n_heads that does not divide hidden_dim raises ValueError."""
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(hidden_dim=128, n_heads=5)
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=3)
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(hidden_dim=128, n_heads=0)


# ---------------------------------------------------------------------------
# Test 4: dropout default = 0.1 (Vaswani 2017 §5.4)
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_dropout():
    """Default dropout is 0.1 per Vaswani 2017 §5.4."""
    attn = MultiHeadPocketCrossAttention()
    assert attn.dropout == DEFAULT_DROPOUT, (
        f"dropout default must be {DEFAULT_DROPOUT}, got {attn.dropout}"
    )
    assert attn.dropout == 0.1, (
        f"dropout default must be 0.1 (Vaswani 2017 §5.4), got {attn.dropout}"
    )


# ---------------------------------------------------------------------------
# Test 5: invalid dropout raises
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_dropout_invalid():
    """Dropout outside [0, 1) raises ValueError."""
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(dropout=1.0)
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(dropout=-0.1)
    with pytest.raises(ValueError):
        MultiHeadPocketCrossAttention(dropout=1.5)


# ---------------------------------------------------------------------------
# Test 6: pocket input projection (pocket_in_dim != hidden_dim)
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_pocket_projection():
    """When pocket_in_dim differs from hidden_dim, a projection is added."""
    # Same dim → identity projection.
    attn_same = MultiHeadPocketCrossAttention(
        hidden_dim=64, n_heads=4, pocket_in_dim=64,
    )
    assert isinstance(attn_same.pocket_proj, nn.Identity), (
        "pocket_proj must be Identity when pocket_in_dim == hidden_dim"
    )
    # Different dim → Linear projection.
    attn_diff = MultiHeadPocketCrossAttention(
        hidden_dim=128, n_heads=8, pocket_in_dim=42,
    )
    assert isinstance(attn_diff.pocket_proj, nn.Linear), (
        "pocket_proj must be Linear when pocket_in_dim != hidden_dim"
    )
    assert attn_diff.pocket_proj.in_features == 42, (
        f"pocket_proj.in_features must be 42, got {attn_diff.pocket_proj.in_features}"
    )
    assert attn_diff.pocket_proj.out_features == 128, (
        f"pocket_proj.out_features must be 128, got {attn_diff.pocket_proj.out_features}"
    )
    # Forward pass works with the projection.
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=128)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=42)
    out = attn_diff(atom, pocket)
    assert out.shape == (1, 4, 128)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 7: mask handling — masked pocket positions excluded from attention
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_mask_handling():
    """Masked pocket positions are excluded from the attention softmax."""
    torch.manual_seed(7)
    attn = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    attn.eval()
    batch, n_atoms, n_pocket, hidden = 2, 3, 5, 64
    atom = _make_atom_features(batch=batch, n_atoms=n_atoms, hidden_dim=hidden, seed=0)
    pocket = _make_pocket_features(batch=batch, n_pocket=n_pocket, in_dim=hidden, seed=1)
    # Mask out the last 2 pocket positions.
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])
    with torch.no_grad():
        out_masked = attn(atom, pocket, mask=mask)
        out_unmasked = attn(atom, pocket)
    # The two outputs MUST differ (mask changes the softmax distribution).
    assert not torch.allclose(out_masked, out_unmasked, atol=1e-6), (
        "mask should change the attention output"
    )
    assert torch.isfinite(out_masked).all()
    assert torch.isfinite(out_unmasked).all()


# ---------------------------------------------------------------------------
# Test 8: residual path — output is not zero for non-trivial input
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_residual_path():
    """Residual + LayerNorm preserves the input shape and is non-zero for
    non-trivial input.
    """
    torch.manual_seed(8)
    attn = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    attn.eval()
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1)
    with torch.no_grad():
        out = attn(atom, pocket)
    assert out.shape == atom.shape
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        "non-trivial input should produce non-zero output"
    )


# ---------------------------------------------------------------------------
# Test 9: gradients flow through the multi-head block
# ---------------------------------------------------------------------------
def test_multi_head_cross_attention_gradient_flows():
    """Backward pass produces non-zero gradients w.r.t. atom_features
    AND pocket_features."""
    torch.manual_seed(9)
    attn = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0).clone().requires_grad_(True)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1).clone().requires_grad_(True)
    out = attn(atom, pocket)
    loss = out.sum()
    loss.backward()
    assert atom.grad is not None, "no gradient on atom features"
    assert pocket.grad is not None, "no gradient on pocket features"
    assert torch.isfinite(atom.grad).all(), "non-finite gradient on atom features"
    assert torch.isfinite(pocket.grad).all(), "non-finite gradient on pocket features"
    assert float(atom.grad.abs().sum().item()) > 0.0
    assert float(pocket.grad.abs().sum().item()) > 0.0


# ---------------------------------------------------------------------------
# Test 10: PocketAwareAtomBias output shape
# ---------------------------------------------------------------------------
def test_pocket_aware_atom_bias_shape():
    """Forward returns (batch, n_atoms, hidden_dim) bias tensor."""
    torch.manual_seed(10)
    bias_mod = PocketAwareAtomBias(hidden_dim=128, max_atoms=64)
    atom = _make_atom_features(batch=2, n_atoms=6, hidden_dim=128)
    pocket = _make_pocket_features(batch=2, n_pocket=8, in_dim=128)
    out = bias_mod(atom, pocket)
    assert out.shape == (2, 6, 128), (
        f"expected (2, 6, 128), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "bias output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 11: zero pocket → zero bias
# ---------------------------------------------------------------------------
def test_pocket_aware_atom_bias_zero_pocket():
    """When pocket_features is all zeros (and gate is zero-init), the bias
    is identically zero.
    """
    torch.manual_seed(11)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    # Sanity: the gate is initialised to zero.
    assert float(bias_mod.gate.item()) == 0.0, (
        f"gate must be initialised to 0.0, got {bias_mod.gate.item()}"
    )
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    zero_pocket = torch.zeros(1, 5, 64)
    out = bias_mod(atom, zero_pocket)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        f"zero pocket + zero gate must produce zero bias, got max abs "
        f"{float(out.abs().max().item())}"
    )


# ---------------------------------------------------------------------------
# Test 12: bias mask handling — masked positions excluded from mean/max pool
# ---------------------------------------------------------------------------
def test_pocket_aware_atom_bias_mask_handling():
    """Mask-aware mean and max pools exclude masked positions."""
    torch.manual_seed(12)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    # Force gate to 1.0 so the descriptor projects through.
    with torch.no_grad():
        bias_mod.gate.fill_(1.0)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1)
    mask_all = torch.ones(1, 5, dtype=torch.bool)
    mask_partial = torch.tensor([[True, True, False, False, False]])
    out_all = bias_mod(atom, pocket, pocket_mask=mask_all)
    out_partial = bias_mod(atom, pocket, pocket_mask=mask_partial)
    # The two outputs MUST differ (mask changes the pool).
    assert not torch.allclose(out_all, out_partial, atol=1e-6), (
        "mask should change the bias output"
    )
    assert torch.isfinite(out_all).all()
    assert torch.isfinite(out_partial).all()


# ---------------------------------------------------------------------------
# Test 13: gate is initialised to zero
# ---------------------------------------------------------------------------
def test_pocket_aware_atom_bias_gate_init_zero():
    """The bias gate is a learnable scalar initialised to 0.0."""
    bias_mod = PocketAwareAtomBias()
    assert bias_mod.gate.shape == (1,), (
        f"gate shape must be (1,), got {tuple(bias_mod.gate.shape)}"
    )
    assert float(bias_mod.gate.item()) == 0.0, (
        f"gate must be initialised to 0.0, got {float(bias_mod.gate.item())}"
    )


# ---------------------------------------------------------------------------
# Test 14: gradients flow through the bias module
# ---------------------------------------------------------------------------
def test_pocket_aware_atom_bias_gradient_flows():
    """Backward pass produces non-zero gradients w.r.t. pocket_features
    AND the gate parameter (after the gate is forced non-zero).
    """
    torch.manual_seed(14)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    # Force the gate to 1.0 so gradient can flow through the gate
    # parameter (zero-init gate blocks its own gradient at step 0).
    with torch.no_grad():
        bias_mod.gate.fill_(1.0)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(
        batch=1, n_pocket=5, in_dim=64, seed=1,
    ).clone().requires_grad_(True)
    out = bias_mod(atom, pocket)
    loss = out.sum()
    loss.backward()
    assert pocket.grad is not None, "no gradient on pocket features"
    assert torch.isfinite(pocket.grad).all(), "non-finite gradient on pocket features"
    assert float(pocket.grad.abs().sum().item()) > 0.0
    assert bias_mod.gate.grad is not None, "no gradient on gate"
    assert torch.isfinite(bias_mod.gate.grad).all(), "non-finite gate gradient"


# ---------------------------------------------------------------------------
# Test 15: M4 composite — all three sub-modules
# ---------------------------------------------------------------------------
def test_m4_composite_attention_with_all_three():
    """Composite runs all three sub-modules and sums their residuals."""
    torch.manual_seed(15)
    single = _StubSingleHead(hidden_dim=64)
    multi = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    composite = M4CompositeAttention(
        single_head=single, multi_head=multi, atom_bias=bias_mod,
    )
    atom = _make_atom_features(batch=2, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=2, n_pocket=5, in_dim=64, seed=1)
    mask = _make_pocket_mask(batch=2, n_pocket=5, valid_lens=[5, 5])
    out = composite(atom, pocket, pocket_mask=mask)
    assert out.shape == (2, 4, 64), (
        f"expected (2, 4, 64), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "composite output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 16: M4 composite — multi-head only
# ---------------------------------------------------------------------------
def test_m4_composite_attention_with_only_multi():
    """Composite runs with only the multi-head block (single_head=None,
    atom_bias=None).
    """
    torch.manual_seed(16)
    multi = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    composite = M4CompositeAttention(multi_head=multi)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1)
    out = composite(atom, pocket)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 17: M4 composite — atom-bias only
# ---------------------------------------------------------------------------
def test_m4_composite_attention_with_only_bias():
    """Composite runs with only the atom-bias block."""
    torch.manual_seed(17)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    composite = M4CompositeAttention(atom_bias=bias_mod)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1)
    out = composite(atom, pocket)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 18: M4 composite — single-head only (P1.3 fallback path)
# ---------------------------------------------------------------------------
def test_m4_composite_attention_with_only_single():
    """Composite runs with only the single-head block — backward-compat
    path that recovers the R17 P1.3 contract.
    """
    torch.manual_seed(18)
    single = _StubSingleHead(hidden_dim=64)
    composite = M4CompositeAttention(single_head=single)
    atom = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64, seed=0)
    pocket = _make_pocket_features(batch=1, n_pocket=5, in_dim=64, seed=1)
    out = composite(atom, pocket)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 19: invalid inputs raise ValueError
# ---------------------------------------------------------------------------
def test_m4_composite_attention_invalid_inputs_raise():
    """Composite raises ValueError on bad shapes / dim mismatches."""
    torch.manual_seed(19)
    multi = MultiHeadPocketCrossAttention(hidden_dim=64, n_heads=4, dropout=0.0)
    bias_mod = PocketAwareAtomBias(hidden_dim=64, max_atoms=32)
    composite = M4CompositeAttention(multi_head=multi, atom_bias=bias_mod)
    # 2-D atom features (must be 3-D).
    with pytest.raises(ValueError):
        composite(
            torch.randn(2, 64),
            _make_pocket_features(batch=2, n_pocket=5, in_dim=64),
        )
    # 2-D pocket (must be 3-D).
    with pytest.raises(ValueError):
        composite(
            _make_atom_features(batch=2, n_atoms=4, hidden_dim=64),
            torch.randn(2, 64),
        )
    # Batch mismatch.
    with pytest.raises(ValueError):
        composite(
            _make_atom_features(batch=2, n_atoms=4, hidden_dim=64),
            _make_pocket_features(batch=3, n_pocket=5, in_dim=64),
        )
    # Hidden-dim mismatch on x.
    with pytest.raises(ValueError):
        composite(
            torch.randn(1, 4, 32),  # wrong hidden_dim
            torch.randn(1, 5, 64),
        )
    # Hidden-dim mismatch on pocket.
    with pytest.raises(ValueError):
        composite(
            torch.randn(1, 4, 64),
            torch.randn(1, 5, 32),  # wrong hidden_dim
        )


# ---------------------------------------------------------------------------
# Test 20: module-level constants match the canonical defaults
# ---------------------------------------------------------------------------
def test_module_level_constants():
    """Module-level DEFAULT_* constants are exposed and match the spec."""
    assert DEFAULT_HIDDEN_DIM == 128, (
        f"DEFAULT_HIDDEN_DIM must be 128, got {DEFAULT_HIDDEN_DIM}"
    )
    assert DEFAULT_N_HEADS == 8, (
        f"DEFAULT_N_HEADS must be 8, got {DEFAULT_N_HEADS}"
    )
    assert DEFAULT_DROPOUT == 0.1, (
        f"DEFAULT_DROPOUT must be 0.1, got {DEFAULT_DROPOUT}"
    )
    assert DEFAULT_MAX_ATOMS == 64, (
        f"DEFAULT_MAX_ATOMS must be 64, got {DEFAULT_MAX_ATOMS}"
    )
    # MultiHeadPocketCrossAttention() with no args matches the defaults.
    attn = MultiHeadPocketCrossAttention()
    assert attn.hidden_dim == DEFAULT_HIDDEN_DIM
    assert attn.n_heads == DEFAULT_N_HEADS
    assert attn.dropout == DEFAULT_DROPOUT
    # PocketAwareAtomBias() with no args matches the defaults.
    bias_mod = PocketAwareAtomBias()
    assert bias_mod.hidden_dim == DEFAULT_HIDDEN_DIM
    assert bias_mod.max_atoms == DEFAULT_MAX_ATOMS