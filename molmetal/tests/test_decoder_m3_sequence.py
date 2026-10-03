"""Pytest: M3 decoder rework — SequenceDecoderHead + GumbelTopKBondDecoder.

Twenty tests covering the M3 follow-up to the R17 Path-B decoder rework:

 1.  test_sequence_decoder_shape
 2.  test_sequence_decoder_vocab_size
 3.  test_sequence_decoder_default_hidden_dim
 4.  test_sequence_decoder_default_max_seq_len
 5.  test_sequence_decoder_handles_padding
 6.  test_sequence_decoder_mask_zeros_padded_positions
 7.  test_sequence_decoder_gradient_flows
 8.  test_gumbel_top_k_sparse
 9.  test_gumbel_top_k_top_k
10.  test_gumbel_top_k_no_self_loops
11.  test_gumbel_top_k_deterministic_eval
12.  test_gumbel_top_k_soft_mode
13.  test_m3_composite_decodes
14.  test_m3_composite_output_shapes
15.  test_m3_composite_with_existing_decoder
16.  test_m3_composite_without_existing_decoder
17.  test_m3_composite_handles_single_atom
18.  test_m3_composite_gradient_flows
19.  test_m3_composite_invalid_inputs_raise
20.  test_module_level_constants

All tests run on CPU in <3 s — no GPU or external evaluators required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.decoder_m3_sequence import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_VOCAB_SIZE,
    GumbelTopKBondDecoder,
    M3CompositeDecoder,
    M3CompositeOutput,
    SequenceDecoderHead,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_atom_features(
    batch: int = 2,
    n_atoms: int = 6,
    hidden_dim: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Random per-atom features."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_atoms, hidden_dim, generator=g)


def _make_bond_features(
    batch: int = 2,
    n_atoms: int = 6,
    seed: int = 0,
    hidden_bond: int = 1,
) -> torch.Tensor:
    """Random per-pair bond features."""
    g = torch.Generator().manual_seed(seed)
    if hidden_bond == 1:
        return torch.randn(batch, n_atoms, n_atoms, generator=g)
    return torch.randn(batch, n_atoms, n_atoms, hidden_bond, generator=g)


def _make_mask(
    batch: int,
    max_seq_len: int,
    valid_lens: list,
) -> torch.Tensor:
    """Build a (B, max_seq_len) bool mask with the given per-sample lengths."""
    m = torch.zeros(batch, max_seq_len, dtype=torch.bool)
    for b, ln in enumerate(valid_lens):
        ln = min(int(ln), max_seq_len)
        m[b, :ln] = True
    return m


# ---------------------------------------------------------------------------
# Test 1: SequenceDecoderHead output shape
# ---------------------------------------------------------------------------
def test_sequence_decoder_shape():
    """Forward returns (batch, max_seq_len, vocab_size) tensor."""
    torch.manual_seed(0)
    seq = SequenceDecoderHead(hidden_dim=128, max_seq_len=64, vocab_size=42)
    z = _make_atom_features(batch=2, n_atoms=6)
    bond = _make_bond_features(batch=2, n_atoms=6)
    mask = _make_mask(batch=2, max_seq_len=64, valid_lens=[6, 6])
    logits = seq(z, bond, mask)
    assert logits.shape == (
        2,
        64,
        42,
    ), f"expected (2, 64, 42), got {tuple(logits.shape)}"


# ---------------------------------------------------------------------------
# Test 2: vocab_size matches PRIMITIVE_ATOMS (42)
# ---------------------------------------------------------------------------
def test_sequence_decoder_vocab_size():
    """vocab_size default = 42 = len(PRIMITIVE_ATOMS)."""
    from molmetal_lam.atoms.combinators import PRIMITIVE_ATOMS

    seq = SequenceDecoderHead()
    assert seq.vocab_size == len(PRIMITIVE_ATOMS), (
        f"vocab_size {seq.vocab_size} must equal len(PRIMITIVE_ATOMS) "
        f"{len(PRIMITIVE_ATOMS)}"
    )
    assert seq.vocab_size == 42, f"vocab_size should be 42, got {seq.vocab_size}"


# ---------------------------------------------------------------------------
# Test 3: default hidden_dim = 128 (Alcaide 2024)
# ---------------------------------------------------------------------------
def test_sequence_decoder_default_hidden_dim():
    """Default hidden_dim is 128 per the Alcaide 2024 §3.1 anchor."""
    seq = SequenceDecoderHead()
    assert (
        seq.hidden_dim == 128
    ), f"hidden_dim default must be 128 (Alcaide 2024), got {seq.hidden_dim}"


# ---------------------------------------------------------------------------
# Test 4: default max_seq_len = 64
# ---------------------------------------------------------------------------
def test_sequence_decoder_default_max_seq_len():
    """Default max_seq_len is 64 (covers typical SBDD ligands with 2x headroom)."""
    seq = SequenceDecoderHead()
    assert (
        seq.max_seq_len == 64
    ), f"max_seq_len default must be 64, got {seq.max_seq_len}"


# ---------------------------------------------------------------------------
# Test 5: handles padding when n_atoms < max_seq_len
# ---------------------------------------------------------------------------
def test_sequence_decoder_handles_padding():
    """When n_atoms < max_seq_len, forward pads with zeros and runs cleanly."""
    torch.manual_seed(0)
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    z = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64)
    bond = _make_bond_features(batch=1, n_atoms=4)
    mask = _make_mask(batch=1, max_seq_len=16, valid_lens=[4])
    logits = seq(z, bond, mask)
    assert logits.shape == (1, 16, 42)
    assert torch.isfinite(logits).all(), "padding must not produce non-finite"


# ---------------------------------------------------------------------------
# Test 6: padded positions have zero logits (mask enforcement)
# ---------------------------------------------------------------------------
def test_sequence_decoder_mask_zeros_padded_positions():
    """Padded positions (``mask=False``) have logits == 0.

    This guarantees callers can use ``argmax`` without worrying
    about which positions are real.
    """
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    z = _make_atom_features(batch=1, n_atoms=4, hidden_dim=64)
    bond = _make_bond_features(batch=1, n_atoms=4)
    # Mask only the first 4 positions as valid; positions 4..15 are padding.
    mask = _make_mask(batch=1, max_seq_len=16, valid_lens=[4])
    seq.eval()  # disable dropout for deterministic comparison
    logits = seq(z, bond, mask)
    # Positions 4..15 must all be exactly zero (we mask-fill them).
    for pos in range(4, 16):
        assert torch.allclose(
            logits[0, pos],
            torch.zeros(42),
            atol=1e-6,
        ), f"position {pos} should be all-zero, got {logits[0, pos]}"


# ---------------------------------------------------------------------------
# Test 7: gradients flow through the sequence decoder
# ---------------------------------------------------------------------------
def test_sequence_decoder_gradient_flows():
    """Backward pass produces non-zero gradients w.r.t. atom_features."""
    torch.manual_seed(0)
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    z = (
        _make_atom_features(batch=1, n_atoms=4, hidden_dim=64)
        .clone()
        .requires_grad_(True)
    )
    bond = _make_bond_features(batch=1, n_atoms=4)
    mask = _make_mask(batch=1, max_seq_len=16, valid_lens=[4])
    logits = seq(z, bond, mask)
    loss = logits.sum()
    loss.backward()
    assert z.grad is not None, "no gradient on atom features"
    assert torch.isfinite(z.grad).all(), "non-finite gradients"
    assert (
        float(z.grad.abs().sum().item()) > 0.0
    ), "gradients should be non-zero for a fresh-init Transformer"


# ---------------------------------------------------------------------------
# Test 8: GumbelTopKBondDecoder output is binary (sparse)
# ---------------------------------------------------------------------------
def test_gumbel_top_k_sparse():
    """Forward returns a binary tensor (entries in {0, 1})."""
    torch.manual_seed(0)
    gumbel = GumbelTopKBondDecoder(temperature=1.0, top_k=4, hard=True)
    bond = torch.randn(2, 6, 6)
    adj = gumbel(bond)
    assert adj.shape == (2, 6, 6)
    unique_vals = torch.unique(adj)
    # All entries must be 0 or 1 in hard mode.
    assert all(
        float(v.item()) in (0.0, 1.0) for v in unique_vals
    ), f"hard mode output should be binary, got unique values {unique_vals}"


# ---------------------------------------------------------------------------
# Test 9: exactly top_k edges per atom
# ---------------------------------------------------------------------------
def test_gumbel_top_k_top_k():
    """Each atom has exactly top_k edges selected."""
    torch.manual_seed(0)
    gumbel = GumbelTopKBondDecoder(temperature=1.0, top_k=4, hard=True)
    gumbel.eval()  # deterministic — no Gumbel noise
    bond = torch.randn(2, 6, 6)
    adj = gumbel(bond, training=False)
    # Per-row (atom) edge count.  Sum along the partner dim (dim=-1).
    per_atom_count = adj.sum(dim=-1)
    expected = torch.full_like(per_atom_count, 4.0)
    assert torch.allclose(per_atom_count, expected), (
        f"each atom should have exactly top_k=4 edges, "
        f"got per-atom counts {per_atom_count.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 10: no self-loops on the diagonal
# ---------------------------------------------------------------------------
def test_gumbel_top_k_no_self_loops():
    """The diagonal of the adjacency matrix is always zero."""
    gumbel = GumbelTopKBondDecoder(top_k=4)
    bond = torch.randn(1, 6, 6)
    adj = gumbel(bond)
    diag = torch.diagonal(adj, dim1=1, dim2=2)
    assert torch.allclose(
        diag, torch.zeros_like(diag)
    ), f"diagonal must be zero (no self-loops), got {diag.tolist()}"


# ---------------------------------------------------------------------------
# Test 11: deterministic in eval mode (no Gumbel noise)
# ---------------------------------------------------------------------------
def test_gumbel_top_k_deterministic_eval():
    """Two eval-mode forward passes return identical results."""
    gumbel = GumbelTopKBondDecoder(temperature=1.0, top_k=3)
    gumbel.eval()
    bond = torch.randn(1, 5, 5)
    out1 = gumbel(bond, training=False)
    out2 = gumbel(bond, training=False)
    assert torch.allclose(out1, out2), "eval mode must be deterministic"


# ---------------------------------------------------------------------------
# Test 12: soft mode produces values in [0, 1]
# ---------------------------------------------------------------------------
def test_gumbel_top_k_soft_mode():
    """In soft mode (hard=False), output is in [0, 1] (sigmoid-scaled)."""
    gumbel = GumbelTopKBondDecoder(temperature=1.0, top_k=2, hard=False)
    bond = torch.randn(1, 5, 5)
    adj = gumbel(bond)
    assert adj.shape == (1, 5, 5)
    assert (
        (adj >= 0.0).all() and (adj <= 1.0).all()
    ), f"soft mode entries must be in [0, 1], got min={adj.min()}, max={adj.max()}"
    # Diagonal still zero.
    diag = torch.diagonal(adj, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.zeros_like(diag))


# ---------------------------------------------------------------------------
# Test 13: M3CompositeDecoder returns valid M3CompositeOutput
# ---------------------------------------------------------------------------
def test_m3_composite_decodes():
    """Composite forward returns M3CompositeOutput with all expected fields."""
    torch.manual_seed(0)
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    gumbel = GumbelTopKBondDecoder(top_k=3)
    composite = M3CompositeDecoder(
        seq_decoder=seq,
        gumbel_decoder=gumbel,
        pocket_dim=32,
    )
    z = _make_atom_features(batch=2, n_atoms=6, hidden_dim=64)
    pocket = torch.randn(2, 32)
    out = composite(z, pocket)
    assert isinstance(
        out, M3CompositeOutput
    ), f"composite must return M3CompositeOutput, got {type(out).__name__}"
    # Without an existing decoder, rework_bond_logits should be None.
    assert out.rework_bond_logits is None
    # Sequence logits and bond adjacency should be present.
    assert out.sequence_logits.shape == (2, 16, 42)
    assert out.bond_adjacency.shape == (2, 6, 6)


# ---------------------------------------------------------------------------
# Test 14: composite output shapes match the inputs
# ---------------------------------------------------------------------------
def test_m3_composite_output_shapes():
    """Sequence logits = (B, max_seq_len, V); bond adjacency = (B, N, N)."""
    seq = SequenceDecoderHead(hidden_dim=128, max_seq_len=32, vocab_size=42)
    composite = M3CompositeDecoder(seq_decoder=seq, pocket_dim=64)
    z = _make_atom_features(batch=3, n_atoms=10, hidden_dim=128)
    pocket = torch.randn(3, 64)
    out = composite(z, pocket)
    assert out.sequence_logits.shape == (3, 32, 42)
    assert out.bond_adjacency.shape == (3, 10, 10)
    # Bond adjacency is binary (default hard=True).
    assert torch.all((out.bond_adjacency == 0) | (out.bond_adjacency == 1))


# ---------------------------------------------------------------------------
# Test 15: composite wires the existing decoder_rework (when provided)
# ---------------------------------------------------------------------------
def test_m3_composite_with_existing_decoder():
    """When an existing decoder is supplied, rework_bond_logits is non-None.

    Uses a stub decoder that exposes ``forward(z, pocket)`` so we don't
    pull in the heavy decoder_rework + RDKit dependency chain.
    """

    class _StubExistingDecoder:
        def __init__(self):
            self.calls = 0

        def forward(self, z, pocket):
            self.calls += 1
            # Return a 1-D tensor of "bond logits" so the composite
            # contract is satisfied.
            return torch.zeros(z.shape[0] * z.shape[1] * (z.shape[1] - 1) // 2)

    stub = _StubExistingDecoder()
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    composite = M3CompositeDecoder(
        existing_decoder=stub,
        seq_decoder=seq,
        pocket_dim=32,
    )
    z = _make_atom_features(batch=2, n_atoms=6, hidden_dim=64)
    pocket = torch.randn(2, 32)
    out = composite(z, pocket)
    assert stub.calls == 1, (
        f"existing_decoder.forward should be called exactly once, " f"got {stub.calls}"
    )
    assert (
        out.rework_bond_logits is not None
    ), "rework_bond_logits must be populated when existing_decoder is given"
    assert out.rework_bond_logits.dim() == 1


# ---------------------------------------------------------------------------
# Test 16: composite works without an existing decoder (head-only mode)
# ---------------------------------------------------------------------------
def test_m3_composite_without_existing_decoder():
    """When existing_decoder is None, the composite runs in head-only mode."""
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    composite = M3CompositeDecoder(
        existing_decoder=None,
        seq_decoder=seq,
        pocket_dim=32,
    )
    z = _make_atom_features(batch=1, n_atoms=5, hidden_dim=64)
    pocket = torch.randn(1, 32)
    out = composite(z, pocket)
    assert out.rework_bond_logits is None
    # Sequence + bond adjacency still produced.
    assert out.sequence_logits.shape == (1, 16, 42)
    assert out.bond_adjacency.shape == (1, 5, 5)


# ---------------------------------------------------------------------------
# Test 17: composite handles a single-atom batch gracefully
# ---------------------------------------------------------------------------
def test_m3_composite_handles_single_atom():
    """A batch with n_atoms=1 must NOT crash; bond adjacency is (1, 1, 1) zeros."""
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=8, vocab_size=42)
    composite = M3CompositeDecoder(seq_decoder=seq, pocket_dim=32)
    z = _make_atom_features(batch=1, n_atoms=1, hidden_dim=64)
    pocket = torch.randn(1, 32)
    mask = _make_mask(batch=1, max_seq_len=8, valid_lens=[1])
    out = composite(z, pocket, mask=mask)
    # Single atom → no possible edges.
    assert out.bond_adjacency.shape == (1, 1, 1)
    assert float(out.bond_adjacency.sum().item()) == 0.0
    # Sequence logits still produced.
    assert out.sequence_logits.shape == (1, 8, 42)


# ---------------------------------------------------------------------------
# Test 18: gradients flow through the composite forward
# ---------------------------------------------------------------------------
def test_m3_composite_gradient_flows():
    """Backward pass produces non-zero gradients w.r.t. z and pocket."""
    torch.manual_seed(0)
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    composite = M3CompositeDecoder(seq_decoder=seq, pocket_dim=32)
    z = (
        _make_atom_features(batch=1, n_atoms=4, hidden_dim=64)
        .clone()
        .requires_grad_(True)
    )
    pocket = torch.randn(1, 32).clone().requires_grad_(True)
    out = composite(z, pocket)
    loss = out.sequence_logits.sum() + out.bond_adjacency.sum()
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().sum().item()) > 0.0
    assert pocket.grad is not None and float(pocket.grad.abs().sum().item()) > 0.0


# ---------------------------------------------------------------------------
# Test 19: composite raises on invalid inputs
# ---------------------------------------------------------------------------
def test_m3_composite_invalid_inputs_raise():
    """Composite forward rejects wrong-shape z and pocket tensors."""
    seq = SequenceDecoderHead(hidden_dim=64, max_seq_len=16, vocab_size=42)
    composite = M3CompositeDecoder(seq_decoder=seq, pocket_dim=32)
    # 2-D z (should be 3-D).
    z2d = torch.randn(2, 6)
    pocket_ok = torch.randn(2, 32)
    with pytest.raises(ValueError, match="z must be"):
        composite(z2d, pocket_ok)
    # Wrong pocket dim.
    z_ok = _make_atom_features(batch=2, n_atoms=6, hidden_dim=64)
    pocket_wrong = torch.randn(2, 16)
    with pytest.raises(ValueError, match="pocket last dim"):
        composite(z_ok, pocket_wrong)
    # Wrong batch alignment.
    pocket_mismatch = torch.randn(3, 32)
    with pytest.raises(ValueError, match="z batch"):
        composite(z_ok, pocket_mismatch)


# ---------------------------------------------------------------------------
# Test 20: module-level constants are exported and match the defaults
# ---------------------------------------------------------------------------
def test_module_level_constants():
    """The module-level DEFAULT_* constants match the class defaults."""
    assert DEFAULT_HIDDEN_DIM == 128
    assert DEFAULT_MAX_SEQ_LEN == 64
    assert DEFAULT_VOCAB_SIZE == 42
    assert DEFAULT_TEMPERATURE == 1.0
    assert DEFAULT_TOP_K == 4
    # The SequenceDecoderHead default constructor uses these values.
    seq = SequenceDecoderHead()
    assert seq.hidden_dim == DEFAULT_HIDDEN_DIM
    assert seq.max_seq_len == DEFAULT_MAX_SEQ_LEN
    assert seq.vocab_size == DEFAULT_VOCAB_SIZE
    # Same for the Gumbel top-k head.
    gumbel = GumbelTopKBondDecoder()
    assert gumbel.temperature == DEFAULT_TEMPERATURE
    assert gumbel.top_k == DEFAULT_TOP_K
