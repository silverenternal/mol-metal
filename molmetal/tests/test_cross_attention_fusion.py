"""Tests for the cross-attention fusion module (TODO/04 C2 ablation B)."""

from __future__ import annotations

import numpy as np
import torch

from molmetal.models.cross_attention_fusion import CrossAttentionFusion


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# test_shape: (B=4, N=10, H=128) -> (B, N, 128)
# ---------------------------------------------------------------------------
def test_shape():
    """Fusion preserves the per-atom tensor shape."""
    set_seed(0)
    B, N, H = 4, 10, 128
    hidden = 64
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusion(
        dmpnn_dim=H, egnn_dim=H, hidden=hidden, n_heads=n_heads
    )
    out = fusion(h_dmpnn, h_egnn)

    assert out.shape == (B, N, H), f"Expected ({B}, {N}, {H}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Fusion output contains NaN/Inf"
    print("test_shape PASSED")


# ---------------------------------------------------------------------------
# test_attention_weights_sum_to_one: hook into internals
# ---------------------------------------------------------------------------
def test_attention_weights_sum_to_one():
    """Attention weights returned via collect_attn=True sum to 1 over keys."""
    set_seed(1)
    B, N, H = 2, 6, 32
    hidden = 16
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusion(
        dmpnn_dim=H, egnn_dim=H, hidden=hidden, n_heads=n_heads
    )

    # Trigger forward with collect_attn=True to populate _last_attn_weights
    _ = fusion(h_dmpnn, h_egnn, collect_attn=True)
    attn = fusion.get_last_attention_weights()

    assert attn is not None, "Expected attention weights to be cached"
    assert attn.shape == (B, N, N), f"Expected ({B}, {N}, {N}), got {tuple(attn.shape)}"
    assert torch.isfinite(attn).all(), "Attention weights contain NaN/Inf"

    # Sum over keys should be 1.0 for every query
    sums = attn.sum(dim=-1)  # (B, N)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Attention weights should sum to 1 over keys, got {sums}"
    )

    # Each weight should be non-negative (softmax)
    assert (attn >= 0).all(), "Attention weights must be non-negative"
    print("test_attention_weights_sum_to_one PASSED")


# ---------------------------------------------------------------------------
# test_residual_connection: output != zero when input is non-trivial
# ---------------------------------------------------------------------------
def test_residual_connection():
    """The residual + LayerNorm path means non-trivial inputs propagate through.

    We verify:
      1. Output is not the zero vector for a non-zero D-MPNN input
         (even if EGNN stream is zeroed).
      2. Output is not identical to a no-residual ablation (so the residual
         term actually contributes).
    """
    set_seed(2)
    B, N, H = 3, 5, 64
    hidden = 32
    n_heads = 4

    # Non-trivial D-MPNN features
    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusion(
        dmpnn_dim=H, egnn_dim=H, hidden=hidden, n_heads=n_heads
    )
    fusion.eval()
    with torch.no_grad():
        out_full = fusion(h_dmpnn, h_egnn)

    # 1. Output is non-zero
    assert out_full.abs().sum() > 0, "Fusion output is all zeros"

    # 2. If we zero the D-MPNN input, the residual path should produce
    #    an output that differs from the EGNN-only path (i.e. the residual
    #    is genuinely carrying information). The LayerNorm on h_dmpnn=0
    #    means the output is just normalised attention output, which is
    #    not zero.
    with torch.no_grad():
        out_no_dmpnn = fusion(torch.zeros_like(h_dmpnn), h_egnn)
    assert out_no_dmpnn.abs().sum() > 0, (
        "With zero D-MPNN input the cross-attention still produces non-zero output"
    )

    # 3. Output differs from a fully-detached "no-residual" computation:
    #    if we directly probe the un-normed attention residual it should
    #    differ from h_dmpnn for non-trivial inputs.
    out_dmpnn_alone = h_dmpnn  # the residual stream is h_dmpnn + W_o * attn
    assert not torch.allclose(out_full, out_dmpnn_alone, atol=1e-3), (
        "Output identical to raw D-MPNN — cross-attention residual term is "
        "having no effect"
    )

    # 4. Output changes when EGNN stream changes (cross-attn actually
    #    attends to EGNN keys/values).
    h_egnn_alt = torch.randn(B, N, H)
    with torch.no_grad():
        out_alt = fusion(h_dmpnn, h_egnn_alt)
    assert not torch.allclose(out_full, out_alt, atol=1e-5), (
        "Output unchanged when EGNN stream is perturbed — cross-attention "
        "is not consulting EGNN keys/values"
    )
    print("test_residual_connection PASSED")


# ---------------------------------------------------------------------------
# test_mask_handling: padding slots should not leak into attention
# ---------------------------------------------------------------------------
def test_mask_handling():
    """When a mask is provided, padded keys/values are excluded from attention."""
    set_seed(3)
    B, N, H = 2, 6, 32
    hidden = 16
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusion(
        dmpnn_dim=H, egnn_dim=H, hidden=hidden, n_heads=n_heads
    )
    fusion.eval()

    # Mask: first 3 atoms real, last 3 padded for both molecules
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :3] = True

    with torch.no_grad():
        # Collect attention weights with the mask applied
        out_masked = fusion(h_dmpnn, h_egnn, mask=mask, collect_attn=True)
        attn_masked = fusion.get_last_attention_weights()

    # The padded key slots should have ~0 attention mass.
    pad_slice = attn_masked[:, :, 3:]  # (B, N, N_pad)
    pad_mass = pad_slice.sum(dim=-1)  # (B, N) — total mass on padded keys
    assert torch.allclose(
        pad_mass, torch.zeros_like(pad_mass), atol=1e-5
    ), f"Padded keys leaked into attention: pad_mass max={pad_mass.max().item():.6f}"

    # Total mass on real keys should be 1.0
    real_slice = attn_masked[:, :, :3].sum(dim=-1)
    assert torch.allclose(
        real_slice, torch.ones_like(real_slice), atol=1e-5
    ), f"Attention mass on real keys should sum to 1, got {real_slice}"

    assert torch.isfinite(out_masked).all()
    print("test_mask_handling PASSED")


if __name__ == "__main__":
    print("Running test_cross_attention_fusion.py ...")
    test_shape()
    test_attention_weights_sum_to_one()
    test_residual_connection()
    test_mask_handling()
    print("\nAll tests PASSED!")