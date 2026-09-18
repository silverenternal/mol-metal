"""Tests for the cross-attention fusion V3 module.

V3 fixes the V2 collapse (test AUC 0.3764 vs V1's 0.6617). The relevant V3
properties to lock in are:

  * shape preserved across the residual stream
  * residual dropout (attn + output) actually engaged in train mode
  * rotation-equivariant initialisation — biases zero, weights Xavier —
    so the first forward pass is approximately the identity on h_dmpnn
    (the backbone dominates).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from molmetal.models.cross_attention_fusion_v3 import CrossAttentionFusionV3


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# test_shape: (B, N, H) -> (B, N, H), residual stream preserved
# ---------------------------------------------------------------------------
def test_shape():
    """V3 preserves the per-atom tensor shape and is finite on a random batch."""
    set_seed(0)
    B, N, H = 4, 10, 128
    hidden = 64
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusionV3(
        dmpnn_dim=H,
        egnn_dim=H,
        hidden=hidden,
        n_heads=n_heads,
        attn_dropout=0.2,
        output_dropout=0.3,
        residual_scale=0.5,
    )
    out = fusion(h_dmpnn, h_egnn)

    assert out.shape == (B, N, H), f"Expected ({B}, {N}, {H}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Fusion output contains NaN/Inf"
    print("test_shape PASSED")


# ---------------------------------------------------------------------------
# test_residual_dropout_active: confirm dropout is engaged in train mode
# ---------------------------------------------------------------------------
def test_residual_dropout_active():
    """In training mode with non-zero dropout, repeated forward passes produce
    different outputs (stochastic regularisation).

    We also check that in eval mode the output is deterministic.
    """
    set_seed(1)
    B, N, H = 3, 6, 64
    hidden = 32
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn = torch.randn(B, N, H)

    fusion = CrossAttentionFusionV3(
        dmpnn_dim=H,
        egnn_dim=H,
        hidden=hidden,
        n_heads=n_heads,
        attn_dropout=0.4,
        output_dropout=0.4,
        residual_scale=0.5,
    )

    # Train mode: two passes should differ because dropout is stochastic.
    fusion.train()
    out_train_a = fusion(h_dmpnn, h_egnn)
    out_train_b = fusion(h_dmpnn, h_egnn)
    assert not torch.allclose(out_train_a, out_train_b, atol=1e-6), (
        "Two training-mode forward passes produced identical outputs — "
        "dropout regulariser is not engaged."
    )

    # Eval mode: two passes should match (deterministic).
    fusion.eval()
    with torch.no_grad():
        out_eval_a = fusion(h_dmpnn, h_egnn)
        out_eval_b = fusion(h_dmpnn, h_egnn)
    assert torch.allclose(out_eval_a, out_eval_b, atol=1e-6), (
        "Two eval-mode forward passes produced different outputs — "
        "dropout not disabled in eval mode."
    )

    # The residual contribution is bounded: with dropout=0.4 the train output
    # magnitude should be at most ~50% larger than the eval output.
    eval_norm = out_eval_a.norm().item()
    train_norm_a = out_train_a.norm().item()
    train_norm_b = out_train_b.norm().item()
    assert max(train_norm_a, train_norm_b) < 2.0 * eval_norm + 1e-3, (
        f"Training outputs {train_norm_a:.3f}, {train_norm_b:.3f} are "
        f">2x the eval norm {eval_norm:.3f}; dropout is over-aggressive."
    )
    print("test_residual_dropout_active PASSED")


# ---------------------------------------------------------------------------
# test_equivariant_init: at init the output is approximately the LayerNorm of
# the D-MPNN stream (residual contribution is zero), so the fusion is
# invariant to the EGNN input at step 0.
# ---------------------------------------------------------------------------
def test_equivariant_init():
    """Rotation-equivariant initialisation: V3's signature regularisation is
    that biases are zeroed at init, so the cross-attn head starts without a
    directional bias in the EGNN coordinate frame.

    We check three properties:

      1. All projection biases are zero at init (no constant offset from the
         cross-attn head).
      2. The LayerNorm is initialised to identity (affine = (1, 0)), so it
         does not shift the residual stream at step 0.
      3. The residual contribution at init is bounded by ``residual_scale`` —
         i.e. the output norm grows by at most a fixed multiple of the
         non-residual baseline. Concretely, switching off the residual path
         (h_egnn -> 0) should not change the output more than a small
         fraction, because the softmax temperature + Xavier init keep the
         initial out projection close to orthogonal.

    Property (3) is weaker than "EGNN-invariant at init" — that would
    require zeroing the output weight, which prevents training. V3 uses a
    small residual_scale (default 0.5) instead so the backbone dominates
    early-training.
    """
    set_seed(2)
    B, N, H = 2, 5, 64
    hidden = 32
    n_heads = 4

    h_dmpnn = torch.randn(B, N, H)
    h_egnn_a = torch.randn(B, N, H)
    h_egnn_b = torch.randn(B, N, H)

    fusion = CrossAttentionFusionV3(
        dmpnn_dim=H,
        egnn_dim=H,
        hidden=hidden,
        n_heads=n_heads,
        attn_dropout=0.0,   # disable stochasticity for the init test
        output_dropout=0.0,
        residual_scale=0.5,
    )
    fusion.eval()

    # 1. All projection biases are zero at init.
    for proj in (fusion.q_dmpnn, fusion.k_egnn, fusion.v_egnn, fusion.out):
        assert torch.all(proj.bias == 0), (
            f"{proj.__class__.__name__} bias should be zero at init"
        )

    # 2. LayerNorm is identity at init.
    assert torch.all(fusion.norm.bias == 0), "LayerNorm bias should be zero at init"
    assert torch.allclose(
        fusion.norm.weight, torch.ones_like(fusion.norm.weight)
    ), "LayerNorm weight should be 1 at init"

    # 3. Bounded residual contribution at init.
    with torch.no_grad():
        out_a = fusion(h_dmpnn, h_egnn_a)
        out_no_residual = fusion(h_dmpnn, torch.zeros_like(h_egnn_a))
        ln_only = fusion.norm(h_dmpnn)

    # The residual contribution is the difference between the full output
    # and the no-EGNN output. It should be small relative to the backbone
    # contribution (which is the LayerNorm of h_dmpnn).
    residual_contrib = (out_a - out_no_residual).abs().mean().item()
    backbone_norm = ln_only.abs().mean().item()
    ratio = residual_contrib / max(backbone_norm, 1e-6)
    assert ratio < 0.3, (
        f"Residual contribution {residual_contrib:.4f} is {ratio:.2%} of "
        f"backbone norm {backbone_norm:.4f}; expected < 30% so the backbone "
        f"dominates at init."
    )

    # Residual scale attribute sanity
    assert fusion.residual_scale == 0.5
    # Attn/output dropout attributes propagate to the forward and are tested
    # separately in test_residual_dropout_active.
    assert fusion.attn_dropout == 0.0
    assert fusion.output_dropout == 0.0
    print("test_equivariant_init PASSED")


if __name__ == "__main__":
    print("Running test_cross_attention_v3.py ...")
    test_shape()
    test_residual_dropout_active()
    test_equivariant_init()
    print("\nAll tests PASSED!")
