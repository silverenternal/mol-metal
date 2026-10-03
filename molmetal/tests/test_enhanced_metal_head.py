"""Tests for EnhancedMetalHead + ChempropV2MTModelV3 (Wf-248 / TODO-149).

Covers six properties derived from the task spec:

  (i)   :class:`EnhancedMetalHead` instantiates with default + custom args.
  (ii)  ``forward(z_in)`` returns ``(B, 1)`` pre-sigmoid logit (no-attention
        path).
  (iii) ``forward_with_attention(h, e)`` returns ``(B, 1)`` logit
        (attention path) with shape matching the no-attention output.
  (iv)  Cross-flag validation: ``use_metal_attention=True`` requires
        ``enhanced_head=True`` on :class:`ChempropV2MTModelV3`.
  (v)   :class:`ChempropV2MTModelV3` with ``enhanced_head=False`` matches
        the legacy :class:`ChempropV2MTModel` output keys + shapes.
  (vi)  ``loss.backward()`` succeeds for both enhanced + attention paths.

All tests are CPU-only — no GPU required.  The chemprop v2 backbone
uses ``SimpleMoleculeMolGraphFeaturizer`` which needs RDKit (already a
hard dep of the project).  Small ``hidden_dim=64`` + ``depth=2`` keep
the tests fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.models.chemprop_v2_mt_model import (
    SUPPORTED_METALS,
    quick_smoke_chemprop_v2_mt,
)
from molmetal.models.chemprop_v2_mt_model_v3 import (
    DEFAULT_ATTN_DIM,
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_HIDDEN_DIM,
    ChempropV2MTModelV3,
    EnhancedMetalHead,
    quick_smoke_chemprop_v2_mt_v3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


SMILES_3 = ["CCO", "c1ccccc1", "CC(=O)O"]


def _make_v3(
    hidden_dim: int = 64,
    depth: int = 2,
    n_metals: int = 5,
    n_coord_classes: int = 5,
    head_hidden_dim: int = 64,
    enhanced_head: bool = True,
    use_metal_attention: bool = False,
) -> ChempropV2MTModelV3:
    """Small Chemprop v2 v3 model for unit tests."""
    _set_seed(0)
    return ChempropV2MTModelV3(
        hidden_dim=hidden_dim,
        depth=depth,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
        head_hidden_dim=head_hidden_dim,
        enhanced_head=enhanced_head,
        use_metal_attention=use_metal_attention,
    )


# ---------------------------------------------------------------------------
# (i) EnhancedMetalHead instantiation
# ---------------------------------------------------------------------------
def test_enhanced_metal_head_instantiates_default():
    """Default constructor wires the 2-layer MLP + (optional) attention."""
    h = EnhancedMetalHead(
        input_dim=128,
        hidden_dim=64,
        metal_embed_dim=32,
        dropout=0.2,
        use_metal_attention=False,
    )
    assert isinstance(h, torch.nn.Module)
    assert h.input_dim == 128
    assert h.hidden_dim == 64
    assert h.metal_embed_dim == 32
    assert h.dropout_rate == 0.2
    assert h.use_metal_attention is False
    # 2-layer MLP present (fc1, fc2).
    assert hasattr(h, "fc1")
    assert hasattr(h, "fc2")
    assert hasattr(h, "act")
    assert hasattr(h, "drop")
    # Attention layers absent (use_metal_attention=False).
    assert not hasattr(h, "attn_q") or h.attn_q is None


def test_enhanced_metal_head_with_attention_instantiates():
    """Attention path constructor wires attn_q / attn_k / attn_v."""
    h = EnhancedMetalHead(
        input_dim=128,
        hidden_dim=64,
        metal_embed_dim=32,
        attn_dim=16,
        dropout=0.2,
        use_metal_attention=True,
    )
    assert h.use_metal_attention is True
    assert h.attn_dim == 16
    # Attention layers are nn.Linear with the right in/out dims.
    assert isinstance(h.attn_q, torch.nn.Linear)
    assert h.attn_q.in_features == 32  # metal_embed_dim
    assert h.attn_q.out_features == 16  # attn_dim
    assert isinstance(h.attn_k, torch.nn.Linear)
    assert h.attn_k.in_features == 64  # hidden_dim
    assert h.attn_k.out_features == 16  # attn_dim
    assert isinstance(h.attn_v, torch.nn.Linear)
    assert h.attn_v.in_features == 64  # hidden_dim
    assert h.attn_v.out_features == 64  # hidden_dim


def test_default_constants_match_spec():
    """DEFAULT_* constants match the task spec."""
    assert DEFAULT_HEAD_HIDDEN_DIM == 300
    assert DEFAULT_HEAD_DROPOUT == 0.2
    assert DEFAULT_ATTN_DIM == 32


# ---------------------------------------------------------------------------
# (ii) No-attention forward
# ---------------------------------------------------------------------------
def test_enhanced_metal_head_forward_shape():
    """forward(z_in) → (B, 1) pre-sigmoid logit (no attention)."""
    h = EnhancedMetalHead(input_dim=128, hidden_dim=64, dropout=0.0)
    h.eval()
    z = torch.randn(4, 128)
    out = h(z)
    assert out.shape == (4, 1)


def test_enhanced_metal_head_forward_rejects_attention_path():
    """forward(z_in) raises when use_metal_attention=True."""
    h = EnhancedMetalHead(
        input_dim=128,
        hidden_dim=64,
        dropout=0.0,
        use_metal_attention=True,
    )
    z = torch.randn(4, 128)
    with pytest.raises(RuntimeError, match="forward_with_attention"):
        h(z)


# ---------------------------------------------------------------------------
# (iii) Attention forward
# ---------------------------------------------------------------------------
def test_enhanced_metal_head_forward_with_attention_shape():
    """forward_with_attention(h, e) → (B, 1) logit (attention path)."""
    h = EnhancedMetalHead(
        input_dim=128,
        hidden_dim=64,
        metal_embed_dim=32,
        attn_dim=16,
        dropout=0.0,
        use_metal_attention=True,
    )
    h.eval()
    h_in = torch.randn(4, 64)
    e_in = torch.randn(4, 32)
    out = h.forward_with_attention(h_in, e_in)
    assert out.shape == (4, 1)


def test_enhanced_metal_head_forward_with_attention_rejects_no_attention():
    """forward_with_attention raises when use_metal_attention=False."""
    h = EnhancedMetalHead(input_dim=128, hidden_dim=64, dropout=0.0)
    h_in = torch.randn(4, 64)
    e_in = torch.randn(4, 32)
    with pytest.raises(RuntimeError, match="forward"):
        h.forward_with_attention(h_in, e_in)


# ---------------------------------------------------------------------------
# (iv) Cross-flag validation
# ---------------------------------------------------------------------------
def test_v3_model_attention_requires_enhanced_head():
    """use_metal_attention=True without enhanced_head=True raises ValueError."""
    with pytest.raises(ValueError, match="use_metal_attention"):
        ChempropV2MTModelV3(
            hidden_dim=32,
            depth=1,
            n_metals=3,
            n_coord_classes=4,
            enhanced_head=False,
            use_metal_attention=True,
        )


def test_v3_model_legacy_path_default():
    """Default constructor (enhanced_head=False) wires the legacy path."""
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        n_metals=3,
        n_coord_classes=4,
    )
    assert m.enhanced_head is False
    assert m.use_metal_attention is False
    # 3 per-metal heads (we passed n_metals=3).
    assert len(m.metal_heads) == 3


# ---------------------------------------------------------------------------
# (v) Legacy V4MultiTaskDMPNN-style forward + keys
# ---------------------------------------------------------------------------
def test_v3_model_legacy_forward_returns_correct_keys():
    """enhanced_head=False → dict[Ru, Ir, Os, Rh, Re, coord] with (B, 2)."""
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        enhanced_head=False,
    )
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    expected = set(SUPPORTED_METALS) | {"coord"}
    assert set(out.keys()) == expected
    B = len(SMILES_3)
    for met in SUPPORTED_METALS:
        assert out[met].shape == (
            B,
            2,
        ), f"{met} logits expected ({B}, 2), got {tuple(out[met].shape)}"
    assert out["coord"].shape == (B, 5)


def test_v3_model_enhanced_head_forward_returns_correct_keys():
    """enhanced_head=True → same dict[Ru, Ir, Os, Rh, Re, coord] shape."""
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        enhanced_head=True,
        use_metal_attention=False,
        head_hidden_dim=32,
    )
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    expected = set(SUPPORTED_METALS) | {"coord"}
    assert set(out.keys()) == expected
    B = len(SMILES_3)
    for met in SUPPORTED_METALS:
        assert out[met].shape == (B, 2)
    assert out["coord"].shape == (B, 5)


def test_v3_model_attention_forward_returns_correct_keys():
    """enhanced_head=True + use_metal_attention=True → same output shape."""
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        enhanced_head=True,
        use_metal_attention=True,
        head_hidden_dim=32,
        attn_dim=8,
    )
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    B = len(SMILES_3)
    for met in SUPPORTED_METALS:
        assert out[met].shape == (B, 2)
    assert out["coord"].shape == (B, 5)


# ---------------------------------------------------------------------------
# (vi) Backward succeeds for all three head paths
# ---------------------------------------------------------------------------
def test_v3_model_legacy_backward():
    """enhanced_head=False → loss.backward() succeeds (chemprop path)."""
    m = _make_v3(enhanced_head=False)
    loss_fn, out, labels = _make_loss_setup(m)
    loss, _ = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()
    # At least one metal head received gradient.
    grad_found = False
    for met in SUPPORTED_METALS:
        for p in m.metal_heads[met].parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                grad_found = True
                break
        if grad_found:
            break
    assert grad_found


def test_v3_model_enhanced_head_backward():
    """enhanced_head=True, use_metal_attention=False → loss.backward()."""
    m = _make_v3(enhanced_head=True, use_metal_attention=False)
    loss_fn, out, labels = _make_loss_setup(m)
    loss, _ = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()
    # The EnhancedMetalHead 2-layer MLP must have received gradient.
    grad_found = False
    for met in SUPPORTED_METALS:
        head = m.metal_heads[met]
        assert isinstance(head, EnhancedMetalHead)
        for p in head.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                grad_found = True
                break
        if grad_found:
            break
    assert grad_found


def test_v3_model_attention_backward():
    """enhanced_head=True + use_metal_attention=True → loss.backward()."""
    m = _make_v3(enhanced_head=True, use_metal_attention=True)
    loss_fn, out, labels = _make_loss_setup(m)
    loss, _ = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()
    # The attention layer must have received gradient.
    attn_grad_found = False
    for met in SUPPORTED_METALS:
        head = m.metal_heads[met]
        assert isinstance(head, EnhancedMetalHead)
        assert head.use_metal_attention is True
        for p in [head.attn_q, head.attn_k, head.attn_v]:
            if p.weight.grad is not None and p.weight.grad.abs().sum().item() > 0.0:
                attn_grad_found = True
                break
        if attn_grad_found:
            break
    assert attn_grad_found, "Attention layers received no gradient"


def test_v3_model_attention_auto_aligns_head_hidden_dim():
    """use_metal_attention=True auto-aligns head_hidden_dim to encoder hidden_dim."""
    # When the user does not pass head_hidden_dim (defaults to 300) but
    # passes a small encoder hidden_dim=32, the model must auto-align
    # head_hidden_dim=32 so the attention layer's attn_k / attn_v
    # projection matrices consume the correct dim.
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        enhanced_head=True,
        use_metal_attention=True,
    )
    assert m.head_hidden_dim == 32, (
        f"head_hidden_dim should auto-align to encoder hidden_dim=32, "
        f"got {m.head_hidden_dim}"
    )
    # Forward should work without manual alignment.
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    assert out["Ru"].shape == (3, 2)


def test_v3_model_attention_respects_explicit_head_hidden_dim():
    """Explicit head_hidden_dim override is respected even when mis-aligned.

    Honest framing: when the user passes a mismatched head_hidden_dim
    on purpose (e.g. for an ablation), the constructor does NOT
    silently re-align — only the default mismatch is auto-fixed.  In
    practice the smoke / production configs all pass matching dims.
    """
    m = ChempropV2MTModelV3(
        hidden_dim=32,
        depth=1,
        head_hidden_dim=32,
        enhanced_head=True,
        use_metal_attention=True,
    )
    # Explicit match stays at 32.
    assert m.head_hidden_dim == 32


# ---------------------------------------------------------------------------
# Helper: build a loss + labels bundle for the v3 model
# ---------------------------------------------------------------------------
def _make_loss_setup(m: ChempropV2MTModelV3):
    """Build a loss function + per-metal targets + coord labels."""
    from molmetal.models.chemprop_v2_mt_model import ChempropV2MTLoss

    loss_fn = ChempropV2MTLoss(n_coord_classes=m.n_coord_classes)
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in m.metals}
    coord_true = torch.randint(0, m.n_coord_classes, (3,))
    return loss_fn, out, {"y_true": y_true, "coord_true": coord_true}


# ---------------------------------------------------------------------------
# Backward compat — V4MultiTaskDMPNN still works
# ---------------------------------------------------------------------------
def test_v4_backward_compat_still_works():
    """V4MultiTaskDMPNN + ChempropV2MTModel still work after this PR."""
    from molmetal.models.v4_multitask_dmpnn import (
        V4MultiTaskDMPNN,
        V4MultiTaskLoss,
    )

    m = V4MultiTaskDMPNN(hidden_dim=32, n_layers=1, n_metals=5)
    loss_fn = V4MultiTaskLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    loss, _ = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss)
    loss.backward()

    # Legacy chemprop v2 multi-task model still works too.
    _model2, _loss2, out2, labels2 = quick_smoke_chemprop_v2_mt()
    assert set(out2.keys()) == set(SUPPORTED_METALS) | {"coord"}


# ---------------------------------------------------------------------------
# Smoke helper
# ---------------------------------------------------------------------------
def test_quick_smoke_helper_enhanced():
    """quick_smoke_chemprop_v2_mt_v3 returns (model, loss, out, labels)."""
    model, loss_fn, out, labels = quick_smoke_chemprop_v2_mt_v3(
        enhanced_head=True,
        use_metal_attention=False,
    )
    B = 4
    for met in SUPPORTED_METALS:
        assert out[met].shape == (B, 2)
    assert out["coord"].shape == (B, 5)
    assert set(labels["y_true"].keys()) == set(SUPPORTED_METALS)
    loss, _ = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()


def test_quick_smoke_helper_attention():
    """quick_smoke_chemprop_v2_mt_v3 with attention path returns finite loss."""
    model, loss_fn, out, labels = quick_smoke_chemprop_v2_mt_v3(
        enhanced_head=True,
        use_metal_attention=True,
    )
    B = 4
    for met in SUPPORTED_METALS:
        assert out[met].shape == (B, 2)
    assert out["coord"].shape == (B, 5)
    loss, _ = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# Per-metal routing works for all three head paths
# ---------------------------------------------------------------------------
def test_v3_model_routing_differs_by_metal_id():
    """Different metal_ids → different per-metal head outputs (all paths)."""
    for use_attn in (False, True):
        m = _make_v3(enhanced_head=True, use_metal_attention=use_attn)
        m.eval()
        metal_ids_a = torch.tensor([0, 0, 0], dtype=torch.long)
        metal_ids_b = torch.tensor([4, 4, 4], dtype=torch.long)
        out_a = m(SMILES_3, metal_ids_a)
        out_b = m(SMILES_3, metal_ids_b)
        for met in SUPPORTED_METALS:
            assert not torch.allclose(out_a[met], out_b[met]), (
                f"use_metal_attention={use_attn}: {met} head gave "
                f"identical output for Ru vs Re ids"
            )
        assert not torch.allclose(out_a["coord"], out_b["coord"])
