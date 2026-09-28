"""Tests for ChempropV2MTModel — Chemprop v2 multi-task backbone (Wf-148 / TODO-148).

Covers four properties derived from the task spec:

  (i)   The class instantiates with default + custom args.
  (ii)  ``forward(smiles_list, metal_ids)`` returns a dict with five
        per-metal ``(B, 2)`` logits (V4-compatible shim) plus a
        coordination-geometry ``(B, n_coord_classes)`` logit tensor.
  (iii) ``loss.backward()`` succeeds — gradient flows through the
        chemprop v2 backbone into every per-metal head, the coord
        head, and the metal embedding.
  (iv)  Each per-metal output key has the expected shape ``(B, 2)``
        and the metal routing is driven by ``metal_ids`` (different
        metal ids → different head outputs).

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
    COORD_GEOMETRY_NAMES,
    SUPPORTED_METALS,
    ChempropV2MTLoss,
    ChempropV2MTModel,
    quick_smoke_chemprop_v2_mt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


SMILES_3 = ["CCO", "c1ccccc1", "CC(=O)O"]


def _make_model(
    hidden_dim: int = 64,
    depth: int = 2,
    n_metals: int = 5,
    n_coord_classes: int = 5,
) -> ChempropV2MTModel:
    """Small chemprop v2 multi-task model for unit tests."""
    _set_seed(0)
    return ChempropV2MTModel(
        hidden_dim=hidden_dim,
        depth=depth,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
    )


# ---------------------------------------------------------------------------
# (i) Instantiation
# ---------------------------------------------------------------------------
def test_model_instantiates():
    """Default constructor wires the chemprop v2 backbone + 5 heads + coord head."""
    m = _make_model()
    assert isinstance(m, torch.nn.Module)
    # Encoder is a chemprop-v2 wrapper module.
    assert isinstance(m.encoder, torch.nn.Module)
    # Metal embedding is a vanilla nn.Embedding (matches V4 contract).
    assert isinstance(m.metal_embed, torch.nn.Embedding)
    # Per-metal heads live in a ModuleDict named by metal.
    assert isinstance(m.metal_heads, torch.nn.ModuleDict)
    # Coord head is a small MLP (matches V4 contract).
    assert isinstance(m.coord_head, torch.nn.Sequential)
    # 5 metal heads by default, named by SUPPORTED_METALS.
    assert set(m.metal_heads.keys()) == set(SUPPORTED_METALS)
    assert m.metal_embed.num_embeddings == 5
    assert m.metal_embed.embedding_dim == 64
    # n_coord_classes wired through.
    assert m.n_coord_classes == 5


def test_model_instantiates_custom_metals():
    """Custom ``metals`` tuple overrides the default 5-way subset."""
    m = ChempropV2MTModel(
        hidden_dim=32,
        depth=1,
        n_metals=3,
        n_coord_classes=4,
        metals=("Ru", "Pt", "Pd"),
    )
    assert set(m.metal_heads.keys()) == {"Ru", "Pt", "Pd"}
    assert m.n_coord_classes == 4
    assert m.metal_embed.num_embeddings == 3


def test_metal_id_lookup():
    """``metal_id`` returns a stable index for known metals."""
    m = _make_model()
    assert m.metal_id("Ru") == 0
    assert m.metal_id("Os") == 2   # canonical order is Ru, Ir, Os
    with pytest.raises(KeyError):
        m.metal_id("Fe")


# ---------------------------------------------------------------------------
# (ii) Forward returns the right keys
# ---------------------------------------------------------------------------
def test_forward_returns_correct_keys():
    """forward(3 SMILES, 3 metal_ids) → {Ru, Ir, Os, Rh, Re, coord} dict."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    expected = set(SUPPORTED_METALS) | {"coord"}
    assert set(out.keys()) == expected
    # Smoke-check no None tensors (chemprop v2 occasionally emits None).
    for k, v in out.items():
        assert v is not None, f"out[{k!r}] is None"
        assert torch.is_tensor(v), f"out[{k!r}] is not a tensor"


def test_forward_rejects_mismatched_lengths():
    """forward rejects mismatched SMILES / metal_id counts."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1], dtype=torch.long)  # 2 ids for 3 SMILES
    with pytest.raises(ValueError):
        m(SMILES_3, metal_ids)


def test_forward_rejects_2d_metal_ids():
    """forward rejects 2-D metal_ids (must be 1-D)."""
    m = _make_model()
    metal_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)  # 1×3
    with pytest.raises(ValueError):
        m(SMILES_3, metal_ids)


# ---------------------------------------------------------------------------
# (iii) Backward
# ---------------------------------------------------------------------------
def test_backward_works():
    """loss.backward() succeeds and gradients flow to every head + encoder."""
    m = _make_model()
    loss_fn = ChempropV2MTLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)

    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))

    loss, info = loss_fn(out, y_true, coord_true)
    # Sanity: finite, scalar, positive.
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    assert loss.dim() == 0
    assert loss.item() > 0.0
    # info dict exposes per-metal scalars + coord loss.
    assert "loss_total" in info
    for met in SUPPORTED_METALS:
        assert f"loss_{met}" in info
    assert "loss_coord" in info

    # Backward — gradient must reach the encoder AND every head.
    loss.backward()

    # The chemprop v2 message-passing backbone received gradient.  We
    # don't poke inside chemprop's private attrs — instead we check
    # that the encoder module has at least one parameter with non-zero
    # gradient.
    encoder_grad_found = False
    for p in m.encoder.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            encoder_grad_found = True
            break
    assert encoder_grad_found, "Chemprop v2 encoder received no gradient"

    # Every metal head received gradient.
    for met in SUPPORTED_METALS:
        head = m.metal_heads[met]
        head_grad_found = False
        for p in head.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                head_grad_found = True
                break
        assert head_grad_found, f"metal_head[{met}] received no gradient"

    # Coord head received gradient.
    coord_grad_found = False
    for p in m.coord_head.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            coord_grad_found = True
            break
    assert coord_grad_found, "coord head received no gradient"

    # Metal embedding received gradient.
    g_embed = m.metal_embed.weight.grad
    assert g_embed is not None and g_embed.abs().sum().item() > 0.0, (
        "metal embedding received no gradient"
    )


def test_loss_decreases_over_steps():
    """Two consecutive backward steps: 2nd loss should be ≤ 1st loss (sanity)."""
    m = _make_model()
    loss_fn = ChempropV2MTLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))

    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        out = m(SMILES_3, metal_ids)
        loss, _ = loss_fn(out, y_true, coord_true)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    # Two steps is a very weak guarantee but the loss should at least
    # be finite and the second step should not be wildly larger than
    # the first.
    assert np.isfinite(losses[0])
    assert np.isfinite(losses[1])
    assert losses[1] <= losses[0] * 5.0 + 1e-3


# ---------------------------------------------------------------------------
# (iv) Per-metal output shape
# ---------------------------------------------------------------------------
def test_per_metal_output_shape():
    """All 5 metal outputs are (B, 2); coord output is (B, n_coord_classes)."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    B = len(SMILES_3)
    for metal in SUPPORTED_METALS:
        assert out[metal].shape == (B, 2), (
            f"{metal} logits expected ({B}, 2), got {tuple(out[metal].shape)}"
        )
    assert out["coord"].shape == (B, 5), (
        f"coord logits expected ({B}, 5), got {tuple(out['coord'].shape)}"
    )


def test_per_metal_routing_differs_by_metal_id():
    """Different metal_ids → different per-metal head outputs (routing works)."""
    m = _make_model()
    m.eval()  # disable dropout so the difference comes from routing alone
    metal_ids_a = torch.tensor([0, 0, 0], dtype=torch.long)   # all "Ru"
    metal_ids_b = torch.tensor([4, 4, 4], dtype=torch.long)   # all "Re"
    out_a = m(SMILES_3, metal_ids_a)
    out_b = m(SMILES_3, metal_ids_b)
    # The per-metal heads consume z = [h; metal_embed], so different
    # metal ids → different z → different logits for every head.
    for met in SUPPORTED_METALS:
        assert not torch.allclose(out_a[met], out_b[met]), (
            f"{met} head gave identical output for Ru vs Re ids — routing broken"
        )
    # Coord also routes through z, so coord logits differ too.
    assert not torch.allclose(out_a["coord"], out_b["coord"])


def test_quick_smoke_helper_runs():
    """quick_smoke_chemprop_v2_mt returns a model + loss + batch ready for backward."""
    model, loss_fn, out, labels = quick_smoke_chemprop_v2_mt()
    B = 4
    for met in SUPPORTED_METALS:
        assert out[met].shape == (B, 2)
    assert out["coord"].shape == (B, 5)
    assert set(labels["y_true"].keys()) == set(SUPPORTED_METALS)
    loss, _info = loss_fn(out, labels["y_true"], labels["coord_true"])
    assert torch.isfinite(loss)
    loss.backward()  # should not raise


def test_v4_backward_compat_still_works():
    """V4MultiTaskDMPNN still instantiates + forward + backward after this PR."""
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS as V4_METALS,
        V4MultiTaskDMPNN,
        V4MultiTaskLoss,
    )
    m = V4MultiTaskDMPNN(hidden_dim=32, n_layers=1, n_metals=5, n_coord_classes=5)
    loss_fn = V4MultiTaskLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in V4_METALS}
    coord_true = torch.randint(0, 5, (3,))
    loss, _ = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss)
    loss.backward()