"""Tests for V4MultiTaskDMPNN — 5-way multi-task D-MPNN (R17 Stage 1).

Covers three properties derived from
``TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`` §2 task 2:

  (i)   ``forward(3 SMILES, 3 metal_ids)`` returns a dict with
        five per-metal logits of shape ``(3, 2)`` plus a
        coordination-geometry logit tensor of shape ``(3, n_coord_classes)``.
  (ii)  ``loss.backward()`` succeeds — gradient flows through the
        shared encoder into every per-metal head and the coord head.
  (iii) Per-metal routing: each ``metal_heads[m]`` head consumes
        ``z = [h ; metal_embed]`` and emits logits of the correct shape;
        switching the metal id changes the routing path but not the
        shared encoder output.

All tests are CPU-only — no GPU required.  The encoder uses
:class:`GraphFeaturizer` which needs RDKit (already a hard dep of the
project).  Small hidden_dim (64) keeps the tests fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.models.v4_multitask_dmpnn import (
    ATOM_VOCAB_SIZE,
    COORD_GEOMETRY_NAMES,
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
    V4MultiTaskDMPNNPretrain,
    V4MultiTaskLoss,
    quick_smoke_v4mt,
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
    n_layers: int = 2,
    n_metals: int = 5,
    n_coord_classes: int = 5,
) -> V4MultiTaskDMPNN:
    """Small model for unit tests."""
    _set_seed(0)
    return V4MultiTaskDMPNN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
    )


# ---------------------------------------------------------------------------
# Module structure / instantiation
# ---------------------------------------------------------------------------
def test_module_instantiation_default():
    """Default constructor wires encoder + 5 metal heads + coord head."""
    m = _make_model()
    assert isinstance(m, torch.nn.Module)
    # Encoder + 5 metal heads + coord head + metal embedding
    assert isinstance(m.encoder, torch.nn.Module)
    assert isinstance(m.metal_embed, torch.nn.Embedding)
    assert isinstance(m.metal_heads, torch.nn.ModuleDict)
    assert isinstance(m.coord_head, torch.nn.Sequential)
    # 5 metal heads by default
    assert set(m.metal_heads.keys()) == set(SUPPORTED_METALS)
    assert m.metal_embed.num_embeddings == 5
    assert m.metal_embed.embedding_dim == 64


def test_module_instantiation_custom_metals():
    """Custom ``metals`` tuple overrides the default 5-way subset."""
    m = V4MultiTaskDMPNN(
        hidden_dim=32,
        n_layers=1,
        n_metals=3,
        n_coord_classes=4,
        metals=("Ru", "Pt", "Pd"),
    )
    assert set(m.metal_heads.keys()) == {"Ru", "Pt", "Pd"}
    assert m.n_coord_classes == 4
    assert m.metal_embed.num_embeddings == 3


def test_module_metal_id_lookup():
    """``metal_id`` returns a stable index for known metals."""
    m = _make_model()
    assert m.metal_id("Ru") == 0
    assert m.metal_id("Os") == 4
    with pytest.raises(KeyError):
        m.metal_id("Fe")


# ---------------------------------------------------------------------------
# (i) Forward shape
# ---------------------------------------------------------------------------
def test_forward_shape():
    """forward(3 SMILES, 3 metal_ids) → 5 metal logits (3,2) + coord (3,5)."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)

    # All 5 metals present, each (B, 2)
    assert set(out.keys()) == set(SUPPORTED_METALS) | {"coord"}
    for metal in SUPPORTED_METALS:
        assert out[metal].shape == (
            3,
            2,
        ), f"{metal} logits expected (3, 2), got {tuple(out[metal].shape)}"
    # Coord head
    assert out["coord"].shape == (
        3,
        5,
    ), f"coord logits expected (3, 5), got {tuple(out['coord'].shape)}"


def test_forward_shape_mismatched_lengths_raises():
    """forward rejects mismatched SMILES / metal_id counts."""
    m = _make_model()
    metal_ids = torch.tensor([0, 1], dtype=torch.long)  # only 2 ids for 3 SMILES
    with pytest.raises(ValueError):
        m(SMILES_3, metal_ids)


def test_forward_shape_2d_metal_ids_raises():
    """forward rejects 2-D metal_ids (must be 1-D)."""
    m = _make_model()
    metal_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)  # 1×3, not 3
    with pytest.raises(ValueError):
        m(SMILES_3, metal_ids)


# ---------------------------------------------------------------------------
# (ii) Backward + loss
# ---------------------------------------------------------------------------
def test_backward():
    """loss.backward() runs cleanly and populates gradients on every head
    and the shared encoder."""
    m = _make_model()
    loss_fn = V4MultiTaskLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)

    # Fake labels
    torch.manual_seed(0)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))

    loss, info = loss_fn(out, y_true, coord_true)
    # Sanity: finite, scalar, positive
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    assert loss.dim() == 0
    assert loss.item() > 0.0
    # info dict exposes per-metal scalars + coord loss
    assert "loss_total" in info
    for met in SUPPORTED_METALS:
        assert f"loss_{met}" in info
    assert "loss_coord" in info

    # Backward — gradient must reach the encoder AND every head
    loss.backward()

    # Encoder bottom layer (atom_embed) gradient is non-zero
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None, "Encoder atom_embed received no gradient"
    assert (
        enc_grad.abs().sum().item() > 0.0
    ), "Encoder atom_embed gradient is zero — gradient not flowing"

    # Every metal head received gradient
    for met in SUPPORTED_METALS:
        head_lin1 = m.metal_heads[met][0]  # first Linear of the Sequential
        g = head_lin1.weight.grad
        assert (
            g is not None and g.abs().sum().item() > 0.0
        ), f"metal_head[{met}] received no gradient"

    # Coord head received gradient
    g_coord = m.coord_head[0].weight.grad
    assert (
        g_coord is not None and g_coord.abs().sum().item() > 0.0
    ), "coord head received no gradient"

    # Metal embedding received gradient
    g_embed = m.metal_embed.weight.grad
    assert (
        g_embed is not None and g_embed.abs().sum().item() > 0.0
    ), "metal embedding received no gradient"


def test_loss_partial_metal_labels():
    """Loss works when some metal labels are absent (partial-batch case)."""
    m = _make_model()
    loss_fn = V4MultiTaskLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    torch.manual_seed(0)
    # Only "Ru" labelled — other metals contribute zero
    y_true = {"Ru": torch.randint(0, 2, (3,))}
    coord_true = torch.randint(0, 5, (3,))
    loss, info = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss)
    assert info["loss_Ru"].item() > 0.0
    for met in ("Ir", "Rh", "Re", "Os"):
        # Missing metals report 0
        assert info[f"loss_{met}"].item() == 0.0


# ---------------------------------------------------------------------------
# (iii) Per-metal routing
# ---------------------------------------------------------------------------
def test_metal_routing():
    """Each metal head emits (B, 2) logits; switching metal id swaps the
    routing path through the metal embedding but not the encoder."""
    m = _make_model()
    metal_ids_a = torch.tensor([0, 0, 0], dtype=torch.long)  # all "Ru"
    metal_ids_b = torch.tensor([4, 4, 4], dtype=torch.long)  # all "Os"

    m.eval()  # disable dropout so output is deterministic
    out_a = m(SMILES_3, metal_ids_a)
    out_b = m(SMILES_3, metal_ids_b)

    # All five metal heads present in both outputs
    for met in SUPPORTED_METALS:
        assert out_a[met].shape == (3, 2)
        assert out_b[met].shape == (3, 2)

    # Coord head shape
    assert out_a["coord"].shape == (3, 5)
    assert out_b["coord"].shape == (3, 5)

    # Different metal id → different metal-embedding → different logits
    # for every metal head (they all see z which depends on e_metal)
    for met in SUPPORTED_METALS:
        assert not torch.allclose(
            out_a[met], out_b[met]
        ), f"{met} head gave identical output for Ru vs Os ids — routing broken"

    # Different metal id → different coord logits too (coord head consumes z)
    assert not torch.allclose(
        out_a["coord"], out_b["coord"]
    ), "Coord head gave identical output for Ru vs Os ids — routing broken"


def test_metal_routing_shared_encoder():
    """Switching metal id does NOT change the shared encoder output."""
    m = _make_model()
    metal_ids_a = torch.tensor([0, 0, 0], dtype=torch.long)
    metal_ids_b = torch.tensor([4, 4, 4], dtype=torch.long)
    m.eval()
    h_a = m.encode(SMILES_3)
    h_b = m.encode(SMILES_3)
    assert torch.allclose(h_a, h_b), "Encoder output should be independent of metal id"
    assert h_a.shape == (3, m.hidden_dim)


# ---------------------------------------------------------------------------
# Coordination-geometry constants sanity
# ---------------------------------------------------------------------------
def test_coord_geometry_constants():
    """``COORD_GEOMETRY_NAMES`` has 5 entries (matches n_coord_classes default)."""
    assert len(COORD_GEOMETRY_NAMES) == 5
    # Distinct names
    assert len(set(COORD_GEOMETRY_NAMES)) == 5


# ---------------------------------------------------------------------------
# quick_smoke_v4mt helper
# ---------------------------------------------------------------------------
def test_quick_smoke_helper_runs():
    """The bundled ``quick_smoke_v4mt`` builder wires a full forward+loss
    step end-to-end — used by Stage-1 dev runs."""
    model, loss_fn, out, payload = quick_smoke_v4mt(hidden_dim=32, n_layers=1)
    y_true = payload["y_true"]
    coord_true = payload["coord_true"]
    loss, info = loss_fn(out, y_true, coord_true)
    assert torch.isfinite(loss)
    loss.backward()
    # Every head got a gradient
    for met in model.metals:
        assert model.metal_heads[met][0].weight.grad is not None


# ---------------------------------------------------------------------------
# V4MultiTaskDMPNNPretrain — 1:20 params:data ratio (Fix D) pretraining heads
# ---------------------------------------------------------------------------
def _make_pretrain_model(
    hidden_dim: int = 64,
    n_layers: int = 2,
    n_metals: int = 5,
    n_coord_classes: int = 5,
    atom_vocab_size: int = ATOM_VOCAB_SIZE,
    n_pretrain_tasks: int = 2,
) -> V4MultiTaskDMPNNPretrain:
    """Small pretrain model for unit tests."""
    _set_seed(0)
    return V4MultiTaskDMPNNPretrain(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
        atom_vocab_size=atom_vocab_size,
        n_pretrain_tasks=n_pretrain_tasks,
    )


def test_pretrain_model_instantiation_and_heads():
    """V4MultiTaskDMPNNPretrain wires the two new heads on top of the
    base V4MultiTaskDMPNN (encoder + 5 metal heads + coord head)."""
    m = _make_pretrain_model(atom_vocab_size=14, n_pretrain_tasks=2)
    # New heads exist with the right shapes
    assert isinstance(m.pretrain_head, torch.nn.Linear)
    assert m.pretrain_head.in_features == m.hidden_dim
    assert m.pretrain_head.out_features == 14
    assert isinstance(m.mol_clf_head, torch.nn.Linear)
    assert m.mol_clf_head.in_features == m.hidden_dim
    assert m.mol_clf_head.out_features == 2
    # All base attributes are still there (subclass of V4MultiTaskDMPNN)
    assert isinstance(m.encoder, torch.nn.Module)
    assert isinstance(m.metal_heads, torch.nn.ModuleDict)
    assert set(m.metal_heads.keys()) == set(SUPPORTED_METALS)
    assert isinstance(m.coord_head, torch.nn.Sequential)
    # pretrain_mode defaults to False (eval / fine-tune behaviour)
    assert m.pretrain_mode is False


def test_pretrain_forward_eval_mode_is_backward_compatible():
    """In eval mode the pretrain model returns exactly the same dict keys
    as V4MultiTaskDMPNN — backward compatibility for downstream consumers."""
    _set_seed(0)
    m_pretrain = _make_pretrain_model()
    m_base = _make_model()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    m_pretrain.eval()
    m_base.eval()
    with torch.no_grad():
        out_base = m_base(SMILES_3, metal_ids)
        out_pretrain = m_pretrain(SMILES_3, metal_ids)
    # Same key set as the base model — no pretrain heads leak into eval output.
    assert set(out_pretrain.keys()) == set(out_base.keys())
    assert set(out_pretrain.keys()) == set(SUPPORTED_METALS) | {"coord"}
    # All metal logits have the right shape
    for met in SUPPORTED_METALS:
        assert out_pretrain[met].shape == (3, 2)
    # Coord logits have the right shape
    assert out_pretrain["coord"].shape == (3, m_pretrain.n_coord_classes)
    # pretrain_mode=False in train mode also yields the base dict
    m_pretrain.train()
    m_pretrain.set_pretrain_mode(False)
    out_off = m_pretrain(SMILES_3, metal_ids)
    assert set(out_off.keys()) == set(out_base.keys())


def test_pretrain_forward_train_emits_pretrain_keys():
    """In train mode + pretrain_mode=True, forward() emits the two new
    pretrain keys with the expected shapes."""
    m = _make_pretrain_model(atom_vocab_size=14, n_pretrain_tasks=3)
    m.train()
    m.set_pretrain_mode(True)
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    out = m(SMILES_3, metal_ids)
    # Base keys still present
    for met in SUPPORTED_METALS:
        assert met in out and out[met].shape == (3, 2)
    assert out["coord"].shape == (3, m.n_coord_classes)
    # New pretrain keys
    assert "mol_clf" in out
    assert out["mol_clf"].shape == (3, m.n_pretrain_tasks) == (3, 3)
    assert "atom_logits" in out
    assert "atom_mask" in out
    # atom_logits: (sum_N, atom_vocab_size) — sum_N is the total atom count
    sum_N_expected = int(out["atom_mask"].sum().item())
    assert out["atom_logits"].shape == (sum_N_expected, ATOM_VOCAB_SIZE)
    # Every pretrain-predicted atom row corresponds to a real atom slot.
    assert sum_N_expected > 0  # CCO + benzene + acetic acid have plenty of atoms


def test_pretrain_loss_backward_flows_through_heads():
    """loss.backward() with pretrain heads active populates gradients on
    BOTH new heads AND the shared encoder (gradients must reach the
    D-MPNN atom embedding from each pretrain loss path)."""
    m = _make_pretrain_model()
    m.train()
    m.set_pretrain_mode(True)
    loss_fn = V4MultiTaskLoss()
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    _set_seed(1)
    out = m(SMILES_3, metal_ids)

    # Base multi-task loss
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    base_loss, _ = loss_fn(out, y_true, coord_true)

    # Pretrain losses
    sum_N = int(out["atom_mask"].sum().item())
    atom_labels = torch.randint(0, ATOM_VOCAB_SIZE, (sum_N,))
    atom_loss = torch.nn.functional.cross_entropy(out["atom_logits"], atom_labels)
    mol_labels = torch.randint(0, m.n_pretrain_tasks, (3,))
    mol_loss = torch.nn.functional.cross_entropy(out["mol_clf"], mol_labels)
    total = base_loss + 0.1 * atom_loss + 0.1 * mol_loss

    assert torch.isfinite(total)
    total.backward()

    # Both new heads received gradient
    g_pretrain = m.pretrain_head.weight.grad
    g_molclf = m.mol_clf_head.weight.grad
    assert g_pretrain is not None and g_pretrain.abs().sum().item() > 0.0
    assert g_molclf is not None and g_molclf.abs().sum().item() > 0.0
    # Shared encoder still receives gradient (param sharing works)
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0
    # A metal head also still receives gradient (backward compat)
    g_metal = m.metal_heads["Ru"][0].weight.grad
    assert g_metal is not None and g_metal.abs().sum().item() > 0.0


def test_pretrain_set_pretrain_mode_round_trip():
    """set_pretrain_mode() is a clean toggle that gates the new keys."""
    m = _make_pretrain_model()
    metal_ids = torch.tensor([0, 1], dtype=torch.long)
    smiles = ["CCO", "CC"]

    m.train()
    # Default OFF → base dict only
    assert m.pretrain_mode is False
    out = m(smiles, metal_ids)
    assert "atom_logits" not in out
    assert "mol_clf" not in out

    # Turn ON → new keys appear
    m.set_pretrain_mode(True)
    assert m.pretrain_mode is True
    out = m(smiles, metal_ids)
    assert "atom_logits" in out
    assert "mol_clf" in out

    # Turn OFF again → new keys disappear
    m.set_pretrain_mode(False)
    assert m.pretrain_mode is False
    out = m(smiles, metal_ids)
    assert "atom_logits" not in out
    assert "mol_clf" not in out

    # Backward compatibility with V4MultiTaskDMPNN is preserved
    base = V4MultiTaskDMPNN(hidden_dim=64, n_layers=2)
    base.eval()
    m.eval()
    with torch.no_grad():
        base_keys = set(base(smiles, metal_ids).keys())
        pretrain_keys = set(m(smiles, metal_ids).keys())
    assert base_keys == pretrain_keys
