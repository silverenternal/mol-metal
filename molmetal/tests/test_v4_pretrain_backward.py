"""Regression tests for V4MultiTaskDMPNNPretrain backward pass.

Fix 4: validates that backward() succeeds through the two pretraining heads
(``pretrain_head`` masked-atom-prediction, ``mol_clf_head`` molecular
classifier) AND that an optimizer step actually mutates weights.

These tests guard against the regression where the pretrain heads were
attached to the module but their parameters weren't picked up by the
optimizer (e.g. because of a missing ``register_parameter`` call, an
attribute that was a plain tensor instead of an ``nn.Linear``, or a
``set_pretrain_mode`` flag that silently disabled the heads).

All tests are CPU-only — no GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.models.v4_multitask_dmpnn import (
    ATOM_VOCAB_SIZE,
    SUPPORTED_METALS,
    V4MultiTaskDMPNNPretrain,
    V4MultiTaskLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


SMILES_3 = ["CCO", "c1ccccc1", "CC(=O)O"]


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


def _set_pretrain_and_forward(m: V4MultiTaskDMPNNPretrain) -> dict:
    """Standard helper: enable pretrain mode and run a forward pass on
    SMILES_3. Returns the output dict."""
    m.train()
    m.set_pretrain_mode(True)
    metal_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    return m(SMILES_3, metal_ids)


# ---------------------------------------------------------------------------
# Test 1 — masked-atom-prediction backward
# ---------------------------------------------------------------------------
def test_pretrain_masked_atom_loss_backward():
    """``masked_logits.pow(2).mean().backward()`` succeeds and reaches
    both the pretrain_head AND the shared encoder."""
    m = _make_pretrain_model()
    out = _set_pretrain_and_forward(m)
    assert "atom_logits" in out, "atom_logits missing — pretrain mode not on"
    masked_logits = out["atom_logits"]

    # Sanity on the head output
    assert masked_logits.dim() == 2, (
        f"atom_logits must be 2-D (sum_N, vocab), got shape "
        f"{tuple(masked_logits.shape)}"
    )
    assert masked_logits.shape[-1] == ATOM_VOCAB_SIZE

    # The simplest possible backward — sum-of-squares proxy loss
    loss = masked_logits.pow(2).mean()
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    loss.backward()

    # pretrain_head (Linear) must have a non-None, non-zero gradient
    g_head = m.pretrain_head.weight.grad
    assert g_head is not None, "pretrain_head.weight.grad is None — head not in graph"
    assert g_head.abs().sum().item() > 0.0, (
        "pretrain_head.weight.grad is zero — backward did not flow"
    )

    # The shared encoder's atom embedding should ALSO have a gradient
    # (proves the head is hooked into the encoder's parameter graph).
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None, "Encoder atom_embed received no gradient"
    assert enc_grad.abs().sum().item() > 0.0, (
        "Encoder atom_embed gradient is zero — gradient not flowing from "
        "pretrain_head back into the shared encoder"
    )


# ---------------------------------------------------------------------------
# Test 2 — task-classifier backward
# ---------------------------------------------------------------------------
def test_pretrain_task_loss_backward():
    """``task_logits.pow(2).mean().backward()`` succeeds and reaches
    both the mol_clf_head AND the shared encoder."""
    m = _make_pretrain_model(n_pretrain_tasks=3)
    out = _set_pretrain_and_forward(m)
    assert "mol_clf" in out, "mol_clf missing — pretrain mode not on"
    task_logits = out["mol_clf"]

    # Sanity on the head output
    assert task_logits.shape == (3, m.n_pretrain_tasks), (
        f"mol_clf shape expected (3, {m.n_pretrain_tasks}), got "
        f"{tuple(task_logits.shape)}"
    )

    # Backward through the simplest possible loss
    loss = task_logits.pow(2).mean()
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    loss.backward()

    # mol_clf_head must have a non-None, non-zero gradient
    g_head = m.mol_clf_head.weight.grad
    assert g_head is not None, "mol_clf_head.weight.grad is None — head not in graph"
    assert g_head.abs().sum().item() > 0.0, (
        "mol_clf_head.weight.grad is zero — backward did not flow"
    )

    # The shared encoder should ALSO have a gradient from the task head.
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None, "Encoder atom_embed received no gradient"
    assert enc_grad.abs().sum().item() > 0.0, (
        "Encoder atom_embed gradient is zero — gradient not flowing from "
        "mol_clf_head back into the shared encoder"
    )


# ---------------------------------------------------------------------------
# Test 3 — combined loss backward through all heads
# ---------------------------------------------------------------------------
def test_pretrain_combined_loss_backward():
    """Combined base+pretrain loss flows through every head and the
    shared encoder. This is the realistic training step that Fix 4
    must support."""
    m = _make_pretrain_model()
    out = _set_pretrain_and_forward(m)
    loss_fn = V4MultiTaskLoss()

    _set_seed(1)
    # Base multi-task loss (5 metal CE + coord CE)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    base_loss, _ = loss_fn(out, y_true, coord_true)

    # Pretrain losses — masked atom + molecular task
    sum_N = int(out["atom_mask"].sum().item())
    atom_labels = torch.randint(0, ATOM_VOCAB_SIZE, (sum_N,))
    atom_loss = torch.nn.functional.cross_entropy(
        out["atom_logits"], atom_labels
    )
    mol_labels = torch.randint(0, m.n_pretrain_tasks, (3,))
    mol_loss = torch.nn.functional.cross_entropy(out["mol_clf"], mol_labels)

    total = base_loss + 0.1 * atom_loss + 0.1 * mol_loss
    assert torch.isfinite(total), f"Combined loss is non-finite: {total.item()}"
    total.backward()

    # Every pretrain head and every base head must have received gradient
    g_pretrain = m.pretrain_head.weight.grad
    g_molclf = m.mol_clf_head.weight.grad
    assert g_pretrain is not None and g_pretrain.abs().sum().item() > 0.0, (
        "pretrain_head received no gradient under combined loss"
    )
    assert g_molclf is not None and g_molclf.abs().sum().item() > 0.0, (
        "mol_clf_head received no gradient under combined loss"
    )

    # A metal head also still receives gradient (backward compat)
    g_metal = m.metal_heads["Ru"][0].weight.grad
    assert g_metal is not None and g_metal.abs().sum().item() > 0.0, (
        "metal_head['Ru'] received no gradient under combined loss"
    )

    # Coord head also receives gradient
    g_coord = m.coord_head[0].weight.grad
    assert g_coord is not None and g_coord.abs().sum().item() > 0.0, (
        "coord_head received no gradient under combined loss"
    )

    # The shared encoder's atom embedding must have gradient (param sharing)
    enc_grad = m.encoder.dmpnn.atom_embed[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0, (
        "Shared encoder atom_embed received no gradient — heads not hooked"
    )

    # Metal embedding must also have gradient (consumed by every head)
    g_embed = m.metal_embed.weight.grad
    assert g_embed is not None and g_embed.abs().sum().item() > 0.0, (
        "metal_embed received no gradient under combined loss"
    )


# ---------------------------------------------------------------------------
# Test 4 — gradients are non-None after backward
# ---------------------------------------------------------------------------
def test_pretrain_gradients_non_zero():
    """Every learnable parameter touched by the pretrain heads must end
    with a non-None gradient after ``.backward()`` is called. This is
    the strict gate Fix 4 introduces — silent gradient absence is the
    regression mode this test was added to catch."""
    m = _make_pretrain_model()
    out = _set_pretrain_and_forward(m)
    loss_fn = V4MultiTaskLoss()

    _set_seed(2)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    base_loss, _ = loss_fn(out, y_true, coord_true)
    sum_N = int(out["atom_mask"].sum().item())
    atom_labels = torch.randint(0, ATOM_VOCAB_SIZE, (sum_N,))
    atom_loss = torch.nn.functional.cross_entropy(
        out["atom_logits"], atom_labels
    )
    mol_labels = torch.randint(0, m.n_pretrain_tasks, (3,))
    mol_loss = torch.nn.functional.cross_entropy(out["mol_clf"], mol_labels)
    total = base_loss + 0.1 * atom_loss + 0.1 * mol_loss
    total.backward()

    # --- Pretrain heads: must NOT be detached from the computation graph ---
    assert m.pretrain_head.weight.grad is not None, (
        "pretrain_head.weight.grad is None — head not in autograd graph"
    )
    if m.pretrain_head.bias is not None:
        assert m.pretrain_head.bias.grad is not None, (
            "pretrain_head.bias.grad is None"
        )
    assert m.mol_clf_head.weight.grad is not None, (
        "mol_clf_head.weight.grad is None — head not in autograd graph"
    )
    if m.mol_clf_head.bias is not None:
        assert m.mol_clf_head.bias.grad is not None, (
            "mol_clf_head.bias.grad is None"
        )

    # --- Encoder bottom layer: must be in graph (param sharing works) ---
    assert m.encoder.dmpnn.atom_embed[0].weight.grad is not None, (
        "encoder.dmpnn.atom_embed[0].weight.grad is None"
    )

    # --- Metal embedding: must be in graph (consumed by every head) ---
    assert m.metal_embed.weight.grad is not None, (
        "metal_embed.weight.grad is None"
    )

    # --- At least one per-metal head: must be in graph ---
    for met in SUPPORTED_METALS:
        head_lin1 = m.metal_heads[met][0]
        assert head_lin1.weight.grad is not None, (
            f"metal_heads[{met!r}][0].weight.grad is None"
        )

    # --- Coord head: must be in graph ---
    assert m.coord_head[0].weight.grad is not None, (
        "coord_head[0].weight.grad is None"
    )

    # And as a final sanity check: NO learnable parameter that requires
    # grad is left without a gradient — catch detached tensors / detached
    # params / accidentally-frozen modules.
    ungrad = []
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is None:
            ungrad.append(name)
    assert not ungrad, (
        f"{len(ungrad)} parameters require_grad=True but have .grad=None — "
        f"detached subgraph. First few: {ungrad[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 5 — AdamW optimizer step actually mutates weights
# ---------------------------------------------------------------------------
def test_pretrain_optimizer_step():
    """A full optimizer step (zero_grad → forward → backward → step)
    must change every pretrain-head weight AND every shared-encoder
    weight. If a head's weight is unchanged after the step, the head
    wasn't picked up by the optimizer — the exact regression Fix 4
    exists to prevent."""
    _set_seed(0)
    m = _make_pretrain_model()

    # Pretrain-head weights BEFORE the step (detached clones for comparison)
    pretrain_head_w_before = m.pretrain_head.weight.detach().clone()
    pretrain_head_b_before = (
        m.pretrain_head.bias.detach().clone()
        if m.pretrain_head.bias is not None
        else None
    )
    mol_clf_head_w_before = m.mol_clf_head.weight.detach().clone()
    mol_clf_head_b_before = (
        m.mol_clf_head.bias.detach().clone()
        if m.mol_clf_head.bias is not None
        else None
    )
    enc_atom_embed_w_before = (
        m.encoder.dmpnn.atom_embed[0].weight.detach().clone()
    )

    # Build the optimizer over model.parameters() — this is the standard
    # call site, and any head that's NOT in ``m.parameters()`` will be
    # silently skipped.
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # Confirm the heads ARE in the parameters() iterator before stepping
    pretrain_head_param_ids = {id(p) for p in m.pretrain_head.parameters()}
    mol_clf_head_param_ids = {id(p) for p in m.mol_clf_head.parameters()}
    optim_param_ids = {id(p) for p in m.parameters()}
    assert pretrain_head_param_ids.issubset(optim_param_ids), (
        "pretrain_head params are NOT in model.parameters() — "
        "optimizer will silently skip them"
    )
    assert mol_clf_head_param_ids.issubset(optim_param_ids), (
        "mol_clf_head params are NOT in model.parameters() — "
        "optimizer will silently skip them"
    )

    optimizer.zero_grad()

    out = _set_pretrain_and_forward(m)
    loss_fn = V4MultiTaskLoss()

    _set_seed(3)
    y_true = {met: torch.randint(0, 2, (3,)) for met in SUPPORTED_METALS}
    coord_true = torch.randint(0, 5, (3,))
    base_loss, _ = loss_fn(out, y_true, coord_true)
    sum_N = int(out["atom_mask"].sum().item())
    atom_labels = torch.randint(0, ATOM_VOCAB_SIZE, (sum_N,))
    atom_loss = torch.nn.functional.cross_entropy(
        out["atom_logits"], atom_labels
    )
    mol_labels = torch.randint(0, m.n_pretrain_tasks, (3,))
    mol_loss = torch.nn.functional.cross_entropy(out["mol_clf"], mol_labels)
    total = base_loss + 0.1 * atom_loss + 0.1 * mol_loss
    total.backward()

    # Sanity: heads have a gradient before the step
    assert m.pretrain_head.weight.grad is not None
    assert m.pretrain_head.weight.grad.abs().sum().item() > 0.0
    assert m.mol_clf_head.weight.grad is not None
    assert m.mol_clf_head.weight.grad.abs().sum().item() > 0.0

    optimizer.step()

    # ----- Verify the weights changed -----
    # pretrain_head.weight
    delta_pretrain_head_w = (
        m.pretrain_head.weight.detach() - pretrain_head_w_before
    ).abs().sum().item()
    assert delta_pretrain_head_w > 0.0, (
        f"pretrain_head.weight unchanged after AdamW step "
        f"(delta={delta_pretrain_head_w}) — head not optimized"
    )

    # pretrain_head.bias (if present)
    if pretrain_head_b_before is not None:
        delta_pretrain_head_b = (
            m.pretrain_head.bias.detach() - pretrain_head_b_before
        ).abs().sum().item()
        assert delta_pretrain_head_b > 0.0, (
            f"pretrain_head.bias unchanged after AdamW step "
            f"(delta={delta_pretrain_head_b})"
        )

    # mol_clf_head.weight
    delta_mol_clf_head_w = (
        m.mol_clf_head.weight.detach() - mol_clf_head_w_before
    ).abs().sum().item()
    assert delta_mol_clf_head_w > 0.0, (
        f"mol_clf_head.weight unchanged after AdamW step "
        f"(delta={delta_mol_clf_head_w}) — head not optimized"
    )

    # mol_clf_head.bias (if present)
    if mol_clf_head_b_before is not None:
        delta_mol_clf_head_b = (
            m.mol_clf_head.bias.detach() - mol_clf_head_b_before
        ).abs().sum().item()
        assert delta_mol_clf_head_b > 0.0, (
            f"mol_clf_head.bias unchanged after AdamW step "
            f"(delta={delta_mol_clf_head_b})"
        )

    # encoder atom_embed (param sharing)
    delta_enc = (
        m.encoder.dmpnn.atom_embed[0].weight.detach()
        - enc_atom_embed_w_before
    ).abs().sum().item()
    assert delta_enc > 0.0, (
        f"encoder.dmpnn.atom_embed[0].weight unchanged after AdamW step "
        f"(delta={delta_enc}) — shared encoder not updated"
    )
