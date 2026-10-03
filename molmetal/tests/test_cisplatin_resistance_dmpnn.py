"""Tests for :mod:`molmetal.models.cisplatin_resistance_dmpnn`.

Phase A / Stage 2.3 — 3-cell-line multi-task D-MPNN with resistance
features.

Smoke tests are kept light (small batch, tiny ``hidden_dim``) so the
suite runs on CPU in <5 s.  The three tests cover the spec's three
acceptance criteria:

1. Forward-shape contract — output is ``dict[cell_line -> (B, 2)]``.
2. Focal loss is non-zero on a high-class-imbalance batch.
3. Per-cell routing — perturbing one head's parameters only affects
   that cell's output; the other cells' logits are bit-exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.cisplatin_resistance_dmpnn import (
    DEFAULT_CELL_LINES,
    DEFAULT_N_RESIST_FEAT,
    CisplatinResistanceDMPNN,
    cisplatin_resistance_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_model(
    hidden_dim: int = 32,
    n_resist_feat: int = DEFAULT_N_RESIST_FEAT,
    seed: int = 0,
) -> CisplatinResistanceDMPNN:
    """Build a small model suitable for CPU smoke tests."""
    torch.manual_seed(seed)
    return CisplatinResistanceDMPNN(
        hidden_dim=hidden_dim,
        n_layers=1,
        n_cells=3,
        n_resist_feat=n_resist_feat,
    )


_TOY_SMILES = [
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CC(C)O",
    "CCC",
    "c1ccncc1",
    "CCS",
    "CC(=O)N",
    "C1CCCCC1",
]
_TOY_CELL_IDS = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def _toy_batch(B: int = 4):
    """Return a tiny batch: SMILES + cell ids + resistance features.

    ``_TOY_SMILES`` / ``_TOY_CELL_IDS`` are padded up to 10 rows so a
    caller can request any ``B in [1, 10]``; B > 10 cycles the bank.
    """
    if B > len(_TOY_SMILES):
        smi = [_TOY_SMILES[i % len(_TOY_SMILES)] for i in range(B)]
        cid = [_TOY_CELL_IDS[i % len(_TOY_CELL_IDS)] for i in range(B)]
    else:
        smi = _TOY_SMILES[:B]
        cid = _TOY_CELL_IDS[:B]
    return (
        smi,
        torch.tensor(cid, dtype=torch.long),
        torch.randn(B, DEFAULT_N_RESIST_FEAT),
    )


# ---------------------------------------------------------------------------
# 1. Forward shape
# ---------------------------------------------------------------------------
def test_forward_shape():
    """Output is ``dict[cell_line -> (B, 2)]`` with the three spec keys."""
    model = _make_model()
    smiles, cell_ids, resist = _toy_batch(B=4)
    model.eval()
    with torch.no_grad():
        out = model(smiles, cell_ids, resist)

    # Key set exactly matches the default cell-line roster.
    assert set(out.keys()) == set(
        DEFAULT_CELL_LINES
    ), f"got keys {set(out.keys())}, expected {set(DEFAULT_CELL_LINES)}"
    # Each value is (B, 2) float32 logits.
    for cl, lg in out.items():
        assert lg.shape == (4, 2), f"{cl}: got {tuple(lg.shape)}"
        assert lg.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. Focal loss on a high-class-imbalance batch
# ---------------------------------------------------------------------------
def test_focal_loss_balance():
    """8 negatives + 2 positives -> loss is finite, positive, and non-zero.

    Focal loss with auto-balancing (``alpha < 0``) must produce a
    non-trivial gradient signal even when one class dominates.  A
    *zero* loss here would mean the model is collapsing to the prior
    (e.g. all-zeros output) — exactly the failure mode the focal loss
    is meant to prevent.
    """
    model = _make_model()
    smiles, cell_ids, resist = _toy_batch(B=10)
    model.eval()
    with torch.no_grad():
        out = model(smiles, cell_ids, resist)

    # High class imbalance: 8 negatives, 2 positives per cell line.
    targets = {
        "MCF-7": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
        "A549": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
        "A2780cis": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
    }
    loss = cisplatin_resistance_loss(out, targets)
    assert torch.isfinite(loss), f"loss is non-finite: {loss.item()}"
    assert loss.item() != 0.0, "focal loss collapsed to zero on imbalanced batch"
    # Sanity-check: focal(p, y) > 0 for any p in (0, 1), y in {0, 1}.
    assert loss.item() > 0.0, f"loss should be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# 3. Per-cell routing
# ---------------------------------------------------------------------------
def test_cell_routing():
    """Perturbing one cell's head affects only that cell's logits.

    The cell embedding is shared across all rows of a batch, and each
    cell line has its **own** head.  Two properties are checked:

    (a) different ``cell_ids`` for the same SMILES produce different
        logits (because the cell embedding changes ``z``);
    (b) zeroing one head's final linear layer leaves the *other*
        cells' logits bit-exact (no parameter sharing between heads).
    """
    model = _make_model()
    smiles = ["CCO", "CCN"]  # B=2
    resist = torch.randn(2, DEFAULT_N_RESIST_FEAT)
    model.eval()

    with torch.no_grad():
        out_mcf7 = model(smiles, torch.tensor([0, 0]), resist)
        out_a549 = model(smiles, torch.tensor([1, 1]), resist)

    # (a) Same molecule routed through MCF-7 vs A549 should differ —
    #     the cell-embedding path (32 dims) feeds every head.
    for sm_i in range(2):
        mcf7 = out_mcf7["MCF-7"][sm_i]
        a549 = out_a549["MCF-7"][sm_i]  # same head, different cell embed
        assert not torch.allclose(
            mcf7, a549, atol=1e-6
        ), "cell embedding did not influence the MCF-7 head"

    # (b) Zero one head's final Linear; the other heads must be unaffected.
    cell_ids = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        out_before = model(smiles, cell_ids, resist)
        # Zero the A549 head's last linear layer.
        last_linear = model.cell_heads["A549"][-1]  # nn.Linear(128, 2)
        last_linear.weight.zero_()
        last_linear.bias.zero_()
        out_after = model(smiles, cell_ids, resist)

    # MCF-7 + A2780cis unchanged.
    for cl in ("MCF-7", "A2780cis"):
        assert torch.allclose(
            out_before[cl], out_after[cl], atol=1e-6
        ), f"{cl} head was perturbed by zeroing A549's head"
    # A549 changed.
    assert not torch.allclose(
        out_before["A549"], out_after["A549"], atol=1e-6
    ), "zeroing A549's head did not change A549's logits"
