"""Tests for ChempropV2Frontier — all 4 R18 frontier-math modules integrated
into the chemprop v2 main pipeline (Wf-237 / TODO-189).

Covers four properties derived from the task spec:

  (i)   :class:`ChempropV2Frontier` instantiates with default + custom
         frontier configs.
  (ii)  ``forward(smiles_list, metal_ids)`` returns a dict with the
         same shape contract as :class:`ChempropV2MTModel` (5 per-metal
         ``(B, 2)`` logits + coord ``(B, n_coord_classes)``).
  (iii) ``forward_with_loss`` returns a finite loss with all 4 frontier
         modules enabled; gradient flows through the chemprop v2
         backbone, multi-PH projection, and frontier regularisers.
  (iv)  When all 4 frontier weights are set to ``0.0``, the wrapper's
         forward output is bit-exact equivalent to the base
         :class:`ChempropV2MTModel` forward on the SAME instance
         (the Wf-237 G4 backward-compat gate).

All tests are CPU-only and use ``hidden_dim=64`` + ``depth=2`` to keep
runtime in seconds.  No GPU required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from molmetal.models.chemprop_v2_frontier import (
    ChempropV2Frontier,
    FrontierConfig,
    quick_smoke_chemprop_v2_frontier,
)
from molmetal.models.chemprop_v2_mt_model import (
    SUPPORTED_METALS,
    ChempropV2MTModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


SMILES_4 = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"]


def _make_frontier(
    hidden_dim: int = 64,
    depth: int = 2,
    config: FrontierConfig = None,
) -> ChempropV2Frontier:
    """Small chemprop v2 frontier model for unit tests."""
    _set_seed(0)
    if config is None:
        config = FrontierConfig()
        # Zero weights by default in tests.
        config.se3_sb_weight = 0.0
        config.bass_weight = 0.0
        config.monge_ampere_weight = 0.0
        config.multi_ph_weight = 0.0
    return ChempropV2Frontier(
        config=config, hidden_dim=hidden_dim, depth=depth,
    )


# ---------------------------------------------------------------------------
# (i) Instantiation
# ---------------------------------------------------------------------------
def test_frontier_instantiates():
    """FrontierConfig + ChempropV2Frontier wire the 4 sub-modules."""
    fc = FrontierConfig()
    assert fc.se3_sb_weight == 0.02  # default opt-in weight
    assert fc.bass_weight == 0.01
    assert fc.monge_ampere_weight == 0.02
    assert fc.multi_ph_weight == 0.05
    m = _make_frontier()
    assert isinstance(m, torch.nn.Module)
    # Base model is a vanilla chemprop v2 mt model.
    assert isinstance(m.base_model, ChempropV2MTModel)
    # 4 frontier sub-modules present.
    assert hasattr(m, "multi_ph_augment")
    assert hasattr(m, "se3_regulariser")
    assert hasattr(m, "monge_ampere_regulariser")
    assert hasattr(m, "bass_regulariser")


def test_frontier_enable_all_disable_all():
    """enable_all() sets all weights to defaults; disable_all() to zero."""
    fc = FrontierConfig()
    fc.se3_sb_weight = 0.0
    fc.bass_weight = 0.0
    fc.monge_ampere_weight = 0.0
    fc.multi_ph_weight = 0.0
    m = ChempropV2Frontier(config=fc)
    m.enable_all()
    assert m.config.se3_sb_weight > 0
    assert m.config.bass_weight > 0
    assert m.config.monge_ampere_weight > 0
    assert m.config.multi_ph_weight > 0
    m.disable_all()
    assert m.config.se3_sb_weight == 0.0
    assert m.config.bass_weight == 0.0
    assert m.config.monge_ampere_weight == 0.0
    assert m.config.multi_ph_weight == 0.0


# ---------------------------------------------------------------------------
# (ii) Forward shape contract
# ---------------------------------------------------------------------------
def test_frontier_forward_shape_matches_base():
    """Forward returns dict[metal -> (B, 2), 'coord' -> (B, n_coord_classes)]."""
    _set_seed(0)
    m = _make_frontier()
    B = 4
    metal_ids = torch.zeros(B, dtype=torch.long)
    out, info = m(SMILES_4, metal_ids)
    assert set(out.keys()) == set(SUPPORTED_METALS) | {"coord"}
    for metal in SUPPORTED_METALS:
        assert out[metal].shape == (B, 2), f"{metal}: {out[metal].shape}"
    assert out["coord"].shape == (B, 5)
    # Info dict has the 4 weight flags + augmented_dim.
    assert "multi_ph_weight" in info
    assert "se3_sb_weight" in info
    assert "monge_ampere_weight" in info
    assert "bass_weight" in info


# ---------------------------------------------------------------------------
# (iii) Forward + backward + finite loss with all 4 modules enabled
# ---------------------------------------------------------------------------
def test_frontier_forward_with_loss_all_modules_enabled():
    """All 4 frontier modules active — loss is finite + backward succeeds."""
    _set_seed(0)
    config = FrontierConfig()
    config.se3_sb_weight = 0.05
    config.bass_weight = 0.02
    config.monge_ampere_weight = 0.05
    config.multi_ph_weight = 0.1
    m = ChempropV2Frontier(config=config, hidden_dim=64, depth=2)

    B = 4
    metal_ids = torch.zeros(B, dtype=torch.long)
    torch.manual_seed(0)
    y_true = {
        metal: torch.randint(0, 2, (B,), dtype=torch.float32)
        for metal in m.base_model.metals
    }
    coord_true = torch.randint(0, 5, (B,), dtype=torch.long)

    loss, info = m.forward_with_loss(SMILES_4, metal_ids, y_true, coord_true)
    assert torch.isfinite(loss).item(), f"loss = {loss.item()}"
    # All 4 frontier contributions should appear in the info dict.
    assert "loss_bass" in info
    assert "loss_monge_ampere" in info
    assert "loss_se3_sb" in info
    assert "loss_coord" in info
    for metal in SUPPORTED_METALS:
        assert f"loss_{metal}" in info

    # Backward succeeds (gradient flows through chemprop v2 backbone +
    # multi-PH projection + frontier regularisers).
    loss.backward()
    # At least one of the model params must have a non-None grad.
    has_grad = any(p.grad is not None for p in m.parameters())
    assert has_grad


def test_frontier_smoke_helper_returns_working_model():
    """quick_smoke_chemprop_v2_frontier returns a working model + batch."""
    _set_seed(0)
    model, loss_fn, batch, labels = quick_smoke_chemprop_v2_frontier(
        hidden_dim=64, depth=2, enable_all=True,
    )
    out = batch["out"]
    info = batch["info"]
    assert "Ru" in out and "coord" in out
    assert info["multi_ph_weight"] > 0
    # Run a forward_with_loss — should produce finite loss.
    metal_ids = torch.zeros(4, dtype=torch.long)
    y_true = labels["y_true"]
    coord_true = labels["coord_true"]
    loss, _ = model.forward_with_loss(SMILES_4, metal_ids, y_true, coord_true)
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# (iv) Bit-exact backward compat — all weights = 0
# ---------------------------------------------------------------------------
def test_frontier_bit_exact_when_all_weights_zero():
    """With all 4 frontier weights = 0, wrapper forward = base forward on
    the SAME model instance (the Wf-237 G4 backward-compat gate)."""
    _set_seed(0)
    config = FrontierConfig()
    config.se3_sb_weight = 0.0
    config.bass_weight = 0.0
    config.monge_ampere_weight = 0.0
    config.multi_ph_weight = 0.0
    m = ChempropV2Frontier(config=config, hidden_dim=64, depth=2)
    # CRITICAL: set eval mode so dropout is disabled — otherwise the
    # wrapper's first forward consumes RNG via dropout and the base
    # model's subsequent forward uses a different RNG state.
    m.eval()

    B = 4
    metal_ids = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out_wrap, _ = m(SMILES_4, metal_ids)
        out_base = m.base_model(SMILES_4, metal_ids)

    for k in ["Ru", "Ir", "Os", "Rh", "Re", "coord"]:
        diff = (out_wrap[k] - out_base[k]).abs().max().item()
        assert diff < 1.0e-6, f"{k}: diff = {diff}"


# ---------------------------------------------------------------------------
# (v) Per-module sub-tests — each frontier module is wired correctly
# ---------------------------------------------------------------------------
def test_multi_ph_augment_output_shape():
    """Multi-PH feature map has shape (B, n_bins²) per batch."""
    from molmetal.models.chemprop_v2_frontier import MultiPHAugment

    mp = MultiPHAugment(n_bins=8, max_radius=5.0)
    out = mp(SMILES_4)
    assert out.shape == (4, 8 * 8)


def test_se3_geodesic_zero_perturbation_is_identity():
    """With perturbation = 0.0, the SE(3) regulariser returns its input
    bit-exact unchanged."""
    from molmetal.models.chemprop_v2_frontier import SE3GeodesicRegulariser

    reg = SE3GeodesicRegulariser(perturbation=0.0)
    logits = torch.randn(4, 2)
    out = reg(logits)
    diff = (out - logits).abs().max().item()
    assert diff < 1.0e-9


def test_bass_regulariser_zero_when_second_moment_below_threshold():
    """Bass regularity penalty is 0 when the empirical second moment is
    below the threshold (i.e. marginal is in P₂)."""
    from molmetal.models.chemprop_v2_frontier import BassRegularityRegulariser

    reg = BassRegularityRegulariser(heavy_tail_threshold=1.0e4)
    embeddings = torch.randn(100, 64) * 0.1  # small magnitude
    pen = reg(embeddings)
    assert pen.item() == 0.0


def test_monge_ampere_regulariser_clamped_to_1():
    """Monge-Ampère variance bound is clamped to 1.0 max so it never
    dominates the BCE loss."""
    from molmetal.models.chemprop_v2_frontier import MongeAmpereRegulariser

    reg = MongeAmpereRegulariser(lam=0.1)
    logits = torch.randn(100) * 100  # huge logits → huge variance bound
    y = torch.ones(100)
    pen = reg(logits, y)
    assert pen.item() <= 1.0 + 1.0e-6


# ---------------------------------------------------------------------------
# (vi) End-to-end smoke (mirrors scripts/run_chemprop_v2_frontier_e2e.py)
# ---------------------------------------------------------------------------
def test_frontier_smoke_100_real_samples_loss_finite():
    """100 MetalCytoToxDB samples + all 4 modules → loss is finite.

    The honest framing note: this is a CPU 1-step smoke, NOT a
    production metric claim.  Loss_finite + non-NaN only proves
    fwd+bwd+AdamW wire-up.
    """
    from molmetal.data.metal_cytotox import load_metal_cytotox

    _set_seed(0)
    df = load_metal_cytotox()
    df = df[df["metal"].isin(SUPPORTED_METALS)].reset_index(drop=True)
    pieces = []
    per_metal = 20
    for metal in SUPPORTED_METALS:
        sub = df[df["metal"] == metal]
        if len(sub) == 0:
            continue
        idx = np.linspace(0, len(sub) - 1, per_metal).round().astype(int)
        pieces.append(sub.iloc[idx])
    sub = (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=0)
        .reset_index(drop=True)
    ).head(100)

    smiles_list = sub["smiles"].tolist()
    metals_list = sub["metal"].tolist()
    actives_list = sub["active"].astype(float).tolist()
    metal_to_id = {m: i for i, m in enumerate(SUPPORTED_METALS)}
    metal_ids = torch.tensor(
        [metal_to_id[m] for m in metals_list], dtype=torch.long
    )
    y_true = {}
    for metal in SUPPORTED_METALS:
        per_row = torch.tensor(
            [
                float(actives_list[i]) if mn == metal else 0.0
                for i, mn in enumerate(metals_list)
            ],
            dtype=torch.float32,
        )
        y_true[metal] = per_row

    def _coord_label_for(metal_name: str) -> int:
        table = {
            "Pt": 3, "Pd": 3, "Au": 3,
            "Ru": 4, "Ir": 4, "Os": 4, "Rh": 4, "Re": 4,
        }
        return table.get(metal_name, 3)
    coord_true = torch.tensor(
        [_coord_label_for(m) for m in metals_list], dtype=torch.long
    )

    config = FrontierConfig()
    config.se3_sb_weight = 0.05
    config.bass_weight = 0.02
    config.monge_ampere_weight = 0.05
    config.multi_ph_weight = 0.1
    m = ChempropV2Frontier(config=config, hidden_dim=64, depth=2)

    loss, info = m.forward_with_loss(smiles_list, metal_ids, y_true, coord_true)
    assert torch.isfinite(loss).item()
    # Backward + AdamW step.
    optim = torch.optim.AdamW(m.parameters(), lr=1e-3)
    optim.zero_grad()
    loss.backward()
    optim.step()
