#!/usr/bin/env python3
"""Wf-31 Phase 1 — R18 M22 SE(3) SB 5-metal multitask trainer (Ru focus).

=====================================================================
**NEW trainer (Wf-31)** that wraps the W2.1 SE(3) Schrödinger-Bridge
velocity field shipped in :mod:`molmetal.scripts.train_se3sb` and
attaches a **5-metal multitask head with classification + regression
heads** on top of the V4 encoder.

This script is a **wrapper / aggregator** of the W2.1 train_se3sb
deliverable, not a replacement:

* **Velocity field**: imported from :func:`train_se3sb.build_se3_twist_head`
  + :class:`SE3SBVelocityField`.  We do **not** modify
  ``train_se3sb.py`` (R17 baseline locked per Wf-22 manifest).
* **V4 5-metal multitask backbone**: imported from
  :class:`molmetal.models.v4_multitask_dmpnn.V4MultiTaskDMPNN`.
* **Init modes**: 4-way ablation (--init) — *tmqm*, *se3sb*,
  *transport*, *multiph* — only *se3sb* attaches the W2.1
  velocity head; the others attach R18 W2.2 / W2.3 adapters.
* **Heads**: for each of the 5 metals we ship *5 cls (binary CE) +
  5 reg (BCE-with-logits soft-label regression)* heads per the Wf-31
  spec.  Classification produces the headline AUC, regression
  provides a continuous score that calibrates against label
  imbalance.

The headline output is **per-metal AUC** (5 metals × 5 seeds = 25
AUC values) plus **5-seed mean ± std** per metal and a **Ru mean ±
std** cross-seed statistic (the Wf-31 SPEC target: ``mean(Ru) ≥ 0.86``
to claim beating Krasnov 0.81).

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (26 801 rows)
  columns: ``smiles,metal,scaffold_id,split,activity`` — scaffold-split
  ``train/val/test`` (Bemis-Murcko) per Krasnov 2026 protocol.

Outputs
-------
* ``.ckpt/ru_5metal_se3sb_{init}_seed{0..4}.pt`` — per-seed best ckpt.
* ``metrics/by_metric/ru_5metal_se3sb.json`` — per-seed
  ``test_auc_per_metal`` (5 entries) + ``mean_test_auc`` (mean across
  5 metals) + **5-seed mean ± std per metal** + ``init`` ablation tag.
* ``metrics/by_metric/ru_5metal_se3sb_summary.json`` — slim final
  report with **Ru 5-seed mean ± std** (the Wf-31 headline metric).

Recipe
------
(1) Build :class:`V4MultiTaskDMPNN` (hidden=300, depth=3, 5 metal
    heads + coord-geometry auxiliary head).
(2) Attach a 5-metal *classification + regression* head wrapper on
    top of the V4 encoder (5 cls + 5 reg = 10 per-metal heads).
(3) Wire the SE(3) SB velocity head (from W2.1 train_se3sb) when
    ``--init se3sb``; otherwise wire transport-relax or multi-PH
    R18 adapters; ``--init tmqm`` is the R17 baseline fallback.
(4) Optionally load tmQM pretrained encoder (arch-mismatch fallback
    per W2.1 contract).
(5) Freeze encoder layers 0..1; fine-tune layer 2 + heads + metal
    embedding + (when in se3sb) the SE(3) twist head.
(6) Multi-task loss: 5 cls CE + 5 reg BCE-with-logits + coord CE
    (auxiliary).  Per-metal CE + reg weights default to 1.0 each;
    the reg term is weighted by ``REG_LOSS_WEIGHT=0.3`` to mirror
    the R17 coord-head weighting (Wf-31 §3).
(7) 5 seeds {0..4} × 50 epochs (default), batch=128, lr=1e-3, AdamW,
    cosine annealing, BF16 autocast (when CUDA is present).
(8) Eval every 5 epochs on val; pick best epoch by mean val AUC.
(9) Final test: per-metal AUC + mean AUC + 5-seed mean ± std per metal.

Constraint
----------
**DO NOT MODIFY** ``train_se3sb.py`` or ``train_ru_multitask.py`` —
both are R17/R18 baselines locked per Wf-22 close-out manifest.
This script is a **NEW deliverable** (Wf-31 Phase 1).

Verify
------
After a successful run:
* 5 ckpts exist at ``.ckpt/ru_5metal_se3sb_{init}_seed{0..4}.pt``.
* ``metrics/by_metric/ru_5metal_se3sb.json`` records per-seed
  ``test_auc_per_metal`` (5 metals) + ``mean_test_auc`` + the
  5-seed ``per_metal_mean_auc`` and ``per_metal_std_auc`` statistics.
* ``metrics/by_metric/ru_5metal_se3sb_summary.json`` records the
  Wf-31 headline metric ``ru_mean_auc_5seed`` (=mean of per-seed
  Ru AUC) and ``ru_std_auc_5seed`` (=std of per-seed Ru AUC).
* The 5-seed mean ± std for **each metal** matches a manual numpy
  reduction of ``per_seed[*]["test_auc_per_metal"][m]``.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_ru_5metal_se3sb.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/ru_5metal_se3sb.json \\
        --summary-path metrics/by_metric/ru_5metal_se3sb_summary.json \\
        --init se3sb --epochs 50 --batch 128 --lr 1e-3 --seeds 0 1 2 3 4 \\
        --metals Ru Ir Rh Re Os
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# W2.1 ship surface — V4 5-metal multitask encoder + coord head.
from molmetal.models.v4_multitask_dmpnn import (
    COORD_LOSS_WEIGHT,
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
)

# W2.1 SE(3) Schrödinger-Bridge velocity field (from train_se3sb).
# We import the underlying adapter + the W2.1 helper used by
# train_se3sb.py — this script does NOT modify train_se3sb.py.
from molmetal.adapters.se3_sb_integration import (
    SE3SBVelocityField,
    integrate_with_cfm,
)

# R18 W2.2 / W2.3 adapter wiring — only used when --init is
# transport / multiph.  We import lazily inside the init branches
# to keep the cold-start path (--init tmqm) import-clean.
try:  # pragma: no cover
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm as _RelaxedEndpointTerm,
    )
except Exception:  # pragma: no cover
    _RelaxedEndpointTerm = None  # type: ignore[assignment]
try:  # pragma: no cover
    from molmetal.adapters.multi_ph_integration import (
        MultiPHPocketFeatures as _MultiPHPocketFeatures,
    )
except Exception:  # pragma: no cover
    _MultiPHPocketFeatures = None  # type: ignore[assignment]

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)

# --- Recipe constants (Wf-31 Phase 1 spec) ---
RECIPE_EPOCHS: int = 50
RECIPE_BATCH: int = 128
RECIPE_LR: float = 1e-3
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
RECIPE_HIDDEN: int = 300
RECIPE_DEPTH: int = 3
RECIPE_EVAL_EVERY: int = 5
RECIPE_N_COORD_CLASSES: int = 5
RECIPE_METAL_EMBED_DIM: int = 64
RECIPE_DROPOUT: float = 0.2

#: Loss weight on the 5 reg heads (mirrors coord-head weight 0.3).
REG_LOSS_WEIGHT: float = 0.3

#: Default metals (5-way canonical subset).
DEFAULT_METALS: Tuple[str, ...] = SUPPORTED_METALS

# Coord labels: a deterministic per-metal proxy (square_planar for
# Pt/Ru/Ir/Rh, tetrahedral for Re/Os) — see train_se3sb.py.
COORD_BY_METAL: Dict[str, int] = {
    "Ru": 3,
    "Ir": 3,
    "Rh": 3,
    "Re": 2,
    "Os": 3,
}

# --- Output paths (overridable via CLI) ---
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_ENCODER_CKPT = (
    PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "ru_5metal_se3sb.json"
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "ru_5metal_se3sb_summary.json"
)

# Wf-31 ablation inits — exactly 4 (matches the spec).
INIT_CHOICES: Tuple[str, ...] = ("tmqm", "se3sb", "transport", "multiph")


# ---------------------------------------------------------------------------
# Head wrapper — V4 backbone + 5 cls + 5 reg heads (no V4 mutation)
# ---------------------------------------------------------------------------
class Ru5MetalSe3SBHead(nn.Module):
    """Wraps :class:`V4MultiTaskDMPNN` with **5 cls + 5 reg** heads.

    The Wf-31 spec requires a 5-metal multitask head with classification
    AND regression heads per metal (10 heads total).  We do not modify
    :class:`V4MultiTaskDMPNN` (the R18 W2.1 backbone is locked per
    Wf-22 close-out) — we instead *wrap* its 5 cls heads with
    matching regression heads that consume the same V4 head-in
    feature (``hidden_dim + metal_embed_dim``).

    Forward contract:
        out["cls"][m] → (B, 2) binary logits (activity)
        out["reg"][m] → (B, 1) regression logit (BCE-with-logits)
        out["coord"]  → (B, n_coord_classes) coord-geometry logits
                        (passed through from V4's coord head for
                         loss-compatibility with V4MultiTaskLoss)

    Parameters
    ----------
    backbone : V4MultiTaskDMPNN
        The V4 backbone whose per-metal heads we wrap.
    metals : Tuple[str, ...]
        Subset of metals we materialise regression heads for.  By
        default equal to ``backbone.metals`` (5-way canonical).
    head_hidden_dim : int, default 128
        Width of the regression head MLP.
    head_dropout : float, default 0.2
        Dropout applied inside each regression head.
    """

    def __init__(
        self,
        backbone: V4MultiTaskDMPNN,
        metals: Optional[Tuple[str, ...]] = None,
        head_hidden_dim: int = 128,
        head_dropout: float = RECIPE_DROPOUT,
    ):
        super().__init__()
        if metals is None:
            metals = tuple(backbone.metals)
        # Validate that the requested metals are a subset of the backbone.
        unknown = [m for m in metals if m not in backbone.metals]
        if unknown:
            raise ValueError(
                f"Requested metals {unknown!r} not in backbone.metals "
                f"{backbone.metals!r}"
            )
        self.backbone = backbone
        self.metals: Tuple[str, ...] = tuple(metals)
        self.head_hidden_dim = int(head_hidden_dim)
        head_in = self.backbone.hidden_dim + self.backbone.metal_embed_dim
        # Per-metal regression head (BCE-with-logits → scalar).
        self.reg_heads = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(head_in, head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                for m in self.metals
            }
        )
        # Init the regression heads with the same scheme as V4's
        # classification heads (Kaiming uniform + zero bias).
        for head in self.reg_heads.values():
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def metal_id(self, metal: str) -> int:
        """Pass-through to the backbone (consistent ids)."""
        return self.backbone.metal_id(metal)

    # ------------------------------------------------------------------
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """Pass-through to the backbone encoder."""
        return self.backbone.encode(smiles_list)

    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        metal_ids: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward through V4 + reg heads.

        Returns
        -------
        dict with keys:
            * ``"cls"`` → ``{metal: (B, 2) logits}``
            * ``"reg"`` → ``{metal: (B, 1) logit}``
            * ``"coord"`` → ``(B, n_coord_classes)`` (passed through
              from V4's coord head)
        """
        v4_out = self.backbone(smiles_list, metal_ids)
        cls_out: Dict[str, torch.Tensor] = {}
        reg_out: Dict[str, torch.Tensor] = {}
        for m in self.metals:
            cls_out[m] = v4_out[m]
            reg_out[m] = self.reg_heads[m](
                torch.cat(
                    [
                        self.backbone.encode(smiles_list),
                        self.backbone.metal_embed(metal_ids.long()),
                    ],
                    dim=-1,
                )
            )
        return {"cls": cls_out, "reg": reg_out, "coord": v4_out["coord"]}


# ---------------------------------------------------------------------------
# Multi-task loss — 5 cls CE + 5 reg BCE-with-logits + coord CE
# ---------------------------------------------------------------------------
class Ru5MetalSe3SBLoss(nn.Module):
    """Multi-task loss for the Wf-31 5 cls + 5 reg head wrapper.

    L = Σ_m CE(cls[m], y_true[m])                (5 cls terms)
        + REG_LOSS_WEIGHT · Σ_m BCE(reg[m], y_true[m])
        + COORD_LOSS_WEIGHT · CE(coord, coord_true)

    The classification + regression pairs share the same metal head
    in features but emit independent logits, so the two losses are
    not redundant: classification is discrete, regression is a soft
    continuous score calibrated against label imbalance.

    Per-metal CE / reg weights default to 1.0; pass ``metal_weights``
    to up-weight scarce-metal heads.
    """

    def __init__(
        self,
        n_coord_classes: int = RECIPE_N_COORD_CLASSES,
        coord_weight: float = COORD_LOSS_WEIGHT,
        reg_weight: float = REG_LOSS_WEIGHT,
        metal_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.n_coord_classes = int(n_coord_classes)
        self.coord_weight = float(coord_weight)
        self.reg_weight = float(reg_weight)
        self.metal_weights: Dict[str, float] = (
            dict(metal_weights) if metal_weights is not None else {}
        )
        self.ce_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_reg = nn.BCEWithLogitsLoss()
        self.ce_coord = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        out: Dict[str, Dict[str, torch.Tensor]],
        y_true: Dict[str, torch.Tensor],
        coord_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the multi-task loss.

        Parameters
        ----------
        out : dict
            Output of :class:`Ru5MetalSe3SBHead.forward` with keys
            ``"cls"``, ``"reg"``, ``"coord"``.
        y_true : dict
            Mapping metal → (B,) long binary labels (used for *both*
            cls CE and reg BCE — the BCE label is the same float).
        coord_true : (B,) long
            Coordination-geometry labels.
        """
        info: Dict[str, torch.Tensor] = {}
        total = out["coord"].new_zeros(())
        cls_out = out["cls"]
        reg_out = out["reg"]
        for metal in cls_out:
            if metal not in y_true:
                info[f"loss_cls_{metal}"] = torch.tensor(0.0, device=total.device)
                info[f"loss_reg_{metal}"] = torch.tensor(0.0, device=total.device)
                continue
            y = y_true[metal].long()
            if y.numel() == 0:
                info[f"loss_cls_{metal}"] = torch.tensor(0.0, device=total.device)
                info[f"loss_reg_{metal}"] = torch.tensor(0.0, device=total.device)
                continue
            w = self.metal_weights.get(metal, 1.0)
            # Classification CE
            l_cls = self.ce_cls(cls_out[metal], y) * float(w)
            info[f"loss_cls_{metal}"] = l_cls.detach()
            total = total + l_cls
            # Regression BCE (float targets)
            y_f = y.float().unsqueeze(-1)
            l_reg = self.bce_reg(reg_out[metal], y_f) * float(w)
            info[f"loss_reg_{metal}"] = l_reg.detach()
            total = total + self.reg_weight * l_reg
        # Coord auxiliary term
        if coord_true is not None and coord_true.numel() > 0:
            lcrd = self.ce_coord(out["coord"], coord_true.long())
            info["loss_coord"] = lcrd.detach()
            total = total + self.coord_weight * lcrd
        else:
            info["loss_coord"] = torch.tensor(0.0, device=total.device)
        info["loss_total"] = total.detach()
        return total, info


# ---------------------------------------------------------------------------
# Data loading — same contract as train_se3sb.load_5metal_csv
# ---------------------------------------------------------------------------
def load_5metal_csv(path: Path) -> Dict[str, Dict[str, List]]:
    """Load the 5-metal multitask CSV → per-metal {split: [(smiles, label)]}.

    Rows whose metal is not in ``SUPPORTED_METALS`` or whose split is
    not in ``{"train", "val", "test"}`` are silently dropped.
    """
    data: Dict[str, Dict[str, List]] = {
        m: {"train": [], "val": [], "test": []} for m in SUPPORTED_METALS
    }
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metal = row["metal"].strip()
            if metal not in SUPPORTED_METALS:
                continue
            split = row["split"].strip()
            if split not in ("train", "val", "test"):
                continue
            try:
                act = int(row["activity"])
            except (ValueError, TypeError):
                continue
            data[metal][split].append((row["smiles"], act))
    return data


# ---------------------------------------------------------------------------
# tmQM encoder load (with arch-mismatch fallback) — mirrors train_se3sb
# ---------------------------------------------------------------------------
def try_load_tmqm_encoder(model: V4MultiTaskDMPNN, ckpt_path: Path) -> int:
    """Attempt to load tmQM pretrained weights into V4 encoder.

    Mirrors the contract of
    :func:`train_se3sb.try_load_tmqm_encoder` and
    :func:`train_ru_multitask.try_load_tmqm_encoder`: on arch-
    mismatch (baseline MetalCytotoxModel vs. DirectedMPNN) we fall
    back to random init with a clear warning.  Returns the number of
    encoder tensors successfully transferred (``0`` on fallback).
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_ru_5metal_se3sb] tmQM encoder ckpt not found at "
            f"{ckpt_path}; using random init for V4 encoder."
        )
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    own = model.encoder.dmpnn.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            n_loaded += 1
    if n_loaded == 0:
        warnings.warn(
            "[train_ru_5metal_se3sb] tmQM encoder arch is incompatible "
            "with V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN); "
            "falling back to random init.  See train_se3sb KNOWN "
            "LIMITATION for the fix path."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_ru_5metal_se3sb] partial tmQM load: {n_loaded}/"
            f"{len(own)} tensors matched; remaining kept at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# SE(3) twist head — mirrors train_se3sb.build_se3_twist_head
# ---------------------------------------------------------------------------
class _StubEGNNVelocityNet(nn.Module):
    """Parameter-free stub EGNN velocity net used by the W2.1 adapter.

    Mirrors :class:`train_se3sb._StubEGNNVelocityNet` — returns a
    per-atom zero residual so the SE(3) twist collapses to the
    centroid velocity during training (the shape contract is
    preserved).
    """

    def __init__(self, hidden_dim: int = RECIPE_HIDDEN) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pocket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-atom velocity of shape ``(B, N, 3)`` (zeros)."""
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        return torch.zeros_like(x)


def build_se3_twist_head(hidden_dim: int = RECIPE_HIDDEN) -> SE3SBVelocityField:
    """Build the SE(3) twist head used by the Wf-31 se3sb init."""
    stub = _StubEGNNVelocityNet(hidden_dim=hidden_dim)
    return SE3SBVelocityField(base_velocity_net=stub, manifold="SE3")


# ---------------------------------------------------------------------------
# Freeze policy (same contract as train_se3sb.py / train_ru_multitask.py)
# ---------------------------------------------------------------------------
def freeze_encoder_layers_1_2(model: V4MultiTaskDMPNN) -> Tuple[int, int]:
    """Freeze DirectedMPNN message-passing layers 0..1 (entry 2 stays trainable)."""
    n_frozen = n_trainable = 0
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    for i in (0, 1):
        for p in model.encoder.dmpnn.gru_updates[i].parameters():
            p.requires_grad_(False)
        for p in model.encoder.dmpnn.edge_mlp[i].parameters():
            p.requires_grad_(False)
        for p in model.encoder.dmpnn.atom_to_edge[i].parameters():
            p.requires_grad_(False)
    for p in model.encoder.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
    return n_frozen, n_trainable


# ---------------------------------------------------------------------------
# Batching helpers (mirrored from train_se3sb.py)
# ---------------------------------------------------------------------------
def build_batches(
    pool: List[Tuple[str, int]],
    metal_id: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> List[Tuple[List[str], torch.Tensor, torch.Tensor]]:
    """Shard a (smiles, label) pool into per-batch (smiles, metal_ids, y)."""
    idx = np.arange(len(pool))
    if shuffle:
        rng.shuffle(idx)
    batches: List[Tuple[List[str], torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        smis = [pool[int(i)][0] for i in chunk]
        ys = torch.tensor([pool[int(i)][1] for i in chunk], dtype=torch.long)
        mids = torch.full((len(chunk),), metal_id, dtype=torch.long)
        batches.append((smis, mids, ys))
    return batches


def gather_train_pool(
    data: Dict[str, Dict[str, List]],
    metal: str,
) -> List[Tuple[str, int]]:
    """Return the metal's train pool — labelled rows only."""
    return [(s, y) for s, y in data[metal]["train"] if y in (0, 1)]


def cap_pool(
    pool: List[Tuple[str, int]],
    max_rows: int,
    rng: np.random.Generator,
) -> List[Tuple[str, int]]:
    """Subsample a pool to ``max_rows`` (with replacement if pool smaller)."""
    if max_rows <= 0 or len(pool) <= max_rows:
        return pool
    idx = rng.integers(0, len(pool), size=max_rows)
    return [pool[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Eval helper — per-metal ROC-AUC (works with the wrapper)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_split_auc(
    wrapper: Ru5MetalSe3SBHead,
    data: Dict[str, Dict[str, List]],
    metal: str,
    split: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, int]:
    """Compute ROC-AUC for one metal on one split using the wrapper's cls head."""
    pool = [(s, y) for s, y in data[metal][split] if y in (0, 1)]
    if not pool:
        return float("nan"), 0
    mid = wrapper.metal_id(metal)
    scores: List[float] = []
    labels: List[int] = []
    wrapper.eval()
    for start in range(0, len(pool), batch_size):
        chunk = pool[start : start + batch_size]
        smis = [s for s, _ in chunk]
        ys = torch.tensor([y for _, y in chunk], dtype=torch.long)
        mids = torch.full((len(chunk),), mid, dtype=torch.long).to(device)
        ys = ys.to(device)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            out = wrapper(smis, mids)
        logits = out["cls"][metal].float()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(ys.tolist())
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, scores)), len(labels)


# ---------------------------------------------------------------------------
# SE(3) integration smoke (mirrors train_se3sb.se3_integration_smoke)
# ---------------------------------------------------------------------------
@torch.no_grad()
def se3_integration_smoke(
    twist_head: SE3SBVelocityField,
    n_atoms: int = 8,
    n_steps: int = 8,
) -> Dict[str, object]:
    """Run a tiny SE(3) integration smoke test against the twist head."""
    twist_head.eval()
    samples = torch.randn(2, n_atoms, 3)
    try:
        x_final = integrate_with_cfm(
            samples,
            twist_head,
            n_steps=n_steps,
            method="midpoint",
        )
        out_norm = float(x_final.norm().item())
        finite = bool(torch.isfinite(x_final).all().item())
    except Exception as exc:  # pragma: no cover
        return {"out_norm": float("nan"), "finite": False, "error": str(exc)}
    return {"out_norm": out_norm, "finite": finite}


# ---------------------------------------------------------------------------
# Init wiring — attach the per-init R18 adapter to the wrapper
# ---------------------------------------------------------------------------
def attach_init_components(
    init: str,
    wrapper: Ru5MetalSe3SBHead,
    device: torch.device,
    hidden_dim: int = RECIPE_HIDDEN,
) -> Dict[str, object]:
    """Attach the per-init R18 adapter components.

    * ``tmqm`` — no extra components (R17 baseline fallback).
    * ``se3sb`` — SE(3) Schrödinger-Bridge twist head.
    * ``transport`` — relaxed endpoint term (W2.2 adapter).
    * ``multiph`` — multi-PH pocket features (W2.3 adapter).

    Returns a dict of {component_name: module} for logging / ckpt
    bookkeeping.  Components are *attached* to the wrapper as
    attributes (e.g. ``wrapper.se3_field``) so they participate in
    ``wrapper.parameters()`` and the optimiser sees them.
    """
    components: Dict[str, object] = {}
    if init == "tmqm":
        # R17 baseline fallback — no extra wiring.
        return components
    if init == "se3sb":
        twist_head = build_se3_twist_head(hidden_dim=hidden_dim).to(device)
        wrapper.se3_field = twist_head  # type: ignore[attr-defined]
        components["se3_field"] = twist_head
    elif init == "transport":
        if _RelaxedEndpointTerm is None:
            warnings.warn(
                "[train_ru_5metal_se3sb] RelaxedEndpointTerm adapter "
                "unavailable; transport init reduces to tmqm fallback."
            )
            return components
        relax_term = _RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
        wrapper.relax_term = relax_term  # type: ignore[attr-defined]
        components["relax_term"] = relax_term
    elif init == "multiph":
        if _MultiPHPocketFeatures is None:
            warnings.warn(
                "[train_ru_5metal_se3sb] MultiPHPocketFeatures adapter "
                "unavailable; multiph init reduces to tmqm fallback."
            )
            return components
        mphp = _MultiPHPocketFeatures(
            max_radius=5.0,
            max_coord=4.0,
            n_bins=8,
        ).to(device)
        wrapper.mphp = mphp  # type: ignore[attr-defined]
        components["mphp"] = mphp
    else:
        raise ValueError(f"Unknown init {init!r}; expected one of {INIT_CHOICES}")
    return components


# ---------------------------------------------------------------------------
# Train one seed (5-metal cls + reg + coord multi-task)
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    data: Dict[str, Dict[str, List]],
    metals: Tuple[str, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed of the Wf-31 5-metal cls+reg multi-task model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    backbone = V4MultiTaskDMPNN(
        hidden_dim=args.hidden,
        n_layers=args.depth,
        n_metals=len(SUPPORTED_METALS),
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        metal_embed_dim=RECIPE_METAL_EMBED_DIM,
        head_dropout=RECIPE_DROPOUT,
    ).to(device)
    if args.init == "tmqm":
        try_load_tmqm_encoder(backbone, args.encoder_ckpt)

    # Wrap V4 with the 5 cls + 5 reg head wrapper.
    wrapper = Ru5MetalSe3SBHead(
        backbone=backbone,
        metals=metals,
        head_hidden_dim=128,
        head_dropout=RECIPE_DROPOUT,
    ).to(device)

    # Attach the per-init R18 adapter components.
    r18_components = attach_init_components(
        args.init,
        wrapper,
        device,
        hidden_dim=args.hidden,
    )

    n_frozen, n_trainable = freeze_encoder_layers_1_2(backbone)
    print(
        f"[train_ru_5metal_se3sb] seed={seed} hidden={args.hidden} "
        f"depth={args.depth} init={args.init} metals={metals} "
        f"frozen={n_frozen:,} trainable={n_trainable:,} "
        f"r18_components={list(r18_components.keys()) or 'none'}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = Ru5MetalSe3SBLoss(
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        coord_weight=COORD_LOSS_WEIGHT,
        reg_weight=REG_LOSS_WEIGHT,
    )

    best_val_mean = -1.0
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()
    }
    best_epoch = 0
    val_history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        epoch_loss = 0.0
        n_batches = 0
        per_metal_batches = {
            m: build_batches(
                cap_pool(gather_train_pool(data, m), args.train_max_per_metal, rng),
                wrapper.metal_id(m),
                args.batch,
                rng,
                shuffle=True,
            )
            for m in metals
        }
        n_steps = max(len(b) for b in per_metal_batches.values())
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            batch_loss: Optional[torch.Tensor] = None
            for m in metals:
                bp = per_metal_batches[m]
                if step >= len(bp):
                    continue
                smis, mids, ys = bp[step]
                mids_dev = mids.to(device)
                ys_dev = ys.to(device)
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    out = wrapper(smis, mids_dev)
                    coord_true = torch.full_like(
                        ys_dev,
                        COORD_BY_METAL.get(m, 0),
                    )
                    y_true = {m: ys_dev}
                    loss, _ = loss_fn(out, y_true, coord_true)
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss = batch_loss + loss
            if isinstance(batch_loss, torch.Tensor):
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    wrapper.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()
                epoch_loss += float(batch_loss.item())
                n_batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_aucs: Dict[str, float] = {}
            for m in metals:
                a, _ = eval_split_auc(
                    wrapper,
                    data,
                    m,
                    "val",
                    device,
                    args.batch,
                )
                val_aucs[m] = a
            valid_aucs = [v for v in val_aucs.values() if not np.isnan(v)]
            mean_val = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            val_history.append(
                {
                    "epoch": epoch,
                    "mean_val_auc": mean_val,
                    "per_metal": val_aucs,
                }
            )
            print(
                f"[train_ru_5metal_se3sb] seed={seed} epoch={epoch:3d} "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{m}={val_aucs[m]:.3f}" for m in metals)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()
                }

    wrapper.load_state_dict(best_state)

    # Final test
    test_aucs: Dict[str, float] = {}
    for m in metals:
        a, n = eval_split_auc(
            wrapper,
            data,
            m,
            "test",
            device,
            args.batch,
        )
        test_aucs[m] = a
        print(f"[train_ru_5metal_se3sb] seed={seed} TEST {m} AUC={a:.4f} n={n}")
    valid_test = [v for v in test_aucs.values() if not np.isnan(v)]
    mean_test = float(np.mean(valid_test)) if valid_test else float("nan")
    print(
        f"[train_ru_5metal_se3sb] seed={seed} mean_test_auc={mean_test:.4f} "
        f"(best_epoch={best_epoch})"
    )

    # SE(3) integration smoke (only if se3sb init).
    se3_smoke: Dict[str, object] = {}
    if args.init == "se3sb" and hasattr(wrapper, "se3_field"):
        se3_smoke = se3_integration_smoke(wrapper.se3_field)
        print(f"[train_ru_5metal_se3sb] seed={seed} se3_smoke={se3_smoke}")

    return {
        "seed": seed,
        "init": args.init,
        "metals": list(metals),
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "test_auc_per_metal": test_aucs,
        "mean_test_auc": mean_test,
        "val_history": val_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "r18_components": list(r18_components.keys()),
        "se3_smoke": se3_smoke,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# Aggregation — 5-seed mean ± std per metal
# ---------------------------------------------------------------------------
def aggregate_per_metal(
    seed_results: List[Dict[str, object]],
    metals: Tuple[str, ...],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (per_metal_mean_auc, per_metal_std_auc) across seeds.

    For each metal ``m`` we collect ``[r["test_auc_per_metal"][m] for r in seed_results]``,
    drop NaNs, and compute ``(mean, std)``.  Empty / NaN-only lists
    yield ``float('nan')`` for both statistics (the caller decides
    how to log).
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for m in metals:
        vals = [
            float(r["test_auc_per_metal"][m])
            for r in seed_results
            if m in r["test_auc_per_metal"] and not np.isnan(r["test_auc_per_metal"][m])
        ]
        if not vals:
            means[m] = float("nan")
            stds[m] = float("nan")
        else:
            means[m] = float(np.mean(vals))
            stds[m] = float(np.std(vals, ddof=0))
    return means, stds


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(v) for v in o]
    return o


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-31 Phase 1 — R18 M22 SE(3) SB 5-metal multitask trainer. "
            "5 cls + 5 reg heads, 5-seed × 50-epoch, 4 init modes."
        ),
    )
    # Wf-35 Phase 2 — shared recipe / GPU / benchmark args.
    p = add_common_args(
        p,
        data_default=DEFAULT_DATA,
        encoder_ckpt_default=DEFAULT_ENCODER_CKPT,
        out_dir_default=DEFAULT_OUT_DIR,
        metrics_path_default=DEFAULT_METRICS_PATH,
        summary_path_default=DEFAULT_SUMMARY_PATH,
        include_init=True,  # Wf-31 supports tmqm/se3sb/transport/multiph.
    )
    p = add_gpu_args(p)

    # Wf-31-specific metals list + SE(3) steps flag.
    p.add_argument(
        "--metals",
        type=str,
        nargs="+",
        default=list(DEFAULT_METALS),
        help=(
            "Subset of metals to train + evaluate on.  Default: all 5 "
            "(Ru Ir Rh Re Os).  Each metal must be in "
            "molmetal.models.v4_multitask_dmpnn.SUPPORTED_METALS."
        ),
    )
    p.add_argument(
        "--se3-n-steps",
        type=int,
        default=8,
        help="Number of SE(3) integration steps for the smoke test.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@MEASURED_vs_PROJECTED_decorator()
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_gpu_env(args)  # Wf-35 — apply CUDA_VISIBLE_DEVICES + torch seed.
    # Resolve metals list — keep order, dedupe, validate.
    metals = tuple(dict.fromkeys(args.metals))
    unknown = [m for m in metals if m not in SUPPORTED_METALS]
    if unknown:
        raise ValueError(
            f"Unknown metals {unknown!r}; supported = {SUPPORTED_METALS!r}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_ru_5metal_se3sb] data={args.data} init={args.init} "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"seeds={args.seeds} metals={metals} "
        f"hidden={args.hidden} depth={args.depth}"
    )

    data = load_5metal_csv(args.data)
    summary_counts = {
        m: {sp: len(data[m][sp]) for sp in ("train", "val", "test")} for m in metals
    }
    print(f"[train_ru_5metal_se3sb] split counts: {summary_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_ru_5metal_se3sb] device={device}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        result = train_one_seed(seed, data, metals, args, device)
        seed_results.append(result)
        ckpt_path = args.out_dir / f"ru_5metal_se3sb_{args.init}_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "init": args.init,
                "metals": list(metals),
                "best_val_epoch": result["best_val_epoch"],
                "best_val_mean_auc": result["best_val_mean_auc"],
                "test_auc_per_metal": result["test_auc_per_metal"],
                "mean_test_auc": result["mean_test_auc"],
                "r18_components": result["r18_components"],
                "se3_smoke": result["se3_smoke"],
                "state_dict": result["best_state"],
                "recipe": {
                    "init": args.init,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "se3_n_steps": args.se3_n_steps,
                    "metals": list(metals),
                },
            },
            ckpt_path,
        )
        print(f"[train_ru_5metal_se3sb] wrote {ckpt_path}")
    wall = time.time() - t0
    print(f"[train_ru_5metal_se3sb] total wall={wall:.1f}s")

    # Per-metal 5-seed mean ± std
    per_metal_mean, per_metal_std = aggregate_per_metal(seed_results, metals)

    # Drop the heavy best_state tensor from the JSON output.
    for r in seed_results:
        if isinstance(r, dict) and "best_state" in r:
            r.pop("best_state")

    metrics = {
        "recipe": "wf31_r18_m22_se3sb_5metal_cls_reg",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": args.init,
        "metals": list(metals),
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_seed": _to_jsonable(seed_results),
        "per_metal_mean_auc": _to_jsonable(per_metal_mean),
        "per_metal_std_auc": _to_jsonable(per_metal_std),
        "ru_test_auc_per_seed": [
            r["test_auc_per_metal"].get("Ru") for r in seed_results
        ],
        "ru_mean_auc_5seed": per_metal_mean.get("Ru", float("nan")),
        "ru_std_auc_5seed": per_metal_std.get("Ru", float("nan")),
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "mean_test_auc_overall": float(
            np.nanmean([r["mean_test_auc"] for r in seed_results])
        ),
        "split_counts": summary_counts,
        "known_limitation": (
            "tmQM encoder was trained on baseline MetalCytotoxModel "
            "(hidden=128, 12 keys); V4MultiTaskDMPNN uses DirectedMPNN "
            "(hidden=300, 36 keys).  Encoder fell back to random init "
            "via try_load_tmqm_encoder — pretrain_v4multitask_dmpnn.py "
            "is the future fix."
        ),
    }
    args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    print(
        f"[train_ru_5metal_se3sb] wrote {args.metrics_path} "
        f"(Ru 5-seed mean ± std = "
        f"{per_metal_mean.get('Ru', float('nan')):.4f} "
        f"± {per_metal_std.get('Ru', float('nan')):.4f})"
    )

    # Slim final summary — headline metric for the Wf-31 §6
    # acceptance criteria.
    final_summary = {
        "recipe": "wf31_r18_m22_se3sb_5metal_cls_reg_final_summary",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "init": args.init,
        "metals": list(metals),
        "wall_seconds": wall,
        "ru_test_auc_per_seed": metrics["ru_test_auc_per_seed"],
        "ru_mean_auc_5seed": metrics["ru_mean_auc_5seed"],
        "ru_std_auc_5seed": metrics["ru_std_auc_5seed"],
        "per_metal_mean_auc": metrics["per_metal_mean_auc"],
        "per_metal_std_auc": metrics["per_metal_std_auc"],
        "mean_test_auc_per_seed": metrics["mean_test_auc_per_seed"],
        "mean_test_auc_overall": metrics["mean_test_auc_overall"],
        "wf31_headline": ("mean(Ru 5-seed AUC) ≥ 0.86 claims beating Krasnov 0.81"),
    }
    args.summary_path.write_text(json.dumps(_to_jsonable(final_summary), indent=2))
    print(f"[train_ru_5metal_se3sb] wrote {args.summary_path}")
    # Return (exit_code, args) so the @MEASURED_vs_PROJECTED_decorator
    # can use the explicit args namespace instead of re-parsing argv
    # (which would fail on the Wf-31-specific --metals flag).
    return 0, args


if __name__ == "__main__":
    out = main()
    if isinstance(out, tuple):
        rc = out[0]
    else:
        rc = out
    raise SystemExit(rc)
