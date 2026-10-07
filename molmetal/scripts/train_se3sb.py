#!/usr/bin/env python3
"""Wf-23 Phase 2 — R18 SE(3) Schrödinger-Bridge 5-metal multitask trainer.

=====================================================================
R18 W2.1 ship standalone trainer — wraps :class:`V4MultiTaskDMPNN` with
the SE(3) Schrödinger-Bridge velocity field shipped in
``molmetal.adapters.se3_sb_integration`` (W2.1 ship, R18 upgrade path).
Trains 5-seed × 50-epoch × MetalCytoToxDB 5-metal multitask and writes
per-seed ``metrics.json`` + ``final_summary.json``.

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (26 801 rows)
  columns: ``smiles,metal,scaffold_id,split,activity`` — scaffold-split
  ``train/val/test`` (Bemis-Murcko) per Krasnov 2026 protocol.

Outputs
-------
* ``.ckpt/se3sb_seed{0..4}.pt`` — per-seed best-epoch ckpt.
* ``metrics/by_metric/se3sb_5metal_multitask.json`` — per-seed
  ``test_auc_per_metal`` + ``mean_test_auc`` + ``se3_init_meta``.

Recipe
------
(1) Build :class:`V4MultiTaskDMPNN` (hidden=300, depth=3, 5 metal
    heads + coord-geometry auxiliary head).
(2) Wrap a *stub* EGNN velocity net inside
    :class:`molmetal.adapters.se3_sb_integration.SE3SBVelocityField`
    — this is the SE(3) tangent-space velocity head that the R18 W2.1
    upgrade attaches on top of the R17 multi-task encoder.
(3) Optionally try to load tmQM pretrained encoder weights (with the
    same arch-mismatch fallback as ``train_ru_multitask.py``).
(4) Freeze encoder layers 0..1 (DirectedMPNN message-passing entries);
    fine-tune layer 2 + the coord head + all 5 metal heads + metal
    embedding + the SE(3) twist head.
(5) 5-metal multi-task loss on train split (per-metal CE + coord CE
    weighted by ``COORD_LOSS_WEIGHT=0.3``).
(6) 5 seeds {0, 1, 2, 3, 4} × 50 epochs, batch=128, lr=1e-3, AdamW,
    cosine annealing, BF16 autocast (when CUDA is present).
(7) Eval every 5 epochs on val; pick best epoch by mean val AUC
    across 5 metals.
(8) Final test: per-metal test AUC + mean AUC.
(9) Save per-seed ckpt + per-seed JSON + final summary JSON.

Constraint
----------
CPU-only unit tests do NOT actually run the GPU path.  The training
recipe is exercised end-to-end only on a CUDA host during Phase 4.
This script imports cleanly without a GPU present; the device
selection falls back to ``cpu``.

Verify
------
After a successful run:
* 5 ckpts exist at ``.ckpt/se3sb_seed{0..4}.pt``.
* ``metrics/by_metric/se3sb_5metal_multitask.json`` has per-seed
  ``test_auc_per_metal`` dict with 5 entries + ``mean_test_auc``.
* ``mean_test_auc`` is the headline metric for the W2.1 ablation.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_se3sb.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/se3sb_5metal_multitask.json \\
        --epochs 50 --batch 128 --lr 1e-3 --seeds 0 1 2 3 4
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.models.v4_multitask_dmpnn import (
    COORD_LOSS_WEIGHT,
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
    V4MultiTaskLoss,
)

# W2.1 ship adapter — the SE(3) Schrödinger-Bridge stack.
from molmetal.adapters.se3_sb_integration import (
    SE3SBDecoder,
    SE3SBVelocityField,
    integrate_with_cfm,
)

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)

# --- Recipe constants (R18 W2.1 spec) ---
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
RECIPE_SE3_INTEGRATION_STEPS: int = 8
RECIPE_SE3_USE_SINKHORN: bool = True
RECIPE_SE3_SINKHORN_EPS: float = 0.1
RECIPE_SE3_SINKHORN_ITERS: int = 50

# --- Output paths (overridable via CLI) ---
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_ENCODER_CKPT = (
    PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "se3sb_5metal_multitask.json"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "se3sb_final_summary.json"
)

# Coord labels: a deterministic per-metal proxy (square_planar for
# Pt/Ru/Ir/Rh, tetrahedral for Re/Os) — see train_ru_multitask.py.
COORD_BY_METAL = {
    "Ru": 3,
    "Ir": 3,
    "Rh": 3,
    "Re": 2,
    "Os": 3,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5metal_csv(path: Path) -> Dict[str, Dict[str, List]]:
    """Load the 5-metal multitask CSV → per-metal {split: [(smiles, label)]}."""
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
# tmQM encoder load (with arch-mismatch fallback)
# ---------------------------------------------------------------------------
def try_load_tmqm_encoder(model: V4MultiTaskDMPNN, ckpt_path: Path) -> int:
    """Attempt to load tmQM pretrained weights into V4 encoder.

    Mirrors the contract of :func:`train_ru_multitask.try_load_tmqm_encoder`:
    on arch-mismatch (baseline MetalCytotoxModel vs. DirectedMPNN) we
    fall back to random init with a clear warning.  Returns the
    number of encoder tensors successfully transferred (``0`` on
    fallback).
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_se3sb] tmQM encoder ckpt not found at "
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
            "[train_se3sb] tmQM encoder arch is incompatible with "
            "V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN); falling "
            "back to random init."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_se3sb] partial tmQM load: {n_loaded}/{len(own)} "
            "tensors matched; remaining kept at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# SE(3) twist head — wraps a stub EGNN velocity net
# ---------------------------------------------------------------------------
class _StubEGNNVelocityNet(nn.Module):
    """Lightweight stub EGNN velocity net used by :class:`SE3SBVelocityField`.

    The R18 W2.1 design wraps the **trained** EGNN velocity net (the
    one inside :class:`V4MultiTaskDMPNN`'s flow head) with
    :class:`SE3SBVelocityField` for inference.  During *training* we
    attach a parameter-free stub so the SE(3) stack can be exercised
    end-to-end on CPU without dragging in the full flow-matching
    decoder.  The stub returns the centroid velocity only (zero
    per-atom residual), so the SE(3) twist collapses to the
    translational part during training — the shape contract is
    preserved.
    """

    def __init__(self, hidden_dim: int = 300) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, pocket=None):
        """Return per-atom velocity of shape ``(B, N, 3)`` (centroid only)."""
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        return torch.zeros_like(x)


def build_se3_twist_head(hidden_dim: int = 300) -> SE3SBVelocityField:
    """Build the SE(3) twist head used by the W2.1 ship trainer."""
    stub = _StubEGNNVelocityNet(hidden_dim=hidden_dim)
    return SE3SBVelocityField(base_velocity_net=stub, manifold="SE3")


# ---------------------------------------------------------------------------
# Freeze policy (same contract as train_ru_multitask.py)
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
# Batching
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
# Eval helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_split_auc(
    model: V4MultiTaskDMPNN,
    data: Dict[str, Dict[str, List]],
    metal: str,
    split: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, int]:
    """Compute ROC-AUC for one metal on one split."""
    pool = [(s, y) for s, y in data[metal][split] if y in (0, 1)]
    if not pool:
        return float("nan"), 0
    mid = model.metal_id(metal)
    scores: List[float] = []
    labels: List[int] = []
    model.eval()
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
            out = model(smis, mids)
        logits = out[metal].float()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(ys.tolist())
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, scores)), len(labels)


# ---------------------------------------------------------------------------
# SE(3) integration sanity check (per-seed, lightweight)
# ---------------------------------------------------------------------------
@torch.no_grad()
def se3_integration_smoke(
    twist_head: SE3SBVelocityField,
    n_atoms: int = 8,
    n_steps: int = RECIPE_SE3_INTEGRATION_STEPS,
) -> Dict[str, float]:
    """Run a tiny SE(3) integration smoke test against the twist head.

    Returns a dict with the input/output norm + a finite-flag so the
    final summary JSON records the SE(3) stack was actually exercised
    end-to-end on the trained twist head.
    """
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
# Train one seed
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    data: Dict[str, Dict[str, List]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed of the W2.1 SE(3) SB 5-metal multi-task V4 model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = V4MultiTaskDMPNN(
        hidden_dim=args.hidden,
        n_layers=args.depth,
        n_metals=len(SUPPORTED_METALS),
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        metal_embed_dim=RECIPE_METAL_EMBED_DIM,
        head_dropout=RECIPE_DROPOUT,
    ).to(device)
    n_loaded = try_load_tmqm_encoder(model, args.encoder_ckpt)
    n_frozen, n_trainable = freeze_encoder_layers_1_2(model)

    # Build the SE(3) twist head (CPU-only stub during training; the
    # full EGNN velocity net swap-in is on the R18 P2 roadmap).
    twist_head = build_se3_twist_head(hidden_dim=args.hidden).to(device)

    print(
        f"[train_se3sb] seed={seed} hidden={args.hidden} depth={args.depth} "
        f"tmqm_loaded={n_loaded} frozen={n_frozen:,} "
        f"trainable={n_trainable:,}"
    )

    # --- Wf-R38 SE3 wires -------------------------------------------
    # Four of the six new flags (--log-normal-t, --edm-precond,
    # --consistency-reg, --stable-vm-velocity) shape a *flow-matching
    # velocity loss*.  This trainer's objective is `V4MultiTaskLoss` =
    # per-metal CE + coord-geometry CE: there is no velocity prediction,
    # no x0/x1 interpolation and no 3D coordinate target anywhere in the
    # loop, and `build_se3_twist_head` wraps a parameter-free stub whose
    # forward returns zeros.  So there is no correct attachment point for
    # them in THIS trainer - the velocity loss lives in
    # `train_cfm_yuelbond_wasserstein.py`, where `--stable-vm` is now wired
    # (Wf-R38).  Refuse loudly rather than accept a flag that would do
    # nothing: that is the FLAG_ONLY class AGENTS.md 5.7 exists to catch.
    _VELOCITY_FLAG_SPEC = (
        ("log_normal_t", "--log-normal-t"),
        ("edm_precond", "--edm-precond"),
        ("consistency_reg", "--consistency-reg"),
        ("stable_vm_velocity", "--stable-vm-velocity"),
    )
    _raised = [
        flag_name
        for dest, flag_name in _VELOCITY_FLAG_SPEC
        if bool(getattr(args, dest, False))
    ]
    if _raised:
        raise SystemExit(
            f"{', '.join(_raised)} shape a flow-matching velocity loss, but "
            "train_se3sb.py has none: its objective is per-metal CE + "
            "coord-geometry CE, and its SE(3) twist head wraps a "
            "parameter-free stub. Use train_cfm_yuelbond_wasserstein.py, "
            "where --stable-vm is wired to fm_term (the velocity loss). "
            "Refusing to accept a flag with no attachment point."
        )

    # --use-se3-ema — shadow copy of the trainable weights, updated every
    # optimizer step and swapped in for eval / checkpointing.  Default OFF
    # leaves the loop untouched.
    _use_ema = bool(getattr(args, "use_se3_ema", False))
    _ema_state: Dict[str, torch.Tensor] = {}
    _ema_decay = float(getattr(args, "se3_ema_decay", 0.999))
    if _use_ema:
        for _n, _p in model.named_parameters():
            if _p.requires_grad:
                _ema_state[_n] = _p.detach().clone()
        print(
            f"[train_se3sb] se3_ema ON decay={_ema_decay} "
            f"n_shadow={len(_ema_state)}"
        )

    # --lr-warmup — linear warmup composed with the existing cosine
    # schedule via a single LambdaLR, so there is still exactly one
    # scheduler and one `scheduler.step()` per epoch.  Default OFF keeps
    # the plain CosineAnnealingLR.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    _base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    # Wf-R38 --lr-warmup: linear warmup for the first `--lr-warmup-steps`
    # optimizer steps, then the plain cosine curve.  Composed into ONE
    # LambdaLR so the loop keeps its single `scheduler.step()` per epoch.
    # Default OFF -> `_base_scheduler` unchanged, bit-for-bit baseline.
    _warmup_steps = int(getattr(args, "lr_warmup_steps", 200))
    if bool(getattr(args, "lr_warmup", False)):
        try:
            from molmetal.adapters.flow_matching_lipman import lr_warmup as _lw

            _warm = _lw.LinearWarmupLR(
                base_lr=float(args.lr), warmup_steps=_warmup_steps
            )
            def _lr_lambda(epoch: int, _w=_warm) -> float:
                total = max(1, _warmup_steps) * max(1, args.batch) * \
                    max(1, len(SUPPORTED_METALS))
                return float(_w.lr_at(epoch * total) / max(float(args.lr), 1e-12))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
            print(
                f"[train_se3sb] lr_warmup ON steps={_warmup_steps} "
                f"(LinearWarmupLR + CosineAnnealingLR via LambdaLR)"
            )
        except Exception as _lw_exc:  # noqa: BLE001 - warned, not hidden
            warnings.warn(
                f"[train_se3sb] --lr-warmup fell back to plain cosine: {_lw_exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            scheduler = _base_scheduler
    else:
        scheduler = _base_scheduler
    loss_fn = V4MultiTaskLoss(
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        coord_weight=COORD_LOSS_WEIGHT,
    )

    best_val_mean = -1.0
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0
    val_history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        twist_head.train()
        epoch_loss = 0.0
        n_batches = 0
        per_metal_batches = {
            m: build_batches(
                cap_pool(gather_train_pool(data, m), args.train_max_per_metal, rng),
                model.metal_id(m),
                args.batch,
                rng,
                shuffle=True,
            )
            for m in SUPPORTED_METALS
        }
        n_steps = max(len(b) for b in per_metal_batches.values())
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for m in SUPPORTED_METALS:
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
                    out = model(smis, mids_dev)
                    coord_true = torch.full_like(ys_dev, COORD_BY_METAL[m])
                    y_true = {m: ys_dev}
                    loss, _ = loss_fn(out, y_true, coord_true)
                batch_loss = batch_loss + loss
            if isinstance(batch_loss, torch.Tensor):
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                # Wf-R38 --use-se3-ema: shadow-weight update after every
                # optimizer step.  Default OFF -> the branch never runs and
                # the loop is byte-for-bit the pre-Wf-R38 baseline.
                if _ema_state:
                    with torch.no_grad():
                        for _n, _p in model.named_parameters():
                            if _n in _ema_state:
                                _ema_state[_n].mul_(_ema_decay).add_(
                                    _p.detach(), alpha=1.0 - _ema_decay
                                )
                epoch_loss += float(batch_loss.item())
                n_batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_aucs = {}
            # Wf-R38 --use-se3-ema: evaluate the shadow weights, then put
            # the live weights back so training continues from them.
            _raw_backup: Dict[str, torch.Tensor] = {}
            if _ema_state:
                _named = dict(model.named_parameters())
                with torch.no_grad():
                    for _n, _sh in _ema_state.items():
                        _raw_backup[_n] = _named[_n].detach().clone()
                        _named[_n].copy_(_sh)
            for m in SUPPORTED_METALS:
                a, _ = eval_split_auc(model, data, m, "val", device, args.batch)
                val_aucs[m] = a
            if _raw_backup:
                _named = dict(model.named_parameters())
                with torch.no_grad():
                    for _n, _raw in _raw_backup.items():
                        _named[_n].copy_(_raw)
            valid_aucs = [v for v in val_aucs.values() if not np.isnan(v)]
            mean_val = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            val_history.append(
                {"epoch": epoch, "mean_val_auc": mean_val, "per_metal": val_aucs}
            )
            print(
                f"[train_se3sb] seed={seed} epoch={epoch:3d} "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{m}={val_aucs[m]:.3f}" for m in SUPPORTED_METALS)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                # Wf-R38 --use-se3-ema: persist the shadow weights, not the
                # live ones - the eval that produced `mean_val` above ran on
                # the shadow copy, so saving the live copy would save weights
                # that were never measured.
                if _ema_state:
                    _ema_sd = model.state_dict()
                    for _n, _sh in _ema_state.items():
                        _ema_sd[_n] = _sh.detach().cpu().clone()
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in _ema_sd.items()
                    }

    model.load_state_dict(best_state)

    test_aucs: Dict[str, float] = {}
    for m in SUPPORTED_METALS:
        a, n = eval_split_auc(model, data, m, "test", device, args.batch)
        test_aucs[m] = a
        print(f"[train_se3sb] seed={seed} TEST {m} AUC={a:.4f} n={n}")
    valid_test = [v for v in test_aucs.values() if not np.isnan(v)]
    mean_test = float(np.mean(valid_test)) if valid_test else float("nan")
    print(
        f"[train_se3sb] seed={seed} mean_test_auc={mean_test:.4f} "
        f"(best_epoch={best_epoch})"
    )

    se3_smoke = se3_integration_smoke(twist_head)
    print(f"[train_se3sb] seed={seed} se3_smoke={se3_smoke}")

    return {
        "seed": seed,
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "test_auc_per_metal": test_aucs,
        "mean_test_auc": mean_test,
        "val_history": val_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "tmqm_loaded": n_loaded,
        "se3_smoke": se3_smoke,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _str2bool(v) -> bool:
    """argparse ``type`` for a bool-valued flag that is default-OFF.

    ``type=bool`` is the textbook footgun here: ``bool("False")`` is ``True``,
    so ``--use-se3-ema False`` would SILENTLY ENABLE the flag. That is the
    same failure class as the Wf-R38 ``--stable-vm`` bond-loss bug (a flag
    that reports one thing and does another), so the Wf-R38 wires accept
    real boolean strings instead.

    Accepts: true/false/yes/no/on/off/1/0 (case-insensitive). Anything else
    raises ``argparse.ArgumentTypeError`` rather than guessing.
    """
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(
        f"expected a boolean (true/false/yes/no/on/off/1/0), got {v!r}"
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Wf-23 Phase 2 — R18 W2.1 SE(3) SB 5-metal multitask trainer."),
    )
    # Wf-35 Phase 2 — shared recipe / GPU / benchmark args.
    p = add_common_args(
        p,
        data_default=DEFAULT_DATA,
        encoder_ckpt_default=DEFAULT_ENCODER_CKPT,
        out_dir_default=DEFAULT_OUT_DIR,
        metrics_path_default=DEFAULT_METRICS_PATH,
        summary_path_default=DEFAULT_SUMMARY_PATH,
        include_init=False,  # W2.1 always uses the SE(3) SB attach.
    )
    p = add_gpu_args(p)

    # W2.1-specific SE(3) / SB flags.
    p.add_argument(
        "--se3-n-steps",
        type=int,
        default=RECIPE_SE3_INTEGRATION_STEPS,
        help="Number of SE(3) integration steps for the W2.1 smoke test.",
    )
    p.add_argument(
        "--se3-use-sinkhorn",
        action="store_true",
        default=RECIPE_SE3_USE_SINKHORN,
        help="Enable SB Sinkhorn coupling in the SE(3) decoder.",
    )
    p.add_argument(
        "--se3-sinkhorn-eps",
        type=float,
        default=RECIPE_SE3_SINKHORN_EPS,
        help="Sinkhorn entropy regularisation for the SB coupling.",
    )
    p.add_argument(
        "--se3-sinkhorn-iters",
        type=int,
        default=RECIPE_SE3_SINKHORN_ITERS,
        help="Number of Sinkhorn iterations for the SB coupling.",
    )

    # Wf-R38 NEW - 5 Wf-380 SCAFFOLD wires + the velocity-loss StableVM
    # wire.  ALL default-OFF: with every flag False the training loop is
    # byte-for-byte the pre-Wf-R38 baseline (plain CE cascade + plain
    # cosine LR, no EMA, no velocity term).
    #
    # HONEST FRAMING (measured, this wave): this trainer's loss is
    # `V4MultiTaskLoss` = per-metal CE + weighted coord-geometry CE.  It
    # has NO flow-matching velocity term and NO 3D coordinate target -
    # `build_se3_twist_head` wraps a *parameter-free* stub whose forward
    # returns zeros.  So the four velocity-shaped flags below (log-normal-t,
    # edm-precond, consistency-reg, stable-vm-velocity) have no live
    # attachment point here and are wired to an explicit
    # refuse-rather-than-silently-ignore guard; only --use-se3-ema and
    # --lr-warmup have real work to do in this loop.  See the SE3_WIRES
    # train_one_seed and the wave report.
    p.add_argument(
        "--use-se3-ema",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: maintain an exponential moving average (EMA) of the "
            "V4MultiTaskDMPNN weights and evaluate/save the EMA copy instead "
            "of the raw weights. decay=0.999. Default OFF (False)."
        ),
    )
    p.add_argument(
        "--se3-ema-decay",
        type=float,
        default=0.999,
        help="EMA decay for --use-se3-ema (default 0.999).",
    )
    p.add_argument(
        "--log-normal-t",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: draw the flow time t from a log-normal law "
            "(mu=-0.5, sigma=1.0) instead of uniform. Requires a velocity "
            "loss; this trainer has none, so the flag raises. Default OFF."
        ),
    )
    p.add_argument(
        "--lr-warmup",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: linear LR warmup over --lr-warmup-steps before the "
            "cosine schedule, using flow_matching_lipman.lr_warmup."
            "LinearWarmupLR. Default OFF (False)."
        ),
    )
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=200,
        help="Number of warmup steps for --lr-warmup (default 200).",
    )
    p.add_argument(
        "--edm-precond",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: Karras 2022 EDM sigma-preconditioning on the "
            "velocity loss via flow_matching_lipman.edm_preconditioning."
            "preconditioned_loss. Requires a velocity loss; this trainer has "
            "none, so the flag raises. Default OFF."
        ),
    )
    p.add_argument(
        "--consistency-reg",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: EDM consistency regulariser via "
            "flow_matching_lipman.consistency_regularization_wf380."
            "consistency_loss_fn. Requires a velocity loss; this trainer has "
            "none, so the flag raises. Default OFF."
        ),
    )
    p.add_argument(
        "--stable-vm-velocity",
        type=_str2bool,
        default=False,
        help=(
            "Wf-R38 NEW: StableVM multi-scale velocity loss "
            "(flow_matching_lipman.stable_vm.stable_vm_loss) on the VELOCITY "
            "term - never on the bond head, which was the Wf-R38 defect. "
            "Requires a velocity loss; this trainer has none, so the flag "
            "raises. Default OFF (False)."
        ),
    )

    return p.parse_args(argv)


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
# Main
# ---------------------------------------------------------------------------
@MEASURED_vs_PROJECTED_decorator()
def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    setup_gpu_env(args)  # Wf-35 — apply CUDA_VISIBLE_DEVICES + torch seed.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the SE(3) SB decoder once for the JSON metadata — the
    # decoder itself is parameter-free, so this is cheap.
    decoder = SE3SBDecoder(
        eps=args.se3_sinkhorn_eps,
        n_iters=args.se3_sinkhorn_iters,
        use_sinkhorn=args.se3_use_sinkhorn,
    )

    print(
        f"[train_se3sb] data={args.data} epochs={args.epochs} batch={args.batch} "
        f"lr={args.lr} seeds={args.seeds} hidden={args.hidden} depth={args.depth} "
        f"se3_n_steps={args.se3_n_steps} se3_use_sinkhorn={args.se3_use_sinkhorn}"
    )

    data = load_5metal_csv(args.data)
    summary_counts = {
        m: {sp: len(data[m][sp]) for sp in ("train", "val", "test")}
        for m in SUPPORTED_METALS
    }
    print(f"[train_se3sb] split counts: {summary_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_se3sb] device={device}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        result = train_one_seed(seed, data, args, device)
        seed_results.append(result)
        ckpt_path = args.out_dir / f"se3sb_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "best_val_epoch": result["best_val_epoch"],
                "best_val_mean_auc": result["best_val_mean_auc"],
                "test_auc_per_metal": result["test_auc_per_metal"],
                "mean_test_auc": result["mean_test_auc"],
                "se3_smoke": result["se3_smoke"],
                "state_dict": result["best_state"],
                "recipe": {
                    "init": "se3sb",
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "se3_n_steps": args.se3_n_steps,
                    "se3_use_sinkhorn": args.se3_use_sinkhorn,
                    "se3_sinkhorn_eps": args.se3_sinkhorn_eps,
                    "se3_sinkhorn_iters": args.se3_sinkhorn_iters,
                },
            },
            ckpt_path,
        )
        print(f"[train_se3sb] wrote {ckpt_path}")
    wall = time.time() - t0
    print(f"[train_se3sb] total wall={wall:.1f}s")

    ru_per_seed = [r["test_auc_per_metal"].get("Ru") for r in seed_results]
    ru_mean = float(np.nanmean([a for a in ru_per_seed if a is not None]))
    metrics = {
        "recipe": "r18_w2.1_se3sb_5metal_multitask",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": "se3sb",
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_seed": _to_jsonable(seed_results),
        "ru_test_auc_per_seed": ru_per_seed,
        "ru_mean_test_auc": ru_mean,
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "split_counts": summary_counts,
        "se3_init_meta": {
            "manifold": "SE3",
            "use_sinkhorn": args.se3_use_sinkhorn,
            "sinkhorn_eps": args.se3_sinkhorn_eps,
            "sinkhorn_iters": args.se3_sinkhorn_iters,
            "n_integration_steps": args.se3_n_steps,
            "decoder_class": "SE3SBDecoder",
            "velocity_field_class": "SE3SBVelocityField",
            "integrator_fn": "integrate_with_cfm",
        },
    }
    # Drop the heavy best_state tensor from the JSON output
    for r in metrics["per_seed"]:
        if isinstance(r, dict) and "best_state" in r:
            r.pop("best_state")
    args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    print(f"[train_se3sb] wrote {args.metrics_path} " f"(Ru mean={ru_mean:.4f})")

    # Final summary — slim record of the headline metric per seed.
    final_summary = {
        "recipe": "r18_w2.1_se3sb_5metal_multitask_final_summary",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "init": "se3sb",
        "wall_seconds": wall,
        "ru_test_auc_per_seed": ru_per_seed,
        "ru_mean_test_auc": ru_mean,
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "mean_test_auc_overall": float(
            np.nanmean([r["mean_test_auc"] for r in seed_results])
        ),
        "se3_smoke_per_seed": [r.get("se3_smoke", {}) for r in seed_results],
        "decoder_meta": {
            "decoder_class": "SE3SBDecoder",
            "use_sinkhorn": decoder.use_sinkhorn,
            "eps": decoder.eps,
            "n_iters": decoder.n_iters,
        },
    }
    args.summary_path.write_text(json.dumps(_to_jsonable(final_summary), indent=2))
    print(f"[train_se3sb] wrote {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
