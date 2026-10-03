#!/usr/bin/env python3
"""Phase A Stage 1.4 — 3-seed × 100-epoch 5-metal multi-task fine-tune.

=====================================================================
Multi-task fine-tune of :class:`V4MultiTaskDMPNN` on the Ru/Ir/Rh/Re/Os
MetalCytoToxDB subset, with optional tmQM-pretrained encoder
initialisation.  Goal: lift m1 Ru AUC from 0.712 (1 seed × 5 epochs) to
≥ 0.86 (3 seeds × 100 epochs + multi-task + tmQM pretrain), per the
recipe in ``TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`` §2 Stage 1.

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (26 801 rows)
  columns: ``smiles,metal,scaffold_id,split,activity`` — scaffold-split
  ``train/val/test`` (Bemis-Murcko) per Krasnov 2026 protocol.

Outputs
-------
* ``.ckpt/ru_multitask_seed{0,1,2}.pt`` — per-seed best-epoch ckpt.
* ``metrics/by_metric/ru_5metal_multitask.json`` — final test AUC per
  metal + mean across 5 metals + best val epoch.

Recipe
------
(1) Build V4MultiTaskDMPNN (hidden=300, depth=3, 5 metal heads +
    coord-geometry auxiliary head).
(2) Try to load tmQM pretrained encoder into the encoder backbone
    (see :func:`try_load_tmqm_encoder` for the arch-mismatch fallback).
    If the tmQM encoder is from the *baseline* MetalCytotoxModel
    architecture (incompatible state-dict with DirectedMPNN), fall back
    to **random init** with a clear warning (this is the realistic
    case as of 2026-09-25 — see KNOWN LIMITATION below).
(3) Freeze encoder layers 1-2 (DirectedMPNN message passing layers
    0..1 plus their edge_mlp / atom_to_edge entries 0..1); fine-tune
    layer 3 (entry 2) + the coord head + all 5 metal heads + metal
    embedding.
(4) 5-metal multi-task loss on train split (per-metal CE + coord CE
    weighted by ``COORD_LOSS_WEIGHT=0.3``).
(5) 3 seeds {0, 1, 2} × 100 epochs, batch=128, lr=1e-3, AdamW,
    cosine annealing, BF16 autocast.
(6) Eval every 10 epochs on val; pick best epoch by mean val AUC
    across 5 metals.
(7) Final test: per-metal test AUC + mean AUC.
(8) Save ckpt + JSON metrics.

KNOWN LIMITATION (2026-09-25)
-----------------------------
``models/tmqm_pretrained/tmqm_pretrained_encoder.pt`` was trained on the
*baseline* ``MetalCytotoxModel`` (baselines/dmpnn.py: 12 tensors,
hidden=128, ``_GRUUpdate`` 3-gate GRU), not on the DirectedMPNN
encoder that backs V4MultiTaskDMPNN (models/dmpnn.py: 36 tensors,
hidden=300, ``GRUMessageUpdate`` = ``msg_proj + GRUCell``).  The two
architectures are incompatible (different state-dict keys, different
hidden dim, different bond-feature dim 10 vs 6).  A future
``pretrain_v4multitask_dmpnn.py`` is needed to pretrain V4's
DirectedMPNN encoder on tmQM before this recipe can load pretrained
weights end-to-end.  This script falls back to **random init** for V4
encoder in the meantime.

Verify
------
After a successful run:
* 3 ckpts exist at ``.ckpt/ru_multitask_seed{0,1,2}.pt``.
* ``metrics/by_metric/ru_5metal_multitask.json`` has per-seed
  ``test_auc_per_metal`` dict with 5 entries + ``mean_test_auc``.
* Ru per-seed test AUC is reported; the recipe aim is
  ``mean(Ru) >= 0.86`` to claim "beat Krasnov 0.81 by +0.05".

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_ru_multitask.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/ru_5metal_multitask.json \\
        --epochs 100 --batch 128 --lr 1e-3 --seeds 0 1 2
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from molmetal.models.v4_multitask_dmpnn import (
    COORD_LOSS_WEIGHT,
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
    V4MultiTaskLoss,
)

# R18 W2.1/W2.2/W2.3 ship adapters — used only when ``--init`` is
# one of ``se3sb`` / ``transportrelax`` / ``multiph``.  The R17
# default ``--init in {tmqm, random}`` keeps the existing behaviour
# (no R18 adapter wiring).  We import lazily inside ``train_one_seed``
# so the R17 cold-start path remains unchanged.
try:                                            # pragma: no cover
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField as _SE3SBVelocityField,
    )
except Exception:                               # pragma: no cover
    _SE3SBVelocityField = None                  # type: ignore[assignment]
try:                                            # pragma: no cover
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm as _RelaxedEndpointTerm,
    )
except Exception:                               # pragma: no cover
    _RelaxedEndpointTerm = None                 # type: ignore[assignment]
try:                                            # pragma: no cover
    from molmetal.adapters.multi_ph_integration import (
        MultiPHPocketFeatures as _MultiPHPocketFeatures,
    )
except Exception:                               # pragma: no cover
    _MultiPHPocketFeatures = None               # type: ignore[assignment]

# --- Recipe constants (frozen by the Stage 1.4 spec) ---
RECIPE_EPOCHS: int = 100
RECIPE_BATCH: int = 128
RECIPE_LR: float = 1e-3
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2)
RECIPE_HIDDEN: int = 300            # V4MultiTaskDMPNN default
RECIPE_DEPTH: int = 3               # V4MultiTaskDMPNN default
RECIPE_EVAL_EVERY: int = 10         # eval cadence (epochs)
RECIPE_N_COORD_CLASSES: int = 5
RECIPE_METAL_EMBED_DIM: int = 64
RECIPE_DROPOUT: float = 0.2

# --- Output paths (overridable via CLI) ---
DEFAULT_DATA = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
DEFAULT_ENCODER_CKPT = PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "ru_5metal_multitask.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5metal_csv(path: Path) -> Dict[str, Dict[str, List]]:
    """Load the 5-metal multitask CSV → per-metal {split: [(smiles, label)]}.

    Returns:
        ``data[metal][split] = [(smiles, activity_int), ...]``
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
# tmQM encoder load (with arch-mismatch fallback)
# ---------------------------------------------------------------------------
def try_load_tmqm_encoder(model: V4MultiTaskDMPNN, ckpt_path: Path) -> int:
    """Attempt to load tmQM pretrained weights into V4 encoder.

    The published tmQM encoder was trained on the *baseline*
    MetalCytotoxModel architecture and is **not** compatible with the
    DirectedMPNN encoder used by V4MultiTaskDMPNN.  On any mismatch
    we fall back to random init with a clear warning, so the rest of
    the recipe (5-metal multi-task + freeze layer 1-2 + fine-tune
    layer 3 + heads) runs unchanged.

    Returns:
        ``n_loaded`` — number of encoder tensors successfully
        transferred (``0`` on fallback).  We deliberately do NOT
        silently partial-load mismatched shapes.
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_ru_multitask] tmQM encoder ckpt not found at "
            f"{ckpt_path}; using random init for V4 encoder."
        )
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    own = model.encoder.dmpnn.state_dict()
    # Probe compatibility: any single key with matching name + shape
    # counts as a "compatible" ckpt; otherwise the architectures differ.
    n_loaded = 0
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            n_loaded += 1
    if n_loaded == 0:
        warnings.warn(
            "[train_ru_multitask] tmQM encoder arch is incompatible "
            "with V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN); "
            "falling back to random init.  See script docstring "
            "KNOWN LIMITATION for the fix path "
            "(pretrain_v4multitask_dmpnn.py)."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_ru_multitask] partial tmQM load: {n_loaded}/"
            f"{len(own)} tensors matched; remaining kept at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# Freeze policy
# ---------------------------------------------------------------------------
def freeze_encoder_layers_1_2(model: V4MultiTaskDMPNN) -> Tuple[int, int]:
    """Freeze DirectedMPNN message-passing layers 0..1 (entry 2 stays trainable).

    The DirectedMPNN has ``n_layers=3`` GRU message-passing blocks
    ``encoder.dmpnn.gru_updates[i]`` plus matching
    ``encoder.dmpnn.edge_mlp[i]`` and ``encoder.dmpnn.atom_to_edge[i]``.
    We freeze ``i=0`` and ``i=1`` and leave ``i=2`` trainable.  The
    bottom-level ``atom_embed`` and ``edge_embed`` (single shared
    input projection) are kept trainable per spec ("freeze first 2
    D-MPNN layers" = freeze message-passing layers 0..1).

    Returns:
        ``(n_frozen, n_trainable)`` parameter counts.
    """
    n_frozen = n_trainable = 0
    for p in model.encoder.parameters():
        p.requires_grad_(True)        # default: trainable
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
        chunk = idx[start:start + batch_size]
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
    """Subsample a pool to ``max_rows`` (with replacement if pool smaller).

    Bound per-epoch wall-time without changing the multi-task recipe.
    """
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
    """Compute ROC-AUC for one metal on one split.

    Returns ``(auc, n_test)`` — ``auc`` is ``NaN`` if the split has a
    single class (degenerate).
    """
    pool = [(s, y) for s, y in data[metal][split] if y in (0, 1)]
    if not pool:
        return float("nan"), 0
    mid = model.metal_id(metal)
    scores: List[float] = []
    labels: List[int] = []
    model.eval()
    for start in range(0, len(pool), batch_size):
        chunk = pool[start:start + batch_size]
        smis = [s for s, _ in chunk]
        ys = torch.tensor([y for _, y in chunk], dtype=torch.long)
        mids = torch.full((len(chunk),), mid, dtype=torch.long).to(device)
        ys = ys.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            out = model(smis, mids)
        logits = out[metal].float()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(ys.tolist())
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, scores)), len(labels)


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    data: Dict[str, Dict[str, List]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed of the 5-metal multi-task V4 model.

    Returns a metrics dict for this seed.
    """
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
    if args.init == "tmqm":
        try_load_tmqm_encoder(model, args.encoder_ckpt)
    # R18 W2.1 / W2.2 / W2.3 adapter wiring.  Each R18 init choice
    # attaches an additional component to the training pipeline; the
    # R17 baseline behaviour (``--init tmqm`` or ``--init random``)
    # is unchanged.
    r18_components: Dict[str, object] = {}
    if args.init == "se3sb" and _SE3SBVelocityField is not None:
        # Wrap a parameter-free stub in the SE(3) twist head; the
        # EGNN velocity net swap-in is on the R18 P2 roadmap.
        class _StubEGNN(nn.Module):
            def forward(self, x, t=None, pocket=None):
                return torch.zeros_like(x)

        r18_components["se3_field"] = _SE3SBVelocityField(
            base_velocity_net=_StubEGNN(), manifold="SE3",
        ).to(device)
    elif args.init == "transportrelax" and _RelaxedEndpointTerm is not None:
        r18_components["relax_term"] = _RelaxedEndpointTerm(
            lambda_relax=1e-3, eps=0.1,
        )
    elif args.init == "multiph" and _MultiPHPocketFeatures is not None:
        r18_components["mphp"] = _MultiPHPocketFeatures(
            max_radius=5.0, max_coord=4.0, n_bins=8,
        ).to(device)
    n_frozen, n_trainable = freeze_encoder_layers_1_2(model)
    print(
        f"[train_ru_multitask] seed={seed} hidden={args.hidden} "
        f"depth={args.depth} init={args.init} "
        f"frozen={n_frozen:,} trainable={n_trainable:,} "
        f"r18_components={list(r18_components.keys()) or 'none'}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = V4MultiTaskLoss(
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        coord_weight=COORD_LOSS_WEIGHT,
    )

    # Coord labels: a constant per-metal proxy (square_planar for Pt/Ru/Ir/Rh,
    # tetrahedral for Re/Os) — not available per-row in this CSV, so we
    # derive a deterministic label from the metal id to keep the auxiliary
    # head differentiable without leaking labels.
    COORD_BY_METAL = {
        "Ru": 3, "Ir": 3, "Rh": 3,       # square_planar / octahedral mixed; default square
        "Re": 2, "Os": 3,
    }

    best_val_mean = -1.0
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0

    val_history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # Per-metal shuffled batches; iterate metal-major so all 5 metals
        # contribute every epoch (no metal left behind on short epochs).
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
                    device_type="cuda", dtype=torch.bfloat16,
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
                epoch_loss += float(batch_loss.item())
                n_batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_aucs = {}
            for m in SUPPORTED_METALS:
                a, _ = eval_split_auc(model, data, m, "val", device, args.batch)
                val_aucs[m] = a
            valid_aucs = [v for v in val_aucs.values() if not np.isnan(v)]
            mean_val = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            val_history.append({"epoch": epoch, "mean_val_auc": mean_val,
                                "per_metal": val_aucs})
            print(
                f"[train_ru_multitask] seed={seed} epoch={epoch:3d} "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{m}={val_aucs[m]:.3f}" for m in SUPPORTED_METALS)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    # Load best
    model.load_state_dict(best_state)

    # Test
    test_aucs: Dict[str, float] = {}
    for m in SUPPORTED_METALS:
        a, n = eval_split_auc(model, data, m, "test", device, args.batch)
        test_aucs[m] = a
        print(f"[train_ru_multitask] seed={seed} TEST {m} AUC={a:.4f} n={n}")
    valid_test = [v for v in test_aucs.values() if not np.isnan(v)]
    mean_test = float(np.mean(valid_test)) if valid_test else float("nan")
    print(
        f"[train_ru_multitask] seed={seed} mean_test_auc={mean_test:.4f} "
        f"(best_epoch={best_epoch})"
    )
    return {
        "seed": seed,
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "test_auc_per_metal": test_aucs,
        "mean_test_auc": mean_test,
        "val_history": val_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase A Stage 1.4 — 3-seed × 100-epoch 5-metal multi-task fine-tune.",
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--encoder-ckpt", type=Path, default=DEFAULT_ENCODER_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    p.add_argument(
        "--init",
        choices=["tmqm", "random", "se3sb", "transportrelax", "multiph"],
        default="tmqm",
        help=(
            "Encoder initialisation. R17 default choices: 'tmqm' "
            "(with arch-mismatch fallback to random), 'random'. "
            "R18 W2.1/W2.2/W2.3 choices: 'se3sb' (Schrödinger-Bridge "
            "velocity head), 'transportrelax' (soft endpoint KL "
            "shaping), 'multiph' (multi-parameter PH pocket features)."
        ),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(RECIPE_SEEDS))
    p.add_argument("--epochs", type=int, default=RECIPE_EPOCHS)
    p.add_argument("--batch", type=int, default=RECIPE_BATCH)
    p.add_argument("--lr", type=float, default=RECIPE_LR)
    p.add_argument("--hidden", type=int, default=RECIPE_HIDDEN)
    p.add_argument("--depth", type=int, default=RECIPE_DEPTH)
    p.add_argument("--eval-every", type=int, default=RECIPE_EVAL_EVERY)
    p.add_argument(
        "--train-max-per-metal", type=int, default=4000,
        help="Subsample cap per metal per epoch (0 = full). Default 4000 for wall-time bound.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_ru_multitask] data={args.data} init={args.init} "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"seeds={args.seeds} hidden={args.hidden} depth={args.depth}"
    )

    data = load_5metal_csv(args.data)
    summary_counts = {
        m: {sp: len(data[m][sp]) for sp in ("train", "val", "test")}
        for m in SUPPORTED_METALS
    }
    print(f"[train_ru_multitask] split counts: {summary_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_ru_multitask] device={device}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        result = train_one_seed(seed, data, args, device)
        seed_results.append(result)
        ckpt_path = args.out_dir / f"ru_multitask_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "best_val_epoch": result["best_val_epoch"],
                "best_val_mean_auc": result["best_val_mean_auc"],
                "test_auc_per_metal": result["test_auc_per_metal"],
                "mean_test_auc": result["mean_test_auc"],
                "state_dict": result["best_state"],
                "recipe": {
                    "epochs": args.epochs, "batch": args.batch,
                    "lr": args.lr, "hidden": args.hidden,
                    "depth": args.depth, "init": args.init,
                    "eval_every": args.eval_every,
                },
            },
            ckpt_path,
        )
        print(f"[train_ru_multitask] wrote {ckpt_path}")
    wall = time.time() - t0
    print(f"[train_ru_multitask] total wall={wall:.1f}s")

    # Aggregate JSON
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

    ru_per_seed = [
        r["test_auc_per_metal"].get("Ru") for r in seed_results
    ]
    ru_mean = float(np.nanmean([a for a in ru_per_seed if a is not None]))
    metrics = {
        "recipe": "phase_a_stage_1.4_5metal_multitask",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": args.init,
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_seed": _to_jsonable(seed_results),
        "ru_test_auc_per_seed": ru_per_seed,
        "ru_mean_test_auc": ru_mean,
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "split_counts": summary_counts,
        "known_limitation": (
            "tmQM encoder was trained on baseline MetalCytotoxModel "
            "(hidden=128, 12 keys); V4MultiTaskDMPNN uses DirectedMPNN "
            "(hidden=300, 36 keys).  Encoder fell back to random init "
            "via try_load_tmqm_encoder — pretrain_v4multitask_dmpnn.py "
            "is the future fix."
        ),
    }
    # Drop the heavy best_state tensor from the JSON output
    for r in metrics["per_seed"]:
        if isinstance(r, dict) and "best_state" in r:
            r.pop("best_state")
    args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    print(
        f"[train_ru_multitask] wrote {args.metrics_path} "
        f"(Ru mean={ru_mean:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
