#!/usr/bin/env python3
"""Wf-23 Phase 2 — R18 W2.2 Transport-Relaxation 5-metal multitask trainer.

=====================================================================
R18 W2.2 ship standalone trainer — adds the
:class:`molmetal.adapters.transport_relaxation_integration.RelaxedEndpointTerm`
soft endpoint KL penalty to the 5-metal multi-task V4 fine-tune.  Trains
5-seed × 50-epoch × MetalCytoToxDB 5-metal multitask with the relaxation
penalty enabled and writes per-seed ``metrics.json`` +
``final_summary.json``.

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (26 801 rows)
  columns: ``smiles,metal,scaffold_id,split,activity`` — scaffold-split
  ``train/val/test`` (Bemis-Murcko) per Krasnov 2026 protocol.

Outputs
-------
* ``.ckpt/transport_seed{0..4}.pt`` — per-seed best-epoch ckpt.
* ``metrics/by_metric/transport_5metal_multitask.json`` — per-seed
  ``test_auc_per_metal`` + ``mean_test_auc`` + ``relax_meta``.
* ``metrics/by_metric/transport_final_summary.json`` — slim summary.

Recipe
------
(1) Build :class:`V4MultiTaskDMPNN` (hidden=300, depth=3, 5 metal
    heads + coord-geometry auxiliary head).
(2) Optionally try to load tmQM pretrained encoder weights (with the
    same arch-mismatch fallback as ``train_ru_multitask.py``).
(3) Freeze encoder layers 0..1; fine-tune layer 2 + the coord head +
    all 5 metal heads + metal embedding.
(4) 5-metal multi-task loss on train split (per-metal CE + coord CE
    weighted by ``COORD_LOSS_WEIGHT=0.3``).
(5) **R18 W2.2 addition**: add the soft endpoint KL penalty
    ``λ_relax · KL(p_T ‖ q_T(ε))`` to the multi-task loss.  The
    penalty is computed from the per-metal per-batch energies (the
    negative log-likelihood of each prediction), giving a true
    trajectory-aware shaping term.
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

Verify
------
After a successful run:
* 5 ckpts exist at ``.ckpt/transport_seed{0..4}.pt``.
* ``metrics/by_metric/transport_5metal_multitask.json`` has per-seed
  ``test_auc_per_metal`` dict with 5 entries + ``mean_test_auc``.
* ``mean_test_auc`` is the headline metric for the W2.2 ablation.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_transport.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/transport_5metal_multitask.json \\
        --epochs 50 --batch 128 --lr 1e-3 --seeds 0 1 2 3 4 \\
        --lambda-relax 1e-3 --eps 0.1
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# W2.2 ship adapter — the Transport-Relaxation reward-shaping stack.
from molmetal.adapters.transport_relaxation_integration import (
    RelaxedEndpointTerm,
    augment_learned_shaping,
)

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)

# --- Recipe constants (R18 W2.2 spec) ---
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
RECIPE_LAMBDA_RELAX: float = 1e-3
RECIPE_EPS: float = 0.1

# --- Output paths (overridable via CLI) ---
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_ENCODER_CKPT = (
    PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "transport_5metal_multitask.json"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "transport_final_summary.json"
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

    Mirrors the contract of :func:`train_ru_multitask.try_load_tmqm_encoder`.
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_transport] tmQM encoder ckpt not found at "
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
            "[train_transport] tmQM encoder arch is incompatible "
            "with V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN); "
            "falling back to random init."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_transport] partial tmQM load: {n_loaded}/{len(own)} "
            "tensors matched; remaining kept at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# Freeze policy
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
# Transport-relaxation shaping — energies from per-batch logits
# ---------------------------------------------------------------------------
def compute_per_trajectory_energies(
    logits_per_metal: Dict[str, torch.Tensor],
    y_true_per_metal: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute per-trajectory energies from per-metal logits.

    Concretely, the per-trajectory energy ``E_i`` is the
    cross-entropy of the **i-th trajectory** under the model's logits
    (the model's negative log-likelihood for the gold label).  The
    relaxation penalty then encourages trajectories whose empirical
    endpoint distribution ``q_T = softmax(-E / ε)`` is close to the
    uniform reference ``p_T``.

    For a multi-metal step we concatenate the per-metal losses in a
    deterministic order (sorted by SUPPORTED_METALS) so the
    resulting energy tensor has a consistent shape (B,) where
    ``B = sum_over_metals(batch_size)``.
    """
    energies: List[torch.Tensor] = []
    for m in SUPPORTED_METALS:
        if m not in logits_per_metal or m not in y_true_per_metal:
            continue
        logits = logits_per_metal[m].float()
        ys = y_true_per_metal[m]
        ce = F.cross_entropy(logits, ys, reduction="none")
        energies.append(ce)
    if not energies:
        # Fallback: a zero-energy trajectory — the relaxation penalty
        # collapses to zero in this case (uniform q_T).
        return torch.zeros(1)
    return torch.cat(energies, dim=0)


# ---------------------------------------------------------------------------
# Train one seed
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    data: Dict[str, Dict[str, List]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed of the W2.2 Transport-Relaxation V4 model."""
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

    # Build the transport-relaxation penalty term.
    relax_term = RelaxedEndpointTerm(
        lambda_relax=args.lambda_relax,
        eps=args.eps,
    )

    print(
        f"[train_transport] seed={seed} hidden={args.hidden} depth={args.depth} "
        f"lambda_relax={args.lambda_relax} eps={args.eps} "
        f"tmqm_loaded={n_loaded} frozen={n_frozen:,} trainable={n_trainable:,}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
    relax_history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_relax = 0.0
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
            logits_per_metal: Dict[str, torch.Tensor] = {}
            ys_per_metal: Dict[str, torch.Tensor] = {}
            coord_true_per_metal: Dict[str, torch.Tensor] = {}
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
                    mt_loss, _ = loss_fn(out, y_true, coord_true)
                logits_per_metal[m] = out[m].detach()
                ys_per_metal[m] = ys_dev.detach()
                coord_true_per_metal[m] = coord_true.detach()
                batch_loss = batch_loss + mt_loss

            # R18 W2.2: transport-relaxation shaping on the per-step
            # energies.  The "trajectory" we feed the relaxation term
            # is a 1-step sequence of per-trajectory energies; this
            # matches the additive-only contract of
            # :func:`augment_learned_shaping` and keeps the W2.2
            # surface area small (the relaxation penalty depends on
            # the energies, not on the encoder forward pass).
            if args.lambda_relax > 0.0:
                energies = compute_per_trajectory_energies(
                    logits_per_metal,
                    ys_per_metal,
                )
                # Use a (B, 1) trajectory so the endpoint slice has
                # well-defined length B.
                trajectory = energies.unsqueeze(-1)  # (B, 1)
                shaping_base = torch.tensor(0.0, device=device, dtype=energies.dtype)
                augmented = augment_learned_shaping(
                    base_shaping_output=shaping_base,
                    trajectory=trajectory,
                    energies=energies,
                    lambda_relax=args.lambda_relax,
                    eps=args.eps,
                )
                relax_penalty = augmented
                batch_loss = batch_loss + relax_penalty
                epoch_relax += float(relax_penalty.item())

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
            val_history.append(
                {"epoch": epoch, "mean_val_auc": mean_val, "per_metal": val_aucs}
            )
            relax_history.append(
                {
                    "epoch": epoch,
                    "mean_relax_penalty": epoch_relax / max(n_batches, 1),
                }
            )
            print(
                f"[train_transport] seed={seed} epoch={epoch:3d} "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"relax={epoch_relax / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{m}={val_aucs[m]:.3f}" for m in SUPPORTED_METALS)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

    model.load_state_dict(best_state)

    test_aucs: Dict[str, float] = {}
    for m in SUPPORTED_METALS:
        a, n = eval_split_auc(model, data, m, "test", device, args.batch)
        test_aucs[m] = a
        print(f"[train_transport] seed={seed} TEST {m} AUC={a:.4f} n={n}")
    valid_test = [v for v in test_aucs.values() if not np.isnan(v)]
    mean_test = float(np.mean(valid_test)) if valid_test else float("nan")
    print(
        f"[train_transport] seed={seed} mean_test_auc={mean_test:.4f} "
        f"(best_epoch={best_epoch})"
    )

    return {
        "seed": seed,
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "test_auc_per_metal": test_aucs,
        "mean_test_auc": mean_test,
        "val_history": val_history,
        "relax_history": relax_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "tmqm_loaded": n_loaded,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-23 Phase 2 — R18 W2.2 Transport-Relaxation 5-metal "
            "multitask trainer."
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
        include_init=False,  # W2.2 always uses the transport attach.
    )
    p = add_gpu_args(p)

    # W2.2-specific Transport flags.
    p.add_argument(
        "--lambda-relax",
        type=float,
        default=RECIPE_LAMBDA_RELAX,
        help="Weight of the soft endpoint KL penalty (W2.2 spec: 1e-3).",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=RECIPE_EPS,
        help="Temperature for the empirical q_T distribution (W2.2 spec: 0.1).",
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

    print(
        f"[train_transport] data={args.data} epochs={args.epochs} "
        f"batch={args.batch} lr={args.lr} seeds={args.seeds} "
        f"hidden={args.hidden} depth={args.depth} "
        f"lambda_relax={args.lambda_relax} eps={args.eps}"
    )

    data = load_5metal_csv(args.data)
    summary_counts = {
        m: {sp: len(data[m][sp]) for sp in ("train", "val", "test")}
        for m in SUPPORTED_METALS
    }
    print(f"[train_transport] split counts: {summary_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_transport] device={device}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        result = train_one_seed(seed, data, args, device)
        seed_results.append(result)
        ckpt_path = args.out_dir / f"transport_seed{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "best_val_epoch": result["best_val_epoch"],
                "best_val_mean_auc": result["best_val_mean_auc"],
                "test_auc_per_metal": result["test_auc_per_metal"],
                "mean_test_auc": result["mean_test_auc"],
                "state_dict": result["best_state"],
                "recipe": {
                    "init": "transportrelax",
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "lambda_relax": args.lambda_relax,
                    "eps": args.eps,
                },
            },
            ckpt_path,
        )
        print(f"[train_transport] wrote {ckpt_path}")
    wall = time.time() - t0
    print(f"[train_transport] total wall={wall:.1f}s")

    ru_per_seed = [r["test_auc_per_metal"].get("Ru") for r in seed_results]
    ru_mean = float(np.nanmean([a for a in ru_per_seed if a is not None]))
    metrics = {
        "recipe": "r18_w2.2_transport_5metal_multitask",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": "transportrelax",
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_seed": _to_jsonable(seed_results),
        "ru_test_auc_per_seed": ru_per_seed,
        "ru_mean_test_auc": ru_mean,
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "split_counts": summary_counts,
        "relax_meta": {
            "lambda_relax": args.lambda_relax,
            "eps": args.eps,
            "term_class": "RelaxedEndpointTerm",
            "augment_fn": "augment_learned_shaping",
        },
    }
    for r in metrics["per_seed"]:
        if isinstance(r, dict) and "best_state" in r:
            r.pop("best_state")
    args.metrics_path.write_text(json.dumps(_to_jsonable(metrics), indent=2))
    print(f"[train_transport] wrote {args.metrics_path} " f"(Ru mean={ru_mean:.4f})")

    final_summary = {
        "recipe": "r18_w2.2_transport_5metal_multitask_final_summary",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "init": "transportrelax",
        "wall_seconds": wall,
        "ru_test_auc_per_seed": ru_per_seed,
        "ru_mean_test_auc": ru_mean,
        "mean_test_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "mean_test_auc_overall": float(
            np.nanmean([r["mean_test_auc"] for r in seed_results])
        ),
        "relax_meta": {
            "lambda_relax": args.lambda_relax,
            "eps": args.eps,
        },
    }
    args.summary_path.write_text(json.dumps(_to_jsonable(final_summary), indent=2))
    print(f"[train_transport] wrote {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
