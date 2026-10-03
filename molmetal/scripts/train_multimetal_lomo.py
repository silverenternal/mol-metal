#!/usr/bin/env python3
"""Wf-32 Phase 2 — Multi-metal Leave-One-Metal-Out (LOMO) cross-validation.

=====================================================================
**NEW trainer (Wf-32)** that runs leave-one-metal-out (LOMO) cross
validation across the 5 precious metals (Ru, Ir, Rh, Re, Os) using
the Wf-31 R18 M22 SE(3) SB 5-metal multitask backbone.

This script is a **wrapper / aggregator** of the Wf-31 deliverables
in :mod:`molmetal.scripts.train_ru_5metal_se3sb.py`, not a
replacement.  The Wf-22 close-out manifest locks the Wf-31 trainer
and we **must not** modify it.  This script:

* **Imports** :class:`Ru5MetalSe3SBHead` + :class:`Ru5MetalSe3SBLoss`
  + the rest of the Wf-31 surface area.
* **Adds a LOMO splitter** that, for each held-out metal M, restricts
  training to the *other 4 metals* and evaluates on M.  This is
  stricter than the Wf-31 scaffold split (which holds out by
  Bemis-Murcko scaffold).
* **Supports a new init mode** ``hilbert`` (Wf-9 P1-4) on top of the
  Wf-31 init modes ``tmqm`` and ``se3sb`` — the Hilbert adapter wires
  :class:`HilbertASTSRComposite` as a reward channel that is
  consumed by the per-metal reg head (the regression branch of the
  Wf-31 wrapper).
* **Reports per-fold AUC** (5 folds × N seeds) plus **mean ± std**
  across the 5×N=25 runs, the Wf-32 headline metric.

The Wf-32 headline target is **mean LOMO test AUC** (5 folds × 5
seeds = 25 runs), to be compared against the R17 baseline
``metal_lomo_auc.json`` value of ``0.535`` (Wf-32 SPEC).

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (26 802 rows)
  columns: ``smiles,metal,scaffold_id,split,activity`` — already
  scaffold-split train/val/test per Krasnov 2026 protocol.

Outputs
-------
* ``.ckpt/multimetal_lomo_{init}_leave{metal}_seed{seed}.pt`` — per
  (held-out metal, seed) best ckpt.
* ``metrics/by_metric/multimetal_lomo_auc.json`` — per-fold
  ``test_auc_per_seed`` (5 seeds) + ``per_fold_mean_auc`` +
  ``per_fold_std_auc`` + ``mean_test_auc`` (across 25 runs) +
  ``wf32_headline`` string.
* ``metrics/by_metric/multimetal_lomo_auc_summary.json`` — slim
  final report (per-fold AUC + headline + recipe).

Recipe
------
(1) For each held-out metal M ∈ {Ru, Ir, Rh, Re, Os}:
    - Build the 5-metal multitask backbone (V4 + 5 cls + 5 reg
      heads), but mask out the M head at training time (its data
      never enters train/val).
    - The remaining 4 metals' data goes to train/val.
    - Test is restricted to rows where ``metal == M`` (the held-out
      metal).
(2) 5 seeds {0..4} × 50 epochs (default), batch=128, lr=1e-3, AdamW,
    cosine annealing, BF16 autocast when CUDA present.
(3) Eval every 5 epochs on val (restricted to 4 train metals); pick
    best epoch by mean val AUC across 4 metals.
(4) Final test: per-fold AUC on the held-out metal + 5-seed
    mean ± std.
(5) Aggregate: 5 folds × 5 seeds = 25 runs; report mean ± std.

Constraint
----------
**DO NOT MODIFY** ``train_ru_5metal_se3sb.py`` — Wf-31 deliverable
locked per Wf-22 close-out manifest.  This script is a **NEW
deliverable** (Wf-32 Phase 2) that imports from it.

Verify
------
After a successful run:
* 25 ckpts exist at ``.ckpt/multimetal_lomo_{init}_leave{metal}_seed{0..4}.pt``.
* ``metrics/by_metric/multimetal_lomo_auc.json`` records per-fold
  ``test_auc_per_seed`` (5 seeds) + ``per_fold_mean_auc`` +
  ``per_fold_std_auc`` + headline ``mean_test_auc``.
* ``metrics/by_metric/multimetal_lomo_auc_summary.json`` records
  the Wf-32 headline ``mean_lomo_auc_25run`` (across 25 runs =
  5 folds × 5 seeds).

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_multimetal_lomo.py \\
        --data .data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/multimetal_lomo_auc.json \\
        --summary-path metrics/by_metric/multimetal_lomo_auc_summary.json \\
        --init se3sb --epochs 50 --batch 128 --lr 1e-3 \\
        --seeds 0 1 2 3 4 \\
        --folds leave-Ru-out leave-Ir-out leave-Rh-out leave-Re-out leave-Os-out
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Wf-31 deliverables — V4 backbone + 5 cls + 5 reg head wrapper +
# loss + CSV loader + tmQM encoder fallback.
from molmetal.models.v4_multitask_dmpnn import (
    COORD_LOSS_WEIGHT,
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
)

import train_ru_5metal_se3sb as _w31_mod  # type: ignore[import-not-found]  # noqa: F401  — exposed for downstream introspection
from train_ru_5metal_se3sb import (      # type: ignore[import-not-found]
    COORD_BY_METAL,
    RECIPE_BATCH,
    RECIPE_DEPTH,
    RECIPE_DROPOUT,
    RECIPE_EVAL_EVERY,
    RECIPE_HIDDEN,
    RECIPE_LR,
    RECIPE_METAL_EMBED_DIM,
    RECIPE_N_COORD_CLASSES,
    REG_LOSS_WEIGHT,
    Ru5MetalSe3SBHead,
    Ru5MetalSe3SBLoss,
    attach_init_components,
    build_batches,
    build_se3_twist_head,
    cap_pool,
    gather_train_pool,
    load_5metal_csv,
    try_load_tmqm_encoder,
)

# W2.1 SE(3) Schrödinger-Bridge velocity field — used by ``--init se3sb``.
from molmetal.adapters.se3_sb_integration import (
    SE3SBVelocityField,  # noqa: F401 — re-exported for downstream tests.
    integrate_with_cfm,  # noqa: F401 — re-exported for downstream tests.
)

# Wf-9 P1-4 Hilbert primal-dual composite — used by ``--init hilbert``.
# Lazy import inside ``attach_hilbert_components`` so the cold-start
# path (``--init tmqm`` / ``--init se3sb``) does not require the
# optional PySR dependency.
try:                                            # pragma: no cover
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite as _HilbertASTSRComposite,
    )
except Exception:                               # pragma: no cover
    _HilbertASTSRComposite = None               # type: ignore[assignment]

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)

# --- Recipe constants (Wf-32 Phase 2 spec) ---
RECIPE_EPOCHS: int = 50
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
HEAD_HIDDEN: int = 128
HILBERT_DIM: int = 64

#: Default held-out metals — exactly the 5-way canonical subset.
DEFAULT_HELD_OUT_METALS: Tuple[str, ...] = SUPPORTED_METALS

#: Wf-32 init modes — 3-way ablation (tmqm baseline + se3sb SE(3)
#: Schrödinger-Bridge + hilbert primal-dual composite).
INIT_CHOICES: Tuple[str, ...] = ("tmqm", "se3sb", "hilbert")

#: Default folds — leave-Ru-out … leave-Os-out.
DEFAULT_FOLDS: Tuple[str, ...] = tuple(
    f"leave-{m}-out" for m in DEFAULT_HELD_OUT_METALS
)


# ---------------------------------------------------------------------------
# Helpers — fold parsing
# ---------------------------------------------------------------------------
def parse_fold_token(fold: str) -> str:
    """Parse a ``leave-{metal}-out`` token → ``metal`` (str).

    Accepts ``leave-Ru-out`` / ``leave-Ru`` / ``Ru`` (any
    case-insensitive prefix); raises ``ValueError`` on malformed
    input.  The returned metal uses the canonical capitalisation
    (``Ru`` / ``Ir`` / ``Rh`` / ``Re`` / ``Os``) to match
    ``SUPPORTED_METALS``.
    """
    if fold is None:
        raise ValueError("fold token cannot be None")
    f = fold.strip()
    if not f:
        raise ValueError("fold token cannot be empty")
    upper = f.upper()
    # Accept canonical ``leave-{metal}-out`` / ``leave-{metal}``.
    if upper.startswith("LEAVE-") and upper.endswith("-OUT"):
        m = upper[len("LEAVE-"):-len("-OUT")]
    elif upper.startswith("LEAVE-"):
        m = upper[len("LEAVE-"):]
    else:
        m = upper
    # Match case-insensitively against SUPPORTED_METALS; return the
    # canonical capitalisation (Ru / Ir / …).
    match = next((sm for sm in SUPPORTED_METALS if sm.upper() == m), None)
    if match is None:
        raise ValueError(
            f"fold token {fold!r} → metal {m!r} not in "
            f"SUPPORTED_METALS {SUPPORTED_METALS!r}"
        )
    return match


def split_folds(folds: List[str]) -> List[str]:
    """Parse a list of fold tokens → list of metals (no dups, validate)."""
    seen: List[str] = []
    for f in folds:
        m = parse_fold_token(f)
        if m not in seen:
            seen.append(m)
    if not seen:
        raise ValueError("at least one fold must be supplied")
    return seen


# ---------------------------------------------------------------------------
# LOMO data view — mask the held-out metal out of train/val
# ---------------------------------------------------------------------------
def build_lomo_view(
    data: Dict[str, Dict[str, List]],
    held_out_metal: str,
) -> Tuple[Dict[str, Dict[str, List]], Dict[str, int]]:
    """Restrict the 5-metal data dict so the held-out metal is *only*
    present in ``test`` (never in ``train`` / ``val``).

    Returns
    -------
    lomo_view : dict
        ``{metal: {split: [(smiles, label)]}}`` — train / val
        restricted to the 4 non-held-out metals; test restricted to
        the held-out metal.
    counts : dict
        ``{metal: {split: n_rows}}`` — diagnostic counts per
        (metal, split).
    """
    if held_out_metal not in SUPPORTED_METALS:
        raise ValueError(
            f"held_out_metal {held_out_metal!r} not in "
            f"SUPPORTED_METALS {SUPPORTED_METALS!r}"
        )
    lomo_view: Dict[str, Dict[str, List]] = {
        m: {"train": [], "val": [], "test": []} for m in SUPPORTED_METALS
    }
    counts: Dict[str, int] = {}
    for m in SUPPORTED_METALS:
        for sp in ("train", "val", "test"):
            if m == held_out_metal:
                if sp == "test":
                    pool = data[m][sp]
                else:
                    pool = []
            else:
                if sp == "test":
                    pool = []
                else:
                    pool = data[m][sp]
            lomo_view[m][sp] = list(pool)
            counts[f"{m}/{sp}"] = len(pool)
    return lomo_view, counts


# ---------------------------------------------------------------------------
# Hilbert init — wire the Wf-9 P1-4 composite as a reg-head side-channel
# ---------------------------------------------------------------------------
def attach_hilbert_components(
    wrapper: Ru5MetalSe3SBHead,
    device: torch.device,
    population_dim: int = HILBERT_DIM,
    n_iters: int = 32,
) -> Dict[str, object]:
    """Attach the Wf-9 P1-4 Hilbert primal-dual composite as a reg
    head side-channel.

    The composite is a *reward channel* — it does not modify the
    backbone / heads directly.  We attach it as an attribute so
    downstream code (e.g. the verify script) can introspect the
    wiring.  The composite itself is a CPU numpy-only optimiser
    (see :mod:`molmetal.molmetal_lam.reward.hilbert_primal_dual`).

    Parameters
    ----------
    wrapper : Ru5MetalSe3SBHead
        The Wf-31 5-metal cls+reg head wrapper.
    device : torch.device
        Currently CPU; the Hilbert composite is a CPU-only
        optimizer.
    population_dim : int, default ``64``
        Hilbert space dimension (matches the coupling adapter 32 → 64
        bridge in ``coupling_adapter.py``).
    n_iters : int, default ``32``
        Number of primal-dual iterations to run on the Hilbert side
        when :meth:`fit` is called.  Bounded small so the wiring
        does not dominate wall-time.

    Returns
    -------
    components : dict
        ``{"hilbert_composite": HilbertASTSRComposite}`` on success;
        ``{}`` if the optional Hilbert module is unavailable in the
        venv (with a clear warning).
    """
    if _HilbertASTSRComposite is None:
        warnings.warn(
            "[train_multimetal_lomo] HilbertASTSRComposite adapter "
            "unavailable; hilbert init reduces to tmqm fallback."
        )
        return {}
    composite = _HilbertASTSRComposite(
        use_hilbert=True,
        population_dim=int(population_dim),
    )
    wrapper.hilbert_composite = composite          # type: ignore[attr-defined]
    wrapper.hilbert_n_iters = int(n_iters)         # type: ignore[attr-defined]
    return {"hilbert_composite": composite}


# ---------------------------------------------------------------------------
# Eval helper — per-fold ROC-AUC (test restricted to held-out metal)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_lomo_test_auc(
    wrapper: Ru5MetalSe3SBHead,
    lomo_view: Dict[str, Dict[str, List]],
    held_out_metal: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, int]:
    """Compute ROC-AUC on the held-out metal's test rows.

    Mirrors ``eval_split_auc`` from the Wf-31 trainer, but the test
    pool is restricted to ``lomo_view[held_out_metal]["test"]``.
    """
    pool = [(s, y) for s, y in lomo_view[held_out_metal]["test"] if y in (0, 1)]
    if not pool:
        return float("nan"), 0
    mid = wrapper.metal_id(held_out_metal)
    scores: List[float] = []
    labels: List[int] = []
    wrapper.eval()
    for start in range(0, len(pool), batch_size):
        chunk = pool[start:start + batch_size]
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
        logits = out["cls"][held_out_metal].float()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(ys.tolist())
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, scores)), len(labels)


@torch.no_grad()
def eval_lomo_val_auc(
    wrapper: Ru5MetalSe3SBHead,
    lomo_view: Dict[str, Dict[str, List]],
    train_metals: Tuple[str, ...],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, Dict[str, float]]:
    """Compute mean val AUC across the 4 train metals (held-out
    metal excluded from val)."""
    val_aucs: Dict[str, float] = {}
    for m in train_metals:
        pool = [(s, y) for s, y in lomo_view[m]["val"] if y in (0, 1)]
        if not pool:
            val_aucs[m] = float("nan")
            continue
        mid = wrapper.metal_id(m)
        scores: List[float] = []
        labels: List[int] = []
        wrapper.eval()
        for start in range(0, len(pool), batch_size):
            chunk = pool[start:start + batch_size]
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
            logits = out["cls"][m].float()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(ys.tolist())
        if len(set(labels)) < 2:
            val_aucs[m] = float("nan")
        else:
            val_aucs[m] = float(roc_auc_score(labels, scores))
    valid = [v for v in val_aucs.values() if not np.isnan(v)]
    mean_val = float(np.mean(valid)) if valid else float("nan")
    return mean_val, val_aucs


# ---------------------------------------------------------------------------
# SE(3) integration smoke — mirrors the Wf-31 smoke
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
            samples, twist_head, n_steps=n_steps, method="midpoint",
        )
        out_norm = float(x_final.norm().item())
        finite = bool(torch.isfinite(x_final).all().item())
    except Exception as exc:                                # pragma: no cover
        return {"out_norm": float("nan"), "finite": False, "error": str(exc)}
    return {"out_norm": out_norm, "finite": finite}


# ---------------------------------------------------------------------------
# Hilbert fit smoke — runs a tiny PD pass on a constant feature / target
# ---------------------------------------------------------------------------
def hilbert_smoke(composite, n_rows: int = 4, n_feats: int = 8) -> Dict[str, object]:
    """Run a tiny Hilbert primal-dual fit to confirm wiring is alive."""
    try:
        import numpy as _np
        feats = _np.zeros((n_rows, n_feats), dtype=float)
        feats[:, 0] = 1.0
        tgt = _np.array([0.1, 0.2, 0.3, 0.4], dtype=float)[:n_rows]
        t0 = time.perf_counter()
        result = composite.fit(feats, tgt)
        wall = float(time.perf_counter() - t0)
        return {
            "wall_s": wall,
            "r2": float(result.get("r2", float("nan"))),
            "backend": str(result.get("backend", "?")),
            "active_fitted": bool(composite.active_fitted()),
        }
    except Exception as exc:                                # pragma: no cover
        return {"wall_s": float("nan"), "r2": float("nan"), "error": str(exc)}


# ---------------------------------------------------------------------------
# Train one (fold, seed) — LOMO split + Wf-31 backbone + init ablation
# ---------------------------------------------------------------------------
def train_one_fold_seed(
    held_out_metal: str,
    seed: int,
    data: Dict[str, Dict[str, List]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one (fold, seed) and return the per-fold AUC on the held-out metal."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_metals: Tuple[str, ...] = tuple(
        m for m in SUPPORTED_METALS if m != held_out_metal
    )

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

    wrapper = Ru5MetalSe3SBHead(
        backbone=backbone,
        metals=SUPPORTED_METALS,
        head_hidden_dim=HEAD_HIDDEN,
        head_dropout=RECIPE_DROPOUT,
    ).to(device)

    # Attach init components.  ``tmqm`` / ``se3sb`` use the Wf-31
    # helper.  ``hilbert`` is the Wf-32 new branch.
    r18_components: Dict[str, object] = {}
    if args.init == "hilbert":
        r18_components = attach_hilbert_components(
            wrapper, device, population_dim=HILBERT_DIM, n_iters=32,
        )
    else:
        r18_components = attach_init_components(
            args.init, wrapper, device, hidden_dim=args.hidden,
        )

    n_frozen, n_trainable = _freeze_encoder_layers_1_2(backbone)
    print(
        f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
        f"seed={seed} init={args.init} train_metals={train_metals} "
        f"frozen={n_frozen:,} trainable={n_trainable:,} "
        f"components={list(r18_components.keys()) or 'none'}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = Ru5MetalSe3SBLoss(
        n_coord_classes=RECIPE_N_COORD_CLASSES,
        coord_weight=COORD_LOSS_WEIGHT,
        reg_weight=REG_LOSS_WEIGHT,
    )

    # Build LOMO view — held-out metal only in test.
    lomo_view, counts = build_lomo_view(data, held_out_metal)
    print(
        f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
        f"counts={counts}"
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
                cap_pool(gather_train_pool(lomo_view, m),
                         args.train_max_per_metal, rng),
                wrapper.metal_id(m),
                args.batch,
                rng,
                shuffle=True,
            )
            for m in train_metals
        }
        n_steps = max(len(b) for b in per_metal_batches.values())
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            batch_loss: Optional[torch.Tensor] = None
            for m in train_metals:
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
                    wrapper.parameters(), max_norm=5.0,
                )
                optimizer.step()
                epoch_loss += float(batch_loss.item())
                n_batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mean_val, val_aucs = eval_lomo_val_auc(
                wrapper, lomo_view, train_metals, device, args.batch,
            )
            val_history.append({
                "epoch": epoch,
                "mean_val_auc": mean_val,
                "per_metal": val_aucs,
            })
            print(
                f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
                f"seed={seed} epoch={epoch:3d} "
                f"loss={epoch_loss / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{m}={val_aucs[m]:.3f}" for m in train_metals)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in wrapper.state_dict().items()
                }

    wrapper.load_state_dict(best_state)
    test_auc, n_test = eval_lomo_test_auc(
        wrapper, lomo_view, held_out_metal, device, args.batch,
    )
    print(
        f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
        f"seed={seed} TEST {held_out_metal} AUC={test_auc:.4f} n_test={n_test}"
    )

    # Per-init smoke — SE(3) integration for ``--init se3sb``, Hilbert
    # PD fit for ``--init hilbert``.
    se3_smoke: Dict[str, object] = {}
    hilbert_smoke_out: Dict[str, object] = {}
    if args.init == "se3sb" and hasattr(wrapper, "se3_field"):
        se3_smoke = se3_integration_smoke(wrapper.se3_field)
        print(
            f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
            f"seed={seed} se3_smoke={se3_smoke}"
        )
    if args.init == "hilbert" and hasattr(wrapper, "hilbert_composite"):
        hilbert_smoke_out = hilbert_smoke(wrapper.hilbert_composite)
        print(
            f"[train_multimetal_lomo] fold=leave-{held_out_metal}-out "
            f"seed={seed} hilbert_smoke={hilbert_smoke_out}"
        )

    return {
        "fold": held_out_metal,
        "seed": seed,
        "init": args.init,
        "test_auc": test_auc,
        "n_test": n_test,
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "train_metals": list(train_metals),
        "val_history": val_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "components": list(r18_components.keys()),
        "se3_smoke": se3_smoke,
        "hilbert_smoke": hilbert_smoke_out,
        "best_state": best_state,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Internal helper — mirrors ``train_ru_5metal_se3sb.freeze_encoder_layers_1_2``
# ---------------------------------------------------------------------------
def _freeze_encoder_layers_1_2(model: V4MultiTaskDMPNN) -> Tuple[int, int]:
    """Freeze DirectedMPNN message-passing layers 0..1; layer 2 + heads + (when in se3sb) the SE(3) twist head remain trainable."""
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    for i in (0, 1):
        for p in model.encoder.dmpnn.gru_updates[i].parameters():
            p.requires_grad_(False)
        for p in model.encoder.dmpnn.edge_mlp[i].parameters():
            p.requires_grad_(False)
        for p in model.encoder.dmpnn.atom_to_edge[i].parameters():
            p.requires_grad_(False)
    n_frozen = n_trainable = 0
    for p in model.encoder.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
    return n_frozen, n_trainable


# ---------------------------------------------------------------------------
# Aggregation — 5-fold × 5-seed statistics
# ---------------------------------------------------------------------------
def aggregate_per_fold(
    fold_results: Dict[str, List[Dict[str, object]]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-fold (5-seed mean, 5-seed std) over the fold→results dict."""
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for m, rs in fold_results.items():
        vals = [
            float(r["test_auc"]) for r in rs
            if not np.isnan(float(r["test_auc"]))
        ]
        if not vals:
            means[m] = float("nan")
            stds[m] = float("nan")
        else:
            means[m] = float(np.mean(vals))
            stds[m] = float(np.std(vals, ddof=0))
    return means, stds


def aggregate_overall(
    fold_results: Dict[str, List[Dict[str, object]]],
) -> Tuple[float, float]:
    """Mean ± std across all 25 runs (5 folds × 5 seeds)."""
    vals = [
        float(r["test_auc"]) for rs in fold_results.values() for r in rs
        if not np.isnan(float(r["test_auc"]))
    ]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


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
            "Wf-32 Phase 2 — Multi-metal LOMO CV across 5 precious metals. "
            "5 folds × 5 seeds × 3 init modes (tmqm / se3sb / hilbert)."
        ),
    )
    # Wf-35 Phase 2 — shared recipe / GPU / benchmark args.
    # ``include_init=False`` because Wf-32 extends the init choices
    # with ``hilbert`` (Wf-9 P1-4) on top of the shared ``tmqm /
    # random / se3sb / transport / multiph`` set; we register our
    # own --init action below so the union of choices is accepted.
    p = add_common_args(
        p,
        data_default=(
            PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
        ),
        encoder_ckpt_default=(
            PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
        ),
        out_dir_default=PROJECT_ROOT / ".ckpt",
        metrics_path_default=(
            PROJECT_ROOT / "metrics" / "by_metric" / "multimetal_lomo_auc.json"
        ),
        summary_path_default=(
            PROJECT_ROOT / "metrics" / "by_metric" / "multimetal_lomo_auc_summary.json"
        ),
        include_init=False,
    )
    p.add_argument(
        "--init",
        choices=list(INIT_CHOICES),
        default="se3sb",
        help=(
            "Encoder initialisation. Wf-32 ablation modes: "
            "'tmqm' (R17 baseline fallback, with arch-mismatch "
            "warning), 'se3sb' (Wf-31 W2.1 SE(3) "
            "Schrödinger-Bridge velocity head), 'hilbert' (Wf-9 "
            "P1-4 Hilbert primal-dual composite side-channel)."
        ),
    )
    p = add_gpu_args(p)

    # Wf-32-specific LOMO flags (metals + folds + SE(3) steps).
    p.add_argument(
        "--metals",
        type=str,
        nargs="+",
        default=list(SUPPORTED_METALS),
        help=(
            "Subset of metals to consider (the head pool). "
            "Folds are restricted to this subset; each held-out "
            "metal must be in the head pool.  Default: all 5 "
            "(Ru Ir Rh Re Os)."
        ),
    )
    p.add_argument(
        "--folds",
        type=str,
        nargs="+",
        default=list(DEFAULT_FOLDS),
        help=(
            "Fold tokens — ``leave-{metal}-out`` / ``leave-{metal}`` / "
            "``{metal}`` (case-insensitive).  Default: the 5 canonical "
            "leave-Ru-out … leave-Os-out tokens."
        ),
    )
    p.add_argument(
        "--se3-n-steps", type=int, default=8,
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

    # Resolve the held-out metal list — order preserved, deduped,
    # validated, intersected with the head pool.
    head_pool = tuple(dict.fromkeys(args.metals))
    unknown_head = [m for m in head_pool if m not in SUPPORTED_METALS]
    if unknown_head:
        raise ValueError(
            f"Unknown metals {unknown_head!r}; supported = "
            f"{SUPPORTED_METALS!r}"
        )
    held_out_metals = split_folds(args.folds)
    bad = [m for m in held_out_metals if m not in head_pool]
    if bad:
        raise ValueError(
            f"Held-out metals {bad!r} not in head pool {head_pool!r}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_multimetal_lomo] data={args.data} init={args.init} "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"seeds={args.seeds} folds={held_out_metals} "
        f"hidden={args.hidden} depth={args.depth}"
    )

    data = load_5metal_csv(args.data)
    summary_counts = {
        m: {sp: len(data[m][sp]) for sp in ("train", "val", "test")}
        for m in head_pool
    }
    print(f"[train_multimetal_lomo] split counts: {summary_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_multimetal_lomo] device={device}")

    t0 = time.time()
    fold_results: Dict[str, List[Dict[str, object]]] = {
        m: [] for m in held_out_metals
    }
    for held_out in held_out_metals:
        for seed in args.seeds:
            result = train_one_fold_seed(
                held_out, seed, data, args, device,
            )
            fold_results[held_out].append(result)
            ckpt_path = (
                args.out_dir
                / f"multimetal_lomo_{args.init}_leave{held_out}_seed{seed}.pt"
            )
            torch.save(
                {
                    "held_out": held_out,
                    "seed": seed,
                    "init": args.init,
                    "test_auc": result["test_auc"],
                    "n_test": result["n_test"],
                    "best_val_epoch": result["best_val_epoch"],
                    "best_val_mean_auc": result["best_val_mean_auc"],
                    "train_metals": result["train_metals"],
                    "components": result["components"],
                    "se3_smoke": result["se3_smoke"],
                    "hilbert_smoke": result["hilbert_smoke"],
                    "state_dict": result["best_state"],
                    "recipe": {
                        "init": args.init,
                        "epochs": args.epochs, "batch": args.batch,
                        "lr": args.lr, "hidden": args.hidden,
                        "depth": args.depth,
                        "se3_n_steps": args.se3_n_steps,
                        "folds": list(held_out_metals),
                        "seeds": list(args.seeds),
                        "head_pool": list(head_pool),
                    },
                },
                ckpt_path,
            )
            print(
                f"[train_multimetal_lomo] wrote {ckpt_path}"
            )
    wall = time.time() - t0
    print(f"[train_multimetal_lomo] total wall={wall:.1f}s")

    per_fold_mean, per_fold_std = aggregate_per_fold(fold_results)
    overall_mean, overall_std = aggregate_overall(fold_results)

    # Drop the heavy best_state tensor from the JSON output.
    for rs in fold_results.values():
        for r in rs:
            if isinstance(r, dict) and "best_state" in r:
                r.pop("best_state")

    metrics = {
        "recipe": "wf32_m19_multimetal_lomo_cv",
        "n_seeds": len(args.seeds),
        "n_folds": len(held_out_metals),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": args.init,
        "folds": list(held_out_metals),
        "head_pool": list(head_pool),
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_fold": {
            m: {
                "test_auc_per_seed": [
                    float(r["test_auc"]) for r in fold_results[m]
                ],
                "test_auc_per_seed_n_test": [
                    int(r["n_test"]) for r in fold_results[m]
                ],
                "best_val_mean_auc_per_seed": [
                    float(r["best_val_mean_auc"]) for r in fold_results[m]
                ],
                "best_val_epoch_per_seed": [
                    int(r["best_val_epoch"]) for r in fold_results[m]
                ],
                "val_history_per_seed": [
                    r["val_history"] for r in fold_results[m]
                ],
                "components_per_seed": [
                    r["components"] for r in fold_results[m]
                ],
                "se3_smoke_per_seed": [r["se3_smoke"] for r in fold_results[m]],
                "hilbert_smoke_per_seed": [
                    r["hilbert_smoke"] for r in fold_results[m]
                ],
            }
            for m in held_out_metals
        },
        "per_fold_mean_auc": per_fold_mean,
        "per_fold_std_auc": per_fold_std,
        "mean_test_auc_overall": overall_mean,
        "std_test_auc_overall": overall_std,
        "n_runs": sum(len(rs) for rs in fold_results.values()),
        "split_counts": summary_counts,
        "wf32_headline": (
            "mean LOMO test AUC across 5 folds × 5 seeds = 25 runs; "
            "compared against R17 baseline metal_lomo_auc.json = 0.535"
        ),
        "r17_baseline_lomo_auc": 0.535,
        "known_limitation": (
            "tmQM encoder was trained on baseline MetalCytotoxModel "
            "(hidden=128, 12 keys); V4MultiTaskDMPNN uses DirectedMPNN "
            "(hidden=300, 36 keys).  Encoder fell back to random init "
            "via try_load_tmqm_encoder — pretrain_v4multitask_dmpnn.py "
            "is the future fix."
        ),
    }
    args.metrics_path.write_text(
        json.dumps(_to_jsonable(metrics), indent=2)
    )
    print(
        f"[train_multimetal_lomo] wrote {args.metrics_path} "
        f"(overall mean ± std = {overall_mean:.4f} ± {overall_std:.4f})"
    )

    # Slim final summary — Wf-32 headline metric.
    final_summary = {
        "recipe": "wf32_m19_multimetal_lomo_cv_final_summary",
        "n_seeds": len(args.seeds),
        "n_folds": len(held_out_metals),
        "epochs": args.epochs,
        "init": args.init,
        "folds": list(held_out_metals),
        "head_pool": list(head_pool),
        "wall_seconds": wall,
        "per_fold_mean_auc": per_fold_mean,
        "per_fold_std_auc": per_fold_std,
        "mean_test_auc_overall": overall_mean,
        "std_test_auc_overall": overall_std,
        "r17_baseline_lomo_auc": 0.535,
        "wf32_headline": (
            "mean(LOMO test AUC across 25 runs) — beat R17 0.535"
        ),
    }
    args.summary_path.write_text(
        json.dumps(_to_jsonable(final_summary), indent=2)
    )
    print(f"[train_multimetal_lomo] wrote {args.summary_path}")
    # Return ``(exit_code, args_namespace)`` so the
    # :func:`MEASURED_vs_PROJECTED_decorator` wrapper uses the
    # already-parsed namespace (it would otherwise re-parse argv
    # with a *different* ``argparse.ArgumentParser`` that registers
    # ``--folds-seed`` via :func:`add_benchmark_args`, which would
    # collide with our ``--folds`` token).
    return 0, args


if __name__ == "__main__":
    raise SystemExit(main())
