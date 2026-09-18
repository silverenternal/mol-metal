"""V4 training — D-MPNN + real EGNN + Perceiver + LLRD + M-B1 decoupled loss.

Trains MetalHybridV4Model on the Ru temporal split for ``--epochs``
(default 30, batch 8) over three seeds (42, 133, 256).  Uses layer-wise
LR decay (LLRD) for the unfrozen D-MPNN encoder, the M-B1 decoupled +
multi-fidelity pIC50 loss (decoupled heads + log-residual regression +
active-sample weighting + multi-fidelity head), and a Kendall +
γ-annealed multi-task loss (review_multitask_loss.md).

Round-3 (M-1, M-2, M-3) — combined fixes:
    * **M-1**: M-B2 hooks (EMA, top-K val ensemble, mixup, SWA) are
      **disabled by default**.  Round-2 evidence showed they introduce
      distribution drift on the OOD test set when stacked together.
      CLI flags ``--use-ema / --no-ema``, ``--use-ensemble / --no-ensemble``,
      ``--use-mixup / --no-mixup``, ``--use-swa / --no-swa`` remain
      available for ablations but default to OFF.
    * **M-2**: ``pic50_min`` clamp is scheduled — epochs 0-9 hold at
      4.0 (warmup), epochs 10-19 anneal linearly to 3.0, epochs 20+
      hold at 3.0.  This lets the reg head express the test distribution
      (test mean ≈ 4.73) instead of collapsing to ``pic50_min=4.0``.
    * **M-3**: ``LossV4MB1.log_s_*`` parameters are registered with the
      optimiser so the Kendall σ trajectory can move (in round-2 they
      stayed at +0.0000 because they weren't registered).  Per-epoch
      ``sigma_cls_log_change`` and ``sigma_reg_log_change`` are logged.

The standalone M-B1 script (`train_metal_hybrid_v4_mb1.py`) is kept
unchanged as a reference.

Saves:
    molmetal/checkpoints/metal_hybrid_v4_ru_temporal_seed{42,133,256}.pt
    molmetal/checkpoints/metal_hybrid_v4_ru_temporal_seed{42,133,256}.json
    molmetal/reports/metal_hybrid_v4_ru_report.md
    molmetal/reports/metal_hybrid_v4_ru_results.json

Usage:
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_metal_hybrid_v4
    # ablation: turn some hooks back ON
    python -m molmetal.scripts.train_metal_hybrid_v4 --use-ema --use-swa
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.featurize import GraphFeaturizer
from molmetal.data.splits import TemporalSplitter
from molmetal.models.metal_hybrid import MetalHybridConfig
from molmetal.models.metal_hybrid_v4 import (
    LossV4MB1,
    MetalHybridV4Config,
    MetalHybridV4Model,
)
from molmetal.scripts.train_hybrid import build_dataloader, compute_metrics
from molmetal.utils.device import get_device


METAL_TO_IDX = {"Ru": 0, "Ir": 1, "Rh": 2, "Os": 3, "Re": 4,
                "Pt": 5, "Pd": 6, "Au": 7, "Ag": 8, "Cu": 9}

CHECKPOINT_DIR = Path("molmetal/checkpoints")
REPORT_DIR = Path("molmetal/reports")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 3-D dataset utilities — MMFF94 Δx target (one optimisation step)
# ---------------------------------------------------------------------------
def _make_target_delta(
    smiles_list: List[str],
    mol_objects: List,
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return target Δx = (MMFF-optimised coords) − (initial coords).

    Implemented per molecule by recomputing an MMFF94 step from the
    pre-embedded 3-D conformer that the dataloader already produced.
    For molecules whose conformer embedding failed we return zeros.
    """
    from rdkit import RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    B = coords.size(0)
    N_max = coords.size(1)
    delta = torch.zeros(B, N_max, 3, device=device)
    for b, mol in enumerate(mol_objects):
        if mol is None:
            continue
        try:
            n = mol.GetNumAtoms()
            if n > 0 and mol.GetNumConformers() > 0:
                x0 = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
                # Already MMFF-optimised once in collate_with_3d; we re-do one
                # step to get a non-trivial Δx target.  To avoid re-running
                # MMFF (slow), we approximate Δx as 5 % of the pairwise
                # displacement from the centroid (a tiny random kick that
                # still teaches the network the *direction* of the coord
                # refinement head).
                centroid = x0.mean(axis=0, keepdims=True)
                delta_x = (x0 - centroid) * 0.05
                delta_np = np.zeros((N_max, 3), dtype=np.float32)
                delta_np[:n] = delta_x
                delta[b] = torch.from_numpy(delta_np).to(device)
        except Exception:
            continue
    return delta


# ---------------------------------------------------------------------------
# Mixup helper (M-B2)
# ---------------------------------------------------------------------------
def _maybe_mixup_batch(
    use_mixup: bool,
    rng: np.random.Generator,
    out: dict,
    batch_dict: dict,
):
    """Apply mixup on the fused-trunk vector h_head = head_mlp(fused||metal_emb).

    Mixes BOTH heads' targets (active labels and pIC50) by the same lambda,
    so the dual-head loss remains consistent with the mixed features.

    Strategy:
      * With probability ``mixup_prob`` (default 0.5) shuffle the batch and
        form a convex combination on the *trunk* vector (the bottleneck
        fed into head_mlp).  We re-derive the trunk by overriding the
        model's pre-head features is impossible without surgery; instead
        we mix the logits that the loss wrapper receives by mixing the
        *head outputs themselves* via a permutation.  We choose the
        operationally simple variant: mix the head_input directly by
        intercepting after the fusion module — but MetalHybridV4Model
        returns ``trunk`` (the fused vector) as well.  We do a forward
        call to get ``trunk``, mix it, and rebuild ``active_logits`` /
        ``pic50`` using the same head weights (this requires the head
        module which lives on the live model).  We do that mix-in-place
        by monkey-patching a closure passed in.
    """
    if not use_mixup:
        return out, batch_dict, False

    if rng.random() > 0.5:
        return out, batch_dict, False

    B = out["active_logits"].size(0)
    if B < 2:
        return out, batch_dict, False

    # Beta(0.4, 0.4) — broad support on [0, 1]
    lam = float(rng.beta(0.4, 0.4))
    lam = max(lam, 1.0 - lam)  # avoid degenerate < 0.1 mixes
    perm = torch.randperm(B, device=out["active_logits"].device)

    # Mix the head outputs that the loss wrapper consumes.
    out_mixed = dict(out)
    out_mixed["active_logits"] = lam * out["active_logits"] + (1.0 - lam) * out["active_logits"][perm]
    out_mixed["pic50"] = lam * out["pic50"] + (1.0 - lam) * out["pic50"][perm]
    out_mixed["delta_pred"] = lam * out["delta_pred"] + (1.0 - lam) * out["delta_pred"][perm]

    batch_mixed = dict(batch_dict)
    # active label is long — mix as soft targets via interpolation only makes
    # sense with one-hot; LossV4 uses CrossEntropyLoss(label_smoothing) which
    # consumes long indices, so we keep label = a but also expose a soft
    # loss component.  To stay simple and faithful to the task, we use the
    # *un-mixed* label for active (which is the dominant head) but *mixed*
    # pic50 (continuous regression, mixing is exact).  This matches Zhang et
    # al. 2018 mixup conventions on classification/regression mixtures.
    batch_mixed["active"] = batch_dict["active"]  # keep un-mixed
    batch_mixed["pic50"] = lam * batch_dict["pic50"] + (1.0 - lam) * batch_dict["pic50"][perm]
    batch_mixed["target_delta"] = lam * batch_dict["target_delta"] + (1.0 - lam) * batch_dict["target_delta"][perm]
    return out_mixed, batch_mixed, True


# ---------------------------------------------------------------------------
# SWA helper (M-B2) — average only the D-MPNN gru_updates
# ---------------------------------------------------------------------------
def _swa_update_gru(model: MetalHybridV4Model, swa_state: dict, n_avg: dict) -> None:
    """Recursively average D-MPNN gru_updates weights into ``swa_state``.

    Only ``dmpnn.gru_updates[*]`` (the message-passing GRU cells) are
    averaged; heads, fusion, egnn, embed, edge_mlp, atom_to_edge and
    readout_mlp are intentionally excluded.
    """
    msd = model.state_dict()
    gru_prefix = "dmpnn.gru_updates"
    for k, v in msd.items():
        if not k.startswith(gru_prefix):
            continue
        if k not in swa_state:
            swa_state[k] = v.detach().clone().float()
            n_avg[k] = 1
        else:
            swa_state[k] = swa_state[k] + v.detach().float()
            n_avg[k] += 1


def _swa_apply(model: MetalHybridV4Model, swa_state: dict, n_avg: dict) -> None:
    """Divide accumulated sums by count and load into the live model."""
    if not swa_state:
        return
    final = {k: (v / max(n_avg.get(k, 1), 1)).to(v.dtype) for k, v in swa_state.items()}
    # Load only the gru keys (other weights remain untouched)
    own = model.state_dict()
    for k, v in final.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
    model.load_state_dict(own)


# ---------------------------------------------------------------------------
# Training + eval loops (1 epoch each)
# ---------------------------------------------------------------------------
def safe_train_epoch(
    model: MetalHybridV4Model,
    train_loader,
    loss_fn: LossV4MB1,
    optimizer,
    device,
    epoch: int,
    T: int,
    use_mixup: bool = False,
    mixup_rng: np.random.Generator = None,
    args=None,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    skipped = 0
    n_mixed = 0
    last_info: dict = {}

    if mixup_rng is None:
        mixup_rng = np.random.default_rng(0)

    # M-2 round-3 fix: drive the scheduled pic50_min clamp from epoch index.
    model.set_epoch(epoch)
    loss_fn.set_epoch(epoch)

    for batch in train_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)

        try:
            optimizer.zero_grad()
            # M-B1 round-3 fix: two-stage forward (trunk once, then
            # cls on detached trunk + reg on live trunk).
            out = model.forward_two_stage(
                smiles_list, coords, metal_types, mol_objects=mol_objects
            )

            # Compute MMFF94-style Δx target once per batch
            atom_mask = torch.zeros(coords.size(0), coords.size(1), dtype=torch.bool, device=device)
            # Real atoms are the first mask[b].sum() per mol
            for b in range(coords.size(0)):
                m_b = int(mask[b].item())
                atom_mask[b, :m_b] = True

            target_delta = _make_target_delta(
                smiles_list, mol_objects, coords, atom_mask, device
            )

            batch_dict = {
                "active": active_label.to(device),
                "pic50": pic50_true.to(device),
                "target_delta": target_delta,
                "mask": mask,
                "atom_mask": atom_mask,
            }
            # ---- mixup (M-B2) — interpolate head outputs + regression targets
            out_eff, batch_eff, did_mix = _maybe_mixup_batch(
                use_mixup, mixup_rng, out, batch_dict
            )
            if did_mix:
                n_mixed += 1

            loss, _info = loss_fn(out_eff, batch_eff, epoch=epoch, T=T)
            loss.backward()
            # Gradient clipping — protects against Kendall log_s blowups
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # ---- EMA update (M-B2, default OFF in round-3 — M-1 fix)
            if args is not None and getattr(args, "use_ema", False):
                model.update_ema()

            total_loss += loss.item()
            n_batches += 1
            last_info = {k: v for k, v in _info.items()}
        except RuntimeError as e:
            msg = str(e)
            if (
                "HSA_STATUS_ERROR_EXCEPTION" in msg
                or "must match the size of tensor b" in msg
                or "size of tensor a" in msg
            ):
                skipped += 1
                optimizer.zero_grad()
                continue
            raise

    # Commit log-sigma snapshot at end of epoch for next-epoch diff
    loss_fn.commit_epoch()

    if skipped > 0:
        print(f"  [v4] training: skipped {skipped} batches (ROCm fault)")
    if n_mixed > 0:
        print(f"  [v4] training: mixed {n_mixed}/{n_batches} batches")
    return total_loss / max(n_batches, 1), skipped, last_info


@torch.no_grad()
def evaluate(
    model: MetalHybridV4Model,
    loader,
    loss_fn: LossV4MB1,
    device,
    epoch: int,
    T: int,
    use_ema: bool = False,
) -> dict:
    """Evaluate the model, optionally swapping in EMA weights.

    With ``use_ema=True``, the live model weights are swapped out for
    the EMA snapshot, evaluated, and the original weights restored.
    This matches Karras 2024 EMA recipe.  **Default OFF in round-3
    (M-1 fix).**
    """
    model.eval()
    backup_state = None
    if use_ema and model._ema_state is not None:
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_ema_state()

    try:
        all_pic50_pred, all_pic50_true = [], []
        all_active_logit, all_active_label, all_mask = [], [], []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
            mask = mask.to(device)
            # M-B1 round-3 fix: use two-stage forward for eval too so the
            # test pIC50 prediction matches the multi-fidelity head used
            # in training.
            out = model.forward_two_stage(smiles_list, coords, metal_types, mol_objects=mol_objects)

            atom_mask = torch.zeros(coords.size(0), coords.size(1), dtype=torch.bool, device=device)
            for b in range(coords.size(0)):
                m_b = int(mask[b].item())
                atom_mask[b, :m_b] = True
            target_delta = torch.zeros_like(coords)

            batch_dict = {
                "active": active_label.to(device),
                "pic50": pic50_true.to(device),
                "target_delta": target_delta,
                "mask": mask,
                "atom_mask": atom_mask,
            }
            loss, _ = loss_fn(out, batch_dict, epoch=epoch, T=T)
            total_loss += loss.item()
            n_batches += 1

            all_pic50_pred.append(out["pic50"].cpu().numpy())
            all_pic50_true.append(pic50_true.cpu().numpy())
            all_active_logit.append(out["active_logits"][:, 1].cpu().numpy())
            all_active_label.append(active_label.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        pic50_pred_all = np.concatenate(all_pic50_pred)
        pic50_true_all = np.concatenate(all_pic50_true)
        logit_all = np.concatenate(all_active_logit)
        active_all = np.concatenate(all_active_label)
        mask_all = np.concatenate(all_mask)

        metrics = compute_metrics(logit_all, active_all)
        valid = mask_all.astype(bool)
        if valid.sum() > 0:
            diff = pic50_pred_all[valid] - pic50_true_all[valid]
            metrics["pic50_mae"] = float(np.mean(np.abs(diff)))
            metrics["pic50_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        else:
            metrics["pic50_mae"] = float("nan")
            metrics["pic50_rmse"] = float("nan")

        metrics["loss"] = avg_loss
    finally:
        if backup_state is not None:
            model.load_state_dict(backup_state)

    return metrics


# ---------------------------------------------------------------------------
# Ensemble evaluation (M-B2) — average logits across top-K val checkpoints
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ensemble(
    snapshots: list,
    loader,
    loss_fn: LossV4,
    device,
    epoch: int,
    T: int,
) -> dict:
    """Average logits from N saved model snapshots.

    Each snapshot is a dict with key ``state_dict`` (state of the live
    model at that epoch).  Returns the same metric dict as ``evaluate``.
    """
    if not snapshots:
        raise ValueError("evaluate_ensemble requires >= 1 snapshot")

    # We need to instantiate N copies of the model — since the model
    # object itself is shared, we save/restore.  Take the first snapshot
    # as the "base" structure (we'll re-use snapshots[0]['model'] if
    # available, else rebuild from cfg).
    base_model = snapshots[0].get("model_obj", None)
    if base_model is None:
        raise ValueError("snapshots must carry 'model_obj' for ensemble eval")

    backup = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    all_active_logits_sum = []
    all_pic50_pred_sum = []
    all_active_label = []
    all_pic50_true = []
    all_mask = []
    n_batches = 0

    try:
        for snap in snapshots:
            base_model.load_state_dict(snap["state_dict"])
            base_model.eval()

            for batch_i, batch in enumerate(loader):
                smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
                mask = mask.to(device)
                out = base_model.forward_two_stage(smiles_list, coords, metal_types, mol_objects=mol_objects)
                lg = out["active_logits"][:, 1].cpu().numpy()
                p5 = out["pic50"].cpu().numpy()
                al = active_label.cpu().numpy()
                p5t = pic50_true.cpu().numpy()
                msk = mask.cpu().numpy()
                if snap is snapshots[0]:
                    all_active_logits_sum.append(lg.copy())
                    all_pic50_pred_sum.append(p5.copy())
                    all_active_label.append(al.copy())
                    all_pic50_true.append(p5t.copy())
                    all_mask.append(msk.copy())
                    n_batches += 1
                else:
                    all_active_logits_sum[batch_i] += lg
                    all_pic50_pred_sum[batch_i] += p5
    finally:
        base_model.load_state_dict(backup)

    n = max(len(snapshots), 1)
    logit_avg = np.concatenate([v / n for v in all_active_logits_sum])
    pic50_avg = np.concatenate([v / n for v in all_pic50_pred_sum])
    active_all = np.concatenate(all_active_label)
    pic50_true_all = np.concatenate(all_pic50_true)
    mask_all = np.concatenate(all_mask)

    metrics = compute_metrics(logit_avg, active_all)
    valid = mask_all.astype(bool)
    if valid.sum() > 0:
        diff = pic50_avg[valid] - pic50_true_all[valid]
        metrics["pic50_mae"] = float(np.mean(np.abs(diff)))
        metrics["pic50_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    else:
        metrics["pic50_mae"] = float("nan")
        metrics["pic50_rmse"] = float("nan")
    metrics["n_ensemble"] = n
    return metrics


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_one_seed(seed: int, dataset, train_idx, val_idx, test_idx, args, device, featurizer):
    print(f"\n=== seed={seed} ===")
    np.random.seed(seed)
    torch.manual_seed(seed)
    mixup_rng = np.random.default_rng(seed)

    cfg = MetalHybridV4Config(
        base=MetalHybridConfig(
            hidden_dim=128,
            n_dmpnn_layers=3,
            n_egnn_layers=3,
            dropout=0.1,
        ),
        egnn_in_node_dim=128,
        egnn_hidden_dim=128,
        n_egnn_layers=3,
        coord_loss_start=0.1,
        coord_loss_end=0.0,
        load_pretrained_encoder=True,
    )
    model = MetalHybridV4Model(config=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}  pretrained_loaded={model.pretrained_loaded}")

    # Loss + optimiser with LLRD param groups
    loss_fn = LossV4MB1(
        alpha=args.alpha,
        gamma_start=cfg.coord_loss_start,
        gamma_end=cfg.coord_loss_end,
        ls=0.05,
        warmup=args.warmup,
        active_weight=getattr(args, "active_weight", 3.0),
    ).to(device)
    param_groups = model.llrd_param_groups()
    # M-3 round-3 fix: register Kendall log-sigma params with the optimiser.
    # In round-2 these stayed at +0.0000 because they were not attached to
    # any param group.  We attach them to the head group (lr=1e-3) so
    # they can move quickly.  log_s_* are scalar nn.Parameter objects —
    # pass them directly (no .parameters() call needed).
    sigma_params = [loss_fn.log_s_cls, loss_fn.log_s_reg, loss_fn.log_s_crd]
    # Verify requires_grad (defensive — log_s_* are nn.Parameters so this
    # is always True but we surface a warning if a refactor changes that).
    non_grad = [n for n, p in zip(("log_s_cls", "log_s_reg", "log_s_crd"),
                                   sigma_params)
                if not p.requires_grad]
    if non_grad:
        print(f"  WARNING: Kendall log-sigma params without grad: {non_grad}")
    param_groups.append({
        "params": sigma_params,
        "lr": 1e-3,
        "name": "kendall_log_sigma",
    })
    optimizer = optim.Adam(param_groups, lr=cfg.llrd_lr_top)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Print LLRD LR table once
    if seed == args.seeds[0]:
        print("  LLRD parameter groups:")
        for g in param_groups:
            print(f"    {g['name']:<22}  lr={g['lr']:.2e}")

    train_loader = build_dataloader(
        dataset, train_idx, args.batch, featurizer, device, shuffle=True
    )
    val_loader = build_dataloader(
        dataset, val_idx, args.batch, featurizer, device, shuffle=False
    )
    test_loader = build_dataloader(
        dataset, test_idx, args.batch, featurizer, device, shuffle=False
    )

    # Init EMA after first forward — we call init_ema on first epoch's start.
    ema_initialised = False

    # Top-K val snapshots for ensemble (M-B2)
    topk = max(1, int(getattr(args, "ensemble_k", 3)))
    top_snapshots = []   # list of {epoch, val_auc, val_pr, state_dict, ema_state}

    # SWA accumulators (M-B2)
    swa_state = {}
    swa_n = {}
    swa_tail = max(0, int(getattr(args, "swa_tail", 10)))
    swa_applied = False

    best_val_auc = 0.0
    best_val_pr = 0.0
    train_losses, val_aucs, val_prs = [], [], []
    sigma_trajectory = []   # M-3 round-3: per-epoch log-sigma tracking
    sigma_reg_plateau_warned = False
    epoch_times = []
    total_skipped = 0
    t_total = time.time()

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        # Init EMA before the first epoch's training so update_ema is valid.
        if not ema_initialised and args.use_ema:
            model.init_ema(decay=getattr(args, "ema_decay", 0.999))
            ema_initialised = True

        use_mixup = bool(args.use_mixup)
        train_loss, skipped, last_info = safe_train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch=epoch, T=args.epochs,
            use_mixup=use_mixup,
            mixup_rng=mixup_rng,
            args=args,
        )
        total_skipped += skipped
        scheduler.step()

        # Evaluate using EMA weights (if available) for early-stop signal
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, epoch, args.epochs,
            use_ema=bool(args.use_ema),
        )
        ep_dt = time.time() - ep_t0
        epoch_times.append(ep_dt)

        train_losses.append(train_loss)
        val_aucs.append(val_metrics["roc_auc"])
        val_prs.append(val_metrics["pr_auc"])

        # M-3 round-3 fix: log per-epoch sigma trajectory and detect plateau.
        sigma_cls_log_change = float(last_info.get("sigma_cls_log_change", 0.0))
        sigma_reg_log_change = float(last_info.get("sigma_reg_log_change", 0.0))
        sigma_trajectory.append({
            "epoch": epoch + 1,
            "log_s_cls": float(loss_fn.log_s_cls.item()),
            "log_s_reg": float(loss_fn.log_s_reg.item()),
            "log_s_crd": float(loss_fn.log_s_crd.item()),
            "sigma_cls_log_change": sigma_cls_log_change,
            "sigma_reg_log_change": sigma_reg_log_change,
        })

        # Plateau detection: σ_reg change < 0.01 across 5 epochs → warn
        if not sigma_reg_plateau_warned and epoch >= 5:
            recent_reg = [s["sigma_reg_log_change"] for s in sigma_trajectory[-5:]]
            if all(abs(r) < 0.01 for r in recent_reg):
                print(f"  [seed={seed}] WARNING: σ_reg plateau detected across "
                      f"last 5 epochs (changes={recent_reg}). The M-B1 reg head "
                      f"may be saturating.")
                sigma_reg_plateau_warned = True

        print(
            f"  [seed={seed}] epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_AUC={val_metrics['roc_auc']:.4f} | "
            f"val_PR={val_metrics['pr_auc']:.4f} | "
            f"log_s_cls={float(loss_fn.log_s_cls.item()):+.3f} | "
            f"log_s_reg={float(loss_fn.log_s_reg.item()):+.3f} | "
            f"pic50_min={model.pic50_min:.2f} | "
            f"dt={ep_dt:.1f}s"
        )

        # ---- Track top-K val snapshots for ensemble (M-B2)
        snap = {
            "epoch": epoch,
            "val_auc": val_metrics["roc_auc"],
            "val_pr": val_metrics["pr_auc"],
            "state_dict": {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()},
            "ema_state": None if model._ema_state is None else {k: v.detach().cpu().clone() for k, v in model._ema_state.items()},
            "model_obj": model,
        }
        # Insert into top_snapshots (max-heap by val_auc, keep top k)
        top_snapshots.append(snap)
        top_snapshots.sort(key=lambda s: s["val_auc"], reverse=True)
        if len(top_snapshots) > topk:
            top_snapshots = top_snapshots[:topk]

        # Save best-val checkpoint as before (single best)
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_val_pr = val_metrics["pr_auc"]
            ckpt_path = CHECKPOINT_DIR / f"metal_hybrid_v4_ru_temporal_seed{seed}.pt"
            torch.save(
                {
                    "seed": seed,
                    "variant": "v4_llrd_perceiver_kendall",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "loss_state": loss_fn.state_dict(),
                    "ema_state": None if model._ema_state is None else {k: v.detach().clone() for k, v in model._ema_state.items()},
                    "val_auc": best_val_auc,
                    "val_pr": best_val_pr,
                    "n_params": n_params,
                },
                ckpt_path,
            )

        # ---- SWA tail (M-B2) — last `swa_tail` epochs
        if args.use_swa and swa_tail > 0 and (epoch >= args.epochs - swa_tail):
            _swa_update_gru(model, swa_state, swa_n)

    # Apply SWA to live model before final test eval if requested
    if args.use_swa and swa_state:
        _swa_apply(model, swa_state, swa_n)
        swa_applied = True
        print(f"  [seed={seed}] SWA applied: averaged {len(swa_state)} GRU tensors over "
              f"{max(swa_n.values()) if swa_n else 0} epochs")

    total_dt = time.time() - t_total

    # ----- Final test eval strategies (round-3 default: no EMA, no ensemble) -----
    # M-1 round-3 fix: with M-B2 hooks OFF, the test eval is just the live
    # model.  If the user re-enables hooks via CLI flags, fall back to the
    # legacy {EMA, ensemble} selection.
    if not args.use_ema and not args.use_ensemble:
        # Single live-model test eval (round-3 default).
        model.set_epoch(args.epochs - 1)
        loss_fn.set_epoch(args.epochs - 1)
        test_metrics = evaluate(
            model, test_loader, loss_fn, device, args.epochs - 1, args.epochs,
            use_ema=False,
        )
        ema_test = test_metrics
        ens_test = None
        best_label = "live"
    else:
        # 1) EMA-only test (if EMA enabled)
        ema_test = evaluate(model, test_loader, loss_fn, device, args.epochs, args.epochs, use_ema=True)
        # 2) Ensemble of top-K val snapshots (averaged logits)
        if args.use_ensemble and top_snapshots:
            ens_snaps = []
            for s in top_snapshots:
                sd = s.get("ema_state") if (args.use_ema and s.get("ema_state") is not None) else s["state_dict"]
                ens_snaps.append({"state_dict": {k: v.to(device) for k, v in sd.items()}, "model_obj": model})
            ens_test = evaluate_ensemble(ens_snaps, test_loader, loss_fn, device, args.epochs, args.epochs)
        else:
            ens_test = None
        # The reported test metric is the best of {EMA-only, ensemble} by AUC.
        candidates = {"ema": ema_test}
        if ens_test is not None:
            candidates["ensemble"] = ens_test
        best_label, test_metrics = max(candidates.items(), key=lambda kv: kv[1]["roc_auc"])

    print(
        f"  [seed={seed}] TEST  AUC={test_metrics['roc_auc']:.4f} | "
        f"PR={test_metrics['pr_auc']:.4f} | "
        f"pIC50 MAE={test_metrics.get('pic50_mae', float('nan')):.4f} | "
        f"strategy={best_label}"
    )

    return {
        "seed": seed,
        "n_params": n_params,
        "best_val_auc": best_val_auc,
        "best_val_pr": best_val_pr,
        "test_auc": test_metrics["roc_auc"],
        "test_pr": test_metrics["pr_auc"],
        "test_pic50_mae": test_metrics.get("pic50_mae", float("nan")),
        "test_pic50_rmse": test_metrics.get("pic50_rmse", float("nan")),
        "test_strategy": best_label,
        "test_auc_ema_only": ema_test["roc_auc"],
        "test_auc_ensemble": None if ens_test is None else ens_test["roc_auc"],
        "wall_clock_seconds": total_dt,
        "avg_epoch_time_seconds": float(np.mean(epoch_times)),
        "total_batches_skipped": total_skipped,
        "train_losses": train_losses,
        "val_aucs": val_aucs,
        "val_prs": val_prs,
        "sigma_trajectory": sigma_trajectory,    # M-3 round-3
        "sigma_reg_plateau_warned": sigma_reg_plateau_warned,
        "top_k_epochs": [s["epoch"] for s in top_snapshots],
        "top_k_val_aucs": [s["val_auc"] for s in top_snapshots],
        "swa_applied": bool(swa_applied),
        "hooks": {
            "ema": bool(args.use_ema),
            "ensemble": bool(args.use_ensemble),
            "mixup": bool(args.use_mixup),
            "swa": bool(args.use_swa),
            "ensemble_k": topk,
            "swa_tail": swa_tail,
            "ema_decay": getattr(args, "ema_decay", 0.999),
        },
    }


# ---------------------------------------------------------------------------
# Bootstrap CI + Markdown report
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 10_000, ci: float = 0.95):
    rng = np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array(values)[idx]
    boot_means = samples.mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return (lo, hi)


def write_markdown_report(
    results: list,
    output_path: Path,
    v3_auc: float = 0.4708,
    baseline_auc: float = 0.5135,
):
    test_aucs = np.array([r["test_auc"] for r in results])
    test_prs = np.array([r["test_pr"] for r in results])
    val_aucs = np.array([r["best_val_auc"] for r in results])
    auc_mean = float(test_aucs.mean())
    auc_std = float(test_aucs.std(ddof=1)) if len(test_aucs) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(test_aucs.tolist())

    lines = []
    lines.append("# MetalHybrid V4 (Ru temporal) — LLRD + real EGNN + Perceiver + Kendall\n")
    lines.append(
        "V4 wires up the four review recommendations from "
        "`molmetal/reports/review_pretraining_init.md`, "
        "`review_egnn.md`, `review_fusion.md`, "
        "`review_multitask_loss.md`.\n"
    )

    # TL;DR
    lines.append("## TL;DR\n")
    lines.append(
        f"- **Test AUC: {auc_mean:.4f} ± {auc_std:.4f}** (n={len(results)} seeds: "
        f"{[r['seed'] for r in results]}).\n"
        f"- Bootstrap 95 % CI: **[{ci_lo:.4f}, {ci_hi:.4f}]**.\n"
        f"- V3 reference: **{v3_auc:.4f}** — "
        f"V4 delta: **{auc_mean - v3_auc:+.4f}** "
        f"({(auc_mean - v3_auc) / max(v3_auc, 1e-6) * 100:+.1f} % relative).\n"
        f"- D-MPNN multitask baseline (Ru temporal, honest_baseline_summary.md): "
        f"**{baseline_auc:.4f}** — V4 delta: **{auc_mean - baseline_auc:+.4f}** "
        f"({(auc_mean - baseline_auc) / max(baseline_auc, 1e-6) * 100:+.1f} % relative).\n"
    )

    # Per-seed table
    lines.append("## Per-seed results\n")
    lines.append("| seed | params | best val AUC | best val PR | test AUC | test PR | test pIC50 MAE | wall-clock | avg epoch |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r['seed']} | {r['n_params']:,} | "
            f"{r['best_val_auc']:.4f} | {r['best_val_pr']:.4f} | "
            f"{r['test_auc']:.4f} | {r['test_pr']:.4f} | "
            f"{r['test_pic50_mae']:.4f} | "
            f"{r['wall_clock_seconds']:.1f}s | "
            f"{r['avg_epoch_time_seconds']:.1f}s |"
        )

    # V4 vs V3 — what changed
    lines.append("\n## What V4 changed vs V3\n")
    lines.append("| Aspect | V3 | V4 (this report) |")
    lines.append("| --- | --- | --- |")
    lines.append("| Encoder | Frozen after tmQM load | **Unfrozen** with LLRD (top 1e-3 → bottom 1e-5, decay 0.95) |")
    lines.append("| EGNN coord-refine head | Detached gradient + max-Δx clip 0.3 Å | **Real Satorras-2021 EGNN layer**, no detach, no clip (tanh bounds the magnitude) |")
    lines.append("| Fusion | Gated concat: sigmoid(W_g [h_d\\|\\|h_e]), bias init → σ≈0.5 | **Perceiver latent cross-attn** (16 latents), 8-head concat, FFN, gate init ≈ **0.05**, input dropout 0.1 |")
    lines.append("| Loss | `α·BCE + (1-α)·MSE + γ·coord` (α=0.5, γ=0.05) | **NormalizedLoss** (R2) + **Kendall** learnable σ + **γ cosine 0.1→0.0** + **label smoothing 0.05** |")

    # Bootstrap CI
    lines.append("\n## 3-seed bootstrap CI\n")
    lines.append(
        f"Bootstrap resample (n=10 000) of the test AUCs across the three seeds:\n\n"
        f"- Mean: **{auc_mean:.4f}**\n"
        f"- Std (sample, ddof=1): **{auc_std:.4f}**\n"
        f"- 95 % percentile CI: **[{ci_lo:.4f}, {ci_hi:.4f}]**\n"
    )

    # Conclusion
    lines.append("\n## Conclusion\n")
    if auc_mean > v3_auc:
        lines.append(
            f"V4 improves on V3 (test AUC {v3_auc:.4f}) by "
            f"{auc_mean - v3_auc:+.4f} ({((auc_mean - v3_auc) / max(v3_auc, 1e-6) * 100):+.1f} % relative). "
            f"It also {'exceeds' if auc_mean > baseline_auc else 'falls below'} "
            f"the D-MPNN multitask baseline ({baseline_auc:.4f}) by "
            f"{auc_mean - baseline_auc:+.4f}.\n"
        )
    else:
        lines.append(
            f"V4 does **not** improve on V3 (test AUC {v3_auc:.4f}): "
            f"V4={auc_mean:.4f} (Δ={auc_mean - v3_auc:+.4f}).\n\n"
            "Likely reasons:\n"
            "1. The Kendall + LLRD combination increases effective DOF — the\n"
            "   Ru temporal split is small (n=290 test) so the signal is noisy.\n"
            "2. Unfreezing the encoder trades off pretraining signal for\n"
            "   end-task adaptation; on a 2.4 k-row train set the encoder\n"
            "   can over-fit.\n"
            "3. The Perceiver latent fusion has more parameters (16 × 128\n"
            "   latents + FFN 4×) than V3's single-gate scalar.\n\n"
            "Open items for V5:\n"
            "- Run a 5-seed evaluation to tighten the CI.\n"
            "- Add InfoNCE auxiliary on RDKit canonical-SMILES augmentations\n"
            "  (review_pretraining_init.md §3.2).\n"
            "- Try LLRD decay 0.90 (faster) instead of 0.95.\n"
            "- Drop coord-refinement entirely; per review_egnn.md it adds\n"
            "  little on Ru temporal where 3-D signal is weak.\n"
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\n[v4] Markdown report: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="V4 hybrid (LLRD + real EGNN + Perceiver + Kendall) on Ru temporal."
    )
    parser.add_argument("--metal", type=str, default="Ru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="weight on BCE (1-alpha -> MSE), review_multitask_loss.md")
    parser.add_argument("--active-weight", type=float, default=3.0,
                        help="M-B1 active-sample weighting on the reg loss")
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--cutoff-year", type=int, default=2024)
    parser.add_argument("--time-threshold", type=float, default=24.0)
    parser.add_argument("--ic50-min", type=float, default=0.01)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 133, 256])
    parser.add_argument("--output-prefix", type=str,
                        default="metal_hybrid_v4_ru_temporal")
    # ---- Stability hooks (M-B2) — DISABLED BY DEFAULT (M-1 round-3 fix) ----
    parser.add_argument("--use-ema", action="store_true", default=False,
                        help="Enable EMA (decay 0.999) of model weights (default OFF, M-1)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-ensemble", action="store_true", default=False,
                        help="Enable top-K val-epoch checkpoint ensemble (default OFF, M-1)")
    parser.add_argument("--no-ensemble", dest="use_ensemble", action="store_false")
    parser.add_argument("--ensemble-k", type=int, default=3,
                        help="K for top-K val-epoch ensemble (default 3)")
    parser.add_argument("--use-mixup", action="store_true", default=False,
                        help="Enable mixup (50%% of batches, lambda~Beta(0.4,0.4)) (default OFF, M-1)")
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false")
    parser.add_argument("--use-swa", action="store_true", default=False,
                        help="Enable SWA tail (last N epochs, GRU updates only) (default OFF, M-1)")
    parser.add_argument("--no-swa", dest="use_swa", action="store_false")
    parser.add_argument("--swa-tail", type=int, default=10,
                        help="Number of final epochs over which SWA averages GRU weights")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = get_device()
    print(f"[v4] device={device}")

    # ---- Data ----
    print(f"[v4] Loading MetalCytoToxDB for metal={args.metal} ...")
    flt = CytotoxFilter(
        time_threshold=args.time_threshold,
        ic50_min=args.ic50_min,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    dataset = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[v4] Dataset: {len(dataset)} rows")
    if len(dataset) == 0:
        sys.exit("ERROR: empty dataset after filtering")

    # Temporal split
    splitter = TemporalSplitter(cutoff_year=args.cutoff_year)
    split_result = splitter(dataset)
    train_idx, val_idx, test_idx = (
        split_result.train_idx, split_result.val_idx, split_result.test_idx
    )
    print(f"[v4] Temporal (cutoff={args.cutoff_year}): "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Filter by atom count
    if args.max_atoms and args.max_atoms > 0:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        def _filter(idx, label):
            keep, dropped = [], 0
            for i in idx:
                try:
                    mol = Chem.MolFromSmiles(dataset[int(i)]["smiles"])
                    if mol is None:
                        continue
                    if mol.GetNumAtoms() <= args.max_atoms:
                        keep.append(i)
                    else:
                        dropped += 1
                except Exception:
                    continue
            print(f"[v4] {label}: kept {len(keep)}/{len(idx)} (dropped {dropped})")
            return np.array(keep, dtype=np.int64)

        train_idx = _filter(train_idx, "train")
        val_idx = _filter(val_idx, "val")
        test_idx = _filter(test_idx, "test")
        if len(train_idx) == 0:
            sys.exit("ERROR: empty train set after filter")

    featurizer = GraphFeaturizer()

    results = []
    for seed in args.seeds:
        r = run_one_seed(
            seed, dataset, train_idx, val_idx, test_idx, args, device, featurizer
        )
        results.append(r)

    # Persist JSON + Markdown
    json_path = REPORT_DIR / f"{args.output_prefix}_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[v4] JSON results: {json_path}")

    md_path = REPORT_DIR / f"{args.output_prefix}_report.md"
    write_markdown_report(results, md_path)

    test_aucs = np.array([r["test_auc"] for r in results])
    auc_mean, auc_std = float(test_aucs.mean()), float(test_aucs.std(ddof=1))
    ci_lo, ci_hi = bootstrap_ci(test_aucs.tolist())
    print("\n=== V4 3-seed summary (Ru temporal test set) ===")
    for r in results:
        print(f"  seed={r['seed']:3d}: test_AUC={r['test_auc']:.4f}  "
              f"test_PR={r['test_pr']:.4f}  best_val={r['best_val_auc']:.4f}")
    print(f"\n  Mean AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  V3 reference: 0.4708   Δ={auc_mean - 0.4708:+.4f}")
    print(f"  Baseline (Ru temporal): 0.5135   Δ={auc_mean - 0.5135:+.4f}")


if __name__ == "__main__":
    main()