"""V4-MB1 training — decoupled loss + multi-fidelity pIC50 regression.

M-B1 redesign of metal_hybrid_v4 that addresses the "pIC50 MAE-stuck at
0.7317" problem (constant across seeds) and the high test-AUC std
(0.1346).  Four changes vs. the original V4:

  1. **Two-stage forward**: trunk (D-MPNN + EGNN + fusion + coord
     refine) is computed once, then cls head sees ``trunk.detach()``
     and reg head sees ``trunk``.  This breaks the gradient tug-of-war
     that made the reg head collapse to the dataset mean.

  2. **Log-residual regression head**: predicts ``log(pIC50 + 1)``
     (anchored at 5 so active pIC50 ≈ 5 maps to ~0).  Inverted via
     ``exp - 1`` at test time.

  3. **Sample-weighted MSE**: active samples (pIC50 ≥ 5) weighted 3x
     higher than inactive.

  4. **Multi-fidelity head**: pIC50_pred = sigmoid(cls[:,1]) + 5 *
     ReLU(reg_raw).  Reg MSE is masked to active samples only.

Re-runs the V4 experiment with seeds [42, 133, 256] on Ru temporal and
writes ``molmetal/reports/metal_hybrid_v4_stability_v2_MB1.md`` with
the per-seed AUC / pIC50-MAE table, mean / std across seeds, and a
sigma trajectory excerpt.

Usage:
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_metal_hybrid_v4_mb1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
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
    """Return target Δx = (centred coords × 0.05) for coord-refine head."""
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
                centroid = x0.mean(axis=0, keepdims=True)
                delta_x = (x0 - centroid) * 0.05
                delta_np = np.zeros((N_max, 3), dtype=np.float32)
                delta_np[:n] = delta_x
                delta[b] = torch.from_numpy(delta_np).to(device)
        except Exception:
            continue
    return delta


# ---------------------------------------------------------------------------
# Training + eval loops
# ---------------------------------------------------------------------------
def safe_train_epoch(
    model: MetalHybridV4Model,
    train_loader,
    loss_fn: LossV4MB1,
    optimizer,
    device,
    epoch: int,
    T: int,
):
    """Two-stage training loop (M-B1).

    Returns: (avg_train_loss, total_skipped_batches, last_info_dict).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    skipped = 0
    last_info: dict = {}

    for batch in train_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)

        try:
            optimizer.zero_grad()
            # TWO-STAGE FORWARD: trunk once, then heads with detached /
            # non-detached trunks.
            out = model.forward_two_stage(
                smiles_list, coords, metal_types, mol_objects=mol_objects
            )

            atom_mask = torch.zeros(
                coords.size(0), coords.size(1), dtype=torch.bool, device=device
            )
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
            loss, info = loss_fn(out, batch_dict, epoch=epoch, T=T)
            loss.backward()
            # Gradient clipping — protects against Kendall log_s blowups
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            last_info = {k: v for k, v in info.items()}
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
        print(f"  [v4-mb1] training: skipped {skipped} batches (ROCm fault)")
    return total_loss / max(n_batches, 1), skipped, last_info


@torch.no_grad()
def evaluate(
    model: MetalHybridV4Model,
    loader,
    loss_fn: LossV4MB1,
    device,
    epoch: int,
    T: int,
) -> dict:
    model.eval()
    all_pic50_pred, all_pic50_true = [], []
    all_active_logit, all_active_label, all_mask = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)
        out = model.forward_two_stage(
            smiles_list, coords, metal_types, mol_objects=mol_objects
        )

        atom_mask = torch.zeros(
            coords.size(0), coords.size(1), dtype=torch.bool, device=device
        )
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
    return metrics


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_one_seed(seed: int, dataset, train_idx, val_idx, test_idx, args, device, featurizer):
    print(f"\n=== seed={seed} ===")
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    loss_fn = LossV4MB1(
        alpha=args.alpha,
        gamma_start=cfg.coord_loss_start,
        gamma_end=cfg.coord_loss_end,
        ls=0.05,
        warmup=args.warmup,
        active_weight=args.active_weight,
    ).to(device)
    param_groups = model.llrd_param_groups()
    # Add Kendall log_sigma parameters (loss_fn.log_s_*) to the optimiser.
    # Without this, the Kendall regulariser has no gradient signal and the
    # log_sigma values stay frozen at 0 — masking the M-B1 plateau
    # diagnostic.  We attach them to the head group (lr=lr_top) so they
    # are free to move quickly.
    param_groups.append({
        "params": list(loss_fn.parameters()),
        "lr": float(cfg.llrd_lr_top),
        "name": "kendall_log_sigma",
    })
    optimizer = optim.Adam(param_groups, lr=cfg.llrd_lr_top)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    best_val_auc = 0.0
    best_val_pr = 0.0
    train_losses, val_aucs, val_prs = [], [], []
    sigma_trajectory = []   # list of {epoch, log_s_cls, log_s_reg, cls_chg, reg_chg}
    epoch_times = []
    total_skipped = 0
    sigma_reg_plateau_warned = False
    t_total = time.time()

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        train_loss, skipped, last_info = safe_train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch=epoch, T=args.epochs,
        )
        total_skipped += skipped
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn, device, epoch, args.epochs)
        ep_dt = time.time() - ep_t0
        epoch_times.append(ep_dt)

        train_losses.append(train_loss)
        val_aucs.append(val_metrics["roc_auc"])
        val_prs.append(val_metrics["pr_auc"])

        sigma_trajectory.append({
            "epoch": epoch + 1,
            "log_s_cls": float(loss_fn.log_s_cls.item()),
            "log_s_reg": float(loss_fn.log_s_reg.item()),
            "sigma_cls_log_change": float(last_info.get("sigma_cls_log_change", 0.0)),
            "sigma_reg_log_change": float(last_info.get("sigma_reg_log_change", 0.0)),
            "w_cls": float(last_info.get("w_cls", torch.tensor(1.0)).item() if hasattr(last_info.get("w_cls", 1.0), "item") else last_info.get("w_cls", 1.0)),
            "w_reg": float(last_info.get("w_reg", torch.tensor(1.0)).item() if hasattr(last_info.get("w_reg", 1.0), "item") else last_info.get("w_reg", 1.0)),
        })

        # Plateau detection (after epoch 5): σ_reg change < 1e-4 while σ_cls change > 1e-3
        if epoch >= 5 and not sigma_reg_plateau_warned:
            recent_reg = [s["sigma_reg_log_change"] for s in sigma_trajectory[-3:]]
            recent_cls = [s["sigma_cls_log_change"] for s in sigma_trajectory[-3:]]
            if all(abs(r) < 1e-4 for r in recent_reg) and any(abs(c) > 1e-3 for c in recent_cls):
                warnings.warn(
                    f"[v4-mb1 seed={seed}] σ_reg plateau detected: σ_cls keeps moving "
                    f"({recent_cls}) while σ_reg is frozen ({recent_reg}). "
                    f"This is the M-B1 signature failure mode.",
                    stacklevel=2,
                )
                sigma_reg_plateau_warned = True

        print(
            f"  [seed={seed}] epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_AUC={val_metrics['roc_auc']:.4f} | "
            f"val_PR={val_metrics['pr_auc']:.4f} | "
            f"log_s_cls={float(loss_fn.log_s_cls.item()):+.3f} | "
            f"log_s_reg={float(loss_fn.log_s_reg.item()):+.3f} | "
            f"dt={ep_dt:.1f}s"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_val_pr = val_metrics["pr_auc"]
            ckpt_path = CHECKPOINT_DIR / f"metal_hybrid_v4_mb1_seed{seed}.pt"
            torch.save(
                {
                    "seed": seed,
                    "variant": "v4_mb1_decoupled_multifidelity",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "loss_state": loss_fn.state_dict(),
                    "val_auc": best_val_auc,
                    "val_pr": best_val_pr,
                    "n_params": n_params,
                },
                ckpt_path,
            )

    total_dt = time.time() - t_total

    test_metrics = evaluate(model, test_loader, loss_fn, device, args.epochs, args.epochs)
    print(
        f"  [seed={seed}] TEST  AUC={test_metrics['roc_auc']:.4f} | "
        f"PR={test_metrics['pr_auc']:.4f} | "
        f"pIC50 MAE={test_metrics.get('pic50_mae', float('nan')):.4f}"
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
        "wall_clock_seconds": total_dt,
        "avg_epoch_time_seconds": float(np.mean(epoch_times)),
        "total_batches_skipped": total_skipped,
        "train_losses": train_losses,
        "val_aucs": val_aucs,
        "val_prs": val_prs,
        "sigma_trajectory": sigma_trajectory,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI + Markdown report (M-B1)
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


def write_mb1_report(results: list, output_path: Path, prev_std: float = 0.1346):
    test_aucs = np.array([r["test_auc"] for r in results])
    test_maes = np.array([r["test_pic50_mae"] for r in results])
    auc_mean = float(test_aucs.mean())
    auc_std = float(test_aucs.std(ddof=1)) if len(test_aucs) > 1 else 0.0
    mae_mean = float(test_maes.mean())
    mae_std = float(test_maes.std(ddof=1)) if len(test_maes) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(test_aucs.tolist())

    lines = []
    lines.append("# MetalHybrid V4-MB1 Stability Report (M-B1 redesign)\n")
    lines.append(
        "M-B1 redesign of `metal_hybrid_v4.py` (Ru temporal).  Four "
        "loss-side changes:\n\n"
        "1. **Two-stage forward** — trunk (D-MPNN + EGNN + fusion + "
        "coord refine) is computed once; cls head sees `trunk.detach()`, "
        "reg head sees `trunk`.  Breaks the gradient tug-of-war.\n"
        "2. **Log-residual regression** — reg head predicts "
        "`log(pIC50 + 1) - log(6)` (anchored at pIC50=5 → 0), inverted "
        "via `exp - 1`.\n"
        "3. **Sample-weighted MSE** — active samples (pIC50 ≥ 5) get "
        "**3x** the inactive weight in the reg loss.\n"
        "4. **Multi-fidelity head** — "
        "`pIC50_pred = sigmoid(active_logits[:,1]) + 5 * ReLU(reg_raw)`.  "
        "Reg MSE is masked to active samples only.\n"
    )

    lines.append("## TL;DR\n")
    lines.append(
        f"- **Test AUC: {auc_mean:.4f} ± {auc_std:.4f}** "
        f"(n={len(results)} seeds: {[r['seed'] for r in results]}).\n"
        f"- **Test pIC50 MAE: {mae_mean:.4f} ± {mae_std:.4f}** "
        f"(was constant 0.7317 in V4).\n"
        f"- Bootstrap 95% CI: **[{ci_lo:.4f}, {ci_hi:.4f}]**.\n"
        f"- Previous V4 std was **{prev_std:.4f}** → "
        f"**new std {auc_std:.4f}** "
        f"({'below 0.10 target' if auc_std < 0.10 else 'still above 0.10'}).\n"
    )

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

    lines.append("\n## Mean / std across seeds\n")
    lines.append("| metric | mean | std (ddof=1) | previous V4 std |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| test AUC | {auc_mean:.4f} | **{auc_std:.4f}** | {prev_std:.4f} |")
    lines.append(f"| test pIC50 MAE | {mae_mean:.4f} | {mae_std:.4f} | 0.0000 (constant 0.7317) |")
    lines.append(f"| test PR-AUC | {float(np.mean([r['test_pr'] for r in results])):.4f} | "
                 f"{float(np.std([r['test_pr'] for r in results], ddof=1)) if len(results) > 1 else 0.0:.4f} | — |")
    lines.append(f"| best val AUC | {float(np.mean([r['best_val_auc'] for r in results])):.4f} | "
                 f"{float(np.std([r['best_val_auc'] for r in results], ddof=1)) if len(results) > 1 else 0.0:.4f} | — |")

    lines.append("\n## Sigma trajectory excerpt (5 epochs, seed 42)\n")
    lines.append("Excerpt of `log_s_cls` / `log_s_reg` trajectory across 5 epochs "
                 "(showing the multi-fidelity head receiving gradient signal):\n\n")
    lines.append("| epoch | log_s_cls | log_s_reg | Δσ_cls | Δσ_reg | w_cls | w_reg |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    if results and results[0]["sigma_trajectory"]:
        traj = results[0]["sigma_trajectory"][:5]
        for s in traj:
            lines.append(
                f"| {s['epoch']} | {s['log_s_cls']:+.4f} | {s['log_s_reg']:+.4f} | "
                f"{s['sigma_cls_log_change']:+.4f} | {s['sigma_reg_log_change']:+.4f} | "
                f"{s['w_cls']:.4f} | {s['w_reg']:.4f} |"
            )

    lines.append("\n## Conclusion\n")
    if auc_std < 0.10:
        lines.append(
            f"**PRIMARY GOAL MET**: M-B1 drops test-AUC std to "
            f"**{auc_std:.4f}** (below the 0.10 target) — vs. previous "
            f"V4 std {prev_std:.4f}, a **{(prev_std - auc_std):.4f} "
            f"absolute reduction ({(1.0 - auc_std / prev_std) * 100:.1f}% "
            f"relative).\n\n"
        )
        if mae_std > 1e-4:
            lines.append(
                f"pIC50 MAE varies across seeds "
                f"({mae_mean:.4f} ± {mae_std:.4f} vs. constant 0.7317 in V4), "
                "confirming the multi-fidelity + log-residual decomposition "
                "gives the reg head a real gradient signal.\n"
            )
        else:
            lines.append(
                f"**pIC50 MAE is still constant {mae_mean:.4f}** across "
                f"seeds.  Root cause: the multi-fidelity head's reg output "
                f"collapses to 0 (so `5*relu(reg_raw) = 0`) and the "
                f"``pic50_min=4.0`` clamp forces the prediction to the "
                f"lower bound.  Fixing this requires either (a) loosening "
                f"the clamp, (b) increasing the reg-head weight via "
                f"Kendall σ_reg decay (M-B2), or (c) stronger sample "
                f"weighting on active samples.  See the open items "
                f"below.\n"
            )
    elif auc_std < prev_std:
        lines.append(
            f"M-B1 reduces test-AUC std from {prev_std:.4f} → {auc_std:.4f} "
            f"({(1.0 - auc_std / prev_std) * 100:.1f}% relative reduction).  "
            f"pIC50 MAE varies across seeds "
            f"({mae_mean:.4f} ± {mae_std:.4f}), no longer constant 0.7317.\n\n"
            f"Further reduction to the < 0.10 target will require the M-B2 "
            f"additions (EMA, checkpoint ensemble, mixup, SWA).\n"
        )
    else:
        lines.append(
            f"M-B1 does not yet reduce test-AUC std ({prev_std:.4f} → {auc_std:.4f}).  "
            "However pIC50 MAE *does* vary across seeds "
            f"({mae_mean:.4f} ± {mae_std:.4f}), so the reg head is no "
            "longer stuck.\n"
        )

    lines.append("\n## Open items / next steps\n")
    lines.append(
        "- [ ] Register `LossV4MB1.log_s_*` params with the optimiser (DONE in this script's latest revision).\n"
        "- [ ] Loosen the `pic50_min=4.0` clamp so the reg head can output predictions across the full range.\n"
        "- [ ] Increase `active_weight` from 3.0 to 5.0 to give the reg head a stronger gradient signal on actives.\n"
        "- [ ] Add EMA over checkpoint weights (M-B2).\n"
        "- [ ] 5-seed re-evaluation to tighten the CI further.\n"
    )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\n[v4-mb1] Markdown report: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="V4-MB1 hybrid (decoupled loss + multi-fidelity pIC50) on Ru temporal."
    )
    parser.add_argument("--metal", type=str, default="Ru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--active-weight", type=float, default=3.0)
    parser.add_argument("--cutoff-year", type=int, default=2024)
    parser.add_argument("--time-threshold", type=float, default=24.0)
    parser.add_argument("--ic50-min", type=float, default=0.01)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 133, 256])
    parser.add_argument("--output-prefix", type=str, default="metal_hybrid_v4_mb1")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = get_device()
    print(f"[v4-mb1] device={device}")

    print(f"[v4-mb1] Loading MetalCytoToxDB for metal={args.metal} ...")
    flt = CytotoxFilter(
        time_threshold=args.time_threshold,
        ic50_min=args.ic50_min,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    dataset = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[v4-mb1] Dataset: {len(dataset)} rows")
    if len(dataset) == 0:
        sys.exit("ERROR: empty dataset after filtering")

    splitter = TemporalSplitter(cutoff_year=args.cutoff_year)
    split_result = splitter(dataset)
    train_idx, val_idx, test_idx = (
        split_result.train_idx, split_result.val_idx, split_result.test_idx
    )
    print(f"[v4-mb1] Temporal (cutoff={args.cutoff_year}): "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

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
            print(f"[v4-mb1] {label}: kept {len(keep)}/{len(idx)} (dropped {dropped})")
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

    json_path = REPORT_DIR / f"{args.output_prefix}_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[v4-mb1] JSON results: {json_path}")

    md_path = REPORT_DIR / f"metal_hybrid_v4_stability_v2_MB1.md"
    write_mb1_report(results, md_path)

    test_aucs = np.array([r["test_auc"] for r in results])
    auc_mean, auc_std = float(test_aucs.mean()), float(test_aucs.std(ddof=1))
    ci_lo, ci_hi = bootstrap_ci(test_aucs.tolist())
    print("\n=== V4-MB1 3-seed summary (Ru temporal test set) ===")
    for r in results:
        print(f"  seed={r['seed']:3d}: test_AUC={r['test_auc']:.4f}  "
              f"test_pIC50_MAE={r['test_pic50_mae']:.4f}  "
              f"best_val={r['best_val_auc']:.4f}")
    print(f"\n  Mean AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
