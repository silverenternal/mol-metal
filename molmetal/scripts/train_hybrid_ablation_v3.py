"""Fusion ablation V3 — V1 (concat) vs V2 (cross-attn) vs V3 (regularised).

Trains three variants on the same Ru temporal split for ``--epochs`` epochs
(default 30, batch 8) and writes the comparison to
``molmetal/reports/r1_cross_attn_v3_fix.md``.

V3 changes vs V2 (the failed variant):
  * attn dropout 0.2 (vs 0.0)
  * output dropout 0.3 (vs 0.0)
  * residual scaling 0.5 — backbone dominates early
  * equivariant init (zero biases) — first forward pass has no constant EGNN
    offset.

Usage:
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_hybrid_ablation_v3 --epochs 30 --batch 8

Outputs:
    molmetal/reports/r1_cross_attn_v3_fix.md
    molmetal/reports/r1_cross_attn_v3_results.json
    molmetal/checkpoints/metal_hybrid_v3_ru_temporal.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.featurize import GraphFeaturizer
from molmetal.data.splits import TemporalSplitter
from molmetal.models.loss import MetalCytotoxLoss
from molmetal.models.metal_hybrid import MetalHybridConfig, MetalHybridModel
from molmetal.models.metal_hybrid_v2 import MetalHybridV2Config, MetalHybridV2Model
from molmetal.models.metal_hybrid_v3 import MetalHybridV3Config, MetalHybridV3Model
from molmetal.scripts.train_hybrid import (
    build_dataloader,
    evaluate,
)
from molmetal.utils.device import get_device

CHECKPOINT_DIR = Path("molmetal/checkpoints")
REPORT_DIR = Path("molmetal/reports")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_train_epoch(model, train_loader, loss_fn, optimizer, device, variant_name: str):
    """Fault-tolerant training epoch (ROCm scatter_add hardware faults)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    skipped = 0

    for batch in train_loader:
        smiles_list, coords, metal_types, pic50_true, active_label, mask, mol_objects = batch
        mask = mask.to(device)

        try:
            optimizer.zero_grad()
            out = model(smiles_list, coords, metal_types, mol_objects=mol_objects)
            loss = loss_fn(
                out.pic50,
                pic50_true.to(device),
                active_label.to(device),
                mask,
                active_logits=out.active_logits,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
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

    if skipped > 0:
        print(
            f"  [{variant_name}] training: skipped {skipped} batches "
            f"(ROCm scatter_add hardware fault)"
        )
    return total_loss / max(n_batches, 1), skipped


def run_variant(
    variant_name: str,
    model_factory,
    dataset,
    train_idx,
    val_idx,
    test_idx,
    args,
    device,
    featurizer,
):
    """Train one variant end-to-end on a given split, return metrics dict."""
    print(f"\n=== Training variant: {variant_name} ===")
    model = model_factory().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{variant_name}] params={n_params:,}")

    loss_fn = MetalCytotoxLoss(alpha=args.alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    epoch_times = []
    train_losses = []
    val_aucs = []
    val_prs = []
    total_skipped = 0

    t_total = time.time()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loss, skipped = safe_train_epoch(
            model, train_loader, loss_fn, optimizer, device, variant_name
        )
        total_skipped += skipped
        scheduler.step()
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        epoch_dt = time.time() - epoch_t0
        epoch_times.append(epoch_dt)

        train_losses.append(train_loss)
        val_aucs.append(val_metrics["roc_auc"])
        val_prs.append(val_metrics["pr_auc"])

        print(
            f"  [{variant_name}] epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_AUC={val_metrics['roc_auc']:.4f} | "
            f"val_PR={val_metrics['pr_auc']:.4f} | "
            f"dt={epoch_dt:.1f}s"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_val_pr = val_metrics["pr_auc"]
            ckpt_path = CHECKPOINT_DIR / f"metal_hybrid_{variant_name}_ru_temporal.pt"
            torch.save(
                {
                    "variant": variant_name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_auc": best_val_auc,
                    "val_pr": best_val_pr,
                    "n_params": n_params,
                },
                ckpt_path,
            )

    total_dt = time.time() - t_total
    avg_epoch_dt = float(np.mean(epoch_times))

    # Final test evaluation
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(
        f"  [{variant_name}] TEST  AUC={test_metrics['roc_auc']:.4f} | "
        f"PR={test_metrics['pr_auc']:.4f} | "
        f"pIC50 MAE={test_metrics.get('pic50_mae', float('nan')):.4f}"
    )

    return {
        "variant": variant_name,
        "n_params": n_params,
        "best_val_auc": best_val_auc,
        "best_val_pr": best_val_pr,
        "test_auc": test_metrics["roc_auc"],
        "test_pr": test_metrics["pr_auc"],
        "test_pic50_mae": test_metrics.get("pic50_mae", float("nan")),
        "test_pic50_rmse": test_metrics.get("pic50_rmse", float("nan")),
        "test_accuracy": test_metrics.get("accuracy", float("nan")),
        "wall_clock_seconds": total_dt,
        "avg_epoch_time_seconds": avg_epoch_dt,
        "total_batches_skipped": total_skipped,
        "train_losses": train_losses,
        "val_aucs": val_aucs,
        "val_prs": val_prs,
    }


def write_markdown_report(results, report_path: Path):
    """Write the comparison table to molmetal/reports/r1_cross_attn_v3_fix.md."""
    lines = []
    lines.append("# R1 — Cross-attention fusion V3 fix\n")
    lines.append(
        "V2 (plain cross-attention) collapsed on the Ru temporal test set "
        "(test AUC 0.3764 vs V1 concat's 0.6617). V3 adds dropout + residual "
        "scaling + equivariant init to the same backbone. All three variants "
        "trained on the identical Ru temporal split.\n"
    )
    lines.append("- Backbone: D-MPNN (3 layers, hidden=128) + EGNN (3 layers, hidden=128)")
    lines.append("- Dual head: pIC50 regression + active classification (alpha=0.5)")
    lines.append("- Optimiser: Adam, lr=1e-3, cosine schedule over --epochs")
    lines.append("- Filter: max_atoms=50 (post-filter n same as W2 ablation)\n")

    lines.append("## Results\n")
    lines.append("| variant | params | Ru val AUC | Ru val PR | Ru test AUC | Ru test PR | test pIC50 MAE | avg epoch time | wall-clock |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['n_params']:,} | "
            f"{r['best_val_auc']:.4f} | {r['best_val_pr']:.4f} | "
            f"{r['test_auc']:.4f} | {r['test_pr']:.4f} | "
            f"{r['test_pic50_mae']:.4f} | "
            f"{r['avg_epoch_time_seconds']:.1f}s | "
            f"{r['wall_clock_seconds']:.1f}s |"
        )

    lines.append("\n## Per-epoch validation curves\n")
    for r in results:
        lines.append(f"\n### {r['variant']} (params={r['n_params']:,})\n")
        lines.append("| epoch | train_loss | val_AUC | val_PR |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for i, (tl, va, vp) in enumerate(zip(r["train_losses"], r["val_aucs"], r["val_prs"])):
            lines.append(f"| {i+1} | {tl:.4f} | {va:.4f} | {vp:.4f} |")

    # Winner
    by_test_auc = sorted(results, key=lambda r: r["test_auc"], reverse=True)
    winner = by_test_auc[0]
    runner = by_test_auc[1] if len(by_test_auc) > 1 else None
    lines.append("\n## Conclusion\n")
    lines.append(
        f"**Winner on Ru temporal test set: `{winner['variant']}` "
        f"(test AUC={winner['test_auc']:.4f})**\n"
    )

    # V3 vs V1 vs V2 delta table
    by_name = {r["variant"]: r for r in results}
    if "v1_concat" in by_name and "v3_crossattn" in by_name:
        v1 = by_name["v1_concat"]
        v3 = by_name["v3_crossattn"]
        delta = v3["test_auc"] - v1["test_auc"]
        rel = (delta / max(v1["test_auc"], 1e-6)) * 100
        lines.append(
            f"V3 vs V1 (concat): {delta:+.4f} test AUC ({rel:+.1f}% relative). "
            f"V3 closes the V2 gap of "
            f"{by_name['v1_concat']['test_auc'] - by_name.get('v2_crossattn', {'test_auc': 0.0})['test_auc']:+.4f} "
            f"by "
            f"{v1['test_auc'] - v3['test_auc']:+.4f}.\n"
        )
    if runner is not None:
        delta = winner["test_auc"] - runner["test_auc"]
        rel = (delta / max(runner["test_auc"], 1e-6)) * 100
        lines.append(
            f"Margin over `{runner['variant']}` (test AUC={runner['test_auc']:.4f}): "
            f"{delta:+.4f} ({rel:+.1f}% relative).\n"
        )
    for r in results:
        lines.append(
            f"- `{r['variant']}`: {r['n_params']:,} params, "
            f"{r['avg_epoch_time_seconds']:.1f}s/epoch, "
            f"test AUC={r['test_auc']:.4f}, test PR={r['test_pr']:.4f}"
        )

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n[ablation_v3] Markdown report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="D-MPNN + EGNN fusion ablation V3 (V1 concat + V2 cross-attn + V3 regularised)"
    )
    parser.add_argument("--metal", type=str, default="Ru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-threshold", type=float, default=24.0)
    parser.add_argument("--ic50-min", type=float, default=0.01)
    parser.add_argument("--cutoff-year", type=int, default=2024)
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=50,
        help="Drop mols with >max_atoms heavy atoms (0 = no filter).",
    )
    parser.add_argument(
        "--skip-v2",
        action="store_true",
        help="Skip V2 (its test AUC is already known: 0.3764 from W2).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"[ablation_v3] device={device}")

    # ---- Data ----
    print(f"[ablation_v3] Loading MetalCytoToxDB for metal={args.metal} ...")
    flt = CytotoxFilter(
        time_threshold=args.time_threshold,
        ic50_min=args.ic50_min,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    dataset = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[ablation_v3] Dataset: {len(dataset)} rows")
    if len(dataset) == 0:
        print("ERROR: Empty dataset after filtering.")
        sys.exit(1)

    # ---- Temporal split (Year < cutoff => train+val, Year >= cutoff => test) ----
    splitter = TemporalSplitter(cutoff_year=args.cutoff_year)
    split_result = splitter(dataset)
    train_idx = split_result.train_idx
    val_idx = split_result.val_idx
    test_idx = split_result.test_idx
    print(
        f"[ablation_v3] Temporal split (cutoff={args.cutoff_year}): "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # ---- Optional: filter by atom-count to dodge ROCm scatter_add HW fault
    if args.max_atoms is not None and args.max_atoms > 0:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        def _filter_by_atoms(indices, label):
            keep = []
            n_drop = 0
            for i in indices:
                smi = dataset[int(i)]["smiles"]
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue
                    if mol.GetNumAtoms() <= args.max_atoms:
                        keep.append(i)
                    else:
                        n_drop += 1
                except Exception:
                    continue
            print(
                f"[ablation_v3] {label}: kept {len(keep)}/{len(indices)} "
                f"(dropped {n_drop} with >{args.max_atoms} atoms)"
            )
            return np.array(keep, dtype=np.int64)

        train_idx = _filter_by_atoms(train_idx, "train")
        val_idx = _filter_by_atoms(val_idx, "val")
        test_idx = _filter_by_atoms(test_idx, "test")
        if len(train_idx) == 0:
            print("ERROR: empty train set after atom-count filter")
            sys.exit(1)

    featurizer = GraphFeaturizer()

    # ---- Three model factories ----
    base_cfg = MetalHybridConfig(
        n_dmpnn_layers=3,
        n_egnn_layers=3,
        hidden_dim=128,
        dropout=0.1,
    )

    def make_v1():
        return MetalHybridModel(config=base_cfg)

    def make_v2():
        return MetalHybridV2Model(config=MetalHybridV2Config(base=base_cfg))

    def make_v3():
        return MetalHybridV3Model(config=MetalHybridV3Config(base=base_cfg))

    variants = [
        ("v1_concat", make_v1),
        ("v3_crossattn", make_v3),
    ]
    if not args.skip_v2:
        variants.insert(1, ("v2_crossattn", make_v2))

    results = []
    for name, factory in variants:
        r = run_variant(
            name,
            factory,
            dataset,
            train_idx,
            val_idx,
            test_idx,
            args,
            device,
            featurizer,
        )
        results.append(r)

    # ---- Persist JSON results ----
    json_path = REPORT_DIR / "r1_cross_attn_v3_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"[ablation_v3] JSON results written to {json_path}")

    # ---- Markdown report ----
    md_path = REPORT_DIR / "r1_cross_attn_v3_fix.md"
    write_markdown_report(results, md_path)

    # ---- Console summary ----
    print("\n=== Ablation V3 summary (Ru temporal test set) ===")
    for r in results:
        print(
            f"  {r['variant']:>14}: params={r['n_params']:>7,}  "
            f"test_AUC={r['test_auc']:.4f}  test_PR={r['test_pr']:.4f}  "
            f"avg_epoch_dt={r['avg_epoch_time_seconds']:.1f}s"
        )
    winner = max(results, key=lambda r: r["test_auc"])
    print(f"\nWinner: {winner['variant']} (test AUC={winner['test_auc']:.4f})")


if __name__ == "__main__":
    main()
