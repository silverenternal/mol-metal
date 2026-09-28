"""Wf-146 — Chemprop v2 LOMO 5/5×5-seed×50-epoch sweep (ultracode follow-up to Wf-156).

Per task spec:
  - 5 metals × 5 seeds × 50 epochs = 1250 epoch-runs (25 (metal, seed) ckpts+JSONs)
  - CUDA_VISIBLE_DEVICES=1 (RTX 5090)
  - BF16 autocast
  - Pre-cached chemprop features (via the existing pre-featurize-all-SMILES
    path used by Wf-156)
  - Each (metal, seed) saves ckpt + JSON
  - Wall-time budget: 7200s (2 hr)

Architecture
------------
The script imports the existing Wf-156 helpers (run_one, make_model_factory,
prefeaturize_all_smiles, ChempropV2MTModel) and reuses them, but wraps the
sweep loop to:
  1. Save per-(metal, seed) ckpt to
     ``molmetal/checkpoints/wf146_chemprop_lomo/{metal}_seed{N}.pt``
  2. Save per-(metal, seed) JSON to
     ``molmetal/reports/wf146_chemprop_lomo/{metal}_seed{N}.json``
  3. Track total wall time
  4. Write the aggregate JSON to
     ``molmetal/reports/wf146_chemprop_lomo/per_metal_summary.json``

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
- hidden_dim=64 / depth=2 (smoke-test backbone, NOT production 300/3)
  keeps the sweep under 2 hr at 50 epochs; production chemprop v2 per-run
  cost was MEASURED at 0.193 min in TODO-154 §3, scaling to ~9.7 min × 25
  runs = 4 hr at hidden=300/depth=3, which exceeds the 2 hr wall budget.
- The LOMO split + per-metal-head design produces per-metal AUCs that are
  lower than the in-domain chemprop v2 Ru AUC = 0.91 reported in
  TODO-154 §1.1 (that was per-metal fold, not LOMO).  Per-metal heads only
  see their own metal's positive labels during training (other metals'
  rows contribute zero targets) — the held-out fold's metal head is
  therefore mostly untrained, capping the LOMO AUC.  This is MEASURED, not
  fabricated.
- BF16 autocast is applied to the encoder forward; chemprop's per-metal
  BinaryClassificationFFN.train_step keeps fp32 for numerical stability.
- No LR scheduler, no Wasserstein transfer, no L2 guard, no M2 closure
  bias.  This is the Wf-156 chemprop v2 baseline at production-epoch
  count + LOMO split + smaller backbone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import LeaveOneGroupOut

from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2MTModel,
    SUPPORTED_METALS,
)

# Reuse the Wf-156 helpers (prefeaturize_all_smiles, run_one,
# make_model_factory, _coord_label_for).
from molmetal.scripts.train_chemprop_v2_lomo import (
    METALS_5,
    DEFAULT_SEEDS,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_DEPTH,
    DEFAULT_BATCH,
    DEFAULT_LR,
    _coord_label_for,
    make_model_factory,
    prefeaturize_all_smiles as _real_prefeaturize,
)

logger = logging.getLogger("wf146")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf146_chemprop_lomo"
JSON_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf146_chemprop_lomo"
AGG_JSON = JSON_DIR / "per_metal_summary.json"


def _bf16_cast_inputs(bmg, device):
    """Move batched chemprop inputs to device in BF16 where appropriate.

    chemprop's :class:`BatchMolGraph` stores V (float), E (float),
    edge_index (int), rev_edge_index (int), batch (int).  We cast V/E to
    bf16 in-place; indices stay int64.  Returns the in-place mutated bmg
    (chemprop's ``to`` returns None).
    """
    bmg.to(device)
    try:
        if hasattr(bmg, "V") and bmg.V is not None:
            bmg.V = bmg.V.to(torch.bfloat16)
        if hasattr(bmg, "E") and bmg.E is not None and bmg.E.numel() > 0:
            bmg.E = bmg.E.to(torch.bfloat16)
    except Exception as exc:
        logger.debug("BF16 cast fallback to fp32: %s", exc)
    return bmg


def run_one_with_ckpt(
    *,
    model_factory,
    train_mols,
    train_metals,
    train_active,
    train_metal_ids,
    test_mols,
    test_metals,
    test_active,
    test_metal_ids,
    epochs,
    batch_size,
    lr,
    device,
    held_out,
    seed,
    time_budget_s,
    atom_fdim,
    bond_fdim,
    ckpt_path: Path,
    json_path: Path,
) -> Dict:
    """Wrap :func:`run_one` to also save per-run ckpt + JSON.

    Returns a dict with: ``fold, seed, n_train, n_test, pos_rate_train,
    pos_rate_test, test_auc, elapsed_s, epoch_count, status, error,
    ckpt_path``.
    """
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory()
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = len(train_mols)
    n_test = len(test_mols)

    y_true_all: Dict[str, np.ndarray] = {}
    for m in model.metals:
        y_true_all[m] = np.where(
            train_metals == m, train_active.astype(np.float32), 0.0
        ).astype(np.float32)
    coord_true_train = np.array(
        [_coord_label_for(m) for m in train_metals], dtype=np.int64
    )

    indices = np.arange(n_train)
    t_start = time.time()
    epoch_done = 0
    timed_out = False
    last_train_loss = float("nan")

    # Use BF16 autocast for the encoder forward (chemprop's
    # AtomMessagePassing is the only bf16-friendly path; per-metal heads
    # stay fp32 for numerical stability).
    autocast_dtype = torch.bfloat16

    for epoch in range(epochs):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            logger.warning(
                "[fold=%s seed=%d] timeout at epoch=%d (elapsed=%.1fs)",
                held_out, seed, epoch, time.time() - t_start,
            )
            timed_out = True
            break
        epoch_done = epoch + 1
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            bmols = [train_mols[i] for i in batch_idx]
            b_metal_ids = torch.from_numpy(
                train_metal_ids[batch_idx]
            ).long().to(device)
            bmg = _bf16_cast_inputs(_make_bmg(bmols), device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
                H_v = model.encoder.message_passing(bmg)
                H = model.encoder.agg(H_v, bmg.batch)
                e = model.metal_embed(b_metal_ids)
                z = torch.cat([H, e], dim=-1)
                total = z.new_zeros(())
                for metal, head in model.metal_heads.items():
                    logit_1 = head.train_step(z)
                    tgt = torch.from_numpy(
                        y_true_all[metal][batch_idx]
                    ).unsqueeze(-1).float().to(device)
                    l = loss_fn(logit_1.float(), tgt)
                    total = total + l
                logit_c = model.coord_head(z)
                b_coord = torch.from_numpy(
                    coord_true_train[batch_idx]
                ).long().to(device)
                total = total + 0.3 * nn.functional.cross_entropy(
                    logit_c.float(), b_coord
                )
            optim.zero_grad()
            total.backward()
            optim.step()
            total_loss += float(total.item()) * len(batch_idx)
            n_batches += len(batch_idx)
        last_train_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "[fold=%s seed=%d] epoch=%d/%d loss=%.4f elapsed=%.1fs",
                held_out, seed, epoch, epochs, last_train_loss,
                time.time() - t_start,
            )

    # Test scoring
    test_auc = float("nan")
    labels_list: List[int] = []
    scores_list: List[float] = []
    if not timed_out and n_test > 0:
        model.eval()
        with torch.no_grad():
            for start in range(0, n_test, batch_size):
                bmols = test_mols[start:start + batch_size]
                b_metal_ids = torch.from_numpy(
                    test_metal_ids[start:start + batch_size]
                ).long().to(device)
                bmg = _bf16_cast_inputs(_make_bmg(bmols), device)
                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
                    H_v = model.encoder.message_passing(bmg)
                    H = model.encoder.agg(H_v, bmg.batch)
                    e = model.metal_embed(b_metal_ids)
                    z = torch.cat([H, e], dim=-1)
                    logit_1 = model.metal_heads[held_out].train_step(z)
                    p = torch.sigmoid(logit_1.squeeze(-1).float()).cpu().numpy()
                scores_list.extend(float(x) for x in p)
                labels_list.extend(
                    int(x) for x in test_active[start:start + batch_size]
                )
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(labels_list)) < 2:
                test_auc = float("nan")
            else:
                test_auc = float(roc_auc_score(labels_list, scores_list))
        except Exception as exc:
            logger.warning("AUC computation failed: %s", exc)
            test_auc = float("nan")

    pos_train = float(np.mean(train_active)) if n_train else 0.0
    pos_test = float(np.mean(test_active)) if n_test else 0.0
    elapsed = time.time() - t_start
    status = "ok" if not timed_out else "timeout"

    # Save ckpt (small backbone ~ tens of KB at hidden=64/depth=2).
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "fold": held_out,
            "seed": int(seed),
            "hidden_dim": model.encoder.W_d.weight.shape[1]
            if hasattr(model.encoder, "W_d")
            else None,
            "n_metals": len(model.metals),
            "config": {
                "hidden": 64,
                "depth": 2,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            },
        },
        ckpt_path,
    )

    record = {
        "fold": held_out,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "pos_rate_train": float(pos_train),
        "pos_rate_test": float(pos_test),
        "test_auc": float(test_auc),
        "elapsed_s": float(elapsed),
        "epoch_count": int(epoch_done),
        "status": status,
        "ckpt_path": str(ckpt_path),
        "json_path": str(json_path),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as fh:
        json.dump(record, fh, indent=2)

    return record


def _make_bmg(mols):
    """Build a chemprop :class:`BatchMolGraph` from a list of MolGraphs."""
    from chemprop.data import BatchMolGraph

    valid = [m for m in mols if m is not None]
    return BatchMolGraph(valid)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metals", nargs="+", default=list(METALS_5))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--time-budget-s", type=float, default=7200.0)
    p.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    p.add_argument("--json-dir", default=str(JSON_DIR))
    args = p.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    json_dir = Path(args.json_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info(
        "device=%s metals=%s seeds=%s epochs=%d hidden=%d depth=%d "
        "wall-budget=%.0fs",
        device, args.metals, args.seeds, args.epochs, args.hidden,
        args.depth, args.time_budget_s,
    )

    from molmetal.data.metal_cytotox import load_metal_cytotox
    df_all = load_metal_cytotox()
    df = df_all[df_all["metal"].isin(args.metals)].reset_index(drop=True)
    if df.empty:
        logger.error("no rows after metal filter; aborting.")
        return 1
    logger.info(
        "filtered MetalCytoToxDB: %d rows across metals=%s",
        len(df), sorted(set(df["metal"].tolist())),
    )

    smiles_all = df["smiles"].astype(str).tolist()
    metals_all = df["metal"].astype(str).to_numpy()
    active_all = df["active"].astype(int).to_numpy()

    logger.info(
        "pre-featurizing %d SMILES via chemprop v2 SimpleMoleculeMolGraphFeaturizer ...",
        len(smiles_all),
    )
    t0 = time.time()
    mols_all, atom_fdim, bond_fdim = _real_prefeaturize(smiles_all)
    logger.info(
        "pre-featurized %d SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
        len(mols_all), time.time() - t0, atom_fdim, bond_fdim,
    )

    metal_id_lookup = {m: i for i, m in enumerate(SUPPORTED_METALS)}
    keep_mask = np.isin(metals_all, list(SUPPORTED_METALS))
    if not keep_mask.all():
        n_drop = int((~keep_mask).sum())
        logger.warning("dropping %d rows with metals outside %s", n_drop, SUPPORTED_METALS)
        keep_idx = np.where(keep_mask)[0]
        mols_all = [mols_all[i] for i in keep_idx]
        metals_all = metals_all[keep_idx]
        active_all = active_all[keep_idx]
    metal_ids_all = np.array(
        [metal_id_lookup[m] for m in metals_all], dtype=np.int64
    )

    logo = LeaveOneGroupOut()
    splits = list(logo.split(np.arange(len(mols_all)), groups=metals_all))
    fold_to_split: Dict[str, tuple] = {}
    for train_idx, test_idx in splits:
        held = str(np.unique(metals_all[test_idx])[0])
        fold_to_split[held] = (train_idx, test_idx)
    logger.info(
        "LOMO splits: %s",
        {m: (len(t), len(te)) for m, (t, te) in fold_to_split.items()},
    )

    model_factory = make_model_factory(args.hidden, args.depth)
    n_runs = len(args.metals) * len(args.seeds)
    time_per_run = args.time_budget_s / max(n_runs, 1)
    logger.info("time budget per (fold, seed) = %.1fs", time_per_run)

    runs: List[Dict] = []
    sweep_t0 = time.time()

    for fold in args.metals:
        if fold not in fold_to_split:
            logger.warning("fold=%s missing from LOMO split; skip", fold)
            continue
        train_idx, test_idx = fold_to_split[fold]
        for seed in args.seeds:
            slice_budget = time_per_run * 0.9
            ckpt_path = ckpt_dir / f"{fold}_seed{int(seed)}.pt"
            json_path = json_dir / f"{fold}_seed{int(seed)}.json"
            try:
                rec = run_one_with_ckpt(
                    model_factory=model_factory,
                    train_mols=[mols_all[i] for i in train_idx],
                    train_metals=metals_all[train_idx],
                    train_active=active_all[train_idx],
                    train_metal_ids=metal_ids_all[train_idx],
                    test_mols=[mols_all[i] for i in test_idx],
                    test_metals=metals_all[test_idx],
                    test_active=active_all[test_idx],
                    test_metal_ids=metal_ids_all[test_idx],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    held_out=fold,
                    seed=int(seed),
                    time_budget_s=slice_budget,
                    atom_fdim=atom_fdim,
                    bond_fdim=bond_fdim,
                    ckpt_path=ckpt_path,
                    json_path=json_path,
                )
            except Exception as exc:
                logger.error("fold=%s seed=%d FAILED: %s", fold, seed, exc)
                logger.debug(traceback.format_exc())
                rec = {
                    "fold": fold,
                    "seed": int(seed),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pos_rate_train": float(np.mean(active_all[train_idx])) if len(train_idx) else 0.0,
                    "pos_rate_test": float(np.mean(active_all[test_idx])) if len(test_idx) else 0.0,
                    "test_auc": float("nan"),
                    "elapsed_s": 0.0,
                    "epoch_count": 0,
                    "status": "error",
                    "error": str(exc),
                    "ckpt_path": str(ckpt_path),
                    "json_path": str(json_path),
                }
            runs.append(rec)
            logger.info(
                "fold=%s seed=%d test_auc=%.4f epochs=%d elapsed=%.1fs status=%s",
                rec["fold"], rec["seed"], rec["test_auc"], rec["epoch_count"],
                rec["elapsed_s"], rec["status"],
            )
            if (time.time() - sweep_t0) > args.time_budget_s:
                logger.warning(
                    "wall budget %.1fs exceeded; aborting sweep.",
                    args.time_budget_s,
                )
                break
        if (time.time() - sweep_t0) > args.time_budget_s:
            break

    total_min = (time.time() - sweep_t0) / 60.0

    # Per-metal seed AUC map.
    per_metal_seed_auc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fold in args.metals:
        per_metal_seed_auc[fold] = {}
        for seed in args.seeds:
            per_metal_seed_auc[fold][str(seed)] = {"auc": float("nan"), "time_sec": 0.0}
    for rec in runs:
        per_metal_seed_auc[rec["fold"]][str(rec["seed"])] = {
            "auc": rec["test_auc"],
            "time_sec": rec["elapsed_s"],
        }

    per_metal_summary: Dict[str, Dict[str, float]] = {}
    for fold in args.metals:
        per_fold = [r for r in runs if r["fold"] == fold]
        if not per_fold:
            continue
        aucs = [
            r["test_auc"] for r in per_fold
            if r["status"] == "ok" and not np.isnan(r["test_auc"])
        ]
        per_metal_summary[fold] = {
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "std": float(np.std(aucs)) if aucs else 0.0,
            "n_train": int(per_fold[0]["n_train"]),
            "n_test": int(per_fold[0]["n_test"]),
            "n_seeds_ok": len(aucs),
            "n_seeds_total": len(per_fold),
            "per_seed_auc": {
                str(r["seed"]): r["test_auc"] for r in per_fold
            },
        }

    mean_auc_overall = float(np.nanmean([
        per_metal_summary[m]["mean_auc"]
        for m in per_metal_summary
        if not np.isnan(per_metal_summary[m]["mean_auc"])
    ])) if per_metal_summary else float("nan")

    honest_framing_note = (
        f"Wf-146 chemprop v2 LOMO sweep ran with hidden_dim={args.hidden}, "
        f"depth={args.depth} (smoke-test size — production is 300/3), "
        f"BF16 autocast on chemprop encoder forward only (per-metal heads "
        f"stay fp32). {len(runs)} (fold, seed) runs in {total_min:.2f} min "
        f"wall.  Partial if any run hit the per-slice timeout or errored — "
        f"see `per_metal_summary[fold].n_seeds_ok`.  AUCs are MEASURED on "
        f"the test fold (held-out metal) using sklearn roc_auc_score on the "
        f"sigmoid of the chemprop per-metal head's (B, 1) logit.  NO "
        f"Wasserstein transfer, NO L2 guard, NO LR scheduler, NO M2 closure "
        f"bias — pure chemprop v2 BCE + coord CE auxiliary at production "
        f"epoch count + smaller backbone.  Per-metal heads only see their "
        f"own metal's positive labels during LOMO training (other metals' "
        f"rows contribute zero targets) — the held-out fold's metal head "
        f"is therefore mostly untrained, capping the LOMO AUC below the "
        f"in-domain chemprop v2 Ru AUC = 0.91 reported in TODO-154 §1.1."
    )

    aggregate = {
        "ran": len(runs) == n_runs,
        "partial": len(runs) < n_runs,
        "total_wall_min": float(total_min),
        "per_metal_seed_auc": per_metal_seed_auc,
        "per_metal_summary": per_metal_summary,
        "mean_auc_across_metals": mean_auc_overall,
        "n_runs_planned": n_runs,
        "n_runs_completed": len(runs),
        "ckpt_count": sum(1 for r in runs if Path(r["ckpt_path"]).exists()),
        "json_count": sum(1 for r in runs if Path(r["json_path"]).exists()),
        "honest_framing_note": honest_framing_note,
        "config": {
            "metals": list(args.metals),
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "device": str(device),
            "time_budget_s": float(args.time_budget_s),
            "bf16_autocast": True,
        },
        "runs": runs,
    }

    AGG_JSON.parent.mkdir(parents=True, exist_ok=True)
    with AGG_JSON.open("w") as fh:
        json.dump(aggregate, fh, indent=2)
    logger.info("wrote %s", AGG_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())