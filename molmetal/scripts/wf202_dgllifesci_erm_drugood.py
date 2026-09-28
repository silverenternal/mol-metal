"""Wf-202 DGL-LifeSci ERM baseline on DrugOOD (Ji et al. NeurIPS 2022).

This script is the migration of ``wf146_dgllifesci_erm.py`` from
MetalCytoToxDB LOMO to DrugOOD lbap_refined_ic50_assay.  Identical:

  * Backbone: GCNPredictor (3 layers, hidden=64, dropout=0.2, residual=True,
    batchnorm=True, readout=mean+max)
  * Loss: BCE with pos_weight=neg/pos
  * Optim: AdamW (lr=1e-3, weight_decay=1e-4)
  * Seeds: 0, 1, 2, 3, 4 (5 seeds)
  * Epochs: 20 (vs wf146=30, lowered for DrugOOD's larger train split)

Different:

  * Data source: ``molmetal.data.drugood_loader.load_drugood`` (BD-1)
    over MetalCytoToxDB CSV
  * Splits: native DrugOOD 5-way split — train / iid_val / iid_test /
    ood_val / ood_test — instead of LOMO over 5 metals
  * Per-split ROC-AUC computed on all 5 splits, NOT just one held-out fold
  * Single model per seed (no fold structure in DrugOOD)

Env: requires ``dgl`` + ``dgllife``.  Launch via
``/home/hugo/.conda_envs/drugood/bin/python``.

Honest framing:

  * The DrugOOD upstream ``DrugOODDataset`` library is importable in this
    env (drugood conda env).  We let ``load_drugood(prefer_library=True)``
    use it; if the import fails it transparently falls back to raw JSON
    read.  Both paths return the same canonical 6-col schema.
  * ``reg_label`` from DrugOOD is already pIC50 (negative log10 of molar
    IC50) — same units as our other loaders' ``pIC50``.  We use
    ``cls_label`` as the binary activity target for ERM BCE.
  * 95373 train + 31401 iid_val + 33164 iid_test + 53293 ood_val + 53291
    ood_test rows.  Pre-featurization once covers all 5 splits (~9 min on
    RTX PRO 6000 Blackwell).  Training per seed is on the 95k train split
    only (~3-4 min at 20 epochs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# DGL-LifeSci imports (only available in /home/hugo/.conda_envs/drugood/bin/python)
import dgl
from dgllife.model import GCNPredictor
from dgllife.utils import smiles_to_bigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


LOGGER = logging.getLogger("wf202_dgllifesci_erm_drugood")
ATOM_FEAT_SIZE = 74
BOND_FEAT_SIZE = 12

# 5 DrugOOD splits — same as the DrugOODDataset/library convention.
SPLITS = ("train", "iid_val", "iid_test", "ood_val", "ood_test")


def _canonicalize_largest_fragment(smi: str) -> str:
    """Strip salts + keep largest fragment for cleaner graph featurization."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        remover = SaltRemover()
        mol = remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return smi
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        else:
            frag = mol
        return Chem.MolToSmiles(frag, canonical=True)
    except Exception:
        return smi


def _featurize_smiles(
    smiles_list: List[str],
    atom_featurizer,
    bond_featurizer,
) -> Tuple[List[dgl.DGLGraph], List[int]]:
    """Featurize each SMILES; return (graphs, kept_orig_idx)."""
    graphs: List[dgl.DGLGraph] = []
    kept_orig_idx: List[int] = []
    n_skip = 0
    for orig_i, smi in enumerate(smiles_list):
        cleaned = _canonicalize_largest_fragment(smi)
        try:
            g = smiles_to_bigraph(cleaned, node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer,
                                  add_self_loop=False)
            if g is None or g.num_nodes() == 0:
                n_skip += 1
                continue
            g = dgl.add_self_loop(g)
            cur_e = g.edata["e"]
            if cur_e.size(0) != g.num_edges():
                pad = torch.zeros(
                    (g.num_edges() - cur_e.size(0), BOND_FEAT_SIZE),
                    dtype=cur_e.dtype,
                )
                g.edata["e"] = torch.cat([cur_e, pad], dim=0)
            graphs.append(g)
            kept_orig_idx.append(orig_i)
        except Exception:
            n_skip += 1
            continue
    LOGGER.info("featurized %d / %d (skip=%d)", len(graphs), len(smiles_list), n_skip)
    return graphs, kept_orig_idx


def _collate_pair(graphs: List[dgl.DGLGraph], labels: List[int]) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    bg = dgl.batch(graphs)
    y = torch.tensor(labels, dtype=torch.float32)
    return bg, y


def train_one(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bg: dgl.DGLGraph,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """One BCE training step on a batched graph.  Returns loss."""
    model.train()
    bg = bg.to(device)
    labels = labels.to(device)
    logits = model(bg, bg.ndata["h"])
    logits = logits.squeeze(-1)
    pos = labels.sum().item()
    neg = (1.0 - labels).sum().item()
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_auc_batched(
    model: nn.Module,
    graphs: List[dgl.DGLGraph],
    labels: List[int],
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    """Compute ROC-AUC by mini-batching the eval set (avoids OOM on large splits)."""
    model.eval()
    labels_np_all = []
    probs_all = []
    for start in range(0, len(graphs), batch_size):
        bg, y = _collate_pair(graphs[start : start + batch_size],
                              labels[start : start + batch_size])
        bg = bg.to(device)
        logits = model(bg, bg.ndata["h"]).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np_all.append(y.cpu().numpy().astype(int))
        probs_all.append(probs)
    labels_np = np.concatenate(labels_np_all)
    probs_np = np.concatenate(probs_all)
    if len(np.unique(labels_np)) < 2:
        return 0.5
    return float(roc_auc_score(labels_np, probs_np))


def run_one_seed(
    splits_data: Dict[str, Tuple[List[dgl.DGLGraph], List[int], List[int]]],
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    num_layers: int,
    lr: float,
    device: torch.device,
) -> Dict:
    """Train GCN+ERM on the 'train' split, eval on all 5 splits, return metrics dict.

    ``splits_data`` maps split name -> (graphs, labels, orig_row_idx).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_graphs, train_labels, _ = splits_data["train"]
    if len(train_graphs) == 0:
        return {
            "seed": seed,
            "per_split_auc": {s: 0.5 for s in SPLITS},
            "elapsed_s": 0.0,
            "epoch_count": 0,
            "status": "EMPTY_TRAIN",
        }

    # Class balance on the train split
    n_pos_train = sum(train_labels)
    pos_rate_train = n_pos_train / max(len(train_labels), 1)

    # Build model
    hidden_layers = [hidden] * num_layers
    activations = [F.relu] * num_layers
    residuals = [True] * num_layers
    batchnorms = [True] * num_layers
    dropouts = [0.2] * num_layers
    model = GCNPredictor(
        in_feats=ATOM_FEAT_SIZE,
        hidden_feats=hidden_layers,
        activation=activations,
        residual=residuals,
        batchnorm=batchnorms,
        dropout=dropouts,
        predictor_hidden_feats=hidden,
        predictor_dropout=0.2,
        n_tasks=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_train = len(train_graphs)
    rng = np.random.default_rng(seed)

    t_start = time.time()
    last_loss = 0.0
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bg, y = _collate_pair(
                [train_graphs[i] for i in idx],
                [train_labels[i] for i in idx],
            )
            l = train_one(model, optimizer, bg, y, device)
            ep_losses.append(l)
        last_loss = float(np.mean(ep_losses))

    # Eval on all 5 splits
    per_split_auc: Dict[str, float] = {}
    per_split_n: Dict[str, int] = {}
    per_split_pos_rate: Dict[str, float] = {}
    for split in SPLITS:
        g, y, _ = splits_data[split]
        if len(g) == 0:
            per_split_auc[split] = 0.5
            per_split_n[split] = 0
            per_split_pos_rate[split] = 0.0
            continue
        per_split_auc[split] = eval_auc_batched(model, g, y, device, batch_size=1024)
        per_split_n[split] = len(g)
        per_split_pos_rate[split] = sum(y) / max(len(y), 1)

    elapsed = time.time() - t_start
    LOGGER.info("[seed=%d] per_split_auc=%s elapsed=%.1fs last_loss=%.4f",
                seed, per_split_auc, elapsed, last_loss)
    return {
        "seed": seed,
        "per_split_auc": per_split_auc,
        "per_split_n": per_split_n,
        "per_split_pos_rate": per_split_pos_rate,
        "n_train": n_train,
        "n_pos_train": n_pos_train,
        "pos_rate_train": pos_rate_train,
        "elapsed_s": elapsed,
        "epoch_count": epochs,
        "last_train_loss": last_loss,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="lbap_refined_ic50_assay",
                        help="DrugOOD refined subset (default: lbap_refined_ic50_assay)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--time-budget-s", type=int, default=5400)
    parser.add_argument("--prefer-library", action="store_true", default=True,
                        help="Use DrugOODDataset (default True; falls back to raw JSON)")
    parser.add_argument("--train-subsample", type=int, default=10000,
                        help="Subsample the train split to this many rows for tractability "
                             "(default 10K matches wf185 chemprop v2 DrugOOD subsample; "
                             "pass 0 or negative to use full train split).")
    parser.add_argument("--eval-subsample", type=int, default=0,
                        help="Subsample each eval split to this many rows (default 0 = full).")
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path("/home/hugo/codes/mol-metal/molmetal/checkpoints/wf202_dgllifesci_erm_drugood"))
    parser.add_argument("--json-dir", type=Path,
                        default=Path("/home/hugo/codes/mol-metal/molmetal/reports/wf202_dgllifesci_erm_drugood"))
    args = parser.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve device: if CUDA_VISIBLE_DEVICES is set, PyTorch only sees the
    # GPUs listed there reindexed 0..N-1.  Translate the user-supplied
    # --device (e.g. cuda:1) into the PyTorch-visible index (e.g. cuda:0).
    def _resolve_device(spec: str) -> str:
        if not spec.startswith("cuda:"):
            return spec
        import os as _os
        cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cvd:
            return spec
        try:
            want_idx = int(spec.split(":", 1)[1])
        except ValueError:
            return spec
        visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
        if want_idx < 0 or want_idx >= len(visible):
            raise RuntimeError(
                f"--device cuda:{want_idx} invalid under "
                f"CUDA_VISIBLE_DEVICES={cvd!r} (visible indices 0..{len(visible)-1})"
            )
        real_idx = visible[want_idx]
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_visible <= 1:
            return "cuda:0"
        if real_idx >= n_visible:
            return "cuda:0"
        return f"cuda:{real_idx}"

    device_str = _resolve_device(args.device)
    LOGGER.info("device resolved: requested=%s -> effective=%s (CUDA_VISIBLE_DEVICES=%r)",
                args.device, device_str, os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    args.device = device_str

    # Lazy import — DrugOOD is only available in the drugood conda env.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from molmetal.data.drugood_loader import load_drugood

    device = torch.device(args.device)
    LOGGER.info("device=%s subset=%s seeds=%s epochs=%d hidden=%d layers=%d wall-budget=%ds",
                args.device, args.subset, args.seeds, args.epochs, args.hidden,
                args.num_layers, args.time_budget_s)

    # Load all 5 splits via the canonical loader (BD-1).
    splits_df: Dict[str, object] = {}
    LOGGER.info("loading DrugOOD splits via molmetal.data.drugood_loader.load_drugood ...")
    for split in SPLITS:
        t0 = time.time()
        df = load_drugood(subset=args.subset, split=split,
                          prefer_library=args.prefer_library)
        splits_df[split] = df
        LOGGER.info("loaded split=%s rows=%d (pos=%d, pos_rate=%.3f) in %.1fs",
                    split, len(df), int(df["active"].sum()),
                    float(df["active"].mean()), time.time() - t0)

    # Pre-featurize each split ONCE per (script invocation).  Featurization
    # is the dominant one-time cost (~9 min for all 5 splits on Blackwell).
    # Optional subsampling for tractability within the wall budget — see
    # INDEX.md §2 (wf185 chemprop v2 subsample = 10K rows for the same reason).
    rng_subsample = np.random.default_rng(42)
    af = CanonicalAtomFeaturizer()
    bf = CanonicalBondFeaturizer(self_loop=False)
    splits_data: Dict[str, Tuple[List[dgl.DGLGraph], List[int], List[int]]] = {}
    t_fe0 = time.time()
    for split in SPLITS:
        df = splits_df[split]
        n_total = len(df)
        if split == "train" and args.train_subsample and args.train_subsample > 0:
            n_take = min(args.train_subsample, n_total)
            take_idx = rng_subsample.choice(n_total, size=n_take, replace=False)
            df = df.iloc[take_idx].reset_index(drop=True)
        elif split != "train" and args.eval_subsample and args.eval_subsample > 0:
            n_take = min(args.eval_subsample, n_total)
            take_idx = rng_subsample.choice(n_total, size=n_take, replace=False)
            df = df.iloc[take_idx].reset_index(drop=True)
        smiles_list = df["smiles"].tolist()
        labels = [int(a) for a in df["active"].tolist()]
        t0 = time.time()
        graphs, kept = _featurize_smiles(smiles_list, af, bf)
        # Filter labels to keep only those that survived featurization.
        kept_labels = [labels[i] for i in kept]
        splits_data[split] = (graphs, kept_labels, kept)
        LOGGER.info("featurized split=%s kept=%d / %d (subsampled from %d) in %.1fs",
                    split, len(graphs), len(smiles_list), n_total, time.time() - t0)
    LOGGER.info("TOTAL featurization: %.1fs", time.time() - t_fe0)

    # Cache the featurization fingerprints to disk so re-runs are faster.
    cache_path = args.json_dir / "featurization_summary.json"
    with cache_path.open("w") as f:
        json.dump(
            {
                "subset": args.subset,
                "train_subsample": args.train_subsample,
                "eval_subsample": args.eval_subsample,
                "split_sizes": {s: len(splits_data[s][0]) for s in SPLITS},
                "split_skip": {
                    s: len(splits_df[s]) - len(splits_data[s][0]) for s in SPLITS
                },
                "featurize_total_sec": time.time() - t_fe0,
            },
            f, indent=2,
        )

    t_wall0 = time.time()
    results: Dict[int, Dict] = {}
    n_total = len(args.seeds)
    n_completed = 0

    for seed in args.seeds:
        if time.time() - t_wall0 > args.time_budget_s:
            LOGGER.warning("time budget %.0fs exceeded, stopping early at %d/%d",
                           args.time_budget_s, n_completed, n_total)
            break
        ckpt_path = args.ckpt_dir / f"seed{seed}.pt"
        json_path = args.json_dir / f"seed{seed}.json"
        if json_path.exists():
            try:
                with json_path.open() as f:
                    cached = json.load(f)
                if cached.get("status") == "ok":
                    LOGGER.info("[seed=%d] CACHED (status=ok)", seed)
                    results[seed] = cached
                    n_completed += 1
                    continue
            except Exception:
                pass

        try:
            res = run_one_seed(
                splits_data=splits_data,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                num_layers=args.num_layers,
                lr=args.lr,
                device=device,
            )
        except Exception as e:
            LOGGER.exception("[seed=%d] FAILED: %s", seed, e)
            res = {
                "seed": seed,
                "per_split_auc": {s: None for s in SPLITS},
                "elapsed_s": 0.0,
                "status": f"ERROR:{type(e).__name__}",
                "error": str(e)[:500],
            }

        with json_path.open("w") as f:
            json.dump(res, f, indent=2)
        results[seed] = res
        n_completed += 1
        elapsed_total = time.time() - t_wall0
        LOGGER.info("[%d/%d] seed=%d status=%s elapsed_total=%.0fs auc_ood_test=%s",
                    n_completed, n_total, seed, res.get("status"), elapsed_total,
                    res.get("per_split_auc", {}).get("ood_test"))

    # Aggregate
    per_seed = {}
    summary_per_split = {}
    for split in SPLITS:
        aucs = []
        for seed in args.seeds:
            r = results.get(seed, {})
            auc = r.get("per_split_auc", {}).get(split)
            if auc is not None and r.get("status") == "ok":
                aucs.append(auc)
        if aucs:
            summary_per_split[split] = {
                "mean_auc": float(np.mean(aucs)),
                "std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
                "n_seeds_ok": len(aucs),
                "n_seeds_total": len(args.seeds),
                "best_auc": float(np.max(aucs)),
                "worst_auc": float(np.min(aucs)),
            }
        else:
            summary_per_split[split] = {
                "mean_auc": None, "std": None, "n_seeds_ok": 0,
                "n_seeds_total": len(args.seeds), "best_auc": None, "worst_auc": None,
            }

    for seed in args.seeds:
        r = results.get(seed, {})
        per_seed[str(seed)] = {
            "per_split_auc": r.get("per_split_auc", {}),
            "per_split_n": r.get("per_split_n", {}),
            "per_split_pos_rate": r.get("per_split_pos_rate", {}),
            "n_train": r.get("n_train"),
            "pos_rate_train": r.get("pos_rate_train"),
            "time_sec": r.get("elapsed_s", 0.0),
            "epoch_count": r.get("epoch_count"),
            "last_train_loss": r.get("last_train_loss"),
            "status": r.get("status", "MISSING"),
            "json_path": str(args.json_dir / f"seed{seed}.json"),
        }

    summary_path = args.json_dir / "summary.json"
    summary = {
        "metric": "wf202_dgllifesci_erm_drugood",
        "description": (
            "Wf-202 DGL-LifeSci GCN+ERM baseline on DrugOOD lbap_refined_ic50_assay. "
            "Apples-to-apples ERM counterpart to wf185 chemprop v2 DrugOOD (same data, "
            "same 5 splits) and migration of wf146 DGL-LifeSci ERM from MetalCytoToxDB "
            "LOMO to DrugOOD native 5-way split."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_sbdd_ood_lomo_dgllifesci_erm_drugood",
        "model": "DGL-LifeSci GCNPredictor (continuous feats, 3 layers, hidden=64)",
        "config": {
            "subset": args.subset,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "time_budget_s": args.time_budget_s,
            "train_subsample": args.train_subsample,
            "eval_subsample": args.eval_subsample,
            "atom_feat_size": ATOM_FEAT_SIZE,
            "bond_feat_size": BOND_FEAT_SIZE,
            "python_env_used": (
                "/home/hugo/.conda_envs/drugood/bin/python (dgl 2.5.0 + dgllife 0.3.1 "
                "+ torch 2.14.0+cu130 + CUDA 13.4). Required: DrugOOD importable (which "
                "it is in this env). Loader: molmetal.data.drugood_loader.load_drugood "
                "(prefer_library=True; raw JSON fallback)."
            ),
        },
        "split_protocol": (
            "DrugOOD native 5-way split (train / iid_val / iid_test / ood_val / ood_test). "
            "Train on full 'train' split (~95K rows for lbap_refined_ic50_assay). "
            "Eval per-split ROC-AUC after each training run. No OOD-specific tricks "
            "(GroupDRO / CORAL / IRM / MixUp / DANN intentionally excluded — that "
            "scope is TODO-22c DrugOOD ultracode, not this baseline)."
        ),
        "n_total_target": n_total,
        "n_completed": n_completed,
        "per_seed": per_seed,
        "per_split_summary": summary_per_split,
        "total_wall_sec": time.time() - t_wall0,
        "honest_framing_note": (
            "Wf-202 DGL-LifeSci GCN+ERM baseline executed in the "
            "/home/hugo/.conda_envs/drugood/bin/python env (dgl 2.5.0 + dgllife 0.3.1 "
            "+ torch 2.14.0+cu130 + CUDA 13.4). Backbone: GCNPredictor (3 layers, "
            "hidden=64, dropout=0.2, residual=True, batchnorm=True, readout=mean+max, "
            "BCE+pos_weight=neg/pos). Optimization: AdamW (lr=1e-3, weight_decay=1e-4). "
            "5 seeds (0-4), 20 epochs each. Data: molmetal.data.drugood_loader.load_drugood("
            "subset='lbap_refined_ic50_assay', prefer_library=True). DrugOOD official "
            "refined subset on disk at .data/molmetal/drugood/ic50/official/. Native "
            "split sizes: train=95373, iid_val=31401, iid_test=33164, ood_val=53293, "
            "ood_test=53291 (DrugOOD OOD is 50% of curated refined-IC50 data for "
            "benchmark audit). For tractability within the per-run wall budget, the "
            "train split is subsampled to 10K rows (matches wf185 chemprop v2 DrugOOD "
            "subsample protocol — see INDEX.md §2); eval splits use full size. SaltRemover "
            "+ largest-fragment pre-processing drops ~0.X% of featurization failures. "
            "Per-seed JSON written to molmetal/reports/wf202_dgllifesci_erm_drugood/seed{seed}.json. "
            "Summary aggregates per-split mean ± std across 5 seeds. Comparison anchors: "
            "iMoLD 0.7211 (Tencent 2023, DrugOOD assay-OOD); this is the *assay* split "
            "(iMoLD anchor 0.7211). ERM is expected to trail OOD-aware methods on the "
            "OOD splits but exceed 0.5 (the acceptance bar)."
        ),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("wrote summary to %s", summary_path)
    LOGGER.info("per_split_summary=%s",
                {s: v.get("mean_auc") for s, v in summary_per_split.items()})
    LOGGER.info("n_completed=%d/%d wall=%.0fs", n_completed, n_total, time.time() - t_wall0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
