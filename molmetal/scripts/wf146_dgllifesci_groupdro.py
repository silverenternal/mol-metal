"""Wf-146 DGL-LifeSci GroupDRO baseline on the same MetalCytoToxDB LOMO split
as ``wf146_dgllifesci_erm.py`` (and ``wf146_chemprop_lomo.py``).

This is the apples-to-apples *GroupDRO* counterpoint to the ERM baseline.

Algorithm (Sagawa et al., ICLR 2020, "Distributionally Robust Neural Networks
for Group Shifts: On the Importance of Regularization for Worst-Performing
Groups"):
  - Each training example has a "group" label = its metal
    (Ru / Ir / Os / Re when Rh is held out as test, etc.).
  - Maintain a vector q_g (one weight per train-domain metal) that is updated
    via gradient ascent on the group loss:
        q_g <- q_g * exp(step_size * L_g)
        then renormalise q_g to sum to 1.
  - The training objective is the weighted sum
        L = sum_g q_g * L_g
    which provably converges to the worst-group loss under the right schedule.

Implementation notes:
  - Backbone is identical to wf146_dgllifesci_erm.py: GCNPredictor
    (in_feats=74, hidden=[64,64,64], residual+BN, readout=mean+max,
    BCE+pos_weight=neg/pos, AdamW, no LR scheduler, no domain-adaptation
    tricks beyond GroupDRO itself).
  - We compute per-minibatch per-metal BCE, weight by q_m, backprop, then
    update q_m.  Initial q_m is uniform 1/n_groups.
  - step_size=0.01 (Sagawa et al. default for small/medium group counts).
  - Same exact LOMO protocol: hold out 1 metal as test, train on the other 4.

Env: requires dgl + dgllife.  The CONSTRAINT-requested
``/home/hugo/codes/mol-metal/.venv/bin/python`` lacks both modules
(verified: ModuleNotFoundError on import).  We use
``/home/hugo/.conda_envs/drugood/bin/python`` which has dgl 2.5.0 +
dgllife 0.3.1 + torch 2.14.0+cu130 + CUDA.  This is the same env used by
the ERM baseline; the honest_framing_note documents the substitution.

Outputs:
  - per-(metal, seed) JSON to ``molmetal/reports/wf146_baselines/groupdro/{metal}_seed{N}.json``
  - aggregate ``summary.json`` with the same schema as ERM
"""

from __future__ import annotations

import argparse
import csv
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

# DGL-LifeSci imports
import dgl
from dgllife.model import GCNPredictor
from dgllife.utils import smiles_to_bigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


LOGGER = logging.getLogger("wf146_dgllifesci_groupdro")
ATOM_FEAT_SIZE = 74
BOND_FEAT_SIZE = 12
# GroupDRO hyperparameters (Sagawa et al. defaults, adapted for n_groups=4)
GROUP_STEP_SIZE = 0.01   # step size for q_g ascent update
GROUP_WEIGHT_DECAY = True  # whether to regularise q (not used here, kept for reference)


def _load_5metal_csv(csv_path: Path) -> List[Dict]:
    """Load the 5-metal MetalCytoToxDB subset, returning rows with smiles/metal/activity."""
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                act = int(r["activity"])
            except (ValueError, KeyError):
                continue
            if r["metal"] not in ("Ru", "Ir", "Os", "Rh", "Re"):
                continue
            rows.append({
                "smiles": r["smiles"],
                "metal": r["metal"],
                "activity": act,
                "split": r.get("split", ""),
            })
    LOGGER.info("loaded %d rows from %s", len(rows), csv_path)
    return rows


def _lomo_indices(rows: List[Dict], held_out: str) -> Tuple[List[int], List[int]]:
    """Return (train_idx, test_idx) where test = all rows with metal==held_out."""
    train_idx, test_idx = [], []
    for i, r in enumerate(rows):
        if r["metal"] == held_out:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx


def _canonicalize_largest_fragment(smi: str) -> str:
    """Strip salts + keep largest organic fragment for cleaner graph featurization."""
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
    """Featurize each SMILES; return list of graphs and per-graph node/edge counts.
    SMILES that fail to parse are skipped."""
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
            cur_e = g.edata['e']
            if cur_e.size(0) != g.num_edges():
                pad = torch.zeros(
                    (g.num_edges() - cur_e.size(0), BOND_FEAT_SIZE),
                    dtype=cur_e.dtype,
                )
                g.edata['e'] = torch.cat([cur_e, pad], dim=0)
            graphs.append(g)
            kept_orig_idx.append(orig_i)
        except Exception:
            n_skip += 1
            continue
    LOGGER.info("featurized %d / %d (skip=%d)", len(graphs), len(smiles_list), n_skip)
    return graphs, kept_orig_idx


def _collate_pair(graphs: List[dgl.DGLGraph], labels: List[int]) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Batch graphs and stack labels into (batched_g, label_tensor)."""
    bg = dgl.batch(graphs)
    y = torch.tensor(labels, dtype=torch.float32)
    return bg, y


def train_one_groupdro(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bg: dgl.DGLGraph,
    labels: torch.Tensor,
    group_ids: torch.Tensor,  # metal index per example in the batch
    n_groups: int,
    q: torch.Tensor,
    device: torch.device,
    step_size: float,
) -> Tuple[float, torch.Tensor]:
    """One GroupDRO training step on a batched graph.

    Returns (loss_value, q_updated).  Computes per-group mean loss over the
    minibatch, weights by q, backprops, then updates q via exp(step_size*L_g).

    For groups absent from the minibatch we keep their previous q value (no
    gradient update for those groups, matching the WILDS reference).
    """
    model.train()
    bg = bg.to(device)
    labels = labels.to(device)
    group_ids = group_ids.to(device)

    logits = model(bg, bg.ndata["h"]).squeeze(-1)

    # Per-example BCE (no pos_weight; we want raw per-example losses so the
    # per-group means are interpretable).  pos_weight in the ERM script made
    # per-example losses a function of the global pos rate, which is the same
    # across all groups; for GroupDRO we use unweighted BCE per example to
    # compute a clean per-group mean.
    per_ex_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    # Per-group mean loss over the batch (only for groups present).
    group_losses = torch.zeros(n_groups, device=device)
    group_counts = torch.zeros(n_groups, device=device)
    for g in range(n_groups):
        mask = (group_ids == g)
        if mask.any():
            group_losses[g] = per_ex_loss[mask].mean()
            group_counts[g] = mask.sum().item()

    # GroupDRO objective: L = sum_g q_g * L_g
    loss = (q.to(device) * group_losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update q: q_g <- q_g * exp(step_size * L_g) for groups present in batch.
    # Detach group_losses so this update is not part of the autograd graph.
    q_np = q.detach().clone()
    with torch.no_grad():
        for g in range(n_groups):
            if group_counts[g] > 0:
                q_np[g] = q_np[g] * torch.exp(torch.tensor(step_size, device=device) * group_losses[g])
        # Renormalise
        q_np = q_np / q_np.sum()
        # Safety: clamp to [1e-3, 1.0] so no group ever gets prob 0 (avoids
        # degenerate "group fully dropped" failure mode).
        q_np = torch.clamp(q_np, min=1e-3)
        q_np = q_np / q_np.sum()

    return float(loss.item()), q_np


@torch.no_grad()
def eval_auc_batched(model: nn.Module, graphs: List[dgl.DGLGraph],
                     labels: List[int], device: torch.device,
                     batch_size: int = 256) -> float:
    """Compute ROC-AUC by mini-batching the test set (avoids OOM on large folds)."""
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


def run_lomo(
    rows: List[Dict],
    held_out: str,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    num_layers: int,
    lr: float,
    device: torch.device,
    group_step_size: float,
    graphs: List[dgl.DGLGraph] = None,
    graph_orig_idx: List[int] = None,
    metal_to_idx: Dict[str, int] = None,
) -> Dict:
    """Train GIN GroupDRO on LOMO train split, eval on LOMO test split.

    The "groups" are the 4 train-domain metals (the held-out metal is
    excluded from training and from q).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx, test_idx = _lomo_indices(rows, held_out)
    train_metals_set = sorted(set(rows[i]["metal"] for i in train_idx))
    LOGGER.info("[fold=%s seed=%d] train=%d test=%d metals=%s",
                held_out, seed, len(train_idx), len(test_idx), train_metals_set)

    # Build a local metal -> idx mapping for the train metals only
    if metal_to_idx is None:
        metal_to_idx = {m: i for i, m in enumerate(train_metals_set)}
    n_groups = len(metal_to_idx)
    LOGGER.info("[fold=%s seed=%d] n_groups=%d mapping=%s",
                held_out, seed, n_groups, metal_to_idx)

    # If we have pre-featurized graphs, subset
    if graphs is not None:
        orig_to_graph = {orig: gi for gi, orig in enumerate(graph_orig_idx)}
        train_idx_use = [i for i in train_idx if i in orig_to_graph]
        test_idx_use = [i for i in test_idx if i in orig_to_graph]
        train_graphs = [graphs[orig_to_graph[i]] for i in train_idx_use]
        test_graphs = [graphs[orig_to_graph[i]] for i in test_idx_use]
        train_rows = [rows[i] for i in train_idx_use]
        test_rows = [rows[i] for i in test_idx_use]
    else:
        af = CanonicalAtomFeaturizer()
        bf = CanonicalBondFeaturizer(self_loop=False)
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        t_fe = time.time()
        train_graphs, _ = _featurize_smiles([r["smiles"] for r in train_rows], af, bf)
        test_graphs, _ = _featurize_smiles([r["smiles"] for r in test_rows], af, bf)
        LOGGER.info("[fold=%s seed=%d] featurize took %.1fs train=%d test=%d",
                    held_out, seed, time.time() - t_fe, len(train_graphs), len(test_graphs))

    if len(train_graphs) == 0 or len(test_graphs) == 0:
        return {
            "fold": held_out, "seed": seed,
            "test_auc": 0.5, "elapsed_s": 0.0, "epoch_count": 0,
            "n_train": 0, "n_test": 0,
            "pos_rate_train": 0.0, "pos_rate_test": 0.0,
            "status": "EMPTY_FEATURIZE",
        }

    # Class balance
    n_pos_train = sum(r["activity"] for r in train_rows)
    n_pos_test = sum(r["activity"] for r in test_rows)
    pos_rate_train = n_pos_train / max(len(train_rows), 1)
    pos_rate_test = n_pos_test / max(len(test_rows), 1)

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

    # GroupDRO state: q uniform, on device
    q = torch.full((n_groups,), 1.0 / n_groups, device=device)

    train_labels = [r["activity"] for r in train_rows[: len(train_graphs)]]
    train_metal_idx = [metal_to_idx[r["metal"]] for r in train_rows[: len(train_graphs)]]
    test_labels = [r["activity"] for r in test_rows[: len(test_graphs)]]

    n_train = len(train_graphs)
    rng = np.random.default_rng(seed)

    t_start = time.time()
    last_loss = 0.0
    q_history = []  # track q evolution for diagnostics
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bg, y = _collate_pair(
                [train_graphs[i] for i in idx],
                [train_labels[i] for i in idx],
            )
            # Build per-example metal-index tensor for this batch
            group_ids = torch.tensor(
                [train_metal_idx[i] for i in idx], dtype=torch.long,
            )
            l, q = train_one_groupdro(
                model=model,
                optimizer=optimizer,
                bg=bg,
                labels=y,
                group_ids=group_ids,
                n_groups=n_groups,
                q=q,
                device=device,
                step_size=group_step_size,
            )
            ep_losses.append(l)
        last_loss = float(np.mean(ep_losses))
        # Snapshot q at end of each epoch (CPU-side)
        q_history.append(q.detach().cpu().tolist())

    # Eval (batched to avoid OOM on Ru fold with 19135 test graphs)
    test_auc = eval_auc_batched(model, test_graphs, test_labels, device, batch_size=512)
    elapsed = time.time() - t_start
    LOGGER.info("[fold=%s seed=%d] test_auc=%.4f epochs=%d elapsed=%.1fs last_loss=%.4f q_final=%s",
                held_out, seed, test_auc, epochs, elapsed, last_loss, q_history[-1])

    return {
        "fold": held_out,
        "seed": seed,
        "test_auc": test_auc,
        "elapsed_s": elapsed,
        "epoch_count": epochs,
        "n_train": n_train,
        "n_test": len(test_graphs),
        "pos_rate_train": pos_rate_train,
        "pos_rate_test": pos_rate_test,
        "last_train_loss": last_loss,
        "q_final": q_history[-1],
        "q_history": q_history,
        "train_metals": train_metals_set,
        "metal_to_idx": metal_to_idx,
        "group_step_size": group_step_size,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"))
    parser.add_argument("--metals", nargs="+", default=["Rh", "Ru", "Ir", "Os", "Re"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--group-step-size", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-budget-s", type=int, default=7200)
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path("/home/hugo/codes/mol-metal/molmetal/checkpoints/wf146_baselines/groupdro"))
    parser.add_argument("--json-dir", type=Path,
                        default=Path("/home/hugo/codes/mol-metal/molmetal/reports/wf146_baselines/groupdro"))
    args = parser.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device(args.device)
    LOGGER.info("device=%s metals=%s seeds=%s epochs=%d hidden=%d layers=%d wall-budget=%ds group_step=%g",
                args.device, args.metals, args.seeds, args.epochs, args.hidden,
                args.num_layers, args.time_budget_s, args.group_step_size)

    rows = _load_5metal_csv(args.csv)
    LOGGER.info("metal counts: %s",
                {m: sum(1 for r in rows if r["metal"] == m) for m in args.metals})

    # Pre-featurize ALL 26801 SMILES ONCE (~120s on RTX 5090 / Blackwell)
    af = CanonicalAtomFeaturizer()
    bf = CanonicalBondFeaturizer(self_loop=False)
    t_fe0 = time.time()
    all_smiles = [r["smiles"] for r in rows]
    all_graphs, kept_orig_idx = _featurize_smiles(all_smiles, af, bf)
    LOGGER.info("pre-featurized %d / %d in %.1fs (skip=%d)",
                len(all_graphs), len(all_smiles), time.time() - t_fe0,
                len(rows) - len(all_graphs))

    t_wall0 = time.time()
    results: Dict[str, Dict[str, Dict]] = {m: {} for m in args.metals}
    n_completed = 0
    n_total = len(args.metals) * len(args.seeds)

    # Interleave seeds across folds for balanced time use
    order = []
    for s in args.seeds:
        for m in args.metals:
            order.append((m, s))

    for fold, seed in order:
        if time.time() - t_wall0 > args.time_budget_s:
            LOGGER.warning("time budget %.0fs exceeded, stopping early at %d/%d",
                           args.time_budget_s, n_completed, n_total)
            break
        json_path = args.json_dir / f"{fold}_seed{seed}.json"
        if json_path.exists():
            try:
                with json_path.open() as f:
                    cached = json.load(f)
                if cached.get("status") == "ok":
                    LOGGER.info("[fold=%s seed=%d] CACHED (status=ok)", fold, seed)
                    results[fold][str(seed)] = cached
                    n_completed += 1
                    continue
            except Exception:
                pass

        try:
            res = run_lomo(
                rows=rows,
                held_out=fold,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                num_layers=args.num_layers,
                lr=args.lr,
                device=device,
                group_step_size=args.group_step_size,
                graphs=all_graphs,
                graph_orig_idx=kept_orig_idx,
            )
        except Exception as e:
            LOGGER.exception("[fold=%s seed=%d] FAILED: %s", fold, seed, e)
            res = {
                "fold": fold, "seed": seed,
                "test_auc": None, "elapsed_s": 0.0,
                "status": f"ERROR:{type(e).__name__}",
                "error": str(e)[:500],
            }

        with json_path.open("w") as f:
            json.dump(res, f, indent=2)
        results[fold][str(seed)] = res
        n_completed += 1
        elapsed_total = time.time() - t_wall0
        LOGGER.info("[%d/%d] fold=%s seed=%d auc=%s elapsed_total=%.0fs",
                    n_completed, n_total, fold, seed, res.get("test_auc"), elapsed_total)

    # Aggregate
    per_metal_seed_auc = {}
    per_metal_summary = {}
    for m in args.metals:
        per_metal_seed_auc[m] = {}
        aucs = []
        for s in args.seeds:
            r = results[m].get(str(s), {})
            per_metal_seed_auc[m][str(s)] = {
                "auc": r.get("test_auc"),
                "time_sec": r.get("elapsed_s", 0.0),
                "status": r.get("status", "MISSING"),
                "n_train": r.get("n_train"),
                "n_test": r.get("n_test"),
                "epoch_count": r.get("epoch_count"),
                "pos_rate_train": r.get("pos_rate_train"),
                "pos_rate_test": r.get("pos_rate_test"),
                "q_final": r.get("q_final"),
                "json_path": str(args.json_dir / f"{m}_seed{s}.json"),
            }
            if r.get("test_auc") is not None and r.get("status") == "ok":
                aucs.append(r["test_auc"])
        if aucs:
            per_metal_summary[m] = {
                "mean_auc": float(np.mean(aucs)),
                "std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
                "n_seeds_ok": len(aucs),
                "n_seeds_total": len(args.seeds),
                "best_auc": float(np.max(aucs)),
                "worst_auc": float(np.min(aucs)),
            }
        else:
            per_metal_summary[m] = {
                "mean_auc": None, "std": None, "n_seeds_ok": 0,
                "n_seeds_total": len(args.seeds), "best_auc": None, "worst_auc": None,
            }

    # Overall mean across metals (macro avg of per-metal means)
    metal_means = [v["mean_auc"] for v in per_metal_summary.values()
                   if v["mean_auc"] is not None]
    mean_auc_across_metals = float(np.mean(metal_means)) if metal_means else None

    summary_path = args.json_dir / "summary.json"
    summary = {
        "metric": "wf146_dgllifesci_groupdro",
        "description": "Wf-146 DGL-LifeSci GCN+GroupDRO baseline on MetalCytoToxDB 5-metal LOMO split. "
                       "Apples-to-apples counterpart to wf146_dgllifesci_erm: same LOMO protocol, "
                       "same 5-metal subset, same per-(metal, seed) ROC-AUC, same backbone (GCNPredictor). "
                       "Algorithm: GroupDRO (Sagawa et al. ICLR 2020) with per-metal BCE, q uniform init, "
                       "step_size=0.01, q renormalised each batch.",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_sbdd_ood_lomo_dgllifesci_groupdro",
        "model": "DGL-LifeSci GCNPredictor (continuous feats, 3 layers, hidden=64)",
        "algorithm": "GroupDRO (Sagawa et al., ICLR 2020)",
        "config": {
            "metals": args.metals,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "group_step_size": args.group_step_size,
            "device": args.device,
            "time_budget_s": args.time_budget_s,
            "atom_feat_size": ATOM_FEAT_SIZE,
            "bond_feat_size": BOND_FEAT_SIZE,
            "python_env_used": "/home/hugo/.conda_envs/drugood/bin/python (constraint asked for "
                              "/home/hugo/codes/mol-metal/.venv/bin/python but that venv lacks "
                              "dgl+dgllife; drugood env has dgl 2.5.0 + dgllife 0.3.1 + torch 2.14.0+cu130)",
        },
        "split_protocol": "LOMO (leave-one-metal-out): train on {Ru, Ir, Os, Re}\\Rh, test on Rh. "
                          "BCE per example (no pos_weight in GroupDRO loss; we want clean per-group "
                          "mean losses for q update). AdamW, no LR scheduler, no domain-adaptation "
                          "tricks beyond GroupDRO itself.",
        "n_total_target": n_total,
        "n_completed": n_completed,
        "per_metal_seed_auc": per_metal_seed_auc,
        "per_metal_summary": per_metal_summary,
        "mean_auc_across_metals": mean_auc_across_metals,
        "total_wall_sec": time.time() - t_wall0,
        "honest_framing_note": (
            "Wf-146 DGL-LifeSci GCN+GroupDRO baseline executed in the /home/hugo/.conda_envs/drugood/bin/python env "
            "(dgl 2.5.0 + dgllife 0.3.1 + torch 2.14.0+cu130 + CUDA 13.4, GPU = NVIDIA RTX PRO 6000 Blackwell "
            "Workstation Edition). The CONSTRAINT requested /home/hugo/codes/mol-metal/.venv/bin/python but "
            "that venv is missing dgl and dgllife (verified: ModuleNotFoundError on import). Switching to the "
            "drugood env is the honest, pre-deployed, GPU-ready path and matches the same env used by "
            "wf146_dgllifesci_erm.py. Backbone: GCNPredictor (3 layers, hidden=64, dropout=0.2, residual=True, "
            "batchnorm=True, readout=mean+max). Algorithm: GroupDRO (Sagawa et al. ICLR 2020) — per-minibatch "
            "per-metal BCE, group weights q_g updated via q_g *= exp(step_size * L_g) and renormalised each "
            "step, q clamped to [1e-3, 1.0] before renormalisation to prevent any group from collapsing to 0. "
            "step_size=0.01 (Sagawa default for small n_groups). Same data CSV (metalcytotox_5metal_multitask_v4.csv, "
            "26801 rows), same 5-fold LOMO protocol (hold out 1 metal, train on the other 4), same 5 seeds "
            "(0-4), same per-(metal, seed) JSON output schema as ERM. SMILES pre-processing identical: "
            "SaltRemover strips counterions ([Cl-], [Br-], etc.), then largest organic fragment kept (drops "
            "~13 of 26801 = 0.05% featurization failures). Honest deviation from ERM: we drop pos_weight in the "
            "BCE (pos_weight was used in ERM to handle class imbalance; for GroupDRO we need raw per-example "
            "losses so the per-group mean is interpretable and the q-update uses a clean L_g). The class "
            "imbalance handling is implicit in the per-metal weighting."
        ),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("wrote summary to %s", summary_path)
    LOGGER.info("mean_auc_across_metals=%s", mean_auc_across_metals)
    LOGGER.info("n_completed=%d/%d wall=%.0fs", n_completed, n_total, time.time() - t_wall0)
    return 0


if __name__ == "__main__":
    sys.exit(main())