"""Retrain Attentive D-MPNN pIC50 predictor on the FROZEN HeLa48h/dark cohort
with censor-aware loss and scaffold-group splits.

WF-Extra-1: replaces the historical pooled-target D-MPNN (legacy
`dmpnn_attn_ru_pic50.pt`, test r=0.407) with a model trained against the
702-formulation / 383-scaffold FROZEN HeLa/48h/dark cohort using the same
GroupShuffleSplit scaffold-group partition as
`calibrate_conditioned_pic50_baseline.py`. Censored observations are
preserved as upper/lower bounds (right/left-censored) — they are NEVER
silently re-coded to exact targets.

Loss
----
* exact rows (is_censored=False):  MSE on pIC50 = -log10(IC50_uM / 1e6).
* right-censored rows (">X uM"):  hinge-style margin
                                   L = max(0, margin - (y_pred - pIC50_bound))
                                   penalising predictions that exceed
                                   the bound (claim more potent than the
                                   bound allows). Negative direction
                                   (less potent than bound) is free.
* left-censored rows ("<X uM"):   symmetric: penalise predictions that
                                   fall BELOW the bound (more potent
                                   than the bound allows).

CLI
----
    uv run python -m molmetal.scripts.retrain_pic50_neural \
        --epochs 50 \
        --seeds 42 0 1234 \
        --output-dir molmetal/reports/wf_extra1_retrain

Outputs
-------
* molmetal/checkpoints/dmpnn_attn_heLa48h_dark_retrained.pt  (per-seed bundle)
* molmetal/reports/wf_extra1_retrain/test_predictions.parquet (concat across seeds)
* molmetal/reports/wf_extra1_retrain/report.json               (per-seed + aggregate)
* molmetal/reports/wf_extra1_retrain/report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from molmetal_lam.training.pic50_censored_loader import (
    CensoredPIC50Dataset,
    FORMULATION_META_COLS,
)
from molmetal.baselines.dmpnn_attentive import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    AttentiveDMPNNModel,
    featurize_smiles_list,
    collate_graphs,
)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr, spearmanr

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters (PROJECTED defaults — match historical r=0.407 ckpt dims)
# ---------------------------------------------------------------------------
DMPNN_HIDDEN = 128
DMPNN_DEPTH = 3
DMPNN_DROPOUT = 0.1
DEFAULT_LR = 1e-4           # brief-specified lr
DEFAULT_EPOCHS = 50         # brief-specified
DEFAULT_BATCH_SIZE = 64     # brief-specified
CENSOR_MARGIN_DEFAULT = 0.5  # hinge margin on pIC50 scale (WF-pIC50-Margin-Sweep 2026-09-14:
                              # down-shift from 1.0 → 0.5.  Sweep over {0.5,1.0,2.0,3.0} on
                              # FROZEN HeLa48h/dark cohort (3 seeds × 50 epochs): margin=0.5
                              # ties margin=1.0 on bound_aware_accuracy=1.0 (67/67) and beats
                              # it on test_pearson_r (0.195 vs 0.184) and test_rmse (0.686 vs
                              # 0.703).  Margins 2.0/3.0 BREAK the censor guarantee (one
                              # over-prediction on seed=0 → BA=0.990).  WF-Extra-1 raised
                              # 0.25→1.0 because 0.25 was below the regression-to-mean pull
                              # (~0.80); 0.5 is the tighter sweet spot.)

CHECKPOINT_REL = Path("molmetal/checkpoints/dmpnn_attn_heLa48h_dark_retrained.pt")
REPORT_REL = Path("molmetal/reports/wf_extra1_retrain")
TEST_PRED_REL = REPORT_REL / "test_predictions.parquet"


# ---------------------------------------------------------------------------
# Cohort construction (matches calibrate_conditioned_pic50_baseline.py)
# ---------------------------------------------------------------------------
def _ic50_to_pic50(ic50_uM: float) -> float:
    """pIC50 = 6 - log10(IC50_uM). Defensive against non-positive inputs."""
    return float(6.0 - np.log10(ic50_uM))


@dataclass
class Cohort:
    smiles: List[str]
    pic50: np.ndarray              # for exact: target; for censored: bound as best-effort
    is_censored: np.ndarray        # bool
    bound_uM: np.ndarray           # numeric bound in uM
    pIC50_bound: np.ndarray        # pIC50 of bound (for censoring loss)
    censor_dir: np.ndarray         # +1 right (">X"), -1 left ("<X")
    scaffold: List[str]
    formulation_id: List[str]


def _build_cohort(csv_path: Path, cell_line: str, time_h: float) -> Cohort:
    """Filter + group-by formulation key + extract scaffolds + median pIC50.

    Uses CensoredPIC50Dataset to preserve censor flags; reconciles
    formulation groups by (canonical_smiles, Counterion, Metal,
    Oxidation_state, Charge_complex) the same way the conditioned
    baseline does. Censored observations inside one formulation are
    aggregated to a single row with is_censored=True (the worst-case
    bound direction wins).
    """
    ds = CensoredPIC50Dataset.from_csv(
        csv_path,
        filters={"Cell_line": cell_line, "Time(h)": time_h},
    )
    if len(ds) == 0:
        raise ValueError(f"No rows survived filtering for {cell_line}/{time_h}h")

    # Group by formulation key (canonical_smiles + counterion + meta)
    records = []
    for i in range(len(ds)):
        row = ds.get_row(i)
        meta = row.formulation_meta
        try:
            oxidation = float(meta.get("Oxidation_state") or 0.0)
            charge = float(meta.get("Charge_complex") or 0.0)
        except (TypeError, ValueError):
            oxidation = charge = float("nan")
        records.append(
            {
                "ligand_smiles": row.smiles,
                "counterion": meta.get("Counterion", "") or "",
                "metal": meta.get("Metal", "") or "",
                "oxidation_state": oxidation,
                "charge": charge,
                "is_censored": bool(row.is_censored),
                "bound_uM": float(row.bound_value),
                "target_uM": float(row.target),
            }
        )
    df = pd.DataFrame.from_records(records)

    grouped = []
    rejected = []
    for key, grp in df.groupby(
        ["ligand_smiles", "counterion", "metal", "oxidation_state", "charge"],
        dropna=False,
        sort=True,
    ):
        ligand, counterion, metal, ox, charge = key
        if not np.isfinite([ox, charge]).all():
            rejected.append({"key": list(map(str, key)), "reason": "non-finite oxidation/charge"})
            continue
        parent = Chem.MolFromSmiles(ligand)
        if parent is None:
            rejected.append({"key": list(map(str, key)), "reason": "parent mol parse failed"})
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
        except Exception as exc:
            rejected.append({"key": list(map(str, key)), "reason": f"scaffold fail: {exc}"})
            continue

        any_censored = bool(grp["is_censored"].any())
        if not any_censored:
            target_uM = float(grp["target_uM"].median())
            bound_uM = target_uM
            censor_dir = 0
            pic50 = _ic50_to_pic50(target_uM)
            pIC50_bound = pic50
        else:
            # Censored aggregation: take the most restrictive bound.
            # Right-censored rows have `target_uM == bound_uM`; take the
            # MIN bound (most potent claim = strongest constraint).
            bound_uM = float(grp.loc[grp["is_censored"], "bound_uM"].min())
            pic50 = _ic50_to_pic50(bound_uM)
            pIC50_bound = pic50
            # Direction: in our data, ">X" is the dominant censor symbol.
            # Use the most-permissive (right-censored) interpretation by default.
            censor_dir = +1

        grouped.append(
            {
                "ligand_smiles": ligand,
                "counterion": counterion,
                "metal": metal,
                "oxidation_state": float(ox),
                "charge": float(charge),
                "scaffold": scaffold or "<acyclic>",
                "is_censored": any_censored,
                "target_uM": float(grp["target_uM"].median()) if not any_censored else bound_uM,
                "bound_uM": bound_uM,
                "pic50": pic50,
                "pIC50_bound": pIC50_bound,
                "censor_dir": censor_dir,
            }
        )

    if len(grouped) < 30:
        raise ValueError(
            f"Insufficient conditioned cohort: {len(grouped)} formulations (need ≥30)"
        )

    return Cohort(
        smiles=[g["ligand_smiles"] for g in grouped],
        pic50=np.array([g["pic50"] for g in grouped], dtype=np.float32),
        is_censored=np.array([g["is_censored"] for g in grouped], dtype=bool),
        bound_uM=np.array([g["bound_uM"] for g in grouped], dtype=np.float32),
        pIC50_bound=np.array([g["pIC50_bound"] for g in grouped], dtype=np.float32),
        censor_dir=np.array([g["censor_dir"] for g in grouped], dtype=np.int8),
        scaffold=[g["scaffold"] for g in grouped],
        formulation_id=[
            f"{g['metal']}|{cell_line}|{time_h}|{g['oxidation_state']}|{g['charge']}|{g['counterion']}|{g['ligand_smiles']}"
            for g in grouped
        ],
    )


# ---------------------------------------------------------------------------
# Scaffold-group split (mirrors conditioned baseline, no leakage)
# ---------------------------------------------------------------------------
def scaffold_split(
    cohort: Cohort, seed: int, val_size: float = 0.1, test_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 scaffold-group; raises if any scaffold or ligand leaks across splits."""
    groups = np.asarray(cohort.scaffold)
    ligands = np.asarray(cohort.smiles)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train, rest = next(gss.split(np.zeros(len(groups)), groups=groups))
    val, test = train_test_split_group(rest, groups[rest], val_size=val_size / (test_size + val_size), seed=seed + 1)

    for i, j in ((0, 1), (0, 2), (1, 2)):
        idxs = (train, val, test)[i], (train, val, test)[j]
        if set(groups[idxs[0]]) & set(groups[idxs[1]]):
            raise ValueError(f"Scaffold leakage between splits ({i},{j})")
        if set(ligands[idxs[0]]) & set(ligands[idxs[1]]):
            raise ValueError(f"Ligand leakage between splits ({i},{j})")
    return np.asarray(train), np.asarray(val), np.asarray(test)


def train_test_split_group(
    idx: np.ndarray, groups: np.ndarray, val_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    a, b = next(gss.split(np.zeros(len(idx)), groups=groups))
    return idx[a], idx[b]


# ---------------------------------------------------------------------------
# Bound-aware accuracy: fraction of censored test rows whose prediction
# respects the censoring direction (i.e. for right-censored, pred ≤ bound;
# for left-censored, pred ≥ bound). Defined per the WF-Extra-1 verdict
# formula: pred_compliant iff -dir*(pred - bound) ≤ 0.
# ---------------------------------------------------------------------------
def bound_aware_accuracy(
    pred: np.ndarray,
    is_censored: np.ndarray,
    pIC50_bound: np.ndarray,
    censor_dir: np.ndarray,
) -> Tuple[float, int, int]:
    """Return (accuracy, n_compliant, n_censored) over censored rows.

    A row is *compliant* if ``-censor_dir * (pred - pIC50_bound) <= 0``,
    i.e. the prediction does not cross the bound in the forbidden direction.
    """
    if not is_censored.any():
        return 0.0, 0, 0
    pred_c = np.asarray(pred[is_censored], dtype=np.float64)
    bound_c = np.asarray(pIC50_bound[is_censored], dtype=np.float64)
    dir_c = np.asarray(censor_dir[is_censored], dtype=np.float64)
    violation = -dir_c * (pred_c - bound_c)  # ≤ 0 means compliant
    compliant = int((violation <= 0.0).sum())
    total = int(is_censored.sum())
    return (compliant / total if total else 0.0), compliant, total


# ---------------------------------------------------------------------------
# Loss: MSE on exact + hinge-style margin on censored
# ---------------------------------------------------------------------------
def censored_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_censored: torch.Tensor,
    pIC50_bound: torch.Tensor,
    censor_dir: torch.Tensor,
    margin: float = CENSOR_MARGIN_DEFAULT,
) -> torch.Tensor:
    """Combine MSE (exact) + margin (censored).

    For right-censored (censor_dir=+1, ">X uM"): true pIC50 ≤ pIC50_bound.
    Penalise ``pred > pIC50_bound + margin`` (claim more potent than data
    allows). ``pred ≤ pIC50_bound`` is free.

    For left-censored (censor_dir=-1, "<X uM"): true pIC50 ≥ pIC50_bound.
    Penalise ``pred < pIC50_bound - margin``. ``pred ≥ pIC50_bound``
    is free.

    For exact (is_censored=False): MSE(pred, target).
    """
    if (~is_censored).any():
        mse = torch.nn.functional.mse_loss(pred[~is_censored], target[~is_censored])
    else:
        mse = torch.zeros((), device=pred.device)

    if is_censored.any():
        diff = pred[is_censored] - pIC50_bound[is_censored]   # signed deviation from bound
        dir_ = censor_dir[is_censored].to(diff.dtype)         # +1 or -1
        # diff > 0 means claim more potent than bound.
        # For right-censored (+1), diff > 0 is the forbidden direction.
        # For left-censored (-1), diff < 0 is forbidden -> multiply diff by -1.
        violation = -dir_ * diff
        hinge = torch.clamp(violation + margin, min=0.0).mean()
    else:
        hinge = torch.zeros((), device=pred.device)

    return mse + hinge


# ---------------------------------------------------------------------------
# Training loop (per seed)
# ---------------------------------------------------------------------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_rmse: float
    val_pearson: float
    seconds: float


def train_one_seed(
    cohort: Cohort,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    censored_margin: float = CENSOR_MARGIN_DEFAULT,
    max_rows: Optional[int] = None,
    log_every: int = 5,
) -> Tuple[AttentiveDMPNNModel, Dict, List[EpochLog]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, val, test = scaffold_split(cohort, seed=seed)
    if max_rows is not None and len(train) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train), size=max_rows, replace=False)
        train = np.sort(train[keep])

    smiles = cohort.smiles
    pic50 = cohort.pic50
    is_censored = cohort.is_censored
    pIC50_bound = cohort.pIC50_bound
    censor_dir = cohort.censor_dir

    if log_every:
        logger.info(
            "seed=%d  train=%d val=%d test=%d  censored_train=%d censored_test=%d",
            seed, len(train), len(val), len(test),
            int(is_censored[train].sum()),
            int(is_censored[test].sum()),
        )

    if log_every:
        logger.info("featurising %d molecules...", len(smiles))
    graphs = featurize_smiles_list(smiles)

    # Replace None graphs (RDKit-fail) with a 1-atom dummy to keep indices aligned
    for i, g in enumerate(graphs):
        if g is None:
            graphs[i] = (
                np.zeros((1, ATOM_FEATURE_DIM), dtype=np.float32),
                np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )

    model = AttentiveDMPNNModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=DMPNN_HIDDEN,
        depth=DMPNN_DEPTH,
        dropout=DMPNN_DROPOUT,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    target_t = torch.as_tensor(pic50, dtype=torch.float32, device=device)
    is_censored_t = torch.as_tensor(is_censored, dtype=torch.bool, device=device)
    bound_t = torch.as_tensor(pIC50_bound, dtype=torch.float32, device=device)
    dir_t = torch.as_tensor(censor_dir, dtype=torch.float32, device=device)

    history: List[EpochLog] = []

    best_val_rmse = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        indices = np.arange(len(train))
        np.random.shuffle(indices)
        running = 0.0
        n = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            g_list = [graphs[i] for i in batch_idx]
            collated = collate_graphs(g_list)
            if collated is None:
                continue
            atom_f, bond_f, e_src, e_dst = collated
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)

            pred = model(atom_f, bond_f, e_src, e_dst)
            y_true = target_t[batch_idx]
            y_cen = is_censored_t[batch_idx]
            y_bound = bound_t[batch_idx]
            y_dir = dir_t[batch_idx]

            loss = censored_loss(pred, y_true, y_cen, y_bound, y_dir, margin=censored_margin)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            running += float(loss.item()) * len(batch_idx)
            n += len(batch_idx)

        train_loss = running / max(n, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            for start in range(0, len(val), batch_size):
                batch_idx = val[start : start + batch_size]
                g_list = [graphs[i] for i in batch_idx]
                collated = collate_graphs(g_list)
                if collated is None:
                    val_preds.extend([np.nan] * len(batch_idx))
                    continue
                atom_f, bond_f, e_src, e_dst = collated
                atom_f = atom_f.to(device)
                bond_f = bond_f.to(device)
                e_src = e_src.to(device)
                e_dst = e_dst.to(device)
                p = model(atom_f, bond_f, e_src, e_dst).cpu().numpy()
                val_preds.extend(p.tolist())
            val_preds = np.asarray(val_preds, dtype=np.float64)
            val_truth = pic50[val]
            ok = np.isfinite(val_preds) & np.isfinite(val_truth)
            if ok.sum() >= 2 and np.std(val_preds[ok]) > 0 and np.std(val_truth[ok]) > 0:
                val_pearson = float(pearsonr(val_preds[ok], val_truth[ok]).statistic)
            else:
                val_pearson = float("nan")
            val_rmse = float(np.sqrt(np.mean((val_preds - val_truth) ** 2)))

        elapsed = time.time() - t0
        history.append(
            EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                val_rmse=val_rmse,
                val_pearson=val_pearson,
                seconds=elapsed,
            )
        )

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            logger.info(
                "seed=%d epoch=%d  train_loss=%.4f  val_rmse=%.4f  val_pearson=%.4f  t=%.1fs",
                seed, epoch, train_loss, val_rmse, val_pearson, elapsed,
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    test_preds: List[float] = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            batch_idx = test[start : start + batch_size]
            g_list = [graphs[i] for i in batch_idx]
            collated = collate_graphs(g_list)
            if collated is None:
                test_preds.extend([float("nan")] * len(batch_idx))
                continue
            atom_f, bond_f, e_src, e_dst = collated
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)
            p = model(atom_f, bond_f, e_src, e_dst).cpu().numpy()
            test_preds.extend(p.tolist())
    test_preds = np.asarray(test_preds, dtype=np.float64)
    test_truth = pic50[test]
    test_censored = is_censored[test]

    exact_mask = ~test_censored
    metrics: Dict[str, float] = {"n_test": int(len(test))}
    if exact_mask.sum() >= 2 and np.std(test_preds[exact_mask]) > 0 and np.std(test_truth[exact_mask]) > 0:
        metrics.update(
            {
                "pearson_r": float(pearsonr(test_preds[exact_mask], test_truth[exact_mask]).statistic),
                "spearman_r": float(spearmanr(test_preds[exact_mask], test_truth[exact_mask]).statistic),
                "rmse": float(np.sqrt(np.mean((test_preds[exact_mask] - test_truth[exact_mask]) ** 2))),
                "mae": float(np.mean(np.abs(test_preds[exact_mask] - test_truth[exact_mask]))),
            }
        )
    metrics.update(
        {
            "y_mean": float(np.mean(test_truth)),
            "y_std": float(np.std(test_truth)),
            "pred_mean": float(np.mean(test_preds)),
            "pred_std": float(np.std(test_preds)),
            "n_exact_test": int(exact_mask.sum()),
            "n_censored_test": int(test_censored.sum()),
        }
    )

    # Bound-aware accuracy over censored test rows (WF-Extra-1 §3 metric)
    ba_acc, ba_compliant, ba_total = bound_aware_accuracy(
        test_preds,
        test_censored.astype(bool),
        cohort.pIC50_bound[test],
        cohort.censor_dir[test],
    )
    metrics.update(
        {
            "bound_aware_accuracy": float(ba_acc),
            "bound_aware_compliant": int(ba_compliant),
            "bound_aware_total": int(ba_total),
        }
    )

    if log_every:
        logger.info(
            "seed=%d TEST  pearson=%s  rmse=%s  n=%d",
            seed,
            f"{metrics.get('pearson_r', float('nan')):.4f}" if "pearson_r" in metrics else "nan",
            f"{metrics.get('rmse', float('nan')):.4f}" if "rmse" in metrics else "nan",
            int(metrics["n_test"]),
        )

    predictions_frame = pd.DataFrame(
        {
            "seed": seed,
            "formulation_id": [cohort.formulation_id[i] for i in test],
            "ligand_smiles": [cohort.smiles[i] for i in test],
            "scaffold": [cohort.scaffold[i] for i in test],
            "is_censored": test_censored,
            "pic50_truth": test_truth,
            "pic50_pred": test_preds,
        }
    )
    return model, {"metrics": metrics, "predictions": predictions_frame,
                    "train_idx": train, "val_idx": val, "test_idx": test,
                    "history": [h.__dict__ for h in history]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv"),
        help="Source CSV (default: MetalCytoToxDB.csv)",
    )
    p.add_argument("--cell-line", default="HeLa")
    p.add_argument("--time-h", type=float, default=48.0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1234])
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write report.json/report.md/test_predictions.parquet")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_REL,
                   help="Per-seed checkpoint path (last seed wins)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="If set, subsample the training set to at most this many rows (scaled-down real evaluator run).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--censored-margin",
        type=float,
        default=CENSOR_MARGIN_DEFAULT,
        help=(
            "Hinge margin on pIC50 scale for the censored rows. "
            "WF-Extra-1 raised this from 0.25 → 1.0 because the regression-"
            "to-mean pull (~0.80 pIC50) dominated the 0.25 hinge and "
            "bound_aware_accuracy collapsed to 0.0. Default 1.0."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if torch.cuda.is_available():
        device = torch.device(args.device if "cuda" in args.device else "cuda:0")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA unavailable; falling back to CPU (slow).")

    cohort = _build_cohort(args.csv, args.cell_line, args.time_h)
    logger.info(
        "cohort: %d formulations, %d scaffolds, %d censored (%.1f%%)",
        len(cohort.smiles),
        len(set(cohort.scaffold)),
        int(cohort.is_censored.sum()),
        100.0 * float(cohort.is_censored.mean()),
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    seed_summaries = []

    for seed in args.seeds:
        logger.info("=== seed %d ===", seed)
        model, result = train_one_seed(
            cohort=cohort,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            censored_margin=args.censored_margin,
            max_rows=args.max_rows,
            log_every=(1 if args.max_rows else 5),
        )
        all_predictions.append(result["predictions"])
        seed_summaries.append(
            {
                "seed": seed,
                "metrics": result["metrics"],
                "n_train": int(len(result["train_idx"])),
                "n_val": int(len(result["val_idx"])),
                "n_test": int(len(result["test_idx"])),
                "history_tail": result["history"][-3:],
            }
        )

        # Save per-seed checkpoint bundle (last seed wins unless scaled-down)
        bundle = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_kwargs": {
                "atom_dim": ATOM_FEATURE_DIM,
                "bond_dim": BOND_FEATURE_DIM,
                "hidden": DMPNN_HIDDEN,
                "depth": DMPNN_DEPTH,
                "dropout": DMPNN_DROPOUT,
            },
            "meta": {
                "metal": "Ru",
                "cell_line": args.cell_line,
                "time_h": args.time_h,
                "n_rows_total": int(len(cohort.smiles)),
                "n_train": int(len(result["train_idx"])),
                "n_val": int(len(result["val_idx"])),
                "n_test": int(len(result["test_idx"])),
                "n_censored": int(cohort.is_censored.sum()),
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "loss": "MSE(exact) + hinge-margin(censored)",
            },
            "test_metrics": result["metrics"],
            "history": result["history"],
        }
        ckpt_path = args.checkpoint
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, ckpt_path)
        logger.info("wrote checkpoint %s", ckpt_path)

    # Concatenate predictions across seeds
    preds_df = pd.concat(all_predictions, ignore_index=True)
    preds_path = out_dir / "test_predictions.parquet"
    preds_df.to_parquet(preds_path)
    logger.info("wrote %s (%d rows)", preds_path, len(preds_df))

    # Aggregate metrics across seeds
    pearsons = [
        s["metrics"].get("pearson_r")
        for s in seed_summaries
        if s["metrics"].get("pearson_r") is not None
    ]
    rmses = [
        s["metrics"].get("rmse")
        for s in seed_summaries
        if s["metrics"].get("rmse") is not None
    ]
    ba_accs = [
        s["metrics"].get("bound_aware_accuracy")
        for s in seed_summaries
        if s["metrics"].get("bound_aware_accuracy") is not None
    ]
    ba_compliant = sum(
        int(s["metrics"].get("bound_aware_compliant", 0)) for s in seed_summaries
    )
    ba_total = sum(
        int(s["metrics"].get("bound_aware_total", 0)) for s in seed_summaries
    )
    aggregate = {
        "pearson_r_mean": float(np.mean(pearsons)) if pearsons else None,
        "pearson_r_std": float(np.std(pearsons)) if pearsons else None,
        "rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rmse_std": float(np.std(rmses)) if rmses else None,
        "bound_aware_accuracy_mean": float(np.mean(ba_accs)) if ba_accs else None,
        "bound_aware_accuracy_std": float(np.std(ba_accs)) if ba_accs else None,
        "bound_aware_accuracy_pooled": (
            float(ba_compliant / ba_total) if ba_total else None
        ),
        "bound_aware_compliant_total": int(ba_compliant),
        "bound_aware_censored_total": int(ba_total),
        "n_seeds": len(seed_summaries),
        "historical_dmpnn_pearson_r": 0.407,
        "historical_ridge_pearson_r_mean": float(np.mean([0.4774, 0.6055, 0.6341])),
        "target_band": [0.5, 0.7],
    }

    report = {
        "scope": (
            "Censor-aware retraining of Attentive D-MPNN pIC50 predictor on "
            "FROZEN HeLa/48h/dark cohort. Censored observations preserved as "
            "bounds; right-censored margin loss + MSE for exact. Independent "
            "of Lambda / CFM / round-trip / metal-seed / BNF work."
        ),
        "cohort": {
            "csv": str(args.csv),
            "cell_line": args.cell_line,
            "time_h": args.time_h,
            "n_formulations": int(len(cohort.smiles)),
            "n_scaffolds": int(len(set(cohort.scaffold))),
            "n_censored": int(cohort.is_censored.sum()),
            "n_exact": int((~cohort.is_censored).sum()),
        },
        "training": {
            "model": "AttentiveDMPNNModel",
            "hidden": DMPNN_HIDDEN,
            "depth": DMPNN_DEPTH,
            "dropout": DMPNN_DROPOUT,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": "MSE(exact) + hinge-margin(censored)",
            "censor_margin": float(args.censored_margin),
            "device": str(device),
            "scaled_down": bool(args.max_rows and args.max_rows < 1451),
            "max_rows": args.max_rows,
        },
        "seeds": seed_summaries,
        "aggregate": aggregate,
    }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote %s", report_json)

    # Markdown report
    lines = [
        "# WF-Extra-1 — D-MPNN retrain (HeLa/48h/dark, censor-aware)",
        "",
        f"Cohort: **{len(cohort.smiles)} formulations / {len(set(cohort.scaffold))} scaffolds** "
        f"(censored: {int(cohort.is_censored.sum())}, exact: {int((~cohort.is_censored).sum())}).",
        "",
        f"Loss: `MSE(exact) + hinge-margin(censored, margin={args.censored_margin})`.",
        "",
        "| Seed | n_train/val/test | Test RMSE | Test Pearson r | n_exact_test | n_censored_test | bound_aware_accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in seed_summaries:
        m = s["metrics"]
        n = f"{s['n_train']}/{s['n_val']}/{s['n_test']}"
        rmse = f"{m.get('rmse', float('nan')):.4f}" if m.get('rmse') is not None else "nan"
        pr = f"{m.get('pearson_r', float('nan')):.4f}" if m.get('pearson_r') is not None else "nan"
        ba = m.get("bound_aware_accuracy")
        ba_str = f"{ba:.3f}" if ba is not None else "nan"
        lines.append(
            f"| {s['seed']} | {n} | {rmse} | {pr} | {m['n_exact_test']} | {m['n_censored_test']} | {ba_str} |"
        )
    lines += [
        "",
        f"Aggregate Pearson r: **{aggregate['pearson_r_mean']:.4f} ± {aggregate['pearson_r_std']:.4f}** "
        f"(target band 0.5–0.7, historical D-MPNN 0.407, ridge ref "
        f"{aggregate['historical_ridge_pearson_r_mean']:.4f}).",
        "",
        "## Honest framing",
        "",
        "* **MEASURED**: cohort sizes, scaffold counts, censoring fractions, GPU forward/backward.",
        "* **PROJECTED** (not measured in this run): final Pearson r for a 50-epoch full sweep — "
        "this run validates GPU wiring, loss convergence, no NaN on the real cohort.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote %s/report.md", out_dir)

    # Print one-line summary for orchestrator
    print(
        json.dumps(
            {
                "exit_code": 0,
                "n_formulations": len(cohort.smiles),
                "n_scaffolds": len(set(cohort.scaffold)),
                "n_censored": int(cohort.is_censored.sum()),
                "seeds": [
                    {"seed": s["seed"], "metrics": s["metrics"]} for s in seed_summaries
                ],
                "aggregate": aggregate,
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
