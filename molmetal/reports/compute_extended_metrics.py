"""Compute extended metrics for the honest baseline report.

Extended metrics:
  1. Per-cell-line AUC for Ru (top-5 cell lines by frequency)
  2. Diversity: mean pairwise Tanimoto distance among top-10 predictions
  3. Novelty: max Tanimoto between top-10 predictions and training set
  4. Calibration: Brier score + calibration curve

Usage:
  source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
  python molmetal/reports/compute_extended_metrics.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

from molmetal.baselines.eval_utils import morgan_features, per_cell_line_auc
from molmetal.baselines.morgan_xgb import MORGAN_RADIUS, MORGAN_NBITS, XGB_PARAMS
from molmetal.baselines.rf_baseline import RF_PARAMS
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import (
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TemporalSplitter,
)

RDLogger.DisableLog("rdApp.*")

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"

SPLITS_TO_RUN = [
    ("random", lambda seed: RandomSplitter(seed=seed)),
    ("ligand_dedup", lambda seed: LigandDeduplicatedSplitter(strategy="largest_first", seed=seed)),
    ("scaffold", lambda seed: ScaffoldSplitter(strategy="largest_first", seed=seed)),
    ("temporal", lambda seed: TemporalSplitter(cutoff_year=2024)),
]

METALS = ["Ru", "Ir"]
MODELS = ["xgb", "rf"]
SEED = 42
TOP_N_CELL_LINES = 5


# ---------------------------------------------------------------------------
# Tanimoto helpers
# ---------------------------------------------------------------------------
def _morgan_fp(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS):
    """Return Morgan FP as uint8 numpy array, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        return arr
    except Exception:
        return None


def _tanimoto_fp_arrays(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Bit-vector Tanimoto between two uint8 arrays."""
    inter = int(np.bitwise_and(fp1, fp2).sum())
    union = int(np.bitwise_or(fp1, fp2).sum())
    if union == 0:
        return 0.0
    return inter / union


def _tanimoto_to_set_arrays(fp: np.ndarray, fp_list: List[np.ndarray]) -> np.ndarray:
    """Vectorised Tanimoto of one fp to a list of fps. Returns array of sims."""
    if not fp_list:
        return np.array([0.0], dtype=np.float64)
    stack = np.stack(fp_list, axis=0).astype(np.uint16)
    a = fp.astype(np.uint16)
    inter = np.bitwise_and(stack, a).sum(axis=1)
    union = np.bitwise_or(stack, a).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(union > 0, inter / union, 0.0)
    return sims.astype(np.float64)


# ---------------------------------------------------------------------------
# Extended metric computations
# ---------------------------------------------------------------------------
def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true.astype(float)) ** 2))


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return {"bin_centers": [], "bin_predicted": [], "bin_actual": [], "bin_counts": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    centers = (bins[:-1] + bins[1:]) / 2
    predicted = np.zeros(n_bins, dtype=np.float64)
    actual = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_indices == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            predicted[b] = float(y_prob[mask].mean())
            actual[b] = float(y_true[mask].mean())

    return {
        "bin_centers": centers.tolist(),
        "bin_predicted": predicted.tolist(),
        "bin_actual": actual.tolist(),
        "bin_counts": counts.tolist(),
    }


def compute_top10_diversity(smiles_list: List[str], y_prob: np.ndarray) -> float:
    """Mean pairwise Tanimoto distance among top-10 predictions. High distance = diverse."""
    top_k = min(10, len(smiles_list))
    if top_k < 2:
        return float("nan")

    top_indices = np.argpartition(-y_prob, top_k - 1)[:top_k]
    fps = []
    for idx in top_indices:
        fp = _morgan_fp(smiles_list[int(idx)])
        if fp is not None:
            fps.append(fp)

    if len(fps) < 2:
        return float("nan")

    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = _tanimoto_fp_arrays(fps[i], fps[j])
            dists.append(1.0 - sim)

    return float(np.mean(dists))


def compute_top10_novelty(top10_smiles: List[str], train_smiles: List[str]) -> float:
    """Mean of max Tanimoto between each top-10 and closest training neighbour.
    High = good (dissimilar to training set = novel)."""
    train_fps = [fp for smi in train_smiles if (fp := _morgan_fp(smi)) is not None]
    if not train_fps:
        return float("nan")

    max_sims = []
    for smi in top10_smiles:
        fp = _morgan_fp(smi)
        if fp is None:
            continue
        sims = _tanimoto_to_set_arrays(fp, train_fps)
        max_sims.append(float(sims.max()))

    if not max_sims:
        return float("nan")
    return float(np.mean(max_sims))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_dataset(metal: str):
    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    smiles = ds.smiles
    y_all = ds.active.astype(int)
    cell_lines = ds.df["Cell_line"].astype(str).to_numpy()
    keep = np.array([bool(s) for s in smiles], dtype=bool)
    return ds, smiles[keep], y_all[keep], cell_lines[keep]


def run_pipeline(
    metal: str,
    model_name: str,
    split_name: str,
    splitter_fn,
) -> dict:
    """Train model and compute all extended metrics."""
    # Load data
    ds, smiles_all, y_all, cell_lines_all = load_dataset(metal)

    # Apply split
    split_result = splitter_fn(SEED)(ds)

    # Map to kept indices only (some ds indices may be out-of-range after keep mask)
    keep = np.array([bool(s) for s in ds.smiles], dtype=bool)
    keep_idx = np.where(keep)[0]

    def _map_idx(idxs):
        # idxs are absolute indices into the original (unfiltered) ds
        # we need indices into the filtered (keep) array
        idx_set = set(int(i) for i in idxs)
        return np.array([j for j, ki in enumerate(keep_idx) if ki in idx_set], dtype=np.int64)

    idx_train = _map_idx(split_result.train_idx)
    idx_val = _map_idx(split_result.val_idx)
    idx_test = _map_idx(split_result.test_idx)

    # Recompute x_all after filtering
    x_all = morgan_features(smiles_all, MORGAN_RADIUS, MORGAN_NBITS)
    x_train, y_train = x_all[idx_train], y_all[idx_train]
    x_val, y_val = x_all[idx_val], y_all[idx_val]
    x_test, y_test = x_all[idx_test], y_all[idx_test]
    cl_test = cell_lines_all[idx_test]
    smiles_test = [smiles_all[int(i)] for i in idx_test]
    smiles_train = [smiles_all[int(i)] for i in idx_train]

    # Train
    if model_name == "xgb":
        from xgboost import XGBClassifier
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - pos))
        params = dict(XGB_PARAMS)
        params["scale_pos_weight"] = float(neg) / float(pos)
        params["random_state"] = SEED
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        from sklearn.ensemble import RandomForestClassifier
        params = dict(RF_PARAMS)
        params["random_state"] = SEED
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)

    test_score = model.predict_proba(x_test)[:, 1]

    # Top-10 predicted molecules (by probability)
    top10_k = min(10, len(smiles_test))
    top10_indices = np.argpartition(-test_score, top10_k - 1)[:top10_k]
    top10_smiles = [smiles_test[int(i)] for i in top10_indices]

    # Compute extended metrics
    brier = compute_brier_score(y_test, test_score)
    calib = compute_calibration_curve(y_test, test_score)
    diversity = compute_top10_diversity(smiles_test, test_score)
    novelty = compute_top10_novelty(top10_smiles, smiles_train)
    cl_auc = per_cell_line_auc(cl_test, y_test, test_score, min_count=30)

    # Top-N cell lines (Ru only)
    top_cell_lines = {}
    if metal == "Ru":
        cl_counts: Dict[str, int] = defaultdict(int)
        for cl in cl_test:
            cl_counts[str(cl)] += 1
        sorted_cls = sorted(cl_counts.items(), key=lambda x: -x[1])[:TOP_N_CELL_LINES]
        for cl_name, _ in sorted_cls:
            if cl_name in cl_auc:
                top_cell_lines[cl_name] = cl_auc[cl_name]

    return {
        "metal": metal,
        "model": model_name,
        "split": split_name,
        "n_test": int(len(y_test)),
        "brier_score": round(brier, 4),
        "calibration": calib,
        "diversity_top10": round(diversity, 4) if diversity == diversity else None,
        "novelty_top10_vs_train": round(novelty, 4) if novelty == novelty else None,
        "per_cell_line_auc": {k: round(v, 4) for k, v in cl_auc.items()},
        "top_cell_lines_ru": {k: round(v, 4) for k, v in top_cell_lines.items()},
    }


def main() -> dict:
    t0 = time.time()
    all_results: Dict = {}

    for metal in METALS:
        print(f"\n=== {metal} ===")
        all_results[metal] = {}

        for model_name in MODELS:
            print(f"  {model_name}...", end="", flush=True)
            all_results[metal][model_name] = {}

            for split_name, splitter_fn in SPLITS_TO_RUN:
                key = split_name
                print(f" {split_name}", end="", flush=True)
                try:
                    result = run_pipeline(metal, model_name, split_name, splitter_fn)
                    all_results[metal][model_name][key] = result
                    print(f"[OK]", end="", flush=True)
                except Exception as ex:
                    print(f"[ERR:{ex}]", end="", flush=True)
                    all_results[metal][model_name][key] = {"error": str(ex)}
            print()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save
    out_path = REPORTS_DIR / "extended_metrics_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
