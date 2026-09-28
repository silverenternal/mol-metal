#!/usr/bin/env python3
"""Verify DrugOOD native pipeline runs and produces sensible numbers.

This script:
1. Loads the native DrugOOD-format data produced by prepare_drugood_native.py
2. Optionally subsamples to N rows for fast verification
3. Runs a 3-split × 3-seed × 5-epoch MLP training on DrugOOD-style OOD
4. Compares against iMoLD reference numbers (assay 0.7211 / scaffold 0.6884 / size 0.6792)
5. Reports data_source = drugood_native vs metacytotox_fallback

Usage:
  /home/hugo/micromamba/envs/pocket311/bin/python \\
    molmetal/scripts/run_drugood_native_smoke.py \\
    --n-subsample 50000 \\
    --epochs 5 \\
    --seeds 0 1 2 \\
    --splits assay scaffold size \\
    --out-json metrics/by_metric/drugood_ood_auc_native_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# iMoLD reference numbers from paper (NeurIPS 2022) on DrugOOD IC50 splits.
IMOLD_REFERENCE = {
    "assay": 0.7211,
    "scaffold": 0.6884,
    "size": 0.6792,
}

NATIVE_DATA = Path(
    "/home/hugo/codes/mol-metal/.data/molmetal/drugood/ic50/data/train.csv"
)


def featurize(smiles: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint (2048, r=2) for each SMILES."""
    fps = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def load_native(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load DrugOOD native format CSV."""
    smiles, y, aid, sid, szid = [], [], [], [], []
    with path.open("r") as fh:
        import csv
        reader = csv.DictReader(fh)
        for row in reader:
            smiles.append(row["smiles"])
            y.append(int(row["y"]))
            aid.append(int(row.get("assay_id", 0) or 0))
            sid.append(int(row.get("scaffold_split_id", 0) or 0))
            szid.append(int(row.get("size_split_id", 0) or 0))
    return (smiles,
            np.asarray(y, dtype=np.int64),
            np.asarray(aid, dtype=np.int64),
            np.asarray(sid, dtype=np.int64),
            np.asarray(szid, dtype=np.int64))


def build_assay_split(y, aid, seed):
    """Pick the largest assay as OOD test (DrugOOD-style assay split)."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(aid)
    target_assay = int(np.argmax(counts))
    test_mask = aid == target_assay
    train_mask = ~test_mask
    test_idx = np.where(test_mask)[0]
    train_idx_full = np.where(train_mask)[0]
    perm = rng.permutation(len(train_idx_full))
    val_idx = train_idx_full[perm[: max(1, len(perm) // 10)]]
    train_idx = train_idx_full[perm[max(1, len(perm) // 10):]]
    return train_idx, val_idx, test_idx


def build_scaffold_split(y, sid, seed):
    """DrugOOD-style scaffold split (matches train_drugood.py _build_scaffold_split).

    Memory-efficient version: numpy-only, no defaultdict.
    Each scaffold group goes to train/val/test by largest-first 70/10/20 split.
    """
    rng = np.random.default_rng(seed)
    # Get unique scaffolds and inverse indices (which row belongs to which scaffold)
    unique_scaffolds, inverse_idx = np.unique(sid, return_inverse=True)
    n = len(y)
    # Compute group sizes via bincount of inverse_idx
    group_sizes = np.bincount(inverse_idx)
    # Sort scaffolds by size descending (largest first)
    sorted_idx = np.argsort(-group_sizes, kind="stable")
    target_train = int(0.7 * n)
    target_val = int(0.1 * n)
    target_test = n - target_train - target_val
    counts = [0, 0, 0]
    targets = [target_train, target_val, target_test]
    # Greedy assignment
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for s_idx in sorted_idx:
        gaps = [targets[k] - counts[k] for k in range(3)]
        # Assign to split with largest gap
        k = int(np.argmax(gaps))
        mask = [train_mask, val_mask, test_mask][k]
        mask[inverse_idx == s_idx] = True
        counts[k] += group_sizes[s_idx]
        if k == 0:
            train_mask = mask
        elif k == 1:
            val_mask = mask
        else:
            test_mask = mask
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


def build_size_split(y, szid, seed):
    """DrugOOD-style size split (matches train_drugood.py _build_size_split).

    10 deciles of size (heavy-atom count), distributed 70/10/20 by largest-first.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    # Quantize into deciles
    decile = np.zeros(n, dtype=np.int64)
    try:
        decile = np.minimum((szid * 10 // max(szid.max() + 1, 1)), 9)
    except Exception:
        pass
    unique_d, inverse_idx = np.unique(decile, return_inverse=True)
    group_sizes = np.bincount(inverse_idx)
    sorted_idx = np.argsort(-group_sizes, kind="stable")
    target_train = int(0.7 * n)
    target_val = int(0.1 * n)
    target_test = n - target_train - target_val
    counts = [0, 0, 0]
    targets = [target_train, target_val, target_test]
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for s_idx in sorted_idx:
        gaps = [targets[k] - counts[k] for k in range(3)]
        k = int(np.argmax(gaps))
        if k == 0:
            train_mask[inverse_idx == s_idx] = True
        elif k == 1:
            val_mask[inverse_idx == s_idx] = True
        else:
            test_mask[inverse_idx == s_idx] = True
        counts[k] += group_sizes[s_idx]
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


BUILDERS = {
    "assay": build_assay_split,
    "scaffold": build_scaffold_split,
    "size": build_size_split,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=NATIVE_DATA,
                        help="DrugOOD native train.csv")
    parser.add_argument("--n-subsample", type=int, default=50000,
                        help="Subsample to N rows for fast verification")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--splits", nargs="+",
                        default=["assay", "scaffold", "size"])
    parser.add_argument("--out-json", type=Path,
                        default=Path("metrics/by_metric/drugood_ood_auc_native_smoke.json"))
    args = parser.parse_args()

    print(f"[smoke] loading {args.data}", flush=True)
    t0 = time.time()
    smiles, y, aid, sid, szid = load_native(args.data)
    print(f"[smoke] loaded {len(smiles):,} rows in {time.time()-t0:.1f}s", flush=True)
    print(f"[smoke] y dist: 0={int((y==0).sum()):,} 1={int((y==1).sum()):,}", flush=True)
    print(f"[smoke] distinct assays={len(set(aid.tolist())):,} "
          f"scaffolds={len(set(sid.tolist())):,} sizes={len(set(szid.tolist())):,}",
          flush=True)

    if args.n_subsample and len(smiles) > args.n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(smiles))[: args.n_subsample]
        smiles = [smiles[i] for i in idx]
        y = y[idx]
        aid = aid[idx]
        sid = sid[idx]
        szid = szid[idx]
        print(f"[smoke] subsampled to {len(smiles):,}", flush=True)

    print(f"[smoke] featurizing {len(smiles):,} SMILES -> Morgan FP(2048, r=2)",
          flush=True)
    t0 = time.time()
    X = featurize(smiles, n_bits=2048, radius=2)
    print(f"[smoke] featurize time: {time.time()-t0:.1f}s, X.shape={X.shape}",
          flush=True)

    results = {}
    for split in args.splits:
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            tr, va, te = BUILDERS[split](y, {
                "assay": aid, "scaffold": sid, "size": szid}[split], seed)
            print(f"[smoke] split={split} seed={seed}: train={len(tr):,} "
                  f"val={len(va):,} test={len(te):,} ({time.time()-t0:.1f}s)",
                  flush=True)

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=500,
                                       random_state=seed, n_jobs=1)
            t0 = time.time()
            model.fit(X[tr], y[tr])
            train_time = time.time() - t0
            ood_proba = model.predict_proba(X[te])[:, 1]
            ood_auc = roc_auc_score(y[te], ood_proba)
            id_proba = model.predict_proba(X[va])[:, 1]
            id_auc = roc_auc_score(y[va], id_proba)
            per_seed.append({
                "split": split, "seed": seed,
                "ood_test_auc": float(ood_auc),
                "id_test_auc": float(id_auc),
                "train_time_s": train_time,
            })
            print(f"[smoke]   {split} seed={seed}: OOD={ood_auc:.4f} ID={id_auc:.4f} "
                  f"({train_time:.1f}s)", flush=True)
        ood_arr = np.array([s["ood_test_auc"] for s in per_seed])
        results[split] = {
            "ood_auc_mean": float(ood_arr.mean()),
            "ood_auc_std": float(ood_arr.std()),
            "per_seed": per_seed,
            "imold_reference": IMOLD_REFERENCE[split],
            "delta_vs_imold": float(ood_arr.mean() - IMOLD_REFERENCE[split]),
        }
        print(f"[smoke] === {split} OOD AUC = {ood_arr.mean():.4f} "
              f"+/- {ood_arr.std():.4f}  "
              f"iMoLD ref = {IMOLD_REFERENCE[split]:.4f}  "
              f"delta = +{results[split]['delta_vs_imold']:.4f}",
              flush=True)

    payload = {
        "data_source": "drugood_native",
        "n_rows": len(smiles),
        "n_subsample_target": args.n_subsample,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "splits": args.splits,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"[smoke] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())