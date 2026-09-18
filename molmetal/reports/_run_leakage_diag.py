"""End-to-end leakage diagnosis runner for L1.

Produces a JSON file with all the numbers needed for the diagnosis markdown
report (and prints a human-readable summary).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from molmetal.baselines.eval_utils import (
    compute_metrics,
    morgan_features,
)
from molmetal.baselines.morgan_xgb import MorganXGBBaseline, XGB_PARAMS
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.leakage_utils import (
    bootstrap_auc,
    canonicalize_smiles,
    smiles_overlap,
)
from molmetal.data.splits import RandomSplitter


REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
OUT_PATH = REPORTS_DIR / "leakage_diagnosis_data.json"


def _bucket(dup_count: int) -> str:
    if dup_count == 1:
        return "1"
    if dup_count <= 5:
        return "2-5"
    if dup_count <= 20:
        return "6-20"
    if dup_count <= 100:
        return "21-100"
    return "100+"


def diagnose_metal(metal: str, seed: int = 42) -> dict:
    """Run all leakage diagnostics for one metal centre."""
    print(f"\n{'='*70}\n[{metal}] loading dataset")
    ds = MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(metal_whitelist=[metal])
    )
    n_raw = len(ds)
    smiles_raw = ds.smiles
    print(f"[{metal}] n_raw_rows = {n_raw}")

    # ---- 1) canonicalise SMILES + dedup stats ----
    canon = np.array([canonicalize_smiles(s) for s in smiles_raw], dtype=object)

    # parse failure count: a parse failure means RDKit returned None
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    n_fail = 0
    for raw in smiles_raw:
        try:
            if Chem.MolFromSmiles(str(raw)) is None:
                n_fail += 1
        except Exception:
            n_fail += 1

    n_unique = int(len(set(canon.tolist())))
    counts = Counter(canon.tolist())
    dup_counter = Counter(counts.values())  # number-of-rows → freq
    bucket_counts = Counter()
    for k, v in dup_counter.items():
        bucket_counts[_bucket(k)] += v  # how many unique SMILES have this dup-count

    top5 = counts.most_common(5)

    print(f"[{metal}] n_parse_failures = {n_fail}")
    print(f"[{metal}] n_unique_canonical = {n_unique}")
    print(f"[{metal}] duplicate-count buckets (unique-SMILES count): "
          f"{dict(sorted(bucket_counts.items()))}")
    print(f"[{metal}] top 5 most-duplicated SMILES:")
    for smi, c in top5:
        print(f"    count={c:4d}  smiles={smi[:80]}")

    # ---- 2) split overlap via RandomSplitter ----
    splitter = RandomSplitter(seed=seed)
    split = splitter(ds)
    n_train, n_val, n_test = len(split.train_idx), len(split.val_idx), len(split.test_idx)
    print(f"[{metal}] split sizes: train={n_train} val={n_val} test={n_test}")

    tv = smiles_overlap(split.train_idx, split.val_idx, smiles_raw.tolist())
    tt = smiles_overlap(split.train_idx, split.test_idx, smiles_raw.tolist())
    vt = smiles_overlap(split.val_idx, split.test_idx, smiles_raw.tolist())

    # ---- 3) re-train XGBoost for bootstrap CI on test predictions ----
    print(f"[{metal}] training XGBoost for bootstrap...")
    t0 = time.time()
    y_all = ds.active.astype(int)
    keep = np.array([bool(s) for s in smiles_raw], dtype=bool)
    smiles_k = smiles_raw[keep]
    y_k = y_all[keep]
    cell_lines = ds.df["Cell_line"].astype(str).to_numpy()[keep]

    # use the same RandomSplitter-based split, but consistent with baseline.py
    # baseline.py uses sklearn train_test_split stratify with seed=42; here we
    # do the same for fair comparison with the reported numbers.
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y_k))
    idx_train, idx_tmp, _, y_tmp = train_test_split(
        idx, y_k, test_size=0.2, stratify=y_k, random_state=seed
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )

    x = morgan_features(smiles_k, radius=2, n_bits=2048)
    x_train, y_train = x[idx_train], y_k[idx_train]
    x_val, y_val = x[idx_val], y_k[idx_val]
    x_test, y_test = x[idx_test], y_k[idx_test]

    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(neg) / float(pos)
    params["random_state"] = seed

    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    test_score = model.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_score)
    elapsed = time.time() - t0
    print(f"[{metal}] XGB test_metrics={test_metrics} elapsed={elapsed:.1f}s")

    # ---- 4) bootstrap CI ----
    print(f"[{metal}] bootstrapping (1000 iters)...")
    mean, lo, hi, aucs = bootstrap_auc(y_test, test_score, n_bootstrap=1000, seed=seed)
    print(f"[{metal}] bootstrap AUC mean={mean:.4f} 95% CI=[{lo:.4f}, {hi:.4f}]")

    # ---- 5) seen vs unseen bucket AUC ----
    train_canon_set = {canonicalize_smiles(str(smiles_k[i])) for i in idx_train}
    test_canon = np.array(
        [canonicalize_smiles(str(smiles_k[i])) for i in idx_test], dtype=object
    )
    seen_mask = np.array([c in train_canon_set for c in test_canon], dtype=bool)

    seen_metrics = compute_metrics(y_test[seen_mask], test_score[seen_mask])
    unseen_metrics = compute_metrics(y_test[~seen_mask], test_score[~seen_mask])
    print(f"[{metal}] seen-unique AUC={seen_metrics['roc_auc']:.4f}  n={seen_metrics['n']}")
    print(f"[{metal}] unseen     AUC={unseen_metrics['roc_auc']:.4f}  n={unseen_metrics['n']}")

    return {
        "metal": metal,
        "n_raw_rows": int(n_raw),
        "n_parse_failures": int(n_fail),
        "n_unique_canon": n_unique,
        "duplicate_bucket_unique_smis": dict(sorted(bucket_counts.items())),
        "top5_most_duplicated": [
            {"smiles": s[:200], "count": int(c)} for s, c in top5
        ],
        "split_sizes": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "overlap_train_val": tv,
        "overlap_train_test": tt,
        "overlap_val_test": vt,
        "xgb_test_metrics": {k: float(v) if not isinstance(v, int) else int(v)
                             for k, v in test_metrics.items()},
        "xgb_train_size": int(len(idx_train)),
        "xgb_val_size": int(len(idx_val)),
        "xgb_test_size": int(len(idx_test)),
        "xgb_elapsed_seconds": elapsed,
        "bootstrap": {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n_bootstrap": 1000,
            "seed": seed,
            "n_test_used": int(y_test.size),
        },
        "seen_unseen": {
            "seen": seen_metrics,
            "unseen": unseen_metrics,
            "n_seen_rows": int(seen_mask.sum()),
            "n_unseen_rows": int((~seen_mask).sum()),
        },
    }


def main() -> None:
    out: dict = {}
    for metal in ("Ru", "Ir"):
        out[metal] = diagnose_metal(metal)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[ok] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
