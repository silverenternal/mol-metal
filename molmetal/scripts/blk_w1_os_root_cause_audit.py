#!/usr/bin/env python3
"""W1 — Os root-cause audit: why is the Os within-metal AUC below chance?

Motivation (todo.json :: submission_plan_2026_10_13.weak_metrics_workstream W1)
------------------------------------------------------------------------------
Os within-metal ROC-AUC is 0.4960 (fingerprint+XGB) and 0.4187 (chemprop v2),
i.e. **below chance under both scorers**, and its K-shot cell at K=40 is empty
because Os has only 36 unique ligands (K=40 cannot draw a support set).

A model that cannot beat chance on a 36-ligand cohort is more likely to be a
**cohort/label pathology** than a capability gap, so this script audits the
cohort before any modelling lever is spent.  It measures, per metal:

1. cohort composition (raw rows, unique ligands, duplication, pos rate);
2. **label noise**: duplicate groups whose rows disagree on the label;
3. **cell-line concentration**: how much of the cohort one cell line owns,
   and how often the SAME ligand is measured active in one line and inactive
   in another (which makes the dedup majority vote meaningless);
4. **cross-metal ligand sharing**: the fraction of unique ligands that also
   appear under another metal (the LOMO leakage rule drops these);
5. within-metal AUC under three label rules on the SAME folds:
   majority-active (current), majority-inactive (conservative), and a
   **single-cell-line** subset (the sharpest test of whether the mixed-line
   majority vote is what destroys the signal).

MEASURED only.  No projection, no extrapolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METALS = ("Ru", "Ir", "Rh", "Os", "Re")

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}


def _load_env_default() -> None:
    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


def _collapse(smiles, y, tie_rule: str) -> Tuple[List[str], np.ndarray]:
    import pandas as pd

    df = pd.DataFrame({"s": [str(x) for x in smiles], "y": np.asarray(y, dtype=int)})
    if tie_rule == "active":
        agg = df.groupby("s")["y"].agg(lambda v: int(v.sum() * 2 >= len(v)))
    else:
        agg = df.groupby("s")["y"].agg(lambda v: int(v.sum() * 2 > len(v)))
    return list(agg.index), agg.to_numpy(dtype=int)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _cv_auc(
    smiles: List[str],
    y: np.ndarray,
    seeds: Tuple[int, ...],
    n_splits: int = 5,
    radius: int = 2,
    nbits: int = 2048,
) -> Tuple[float, List[float]]:
    """5-fold stratified CV AUC, mean over ``seeds``; folds rebuilt per seed."""
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    from molmetal.baselines.eval_utils import morgan_features

    X = morgan_features(np.asarray(smiles), radius, nbits)
    per_seed = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(np.zeros(len(y)), y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            params = dict(XGB_PARAMS, random_state=seed)
            pos = max(1, int(y[tr].sum()))
            neg = max(1, int(len(tr) - pos))
            params["scale_pos_weight"] = neg / pos
            clf = XGBClassifier(**params)
            clf.fit(X[tr], y[tr], verbose=False)
            aucs.append(_auc(y[te], clf.predict_proba(X[te])[:, 1]))
        if aucs:
            per_seed.append(float(np.mean(aucs)))
    return (float(np.mean(per_seed)) if per_seed else float("nan")), per_seed


def _cohort_stats(df, metal: str) -> Dict[str, object]:
    """Composition + label-noise + cell-line concentration for one metal."""
    sub = df[df["Metal"].astype(str) == metal]
    smi = [str(x) for x in sub["SMILES_Ligands"]]
    ic50 = sub["IC50_Dark_value"].to_numpy(dtype=float)
    # loader definition: active := IC50_Dark_value < 10 uM (see _add_active)
    active = (ic50 < 10.0).astype(int)
    lines = sub["Cell_line"].astype(str).to_numpy()

    # duplicate-group label disagreement
    groups: Dict[str, List[int]] = {}
    for s, a in zip(smi, active):
        groups.setdefault(s, []).append(int(a))
    mixed = sum(1 for v in groups.values() if 0 < sum(v) < len(v))
    ties = sum(1 for v in groups.values() if len(v) > 1 and sum(v) * 2 == len(v))

    # same ligand measured in >1 cell line, disagreeing
    per_ligand_lines: Dict[str, Dict[str, int]] = {}
    for s, a, c in zip(smi, active, lines):
        per_ligand_lines.setdefault(s, {}).setdefault(c, int(a))
    multi_line = {s: v for s, v in per_ligand_lines.items() if len(v) > 1}
    multi_line_disagree = sum(1 for v in multi_line.values() if len(set(v.values())) > 1)

    # cell-line concentration
    uniq, counts = np.unique(lines, return_counts=True)
    order = np.argsort(-counts)
    top_lines = [
        {"cell_line": str(uniq[i]), "n_rows": int(counts[i]),
         "share": float(counts[i] / len(lines))}
        for i in order[:5]
    ]

    return {
        "n_rows_raw": int(len(sub)),
        "n_unique": int(len(groups)),
        "duplication_rate": float(1.0 - len(groups) / max(1, len(sub))),
        "pos_rate_raw": float(active.mean()),
        "n_duplicate_groups_mixed_label": int(mixed),
        "n_duplicate_groups_exact_tie": int(ties),
        "n_ligands_multi_cellline": int(len(multi_line)),
        "n_ligands_multi_cellline_disagreeing": int(multi_line_disagree),
        "n_distinct_cell_lines": int(len(uniq)),
        "top_cell_lines": top_lines,
    }


def _cross_metal_sharing(df) -> Dict[str, object]:
    per_metal = {
        m: set(str(x) for x in df[df["Metal"].astype(str) == m]["SMILES_Ligands"])
        for m in METALS
    }
    out = {}
    for m in METALS:
        others = set().union(*[per_metal[o] for o in METALS if o != m])
        shared = per_metal[m] & others
        out[m] = {
            "n_unique": int(len(per_metal[m])),
            "n_shared_with_other_metals": int(len(shared)),
            "shared_fraction": float(len(shared) / max(1, len(per_metal[m]))),
        }
    return out


def _within_metal_sensitivity(df, metal: str, seeds: Tuple[int, ...]) -> Dict[str, object]:
    """AUC under three label rules; plus a single-cell-line subset."""
    sub = df[df["Metal"].astype(str) == metal]
    smi = [str(x) for x in sub["SMILES_Ligands"]]
    ic50 = sub["IC50_Dark_value"].to_numpy(dtype=float)
    # loader definition: active := IC50_Dark_value < 10 uM (see _add_active)
    active = (ic50 < 10.0).astype(int)
    lines = sub["Cell_line"].astype(str).to_numpy()

    out: Dict[str, object] = {}

    for rule in ("active", "inactive"):
        s_u, y_u = _collapse(smi, active, rule)
        mean, per_seed = _cv_auc(s_u, y_u, seeds)
        out[f"dedup_majority_{rule}"] = {
            "n_unique": int(len(s_u)),
            "pos_rate": float(y_u.mean()),
            "cv_auc_mean": mean,
            "cv_auc_per_seed": per_seed,
        }

    # single-cell-line subsets: keep a line only if it has enough unique ligands
    uniq, counts = np.unique(lines, return_counts=True)
    subsets = {}
    for line, cnt in zip(uniq, counts):
        mask = lines == line
        s_sub = [s for s, k in zip(smi, mask) if k]
        y_sub = active[mask]
        s_u, y_u = _collapse(s_sub, y_sub, "active")
        if len(s_u) < 20 or len(np.unique(y_u)) < 2:
            continue
        mean, per_seed = _cv_auc(s_u, y_u, seeds)
        subsets[str(line)] = {
            "n_rows_raw": int(mask.sum()),
            "n_unique": int(len(s_u)),
            "pos_rate": float(y_u.mean()),
            "cv_auc_mean": mean,
            "cv_auc_per_seed": per_seed,
        }
    out["single_cell_line_subsets"] = subsets
    if subsets:
        best = max(subsets.items(), key=lambda kv: (kv[1]["cv_auc_mean"]
                                                     if kv[1]["cv_auc_mean"] == kv[1]["cv_auc_mean"] else -1))
        out["best_single_cell_line"] = {"cell_line": best[0], **best[1]}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metals", nargs="+", default=["Os", "Ru", "Ir", "Rh", "Re"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 49, 123])
    p.add_argument("--out-metric", default="metrics/by_metric/w1_os_root_cause_audit.json")
    args = p.parse_args(argv)

    _load_env_default()
    import pandas as pd

    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    frames = []
    for m in args.metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        d = ds.df.copy()
        d["Metal"] = m
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    seeds = tuple(args.seeds)
    report: Dict[str, object] = {
        "metric": "w1_os_root_cause_audit",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/blk_w1_os_root_cause_audit.py",
        "description": (
            "W1 root-cause audit for the below-chance Os within-metal AUC. "
            "Measures cohort composition, duplicate-group label noise, "
            "cell-line concentration and same-ligand cross-cell-line label "
            "disagreement, cross-metal ligand sharing, and within-metal AUC "
            "under three label rules (majority-active / majority-inactive / "
            "single-cell-line subset)."
        ),
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM)",
            "seeds": list(seeds),
            "n_splits": 5,
            "scorer": "morgan_fp(radius=2, 2048bit) + xgboost",
            "note": "K-shot K=40 for Os is empty because Os has <40 unique ligands.",
        },
        "cohort_composition": {m: _cohort_stats(df, m) for m in args.metals},
        "cross_metal_sharing": _cross_metal_sharing(df),
        "within_metal_sensitivity": {
            m: _within_metal_sensitivity(df, m, seeds) for m in args.metals
        },
    }

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out, json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    print(f"{'metal':5} {'raw':>6} {'uniq':>6} {'pos%':>6} {'mixed':>6} {'ties':>5} "
          f"{'multiLn':>8} {'disagr':>7} {'shared%':>8} {'AUC(act)':>9} {'AUC(inact)':>10}")
    for m in args.metals:
        c = report["cohort_composition"][m]
        sh = report["cross_metal_sharing"][m]
        s = report["within_metal_sensitivity"][m]
        print(
            f"{m:5} {c['n_rows_raw']:6d} {c['n_unique']:6d} "
            f"{100*c['pos_rate_raw']:6.1f} {c['n_duplicate_groups_mixed_label']:6d} "
            f"{c['n_duplicate_groups_exact_tie']:5d} "
            f"{c['n_ligands_multi_cellline']:8d} "
            f"{c['n_ligands_multi_cellline_disagreeing']:7d} "
            f"{100*sh['shared_fraction']:8.1f} "
            f"{s['dedup_majority_active']['cv_auc_mean']:9.4f} "
            f"{s['dedup_majority_inactive']['cv_auc_mean']:10.4f}"
        )
    print()
    for m in args.metals:
        sub = report["within_metal_sensitivity"][m]["single_cell_line_subsets"]
        if not sub:
            continue
        best = report["within_metal_sensitivity"][m].get("best_single_cell_line", {})
        print(f"{m}: {len(sub)} single-cell-line subsets with n>=20; "
              f"best = {best.get('cell_line')} n={best.get('n_unique')} "
              f"AUC={best.get('cv_auc_mean', float('nan')):.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
