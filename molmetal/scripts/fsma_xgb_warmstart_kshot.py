#!/usr/bin/env python3
"""P1.5 / C5 — does FSMA transfer to the fingerprint + GBDT scorer?

Why this exists
---------------
The FSMA K-shot result (C1) is measured on the chemprop v2 graph backbone.
But the paper's *within-metal* column belongs to Morgan-fingerprint + XGBoost
(MEASURED: 0.8271 Ru / 0.8430 Ir on the 24h cohort, and higher on the paper
funnel).  If adaptation only works on a graph net, the method is narrower than
claimed.  This script runs the same LOMO K-shot protocol with XGBoost, where
"adaptation" is **warm-started gradient boosting**: fit the source-metal model,
then continue boosting on the K support ligands via ``xgb_model=booster``.

Arms (all scored on the IDENTICAL query rows -> paired):
  * ``zero_shot``       — 500 rounds on the source pool, no adaptation;
  * ``warm_start_k``    — +50 rounds continued on the K support ligands;
  * ``from_scratch_k``  — 550 rounds on the K support ligands only;
  * ``concat_retrain_k``— 550 rounds on (source pool ++ K support).

``from_scratch_k`` and ``concat_retrain_k`` are the C2 controls: they hold the
tree budget fixed and separate "the adaptation" from "the extra K labels".

Leakage guard: the query rows never enter any fit.  Support rows are drawn from
the held-out metal's test set with the same label-stratified sampler the
chemprop runner uses.

MEASURED only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS = (42, 43, 44, 49, 123)

XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(np.asarray(y).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _draw_support(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Label-stratified K-shot support indices (mirrors the chemprop runner)."""
    rng = np.random.default_rng(int(seed))
    n = len(y)
    sup: List[int] = []
    for lab in (0, 1):
        li = np.where(y == lab)[0].copy()
        rng.shuffle(li)
        k_lab = max(1, int(round(k * len(li) / n)))
        sup.extend(li[:k_lab].tolist())
    return np.asarray(sorted(set(sup)), dtype=int)


def _fit(X, y, seed, rounds, base=None):
    from xgboost import XGBClassifier

    params = dict(XGB_PARAMS, random_state=int(seed), n_estimators=int(rounds))
    pos = max(1, int(np.asarray(y).sum()))
    neg = max(1, int(len(y) - pos))
    params["scale_pos_weight"] = neg / pos
    clf = XGBClassifier(**params)
    if base is None:
        clf.fit(X, y, verbose=False)
    else:
        clf.fit(X, y, xgb_model=base.get_booster(), verbose=False)
    return clf


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--base-rounds", type=int, default=500)
    p.add_argument("--ft-rounds", type=int, default=50)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--cohort-spec", choices=("24h", "paper_funnel"),
                   default="paper_funnel")
    p.add_argument("--out-metric",
                   default="metrics/by_metric/fsma_xgb_warmstart_kshot.json")
    args = p.parse_args(argv)

    from molmetal.scripts.metal_activity_transfer_probe import _featurise
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import build_probe_cohort

    per_metal, dup = build_probe_cohort(
        list(args.metals), 24.0, 0.01, cohort_spec=args.cohort_spec
    )
    feats = {
        m: _featurise(list(per_metal[m][0]), args.radius, args.nbits)
        for m in args.metals
    }
    report: Dict[str, object] = {
        "metric": "fsma_xgb_warmstart_kshot",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fsma_xgb_warmstart_kshot.py",
        "description": (
            "C5 cross-backbone check: the FSMA LOMO K-shot protocol run with "
            "Morgan-fingerprint + XGBoost, where adaptation = warm-started "
            "gradient boosting (xgb_model continuation) on the K support "
            "ligands.  Budget-matched controls from_scratch_k / "
            "concat_retrain_k separate adaptation from sample size."
        ),
        "protocol": {
            "cohort_spec": args.cohort_spec,
            "scorer": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
            "base_rounds": int(args.base_rounds),
            "ft_rounds": int(args.ft_rounds),
            "control_rounds": int(args.base_rounds + args.ft_rounds),
            "split": "leave-one-metal-out, ligand-disjoint (leaked ligands dropped)",
            "support_sampler": "label-stratified random",
            "seeds": [int(s) for s in args.seeds],
            "cohort_n_unique": {m: int(dup[m]["n_unique"]) for m in args.metals},
        },
        "per_metal": {},
    }

    # LOMO leakage rule: drop source ligands that appear in the held-out metal.
    smiles_sets = {m: set(per_metal[m][0]) for m in args.metals}
    for held in args.metals:
        src = [m for m in args.metals if m != held]
        leak = smiles_sets[held]
        trX, trY = [], []
        for m in src:
            keep = np.array([s not in leak for s in per_metal[m][0]])
            trX.append(feats[m][keep])
            trY.append(per_metal[m][1][keep])
        Xtr = np.vstack(trX) if trX else np.zeros((0, args.nbits), dtype=np.float32)
        ytr = np.concatenate(trY) if trY else np.zeros(0, dtype=int)
        Xte = feats[held]
        yte = per_metal[held][1]

        metal_block: Dict[str, object] = {
            "n_train_source": int(len(ytr)), "n_test": int(len(yte)),
            "pos_rate_test": float(yte.mean()),
            "by_k": {},
        }
        for k in args.ks:
            if len(yte) <= k + 10 or len(ytr) < 50:
                metal_block["by_k"][str(k)] = {"status": "SKIPPED",
                                               "reason": f"n_test={len(yte)} <= K+10"}
                continue
            cells: Dict[str, List[float]] = {
                "zero_shot": [], "warm_start_k": [], "from_scratch_k": [],
                "concat_retrain_k": [],
            }
            for seed in args.seeds:
                sup = _draw_support(yte, k, seed)
                q = np.setdiff1d(np.arange(len(yte)), sup)
                base = _fit(Xtr, ytr, seed, args.base_rounds)
                p0 = base.predict_proba(Xte)[:, 1]
                cells["zero_shot"].append(_auc(yte[q], p0[q]))
                warm = _fit(Xte[sup], yte[sup], seed, args.ft_rounds, base=base)
                pw = warm.predict_proba(Xte)[:, 1]
                cells["warm_start_k"].append(_auc(yte[q], pw[q]))
                fs = _fit(Xte[sup], yte[sup], seed, args.base_rounds + args.ft_rounds)
                pf = fs.predict_proba(Xte)[:, 1]
                cells["from_scratch_k"].append(_auc(yte[q], pf[q]))
                Xc = np.vstack([Xtr, Xte[sup]])
                yc = np.concatenate([ytr, yte[sup]])
                cr = _fit(Xc, yc, seed, args.base_rounds + args.ft_rounds)
                pc = cr.predict_proba(Xte)[:, 1]
                cells["concat_retrain_k"].append(_auc(yte[q], pc[q]))
            metal_block["by_k"][str(k)] = {
                arm: {
                    "mean": float(np.nanmean(v)),
                    "per_seed": [float(x) for x in v],
                    "n_seeds_ok": int(np.sum(~np.isnan(v))),
                }
                for arm, v in cells.items()
            }
            metal_block["by_k"][str(k)]["paired_delta_warm_vs_zero"] = {
                "mean": float(np.nanmean(np.asarray(cells["warm_start_k"])
                                         - np.asarray(cells["zero_shot"]))),
                "per_seed": [float(a - b) for a, b in zip(cells["warm_start_k"],
                                                          cells["zero_shot"])],
            }
            metal_block["by_k"][str(k)]["paired_delta_warm_vs_concat"] = {
                "mean": float(np.nanmean(np.asarray(cells["warm_start_k"])
                                         - np.asarray(cells["concat_retrain_k"]))),
                "per_seed": [float(a - b) for a, b in zip(cells["warm_start_k"],
                                                          cells["concat_retrain_k"])],
            }
        report["per_metal"][held] = metal_block

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out, json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    print(f"{'metal':5} {'K':>4} {'zero':>7} {'warm':>7} {'scratch':>8} "
          f"{'concat':>7} {'w-z':>7} {'w-c':>7}")
    for m in args.metals:
        for k in args.ks:
            b = report["per_metal"][m]["by_k"].get(str(k), {})
            if b.get("status") == "SKIPPED":
                continue
            print(f"{m:5} {k:4d} {b['zero_shot']['mean']:7.4f} "
                  f"{b['warm_start_k']['mean']:7.4f} "
                  f"{b['from_scratch_k']['mean']:8.4f} "
                  f"{b['concat_retrain_k']['mean']:7.4f} "
                  f"{b['paired_delta_warm_vs_zero']['mean']:+7.4f} "
                  f"{b['paired_delta_warm_vs_concat']['mean']:+7.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
