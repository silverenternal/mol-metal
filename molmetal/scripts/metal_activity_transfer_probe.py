#!/usr/bin/env python3
"""Within-metal vs cross-metal anticancer activity — the transfer-penalty probe.

=====================================================================
What this measures, and why both halves are needed
=====================================================================
The field's headline numbers (Krasnov et al., J. Med. Chem. 2026, 69(8),
8838; MetalCytoToxDB) are **within-metal** ROC-AUC: 0.81 for Ru, 0.73
for Ir. That is a same-metal train/test split — the model has already
seen that metal's coordination chemistry during training.

Cross-metal (leave-one-metal-out) is a strictly harder question and
**nobody in the comparison set has published it**. Reporting only one
half produces a wrong conclusion in either direction:

* quoting within-metal AUC as if it were cross-metal overstates us;
* quoting cross-metal AUC as if it were the field's number understates
  the field.

So this script measures BOTH, on the SAME deduplicated cohort, with the
SAME model, and reports the difference as the transfer penalty. The
penalty is the finding; neither half is meaningful alone.

Why deduplication is load-bearing (MEASURED 2026-10-10)
--------------------------------------------------------
Raw MetalCytoToxDB rows: 5,271 across 5 metals, but only **1,908 unique
ligands** — 64% duplication, one ligand measured up to 15 times. Under
a Morgan fingerprint + GBDT scorer, a ligand present in both the train
metals and the held-out metal is simply memorised, so cross-metal
apparent transfer inflates. Measured effect of removing duplicates:

    pooled ROC-AUC   0.6444 (raw)  ->  0.5894 (deduplicated)
    top-5% lift      1.88         ->  1.54
    Os fold AUC      0.6912       ->  0.5318   (47% of Os ligands also
                                                appear in other metals)
    Re fold AUC      0.5701       ->  0.4179   (below chance)

Duplicates are collapsed by majority label. The predicate is `sum * 2 >= n`,
so an exact tie resolves to ACTIVE -- the ANTI-conservative direction for an
activity claim. An earlier version of this docstring said "a tie resolves to
inactive, which is the conservative choice"; that was wrong and is corrected
here 2026-10-11. The effect is real, not cosmetic: 80 ligand groups tie on
MetalCytoToxDB and all 80 are labelled active, and re-running within-metal
under the conservative rule moves Ru 0.8363 -> 0.7929, below Krasnov's
published 0.81, while Ir is unchanged. See
`metrics/by_metric/metal_activity_transfer_tie_sensitivity.json` and the
`collapse_duplicates` docstring below.

Honest-negative carried by this script
--------------------------------------
``Re`` cross-metal lands **below chance** (0.418 after dedup, from 0.928
within-metal). ``Rh`` (n=51 unique) and ``Os`` (n=36 unique) are too small
to resolve at all and are reported but not gated. Any paper claim must
carry the per-metal table, not the pooled mean, because the pooled mean
is dominated by whichever metal is largest (Ru is 64% of the cohort).

Model
-----
Morgan/ECFP4 (radius 2, 2048 bit) + XGBoost, with ``baseline_lomo_cv``'s
exact hyper-parameters so the two scripts' numbers are comparable. NOT
our D-MPNN, deliberately: this probe asks whether the *signal* exists in
the data before any GPU time is spent. Note ``baseline_lomo_cv``'s
per-fold ``hit_rate_top5pct`` is NOT quoted anywhere — its folds are
87-3668 rows, so top-5% is 4-183 molecules and the number is dominated
by k, not by chemistry.

Usage
-----
::

    .venv/bin/python molmetal/scripts/metal_activity_transfer_probe.py \
        --out-metric metrics/by_metric/metal_activity_transfer.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

METALS = ["Ru", "Ir", "Rh", "Os", "Re"]

# Krasnov et al. 2026 published WITHIN-Metal ROC-AUC. Recorded so the
# within-metal half of this script is checked against the literature
# rather than merely asserted. `None` = not reported for that metal.
KRASNOV_WITHIN_METAL_AUC = {"Ru": 0.81, "Ir": 0.73, "Rh": None, "Os": None, "Re": None}

# Mirrors baseline_lomo_cv's XGBClassifier params exactly.
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

# Below this many unique ligands a metal's AUC is not resolvable and must
# not be gated. At n=36 a 95% CI on AUC is roughly +-0.16.
MIN_UNIQUE_FOR_GATE = 200


def _load_env_default() -> None:
    """``cytotox.DEFAULT_CSV`` honours MOLMETAL_DATA_DIR; mirror env.sh 1b.

    Without this the loader falls back to a /mnt/storage mount that does
    not exist on this host — measured FileNotFoundError 2026-10-10.
    """
    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


def collapse_duplicates(
    smiles: np.ndarray, y: np.ndarray
) -> Tuple[List[str], np.ndarray, int]:
    """One row per unique ligand; label = majority, **ties resolve to ACTIVE**.

    Returns ``(unique_smiles, unique_labels, n_raw_rows)``.

    ⚠️ The tie rule is ``sum * 2 >= n``, so an exact tie (sum*2 == n) yields
    ``True`` → label 1. That is the **anti-conservative** direction for an
    activity claim, and earlier versions of this docstring (and of the matching
    trainer) described it as "ties resolve to inactive, the conservative
    choice" — that was wrong, verified 2026-10-10 by evaluating the expression
    on n=2/sum=1 and n=4/sum=2 (both → label 1).

    It is not a harmless typo: on MetalCytoToxDB 80 ligand groups tie (Ru 59,
    Ir 19, Rh 2), and all 80 are labelled active. Re-running within-metal with
    the conservative rule ``sum * 2 > n`` moves Ru 0.8363 → 0.7929 (below
    Krasnov's published 0.81) while Ir is unchanged (0.8295 → 0.8319). See
    ``metrics/by_metric/metal_activity_transfer_tie_sensitivity.json``.

    Keep the rule, but state its direction honestly. The tie itself is genuine
    label noise (see ``todo.json :: root_cause_analysis.RC1``), so neither
    direction is "right"; what matters is that no claim is presented as robust
    to a choice it is not robust to.
    """
    import pandas as pd

    df = pd.DataFrame({"s": [str(x) for x in smiles], "y": np.asarray(y, dtype=int)})
    agg = df.groupby("s")["y"].agg(lambda v: int(v.sum() * 2 >= len(v)))
    return list(agg.index), agg.to_numpy(dtype=int), len(df)


def _make_model(y_train: np.ndarray, seed: int):
    from xgboost import XGBClassifier

    params = dict(XGB_PARAMS, random_state=seed)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    params["scale_pos_weight"] = neg / pos
    return XGBClassifier(**params)


def _featurise(smiles: List[str], radius: int, nbits: int) -> np.ndarray:
    from molmetal.baselines.eval_utils import morgan_features

    return morgan_features(np.asarray(smiles), radius, nbits)


def within_metal(
    smiles: List[str], y: np.ndarray, seed: int, radius: int, nbits: int, n_splits: int = 5
) -> Dict[str, object]:
    """Stratified k-fold CV inside a single metal — the field's own setting."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(set(y.tolist())) < 2 or len(y) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y)} too small or single-class"}

    X = _featurise(smiles, radius, nbits)
    aucs, aps = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = _make_model(y[tr], seed)
        clf.fit(X[tr], y[tr], verbose=False)
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], prob)))
        aps.append(float(average_precision_score(y[te], prob)))
    return {
        "status": "MEASURED",
        "n_unique": int(len(y)),
        "n_splits": int(n_splits),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_fold_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(aps)),
        "pos_rate": float(y.mean()),
    }


def cross_metal(
    per_metal: Dict[str, Tuple[List[str], np.ndarray]],
    seed: int,
    radius: int,
    nbits: int,
) -> Dict[str, object]:
    """Leave-one-metal-out. A ligand seen in any other metal is dropped
    from the training pool, so no fingerprint can straddle the split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    pooled_p: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []
    pooled_m: List[np.ndarray] = []
    folds: List[Dict[str, object]] = []

    for held_out in per_metal:
        te_s, te_y = per_metal[held_out]
        if len(set(te_y.tolist())) < 2:
            folds.append({"metal": held_out, "status": "SKIPPED", "reason": "single-class"})
            continue
        tr_s: List[str] = []
        tr_y: List[int] = []
        n_leaked = 0
        for m, (s_list, y_arr) in per_metal.items():
            if m == held_out:
                continue
            blocked = set(te_s)
            for s, yv in zip(s_list, y_arr.tolist()):
                if s in blocked:
                    n_leaked += 1
                    continue
                tr_s.append(s)
                tr_y.append(yv)
        if len(set(tr_y)) < 2:
            folds.append({"metal": held_out, "status": "SKIPPED", "reason": "train_pool_single_class"})
            continue
        ytr = np.asarray(tr_y, dtype=int)
        Xtr = _featurise(tr_s, radius, nbits)
        Xte = _featurise(te_s, radius, nbits)
        clf = _make_model(ytr, seed)
        clf.fit(Xtr, ytr, verbose=False)
        prob = clf.predict_proba(Xte)[:, 1]
        folds.append(
            {
                "metal": held_out,
                "status": "MEASURED",
                "roc_auc": float(roc_auc_score(te_y, prob)),
                "pr_auc": float(average_precision_score(te_y, prob)),
                "n_train_unique": int(len(tr_y)),
                "n_test_unique": int(len(te_y)),
                "n_train_rows_dropped_as_leaked": int(n_leaked),
                "pos_rate_test": float(te_y.mean()),
                "resolvable": bool(len(te_y) >= MIN_UNIQUE_FOR_GATE),
            }
        )
        pooled_p.append(prob)
        pooled_y.append(te_y)
        pooled_m.append(np.array([held_out] * len(te_y), dtype=object))

    if not pooled_p:
        return {"status": "INSUFFICIENT_FOLDS", "per_fold": folds}

    p_all = np.concatenate(pooled_p)
    y_all = np.concatenate(pooled_y)
    m_all = np.concatenate(pooled_m)
    order = np.argsort(-p_all, kind="stable")
    y_sorted = y_all[order]
    base = float(y_all.mean())
    n = len(y_all)

    enrichment = {}
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(frac * n)))
        hits = int(y_sorted[:k].sum())
        hit = hits / k
        se = (hit * (1.0 - hit) / k) ** 0.5
        enrichment[f"top_{int(frac * 100)}pct"] = {
            "k": k,
            "n_hits": hits,
            "hit_rate": hit,
            "base_rate": base,
            "lift": (hit / base) if base > 0 else float("nan"),
            "hit_rate_ci95": [max(0.0, hit - 1.96 * se), min(1.0, hit + 1.96 * se)],
        }

    return {
        "status": "MEASURED",
        "per_fold": folds,
        "pooled": {
            "n": int(n),
            "base_rate": base,
            "roc_auc": float(roc_auc_score(y_all, p_all)),
            "pr_auc": float(average_precision_score(y_all, p_all)),
            "n_tied_scores": int(n - len(np.unique(p_all))),
            "enrichment": enrichment,
            "peak_enrichment_k": max(enrichment.items(), key=lambda kv: kv[1]["lift"])[0],
        },
    }


def run(
    metals: List[str],
    seed: int,
    radius: int,
    nbits: int,
    include_raw: bool = False,
) -> Dict[str, object]:
    _load_env_default()
    import pandas as pd

    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    raw: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dedup: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dup_stats: Dict[str, object] = {}

    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        s_raw = [str(x) for x in ds.smiles]
        y_raw = np.asarray(ds.active, dtype=int)
        s_u, y_u, n_raw = collapse_duplicates(ds.smiles, ds.active)
        raw[m] = (s_raw, y_raw)
        dedup[m] = (s_u, y_u)
        dup_stats[m] = {
            "n_rows_raw": int(n_raw),
            "n_unique": int(len(s_u)),
            "duplication_rate": 1.0 - len(s_u) / max(1, n_raw),
            "pos_rate_raw": float(y_raw.mean()),
            "pos_rate_dedup": float(y_u.mean()),
        }

    report: Dict[str, object] = {
        "metric": "metal_activity_transfer",
        "description": (
            "Anticancer cytotoxicity AUC for noble-metal complexes, measured "
            "BOTH within-metal (5-fold CV, the setting Krasnov et al. 2026 "
            "report) and cross-metal (leave-one-metal-out, which the "
            "comparison set does not publish), on the SAME deduplicated "
            "cohort with the SAME scorer. The difference is the transfer "
            "penalty and is the actual finding."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/metal_activity_transfer_probe.py",
        "model": f"morgan_fp(radius={radius}, {nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "seed": seed,
        "dedup_rule": "one row per unique ligand; label = majority; ties -> ACTIVE (sum*2>=n; anti-conservative direction; see tie_sensitivity field)",
        "leakage_rule": "a ligand appearing in ANY other metal is dropped from the LOMO train pool",
        "krasnov_within_metal_auc": KRASNOV_WITHIN_METAL_AUC,
        "duplication": dup_stats,
    }

    within: Dict[str, object] = {}
    for m in metals:
        s_u, y_u = dedup[m]
        within[m] = within_metal(s_u, y_u, seed, radius, nbits)
        within[m]["krasnov_reported_auc"] = KRASNOV_WITHIN_METAL_AUC.get(m)
    report["within_metal_cv"] = within

    cross = cross_metal({m: dedup[m] for m in dedup}, seed, radius, nbits)
    report["cross_metal_lomo_dedup"] = cross

    # ---- the finding: transfer penalty per metal ----
    penalty: Dict[str, object] = {}
    for m in metals:
        w = within[m]
        folds = {f["metal"]: f for f in cross.get("per_fold", []) if f.get("status") == "MEASURED"}
        if w.get("status") != "MEASURED" or m not in folds:
            continue
        f = folds[m]
        penalty[m] = {
            "within_auc": w["roc_auc"],
            "cross_auc": f["roc_auc"],
            "transfer_penalty": float(w["roc_auc"] - f["roc_auc"]),
            "cross_below_chance": bool(f["roc_auc"] < 0.5),
            "resolvable": bool(f.get("resolvable", False)),
            "krasnov_reported_auc": KRASNOV_WITHIN_METAL_AUC.get(m),
        }
    report["transfer_penalty"] = penalty

    resolvable = [m for m, v in penalty.items() if v["resolvable"]]
    report["honest_negative"] = {
        "below_chance_cross_metal": [m for m, v in penalty.items() if v["cross_below_chance"]],
        "unresolvable_small_folds": [
            m for m, v in penalty.items() if not v["resolvable"]
        ],
        "note": (
            "Ru is 64% of the deduplicated cohort, so the pooled cross-metal "
            "mean is dominated by Ru and must not be quoted without this "
            "per-metal table."
        ),
    }

    if include_raw:
        report["cross_metal_lomo_RAW_undeduplicated"] = cross_metal(raw, seed, radius, nbits)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Within- vs cross-metal activity probe.")
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="also run LOMO on the un-deduplicated rows, to quantify the leakage effect",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_activity_transfer.json",
    )
    args = p.parse_args(argv)

    rep = run(
        metals=list(args.metals),
        seed=args.seed,
        radius=args.radius,
        nbits=args.nbits,
        include_raw=args.include_raw,
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float))

    print(f"{'metal':>5} {'n_uniq':>7} {'within':>8} {'krasnov':>8} {'cross':>8} {'penalty':>9}")
    for m in METALS:
        w = rep["within_metal_cv"].get(m, {})
        pen = rep["transfer_penalty"].get(m)
        if pen is None:
            print(f"{m:>5} {w.get('n_unique', 0):7d} "
                  f"{w.get('roc_auc', float('nan')):8.4f} "
                  f"{str(w.get('krasnov_reported_auc') or 'n/a'):>8} {'-':>8} {'-':>9}")
            continue
        print(f"{m:>5} {w['n_unique']:7d} {pen['within_auc']:8.4f} "
              f"{str(pen['krasnov_reported_auc'] or 'n/a'):>8} {pen['cross_auc']:8.4f} "
              f"{pen['transfer_penalty']:9.4f}"
              + ("  BELOW-CHANCE" if pen["cross_below_chance"] else ""))

    pool = rep["cross_metal_lomo_dedup"].get("pooled")
    if pool:
        print(f"\npooled cross-metal n={pool['n']} AUC={pool['roc_auc']:.4f} "
              f"peak enrichment at {pool['peak_enrichment_k']}")
        for k, c in pool["enrichment"].items():
            print(f"  {k:>10}  k={c['k']:5d}  hit={c['hit_rate']:.3f}  lift={c['lift']:.2f}")
    hn = rep["honest_negative"]
    print(f"\nhonest-negative: below-chance={hn['below_chance_cross_metal']} "
          f"unresolvable={hn['unresolvable_small_folds']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())