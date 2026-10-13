#!/usr/bin/env python
"""L4 — is the "Ru beats Krasnov 0.81" claim robust to the tie rule?

The within-metal Ru AUC is 0.8363 when duplicate ligand groups that tie
(``sum*2 == n``, i.e. exactly half the measurements active) are labelled
ACTIVE, and 0.7929 when they are labelled INACTIVE.  The published
comparator is 0.81, so the direction of the claim is decided by an arbitrary
label convention rather than by chemistry.

This script asks whether a *third* disposition --- DROP the contradictory
groups entirely, so no tie rule is applied at all --- lands the Ru AUC on
one side of 0.81 under every seed.  It reports, for each disposition and
each seed, both the per-fold-mean AUC (what the probe reports) and the
pooled out-of-fold AUC (a single AUC over all held-out ligands), plus a
bootstrap CI on the pooled estimate.

Zero GPU.  Reuses ``metal_activity_transfer_probe.within_metal`` and the
same MetalCytotoxDataset / CytotoxFilter cohort the headline uses.

MEASURED only; every number is written to the output JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from molmetal.scripts.metal_activity_transfer_probe import (  # noqa: E402
    KRASNOV_WITHIN_METAL_AUC,
    _load_env_default,
    within_metal,
)


def _load_metal(metal: str):
    # set MOLMETAL_DATA_DIR BEFORE importing cytotox — its DEFAULT_CSV is a
    # module-level constant bound at import time.
    _load_env_default()
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    ds = MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[metal])
    )
    return [str(x) for x in ds.smiles], np.asarray(ds.active, dtype=int)


def _collapse(smiles, y, disposition: str):
    """Return (unique_smiles, labels) under one tie disposition.

    active   : ties -> 1   (predicate sum*2 >= n; the probe default)
    inactive : ties -> 0   (predicate sum*2 >  n; conservative)
    drop     : drop every tied group, so no tie rule is applied
    """
    import pandas as pd

    df = pd.DataFrame({"s": [str(x) for x in smiles], "y": np.asarray(y, dtype=int)})
    grp = df.groupby("s")["y"]
    rows = []
    n_ties = 0
    for s, v in grp:
        tot = int(v.sum())
        n = len(v)
        if tot * 2 == n:
            n_ties += 1
            if disposition == "active":
                rows.append((s, 1))
            elif disposition == "inactive":
                rows.append((s, 0))
            elif disposition == "drop":
                continue
            else:
                raise ValueError(disposition)
        else:
            rows.append((s, int(tot * 2 > n)))
    rows.sort(key=lambda r: r[0])
    us = [r[0] for r in rows]
    uy = np.asarray([r[1] for r in rows], dtype=int)
    return us, uy, n_ties


def _pooled_auc(oof_y, oof_p):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(oof_y), np.asarray(oof_p)))


def _bootstrap_ci(oof_y, oof_p, n_boot=10000, seed=42):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(oof_y)
    p = np.asarray(oof_p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metal", default="Ru")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 49, 123])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument(
        "--out",
        default="metrics/by_metric/l4_ru_within_metal_tie_robustness.json",
    )
    args = ap.parse_args()

    raw_s, raw_y = _load_metal(args.metal)
    krasnov = KRASNOV_WITHIN_METAL_AUC.get(args.metal)

    report: dict = {
        "metric": "l4_ru_within_metal_tie_robustness",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/l4_ru_tie_robustness.py",
        "metal": args.metal,
        "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026)",
        "seeds": args.seeds,
        "n_rows_raw": int(len(raw_y)),
        "krasnov_published_auc": krasnov,
        "why": (
            "L4: the within-metal Ru claim vs Krasnov 0.81 flips with the "
            "duplicate tie rule. Test whether dropping the contradictory "
            "groups (no tie rule at all) lands it on one side under every seed."
        ),
        "dispositions": {},
    }

    for disp in ("active", "inactive", "drop"):
        us, uy, n_ties = _collapse(raw_s, raw_y, disp)
        per_seed = []
        for sd in args.seeds:
            r = within_metal(
                us, uy, sd, args.radius, args.nbits, return_scores=True
            )
            oof_y = r["oof_labels"]
            oof_p = r["oof_probs"]
            per_seed.append(
                {
                    "seed": sd,
                    "fold_mean_auc": float(r["roc_auc"]),
                    "fold_std": float(r["roc_auc_fold_std"]),
                    "pooled_oof_auc": _pooled_auc(oof_y, oof_p),
                }
            )
        # bootstrap CI on the pooled OOF estimate, seed 42 run
        r42 = within_metal(us, uy, args.seeds[0], args.radius, args.nbits, return_scores=True)
        lo, hi = _bootstrap_ci(r42["oof_labels"], r42["oof_probs"], args.n_boot)
        fold_means = [p["fold_mean_auc"] for p in per_seed]
        pooled = [p["pooled_oof_auc"] for p in per_seed]
        report["dispositions"][disp] = {
            "n_unique": int(len(uy)),
            "n_tie_groups_handled": n_ties,
            "pos_rate": float(uy.mean()),
            "per_seed": per_seed,
            "fold_mean_auc": {
                "min": float(np.min(fold_means)),
                "max": float(np.max(fold_means)),
                "mean": float(np.mean(fold_means)),
            },
            "pooled_oof_auc": {
                "min": float(np.min(pooled)),
                "max": float(np.max(pooled)),
                "mean": float(np.mean(pooled)),
                "ci95_seed42": [lo, hi],
            },
            "beats_krasnov": {
                "fold_mean_all_seeds": bool(np.min(fold_means) > krasnov),
                "pooled_all_seeds": bool(np.min(pooled) > krasnov),
            },
        }

    # verdict
    d = report["dispositions"]
    robust = d["drop"]["beats_krasnov"]["pooled_all_seeds"] and d["drop"]["beats_krasnov"][
        "fold_mean_all_seeds"
    ]
    report["verdict"] = {
        "drop_is_robust_win": bool(robust),
        "summary": (
            f"active rule Ru {d['active']['fold_mean_auc']['min']:.4f}-"
            f"{d['active']['fold_mean_auc']['max']:.4f}; inactive "
            f"{d['inactive']['fold_mean_auc']['min']:.4f}-"
            f"{d['inactive']['fold_mean_auc']['max']:.4f}; drop "
            f"{d['drop']['fold_mean_auc']['min']:.4f}-"
            f"{d['drop']['fold_mean_auc']['max']:.4f} (vs {krasnov})"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=1))
    for disp in ("active", "inactive", "drop"):
        x = report["dispositions"][disp]
        print(
            f"{disp:9s} n={x['n_unique']:5d} fold_mean "
            f"{x['fold_mean_auc']['min']:.4f}-{x['fold_mean_auc']['max']:.4f} "
            f"| pooled {x['pooled_oof_auc']['min']:.4f}-{x['pooled_oof_auc']['max']:.4f} "
            f"| beats {krasnov}: {x['beats_krasnov']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
