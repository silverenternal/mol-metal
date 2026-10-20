#!/usr/bin/env python3
"""C3 — the delta-sign metric has a 0.60 floor, and nobody was measuring it.

``sign_at_gate`` is reported against 0.5 as if the two signs were balanced.
On the 218 gated pairs they are not: 60.1% are positive, and the corpus contains
only two metal pairs (Os-Ru 153, Ir-Rh 64), both leaning positive.  A model that
always answers "the alphabetically later metal is better" therefore scores
0.6009, and the published 0.5826 is *below* it.

This audit reports, for every route that has row-level predictions on disk:

* raw sign accuracy (the published metric);
* the majority-class rate (the trivial floor);
* balanced sign accuracy = (TPR + TNR) / 2, which is 0.5 for any constant
  predictor and is the only one of the three that answers "did it learn the
  ordering" rather than "did it learn the base rate";
* a DOI-clustered bootstrap interval for the balanced version.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_delta_metric_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_feature_expansion import select_cohort  # noqa: E402
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 20261020
ARM = "ligand+both"


def balanced_sign(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pos = obs > 0
    neg = ~pos
    tpr = float(np.mean(pred[pos] > 0)) if pos.any() else float("nan")
    tnr = float(np.mean(pred[neg] < 0)) if neg.any() else float("nan")
    return {
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "balanced": round((tpr + tnr) / 2.0, 4),
    }


def bootstrap_balanced(
    obs: np.ndarray, pred: np.ndarray, clusters: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> List[float]:
    uniq = np.unique(clusters)
    index = {u: np.where(clusters == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        rows = np.concatenate([index[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        o, p = obs[rows], pred[rows]
        if (o > 0).all() or (o < 0).all():
            continue
        values.append(balanced_sign(o, p)["balanced"])
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def level_model_delta(directory: Path, ia: np.ndarray, ib: np.ndarray) -> Optional[np.ndarray]:
    paths = sorted(directory.glob("oof_seed*.npz"))
    if not paths:
        return None
    preds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            preds.append(data[f"doi__{ARM}"].astype(float))
    mean = np.mean(np.column_stack(preds), axis=1)
    return mean[ib] - mean[ia]


def direct_delta(npz_path: Path, y: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> Optional[np.ndarray]:
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        pair_a = data["pair_a"].astype(int)
        pair_b = data["pair_b"].astype(int)
        pred = data["pair_pred"].astype(float)
    # the dump is over ordered pairs; map each (a, b) to its prediction
    lookup = {(int(a), int(b)): float(p) for a, b, p in zip(pair_a, pair_b, pred)}
    out = np.array([lookup.get((int(a), int(b)), np.nan) for a, b in zip(ia, ib)])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    parser.add_argument(
        "--level-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_seed_ensemble",
    )
    parser.add_argument(
        "--direct-pairs",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_pairwise_reranker" / "pair_oof_doi.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_delta_metric_audit.json",
    )
    args = parser.parse_args(argv)

    frame = select_cohort(args.long)
    y = frame["value_log10"].to_numpy(dtype=float)
    pairs = pair_table(frame)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    obs_all = y[ib] - y[ia]
    gated = np.abs(obs_all) >= GATE_THRESHOLD
    obs = obs_all[gated]
    clusters = frame["doi"].astype(str).to_numpy()[ia][gated]
    metal_pairs = pairs[["metal_a", "metal_b"]].astype(str).agg("-".join, axis=1).to_numpy()[gated]

    routes: Dict[str, object] = {}

    level = level_model_delta(args.level_dir, ia, ib)
    if level is not None:
        pred = level[gated]
        ok = np.isfinite(pred)
        routes["level_differencing"] = {
            "n": int(ok.sum()),
            "raw_sign": round(float(np.mean(np.sign(pred[ok]) == np.sign(obs[ok]))), 4),
            **balanced_sign(obs[ok], pred[ok]),
            "balanced_ci95": bootstrap_balanced(obs[ok], pred[ok], clusters[ok]),
        }

    direct = direct_delta(args.direct_pairs, y, ia, ib)
    if direct is not None:
        pred = direct[gated]
        ok = np.isfinite(pred)
        routes["direct_regression"] = {
            "n": int(ok.sum()),
            "raw_sign": round(float(np.mean(np.sign(pred[ok]) == np.sign(obs[ok]))), 4),
            **balanced_sign(obs[ok], pred[ok]),
            "balanced_ci95": bootstrap_balanced(obs[ok], pred[ok], clusters[ok]),
        }

    # trivial baselines
    majority = float((obs > 0).mean())
    always_positive = np.ones_like(obs)
    pair_prior = np.zeros_like(obs)
    for name in np.unique(metal_pairs):
        mask = metal_pairs == name
        pair_prior[mask] = 1.0 if float((obs[mask] > 0).mean()) > 0.5 else -1.0

    payload = {
        "artifact": "msrs_delta_metric_audit",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_delta_metric_audit.py",
            "git_commit": _git_commit(),
            "level_predictions": str(args.level_dir.relative_to(PROJECT_ROOT)),
            "direct_predictions": str(args.direct_pairs.relative_to(PROJECT_ROOT))
            if args.direct_pairs.is_relative_to(PROJECT_ROOT)
            else str(args.direct_pairs),
        },
        "design": {
            "gate_threshold": GATE_THRESHOLD,
            "n_gated_pairs": int(gated.sum()),
            "metric_note": "balanced = (TPR + TNR)/2; 0.5 for any constant predictor",
            "bootstrap": {"cluster": "doi", "n_boot": N_BOOT, "seed": BOOT_SEED},
        },
        "pair_composition": {
            name: {
                "n": int((metal_pairs == name).sum()),
                "fraction_positive": round(float((obs[metal_pairs == name] > 0).mean()), 4),
            }
            for name in np.unique(metal_pairs)
        },
        "baselines": {
            "majority_class_rate": round(majority, 4),
            "always_positive_sign": round(float(np.mean(np.sign(always_positive) == np.sign(obs))), 4),
            "metal_pair_prior_sign": round(float(np.mean(np.sign(pair_prior) == np.sign(obs))), 4),
            "metal_pair_prior_balanced": balanced_sign(obs, pair_prior)["balanced"],
        },
        "routes": routes,
        "verdict": {
            "floor": round(majority, 4),
            "published_level_raw": routes.get("level_differencing", {}).get("raw_sign"),
            "note": "raw sign must be read against the majority-class rate, not 0.5",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(f"[c3a] gated={int(gated.sum())} majority={majority:.4f} pair_prior={payload['baselines']['metal_pair_prior_sign']}")
    for name, row in routes.items():
        print(
            f"[c3a] {name:20s} raw={row['raw_sign']:.4f} balanced={row['balanced']:.4f} "
            f"ci={row['balanced_ci95']} tpr={row['tpr']} tnr={row['tnr']}"
        )
    print(f"[c3a] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
