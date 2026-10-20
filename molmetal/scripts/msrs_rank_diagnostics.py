#!/usr/bin/env python3
"""C3 diagnostics — why the ordering claim is not identified by this cohort.

The three ordering routes (level difference 0.5826, fusion 0.5505, direct delta
0.5183) all miss the pre-registered 0.65 sign gate.  Before treating that as a
model failure this script asks three cheaper questions, each answerable from the
pair dump alone:

1. **Coverage.**  A "which metal is better" claim needs pairs across the metal
   graph.  How many of the 10 possible metal pairs does the cohort actually
   contain, and how many survive the |delta| >= 0.2091 gate?

2. **Is a zero-parameter prior already competitive?**  Rank by the training
   fold's per-metal mean potency and compare signs.  If the learned model does
   not beat this, the model is not adding information about metals.

3. **How much of the remaining error is measurement noise?**  Rows that share
   (ligand, metal, cell line, time) are replicate measurements of the same
   quantity.  Their within-group spread sets the smallest delta that can be
   resolved, and therefore the ceiling on sign accuracy for any model.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_rank_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_direct import cluster_bootstrap  # noqa: E402
from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Os", "Rh", "Re")
REPLICATE_KEYS = ("ligand_canonical", "metal", "cell_line_key", "time_h")
MIN_PAIRS_FOR_PAIR_LEVEL = 20


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def sign_acc(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(obs)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_pairwise_reranker" / "pair_oof_doi.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_rank_diagnostics.json",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with np.load(args.pairs, allow_pickle=False) as data:
        frame = pd.DataFrame({k: data[k] for k in ("y", "metal", "doi", "ligand_canonical", "cell_line_key", "time_h", "fold")})
        pair_a = data["pair_a"]
        pair_b = data["pair_b"]
        pair_obs = data["pair_obs"]
        pair_pred = data["pair_pred"]

    # One orientation per unordered pair: lower metal rank in `a` (same
    # convention as msrs_delta_direct.canonical_pairs).
    rank = {m: i for i, m in enumerate(sorted(frame["metal"].unique()))}
    ma = np.array([rank[m] for m in frame["metal"].to_numpy()[pair_a]])
    mb = np.array([rank[m] for m in frame["metal"].to_numpy()[pair_b]])
    keep = ma < mb
    pair_a, pair_b = pair_a[keep], pair_b[keep]
    obs, pred = pair_obs[keep], pair_pred[keep]

    metal_a = frame["metal"].to_numpy()[pair_a]
    metal_b = frame["metal"].to_numpy()[pair_b]
    fold_of_pair = frame["fold"].to_numpy()[pair_a]
    doi_of_pair = frame["doi"].to_numpy()[pair_a]
    ligand_of_pair = frame["ligand_canonical"].to_numpy()[pair_a]
    gated = np.abs(obs) >= GATE_THRESHOLD

    # --- 2. zero-parameter prior: training-fold per-metal mean potency -------
    prior = np.full(len(obs), np.nan)
    for k in sorted(frame["fold"].unique()):
        means = frame[frame["fold"] != k].groupby("metal")["y"].mean()
        sel = fold_of_pair == k
        prior[sel] = means.reindex(metal_b[sel]).to_numpy() - means.reindex(metal_a[sel]).to_numpy()
    if not np.isfinite(prior).all():
        raise RuntimeError("metal prior undefined for some pairs (unseen metal in a training fold?)")

    def block(p: np.ndarray, mask: np.ndarray, with_boot: bool) -> Dict[str, object]:
        out = {
            "n_pairs": int(mask.sum()),
            "sign_accuracy": round(sign_acc(obs[mask], p[mask]), 4),
            "mae_delta": round(float(np.mean(np.abs(p[mask] - obs[mask]))), 4),
        }
        if with_boot:
            # gate=0.0: the mask already applies GATE_THRESHOLD where needed.
            boot = cluster_bootstrap(obs[mask], p[mask], doi_of_pair[mask], 0.0,
                                     n_boot=args.n_boot, seed=args.seed)
            out["bootstrap"] = boot
        return out

    routes = {
        "learned_pair_model": {
            "gated": block(pred, gated, True),
            "ungated": block(pred, np.ones(len(obs), bool), False),
        },
        "metal_mean_prior": {
            "gated": block(prior, gated, True),
            "ungated": block(prior, np.ones(len(obs), bool), False),
        },
    }

    # --- 1. coverage --------------------------------------------------------
    pair_key = np.array([f"{a}<{b}" for a, b in zip(metal_a, metal_b)])
    coverage: List[Dict[str, object]] = []
    for key in sorted(set(pair_key.tolist())):
        sel = pair_key == key
        g = sel & gated
        ok_model = np.sign(pred[g]) == np.sign(obs[g])
        ok_prior = np.sign(prior[g]) == np.sign(obs[g])
        coverage.append(
            {
                "metal_pair": key,
                "n_pairs": int(sel.sum()),
                "n_gated": int(g.sum()),
                "n_dois": int(pd.unique(doi_of_pair[sel]).size),
                "n_ligands": int(pd.unique(ligand_of_pair[sel]).size),
                "obs_mean_delta": round(float(obs[g].mean()), 4) if g.any() else None,
                "learned_sign": round(float(ok_model.mean()), 4) if g.any() else None,
                "learned_ci95_wilson": wilson(int(ok_model.sum()), int(g.sum())),
                "prior_sign": round(float(ok_prior.mean()), 4) if g.any() else None,
                "prior_ci95_wilson": wilson(int(ok_prior.sum()), int(g.sum())),
            }
        )

    all_pairs_seen = sorted(set(pair_key.tolist()))
    possible = len(METALS) * (len(METALS) - 1) // 2
    gated_pairs_seen = sorted(set(pair_key[gated].tolist()))

    # --- 3. replicate noise ceiling ----------------------------------------
    grouped = frame.groupby(list(REPLICATE_KEYS))["y"]
    sizes = grouped.size()
    spread = grouped.std().dropna()
    sigma_mean = float(spread.mean())
    sigma_median = float(spread.median())
    from scipy.stats import norm  # local: keeps the import cost off the hot path

    ceiling = {}
    for name, sigma in (("mean_sd", sigma_mean), ("median_sd", sigma_median)):
        probs = norm.cdf(np.abs(obs[gated]) / (np.sqrt(2.0) * sigma))
        ceiling[name] = {
            "sigma": round(sigma, 4),
            "ceiling_sign_accuracy": round(float(probs.mean()), 4),
            "share_of_gated_pairs_resolvable": round(float((np.abs(obs[gated]) >= 1.96 * np.sqrt(2.0) * sigma).mean()), 4),
        }

    payload = {
        "artifact": "msrs_rank_diagnostics",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_rank_diagnostics.py",
            "git_commit": _git_commit(),
            "pair_dump": str(args.pairs.relative_to(PROJECT_ROOT)),
            "note": "pair predictions are the DOI-held-out OOF values from the pairwise reranker",
        },
        "design": {
            "gate_threshold": GATE_THRESHOLD,
            "n_unordered_pairs": int(len(obs)),
            "n_gated_pairs": int(gated.sum()),
            "same_doi_share": round(float(np.mean(frame["doi"].to_numpy()[pair_a] == frame["doi"].to_numpy()[pair_b])), 4),
            "n_boot": args.n_boot,
            "replicate_keys": list(REPLICATE_KEYS),
            "min_pairs_for_pair_level": MIN_PAIRS_FOR_PAIR_LEVEL,
        },
        "coverage": {
            "metal_pairs_possible": possible,
            "metal_pairs_present": len(all_pairs_seen),
            "metal_pairs_gated": len(gated_pairs_seen),
            "gated_pairs_missing": sorted(
                set(f"{a}<{b}" for i, a in enumerate(sorted(METALS)) for b in sorted(METALS)[i + 1:])
                - set(gated_pairs_seen)
            ),
            "per_pair": coverage,
        },
        "routes": routes,
        "replicate_noise": {
            "n_replicate_groups": int((sizes > 1).sum()),
            "n_rows_in_replicates": int(sizes[sizes > 1].sum()),
            **ceiling,
        },
        "verdict": {
            "learned_beats_prior_gated": bool(
                routes["learned_pair_model"]["gated"]["sign_accuracy"]
                > routes["metal_mean_prior"]["gated"]["sign_accuracy"]
            ),
            "gated_sign_gap": round(
                routes["metal_mean_prior"]["gated"]["sign_accuracy"]
                - routes["learned_pair_model"]["gated"]["sign_accuracy"],
                4,
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for row in coverage:
        print(
            f"[c3d] {row['metal_pair']:8s} pairs={row['n_pairs']:4d} gated={row['n_gated']:4d} "
            f"learned={row['learned_sign']} prior={row['prior_sign']}"
        )
    print(
        f"[c3d] gated metal pairs {len(gated_pairs_seen)}/{possible}  "
        f"learned={routes['learned_pair_model']['gated']['sign_accuracy']} "
        f"prior={routes['metal_mean_prior']['gated']['sign_accuracy']} "
        f"ceiling={ceiling['mean_sd']['ceiling_sign_accuracy']}"
    )
    print(f"[c3d] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
