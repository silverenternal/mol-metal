#!/usr/bin/env python3
"""MET-3 — effect-size metrics for the metal swap: how big, not just which way.

The only delta number in the project is a sign accuracy (0.6152).  A sign can
be right while the magnitude is wrong by a factor of three, and the decision
layer consumes magnitudes, not signs.  This script turns the matched-pair
predictions from MET-2 into the metrics a reader needs to judge a predicted
metal swap:

  * sign accuracy with a DOI-clustered bootstrap CI;
  * MAE(delta) and RMSE(delta) on the log10 scale;
  * calibration of predicted delta against observed delta
    (slope, intercept, R2, and a through-origin slope);
  * the same broken down by metal pair and by |delta| stratum.

Pairs are the same objects MET-2 scored: rows sharing
(ligand_canonical, cell_line_key, doi, time_h) with at least two distinct
metals, taken in a canonical metal order so "Ru-Ir" never appears twice.
Predictions are out-of-fold for every protocol, so a pair inside a held-out
paper is scored by a model that never saw that paper.

Acceptance (pre-registered in todo.json): on the ``doi`` protocol at the
0.2091 noise floor, sign accuracy >= 0.65 and the CI lower bound > 0.55.
Anything else is a boundary result, not a finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
ARM = "ligand+both"
THRESHOLDS = (0.0, 0.1, 0.2091, 0.5)
SIGN_GATE = 0.65
CI_LOWER_GATE = 0.55
GATE_THRESHOLD = 0.2091
N_BOOT = 2000
SEED = 20261018


# ---------------------------------------------------------------------------
# Metrics on one pair set
# ---------------------------------------------------------------------------
def delta_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    """Metrics for observed vs predicted deltas (both already subset)."""
    if len(obs) == 0:
        return {"n_pairs": 0, "status": "EMPTY"}
    err = pred - obs
    out: Dict[str, object] = {
        "n_pairs": int(len(obs)),
        "mae_delta": round(float(np.mean(np.abs(err))), 4),
        "rmse_delta": round(float(np.sqrt(np.mean(err**2))), 4),
        "bias_delta": round(float(np.mean(err)), 4),
        "sign_accuracy": round(float(np.mean(np.sign(pred) == np.sign(obs))), 4),
        "mean_abs_observed_delta": round(float(np.mean(np.abs(obs))), 4),
        "mean_abs_predicted_delta": round(float(np.mean(np.abs(pred))), 4),
    }
    if float(np.std(pred)) > 0:
        slope, intercept = np.polyfit(pred, obs, 1)
        pred_c = pred - pred.mean()
        obs_c = obs - obs.mean()
        denom = float(np.sum(pred_c**2))
        r2 = float(np.sum(pred_c * obs_c) ** 2) / (
            denom * float(np.sum(obs_c**2))
        ) if denom > 0 and float(np.sum(obs_c**2)) > 0 else 0.0
        out["calibration"] = {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r2": round(float(r2), 4),
            "slope_through_origin": round(
                float(np.sum(pred * obs) / max(np.sum(pred**2), 1e-12)), 4
            ),
        }
    else:
        out["calibration"] = {"status": "ZERO_VARIANCE"}
    return out


def _pair_arrays(
    y: np.ndarray, pred: np.ndarray, ia: np.ndarray, ib: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return y[ib] - y[ia], pred[ib] - pred[ia]


def cluster_bootstrap(
    y: np.ndarray,
    pred: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    doi: np.ndarray,
    keep: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, object]:
    """Bootstrap over DOIs, not over pairs.

    Two pairs from the same paper share ligands, cell lines and assay protocol;
    resampling pairs would treat them as independent and shrink the interval.
    """
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return {"status": "TOO_FEW_PAIRS"}
    clusters = doi[ia][idx]
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 3:
        return {"status": "TOO_FEW_DOIS", "n_dois": int(len(uniq))}
    buckets = [idx[inv == g] for g in range(len(uniq))]
    obs_all, pred_all = _pair_arrays(y, pred, ia, ib)

    rng = np.random.default_rng(seed)
    signs, maes, slopes = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([buckets[p] for p in pick])
        obs, prd = obs_all[sel], pred_all[sel]
        signs.append(float(np.mean(np.sign(prd) == np.sign(obs))))
        maes.append(float(np.mean(np.abs(prd - obs))))
        if float(np.std(prd)) > 0:
            slopes.append(float(np.polyfit(prd, obs, 1)[0]))
    out: Dict[str, object] = {"n_dois": int(len(uniq)), "n_boot": int(len(signs))}
    for name, vals in (("sign_accuracy", signs), ("mae_delta", maes), ("slope", slopes)):
        if not vals:
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = {
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "mean": round(float(np.mean(vals)), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------
def by_metal_pair(
    obs: np.ndarray, pred: np.ndarray, names: np.ndarray
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in sorted(set(names.tolist())):
        m = names == name
        block = delta_metrics(obs[m], pred[m])
        out[str(name)] = {
            "n_pairs": block["n_pairs"],
            "sign_accuracy": block.get("sign_accuracy"),
            "mae_delta": block.get("mae_delta"),
            "calibration": block.get("calibration"),
        }
    return out


def by_stratum(obs: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    edges = [0.0, 0.1, 0.2091, 0.5, 1.0, np.inf]
    out: Dict[str, object] = {}
    mag = np.abs(obs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag < hi)
        label = f"[{lo:g},{hi:g})" if np.isfinite(hi) else f"[{lo:g},inf)"
        block = delta_metrics(obs[m], pred[m])
        out[label] = {
            "n_pairs": block["n_pairs"],
            "sign_accuracy": block.get("sign_accuracy"),
            "mae_delta": block.get("mae_delta"),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_generalization" / "oof.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_delta_metrics.json",
    )
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    data = np.load(args.oof, allow_pickle=False)
    y = data["y"].astype(float)
    doi = data["doi"].astype(str)
    metal = data["metal"].astype(str)
    frame = pd.DataFrame(
        {
            "ligand_canonical": data["ligand_canonical"].astype(str),
            "cell_line_key": data["cell_line_key"].astype(str),
            "doi": doi,
            "time_h": data["time_h"].astype(float),
            "metal": metal,
        }
    )
    pairs = pair_table(frame)
    if pairs.empty:
        raise SystemExit("no matched pairs")
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    pair_names = np.array(
        [f"{a}-{b}" for a, b in zip(pairs["metal_a"], pairs["metal_b"])]
    )
    print(f"[met3] pairs={len(pairs)} dois={pairs['doi'].nunique()}")

    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        key = f"{protocol}__{args.arm}"
        if key not in data:
            protocols[protocol] = {"status": "MISSING_OOF", "key": key}
            continue
        fold = data[f"{protocol}__fold"]
        pred = data[key]
        # A pair is scored only when both rows were held out at some point.
        valid = (fold[ia] >= 0) & (fold[ib] >= 0) & np.isfinite(pred[ia]) & np.isfinite(pred[ib])
        obs_all, pred_all = _pair_arrays(y, pred, ia, ib)

        blocks: Dict[str, object] = {}
        for th in THRESHOLDS:
            keep = valid & (np.abs(obs_all) >= th)
            block = delta_metrics(obs_all[keep], pred_all[keep])
            block["bootstrap"] = cluster_bootstrap(
                y, pred, ia, ib, doi, keep, n_boot=args.n_boot, seed=args.seed
            )
            blocks[f"{th:g}"] = block

        keep0 = valid & (np.abs(obs_all) >= 0.0)
        protocols[protocol] = {
            "n_pairs_scored": int(valid.sum()),
            "n_dois": int(pd.Series(doi[ia][valid]).nunique()),
            "thresholds": blocks,
            "by_metal_pair": by_metal_pair(
                obs_all[keep0], pred_all[keep0], pair_names[keep0]
            ),
            "by_abs_delta": by_stratum(obs_all[keep0], pred_all[keep0]),
        }
        main_block = blocks[f"{GATE_THRESHOLD:g}"]
        print(
            f"[met3] {protocol:9s} n={main_block.get('n_pairs')} "
            f"sign={main_block.get('sign_accuracy')} "
            f"mae={main_block.get('mae_delta')} "
            f"slope={main_block.get('calibration', {}).get('slope')}"
        )

    doi_block = protocols.get("doi", {}).get("thresholds", {}).get(
        f"{GATE_THRESHOLD:g}", {}
    )
    sign = doi_block.get("sign_accuracy")
    ci = doi_block.get("bootstrap", {}).get("sign_accuracy", {}).get("ci95")
    passed = bool(
        sign is not None
        and sign >= SIGN_GATE
        and ci is not None
        and ci[0] > CI_LOWER_GATE
    )
    verdict = "PASS" if passed else "NEGATIVE"

    payload = {
        "artifact": "msrs_delta_metrics",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof),
            "oof_sha256": _sha256(args.oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "split_schema": (
                "consumes MET-2's out-of-fold predictions; four protocols, so the "
                "dual-split gate does not apply"
            ),
            "arm": args.arm,
            "protocols": list(args.protocols),
            "pair_key": ["ligand_canonical", "cell_line_key", "doi", "time_h"],
            "thresholds": list(THRESHOLDS),
            "gate_threshold": GATE_THRESHOLD,
            "sign_gate": SIGN_GATE,
            "ci_lower_gate": CI_LOWER_GATE,
            "bootstrap": {"unit": "doi", "n_boot": args.n_boot},
            "note": (
                "pairs are keyed on the complex, not the main fragment, because "
                "the features are built on ligand_canonical; the 1284-pair count "
                "in todo.json came from the fragment-keyed surface model"
            ),
        },
        "protocols": protocols,
        "verdict": verdict,
        "verdict_rule": (
            f"doi protocol at |delta| >= {GATE_THRESHOLD}: sign accuracy >= "
            f"{SIGN_GATE} AND DOI-clustered CI lower > {CI_LOWER_GATE}"
        ),
        "gate_measurement": {
            "sign_accuracy": sign,
            "ci95": ci,
            "n_pairs": doi_block.get("n_pairs"),
            "mae_delta": doi_block.get("mae_delta"),
            "calibration": doi_block.get("calibration"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(
        args.out, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[met3] verdict={verdict} sign={sign} ci={ci}")
    print(f"[met3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
