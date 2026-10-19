#!/usr/bin/env python3
"""STR-1 — is the single-seed OOF behind MET-3/4/5 dominated by seed noise?

MET-3 (delta), MET-4 (calibration) and MET-5 (decision) all consume one
out-of-fold prediction file (``results/msrs_generalization/oof.npz``, seed
20261018).  MET-1 measured a seed SD of 0.0102 R2 on the blend, so a single
seed is not obviously enough for a second-decimal claim.

This script loads N per-seed OOF files produced by
``msrs_generalization.py --seed S --oof-out ...``, averages the predictions,
and recomputes the delta / calibration / decision blocks on the average.  It
also reports the per-seed spread of every headline number.

Output: ``metrics/by_metric/msrs_seed_ensemble.json``.

Usage:
    .venv/bin/python molmetal/scripts/msrs_seed_ensemble.py \
        --dir results/msrs_seed_ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_calibration import (  # noqa: E402
    cohort_sizes,
    run_protocol as calibration_run,
)
from molmetal.scripts.msrs_decision_curve import (  # noqa: E402
    run_protocol as decision_run,
)
from molmetal.scripts.msrs_delta_metrics import (  # noqa: E402
    GATE_THRESHOLD,
    THRESHOLDS,
    _pair_arrays,
    cluster_bootstrap,
    delta_metrics,
)
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
ARM = "ligand+both"
DEFAULT_DIR = PROJECT_ROOT / "results" / "msrs_seed_ensemble"
DEFAULT_OUT = PROJECT_ROOT / "metrics" / "by_metric" / "msrs_seed_ensemble.json"
N_BOOT = 2000
SEED_SD_GATE = 0.05
SINGLE_SEED_ARTIFACTS = {
    "generalization": "msrs_generalization.json",
    "delta": "msrs_delta_metrics.json",
    "calibration": "msrs_calibration.json",
    "decision": "msrs_decision_curve.json",
}


def _load_oof(paths: List[Path]) -> Dict[str, np.ndarray]:
    """Stack per-seed OOF files after checking the rows line up exactly."""
    if not paths:
        raise SystemExit("no OOF files found")
    loaded = [np.load(p, allow_pickle=False) for p in paths]
    first = loaded[0]
    for path, data in zip(paths[1:], loaded[1:]):
        if not np.array_equal(first["y"], data["y"]):
            raise SystemExit(f"{path}: y differs from {paths[0]}")
        for protocol in PROTOCOLS:
            fkey = f"{protocol}__fold"
            if not np.array_equal(first[fkey], data[fkey]):
                raise SystemExit(f"{path}: {fkey} differs from {paths[0]}")

    out: Dict[str, Any] = {
        "y": first["y"].astype(float),
        "doi": first["doi"].astype(str),
        "metal": first["metal"].astype(str),
        "cell_line_key": first["cell_line_key"].astype(str),
        "ligand_canonical": first["ligand_canonical"].astype(str),
        "time_h": first["time_h"].astype(float),
    }
    for protocol in PROTOCOLS:
        out[f"{protocol}__fold"] = first[f"{protocol}__fold"]
        out[f"{protocol}__preds"] = np.column_stack(
            [d[f"{protocol}__{ARM}"].astype(float) for d in loaded]
        )
    out["seeds"] = [p.stem.replace("oof_seed", "") for p in paths]
    return out


def _per_seed(y: np.ndarray, preds: np.ndarray, fold: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> List[Dict[str, Any]]:
    ev = fold >= 0
    valid = (fold[ia] >= 0) & (fold[ib] >= 0)
    obs = y[ib] - y[ia]
    keep = valid & (np.abs(obs) >= GATE_THRESHOLD)
    rows = []
    for j in range(preds.shape[1]):
        p = preds[:, j]
        d = p[ib] - p[ia]
        rows.append(
            {
                **score(y[ev], p[ev]),
                "sign_at_gate": round(float(np.mean(np.sign(d[keep]) == np.sign(obs[keep]))), 4),
                "mae_delta": round(float(np.mean(np.abs(d[keep] - obs[keep]))), 4),
            }
        )
    return rows


def _summary(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        if vals.size:
            out[key] = {
                "mean": round(float(vals.mean()), 4),
                "sd": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
            }
    return out


def _pick(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Nested ``.get`` that returns ``default`` on any missing level."""
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _reference(metrics_dir: Path) -> Dict[str, Any]:
    """The published single-seed numbers, read rather than recomputed.

    Calibration and the decision curve were only measured on ``ligand`` and
    ``doi`` (MET-4 / MET-5); the missing protocols stay ``None`` rather than
    being invented.
    """
    ref: Dict[str, Any] = {}
    try:
        gen = json.loads((metrics_dir / SINGLE_SEED_ARTIFACTS["generalization"]).read_text())
        delta = json.loads((metrics_dir / SINGLE_SEED_ARTIFACTS["delta"]).read_text())
        cal = json.loads((metrics_dir / SINGLE_SEED_ARTIFACTS["calibration"]).read_text())
        dec = json.loads((metrics_dir / SINGLE_SEED_ARTIFACTS["decision"]).read_text())
    except FileNotFoundError as exc:
        return {"status": "MISSING", "error": str(exc)}
    gate = f"{GATE_THRESHOLD:g}"
    for protocol in PROTOCOLS:
        ref[protocol] = {
            "r2": _pick(gen, "splits", protocol, "arms", ARM, "r2"),
            "sign_at_gate": _pick(
                delta, "protocols", protocol, "thresholds", gate, "sign_accuracy"
            ),
            "mae_delta": _pick(
                delta, "protocols", protocol, "thresholds", gate, "mae_delta"
            ),
            "calibration_coverage": _pick(
                cal, "protocols", protocol, "variants", "global", "coverage"
            ),
            "calibration_median_width": _pick(
                cal, "protocols", protocol, "variants", "global", "median_width"
            ),
            "decision_regret_full": _pick(
                dec, "protocols", protocol, "full_coverage", "model", "mean_regret"
            ),
        }
    return ref


def build(directory: Path = DEFAULT_DIR, metrics_dir: Path = PROJECT_ROOT / "metrics" / "by_metric") -> Dict[str, Any]:
    paths = sorted(directory.glob("oof_seed*.npz"))
    data = _load_oof(paths)
    y = data["y"]
    doi = data["doi"]
    frame = pd.DataFrame(
        {
            "ligand_canonical": data["ligand_canonical"],
            "cell_line_key": data["cell_line_key"],
            "doi": doi,
            "time_h": data["time_h"],
            "metal": data["metal"],
        }
    )
    pairs = pair_table(frame)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    cohort_k = cohort_sizes(frame)
    reference = _reference(metrics_dir)

    protocols: Dict[str, Any] = {}
    for protocol in PROTOCOLS:
        t0 = time.time()
        fold = data[f"{protocol}__fold"]
        preds = data[f"{protocol}__preds"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred = np.nanmean(preds, axis=1)
        ev = fold >= 0

        obs_all, pred_all = _pair_arrays(y, pred, ia, ib)
        valid = (fold[ia] >= 0) & (fold[ib] >= 0) & np.isfinite(pred[ia]) & np.isfinite(pred[ib])
        blocks: Dict[str, Any] = {}
        for th in THRESHOLDS:
            keep = valid & (np.abs(obs_all) >= th)
            block = delta_metrics(obs_all[keep], pred_all[keep])
            block["bootstrap"] = cluster_bootstrap(
                y, pred, ia, ib, doi, keep, n_boot=N_BOOT, seed=20261018
            )
            blocks[f"{th:g}"] = block

        cal = calibration_run(
            y, pred, fold, data["metal"], data["cell_line_key"], cohort_k
        )
        dec = decision_run(frame, y, pred, protocol, seed=20261018)
        per_seed = _per_seed(y, preds, fold, ia, ib)

        gate = f"{GATE_THRESHOLD:g}"
        protocols[protocol] = {
            "n_evaluated": int(ev.sum()),
            "n_pairs_scored": int(valid.sum()),
            "per_seed": per_seed,
            "seed_summary": _summary(per_seed, ["r2", "mae", "rmse", "sign_at_gate", "mae_delta"]),
            "ensemble": {
                "r2": round(float(score(y[ev], pred[ev])["r2"]), 4),
                "mae": round(float(score(y[ev], pred[ev])["mae"]), 4),
                "rmse": round(float(score(y[ev], pred[ev])["rmse"]), 4),
                "delta": blocks,
                "delta_gate": {
                    "sign_accuracy": blocks[gate]["sign_accuracy"],
                    "ci95": blocks[gate].get("bootstrap", {}).get("sign_accuracy", {}).get("ci95"),
                    "mae_delta": blocks[gate]["mae_delta"],
                },
                "calibration": {
                    name: {
                        k: cal["variants"][name].get(k)
                        for k in ("coverage", "median_width", "winkler")
                    }
                    | {"pit_ks": cal["variants"][name].get("pit", {}).get("ks_stat")}
                    for name in ("global", "metal", "difficulty")
                },
                "decision_full_coverage": {
                    k: dec["full_coverage"]["model"].get(k)
                    for k in ("mean_regret", "top1_accuracy")
                },
                "decision_coverage_0.5": next(
                    (
                        {
                            "mean_regret": row["model"]["mean_regret"],
                            "top1_accuracy": row["model"]["top1_accuracy"],
                            "p_holm": row["model_vs_global_best"]["p_holm"],
                        }
                        for row in dec["curve"]
                        if abs(row["coverage"] - 0.5) < 1e-9
                    ),
                    None,
                ),
            },
            "single_seed_reference": reference.get(protocol) if isinstance(reference, dict) else None,
            "wall_s": round(time.time() - t0, 1),
        }

        ref = (reference or {}).get(protocol) or {}
        protocols[protocol]["delta_vs_single_seed"] = {
            "r2": _diff(protocols[protocol]["ensemble"]["r2"], ref.get("r2")),
            "sign_at_gate": _diff(protocols[protocol]["ensemble"]["delta_gate"]["sign_accuracy"], ref.get("sign_at_gate")),
            "mae_delta": _diff(protocols[protocol]["ensemble"]["delta_gate"]["mae_delta"], ref.get("mae_delta")),
            "median_width": _diff(
                protocols[protocol]["ensemble"]["calibration"]["global"]["median_width"],
                ref.get("calibration_median_width"),
            ),
            "decision_regret_full": _diff(
                protocols[protocol]["ensemble"]["decision_full_coverage"]["mean_regret"],
                ref.get("decision_regret_full"),
            ),
        }
        print(
            f"[str1] {protocol:9s} R2={protocols[protocol]['ensemble']['r2']:.4f} "
            f"sign={protocols[protocol]['ensemble']['delta_gate']['sign_accuracy']:.4f} "
            f"(seed sd {protocols[protocol]['seed_summary'].get('sign_at_gate', {}).get('sd')})"
        )

    max_sd = max(
        (p["seed_summary"].get("sign_at_gate", {}).get("sd", 0.0) for p in protocols.values()),
        default=0.0,
    )
    verdict = "SEED_NOISE_DOMINATED" if max_sd > SEED_SD_GATE else "STABLE"

    return {
        "artifact": "msrs_seed_ensemble",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_seed_ensemble.py",
            "git_commit": _git_commit(),
            "oof_files": [_display_path(p) for p in paths],
            "oof_sha256": {p.name: _sha256(p) for p in paths},
            "seeds": data["seeds"],
        },
        "design": {
            "protocols": list(PROTOCOLS),
            "arm": ARM,
            "aggregation": "mean of per-seed out-of-fold predictions; folds are fixed across seeds",
            "n_boot": N_BOOT,
            "seed_sd_gate": SEED_SD_GATE,
            "note": "per-seed metrics use the same rows; the ensemble prediction is not refit",
        },
        "protocols": protocols,
        "verdict": verdict,
        "verdict_rule": f"max per-protocol seed SD of delta sign at |d|>={GATE_THRESHOLD} > {SEED_SD_GATE} => SEED_NOISE_DOMINATED",
        "max_sign_seed_sd": round(float(max_sd), 4),
    }


def _diff(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    artifact = build(args.dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(f"[str1] verdict={artifact['verdict']} max_sign_sd={artifact['max_sign_seed_sd']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
