#!/usr/bin/env python3
"""Wf-278A — TODO-181 G5 50-epoch full sweep re-aggregator (path+schema fix).

==============================================================================

The original G5 launcher (`scripts/launch_wf124_p3_g5.sh`) ran an inline
Python aggregation at end-of-sweep:

    for p in seed_dir.glob('per_seed_metrics_seed*.json'):
        runs.append(...)

This failed for two reasons (Wf-310A diagnosis):

  1. **Path mismatch** — the Wf-124 P3 trainer writes per_seed_metrics.json
     to a NESTED location:
        {out_dir}/{Metal}/seed{N}/{metal_lower}/per_seed_metrics.json
     but the launcher's inline glob walked only:
        {out_dir}/{Metal}/seed{N}/per_seed_metrics_seed*.json
     (missing the {metal_lower} intermediate directory). Result: 0 hits.

  2. **Schema mismatch** — when the launcher DID match (it didn't), the
     inline script stamped `{'metal': ..., 'seed': ..., 'metrics_path': ...}`
     into training_summary.json without ANY of the MEASURED test_auc /
     test_pearson / test_rmse values. The summary was structurally empty.

This aggregator fixes BOTH bugs:

  - Walks the correct nested tree
        <input-dir>/{Metal}/seed{N}/{metal_lower}/per_seed_metrics.json
        <input-dir>/{Metal}/seed{N}/{metal_lower}/per_seed_metrics_seed{N}.json
  - Reads the FLAT-schema wrapper `per_seed_metrics.json` (which is what the
    Wf-124 P3 trainer writes for G5) for primary aggregation, falling back
    to the per-seed seed-numbered file with the G4 nested schema if the
    wrapper is missing.
  - Aggregates per-metal mean ± std across seeds 0/1/2.
  - Emits training_summary.json with real `total_pairs` + per-metal
    breakdown + 5-metal grand aggregate + per-pair table.

Cf. the Wf-278A G4 aggregator (same module name) for the 30-epoch sibling
sweep (TODO-180 G4) at results/wf124_p3_remaining_metals_g4/. Both share the
same nested-layout problem; the schema difference (G5 flat vs G4 nested)
is the new wrinkle.

Usage:
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/wf278a_g5_aggregate.py \\
        --input-dir results/wf124_p3_50epoch_full_sweep_g5 \\
        --output   results/wf124_p3_50epoch_full_sweep_g5/training_summary.json

Output schema:
    {
        "wf_tag": "Wf-278A — TODO-181 G5 50-epoch re-aggregation (path+schema fix)",
        "input_dir": "results/wf124_p3_50epoch_full_sweep_g5",
        "metals": ["Ru", "Ir", "Os", "Rh", "Re"],
        "epochs": 50,
        "backbone": {"hidden": 300, "depth": 3},
        "total_pairs": 13,
        "expected_pairs": 15,
        "missing_pairs": ["Ir/seed0", "Ru/seed2"],
        "per_metal": {
            "Ru": {
                "metal": "Ru", "n_seeds": 2, "n_seeds_ok": 2,
                "test_auc_mean": ..., "test_auc_std": ...,
                "test_pearson_mean": ..., "test_rmse_mean": ...,
                "test_auc_per_seed": {"0": ..., "1": ...},
                ...
            },
            ...
        },
        "aggregate_5metal": {...},
        "ru_auc_target_projected": 0.86,
        "ru_auc_baseline_single_task_chemprop_v2": 0.7123,
        "honest_framing_note": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FIVE_METALS = ["Ru", "Ir", "Os", "Rh", "Re"]
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EPOCHS = 50


def _load_wrapper_schema(psm_path: Path, canonical_metal: str) -> Optional[Dict]:
    """Load per_seed_metrics.json wrapper (FLAT G5 schema). Returns None on miss."""
    try:
        data = json.loads(psm_path.read_text())
    except Exception:
        return None
    per_seed = data.get("per_seed") or []
    if not per_seed:
        return None
    ps = per_seed[0]
    if ps.get("status", "OK") != "OK":
        return None
    return {
        "metal": canonical_metal,
        "seed": int(ps.get("seed", 0)),
        "n_train": int(ps.get("n_train", 0)),
        "n_val": int(ps.get("n_val", 0)),
        "n_test": int(ps.get("n_test", 0)),
        "elapsed_seconds": ps.get("elapsed_seconds"),
        "test_auc": ps.get("test_auc"),
        "val_auc": ps.get("val_auc"),
        "test_pearson": ps.get("test_pearson"),
        "test_rmse": ps.get("test_rmse"),
        "test_mae": ps.get("test_mae"),
        "wasserstein_history": ps.get("wasserstein_history") or [],
        "ckpt_path": ps.get("ckpt_path"),
        "status": str(ps.get("status", "UNKNOWN")),
        "source_file": str(psm_path),
        "schema": "G5_flat_wrapper",
    }


def _load_perseed_schema(psm_path: Path) -> Optional[Dict]:
    """Load per_seed_metrics_seed{N}.json (NESTED G4 schema). Returns None on miss."""
    try:
        data = json.loads(psm_path.read_text())
    except Exception:
        return None
    if data.get("status", "OK") != "OK":
        return None
    test_metrics = data.get("test_metrics") or {}
    test_pic50 = data.get("test_pic50_metrics") or {}
    val_metrics = data.get("val_metrics") or {}
    return {
        "metal": data.get("metal") or psm_path.parent.name.upper(),
        "seed": int(data.get("seed", 0)),
        "n_train": int(data.get("n_train", 0)),
        "n_val": int(data.get("n_val", 0)),
        "n_test": int(data.get("n_test", 0)),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "test_auc": test_metrics.get("roc_auc"),
        "val_auc": val_metrics.get("roc_auc"),
        "test_pearson": test_pic50.get("pearson"),
        "test_rmse": test_pic50.get("rmse"),
        "test_mae": test_pic50.get("mae"),
        "wasserstein_history": data.get("wasserstein_history") or [],
        "ckpt_path": data.get("ckpt_path"),
        "status": str(data.get("status", "UNKNOWN")),
        "source_file": str(psm_path),
        "schema": "G4_nested_perseed",
    }


def _load_pair(g5_dir: Path, metal: str, seed: int) -> Optional[Dict]:
    """Try wrapper first (G5 flat), fall back to per-seed file (G4 nested)."""
    metal_lower = metal.lower()
    pair_dir = g5_dir / metal / f"seed{seed}"
    if not pair_dir.is_dir():
        return None
    sub = pair_dir / metal_lower
    wrapper = sub / "per_seed_metrics.json"
    if wrapper.exists():
        rec = _load_wrapper_schema(wrapper, metal)
        if rec is not None:
            return rec
    perseed = sub / f"per_seed_metrics_seed{seed}.json"
    if perseed.exists():
        rec = _load_perseed_schema(perseed)
        if rec is not None:
            return rec
    return None


def collect_all_pairs(g5_dir: Path) -> Tuple[List[Dict], List[Tuple[str, int]]]:
    """Walk G5 dir. Return (records, missing_pairs)."""
    records: List[Dict] = []
    missing: List[Tuple[str, int]] = []
    for metal in FIVE_METALS:
        for seed in DEFAULT_SEEDS:
            rec = _load_pair(g5_dir, metal, seed)
            if rec is None:
                missing.append((metal, seed))
            else:
                records.append(rec)
    records.sort(key=lambda r: (r["metal"], r["seed"]))
    return records, missing


def _safe_mean(xs: List[float]) -> float:
    finite = [x for x in xs if x is not None and not (isinstance(x, float) and x != x)]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _safe_std(xs: List[float]) -> float:
    finite = [x for x in xs if x is not None and not (isinstance(x, float) and x != x)]
    return float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0


def per_metal_summary(metal: str, seeds: List[Dict]) -> Dict:
    """Per-metal mean ± std across the available seeds."""
    if not seeds:
        return {"metal": metal, "n_seeds": 0, "status": "NO_SEEDS"}
    aucs = [s["test_auc"] for s in seeds]
    pearsons = [s["test_pearson"] for s in seeds]
    rmses = [s["test_rmse"] for s in seeds]
    maes = [s["test_mae"] for s in seeds]
    val_aucs = [s["val_auc"] for s in seeds]
    w_finals = [
        s["wasserstein_history"][-1] if s.get("wasserstein_history") else float("nan")
        for s in seeds
    ]
    elapsed = [s.get("elapsed_seconds") for s in seeds]
    return {
        "metal": metal,
        "n_seeds": len(seeds),
        "n_seeds_total": 3,
        "test_auc_mean": _safe_mean(aucs),
        "test_auc_std": _safe_std(aucs),
        "test_auc_per_seed": {
            str(s["seed"]): (float(s["test_auc"]) if s["test_auc"] is not None else None)
            for s in seeds
        },
        "val_auc_mean": _safe_mean(val_aucs),
        "val_auc_std": _safe_std(val_aucs),
        "test_pearson_mean": _safe_mean(pearsons),
        "test_pearson_std": _safe_std(pearsons),
        "test_rmse_mean": _safe_mean(rmses),
        "test_rmse_std": _safe_std(rmses),
        "test_mae_mean": _safe_mean(maes),
        "test_mae_std": _safe_std(maes),
        "w_final_mean": _safe_mean(w_finals),
        "w_final_std": _safe_std(w_finals),
        "elapsed_seconds_mean": _safe_mean(elapsed),
        "elapsed_seconds_total": sum((e for e in elapsed if e is not None), 0.0),
        "n_train": int(seeds[0].get("n_train", 0)),
        "n_val": int(seeds[0].get("n_val", 0)),
        "n_test": int(seeds[0].get("n_test", 0)),
        "epochs": DEFAULT_EPOCHS,
        "hidden": int(seeds[0].get("hidden", 0)) if seeds[0].get("hidden") else 300,
        "depth": int(seeds[0].get("depth", 0)) if seeds[0].get("depth") else 3,
        "ckpt_paths": [s["ckpt_path"] for s in seeds],
        "source_files": [s["source_file"] for s in seeds],
        "schema_per_seed": {str(s["seed"]): s["schema"] for s in seeds},
        "status": "OK" if len(seeds) >= 1 else "PARTIAL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wf-278A G5 50-epoch re-aggregator")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="G5 sweep dir, e.g. results/wf124_p3_50epoch_full_sweep_g5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output training_summary.json path",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=15,
        help="Expected total (metal, seed) pairs (default 15).",
    )
    args = parser.parse_args(argv)

    g5_dir: Path = args.input_dir
    if not g5_dir.is_dir():
        print(f"[wf278a-g5] ERROR: {g5_dir} does not exist", flush=True)
        return 2

    records, missing = collect_all_pairs(g5_dir)
    n_total = len(records)
    n_expected = int(args.expected_pairs)

    summary_per_metal: Dict[str, Dict] = {}
    for metal in FIVE_METALS:
        metal_seeds = [s for s in records if s["metal"] == metal]
        summary_per_metal[metal] = per_metal_summary(metal, metal_seeds)

    all_aucs = [r["test_auc"] for r in records]
    all_pearson = [r["test_pearson"] for r in records]
    all_rmse = [r["test_rmse"] for r in records]

    summary = {
        "wf_tag": "Wf-278A — TODO-181 G5 50-epoch re-aggregation (path+schema fix)",
        "input_dir": str(g5_dir),
        "metals": FIVE_METALS,
        "seeds": list(DEFAULT_SEEDS),
        "epochs": DEFAULT_EPOCHS,
        "backbone": {"hidden": 300, "depth": 3},
        "recipe": {
            "alpha": 0.5,
            "wasserstein_weight": 0.05,
            "use_l2_guard": True,
            "ablation_config": "mtl_baseline",
            "w2_gradient_mode": "sliced",
            "n_mc": 256,
            "batch_size": 256,
            "lr": 1e-3,
        },
        "total_pairs": n_total,
        "todo_181_g5_total_pairs": n_total,
        "expected_pairs": n_expected,
        "missing_pairs": [f"{m}/seed{s}" for (m, s) in missing],
        "per_pair_table": [
            {
                "metal": r["metal"],
                "seed": r["seed"],
                "test_auc": r["test_auc"],
                "val_auc": r["val_auc"],
                "test_pearson": r["test_pearson"],
                "test_rmse": r["test_rmse"],
                "test_mae": r["test_mae"],
                "elapsed_seconds": r["elapsed_seconds"],
                "ckpt_path": r["ckpt_path"],
                "source_file": r["source_file"],
                "schema": r["schema"],
                "status": r["status"],
            }
            for r in records
        ],
        "per_metal": summary_per_metal,
        "aggregate_5metal": {
            "n_pairs": n_total,
            "test_auc_mean": _safe_mean(all_aucs),
            "test_auc_std": _safe_std(all_aucs),
            "test_pearson_mean": _safe_mean(all_pearson),
            "test_pearson_std": _safe_std(all_pearson),
            "test_rmse_mean": _safe_mean(all_rmse),
            "test_rmse_std": _safe_std(all_rmse),
        },
        "ru_auc_target_projected": 0.86,
        "ru_auc_baseline_single_task_chemprop_v2": 0.7123,
        "ru_auc_baseline_30epoch_g4": None,  # filled in by Wf-310B cascade if G4 summary present
        "honest_framing_note": (
            "Wf-310A path+schema fix. Original launch_wf124_p3_g5.sh inline "
            "aggregation failed for two reasons: (1) glob walked the wrong "
            "directory level — missed the {metal_lower} intermediate dir; "
            "(2) inline summary only stamped paths, not MEASURED test_auc. "
            "This aggregator walks the correct nested tree and reads the G5 "
            "flat wrapper schema (per_seed_metrics.json) primarily, falling "
            "back to the G4 nested per-seed schema when needed. Per party-"
            "twenty-spirit §3.5, missing pairs are reported honestly."
        ),
        "cascade_ready": True,
        "cascade_partners": [
            "results/wf124_p3_remaining_metals_g4/training_summary.json (30ep sibling)",
        ],
    }

    # Fill ru_auc_baseline_30epoch_g4 if cascade summary is present
    g4_summary_path = g5_dir.parent / "wf124_p3_remaining_metals_g4" / "training_summary.json"
    if g4_summary_path.exists():
        try:
            g4 = json.loads(g4_summary_path.read_text())
            ru_g4 = (g4.get("per_metal") or {}).get("Ru") or {}
            if ru_g4.get("test_auc_mean") is not None:
                summary["ru_auc_baseline_30epoch_g4"] = ru_g4["test_auc_mean"]
                summary["ru_30ep_vs_50ep_delta"] = (
                    (summary_per_metal.get("Ru") or {}).get("test_auc_mean", float("nan"))
                    - ru_g4["test_auc_mean"]
                )
        except Exception as exc:
            print(f"[wf278a-g5] WARN: could not load G4 cascade: {exc!r}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[wf278a-g5] wrote {args.output}", flush=True)

    # Console summary
    print("", flush=True)
    print("=" * 70, flush=True)
    print("Wf-278A — TODO-181 G5 50-epoch re-aggregation", flush=True)
    print("=" * 70, flush=True)
    print(f"input_dir         : {g5_dir}", flush=True)
    print(f"total_pairs       : {n_total} / {n_expected}", flush=True)
    if missing:
        print(f"missing_pairs     : {[(m, s) for (m, s) in missing]}", flush=True)
    for metal in FIVE_METALS:
        s = summary_per_metal[metal]
        if s.get("status") == "NO_SEEDS":
            print(f"  {metal}: NO_SEEDS", flush=True)
            continue
        print(
            f"  {metal}: test_auc = {s['test_auc_mean']:.4f} ± {s['test_auc_std']:.4f} "
            f"(n={s['n_seeds']}/{s['n_seeds_total']})  "
            f"pearson={s['test_pearson_mean']:.3f}±{s['test_pearson_std']:.3f}  "
            f"rmse={s['test_rmse_mean']:.3f}±{s['test_rmse_std']:.3f}",
            flush=True,
        )
    agg = summary["aggregate_5metal"]
    print(
        f"5-metal aggregate : test_auc = {agg['test_auc_mean']:.4f} ± {agg['test_auc_std']:.4f} "
        f"(n={agg['n_pairs']})",
        flush=True,
    )
    if summary.get("ru_auc_baseline_30epoch_g4") is not None:
        ru_g5 = summary_per_metal["Ru"]["test_auc_mean"]
        print(
            f"Ru 30ep→50ep Δ    : {summary['ru_30ep_vs_50ep_delta']:+.4f}  "
            f"(G4 30ep baseline {summary['ru_auc_baseline_30epoch_g4']:.4f} → "
            f"G5 50ep {ru_g5:.4f})",
            flush=True,
        )
    print("", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
