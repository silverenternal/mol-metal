#!/usr/bin/env python3
"""Aggregate the Lambda phase-4 100pocket x 3-seed sweep.

Inputs (created by ``run_seed_sweep.sh``):
    logs/lambda_p4_sweep/seed_0/report.json
    logs/lambda_p4_sweep/seed_1/report.json
    logs/lambda_p4_sweep/seed_2/report.json

Each report.json contains:
    aggregate: { sa_mean, qed_mean, validity_rate, diversity_tanimoto,
                  synthesizability_rate, n_distinct (cell-average), ... }
    cells: [ { pocket_id, seed, n_distinct, validity_rate, ... }, ... ]

For every metric we report:
    per-seed value, mean ± std across seeds, and the spec target.

Outputs:
    logs/lambda_p4_sweep/aggregate.csv
    logs/lambda_p4_sweep/aggregate.json
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent
SEEDS = ("seed_0", "seed_1", "seed_2")

# Metric name -> (extractor, paper target, comparator)
#   extractor: pulls the scalar from one seed's report.json aggregate
#   target:    the paper-grade target (None = none)
#   comparator: '>=' / '<=' / '==' / 'approx' for hit/miss colouring
METRICS: List[Dict[str, Any]] = [
    {
        "name": "n_distinct",
        "extractor": lambda rep: _cell_mean(rep, "n_distinct"),
        "target": 3,
        "comparator": ">=",
    },
    {
        "name": "sa_mean",
        "extractor": lambda rep: float(rep.get("aggregate", {}).get("sa_mean", float("nan"))),
        "target": 3.5,
        "comparator": "<",
    },
    {
        "name": "qed_mean",
        "extractor": lambda rep: float(rep.get("aggregate", {}).get("qed_mean", float("nan"))),
        "target": 0.55,
        "comparator": ">=",
    },
    {
        "name": "div_tanimoto",
        "extractor": lambda rep: float(rep.get("aggregate", {}).get("diversity_tanimoto", float("nan"))),
        "target": 0.6,
        "comparator": ">=",
    },
    {
        "name": "synth_rate",
        "extractor": lambda rep: float(rep.get("aggregate", {}).get("synthesizability_rate", float("nan"))),
        "target": 0.95,
        "comparator": ">=",
    },
    {
        "name": "validity_rate",
        "extractor": lambda rep: float(rep.get("aggregate", {}).get("validity_rate", float("nan"))),
        "target": 1.0,
        "comparator": "==",
    },
]


def _cell_mean(rep: Dict[str, Any], field: str) -> float:
    cells = rep.get("cells") or []
    vals: List[float] = []
    for c in cells:
        v = c.get(field)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            vals.append(float(v))
    if not vals:
        return float("nan")
    return float(statistics.fmean(vals))


def _safe_stat(values: List[float], fn) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return float(fn(finite))


def _hit(value: float, target: float, comparator: str) -> str:
    if math.isnan(value):
        return "n/a"
    if comparator == ">=":
        return "PASS" if value >= target else "MISS"
    if comparator == "<":
        return "PASS" if value < target else "MISS"
    if comparator == "==":
        return "PASS" if abs(value - target) < 1e-9 else "MISS"
    return "n/a"


def main() -> None:
    seed_reports: Dict[str, Dict[str, Any]] = {}
    seed_values: Dict[str, Dict[str, float]] = {}
    seed_meta: Dict[str, Dict[str, Any]] = {}
    for s in SEEDS:
        path = ROOT / s / "report.json"
        if not path.exists():
            print(f"[aggregate] missing {path}")
            continue
        rep = json.loads(path.read_text())
        seed_reports[s] = rep
        seed_meta[s] = {
            "path": str(path),
            "n_cells": len(rep.get("cells") or []),
            "config": rep.get("config", {}),
            "elapsed_s_total": rep.get("elapsed_s_total"),
        }
        seed_values[s] = {m["name"]: m["extractor"](rep) for m in METRICS}

    if not seed_reports:
        print("[aggregate] no seed reports found under logs/lambda_p4_sweep/")
        raise SystemExit(1)

    # Compute mean ± std across seeds for every metric.
    rows: List[Dict[str, Any]] = []
    for m in METRICS:
        per_seed = [seed_values[s].get(m["name"], float("nan")) for s in SEEDS if s in seed_values]
        mean = _safe_stat(per_seed, statistics.fmean)
        std = (
            _safe_stat(per_seed, statistics.stdev)
            if sum(1 for v in per_seed if not math.isnan(v)) >= 2
            else 0.0
        )
        rows.append({
            "metric": m["name"],
            "target": m["target"],
            "comparator": m["comparator"],
            "mean": mean,
            "std": std,
            "n_seeds": int(sum(1 for v in per_seed if not math.isnan(v))),
            "hit": _hit(mean, m["target"], m["comparator"]),
            **{s: seed_values.get(s, {}).get(m["name"], float("nan")) for s in SEEDS},
        })

    # ---- CSV ------------------------------------------------------------
    csv_path = ROOT / "aggregate.csv"
    fieldnames = ["metric", "target", "comparator", "mean", "std", "n_seeds", "hit"] + list(SEEDS)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    print(f"[aggregate] wrote {csv_path}")

    # ---- JSON -----------------------------------------------------------
    json_path = ROOT / "aggregate.json"
    payload = {
        "schema_version": 1,
        "tool": "logs/lambda_p4_sweep/aggregate.py",
        "spec": {
            "n_pockets": (seed_meta.get(SEEDS[0], {}).get("config", {}) or {}).get("n_pockets"),
            "seeds": (seed_meta.get(SEEDS[0], {}).get("config", {}) or {}).get("seeds"),
            "n_simulations": (seed_meta.get(SEEDS[0], {}).get("config", {}) or {}).get("n_simulations"),
            "n_top_k": (seed_meta.get(SEEDS[0], {}).get("config", {}) or {}).get("n_top_k"),
        },
        "n_seeds": len(seed_reports),
        "seeds": seed_meta,
        "metrics": [
            {
                "metric": r["metric"],
                "target": r["target"],
                "comparator": r["comparator"],
                "mean": r["mean"],
                "std": r["std"],
                "n_seeds": r["n_seeds"],
                "hit": r["hit"],
                "per_seed": {s: seed_values.get(s, {}).get(r["metric"], float("nan")) for s in SEEDS},
            }
            for r in rows
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[aggregate] wrote {json_path}")

    # ---- Console summary -----------------------------------------------
    print("\nmetric             target   comparator  mean ± std       hit")
    print("-" * 72)
    for r in rows:
        mean = r["mean"]
        std = r["std"]
        mean_s = f"{mean:.4f}" if not math.isnan(mean) else "n/a"
        std_s = f"{std:.4f}" if not (isinstance(std, float) and math.isnan(std)) else "n/a"
        target_s = f"{r['target']:.4f}" if isinstance(r["target"], float) else str(r["target"])
        print(f"{r['metric']:<18} {target_s:>8} {r['comparator']:^11} {mean_s:>10} ± {std_s:<10} {r['hit']}")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    if v is None:
        return ""
    return str(v)


if __name__ == "__main__":
    main()
