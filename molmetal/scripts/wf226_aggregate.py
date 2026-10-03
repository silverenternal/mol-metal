#!/usr/bin/env python3
"""Wf-226 aggregator — compute per-metal mean ± std from the 7-pair sweep."""

import json
import statistics
from pathlib import Path

REPORTS = Path("/home/hugo/codes/mol-metal/molmetal/reports")
OUT_DIR = REPORTS / "wf226_lomo_sweep"

per_metal: dict[str, list[float]] = {}
aucs_per_pair: list[dict] = []
for metal_dir in sorted(OUT_DIR.iterdir()):
    if not metal_dir.is_dir():
        continue
    for jf in sorted(metal_dir.glob("per_seed_metrics_seed*.json")):
        d = json.loads(jf.read_text())
        if "test_auc" in d and d.get("error") is None:
            auc = float(d["test_auc"])
            per_metal.setdefault(metal_dir.name, []).append(auc)
            aucs_per_pair.append(
                {
                    "fold": metal_dir.name,
                    "seed": d["seed"],
                    "test_auc": auc,
                    "val_auc": d.get("val_auc"),
                    "elapsed_s": d.get("elapsed_seconds"),
                }
            )

per_metal_stats = {}
for m, vals in per_metal.items():
    per_metal_stats[m] = {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "n_seeds": len(vals),
        "seeds": sorted([int(j["seed"]) for j in aucs_per_pair if j["fold"] == m]),
        "values": vals,
    }

agg = {
    "metric": "metal_lomo_auc_wasserstein_wf226",
    "scope": "tier2_sbdd_ood_lomo_wasserstein_wf226",
    "description": (
        "Wf-226 — 50-epoch CPU sweep of 7 PENDING Wf-131 P3 LOMO "
        "(metal, seed) pairs with hidden=64, depth=2 backbone. "
        "MEASURED end-to-end on 2026-09-28."
    ),
    "gate_target": 0.7,
    "per_metal": per_metal_stats,
    "n_pairs_total": len(aucs_per_pair),
    "all_aucs": [j["test_auc"] for j in aucs_per_pair],
    "overall_mean": float(statistics.mean(j["test_auc"] for j in aucs_per_pair)),
    "overall_std": float(statistics.pstdev(j["test_auc"] for j in aucs_per_pair))
    if len(aucs_per_pair) > 1
    else 0.0,
    "wall_time_seconds": 1382.317697763443,
    "wall_time_minutes": 1382.317697763443 / 60.0,
    "epochs": 50,
    "hidden": 64,
    "depth": 2,
    "n_parallel": 4,
    "sweep_pairs": [
        ("Rh", 3),
        ("Rh", 4),
        ("Re", 0),
        ("Re", 1),
        ("Re", 2),
        ("Re", 3),
        ("Re", 4),
    ],
}
out_path = OUT_DIR / "wf226_per_metal_aggregate.json"
out_path.write_text(json.dumps(agg, indent=2, default=str))
print(f"Wrote {out_path}")
print(
    json.dumps(
        {
            m: {"mean": s["mean"], "std": s["std"], "n_seeds": s["n_seeds"]}
            for m, s in per_metal_stats.items()
        },
        indent=2,
    )
)
print(
    f"\nOverall: {agg['overall_mean']:.4f} ± {agg['overall_std']:.4f} "
    f"(n={agg['n_pairs_total']} pairs)"
)
print(
    f"Wall: {agg['wall_time_minutes']:.1f} min ({agg['wall_time_seconds']/3600:.2f} h)"
)
