#!/usr/bin/env python3
"""Wf-318B: compute per-metal cascade delta G4 30ep → G5 50ep for all 5 metals.

==============================================================================

Reads:
  * results/wf124_p3_remaining_metals_g4/training_summary.json (G4 30ep baseline)
  * metrics/by_metric/wf181_g5_50epoch_full_sweep.json        (G5 50ep)
  * results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/training_summary.json
    (Ir seed=0 retry at 0.6888 — incorporated for n=3 effective coverage)

Writes:
  * metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json

Per-metal cascade delta per party-twenty-spirit §3.5 honest framing:
  - delta       = G5_mean - G4_mean
  - delta_pct   = delta / G4_mean × 100
  - std_pooled  = sqrt((G4_std² × n_g4 + G5_std² × n_g5) / (n_g4 + n_g5))
  - t_stat      = delta / std_pooled × sqrt(n_g4 × n_g5 / (n_g4 + n_g5))

CPU-only, deterministic. No GPU launch.

Usage::

    python molmetal/scripts/compute_cascade_delta_5metal.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
G4_PATH = REPO_ROOT / "results" / "wf124_p3_remaining_metals_g4" / "training_summary.json"
G5_PATH = METRICS_DIR / "wf181_g5_50epoch_full_sweep.json"
IR_RETRY_PATH = (
    REPO_ROOT
    / "results"
    / "wf124_p3_50epoch_full_sweep_g5"
    / "Ir"
    / "seed0"
    / "training_summary.json"
)
OUT_PATH = METRICS_DIR / "wf181_g5_50epoch_cascade_delta_5metal.json"


def _pooled_std(std1: float, n1: int, std2: float, n2: int) -> float:
    return math.sqrt((std1 ** 2 * n1 + std2 ** 2 * n2) / (n1 + n2))


def _t_stat(delta: float, std_pooled: float, n1: int, n2: int) -> float:
    if std_pooled == 0:
        return float("inf") if delta != 0 else 0.0
    return delta / std_pooled * math.sqrt(n1 * n2 / (n1 + n2))


def _g5_std_from_per_seed(per_seed: dict[int, float]) -> tuple[float, int]:
    aucs = list(per_seed.values())
    n = len(aucs)
    if n == 0:
        return 0.0, 0
    mean = sum(aucs) / n
    if n == 1:
        return 0.0, n
    var = sum((v - mean) ** 2 for v in aucs) / (n - 1)
    return math.sqrt(var), n


def main() -> None:
    g4 = json.loads(G4_PATH.read_text())
    g5 = json.loads(G5_PATH.read_text())
    ir_seed0_retry = json.loads(IR_RETRY_PATH.read_text())

    # Incorporate Ir seed=0 retry
    g5_per_seed_per_metal: dict[str, dict[int, float]] = {}
    for metal in g5["metals"]:
        seeds = {}
        for seed_str, entry in g5["per_metal_seed_auc"].get(metal, {}).items():
            seeds[int(seed_str)] = entry["auc"]
        if metal == "Ir":
            seeds[0] = ir_seed0_retry["per_metal"]["Ir"]["test_auc_mean"]
        g5_per_seed_per_metal[metal] = seeds

    results = {}
    for metal in g5["metals"]:
        g4_pm = g4["per_metal"][metal]
        g4_mean = g4_pm["test_auc_mean"]
        g4_std = g4_pm["test_auc_std"]
        n_g4 = g4_pm["n_seeds"]

        g5_seeds = g5_per_seed_per_metal[metal]
        if not g5_seeds:
            continue
        g5_mean = sum(g5_seeds.values()) / len(g5_seeds)
        g5_std, n_g5 = _g5_std_from_per_seed(g5_seeds)

        g4_seeds_set = {int(k) for k in g4_pm["test_auc_per_seed"].keys()}
        overlap_seeds = sorted(g4_seeds_set & set(g5_seeds.keys()))
        n_overlap = len(overlap_seeds)

        delta = g5_mean - g4_mean
        delta_pct = delta / g4_mean * 100 if g4_mean else 0.0
        std_pool = _pooled_std(g4_std, n_g4, g5_std, n_g5)
        t = _t_stat(delta, std_pool, n_g4, n_g5)

        if n_overlap == 3:
            hf = "FULL 3-seed overlap; both G4 and G5 measured on same 3 seeds"
        elif n_overlap == 2 and metal == "Ru":
            hf = (
                "n=2-seed overlap (seeds 0+1); Ru/seed=2 still in-flight "
                "in G5 sweep (epoch 27/50 as of 2026-09-29)"
            )
        else:
            hf = f"n={n_overlap}-seed overlap only"

        results[metal] = {
            "metal": metal,
            "g4_baseline": g4_mean,
            "g4_baseline_std": g4_std,
            "n_seeds_g4": n_g4,
            "g4_per_seed": {int(k): v for k, v in g4_pm["test_auc_per_seed"].items()},
            "g5_50ep": g5_mean,
            "g5_50ep_std": g5_std,
            "n_seeds_g5": n_g5,
            "g5_per_seed": g5_seeds,
            "delta": delta,
            "delta_pct": delta_pct,
            "std_pooled": std_pool,
            "t_stat": t,
            "n_seeds_overlap": n_overlap,
            "overlap_seeds": overlap_seeds,
            "honest_framing": hf,
        }

    output = {
        "metric": "wf181_g5_50epoch_cascade_delta_5metal",
        "description": (
            "Wf-318B: Wf-310A cascade delta 30ep→50ep extended to all 5 metals "
            "(Ru/Ir/Os/Rh/Re). G4 30ep per-metal baseline from "
            "results/wf124_p3_remaining_metals_g4/training_summary.json "
            "(MEASURED 5 metal × 3 seed); G5 50ep per-metal from "
            "metrics/by_metric/wf181_g5_50epoch_full_sweep.json "
            "(MEASURED 13/15 pairs + Ir/seed=0 retry at 0.6888 incorporated = "
            "14/15; Ru/seed=2 still in-flight). Per-metal cascade delta "
            "(G5 - G4) reported with delta_pct, std_pooled, t_stat "
            "(unpaired two-sample t-test formula). Honest framing per "
            "party-twenty-spirit §3.5: Ru n=2-seed overlap (seed=2 in-flight), "
            "Ir n=3-seed effective (post-retry), Os/Rh/Re n=3 full coverage."
        ),
        "unit": "ROC-AUC delta (0-1, higher better)",
        "scope": (
            "tier1_precious_metal_anticancer / MetalCytoToxDB 5-metal / "
            "cascade 30ep→50ep / D-MPNN multi-task Wasserstein pIC50"
        ),
        "source": "molmetal/scripts/compute_cascade_delta_5metal.py (Wf-318B)",
        "split_protocol": "scaffold (Bemis-Murcko) per Wf-124 P3 baseline + TODO-181 G5 50ep",
        "krasnov_protocol_aligned": True,
        "metals": ["Ru", "Ir", "Os", "Rh", "Re"],
        "per_metal_table": results,
        "summary": {
            "n_metal_deltas_computed": len(results),
            "delta_positive_count": sum(1 for r in results.values() if r["delta"] > 0),
            "delta_negative_count": sum(1 for r in results.values() if r["delta"] < 0),
            "delta_zero_count": sum(1 for r in results.values() if r["delta"] == 0),
            "aggregate_5metal_delta": sum(r["delta"] for r in results.values()) / len(results),
            "max_delta_metal": max(results.items(), key=lambda kv: kv[1]["delta"])[0],
            "min_delta_metal": min(results.items(), key=lambda kv: kv[1]["delta"])[0],
        },
        "g4_source_path": "results/wf124_p3_remaining_metals_g4/training_summary.json",
        "g5_source_path": "metrics/by_metric/wf181_g5_50epoch_full_sweep.json",
        "ir_seed0_retry_source_path": (
            "results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/training_summary.json"
        ),
        "wf_tag": "Wf-318B — TODO-181 cascade delta 30ep→50ep extended to all 5 metals",
        "ran_at_utc": "2026-09-29T01:45:23+00:00",
        "consolidation_date": "2026-09-29",
        "consolidation_workflow": "Wf-318B",
        "followup_note": (
            "Ru/seed=2 still in-flight (epoch 27/50 as of 2026-09-29); "
            "re-run cascade delta after Ru/seed=2 lands to update Ru row "
            "from n=2 to n=3."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[Wf-318B] wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    print(f"[Wf-318B] n_metal_deltas_computed = {len(results)}")
    for metal, r in results.items():
        print(
            f"  {metal}: Δ={r['delta']:+.4f} ({r['delta_pct']:+.2f}%) "
            f"t_stat={r['t_stat']:.2f} n_overlap={r['n_seeds_overlap']}"
        )


if __name__ == "__main__":
    main()
