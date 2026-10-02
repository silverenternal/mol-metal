#!/usr/bin/env python3
"""Wf-379-A1 — Update metrics/by_metric/metal_lomo_auc.json with 7-run MEASURED results.

Reads per-seed metrics JSON files from
molmetal/reports/wf131_p3_deferred_runs/{Rh,Re}/seed{N}/per_seed_metrics_seed{N}.json
and appends a new wf379a1_7runs_update block to metal_lomo_auc.json with:
- per-metal per-seed test_auc values
- aggregate mean ± std per metal
- 95% confidence interval per metal
- Krasnov 0.86 gate check per metal
- Honest framing per party-twenty-spirit §3.5

Usage:
  python3 molmetal/scripts/wf379a1_update_m1_auc.py
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
RUNS_DIR = ROOT / "molmetal/reports/wf131_p3_deferred_runs"
METRICS_JSON = ROOT / "metrics/by_metric/metal_lomo_auc.json"

KRASNOV_GATE = 0.86

# 7 DEFERRED (metal, seed) pairs from Wf-379-A1
DEFERRED_PAIRS = [
    ("Rh", 3),
    ("Rh", 4),
    ("Re", 0),
    ("Re", 1),
    ("Re", 2),
    ("Re", 3),
    ("Re", 4),
]


def main() -> int:
    if not RUNS_DIR.exists():
        print(f"FATAL: {RUNS_DIR} does not exist", file=sys.stderr)
        return 1

    per_seed: dict = {"Rh": {}, "Re": {}}
    missing = []

    for metal, seed in DEFERRED_PAIRS:
        f = RUNS_DIR / metal / f"seed{seed}" / f"per_seed_metrics_seed{seed}.json"
        if not f.exists():
            missing.append(str(f))
            continue
        with f.open() as fh:
            payload = json.load(fh)
        test_auc = payload.get("test_auc")
        val_auc = payload.get("val_auc")
        w2_final = payload.get("w2_final")
        per_seed[metal][seed] = {
            "test_auc": test_auc,
            "val_auc": val_auc,
            "w2_final": w2_final,
            "source": str(f),
        }

    if missing:
        print(f"WARN: missing per_seed_metrics files: {missing}", file=sys.stderr)

    # Aggregate
    per_metal_agg = {}
    for metal in ("Rh", "Re"):
        aucs = [
            v["test_auc"]
            for v in per_seed[metal].values()
            if v.get("test_auc") is not None
        ]
        if not aucs:
            per_metal_agg[metal] = {
                "n_seeds_used": 0,
                "mean_test_auc": None,
                "std_test_auc": None,
                "ci95_low": None,
                "ci95_high": None,
                "krasnov_gate_pass": False,
                "krasnov_gate_margin": None,
                "verdict": "NOT_MEASURED",
            }
            continue
        n = len(aucs)
        mean = sum(aucs) / n
        var = sum((v - mean) ** 2 for v in aucs) / n  # population std
        std = var ** 0.5
        # 95% CI using z = 1.96 (small sample approx)
        ci_half = 1.96 * std / (n ** 0.5) if n > 0 else 0.0
        gate_pass = mean >= KRASNOV_GATE
        gate_margin = mean - KRASNOV_GATE
        verdict = (
            "MEASURED_PASS_GATE" if gate_pass
            else "MEASURED_FAIL_GATE" if n >= 3
            else "HONEST_NEGATIVE_1SEED" if n == 1
            else "PARTIAL_2SEED"
        )
        per_metal_agg[metal] = {
            "n_seeds_used": n,
            "per_seed_aucs": per_seed[metal],
            "mean_test_auc": round(mean, 4),
            "std_test_auc": round(std, 4),
            "ci95_low": round(mean - ci_half, 4),
            "ci95_high": round(mean + ci_half, 4),
            "krasnov_gate_target": KRASNOV_GATE,
            "krasnov_gate_pass": gate_pass,
            "krasnov_gate_margin": round(gate_margin, 4),
            "verdict": verdict,
        }

    new_block = {
        "date_utc": "2026-09-30T00:55:00+00:00",
        "wf_tag": "Wf-379-A1",
        "wrapper": "scripts/launch_wf131_p3_deferred_runs.sh",
        "data_source": RUNS_DIR,
        "actual_data_path": str(RUNS_DIR),
        "actual_data_status": "COMPLETE" if not missing else f"PARTIAL ({len(missing)} missing)",
        "n_total_runs": len(DEFERRED_PAIRS),
        "n_completed_runs": len(DEFERRED_PAIRS) - len(missing),
        "missing_runs": missing,
        "gpu": "cuda:1 (RTX 5090 GPU-1, 32 GB)",
        "epochs": 50,
        "krasnov_gate_target": KRASNOV_GATE,
        "per_metal_aggregate": per_metal_agg,
        "per_seed_aucs_combined": per_seed,
        "honest_framing": (
            "BUGGY Wf-125 P1 cohort per Wf-354-A1 §5 + Wf-355-B1 §1.3; "
            "Re fold DEFERRED 0/5 prior; Rh fold 3/5 prior (mean 0.6874 ± 0.3584). "
            "Wf-379-A1 resolves the 7 DEFERRED runs (Rh seeds 3+4 + Re seeds 0-4) "
            "to establish the per-metal AUC envelope on the BUGGY cohort."
        ),
        "status": (
            "COMPLETE" if not missing
            else f"PARTIAL ({len(missing)} of {len(DEFERRED_PAIRS)} missing)"
        ),
    }

    # Read existing metrics
    with METRICS_JSON.open() as fh:
        data = json.load(fh)
    data["wf379a1_7runs_update"] = new_block
    METRICS_JSON.write_text(json.dumps(data, indent=2))
    print(f"[OK] wrote {METRICS_JSON}")
    for metal, agg in per_metal_agg.items():
        if agg.get("mean_test_auc") is not None:
            print(
                f"     {metal}: mean={agg['mean_test_auc']} ± {agg['std_test_auc']} "
                f"95%CI=[{agg['ci95_low']}, {agg['ci95_high']}] "
                f"verdict={agg['verdict']} "
                f"krasnov_margin={agg['krasnov_gate_margin']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())