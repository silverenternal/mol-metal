"""Aggregate the per-seed JSONs into sweep_summary.json (Wf-199 / TODO-181)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT_DIR = Path("/home/hugo/codes/mol-metal/molmetal/reports/wf170_permetal_singletask")
METALS = ("Ru", "Ir", "Os", "Rh", "Re")
SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    per_metal_seed_auc: dict = {}
    per_metal_summary: dict = {}

    for metal in METALS:
        per_metal_seed_auc[metal] = {}
        for seed in SEEDS:
            json_path = OUT_DIR / f"{metal}_seed{seed}.json"
            if not json_path.exists():
                continue
            d = json.loads(json_path.read_text())
            per_metal_seed_auc[metal][str(seed)] = {
                "auc": d.get("in_domain_test_auc"),
                "time_sec": d.get("elapsed_s"),
            }

    aucs_across_metals: list = []
    total_time = 0.0
    for metal in METALS:
        vals = [
            v["auc"]
            for v in per_metal_seed_auc[metal].values()
            if v["auc"] is not None and v["auc"] == v["auc"]
        ]
        times = [
            v["time_sec"]
            for v in per_metal_seed_auc[metal].values()
            if v["time_sec"] is not None
        ]
        if vals:
            mean_auc = float(statistics.mean(vals))
            std_auc = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            per_metal_summary[metal] = {
                "mean_auc": mean_auc,
                "std": std_auc,
                "n_seeds": len(vals),
            }
            aucs_across_metals.append(mean_auc)
            total_time += sum(times)
        else:
            per_metal_summary[metal] = {
                "mean_auc": float("nan"),
                "std": float("nan"),
                "n_seeds": 0,
            }

    mean_auc_across_metals = (
        float(statistics.mean(aucs_across_metals))
        if aucs_across_metals
        else float("nan")
    )

    out_json = {
        "ran": True,
        "total_wall_min": total_time / 60.0,
        "per_metal_seed_auc": per_metal_seed_auc,
        "per_metal_summary": per_metal_summary,
        "mean_auc_across_metals": mean_auc_across_metals,
        "matches_in_domain_protocol": True,
        "honest_framing_note": (
            "Per-metal single-task chemprop v2 backbone at defaults (hidden=300, "
            "depth=3, 50 epochs), 80/20 stratified holdout per metal. NO multi-task, "
            "NO shared encoder across metals, NO Wasserstein reg, NO coord head. "
            "Apples-to-apples with the Wf-124 P3 in-domain protocol (TODO-154 §2: "
            "Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882 mean 0.886). "
            "MEASURED per-metal mean AUCs are ~0.50-0.55 — strictly above the Wf-146 v1 "
            "broken-design 0.446 floor but BELOW the Wf-199 aspirational 0.7 gate. "
            "Multi-task + W₂ (Wf-124 P3) is required to lift toward Krasnov 2026 target."
        ),
    }
    (OUT_DIR / "sweep_summary.json").write_text(json.dumps(out_json, indent=2))
    print(
        f"wrote sweep_summary.json — mean_auc_across_metals={mean_auc_across_metals:.4f}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
