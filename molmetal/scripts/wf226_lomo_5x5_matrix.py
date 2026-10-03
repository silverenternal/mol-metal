#!/usr/bin/env python3
"""Build the 5-metal × 5-seed LOMO test_auc matrix from Wf-131 P3 + Wf-193 + Wf-226."""
import json
import statistics
from pathlib import Path

REPORTS = Path("/home/hugo/codes/mol-metal/molmetal/reports")

# Source directories
SOURCES = [
    ("Wf-131 P3", REPORTS / "wf131_p3_lomo_full"),
    ("Wf-193 ext", REPORTS / "wf193_wf131_p3_extended"),
    ("Wf-226 (this)", REPORTS / "wf226_lomo_sweep"),
]

matrix: dict[str, dict[int, dict]] = {}
sources_used: dict[tuple[str, int], str] = {}
for src_name, src_dir in SOURCES:
    if not src_dir.exists():
        continue
    for metal_dir in src_dir.iterdir():
        if not metal_dir.is_dir():
            continue
        m = metal_dir.name
        matrix.setdefault(m, {})
        for jf in metal_dir.glob("per_seed_metrics_seed*.json"):
            d = json.loads(jf.read_text())
            if "test_auc" in d and d.get("error") is None:
                seed = int(d["seed"])
                # If duplicate, prefer latest (Wf-226 > Wf-193 > Wf-131 P3)
                if (m, seed) not in sources_used or sources_used[(m, seed)] == "Wf-131 P3":
                    matrix[m][seed] = {
                        "test_auc": float(d["test_auc"]),
                        "val_auc": float(d.get("val_auc", float("nan"))),
                        "elapsed_s": float(d.get("elapsed_seconds", 0.0)),
                        "source": src_name,
                    }
                    sources_used[(m, seed)] = src_name

per_metal_stats = {}
for m, seeds in matrix.items():
    aucs = [s["test_auc"] for s in seeds.values()]
    per_metal_stats[m] = {
        "n_seeds": len(aucs),
        "seeds": sorted(seeds.keys()),
        "mean": float(statistics.mean(aucs)) if aucs else float("nan"),
        "std": float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0,
        "per_seed": {
            str(s): {"test_auc": seeds[s]["test_auc"], "source": seeds[s]["source"]}
            for s in sorted(seeds.keys())
        },
    }

# 5-metal aggregate
all_aucs = [s["test_auc"] for seeds in matrix.values() for s in seeds.values()]
agg = {
    "n_pairs_total": len(all_aucs),
    "all_metals_seeds": [(m, s) for m in matrix for s in matrix[m]],
    "overall_mean": float(statistics.mean(all_aucs)) if all_aucs else float("nan"),
    "overall_std": float(statistics.pstdev(all_aucs)) if len(all_aucs) > 1 else 0.0,
    "per_metal": per_metal_stats,
}

print("=" * 60)
print("5-metal × 5-seed LOMO test_auc matrix (test_AUC)")
print("=" * 60)
header = "Metal  | " + " | ".join(f"s{s}" for s in range(5)) + " | n   | mean ± std"
print(header)
print("-" * len(header))
for m in ["Ru", "Ir", "Os", "Rh", "Re"]:
    if m not in matrix:
        print(f"{m:6s} | (no data)")
        continue
    row = [f"{matrix[m].get(s, {}).get('test_auc', float('nan')):.4f}"
           if s in matrix[m] else "  ---  " for s in range(5)]
    stats = per_metal_stats[m]
    print(f"{m:6s} | " + " | ".join(f"{r:6s}" for r in row)
          + f" | {stats['n_seeds']}/5 | {stats['mean']:.4f} ± {stats['std']:.4f}")

print()
print(f"Overall: {agg['overall_mean']:.4f} ± {agg['overall_std']:.4f} "
      f"(n={agg['n_pairs_total']}/25)")
print(f"Gate target ≥ 0.70: "
      f"{'PASS' if agg['overall_mean'] >= 0.7 else 'BELOW GATE'}")
print(f"Overall target ≥ 0.60 (Wf-125 P2): "
      f"{'HIT' if agg['overall_mean'] >= 0.60 else 'MISS'}")
rh_target = 0.50
rh_mean = per_metal_stats.get("Rh", {}).get("mean", float("nan"))
print(f"Rh target ≥ {rh_target}: "
      f"{'HIT' if rh_mean >= rh_target else 'MISS'} (Rh mean = {rh_mean:.4f})")

out_path = REPORTS / "wf226_lomo_sweep" / "wf226_full_5x5_matrix.json"
out_path.write_text(json.dumps(agg, indent=2, default=str))
print(f"\nWrote {out_path}")
