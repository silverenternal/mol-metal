"""Wf-310D PART 1: DrugOOD DANN full GPU aggregation template.

==============================================================================
Wf-310D (Wf-301A full run aggregation, follow-up of Wf-306A Phase-A):

Inputs (read-only, written by Wf-301A full 5-seed × 50-epoch GPU run launched
via scripts/launch_wf306a_drugood_vina_gpu.sh --phase A):

* molmetal/reports/wf301a_drugood_dann/seed<N>.json   (N in {0,1,2,3,4})
* molmetal/reports/wf301a_drugood_dann/aggregate.json (smoke checkpoint)

Outputs (this script writes):

* molmetal/reports/wf310d_drugood_full/aggregate.json  (paper-grade)
* molmetal/reports/wf310d_drugood_full/claim_cards.md  (mechanism claims)
* molmetal/reports/wf310d_drugood_full/aggregation_plan.md (template doc)

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* Wf-185 v2 baseline = 0.6707 ± 0.0042 (chemprop v2, lbap_refined_ic50
  scaffold split).  Different substrate from DrugOOD.
* Wf-301A smoke (1 epoch, 3 seeds) test_auc = 0.8024 mean (CPU).
* Field-avg target per TODO-130 G5 honest reporting: 0.55 - 0.65.
* Full run delta_vs_wf185 is across DIFFERENT subsets; cross-substrate
  comparison is honest-negative flagged per Wf-301A honest-negative.

Usage
-----
  python molmetal/scripts/wf310d_drugood_dann_full_aggregate.py \\
      --per-seed-dir molmetal/reports/wf301a_drugood_dann \\
      --smoke-aggregate molmetal/reports/wf301a_drugood_dann/aggregate.json \\
      --out-dir molmetal/reports/wf310d_drugood_full

Per-seed JSON shape (Wf-301A output)
------------------------------------
  {
    "seed": int,
    "train_auc": float,
    "val_auc": float,
    "test_auc": float,
    "elapsed_s": float,
    "n_train": int, "n_val": int, "n_test": int,
    "use_dann": bool, "n_domains": int
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Wf-185 chemprop v2 baseline (lbap_refined_ic50_scaffold split).
WF185_BASELINE_MEAN: float = 0.6707
WF185_BASELINE_STD: float = 0.0042
WF185_BASELINE_SEED_SLACK: float = 0.02
# Field-avg target per TODO-130 G5 honest reporting.
FIELD_AVG_TARGET_LO: float = 0.55
FIELD_AVG_TARGET_HI: float = 0.65
# Wf-301A smoke CPU checkpoint (1 epoch, 3 seeds; cross-comparison axis).
WF301A_SMOKE_MEAN: float = 0.8024
WF301A_SMOKE_N_SEEDS: int = 3


def _log(msg: str) -> None:
    print(f"[wf310d-agg] {msg}", flush=True)


def _load_per_seed(per_seed_dir: Path, n_seeds: int = 5) -> List[Dict[str, float]]:
    """Load seed0.json..seed4.json (skip missing)."""
    rows: List[Dict[str, float]] = []
    for seed in range(n_seeds):
        p = per_seed_dir / f"seed{seed}.json"
        if not p.exists():
            _log(f"WARN: missing {p} — skipping")
            continue
        rows.append(json.loads(p.read_text()))
    return rows


def _verdict(test_mean: float, test_std: float, n_seeds: int) -> str:
    """Pass-gate verdict per Wf-301A pass-gate convention."""
    passes_wf185 = bool(test_mean >= WF185_BASELINE_MEAN)
    passes_field_avg = bool(test_mean >= FIELD_AVG_TARGET_LO)
    passes_wf185_minus_slack = bool(
        test_mean >= WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK
    )

    if passes_wf185:
        return (
            f"PASS_WF185 -- DANN full {test_mean:.4f} >= Wf-185 baseline "
            f"{WF185_BASELINE_MEAN:.4f}; the Wf-253-D 0.4544 collapse is "
            f"addressable by ECFP4 + scaffold-bin DANN ({n_seeds}-seed full)."
        )
    if passes_wf185_minus_slack:
        return (
            f"PASS_FIELD_AVG -- DANN full {test_mean:.4f} >= field-avg lo "
            f"{FIELD_AVG_TARGET_LO:.2f} and within seed-noise slack of "
            f"Wf-185 v2 ({WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK:.4f}); "
            f"partial recovery ({n_seeds}-seed full)."
        )
    if passes_field_avg:
        return (
            f"PASS_FIELD_AVG_ONLY -- DANN full {test_mean:.4f} in "
            f"field-avg band {FIELD_AVG_TARGET_LO}-{FIELD_AVG_TARGET_HI}; "
            f"Wf-185 not matched but field-avg band cleared ({n_seeds}-seed full)."
        )
    return (
        f"HONEST_NEGATIVE -- DANN full {test_mean:.4f} < field-avg lo "
        f"{FIELD_AVG_TARGET_LO:.2f}; scaffold-bin DANN does NOT rescue the "
        f"Wf-253-D 0.4544 collapse ({n_seeds}-seed full)."
    )


def _claim_cards(
    test_mean: float,
    test_std: float,
    delta_wf185: float,
    delta_smoke: float,
    n_seeds: int,
    wall_min: float,
) -> List[str]:
    """Emit paper-grade claim cards per Wf-301A convention."""
    cards = [
        f"# Wf-310D DrugOOD DANN full aggregation -- claim cards ({n_seeds}-seed × 50-epoch)",
        "",
        "## Card 1 -- per-seed ood_test_auc",
        "",
        "| seed | test_auc | elapsed_s | n_test |",
        "|------|----------|-----------|--------|",
    ]
    return cards


def _format_claim_cards(
    per_seed: List[Dict[str, float]],
    test_mean: float,
    test_std: float,
    delta_wf185: float,
    delta_smoke: float,
    n_seeds: int,
    wall_min: float,
) -> str:
    """Format per-seed table + MEASURED-vs-PROJECTED claim cards."""
    lines = [
        "# Wf-310D DrugOOD DANN full aggregation -- claim cards",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"n_seeds: {n_seeds}  test_auc_mean: {test_mean:.4f}  "
        f"test_auc_std: {test_std:.4f}",
        f"wall_min: {wall_min:.2f}  delta_vs_wf185: {delta_wf185:+.4f}",
        f"delta_vs_wf301a_smoke: {delta_smoke:+.4f}",
        "",
        "## Card 1 -- per-seed ood_test_auc (MEASURED)",
        "",
        "| seed | test_auc | train_auc | val_auc | elapsed_s | n_test |",
        "|------|----------|-----------|---------|-----------|--------|",
    ]
    for r in per_seed:
        lines.append(
            f"| {r.get('seed', '?')} | {r.get('test_auc', float('nan')):.4f} | "
            f"{r.get('train_auc', float('nan')):.4f} | "
            f"{r.get('val_auc', float('nan')):.4f} | "
            f"{r.get('elapsed_s', 0.0):.1f} | "
            f"{r.get('n_test', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Card 2 -- pass-gate verdict (MEASURED)",
            "",
            f"- Wf-185 v2 baseline: {WF185_BASELINE_MEAN:.4f} ± {WF185_BASELINE_STD:.4f}",
            f"- Wf-301A smoke (CPU, 1-epoch, 3-seed): {WF301A_SMOKE_MEAN:.4f}",
            f"- Field-avg target band: {FIELD_AVG_TARGET_LO:.2f} - {FIELD_AVG_TARGET_HI:.2f}",
            f"- Full DANN {n_seeds}-seed: {test_mean:.4f} ± {test_std:.4f}",
            f"- delta_vs_wf185: {delta_wf185:+.4f}",
            f"- delta_vs_wf301a_smoke: {delta_smoke:+.4f}  (full vs smoke -- "
            f"both ECFP4 + DANN, different epoch count + different seed count)",
            "",
            "## Card 3 -- honest-negative flags (per Wf-301A honest-negative)",
            "",
            "- Wf-185 Δ is across DIFFERENT subsets (DrugOOD scaffold vs "
            "  lbap_refined_ic50 scaffold); cross-substrate comparison is "
            "  directional only.",
            "- Full run vs smoke Δ mixes epoch-budget effect with seed-noise.",
            "- DANN scaffold-binning may not generalise beyond top-15 + OOV "
            "  scaffold frequency (per Wf-301A honest-negative §HONEST_NEGATIVE).",
            "",
            "## Card 4 -- wall + GPU cost (PROJECTED)",
            "",
            f"- Wall: {wall_min:.1f} min ({wall_min/60:.2f} hr) -- MEASURED.",
            "- GPU: cuda:1 (RTX 5090) per Wf-306A launch_wf306a_drugood_vina_gpu.sh.",
            "- ~3-4 hr GPU-1 PROJECTED per Wf-306A Phase-A budget.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--per-seed-dir",
        default="molmetal/reports/wf301a_drugood_dann",
        help="Directory containing seed<N>.json from Wf-301A full run.",
    )
    p.add_argument(
        "--smoke-aggregate",
        default="molmetal/reports/wf301a_drugood_dann/aggregate.json",
        help="Wf-301A smoke aggregate.json (for Δ smoke tracking).",
    )
    p.add_argument("--out-dir", default="molmetal/reports/wf310d_drugood_full")
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args(argv)

    per_seed_dir = Path(args.per_seed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = _load_per_seed(per_seed_dir, n_seeds=args.n_seeds)
    if not per_seed:
        _log(f"ERROR: no per-seed JSONs found in {per_seed_dir}")
        return 1
    n_seeds = len(per_seed)

    test_aucs = [
        r["test_auc"]
        for r in per_seed
        if "test_auc" in r and not np.isnan(r["test_auc"])
    ]
    val_aucs = [
        r["val_auc"] for r in per_seed if "val_auc" in r and not np.isnan(r["val_auc"])
    ]
    train_aucs = [
        r["train_auc"]
        for r in per_seed
        if "train_auc" in r and not np.isnan(r["train_auc"])
    ]
    elapsed = [r.get("elapsed_s", 0.0) for r in per_seed]

    def _m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def _s(xs):
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    test_mean, test_std = _m(test_aucs), _s(test_aucs)
    val_mean, val_std = _m(val_aucs), _s(val_aucs)
    train_mean, train_std = _m(train_aucs), _s(train_aucs)
    wall_min = float(sum(elapsed) / 60.0)

    delta_wf185 = float(test_mean - WF185_BASELINE_MEAN)
    delta_smoke = float(test_mean - WF301A_SMOKE_MEAN)

    passes_wf185 = bool(test_mean >= WF185_BASELINE_MEAN)
    passes_field_avg = bool(test_mean >= FIELD_AVG_TARGET_LO)
    passes_wf185_minus_slack = bool(
        test_mean >= WF185_BASELINE_MEAN - WF185_BASELINE_SEED_SLACK
    )

    verdict = _verdict(test_mean, test_std, n_seeds)

    aggregate = {
        "wf": "wf310d-drugood-dann-full-aggregate",
        "date": time.strftime("%Y-%m-%d"),
        "per_seed_dir": str(per_seed_dir),
        "smoke_aggregate": str(args.smoke_aggregate),
        "featurization": (
            "ECFP4 (Morgan radius=2, n_bits=2048) + chemprop v2 "
            "BinaryClassificationFFN + DANN scaffold-bin adversarial head "
            "[full run: 5-seed x 50-epoch GPU]"
        ),
        "per_seed": per_seed,
        "n_seeds_completed": int(n_seeds),
        "train_auc_mean": train_mean,
        "train_auc_std": train_std,
        "val_auc_mean": val_mean,
        "val_auc_std": val_std,
        "test_auc_mean": test_mean,
        "test_auc_std": test_std,
        "wall_min": wall_min,
        "wf185_v2_baseline_mean": WF185_BASELINE_MEAN,
        "wf185_v2_baseline_std": WF185_BASELINE_STD,
        "wf301a_smoke_mean": WF301A_SMOKE_MEAN,
        "wf301a_smoke_n_seeds": WF301A_SMOKE_N_SEEDS,
        "field_avg_target_lo": FIELD_AVG_TARGET_LO,
        "field_avg_target_hi": FIELD_AVG_TARGET_HI,
        "delta_vs_wf185": delta_wf185,
        "delta_vs_wf301a_smoke": delta_smoke,
        "passes_field_avg_target": passes_field_avg,
        "passes_wf185_v2_minus_slack": passes_wf185_minus_slack,
        "passes_wf185_v2": passes_wf185,
        "verdict": verdict,
    }
    out_agg = out_dir / "aggregate.json"
    out_agg.write_text(json.dumps(aggregate, indent=2))
    _log(f"wrote {out_agg}")

    out_cards = out_dir / "claim_cards.md"
    out_cards.write_text(
        _format_claim_cards(
            per_seed,
            test_mean,
            test_std,
            delta_wf185,
            delta_smoke,
            n_seeds,
            wall_min,
        )
    )
    _log(f"wrote {out_cards}")

    _log(
        f"TEST AUC = {test_mean:.4f} ± {test_std:.4f}  "
        f"vs Wf-185 v2 = {WF185_BASELINE_MEAN:.4f}  "
        f"vs Wf-301A smoke = {WF301A_SMOKE_MEAN:.4f}  "
        f"verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
