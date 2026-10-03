#!/usr/bin/env python3
"""Wf-197 — Post-process attribution re-compute.

The launcher's in-memory `compute_attribution` had a bug in the 2-way
interaction formula (the bit-position logic was wrong for I_MT_L2 and
I_W_L2). This script re-computes the correct attribution from the raw
results in `ablation_summary.json`.

Run after `wf197_ablation_launcher.py` completes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "molmetal"
    / "reports"
    / "wf197_ablation"
)


def main() -> int:
    summary_path = OUT_DIR / "ablation_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())
    results = summary["results"]

    # Build by_code from raw results (test_auc may be None if config didn't complete)
    by_code: dict[str, dict] = {}
    for r in results:
        if r.get("test_auc") is not None:
            by_code[r["code"]] = r["test_auc"]

    def auc(code: str) -> float | None:
        return by_code.get(code)

    if "000" not in by_code:
        print("ERROR: 000 baseline not measured", file=sys.stderr)
        return 1

    baseline = auc("000")

    # Single-lever lifts (correct in original)
    single_lift = {
        "multi_task": (auc("100") or 0) - baseline if auc("100") is not None else None,
        "wasserstein": (auc("010") or 0) - baseline if auc("010") is not None else None,
        "l2_guard": (auc("001") or 0) - baseline if auc("001") is not None else None,
    }

    # 2-way interactions: I_AB = AUC(1,1) - AUC(1,0) - AUC(0,1) + AUC(0,0)
    # Bit positions: 0 = multi_task (MT), 1 = wasserstein (W), 2 = l2_guard (L2)

    def cfg(a_pos: int, b_pos: int, a: int, b: int) -> str:
        bits = ["0", "0", "0"]
        bits[a_pos] = str(a)
        bits[b_pos] = str(b)
        return "".join(bits)

    def interaction(a_pos: int, b_pos: int) -> float | None:
        a_v = auc(cfg(a_pos, b_pos, 1, 0))
        b_v = auc(cfg(a_pos, b_pos, 0, 1))
        ab_v = auc(cfg(a_pos, b_pos, 1, 1))
        c_v = auc(cfg(a_pos, b_pos, 0, 0))
        if any(v is None for v in [a_v, b_v, ab_v, c_v]):
            return None
        return ab_v - a_v - b_v + c_v

    interactions = {
        "I_MT_W": interaction(0, 1),
        "I_MT_L2": interaction(0, 2),
        "I_W_L2": interaction(1, 2),
    }

    full_recipe_lift = (auc("111") - baseline) if auc("111") is not None else None
    sum_single = sum(v for v in single_lift.values() if v is not None)
    sum_inter = sum(v for v in interactions.values() if v is not None)
    additivity_gap = None
    if full_recipe_lift is not None and sum_single is not None:
        additivity_gap = round(full_recipe_lift - sum_single - sum_inter, 6)

    corrected = {
        "baseline_auc": baseline,
        "single_lever_lift": single_lift,
        "two_way_interactions": interactions,
        "full_recipe_lift": full_recipe_lift,
        "sum_single_levers": sum_single,
        "sum_interactions": sum_inter,
        "additivity_gap": additivity_gap,
        "by_code": by_code,
    }

    summary["attribution"] = corrected
    summary["attribution_recomputed_note"] = (
        "Attribution re-computed post-launcher with correct bit-position logic. "
        "Original launcher had bug in I_MT_L2 / I_W_L2 interactions."
    )

    # Write back
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote corrected summary: {summary_path}")
    print(json.dumps(corrected, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
