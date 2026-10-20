#!/usr/bin/env python3
"""Restore the Wf-31 provenance blocks that two metric files lost.

``molmetal/tests/test_wf31_phase5_lift_analysis.py`` expects
``wf31_phase5_lift_update`` in three metric files.  ``nci60_precious_9cell_auc.json``
has one; ``metalcytotox_ru_auc.json`` and ``diversity_tanimoto.json`` do not —
they were re-written by later runs (Wf-124 P3 and Wf-129 P2 respectively), and
each rewrite dropped the block along with the Wf-31 measurement entry.  Twelve
tests have been red ever since, and pre-flight never noticed because it does not
run this suite.

This script rebuilds the blocks from the analysis that produced them,
``molmetal/reports/wf31_m1_m2_m7_lift_analysis.json``, plus
``molmetal/reports/wf31_m7_summary.json`` for the expressivity boolean.  It
invents nothing: every field is copied or derived from those two files, and the
block records which one it came from.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf_a56_restore_wf31_blocks.py --dry-run
    .venv/bin/python molmetal/scripts/wf_a56_restore_wf31_blocks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS = "molmetal/reports/wf31_m1_m2_m7_lift_analysis.json"
M7_SUMMARY = "molmetal/reports/wf31_m7_summary.json"

BLOCK_KEY = "wf31_phase5_lift_update"

# metric_id in the analysis -> (metric file, whether a measurement row is needed)
TARGETS = {
    "m1": ("metrics/by_metric/metalcytotox_ru_auc.json", False),
    "m7": ("metrics/by_metric/diversity_tanimoto.json", True),
}

M7_RUN = "wf31_phase4_multi_ph_diversity_bench"


def _build_block(entry: Dict, analysis_rel: str, idx: int) -> Dict:
    """Flatten one analysis entry into the block the tests expect."""
    measured = entry.get("r18_wf31_measured") or {}
    block: Dict[str, object] = {
        "metric_id": entry["metric_id"],
        "metric_name": entry.get("metric_name"),
        "workflow": "Wf-31",
        "phase": "Phase 5 — MEASURED vs R17 baseline lift comparison",
        "source_analysis": f"{analysis_rel}::metrics[{idx}]",
        "r17_baseline_value": (entry.get("r17_baseline") or {}).get("value"),
        "r17_baseline_source": (entry.get("r17_baseline") or {}).get("source"),
        "r17_baseline_status": (entry.get("r17_baseline") or {}).get("status"),
        "r18_wf31_status": measured.get("status"),
        "r18_wf31_measured_value": measured.get("value"),
        "r18_wf31_source": measured.get("source"),
        "r18_wf31_note": measured.get("note"),
        "r18_wf31_defer_reason": measured.get("reason") or entry.get("verdict_detail"),
        "lift_r18_minus_r17": entry.get("lift_r18_minus_r17"),
        "pass_target_r18": entry.get("pass_target_r18"),
        "sota_target": entry.get("sota_target"),
        "ship_gate": entry.get("ship_gate"),
        "ship_gate_passed": entry.get("ship_gate_passed"),
        "verdict": entry.get("verdict"),
        "verdict_detail": entry.get("verdict_detail"),
        "r20_plan_item": entry.get("r20_plan_item"),
    }
    if "wf22_history" in entry:
        block["wf22_history"] = entry["wf22_history"]
    if "lift_r18_minus_r17_interpretation" in entry:
        block["lift_r18_minus_r17_naive"] = entry.get("lift_r18_minus_r17_naive")
        block["lift_r18_minus_r17_interpretation"] = entry[
            "lift_r18_minus_r17_interpretation"
        ]
    if "r18_wf31_expressivity_measured" in entry:
        block["r18_wf31_expressivity_measured"] = entry[
            "r18_wf31_expressivity_measured"
        ]
    return block


def _m7_measurement(summary: Dict) -> Dict:
    """The Wf-31 m7 row that belongs in ``measurements[]``."""
    div = summary["diversity"]
    return {
        "run": f"{M7_RUN} (PlatinAI 5K random subset, n_samples=500, seed=42, "
        f"n_bins=16)",
        "value": div["mean_tanimoto_distance"],
        "n_samples": summary.get("n_samples"),
        "status": summary.get("status", "MEASURED"),
        "date_utc": summary.get("date"),
        "wall_seconds": summary.get("expressivity", {}).get("multi_ph_wall_s"),
        "note": (
            "Multi-PH diversity bench. Cohort differs from the Lambda-MCTS "
            "rows above: this measures the intrinsic diversity of a fixed "
            "PlatinAI 5K subset, not the model's generation. See "
            "wf31_phase5_lift_update.lift_r18_minus_r17_interpretation — the "
            "naive lift is APPLES_TO_ORANGES."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild even if present")
    args = p.parse_args(argv)

    analysis_path = PROJECT_ROOT / ANALYSIS
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    by_id = {m["metric_id"]: (i, m) for i, m in enumerate(analysis["metrics"])}

    summary_path = PROJECT_ROOT / M7_SUMMARY
    m7_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )

    rc = 0
    for metric_id, (rel, needs_measurement) in TARGETS.items():
        if metric_id not in by_id:
            print(f"[wf31] ERROR: {ANALYSIS} has no {metric_id} entry", file=sys.stderr)
            rc = 2
            continue
        idx, entry = by_id[metric_id]
        path = PROJECT_ROOT / rel
        doc = json.loads(path.read_text(encoding="utf-8"))

        if BLOCK_KEY in doc and not args.force:
            print(f"[wf31] {rel}: block already present — skip (use --force to rebuild)")
            continue

        block = _build_block(entry, ANALYSIS, idx)
        doc[BLOCK_KEY] = block

        added_measurement = False
        if needs_measurement:
            runs = [m.get("run", "") for m in doc.get("measurements", [])]
            if not any(M7_RUN in r for r in runs):
                if m7_summary is None:
                    print(
                        f"[wf31] ERROR: {M7_SUMMARY} missing — cannot restore the "
                        f"m7 measurement row without it",
                        file=sys.stderr,
                    )
                    rc = 2
                    continue
                doc.setdefault("measurements", []).append(
                    _m7_measurement(m7_summary)
                )
                added_measurement = True
                # Keep aggregate honest: best_measured is scoped to the
                # Lambda-MCTS trajectory, and the new row is a different cohort.
                agg = doc.setdefault("aggregate", {})
                agg["scope_note"] = (
                    "best_measured is scoped to the Lambda-MCTS click-diversity "
                    f"trajectory. The {M7_RUN} row is a different cohort "
                    "(PlatinAI 5K intrinsic diversity) and is deliberately not "
                    "folded into it — see that row's note."
                )
                block["r18_section_3_6_t5_multi_ph_theoretical_lift_passed"] = bool(
                    m7_summary.get("expressivity", {}).get(
                        "multi_ph_more_expressive"
                    )
                )
                doc[BLOCK_KEY] = block

        print(
            f"[wf31] {rel}: status={block['r18_wf31_status']} "
            f"value={block['r18_wf31_measured_value']}"
            + (" (+measurement row)" if added_measurement else "")
        )
        if args.dry_run:
            print("[wf31] dry-run — nothing written")
            continue

        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(path, json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"[wf31] wrote {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
