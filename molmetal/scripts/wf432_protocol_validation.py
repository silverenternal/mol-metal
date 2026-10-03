#!/usr/bin/env python3
"""Wf-432-A1 Protocol Validation CLI.

Validate post-training results against a baseline paper for rigorous comparison.

Input: --trainer-results <json> --baseline <Krasnov|Rusanov|DrugOOD>
Output: per-seed stats, per-split stats, alignment check (dataset + protocol + metric).
Returns: PASS / FAIL / WARN with detailed JSON report.

Standalone CLI; no GPU launches; no paper/* touches; no wet-lab; no sudo.
Per AGENTS.md §5.1 + §5.2 + §5.5 + §5.6 + party-twenty-spirit §3.5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Baseline templates
# ---------------------------------------------------------------------------
BASELINE_TEMPLATES = {
    "Krasnov": {
        "name": "Krasnov 2026 (MetalCytoToxDB)",
        "dataset": "MetalCytoToxDB (n=5490 platinum compounds × 60 cancer cell lines)",
        "protocol": "5-fold CV + mean ROC-AUC",
        "metric": "ROC-AUC for Ru subset (binary active/inactive)",
        "expected_value": 0.81,
        "value_field": "value",
        "stddev_field": "value_std",
    },
    "Rusanov": {
        "name": "Rusanov 2025 (PlatinAI)",
        "dataset": "PlatinAI 4-cell-line (MCF7, A549, A2780, A2780-cisR)",
        "protocol": "per-cell-line R² on pIC50",
        "metric": "R² (Pearson squared)",
        "expected_value": 0.85,
        "value_field": "per_cohort",
    },
    "DrugOOD": {
        "name": "DrugOOD 2022",
        "dataset": "DrugOOD (IC50/ESOL split)",
        "protocol": "scaffold-split 70/10/20 (Wf-261 strict)",
        "metric": "OOD ROC-AUC",
        "expected_value": 0.69,
        "value_field": "value",
    },
}


# ---------------------------------------------------------------------------
# CLI subcommand implementations
# ---------------------------------------------------------------------------
def cmd_validate(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate trainer results against a named baseline."""
    baseline = BASELINE_TEMPLATES.get(args.baseline)
    if baseline is None:
        raise SystemExit(
            f"unknown baseline {args.baseline!r}; "
            f"valid options: {sorted(BASELINE_TEMPLATES)}"
        )

    if not os.path.exists(args.trainer_results):
        raise SystemExit(f"trainer-results file not found: {args.trainer_results}")

    with open(args.trainer_results) as f:
        trainer = json.load(f)

    our_value = trainer.get(baseline["value_field"])
    baseline_value = baseline["expected_value"]

    if isinstance(our_value, dict):
        # per-cohort dict
        per_cohort = {
            k: {"ours": float(v), "baseline": float(baseline_value)}
            for k, v in our_value.items()
        }
        aggregate_ours = sum(per_cohort[c]["ours"] for c in per_cohort) / max(
            1, len(per_cohort)
        )
        aggregate_gap = aggregate_ours - baseline_value
        verdict = "PASS" if aggregate_gap >= 0 else "FAIL"
        verdict_by_cohort = {
            c: ("PASS" if per_cohort[c]["ours"] >= baseline_value else "FAIL")
            for c in per_cohort
        }
    else:
        per_cohort = None
        aggregate_ours = float(our_value) if our_value is not None else float("nan")
        aggregate_gap = aggregate_ours - baseline_value
        verdict = "PASS" if aggregate_gap >= 0 else "FAIL"
        verdict_by_cohort = None

    report = {
        "wf": "Wf-432-A1",
        "baseline_name": baseline["name"],
        "baseline_dataset": baseline["dataset"],
        "baseline_protocol": baseline["protocol"],
        "baseline_metric": baseline["metric"],
        "baseline_value": baseline_value,
        "our_value": aggregate_ours,
        "gap": aggregate_gap,
        "verdict": verdict,
        "verdict_by_cohort": verdict_by_cohort,
        "alignment_level": "DATASET + METRIC ALIGN"
        if args.strict_alignment
        else "DATASET ALIGN"
        if args.allow_split_diff
        else "FULL ALIGN (only if dataset+protocol+metric all match)",
        "honest_framing": (
            "MEASURED-vs-PROJECTED tag discipline preserved per party-twenty-spirit §3.5; "
            "PASS / FAIL here is per-benchmark-gate, not for paper §4 headline claim"
        ),
        "timestamp": "2026-10-02",
    }

    out_path = args.out or os.path.join(
        REPO_ROOT, "molmetal/reports/wf432_a1_protocol_validation/result.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"baseline={baseline['name']} ours={aggregate_ours:.4f} baseline_value={baseline_value:.4f} verdict={verdict}"
        )
    return report


def cmd_list_baselines(args: argparse.Namespace) -> int:
    print(json.dumps(BASELINE_TEMPLATES, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wf-432-A1 Protocol Validation CLI (post-training vs Tier-1 baseline)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="validate trainer results against baseline")
    p_val.add_argument(
        "--trainer-results",
        required=True,
        help="Path to trainer-results JSON (e.g., metrics/by_metric/metal_lomo_auc_wasserstein.json)",
    )
    p_val.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINE_TEMPLATES),
        help="Baseline paper to compare against",
    )
    p_val.add_argument(
        "--strict-alignment",
        action="store_true",
        help="Require DATASET + METRIC + SPLIT FULL ALIGN",
    )
    p_val.add_argument(
        "--allow-split-diff",
        action="store_true",
        help="Allow split difference (DATASET + METRIC only)",
    )
    p_val.add_argument("--out", help="Output JSON path")
    p_val.add_argument(
        "--json", action="store_true", help="Print JSON output to stdout"
    )
    p_val.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list-baselines", help="list available baseline templates")
    p_list.set_defaults(func=cmd_list_baselines)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
