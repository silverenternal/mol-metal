#!/usr/bin/env python3
"""Wf-432-A4 protocol alignment validator.

Validate that a trainer's post-training results are aligned with a baseline protocol:
- same dataset path / patterns
- same split protocol (scaffold / random / temporal)
- same metric definition (ROC-AUC, R², etc.)

Output: alignment report (DATASET_ALIGN / DATASET+METRIC / FULL / NOT_ALIGN) with §6 framing recommendation.
Standalone CLI; no GPU; no paper/*; no wet-lab; no sudo.
Per AGENTS.md §5.1 + §5.2 + §5.5 + §5.6 + party-twenty-spirit §3.5.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Protocol templates
# ---------------------------------------------------------------------------
PROTOCOL_TEMPLATES = {
    "Krasnov": {
        "name": "Krasnov 2026 (MetalCytoToxDB)",
        "dataset_pattern": r".*(metalcytotox|metal_cyto_5|metalcytotox_5).*",
        "split_protocol": "5-fold CV",
        "metric_definition": "mean ROC-AUC for Ru subset",
    },
    "Rusanov": {
        "name": "Rusanov 2025 (PlatinAI)",
        "dataset_pattern": r".*(platinai|platin_ai|mbfinder).*",
        "split_protocol": "per-cell-line train/val/test",
        "metric_definition": "per-cell-line R² on pIC50",
    },
    "DrugOOD": {
        "name": "DrugOOD 2022",
        "dataset_pattern": r".*drugood.*",
        "split_protocol": "scaffold-split (Wf-261 strict)",
        "metric_definition": "OOD ROC-AUC",
    },
    "TransDiffSBDD": {
        "name": "TransDiffSBDD / MolCRAFT (Tier-2 reference)",
        "dataset_pattern": r".*(crossdocked100|mmp13_real).*",
        "split_protocol": "scaffold-split 70/10/20",
        "metric_definition": "Vina kcal/mol median + Diversity Tanimoto + RMSD < 2Å",
    },
}


def _match_pattern(value: str, pattern: str) -> bool:
    """Match a string against a regex pattern (case-insensitive)."""
    return bool(re.search(pattern, value, re.IGNORECASE))


def _resolve_split_protocol(json_data: Dict[str, Any]) -> str:
    """Resolve split protocol from JSON metadata fields."""
    # Try common fields
    if "split_protocol" in json_data:
        return str(json_data["split_protocol"])
    if "split" in json_data:
        return str(json_data["split"])
    if "n_folds" in json_data:
        return f"{json_data['n_folds']}-fold CV"
    if "n_splits_effective" in json_data:
        return f"{json_data['n_splits_effective']} splits"
    return "unknown"


def _resolve_metric_definition(json_data: Dict[str, Any]) -> str:
    """Resolve metric definition from JSON metadata.

    Prefer `metric_definition` (verbose description) over `metric` (short identifier)
    so that comparisons against baseline `metric_definition` produce meaningful matches.
    """
    if "metric_definition" in json_data:
        return str(json_data["metric_definition"])
    if "metric" in json_data:
        return str(json_data["metric"])
    return "unknown"


def _resolve_dataset(json_data: Dict[str, Any]) -> str:
    """Resolve dataset path/pattern from JSON metadata."""
    for field in (
        "data_source",
        "data_source_path",
        "data_csv",
        "cohort_csv",
        "dataset",
    ):
        if field in json_data:
            return str(json_data[field])
    return "unknown"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_validate(args: argparse.Namespace) -> int:
    if not os.path.exists(args.trainer_results):
        raise SystemExit(f"trainer-results not found: {args.trainer_results}")
    if not os.path.exists(args.baseline_protocol):
        raise SystemExit(f"baseline-protocol not found: {args.baseline_protocol}")

    with open(args.trainer_results) as f:
        trainer = json.load(f)
    with open(args.baseline_protocol) as f:
        baseline = json.load(f)

    trainer_dataset = _resolve_dataset(trainer)
    baseline_dataset = baseline.get("dataset", "unknown")
    baseline_pattern = baseline.get("dataset_pattern", "")
    dataset_aligned = (
        bool(_match_pattern(trainer_dataset, baseline_pattern))
        if baseline_pattern
        else False
    )

    trainer_split = _resolve_split_protocol(trainer)
    baseline_split = baseline.get("split_protocol", "unknown")
    split_aligned = trainer_split == baseline_split

    trainer_metric = _resolve_metric_definition(trainer)
    baseline_metric = baseline.get("metric_definition", "unknown")
    metric_aligned = trainer_metric == baseline_metric

    if dataset_aligned and split_aligned and metric_aligned:
        alignment = "FULL"
        section_6_recommendation = "no §6 limitation needed for this metric"
    elif dataset_aligned and metric_aligned:
        alignment = "DATASET + METRIC ALIGN"
        section_6_recommendation = f"Split protocol differs: ours={trainer_split} vs baseline={baseline_split}; document in §6"
    elif dataset_aligned:
        alignment = "DATASET ALIGN"
        section_6_recommendation = (
            f"Split + metric differ; document both in §6 (ours split={trainer_split}, metric={trainer_metric}; "
            f"baseline split={baseline_split}, metric={baseline_metric})"
        )
    else:
        alignment = "NOT_ALIGN"
        section_6_recommendation = (
            "do NOT include in §4 comparison tables; document in §6 as out-of-scope"
        )

    report = {
        "wf": "Wf-432-A4",
        "trainer_results_path": args.trainer_results,
        "baseline_protocol_path": args.baseline_protocol,
        "checks": {
            "dataset_aligned": dataset_aligned,
            "split_protocol_aligned": split_aligned,
            "metric_definition_aligned": metric_aligned,
        },
        "details": {
            "our_dataset": trainer_dataset,
            "baseline_dataset": baseline_dataset,
            "our_split_protocol": trainer_split,
            "baseline_split_protocol": baseline_split,
            "our_metric_definition": trainer_metric,
            "baseline_metric_definition": baseline_metric,
        },
        "alignment_level": alignment,
        "section_6_recommendation": section_6_recommendation,
        "honest_framing": "MEASURED-vs-PROJECTED tag discipline preserved per party-twenty-spirit §3.5",
        "timestamp": "2026-10-02",
    }

    out_path = args.out or os.path.join(
        REPO_ROOT, "molmetal/reports/wf432_a4_protocol_alignment/result.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"alignment={alignment}  section_6={section_6_recommendation[:80]}...")
    return 0


def cmd_list_protocols(args: argparse.Namespace) -> int:
    print(json.dumps(PROTOCOL_TEMPLATES, indent=2, ensure_ascii=False))
    return 0


def cmd_emit_protocol(args: argparse.Namespace) -> int:
    """Emit a baseline-protocol JSON file from a template."""
    template = PROTOCOL_TEMPLATES.get(args.protocol)
    if template is None:
        raise SystemExit(
            f"unknown protocol {args.protocol!r}; valid options: {sorted(PROTOCOL_TEMPLATES)}"
        )

    out = dict(template)
    out["protocol"] = args.protocol

    out_path = args.out or os.path.join(
        REPO_ROOT,
        f"molmetal/reports/wf432_a4_protocol_alignment/{args.protocol}_protocol.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wf-432-A4 protocol alignment validator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser(
        "validate", help="validate trainer results align with a baseline protocol"
    )
    p_val.add_argument(
        "--trainer-results",
        required=True,
        help="Path to trainer results JSON (e.g. metrics/by_metric/<metric>.json)",
    )
    p_val.add_argument(
        "--baseline-protocol", required=True, help="Path to baseline protocol JSON"
    )
    p_val.add_argument("--out", help="Output report path")
    p_val.add_argument(
        "--json", action="store_true", help="Print JSON output to stdout"
    )
    p_val.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list-protocols", help="list available protocol templates")
    p_list.set_defaults(func=cmd_list_protocols)

    p_emit = sub.add_parser(
        "emit-protocol", help="emit a baseline-protocol JSON from template"
    )
    p_emit.add_argument("--protocol", required=True, choices=sorted(PROTOCOL_TEMPLATES))
    p_emit.add_argument("--out", help="Output JSON path")
    p_emit.set_defaults(func=cmd_emit_protocol)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
