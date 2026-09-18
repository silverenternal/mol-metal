#!/usr/bin/env python3
"""WF-R16-Deflex-v2 aggregator: diff two 30-cell arms (lambda_only vs deflex_f5)
and report per-cell + aggregate learned-shaping lift across
div_tanimoto / ref_tanimoto / validity / synth_rate.

Usage:  uv run python molmetal/reports/wf_r16_deflex_v2_verify/aggregate.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# r4_lambda_only_run.py prepends "wf_lambda1_" to --output-dir.
LAMBDA_PATH = (
    ROOT.parent
    / "wf_lambda1_wf_r16_deflex_v2_verify"
    / "lambda_only_arm"
    / "report.json"
)
DEFLEX_PATH = (
    ROOT.parent
    / "wf_lambda1_wf_r16_deflex_v2_verify"
    / "deflex_f5_arm"
    / "report.json"
)
OUT_FINAL = ROOT / "final.md"
OUT_JSON = ROOT / "final.json"

METRICS = [
    "diversity_tanimoto",     # div_tanimoto (post R16 schema)
    "reference_tanimoto",     # ref_tanimoto
    "validity_rate",          # validity
    "synthesizability_rate",  # synth_rate
]


def load_report(path: Path) -> dict[str, dict]:
    """Return {pocket_id__seed: cell_dict}."""
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for cell in data.get("cells", []):
        key = f"{cell['pocket_id']}__{cell.get('seed', '?')}"
        out[key] = cell
    return out


def per_cell_lift(l_cells: dict, d_cells: dict) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(l_cells.keys() & d_cells.keys()):
        l = l_cells[key]
        d = d_cells[key]
        row = {"cell": key}
        for m in METRICS:
            lv = float(l.get(m, 0.0))
            dv = float(d.get(m, 0.0))
            row[f"{m}_lambda"] = lv
            row[f"{m}_deflex"] = dv
            row[f"{m}_lift"] = dv - lv
        rows.append(row)
    return rows


def aggregate_lift(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in METRICS:
        lifts = [r[f"{m}_lift"] for r in rows]
        out[m] = {
            "n": len(lifts),
            "mean": round(statistics.fmean(lifts), 4),
            "std": round(statistics.pstdev(lifts), 4) if len(lifts) > 1 else 0.0,
            "min": round(min(lifts), 4),
            "max": round(max(lifts), 4),
        }
    return out


def verdict(agg: dict[str, dict]) -> tuple[str, str]:
    """Return (verdict_label, verdict_text)."""
    div = agg["diversity_tanimoto"]
    mean = div["mean"]
    if mean >= 0.05:
        return (
            "PASS",
            f"mean div_tanimoto lift = +{mean:.4f} >= +0.05 gate; promote Deflex.",
        )
    if mean >= 0.0:
        return (
            "NEUTRAL",
            f"mean div_tanimoto lift = +{mean:.4f} (in [0, +0.05)); honest NEUTRAL.",
        )
    return (
        "REGRESSION",
        f"mean div_tanimoto lift = {mean:.4f} < 0; halt Deflex promotion.",
    )


def main() -> None:
    if not LAMBDA_PATH.exists():
        raise SystemExit(f"missing {LAMBDA_PATH}")
    if not DEFLEX_PATH.exists():
        raise SystemExit(f"missing {DEFLEX_PATH}")
    l_cells = load_report(LAMBDA_PATH)
    d_cells = load_report(DEFLEX_PATH)
    rows = per_cell_lift(l_cells, d_cells)
    agg = aggregate_lift(rows)
    label, text = verdict(agg)

    final = {
        "lambda_only_path": str(LAMBDA_PATH),
        "deflex_path": str(DEFLEX_PATH),
        "n_cells": len(rows),
        "aggregate": agg,
        "verdict_label": label,
        "verdict_text": text,
        "per_cell": rows,
    }
    OUT_JSON.write_text(json.dumps(final, indent=2))

    lines = [
        "# WF-R16-Deflex-v2 — Verification + 30-cell Re-validation",
        "",
        "**Date:** 2026-09-18  ",
        "**Workflow:** R16b GPU ultracode, Phase E3 (Deflex verify)  ",
        "**Pre-flight:** BUG-2 (pocket_macro_inference CWD path) SHIPPED + verified.",
        "",
        "## 1. CLI invocation",
        "",
        "Lambda-only arm (control):",
        "```",
        "uv run python molmetal/scripts/r4_lambda_only_run.py \\",
        "    --pockets 10 --seeds 42 0 1234 \\",
        "    --n-simulations 1000 --n-samples 8 --n-top-k 20 \\",
        "    --output-dir wf_r16_deflex_v2_verify/lambda_only_arm \\",
        "    --quiet",
        "```",
        "",
        "Deflex F5 learned-shaping arm (test):",
        "```",
        "uv run python molmetal/scripts/r4_lambda_only_run.py \\",
        "    --pockets 10 --seeds 42 0 1234 \\",
        "    --n-simulations 1000 --n-samples 8 --n-top-k 20 \\",
        "    --use-learned-shaping --learned-shaping-weight 1.0 \\",
        "    --output-dir wf_r16_deflex_v2_verify/deflex_f5_arm \\",
        "    --quiet",
        "```",
        "",
        f"## 2. Per-cell results ({len(rows)} cells)",
        "",
        "| cell | div_alpha_L | div_alpha_D | div_lift | ref_tan_L | ref_tan_D | ref_lift | valid_L | valid_D | valid_lift | synth_L | synth_D | synth_lift |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} "
            f"| {r['diversity_tanimoto_lambda']:.3f} "
            f"| {r['diversity_tanimoto_deflex']:.3f} "
            f"| {r['diversity_tanimoto_lift']:+.3f} "
            f"| {r['reference_tanimoto_lambda']:.3f} "
            f"| {r['reference_tanimoto_deflex']:.3f} "
            f"| {r['reference_tanimoto_lift']:+.3f} "
            f"| {r['validity_rate_lambda']:.3f} "
            f"| {r['validity_rate_deflex']:.3f} "
            f"| {r['validity_rate_lift']:+.3f} "
            f"| {r['synthesizability_rate_lambda']:.3f} "
            f"| {r['synthesizability_rate_deflex']:.3f} "
            f"| {r['synthesizability_rate_lift']:+.3f} |"
        )

    lines += [
        "",
        "## 3. Aggregate (mean +/- std across 30 cells)",
        "",
        "| metric | mean lift | std lift | min | max |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        s = agg[m]
        lines.append(
            f"| {m} | {s['mean']:+.4f} | {s['std']:.4f} | {s['min']:+.4f} | {s['max']:+.4f} |"
        )

    lines += [
        "",
        f"## 4. VERDICT: **{label}**",
        "",
        text,
        "",
        "## 5. Gate rationale",
        "",
        "- PASS (mean div_tan lift >= +0.05): promote Deflex §3.5 + §4 Table 1 + §5 ablation.",
        "- NEUTRAL (0 <= lift < +0.05): honest finding, document as POOR-LIFT, no promotion.",
        "- REGRESSION (lift < 0): halt Deflex promotion; roll back learned-shaping wire.",
        "",
    ]

    if label == "PASS":
        lines += [
            "## 6. Follow-up",
            "",
            "PASS gate met. Re-validate `--learned-shaping-weight 0.3` next to",
            "match SA-penalty weight convention (see WF-SA-Penalty-Guidance).",
        ]
    elif label == "NEUTRAL":
        lines += [
            "## 6. Follow-up",
            "",
            "NEUTRAL: lift is positive but below gate. Document as honest finding.",
            "No promotion to MEASURED; keep §3.5 / §4 / §5 DESIGN.",
        ]
    else:
        lines += [
            "## 6. Follow-up",
            "",
            "REGRESSION: learned-shaping wire degrades diversity. Roll back",
            "`register_learned_shaping_channel` registration in r4_lambda_only_run.py:2997-3015",
            "and keep labels but reset to DESIGN.",
        ]

    OUT_FINAL.write_text("\n".join(lines) + "\n")
    print(f"VERDICT: {label}")
    print(text)
    print(f"wrote {OUT_FINAL}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()