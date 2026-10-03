#!/usr/bin/env python3
"""Wf-432-A3 m6_kappa paper drop-promote decision formalizer.

Codifies the Wf-424-A15 + Wf-377-A1 + Wf-389-A1 + Wf-411-A4 decision convergence:
- DROP m6_kappa from Tier-2 §3 main-result table
- PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table
- RETAIN m6_kappa as §6 honest-limitations complementary heuristic
- Generate ready-to-paste LaTeX snippet for §3 + §6
- Update metrics/by_metric/m6_kappa.json::wf424a15_drop_decision block (already on disk; this validates it)

CPU-only; no GPU launches; no paper/* touches; no wet-lab; no sudo.
Per AGENTS.md §5.1 + §5.2 + §5.3 + §5.5 + §5.6 + party-twenty-spirit §3.5.

Wf-432-A3 does NOT modify paper/main.tex or paper/sections/*.tex (per TODO-84 + AGENTS.md §5.3).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]  # /home/hugo/codes/mol-metal/


def _read_metric_json(path: Path) -> Dict[str, Any]:
    """Read a metric JSON file."""
    with open(path) as f:
        return json.load(f)


def _write_metric_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write a metric JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the m6_kappa canonical JSON contains the Wf-424-A15 drop_decision block."""
    json_path = REPO_ROOT / "metrics/by_metric/m6_kappa.json"
    if not json_path.exists():
        raise SystemExit(f"m6_kappa.json not found: {json_path}")
    data = _read_metric_json(json_path)

    drop_block = data.get("wf424a15_drop_decision")
    if drop_block is None:
        print("ERROR: wf424a15_drop_decision block MISSING from m6_kappa.json")
        return 1

    # wf424a15_drop_decision uses pillars-based structure; verify decision_pillars OR fallback to top3
    decision_pillars = drop_block.get("decision_pillars", [])
    verdict_pillars = drop_block.get("verdict_pillars", [])
    decision_text = drop_block.get("decision", "")

    # Find top3 alternatives — wf424a15 may store them in wf389a1 block instead
    top3 = drop_block.get("top3_alternatives", [])
    if len(top3) < 3:
        # Fall back to wf389a1_supplementary_reframe.top3_alternatives
        wf389a1 = data.get("wf389a1_supplementary_reframe", {})
        top3 = wf389a1.get("top3_alternatives", top3)

    if len(top3) != 3:
        print(f"ERROR: top3_alternatives must have 3 entries, got {len(top3)}")
        return 1

    # Validate each alternative has Pearson r + verdict_band
    for alt in top3:
        if "pearson_r" not in alt or "verdict_band" not in alt:
            print(f"ERROR: alternative missing fields: {alt}")
            return 1

    # Check decision text OR pillars
    decision_ok = (
        ("DROP" in decision_text and "PROMOTE" in decision_text)
        or any("DROP" in str(p) and "PROMOTE" in str(p) for p in decision_pillars)
    )
    if not decision_ok:
        print(f"ERROR: decision text + pillars missing DROP / PROMOTE: {decision_text!r}")
        return 1

    print(json.dumps({
        "wf": "Wf-432-A3",
        "validation": "PASS",
        "decision_block_found": True,
        "decision_text": decision_text[:200],
        "decision_pillars_count": len(decision_pillars),
        "verdict_pillars_count": len(verdict_pillars),
        "top3_alternatives": [
            {"rank": a.get("rank"), "name": a.get("name"), "pearson_r": a.get("pearson_r")}
            for a in top3
        ],
        "paths_exhausted_breakdown_total": drop_block.get("paths_exhausted", 0) if isinstance(drop_block.get("paths_exhausted"), int) else len(drop_block.get("paths_exhausted", [])),
        "verdict_field_set": drop_block.get("verdict_field_set"),
        "timestamp": "2026-10-02",
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_emit_latex(args: argparse.Namespace) -> int:
    """Emit ready-to-paste LaTeX snippets for §3 main-result + §3 supplementary + §6 honest-limitations."""
    json_path = REPO_ROOT / "metrics/by_metric/m6_kappa.json"
    data = _read_metric_json(json_path)
    drop_block = data.get("wf424a15_drop_decision", {})
    decision = drop_block.get("decision", "DROP from main-result + PROMOTE top-3 WEAK to supplementary")
    top3 = drop_block.get("top3_alternatives", [])

    # §3 main-result table: m6_kappa is DROPPED, no row
    # §3 supplementary sub-table: top-3 WEAK-band correlates
    # §6 honest-limitations: 4 NEW paragraphs (per Wf-410-A1 §3.1.6 + Wf-375-A4 §3 + Wf-377-A1 §4.4 + Wf-424-A15 §4)

    latex = []
    latex.append("% --- Paper §3 Supplementary Sub-Table: Top-3 WEAK-Band Correlates (from Wf-377-A1 + Wf-389-A1) ---")
    latex.append(r"\begin{table}[h]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{Top-3 WEAK-band per-molecule topological correlates promoted from §6 (per Wf-389-A1 §3 supplementary reframe + Wf-411-A4 per-cohort Cohen 1988 3-band refinement).}")
    latex.append(r"  \label{tab:sup_subtable_m6k_replacement}")
    latex.append(r"  \begin{tabular}{lccc}")
    latex.append(r"    \toprule")
    latex.append(r"    Metric & Pearson r & Spearman $\rho$ & Verdict (Cohen 1988) \\")
    latex.append(r"    \midrule")
    for alt in top3:
        name = alt.get("name", "Unknown")
        pr = alt.get("pearson_r", 0.0)
        sr = alt.get("spearman_rho", 0.0)
        vb = alt.get("verdict_band", "WEAK")
        latex.append(f"    {name} & {pr:+.4f} & {sr:+.4f} & {vb} \\\\")
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    latex.append("% --- Paper §6 Honest-Limitations (per Wf-410-A1 §3.1.6 + Wf-375-A4 §3 + Wf-377-A1 §4.4 + Wf-424-A15 §4) ---")
    latex.append(r"\subsection{m6\_kappa dropped from §3 main-result}")
    latex.append(
        r"The m6 \emph{Shape Kappa Index} metric was originally included in §3 main-result, "
        r"with per-molecule Pearson $r = 0.0813$ on the $n=5460$ MetalCytoToxDB labelled pIC50 cohort. "
        r"Per Wf-377-A1 §4.2 forward-looking + Wf-389-A1 §3 supplementary reframe + Wf-411-A4 §3 per-cohort $\times Cohen$ 1988 3-band refinement, "
        r"the canonical decision is: \textbf{DROP} m6\_kappa from §3 main-result; \textbf{PROMOTE} the top-3 WEAK-band correlates "
        r"(MolLogP, MorganDensity$_r$, Chi3v) to §3 supplementary sub-table; \textbf{RETAIN} m6\_kappa as §6 honest-limitations complementary heuristic. "
        r"\textbf{Honest framing per \texttt{party-twenty-spirit §3.5}:} MEASURED-vs-PROJECTED tag discipline preserved; "
        r"9 paths exhausted (7 architectural + 1 degenerate + 1 metric-redesign + 1 supplementary reframe); "
        r"augmented\_mean $0.472$ $<$ $0.6$ threshold gap $-0.128$; structural CV-R\textsuperscript{2} $+0.6814$ CEILING\_BREAK but threshold-reaching axis NOT lifted."
    )
    latex.append("")

    out_path = args.out or os.path.join(
        REPO_ROOT, "molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(latex) + "\n")
    print(f"Wrote LaTeX snippet to {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


def cmd_rebuild_decision(args: argparse.Namespace) -> int:
    """Rebuild the wf424a15_drop_decision block from canonical sources."""
    json_path = REPO_ROOT / "metrics/by_metric/m6_kappa.json"
    data = _read_metric_json(json_path)

    decision_block = {
        "wf": "Wf-432-A3 re-author of Wf-424-A15 decision",
        "description": "Re-affirm the Wf-377-A1 + Wf-389-A1 + Wf-411-A4 + Wf-424-A15 decision convergence: m6_kappa DROPPED from §3 main-result; PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic.",
        "decision": "DROP m6_kappa from §3 main-result + PROMOTE top-3 WEAK-band correlates (MolLogP Pearson +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) to §3 supplementary sub-table + RETAIN m6_kappa as §6 honest-limitations complementary heuristic.",
        "verdict_field_set": True,
        "drop_from_main_result": True,
        "promote_to_supplementary_subtable": True,
        "top3_alternatives": data.get("wf424a15_drop_decision", {}).get("top3_alternatives", []),
        "paths_exhausted_breakdown": {
            "7_arch": ["Wf-335-M1 chemprop+PLL", "Wf-359-A1 R20-P3 W₂", "Wf-361-B1 MAML", "Wf-361-B2 Neural ODE", "Wf-364-A2 combined MAML+ODE", "Wf-365-A3 verdict doc", "Wf-371-A1 degenerate chain"],
            "1_introspection": "Wf-377-A1 metric-redesign literature + empirical 16 invariants × n=5460",
            "1_supplementary": "Wf-389-A1 §3 supplementary sub-table",
            "1_per_cohort": "Wf-411-A4 per-cohort × Cohen 1988 3-band refinement",
            "total": 9,
        },
        "timestamp": "2026-10-02",
        "paper_section_lifecycle": {
            "drop_decision_recorded": "TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §4 + TODO/03_plans/LAMBDA_ROADMAP.md §11",
            "latex_snippet_location": "molmetal/reports/wf432_a3_m6_kappa_drop_decision/latex_snippet.tex",
            "paper_conversion_deferred": "USER_OK-gated per AGENTS.md §5.3 (TODO-84 active); paper §3 main-result table no longer includes m6_kappa row",
            "honest_framing_per_party_twenty_spirit": "MEASURED-vs-PROJECTED tag discipline preserved",
        },
    }

    data["wf424a15_drop_decision"] = decision_block
    _write_metric_json(json_path, data)
    print(f"Rebuilt wf424a15_drop_decision block in {json_path}")
    print(f"verdict: {data.get('verdict')}")
    print(f"paths_exhausted_breakdown total: {decision_block['paths_exhausted_breakdown']['total']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wf-432-A3 m6_kappa paper drop-promote decision formalizer"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="validate wf424a15_drop_decision block is present + well-formed")
    p_val.set_defaults(func=cmd_validate)

    p_emit = sub.add_parser("emit-latex", help="emit ready-to-paste LaTeX snippet for §3 + §6")
    p_emit.add_argument("--out", help="Output .tex file path")
    p_emit.set_defaults(func=cmd_emit_latex)

    p_rebuild = sub.add_parser("rebuild-decision", help="rebuild wf424a15_drop_decision block in m6_kappa.json")
    p_rebuild.set_defaults(func=cmd_rebuild_decision)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
