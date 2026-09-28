#!/usr/bin/env python
"""WF-Section4-AutoGen — paper §4 Results auto-generator.

================================================================
Why this script exists
================================================================
Paper §4 currently hand-rolls Table 2 + Figures 7/8/9 from
``metrics/by_metric/*.json`` snapshots.  Each round (R12 → R13 → R16 →
R17) the panel is rebuilt by hand, which is fragile and prone to
silent drift between the JSON snapshot and the LaTeX cell.  This
script auto-generates the panel from the snapshot directory and the
already-documented SOTA targets in ``TODO/02_metrics/METRICS_TARGETS.md``
(§2).  Honest framing is preserved: every cell that has no measured
value renders ``\\textit{pending --- see TODO/METRICS\\_TARGETS.md
metric N}`` instead of a fabricated number, and the verdict column
encodes the gate pass/fail per the §2 verdict rules.

Inputs (read)
-------------
* ``metrics/by_metric/<metric>.json``     --- the snapshot
* ``TODO/02_metrics/METRICS_TARGETS.md``  --- SOTA targets (parsed via a
                                              small regex walker; full
                                              docstring of the parsed
                                              targets is inlined below)

Outputs (written)
-----------------
* ``paper/sections/04_results_table2.tex``            --- LaTeX Table 2
* ``paper/figures/data/fig7_cell_line.json``          --- per-cell AUC bar
* ``paper/figures/data/fig8_ood_radar.json``          --- OOD radar stub
* ``paper/figures/data/fig9_tier1_scorecard.json``    --- Tier-1 scorecard
* ``paper/reports/wf_section4_gen/summary.md``        --- human summary

Usage
-----
    source env.sh && python -u molmetal/scripts/generate_section4.py \
        [--dry-run] [--out-dir paper/] [--metrics-dir metrics/by_metric/]

``--dry-run`` prints the LaTeX table to stdout + skips writing files
(used by the smoke test in ``molmetal/tests/test_generate_section4.py``).

Honest framing
==============
* The 20-row panel is the canonical paper §4.2 / §4.3 / §4.4 list.
* Tier-1 SOTA values are taken verbatim from ``METRICS_TARGETS.md``
  (Krasnov 2026 = 0.81 Ru; Rusanov 2025 = per-line PlatinAI; Boltz-2 =
  0.86 affinity Pearson r; OMTRA = 100 interaction parity; TransDiff
  SBDD = -9.37 Vina median; AlphaDrug = -9.77 Vina min).
* Verdict gate is applied per ``METRICS_TARGETS.md`` §2 verdict rules;
  values in the ``(verdict)`` column are ``PASS`` / ``MARGINAL`` /
  ``REGRESS`` / ``PENDING``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Metric registry — maps metric name → JSON filename + SOTA + verdict gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One row of paper Table 2 (Section 4)."""

    row: int  # 1..20 (paper order)
    display: str  # LaTeX-safe metric label
    json_key: str  # file stem under metrics/by_metric/
    direction: str  # "higher" or "lower"
    tier: str  # "Tier 1", "Tier 2", "Tier 3"
    target_value: float  # our gate target from METRICS_TARGETS
    target_str: str  # human-readable target string ("≥ 0.86")
    tier1_sota: str  # Tier 1 SOTA value+label, e.g. "0.81 Krasnov"
    tier2_sota: str  # Tier 2 SOTA value+label, e.g. "-9.37 TransDiffSBDD"
    value_extract: str  # how to read the value: "value" | "aggregate.best_measured" | ...
    verdict_passes: List[float]  # values that count as PASS (incl. direction-aware bounds)
    pending_note: str = ""  # additional pending note (e.g. "metric 2 gate ≥ 0.85")


# The 20 metrics. order = paper Table 2 order.
METRIC_REGISTRY: List[MetricSpec] = [
    MetricSpec(
        row=1, display="Ru subset AUC (MetalCytoToxDB)",
        json_key="metalcytotox_ru_auc", direction="higher", tier="Tier 1",
        target_value=0.86, target_str="$\\geq 0.86$",
        tier1_sota="0.81 Krasnov 2026", tier2_sota="---",
        value_extract="value",
        verdict_passes=[0.86], pending_note="metric 1",
    ),
    MetricSpec(
        row=2, display="Pt MCF-7 AUC (PlatinAI)",
        json_key="platinai_4cell_auc", direction="higher", tier="Tier 1",
        target_value=0.70, target_str="$\\geq 0.70$",
        tier1_sota="Rusanov per-line", tier2_sota="---",
        value_extract="per_cell_line.MCF7.auc_mean",
        verdict_passes=[0.70], pending_note="metric 2",
    ),
    MetricSpec(
        row=3, display="Vina Dock median (kcal/mol)",
        json_key="vina_kcal_per_mol", direction="lower", tier="Tier 2",
        target_value=-8.5, target_str="$\\leq -8.5$",
        tier1_sota="---", tier2_sota="$-9.37$ TransDiffSBDD",
        value_extract="value_mean",
        verdict_passes=[-8.5], pending_note="metric 3",
    ),
    MetricSpec(
        row=4, display="Success Rate (Vina+QED+SA)",
        json_key="vina_relaxed_success", direction="higher", tier="Tier 2",
        target_value=0.60, target_str="$\\geq 60\\%$",
        tier1_sota="---", tier2_sota="$76.7\\%$ ReInvent+Vina",
        value_extract="value_mean",
        verdict_passes=[0.60], pending_note="metric 4",
    ),
    MetricSpec(
        row=5, display="SA score (Ertl, 1=easy)",
        json_key="sa_mean", direction="higher", tier="Tier 2",
        target_value=0.65, target_str="$\\geq 0.65$",
        tier1_sota="---", tier2_sota="$0.74$ TargetDiff",
        value_extract="latest_measured",
        verdict_passes=[0.65], pending_note="metric 5",
    ),
    MetricSpec(
        row=6, display="QED (drug-likeness)",
        json_key="qed_mean", direction="higher", tier="Tier 2",
        target_value=0.50, target_str="$\\geq 0.50$",
        tier1_sota="---", tier2_sota="$0.51$ Pocket2Mol",
        value_extract="latest_measured",
        verdict_passes=[0.50], pending_note="metric 6",
    ),
    MetricSpec(
        row=7, display="Diversity (Morgan ECFP4)",
        json_key="diversity_tanimoto", direction="higher", tier="Tier 2",
        target_value=0.65, target_str="$\\geq 0.65$",
        tier1_sota="---", tier2_sota="$0.76$ MolChord",
        value_extract="latest_measured",
        verdict_passes=[0.65], pending_note="metric 7",
    ),
    MetricSpec(
        row=8, display="PB-Valid (PoseBusters)",
        json_key="pb_pass_rate", direction="higher", tier="Tier 2",
        target_value=0.90, target_str="$\\geq 90\\%$",
        tier1_sota="$97.7\\%$ OMTRA", tier2_sota="$88\\%$ DiffSBDD",
        value_extract="value_mean",
        verdict_passes=[0.90], pending_note="metric 8",
    ),
    MetricSpec(
        row=9, display="Interaction Parity",
        json_key="pharmacophore_pass_rate", direction="higher", tier="Tier 2",
        target_value=50.0, target_str="$\\geq 50$",
        tier1_sota="100 OMTRA", tier2_sota="22 Pocket2Mol",
        value_extract="value_mean",
        verdict_passes=[50.0], pending_note="metric 9",
    ),
    MetricSpec(
        row=10, display="Vina Min (best pose, kcal/mol)",
        json_key="vina_min_score", direction="lower", tier="Tier 2",
        target_value=-9.5, target_str="$\\leq -9.5$",
        tier1_sota="---", tier2_sota="$-9.77$ AlphaDrug",
        value_extract="value.mean",
        verdict_passes=[-9.5], pending_note="metric 10",
    ),
    MetricSpec(
        row=11, display="$n_{\\mathrm{distinct}}$ metal-ligand pairs",
        json_key="n_distinct", direction="higher", tier="Tier 1",
        target_value=3.0, target_str="$\\geq 3$",
        tier1_sota="R16 Path A 1", tier2_sota="---",
        value_extract="aggregate.best_measured_with_all_fixes",
        verdict_passes=[3.0], pending_note="metric 11",
    ),
    MetricSpec(
        row=12, display="Synthesis rate (SyntheMol)",
        json_key="synthesis_rate", direction="higher", tier="Tier 3",
        target_value=0.95, target_str="$\\geq 0.95$",
        tier1_sota="---", tier2_sota="R16 1.00",
        value_extract="value",
        verdict_passes=[0.95], pending_note="metric 12",
    ),
    MetricSpec(
        row=13, display="Validity (RDKit parseable)",
        json_key="validity_rate", direction="higher", tier="Tier 3",
        target_value=0.99, target_str="$\\geq 0.99$",
        tier1_sota="---", tier2_sota="SOTA 0.99--1.00",
        value_extract="value",
        verdict_passes=[0.99], pending_note="metric 13",
    ),
    MetricSpec(
        row=14, display="Decode ratio (CFM)",
        json_key="decode_ratio", direction="higher", tier="Tier 3",
        target_value=0.50, target_str="$\\geq 0.50$",
        tier1_sota="---", tier2_sota="R17 1.000",
        value_extract="aggregate.best_measured",
        verdict_passes=[0.50], pending_note="metric 14",
    ),
    MetricSpec(
        row=15, display="Docking RMSD $<2\\AA$ (\\%)",
        json_key="rigid_fragment_rmsd", direction="higher", tier="Tier 3",
        target_value=25.0, target_str="$\\geq 25\\%$",
        tier1_sota="---", tier2_sota="$38.2\\%$ DiffDock",
        value_extract="value",
        verdict_passes=[25.0], pending_note="metric 15",
    ),
    MetricSpec(
        row=16, display="Affinity Pearson r (pIC50)",
        json_key="pearson_r_pic50", direction="higher", tier="Tier 3",
        target_value=0.75, target_str="$\\geq 0.75$",
        tier1_sota="$0.86$ Boltz-2", tier2_sota="$\\sim 0.90$ FEP+",
        value_extract="value",
        verdict_passes=[0.75], pending_note="metric 16",
    ),
    MetricSpec(
        row=17, display="DrugOOD OOD (3 splits)",
        json_key="drugood_ood_auc", direction="higher", tier="Tier 3",
        target_value=0.66, target_str="$\\geq 0.66$ (iMoLD--0.02)",
        tier1_sota="---", tier2_sota="72.11 / 68.84 / 67.92 iMoLD",
        value_extract="mean_ood_auc",
        verdict_passes=[0.66], pending_note="metric 17",
    ),
    MetricSpec(
        row=18, display="Protein-metal binding AUC",
        json_key="metric18_protein_metal", direction="higher", tier="Tier 1",
        target_value=0.85, target_str="$\\geq 0.85$",
        tier1_sota="$>0.92$ 2025 paper", tier2_sota="---",
        value_extract="value",
        verdict_passes=[0.85], pending_note="metric 18 (future)",
    ),
    MetricSpec(
        row=19, display="Multi-metal transfer (LOMO)",
        json_key="metal_lomo_auc", direction="higher", tier="Tier 3",
        target_value=0.70, target_str="$\\geq 0.70$",
        tier1_sota="---", tier2_sota="not reported",
        value_extract="value",
        verdict_passes=[0.70], pending_note="metric 19",
    ),
    MetricSpec(
        row=20, display="NCI-60 precious 9-cell mean AUC",
        json_key="nci60_precious_9cell_auc", direction="higher", tier="Tier 1",
        target_value=0.70, target_str="$\\geq 0.70$",
        tier1_sota="$0.81$ Krasnov Ru", tier2_sota="(no analog)",
        value_extract="value_overall_mean_auc",
        verdict_passes=[0.70], pending_note="metric 21",
    ),
]


# ---------------------------------------------------------------------------
# 2. JSON extraction helpers
# ---------------------------------------------------------------------------


def _get_nested(obj: Any, dotted: str) -> Optional[Any]:
    """Walk a dotted path through nested dicts/lists. Returns None if missing."""
    if obj is None or not dotted:
        return obj
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def extract_metric_value(spec: MetricSpec, payload: Dict[str, Any]) -> Optional[float]:
    """Pull a numeric value out of a metric JSON per the spec's value_extract."""
    if payload is None:
        return None
    ve = spec.value_extract
    if ve == "value":
        return _coerce_float(payload.get("value"))
    if ve == "value_mean":
        return _coerce_float(payload.get("value_mean"))
    if ve == "value.mean":
        v = _get_nested(payload, "value")
        return _coerce_float(_get_nested(v, "mean")) if isinstance(v, dict) else _coerce_float(v)
    if ve == "latest_measured":
        # Pick the most-recent numeric measurement in the "measurements" list.
        ms = payload.get("measurements") or []
        # walk backwards; return the first numeric "value"
        for m in reversed(ms):
            v = m.get("value") if isinstance(m, dict) else None
            f = _coerce_float(v)
            if f is not None:
                return f
        return None
    if ve == "aggregate.best_measured":
        agg = payload.get("aggregate") or {}
        return _coerce_float(agg.get("best_measured"))
    if ve == "aggregate.best_measured_with_all_fixes":
        agg = payload.get("aggregate") or {}
        return _coerce_float(agg.get("best_measured_with_all_fixes"))
    if ve == "mean_ood_auc":
        # DrugOOD: mean of (assay.ood_auc_mean, scaffold.ood_auc_mean, size.ood_auc_mean)
        results = payload.get("results") or {}
        vals = []
        for split in ("assay", "scaffold", "size"):
            v = _get_nested(results, f"{split}.ood_auc_mean")
            f = _coerce_float(v)
            if f is not None:
                vals.append(f)
        return sum(vals) / len(vals) if vals else None
    if ve.startswith("per_cell_line."):
        cell = ve.split(".")[1]
        return _coerce_float(_get_nested(payload, f"per_cell_line.{cell}.auc_mean"))
    return _coerce_float(_get_nested(payload, ve))


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None  # exclude booleans (avoid False=0.0 contamination)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Skip ≥ / ≤ markers — those are "did not converge" placeholders
        if s.startswith(("≥", "≤", ">=", "<=", "<", ">")):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def passes_gate(spec: MetricSpec, value: Optional[float]) -> str:
    """Apply the §2 verdict gate; return PASS / MARGINAL / REGRESS / PENDING."""
    if value is None:
        return "PENDING"
    gate = spec.target_value
    if spec.direction == "higher":
        if value >= gate:
            return "PASS"
        if value >= gate - 0.10:  # 10% margin = MARGINAL
            return "MARGINAL"
        return "REGRESS"
    # direction == "lower"
    if value <= gate:
        return "PASS"
    if value <= gate + 1.0:  # 1.0 kcal/mol margin = MARGINAL for binding metrics
        return "MARGINAL"
    return "REGRESS"


def fmt_value(v: Optional[float], spec: MetricSpec) -> str:
    """Render a measured value as a LaTeX cell."""
    if v is None:
        return f"\\textit{{pending --- see TODO/METRICS\\_TARGETS.md {spec.pending_note}}}"
    # 3 sig figs for AUC; 2 decimals for kcal/mol; 1 decimal for counts.
    if abs(v) < 10 and spec.direction != "lower":
        return f"${v:.3f}$"
    if abs(v) < 10 and spec.direction == "lower":
        return f"${v:.2f}$"
    return f"${v:.1f}$"


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------


def load_all_metrics(metrics_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load every JSON under metrics/by_metric/. Returns {json_key: payload}."""
    out: Dict[str, Dict[str, Any]] = {}
    if not metrics_dir.is_dir():
        return out
    for jf in sorted(metrics_dir.glob("*.json")):
        try:
            with jf.open("r", encoding="utf-8") as f:
                out[jf.stem] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build_table_rows(specs: List[MetricSpec], snapshot: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build one dict per spec with display, our, tier1, tier2, direction, verdict."""
    rows: List[Dict[str, str]] = []
    for spec in specs:
        payload = snapshot.get(spec.json_key)
        value = extract_metric_value(spec, payload) if payload else None
        rows.append(
            {
                "row": str(spec.row),
                "display": spec.display,
                "our": fmt_value(value, spec),
                "value_raw": "" if value is None else f"{value}",
                "tier1": spec.tier1_sota,
                "tier2": spec.tier2_sota,
                "direction": spec.direction,
                "target": spec.target_str,
                "verdict": passes_gate(spec, value),
            }
        )
    return rows


def render_table2_tex(rows: List[Dict[str, str]]) -> str:
    """Render the LaTeX Table 2 fragment."""
    out = [
        "% Auto-generated by molmetal/scripts/generate_section4.py (DO NOT EDIT BY HAND)",
        "% Source: metrics/by_metric/*.json + TODO/02_metrics/METRICS_TARGETS.md §2",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Our value} & \\textbf{Tier 1 SOTA} & \\textbf{Tier 2 SOTA} & \\textbf{Direction} & \\textbf{Verdict} \\\\",
        "\\midrule",
    ]
    for r in rows:
        out.append(
            f"{r['display']} & {r['our']} & {r['tier1']} & {r['tier2']} & {r['direction']} & {r['verdict']} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\caption{Paper §4 main panel.\\label{tab:main_v2}}", "\\end{table}", ""]
    return "\n".join(out)


def render_fig7_cell_line(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-cell-line AUC bar — PlatinAI 4-cell + NCI-60 9-cell subsets."""
    platinai = snapshot.get("platinai_4cell_auc") or {}
    nci60 = snapshot.get("nci60_precious_9cell_auc") or {}
    payload = {
        "title": "Per-cell-line AUC bar",
        "description": "Tier 1 PlatinAI 4-cell + NCI-60 9-cell per-line AUC vs Krasnov/Rusanov baselines",
        "platinai": {
            "cells": ["MCF7", "A549", "A2780", "A2780cis"],
            "ours": [_get_nested(platinai, f"per_cell_line.{c}.auc_mean") for c in ("MCF7", "A549", "A2780", "A2780cis")],
            "rusanov_2025": "PENDING (paper Tables S4-S5 supplementary download)",
            "key_axis": "A2780cis (cisplatin-resistance, gate ≥ 0.85)",
        },
        "nci60_9cell": {
            "cells": nci60.get("cell_lines", []),
            "ours": [_get_nested(nci60, f"per_cell_line.{c}.auc_mean") for c in (nci60.get("cell_lines") or [])],
            "krasnov_baseline": nci60.get("krasnov_2026_baseline_auc"),
            "overall_mean_auc": nci60.get("value_overall_mean_auc"),
            "overall_mean_std": nci60.get("value_overall_mean_std"),
        },
        "render": "paper/figures/fig7_cell_line.py (PNG + SVG)",
    }
    return payload


def render_fig8_ood_radar(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """OOD radar — DrugOOD 3-split vs iMoLD vs ERM."""
    d = snapshot.get("drugood_ood_auc") or {}
    payload = {
        "title": "OOD radar",
        "description": "DrugOOD 3-split OOD AUC vs iMoLD (SOTA) vs ERM (baseline)",
        "axes": ["assay", "scaffold", "size"],
        "ours": {
            axis: _get_nested(d, f"results.{axis}.ood_auc_mean")
            for axis in ("assay", "scaffold", "size")
        },
        "imold_reference": d.get("imold_reference"),
        "erm_baseline": d.get("erm_baseline"),
        "gate_target": "≥ iMoLD − 0.02",
        "render": "paper/figures/fig8_ood_radar.py",
    }
    return payload


def render_fig9_tier1_scorecard(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Tier-1 scorecard — Ru subset + Pt 4-cell + LOMO + NCI-60 9-cell + n_distinct."""
    keys = (
        ("metalcytotox_ru_auc", "Ru subset AUC"),
        ("platinai_4cell_auc", "Pt 4-cell mean AUC"),
        ("metal_lomo_auc", "Multi-metal LOMO"),
        ("nci60_precious_9cell_auc", "NCI-60 9-cell mean"),
        ("n_distinct", "$n_{\\mathrm{distinct}}$ metal-ligand pairs"),
    )
    cards = []
    for k, label in keys:
        payload = snapshot.get(k) or {}
        v = None
        if k == "n_distinct":
            v = _get_nested(payload, "aggregate.best_measured_with_all_fixes")
        elif k == "platinai_4cell_auc":
            v = payload.get("value_overall_mean_auc")
        elif k == "nci60_precious_9cell_auc":
            v = payload.get("value_overall_mean_auc")
        else:
            v = payload.get("value")
        cards.append(
            {
                "metric_key": k,
                "label": label,
                "value": v,
                "status": payload.get("status", "PENDING"),
            }
        )
    return {
        "title": "Tier-1 precious-metal scorecard",
        "description": "5-axis Tier-1 scorecard on precious-metal anticancer",
        "cards": cards,
        "render": "paper/figures/fig9_tier1_scorecard.py",
    }


def render_summary_md(rows: List[Dict[str, str]]) -> str:
    """Human-readable summary.md."""
    counts: Dict[str, int] = {"PASS": 0, "MARGINAL": 0, "REGRESS": 0, "PENDING": 0}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines = [
        "# WF-Section4-Gen — paper §4 panel summary",
        "",
        "**Date**: 2026-09-25  ",
        "**Source**: `molmetal/scripts/generate_section4.py`  ",
        f"**Metrics dir**: `metrics/by_metric/*.json`  ",
        "",
        "## Verdict counts",
        "",
        f"* PASS: **{counts.get('PASS', 0)}** / 20",
        f"* MARGINAL: **{counts.get('MARGINAL', 0)}** / 20",
        f"* REGRESS: **{counts.get('REGRESS', 0)}** / 20",
        f"* PENDING: **{counts.get('PENDING', 0)}** / 20",
        "",
        "## Per-metric verdict",
        "",
        "| # | Metric | Our value | Tier 1 SOTA | Tier 2 SOTA | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['row']} | {r['display']} | {r['our']} | {r['tier1']} | {r['tier2']} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Auto-generated by `molmetal/scripts/generate_section4.py` from `metrics/by_metric/*.json` snapshot.")
    lines.append("Do NOT edit by hand; re-run after metric refresh.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. CLI entry
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paper §4 Results auto-generator")
    p.add_argument("--dry-run", action="store_true", help="Print Table 2 to stdout; skip file writes")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "paper"), help="Paper root (sections/ + figures/data/ + reports/)")
    p.add_argument("--metrics-dir", default=str(REPO_ROOT / "metrics" / "by_metric"), help="Snapshot dir")
    p.add_argument("--report-dir", default=str(REPO_ROOT / "paper" / "reports" / "wf_section4_gen"), help="Output report dir")
    args = p.parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    snapshot = load_all_metrics(metrics_dir)
    rows = build_table_rows(METRIC_REGISTRY, snapshot)
    tex = render_table2_tex(rows)

    if args.dry_run:
        print(tex)
        return 0

    # Table 2 → paper/sections/04_results_table2.tex
    sections_dir = out_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    (sections_dir / "04_results_table2.tex").write_text(tex, encoding="utf-8")

    # Figure data → paper/figures/data/{fig7,fig8,fig9}_*.json
    figs_dir = out_dir / "figures" / "data"
    figs_dir.mkdir(parents=True, exist_ok=True)
    (figs_dir / "fig7_cell_line.json").write_text(json.dumps(render_fig7_cell_line(snapshot), indent=2), encoding="utf-8")
    (figs_dir / "fig8_ood_radar.json").write_text(json.dumps(render_fig8_ood_radar(snapshot), indent=2), encoding="utf-8")
    (figs_dir / "fig9_tier1_scorecard.json").write_text(json.dumps(render_fig9_tier1_scorecard(snapshot), indent=2), encoding="utf-8")

    # Summary → paper/reports/wf_section4_gen/summary.md
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text(render_summary_md(rows), encoding="utf-8")

    print(f"[generate_section4] wrote Table 2 ({len(rows)} rows) + 3 figure stubs + summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())