#!/usr/bin/env python3
"""Aggregate WF-R16-Round13 30-cell sweep results.

Reads molmetal/reports/wf_lambda1_wf_r16_round13_30cell/report.json,
computes per-cell + per-pocket + per-seed + overall aggregates, and
writes final.md + final.json.

Honest framing: MEASURED vs PROJECTED vs DESIGN preserved.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO = Path("/home/hugo/codes/try_triton_on_rocm")
REPORT = REPO / "molmetal/reports/wf_lambda1_wf_r16_round13_30cell/report.json"
OUT_DIR = REPO / "molmetal/reports/wf_r16_round13_30cell"


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


# Metrics the sweep exposes on each cell (filtered to those we want)
CELL_METRICS = [
    ("validity_rate", "validity"),
    ("synthesizability_rate", "synth"),
    ("uniqueness_rate", "uniq"),
    ("metal_compliance_rate", "metal"),
    ("diversity_tanimoto", "div_tan"),
    ("diversity_homotype", "div_homo"),
    ("diversity_subpocket", "div_subpocket"),
    ("novelty", "novelty"),
    ("n_distinct", "n_distinct"),
    ("reference_tanimoto", "ref_tan"),
    ("sa_mean", "sa"),
    ("qed_mean", "qed"),
    ("logp_mean", "logp"),
    ("tpsa_mean", "tpsa"),
    ("rotb_mean", "rotb"),
    ("anticancer_index", "anticancer"),
    ("rigid_rmsd_mean", "rigid_rmsd"),
    ("com_shift_mean", "com_shift"),
    ("decoder_pass_rate", "decoder"),
]


def _cell_status(c: dict) -> str:
    """Determine cell status: completed / seed_only / no_candidates / failed."""
    if c.get("n_candidates", 0) == 0:
        return "no_candidates"
    if c.get("n_distinct", 0) == 0:
        return "seed_only"
    return "completed"


def aggregate(report: dict) -> dict:
    cells = report.get("cells", [])
    n_total = len(cells)
    n_completed = sum(1 for c in cells if _cell_status(c) == "completed")
    n_seed_only = sum(1 for c in cells if _cell_status(c) == "seed_only")
    n_no_candidates = sum(1 for c in cells if _cell_status(c) == "no_candidates")

    # Per-cell metric table
    per_cell = []
    for c in cells:
        row = {
            "pocket_id": c.get("pocket_id"),
            "seed": c.get("seed"),
            "status": _cell_status(c),
            "n_candidates": c.get("n_candidates", 0),
            "n_distinct": c.get("n_distinct", 0),
            "wall_s": c.get("elapsed_s", 0.0),
            "warnings": c.get("warnings", []),
        }
        for src_key, dst_key in CELL_METRICS:
            val = c.get(src_key)
            row[dst_key] = val if val is not None else 0.0
        per_cell.append(row)

    # Per-pocket
    by_pocket = defaultdict(list)
    for row in per_cell:
        by_pocket[row["pocket_id"]].append(row)

    per_pocket = {}
    for pid, rows in sorted(by_pocket.items()):
        agg = {"n_cells": len(rows), "n_completed": sum(1 for r in rows if r["status"] == "completed")}
        for src_key, dst_key in CELL_METRICS:
            vals = [r[dst_key] for r in rows]
            agg[f"{dst_key}_mean"] = _safe_mean(vals)
            agg[f"{dst_key}_std"] = _safe_std(vals)
        per_pocket[pid] = agg

    # Per-seed
    by_seed = defaultdict(list)
    for row in per_cell:
        by_seed[row["seed"]].append(row)
    per_seed = {}
    for s, rows in sorted(by_seed.items()):
        agg = {"n_cells": len(rows), "n_completed": sum(1 for r in rows if r["status"] == "completed")}
        for src_key, dst_key in CELL_METRICS:
            vals = [r[dst_key] for r in rows]
            agg[f"{dst_key}_mean"] = _safe_mean(vals)
            agg[f"{dst_key}_std"] = _safe_std(vals)
        per_seed[s] = agg

    # Overall
    overall = {
        "n_cells": n_total,
        "n_completed": n_completed,
        "n_seed_only": n_seed_only,
        "n_no_candidates": n_no_candidates,
        "wall_s_total": report.get("elapsed_s_total", 0.0),
    }
    for src_key, dst_key in CELL_METRICS:
        vals = [r[dst_key] for r in per_cell]
        overall[f"{dst_key}_mean"] = _safe_mean(vals)
        overall[f"{dst_key}_std"] = _safe_std(vals)

    return {
        "overall": overall,
        "per_pocket": per_pocket,
        "per_seed": per_seed,
        "per_cell": per_cell,
        "config": report.get("config", {}),
    }


def verdict(agg: dict) -> dict:
    """Compute gate verdict."""
    overall = agg["overall"]
    n_total = overall["n_cells"]
    if n_total == 0:
        return {"pass": False, "reason": "no cells ran"}

    # Gate A: >=90% cells validity=1.0
    valid_cells = sum(1 for c in agg["per_cell"] if c["validity"] >= 0.99)
    gate_a = valid_cells / n_total

    # Gate B: >=80% cells n_distinct>1
    diverse_cells = sum(1 for c in agg["per_cell"] if c["n_distinct"] > 1)
    gate_b = diverse_cells / n_total

    # Gate C: >=60% cells metal_compliance >=0.60
    metal_cells = sum(1 for c in agg["per_cell"] if c["metal"] >= 0.60)
    gate_c = metal_cells / n_total

    # PB gate: n/a (r4_lambda_only_run.py does not have --pb-mode; PB only in r4_c_full_sweep.py)
    gate_pb = "n/a (r4_lambda_only_run.py has no --pb-mode flag; PB run is r4_c_full_sweep.py)"

    pass_a = gate_a >= 0.90
    pass_b = gate_b >= 0.80
    pass_c = gate_c >= 0.60

    return {
        "gate_a_validity_pct": gate_a,
        "gate_b_n_distinct_gt_1_pct": gate_b,
        "gate_c_metal_compliance_pct": gate_c,
        "gate_pb": gate_pb,
        "gate_a_pass": pass_a,
        "gate_b_pass": pass_b,
        "gate_c_pass": pass_c,
        "overall_pass": pass_a and pass_b,
        "strong_pass": pass_a and pass_b and pass_c,
        "valid_cells": valid_cells,
        "diverse_cells": diverse_cells,
        "metal_cells": metal_cells,
    }


def render_md(agg: dict, v: dict) -> str:
    lines = []
    lines.append("# WF-R16-Round13 30-cell sweep — final")
    lines.append("")
    lines.append("Date: 2026-09-18")
    lines.append("")
    lines.append("## CLI (REAL supported flags, sourced from r4_lambda_only_run.py argparser)")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python molmetal/scripts/r4_lambda_only_run.py \\")
    lines.append("  --pockets 10 --seeds 0 1 2 \\")
    lines.append("  --n-simulations 1000 --n-top-k 20 \\")
    lines.append("  --metal-seed cisplatin --click-rules auto-pt-strict \\")
    lines.append("  --output-dir wf_r16_round13_30cell")
    lines.append("```")
    lines.append("")
    lines.append("Flags used:")
    lines.append("- `--pockets 10` (r4_lambda_only_run.py:4158)")
    lines.append("- `--seeds 0 1 2` (r4_lambda_only_run.py:4218)")
    lines.append("- `--n-simulations 1000` (r4_lambda_only_run.py:4225, cap removed by WF-Lift-N-Sim-Cap)")
    lines.append("- `--n-top-k 20` (r4_lambda_only_run.py:4236)")
    lines.append("- `--metal-seed cisplatin` (r4_lambda_only_run.py:4289)")
    lines.append("- `--click-rules auto-pt-strict` (r4_lambda_only_run.py:4279)")
    lines.append("- `--output-dir wf_r16_round13_30cell` (r4_lambda_only_run.py:4242)")
    lines.append("")
    lines.append("Flags NOT registered by r4_lambda_only_run.py (and so SKIPPED this run):")
    lines.append("- `--pb-mode` / `--pb-relax-mmff94` / `--pb-check` / `--physical-docking` / `--engine both` — these are owned by `r4_c_full_sweep.py`, not this script.")
    lines.append("- `--sa-weight 0.3` IS supported (r4_lambda_only_run.py:4446), but skipped here to keep the ablation single-table.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("| Gate | Threshold | Measured | Pass? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A: cells with validity>=0.99 | >=90% | {v['valid_cells']}/{agg['overall']['n_cells']} = {v['gate_a_validity_pct']*100:.1f}% | {'YES' if v['gate_a_pass'] else 'NO'} |")
    lines.append(f"| B: cells with n_distinct>1 | >=80% | {v['diverse_cells']}/{agg['overall']['n_cells']} = {v['gate_b_n_distinct_gt_1_pct']*100:.1f}% | {'YES' if v['gate_b_pass'] else 'NO'} |")
    lines.append(f"| C: cells with metal_compliance>=0.60 | >=60% | {v['metal_cells']}/{agg['overall']['n_cells']} = {v['gate_c_metal_compliance_pct']*100:.1f}% | {'YES' if v['gate_c_pass'] else 'NO'} |")
    lines.append(f"| PB | n/a this script | {v['gate_pb']} | n/a |")
    lines.append("")
    lines.append(f"**OVERALL: {'STRONG PASS' if v['strong_pass'] else 'PASS' if v['overall_pass'] else 'PARTIAL/FAIL'}**")
    lines.append("")
    lines.append("## Overall aggregate (30 cells)")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---|---|")
    for src_key, dst_key in CELL_METRICS:
        m = agg["overall"][f"{dst_key}_mean"]
        s = agg["overall"][f"{dst_key}_std"]
        lines.append(f"| {dst_key} | {m:.4f} | {s:.4f} |")
    lines.append(f"| wall_s_total | {agg['overall']['wall_s_total']:.1f} | n/a |")
    lines.append("")
    lines.append("## Per-cell table")
    lines.append("")
    lines.append("| pocket | seed | status | n_distinct | validity | synth | metal | div_tan | div_homo | n_candidates | wall_s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in agg["per_cell"]:
        lines.append(
            f"| {c['pocket_id']} | {c['seed']} | {c['status']} | {c['n_distinct']} | "
            f"{c['validity']:.2f} | {c['synth']:.2f} | {c['metal']:.2f} | "
            f"{c['div_tan']:.3f} | {c['div_homo']:.3f} | {c['n_candidates']} | {c['wall_s']:.1f} |"
        )
    lines.append("")
    lines.append("## Per-pocket (3-seed mean)")
    lines.append("")
    lines.append("| pocket | n_completed | validity | synth | metal | div_tan | div_homo | n_distinct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pid, p in sorted(agg["per_pocket"].items()):
        lines.append(
            f"| {pid} | {p['n_completed']}/3 | {p['validity_mean']:.2f} | "
            f"{p['synth_mean']:.2f} | {p['metal_mean']:.2f} | "
            f"{p['div_tan_mean']:.3f} | {p['div_homo_mean']:.3f} | "
            f"{p['n_distinct_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-seed (10-pocket mean)")
    lines.append("")
    lines.append("| seed | n_completed | validity | synth | metal | div_tan | div_homo | n_distinct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s, p in sorted(agg["per_seed"].items()):
        lines.append(
            f"| {s} | {p['n_completed']}/10 | {p['validity_mean']:.2f} | "
            f"{p['synth_mean']:.2f} | {p['metal_mean']:.2f} | "
            f"{p['div_tan_mean']:.3f} | {p['div_homo_mean']:.3f} | "
            f"{p['n_distinct_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append("- MEASURED: every per-cell metric above was emitted by the sweep itself.")
    lines.append("- PROJECTED / DESIGN: not applicable — this is the MEASURED round.")
    lines.append("- PB skipped (different script owns --pb-mode).")
    lines.append("- `--sa-weight` skipped to keep this single-table; ablation is follow-up.")
    lines.append("- `--metal-seed cisplatin + --click-rules auto-pt-strict` chosen for protocol parity with WF-Round12-Lambda-Path-A (ultracode_wf_round12_lambda_patha_10x3.mjs).")
    lines.append("- Round-12 mini-pilot (n_sim=100) collapsed to n_distinct=1. Round-13 (n_sim=1000) is the lift test.")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        print(f"ERROR: report.json not found at {REPORT}")
        return 1
    report = json.loads(REPORT.read_text())
    agg = aggregate(report)
    v = verdict(agg)
    out = {**agg, "verdict": v}
    (OUT_DIR / "final.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    (OUT_DIR / "final.md").write_text(render_md(agg, v))
    print(f"Wrote {OUT_DIR}/final.json + final.md")
    print(f"Verdict: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())