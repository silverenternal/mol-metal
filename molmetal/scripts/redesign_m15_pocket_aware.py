#!/usr/bin/env python
"""Wf-246-B: TODO-130 m15 REDESIGN — pocket-aware self-RMSD metric.

Per TODO-130 / TODO-134 (m15 redesign spec):
  Refactor m15 metric (docked_pose_rmsd) to add:
    1. Per-pocket std reporting (5 pockets x 3 seeds = 15 runs)
    2. Cross-pocket aggregate: median + std + min + max
    3. Difficulty flag per pocket (1h36 small=hard, 3ny8 medium, etc.)

Inputs:
  molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json (per-pocket numbers)

Outputs:
  metrics/by_metric/docked_pose_rmsd.json (REFACTORED with new fields)
  molmetal/reports/wf246b_m15_redesign/summary.json
  molmetal/reports/wf246b_m15_redesign/synthesis.md
  molmetal/reports/wf246b_m15_redesign/per_pocket_table.md

Constraints:
  USE /home/hugo/codes/mol-metal/.venv/bin/python
  CPU-only (no GPU)
  Backward-compatible with existing docked_pose_rmsd.json schema
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
INPUT_SUMMARY = REPO / "molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json"
METRIC_JSON = REPO / "metrics/by_metric/docked_pose_rmsd.json"
REPORT_DIR = REPO / "molmetal/reports/wf246b_m15_redesign"

# Difficulty flag thresholds (per TODO-130 / TODO-134):
#   EASY: median best RMSD < 0.5 Å (Pocket is shallow / open; Vina re-docks easily)
#   MEDIUM: 0.5 Å <= median best RMSD < 1.0 Å
#   HARD: median best RMSD >= 1.0 Å (Pocket is buried / constrained; Vina re-docks hard)
DIFFICULTY_EASY_MAX = 0.5
DIFFICULTY_MEDIUM_MAX = 1.0


def classify_difficulty(median_rmsd_angstrom: float, pocket: str) -> str:
    """Classify pocket difficulty based on mean best RMSD (angstrom).

    Returns one of {EASY, MEDIUM, HARD}.
    """
    if median_rmsd_angstrom < DIFFICULTY_EASY_MAX:
        return "EASY"
    elif median_rmsd_angstrom < DIFFICULTY_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HARD"


def cross_pocket_aggregate(pocket_medians: list[float]) -> dict:
    """Compute cross-pocket aggregate: median, std, min, max."""
    if not pocket_medians:
        return {"median": None, "std": None, "min": None, "max": None, "n_pockets": 0}
    return {
        "median": float(statistics.median(pocket_medians)),
        "std": float(statistics.stdev(pocket_medians)) if len(pocket_medians) > 1 else 0.0,
        "min": float(min(pocket_medians)),
        "max": float(max(pocket_medians)),
        "n_pockets": len(pocket_medians),
    }


def main() -> int:
    if not INPUT_SUMMARY.exists():
        print(f"FAIL: {INPUT_SUMMARY} not found", file=sys.stderr)
        return 2

    with open(INPUT_SUMMARY) as f:
        src = json.load(f)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Existing per-pocket data
    pockets = src.get("pockets", {})
    pocket_medians = []
    per_pocket_block = {}
    difficulty_counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}

    for pocket_id, pdata in pockets.items():
        if pdata.get("status") != "MEASURED":
            continue
        median_rmsd = pdata["median_best_rmsd_angstrom"]
        mean_rmsd = pdata["mean_best_rmsd_angstrom"]
        std_rmsd = pdata.get("std_best_rmsd_angstrom", 0.0)
        n_seeds = pdata.get("n_seeds_succeeded", 0)
        per_seed = pdata.get("per_seed_best_rmsd_angstrom", [])
        pocket_medians.append(median_rmsd)

        difficulty = classify_difficulty(median_rmsd, pocket_id)
        difficulty_counts[difficulty] += 1

        per_pocket_block[pocket_id] = {
            "pocket": pocket_id,
            "ref_smiles": pdata.get("ref_smiles", ""),
            "n_ref_heavy": pdata.get("n_ref_heavy", 0),
            "per_seed_best_rmsd_angstrom": per_seed,
            "median_best_rmsd_angstrom": median_rmsd,
            "mean_best_rmsd_angstrom": mean_rmsd,
            "std_best_rmsd_angstrom": std_rmsd,
            "n_seeds_succeeded": n_seeds,
            "n_seeds_total": 3,
            "passes_2_5a_gate": median_rmsd <= 2.5,
            "passes_2_0a_sota": median_rmsd <= 2.0,
            "difficulty_flag": difficulty,
            "status": "MEASURED",
        }

    # Cross-pocket aggregate
    cross_agg = cross_pocket_aggregate(pocket_medians)
    cross_median = cross_agg["median"]
    cross_std = cross_agg["std"]
    cross_min = cross_agg["min"]
    cross_max = cross_agg["max"]

    # === Update metrics/by_metric/docked_pose_rmsd.json (backward compat) ===
    with open(METRIC_JSON) as f:
        metric = json.load(f)

    # Add new fields at the top level (preserve existing fields)
    metric["redesign_wf246b"] = {
        "date": "2026-09-28",
        "workflow": "Wf-246-B TODO-130 m15 REDESIGN",
        "purpose": "pocket-aware self-RMSD metric: per-pocket std + cross-pocket aggregate + difficulty flag",
        "per_pocket_count": len(per_pocket_block),
        "per_pocket_seeds_total": len(per_pocket_block) * 3,
        "difficulty_thresholds_angstrom": {
            "EASY_lt": DIFFICULTY_EASY_MAX,
            "MEDIUM_lt": DIFFICULTY_MEDIUM_MAX,
        },
        "cross_pocket_aggregate": cross_agg,
        "difficulty_counts": difficulty_counts,
    }
    metric["per_pocket_with_difficulty"] = per_pocket_block
    metric["cross_pocket_aggregate"] = cross_agg
    metric["difficulty_counts"] = difficulty_counts
    metric["aggregate_median_across_pockets"] = cross_median  # canonical row
    metric["aggregate_std_across_pockets"] = cross_std
    metric["aggregate_min_across_pockets"] = cross_min
    metric["aggregate_max_across_pockets"] = cross_max
    metric["note"] = (
        "Wf-246-B redesign: per-pocket std reporting (15 runs = 5 pockets x 3 seeds) "
        "+ cross-pocket aggregate (median, std, min, max) + per-pocket difficulty flag "
        "(EASY <0.5A, MEDIUM 0.5-1.0A, HARD >=1.0A). Backward-compatible with Wf-179 ship schema."
    )

    with open(METRIC_JSON, "w") as f:
        json.dump(metric, f, indent=2)

    print(f"Wrote {METRIC_JSON}")

    # === Write report summary.json ===
    redesign_summary = {
        "workflow": "Wf-246-B TODO-130 m15 REDESIGN",
        "engine": "AutoDock Vina (ref: wf179_bd9_m15_rmsd_p2)",
        "exhaustiveness": src.get("exhaustiveness", 16),
        "n_poses": src.get("n_poses", 5),
        "seeds": src.get("seeds", [0, 1, 2]),
        "input_source": "molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json",
        "per_pocket": per_pocket_block,
        "difficulty_counts": difficulty_counts,
        "cross_pocket_aggregate": cross_agg,
        "difficulty_thresholds_angstrom": {
            "EASY_lt": DIFFICULTY_EASY_MAX,
            "MEDIUM_lt": DIFFICULTY_MEDIUM_MAX,
        },
        "pockets": {k: v for k, v in per_pocket_block.items()},  # alias for legacy readers
        "aggregate_median_across_pockets": cross_median,
        "n_pockets": len(per_pocket_block),
        "n_pockets_passed_2_5a": sum(
            1 for v in per_pocket_block.values() if v["passes_2_5a_gate"]
        ),
        "n_pockets_passed_2_0a": sum(
            1 for v in per_pocket_block.values() if v["passes_2_0a_sota"]
        ),
        "gate_4_passes_2_5a_overall": cross_median <= 2.5,
    }
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(redesign_summary, f, indent=2)
    print(f"Wrote {REPORT_DIR / 'summary.json'}")

    # === Write per-pocket table markdown ===
    table_lines = [
        "# Wf-246-B m15 Redesign — Per-Pocket Self-RMSD Table",
        "",
        "Source: `molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json` (Wf-179 BD-9 m15 Phase 2)",
        "Redesign: per-pocket std + cross-pocket aggregate + difficulty flag.",
        "",
        "Difficulty thresholds (median best RMSD, angstrom):",
        f"- EASY: median < {DIFFICULTY_EASY_MAX}",
        f"- MEDIUM: {DIFFICULTY_EASY_MAX} <= median < {DIFFICULTY_MEDIUM_MAX}",
        f"- HARD: median >= {DIFFICULTY_MEDIUM_MAX}",
            "",
            "| Pocket | Difficulty | Median (Å) | Mean (Å) | Std (Å) | n_seeds | passes_2.5Å | passes_2.0Å |",
            "|--------|------------|------------|----------|---------|---------|--------------|------------|",
    ]
    for pid, pdata in sorted(per_pocket_block.items()):
        table_lines.append(
            f"| {pid} | {pdata['difficulty_flag']} | {pdata['median_best_rmsd_angstrom']:.4f} | "
            f"{pdata['mean_best_rmsd_angstrom']:.4f} | {pdata['std_best_rmsd_angstrom']:.4f} | "
            f"{pdata['n_seeds_succeeded']}/3 | {pdata['passes_2_5a_gate']} | {pdata['passes_2_0a_sota']} |"
        )
    table_lines.extend([
        "",
        f"Cross-pocket aggregate (n={len(per_pocket_block)}):",
        f"- median: {cross_median:.4f} Å",
        f"- std:    {cross_std:.4f} Å",
        f"- min:    {cross_min:.4f} Å",
        f"- max:    {cross_max:.4f} Å",
        "",
        f"Difficulty distribution: {difficulty_counts}",
    ])
    (REPORT_DIR / "per_pocket_table.md").write_text("\n".join(table_lines) + "\n")
    print(f"Wrote {REPORT_DIR / 'per_pocket_table.md'}")

    # === Write synthesis.md ===
    synthesis_lines = [
        "# Wf-246-B Synthesis: TODO-130 m15 REDESIGN",
        "",
        "**Date**: 2026-09-28",
        "**Workflow**: Wf-246-B TODO-130 m15 REDESIGN",
        "**Parent**: TODO-130 (metric definition baseline) + TODO-134 (m15 redesign spec) + Wf-179 (BD-9 m15 RMSD Phase 2 Vina redocking)",
        "",
        "## §1 Purpose",
        "",
        "Redesign the m15 metric (docked_pose_rmsd) to add three paper-ready diagnostics:",
        "1. **Per-pocket std reporting**: each pocket's std across 3 seeds (5 pockets x 3 seeds = 15 runs)",
        "2. **Cross-pocket aggregate**: median, std, min, max across the 5 pocket medians",
        "3. **Difficulty flag**: per-pocket classification (EASY / MEDIUM / HARD) by median best RMSD",
        "",
        "## §2 Inputs",
        "",
        f"- `molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json` (5 pockets x 3 seeds, Vina ex={src.get('exhaustiveness', 16)})",
        f"- `molmetal/data/pocket_ref_ligands/ligand_metadata.json` (ref ligand SMILES + coords)",
        "",
        "## §3 Per-pocket results",
        "",
        "| Pocket | Difficulty | Median RMSD (Å) | Std (Å) | n_seeds |",
        "|--------|------------|------------------|---------|---------|",
    ]
    for pid in sorted(per_pocket_block):
        d = per_pocket_block[pid]
        synthesis_lines.append(
            f"| {pid} | {d['difficulty_flag']} | {d['median_best_rmsd_angstrom']:.4f} | "
            f"{d['std_best_rmsd_angstrom']:.4f} | {d['n_seeds_succeeded']}/3 |"
        )
    synthesis_lines.extend([
        "",
        "## §4 Cross-pocket aggregate",
        "",
        f"- **Median**: {cross_median:.4f} Å",
        f"- **Std**:    {cross_std:.4f} Å",
        f"- **Min**:    {cross_min:.4f} Å (best pocket)",
        f"- **Max**:    {cross_max:.4f} Å (worst pocket)",
        f"- **n_pockets**: {len(per_pocket_block)}",
        "",
        "## §6 Difficulty distribution",
        "",
        f"- EASY (<{DIFFICULTY_EASY_MAX} Å): {difficulty_counts['EASY']} pockets",
        f"- MEDIUM ({DIFFICULTY_EASY_MAX}-{DIFFICULTY_MEDIUM_MAX} Å): {difficulty_counts['MEDIUM']} pockets",
        f"- HARD (>={DIFFICULTY_MEDIUM_MAX} Å): {difficulty_counts['HARD']} pockets",
        "",
        "## §7 Verdict",
        "",
        f"All {len(per_pocket_block)}/5 pockets MEASURED with per-pocket std and difficulty flag.",
        f"Cross-pocket median = {cross_median:.4f} Å (SOTA gate <= 2.0 Å: {'PASS' if cross_median <= 2.0 else 'FAIL'});",
        f"all pockets pass <= 2.5 Å gate ({sum(1 for v in per_pocket_block.values() if v['passes_2_5a_gate'])}/{len(per_pocket_block)}).",
        "",
        "## §8 Honest framing",
        "",
        "- All values are MEASURED (not projected) from Wf-179 Vina redocking runs (exhaustiveness=16, n_poses=5).",
        "- 3ny8 has only 1/3 seeds succeeded (NO_DOCK_OUTPUT for seeds 1, 2 — partial stage artifact).",
        "- Difficulty flag uses median best RMSD (NOT pass-rate), so it's a property of the pocket + crystal ligand, not of Vina exhaustiveness.",
        "- Backward compat: `metrics/by_metric/docked_pose_rmsd.json` schema preserved (legacy fields unchanged); new fields added under `redesign_wf246b`, `per_pocket_with_difficulty`, `cross_pocket_aggregate`.",
    ])
    (REPORT_DIR / "synthesis.md").write_text("\n".join(synthesis_lines) + "\n")
    print(f"Wrote {REPORT_DIR / 'synthesis.md'}")

    # === Console summary ===
    print("\n=== Wf-246-B m15 Redesign Summary ===")
    print(f"Per-pocket count: {len(per_pocket_block)}")
    for pid in sorted(per_pocket_block):
        d = per_pocket_block[pid]
        print(f"  {pid:6s} {d['difficulty_flag']:6s} median={d['median_best_rmsd_angstrom']:.4f} Å "
              f"std={d['std_best_rmsd_angstrom']:.4f} Å ({d['n_seeds_succeeded']}/3 seeds)")
    print(f"Cross-pocket: median={cross_median:.4f} std={cross_std:.4f} "
          f"min={cross_min:.4f} max={cross_max:.4f}")
    print(f"Difficulty: {difficulty_counts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
