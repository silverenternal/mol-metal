#!/usr/bin/env python3
"""Refresh metrics/by_metric/{hilbert_4cell_r2,metalcytotox_combined_r2}.json
with the Wf-127 Phase 4 (full E2E retrain with L2 + scale-guard) measured values.

Reads:
  molmetal/reports/wf127_p4_platinai_4cell/{cell_line}/per_seed_metrics.json
  molmetal/reports/wf127_p4_metalcytotox_5metal/{metal}/per_seed_metrics.json

Writes:
  metrics/by_metric/hilbert_4cell_r2.json (REFRESHED)
  metrics/by_metric/metalcytotox_combined_r2.json (REFRESHED)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_METRICS_DIR = _REPO_ROOT / "metrics" / "by_metric"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _aggregate_4cell(out_dir: Path) -> Dict[str, Any]:
    """Read PlatinAI 4-cell per-cell-line per_seed_metrics.json and aggregate."""
    cell_lines = ["mcf7", "a549", "a2780", "a2780cis"]
    per_cell_line: Dict[str, Dict[str, Any]] = {}
    cell_means: List[float] = []

    for cl in cell_lines:
        p = out_dir / cl / "per_seed_metrics.json"
        if not p.exists():
            print(f"  [warn] missing {p}", file=sys.stderr)
            continue
        data = _read_json(p)
        per_cell_line[cl] = {
            "combined_r2_mean": data.get("combined_r2_mean"),
            "combined_r2_std": data.get("combined_r2_std"),
            "hilbert_r2_mean": data.get("hilbert_r2_mean"),
            "hilbert_r2_std": data.get("hilbert_r2_std"),
            "astr_r2_mean": data.get("astr_r2_mean"),
            "astr_r2_std": data.get("astr_r2_std"),
            "n_seeds": data.get("n_seeds"),
            "source": str(p.relative_to(_REPO_ROOT)),
        }
        if data.get("combined_r2_mean") is not None:
            cell_means.append(float(data["combined_r2_mean"]))

    agg_mean = float(sum(cell_means) / len(cell_means)) if cell_means else None
    # std over cell-line means
    if len(cell_means) >= 2:
        mu = agg_mean
        var = sum((x - mu) ** 2 for x in cell_means) / (len(cell_means) - 1)
        agg_std = math.sqrt(var)
    else:
        agg_std = None

    return {
        "n_cell_lines": len(cell_means),
        "aggregate_combined_r2_mean": agg_mean,
        "aggregate_combined_r2_std": agg_std,
        "aggregate_cell_lines": cell_lines,
        "per_cell_line": per_cell_line,
    }


def _aggregate_5metal(out_dir: Path) -> Dict[str, Any]:
    """Read MetalCytoToxDB 5-metal per-metal per_seed_metrics.json and aggregate."""
    metals = ["Ru", "Ir", "Os", "Rh", "Re"]
    per_metal: Dict[str, Dict[str, Any]] = {}
    metal_means: List[float] = []

    for m in metals:
        p = out_dir / m / "per_seed_metrics.json"
        if not p.exists():
            print(f"  [warn] missing {p}", file=sys.stderr)
            continue
        data = _read_json(p)
        per_metal[m] = {
            "combined_r2_mean": data.get("combined_r2_mean"),
            "combined_r2_std": data.get("combined_r2_std"),
            "hilbert_r2_mean": data.get("hilbert_r2_mean"),
            "hilbert_r2_std": data.get("hilbert_r2_std"),
            "astr_r2_mean": data.get("astr_r2_mean"),
            "astr_r2_std": data.get("astr_r2_std"),
            "n_seeds": data.get("n_seeds"),
            "source": str(p.relative_to(_REPO_ROOT)),
        }
        if data.get("combined_r2_mean") is not None:
            metal_means.append(float(data["combined_r2_mean"]))

    agg_mean = float(sum(metal_means) / len(metal_means)) if metal_means else None
    if len(metal_means) >= 2:
        mu = agg_mean
        var = sum((x - mu) ** 2 for x in metal_means) / (len(metal_means) - 1)
        agg_std = math.sqrt(var)
    else:
        agg_std = None

    # gate: ≥ 0 for ≥ 1 metal
    n_pass = sum(1 for v in metal_means if v >= 0)
    n_total = len(metal_means)
    verdict = "PASS" if n_pass >= 1 else "FAIL"
    verdict_target = "combined R² >= 0 for >= 1 of 5 metals"

    return {
        "n_metals": len(metal_means),
        "aggregate_combined_r2_mean": agg_mean,
        "aggregate_combined_r2_std": agg_std,
        "aggregate_metals": metals,
        "per_metal": per_metal,
        "verdict": verdict,
        "verdict_target": verdict_target,
        "verdict_n_pass": n_pass,
        "verdict_n_total": n_total,
    }


def refresh_hilbert_4cell_r2(out_dir: Path) -> None:
    """Refresh metrics/by_metric/hilbert_4cell_r2.json with Wf-127 P4 result.

    Adds (does not overwrite) a ``wf127_p4_l2_scale_guard`` entry under
    a new top-level ``wf127_p4_guarded`` key; preserves existing Wf-122
    P3 / Wf-129 P3 entries.
    """
    target = _METRICS_DIR / "hilbert_4cell_r2.json"
    current = _read_json(target)
    agg = _aggregate_4cell(out_dir)

    # gate: ≥ 0 for ≥ 1 cell line
    cell_means = [
        v["combined_r2_mean"]
        for v in agg["per_cell_line"].values()
        if v.get("combined_r2_mean") is not None
    ]
    n_pass = sum(1 for v in cell_means if v >= 0)
    n_total = len(cell_means)
    verdict = "PASS" if n_pass >= 1 else "FAIL"
    verdict_target = "combined R² >= 0 for >= 1 of 4 cell-lines"

    refreshed: Dict[str, Any] = dict(current) if current else {}
    refreshed["metric"] = "hilbert_4cell_r2"
    # Keep existing description/scope/source stable; only update timestamp.
    refreshed["timestamp"] = "2026-09-27"
    refreshed["wf127_p4_guarded"] = {
        "description": (
            "Wf-127 P4 full E2E retrain with --use-l2-guard + "
            "--use-astr-scale-guard (l2_weight=0.01, scale_clip=(-10, 10))."
        ),
        "aggregate_combined_r2_mean": agg["aggregate_combined_r2_mean"],
        "aggregate_combined_r2_std": agg["aggregate_combined_r2_std"],
        "aggregate_cell_lines": agg["aggregate_cell_lines"],
        "per_cell_line": agg["per_cell_line"],
        "guards_applied": [
            "Wf-127 P2 L2 weight-decay guard (l2_weight=0.01 on Hilbert primal+dual)",
            "Wf-127 P3 ASTSR scale-mismatch guard (scale_clip=(-10, 10) on ASTSR signature)",
        ],
        "verdict": verdict,
        "verdict_target": verdict_target,
        "verdict_n_pass": n_pass,
        "verdict_n_total": n_total,
        "n_seeds_per_cell": 5,
        "n_folds_per_seed": 5,
        "epochs": 50,
        "device": "cpu",
        "wf_phase": "Wf-127 P4 full E2E retrain with L2 + scale-guard",
        "close_out_doc": "TODO/completed/127_l5_hilbert_astsr_rescue.md",
        "source_reports_dir": str(out_dir.relative_to(_REPO_ROOT)),
    }
    refreshed["wf_phase"] = (
        "Wf-127 P4 full E2E retrain with L2 + scale-guard (latest); "
        "see wf127_p4_guarded for the P4 MEASURED values; "
        "see wf129_p3_multiph_pll_guarded_l7_lift_caveat for Wf-129 P3 "
        "(more recent Multi-PH+PLL guarded retrain, R² = -0.078)"
    )
    _write_json(target, refreshed)
    print(f"[wf127-p4] refreshed {target}")


def refresh_metalcytotox_combined_r2(out_dir: Path) -> None:
    """Refresh metrics/by_metric/metalcytotox_combined_r2.json (add Wf-127 P4 method)."""
    target = _METRICS_DIR / "metalcytotox_combined_r2.json"
    current = _read_json(target)
    agg = _aggregate_5metal(out_dir)

    if current:
        refreshed = dict(current)
        # Add Wf-127 P4 method
        if "per_method" not in refreshed:
            refreshed["per_method"] = {}
        refreshed["per_method"]["wf127_p4_5metal_l2_scale_guard"] = {
            "overall_r2_mean": agg["aggregate_combined_r2_mean"],
            "overall_r2_std": agg["aggregate_combined_r2_std"],
            "n_seeds": sum(
                v.get("n_seeds") or 0 for v in agg["per_metal"].values()
            ),
            "n_metals": agg["n_metals"],
            "cohorts": agg["aggregate_metals"],
            "per_metal": {k: v["combined_r2_mean"] for k, v in agg["per_metal"].items()},
            "guards": "Wf-127 P2 L2 weight-decay + Wf-127 P3 ASTSR scale-clip (-10, 10)",
            "epochs": 30,
            "verdict": agg["verdict"],
            "verdict_target": agg["verdict_target"],
            "verdict_n_pass": agg["verdict_n_pass"],
            "verdict_n_total": agg["verdict_n_total"],
            "source": str(out_dir.relative_to(_REPO_ROOT)) + "/synthesis.md",
        }
        if "methods" in refreshed:
            if "wf127_p4_5metal_l2_scale_guard" not in refreshed["methods"]:
                refreshed["methods"].append("wf127_p4_5metal_l2_scale_guard")
        if "n_methods" in refreshed:
            refreshed["n_methods"] = len(refreshed["methods"])
        refreshed["timestamp"] = "2026-09-27"
        refreshed["description"] = (
            "MetalCytoToxDB 5-metal (Ru/Ir/Os/Rh/Re) regression R² aggregated from "
            "Wf-77 ASTSR baseline + Wf-79 3-way PLL experiments + Wf-128 P2 3-way REAL "
            "MetalCytoToxDB + Wf-127 P4 (L2 + scale-guard re-bench). Per-method "
            "breakdown kept as ``per_method``."
        )
        refreshed["wf_phase"] = (
            "Wf-87 Phase 2 promotion from Wf-77/79 MetalCytoTox outputs + "
            "Wf-128 P2 3-way REAL + Wf-127 P4 L2+scale-guard re-bench"
        )
    else:
        refreshed = {
            "metric": "metalcytotox_combined_r2",
            "per_method": {},
        }

    _write_json(target, refreshed)
    print(f"[wf127-p4] refreshed {target}")


def main(argv: List[str]) -> int:
    platinci_dir = Path(
        argv[1] if len(argv) > 1 else "molmetal/reports/wf127_p4_platinai_4cell"
    )
    metal_dir = Path(
        argv[2] if len(argv) > 2 else "molmetal/reports/wf127_p4_metalcytotox_5metal"
    )
    if not platinci_dir.is_absolute():
        platinci_dir = _REPO_ROOT / platinci_dir
    if not metal_dir.is_absolute():
        metal_dir = _REPO_ROOT / metal_dir

    refresh_hilbert_4cell_r2(platinci_dir)
    refresh_metalcytotox_combined_r2(metal_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
