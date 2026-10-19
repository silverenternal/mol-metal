#!/usr/bin/env python3
"""Rebuild the NCI-60 9-cell aggregate block from the real per-cell table.

``metrics/by_metric/nci60_precious_9cell_auc.json`` carried two tables that
disagreed: a real ``per_cell_line`` block (5 seeds, 9 cells, mean 0.7294) and a
``wf312b_nci60_smoke_aggregate`` block whose per-cell values were placeholders
— 7 of 9 cells were the identical 0.80072, and DU145 had been given that same
number purely to reach the 9-key floor the pre-flight gate enforces.

The gate validates the placeholder block's internal consistency, never its
agreement with the real table, so it passed.  Meanwhile every document in the
repo quoted 0.8007 as the measured m21 result.

This script makes the block honest: it is derived from ``per_cell_line``, so
the gate now validates real numbers, and it adds a ``resolvable_subset`` that
reports what the panel looks like once single-digit test sets are excluded.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf_a55_nci60_aggregate_refresh.py
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

DEFAULT_ARTIFACT = "metrics/by_metric/nci60_precious_9cell_auc.json"
BLOCK_KEY = "wf312b_nci60_smoke_aggregate"

# Below this many test molecules a per-cell AUC cannot support a claim: a
# 3-molecule test set can reach 1.0 by ranking one positive first.
MIN_RESOLVABLE_N_TEST = 10


def _simple_mean(values: List[float]) -> float:
    return sum(values) / len(values)


def build_block(per_cell: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Build the aggregate block from the real per-cell measurements."""
    breakdown = {
        cell: {
            "mean_auc": float(v["auc_mean"]),
            "std_auc": float(v.get("auc_std", 0.0)),
            "n_train": int(v.get("n_train", 0)),
            "n_test": int(v.get("n_test", 0)),
        }
        for cell, v in per_cell.items()
    }
    means = [b["mean_auc"] for b in breakdown.values()]

    highlight_cell = "HCT116" if "HCT116" in breakdown else max(
        breakdown, key=lambda c: breakdown[c]["mean_auc"]
    )
    hn_cell = min(breakdown, key=lambda c: breakdown[c]["mean_auc"])
    hn_value = breakdown[hn_cell]["mean_auc"]

    resolvable = {
        c: b for c, b in breakdown.items() if b["n_test"] >= MIN_RESOLVABLE_N_TEST
    }
    excluded = sorted(c for c in breakdown if c not in resolvable)
    weighted_den = sum(b["n_test"] for b in breakdown.values())
    weighted = (
        sum(b["mean_auc"] * b["n_test"] for b in breakdown.values()) / weighted_den
        if weighted_den
        else float("nan")
    )

    return {
        "per_cell_breakdown": breakdown,
        "overall_mean": _simple_mean(means),
        "overall_mean_tolerance": 1e-4,
        "highlight": {
            "cell": highlight_cell,
            "value": breakdown[highlight_cell]["mean_auc"],
        },
        "honest_negative": {
            "cell": hn_cell,
            "value": hn_value,
            "honest_negative_reason": (
                f"{hn_cell} MEASURED at {hn_value:.4f}, below the 0.5 "
                f"random-chance floor on n_test={breakdown[hn_cell]['n_test']}. "
                f"Recorded as an honest-negative; the panel mean is not "
                f"rescued by it."
            ),
            "badge": "honest-negative",
        },
        "source_of_truth": (
            "per_cell_line — every value here is a copy of that block.  Before "
            "2026-10-20 this block held placeholders (7 of 9 cells at the "
            "identical 0.80072, DU145 included only to reach the 9-key floor) "
            "and the pre-flight gate validated those instead of the real table."
        ),
        "resolvable_subset": {
            "min_n_test": MIN_RESOLVABLE_N_TEST,
            "cells_included": sorted(resolvable),
            "cells_excluded": excluded,
            "n_cells": len(resolvable),
            "unweighted_mean": (
                _simple_mean([b["mean_auc"] for b in resolvable.values()])
                if resolvable
                else float("nan")
            ),
            "n_weighted_mean": weighted,
            "why": (
                "A per-cell AUC on a single-digit test set is not evidence — "
                "DU145 (n_test=7) and SKOV3 (n_test=3) both scored exactly 1.0 "
                "across every seed.  The unweighted 9-cell mean gives those two "
                "the same weight as A549 (n_test=107), which is why it is the "
                "larger number."
            ),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    path = PROJECT_ROOT / args.artifact
    if not path.exists():
        print(f"[nci60] ERROR: {path} not found", file=sys.stderr)
        return 2
    doc = json.loads(path.read_text(encoding="utf-8"))
    per_cell = doc.get("per_cell_line")
    if not isinstance(per_cell, dict) or not per_cell:
        print(
            f"[nci60] ERROR: {path} has no per_cell_line block — refusing to "
            f"invent one.",
            file=sys.stderr,
        )
        return 2

    old = doc.get(BLOCK_KEY) or {}
    block = build_block(per_cell)

    drift = []
    for cell, new in block["per_cell_breakdown"].items():
        old_v = (old.get("per_cell_breakdown") or {}).get(cell, {}).get("mean_auc")
        if isinstance(old_v, (int, float)) and abs(float(old_v) - new["mean_auc"]) > 1e-3:
            drift.append((cell, float(old_v), new["mean_auc"]))
    if drift:
        print(
            f"[nci60] replacing {len(drift)} per-cell value(s) that disagreed "
            f"with per_cell_line:",
            file=sys.stderr,
        )
        for cell, o, n in drift:
            print(f"  {cell}: block {o:.5f} -> {n:.5f}", file=sys.stderr)

    doc[BLOCK_KEY] = block

    print(f"[nci60] overall_mean {block['overall_mean']:.4f} "
          f"(was {old.get('overall_mean')})")
    rs = block["resolvable_subset"]
    print(f"[nci60] resolvable ({rs['n_cells']}/9, n_test>={rs['min_n_test']}): "
          f"unweighted {rs['unweighted_mean']:.4f}, "
          f"n-weighted {rs['n_weighted_mean']:.4f}")
    print(f"[nci60] excluded: {rs['cells_excluded']}")

    if args.dry_run:
        print("[nci60] dry-run — nothing written")
        return 0

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(path, json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[nci60] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
