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
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT = "metrics/by_metric/nci60_precious_9cell_auc.json"
BLOCK_KEY = "wf312b_nci60_smoke_aggregate"

# Below this many test molecules a per-cell AUC cannot support a claim: a
# 3-molecule test set can reach 1.0 by ranking one positive first.
#
# This is only the fallback.  The artifact already carries
# ``test_set_size_funnel.primary_filter`` (BLK-PA16, ``n_test_ge_20``) — that is
# the file's own declared primary, and this script must mirror it rather than
# introduce a second, quieter definition.  Measured 2026-10-20: the first
# version of this script hardcoded 10 and wrote 0.6521 into
# ``resolvable_subset`` while ``test_set_size_funnel.primary_mean_auc`` said
# 0.5923 — two "honest" numbers in one file, which is the same defect class the
# script was written to remove.
MIN_RESOLVABLE_N_TEST = 10


def _resolve_threshold(doc: Dict[str, object]) -> Tuple[int, str]:
    """Return ``(min_n_test, where_it_came_from)``.

    Prefers the artifact's own ``test_set_size_funnel.primary_filter`` so the
    rebuilt block cannot disagree with the funnel that sits beside it.
    """
    funnel = doc.get("test_set_size_funnel")
    if isinstance(funnel, dict):
        primary = funnel.get("primary_filter")
        if isinstance(primary, str) and primary.startswith("n_test_ge_"):
            try:
                return int(primary.rsplit("_", 1)[1]), "test_set_size_funnel.primary_filter"
            except ValueError:
                pass
    return MIN_RESOLVABLE_N_TEST, "MIN_RESOLVABLE_N_TEST (funnel absent)"


def _simple_mean(values: List[float]) -> float:
    return sum(values) / len(values)


def build_block(
    per_cell: Dict[str, Dict[str, object]],
    min_n_test: int = MIN_RESOLVABLE_N_TEST,
    threshold_source: str = "MIN_RESOLVABLE_N_TEST",
) -> Dict[str, object]:
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
        c: b for c, b in breakdown.items() if b["n_test"] >= min_n_test
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
            "min_n_test": min_n_test,
            "threshold_source": threshold_source,
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
                f"A per-cell AUC on a test set smaller than {min_n_test} is not "
                f"evidence.  Excluded here: "
                + ", ".join(
                    f"{c} (n_test={breakdown[c]['n_test']}, "
                    f"AUC={breakdown[c]['mean_auc']:.4f})"
                    for c in excluded
                )
                + ".  The unweighted 9-cell mean gives each of those the same "
                "weight as A549 (n_test=107), which is why it is the larger "
                "number."
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
    min_n_test, threshold_source = _resolve_threshold(doc)
    block = build_block(per_cell, min_n_test, threshold_source)

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

    # The funnel and the rebuilt block must state the same honest number.  If
    # they can drift, one file carries two answers again.
    funnel = doc.get("test_set_size_funnel")
    if isinstance(funnel, dict):
        primary_mean = funnel.get("primary_mean_auc")
        rs = block["resolvable_subset"]
        if isinstance(primary_mean, (int, float)) and rs["n_cells"]:
            delta = abs(float(primary_mean) - rs["unweighted_mean"])
            if delta > 1e-6:
                print(
                    f"[nci60] ERROR: rebuilt resolvable_subset says "
                    f"{rs['unweighted_mean']:.6f} but "
                    f"test_set_size_funnel.primary_mean_auc says "
                    f"{float(primary_mean):.6f} (Δ={delta:.2e}).  Refusing to "
                    f"write two different honest numbers into one artifact.",
                    file=sys.stderr,
                )
                return 2
            print(
                f"[nci60] resolvable_subset agrees with "
                f"test_set_size_funnel.primary_mean_auc ({float(primary_mean):.4f})"
            )

    print(f"[nci60] overall_mean {block['overall_mean']:.4f} "
          f"(was {old.get('overall_mean')})")
    rs = block["resolvable_subset"]
    print(f"[nci60] resolvable ({rs['n_cells']}/9, n_test>={rs['min_n_test']} "
          f"from {rs['threshold_source']}): "
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
