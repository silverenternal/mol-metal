#!/usr/bin/env python
"""Why the NCI-60 9-cell split has 3-107 test rows per cell.

Question
--------
``nci60_precious_9cell_auc.json`` reports a 9-cell mean of 0.7294, but its
per-cell test sets are 48 / 28 / 107 / 7 / 31 / 11 / 90 / 18 / 3 rows.  Two
cells (DU145, SKOV3) sit at 7 and 3, where ROC-AUC can only take four values.
Is the cohort short of data, or is the split handing test the leftovers?

Answer (measured here, not asserted)
------------------------------------
The cohort is not short: 7,823 labelled rows over 5,207 unique molecules and
nine cell lines.  ``ScaffoldSplitter._assign_groups`` budgets its 70/10/20
capacities in *scaffold counts*, and ``largest_first`` spends those groups on
the largest scaffolds first.  With 1,151 scaffolds, train takes the first
806 groups (the big ones) and test receives 232 leftover singleton scaffolds:
7,320 / 160 / 343 rows, i.e. 93.6 / 2.0 / 4.4 % instead of 70/10/20.

The fix measured here is ``scaffold_coverage_70_10_20``: the same scaffold
grouping and the same one-scaffold-one-split invariant, but the splitter
targets 70/10/20 of *each cell line's labelled rows* and refines the greedy
assignment with whole-scaffold moves.  Every cell then gets >= 25 test rows and
>= 22 val rows.

Nothing is reimplemented: both splits come from
``train_dmpnn_multitask_wasserstein.load_nci60_9cell_split_table``, the same
function the trainer calls.

Output: ``metrics/by_metric/nci60_split_coverage_audit.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molmetal.data.splits import ScaffoldSplitter, _bemis_murcko_key  # noqa: E402
from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    NCI60_COVERAGE_SALT,
    _SmilesOnlyAdapter,
    load_nci60_9cell_split_table,
)

CSV = ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
OUT = ROOT / "metrics" / "by_metric" / "nci60_split_coverage_audit.json"

#: Rows needed before a per-cell ROC-AUC is reported.  The m3 line uses 30
#: (``train_cisplatin_resistance.MIN_EVAL_ROWS``); this artifact's own funnel
#: uses 20.  Both are reported so neither is silently substituted for the other.
EVAL_BARS = (20, 30)


def _per_split(df: pd.DataFrame) -> dict:
    all_cells = sorted(df["cell_line"].unique().tolist())
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        cells = {}
        for cell in all_cells:
            cs = sub[sub["cell_line"] == cell]
            n = int(len(cs))
            pos = int(cs["active"].sum())
            cells[str(cell)] = {
                "labelled": n,
                "positive": pos,
                "negative": n - pos,
                "evaluable": {str(b): bool(n >= b and 0 < pos < n) for b in EVAL_BARS},
            }
        out[split] = {
            "rows": int(len(sub)),
            "molecules": int(sub["smiles"].nunique()),
            "row_fraction": round(len(sub) / len(df), 4),
            "cells": cells,
        }
    return out


def _scaffold_stats(df: pd.DataFrame) -> dict:
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df["smiles"].astype(str)):
        key = _bemis_murcko_key(smi) or f"__empty__{smi}"
        scaffolds[key].append(i)
    groups = sorted(scaffolds.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_groups = len(groups)
    n_train_groups = max(1, int(round(0.7 * n_groups)))
    train_eligible = {key for key, _ in groups[:n_train_groups]}

    row_scaffold = {}
    for key, rows in scaffolds.items():
        for r in rows:
            row_scaffold[r] = key
    df = df.reset_index(drop=True)
    concentration = {}
    for cell, cs in df.groupby("cell_line", sort=True):
        idx = cs.index.to_numpy()
        in_train = sum(1 for i in idx if row_scaffold[i] in train_eligible)
        concentration[str(cell)] = {
            "labelled": int(len(cs)),
            "in_largest_70pct_of_groups": int(in_train),
            "share": round(in_train / len(cs), 4) if len(cs) else 0.0,
        }

    spanning = defaultdict(set)
    for i, key in row_scaffold.items():
        spanning[key].add(df.loc[i, "split"])
    return {
        "n_scaffolds": n_groups,
        "n_singleton_scaffolds": int(sum(1 for _, g in groups if len(g) == 1)),
        "largest_scaffold_rows": int(len(groups[0][1])) if groups else 0,
        "n_groups_budgeted_to_train": n_train_groups,
        "largest_first_concentration": concentration,
        "scaffolds_spanning_splits": int(sum(1 for v in spanning.values() if len(v) > 1)),
    }


def _summary(df: pd.DataFrame) -> dict:
    per = _per_split(df)
    return {
        "rows": {s: per[s]["rows"] for s in ("train", "val", "test")},
        "row_fraction": {s: per[s]["row_fraction"] for s in ("train", "val", "test")},
        "molecules": {s: per[s]["molecules"] for s in ("train", "val", "test")},
        "per_split": per,
        "min_test_rows": min(v["labelled"] for v in per["test"]["cells"].values()),
        "min_val_rows": min(v["labelled"] for v in per["val"]["cells"].values()),
        "cells_below_30_test": sorted(
            c for c, v in per["test"]["cells"].items() if v["labelled"] < 30
        ),
        "cells_below_20_test": sorted(
            c for c, v in per["test"]["cells"].items() if v["labelled"] < 20
        ),
        "scaffolds_spanning_splits": _scaffold_stats(df)["scaffolds_spanning_splits"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    as_is = load_nci60_9cell_split_table(args.csv, "scaffold_70_10_10")
    coverage = load_nci60_9cell_split_table(args.csv, "scaffold_coverage_70_10_20")
    if len(as_is) != len(coverage):
        raise SystemExit(
            f"split row counts differ ({len(as_is)} vs {len(coverage)}) — "
            "the two protocols must partition the same cohort"
        )

    cohort = pd.read_csv(args.csv)
    payload = {
        "metric": "nci60_split_coverage_audit",
        "description": (
            "Per-cell-line coverage of the two NCI-60 9-cell split protocols: "
            "the shipped m21 scaffold split (fractions budgeted in scaffold "
            "counts) and the opt-in coverage-stratified split (fractions "
            "budgeted in per-cell labelled rows, with whole-scaffold "
            "refinement)."
        ),
        "unit": "labelled rows per split (one row = one (molecule, cell) pair)",
        "scope": "data_protocol_audit",
        "source": (
            "molmetal/scripts/wf_a62_nci60_split_coverage_audit.py over "
            "molmetal/data/nci60_precious_9cell.csv, calling "
            "train_dmpnn_multitask_wasserstein.load_nci60_9cell_split_table "
            "for both protocols"
        ),
        "cohort": {
            "path": "molmetal/data/nci60_precious_9cell.csv",
            "rows": int(len(cohort)),
            "unique_molecules": int(cohort["smiles"].nunique()),
            "cell_lines": sorted(cohort["cell_line"].unique().tolist()),
            "label": "CSV 'active' column (IC50 < 10 uM), used verbatim",
        },
        "evaluation_bars": {
            "nci60_funnel": 20,
            "m3_min_eval_rows": 30,
            "note": (
                "Both are reported per cell. The shipped artifact's own funnel "
                "uses n_test >= 20; the m3 line uses 30. Neither is silently "
                "substituted for the other."
            ),
        },
        "as_is_scaffold_70_10_10": {
            "protocol": (
                "ScaffoldSplitter(largest_first, fractions=0.7/0.1/0.2, seed=0) "
                "on unique SMILES; capacities are scaffold counts"
            ),
            **_summary(as_is),
            "scaffold_stats": _scaffold_stats(as_is),
        },
        "applied_coverage_split": {
            "protocol": (
                "ScaffoldSplitter(coverage=per-cell masks, "
                f"coverage_salt={NCI60_COVERAGE_SALT}, coverage_refine=True)"
            ),
            "salt_selection": (
                "searched 0-299 for max(min test rows + min val rows) subject "
                "to every cell >= 20 test rows; coverage only, no model was "
                "trained or scored"
            ),
            **_summary(coverage),
            "scaffold_stats": _scaffold_stats(coverage),
        },
        "verdict": (
            "The 3-107 test rows are produced by the split, not by the cohort. "
            "Budgeting 70/10/20 in scaffold counts spends the largest 70% of "
            "groups on train, so test receives singleton scaffolds: 4.4% of "
            "rows and five cells below 30. The coverage split keeps the same "
            "one-scaffold-one-split invariant and gives every cell >= 25 test "
            "and >= 22 val rows."
        ),
        "status": "MEASURED",
        "generated_by": "molmetal/scripts/wf_a62_nci60_split_coverage_audit.py",
    }

    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    payload["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    for name in ("as_is_scaffold_70_10_10", "applied_coverage_split"):
        s = payload[name]
        print(
            f"{name}: rows={s['rows']} min_test={s['min_test_rows']} "
            f"min_val={s['min_val_rows']} below30={s['cells_below_30_test']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
