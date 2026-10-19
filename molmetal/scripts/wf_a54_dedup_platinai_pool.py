#!/usr/bin/env python3
"""Deduplicate the merged PlatinAI + MetalCytoToxDB training pool.

The pool at ``.data/molmetal/platinai_4cells_with_5metal_train.csv`` contains
the same (complex, cell line) measurement more than once — either as a byte
identical row or as a replicate from a second paper.  Nothing downstream
notices, so those compounds are silently up-weighted by however many times
they were published.

This script writes a deduplicated **variant** beside the original and a
measured report.  It never overwrites its input: the pool is consumed by work
outside this script's ownership, and replacing a dataset under a running
consumer is not a data fix, it is an outage.

Dedup key is ``(smiles, metal, cell_line)`` — the same ligand with a different
metal is a different compound, and the same compound on a different cell line
is a different measurement.  ``keep="first"`` in file order, matching
``molmetal.data.cytotox.PAPER_DEDUP_KEYS``.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf_a54_dedup_platinai_pool.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IN = ".data/molmetal/platinai_4cells_with_5metal_train.csv"
DEFAULT_OUT = ".data/molmetal/platinai_4cells_with_5metal_train_dedup.csv"
DEFAULT_METRIC = "metrics/by_metric/platinai_pool_dedup.json"

# Same complex, same cell line.  Metal is part of the identity: a Ru complex
# and its Pt analogue share the ligand SMILES but are different compounds.
DEDUP_KEYS: tuple = ("smiles", "metal", "cell_line")


def measure(df: pd.DataFrame, keys: List[str]) -> Dict[str, object]:
    """Duplicate census for *df* under *keys*."""
    grouped = df.groupby(keys, dropna=False)
    sizes = grouped.size()
    excess = int((sizes - 1).clip(lower=0).sum())
    conflicts = 0
    if "pic50" in df.columns:
        # Pairs whose repeats disagree on the value: keep="first" picks one
        # arbitrarily, so the count must be reported rather than absorbed.
        conflicts = int((grouped["pic50"].nunique() > 1).sum())
    return {
        "rows": int(len(df)),
        "unique_keys": int(len(sizes)),
        "duplicate_groups": int((sizes > 1).sum()),
        "excess_rows": excess,
        "exact_full_row_duplicates": int(df.duplicated().sum()),
        "groups_with_conflicting_pic50": conflicts,
    }


def label_definition_check(df: pd.DataFrame) -> Dict[str, object]:
    """Recover each source's implicit ``activity`` threshold, in pIC50.

    The merged pool concatenates two sources whose binary labels were built
    with different cut-offs, and the ``activity`` column does not say so.  A
    model trained on the concatenation is fitting two different tasks at once.
    Recovering the threshold per source makes that visible instead of latent.
    """
    if not {"source", "pic50", "activity"}.issubset(df.columns):
        return {"skipped": "source/pic50/activity column missing"}
    out: Dict[str, object] = {"per_source": {}}
    for src, sub in df.groupby("source"):
        sub = sub.dropna(subset=["pic50"])
        if sub.empty:
            continue
        best_th, best_acc = None, -1.0
        for th in np.arange(3.0, 9.0, 0.01):
            acc = float(((sub["pic50"] >= th).astype(int) == sub["activity"]).mean())
            if acc > best_acc:
                best_th, best_acc = float(th), acc
        out["per_source"][str(src)] = {
            "n": int(len(sub)),
            "best_pic50_threshold": round(best_th, 2),
            "implied_ic50_uM": round(float(10 ** (6 - best_th)), 4),
            "accuracy": round(best_acc, 4),
        }
    thresholds = [
        v["best_pic50_threshold"] for v in out["per_source"].values()
    ]
    out["thresholds_differ"] = (
        bool(thresholds) and (max(thresholds) - min(thresholds)) > 0.5
    )
    out["why"] = (
        "If the recovered thresholds differ by more than 0.5 pIC50 (a factor of "
        "~3 in IC50), the sources' 'activity' columns mean different things and "
        "must not be concatenated into one binary target without re-deriving it."
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in-csv", default=DEFAULT_IN)
    p.add_argument("--out-csv", default=DEFAULT_OUT)
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output CSV.  Off by default so a rerun "
             "cannot silently replace a variant someone is already using.",
    )
    args = p.parse_args(argv)

    src = PROJECT_ROOT / args.in_csv
    dst = PROJECT_ROOT / args.out_csv
    if not src.exists():
        print(f"[dedup] ERROR: input not found: {src}", file=sys.stderr)
        return 2
    if dst.exists() and not args.force:
        print(
            f"[dedup] REFUSING: {dst} exists.  Pass --force to regenerate.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(src)
    keys = list(DEDUP_KEYS)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        print(f"[dedup] ERROR: missing key columns {missing}", file=sys.stderr)
        return 2

    before = measure(df, keys)

    # A dedup that moves a row between splits would change the protocol, not
    # just the weighting.  Check before writing anything.
    cross = int((df.groupby(keys, dropna=False)["split"].nunique() > 1).sum()) \
        if "split" in df.columns else 0
    if cross:
        print(
            f"[dedup] REFUSING: {cross} (complex, cell) group(s) appear in more "
            f"than one split.  Deduplicating would silently move rows across "
            f"the train/test boundary; fix the split first.",
            file=sys.stderr,
        )
        return 2

    dedup = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    after = measure(dedup, keys)

    per_split = {
        "before": {k: int(v) for k, v in df["split"].value_counts().items()}
        if "split" in df.columns else {},
        "after": {k: int(v) for k, v in dedup["split"].value_counts().items()}
        if "split" in dedup.columns else {},
    }
    per_metal = {
        "before": {k: int(v) for k, v in df["metal"].value_counts().items()},
        "after": {k: int(v) for k, v in dedup["metal"].value_counts().items()},
    }

    dedup.to_csv(dst, index=False)

    label_check = label_definition_check(df)
    if label_check.get("thresholds_differ"):
        print(
            "[dedup] WARNING: the sources' 'activity' columns use different "
            "pIC50 thresholds — see label_definition_check in the report.",
            file=sys.stderr,
        )

    report = {
        "metric": "platinai_pool_dedup",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "row counts",
        "scope": "tier1_precious_metal_anticancer / training-pool hygiene",
        "source": "molmetal/scripts/wf_a54_dedup_platinai_pool.py",
        "input": args.in_csv,
        "output": args.out_csv,
        "dedup_keys": keys,
        "keep": "first",
        "before": before,
        "after": after,
        "rows_removed": before["rows"] - after["rows"],
        "per_split": per_split,
        "per_metal": per_metal,
        "cross_split_groups": cross,
        "label_definition_check": label_check,
        "consumers": (
            "NONE.  MEASURED 2026-10-20: no config and no trainer in this repo "
            "reads this file — melt_platinai_4cell.py writes it and nothing "
            "loads it.  It is an orphan build artefact, so the risk is that a "
            "future run picks it up and inherits its two label definitions."
        ),
        "why": (
            "The same (complex, cell line) measurement appeared more than once "
            "in the merged pool, so those compounds carried extra weight in "
            "proportion to how many times they were published.  Deduplicating "
            "makes every measurement count once."
        ),
        "honest_framing_note": (
            "MEASURED row counts only.  This script does not retrain anything: "
            "any metric produced on the undeduplicated pool is still on disk and "
            "still describes that pool.  Re-run a model on the dedup CSV before "
            "quoting a number as deduplicated."
        ),
        "conflict_policy": (
            "Where repeats disagree on pic50 the first row in file order is kept "
            "(the repo convention in molmetal.data.cytotox.PAPER_DEDUP_KEYS).  "
            "groups_with_conflicting_pic50 counts how many groups this decision "
            "applied to; it is not a tie-break on quality."
        ),
    }
    metric_path = PROJECT_ROOT / args.metric
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # A .meta.json beside the data file, matching the convention the other
    # .data/molmetal artefacts already follow.  The warning has to live where
    # the next reader looks, not only in a metrics/ report they may never open.
    meta_path = src.with_suffix(".meta.json")
    meta = {
        "n_rows": int(len(df)),
        "n_rows_deduplicated": int(len(dedup)),
        "status": "ORPHAN_BUILD_ARTEFACT",
        "consumers": "none — no config or trainer in this repo reads this file",
        "label_definition": {
            src_name: v["best_pic50_threshold"]
            for src_name, v in label_check.get("per_source", {}).items()
        },
        "label_definition_warning": (
            "The sources' 'activity' columns use DIFFERENT pIC50 cut-offs: "
            + ", ".join(
                f"{k} -> {v['best_pic50_threshold']} pIC50 "
                f"(~{v['implied_ic50_uM']} uM)"
                for k, v in label_check.get("per_source", {}).items()
            )
            + ".  Do not concatenate them into one binary target without "
            "re-deriving the label.  The deduplicated variant is "
            f"{args.out_csv}; see {args.metric}."
        )
        if label_check.get("thresholds_differ")
        else "labels agree across sources",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[dedup] wrote {meta_path}")

    print(f"[dedup] {before['rows']} -> {after['rows']} rows "
          f"(-{report['rows_removed']})")
    print(f"[dedup] duplicate groups {before['duplicate_groups']}, "
          f"exact full-row dups {before['exact_full_row_duplicates']}, "
          f"conflicting pic50 {before['groups_with_conflicting_pic50']}")
    print(f"[dedup] wrote {dst}")
    print(f"[dedup] wrote {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
