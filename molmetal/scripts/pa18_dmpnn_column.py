#!/usr/bin/env python3
"""PA18 — aggregate the per-metal D-MPNN DOI-CV summaries into one column.

``run_m1_paper_cv.py --cohort-only`` writes ``<out-dir>/<metal>/summary.json``
per metal.  ``pa18_model_class_5metal.py --dmpnn-json`` expects a flat
``{metal: {mean_auc, std_auc}}`` mapping.  This script bridges the two and
refuses to emit a number for any metal whose run is incomplete — a partial
fold set is not a mean (AGENTS.md §5.2).

Usage
-----
::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    .venv/bin/python molmetal/scripts/pa18_dmpnn_column.py \\
        --runs-dir /tmp/pa18_dmpnn \\
        --out /tmp/pa18_dmpnn/dmpnn_column.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_METALS = ("Ru", "Ir", "Rh", "Os", "Re")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True, help="parent of <metal>/summary.json")
    ap.add_argument("--metals", nargs="+", default=list(DEFAULT_METALS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    runs = Path(args.runs_dir)
    column: dict = {}
    missing: list[str] = []
    for metal in args.metals:
        summary_path = runs / metal / "summary.json"
        if not summary_path.is_file():
            missing.append(f"{metal}: no {summary_path}")
            continue
        s = json.loads(summary_path.read_text())
        if not s.get("complete"):
            missing.append(
                f"{metal}: incomplete "
                f"({s.get('n_folds_run')}/{s.get('n_requested_folds')} folds)"
            )
            continue
        column[metal] = {
            "mean_auc": s["mean"],
            "std_auc": s["std"],
            "per_fold_auc": s["per_fold_auc"],
            "n_folds": s["n_folds_run"],
            "complete": True,
            "source": str(summary_path),
            "protocol": s.get("protocol"),
        }
        print(
            f"[pa18-dmpnn] {metal}: mean={s['mean']:.4f} +/- {s['std']:.4f} "
            f"({s['n_folds_run']} folds)",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(column, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[pa18-dmpnn] wrote {out}", flush=True)
    if missing:
        print("[pa18-dmpnn] MISSING/INCOMPLETE:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
