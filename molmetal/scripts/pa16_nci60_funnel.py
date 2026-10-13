#!/usr/bin/env python3
"""PA16 — NCI-60 9-cell panel: a test-set-size funnel (BLK-PA16).

The fixed-splitter rerun (BLK-PA12) left the NCI-60 panel headline at
``0.7294 +/- 0.2137`` with two single-digit test sets (SKOV3 n_test=3,
DU145 n_test=7) whose AUC of 1.0 is not informative, plus two large lines near
chance (A2780cis 0.3356, A2780 0.5104).  A mean that averages a 3-molecule test
set with a 107-molecule one is not readable.

This script recomputes the panel mean under explicit minimum-test-set-size
filters and *adds* the result to the canonical metric as a new
``test_set_size_funnel`` block.  Nothing existing is overwritten (AGENTS.md
§5.5): ``value_overall_mean_auc`` and every ``per_cell_line`` entry are left
verbatim, so the unfiltered number stays on the record.

It is a pure recompute from the values already on disk — no training, no
network — and is idempotent.

Usage
-----
::

    .venv/bin/python molmetal/scripts/pa16_nci60_funnel.py
    .venv/bin/python molmetal/scripts/pa16_nci60_funnel.py --metric path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_METRIC = "metrics/by_metric/nci60_precious_9cell_auc.json"
KRASNOV_BASELINE = 0.81
FILTER_THRESHOLDS = (10, 20)


def _subset_block(rows: List[tuple], n_min: int) -> Dict[str, object]:
    sel = [r for r in rows if r[3] >= n_min]
    aucs = np.asarray([r[1] for r in sel], dtype=np.float64)
    return {
        "n_test_min": int(n_min),
        "n_cells": int(len(sel)),
        "cells": [r[0] for r in sel],
        "excluded_cells": [r[0] for r in rows if r[3] < n_min],
        "mean_auc": float(aucs.mean()) if aucs.size else None,
        "std_across_cells": float(aucs.std()) if aucs.size else None,
        "per_cell": {
            r[0]: {"auc_mean": float(r[1]), "n_test": int(r[3])} for r in sel
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metric", default=DEFAULT_METRIC)
    args = p.parse_args(argv)

    path = PROJECT_ROOT / args.metric
    data = json.loads(path.read_text())
    pcl = data["per_cell_line"]
    rows = [
        (m, float(v["auc_mean"]), float(v.get("auc_std") or 0.0), int(v["n_test"]))
        for m, v in pcl.items()
        if v.get("auc_mean") is not None
    ]

    all_aucs = np.asarray([r[1] for r in rows], dtype=np.float64)
    weights = np.asarray([r[3] for r in rows], dtype=np.float64)
    n_weighted = float((all_aucs * weights).sum() / weights.sum())

    blocks = {f"n_test_ge_{t}": _subset_block(rows, t) for t in FILTER_THRESHOLDS}

    # Headline filter is the BLK-PA16 spec's n_test >= 20; n_test >= 10 is
    # carried as a sensitivity (it keeps PC3 n=18 and HL60 n=11).
    primary = "n_test_ge_20"
    pm = blocks[primary]["mean_auc"]
    verdict = (
        f"Under the BLK-PA16 n_test>=20 filter the panel mean is {pm:.4f} over "
        f"{blocks[primary]['n_cells']} cell lines, versus "
        f"{float(all_aucs.mean()):.4f} over all {len(rows)}.  The n_test>=10 "
        f"sensitivity gives {blocks['n_test_ge_10']['mean_auc']:.4f} over "
        f"{blocks['n_test_ge_10']['n_cells']}.  Both are BELOW the "
        f"{KRASNOV_BASELINE:.2f} comparator, so on this panel the honest "
        f"reading is an explicit miss \u2014 not a tie, and not a win.  The "
        f"comparison is also protocol-mismatched (scaffold split here vs "
        f"Krasnov's random within-metal CV)."
    )

    funnel = {
        "why": (
            "BLK-PA16: the 9-cell mean mixes single-digit test sets (SKOV3 n=3, "
            "DU145 n=7) where AUC=1.0 is uninformative with large near-chance "
            "lines.  Recompute under explicit minimum-test-set-size filters."
        ),
        "source": "recomputed from per_cell_line[*].auc_mean / .n_test on disk",
        "n_cells_measured": len(rows),
        "all_cells_mean_auc": float(all_aucs.mean()),
        "all_cells_std_across_cells": float(all_aucs.std()),
        "n_test_weighted_mean_auc": n_weighted,
        "subsets": blocks,
        "primary_filter": primary,
        "primary_mean_auc": pm,
        "krasnov_2026_baseline_auc": KRASNOV_BASELINE,
        "protocol_mismatch_note": (
            "Krasnov 2026's 0.81 is a within-metal stratified 5-fold CV (random "
            "split).  This panel is measured on a Bemis-Murcko scaffold split "
            "(whole scaffolds held out), which is strictly harder.  The two "
            "numbers are not like-for-like; the scaffold-split panel should be "
            "compared to a random-split re-measurement of the same panel."
        ),
        "verdict": verdict,
        "honest_framing": (
            "MEASURED, recomputed from on-disk values.  No cell line is dropped "
            "from the record: the unfiltered mean and every per_cell_line entry "
            "are preserved above."
        ),
    }

    data["test_set_size_funnel"] = funnel

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(path, json.dumps(data, indent=2, default=float, ensure_ascii=False))

    print(f"cells measured: {len(rows)}")
    print(f"all-cells mean: {float(all_aucs.mean()):.4f}")
    for k, b in blocks.items():
        print(f"  {k}: n_cells={b['n_cells']} mean={b['mean_auc']:.4f} "
              f"std={b['std_across_cells']:.4f} excluded={b['excluded_cells']}")
    print(f"n_test-weighted mean: {n_weighted:.4f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
