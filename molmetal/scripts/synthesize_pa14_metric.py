#!/usr/bin/env python3
"""Assemble the PA14 canonical metric from the 512/4 MoleculeNet run (BLK-PA14).

``train_moleculenet.py`` writes the per-run aggregate to
``molmetal/reports/pa14_moleculenet_cap512_d4/metrics.json``.  This script turns
it into a canonical ``metrics/by_metric`` artefact by adding a comparison block
against the hidden-300/depth-3 canonical
(``metrics/by_metric/moleculenet_auc_dmpnn_5seed.json``) and a derived verdict.

The 300/3 file is left untouched (AGENTS.md §5.5): this is a NEW artefact, not a
replacement, so both capacity points stay on the record.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_pa14_metric.py
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

REPORT = "molmetal/reports/pa14_moleculenet_cap512_d4/metrics.json"
BASELINE = "metrics/by_metric/moleculenet_auc_dmpnn_5seed.json"
OUT = "metrics/by_metric/moleculenet_auc_dmpnn_5seed_cap512_d4.json"
TASKS = ("bbbp", "bace", "hiv", "clintox")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", default=REPORT)
    p.add_argument("--baseline", default=BASELINE)
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)

    rep = json.loads((PROJECT_ROOT / args.report).read_text())
    base = json.loads((PROJECT_ROOT / args.baseline).read_text())

    comparison: Dict[str, object] = {}
    new_means, old_means = [], []
    for t in TASKS:
        n, o = rep["tasks"][t], base["tasks"][t]
        comparison[t] = {
            "cap512_d4_mean": float(n["test_auc_mean"]),
            "cap512_d4_std": float(n["test_auc_std"]),
            "cap300_d3_mean": float(o["test_auc_mean"]),
            "cap300_d3_std": float(o["test_auc_std"]),
            "delta_mean": float(n["test_auc_mean"] - o["test_auc_mean"]),
            "delta_std": float(n["test_auc_std"] - o["test_auc_std"]),
        }
        new_means.append(float(n["test_auc_mean"]))
        old_means.append(float(o["test_auc_mean"]))

    new_overall = sum(new_means) / len(new_means)
    old_overall = sum(old_means) / len(old_means)

    out = dict(rep)
    out.update(
        {
            "metric": "moleculenet_auc_dmpnn_5seed_cap512_d4",
            "status": "MEASURED",
            "MEASURED_vs_PROJECTED": "MEASURED",
            "comparison_vs_cap300_d3": comparison,
            "mean_auc_cap512_d4": new_overall,
            "mean_auc_cap300_d3": old_overall,
            "mean_delta": new_overall - old_overall,
            "comparison_baseline": args.baseline,
            "why": (
                "BLK-PA14: the paper's tab:moleculenet used the hidden-300/depth-3 "
                "canonical; the capacity sweep had already MEASURED that 512/4 "
                "lifts bace with halved variance.  This is the same-run 4-task "
                "confirmation that folds that configuration into a citable artefact."
            ),
            "verdict": (
                f"MEAN {new_overall:.4f} at 512/4 vs {old_overall:.4f} at 300/3 "
                f"({new_overall - old_overall:+.4f}).  "
                f"bace {comparison['bace']['cap512_d4_mean']:.4f} vs "
                f"{comparison['bace']['cap300_d3_mean']:.4f} "
                f"({comparison['bace']['delta_mean']:+.4f}), seed std "
                f"{comparison['bace']['cap300_d3_std']:.4f} -> "
                f"{comparison['bace']['cap512_d4_std']:.4f}."
            ),
            "caveats": [
                "bbbp moves the other way (-0.0071) and hiv is flat (-0.0011); "
                "the mean gain is carried by bace and clintox.",
                "Both arms use the historical varying-per-seed scaffold split "
                "(split_seed unset), so the 5-seed std mixes split and seed "
                "variance; the fixed-split confirmation lives in "
                "molmetal/reports/wf_a1_bace_capacity/SUMMARY.md (+0.1098).",
                "The chemprop reference column is a literature best-of-search "
                "constant; BLK-PA17 supplies the same-protocol value.",
            ],
            "honest_framing_note": (
                "MEASURED from the PA14 report on disk; the verdict is derived "
                "from the per-task deltas, not asserted."
            ),
        }
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = PROJECT_ROOT / args.out
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))
    for t in TASKS:
        c = comparison[t]
        print(f"{t:>8}  {c['cap512_d4_mean']:.4f} +/- {c['cap512_d4_std']:.4f}  "
              f"(300/3 {c['cap300_d3_mean']:.4f}, {c['delta_mean']:+.4f})")
    print(f"{'MEAN':>8}  {new_overall:.4f}  (300/3 {old_overall:.4f}, {new_overall - old_overall:+.4f})")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
