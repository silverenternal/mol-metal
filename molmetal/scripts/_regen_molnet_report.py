"""Regenerate the MoleculeNet Tier 2 report from cached per-seed JSONs.

Used to recover from a report-generation KeyError after the main
4-tasks × 3-seeds run completed. Reads per-seed JSONs from
``logs/metric_molnet/<task>_seed<n>.json`` and re-writes the aggregate
report at ``molmetal/reports/wf_tier2_molnet/final.md`` and the metric
JSON at ``metrics/by_metric/moleculenet_auc.json``.

This file is invoked manually; the primary train_moleculenet.py now
calls its own reporter.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.train_moleculenet import (
    _aggregate,
    _write_report,
    CHEMPROP_DMPNN_BASELINE,
    CHEMPROP_RF_BASELINE,
)

OUT_DIR = PROJECT_ROOT / "logs" / "metric_molnet"
METRICS_DIR = PROJECT_ROOT / "metrics" / "by_metric"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf_tier2_molnet"

TASKS = ["bbbp", "bace", "hiv", "clintox"]


def main() -> int:
    all_results: dict = {}
    for task in TASKS:
        per_seed = []
        for s in (0, 1, 2):
            p = OUT_DIR / f"{task}_seed{s}.json"
            if p.exists():
                per_seed.append(json.loads(p.read_text()))
        if not per_seed:
            print(f"[regen] no per-seed JSONs for task {task!r}; skipping", flush=True)
            continue
        agg = _aggregate(per_seed)
        agg["task"] = task
        agg["chemprop_dmpnn_baseline"] = CHEMPROP_DMPNN_BASELINE[task]
        agg["chemprop_rf_baseline"] = CHEMPROP_RF_BASELINE[task]
        agg["per_seed"] = per_seed
        all_results[task] = agg

    # Re-write aggregate metric JSON
    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "model": "morgan_mlp_v4",
        "featurization": "morgan_fp_2048_r2",
        "split": "scaffold (Bemis-Murcko, 70/10/20)",
        "seeds": [0, 1, 2],
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "hidden": 256,
        "tasks": {
            task: {k: v for k, v in all_results[task].items() if k != "per_seed"}
            for task in TASKS if task in all_results
        },
        "gate_mean_auc_threshold": 0.80,
        "chemprop_dmpnn_baseline_mean": float(np.mean([CHEMPROP_DMPNN_BASELINE[t] for t in TASKS])),
    }
    out_metric = METRICS_DIR / "moleculenet_auc.json"
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    out_metric.write_text(json.dumps(payload, indent=2))
    print(f"[regen] wrote {out_metric}", flush=True)

    mean_auc = float(np.mean([all_results[t]["test_auc_mean"] for t in TASKS if t in all_results]))
    gate_pass = mean_auc >= 0.80
    print(f"[regen] mean AUC = {mean_auc:.4f}  (gate ≥ 0.80)  {'PASS' if gate_pass else 'BELOW GATE'}", flush=True)

    out_report = REPORTS_DIR / "final.md"
    _write_report(out_report, all_results, gate_pass, TASKS)
    print(f"[regen] wrote {out_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())