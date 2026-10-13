#!/usr/bin/env python
"""L1 — NCI60 9-cell capacity probe analysis (3-seed).

The shipped NCI60 9-cell run uses hidden=128/depth=3 (m21's architecture).
MoleculeNet showed hidden=512/depth=4 is much stronger there.  This script
compares a 3-seed capacity probe (512/4, seeds 0,1,2) against the shipped
128/3 5-seed baseline on the IDENTICAL scaffold split.

The probe shares the split with the baseline (same n_train/n_val/n_test per
cell), so the cleanest comparison is PAIRED BY (CELL, SEED):
    delta = probe_auc[cell,seed] - baseline_auc[cell,seed]   for seed in {0,1,2}
We also report the probe 3-seed mean vs the baseline 5-seed mean as a
secondary, cell-level reference.

MEASURED only; every number comes from the per-cell reports on disk.
Partial probe directories are tolerated (cells not yet written are skipped
and flagged), so this can be run mid-flight to preview.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CELL_DIRS = (
    "a2780",
    "a2780cis",
    "a549",
    "du145",
    "hct116",
    "hl60",
    "mcf7",
    "pc3",
    "skov3",
)
N_TEST_FILTER = 20  # PA16 funnel: cells with a meaningful test set
_SEED_RE = re.compile(r"per_seed_metrics_seed(\d+)\.json$")


def _probe_seed_aucs(cell_dir: Path) -> Dict[int, Dict[str, object]]:
    """Return {seed: {auc, n_test, n_train, n_val, hidden, depth}} for a cell."""
    out: Dict[int, Dict[str, object]] = {}
    if not cell_dir.is_dir():
        return out
    for f in sorted(cell_dir.glob("per_seed_metrics_seed*.json")):
        m = _SEED_RE.search(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        if d.get("status") != "OK":
            continue
        out[int(m.group(1))] = {
            "auc": float(d["test_metrics"]["roc_auc"]),
            "n_train": d.get("n_train"),
            "n_val": d.get("n_val"),
            "n_test": d.get("n_test"),
            "hidden": d.get("hidden"),
            "depth": d.get("depth"),
            "metal": d.get("metal"),
        }
    return out


def _baseline_seed_aucs(cell_dir: Path) -> Dict[int, Dict[str, object]]:
    f = cell_dir / "per_seed_metrics.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    out: Dict[int, Dict[str, object]] = {}
    for r in d["per_seed"]:
        out[int(r["seed"])] = {"auc": float(r["test_auc"])}
    out["_agg"] = {  # type: ignore[assignment]
        "mean": float(d["aggregate"]["test_auc_mean"]),
        "std": float(d["aggregate"]["test_auc_std"]),
        "n_train": d["aggregate"].get("n_train"),
        "n_val": d["aggregate"].get("n_val"),
        "n_test": d["aggregate"].get("n_test"),
    }
    return out


def _paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = delta.size
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = float(delta[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_cells": int(n),
        "mean_delta": float(delta.mean()),
        "delta_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(float(lo) > 0.0 or float(hi) < 0.0),
        "p_one_sided_greater": float(np.mean(means <= 0.0)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--probe-dir",
        action="append",
        default=None,
        help="probe results dir (repeatable; seeds are read from the filenames)",
    )
    p.add_argument(
        "--baseline-dir",
        default="results/nci60_9cell_5seed_50ep",
    )
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=42)
    p.add_argument(
        "--out-metric", default="metrics/by_metric/l1_nci60_cap512_d4_probe.json"
    )
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args(argv)

    probe_dirs = [
        PROJECT_ROOT / d
        for d in (
            args.probe_dir
            or [
                "results/nci60_9cell_probe_seed0_cap512_d4",
                "results/nci60_9cell_probe_seed12_cap512_d4",
            ]
        )
    ]
    base_dir = PROJECT_ROOT / args.baseline_dir

    rows: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []
    probe_seeds_seen: set = set()
    for cell in CELL_DIRS:
        # merge probe seeds across all probe dirs (later dirs may add seeds)
        pauc: Dict[int, Dict[str, object]] = {}
        for pd in probe_dirs:
            pauc.update(_probe_seed_aucs(pd / cell))
        bauc = _baseline_seed_aucs(base_dir / cell)
        if not pauc or "_agg" not in bauc:
            missing.append(cell)
            continue
        probe_seeds_seen.update(pauc.keys())
        shared = sorted(set(pauc) & {k for k in bauc if isinstance(k, int)})
        agg = bauc["_agg"]
        probe_mean = float(np.mean([pauc[s]["auc"] for s in sorted(pauc)]))
        rows[cell] = {
            "metal": pauc[sorted(pauc)[0]]["metal"],
            "probe_seed_auc": {int(s): float(pauc[s]["auc"]) for s in sorted(pauc)},
            "probe_mean": probe_mean,
            "baseline_seed_auc": {
                int(k): float(v["auc"]) for k, v in bauc.items() if isinstance(k, int)
            },
            "baseline_5seed_mean_auc": float(agg["mean"]),
            "baseline_5seed_std_auc": float(agg["std"]),
            "shared_seeds": shared,
            "delta_paired_by_seed": {
                int(s): float(pauc[s]["auc"]) - float(bauc[s]["auc"]) for s in shared
            },
            "delta_probe_mean_vs_baseline_mean": probe_mean - float(agg["mean"]),
            "n_train": agg["n_train"],
            "n_val": agg["n_val"],
            "n_test": agg["n_test"],
            "probe_hidden": pauc[sorted(pauc)[0]]["hidden"],
            "probe_depth": pauc[sorted(pauc)[0]]["depth"],
        }

    if args.require_complete and missing:
        print(f"[FAIL] probe incomplete, missing cells: {missing}")
        return 2

    cells_all = list(rows.keys())
    cells_filt = [c for c in cells_all if rows[c]["n_test"] >= N_TEST_FILTER]

    def _paired(cells):
        vals = [
            rows[c]["delta_paired_by_seed"][s]
            for c in cells
            for s in rows[c]["shared_seeds"]
        ]
        return np.asarray(vals, dtype=np.float64)

    def _by_cell_mean_delta(cells):
        return np.asarray(
            [rows[c]["delta_probe_mean_vs_baseline_mean"] for c in cells],
            dtype=np.float64,
        )

    def _macro(cells):
        if not cells:
            return None
        probe = np.asarray([rows[c]["probe_mean"] for c in cells])
        base_m = np.asarray([rows[c]["baseline_5seed_mean_auc"] for c in cells])
        base_s0 = np.asarray(
            [rows[c]["baseline_seed_auc"].get(0, np.nan) for c in cells]
        )
        dp = _paired(cells)
        dc = _by_cell_mean_delta(cells)
        return {
            "n_cells": len(cells),
            "probe_3seed_macro_mean": float(probe.mean()),
            "baseline_seed0_macro_mean": float(np.nanmean(base_s0)),
            "baseline_5seed_macro_mean": float(base_m.mean()),
            "paired_by_cell_seed": _paired_bootstrap(
                dp, args.n_bootstrap, args.boot_seed
            ),
            "by_cell_mean_delta": _paired_bootstrap(
                dc, args.n_bootstrap, args.boot_seed
            ),
        }

    overall = _macro(cells_all)
    funnel = _macro(cells_filt)

    # per-probe-seed macro (seed variance of the lever itself)
    per_probe_seed_macro = {}
    for s in sorted(probe_seeds_seen):
        present = [c for c in cells_all if s in rows[c]["probe_seed_auc"]]
        if not present:
            continue
        per_probe_seed_macro[str(s)] = {
            "all9_mean": float(
                np.mean([rows[c]["probe_seed_auc"][s] for c in present])
            ),
            "n_cells": len(present),
        }
        present_f = [c for c in present if rows[c]["n_test"] >= N_TEST_FILTER]
        if present_f:
            per_probe_seed_macro[str(s)]["funnel_mean"] = float(
                np.mean([rows[c]["probe_seed_auc"][s] for c in present_f])
            )

    report = {
        "metric": "l1_nci60_cap512_d4_probe",
        "status": "MEASURED" if not missing else "MEASURED_PARTIAL",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better), per cell line",
        "scope": "tier1_precious_metal_anticancer",
        "source": "molmetal/scripts/l1_nci60_capacity_probe_analysis.py",
        "description": (
            "L1 capacity lever: does hidden=512/depth=4 beat the shipped "
            "hidden=128/depth=3 on the NCI60 9-cell precious-metal panel?  "
            "3-seed probe (seeds 0,1,2) on the identical scaffold split, "
            "compared paired-by-(cell,seed) against the shipped 5-seed baseline."
        ),
        "protocol": {
            "probe_configs": [
                "configs/training/nci60_9cell_probe_seed0_cap512_d4.yaml",
                "configs/training/nci60_9cell_probe_seed12_cap512_d4.yaml",
            ],
            "baseline_config": "configs/training/nci60_9cell_5seed_50ep_v1.yaml",
            "probe": "hidden=512, depth=4, seeds 0,1,2",
            "baseline": "hidden=128, depth=3, seeds 0..4",
            "split": "identical scaffold split (n_train/n_val/n_test match per cell)",
            "significance": (
                "paired bootstrap over the (cell, seed) pairs with a shared seed "
                "(3 seeds x 9 cells = 27 cells), 10000 resamples, seed 42; "
                "also a cell-level mean-delta bootstrap"
            ),
            "n_test_filter": N_TEST_FILTER,
        },
        "probe_seeds": sorted(probe_seeds_seen),
        "cells_present": sorted(cells_all),
        "cells_missing": missing,
        "per_cell": rows,
        "per_probe_seed_macro": per_probe_seed_macro,
        "overall_all_cells": overall,
        "overall_n_test_ge_filter": {
            "filter": f"n_test >= {N_TEST_FILTER}",
            "cells": sorted(cells_filt),
            **(funnel or {}),
        },
        "krasnov_2026_baseline_auc": 0.81,
        "shipped_baseline_mean_auc": 0.7293802091886383,
    }

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"probe seeds: {sorted(probe_seeds_seen)}  cells present: {len(cells_all)}/9")
    print(
        f"{'cell':9s} {'p_s0':>6s} {'p_s1':>6s} {'p_s2':>6s} {'p_mean':>7s} "
        f"{'b_s0':>6s} {'b_mean':>7s} {'d_pair':>7s} {'n_test':>7s}"
    )
    for c in cells_all:
        r = rows[c]
        ps = r["probe_seed_auc"]
        dp = np.mean(list(r["delta_paired_by_seed"].values()))
        print(
            f"{c:9s} {ps.get(0, float('nan')):6.4f} {ps.get(1, float('nan')):6.4f} "
            f"{ps.get(2, float('nan')):6.4f} {r['probe_mean']:7.4f} "
            f"{r['baseline_seed_auc'].get(0, float('nan')):6.4f} "
            f"{r['baseline_5seed_mean_auc']:7.4f} {dp:+7.4f} {r['n_test']:7d}"
        )
    for name, blk in (("ALL9", overall), (f"n>={N_TEST_FILTER}", funnel)):
        if not blk:
            continue
        pp = blk["paired_by_cell_seed"]
        cc = blk["by_cell_mean_delta"]
        print(
            f"[{name}] probe={blk['probe_3seed_macro_mean']:.4f} "
            f"base_s0={blk['baseline_seed0_macro_mean']:.4f} "
            f"base5={blk['baseline_5seed_macro_mean']:.4f} | "
            f"paired(cell,seed) d={pp['mean_delta']:+.4f} "
            f"CI[{pp['ci95'][0]:+.4f},{pp['ci95'][1]:+.4f}] p={pp['p_one_sided_greater']:.4f} | "
            f"bycell d={cc['mean_delta']:+.4f} p={cc['p_one_sided_greater']:.4f}"
        )
    print("per-probe-seed macro:", json.dumps(per_probe_seed_macro, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
