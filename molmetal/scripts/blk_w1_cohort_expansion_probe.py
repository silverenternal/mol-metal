#!/usr/bin/env python3
"""W1 — cohort-expansion probe: is the small-metal floor a data artefact?

Motivation (todo.json :: submission_plan_2026_10_13.weak_metrics_workstream W1)
------------------------------------------------------------------------------
The probe cohort applies ``CytotoxFilter(time_threshold=24.0, ic50_min=0.01)``.
MEASURED 2026-10-13: that 24 h window is what makes Os/Rh/Re tiny — the
underlying MetalCytoToxDB has 333/339/250 unique Os/Rh/Re ligands, but the
24 h filter keeps only 36/51/74, i.e. below ``MIN_UNIQUE_FOR_GATE = 200``.
The paper we compare against (Krasnov et al. 2026) applies NO time filter, so
the 24 h window is our own convention, not a property of the data.

This script re-measures within-metal 5-fold CV and cross-metal LOMO on the
SAME scorer under two cohort specs:

* ``current``     — ``time_threshold=24.0, ic50_min=0.01`` (the probe's cohort);
* ``paper_funnel`` — ``build_paper_cohort`` (Krasnov-style: IC50-present, metal,
  measurement-level dedup, non-cancer exclusion; no time window).

MEASURED only.  The point is to decide whether the below-chance Os AUC is a
capability gap (it is not, if a bigger cohort resolves it) or a volume floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
MIN_UNIQUE_FOR_GATE = 200


def _load_env_default() -> None:
    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


def _cohorts():
    """Return {spec: {metal: (unique_smiles, y)}} for the two cohort specs."""
    import pandas as pd

    from molmetal.data.cytotox import (
        CytotoxFilter,
        MetalCytotoxDataset,
        build_paper_cohort,
    )
    from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates

    raw = pd.read_csv(PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
                      low_memory=False)
    out: Dict[str, Dict[str, Tuple[List[str], np.ndarray]]] = {
        "current": {}, "paper_funnel": {}
    }
    for m in METALS:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        s_u, y_u, _ = collapse_duplicates(ds.smiles, ds.active)
        out["current"][m] = (s_u, y_u)

        sub = build_paper_cohort(raw, metal=m)
        ic50 = sub["IC50_Dark_value"].astype(float)
        y_raw = (ic50 < 10.0).to_numpy(dtype=int)
        s_u2, y_u2, _ = collapse_duplicates(
            sub["SMILES_Ligands"].astype(str).to_numpy(), y_raw
        )
        out["paper_funnel"][m] = (s_u2, y_u2)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 49, 123])
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--out-metric",
                   default="metrics/by_metric/w1_cohort_expansion_probe.json")
    args = p.parse_args(argv)

    _load_env_default()
    from molmetal.scripts.metal_activity_transfer_probe import cross_metal, within_metal

    cohorts = _cohorts()
    seeds = tuple(args.seeds)

    report: Dict[str, object] = {
        "metric": "w1_cohort_expansion_probe",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/blk_w1_cohort_expansion_probe.py",
        "description": (
            "W1 cohort-expansion probe. Re-measures within-metal 5-fold CV and "
            "cross-metal LOMO on the same scorer under the probe's 24h cohort "
            "and under the Krasnov-style paper funnel, to test whether the "
            "below-chance Os AUC is a volume floor rather than a capability gap."
        ),
        "protocol": {
            "scorer": "morgan_fp(radius=2, 2048bit) + xgboost",
            "seeds": list(seeds),
            "n_splits": 5,
            "cohort_specs": {
                "current": "CytotoxFilter(time_threshold=24.0, ic50_min=0.01)",
                "paper_funnel": "build_paper_cohort (no time window; Krasnov-style)",
            },
            "min_unique_for_gate": MIN_UNIQUE_FOR_GATE,
        },
        "cohorts": {},
    }

    for spec, per_metal in cohorts.items():
        block: Dict[str, object] = {"per_metal": {}}
        for m in METALS:
            s_u, y_u = per_metal[m]
            w_aucs = []
            for s in seeds:
                r = within_metal(s_u, y_u, s, args.radius, args.nbits)
                if r.get("status") == "MEASURED":
                    w_aucs.append(float(r["roc_auc"]))
            block["per_metal"][m] = {
                "n_unique": int(len(s_u)),
                "pos_rate": float(y_u.mean()),
                "resolvable": bool(len(s_u) >= MIN_UNIQUE_FOR_GATE),
                "n_seeds_ok": len(w_aucs),
                "within_metal_auc_mean": float(np.mean(w_aucs)) if w_aucs else float("nan"),
                "within_metal_auc_per_seed": w_aucs,
            }
        cm = cross_metal(per_metal, seeds[0], args.radius, args.nbits)
        block["cross_metal_lomo"] = cm
        report["cohorts"][spec] = block

    # paired comparison of the two cohort specs on within-metal AUC
    comp = {}
    for m in METALS:
        a = report["cohorts"]["current"]["per_metal"][m]
        b = report["cohorts"]["paper_funnel"]["per_metal"][m]
        da = np.asarray(a["within_metal_auc_per_seed"])
        db = np.asarray(b["within_metal_auc_per_seed"])
        d = db - da
        comp[m] = {
            "n_current": a["n_unique"], "n_paper_funnel": b["n_unique"],
            "auc_current": a["within_metal_auc_mean"],
            "auc_paper_funnel": b["within_metal_auc_mean"],
            "delta": float(d.mean()),
            "delta_per_seed": [float(x) for x in d],
        }
    report["cohort_spec_comparison"] = comp

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(out, json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    print(f"{'metal':5} {'n_24h':>6} {'n_paper':>8} {'AUC 24h':>9} {'AUC paper':>10} {'delta':>8}")
    for m in METALS:
        c = comp[m]
        print(f"{m:5} {c['n_current']:6d} {c['n_paper_funnel']:8d} "
              f"{c['auc_current']:9.4f} {c['auc_paper_funnel']:10.4f} "
              f"{c['delta']:+8.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
