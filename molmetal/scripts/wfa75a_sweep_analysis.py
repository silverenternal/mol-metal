#!/usr/bin/env python
"""Wf-A75a — guidance-weight sweep analysis (MEASURED only).

Reads N result dirs produced by train_cfm_yuelbond_wasserstein.py
(each carrying per_epoch_metrics.json) and emits one comparison JSON
that answers the Wf-A75a question:

    Which bond-length guidance weight keeps the Wf-A75 fragment/bond_f1
    win WITHOUT collapsing gen_validity?

Selection rule (stated up front, so the answer is not chosen after the
fact): among all (weight, epoch) cells with gen_validity >= 0.90, pick
the highest bond_f1; ties broken by gen_largest_fragment_frac.  The 0.90
floor is the "decode validity is not broken" line — the Wf-A75 run at
weight 0.1 fell to 0.479 by ep 4.

Every emitted number carries the source file it came from (AGENTS.md
§5.2).  No metric is computed here; all values are read from disk.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Bond-class index -> name, per YuelBondBondDecoder's 5-class head.
CLASS_NAMES = {0: "NO_BOND", 1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}

# Fields copied verbatim from per_epoch_metrics.json (no derivation).
SCALAR_FIELDS = [
    "bond_f1",
    "bond_precision",
    "bond_recall",
    "bond_type_macro_f1_pooled",
    "bond_per_class_macro_pr_auc",
    "gen_validity",
    "gen_largest_fragment_frac",
    "gen_single_fragment_ratio",
    "gen_metal_ratio",
    "gen_mean_heavy_atoms",
    "gen_seed_spread",
    "decode_ratio_rdkit_strict",
    "train_loss",
    "wall_time_s",
]


def load_run(spec: str) -> dict:
    weight_s, path_s = spec.split("=", 1)
    weight = float(weight_s)
    root = pathlib.Path(path_s)
    metrics_path = root / "per_epoch_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"[wfa75a] missing {metrics_path}")
    blob = json.loads(metrics_path.read_text())
    epochs = []
    for row in blob["per_epoch_metrics"]:
        rec = {"epoch": row["epoch"]}
        for f in SCALAR_FIELDS:
            rec[f] = row.get(f)
        rec["per_class_pr_auc"] = {
            CLASS_NAMES[c]: row.get(f"bond_class_{c}_pr_auc") for c in CLASS_NAMES
        }
        epochs.append(rec)
    return {
        "guidance_weight": weight,
        "source_path": str(metrics_path),
        "config_epochs": blob.get("config", {}).get("epochs"),
        "device": blob.get("config", {}).get("device"),
        "seed": blob.get("config", {}).get("seed"),
        "epochs": epochs,
    }


def pick_best(runs: list[dict], validity_floor: float) -> dict | None:
    cells = []
    for run in runs:
        for ep in run["epochs"]:
            gv = ep.get("gen_validity")
            if gv is None or gv < validity_floor:
                continue
            cells.append(
                {
                    "guidance_weight": run["guidance_weight"],
                    "epoch": ep["epoch"],
                    "gen_validity": gv,
                    "bond_f1": ep.get("bond_f1"),
                    "gen_largest_fragment_frac": ep.get("gen_largest_fragment_frac"),
                    "source_path": run["source_path"],
                }
            )
    if not cells:
        return None
    cells.sort(
        key=lambda c: (
            c["bond_f1"] if c["bond_f1"] is not None else -1.0,
            c["gen_largest_fragment_frac"]
            if c["gen_largest_fragment_frac"] is not None
            else -1.0,
        ),
        reverse=True,
    )
    return cells[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="WEIGHT=DIR",
        help="e.g. 0.05=results/cfm_5metal_wfa75a_w005",
    )
    ap.add_argument(
        "--out",
        default="metrics/by_metric/cfm_wfa75a_guidance_sweep.json",
    )
    ap.add_argument(
        "--validity-floor",
        type=float,
        default=0.90,
        help="gen_validity floor for the 'validity not broken' line",
    )
    args = ap.parse_args(argv)

    runs = sorted((load_run(s) for s in args.run), key=lambda r: r["guidance_weight"])
    best = pick_best(runs, args.validity_floor)

    payload = {
        "metric_id": "cfm_wfa75a_guidance_weight_sweep",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "question": (
            "Which bond-length guidance weight keeps the Wf-A75 "
            "bond_f1 / largest_fragment_frac win without collapsing gen_validity?"
        ),
        "selection_rule": (
            f"max bond_f1 among (weight, epoch) cells with "
            f"gen_validity >= {args.validity_floor}; tie-break "
            f"gen_largest_fragment_frac"
        ),
        "validity_floor": args.validity_floor,
        "runs": runs,
        "winner": best,
        "honest_framing": [
            "All values are read verbatim from each run's per_epoch_metrics.json; "
            "nothing is recomputed or extrapolated.",
            "5-epoch smokes measure direction, not converged performance; the "
            "30-epoch retrain (Wf-A76) is the paper-facing run.",
            "gen_validity is measured on 96 samples (3 seeds x 32); a single "
            "seed's ep-to-ep swing is not a trend.",
        ],
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    # Console table (weight x epoch).
    print(f"{'w':>5} {'ep':>3} {'gen_valid':>9} {'bond_f1':>8} {'frag_frac':>9} "
          f"{'SINGLE':>7} {'AROM':>6} {'DOUBLE':>7}")
    for run in runs:
        for ep in run["epochs"]:
            pc = ep["per_class_pr_auc"]
            print(
                f"{run['guidance_weight']:>5.2f} {ep['epoch']:>3} "
                f"{(ep['gen_validity'] or 0):>9.3f} {(ep['bond_f1'] or 0):>8.3f} "
                f"{(ep['gen_largest_fragment_frac'] or 0):>9.3f} "
                f"{(pc['SINGLE'] or 0):>7.3f} {(pc['AROMATIC'] or 0):>6.3f} "
                f"{(pc['DOUBLE'] or 0):>7.4f}"
            )
    print()
    if best is None:
        print(f"[wfa75a] NO cell clears gen_validity >= {args.validity_floor}")
    else:
        print(
            f"[wfa75a] winner: w={best['guidance_weight']} ep={best['epoch']} "
            f"gen_validity={best['gen_validity']:.3f} bond_f1={best['bond_f1']:.3f} "
            f"frag_frac={best['gen_largest_fragment_frac']:.3f}"
        )
    print(f"[wfa75a] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
