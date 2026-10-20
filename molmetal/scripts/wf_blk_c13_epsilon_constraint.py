# SPDX-License-Identifier: Apache-2.0
"""Epsilon-constraint slices of the CFM (generation x bond-type) Pareto front.

WHY THIS SHAPE OF ANALYSIS
--------------------------
A product like ``gen_validity x bond_type_macro_f1`` is a SCALARISATION: it
collapses the front to one number and throws away its shape. Epsilon-constraint
does the opposite — it fixes ONE objective to a bound (the hyperplane) and
optimises the other over everything that survives. The bound is the only
preference input, so the trade-off stops being hidden inside a weight.

MEASURED PREMISE
----------------
On both recorded 29-epoch runs there is NO epoch with gen_validity >= 0.85 AND
bond_type_macro_f1 >= 0.50 — so the front is a genuine multi-point curve
(control 7 non-dominated epochs, fm_ab 9), not a degenerate line. epsilon-
constraint is the right tool rather than an ornamental one.

WHY THIS HAD TO COME AFTER THE GATE FIX
---------------------------------------
This was written after ``combined`` stopped being the constant 0.0. Slicing a
Pareto front is pointless while the selector feeding it never moves: the run
had already recorded ``best_epoch = 0`` in both arms, i.e. its own judgement
was that the best model in 29 epochs was the one from before training.

WHAT epsilon IS, AND IS NOT
---------------------------
epsilon is a PREFERENCE, not a measurement. Nothing here discovers the right
value. The knee (the front point furthest from the chord joining its two
extremes) is reported as the standard evidence-based default, so the choice
can be made against a number rather than a preference.

SCOPE — read before quoting anything
-------------------------------------
Two runs, one seed each, 30 epochs. The fronts are computed from the SAME runs
the slices are read off, so this is in-sample by construction. It shows the
trade-off is real and locates its knee. It does NOT show that any particular
epsilon yields a better model, and it says nothing about whether the front
itself is fundamental — BLK-C12 records 8 of 12 loss terms dead, so this may
be the front of a broken objective rather than a physical constraint.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.eval.bond_metric_schema import (
    BOND_AP_UNDECLARED,
    bond_ap_definition,
    bond_ap_legend,
    require_single_definition,
)


RUNS = {
    "control": "molmetal/reports/cfm_fm_control_2026_10_09/per_epoch_metrics.csv",
    "fm_ab": "molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.csv",
}

#: The bound on the GENERATION objective. Everything at or above this
#: gen_validity is admissible; among those, bond type is maximised.
EPSILON_GRID = (0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)

OBJECTIVES = {
    # name -> column in per_epoch_metrics.csv
    "gen_validity": None,  # derived: gen_validity_val / gen_validity_n
    "bond_type_macro_f1": "bond_type_macro_f1_pooled",
    "bond_pr_auc": "bond_pr_auc",
}


def load(rel: str) -> Dict[str, Any]:
    rows = list(csv.DictReader((REPO / rel).open(encoding="utf-8")))
    def col(name):
        return np.array([float(r[name]) for r in rows])
    gv_n = col("gen_validity_n")
    gv = col("gen_validity_val")
    # Wf-A99 (2026-10-20) — these two runs are dated 2026-10-09, i.e. before
    # Wf-A97 corrected the metric, so their `bond_pr_auc` column is the
    # NO_BOND one-vs-rest AP (majority class).  The stamp is read from the
    # CSV when present and reported as `undeclared` otherwise; the module
    # never guesses a definition from a date.
    definition = BOND_AP_UNDECLARED
    if rows and "bond_pr_auc_definition" in rows[0]:
        definition = bond_ap_definition(rows[-1])
    return {
        "epoch": col("epoch").astype(int),
        "gen_validity": np.where(gv_n > 0, gv / np.where(gv_n > 0, gv_n, 1), 0.0),
        "bond_type_macro_f1": col(OBJECTIVES["bond_type_macro_f1"]),
        "bond_pr_auc": col(OBJECTIVES["bond_pr_auc"]),
        "bond_pr_auc_definition": definition,
    }


def pareto_front(xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> List[int]:
    """Indices of non-dominated points, both objectives maximised.

    A point is dominated when another is >= on both and > on at least one.
    """
    front = []
    for i in range(len(xs)):
        dominated = any(
            (xs[j] >= xs[i] and ys[j] >= ys[i])
            and (xs[j] > xs[i] or ys[j] > ys[i])
            for j in range(len(xs))
            if j != i
        )
        if not dominated:
            front.append(i)
    return front


def knee_index(xs: np.ndarray, ys: np.ndarray) -> int:
    """Front point furthest from the chord joining the two extremes.

    Normalised first: the two objectives live on different scales (gen_validity
    is a fraction of 48 samples, bond-type macro-F1 is a macro average), so an
    un-normalised distance would pick whichever axis happens to be wider.
    """
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    nx = (xs - x0) / (x1 - x0) if x1 > x0 else np.zeros_like(xs)
    ny = (ys - y0) / (y1 - y0) if y1 > y0 else np.zeros_like(ys)
    d = np.abs(nx + ny - 1.0) / np.sqrt(2.0)
    return int(np.argmax(d))


def slice_front(
    epochs: np.ndarray, xs: np.ndarray, ys: np.ndarray, eps: float, yname: str
) -> Dict:
    """Maximise y subject to x >= eps. Never silently relaxes the bound."""
    ok = xs >= eps - 1e-12
    if not ok.any():
        return {
            "epsilon": eps,
            "feasible": False,
            "why": (
                f"no epoch has gen_validity >= {eps}; the front's maximum is "
                f"{xs.max():.4f}"
            ),
            "best_epoch": None,
            yname: None,
            "gen_validity_at_best": None,
            "n_epochs_satisfying": 0,
        }
    ys_masked = np.where(ok, ys, -np.inf)
    i = int(np.argmax(ys_masked))
    return {
        "epsilon": eps,
        "feasible": True,
        "best_epoch": int(epochs[i]),
        yname: round(float(ys[i]), 4),
        "gen_validity_at_best": round(float(xs[i]), 4),
        "n_epochs_satisfying": int(ok.sum()),
    }


def analyse(name: str, run: Dict[str, np.ndarray], second: str) -> Dict:
    xs = run["gen_validity"]
    ys = run[second]
    epochs = run["epoch"]
    front = pareto_front(xs, ys, epochs)
    front.sort(key=lambda i: -xs[i])
    fx = xs[front]
    fy = ys[front]
    k = knee_index(fx, fy)
    return {
        "run": name,
        "second_objective": second,
        # Wf-A99 — the definition only exists for bond_pr_auc; macro-F1 has
        # one meaning and does not need a stamp.
        "second_objective_definition": (
            run.get("bond_pr_auc_definition") if second == "bond_pr_auc" else None
        ),
        "n_epochs": int(len(epochs)),
        "pareto_front": [
            {
                "epoch": int(epochs[i]),
                "gen_validity": round(float(xs[i]), 4),
                second: round(float(ys[i]), 4),
            }
            for i in front
        ],
        "knee": {
            "epoch": int(epochs[front[k]]),
            "gen_validity": round(float(fx[k]), 4),
            second: round(float(fy[k]), 4),
            "implied_epsilon": round(float(fx[k]), 4),
            "note": (
                "furthest from the chord joining the front's two extremes, both "
                "axes normalised. A standard evidence-based default, not a "
                "measurement of the right trade-off."
            ),
        },
        "epsilon_slices": [
            slice_front(epochs, xs, ys, e, second) for e in EPSILON_GRID
        ],
        # The 0.50 bar is the macro-F1 threshold the OpenBabel reference is
        # quoted against (tmqm_bond_perception.json: macro-F1 0.539). It is
        # NOT a bar that transfers to bond_pr_auc, which is on a different
        # scale, so the joint-feasibility check is reported for the macro-F1
        # analysis only. Reusing it elsewhere would compare a macro average
        # against an average precision and call the result a finding.
        "no_checkpoint_satisfies_both": (
            {
                "rule": "gen_validity >= 0.85 AND bond_type_macro_f1 >= 0.50",
                "threshold_basis": "OpenBabel reference macro-F1 = 0.539",
                "epochs": [
                    int(epochs[i])
                    for i in range(len(epochs))
                    if xs[i] >= 0.85 and ys[i] >= 0.50
                ],
                "note": "empty means the front is a real trade-off, not an artifact",
            }
            if second == "bond_type_macro_f1"
            else None
        ),
    }


def main() -> int:
    out = {
        "purpose": (
            "Locate the knee and the epsilon-constraint slices of the "
            "generation x bond-type Pareto front on the two recorded 29-epoch "
            "runs. In-sample by construction; epsilon is a preference."
        ),
        "scope": {
            "runs": list(RUNS),
            "seeds": "one seed per run (seed 0)",
            "in_sample": True,
            "not_established": (
                "No claim that any epsilon produces a better model. No claim "
                "that the front is fundamental — BLK-C12 records 8 of 12 loss "
                "terms dead, so this may be the front of a broken objective."
            ),
        },
        "epsilon_is_a_preference": (
            "epsilon is the bound on the generation objective. It is not "
            "discovered here; the knee is reported so the choice can be made "
            "against a number."
        ),
        "analyses": [],
    }
    _runs = {name: load(rel) for name, rel in RUNS.items()}
    _defs = {name: r["bond_pr_auc_definition"] for name, r in _runs.items()}
    out["bond_pr_auc_definition"] = _defs
    out["bond_pr_auc_definition_legend"] = {
        d: bond_ap_legend(d) for d in sorted(set(_defs.values()))
    }
    # Refuses to build a frontier if a future re-run stamps one arm with the
    # post-Wf-A97 definition and leaves the other unstamped — those numbers
    # are not on the same scale and a Pareto front over them is meaningless.
    require_single_definition(
        [{"run": n, "bond_pr_auc_definition": d} for n, d in _defs.items()],
        context="blk_c13 epsilon-constraint frontier",
    )
    for name, rel in RUNS.items():
        run = _runs[name]
        for second in ("bond_type_macro_f1", "bond_pr_auc"):
            a = analyse(name, run, second)
            out["analyses"].append(a)
            print(f"== {name} / {second}")
            print("   front:", [
                (p["epoch"], p["gen_validity"], p[second]) for p in a["pareto_front"]
            ])
            kn = a["knee"]
            print(f"   knee: ep={kn['epoch']} gv={kn['gen_validity']} "
                  f"{second}={kn[second]}")
            nb = a["no_checkpoint_satisfies_both"]
            print("   no-epoch-satisfies-both:",
                  "n/a (threshold is macro-F1-specific)" if nb is None else nb["epochs"])
            print(f"   {'eps':>5}  {'best_ep':>8}  {'gv':>7}  {second:>8}")
            for s in a["epsilon_slices"]:
                if not s["feasible"]:
                    print(f"   {s['epsilon']:>5}  INFEASIBLE ({s['why']})")
                else:
                    print(f"   {s['epsilon']:>5}  {s['best_epoch']:>8}  "
                          f"{s['gen_validity_at_best']:>7}  "
                          f"{s[second]:>8}")
            print()

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = REPO / "metrics" / "by_metric" / "blk_c13_epsilon_constraint.json"
    atomic_write_file(str(dest), json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())