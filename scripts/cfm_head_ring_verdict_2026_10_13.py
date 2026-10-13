#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Record the head-behaviour verdict: the bond head is NOT the ring defect.

THE MEASUREMENT
---------------
A4 was refuted on CAPACITY -- a linear model on raw distances classifies
ring size at 1.000 held-out (metrics/by_metric/cfm_a4_receptive_field.json).
Capacity is not behaviour, and behaviour is what decides where the single
authorised retrain slot goes.

Loaded the shipped checkpoint's head (strict=True, all6 Z table width 18)
and ran it on CLEAN tmQM geometry -- fully in-distribution for a head trained
on clean CSD structures. Counted the cycles its argmax bond predictions close
against the cycles the truth Wiberg bonds close on the same 120 clouds:

    5/6-membered   TRUTH 76.10%    HEAD 82.62%
    3-membered     TRUTH 18.26%    HEAD 12.79%
    total rings    TRUTH 657       HEAD 305

The head's ring-size distribution is CLOSER to the real corpus than the
truth bonds are. It UNDER-produces 3-rings and OVER-produces 5/6-rings.

WHY THIS MATTERS
----------------
The 89.81 percent three-membered rate in GENERATED output (like-for-like
real 12.38 percent, metrics/by_metric/cfm_opt2_ring_dative_correction.json)
cannot be blamed on a head that, fed clean geometry, gets ring size right.
The defect lives UPSTREAM, in the geometry the velocity field supplies --
which is what ARCH-4 and ARCH-1 address, and not the head retrain slot.

THE HONEST REMAINDER, recorded so it cannot be quietly dropped
-------------------------------------------------------------
The head predicts 305 rings against 657 in truth: it drops bonds. A correct
ring-size DISTRIBUTION is not a correct ring-size COUNT. So this rules out
"the head cannot do ring closure, so retrain it for that" -- it does NOT rule
out spending the slot on the SEPARATE macro-F1 deficit against OpenBabel
(0.4784 vs 0.5051, metrics/by_metric/tmqm_bond_perception_3seed.json),
which is untouched by any of this.

SMOKE SCALE: 120 clouds, one checkpoint, one seed, argmax decisions, CPU.

THE RULE
--------
Annotate. Nothing closed, nothing deleted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
TODO = REPO / "cfm_todo.json"
BEHAVIOUR = "metrics/by_metric/cfm_head_ring_size_behaviour.json"
BEHAVIOUR_SCRIPT = "molmetal/scripts/wf_head_ring_size_behaviour_2026_10_13.py"
CAPACITY = "metrics/by_metric/cfm_a4_receptive_field.json"
GENERATED_SIDE = "metrics/by_metric/cfm_opt2_ring_dative_correction.json"
TEST = "molmetal/tests/test_head_ring_size_behaviour.py"


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))
    for p in (BEHAVIOUR, BEHAVIOUR_SCRIPT, CAPACITY):
        if not (REPO / p).is_file():
            print(f"REFUSING: {p} not on disk", file=sys.stderr)
            return 2

    b = json.loads((REPO / BEHAVIOUR).read_text(encoding="utf-8"))

    d["head_is_not_the_ring_defect_2026_10_13"] = {
        "verdict": (
            "THE BOND HEAD IS NOT THE RING DEFECT. On clean geometry -- its "
            "home turf, no sampling, no ODE, no noise -- the trained head's "
            "ring-size distribution is CLOSER to the real corpus than the "
            "truth bonds are."
        ),
        "measured": {
            "n_clouds": b["setup"]["n_clouds_used"],
            "checkpoint": b["setup"]["checkpoint"],
            "checkpoint_epoch": b["setup"]["checkpoint_epoch"],
            "state_dict_loaded": b["setup"]["state_dict_loaded"],
            "decision_rule": b["setup"]["decision_rule"],
            "frac_5_or_6": {"truth": b["frac_5_or_6"]["truth"],
                            "head": b["frac_5_or_6"]["head"]},
            "frac_3_membered": {"truth": b["frac_3_membered"]["truth"],
                                "head": b["frac_3_membered"]["head"]},
            "total_rings": {"truth": b["n_truth_rings"],
                            "head": b["n_head_rings"]},
            "path": BEHAVIOUR,
        },
        "how_this_completes_the_A4_refutation": (
            "A4 claimed ring size is UNREPRESENTABLE in a per-edge head and "
            "used that to argue against the head retrain. Two measurements "
            "now remove that argument: (1) CAPACITY -- a plain "
            "LogisticRegression on the sorted pairwise distances of a ring "
            "classifies ring size at 1.000 held-out with per-class recall "
            "1.000, so the information is in the head's input; (2) BEHAVIOUR "
            "-- the trained head, fed clean geometry, actually closes rings of "
            "the right size. Capacity said it could; behaviour says it does."
        ),
        "where_the_defect_must_live": (
            "UPSTREAM, in the geometry the velocity field supplies. This is "
            "what ARCH-4 (make sampled geometry in-distribution) and ARCH-1 "
            "(the head never sees a noisy sample, verified at "
            "train_cfm_yuelbond_wasserstein.py:3437/:3454) address. It is NOT "
            "the head retrain slot. Corroborating: generated 3-membered 89.81 "
            "percent vs like-for-like real 12.38 percent ("
            + GENERATED_SIDE + ")."
        ),
        "the_honest_remainder_DO_NOT_DROP_THIS": (
            "The head predicts 305 rings against 657 in truth: it DROPS "
            "BONDS. A correct ring-size DISTRIBUTION is not a correct "
            "ring-size COUNT. So this rules out 'the head cannot do ring "
            "closure, so retrain it for that' -- it does NOT rule out "
            "spending the slot on the SEPARATE macro-F1 deficit against "
            "OpenBabel (0.4784 vs 0.5051, "
            "metrics/by_metric/tmqm_bond_perception_3seed.json), which this "
            "measurement does not touch."
        ),
        "smoke_scale": (
            "120 clouds, one checkpoint, one seed, argmax decisions (the "
            "checkpoint's own bond_decision is 'threshold' at 0.5), CPU, no "
            "GPU. Directional, not a settled number."
        ),
        "red_checked_by": TEST + " (6 tests). Two mutations were run and both "
        "were observed to fail: randomising the head's weights after loading, "
        "and dropping the .T transpose on the (2, E) truth-bond array. A third "
        "bug in the test itself was caught this way -- it read setup keys from "
        "the top level and raised KeyError, which pytest reports as an ERROR "
        "rather than a failure.",
        "paths": [BEHAVIOUR, BEHAVIOUR_SCRIPT, CAPACITY, TEST],
    }

    # OPT-4 rested on the structural argument. Annotate rather than delete.
    plan = d.get("optimisation_plan_2026_10_13", {})
    up = plan.get("ARCHITECTURE_UPGRADE_2026_10_13")
    if isinstance(up, dict):
        for it in up.get("items", []):
            if it.get("id") == "ARCH-2_cycle_aware_bond_head":
                it["PREMISE_FURTHER_WEAKENED_2026_10_13"] = (
                    "Two independent refutations now bear on this item. "
                    "CAPACITY: ring size is recoverable from the head's input "
                    "at 1.000 held-out accuracy. BEHAVIOUR: the trained head "
                    "on CLEAN geometry already produces a ring-size "
                    "distribution closer to the real corpus than truth "
                    "(5/6-membered 82.62% vs 76.10%). So a cycle-aware "
                    "readout is not 'the only way to express ring size' -- it "
                    "is a possible IMPROVEMENT over a head that already gets "
                    "the distribution roughly right, aimed at the 305-vs-657 "
                    "under-count rather than at ring size per se. Its cheap "
                    "no-retrain gate is now the decisive experiment: if a "
                    "cycle readout fails to move the metric, the head's local "
                    "geometry already carries what there is."
                )

    d["updated"] = (
        "2026-10-13 (v7.3: HEAD BEHAVIOUR measured -- on clean geometry the "
        "trained head's ring-size distribution is closer to the real corpus "
        "than truth (5/6-membered 82.62% vs 76.10%, 3-membered 12.79% vs "
        "18.26%). The bond head is not the ring defect; the defect is "
        "upstream in the sampled geometry. The head still drops bonds "
        "(305 vs 657 rings), so the separate macro-F1 deficit is untouched.)"
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    back = json.loads(TODO.read_text(encoding="utf-8"))
    k = back["head_is_not_the_ring_defect_2026_10_13"]["measured"]
    assert k["frac_3_membered"]["head"] <= k["frac_3_membered"]["truth"]
    assert k["frac_5_or_6"]["head"] >= k["frac_5_or_6"]["truth"]
    assert k["total_rings"]["head"] < k["total_rings"]["truth"]
    print("OK: head-behaviour verdict recorded")
    print(f"OK: 5/6 head {k['frac_5_or_6']['head']:.4f} >= truth "
          f"{k['frac_5_or_6']['truth']:.4f}")
    print(f"OK: 3-ring head {k['frac_3_membered']['head']:.4f} <= truth "
          f"{k['frac_3_membered']['truth']:.4f}")
    print(f"OK: under-count caveat retained ({k['total_rings']['head']} vs "
          f"{k['total_rings']['truth']} rings)")
    print(f"OK: blockers untouched: {[b2['id'] for b2 in back['blockers']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
