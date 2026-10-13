#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Upgrade cfm_todo.json to v3: write down what the loss line is FOR.

WHY THIS EXISTS
---------------
On 2026-10-12 the user stated the objective explicitly, and the file did not
contain it anywhere:

    "改loss的目标是为了拿到更好的qed、admet和sa以及贵金属抗癌药的专用指标"
    (the goal of changing the loss is to get better QED, ADMET and SA, and
    precious-metal-anticancer-specific metrics)

That reframes what BLK-C-L1's refutation means. L1 answered "is the velocity
reduction broken?" -- NO, do not touch it. It did NOT answer "should the
objective function serve the downstream properties?", and the difference is
the whole remaining agenda. A file that records only the first answer reads
as "the loss line is finished", which is the opposite of the truth.

It also fixes an internal contradiction the Act pass left behind: the
BLK-C-L3 blocker was updated to PARTIALLY CONFIRMED while
``execution_order.phases[L2]`` still said "PREMISE FAILED ... DO NOT BUILD".
Two blocks of the same file telling an agent opposite things about the same
task is how the wrong one gets followed.

Writes atomically (AGENTS.md 5.6). Prints what it changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TODO = REPO / "cfm_todo.json"

OBJECTIVE = {
    "stated_verbatim_2026_10_12": (
        "改loss的目标是为了拿到更好的qed、admet和sa以及贵金属抗癌药的专用指标"
    ),
    "in_english": (
        "The purpose of changing the loss is to obtain better QED, ADMET and SA, "
        "and precious-metal-anticancer-specific metrics."
    ),
    "what_this_changes_about_L1": (
        "L1 refuted ONE hypothesis -- that the velocity loss is mis-reduced so "
        "that a metal-centred tail dominates the gradient. It measured 192 per-cloud "
        "rows and found the opposite sign (r(loss, n_atoms) = +0.408) and a "
        "reduction worth 4.2%. That closes a bug hunt. It says nothing about "
        "whether the objective function serves the properties above. Reading L1 as "
        "'the loss line is finished' inverts its actual conclusion, which is that "
        "the ONE part of the loss that was under suspicion is fine, and the "
        "eleven other parts were never checked against the objective at all."
    ),
    "the_objective_is_NOT": [
        "NOT lower fm_term. fm_term is a velocity regression; it is the mechanism, "
        "not the goal, and 75.4% of its trivial-predictor upper bound is already "
        "the target's own variance (metrics/by_metric/cfm_fm_term_percloud.json).",
        "NOT higher bond_type macro-F1. That is a perception metric on REAL "
        "geometry; it says nothing about the generated molecules.",
        "NOT connectivity. L4 withdrew 'connectivity is not the bottleneck' as "
        "stated, and the real tmQM comparison band turned out to be decoder "
        "output rather than real tmQM.",
    ],
    "the_objective_IS": {
        "primary_properties": [
            {"metric": "QED", "ours_MEASURED": 0.0944,
             "real_tmqm_reference_MEASURED": 0.2685,
             "source": "metrics/by_metric/cfm_checkpoint_property_benchmark.json",
             "caution": "QED's ADS functions were fitted on ~771 oral ORGANIC drugs "
                        "(Bickerton 2012). Metal centres, coordination numbers and "
                        "inorganic geometry are outside its calibration domain, and "
                        "metal-complex generation papers essentially do not report it. "
                        "0.2685 on real tmQM is near the metal ceiling. Chasing QED is "
                        "the user's stated goal and it is also the metric most likely "
                        "to reward metal-unrealistic output. Track it; do not let it "
                        "be the only objective."},
            {"metric": "SA (Ertl)", "ours_MEASURED": 6.4423,
             "real_tmqm_reference_MEASURED": 5.0269,
             "normalised_ours": 0.356,
             "closest_published_MEASURED": "0.68-0.80 (LigandDiff, JCIM 2024, "
                                           "Pt/Pd/Ru complexes)",
             "source": "docs/CFM_STATUS_2026_10_12.md Sec 2",
             "verdict": "SA is the real, fixable, comparable gap. This is the "
                        "single strongest argument for a molecule-level objective."},
        ],
        "admet_triple": {
            "note": "measured on the same checkpoint, NOT ON DISK for a swept arm",
            "ames_mutagen": {"ours": 0.0000, "real_tmqm": 0.0173},
            "logPapp": {"ours": -3.0387, "real_tmqm": -3.0794},
            "logS": {"ours": -8.0099, "real_tmqm": -8.3385},
            "reading": "All three sit essentially ON the real-tmQM band. tmQM is a "
                       "DFT-optimised structure set, not a drug-like library, so a "
                       "near-zero mutagenic rate is not evidence of a usable "
                       "cytotoxic scaffold -- it is evidence that the ADMET proxies "
                       "cannot see the difference. These three are NOT the lever. "
                       "PROJECTED as diagnostic, not as target.",
        },
        "precious_metal_anticancer_specific": {
            "why_this_is_the_axis_nobody_else_writes_down": (
                "The published comparators (LigandDiff, multi-LigandDiff, JACS Au "
                "JT-VAE, OM-Diff, tmQMg-L) report validity/connectivity/p_valc/DFT "
                "properties. None of them report coordination geometry, because for "
                "metal complexes THAT is the thing that determines activity -- it is "
                "what makes a platinum complex a cisplatin rather than a random "
                "PtCl2(NH3) adduct."
            ),
            "concrete_measures_on_disk_today": [
                {"measure": "metal-ligand closest contact",
                 "unguided_MEASURED": 1.234, "guided_MEASURED": 1.602,
                 "real_tmqm_MEASURED": 1.953, "unit": "Angstrom",
                 "source": "metrics/by_metric/cfm_l3_guidance_ab.json",
                 "reading": "Guidance closed 0.72 A of a 0.72 A gap and left 0.35 A "
                            "OPEN. This is the single most legible remaining gap in "
                            "the whole line and it is precious-metal-specific."},
                {"measure": "global min pair distance",
                 "unguided_MEASURED": 0.570, "guided_MEASURED": 1.155,
                 "real_tmqm_MEASURED": 1.172,
                 "source": "metrics/by_metric/cfm_l3_guidance_ab.json",
                 "reading": "CLOSED (abs error 0.602 -> 0.017)."},
                {"measure": "coordination number",
                 "generated_MEASURED": 4.87, "real_tmqm_MEASURED": 4.91,
                 "source": "metrics/by_metric/cfm_embed_failure_localisation.json",
                 "reading": "Already at parity. NOT a lever -- which is itself "
                            "informative: the model gets the COUNT right and the "
                            "GEOMETRY wrong."},
                {"measure": "RDKit 3D-embeddable",
                 "generated_MEASURED": 0.8182, "source": "same",
                 "reading": "11 of 12 failures are fixed by deleting the metal atom. "
                            "The bond head emits a metal coordination sphere ETKDG "
                            "cannot embed. BLK-C-L5."},
            ],
            "not_yet_measurable_here": [
                "square-planar vs tetrahedral geometry at Pt(II)/Pd(II)/Ru(II) -- "
                "this is THE pharmacophore for the cisplatin class and nothing in "
                "the current record measures it. BUILD IT BEFORE optimising anything "
                "against it, or an objective will be tuned against a proxy that "
                "cannot see the property that matters.",
                "oxidation state / charge formalism",
                "ligand-field preference (which donors coordinate: N vs O vs S)",
            ],
        },
    },
    "consequence_for_the_loss_line": (
        "The loss needs a MOLECULE-LEVEL objective that scores a decoded complex as "
        "a complex. Of the 12 existing terms, only bond_ce touches the decoded "
        "molecule's chemistry and it is a PER-EDGE cross-entropy (cfm_loss_term_"
        "liveness.json::no_generation_objective_is_real). The other 11 are velocity "
        "regression, regularisers, or dead. That gap is BLK-C-L2's original ask and "
        "it is untouched by L1."
    ),
    "guardrail": (
        "The literature objection to a differentiable QED/SA loss is real and is "
        "recorded in BLK-C-L3::why_not_a_QED_or_SA_loss_term. It is an objection to "
        "QED and SA SPECIFICALLY -- black-box, metal-uncalibrated, fragment-biased. "
        "It is NOT an objection to a molecule-level objective in general. Sampling-"
        "time guidance already MEASURED a 0.585 A improvement in closest contact on "
        "the existing checkpoint with no retraining, and that is a molecule-level "
        "effect. Do not let the QED/SA critique be read as 'no molecule-level "
        "objective ever'."
    ),
}


NEW_BLOCKER = {
    "id": "BLK-C-OBJ",
    "line": "CFM",
    "priority": 2,
    "title": "The loss has no molecule-level objective, and the user's stated goals "
             "(QED / ADMET / SA / precious-metal-specific) are all downstream of one",
    "status": "OPEN 2026-10-12 -- AUTHORISED BY USER DIRECTIVE, NOT YET DESIGNED",
    "established": (
        "metrics/by_metric/cfm_loss_term_liveness.json::no_generation_objective_is_real: "
        "of the 4 terms live under the shipped config, only bond_ce touches the "
        "decoded molecule's chemistry and it is a PER-EDGE cross-entropy. fm_term, w2 "
        "and cb are a velocity regression and two regularisers. Nothing scores the "
        "molecule as a whole. 8 of 12 terms are gradient-dead."
    ),
    "what_L1_did_NOT_decide": (
        "L1 measured the velocity loss in isolation and cleared it. It never asked "
        "whether the objective serves the properties in "
        "cfm_todo.json::the_objective_2026_10_12. Those are different questions and "
        "the second one is this blocker."
    ),
    "the_cheapest_proven_mechanism_already_exists": (
        "Sampling-time guidance, prototyped on the existing checkpoint with NO "
        "retraining, moved closest contact 0.570 -> 1.155 A against a real-tmqm 1.172 "
        "(metrics/by_metric/cfm_l3_guidance_ab.json). That is a molecule-level effect "
        "delivered without touching the loss. It is the reason to be optimistic; it is "
        "NOT yet evidence about QED/SA/ADMET, because decode was not run."
    ),
    "required_before_any_design": [
        "MEASURE the precious-metal-specific geometry axes that do not exist yet "
        "(square-planar vs tetrahedral at Pt/Pd/Ru, oxidation state, donor "
        "preference). Do not tune an objective against a proxy that cannot see the "
        "property that makes a complex a drug. THIS BLOCKS THE REST OF THE BLOCKER.",
        "Establish whether the guidance-improved geometry decodes to better SMILES. "
        "Everything downstream of geometry is currently unmeasured.",
    ],
    "not_authorised_yet": [
        "A differentiable QED term (see BLK-C-L3 for the four independent objections).",
        "Enabling any of the 8 dead terms without reading its own premise first -- "
        "three of them were proposed today and two had already been refuted by the "
        "time they were written.",
        "Any retrain. Per the user's 2026-10-12 directive the loss and architecture "
        "work is to be finished and then a single retrain run.",
    ],
    "source_of_the_directive": "user, 2026-10-12, verbatim in "
                               "the_objective_2026_10_12.stated_verbatim_2026_10_12",
}


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))

    d["the_objective_2026_10_12"] = OBJECTIVE

    # --- BLK-C-OBJ -----------------------------------------------------------
    ids = {b["id"] for b in d["blockers"]}
    if "BLK-C-OBJ" not in ids:
        l2 = next(i for i, b in enumerate(d["blockers"]) if b["id"] == "BLK-C-L2")
        d["blockers"].insert(l2, NEW_BLOCKER)

    # --- resolve the DO-NOT-BUILD contradiction -----------------------------
    for ph in d["execution_order"]["phases"]:
        if ph["phase"] != "L2":
            continue
        ph["status"] = (
            "RE-RANKED 2026-10-12, then RE-OPENED by the user directive that states "
            "what the loss is for (cfm_todo.json::the_objective_2026_10_12). L1's "
            "refutation closed a bug hunt, not the objective design. The one item "
            "that lost its premise (L3) had it RESTORED by measurement on the same "
            "day -- see items below."
        )
        ph["items"] = [
            "soft_valence scale: ALREADY FIXED AND GUARDED (commit c3ee284b64). The "
            "32537x premise was stale on arrival and is now retracted at every live "
            "source. Nothing to do.",
            "L3 sampling-time guidance: DO NOT BUILD IS WITHDRAWN. The earlier "
            "reason ('guidance acts on coordinates, the failure is on a graph that "
            "discards them') conflated two objects. Measured on the existing "
            "checkpoint, no retraining: closest contact 0.570 -> 1.155 A against real "
            "tmQM 1.172 (metrics/by_metric/cfm_l3_guidance_ab.json). NEXT: a strength "
            "sweep WITH decode, so we learn whether better geometry yields better "
            "SMILES. That is the first step of BLK-C-OBJ.",
            "BLK-C-OBJ molecule-level objective: NEW, and it is what the user asked "
            "for. BLOCKED on measuring the precious-metal-specific geometry axes "
            "first (square-planar vs tetrahedral at Pt/Pd/Ru, oxidation state, donor "
            "preference) -- none of them exist in the current record.",
            "L5 18.8% non-embeddable: CAUSE ESTABLISHED. A bond-head defect at the "
            "metal coordination sphere, not a velocity-stage one. This is the one "
            "concrete target L0/L1 produced.",
            "L2 dead terms: still unexamined. On hold, not authorised.",
        ]

    # --- L0 items: record that the Vina sweep is now unblocked --------------
    for ph in d["execution_order"]["phases"]:
        if ph["phase"] != "L0":
            continue
        ph["items"] = [
            i for i in ph["items"] if not i.startswith("Finish the Vina sweep")
        ] + [
            "Re-run the Vina sweep: UNBLOCKED 2026-10-12. The third blocker is gone "
            "(surrogate-rewritten ligands are now recorded as "
            "MEASURED_SURROGATE_ATOM, not MEASURED_CHEMOTYPE_VDW -- "
            "molmetal/tests/test_vina_surrogate_status.py, red check reported). "
            "The earlier sweep died in pocket 2 of 10 with no traceback; 9 of 10 "
            "pockets remain NOT ON DISK, so NO generated-vs-control statement is "
            "currently available. Report per-pocket error counts split by exception "
            "type and always state the denominator -- the earlier n=51 was a "
            "survivor count over an unstated attempted set, which is selection bias.",
            "Build the precious-metal geometry measures (square-planar vs "
            "tetrahedral, oxidation state, donor preference) BEFORE any objective is "
            "tuned against them. BLOCKS BLK-C-OBJ.",
        ]

    d["updated"] = "2026-10-12 (v3: objective recorded; L2 re-opened; Vina unblocked)"

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(
        TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"cfm_todo.json v3 written: {TODO.stat().st_size} bytes, "
          f"{len(d['blockers'])} blockers")
    print("  + the_objective_2026_10_12 (user directive, verbatim)")
    print("  + BLK-C-OBJ (molecule-level objective, author-facing)")
    print("  ~ L2 phase items rewritten (L3 'DO NOT BUILD' withdrawn)")
    print("  ~ L0 phase items: Vina sweep unblocked, geometry measures added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())