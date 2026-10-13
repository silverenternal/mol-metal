#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""cfm_todo.json v4 — remove the blockers that are resolved, with evidence.

WHY THIS EXISTS
---------------
Two rounds of workflows closed eight blockers. They were still sitting in
``blockers`` next to genuinely open work, which makes the file useless as a
work list: a reader cannot tell which entries are live without re-deriving each
one from a metrics JSON.

THE RULE APPLIED, and it is deliberately strict
------------------------------------------------
An entry leaves ``blockers`` only if its QUESTION HAS AN ANSWER ON DISK. Not if
it is nearly answered, not if its fix is prepared, not if it was closed by a
directive — the question has to be settled. Every removal below carries the
path that settles it, so the claim is checkable with one read.

Entries are MOVED to ``resolved_2026_10_12``, not deleted. AGENTS.md 5.5 forbids
rewriting history: a file that once said "OPEN — this blocks everything" and
now says nothing at all cannot be audited by the person who read the earlier
version. What leaves the ACTIVE plan is what the user asked for; what leaves
the FILE entirely is nothing.

WRITES ATOMICALLY per AGENTS.md 5.6. Prints every removal with its evidence
path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TODO = REPO / "cfm_todo.json"

# --------------------------------------------------------------------------
# REMOVED: question answered, evidence on disk.
# `because` must state the ANSWER, not the activity. "We fixed it" is not an
# answer; "the ODE is integrated and 53/64 clouds decoded" is.
# --------------------------------------------------------------------------
REMOVED = {
    "BLK-C-M1": {
        "question": "patience 3 early-stops runs that have not converged; seed0's "
                    "headline 0.9194 is a lower bound",
        "answer": "ANSWERED AND CLOSED. Root cause: DEFAULT_EARLY_STOP_PATIENCE=3 "
                  "(train_cfm_yuelbond_wasserstein.py:720) with no config setting it. "
                  "The patience=10 config exists and the run reached gate 0.9532 at "
                  "epoch 27 with no down-epoch in its last six, then was stopped at the "
                  "user's direction to benchmark the checkpoint first. 0.9532 beats the "
                  "0.9194 that was quoted as the config's headline.",
        "evidence": [
            "configs/cfm_patience10_seed0_2026_10_12.yaml (early_stop_patience: 10)",
            "molmetal/reports/cfm_patience10_seed0/events.jsonl",
            "docs/CFM_STATUS_2026_10_12.md sec 4",
        ],
        "closed_by": "user directive 2026-10-12 (checkpoint retained, benchmark first)",
    },
    "BLK-C-M2": {
        "question": "No decode path integrates the ODE, so m4 QED / m5 SA / m7 diversity "
                    "are UNMEASURED for free generation",
        "answer": "ANSWERED AND FIXED. The decode dump used the bond head only and "
                  "never ran the integrator. It now calls the trainer's own generation "
                  "path, evaluate_generation_validity -> generate_clouds_from_noise, and "
                  "that path is verified in production: four sweep arms decoded 22, 23, "
                  "32 and 53 molecules out of 64 clouds each. QED / SA / ADMET are "
                  "therefore measured on ODE-generated geometry, not on a "
                  "reconstruction.",
        "evidence": [
            "molmetal/scripts/wf_cfm_decode_dump_smiles.py:34,187-189",
            "molmetal/reports/cfm_l3_guidance_sweep_seed0/arm_strength*.json (decode block)",
            "metrics/by_metric/cfm_l3_guidance_sweep.json (properties_on_decoded)",
        ],
    },
    "BLK-C-M5": {
        "question": "m4 QED definition fix is correct but its measured magnitude is "
                    "provenance-lost",
        "answer": "ANSWERED. The comparison band was 636 rows of our own decoder output, "
                  "so the magnitude was measured against the generator. Replaced by a "
                  "real tmQM band on the six train metals: QED 0.234783, n=31,150. Both "
                  "the generated value and the reference now have a stated cohort.",
        "evidence": [
            "metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json",
            "metrics/by_metric/cfm_reference_band_audit.json",
        ],
    },
    "BLK-C-M7": {
        "question": "Re-audit every past CFM A/B — they were all run at patience 3",
        "answer": "ANSWERED, NEGATIVELY. Triage found only 2 of 20 runs cut off within "
                  "patience of their own best, and both are 1-epoch eval runs, not A/B "
                  "arms. The feared scope is NOT supported and the historical A/B results "
                  "do not need re-running. One over-claim was withdrawn in the process: "
                  "'most past A/B results are early-stop artefacts' was mine and was wrong.",
        "evidence": ["cfm_todo.json::corrections_2026_10_12",
                     "docs/CFM_STATUS_2026_10_12.md sec 4"],
    },
    "BLK-C-L1": {
        "question": "WHY does a live, dominant loss term (fm_term, 12.46 of ~14.5) fail to "
                    "descend?",
        "answer": "ANSWERED, AND THE HYPOTHESIS WAS WRONG. Measured over 192 per-cloud "
                  "rows: the reduction is worth 4.2% (9.325 element-mean vs 8.948 "
                  "cloud-mean), r(loss, n_atoms) = +0.408 has the OPPOSITE sign to the "
                  "metal-tail hypothesis, metal atoms carry 1.04% of the squared error, "
                  "and the trivial v=0 predictor scores 12.363, so the model removes "
                  "24.6% of a measured upper bound. NO reduction change is recommended. "
                  "A second hypothesis (un-centred CoM) was also refuted earlier: the "
                  "loader already centres, at tmqm_clouds.py:443 and :450.",
        "evidence": ["metrics/by_metric/cfm_fm_term_percloud.json",
                     "molmetal/scripts/wf_blkc_l1_fm_term_percloud.py"],
    },
    "BLK-C-L4": {
        "question": "Is the connectivity advantage an artefact of a broken aromatic-bond "
                    "valence evaluator (arXiv:2505.00169)?",
        "answer": "ANSWERED, AND IT WAS NOT OUR BUG. RDKit's GetTotalValence() kekulizes "
                  "internally, so cfm_property_panel.py:291-311 is immune by "
                  "construction, and the 0.45-0.63 band is unreachable — both cohorts "
                  "score 1.0000 under a correct Kekule evaluator. BUT the verdict that "
                  "replaced it is NARROWER: all three evaluators skip Z>20, i.e. every "
                  "metal centre, which is the exact axis Sec 2 blames, so 'connectivity "
                  "is not the bottleneck' is withdrawn as stated.",
        "evidence": ["metrics/by_metric/cfm_metric_validity_audit.json",
                     "molmetal/scripts/wf_blkc4_metric_validity_audit.py",
                     "molmetal/tests/test_wf_blkc4_metric_validity_audit.py"],
    },
    "BLK-C-L5": {
        "question": "Are the 18.8% non-embeddable molecules a velocity-stage failure "
                    "localised at the metal coordination centre?",
        "answer": "ANSWERED, AND THE LOCALISATION WAS A BASE RATE. With no control, "
                  "'11 of 12 embed once the metal is deleted' looks damning. Measured "
                  "against 600 REAL tmQM molecules on the same protocol, the real cohort "
                  "is repaired 95.0% of the time against our 91.7%. Real tmQM itself "
                  "embeds at only 0.6333, so our 0.8182 is BETTER than the reference "
                  "corpus. The 18.8% figure is not an honest-negative about the model.",
        "evidence": ["metrics/by_metric/cfm_l5_metal_deletion_control.json",
                     "molmetal/scripts/wf_blkc_l5_metal_deletion_control.py",
                     "molmetal/tests/test_wf_blkc_l5_metal_deletion_control.py"],
    },
    "BLK-C-OBJ": {
        "question": "There is no molecule-level objective; the user directed that loss "
                    "changes target QED / ADMET / SA / precious-metal-specific metrics",
        "answer": "DECIDED per target. QED: reachable but NOT via a QED term — ROTB and "
                  "ALERTS carry the gap and both trace to acyclic sp3 backbones; the "
                  "lever is bond_ce, which is currently LOSING. ADMET: change nothing — "
                  "no proxy reads the metal, so none can express a metal-specific fact. "
                  "SA: fragment rarity points the WRONG way; the gap is a "
                  "stereocentre-penalty consequence of the same acyclic defect, so SA and "
                  "QED are ONE problem. Precious-metal: measurable only from retained "
                  "coordinates, because ETKDG erases square-planarity in 91 of 99 paired "
                  "real CN=4 molecules.",
        "evidence": ["metrics/by_metric/cfm_blk_c_obj_decision.json",
                     "metrics/by_metric/cfm_objective_inventory.json",
                     "metrics/by_metric/cfm_sa_gao_coley_reference.json"],
    },
}


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))
    before = len(d["blockers"])

    kept, moved = [], []
    for b in d["blockers"]:
        bid = b.get("id")
        if bid in REMOVED:
            entry = {
                "id": bid,
                "title": b.get("title"),
                "removed_from_blockers": "2026-10-12 (cfm_todo.json v4)",
                **REMOVED[bid],
            }
            moved.append(entry)
        else:
            kept.append(b)
    d["blockers"] = kept

    missing = set(REMOVED) - {m["id"] for m in moved}
    if missing:
        print(f"WARNING: these ids were not present in blockers: {sorted(missing)}",
              file=sys.stderr)

    archive = d.setdefault("resolved_2026_10_12", [])
    if not isinstance(archive, list):
        archive = [archive]
    archive.extend(moved)

    # --- refresh the two blockers whose ANSWER landed today ------------------
    for b in kept:
        if b.get("id") == "BLK-C-L3":
            b["status"] = ("MEASURED 2026-10-12 -- the decode-enabled sweep is DONE and "
                           "the answer is NOT clean. Sampling-time guidance raises the "
                           "decode rate from 0.344 to 0.828 across strengths "
                           "0.0/0.01/0.05/0.15 on a paired cohort (same seed, same "
                           "templates), BUT: the best-geometry arm (0.05, closest to "
                           "real tmQM on total absolute error) is NOT the best-decoding "
                           "arm (0.15); 0.15 overshoots on p01_pair_distance and "
                           "radius_of_gyration; and QED moves the OPPOSITE way, 0.2358 "
                           "-> 0.1327. The decode gain is not a quality gain. The "
                           "property columns are also selection-confounded: each arm is "
                           "scored on the molecules THAT ARM decoded (n=22/23/32/53).")
            b["SWEEP_2026_10_12"] = {
                "measurements": "metrics/by_metric/cfm_l3_guidance_sweep.json",
                "arms": [0.0, 0.01, 0.05, 0.15],
                "cohort": "64 tmQM val clouds, midpoint/50, seed 1234, PAIRED across arms",
                "decode_rate": {"0.0": 0.34375, "0.01": 0.359375,
                                "0.05": 0.5, "0.15": 0.828125},
                "qed_on_decoded": {"0.0": 0.2358, "0.05": 0.1873, "0.15": 0.1327},
                "best_geometry_strength": 0.05,
                "best_decode_strength": 0.15,
                "contradiction": "geometry winner != molecule winner",
            }
        if b.get("id") == "BLK-C-M4":
            b["REFINED_2026_10_12"] = (
                "The losing axis is now LOCALISED rather than merely observed. Measured "
                "on 798 real tmQM complexes against 15,851 Wiberg-aromatic bonds: the "
                "head DOES emit the aromatic class (precision 0.8889), so this is not a "
                "never-predicted class hidden by a macro average. But 70.7% of true "
                "aromatic bonds (BO 1.3-1.7) collapse to the single-bond class, while "
                "single-bond recall is 0.966. The defect is class-4 vs class-1 RANKING "
                "on intermediate-bond-order pairs. Separately the generation gap is 1.6x "
                "the head gap, so a head-only fix cannot close it."
            )
            b["evidence"] = ["molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py"]

    # --- next_actions: drop the refuted P1, refresh the P0 ------------------
    d["next_actions"] = [
        n for n in d.get("next_actions", [])
        if not str(n.get("action", "")).startswith("BOND HEAD, METAL COORDINATION SPHERE")
    ]
    for n in d["next_actions"]:
        if str(n.get("action", "")).startswith("decode-enabled guidance sweep"):
            n["updated_2026_10_12"] = (
                "Two independent defects must ride along or the run repeats the "
                "unmeasurable cohort: (1) load_tmqm_clouds(n_clouds=...) takes a "
                "window that contained ZERO Pt -- training split is 24.83% Pt, so this "
                "is a sampling artefact and min_per_metal>=1 is the fix, no retrain; "
                "(2) the sampled cloud coordinates must be RETAINED, because ETKDG "
                "erases square-planarity in 91 of 99 paired real CN=4 molecules and any "
                "geometry verdict taken from SMILES describes RDKit. Both are landed "
                "and guarded; the run has not been launched."
            )

    # --- phases: the L0-L1 question-asking phases are over -------------------
    d["execution_order"]["phases"] = [
        {
            "phase": "L2",
            "name": "Make the objective measurable",
            "gate": "BLK-C-OBJ decided (CLOSED 2026-10-12)",
            "status": "DECIDED. Two sampler/cohort defects blocked measurement and both "
                      "are now fixed: the zero-Pt window (min_per_metal, guarded) and the "
                      "ETKDG round-trip that destroys coordination geometry. What is left "
                      "is to RUN the decode-enabled sweep that carries both fixes.",
            "items": [
                "Launch the P0 sweep with min_per_metal>=1 and retained coordinates.",
                "L3 sampling-time guidance: MEASURED, with a contradiction (see "
                "BLK-C-L3.SWEEP_2026_10_12). Do not report the decode gain alone.",
                "L5 18.8% non-embeddable: PREMINE REFUTED -- it is a base rate of the "
                "embedder, and our rate is better than real tmQM.",
            ],
        },
        {
            "phase": "L3",
            "name": "Retrain",
            "gate": "a specific loss or architecture change justified by a MEASURED "
                    "premise, not a projected one",
            "status": "NOT AUTHORISED YET. The one change with a measured premise is "
                      "bond_ce class-4 vs class-1 ranking on intermediate bond orders "
                      "(BLK-C-M4). Spending the single authorised retrain on it is a "
                      "decision the user has not made.",
            "items": [
                "Only after the P0 sweep reports whether the generation-side gap or the "
                "head-side gap dominates.",
            ],
        },
    ]

    d["updated"] = "2026-10-12 (v4: eight resolved blockers removed from the active plan)"

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    print(f"cfm_todo.json v4: {before} -> {len(kept)} active blockers, "
          f"{len(moved)} moved to resolved_2026_10_12\n")
    for m in moved:
        print(f"  REMOVED {m['id']:12} {m['answer'][:74]}...")
    print("\nKEPT (question still open):")
    for b in kept:
        print(f"  {b['id']:12} P{b.get('priority','?')}  {str(b.get('status',''))[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())