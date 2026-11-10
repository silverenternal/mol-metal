#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""cfm_todo.json v7 — add the model-internal architecture programme.

WHY THIS EXISTS
---------------
The file had a five-item plan (OPT-1..OPT-5) in which every item was a
CONFIG change, a decode-time postprocess, or a loss weight. None of them
changed the model. That is the gap the user named: 「基于现状要有对模型内结构的优化」
— the plan needs optimisation of the model's INTERNAL STRUCTURE.

The items below are grounded in four structural facts, each verified against
source in this pass (not asserted from a design document):

A1. THE BOND HEAD NEVER SEES A NOISY SAMPLE. `CFMJointModel.forward`
    (train_cfm_yuelbond_wasserstein.py:1775) calls `self.head(cloud)` on the
    clean cloud and evaluates the velocity field at `t_b = 1`, which under
    `x_t = (1-t)x_0 + t x_1` is exactly `x_1`. Verified: no `(1-t)*x0`
    interpolation appears anywhere in that forward. So bond_ce trains a
    PERCEPTION module on clean CSD geometry, while at inference the head is
    asked to read geometry the velocity field produced by integrating from
    noise. Nothing in the objective ever asks the head to be robust along the
    trajectory it will actually be used on. This is the single largest
    train/inference gap in the model and it is NOT in the previous plan.

A2. THERE IS NO MOLECULE-LEVEL TERM. The loss sum at :3816 is
    bond_ce_term + w2 + 0.05*cb + diversity + shortcut_distill + consistency
    + so3 + edm + fm + assemble + entropy + soft_valence. Verified: no term
    consumes a decoded graph, an RDKit Mol, SMILES, a ring, or a cycle.
    bond_ce is a PER-EDGE cross-entropy; everything else is a regulariser or
    a velocity-field term. Consequences follow directly from the measured
    defects: nothing penalises 3-membered rings (89.81% gen vs 12.38%
    real like-for-like), nothing penalises 6.26 potential stereocentres per
    molecule, and nothing reads QED.

A3. sa_loss IS COMPUTED AND THEN DISCARDED. The trainer builds `sa_loss`
    from per-cloud SA-like scores (:3107-3136) and it does NOT appear in the
    `loss = (...)` sum — verified by string check on the sum block. It was
    deliberately removed and the removal is recorded at :3880. Note the
    scores it consumes are a SCAFFOLD surrogate
    (`_synthesize_sa_scores_from_clouds`, from n_atoms + n_distinct_elements)
    unless an outer scope injects real per-cloud SA, which nothing does. So
    this is not a one-line re-wire: the input is a placeholder.

A4. THE BOND HEAD IS PURELY PER-EDGE. `YuelBondDecoderHead` message-passes
    over the radius-graph edges and has no ring, cycle, or global term —
    verified, the strings ring/cycle/RingInfo do not appear in the class
    body. Ring SIZE is therefore unrepresentable: a per-edge classifier can
    rank "this pair is aromatic" but nothing in it can prefer closing a
    5-ring over a 3-ring, because that decision is a property of a CYCLE,
    not of any single edge.

THE RING NUMBER IS ALSO SMALLER THAN ROUND 8 REPORTED
-----------------------------------------------------
Round 8 recorded 89.8% generated 3-membered rings against 0.43% real, a
180x gap, and built OPT-5 (a ring-template prior) on it. The real denominator
was measured through RDKit ring perception, and real tmQM encodes EVERY
metal-ligand bond as DATIVE, which ring perception does not traverse — so the
real side never counted a single chelate ring. Like-for-like (dative->single,
same 9183-molecule subset both arms):

    generated          3-ring 89.81%   5/6  1.85%
    real as-written    3-ring  0.90%   5/6 94.78%   <- dative rings invisible
    real dative->single 3-ring 12.38%  5/6 80.50%   <- like-for-like
    ratio 7.3x, not 184x

The DIRECTION survives and ring size is still the right axis. The MAGNITUDE
is a fraction of what was reported, and 70.5% of the corpus does not convert
cleanly, which is recorded rather than averaged away.

THE RULE
--------
An entry leaves `blockers` ONLY if its question is answered on disk with a
path. Nothing here marks anything CLOSED. Every architecture item is written
as a hypothesis with its own measurement gate, NOT as a promised lift: no
PROJECTED number appears without a wall budget and a way to be wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TODO = REPO / "cfm_todo.json"

RING_CORR = "metrics/by_metric/cfm_opt2_ring_dative_correction.json"
RING_SCRIPT = "molmetal/scripts/wf_cfm_opt2_ring_dative_correction_2026_10_13.py"
OPT1 = "metrics/by_metric/cfm_opt1_stereo_suppression.json"
GEOM = "metrics/by_metric/cfm_generated_geometry_defect.json"
LIVENESS = "metrics/by_metric/cfm_loss_term_liveness.json"
GUIDANCE = "metrics/by_metric/cfm_l3_guidance_ab.json"
FORK_TEST = "molmetal/tests/test_sascorer_fork_divergence.py"


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))

    for p in (RING_CORR, RING_SCRIPT, OPT1, GEOM, LIVENESS):
        if not (REPO / p).is_file():
            print(f"REFUSING: {p} is not on disk", file=sys.stderr)
            return 2

    plan = d.get("optimisation_plan_2026_10_13")
    if not isinstance(plan, dict):
        print("REFUSING: optimisation_plan_2026_10_13 missing", file=sys.stderr)
        return 2

    plan["ARCHITECTURE_UPGRADE_2026_10_13"] = {
        "why": (
            "The five-item plan was entirely config / postprocess / loss-weight. "
            "None of it changed the model. The user asked for optimisation of "
            "the model's INTERNAL STRUCTURE (「基于现状要有对模型内结构的优化」). "
            "These four items do, each keyed to a structural fact verified "
            "against source in this pass rather than to a design document."
        ),
        "structural_facts_verified_2026_10_13": {
            "A1_bond_head_never_sees_a_noisy_sample": {
                "claim": (
                    "CFMJointModel.forward calls self.head(cloud) on the CLEAN "
                    "cloud and evaluates the velocity field at t_b = 1, which "
                    "under x_t = (1-t)x_0 + t x_1 is exactly x_1."
                ),
                "evidence": (
                    "train_cfm_yuelbond_wasserstein.py:1775 forward; "
                    "'out = self.head(cloud)'; 't_b = torch.ones(1, ...)' with "
                    "the in-code comment 't = 1 is exactly x_1 -- the cloud's "
                    "real coordinates'; a regex for '(1 - t) * x0' over the "
                    "whole forward body returns NO match."
                ),
                "consequence": (
                    "bond_ce trains a PERCEPTION module on clean CSD geometry. "
                    "At inference the same head must read geometry the velocity "
                    "field produced by integrating from noise. The objective "
                    "never asks the head to be robust along the trajectory it "
                    "is actually used on. This is the largest train/inference "
                    "gap in the model and it was absent from the previous plan."
                ),
                "corroborating_measurement": (
                    "Independent evidence that generated geometry is off the "
                    "training distribution: p01_pair_distance 1.131 A generated "
                    "vs 1.337 A real, and the aromatic Wiberg band starts at "
                    "~1.39 A -- the generated p01 sits BELOW the whole band. "
                    "Path: " + GEOM + " and "
                    "molmetal/reports/cfm_aromatic_bias_sweep_seed0/arm_bias0.0.json"
                ),
            },
            "A2_no_molecule_level_term": {
                "claim": (
                    "The loss sum is per-edge + regularisers. No term consumes a "
                    "decoded graph, an RDKit Mol, SMILES, a ring, or a cycle."
                ),
                "evidence": (
                    "train_cfm_yuelbond_wasserstein.py:3816 `loss = (` sums: "
                    "bond_ce_term, wasserstein_weight*w2_val, 0.05*cb_loss, "
                    "diversity_term, shortcut_distill_loss, consistency_term, "
                    "so3_term, edm_term, fm_term, assemble_term, entropy_term, "
                    "soft_valence_term. A string check for mol/smiles/qed/sa_/"
                    "ring/cycle over those term names returns nothing."
                ),
                "consequence": (
                    "Every measured molecule-level defect is invisible to the "
                    "objective by construction: 89.81% three-membered rings, "
                    "6.26 potential stereocentres per molecule, QED 0.0944. "
                    "bond_ce is a PER-EDGE cross-entropy and cannot see any of "
                    "them; a 3-ring and a 5-ring cost the same per edge."
                ),
                "measurement": (
                    "8 of 12 terms are exact zeros under the shipped config, "
                    "and of the 4 live ones only bond_ce touches chemistry. "
                    "Path: " + LIVENESS
                ),
            },
            "A3_sa_loss_computed_then_discarded": {
                "claim": (
                    "The trainer builds sa_loss at :3107-3136 and it does NOT "
                    "enter the loss sum; the removal is deliberate and recorded "
                    "at :3880."
                ),
                "evidence": (
                    "'sa_loss = torch.zeros' appears in the trainer; a string "
                    "check for 'sa_loss' over the loss-sum block returns False."
                ),
                "why_it_is_NOT_a_one_line_rewire": (
                    "The scores it consumes are a SCAFFOLD surrogate from "
                    "_synthesize_sa_scores_from_clouds (n_atoms + "
                    "n_distinct_elements), unless an outer scope injects real "
                    "per-cloud SA -- and nothing does. Re-adding the term as-is "
                    "would train on a placeholder that cannot see chemistry."
                ),
                "measurement": (
                    "Path: train_cfm_yuelbond_wasserstein.py:3091-3136 "
                    "(the Wf-404-A3 comment states the surrogate contract "
                    "explicitly)."
                ),
            },
            "A4_bond_head_is_purely_per_edge": {
                "claim": (
                    "YuelBondDecoderHead message-passes over radius-graph edges "
                    "and contains no ring, cycle, or global term."
                ),
                "evidence": (
                    "The strings 'ring', 'cycle', 'RingInfo' do not appear "
                    "anywhere in the YuelBondDecoderHead class body."
                ),
                "consequence": (
                    "Ring SIZE is unrepresentable in this head. A per-edge "
                    "classifier can rank 'this pair is aromatic' but nothing in "
                    "it can prefer closing a 5-ring over a 3-ring, because that "
                    "is a property of a CYCLE, not of any single edge. This is "
                    "why OPT-4 (a better per-edge head) cannot close the ring "
                    "defect even in principle -- and it is a STRUCTURAL argument, "
                    "not the load-ratio argument round 8 used."
                ),
            },
        },
        "items": [
            {
                "priority": 1,
                "id": "ARCH-1_trained_on_the_trajectory",
                "name": "Train the bond head on NOISY x_t, not only clean x_1",
                "targets": "A1 -- the largest train/inference gap in the model",
                "mechanism": (
                    "Sample t ~ U(eps,1), form x_t = (1-t)x_0 + t x_1 from the "
                    "training cloud, and compute bond_ce on the head's logits at "
                    "x_t rather than at x_1. The target labels are unchanged "
                    "(they come from tmQM's Wiberg blocks, which describe the "
                    "clean molecule); only the INPUT the head reads becomes "
                    "on-trajectory. Optionally mix clean and noisy per step so "
                    "the clean-geometry regime is not abandoned."
                ),
                "why_this_is_first": (
                    "Every other molecule-level term below is a NEW loss whose "
                    "gradient flows through the head. If the head cannot read "
                    "on-trajectory geometry, those gradients are being fitted "
                    "against a module that behaves differently at inference than "
                    "in training. Fix the substrate first."
                ),
                "cost": "No new hyperparameters needed beyond a t-distribution; "
                        "one forward pass per cloud as today (the cloud is "
                        "already built per step).",
                "decides": (
                    "Whether the head's failure is a CAPACITY problem or a "
                    "TRAIN/INFERENCE MISMATCH problem. This is the cheapest "
                    "possible answer and it reframes whether OPT-4's retrain "
                    "slot is worth spending at all."
                ),
                "falsifiable_prediction": (
                    "If A1 is the binding constraint, evaluating the EXISTING "
                    "checkpoint's head on x_t drawn from the training t-"
                    "distribution will show bond accuracy degrading as t "
                    "decreases, monotonically. That is a CPU/no-retrain "
                    "measurement on an existing checkpoint and it should be run "
                    "BEFORE any retrain, as the gate on this item."
                ),
                "gate_before_retrain": (
                    "Run the degradation curve first. If accuracy does NOT "
                    "degrade with t, A1 is not the binding constraint and this "
                    "item is downgraded rather than retrained."
                ),
                "risk": (
                    "Noising the input may make the head's job harder without "
                    "making it better at inference, if the real inference "
                    "distribution is narrower than the training t-distribution. "
                    "The gate above is designed to catch exactly that."
                ),
                "verdict_label": "PROJECTED — hypothesis, not measured lift. "
                                 "Gate is cheap and must pass first.",
            },
            {
                "priority": 2,
                "id": "ARCH-2_cycle_aware_bond_head",
                "name": "Give the bond head a cycle-level term so ring size is representable",
                "targets": "A4 + the ring-size defect",
                "mechanism": (
                    "Add a second-stage readout over the head's per-edge "
                    "probabilities that scores whole CYCLES rather than edges: "
                    "for each candidate cycle in the decoded candidate graph, "
                    "aggregate its edge probabilities and reward cycles of size "
                    "5 or 6, penalise size 3. This is a post-hoc module over "
                    "existing logits, so it can be built and measured WITHOUT "
                    "retraining the head -- which is the point: it separates "
                    "'the head lacks ring representation' from 'the head is "
                    "right and the sampler never asks it about 5-rings'."
                ),
                "why_it_is_structural_not_a_weight": (
                    "A4 shows the head has no cycle-level term at all. No "
                    "per-edge classifier, however good, can express 'prefer a "
                    "5-ring', because that is a property of a cycle. This is why "
                    "OPT-4 cannot close the ring defect in principle, not merely "
                    "with the current budget."
                ),
                "cost": "CPU. A decoder-side module over logits; no retrain.",
                "decides": (
                    "Whether ring size is a HEAD-CAPACITY defect or a SAMPLER-"
                    "SUPPLY defect. Both round 8 and the current plan assume the "
                    "latter; this measures it."
                ),
                "falsifiable_prediction": (
                    "If the head CAN rank cycles correctly when asked, then a "
                    "cycle-aware readout will move the 3-ring fraction toward "
                    "12.38% without retraining. If it cannot, the readout will "
                    "not move it and the defect is upstream in geometry."
                ),
                "measurement_gate": (
                    "Report generated 3-ring fraction and 5/6-membered fraction "
                    "against the CORRECTED like-for-like real arm (12.38% / "
                    "80.50%), NOT against round 8's 0.43% / 96.92%, which "
                    "measured through RDKit ring perception that does not "
                    "traverse the DATIVE bonds real tmQM uses. Path: " + RING_CORR
                ),
                "verdict_label": "PROJECTED — and the corrected target is 7.3x, "
                                 "not the 184x round 8 planned against.",
            },
            {
                "priority": 3,
                "id": "ARCH-3_molecule_level_objective",
                "name": "One differentiable molecule-level term in the loss",
                "targets": "A2 — the reason every measured defect is invisible",
                "mechanism": (
                    "BLK-C-L3's conclusion stands: guidance is the cheapest "
                    "3D-proven lever and it needs no retrain. But guidance acts "
                    "at SAMPLING time, so it cannot fix a model whose training "
                    "objective never looks at a molecule. This item adds ONE "
                    "molecule-level term. The candidate that needs no surrogate "
                    "placeholder is a differentiable ring-size / cycle term over "
                    "the head's own edge probabilities -- which is ARCH-2's "
                    "objective, read as a loss rather than a decode-time "
                    "readout. The alternative, a real SA term, is BLOCKED on A3: "
                    "the existing sa_loss consumes a scaffold surrogate, so "
                    "re-adding it would train on a placeholder."
                ),
                "why_not_just_turn_on_the_dead_terms": (
                    "8 of 12 loss terms are exact zeros under the shipped "
                    "config, and six of those eight are gated on a config value "
                    "that appears in ZERO config files. Turning them on is a "
                    "config edit, not a structural change, and BLK-C-L2 "
                    "requires reviving them BY THEIR OWN PREMISE."
                ),
                "cost": "ARCH-2's term reuses logits already computed per step, "
                        "so the marginal cost is the cycle enumeration.",
                "decides": "Whether this project ever has a molecule-level objective.",
                "explicitly_not_claimed": (
                    "This is NOT a promise that adding the term lifts QED or SA. "
                    "The measured SA lever is 1.2743 SA units attributable to "
                    "the stereocentre axis under a size-matched control (path: "
                    + OPT1 + "), and that was obtained by DELETING atoms -- a "
                    "score-level postprocess, not something a loss term should "
                    "be asked to imitate."
                ),
                "verdict_label": "PROJECTED — architecture, not a measured lift.",
            },
            {
                "priority": 4,
                "id": "ARCH-4_geometry_before_chemistry",
                "name": "Make the generated geometry in-distribution before optimising what is read from it",
                "targets": "A1's corollary — the head is asked to read geometry it was never trained on",
                "mechanism": (
                    "MEASURED and already partly shipped: every one of 24 "
                    "generated clouds carries at least one hard steric clash "
                    "against 0 of 24 real; mean closest contact 0.570 A vs "
                    "1.172 A real; metal-ligand first shell 1.234 A vs 1.953 A "
                    "real. Guidance moves every geometry metric toward real "
                    "(hard clashes 143->17, min distance 0.570->1.155) but 13 of "
                    "24 clouds still clash."
                ),
                "why_it_is_first_in_time_even_though_it_is_priority_4": (
                    "ARCH-1, ARCH-2 and ARCH-3 all train or read the head. If "
                    "the geometry stays out of distribution, they are all "
                    "optimising against a substrate that does not resemble "
                    "inference. The guidance A/B is at smoke scale (24 clouds, "
                    "20 steps, one checkpoint, one seed, one strength) and its "
                    "DECODE effect is explicitly unmeasured -- the next step is "
                    "a strength sweep plus a decode-enabled A/B, not a retrain. "
                    "Path: " + GUIDANCE
                ),
                "decides": (
                    "How much of the ring and stereocentre defect is upstream in "
                    "geometry rather than in the head. This is the confound that "
                    "has repeatedly made head-side and sampler-side "
                    "interpretations unfalsifiable."
                ),
                "verdict_label": "PARTLY MEASURED at smoke scale; decode effect NOT ON DISK.",
            },
        ],
        "what_this_program_does_NOT_claim": [
            "No PROJECTED QED, ADMET or SA lift is promised. Every architecture "
            "item is a hypothesis with a gate, and the gates are cheap enough "
            "to run before any retrain is authorised.",
            "ARCH-1's premise (train/inference mismatch) is a CODE fact -- the "
            "head is evaluated at t=1 and never at a noisy x_t. Its "
            "CONSEQUENCE is a hypothesis, and the degradation-curve gate is "
            "what separates them.",
            "The ring defect is 7.3x like-for-like, not the 184x round 8 "
            "reported, and 70.5% of the real corpus does not convert cleanly "
            "under the dative->single rewrite that makes the comparison "
            "possible at all.",
            "Nothing here reopens the withdrawn 22.8x pad_groups figure or "
            "changes any CLOSED entry.",
        ],
        "the_single_authorised_retrain_slot": (
            "STILL UNSPENT and still not earmarked. ARCH-1 through ARCH-4 are "
            "deliberately ordered so that the cheap gates run FIRST: the "
            "t-degradation curve (CPU, no retrain), the cycle-aware readout "
            "(CPU, no retrain), the guidance strength sweep with decode "
            "(GPU, short). If those gates show the defects are head-capacity "
            "rather than substrate or supply, the slot goes to ARCH-1's "
            "training change. If they do not, the slot should NOT be spent on a "
            "bond-head retrain at all -- which is the outcome round 8's "
            "deprioritisation of OPT-4 was reaching for by a weaker argument."
        ),
        "precedence_against_the_old_plan": (
            "ARCH items sit ABOVE OPT-1/OPT-4/OPT-5 and PARALLEL to OPT-3. "
            "OPT-3 (min_per_metal) remains the cheapest CONFIG-only item and "
            "still gates the precious-metal axis. OPT-1's decode-time arm is "
            "superseded by the corrected measurement: it is a score-level "
            "postprocess that deletes atoms, and the corrected figure is 1.2743 "
            "attributable SA units, not the 0 the broken scorer produced."
        ),
    }

    d["ring_size_correction_2026_10_13"] = {
        "what_round_8_reported": {
            "generated_3_membered": 0.8981,
            "real_3_membered": 0.0043,
            "ratio": 184.0,
            "source": "metrics/by_metric/cfm_opt2_ring_content.json",
        },
        "the_confound": (
            "RDKit ring perception does not traverse DATIVE bonds. Real tmQM "
            "encodes EVERY metal-ligand bond as DATIVE (->), so the as-written "
            "real profile contains ZERO metal-containing rings -- the chelate "
            "rings that dominate six-metal anticancer chemistry were never "
            "counted on the reference side."
        ),
        "corrected_like_for_like": {
            "generated_3_membered": 0.8981,
            "real_3_membered_same_subset_as_written": 0.0090,
            "real_3_membered_dative_to_single": 0.1238,
            "generated_5_or_6": 0.0185,
            "real_5_or_6_dative_to_single": 0.8050,
            "ratio_3_membered": 7.3,
        },
        "honest_limitation": (
            "Only 9183 of 31150 molecules (29.5%) survive DATIVE->SINGLE plus "
            "sanitisation. BOTH arms are computed on that same subset so the "
            "dative effect is not confounded with survivorship, but the subset "
            "is selected and 70.5% of the corpus is unmeasured in the corrected "
            "arm. A full-coverage arm needs ring perception that tolerates "
            "dative bonds, which RDKit does not offer."
        ),
        "what_survives": (
            "The direction and the axis. Generated still over-produces "
            "three-membered rings and under-produces 5/6-membered ones. What "
            "changes is the magnitude, and therefore any lift quoted against "
            "the 184x figure."
        ),
        "what_is_unchanged": (
            "Fragmentation was never the story (largest_frag_frac 0.9984 gen vs "
            "0.9996 real), and ring SIZE remains the right axis over ring count."
        ),
        "paths": [RING_CORR, RING_SCRIPT],
    }

    d["updated"] = (
        "2026-10-13 (v7: ARCHITECTURE_UPGRADE added — four model-internal items "
        "keyed to source-verified structural facts; ring-size finding corrected "
        "from 184x to 7.3x like-for-like after the DATIVE bond confound)"
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    back = json.loads(TODO.read_text(encoding="utf-8"))
    up = back["optimisation_plan_2026_10_13"]["ARCHITECTURE_UPGRADE_2026_10_13"]
    facts = up["structural_facts_verified_2026_10_13"]
    assert len(facts) == 4, f"expected 4 structural facts, got {len(facts)}"
    assert len(up["items"]) == 4, f"expected 4 arch items, got {len(up['items'])}"
    assert "ring_size_correction_2026_10_13" in back
    for it in up["items"]:
        assert "gate" in " ".join(it.keys()) or "decides" in it, (
            f"{it['id']} has neither a gate nor a decides clause -- every "
            "architecture item must be falsifiable"
        )
    print(f"OK: 4 structural facts, {len(up['items'])} architecture items")
    print(f"OK: ring correction recorded (184x -> {back['ring_size_correction_2026_10_13']['corrected_like_for_like']['ratio_3_membered']}x)")
    print(f"OK: blockers untouched: {[b['id'] for b in back['blockers']]}")
    print(f"OK: resolved entries untouched: {len(back.get('resolved_2026_10_12', [])) + len(back.get('resolved_2026_10_13', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
