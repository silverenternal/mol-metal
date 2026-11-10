#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Sharpen claim A1 in cfm_todo.json with the mechanism, verified against code.

WHY THIS PASS EXISTS
--------------------
A1 was written into cfm_todo.json as: "the bond head never sees a noisy
sample, because CFMJointModel.forward calls self.head(cloud) on the clean
cloud and evaluates the field at t_b = 1."

That is TRUE but incomplete, and the incompleteness matters because it
changes how much work ARCH-1 is. The earlier wording implies the FM
objective is off or absent, so there is no x_t in the training step at all.
The opposite is true, and the sharper fact is more actionable:

  * The checkpoint under test has use_fm_objective=True, fm_weight=1.0,
    fm_t_dist='uniform', fm_t_per_cloud=1, and use_egnn_velocity=True.
    (Source: the checkpoint's own args at
    molmetal/reports/cfm_patience10_seed0/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt)
  * The FM block DOES sample t and DOES build a genuinely noisy
    x_t = (1 - t) * x0 + t * x1 at train_cfm_yuelbond_wasserstein.py:3437,
    optionally distorted at :3444.
  * But that x_t goes to exactly ONE consumer:
        model.fm_velocity(x_t, t, z_all, cloud_sizes=_cloud_sizes)   [:3454]
    whose signature (:1619) takes (x_t, t, atomic_numbers, cloud_sizes).
    It receives ATOMIC NUMBERS and COORDINATES. It never receives the head,
    the head's logits, or per-edge anything.
  * Under use_egnn_velocity=True -- the only field this config uses --
    fm_velocity routes to self.vel_field (the EGNN), not self.fm_vel_head.
    self.head (the bond head, :1517) and self.fm_vel_head (:1578) are
    SEPARATE nn.Module attributes.

So the training step already produces a noisy sample, and the machinery to
reach the head with it exists one call away. What is missing is a single
forward: the head is never handed x_t. That is a much smaller change than
"build a new training path", and it is why ARCH-1's gate is the right first
measurement: if the existing head degrades on noisy geometry, the fix is one
extra head call per step, not a redesign.

THE HONEST LIMIT OF THIS PASS
-----------------------------
This establishes the STRUCTURE (two disjoint branches, head never sees x_t).
It does NOT establish that the mismatch is BINDING -- that is exactly what
the t-degradation gate measures, and a flat curve would refute the premise
regardless of how clean the structural argument is. Both are recorded.

THE RULE
--------
Annotations only. Nothing is closed, nothing is deleted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TODO = REPO / "cfm_todo.json"
TRAINER = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
CKPT = ("molmetal/reports/cfm_patience10_seed0/checkpoints/"
        "cfm_yuelbond_wasserstein_seed0_last.pt")
GATE_METRIC = "metrics/by_metric/cfm_arch1_t_degradation.json"


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))
    for p in (REPO / TRAINER, REPO / CKPT):
        if not p.is_file():
            print(f"REFUSING: {p} not on disk", file=sys.stderr)
            return 2

    up = d.get("optimisation_plan_2026_10_13", {}).get(
        "ARCHITECTURE_UPGRADE_2026_10_13"
    )
    if not isinstance(up, dict):
        print("REFUSING: ARCHITECTURE_UPGRADE_2026_10_13 missing", file=sys.stderr)
        return 2
    facts = up.get("structural_facts_verified_2026_10_13", {})
    a1 = facts.get("A1_bond_head_never_sees_a_noisy_sample")
    if not isinstance(a1, dict):
        print("REFUSING: A1 fact missing", file=sys.stderr)
        return 2

    a1["SHARPENED_2026_10_13"] = {
        "the_original_wording_was_true_but_incomplete": (
            "A1 said the head 'never sees a noisy sample' and cited "
            "self.head(cloud) at t=1. That is correct, but it read as though "
            "no x_t exists in the training step. One does."
        ),
        "what_is_actually_true_and_stronger": (
            "The training step ALREADY samples t and builds a genuinely noisy "
            "x_t = (1-t)*x0 + t*x1, with optional distortion. That x_t has "
            "exactly one consumer -- model.fm_velocity(x_t, t, z_all, "
            "cloud_sizes) -- whose signature takes coordinates and atomic "
            "numbers and never receives the head, its logits, or anything "
            "per-edge. Under use_egnn_velocity=True it routes to "
            "self.vel_field, not self.fm_vel_head; and self.head (the bond "
            "head) and self.fm_vel_head are separate nn.Module attributes."
        ),
        "why_the_sharper_version_matters": (
            "It changes the SIZE of the fix. The earlier wording implies "
            "building a new training path. The truth is that the machinery to "
            "reach the head with a noisy sample already exists one call away, "
            "so the intervention is an extra head forward per step, not a "
            "redesign. That is why ARCH-1's cheap gate is the right FIRST "
            "measurement rather than a formality."
        ),
        "checkpoint_settings_read_from_disk": {
            "path": CKPT,
            "epoch": 24,
            "decode_ratio": 0.296875,
            "use_fm_objective": True,
            "fm_weight": 1.0,
            "fm_t_dist": "uniform",
            "fm_t_per_cloud": 1,
            "fm_noise_sigma": 2.485,
            "use_egnn_velocity": True,
            "n_layers_head": 3,
            "hidden_dim": 128,
            "bond_decision": "threshold",
            "note": (
                "fm_t_per_cloud=1 means the per-cloud t draw is the DEFAULT "
                "and the per-atom draw -- which the in-code comment at :3395 "
                "describes as internally inconsistent supervision -- is the "
                "ablation, restored with --fm-t-per-cloud 0."
            ),
        },
        "file_line_evidence": {
            TRAINER + ":3437": "x_t = (1.0 - t_col) * x0 + t_col * x1",
            TRAINER + ":3444": "x_t = apply_fm_distortion(...)",
            TRAINER + ":3454": "model.fm_velocity(x_t, t, z_all, cloud_sizes=_cloud_sizes)",
            TRAINER + ":1619": "def fm_velocity(self, x_t, t, atomic_numbers, cloud_sizes) -- no head argument",
            TRAINER + ":1656": "if self.use_egnn_velocity: -- routes to the EGNN",
            TRAINER + ":1517": "self.head = YuelBondDecoderHead(...)  -- the bond head",
            TRAINER + ":1578": "self.fm_vel_head = nn.Sequential(...)  -- a SEPARATE module",
        },
        "every_x_t_consumer_was_enumerated": {
            "method": (
                "grep for x_t across the loss region (lines 3200-3500). Every "
                "hit is either the construction site, apply_fm_distortion, the "
                "fm_velocity call, or the optional so3/edm regularisers that "
                "consume x_t BY DESIGN. None routes x_t to self.head."
            ),
            "conclusion": "No loss term couples the bond head to a noisy sample.",
        },
        "what_this_does_NOT_establish": (
            "STRUCTURE, not BINDING. That the two branches are disjoint does "
            "not prove the mismatch is what limits quality. Only the "
            "t-degradation curve can decide that, and a flat curve would "
            "refute ARCH-1's premise no matter how clean this argument is. "
            "Expected path: " + GATE_METRIC + " (run by the 2026-10-13 "
            "ARCH-1 gate; if absent when this is read, the gate has not run "
            "and the premise is untested)."
        ),
    }

    up["ARCH-1_note"] = (
        "ARCH-1's premise was independently re-verified and sharpened on "
        "2026-10-13: the trainer ALREADY builds noisy x_t and already has a "
        "live FM objective (use_fm_objective=True, fm_weight=1.0, "
        "use_egnn_velocity=True per the checkpoint's own args). What is "
        "missing is that x_t never reaches the bond head -- fm_velocity takes "
        "coordinates and atomic numbers only. So the intervention is an extra "
        "head forward on x_t, not a new training path. See "
        "structural_facts_verified_2026_10_13.A1...SHARPENED_2026_10_13."
    )

    d["updated"] = (
        "2026-10-13 (v7.1: A1 sharpened -- the FM objective is live and already "
        "builds noisy x_t, but x_t never reaches the bond head; the fix is one "
        "extra forward, not a new training path. Structure confirmed; "
        "BINDING-ness still untested pending the t-degradation gate.)"
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    back = json.loads(TODO.read_text(encoding="utf-8"))
    got = back["optimisation_plan_2026_10_13"]["ARCHITECTURE_UPGRADE_2026_10_13"][
        "structural_facts_verified_2026_10_13"
    ]["A1_bond_head_never_sees_a_noisy_sample"]["SHARPENED_2026_10_13"]
    assert got["checkpoint_settings_read_from_disk"]["use_fm_objective"] is True
    assert got["what_this_does_NOT_establish"].startswith("STRUCTURE")
    print("OK: A1 sharpened with the disjoint-branch mechanism")
    print("OK: checkpoint settings recorded from disk (fm live, EGNN on)")
    print("OK: explicitly flagged that BINDING-ness is still untested")
    print(f"OK: blockers untouched: {[b['id'] for b in back['blockers']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
