#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Correct OPT-1 in cfm_todo.json: it was measured through a broken scorer.

WHAT WENT WRONG
---------------
Round 8 closed OPT-1 as INFEASIBLE_AT_DECODE_TIME, reporting SA moving the
WRONG WAY (3.3039 -> 4.4189, +1.115) under every intervention that lowered
n_chiral. Those numbers were produced through
``molmetal/eval/scoring/sascorer.py``, a vendored fork whose fragment term
is a hardcoded constant:

  * the fork calls ``GetCountFingerprint`` on a ``fpSize=2048`` generator,
    producing 11-bit hashed ids, while the Ertl table is keyed on the 32-bit
    ids ``GetSparseCountFingerprint`` produces;
  * MEASURED: 0 of 66 molecules match a single fragment bit through the fork
    (canonical matches 41.35 bits/molecule on average);
  * with ``nf == 0`` the fork returns its ``else`` branch, ``score1 = 5.0``,
    for every molecule. The term Ertl built the method around is constant.

The fork is also unrunnable: its table lives at a .gitignore'd path that is
absent, so every call raises FileNotFoundError. Three scorer modules import
it (pocket2mol_score, softmol_score, targetdiff_score) and none can return a
score.

Under the canonical scorer the sign REVERSES. With a size-matched control --
dropping the same COUNT of non-stereocentre atoms, because SA carries a size
penalty and the intervention deletes 6.26 atoms per molecule -- the
stereocentre-attributable effect survives:

    baseline                          SA 6.4423  QED 0.0944  atoms 37.15
    A  remove stereocentre atoms      SA 4.8278  (-1.6145)  atoms 30.89
    B  size-matched control           SA 6.1021  (-0.3402)  atoms 31.02
    stereocentre-attributable        d_sa  -1.2743   d_qed  +0.0381

WHAT SURVIVES FROM ROUND 8
--------------------------
``set_tag`` and ``remove_annotation`` stay vacuous. That claim is
scorer-independent: ``FindMolChiralCenters(includeUnassigned=True)`` counts
assigned centres as well as unassigned ones, so tagging a centre does not
lower the count the penalty is computed from. Only the graph-modifying arms
change sign.

WHAT THIS DOES NOT CLAIM
------------------------
Dropping the atoms is not a chemistry improvement -- it deletes 6.26 atoms
per molecule and yields a 12-atom fragment in some cases. It is a
SCORE-LEVEL postprocess and it would be dishonest to ship it as a generator
improvement. What it establishes is narrower and load-bearing: the
stereocentre axis carries 1.27 SA units of real, attributable signal, so
there IS something for a sampler-side constraint to aim at. Whether a
generator can reach it without deleting chemistry is open.

THE RULE
--------
An entry leaves ``blockers`` ONLY if its question is answered on disk with a
path. Nothing here marks anything CLOSED; this script replaces a wrong
answer with a corrected one and re-opens the question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
TODO = REPO / "cfm_todo.json"

METRIC = "metrics/by_metric/cfm_opt1_stereo_suppression.json"
FORK_TEST = "molmetal/tests/test_sascorer_fork_divergence.py"
SCRIPT = "molmetal/scripts/wf_cfm_opt1_stereo_suppression_2026_10_13.py"
REPO_SA = "metrics/by_metric/cfm_free_generation_panel_seed0.json::panel.sa_mean"


def main() -> int:
    d = json.loads(TODO.read_text(encoding="utf-8"))

    for p in (METRIC, FORK_TEST, SCRIPT):
        if not (REPO / p).is_file():
            print(f"REFUSING: {p} is not on disk", file=sys.stderr)
            return 2

    opt1 = d.get("OPT_1_MEASURED_2026_10_13")
    if not isinstance(opt1, dict):
        print("REFUSING: OPT_1_MEASURED_2026_10_13 missing or not an object",
              file=sys.stderr)
        return 2

    old_verdict = opt1.get("MEASURED_2026_10_13", {}).get("verdict")
    opt1["CORRECTED_2026_10_13"] = {
        "what_was_wrong": (
            "The verdict '%s' was measured through a broken scorer. "
            "molmetal/eval/scoring/sascorer.py folds its Morgan fingerprint to "
            "fpSize=2048 (11-bit hashed ids) while the Ertl table is keyed on "
            "the 32-bit ids GetSparseCountFingerprint yields. MEASURED: 0 of 66 "
            "molecules match a single fragment bit through the fork; canonical "
            "matches 41.3485 bits/molecule (max 75). With nf==0 the fork takes "
            "its else branch and returns score1 = 5.0 for every molecule -- the "
            "fragment term Ertl built the method around is a CONSTANT. Four "
            "further divergences compound it: unmatched bits are skipped rather "
            "than charged the canonical -4; the divisor is distinct matched bits "
            "rather than total fragment count; score3 is a different function; "
            "and the fork adds a score4 term canonical does not have."
        ) % old_verdict,
        "measured_scorer_agrees_with_the_repo": (
            "Canonical SA over the same 66 SMILES is 6.442335, which is what "
            + REPO_SA + " records and what five dependent metric files quote. "
            "The fork gives 3.303854 on the identical cohort -- a 3.138481 SA "
            "unit understatement. Round 8's probe ran through the fork, which "
            "is why it reported a baseline (3.3039) that appears nowhere else "
            "in the repo."
        ),
        "the_fork_cannot_run_at_all": (
            "compute_sa_score opens molmetal/eval/scoring/fpscores.pkl.gz, "
            "which .gitignore line 111 hides as 'regenerable via sascorer.py's "
            "pickle download path'. There is no download path in the file "
            "(no urlretrieve / requests / urllib). The path is absent on disk, "
            "so every call raises FileNotFoundError. Three modules import it "
            "and none can return a score: pocket2mol_score, softmol_score, "
            "targetdiff_score. sa_penalty.py escapes only because it tries "
            "RDKit's bundled scorer FIRST."
        ),
        "corrected_measurement": {
            "baseline": {"sa": 6.4423, "qed": 0.0944, "atoms_per_mol": 37.15},
            "arm_A_remove_stereocentre_atoms": {
                "sa": 4.8278, "d_sa": -1.6145, "atoms_per_mol": 30.89,
                "n_survived": 66,
            },
            "arm_B_size_matched_control": {
                "sa": 6.1021, "d_sa": -0.3402, "atoms_per_mol": 31.02,
                "what_it_is": (
                    "drops the SAME COUNT of atoms chosen from the "
                    "NON-stereocentre atoms. SA carries a size penalty "
                    "(nAtoms**1.005 - nAtoms) and the intervention deletes 6.26 "
                    "atoms per molecule, so without this arm an SA improvement "
                    "would be indistinguishable from simply having fewer atoms."
                ),
            },
            "stereocentre_attributable": {"d_sa": -1.2743, "d_qed": +0.0381},
            "path": METRIC,
        },
        "corrected_verdict": (
            "OPT-1 IS EXECUTABLE AT DECODE TIME AND THE SIGN IS THE OPPOSITE OF "
            "WHAT ROUND 8 REPORTED. Dropping the stereocentre atoms LOWERS SA "
            "by 1.61 (4.83 vs 6.44) and lifts QED by 0.10 under the canonical "
            "scorer. 1.2743 SA units of that survives a size-matched control, so "
            "the effect is attributable to WHICH atoms were removed, not how "
            "many. Round 8's honest-negative is an artefact of the broken fork."
        ),
        "what_survives_from_round_8": (
            "set_tag and remove_annotation remain VACUOUS, and that is "
            "scorer-independent: FindMolChiralCenters(includeUnassigned=True) "
            "counts assigned centres too, so tagging a centre cannot lower the "
            "count the penalty reads. The vacuous arms still yield zero. Only "
            "the graph-modifying arms change sign."
        ),
        "what_this_does_NOT_claim": (
            "Dropping the atoms is a SCORE-LEVEL postprocess, not a chemistry "
            "improvement: it deletes 6.26 atoms per molecule and leaves a "
            "12-atom fragment in the worst cases. Shipping it as a generator "
            "improvement would be dishonest. What it establishes is narrower "
            "and load-bearing -- the stereocentre axis carries 1.27 SA units of "
            "real attributable signal, so a sampler-side constraint has "
            "something to aim at. Whether a generator can reach it WITHOUT "
            "deleting chemistry is open and is the re-opened question."
        ),
        "score_level_edit_still_pending_user_verdict": (
            "The includeUnassigned=True -> False metric edit is UNCHANGED and "
            "still requires a user verdict. It is a metric REDEFINITION: it "
            "would close the centre-count gap by changing what is counted, not "
            "by improving the generator. Round 8's literature objection stands "
            "-- chiral Ru(III) anticancer ratios of 3-260x are reported, and "
            "oxaliplatin (R,R-DACH) is the FDA-approved chiral Pt exception."
        ),
        "path": METRIC,
        "regenerate_with": SCRIPT,
        "red_checked_by": FORK_TEST + " (8 tests; see its docstring for the "
        "three mutation runs and the two wrong-reason attempts the mutation "
        "pass caught)",
    }

    # The stale headline table must not be left readable as current.
    block = opt1.setdefault("MEASURED_2026_10_13", {})
    if isinstance(block, dict):
        block["verdict"] = (
            "SUPERSEDED 2026-10-13 -- see CORRECTED_2026_10_13. This verdict "
            "was measured through a scorer whose fragment term is a constant "
            "(0 of 66 fragment matches). The SA column below is on the broken "
            "fork's scale and must not be quoted."
        )
        block["SA_SCALE_WARNING"] = (
            "Every SA number in this block is on the in-tree fork's scale "
            "(baseline 3.3039). The repo's SA scale is 6.4423. Do not mix them."
        )

    d.setdefault("sascorer_fork_defect_2026_10_13", {
        "summary": (
            "molmetal/eval/scoring/sascorer.py does not implement Ertl 2009 and "
            "cannot run. Its fragment term is a hardcoded 5.0 for every "
            "molecule measured, because a folded fpSize=2048 Morgan fingerprint "
            "can never hit a table keyed on sparse 32-bit ids."
        ),
        "measured": {
            "cohort": "molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv (66 rows)",
            "fork_mean_sa": 3.303854,
            "canonical_mean_sa": 6.442335,
            "understatement_sa_units": 3.138481,
            "fork_fragment_matches_per_mol": 0.0,
            "canonical_fragment_matches_per_mol": 41.3485,
            "molecules_with_any_fork_match": "0 / 66",
        },
        "cannot_run": {
            "reason": "fpscores.pkl.gz is at a .gitignore'd path and absent; "
                      "compute_sa_score raises FileNotFoundError on every call",
            "modules_that_import_it_and_cannot_score": [
                "molmetal/eval/scoring/pocket2mol_score.py",
                "molmetal/eval/scoring/softmol_score.py",
                "molmetal/eval/scoring/targetdiff_score.py",
            ],
            "the_one_escapee": "molmetal/adapters/flow_matching_lipman/sa_penalty.py "
            "-- it tries RDKit's bundled scorer FIRST and only falls through "
            "to the fork",
        },
        "still_to_decide": (
            "FIX (correct the fork to sparse + the three formula terms) or "
            "DELETE (make every caller use RDKit's bundled scorer, as "
            "sa_penalty.py already does). NOT DONE HERE -- both change what "
            "other numbers in the repo mean, and the caller sweep is wider than "
            "this pass. Recorded rather than silently repaired."
        ),
        "red_checked_by": FORK_TEST,
        "paths": [FORK_TEST, METRIC, REPO_SA],
    })

    d["updated"] = (
        "2026-10-13 (OPT-1 corrected: round 8's honest-negative was measured "
        "through a broken SA fork whose fragment term is a constant; under the "
        "canonical scorer the sign reverses and 1.2743 SA units survive a "
        "size-matched control)"
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(TODO, json.dumps(d, indent=2, ensure_ascii=False) + "\n")

    back = json.loads(TODO.read_text(encoding="utf-8"))
    assert "CORRECTED_2026_10_13" in back["OPT_1_MEASURED_2026_10_13"]
    assert back["OPT_1_MEASURED_2026_10_13"]["MEASURED_2026_10_13"]["verdict"].startswith(
        "SUPERSEDED"
    )
    print("OK: OPT_1_MEASURED_2026_10_13 carries the correction and the old "
          "verdict is marked SUPERSEDED")
    print("OK: sascorer_fork_defect_2026_10_13 present")
    print(f"blockers still open: {[b['id'] for b in back['blockers']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
