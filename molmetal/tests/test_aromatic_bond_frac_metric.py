# SPDX-License-Identifier: Apache-2.0
"""``aromatic_bond_frac`` — the metric the BLK-C-M4 offset has to be read with.

WHY THIS EXISTS
---------------
The aromatic logit offset changes which bonds the DECODER draws as AROMATIC.
Until 2026-10-12 nothing in the property panel recorded aromaticity at all, so
a sweep of that offset would have reported only decode rate -- a number that
went UP for the wrong reason in the L3 guidance sweep (0.344 -> 0.828 while
QED fell 0.2358 -> 0.1327). A lever with no metric that can see it is a lever
that gets aimed blind.

So the definition is pinned here rather than left to whichever implementation
happens to be edited next.

THE DEFINITION, and the two things it deliberately is NOT
--------------------------------------------------------
Per MOLECULE aromatic-bond fraction, then averaged over the cohort.

1. NOT the pooled ratio ``sum(aromatic) / sum(bonds)``. That is dominated by
   the largest molecules in the cohort, so a run could raise it by emitting a
   few big aromatic molecules while its typical molecule stayed flat. The
   two disagree on exactly the cohorts where they must not.

2. NOT skipping zero-bond molecules. If they were skipped, a cohort would
   IMPROVE by decoding fewer real molecules -- the survivor-conditioning
   defect ``cfm_l3_guidance_sweep.json`` already documents (52 docked records
   with ``_survivor_conditioned: true`` and ``_mean_excludes: 14``).

RED CHECK: change the accumulation to a pooled ``sum/sum`` and
``test_pooled_and_per_molecule_disagree`` goes red; make the zero-bond case
skip and ``test_zero_bond_molecule_scores_zero`` goes red.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cfm_checkpoint_benchmark import TARGETS, score_cohort  # noqa: E402

BENZENE = "c1ccccc1"
CYCLOHEXANE = "C1CCCCC1"
ETHANOL = "CCO"
ETHANE = "CC"
# A single heavy atom has ZERO bonds. `CC` (ethane) has ONE, so it does not
# exercise the zero-bond branch -- an earlier version of this file used ethane
# and the test passed against a mutation that appended `None` instead of 0.0,
# because the branch was never reached. `C` is methane with implicit Hs; the
# heavy-atom bond count RDKit reports is 0, which is what the branch tests.
ZERO_BOND = "C"
# Standalone metal ion, the shape a fragment decode collapses to. Also 0 bonds.
ZERO_BOND_METAL = "[Pt]"


def test_benzene_scores_one_and_cyclohexane_scores_zero():
    r = score_cohort("t", [BENZENE, CYCLOHEXANE])
    assert r["aromatic_bond_frac"]["mean"] == 0.5, (
        "benzene must score 1.0 and cyclohexane 0.0, so the pair averages 0.5; "
        f"got {r['aromatic_bond_frac']['mean']}"
    )


def test_zero_bond_molecule_scores_zero_rather_than_being_skipped():
    """Skipping would let a cohort improve by decoding less."""
    assert Chem.MolFromSmiles(ZERO_BOND).GetNumBonds() == 0, (
        "the premise molecule must have zero bonds or this test proves nothing"
    )
    with_zero = score_cohort("t", [BENZENE, ZERO_BOND])
    without_zero = score_cohort("t", [BENZENE])
    assert with_zero["aromatic_bond_frac"]["mean"] == 0.5
    assert without_zero["aromatic_bond_frac"]["mean"] == 1.0
    assert with_zero["aromatic_bond_frac"]["n"] == 2, (
        "the zero-bond molecule must still be COUNTED; a cohort that drops it "
        "improves its mean without generating anything better"
    )
    # The fragment shape a collapsed decode produces.
    frag = score_cohort("t", [BENZENE, ZERO_BOND_METAL])
    assert frag["aromatic_bond_frac"]["mean"] == 0.5
    assert frag["aromatic_bond_frac"]["n"] == 2


def test_metric_is_present_for_every_cohort_not_silently_absent():
    """Guards the ``except: pass`` around the computation.

    The first red-check attempt changed the zero-bond branch to append
    ``None`` instead of 0.0. That did NOT turn any test red: ``list.append
    (None)`` succeeds, the later ``st.mean`` then raises, and the surrounding
    ``except Exception: pass`` swallowed it -- so the metric simply VANISHED
    from the output and every mean-reading assertion above still passed on the
    cohorts that had bonds.

    That is the silent-loss failure mode this file exists to prevent: an
    absent metric reads exactly like a flat one. So the presence of the key
    is asserted directly, on a cohort that has no zero-bond molecule at all.
    """
    r = score_cohort("t", [BENZENE, CYCLOHEXANE, ETHANOL])
    assert "aromatic_bond_frac" in r, (
        "the metric vanished from the output entirely; a missing metric reads "
        "as 'no change', which is how a dead measurement gets reported as a "
        "null result"
    )
    assert r["aromatic_bond_frac"]["mean"] is not None
    assert r["aromatic_bond_frac"]["n"] == 3


def test_pooled_and_per_molecule_disagree_where_it_matters():
    """The two definitions part company on a small-molecule + big-ring cohort.

    One molecule with six aromatic bonds and one with none:
      per-molecule mean = (1.0 + 0.0) / 2 = 0.5
      pooled ratio     = 6 / (6 + 2)   = 0.75
    A big benzene next to a small alkane is the ordinary shape of a decoded
    cohort, so the two are NOT interchangeable and the panel must be the
    per-molecule one.
    """
    r = score_cohort("t", [BENZENE, ETHANE])
    per_molecule = r["aromatic_bond_frac"]["mean"]
    # 6 aromatic bonds out of 6 (benzene) + 1 (ethane) = 6/7 pooled.
    pooled = 6 / 7
    assert per_molecule == 0.5
    assert abs(pooled - per_molecule) > 0.2, (
        "this cohort is supposed to separate the two definitions; if it no "
        "longer does, the test is not testing the thing it claims"
    )


def test_target_is_the_measured_real_tmqm_band_not_a_guess():
    spec = TARGETS["aromatic_bond_frac"]
    assert spec["direction"] == "higher"
    assert spec["target"] == 0.454591, (
        "the target must stay pinned to the MEASURED real tmQM six-metal band "
        f"(n=31,150), not drift to a literature-sounding number; got {spec['target']}"
    )
    assert "MEASURED" in spec["source"]


def test_a_generated_like_cohort_scores_far_below_the_target():
    """A sanity check on the metric's scale against a known-bad cohort.

    Generated CFM output measured 0.001229 aromatic bond fraction pre-fix. An
    alkane/decane cohort is the same story: near zero, far from 0.454591.
    If this test ever fails, the metric is reading something other than
    aromaticity.
    """
    r = score_cohort("t", [ETHANE, ETHANOL, "CCCCCCCC", "CC(C)C"])
    assert r["aromatic_bond_frac"]["mean"] == 0.0
    assert r["aromatic_bond_frac"]["meets_target"] is False


def test_unparseable_smiles_do_not_count_toward_n():
    """n is the PARSED count; a failed parse must not dilute the mean."""
    r = score_cohort("t", [BENZENE, "not_a_molecule", CYCLOHEXANE])
    assert r["n_parsed"] == 2
    assert r["n_failed"] == 1
    assert r["aromatic_bond_frac"]["n"] == 2
    assert r["aromatic_bond_frac"]["mean"] == 0.5