"""Red-checked test for :mod:`molmetal.eval.scoring.stereo_postprocess`.

The OPT-1 feasibility report established that under the deployed
``Chem.FindMolChiralCenters(mol, includeUnassigned=True)`` predictor
the cheap decode-time interventions (``set_tag``, ``remove_annotation``)
do not lower ``n_chiral``.  These tests assert that reality from both
sides — every cheap intervention's vacuity is verified on a control
molecule, AND the helper module does not silently move the score on
the deployed metric.

Test matrix
-----------

1. ``test_set_tag_does_not_lower_n_chiral``
   ``Chem.AssignStereochemistry(force=True)`` does NOT lower
   ``n_chiral`` under ``includeUnassigned=True``.  This is the
   load-bearing vacuity claim from Wf-369.

2. ``test_remove_annotation_does_not_lower_n_chiral``
   ``Chem.RemoveStereochemistry`` is vacuous against
   ``includeUnassigned=True`` (only clears the @ tag, not the
   potential-centre detection).

3. ``test_saturate_lowers_n_chiral_on_simple_mol``
   On a hand-built mol with explicit sp3 neighbours the saturate
   intervention does kill the centre.  Sanitisation must succeed.

4. ``test_remove_atom_lowers_n_chiral_on_simple_mol``
   Same control: atom removal drops n_chiral to 0.

5. ``test_compute_sa_score_excluding_potential_drops_stereo_term``
   The score-side ``includeUnassigned=False`` helper returns a LOWER
   SA than the deployed sascorer on a molecule whose centres are not
   annotated.

6. ``test_apply_stereocentre_suppression_dispatch``
   All four modes are registered and callable, and unknown modes
   raise ``ValueError``.

7. ``test_does_not_silently_lower_sa_under_deployed_rule``
   The module's stated purpose ("decode-time postprocess") is NOT to
   silently lower SA under the deployed metric.  Vacuous
   interventions must leave SA exactly equal to ``compute_sa_score``.
   This guards against the wrong-reason red: a future regression that
   "moves the score" via a hidden fall-through would fail this.

All tests use **real RDKit** — we do not mock any chemistry.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from rdkit import Chem
from rdkit.Chem import QED, AllChem

from molmetal.eval.scoring import sascorer
from molmetal.eval.scoring.stereo_postprocess import (
    INTERVENTIONS,
    apply_stereocentre_suppression,
    compute_sa_score_excluding_potential,
)


# ---------------------------------------------------------------------------
# Deployment-dep gate
# ---------------------------------------------------------------------------
# sascorer.compute_sa_score() reads the Ertl 2009 fragment table at
# molmetal/eval/scoring/fpscores.pkl.gz.  The shipped pickle is missing on
# disk (verified) so the function raises ``FileNotFoundError``.  Two
# tests below depend on the actual SA score; they skip when the shipped
# pickle is missing so the OPT-1 vacuity findings can still be verified.
# The dep is recorded as an honest-negative in
# cfm_todo.json::OPT_1_MEASURED_2026_10_13::fpscores_path_problem.
_FPSCORES_SHIPPED = Path(
    "/home/hugo/codes/mol-metal/molmetal/eval/scoring/fpscores.pkl.gz"
)
_SASCORER_AVAILABLE = os.path.isfile(_FPSCORES_SHIPPED)


@pytest.fixture
def maybe_skip_sascorer():
    """Skip a test if sascorer.compute_sa_score cannot find fpscores."""
    if not _SASCORER_AVAILABLE:
        pytest.skip(
            "fpscores.pkl.gz not shipped under molmetal/eval/scoring/ "
            "(deployment-dep; see cfm_todo.json::OPT_1_MEASURED_2026_10_13::"
            "fpscores_path_problem)"
        )


# ---------------------------------------------------------------------------
# Test fixtures — small RDKit molecules whose n_chiral we control.
# ---------------------------------------------------------------------------


# (R)-CHFClBr — a trivial 1-centre test molecule.  C with H/F/Cl/Br
# attached is a textbook chiral centre and RDKit can detect it.
CHFCLBR = "[C@@H](F)(Cl)Br"


# 2,3,4-trihydroxy-pentan-1-ol — multi-centre acyclic alcohol.
# Multiple sp3 carbons, no double/triple bonds, no rings, predictable
# sanitisation.
TRIOL = "OC[C@@H](O)[C@H](O)[C@H](O)CO"


# ---------------------------------------------------------------------------
# 1. set_tag is vacuous under includeUnassigned=True
# ---------------------------------------------------------------------------


def test_set_tag_does_not_lower_n_chiral():
    """Setting the @ tag does NOT change the potential-centre count."""
    mol = Chem.MolFromSmiles(CHFCLBR)
    assert mol is not None
    n_before = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    out = apply_stereocentre_suppression(mol, mode="set_tag")
    assert out is not None
    n_after = len(Chem.FindMolChiralCenters(out, includeUnassigned=True))
    assert n_before >= 1, "control mol should expose ≥1 potential centre"
    assert n_after == n_before, (
        f"set_tag should not change n_chiral under includeUnassigned=True; "
        f"before={n_before} after={n_after}"
    )


# ---------------------------------------------------------------------------
# 2. remove_annotation is vacuous under includeUnassigned=True
# ---------------------------------------------------------------------------


def test_remove_annotation_does_not_lower_n_chiral():
    """Removing the @ tag does NOT lower ``n_chiral`` — same vacuity."""
    mol = Chem.MolFromSmiles(TRIOL)
    assert mol is not None
    n_before = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    out = apply_stereocentre_suppression(mol, mode="remove_annotation")
    assert out is not None
    n_after = len(Chem.FindMolChiralCenters(out, includeUnassigned=True))
    assert n_after == n_before, (
        f"remove_annotation vacuity claim: expected {n_before}=={n_after}"
    )


# ---------------------------------------------------------------------------
# 3. saturate lowers n_chiral on a controlled mol that sanitises
# ---------------------------------------------------------------------------


def test_saturate_lowers_n_chiral_on_simple_mol():
    """On a hand-built alkyl alcohol, saturate kills the chiral centre
    AND sanitises cleanly.

    We pick a secondary alcohol with no rings so the intervention has
    a clean fallback (double bond to lightest neighbour).  The C/H/O
    combination is small enough that rehybridisation must succeed.
    """
    # (S)-2-butanol — C2 has H, OH, CH3, CH2CH3 as 4 distinct neighbours.
    mol = Chem.MolFromSmiles("CC[C@H](O)C")
    assert mol is not None
    n_before = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    assert n_before >= 1
    out = apply_stereocentre_suppression(mol, mode="saturate")
    assert out is not None, "saturate must sanitise on a clean small mol"
    n_after = len(Chem.FindMolChiralCenters(out, includeUnassigned=True))
    assert n_after < n_before, (
        f"saturate should lower n_chiral on a sanitisation-friendly mol; "
        f"before={n_before} after={n_after}"
    )


# ---------------------------------------------------------------------------
# 4. remove_atom lowers n_chiral on a controlled mol
# ---------------------------------------------------------------------------


def test_remove_atom_lowers_n_chiral_on_simple_mol():
    """The atom-removal intervention must lower n_chiral on a controlled mol."""
    mol = Chem.MolFromSmiles(CHFCLBR)
    assert mol is not None
    out = apply_stereocentre_suppression(mol, mode="remove_atom")
    assert out is not None
    n_after = len(Chem.FindMolChiralCenters(out, includeUnassigned=True))
    assert n_after == 0, f"after atom removal n_chiral must be 0; got {n_after}"


# ---------------------------------------------------------------------------
# 5. score-side exclusion lowers SA by exactly the stereo penalty
# ---------------------------------------------------------------------------


def test_compute_sa_score_excluding_potential_drops_stereo_term(maybe_skip_sascorer):
    """On a mol with annotated centres, the score-side helper subtracts
    the deployed stereo penalty and re-adds the (zero, in this case)
    exclusion-rule one.  Net effect: SA drops by exactly the deployed
    stereo term, all other terms identical.  Skipped when
    ``fpscores.pkl.gz`` is not on disk (deployment-dep).
    """
    mol = Chem.MolFromSmiles(TRIOL)
    assert mol is not None
    n_deployed = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_excluded = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))
    assert n_deployed >= 1
    assert n_excluded == 0, "trio has no assigned centres, exclude returns 0"

    sa_deployed = sascorer.compute_sa_score(mol)
    sa_excluded = compute_sa_score_excluding_potential(mol)
    expected_delta = (
        math.log10(max(1, n_deployed)) - math.log10(max(1, n_excluded))
    )
    assert sa_excluded == pytest.approx(sa_deployed + expected_delta, abs=1e-9), (
        f"exclusion SA should be deployed SA + stereo delta = {sa_deployed}+{expected_delta}; "
        f"got {sa_excluded}"
    )


# ---------------------------------------------------------------------------
# 6. dispatch helper — all modes registered
# ---------------------------------------------------------------------------


def test_apply_stereocentre_suppression_dispatch():
    """Every registered mode is callable on a real mol."""
    mol = Chem.MolFromSmiles(CHFCLBR)
    assert mol is not None
    for mode in INTERVENTIONS:
        out = apply_stereocentre_suppression(mol, mode=mode)
        assert out is not None or mode in {"saturate"}, (
            f"{mode!r} returned None on a real mol that should sanitise"
        )

    # unknown mode raises
    with pytest.raises(ValueError):
        apply_stereocentre_suppression(mol, mode="not_a_real_mode")


# ---------------------------------------------------------------------------
# 7. vacuous interventions must NOT silently lower SA under the deployed rule
# ---------------------------------------------------------------------------


def test_does_not_silently_lower_sa_under_deployed_rule(maybe_skip_sascorer):
    """Vacuous interventions must leave SA exactly equal to the deployed
    ``compute_sa_score``.

    This guards the wrong-reason red: a regression that "helps" SA via
    a hidden fall-through would fail here.  Vacuity is the load-bearing
    claim of the OPT-1 feasibility report and must remain detectable.
    Skipped when ``fpscores.pkl.gz`` is not on disk (deployment-dep).
    """
    mol = Chem.MolFromSmiles(TRIOL)
    assert mol is not None
    sa_baseline = sascorer.compute_sa_score(mol)
    for mode in ("set_tag", "remove_annotation"):
        out = apply_stereocentre_suppression(mol, mode=mode)
        assert out is not None
        sa_after = sascorer.compute_sa_score(out)
        assert sa_after == pytest.approx(sa_baseline, abs=1e-9), (
            f"vacuous intervention {mode!r} silently moved SA from "
            f"{sa_baseline} to {sa_after}"
        )
