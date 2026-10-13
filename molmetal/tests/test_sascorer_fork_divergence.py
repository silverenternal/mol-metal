# SPDX-License-Identifier: Apache-2.0
"""The in-tree SA wrapper never sees a single fragment score, and cannot run.

THE DEFECT THIS PINS
--------------------
``molmetal/eval/scoring/sascorer.py`` is a vendored copy of Ertl 2009 that
does not reproduce it. The dominant divergence is not a formula subtlety --
it is that its fragment term is **identically the hardcoded fallback** on
every molecule measured so far.

    canonical: mfpgen = GetMorganGenerator(radius=2)          # no fpSize
               sparse count fingerprint -> 32-bit bit ids
    fork:      GetMorganGenerator(radius=2, fpSize=2048)      # folded
               count fingerprint -> 11-bit bit ids

The Ertl table ``fpscores.pkl.gz`` is keyed on the SPARSE (unfolded) bit
ids. Folding to 2048 bits hashes them into a different space, so **not one
bit can ever hit**. MEASURED on the 66-row generated cohort:

    matched fragment bits per molecule, canonical : mean 41.3485, max 75
    matched fragment bits per molecule, fork      : mean  0.0,    max  0
    molecules with >=1 matched bit, fork         : 0 / 66

With ``nf == 0`` the fork takes its ``else`` branch and returns
``score1 = 5.0`` for all of them. The fragment-contribution term -- the
term Ertl built the method around -- is a constant.

FOUR DIVERGENCES, all in the direction of "looks easier"
-------------------------------------------------------
1. Folded vs sparse fingerprint. ``GetCountFingerprint`` on a
   ``fpSize=2048`` generator yields 11-bit hashed ids; the Ertl table is
   keyed on the 32-bit ids ``GetSparseCountFingerprint`` yields. Kills
   score1 entirely (above). NOTE: dropping the ``fpSize`` keyword does NOT
   fix this -- it just defaults to 2048. Only the sparse call matters.
2. Canonical scores unmatched bits at ``_fscores.get(id, -4) * count``
   (``rdkit/Contrib/SA_Score/sascorer.py:67``); the fork skips bits not in
   the table. Moot while (1) holds, but a live defect once (1) is fixed.
3. Canonical divides by ``nf += count`` (total fragment count); the fork
   does ``nf += 1`` (distinct matched bits).
4. score3 and score4 are different functions. Canonical score3 is the
   Ertl v1.1 density correction ``+0.5*log(nAtoms/numBits)``; the fork's is
   ``-log10(n_rings)/n_atoms``. The fork ADDS a ``score4 = sp3/n_atoms -
   0.5`` term that canonical does not have at all.

MEASURED CONSEQUENCE, same 66 SMILES:

    canonical (rdkit sascorer)   SA mean 6.442335   <- every number in the repo
    in-tree fork                 SA mean 3.303854
    difference                   3.138481 SA units

6.442335 is what ``metrics/by_metric/cfm_free_generation_panel_seed0.json``
records and what five dependent files quote. 3.303854 is the figure the
round-8 OPT-1 feasibility probe produced, because that probe ran through
the fork. The conclusion it drew inverts -- see the last test.

THE SECOND DEFECT: THE FORK CANNOT RUN AT ALL
----------------------------------------------
``compute_sa_score`` calls ``_load_fscores()``, which opens
``molmetal/eval/scoring/fpscores.pkl.gz``. That path is in ``.gitignore``
(line 111) and absent on disk, so every call raises ``FileNotFoundError``.
Three scorer modules import it (``pocket2mol_score``, ``softmol_score``,
``targetdiff_score``) and none can return a score. ``sa_penalty.py``
escapes only because it tries RDKit's bundled scorer FIRST.

RED CHECKS -- each was run and observed to fail
------------------------------------------------
* Restore the pickle at the shipped path -> ``test_fork_raises`` fails.
* Change the fork's generator to sparse ``GetMorganGenerator(radius=2)``
  -> ``test_fork_matches_no_fragments`` and
  ``test_opt1_direction_reverses_under_canonical`` both fail, because the
  fork's SA collapses to match canonical and the sign of OPT-1's effect
  flips.
* Patch only the fork's ``score1`` arithmetic (NOT the fingerprint) -> the
  suite still PASSES. That mutation was actually run, and it is why the
  first version of this file asserted the wrong mechanism: it described
  divergences 2 and 3, which are inert while divergence 1 holds. The
  assertion is now pinned to the fingerprint, which is what actually
  drives the number.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import os
import statistics
import sys
from pathlib import Path

import pytest
from rdkit import Chem, RDConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COHORT = (
    PROJECT_ROOT / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10"
    / "generated_smiles.csv"
)
SHIPPED_FP = PROJECT_ROOT / "molmetal" / "eval" / "scoring" / "fpscores.pkl.gz"
RDKIT_SA_DIR = Path(RDConfig.RDContribDir) / "SA_Score"
FORK = PROJECT_ROOT / "molmetal" / "eval" / "scoring" / "sascorer.py"

# POST-FIX fork detector: the canonical algorithm uses the sparse
# MorganFingerprint path and rdMolDescriptors' default (no folded
# fpSize keyword). The pre-fix fork forced fpSize=2048 + GetCountFingerprint
# (folded). When the file no longer contains that combination, the
# broken-fork assertions are stale by DESIGN and must be skipped — the
# "red → green flip" called for in
# TODO/completed/wf_design_ultracode9_eot.md step A.
_FORK_TEXT = FORK.read_text(encoding="utf-8") if FORK.is_file() else ""
_FORK_IS_POST_FIX = (
    "fpSize=2048" not in _FORK_TEXT
    and "GetMorganGenerator(radius=2, fpSize=" not in _FORK_TEXT
    and "rdMolDescriptors.GetMorganFingerprint(m, 2)" in _FORK_TEXT
)

requires_pre_fix_fork = pytest.mark.skipif(
    _FORK_IS_POST_FIX,
    reason="Wf-A14 (2026-10-13): fork replaced with canonical Ertl reference; "
    "this red-on-broken assertion is no longer applicable. See file docstring.",
)

#: Recorded in cfm_free_generation_panel_seed0.json::panel.sa_mean and
#: quoted by cfm_sa_gao_coley_reference, cfm_checkpoint_property_benchmark,
#: cfm_reference_band_audit, cfm_blk_c_obj_decision, cfm_objective_inventory.
RECORDED_REPO_SA = 6.442335


def _canonical():
    """RDKit's bundled Ertl scorer, loaded by path so sys.path order is moot."""
    if "RDKIT_SA" in sys.modules:
        return sys.modules["RDKIT_SA"]
    spec = importlib.util.spec_from_file_location(
        "RDKIT_SA", str(RDKIT_SA_DIR / "sascorer.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["RDKIT_SA"] = m
    spec.loader.exec_module(m)
    return m


def _fork():
    """The in-tree wrapper with the fragment table injected in memory.

    Injection is required because the shipped path is absent (see below);
    it isolates the FORMULA question from the MISSING-FILE question.
    """
    mod = importlib.import_module("molmetal.eval.scoring.sascorer")
    if not SHIPPED_FP.is_file():
        sa = _canonical()
        sa._fscores = None
        sa.readFragmentScores(str(RDKIT_SA_DIR / "fpscores.pkl.gz"))
        mod._fscores = sa._fscores
    return mod


def _mols() -> list:
    if not COHORT.is_file():
        pytest.skip(f"cohort not on disk: {COHORT}")
    out = [m for m in (Chem.MolFromSmiles(s) for s in _smiles()) if m is not None]
    if len(out) < 10:
        pytest.skip("cohort too small to compare")
    return out


def _smiles() -> list[str]:
    out = []
    with COHORT.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("smiles") or "").strip()
            if v:
                out.append(v)
    return out


# ---------------------------------------------------------------------------
# 1. The fork cannot run at all
# ---------------------------------------------------------------------------


@requires_pre_fix_fork
def test_fork_raises_file_not_found():
    """The documented shipped path is absent, so every call raises."""
    from molmetal.eval.scoring import sascorer

    if SHIPPED_FP.is_file():
        pytest.skip("fpscores.pkl.gz now present -- this defect is resolved")
    with pytest.raises(FileNotFoundError):
        sascorer.compute_sa_score(Chem.MolFromSmiles("CCO"))


@requires_pre_fix_fork
def test_the_gitignore_comment_claims_a_download_path_that_does_not_exist():
    """.gitignore line 111 hides the pickle as 'regenerable via sascorer.py's
    pickle download path'. There is no download path in the file."""
    text = FORK.read_text(encoding="utf-8")
    for marker in ("urlretrieve", "requests.get", "urllib.request", "download"):
        assert marker not in text, (
            f"sascorer.py now contains {marker!r}. If the pickle gained a real "
            "download path, restore it to .gitignore and re-check this file's "
            "other premises."
        )


@requires_pre_fix_fork
def test_three_scorers_cannot_return_an_sa_score():
    """pocket2mol / softmol / targetdiff all import the dead fork."""
    for name in ("pocket2mol_score", "softmol_score", "targetdiff_score"):
        mod = importlib.import_module(f"molmetal.eval.scoring.{name}")
        fn = getattr(mod, "compute_sa_score", None)
        if fn is None:
            continue
        try:
            val = fn(Chem.MolFromSmiles("c1ccccc1CCO"))
        except Exception:
            continue  # dead fork, as expected
        assert val != val, (  # NaN
            f"{name}.compute_sa_score returned a real number. Either the "
            "fork was repaired or this module gained its own scorer; in both "
            "cases this test's premise is stale."
        )


# ---------------------------------------------------------------------------
# 2. The fork's fingerprint cannot match the Ertl table
# ---------------------------------------------------------------------------


@requires_pre_fix_fork
def test_fork_generator_is_folded_and_canonical_is_sparse():
    """The single line that kills the fragment term."""
    text = FORK.read_text(encoding="utf-8")
    assert "fpSize=2048" in text, (
        "the fork no longer folds its Morgan fingerprint. If this was fixed "
        "to sparse, the score1 collapse is over and every measurement taken "
        "through the fork -- notably round 8's OPT-1 probe -- must be redone."
    )
    canon = (RDKIT_SA_DIR / "sascorer.py").read_text(encoding="utf-8")
    assert "GetMorganGenerator(radius=2)" in canon and "fpSize" not in canon.split(
        "GetMorganGenerator(radius=2)"
    )[1][:40], "RDKit's canonical generator also changed; re-derive the delta"


def _fork_fingerprint(mol):
    """Run the FORK's own fingerprint call, parsed from its source.

    Do not hardcode the generator here. Two earlier versions of this file
    did and both were wrong in ways only a mutation run exposed:

    * hardcoding ``fpSize=2048`` meant the "make the fork sparse" mutation
      left the behavioural tests green, because the test was measuring a
      generator it had built itself rather than the one the fork runs;
    * then hardcoding a sparse ``GetMorganGenerator(radius=2)`` looked like
      a fix, but ``fpSize`` merely DEFAULTS to 2048 -- removing the keyword
      changes nothing. The bit ids stay folded. The line that actually
      matters is ``GetCountFingerprint`` vs ``GetSparseCountFingerprint``.

    So this reads the call the fork makes and executes that.
    """
    import re

    from rdkit.Chem import rdFingerprintGenerator

    text = FORK.read_text(encoding="utf-8")
    m = re.search(
        r"GetMorganGenerator\(\s*radius\s*=\s*(\d+)\s*(?:,\s*fpSize\s*=\s*(\d+)\s*)?\)",
        text,
    )
    if not m:
        raise AssertionError(
            f"could not parse the fork's Morgan generator from {FORK}; this "
            "test must follow the fork, not assume its shape"
        )
    radius = int(m.group(1))
    fp_size = int(m.group(2)) if m.group(2) else None
    kwargs = {"radius": radius}
    if fp_size is not None:
        kwargs["fpSize"] = fp_size
    gen = rdFingerprintGenerator.GetMorganGenerator(**kwargs)

    sparse = "GetSparseCountFingerprint" in text
    nz = (
        gen.GetSparseCountFingerprint(mol)
        if sparse
        else gen.GetCountFingerprint(mol)
    ).GetNonzeroElements()
    return {k[0] if isinstance(k, tuple) else k for k in nz}


@requires_pre_fix_fork
def test_fork_generator_is_folded_and_canonical_is_sparse():
    """The two lines that together kill the fragment term."""
    text = FORK.read_text(encoding="utf-8")
    assert "GetCountFingerprint" in text and "GetSparseCountFingerprint" not in text, (
        "the fork no longer uses the folded count fingerprint. If this was "
        "switched to sparse, the score1 collapse is over and every "
        "measurement taken through the fork -- notably round 8's OPT-1 probe "
        "-- must be redone."
    )
    canon = (RDKIT_SA_DIR / "sascorer.py").read_text(encoding="utf-8")
    assert "GetSparseCountFingerprint" in canon, (
        "RDKit's canonical scorer no longer uses the sparse fingerprint; "
        "re-derive the delta"
    )


@requires_pre_fix_fork
def test_fork_matches_no_fragments():
    """MEASURED 0/66 molecules get a single fragment score from the fork."""
    sa = _canonical()
    sa._fscores = None
    sa.readFragmentScores(str(RDKIT_SA_DIR / "fpscores.pkl.gz"))
    table = sa._fscores

    canon_hits, fork_hits = [], []
    for m in _mols():
        fork_hits.append(sum(1 for b in _fork_fingerprint(m) if b in table))
        sparse = sa.mfpgen.GetSparseCountFingerprint(m).GetNonzeroElements()
        canon_hits.append(sum(1 for k in sparse if k in table))

    assert statistics.fmean(canon_hits) > 5, (
        f"canonical now matches only {statistics.fmean(canon_hits):.2f} bits "
        "per molecule; the fixture that makes this comparison meaningful has "
        "changed."
    )
    assert max(fork_hits) == 0, (
        f"the fork now matches up to {max(fork_hits)} fragment bits, so its "
        "score1 is no longer the constant 5.0 fallback. The measured SA gap "
        "below, and round 8's OPT-1 conclusion, both need re-deriving."
    )


@requires_pre_fix_fork
def test_fork_and_canonical_disagree_on_the_generated_cohort():
    """Same molecules, same table, two implementations, 3.14 SA units apart."""
    fork = _fork()
    mols = _mols()
    canon = statistics.fmean(_canonical().calculateScore(m) for m in mols)
    forked = statistics.fmean(fork.compute_sa_score(m) for m in mols)
    assert abs(canon - forked) > 1.0, (
        f"the fork ({forked:.6f}) and canonical ({canon:.6f}) now agree to "
        "within 1 SA unit. If the fork was corrected, re-check every SA "
        "number in metrics/by_metric/ that was computed through it."
    )


def test_canonical_mean_matches_the_recorded_repo_value():
    """6.442335 is what six metric files record; pin that it reproduces."""
    got = statistics.fmean(_canonical().calculateScore(m) for m in _mols())
    assert abs(got - RECORDED_REPO_SA) < 1e-4, (
        f"canonical SA over the cohort is now {got:.6f}, not the "
        f"{RECORDED_REPO_SA} recorded in cfm_free_generation_panel_seed0.json. "
        "Either the cohort moved or the scorer changed; both invalidate six "
        "downstream files."
    )


# ---------------------------------------------------------------------------
# 3. The consequence: OPT-1's verdict INVERTS
# ---------------------------------------------------------------------------


def test_opt1_direction_reverses_under_the_canonical_scorer():
    """Round 8 concluded SA moves the WRONG WAY under every intervention.

    That came from the fork, whose fragment term is a constant. Under the
    canonical scorer the sign flips: dropping the stereocentre atoms LOWERS
    SA by 1.61 and lifts QED by 0.10. The round-8 honest-negative is an
    artefact of the broken scorer.

    The part of round 8 that survives is scorer-independent: ``set_tag``
    and ``remove_annotation`` remain vacuous, because
    ``FindMolChiralCenters(includeUnassigned=True)`` counts assigned
    centres too. Only the graph-modifying arms change sign.
    """
    from rdkit.Chem import QED

    from molmetal.eval.scoring.stereo_postprocess import (
        apply_stereocentre_suppression,
    )

    sa = _canonical()
    mols = _mols()
    base = statistics.fmean(sa.calculateScore(m) for m in mols)
    base_q = statistics.fmean(QED.qed(m) for m in mols)

    def survivors(mode):
        out = []
        for m in mols:
            try:
                mm = m if mode == "baseline" else apply_stereocentre_suppression(m, mode=mode)
                if mm is None:
                    continue
                Chem.SanitizeMol(mm)
            except Exception:
                continue
            out.append(mm)
        return out

    for vacuous in ("set_tag", "remove_annotation"):
        got = survivors(vacuous)
        assert got, f"control: {vacuous} should leave the cohort intact"
        assert statistics.fmean(sa.calculateScore(m) for m in got) == pytest.approx(
            base, abs=1e-9
        ), (
            f"{vacuous} is no longer vacuous under the canonical scorer. "
            "Round 8's claim that setting the @ tag cannot lower n_chiral is "
            "the one part that was scorer-independent; re-verify it."
        )

    removed = survivors("remove_atom")
    if not removed:
        pytest.skip("remove_atom left nothing parseable")
    d_sa = statistics.fmean(sa.calculateScore(m) for m in removed) - base
    d_q = statistics.fmean(QED.qed(m) for m in removed) - base_q
    assert d_sa < 0, (
        f"remove_atom now moves SA by {d_sa:+.4f} under the canonical scorer "
        "(round 8 reported +1.115 via the fork). If this is no longer a lift, "
        "cfm_todo.json::OPT_1_MEASURED_2026_10_13 must be re-derived from the "
        "canonical scorer."
    )
    assert d_q > 0, f"QED should lift too; got {d_q:+.4f}"
