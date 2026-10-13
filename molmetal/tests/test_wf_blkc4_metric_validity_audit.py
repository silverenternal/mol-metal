# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""BLK-C-L4 — the aromatic-valency audit's load-bearing claims, pinned.

The audit's verdict is "our connectivity advantage does NOT collapse", which is
a negative result resting on a positive claim about RDKit: ``GetTotalValence()``
already charges aromatic bonds their Kekule order, so the shipped check never
had the arXiv:2505.00169 bug in the first place.  If that ever changes, the
verdict silently inverts, so it is pinned here rather than left to a JSON.

These tests are RED without the behaviours they name; each is annotated with
what fails if the corresponding line is reverted.
"""

import json
import os
import subprocess
import sys

import pytest

# this file is <repo>/molmetal/tests/<it>.py — three levels up is the repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from molmetal.scripts.wf_blkc4_metric_validity_audit import (  # noqa: E402
    _independent_descriptors,
    connectivity_ok,
    score_row,
    valence_ok,
)


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    return m


@pytest.mark.parametrize(
    "smiles,symbol,expected",
    [
        ("c1ccccc1", "C", 4),      # aromatic C: 1.5x2 would be 3.0, Kekule is 4
        ("c1cc[nH]c1", "N", 3),    # pyrrole N
        ("n1ccccc1", "N", 3),      # pyridine N
        ("c1ccoc1", "O", 2),       # furan O
    ],
)
def test_rdkit_total_valence_is_already_aromatic_aware(smiles, symbol, expected):
    """RED if RDKit regresses to charging aromatic bonds 1.5 (or 1.0).

    This is the audit's premise.  Asserted per-element so a partial regression
    (say, heteroatoms only) is caught rather than masked by a benzene-only test.
    """
    got = [a.GetTotalValence() for a in _mol(smiles).GetAtoms()
           if a.GetSymbol() == symbol]
    assert got and all(v == expected for v in got), f"{smiles} {symbol}: {got}"


def test_kekulize_does_not_change_get_total_valence():
    """RED if the ``kekule`` mode stops being equivalent to RDKit's own answer.

    ``valence_ok(..., "kekule")`` exists to be the *corrected* evaluator.  If
    Kekulize ever shifted valences, the corrected check would disagree with
    RDKit for reasons unrelated to chemistry and every number below would be
    measuring that, not the molecules.
    """
    for smiles in ("c1ccccc1", "c1cc[nH]c1", "c1ccoc1", "Cc1ccccc1"):
        m = _mol(smiles)
        before = [a.GetTotalValence() for a in m.GetAtoms()]
        m2 = Chem.Mol(m)
        Chem.Kekulize(m2, clearAromaticFlags=True)
        assert before == [a.GetTotalValence() for a in m2.GetAtoms()], smiles


def test_buggy_evaluator_differs_from_corrected():
    """RED if the ``buggy`` mode is removed or aliased to ``kekule``.

    Without a genuinely distinct buggy mode the side-by-side has nothing to
    compare, and the report would silently degrade to a single column.
    """
    benzenoid = _mol("c1ccc(cc1)c1ccccc1")  # biphenyl, all aromatic carbons
    assert valence_ok(benzenoid, "buggy")["ok"] is False   # sums to 3, not 4
    assert valence_ok(benzenoid, "kekule")["ok"] is True   # kekulized to 4


def test_naive_evaluator_over_penalises_furan_oxygen():
    """RED if ``naive`` stops being the over-penalising middle option.

    Charging aromatic bonds 1.5 looks like the fix but fails furan O (3.0 vs an
    allowed 2) — which is why the verdict rests on ``kekule``.  If a future edit
    makes ``naive`` correct, this test says so loudly instead of letting the
    docstring's "WRONG on its own" become a lie.
    """
    assert valence_ok(_mol("c1ccoc1"), "naive")["ok"] is False
    assert valence_ok(_mol("c1ccoc1"), "kekule")["ok"] is True


def test_connectivity_predicate_is_fragment_count_not_valency():
    """RED if ``connectivity_ok`` grows a valency term.

    The headline metric is a PURE connectivity test, which is exactly why it is
    blind to chemistry.  Folding valence in would make the two cohorts'
    connectivity numbers quietly valency numbers and the side-by-side
    meaningless.
    """
    assert connectivity_ok(_mol("C.C.C")) is False        # 3 fragments
    assert connectivity_ok(_mol("CCC")) is True          # 1 fragment
    # A single fragment that is chemically absurd still passes: that IS the
    # blind spot, and pinning it stops anyone "fixing" it into the headline.
    # Hexavalent carbon cannot be sanitized, so it is parsed with
    # sanitize=False — which is exactly the point: the connectivity predicate
    # never sanitizes, so it never asks.
    hyper = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
    assert hyper is not None
    assert connectivity_ok(hyper) is True
    assert valence_ok(hyper, "kekule")["ok"] is False


def test_sa_score_is_not_silently_none():
    """RED if the SA import ever starts failing again.

    The first full run of this audit reported ``sa_erlt: None`` for all 702
    molecules while looking entirely healthy, because running the script as
    ``molmetal/scripts/<it>.py`` puts ``molmetal/scripts`` — not the repo root
    — on ``sys.path``, so ``import molmetal.adapters...`` raised and the bare
    ``except`` turned the failure into a None that a mean over an empty list
    reports as 0.0, i.e. as a chemistry result.  A scorer that cannot load must
    be loud.

    Asserted in BOTH modes on purpose.  In-process (this file's own import)
    the repo root is already on ``sys.path`` courtesy of conftest, so only a
    subprocess reproduces the real failure; asserting in-process alone gives a
    green test over a broken script.
    """
    out = score_row("c1ccccc1")
    assert out is not None
    assert out.get("sa_erlt_error") is None, out.get("sa_erlt_error")
    assert isinstance(out.get("sa_erlt"), float), f"SA not scored: {out}"

    # The mode that actually broke: invoked by path, exactly as the audit is.
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "molmetal", "scripts",
                                      "wf_blkc4_metric_validity_audit.py"),
         "--help"],
        capture_output=True, text=True, cwd="/tmp", timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_sa_actually_scores_when_script_is_run_by_path(tmp_path):
    """RED if the script-mode path yields ``sa_erlt: None`` on a real cohort.

    The subprocess check above only proves the script IMPORTS; this runs it end
    to end on a 1-molecule CSV and reads the SA back out of the JSON, which is
    the assertion that actually failed the first time.
    """
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("epoch,seed,smiles\n0,0,c1ccccc1\n")
    out_json = tmp_path / "audit.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "molmetal", "scripts",
                                      "wf_blkc4_metric_validity_audit.py"),
         "--generated", str(csv_path), "--real", str(csv_path),
         "--out", str(out_json)],
        capture_output=True, text=True, cwd="/tmp", timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(out_json.read_text())
    row = payload["cohorts"]["generated"]["per_row"][0]
    assert row.get("sa_erlt_error") is None, row.get("sa_erlt_error")
    assert isinstance(row.get("sa_erlt"), float), f"SA not scored: {row}"
    assert payload["cohorts"]["generated"]["other"]["sa_erlt_mean"] is not None


def test_independent_scorer_absence_is_declared_not_silent():
    """RED if the "no independent scorer" fact is ever dropped from the report.

    The first version of this block returned ``{"mordred_available": True}``
    with no numbers — present, healthy-looking, and empty, the worst possible
    shape for a cross-check.  Whether mordred works is environment-dependent, so
    the test pins the SHAPE of the declaration rather than its truth: the two
    ``*_available`` keys must always be present and explicitly False or True,
    never absent and never a bare empty dict.
    """
    got = _independent_descriptors(["c1ccccc1", "CCO"])
    assert got.get("independent_qed_scorer_available") is not None
    assert got.get("independent_sa_scorer_available") is not None
    assert isinstance(got.get("independent_sa_scorer_available"), bool)
    assert got.get("reason"), "the reason must travel with the flags"
    # If either scorer ever becomes available, a number must come with it.
    if got.get("independent_sa_scorer_available"):
        assert any(k.endswith("_sa_mean") for k in got), got
    # Whatever IS independently measurable must actually produce a number.
    assert isinstance(got.get("mordred_molwt_mean"), float), got
    assert got.get("mordred_molwt_n_scored") == 2, got


def test_generated_cohort_is_essentially_aromatic_free():
    """RED if the audit's asymmetry premise stops holding.

    Only 1 of 66 generated molecules contains an aromatic atom while 481 of
    636 real ones do.  The aromatic-evaluator question therefore barely bites
    the generated cohort at all — the audit must keep reporting the asymmetry
    rather than averaging over it.
    """
    import csv

    gen = "/home/hugo/codes/mol-metal/molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv"
    if not os.path.exists(gen):
        pytest.skip("generated cohort CSV not on disk")
    with open(gen, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("smiles") or "").strip()]
    n_arom = sum(
        1 for r in rows
        if (m := Chem.MolFromSmiles(r["smiles"])) is not None
        and any(a.GetIsAromatic() for a in m.GetAtoms())
    )
    assert n_arom <= 2, f"generated cohort aromatic load moved: {n_arom}/{len(rows)}"
