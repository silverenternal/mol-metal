# SPDX-License-Identifier: Apache-2.0
"""m7 diversity must be a reproducible MOSES IntDiv1, not an ad-hoc number.

WHY THIS FILE EXISTS
--------------------
m7 used to exist only as a number in a throwaway script, so nothing could
check it. Worse, it was read as a STRENGTH: 0.90-0.93 against a Guan 2023
target of 0.860 looks like a clear win. It is not.

Our formula was already MOSES IntDiv1 (`1 - mean Tanimoto similarity`). The
benchmark values are:

    MOSES training set (ZINC Clean Leads, 1.9M)   IntDiv1 0.857
    CharRNN / VAE / AAE / LatentGAN                 0.851 - 0.874
    n-gram / HMM combinatorial baselines            0.847 - 0.874

So the CFM cohort scores ABOVE the data it was trained on. The field's reading
of that (Vendi Score, arXiv:2210.02410) is that values above ~0.88 on MOSES are
rare and usually indicate distribution shift rather than genuine novelty — their
HMM model posts one of the highest IntDiv1 scores (0.871) while ranking LOWEST
on Vendi Score. IntDiv also barely discriminates between models at all; it is a
mode-collapse alarm, not a quality score.

The same source notes GA/search methods collapsing to 0.13-0.15 on JNK3/sEH,
which is what the Lambda MCTS path's 0.1366 looks like — and on THAT path the
low number is the diagnostic one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = REPO_ROOT / "molmetal" / "scripts"

MOSES_TRAIN_SET = 0.857
ABOVE_REFERENCE = 0.88


@pytest.fixture(scope="module")
def panel():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import cfm_property_panel as mod

    return mod


def test_internal_diversity_is_reachable_from_the_panel(panel):
    """m7 must not live only in a throwaway script."""
    rep = panel.build_panel(["CCO", "c1ccccc1", "CC(=O)O"])
    assert "internal_diversity" in rep["aggregate"]


def test_matches_the_moses_configuration(panel):
    out = panel.internal_diversity(["CCO", "c1ccccc1", "CC(=O)O", "CCN"])
    assert out["radius"] == 2, "MOSES ECFP4 is radius 2"
    assert out["n_bits"] == 1024, (
        f"MOSES uses 1024 bits, got {out['n_bits']} — the project was using "
        "2048, which is not the reference configuration"
    )
    assert out["self_pairs"] is True, (
        "the MOSES formula sums over |G|^2 ordered pairs INCLUDING self-pairs; "
        "excluding them shifts IntDiv1 by about 1/n"
    )


def test_identical_molecules_score_zero_not_undefined(panel):
    """All-same cohort -> 0.0, the collapse signature.

    The cohort dedupes to ONE molecule. That is not "too few to measure" — it
    is total mode collapse, and IntDiv is a collapse alarm, so reporting None
    would hide the very signal the metric exists for. IntDiv over |G|=1 is
    1 - T(m,m) = 0 by the same formula.
    """
    out = panel.internal_diversity(["c1ccccc1"] * 8)
    assert out["intdiv1"] == pytest.approx(0.0, abs=1e-9)
    assert "collapse" in out["note"].lower()


def test_disjoint_molecules_score_near_the_achievable_ceiling(panel):
    """The ceiling is 1 - 1/n, not 1.0, because self-pairs are in the sum.

    With the MOSES formula summing over |G|^2 ordered pairs INCLUDING the
    diagonal, a perfectly disjoint set of n still contributes n self-pairs at
    T=1. For n=5 that caps IntDiv1 at 0.8. Asserting "> 0.8" would have been a
    guess that fails for a correct implementation.
    """
    smis = ["c1ccccc1", "C1CCCCC1", "CCOC(=O)C", "N#Cc1ccncc1", "O=S(=O)(O)O"]
    out = panel.internal_diversity(smis)
    ceiling = 1.0 - 1.0 / len(smis)
    assert out["intdiv1"] == pytest.approx(ceiling, abs=0.05)


def test_reports_the_reference_and_flags_being_above_it(panel):
    out = panel.internal_diversity(["CCO", "c1ccccc1"])
    assert out["moses_train_set_reference"] == MOSES_TRAIN_SET
    assert "moses_model_range" in out
    # A high value must be visibly suspicious, not silently flattering.
    assert "above_reference_flag" in out
    assert "do not present a high value" in out["note"].lower()


def test_sample_size_is_reported_not_hidden(panel):
    smis = [f"CC{'C' * i}" for i in range(1, 40)]
    out = panel.internal_diversity(smis, max_mols=20)
    assert out["n_used"] == 20
    assert out["n_unique"] >= 20
    assert out["n_used"] < out["n_unique"], (
        "the subsample must be visible so the number is not read as covering "
        "the whole cohort"
    )


def test_unparseable_cohort_returns_none(panel):
    """Zero parseable molecules genuinely is undefined — not collapse."""
    out = panel.internal_diversity(["not a smiles", "also not"])
    assert out["intdiv1"] is None
    assert "no molecule parsed" in out["note"]
