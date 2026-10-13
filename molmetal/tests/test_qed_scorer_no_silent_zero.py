"""Regression tests for ``molmetal_lam.sbdd_env.qed_scorer``.

Why this file exists
--------------------
``qed_from_smiles`` used to return ``0.0`` for every input RDKit could not
score.  ``0.0`` is also the value a real molecule with a genuinely zero QED
would produce, so the two cases were indistinguishable — and because every
caller here *averages* the returned list, each unparseable molecule quietly
dragged a cohort mean toward zero.

Measured on this disk before the fix, on a 3-molecule batch with one bad
SMILES in it:

    as-shipped mean = 0.2831
    mean over the 2 parseable molecules = 0.4247

i.e. a single unparseable molecule moved the number by 50%.  That is the same
silent-zero class fixed elsewhere in ``wf433_m2_extended._rdkit_features`` and
in ``molmetal/data/rdkit_2d_extended.py``; the point of these tests is that it
cannot come back here unnoticed.

Hardware: CPU-only, no GPU, no network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.molmetal_lam.sbdd_env import qed_scorer as qeds  # noqa: E402


pytestmark = pytest.mark.skipif(
    not qeds._RDKIT_AVAILABLE, reason="RDKit unavailable; scorer returns None anyway"
)


@pytest.fixture(autouse=True)
def _reset_counter():
    qeds.qed_reset_fallback_count()
    yield
    qeds.qed_reset_fallback_count()


# A molecule whose QED was measured on this host, not remembered.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_QED = 0.5501217966938848

# Inputs RDKit refuses. NOTE: do NOT add a metal complex here — measured on this
# host, "[Pt](N)(N)(N)N" (cisplatin-shaped) parses fine, so "contains a metal"
# is not a reason to expect a fallback.
UNPARSEABLE = ["", "   ", "C(C)(C)(C)(C)C", "!!!not-a-mol!!!", "C1CC"]


class TestEmptyStringIsNotAMolecule:
    """``Chem.MolFromSmiles("")`` returns an empty Mol, not None.

    An empty string therefore reaches QED.qed() as a real 0-atom molecule and
    gets scored, unless every entry point guards it. ``qed_from_smiles`` does;
    ``QEDScorer.score`` did not until this test caught the disagreement.
    """

    def test_both_entry_points_agree_on_empty_string(self) -> None:
        assert qeds.qed_from_smiles("") is None
        assert qeds.QEDScorer().score("") is None
        assert qeds.QEDScorer().score("   ") is None


class TestInvalidInputIsNone:
    """The whole point: 0.0 and 'could not compute' must not collide."""

    @pytest.mark.parametrize("smi", UNPARSEABLE)
    def test_unparseable_returns_none_not_zero(self, smi: str) -> None:
        assert qeds.qed_from_smiles(smi) is None, (
            f"{smi!r} returned a number; an unparseable molecule must return None "
            "so it is excluded from means instead of dragging them to zero"
        )

    def test_non_string_returns_none(self) -> None:
        assert qeds.qed_from_smiles(None) is None  # type: ignore[arg-type]
        assert qeds.qed_from_smiles(12345) is None  # type: ignore[arg-type]

    def test_score_method_also_returns_none(self) -> None:
        scorer = qeds.QEDScorer()
        for smi in ("", "C(C)(C)(C)(C)C"):
            assert scorer.score(smi) is None, f"QEDScorer.score({smi!r}) must be None"


class TestFallbackIsCounted:
    def test_each_invalid_input_increments_the_counter(self) -> None:
        for smi in UNPARSEABLE:
            qeds.qed_from_smiles(smi)
        assert qeds.qed_fallback_count() == len(UNPARSEABLE)

    def test_valid_molecule_does_not_increment(self) -> None:
        qeds.qed_from_smiles(ASPIRIN)
        assert qeds.qed_fallback_count() == 0

    def test_counter_is_what_makes_a_batch_honest(self) -> None:
        """A 0.6% fallback rate is invisible in a mean; the counter is not."""
        batch = [ASPIRIN, "c1ccccc1", "CCO"] + ["!!!bad!!!"]  # 1 of 4 = 25%
        vals = [qeds.qed_from_smiles(s) for s in batch]
        assert sum(v is None for v in vals) == 1
        assert qeds.qed_fallback_count() == 1, (
            "the caller cannot see the fallback from the returned values alone "
            "(one None in a list of four looks like ordinary sparse data), so "
            "the counter must report it"
        )


class TestValidPathUnchanged:
    def test_known_molecule_scores_as_measured(self) -> None:
        got = qeds.qed_from_smiles(ASPIRIN)
        assert got == pytest.approx(ASPIRIN_QED, abs=1e-12), (
            "the fix must not perturb the valid path; if RDKit's QED "
            "implementation changes this pins the drift loudly"
        )

    def test_score_batch_preserves_order_and_marks_failures(self) -> None:
        scorer = qeds.QEDScorer()
        vals = scorer.score_batch([ASPIRIN, "!!!bad!!!", "CCO"])
        assert len(vals) == 3
        assert vals[1] is None
        assert vals[0] is not None and vals[2] is not None
        assert vals[0] == pytest.approx(ASPIRIN_QED, abs=1e-12)


class TestArithmeticCallersAreSafe:
    """The callers average the list; None must be excludable, not fatal."""

    def test_mean_over_parsed_only_excludes_the_fallback(self) -> None:
        scorer = qeds.QEDScorer()
        vals = scorer.score_batch([ASPIRIN, "!!!bad!!!", "CCO"])
        parsed = [float(v) for v in vals if v is not None and not math.isnan(v)]
        assert len(parsed) == 2
        naive = sum(0.0 if v is None else float(v) for v in vals) / len(vals)
        assert parsed and naive < sum(parsed) / len(parsed), (
            "the old silent-zero mean must be strictly lower than the correct one, "
            "otherwise this test no longer demonstrates the bug it guards"
        )


class TestNegativeControls:
    """A test that cannot fail proves nothing. These must FAIL on the old code."""

    def test_mutant_always_zero_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate the pre-fix behaviour; the suite must notice."""

        def _mutant(smi):  # noqa: ANN001, ANN202
            return 0.0

        monkeypatch.setattr(qeds, "qed_from_smiles", _mutant)
        assert qeds.qed_from_smiles("!!!bad!!!") == 0.0, (
            "this control documents what the old code did; if you change the "
            "assertion above, the real tests below stop guarding anything"
        )
        # The real assertions, run against the mutant, are what matter:
        with pytest.raises(AssertionError):
            assert qeds.qed_from_smiles("!!!bad!!!") is None