"""Guards for the tmQM bond-perception scoring function.

2026-10-04 — `score_bonds` shipped two bugs on first write, both of which made
the reference table flattering rather than wrong-looking, which is the
dangerous kind:

1. **Scope filter applied to the truth.**  ``pred_keys = set(pred) &
   set(truth)`` forced precision to 1.0 by construction, so a method that
   missed bonds still scored F1(exist) = 1.000.  ``restrict_to`` names the
   SCOPE a baseline can address (non-metal pairs, for a metal-stripped
   method); it must never narrow the prediction to the ground truth.

2. **F1(exist) and F1(type) computed by the same expression.**  The predicted
   class set had already been filtered by class before both loops, so the two
   columns were equal by construction and the "type" column measured nothing.

Both are caught below.
"""

from __future__ import annotations

import pytest

from molmetal.scripts.bench_tmqm_bond_perception import bo_to_type, score_bonds


def test_missing_bonds_are_not_forgiven_by_scope():
    """A scope filter must not hide false negatives.

    Ground truth has two bonds inside scope; the prediction finds only one.
    Recall must suffer — this is bug (1) above.
    """
    truth = {(1, 2): 1.0, (3, 4): 1.0}  # both in scope
    pred = {(1, 2): "single"}  # (3, 4) missed entirely
    s = score_bonds(pred, truth, restrict_to={(1, 2), (3, 4)})
    assert s["f1_exist"] < 1.0, (
        "F1(exist)=1.0 while a ground-truth bond was missed — the scope filter "
        "is narrowing the prediction to the truth again"
    )
    assert s["f1_exist"] == pytest.approx(2 * 1 / (1 + 2), abs=1e-9)
    assert s["exact"] == 0.0
    assert s["exact_type"] == 0.0


def test_in_scope_false_positive_is_penalised():
    """A wrong bond INSIDE the scope must cost precision.

    This is the case that actually exposes bug (1).  The other scope tests use
    a prediction that is a subset of the truth, so intersecting the prediction
    with the truth is a no-op for them and they pass with or without the bug.
    Here the prediction contains a pair that is in scope but NOT a real bond;
    if the scoring narrows the prediction to the truth, that false positive
    vanishes and F1 comes back as 1.0.
    """
    truth = {(1, 2): 1.0}
    pred = {(1, 2): "single", (3, 4): "single"}  # (3,4) in scope, but no bond
    s = score_bonds(pred, truth, restrict_to={(1, 2), (3, 4)})
    # TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=2/3
    assert s["f1_exist"] == pytest.approx(2 / 3, abs=1e-9), (
        f"in-scope false positive was forgiven (F1={s['f1_exist']}) — the "
        "prediction is being intersected with the truth again"
    )
    assert s["exact"] == 0.0


def test_scope_does_not_penalise_out_of_scope_false_positives():
    """Bonds the baseline is not asked about must not count against it.

    (5, 6) is outside the scope, so predicting or missing it is irrelevant.
    """
    truth = {(1, 2): 1.0, (5, 6): 1.0}
    pred = {(1, 2): "single", (5, 6): "single"}  # perfectly right, but 5-6 out of scope
    s = score_bonds(pred, truth, restrict_to={(1, 2)})
    assert s["f1_exist"] == pytest.approx(1.0)
    assert s["exact_type"] == pytest.approx(1.0)
    assert s["n_true"] == 1 and s["n_pred"] == 1


def test_existence_and_type_f1_are_distinct():
    """Right connectivity + wrong orders must separate the two columns.

    This is bug (2): if both columns come from one expression they stay equal
    and the type column is dead.
    """
    # Two single bonds in truth; the prediction finds both pairs but calls
    # both of them double.
    truth = {(1, 2): 1.0, (2, 3): 1.0}
    pred = {(1, 2): "double", (2, 3): "double"}
    s = score_bonds(pred, truth)
    assert s["f1_exist"] == pytest.approx(1.0), "connectivity is perfect"
    assert s["macro_f1_type"] == pytest.approx(0.0), (
        "every bond order is wrong, so macro-F1 over TYPE must be 0 — it "
        "equalling F1(exist) means the two columns are one expression"
    )
    assert s["exact"] == 1.0, "the bond SET matches"
    assert s["exact_type"] == 0.0, "but the orders do not"


def test_exact_type_requires_both():
    truth = {(1, 2): 1.0}
    assert score_bonds({(1, 2): "single"}, truth)["exact_type"] == pytest.approx(1.0)
    assert score_bonds({(1, 2): "double"}, truth)["exact_type"] == pytest.approx(0.0)
    assert score_bonds({}, truth)["exact_type"] == pytest.approx(0.0)


def test_bo_to_type_boundaries():
    """Wiberg BO -> nearest of single/double/triple."""
    assert bo_to_type(0.9) == "single"
    assert bo_to_type(1.0) == "single"
    assert bo_to_type(1.4) == "single"
    assert bo_to_type(1.6) == "double"
    assert bo_to_type(2.0) == "double"
    assert bo_to_type(2.6) == "triple"
    assert bo_to_type(3.0) == "triple"


def test_offsets_do_not_displace_precision_and_recall():
    """A split of errors between FP and FN must be reflected, not rounded away."""
    truth = {(1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0}
    pred = {(1, 2): "single", (2, 3): "single", (9, 9): "single"}
    s = score_bonds(pred, truth)
    # TP=2, FP=1, FN=1  ->  P = 2/3, R = 2/3, F1 = 2/3
    assert s["f1_exist"] == pytest.approx(2 / 3, abs=1e-9)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
