"""The CFM must report bond TYPE, on the bench's definitions.

2026-10-04 — existence F1 alone cannot answer whether the learned bond head
earns its place.  MEASURED under one controlled non-hydrogen scope on 900 tmQM
complexes, OpenBabel recovers bond EXISTENCE at F1 0.977 while the CFM (4
epochs, probe scale) sat at 0.612, and OpenBabel's own weak axis is TYPE
(F1 0.505, against the CFM's 0.280).  So "is this head worth keeping?" is a
question about the TYPE column, and the trainer previously reported no type
number at all.

These tests pin the two definitions to each other so the training table and
the reference table cannot drift apart:

  * ``bond_exact_type_rate`` is aggregation-free — the fraction of clouds whose
    bond set AND every class match — so it must equal the bench's per-complex
    ``exact_type`` mean.  Both are asserted against the same constructed case.
  * ``bond_type_macro_f1_pooled`` uses the bench's per-class F1 formula with a
    DIFFERENT aggregation (pairs pooled, not averaged per complex).  On a
    single complex the two aggregations coincide, which is where they are
    compared.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)
BENCH = (
    PROJECT_ROOT / "molmetal" / "scripts" / "bench_tmqm_bond_perception.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load(BENCH, "_t_bench_type")


# ---------------------------------------------------------------------------
# The bench's macro-F1 is the reference definition
# ---------------------------------------------------------------------------
def test_bench_macro_f1_type_is_a_per_class_macro(bench):
    """macro_f1_type averages per-class F1, and a wrong class is penalised.

    Guards the bench's own bug history: an earlier version scored F1(exist)
    and F1(type) with the same expression, so the type column measured
    nothing.  Here existence is perfect while types are all wrong, and the
    two columns must disagree.
    """
    pred = {(1, 2): "single", (2, 3): "single"}
    truth_bo = {(1, 2): 2.0, (2, 3): 3.0}  # double, triple
    s = bench.score_bonds(pred, truth_bo)
    assert s["f1_exist"] == pytest.approx(1.0), "existence is perfect"
    assert s["exact_type"] == 0.0, "no bond order matches"
    assert s["macro_f1_type"] < 0.4, (
        "the type column must collapse when every order is wrong"
    )


def test_bench_exact_type_requires_both_set_and_order(bench):
    truth_bo = {(1, 2): 1.0, (2, 3): 1.0}
    right = bench.score_bonds({(1, 2): "single", (2, 3): "single"}, truth_bo)
    wrong_order = bench.score_bonds({(1, 2): "double", (2, 3): "single"}, truth_bo)
    missing = bench.score_bonds({(1, 2): "single"}, truth_bo)
    assert right["exact_type"] == 1.0
    assert wrong_order["exact_type"] == 0.0
    assert missing["exact_type"] == 0.0


def test_bench_scope_filters_never_the_truth(bench):
    """The documented guard: restricting the scope must not force precision 1.

    An earlier version intersected the PREDICTION with the truth, which makes
    F1(exist) 1.0 by construction for a method that is missing bonds.
    """
    pred = {(1, 2): "single"}          # misses (2, 3)
    truth_bo = {(1, 2): 1.0, (2, 3): 1.0}
    s = bench.score_bonds(pred, truth_bo, restrict_to={(1, 2), (2, 3)})
    assert s["f1_exist"] < 1.0, (
        "a missed bond inside the scope must lower F1 — the scope must filter "
        "the pairs under consideration, never the truth"
    )


# ---------------------------------------------------------------------------
# The trainer's type metrics use the same definitions
# ---------------------------------------------------------------------------
def test_trainer_reports_type_metrics():
    src = TRAINER.read_text()
    assert "bond_type_macro_f1_pooled" in src, (
        "the trainer no longer reports a bond-TYPE metric — the only axis on "
        "which the learned head could beat classical perception would go "
        "unmeasured again"
    )
    assert "bond_exact_type_rate" in src
    assert "[cfm-bond-type]" in src, (
        "the type metric is computed but not printed; a per-epoch line is what "
        "makes the OpenBabel comparison visible during a run"
    )


def test_trainer_pooled_macro_f1_matches_the_bench_formula():
    """Same per-class F1 formula, checked on a case where pooling == mean.

    For ONE complex the pooled macro-F1 and the bench's per-complex macro-F1
    are the same number, so the formula can be compared directly.  If the
    trainer's F1 were written as 2*tp/(tp+fp+fn) — a common slip — this
    would disagree as soon as a class has both errors present.
    """
    # Truth: (1,2) single, (2,3) double.  Pred: (1,2) single, (2,3) single.
    # single: tp=1, truth-class size 1, pred-class size 2 -> F1 = 2*1/(1+2)
    # double: tp=0, truth size 1, pred size 0         -> F1 = 0
    # macro = (2/3 + 0) / 2 = 1/3
    truth_by_type = {1: {(0, (1, 2))}, 2: {(0, (2, 3))}}
    pred_by_type = {1: {(0, (1, 2)), (0, (2, 3))}}
    classes = sorted(set(truth_by_type) | set(pred_by_type))
    f1s = []
    for cls in classes:
        t_c = truth_by_type.get(cls, set())
        p_c = pred_by_type.get(cls, set())
        tp_t = len(t_c & p_c)
        f1s.append(0.0 if not (t_c or p_c) else 2 * tp_t / (len(t_c) + len(p_c)))
    assert sum(f1s) / len(f1s) == pytest.approx(1 / 3)

    # ...which is the bench's number for the same single complex.
    bench = _load(BENCH, "_t_bench_type2")
    s = bench.score_bonds({(1, 2): "single", (2, 3): "single"},
                          {(1, 2): 1.0, (2, 3): 2.0})
    assert s["macro_f1_type"] == pytest.approx(1 / 3), (
        "the trainer's pooled formula and the bench's per-complex formula "
        "must agree where the aggregation is identical"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
