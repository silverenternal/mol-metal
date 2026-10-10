# SPDX-License-Identifier: Apache-2.0
"""The `combined` early-stop gate must MOVE. Pin it against the recorded runs.

THE DEFECT THIS GUARDS
----------------------
`combined = gen_validity x bond_exact_type_rate`. `bond_exact_type_rate`
requires EVERY bond in a ~45-bond complex to match the Wiberg truth exactly.
MEASURED on both recorded 29-epoch runs:

    value set of bond_exact_type_rate over all epochs
        cfm_fm_control_2026_10_09 -> {0.0000}
        cfm_fm_ab_2026_10_09      -> {0.0000}
    early_stop_value              0.000000 -> 0.000000 (both arms)

So the criterion that selects which checkpoint survives was a constant. It
could not tell a trained model from an untrained one, and the run stopped on
it regardless. The code already DETECTED this — it writes "bond_exact_type_rate
is at its floor 0.000" into the note — and then gated on the constant anyway.

WHY REPLAY THE REAL CURVES IN A TEST
------------------------------------
Asserting the arithmetic on hand-picked numbers would pass while the real runs
kept failing. These tests read the CSVs the runs actually wrote and assert the
resolved criterion takes more than one distinct value across them. If someone
reverts the fallback, this fails on real data rather than on a contrived case.

WHAT IT DOES NOT CLAIM
----------------------
That `combined` is now a GOOD criterion. The swap was chosen by reading the
same two runs it is replayed on, so it is in-sample. This file guards against
regression to a constant; it is not evidence of improvement.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RECORDED_RUNS = {
    "control": "molmetal/reports/cfm_fm_control_2026_10_09/per_epoch_metrics.csv",
    "fm_ab": "molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.csv",
}


def _load_trainer():
    """Import the trainer module by path — it is a script, not a package API."""
    path = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location(
        "cfm_trainer_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_curve(rel: str):
    p = REPO_ROOT / rel
    if not p.is_file():
        pytest.skip(f"recorded run not present at {p}")
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    out = []
    for r in rows:
        gv_n = float(r["gen_validity_n"])
        out.append(
            {
                "epoch": int(r["epoch"]),
                "gen_validity": float(r["gen_validity_val"]) / gv_n if gv_n else 0.0,
                "gen_n": int(gv_n),
                "decode_ratio": float(r["decode_ratio"]),
                "bond_exact_type_rate": float(r["bond_exact_type_rate"]),
                "bond_n_scored": float(r["bond_n_scored"]),
                "bond_type_macro_f1": float(r["bond_type_macro_f1_pooled"]),
            }
        )
    return out


@pytest.mark.parametrize("run", sorted(RECORDED_RUNS))
def test_recorded_exact_type_rate_is_the_structural_zero(run):
    """The premise. If exact-type ever becomes non-zero this whole file is
    stale and the fallback should be revisited deliberately, not silently."""
    curve = _read_curve(RECORDED_RUNS[run])
    vals = {c["bond_exact_type_rate"] for c in curve}
    assert vals == {0.0}, (
        f"{run}: bond_exact_type_rate took values {sorted(vals)}, not a "
        "structural zero — the fallback's justification has changed and the "
        "swap decision must be re-made explicitly"
    )


@pytest.mark.parametrize("run", sorted(RECORDED_RUNS))
def test_combined_gate_is_not_constant_on_recorded_runs(run):
    """The regression guard: the gate must take more than one value."""
    resolve = _load_trainer().resolve_early_stop_criterion
    curve = _read_curve(RECORDED_RUNS[run])
    values = [
        resolve(
            "combined",
            decode_ratio=c["decode_ratio"],
            gen_validity=c["gen_validity"],
            gen_n=c["gen_n"],
            bond_exact_type_rate=c["bond_exact_type_rate"],
            bond_n_scored=c["bond_n_scored"],
            bond_type_macro_f1=c["bond_type_macro_f1"],
        )[1]
        for c in curve
    ]
    assert len(set(round(v, 6) for v in values)) > 1, (
        f"{run}: `combined` resolved to a constant {values[0]!r} across all "
        f"{len(values)} epochs. The gate cannot select a checkpoint if it "
        "never moves — this is the defect this file exists to prevent."
    )


def test_combined_uses_type_f1_when_exact_type_is_zero():
    resolve = _load_trainer().resolve_early_stop_criterion
    name, value, note = resolve(
        "combined",
        decode_ratio=0.8,
        gen_validity=0.875,
        gen_n=48,
        bond_exact_type_rate=0.0,
        bond_n_scored=192.0,
        bond_type_macro_f1=0.3634,
    )
    assert name == "combined"
    assert abs(value - 0.875 * 0.3634) < 1e-12, (
        "the fallback did not actually use bond_type_macro_f1"
    )
    assert "bond_type_macro_f1" in note and "structurally unreachable" in note, (
        "the swap must be recorded in the note — a reader must never have to "
        "guess which number gated"
    )


def test_without_type_f1_the_old_behaviour_is_preserved():
    """Callers that do not supply the new kwarg keep the old semantics.

    Without this, the fallback would fire for any caller that merely forgot
    the argument, turning a silent omission into a silent metric swap.
    """
    resolve = _load_trainer().resolve_early_stop_criterion
    name, value, _note = resolve(
        "combined",
        decode_ratio=1.0,
        gen_validity=1.0,
        gen_n=3,
        bond_exact_type_rate=0.0,
        bond_n_scored=5.0,
    )
    assert name == "combined" and value == 0.0


def test_bond_type_f1_is_selectable_on_its_own():
    resolve = _load_trainer().resolve_early_stop_criterion
    name, value, _ = resolve(
        "bond_type_f1",
        decode_ratio=0.8,
        gen_validity=0.9,
        gen_n=48,
        bond_exact_type_rate=0.0,
        bond_n_scored=192.0,
        bond_type_macro_f1=0.377,
    )
    assert name == "bond_type_f1" and abs(value - 0.377) < 1e-12


def test_bond_type_f1_is_an_accepted_criterion_name():
    mod = _load_trainer()
    assert "bond_type_f1" in mod.EARLY_STOP_METRICS

# ---------------------------------------------------------------------------
# epsilon-constraint criterion. The strong assertion is the last one: the
# wired criterion must REPRODUCE the knee that motivated it, on the real runs.
# ---------------------------------------------------------------------------

def _resolve_curve(resolve, curve, metric, epsilon=None):
    vals = []
    for c in curve:
        kw = dict(
            decode_ratio=c["decode_ratio"],
            gen_validity=c["gen_validity"],
            gen_n=c["gen_n"],
            bond_exact_type_rate=c["bond_exact_type_rate"],
            bond_n_scored=c["bond_n_scored"],
            bond_type_macro_f1=c["bond_type_macro_f1"],
        )
        if epsilon is not None:
            kw["early_stop_epsilon"] = epsilon
        vals.append(resolve(metric, **kw))
    return vals


def test_satisfying_epochs_outrank_violating_ones():
    """The negation is the whole mechanism, so pin the ordering directly.

    A satisfying epoch with a WORSE macro-F1 must still beat a violating epoch
    with a better one, or the constraint is not being enforced.
    """
    resolve = _load_trainer().resolve_early_stop_criterion
    _, good, _ = resolve(
        "bond_type_f1_eps",
        decode_ratio=0.8, gen_validity=0.90, gen_n=48,
        bond_exact_type_rate=0.0, bond_n_scored=192.0,
        bond_type_macro_f1=0.10, early_stop_epsilon=0.85,
    )
    _, bad, _ = resolve(
        "bond_type_f1_eps",
        decode_ratio=0.8, gen_validity=0.10, gen_n=48,
        bond_exact_type_rate=0.0, bond_n_scored=192.0,
        bond_type_macro_f1=0.37, early_stop_epsilon=0.85,
    )
    assert good > bad, (
        f"a violating epoch scored {bad} and outranked a satisfying one at "
        f"{good} — the epsilon constraint is not enforced"
    )
    assert bad < 0.0 <= good, "the two groups are not cleanly separated"


@pytest.mark.parametrize("run,expected", [("control", 17), ("fm_ab", 20)])
def test_epsilon_criterion_reproduces_the_knee(run, expected):
    """The criterion must select the knee the analysis found.

    blk_c13_epsilon_constraint.json located the knee at ep17 (control) and
    ep20 (fm_ab) on gen_validity x bond_type_macro_f1. If the wired criterion
    selects anything else, the implementation and the analysis that justified
    it have drifted apart — which is exactly the failure mode where a number
    gets quoted from an analysis that no longer describes the code.
    """
    resolve = _load_trainer().resolve_early_stop_criterion
    curve = _read_curve(RECORDED_RUNS[run])
    resolved = _resolve_curve(resolve, curve, "bond_type_f1_eps", epsilon=0.85)
    best_i = max(range(len(resolved)), key=lambda i: resolved[i][1])
    assert int(curve[best_i]["epoch"]) == expected, (
        f"{run}: epsilon-constraint selected epoch "
        f"{int(curve[best_i]['epoch'])}, but the knee analysis located "
        f"{expected}. Re-run wf_blk_c13_epsilon_constraint.py and reconcile "
        "before quoting either number."
    )


@pytest.mark.parametrize("run", sorted(RECORDED_RUNS))
def test_epsilon_criterion_is_not_constant(run):
    resolve = _load_trainer().resolve_early_stop_criterion
    curve = _read_curve(RECORDED_RUNS[run])
    values = [v for _, v, _ in _resolve_curve(resolve, curve, "bond_type_f1_eps", 0.85)]
    assert len(set(round(v, 6) for v in values)) > 1, (
        f"{run}: bond_type_f1_eps is constant across the run"
    )
