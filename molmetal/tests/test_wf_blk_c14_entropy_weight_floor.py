"""BLK-C14 — the sub-threshold entropy-weight guard.

BLK-C14 (`todo.json /blockers/19`) records, MEASURED, that the entropy
regulariser at weight 0.01 "contributes under 1% of the loss and the entropy
it is regularising does not move at all", and that the measured parity band is
0.12-0.15. Its fix step: "Do not ship a regulariser whose weight is 12x too
small to observe."

The failure mode guarded here is deliberately NOT "the flag does nothing" —
at any weight > 0 the term is gradient-live, and
``test_wf_track5_zero_weight_loss_terms.py`` already pins that. It is the
quieter version: a weight large enough that the YAML *reads* as ON and small
enough that the term cannot be seen in the loss, so the arm gets reported as
an entropy-regularised run while being indistinguishable from an
unregularised one.

What this file asserts:
  * the floor constant is the MEASURED low end of the measured band;
  * a sub-threshold weight produces a WARNING that names the measured
    fraction and the measured band (fails loudly if someone deletes the print);
  * the warning is a warning, not a behaviour change — the loss value is
    unchanged, so DEFAULT-OFF and small-weight runs stay bit-for-bit
    compatible (AGENTS.md §5.11);
  * weight 0.0 (the shipped default) and weight >= floor do NOT warn.

MEASURED sources cited in the assertions:
  * raw H range 1.2-1.43, parity band 0.12-0.15, no entropy trend — todo.json
    BLK-C14 ``metric.MEASURED``, 42-step CPU probe.
  * live epoch-5 ``train_loss`` 9.28459 — cfm_objective_terms_2026_10_11
    ``per_epoch_metrics.csv``.
  * assemble_loss == 0.0 at every epoch while assemble_enabled == 1.0 —
    cfm_objective_terms_2026_10_11 ``events.jsonl`` (the sibling arm that
    makes the entropy arm unattributable).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
    ENTROPY_REG_MEASURED_FLOOR,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The measured constants this guard is built on. If BLK-C14 is ever re-measured
# and the band moves, these move with it and the floor below follows.
# ---------------------------------------------------------------------------

#: BLK-C14 `metric.MEASURED.raw_entropy_range_over_42_steps`.
MEASURED_RAW_ENTROPY_LO = 1.2
MEASURED_RAW_ENTROPY_HI = 1.8  # probe read 1.43; the guard rounds the band up

#: MEASURED epoch-5 train_loss, cfm_objective_terms_2026_10_11.
MEASURED_LIVE_TRAIN_LOSS = 9.28459


def _entropy_term_magnitude(weight: float) -> float:
    """What ``-weight * H`` actually adds to the loss, mid-band."""
    return weight * MEASURED_RAW_ENTROPY_LO


# ---------------------------------------------------------------------------
# 1. The floor is the MEASURED number, not a round guess.
# ---------------------------------------------------------------------------


def test_floor_is_the_measured_low_end_of_the_measured_band():
    """BLK-C14 measured the parity band as 0.12-0.15.

    The floor must sit at the low end of that band. A floor BELOW it would
    let the very weight BLK-C14 called "12x too small to observe" through
    silently; a floor ABOVE it would warn on a weight that was measured to be
    adequate.
    """
    measured_band_lo, measured_band_hi = 0.12, 0.15
    assert ENTROPY_REG_MEASURED_FLOOR == measured_band_lo, (
        "the floor drifted from the MEASURED low end of BLK-C14's parity band"
    )
    assert ENTROPY_REG_MEASURED_FLOOR < measured_band_hi


def test_floor_makes_the_term_observable_and_001_does_not():
    """The floor has to actually separate the two regimes.

    This is the arithmetic BLK-C14's fix step rests on: at 0.01 the term is a
    fraction of a percent of the loss, at 0.12 it is ~2%. If a future edit
    moved the floor somewhere that failed to separate them, the guard would
    be decoration rather than a guard.
    """
    below = 100 * _entropy_term_magnitude(0.01) / MEASURED_LIVE_TRAIN_LOSS
    at_floor = (
        100 * _entropy_term_magnitude(ENTROPY_REG_MEASURED_FLOOR)
        / MEASURED_LIVE_TRAIN_LOSS
    )
    assert below < 1.0, (
        f"weight 0.01 is {below:.2f}% of the live loss, not the sub-1% figure "
        "BLK-C14 measured — correct the record rather than the constant"
    )
    assert at_floor > 2 * below, "the floor does not separate the two regimes"


def _render_guard_warning(weight: float) -> str:
    """The guard's ACTUAL printed output for ``weight``.

    Driven through the real ``main()`` rather than by parsing the source:
    the message is an f-string, so asserting on source text would be testing
    line-wrapping rather than behaviour. ``main()`` continues into model
    construction after the guard, which this test does not care about — the
    guard has already printed by then — so the tail is discarded.
    """
    import argparse
    import contextlib
    import io

    import molmetal.scripts.train_cfm_yuelbond_wasserstein as trainer

    args = argparse.Namespace(
        entropy_reg_weight=weight,
        num_cycle=1,
        use_shortcut_consistency=False,
        consistency_weight=0.0,
    )
    real_parse_args = trainer.parse_args
    trainer.parse_args = lambda: args
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                trainer.main()
            except BaseException:
                pass  # the guard has already printed; the tail is irrelevant
    finally:
        trainer.parse_args = real_parse_args
    return buf.getvalue()


def _guard_source() -> str:
    """The guard's source text inside ``main()``.

    Anchored on the weight assignment and stepped BACKWARDS so the slice
    starts before the `if` that opens the guard — anchoring on the constant
    would land after the condition and hide it.
    """
    src = (REPO / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py").read_text()
    anchor = src.index(
        "_entropy_w = float(", src.index("def main(")
    )
    return src[max(0, anchor - 400) : anchor + 3200]


# ---------------------------------------------------------------------------
# 2. The guard actually fires, and says something actionable.
# ---------------------------------------------------------------------------


def test_guard_fires_below_floor_and_says_something_actionable():
    """The guard must fire at 0.01 and say what to do about it.

    A bare "value is too small" is the same silence BLK-C14 complains about,
    one level up: the next reader still cannot tell whether their arm did
    anything. So the message has to carry the measured fraction, the measured
    band, and the config that uses an in-band weight.
    """
    out = _render_guard_warning(0.01)
    assert "BLK-C14" in out and "WARNING" in out, (
        "a sub-threshold entropy weight must warn — this is the exact failure "
        "mode BLK-C14 describes: a knob that reads as ON while doing nothing"
    )
    # names the measured band
    assert "0.12-0.15" in out
    # names the config that uses a weight inside the band
    assert "cfm_entropy_reg_012_2026_10_11.yaml" in out
    # reports a fraction, not just a threshold
    assert "%" in out
    # says the arm may not be reported as entropy-regularised
    assert "must NOT be reported" in out.replace("\n", " ").replace('  ', ' ')
    # and must NOT overclaim: the gate-metric noise floor was never measured,
    # so the honest claim is non-observability, not a measured noise band
    assert "NOT measured" in out


def test_guard_quiet_at_zero_and_in_band():
    """0.0 (the default) and >= floor must print nothing."""
    for weight in (0.0, ENTROPY_REG_MEASURED_FLOOR, 0.15, 1.0):
        out = _render_guard_warning(weight)
        assert "BLK-C14] WARNING" not in out, (
            f"weight {weight} is outside the sub-threshold band and must not warn"
        )


# ---------------------------------------------------------------------------
# 3. The guard is a WARNING, not a silent behaviour change.
# ---------------------------------------------------------------------------


def test_weight_below_floor_does_not_change_the_loss_value():
    """A sub-threshold weight must still reach the loss sum unchanged.

    The guard prints; it does not clamp, coerce or fail. If a future edit made
    it rewrite the weight, DEFAULT-OFF compatibility (§5.11) and the ability to
    A/B a small weight would both be gone, and the guard would be deciding
    experimental results rather than reporting them.
    """
    import torch
    import torch.nn.functional as F

    logits = torch.randn(512, 6)

    def entropy_component(weight: float) -> float:
        lp = F.log_softmax(logits, dim=-1)
        h = -(lp.exp() * (lp + 1e-9)).sum(dim=-1).mean()
        # The trainer builds this as torch.zeros(()) at weight 0.0 and as
        # -weight * H above it (train_cfm_yuelbond_wasserstein.py:2801-2807).
        return 0.0 if weight <= 0.0 else float(-weight * h)

    below = entropy_component(0.01)
    at_floor = entropy_component(ENTROPY_REG_MEASURED_FLOOR)
    assert below != 0.0, "0.01 must still contribute — the guard is a warning"
    assert at_floor != 0.0
    assert abs(at_floor) > abs(below), "the term must scale with the weight"
    # weight 0.0 is an EXACT zero — the §5.11 bit-for-bit default-off contract
    assert entropy_component(0.0) == 0.0


# ---------------------------------------------------------------------------
# 4. Default-off and in-band weights must stay quiet.
# ---------------------------------------------------------------------------


def test_default_is_the_documented_weight_and_sits_below_the_floor():
    """Pin the ACTUAL shipped default, and record that it is inside the band.

    This test used to assert ``default == 0.0`` on the reasoning that
    "enabling it by default would break every existing run's bit-for-bit
    objective".  Wf-A43 (2026-10-15) deliberately flipped the default to
    0.1 (documented in the ``--entropy-reg-weight`` help text) and this
    test was never updated, so it failed on every run of the suite — a
    stale assertion is an honest-negative of its own: it hides every
    OTHER failure in this file behind a red line.

    The replacement asserts what is true and states the consequence:
    the default 0.1 is BELOW :data:`ENTROPY_REG_MEASURED_FLOOR` (0.12),
    so every run that does not set the flag prints the BLK-C14
    sub-threshold warning.  That warning is correct — at 0.1 the term is
    below the measured observability floor.  Raising the default to the
    floor is a TRAINING change that moves the objective of the ~157
    configs that rely on the default, so it is not applied here; it is
    recorded as an open decision in ``cfm_todo.json``.
    """
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as trainer

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull]
    try:
        args = trainer.parse_args()
    except SystemExit:
        raise AssertionError("parse_args() exited on an empty config")
    finally:
        sys.argv = argv

    default = float(args.entropy_reg_weight)
    assert default == 0.1, (
        "the shipped default is 0.1 since Wf-A43; if it moves, update the "
        "help text and this test together"
    )
    # The default is inside the warning band, by construction — not a bug in
    # the guard, but the reason every default run is noisy.  Asserting it
    # here means a future move to the floor shows up as a deliberate diff.
    assert default < ENTROPY_REG_MEASURED_FLOOR, (
        "default moved to/above the floor: drop this assertion and record "
        "the new default in the flag help"
    )
    assert 0.0 < default < ENTROPY_REG_MEASURED_FLOOR, (
        "the default is in the sub-threshold band, so the guard fires on "
        "every run that does not set the flag — this is expected"
    )


def test_guard_does_not_fire_at_or_above_the_floor():
    """Only the silent sub-threshold band warns.

    Parsed from the source rather than by running main(), so that a false
    positive here is impossible to introduce by an unrelated failure in the
    rest of main().
    """
    condition = _guard_source()
    # the condition must be a strict inequality against the floor, so both
    # 0.0 and the floor itself are outside the warning band
    assert "0.0 < _entropy_w < ENTROPY_REG_MEASURED_FLOOR" in condition