"""The CFM checkpoint gate must select on bond F1, not decode_ratio.

2026-10-04 — MEASURED on tmQM Ru, one 6-epoch run:

    epoch          1      2      3      4      5
    bond_f1      0.065  0.124  0.235  0.462  0.439
    decode_ratio 0.969  0.883  0.617  0.188  0.188

The two move in OPPOSITE directions, and decode_ratio has the wrong sign: it
reads high while the head predicts BOND_NO_BOND for nearly every pair (loose
atoms parse trivially) and falls as real bonds appear (connected molecules
with a wrong valence get rejected by RDKit).  With `best_decode_ratio` as the
gate the trainer saved epoch 1 — the checkpoint with bond F1 0.065, the one
that had learned the least chemistry — and reported it as the best result.

decode_ratio is still recorded and still printed.  It just no longer decides
which weights survive.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = (
    PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)


def test_gate_uses_bond_f1_when_truth_is_available():
    src = TRAINER.read_text()
    assert '_bond_acc["bond_f1"]' in src, (
        "the checkpoint gate no longer consults bond_f1 — it would select on "
        "decode_ratio again, which is anti-correlated with bond correctness "
        "and saves the least-trained epoch as the best one"
    )
    assert "if _gate > _best_gate:" in src, (
        "the gate comparison was changed; confirm the criterion is still the "
        "bond-F1 gate rather than decode_ratio"
    )


def test_decode_ratio_is_still_recorded():
    """Demoted, not deleted — it is still a reported number."""
    src = TRAINER.read_text()
    assert "decode_ratio=float(decode_ratio)" in src, (
        "decode_ratio is no longer emitted; it must still be recorded for "
        "continuity with earlier runs even though it does not gate"
    )


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# The decision rule the F1 is computed under
# ---------------------------------------------------------------------------
def test_bond_decision_threshold_splits_existence_from_type():
    """`threshold` must decide EXISTENCE separately from the TYPE.

    2026-10-04 — MEASURED on the 64-cloud tmQM cohort, the two rules differ by
    0.45 F1 on the SAME logits (argmax 0.079 vs threshold 0.570 at 4 epochs
    unweighted), and class weighting swung argmax to the other failure mode
    (recall 0.97 / precision 0.32).  Both rules are now scored every epoch so
    the choice is measured rather than assumed.
    """
    import importlib.util
    import sys

    import torch

    spec = importlib.util.spec_from_file_location("_t_gate", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_gate"] = mod
    spec.loader.exec_module(mod)

    # One pair that is very likely bonded, one that is very likely not.
    # Columns: [NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC].
    logits = torch.tensor(
        [
            [-4.0, 4.0, 0.0, 0.0, 0.0],   # P(bonded) ~ 1.0, type SINGLE
            [4.0, -2.0, -2.0, -2.0, -2.0],  # P(bonded) low
        ]
    )
    got = mod._bond_decision(logits, "threshold", 0.5)
    assert got == [1, 0], f"expected [SINGLE, NO_BOND], got {got}"

    # The cut must decide the marginal pair — the knob argmax does not
    # expose.  NOTE the class count sets the scale: uniform logits over 5
    # classes put P(bonded) at 0.8 (= 1 - 1/5), NOT 0.5, so a "neutral"
    # threshold is not neutral here.
    logits_mid = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]])  # P(bonded) ~ 0.61
    p = 1.0 - float(torch.softmax(logits_mid.float(), -1)[0, 0])
    assert mod._bond_decision(logits_mid, "threshold", 0.5) == [1], (
        f"P(bonded)={p:.3f} >= 0.5 must predict a bond"
    )
    assert mod._bond_decision(logits_mid, "threshold", 0.9) == [0], (
        f"P(bonded)={p:.3f} < 0.9 must predict none — a cut above the "
        "model's confidence must be able to suppress a bond"
    )


def test_bond_decision_argmax_is_unchanged():
    """The historical rule must stay available and behave exactly as before."""
    import importlib.util
    import sys

    import torch

    spec = importlib.util.spec_from_file_location("_t_gate2", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_gate2"] = mod
    spec.loader.exec_module(mod)

    logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0, 0.0]])
    assert mod._bond_decision(logits, "argmax", 0.9) == [1, 0], (
        "argmax must ignore the threshold entirely"
    )


def test_both_rules_are_scored_every_epoch():
    """Reporting one rule would hide the gap that motivates the choice."""
    src = TRAINER.read_text()
    assert "_bond_decision(" in src and "bond_f1_alt_rule" in src, (
        "evaluate_bond_accuracy no longer scores the alternative decision "
        "rule — the argmax/threshold gap would go unmeasured and the config "
        "choice would be a guess again"
    )
