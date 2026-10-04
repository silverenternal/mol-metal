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
    """The gate must read a BOND metric, and must be one that can improve.

    2026-10-05 — this test previously demanded the literal
    ``_bond_acc["bond_f1"]``, which was correct when written and wrong after
    the large-cohort run exposed saturation: MEASURED over 26,715 clouds,
    ``bond_f1`` reached 0.996 by epoch 5 and never moved again, while
    ``bond_exact_type_rate`` climbed 0.254 -> 0.415 over the same epochs.
    Gating on the saturated metric stopped the run at epoch 13 of 20
    (patience=10) and left the epoch-3 weights as "best" — the artifact with
    exact-match 0.254 instead of 0.415.

    So the invariant is not a particular metric's name. It is: the gate reads a
    bond metric, it is not decode_ratio, and it is not an existence metric
    that saturates. Pinning the name would have blocked the fix; pinning the
    property stops the same regression recurring.
    """
    src = TRAINER.read_text()
    assert "_bond_acc.get(" in src, (
        "the checkpoint gate no longer reads a bond metric at all — it would "
        "select on decode_ratio again, which is anti-correlated with bond "
        "correctness and saves the least-trained epoch as the best one"
    )
    assert "if _gate > _best_gate:" in src, (
        "the gate comparison was changed; confirm the criterion is still a "
        "bond-derived gate rather than decode_ratio"
    )
    # The saturating existence metric must not be the criterion again.
    gate_line = [
        ln
        for ln in src.splitlines()
        if "_gate = (" in ln or ln.strip().startswith("float(_bond_acc.get(")
    ]
    assert gate_line, "could not locate the gate assignment to audit it"
    assert not any('"bond_f1"' in ln for ln in gate_line), (
        "bond_f1 saturates on the large cohort (0.996 by epoch 5) and must not "
        "gate; use bond_exact_type_rate, which was still improving"
    )
    assert any('"bond_exact_type_rate"' in ln for ln in gate_line), (
        "the gate should be bond_exact_type_rate — the fraction of complexes "
        "whose bond set AND every order match"
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


def test_gate_name_and_value_are_defined_before_the_epoch_loop():
    """The run summary reads both AFTER the loop, so both must pre-exist.

    2026-10-05 — `_gate_name` and `_best_gate` were assigned only inside the
    epoch loop, and the summary block reads them once training is over.  A
    0-epoch run (exhausted resume budget, or an early break that skips the
    gate block) therefore raised `UnboundLocalError` at report time — after
    the training, which is the worst moment to lose a run.

    This is the same shape as the `_bond_acc` UnboundLocalError found
    earlier the same day, so the check is deliberately structural: it asserts
    the assignment precedes the loop rather than trying to run a 0-epoch
    training, which would cost a corpus load.
    """
    lines = TRAINER.read_text().splitlines()

    def _first_assign(name: str) -> int:
        for i, ln in enumerate(lines, 1):
            s = ln.strip()
            if s.startswith(f"{name} =") or s.startswith(f"{name}:"):
                return i
        raise AssertionError(f"{name} is never assigned in the trainer")

    loop = next(
        i for i, ln in enumerate(lines, 1) if "for epoch in range(" in ln
    )
    for name in ("_gate_name", "_best_gate", "_bond_acc"):
        at = _first_assign(name)
        assert at < loop, (
            f"{name} is first assigned at line {at}, after the epoch loop at "
            f"{loop}; a 0-epoch run would raise UnboundLocalError when the "
            "summary reads it after training"
        )


def test_hit_target_is_measured_against_the_gate_not_decode_ratio():
    """`hit_target: false` must not be possible for a run that improved.

    2026-10-05 — `hit_target` compared `best_decode_ratio` to
    `target_decode_ratio`.  decode_ratio FALLS as the bond head improves
    (loose atoms parse; connected molecules with a wrong valence do not), so
    the 2026-10-04 run improved its gate metric every single epoch and still
    reported `hit_target: false`.  The verdict now follows the gate.
    """
    src = TRAINER.read_text()
    assert '"gate_metric": _gate_name' in src, (
        "the summary no longer names the metric it gated on"
    )
    assert "hit_target=bool(" in src, "hit_target must be derived, not implied"
    # The old comparison must not survive anywhere in the summary/event path.
    assert "hit_target=bool(best_decode_ratio >=" not in src, (
        "hit_target is comparing decode_ratio again — that is the "
        "anti-correlated metric, and it makes an improving run read as a "
        "failure"
    )
    assert '"hit_target": best_decode_ratio >=' not in src, (
        "the JSON summary's hit_target is comparing decode_ratio again"
    )


def test_run_summary_reports_the_type_metrics():
    """The bond TYPE results must be in the summary, not only per-epoch.

    They are the numbers the task is actually about (existence F1 is close to
    saturated by classical toolkits; type perception is the claim), so a
    summary that omits them cannot support the result it is reporting.
    """
    src = TRAINER.read_text()
    for key in (
        '"bond_exact_type_rate"',
        '"bond_type_macro_f1_pooled"',
    ):
        assert key in src, f"{key} is missing from the trainer's run summary"
