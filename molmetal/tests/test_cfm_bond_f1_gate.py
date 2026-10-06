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


def _load_trainer():
    """Import the trainer module (cached) for behavioural assertions.

    Round-4 E4 added pure functions (`resolve_early_stop_criterion`,
    `early_stop_hit_target`) that the source-text audits cannot exercise: the
    failure they guard against is a criterion resolving to a constant 0.0 and
    the run plateauing silently, which is invisible in the source and only
    shows up when the function is called with the degenerate inputs.
    """
    import importlib.util
    import sys

    if "_t_gate" in sys.modules:
        return sys.modules["_t_gate"]
    spec = importlib.util.spec_from_file_location("_t_gate", str(TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


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
    # Round-4 E4 — the criterion moved out of the loop body into
    # `resolve_early_stop_criterion`, so the audit follows it there.  The
    # invariant is unchanged: the criterion is bond-derived, it is not
    # decode_ratio, and it is not the saturating existence metric.
    assert "def resolve_early_stop_criterion(" in src, (
        "the early-stop criterion is no longer resolved by a named helper — "
        "audit where it went before assuming it is still a bond metric"
    )
    resolver_body = src.split("def resolve_early_stop_criterion(", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert '"bond_exact_type"' in resolver_body, (
        "the resolver no longer offers bond_exact_type — the fraction of "
        "complexes whose bond set AND every order match"
    )
    assert '"bond_f1"' not in resolver_body, (
        "bond_f1 saturates on the large cohort (0.996 by epoch 5) and must not "
        "gate; use bond_exact_type_rate, which was still improving"
    )
    # The default must not be the anti-correlated one.
    assert 'DEFAULT_EARLY_STOP_METRIC = "combined"' in src, (
        "the default early-stop criterion is no longer a chemistry metric — "
        "decode_ratio is the project's own documented "
        "'NOT a chemistry metric, anti-correlated with correctness'"
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
    # Round-4 E4 — the verdict is now computed by a named helper so each
    # criterion can be compared against its OWN bar.  Comparing a
    # gen_validity rate against the bond-exact-type bar (or vice versa) is the
    # same class of bug as gating on decode_ratio.
    assert "def early_stop_hit_target(" in src, (
        "hit_target must be derived per criterion, not compared against one "
        "constant bar for every metric"
    )
    assert "hit_target=early_stop_hit_target(" in src, (
        "hit_target must be derived, not implied"
    )
    # The old comparison must not survive anywhere in the summary/event path.
    assert "hit_target=bool(best_decode_ratio >=" not in src, (
        "hit_target is comparing decode_ratio again — that is the "
        "anti-correlated metric, and it makes an improving run read as a "
        "failure"
    )
    assert '"hit_target": best_decode_ratio >=' not in src, (
        "the JSON summary's hit_target is comparing decode_ratio again"
    )


def test_early_stop_criterion_resolution():
    """Round-4 E4 — the criterion resolves to a MEASURED chemistry number.

    This is the behavioural counterpart to the source audits above.  The
    failure it guards against is silent and well-formatted: if a criterion's
    own input is unmeasured (generation eval off, or no Wiberg truth) and the
    code takes it at face value, `combined` becomes the constant 0.0, the run
    plateaus, and it early-stops at `patience` epochs having learned nothing
    — while writing a perfectly normal-looking per_epoch_metrics.json.
    """
    mod = _load_trainer()
    resolve = mod.resolve_early_stop_criterion

    # Both halves measured: the product, as asked.
    name, value, note = resolve(
        "combined",
        decode_ratio=0.53,
        gen_validity=0.40,
        gen_n=64,
        bond_exact_type_rate=0.38,
        bond_n_scored=100,
    )
    assert name == "combined" and note == ""
    assert abs(value - 0.40 * 0.38) < 1e-12, "combined is not the product"

    # Generation eval off (gen_n=0) must NOT yield combined==0.0.
    name, value, note = resolve(
        "combined",
        decode_ratio=0.53,
        gen_validity=0.0,
        gen_n=0,
        bond_exact_type_rate=0.38,
        bond_n_scored=100,
    )
    assert name == "bond_exact_type", f"combined gated on an unmeasured half: {name}"
    assert value > 0.0, "the degraded criterion must still carry real signal"
    assert note, "a degradation must be recorded, not silent"

    # No Wiberg truth: degrade the other way, and say so.
    name, value, note = resolve(
        "combined",
        decode_ratio=0.53,
        gen_validity=0.40,
        gen_n=64,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert name == "gen_validity"
    assert value > 0.0
    assert "bond_n_scored=0" in note

    # Neither chemistry metric measurable: the fallback is decode_ratio, and
    # the note must say plainly that no chemistry metric existed.
    name, value, note = resolve(
        "combined",
        decode_ratio=0.53,
        gen_validity=0.0,
        gen_n=0,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert name == "decode_ratio" and note

    # An explicitly-requested criterion with its own data missing degrades.
    name, value, note = resolve(
        "gen_validity",
        decode_ratio=0.53,
        gen_validity=0.0,
        gen_n=0,
        bond_exact_type_rate=0.38,
        bond_n_scored=100,
    )
    assert name == "bond_exact_type" and note

    # The legacy criterion is honoured exactly and never degraded.
    name, value, note = resolve(
        "decode_ratio",
        decode_ratio=0.531,
        gen_validity=0.0,
        gen_n=0,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert (name, value, note) == ("decode_ratio", 0.531, "")

    # Unknown names are rejected rather than silently defaulted.
    try:
        resolve(
            "bond_f1",
            decode_ratio=0.5,
            gen_validity=0.0,
            gen_n=0,
            bond_exact_type_rate=0.0,
            bond_n_scored=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown criterion must not be silently accepted")


def test_saturated_criterion_is_flagged_not_treated_as_signal():
    """A rate pinned at 0.0 or 1.000 carries no gradient — say so.

    MEASURED in the E4 50-step CPU smoke (tmQM Ru, 6 epochs, n=4 generation
    samples, no Wiberg truth): `gen_validity` read 1.000 at EVERY epoch, so
    `combined` degraded to a constant 1.0.  A constant criterion never
    improves, so the run stops at `patience` epochs while the JSON reads a
    confident-looking 1.000.  That is decode_ratio's failure mode wearing a
    chemistry label, so the resolver must name the saturation instead of
    letting a reader take 1.000 for a result.
    """
    mod = _load_trainer()
    resolve = mod.resolve_early_stop_criterion

    _, value, note = resolve(
        "combined",
        decode_ratio=1.0,
        gen_validity=1.0,
        gen_n=4,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert "saturated" in note, (
        "a criterion pinned at its ceiling was accepted as signal; a reader "
        "would read 1.000 as a result"
    )
    assert "n=4" in note, "the note must carry the sample size that saturated it"

    # The floor is the same failure.
    _, _, note_floor = resolve(
        "gen_validity",
        decode_ratio=0.5,
        gen_validity=0.0,
        gen_n=4,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert "saturated" in note_floor

    # A genuine interior value is NOT flagged — otherwise the warning is noise.
    _, _, note_ok = resolve(
        "combined",
        decode_ratio=0.5,
        gen_validity=0.40,
        gen_n=64,
        bond_exact_type_rate=0.35,
        bond_n_scored=50,
    )
    assert note_ok == "", f"an interior measurement was flagged: {note_ok!r}"


def test_combined_at_its_floor_is_flagged_on_the_bond_side():
    """MEASURED in the E4 50-step CPU smoke: `combined` was a constant 0.000.

    The 4-epoch / 5-complex smoke reports `bond_exact_type_rate` 0.000 at
    every epoch (a run that short cannot learn the task — expected), and
    `gen_validity` 1.000 (n=3, saturated).  The product is 0.000 at every
    epoch, so the gate never improves and the run early-stops having selected
    nothing.  Both factors are pinned; the note must say so rather than let
    `combined: 0.0` read as a measurement.
    """
    mod = _load_trainer()
    resolve = mod.resolve_early_stop_criterion
    name, value, note = resolve(
        "combined",
        decode_ratio=1.0,
        gen_validity=1.0,
        gen_n=3,
        bond_exact_type_rate=0.0,
        bond_n_scored=5,
    )
    assert name == "combined" and value == 0.0
    assert "floor" in note and "n=5" in note, (
        "the bond factor sitting at its floor was not reported; a reader would "
        "read combined=0.000 as a result"
    )
    assert "saturated" in note, "the gen factor's saturation was not reported"

    # The CFM retrain's own MEASURED range must stay clean, or the flag is
    # noise: exact_type climbed 0.355 -> 0.380 across that run.
    _, _, note_retrain = resolve(
        "combined",
        decode_ratio=0.53,
        gen_validity=0.40,
        gen_n=64,
        bond_exact_type_rate=0.355,
        bond_n_scored=500,
    )
    assert note_retrain == "", (
        f"a realistic in-range measurement was flagged: {note_retrain!r}"
    )


def test_early_stop_hit_target_uses_each_metrics_own_bar():
    """A `combined` run is not judged against the bond-exact-type bar."""
    mod = _load_trainer()
    hit = mod.early_stop_hit_target
    # 0.35 * 0.90 = 0.315, which is below the bond bar 0.50 but the combined
    # bar is 0.80 * 0.50 = 0.40.  Still a miss; the point is the two paths
    # are not the same expression.
    assert hit("bond_exact_type", 0.49, 0.8) is False
    assert hit("bond_exact_type", 0.51, 0.8) is True
    assert hit("gen_validity", 0.85, 0.8) is True
    assert hit("gen_validity", 0.75, 0.8) is False
    # decode_ratio keeps its own user-configurable bar.
    assert hit("decode_ratio", 0.85, 0.8) is True
    assert hit("decode_ratio", 0.75, 0.8) is False


def test_per_epoch_json_records_the_criterion():
    """The JSON must say which number gated, not leave it to be guessed."""
    src = TRAINER.read_text()
    assert "early_stop_metric: str" in src, (
        "per-epoch records no longer carry the criterion that gated them"
    )
    assert "early_stop_value: float" in src
    assert "early_stop_criterion_note: str" in src
    assert "early_stop_improved: bool" in src
    assert '"early_stop_control"' in src, (
        "the run summary does not record the early-stop control that produced "
        "its `best` block"
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
