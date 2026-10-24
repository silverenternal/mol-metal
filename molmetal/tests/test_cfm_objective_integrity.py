"""objective_integrity — the artifact says whether the arm is the arm.

Wf-A108h measured the failure this guards: at batch 64 under a 20 GiB allocator
cap, FmOdeTerm raised CUDA OOM on 25 of 25 steps and returned zeros, SBOO3Term
on 4 of 25, and the loss curve looked like a smaller objective rather than a
missing one.  `regularizer_fallbacks` recorded the counts per epoch; this block
states the conclusion once, at the top of the artifact.
"""
from molmetal.scripts import train_cfm_yuelbond_wasserstein as trainer


def test_clean_run_is_not_degraded(monkeypatch):
    monkeypatch.setattr(trainer, "_regularizer_fallbacks", lambda: {})
    monkeypatch.setattr(trainer, "_regularizer_fallback_reasons", lambda: {})
    got = trainer._objective_integrity()
    assert got["degraded"] is False
    assert got["lost_terms"] == []


def test_any_fallback_marks_the_run_degraded(monkeypatch):
    monkeypatch.setattr(
        trainer, "_regularizer_fallbacks", lambda: {"FmOdeTerm": 25, "SBOO3Term": 4}
    )
    monkeypatch.setattr(
        trainer,
        "_regularizer_fallback_reasons",
        lambda: {"FmOdeTerm": {"OutOfMemoryError": 25}, "SBOO3Term": {"OutOfMemoryError": 4}},
    )
    got = trainer._objective_integrity()
    assert got["degraded"] is True
    assert got["lost_terms"] == ["FmOdeTerm", "SBOO3Term"]
    assert got["fallback_counts"]["FmOdeTerm"] == 25
    assert got["fallback_reasons"]["SBOO3Term"]["OutOfMemoryError"] == 4
