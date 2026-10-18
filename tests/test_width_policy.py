"""Tests for ``molmetal.width_policy``.

The policy is the code form of the #184 capacity result: widening pays only on
cohorts large enough to support it. Three things must hold:

  I1. ``fixed`` never changes the caller's hidden, whatever n is -- published
      columns must stay reproducible from their configs.
  I2. ``size-adaptive`` switches exactly at the measured cut and nowhere else.
  I3. The decision dict always reports the width that will actually be used, so
      a run artifact can never be ambiguous about it.

CPU-only, no torch, no data.

Run: pytest tests/test_width_policy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.width_policy import (  # noqa: E402
    MIN_N_FOR_WIDE,
    NARROW_HIDDEN,
    WIDE_HIDDEN,
    select_hidden,
)


@pytest.mark.parametrize("n", [0, 1, 250, 334, 999, 1000, 1281, 4793])
def test_fixed_ignores_n(n: int) -> None:
    for base in (NARROW_HIDDEN, WIDE_HIDDEN, 64):
        hidden, decision = select_hidden(n, policy="fixed", fixed_hidden=base)
        assert hidden == base
        assert decision["policy"] == "fixed"
        assert decision["hidden"] == base
        assert decision["n_unique"] == n


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, NARROW_HIDDEN),
        (250, NARROW_HIDDEN),   # Re
        (333, NARROW_HIDDEN),   # Os
        (334, NARROW_HIDDEN),   # Rh
        (999, NARROW_HIDDEN),
        (1000, WIDE_HIDDEN),    # the cut itself is inclusive
        (1281, WIDE_HIDDEN),    # Ir
        (4793, WIDE_HIDDEN),    # Ru
    ],
)
def test_size_adaptive_cut(n: int, expected: int) -> None:
    hidden, decision = select_hidden(n, policy="size-adaptive")
    assert hidden == expected
    assert decision["hidden"] == expected
    assert decision["policy"] == "size-adaptive"
    assert decision["n_unique"] == n
    assert str(MIN_N_FOR_WIDE) in decision["rule"]


def test_size_adaptive_ignores_fixed_hidden() -> None:
    # fixed_hidden is only meaningful for the fixed policy; passing it must not
    # leak into the adaptive branch and silently override the measurement.
    hidden, _ = select_hidden(4793, policy="size-adaptive", fixed_hidden=64)
    assert hidden == WIDE_HIDDEN
    hidden, _ = select_hidden(250, policy="size-adaptive", fixed_hidden=1024)
    assert hidden == NARROW_HIDDEN


def test_decision_carries_evidence() -> None:
    _, decision = select_hidden(4793, policy="size-adaptive")
    assert decision["basis"].endswith("dmpnn_within_metal_capacity_pair_gpu0.json")
    assert set(decision["validated_on"]) == {"Ru", "Ir", "Rh", "Os", "Re"}
    assert "Post-hoc" in decision["caveat"]


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        select_hidden(100, policy="auto")


def test_trainer_uses_resolved_hidden_per_metal(monkeypatch) -> None:
    """The policy is only useful if ``_run_metal`` actually trains at the width
    it resolved, and records it. Stub out loading/featurisation/training so this
    is a wiring test, not an experiment."""
    import numpy as np

    import molmetal.scripts.dmpnn_within_metal_auc as T

    n = 1200  # >= MIN_N_FOR_WIDE
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    smiles = ["C"] * n

    monkeypatch.setattr(T, "_load_dedup_cohort", lambda metal, spec: (smiles, y, n))
    featurize = lambda s: [object()] * len(s)  # noqa: E731  (injected, not imported)
    monkeypatch.setattr(
        T,
        "_same_folds_as_probe",
        lambda yy, seed, n_splits: [
            (np.arange(n // 2), np.arange(n // 2, n)),
            (np.arange(n // 2, n), np.arange(n // 2)),
        ],
    )
    monkeypatch.setattr(T, "_xgb_fold_aucs", lambda *a, **k: [0.5, 0.5])

    seen_hidden = []

    def fake_train(graphs, y, tr, te, *, hidden, **kwargs):
        seen_hidden.append(hidden)
        return {"auc": 0.6, "acc": 0.5}

    monkeypatch.setattr(T, "_train_one_fold", fake_train)

    parser = T._build_parser()
    args = parser.parse_args(
        ["--metals", "Ru", "--split-seeds", "42", "--n-splits", "2",
         "--train-seeds", "0", "--width-policy", "size-adaptive", "--device", "cpu"]
    )
    args.max_fold_n = 0
    entry = T._run_metal("Ru", args, "cpu", featurize)

    assert entry["status"] == "MEASURED"
    assert entry["hidden_used"] == WIDE_HIDDEN
    assert entry["width_decision"]["policy"] == "size-adaptive"
    assert seen_hidden and set(seen_hidden) == {WIDE_HIDDEN}

    # And the fixed policy must still honour --hidden exactly.
    seen_hidden.clear()
    args.width_policy = "fixed"
    args.hidden = 64
    entry = T._run_metal("Ru", args, "cpu", featurize)
    assert entry["hidden_used"] == 64
    assert set(seen_hidden) == {64}
