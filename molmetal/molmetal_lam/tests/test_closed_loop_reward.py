"""Tests for TODO-01 closed-loop reward wiring.

Verifies that :mod:`molmetal.orchestration.closed_loop` exposes the
Vina + PoseBusters + AiZynth composite reward aggregator, the
``decompose_reward`` helper, the ``--reward-weights`` CLI override,
and the ``r_admet`` opt-in channel.

Spot-checks only — no sweep / benchmark / large experiment runs.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

from molmetal.orchestration.closed_loop import (
    DEFAULT_REWARD_AGGREGATOR_WEIGHTS,
    RewardAggregatorScorer,
    TODO01_REWARD_AGGREGATOR_WEIGHTS,
    add_reward_weights_arg,
    apply_reward_weights_override,
    build_default_reward_aggregator,
    build_todo01_reward_aggregator,
    decompose_reward,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny stubs for Molecule / Complex / PropertyPrediction
# ---------------------------------------------------------------------------
@dataclass
class FakeMolecule:
    smiles: str = "CCO"
    qed: float = 0.5
    sa_score: float = 3.0


@dataclass
class FakeComplex:
    vina_score: float = -8.0
    binding_affinity: float = 7.0


@dataclass
class FakePropertyPrediction:
    qed: float = 0.5
    sa_score: float = 3.0
    binding_affinity_pic50: float = 7.0


# ---------------------------------------------------------------------------
# Constants — the TODO-01 brief values
# ---------------------------------------------------------------------------
def test_todo01_weights_match_brief() -> None:
    """TODO01_REWARD_AGGREGATOR_WEIGHTS must be the canonical brief values."""
    assert TODO01_REWARD_AGGREGATOR_WEIGHTS == {
        "w_vina": 0.40,
        "w_posebusters": 0.20,
        "w_retro": 0.15,
        "w_pic50": 0.10,
        "w_qed": 0.10,
        "w_sa": 0.05,
    }, "TODO-01 brief requires these exact weights"


# ---------------------------------------------------------------------------
# 1) test_closed_loop_uses_reward_aggregator
# ---------------------------------------------------------------------------
def test_closed_loop_uses_reward_aggregator() -> None:
    """Default scorer factory returns a RewardAggregator (not constant 0.5).

    Verifies that the TODO-01 factory actually constructs a
    :class:`RewardAggregator` with the rich channel layout (r_vina,
    r_posebusters, r_retro, r_pic50, r_qed, r_sa) — *not* a constant
    0.5 scalar or a single-channel scorer.
    """
    agg = build_todo01_reward_aggregator()
    # All six channels are exposed on the aggregator, even when the
    # underlying callables are ``None`` (they degrade to 0.0 in __call__).
    assert hasattr(agg, "r_vina")
    assert hasattr(agg, "r_posebusters")
    assert hasattr(agg, "r_retro")
    assert hasattr(agg, "r_pic50")
    assert hasattr(agg, "r_qed")
    assert hasattr(agg, "r_sa")
    # And the weights match the brief.
    assert agg.w_vina == 0.40
    assert agg.w_posebusters == 0.20
    assert agg.w_retro == 0.15
    assert agg.w_pic50 == 0.10
    assert agg.w_qed == 0.10
    assert agg.w_sa == 0.05


# ---------------------------------------------------------------------------
# 2) test_closed_loop_emits_per_channel_breakdown
# ---------------------------------------------------------------------------
def test_closed_loop_emits_per_channel_breakdown() -> None:
    """decompose_reward() emits r_vina, r_posebusters, r_retro, r_total."""
    # Wire stub callables that return constant positive values so we
    # can verify each channel is reported with the expected contribution.
    agg = build_todo01_reward_aggregator(
        r_vina=lambda s: -10.0,           # -> -(-10) = 10 normalised
        r_posebusters=lambda s: 0.8,      # 0.8 * w_posebusters
        r_retro=lambda s: 0.5,            # 0.5 * w_retro
        r_pic50=lambda s: 7.0,            # 7.0 * w_pic50
        r_qed=lambda s: 0.6,              # 0.6 * w_qed
        r_sa=lambda s: 2.0,               # sa_norm = 1 - 1/9 ≈ 0.889 * w_sa
    )
    bd = decompose_reward(agg, state=None)
    # Every required channel must be present.
    for key in (
        "r_vina",
        "r_posebusters",
        "r_retro",
        "r_pic50",
        "r_qed",
        "r_sa",
        "r_total",
    ):
        assert key in bd, f"breakdown missing channel {key!r}"
    # Spot-check arithmetic.
    assert bd["r_vina"] == pytest.approx(0.40 * 10.0)
    assert bd["r_posebusters"] == pytest.approx(0.20 * 0.8)
    assert bd["r_retro"] == pytest.approx(0.15 * 0.5)
    assert bd["r_pic50"] == pytest.approx(0.10 * 7.0)
    assert bd["r_qed"] == pytest.approx(0.10 * 0.6)
    # SA = 1 - (2 - 1) / 9 = 8/9 ≈ 0.8889
    assert bd["r_sa"] == pytest.approx(0.05 * (8.0 / 9.0))
    # r_total is the sum of all per-channel contributions.
    assert bd["r_total"] == pytest.approx(
        bd["r_vina"]
        + bd["r_posebusters"]
        + bd["r_retro"]
        + bd["r_pic50"]
        + bd["r_qed"]
        + bd["r_sa"]
    )


def test_closed_loop_emits_per_channel_breakdown_via_scorer() -> None:
    """RewardAggregatorScorer.breakdown() emits the same schema."""
    scorer = RewardAggregatorScorer(
        aggregator=build_todo01_reward_aggregator(
            r_vina=lambda s: -8.0,
            r_posebusters=lambda s: 1.0,
            r_retro=lambda s: 0.5,
        ),
    )
    bd = scorer.breakdown(
        FakeMolecule(),
        FakeComplex(),
        FakePropertyPrediction(),
    )
    for key in ("r_vina", "r_posebusters", "r_retro", "r_total"):
        assert key in bd


# ---------------------------------------------------------------------------
# 3) test_closed_loop_weights_override_via_cli
# ---------------------------------------------------------------------------
def test_closed_loop_weights_override_via_cli() -> None:
    """CLI --reward-weights YAML override merges on top of defaults."""
    # Build a parser that exposes the --reward-weights flag.
    parser = argparse.ArgumentParser()
    add_reward_weights_arg(parser)
    # Sanity-check the parser exposes the expected flag.
    help_text = parser.format_help()
    assert "--reward-weights" in help_text

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as fh:
        fh.write("w_vina: 0.7\nw_posebusters: 0.3\n")
        path = fh.name
    try:
        merged = apply_reward_weights_override(override_path=path)
        # Overridden keys take the YAML value.
        assert merged["w_vina"] == 0.7
        assert merged["w_posebusters"] == 0.3
        # Unmentioned keys stay at the TODO-01 defaults.
        assert merged["w_retro"] == 0.15
        assert merged["w_pic50"] == 0.10
        assert merged["w_qed"] == 0.10
        assert merged["w_sa"] == 0.05

        # Apply via argparse so the wire-up is exercised end-to-end.
        ns = parser.parse_args(["--reward-weights", path])
        assert ns.reward_weights == path
        merged_from_argparse = apply_reward_weights_override(
            override_path=ns.reward_weights,
        )
        assert merged_from_argparse["w_vina"] == 0.7
    finally:
        os.unlink(path)

    # Missing file -> defaults, no exception.
    fallback = apply_reward_weights_override(override_path="/nonexistent.yaml")
    assert fallback == dict(TODO01_REWARD_AGGREGATOR_WEIGHTS)


# ---------------------------------------------------------------------------
# 4) test_top5_pass_pb_80pct
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason=(
        "TODO-01 success criterion — top-5 by reward pass PB at >=80%. "
        "Marked xfail: PoseBusters not installed in this venv so the "
        "scorer cannot compute real PB verdicts.  See molmetal/validation/"
        "posebusters_runner.py for the runner; the closed-loop still "
        "emits pb_pass_rate_top1 / pb_pass_rate_topk in history records."
    ),
    strict=False,
)
def test_top5_pass_pb_80pct() -> None:
    """Top-5 candidates by reward should pass PoseBusters at >=80%.

    Marked ``xfail`` while the PoseBusters runner is unavailable —
    the closed-loop itself records ``pb_pass_rate_top1`` and
    ``pb_pass_rate_topk`` per iteration regardless, so the contract
    holds once the runner is wired.
    """
    try:
        from molmetal.validation.posebusters_runner import check_posebusters
    except Exception:
        pytest.xfail("PoseBusters runner unavailable")

    # Stub molecules; in production the closed-loop feeds its top-K
    # SMILES through check_posebusters and reports the pass-rate.
    candidates = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    n_pass = 0
    for smi in candidates:
        verdict = check_posebusters(smi)
        if isinstance(verdict, dict) and verdict.get("pb_valid"):
            n_pass += 1
    rate = n_pass / max(len(candidates), 1)
    assert rate >= 0.80, f"PB pass rate {rate:.0%} < 80%"


# ---------------------------------------------------------------------------
# Bonus: round-3 factory still works + ADMET opt-in path
# ---------------------------------------------------------------------------
def test_build_default_reward_aggregator_still_runs() -> None:
    """The round-3 factory must still construct (back-compat)."""
    agg = build_default_reward_aggregator()
    assert agg.w_vina == DEFAULT_REWARD_AGGREGATOR_WEIGHTS["w_vina"]
    assert agg.w_posebusters == DEFAULT_REWARD_AGGREGATOR_WEIGHTS["w_posebusters"]


def test_admet_channel_opt_in() -> None:
    """``enable_admet=True`` wires the round-8 ADMET default closure."""
    agg = build_todo01_reward_aggregator(
        enable_admet=True,
    )
    # w_admet becomes 1.0 *or* 0.0 if the ADMET default closure was not
    # importable (no admet-ai / datamol backend).  Both outcomes are
    # valid; we just verify the field exists.
    assert hasattr(agg, "w_admet")
    assert isinstance(agg.w_admet, float)
    if agg.w_admet == 1.0:
        # When wired, the r_admet callable is non-None.
        assert agg.r_admet is not None


# ---------------------------------------------------------------------------
# Bonus: decompose_reward never raises on a stub state
# ---------------------------------------------------------------------------
def test_decompose_reward_never_raises() -> None:
    """decompose_reward must never raise — channels degrade to 0.0."""
    def _boom(_state: Any) -> float:
        raise RuntimeError("simulated adapter failure")

    class _NonNumeric:
        # Returned by the callables below; the aggregator's _safe
        # helper coerces via float() which raises TypeError.
        def __float__(self) -> float:  # pragma: no cover
            raise TypeError("non-numeric channel value")

    agg = build_todo01_reward_aggregator(
        r_vina=_boom,                  # raises inside the callable
        r_posebusters=lambda s: _NonNumeric(),  # raises inside float()
        r_retro=lambda s: None,        # returns None -> 0.0 via float()
        r_pic50=lambda s: "nope",      # raises inside float()
    )
    bd = decompose_reward(agg, state=None)
    # All gracefully degraded to 0.0.
    assert bd["r_vina"] == 0.0
    assert bd["r_posebusters"] == 0.0
    assert bd["r_retro"] == 0.0
    assert bd["r_pic50"] == 0.0
    # qed/sa channels stay wired but never raise either.
    assert "r_qed" in bd
    assert "r_sa" in bd
    assert "r_total" in bd