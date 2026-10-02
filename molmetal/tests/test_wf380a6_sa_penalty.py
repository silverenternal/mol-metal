"""Wf-380-A6 — SA penalty (TODO-205) test suite.

================================================================
Why this test exists
================================================================
TODO-205 / WEAK_TO_STRONG §2.4 — m4 Success Rate SA/QED sub-criterion
needs an SA-aware sampling penalty so the CFM can be nudged toward
the synthesizable region (SA < 2.8 per Wf-139 P6).  The penalty is
``max(0, sa - target_sa) * sa_weight`` (Ertl & Schuffenhauer 2009 §3
generation-time reweighting).

This module exercises:

* ``sa_aware_sampling_penalty`` — non-negativity + exact-zero below
  target (the core contract that the trainer will rely on).
* ``aggregate_sa_penalty`` — statistics are correct on a synthetic
  cohort with mixed above/below-target samples.
* ``sa_aware_loss_step`` — combined BCE+penalty loss is finite and
  monotonic in the penalty term.
* ``compute_sa_score`` — delegates to the RDKit bundled SA scorer
  correctly on a real SMILES.

All tests are CPU-only; no GPU launches; no RDKit trainer required.

Wf-380-A6 honest framing (party-twenty-spirit §3.5):
* SCAFFOLD tier — actual integration into the CFM trainer is deferred
  to Wf-380-B1.
* PROJECTED m4 SA lift: 3.099 -> 2.5-2.7 (WEAK_TO_STRONG §3.6).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the SA penalty module via direct path so we sidestep the
# ``molmetal`` package import (which pulls torch via
# ``molmetal/adapters/flow_matching_lipman/__init__.py`` and would
# fail in CPU-only test envs without torch installed).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PENALTY_PATH = (
    _REPO_ROOT
    / "molmetal"
    / "adapters"
    / "flow_matching_lipman"
    / "sa_penalty.py"
)


def _load_sa_penalty_module():
    """Load :mod:`sa_penalty.py` without going through ``molmetal.__init__``.

    Registers the module in ``sys.modules`` under its dotted path so
    the @dataclass decorator (and any other module-level helpers
    that introspect ``sys.modules[cls.__module__]``) work correctly.
    """
    spec = importlib.util.spec_from_file_location(
        "molmetal.adapters.flow_matching_lipman.sa_penalty",
        _PENALTY_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    # Pre-register so @dataclass can find __module__ in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sa_penalty_mod():
    """Return the loaded SA penalty module (module-scoped fixture)."""
    return _load_sa_penalty_module()


class TestSAPenaltyContract:
    """Verify the canonical penalty contract used by the trainer."""

    def test_penalty_is_nonnegative(self, sa_penalty_mod):
        """Penalty must be >= 0 for any input SA score."""
        mod = sa_penalty_mod
        for sa in [0.5, 1.0, 2.0, 2.5, 2.79, 2.8, 2.81, 3.0, 4.0, 5.0, 8.5, 10.0]:
            p = mod.sa_aware_sampling_penalty([sa])
            assert p.shape == (1,)
            assert p[0] >= 0.0, f"penalty negative for sa={sa}: {p[0]}"

    def test_penalty_zero_for_sa_le_target(self, sa_penalty_mod):
        """Penalty must be exactly 0 when SA <= target_sa (the synthesizable region)."""
        mod = sa_penalty_mod
        # At or below target: penalty == 0 by construction.
        for sa in [0.0, 0.5, 1.0, 2.0, 2.5, 2.79, 2.8]:
            p = mod.sa_aware_sampling_penalty(
                [sa], target_sa=mod.DEFAULT_TARGET_SA, sa_weight=0.3
            )
            assert p[0] == 0.0, (
                f"penalty nonzero at or below target (sa={sa}, "
                f"target={mod.DEFAULT_TARGET_SA}, p={p[0]})"
            )
        # Just above target: penalty > 0.
        p = mod.sa_aware_sampling_penalty(
            [2.81], target_sa=mod.DEFAULT_TARGET_SA, sa_weight=0.3
        )
        assert p[0] > 0.0, f"penalty zero just above target: {p[0]}"
        # And matches the linear formula: (2.81 - 2.8) * 0.3 = 0.003.
        assert abs(p[0] - 0.003) < 1e-9

    def test_penalty_linear_in_sa_above_target(self, sa_penalty_mod):
        """Above target the penalty is linear in (sa - target)."""
        mod = sa_penalty_mod
        # slope = sa_weight = 0.3, intercept = 0 at sa=target.
        for sa in [3.0, 5.0, 10.0]:
            p = mod.sa_aware_sampling_penalty(
                [sa], target_sa=2.8, sa_weight=0.3
            )
            expected = (sa - 2.8) * 0.3
            assert abs(p[0] - expected) < 1e-9, (
                f"expected {expected} got {p[0]} for sa={sa}"
            )

    def test_penalty_default_target_is_2_8(self, sa_penalty_mod):
        """Default target_sa must be 2.8 (the canonical synthesizable threshold)."""
        mod = sa_penalty_mod
        assert mod.DEFAULT_TARGET_SA == pytest.approx(2.8)

    def test_penalty_default_weight_is_0_3(self, sa_penalty_mod):
        """Default sa_weight must be 0.3 (matches the smoke cohort in sa_mean.json)."""
        mod = sa_penalty_mod
        assert mod.DEFAULT_SA_WEIGHT == pytest.approx(0.3)

    def test_penalty_vectorised(self, sa_penalty_mod):
        """Vectorised SA input must return a same-shape penalty vector."""
        mod = sa_penalty_mod
        sa_arr = np.array([1.0, 2.5, 2.8, 3.0, 4.5, 8.0])
        p = mod.sa_aware_sampling_penalty(sa_arr)
        assert p.shape == sa_arr.shape
        # Below-or-equal target = 0.
        assert p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0
        # Above target.
        assert p[3] == pytest.approx(0.06)
        assert p[4] == pytest.approx(0.51)
        assert p[5] == pytest.approx(1.56)


class TestAggregateSAPenalty:
    """Aggregate stats must be correct on synthetic cohorts."""

    def test_aggregate_mixed_cohort(self, sa_penalty_mod):
        """Mixed above/below-target cohort produces correct mean + fraction."""
        mod = sa_penalty_mod
        agg = mod.aggregate_sa_penalty([2.0, 3.0, 4.0, 5.0])
        assert agg.n == 4
        assert agg.mean_sa == pytest.approx(3.5)
        # 3.0, 4.0, 5.0 above target=2.8 -> fraction = 0.75
        assert agg.fraction_above_target == pytest.approx(0.75)
        # mean_penalty = mean(max(0, sa-2.8)*0.3) for [2.0, 3.0, 4.0, 5.0]
        # = mean([0, 0.06, 0.36, 0.66]) = 1.08 / 4 = 0.27
        assert agg.mean_penalty == pytest.approx(0.27)
        assert agg.target_sa == pytest.approx(2.8)

    def test_aggregate_empty_returns_nan(self, sa_penalty_mod):
        """Empty cohort returns n=0 with NaN stats (graceful)."""
        mod = sa_penalty_mod
        agg = mod.aggregate_sa_penalty([])
        assert agg.n == 0
        for field_name in ("mean_sa", "mean_penalty", "fraction_above_target"):
            v = getattr(agg, field_name)
            assert isinstance(v, float)
            assert v != v  # NaN check

    def test_aggregate_all_below_target_zero_penalty(self, sa_penalty_mod):
        """All SA <= target -> mean_penalty = 0 and fraction_above_target = 0."""
        mod = sa_penalty_mod
        agg = mod.aggregate_sa_penalty([1.0, 1.5, 2.0, 2.5, 2.8])
        assert agg.mean_penalty == pytest.approx(0.0)
        assert agg.fraction_above_target == pytest.approx(0.0)


class TestSAAwareLossStep:
    """The combined loss must be finite and monotonic in the SA penalty term."""

    def test_loss_step_1d_logits_finite(self, sa_penalty_mod):
        mod = sa_penalty_mod
        logits = np.array([0.0, 0.0, 0.0])
        sa_scores = np.array([2.0, 3.0, 5.0])
        loss = mod.sa_aware_loss_step(logits, sa_scores)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_loss_step_2d_logits_finite(self, sa_penalty_mod):
        mod = sa_penalty_mod
        logits = np.array([[0.0, 0.0], [0.0, 0.0]])
        sa_scores = np.array([2.0, 5.0])
        loss = mod.sa_aware_loss_step(logits, sa_scores)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_loss_step_monotonic_in_penalty(self, sa_penalty_mod):
        """Higher mean SA -> higher loss (penalty term is monotone non-decreasing)."""
        mod = sa_penalty_mod
        logits = np.array([0.0, 0.0])
        # Cohort A: low SA (no penalty)
        loss_a = mod.sa_aware_loss_step(logits, np.array([1.0, 2.0]))
        # Cohort B: high SA (penalty kicks in)
        loss_b = mod.sa_aware_loss_step(logits, np.array([6.0, 8.0]))
        assert loss_b > loss_a, (
            f"loss not monotonic in SA penalty: "
            f"loss_a={loss_a} loss_b={loss_b}"
        )


class TestComputeSAScoreDelegation:
    """compute_sa_score must delegate to the RDKit bundled SA scorer."""

    def test_compute_sa_score_ethanol_matches_rdkit_bundled(self, sa_penalty_mod):
        """CCO score must match RDKit's SA_Score.calculateScore directly."""
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import os as _os
        import sys as _sys

        sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in _sys.path:
            _sys.path.append(sa_dir)
        import sascorer as rdkit_sascorer  # noqa: WPS433

        ref = rdkit_sascorer.calculateScore(Chem.MolFromSmiles("CCO"))
        mod_score = sa_penalty_mod.compute_sa_score(Chem.MolFromSmiles("CCO"))
        assert abs(ref - mod_score) < 1e-9, (
            f"delegation mismatch: ref={ref} mod={mod_score}"
        )

    def test_compute_sa_score_none_returns_nan(self, sa_penalty_mod):
        """None mol must return NaN (defensive)."""
        mod = sa_penalty_mod
        score = mod.compute_sa_score(None)
        assert isinstance(score, float)
        assert score != score  # NaN