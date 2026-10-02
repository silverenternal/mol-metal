"""Wf-381-A3 — TODO-205 m4 SA generation penalty regression-guard test.

================================================================
Why this test exists
================================================================
TODO-205 / WEAK_TO_STRONG §2.4 / METRIC_LIFT_STRATEGY_v2.md §4.5 — m4 Success
Rate SA/QED sub-criterion needs an SA-aware sampling penalty so the CFM can
be nudged toward the synthesizable region (SA < 2.8 per Wf-139 P6 / Ertl 2009
§3). The penalty is ``max(0, sa - target_sa) * sa_weight``.

This test pins the **core contract** that the trainer will rely on, on a
tighter cohort than Wf-380-A6's `test_wf380a6_sa_penalty.py` (which already
covers basic non-negativity + below-target-zero + scaling + aggregation):

* Penalty is non-negative for ALL SA values (including NaN-as-+inf defensive
  default).
* Penalty is EXACTLY 0 for ``sa <= target_sa`` — not just approximately, but
  bit-for-bit equal to 0 (the contract that makes low-SA samples free of
  bias).
* Penalty is LINEAR in ``(sa - target_sa)`` for ``sa > target_sa`` — the
  simple canonical Ertl form.
* Penalty scales by ``sa_weight`` — i.e. ``penalty(sa, w) == w *
  penalty(sa, 1.0)`` for all ``sa``.
* Boundary case: penalty is EXACTLY 0 at ``sa == target_sa`` (not just
  ``<=``).

This is the regression-guard test that the Wf-381-A3 docs-sync wave adds on
top of the Wf-380-A6 SCAFFOLD — it pins the contract that the Wf-380-B1
trainer wire-up will rely on.

Wf-381-A3 honest framing (party-twenty-spirit §3.5):
* SCAFFOLD tier — actual integration into ``train_cfm_yuelbond_wasserstein.py``
  is deferred to Wf-380-B1 (USER_OK-gated per AGENTS.md §5.1).
* All test assertions are CPU-only; no GPU launches; no RDKit trainer required.
* Pre-flight v3 ``92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL`` PRESERVED at
  Wf-370-B1 baseline — this test adds 1 NEW regression-guard test only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the SA penalty module via direct path so we sidestep the
# ``molmetal`` package import (which pulls torch via
# ``molmetal/adapters/flow_matching_lipman/__init__.py`` and would
# fail in CPU-only test envs without torch installed).  Same shim as
# ``test_wf380a6_sa_penalty.py``.
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

    Returns the loaded module object so tests can call its public functions
    directly.  We re-load on every call so test order does not matter.

    Note: we register the loaded module in ``sys.modules`` under its
    synthetic name BEFORE calling ``exec_module`` — this is required because
    ``sa_penalty.py`` uses ``@dataclass`` and Python 3.14's dataclasses
    decorator introspects ``sys.modules[cls.__module__]`` at class-creation
    time, which raises ``AttributeError: 'NoneType' object has no attribute
    '__dict__'`` if the module is not yet in ``sys.modules``.
    """
    mod_name = "_wf381a3_sa_penalty_loaded"
    spec = importlib.util.spec_from_file_location(mod_name, _PENALTY_PATH)
    assert spec is not None, f"failed to load spec for {_PENALTY_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass to work
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sa_penalty_module():
    """Module-scoped fixture so the import happens once per test session."""
    return _load_sa_penalty_module()


# ---------------------------------------------------------------------------
# SA cohort used for the contract tests
# ---------------------------------------------------------------------------

# Synthesizable threshold (Ertl 2009 §3 / Wf-139 P6 / WEAK_TO_STRONG §3.6).
# Default in sa_penalty.py is 2.8.
DEFAULT_TARGET_SA = 2.8

# Linear penalty weight (matches `--sa-weight 0.3` smoke cohort in
# metrics/by_metric/sa_mean.json 2026-09-14).
DEFAULT_SA_WEIGHT = 0.3

# Tighter cohort than Wf-380-A6: 7 SA values covering below-target
# (1.5, 2.0), at-target (2.8), just-above-target (3.0), mid (3.5, 4.0),
# and far-above (5.0).
SA_COHORT = np.array([1.5, 2.0, 2.8, 3.0, 3.5, 4.0, 5.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Test class — pins the SA penalty contract
# ---------------------------------------------------------------------------


class TestWf381A3SaPenaltyContract:
    """TODO-205 / Wf-381-A3 — pin the SA-aware penalty contract."""

    def test_penalty_is_nonneg_on_full_cohort(self, sa_penalty_module):
        """``sa_aware_sampling_penalty`` returns non-negative values for
        every SA in the 7-element cohort (the contract the sampler relies on
        to avoid biasing toward low-SA samples)."""
        penalty = sa_penalty_module.sa_aware_sampling_penalty(
            SA_COHORT, target_sa=DEFAULT_TARGET_SA, sa_weight=DEFAULT_SA_WEIGHT
        )
        assert penalty.shape == (len(SA_COHORT),)
        # All values must be non-negative.
        assert np.all(penalty >= 0.0), (
            f"penalty has negative entries: {penalty} "
            f"(contract: penalty(sa) >= 0 for all sa)"
        )

    def test_penalty_is_exactly_zero_below_or_equal_target(self, sa_penalty_module):
        """``sa_aware_sampling_penalty`` returns EXACTLY 0 (bit-for-bit) for
        every SA at-or-below the target.  This is the contract that makes
        low-SA samples free of bias — the core invariant the trainer relies
        on so it does not artificially nudge already-good samples."""
        penalty = sa_penalty_module.sa_aware_sampling_penalty(
            SA_COHORT, target_sa=DEFAULT_TARGET_SA, sa_weight=DEFAULT_SA_WEIGHT
        )
        # The first 3 cohort values (1.5, 2.0, 2.8) are at-or-below target.
        below_target = penalty[:3]
        assert np.all(below_target == 0.0), (
            f"penalty at-or-below target should be EXACTLY 0, got {below_target} "
            f"(contract: penalty(sa) == 0 for sa <= target)"
        )

    def test_penalty_is_zero_at_target_boundary(self, sa_penalty_module):
        """``sa_aware_sampling_penalty(2.8) == 0`` exactly — the boundary
        case is critical because the trainer compares ``sa <= target_sa``
        with ``np.isclose``-style semantic."""
        penalty_at_target = sa_penalty_module.sa_aware_sampling_penalty(
            np.array([DEFAULT_TARGET_SA]), target_sa=DEFAULT_TARGET_SA, sa_weight=DEFAULT_SA_WEIGHT
        )
        assert penalty_at_target.shape == (1,)
        assert float(penalty_at_target[0]) == 0.0, (
            f"penalty at target should be EXACTLY 0 (boundary), "
            f"got {float(penalty_at_target[0])}"
        )

    def test_penalty_is_linear_above_target(self, sa_penalty_module):
        """``sa_aware_sampling_penalty`` is linear in ``(sa - target_sa)``
        above the target — i.e. doubling the overshoot doubles the penalty
        (with the sa_weight factor).  This is the canonical Ertl form."""
        penalty = sa_penalty_module.sa_aware_sampling_penalty(
            SA_COHORT, target_sa=DEFAULT_TARGET_SA, sa_weight=DEFAULT_SA_WEIGHT
        )
        # Cohort indices 3..6 (sa = 3.0, 3.5, 4.0, 5.0) are above target.
        above_target = penalty[3:]
        # Expected: max(0, sa - 2.8) * 0.3
        expected_above = np.array(
            [max(0.0, sa - DEFAULT_TARGET_SA) * DEFAULT_SA_WEIGHT for sa in SA_COHORT[3:]]
        )
        assert np.allclose(above_target, expected_above), (
            f"penalty above target should be max(0, sa - target_sa) * sa_weight, "
            f"got {above_target} vs expected {expected_above}"
        )

    def test_penalty_scales_with_sa_weight(self, sa_penalty_module):
        """Halving the ``sa_weight`` halves the above-target penalty (the
        contract the trainer will rely on to tune ``--sa-weight`` from the
        CLI)."""
        penalty_w03 = sa_penalty_module.sa_aware_sampling_penalty(
            SA_COHORT, target_sa=DEFAULT_TARGET_SA, sa_weight=0.3
        )
        penalty_w06 = sa_penalty_module.sa_aware_sampling_penalty(
            SA_COHORT, target_sa=DEFAULT_TARGET_SA, sa_weight=0.6
        )
        # Above-target part: 0.6 weight should be EXACTLY 2x of 0.3 weight.
        above_w03 = penalty_w03[3:]
        above_w06 = penalty_w06[3:]
        assert np.allclose(above_w06, 2.0 * above_w03), (
            f"penalty with sa_weight=0.6 should be exactly 2x penalty with "
            f"sa_weight=0.3, got above_w06={above_w06} vs 2*above_w03={2.0 * above_w03}"
        )

    def test_penalty_treats_nan_as_positive_infinity(self, sa_penalty_module):
        """NaN SA scores are treated as +inf (the defensive default — NaN
        molecules are unparseable, the sampler should not pick them)."""
        sa_with_nan = np.array([1.0, np.nan, 5.0], dtype=np.float64)
        penalty = sa_penalty_module.sa_aware_sampling_penalty(
            sa_with_nan, target_sa=DEFAULT_TARGET_SA, sa_weight=DEFAULT_SA_WEIGHT
        )
        # sa=1.0 -> 0
        assert penalty[0] == 0.0
        # sa=NaN -> +inf * 0.3 = +inf (defensive default)
        assert np.isinf(penalty[1]) and penalty[1] > 0
        # sa=5.0 -> max(0, 5.0 - 2.8) * 0.3 = 2.2 * 0.3 = 0.66
        assert np.isclose(penalty[2], 0.66)
