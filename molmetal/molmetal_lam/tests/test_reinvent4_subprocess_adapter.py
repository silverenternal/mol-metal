"""Unit tests for the L-4 REINVENT4 subprocess-RPC adapter (Lambda round 3).

Three test cases cover the public surface:

* ``test_reinvent4_adapter_unavailable_returns_none``
    — when the REINVENT4 binary is missing, every ``.score()`` call
    must return ``None`` (graceful fallback).
* ``test_score_aggregator_weighted_sum``
    — default weights ``0.3 + 0.3 + 0.3 + 0.1`` sum to ``1.0`` for a
    perfect molecule, confirming the aggregator's numerical correctness.
* ``test_reward_aggregator_with_reinvent4_channel``
    — :class:`RewardAggregator` constructs cleanly when ``r_reinvent4``
    is provided and degrades to ``0.0`` when the channel returns ``None``
    or the adapter is unavailable.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure the project root is on sys.path so the test runs in isolation.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from molmetal_lam.sbdd_env.reinvent4_subprocess_adapter import (
    REINVENT4Adapter,
    REINVENT4MultipropertyAdapter,
    ScoreAggregator,
    ScoreResult,
    is_reinvent4_binary_available,
)


class TestREINVENT4AdapterUnavailable(unittest.TestCase):
    """When the reinvent4 binary is missing, all score() calls return None."""

    def test_unavailable_returns_none(self) -> None:
        # Construct the adapter against a non-existent binary.  The
        # adapter's runtime contract is: when no reinvent4 binary is
        # available (or the worker subprocess fails to spawn), every
        # ``score()`` call must return ``None`` so the search loop
        # degrades gracefully.
        #
        # The static probe in :func:`is_reinvent4_binary_available` has
        # a fallback path that detects a cloned reference repo on disk
        # (which we DO have here).  We bypass that by patching the
        # probe so it always reports "False" — that is what we are
        # really testing.
        import unittest.mock as mock

        with mock.patch(
            "molmetal_lam.sbdd_env.reinvent4_subprocess_adapter.is_reinvent4_binary_available",
            return_value=False,
        ):
            adapter = REINVENT4Adapter(
                binary="reinvent",
                heartbeat_interval=60.0,
            )
            # The probe returned False → the adapter is unavailable.
            self.assertFalse(adapter.available)
            self.assertFalse(adapter.dead)
            out = adapter.score(["CCO", "c1ccccc1", "CC(=O)O"])
            self.assertEqual(len(out), 3)
            for item in out:
                self.assertIsNone(item)
            # Empty batch → empty list (no spurious None entries).
            self.assertEqual(adapter.score([]), [])
            # Cleanup: no worker was started, but be defensive.
            adapter.close()


class TestJSONLinesBridge(unittest.TestCase):
    """The isolated bridge validates the wire contract without REINVENT4."""

    def test_rdkit_bridge_scores_and_handles_invalid_smiles(self) -> None:
        worker = Path(__file__).parents[2] / "molmetal_lam" / "sbdd_env" / "reinvent4_jsonl_worker.py"
        adapter = REINVENT4Adapter(binary=str(worker), heartbeat_interval=0.05)
        try:
            self.assertTrue(adapter.available)
            values = adapter.score(["CCO", "invalid"])
            self.assertEqual(len(values), 2)
            self.assertTrue(values[0] is not None and values[0].is_complete())
            self.assertIsNotNone(values[1])
            self.assertFalse(values[1].is_complete())
        finally:
            adapter.close()


class TestScoreAggregator(unittest.TestCase):
    """The ScoreAggregator combines four [0,1] components into a single float."""

    def test_weighted_sum(self) -> None:
        agg = ScoreAggregator()  # defaults: 0.3 / 0.3 / 0.3 / 0.1
        # Perfect molecule — every component == 1.0 → total = sum of weights.
        perfect = ScoreResult(qed=1.0, sa=1.0, binding=1.0, novelty=1.0)
        self.assertAlmostEqual(agg.aggregate(perfect), 1.0, places=9)
        self.assertAlmostEqual(agg.aggregate(perfect), agg.total_weight(), places=9)
        # Half-perfect molecule → 0.5 * total_weight = 0.5.
        half = ScoreResult(qed=0.5, sa=0.5, binding=0.5, novelty=0.5)
        self.assertAlmostEqual(agg.aggregate(half), 0.5, places=9)
        # None input → 0.0 (graceful degradation).
        self.assertEqual(agg.aggregate(None), 0.0)
        # Missing component (None) is treated as 0.0.
        partial = ScoreResult(qed=1.0, sa=None, binding=1.0, novelty=None)
        expected = agg.w_qed * 1.0 + agg.w_binding * 1.0  # sa and novelty = 0
        self.assertAlmostEqual(agg.aggregate(partial), expected, places=9)

    def test_out_of_range_clipping(self) -> None:
        """Components outside [0,1] are clipped, not extrapolated."""
        agg = ScoreAggregator()
        bad = ScoreResult(qed=2.0, sa=-0.5, binding=1.5, novelty=0.0)
        # qed clipped to 1.0, sa clipped to 0.0, binding clipped to 1.0.
        expected = agg.w_qed * 1.0 + agg.w_binding * 1.0
        self.assertAlmostEqual(agg.aggregate(bad), expected, places=9)


class TestRewardAggregatorWithREINVENT4Channel(unittest.TestCase):
    """RewardAggregator wires the r_reinvent4 channel cleanly and degrades to 0.0."""

    def test_with_r_reinvent4_stub(self) -> None:
        # Defer import to avoid heavy module-level dependencies at collect time.
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        def stub_channel(_state) -> float:
            return 0.8  # plausible multi-property score for a "good" molecule

        agg = RewardAggregator(r_reinvent4=stub_channel, w_reinvent4=1.0)
        # Constructing with a stub channel must not raise.
        self.assertIsNotNone(agg.r_reinvent4)
        self.assertEqual(agg.w_reinvent4, 1.0)

        # Build a dummy state — RewardAggregator only invokes the callable
        # inside __call__, never during __init__, so any object works.
        class _Sentinel:
            pass

        value = agg(_Sentinel())
        # The stub returns 0.8 and no other channel is set, so the value
        # equals 0.8 with no bonuses.
        self.assertAlmostEqual(value, 0.8, places=9)

        # Now degrade: a stub that returns 0.0 (i.e. "adapter unavailable").
        def zero_channel(_state) -> float:
            return 0.0

        agg_zero = RewardAggregator(r_reinvent4=zero_channel, w_reinvent4=1.0)
        self.assertEqual(agg_zero(_Sentinel()), 0.0)

        # And a stub that raises — the aggregator's _safe wrapper swallows it.
        def bad_channel(_state) -> float:
            raise RuntimeError("simulated REINVENT4 binary crash")

        agg_bad = RewardAggregator(r_reinvent4=bad_channel, w_reinvent4=1.0)
        self.assertEqual(agg_bad(_Sentinel()), 0.0)

    def test_no_r_reinvent4_returns_zero(self) -> None:
        """Backward compatibility — no r_reinvent4 channel yields 0.0 contribution."""
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        agg = RewardAggregator()
        self.assertIsNone(agg.r_reinvent4)

        class _Sentinel:
            pass

        # No bonuses, no channels — value is exactly 0.0.
        self.assertEqual(agg(_Sentinel()), 0.0)


class TestREINVENT4MultipropertyAdapter(unittest.TestCase):
    """WF-Extra-2 — multiproperty subprocess-RPC adapter unit tests."""

    def test_default_components(self) -> None:
        """An adapter constructed with no components falls back to logp+ring+qed."""
        # Use a non-existent worker so __post_init__ short-circuits —
        # we only care that the dataclass default-factory contract works.
        adapter = REINVENT4MultipropertyAdapter(
            worker_python="/nonexistent/python",
            worker_script="/nonexistent/worker.py",
        )
        self.assertFalse(adapter.available)
        self.assertEqual(adapter.last_error, "worker_missing")
        self.assertEqual(
            adapter.components,
            {"logp": 0.4, "ring_count": 0.2, "qed": 0.4},
        )

    def test_invalid_mode_rejected(self) -> None:
        """mode != 'multiproperty' is refused with a structured last_error."""
        adapter = REINVENT4MultipropertyAdapter(
            mode="prior_nll",
            worker_python="/nonexistent/python",
            worker_script="/nonexistent/worker.py",
        )
        self.assertFalse(adapter.available)
        self.assertEqual(adapter.last_error, "config_invalid")

    def test_score_unavailable_returns_zeros(self) -> None:
        """An unavailable adapter returns 0.0 for every element."""
        adapter = REINVENT4MultipropertyAdapter(
            worker_python="/nonexistent/python",
            worker_script="/nonexistent/worker.py",
        )
        out = adapter.score(["CCO", "c1ccccc1", "invalid"])
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertEqual(v, 0.0)
        # Single-string call also returns a list[float].
        out_single = adapter.score("CCO")
        self.assertEqual(out_single, [0.0])

    def test_score_empty_batch(self) -> None:
        """An empty input batch returns an empty list (no spurious zeros)."""
        adapter = REINVENT4MultipropertyAdapter()
        self.assertEqual(adapter.score([]), [])

    def test_zero_weight_components_dropped(self) -> None:
        """Components with weight <= 0 are silently dropped."""
        adapter = REINVENT4MultipropertyAdapter(
            components={"qed": 0.0, "logp": 0.7, "ring_count": -0.1},
            worker_python="/nonexistent/python",
            worker_script="/nonexistent/worker.py",
        )
        self.assertEqual(adapter.components, {"logp": 0.7})

    def test_from_config_rejects_wrong_mode(self) -> None:
        """from_config refuses configs that don't declare mode=multiproperty."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"mode": "prior_nll"}')
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                REINVENT4MultipropertyAdapter.from_config(path)
        finally:
            os.unlink(path)


class TestRewardAggregatorMultipropertyChannel(unittest.TestCase):
    """WF-Extra-2 — wire REINVENT4MultipropertyAdapter into RewardAggregator.

    These tests cover the *contract* that ``r_reinvent4`` now reads from
    the multiproperty adapter instead of returning a hardcoded 0.0.
    """

    def test_reward_aggregator_reinvent4_channel_nonzero(self) -> None:
        """A known-good SMILES gets r_reinvent4 > 0 when adapter is configured.

        We stub the adapter (skip the real subprocess) to confirm the
        aggregator's channel actually calls ``adapter.score(smiles)``
        and surfaces the result through the weighted sum.
        """
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        class _StubAdapter:
            available = True
            last_error = None

            def __init__(self, value: float) -> None:
                self._value = value

            def score(self, smiles_batch):
                # The contract is: return list[float] of the same length.
                return [self._value] * len(list(smiles_batch))

        adapter = _StubAdapter(0.73)
        agg = RewardAggregator()
        # Before registration the channel is None.
        self.assertIsNone(agg.r_reinvent4)
        agg.register_reinvent4_multiproperty_channel(adapter)
        # After registration the channel is populated and w_reinvent4
        # defaults to 1.0 so the channel value flows through unchanged.
        self.assertIsNotNone(agg.r_reinvent4)
        self.assertEqual(agg.w_reinvent4, 1.0)

        # Provide a state with a callable canonical_smiles + a plain-str
        # fallback so we exercise both code paths.
        class _State:
            def canonical_smiles(self):
                return "CCO"

        value = agg(_State())
        # 0.73 is well above 0 — proves the channel is no longer the
        # hardcoded 0.0 stub.
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 0.73, places=9)

    def test_reward_aggregator_reinvent4_channel_accepts_str(self) -> None:
        """Plain string inputs are accepted (canonical_smiles is optional)."""
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        class _StubAdapter:
            available = True
            last_error = None

            def score(self, smiles_batch):
                return [0.42] * len(list(smiles_batch))

        agg = RewardAggregator()
        agg.register_reinvent4_multiproperty_channel(_StubAdapter())
        self.assertAlmostEqual(agg("CCO"), 0.42, places=9)

    def test_reward_aggregator_reinvent4_offline_graceful(self) -> None:
        """If the worker subprocess fails, r_reinvent4 falls back to 0.0.

        We construct an adapter pointing at a non-existent worker
        script so ``available`` is ``False``.  The aggregator must
        leave ``r_reinvent4`` as ``None`` and the resulting value
        must be exactly 0.0 — the channel silently degrades.
        """
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        adapter = REINVENT4MultipropertyAdapter(
            worker_python="/nonexistent/python",
            worker_script="/nonexistent/worker.py",
        )
        self.assertFalse(adapter.available)
        agg = RewardAggregator()
        agg.register_reinvent4_multiproperty_channel(adapter)
        # Channel stays None — _safe degrades None to 0.0.
        self.assertIsNone(agg.r_reinvent4)

        class _Sentinel:
            canonical_smiles = "CCO"

        # The aggregator never raises even when the SMILES is bogus.
        self.assertEqual(agg(_Sentinel()), 0.0)
        self.assertEqual(agg("invalid_smiles"), 0.0)

    def test_reward_aggregator_reinvent4_adapter_exception_graceful(self) -> None:
        """If adapter.score() raises mid-call, the channel returns 0.0.

        This is the *runtime* failure mode (the adapter passed its
        constructor probe, then died on a real call).  The closure
        installed by ``register_reinvent4_multiproperty_channel``
        must catch the exception and return 0.0 without surfacing it.
        """
        from molmetal_lam.search_alg.proof_search import RewardAggregator

        class _DyingAdapter:
            available = True
            last_error = None

            def score(self, smiles_batch):
                raise RuntimeError("simulated REINVENT4 subprocess crash")

        agg = RewardAggregator()
        agg.register_reinvent4_multiproperty_channel(_DyingAdapter())
        self.assertIsNotNone(agg.r_reinvent4)

        class _Sentinel:
            canonical_smiles = "CCO"

        self.assertEqual(agg(_Sentinel()), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
