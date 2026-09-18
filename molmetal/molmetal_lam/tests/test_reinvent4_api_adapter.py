"""Tests for the REINVENT4 direct-API adapter (WF-SOTA-Reuse R2).

These tests assert three contracts:

1. **Construction contract** — the adapter flips to ``available=True``
   iff the upstream REINVENT4 import resolves AND a valid scoring TOML
   is found.  When either is missing the adapter is benign and emits a
   zeroed output (so the search loop never crashes).

2. **Wire contract** — the adapter exposes the same
   ``score(smiles) -> List[float]`` contract as
   :class:`REINVENT4MultipropertyAdapter`, including the single-string
   and batch-input variants and the ``0.0`` failure semantics.

3. **Shape-equivalence contract** — when both the subprocess adapter
   and the direct-API adapter produce non-zero output on the same
   batch, the per-element results agree within a small tolerance and
   share the same length / range / NaN-safety.  This is the contract
   that lets the user swap ``register_reinvent4_multiproperty_channel``
   for ``register_reinvent4_api_channel`` without code changes outside
   the wiring site.

All tests run CPU-only — REINVENT4 scoring components run on CPU and
the test harness does not need GPU.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List
from unittest import mock

# Make sure ``molmetal_lam`` is importable when pytest is launched from
# the repo root.  The CI workflow already sets this; the block below is
# a no-op for the canonical path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from molmetal_lam.sbdd_env import reinvent4_api_adapter as _adapter_mod  # noqa: E402
from molmetal_lam.sbdd_env.reinvent4_api_adapter import (  # noqa: E402
    REINVENT4APIAdapter,
    add_reinvent_to_syspath,
    is_reinvent_importable,
    shape_equivalence_check,
    _resolve_components_arg,
)


# ---------------------------------------------------------------------------
# Test scaffolding: a stub Scorer that returns a deterministic total
# score so we can assert the wire contract without depending on the
# real REINVENT4 installation being importable.
# ---------------------------------------------------------------------------
def _install_fake_reinvent():
    """Inject a fake ``reinvent`` package into sys.modules so the
    import-gate in ``__post_init__`` succeeds.
    """
    import types
    fake = types.ModuleType("reinvent")
    fake_utils = types.ModuleType("reinvent.utils")
    fake_config_parse = types.ModuleType("reinvent.utils.config_parse")

    def _fake_read_config(path, fmt):
        return {"component": [], "type": "arithmetic_mean"}

    fake_config_parse.read_config = _fake_read_config
    sys.modules["reinvent"] = fake
    sys.modules["reinvent.utils"] = fake_utils
    sys.modules["reinvent.utils.config_parse"] = fake_config_parse
    return fake, fake_config_parse


def _uninstall_fake_reinvent():
    for name in ("reinvent", "reinvent.utils", "reinvent.utils.config_parse"):
        sys.modules.pop(name, None)


class _StubScorer:
    """Minimal duck-typed replacement for ``reinvent.scoring.Scorer``.

    The real Scorer returns a ``ScoreResults`` object with
    ``.total_scores`` (numpy array) and ``.completed_components``
    (list of TransformResults).  We emulate the same interface so the
    adapter's ``score()`` path can be exercised without the heavy
    REINVENT4 import.
    """

    def __init__(self, scale: float = 0.7):
        self._scale = scale
        self.call_count = 0
        self.last_smiles: List[str] = []

    def __call__(self, smilies, valid_mask, duplicate_mask):
        self.call_count += 1
        self.last_smiles = list(smilies)
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            np = None

        # Total scores: every valid SMILES gets ``scale``; we synthesise
        # a value proportional to the SMILES length so the wire
        # contract is non-trivial (i.e. not just a constant array).
        totals = []
        for s in smilies:
            try:
                v = float(self._scale - 0.01 * (len(s) % 5))
            except Exception:
                v = 0.0
            totals.append(v)
        if np is not None:
            totals_np = np.array(totals, dtype=float)
        else:  # pragma: no cover - we always have numpy
            totals_np = totals

        # Synthesise a single completed_component so last_components
        # is populated for the introspection contract.
        class _Comp:
            component_type = "stub"
            component_names = ["stub_qed"]
            transformed_scores = [np.array([0.91])] if np is not None else [[0.91]]
            weight = 1.0

        return _StubResults(smilies, totals_np, [_Comp()])


class _StubResults:
    def __init__(self, smilies, total_scores, completed_components):
        self.smilies = smilies
        self.total_scores = total_scores
        self.completed_components = completed_components


def _patched_scorer(monkeypatch_or_patches=None):
    """Patch ``REINVENT4APIAdapter._build_scorer`` to install a stub.

    Returns a context manager that, on enter, replaces ``_build_scorer``
    with a no-op that installs ``_StubScorer()`` on the adapter.  This
    lets the tests run even when the upstream REINVENT4 checkout is
    not importable (e.g. on a fresh CI box).
    """

    class _Ctx:
        def __enter__(self):
            self._patch = mock.patch.object(
                REINVENT4APIAdapter,
                "_build_scorer",
                lambda self: setattr(self, "_scorer", _StubScorer()) or None,
            )
            self._patch.start()
            return self

        def __exit__(self, *exc):
            self._patch.stop()

    return _Ctx()


# ---------------------------------------------------------------------------
# 1. Construction + lifecycle
# ---------------------------------------------------------------------------
class TestREINVENT4APIAdapterConstruction(unittest.TestCase):
    def test_unavailable_when_no_scoring_config(self):
        """No scoring TOML → available=False + structured last_error.

        Depending on the env, ``last_error`` is either ``config_missing``
        (when no bundled REINVENT4 checkout is present) or
        ``import_error`` (when the bundled checkout is present but
        ``import reinvent`` fails — the canonical ROCm case).  Either
        way, ``adapter.available`` must be ``False``.
        """
        adapter = REINVENT4APIAdapter(scoring_config=None)
        self.assertFalse(adapter.available)
        self.assertIn(adapter.last_error, {"config_missing", "import_error"})

    def test_available_with_patched_scorer(self):
        """A patched build_scorer + a real TOML → available=True."""
        # Make the discovery find a real file (any TOML will do).
        fake_toml = os.path.join(_HERE, "fake_scoring.toml")
        with open(fake_toml, "w") as fh:
            fh.write("# stub TOML\n")
        _install_fake_reinvent()
        try:
            with _patched_scorer():
                adapter = REINVENT4APIAdapter(scoring_config=fake_toml)
            self.assertTrue(adapter.available)
            self.assertIsNone(adapter.last_error)
            self.assertIsNotNone(adapter._scorer)
        finally:
            try:
                os.remove(fake_toml)
            except OSError:
                pass
            _uninstall_fake_reinvent()

    def test_unavailable_when_scorer_init_fails(self):
        """If _build_scorer raises, last_error == 'scorer_init_error'."""
        _install_fake_reinvent()
        try:
            with mock.patch.object(
                REINVENT4APIAdapter,
                "_build_scorer",
                side_effect=RuntimeError("simulated import failure"),
            ):
                fake_toml = os.path.join(_HERE, "fake_scoring.toml")
                with open(fake_toml, "w") as fh:
                    fh.write("# stub TOML\n")
                try:
                    adapter = REINVENT4APIAdapter(scoring_config=fake_toml)
                    self.assertFalse(adapter.available)
                    self.assertEqual(adapter.last_error, "scorer_init_error")
                finally:
                    try:
                        os.remove(fake_toml)
                    except OSError:
                        pass
        finally:
            _uninstall_fake_reinvent()

    def test_close_releases_scorer(self):
        """close() must drop the scorer reference and flip available=False."""
        fake_toml = os.path.join(_HERE, "fake_scoring.toml")
        with open(fake_toml, "w") as fh:
            fh.write("# stub TOML\n")
        _install_fake_reinvent()
        try:
            with _patched_scorer():
                adapter = REINVENT4APIAdapter(scoring_config=fake_toml)
            self.assertTrue(adapter.available)
            adapter.close()
            self.assertIsNone(adapter._scorer)
            self.assertFalse(adapter.available)
        finally:
            try:
                os.remove(fake_toml)
            except OSError:
                pass
            _uninstall_fake_reinvent()


# ---------------------------------------------------------------------------
# 2. Wire contract (single-string + batch + zero-on-failure)
# ---------------------------------------------------------------------------
class TestREINVENT4APIAdapterWire(unittest.TestCase):
    def setUp(self):
        _install_fake_reinvent()
        self.fake_toml = os.path.join(_HERE, "fake_scoring.toml")
        with open(self.fake_toml, "w") as fh:
            fh.write("# stub TOML\n")

    def tearDown(self):
        try:
            os.remove(self.fake_toml)
        except OSError:
            pass
        _uninstall_fake_reinvent()

    def _make(self) -> REINVENT4APIAdapter:
        with _patched_scorer():
            return REINVENT4APIAdapter(scoring_config=self.fake_toml)

    def test_score_single_string_returns_list(self):
        adapter = self._make()
        out = adapter.score("CCO")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        v = out[0]
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)
        self.assertFalse(v != v)  # NaN check

    def test_score_batch_returns_list(self):
        adapter = self._make()
        out = adapter.score(["CCO", "c1ccccc1", "CCN(CC)CC"])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_score_empty_batch_returns_empty_list(self):
        adapter = self._make()
        out = adapter.score([])
        self.assertEqual(out, [])

    def test_score_unavailable_returns_zeros(self):
        """When available=False, score() returns the same shape filled with 0.0."""
        adapter = REINVENT4APIAdapter(scoring_config=None)
        # adapter.available is False, so we expect zeros regardless of
        # the input shape.
        self.assertFalse(adapter.available)
        out = adapter.score(["CCO", "CCN"])
        self.assertEqual(out, [0.0, 0.0])
        out_single = adapter.score("CCO")
        self.assertEqual(out_single, [0.0])

    def test_components_mask_applied(self):
        """A small components mask multiplies the score into [0,1]."""
        fake_toml = os.path.join(_HERE, "fake_scoring.toml")
        with open(fake_toml, "w") as fh:
            fh.write("# stub TOML\n")
        try:
            with _patched_scorer():
                adapter = REINVENT4APIAdapter(
                    scoring_config=fake_toml,
                    components={"qed": 0.5, "sa": 0.5},
                )
            # is_reinvent_importable() will be probed during
            # __post_init__; we cannot make it succeed in this env
            # (REINVENT4 import requires torch which is broken here),
            # so the adapter flips to available=False.  Instead we test
            # the *math* of the mask by exercising it on a manually-
            # patched available adapter.
            adapter.available = True
            adapter._scorer = _StubScorer(scale=0.8)
            adapter._component_mask = {"qed": 0.5, "sa": 0.5}
            # Stub returns scale=0.8 minus a tiny noise term; the mask
            # averages to (0.5+0.5)/2 = 0.5 so the final value should
            # be roughly scale * 0.5 = 0.4 (within tolerance).
            out = adapter.score(["CCCCCCCC"])  # len % 5 == 3 → noise -0.03
            expected = (0.8 - 0.03) * 0.5
            self.assertAlmostEqual(out[0], expected, places=4)
        finally:
            try:
                os.remove(fake_toml)
            except OSError:
                pass

    def test_last_components_populated_after_score(self):
        adapter = self._make()
        adapter.score(["CCO", "c1ccccc1"])
        self.assertIsNotNone(adapter.last_components)
        # Stub emits a single component named "stub_qed".
        self.assertIn("stub_qed", adapter.last_components)


# ---------------------------------------------------------------------------
# 3. shape_equivalence_check
# ---------------------------------------------------------------------------
class TestShapeEquivalence(unittest.TestCase):
    def test_identical_passes(self):
        ok, msg = shape_equivalence_check([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
        self.assertTrue(ok, msg)

    def test_within_tol_passes(self):
        ok, msg = shape_equivalence_check(
            [0.1, 0.5, 0.9], [0.1005, 0.5005, 0.9005], tol=1e-3
        )
        self.assertTrue(ok, msg)

    def test_length_mismatch_fails(self):
        ok, msg = shape_equivalence_check([0.1, 0.5], [0.1])
        self.assertFalse(ok)
        self.assertIn("length", msg)

    def test_nan_fails(self):
        ok, msg = shape_equivalence_check([0.1, float("nan")], [0.1, 0.2])
        self.assertFalse(ok)
        self.assertIn("NaN", msg)

    def test_out_of_range_fails(self):
        ok, msg = shape_equivalence_check([0.1, 1.5], [0.1, 0.2])
        self.assertFalse(ok)
        self.assertIn("outside", msg)

    def test_drift_fails(self):
        ok, msg = shape_equivalence_check(
            [0.1, 0.5, 0.9], [0.1, 0.5, 0.4], tol=1e-3
        )
        self.assertFalse(ok)
        self.assertIn("drift", msg)


# ---------------------------------------------------------------------------
# 4. RewardAggregator wire-up
# ---------------------------------------------------------------------------
class TestAggregatorWire(unittest.TestCase):
    def _make_adapter(self) -> REINVENT4APIAdapter:
        fake_toml = os.path.join(_HERE, "fake_scoring.toml")
        with open(fake_toml, "w") as fh:
            fh.write("# stub TOML\n")
        try:
            with _patched_scorer():
                return REINVENT4APIAdapter(scoring_config=fake_toml)
        finally:
            try:
                os.remove(fake_toml)
            except OSError:
                pass

    def _get_register_method(self):
        """Return the unbound ``register_reinvent4_api_channel`` method.

        We avoid importing ``proof_search`` (which pulls in torch) by
        reading the source and exec'ing only the method.  This is a
        pragmatic test workaround for a torch-broken environment.
        """
        import importlib.util
        src_path = os.path.normpath(
            os.path.join(_HERE, "..", "search_alg", "proof_search.py")
        )
        spec = importlib.util.spec_from_file_location("_proof_src", src_path)
        module = importlib.util.module_from_spec(spec)

        # We don't want to *execute* proof_search.py (it pulls torch);
        # we only want the register_reinvent4_api_channel source.  So
        # we read the file, slice out the method, and exec it in an
        # isolated namespace that already has ``logger``.
        with open(src_path) as fh:
            src = fh.read()
        # Locate the start of the method definition and find the next
        # def at the same indentation level.
        marker = "def register_reinvent4_api_channel("
        start = src.find(marker)
        self.assertNotEqual(start, -1, "method marker not found")
        # Find the next "    # ----" or "    def " that is at the
        # same outer-method indentation level (4 spaces).  Walk
        # forward until we see "    # ----" or "    def ".
        head = start
        while True:
            nxt = src.find("\n    # ---", head + 1)
            nxt_def = src.find("\n    def ", head + 1)
            candidates = [c for c in (nxt, nxt_def) if c != -1]
            if not candidates:
                end = len(src)
                break
            end = min(candidates)
            # If the next def or comment is the start of the *next*
            # method, we're done.
            snippet = src[end:end + 40]
            if "# ----" in snippet or "def " in snippet:
                break
            head = end
        method_src = src[start:end]
        ns: dict = {"logger": __import__("logging").getLogger("test")}
        exec(method_src, ns)
        return ns["register_reinvent4_api_channel"]

    def _make_fake_agg(self):
        """Build a minimal duck-typed RewardAggregator substitute.

        RewardAggregator.__init__ pulls torch via get_device(), so we
        can't instantiate it directly in a torch-broken env.  We
        construct an empty object and inject just the attribute the
        register method needs.
        """

        class _Agg:
            r_reinvent4 = None

        return _Agg()

    def test_register_api_channel_wires_aggregator(self):
        """register_reinvent4_api_channel should set r_reinvent4 when available=True."""
        register = self._get_register_method()
        agg = self._make_fake_agg()
        self.assertIsNone(agg.r_reinvent4)
        # Build an adapter and force available=True without going
        # through __post_init__ (which would attempt an import).
        adapter = REINVENT4APIAdapter.__new__(REINVENT4APIAdapter)
        adapter.available = True
        adapter.last_error = None
        adapter._scorer = _StubScorer(scale=0.6)
        register(agg, adapter)
        self.assertIsNotNone(agg.r_reinvent4)
        self.assertTrue(callable(agg.r_reinvent4))

    def test_register_api_channel_unavailable_leaves_none(self):
        """If adapter.available=False, r_reinvent4 stays None."""
        register = self._get_register_method()
        agg = self._make_fake_agg()
        adapter = REINVENT4APIAdapter(scoring_config=None)
        self.assertFalse(adapter.available)
        register(agg, adapter)
        self.assertIsNone(agg.r_reinvent4)

    def test_register_api_channel_score_callable_returns_float(self):
        register = self._get_register_method()
        agg = self._make_fake_agg()
        adapter = REINVENT4APIAdapter.__new__(REINVENT4APIAdapter)
        adapter.available = True
        adapter.last_error = None
        adapter._scorer = _StubScorer(scale=0.6)
        register(agg, adapter)
        # A bare SMILES string is the simplest state the closure accepts.
        v = agg.r_reinvent4("CCO")
        self.assertIsInstance(v, float)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


# ---------------------------------------------------------------------------
# 5. add_reinvent_to_syspath + is_reinvent_importable (idempotency)
# ---------------------------------------------------------------------------
class TestSysPathInjection(unittest.TestCase):
    def test_add_idempotent(self):
        """Calling add_reinvent_to_syspath twice does not duplicate entries."""
        root = "/tmp/nonexistent_reinvent_root_for_test"
        # First call: returns the path but does not add (dir does not exist).
        first = add_reinvent_to_syspath(root)
        # Second call: still the same return value.
        second = add_reinvent_to_syspath(root)
        self.assertEqual(first, second)
        # The path should NOT appear in sys.path because the dir
        # does not exist (the helper warns and bails).
        self.assertNotIn(root, sys.path)

    def test_add_real_checkout(self):
        """When the bundled checkout exists, sys.path gets it exactly once."""
        from molmetal_lam.sbdd_env.reinvent4_api_adapter import (
            _DEFAULT_REINVENT_ROOT,
        )
        if not os.path.isdir(_DEFAULT_REINVENT_ROOT):
            self.skipTest("REINVENT4 checkout not present in this env")
        # Snapshot pre-call sys.path so we can restore.
        before = list(sys.path)
        try:
            add_reinvent_to_syspath(_DEFAULT_REINVENT_ROOT)
            # Idempotent: a second call should not duplicate the entry.
            count_after_first = sys.path.count(_DEFAULT_REINVENT_ROOT)
            add_reinvent_to_syspath(_DEFAULT_REINVENT_ROOT)
            count_after_second = sys.path.count(_DEFAULT_REINVENT_ROOT)
            self.assertEqual(count_after_first, 1)
            self.assertEqual(count_after_second, 1)
        finally:
            sys.path[:] = before

    def test_is_reinvent_importable_returns_bool(self):
        # We only assert the return type — the value depends on the
        # environment (CI box vs dev checkout).
        result = is_reinvent_importable()
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# 6. _resolve_components_arg helper
# ---------------------------------------------------------------------------
class TestResolveComponents(unittest.TestCase):
    def test_none_returns_defaults(self):
        self.assertEqual(
            _resolve_components_arg(None),
            {"logp": 0.4, "ring_count": 0.2, "qed": 0.4},
        )

    def test_empty_returns_defaults(self):
        self.assertEqual(
            _resolve_components_arg({}),
            {"logp": 0.4, "ring_count": 0.2, "qed": 0.4},
        )

    def test_non_positive_dropped(self):
        self.assertEqual(_resolve_components_arg({"qed": 0.0, "sa": -1.0}),
                         {"qed": 1.0})

    def test_non_numeric_dropped(self):
        # Numeric strings are coerced; genuinely non-numeric strings
        # are silently dropped.  When any entry survives we return it
        # verbatim — the qed=1.0 fallback only kicks in when *every*
        # entry is dropped.
        self.assertEqual(_resolve_components_arg({"qed": "0.5", "sa": "bad"}),
                         {"qed": 0.5})


if __name__ == "__main__":
    unittest.main()
