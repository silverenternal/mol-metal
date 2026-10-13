"""Wf-C4 — regression-guard tests for the QED penalty module.

============================================================
Why this test exists
============================================================
BLK-C4 (todo.json) claimed a ``qed_penalty.py`` SCAFFOLD existed and
only needed wiring into the trainer's loss cascade. That premise is
WRONG: no such module exists anywhere in the repo. What exists is
``molmetal/molmetal_lam/sbdd_env/qed_scorer.py``, which returns a bare
``0.0`` for an unparseable SMILES with no counter — the exact
silent-zero failure mode this repo spent 2026-10-09 fixing elsewhere.
So this test guards a NEW build, not a wire-up.

It pins four contracts:

1. A well-formed molecule with a MEASURED QED yields a finite,
   non-negative penalty equal to ``max(0, target - qed) * weight``.
2. An unparseable SMILES exercises the fallback path AND the counter
   reports it (``n_fallback`` / ``fallback_indices``) — a fallback must
   never be a silent 0.0.
3. A batch returns exactly one penalty per input, in input order.
4. **Negative control**: the function does not degenerate to returning
   0 for everything. A molecule whose QED is genuinely maximal is the
   only thing allowed to reach 0; a low-QED molecule must not.

MEASURED reference values in this file were produced by running RDKit
directly (see the assertions' comments) — they are not remembered.
Verified on 2026-10-09 with
``rdkit.Chem.QED.qed(Chem.MolFromSmiles(...))``:

    CC(=O)Oc1ccccc1C(=O)O (aspirin)     -> 0.550122
    CN1C=NC2=C1C(=O)N(C)C(=O)N2C (caffeine) -> 0.538463
    c1ccccc1 (benzene)                  -> 0.442628
    C1CCCCC1 (cyclohexane)              -> 0.422316
    CC(C)Cc1ccc(cc1)C(C)C(=O)O (ibuprofen) -> 0.821600

Honest framing (party-twenty-spirit §3.5): this test is CPU-only and
covers a module that is NOT wired into any trainer. It says nothing
about whether a QED penalty improves any metric.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load via direct path so we sidestep the ``molmetal`` package import
# (``molmetal/adapters/flow_matching_lipman/__init__.py`` pulls torch).
# Same shim as test_wf380a6_sa_penalty.py / test_wf381a3_sa_penalty.py.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PENALTY_PATH = (
    _REPO_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "qed_penalty.py"
)


def _load_qed_penalty_module():
    """Load :mod:`qed_penalty.py` without going through ``molmetal.__init__``.

    Registered in ``sys.modules`` BEFORE ``exec_module`` because
    ``@dataclass`` introspects ``sys.modules[cls.__module__]`` at class
    creation time and fails with a ``NoneType.__dict__`` AttributeError
    otherwise (same requirement documented in test_wf381a3).
    """
    mod_name = "_wf_c4_qed_penalty_loaded"
    spec = importlib.util.spec_from_file_location(mod_name, _PENALTY_PATH)
    assert spec is not None, f"failed to load spec for {_PENALTY_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass to work
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qed_mod():
    """Module-scoped fixture so the import happens once per test session."""
    return _load_qed_penalty_module()


# ---------------------------------------------------------------------------
# Cohort — QED values MEASURED on 2026-10-09, not hardcoded from memory
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_QED = 0.550122
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
CAFFEINE_QED = 0.538463
BENZENE = "c1ccccc1"
BENZENE_QED = 0.442628
CYCLOHEXANE = "C1CCCCC1"
CYCLOHEXANE_QED = 0.422316
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
IBUPROFEN_QED = 0.821600

# Confirmed by RDKit to return None from MolFromSmiles:
#   'not_a_smiles'      -> None (parse error)
#   'C(C)(C)(C)(C)C'    -> None (pentavalent carbon / hypervalent)
UNPARSEABLE = "not_a_smiles"
HYPERVALENT = "C(C)(C)(C)(C)C"

DEFAULT_TARGET_QED = 0.5
DEFAULT_QED_WEIGHT = 0.3


class TestRDKitAvailable:
    """The module must import and find RDKit in this environment."""

    def test_rdkit_is_available(self, qed_mod):
        assert qed_mod.rdkit_available(), (
            "RDKit unavailable — these tests cannot MEASURE a QED and the "
            "whole point of this file is to pin a measured value"
        )


class TestSingleMoleculePenalty:
    """Contract 1: measured QED -> finite, non-negative, correct penalty."""

    def test_aspirin_penalty_matches_measured_qed(self, qed_mod):
        """aspirin QED = 0.550122 (MEASURED) -> penalty =
        max(0, 0.5 - 0.550122) * 0.3 = 0.0 exactly, because aspirin is
        already ABOVE the 0.5 m4 threshold."""
        batch = qed_mod.qed_penalty_batch([ASPIRIN])
        assert batch.n_fallback == 0, "aspirin must parse"
        qed = float(batch.qed[0])
        assert qed == pytest.approx(ASPIRIN_QED, abs=1e-5), (
            f"measured QED drifted from the pinned value: {qed} vs "
            f"{ASPIRIN_QED} (if RDKit changed, re-measure and re-pin)"
        )
        pen = float(batch.penalties[0])
        assert math.isfinite(pen), f"penalty must be finite, got {pen}"
        assert pen >= 0.0, f"penalty must be non-negative, got {pen}"
        assert pen == pytest.approx(0.0, abs=1e-9), (
            f"aspirin QED {qed} >= target {DEFAULT_TARGET_QED} so the hinge "
            f"penalty must be exactly 0, got {pen}"
        )

    def test_low_qed_molecule_gets_positive_penalty(self, qed_mod):
        """benzene QED = 0.442628 (MEASURED) -> penalty =
        max(0, 0.5 - 0.442628) * 0.3 = 0.0172116."""
        batch = qed_mod.qed_penalty_batch([BENZENE])
        assert batch.n_fallback == 0
        qed = float(batch.qed[0])
        assert qed == pytest.approx(BENZENE_QED, abs=1e-5)
        pen = float(batch.penalties[0])
        expected = max(0.0, DEFAULT_TARGET_QED - BENZENE_QED) * DEFAULT_QED_WEIGHT
        assert pen == pytest.approx(expected, abs=1e-6), (
            f"penalty should be max(0, target - qed) * weight = {expected}, "
            f"got {pen}"
        )
        assert pen > 0.0, (
            "benzene is below the 0.5 target and MUST be penalised; a 0 here "
            "means the hinge degenerated"
        )


class TestFallbackPolicy:
    """Contract 2: unparseable SMILES -> max penalty + COUNTED, never silent 0."""

    def test_unparseable_smiles_triggers_fallback_and_counter(self, qed_mod):
        batch = qed_mod.qed_penalty_batch([UNPARSEABLE])
        assert batch.n_fallback == 1, (
            f"unparseable SMILES {UNPARSEABLE!r} must be counted as a "
            f"fallback, got n_fallback={batch.n_fallback}"
        )
        assert batch.fallback_indices == [0], (
            f"fallback index must be recorded, got {batch.fallback_indices}"
        )
        assert math.isnan(float(batch.qed[0])), (
            "an unparseable SMILES has NO QED — it must be NaN, not 0.0, so it "
            "stays distinguishable from a real molecule with QED 0.0"
        )

    def test_fallback_penalty_is_maximum_not_silent_zero(self, qed_mod):
        """The fallback penalty is ``target_qed * qed_weight`` — the max
        attainable, the conservative direction (mirrors SA's NaN -> +inf,
        but finite so it cannot poison a downstream mean with infinity).
        It must NOT be 0.0: a silent 0 would let an unparseable molecule
        look like a perfect one in a batch mean."""
        batch = qed_mod.qed_penalty_batch([UNPARSEABLE])
        pen = float(batch.penalties[0])
        expected_max = DEFAULT_TARGET_QED * DEFAULT_QED_WEIGHT
        assert pen == pytest.approx(expected_max, abs=1e-9), (
            f"fallback penalty should be the maximum {expected_max}, got {pen}"
        )
        assert pen > 0.0, (
            "fallback penalty must NOT be a silent 0.0 — that is the exact "
            "bug class this module exists to prevent"
        )

    def test_hypervalent_molecule_is_also_a_fallback(self, qed_mod):
        """Hypervalent carbon is the documented real failure mode for this
        project's generated SMILES (RDKit returns None for it)."""
        batch = qed_mod.qed_penalty_batch([HYPERVALENT])
        assert batch.n_fallback == 1, (
            f"{HYPERVALENT!r} (hypervalent C) should not parse; "
            f"n_fallback={batch.n_fallback}"
        )
        assert batch.fallback_rate == pytest.approx(1.0)

    def test_empty_string_is_a_fallback_not_a_crash(self, qed_mod):
        batch = qed_mod.qed_penalty_batch([""])
        assert batch.n_fallback == 1
        assert float(batch.penalties[0]) > 0.0

    def test_fallback_is_counted_but_mean_qed_excludes_it(self, qed_mod):
        """The headline mean QED is computed over PARSED molecules only, so a
        fallback cannot deflate the reported QED."""
        batch = qed_mod.qed_penalty_batch([ASPIRIN, UNPARSEABLE, CAFFEINE])
        assert batch.n_fallback == 1
        assert batch.mean_qed_parsed_only == pytest.approx(
            (ASPIRIN_QED + CAFFEINE_QED) / 2.0, abs=1e-4
        ), (
            "mean QED must average only the parsed molecules; averaging over "
            "the fallback (NaN excluded) is the whole point"
        )


class TestBatchOrdering:
    """Contract 3: one penalty per input, in input order."""

    def test_batch_returns_one_penalty_per_input_in_order(self, qed_mod):
        cohort = [ASPIRIN, UNPARSEABLE, BENZENE, CAFFEINE, CYCLOHEXANE]
        batch = qed_mod.qed_penalty_batch(cohort)

        assert len(batch.smiles) == len(cohort)
        assert batch.penalties.shape == (len(cohort),), (
            f"expected one penalty per input ({len(cohort)}), got shape "
            f"{batch.penalties.shape}"
        )
        # Each position must match scoring that molecule on its own.
        for i, smi in enumerate(cohort):
            solo = qed_mod.qed_penalty_batch([smi])
            assert float(batch.penalties[i]) == pytest.approx(
                float(solo.penalties[0])
            ), f"batch position {i} ({smi!r}) does not match its solo penalty"

    def test_order_is_preserved_not_sorted(self, qed_mod):
        """A low-QED-first batch must NOT come back sorted by penalty."""
        cohort = [CYCLOHEXANE, ASPIRIN]  # cyclohexane is the more-penalised
        batch = qed_mod.qed_penalty_batch(cohort)
        assert float(batch.penalties[0]) > float(batch.penalties[1]), (
            "penalties must stay in input order (position 0 = cyclohexane, "
            "the more-penalised); they were sorted"
        )

    def test_empty_batch_is_safe(self, qed_mod):
        batch = qed_mod.qed_penalty_batch([])
        assert batch.n == 0
        assert batch.n_fallback == 0
        assert batch.penalties.shape == (0,)


class TestNegativeControl:
    """Contract 4: the function must not degenerate to returning 0 always."""

    def test_zero_penalty_is_not_reachable_by_a_low_qed_molecule(self, qed_mod):
        """NEGATIVE CONTROL: cyclohexane QED = 0.422316 (MEASURED) is below
        the 0.5 target and MUST yield a strictly positive penalty. A
        degenerate implementation returning 0.0 for everything fails
        here."""
        batch = qed_mod.qed_penalty_batch([CYCLOHEXANE])
        qed = float(batch.qed[0])
        assert qed == pytest.approx(CYCLOHEXANE_QED, abs=1e-5)
        assert qed < DEFAULT_TARGET_QED, (
            "test cohort broken: cyclohexane must be BELOW the target for "
            "this control to be meaningful"
        )
        pen = float(batch.penalties[0])
        assert pen > 0.0, (
            "DEGENERATE: a below-target molecule returned penalty 0. The "
            "hinge is broken (or the function always returns 0)."
        )

    def test_zero_penalty_only_where_qed_reaches_target(self, qed_mod):
        """Across a mixed cohort, penalty == 0 must hold ONLY for
        molecules whose QED is genuinely >= target — and there must be
        at least one such molecule, so the cohort cannot pass vacuously
        if everything returned nonzero."""
        cohort = [ASPIRIN, CAFFEINE, BENZENE, CYCLOHEXANE, IBUPROFEN]
        batch = qed_mod.qed_penalty_batch(cohort)
        assert batch.n_fallback == 0

        zeros = 0
        for i, smi in enumerate(cohort):
            qed = float(batch.qed[i])
            pen = float(batch.penalties[i])
            if pen == pytest.approx(0.0, abs=1e-9):
                zeros += 1
                assert qed >= DEFAULT_TARGET_QED, (
                    f"{smi!r} got penalty 0 but its QED is only {qed} "
                    f"< target {DEFAULT_TARGET_QED}"
                )
            else:
                assert pen > 0.0, f"{smi!r} produced a negative penalty {pen}"

        assert zeros >= 1, (
            "no molecule in the cohort reached penalty 0 — the cohort no "
            "longer covers the 'good molecule is free' side of the contract"
        )
        assert zeros < len(cohort), (
            "EVERY molecule got penalty 0 — the hinge is degenerate and this "
            "control exists to catch exactly that"
        )

    def test_ibuprofen_high_qed_is_the_zero_case(self, qed_mod):
        """ibuprofen QED = 0.821600 (MEASURED), far above target -> 0."""
        batch = qed_mod.qed_penalty_batch([IBUPROFEN])
        assert float(batch.qed[0]) == pytest.approx(IBUPROFEN_QED, abs=1e-5)
        assert float(batch.penalties[0]) == pytest.approx(0.0, abs=1e-9)


class TestMirrorSAPenaltyConvention:
    """The module must follow the SA penalty's shape, with the direction flipped."""

    def test_penalty_is_nonneg_across_the_cohort(self, qed_mod):
        cohort = [ASPIRIN, BENZENE, CYCLOHEXANE, IBUPROFEN, UNPARSEABLE]
        batch = qed_mod.qed_penalty_batch(cohort)
        assert np.all(batch.penalties >= 0.0), (
            f"penalty must be non-negative everywhere: {batch.penalties}"
        )

    def test_penalty_is_linear_below_target(self, qed_mod):
        """Like the SA form, the hinge is linear in (target - qed) below
        the threshold — not a step function, not quadratic."""
        import numpy as np

        qeds = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.9], dtype=np.float64)
        pen = qed_mod.qed_aware_sampling_penalty(qeds)
        expected = np.maximum(0.0, 0.5 - qeds) * 0.3
        assert np.allclose(pen, expected)

    def test_penalty_scales_with_qed_weight(self, qed_mod):
        """Halving the weight halves the penalty — the knob a caller tunes."""
        w03 = qed_mod.qed_penalty_batch([BENZENE], qed_weight=0.3).penalties[0]
        w06 = qed_mod.qed_penalty_batch([BENZENE], qed_weight=0.6).penalties[0]
        assert float(w06) == pytest.approx(2.0 * float(w03))

    def test_qed_weight_zero_turns_the_penalty_off(self, qed_mod):
        """The OFF convention matches SA's: weight 0 -> no term at all,
        even for a fallback (weighting, not parsing, controls this)."""
        batch = qed_mod.qed_penalty_batch([BENZENE, UNPARSEABLE], qed_weight=0.0)
        assert np.all(batch.penalties == 0.0)
        # ...but the fallback is STILL counted, even when contributing 0.
        assert batch.n_fallback == 1

    def test_nan_input_maps_to_maximum_penalty(self, qed_mod):
        """Pre-computed-QED path treats NaN as the worst case, exactly as
        sa_aware_sampling_penalty treats NaN as +inf (finite here)."""
        pen = qed_mod.qed_aware_sampling_penalty(np.array([np.nan, 0.1]))
        assert float(pen[0]) == pytest.approx(0.5 * 0.3, abs=1e-9)
        assert float(pen[1]) == pytest.approx(0.4 * 0.3, abs=1e-9)

    def test_aggregate_reports_fallbacks(self, qed_mod):
        agg = qed_mod.aggregate_qed_penalty([ASPIRIN, UNPARSEABLE, BENZENE])
        assert agg.n == 3
        assert agg.n_fallback == 1
        assert agg.to_dict()["fallback_rate"] == pytest.approx(1.0 / 3.0)
        # benzene (0.4426) is below target, aspirin (0.5501) is above
        assert agg.fraction_below_target == pytest.approx(0.5)
        # mean over PARSED only
        assert agg.mean_qed == pytest.approx(
            (ASPIRIN_QED + BENZENE_QED) / 2.0, abs=1e-4
        )

    def test_aggregate_all_fallback_is_nan_not_crash(self, qed_mod):
        agg = qed_mod.aggregate_qed_penalty([UNPARSEABLE, HYPERVALENT])
        assert agg.n == 2
        assert agg.n_fallback == 2
        assert math.isnan(agg.mean_qed), (
            "with zero parsed molecules the mean QED is undefined (NaN), "
            "not 0.0 — 0.0 would be a fabricated number"
        )
        assert math.isnan(agg.fraction_below_target)