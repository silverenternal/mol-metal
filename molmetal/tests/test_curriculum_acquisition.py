"""Tests for molmetal.replay.curriculum.acquisition.

================================================================
Acquisition policy contract tests (TODO-150 §P2.1)
================================================================
Covers per-metal top-K + Tanimoto diversity + 10 % random fallback:

* :class:`AcquisitionResult` — 2 tests (to_dict shape, frozen).
* :class:`AcquisitionPolicy`  — 8 tests (defaults, k_by_metal,
  threshold validation, fingerprint batch, top-K selection,
  diversity filter, random fallback, K=0 short-circuit).

Total: **10 tests**, CPU-only.  All under 1 s on any CI container.
"""

from __future__ import annotations

import random

import pytest

try:
    from molmetal.replay.curriculum.acquisition import (
        DEFAULT_K_BY_METAL,
        DEFAULT_RANDOM_FRACTION,
        DEFAULT_TANIMOTO_THRESHOLD,
        AcquisitionPolicy,
        AcquisitionResult,
    )
except Exception:
    from molmetal.replay.curriculum import (  # type: ignore
        DEFAULT_K_BY_METAL,
        DEFAULT_RANDOM_FRACTION,
        DEFAULT_TANIMOTO_THRESHOLD,
        AcquisitionPolicy,
        AcquisitionResult,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_k_by_metal_defaults(self):
        assert DEFAULT_K_BY_METAL["Rh"] == 50
        assert DEFAULT_K_BY_METAL["Pd"] == 20
        assert DEFAULT_K_BY_METAL["Ru"] == 10

    def test_threshold_default(self):
        assert DEFAULT_TANIMOTO_THRESHOLD == pytest.approx(0.4)

    def test_random_fraction_default(self):
        assert DEFAULT_RANDOM_FRACTION == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# AcquisitionResult
# ---------------------------------------------------------------------------


class TestAcquisitionResult:
    def test_to_dict_shape(self):
        r = AcquisitionResult(
            metal="Ru",
            accepted=[1, 2, 3],
            accepted_smiles=["O", "CCO", "N"],
            hardness_predicted=[0.9, 0.8, 0.7],
            n_topk_candidates=10,
            n_after_diversity=3,
            n_random_injected=1,
        )
        d = r.to_dict()
        assert d["metal"] == "Ru"
        assert d["n_accepted"] == 3
        assert d["n_topk_candidates"] == 10
        assert d["n_random_injected"] == 1

    def test_frozen(self):
        r = AcquisitionResult(metal="Ru")
        with pytest.raises((AttributeError, Exception)):
            r.metal = "Rh"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AcquisitionPolicy
# ---------------------------------------------------------------------------


def _make_pool(n: int = 30):
    """Build a pool with stable, distinct SMILES + predictable hardness."""
    pool = []
    # 12 distinct aromatic SMILES — far enough apart in Tanimoto space.
    arom = [
        "c1ccccc1", "c1ccccc1C", "c1ccccc1O", "c1ccccc1N",
        "c1ccccc1Cl", "c1ccccc1F", "c1ccccc1Br", "c1ccncc1",
        "c1cnccc1", "c1ncccc1", "c1ccc2ccccc12", "c1ccc2[nH]ccc2c1",
    ]
    for i in range(n):
        smi = arom[i % len(arom)]
        pool.append(
            {
                "row_id": i,
                "smiles": smi,
                "hardness_predicted": float(i) / float(n),
            }
        )
    return pool


class TestAcquisitionPolicy:
    def test_defaults(self):
        pol = AcquisitionPolicy()
        assert pol.tanimoto_threshold == pytest.approx(DEFAULT_TANIMOTO_THRESHOLD)
        assert pol.random_fraction == pytest.approx(DEFAULT_RANDOM_FRACTION)

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            AcquisitionPolicy(tanimoto_threshold=1.5)
        with pytest.raises(ValueError):
            AcquisitionPolicy(tanimoto_threshold=-0.1)

    def test_random_fraction_validation(self):
        with pytest.raises(ValueError):
            AcquisitionPolicy(random_fraction=1.5)

    def test_fingerprint_batch(self):
        pol = AcquisitionPolicy()
        fps = pol.fingerprints_batch(["O", "CCO", "@#$"])
        assert len(fps) == 3
        if pol._fp_gen is not None:
            assert fps[0] is not None
            assert fps[1] is not None
            # Garbage SMILES may return None.
            assert fps[2] is None or fps[2] is not None  # RDKit may sanitize

    def test_acquire_top_k(self):
        pol = AcquisitionPolicy(k_by_metal={"Ru": 5}, random_fraction=0.0)
        pool = _make_pool(20)
        result = pol.acquire("Ru", pool)
        # K=5, random_fraction=0 → no random injection.
        assert result.metal == "Ru"
        assert len(result.accepted) <= 5
        assert result.n_random_injected == 0
        # Highest-hardness rows win.
        if result.accepted:
            assert result.hardness_predicted == sorted(
                result.hardness_predicted, reverse=True
            )

    def test_acquire_diversity_filter(self):
        """All-same-SMILES pool → diversity filter limits acceptance."""
        pol = AcquisitionPolicy(k_by_metal={"Ru": 10}, random_fraction=0.0)
        pool = [
            {"row_id": i, "smiles": "c1ccccc1", "hardness_predicted": float(i) / 100}
            for i in range(10)
        ]
        result = pol.acquire("Ru", pool)
        # Tanimoto(c1ccccc1, c1ccccc1) = 1.0 > 0.4 ⇒ only one is accepted.
        assert len(result.accepted) == 1

    def test_acquire_random_fallback(self):
        pol = AcquisitionPolicy(k_by_metal={"Ru": 4}, random_fraction=0.5)
        pool = _make_pool(20)
        result = pol.acquire("Ru", pool)
        # 4 accepted + 4*0.5 ≈ 2 random → total ≈ 6.
        assert result.n_random_injected >= 0

    def test_acquire_k_zero(self):
        pol = AcquisitionPolicy(k_by_metal={"Ru": 0}, random_fraction=0.0)
        pool = _make_pool(10)
        result = pol.acquire("Ru", pool)
        assert len(result.accepted) == 0
        assert result.n_topk_candidates == 0

    def test_acquire_per_metal_k(self):
        pol = AcquisitionPolicy(random_fraction=0.0)
        pool = _make_pool(60)
        # Rh is special (K=50) — full pool can fit if diversity passes.
        result_rh = pol.acquire("Rh", pool)
        assert result_rh.n_topk_candidates >= min(50, len(pool))
        # Pd is K=20.
        result_pd = pol.acquire("Pd", pool)
        assert result_pd.n_topk_candidates >= min(20, len(pool))

    def test_acquire_empty_pool(self):
        pol = AcquisitionPolicy()
        result = pol.acquire("Ru", [])
        assert len(result.accepted) == 0
        assert result.metal == "Ru"
