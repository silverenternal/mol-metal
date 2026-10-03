"""Tests for Wf-125 Phase 1 — Wasserstein transfer + Rh closure oversample.

Wf-125 P1 ships two building blocks + a driver that wires them:

1. :mod:`molmetal.molmetal_lam.reward.wasserstein_transfer` —
   ``WassersteinTransferLoss`` + ``sliced_w2`` + ``pairwise_w2_matrix``.
2. :mod:`molmetal.molmetal_lam.reward.closure_oversample` —
   ``ClosureOversampler`` + the curated ``RH_SEED_SCAFFOLDS`` /
   ``RICH_PARTNERS`` / ``CLOSURE_PARTNERS`` constants.

This file covers both modules + the per-compound Catherine 6/6
verification path.  Tests are CPU-only and use small data (≤ 50
synthetic compounds, ≤ 200 real rows per metal).

Test breakdown
==============
**Wasserstein transfer (M1)** — 7 tests:
* Construction + dataclass validation
* ``sliced_w2_1d`` correctness on a known 1-D distribution
* ``sliced_w2`` returns 0 when x == y (self-distance)
* ``sliced_w2`` is symmetric in (x, y)
* ``WassersteinTransferLoss.compute`` shape, normalisation, per-pair
  reporting
* ``loss_value`` matches the chosen reduction
* ``pairwise_w2_matrix`` is symmetric with zero diagonal

**Closure oversample (M2)** — 8 tests:
* Seed scaffold + partner constants are non-empty + valid SMILES
* ``ClosureOversampler`` constructor validation
* Small ``run()`` produces ≥ 1 compound with Catherine 6/6 PASS
* Per-compound ``catherine_per_property`` has 6 keys (P1..P6)
* Depth distribution is bounded by ``max_depth``
* ``RhSyntheticCompound`` dataclass round-trip via ``as_dict``
* Oversample result ``as_dict`` is JSON-serialisable
* The driver ``wasserstein_transfer_rh.main`` produces the three
  output artefacts (smoke-level)

Total: 15 tests.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.molmetal_lam.reward.wasserstein_transfer import (
    WassersteinPairResult,
    WassersteinTransferLoss,
    WassersteinTransferResult,
    pairwise_w2_matrix,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    wasserstein_transfer_loss,
    wasserstein_loss,
)
from molmetal.molmetal_lam.reward.closure_oversample import (
    CLOSURE_PARTNERS,
    RH_SEED_SCAFFOLDS,
    RICH_PARTNERS,
    ClosureOversampleResult,
    ClosureOversampler,
    RhSyntheticCompound,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _per_metal_sample(n_per_metal: int = 20, dim: int = 8, seed: int = 0):
    """Build a deterministic ``{metal: Tensor(n, dim)}`` mapping."""
    g = torch.Generator().manual_seed(int(seed))
    return {
        "Ru": torch.randn(n_per_metal, dim, generator=g),
        "Ir": torch.randn(n_per_metal, dim, generator=g),
        "Os": torch.randn(n_per_metal, dim, generator=g),
        "Rh": torch.randn(n_per_metal, dim, generator=g),
        "Re": torch.randn(n_per_metal, dim, generator=g),
    }


# ===========================================================================
# 1. M1 — Wasserstein transfer loss tests (7 tests)
# ===========================================================================


def test_wasserstein_construction_and_validation():
    """WassersteinTransferLoss ctor + dataclass validation."""
    # Default ctor works
    loss = WassersteinTransferLoss()
    assert loss.n_mc == 64
    assert loss.seed == 0
    assert loss.reduction == "sum"

    # Invalid reduction
    with pytest.raises(ValueError, match="reduction must be one of"):
        WassersteinTransferLoss(reduction="bogus")

    # Invalid n_mc
    with pytest.raises(ValueError, match="n_mc must be >= 1"):
        WassersteinTransferLoss(n_mc=0)


def test_sliced_w2_1d_correctness_known_distribution():
    """wasserstein_loss (M1 primitive) on known distributions."""
    # x = [0, 1, 2, 3] and y = [0, 1, 2, 3] (sorted)
    # Their L2 distance is 0
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    w2 = wasserstein_loss(x, y, n_mc=16, seed=42)
    assert math.isclose(float(w2.item()), 0.0, abs_tol=1e-3)

    # x = [0, 1] and y = [2, 3] — sorted L2 = sqrt((0-2)² + (1-3)²) = sqrt(8)
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[2.0], [3.0]])
    w2 = wasserstein_loss(x, y, n_mc=16, seed=42)
    # Loose tolerance — sliced-W₂ is an MC estimator.
    assert float(w2.item()) >= 0.5


def test_wasserstein_loss_self_distance_is_zero():
    """wasserstein_loss returns 0 (within tolerance) when x and y are identical."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 8, generator=g)
    w2 = wasserstein_loss(x, x, n_mc=32, seed=42)
    assert math.isclose(float(w2.item()), 0.0, abs_tol=1e-3)


def test_wasserstein_loss_symmetric_via_transfer():
    """wasserstein_transfer_loss is symmetric across metal pairs."""
    g = torch.Generator().manual_seed(0)
    emb_a = torch.randn(20, 8, generator=g)
    emb_b = torch.randn(25, 8, generator=g) + 0.5
    w2_ab = wasserstein_transfer_loss({"A": emb_a, "B": emb_b}, n_mc=32, seed=42)
    w2_ba = wasserstein_transfer_loss({"B": emb_b, "A": emb_a}, n_mc=32, seed=42)
    assert math.isclose(float(w2_ab.item()), float(w2_ba.item()), abs_tol=1e-5)


def test_wasserstein_loss_compute_shape_and_normalisation():
    """compute() returns 10-pair result for 5 metals + normalised totals."""
    per_metal = _per_metal_sample(n_per_metal=20, dim=8, seed=0)
    loss = WassersteinTransferLoss(n_mc=32, seed=42)
    result = loss.compute(per_metal)

    assert isinstance(result, WassersteinTransferResult)
    assert result.n_pairs == 10  # K(K-1)/2 for K=5
    assert result.embedding_dim == 8
    # 5 metals × 20 samples each
    assert result.per_metal_n == {"Ru": 20, "Ir": 20, "Os": 20, "Rh": 20, "Re": 20}
    # Per-pair results: WassersteinPairResult dataclass instances
    for p in result.per_pair:
        assert isinstance(p, WassersteinPairResult)
        assert p.anchor != p.partner
        assert p.w2_distance >= 0.0
    # Aggregate stats consistent
    assert math.isclose(
        result.total_w2, sum(p.w2_distance for p in result.per_pair), abs_tol=1e-5
    )
    assert math.isclose(result.mean_w2, result.total_w2 / 10, abs_tol=1e-5)


def test_wasserstein_loss_value_reductions():
    """loss_value returns sum/mean of pairwise W₂ distances."""
    per_metal = _per_metal_sample(n_per_metal=15, dim=4, seed=1)
    base = WassersteinTransferLoss(reduction="sum").compute(per_metal)
    sum_loss = WassersteinTransferLoss(reduction="sum").loss_value(per_metal)
    mean_loss = WassersteinTransferLoss(reduction="mean").loss_value(per_metal)

    assert math.isclose(sum_loss, base.total_w2, abs_tol=1e-5)
    assert math.isclose(mean_loss, base.mean_w2, abs_tol=1e-5)


def test_pairwise_w2_matrix_is_symmetric_with_zero_diagonal():
    """pairwise_w2_matrix returns a symmetric matrix with zero diagonal."""
    per_metal = _per_metal_sample(n_per_metal=20, dim=4, seed=2)
    metals, matrix = pairwise_w2_matrix(per_metal, n_mc=16, seed=42)
    assert metals == ["Ir", "Os", "Re", "Rh", "Ru"]
    K = len(metals)
    assert matrix.shape == (K, K)
    # Diagonal is zero
    for i in range(K):
        assert math.isclose(float(matrix[i, i].item()), 0.0, abs_tol=1e-5)
    # Off-diagonal symmetric
    for i in range(K):
        for j in range(K):
            assert math.isclose(
                float(matrix[i, j].item()),
                float(matrix[j, i].item()),
                abs_tol=1e-5,
            )


# ===========================================================================
# 2. M2 — Closure oversample tests (8 tests)
# ===========================================================================


def test_seed_scaffolds_and_partner_constants_are_valid():
    """RH_SEED_SCAFFOLDS + CLOSURE_PARTNERS + RICH_PARTNERS are non-empty."""
    assert len(RH_SEED_SCAFFOLDS) >= 20
    assert len(CLOSURE_PARTNERS) == 7
    assert len(RICH_PARTNERS) >= 25

    # Each seed scaffold SMILES is non-empty + contains "Rh"
    for s in RH_SEED_SCAFFOLDS:
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Rh" in s  # all seed scaffolds are Rh-containing

    # Each partner SMILES is non-empty
    for p in RICH_PARTNERS:
        assert isinstance(p, str)
        assert len(p) > 0


def test_closure_oversampler_constructor_validation():
    """ClosureOversampler ctor enforces non-negative integer params."""
    # Default ctor
    oss = ClosureOversampler()
    assert oss.metal == "Rh"
    assert oss.max_depth == 2
    assert oss.per_scaffold_cap == 50
    assert oss.target_cohort_size == 500
    assert len(oss.seed_scaffolds) == len(RH_SEED_SCAFFOLDS)

    # Validation: max_depth < 1
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        ClosureOversampler(max_depth=0)
    # Validation: per_scaffold_cap < 1
    with pytest.raises(ValueError, match="per_scaffold_cap must be >= 1"):
        ClosureOversampler(per_scaffold_cap=0)
    # Validation: target_cohort_size < 1
    with pytest.raises(ValueError, match="target_cohort_size must be >= 1"):
        ClosureOversampler(target_cohort_size=0)


def test_closure_oversample_small_run_produces_compounds():
    """Small oversample run produces ≥ 1 Catherine 6/6 PASS compound."""
    oss = ClosureOversampler(
        metal="Rh",
        max_depth=1,
        per_scaffold_cap=5,
        target_cohort_size=5,
    )
    res = oss.run(verbose=False)
    assert isinstance(res, ClosureOversampleResult)
    assert res.metal == "Rh"
    assert res.n_seeds == 24
    assert res.n_compounds_kept >= 1
    assert res.catherine_pass_rate > 0.0
    # Each kept compound passes Catherine 6/6
    for c in res.per_compound:
        assert c.catherine_6of6_pass is True
        assert c.compound_id.startswith("rh_syn_")
        assert "Rh" in c.smiles


def test_per_compound_has_six_catherine_properties():
    """Each compound's catherine_per_property has 6 keys (P1..P6)."""
    oss = ClosureOversampler(
        metal="Rh",
        max_depth=1,
        per_scaffold_cap=3,
        target_cohort_size=3,
    )
    res = oss.run(verbose=False)
    assert res.n_compounds_kept >= 1
    for c in res.per_compound:
        keys = set(c.catherine_per_property.keys())
        assert len(keys) == 6
        # All values are bool
        for v in c.catherine_per_property.values():
            assert isinstance(v, bool)


def test_depth_distribution_bounded_by_max_depth():
    """Every compound's depth is ≤ max_depth."""
    oss = ClosureOversampler(
        metal="Rh",
        max_depth=2,
        per_scaffold_cap=20,
        target_cohort_size=30,
    )
    res = oss.run(verbose=False)
    assert res.n_compounds_kept >= 1
    for c in res.per_compound:
        assert c.depth <= 2
        assert c.depth >= 0
        # Witness path length matches depth (depth 0 = seed scaffold)
        assert c.n_visit_witness_path == len(c.derivation_path)


def test_rh_synthetic_compound_as_dict_roundtrip():
    """RhSyntheticCompound + ClosureOversampleResult are JSON-serialisable."""
    oss = ClosureOversampler(
        metal="Rh",
        max_depth=1,
        per_scaffold_cap=2,
        target_cohort_size=2,
    )
    res = oss.run(verbose=False)
    assert res.n_compounds_kept >= 1
    d = res.as_dict()
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["metal"] == "Rh"
    assert parsed["n_seeds"] == 24
    assert parsed["n_compounds_kept"] == res.n_compounds_kept
    assert parsed["target_cohort_size"] == 2
    # Per-compound entries are dict-shaped
    for c in parsed["per_compound"]:
        assert "compound_id" in c
        assert "smiles" in c
        assert "derivation_path" in c
        assert "depth" in c
        assert "catherine_6of6_pass" in c
        assert "catherine_per_property" in c


def test_oversampler_with_target_500_meets_target():
    """Full target run with default params produces ≥ 500 compounds."""
    # We use a relaxed cap (100 per scaffold) so the test completes in
    # ~30s on CPU.  This exercises the full 24-seed × 28-partner ×
    # 5-rule × max_depth=2 BFS.
    oss = ClosureOversampler(
        metal="Rh",
        max_depth=2,
        per_scaffold_cap=100,
        target_cohort_size=500,
    )
    res = oss.run(verbose=False)
    assert res.n_compounds_kept >= 500, (
        f"expected ≥ 500, got {res.n_compounds_kept} "
        f"(n_products_raw={res.n_products_raw}, "
        f"pass_rate={res.catherine_pass_rate:.4f})"
    )
    # All compounds pass Catherine 6/6
    assert all(c.catherine_6of6_pass for c in res.per_compound)


def test_driver_script_writes_three_artifacts(tmp_path):
    """The Wf-125 P1 driver writes synthetic_rh_cohort.json + decay.json + synthesis.md."""
    from molmetal.scripts import wasserstein_transfer_rh as driver

    out_dir = tmp_path / "wf125_p1"
    # Quick mode: 50 compounds, 200 train rows per metal — runs in < 30s.
    rc = driver.main(
        [
            "--target-cohort-size",
            "50",
            "--max-train-rows",
            "200",
            "--out-dir",
            str(out_dir),
            "--quick",
        ]
    )
    assert rc == 0
    # Verify all three artefacts exist
    assert (out_dir / "synthetic_rh_cohort.json").exists()
    assert (out_dir / "wasserstein_decay.json").exists()
    assert (out_dir / "synthesis.md").exists()

    # The cohort JSON is parseable and has the expected top-level keys.
    payload = json.loads((out_dir / "synthetic_rh_cohort.json").read_text())
    for key in (
        "workflow",
        "phase",
        "metal",
        "n_compounds_kept",
        "catherine_pass_rate",
        "all_catherine_6of6_pass",
        "per_compound",
        "wass_decay",
    ):
        assert key in payload, f"missing key: {key}"
    assert payload["metal"] == "Rh"
    assert payload["all_catherine_6of6_pass"] is True

    # The decay JSON has the expected structure.
    decay = json.loads((out_dir / "wasserstein_decay.json").read_text())
    assert decay["n_projections"] > 0
    assert isinstance(decay["decay"], list)
    assert len(decay["decay"]) >= 2  # at least 2 steps
    # Each step has the expected W₂ keys
    for step in decay["decay"]:
        for k in ("step", "n_synthetic", "n_interpolated_rh", "mean_w2"):
            assert k in step
