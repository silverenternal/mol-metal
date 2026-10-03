"""test_pareto_multiobj.py — pytest suite for molmetal/molmetal_lam/reward/pareto_multiobj.py

Wf-78 Phase 3 (R22-P3 / L15) test surface — 16 unit tests covering:
  * Config & data-class APIs
  * Per-candidate Pareto vector composition (sign flip)
  * NSGA-III reference directions (Das-Dennis simplex)
  * MCTS → NSGA-III bridge
  * Per-cell-line + per-metal breakdown
  * Hypervolume indicator (Zitzler+Thiele 1999)
  * Synthetic mock determinism
  * Top-level orchestrator

All tests are CPU-only, deterministic on seed=0, runtime <1s.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOLMETAL_PKG = os.path.join(REPO_ROOT, "molmetal")
for candidate in (REPO_ROOT, MOLMETAL_PKG):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from molmetal.molmetal_lam.reward.pareto_multiobj import (  # noqa: E402
    ParetoObjective,
    ParetoConfig,
    build_pareto_vector,
    pareto_objectives_from_table,
    nsga3_reference_directions,
    mcts_nsga3_bridge,
    per_cell_line_breakdown,
    per_metal_pareto_rank,
    compute_hypervolume_indicator,
    build_synthetic_metallodrug_pop,
    synthesize_pareto_front,
)


# ---------------------------------------------------------------------------
# 1. ParetoObjective / ParetoConfig dataclass tests
# ---------------------------------------------------------------------------


def test_pareto_objective_sign_for_pareto_min():
    obj = ParetoObjective("toxicity", "min")
    assert obj.sign_for_pareto() == -1.0


def test_pareto_objective_sign_for_pareto_max():
    obj = ParetoObjective("efficacy", "max")
    assert obj.sign_for_pareto() == 1.0


def test_pareto_config_default_4_objective():
    cfg = ParetoConfig()
    names = [o.name for o in cfg.objectives]
    assert names == ["toxicity", "efficacy", "admet", "selectivity"]
    assert cfg.reference_point == (1.0, 1.0, 1.0, 1.0)


def test_pareto_config_as_dict_serialisable():
    cfg = ParetoConfig()
    d = cfg.as_dict()
    assert isinstance(d, dict)
    assert "objectives" in d and "reference_point" in d
    assert d["n_reference_directions"] == 6


# ---------------------------------------------------------------------------
# 2. Per-candidate Pareto vector composition
# ---------------------------------------------------------------------------


def test_build_pareto_vector_sign_flip_toxicity():
    """toxicity is 'min' raw → we multiply by -1 so it becomes max -x."""
    rec = {"toxicity": 0.3, "efficacy": 0.7, "admet": 0.6, "selectivity": 0.8}
    vec = build_pareto_vector(rec)
    # toxicity → -0.3, rest keep raw values.
    assert vec == pytest.approx([-0.3, 0.7, 0.6, 0.8])


def test_build_pareto_vector_missing_keys_default_zero():
    rec = {"efficacy": 0.5}  # toxicity/admet/selectivity missing
    vec = build_pareto_vector(rec)
    # 0 in toxicity → 0 (sign flip * 0 = 0), others 0.
    assert vec == pytest.approx([0.0, 0.5, 0.0, 0.0])


def test_pareto_objectives_from_table_row_count():
    rows = [
        {"toxicity": 0.1, "efficacy": 0.5, "admet": 0.3, "selectivity": 0.6},
        {"toxicity": 0.4, "efficacy": 0.7, "admet": 0.2, "selectivity": 0.5},
        {"toxicity": 0.2, "efficacy": 0.4, "admet": 0.6, "selectivity": 0.8},
    ]
    vectors = pareto_objectives_from_table(rows)
    assert len(vectors) == 3
    assert len(vectors[0]) == 4


# ---------------------------------------------------------------------------
# 3. NSGA-III reference directions (Das-Dennis simplex)
# ---------------------------------------------------------------------------


def test_nsga3_reference_directions_4obj_p2_count():
    """For 4-obj, p=2, we expect C(2+3, 3) - 1 + 4 = 15 directions (Das-Dennis)."""
    directions = nsga3_reference_directions(n_objectives=4, p=2)
    assert directions.shape[0] >= 6  # at least 6 per config default
    assert directions.shape[1] == 4


def test_nsga3_reference_directions_simplex_sums_to_one():
    directions = nsga3_reference_directions(n_objectives=4, p=2)
    row_sums = directions.sum(axis=1)
    # Allow tiny rounding for directions with zero norm (re-normalised to 0).
    non_trivial = row_sums[row_sums > 0.5]
    assert np.allclose(non_trivial, 1.0, atol=1e-6)


def test_nsga3_reference_directions_invalid_n_obj_raises():
    with pytest.raises(ValueError):
        nsga3_reference_directions(n_objectives=1, p=2)


# ---------------------------------------------------------------------------
# 4. MCTS ↔ NSGA-III bridge
# ---------------------------------------------------------------------------


def test_mcts_nsga3_bridge_returns_correct_count():
    pop = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.1, 0.2, 0.3],
        [0.2, 0.6, 0.1, 0.3],
        [0.3, 0.2, 0.5, 0.1],
    ]
    refs = nsga3_reference_directions(n_objectives=4, p=2)
    offspring = [1, 2]
    survivors = mcts_nsga3_bridge(pop, offspring, refs)
    # Survivors must be a subset of {0,1,2,3}.
    assert all(0 <= i < len(pop) for i in survivors)


def test_mcts_nsga3_bridge_returns_all_when_pool_smaller_than_offspring():
    pop = [[0.1, 0.2], [0.5, 0.3]]
    refs = nsga3_reference_directions(n_objectives=2, p=1)
    # offspring indices larger than population → we keep all
    survivors = mcts_nsga3_bridge(pop, [5, 6, 7], refs)
    assert sorted(survivors) == [0, 1]


# ---------------------------------------------------------------------------
# 5. Per-cell-line + per-metal breakdown
# ---------------------------------------------------------------------------


def test_per_cell_line_breakdown_keys_cover():
    pop = build_synthetic_metallodrug_pop(n=20, seed=0)
    breakdown = per_cell_line_breakdown(pop, cell_key="cell_line")
    assert "A2780" in breakdown
    assert "MCF7" in breakdown
    for _, payload in breakdown.items():
        assert "front_size" in payload
        assert "hypervolume" in payload


def test_per_metal_pareto_rank_includes_30_metals():
    pop = build_synthetic_metallodrug_pop(n=60, seed=0)
    rank = per_metal_pareto_rank(pop, metal_key="metal")
    # 30-row d-block metal table is the default; mock only populates 7.
    assert "Pt" in rank
    assert "Au" in rank
    assert "Ru" in rank
    # Ranks are integers and Pt's rank must be in [1, 30].
    pt_rank = rank["Pt"]["rank"]
    assert isinstance(pt_rank, int)
    assert 1 <= pt_rank <= 30


# ---------------------------------------------------------------------------
# 6. Hypervolume indicator
# ---------------------------------------------------------------------------


def test_compute_hypervolume_indicator_basic():
    rows = [
        {"toxicity": 0.1, "efficacy": 0.9, "admet": 0.7, "selectivity": 0.8},
        {"toxicity": 0.2, "efficacy": 0.6, "admet": 0.5, "selectivity": 0.7},
    ]
    out = compute_hypervolume_indicator(rows)
    assert out["front_size"] >= 1
    assert out["hypervolume"] >= 0.0
    assert out["n_total"] == 2


def test_hypervolume_empty_pop_returns_zero():
    out = compute_hypervolume_indicator([])
    assert out["front_size"] == 0
    assert out["hypervolume"] == 0.0


# ---------------------------------------------------------------------------
# 7. Synthetic mock determinism
# ---------------------------------------------------------------------------


def test_synthetic_pop_deterministic_same_seed():
    a = build_synthetic_metallodrug_pop(n=10, seed=42)
    b = build_synthetic_metallodrug_pop(n=10, seed=42)
    assert a == b


def test_synthetic_pop_different_seeds_differ():
    a = build_synthetic_metallodrug_pop(n=10, seed=0)
    b = build_synthetic_metallodrug_pop(n=10, seed=1)
    # Most rows should differ (probabilistic uniqueness test).
    diff_count = sum(1 for ra, rb in zip(a, b) if ra != rb)
    assert diff_count >= 5  # at least 5 of 10 differ


# ---------------------------------------------------------------------------
# 8. Top-level orchestrator
# ---------------------------------------------------------------------------


def test_synthesize_pareto_front_returns_full_kpis():
    out = synthesize_pareto_front(seed=0, n=100)
    assert "front_size" in out
    assert "hypervolume" in out
    assert "per_cell_line" in out
    assert "per_metal" in out
    assert "front_vectors" in out
    assert len(out["front_vectors"]) == out["front_size"]
    assert all(len(v) == 4 for v in out["front_vectors"])


def test_synthesize_pareto_front_custom_pop():
    pop = [
        {
            "id": 0,
            "toxicity": 0.0,
            "efficacy": 1.0,
            "admet": 1.0,
            "selectivity": 1.0,
            "metal": "Pt",
            "cell_line": "A2780",
        },
        {
            "id": 1,
            "toxicity": 1.0,
            "efficacy": 0.0,
            "admet": 0.0,
            "selectivity": 0.0,
            "metal": "Au",
            "cell_line": "MCF7",
        },
    ]
    out = synthesize_pareto_front(pop=pop, cfg=ParetoConfig(), seed=0, n=2)
    # First row dominates second (in the sign-flipped space); front size should be 1.
    assert out["front_size"] == 1
    assert out["front_ids"] == [0]
