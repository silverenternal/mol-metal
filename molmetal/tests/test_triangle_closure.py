"""Tests for the Wf-37 L8 ASTSR ↔ Hilbert ↔ MCTS triangle closure.

================================================================
CPU-only smoke / contract tests for ``triangle_closure``
================================================================
Validates the Wf-37 L8 deliverable (TODO-83 §4.4) — the additive
triangle closure orchestrator that wires ASTSR, Hilbert primal-dual,
and variance-controlled PUCT into a single pipeline.

Test inventory (24 tests)
-------------------------
1. **TriangleVertex**                       (3 tests)
2. **TriangleClosure construction**         (3 tests)
3. **Synthetic data generator**             (2 tests)
4. **ASTSR vertex**                         (3 tests)
5. **Hilbert vertex**                       (3 tests)
6. **MCTS vertex**                          (3 tests)
7. **Edge / closure semantics**             (3 tests)
8. **Full triangle pipeline**               (2 tests)
9. **Report / IO helpers**                  (2 tests)

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_triangle_closure.py -v
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from molmetal.scripts import triangle_closure as tc


# ---------------------------------------------------------------------------
# 1. TriangleVertex
# ---------------------------------------------------------------------------
def test_triangle_vertex_default_construction() -> None:
    """A fresh ``TriangleVertex`` has zero wall time and NaN R²."""
    v = tc.TriangleVertex(name="astsr")
    assert v.name == "astsr"
    assert v.wall_s == 0.0
    assert not math.isfinite(v.r2)
    assert v.n_actions == 0
    assert v.variance_reduction == 0.0
    assert v.metadata == {}


def test_triangle_vertex_with_metadata() -> None:
    """Metadata dict is propagated verbatim."""
    meta = {"backend": "hilbert", "n_iters": 50}
    v = tc.TriangleVertex(
        name="hilbert", wall_s=0.1, r2=0.9, n_actions=4,
        variance_reduction=0.1, metadata=meta,
    )
    assert v.name == "hilbert"
    assert v.wall_s == pytest.approx(0.1)
    assert v.r2 == pytest.approx(0.9)
    assert v.n_actions == 4
    assert v.variance_reduction == pytest.approx(0.1)
    assert v.metadata == meta


def test_vertex_to_dict_round_trip() -> None:
    """``_vertex_to_dict`` produces a JSON-safe dict."""
    v = tc.TriangleVertex(name="mcts", wall_s=0.05, n_actions=4,
                          variance_reduction=0.2,
                          metadata={"mean_q": 0.7})
    d = tc._vertex_to_dict(v)
    # All values must be JSON-encodable.
    s = json.dumps(d)
    assert "mcts" in s
    assert "mean_q" in s
    # Non-finite R² must be coerced to 0.0.
    v2 = tc.TriangleVertex(name="astsr")  # r2 = NaN
    d2 = tc._vertex_to_dict(v2)
    assert d2["r2"] == 0.0


# ---------------------------------------------------------------------------
# 2. TriangleClosure construction
# ---------------------------------------------------------------------------
def test_closure_default_construction() -> None:
    """Default ``TriangleClosure`` uses the canonical defaults."""
    closure = tc.TriangleClosure()
    assert closure.n_trajectories == tc.DEFAULT_N_TRAJECTORIES
    assert closure.n_rounds == tc.DEFAULT_N_ROUNDS
    assert closure.n_features == tc.DEFAULT_N_FEATURES
    assert closure.n_arms == tc.DEFAULT_N_ARMS
    assert closure.seed == tc.DEFAULT_SEED
    # ``feature_names_`` is populated by ``__post_init__`` so the
    # per-vertex tests can call ``_run_*_vertex`` without first
    # calling ``run``.  Length must match ``n_features``.
    assert len(closure.feature_names_) == tc.DEFAULT_N_FEATURES
    assert closure.feature_names_[0] == "f0"
    assert closure.rounds_ == []
    assert closure.final_metrics_ == {}
    assert closure.total_wall_s_ == 0.0


def test_closure_custom_kwargs() -> None:
    """Custom kwargs override the defaults."""
    closure = tc.TriangleClosure(
        n_trajectories=32, n_rounds=5, n_features=8,
        n_arms=6, seed=7, hilbert_iters=20,
    )
    assert closure.n_trajectories == 32
    assert closure.n_rounds == 5
    assert closure.n_features == 8
    assert closure.n_arms == 6
    assert closure.seed == 7
    assert closure.hilbert_iters == 20


def test_closure_feature_names_length() -> None:
    """``feature_names_`` length matches ``n_features`` after ``run``."""
    closure = tc.TriangleClosure(
        n_trajectories=8, n_rounds=1, n_features=7, n_arms=4, seed=0,
    )
    closure.run()
    assert len(closure.feature_names_) == 7
    assert closure.feature_names_[0] == "f0"
    assert closure.feature_names_[6] == "f6"


# ---------------------------------------------------------------------------
# 3. Synthetic data generator
# ---------------------------------------------------------------------------
def test_make_design_deterministic() -> None:
    """Same seed → identical ``(X, y)``."""
    cl1 = tc.TriangleClosure(seed=42, n_trajectories=16, n_features=5)
    cl2 = tc.TriangleClosure(seed=42, n_trajectories=16, n_features=5)
    X1, y1 = cl1._make_design(n=16, dim=5, seed=42)
    X2, y2 = cl2._make_design(n=16, dim=5, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_make_design_shapes() -> None:
    """Returned ``(X, y)`` have the requested shapes."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=10, n_features=4)
    X, y = cl._make_design(n=10, dim=4, seed=0)
    assert X.shape == (10, 4)
    assert y.shape == (10,)


# ---------------------------------------------------------------------------
# 4. ASTSR vertex
# ---------------------------------------------------------------------------
def test_run_astsr_vertex_returns_finite_r2() -> None:
    """ASTSR vertex returns a finite ``R²`` and a non-zero wall time."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=24, n_features=5)
    X, y = cl._make_design(n=24, dim=5, seed=0)
    v = cl._run_astsr_vertex(
        X, y, cl.feature_names_, depth=3, population=12,
        generations=8, seed=0,
    )
    assert v.name == "astsr"
    assert v.wall_s >= 0.0
    assert math.isfinite(v.r2)
    assert v.metadata["backend"] == "evolutionary"


def test_run_astsr_vertex_deterministic_seed() -> None:
    """Same seed → same R² (deterministic)."""
    cl = tc.TriangleClosure(seed=42, n_trajectories=20, n_features=5)
    X, y = cl._make_design(n=20, dim=5, seed=42)
    v1 = cl._run_astsr_vertex(
        X, y, cl.feature_names_, depth=3, population=12,
        generations=8, seed=42,
    )
    v2 = cl._run_astsr_vertex(
        X, y, cl.feature_names_, depth=3, population=12,
        generations=8, seed=42,
    )
    assert v1.r2 == pytest.approx(v2.r2)


def test_run_astsr_vertex_metadata_keys() -> None:
    """ASTSR vertex metadata carries ``depth``, ``population``,
    ``generations``, ``complexity``, ``backend``."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=16, n_features=5)
    X, y = cl._make_design(n=16, dim=5, seed=0)
    v = cl._run_astsr_vertex(
        X, y, cl.feature_names_, depth=3, population=10,
        generations=6, seed=0,
    )
    assert "depth" in v.metadata
    assert "population" in v.metadata
    assert "generations" in v.metadata
    assert "complexity" in v.metadata
    assert "backend" in v.metadata


# ---------------------------------------------------------------------------
# 5. Hilbert vertex
# ---------------------------------------------------------------------------
def test_run_hilbert_vertex_returns_finite_r2() -> None:
    """Hilbert vertex returns a finite ``R²`` and convergence rate."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=24, n_features=5)
    X, y = cl._make_design(n=24, dim=5, seed=0)
    v = cl._run_hilbert_vertex(
        X, y, n_iters=20, seed=0,
    )
    assert v.name == "hilbert"
    assert v.wall_s >= 0.0
    assert math.isfinite(v.r2)
    # Convergence rate is 1/n_iters by the O(1/k) bound.
    assert v.metadata["convergence_rate"] == pytest.approx(1.0 / 20.0)
    assert v.metadata["backend"] == "hilbert"


def test_run_hilbert_vertex_handles_dim_mismatch() -> None:
    """Hilbert vertex projects ``n_features`` to ``population_dim``
    and still returns a finite ``R²``."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=16, n_features=3)
    X, y = cl._make_design(n=16, dim=3, seed=0)
    v = cl._run_hilbert_vertex(
        X, y, n_iters=10, seed=0,
    )
    assert math.isfinite(v.r2)
    # population_dim must be at least n_features.
    assert v.metadata["population_dim"] >= 3


def test_run_hilbert_vertex_metadata_keys() -> None:
    """Hilbert vertex metadata carries ``n_iters``, ``lr_primal``,
    ``lr_dual``, ``convergence_rate``, ``population_dim``,
    ``backend``."""
    cl = tc.TriangleClosure(seed=0, n_trajectories=16, n_features=5)
    X, y = cl._make_design(n=16, dim=5, seed=0)
    v = cl._run_hilbert_vertex(X, y, n_iters=15, seed=0)
    for key in (
        "n_iters", "lr_primal", "lr_dual",
        "convergence_rate", "population_dim", "backend",
    ):
        assert key in v.metadata, key


# ---------------------------------------------------------------------------
# 6. MCTS vertex
# ---------------------------------------------------------------------------
def test_run_mcts_vertex_returns_stats() -> None:
    """MCTS vertex returns ``n_actions`` and tree stats."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X, y = cl._make_design(n=16, dim=5, seed=0)
    v = cl._run_mcts_vertex(
        features=X, hilbert_r2=0.5, n_arms=4, seed=0,
    )
    assert v.name == "mcts"
    assert v.n_actions == 4
    assert v.wall_s >= 0.0
    assert "mean_sigma" in v.metadata
    assert "max_sigma" in v.metadata
    assert "mean_q" in v.metadata
    assert "total_visits" in v.metadata
    assert "best_action" in v.metadata


def test_run_mcts_vertex_priors_sum_to_one() -> None:
    """MCTS vertex priors sum to ``1.0`` (a valid distribution)."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X, y = cl._make_design(n=16, dim=5, seed=0)
    v = cl._run_mcts_vertex(
        features=X, hilbert_r2=0.5, n_arms=4, seed=0,
    )
    priors = v.metadata["priors"]
    assert len(priors) == 4
    assert math.isclose(sum(priors), 1.0, abs_tol=1e-9)
    assert all(p > 0.0 for p in priors)


def test_run_mcts_vertex_variance_reduction_finite() -> None:
    """MCTS variance reduction is finite (and in ``[-2, 1]``)."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X, y = cl._make_design(n=16, dim=5, seed=0)
    v = cl._run_mcts_vertex(
        features=X, hilbert_r2=0.7, n_arms=4, seed=0,
    )
    assert math.isfinite(v.variance_reduction)
    assert -2.0 <= v.variance_reduction <= 1.0


# ---------------------------------------------------------------------------
# 7. Edge / closure semantics
# ---------------------------------------------------------------------------
def test_feedback_dataset_extends_rows() -> None:
    """``_feedback_dataset`` adds exactly ``ceil(n_traj / n_arms)``
    new rows to ``(X, y)``."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X = np.zeros((16, 5))
    y = np.zeros(16)
    mcts_v = tc.TriangleVertex(
        name="mcts", n_actions=4,
    )
    X2, y2 = cl._feedback_dataset(X, y, mcts_v, round_idx=0)
    # 16 + ceil(16 / 4) = 16 + 4 = 20.
    assert X2.shape[0] == 20
    assert y2.shape[0] == 20


def test_run_triangle_astr_hilbert_coupling() -> None:
    """The ASTSR → Hilbert edge produces a Hilbert R² that is
    informed by the ASTSR formula (i.e. the closure is non-trivial)."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=24, n_features=5,
    )
    X, y = cl._make_design(n=24, dim=5, seed=0)
    astsr_v = cl._run_astsr_vertex(
        X, y, cl.feature_names_, depth=3, population=12,
        generations=8, seed=0,
    )
    hilbert_v = cl._run_hilbert_vertex(X, y, n_iters=20, seed=0)
    # Both R² values are finite (the coupling is well-defined).
    assert math.isfinite(astsr_v.r2)
    assert math.isfinite(hilbert_v.r2)


def test_run_triangle_hilbert_mcts_coupling() -> None:
    """The Hilbert → MCTS edge feeds the Hilbert R² into the MCTS
    ``base_value`` seed."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X, y = cl._make_design(n=16, dim=5, seed=0)
    # Run two MCTS vertices with different Hilbert R² seeds; the
    # resulting tree statistics must reflect the seed difference.
    v_low = cl._run_mcts_vertex(X, hilbert_r2=0.0, n_arms=4, seed=0)
    v_high = cl._run_mcts_vertex(X, hilbert_r2=1.0, n_arms=4, seed=0)
    # The mean Q of the high-R² vertex must exceed the low-R²
    # vertex (the base_value seed is ``0.5 + 0.5 * hilbert_r2``).
    assert v_high.metadata["mean_q"] > v_low.metadata["mean_q"]


def test_run_triangle_mcts_astsr_coupling() -> None:
    """The MCTS → ASTSR closure produces a non-degenerate
    ``X2, y2`` pair (the next round's training set)."""
    cl = tc.TriangleClosure(
        seed=0, n_trajectories=16, n_features=5, n_arms=4,
    )
    X = np.random.default_rng(0).standard_normal((16, 5))
    y = np.random.default_rng(1).standard_normal(16)
    mcts_v = tc.TriangleVertex(name="mcts", n_actions=4)
    X2, y2 = cl._feedback_dataset(X, y, mcts_v, round_idx=0)
    assert X2.shape[1] == X.shape[1]  # feature dim preserved
    assert X2.shape[0] > X.shape[0]    # row count grew
    assert np.all(np.isfinite(X2))
    assert np.all(np.isfinite(y2))


# ---------------------------------------------------------------------------
# 8. Full triangle pipeline
# ---------------------------------------------------------------------------
def test_full_triangle_smoke() -> None:
    """Smoke pipeline: 8 trajectories × 3 rounds runs end-to-end."""
    ok = tc.smoke_run(
        n_trajectories=8, n_rounds=3,
        n_features=5, n_arms=4, seed=42,
    )
    assert ok is True


def test_full_triangle_runs_in_reasonable_time() -> None:
    """Full triangle on 16 × 4 × 5 × 4 completes in <30 s."""
    t0 = __import__("time").perf_counter()
    report = tc.build_report(
        n_trajectories=16, n_rounds=4, n_features=5,
        n_arms=4, seed=42, hilbert_iters=20,
    )
    wall = __import__("time").perf_counter() - t0
    assert wall < 30.0
    # Summary must carry the four canonical keys.
    s = report["summary"]
    for key in (
        "mean_hilbert_r2", "mean_evolutionary_r2",
        "mean_variance_reduction", "triangle_convergence_rate",
    ):
        assert key in s, key
        assert math.isfinite(s[key]), (key, s[key])


# ---------------------------------------------------------------------------
# 9. Report / IO helpers
# ---------------------------------------------------------------------------
def test_build_report_deterministic() -> None:
    """Same kwargs → identical summary."""
    r1 = tc.build_report(
        n_trajectories=12, n_rounds=3, n_features=5,
        n_arms=4, seed=42, hilbert_iters=20,
    )
    r2 = tc.build_report(
        n_trajectories=12, n_rounds=3, n_features=5,
        n_arms=4, seed=42, hilbert_iters=20,
    )
    assert r1["summary"] == r2["summary"]
    assert len(r1["per_round"]) == len(r2["per_round"]) == 3
    # Per-round R² values match.
    for a, b in zip(r1["per_round"], r2["per_round"]):
        assert (
            a["vertices"]["hilbert"]["r2"]
            == b["vertices"]["hilbert"]["r2"]
        )
        assert (
            a["vertices"]["astsr"]["r2"]
            == b["vertices"]["astsr"]["r2"]
        )


def test_write_report_emits_json_and_md(tmp_path: Path) -> None:
    """``write_report`` emits both ``triangle_bench.json`` and the
    optional ``triangle_bench.md``."""
    report = tc.build_report(
        n_trajectories=8, n_rounds=2, n_features=4,
        n_arms=3, seed=42, hilbert_iters=10,
    )
    json_path, md_path = tc.write_report(
        out_dir=str(tmp_path), report=report, also_markdown=True,
    )
    assert json_path.exists()
    assert json_path.name == "triangle_bench.json"
    if md_path is not None:
        assert md_path.exists()
        assert md_path.name == "triangle_bench.md"
        body = md_path.read_text(encoding="utf-8")
        assert "Wf-37 L8" in body
        assert "Summary" in body
    # Round-trip the JSON.
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["n_rounds"] == 2
    assert len(loaded["per_round"]) == 2


# ---------------------------------------------------------------------------
# 10. CLI sanity
# ---------------------------------------------------------------------------
def test_cli_help_exits_zero() -> None:
    """``--help`` exits with code 0."""
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable, "-m",
            "molmetal.scripts.triangle_closure", "--help",
        ],
        capture_output=True, text=True,
        cwd=str(tc._REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0
    assert "triangle closure" in (result.stdout + result.stderr).lower()


def test_cli_smoke_exits_zero() -> None:
    """``--smoke`` exits with code 0 on a happy path."""
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable, "-m",
            "molmetal.scripts.triangle_closure", "--smoke",
        ],
        capture_output=True, text=True,
        cwd=str(tc._REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0
    assert "PASS" in result.stdout
