"""Pytest: Integration 3 — W-Flow -> CFM ODE solver replacement.

================================================================
Mol-Metal R18 — Integration 3 verify
================================================================
25 unit tests covering the new ``w_flow_cfm_integration`` module:
:class:`WFlowCFMSolver`, :class:`WFlowCFMSelector`, and
:func:`benchmark_w_flow_cfm`.  All tests are CPU-only and run in
<30 s on a single thread (the R18 Integration 3 budget).  No GPU,
no Lipman adapter, no upstream flow_matching clone required —
every test uses the self-contained ``v(x, t) = x`` fallback or a
deterministic random velocity field.

Test coverage (25 tests, matches the 25-test spec):

1.  test_w_flow_cfm_solver_shape
2.  test_w_flow_cfm_trajectory_matches_base
3.  test_w_flow_speedup
4.  test_w_flow_selector_picks_base
5.  test_w_flow_selector_picks_w_flow
6.  test_w_flow_cfm_solver_default_strategy_is_w_flow
7.  test_w_flow_cfm_solver_solve_with_strategy_base
8.  test_w_flow_cfm_solver_solve_with_strategy_w_flow
9.  test_w_flow_cfm_solver_invalid_strategy_raises
10. test_w_flow_cfm_solver_diag_reset_on_solve
11. test_w_flow_cfm_solver_set_base_solver_chaining
12. test_w_flow_cfm_solver_invalid_n_windows_raises
13. test_w_flow_cfm_solver_invalid_n_steps_base_raises
14. test_w_flow_cfm_solver_wall_time_speedup_finite
15. test_w_flow_cfm_solver_wall_time_speedup_zero_runs_raises
16. test_w_flow_cfm_solver_solve_base_uses_default_method
17. test_w_flow_cfm_selector_low_max_atoms_invalid_raises
18. test_w_flow_cfm_selector_high_max_atoms_le_low_raises
19. test_w_flow_cfm_selector_auto_complexity_without_atoms_raises
20. test_w_flow_cfm_selector_unknown_complexity_raises
21. test_w_flow_cfm_selector_build_base_solver_chaining
22. test_w_flow_cfm_selector_select_strategy_medium_returns_base
23. test_w_flow_cfm_selector_select_solver_returns_object
24. test_benchmark_w_flow_cfm_report_keys
25. test_benchmark_w_flow_cfm_within_tol_default

All tests are deterministic (fixed seeds) so the suite is
reproducible across hosts.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman._reference.solver import (  # noqa: E402
    ODESolver,
)
from molmetal.adapters.flow_matching_lipman.w_flow import (  # noqa: E402
    WaveformRelaxationSolver,
    _default_linear_velocity_fn,
)
from molmetal.adapters.flow_matching_lipman.w_flow_cfm_integration import (  # noqa: E402
    COMPLEXITY_THRESHOLDS,
    DEFAULT_N_WINDOWS,
    SolverStrategy,
    WFlowCFMSelector,
    WFlowCFMSolver,
    benchmark_w_flow_cfm,
    make_default_w_flow_cfm_solver,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def z0():
    """Deterministic (1, 8, 3) noise tensor for reproducible tests."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 8, 3, generator=g)


@pytest.fixture
def linear_velocity_fn():
    """``v(x, t) = x`` analytic solution: ``x(t) = x_0 * exp(t)``."""
    return _default_linear_velocity_fn


@pytest.fixture
def base_solver(linear_velocity_fn):
    """Reference base solver = R17 legacy midpoint ODESolver.

    ``n_steps=40`` matches the canonical R18 P2-1 benchmark and
    keeps the comparison fair with
    ``n_windows=4 * n_substeps_per_window=10``.
    """
    return ODESolver(
        velocity_model=linear_velocity_fn,
        method="midpoint",
        n_steps=40,
    )


@pytest.fixture
def w_flow_solver():
    """W-Flow solver with R18 P2-1 defaults."""
    return WaveformRelaxationSolver(
        n_windows=4,
        n_iters_per_window=10,
        tol=1e-6,
        method="midpoint",
        n_substeps_per_window=10,
    )


@pytest.fixture
def integration_solver(base_solver, w_flow_solver):
    """A :class:`WFlowCFMSolver` with both base + W-Flow solvers wired."""
    return WFlowCFMSolver(
        base_cfm_solver=base_solver,
        w_flow_solver=w_flow_solver,
        n_windows=4,
        n_steps_base=40,
        method_base="midpoint",
    )


@pytest.fixture
def selector(base_solver, w_flow_solver):
    """A :class:`WFlowCFMSelector` with both solvers wired."""
    return WFlowCFMSelector(
        base_solver=base_solver,
        w_flow_solver=w_flow_solver,
        low_max_atoms=COMPLEXITY_THRESHOLDS["LOW_MAX_ATOMS"],
        high_max_atoms=COMPLEXITY_THRESHOLDS["HIGH_MAX_ATOMS"],
    )


# ---------------------------------------------------------------------------
# Test 1: WFlowCFMSolver.solve preserves the (n, dim) shape
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_shape(z0, integration_solver, linear_velocity_fn):
    """``WFlowCFMSolver.solve`` returns a tensor with the same shape as ``z0``."""
    with torch.no_grad():
        out = integration_solver.solve(linear_velocity_fn, z0.clone(), (0.0, 1.0))

    assert out.shape == z0.shape, (
        f"WFlowCFMSolver.solve shape {tuple(out.shape)} != z0 shape {tuple(z0.shape)}"
    )
    assert out.dtype == z0.dtype, (
        f"WFlowCFMSolver.solve dtype {out.dtype} != z0 dtype {z0.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2: WFlowCFMSolver trajectory matches the base solver within tol
# ---------------------------------------------------------------------------
def test_w_flow_cfm_trajectory_matches_base(
    z0, integration_solver, linear_velocity_fn
):
    """W-Flow's trajectory (via :class:`WFlowCFMSolver`) matches the base solver.

    We compare the integrated state at ``t=1.0`` against the
    legacy midpoint :class:`ODESolver` on the canonical ``(1, 8, 3)``
    workload.  Per the W-Flow paper §4, the geometric convergence
    rate ensures the two trajectories agree to within ``1e-1`` in
    max-abs-diff for ``n_iters_per_window=10``.
    """
    with torch.no_grad():
        w_state = integration_solver.solve(linear_velocity_fn, z0.clone(), (0.0, 1.0))
        base_state = integration_solver.solve_base(
            linear_velocity_fn,
            z0.clone(),
            n_steps=40,
            method="midpoint",
            t_span=(0.0, 1.0),
        )

    diff = (w_state - base_state).abs().max().item()
    assert diff < 0.1, (
        f"WFlowCFMSolver trajectory diverges from base by {diff:.4e}; "
        f"expected < 0.1 (W-Flow paper §4 mid-conv bound)."
    )


# ---------------------------------------------------------------------------
# Test 3: W-Flow wall-time speedup is a finite positive number
# ---------------------------------------------------------------------------
def test_w_flow_speedup(integration_solver):
    """``WFlowCFMSolver.wall_time_speedup`` returns a finite positive number.

    On the CPU skeleton we don't require a sub-1.0 speedup (the
    paper's 100x factor requires GPU window parallelism) — we
    only assert the measurement succeeds and is finite/positive.
    """
    speedup = integration_solver.wall_time_speedup(n_runs=2)
    assert math.isfinite(speedup), (
        f"wall_time_speedup must be finite; got {speedup!r}"
    )
    assert speedup > 0.0, (
        f"wall_time_speedup must be > 0; got {speedup!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: WFlowCFMSelector picks the base solver for low complexity
# ---------------------------------------------------------------------------
def test_w_flow_selector_picks_base(selector):
    """``complexity='low'`` -> :data:`SolverStrategy.BASE`."""
    assert selector.select_strategy("low") == SolverStrategy.BASE
    assert selector.select_strategy("low", n_atoms=4) == SolverStrategy.BASE
    assert selector.select_strategy("low", n_atoms=COMPLEXITY_THRESHOLDS["LOW_MAX_ATOMS"] - 1) == SolverStrategy.BASE


# ---------------------------------------------------------------------------
# Test 5: WFlowCFMSelector picks the W-Flow solver for high complexity
# ---------------------------------------------------------------------------
def test_w_flow_selector_picks_w_flow(selector):
    """``complexity='high'`` -> :data:`SolverStrategy.W_FLOW`."""
    assert selector.select_strategy("high") == SolverStrategy.W_FLOW
    assert selector.select_strategy("high", n_atoms=200) == SolverStrategy.W_FLOW
    assert (
        selector.select_strategy("high", n_atoms=COMPLEXITY_THRESHOLDS["HIGH_MAX_ATOMS"])
        == SolverStrategy.W_FLOW
    )


# ---------------------------------------------------------------------------
# Test 6: WFlowCFMSolver default strategy is W-Flow
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_default_strategy_is_w_flow(
    z0, integration_solver, linear_velocity_fn
):
    """``WFlowCFMSolver.solve`` records ``last_strategy == W_FLOW``."""
    with torch.no_grad():
        integration_solver.solve(linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert integration_solver.last_strategy == SolverStrategy.W_FLOW, (
        f"Default strategy must be W_FLOW; got {integration_solver.last_strategy!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: solve_with(strategy='base') uses the base solver
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_solve_with_strategy_base(
    z0, integration_solver, linear_velocity_fn
):
    """``solve_with(strategy='base')`` returns the legacy trajectory.

    The diagnostic ``last_strategy`` must be updated to ``"base"``.
    """
    with torch.no_grad():
        out = integration_solver.solve_with(
            linear_velocity_fn, z0.clone(), SolverStrategy.BASE,
            n_steps=40, method="midpoint",
        )

    assert out.shape == z0.shape
    assert integration_solver.last_strategy == SolverStrategy.BASE, (
        f"solve_with(strategy='base') must update last_strategy; "
        f"got {integration_solver.last_strategy!r}"
    )


# ---------------------------------------------------------------------------
# Test 8: solve_with(strategy='w_flow') uses the W-Flow solver
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_solve_with_strategy_w_flow(
    z0, integration_solver, linear_velocity_fn
):
    """``solve_with(strategy='w_flow')`` returns the W-Flow trajectory."""
    with torch.no_grad():
        out = integration_solver.solve_with(
            linear_velocity_fn, z0.clone(), SolverStrategy.W_FLOW,
        )

    assert out.shape == z0.shape
    assert integration_solver.last_strategy == SolverStrategy.W_FLOW


# ---------------------------------------------------------------------------
# Test 9: solve_with raises on unknown strategy
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_invalid_strategy_raises(
    z0, integration_solver, linear_velocity_fn
):
    """``solve_with(strategy='garbage')`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        integration_solver.solve_with(
            linear_velocity_fn, z0.clone(), "garbage",
        )


# ---------------------------------------------------------------------------
# Test 10: diagnostic state resets on every solve
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_diag_reset_on_solve(
    z0, integration_solver, linear_velocity_fn
):
    """``last_n_outer_iters`` and ``last_final_residual`` reset on each ``solve`` call."""
    with torch.no_grad():
        integration_solver.solve(linear_velocity_fn, z0.clone(), (0.0, 1.0))
    first_iters = integration_solver.last_n_outer_iters
    first_residual = integration_solver.last_final_residual
    assert first_iters >= 1
    assert math.isfinite(first_residual)

    with torch.no_grad():
        integration_solver.solve(linear_velocity_fn, z0.clone(), (0.0, 1.0))
    assert integration_solver.last_n_outer_iters >= 1
    assert math.isfinite(integration_solver.last_final_residual)
    # Both solves are deterministic; iteration counts must match.
    assert integration_solver.last_n_outer_iters == first_iters


# ---------------------------------------------------------------------------
# Test 11: set_base_cfm_solver is chainable
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_set_base_solver_chaining(z0, integration_solver):
    """``set_base_cfm_solver`` returns ``self`` for chaining."""
    out = integration_solver.set_base_cfm_solver(None)
    assert out is integration_solver, (
        "set_base_cfm_solver must return self for chaining"
    )


# ---------------------------------------------------------------------------
# Test 12: invalid n_windows raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_invalid_n_windows_raises():
    """``WFlowCFMSolver(n_windows=0)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_windows must be >= 1"):
        WFlowCFMSolver(n_windows=0)
    with pytest.raises(ValueError, match="n_windows must be >= 1"):
        WFlowCFMSolver(n_windows=-2)


# ---------------------------------------------------------------------------
# Test 13: invalid n_steps_base raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_invalid_n_steps_base_raises():
    """``WFlowCFMSolver(n_steps_base=0)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_steps_base must be >= 1"):
        WFlowCFMSolver(n_steps_base=0)


# ---------------------------------------------------------------------------
# Test 14: wall_time_speedup returns a finite positive number
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_wall_time_speedup_finite(integration_solver):
    """``wall_time_speedup`` with small ``n_runs`` is finite and positive."""
    speedup = integration_solver.wall_time_speedup(n_runs=2)
    assert math.isfinite(speedup)
    assert speedup > 0.0


# ---------------------------------------------------------------------------
# Test 15: wall_time_speedup with n_runs=0 raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_wall_time_speedup_zero_runs_raises(integration_solver):
    """``wall_time_speedup(n_runs=0)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_runs must be >= 1"):
        integration_solver.wall_time_speedup(n_runs=0)


# ---------------------------------------------------------------------------
# Test 16: solve_base uses the configured default method when not overridden
# ---------------------------------------------------------------------------
def test_w_flow_cfm_solver_solve_base_uses_default_method(
    z0, integration_solver, linear_velocity_fn
):
    """``solve_base(z0)`` (no overrides) uses ``self.method_base`` / ``self.n_steps_base``.

    We construct a fresh :class:`WFlowCFMSolver` with
    ``method_base="euler"`` and ``n_steps_base=20`` and verify
    that :meth:`solve_base` calls an :class:`ODESolver` with the
    same ``method`` / ``n_steps``.  We confirm this indirectly by
    verifying the call doesn't raise and the output has the
    expected shape.
    """
    custom_base = ODESolver(
        velocity_model=linear_velocity_fn,
        method="euler",
        n_steps=20,
    )
    solver = WFlowCFMSolver(
        base_cfm_solver=custom_base,
        w_flow_solver=WaveformRelaxationSolver(
            n_windows=2, n_iters_per_window=2, tol=1e-6,
            method="euler", n_substeps_per_window=10,
        ),
        n_windows=2,
        n_steps_base=20,
        method_base="euler",
    )
    with torch.no_grad():
        out = solver.solve_base(linear_velocity_fn, z0.clone())
    assert out.shape == z0.shape


# ---------------------------------------------------------------------------
# Test 17: invalid low_max_atoms raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_low_max_atoms_invalid_raises():
    """``WFlowCFMSelector(low_max_atoms=0)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="low_max_atoms must be >= 1"):
        WFlowCFMSelector(low_max_atoms=0, high_max_atoms=10)


# ---------------------------------------------------------------------------
# Test 18: high_max_atoms <= low_max_atoms raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_high_max_atoms_le_low_raises():
    """``high_max_atoms <= low_max_atoms`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="high_max_atoms must be > low_max_atoms"):
        WFlowCFMSelector(low_max_atoms=10, high_max_atoms=10)
    with pytest.raises(ValueError, match="high_max_atoms must be > low_max_atoms"):
        WFlowCFMSelector(low_max_atoms=20, high_max_atoms=5)


# ---------------------------------------------------------------------------
# Test 19: 'auto' complexity without n_atoms raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_auto_complexity_without_atoms_raises(selector):
    """``select_strategy('auto')`` without ``n_atoms`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="requires n_atoms"):
        selector.select_strategy("auto")


# ---------------------------------------------------------------------------
# Test 20: unknown complexity descriptor raises ValueError
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_unknown_complexity_raises(selector):
    """``select_strategy('garbage')`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown complexity"):
        selector.select_strategy("garbage")


# ---------------------------------------------------------------------------
# Test 21: build_base_solver returns an ODESolver and chains
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_build_base_solver_chaining(selector, linear_velocity_fn):
    """``build_base_solver`` returns an :class:`ODESolver` and binds it to ``self``."""
    out = selector.build_base_solver(linear_velocity_fn, n_steps=40, method="midpoint")
    assert isinstance(out, ODESolver), (
        f"build_base_solver must return ODESolver; got {type(out).__name__}"
    )
    # And it's bound to self for subsequent select_solver calls.
    assert selector.base_solver is out


# ---------------------------------------------------------------------------
# Test 22: 'medium' complexity returns the base solver
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_select_strategy_medium_returns_base(selector):
    """``complexity='medium'`` -> :data:`SolverStrategy.BASE` (CPU skeleton default)."""
    assert selector.select_strategy("medium") == SolverStrategy.BASE
    # Even when n_atoms falls in the medium band, we conservatively
    # default to base (W-Flow overhead on CPU is not worth it for
    # medium-sized molecules).
    mid_atoms = (
        COMPLEXITY_THRESHOLDS["LOW_MAX_ATOMS"]
        + COMPLEXITY_THRESHOLDS["HIGH_MAX_ATOMS"]
    ) // 2
    assert (
        selector.select_strategy("auto", n_atoms=mid_atoms) == SolverStrategy.BASE
    )


# ---------------------------------------------------------------------------
# Test 23: select_solver returns the actual solver object
# ---------------------------------------------------------------------------
def test_w_flow_cfm_selector_select_solver_returns_object(selector, base_solver, w_flow_solver):
    """``select_solver`` returns the bound solver object for the chosen strategy."""
    base_obj = selector.select_solver("low")
    wflow_obj = selector.select_solver("high")
    # The base object is the same instance we supplied.
    assert base_obj is base_solver, (
        "select_solver('low') must return the bound base_solver instance"
    )
    # The W-Flow object is the W-Flow solver we supplied.
    assert wflow_obj is w_flow_solver, (
        "select_solver('high') must return the bound w_flow_solver instance"
    )


# ---------------------------------------------------------------------------
# Test 24: benchmark_w_flow_cfm returns the documented key set
# ---------------------------------------------------------------------------
def test_benchmark_w_flow_cfm_report_keys(z0, base_solver, w_flow_solver):
    """``benchmark_w_flow_cfm`` returns all 10 documented keys."""
    report = benchmark_w_flow_cfm(
        base_solver=base_solver,
        w_flow_solver=w_flow_solver,
        n_runs=1,
        z0=z0.clone(),
        t=1.0,
        n_steps_base=20,
        tol_accuracy=0.05,
    )
    expected_keys = {
        "strategy",
        "w_total_time_s",
        "base_total_time_s",
        "speedup",
        "max_abs_diff",
        "within_tol",
        "convergence",
        "n_runs",
        "z0_shape",
        "n_windows",
    }
    assert set(report.keys()) == expected_keys, (
        f"benchmark_w_flow_cfm keys {set(report.keys())} != expected {expected_keys}"
    )
    assert report["strategy"] == SolverStrategy.W_FLOW
    assert report["n_runs"] == 1
    assert tuple(report["z0_shape"]) == tuple(z0.shape)
    assert report["n_windows"] == w_flow_solver.n_windows
    assert isinstance(report["convergence"], dict)


# ---------------------------------------------------------------------------
# Test 25: benchmark_w_flow_cfm accuracy is within tol by default
# ---------------------------------------------------------------------------
def test_benchmark_w_flow_cfm_within_tol_default(z0, base_solver, w_flow_solver):
    """``benchmark_w_flow_cfm`` reports ``within_tol=True`` by default.

    The default ``tol_accuracy=1e-2`` is loose enough that the
    W-Flow trajectory (with R18 P2-1 defaults) is within the
    legacy midpoint ``ODESolver`` trajectory at ``n_steps=40`` on
    the canonical ``(1, 8, 3)`` workload with the trivial
    ``v(x, t) = x`` velocity field.
    """
    report = benchmark_w_flow_cfm(
        base_solver=base_solver,
        w_flow_solver=w_flow_solver,
        n_runs=1,
        z0=z0.clone(),
        t=1.0,
        n_steps_base=40,
    )
    assert report["within_tol"] is True, (
        f"W-Flow trajectory not within default tol: "
        f"max_abs_diff={report['max_abs_diff']:.4e}, "
        f"speedup={report['speedup']:.4f}"
    )
    assert math.isfinite(report["max_abs_diff"])