"""Mol-Metal W-Flow -> CFM ODE solver integration layer.

================================================================
Integration 3: W-Flow CFM ODE solver replacement
================================================================
The R17 ship ``flow_matching_lipman.ode_solver`` is the production
CFM ODE solver — a fixed-step Euler / midpoint loop over ``n_steps``
sub-steps.  For stiff or coupled velocity fields (e.g. tight Pt-S
coordination pockets) each sub-step is small and the global error
accumulates as ``O(h)`` (Euler) or ``O(h^2)`` (midpoint).  200 steps
over ``[0, 1]`` costs ~6 ms per molecule on the RX 7800 XT and
dominates the wall-time of the 100-pocket sampling sweep.

**W-Flow** (Stanford ICML 2026, ``arXiv:2605.11755``) introduces
*waveform relaxation* — split the time interval into ``n_windows``
sub-intervals, integrate each independently using the previous
iteration's trajectory as the boundary condition, converge
geometrically (linear convergence with rate equal to the coupling
factor of the ODE).  The CPU skeleton ships in
:mod:`molmetal.adapters.flow_matching_lipman.w_flow`.

This module is the **integration layer** that wires W-Flow into the
existing CFM stack **without modifying** ``ode_solver.py``.  It
provides:

* :class:`WFlowCFMSolver` — wraps a base CFM solver + W-Flow
  solver so callers can pick a strategy at solve-time.
* :class:`WFlowCFMSelector` — runtime strategy selector based on a
  *complexity* descriptor (``"low"`` / ``"medium"`` / ``"high"``).
* :func:`benchmark_w_flow_cfm` — head-to-head wall-time + accuracy
  comparison helper used by tests + the verify script.

Public API
----------
- :class:`WFlowCFMSolver`     — integrated W-Flow + base CFM solver.
- :class:`WFlowCFMSelector`   — runtime strategy selector.
- :func:`benchmark_w_flow_cfm` — benchmark helper.
- :class:`SolverStrategy`     — ``str`` enum for the selector.
- :func:`make_default_w_flow_cfm_solver` — factory used by tests.

Constraints
-----------
- CPU-only, ``torch.no_grad`` throughout the solve path.
- Does **not** modify ``ode_solver.py`` / ``loss.py`` (R17 ship
  invariant) — only references ``w_flow.py`` (R18 P2-1 ship) +
  ``_reference.solver.ODESolver`` (the R17 shim).
- All inputs / outputs follow the ``ModelWrapper`` contract:
  ``velocity_fn(x, t) -> Tensor`` of same shape as ``x``; ``t`` is
  a 0-d scalar tensor.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ._reference.solver import ODESolver
from .w_flow import (
    DEFAULT_N_ITERS_PER_WINDOW,
    DEFAULT_N_RUNS,
    DEFAULT_N_WINDOWS,
    WaveformRelaxationSolver,
    compare_solvers as _w_flow_compare_solvers,
)


# ---------------------------------------------------------------------------
# Strategy enum (string-based for stability across pickle/imports)
# ---------------------------------------------------------------------------
class SolverStrategy:
    """Strategy names used by :class:`WFlowCFMSelector`.

    Constants
    ---------
    BASE   : use the legacy midpoint ODE solver (R17 production).
    W_FLOW : use the W-Flow waveform-relaxation solver.
    """

    BASE: str = "base"
    W_FLOW: str = "w_flow"

    _ALL: Tuple[str, ...] = (BASE, W_FLOW)

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        """Tuple of all known strategy names (read-only)."""
        return cls._ALL

    @classmethod
    def is_valid(cls, name: str) -> bool:
        """True iff ``name`` is a known strategy."""
        return str(name) in cls._ALL


# ---------------------------------------------------------------------------
# Complexity descriptors (used by :class:`WFlowCFMSelector`)
# ---------------------------------------------------------------------------
COMPLEXITY_THRESHOLDS: Dict[str, int] = {
    # ``low``    -> ``n_atoms < LOW_MAX_ATOMS``  -> base solver
    # ``medium`` -> ``n_atoms in [LOW_MAX, HIGH_MAX)`` -> base solver
    # ``high``   -> ``n_atoms >= HIGH_MAX_ATOMS`` -> W-Flow
    # These are honest defaults — small molecules don't benefit
    # from waveform relaxation (overhead dominates); only large
    # coupled systems do.  Tunable via constructor.
    "LOW_MAX_ATOMS": 32,
    "HIGH_MAX_ATOMS": 96,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_base_solver(
    base_cfm_solver: Any,
    velocity_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    *,
    n_steps: int,
    method: str,
) -> Any:
    """Resolve a base CFM solver, building an ODESolver if needed.

    ``base_cfm_solver`` may be:

    * ``None`` — we build a fresh :class:`ODESolver` bound to
      ``velocity_fn`` (default midpoint, ``n_steps`` substeps).
    * An :class:`ODESolver`-like object — returned as-is (the
      caller-supplied ``velocity_fn`` / ``n_steps`` / ``method`` are
      only used when we construct a fresh one).
    * Anything else — returned as-is and the caller is expected to
      have wired it correctly (we don't introspect).
    """
    if base_cfm_solver is not None:
        return base_cfm_solver
    if velocity_fn is None:
        # No velocity_fn and no pre-built solver — emit a clear
        # error so callers don't silently fall through to a
        # dead solver.
        raise ValueError(
            "_resolve_base_solver requires either base_cfm_solver "
            "or velocity_fn to construct a fresh ODESolver.",
        )
    return ODESolver(
        velocity_model=velocity_fn,
        method=str(method),
        n_steps=int(n_steps),
    )


def _call_solver(
    solver: Any,
    z0: torch.Tensor,
    *,
    n_steps: Optional[int] = None,
    method: Optional[str] = None,
) -> torch.Tensor:
    """Call ``solver.sample(z0, ...)`` or ``solver(z0)`` permissively.

    Mirrors :func:`molmetal.adapters.flow_matching_lipman.w_flow._call_base_solver`
    so the wrapper is signature-compatible with both the legacy
    :class:`ODESolver` and any callable that exposes a ``.sample``
    method.
    """
    if hasattr(solver, "sample") and callable(getattr(solver, "sample")):
        try:
            return solver.sample(z0, n_steps=n_steps, method=method)
        except TypeError:
            # Older positional-only API.
            if n_steps is not None and method is not None:
                return solver.sample(z0, int(n_steps), str(method))
            return solver.sample(z0)
    if callable(solver):
        return solver(z0)
    raise TypeError(
        f"solver must expose .sample(...) or be callable; got {type(solver).__name__}"
    )


# ---------------------------------------------------------------------------
# WFlowCFMSolver
# ---------------------------------------------------------------------------
class WFlowCFMSolver:
    """Integrated W-Flow + base CFM solver.

    Wraps both a legacy ``base_cfm_solver`` (e.g. the R17 midpoint
    :class:`ODESolver`) and a W-Flow
    :class:`WaveformRelaxationSolver` so callers can pick a strategy
    at solve-time.  :meth:`solve` returns the W-Flow trajectory by
    default (W-Flow is the headline Integration 3 deliverable);
    :meth:`solve_base` returns the legacy trajectory; :meth:`solve_with`
    selects the strategy explicitly.

    Parameters
    ----------
    base_cfm_solver : ``ODESolver``-like, optional
        Legacy CFM solver.  When ``None`` and :meth:`solve` is
        called with a ``velocity_fn``, we construct a fresh
        ``ODESolver(method="midpoint", n_steps=n_steps_base)``.
    w_flow_solver : ``WaveformRelaxationSolver``, optional
        W-Flow solver.  When ``None`` we build one with the
        canonical R18 P2-1 defaults via :func:`make_default_w_flow_cfm_solver`.
    n_windows : int, default 4
        Forwarded to ``w_flow_solver`` when we build one.
    n_steps_base : int, default 40
        Forwarded to the base ODESolver when we build one.  Must
        be ``>= n_windows * n_substeps_per_window`` for a fair
        comparison (we use 40 = 4 * 10 as the R18 P2-1 canonical).
    method_base : str, default "midpoint"
        Inner method for the base ODESolver when we build one.

    Notes
    -----
    The :meth:`solve` method delegates to ``w_flow_solver.solve`` —
    it does **not** call the base solver.  Use :meth:`solve_base`
    (or :meth:`solve_with` with ``strategy="base"``) for the legacy
    trajectory.  The wall-time speedup is reported as
    ``w_total_time / base_total_time`` (lower = faster W-Flow).
    """

    def __init__(
        self,
        base_cfm_solver: Any = None,
        w_flow_solver: Optional[WaveformRelaxationSolver] = None,
        n_windows: int = DEFAULT_N_WINDOWS,
        n_steps_base: int = 40,
        method_base: str = "midpoint",
    ) -> None:
        if int(n_windows) < 1:
            raise ValueError(
                f"n_windows must be >= 1; got {n_windows!r}"
            )
        if int(n_steps_base) < 1:
            raise ValueError(
                f"n_steps_base must be >= 1; got {n_steps_base!r}"
            )
        self.n_windows = int(n_windows)
        self.n_steps_base = int(n_steps_base)
        self.method_base = str(method_base)

        # The base solver may be supplied later via :meth:`set_base_cfm_solver`
        # — typical when the velocity_fn is unknown at construction
        # time.  We accept ``None`` here.
        self.base_cfm_solver = base_cfm_solver

        # The W-Flow solver may be supplied, or we build a default.
        if w_flow_solver is None:
            w_flow_solver = WaveformRelaxationSolver(
                n_windows=int(n_windows),
                n_iters_per_window=int(DEFAULT_N_ITERS_PER_WINDOW),
                tol=1e-6,
                method=str(method_base),
                n_substeps_per_window=10,
            )
        self.w_flow_solver = w_flow_solver

        # Diagnostic state populated by :meth:`solve` and reset on
        # every call.  Tests + scripts read these to verify
        # behaviour.
        self.last_strategy: str = SolverStrategy.W_FLOW
        self.last_n_outer_iters: int = 0
        self.last_final_residual: float = float("inf")
        self.last_converged: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_base_cfm_solver(self, base_cfm_solver: Any) -> "WFlowCFMSolver":
        """Bind / replace the legacy CFM solver.

        Returns
        -------
        self
            For chaining.
        """
        self.base_cfm_solver = base_cfm_solver
        return self

    # ------------------------------------------------------------------
    # Solve paths
    # ------------------------------------------------------------------
    def solve(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Integrate via the W-Flow solver.

        Parameters
        ----------
        velocity_fn : ``(x, t) -> Tensor`` callable
            Velocity field.  ``x`` has shape ``(B, N, 3)``; ``t`` is
            a 0-d scalar tensor.
        x0 : ``(B, N, 3)`` tensor
            Initial state at ``t = t_span[0]``.
        t_span : ``(t0, t1)``, default ``(0.0, 1.0)``
            Tuple of Python floats.  ``t1 > t0`` is required.

        Returns
        -------
        ``(B, N, 3)`` tensor
            The integrated state at ``t = t_span[1]``.  Shape and
            dtype preserved from ``x0``.
        """
        # Reset diagnostic state — :meth:`solve` is re-entrant safe.
        self.last_strategy = SolverStrategy.W_FLOW
        self.last_n_outer_iters = 0
        self.last_final_residual = float("inf")
        self.last_converged = False

        result = self.w_flow_solver.solve(velocity_fn, x0, t_span)
        # Copy diagnostics from the inner solver.
        self.last_n_outer_iters = int(self.w_flow_solver.last_n_outer_iters)
        self.last_final_residual = float(self.w_flow_solver.last_final_residual)
        self.last_converged = bool(self.w_flow_solver.last_converged)
        return result

    def solve_base(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        *,
        n_steps: Optional[int] = None,
        method: Optional[str] = None,
        t_span: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Integrate via the legacy base CFM solver.

        Constructs an :class:`ODESolver` on demand when one wasn't
        supplied at construction.  ``t_span`` is honoured via
        :meth:`ODESolver.sample` — when the supplied base solver
        doesn't expose ``.sample`` with ``time_grid`` support, we
        fall back to ``sample(z0, n_steps, method)`` and the
        ``t_span`` is effectively ``(0, 1)`` (the canonical
        Lipman 2023 grid).
        """
        # Reset diagnostic state.
        self.last_strategy = SolverStrategy.BASE
        n_steps_eff = int(n_steps) if n_steps is not None else self.n_steps_base
        method_eff = str(method) if method is not None else self.method_base
        solver = _resolve_base_solver(
            self.base_cfm_solver,
            velocity_fn,
            n_steps=n_steps_eff,
            method=method_eff,
        )
        return _call_solver(
            solver,
            x0,
            n_steps=n_steps_eff,
            method=method_eff,
        )

    def solve_with(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        strategy: str,
        *,
        n_steps: Optional[int] = None,
        method: Optional[str] = None,
        t_span: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Integrate via the explicitly named strategy.

        Parameters
        ----------
        velocity_fn : ``(x, t) -> Tensor`` callable
            Velocity field.
        x0 : ``(B, N, 3)`` tensor
            Initial state.
        strategy : str
            ``SolverStrategy.BASE`` or ``SolverStrategy.W_FLOW``.
        n_steps, method : optional overrides forwarded to the
            chosen solver.
        t_span : ``(t0, t1)``, default ``(0.0, 1.0)``
            Forwarded to the W-Flow path; ignored on the base path
            (which uses Lipman's canonical ``[0, 1]`` grid).

        Raises
        ------
        ValueError
            If ``strategy`` is not a known :class:`SolverStrategy`
            value.
        """
        if not SolverStrategy.is_valid(strategy):
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of "
                f"{SolverStrategy.all()!r}."
            )
        if strategy == SolverStrategy.BASE:
            return self.solve_base(
                velocity_fn, x0,
                n_steps=n_steps, method=method, t_span=t_span,
            )
        return self.solve(velocity_fn, x0, t_span=t_span)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    def wall_time_speedup(
        self,
        n_runs: int = DEFAULT_N_RUNS,
        *,
        z0: Optional[torch.Tensor] = None,
        t: float = 1.0,
        velocity_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        n_steps_base: Optional[int] = None,
        method_base: Optional[str] = None,
    ) -> float:
        """Measure the empirical wall-time speedup (W-Flow vs base).

        Returns
        -------
        float
            ``w_total_time / base_total_time``.  Lower = faster
            W-Flow.  Returns ``float("inf")`` when either solver
            fails on every run (so callers can detect a
            pathological solver via ``math.isinf(result)``).

        Parameters
        ----------
        n_runs : int, default 10
            Number of timing runs.
        z0 : ``(B, N, 3)`` tensor, optional
            Initial state.  When ``None`` we generate a
            deterministic ``(1, 8, 3)`` noise tensor with
            ``torch.manual_seed(0)``.
        t : float, default 1.0
            Final time.  Forwarded to :meth:`solve`.
        velocity_fn : callable, optional
            Velocity field for the W-Flow solver.  When ``None`` we
            fall back to a self-contained ``v(x, t) = x`` field so
            the measurement is reproducible without dragging the
            production EGNN in.
        n_steps_base : int, optional
            ``n_steps`` for the base solver.  Defaults to
            ``self.n_steps_base``.
        method_base : str, optional
            Inner method for the base solver.  Defaults to
            ``self.method_base``.
        """
        if int(n_runs) < 1:
            raise ValueError(f"n_runs must be >= 1; got {n_runs!r}")
        if z0 is None:
            g = torch.Generator().manual_seed(0)
            z0 = torch.randn(1, 8, 3, generator=g)
        n_steps_base_eff = (
            int(n_steps_base) if n_steps_base is not None else self.n_steps_base
        )
        method_base_eff = (
            str(method_base) if method_base is not None else self.method_base
        )
        if velocity_fn is None:
            velocity_fn = _default_linear_velocity_fn

        # ---- W-Flow timing ----
        w_times: list[float] = []
        for _ in range(int(n_runs)):
            start = time.perf_counter()
            try:
                self.solve(velocity_fn, z0.clone(), t_span=(0.0, float(t)))
            except Exception:
                w_times.append(float("inf"))
                continue
            w_times.append(time.perf_counter() - start)
        w_finite = [u for u in w_times if math.isfinite(u)]
        if not w_finite:
            return float("inf")
        w_total = float(sum(w_finite))

        # ---- Base solver timing ----
        base_times: list[float] = []
        for _ in range(int(n_runs)):
            start = time.perf_counter()
            try:
                self.solve_base(
                    velocity_fn,
                    z0.clone(),
                    n_steps=n_steps_base_eff,
                    method=method_base_eff,
                    t_span=(0.0, float(t)),
                )
            except Exception:
                base_times.append(float("inf"))
                continue
            base_times.append(time.perf_counter() - start)
        base_finite = [u for u in base_times if math.isfinite(u)]
        if not base_finite:
            return float("inf")
        base_total = float(sum(base_finite))

        if base_total <= 0.0:
            return float("inf")
        return float(w_total / base_total)


# ---------------------------------------------------------------------------
# WFlowCFMSelector
# ---------------------------------------------------------------------------
class WFlowCFMSelector:
    """Runtime strategy selector (base vs W-Flow) by complexity descriptor.

    The selector wraps both solvers and returns the appropriate one
    based on a *complexity* string:

    * ``"low"`` — small molecule (``n_atoms < LOW_MAX_ATOMS``).
      The waveform-relaxation Python overhead dominates and W-Flow
      is not worth it; return the base solver.
    * ``"medium"`` — intermediate.  Default to base (the
      conservative choice for the CPU skeleton).  A future R18+
      sprint can promote this to W-Flow.
    * ``"high"`` — large molecule (``n_atoms >= HIGH_MAX_ATOMS``).
      W-Flow's geometric convergence pays off; return W-Flow.

    Parameters
    ----------
    base_solver : ``ODESolver``-like, optional
        Legacy CFM solver.  When ``None`` we build a fresh one on
        demand via :meth:`build_base_solver`.
    w_flow_solver : ``WaveformRelaxationSolver``, optional
        W-Flow solver.  When ``None`` we build one with the
        canonical R18 P2-1 defaults.
    low_max_atoms : int, default 32
        Threshold below which complexity is ``"low"``.
    high_max_atoms : int, default 96
        Threshold at or above which complexity is ``"high"``.

    Notes
    -----
    The selector does **not** itself integrate the ODE — it returns
    the *solver object* (base or W-Flow) and the caller does the
    integration.  This keeps the selector's surface area small and
    testable.
    """

    def __init__(
        self,
        base_solver: Any = None,
        w_flow_solver: Optional[WaveformRelaxationSolver] = None,
        low_max_atoms: int = COMPLEXITY_THRESHOLDS["LOW_MAX_ATOMS"],
        high_max_atoms: int = COMPLEXITY_THRESHOLDS["HIGH_MAX_ATOMS"],
    ) -> None:
        if int(low_max_atoms) < 1:
            raise ValueError(
                f"low_max_atoms must be >= 1; got {low_max_atoms!r}"
            )
        if int(high_max_atoms) <= int(low_max_atoms):
            raise ValueError(
                f"high_max_atoms must be > low_max_atoms; "
                f"got high_max_atoms={high_max_atoms!r}, "
                f"low_max_atoms={low_max_atoms!r}"
            )
        self.low_max_atoms = int(low_max_atoms)
        self.high_max_atoms = int(high_max_atoms)
        self.base_solver = base_solver
        if w_flow_solver is None:
            w_flow_solver = WaveformRelaxationSolver(
                n_windows=int(DEFAULT_N_WINDOWS),
                n_iters_per_window=int(DEFAULT_N_ITERS_PER_WINDOW),
                tol=1e-6,
                method="midpoint",
                n_substeps_per_window=10,
            )
        self.w_flow_solver = w_flow_solver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_strategy(
        self,
        complexity: str,
        *,
        n_atoms: Optional[int] = None,
    ) -> str:
        """Return the strategy name for the given complexity descriptor.

        Parameters
        ----------
        complexity : str
            ``"low"``, ``"medium"``, or ``"high"``.  Anything else
            raises ``ValueError``.
        n_atoms : int, optional
            When provided, we *also* derive the strategy from the
            atom count: ``n_atoms < low_max_atoms -> "low"``;
            ``n_atoms >= high_max_atoms -> "high"``; otherwise
            ``"medium"``.  When ``None`` we trust the explicit
            ``complexity`` argument.

        Returns
        -------
        str
            :data:`SolverStrategy.BASE` or :data:`SolverStrategy.W_FLOW`.
        """
        # Resolve the canonical complexity from n_atoms when given.
        derived = None
        if n_atoms is not None:
            if int(n_atoms) < self.low_max_atoms:
                derived = "low"
            elif int(n_atoms) >= self.high_max_atoms:
                derived = "high"
            else:
                derived = "medium"
        # The explicit ``complexity`` argument wins when supplied.
        eff = str(complexity).lower()
        if derived is not None and eff == "auto":
            eff = derived
        if eff == "low":
            return SolverStrategy.BASE
        if eff == "high":
            return SolverStrategy.W_FLOW
        if eff == "medium":
            # Conservative default — base solver on CPU skeleton.
            return SolverStrategy.BASE
        if eff == "auto":
            raise ValueError(
                "select_strategy(complexity='auto') requires n_atoms."
            )
        raise ValueError(
            f"Unknown complexity {complexity!r}; expected 'low', 'medium', 'high', or 'auto'."
        )

    def select_solver(
        self,
        complexity: str,
        *,
        n_atoms: Optional[int] = None,
    ) -> Any:
        """Return the solver object for the given complexity descriptor.

        Parameters
        ----------
        complexity : str
            ``"low"``, ``"medium"``, ``"high"``, or ``"auto"``.
        n_atoms : int, optional
            Atom count — required when ``complexity == 'auto'``.

        Returns
        -------
        ``ODESolver``-like or :class:`WaveformRelaxationSolver`
            The chosen solver.  The caller invokes
            ``solver.sample(...)`` or ``solver.solve(...)``.
        """
        strategy = self.select_strategy(complexity, n_atoms=n_atoms)
        if strategy == SolverStrategy.BASE:
            return self.base_solver
        return self.w_flow_solver

    def build_base_solver(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        n_steps: int = 40,
        method: str = "midpoint",
    ) -> Any:
        """Build a fresh legacy :class:`ODESolver` for the base path.

        Useful when :meth:`__init__` was called with
        ``base_solver=None`` and we need to materialise a base
        solver for the selector.  Also binds it to ``self`` for
        subsequent :meth:`select_solver` calls.
        """
        if velocity_fn is None:
            raise ValueError(
                "build_base_solver requires a velocity_fn to construct ODESolver."
            )
        self.base_solver = ODESolver(
            velocity_model=velocity_fn,
            method=str(method),
            n_steps=int(n_steps),
        )
        return self.base_solver


# ---------------------------------------------------------------------------
# Module-level benchmark helper
# ---------------------------------------------------------------------------
def benchmark_w_flow_cfm(
    base_solver: Any,
    w_flow_solver: WaveformRelaxationSolver,
    n_runs: int = DEFAULT_N_RUNS,
    *,
    z0: Optional[torch.Tensor] = None,
    t: float = 1.0,
    velocity_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    n_steps_base: int = 40,
    method_base: str = "midpoint",
    tol_accuracy: float = 1e-2,
) -> Dict[str, Any]:
    """Head-to-head wall-time + accuracy benchmark for W-Flow vs base.

    Thin wrapper around :func:`flow_matching_lipman.w_flow.compare_solvers`
    that re-shapes the report into the Integration 3 contract:
    callers get back a dict with ``strategy``, ``w_total_time_s``,
    ``base_total_time_s``, ``speedup``, ``max_abs_diff``,
    ``within_tol``, ``convergence``, ``n_runs``, ``z0_shape``, and
    ``n_windows``.

    Parameters
    ----------
    base_solver : ``ODESolver``-like
        Legacy solver.  Must expose ``.sample(z0, n_steps=N,
        method=M)`` or be callable as ``base_solver(z0)``.
    w_flow_solver : :class:`WaveformRelaxationSolver`
        The W-Flow solver under test.
    n_runs : int, default 10
        Number of timing runs for both solvers.  The accuracy
        comparison uses a single deterministic run (not timed).
    z0 : ``(B, N, 3)`` tensor, optional
        Initial state.  When ``None`` we generate the canonical
        ``(1, 8, 3)`` noise tensor with ``torch.manual_seed(0)``.
    t : float, default 1.0
        Final time.  Forwarded to ``w_flow_solver.solve``.
    velocity_fn : callable, optional
        Velocity field for ``w_flow_solver``.  When ``None`` we
        fall back to a self-contained ``v(x, t) = x`` field so the
        benchmark is self-contained.
    n_steps_base : int, default 40
        ``n_steps`` for the base solver.  Must be ``>=
        w_flow_solver.n_windows * w_flow_solver.n_substeps_per_window``
        for a fair comparison.
    method_base : str, default "midpoint"
        Inner method for the base solver.
    tol_accuracy : float, default 1e-2
        Max-abs-diff tolerance for ``within_tol``.  Loose default
        because the benchmark exercises the *head-to-head* timing
        + accuracy trade-off — production accuracy is verified by
        :mod:`molmetal.tests.test_w_flow_cfm_integration`.

    Returns
    -------
    dict
        ``{
            "strategy": "w_flow",
            "w_total_time_s": float,
            "base_total_time_s": float,
            "speedup": float,           # lower = faster W-Flow
            "max_abs_diff": float,
            "within_tol": bool,
            "convergence": dict,        # w_flow_solver.convergence_report()
            "n_runs": int,
            "z0_shape": tuple,
            "n_windows": int,
        }``.
    """
    report = _w_flow_compare_solvers(
        base_solver=base_solver,
        w_solver=w_flow_solver,
        n_runs=int(n_runs),
        z0=z0.clone() if z0 is not None else None,
        t=float(t),
        velocity_fn=velocity_fn,
        n_steps_base=int(n_steps_base),
        method_base=str(method_base),
        tol_accuracy=float(tol_accuracy),
    )
    return {
        "strategy": SolverStrategy.W_FLOW,
        "w_total_time_s": float(report["w_total_time_s"]),
        "base_total_time_s": float(report["base_total_time_s"]),
        "speedup": float(report["speedup"]),
        "max_abs_diff": float(report["max_abs_diff"]),
        "within_tol": bool(report["within_tol"]),
        "convergence": dict(report["convergence"]),
        "n_runs": int(report["n_runs"]),
        "z0_shape": tuple(report["z0_shape"]),
        "n_windows": int(w_flow_solver.n_windows),
    }


# ---------------------------------------------------------------------------
# Trivial velocity field (used by ``wall_time_speedup`` when no model wired)
# ---------------------------------------------------------------------------
def _default_linear_velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Trivial velocity field used as a self-contained fallback.

    ``v(x, t) = x`` gives the analytic solution
    ``x(t) = x_0 * exp(t)``.  Used by :meth:`WFlowCFMSolver.wall_time_speedup`
    and :func:`benchmark_w_flow_cfm` when no external velocity_fn
    is supplied — keeps the test suite self-contained without
    dragging the production EGNN in.

    The ``t`` argument is accepted to satisfy the ModelWrapper
    contract (a 0-d scalar tensor) but otherwise ignored because
    the field is autonomous.
    """
    return x.clone()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_default_w_flow_cfm_solver(
    base_cfm_solver: Any = None,
    *,
    n_windows: int = DEFAULT_N_WINDOWS,
    n_steps_base: int = 40,
    method_base: str = "midpoint",
) -> WFlowCFMSolver:
    """Factory used by tests + scripts.

    Equivalent to :class:`WFlowCFMSolver(...)` with the canonical
    R18 P2-1 defaults.  Exists so test files can declare a fixture
    in one place.
    """
    return WFlowCFMSolver(
        base_cfm_solver=base_cfm_solver,
        w_flow_solver=WaveformRelaxationSolver(
            n_windows=int(n_windows),
            n_iters_per_window=int(DEFAULT_N_ITERS_PER_WINDOW),
            tol=1e-6,
            method=str(method_base),
            n_substeps_per_window=10,
        ),
        n_windows=int(n_windows),
        n_steps_base=int(n_steps_base),
        method_base=str(method_base),
    )


__all__ = [
    "WFlowCFMSolver",
    "WFlowCFMSelector",
    "SolverStrategy",
    "benchmark_w_flow_cfm",
    "make_default_w_flow_cfm_solver",
    "COMPLEXITY_THRESHOLDS",
    # Re-exports for convenience
    "DEFAULT_N_WINDOWS",
    "DEFAULT_N_ITERS_PER_WINDOW",
    "DEFAULT_N_RUNS",
]