"""Integration 3 — W-Flow -> CFM ODE solver integration smoke test.

================================================================
Mol-Metal R18 — Integration 3 verify
================================================================
Usage::

    python molmetal/scripts/verify_w_flow_cfm.py --smoke
    python molmetal/scripts/verify_w_flow_cfm.py --n-runs 10 --n-windows 8

Runs an end-to-end comparison of :class:`WFlowCFMSolver` against
the legacy R17 midpoint :class:`ODESolver` and prints the
wall-time + accuracy.  Designed for CI: no torch model
dependency, finishes in <30 s on a single CPU thread (smoke mode
is <5 s).

Output is one section with ``strategy``, ``w_total_time_s``,
``base_total_time_s``, ``speedup``, ``max_abs_diff``,
``within_tol``, and ``n_windows``.

Constraints
-----------
- CPU-only (``torch.no_grad`` throughout).
- Does **not** modify ``molmetal/adapters/flow_matching_lipman/ode_solver.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make ``molmetal`` importable when run as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman._reference.solver import (  # noqa: E402
    ODESolver,
)
from molmetal.adapters.flow_matching_lipman.w_flow import (  # noqa: E402
    _default_linear_velocity_fn,
)
from molmetal.adapters.flow_matching_lipman.w_flow_cfm_integration import (  # noqa: E402
    SolverStrategy,
    WFlowCFMSolver,
    benchmark_w_flow_cfm,
)


def _run_one(
    n_runs: int,
    n_windows: int,
    n_steps_base: int,
    t: float,
    smoke: bool,
) -> None:
    """Run one benchmark + print a single result block."""
    print(
        f"\n=== W-Flow vs base | n_windows={n_windows} "
        f"| n_steps_base={n_steps_base} | n_runs={n_runs} | t={t} ==="
    )
    # Self-contained trivial velocity field — no Lipman adapter needed.
    velocity_fn = _default_linear_velocity_fn
    base_solver = ODESolver(
        velocity_model=velocity_fn,
        method="midpoint",
        n_steps=n_steps_base,
    )
    w_solver = WFlowCFMSolver(
        base_cfm_solver=base_solver,
        n_windows=n_windows,
        n_steps_base=n_steps_base,
        method_base="midpoint",
    )
    # Deterministic (1, 8, 3) input.
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, 8, 3, generator=g)

    if smoke:
        # Smoke mode: tiny benchmark + manual accuracy check.
        speedup = w_solver.wall_time_speedup(
            n_runs=n_runs, z0=z0.clone(), t=t, velocity_fn=velocity_fn,
        )
        with torch.no_grad():
            w_state = w_solver.solve(velocity_fn, z0.clone(), (0.0, t))
            base_state = w_solver.solve_base(
                velocity_fn, z0.clone(),
                n_steps=n_steps_base, method="midpoint",
                t_span=(0.0, t),
            )
        max_abs_diff = float((w_state - base_state).abs().max().item())
        print(f"  strategy       : {SolverStrategy.W_FLOW}")
        print(f"  speedup (W/B)  : {speedup:.4f}  (lower = faster W-Flow)")
        print(f"  max_abs_diff   : {max_abs_diff:.6e}")
        print(f"  within tol (0.1): {bool(max_abs_diff < 0.1)}")
    else:
        # Full benchmark via ``benchmark_w_flow_cfm``.
        report = benchmark_w_flow_cfm(
            base_solver=base_solver,
            w_flow_solver=w_solver.w_flow_solver,
            n_runs=n_runs,
            z0=z0.clone(),
            t=t,
            velocity_fn=velocity_fn,
            n_steps_base=n_steps_base,
            method_base="midpoint",
            tol_accuracy=1e-2,
        )
        for key in (
            "strategy", "w_total_time_s", "base_total_time_s", "speedup",
            "max_abs_diff", "within_tol", "n_windows", "n_runs", "z0_shape",
        ):
            val = report[key]
            if isinstance(val, float):
                print(f"  {key:14s}: {val:.6f}")
            else:
                print(f"  {key:14s}: {val}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Integration 3 W-Flow -> CFM ODE solver "
            "replacement (R18 P2-1)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the canonical smoke benchmark (fast path).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of timing runs (default 5).",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=4,
        help="W-Flow n_windows (default 4).",
    )
    parser.add_argument(
        "--n-steps-base",
        type=int,
        default=40,
        help="Base solver n_steps (default 40).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Integration endpoint (default 1.0).",
    )
    args = parser.parse_args()

    # Smoke mode: tiny defaults for fast CI feedback.
    if args.smoke:
        n_runs = 2
        n_windows = 2
        n_steps_base = 20
    else:
        n_runs = args.n_runs
        n_windows = args.n_windows
        n_steps_base = args.n_steps_base

    print("Integration 3 — W-Flow -> CFM ODE solver verify")
    print(f"  project root   : {PROJECT_ROOT}")
    print(f"  torch version  : {torch.__version__}")
    _run_one(
        n_runs=int(n_runs),
        n_windows=int(n_windows),
        n_steps_base=int(n_steps_base),
        t=float(args.t),
        smoke=bool(args.smoke),
    )
    print("\nAll smoke checks completed (CPU only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
