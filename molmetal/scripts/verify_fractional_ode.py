"""R18 P1-5 — fractional-order ODE CPU skeleton smoke test.

Usage::

    python molmetal/scripts/verify_fractional_ode.py --smoke
    python molmetal/scripts/verify_fractional_ode.py --alpha 0.7 --steps 60

Runs an end-to-end comparison of the :class:`FractionalCFM` wrapper
against the integer-order (linear) base CFM and prints the wall-time
and accuracy.  Designed for CI: no torch dependency, finishes in
<1 s on a single CPU thread.

Output is one section per alpha value tested (default: ``[0.5,
0.9]``) with mean wall-time over ``n_runs`` runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make ``molmetal`` importable when run as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman.fractional_ode import (
    CaputoDerivative,
    FractionalCFM,
    FractionalODESolver,
    fractional_convergence_rate,
)


def _linear_base_cfm(z0: np.ndarray, t: float) -> np.ndarray:
    """z(t) = z0 + t · v, with a fixed drift."""
    v = np.array([1.0, 0.5, -0.3])
    return z0 + t * v


def _run_one(alpha: float, n_steps: int, n_runs: int, t: float) -> None:
    print(f"\n=== alpha = {alpha:.2f} | n_steps = {n_steps} | t = {t} ===")
    cfm = FractionalCFM(
        base_cfm=_linear_base_cfm,
        alpha=alpha,
    )
    # Override the default ``n_steps=100`` on the underlying solver so the
    # smoke test runs faster.
    cfm._solver = FractionalODESolver(alpha=alpha, n_steps=n_steps)
    z0 = np.array([1.0, -0.5, 0.7])
    out = cfm.compare_with_base(z0, t=t, n_runs=n_runs)
    print(
        f"  fractional wall-time (mean over {n_runs} runs): "
        f"{out['fractional']['wall_time'] * 1000:.3f} ms"
    )
    print(
        f"  base wall-time      (mean over {n_runs} runs): "
        f"{out['base']['wall_time'] * 1000:.3f} ms"
    )
    print(f"  fractional endpoint norm: {out['fractional']['endpoint_norm']:.6f}")
    print(f"  base      endpoint norm: {out['base']['endpoint_norm']:.6f}")
    print(f"  abs diff  : {out['accuracy']['abs_diff']:.6e}")
    print(f"  rel diff  : {out['accuracy']['rel_diff']:.6e}")
    # Convergence-rate diagnostic at k = n_steps.
    rate = fractional_convergence_rate(alpha=alpha, n_iters=n_steps)
    print(f"  convergence rate O(k^-alpha) at k={n_steps}: {rate:.6e}")


def _run_caputo_sanity() -> None:
    """Quick Caputo derivative analytical sanity check."""
    print("\n=== Caputo derivative sanity ===")
    cap = CaputoDerivative(alpha=0.5, n_steps=64)
    t = np.linspace(0.0, 1.0, 65)
    f_const = np.full_like(t, 3.7)
    d_const = cap.compute(f_const, t)
    print(f"  Caputo(constant = 3.7) max abs: {np.max(np.abs(d_const)):.3e}")
    f_lin = 1.0 + 2.0 * t
    d_lin = cap.compute(f_lin, t)
    print(f"  Caputo(1 + 2t)  max abs (tail): {np.max(np.abs(d_lin[10:])):.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the fractional-order ODE CPU skeleton "
            "(R18 P1-5 — 数学学报 2026)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the canonical smoke test (alpha in {0.5, 0.9}, n_steps=20).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Single alpha to test.  Defaults to the smoke-test set.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override n_steps (default depends on --smoke).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timing runs (default 5).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Integration endpoint (default 1.0).",
    )
    args = parser.parse_args()

    if args.smoke or (args.alpha is None and args.steps is None):
        alphas = [0.5, 0.9]
        steps = args.steps if args.steps is not None else 20
    else:
        alphas = [args.alpha]
        steps = args.steps if args.steps is not None else 50

    print("R18 P1-5 — fractional ODE CPU skeleton verify")
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  numpy version: {np.__version__}")
    _run_caputo_sanity()
    for alpha in alphas:
        _run_one(
            alpha=float(alpha),
            n_steps=int(steps),
            n_runs=int(args.runs),
            t=float(args.t),
        )
    print("\nAll smoke checks completed (CPU only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
