"""Verify Hilbert-space primal-dual optimisation skeleton (TODO-34, 2026-09-26).

Smoke verification that imports the new
:mod:`molmetal.molmetal_lam.reward.hilbert_primal_dual` module, runs
the full pipeline (Hilbert vs evolutionary wall-time + R² comparison
on a 30-row toy dataset) on CPU, and prints a one-line summary.

Run::

    uv run --no-sync python molmetal/scripts/verify_hilbert_primal_dual.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    """Smoke verification entry point — return 0 on pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 30-second CPU smoke verification.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of seeds for the Hilbert-vs-evolutionary comparison.",
    )
    args = parser.parse_args()

    # ---- Imports --------------------------------------------------------
    try:
        import numpy as np

        from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
            HilbertSpace,
            LambdaASTSRHilbert,
            PrimalDualOptimizer,
            primal_dual_convergence_rate,
        )
    except Exception as exc:
        print(
            f"Hilbert primal-dual: imports FAIL ({type(exc).__name__}:{exc})"
        )
        return 1

    n_fail = 0
    n_total = 0

    # ---- 1. Hilbert space smoke ----------------------------------------
    n_total += 1
    try:
        h = HilbertSpace(dim=16)
        assert h.basis_matrix.shape == (h.n_quad, 16)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(h.n_quad)
        g = rng.standard_normal(h.n_quad)
        ip = h.inner_product(f, g)
        assert np.isfinite(ip)
    except Exception:
        n_fail += 1

    # ---- 2. PD optimiser single-step ----------------------------------
    n_total += 1
    try:
        opt = PrimalDualOptimizer(lr_primal=0.01, lr_dual=0.01, n_iters=10)
        x = np.zeros(4)
        y = np.zeros(4)

        def _obj(xk):
            return float(np.sum(xk ** 2)), 2.0 * xk

        x_new, y_new = opt.step(x, y, _obj)
        assert x_new.shape == (4,) and y_new.shape == (4,)
    except Exception:
        n_fail += 1

    # ---- 3. LambdaASTSRHilbert compare with evolutionary ---------------
    n_total += 1
    try:
        # 30-row toy linear problem; deterministic seed for reproducibility.
        rng = np.random.default_rng(42)
        dim = 8
        X = rng.standard_normal((30, dim))
        w_true = rng.standard_normal(dim)
        y = X @ w_true + 0.01 * rng.standard_normal(30)

        hilbert = LambdaASTSRHilbert(
            population_dim=dim, lr_primal=0.01, lr_dual=0.01,
            n_iters=20, seed=0,
        )
        result = hilbert.compare_with_evolutionary(
            X, y, n_seeds=int(args.n_seeds),
        )
        # Print the comparison table.
        print(
            f"  Hilbert      wall={result['hilbert_wall_s']:.4f}s  "
            f"R²={result['hilbert_r2']:.4f}"
        )
        print(
            f"  Evolutionary wall={result['evolutionary_wall_s']:.4f}s  "
            f"R²={result['evolutionary_r2']:.4f}"
        )
        print(f"  speedup={result['speedup']:.2f}x")
        # Sanity check: both R²s must be in [-1, 1].
        if not (-1.0 <= result["hilbert_r2"] <= 1.0):
            n_fail += 1
        if not (-1.0 <= result["evolutionary_r2"] <= 1.0):
            n_fail += 1
    except Exception as exc:
        print(f"  comparison FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # ---- 4. Convergence-rate check -------------------------------------
    n_total += 1
    try:
        rate_10 = primal_dual_convergence_rate(10)
        rate_100 = primal_dual_convergence_rate(100)
        if rate_10 <= rate_100:
            n_fail += 1
    except Exception:
        n_fail += 1

    # ---- Summary --------------------------------------------------------
    print(
        f"Hilbert primal-dual: imports OK, smoke {n_fail}/{n_total} fail"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
