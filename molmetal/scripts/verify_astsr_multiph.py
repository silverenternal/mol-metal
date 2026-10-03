"""Verify ASTSR + Multi-PH joint Hilbert integration (Wf-14 phase-3, 2026-09-26).

Run::

    uv run --no-sync python molmetal/scripts/verify_astsr_multiph.py --smoke

Exercises the joint Hilbert regression layer end-to-end on a tiny
synthetic design matrix and prints the head-to-head wall-time + R²
comparison.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    """Smoke verification entry point — return 0 on pass."""
    parser = argparse.ArgumentParser(
        description="Smoke-check ASTSR + Multi-PH joint Hilbert integration.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke check.",
    )
    parser.add_argument("--n-seeds", type=int, default=2)
    args = parser.parse_args()

    try:
        import numpy as np

        from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
            JointFeatureConcatenator,
            JointHilbertOptimizer,
            JointVsSeparateComparator,
        )
        from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
            LambdaASTSRHilbert,
        )
    except Exception as exc:
        print(
            f"ASTSR + Multi-PH joint Hilbert: imports FAIL "
            f"({type(exc).__name__}:{exc})"
        )
        return 1

    n_fail, n_total = 0, 0

    # Toy joint design: (astr_X, multi_ph_X, y)
    rng = np.random.default_rng(42)
    n, d_astr, n_bins = 24, 4, 3
    astr_X = rng.standard_normal((n, d_astr))
    mph_X = rng.standard_normal((n, n_bins, n_bins))
    w = rng.standard_normal(d_astr)
    y = astr_X @ w + 0.01 * rng.standard_normal(n)

    # 1. Joint Hilbert fit
    n_total += 1
    try:
        opt = JointHilbertOptimizer(hilbert_dim=8, normalize=False)
        res = opt.fit(astr_X, mph_X, y)
        assert math.isfinite(res["r2"]) and math.isfinite(res["wall_s"])
        print(
            f"  Joint Hilbert fit      wall={res['wall_s']:.4f}s  "
            f"R²={res['r2']:.4f}  join_dim={res['join_dim']}"
        )
    except Exception as exc:
        print(f"  Joint Hilbert fit FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 2. Joint predict
    n_total += 1
    try:
        y_hat = opt.predict(astr_X, mph_X)
        assert y_hat.shape == (n,) and np.all(np.isfinite(y_hat))
        print(f"  Joint predict          shape={y_hat.shape} (finite)")
    except Exception as exc:
        print(f"  Joint predict FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 3. Concatenator shape + normalize
    n_total += 1
    try:
        cat = JointFeatureConcatenator()
        joined = cat.concatenate(astr_X, mph_X)
        assert joined.shape == (n, d_astr + n_bins**2)
        normed = cat.fit_normalize(joined)
        assert np.all(np.isfinite(normed))
        print(
            f"  Concatenator           joined_shape={joined.shape}  "
            f"normalized_mean={np.mean(normed, axis=0).max():.2e}"
        )
    except Exception as exc:
        print(f"  Concatenator FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 4. Joint vs separate comparator
    n_total += 1
    try:

        def _factory(population_dim: int = 4, seed: int = 0) -> LambdaASTSRHilbert:
            return LambdaASTSRHilbert(
                population_dim=int(population_dim),
                lr_primal=0.05,
                lr_dual=0.05,
                n_iters=80,
                seed=int(seed),
            )

        comp = JointVsSeparateComparator(
            separate_hilbert=_factory,
            joint_hilbert=opt,
        )
        out = comp.compare((astr_X, mph_X), y, n_seeds=int(args.n_seeds))
        print(
            f"  Joint   (comparator)   wall={out['joint_wall_s']:.4f}s  "
            f"R²={out['joint_r2']:.4f}"
        )
        print(
            f"  Separate (comparator)  wall={out['separate_wall_s']:.4f}s  "
            f"R²={out['separate_r2']:.4f}"
        )
        print(f"  speedup={out['speedup']:.2f}x  (n_seeds={out['n_seeds']})")
    except Exception as exc:
        print(f"  JointVsSeparate FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    print(
        f"ASTSR + Multi-PH joint Hilbert: imports OK, smoke " f"{n_fail}/{n_total} fail"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
