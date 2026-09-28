"""Verify Hilbert ↔ LambdaASTSR integration (TODO-34 §L-R18-P1-5, 2026-09-26).

Run::

    uv run --no-sync python molmetal/scripts/verify_hilbert_astsr.py --smoke
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU smoke check.")
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    try:
        import numpy as np

        from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
            HilbertASTSRComposite,
            HilbertEvolutionaryComparator,
            benchmark_hilbert_vs_evolutionary,
            make_hilbert_astsr_reward_channel,
        )
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR
    except Exception as exc:
        print(
            f"Hilbert ↔ ASTSR integration: imports FAIL "
            f"({type(exc).__name__}:{exc})"
        )
        return 1

    n_fail, n_total = 0, 0

    # Build a 30-row toy design matrix.
    rng = np.random.default_rng(42)
    dim = 8
    X = rng.standard_normal((30, dim))
    w_true = rng.standard_normal(dim)
    y = X @ w_true + 0.01 * rng.standard_normal(30)
    fc = tuple(f"f{i}" for i in range(dim))

    # 1. Composite fit (Hilbert arm)
    n_total += 1
    try:
        composite = HilbertASTSRComposite(
            feature_columns=fc, population_dim=dim, use_hilbert=True,
        )
        res_h = composite.fit(X, y)
        assert res_h["backend"] == "hilbert" and math.isfinite(res_h["r2"])
        print(f"  Hilbert (composite.fit)   wall={res_h['wall_s']:.4f}s  R²={res_h['r2']:.4f}")
    except Exception as exc:
        print(f"  composite Hilbert fit FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 2. Switch + evolutionary fit
    n_total += 1
    try:
        composite.switch_to_hilbert(False)
        res_e = composite.fit(X, y)
        assert res_e["backend"] == "evolutionary" and math.isfinite(res_e["r2"])
        print(f"  Evolutionary (composite)  wall={res_e['wall_s']:.4f}s  R²={res_e['r2']:.4f}")
    except Exception as exc:
        print(f"  composite Evolutionary fit FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 3. Head-to-head comparator
    n_total += 1
    try:
        comparator = HilbertEvolutionaryComparator(
            hilbert_astsr=HilbertASTSRComposite(
                feature_columns=fc, population_dim=dim,
            ),
            evolutionary_astsr=LambdaASTSR(feature_names=fc),
        )
        out = comparator.compare(X, y, n_seeds=int(args.n_seeds))
        print(
            f"  Hilbert      (comparator) wall={out['hilbert_wall_s']:.4f}s  "
            f"R²={out['hilbert_r2']:.4f}"
        )
        print(
            f"  Evolutionary (comparator) wall={out['evolutionary_wall_s']:.4f}s  "
            f"R²={out['evolutionary_r2']:.4f}"
        )
        print(f"  speedup={out['speedup']:.2f}x  (n_seeds={out['n_seeds']})")
    except Exception as exc:
        print(f"  comparator FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 4. Multi-run benchmark
    n_total += 1
    try:
        bench = benchmark_hilbert_vs_evolutionary(
            X, y, n_runs=int(args.n_runs),
            feature_columns=fc, population_dim=dim,
        )
        print(
            f"  benchmark (n_runs={bench['n_runs']})  "
            f"hilbert_mean={bench['hilbert_wall_s_mean']:.4f}s±"
            f"{bench['hilbert_wall_s_std']:.4f}  "
            f"evol_mean={bench['evolutionary_wall_s_mean']:.4f}s±"
            f"{bench['evolutionary_wall_s_std']:.4f}"
        )
    except Exception as exc:
        print(f"  benchmark FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    # 5. Reward-channel smoke
    n_total += 1
    try:
        composite2 = HilbertASTSRComposite(
            feature_columns=fc, population_dim=dim, use_hilbert=True,
        )
        composite2.fit(X, y)
        closure = make_hilbert_astsr_reward_channel(composite2, enabled=True)
        row = {fn: float(X[0, j]) for j, fn in enumerate(fc)}
        v = closure(row)
        assert math.isfinite(v), f"non-finite reward: {v}"
    except Exception as exc:
        print(f"  reward-channel FAIL ({type(exc).__name__}:{exc})")
        n_fail += 1

    print(
        f"Hilbert ↔ ASTSR integration: imports OK, smoke "
        f"{n_fail}/{n_total} fail"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
