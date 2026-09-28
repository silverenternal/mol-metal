#!/usr/bin/env python
"""Smoke harness for the variance-controlled PUCT integration (TODO-30 P5.1).

================================================================
Compare Dirichlet vs Score-Repellent exploration noise on a
4-arm canonical PUCT benchmark.
================================================================
Usage::

    uv run --no-sync python -m molmetal.scripts.verify_variance_puct --smoke
    uv run --no-sync python -m molmetal.scripts.verify_variance_puct --n-samples 4096

Outputs the per-arm variance + KL divergence between the two
samplers and prints a PASS/FAIL line for the
"score-repellent variance ≤ Dirichlet variance" contract.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _run_smoke(n_samples: int) -> int:
    # Lazy import so this script is usable in environments where
    # torch / score_repellent_mc are not installed (CPU-only
    # smoke runs use only numpy).
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (
        DEFAULT_DIRICHLET_ALPHA,
        DEFAULT_SCORE_REPELLENT_ALPHA,
        DirichletVarianceComparator,
    )
    cmp = DirichletVarianceComparator(
        alpha=DEFAULT_DIRICHLET_ALPHA,
        score_repellent_alpha=DEFAULT_SCORE_REPELLENT_ALPHA,
        n_iters=50,
    )
    report = cmp.compare(n_samples=int(n_samples))
    # Pretty-print.
    print("=" * 64)
    print("Variance-Controlled PUCT — Dirichlet vs Score-Repellent")
    print("=" * 64)
    print(f"n_samples                : {report['n_samples']}")
    print(f"alpha (Dirichlet)        : {report['alpha_dirichlet']}")
    print(f"alpha (Score-Repellent)  : {report['alpha_score_repellent']}")
    print()
    print(f"per-arm var (Dirichlet)        : "
          f"{[round(v, 6) for v in report['per_arm_var_dirichlet']]}")
    print(f"per-arm var (Score-Repellent)  : "
          f"{[round(v, 6) for v in report['per_arm_var_score_repellent']]}")
    print()
    print(f"total var (Dirichlet)          : "
          f"{report['total_var_dirichlet']:.6f}")
    print(f"total var (Score-Repellent)    : "
          f"{report['total_var_score_repellent']:.6f}")
    print(f"KL(Dirichlet || Score-Rep)     : "
          f"{report['kl_divergence']:.6f}")
    # Contract: score-repellent variance ≤ Dirichlet variance
    # per arm (allow a tiny tolerance for floating-point noise).
    pass_contract = all(
        sr <= di + 1.0e-3
        for sr, di in zip(
            report["per_arm_var_score_repellent"],
            report["per_arm_var_dirichlet"],
        )
    )
    print()
    if pass_contract:
        print("[PASS] score-repellent variance <= Dirichlet variance (per-arm)")
        return 0
    print("[FAIL] score-repellent variance exceeds Dirichlet variance")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small smoke comparison (default n_samples=256).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1024,
        help="Number of independent draws per sampler (default 1024).",
    )
    args = parser.parse_args(argv)
    n = 256 if args.smoke else args.n_samples
    return _run_smoke(n)


if __name__ == "__main__":
    sys.exit(main())
