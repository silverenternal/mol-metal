"""Smoke verification for the score-repellent Monte Carlo sampler.

============================================================
Score-Repellent MC smoke (R18 P2-5, ICML 2026)
============================================================
Runs a small end-to-end comparison of the two exploration-noise
paths used by the MLC MCTS root:

    (a) Dirichlet replacement  -- the R18 default (variance-controlled)
    (b) Vanilla Dirichlet      -- the AlphaZero baseline

The script prints a one-page variance comparison so reviewers can
verify the R18 variance-reduction claim at a glance.

Run via::

    uv run --no-sync python molmetal/scripts/verify_score_repellent_mc.py --smoke

Optional flags
--------------
``--n-samples N``       number of draws per arm  (default 256)
``--dim D``             action-space dimension   (default 8)
``--alpha A``           Dirichlet concentration  (default 1.0)
``--n-iters K``         chain length per draw    (default 32)
``--step-size H``       score-repellent step eta (default 0.1)
``--seed S``            reserved; default 2025

References
----------
* Burch & Bardsley 2022, "Score-based Repulsive MCMC".
* Silver 2016, "Mastering the game of Go", *Nature* 529.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap sys.path so ``python molmetal/scripts/verify_*.py --help``
# can resolve the ``molmetal`` package without uv-installed env.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

# Lazy import so the script works whether or not the package has
# been installed into the active environment (matches the
# convention in the rest of the ``scripts/`` directory).
try:
    from molmetal_lam.search_alg.score_repellent_mc import (
        DEFAULT_N_ITERS,
        DEFAULT_STEP_SIZE,
        DirichletReplacement,
        ScoreRepellentSampler,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.score_repellent_mc import (  # type: ignore
        DEFAULT_N_ITERS,
        DEFAULT_STEP_SIZE,
        DirichletReplacement,
        ScoreRepellentSampler,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _running_variance(samples: torch.Tensor) -> float:
    """Return the variance of the *per-arm mean* across draws.

    This is the variance of the empirical mean estimator for one
    arm, which is what the PUCT bonus consumes.  Lower variance
    == lower-variance exploration == better statistical
    efficiency of the MCTS rollout (Burch & Bardsley 2022 §4.1).
    """
    arr = samples.detach().cpu().numpy()
    return float(arr.mean(axis=0).var(ddof=1))


def _marginal_stats(samples: torch.Tensor) -> tuple:
    """Return ``(mean_per_arm, std_per_arm)`` arrays."""
    arr = samples.detach().cpu().numpy()
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score-repellent MC smoke verification (R18 P2-5).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the smoke harness end-to-end (default behaviour).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=256,
        help="Number of draws per arm (default 256).",
    )
    parser.add_argument(
        "--dim", type=int, default=8,
        help="Action-space dimension (default 8).",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Dirichlet concentration alpha (default 1.0).",
    )
    parser.add_argument(
        "--n-iters", type=int, default=DEFAULT_N_ITERS,
        help="Chain length per draw (default DEFAULT_N_ITERS).",
    )
    parser.add_argument(
        "--step-size", type=float, default=DEFAULT_STEP_SIZE,
        help="Score-repellent step eta (default DEFAULT_STEP_SIZE).",
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Master RNG seed (default 2025).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rng = np.random.default_rng(int(args.seed))
    print("=" * 72)
    print("Score-Repellent MC smoke harness (R18 P2-5)")
    print("=" * 72)
    print(f"  n_samples = {args.n_samples}")
    print(f"  dim       = {args.dim}")
    print(f"  alpha     = {args.alpha}")
    print(f"  n_iters   = {args.n_iters}")
    print(f"  step_size = {args.step_size}")
    print(f"  seed      = {args.seed}")
    print()

    # ----- (a) Vanilla Dirichlet baseline -------------------------------
    dirichlet = DirichletReplacement(
        alpha=args.alpha, score_repellent=False, seed=int(args.seed),
    )
    eta_d = dirichlet.sample(n_samples=args.n_samples, dim=args.dim)
    var_d = _running_variance(eta_d)
    mean_d, std_d = _marginal_stats(eta_d)
    print("[Dirichlet baseline] (AlphaZero-style, score_repellent=False)")
    print(f"  per-arm mean    = {np.round(mean_d, 4).tolist()}")
    print(f"  per-arm std     = {np.round(std_d, 4).tolist()}")
    print(f"  variance(mean)  = {var_d:.6f}")
    print()

    # ----- (b) Score-repellent replacement -------------------------------
    rep = DirichletReplacement(
        alpha=args.alpha, score_repellent=True,
        step_size=args.step_size, n_iters=args.n_iters,
        seed=int(args.seed),
    )
    eta_sr = rep.sample(n_samples=args.n_samples, dim=args.dim)
    var_sr = _running_variance(eta_sr)
    mean_sr, std_sr = _marginal_stats(eta_sr)
    print("[Score-repellent]  (R18 P2-5, score_repellent=True)")
    print(f"  per-arm mean    = {np.round(mean_sr, 4).tolist()}")
    print(f"  per-arm std     = {np.round(std_sr, 4).tolist()}")
    print(f"  variance(mean)  = {var_sr:.6f}")
    print()

    # ----- (c) Theory: alpha = 2 eta L ----------------------------------
    sampler = ScoreRepellentSampler(
        step_size=args.step_size,
        score_lipschitz=1.0,
        seed=int(args.seed),
    )
    alpha_theory = sampler.variance_bound()
    print(f"[Theory] Burch & Bardsley 2022 alpha = 2 eta L = {alpha_theory:.4f}")
    print()

    # ----- (d) Variance comparison --------------------------------------
    print("-" * 72)
    print("Variance comparison:")
    if var_d > 0.0:
        ratio = var_sr / var_d
        print(f"  variance(score-repellent) / variance(Dirichlet) = {ratio:.4f}")
        print(
            f"  variance reduction (1 - ratio) = {(1.0 - ratio):.4f}"
        )
    else:
        print("  WARNING: Dirichlet variance is zero (degenerate alpha).")
    print()
    print("PASS: smoke harness complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
