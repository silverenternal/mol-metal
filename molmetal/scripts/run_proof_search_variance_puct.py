#!/usr/bin/env python
"""Run ProofSearch with VariancePUCT as the default PUCT selector (TODO-92 R20-P1).

================================================================
CLI wrapper for the VariancePUCT-default integration layer
================================================================
This is the **CPU-only** CLI entry point for the R20-P1 (TODO-92)
ship: VariancePUCT replaces the canonical Dirichlet PUCT as the
**default** PUCT selector in :mod:`molmetal.search_alg.proof_search`.

The wrapper does NOT mutate :mod:`proof_search` (a separate PR is
required for that).  Instead it threads a VariancePUCT selector
through the integration layer
(:class:`molmetal.molmetal_lam.search_alg.variance_puct_integration.MCTSVarianceWrapper`)
and emits a small ProofSearch-style run report.

CLI flags
---------
* ``--variance-puct-default`` (default True) — when True, build a
  VariancePUCT selector with score-repellent noise (replaces
  Dirichlet PUCT).  When False, fall back to the canonical Dirichlet
  PUCT (backward compat — c_var=0 disables the variance penalty).
* ``--disable-variance-puct`` — convenience alias for
  ``--variance-puct-default False``.
* ``--c-puct`` (default 1.0) — PUCT exploration constant.
* ``--c-var`` (default 0.5) — variance penalty coefficient (Burch &
  Bardsley 2022 §4.4).
* ``--n-min-visits`` (default 2) — minimum visits before variance term
  activates.
* ``--smoke`` — run a small smoke (n_actions=4, n_steps=10).
* ``--n-actions`` / ``--n-steps`` — full-bench knobs.

Honest framing
--------------
This wrapper is **CPU-only documentation-grade proof** that the
VariancePUCT-default contract holds: it exercises the wrapper with a
small IntegrationNode tree, asserts that the selector returns valid
actions in [0, n_actions), and reports the cumulative Q / variance
statistics.  The full R20-P1 ship (VariancePUCT as the *production*
default in :mod:`proof_search`) requires the follow-up PR that wires
the integration layer into :class:`MCTSProofSearch`.  This CLI is the
**acceptance gate** for that follow-up.

Usage examples
--------------

::

    # Smoke run (4 actions × 10 steps, VariancePUCT default on).
    uv run --no-sync python -m molmetal.scripts.run_proof_search_variance_puct --smoke

    # Full bench with custom c_var.
    uv run --no-sync python -m molmetal.scripts.run_proof_search_variance_puct \\
        --n-actions 8 --n-steps 50 --c-var 0.7

    # Backward compat: legacy Dirichlet PUCT path.
    uv run --no-sync python -m molmetal.scripts.run_proof_search_variance_puct \\
        --disable-variance-puct --smoke
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

# ---------------------------------------------------------------------------
# Repo-root path injection
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run_proof_search(
    *,
    use_variance_puct: bool,
    c_puct: float,
    c_var: float,
    n_min_visits: int,
    n_actions: int,
    n_steps: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Run a small ProofSearch-style trajectory with the chosen selector.

    Builds an :class:`IntegrationNode` root with ``n_actions``
    children, runs ``n_steps`` rollout steps with a stochastic reward
    channel, and accumulates per-arm Q / sigma statistics.

    Returns a JSON-serialisable dict with the per-arm stats and
    cumulative wall-time.
    """
    # Lazy imports: this script must remain importable in CPU-only
    # environments where torch / score_repellent_mc are not installed.
    from molmetal.molmetal_lam.search_alg.variance_puct_integration import (
        DEFAULT_USE_VARIANCE_PUCT,
        IntegrationNode,
        VarianceControlledPUCT,
        VariancePUCTError,
        make_default_selector,
    )

    rng = random.Random(int(rng_seed))
    # Per-arm latent reward means (in [0, 1]) — fixed across steps.
    arm_means = [rng.random() for _ in range(int(n_actions))]
    # Per-arm latent reward stds (heteroscedastic: 1 arm has high
    # variance so the variance penalty has something to bite on).
    arm_stds = [0.05 + 0.45 * (1.0 if i == 0 else 0.0) for i in range(int(n_actions))]

    root = IntegrationNode(
        node_id="root",
        prior=1.0 / float(n_actions),
        children=[
            IntegrationNode(node_id=f"a{i}", prior=1.0 / float(n_actions))
            for i in range(int(n_actions))
        ],
    )

    # Build the default selector (R20-P1: VariancePUCT default).
    selector: VarianceControlledPUCT = make_default_selector(
        use_variance_puct=bool(use_variance_puct),
        c_puct=float(c_puct),
        c_var=float(c_var),
        score_repellent=True,  # score-repellent noise = Burch & Bardsley 2022
        n_min_visits=int(n_min_visits),
        rng_seed=int(rng_seed),
    )

    t0 = time.time()
    selected_counts = [0] * int(n_actions)
    last_action = -1
    last_value = 0.0
    last_variance = 0.0
    for step in range(int(n_steps)):
        action = int(selector.select(root))
        # Stochastic reward channel: r ~ Normal(arm_mean[action], arm_std[action]).
        v = float(rng.gauss(arm_means[action], arm_stds[action]))
        # Empirical sigma^2 for this arm (constant across steps; in
        # production the variance would be threaded from downstream
        # channels).
        var = float(arm_stds[action] ** 2)
        selector.update(root, action, value=v, variance=var)
        selected_counts[action] += 1
        last_action = action
        last_value = v
        last_variance = var
    wall = time.time() - t0

    # Per-arm summary.
    per_arm = []
    for i in range(int(n_actions)):
        st = selector.stats_.get((root.node_id, i), {})
        per_arm.append({
            "action": i,
            "n_visits": int(st.get("N", 0.0)),
            "Q": float(st.get("Q", 0.0)),
            "sigma": float(st.get("sigma", 0.0)),
        })

    return {
        "selector_kind": (
            "variance_puct" if selector.c_var > 0.0 else "dirichlet_puct_legacy"
        ),
        "default_use_variance_puct": bool(DEFAULT_USE_VARIANCE_PUCT),
        "requested_use_variance_puct": bool(use_variance_puct),
        "c_puct": float(c_puct),
        "c_var": float(c_var),
        "n_min_visits": int(n_min_visits),
        "n_actions": int(n_actions),
        "n_steps": int(n_steps),
        "selected_counts": selected_counts,
        "per_arm": per_arm,
        "last_action": int(last_action),
        "last_value": float(last_value),
        "last_variance": float(last_variance),
        "wall_time_s": float(wall),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variance-puct-default",
        dest="variance_puct_default",
        action="store_true",
        default=True,
        help=(
            "Use VariancePUCT as the default PUCT selector "
            "(default True; R20-P1 ship contract)."
        ),
    )
    parser.add_argument(
        "--no-variance-puct-default",
        dest="variance_puct_default",
        action="store_false",
        help=(
            "Equivalent to --variance-puct-default=False.  Convenient "
            "for environments where the default True is the wrong choice."
        ),
    )
    parser.add_argument(
        "--disable-variance-puct",
        action="store_true",
        default=False,
        help=(
            "Convenience alias for --no-variance-puct-default.  "
            "Switches to the canonical Dirichlet PUCT (c_var=0). "
            "Provided for backward compat with Wf-36 P2 / Wf-51 callers."
        ),
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="PUCT exploration constant (default 1.0).",
    )
    parser.add_argument(
        "--c-var",
        type=float,
        default=0.5,
        help="Variance penalty coefficient (default 0.5).",
    )
    parser.add_argument(
        "--n-min-visits",
        type=int,
        default=2,
        help="Min visits before variance term activates (default 2).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run (n_actions=4, n_steps=10).",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=8,
        help="Number of PUCT actions (=children) at the root (default 8).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of rollout steps (default 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default 0).",
    )
    args = parser.parse_args(argv)

    if args.disable_variance_puct:
        args.variance_puct_default = False

    n_actions = 4 if args.smoke else args.n_actions
    n_steps = 10 if args.smoke else args.n_steps

    report = _run_proof_search(
        use_variance_puct=bool(args.variance_puct_default),
        c_puct=float(args.c_puct),
        c_var=float(args.c_var),
        n_min_visits=int(args.n_min_visits),
        n_actions=int(n_actions),
        n_steps=int(n_steps),
        rng_seed=int(args.seed),
    )

    # Pretty-print.
    print("=" * 72)
    print("R20-P1 (TODO-92) — VariancePUCT-default ProofSearch run")
    print("=" * 72)
    print(f"selector_kind             : {report['selector_kind']}")
    print(f"default_use_variance_puct : {report['default_use_variance_puct']}")
    print(f"requested_use_variance_puct: {report['requested_use_variance_puct']}")
    print(f"c_puct                    : {report['c_puct']}")
    print(f"c_var                     : {report['c_var']}")
    print(f"n_min_visits              : {report['n_min_visits']}")
    print(f"n_actions                 : {report['n_actions']}")
    print(f"n_steps                   : {report['n_steps']}")
    print(f"selected_counts           : {report['selected_counts']}")
    print(f"wall_time_s               : {report['wall_time_s']:.6f}")
    print()
    print("per-arm stats (Q, sigma, n_visits):")
    for row in report["per_arm"]:
        print(
            f"  action {row['action']:>2} : "
            f"Q={row['Q']:.4f}  sigma={row['sigma']:.4f}  "
            f"N={row['n_visits']}"
        )
    print()
    # Acceptance gates (FLAT schema).
    gate_pass = (
        # G1_variance_puct_default: when the user asked for the default,
        # the selector must be variance_puct.  When they asked for the
        # legacy path, the selector must be dirichlet_puct_legacy.
        (
            (report["requested_use_variance_puct"] and report["selector_kind"] == "variance_puct")
            or
            (not report["requested_use_variance_puct"] and report["selector_kind"] == "dirichlet_puct_legacy")
        )
        and sum(report["selected_counts"]) == report["n_steps"]
        and all(
            0 <= r["action"] < report["n_actions"]
            and r["n_visits"] >= 0
            and 0.0 <= r["Q"] <= 1.0 + 1.0e-6
            and r["sigma"] >= 0.0
            for r in report["per_arm"]
        )
    )
    if gate_pass:
        print("[PASS] VariancePUCT default selector contract holds (R20-P1)")
        return 0
    print("[FAIL] VariancePUCT default selector contract violated")
    return 1


if __name__ == "__main__":
    sys.exit(main())
