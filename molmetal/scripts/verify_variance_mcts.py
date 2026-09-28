"""Smoke verification for the variance-controlled MCTS integration layer.

================================================================
CLI: Variance vs Dirichlet MCTS variance + wall-time
================================================================
Runs the variance-MCTS integration layer (TODO-30 P5.2) on a stub
MCTS instance and reports:

* ``variance_reduction``:  ``1 - mean_var(variance) / mean_var(base)``
* ``wall_time_ratio``:     ``wall_time_s(variance) / wall_time_s(base)``
* ``success_rate``:        fraction of runs emitting >= 1 candidate

Usage::

    uv run --no-sync python -m molmetal.scripts.verify_variance_mcts --smoke
    uv run --no-sync python -m molmetal.scripts.verify_variance_mcts \\
        --n-runs 5 --max-depth 3

The script is CPU-only and completes in <5 s on the canonical 4-arm
benchmark.  It does NOT mutate ``proof_search.py`` and does NOT
require RDKit or a real molecular setup — the MCTSProofSearch is
stubbed by a lightweight class that mimics the public surface.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List


def _build_stub_search() -> Any:
    """Build a stub ``MCTSProofSearch``-like instance.

    The stub carries the public surface used by
    :func:`compare_variance_vs_dirichlet_mcts`:

    * ``tile_library``
    * ``search(initial_state, max_depth)``
    * ``score_final(state)``
    """

    class _Stub:
        def __init__(self) -> None:
            self.tile_library = ["stub_root"]
            # Three candidates with score-vector (0.3, 0.7, 0.5).
            self._candidates = ["c1", "c2", "c3"]
            self._scores = [0.3, 0.7, 0.5]
            self.leaf_value_var = 0.04
            self.leaf_value_std = 0.2
            self.puct_exploit_ratio_var = 0.97
            self.history = [{"step": 0}]

        def search(self, initial_state: Any, max_depth: int = 3) -> List[Any]:
            return list(self._candidates)

        def score_final(self, state: Any) -> float:
            try:
                idx = self._candidates.index(state)
            except ValueError:
                return 0.0
            return float(self._scores[idx % len(self._scores)])

    return _Stub()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Variance vs Dirichlet MCTS smoke harness",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the canonical 4-run smoke harness.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=4,
        help="Number of repeated search calls (default 4).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Rollout depth budget per call (default 3).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the report as a single JSON line.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.n_runs = 4
        args.max_depth = 3

    # Imported here so the script is parseable even when the package
    # tree is not fully on sys.path.
    from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (
        VarianceMCTSWrapper,
        compare_variance_vs_dirichlet_mcts,
    )

    base = _build_stub_search()
    # Variance-controlled wrapper around a fresh stub.
    variance = VarianceMCTSWrapper(base_search=_build_stub_search())

    t0 = time.time()
    report = compare_variance_vs_dirichlet_mcts(
        base_search=base,
        variance_search=variance,
        n_runs=int(args.n_runs),
        max_depth=int(args.max_depth),
    )
    wall = time.time() - t0

    if args.json:
        out = dict(report)
        out["script_wall_time_s"] = float(wall)
        print(json.dumps(out))
        return 0

    print("=" * 70)
    print("Variance vs Dirichlet MCTS smoke harness (TODO-30 P5.2)")
    print("=" * 70)
    print(f"n_runs      : {report['n_runs']}")
    print(f"max_depth   : {report['max_depth']}")
    print(f"script_wall : {wall:.3f} s")
    print("-" * 70)
    print(f"base.mean_var      : {report['base']['mean_var']:.6f}")
    print(f"variance.mean_var  : {report['variance']['mean_var']:.6f}")
    print(f"variance_reduction : {report['variance_reduction']:.4f}")
    print("-" * 70)
    print(f"base.wall_time_s    : {report['base']['wall_time_s']:.4f}")
    print(f"variance.wall_time_s: {report['variance']['wall_time_s']:.4f}")
    print(f"wall_time_ratio     : {report['wall_time_ratio']:.4f}")
    print("-" * 70)
    print(f"base.success_rate    : {report['base']['success_rate']:.4f}")
    print(f"variance.success_rate: {report['variance']['success_rate']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())