#!/usr/bin/env python
"""Smoke harness for the MCTS replay buffer (TODO-31).

================================================================
Lightweight verify-script — buffer stats + replay sample quality
================================================================
Usage::

    uv run --no-sync python -m molmetal.scripts.verify_mcts_replay --smoke
    uv run --no-sync python -m molmetal.scripts.verify_mcts_replay --n-samples 200

Populates a replay buffer with ``--n-samples`` synthetic entries,
reports the buffer's stats dict, draws a sample, and prints the
loss estimates + replay-sample quality.

The script is CPU-only and never touches
:mod:`proof_search` / :mod:`score_repellent_mc` /
:mod:`variance_puct_integration` / :mod:`variance_mcts_integration`.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np


def _build_stub_search():
    """Build a stub base_search that exposes public diagnostics only.

    No real MCTSProofSearch is constructed — the wrapper accepts any
    object with a ``.search(root)`` method and reads diagnostics off
    ``history[-1]``.
    """

    class _StubMol:
        def __init__(self, smiles):
            self._smi = smiles

        def canonical_smiles(self):
            return self._smi

        def __repr__(self):
            return f"_StubMol({self._smi!r})"

        def __str__(self):
            return self._smi

    candidates = [
        _StubMol("CCO"),
        _StubMol("CCN"),
        _StubMol("CCC"),
        _StubMol("CCCC"),
    ]
    scores = [0.3, 0.7, 0.5, 0.9]
    priors = [0.25, 0.25, 0.25, 0.25]
    visits = [4, 8, 2, 16]
    variances = [0.01, 0.02, 0.015, 0.005]

    class _StubSearch:
        history = [
            {
                "candidates": list(candidates),
                "scores": list(scores),
                "priors": list(priors),
                "visits": list(visits),
                "variances": list(variances),
            }
        ]
        debug_trace = []
        _root_children_cache_for_search = None

        def search(self, root):
            self.history.append(
                {
                    "candidates": list(candidates),
                    "scores": list(scores),
                    "priors": list(priors),
                    "visits": list(visits),
                    "variances": list(variances),
                }
            )
            return int(np.argmax(np.asarray(scores)))

    return _StubSearch()


def _run_smoke(n_samples: int) -> int:
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayWrapper,
        ReplayGuidedMCTS,
    )

    # 1. Build a stub MCTS + buffer.
    base = _build_stub_search()
    buf = MCTSReplayBuffer(max_size=max(n_samples * 2, 64), seed=42)
    wrapper = MCTSReplayWrapper(
        base_search=base,
        buffer=buf,
        n_top_actions=4,
    )

    # 2. Drive n_samples replay-augmented searches.
    n_searches = max(1, n_samples // 4)
    for i in range(n_searches):
        wrapper.search(root=f"root_iter_{i}")

    # 3. Train (loss estimates) on a fresh replay batch.
    loss = wrapper.train_value_network()

    # 4. Bias the prior of a fresh search from the accumulated replay.
    rg = ReplayGuidedMCTS(base_search=base, replay_buffer=buf, alpha_replay=0.5)
    biased = rg.bias_prior(root="CCO", n_actions=4)

    # 5. Pretty-print the report.
    stats = buf.stats()
    print("=" * 64)
    print("MCTS Replay Buffer — Smoke Verify (TODO-31)")
    print("=" * 64)
    print(f"n_searches_run       : {wrapper.n_searches_}")
    print(f"n_entries_recorded   : {wrapper.n_entries_recorded_}")
    print(f"buffer.size          : {stats['size']}")
    print(f"buffer.capacity      : {stats['capacity']}")
    print(f"buffer.n_added       : {stats['n_added']}")
    print(f"buffer.n_evicted     : {stats['n_evicted']}")
    print(f"buffer.mean_value    : {stats['mean_value']:.6f}")
    print(f"buffer.std_value     : {stats['std_value']:.6f}")
    print(f"buffer.mean_prior    : {stats['mean_prior']:.6f}")
    print(f"buffer.mean_variance : {stats['mean_variance']:.6f}")
    print(f"buffer.mean_visit    : {stats['mean_visit_count']:.6f}")
    print(
        f"buffer.value_range   : "
        f"({stats['value_range'][0]:.4f}, {stats['value_range'][1]:.4f})"
    )
    print()
    print("Replay-guided prior (root='CCO', n_actions=4):")
    print(f"  bias_prior         : {biased.tolist()}")
    print()
    print("Loss estimates (train_value_network, fresh batch):")
    print(f"  value_loss         : {loss['value_loss']:.6f}")
    print(f"  policy_loss        : {loss['policy_loss']:.6f}")
    print(f"  batch_size         : {loss['batch_size']}")
    print(f"  mean_value         : {loss['mean_value']:.6f}")
    print(f"  mean_prior         : {loss['mean_prior']:.6f}")
    print()

    # 6. PASS / FAIL contract — buffer must be non-empty, prior must
    # be a valid probability vector.
    contract_pass = (
        stats["size"] > 0
        and stats["n_added"] >= wrapper.n_entries_recorded_
        and abs(float(biased.sum()) - 1.0) < 1.0e-6
        and (biased >= 0.0).all()
        and loss["batch_size"] > 0
    )
    if contract_pass:
        print("[PASS] MCTS replay buffer smoke verify OK")
        return 0
    print("[FAIL] MCTS replay buffer smoke verify FAILED")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small smoke verify (default n_samples=64).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=256,
        help="Number of synthetic replay entries to push (default 256).",
    )
    args = parser.parse_args(argv)
    n = 64 if args.smoke else args.n_samples
    return _run_smoke(n)


if __name__ == "__main__":
    sys.exit(main())
