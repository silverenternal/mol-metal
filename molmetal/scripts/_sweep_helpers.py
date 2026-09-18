"""Shared sweep configuration for the Lambda MCTS 100-pocket runs.

This module centralises the knobs that drive the per-pocket MCTS
search and the per-pocket sweep driver (``lambda_100pocket_sweep.py``)
so the SOTA-comparable defaults are defined exactly once and can be
overridden in tests / ablation runs without copy-pasting magic
numbers across scripts.

Background (round-0, see TODO/14_round0_lambda/round0_findings.md)
-----------------------------------------------------------------

At the Phase-0 defaults (``top_k=5``, ``max_depth=2``) the sweep
emitted only ~1.7 candidates per pocket at ``best_score=1.7046``,
which is not SOTA-comparable — 3D SBDD papers (TransDiff, MolCRAFT,
Pocket2Mol, TargetDiff, DecompDiff) report 50-1000 candidates/pocket
to match their evaluation protocol.

Two knobs control n_candidates:

    * ``top_k``        — number of top-scoring candidates returned by
      :meth:`MCTSProofSearch.search` (primary n_candidates knob).
    * ``max_depth``    — β-reduction chain length per simulation
      (controls rollout depth and leaf diversity).

The defaults below are calibrated to give ~20 candidates per pocket
(lower edge of the SOTA 50-100 band) while keeping wall-time bounded
on the RX 7800 XT, single thread, Python MCTS.

Wall-time projection (round-0 measurement → round-4 projection)
---------------------------------------------------------------

    * 12-tile library, branching=60,   top_k=5,  max_depth=2  →   14s/cell
    * 204-tile library, branching=1020, top_k=5,  max_depth=2  →  298s/cell
    * 204-tile library, branching=1020, top_k=20, max_depth=3  →  ~480s/cell
        (projection: +60% from deeper rollouts + ~4x more leaves to score)

For a 100-pocket sweep at the round-4 defaults this is ~13h
(204-tile) or ~2h (12-tile, ``--no-fragments``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SweepConfig:
    """Per-pocket MCTS sweep configuration (round-4 SOTA-comparable defaults).

    Attributes
    ----------
    top_k : int, default 20
        Number of top-scoring candidates returned by
        :meth:`MCTSProofSearch.search`.  ``top_k`` is the primary
        n_candidates/pocket knob.  Default 20 (was 5 in Phase-0) is
        the lower edge of the SOTA 50-100 band.  Set to 50-100 in
        paper runs to fully match the SOTA evaluation protocol.
    max_depth : int, default 3
        β-reduction chain length per simulation.  Default 3 (was 2
        in Phase-0) gives enough leaves for ``top_k=20`` after the
        typecheck / binding filter.  ``max_depth`` is the secondary
        n_candidates knob — bump it (e.g. 4-5) only if you also raise
        ``top_k`` past 50.
    n_simulations : int, default 200
        MCTS simulation budget per pocket.  Default 200 (was 1000 in
        Phase-0).  Round-0 measurements show UCB converges within ~30
        rollouts at the 12-tile Phase-0 branching — so 200 sims gives
        a ~6x safety margin while keeping wall-time bounded on the
        single-thread Python MCTS.
    patience : int, default 30
        Early-stop patience: when ``early_stop=True`` and the best
        leaf score has not improved for ``patience`` consecutive
        simulations, the search breaks early.  Default 30 is the
        round-0 calibrated value (UCB converges within 16-37 sims at
        1020-branching).
    early_stop : bool, default True
        Enable MCTS early-stop (saves compute when UCB has converged).
        Set to False for ablation / paper-figure runs that require
        the full ``n_simulations`` budget.

    Notes
    -----
    The dataclass is *non-frozen* (``frozen=False`` is the dataclass
    default) so callers can override individual fields in-place, e.g.::

        cfg = SweepConfig()
        cfg.top_k = 50        # bump n_candidates/pocket
        cfg.n_simulations = 500
    """

    top_k: int = 20
    max_depth: int = 3
    n_simulations: int = 200
    patience: int = 30
    early_stop: bool = True


__all__ = ["SweepConfig"]
