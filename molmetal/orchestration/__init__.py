"""Orchestration layer — the closed-loop drug-design driver.

This is the only layer that knows about *all five* ports at once.  It
composes a :class:`~molmetal.ports.MoleculeGenerator`, a
:class:`~molmetal.ports.DockingEngine`, a
:class:`~molmetal.ports.PropertyPredictor` and a
:class:`~molmetal.ports.ScoringFunction` into the iterative

    generate → dock → predict → score → refine

cycle described by the :class:`~molmetal.ports.DesignLoop` port.

Public API::

    from molmetal.orchestration import ClosedLoop, WeightedSumScorer

    loop = ClosedLoop(generator, docker, predictor, WeightedSumScorer())
    loop.setup(device="cuda")
    top = loop.run(pocket, DesignLoopConfig(n_iterations=3, n_top_k=100))
    history = loop.get_history()
"""

from __future__ import annotations

from molmetal.orchestration.closed_loop import (
    DEFAULT_WEIGHTS,
    ClosedLoop,
    WeightedSumScorer,
)

__all__ = ["ClosedLoop", "WeightedSumScorer", "DEFAULT_WEIGHTS"]
