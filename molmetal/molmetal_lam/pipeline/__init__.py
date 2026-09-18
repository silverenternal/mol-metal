"""End-to-end Molecular Lambda Calculus (MLC) drug-design pipeline.

============================================================
The closed-loop pipeline
============================================================
This package wires the eleven MLC layers into a single callable loop:

    search_alg.MCTSProofSearch   ─┐
                                  ├── LamClickDesignLoop ──┐
    sbdd_env.REINVENT4Scorer     ─┤                       │
                                  │                       │
    lam_chem.HeuristicRegressor  ─┤   → closed-loop       │
                                  │     iteration:        │
    pipeline.extract_features    ─┘   1) MCTS search      │
                                            candidates       │
                                      2) RDKit descriptors │
                                      3) REINVENT score    │
                                      4) PySR / sklearn    │
                                         heuristic fit     │
                                      5) splice equation   │
                                         into next-iter    │
                                         MCTS heuristic    │
                                                                │
                                  ──────────────────────────────┘

The loop is intentionally small (5 files, ~400 LOC) — the heavy lifting
lives in the layers it composes.  Keeping the pipeline thin makes it
easy to swap any single layer (e.g. drop REINVENT4 for DiffDock) without
touching the others.

Public API
----------
:class:`LamClickDesignLoop`    the closed-loop orchestrator
:func:`molecule_to_features`   8-d RDKit feature vector for any term
:func:`extract_paper_equation` render fitted heuristic as LaTeX-ish text
"""

from molmetal_lam.pipeline.closed_loop import (
    LamClickDesignLoop,
    extract_paper_equation,
)
from molmetal_lam.pipeline.cross_layer_metrics import (
    SA_score_mean,
    end_to_end_yield_proxy,
    retrosynth_feasibility,
    synthesis_success,
)
from molmetal_lam.pipeline.extract_features import (
    FEATURE_NAMES,
    molecule_to_features,
)

__all__ = [
    "LamClickDesignLoop",
    "extract_paper_equation",
    "FEATURE_NAMES",
    "molecule_to_features",
    "synthesis_success",
    "SA_score_mean",
    "retrosynth_feasibility",
    "end_to_end_yield_proxy",
    "__version__",
]

__version__ = "0.0.1-pipeline"
