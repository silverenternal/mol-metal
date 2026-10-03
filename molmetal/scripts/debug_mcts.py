#!/usr/bin/env python3
"""Standalone CLI wrapper for MCTSProofSearch with ``--debug-mcts``.

TODO-IMPROVEMENT_BACKLOG item B3 — the canonical production runner
(:file:`molmetal/scripts/r4_lambda_only_run.py`) is heavy (D-MPNN
loading, RDKit docking, 100-pocket sweeps, ...) so on-demand debugging
needs a slim entrypoint.  This wrapper:

* Builds a minimal :class:`MCTSProofSearch` instance using the same
  ``_IdentityRule`` style mock chemistry the unit tests use (4 small
  tile SMILES, a single trivial reaction rule).
* Runs ``n_simulations`` MCTS rollouts from a starting molecule.
* When ``--debug-mcts`` is set, every iteration prints the PUCT formula
  components (``Q``, ``P``, ``N_child``, ``sqrt(N_parent)``, ``cpuct``,
  ``U``), the leaf's canonical SMILES, the best action (= ``rule_name``
  that produced the leaf), and the rollout reward.
* When ``--debug-mcts`` is **not** set, the script runs silently so
  callers can pipe it through CI without log noise.

Usage::

    python -m molmetal.scripts.debug_mcts --seed 0 --depth 3 --sims 10 --debug-mcts
    python -m molmetal.scripts.debug_mcts --seed 0 --depth 2 --sims 5
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List

# Allow ``python -m molmetal.scripts.debug_mcts`` and direct-script
# execution both.  When invoked as a module under the ``molmetal``
# top-level package, the absolute imports against ``molmetal_lam.*``
# resolve cleanly.  When invoked directly (``python debug_mcts.py``)
# we add the project root to ``sys.path`` and fall back to the same
# absolute imports.
if __package__ in (None, ""):
    sys.path.insert(0, "/home/hugo/codes/mol-metal")
from molmetal_lam.binding.types import BindingSite
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import ReactionRule
from molmetal_lam.search_alg.proof_search import MCTSProofSearch


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors the unit-test fixtures in
# :file:`molmetal/tests/test_proof_search_strengthened.py` so the wrapper
# runs without RDKit / D-MPNN / DiffDock / docking oracle backends.
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Trivial reaction: concatenates atoms of ``state`` and ``tile``.

    Every (state, tile) pair produces a single product whose atom list
    is ``state.atoms + tile.atoms``.  Gives the MCTS enough branching
    to exercise the selection / expansion code without real chemistry.
    """

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            from molmetal_lam.molecules.closed_term import (
                MoleculeClosedTerm as _MCT,
            )
            from molmetal_lam.bonds.application import FreeSiteLedger

            new_atoms = list(state.atoms) + list(tile.atoms)
            return [
                _MCT(
                    atoms=new_atoms,
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"_IdentityRule({self.name!r})"


def _default_tiles() -> List[MoleculeClosedTerm]:
    """Tiny 4-tile library — matches the unit-test fixture."""
    return [
        MoleculeClosedTerm.from_smiles(smi) for smi in ("CCO", "CCN", "CCCl", "CCS")
    ]


def _default_rules() -> Dict[str, ReactionRule]:
    return {"identity": _IdentityRule()}


def _default_binding_site() -> BindingSite:
    return BindingSite(name="debug_site", constraints=[], geometry_hints={})


def _build_search(
    *,
    seed: int,
    sims: int,
    debug: bool,
    c_puct: float = 1.4,
) -> MCTSProofSearch:
    """Construct a minimal :class:`MCTSProofSearch` for the CLI run."""
    return MCTSProofSearch(
        tile_library=_default_tiles(),
        rules=_default_rules(),
        target_predicates=[],
        binding_site=_default_binding_site(),
        scorer=lambda s: float(s.n_atoms) / 10.0,
        n_simulations=int(sims),
        c_puct=float(c_puct),
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.0,  # disable Dirichlet noise for determinism
        early_stop=False,  # honour the full budget deterministically
        debug=bool(debug),
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "MCTSProofSearch debug CLI.  Pass --debug-mcts to print the "
            "per-iteration PUCT trace (step, PUCT components, best action, "
            "reward)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Max β-reduction chain length per simulation (default: 3).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=10,
        help="Number of MCTS simulations to run (default: 10).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.4,
        help="PUCT exploration constant (default: 1.4).",
    )
    parser.add_argument(
        "--start-smiles",
        type=str,
        default="CCO",
        help="Starting molecule SMILES (default: CCO).",
    )
    parser.add_argument(
        "--debug-mcts",
        action="store_true",
        help=(
            "TODO-IMPROVEMENT_BACKLOG B3 — print per-iteration PUCT "
            "trace: step, node being expanded, PUCT formula components "
            "(Q, P, N_child, sqrt(N_parent), cpuct), best action "
            "selected, and reward received.  Default OFF (silent)."
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help=(
            "Print a JSON-style summary at the end of the run with the "
            "final best_score, mean_score, n_states_explored, and the "
            "full debug_trace list (when --debug-mcts is also set)."
        ),
    )
    args = parser.parse_args(argv)

    # Build the search and run.
    try:
        search = _build_search(
            seed=int(args.seed),
            sims=int(args.sims),
            debug=bool(args.debug_mcts),
            c_puct=float(args.c_puct),
        )
    except Exception as exc:
        print(f"[debug-mcts] search_construct_failed: {exc}", file=sys.stderr)
        return 2

    initial = MoleculeClosedTerm.from_smiles(str(args.start_smiles))
    try:
        cands = search.search(initial, max_depth=int(args.depth))
    except Exception as exc:
        print(f"[debug-mcts] search_run_failed: {exc}", file=sys.stderr)
        return 3

    if args.print_summary:
        # Surface the run-level diagnostics in a compact, copy-pasteable
        # form.  This is the canonical output of the wrapper script.
        try:
            best_score = (
                float(search.history[-1]["best_score"]) if search.history else 0.0
            )
            mean_score = (
                float(search.history[-1]["mean_score"]) if search.history else 0.0
            )
            n_states = (
                int(search.history[-1]["n_states_explored"]) if search.history else 0
            )
        except Exception:
            best_score = 0.0
            mean_score = 0.0
            n_states = 0
        # Emit a structured summary line — easy to grep.
        print(
            f"[debug-mcts] summary: best_score={best_score:.4f} "
            f"mean_score={mean_score:.4f} n_states_explored={n_states} "
            f"n_iter={len(search.history)} n_candidates={len(cands)} "
            f"nfe={int(search.nfe)}"
        )
        if args.debug_mcts and getattr(search, "debug_trace", None):
            print(f"[debug-mcts] debug_trace_len=" f"{len(search.debug_trace)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
