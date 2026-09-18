"""lam_demo — End-to-end demo of the LamClickDesignLoop.

============================================================
What this script does
============================================================
1. Builds a tiny :class:`MCTSProofSearch` configured against the
   12-tile standard click library + the :data:`REACTION_RULES` registry.
2. Wires in a :class:`REINVENT4Scorer` (RDKit-only fallback in this
   environment) and a :class:`HeuristicRegressor` (PySR if available,
   else sklearn-Ridge / RandomForest).
3. Runs :meth:`LamClickDesignLoop.run` for 3 iterations on a fake
   pocket (no real PDB download required — the pocket_loader returns
   a stub BindingSite).
4. Prints the best SMILES + best lambda expression + extracted paper
   equation for each iteration.

This is a *smoke test*, not a benchmark — the goal is to verify the
loop runs end-to-end and the heuristic regressor produces a sensible
closed-form expression at the end.

Usage
-----
    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.molmetal_lam.scripts.lam_demo
"""
from __future__ import annotations

import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap: ensure the package root is importable when run as
# ``python -m molmetal.molmetal_lam.scripts.lam_demo``.
# ---------------------------------------------------------------------------
_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor  # noqa: E402
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # noqa: E402
from molmetal_lam.pipeline.closed_loop import LamClickDesignLoop  # noqa: E402
from molmetal_lam.sbdd_env.reinvent_wrapper import REINVENT4Scorer  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight MCTS stub for the demo.  The full :class:`MCTSProofSearch`
# needs tile-library + reaction-rule + type-predicate wiring; here we
# only need it to *return a non-empty list of candidates*.  The demo
# then exercises the rest of the loop (scoring, feature extraction,
# heuristic fit) which is the load-bearing part.
# ---------------------------------------------------------------------------
class _DemoMCTS:
    """Tiny proof-search surrogate: 6 hand-picked SMILES per call.

    Each iteration cycles through a different seed tile so the loop
    sees a *changing* candidate pool and the heuristic actually has
    something to fit.
    """

    def __init__(self) -> None:
        self.smiles_pool: List[str] = [
            "CCO",
            "c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "OCC(O)CO",
            "NCCO",
        ]
        self.call_count: int = 0
        self.last_max_depth: int = 0

    def search(self, initial_state: Any = None, max_depth: int = 3) -> List[MoleculeClosedTerm]:
        self.call_count += 1
        self.last_max_depth = int(max_depth)
        # Rotate the pool per iteration so the loop sees new candidates.
        offset = (self.call_count - 1) % len(self.smiles_pool)
        rotated = self.smiles_pool[offset:] + self.smiles_pool[:offset]
        out: List[MoleculeClosedTerm] = []
        for s in rotated[:4]:
            try:
                out.append(MoleculeClosedTerm.from_smiles(s, embed_3d=False))
            except Exception:
                out.append(MoleculeClosedTerm())
        return out


def _fake_pocket_loader(pdb_id: str) -> Dict[str, Any]:
    """Stub: returns a dict shaped like a BindingSite descriptor."""
    return {
        "pdb_id": pdb_id,
        "kind": "fake-pocket",
        "geometry_hints": {"preferred_donors": ["N", "O"]},
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("lam_demo")

    # ---- 1) Build the loop components
    mcts = _DemoMCTS()
    scorer = REINVENT4Scorer()
    hr = HeuristicRegressor(niterations=4, random_state=0)
    loop = LamClickDesignLoop(
        mcts=mcts,
        scorer=scorer,
        pocket_loader=_fake_pocket_loader,
        symbolic_reg=hr,
        rng=random.Random(0),
    )

    # ---- 2) Run the closed loop
    log.info("=" * 70)
    log.info("LamClickDesignLoop demo — 3 iterations on fake pocket 'demo'")
    log.info("=" * 70)
    results = loop.run(pdb_id="demo", n_iterations=3, top_k=4, max_depth=3)

    # ---- 3) Print diagnostics
    for rec in results:
        print()
        print(f"--- iteration {rec['iteration']} ---")
        print(f"  best_smiles      : {rec['best_smiles']!r}")
        print(f"  best_lambda_expr : {rec['best_lambda_expr']!r}")
        print(f"  best_score       : {rec['best_score']:.4f}")
        print(f"  extracted_formula: {rec['extracted_formula']!r}")
        if rec["top_k_smiles"]:
            print(f"  top_k            : {rec['top_k_smiles']}")
            print(f"  top_k_scores     : {[round(s, 3) for s in rec['top_k_scores']]}")

    print()
    print("=" * 70)
    print(f"FINAL paper_equation : {loop.paper_equation!r}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
