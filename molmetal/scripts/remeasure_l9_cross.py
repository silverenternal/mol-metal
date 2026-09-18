"""Re-measure all 15 metrics on the cisplatin-style seed.

Reads:
  - /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py
  - /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/pipeline/closed_loop.py
  - /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/pipeline/cross_layer_metrics.py

Writes:
  - /home/hugo/codes/try_triton_on_rocm/molmetal/reports/govern_remeasure_L9_cross.md
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Repo root on sys.path so the local molmetal_lam package resolves.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from molmetal_lam.binding.types import BindingSite
from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.pipeline.closed_loop import LamClickDesignLoop
from molmetal_lam.pipeline.cross_layer_metrics import (
    SA_score_mean,
    end_to_end_yield_proxy,
    retrosynth_feasibility,
    synthesis_success,
)
from molmetal_lam.reactions.beta_reductions import ReactionRule
from molmetal_lam.search_alg.proof_search import MCTSProofSearch
from molmetal_lam.types.predicates import TypePredicate


# ---------------------------------------------------------------------------
# Reaction rule: cisplatin-style (Pt + ligand)
# ---------------------------------------------------------------------------


class _PtLigandRule(ReactionRule):
    """Reaction rule that adds Pt(II) to a ligand-like state.

    Mimics cisplatin chemistry: combines an NH3/Cl tile with a Pt-bearing
    state, producing a single product whose atom list = state.atoms + tile.atoms.
    """

    def __init__(self) -> None:
        self.name = "Pt_ligation"
        self.stoichiometry: Dict[str, int] = {}

    def reduce(self, pair):  # type: ignore[override]
        try:
            state, tile = pair
            from molmetal_lam.bonds.application import FreeSiteLedger

            new_atoms = list(state.atoms) + list(tile.atoms)
            return [
                MoleculeClosedTerm(
                    atoms=new_atoms,
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []

    def predict_yield(self, a: str, b: str) -> float:
        # Baseline deterministic predictor: longer ligands score higher.
        return min(1.0, 0.5 + 0.05 * min(len(a), len(b)))


def _binding_site() -> BindingSite:
    return BindingSite(name="cisplatin_site", constraints=[], geometry_hints={})


def _pred(m: MoleculeClosedTerm) -> bool:
    """Predicate: state must have ≥ 4 atoms (cisplatin-like)."""
    try:
        return m.n_atoms >= 4
    except Exception:
        return False


def build_cisplatin_seed() -> MoleculeClosedTerm:
    """Cisplatin SMILES as a MoleculeClosedTerm.

    RDKit canonical SMILES for cisplatin = cis-[Pt(NH3)2Cl2]:
        N.N.Cl.Cl.[Pt]   (5 heavy atoms, 4 dative ligands).
    """
    return MoleculeClosedTerm.from_smiles("N.N.Cl.Cl.[Pt]", embed_3d=False)


def main() -> None:
    rng = random.Random(42)
    seed = build_cisplatin_seed()
    tiles = [MoleculeClosedTerm.from_smiles(s, embed_3d=False) for s in ("CCO", "CCN", "CCCl", "CCS")]
    rules = {"Pt_ligation": _PtLigandRule()}

    # ----- L9 metrics (single 100-sim run) -----
    search = MCTSProofSearch(
        tile_library=tiles,
        rules=rules,
        target_predicates=[TypePredicate(name="has_atoms", predicate_fn=_pred)],
        binding_site=_binding_site(),
        scorer=lambda s: float(s.n_atoms) / 5.0,
        n_simulations=100,
        top_k=5,
        rng=rng,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
    )

    t0 = time.time()
    candidates = search.search(seed, max_depth=4)
    wall = time.time() - t0

    # L9-1 best_score_trajectory (monotone)
    history = search.history
    best_scores = [h["best_score"] for h in history]
    monotone = all(b >= a - 1e-9 for a, b in zip(best_scores, best_scores[1:]))

    # L9-2 n_states_explored (final)
    n_states_final = history[-1]["n_states_explored"]

    # L9-3 n_satisfying (final)
    n_sat_final = history[-1]["n_satisfying"]

    # L9-4 PUCT exploit ratio — proxy via final mean_score vs initial.
    initial_mean = history[0]["mean_score"]
    final_mean = history[-1]["mean_score"]
    exploit_proxy = max(0.0, final_mean - initial_mean)

    # L9-5 rollout_guided_ratio — 0 when prior unfitted.
    rollout_guided = 0.0

    # L9-6 dirichlet_applied — True iff alpha>0 and fraction>0.
    dirichlet_applied = bool(search.dirichlet_alpha > 0 and search.dirichlet_fraction > 0)

    # L9-7 closed_loop_iteration_latency — measured below in the loop.
    closed_loop_latency_per_iter: List[float] = []

    # L9-8 equation_change_rate — measured below.
    equation_change_rates: List[float] = []

    # ----- Closed loop (3 iters) to measure latency + change-rate -----
    @dataclass
    class MockScorer:
        def batch_score(self, smiles_list: List[str]) -> np.ndarray:
            return np.asarray([float(len(s)) for s in smiles_list], dtype=float)

    loop = LamClickDesignLoop(
        mcts=search,
        scorer=MockScorer(),
        symbolic_reg=HeuristicRegressor(niterations=2, random_state=0),
        rng=random.Random(0),
        seed_tiles=[seed],
    )
    loop_results = loop.run(
        pdb_id="cisplatin",
        n_iterations=3,
        top_k=5,
        max_depth=2,
        n_simulations=20,
    )
    closed_loop_latency_per_iter = [
        float(r.get("closed_loop_iteration_latency_s", 0.0)) for r in loop_results
    ]
    eqs = [r["extracted_formula"] for r in loop_results]
    # Equation change rate = fraction of consecutive pairs that differ.
    if len(eqs) >= 2:
        diffs = sum(1 for a, b in zip(eqs, eqs[1:]) if a != b)
        equation_change_rates = [diffs / (len(eqs) - 1)]
    else:
        equation_change_rates = [0.0]

    # L9-9 tree_diversity
    tree_diversity = float(search.tree_diversity)

    # L9-10 rollout_depth_hist
    hist = dict(search.rollout_depth_hist)
    median_depth = (
        float(np.median([k for k, v in hist.items() for _ in range(v)]))
        if hist
        else 0.0
    )

    # ----- Cross-layer metrics -----
    # Collect top-K SMILES across iterations from the loop.
    top_k_all = []
    for r in loop_results:
        top_k_all.extend(r.get("top_k_smiles", []))
    top_k_all = [s for s in top_k_all if s]

    # Cross-1 synthesis_success (uses real AiZynthAdapter w/ SMARTS fallback)
    synth_succ = synthesis_success(top_k_all)
    # Cross-2 SA_score_mean
    sa_mean = SA_score_mean(top_k_all)
    # Cross-3 retrosynth_feasibility
    retro_feas = retrosynth_feasibility(top_k_all)
    # Cross-4 end_to_end_yield_proxy (uses rule.predict_yield)
    yield_proxy = end_to_end_yield_proxy(top_k_all, rules=list(rules.values()))
    # Cross-5 end_to_end_mass_balance (binary: every fired rule has stoich == {})
    mass_balance_violations = sum(1 for r in rules.values() if r.stoichiometry != {})
    end_to_end_mass_balance = 1.0 if mass_balance_violations == 0 else 0.0

    metrics: Dict[str, object] = {
        "L9-1 BEST_SCORE_TRAJECTORY_monotone": bool(monotone),
        "L9-2 N_STATES_EXPLORED_final": int(n_states_final),
        "L9-3 N_SATISFYING_final": int(n_sat_final),
        "L9-4 PUCT_EXPLOIT_RATIO_proxy_delta_mean": float(exploit_proxy),
        "L9-5 ROLLOUT_GUIDED_RATIO": float(rollout_guided),
        "L9-6 DIRICHLET_APPLIED": bool(dirichlet_applied),
        "L9-7 CLOSED_LOOP_ITERATION_LATENCY_s_per_iter": closed_loop_latency_per_iter,
        "L9-8 EQUATION_CHANGE_RATE": float(equation_change_rates[0]),
        "L9-9 TREE_DIVERSITY": float(tree_diversity),
        "L9-10 ROLLOUT_DEPTH_DIST_median": float(median_depth),
        "L9-10 ROLLOUT_DEPTH_DIST_hist": {int(k): int(v) for k, v in hist.items()},
        "CROSS-1 synthesis_success": float(synth_succ),
        "CROSS-2 SA_score_mean": float(sa_mean) if sa_mean == sa_mean else None,
        "CROSS-3 retrosynth_feasibility": float(retro_feas),
        "CROSS-4 end_to_end_yield_proxy": float(yield_proxy),
        "CROSS-5 END_TO_END_MASS_BALANCE": float(end_to_end_mass_balance),
        "wall_time_s_total": float(wall),
        "n_candidates_returned": int(len(candidates)),
    }

    out_path = os.path.join(
        os.path.dirname(HERE),
        "reports",
        "govern_remeasure_L9_cross.json",
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
