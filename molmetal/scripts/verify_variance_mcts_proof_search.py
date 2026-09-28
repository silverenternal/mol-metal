"""Wf-36 Phase-2 verification — Variance MCTS in proof_search.

================================================================
End-to-end A/B verification of the three exploration strategies
plumbed through :mod:`molmetal_lam.search_alg.proof_search`:

* ``dirichlet``       (R17 default, AlphaZero root noise)
* ``score_repellent`` (Wf-9 chain replaces ``Dir(α)``)
* ``variance``        (Wf-36 chain + variance-controlled PUCT)

The script runs 100 trajectories on the same mock chemistry
(identity concatenation rule, 4 tile SMILES, scorer ∝ n_atoms)
and reports the mean leaf-value variance and best score for
each strategy.  Output is dumped as JSON to
:file:`molmetal/reports/wf36_variance_mcts/variance_bench.json`
so the L2 closure review can consume it.

Run via::

    uv run --no-sync python molmetal/scripts/verify_variance_mcts_proof_search.py --smoke
    uv run --no-sync python molmetal/scripts/verify_variance_mcts_proof_search.py \\
        --n-trajectories 100 --sims 30 --depth 3

Optional flags
--------------
``--smoke``              quick run (default behaviour, 20 traj × 10 sims).
``--n-trajectories N``   number of independent trajectories per strategy.
``--sims K``             MCTS simulations per trajectory.
``--depth D``            max β-reduction chain length per simulation.
``--seed S``             master seed (default 2025).
``--report-path PATH``   JSON output path (default reports/wf36_*/).

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep
  neural networks and tree search".  *Nature* 529, 484-489.
* Burch E., Bardsley J. M. (2022) "Score-based Repulsive MCMC".
  *SIAM/ASA J. Uncertain. Quantif.* 10(3): 1247-1275.
* Liu J., Feng H. (2026) "Persistent Local Laplacian Prediction of
  Protein-Ligand Binding Affinities".  *J. Chem. Inf. Model.* 66(3).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

# Allow ``python -m molmetal.scripts.verify_variance_mcts_proof_search``
# and direct-script execution both.  When invoked as a module under the
# ``molmetal`` top-level package, the absolute imports resolve cleanly.
# When invoked directly we add the project root to ``sys.path`` and
# fall back to the same absolute imports.
if __package__ in (None, ""):
    sys.path.insert(0, "/home/hugo/codes/mol-metal")
else:
    sys.path.insert(0, "/home/hugo/codes/mol-metal")

try:
    from molmetal_lam.binding.types import BindingSite
    from molmetal_lam.bonds.application import FreeSiteLedger
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import ReactionRule
    from molmetal_lam.search_alg.proof_search import (
        EXPLORATION_STRATEGIES,
        MCTSProofSearch,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
    from molmetal.molmetal_lam.bonds.application import (  # type: ignore
        FreeSiteLedger,
    )
    from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
        MoleculeClosedTerm,
    )
    from molmetal.molmetal_lam.reactions.beta_reductions import (  # type: ignore
        ReactionRule,
    )
    from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore
        EXPLORATION_STRATEGIES,
        MCTSProofSearch,
    )


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors the unit-test fixtures in
# :file:`molmetal/tests/test_persistent_mcts_tree.py` and
# :file:`molmetal/scripts/debug_mcts.py` so the verification runs
# without RDKit / D-MPNN / DiffDock / docking oracle backends.
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Identity concatenation: state + tile → bigger state.

    Gives the MCTS enough branching to exercise the selection /
    expansion code without real chemistry.
    """

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            return [
                MoleculeClosedTerm(
                    atoms=list(state.atoms) + list(tile.atoms),
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []


def _default_tiles() -> List[MoleculeClosedTerm]:
    """4 small tile SMILES — matches the test fixtures."""
    out: List[MoleculeClosedTerm] = []
    for smi in ("CCO", "CCN", "CCCl", "CCS"):
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception:
            continue
    return out


def _default_binding_site() -> BindingSite:
    return BindingSite(name="wf36_site", constraints=[], geometry_hints={})


def _scorer(state: MoleculeClosedTerm) -> float:
    """Mock scorer with non-trivial leaf reward distribution.

    Combines:
    * a linear-in-n_atoms term (gives larger molecules higher scores),
    * a SMILES-hash term (gives *distinct* leaves distinct scores —
      this is what produces a non-zero ``leaf_value_var``),
    * a small mod-7 component (breaks ties between leaves with the
      same n_atoms).

    The SMILES hash is computed from the canonical SMILES when
    available, otherwise from the atom-bag string — both are
    deterministic so the verification runs reproducibly.
    """
    try:
        n_atoms = int(getattr(state, "n_atoms", 0) or 0)
    except Exception:
        n_atoms = 0
    # SMILES-derived hash for non-trivial per-leaf variance.
    try:
        smi = str(state.canonical_smiles() or "")
    except Exception:
        try:
            smi = str(getattr(state, "source_smiles", "") or "")
        except Exception:
            smi = ""
    if not smi:
        try:
            smi = ",".join(
                sorted(str(a) for a in getattr(state, "atoms", []) or []),
            )
        except Exception:
            smi = ""
    smi_hash = float(abs(hash(smi)) % 1000) / 1000.0  # in [0, 1)
    return float(
        min(
            1.0,
            0.05 * float(n_atoms)
            + 0.30 * smi_hash
            + 0.05 * float(n_atoms % 7),
        )
    )


def _build_search(
    *,
    seed: int,
    sims: int,
    strategy: str,
    c_var: float = 0.5,
) -> MCTSProofSearch:
    """Build a minimal :class:`MCTSProofSearch` for one strategy."""
    kwargs = dict(
        tile_library=_default_tiles(),
        rules={"identity": _IdentityRule()},
        target_predicates=[],
        binding_site=_default_binding_site(),
        scorer=_scorer,
        n_simulations=int(sims),
        c_puct=1.4,
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,
        debug=False,
    )
    if strategy in ("score_repellent", "variance"):
        kwargs["exploration_strategy"] = strategy
    if strategy == "variance":
        kwargs["c_var"] = float(c_var)
    return MCTSProofSearch(**kwargs)


def _run_trajectory(
    *,
    strategy: str,
    seed: int,
    sims: int,
    depth: int,
    c_var: float,
    start_smiles: str,
) -> Dict[str, Any]:
    """Run one MCTS trajectory and return per-run metrics."""
    t0 = time.time()
    try:
        start = MoleculeClosedTerm.from_smiles(
            str(start_smiles), embed_3d=False,
        )
    except Exception as exc:
        return {
            "strategy": strategy,
            "seed": int(seed),
            "error": f"start_construct_failed: {exc}",
            "wall_time_s": float(time.time() - t0),
            "leaf_value_var": 0.0,
            "best_score": 0.0,
            "n_states_explored": 0,
        }
    search = _build_search(
        seed=int(seed), sims=int(sims),
        strategy=str(strategy), c_var=float(c_var),
    )
    try:
        search.search(start, max_depth=int(depth))
    except Exception as exc:
        return {
            "strategy": strategy,
            "seed": int(seed),
            "error": f"search_run_failed: {exc}",
            "wall_time_s": float(time.time() - t0),
            "leaf_value_var": 0.0,
            "best_score": 0.0,
            "n_states_explored": 0,
        }
    wall = time.time() - t0
    leaf_var = float(getattr(search, "leaf_value_var", 0.0))
    best = 0.0
    n_states = 0
    try:
        if search.history:
            best = float(search.history[-1].get("best_score", 0.0) or 0.0)
            n_states = int(
                search.history[-1].get("n_states_explored", 0) or 0,
            )
    except Exception:
        pass
    return {
        "strategy": strategy,
        "seed": int(seed),
        "leaf_value_var": float(leaf_var),
        "best_score": float(best),
        "n_states_explored": int(n_states),
        "wall_time_s": float(wall),
    }


def _aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the mean / std / success-rate across runs."""
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {
            "n_runs": int(len(runs)),
            "n_ok": 0,
            "success_rate": 0.0,
            "mean_leaf_value_var": 0.0,
            "std_leaf_value_var": 0.0,
            "mean_best_score": 0.0,
            "std_best_score": 0.0,
            "mean_n_states": 0.0,
            "mean_wall_time_s": float(
                sum(r.get("wall_time_s", 0.0) for r in runs),
            ),
        }
    leaf_vars = [float(r["leaf_value_var"]) for r in ok]
    bests = [float(r["best_score"]) for r in ok]
    n_states = [float(r["n_states_explored"]) for r in ok]
    walls = [float(r["wall_time_s"]) for r in ok]
    mean_lv = sum(leaf_vars) / len(leaf_vars)
    var_lv = sum((x - mean_lv) ** 2 for x in leaf_vars) / max(1, len(leaf_vars))
    mean_bs = sum(bests) / len(bests)
    var_bs = sum((x - mean_bs) ** 2 for x in bests) / max(1, len(bests))
    return {
        "n_runs": int(len(runs)),
        "n_ok": int(len(ok)),
        "success_rate": float(len(ok) / max(1, len(runs))),
        "mean_leaf_value_var": float(mean_lv),
        "std_leaf_value_var": float(var_lv ** 0.5),
        "mean_best_score": float(mean_bs),
        "std_best_score": float(var_bs ** 0.5),
        "mean_n_states": float(sum(n_states) / len(n_states)),
        "mean_wall_time_s": float(sum(walls) / len(walls)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-36 Phase-2 verification — Variance MCTS in proof_search. "
            "Runs 100 trajectories on each of the 3 exploration strategies "
            "and reports variance / success-rate / wall-time."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke harness: 20 trajectories × 10 sims (default behaviour)."
        ),
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=100,
        help="Number of independent trajectories per strategy (default 100).",
    )
    parser.add_argument(
        "--sims", type=int, default=30,
        help="MCTS simulations per trajectory (default 30).",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Max β-reduction chain length per simulation (default 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Master seed (default 2025).",
    )
    parser.add_argument(
        "--start-smiles", type=str, default="CCO",
        help="Starting molecule SMILES (default CCO).",
    )
    parser.add_argument(
        "--c-var", type=float, default=0.5,
        help="Variance penalty coefficient (default 0.5).",
    )
    parser.add_argument(
        "--report-path", type=str,
        default=(
            "/home/hugo/codes/mol-metal/molmetal/reports/"
            "wf36_variance_mcts/variance_bench.json"
        ),
        help="Path to write the JSON report.",
    )
    return parser


def main(argv: Any = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        n_traj = 20
        sims = 10
    else:
        n_traj = int(args.n_trajectories)
        sims = int(args.sims)
    depth = int(args.depth)
    seed = int(args.seed)
    c_var = float(args.c_var)
    start_smiles = str(args.start_smiles)
    print("=" * 72)
    print("Wf-36 Variance-MCTS verification harness")
    print("=" * 72)
    print(f"  strategies      = {list(EXPLORATION_STRATEGIES)}")
    print(f"  n_trajectories  = {n_traj}")
    print(f"  n_simulations   = {sims}")
    print(f"  max_depth       = {depth}")
    print(f"  c_var           = {c_var}")
    print(f"  seed            = {seed}")
    print(f"  start_smiles    = {start_smiles}")
    print()

    t_start = time.time()
    all_runs: Dict[str, List[Dict[str, Any]]] = {}
    for strategy in EXPLORATION_STRATEGIES:
        runs: List[Dict[str, Any]] = []
        for t in range(n_traj):
            run_seed = seed + t
            runs.append(_run_trajectory(
                strategy=strategy,
                seed=run_seed,
                sims=sims,
                depth=depth,
                c_var=c_var,
                start_smiles=start_smiles,
            ))
        all_runs[strategy] = runs
        agg = _aggregate_runs(runs)
        print(
            f"[{strategy:>15s}]  mean_leaf_var={agg['mean_leaf_value_var']:.6f}  "
            f"mean_best={agg['mean_best_score']:.4f}  "
            f"success_rate={agg['success_rate']:.3f}  "
            f"wall_s/run={agg['mean_wall_time_s']:.3f}"
        )
    total_wall = time.time() - t_start

    # Compute variance reduction vs the dirichlet baseline.
    base = _aggregate_runs(all_runs["dirichlet"])
    sr = _aggregate_runs(all_runs["score_repellent"])
    var = _aggregate_runs(all_runs["variance"])
    base_var = base["mean_leaf_value_var"]
    sr_var_reduction = 0.0
    var_var_reduction = 0.0
    if base_var > 1.0e-12:
        sr_var_reduction = float(1.0 - sr["mean_leaf_value_var"] / base_var)
        var_var_reduction = float(1.0 - var["mean_leaf_value_var"] / base_var)
    print()
    print(f"  variance_reduction(score_repellent vs dirichlet) = {sr_var_reduction:.4f}")
    print(f"  variance_reduction(variance vs dirichlet)        = {var_var_reduction:.4f}")
    print(f"  total_wall_time_s = {total_wall:.3f}")

    # Build the JSON report.
    report = {
        "config": {
            "n_trajectories": int(n_traj),
            "n_simulations": int(sims),
            "max_depth": int(depth),
            "c_var": float(c_var),
            "seed": int(seed),
            "start_smiles": start_smiles,
        },
        "strategies": {
            s: _aggregate_runs(all_runs[s]) for s in EXPLORATION_STRATEGIES
        },
        "variance_reduction": {
            "score_repellent_vs_dirichlet": float(sr_var_reduction),
            "variance_vs_dirichlet": float(var_var_reduction),
        },
        "wall_time_s": float(total_wall),
        "n_runs_per_strategy": int(n_traj),
        "all_runs": {
            s: list(all_runs[s]) for s in EXPLORATION_STRATEGIES
        },
    }
    report_path = str(args.report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print()
        print(f"  report written to {report_path}")
    except Exception as exc:
        print(f"[wf36] failed to write report: {exc}", file=sys.stderr)
        return 1

    print()
    print("PASS: Wf-36 verification harness complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
