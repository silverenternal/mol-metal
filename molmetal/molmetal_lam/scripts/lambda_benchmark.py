"""D4 convergence benchmark for the Lambda MCTS proof-search.

GOAL: Sweep the (n_simulations, branching) plane on a small set of
CrossDocked pockets and record convergence behaviour:

* **n_simulations** : MCTS rollout count per pocket (100 / 500 default,
  ``--full`` raises it to [100, 500, 1000, 5000]).
* **branching**    : MCTS branching factor, controlled via the tile
  library size.  ``12`` -> :func:`build_tile_library` (Phase-0 standard
  12 tiles); ``1020`` -> :func:`FRAGMENT_LIBRARY_200_TILES` (Phase-1
  SMARTS-diverse library; we repeat it 5x to mimic the round-4 1020
  effective branching that combines 200 tiles x 5 click rules).

For each (n_simulations, branching) cell we run one pocket from the
``--pockets`` directory, record:

    n_simulations, branching, pocket_id, best_score, n_candidates,
    lipinski_pass_rate, wall_seconds,
    convergence_iter_to_0.95_best

where ``convergence_iter_to_0.95_best`` is the iteration at which
``search.history[i]["best_score"]`` first crossed 95% of the final
``best_score``.  A small value means fast convergence; ``n_simulations``
means the search never reached its own 95% mark.

Output: ``{prefix}.csv``, ``{prefix}.json``, ``{prefix}.md`` (mirrors
the :mod:`lambda_100pocket_sweep` IO contract).

Reuse strategy
--------------
Seed extraction, reward-building, lipinski/smi helpers and branching
factor -> tile library conversion live in
:mod:`molmetal_lam.scripts._sweep_helpers` and are imported directly
here (this used to be done via ``importlib.util.spec_from_file_location``,
which worked but was brittle).  Branching factor is swapped by passing
a different ``tile_library`` into ``run_one_pocket_benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from molmetal_lam.binding.types import PROTEASE_GENERIC  # noqa: E402
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # noqa: E402
from molmetal_lam.reactions.beta_reductions import REACTION_RULES  # noqa: E402
from molmetal_lam.scripts._sweep_helpers import (  # noqa: E402
    BRANCHING_LABELS,
    build_reward,
    build_seed_smiles,
    build_seed_term,
    build_tile_library_for_branching,
    lipinski_pass,
    smi_of,
    vina_proxy,
)
from molmetal_lam.search_alg.proof_search import MCTSProofSearch  # noqa: E402
from molmetal_lam.types.predicates import LIPINSKI  # noqa: E402


# ---------------------------------------------------------------------------
# Branching-factor control
# ---------------------------------------------------------------------------

#: MCTS branching factor options.  Now sourced from
#: :data:`molmetal_lam.scripts._sweep_helpers.BRANCHING_LABELS` so both
#: sweep scripts agree on a single set of supported values.


# ---------------------------------------------------------------------------
# Single-pocket sweep with convergence tracking
# ---------------------------------------------------------------------------

# Legacy aliases preserved so the rest of this file reads naturally.
_smi_of = smi_of
_lipinski_pass = lipinski_pass
_build_reward = build_reward
_build_tile_library_for_branching = build_tile_library_for_branching


def _convergence_iter(history: List[dict], target_fraction: float = 0.95) -> int:
    """Return the iteration at which ``best_score`` first reached
    ``target_fraction`` of the final ``best_score``.

    If the search never crossed the threshold, returns the total
    iteration count (= ``n_simulations``), which signals "did not
    converge".  ``history`` is the ``MCTSProofSearch.history`` list
    (one entry per simulation step).
    """
    if not history:
        return 0
    final_best = max(float(h.get("best_score", 0.0)) for h in history)
    threshold = float(final_best) * float(target_fraction)
    for h in history:
        if float(h.get("best_score", 0.0)) >= threshold:
            return int(h.get("iteration", 0))
    return len(history)


def run_one_pocket_benchmark(
    pocket_dir: str,
    n_simulations: int,
    branching: int,
    seed_offset: int,
    max_depth: int = 3,
) -> Dict[str, object]:
    """Run one Lambda MCTS sweep and capture convergence metrics.

    Same shape as ``_sweep.run_one_pocket`` but additionally records:

    * ``convergence_iter_to_0.95_best`` — see :func:`_convergence_iter`
    * ``branching_label`` — human-readable description
    """
    pocket_id = os.path.basename(pocket_dir)
    rng = random.Random(20260912 + seed_offset)

    # Seed = first molecule of the pocket ligand file (if present),
    # else cyclopentadiene fallback (matches the 100pocket sweep).
    seed_smiles: Optional[str] = build_seed_smiles(pocket_dir)
    if seed_smiles is None:
        seed_smiles = "C1=CCC=C1"

    try:
        seed = MoleculeClosedTerm.from_smiles(seed_smiles, embed_3d=False)
    except Exception:
        return {
            "pocket_id": pocket_id,
            "branching": branching,
            "branching_label": BRANCHING_LABELS.get(branching, str(branching)),
            "n_simulations": n_simulations,
            "status": "seed_parse_fail",
            "best_score": 0.0,
            "n_candidates": 0,
            "lipinski_pass_rate": 0.0,
            "wall_seconds": 0.0,
            "convergence_iter_to_0.95_best": n_simulations,
        }

    # L-3 tile library — size controlled by ``branching``.
    tile_terms = _build_tile_library_for_branching(branching)

    # Reward: SA + QED (same as the 100pocket sweep).
    reward = build_reward()

    search = MCTSProofSearch(
        tile_library=tile_terms,
        rules=dict(REACTION_RULES),
        target_predicates=[LIPINSKI],
        binding_site=PROTEASE_GENERIC,
        reward=reward,
        n_simulations=n_simulations,
        c_puct=1.4,
        top_k=5,
        rng=rng,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
    )

    t0 = time.time()
    try:
        candidates = search.search(seed, max_depth=max_depth)
    except Exception as exc:
        return {
            "pocket_id": pocket_id,
            "branching": branching,
            "branching_label": BRANCHING_LABELS.get(branching, str(branching)),
            "n_simulations": n_simulations,
            "status": f"search_fail: {type(exc).__name__}",
            "best_score": 0.0,
            "n_candidates": 0,
            "lipinski_pass_rate": 0.0,
            "wall_seconds": time.time() - t0,
            "convergence_iter_to_0.95_best": n_simulations,
        }
    wall = time.time() - t0

    cand_records: List[Tuple[float, str, bool]] = []
    for c in candidates[:20]:
        smi = _smi_of(c)
        cand_records.append((0.0, smi, _lipinski_pass(smi)))

    # Pull best_score from the per-iteration history rather than the
    # final aggregated reward — the history best_score is a strict
    # upper bound on the search progress, and it lets us compute the
    # convergence metric without altering the search internals.
    history = getattr(search, "history", []) or []
    best_score = max(
        (float(h.get("best_score", 0.0)) for h in history),
        default=0.0,
    )
    n_lipinski_pass = sum(1 for _, _, lp in cand_records if lp)
    n_cands = len(cand_records)

    return {
        "pocket_id": pocket_id,
        "branching": branching,
        "branching_label": BRANCHING_LABELS.get(branching, str(branching)),
        "n_simulations": n_simulations,
        "status": "ok",
        "seed_smiles": seed_smiles,
        "best_score": best_score,
        "n_candidates": n_cands,
        "lipinski_pass_rate": (n_lipinski_pass / n_cands) if n_cands else 0.0,
        "wall_seconds": wall,
        "convergence_iter_to_0.95_best": _convergence_iter(history),
    }


# ---------------------------------------------------------------------------
# CSV / JSON / MD IO
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "n_simulations",
    "branching",
    "branching_label",
    "pocket_id",
    "status",
    "best_score",
    "n_candidates",
    "lipinski_pass_rate",
    "wall_seconds",
    "convergence_iter_to_0.95_best",
]


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"per_run": rows}, f, indent=2, default=str)


def write_markdown(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    lines = [
        "# Lambda MCTS convergence benchmark",
        "",
        f"- total runs: {len(rows)}",
        f"- successful runs: {n_ok}",
        "",
        "| n_sims | branching | label | pocket | best_score | n_cand | lipinski | wall_s | conv_iter_0.95 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('n_simulations', '')} | {r.get('branching', '')} | "
            f"{r.get('branching_label', '')} | {r.get('pocket_id', '')} | "
            f"{float(r.get('best_score', 0.0)):.4f} | "
            f"{r.get('n_candidates', 0)} | "
            f"{float(r.get('lipinski_pass_rate', 0.0)):.3f} | "
            f"{float(r.get('wall_seconds', 0.0)):.2f} | "
            f"{r.get('convergence_iter_to_0.95_best', '')} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pockets",
        required=True,
        help="Directory containing CrossDocked pocket sub-directories.",
    )
    p.add_argument(
        "--output-prefix",
        required=True,
        help="Output file prefix (writes .csv, .json, .md).",
    )
    p.add_argument(
        "--pocket-offset",
        type=int,
        default=0,
        help="Skip this many pocket sub-directories (alphabetical order).",
    )
    p.add_argument(
        "--n-pockets",
        type=int,
        default=1,
        help="Number of pockets to run per (n_sims, branching) cell.",
    )
    p.add_argument(
        "--n-simulations-list",
        type=int,
        nargs="+",
        default=[100, 500],
        help="List of MCTS simulation counts per pocket.",
    )
    p.add_argument(
        "--branching-list",
        type=int,
        nargs="+",
        default=[12, 1020],
        help=f"List of branching factors (tile-library sizes). "
        f"Supported: {sorted(BRANCHING_LABELS)}",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="MCTS max depth (default 3; matches the 100pocket sweep).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Use the full n_simulations list [100, 500, 1000, 5000].",
    )
    args = p.parse_args()

    if args.full:
        args.n_simulations_list = [100, 500, 1000, 5000]

    if not os.path.isdir(args.pockets):
        print(f"[bench] pockets dir not found: {args.pockets}", file=sys.stderr)
        return 1

    # Sort pockets alphabetically; slice the requested window.
    entries = sorted(os.listdir(args.pockets))
    pocket_window = entries[
        args.pocket_offset : args.pocket_offset + args.n_pockets
    ]
    if not pocket_window:
        print(f"[bench] no pockets found in window", file=sys.stderr)
        return 1

    csv_path = args.output_prefix + ".csv"
    json_path = args.output_prefix + ".json"
    md_path = args.output_prefix + ".md"

    print(
        f"[bench] pockets={len(pocket_window)} "
        f"n_simulations_list={args.n_simulations_list} "
        f"branching_list={args.branching_list}"
    )

    rows: List[Dict[str, object]] = []
    t_total = time.time()
    for branching in args.branching_list:
        if branching not in BRANCHING_LABELS:
            print(
                f"[bench] warning: branching={branching} has no label; "
                "using raw value",
                file=sys.stderr,
            )
        for n_sims in args.n_simulations_list:
            for idx, name in enumerate(pocket_window):
                pocket_dir = os.path.join(args.pockets, name)
                seed_offset = args.pocket_offset + idx + branching + n_sims
                print(
                    f"[bench] branching={branching} n_sims={n_sims} "
                    f"pocket={name}"
                )
                row = run_one_pocket_benchmark(
                    pocket_dir,
                    n_simulations=n_sims,
                    branching=branching,
                    seed_offset=seed_offset,
                    max_depth=args.max_depth,
                )
                rows.append(row)
                # Periodic flush so partial results survive a crash.
                write_csv(rows, csv_path)
                write_json(rows, json_path)

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_markdown(rows, md_path)
    print(f"[bench] DONE in {time.time() - t_total:.1f}s")
    print(f"[bench] Wrote: {csv_path}")
    print(f"[bench] Wrote: {json_path}")
    print(f"[bench] Wrote: {md_path}")
    print(f"[bench] rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())