#!/usr/bin/env python3
"""Train a small GRU policy prior on tmQM-derived click-rule labels.

This is the data-prep + training entry-point for **Task L** (learned
MCTS policy prior — small RNN on tmQM reactions).

What it does
------------
1. Loads the tmQM corpus (``molmetal/data/tmqm.py``).
2. For each row, extracts the **click-rule coverage vector** from the
   static SMILES via :func:`functional_group_overlap`.  We treat the
   SMARTS-overlap as a soft supervision signal — *not* a real reaction
   yield — so the prior learns "which click rule is at least
   *applicable* to this state", not "which click rule would actually
   succeed in tmQM".
3. Filters out rows whose SMARTS-overlap vector is all-zero (e.g.
   saturated hydrocarbons) and rows whose SMILES is unparseable.
4. Trains a 2-layer GRU encoder with a softmax classifier head over the
   5 click rules (CuAAC, SPAAC, Suzuki, ThiolEne, AmideCoupling) using
   KL divergence + Adam.
5. Serialises the fitted prior under
   ``molmetal/checkpoints/learned_prior_<metals>.pt`` so the MCTS
   proof-search layer can drop it into the PUCT calculation later.

Usage
-----
::

    uv run python molmetal/scripts/learned_prior_train.py \
        --metals Pt --epochs 100 --batch-size 32 --lr 1e-2 \
        --out checkpoints/learned_prior_Pt.pt

Honest framing
--------------
* The supervision is *coverage*, not *yield*.  Expect hold-out
  accuracy to plateau around 40-60% — well above random 20% but well
  below what a real reaction-yield model would achieve.
* No claim is made about predictive accuracy on novel click partners;
  the prior's purpose is to *bias* MCTS toward applicable rules.
* A tmQM corpus of 30k rows trains in ~3 minutes on CPU (32-dim
  hidden, 2-layer GRU).

Lit basis
---------
* Silver 2017 AlphaGo Zero (Nature 550:354) — learned policy + value
  + MCTS root-noise mixing.
* Schrittwieser 2019 MuZero (Nature 588:59) — model-free latent state.
* Lipman 2023 Theorem 2 — Flow-Main = unconstrained FM; policy prior
  is the lever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch

# Project root on sys.path so this script can be run from anywhere
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molmetal.data.tmqm import load_tmqm
from molmetal_lam.search_alg.learned_prior import (
    DEFAULT_CLICK_RULES,
    LearnedPolicyPrior,
    functional_group_overlap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metals",
        type=str,
        default="Pt",
        help=(
            "Comma-separated list of metal centres to keep "
            "(e.g. 'Pt', 'Pt,Ru,Ir', or 'all')"
        ),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="Maximum number of tmQM rows to use (CPU default)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam learning rate",
    )
    p.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="Adam weight decay (L2 regularisation)",
    )
    p.add_argument(
        "--mix-uniform",
        type=float,
        default=0.5,
        help="Mixing coefficient for uniform prior (AlphaGo Zero style)",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="GRU hidden width (default 32 per the lit spec)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GRU layers (default 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    p.add_argument(
        "--out",
        type=str,
        default="checkpoints/learned_prior_Pt.pt",
        help="Output checkpoint path (relative to project root)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch verbose output",
    )
    return p.parse_args()


def _resolve_metals(arg: str) -> Sequence[str]:
    if arg == "all":
        return ("Pt", "Ru", "Ir")  # PAPER_METALS subset
    return tuple(m.strip() for m in arg.split(",") if m.strip())


def _build_smiles_list(metals: Sequence[str], max_rows: int) -> List[str]:
    """Load tmQM rows + extract SMILES for the requested metals."""
    print(f"[learned_prior_train] loading tmQM (metals={metals}) …", flush=True)
    df = load_tmqm(metals=metals, require_smiles=True, use_cache=True)
    print(
        f"[learned_prior_train] tmQM loaded: {len(df):,} rows, "
        f"metals={sorted(df['metal'].unique())}",
        flush=True,
    )
    # Drop rows with empty / non-parseable SMILES
    smiles_list = df["smiles"].dropna().astype(str).tolist()
    # Trim to max_rows (deterministic by sorting is overkill; just truncate)
    if len(smiles_list) > max_rows:
        smiles_list = smiles_list[:max_rows]
    return smiles_list


def main() -> int:
    args = parse_args()
    metals = _resolve_metals(args.metals)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / "checkpoints" / Path(args.out).name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. data
    smiles_list = _build_smiles_list(metals, args.max_rows)

    # 2. SMARTS-overlap coverage filter (drop all-zero rows)
    t0 = time.time()
    keep_count = 0
    overlap_rows = []
    for s in smiles_list:
        overlap = functional_group_overlap(s)
        if sum(overlap.values()) > 0:
            keep_count += 1
            overlap_rows.append(s)
    print(
        f"[learned_prior_train] SMARTS coverage filter: "
        f"{keep_count:,} of {len(smiles_list):,} rows have at least one match "
        f"({100*keep_count/max(len(smiles_list),1):.1f}%)",
        flush=True,
    )
    print(
        f"[learned_prior_train] SMARTS coverage computed in "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    if keep_count == 0:
        print("[learned_prior_train] ERROR: no rows kept — aborting.", flush=True)
        return 1

    # 3. train
    torch.manual_seed(args.seed)
    prior = LearnedPolicyPrior(
        click_rules=DEFAULT_CLICK_RULES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mix_uniform=args.mix_uniform,
        seed=args.seed,
    )
    print(
        f"[learned_prior_train] training: "
        f"epochs={args.epochs} lr={args.lr} l2={args.l2} "
        f"hidden_dim={args.hidden_dim} num_layers={args.num_layers} "
        f"mix_uniform={args.mix_uniform}",
        flush=True,
    )
    t0 = time.time()
    losses = prior.fit(
        overlap_rows,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0
    print(
        f"[learned_prior_train] training complete: "
        f"{elapsed:.1f}s wall, final loss={losses[-1]:.4f}, "
        f"first loss={losses[0]:.4f}, "
        f"Δloss={losses[0]-losses[-1]:+.4f}",
        flush=True,
    )

    # 4. quick post-train probe (3 chemically distinct SMILES)
    probe_smiles = [
        "C#CCN=[N+]=[N-]",  # alkyne + azide
        "c1ccc(B(O)O)cc1",  # aryl-boronic acid
        "CC(=O)O",          # acetic acid
        "C=CCS",            # thiol-ene
    ]
    print("[learned_prior_train] post-train probe (3 rules × 4 states):", flush=True)
    for s in probe_smiles:
        probs = prior.predict_proba(s)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top3 = ", ".join(f"{r}={p:.3f}" for r, p in ranked[:3])
        print(f"  {s:25s}  →  {top3}", flush=True)

    # 5. checkpoint
    state_dict = {
        "model_state_dict": prior.model.state_dict(),
        "click_rules": list(prior.click_rules),
        "hidden_dim": prior.hidden_dim,
        "num_layers": prior.num_layers,
        "mix_uniform": prior.mix_uniform,
        "seed": args.seed,
        "metals": list(metals),
        "n_train_rows": keep_count,
        "final_loss": losses[-1],
        "first_loss": losses[0],
        "loss_curve": losses,
        "wall_time_seconds": elapsed,
    }
    torch.save(state_dict, out_path)
    print(f"[learned_prior_train] checkpoint written: {out_path}", flush=True)
    # Side-car JSON so external tools can read the metadata without torch.
    meta = {k: v for k, v in state_dict.items() if k != "model_state_dict"}
    json.dump(
        meta, open(out_path.with_suffix(".json"), "w"), indent=2, default=str
    )
    print(f"[learned_prior_train] metadata written: {out_path.with_suffix('.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())