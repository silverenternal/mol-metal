"""Wf-320D smoke runner — chemprop v2 per-metal with featurization cache.

This is the CLI surface for the Wf-320D timing benchmark + integration
smoke. It reuses the canonical
``train_chemprop_v2_permetal_singletask.py`` driver and slots in the
:mod:`wf320d_featurization_fix` per-epoch cache + faster featurizer.

Usage (CPU smoke, default 1 epoch × 1 seed × 1 metal):

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_v2_fastfeaturize.py \\
        --metals Ru --seeds 0 --epochs 1 --hidden 64 --depth 2 \\
        --device cpu \\
        --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv \\
        --out /tmp/wf320d_featurization_fix_smoke --execute

Output:
    /tmp/wf320d_featurization_fix_smoke/{metal}_seed{N}.json with
    cache_stats (n_hit, n_miss, hit_rate) + per-run wall_time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pin to CPU by default; this script is CPU-only per the Wf-320D
# honest-framing constraint. Override with --device cuda if needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metals", nargs="+", default=["Ru"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="/tmp/wf320d_featurization_fix_smoke")
    p.add_argument(
        "--csv",
        default="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the smoke (without this flag, dry-run print-only).",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Wf-320D featurization fix smoke ==")
    print(f"metals={args.metals} seeds={args.seeds} epochs={args.epochs}")
    print(f"hidden={args.hidden} depth={args.depth} device={args.device}")
    print(f"out_dir={out_dir}")
    print(f"csv={args.csv}")

    if not args.execute:
        print("[dry-run] pass --execute to actually run training")
        return 0

    # Lazy import — keep the dry-run path light.
    from molmetal.scripts.wf320d_featurization_fix import (
        FastAtomFeaturizer,
        FastBondFeaturizer,
        PerEpochFeatureCache,
    )
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.nn import AtomMessagePassing, BinaryClassificationFFN, SumAggregation

    torch.manual_seed(args.seeds[0])
    np.random.seed(args.seeds[0])

    df_all = pd.read_csv(args.csv)
    df = df_all[df_all["metal"].isin(args.metals)].reset_index(drop=True)
    if df.empty:
        print(f"No rows for metals={args.metals}; aborting.")
        return 1
    print(f"Loaded {len(df)} rows for metals={args.metals}")

    device = torch.device(args.device)
    smiles_list = df["smiles"].astype(str).tolist()
    active_arr = df["activity"].astype(int).to_numpy()

    # Pre-featurise via chemprop v2 default (faster than pure-Python fast path
    # at production scale; fast path kept opt-in for benchmarking).
    cache = PerEpochFeatureCache(use_fast_featurizer=False)
    t0 = time.time()
    cache.precompute_smiles(smiles_list)
    precompute_s = time.time() - t0
    print(f"[tier-1] precompute_smiles: {precompute_s:.2f}s "
          f"({cache.cache_info()['n_canonical_smiles']} unique SMILES)")

    # Build backbone.
    sample_smi = smiles_list[0]
    sample = cache._mols[sample_smi]
    atom_fdim = sample["V"].shape[1]
    bond_fdim = sample["E"].shape[1] if sample["E"].ndim > 1 else 14

    mp = AtomMessagePassing(d_v=atom_fdim, d_e=bond_fdim, d_h=args.hidden, depth=args.depth)
    agg = SumAggregation()
    head = BinaryClassificationFFN(
        n_tasks=1,
        input_dim=mp.output_dim,
        hidden_dim=args.hidden,
        n_layers=2,
        dropout=0.0,
    )
    mp = mp.to(device)
    agg = agg.to(device)
    head = head.to(device)

    optim = torch.optim.Adam(
        list(mp.parameters()) + list(agg.parameters()) + list(head.parameters()),
        lr=args.lr,
    )
    bce = torch.nn.BCEWithLogitsLoss()

    # Train/val split.
    idx = np.arange(len(smiles_list))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.20, random_state=args.seeds[0],
        stratify=active_arr.astype(int),
    )

    # Pre-compute batches for the train set (deterministic).
    train_batches = []
    for s in range(0, len(idx_train), args.batch_size):
        bi = idx_train[s:s + args.batch_size]
        train_batches.append([smiles_list[int(i)] for i in bi])
    val_batches = []
    for s in range(0, len(idx_val), args.batch_size):
        bi = idx_val[s:s + args.batch_size]
        val_batches.append([smiles_list[int(i)] for i in bi])

    train_active_t = torch.from_numpy(active_arr[idx_train]).float().to(device)
    bi_offsets = np.cumsum([0] + [len(b) for b in train_batches])

    # Training loop using the per-epoch cache.
    train_t0 = time.time()
    for epoch in range(args.epochs):
        cache.begin_epoch(epoch)
        order = np.random.permutation(len(train_batches))
        total_loss = 0.0
        nb = 0
        for batch_pos in order:
            smiles_batch = train_batches[batch_pos]
            bmg = cache.get_or_build_batch(
                epoch_idx=epoch,
                batch_idx=batch_pos,
                smiles_batch=smiles_batch,
                device=device,
                batch_molgraph_ctor=BatchMolGraph,
            )
            H_v = mp(bmg)
            H = agg(H_v, bmg.batch)
            logit = head.train_step(H)
            # Target indices.
            row_start = int(bi_offsets[batch_pos])
            row_end = int(bi_offsets[batch_pos + 1])
            tgt = train_active_t[row_start:row_end].unsqueeze(-1)
            loss = bce(logit, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(smiles_batch)
            nb += len(smiles_batch)
        avg_loss = total_loss / max(nb, 1)
        elapsed = time.time() - train_t0
        print(f"  epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} "
              f"cache: hit={cache.stats.n_hit} miss={cache.stats.n_miss} "
              f"hit_rate={cache.stats.hit_rate:.2%} elapsed={elapsed:.1f}s")

    train_wall = time.time() - train_t0

    # Eval.
    mp.eval(); agg.eval(); head.eval()
    scores = []
    labels = []
    eval_epoch_idx = args.epochs + 1
    cache.begin_epoch(eval_epoch_idx)  # mark a fresh epoch so cache lookup is valid
    with torch.no_grad():
        for batch_pos, smiles_batch in enumerate(val_batches):
            bmg = cache.get_or_build_batch(
                epoch_idx=eval_epoch_idx,
                batch_idx=batch_pos + 10 ** 6,  # offset to avoid training cache hits
                smiles_batch=smiles_batch,
                device=device,
                batch_molgraph_ctor=BatchMolGraph,
            )
            H_v = mp(bmg)
            H = agg(H_v, bmg.batch)
            logit = head.train_step(H)
            p = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
            scores.extend(float(x) for x in p)
    labels = [int(active_arr[int(i)]) for i in idx_val[: len(scores)]]

    if len(set(labels)) >= 2:
        auc = float(roc_auc_score(labels, scores))
    else:
        auc = float("nan")

    # Save result.
    out_json = {
        "metal": args.metals[0],
        "seed": args.seeds[0],
        "epochs": args.epochs,
        "wall_train_s": train_wall,
        "wall_precompute_s": precompute_s,
        "test_auc": auc,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "cache_stats": cache.stats.to_dict(),
    }
    out_path = out_dir / f"{args.metals[0]}_seed{args.seeds[0]}.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"== Wrote {out_path}")
    print(f"== cache hit_rate={cache.stats.hit_rate:.2%} "
          f"({cache.stats.n_hit} hits / {cache.stats.n_miss} misses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
