#!/usr/bin/env python3
"""PA17 — chemprop v2 (official D-MPNN) on MoleculeNet under THIS repo's split.

WHY (BLK-PA17)
--------------
``paper_a`` Table 4's "chemprop D-MPNN" column is a **literature
best-of-hyperparameter-search** constant (Wu et al. 2018: bbbp 0.940 /
bace 0.876 / hiv 0.823 / clintox 0.894).  ``train_moleculenet.py`` says so
itself (``wf_a1_bace_focal/report.md``: "NOT reproduced under our protocol",
"split-match not proven").  A reviewer can therefore argue the gap is an
artefact of comparing a tuned literature number against a fixed-hyperparameter
run on a different split.

This script removes that objection by running the **official chemprop v2
message-passing backbone** (``AtomMessagePassing``, ``d_h``/``depth`` from the
config) end-to-end on the **same Bemis-Murcko scaffold split** this repo uses —
``train_moleculenet._scaffold_split`` is imported, not re-implemented — with the
same 5 seeds, epochs and batch size as the canonical
``metrics/by_metric/moleculenet_auc_dmpnn_5seed.json``.  The only remaining
difference is the model class itself, which is the point.

Defaults are chemprop's, not ours
---------------------------------
chemprop v2 trains with an **unweighted** BCE loss and its own early stopping.
This runner defaults ``--class-weight off`` (chemprop default) and fixed
``--epochs`` (no early stopping, matching this repo's fixed-epoch protocol so
the test set is scored once, at the end).  Both are exposed as flags.

Honest framing (AGENTS.md §5.2)
-------------------------------
Every AUC is MEASURED with ``sklearn.metrics.roc_auc_score`` on the sigmoid of
the pooled head's logit over the task's held-out test split.  If the same-
protocol chemprop value is close to the literature value, the gap stands and is
reported as such; if it is lower, the literature column is best-of-search and
the comparison was never like-for-like.

Usage
-----
::

    .venv/bin/python molmetal/scripts/train_chemprop_v2_moleculenet.py \
        --config configs/pa17_chemprop_v2_moleculenet.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.data.molgraph import MolGraph  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Protocol alignment: reuse the repo's own loader + split so the chemprop run and
# the D-MPNN run differ only in model class.
from molmetal.scripts.train_moleculenet import (  # noqa: E402
    CHEMPROP_DMPNN_BASELINE,
    _load_task,
    _scaffold_split,
)
# Reuse the chemprop v2 encoder + pooled binary head + featurisation cache.
from molmetal.scripts.train_chemprop_lomo_paper_cohort import (  # noqa: E402
    ChempropBinaryLOMO,
    _batched,
    featurize_cache,
)

DEFAULT_TASKS: Tuple[str, ...] = ("bbbp", "bace", "hiv", "clintox")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 64
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 300  # chemprop v2 production default
DEFAULT_DEPTH = 3  # chemprop v2 production default
DEFAULT_FFN_HIDDEN = 300
DEFAULT_FFN_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "pa17_chemprop_v2_moleculenet"

logger = logging.getLogger("pa17_chemprop_v2_moleculenet")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _pos_weight(y: np.ndarray, mode: str) -> Optional[float]:
    if mode == "off":
        return None
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return None
    return n_neg / n_pos


def train_one_seed(
    *,
    smiles: List[str],
    y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    depth: int,
    ffn_hidden: int,
    ffn_layers: int,
    dropout: float,
    class_weight: str,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    tr_idx, _va_idx, te_idx = _scaffold_split(smiles, seed=int(seed))

    t0 = time.time()
    cache, atom_fdim, bond_fdim = featurize_cache(smiles)
    mgs = [cache[s] for s in smiles]

    model = ChempropBinaryLOMO(
        hidden_dim=hidden,
        depth=depth,
        ffn_hidden=ffn_hidden,
        ffn_layers=ffn_layers,
        dropout=dropout,
        metal_mode="none",
    )
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    pw = _pos_weight(y[tr_idx], class_weight)
    pos_weight = torch.tensor(pw, device=device) if pw is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    order = np.arange(len(tr_idx))
    last_loss = float("nan")
    for _ep in range(epochs):
        np.random.shuffle(order)
        total = 0.0
        for start in range(0, order.size, batch_size):
            bidx = tr_idx[order[start : start + batch_size]]
            bmg = _batched(mgs, bidx, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logit = model.logits(bmg)
            tgt = torch.from_numpy(y[bidx]).float().to(device)
            loss = loss_fn(logit.float(), tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item()) * bidx.size
        last_loss = total / max(order.size, 1)

    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, te_idx.size, batch_size):
            bidx = te_idx[start : start + batch_size]
            bmg = _batched(mgs, bidx, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logit = model.logits(bmg)
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)

    auc = float("nan")
    if p.size and len(set(y[te_idx].tolist())) >= 2:
        auc = float(roc_auc_score(y[te_idx], p))
    return {
        "seed": int(seed),
        "n_train": int(tr_idx.size),
        "n_test": int(te_idx.size),
        "test_auc": auc,
        "final_train_loss": float(last_loss),
        "elapsed_s": float(time.time() - t0),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=str, default=None, help="§5.11 YAML config path")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--ffn-hidden", type=int, default=DEFAULT_FFN_HIDDEN)
    p.add_argument("--ffn-layers", type=int, default=DEFAULT_FFN_LAYERS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument(
        "--class-weight",
        choices=("on", "off"),
        default="off",
        help="chemprop v2 default is 'off' (unweighted BCE); 'on' applies neg/pos.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--metric-name", type=str, default="pa17_chemprop_v2_moleculenet")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else out_dir / "metrics.json"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    use_amp = bool(args.bf16) and device.type == "cuda"

    logger.info(
        "device=%s bf16=%s tasks=%s seeds=%s epochs=%d hidden=%d depth=%d "
        "ffn=%dx%d dropout=%.2f batch=%d lr=%.1e class_weight=%s",
        device, args.bf16, args.tasks, args.seeds, args.epochs, args.hidden,
        args.depth, args.ffn_layers, args.ffn_hidden, args.dropout,
        args.batch_size, args.lr, args.class_weight,
    )

    tasks_out: Dict[str, object] = {}
    for task in args.tasks:
        smiles, y = _load_task(task)
        y = np.asarray(y, dtype=np.int64)
        logger.info("task=%s rows=%d pos_rate=%.3f", task, len(smiles), float(y.mean()))
        runs: List[Dict[str, object]] = []
        for seed in args.seeds:
            rec = train_one_seed(
                smiles=smiles, y=y, seed=int(seed), epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
                depth=args.depth, ffn_hidden=args.ffn_hidden,
                ffn_layers=args.ffn_layers, dropout=args.dropout,
                class_weight=args.class_weight, device=device, use_amp=use_amp,
            )
            runs.append(rec)
            logger.info(
                "task=%s seed=%d test_auc=%.4f loss=%.4f %.1fs",
                task, seed, rec["test_auc"], rec["final_train_loss"], rec["elapsed_s"],
            )
        aucs = [r["test_auc"] for r in runs if not np.isnan(r["test_auc"])]
        tasks_out[task] = {
            "n_seeds_total": len(runs),
            "n_seeds_valid": len(aucs),
            "test_auc_mean": float(np.mean(aucs)) if aucs else None,
            "test_auc_std": float(np.std(aucs)) if aucs else None,
            "per_seed_auc": [r["test_auc"] for r in runs],
            "n_test": runs[0]["n_test"] if runs else None,
            "n_train": runs[0]["n_train"] if runs else None,
            "chemprop_dmpnn_literature_baseline": CHEMPROP_DMPNN_BASELINE[task],
            "delta_vs_literature": (
                float(np.mean(aucs) - CHEMPROP_DMPNN_BASELINE[task]) if aucs else None
            ),
        }

    means = [
        v["test_auc_mean"] for v in tasks_out.values() if v["test_auc_mean"] is not None
    ]
    report = {
        "metric": str(args.metric_name),
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "date": time.strftime("%Y-%m-%d"),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer / MoleculeNet backbone control",
        "source": "molmetal/scripts/train_chemprop_v2_moleculenet.py",
        "model": (
            f"chemprop v2 AtomMessagePassing (d_h={args.hidden}, depth={args.depth}) "
            f"+ BinaryClassificationFFN ({args.ffn_layers}x{args.ffn_hidden})"
        ),
        "backbone": "chemprop_v2",
        "featurization": "chemprop_v2 SimpleMoleculeMolGraphFeaturizer (atom+bond)",
        "split": "scaffold (Bemis-Murcko, 70/10/20) via train_moleculenet._scaffold_split",
        "seeds": [int(s) for s in args.seeds],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "class_weight": str(args.class_weight),
        "tasks": tasks_out,
        "mean_auc": float(np.mean(means)) if means else None,
        "chemprop_dmpnn_literature_mean": float(np.mean(list(CHEMPROP_DMPNN_BASELINE.values()))),
        "comparison_baseline": "metrics/by_metric/moleculenet_auc_dmpnn_5seed.json",
        "why": (
            "BLK-PA17: the paper's chemprop column is a literature best-of-search "
            "constant on an unproven split match.  This run measures chemprop v2 "
            "on THIS repo's scaffold split so the comparison is same-protocol."
        ),
        "honest_framing_note": (
            "MEASURED only.  Same scaffold split, same seeds, same epochs/batch as "
            "the canonical D-MPNN MoleculeNet metric; the only difference is the "
            "model class.  class_weight defaults to chemprop's 'off'."
        ),
        "config": {
            "tasks": list(args.tasks),
            "seeds": [int(s) for s in args.seeds],
            "device": str(device),
            "bf16_autocast": bool(use_amp),
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out_json, json.dumps(report, indent=2, default=float, ensure_ascii=False))
    logger.info("wrote %s", out_json)
    for task, v in tasks_out.items():
        logger.info(
            "task=%s mean=%.4f (literature %.3f, delta %+.4f)",
            task, v["test_auc_mean"] or float("nan"),
            v["chemprop_dmpnn_literature_baseline"], v["delta_vs_literature"] or float("nan"),
        )
    if means:
        logger.info("mean over tasks = %.4f", float(np.mean(means)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
