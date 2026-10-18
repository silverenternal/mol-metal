#!/usr/bin/env python
"""BLK-P1 — D-MPNN *within-metal* ROC-AUC on the probe's deduplicated cohort.

=====================================================================
Why this file exists
=====================================================================
The paper's positive metal-cytotox result (within-metal Ru 0.8363 / Ir
0.8295) is measured by ``metal_activity_transfer_probe.py``, whose ``model``
field says ``morgan_fp(radius=2, 2048bit) + xgboost`` — i.e. a GBDT, not our
D-MPNN. ``docs/00_MODEL_NARRATIVE.md`` lists the D-MPNN as one of the two
models of the predictive line, so a reviewer is entitled to ask "what is
*your* D-MPNN's within-metal number?" and the repo previously had no answer
on the probe's cohort (BLK-P1).

This script produces that number, and it does so in the only way that makes
the comparison honest: **same cohort, same folds, swap the scorer.**

* cohort   — byte-identical to the probe: ``CytotoxFilter(time_threshold=24.0,
             ic50_min=0.01, metal_whitelist=[m])`` then
             ``collapse_duplicates`` (majority label, ties -> ACTIVE).
* folds    — ``StratifiedKFold(n_splits=5, shuffle=True, random_state=s)``, the
             exact constructor call ``within_metal`` uses. ``split()`` consumes
             only ``(n_samples, y)``, so rebuilding the folds from a zero array
             of the right length reproduces them exactly. The default
             ``--split-seeds`` set is the same 5 seeds PA3 uses, and includes
             42 — the probe's own fold seed — so the ``split_seed == 42`` slice
             is *paired* with the published XGB within column.
* scorer   — ``MetalCytotoxModel`` (the D-MPNN backbone + a 2-way head), vs
             the probe's XGBoost on the same folds.

Both scorers are re-measured here on the identical folds, so the per-metal
delta is *paired* (same train/test ligands), not two numbers from two papers.

Reporting rule (BLK-P1 Step 2)
------------------------------
Per-metal table + the Krasnov comparator column. **No pooled mean**: Ru is
64% of the deduplicated cohort, so a pooled average is dominated by Ru and
would misrepresent Rh/Os/Re (n=51/36/74, below ``MIN_UNIQUE_FOR_GATE=200``
and therefore flagged ``resolvable: false``, not gated).

Honest-negative fallback
------------------------
If the D-MPNN lands *below* the XGB within-metal column, the honest reading
is that the activity-prediction backbone should be named "GBDT by protocol"
and the D-MPNN restricted to the roles it does win (bond head, search value).
That is a legitimate outcome, not a failure, and this script does not hide
it — the delta is written to disk either way.

Usage
-----
::

    # canonical: hyperparameters live in the YAML (AGENTS.md §5.11)
    .venv/bin/python molmetal/scripts/dmpnn_within_metal_auc.py \
        --config configs/dmpnn_within_metal_auc.yaml

    # smoke (CPU, 2 epochs, 1 metal, 1 split seed) — explicit CLI wins over YAML
    .venv/bin/python molmetal/scripts/dmpnn_within_metal_auc.py \
        --metals Ru --split-seeds 42 --epochs 2 --device cpu --max-fold-n 200
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU thread control (perf, not correctness).
# The D-MPNN forward here loops over molecules in Python and does a tiny
# matmul per molecule; torch's default intra-op parallelism then spawns ~90
# threads that thrash on sub-millisecond work (measured: 2231% CPU, ~7 min per
# split seed). Capping to a single thread is ~an order of magnitude faster and
# does not change any number (each molecule's forward is independent).
# Must be set before torch is imported, so it lives above every import.
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from molmetal.scripts.metal_activity_transfer_probe import (  # noqa: E402
    KRASNOV_WITHIN_METAL_AUC,
    METALS,
    MIN_UNIQUE_FOR_GATE,
    _featurise,
    _load_env_default,
    _make_model,
    collapse_duplicates,
)

# Set MOLMETAL_DATA_DIR *before* importing the fine-tune module: that module
# imports ``molmetal.data.cytotox`` at import time, which binds its
# ``DEFAULT_CSV`` constant to whatever the env says right then. Doing this
# after the import would leave the loader pointing at /mnt/storage (absent on
# this host) — measured FileNotFoundError 2026-10-12.
_load_env_default()

# The D-MPNN binary head + its train/eval helpers are reused verbatim from the
# tmQM fine-tune script so the architecture is the same object, not a copy.
from molmetal.scripts.finetune_metalcytotox_from_tmqm import (  # noqa: E402
    MetalCytotoxModel,
    _evaluate,
    _train_one_epoch,
)
from molmetal.width_policy import select_hidden  # noqa: E402

# Belt-and-braces with the OMP_* env vars above: if torch was already imported
# transitively, the env vars are too late, so cap the intra-op pool explicitly.
import torch as _torch  # noqa: E402

_torch.set_num_threads(1)
try:
    _torch.set_num_interop_threads(1)
except RuntimeError:
    # already set / interop pool already started — harmless, intra-op cap stands
    pass


class MaskedMetalCytotoxModel(MetalCytotoxModel):
    """``MetalCytotoxModel`` with a padding-mask in the pooled readout.

    ``MetalCytotoxModel.forward`` pools over ``atom_feats[b, :V]`` with ``V`` the
    *padded* width (``max_atoms=120``), so every molecule carries its zero-filled
    padding atoms into the sum pool — and ``relu(atom_emb(0))`` is a non-zero
    bias vector, not zero. MEASURED on the Ru dedup cohort: median ligand 35
    heavy atoms, so ~68% of pooled slots are padding. This subclass uses each
    molecule's true non-zero atom/edge count instead, which is what a standard
    batched D-MPNN does (chemprop scatters by batch index).

    Weights and architecture are identical to ``MetalCytotoxModel`` — only the
    pooling mask differs — so ``--mask-padding`` isolates that one effect.
    """

    def embed(self, atom_feats, bond_feats, edge_src, edge_dst):
        """The pooled ``2 * hidden`` graph vector that ``forward`` feeds the head.

        Split out of ``forward`` (BLK-F1) so the fusion runner can read the
        *same* representation the D-MPNN head sees and hand it to a second
        scorer, instead of re-implementing the masked pooling loop. Behaviour
        of ``forward`` is byte-identical to before the split.
        """
        import torch

        B = atom_feats.size(0)
        device = atom_feats.device
        pooled = []
        for b in range(B):
            atom_nonzero = atom_feats[b].abs().sum(dim=-1) > 0
            V = int(atom_nonzero.sum().item())
            valid_edges = bond_feats[b].abs().sum(dim=-1) > 0
            E = int(valid_edges.sum().item())
            if V == 0 or E == 0:
                pooled.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled.append(
                self._encode_one(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        return torch.stack(pooled, dim=0)

    def forward(self, atom_feats, bond_feats, edge_src, edge_dst):
        return self.head(self.embed(atom_feats, bond_feats, edge_src, edge_dst))


def _load_dedup_cohort(metal: str, cohort_spec: str = "24h") -> Tuple[List[str], np.ndarray, int]:
    """The probe's exact cohort: filter -> collapse_duplicates(ties->active).

    ``cohort_spec="paper_funnel"`` (W1, 2026-10-13) uses the Krasnov-style
    funnel instead of the 24 h window — the paper funnel is what makes
    Os/Rh/Re resolvable (n=333/334/250 vs 36/51/74).

    ``MOLMETAL_DATA_DIR`` must be set before importing ``cytotox`` because its
    ``DEFAULT_CSV`` is a module-level constant bound at import time.
    """
    _load_env_default()
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    if cohort_spec == "paper_funnel":
        import os as _os

        import pandas as pd

        from molmetal.data.cytotox import build_paper_cohort

        data_dir = Path(
            _os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
        )
        raw = pd.read_csv(data_dir / "MetalCytoToxDB.csv", low_memory=False)
        sub = build_paper_cohort(raw, metal=metal)
        ic50 = sub["IC50_Dark_value"].astype(float)
        y_raw = (ic50 < 10.0).to_numpy(dtype=int)  # loader definition
        unique_smiles, unique_y, n_raw = collapse_duplicates(
            sub["SMILES_Ligands"].astype(str).to_numpy(), y_raw, tie_rule="active"
        )
        return unique_smiles, np.asarray(unique_y, dtype=int), int(n_raw)

    if cohort_spec != "24h":
        raise ValueError(f"unknown cohort_spec {cohort_spec!r} (expected 24h|paper_funnel)")

    ds = MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[metal])
    )
    unique_smiles, unique_y, n_raw = collapse_duplicates(ds.smiles, ds.active, tie_rule="active")
    return unique_smiles, np.asarray(unique_y, dtype=int), int(n_raw)


def _same_folds_as_probe(y: np.ndarray, split_seed: int, n_splits: int):
    """Reproduce ``within_metal``'s folds.

    ``StratifiedKFold.split`` only reads ``(n_samples, y)`` from its first
    argument, so a zero placeholder yields the identical partition that the
    probe builds on its Morgan matrix.
    """
    from sklearn.model_selection import StratifiedKFold

    placeholder = np.zeros((len(y), 1), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    return list(skf.split(placeholder, y))


def _xgb_fold_aucs(smiles, y, folds, split_seed, radius, nbits) -> List[float]:
    """XGBoost on the *same* folds — the paired reference column."""
    from sklearn.metrics import roc_auc_score

    X = _featurise(smiles, radius, nbits)
    aucs = []
    for tr, te in folds:
        clf = _make_model(y[tr], split_seed)
        clf.fit(X[tr], y[tr], verbose=False)
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], prob)))
    return aucs


def _train_one_fold(
    graphs: List,
    targets: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    train_seed: int,
    hidden: int,
    depth: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    max_atoms: int,
    max_edges: int,
    class_weight: bool,
    mask_padding: bool,
    device,
) -> Dict[str, float]:
    """Train a fresh D-MPNN on ``tr`` and score ``te``. Returns {'auc','acc'}."""
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)

    model_cls = MaskedMetalCytotoxModel if mask_padding else MetalCytotoxModel
    model = model_cls(hidden=hidden, depth=depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weight:
        pos = max(1, int(targets[tr].sum()))
        neg = max(1, int(len(tr) - pos))
        w = torch.tensor([1.0, neg / pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        _train_one_epoch(
            model,
            optimizer,
            criterion,
            graphs,
            targets,
            tr,
            batch_size=batch_size,
            max_atoms=max_atoms,
            max_edges=max_edges,
            device=device,
            rng=rng,
        )
    return _evaluate(
        model,
        graphs,
        targets,
        te,
        batch_size=batch_size,
        max_atoms=max_atoms,
        max_edges=max_edges,
        device=device,
    )


def _resolve_device(name: str):
    import torch

    if name and name != "cpu" and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _run_metal(
    metal: str,
    args: argparse.Namespace,
    device,
    featurize_smiles_list,
) -> Dict[str, object]:
    t0 = time.time()
    smiles, y, n_raw = _load_dedup_cohort(metal, getattr(args, "cohort_spec", "24h"))
    if args.max_fold_n and len(y) > args.max_fold_n:
        # smoke only: truncate deterministically so the probe path stays untouched
        rng = np.random.default_rng(args.split_seeds[0])
        keep = np.sort(rng.choice(len(y), size=args.max_fold_n, replace=False))
        smiles = [smiles[i] for i in keep]
        y = y[keep]
    n_unique = int(len(y))
    # Resolve width AFTER n_unique is known: the size-adaptive policy keys on it.
    # Under the default "fixed" policy this returns args.hidden unchanged, so
    # published columns remain reproducible from their configs.
    hidden, width_decision = select_hidden(
        n_unique,
        policy=getattr(args, "width_policy", "fixed"),
        fixed_hidden=args.hidden,
    )
    entry: Dict[str, object] = {
        "n_rows_raw": n_raw,
        "n_unique": n_unique,
        "pos_rate": float(y.mean()) if n_unique else float("nan"),
        "resolvable": bool(n_unique >= MIN_UNIQUE_FOR_GATE),
        "krasnov_reported_auc": KRASNOV_WITHIN_METAL_AUC.get(metal),
        "hidden_used": int(hidden),
        "width_decision": width_decision,
    }

    if n_unique < 2 * args.n_splits or len(set(y.tolist())) < 2:
        entry["status"] = "SKIPPED"
        entry["reason"] = f"n={n_unique} too small or single-class"
        return entry

    graphs = featurize_smiles_list(list(smiles))
    entry["n_invalid_graphs"] = int(sum(g is None for g in graphs))

    per_split: List[Dict[str, object]] = []
    all_dmpnn: List[float] = []
    all_xgb: List[float] = []
    paired42: Optional[Dict[str, float]] = None

    for split_seed in args.split_seeds:
        folds = _same_folds_as_probe(y, split_seed, args.n_splits)
        xgb_fold_aucs = _xgb_fold_aucs(
            smiles, y, folds, split_seed, args.radius, args.nbits
        )
        fold_rows: List[Dict[str, object]] = []
        dmpnn_fold_means: List[float] = []
        for k, (tr, te) in enumerate(folds):
            # drop invalid graphs from each side (keep fold membership aligned)
            tr = np.asarray([i for i in tr if graphs[int(i)] is not None], dtype=int)
            te = np.asarray([i for i in te if graphs[int(i)] is not None], dtype=int)
            if len(tr) == 0 or len(te) == 0 or len(set(y[te].tolist())) < 2:
                fold_rows.append({"fold": k, "status": "SKIPPED"})
                continue
            seed_aucs, seed_accs = [], []
            for sd in args.train_seeds:
                ev = _train_one_fold(
                    graphs,
                    y,
                    tr,
                    te,
                    train_seed=sd,
                    hidden=hidden,
                    depth=args.depth,
                    dropout=args.dropout,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_atoms=args.max_atoms,
                    max_edges=args.max_edges,
                    class_weight=args.class_weight,
                    mask_padding=args.mask_padding,
                    device=device,
                )
                seed_aucs.append(float(ev["auc"]))
                seed_accs.append(float(ev["acc"]))
            dmpnn_fold_means.append(float(np.mean(seed_aucs)))
            all_dmpnn.append(float(np.mean(seed_aucs)))
            all_xgb.append(xgb_fold_aucs[k])
            fold_rows.append(
                {
                    "fold": k,
                    "status": "MEASURED",
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "pos_rate_test": float(y[te].mean()),
                    "dmpnn_seed_aucs": seed_aucs,
                    "dmpnn_auc": float(np.mean(seed_aucs)),
                    "dmpnn_acc": float(np.mean(seed_accs)),
                    "xgb_auc": xgb_fold_aucs[k],
                    "delta_dmpnn_minus_xgb": float(np.mean(seed_aucs) - xgb_fold_aucs[k]),
                }
            )
        if dmpnn_fold_means:
            row = {
                "split_seed": split_seed,
                "dmpnn_auc": float(np.mean(dmpnn_fold_means)),
                "dmpnn_auc_fold_std": float(np.std(dmpnn_fold_means)),
                "xgb_auc": float(np.mean(xgb_fold_aucs)),
                "delta_dmpnn_minus_xgb": float(
                    np.mean(dmpnn_fold_means) - np.mean(xgb_fold_aucs)
                ),
                "per_fold": fold_rows,
            }
            per_split.append(row)
            print(
                f"[blk-p1]   {metal} split_seed={split_seed} "
                f"dmpnn={row['dmpnn_auc']:.4f} xgb={row['xgb_auc']:.4f} "
                f"delta={row['delta_dmpnn_minus_xgb']:+.4f}",
                flush=True,
            )
            if split_seed == 42:
                paired42 = {
                    "dmpnn_auc": row["dmpnn_auc"],
                    "xgb_auc": row["xgb_auc"],
                    "delta_dmpnn_minus_xgb": row["delta_dmpnn_minus_xgb"],
                }

    if not all_dmpnn:
        entry["status"] = "SKIPPED"
        entry["reason"] = "no measurable folds"
        entry["per_split"] = per_split
        return entry

    dmpnn_mean = float(np.mean(all_dmpnn))
    xgb_mean = float(np.mean(all_xgb))
    entry.update(
        {
            "status": "MEASURED",
            "n_splits": args.n_splits,
            "split_seeds": list(args.split_seeds),
            "n_folds_measured": int(len(all_dmpnn)),
            "dmpnn_auc": dmpnn_mean,
            "dmpnn_auc_across_folds_std": float(np.std(all_dmpnn)),
            "xgb_auc_same_folds": xgb_mean,
            "delta_dmpnn_minus_xgb": float(dmpnn_mean - xgb_mean),
            "dmpnn_beats_xgb": bool(dmpnn_mean > xgb_mean),
            "dmpnn_beats_krasnov": (
                bool(dmpnn_mean > KRASNOV_WITHIN_METAL_AUC[metal])
                if KRASNOV_WITHIN_METAL_AUC.get(metal) is not None
                else None
            ),
            "paired_split_seed_42": paired42,
            "per_split": per_split,
            "wall_sec": round(time.time() - t0, 1),
        }
    )
    print(
        f"[blk-p1] {metal:>3} n={n_unique:5d} dmpnn={dmpnn_mean:.4f}"
        f"±{entry['dmpnn_auc_across_folds_std']:.4f} xgb={xgb_mean:.4f} "
        f"delta={entry['delta_dmpnn_minus_xgb']:+.4f} "
        f"krasnov={entry['krasnov_reported_auc']} "
        f"({entry['wall_sec']}s, dev={device})",
        flush=True,
    )
    return entry


def run(args: argparse.Namespace) -> Dict[str, object]:
    from molmetal.baselines.dmpnn import featurize_smiles_list

    if getattr(args, "deterministic", False):
        # cuBLAS needs a fixed workspace for deterministic matmul on CUDA.
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        _torch.use_deterministic_algorithms(True, warn_only=True)

    device = _resolve_device(args.device)

    report: Dict[str, object] = {
        "metric": "dmpnn_within_metal_auc",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "D-MPNN within-metal ROC-AUC on the probe's deduplicated cohort, "
            "measured on the SAME folds the probe's XGBoost within-metal column "
            "uses. Fills BLK-P1: the paper names the D-MPNN as a predictive-line "
            "model but had no within-metal number for it on this cohort."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/dmpnn_within_metal_auc.py",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "cohort": (
            "identical to metal_activity_transfer_probe: CytotoxFilter("
            "time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m]) then "
            "collapse_duplicates(tie_rule=active)"
            if getattr(args, "cohort_spec", "24h") == "24h"
            else "paper funnel (build_paper_cohort, no time window) then "
            "collapse_duplicates(tie_rule=active)"
        ),
        "cohort_spec": getattr(args, "cohort_spec", "24h"),
        "fold_protocol": (
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed); "
            "identical to the probe's within_metal. split_seed=42 reproduces the "
            "probe's published folds exactly (paired slice)."
        ),
        "dmpnn_arch": {
            "model": (
                "MaskedMetalCytotoxModel (D-MPNN backbone + 2-way head, "
                "padding-masked pooled readout)"
                if args.mask_padding
                else "MetalCytotoxModel (D-MPNN backbone + 2-way head)"
            ),
            "hidden": args.hidden,
            "hidden_policy": getattr(args, "width_policy", "fixed"),
            "hidden_note": (
                "hidden is the fixed value; with width_policy=size-adaptive the width "
                "actually used is per_metal.<M>.hidden_used"
                if getattr(args, "width_policy", "fixed") != "fixed"
                else "uniform across metals"
            ),
            "depth": args.depth,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "class_weight": bool(args.class_weight),
            "mask_padding": bool(args.mask_padding),
            "train_seeds": list(args.train_seeds),
        },
        "xgb_reference": {
            "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
            "note": "re-measured on the SAME folds; the delta is paired",
        },
        "split_seeds": list(args.split_seeds),
        "device": str(device),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "krasnov_within_metal_auc": KRASNOV_WITHIN_METAL_AUC,
        "min_unique_for_gate": MIN_UNIQUE_FOR_GATE,
        "per_metal": {},
    }

    for metal in args.metals:
        report["per_metal"][metal] = _run_metal(
            metal, args, device, featurize_smiles_list
        )

    # ---- per-metal table only; no pooled mean (BLK-P1 Step 2) ----
    report["honest_negative"] = {
        "pooled_mean_reported": False,
        "why": (
            "Ru is 64% of the deduplicated cohort, so a pooled mean is a Ru "
            "mean wearing a 5-metal label. The per-metal table is the unit."
        ),
        "unresolvable_metals": [
            m for m, v in report["per_metal"].items() if not v.get("resolvable", False)
        ],
        "dmpnn_below_xgb": [
            m
            for m, v in report["per_metal"].items()
            if v.get("status") == "MEASURED" and not v.get("dmpnn_beats_xgb", True)
        ],
        "fallback_if_dmpnn_loses": (
            "If the D-MPNN lands below the XGB within-metal column, the "
            "activity-prediction backbone should be named 'GBDT by protocol' "
            "and the D-MPNN restricted to its winning roles (bond head, search "
            "value). Written to disk either way."
        ),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D-MPNN within-metal AUC (BLK-P1).")
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument(
        "--split-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 49, 123],
        help="fold seeds. 42 is the probe's own fold seed; the default set "
        "matches PA3's 5-seed protocol so all three columns are comparable.",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="D-MPNN training seeds, averaged within each fold",
    )
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument(
        "--width-policy",
        choices=("fixed", "size-adaptive"),
        default="fixed",
        help=(
            "fixed (default) uses --hidden for every metal. size-adaptive picks "
            "hidden=512 when the deduplicated cohort has n>=1000 and 128 below that, "
            "based on the measured #184 capacity pair "
            "(metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json). The width "
            "actually used is recorded per metal as hidden_used."
        ),
    )
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=240)
    p.add_argument(
        "--class-weight",
        action="store_true",
        default=True,
        help="weight CrossEntropyLoss by inverse class frequency (matches the "
        "probe's scale_pos_weight). Use --no-class-weight to disable.",
    )
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    p.add_argument(
        "--mask-padding",
        action="store_true",
        default=True,
        help="use MaskedMetalCytotoxModel: pool over each molecule's TRUE atom "
        "count instead of the padded width (the reused MetalCytotoxModel does "
        "not mask, and ~68%% of the Ru cohort's pooled slots are padding). "
        "Use --no-mask-padding to reproduce the unmasked model exactly.",
    )
    p.add_argument("--no-mask-padding", dest="mask_padding", action="store_false")
    p.add_argument(
        "--cohort-spec",
        choices=("24h", "paper_funnel"),
        default="24h",
        help=(
            "24h = historical probe cohort; paper_funnel = Krasnov-style "
            "build_paper_cohort (no time window), which resolves Os/Rh/Re. "
            "See W1 / blk_w1_cohort_expansion_probe.py."
        ),
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force bit-reproducible kernels (torch.use_deterministic_algorithms, "
            "cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG). Required for CUDA runs."
        ),
    )
    p.add_argument(
        "--max-fold-n",
        type=int,
        default=0,
        help="smoke only: truncate each metal to N ligands. 0 = full cohort.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (AGENTS.md §5.11). Keys resolve to the dests above; "
        "an explicit CLI flag still wins.",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/dmpnn_within_metal_auc.json",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Module-level entry point for scripts/_config_binding_gate.py."""
    p = _build_parser()
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        _scripts_dir = str(Path(__file__).resolve().parent)
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    rep = run(args)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))

    print(f"\n{'metal':>5} {'n_uniq':>7} {'dmpnn':>8} {'xgb':>8} {'delta':>8} "
          f"{'krasnov':>8}  resolvable")
    for m in args.metals:
        v = rep["per_metal"].get(m, {})
        if v.get("status") != "MEASURED":
            print(f"{m:>5} {v.get('n_unique', 0):7d}  SKIPPED: {v.get('reason', '')}")
            continue
        print(
            f"{m:>5} {v['n_unique']:7d} {v['dmpnn_auc']:8.4f} "
            f"{v['xgb_auc_same_folds']:8.4f} {v['delta_dmpnn_minus_xgb']:+8.4f} "
            f"{str(v['krasnov_reported_auc'] or 'n/a'):>8}  {v['resolvable']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
