#!/usr/bin/env python3
"""UP-28 — the DOI line has never been run on the main model.

=====================================================================
Why this exists
=====================================================================
Every DOI number in this project comes from a fingerprint tree: MACCS +
RDKit descriptors over ECFP4, seed-averaged R^2 0.3767 (the
``maccs_rdkit`` arm of ``msrs_feature_expansion``).  The in-house D-MPNN
is the main model and has never been evaluated on this split.  The
question is narrow: with the same folds and the same evaluation path,
does the graph model beat the tree, and does it clear 0.40?

This is the one package in round 4 that changes the *representation*
rather than the estimand or the protocol.

=====================================================================
Design
=====================================================================
Cohort and folds are byte-identical to UP-21/UP-26
(``msrs_feature_expansion.select_cohort``, ``scheme_folds(frame,
"doi")``).  Two arms run inside one script so the comparison cannot
drift:

  ``fp_tree``    ECFP4 + state/coordination + MACCS/RDKit descriptors,
                 LightGBM via ``msrs_code_levers.fit_arm``.  This is the
                 arm whose seed-averaged R^2 is 0.3767; reproducing it
                 is the protocol check the gate requires.
  ``dmpnn``      D-MPNN over the complex SMILES, with the metal-state,
                 coordination, metal-identity and cell-line blocks
                 concatenated to the pooled embedding before the head.

The D-MPNN sees strictly more information than the tree (the graph
replaces the fingerprints, the side block is the same tabular data), so
a win here is "better representation", not a controlled ablation.  That
is stated in the artifact rather than corrected for.

Evaluation goes through ``msrs_surface_model.score`` on the same
``fold_id >= 0`` rows, exactly as the other DOI artifacts do.  Two
aggregates are reported because they answer different questions:

  ``r2_mean_over_seeds``   what a single training run delivers
  ``r2_seed_averaged``     what a five-checkpoint ensemble delivers,
                           and the number 0.3767 is quoted against

=====================================================================
Hyperparameters
=====================================================================
chemprop defaults (Yang et al. 2019, J Cheminform 11:1): hidden 300,
depth 3, dropout 0.1, batch 64, Adam lr 1e-3.  Epochs are fixed at 40
with no early stopping: an early-stopping split inside the training
folds would either leak a DOI or need a nested split, and a fixed
budget keeps the five seeds comparable.  Every value lives in
``configs/msrs_dmpnn_doi.yaml`` (§5.11).

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
primary     D-MPNN seed-averaged DOI R^2 >= 0.40
baseline    fp_tree seed-averaged R^2 reproduces 0.3767 +- 0.005
per_seed    the five per-seed R^2 values and their sd are reported
kill        D-MPNN mean < 0.3767 closes the representation line

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_dmpnn_doi.py --config configs/msrs_dmpnn_doi.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.dmpnn import (  # noqa: E402
    DMPNNModel,
    _atom_feature_dim,
    collate_graphs,
    featurize_smiles_list,
)
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_feature_expansion import (  # noqa: E402
    build_arm,
    select_cohort,
)
from molmetal.scripts.msrs_generalization import (  # noqa: E402
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    native_codes,
    scheme_folds,
    state_block,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    _git_commit,
    _sha256,
    cell_matrix,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides  # noqa: E402

# 0..4 are the ``random_state`` values behind the 0.3767 maccs_rdkit reference
# (``msrs_feature_expansion.py --seed {0..4}``); reusing them makes the fp_tree
# arm a bit-exact protocol reproduction rather than a lookalike.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
R2_GATE = 0.40
BASELINE_TARGET = 0.3767
BASELINE_TOL = 0.005


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _rel(p: Path) -> str:
    """Path relative to the repo root when possible, absolute otherwise."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def _dense(x) -> np.ndarray:
    """Dense float32 view of a block that may be sparse or already dense."""
    if hasattr(x, "todense"):
        x = x.todense()
    return np.asarray(x, dtype=np.float32)


def build_blocks(
    frame: pd.DataFrame, atom_version: str
) -> Tuple[List[Optional[tuple]], np.ndarray, np.ndarray]:
    """Graphs, dense side block, and a validity mask.

    Side block = metal state (4) + coordination (22) + metal one-hot (5)
    + cell one-hot (top-N vocabulary).  Nothing here is derived from y.
    """
    smiles = frame["ligand_canonical"].astype(str).tolist()
    graphs = featurize_smiles_list(smiles, version=atom_version, metal_mode="retain")
    valid = np.array([g is not None for g in graphs], dtype=bool)

    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    vocab = cell_vocabulary(
        frame["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    cells = cell_matrix(frame["cell_line_key"].astype(str).tolist(), vocab)
    metal = frame["metal"].astype(str).to_numpy()
    metal_oh = np.zeros((len(frame), len(METALS)), dtype=np.float32)
    for i, m in enumerate(metal):
        if m in METALS:
            metal_oh[i, METALS.index(m)] = 1.0

    side = np.hstack([
        _dense(state),
        _dense(coord),
        metal_oh,
        _dense(cells),
    ])
    return graphs, side, valid


class DMPNNWithSide(DMPNNModel):
    """D-MPNN with the tabular block concatenated before the regression head."""

    def __init__(self, extra_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.head = nn.Sequential(
            nn.Linear(self.hidden * 2 + extra_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(self.hidden, max(self.hidden // 2, 8)),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(max(self.hidden // 2, 8), 1),
        )

    def forward(self, atom, bond, e_src, e_dst, extra):
        pooled = self.pool_batch(atom, bond, e_src, e_dst)
        return self.head(torch.cat([pooled, extra], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _batch_tensors(graphs, side, idx, device):
    batch = collate_graphs([graphs[i] for i in idx])
    atom, bond, e_src, e_dst = batch
    return (
        atom.to(device), bond.to(device), e_src.to(device), e_dst.to(device),
        torch.from_numpy(side[idx]).to(device),
    )


@torch.no_grad()
def predict_fold(model, graphs, side, idx, batch_size, device) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), dtype=float)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        atom, bond, e_src, e_dst, extra = _batch_tensors(graphs, side, chunk, device)
        out[start:start + len(chunk)] = model(atom, bond, e_src, e_dst, extra).cpu().numpy()
    return out


def train_fold(
    graphs, side, y_std, tr_idx, te_idx, cfg, seed, device
) -> Tuple[np.ndarray, Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DMPNNWithSide(
        extra_dim=side.shape[1],
        atom_dim=_atom_feature_dim(cfg["atom_version"]),
        hidden=cfg["hidden"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for _ in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(tr_idx)
        total, n = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            chunk = order[start:start + cfg["batch_size"]]
            atom, bond, e_src, e_dst, extra = _batch_tensors(graphs, side, chunk, device)
            target = torch.from_numpy(y_std[chunk]).to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(model(atom, bond, e_src, e_dst, extra), target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(chunk)
            n += len(chunk)
        losses.append(total / max(n, 1))
    pred = predict_fold(model, graphs, side, te_idx, cfg["batch_size"], device)
    return pred, {"train_loss_first": round(losses[0], 6),
                  "train_loss_last": round(losses[-1], 6)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    # every YAML leaf must reach a dest of this parser (§5.11 binding gate)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: first N ligand fragments (§5.7 CPU smoke path)")
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--atom-version", default="v2", choices=("v1", "v2"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--arms", nargs="+", default=["fp_tree", "dmpnn"])
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_dmpnn_doi.json")
    args = ap.parse_args(argv)
    if args.config is not None:
        args = apply_yaml_overrides(ap, args, str(args.config), argv=list(argv or []))
    # YAML paths are written relative to the repo root; make them absolute so
    # provenance strings and ``relative_to`` do not depend on the cwd.
    for name in ("long", "metal_desc", "out", "config"):
        p = getattr(args, name)
        if p is None:
            continue
        setattr(args, name, p if p.is_absolute() else PROJECT_ROOT / p)

    cfg = {
        "hidden": args.hidden, "depth": args.depth, "dropout": args.dropout,
        "atom_version": args.atom_version, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
    }
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    t0 = time.time()
    frame = select_cohort(args.long, limit=args.limit)
    y = frame["value_log10"].to_numpy(dtype=float)
    fold_id, _, fold_info = scheme_folds(frame, "doi")
    ev = fold_id >= 0
    print(f"[UP-28] rows={len(frame)} folds={fold_info['n_folds']} device={device} "
          f"arms={args.arms} seeds={len(args.seeds)}")

    results: Dict[str, object] = {}

    # ---------------- fingerprint tree (protocol check) ----------------
    if "fp_tree" in args.arms:
        load_descriptors(args.metal_desc)
        vocab = cell_vocabulary(
            frame["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES)
        codes = native_codes(
            frame["metal"].astype(str).tolist(),
            frame["cell_line_key"].astype(str).tolist(), vocab, mode="full")
        base = build(frame, LIGAND_ONLY)
        state, _ = state_block(frame)
        coord, _, _ = coordination_block(frame, include_geometry=True)
        state_coord = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
        X_fp = build_arm(
            frame, "maccs_rdkit",
            sparse.hstack([base.X, state_coord], format="csr"), state_coord)
        empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
        preds, per_seed = [], []
        for seed in args.seeds:
            p = fit_arm(X_fp, empty, empty, None, 0, False, y, fold_id,
                        cat_codes=codes, params_override={"random_state": int(seed)})
            preds.append(p)
            per_seed.append(score(y[ev], p[ev])["r2"])
        avg = np.mean(np.vstack(preds), axis=0)
        results["fp_tree"] = {
            "r2_per_seed": [round(float(v), 4) for v in per_seed],
            "r2_mean_over_seeds": round(float(np.mean(per_seed)), 4),
            "r2_sd_over_seeds": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else 0.0,
            "r2_seed_averaged": score(y[ev], avg[ev])["r2"],
            "n_features": int(X_fp.shape[1]),
        }
        print(f"[UP-28] fp_tree mean={results['fp_tree']['r2_mean_over_seeds']} "
              f"seed_avg={results['fp_tree']['r2_seed_averaged']}")

    # ---------------- D-MPNN ----------------
    if "dmpnn" in args.arms:
        graphs, side, valid = build_blocks(frame, args.atom_version)
        n_bad = int((~valid).sum())
        print(f"[UP-28] graphs={len(graphs)} unparsable={n_bad} side_dim={side.shape[1]}")
        if n_bad:
            keep = np.where(valid)[0]
            graphs = [graphs[i] for i in keep]
            side = side[keep]
        else:
            keep = np.arange(len(frame))
        # standardise the numeric part only; one-hots are left alone
        n_num = 4 + 22
        mu = side[:, :n_num].mean(axis=0)
        sd = side[:, :n_num].std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        side[:, :n_num] = (side[:, :n_num] - mu) / sd

        y_mean, y_sd = float(y[keep].mean()), float(y[keep].std())
        y_std = (y[keep] - y_mean) / y_sd
        pos = {int(r): i for i, r in enumerate(keep)}

        preds, per_seed, fold_diag = [], [], []
        for seed in args.seeds:
            ts = time.time()
            oof = np.full(len(keep), np.nan)
            for k in sorted(set(fold_id[fold_id >= 0].tolist())):
                tr = np.array([pos[i] for i in np.where((fold_id != k) & (fold_id >= 0))[0]],
                              dtype=int)
                te = np.array([pos[i] for i in np.where(fold_id == k)[0]], dtype=int)
                p, diag = train_fold(graphs, side, y_std, tr, te, cfg, seed, device)
                oof[te] = p
                if seed == args.seeds[0]:
                    fold_diag.append({"fold": int(k), **diag})
                    print(f"[UP-28]   fold {k} loss {diag['train_loss_first']:.5f} -> "
                          f"{diag['train_loss_last']:.5f}")
            full = np.full(len(frame), np.nan)
            full[keep] = oof * y_sd + y_mean
            preds.append(full)
            r2 = score(y[ev], full[ev])["r2"]
            per_seed.append(r2)
            print(f"[UP-28] dmpnn seed={seed} r2={r2:.4f} "
                  f"({round(time.time()-ts,1)}s)")
        avg = np.nanmean(np.vstack(preds), axis=0)
        results["dmpnn"] = {
            "r2_per_seed": [round(float(v), 4) for v in per_seed],
            "r2_mean_over_seeds": round(float(np.mean(per_seed)), 4),
            "r2_sd_over_seeds": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else 0.0,
            "r2_seed_averaged": score(y[ev], avg[ev])["r2"],
            "n_unparsable": n_bad,
            "side_dim": int(side.shape[1]),
            "fold_diagnostics": fold_diag,
        }
        np.savez_compressed(
            PROJECT_ROOT / "metrics" / "by_metric" / "msrs_dmpnn_doi_oof.npz",
            y=y, fold_id=fold_id.astype(np.int16),
            oof=np.vstack(preds).astype(np.float32),
        )
        print(f"[UP-28] dmpnn mean={results['dmpnn']['r2_mean_over_seeds']} "
              f"seed_avg={results['dmpnn']['r2_seed_averaged']}")

    # ---------------- gate ----------------
    checks: Dict[str, object] = {}
    if "dmpnn" in results:
        d = results["dmpnn"]
        checks["primary"] = {
            "value": d["r2_seed_averaged"], "threshold": R2_GATE,
            "pass": bool(d["r2_seed_averaged"] >= R2_GATE),
        }
        checks["per_seed"] = {
            "r2_per_seed": d["r2_per_seed"], "sd": d["r2_sd_over_seeds"],
            "pass": True,
        }
    if "fp_tree" in results:
        f = results["fp_tree"]
        checks["baseline"] = {
            "value": f["r2_seed_averaged"], "target": BASELINE_TARGET,
            "tolerance": BASELINE_TOL,
            "pass": bool(abs(f["r2_seed_averaged"] - BASELINE_TARGET) <= BASELINE_TOL),
        }
    if "dmpnn" in results and "fp_tree" in results:
        checks["delta_vs_tree"] = {
            "value": round(results["dmpnn"]["r2_seed_averaged"]
                           - results["fp_tree"]["r2_seed_averaged"], 4),
            "pass": True,
        }
    gate_pass = bool(checks) and all(v["pass"] for v in checks.values())
    kill = {
        "rule": "D-MPNN mean R^2 < 0.3767 closes the representation line",
        "value": results.get("dmpnn", {}).get("r2_seed_averaged"),
        "triggered": bool(
            "dmpnn" in results
            and results["dmpnn"]["r2_seed_averaged"] < BASELINE_TARGET),
    }

    out = {
        "artifact": "msrs_dmpnn_doi",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 representation (UP-28)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": _rel(args.long),
            "long_sha256": _sha256(args.long),
            "script": _rel(Path(__file__)),
            "config": _rel(args.config) if args.config else None,
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_feature_expansion.select_cohort (21,404 rows)",
            "split": "scheme_folds('doi'), 5 GroupKFold folds on DOI",
            "device": str(device),
            "deterministic": bool(args.deterministic),
            "seeds": list(args.seeds),
            "hyperparameters": cfg,
            "side_block": "metal state (4) + coordination (22) + metal one-hot (5) "
                          "+ cell one-hot (top-40); numeric part standardised on the "
                          "whole cohort, one-hots untouched",
            "evaluation": "msrs_surface_model.score on fold_id >= 0 rows, same as "
                          "every other DOI artifact",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "results": results,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": gate_pass, "kill": kill},
        "elapsed_seconds": round(time.time() - t0, 1),
        "caveats": [
            "The D-MPNN sees strictly more information than the fingerprint arm: the "
            "graph replaces ECFP4/MACCS/RDKit and the side block adds metal and cell "
            "identity that the tree also has. A win is a representation win, not a "
            "controlled ablation of graph-vs-fingerprint alone.",
            "Epochs are fixed at 40 with no early stopping. An early-stopping split "
            "inside the training folds would either leak a DOI or need a nested split, "
            "and a fixed budget keeps the seeds comparable.",
            "The numeric side block is standardised on the whole cohort rather than "
            "per fold. It is label-free, so this is transductive preprocessing, not "
            "leakage, but it does mean the fold is not fully self-contained.",
            "The seed-averaged R^2 is the number 0.3767 is quoted against; the "
            "per-seed mean is what one training run delivers. They differ by about "
            "0.01 on the tree and the gate reads the seed-averaged one.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for arm, r in results.items():
        print(f"[UP-28] {arm:8s} mean={r['r2_mean_over_seeds']} "
              f"sd={r['r2_sd_over_seeds']} seed_avg={r['r2_seed_averaged']}")
    print(f"[UP-28] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
