#!/usr/bin/env python3
"""STR-2a — D-MPNN out-of-fold predictions on the four generalisation protocols.

MET-2 measured the *tree* under four partitions and found the DOI collapse
(R2 0.3332).  The fusion arm was only ever measured on a ligand-grouped
partition (FIX-M2: +0.0273 R2), so "the blend rescues the paper-held-out
number" is currently an untested hope.  This script produces the missing
half of the fusion: D-MPNN OOF predictions on exactly the folds that
``msrs_generalization.scheme_folds`` defines.

Design:

* Rows are selected by ``select_cohort`` — the same 21,404 rows as MET-2.
* Only the ``metal_state`` arm is trained (FIX-M2's GNN arm); the other two
  D-MPNN arms were already shown to be weaker.
* Folds come from ``scheme_folds``, so a GNN prediction and a tree prediction
  for the same row were produced by models that never saw that row's group.
* Inner validation for early stopping holds out whole *protocol* groups
  (DOI for the DOI protocol, year for the year protocol), not ligands.  Using
  ligand groups there would leak the paper into the early-stopping signal.
* One seed per invocation; the fusion script averages five.

Usage::

    .venv/bin/python molmetal/scripts/msrs_dmpnn_protocols.py \
        --config configs/msrs_dmpnn_surface_gpu0.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_dmpnn_surface import (  # noqa: E402
    DEFAULT_ARM,
    attach_conditioning,
    build_graph_cache,
    cell_one_hot,
    conditioning_matrix,
    graph_key,
    load_config,
    select_cohort,
    train_fold,
    _predict,
)
from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SCHEMES = ("ligand", "doi", "scaffold", "year")
ARM_NAME = "gnn_metal_state"


def _display_path(path: Path) -> str:
    """Repo-relative when possible; the literal argument otherwise."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Cohort and inputs
# ---------------------------------------------------------------------------
def build_inputs(
    long_path: Path,
    endpoint: str,
    limit: int,
    top_cell_lines: int,
) -> Tuple[pd.DataFrame, np.ndarray, List, np.ndarray, Dict[str, object]]:
    """Frame, labels, per-row graphs, fold-safe group keys, summary.

    The returned graph list is row-aligned: one conditioning vector per row,
    because the cell-line one-hot differs between rows that share a graph.
    """
    frame = select_cohort(long_path, endpoint, limit)
    if frame["oxidation_state"].isna().any():
        # scheme_folds / MET-2 require an oxidation state; drop rather than
        # silently conditioning on a zero.
        frame = frame[frame["oxidation_state"].notna()].reset_index(drop=True)

    cache = build_graph_cache(frame, use_metal_node=True)
    keys = [graph_key(r) for r in frame[
        ["ligand_canonical", "metal", "oxidation_state", "charge_complex"]
    ].itertuples(index=False)]
    ok = np.array([cache[k] is not None for k in keys])
    if not ok.all():
        frame = frame[ok].reset_index(drop=True)
        keys = [k for k, keep in zip(keys, ok) if keep]

    y = frame["value_log10"].to_numpy(dtype=float)
    vocab = (
        frame["cell_line_key"]
        .value_counts()
        .head(int(top_cell_lines))
        .index.tolist()
    )
    cell_oh = cell_one_hot(frame, vocab)
    cond = conditioning_matrix(frame, str(DEFAULT_ARM["metal_mode"]), cell_oh)
    graphs = [attach_conditioning(cache[k], cond[i]) for i, k in enumerate(keys)]

    info = {
        "rows": int(len(frame)),
        "ligands_main_fragment": int(frame["ligand_main_fragment"].nunique()),
        "complexes_canonical": int(frame["ligand_canonical"].nunique()),
        "unique_graphs": int(len(cache)),
        "unparseable_graphs": int((~ok).sum()),
        "cell_vocabulary_size": int(len(vocab) + 1),
        "cond_dim": int(cond.shape[1]),
    }
    return frame, y, graphs, cond, info


# ---------------------------------------------------------------------------
# One protocol
# ---------------------------------------------------------------------------
def run_protocol(
    frame: pd.DataFrame,
    y: np.ndarray,
    graphs: Sequence[Optional[Tuple]],
    scheme: str,
    cfg: Dict[str, object],
    device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Train one model per fold; return (fold_id, oof, entry)."""
    import torch

    fold_id, groups, fold_info = scheme_folds(frame, scheme)
    seed = int(cfg["train"]["seed"])
    y_mu, y_sd = float(y.mean()), float(y.std(ddof=0)) or 1.0
    y_z = (y - y_mu) / y_sd

    oof = np.full(len(y), np.nan, dtype=np.float64)
    inner: List[float] = []
    t0 = time.time()
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = np.flatnonzero(fold_id != k)
        te = np.flatnonzero(fold_id == k)

        # Inner validation holds out whole protocol groups.
        rng = np.random.default_rng(seed + k)
        tr_groups = np.array(
            [g for g in groups[tr] if g is not None and str(g) != "nan"]
        )
        uniq = np.unique(tr_groups)
        n_val = max(
            1,
            int(round(len(uniq) * float(cfg["train"]["inner_val_fraction"]))),
        )
        val_groups = set(
            rng.choice(uniq, size=min(n_val, max(1, len(uniq) - 1)), replace=False)
        )
        is_val = np.array([g in val_groups for g in groups[tr]])
        fit_idx, val_idx = tr[~is_val], tr[is_val]
        if len(val_idx) == 0:
            fit_idx, val_idx = tr, tr

        model, best_mae = train_fold(
            [graphs[i] for i in fit_idx],
            y_z[fit_idx],
            [graphs[i] for i in val_idx],
            y_z[val_idx],
            cond_dim=int(conditioning_dim(graphs)),
            arm=DEFAULT_ARM,
            model_cfg=cfg["model"],
            train_cfg=cfg["train"],
            device=device,
            seed=seed + k,
        )
        pred_z = _predict(model, [graphs[i] for i in te], device)
        oof[te] = pred_z * y_sd + y_mu
        inner.append(round(best_mae * y_sd, 4))
        print(
            f"    {scheme:9s} fold={k} inner_val_mae={best_mae * y_sd:.4f} "
            f"({time.time() - t0:.0f}s)"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof[fold_id >= 0]).any():
        raise RuntimeError(f"{scheme}: fold training left evaluated rows unscored")

    entry: Dict[str, object] = {
        **fold_info,
        **score(y[fold_id >= 0], oof[fold_id >= 0]),
        "inner_val_mae": inner,
        "wall_s": round(time.time() - t0, 1),
    }
    print(
        f"[str2a] {scheme:9s} R2={entry['r2']:.4f} "
        f"MAE={entry['mae']:.4f} folds={fold_info['n_folds']}"
    )
    return fold_id, oof, entry


def conditioning_dim(graphs: Sequence[Optional[Tuple]]) -> int:
    for g in graphs:
        if g is not None:
            return int(len(g[4]))
    raise ValueError("no graph carries a conditioning vector")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artifact path; default results/msrs_dmpnn_protocols/seed{N}.json",
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=None,
        help="OOF npz; default results/msrs_dmpnn_protocols/oof_seed{N}.npz",
    )
    args = parser.parse_args(argv)

    import torch

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    seed = int(cfg["train"]["seed"])
    if args.limit is not None:
        cfg["data"]["limit"] = int(args.limit)

    compute_cfg = cfg["compute"]
    if compute_cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(compute_cfg.get("num_threads", 1)))
    device = torch.device(str(compute_cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("config asks for CUDA but torch.cuda.is_available() is False")

    long_path = PROJECT_ROOT / str(cfg["data"]["long"])
    t0 = time.time()
    frame, y, graphs, _cond, info = build_inputs(
        long_path,
        str(cfg["data"]["endpoint"]),
        int(cfg["data"].get("limit", 0)),
        int(cfg["data"]["top_cell_lines"]),
    )
    print(
        f"[str2a] seed={seed} rows={info['rows']} device={device} "
        f"graphs={info['unique_graphs']} ({time.time() - t0:.0f}s)"
    )

    schemes: Dict[str, object] = {}
    oof_arrays: Dict[str, np.ndarray] = {
        "y": y,
        "metal": frame["metal"].astype(str).to_numpy(),
        "cell_line_key": frame["cell_line_key"].astype(str).to_numpy(),
        "doi": frame["doi"].astype(str).to_numpy(),
        "time_h": frame["time_h"].to_numpy(),
        "ligand_canonical": frame["ligand_canonical"].astype(str).to_numpy(),
        "ligand_main_fragment": frame["ligand_main_fragment"].astype(str).to_numpy(),
    }
    for scheme in args.schemes:
        fold_id, oof, entry = run_protocol(frame, y, graphs, scheme, cfg, device)
        oof_arrays[f"{scheme}__fold"] = fold_id.astype(np.int16)
        oof_arrays[f"{scheme}__{ARM_NAME}"] = oof
        schemes[scheme] = entry

    out = args.out or (
        PROJECT_ROOT / "results" / "msrs_dmpnn_protocols" / f"seed{seed}.json"
    )
    oof_out = args.oof_out or (
        PROJECT_ROOT / "results" / "msrs_dmpnn_protocols" / f"oof_seed{seed}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_out, **oof_arrays)

    payload = {
        "artifact": "msrs_dmpnn_protocols",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "config": _display_path(args.config),
            "config_sha256": _sha256(args.config),
            "script": _display_path(Path(__file__)),
            "git_commit": _git_commit(),
            "oof": _display_path(oof_out),
        },
        "design": {
            **info,
            "seed": seed,
            "arm": ARM_NAME,
            "arm_spec": dict(DEFAULT_ARM),
            "folds": "scheme_folds from msrs_generalization (same as MET-2)",
            "inner_validation": "whole protocol groups held out",
            "device": str(device),
        },
        "schemes": schemes,
    }
    atomic_write_file(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[str2a] wrote {oof_out}")
    print(f"[str2a] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
