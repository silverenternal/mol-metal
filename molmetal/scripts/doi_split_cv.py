"""DOI-split 10-fold CV CLI — runs fold k of the paper-replication protocol.

Thin wrapper around :mod:`molmetal.data.cytotox`:

    paper cohort (Ru, measurement-level dedup, non-cancer exclusion)
        -> DOI-grouped folds (all rows of a DOI stay on one side)
        -> fold k definition written as JSON

The JSON's ``train_idx`` / ``val_idx`` are positional row indices **into the
CSV** (i.e. ``pd.read_csv(csv).iloc[train_idx]``), so any consumer can slice
the frame without re-deriving the cohort.

OPT-IN only: nothing in the trainer reads this file, no existing default is
touched.  Not wired into ``train_dmpnn_multitask_wasserstein.py`` — the
trainer's single-split design is unchanged; this CLI materialises fold k so a
caller can consume it.

Usage
-----
    MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal \
        .venv/bin/python -m molmetal.scripts.doi_split_cv --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.data import cytotox  # noqa: E402


def build_fold_payload(
    df: pd.DataFrame,
    *,
    fold: int,
    n_folds: int = 10,
    seed: int = 0,
    metal: str | None = "Ru",
    cohort: bool = True,
    csv_path: str | None = None,
    top_cell_lines: int | None = None,
) -> dict:
    """Fold-k definition payload (indices are positional into ``df``)."""
    if not 0 <= int(fold) < int(n_folds):
        raise ValueError(f"--fold must be in [0, {n_folds}), got {fold}")
    if cohort:
        _cl = None
        if top_cell_lines:
            from molmetal.data.cytotox import top_cell_lines as _top

            _probe = df.loc[cytotox.build_paper_cohort_indices(df, metal=metal)]
            _cl = _top(_probe, int(top_cell_lines))
            print(f"[doi-cv] top-{top_cell_lines} cell lines: {_cl}", flush=True)
        cohort_idx = cytotox.build_paper_cohort_indices(
            df, metal=metal, cell_lines=_cl
        )
    else:
        cohort_idx = np.arange(len(df))
    sub = df.iloc[cohort_idx]
    # 2026-10-04 FIX — `smiles` was not passed, so this produced a DOI-ONLY
    # split: coherent by DOI, incoherent by complex.  The D-MPNN trainer
    # resolves each row's split through a SMILES-keyed dict, so a complex
    # appearing in two articles would have its articles land in different
    # folds and the trainer would silently collapse them — the exact leak
    # `assign_doi_folds(smiles=...)` exists to prevent (MEASURED, 6.1% of
    # cohort complexes span >1 DOI, up to 13).  MEASURED downstream, the
    # leak surfaced as 36 DOI groups appearing on both sides of a fold
    # built from this JSON.
    fold_ids = cytotox.assign_doi_folds(
        sub["DOI"].to_numpy(),
        n_folds=n_folds,
        seed=seed,
        smiles=sub["SMILES_Ligands"],
    )
    k = int(fold)
    val_idx = cohort_idx[np.flatnonzero(fold_ids == k)]
    train_idx = cohort_idx[np.flatnonzero(fold_ids != k)]
    val_dois = set(sub["DOI"].iloc[np.flatnonzero(fold_ids == k)].astype(str))
    train_dois = set(sub["DOI"].iloc[np.flatnonzero(fold_ids != k)].astype(str))
    return {
        "csv": csv_path,
        "protocol": "krasnov2026-metalcytotox-v2/doi-split-10cv",
        "cohort": bool(cohort),
        "metal": metal,
        "n_folds": int(n_folds),
        "seed": int(seed),
        "fold": k,
        "cohort_rows": int(cohort_idx.size),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_train_dois": len(train_dois),
        "n_val_dois": len(val_dois),
        "doi_overlap_train_val": len(train_dois & val_dois),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, required=True, help="validation fold k")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV to split (default: cytotox.DEFAULT_CSV, honours MOLMETAL_DATA_DIR)",
    )
    ap.add_argument(
        "--metal",
        default="Ru",
        help="metal for the paper cohort; 'all' disables the metal filter",
    )
    ap.add_argument(
        "--top-cell-lines",
        type=int,
        default=None,
        help="Restrict the cohort to the top-N cell lines by row volume.  "
        "N=23 reconstructs the paper's 23-cell-line scope; MEASURED it gives "
        "11,517 rows / 4,502 complexes / 23 lines vs the paper's "
        "11,294 / 4,399 / 23 (+2%).  The paper does not publish WHICH lines, "
        "so this subset is OUR choice and must be declared as such.",
    )
    ap.add_argument(
        "--no-cohort",
        action="store_true",
        help="split the raw CSV rows (skip the paper funnel)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper_replication/doi_cv"),
        help="directory for fold_k.json",
    )
    args = ap.parse_args(argv)

    csv_path = Path(args.csv) if args.csv is not None else cytotox.DEFAULT_CSV
    if not csv_path.exists():
        print(f"FAIL: CSV not found at {csv_path} (set MOLMETAL_DATA_DIR?)")
        return 1
    df = pd.read_csv(csv_path, low_memory=False)
    metal = None if str(args.metal).lower() == "all" else args.metal
    payload = build_fold_payload(
        df,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        metal=metal,
        cohort=not args.no_cohort,
        csv_path=str(csv_path),
        top_cell_lines=args.top_cell_lines,
    )
    out_file = Path(args.out) / f"fold_{payload['fold']}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[doi-cv] fold {payload['fold']}/{payload['n_folds']} "
        f"cohort_rows={payload['cohort_rows']} "
        f"n_train={payload['n_train']} n_val={payload['n_val']} "
        f"doi_overlap={payload['doi_overlap_train_val']} -> {out_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
