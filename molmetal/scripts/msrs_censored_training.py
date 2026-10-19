#!/usr/bin/env python3
"""STR-3 — train on the 5,024 censored potency rows, evaluate on the 21,404 exact rows.

18.7% of the potency table is right-censored (median bound 100 uM) and was
dropped by every model so far.  Those rows sit at the *inactive* end, which is
exactly where a delta model needs signal.  This script adds them to the
training set only; evaluation stays on the same 21,404 rows and the same
``scheme_folds`` partitions as MET-2 / STR-1.

Two censored objectives:

``tobit``  right-censored normal NLL, the tested AUD-D1 implementation
           (``msrs_code_levers.tobit_objective``)
``hinge``  the pre-registered form: ``max(0, bound - pred)`` for right-censored
           rows, squared error for exact rows

Fold assignment for a censored row is inherited from its group (ligand / DOI /
scaffold / year) computed on the exact rows.  A censored row whose group never
appears in the evaluation set gets ``fold = -1``: it trains in every fold and
is never scored.

Usage::

    .venv/bin/python molmetal/scripts/msrs_censored_training.py --seed 20261018
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.features.cell import native_codes  # noqa: E402
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_external_validation import (  # noqa: E402
    MIN_YEAR_ROWS,
    murcko_scaffold,
    parse_year,
    year_group,
)
from molmetal.scripts.msrs_fusion_protocols import paired_r2_delta  # noqa: E402
from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    state_block,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
OBJECTIVES = ("tobit", "hinge")
ARM = "ligand+both"
DOI_GATE = 0.36
DOI_KILL = 0.34
LIGAND_TOLERANCE = 0.005
LIGAND_KILL = 0.01
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def select_rows(
    long_path: Path, endpoint: str = ENDPOINT, limit: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact rows (evaluated) followed by censored rows (training only)."""
    long = pd.read_parquet(long_path)
    base = long[
        (long["endpoint"] == endpoint)
        & long["value_log10"].notna()
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy()
    if limit:
        keep = base["ligand_main_fragment"].drop_duplicates().head(int(limit))
        base = base[base["ligand_main_fragment"].isin(set(keep))]
    exact = base[base["censoring"].astype(str) == "none"].reset_index(drop=True)
    censored = base[base["censoring"].astype(str) != "none"].reset_index(drop=True)
    frame = pd.concat([exact, censored], ignore_index=True)
    mask = np.zeros(len(frame), dtype=bool)
    mask[len(exact):] = True
    return frame, mask


# ---------------------------------------------------------------------------
# Fold inheritance
# ---------------------------------------------------------------------------
def group_key(
    frame: pd.DataFrame,
    scheme: str,
    year_counts: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """Same key ``scheme_folds`` uses, computed without splitting."""
    if scheme == "ligand":
        return frame["ligand_canonical"].astype(str).to_numpy()
    if scheme == "doi":
        return frame["doi"].astype(str).to_numpy()
    if scheme == "scaffold":
        return frame["ligand_canonical"].map(murcko_scaffold).to_numpy()
    if scheme == "year":
        if year_counts is None:
            raise ValueError("year scheme needs counts from the exact rows")
        return (
            frame["doi"]
            .map(lambda d: year_group(d, year_counts, MIN_YEAR_ROWS))
            .to_numpy()
        )
    raise ValueError(f"unknown scheme {scheme!r}")


def inherit_folds(
    exact: pd.DataFrame,
    combined: pd.DataFrame,
    scheme: str,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Fold id for every row; censored rows inherit their group's fold."""
    fold_exact, groups_exact, info = scheme_folds(exact, scheme)
    mapping: Dict[object, int] = {}
    for group, fold in zip(groups_exact, fold_exact):
        if fold >= 0:
            mapping[group] = int(fold)

    year_counts = None
    if scheme == "year":
        counts = exact["doi"].map(parse_year).value_counts().to_dict()
        year_counts = {int(k): int(v) for k, v in counts.items()}
    groups_all = group_key(combined, scheme, year_counts)

    folds = np.full(len(combined), -1, dtype=int)
    n_exact = len(exact)
    folds[:n_exact] = fold_exact
    unseen = 0
    for i in range(n_exact, len(combined)):
        group = groups_all[i]
        if group is None or str(group) == "nan":
            continue
        fold = mapping.get(group)
        if fold is None:
            unseen += 1
        else:
            folds[i] = fold

    stats = {
        **info,
        "censored_rows": int(len(combined) - n_exact),
        "censored_with_unseen_group": int(unseen),
        "censored_training_in_all_folds": int(
            ((folds < 0) & (np.arange(len(combined)) >= n_exact)).sum()
        ),
    }
    return folds, stats


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def hinge_objective(censored: np.ndarray):
    """Pre-registered hinge: censored rows only pay when pred < bound."""

    def obj(preds: np.ndarray, dataset):
        y = dataset.get_label()
        below = censored & (preds < y)
        grad = np.where(
            censored,
            np.where(below, -1.0, 0.0),
            2.0 * (preds - y),
        )
        hess = np.where(censored, 1.0, 2.0)
        return grad.astype(float), hess.astype(float)

    return obj


OBJECTIVE_FACTORY = {"hinge": hinge_objective, "tobit": None}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(
    combined: pd.DataFrame,
    exact: pd.DataFrame,
    metal_desc: Path,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    load_descriptors(metal_desc)
    vocab = cell_vocabulary(
        exact["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        combined["metal"].astype(str).tolist(),
        combined["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )
    bundle = build(combined, LIGAND_ONLY)
    state, _ = state_block(combined)
    coord, _, _ = coordination_block(combined, include_geometry=True)
    both = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    X = sparse.hstack([bundle.X, both], format="csr")
    return X, codes


# ---------------------------------------------------------------------------
# One protocol
# ---------------------------------------------------------------------------
def run_protocol(
    X: sparse.csr_matrix,
    codes: np.ndarray,
    y: np.ndarray,
    censored: np.ndarray,
    folds: np.ndarray,
    n_exact: int,
    protocol: str,
    objective: str,
    seed: int,
    baseline: Optional[np.ndarray],
    clusters: np.ndarray,
) -> Dict[str, object]:
    empty = sparse.csr_matrix((len(y), 0), dtype=np.float32)
    t0 = time.time()
    pred = fit_arm(
        X,
        empty,
        empty,
        None,
        0,
        False,
        y,
        folds,
        censored=censored,
        cat_codes=codes,
        params_override={"random_state": int(seed)},
        objective_fn=OBJECTIVE_FACTORY[objective],
    )
    exact_mask = np.zeros(len(y), dtype=bool)
    exact_mask[:n_exact] = True
    ev = exact_mask & (folds >= 0)
    entry: Dict[str, object] = {
        **score(y[ev], pred[ev]),
        "n_evaluated": int(ev.sum()),
        "n_train_rows": int(len(y)),
        "n_censored_train": int(censored.sum()),
        "wall_s": round(time.time() - t0, 1),
    }
    if baseline is not None:
        if len(baseline) != n_exact:
            raise ValueError(
                f"baseline has {len(baseline)} rows, expected {n_exact} exact rows"
            )
        base_full = np.full(len(y), np.nan)
        base_full[:n_exact] = baseline
        clusters_full = np.full(len(y), "", dtype=object)
        clusters_full[:n_exact] = clusters
        finite = ev & np.isfinite(base_full)
        entry["vs_tree"] = paired_r2_delta(
            y[finite],
            base_full[finite],
            pred[finite],
            clusters_full[finite],
            n_boot=N_BOOT,
        )
    print(
        f"[str3] {protocol:9s} {objective:5s} R2={entry['r2']:.4f} "
        f"MAE={entry['mae']:.4f} ({entry['wall_s']}s)"
    )
    return entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligand fragments")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES), choices=list(OBJECTIVES))
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_censored_training.json",
    )
    args = parser.parse_args(argv)

    frame, censored = select_rows(args.long, limit=args.limit)
    n_exact = int((~censored).sum())
    exact = frame.iloc[:n_exact].reset_index(drop=True)
    print(
        f"[str3] exact={n_exact} censored={int(censored.sum())} "
        f"({censored.mean():.1%} of training rows)"
    )

    t0 = time.time()
    X, codes = build_features(frame, exact, args.metal_desc)
    print(f"[str3] features={X.shape} in {time.time() - t0:.0f}s")

    y = frame["value_log10"].to_numpy(dtype=float)
    oof_path = PROJECT_ROOT / "results" / "msrs_seed_ensemble" / f"oof_seed{args.seed}.npz"
    if not oof_path.exists():
        oof_path = PROJECT_ROOT / "results" / "msrs_generalization" / "oof.npz"
    baseline = {}
    if args.limit:
        print("[str3] smoke mode: baseline disabled (row count differs)")
    elif oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as data:
            for protocol in PROTOCOLS:
                baseline[protocol] = data[f"{protocol}__{ARM}"].astype(float)
        print(f"[str3] baseline={oof_path.name}")
    else:
        print(f"[str3] WARNING no baseline OOF found")

    doi_all = exact["doi"].astype(str).to_numpy()
    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        folds, stats = inherit_folds(exact, frame, protocol)
        print(
            f"[str3] {protocol:9s} folds={stats['n_folds']} "
            f"censored_in_train={stats['censored_rows']} "
            f"unseen_group={stats['censored_with_unseen_group']}"
        )
        protocols[protocol] = {"folds": stats, "arms": {}}
        for objective in args.objectives:
            entry = run_protocol(
                X, codes, y, censored, folds, n_exact, protocol, objective,
                args.seed, baseline.get(protocol), doi_all,
            )
            protocols[protocol]["arms"][objective] = entry

    # Verdicts per objective, on DOI and ligand.
    verdicts: Dict[str, object] = {}
    for objective in args.objectives:
        doi = protocols.get("doi", {}).get("arms", {}).get(objective)
        lig = protocols.get("ligand", {}).get("arms", {}).get(objective)
        if doi is None:
            verdicts[objective] = {"status": "NOT_RUN"}
            continue
        if "vs_tree" not in doi:
            verdicts[objective] = {"status": "SMOKE", "doi_r2": doi["r2"]}
            continue
        ci = doi["vs_tree"]
        lig_ref = None
        if lig is not None and "vs_tree" in lig:
            # r2 - delta = the same-seed tree's R2 on the same rows.
            lig_ref = round(float(lig["r2"] - lig["vs_tree"]["delta_r2"]), 4)
        if doi["r2"] >= DOI_GATE and ci["ci_low"] > 0:
            status = "PASS"
        elif doi["r2"] < DOI_KILL or (
            lig is not None and lig_ref is not None and lig["r2"] < lig_ref - LIGAND_KILL
        ):
            status = "KILL"
        else:
            status = "PARTIAL"
        verdicts[objective] = {
            "status": status,
            "doi_r2": doi["r2"],
            "doi_delta": ci["delta_r2"],
            "doi_ci": [ci["ci_low"], ci["ci_high"]],
            "ligand_r2": None if lig is None else lig["r2"],
            "ligand_tree_r2": lig_ref,
        }

    payload = {
        "artifact": "msrs_censored_training",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_censored_training.py",
            "git_commit": _git_commit(),
            "long_sha256": _sha256(args.long),
            "baseline_oof": (
                str(oof_path.relative_to(PROJECT_ROOT)) if oof_path.exists() else None
            ),
        },
        "design": {
            "seed": int(args.seed),
            "n_exact": n_exact,
            "n_censored": int(censored.sum()),
            "censored_share": round(float(censored.mean()), 4),
            "objectives": list(args.objectives),
            "arm": ARM,
            "folds": "scheme_folds on the exact rows; censored rows inherit their group's fold",
            "evaluation": "exact rows only, fold >= 0",
            "doi_gate": DOI_GATE,
            "doi_kill": DOI_KILL,
            "ligand_tolerance": LIGAND_TOLERANCE,
            "ligand_kill": LIGAND_KILL,
            "ligand_reference": "same-seed tree on the same rows (not the 0.5828 stack)",
            "n_features": int(X.shape[1]),
        },
        "protocols": protocols,
        "verdicts": verdicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    for objective, verdict in verdicts.items():
        print(f"[str3] verdict[{objective}]={verdict.get('status')} {verdict}")
    print(f"[str3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
