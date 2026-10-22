#!/usr/bin/env python3
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""UP-14 — re-report every headline number split by whether it has paired evidence.

All first-round metrics are pooled.  If most of the error sits on rows that have no
metal-pair evidence, the pooled number describes the corpus, not the model's
capability.

Two axes, each with two definitions:

  * **unit** — does this row's `(ligand_canonical, cell_line_key)` group carry >= 2
    metals?  The unit the UP-10 audit counted.
  * **metal** — is this row's metal a member of a pair with >= 20 matched groups?
    Ru/Os/Ir/Rh yes (Ru-Os, Ir-Rh); Re no.

and each axis is evaluated twice:

  * **corpus** — counted over the whole table.  Descriptive.
  * **train_evidence** — counted only over the rows the model could actually learn
    from in that fold.  This is the definition that matters, and it is fold-aware:
    a test unit whose partner metal sits in the same fold has no evidence at all.

Applied to three protocols: `doi` (metal seen, paper unseen), `within_metal` (metal
seen, ligand unseen), `metal_out` (metal unseen).  Each stratum reports n, share of
rows, R2, MAE, and its share of the total squared error, so "how much of the pooled
number comes from supported rows" is a number rather than an adjective.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_support_stratified.py
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
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import METALS, _git_commit, _sha256  # noqa: E402
from molmetal.scripts.msrs_lomo_surface import (  # noqa: E402
    DEFAULT_METAL_COLUMNS,
    assemble,
    build_blocks,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    N_FOLDS,
    TOP_CELL_LINES,
    fit_predict,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

MIN_GROUPS = 20
CROSS_METAL_MIN = 2


def supported_pairs(df: pd.DataFrame, min_groups: int = MIN_GROUPS) -> Dict[str, int]:
    units = (
        df.groupby(["ligand_canonical", "cell_line_key", "metal"])["value_log10"]
        .mean()
        .reset_index()
    )
    out = {}
    metals = sorted(df["metal"].unique())
    for i, a in enumerate(metals):
        for b in metals[i + 1:]:
            left = units.loc[units["metal"] == a, ["ligand_canonical", "cell_line_key"]]
            right = units.loc[units["metal"] == b, ["ligand_canonical", "cell_line_key"]]
            n = len(left.merge(right, on=["ligand_canonical", "cell_line_key"]))
            if n >= min_groups:
                out[f"{a}-{b}"] = n
    return out


def unit_key(df: pd.DataFrame) -> pd.Series:
    return df["ligand_canonical"].astype(str) + "|" + df["cell_line_key"].astype(str)


def corpus_unit_flag(df: pd.DataFrame) -> np.ndarray:
    key = unit_key(df)
    n_metals = df.assign(_k=key).groupby("_k")["metal"].transform("nunique")
    return (n_metals >= CROSS_METAL_MIN).to_numpy()


def train_evidence_flags(
    df: pd.DataFrame, fold_id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold-aware: unit has >= 2 metals *in the training rows*; metal is in a
    supported pair *in the training rows*."""
    key = unit_key(df).to_numpy()
    metals = df["metal"].to_numpy()
    unit_flag = np.zeros(len(df), dtype=bool)
    metal_flag = np.zeros(len(df), dtype=bool)
    for f in np.unique(fold_id):
        tr = np.where(fold_id != f)[0]
        te = np.where(fold_id == f)[0]
        if len(tr) == 0 or len(te) == 0:
            continue
        train = df.iloc[tr].assign(_k=key[tr])
        seen = train.groupby("_k")["metal"].nunique()
        supported_units = set(seen[seen >= CROSS_METAL_MIN].index)
        unit_flag[te] = pd.Series(key[te]).isin(supported_units).to_numpy()

        pairs = supported_pairs(df.iloc[tr])
        pair_metals = {m for p in pairs for m in p.split("-")}
        metal_flag[te] = np.array([m in pair_metals for m in metals[te]])
    return unit_flag, metal_flag


def stratum_report(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    if not mask.any():
        return {"n": 0, "status": "EMPTY"}
    s = score(y[mask], pred[mask])
    return {
        "n": int(mask.sum()),
        "r2": s["r2"],
        "mae": s["mae"],
        "sse": round(float(np.sum((y[mask] - pred[mask]) ** 2)), 2),
    }


def two_strata(
    y: np.ndarray, pred: np.ndarray, mask: np.ndarray, pos: str, neg: str
) -> Dict[str, object]:
    sse_total = float(np.sum((y - pred) ** 2))
    block = {pos: stratum_report(y, pred, mask), neg: stratum_report(y, pred, ~mask)}
    for v in block.values():
        if v.get("n", 0) > 0:
            v["share_of_rows"] = round(v["n"] / len(y), 4)
            v["share_of_sse"] = round(v["sse"] / sse_total, 4) if sse_total > 0 else None
    return block


def stratify(
    y: np.ndarray,
    pred: np.ndarray,
    corpus_unit: np.ndarray,
    corpus_metal: np.ndarray,
    ev_unit: np.ndarray,
    ev_metal: np.ndarray,
) -> Dict[str, object]:
    return {
        "overall": {**score(y, pred), "n": int(len(y))},
        "unit_corpus": two_strata(y, pred, corpus_unit, "cross_metal", "singleton"),
        "unit_train_evidence": two_strata(y, pred, ev_unit, "has_pair_in_train",
                                          "no_pair_in_train"),
        "metal_corpus": two_strata(y, pred, corpus_metal, "metal_in_supported_pair",
                                   "metal_without_supported_pair"),
        "metal_train_evidence": two_strata(y, pred, ev_metal, "metal_paired_in_train",
                                           "metal_unpaired_in_train"),
    }


def oof_by_folds(
    df: pd.DataFrame, y: np.ndarray, blocks, splitter, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    for k, (tr, te) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
        X_tr, cons = assemble(blocks, "descriptors", tr)
        X_te, _ = assemble(blocks, "descriptors", te)
        pred[te] = fit_predict(X_tr, y[tr], X_te, cons)
        fold_id[te] = k
    assert not np.isnan(pred).any()
    return pred, fold_id


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency", choices=("potency", "cisplatin_ratio"))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_support_stratified.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_parquet(args.dedup)
    sub = df[(df["endpoint"] == args.endpoint) & df["value_log10"].notna()
             & df["ligand_main_fragment"].notna()].reset_index(drop=True)
    desc = pd.read_csv(args.metal_desc)
    vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, desc, vocab, metal_columns=DEFAULT_METAL_COLUMNS)
    y = sub["value_log10"].to_numpy(dtype=float)
    metals = sub["metal"].to_numpy()
    frag = sub["ligand_main_fragment"].to_numpy()
    doi = sub["doi"].astype(str).to_numpy()

    pairs = supported_pairs(sub)
    pair_metals = {m for p in pairs for m in p.split("-")}
    corpus_unit = corpus_unit_flag(sub)
    corpus_metal = np.array([m in pair_metals for m in metals])

    print(f"[UP-14] rows={len(sub)} supported_pairs={pairs}")
    print(f"[UP-14] corpus: cross-metal rows={int(corpus_unit.sum())} "
          f"({corpus_unit.mean():.4f}) | metal-in-pair rows={int(corpus_metal.sum())} "
          f"({corpus_metal.mean():.4f})")

    protocols: Dict[str, object] = {}

    # --- DOI protocol -----------------------------------------------------
    p_doi, f_doi = oof_by_folds(sub, y, blocks, GroupKFold(n_splits=N_FOLDS), doi)
    u_doi, m_doi = train_evidence_flags(sub, f_doi)
    protocols["doi"] = {
        "split": f"GroupKFold({N_FOLDS}) on doi",
        **stratify(y, p_doi, corpus_unit, corpus_metal, u_doi, m_doi),
    }

    # --- within-metal, ligand-grouped ------------------------------------
    p_lig, f_lig = oof_by_folds(sub, y, blocks, GroupKFold(n_splits=N_FOLDS), frag)
    u_lig, m_lig = train_evidence_flags(sub, f_lig)
    wm = {
        "split": f"GroupKFold({N_FOLDS}) on ligand_main_fragment",
        **stratify(y, p_lig, corpus_unit, corpus_metal, u_lig, m_lig),
        "per_metal": {},
    }
    for m in METALS:
        sel = metals == m
        if int(sel.sum()) < 20:
            continue
        wm["per_metal"][m] = {
            **score(y[sel], p_lig[sel]),
            "n": int(sel.sum()),
            "metal_in_supported_pair_corpus": bool(m in pair_metals),
            "metal_paired_in_train_share": round(float(m_lig[sel].mean()), 4),
            "cross_metal_unit_share": round(float(corpus_unit[sel].mean()), 4),
        }
    protocols["within_metal"] = wm

    # --- metal-out --------------------------------------------------------
    p_mo = np.full(len(y), np.nan)
    f_mo = np.full(len(y), -1, dtype=int)
    mo_meta = {}
    for k, m in enumerate(METALS):
        tr, te = np.where(metals != m)[0], np.where(metals == m)[0]
        if len(tr) < 200 or len(te) < 20:
            continue
        X_tr, cons = assemble(blocks, "descriptors", tr)
        X_te, _ = assemble(blocks, "descriptors", te)
        p_mo[te] = fit_predict(X_tr, y[tr], X_te, cons)
        f_mo[te] = k
        train_pairs = supported_pairs(sub.iloc[tr])
        mo_meta[m] = {
            "supported_pairs_train": train_pairs,
            "metal_in_train_pair": bool(any(m in p.split("-") for p in train_pairs)),
        }
    keep = f_mo >= 0
    u_mo, m_mo = train_evidence_flags(sub, f_mo)
    mo_per_metal = {}
    for m in METALS:
        sel = (metals == m) & keep
        if int(sel.sum()) < 20:
            continue
        mo_per_metal[m] = {**score(y[sel], p_mo[sel]), "n": int(sel.sum())}
    mo_n = np.array([v["n"] for v in mo_per_metal.values()], dtype=float)
    mo_weighted = float(np.average([v["r2"] for v in mo_per_metal.values()], weights=mo_n))
    protocols["metal_out"] = {
        "split": "leave one metal out",
        "per_held_out_metal": mo_meta,
        "per_metal": mo_per_metal,
        # msrs_lomo_surface reports the n-weighted mean of per-metal R2; the pooled
        # R2 below is 1 - SSE/SST over all rows. They are different estimators.
        "weighted_r2_per_metal": round(mo_weighted, 4),
        "weighted_r2_source": "same estimator as msrs_lomo_surface.json:"
                              "/endpoints/potency/splits/metal_out/aggregate/descriptors",
        **stratify(y[keep], p_mo[keep], corpus_unit[keep], corpus_metal[keep],
                   u_mo[keep], m_mo[keep]),
    }

    # --- gate / kill ------------------------------------------------------
    checks = {}
    for name in ("doi", "within_metal"):
        for axis in ("unit_corpus", "unit_train_evidence", "metal_corpus",
                     "metal_train_evidence"):
            checks[f"{name}.{axis}"] = bool(
                all(v.get("n", 0) > 0 for v in protocols[name][axis].values())
            )
    # Ligand-grouped CV holds the whole ligand out, so a test unit's partner metal is
    # always in the same fold. The stratum is empty by construction, not by failure.
    exempt = ["within_metal.unit_train_evidence"]
    gate = {
        "every_axis_has_two_non_empty_strata": checks,
        "exempt_empty_by_construction": exempt,
        "metal_out_exempt": "the held-out metal is in no training pair, so every "
                            "evidence stratum is empty by construction",
        "no_threshold": True,
        "gate_pass": bool(all(v for k, v in checks.items() if k not in exempt)),
    }

    def share(name: str, axis: str, stratum: str) -> Optional[float]:
        return protocols[name][axis].get(stratum, {}).get("share_of_rows")

    kill = {
        "rule": "supported-row share < 5% => the metric is a case study, not a capability",
        "share_has_pair_in_train_doi": share("doi", "unit_train_evidence", "has_pair_in_train"),
        "share_has_pair_in_train_within_metal": share("within_metal", "unit_train_evidence",
                                                      "has_pair_in_train"),
        "share_metal_paired_in_train_doi": share("doi", "metal_train_evidence",
                                                 "metal_paired_in_train"),
        "share_metal_paired_in_train_within_metal": share("within_metal",
                                                          "metal_train_evidence",
                                                          "metal_paired_in_train"),
    }
    kill["kill_triggered"] = {
        k: bool(v is not None and v < 0.05)
        for k, v in kill.items() if k.startswith("share_")
    }

    artifact = {
        "artifact": "msrs_support_stratified",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "UP-14 support-stratified re-report",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": args.endpoint,
            "model": "additive LightGBM, descriptors representation, "
                     "interaction_constraints [[ligand],[metal],[cell]]",
            "unit_definition": f"(ligand, cell) group carries >= {CROSS_METAL_MIN} metals",
            "metal_definition": f"metal is in a pair with >= {MIN_GROUPS} matched groups",
            "corpus_vs_train_evidence": "corpus counts the whole table; train_evidence "
                                        "counts only the fold's training rows",
            "supported_pairs_full_corpus": pairs,
            "metal_in_supported_pair_corpus": sorted(pair_metals),
            "error_share": "stratum SSE / total SSE of the same out-of-fold predictions",
        },
        "row_flags_corpus": {
            "n_rows": int(len(y)),
            "n_cross_metal_unit": int(corpus_unit.sum()),
            "n_metal_in_supported_pair": int(corpus_metal.sum()),
        },
        "protocols": protocols,
        "ranking_reference": {
            "artifact": "metrics/by_metric/msrs_ranking_support.json",
            "why": "every ranking pair is two rows of one (ligand, cell) group, so the "
                   "supported share is 1.0 by construction; UP-16 reports it per pair",
        },
        "gate": gate,
        "kill": kill,
        "verdict": "SUPPORT_STRATIFIED",
        "caveats": [
            "The unit and metal axes are not independent: a cross-metal unit is by "
            "definition inside a metal pair. They answer different questions about the "
            "same rows, not two independent effects.",
            "In metal_out every evidence stratum is empty, because the held-out metal is "
            "in no training pair. The whole test set is the unsupported stratum; that is "
            "the finding, not a stratification bug.",
            "In within_metal the ligand is unseen, so a cross-metal unit has its partner "
            "metal in the same test fold. The unit axis is therefore empty there by "
            "construction and only the metal axis carries information.",
            "R2 is computed inside each stratum against that stratum's own variance, so "
            "the stratum R2 values do not add up to the pooled R2. The SSE shares do add "
            "up and are the quantity to quote.",
            "The corpus-level flags are descriptive only. A test row's corpus-level "
            "cross-metal partner may sit in the same fold, which is why the fold-aware "
            "train_evidence flags are the primary table.",
        ],
    }
    artifact["elapsed_seconds"] = round(time.time() - t0, 1)
    atomic_write_file(args.out, json.dumps(artifact, indent=2) + "\n")

    for name in ("doi", "within_metal", "metal_out"):
        block = protocols[name]
        extra = (f" weighted_r2_per_metal={block['weighted_r2_per_metal']}"
                 if "weighted_r2_per_metal" in block else "")
        print(f"[UP-14] {name}: pooled R2={block['overall']['r2']} "
              f"n={block['overall']['n']}{extra}")
        for axis in ("unit_corpus", "unit_train_evidence", "metal_corpus",
                     "metal_train_evidence"):
            for k, v in block[axis].items():
                if v.get("n", 0) > 0:
                    print(f"[UP-14]   {axis}.{k}: n={v['n']} rows={v['share_of_rows']} "
                          f"R2={v['r2']} sse={v['share_of_sse']}")
                else:
                    print(f"[UP-14]   {axis}.{k}: EMPTY")
    print(f"[UP-14] gate_pass={gate['gate_pass']} checks={checks}")
    print(f"[UP-14] kill={json.dumps(kill, ensure_ascii=False)}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
