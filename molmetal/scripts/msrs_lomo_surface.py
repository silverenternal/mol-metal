#!/usr/bin/env python3
"""MSRS-4 — can the surface extrapolate to a metal it has never seen?

=====================================================================
The question
=====================================================================
WP3 measured that adding metal descriptors to a classifier bought +0.0031 AUC
and that metal-as-domain-token lost on 8 of 9 arms. That was a *classification*
answer. This is the continuous version:

    train on four metals, predict the fifth's log10 IC50, and ask whether any
    metal representation beats ignoring the metal entirely.

Three representations, same learner and same ligand/cell features:

    descriptors   IE1, IE2, Pauling EN — the only block that can represent a
                  metal absent from training
    one-hot       five columns; the held-out column is all zeros at test, so
                  the model has never seen that row pattern
    none          the metal is dropped; the ligand effect is applied unchanged

=====================================================================
Two splits
=====================================================================
    metal-out             train on the other four metals, all ligands
    metal-out + ligand-disjoint
                          additionally drop training rows whose ligand appears
                          in the held-out metal. 2.7% of units are cross-metal,
                          so this removes a small but privileged overlap and is
                          the honest "new metal, new ligand" test.

=====================================================================
What counts as success
=====================================================================
Beating the training mean is necessary but weak: the ligand features alone do
that. The metal-specific claim is that the representation places the unseen
metal's *level* correctly. So the report includes, per held-out metal:

    bias = mean(y_hat - y)
    rank correlation between the five predicted and observed metal means
    MAE against the oracle held-out-metal mean (an upper bound that uses the
      held-out labels and is labelled as such)

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_lomo_surface.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    SEED,
    TOP_CELL_LINES,
    cell_matrix,
    fit_predict,
    metal_matrix,
    morgan_matrix,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ENDPOINTS = ("potency", "cisplatin_ratio")
REPRESENTATIONS = ("descriptors", "one_hot", "none")
# The descriptor block every pre-2026-10-21 artifact used.  UP-4 overrides it
# with an eRAC block via --metal-cols; the default keeps old runs byte-identical.
DEFAULT_METAL_COLUMNS = ("ie1_kj", "ie2_kj", "pauling_en")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Design matrices per representation
# ---------------------------------------------------------------------------
def build_blocks(
    df: pd.DataFrame,
    descriptors: pd.DataFrame,
    vocabulary: Sequence[str],
    metal_columns: Sequence[str] = DEFAULT_METAL_COLUMNS,
) -> Dict[str, sparse.csr_matrix]:
    lig = morgan_matrix(df["ligand_main_fragment"].tolist())
    met_full = sparse.csr_matrix(
        metal_matrix(df["metal"].tolist(), descriptors, columns=metal_columns)
    )
    cell = cell_matrix(df["cell_line_key"].tolist(), vocabulary)
    return {
        "ligand": lig,
        "metal_one_hot": met_full[:, : len(METALS)],
        "metal_descriptors": met_full[:, len(METALS) :],
        "cell": cell,
    }


def assemble(
    blocks: Dict[str, sparse.csr_matrix], representation: str, rows: np.ndarray
) -> Tuple[sparse.csr_matrix, List[List[int]]]:
    lig = blocks["ligand"][rows]
    cell = blocks["cell"][rows]
    if representation == "descriptors":
        metal = blocks["metal_descriptors"][rows]
        groups = [list(range(lig.shape[1])),
                  list(range(lig.shape[1], lig.shape[1] + metal.shape[1])),
                  list(range(lig.shape[1] + metal.shape[1], lig.shape[1] + metal.shape[1] + cell.shape[1]))]
    elif representation == "one_hot":
        metal = blocks["metal_one_hot"][rows]
        groups = [list(range(lig.shape[1])),
                  list(range(lig.shape[1], lig.shape[1] + metal.shape[1])),
                  list(range(lig.shape[1] + metal.shape[1], lig.shape[1] + metal.shape[1] + cell.shape[1]))]
    elif representation == "none":
        metal = None
        groups = [list(range(lig.shape[1])),
                  list(range(lig.shape[1], lig.shape[1] + cell.shape[1]))]
    else:
        raise ValueError(representation)
    X = sparse.hstack([lig] + ([metal] if metal is not None else []) + [cell], format="csr")
    return X, groups


# ---------------------------------------------------------------------------
# One metal-out evaluation
# ---------------------------------------------------------------------------
def evaluate_split(
    df: pd.DataFrame,
    blocks: Dict[str, sparse.csr_matrix],
    y: np.ndarray,
    holdout: str,
    ligand_disjoint: bool,
    verbose: bool = True,
) -> Dict[str, object]:
    metals = df["metal"].to_numpy()
    fragments = df["ligand_main_fragment"].to_numpy()
    train = np.where(metals != holdout)[0]
    test = np.where(metals == holdout)[0]

    if ligand_disjoint:
        held = set(fragments[test])
        keep = np.array([f not in held for f in fragments[train]])
        train = train[keep]

    if len(train) < 200 or len(test) < 20:
        return {"status": "INSUFFICIENT", "n_train": int(len(train)), "n_test": int(len(test))}

    preds: Dict[str, np.ndarray] = {}
    for rep in REPRESENTATIONS:
        X_tr, groups = assemble(blocks, rep, train)
        X_te, _ = assemble(blocks, rep, test)
        preds[rep] = fit_predict(X_tr, y[train], X_te, groups)

    y_test = y[test]
    train_mean = float(np.mean(y[train]))
    preds["train_mean"] = np.full(len(test), train_mean)
    preds["oracle_metal_mean"] = np.full(len(test), float(np.mean(y_test)))

    rows: Dict[str, object] = {}
    for name, pred in preds.items():
        rows[name] = {
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "r2": round(float(r2_score(y_test, pred)), 4),
            "bias": round(float(np.mean(pred - y_test)), 4),
        }

    if verbose:
        print(
            f"  {holdout:>2s} n_train={len(train):5d} n_test={len(test):4d}  "
            + "  ".join(f"{r}={rows[r]['mae']:.3f}" for r in REPRESENTATIONS)
        )

    return {
        "status": "OK",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_ligands_test": int(len(set(fragments[test]))),
        "train_mean": round(train_mean, 4),
        "observed_mean": round(float(np.mean(y_test)), 4),
        "models": rows,
        "pred_mean_by_representation": {
            r: round(float(np.mean(preds[r])), 4) for r in REPRESENTATIONS
        },
        "abs_err": {r: np.abs(y_test - preds[r]) for r in REPRESENTATIONS},
        "test_idx": test,
    }


def aggregate_split(per_metal: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    ok = {m: v for m, v in per_metal.items() if v.get("status") == "OK"}
    if not ok:
        return {}
    n = np.array([ok[m]["n_test"] for m in ok], dtype=float)
    out: Dict[str, object] = {}
    for rep in REPRESENTATIONS:
        mae = np.array([ok[m]["models"][rep]["mae"] for m in ok])
        r2 = np.array([ok[m]["models"][rep]["r2"] for m in ok])
        bias = np.array([ok[m]["models"][rep]["bias"] for m in ok])
        out[rep] = {
            "weighted_mae": round(float(np.average(mae, weights=n)), 4),
            "weighted_r2": round(float(np.average(r2, weights=n)), 4),
            "mean_abs_bias": round(float(np.mean(np.abs(bias))), 4),
            "max_abs_bias": round(float(np.max(np.abs(bias))), 4),
            "n_metals_negative_r2": int((r2 < 0).sum()),
        }
    return out


def rank_agreement(per_metal: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    ok = {m: v for m, v in per_metal.items() if v.get("status") == "OK"}
    observed = np.array([ok[m]["observed_mean"] for m in ok])
    out: Dict[str, object] = {}
    for rep in REPRESENTATIONS:
        predicted = np.array([ok[m]["pred_mean_by_representation"][rep] for m in ok])
        rho, p = stats.spearmanr(observed, predicted)
        out[rep] = {
            "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
            "p": round(float(p), 4) if np.isfinite(p) else None,
            "n_metals": int(len(ok)),
            "predicted_means": {m: ok[m]["pred_mean_by_representation"][rep] for m in ok},
            "observed_means": {m: ok[m]["observed_mean"] for m in ok},
        }
    return out


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def paired_tests(per_metal: Dict[str, Dict[str, object]], fragments: np.ndarray) -> Dict[str, object]:
    """Per-ligand mean absolute error, paired across representations."""
    ok = [r for r in per_metal.values() if r.get("status") == "OK"]
    if not ok:
        return {}
    idx = np.concatenate([r["test_idx"] for r in ok])
    lig = fragments[idx]
    frame = pd.DataFrame({"lig": lig, **{f"e_{rep}": np.concatenate([r["abs_err"][rep] for r in ok]) for rep in REPRESENTATIONS}})
    per = frame.groupby("lig").mean()
    out: Dict[str, object] = {}
    for a, b in (("descriptors", "none"), ("descriptors", "one_hot")):
        diff = per[f"e_{a}"] - per[f"e_{b}"]
        if float(diff.std(ddof=1)) == 0.0:
            t, p_one = 0.0, 1.0
        else:
            t, p_two = stats.ttest_rel(per[f"e_{a}"], per[f"e_{b}"])
            p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
        out[f"{a}_vs_{b}"] = {
            "n_ligands": int(len(per)),
            "mean_mae_a": round(float(per[f"e_{a}"].mean()), 4),
            "mean_mae_b": round(float(per[f"e_{b}"].mean()), 4),
            "mean_delta": round(float(diff.mean()), 4),
            "t": round(float(t), 4),
            "p_one_sided_b_better": p_one,
            "p_one_sided_a_better": round(1.0 - p_one, 8),
            "direction": "b better" if diff.mean() > 0 else "a better",
        }
    return out


# ---------------------------------------------------------------------------
# One endpoint
# ---------------------------------------------------------------------------
def run_endpoint(
    df: pd.DataFrame,
    endpoint: str,
    descriptors: pd.DataFrame,
    verbose: bool = True,
    metal_columns: Sequence[str] = DEFAULT_METAL_COLUMNS,
) -> Dict[str, object]:
    t0 = time.time()
    sub = df[df["endpoint"] == endpoint]
    n_total = len(sub)
    n_censored = int(sub["value_log10"].isna().sum())
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return {"status": "EMPTY"}

    vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, descriptors, vocabulary, metal_columns=metal_columns)
    y = sub["value_log10"].to_numpy(dtype=float)

    out: Dict[str, object] = {
        "status": "OK",
        "n_rows": int(len(sub)),
        "n_dropped_censored": n_censored,
        "censored_share": round(n_censored / n_total, 4) if n_total else None,
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "design_facts": {
            "n_dois": int(sub["doi"].nunique()),
            "dois_with_multiple_metals": int(
                sub.groupby("doi")["metal"].nunique().gt(1).sum()
            ),
            "share_units_in_multi_metal_dois": round(
                float(sub.groupby("doi")["metal"].transform("nunique").gt(1).mean()), 4
            ),
        },
        "splits": {},
    }

    if verbose:
        print(f"[{endpoint}] n={len(sub)} ligands={sub['ligand_main_fragment'].nunique()}")

    for split_name, disjoint in (("metal_out", False), ("metal_out_ligand_disjoint", True)):
        per_metal: Dict[str, Dict[str, object]] = {}
        if verbose:
            print(f" split={split_name}")
        for metal in METALS:
            per_metal[metal] = evaluate_split(sub, blocks, y, metal, disjoint, verbose)
        out["splits"][split_name] = {
            "per_metal": {
                m: {k: v for k, v in r.items() if k not in ("abs_err", "test_idx")}
                for m, r in per_metal.items()
            },
            "aggregate": aggregate_split(per_metal),
            "rank_agreement": rank_agreement(per_metal),
            "paired_tests": paired_tests(per_metal, sub["ligand_main_fragment"].to_numpy()),
        }
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_lomo_surface.json"
    )
    parser.add_argument(
        "--metal-cols",
        default=",".join(DEFAULT_METAL_COLUMNS),
        help="comma-separated descriptor columns for the 'descriptors' arm "
             "(default: the pre-2026-10-21 block)",
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    metal_columns = tuple(c.strip() for c in str(args.metal_cols).split(",") if c.strip())
    missing = [c for c in metal_columns if c not in descriptors.columns]
    if missing:
        print(f"FAIL: --metal-cols not in {args.metal_desc}: {missing}")
        return 1
    results = {
        ep: run_endpoint(df, ep, descriptors, metal_columns=metal_columns)
        for ep in ENDPOINTS
    }

    # Holm across the four endpoint x split cells. Two pre-specified directions:
    # "the baseline is better" (does the metal block hurt?) and "the metal block
    # is better" (does it add anything?).
    for contrast in ("descriptors_vs_none", "descriptors_vs_one_hot"):
        for tail, field, out_key in (
            ("b", "p_one_sided_b_better", "holm_p_baseline_better"),
            ("a", "p_one_sided_a_better", "holm_p_metal_block_better"),
        ):
            raw = {}
            for ep, r in results.items():
                for split, v in r.get("splits", {}).items():
                    raw[f"{ep}|{split}"] = v["paired_tests"][contrast][field]
            adj = holm(raw)
            for key, p in adj.items():
                ep, split = key.split("|")
                results[ep]["splits"][split]["paired_tests"][contrast][out_key] = p

    out = {
        "artifact": "msrs_lomo_surface",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 metal extrapolation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc),
            "metal_descriptor_columns": list(metal_columns),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "learner": "additive LightGBM from MSRS-2 (interaction_constraints on ligand/metal/cell)",
            "representations": list(REPRESENTATIONS),
            "splits": ["metal_out", "metal_out_ligand_disjoint"],
            "seed": SEED,
            "oracle_note": "oracle_metal_mean uses the held-out metal's labels and is an upper bound, not a model.",
        },
        "endpoints": results,
        "caveats": [
            "The additive structure is imposed because MSRS-2 showed it ranks metals better. "
            "A metal-specific slope is therefore not available to the extrapolation.",
            "Three descriptors cannot span five metals; a poor rank correlation is a property of "
            "the descriptor table, not of the learner.",
            "Censored rows are dropped (potency 18.4%), so the extrapolation is fitted on the "
            "potent side of each metal's cohort.",
            "LOMO evaluates one metal at a time; the five test cohorts differ in size and in "
            "ligand chemistry, so the per-metal numbers are not exchangeable.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
