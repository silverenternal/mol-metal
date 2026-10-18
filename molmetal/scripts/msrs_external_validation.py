#!/usr/bin/env python
"""MSRS-7 — external / temporal / scaffold validation of the response surface.

The MSRS-2 numbers come from a random, ligand-grouped cross-validation.  That is
enough to show the surface fits, not enough to show it generalises.  This script
re-runs the same two models under three leakage-controlled splits and reports the
decay against the random split:

* ``doi``      — GroupKFold(5) on the DOI.  No paper appears in both train and
  test, so assay protocol, cell-line panel and compound series cannot leak.
  Ligands may still recur across papers, which is the realistic deployment case.
* ``year``     — leave-one-year-out on the year parsed from the DOI.  Papers from
  the held-out year are unseen in time.  Only 30.6% of potency rows carry a
  parseable year in the DOI, so this cell is evaluated on that subset and the
  coverage is reported.  Years with < 100 rows are pooled into one early group,
  otherwise a 4-row holdout would produce a meaningless fold.
* ``scaffold`` — GroupKFold(5) on the Bemis-Murcko scaffold of the ligand.  No
  scaffold appears in both train and test, so the model cannot memorise a ring
  system.

What this cannot do: none of the three splits is a metal holdout, so all of them
validate *interpolation* within the five metals.  MSRS-4 already showed that the
metal itself does not extrapolate, and nothing here changes that.

Usage::

    .venv/bin/python molmetal/scripts/msrs_external_validation.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    LGB_PARAMS,
    METALS,
    N_FOLDS,
    NOISE_FLOOR,
    SEED,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    build_design,
    delta_sign_report,
    holm,
    matched_delta_pairs,
    oof_means,
    paired_ligand_test,
    score,
)

ENDPOINTS = ("potency", "cisplatin_ratio")
SCHEMES = ("random", "doi", "year", "scaffold")
MIN_YEAR_ROWS = 100
EARLY_YEAR_LABEL = "early"

# A year must be delimited by . _ / ( or [ and be followed by a delimiter or the
# end.  Without the delimiters a volume or page number matches by accident.
_YEAR_RE = re.compile(r"(?:[._/(\[])((?:19|20)\d{2})(?=[._)/\]\-]|$)")


# ---------------------------------------------------------------------------
# Grouping keys
# ---------------------------------------------------------------------------
def parse_year(doi: object) -> Optional[int]:
    """Publication year from a DOI, or None when the DOI carries no year."""
    if not isinstance(doi, str):
        return None
    for match in _YEAR_RE.finditer(doi):
        year = int(match.group(1))
        if 1950 <= year <= 2026:
            return year
    return None


def year_group(doi: object, counts: Dict[int, int], min_rows: int = MIN_YEAR_ROWS) -> Optional[str]:
    """Year label, pooling sparse years into one group; None when unparseable."""
    year = parse_year(doi)
    if year is None:
        return None
    return str(year) if counts.get(year, 0) >= min_rows else EARLY_YEAR_LABEL


_SCAFFOLD_CACHE: Dict[str, str] = {}


def murcko_scaffold(smiles: object) -> str:
    """Bemis-Murcko scaffold; acyclic molecules collapse to ``ACYCLIC``."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if not isinstance(smiles, str):
        return "UNPARSED"
    if smiles in _SCAFFOLD_CACHE:
        return _SCAFFOLD_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out = "UNPARSED"
    else:
        out = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC"
    _SCAFFOLD_CACHE[smiles] = out
    return out


def fold_ids(df: pd.DataFrame, scheme: str, n_splits: int = N_FOLDS) -> np.ndarray:
    """Fold assignment for one scheme.  Rows with no group key get -1."""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    if scheme == "random":
        groups = df["ligand_main_fragment"].to_numpy()
    elif scheme == "doi":
        groups = df["doi"].astype(str).to_numpy()
    elif scheme == "scaffold":
        groups = df["scaffold"].to_numpy()
    elif scheme == "year":
        groups = df["year_group"].to_numpy()
    else:
        raise ValueError(f"unknown scheme {scheme}")

    valid = np.array([g is not None and str(g) != "nan" for g in groups])
    folds = np.full(len(df), -1, dtype=int)
    idx = np.flatnonzero(valid)
    sub_groups = groups[valid]
    if scheme == "year":
        splitter = LeaveOneGroupOut()
    else:
        n_groups = len(np.unique(sub_groups))
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    for k, (_, te) in enumerate(splitter.split(np.zeros((len(idx), 1)), groups=sub_groups)):
        folds[idx[te]] = k
    return folds


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
def oof_by_folds(
    X: sparse.csr_matrix,
    y: np.ndarray,
    fold_id: np.ndarray,
    constraints: Optional[List[List[int]]],
) -> np.ndarray:
    """Fit one model per fold id; rows with fold_id < 0 are left as NaN."""
    from lightgbm import LGBMRegressor

    oof = np.full(len(y), np.nan)
    params = dict(LGB_PARAMS)
    if constraints is not None:
        params["interaction_constraints"] = constraints
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = (fold_id >= 0) & (fold_id != k)
        te = fold_id == k
        model = LGBMRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return oof


def _baseline_by_folds(
    y: np.ndarray, metals: np.ndarray, fold_id: np.ndarray
) -> np.ndarray:
    """Per-metal mean from the training folds of each fold id."""
    out = np.full(len(y), np.nan)
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = (fold_id >= 0) & (fold_id != k)
        te = fold_id == k
        overall = float(np.mean(y[tr]))
        means = pd.Series(y[tr]).groupby(metals[tr]).mean()
        out[te] = np.array([means.get(m, overall) for m in metals[te]], dtype=float)
    return out


def _decision_utility(
    frame: pd.DataFrame,
    pred_add: np.ndarray,
    pred_int: np.ndarray,
    pred_metal: np.ndarray,
) -> Dict[str, object]:
    """Top-1 regret per ligand cohort, under whatever split produced the predictions.

    This is the MSRS-6 question re-asked on the controlled splits.  It matters
    because MSRS-7 shows the interaction arm can win the delta sign here even
    though it lost it on the random split; the decision layer only cares whether
    that turns into lower regret than the fold-safe global metal order.
    """
    from molmetal.scripts.msrs_recommendation import build_cohorts, cohort_metrics

    if frame.empty:
        return {"status": "NO_COHORTS"}
    work = frame.assign(
        cohort=frame.groupby(
            ["ligand_canonical", "cell_line_key", "doi", "time_h"], dropna=False, sort=False
        ).ngroup(),
        fold=0,  # carried through build_cohorts; the regret metrics do not use it
        additive=pred_add,
        interaction=pred_int,
        global_best=pred_metal,
    )
    cohorts = build_cohorts(work, ("additive", "interaction", "global_best"))
    if cohorts.empty:
        return {"status": "NO_COHORTS"}

    rows = []
    for cid, g in cohorts.groupby("cohort", sort=True):
        y_obs = g["y"].to_numpy()
        for scorer in ("additive", "interaction", "global_best"):
            m = cohort_metrics(y_obs, g[scorer].to_numpy())
            rows.append({"cohort": int(cid), "scorer": scorer, "regret": m["regret"],
                         "top1_correct": m["top1_correct"]})
    rec = pd.DataFrame(rows)

    out: Dict[str, object] = {"n_cohorts": int(cohorts["cohort"].nunique())}
    for scorer in ("additive", "interaction", "global_best"):
        sub = rec[rec["scorer"] == scorer]
        out[scorer] = {
            "mean_regret": round(float(sub["regret"].mean()), 4),
            "top1_accuracy": round(float(sub["top1_correct"].mean()), 4),
        }
    wide = rec.pivot(index="cohort", columns="scorer", values="regret")
    for scorer in ("additive", "interaction"):
        diff = (wide["global_best"] - wide[scorer]).dropna()
        if len(diff) >= 2 and float(diff.std(ddof=1)) > 0:
            t, p_two = stats.ttest_1samp(diff.to_numpy(), 0.0)
            p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
        else:
            t, p_one = 0.0, 1.0
        out[f"{scorer}_vs_global_best"] = {
            "mean_regret_delta": round(float(diff.mean()), 4) if len(diff) else None,
            "t": round(float(t), 4),
            "p_one_sided": p_one,
            "direction": "model better" if len(diff) and diff.mean() > 0 else "baseline better",
        }
    return out


# ---------------------------------------------------------------------------
# One (endpoint, scheme) cell
# ---------------------------------------------------------------------------
def evaluate_cell(
    df: pd.DataFrame,
    endpoint: str,
    scheme: str,
    descriptors: pd.DataFrame,
    verbose: bool = True,
    min_year_rows: int = MIN_YEAR_ROWS,
) -> Dict[str, object]:
    t0 = time.time()
    sub = df[df["endpoint"] == endpoint].copy()
    n_total = len(sub)
    n_censored = int(sub["value_log10"].isna().sum())
    sub = sub[
        sub["value_log10"].notna()
        & sub["ligand_main_fragment"].notna()
        & sub["ligand_canonical"].notna()
    ].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return {"status": "EMPTY"}

    sub["year_group"] = None
    year_counts = {}
    if scheme == "year":
        parsed = sub["doi"].map(parse_year)
        year_counts = {int(k): int(v) for k, v in parsed.value_counts().items()}
        sub["year_group"] = sub["doi"].map(lambda d: year_group(d, year_counts, min_year_rows))
    if scheme == "scaffold":
        sub["scaffold"] = sub["ligand_main_fragment"].map(murcko_scaffold)

    fold_id = fold_ids(sub, scheme)
    keep = fold_id >= 0
    n_unassigned = int((~keep).sum())
    if keep.sum() < 100:
        return {"status": "TOO_FEW_ROWS", "n_assigned": int(keep.sum())}
    if len(set(fold_id[keep].tolist())) < 2:
        return {"status": "TOO_FEW_GROUPS", "n_groups": len(set(fold_id[keep].tolist()))}

    cell_vocab = sub.loc[keep, "cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, _, groups = build_design(sub, descriptors, cell_vocab)
    y = sub["value_log10"].to_numpy(dtype=float)
    metals = sub["metal"].to_numpy()

    pred_add = oof_by_folds(X, y, fold_id, groups)
    pred_int = oof_by_folds(X, y, fold_id, None)
    pred_metal = _baseline_by_folds(y, metals, fold_id)

    mask = keep & ~np.isnan(pred_add) & ~np.isnan(pred_int) & ~np.isnan(pred_metal)
    if mask.sum() < 100:
        return {"status": "TOO_FEW_ROWS", "n_assigned": int(mask.sum())}

    models = {}
    for name, pred in (("additive", pred_add), ("interaction", pred_int), ("per_metal_mean", pred_metal)):
        entry = score(y[mask], pred[mask])
        entry["mae_by_metal"] = {
            m: round(float(np.mean(np.abs(y[mask & (metals == m)] - pred[mask & (metals == m)]))), 4)
            for m in METALS
            if (mask & (metals == m)).sum() > 0
        }
        models[name] = entry

    err = {
        "additive": np.abs(y[mask] - pred_add[mask]),
        "interaction": np.abs(y[mask] - pred_int[mask]),
        "per_metal_mean": np.abs(y[mask] - pred_metal[mask]),
    }
    sub_mask = sub[mask].reset_index(drop=True)
    paired = {
        "interaction_vs_additive": paired_ligand_test(sub_mask, err["additive"], err["interaction"]),
        "additive_vs_per_metal_mean": paired_ligand_test(sub_mask, err["per_metal_mean"], err["additive"]),
    }

    pairs = matched_delta_pairs(sub_mask)
    delta = delta_sign_report(
        y[mask], pred_add[mask], pred_int[mask], pairs, fold_id[mask]
    )
    decision = _decision_utility(
        sub_mask.assign(y=y[mask]), pred_add[mask], pred_int[mask], pred_metal[mask]
    )

    # Per-fold detail: the year cell is the only one where the fold label itself
    # carries meaning, but the spread is informative for every scheme.
    fold_detail = {}
    for k in sorted(set(fold_id[mask].tolist())):
        f = mask & (fold_id == k)
        if f.sum() == 0:
            continue
        label = str(int(k))
        if scheme == "year":
            label = str(sub.loc[f, "year_group"].iloc[0])
        fold_detail[label] = {
            "n_test": int(f.sum()),
            "n_train": int(((fold_id >= 0) & (fold_id != k)).sum()),
            "mae_additive": round(float(np.mean(np.abs(y[f] - pred_add[f]))), 4),
            "mae_interaction": round(float(np.mean(np.abs(y[f] - pred_int[f]))), 4),
            "mae_per_metal_mean": round(float(np.mean(np.abs(y[f] - pred_metal[f]))), 4),
        }

    if verbose:
        prim = delta.get("by_threshold", {}).get(f"{NOISE_FLOOR:g}", {}) if delta.get("n_pairs") else {}
        print(
            f"[{endpoint}/{scheme}] n={int(mask.sum())} folds={len(fold_detail)} "
            f"add={models['additive']['mae']} int={models['interaction']['mae']} "
            f"metal={models['per_metal_mean']['mae']} "
            f"delta_sign add={prim.get('additive_accuracy')} int={prim.get('interaction_accuracy')} "
            f"regret add={decision.get('additive', {}).get('mean_regret')} "
            f"int={decision.get('interaction', {}).get('mean_regret')} "
            f"global={decision.get('global_best', {}).get('mean_regret')}"
        )

    group_key = {
        "random": "ligand_main_fragment",
        "doi": "doi",
        "year": "year_group",
        "scaffold": "scaffold",
    }[scheme]
    group_series = sub.loc[keep, group_key]

    return {
        "status": "OK",
        "n_rows_total": n_total,
        "n_dropped_censored": n_censored,
        "n_evaluated": int(mask.sum()),
        "n_unassigned": n_unassigned,
        "assignment_coverage": round(float(mask.sum() / max(1, len(sub))), 4),
        "n_folds": int(len(fold_detail)),
        "group_counts": {
            "key": group_key,
            "n_groups": int(group_series.nunique()),
            "largest_group_rows": int(group_series.value_counts().iloc[0]),
        },
        "models": models,
        "paired_tests": paired,
        "delta_sign": delta,
        "decision_utility": decision,
        "folds": fold_detail,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


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
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_external_validation.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results: Dict[str, Dict[str, object]] = {}
    for endpoint in ENDPOINTS:
        results[endpoint] = {}
        for scheme in SCHEMES:
            results[endpoint][scheme] = evaluate_cell(df, endpoint, scheme, descriptors)

    # Decay is measured against the random split of the same endpoint, so the
    # comparison is exact rather than against a number copied from MSRS-2.
    for endpoint, cells in results.items():
        ref = cells.get("random", {})
        if ref.get("status") != "OK":
            continue
        for scheme, cell in cells.items():
            if cell.get("status") != "OK":
                continue
            cell["decay_vs_random"] = {
                "mae_additive_delta": round(
                    cell["models"]["additive"]["mae"] - ref["models"]["additive"]["mae"], 4
                ),
                "mae_interaction_delta": round(
                    cell["models"]["interaction"]["mae"] - ref["models"]["interaction"]["mae"], 4
                ),
                "r2_additive_delta": round(
                    cell["models"]["additive"]["r2"] - ref["models"]["additive"]["r2"], 4
                ),
                "note": "positive MAE delta = worse than the random ligand-grouped split",
            }

    # Holm across the 8 (endpoint x scheme) interaction-vs-additive tests.
    raw = {}
    for endpoint, cells in results.items():
        for scheme, cell in cells.items():
            if cell.get("status") != "OK":
                continue
            raw[f"{endpoint}/{scheme}"] = cell["paired_tests"]["interaction_vs_additive"]["p_one_sided"]
    adjusted = holm(raw)
    for key, p in adjusted.items():
        endpoint, scheme = key.split("/")
        results[endpoint][scheme]["paired_tests"]["interaction_vs_additive"]["holm_p"] = p

    summary = {}
    for endpoint, cells in results.items():
        summary[endpoint] = {
            scheme: {
                "n_evaluated": cell.get("n_evaluated"),
                "n_folds": cell.get("n_folds"),
                "mae_additive": cell.get("models", {}).get("additive", {}).get("mae"),
                "mae_interaction": cell.get("models", {}).get("interaction", {}).get("mae"),
                "mae_per_metal_mean": cell.get("models", {}).get("per_metal_mean", {}).get("mae"),
                "r2_additive": cell.get("models", {}).get("additive", {}).get("r2"),
                "delta_sign_additive": cell.get("delta_sign", {})
                .get("by_threshold", {})
                .get(f"{NOISE_FLOOR:g}", {})
                .get("additive_accuracy"),
                "delta_sign_interaction": cell.get("delta_sign", {})
                .get("by_threshold", {})
                .get(f"{NOISE_FLOOR:g}", {})
                .get("interaction_accuracy"),
                "mae_additive_delta_vs_random": cell.get("decay_vs_random", {}).get("mae_additive_delta"),
                "regret_additive": cell.get("decision_utility", {}).get("additive", {}).get("mean_regret"),
                "regret_interaction": cell.get("decision_utility", {}).get("interaction", {}).get("mean_regret"),
                "regret_global_best": cell.get("decision_utility", {}).get("global_best", {}).get("mean_regret"),
            }
            for scheme, cell in cells.items()
        }

    out = {
        "artifact": "msrs_external_validation",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "splits": {
                "random": f"GroupKFold({N_FOLDS}) on ligand_main_fragment — the MSRS-2 protocol",
                "doi": f"GroupKFold({N_FOLDS}) on doi — no paper in both train and test",
                "year": f"LeaveOneGroupOut on the year parsed from the DOI; years with < {MIN_YEAR_ROWS} rows pooled as '{EARLY_YEAR_LABEL}'",
                "scaffold": f"GroupKFold({N_FOLDS}) on the Bemis-Murcko scaffold",
            },
            "models": ["additive (interaction constraints)", "interaction (no constraints)", "per_metal_mean"],
            "primary_metric": "MAE in log10 units; delta sign accuracy at the MSRS-1 noise floor",
            "decay_reference": "the random split of the same endpoint, recomputed here",
            "seed": SEED,
            "lightgbm": LGB_PARAMS,
            "scope": "all three splits hold out papers, years or scaffolds, never a metal; this validates interpolation only",
        },
        "summary": summary,
        "endpoints": results,
        "caveats": [
            "Only 30.6% of potency rows (29.9% of cisplatin_ratio) carry a parseable year in the DOI, so the year cell is a subset and is not directly comparable in n.",
            "The year split holds out whole papers by publication year, but a paper's DOI year can predate its print issue; treat the labels as approximate.",
            "The DOI split still lets the same ligand appear in train and test under a different paper, which is realistic but not a chemistry holdout.",
            "The scaffold split removes ring-system memorisation but keeps ligand decoration patterns; it is not a full analogue-series holdout.",
            "No split holds out a metal. MSRS-4 covers that question and the answer there is negative.",
            "The decision-utility block re-asks the MSRS-6 question on these splits but was not the "
            "pre-registered primary analysis; MSRS-6's kill criterion was fixed on the random split "
            "and is not reopened post hoc here.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
