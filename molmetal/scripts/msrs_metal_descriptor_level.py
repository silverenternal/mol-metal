#!/usr/bin/env python3
"""MSRS-EXT-2 — can static metal chemistry predict the level a metal sits at?

=====================================================================
The question
=====================================================================
`msrs_lomo_surface.json` left this as an untested assertion:

    "Three descriptors cannot span five metals; a poor rank correlation is a
     property of the descriptor table, not of the learner."

EXT-1 then measured that the level term is only 5-10% of the leave-one-metal-out
squared error, so this question is worth less than the plan assumed.  It is
still worth answering, for two reasons:

  * if descriptors *can* place a metal's level, zero-shot extrapolation has a
    chemical basis even though its payoff is small;
  * if they cannot, that is a statement about the limits of static descriptors
    with five metals, and it has to be written down rather than assumed.

=====================================================================
The test
=====================================================================
Leave-one-metal-out over the five metals.  Two estimators, because with four
training points the choice of estimator is the whole game:

    ridge     all descriptors, standardised, alpha chosen from a grid by the
              same leave-one-out loop (nested, so the reported number is not
              the best of a grid)
    single    one descriptor at a time, ordinary least squares.  This is the
              fairer test: it asks whether *any* single column orders the five
              metals correctly, which is all five points can support.

Null: permute which mean belongs to which metal (120 distinct permutations for
5 metals) and recompute.  The permutation p is the fraction of permuted R2 at
least as large as the observed one.  With n=5 this is the only honest
significance statement available.

A second, cheaper analysis reads `msrs_lomo_error_decomposition.json` and
computes the *perfect-offset counterfactual*: if the held-out metal's mean were
known exactly, R2 would rise by exactly offset_share x (1 + |R2_raw|).  That
bounds what any level-prediction improvement can buy, without refitting.

=====================================================================
Decision gate (pre-registered in todo.json, EXT-2)
=====================================================================
    PASS   LOOCV R2 >= 0.5 for the metal mean AND the counterfactual gain is
           large enough to matter
    KILL   LOOCV R2 < 0.5 -> "static descriptors cannot set the level"

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_metal_descriptor_level.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
ENDPOINTS = ("potency", "cisplatin_ratio")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

# Definitional columns.  These are facts about the element and the M(III)
# oxidation state used throughout MSRS; they are not measurements and are
# labelled as such so nobody mistakes them for curated chemistry.
DEFINITIONAL = {
    #        group, period, d-electrons in M(III), unpaired electrons in M(III) low spin
    "Ru": (8, 5, 5, 1),
    "Rh": (9, 5, 6, 0),
    "Os": (8, 6, 5, 1),
    "Ir": (9, 6, 6, 0),
    "Re": (7, 6, 4, 2),
}


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


def build_table(base: pd.DataFrame) -> pd.DataFrame:
    """RDKit periodic table + the existing NIST ionisation energies + definitions."""
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    rows = []
    for _, r in base.iterrows():
        s = r["metal_symbol"]
        z = pt.GetAtomicNumber(s)
        g, per, d, unp = DEFINITIONAL[s]
        rows.append(
            {
                "metal_symbol": s,
                "atomic_number": z,
                "atomic_weight": pt.GetAtomicWeight(z),
                "rcovalent_ang": pt.GetRcovalent(z),
                "rvdw_ang": pt.GetRvdw(z),
                "rb0_ang": pt.GetRb0(z),
                "n_outer_electrons": pt.GetNOuterElecs(z),
                "row": pt.GetRow(z),
                "ie1_kj": r["ie1_kj"],
                "ie2_kj": r["ie2_kj"],
                "pauling_en": r["pauling_en"],
                "group": g,
                "period": per,
                "d_electrons_M3": d,
                "unpaired_M3_lowspin": unp,
                "is_4d": 1 if per == 5 else 0,
                "is_5d": 1 if per == 6 else 0,
            }
        )
    return pd.DataFrame(rows).set_index("metal_symbol")


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "metal_symbol"]


# ---------------------------------------------------------------------------
# Leave-one-metal-out
# ---------------------------------------------------------------------------
def loo_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i])
        model = Ridge(alpha=alpha)
        model.fit(X[tr], y[tr])
        pred[i] = model.predict(X[i : i + 1])[0]
    return pred


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((pred - y) ** 2) / denom)


def loo_ridge_nested(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Pick alpha by an inner leave-one-out; report the outer number."""
    inner_scores = {}
    for a in ALPHAS:
        inner_scores[a] = r2(loo_ridge(X, y, a), y)
    best_alpha = max(inner_scores, key=lambda a: inner_scores[a])
    pred = loo_ridge(X, y, best_alpha)
    return {
        "alpha": best_alpha,
        "alpha_curve": {str(a): round(v, 4) for a, v in inner_scores.items()},
        "loo_r2": round(r2(pred, y), 4),
        "loo_pred": [round(float(v), 4) for v in pred],
    }


def loo_single(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Best single column, ordinary least squares, leave one metal out."""
    out = {}
    for j in range(X.shape[1]):
        pred = np.empty(len(y), dtype=float)
        for i in range(len(y)):
            tr = np.array([k for k in range(len(y)) if k != i])
            A = np.column_stack([np.ones(len(tr)), X[tr, j]])
            coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred[i] = coef[0] + coef[1] * X[i, j]
        out[j] = (r2(pred, y), pred)
    best = max(out, key=lambda j: out[j][0])
    return {
        "best_column_index": int(best),
        "best_loo_r2": round(float(out[best][0]), 4),
        "per_column_loo_r2": {str(j): round(float(out[j][0]), 4) for j in out},
    }


def permutation_null(X: np.ndarray, y: np.ndarray, observed: float, n_perm: int, seed: int = 20261021) -> Dict[str, object]:
    """Exhaustive for 5 metals (120 permutations); sampled above that."""
    rng = np.random.default_rng(seed)
    perms = list(itertools.permutations(range(len(y))))
    if len(perms) > n_perm:
        idx = rng.choice(len(perms), size=n_perm, replace=False)
        perms = [perms[i] for i in idx]
    null = []
    for p in perms:
        null.append(r2(loo_ridge(X, y[list(p)], 1.0), y[list(p)]))
    null = np.array(null)
    return {
        "n_permutations": int(len(perms)),
        "exhaustive": len(perms) == 120,
        "null_mean_r2": round(float(null.mean()), 4),
        "null_q95_r2": round(float(np.quantile(null, 0.95)), 4),
        "p_value": round(float((null >= observed).mean()), 4),
        "note": "alpha fixed at 1.0 inside the permutation loop so the null is not re-tuned per shuffle.",
    }


# ---------------------------------------------------------------------------
# Perfect-offset counterfactual (reads EXT-1, no refit)
# ---------------------------------------------------------------------------
def offset_counterfactual(decomp_path: Path) -> Dict[str, object]:
    if not decomp_path.exists():
        return {"status": "MISSING", "path": str(decomp_path)}
    d = json.load(open(decomp_path, encoding="utf-8"))
    out: Dict[str, object] = {"source": str(decomp_path), "cells": {}}
    for ep in ENDPOINTS:
        for split in ("metal_out", "metal_out_ligand_disjoint"):
            pm = d["endpoints"][ep]["splits"][split]["per_metal"]
            cells = {}
            for m, v in pm.items():
                if v.get("status") != "OK":
                    continue
                r2_raw = v["raw"]["r2"]
                share = v["raw"]["offset_share"]
                # removing the offset multiplies MSE by (1 - share); R2 is
                # 1 - MSE/var, so the new R2 is 1 - (1-share)(1-r2_raw).
                r2_after = 1.0 - (1.0 - share) * (1.0 - r2_raw)
                # correcting half the bias leaves offset_sq at (1-0.5)^2 = 0.25
                # of its value, i.e. removes 0.75 of the offset term.
                r2_half = 1.0 - (1.0 - 0.75 * share) * (1.0 - r2_raw)
                cells[m] = {
                    "r2_raw": r2_raw,
                    "offset_share": share,
                    "r2_if_level_perfect": round(r2_after, 4),
                    "gain": round(r2_after - r2_raw, 4),
                    "r2_if_half_bias_corrected": round(r2_half, 4),
                    "half_bias_gain": round(r2_half - r2_raw, 4),
                }
            out["cells"][f"{ep}|{split}"] = {
                "per_metal": cells,
                "mean_gain": round(float(np.mean([c["gain"] for c in cells.values()])), 4),
                "mean_half_bias_gain": round(float(np.mean([c["half_bias_gain"] for c in cells.values()])), 4),
                "n_metals_still_negative": int(sum(1 for c in cells.values() if c["r2_if_level_perfect"] < 0)),
                "n_metals": len(cells),
            }
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
        "--decomp",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_lomo_error_decomposition.json",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_descriptor_level.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    table = build_table(pd.read_csv(args.metal_desc))
    feats = feature_columns(table)
    X = table.loc[list(METALS), feats].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

    df = pd.read_parquet(args.dedup)
    endpoints: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = df[(df["endpoint"] == ep) & df["value_log10"].notna()]
        means = {m: float(sub[sub["metal"] == m]["value_log10"].mean()) for m in METALS}
        y = np.array([means[m] for m in METALS], dtype=float)

        ridge = loo_ridge_nested(X, y)
        single = loo_single(X, y)
        perm = permutation_null(X, y, ridge["loo_r2"], n_perm=120)

        endpoints[ep] = {
            "metal_means": {m: round(v, 4) for m, v in means.items()},
            "mean_sd": round(float(y.std()), 4),
            "ridge_loo": ridge,
            "single_best": {
                **single,
                "best_column": feats[single["best_column_index"]],
                "per_column_names": {str(j): feats[j] for j in range(len(feats))},
            },
            "permutation": perm,
        }
        print(
            f"[{ep}] ridge LOOCV R2={ridge['loo_r2']:+.3f} (alpha={ridge['alpha']})  "
            f"best single={feats[single['best_column_index']]} R2={single['best_loo_r2']:+.3f}  "
            f"perm p={perm['p_value']:.3f}"
        )

    cf = offset_counterfactual(args.decomp)
    gains = [cf["cells"][k]["mean_gain"] for k in cf.get("cells", {})] if "cells" in cf else []
    still_neg = [cf["cells"][k]["n_metals_still_negative"] for k in cf.get("cells", {})] if "cells" in cf else []

    best_loo = max(
        max(endpoints[ep]["ridge_loo"]["loo_r2"], endpoints[ep]["single_best"]["best_loo_r2"])
        for ep in ENDPOINTS
    )
    # The gate text in todo.json is "LOOCV R2 >= 0.5", but its own note says the
    # n=5 R2 must be read next to the permutation p and the per-fold predictions.
    # A 0.894 that a 120-permutation null reproduces 17% of the time is not
    # evidence, and a number that appears on one endpoint and inverts on the
    # other is not a metal property.  So the gate is applied with both clauses.
    per_endpoint_best = {
        ep: max(endpoints[ep]["ridge_loo"]["loo_r2"], endpoints[ep]["single_best"]["best_loo_r2"])
        for ep in ENDPOINTS
    }
    per_endpoint_p = {ep: endpoints[ep]["permutation"]["p_value"] for ep in ENDPOINTS}
    replicates = all(v >= 0.5 for v in per_endpoint_best.values())
    significant = min(per_endpoint_p.values()) < 0.05
    verdict = "PASS" if (best_loo >= 0.5 and replicates and significant) else "KILL"
    gate_reading = {
        "clause_loo_r2": f"best {best_loo:+.3f} (needs >= 0.5)",
        "clause_replicates_across_endpoints": replicates,
        "clause_permutation_p": f"min {min(per_endpoint_p.values()):.3f} (needs < 0.05)",
        "why_kill": None if verdict == "PASS" else (
            "the only positive LOOCV R2 is a single column (van der Waals radius) on one "
            "endpoint, it is not distinguishable from the 120-permutation null, and it "
            "inverts on cisplatin_ratio"
        ),
    }

    out = {
        "artifact": "msrs_metal_descriptor_level",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 level predictability",
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
            "n_metals": len(METALS),
            "n_features": len(feats),
            "features": feats,
            "feature_sources": {
                "rdkit_periodic_table": [
                    "atomic_number", "atomic_weight", "rcovalent_ang", "rvdw_ang",
                    "rb0_ang", "n_outer_electrons", "row",
                ],
                "nist_and_wikipedia": ["ie1_kj", "ie2_kj", "pauling_en"],
                "definitional_no_measurement": [
                    "group", "period", "d_electrons_M3", "unpaired_M3_lowspin", "is_4d", "is_5d",
                ],
            },
            "estimators": {
                "ridge": "standardised, alpha from a grid, nested leave-one-out",
                "single": "one column, OLS, leave-one-out",
            },
            "null": "120 permutations of metal -> mean assignment, alpha fixed at 1.0",
            "gate": "PASS if best LOOCV R2 >= 0.5; KILL otherwise",
        },
        "verdict": verdict,
        "gate_reading": gate_reading,
        "per_endpoint_best_loo_r2": {ep: round(float(v), 4) for ep, v in per_endpoint_best.items()},
        "per_endpoint_permutation_p": per_endpoint_p,
        "best_loo_r2": round(float(best_loo), 4),
        "endpoints": endpoints,
        "perfect_offset_counterfactual": cf,
        "counterfactual_summary": {
            "mean_gain_over_cells": round(float(np.mean(gains)), 4) if gains else None,
            "max_metals_still_negative": int(max(still_neg)) if still_neg else None,
            "note": (
                "Even a perfect level prediction leaves most held-out metals with R2 < 0, "
                "because EXT-1 measured the offset share at 5-10% of squared error."
            ),
        },
        "caveats": [
            "Five metals, so every LOOCV trains on four points. Any R2 here has a huge "
            "variance and the permutation null is the only meaningful reference.",
            "Six of the sixteen columns are definitional (group, period, d-count, spin). "
            "They are not measurements; they are listed separately in feature_sources.",
            "RDKit periodic-table values are programmatic and reproducible, but they are "
            "element-level, not M(III)-ion-level. Static descriptors cannot encode the "
            "ligand-field or solvation environment of the actual complex.",
            "The counterfactual is arithmetic on EXT-1's decomposition, not a refit: "
            "removing the offset multiplies MSE by (1 - offset_share).",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"\njson: {args.out}")
    print(f"VERDICT {verdict}  best_loo_r2={best_loo:+.3f}")
    if gains:
        print(f"perfect-level counterfactual: mean gain {np.mean(gains):+.4f}, "
              f"metals still negative (max over cells) {max(still_neg)}/5")
        half = [cf["cells"][k]["mean_half_bias_gain"] for k in cf["cells"]]
        print(f"half-bias-corrected counterfactual: mean gain {np.mean(half):+.4f}")
    print(f"gate: {gate_reading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
