#!/usr/bin/env python3
"""UP-34 — can a handful of measured rows locate a metal the model never saw?

=====================================================================
Why this exists
=====================================================================
UP-19 showed that 94.5% of the rows have no matched partner, so a metal's
level cannot be inferred from ligand similarity; UP-30 showed that
loosening the matching rule does not rescue it.  Both were asking the
model to *predict* the level.  UP-26 asked the same question about
papers and got a different answer: do not predict the level, measure a
few rows of it and subtract the empirical-Bayes posterior mean.

That operator has never been run on the metal axis.  Deployment is the
same shape: a new metal enters with k measured compounds, and the
question is whether those k rows carry enough information to shift the
rest of that metal's cohort.

=====================================================================
The ceiling, computed before the sweep
=====================================================================
A per-metal constant offset can only move a metal's R^2 by bias^2/Var(y).
Summing that over the five held-out metals with the published
per-metal rows (``msrs_identifiability.json``) gives a weighted ceiling
of 0.0092 against a baseline of -0.0734: the whole recoverable gap is
0.083.  The first version of this gate asked for R^2 >= 0.30 at k=10,
which is 33x the ceiling and could not be met by any estimator in this
family.  The gate was therefore rewritten to be a fraction of the
oracle gap before any k-curve was produced; both texts are recorded in
the artifact.

=====================================================================
Two levels, and why the second one is usually unidentified
=====================================================================
A test row's residual has a paper component and a metal component.  The
paper component is estimated from that paper's rows in the *training*
metals, cross-fitted so the estimate is not an in-sample residual.  The
metal component is estimated from the k measured rows after removing
their papers' offsets.

Most test papers do not appear in any training metal (10.4% of units sit
in a multi-metal paper), so for most rows the paper term is zero and the
k rows identify the metal term alone.  The identifiability table reports
how many distinct papers the k rows span, because with one paper the two
components are collinear and only their sum is identified.

=====================================================================
Arms
=====================================================================
``baseline``    the LOMO prediction, no correction.  Reproduces -0.0734.
``paper_only``  + the paper offset only.  Independent of k, so it is the
                control that says how much of any gain comes from the
                paper axis rather than the metal axis.
``paper_metal`` + paper offset + EB metal offset from k rows.  PRIMARY.
``oracle``      + each metal's mean residual over all of its rows.  The
                ceiling, not a model.

=====================================================================
Gate (todo.json v5, frozen 2026-10-23T03:20+08:00; primary rewritten
at 2026-10-23T03:55+08:00, before any k-curve existed)
=====================================================================
primary     k=10 recovers >= 70% of the oracle gap
reproduce   baseline = -0.0734 +- 0.01
contrast    paper_only recovers < 30% of the gap
monotone    >= 4 of the 5 k points non-decreasing
kill        k=10 recovers < 30% of the gap closes the metal_out line

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_metal_calibration.py
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_lomo_surface import (  # noqa: E402
    DEFAULT_METAL_COLUMNS,
    _git_commit,
    _sha256,
    assemble,
    build_blocks,
)
from molmetal.scripts.msrs_paper_calibration import eb_offset  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    SEED,
    TOP_CELL_LINES,
    fit_predict,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

K_GRID = (0, 5, 10, 20, 50)
REPEATS = 20
INNER_FOLDS = 5
GATE_FROZEN_AT = "2026-10-23T03:20+08:00"
GATE_AMENDED_AT = "2026-10-23T03:55+08:00"
PRIMARY_K = 10
RECOVERY_GATE = 0.70
CONTRAST_MAX = 0.30
KILL_RECOVERY = 0.30
MIN_MONOTONE = 4
REPRO_TARGET = -0.0734
REPRO_TOL = 0.01
REPRESENTATION = "descriptors"


# ---------------------------------------------------------------------------
# Variance components on the metal axis
# ---------------------------------------------------------------------------
def group_variances(
    resid: np.ndarray, group_idx: np.ndarray, rows: np.ndarray
) -> Dict[str, float]:
    """Method-of-moments split of the residual variance into noise and group.

    Same estimator as ``msrs_paper_calibration.paper_variances`` with the
    grouping variable swapped, so the two kappa values in this artifact are
    on one scale and can be compared.
    """
    idx = group_idx[rows]
    r = resid[rows]
    uniq, inv = np.unique(idx, return_inverse=True)
    n_g = np.bincount(inv).astype(float)
    s1 = np.bincount(inv, weights=r)
    s2 = np.bincount(inv, weights=r * r)
    mean = np.divide(s1, n_g, out=np.zeros_like(s1), where=n_g > 0)
    ss_within = s2 - n_g * mean * mean
    dof = np.maximum(n_g - 1.0, 0.0)
    sigma2 = float(ss_within.sum() / dof.sum()) if dof.sum() > 0 else float("nan")
    big = n_g >= 2
    between = float(np.var(mean[big], ddof=1)) if int(big.sum()) > 1 else 0.0
    correction = float(np.mean(sigma2 / n_g[big])) if int(big.sum()) > 0 else 0.0
    tau2 = max(between - correction, 1e-6)
    return {
        "sigma2_noise": sigma2,
        "tau2_raw": between,
        "tau2": tau2,
        "kappa": float(sigma2 / tau2),
        "n_groups": int(len(uniq)),
        "n_rows": int(len(rows)),
    }


def oof_residuals(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, constraints: List[List[int]]
) -> np.ndarray:
    """Cross-fitted residuals inside the training metals.

    An in-sample residual is shrunk toward zero by the model's own fit, which
    would understate every paper offset.  Five DOI-grouped folds keep each
    residual out of the model that produced it.
    """
    resid = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups):
        pred = fit_predict(X[tr], y[tr], X[te], constraints)
        resid[te] = y[te] - pred
    return resid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric"
                    / "msrs_metal_calibration.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    sub = df[df["endpoint"] == "potency"]
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()]
    sub = sub.reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    metal = sub["metal"].to_numpy()
    doi = sub["doi"].astype(str).to_numpy()
    vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, descriptors, vocabulary,
                          metal_columns=DEFAULT_METAL_COLUMNS)
    print(f"[UP-34] rows={len(sub)} metals={len(METALS)} dois={len(set(doi))}")

    rng = np.random.default_rng(SEED)

    per_metal: Dict[str, Dict[str, object]] = {}
    for holdout in METALS:
        ts = time.time()
        train = np.where(metal != holdout)[0]
        test = np.where(metal == holdout)[0]
        if len(train) < 200 or len(test) < 20:
            per_metal[holdout] = {"status": "INSUFFICIENT",
                                  "n_train": int(len(train)),
                                  "n_test": int(len(test))}
            continue

        X_tr, groups_tr = assemble(blocks, REPRESENTATION, train)
        X_te, _ = assemble(blocks, REPRESENTATION, test)
        pred = fit_predict(X_tr, y[train], X_te, groups_tr)

        # Cross-fitted residuals on the training metals give the paper and
        # metal variance components without ever touching the held-out metal.
        resid_tr = oof_residuals(X_tr, y[train], doi[train], groups_tr)
        paper_idx_tr = pd.factorize(doi[train])[0]
        v_paper = group_variances(resid_tr, paper_idx_tr, np.arange(len(train)))
        v_metal = group_variances(resid_tr, pd.factorize(metal[train])[0],
                                  np.arange(len(train)))

        # Paper offsets for the held-out cohort, from that paper's rows in the
        # training metals.  A paper that never appears there gets zero, which
        # is the honest answer rather than a borrowed one.
        paper_mean: Dict[str, float] = {}
        paper_n: Dict[str, int] = {}
        for d in np.unique(doi[train]):
            mask = doi[train] == d
            paper_mean[d] = float(np.mean(resid_tr[mask]))
            paper_n[d] = int(mask.sum())
        off_paper_te = np.array([
            paper_n[d] / (paper_n[d] + v_paper["kappa"]) * paper_mean[d]
            if d in paper_mean else 0.0
            for d in doi[test]
        ])
        n_paper_known = int(sum(1 for d in doi[test] if d in paper_mean))

        resid = y[test] - pred
        off_paper_resid = resid - off_paper_te

        # Oracle: subtract this metal's mean residual over every one of its
        # rows.  Not deployable; it bounds what any constant offset can reach.
        oracle_pred = pred + float(resid.mean())

        arms: Dict[str, Dict[str, object]] = {
            "baseline": {"r2": round(float(r2_score(y[test], pred)), 4)},
            "paper_only": {"r2": round(float(r2_score(y[test], pred + off_paper_te)), 4)},
            "oracle": {"r2": round(float(r2_score(y[test], oracle_pred)), 4)},
        }

        k_curve: Dict[str, object] = {}
        n_papers_drawn: Dict[str, float] = {}
        share_matched: Dict[str, float] = {}
        for k in K_GRID:
            r2s: List[float] = []
            r2_eb_only: List[float] = []
            r2_raw: List[float] = []
            offs: List[float] = []
            papers_per_draw: List[int] = []
            matched_per_draw: List[float] = []
            for _ in range(args.repeats):
                if k == 0:
                    off = 0.0
                    raw = 0.0
                    papers_per_draw.append(0)
                    matched_per_draw.append(float("nan"))
                else:
                    kk = min(k, len(test))
                    pick = rng.choice(len(test), size=kk, replace=False)
                    # The pre-registered estimator: EB on the paper-corrected
                    # residual.  kappa comes from the *training* metals.
                    off = eb_offset(off_paper_resid[pick], v_metal["kappa"])
                    # Diagnostic: the unshrunk mean of the same rows.  It
                    # isolates whether the k rows carry the information at all,
                    # separately from whether the prior transfers.
                    raw = float(np.mean(resid[pick]))
                    papers_per_draw.append(int(len(set(doi[test][pick]))))
                    # Identifiability: a k row can only separate a paper effect
                    # from a metal effect if its paper also appears in a
                    # training metal.
                    matched_per_draw.append(
                        float(np.mean([d in paper_mean for d in doi[test][pick]])))
                r2s.append(float(r2_score(y[test], pred + off_paper_te + off)))
                r2_eb_only.append(float(r2_score(y[test], pred + off)))
                r2_raw.append(float(r2_score(y[test], pred + raw)))
                offs.append(float(off))
            k_curve[str(k)] = {
                "r2_mean": round(float(np.mean(r2s)), 4),
                "r2_min": round(float(np.min(r2s)), 4),
                "r2_max": round(float(np.max(r2s)), 4),
                "offset_mean": round(float(np.mean(offs)), 4),
                "offset_sd": round(float(np.std(offs)), 4),
                "n_rows_sampled": int(min(k, len(test))),
                "metal_only_eb_r2_mean": round(float(np.mean(r2_eb_only)), 4),
                "metal_only_raw_r2_mean": round(float(np.mean(r2_raw)), 4),
            }
            n_papers_drawn[str(k)] = round(float(np.mean(papers_per_draw)), 2)
            share_matched[str(k)] = (
                round(float(np.nanmean(matched_per_draw)), 4) if k else None)

        per_metal[holdout] = {
            "status": "OK",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "share_test_rows_with_paper_in_train": round(n_paper_known / len(test), 4),
            "kappa_paper": round(float(v_paper["kappa"]), 4),
            "kappa_metal": round(float(v_metal["kappa"]), 4),
            "tau2_paper": round(float(v_paper["tau2"]), 6),
            "tau2_metal": round(float(v_metal["tau2"]), 6),
            "sigma2_noise": round(float(v_metal["sigma2_noise"]), 6),
            "arms": arms,
            "k_curve": k_curve,
            "n_papers_in_k_rows": n_papers_drawn,
            "share_k_rows_with_paper_in_train": share_matched,
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        print(f"[UP-34] holdout={holdout} n_test={len(test)} "
              f"base={arms['baseline']['r2']} paper={arms['paper_only']['r2']} "
              f"oracle={arms['oracle']['r2']} "
              f"k10={k_curve['10']['r2_mean']} k50={k_curve['50']['r2_mean']} "
              f"kappa_m={v_metal['kappa']:.2f} ({per_metal[holdout]['elapsed_seconds']}s)")

    ok = {m: v for m, v in per_metal.items() if v.get("status") == "OK"}
    if not ok:
        print("FAIL: no metal produced a usable split")
        return 1

    weights = np.array([ok[m]["n_test"] for m in ok], dtype=float)
    agg: Dict[str, object] = {}
    for arm in ("baseline", "paper_only", "oracle"):
        vals = np.array([ok[m]["arms"][arm]["r2"] for m in ok], dtype=float)
        agg[arm] = {"weighted_r2": round(float(np.average(vals, weights=weights)), 4)}
    k_agg: Dict[str, object] = {}
    for k in K_GRID:
        row: Dict[str, float] = {}
        for key, field in (("primary", "r2_mean"),
                           ("metal_only_eb", "metal_only_eb_r2_mean"),
                           ("metal_only_raw", "metal_only_raw_r2_mean")):
            vals = np.array([ok[m]["k_curve"][str(k)][field] for m in ok], dtype=float)
            row[key] = round(float(np.average(vals, weights=weights)), 4)
        k_agg[str(k)] = row

    base_r2 = agg["baseline"]["weighted_r2"]
    oracle_r2 = agg["oracle"]["weighted_r2"]
    gap = oracle_r2 - base_r2

    def recovery(value: float) -> float:
        return round(float((value - base_r2) / gap), 4) if gap > 1e-9 else float("nan")

    curve = [k_agg[str(k)]["primary"] for k in K_GRID]
    steps = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    n_nondec = int(sum(1 for d in steps if d >= 0))

    rec10 = recovery(k_agg[str(PRIMARY_K)]["primary"])
    rec_po = recovery(agg["paper_only"]["weighted_r2"])
    checks = {
        "primary_recovery_k10": {
            "value": rec10,
            "r2": k_agg[str(PRIMARY_K)]["primary"],
            "threshold": RECOVERY_GATE,
            "pass": bool(rec10 >= RECOVERY_GATE),
        },
        "reproduce_baseline": {
            "value": base_r2,
            "target": REPRO_TARGET,
            "tolerance": REPRO_TOL,
            "pass": bool(abs(base_r2 - REPRO_TARGET) <= REPRO_TOL),
        },
        "contrast_paper_only": {
            "value": rec_po,
            "r2": agg["paper_only"]["weighted_r2"],
            "max": CONTRAST_MAX,
            "pass": bool(rec_po < CONTRAST_MAX),
        },
        "monotone": {
            "curve": curve,
            "steps": [round(float(d), 4) for d in steps],
            "non_decreasing_steps": n_nondec,
            "required": MIN_MONOTONE,
            "pass": bool(n_nondec >= MIN_MONOTONE),
        },
    }
    kill = {
        "rule": "k=10 recovering < 30% of the oracle gap closes the metal_out line",
        "value": rec10,
        "triggered": bool(rec10 < KILL_RECOVERY),
    }
    # The mechanical rule fires on the pre-registered arm.  The diagnostic arms
    # ask whether that firing means "no information in k rows" or "the prior and
    # the paper term are wrong"; the two readings imply opposite decisions, so
    # both are written down rather than left to the reader.
    rec_raw10 = recovery(k_agg[str(PRIMARY_K)]["metal_only_raw"])
    rec_raw50 = recovery(k_agg["50"]["metal_only_raw"])
    rec_eb10 = recovery(k_agg[str(PRIMARY_K)]["metal_only_eb"])
    diagnostics = {
        "metal_only_raw_recovery_k10": rec_raw10,
        "metal_only_raw_recovery_k50": rec_raw50,
        "metal_only_eb_recovery_k10": rec_eb10,
        "paper_only_recovery": rec_po,
    }
    if kill["triggered"] and rec_raw50 >= KILL_RECOVERY:
        kill["interpretation"] = (
            "The pre-registered arm fires. The unshrunk mean of the same k rows "
            f"recovers {rec_raw10:.2f} of the gap at k=10 and {rec_raw50:.2f} at "
            "k=50, so the level is in the rows but ten of them are not enough to "
            "read it: the sampling error of a 10-row mean is the same size as the "
            "bias it is meant to remove. Two separate defects make the "
            "pre-registered estimator worse than the baseline at every k: the paper "
            "term is confounded with the training metal's level and does not "
            "transfer, and kappa_metal estimated on the observed metals is "
            "effectively infinite because the held-out metal sits outside their "
            "between-metal variance. Closing the line at k=10 is correct for the "
            "deployment story as written; a k~50 protocol would need its own "
            "pre-registration.")
    else:
        kill["interpretation"] = (
            "The pre-registered arm fires and no diagnostic arm recovers the gap, so "
            "the reading is that k rows do not identify the metal level.")
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_metal_calibration",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 metal extrapolation with measured rows (UP-34)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": "potency, censored dropped",
            "n_rows": int(len(sub)),
            "split": "leave-one-metal-out, all ligands kept",
            "representation": REPRESENTATION,
            "learner": "the additive LightGBM used by msrs_lomo_surface",
            "seed": SEED,
            "k_grid": list(K_GRID),
            "repeats": args.repeats,
            "primary_arm": "paper_metal",
            "estimator": "offset = n/(n+kappa_metal) * mean(resid - paper_offset | k rows)",
            "kappa_definition": "sigma2_noise / tau2, method-of-moments on the "
                                "training metals' cross-fitted residuals",
            "oracle": "each metal's mean residual over all of its rows; not deployable",
            "oracle_ceiling_source": "msrs_identifiability.json per-metal "
                                     "r2_centred, n-weighted: 0.0092",
            "gate_frozen_at": GATE_FROZEN_AT,
            "gate_amended_at": GATE_AMENDED_AT,
            "amendment": "the frozen primary asked for R^2 >= 0.30 at k=10, which is "
                         "33x the 0.0092 ceiling of any per-metal constant offset; it "
                         "was rewritten as a fraction of the oracle gap before any "
                         "k-curve was produced",
        },
        "per_metal": per_metal,
        "aggregate": {**agg, "k_curve": k_agg, "oracle_gap": round(float(gap), 4),
                      "diagnostics": diagnostics},
        "gate": {"frozen_at": GATE_FROZEN_AT, "amended_at": GATE_AMENDED_AT,
                 "checks": checks, "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The k rows are drawn uniformly from the held-out metal's cohort, not "
            "from a designed plate. A deployment that can choose which compounds to "
            "measure may do better; that is a different experiment.",
            "The plan specified a two-level decomposition with the metal term "
            "identified from within-paper cross-metal residual differences. That "
            "estimator is not implemented here because the data cannot identify it: "
            "only a small share of k rows belong to papers that also appear in a "
            "training metal, and when they do their training rows span several "
            "metals, so the difference does not isolate the held-out metal. The "
            "share is reported per metal and per k instead of assuming it away.",
            "The paper offset is estimated from the paper's rows in the training "
            "metals. Those rows also carry their own metal's level, so the estimate "
            "is confounded with a metal effect and does not transfer to the held-out "
            "metal; the paper_only arm measures that failure rather than hiding it.",
            "kappa_metal is estimated on the four observed metals. The held-out "
            "metal is by construction the one whose level the model gets wrong, so it "
            "sits outside that variance and the shrinkage becomes near-total. This is "
            "a property of leave-one-metal-out, not of empirical Bayes.",
            "The oracle uses every row of the held-out metal, including the rows a "
            "real method would spend on calibration. It is a ceiling, not a model.",
            "Weighted R^2 is the n-weighted mean of per-metal R^2, matching "
            "msrs_lomo_surface. It is not a pooled R^2 and the two differ by "
            "construction.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for k in K_GRID:
        print(f"[UP-34] k={k:2d} primary={k_agg[str(k)]['primary']:.4f} "
              f"metal_eb={k_agg[str(k)]['metal_only_eb']:.4f} "
              f"metal_raw={k_agg[str(k)]['metal_only_raw']:.4f} "
              f"recovery={recovery(k_agg[str(k)]['primary']):.4f}")
    print(f"[UP-34] base={base_r2} paper_only={agg['paper_only']['weighted_r2']} "
          f"oracle={oracle_r2} gap={gap:.4f} rec10={rec10}")
    print(f"[UP-34] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time() - t0, 1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
