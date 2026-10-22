#!/usr/bin/env python3
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""UP-12 — two stages: an additive model on everything, an interaction only on paired evidence.

The first round's interaction A/B (`msrs_interaction_ab`) was all-or-nothing: either
the model may cross the ligand x metal feature boundary, or it may not.  Both extremes
lose.  This script tests the middle: stage 1 is the additive model already measured by
`msrs_lomo_surface` (interaction constraints [[ligand], [metal], [cell]]), and stage 2
adds a **rank-1 interaction** whose only fitted object is a ligand projection learned
from *paired residuals* — differences between the two metals of a matched
(ligand, cell-line) group.  Metals that appear in no supported pair get a coefficient
of exactly 0, so the interaction cannot fire where there is no evidence.

    interaction(l, m) = lambda * (w . ecfp4(l)) * v_m
    d(l; a, b)        = resid(l, b) - resid(l, a)      <- the identified quantity
    w                 = ridge fit of d on ecfp4, alpha by inner ligand-grouped CV
    v                 = least squares with v_b - v_a = 1 for every supported pair,
                        sum_m v_m = 0  (one pair => +-0.5)
    lambda            = <u_hat, d> / <u_hat, u_hat>, clipped to [0, 1]

Two protocols, because the frozen one cannot see the lever:

  * **metal_out** (frozen gate).  The held-out metal is absent from training, so it is
    in no supported pair, so v_heldout = 0 and the interaction is **identically zero on
    every test row**.  Stage 1 + stage 2 therefore reproduces stage 1 to the last bit.
    The gate cannot pass, and the reason is the protocol, not the model.  Reported as
    `DEGENERATE`, with the exact identity checked numerically rather than asserted.
  * **within_metal** (the informative protocol).  Ligand-grouped 5-fold CV; the metal is
    present in training, so supported metals carry a non-zero v.  This is where the
    lever is actually tested, against both the additive model and the free-interaction
    model (the UP-1 arm).

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_two_stage.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
    morgan_matrix,
    oof_predict,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

MIN_GROUPS = 20                 # UP-10 kill rule, reused as the support threshold
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
INNER_FOLDS = 3
GATE_FROZEN_AT = "2026-10-21T20:30+08:00"


# ---------------------------------------------------------------------------
# Support structure
# ---------------------------------------------------------------------------
def pair_support(df: pd.DataFrame, min_groups: int = MIN_GROUPS) -> Dict[str, object]:
    """Matched (ligand, cell) groups per metal pair, on the rows handed in.

    Same definition as the UP-10 audit: a group is a (ligand_canonical,
    cell_line_key) that carries both metals.  Only pairs at or above `min_groups`
    are allowed to enter stage 2.
    """
    units = (
        df.groupby(["ligand_canonical", "cell_line_key", "metal"])["value_log10"]
        .mean()
        .reset_index()
    )
    counts: Dict[str, int] = {}
    for a, b in itertools.combinations(sorted(set(df["metal"])), 2):
        left = units.loc[units["metal"] == a, ["ligand_canonical", "cell_line_key"]]
        right = units.loc[units["metal"] == b, ["ligand_canonical", "cell_line_key"]]
        counts[f"{a}-{b}"] = int(len(left.merge(right, on=["ligand_canonical", "cell_line_key"])))
    supported = {k: v for k, v in counts.items() if v >= min_groups}
    return {"counts": counts, "supported": supported}


def metal_coefficients(supported: Sequence[str]) -> Dict[str, float]:
    """v_m with v_b - v_a = 1 per supported pair and sum_m v_m = 0.

    One pair gives (+0.5, -0.5).  Pairs that share a metal are reconciled by least
    squares.  A metal in no supported pair is absent from the dict; callers must
    treat a missing key as 0.
    """
    metals = sorted({m for p in supported for m in p.split("-")})
    if not metals:
        return {}
    rows, rhs = [], []
    for pair in supported:
        a, b = pair.split("-")
        row = [0.0] * len(metals)
        row[metals.index(b)] = 1.0
        row[metals.index(a)] = -1.0
        rows.append(row)
        rhs.append(1.0)
    rows.append([1.0] * len(metals))       # sum_m v_m = 0
    rhs.append(0.0)
    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return {m: float(v) for m, v in zip(metals, sol)}


# ---------------------------------------------------------------------------
# Stage 2: one ligand projection, one coefficient per metal
# ---------------------------------------------------------------------------
def pair_differences(
    df: pd.DataFrame, resid: np.ndarray, supported: Sequence[str]
) -> pd.DataFrame:
    """Per-ligand residual difference d = resid(b) - resid(a) for each supported pair."""
    units = (
        pd.DataFrame(
            {
                "frag": df["ligand_main_fragment"].to_numpy(),
                "metal": df["metal"].to_numpy(),
                "resid": resid,
            }
        )
        .groupby(["frag", "metal"])["resid"]
        .mean()
        .reset_index()
    )
    out = []
    for pair in supported:
        a, b = pair.split("-")
        ra = units.loc[units["metal"] == a].set_index("frag")["resid"].rename("r_a")
        rb = units.loc[units["metal"] == b].set_index("frag")["resid"].rename("r_b")
        joint = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if joint.empty:
            continue
        joint["d"] = joint["r_b"] - joint["r_a"]
        joint["pair"] = pair
        out.append(joint.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["frag", "d"])


def fit_ligand_projection(
    diffs: pd.DataFrame, alpha_grid: Sequence[float] = ALPHAS, seed: int = 20261021
) -> Dict[str, object]:
    """Ridge d ~ ecfp4 with alpha chosen by inner ligand-grouped CV."""
    if diffs.empty:
        return {"status": "EMPTY"}
    frags = diffs["frag"].tolist()
    X = morgan_matrix(frags)
    y = diffs["d"].to_numpy(dtype=float)

    n = len(y)
    k = min(INNER_FOLDS, max(2, n // 10))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold = np.empty(n, dtype=int)
    fold[order] = np.arange(n) % k

    best = None
    for alpha in alpha_grid:
        sse = 0.0
        for f in range(k):
            tr, te = fold != f, fold == f
            if tr.sum() < 5 or te.sum() < 2:
                continue
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X[tr], y[tr])
            sse += float(np.sum((model.predict(X[te]) - y[te]) ** 2))
        if best is None or sse < best[1]:
            best = (alpha, sse)
    alpha = float(best[0])

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    u_hat = model.predict(X)
    denom = float(np.dot(u_hat, u_hat))
    lam = float(np.dot(u_hat, y) / denom) if denom > 1e-12 else 0.0
    lam_clipped = float(np.clip(lam, 0.0, 1.0))
    return {
        "status": "OK",
        "alpha": alpha,
        "lambda_raw": round(lam, 4),
        "lambda": round(lam_clipped, 4),
        "n_rows": int(n),
        "n_ligands": int(diffs["frag"].nunique()),
        "inner_folds": int(k),
        "r2_inner": round(
            float(1.0 - best[1] / max(np.sum((y - y.mean()) ** 2), 1e-12)), 4
        ),
        "model": model,
    }


def interaction_terms(model: Dict[str, object], frags: Sequence[str], metals: Sequence[str],
                      coefs: Dict[str, float]) -> np.ndarray:
    """lambda * (w . ecfp4) * v_m; a metal absent from `coefs` contributes exactly 0."""
    if model.get("status") != "OK":
        return np.zeros(len(frags), dtype=float)
    u = model["model"].predict(morgan_matrix(list(frags))) * model["lambda"]
    v = np.array([coefs.get(m, 0.0) for m in metals], dtype=float)
    return u * v


# ---------------------------------------------------------------------------
# Protocol A: metal-out (the frozen gate)
# ---------------------------------------------------------------------------
def metal_out(df: pd.DataFrame, y: np.ndarray, blocks) -> Dict[str, object]:
    """Stage 2 is fitted for real, then applied to a metal whose v is exactly 0."""
    metals = df["metal"].to_numpy()
    frag = df["ligand_main_fragment"].to_numpy()
    per_metal: Dict[str, object] = {}
    max_abs_delta = 0.0
    for held in METALS:
        train = np.where(metals != held)[0]
        test = np.where(metals == held)[0]
        if len(train) < 200 or len(test) < 20:
            per_metal[held] = {"status": "INSUFFICIENT", "n_train": int(len(train)),
                               "n_test": int(len(test))}
            continue
        support = pair_support(df.iloc[train])
        coefs = metal_coefficients(list(support["supported"]))
        X_tr, groups = assemble(blocks, "descriptors", train)
        X_te, _ = assemble(blocks, "descriptors", test)
        p_stage1 = fit_predict(X_tr, y[train], X_te, groups)

        oof_tr, _ = oof_predict(X_tr, y[train], frag[train], groups)
        diffs = pair_differences(df.iloc[train], y[train] - oof_tr,
                                 list(support["supported"]))
        model = fit_ligand_projection(diffs)
        inter = interaction_terms(model, frag[test], metals[test], coefs)

        v_held = coefs.get(held, 0.0)
        p_two = p_stage1 + inter
        delta = float(np.max(np.abs(p_two - p_stage1)))
        max_abs_delta = max(max_abs_delta, delta)

        per_metal[held] = {
            "status": "OK",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "supported_pairs_train": support["supported"],
            "pair_counts_train": support["counts"],
            "metal_coefficients_train": {k: round(v, 4) for k, v in coefs.items()},
            "v_held_out": round(v_held, 4),
            "stage2_alpha": model.get("alpha"),
            "stage2_lambda": model.get("lambda"),
            "n_pair_rows": model.get("n_rows"),
            "n_test_rows_nonzero_interaction": int(np.count_nonzero(np.abs(inter) > 1e-12)),
            "stage1": score(y[test], p_stage1),
            "stage1_plus_stage2": score(y[test], p_two),
            "max_abs_prediction_delta": delta,
        }
    return {"per_metal": per_metal, "max_abs_delta_over_metals": round(max_abs_delta, 12)}


# ---------------------------------------------------------------------------
# Protocol B: within-metal, ligand-grouped
# ---------------------------------------------------------------------------
def within_metal(
    df: pd.DataFrame, y: np.ndarray, blocks
) -> Dict[str, object]:
    from sklearn.model_selection import GroupKFold

    frag = df["ligand_main_fragment"].to_numpy()
    metals = df["metal"].to_numpy()
    splitter = GroupKFold(n_splits=N_FOLDS)

    preds = {a: np.full(len(y), np.nan) for a in ("additive", "two_stage", "free")}
    fold_rows = []
    for k, (tr, te) in enumerate(splitter.split(np.zeros((len(y), 1)), y, frag)):
        X_tr, groups = assemble(blocks, "descriptors", tr)
        X_te, _ = assemble(blocks, "descriptors", te)

        p_add = fit_predict(X_tr, y[tr], X_te, groups)
        p_free = fit_predict(X_tr, y[tr], X_te, None)
        oof_tr, _ = oof_predict(X_tr, y[tr], frag[tr], groups)

        support = pair_support(df.iloc[tr])
        coefs = metal_coefficients(list(support["supported"]))
        diffs = pair_differences(df.iloc[tr], y[tr] - oof_tr, list(support["supported"]))
        model = fit_ligand_projection(diffs)
        inter = interaction_terms(model, frag[te], metals[te], coefs)

        preds["additive"][te] = p_add
        preds["free"][te] = p_free
        preds["two_stage"][te] = p_add + inter

        fold_rows.append(
            {
                "fold": k,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "supported_pairs": support["supported"],
                "metal_coefficients": {m: round(v, 4) for m, v in coefs.items()},
                "stage2_alpha": model.get("alpha"),
                "stage2_lambda": model.get("lambda"),
                "stage2_lambda_raw": model.get("lambda_raw"),
                "stage2_r2_inner": model.get("r2_inner"),
                "n_pair_rows": model.get("n_rows"),
                "interaction_sd": round(float(np.std(inter)), 4),
                "interaction_mean": round(float(np.mean(inter)), 4),
            }
        )

    assert not any(np.isnan(v).any() for v in preds.values()), "fold coverage incomplete"

    per_metal: Dict[str, object] = {}
    for metal in METALS:
        m = metals == metal
        if int(m.sum()) < 20:
            per_metal[metal] = {"status": "INSUFFICIENT", "n_test": int(m.sum())}
            continue
        per_metal[metal] = {
            "status": "OK",
            "n_test": int(m.sum()),
            "additive": score(y[m], preds["additive"][m]),
            "two_stage": score(y[m], preds["two_stage"][m]),
            "free": score(y[m], preds["free"][m]),
        }
        per_metal[metal]["delta_two_stage_vs_additive"] = round(
            per_metal[metal]["two_stage"]["r2"] - per_metal[metal]["additive"]["r2"], 4
        )

    ok = {m: v for m, v in per_metal.items() if v.get("status") == "OK"}
    n = np.array([ok[m]["n_test"] for m in ok], dtype=float)
    aggregate = {
        arm: {
            "weighted_r2": round(
                float(np.average([ok[m][arm]["r2"] for m in ok], weights=n)), 4
            ),
            "weighted_mae": round(
                float(np.average([ok[m][arm]["mae"] for m in ok], weights=n)), 4
            ),
            "n_metals_negative_r2": int(sum(ok[m][arm]["r2"] < 0 for m in ok)),
        }
        for arm in ("additive", "two_stage", "free")
    }
    aggregate["delta_two_stage_vs_additive"] = round(
        aggregate["two_stage"]["weighted_r2"] - aggregate["additive"]["weighted_r2"], 4
    )
    aggregate["delta_two_stage_vs_free"] = round(
        aggregate["two_stage"]["weighted_r2"] - aggregate["free"]["weighted_r2"], 4
    )
    return {
        "per_metal": per_metal,
        "aggregate": aggregate,
        "folds": fold_rows,
        "predictions": preds,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency", choices=("potency", "cisplatin_ratio"))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_two_stage.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = pd.read_parquet(args.dedup)
    sub = df[(df["endpoint"] == args.endpoint) & df["value_log10"].notna()
             & df["ligand_main_fragment"].notna()].reset_index(drop=True)
    desc = pd.read_csv(args.metal_desc)
    vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, desc, vocab, metal_columns=DEFAULT_METAL_COLUMNS)
    y = sub["value_log10"].to_numpy(dtype=float)
    print(f"[UP-12] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
          f"metals={sub['metal'].nunique()}")

    support = pair_support(sub)
    print(f"[UP-12] supported pairs (full corpus): {support['supported']}")

    mo = metal_out(sub, y, blocks)
    print(f"[UP-12] metal_out max |pred_two_stage - pred_stage1| = "
          f"{mo['max_abs_delta_over_metals']:.3g}")

    wm = within_metal(sub, y, blocks)
    preds = wm.pop("predictions")
    print(f"[UP-12] within_metal aggregate: {json.dumps(wm['aggregate'], ensure_ascii=False)}")
    for m, v in wm["per_metal"].items():
        if v.get("status") == "OK":
            print(f"[UP-12]   {m:2s} additive={v['additive']['r2']:+.4f} "
                  f"two_stage={v['two_stage']['r2']:+.4f} free={v['free']['r2']:+.4f} "
                  f"delta={v['delta_two_stage_vs_additive']:+.4f}")

    # Frozen gate, applied verbatim to metal_out.  The baseline is read from the
    # artifact it was frozen against, not retyped, so the comparison cannot drift.
    lomo_path = PROJECT_ROOT / "metrics" / "by_metric" / "msrs_lomo_surface.json"
    lomo = json.loads(lomo_path.read_text())
    baseline_weighted = float(
        lomo["endpoints"][args.endpoint]["splits"]["metal_out"]["aggregate"]
        ["descriptors"]["weighted_r2"]
    )
    mo_ok = {m: v for m, v in mo["per_metal"].items() if v.get("status") == "OK"}
    n = np.array([mo_ok[m]["n_test"] for m in mo_ok], dtype=float)
    mo_weighted = float(np.average([mo_ok[m]["stage1_plus_stage2"]["r2"] for m in mo_ok],
                                   weights=n))
    per_metal_delta = {
        m: round(mo_ok[m]["stage1_plus_stage2"]["r2"] - mo_ok[m]["stage1"]["r2"], 6)
        for m in mo_ok
    }
    gate = {
        "frozen_at": GATE_FROZEN_AT,
        "baseline_source": "metrics/by_metric/msrs_lomo_surface.json:"
                           f"/endpoints/{args.endpoint}/splits/metal_out/aggregate/"
                           "descriptors/weighted_r2",
        "baseline_metal_out_weighted_r2": baseline_weighted,
        "metal_out_weighted_r2": round(mo_weighted, 4),
        "improvement": round(mo_weighted - baseline_weighted, 4),
        "improvement_required": 0.03,
        "per_metal_delta": per_metal_delta,
        "max_per_metal_drop": round(min(per_metal_delta.values()), 6),
        "no_metal_drops_more_than": 0.02,
        "gate_pass": bool(
            mo_weighted - baseline_weighted >= 0.03
            and min(per_metal_delta.values()) >= -0.02
        ),
        "gate_failure_mode": (
            "DEGENERATE_IDENTICAL_PREDICTIONS"
            if mo["max_abs_delta_over_metals"] == 0.0 else "REAL_DELTA"
        ),
    }

    wm_agg = wm["aggregate"]
    wm_delta = wm_agg["delta_two_stage_vs_additive"]
    wm_supported = ("Ru", "Ir", "Rh", "Os")
    wm_max_drop = min(
        wm["per_metal"][m]["delta_two_stage_vs_additive"]
        for m in wm_supported if wm["per_metal"][m].get("status") == "OK"
    )
    kill = {
        "threshold_improvement": 0.01,
        "threshold_per_metal_drop": 0.02,
        "applied_to": "within_metal (the frozen metal_out gate is degenerate)",
        "improvement": wm_delta,
        "max_supported_metal_drop": round(float(wm_max_drop), 4),
        "kill_triggered": bool(wm_delta < 0.01 or wm_max_drop < -0.02),
    }

    if gate["gate_failure_mode"] == "DEGENERATE_IDENTICAL_PREDICTIONS":
        verdict = "METAL_OUT_CANNOT_TEST_THE_LEVER"
    elif gate["gate_pass"]:
        verdict = "TWO_STAGE_HELPS_ON_METAL_OUT"
    elif kill["kill_triggered"]:
        verdict = "TWO_STAGE_ADDS_NOTHING"
    else:
        verdict = "MIXED"

    out = {
        "artifact": "msrs_two_stage",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "UP-12 two-stage residual model",
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
            "stage1": "LightGBM, interaction_constraints [[ligand],[metal],[cell]], "
                      "descriptors representation (3 metal columns)",
            "stage2": "rank-1: interaction(l,m) = lambda * (w . ecfp4(l)) * v_m",
            "stage2_target": "per-ligand residual difference d = resid(b) - resid(a) on "
                             "matched (ligand, cell) groups of a supported pair",
            "residual_source": "out-of-fold residuals inside the training side "
                               f"(GroupKFold {N_FOLDS} by ligand)",
            "supported_pair_rule": f"n matched groups >= {MIN_GROUPS} (UP-10 kill rule)",
            "metal_coefficient": "least squares, v_b - v_a = 1 per pair, sum_m v_m = 0",
            "unsupported_metal_coefficient": 0.0,
            "alpha_grid": list(ALPHAS),
            "inner_folds": INNER_FOLDS,
            "protocols": ["metal_out", "within_metal"],
        },
        "support_full_corpus": support,
        "metal_out": mo,
        "within_metal": wm,
        "gate": gate,
        "kill": kill,
        "verdict": verdict,
        "caveats": [
            "On metal_out the held-out metal has no supported pair by construction, so "
            "v_held = 0 and the stage-2 term is exactly zero. The frozen gate therefore "
            "measures stage 1 against stage 1. That is a property of the protocol, not a "
            "result about the two-stage model; the within_metal block is the informative "
            "test.",
            "The two-stage arm is evaluated against the same additive stage 1 it is built "
            "on, so its gain is bounded by how much residual structure a rank-1 ligand "
            "projection can capture. It is not a claim about the ceiling.",
            "v_m is a fixed unit vector given the supported pairs, not a fitted quantity. "
            "With one supported pair inside a training fold it is (+0.5, -0.5) exactly; "
            "the only fitted object is the 2048-dimensional ligand projection.",
            "lambda is a single closed-form shrinkage on the training pair differences; it "
            "is not cross-validated, so a small positive lambda is weak evidence of a real "
            "interaction rather than of ridge under-fitting.",
            "Stage-2 residuals are out-of-fold, which removes in-sample overfit but also "
            "shrinks them; a real interaction will appear smaller than it is.",
        ],
    }
    if not (np.isfinite(list(preds["two_stage"])).all()):
        raise RuntimeError("non-finite two-stage prediction")

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"[UP-12] gate: {json.dumps(gate, ensure_ascii=False)}")
    print(f"[UP-12] kill: {json.dumps(kill, ensure_ascii=False)}")
    print(f"[UP-12] verdict={verdict}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
