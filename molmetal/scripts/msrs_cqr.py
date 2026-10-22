#!/usr/bin/env python3
"""UP-20 — CQR: can width and coverage stop trading against each other?

=====================================================================
Why this exists
=====================================================================
The current interval is a hand-built sigma proxy (``difficulty``: a
LightGBM regressor fitted on absolute residuals) followed by a single
global conformal quantile.  That construction buys width by spending
coverage: difficulty-normalised got median width 2.0052 -> 1.9195 but
coverage 0.9004 -> 0.8789.  The two arms were never simultaneously at
nominal.

Conformalized quantile regression (Romano, Patterson & Candes, 2019)
attacks the same trade differently:

    fit q_lo(x), q_hi(x) at levels alpha/2 and 1-alpha/2
    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))      (CQR score)
    Q   = the (1-alpha) conformal quantile of E on the calibration rows
    interval = [q_lo(x) - Q, q_hi(x) + Q]

Coverage is still guaranteed by the conformal step; the *adaptivity* now
comes from a conditional quantile model rather than from a residual
magnitude model, and the conformal step only has to correct a level, not
a whole shape.

=====================================================================
Design
=====================================================================
  * Features and folds are exactly the ``doi`` scheme of
    ``msrs_generalization`` (5 GroupKFold folds on DOI), arm ``ligand+both``.
  * Point predictions are refitted per seed with the same learner, so the
    three comparison arms see the same residuals.
  * Calibration for outer fold k uses every other fold's out-of-fold
    quantile predictions -- the same cross-conformal arrangement as
    ``msrs_calibration.run_protocol``, so the numbers are comparable.
  * A metal-Mondrian variant calibrates Q inside each metal, falling back
    to the global Q when a metal has fewer than MIN_METAL_CAL rows.

=====================================================================
Gate (todo.json, frozen 2026-10-22T12:42+08:00)
=====================================================================
median_width                     <= 1.80 log10 (best so far 1.9195)
marginal_coverage                >= 0.90
worst_metal_group_coverage       >= 0.87
must_beat                        width below difficulty-normalised at
                                 coverage >= 0.90
kill                             width > 1.82 or worst group < 0.85

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_cqr.py
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

from molmetal.scripts.msrs_actionability import (  # noqa: E402
    conformal_quantile,
    coverage_report,
)
from molmetal.scripts.msrs_calibration import (  # noqa: E402
    ALPHA,
    MIN_METAL_CAL,
    cohort_sizes,
    run_protocol,
    winkler_score,
)
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_generalization import (  # noqa: E402
    ENDPOINT,
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
    LGB_PARAMS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
QUANTILES = (0.05, 0.95)
GATE_FROZEN_AT = "2026-10-22T12:42+08:00"
WIDTH_GATE = 1.80
COVERAGE_GATE = 0.90
WORST_GROUP_GATE = 0.87
KILL_WIDTH = 1.82
KILL_GROUP = 0.85


def quantile_oof(
    X: sparse.csr_matrix,
    codes: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    q: float,
    seed: int,
) -> np.ndarray:
    """Out-of-fold quantile predictions, one model per fold."""
    from lightgbm import LGBMRegressor

    oof = np.full(len(y), np.nan)
    params = dict(LGB_PARAMS)
    params.update({
        "objective": "quantile",
        "alpha": float(q),
        "random_state": int(seed),
    })
    cat_idx = list(range(X.shape[1], X.shape[1] + codes.shape[1]))
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = fold_id != k
        te = ~tr
        X_tr = sparse.hstack(
            [X[tr], sparse.csr_matrix(codes[tr].astype(np.float32))], format="csr"
        )
        X_te = sparse.hstack(
            [X[te], sparse.csr_matrix(codes[te].astype(np.float32))], format="csr"
        )
        model = LGBMRegressor(**params)
        model.fit(X_tr, y[tr], categorical_feature=cat_idx)
        oof[te] = model.predict(X_te)
    return oof


def cqr_interval(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    fold_id: np.ndarray,
    alpha: float,
    metal: np.ndarray,
    mondrian: bool,
) -> Dict[str, np.ndarray]:
    """Cross-conformal CQR: calibrate on every fold except the one being scored."""
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    q_used = np.full(len(y), np.nan)
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        te = fold_id == k
        cal = (fold_id >= 0) & ~te
        e_cal = np.maximum(q_lo[cal] - y[cal], y[cal] - q_hi[cal])
        q_global = conformal_quantile(e_cal, alpha)
        if mondrian:
            per_metal = {}
            for m in np.unique(metal[cal]):
                mask = metal[cal] == m
                per_metal[m] = (
                    conformal_quantile(e_cal[mask], alpha)
                    if int(mask.sum()) >= MIN_METAL_CAL else q_global
                )
            qk = np.array([per_metal.get(m, q_global) for m in metal[te]])
        else:
            qk = np.full(int(te.sum()), q_global)
        lo[te] = q_lo[te] - qk
        hi[te] = q_hi[te] + qk
        q_used[te] = qk
    return {"lo": lo, "hi": hi, "q": q_used}


def _worst_group(arm: Dict[str, object]) -> float:
    """Worst per-metal coverage; run_protocol arms only carry ``by_group``."""
    if arm.get("worst_group_coverage") is not None:
        return float(arm["worst_group_coverage"])
    groups = (arm.get("by_group") or {}).values()
    return float(min(g["coverage"] for g in groups)) if groups else float("nan")


def interval_report(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, metal: np.ndarray, alpha: float
) -> Dict[str, object]:
    ok = np.isfinite(lo) & np.isfinite(hi)
    half = (hi[ok] - lo[ok]) / 2.0
    rep = coverage_report(y[ok], (hi[ok] + lo[ok]) / 2.0, half, metal[ok], "cqr")
    rep["winkler"] = winkler_score(y[ok], lo[ok], hi[ok], alpha)
    groups = rep["by_group"]
    worst = min(groups.items(), key=lambda kv: kv[1]["coverage"]) if groups else (None, {})
    rep["worst_group"] = {"group": worst[0], **worst[1]} if groups else None
    rep["worst_group_coverage"] = worst[1].get("coverage") if groups else None
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of seeds to run (prefix of SEEDS)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_cqr.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    metal = sub["metal"].astype(str).to_numpy()
    cell = sub["cell_line_key"].astype(str).to_numpy()
    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(cell.tolist(), top_n=TOP_CELL_LINES)
    codes = native_codes(metal.tolist(), cell.tolist(), vocab, mode="full")

    bundle = build(sub, LIGAND_ONLY)
    state, _ = state_block(sub)
    coord, _, _ = coordination_block(sub, include_geometry=True)
    both = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    X = sparse.hstack([bundle.X, both], format="csr")

    fold_id, _, fold_info = scheme_folds(sub, "doi")
    frame = pd.DataFrame({
        "ligand_canonical": sub["ligand_canonical"].astype(str),
        "cell_line_key": cell,
        "doi": sub["doi"].astype(str),
        "time_h": sub["time_h"].astype(float),
        "metal": metal,
    })
    ck = cohort_sizes(frame)
    empty = sparse.csr_matrix((len(sub), 0), dtype=np.float32)

    print(f"[UP-20] rows={len(sub)} folds={fold_info['n_folds']} "
          f"features={X.shape[1]} seeds={args.seeds}")

    per_seed: Dict[str, object] = {}
    for si, seed in enumerate(SEEDS[: args.seeds]):
        ts = time.time()
        pred = fit_arm(
            X, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        q_lo = quantile_oof(X, codes, y, fold_id, QUANTILES[0], seed)
        q_hi = quantile_oof(X, codes, y, fold_id, QUANTILES[1], seed)

        plain = cqr_interval(y, q_lo, q_hi, fold_id, args.alpha, metal, mondrian=False)
        mond = cqr_interval(y, q_lo, q_hi, fold_id, args.alpha, metal, mondrian=True)
        arms = {
            "cqr": interval_report(y, plain["lo"], plain["hi"], metal, args.alpha),
            "cqr_metal_mondrian": interval_report(y, mond["lo"], mond["hi"], metal, args.alpha),
        }
        base = run_protocol(y, pred, fold_id, metal, cell, ck, alpha=args.alpha)
        for name, v in base["variants"].items():
            arms[name] = v

        per_seed[str(seed)] = {
            "point_r2": round(float(np.corrcoef(y, pred)[0, 1] ** 2), 4),
            "arms": arms,
            "q_median_by_fold": {
                str(k): round(float(np.nanmedian(plain["q"][fold_id == k])), 4)
                for k in sorted(set(fold_id[fold_id >= 0].tolist()))
            },
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        print(f"[UP-20] seed={seed} "
              + " ".join(f"{n}: cov={arms[n].get('coverage')} w={arms[n].get('median_width')}"
                         for n in ("cqr", "cqr_metal_mondrian", "difficulty", "global"))
              + f"  ({per_seed[str(seed)]['elapsed_seconds']}s)")

    names = ("cqr", "cqr_metal_mondrian", "global", "metal", "difficulty")
    agg = {}
    for name in names:
        covs = [per_seed[s]["arms"][name]["coverage"] for s in per_seed]
        widths = [per_seed[s]["arms"][name]["median_width"] for s in per_seed]
        worst = [
            _worst_group(per_seed[s]["arms"][name]) for s in per_seed
        ]
        wink = [per_seed[s]["arms"][name].get("winkler") for s in per_seed
                if per_seed[s]["arms"][name].get("winkler") is not None]
        agg[name] = {
            "coverage_mean": round(float(np.mean(covs)), 4),
            "coverage_min": round(float(np.min(covs)), 4),
            "median_width_mean": round(float(np.mean(widths)), 4),
            "worst_group_coverage_mean": round(float(np.mean(worst)), 4),
            "winkler_mean": round(float(np.mean(wink)), 4) if wink else None,
        }

    cqr = agg["cqr"]
    diff = agg["difficulty"]
    checks = {
        "median_width": {
            "value": cqr["median_width_mean"],
            "threshold": WIDTH_GATE,
            "pass": bool(cqr["median_width_mean"] <= WIDTH_GATE),
        },
        "marginal_coverage": {
            "value": cqr["coverage_mean"],
            "threshold": COVERAGE_GATE,
            "pass": bool(cqr["coverage_mean"] >= COVERAGE_GATE),
        },
        "worst_metal_group_coverage": {
            "value": cqr["worst_group_coverage_mean"],
            "threshold": WORST_GROUP_GATE,
            "pass": bool(cqr["worst_group_coverage_mean"] >= WORST_GROUP_GATE),
        },
        "beats_difficulty_at_coverage": {
            "cqr_width": cqr["median_width_mean"],
            "cqr_coverage": cqr["coverage_mean"],
            "difficulty_width": diff["median_width_mean"],
            "difficulty_coverage": diff["coverage_mean"],
            "coverage_gap": round(cqr["coverage_mean"] - diff["coverage_mean"], 4),
            "pass": bool(
                cqr["coverage_mean"] >= COVERAGE_GATE
                and cqr["median_width_mean"] < diff["median_width_mean"]
            ),
            "note": "Compared at each arm's own coverage; the gap is reported because "
                    "a narrower interval at lower coverage is not a win.",
        },
    }
    kill = {
        "rule": "width > 1.82 or worst group coverage < 0.85",
        "triggered": bool(
            cqr["median_width_mean"] > KILL_WIDTH
            or cqr["worst_group_coverage_mean"] < KILL_GROUP
        ),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_cqr",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 interval calibration (UP-20)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "metal_descriptors": str(args.metal_desc.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": ENDPOINT,
            "n_rows": int(len(sub)),
            "scheme": "doi (GroupKFold, 5 folds)",
            "arm": "ligand+both (Morgan + oxidation state + coordination)",
            "alpha": args.alpha,
            "nominal_coverage": round(1.0 - args.alpha, 4),
            "quantile_levels": list(QUANTILES),
            "seeds": list(SEEDS[: args.seeds]),
            "learner": "LightGBM; objective=quantile for q_lo/q_hi, objective=regression for the point",
            "calibration": "cross-conformal: outer fold k is calibrated on the OOF scores of "
                           "every other fold",
            "mondrian": f"metal groups with >= {MIN_METAL_CAL} calibration rows",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "Point predictions are refitted per seed, so the comparison arms and CQR see "
            "the same residuals; only the interval construction differs.",
            "Cross-conformal coverage is a lower bound: for outer fold k the calibration "
            "rows come from models that saw fold k, so their scores are mildly optimistic.",
            "The quantile models use the same 500-tree budget as the point model. A "
            "quantile objective needs more capacity than a mean objective on the same "
            "features; the budget was not tuned, so this is a floor on CQR's width.",
            "Coverage is marginal over all rows; the per-metal table is the honest one, "
            "and metal-Mondrian is the variant that targets it.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for name in names:
        v = agg[name]
        print(f"[UP-20] {name:20s} cov={v['coverage_mean']} width={v['median_width_mean']} "
              f"worst={v['worst_group_coverage_mean']} winkler={v['winkler_mean']}")
    print(f"[UP-20] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
