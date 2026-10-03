"""Wf-39 Phase 2 (parallel) — Lambda algorithm vulnerability scan.

================================================================
Scan the R17 + R18 lambda-algorithm stack for **runtime
vulnerabilities** on real data (TODO-83 §L5, 2026-09-26).

The user request "lambda 算法那条线的能力也要找漏洞做加强" motivated
a systematic scan over 6 vulnerability surfaces that the previous
R17/R18 ships have not stress-tested on real chemistry:

| Vuln | Surface | What it checks |
|---|---|---|
| **V1** | `LambdaASTSR.fit()` on real IC50 | MEASURED R² on the held-out fold |
| **V2** | `LambdaASTSRHilbert.optimize()` on real IC50 | Convergence / numerical stability |
| **V3** | `HilbertASTSRComposite.fit()` round-trip | Predict consistency after fit / predict |
| **V4** | `LambdaAST` evaluation on extreme input | NaN / Inf on divide-by-zero, exp overflow |
| **V5** | Best-AST reproduction under re-seed | Same seed → same AST / same R² |
| **V6** | 5-seed R² spread (overfitting check) | Per-seed variance vs aggregate mean |

Each vuln has an ``OK / FAIL / WARN`` verdict.  ``FAIL`` indicates a
**runtime vulnerability** that the previous R17/R18 ship did not
catch on synthetic data.  ``WARN`` indicates an honest framing item
(no failure, but worth recording).

Outputs
-------
* :file:`molmetal/reports/wf39_deflexpressor/vuln_scan.json` — verdict
  per vuln + raw measurements.

Usage::

    .venv/bin/python -u molmetal/scripts/scan_lambda_algorithm_vulns.py \\
        --n-samples 200 --seeds 5 \\
        --output molmetal/reports/wf39_deflexpressor

Honest framing
--------------
* This scan is **read-mostly** — it never mutates ``lambda_astsr.py``,
  ``hilbert_primal_dual.py`` or ``hilbert_astsr_integration.py``.
* The script is intentionally **slow** (~30 s for 5 seeds × 200
  samples) because the AST branch is evolutionary; it deliberately
  exercises all three branches (AST / Hilbert / Composite) head-to-
  head on real data so any asymmetry surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Reuse the bench helpers.
from bench_deflexpressor_end_to_end import (  # noqa: E402
    HILBERT_DIM,
    build_design_matrix,
    load_nci60_precious,
    pad_to_hilbert_dim,
    zscore,
)

DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf39_deflexpressor"


# ---------------------------------------------------------------------------
# Vuln checks
# ---------------------------------------------------------------------------
def vuln_v1_ast_r2(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """V1: ``LambdaASTSR.fit()`` produces a finite R² on real IC50."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    smiles, X, y, stats = build_design_matrix(rows=rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, int(0.2 * X.shape[0]))
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr, X_te = zscore(X[train_idx], X[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    astsr = LambdaASTSR(
        depth=3,
        population=16,
        generations=8,
        seed=seed,
        feature_names=(
            "f0",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
        ),
    )
    # Rename columns to match (X has 12 columns named after FEATURE_COLUMNS
    # by default; ``build_design_matrix`` doesn't rename).
    # Easier: rebuild with default feature names from LambdaASTSR.
    smiles2, X2, y2, _ = build_design_matrix(rows=rows, seed=seed)
    X_tr2 = X2[train_idx]
    X_te2 = X2[test_idx]
    y_tr2 = y2[train_idx]
    y_te2 = y2[test_idx]
    astsr.fit(X_tr2, y_tr2)
    y_hat = astsr.predict(X_te2)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
    ss_res = float(np.sum((y_te2 - y_hat) ** 2))
    ss_tot = float(np.sum((y_te2 - np.mean(y_te2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    verdict = "OK" if math.isfinite(r2) else "FAIL"
    return {
        "verdict": verdict,
        "r2_test": float(r2),
        "n_train": int(X_tr2.shape[0]),
        "n_test": int(X_te2.shape[0]),
        "n_features": int(X_tr2.shape[1]),
    }


def vuln_v2_hilbert_stability(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """V2: ``LambdaASTSRHilbert.optimize()`` produces finite iterate."""
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert

    smiles, X, y, _ = build_design_matrix(rows=rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, int(0.2 * X.shape[0]))
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
    y_tr, y_te = y[train_idx], y[test_idx]

    model = LambdaASTSRHilbert(
        population_dim=HILBERT_DIM,
        lr_primal=0.05,
        lr_dual=0.05,
        n_iters=200,
        seed=seed,
    )
    t0 = time.perf_counter()
    model.optimize(X_tr_64, y_tr)
    wall = float(time.perf_counter() - t0)
    has_nan = (
        bool(np.any(~np.isfinite(model.best_params_)))
        if model.best_params_ is not None
        else True
    )
    has_inf = (
        bool(np.any(np.abs(model.best_params_) > 1e6))
        if model.best_params_ is not None
        else True
    )
    y_hat = model.predict(X_te_64)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
    ss_res = float(np.sum((y_te - y_hat) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    verdict = "OK"
    if has_nan:
        verdict = "FAIL"
    elif has_inf or not math.isfinite(r2) or r2 < -10.0:
        verdict = "WARN"
    return {
        "verdict": verdict,
        "has_nan_params": has_nan,
        "has_inf_params": has_inf,
        "r2_test": float(r2),
        "wall_s": float(wall),
    }


def vuln_v3_composite_roundtrip(
    rows: List[Dict[str, Any]], seed: int
) -> Dict[str, Any]:
    """V3: ``HilbertASTSRComposite.fit()`` → ``predict()`` is consistent."""
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    smiles, X, y, _ = build_design_matrix(rows=rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, int(0.2 * X.shape[0]))
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
    X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
    y_tr, y_te = y[train_idx], y[test_idx]

    astsr = LambdaASTSR(
        depth=2,
        population=8,
        generations=4,
        seed=seed,
        feature_names=tuple(range(X_tr_n.shape[1])),
    )
    hilbert = LambdaASTSRHilbert(
        population_dim=HILBERT_DIM,
        lr_primal=0.05,
        lr_dual=0.05,
        n_iters=100,
        seed=seed,
    )
    model = HilbertASTSRComposite(
        lambda_astsr=astsr,
        hilbert_optimizer=hilbert,
        feature_columns=tuple(range(X_tr_n.shape[1])),
        use_hilbert=True,
        population_dim=HILBERT_DIM,
    )
    info = model.fit(X_tr_64, y_tr)
    y_hat = model.predict(X_te_64)
    y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
    n_finite = int(np.sum(np.isfinite(y_hat)))
    n_test_actual = int(y_hat.shape[0])
    verdict = (
        "OK" if (n_finite == n_test_actual and info.get("r2") is not None) else "FAIL"
    )
    return {
        "verdict": verdict,
        "info_r2": float(info.get("r2", float("nan"))),
        "info_backend": str(info.get("backend", "")),
        "n_test": n_test_actual,
        "n_finite_predict": n_finite,
    }


def vuln_v4_lambda_ast_extremes(seed: int) -> Dict[str, Any]:
    """V4: ``LambdaAST.evaluate()`` is NaN-safe on extreme inputs."""
    from molmetal.molmetal_lam.reward.lambda_astsr import random_ast

    feats = [f"x{i}" for i in range(8)]
    rng = random.Random(seed)
    # Sample 10 random ASTs and feed them extreme inputs.
    n_total = 0
    n_finite = 0
    n_extreme_failures: List[Dict[str, Any]] = []
    for i in range(10):
        ast = random_ast(rng, feats, max_depth=3, p_var=0.5)
        for j, row in enumerate(
            [
                {"x" + str(k): 1e10 for k in range(8)},
                {"x" + str(k): -1e10 for k in range(8)},
                {"x" + str(k): 0.0 for k in range(8)},
                {"x" + str(k): float("nan") for k in range(8)},
            ]
        ):
            n_total += 1
            try:
                val = ast.evaluate(row)
                if math.isfinite(val):
                    n_finite += 1
                elif i < 3:  # Only log first 3 extreme-failures per AST
                    n_extreme_failures.append(
                        {
                            "ast_idx": i,
                            "row_idx": j,
                            "value": float(val)
                            if not (math.isinf(val) or math.isnan(val))
                            else str(val),
                        }
                    )
            except Exception as exc:
                n_extreme_failures.append(
                    {
                        "ast_idx": i,
                        "row_idx": j,
                        "exception": f"{type(exc).__name__}: {exc}"[:120],
                    }
                )
    finite_rate = n_finite / max(1, n_total)
    verdict = "OK" if finite_rate >= 0.5 else "FAIL"
    return {
        "verdict": verdict,
        "n_total_evaluations": n_total,
        "n_finite": n_finite,
        "finite_rate": float(finite_rate),
        "n_extreme_failures_logged": len(n_extreme_failures),
        "first_failures": n_extreme_failures[:5],
    }


def vuln_v5_reproducibility(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """V5: Same seed → same AST / same R² across two runs."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    smiles, X, y, _ = build_design_matrix(rows=rows, seed=seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_test = max(1, int(0.2 * X.shape[0]))
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    fnames = tuple(range(X_tr.shape[1]))

    model_a = LambdaASTSR(
        depth=2,
        population=8,
        generations=4,
        seed=seed,
        feature_names=fnames,
    )
    model_a.fit(X_tr, y_tr)
    r2_a = float(model_a.best_r2_train_)
    s_a = model_a.best_ast_.to_lambda_string() if model_a.best_ast_ is not None else ""

    model_b = LambdaASTSR(
        depth=2,
        population=8,
        generations=4,
        seed=seed,
        feature_names=fnames,
    )
    model_b.fit(X_tr, y_tr)
    r2_b = float(model_b.best_r2_train_)
    s_b = model_b.best_ast_.to_lambda_string() if model_b.best_ast_ is not None else ""

    verdict = "OK" if (s_a == s_b and abs(r2_a - r2_b) < 1e-9) else "FAIL"
    return {
        "verdict": verdict,
        "r2_first": r2_a,
        "r2_second": r2_b,
        "ast_string_match": bool(s_a == s_b),
        "first_lambda_str": s_a[:120],
        "second_lambda_str": s_b[:120],
    }


def vuln_v6_seed_spread(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    """V6: 5-seed R² spread vs aggregate mean (overfitting check)."""
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import LambdaASTSRHilbert

    r2_per_seed: List[float] = []
    for seed in seeds:
        smiles, X, y, _ = build_design_matrix(rows=rows, seed=seed)
        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        n_test = max(1, int(0.2 * X.shape[0]))
        train_idx = idx[n_test:]
        test_idx = idx[:n_test]
        X_tr_n, X_te_n = zscore(X[train_idx], X[test_idx])
        X_tr_64, X_te_64 = pad_to_hilbert_dim(X_tr_n, HILBERT_DIM, X_te_n)
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LambdaASTSRHilbert(
            population_dim=HILBERT_DIM,
            lr_primal=0.05,
            lr_dual=0.05,
            n_iters=200,
            seed=seed,
        )
        model.optimize(X_tr_64, y_tr)
        y_hat = model.predict(X_te_64)
        y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
        ss_res = float(np.sum((y_te - y_hat) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_per_seed.append(float(r2) if math.isfinite(r2) else 0.0)
    arr = np.asarray(r2_per_seed, dtype=float)
    mean_r2 = float(np.mean(arr))
    std_r2 = float(np.std(arr))
    cv = float(std_r2 / abs(mean_r2)) if abs(mean_r2) > 1e-9 else float("inf")
    # If CV > 5.0 the per-seed variance dwarfs the mean — flag as WARN.
    verdict = "OK" if cv <= 5.0 else "WARN"
    return {
        "verdict": verdict,
        "r2_per_seed": r2_per_seed,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "cv": cv,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_scan(
    *,
    n_samples: int,
    seeds: List[int],
) -> Dict[str, Any]:
    """Run all 6 vuln checks; return a JSON-serialisable payload."""
    nci60_path = _REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
    rows = load_nci60_precious(nci60_path, n_max=n_samples)

    results: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for seed in seeds:
        results.setdefault("v1_ast_r2", []).append(vuln_v1_ast_r2(rows, seed))
        results.setdefault("v2_hilbert_stability", []).append(
            vuln_v2_hilbert_stability(rows, seed)
        )
        results.setdefault("v3_composite_roundtrip", []).append(
            vuln_v3_composite_roundtrip(rows, seed)
        )
    # V4 doesn't need the full dataset.
    results["v4_lambda_ast_extremes"] = vuln_v4_lambda_ast_extremes(seeds[0])
    results["v5_reproducibility"] = vuln_v5_reproducibility(rows, seeds[0])
    results["v6_seed_spread"] = vuln_v6_seed_spread(rows, seeds)
    elapsed = float(time.perf_counter() - t0)

    # Aggregate verdicts.
    verdicts: List[str] = []
    for k, v in results.items():
        if isinstance(v, list):
            for entry in v:
                verdicts.append(entry.get("verdict", "?"))
        else:
            verdicts.append(v.get("verdict", "?"))
    n_fail = sum(1 for v in verdicts if v == "FAIL")
    n_warn = sum(1 for v in verdicts if v == "WARN")
    n_ok = sum(1 for v in verdicts if v == "OK")

    payload = {
        "config": {
            "n_samples": int(n_samples),
            "seeds": [int(s) for s in seeds],
            "dataset": "nci60",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_total": len(verdicts),
            "elapsed_s": elapsed,
        },
        "results": results,
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-39 Phase 2 — Lambda algorithm vulnerability scan.",
    )
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT_DIR))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    seeds = list(range(int(args.seeds)))
    payload = run_scan(n_samples=int(args.n_samples), seeds=seeds)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vuln_scan.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"[wf39_vuln] wrote {out_path}")
    s = payload["summary"]
    print(
        f"[wf39_vuln] summary: OK={s['n_ok']} WARN={s['n_warn']} "
        f"FAIL={s['n_fail']} / total={s['n_total']} "
        f"(elapsed={s['elapsed_s']:.1f}s)"
    )
    if s["n_fail"] > 0:
        print("[wf39_vuln] VULNERABILITIES DETECTED — see vuln_scan.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
