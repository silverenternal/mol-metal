#!/usr/bin/env python3
"""UP-8 — does adapting on k points beat zero-shot, or is post-hoc calibration the wrong tool?

EXT-3 (`msrs_kshot_calibration.json`) measured that at k=50 no metal reaches R² ≥ 0
under offset / linear / isotonic recalibration.  Its conclusion was "few-shot is
infeasible".  That conclusion is too strong: recalibration only rescales a fixed
prediction, and the bias it corrects is 5–10% of the MSE, so estimating it from
50 points adds more noise than signal.

This script tests the alternative: **adapt the model, not the output**.

    zero_shot   train on the other four metals, predict the held-out metal
    offset      zero-shot + mean shift estimated from k points (EXT-3's baseline)
    ridge       zero-shot + closed-form ridge on the k residuals, fitted in the
                same design space (the "learned head" adaptation)
    proto       zero-shot + similarity-weighted average of the k residuals
                (ProtoNet-style: no gradient, the k points are the prototypes)

Sampling follows EXT-3 exactly — whole DOI groups, so a calibration paper never
also contributes test rows.

Gate (frozen before the run, 2026-10-21):

    PASS  if at k=50 at least 3 of 5 metals reach R² >= 0 on the best arm
    FAIL  otherwise -> "few-shot transfer does not work", now supported by two
         independent adaptation families (output rescaling + model adaptation)

Encoder caveat, stated up front: the design space is Morgan FP + metal block +
cell one-hot, not a D-MPNN embedding.  The D-MPNN checkpoints were deleted on
2026-10-21 and re-encoding every row is a separate run; the question here is
whether *any* adaptation on k points helps, and the answer does not depend on
which fixed encoder produced the features.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

from msrs_kshot_calibration import (  # noqa: E402
    ENDPOINTS,
    METALS,
    SPLITS,
    TOP_CELL_LINES,
    build_design,
    fit_predict,
    r2,
    sample_calibration,
    spearman,
)

OUT = REPO / "metrics" / "by_metric" / "msrs_meta_kshot.json"
K_VALUES = (5, 10, 20, 50)
N_REPEATS = 20
SEED = 20261021
GATE = {"k": 50, "n_metals": 3, "frozen_at": "2026-10-21T15:10+08:00"}


def _unit(m: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m


def adapt_ridge(Xc: sparse.csr_matrix, resid: np.ndarray, Xt: sparse.csr_matrix,
                alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge on the k residuals; intercept unpenalised."""
    A = sparse.hstack([Xc, sparse.csr_matrix(np.ones((Xc.shape[0], 1)))], format="csr")
    B = sparse.hstack([Xt, sparse.csr_matrix(np.ones((Xt.shape[0], 1)))], format="csr")
    A = A.toarray()
    B = B.toarray()
    n_feat = A.shape[1]
    reg = np.eye(n_feat) * alpha
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ resid)
    return B @ coef


def adapt_proto(Xc: sparse.csr_matrix, resid: np.ndarray, Xt: sparse.csr_matrix,
                temperature: float = 0.05) -> np.ndarray:
    """Similarity-weighted residual average (cosine softmax over the k prototypes)."""
    Xc_u = _unit(Xc)
    Xt_u = _unit(Xt)
    sim = np.asarray((Xt_u @ Xc_u.T).todense())
    sim = sim / temperature
    sim -= sim.max(axis=1, keepdims=True)
    w = np.exp(sim)
    w /= w.sum(axis=1, keepdims=True)
    return w @ resid


def run_cell(X, y, metals, fragments, dois, groups, holdout, ligand_disjoint, seed):
    train = np.where(metals != holdout)[0]
    test = np.where(metals == holdout)[0]
    if ligand_disjoint:
        held = set(fragments[test])
        train = train[np.array([f not in held for f in fragments[train]])]
    if len(train) < 200 or len(test) < 50:
        return {"status": "INSUFFICIENT", "n_train": int(len(train)), "n_test": int(len(test))}

    pred = fit_predict(X[train], y[train], X[test], groups)
    y_te = y[test]
    d_te = dois[test]
    base_r2 = r2(pred, y_te)

    Xt = X[test]
    cells: Dict[str, object] = {}
    for arm in ("offset", "ridge", "proto"):
        curve = {}
        for k in K_VALUES:
            scores: List[float] = []
            worse = 0
            rng = np.random.default_rng(seed + k)
            for _rep in range(N_REPEATS):
                cal = sample_calibration(d_te, k, rng)
                if len(cal) == 0:
                    continue
                # calibration rows must not also be scored
                eval_mask = np.ones(len(y_te), dtype=bool)
                eval_mask[cal] = False
                if eval_mask.sum() < 30:
                    continue
                resid = y_te[cal] - pred[cal]
                if arm == "offset":
                    delta = np.full(int(eval_mask.sum()), resid.mean())
                elif arm == "ridge":
                    delta = adapt_ridge(Xt[cal], resid, Xt[eval_mask])
                else:
                    delta = adapt_proto(Xt[cal], resid, Xt[eval_mask])
                r = r2(pred[eval_mask] + delta, y_te[eval_mask])
                r0 = r2(pred[eval_mask], y_te[eval_mask])
                scores.append(r)
                worse += int(r < r0)
            if not scores:
                curve[str(k)] = {"status": "NO_REPEATS"}
                continue
            arr = np.array(scores)
            curve[str(k)] = {
                "mean_r2": round(float(arr.mean()), 4),
                "median_r2": round(float(np.median(arr)), 4),
                "ci95": [round(float(np.quantile(arr, 0.025)), 4),
                         round(float(np.quantile(arr, 0.975)), 4)],
                "share_worse_than_zero_shot": round(worse / len(scores), 4),
                "n_repeats": len(scores),
            }
        cells[arm] = curve

    rho, _p = spearman(pred, y_te)
    return {
        "status": "OK",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "zero_shot_r2": round(base_r2, 4),
        "zero_shot_spearman": round(float(rho), 4),
        "arms": cells,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=REPO / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=REPO / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results: Dict[str, object] = {}
    for ep in ENDPOINTS:
        sub = df[(df["endpoint"] == ep) & df["value_log10"].notna()
                 & df["ligand_main_fragment"].notna()].reset_index(drop=True)
        vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
        X, groups = build_design(sub, descriptors, vocabulary)
        y = sub["value_log10"].to_numpy(dtype=float)
        metals = sub["metal"].to_numpy()
        fragments = sub["ligand_main_fragment"].to_numpy()
        dois = sub["doi"].to_numpy()

        results[ep] = {"status": "OK", "n_rows": int(len(sub)), "splits": {}}
        for split_name, disjoint in SPLITS:
            t0 = time.time()
            cells = {m: run_cell(X, y, metals, fragments, dois, groups, m, disjoint, SEED)
                     for m in METALS}
            results[ep]["splits"][split_name] = {
                "per_metal": cells,
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            print(f"[{ep}/{split_name}] " +
                  " ".join(f"{m}={cells[m].get('zero_shot_r2', float('nan')):+.3f}"
                           for m in METALS))

    # gate: best arm at k=50 on potency / metal_out
    split = results["potency"]["splits"]["metal_out"]["per_metal"]
    per_arm = {}
    for arm in ("offset", "ridge", "proto"):
        vals = {}
        for m in METALS:
            c = split[m]
            if c.get("status") != "OK":
                continue
            entry = c["arms"][arm].get(str(GATE["k"]), {})
            vals[m] = entry.get("mean_r2")
        per_arm[arm] = {
            "mean_r2": vals,
            "n_metals_nonneg": int(sum(1 for v in vals.values() if v is not None and v >= 0)),
        }
    # proto degenerates to offset when the cosine similarities are near-constant
    # (Morgan bits are close to orthogonal), so the softmax over prototypes is
    # uniform and the "weighted" average is the plain mean.  Detect it rather
    # than report a duplicate arm as an independent result.
    degenerate = all(
        per_arm["proto"]["mean_r2"].get(m) == per_arm["offset"]["mean_r2"].get(m)
        for m in METALS
    )
    best_arm = max(per_arm, key=lambda a: per_arm[a]["n_metals_nonneg"])
    verdict = "PASS" if per_arm[best_arm]["n_metals_nonneg"] >= GATE["n_metals"] else "FAIL"

    doc = {
        "artifact": "msrs_meta_kshot",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 few-shot / UP-8",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "script": str(Path(__file__).relative_to(REPO)),
            "seed": SEED,
            "n_repeats": N_REPEATS,
            "encoder": "Morgan FP + metal descriptor block + cell one-hot (build_design)",
        },
        "design": {
            "arms": {
                "zero_shot": "train on the other four metals, no adaptation",
                "offset": "zero-shot + mean residual shift from k points (EXT-3 baseline)",
                "ridge": "zero-shot + closed-form ridge on k residuals in the design space",
                "proto": "zero-shot + cosine-softmax weighted residual average over k prototypes",
            },
            "k_values": list(K_VALUES),
            "sampling": "whole DOI groups (sample_calibration), calibration rows excluded from scoring",
            "gate": GATE,
        },
        "results": results,
        "gate_table": per_arm,
        "diagnostics": {
            "proto_equals_offset": bool(degenerate),
            "reading": "True means the cosine similarities in the Morgan design space are "
                       "near-constant, the softmax over the k prototypes is uniform, and the "
                       "proto arm is the plain mean. It is NOT an independent adaptation family "
                       "in this encoder." if degenerate else "proto and offset differ.",
        },
        "best_arm": best_arm,
        "verdict": verdict,
        "verdict_rule": "PASS iff at k=50 the best arm has >= 3/5 metals with mean R2 >= 0",
        "caveats": [
            "The encoder is a fixed Morgan-FP design matrix, not a meta-trained D-MPNN. "
            "The D-MPNN checkpoints were deleted on 2026-10-21; re-encoding is a separate run.",
            "ridge and proto adapt a residual, so they cannot fix a systematically wrong "
            "ranking; they only move the level. That is the same limitation as offset, "
            "partially lifted.",
            "Calibration rows are drawn per repeat with whole-DOI grouping, so different "
            "repeats see different amounts of data for the same k.",
        ],
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[UP-8] gate k={GATE['k']} " +
          " ".join(f"{a}:{per_arm[a]['n_metals_nonneg']}/5" for a in per_arm) +
          f" -> {verdict}")
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
