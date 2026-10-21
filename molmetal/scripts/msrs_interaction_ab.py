#!/usr/bin/env python3
"""UP-1 — is the cross-metal failure self-inflicted by an additive constraint?

=====================================================================
Why this exists
=====================================================================
``msrs_variance_decomposition.json`` measures that, for the one estimable
metal pair (Ru-Os), the ligand x metal interaction carries **0.712** of the
delta variance (95% CI 0.601-0.776, F=3.47, p=8.2e-25).

``msrs_lomo_surface.json`` then evaluates the held-out-metal surface with
LightGBM constrained to ``interaction_constraints = [[ligand], [metal],
[cell]]`` (``msrs_surface_model.py:213-233``).  Those three groups are
disjoint, so the learner cannot split on ligand and metal at the same time.
The surface that measures the failure is structurally forbidden from
representing the interaction the same project measured as dominant.

MSRS-2 already ran additive-vs-unconstrained under **ligand-grouped** CV and
found the interaction wins (MAE 0.4635 vs 0.4837, p=3e-30).  It was never run
under **metal-out**, where the metal block at test is either an all-zero
one-hot column or three descriptor values the training metals never spanned.
Those two cases are not the same experiment:

    one-hot     the held-out metal is a row pattern the model never saw, and
                every interaction term evaluates to zero -> cannot transfer
    descriptors the held-out metal has real values -> an interaction can in
                principle transfer

So the arms are nested inside a metal representation, and the headline
contrast is descriptor-vs-descriptor.

=====================================================================
Arms
=====================================================================
    additive_desc   [[lig], [desc], [cell]]              (the current surface)
    free_desc       no constraints
    pair_desc       [[lig, desc], [cell]]
    product_desc    explicit ligand x descriptor products, no constraints
    additive_onehot [[lig], [onehot], [cell]]            (reference)
    free_onehot     no constraints                        (reference)

``product_desc`` multiplies every ECFP4 bit by each standardised metal
descriptor, so the *slope* of a ligand bit on potency is allowed to depend on
the metal.  Descriptors are standardised over the five metals (identity only,
no labels).  At test the held-out metal's descriptors are real numbers, so the
product block is non-zero — unlike the one-hot case.

=====================================================================
Primary judgement (frozen in todo.json, 2026-10-21T15:10+08:00)
=====================================================================
    potency / metal_out: weighted R2 of a non-additive arm
        >= additive_desc + 0.05
    AND ligand-clustered paired bootstrap CI of delta R2 has lower bound > 0
    AND metal_out_ligand_disjoint delta has the same sign

Nothing here changes the gate.  If no arm clears it, the failure is
representational, not a constraint artefact, and the script says so.

=====================================================================
Usage
=====================================================================
::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_interaction_ab.py
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
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    LGB_PARAMS,
    METALS,
    SEED,
    TOP_CELL_LINES,
    cell_matrix,
    metal_matrix,
    morgan_matrix,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ENDPOINTS = ("potency", "cisplatin_ratio")
SPLITS = ("metal_out", "metal_out_ligand_disjoint")
ARMS = (
    "additive_desc",
    "free_desc",
    "pair_desc",
    "product_desc",
    "additive_onehot",
    "free_onehot",
)
NON_ADDITIVE = ("free_desc", "pair_desc", "product_desc", "free_onehot")
REFERENCE_ARM = "additive_desc"
SEEDS_SCREEN = (SEED,)
SEEDS_CONFIRM = (SEED, 42, 43, 44, 49)
N_BOOT = 2000
GATE_DELTA = 0.05


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
# Blocks
# ---------------------------------------------------------------------------
def build_blocks(
    df: pd.DataFrame, descriptors: pd.DataFrame, vocabulary: Sequence[str]
) -> Dict[str, object]:
    """Ligand / metal / cell blocks plus a standardised descriptor matrix.

    Descriptor standardisation uses the five metals' tabulated values only —
    no labels, and the held-out metal is included because its chemistry is
    known at test time.  This is what makes the product arm transferable.
    """
    lig = morgan_matrix(df["ligand_main_fragment"].tolist())
    met_full = metal_matrix(df["metal"].tolist(), descriptors)
    onehot = met_full[:, : len(METALS)]
    desc = met_full[:, len(METALS) :].astype(np.float64)
    mu = desc.mean(axis=0, keepdims=True)
    sd = desc.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    desc_z = ((desc - mu) / sd).astype(np.float32)
    cell = cell_matrix(df["cell_line_key"].tolist(), vocabulary)
    return {
        "ligand": lig,
        "metal_one_hot": sparse.csr_matrix(onehot),
        "metal_desc": sparse.csr_matrix(desc_z),
        "metal_desc_dense": desc_z,
        "cell": cell,
    }


def assemble(
    blocks: Dict[str, object], arm: str, rows: np.ndarray
) -> Tuple[sparse.csr_matrix, Optional[List[List[int]]]]:
    lig = blocks["ligand"][rows]
    cell = blocks["cell"][rows]
    n_lig = lig.shape[1]
    n_cell = cell.shape[1]

    if arm in ("additive_desc", "free_desc", "pair_desc"):
        met = blocks["metal_desc"][rows]
        X = sparse.hstack([lig, met, cell], format="csr")
        n_met = met.shape[1]
        if arm == "additive_desc":
            groups = [list(range(n_lig)),
                      list(range(n_lig, n_lig + n_met)),
                      list(range(n_lig + n_met, n_lig + n_met + n_cell))]
        elif arm == "pair_desc":
            groups = [list(range(n_lig + n_met)),
                      list(range(n_lig + n_met, n_lig + n_met + n_cell))]
        else:
            groups = None
        return X, groups

    if arm == "product_desc":
        dense = blocks["metal_desc_dense"][rows]
        parts = [lig]
        for k in range(dense.shape[1]):
            parts.append(sparse.csr_matrix(lig.multiply(dense[:, k : k + 1])))
        parts.append(blocks["metal_desc"][rows])
        parts.append(cell)
        return sparse.hstack(parts, format="csr"), None

    if arm in ("additive_onehot", "free_onehot"):
        met = blocks["metal_one_hot"][rows]
        X = sparse.hstack([lig, met, cell], format="csr")
        n_met = met.shape[1]
        groups = (
            [list(range(n_lig)),
             list(range(n_lig, n_lig + n_met)),
             list(range(n_lig + n_met, n_lig + n_met + n_cell))]
            if arm == "additive_onehot"
            else None
        )
        return X, groups

    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_arm(
    X_tr: sparse.csr_matrix,
    y_tr: np.ndarray,
    X_te: sparse.csr_matrix,
    constraints: Optional[List[List[int]]],
    seed: int,
) -> np.ndarray:
    from lightgbm import LGBMRegressor

    params = dict(LGB_PARAMS)
    params["random_state"] = int(seed)
    if constraints is not None:
        params["interaction_constraints"] = constraints
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


# ---------------------------------------------------------------------------
# One split, all arms, one seed
# ---------------------------------------------------------------------------
def evaluate_split(
    df: pd.DataFrame,
    blocks: Dict[str, object],
    y: np.ndarray,
    holdout: str,
    ligand_disjoint: bool,
    seed: int,
    arms: Sequence[str] = ARMS,
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

    y_test = y[test]
    preds: Dict[str, np.ndarray] = {}
    for arm in arms:
        X_tr, groups = assemble(blocks, arm, train)
        X_te, _ = assemble(blocks, arm, test)
        preds[arm] = fit_arm(X_tr, y[train], X_te, groups, seed)

    models = {
        arm: {
            "mae": round(float(mean_absolute_error(y_test, preds[arm])), 4),
            "r2": round(float(r2_score(y_test, preds[arm])), 4),
            "bias": round(float(np.mean(preds[arm] - y_test)), 4),
            "pred_mean": round(float(np.mean(preds[arm])), 4),
        }
        for arm in arms
    }
    return {
        "status": "OK",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_ligands_test": int(len(set(fragments[test]))),
        "observed_mean": round(float(np.mean(y_test)), 4),
        "train_mean": round(float(np.mean(y[train])), 4),
        "models": models,
        "preds": preds,
        "y_test": y_test,
        "test_idx": test,
        "ligand": fragments[test],
    }


def aggregate(per_metal: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    ok = {m: v for m, v in per_metal.items() if v.get("status") == "OK"}
    if not ok:
        return {}
    n = np.array([ok[m]["n_test"] for m in ok], dtype=float)
    out: Dict[str, object] = {}
    for arm in ARMS:
        r2 = np.array([ok[m]["models"][arm]["r2"] for m in ok])
        mae = np.array([ok[m]["models"][arm]["mae"] for m in ok])
        bias = np.array([ok[m]["models"][arm]["bias"] for m in ok])
        out[arm] = {
            "weighted_r2": round(float(np.average(r2, weights=n)), 4),
            "weighted_mae": round(float(np.average(mae, weights=n)), 4),
            "mean_abs_bias": round(float(np.mean(np.abs(bias))), 4),
            "max_abs_bias": round(float(np.max(np.abs(bias))), 4),
            "n_metals_negative_r2": int((r2 < 0).sum()),
            "per_metal_r2": {m: ok[m]["models"][arm]["r2"] for m in ok},
            "per_metal_bias": {m: ok[m]["models"][arm]["bias"] for m in ok},
        }
    return out


def pool(per_metal: Dict[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
    ok = [v for v in per_metal.values() if v.get("status") == "OK"]
    out = {
        "y": np.concatenate([r["y_test"] for r in ok]),
        "ligand": np.concatenate([r["ligand"] for r in ok]),
    }
    for arm in ARMS:
        out[arm] = np.concatenate([r["preds"][arm] for r in ok])
    return out


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def clustered_bootstrap_delta(
    pooled: Dict[str, np.ndarray], arm: str, n_boot: int = N_BOOT, seed: int = SEED
) -> Dict[str, object]:
    """Ligand-clustered bootstrap of R2(arm) - R2(reference).

    Rows inside one ligand are not independent, and the same ligand appears
    under several metals.  Resampling ligands keeps that structure.
    """
    y = pooled["y"]
    lig = pooled["ligand"]
    uniq, inv = np.unique(lig, return_inverse=True)
    if len(uniq) < 3:
        return {"status": "TOO_FEW_CLUSTERS", "n_clusters": int(len(uniq))}
    buckets = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([buckets[p] for p in pick])
        a = _r2(y[idx], pooled[arm][idx])
        b = _r2(y[idx], pooled[REFERENCE_ARM][idx])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(a - b)
    if not deltas:
        return {"status": "DEGENERATE"}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": int(len(deltas)),
        "n_clusters": int(len(uniq)),
        "delta_r2_point": round(_r2(y, pooled[arm]) - _r2(y, pooled[REFERENCE_ARM]), 4),
        "delta_r2_mean": round(float(np.mean(deltas)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "share_boot_positive": round(float(np.mean(np.array(deltas) > 0)), 4),
    }


# ---------------------------------------------------------------------------
# One endpoint
# ---------------------------------------------------------------------------
def run_endpoint(
    df: pd.DataFrame, endpoint: str, descriptors: pd.DataFrame, verbose: bool = True
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
    blocks = build_blocks(sub, descriptors, vocabulary)
    y = sub["value_log10"].to_numpy(dtype=float)

    out: Dict[str, object] = {
        "status": "OK",
        "n_rows": int(len(sub)),
        "n_dropped_censored": n_censored,
        "censored_share": round(n_censored / n_total, 4) if n_total else None,
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "splits": {},
        "elapsed_seconds": None,
    }

    for split_name, disjoint in (("metal_out", False), ("metal_out_ligand_disjoint", True)):
        per_metal: Dict[str, Dict[str, object]] = {}
        for metal in METALS:
            per_metal[metal] = evaluate_split(sub, blocks, y, metal, disjoint, SEED)
        pooled = pool(per_metal)
        boot = {arm: clustered_bootstrap_delta(pooled, arm) for arm in ARMS if arm != REFERENCE_ARM}
        out["splits"][split_name] = {
            "per_metal": {
                m: {k: v for k, v in r.items() if k not in ("preds", "y_test", "test_idx", "ligand")}
                for m, r in per_metal.items()
            },
            "aggregate": aggregate(per_metal),
            "bootstrap_vs_reference": boot,
        }
        if verbose:
            agg = out["splits"][split_name]["aggregate"]
            print(
                f"[{endpoint}/{split_name}] "
                + "  ".join(f"{a}={agg[a]['weighted_r2']:+.4f}" for a in ARMS)
            )

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Confirmation seeds for the reference arm and the best non-additive arm
# ---------------------------------------------------------------------------
def confirm_seeds(
    df: pd.DataFrame, endpoint: str, descriptors: pd.DataFrame, arm: str, verbose: bool = True
) -> Dict[str, object]:
    sub = df[df["endpoint"] == endpoint]
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()].copy()
    sub = sub.reset_index(drop=True)
    vocabulary = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    blocks = build_blocks(sub, descriptors, vocabulary)
    y = sub["value_log10"].to_numpy(dtype=float)

    per_seed = []
    for seed in SEEDS_CONFIRM:
        r2 = {}
        for name in (REFERENCE_ARM, arm):
            vals, weights = [], []
            for metal in METALS:
                res = evaluate_split(sub, blocks, y, metal, False, seed, arms=(name,))
                if res.get("status") == "OK":
                    vals.append(res["models"][name]["r2"])
                    weights.append(res["n_test"])
            r2[name] = (
                round(float(np.average(vals, weights=weights)), 4) if vals else None
            )
        per_seed.append({"seed": int(seed), **r2, "delta": round(r2[arm] - r2[REFERENCE_ARM], 4)})
        if verbose:
            print(f"  seed {seed}: {REFERENCE_ARM}={r2[REFERENCE_ARM]:+.4f} {arm}={r2[arm]:+.4f}")
    deltas = np.array([r["delta"] for r in per_seed], dtype=float)
    return {
        "arm": arm,
        "per_seed": per_seed,
        "mean_delta": round(float(deltas.mean()), 4),
        "sd_delta": round(float(deltas.std(ddof=1)), 4) if len(deltas) > 1 else 0.0,
        "min_delta": round(float(deltas.min()), 4),
        "max_delta": round(float(deltas.max()), 4),
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
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_interaction_ab.json"
    )
    parser.add_argument("--skip-confirm", action="store_true", help="screen only, no 5-seed confirmation")
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)
    results = {ep: run_endpoint(df, ep, descriptors) for ep in ENDPOINTS}

    # Primary cell: potency / metal_out.
    primary_agg = results["potency"]["splits"]["metal_out"]["aggregate"]
    primary_boot = results["potency"]["splits"]["metal_out"]["bootstrap_vs_reference"]
    disjoint_agg = results["potency"]["splits"]["metal_out_ligand_disjoint"]["aggregate"]

    candidates = {
        arm: {
            "delta_r2_weighted": round(primary_agg[arm]["weighted_r2"] - primary_agg[REFERENCE_ARM]["weighted_r2"], 4),
            "bootstrap": primary_boot[arm],
            "delta_r2_ligand_disjoint": round(
                disjoint_agg[arm]["weighted_r2"] - disjoint_agg[REFERENCE_ARM]["weighted_r2"], 4
            ),
        }
        for arm in NON_ADDITIVE
    }
    best_arm = max(candidates, key=lambda a: candidates[a]["delta_r2_weighted"])
    best = candidates[best_arm]
    ci = best["bootstrap"].get("ci95") or [None, None]
    passes = (
        best["delta_r2_weighted"] >= GATE_DELTA
        and ci[0] is not None
        and ci[0] > 0
        and best["delta_r2_ligand_disjoint"] > 0
    )

    confirmation = None
    if not args.skip_confirm and best["delta_r2_weighted"] > 0:
        confirmation = confirm_seeds(df, "potency", descriptors, best_arm)

    out = {
        "artifact": "msrs_interaction_ab",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 metal extrapolation / UP-1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv if argv is not None else sys.argv[1:]),
            "screen_seed": SEED,
            "confirm_seeds": list(SEEDS_CONFIRM),
        },
        "design": {
            "learner": "LightGBM, MSRS-2 hyper-parameters, n_jobs=1, deterministic",
            "arms": list(ARMS),
            "splits": list(SPLITS),
            "endpoints": list(ENDPOINTS),
            "reference_arm": REFERENCE_ARM,
            "constraints": {
                "additive_desc": "[[ligand], [metal descriptors], [cell]]",
                "pair_desc": "[[ligand, metal descriptors], [cell]]",
                "free_desc": "none",
                "product_desc": "ligand bit x standardised descriptor for each of 3 descriptors, plus the metal block; no constraints",
                "additive_onehot": "[[ligand], [metal one-hot], [cell]]",
                "free_onehot": "none",
            },
            "descriptor_standardisation": "z-score over the five metals' tabulated values (no labels)",
            "bootstrap": "ligand-clustered, 2000 reps, delta R2 vs reference on pooled metal-out rows",
            "gate": {
                "delta_r2_weighted": GATE_DELTA,
                "ci_lower": ">0",
                "ligand_disjoint": "same sign (point estimate > 0)",
                "frozen_at": "2026-10-21T15:10+08:00",
            },
        },
        "endpoints": results,
        "primary": {
            "cell": "potency / metal_out",
            "reference_weighted_r2": primary_agg[REFERENCE_ARM]["weighted_r2"],
            "candidates": candidates,
            "best_arm": best_arm,
            "delta_r2": best["delta_r2_weighted"],
            "ci95": ci,
            "delta_r2_ligand_disjoint": best["delta_r2_ligand_disjoint"],
            "passes_gate": bool(passes),
        },
        "confirmation": confirmation,
        "verdict": (
            "INTERACTION_IS_THE_LEVER"
            if passes
            else "CONSTRAINT_IS_NOT_THE_LEVER"
        ),
        "verdict_rule": (
            f"best non-additive arm on potency/metal_out must reach {REFERENCE_ARM} + {GATE_DELTA}, "
            "with ligand-clustered bootstrap CI lower bound > 0, and the ligand-disjoint delta positive"
        ),
        "caveats": [
            "One seed for the screen; only the best arm is confirmed on five seeds.",
            "The held-out metal's descriptors are known at test time by construction; this is not a "
            "zero-shot element prediction.",
            "Censored rows are dropped as in msrs_lomo_surface (potency 18.4%), so the surface is "
            "fitted on the potent side.",
            "product_desc multiplies 2048 bits by 3 descriptors; a tree can represent the same "
            "interaction through free_desc given enough depth, so a null result on product_desc "
            "alone is not evidence against the interaction.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-1] best={best_arm} delta_r2={best['delta_r2_weighted']:+.4f} ci={ci} "
          f"ligand_disjoint={best['delta_r2_ligand_disjoint']:+.4f} -> {out['verdict']}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
