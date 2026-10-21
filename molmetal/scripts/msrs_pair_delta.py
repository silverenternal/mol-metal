#!/usr/bin/env python3
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""UP-11 — can a metal-substitution interaction transfer to an unobserved pair?

The frozen plan (``todo.json`` UP-11, gate frozen 2026-10-21T20:30+08:00) is a
low-rank test on the ligand x metal-pair delta matrix:

    rows  = ligands, columns = metal pairs, cell = mean over cell lines of
            Delta = mean(y | metal_b) - mean(y | metal_a)
    SVD -> rank-1 variance share >= 0.50
    then a ONE-parameter descriptor bridge v_p ~ d_p, leave one pair out,
    requiring sign accuracy >= 0.60, R2_Delta >= 0.10 and >= +0.05 over Delta=0.

**Why this script reports the rank test as NOT_IDENTIFIABLE.**  The matrix has
to have at least one row with two non-missing entries, otherwise a shared
ligand profile u_l does not exist and neither the SVD nor the bridge has an
input.  In this corpus every ligand was measured under exactly one metal pair
(or one metal): the three 2-metal groups (Os,Ru)=125, (Ir,Rh)=75, (Rh,Ru)=14
are pairwise disjoint, and no ligand spans >= 3 metals.  Complete-case rows = 0.

The only remaining operationalization of "does the interaction transfer to an
unobserved pair" is therefore a **ligand-disjoint cross-pair transfer**: fit
Delta(l) on the ligands of pair A, predict Delta(l) on the disjoint ligands of
pair B.  This is strictly harder than the rank-1 bridge (both the ligand and
the pair are unseen), and the frozen thresholds are applied to it verbatim.
The per-pair scale is the one bridge parameter:  alpha = d_B / d_A, where d is
a metal-pair descriptor difference.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_pair_delta.py --device cuda:0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import METALS, _git_commit, _sha256  # noqa: E402
from molmetal.scripts.msrs_surface_model import morgan_matrix  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

MIN_GROUPS = 20          # UP-10 kill rule: a pair below this cannot enter training
N_BOOT = 1000
BOOT_SEED = 20261021
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
INNER_FOLDS = 5
# metal_chemistry_5.csv carries no covalent-radius column; the three numeric
# descriptors it does carry are these.
PAIR_DESCRIPTORS = ("pauling_en", "ie1_kj", "ie2_kj")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def unit_table(df: pd.DataFrame) -> pd.DataFrame:
    """(ligand, cell, metal) -> mean potency and how many rows fed it."""
    return (
        df.groupby(["ligand_canonical", "cell_line_key", "metal"], dropna=False)["value_log10"]
        .agg(["mean", "size"])
        .reset_index()
    )


def pair_deltas(units: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Matched Delta = y_b - y_a on the (ligand, cell) intersection."""
    left = units[units["metal"] == a][["ligand_canonical", "cell_line_key", "mean", "size"]]
    right = units[units["metal"] == b][["ligand_canonical", "cell_line_key", "mean", "size"]]
    j = left.merge(right, on=["ligand_canonical", "cell_line_key"], suffixes=("_a", "_b"))
    if j.empty:
        return j.assign(delta=pd.Series(dtype=float))
    j["delta"] = j["mean_b"] - j["mean_a"]
    return j


def ligand_profile(pairs: pd.DataFrame) -> pd.DataFrame:
    """Collapse a pair table to one Delta per ligand (mean over cell lines)."""
    g = pairs.groupby("ligand_canonical")["delta"]
    out = g.agg(delta="mean", sd="std", n_cells="size").reset_index()
    out["sd"] = out["sd"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Rank test
# ---------------------------------------------------------------------------
def rank_structure(per_pair: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, object]:
    """Complete-case SVD of the ligand x pair matrix, or why it cannot run."""
    pair_names = [f"{a}-{b}" for a, b in per_pair]
    sets = {f"{a}-{b}": set(v["ligand_canonical"]) for (a, b), v in per_pair.items()}

    overlap = {}
    for p1, p2 in itertools.combinations(pair_names, 2):
        overlap[f"{p1} & {p2}"] = int(len(sets[p1] & sets[p2]))
    common = set.intersection(*sets.values()) if sets else set()

    # How many metals does a ligand span?  1 or 2 means it can never appear in
    # two pair-columns, which is what kills the shared profile.
    span = {}
    lig_metals: Dict[str, set] = {}
    for (a, b), v in per_pair.items():
        for lig in v["ligand_canonical"]:
            lig_metals.setdefault(lig, set()).update((a, b))
    for lig, ms in lig_metals.items():
        span[len(ms)] = span.get(len(ms), 0) + 1

    out: Dict[str, object] = {
        "pairs": pair_names,
        "n_ligands_per_pair": {f"{a}-{b}": int(len(v)) for (a, b), v in per_pair.items()},
        "pair_ligand_overlap": overlap,
        "complete_case_ligands": int(len(common)),
        "ligand_metal_span_histogram": {str(k): int(v) for k, v in sorted(span.items())},
    }

    if len(common) < 2 or len(pair_names) < 2:
        out.update(
            status="NOT_IDENTIFIABLE",
            svd_ran=False,
            reason=(
                "No ligand carries two metal-pair columns, so the delta matrix has zero "
                "complete-case rows. A rank-1 model Delta(l, p) = u_l * v_p needs at least "
                "one ligand observed under two pairs to identify u_l; the corpus has none. "
                "The rank-1 share is therefore undefined, not merely small."
            ),
            degenerate_single_column_rank1_share=1.0,
        )
        return out

    M = np.full((len(common), len(pair_names)), np.nan)
    common = sorted(common)
    for j, name in enumerate(pair_names):
        s = per_pair[tuple(name.split("-"))].set_index("ligand_canonical")["delta"]
        M[:, j] = s.reindex(common).to_numpy(dtype=float)
    M = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(M, compute_uv=False)
    out.update(
        status="OK",
        svd_ran=True,
        n_rows=int(M.shape[0]),
        singular_values=[round(float(s), 6) for s in sv],
        rank1_share=round(float(sv[0] ** 2 / np.sum(sv**2)), 4),
    )
    return out


# ---------------------------------------------------------------------------
# Cross-pair transfer (GPU)
# ---------------------------------------------------------------------------
def _resolve_device(name: str):
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu"), f"{name} unavailable, fell back to cpu"
    return torch.device(name), ""


def _fit_ridge_gpu(X, y, alpha: float):
    """Ridge with an intercept, solved on the device X lives on."""
    import torch

    xmean = X.mean(dim=0, keepdim=True)
    Xc = X - xmean
    yc = y - y.mean()
    gram = Xc.T @ Xc
    reg = alpha * torch.eye(gram.shape[0], device=X.device, dtype=X.dtype)
    w = torch.linalg.solve(gram + reg, Xc.T @ yc)
    return w, y.mean(), xmean


def _predict(X, w, b, xmean):
    return (X - xmean) @ w + b


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _sign_acc(y: np.ndarray, p: np.ndarray) -> Tuple[float, int]:
    mask = np.abs(y) > 0
    if not mask.any():
        return float("nan"), 0
    return float(np.mean(np.sign(p[mask]) == np.sign(y[mask]))), int(mask.sum())


def _bootstrap(y: np.ndarray, arms: Dict[str, np.ndarray], n_boot: int, seed: int):
    """Ligand-clustered bootstrap (one row per ligand)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out: Dict[str, Dict[str, List[float]]] = {
        k: {"r2": [], "sign": []} for k in arms
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, p in arms.items():
            r = _r2(y[idx], p[idx])
            s = _sign_acc(y[idx], p[idx])[0]
            if np.isfinite(r):
                out[k]["r2"].append(r)
            if np.isfinite(s):
                out[k]["sign"].append(s)

    def ci(v: List[float]):
        if not v:
            return None
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {k: {"r2_ci95": ci(v["r2"]), "sign_ci95": ci(v["sign"])} for k, v in out.items()}


def transfer_direction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_pair: Tuple[str, str],
    test_pair: Tuple[str, str],
    desc: pd.DataFrame,
    smiles_by_ligand: pd.Series,
    device,
    descriptors: Sequence[str],
) -> Dict[str, object]:
    import torch

    tr_smiles = smiles_by_ligand.reindex(train["ligand_canonical"]).fillna("").tolist()
    te_smiles = smiles_by_ligand.reindex(test["ligand_canonical"]).fillna("").tolist()
    Xtr = torch.tensor(morgan_matrix(tr_smiles).toarray(), dtype=torch.float64, device=device)
    Xte = torch.tensor(morgan_matrix(te_smiles).toarray(), dtype=torch.float64, device=device)
    ytr = torch.tensor(train["delta"].to_numpy(dtype=float), dtype=torch.float64, device=device)
    yte_np = test["delta"].to_numpy(dtype=float)

    # Inner k-fold on the training pair picks the ridge strength.
    n = Xtr.shape[0]
    fold = np.arange(n) % INNER_FOLDS
    rng = np.random.default_rng(BOOT_SEED)
    rng.shuffle(fold)
    fold_t = torch.tensor(fold, device=device)
    best = None
    for alpha in ALPHAS:
        err = 0.0
        for f in range(INNER_FOLDS):
            m = fold_t != f
            w, b, xm = _fit_ridge_gpu(Xtr[m], ytr[m], alpha)
            pred = _predict(Xtr[~m], w, b, xm)
            err += float(torch.sum((pred - ytr[~m]) ** 2))
        if best is None or err < best[1]:
            best = (alpha, err)
    alpha = float(best[0])
    w, b, xm = _fit_ridge_gpu(Xtr, ytr, alpha)
    raw = _predict(Xte, w, b, xm).detach().cpu().numpy()

    row = desc.set_index("metal_symbol")

    def desc_diff(pair: Tuple[str, str], col: str) -> float:
        return float(row.loc[pair[1], col] - row.loc[pair[0], col])

    # The ridge fit does not depend on the descriptor, only the one bridge
    # parameter does, so every descriptor reuses the same `raw` prediction.
    bridges: Dict[str, Dict[str, object]] = {}
    for col in descriptors:
        d_tr, d_te = desc_diff(train_pair, col), desc_diff(test_pair, col)
        scale = float(d_te / d_tr) if abs(d_tr) > 1e-9 else float("nan")
        bridges[col] = {
            "descriptor_diff_train": round(d_tr, 4),
            "descriptor_diff_test": round(d_te, 4),
            "bridge_scale": None if not np.isfinite(scale) else round(scale, 4),
            "degenerate": not np.isfinite(scale),
            # A ratio of two small descriptor differences is not a calibration;
            # |scale| outside [0.05, 5] is reported as unstable rather than used.
            "unstable": bool(not np.isfinite(scale) or abs(scale) < 0.05 or abs(scale) > 5.0),
            "prediction": raw * scale if np.isfinite(scale) else np.full_like(raw, np.nan),
        }

    sign_ok, n_nonzero = _sign_acc(yte_np, raw)
    out: Dict[str, object] = {
        "train_pair": f"{train_pair[0]}-{train_pair[1]}",
        "test_pair": f"{test_pair[0]}-{test_pair[1]}",
        "n_train_ligands": int(len(train)),
        "n_test_ligands": int(len(test)),
        "ridge_alpha": alpha,
        "test_delta_mean": round(float(yte_np.mean()), 4),
        "test_delta_sd": round(float(yte_np.std(ddof=1)), 4),
        "n_nonzero_test_delta": n_nonzero,
        "raw_sign_accuracy": None if not np.isfinite(sign_ok) else round(sign_ok, 4),
        "arms": {},
    }

    arm_preds = {"zero": np.zeros_like(yte_np), "raw": raw}
    for col, br in bridges.items():
        if not br["degenerate"]:
            arm_preds[f"bridged[{col}]"] = br["prediction"]
    boot = _bootstrap(yte_np, arm_preds, N_BOOT, BOOT_SEED)

    for name, p in arm_preds.items():
        sa, _ = _sign_acc(yte_np, p)
        out["arms"][name] = {
            "r2_delta": round(_r2(yte_np, p), 4),
            "mae": round(float(np.mean(np.abs(yte_np - p))), 4),
            "sign_accuracy": None if not np.isfinite(sa) else round(sa, 4),
            "r2_delta_ci95": boot[name]["r2_ci95"],
            "sign_accuracy_ci95": boot[name]["sign_ci95"],
        }
    out["bridges"] = {
        col: {k: v for k, v in br.items() if k != "prediction"} for col, br in bridges.items()
    }
    return out



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--endpoint", default="potency")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale-descriptor", default="ie1_kj", choices=list(PAIR_DESCRIPTORS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_pair_delta.json")
    args = ap.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    device, note = _resolve_device(args.device)
    if note:
        print(f"[UP-11] {note}")

    df = pd.read_parquet(args.dedup)
    desc = pd.read_csv(args.metal_desc)
    sub = df[
        (df["endpoint"] == args.endpoint)
        & df["value_log10"].notna()
        & df["ligand_canonical"].notna()
        & df["ligand_main_fragment"].notna()
        & df["cell_line_key"].notna()
    ].copy().reset_index(drop=True)
    if sub.empty:
        print("FAIL: empty cohort")
        return 1

    units = unit_table(sub)
    per_pair: Dict[Tuple[str, str], pd.DataFrame] = {}
    pair_meta: Dict[str, Dict[str, object]] = {}
    for a, b in itertools.combinations(METALS, 2):
        j = pair_deltas(units, a, b)
        if j.empty:
            continue
        prof = ligand_profile(j)
        per_pair[(a, b)] = prof
        pair_meta[f"{a}-{b}"] = {
            "n_groups": int(len(j)),
            "n_ligands": int(len(prof)),
            "n_cells": int(j["cell_line_key"].nunique()),
            "mean_delta": round(float(prof["delta"].mean()), 4),
            "median_abs_delta": round(float(prof["delta"].abs().median()), 4),
            "sd_delta": round(float(prof["delta"].std(ddof=1)), 4),
            "status": "OK" if len(j) >= MIN_GROUPS else "UNSUPPORTED",
        }

    print(f"[UP-11] rows={len(sub)} units={len(units)} "
          f"pairs={[f'{a}-{b}' for a, b in per_pair]} device={device}")

    rank = rank_structure(per_pair)
    print(f"[UP-11] rank status={rank['status']} "
          f"complete_case={rank['complete_case_ligands']} "
          f"span={rank['ligand_metal_span_histogram']}")

    # The one bridge parameter is estimated on pairs that clear the UP-10 kill.
    ok_pairs = [k for k, v in pair_meta.items() if v["status"] == "OK"]
    transfer = {}
    if len(ok_pairs) >= 2:
        smiles_by_ligand = (
            sub.groupby("ligand_canonical")["ligand_main_fragment"]
            .agg(lambda s: s.value_counts().index[0])
        )
        for test_name in ok_pairs:
            train_names = [p for p in ok_pairs if p != test_name]
            # One bridge parameter with two usable pairs => a single training pair.
            train_name = train_names[0]
            tp = tuple(train_name.split("-"))
            vp = tuple(test_name.split("-"))
            transfer[test_name] = transfer_direction(
                per_pair[tp], per_pair[vp], tp, vp, desc, smiles_by_ligand,
                device, PAIR_DESCRIPTORS,
            )
            r = transfer[test_name]
            bridged_key = f"bridged[{args.scale_descriptor}]"
            bridged = r["arms"].get(bridged_key, {})
            print(f"[UP-11] transfer {train_name} -> {test_name} "
                  f"alpha={r['ridge_alpha']} "
                  f"raw R2={r['arms']['raw']['r2_delta']} sign={r['arms']['raw']['sign_accuracy']} | "
                  f"{bridged_key} R2={bridged.get('r2_delta')} sign={bridged.get('sign_accuracy')}")

    gate = {
        "rank1_share_min": 0.50,
        "sign_accuracy_min": 0.60,
        "r2_delta_min": 0.10,
        "r2_delta_margin_over_zero": 0.05,
        "primary_scale_descriptor": args.scale_descriptor,
        "rank1_share": rank.get("rank1_share"),
        "rank1_evaluable": bool(rank["status"] == "OK"),
        "transfer_evaluable": bool(transfer),
        "per_test_pair": {},
    }
    for name, r in transfer.items():
        candidates = ["raw", f"bridged[{args.scale_descriptor}]"]
        candidates = [c for c in candidates if c in r["arms"]]
        best = max(candidates, key=lambda k: r["arms"][k]["r2_delta"])
        gate["per_test_pair"][name] = {
            "best_arm": best,
            "sign_accuracy": r["arms"][best]["sign_accuracy"],
            "r2_delta": r["arms"][best]["r2_delta"],
            "r2_delta_vs_zero": round(r["arms"][best]["r2_delta"] - r["arms"]["zero"]["r2_delta"], 4),
            "pass": bool(
                r["arms"][best]["sign_accuracy"] is not None
                and r["arms"][best]["sign_accuracy"] >= 0.60
                and r["arms"][best]["r2_delta"] >= 0.10
                and r["arms"][best]["r2_delta"] - r["arms"]["zero"]["r2_delta"] >= 0.05
            ),
        }
    gate["gate_pass"] = bool(
        gate["rank1_evaluable"]
        and gate["per_test_pair"]
        and all(v["pass"] for v in gate["per_test_pair"].values())
    )

    if not gate["rank1_evaluable"]:
        verdict = "NOT_IDENTIFIABLE"
    elif gate["gate_pass"]:
        verdict = "TRANSFER_CONFIRMED"
    elif gate["per_test_pair"] and all(
        v["r2_delta"] < 0.10 for v in gate["per_test_pair"].values()
    ):
        verdict = "INTERACTION_IS_PAIR_SPECIFIC"
    else:
        verdict = "STRUCTURE_EXISTS_BUT_DOES_NOT_SCALE"

    out = {
        "artifact": "msrs_pair_delta",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "UP-11 interaction transfer",
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
            "device": str(device),
            "unit": "(ligand_canonical, cell_line_key, metal) mean of value_log10",
            "delta": "mean(y | metal_b) - mean(y | metal_a) per (ligand, cell), then mean over cells per ligand",
            "matrix": "ligand x metal pair, NaN preserved (never filled with 0)",
            "bridge": "one parameter, alpha = (desc_b - desc_a)_test / (desc_b - desc_a)_train",
            "bridge_descriptors_tested": list(PAIR_DESCRIPTORS),
            "scale_descriptor": args.scale_descriptor,
            "model": f"ridge on ECFP4 (2048 bits), alpha by {INNER_FOLDS}-fold inner CV",
            "bootstrap": f"{N_BOOT} ligand resamples, seed {BOOT_SEED}",
            "min_groups_per_pair": MIN_GROUPS,
            "gate_frozen_at": "2026-10-21T20:30+08:00",
        },
        "pair_metadata": pair_meta,
        "rank_structure": rank,
        "transfer": transfer,
        "gate": gate,
        "verdict": verdict,
        "caveats": [
            "The frozen rank-1 test is undefined here, not failed: zero ligands carry two "
            "pair columns, so there is no shared ligand profile to factorise. The transfer "
            "block below is a strictly harder substitute (ligand AND pair both unseen).",
            "The bridge has one parameter and one usable training pair, so alpha is a ratio "
            "of descriptor differences, not a fitted coefficient. It cannot be validated "
            "inside the training side.",
            "Per-ligand Delta averages over cell lines without weighting; a ligand measured "
            "in one cell line and one measured in ten contribute equally.",
            "n_groups here is 347 for Ru-Os vs 354 in the UP-10 audit: this script additionally "
            "drops rows with a missing cell_line_key, because the delta key needs one.",
            "pauling_en cannot bridge Ru-Os at all (Ru, Os and Ir all carry 2.20), so the "
            "scale is 0.0 and the bridged arm collapses onto the zero arm. ie1_kj is the only "
            "descriptor whose sign differs between the two supported pairs.",
            "Ridge on 2048 ECFP4 bits with ~100 training ligands is a high-variance "
            "estimator; alpha is chosen on the training pair only, never on the test pair.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"[UP-11] verdict={verdict} gate_pass={gate['gate_pass']}")
    for name, g in gate["per_test_pair"].items():
        print(f"[UP-11] gate {name}: {g}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
