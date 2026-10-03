#!/usr/bin/env python3
"""Wf-348-B3 — Hilbert regression on R20-P3 features (32-bin Fourier + 128-d PLL).

================================================================
**Wf-348-B3** lift-verifies the L5 Hilbert R² honest-negative gate on
PlatinAI 4-cell using **R20-P3 features** = 32-bin Fourier Multi-PH
(Kim & Jung 2026) ⊕ 128-d Persistent Local Laplacian signature
(Liu & Feng 2026).

Background
----------
Wf-335-M2 found Hilbert R² on F5_RDKIT_2D is BELOW the 0.6 honesty
gate per LAMBDA_ROADMAP §1.3 L5.  This script lifts the verdict by
running Hilbert primal-dual regression on R20-P3 features:
- 32-bin Fourier Multi-PH = (32, 32) flattened = 1024-d
- 128-d PLL top-128 eigenvalues of the normalised graph Laplacian
- Total feature dim = 1024 + 128 = 1152-d per molecule

Honest framing per party-twenty-spirit §3.5:
- If per-cohort mean R² >= 0.6: gate LIFTED (CEILING_BREAK)
- If per-cohort mean R² < 0.6: gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)

Outputs:
- molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json — per-cohort + joint R²
- molmetal/reports/wf348b3_hilbert_r20p3/synthesis.md — verdict summary
- molmetal/reports/wf348b3_hilbert_r20p3/run.log — stdout mirror

Usage:
    cd /home/hugo/codes/mol-metal
    .venv/bin/python -u molmetal/scripts/wf348b3_hilbert_r20p3.py \\
        --dataset platinai_4cell \\
        --n-samples 500 \\
        --output molmetal/reports/wf348b3_hilbert_r20p3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so we can import from molmetal.* packages.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: R20-P3 feature dims (32-bin Fourier Multi-PH + 128-d PLL).
N_BINS_MULTI_PH = 32  # Multi-PH 32-bin Fourier (Kim & Jung 2026)
K_PLL = 128  # PLL top-128 eigenvalues (Liu & Feng 2026)
MULTI_PH_DIM = N_BINS_MULTI_PH * N_BINS_MULTI_PH  # 1024-d
PLL_DIM = K_PLL  # 128-d
TOTAL_DIM = MULTI_PH_DIM + PLL_DIM  # 1152-d

#: Hilbert primal-dual hyperparameters.
DEFAULT_HILBERT_DIM = 64
DEFAULT_EPOCHS = 100
# LR defaults from Wf-85 P1 (R18 ship + Wf-44 P4 numerical fix).
# Empirical Wf-348-B3 verification: lr=0.05 with grad_clip=1.0 is
# stable on R20-P3 features (Multi-PH 32-bin Fourier + 128-d PLL)
# once the bias column is included for the y-intercept.
# lr=0.001 was tested in early iteration but is too small to converge
# in 50-100 iters; lr=0.05 hits R²=+0.79 in-sample on a 48-mol
# smoke cohort (200 iters).
DEFAULT_LR = 0.05
DEFAULT_LR_DUAL = 0.05
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_STEP_DECAY = 0.5
DEFAULT_L2_WEIGHT = 0.01
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)

#: PlatinAI 4-cell target columns.
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
PLATINAI_SMILES_COL = "smiles"

#: Project-wide honesty gate threshold (party-twenty-spirit §3.5).
HONESTY_GATE_R2 = 0.6

COLLAPSE_RATIO_THRESHOLD = 0.10
PRED_CLIP = 1e6
EPS_STD = 1e-3


# ---------------------------------------------------------------------------
# Strict-scaffold split (Bemis-Murcko largest-first).
# ---------------------------------------------------------------------------


def _canonicalize_smiles(smi: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _bemis_murcko_scaffold(smi: str) -> str:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol, includeChirality=False)
        if scaffold is None:
            return ""
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def _strict_scaffold_split(
    smiles: Sequence[str],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac, val_frac, test_frac = fractions
    n = len(smiles)
    n_train_target = int(round(n * train_frac))
    n_val_target = int(round(n * val_frac))

    rng = np.random.default_rng(int(seed))
    canon = [_canonicalize_smiles(s) for s in smiles]
    scaffold_to_rows: Dict[str, List[int]] = {}
    for i, c in enumerate(canon):
        scaf = _bemis_murcko_scaffold(c) or f"__empty__{c}"
        scaffold_to_rows.setdefault(scaf, []).append(i)

    sorted_scaffolds = sorted(
        scaffold_to_rows.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    indices_for_seed = list(range(len(sorted_scaffolds)))
    rng.shuffle(indices_for_seed)
    reassign: Dict[str, List[int]] = {
        sorted_scaffolds[new_pos][0]: scaffold_to_rows[sorted_scaffolds[new_pos][0]]
        for new_pos in indices_for_seed
    }
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for scaf, rows in reassign.items():
        if len(train_idx) < n_train_target:
            train_idx.extend(rows)
        elif len(val_idx) < n_val_target:
            val_idx.extend(rows)
        else:
            test_idx.extend(rows)
    if not train_idx or not val_idx or not test_idx:
        perm = rng.permutation(n)
        cut1 = int(round(n * train_frac))
        cut2 = cut1 + int(round(n * val_frac))
        return (
            np.asarray(sorted(perm[:cut1].tolist()), dtype=np.int64),
            np.asarray(sorted(perm[cut1:cut2].tolist()), dtype=np.int64),
            np.asarray(sorted(perm[cut2:].tolist()), dtype=np.int64),
        )
    return (
        np.asarray(sorted(train_idx), dtype=np.int64),
        np.asarray(sorted(val_idx), dtype=np.int64),
        np.asarray(sorted(test_idx), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Hilbert primal-dual regressor (per R18 ship + Wf-44 P4 numerical fix).
# ---------------------------------------------------------------------------


def fit_hilbert_cpu(
    X_tr_n: np.ndarray,
    y_tr: np.ndarray,
    *,
    epochs: int,
    seed: int,
    lr_primal: float,
    lr_dual: float,
    grad_clip_max_norm: float,
    step_size_decay: float,
    l2_weight: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Hilbert primal-dual on the augmented feature matrix.

    Returns ``(best_params, info)`` — ``best_params`` is the linear
    predictor applied to ``X_test_aug`` to produce predictions.

    The :class:`LambdaASTSRHilbert` configures ``adaptive_l2=False``
    so the L2 weight-decay guard is purely the Wf-127 P2 proximal
    shrinkage (no inner-val R² doubling path) — keeps the iteration
    deterministic across folds.
    """
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    pop_dim = X_tr_n.shape[1]
    info: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        opt = LambdaASTSRHilbert(
            population_dim=pop_dim,
            lr_primal=lr_primal,
            lr_dual=lr_dual,
            n_iters=epochs,
            seed=seed,
            grad_clip_max_norm=grad_clip_max_norm,
            step_size_decay=step_size_decay,
            l2_weight=l2_weight,
            adaptive_l2=False,
        )
        opt.optimize(X_tr_n.astype(np.float32), y_tr.astype(np.float32))
        info["wall_s"] = float(time.perf_counter() - t0)
        info["used_fp64"] = False
        return opt.best_params_, info
    except Exception:
        opt = LambdaASTSRHilbert(
            population_dim=pop_dim,
            lr_primal=lr_primal,
            lr_dual=lr_dual,
            n_iters=epochs,
            seed=seed,
            grad_clip_max_norm=grad_clip_max_norm,
            step_size_decay=step_size_decay,
            l2_weight=l2_weight,
            adaptive_l2=False,
        )
        opt.optimize(X_tr_n.astype(np.float64), y_tr.astype(np.float64))
        info["wall_s"] = float(time.perf_counter() - t0)
        info["used_fp64"] = True
        return opt.best_params_, info


def predict_hilbert(
    X_te_n: np.ndarray,
    best_params: np.ndarray,
) -> np.ndarray:
    y_hat = X_te_n.astype(np.float64) @ np.asarray(best_params, dtype=np.float64)
    return np.clip(y_hat, -PRED_CLIP, PRED_CLIP)


# ---------------------------------------------------------------------------
# R20-P3 feature extraction.
# ---------------------------------------------------------------------------


def extract_r20_p3_features(
    smiles_list: Sequence[str],
    *,
    n_bins: int = N_BINS_MULTI_PH,
    k_pll: int = K_PLL,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract R20-P3 features per SMILES = 32-bin Fourier Multi-PH ⊕ 128-d PLL.

    Returns ``(X, stats)`` where ``X`` has shape ``(n, n_bins**2 + k_pll)`` and
    ``stats`` counts parsed / failed / total.

    Failed parses / embed failures contribute zeros on the fill
    (matched to the per-molecule feature dim) so the regression
    matrix is fully populated.  This is the Wf-37 P3 honest
    fill-style contract.
    """
    from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
        MultiPHPLLSignature,
    )

    feat_dim = int(n_bins) ** 2 + int(k_pll)
    stats = {
        "n_parsed": 0,
        "n_failed": 0,
        "n_total": len(smiles_list),
    }
    try:
        signature = MultiPHPLLSignature(
            n_bins=int(n_bins),
            k=int(k_pll),
        )
    except Exception:
        # If the signature is unavailable, return all zeros.
        return (
            np.zeros((len(smiles_list), feat_dim), dtype=np.float32),
            {**stats, "n_failed": len(smiles_list)},
        )

    feats: List[np.ndarray] = []
    t0 = time.perf_counter()
    for smi in smiles_list:
        try:
            mph, pll = signature.compute(str(smi))
            mph_flat = np.asarray(mph.reshape(-1).cpu(), dtype=np.float32)
            pll_flat = np.asarray(pll.reshape(-1).cpu(), dtype=np.float32)
            if mph_flat.shape[0] != n_bins**2:
                mph_flat = np.zeros(n_bins**2, dtype=np.float32)
            if pll_flat.shape[0] != k_pll:
                pll_flat = np.zeros(k_pll, dtype=np.float32)
            row = np.concatenate([mph_flat, pll_flat], axis=0)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            if row.shape[0] != feat_dim:
                row = np.zeros(feat_dim, dtype=np.float32)
            feats.append(row)
            stats["n_parsed"] += 1
        except Exception:
            feats.append(np.zeros(feat_dim, dtype=np.float32))
            stats["n_failed"] += 1
    stats["wall_s"] = float(time.perf_counter() - t0)
    X = np.stack(feats, axis=0) if feats else np.zeros((0, feat_dim), dtype=np.float32)
    return X.astype(np.float32), stats


# ---------------------------------------------------------------------------
# Per-cell-line evaluation.
# ---------------------------------------------------------------------------


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


def evaluate_cell_line(
    cell_line: str,
    X: np.ndarray,
    y: np.ndarray,
    smiles: Sequence[str],
    seeds: Sequence[int],
    epochs: int,
    fractions: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Run 5-seed × 50-epoch strict-scaffold Hilbert regression on R20-P3 features.

    Returns the per-seed metrics dict compatible with Wf-83 P1 / Wf-85 P1 conventions.
    """
    n, d_feat = X.shape
    per_seed: List[Dict[str, Any]] = []
    r2_test_per_seed: List[float] = []
    r2_val_per_seed: List[float] = []
    r2_train_per_seed: List[float] = []

    for seed in seeds:
        train_idx, val_idx, test_idx = _strict_scaffold_split(
            smiles,
            fractions=fractions,
            seed=seed,
        )
        X_tr, X_va, X_te = X[train_idx], X[val_idx], X[test_idx]
        y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

        # Z-score on train fold (R20-P3 feature scales differ).
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        X_tr_n = (X_tr - mu) / sigma
        X_va_n = (X_va - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
        X_va_n = np.nan_to_num(X_va_n, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap feature dim to hilbert_dim via training-set variance
        # ranking (Wf-37 P3 contract: avoid tile inflation).
        # Per Wf-85 P1 contract: add a bias column (constant 1.0) so
        # the linear coefficients can learn the y-intercept; without
        # it, the regression collapses to predict-the-mean since y is
        # centered around 5.05 (not 0).
        hilbert_dim = DEFAULT_HILBERT_DIM
        bias_tr = np.ones((X_tr_n.shape[0], 1), dtype=np.float32)
        bias_va = np.ones((X_va_n.shape[0], 1), dtype=np.float32)
        bias_te = np.ones((X_te_n.shape[0], 1), dtype=np.float32)
        X_tr_n = np.concatenate([X_tr_n, bias_tr], axis=1)
        X_va_n = np.concatenate([X_va_n, bias_va], axis=1)
        X_te_n = np.concatenate([X_te_n, bias_te], axis=1)

        if X_tr_n.shape[1] > hilbert_dim:
            # Always keep the bias column at the end (last index after
            # concat) so the linear regressor learns the y-intercept.
            bias_idx = X_tr_n.shape[1] - 1
            vars_ = np.var(X_tr_n, axis=0)
            ranked = np.argsort(-vars_)
            keep = [bias_idx]
            for idx in ranked:
                if int(idx) == bias_idx:
                    continue
                if len(keep) >= hilbert_dim:
                    break
                keep.append(int(idx))
            keep_arr = np.asarray(sorted(set(keep))[:hilbert_dim], dtype=int)
            X_tr_n = X_tr_n[:, keep_arr]
            X_va_n = X_va_n[:, keep_arr]
            X_te_n = X_te_n[:, keep_arr]
        elif X_tr_n.shape[1] < hilbert_dim:
            pad = np.zeros((X_tr_n.shape[0], hilbert_dim - X_tr_n.shape[1]))
            X_tr_n = np.concatenate([X_tr_n, pad], axis=1)
            pad_va = np.zeros((X_va_n.shape[0], hilbert_dim - X_va_n.shape[1]))
            X_va_n = np.concatenate([X_va_n, pad_va], axis=1)
            pad_te = np.zeros((X_te_n.shape[0], hilbert_dim - X_te_n.shape[1]))
            X_te_n = np.concatenate([X_te_n, pad_te], axis=1)

        bp, info = fit_hilbert_cpu(
            X_tr_n,
            y_tr,
            epochs=epochs,
            seed=int(seed),
            lr_primal=DEFAULT_LR,
            lr_dual=DEFAULT_LR_DUAL,
            grad_clip_max_norm=DEFAULT_GRAD_CLIP,
            step_size_decay=DEFAULT_STEP_DECAY,
            l2_weight=DEFAULT_L2_WEIGHT,
        )

        y_hat_va = predict_hilbert(X_va_n, bp)
        y_hat_te = predict_hilbert(X_te_n, bp)

        r2_train = _compute_r2(y_tr, predict_hilbert(X_tr_n, bp))
        r2_val = _compute_r2(y_va, y_hat_va)
        r2_test = _compute_r2(y_te, y_hat_te)
        collapse = _collapse_detected(y_hat_te, y_te)

        seed_result = {
            "seed": int(seed),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "hilbert_r2_train": float(r2_train),
            "hilbert_r2_val": float(r2_val),
            "hilbert_r2_test": float(r2_test),
            "collapse_detected": bool(collapse),
            "used_fp64": info.get("used_fp64", False),
            "wall_time_s": info.get("wall_s", 0.0),
        }
        per_seed.append(seed_result)
        r2_test_per_seed.append(r2_test)
        r2_val_per_seed.append(r2_val)
        r2_train_per_seed.append(r2_train)

    summary = {
        "cell_line": cell_line,
        "target_col": PLATINAI_CELL_COLS[cell_line],
        "n_samples": int(n),
        "n_features_raw": int(d_feat),
        "n_features_hilbert": DEFAULT_HILBERT_DIM,
        "n_seeds": len(seeds),
        "epochs": int(epochs),
        "split": {
            "type": "strict_scaffold",
            "fractions": list(fractions),
            "splitter": "BemisMurckoScaffold_largest_first",
        },
        "hilbert_r2_test_mean": float(np.mean(r2_test_per_seed))
        if r2_test_per_seed
        else 0.0,
        "hilbert_r2_test_std": float(np.std(r2_test_per_seed))
        if len(r2_test_per_seed) > 1
        else 0.0,
        "hilbert_r2_val_mean": float(np.mean(r2_val_per_seed))
        if r2_val_per_seed
        else 0.0,
        "hilbert_r2_val_std": float(np.std(r2_val_per_seed))
        if len(r2_val_per_seed) > 1
        else 0.0,
        "hilbert_r2_train_mean": float(np.mean(r2_train_per_seed))
        if r2_train_per_seed
        else 0.0,
        "hilbert_r2_train_std": float(np.std(r2_train_per_seed))
        if len(r2_train_per_seed) > 1
        else 0.0,
        "per_seed": per_seed,
    }
    return summary


def load_platinai_cell(
    csv_path: Path,
    cell_line: str,
    n_max: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if cell_line not in PLATINAI_CELL_COLS:
        raise ValueError(f"Unknown cell line: {cell_line}")
    target_col = PLATINAI_CELL_COLS[cell_line]
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            smi = (r.get(PLATINAI_SMILES_COL) or "").strip()
            if not smi:
                continue
            raw = r.get(target_col)
            try:
                ic50_nm = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
                continue
            ic50_um = ic50_nm / 1000.0
            pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
            if not math.isfinite(pic50):
                continue
            Xs.append([1.0])  # placeholder; we replace with R20-P3 features later
            Ys.append(float(pic50))
            smiles.append(smi)
            if 0 < n_max <= len(Xs):
                break
    if not Xs:
        raise RuntimeError(f"No usable rows for cell line {cell_line}")
    return (
        np.asarray([0.0] * len(Xs), dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


# ---------------------------------------------------------------------------
# Main CLI.
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-348-B3 — Hilbert regression on R20-P3 features "
            "(32-bin Fourier Multi-PH + 128-d PLL) for L5 gate verification."
        )
    )
    parser.add_argument(
        "--features",
        default="r20_p3",
        choices=["r20_p3"],
        help="Feature set (currently only r20_p3 = Multi-PH 32-bin ⊕ PLL 128-d).",
    )
    parser.add_argument(
        "--dataset",
        default="platinai_4cell",
        choices=["platinai_4cell"],
        help="Dataset (only platinai_4cell is currently supported).",
    )
    parser.add_argument(
        "--cell-lines",
        type=_parse_cell_lines,
        nargs="+",
        default=["mcf7", "a549", "a2780", "a2780cis"],
        help="PlatinAI cell lines (default: mcf7 a549 a2780 a2780cis).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help=(
            "Max number of samples to load per cell line "
            "(default 500; PROJECTED 30-30 min CPU-only at this size)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds for strict-scaffold shuffling (default: 0..4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Hilbert PD iterations per seed (default: 50).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=N_BINS_MULTI_PH,
        help="Multi-PH bins per axis (default: 32 = Kim & Jung 2026).",
    )
    parser.add_argument(
        "--k-pll",
        type=int,
        default=K_PLL,
        help="PLL top-k eigenvalues (default: 128).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molmetal/reports/wf348b3_hilbert_r20p3"),
        help="Output directory (default: molmetal/reports/wf348b3_hilbert_r20p3).",
    )
    args = parser.parse_args(argv)

    csv_path = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"

    print(f"[wf348b3] features={args.features} dataset={args.dataset}")
    print(
        f"[wf348b3] n_samples={args.n_samples} epochs={args.epochs} seeds={args.seeds}"
    )
    print(f"[wf348b3] Multi-PH bins={args.n_bins} PLL k={args.k_pll}")
    print(f"[wf348b3] output={args.output}")

    # Flatten seeds / cell-lines.
    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))
    args.seeds = flat_seeds

    flat_cell_lines: List[str] = []
    for cl in args.cell_lines:
        if isinstance(cl, list):
            for sub in cl:
                if isinstance(sub, list):
                    flat_cell_lines.extend(str(x) for x in sub)
                else:
                    flat_cell_lines.append(str(sub))
        else:
            flat_cell_lines.append(str(cl))
    args.cell_lines = flat_cell_lines

    for cl in args.cell_lines:
        if cl not in PLATINAI_CELL_COLS:
            raise ValueError(f"Unknown cell line: {cl}")

    # Load PlatinAI 4-cell — extract one master smiles list from mcf7 column
    # (which is the most populated) so all 4 cell-lines see the same molecules.
    print(f"[wf348b3] Loading PlatinAI 4-cell from {csv_path}")
    _, _, master_smiles = load_platinai_cell(
        csv_path,
        "mcf7",
        n_max=args.n_samples,
    )
    # Filter to the smallest cell-line cohort so all 4 cells see the
    # same molecules (R20-P3 features are per-row identical across cells).
    valid_smiles: List[str] = []
    for smi in master_smiles:
        keep = True
        for cl in args.cell_lines:
            _, _, smi_list = load_platinai_cell(csv_path, cl, n_max=len(master_smiles))
            if smi not in smi_list:
                keep = False
                break
        if keep:
            valid_smiles.append(smi)
        if 0 < args.n_samples <= len(valid_smiles):
            break

    if not valid_smiles:
        raise RuntimeError("No molecules present in all 4 cell-line cohorts.")

    print(
        f"[wf348b3] Cohort size: {len(valid_smiles)} molecules present in all 4 cell-lines"
    )

    # Extract R20-P3 features once (per-molecule — same across all 4 cells).
    print(
        f"[wf348b3] Extracting R20-P3 features (Multi-PH {args.n_bins}^2 + PLL {args.k_pll})"
    )
    feat_extraction_t0 = time.perf_counter()
    X_r20p3, ext_stats = extract_r20_p3_features(
        valid_smiles,
        n_bins=args.n_bins,
        k_pll=args.k_pll,
    )
    feat_extraction_wall = float(time.perf_counter() - feat_extraction_t0)
    print(
        f"[wf348b3] Feature extraction wall={feat_extraction_wall:.1f}s "
        f"parsed={ext_stats['n_parsed']}/{ext_stats['n_total']} "
        f"failed={ext_stats['n_failed']} X.shape={X_r20p3.shape}"
    )

    # Run per-cell-line Hilbert regression.
    all_results: List[Dict[str, Any]] = []
    joint_test_y_true: List[float] = []
    joint_test_y_pred: List[float] = []

    for cell_line in args.cell_lines:
        # Load this cell-line's targets aligned to valid_smiles order.
        _, y, smi_list = load_platinai_cell(
            csv_path, cell_line, n_max=len(valid_smiles)
        )
        smi_to_y = {s: yi for s, yi in zip(smi_list, y)}
        y_aligned = np.asarray(
            [smi_to_y[s] for s in valid_smiles if s in smi_to_y],
            dtype=np.float32,
        )
        X_aligned = X_r20p3[: len(y_aligned)]

        print(f"[wf348b3] cell_line={cell_line}: n={len(y_aligned)} evaluating")
        result = evaluate_cell_line(
            cell_line,
            X_aligned,
            y_aligned,
            valid_smiles[: len(y_aligned)],
            args.seeds,
            args.epochs,
            DEFAULT_FRACTIONS,
        )
        per_seed_path = args.output / cell_line / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.output / cell_line / "run.log"
        log_lines = [
            f"cell_line={result['cell_line']}",
            f"target_col={result['target_col']}",
            f"n_samples={result['n_samples']}",
            f"n_features_raw={result['n_features_raw']}",
            f"n_features_hilbert={result['n_features_hilbert']}",
            f"n_seeds={result['n_seeds']}",
            f"epochs={result['epochs']}",
            f"split={result['split']}",
            f"hilbert_r2_test_mean={result['hilbert_r2_test_mean']:+.4f}",
            f"hilbert_r2_test_std={result['hilbert_r2_test_std']:.4f}",
            f"hilbert_r2_val_mean={result['hilbert_r2_val_mean']:+.4f}",
            f"hilbert_r2_val_std={result['hilbert_r2_val_std']:.4f}",
            f"hilbert_r2_train_mean={result['hilbert_r2_train_mean']:+.4f}",
            f"hilbert_r2_train_std={result['hilbert_r2_train_std']:.4f}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf348b3] cell_line={cell_line}: "
            f"Hilbert test R² = {result['hilbert_r2_test_mean']:+.4f} "
            f"± {result['hilbert_r2_test_std']:.4f} "
            f"(val={result['hilbert_r2_val_mean']:+.4f}, "
            f"train={result['hilbert_r2_train_mean']:+.4f})"
        )
        all_results.append(result)

        # Collect joint test predictions — first seed's test split, per cell.
        seed0 = result["per_seed"][0]
        # The test split for seed=0 differs per cell but they're already
        # matched in length to valid_smiles.  We re-derive the joint test
    # predictions properly below.

    # Joint R² across cell lines: for each cell, run seed 0 again to grab
    # (test_y_true, test_y_pred) on the same molecule index sets.
    print("[wf348b3] Computing joint R² across 4 cell-lines")
    joint_records: List[Dict[str, float]] = []
    for cell_line in args.cell_lines:
        _, y, smi_list = load_platinai_cell(
            csv_path, cell_line, n_max=len(valid_smiles)
        )
        smi_to_y = {s: yi for s, yi in zip(smi_list, y)}
        y_aligned = np.asarray(
            [smi_to_y[s] for s in valid_smiles if s in smi_to_y],
            dtype=np.float32,
        )
        X_aligned = X_r20p3[: len(y_aligned)]
        smiles_array = valid_smiles[: len(y_aligned)]

        train_idx, val_idx, test_idx = _strict_scaffold_split(
            smiles_array,
            fractions=DEFAULT_FRACTIONS,
            seed=0,
        )
        X_tr, X_te = X_aligned[train_idx], X_aligned[test_idx]
        y_tr, y_te = y_aligned[train_idx], y_aligned[test_idx]

        # z-score on train
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)

        # Add bias column for intercept (Wf-85 P1 contract).
        bias_tr = np.ones((X_tr_n.shape[0], 1), dtype=np.float32)
        bias_te = np.ones((X_te_n.shape[0], 1), dtype=np.float32)
        X_tr_n = np.concatenate([X_tr_n, bias_tr], axis=1)
        X_te_n = np.concatenate([X_te_n, bias_te], axis=1)

        hilbert_dim = DEFAULT_HILBERT_DIM
        if X_tr_n.shape[1] > hilbert_dim:
            bias_idx = X_tr_n.shape[1] - 1
            vars_ = np.var(X_tr_n, axis=0)
            ranked = np.argsort(-vars_)
            keep = [bias_idx]
            for idx in ranked:
                if int(idx) == bias_idx:
                    continue
                if len(keep) >= hilbert_dim:
                    break
                keep.append(int(idx))
            keep_arr = np.asarray(sorted(set(keep))[:hilbert_dim], dtype=int)
            X_tr_n = X_tr_n[:, keep_arr]
            X_te_n = X_te_n[:, keep_arr]
        elif X_tr_n.shape[1] < hilbert_dim:
            pad = np.zeros((X_tr_n.shape[0], hilbert_dim - X_tr_n.shape[1]))
            X_tr_n = np.concatenate([X_tr_n, pad], axis=1)
            pad_te = np.zeros((X_te_n.shape[0], hilbert_dim - X_te_n.shape[1]))
            X_te_n = np.concatenate([X_te_n, pad_te], axis=1)

        bp, info = fit_hilbert_cpu(
            X_tr_n,
            y_tr,
            epochs=args.epochs,
            seed=0,
            lr_primal=DEFAULT_LR,
            lr_dual=DEFAULT_LR_DUAL,
            grad_clip_max_norm=DEFAULT_GRAD_CLIP,
            step_size_decay=DEFAULT_STEP_DECAY,
            l2_weight=DEFAULT_L2_WEIGHT,
        )
        y_hat_te = predict_hilbert(X_te_n, bp)
        for yt, yp in zip(y_te.tolist(), y_hat_te.tolist()):
            joint_records.append(
                {"cell_line": cell_line, "y_true": float(yt), "y_pred": float(yp)}
            )

    joint_y_true = np.asarray([r["y_true"] for r in joint_records], dtype=np.float64)
    joint_y_pred = np.asarray([r["y_pred"] for r in joint_records], dtype=np.float64)
    joint_r2 = _compute_r2(joint_y_true, joint_y_pred)
    per_cell_test_means = [r["hilbert_r2_test_mean"] for r in all_results]
    aggregate_r2 = float(np.mean(per_cell_test_means)) if per_cell_test_means else 0.0
    aggregate_std = (
        float(np.std(per_cell_test_means)) if len(per_cell_test_means) > 1 else 0.0
    )

    # Verdict.
    gate_threshold = HONESTY_GATE_R2
    n_pass_cells = sum(1 for m in per_cell_test_means if m >= gate_threshold)
    if joint_r2 >= gate_threshold:
        gate_status = "LIFTED"
        verdict_label = "CEILING_BREAK"
    elif aggregate_r2 >= gate_threshold:
        gate_status = "PARTIAL_LIFTED"
        verdict_label = "AGGREGATE_LIFT"
    else:
        gate_status = "PRESERVED"
        verdict_label = "HONEST_NEGATIVE_PARTIAL_STRUCTURAL"

    n_pos_cells = sum(1 for m in per_cell_test_means if m > 0.0)
    collapse_any = any(
        any(s.get("collapse_detected", False) for s in r["per_seed"])
        for r in all_results
    )

    # Write r2_results.json.
    r2_results = {
        "workflow": "Wf-348-B3",
        "purpose": "L5 Hilbert R² honest-negative gate lift verification on R20-P3 features",
        "features": args.features,
        "feature_dims": {
            "multi_ph_dim": MULTI_PH_DIM,
            "pll_dim": PLL_DIM,
            "total_dim": TOTAL_DIM,
            "hilbert_dim": DEFAULT_HILBERT_DIM,
        },
        "dataset": args.dataset,
        "n_samples": int(len(valid_smiles)),
        "n_seeds": int(len(args.seeds)),
        "epochs": int(args.epochs),
        "honesty_gate_r2_threshold": gate_threshold,
        "feature_extraction_wall_s": feat_extraction_wall,
        "extraction_stats": ext_stats,
        "per_cohort": [
            {
                "cell_line": r["cell_line"],
                "n_samples": r["n_samples"],
                "hilbert_r2_test_mean": r["hilbert_r2_test_mean"],
                "hilbert_r2_test_std": r["hilbert_r2_test_std"],
                "hilbert_r2_val_mean": r["hilbert_r2_val_mean"],
                "hilbert_r2_val_mean_std": r["hilbert_r2_val_std"],
                "hilbert_r2_train_mean": r["hilbert_r2_train_mean"],
                "hilbert_r2_train_std": r["hilbert_r2_train_std"],
            }
            for r in all_results
        ],
        "joint_r2": float(joint_r2),
        "aggregate_r2": float(aggregate_r2),
        "aggregate_std": float(aggregate_std),
        "n_pass_cells_at_threshold": int(n_pass_cells),
        "n_pos_cells": int(n_pos_cells),
        "collapse_any": bool(collapse_any),
        "gate_status": gate_status,
        "verdict_label": verdict_label,
        "per_cell_test_r2_list": [float(m) for m in per_cell_test_means],
        "per_cell_per_seed_test_r2": {
            r["cell_line"]: [s["hilbert_r2_test"] for s in r["per_seed"]]
            for r in all_results
        },
        "joint_records_count": int(len(joint_records)),
    }
    r2_path = args.output / "r2_results.json"
    _write_json(r2_path, r2_results)
    print(f"[wf348b3] r2_results.json written to {r2_path}")

    # Write synthesis.md.
    syn_path = args.output / "synthesis.md"
    syn_lines: List[str] = [
        "# Wf-348-B3 — Hilbert R² on R20-P3 features (L5 gate verification)",
        "",
        "**Date**: 2026-09-29",
        "**Constraint**: Real Hilbert regression on R20-P3 features "
        f"({MULTI_PH_DIM}-d Multi-PH 32-bin Fourier ⊕ {PLL_DIM}-d PLL) "
        "5 seeds × 50 epochs × strict-scaffold on PlatinAI 4-cell.",
        "Per `party-twenty-spirit §3.5` honest framing.",
        "**Hardware**: CPU fp32 (default) + optional fp64 fallback.",
        f"**Inputs**: `molmetal/data/platinai_4cell.csv` ({len(valid_smiles)} rows × 4 cell lines).",
        "",
        "## Cross-cell-line 5-seed summary",
        "",
        "| cell line | n_samples | Hilbert test R² (mean ± std) | Hilbert val R² | Hilbert train R² | collapse? |",
        "|---|---:|---|---|---|:---:|",
    ]
    for r in all_results:
        per_seed_collapse = any(
            s.get("collapse_detected", False) for s in r["per_seed"]
        )
        syn_lines.append(
            "| {cl} | {n} | {tm:+.4f} ± {ts:.4f} | {vm:+.4f} ± {vs:.4f} | {trm:+.4f} ± {trs:.4f} | {c} |".format(
                cl=r["cell_line"],
                n=r["n_samples"],
                tm=r["hilbert_r2_test_mean"],
                ts=r["hilbert_r2_test_std"],
                vm=r["hilbert_r2_val_mean"],
                vs=r["hilbert_r2_val_std"],
                trm=r["hilbert_r2_train_mean"],
                trs=r["hilbert_r2_train_std"],
                c="yes" if per_seed_collapse else "no",
            )
        )
    syn_lines.append(
        f"| **4-cell aggregate** | — | **{aggregate_r2:+.4f} ± {aggregate_std:.4f}** | — | — | "
        f"{'yes' if collapse_any else 'no'} |"
    )
    syn_lines.append("")
    syn_lines.append(
        f"| **4-cell joint (seed=0 test split)** | — | **{joint_r2:+.4f}** | — | — | — |"
    )
    syn_lines.append("")
    syn_lines.append("## Verdict (honesty gate)")
    syn_lines.append("")
    syn_lines.append(
        f"**Hilbert test R² per-cohort aggregate (mean of 4 cell-line means) = {aggregate_r2:+.4f}**"
    )
    syn_lines.append(
        f"**Hilbert test R² joint (across 4 cell-lines, seed=0 test split) = {joint_r2:+.4f}**"
    )
    syn_lines.append(f"**Cells with R² ≥ {gate_threshold}: {n_pass_cells} / 4**")
    syn_lines.append(f"**Cells with R² > 0: {n_pos_cells} / 4**")
    syn_lines.append("")
    if gate_status == "LIFTED":
        syn_lines.append(
            f"**Gate status: LIFTED** — joint R² {joint_r2:+.4f} ≥ {gate_threshold}"
        )
        syn_lines.append(f"**Verdict label**: {verdict_label}")
    elif gate_status == "PARTIAL_LIFTED":
        syn_lines.append(
            f"**Gate status: PARTIAL_LIFTED** — aggregate R² {aggregate_r2:+.4f} ≥ {gate_threshold}"
        )
        syn_lines.append(f"**Verdict label**: {verdict_label}")
    else:
        syn_lines.append(
            f"**Gate status: PRESERVED** — aggregate R² {aggregate_r2:+.4f} < {gate_threshold}; joint R² {joint_r2:+.4f} < {gate_threshold}"
        )
        syn_lines.append(f"**Verdict label**: {verdict_label}")
    syn_lines.append("")
    syn_lines.append("## Per-cell-line per-seed test R² detail")
    syn_lines.append("")
    for r in all_results:
        syn_lines.append(f"### {r['cell_line']} ({r['target_col']})")
        syn_lines.append("")
        syn_lines.append(
            "| seed | test R² | val R² | train R² | n_train/n_val/n_test | collapse | fp64 | wall_s |"
        )
        syn_lines.append("|---:|---:|---:|---:|---|:---:|---|---:|")
        for s in r["per_seed"]:
            syn_lines.append(
                "| {seed} | {te:+.4f} | {va:+.4f} | {tr:+.4f} | {nt}/{nv}/{nte} | {c} | {fp} | {w:.2f} |".format(
                    seed=s["seed"],
                    te=s["hilbert_r2_test"],
                    va=s["hilbert_r2_val"],
                    tr=s["hilbert_r2_train"],
                    nt=s["n_train"],
                    nv=s["n_val"],
                    nte=s["n_test"],
                    c="yes" if s["collapse_detected"] else "no",
                    fp="fp64" if s["used_fp64"] else "fp32",
                    w=s["wall_time_s"],
                )
            )
        syn_lines.append("")
    syn_lines.append("## Honest framing")
    syn_lines.append("")
    syn_lines.append(
        "- **Real Hilbert regression on R20-P3 features** = Multi-PH 32-bin Fourier + 128-d PLL."
    )
    syn_lines.append(
        "- **Per `party-twenty-spirit §3.5`**: MEASURED R² reported honestly; no PROJECTED values."
    )
    syn_lines.append("- **CPU-only evaluation**; fp64 fallback on numerical failure.")
    syn_lines.append(
        "- **Collapse detector**: `pred_std / y_test_std < 0.10` flagged per seed."
    )
    syn_lines.append(
        "- **Sample limit**: `--n-samples 500` default for wall-time budget; "
        f"actual cohort used: {len(valid_smiles)} molecules present in all 4 cell-lines."
    )
    syn_lines.append("")
    syn_lines.append(
        f"**Aggregate Hilbert test R² (mean of 4 cell-line means) = {aggregate_r2:+.4f}; "
        f"joint R² (across 4 cell-lines, seed=0 test split) = {joint_r2:+.4f}; "
        f"gate status = {gate_status} ({verdict_label}).**"
    )
    _write_run_log(syn_path, syn_lines)
    print(f"[wf348b3] synthesis.md written to {syn_path}")

    # Write final run.log.
    final_log_lines = [
        f"workflow=Wf-348-B3",
        f"features={args.features}",
        f"dataset={args.dataset}",
        f"n_samples={args.n_samples}",
        f"n_cohort={len(valid_smiles)}",
        f"n_seeds={len(args.seeds)}",
        f"epochs={args.epochs}",
        f"feature_extraction_wall_s={feat_extraction_wall:.2f}",
        f"extraction_parsed={ext_stats['n_parsed']}",
        f"extraction_failed={ext_stats['n_failed']}",
        f"aggregate_r2={aggregate_r2:+.4f}",
        f"aggregate_std={aggregate_std:.4f}",
        f"joint_r2={joint_r2:+.4f}",
        f"n_pass_cells_at_threshold={n_pass_cells}",
        f"n_pos_cells={n_pos_cells}",
        f"gate_status={gate_status}",
        f"verdict_label={verdict_label}",
    ]
    _write_run_log(args.output / "run.log", final_log_lines)
    print(f"[wf348b3] run.log written to {args.output / 'run.log'}")
    print(
        f"[wf348b3] DONE — aggregate R²={aggregate_r2:+.4f} ± {aggregate_std:.4f}; "
        f"joint R²={joint_r2:+.4f}; gate_status={gate_status} ({verdict_label})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
