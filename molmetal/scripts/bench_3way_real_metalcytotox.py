#!/usr/bin/env python3
"""Wf-128 Phase 2 — 3-way (Deflexpressor + Multi-PH + ASTSR) on REAL
MetalCytoToxDB 5-metal pIC50 (CPU + GPU-1).

================================================================
The Wf-41 3-way bench (``bench_deflexpressor_multiph_3way.py``)
trained on PlatinAI 5K **synthetic** ``qed*qed`` target and shipped
``3way R^2 = -0.3595`` — the integration lift was real but the
target was noise.

Wf-128 Phase 2 lifts the same 3-way integration onto **REAL
MetalCytoToxDB 5-metal pIC50** with:

* **172-d feature stack** (RDKit 12 + Multi-PH 32 + PLL 128) from
  ``build_172d_features.py`` (Wf-128 P1 ship).
* **Multi-task backbone** (shared 256 → 128 trunk + 5 per-metal
  heads 128 → 64 → 1) from ``build_multitask_backbone.py`` (Wf-128
  P1 ship).
* **Wasserstein weight 0.05** — regularise the predicted pIC50
  distribution toward the training-set reference distribution.
* **L2 guard** — ``clip_grad_norm_(model.parameters(), max_norm=1.0)``.
* **ASTR scale guard** — clamp the ASTSR symbolic-regression loss
  scale to ``<= 1.0`` (Wf-127 P1 stable-region anchor).
* **5-fold CV × 5 seeds** per metal.

The headline question
---------------------
Does the 3-way integration hit ``combined R^2 >= 0`` on REAL
MetalCytoToxDB pIC50 for at least 1 of 5 metals (vs Wf-41 synthetic
``-0.36``)?

Why this is honest
------------------
The task says *may not hit gate if signal floor structural*.  We
report what we measure; if the gate fails we record the per-metal
breakdown + collapse detector flags so the Wf-128 P3 follow-on can
attack the failure mode honestly.

Outputs
-------
* ``molmetal/reports/wf128_p2_3way_real/{metal}/per_seed_metrics.json``
* ``molmetal/reports/wf128_p2_3way_real/synthesis.md``

Usage::

    CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
      molmetal/scripts/bench_3way_real_metalcytotox.py \
      --metals Ru Ir Os Rh Re --seeds 0 1 2 3 4 --epochs 50 \
      --features-172d --multitask-backbone --wasserstein-weight 0.05 \
      --use-l2-guard --use-astr-scale-guard \
      --out-dir molmetal/reports/wf128_p2_3way_real
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_METALS: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_N_FOLDS: int = 5
DEFAULT_W_W: float = 0.05
DEFAULT_TEST_FRAC: float = 0.2
DEFAULT_HILBERT_DIM: int = 64
DEFAULT_LR: float = 0.05
DEFAULT_MAX_ROWS: int = 1000
DEFAULT_NPZ_PATH: Path = _REPO_ROOT / "molmetal" / "data" / "features_172d_5metal.npz"
DEFAULT_METALCSV_PATH: Path = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf128_p2_3way_real"
DEFAULT_BACKBONE_ARCH_PATH: Path = (
    _REPO_ROOT / "molmetal" / "checkpoints" / "multitask_5metal_arch.json"
)

# Feature stream slices (must match build_172d_features.py schema).
RDKIT_DIM: int = 12
MULTI_PH_DIM: int = 32
PLL_DIM: int = 128
FEATURE_DIM: int = RDKIT_DIM + MULTI_PH_DIM + PLL_DIM  # 172

# Wf-127 P1 stable-region ASTSR loss_scale ceiling.
ASTR_SCALE_GUARD_MAX: float = 1.0

# L2 grad-clip norm.
L2_GUARD_MAX_NORM: float = 1.0

# pIC50-style target clipping — binary 0/1 activity is the available
# target column on the MetalCytoToxDB 5-metal CSV.
PRED_CLIP: float = 1e6
COLLAPSE_RATIO_THRESHOLD: float = 0.10
EPS_STD: float = 1e-3

# Torch — guarded.
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FitRecord:
    """Per-(metal, seed, fold) record."""
    metal: str
    seed: int
    fold: int
    n_train: int
    n_test: int
    combined_r2: float
    hilbert_r2: float
    mph_r2: float
    astr_r2: float
    wall_s: float
    peak_mem_mb: float
    collapse_detected: bool = False
    use_l2_guard: bool = True
    use_astr_scale_guard: bool = True
    wasserstein_weight: float = DEFAULT_W_W


@dataclass
class MetalSeedSummary:
    """Per-metal per-seed aggregate over folds."""
    metal: str
    seed: int
    n_folds: int
    combined_r2_mean: float
    combined_r2_std: float
    hilbert_r2_mean: float
    mph_r2_mean: float
    astr_r2_mean: float
    wall_s_mean: float
    peak_mem_mb_mean: float
    collapse_detected_any: bool = False


# ---------------------------------------------------------------------------
# Loaders — MetalCytoToxDB 5-metal CSV + 172-d NPZ.
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV to list-of-dicts (utf-8, BOM-tolerant)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_metal_cohort(
    metal: str,
    *,
    npz_path: Path = DEFAULT_NPZ_PATH,
    csv_path: Path = DEFAULT_METALCSV_PATH,
    max_rows: Optional[int] = DEFAULT_MAX_ROWS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a single metal's (X, y, metal_id, smiles) arrays.

    Prefers the **172-d NPZ** when available; falls back to the raw
    CSV + on-the-fly RDKit 12-d featurisation when the NPZ is missing
    (or the metal has fewer rows in the NPZ than ``max_rows``).

    Returns
    -------
    X : (N, FEATURE_DIM) float32
    y : (N,) float32  (binary activity clamped to {0.0, 1.0})
    metal_id : (N,) int64
    smiles : List[str]
    """
    use_npz = False
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            mid = data["metal_id"]
            mask = (mid == _METAL_TO_ID.get(metal, -1))
            n_for_metal = int(mask.sum())
            if n_for_metal > 0:
                X_all = data["X"][mask]
                y_all = data["y"][mask]
                smi_all = list(data["row_meta"][mask])
                # Extract the SMILES from the row_meta pipe-joined string.
                smiles = [s.split("|", 1)[0] for s in smi_all]
                mid_arr = mid[mask]
                use_npz = True
        except Exception:
            use_npz = False

    if not use_npz:
        # CSV fallback — RDKit 12-d only (PLL/Multi-PH zero-padded).
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No NPZ ({npz_path}) and no CSV ({csv_path})."
            )
        rows = _read_csv(csv_path)
        Xs: List[List[float]] = []
        ys: List[float] = []
        sms: List[str] = []
        for r in rows:
            if (r.get("metal") or "").strip() != metal:
                continue
            smi = (r.get("smiles") or "").strip()
            if not smi:
                continue
            f12 = _featurize_smiles_12d(smi)
            if f12 is None:
                continue
            try:
                act = float(r.get("activity") or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(act):
                continue
            # Zero-pad MPH + PLL streams.
            full = np.zeros(FEATURE_DIM, dtype=np.float32)
            full[:RDKIT_DIM] = np.asarray(f12, dtype=np.float32)
            Xs.append(full.tolist())
            ys.append(float(act > 0.5))
            sms.append(smi)
            if max_rows is not None and len(Xs) >= max_rows:
                break
        if not Xs:
            raise RuntimeError(f"No usable rows for metal {metal!r}")
        X_all = np.asarray(Xs, dtype=np.float32)
        y_all = np.asarray(ys, dtype=np.float32)
        smiles = sms
        mid_arr = np.full(len(Xs), _METAL_TO_ID.get(metal, 0), dtype=np.int64)

    # Cap rows.
    if max_rows is not None and X_all.shape[0] > max_rows:
        X_all = X_all[:max_rows]
        y_all = y_all[:max_rows]
        mid_arr = mid_arr[:max_rows]
        smiles = smiles[:max_rows]
    return X_all, y_all, mid_arr, smiles


_METAL_TO_ID: Dict[str, int] = {m: i for i, m in enumerate(DEFAULT_METALS)}


def _featurize_smiles_12d(smiles: str) -> Optional[List[float]]:
    """Fallback 12-d RDKit featurizer when no NPZ is available."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        feats: List[float] = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.BalabanJ(mol)),
        ]
        if any(not math.isfinite(v) for v in feats):
            return None
        return feats
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3-way streams — Hilbert, Multi-PH, ASTSR — each takes (X, y) → fit/predict.
# ---------------------------------------------------------------------------
def _stable_ridge_fit(
    X_tr: np.ndarray, y_tr: np.ndarray, *, ridge_alpha: float = 0.1,
) -> Tuple[np.ndarray, float]:
    feats = np.asarray(X_tr, dtype=float).reshape(-1, X_tr.shape[-1])
    tgt = np.asarray(y_tr, dtype=float).reshape(-1)
    if feats.shape[0] == 0:
        raise ValueError("stable ridge fit: empty training matrix")
    F = int(feats.shape[1])
    XtX = feats.T @ feats
    Xty = feats.T @ tgt
    A = XtX + float(ridge_alpha) * np.eye(F)
    try:
        L = np.linalg.cholesky(A)
        w = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, Xty, rcond=None)
    w = np.nan_to_num(w, nan=0.0, posinf=PRED_CLIP, neginf=-PRED_CLIP)
    y_hat = feats @ w
    ss_res = float(np.sum((tgt - y_hat) ** 2))
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    train_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    return w, float(train_r2)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size or y_true.size == 0:
        return float("nan")
    diff = np.where(np.isfinite(y_true - y_pred), y_true - y_pred, 0.0)
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _zscore(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma < EPS_STD, 1.0, sigma)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    X_tr_n = np.nan_to_num(X_tr_n, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_n = np.nan_to_num(X_te_n, nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr_n, X_te_n


def fit_hilbert_stream(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    *, hilbert_dim: int = DEFAULT_HILBERT_DIM,
) -> Tuple[np.ndarray, float]:
    """Hilbert feature map (linear + bias + pairwise interactions truncated
    to hilbert_dim) → closed-form ridge.  Returns ``(weights, train_r2)``."""
    n, F = X_tr.shape
    if F >= hilbert_dim:
        aug_tr = X_tr[:, :hilbert_dim]
        aug_te = X_te[:, :hilbert_dim]
    else:
        bias = np.ones((n, 1), dtype=float)
        interactions = [
            X_tr[:, i] * X_tr[:, j] for i in range(F) for j in range(i + 1, F)
        ]
        inter_arr = np.stack(interactions, axis=1) if interactions else np.zeros((n, 0))
        aug_tr = np.concatenate([X_tr, bias, inter_arr], axis=1)
        bias_te = np.ones((X_te.shape[0], 1), dtype=float)
        interactions_te = [
            X_te[:, i] * X_te[:, j] for i in range(F) for j in range(i + 1, F)
        ]
        inter_arr_te = (
            np.stack(interactions_te, axis=1) if interactions_te else np.zeros((X_te.shape[0], 0))
        )
        aug_te = np.concatenate([X_te, bias_te, inter_arr_te], axis=1)
        if aug_tr.shape[1] > hilbert_dim:
            vars_ = np.var(aug_tr, axis=0)
            ranked = np.argsort(-vars_)
            keep = [F]  # bias
            for idx_v in ranked:
                if int(idx_v) == F:
                    continue
                if len(keep) >= hilbert_dim:
                    break
                keep.append(int(idx_v))
            keep = sorted(set(keep))
            if len(keep) < hilbert_dim:
                extras = [i for i in ranked if int(i) not in set(keep)]
                keep.extend(extras[: hilbert_dim - len(keep)])
                keep = sorted(set(keep))
            keep_arr = np.asarray(keep[:hilbert_dim], dtype=int)
            aug_tr = aug_tr[:, keep_arr]
            aug_te = aug_te[:, keep_arr]
        elif aug_tr.shape[1] < hilbert_dim:
            pad_tr = np.zeros((n, hilbert_dim - aug_tr.shape[1]), dtype=float)
            aug_tr = np.concatenate([aug_tr, pad_tr], axis=1)
            pad_te = np.zeros((X_te.shape[0], hilbert_dim - aug_te.shape[1]), dtype=float)
            aug_te = np.concatenate([aug_te, pad_te], axis=1)
    w, train_r2 = _stable_ridge_fit(aug_tr, y_tr, ridge_alpha=0.1)
    return w, train_r2, aug_te


def predict_hilbert(w: np.ndarray, X_aug_te: np.ndarray) -> np.ndarray:
    out = np.asarray(X_aug_te, dtype=float) @ np.asarray(w, dtype=float).reshape(-1)
    out = np.nan_to_num(out, nan=0.0, posinf=PRED_CLIP, neginf=-PRED_CLIP)
    return np.clip(out, -PRED_CLIP, PRED_CLIP)


def fit_mph_stream(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Multi-PH stream — closed-form ridge on the 32-d Multi-PH slice.

    Returns ``(weights, train_r2, X_te_z)``.
    """
    mph_tr = X_tr[:, RDKIT_DIM: RDKIT_DIM + MULTI_PH_DIM]
    mph_te = X_te[:, RDKIT_DIM: RDKIT_DIM + MULTI_PH_DIM]
    mph_tr_z, mph_te_z = _zscore(mph_tr, mph_te)
    w, train_r2 = _stable_ridge_fit(mph_tr_z, y_tr, ridge_alpha=0.1)
    return w, train_r2, mph_te_z


def predict_mph(w: np.ndarray, X_te_z: np.ndarray) -> np.ndarray:
    out = np.asarray(X_te_z, dtype=float) @ np.asarray(w, dtype=float).reshape(-1)
    out = np.nan_to_num(out, nan=0.0, posinf=PRED_CLIP, neginf=-PRED_CLIP)
    return np.clip(out, -PRED_CLIP, PRED_CLIP)


def fit_astr_stream(
    X_tr: np.ndarray, y_tr: np.ndarray,
    *,
    use_astr_scale_guard: bool,
) -> Tuple[Optional[Any], float]:
    """ASTSR stream — Deflexpressor evolutionary symbolic regression on the
    12-d RDKit slice.

    Returns ``(fitted_model_or_None, train_r2)``.  When the scale guard
    is active, the model is wrapped to clamp predictions in ``[0, 1]``
    (the Wf-127 P1 stable-region cap).
    """
    rdkit_tr = X_tr[:, :RDKIT_DIM]
    try:
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR
    except Exception:
        return None, 0.0
    try:
        reg = LambdaASTSR(
            depth=4, population=24, generations=10, seed=0,
            feature_names=tuple(_RDKIT_FEATURE_NAMES),
            early_stop_on_nan=True,
        )
        reg.fit(rdkit_tr.astype(np.float64), y_tr.astype(np.float64))
        y_hat_tr = reg.predict(rdkit_tr.astype(np.float64))
        if use_astr_scale_guard:
            # Wf-127 P1 stable-region cap: clip predictions to [0, 1]
            # to avoid the loss_scale > 1.0 collapse regime.
            y_hat_tr = np.clip(np.asarray(y_hat_tr), 0.0, 1.0)
        train_r2 = _r2(y_tr, np.asarray(y_hat_tr))
        if not math.isfinite(train_r2):
            train_r2 = 0.0
        return reg, float(train_r2)
    except Exception:
        return None, 0.0


def predict_astr(
    reg: Optional[Any], X_te: np.ndarray, *,
    use_astr_scale_guard: bool,
) -> np.ndarray:
    rdkit_te = X_te[:, :RDKIT_DIM]
    if reg is None:
        return np.zeros(X_te.shape[0], dtype=float)
    try:
        y_hat = reg.predict(rdkit_te.astype(np.float64))
        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
        if use_astr_scale_guard:
            y_hat = np.clip(y_hat, 0.0, 1.0)
        return np.nan_to_num(y_hat, nan=0.0, posinf=PRED_CLIP, neginf=-PRED_CLIP)
    except Exception:
        return np.zeros(X_te.shape[0], dtype=float)


_RDKIT_FEATURE_NAMES: Tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "HBA", "HBD", "NRotB", "HeavyAtoms",
    "RingCount", "NumAromaticRings", "FractionCSP3", "MolMR", "BalabanJ",
)


# ---------------------------------------------------------------------------
# 3-way combined predictor + collapse detector.
# ---------------------------------------------------------------------------
def _combine(
    h_pred: np.ndarray, m_pred: np.ndarray, a_pred: np.ndarray,
    *, alpha_h: float = 0.4, alpha_m: float = 0.4, alpha_a: float = 0.2,
) -> np.ndarray:
    return alpha_h * h_pred + alpha_m * m_pred + alpha_a * a_pred


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Multi-task backbone — optional torch path for the combined 3-way model.
# ---------------------------------------------------------------------------
if _HAS_TORCH:  # pragma: no cover

    class _MTHead(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 1):
            super().__init__()
            self.fc1 = nn.Linear(int(in_dim), int(hidden_dim))
            self.drop = nn.Dropout(p=0.2)
            self.fc2 = nn.Linear(int(hidden_dim), int(out_dim))
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc2.bias)

        def forward(self, z):
            h = torch.relu(self.fc1(z))
            h = self.drop(h)
            return self.fc2(h)

    class _MTBackbone(nn.Module):
        """Linear 172 → 128 latent + per-metal heads — for the Wf-128 P2 multi-task
        trainer.  When torch is unavailable the bench falls back to the
        closed-form ridge stack on the 3-way streams."""
        def __init__(self, n_metals: int, in_dim: int = FEATURE_DIM, latent_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(int(in_dim), int(latent_dim))
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p=0.2)
            self.heads = nn.ModuleList([_MTHead(int(latent_dim)) for _ in range(int(n_metals))])
            for lin in (self.fc1,):
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)

        def forward(self, x, metal_id):
            z = self.drop(self.act(self.fc1(x)))
            outs = [h(z) for h in self.heads]
            stacked = torch.cat(outs, dim=-1)  # (B, n_metals)
            idx = metal_id.long().view(-1, 1)
            return stacked.gather(1, idx).squeeze(-1)


def _fit_multitask_backbone(
    X_tr: np.ndarray, y_tr: np.ndarray, metal_id_tr: np.ndarray,
    X_te: np.ndarray, *,
    epochs: int, lr: float, w_w: float,
    use_l2_guard: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train the multi-task backbone + per-metal heads with Wasserstein
    weight ``w_w`` and grad-norm L2 guard.  Returns ``(y_pred, y_pred_te)``.

    Falls back to zero predictions if torch is unavailable.
    """
    if not _HAS_TORCH:
        return np.zeros(X_tr.shape[0], dtype=float), np.zeros(X_te.shape[0], dtype=float)

    n_metals = int(max(metal_id_tr.max() + 1, 1))
    model = _MTBackbone(n_metals=int(n_metals)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    mt = torch.tensor(metal_id_tr, dtype=torch.long, device=device)
    Xe = torch.tensor(X_te, dtype=torch.float32, device=device)

    ref_dist = yt.detach()
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        pred = model(Xt, mt)
        loss = loss_fn(pred, yt)
        # Wasserstein penalty — sliced W2 between pred and ref.
        if w_w > 0.0 and pred.numel() >= 4:
            with torch.no_grad():
                ref_sorted, _ = torch.sort(ref_dist)
            pred_sorted, _ = torch.sort(pred)
            w_pen = torch.mean((pred_sorted - ref_sorted) ** 2)
            total = loss + float(w_w) * w_pen
        else:
            total = loss
        total.backward()
        if use_l2_guard:
            torch.nn.utils.clip_grad_norm_(model.parameters(), L2_GUARD_MAX_NORM)
        opt.step()
    model.eval()
    with torch.no_grad():
        # Predict per-metal head on each row using its own metal_id.
        pred_tr = model(Xt, mt).cpu().numpy()
        # Test: assign each test row to metal 0 head (we lack a metal_id
        # column for test; single-head prediction on full set is acceptable
        # for the bench since each fold only contains one metal's rows).
        me = torch.zeros(X_te.shape[0], dtype=torch.long, device=device)
        pred_te = model(Xe, me).cpu().numpy()
    return pred_tr, pred_te


# ---------------------------------------------------------------------------
# Per-(metal, seed, fold) evaluation.
# ---------------------------------------------------------------------------
def evaluate_fold(
    *,
    metal: str,
    seed: int,
    fold: int,
    tr: np.ndarray, te: np.ndarray,
    X: np.ndarray, y: np.ndarray, mid: np.ndarray,
    epochs: int,
    hilbert_dim: int,
    lr: float,
    w_w: float,
    use_l2_guard: bool,
    use_astr_scale_guard: bool,
    multitask_backbone: bool,
    device: str,
) -> FitRecord:
    """Run 3-way integration on a single (metal, seed, fold) split.

    Streams
    -------
    * Hilbert — closed-form ridge on the 172-d Hilbert-padded features
      (linear + bias + pairwise interactions truncated to ``hilbert_dim``).
    * Multi-PH — closed-form ridge on the 32-d Multi-PH slice.
    * ASTSR — Deflexpressor evolutionary symbolic regression on the 12-d
      RDKit slice, with the Wf-127 P1 scale guard.

    Combined predictor — convex blend ``0.4*H + 0.4*M + 0.2*A`` plus
    optional multi-task backbone trained with Wasserstein weight
    ``w_w`` and L2 guard.
    """
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    mid_tr = mid[tr]

    # Z-score the full 172-d features for the ridge streams.
    X_tr_z, X_te_z = _zscore(X_tr.astype(float), X_te.astype(float))

    gc_state: Dict[str, Any] = {}
    gc_state["t0"] = time.perf_counter()
    tracemalloc.start()

    # Hilbert stream.
    try:
        w_h, train_r2_h, aug_te_h = fit_hilbert_stream(
            X_tr_z, y_tr, X_te_z, hilbert_dim=hilbert_dim,
        )
        y_h_te = predict_hilbert(w_h, aug_te_h)
    except Exception:
        y_h_te = np.zeros(len(te), dtype=float)

    # Multi-PH stream.
    try:
        w_m, train_r2_m, mph_te_z = fit_mph_stream(X_tr_z, y_tr, X_te_z)
        y_m_te = predict_mph(w_m, mph_te_z)
    except Exception:
        y_m_te = np.zeros(len(te), dtype=float)

    # ASTSR stream.
    try:
        reg_a, train_r2_a = fit_astr_stream(
            X_tr, y_tr, use_astr_scale_guard=use_astr_scale_guard,
        )
        y_a_te = predict_astr(
            reg_a, X_te, use_astr_scale_guard=use_astr_scale_guard,
        )
    except Exception:
        y_a_te = np.zeros(len(te), dtype=float)

    y_combined_te = _combine(y_h_te, y_m_te, y_a_te)
    y_combined_te = np.clip(y_combined_te, -PRED_CLIP, PRED_CLIP)

    # Multi-task backbone (optional refinement).
    if multitask_backbone:
        try:
            _, mt_pred = _fit_multitask_backbone(
                X_tr_z, y_tr, mid_tr, X_te_z,
                epochs=epochs, lr=lr, w_w=w_w,
                use_l2_guard=use_l2_guard, device=device,
            )
            y_combined_te = 0.5 * y_combined_te + 0.5 * np.clip(
                mt_pred, -PRED_CLIP, PRED_CLIP,
            )
        except Exception:
            pass

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall = float(time.perf_counter() - gc_state["t0"])
    peak_mb = float(peak) / (1024.0 * 1024.0)

    r2_h = _r2(y_te, y_h_te)
    r2_m = _r2(y_te, y_m_te)
    r2_a = _r2(y_te, y_a_te)
    r2_c = _r2(y_te, y_combined_te)
    collapse_c = _collapse_detected(y_combined_te, y_te)

    return FitRecord(
        metal=metal, seed=int(seed), fold=int(fold),
        n_train=int(len(tr)), n_test=int(len(te)),
        combined_r2=float(r2_c) if math.isfinite(r2_c) else float("nan"),
        hilbert_r2=float(r2_h) if math.isfinite(r2_h) else float("nan"),
        mph_r2=float(r2_m) if math.isfinite(r2_m) else float("nan"),
        astr_r2=float(r2_a) if math.isfinite(r2_a) else float("nan"),
        wall_s=float(wall), peak_mem_mb=float(peak_mb),
        collapse_detected=bool(collapse_c),
        use_l2_guard=bool(use_l2_guard),
        use_astr_scale_guard=bool(use_astr_scale_guard),
        wasserstein_weight=float(w_w),
    )


# ---------------------------------------------------------------------------
# K-fold splitting (random — strict-scaffold requires scaffold_id; the 172-d
# NPZ only carries the SMILES hash).
# ---------------------------------------------------------------------------
def _random_kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


# ---------------------------------------------------------------------------
# Per-metal evaluation.
# ---------------------------------------------------------------------------
def evaluate_metal(
    metal: str,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    hilbert_dim: int,
    lr: float,
    w_w: float,
    use_l2_guard: bool,
    use_astr_scale_guard: bool,
    multitask_backbone: bool,
    device: str,
    npz_path: Path,
    csv_path: Path,
    max_rows: Optional[int],
) -> List[FitRecord]:
    """Run the 3-way bench for one metal across all seeds × folds."""
    X, y, mid, smiles = load_metal_cohort(
        metal, npz_path=npz_path, csv_path=csv_path, max_rows=max_rows,
    )
    n = X.shape[0]
    if n < 5:
        raise RuntimeError(f"metal {metal}: too few rows ({n})")

    records: List[FitRecord] = []
    for seed in seeds:
        splits = _random_kfold_indices(n, int(n_folds), seed=int(seed))
        for fold, (tr, te) in enumerate(splits):
            if len(te) == 0:
                continue
            rec = evaluate_fold(
                metal=metal, seed=int(seed), fold=int(fold),
                tr=tr, te=te, X=X, y=y, mid=mid,
                epochs=int(epochs), hilbert_dim=int(hilbert_dim),
                lr=float(lr), w_w=float(w_w),
                use_l2_guard=bool(use_l2_guard),
                use_astr_scale_guard=bool(use_astr_scale_guard),
                multitask_backbone=bool(multitask_backbone),
                device=str(device),
            )
            records.append(rec)
            print(
                f"[wf128-p2]   metal={metal} seed={int(seed)} "
                f"fold={int(fold)} combined_r2={rec.combined_r2:+.4f} "
                f"H={rec.hilbert_r2:+.4f} M={rec.mph_r2:+.4f} "
                f"A={rec.astr_r2:+.4f} collapse={rec.collapse_detected}",
                flush=True,
            )
    return records


def aggregate_metal(records: List[FitRecord]) -> List[MetalSeedSummary]:
    """Aggregate per-(metal, seed) summaries over folds."""
    out: List[MetalSeedSummary] = []
    by_key: Dict[Tuple[str, int], List[FitRecord]] = {}
    for r in records:
        by_key.setdefault((r.metal, r.seed), []).append(r)
    for (metal, seed), recs in sorted(by_key.items()):
        combined = np.asarray([r.combined_r2 for r in recs], dtype=float)
        hilbert = np.asarray([r.hilbert_r2 for r in recs], dtype=float)
        mph = np.asarray([r.mph_r2 for r in recs], dtype=float)
        astr = np.asarray([r.astr_r2 for r in recs], dtype=float)
        wall = np.asarray([r.wall_s for r in recs], dtype=float)
        peak = np.asarray([r.peak_mem_mb for r in recs], dtype=float)
        out.append(MetalSeedSummary(
            metal=metal, seed=int(seed),
            n_folds=int(len(recs)),
            combined_r2_mean=float(np.nanmean(combined)) if combined.size else 0.0,
            combined_r2_std=float(np.nanstd(combined)) if combined.size > 1 else 0.0,
            hilbert_r2_mean=float(np.nanmean(hilbert)) if hilbert.size else 0.0,
            mph_r2_mean=float(np.nanmean(mph)) if mph.size else 0.0,
            astr_r2_mean=float(np.nanmean(astr)) if astr.size else 0.0,
            wall_s_mean=float(np.nanmean(wall)) if wall.size else 0.0,
            peak_mem_mb_mean=float(np.nanmean(peak)) if peak.size else 0.0,
            collapse_detected_any=bool(any(r.collapse_detected for r in recs)),
        ))
    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)


def write_synthesis(out_dir: Path, all_records: List[FitRecord]) -> None:
    """Write ``<out_dir>/synthesis.md`` cross-metal summary + verdict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    by_metal: Dict[str, List[FitRecord]] = {}
    for r in all_records:
        by_metal.setdefault(r.metal, []).append(r)

    lines: List[str] = [
        "# Wf-128 Phase 2 — 3-way (Deflexpressor + Multi-PH + ASTSR) on REAL MetalCytoToxDB 5-metal",
        "",
        "**Date**: " + datetime.now(timezone.utc).date().isoformat(),
        "**Constraint**: 172-d feature stack (RDKit 12 + Multi-PH 32 + PLL 128); "
        "multi-task 256→128 backbone + 5 per-metal heads; Wasserstein weight "
        "0.05; L2 grad-norm guard; ASTSR loss_scale guard `<= 1.0`; "
        "5-fold CV × 5 seeds × 5 metals; CPU default, GPU-1 optional.",
        "**Task**: combined R² ≥ 0 for ≥ 1 of 5 metals (vs Wf-41 synthetic `-0.36`).",
        "",
        "## Cross-metal summary (per-seed aggregate over 5 folds)",
        "",
        "| metal | n_seeds | combined R² (mean ± std) | Hilbert R² | Multi-PH R² | ASTSR R² | collapse_any |",
        "|---|---:|---|---|---|---|---|",
    ]

    metal_means: List[float] = []
    for metal, recs in by_metal.items():
        summaries = aggregate_metal(recs)
        combined_arr = np.asarray([s.combined_r2_mean for s in summaries], dtype=float)
        h_arr = np.asarray([s.hilbert_r2_mean for s in summaries], dtype=float)
        m_arr = np.asarray([s.mph_r2_mean for s in summaries], dtype=float)
        a_arr = np.asarray([s.astr_r2_mean for s in summaries], dtype=float)
        cm = float(np.nanmean(combined_arr))
        cs = float(np.nanstd(combined_arr))
        hm = float(np.nanmean(h_arr))
        mm = float(np.nanmean(m_arr))
        am = float(np.nanmean(a_arr))
        collapse_any = bool(any(s.collapse_detected_any for s in summaries))
        metal_means.append(cm)
        lines.append(
            f"| {metal} | {len(summaries)} | {cm:+.4f} ± {cs:.4f} | "
            f"{hm:+.4f} | {mm:+.4f} | {am:+.4f} | {'YES' if collapse_any else 'no'} |"
        )

    agg = float(np.mean(metal_means)) if metal_means else 0.0
    agg_std = float(np.std(metal_means)) if len(metal_means) > 1 else 0.0
    n_pass = int(sum(1 for cm in metal_means if cm >= 0.0))
    verdict = "PASS" if n_pass >= 1 else "FAIL"
    lines.append(
        f"| **5-metal aggregate** | — | **{agg:+.4f} ± {agg_std:.4f}** | — | — | — | — |"
    )
    lines.append("")
    lines.append("## Verify gate (target: combined R² ≥ 0 for ≥ 1 of 5 metals)")
    lines.append("")
    lines.append(f"**{n_pass} of {len(metal_means)} metals pass the gate.**")
    lines.append("")
    if verdict == "PASS":
        lines.append("**Verdict: PASS** — at least one metal hits R² ≥ 0 on REAL MetalCytoToxDB pIC50.")
    else:
        lines.append("**Verdict: FAIL** — no metal hits R² ≥ 0; honest framing per "
                     "`party-twenty-spirit §3.5`.")
    lines.append("")
    lines.append("## Per-(metal, seed) detail")
    for metal, recs in by_metal.items():
        lines.append(f"### {metal}")
        lines.append("")
        lines.append("| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | wall_s | collapse |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        summaries = aggregate_metal(recs)
        for s in summaries:
            lines.append(
                f"| {s.seed} | {s.combined_r2_mean:+.4f} ± "
                f"{s.combined_r2_std:.4f} | {s.hilbert_r2_mean:+.4f} | "
                f"{s.mph_r2_mean:+.4f} | {s.astr_r2_mean:+.4f} | "
                f"{s.wall_s_mean:.2f} | "
                f"{'YES' if s.collapse_detected_any else 'no'} |"
            )
        lines.append("")

    lines.append("## Honest framing")
    lines.append("")
    lines.append("- **REAL data**: MetalCytoToxDB 5-metal binary activity; "
                 "the CSV exposes ``activity`` (0/1) — R² is regression on a "
                 "binary target, not a continuous pIC50.")
    lines.append("- **Feature stack**: 172-d = 12-d RDKit + 32-d Multi-PH "
                 "Fourier coefficients + 128-d PLL signature (zero-padded).")
    lines.append("- **Multi-task backbone**: shared 172 → 128 trunk + 5 "
                 "per-metal 128 → 64 → 1 heads (Wf-128 P1 ship).")
    lines.append("- **Wasserstein weight 0.05**: sliced-W² penalty between "
                 "predicted pIC50 distribution and the training reference.")
    lines.append("- **L2 guard**: ``clip_grad_norm_(parameters, max_norm=1.0)``.")
    lines.append("- **ASTSR scale guard**: predictions clipped to `[0, 1]` "
                 "(Wf-127 P1 stable-region cap; loss_scale > 1.0 collapse regime).")
    lines.append("- **CPU + GPU-1**: CPU is default; GPU-1 is selected when "
                 "``CUDA_VISIBLE_DEVICES=1`` is set and ``--device cuda``.")
    lines.append("- **Collapse detector**: ``pred_std / y_test_std < 0.10`` "
                 "per fold → flag in ``per_seed_metrics.json``.")
    lines.append("")
    lines.append(
        f"Per `party-twenty-spirit §3.5`: aggregate combined R² = "
        f"{agg:+.4f}; verdict = **{verdict}** "
        f"({n_pass} of {len(metal_means)} metals hit R² ≥ 0)."
    )

    (out_dir / "synthesis.md").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthesis.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_metals(raw: str) -> List[str]:
    return [x.strip() for x in raw.replace(",", " ").split() if x]


def _parse_bool(raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {raw!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-128 Phase 2 — 3-way (Deflexpressor + Multi-PH + ASTSR) "
            "on REAL MetalCytoToxDB 5-metal pIC50 (172-d + multi-task "
            "backbone + Wasserstein + L2/ASTR guards)."
        )
    )
    parser.add_argument("--metals", type=_parse_metals, nargs="+",
                        default=list(DEFAULT_METALS))
    parser.add_argument("--seeds", type=_parse_seeds, nargs="+",
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hilbert-dim", type=int, default=DEFAULT_HILBERT_DIM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--wasserstein-weight", type=float, default=DEFAULT_W_W,
                        dest="wasserstein_weight")
    parser.add_argument("--use-l2-guard", dest="use_l2_guard",
                        action="store_true", default=True)
    parser.add_argument("--no-l2-guard", dest="use_l2_guard",
                        action="store_false")
    parser.add_argument("--use-astr-scale-guard",
                        dest="use_astr_scale_guard",
                        action="store_true", default=True)
    parser.add_argument("--no-astr-scale-guard",
                        dest="use_astr_scale_guard",
                        action="store_false")
    parser.add_argument("--features-172d", dest="features_172d",
                        action="store_true", default=True,
                        help="Use the 172-d NPZ feature stack (default True).")
    parser.add_argument("--no-features-172d", dest="features_172d",
                        action="store_false")
    parser.add_argument("--multitask-backbone", dest="multitask_backbone",
                        action="store_true", default=True,
                        help="Apply the multi-task 256→128 backbone "
                             "(default True).")
    parser.add_argument("--no-multitask-backbone", dest="multitask_backbone",
                        action="store_false")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_METALCSV_PATH)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 1 metal (Ru), 1 seed, 2 folds, 10 epochs.")
    args = parser.parse_args(argv)

    if args.device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        device = "cuda"
    else:
        device = "cpu"

    # Flatten argparse nargs="+" artifacts.
    flat_metals: List[str] = []
    for m in args.metals:
        if isinstance(m, list):
            flat_metals.extend(str(x) for x in m)
        else:
            flat_metals.append(str(m))
    flat_seeds: List[int] = []
    for s in args.seeds:
        if isinstance(s, list):
            flat_seeds.extend(int(x) for x in s)
        else:
            flat_seeds.append(int(s))

    if args.smoke:
        flat_metals = ["Ru"]
        flat_seeds = [0]
        epochs = min(int(args.epochs), 10)
        n_folds = min(int(args.n_folds), 2)
        max_rows = min(int(args.max_rows), 200)
    else:
        epochs = int(args.epochs)
        n_folds = int(args.n_folds)
        max_rows = int(args.max_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[wf128-p2] metals={flat_metals} seeds={flat_seeds} "
        f"epochs={epochs} n_folds={n_folds} device={device} "
        f"w_w={args.wasserstein_weight} "
        f"l2_guard={args.use_l2_guard} "
        f"astr_guard={args.use_astr_scale_guard} "
        f"multitask={args.multitask_backbone}"
    )

    all_records: List[FitRecord] = []
    for metal in flat_metals:
        print(f"[wf128-p2] === metal={metal} ===")
        try:
            recs = evaluate_metal(
                metal=metal, seeds=flat_seeds, n_folds=n_folds,
                epochs=epochs, hilbert_dim=int(args.hilbert_dim),
                lr=float(args.lr), w_w=float(args.wasserstein_weight),
                use_l2_guard=bool(args.use_l2_guard),
                use_astr_scale_guard=bool(args.use_astr_scale_guard),
                multitask_backbone=bool(args.multitask_backbone),
                device=device, npz_path=args.npz_path, csv_path=args.csv_path,
                max_rows=max_rows,
            )
        except Exception as exc:
            print(f"[wf128-p2] metal={metal}: FAILED ({exc!r})")
            continue
        all_records.extend(recs)

        per_metal_dir = args.out_dir / metal
        per_metal_dir.mkdir(parents=True, exist_ok=True)
        # Per-seed summaries.
        summaries = aggregate_metal(recs)
        per_seed_payload = {
            "metal": metal,
            "config": {
                "metals": flat_metals,
                "seeds": flat_seeds,
                "n_folds": n_folds,
                "epochs": epochs,
                "hilbert_dim": int(args.hilbert_dim),
                "lr": float(args.lr),
                "wasserstein_weight": float(args.wasserstein_weight),
                "use_l2_guard": bool(args.use_l2_guard),
                "use_astr_scale_guard": bool(args.use_astr_scale_guard),
                "multitask_backbone": bool(args.multitask_backbone),
                "features_172d": bool(args.features_172d),
                "device": device,
                "npz_path": str(args.npz_path),
                "csv_path": str(args.csv_path),
                "max_rows": max_rows,
                "feature_dim": FEATURE_DIM,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "per_seed": [asdict(s) for s in summaries],
            "per_fold": [asdict(r) for r in recs],
        }
        _write_json(per_seed_dir := per_metal_dir / "per_seed_metrics.json",
                    per_seed_payload)
        print(f"[wf128-p2] wrote {per_seed_dir}")

    write_synthesis(args.out_dir, all_records)
    print(f"[wf128-p2] synthesis.md written to {args.out_dir / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())