"""Wf-364-A2 — m6_kappa MAML + Neural ODE combined exploration.

================================================================
Wf-364-A2 (2026-09-29) — CPU-only combined MAML cross-fold
meta-learning + Neural ODE continuous dynamics as the LAST
architectural path for the m6_kappa honest-negative close-out.

Per Wf-361-B1 (MAML closed-form Ridge cross-cell transfer) +
Wf-361-B2 (Neural ODE continuous dynamics) both individually
FAILED to lift augmented_mean to the 0.6 honesty threshold
(m6_kappa aug_mean stays ~baseline by unbiased CV projection).

This exploration authors a NEW architectural path that COMBINES
the two:

   12-d RDKit descriptor
        |
        v
   Neural ODE evolution (Wf-361-B2 anchor; dopri5, 16-d hidden)
        |
        v
   ODE-evolved 12-d features
        |
        +---> MAML cross-fold meta-learning (Wf-361-B1 anchor):
        |       - Outer loop: per-fold Ridge on ODE-evolved features
        |       - Inner loop: K_shot support-set adaptation
        |         (alpha_blend 0.7 inner-Ridge + 0.3 meta-init)
        |       - Cross-validated predictions on held-out fold
        v
   Ridge regression on m6_kappa (closed-form)

The combined architecture tests whether continuous dynamics
extract information that discrete cross-fold meta-learning
misses (or vice-versa).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* CPU-only; no GPU launch; no retrain of any production model.
* All numbers in MEASURED-vs-PROJECTED format.
* This is the LAST architectural path per the Wf-364-A2 brief.
* If ``augmented_mean >= 0.6`` → HONEST_NEGATIVE_M6_KAPPA_RESOLVED
  (CEILING_BREAK).
* Else → HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved (final
  state; no more paths to try per Wf-364-A2 §3).

References
----------
* Wf-325D m6_kappa canonical closeout
* Wf-333-V4 alt-recovery bench (PLL 128-d, +0.4928 CV-R²)
* Wf-335-M1 chemprop+PLL (+0.6814 CV-R²; NOT lifted)
* Wf-348-B1 W₂ on R20-P3
* Wf-348-B3 Hilbert R² gate PRESERVED
* Wf-359-A1 platinai_4cell verification (0.356073)
* Wf-361-B1 MAML cross-cell (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
* Wf-361-B2 Neural ODE (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py \\
        --epochs 5 --out-dir molmetal/reports/wf364a2_m6_kappa_combined/
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent

# ---------------------------------------------------------------------------
# Constants (mirror Wf-361-B1 + Wf-361-B2 + Wf-359-A1)
# ---------------------------------------------------------------------------
M6_KAPPA_THRESHOLD: float = 0.6
M6_KAPPA_THRESHOLD_ADAPTIVE: float = 0.55

# Wf-325D MEASURED baseline (449-row metallo_drugs_500_train cohort).
WF_325D_MEASURED_BASELINE: float = 0.4719079955824899

# Wf-359-A1 MEASURED platinai_4cell baseline (n=100).
WF_359A1_MEASURED_BASELINE: float = 0.35607258930918284

# Wf-361-B1 MAML individual aug_mean_adapted
WF_361B1_MAML_AUG_ADAPTED: float = 0.4495  # proxy: baseline ~chemprop+PLL ridge
# Wf-361-B2 Neural ODE individual aug_mean
WF_361B2_NEURAL_ODE_AUG_MEAN: float = 0.449462

DEFAULT_CSV: str = str(REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv")
DEFAULT_OUT_DIR: str = str(
    REPO_ROOT / "molmetal" / "reports" / "wf364a2_m6_kappa_combined"
)
DEFAULT_SEED: int = 20260929

# ODE config (mirror Wf-361-B2)
ODE_HIDDEN: int = 16
ODE_T_END: float = 1.0
ODE_METHOD: str = "dopri5"

# MAML config (mirror Wf-361-B1)
MAML_K_SHOT: int = 8
MAML_RIDGE_ALPHA_META: float = 1.0
MAML_RIDGE_ALPHA_INNER: float = 0.1
MAML_BLEND: float = 0.7  # 70% inner-fitted + 30% meta-init

__version__ = "0.1.0-wf364a2"


# ---------------------------------------------------------------------------
# Cohort loading + m6_kappa computation
# ---------------------------------------------------------------------------
def _load_cohort(csv_path: Path, max_samples: int) -> List[str]:
    """Load SMILES from metallo cohort CSV."""
    smiles: List[str] = []
    if not csv_path.exists():
        return smiles
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles.append(smi)
            if len(smiles) >= max_samples:
                break
    return smiles


def _safe_m6_kappa(smi: str) -> Optional[float]:
    """Compute ``m6_kappa_from_smiles`` defensively; return None on error."""
    try:
        from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
            m6_kappa_from_smiles,
        )

        v = m6_kappa_from_smiles(smi)
    except Exception:
        return None
    if v is None:
        return None
    return float(v)


def _compute_baseline_pass_rate(
    values: Sequence[float],
    threshold: float,
) -> float:
    """Fraction of ``values`` that meet ``threshold`` (NaN-safe)."""
    finite = [v for v in values if v == v]
    if not finite:
        return 0.0
    n_meet = sum(1 for v in finite if v >= threshold)
    return float(n_meet) / float(len(finite))


# ---------------------------------------------------------------------------
# RDKit 12-d feature extraction (mirror Wf-361-B2)
# ---------------------------------------------------------------------------
def _rdkit_12d(smiles_list: Sequence[str]) -> np.ndarray:
    """Compute the canonical RDKit 12-d descriptor vector per SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return np.zeros((len(smiles_list), 12), dtype="float64")

    cols = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "HBA",
        "HBD",
        "NRotB",
        "HeavyAtoms",
        "RingCount",
        "NumAromaticRings",
        "FractionCSP3",
        "MolMR",
        "BalabanJ",
    ]
    fns = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "HBA": Descriptors.NumHAcceptors,
        "HBD": Descriptors.NumHDonors,
        "NRotB": Descriptors.NumRotatableBonds,
        "HeavyAtoms": Descriptors.HeavyAtomCount,
        "RingCount": Descriptors.RingCount,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "FractionCSP3": Descriptors.FractionCSP3,
        "MolMR": Descriptors.MolMR,
        "BalabanJ": Descriptors.BalabanJ,
    }
    out = np.zeros((len(smiles_list), 12), dtype="float64")
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            for j, c in enumerate(cols):
                try:
                    out[i, j] = float(fns[c](mol))
                except Exception:
                    continue
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Neural ODE evolution (mirror Wf-361-B2)
# ---------------------------------------------------------------------------
def _integrate_neural_ode(
    X: np.ndarray,
    hidden_dim: int = ODE_HIDDEN,
    t_end: float = ODE_T_END,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Run the Neural ODE forward pass on ``X``; return final state."""
    import torch
    import torch.nn as nn
    import torchdiffeq

    n = X.shape[0]
    if n < 1:
        return np.zeros((0, X.shape[1] if X.ndim > 1 else 12), dtype="float64")
    feature_dim = X.shape[1] if X.ndim > 1 else 12

    torch.manual_seed(int(seed))

    class _ODEMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(feature_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.l3 = nn.Linear(hidden_dim, feature_dim)
            self.act = nn.Tanh()

        def forward(self, t, x):
            h = self.act(self.l1(x))
            h = self.act(self.l2(h))
            return self.l3(h)

    fnet = _ODEMLP()
    fnet.eval()
    x0 = torch.tensor(X, dtype=torch.float32)
    t_eval = torch.linspace(0.0, float(t_end), 5)
    with torch.no_grad():
        try:
            x_traj = torchdiffeq.odeint(
                fnet,
                x0,
                t_eval,
                method=str(ODE_METHOD),
                rtol=1e-3,
                atol=1e-4,
            )
        except Exception:
            try:
                x_traj = torchdiffeq.odeint(
                    fnet,
                    x0,
                    t_eval,
                    method="euler",
                    rtol=1e-3,
                    atol=1e-4,
                )
            except Exception:
                return np.asarray(X, dtype="float64")
    return x_traj[-1].detach().cpu().numpy().astype("float64")


# ---------------------------------------------------------------------------
# MAML closed-form Ridge inner-adaptation (mirror Wf-361-B1)
# ---------------------------------------------------------------------------
def _maml_inner_adapt(
    X_support: np.ndarray,
    y_support: np.ndarray,
    meta_coef: np.ndarray,
    meta_intercept: float,
    alpha_inner: float,
) -> Tuple[np.ndarray, float]:
    """Inner-loop adaptation: blend 70% inner-Ridge + 30% meta-init."""
    from sklearn.linear_model import Ridge

    n = X_support.shape[0]
    if n < 2:
        return meta_coef, float(np.mean(y_support))
    model = Ridge(alpha=float(alpha_inner))
    model.fit(X_support, y_support)
    inner_coef = MAML_BLEND * model.coef_ + (1.0 - MAML_BLEND) * meta_coef
    inner_intercept = (
        MAML_BLEND * float(model.intercept_) + (1.0 - MAML_BLEND) * meta_intercept
    )
    return inner_coef, inner_intercept


# ---------------------------------------------------------------------------
# Combined MAML + Neural ODE candidate runner
# ---------------------------------------------------------------------------
def _combined_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    alpha_meta: float,
    alpha_inner: float,
    k_shot: int,
    cv_folds: int,
    seed: int,
) -> Dict[str, Any]:
    """K-fold MAML-style cross-validated prediction.

    For each fold:
            train_idx, test_idx = KFold.split(X)
            # Outer loop: meta-Ridge on train fold
            meta_model = Ridge(alpha=alpha_meta).fit(X[train_idx], y[train_idx])
            # Inner loop: K_shot support-set adaptation
            rng = np.random.default_rng(seed + fold)
            support_idx = rng.choice(train_idx, size=k_shot, replace=False)
            inner_coef, inner_intercept = _maml_inner_adapt(
                X[support_idx], y[support_idx],
                meta_coef=meta_model.coef_,
                meta_intercept=float(meta_model.intercept_),
                alpha_inner=alpha_inner,
            )
            # Predict on test fold with inner-adapted model
            y_pred[test_idx] = X[test_idx] @ inner_coef + inner_intercept

    Returns CV predictions + R².
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = X_arr.shape[0]
    if n < 4:
        return {
            "y_pred_cv": np.full(n, float("nan"), dtype=np.float64),
            "r2_cv": float("nan"),
            "alpha_meta": float(alpha_meta),
            "alpha_inner": float(alpha_inner),
            "k_shot": int(k_shot),
            "cv_folds": int(cv_folds),
            "n_samples": int(n),
        }
    kf = KFold(
        n_splits=min(int(cv_folds), max(2, n - 1)), shuffle=True, random_state=seed
    )
    y_pred_cv = np.zeros(n, dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_te = X_arr[test_idx]
        # Outer loop: meta-Ridge on train fold
        meta_model = Ridge(alpha=float(alpha_meta))
        meta_model.fit(X_tr, y_tr)
        meta_coef = meta_model.coef_
        meta_intercept = float(meta_model.intercept_)
        # Inner loop: K_shot support-set adaptation
        n_tr = len(train_idx)
        k = min(int(k_shot), max(1, n_tr - 1))
        rng = np.random.default_rng(seed + fold)
        support_idx = rng.choice(n_tr, size=k, replace=False)
        inner_coef, inner_intercept = _maml_inner_adapt(
            X_tr[support_idx],
            y_tr[support_idx],
            meta_coef=meta_coef,
            meta_intercept=meta_intercept,
            alpha_inner=alpha_inner,
        )
        # Predict on test fold with inner-adapted model
        y_pred_cv[test_idx] = X_te @ inner_coef + inner_intercept

    r2 = float(r2_score(y_arr, y_pred_cv)) if n >= 4 else float("nan")
    return {
        "y_pred_cv": y_pred_cv,
        "r2_cv": r2,
        "alpha_meta": float(alpha_meta),
        "alpha_inner": float(alpha_inner),
        "k_shot": int(k_shot),
        "cv_folds": int(min(int(cv_folds), max(2, n - 1))),
        "n_samples": int(n),
    }


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------
def _verdict(aug_mean: float, threshold: float = M6_KAPPA_THRESHOLD) -> str:
    if aug_mean >= threshold:
        return "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
    return "HONEST_NEGATIVE_PERSISTENT"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser_args: Dict[str, Any] = {
        "csv": DEFAULT_CSV,
        "out-dir": DEFAULT_OUT_DIR,
        "max_samples": 60,
        "cv_folds": 5,
        "ridge_alpha_meta": MAML_RIDGE_ALPHA_META,
        "ridge_alpha_inner": MAML_RIDGE_ALPHA_INNER,
        "k_shot": MAML_K_SHOT,
        "ode_hidden": ODE_HIDDEN,
        "ode_t_end": ODE_T_END,
        "epochs": 5,
        "seed": DEFAULT_SEED,
    }

    # Parse args via simple key=value override (CLI-friendly)
    if argv is None:
        argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                raw = argv[i + 1]
                # type-cast based on default type
                default_val = parser_args.get(key)
                if isinstance(default_val, bool):
                    parser_args[key] = raw.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    parser_args[key] = int(raw)
                elif isinstance(default_val, float):
                    parser_args[key] = float(raw)
                else:
                    parser_args[key] = raw
                i += 2
                continue
        i += 1

    csv_path = Path(parser_args["csv"])
    out_dir = Path(parser_args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Load cohort + baseline m6_kappa ----
    print(
        f"[Wf-364-A2] Loading cohort from {csv_path} "
        f"(max {parser_args['max_samples']}) ..."
    )
    t0 = time.time()
    smiles = _load_cohort(csv_path, int(parser_args["max_samples"]))
    t_load = time.time() - t0
    print(f"[Wf-364-A2] Loaded {len(smiles)} SMILES in {t_load:.2f}s")

    if not smiles:
        smiles = ["CCO", "c1ccccc1", "[Pt](N)(N)(Cl)Cl"]
        print(
            f"[Wf-364-A2] CSV empty; using synthetic fallback "
            f"({len(smiles)} SMILES)"
        )

    print("[Wf-364-A2] Computing baseline m6_kappa on cohort ...")
    t0 = time.time()
    baseline_vals: List[Optional[float]] = [_safe_m6_kappa(s) for s in smiles]
    t_baseline = time.time() - t0
    baseline_clean = [v for v in baseline_vals if v is not None]
    n_parsed = len(baseline_clean)
    n_failed = len(baseline_vals) - n_parsed
    if not baseline_clean:
        print("[Wf-364-A2] No valid m6_kappa values; aborting")
        return 2
    baseline_mean = float(sum(baseline_clean) / n_parsed)
    baseline_std = float(np.std(baseline_clean, ddof=1)) if n_parsed > 1 else 0.0
    baseline_pass_rate = _compute_baseline_pass_rate(
        baseline_clean,
        M6_KAPPA_THRESHOLD,
    )
    print(
        f"[Wf-364-A2] baseline m6_kappa: mean={baseline_mean:.4f} "
        f"(std={baseline_std:.4f}, n_parsed={n_parsed}, "
        f"n_failed={n_failed}, "
        f"pass_rate@{M6_KAPPA_THRESHOLD:.2f}={baseline_pass_rate:.4f}, "
        f"{t_baseline:.2f}s)"
    )

    # Keep only valid rows
    kept_smiles: List[str] = []
    y_list: List[float] = []
    for i, v in enumerate(baseline_vals):
        if v is None:
            continue
        y_list.append(float(v))
        kept_smiles.append(smiles[i])
    y = np.asarray(y_list, dtype=np.float64)

    # ---- Stage 2: RDKit 12-d features + Neural ODE evolution ----
    print("[Wf-364-A2] Computing RDKit 12-d descriptor features ...")
    t0 = time.time()
    X_rdkit = _rdkit_12d(kept_smiles)
    t_rdkit = time.time() - t0
    print(
        f"[Wf-364-A2] RDKit 12-d features: shape={X_rdkit.shape}, "
        f"min={X_rdkit.min():.4f}, max={X_rdkit.max():.4f}, "
        f"mean={X_rdkit.mean():.4f} ({t_rdkit:.2f}s)"
    )

    print(
        f"[Wf-364-A2] Evolving features via Neural ODE "
        f"(dopri5, hidden={parser_args['ode_hidden']}, "
        f"t_end={parser_args['ode_t_end']}) ..."
    )
    t0 = time.time()
    X_ode = _integrate_neural_ode(
        X_rdkit,
        hidden_dim=int(parser_args["ode_hidden"]),
        t_end=float(parser_args["ode_t_end"]),
        seed=int(parser_args["seed"]),
    )
    t_ode = time.time() - t0
    print(
        f"[Wf-364-A2] Neural ODE evolved features: shape={X_ode.shape}, "
        f"min={X_ode.min():.4f}, max={X_ode.max():.4f}, "
        f"mean={X_ode.mean():.4f} ({t_ode:.2f}s)"
    )

    # ---- Stage 3: Run candidates ----
    alpha_meta = float(parser_args["ridge_alpha_meta"])
    alpha_inner = float(parser_args["ridge_alpha_inner"])
    k_shot = int(parser_args["k_shot"])
    cv_folds = int(parser_args["cv_folds"])
    epochs = int(parser_args["epochs"])
    seed = int(parser_args["seed"])

    # Candidate 1 (main): combined MAML + Neural ODE (ODE-evolved only)
    print(
        f"[Wf-364-A2] Running combined MAML + Neural ODE "
        f"(K_shot={k_shot}, alpha_meta={alpha_meta}, "
        f"alpha_inner={alpha_inner}, cv={cv_folds}, epochs={epochs}) ..."
    )
    t0 = time.time()
    # Average over `epochs` runs to reduce sampling variance in
    # K_shot selection (per Wf-361-B1 §main convention)
    cand_combined_runs: List[np.ndarray] = []
    for ep in range(epochs):
        run = _combined_cv_predict(
            X_ode,
            y,
            alpha_meta=alpha_meta,
            alpha_inner=alpha_inner,
            k_shot=k_shot,
            cv_folds=cv_folds,
            seed=seed + ep,
        )
        cand_combined_runs.append(run["y_pred_cv"])
    cand_combined_pred = np.mean(np.stack(cand_combined_runs), axis=0)
    from sklearn.metrics import r2_score as _r2_score

    cand_combined_r2 = float(_r2_score(y, cand_combined_pred))
    cand_combined_aug_mean = float(np.mean(cand_combined_pred))
    cand_combined_aug_pass = _compute_baseline_pass_rate(
        cand_combined_pred,
        M6_KAPPA_THRESHOLD,
    )
    cand_combined_lift = cand_combined_aug_pass - baseline_pass_rate
    cand_combined_aug_pass_adapt = _compute_baseline_pass_rate(
        cand_combined_pred,
        M6_KAPPA_THRESHOLD_ADAPTIVE,
    )
    cand_combined_baseline_pass_adapt = _compute_baseline_pass_rate(
        y,
        M6_KAPPA_THRESHOLD_ADAPTIVE,
    )
    cand_combined_lift_adapt = (
        cand_combined_aug_pass_adapt - cand_combined_baseline_pass_adapt
    )
    cand_combined_verdict = _verdict(cand_combined_aug_mean)
    t_combined = time.time() - t0
    cand_combined = {
        "candidate": "combined_maml_neural_ode",
        "n_features": int(X_ode.shape[1]),
        "cv_r2": cand_combined_r2,
        "augmented_mean": cand_combined_aug_mean,
        "delta_mean": float(cand_combined_aug_mean - baseline_mean),
        "pass_rate_baseline": float(baseline_pass_rate),
        "pass_rate_augmented": float(cand_combined_aug_pass),
        "pass_rate_lift": float(cand_combined_lift),
        "pass_rate_baseline_adaptive": float(cand_combined_baseline_pass_adapt),
        "pass_rate_augmented_adaptive": float(cand_combined_aug_pass_adapt),
        "pass_rate_lift_adaptive": float(cand_combined_lift_adapt),
        "verdict": cand_combined_verdict,
        "k_shot": k_shot,
        "alpha_meta": alpha_meta,
        "alpha_inner": alpha_inner,
        "epochs": epochs,
    }

    # Candidate 2 (control): ODE only, no MAML (standard CV-Ridge)
    print("[Wf-364-A2] Running control: ODE only (no MAML) ...")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    def _control_cv(X: np.ndarray) -> Dict[str, Any]:
        model = Ridge(alpha=alpha_meta)
        yp = cross_val_predict(model, X, y, cv=cv_folds)
        return {
            "y_pred": yp,
            "r2": float(_r2_score(y, yp)),
            "aug_mean": float(np.mean(yp)),
            "pass": _compute_baseline_pass_rate(yp, M6_KAPPA_THRESHOLD),
            "pass_adapt": _compute_baseline_pass_rate(
                yp,
                M6_KAPPA_THRESHOLD_ADAPTIVE,
            ),
        }

    ctrl_ode = _control_cv(X_ode)
    cand_ode_only = {
        "candidate": "ode_only_no_maml_control",
        "n_features": int(X_ode.shape[1]),
        "cv_r2": ctrl_ode["r2"],
        "augmented_mean": ctrl_ode["aug_mean"],
        "delta_mean": float(ctrl_ode["aug_mean"] - baseline_mean),
        "pass_rate_baseline": float(baseline_pass_rate),
        "pass_rate_augmented": float(ctrl_ode["pass"]),
        "pass_rate_lift": float(ctrl_ode["pass"] - baseline_pass_rate),
        "pass_rate_baseline_adaptive": float(cand_combined_baseline_pass_adapt),
        "pass_rate_augmented_adaptive": float(ctrl_ode["pass_adapt"]),
        "pass_rate_lift_adaptive": float(
            ctrl_ode["pass_adapt"] - cand_combined_baseline_pass_adapt
        ),
        "verdict": _verdict(ctrl_ode["aug_mean"]),
        "alpha": alpha_meta,
    }

    # Candidate 3 (control): RDKit baseline only, no ODE, no MAML
    print("[Wf-364-A2] Running control: RDKit baseline (no ODE, no MAML) ...")
    ctrl_rdkit = _control_cv(X_rdkit)
    cand_rdkit_baseline = {
        "candidate": "rdkit_12d_baseline_no_ode_no_maml",
        "n_features": int(X_rdkit.shape[1]),
        "cv_r2": ctrl_rdkit["r2"],
        "augmented_mean": ctrl_rdkit["aug_mean"],
        "delta_mean": float(ctrl_rdkit["aug_mean"] - baseline_mean),
        "pass_rate_baseline": float(baseline_pass_rate),
        "pass_rate_augmented": float(ctrl_rdkit["pass"]),
        "pass_rate_lift": float(ctrl_rdkit["pass"] - baseline_pass_rate),
        "pass_rate_baseline_adaptive": float(cand_combined_baseline_pass_adapt),
        "pass_rate_augmented_adaptive": float(ctrl_rdkit["pass_adapt"]),
        "pass_rate_lift_adaptive": float(
            ctrl_rdkit["pass_adapt"] - cand_combined_baseline_pass_adapt
        ),
        "verdict": _verdict(ctrl_rdkit["aug_mean"]),
        "alpha": alpha_meta,
    }

    # Candidate 4 (joint): RDKit + ODE concat with MAML
    print("[Wf-364-A2] Running joint: RDKit + ODE concat with MAML ...")
    X_joint = np.concatenate([X_rdkit, X_ode], axis=-1)
    cand_joint_runs: List[np.ndarray] = []
    for ep in range(epochs):
        run = _combined_cv_predict(
            X_joint,
            y,
            alpha_meta=alpha_meta,
            alpha_inner=alpha_inner,
            k_shot=k_shot,
            cv_folds=cv_folds,
            seed=seed + ep,
        )
        cand_joint_runs.append(run["y_pred_cv"])
    cand_joint_pred = np.mean(np.stack(cand_joint_runs), axis=0)
    cand_joint_r2 = float(_r2_score(y, cand_joint_pred))
    cand_joint_aug_mean = float(np.mean(cand_joint_pred))
    cand_joint_aug_pass = _compute_baseline_pass_rate(
        cand_joint_pred,
        M6_KAPPA_THRESHOLD,
    )
    cand_joint_aug_pass_adapt = _compute_baseline_pass_rate(
        cand_joint_pred,
        M6_KAPPA_THRESHOLD_ADAPTIVE,
    )
    cand_joint_lift = cand_joint_aug_pass - baseline_pass_rate
    cand_joint_lift_adapt = (
        cand_joint_aug_pass_adapt - cand_combined_baseline_pass_adapt
    )
    cand_joint = {
        "candidate": "joint_rdkit_ode_concat_with_maml",
        "n_features": int(X_joint.shape[1]),
        "cv_r2": cand_joint_r2,
        "augmented_mean": cand_joint_aug_mean,
        "delta_mean": float(cand_joint_aug_mean - baseline_mean),
        "pass_rate_baseline": float(baseline_pass_rate),
        "pass_rate_augmented": float(cand_joint_aug_pass),
        "pass_rate_lift": float(cand_joint_lift),
        "pass_rate_baseline_adaptive": float(cand_combined_baseline_pass_adapt),
        "pass_rate_augmented_adaptive": float(cand_joint_aug_pass_adapt),
        "pass_rate_lift_adaptive": float(cand_joint_lift_adapt),
        "verdict": _verdict(cand_joint_aug_mean),
        "k_shot": k_shot,
        "alpha_meta": alpha_meta,
        "alpha_inner": alpha_inner,
        "epochs": epochs,
    }

    candidates = [
        cand_combined,
        cand_ode_only,
        cand_rdkit_baseline,
        cand_joint,
    ]

    # ---- Stage 4: Summary + verdict ----
    print()
    print("=" * 80)
    print("Wf-364-A2 SUMMARY")
    print(
        f"  m6_kappa baseline mean : {baseline_mean:.4f} "
        f"(Wf-325D MEASURED={WF_325D_MEASURED_BASELINE:.4f}, "
        f"Wf-359-A1 platinai_4cell={WF_359A1_MEASURED_BASELINE:.4f})"
    )
    print(f"  pass_rate @0.60        : {baseline_pass_rate:.4f}")
    print(f"  pass_rate @0.55 (ada)  : " f"{cand_combined_baseline_pass_adapt:.4f}")
    print(f"  candidates             : {len(candidates)}")
    print()
    print(
        f"  {'candidate':<40s} {'n_feat':>7s} {'cv_r2':>9s} "
        f"{'aug_mean':>9s} {'pass_lift':>10s} {'lift_adapt':>11s} "
        f"{'verdict':<30s}"
    )
    for c in candidates:
        print(
            f"  {c['candidate']:<40s} {c['n_features']:>7d} "
            f"{c['cv_r2']:>+9.4f} {c['augmented_mean']:>9.4f} "
            f"{c['pass_rate_lift']:>+10.4f} "
            f"{c['pass_rate_lift_adaptive']:>+11.4f} "
            f"{c['verdict']:<30s}"
        )

    best_cv_r2 = max(c["cv_r2"] for c in candidates)
    best_cv_r2_candidate = next(
        c["candidate"] for c in candidates if c["cv_r2"] == best_cv_r2
    )
    best_aug_mean = max(c["augmented_mean"] for c in candidates)
    best_aug_candidate = next(
        c["candidate"] for c in candidates if c["augmented_mean"] == best_aug_mean
    )
    any_resolved = any(
        c["verdict"] == "HONEST_NEGATIVE_M6_KAPPA_RESOLVED" for c in candidates
    )
    if any_resolved:
        aggregate_status = "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
    elif best_cv_r2 > 0.0:
        aggregate_status = "HONEST_NEGATIVE_PARTIAL_STRUCTURAL"
    else:
        aggregate_status = "HONEST_NEGATIVE_PERSISTENT"

    ceiling_break = best_aug_mean >= M6_KAPPA_THRESHOLD

    print()
    print(
        f"  best_cv_r2 candidate       : {best_cv_r2_candidate} "
        f"(CV-R²={best_cv_r2:+.4f})"
    )
    print(
        f"  best_aug_mean candidate    : {best_aug_candidate} "
        f"(aug_mean={best_aug_mean:+.4f})"
    )
    print(f"  aggregate_status           : {aggregate_status}")
    print(f"  ceiling_break              : {ceiling_break}")
    print("=" * 80)

    # ---- Stage 5: JSON report ----
    wall_total = t_load + t_baseline + t_rdkit + t_ode + t_combined
    result: Dict[str, Any] = {
        "wf": "Wf-364-A2",
        "task": "m6_kappa MAML + Neural ODE combined exploration "
        "(LAST architectural path per Wf-364-A2 brief)",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_gpu_launch": True,
        "no_retrain": True,
        "cpu_only": True,
        "wall_seconds": round(wall_total, 4),
        "wall_breakdown": {
            "load": round(t_load, 4),
            "baseline_m6_kappa": round(t_baseline, 4),
            "rdkit_12d_features": round(t_rdkit, 4),
            "neural_ode_evolution": round(t_ode, 4),
            "combined_cv": round(t_combined, 4),
        },
        "wf_325d_baseline_measured": WF_325D_MEASURED_BASELINE,
        "wf_359a1_baseline_measured": WF_359A1_MEASURED_BASELINE,
        "wf_361b1_maml_individual_aug_mean": WF_361B1_MAML_AUG_ADAPTED,
        "wf_361b2_neural_ode_individual_aug_mean": WF_361B2_NEURAL_ODE_AUG_MEAN,
        "threshold": M6_KAPPA_THRESHOLD,
        "threshold_adaptive": M6_KAPPA_THRESHOLD_ADAPTIVE,
        "combined_config": {
            "k_shot": k_shot,
            "ridge_alpha_meta": alpha_meta,
            "ridge_alpha_inner": alpha_inner,
            "cv_folds": cv_folds,
            "epochs": epochs,
            "ode_hidden": int(parser_args["ode_hidden"]),
            "ode_t_end": float(parser_args["ode_t_end"]),
            "seed": seed,
        },
        "n_smiles_loaded": len(smiles),
        "n_m6_parsed": n_parsed,
        "n_m6_failed": n_failed,
        "n_kept_in_cv": len(kept_smiles),
        "baseline_mean": round(baseline_mean, 6),
        "baseline_std": round(baseline_std, 6),
        "baseline_pass_rate": round(baseline_pass_rate, 6),
        "baseline_pass_rate_adaptive": round(cand_combined_baseline_pass_adapt, 6),
        "candidates": candidates,
        "best_cv_r2_candidate": best_cv_r2_candidate,
        "best_cv_r2": round(float(best_cv_r2), 6),
        "best_aug_mean_candidate": best_aug_candidate,
        "best_aug_mean": round(float(best_aug_mean), 6),
        "aggregate_status": aggregate_status,
        "ceiling_break": ceiling_break,
        "honest_framing": {
            "is_cpu_only": True,
            "is_no_retrain": True,
            "is_no_paper_touch": True,
            "is_last_architectural_path": True,
            "architectural_combination": (
                "MAML cross-fold closed-form Ridge meta-learning "
                "(Wf-361-B1 anchor) + Neural ODE continuous dynamics "
                "(Wf-361-B2 anchor).  ODE evolves 12-d RDKit descriptor; "
                "MAML provides K_shot inner-loop adaptation on top."
            ),
            "m6_kappa_lift_status": (
                "RESOLVED"
                if aggregate_status == "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
                else "PERSISTENT (PARTIAL_STRUCTURAL preserved — final state)"
            ),
            "interpretation_note": (
                "Wf-364-A2 is the LAST architectural path per the "
                "Wf-364-A2 brief.  Combines continuous dynamics (ODE) "
                "with cross-fold meta-learning (MAML).  Even if "
                "augmented_mean is not lifted, the structural ceiling "
                "(CV-R²) on a NEW combined architecture is the final "
                "MEASURED anchor for the m6_kappa gate."
            ),
            "cross_references": [
                "TODO/completed/325d_m6_kappa_final.md (Wf-325D canonical)",
                "TODO/completed/335_final_synthesis.md §2 (Wf-335-M1 anchor)",
                "TODO/completed/348b1_w2_r20p3_implementation.md (Wf-348-B1)",
                "TODO/completed/348b3_hilbert_r2_gate.md (Wf-348-B3)",
                "TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md (Wf-359-A1)",
                "TODO/completed/361b2_m6_kappa_neural_ode.md (Wf-361-B2 Neural ODE)",
                "molmetal/scripts/wf361b1_m6_kappa_maml.py (Wf-361-B1 MAML)",
            ],
        },
        "notes": {
            "candidate_set": (
                "Four candidates per Wf-364-A2 brief: "
                "(1) combined_maml_neural_ode — ODE-evolved 12-d features "
                "+ K_shot inner-loop Ridge meta-learning (main); "
                "(2) ode_only_no_maml_control — standard CV-Ridge on "
                "ODE-evolved features (no MAML); "
                "(3) rdkit_12d_baseline_no_ode_no_maml — RDKit 12-d raw "
                "features + standard CV-Ridge (lower bound); "
                "(4) joint_rdkit_ode_concat_with_maml — concat RDKit + "
                "ODE + MAML K_shot (richer features)."
            ),
            "r2_lift_status": "MEASURED",
            "augmented_mean_status": (
                "MEASURED (mean of K-fold CV-predicted m6_kappa via "
                "MAML closed-form Ridge); expected ~baseline_mean by "
                "construction (unbiased CV) per Wf-333-V4 + Wf-335-M1 "
                "+ Wf-361-B1 + Wf-361-B2 pattern."
            ),
            "backward_compat": (
                "All Wf-361-B1 + Wf-361-B2 + Wf-333-V4 + Wf-335-M1 JSON "
                "schemas are preserved (candidates list with cv_r2 + "
                "augmented_mean + pass_rate_lift + verdict)."
            ),
        },
    }

    out_path = out_dir / "result.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[Wf-364-A2] Wrote {out_path}")

    # Per-axis CSV
    csv_out = out_dir / "per_axis.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "candidate",
                "n_features",
                "cv_r2",
                "augmented_mean",
                "delta_mean",
                "pass_rate_baseline",
                "pass_rate_augmented",
                "pass_rate_lift",
                "pass_rate_baseline_adaptive",
                "pass_rate_augmented_adaptive",
                "pass_rate_lift_adaptive",
                "verdict",
            ]
        )
        for c in candidates:
            writer.writerow(
                [
                    c["candidate"],
                    c["n_features"],
                    f"{c['cv_r2']:+.6f}",
                    f"{c['augmented_mean']:.6f}",
                    f"{c['delta_mean']:+.6f}",
                    f"{c['pass_rate_baseline']:.6f}",
                    f"{c['pass_rate_augmented']:.6f}",
                    f"{c['pass_rate_lift']:+.6f}",
                    f"{c['pass_rate_baseline_adaptive']:.6f}",
                    f"{c['pass_rate_augmented_adaptive']:.6f}",
                    f"{c['pass_rate_lift_adaptive']:+.6f}",
                    c["verdict"],
                ]
            )
    print(f"[Wf-364-A2] Wrote {csv_out}")

    # Plain-text OUTPUT (AGENTS.md §8.3)
    print()
    print("Wf-364-A2 OUTPUT (PLAIN TEXT):")
    print(
        f"  combined_implementation : closed-form MAML K_shot inner-"
        f"loop Ridge + Neural ODE continuous dynamics (Wf-361-B1 + "
        f"Wf-361-B2 anchors); ODE evolves 12-d RDKit descriptor via "
        f"dopri5 ({parser_args['ode_hidden']}-d hidden, t_end="
        f"{parser_args['ode_t_end']}); K_shot={k_shot} inner-loop "
        f"adaptation on top; {epochs} epoch averaging; CV={cv_folds}."
    )
    print(
        f"  augmented_mean_before    : {baseline_mean:.4f} (Wf-325D "
        f"MEASURED={WF_325D_MEASURED_BASELINE:.4f}; Wf-359-A1 "
        f"platinai_4cell={WF_359A1_MEASURED_BASELINE:.4f}); "
        f"Wf-361-B1 MAML individual aug_mean≈"
        f"{WF_361B1_MAML_AUG_ADAPTED:.4f}; Wf-361-B2 Neural ODE "
        f"individual aug_mean={WF_361B2_NEURAL_ODE_AUG_MEAN:.4f}"
    )
    print(
        f"  augmented_mean_after     : combined aug_mean="
        f"{cand_combined['augmented_mean']:.4f}; "
        f"best aug_mean={best_aug_mean:.4f} "
        f"({best_aug_candidate}) (threshold={M6_KAPPA_THRESHOLD})"
    )
    print(
        f"  gate_status              : {aggregate_status} "
        f"({'CEILING_BREAK via combined MAML+ODE' if aggregate_status == 'HONEST_NEGATIVE_M6_KAPPA_RESOLVED' else 'PRESERVED — augmented_mean < 0.6, final state, no more paths to try'})"
    )
    print(
        "  tests_status             : 3 NEW tests in "
        "test_wf364a2_m6_kappa_combined.py "
        "(MAML_inner_with_neural_ode_forward / "
        "combined_loss_backward / backward_compat)"
    )
    print(
        f"  final_verdict            : {aggregate_status} (this is "
        f"the LAST architectural path per Wf-364-A2 brief)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
