"""Wf-79 Phase 2 — 3-way + PLL combined R^2 benchmark (CPU + GPU-1 RTX 5090).

================================================================
Combine Wf-67 R20-P4 3-way with Wf-66 R20-P3 PLL 128-d on PlatinAI 4-cell.
Compute combined 3-way + PLL R^2 vs 3-way only vs PLL only.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
This is the **CPU smoke** layer for the Wf-79 P2 task. The full
GPU retrain cycle is reserved for R20-P3 (PLL 128-d) + R20-P4
(3-way on MetalCytoToxDB 5-metal) follow-on. Per the harness task
doc, the Wf-67 P4.1 + Wf-66 P3.2 deliverables are PROJECTED @ R20
GPU — they are NOT on disk. We fall back to the **MEASURED
CPU baselines**:

* Wf-41 P1 3-way (ASTSR + Multi-PH + PLL 16-d) on PlatinAI 5K
  synthetic: R^2 = -0.3595 ± 0.2806 (n=100, 3 seeds, 92-d feature)
* Wf-37 P3 Multi-PH + PLL extraction on NCI-60:
  R^2 = 0.087 ± 0.048 (n=200, 3 seeds, 73-d feature)

This Wf-79 P2 CPU smoke runs the same 3-way module on **real
PlatinAI 4-cell pIC50** at smoke parameters (n=200, 16-d PLL,
8-bin Multi-PH, 64-d Hilbert, 5 seeds) and computes:

| Method            | Stream                              |
|-------------------|-------------------------------------|
| ``pll_only``      | PLL (k-d)                           |
| ``ast_only``      | ASTSR (12-d RDKit)                  |
| ``3way``          | ASTSR + Multi-PH + PLL              |
| ``3way_pll_aug``  | ASTSR + Multi-PH + PLL + PLL^2 + ASTSR·PLL cross |
| ``joint``         | ASTSR + Multi-PH (Wf-21 P5 ref)     |

For each method, per-seed R^2 on a deterministic 80/20 split
of PlatinAI 4-cell (real pIC50 = mean of 4 cell-line log10 IC50).
Then we report the **combined 3way_pll_aug - 3way** lift
delta R^2 and the **3-way - PLL-only** lift.

Outputs
-------
* ``molmetal/reports/wf79_p2_3way_pll/platinai_4cell/per_seed_metrics.json``
* ``molmetal/reports/wf79_p2_3way_pll/platinai_4cell/summary.json``
* ``molmetal/reports/wf79_p2_3way_pll/synthesis.md``

Usage::

    source env.sh && /home/hugo/codes/mol-metal/.venv/bin/python -u \\
        molmetal/scripts/bench_3way_pll_combined.py \\
        --dataset platinai_4cell --seeds 0 1 2 3 4 --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Ensure project root is on sys.path.
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))


LOGGER = logging.getLogger("wf79_p2_3way_pll")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)


PLL_K = 16
MPH_N_BINS = 8
MPH_MAX_RADIUS = 5.0
MPH_MAX_COORD = 4.0
HILBERT_DIM = 64
HILBERT_LR = 0.05
HILBERT_ITERS = 60
TEST_FRAC = 0.2


def _load_platinai_4cell(
    csv_path: Path,
    n_samples: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    """Load PlatinAI 4-cell SMILES + mean log10 IC50 (real pIC50 proxy).

    Reads ``complex_id,smiles,ic50_mcf7_nm,ic50_a549_nm,ic50_a2780_nm,
    ic50_a2780cis_nm``.  We take the **mean log10 IC50** across the
    four cell lines as the continuous regression target (a noisy
    but real pIC50 proxy).
    """
    import csv

    smiles_list: List[str] = []
    pic50_list: List[float] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            try:
                vals = []
                for col in (
                    "ic50_mcf7_nm",
                    "ic50_a549_nm",
                    "ic50_a2780_nm",
                    "ic50_a2780cis_nm",
                ):
                    v = float(row.get(col) or "nan")
                    if not math.isnan(v) and v > 0:
                        vals.append(-math.log10(v * 1e-9))
                if len(vals) < 3:
                    continue
                smiles_list.append(smi)
                pic50_list.append(float(np.mean(vals)))
            except Exception:
                continue

    if not smiles_list:
        raise RuntimeError(f"No valid PlatinAI 4-cell rows in {csv_path}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(smiles_list))[: min(n_samples, len(smiles_list))]
    return (
        [smiles_list[i] for i in idx],
        np.asarray(pic50_list, dtype=float)[idx],
    )


def _load_metalcytotox_5metal(
    csv_path: Path,
    n_samples: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    """Load MetalCytoToxDB 5-metal — use activity (binary) + log-pIC50 proxy.

    Reads ``smiles,metal,scaffold_id,split,activity``. The original
    ``activity`` is binary (0/1); we **augment** with a per-row
    random ``N(0, 0.05)`` noise to lift it off the SS_tot-collapsing
    binary target (Wf-25 numerical fix), giving the regressor a
    near-continuous signal.

    For n_samples, we stratify-sample across the 5 metals
    (Ru, Pt, Ir, Pd, Os) to keep the per-metal subset balanced.
    """
    import csv

    by_metal: Dict[str, List[Tuple[str, float]]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            met = (row.get("metal") or "").strip()
            try:
                act = float(row.get("activity") or "nan")
            except Exception:
                continue
            if not smi or not met or math.isnan(act):
                continue
            # Bypass: lift binary target off the SS_tot collapse.
            noisy_act = act + float(
                np.random.default_rng(hash(smi) & 0xFFFFFFFF).normal(0.0, 0.05)
            )
            by_metal.setdefault(met, []).append((smi, noisy_act))

    if not by_metal:
        raise RuntimeError(f"No valid MetalCytoToxDB rows in {csv_path}")

    # Stratified per-metal sampling.
    rng = np.random.default_rng(seed)
    metals = sorted(by_metal.keys())
    per_metal = max(1, n_samples // len(metals))
    smiles_list: List[str] = []
    target_list: List[float] = []
    for met in metals:
        rows = by_metal[met]
        idx = rng.permutation(len(rows))[: min(per_metal, len(rows))]
        for i in idx:
            smi, val = rows[i]
            smiles_list.append(smi)
            target_list.append(val)

    return smiles_list, np.asarray(target_list, dtype=float)


def _ast_features(smiles_list: Sequence[str]) -> np.ndarray:
    """ASTSR-proxy 12-d RDKit descriptor features (Wf-41 P1 baseline)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    feats = np.zeros((len(smiles_list), 12), dtype=float)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            feats[i, 0] = Descriptors.MolWt(mol)
            feats[i, 1] = Descriptors.MolLogP(mol)
            feats[i, 2] = Descriptors.TPSA(mol)
            feats[i, 3] = rdMolDescriptors.CalcNumHBA(mol)
            feats[i, 4] = rdMolDescriptors.CalcNumHBD(mol)
            feats[i, 5] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            feats[i, 6] = rdMolDescriptors.CalcNumRings(mol)
            feats[i, 7] = Descriptors.FractionCSP3(mol)
            feats[i, 8] = Descriptors.HeavyAtomCount(mol)
            feats[i, 9] = Descriptors.NumValenceElectrons(mol)
            feats[i, 10] = Descriptors.NHOHCount(mol)
            feats[i, 11] = Descriptors.NOCount(mol)
        except Exception:
            continue
    return feats


def _multiph_features(
    smiles_list: Sequence[str],
    n_bins: int = MPH_N_BINS,
    max_radius: float = MPH_MAX_RADIUS,
    max_coord: float = MPH_MAX_COORD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-PH flattened feature map via MultiPHPLLSignature.

    Returns a (n_samples, n_bins * n_bins) matrix + a boolean mask
    of valid samples.
    """
    from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
        MultiPHPLLSignature,
    )

    sig = MultiPHPLLSignature(n_bins=int(n_bins), k=int(PLL_K))
    out_dim = int(n_bins) * int(n_bins)
    feats = np.zeros((len(smiles_list), out_dim), dtype=float)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            F, _ = sig.compute(smi)
            arr = np.asarray(F, dtype=float).reshape(-1)
            n = min(arr.size, out_dim)
            feats[i, :n] = arr[:n]
            valid[i] = True
        except Exception:
            valid[i] = False
    return feats, valid


def _pll_features(
    smiles_list: Sequence[str],
    k: int = PLL_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """PLL top-k eigenvalue signature via MultiPHPLLSignature (PLL-only)."""
    from molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration import (
        MultiPHPLLSignature,
    )
    from molmetal.math_skills.multi_parameter_ph import (
        TwoParameterFiltration,
    )

    # We only need PLL — feed a no-op multi-PH filter by passing
    # a minimal n_bins=2 TwoParameterFiltration. The PLL spectrum
    # is computed independently of the multi-PH map.
    mph_min = TwoParameterFiltration(
        max_radius=MPH_MAX_RADIUS,
        max_coord=MPH_MAX_COORD,
        n_bins=2,
    )
    sig = MultiPHPLLSignature(
        multi_ph=mph_min,
        n_bins=2,
        k=int(k),
        knn=8,
    )

    feats = np.zeros((len(smiles_list), k), dtype=float)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            _, lam = sig.compute(smi)
            arr = np.asarray(lam, dtype=float).reshape(-1)
            n = min(arr.size, k)
            feats[i, :n] = arr[:n]
            valid[i] = True
        except Exception:
            continue
    return feats, valid


def _stable_ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Closed-form ridge regression (Wf-39 P2 / Wf-41 P1 fix).

    Avoids the Wf-25 numerical instability on ill-conditioned ``XᵀX``
    when ``F >= N``.  Returns ``(w, train_r2)``.
    """
    n, f = X.shape
    if n < 2 or f < 1:
        return np.zeros(f), 0.0
    XtX = X.T @ X
    # Ridge stabiliser.
    XtX = XtX + float(ridge_alpha) * np.eye(f)
    try:
        w = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    y_hat = X @ w
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return w, float(train_r2)


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.asarray(X @ w, dtype=float).reshape(-1)


def _train_hilbert(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hilbert_dim: int = HILBERT_DIM,
    lr: float = HILBERT_LR,
    iters: int = HILBERT_ITERS,
    ridge_alpha: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Train a closed-form ridge regressor; return (R^2, train R^2, wall, mem).

    Uses closed-form ridge (Wf-41 P1 fix) rather than the PD
    iteration since (a) the design matrix dim varies across methods
    and (b) the closed-form is faster + stable on small N.
    """
    t0 = time.time()
    w, train_r2 = _stable_ridge_fit(
        X_train,
        y_train,
        ridge_alpha=float(ridge_alpha),
    )
    pred_test = _predict(w, X_test)
    pred_train = _predict(w, X_train)
    wall = time.time() - t0

    def r2(y, p):
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 1e-12:
            return 0.0
        return 1.0 - ss_res / ss_tot

    r2_test = float(r2(y_test, pred_test))
    mem = float(X_train.nbytes + X_test.nbytes) / (1024**2)
    return r2_test, train_r2, wall, mem


def _split_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    test_frac: float = TEST_FRAC,
    seed: int = 0,
    hilbert_dim: int = HILBERT_DIM,
) -> Dict[str, float]:
    """Per-seed split + fit; return metrics dict."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    Xtr = X[train_idx]
    Xte = X[test_idx]
    ytr = y[train_idx]
    yte = y[test_idx]

    # Per-column z-score with std floor 1e-3 (Wf-37 numerical fix).
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    r2_te, r2_tr, wall, mem = _train_hilbert(
        Xtr,
        ytr,
        Xte,
        yte,
        hilbert_dim,
    )
    return {
        "r2_test": float(r2_te),
        "r2_train": float(r2_tr),
        "wall_s": float(wall),
        "peak_mem_mb": float(mem),
        "n_features": int(X.shape[1]),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_valid": int(valid.sum()),
    }


def _aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {
            "r2_mean": 0.0,
            "r2_std": 0.0,
            "wall_s_mean": 0.0,
            "wall_s_std": 0.0,
            "n_seeds": 0,
        }
    r2 = np.array([r["r2_test"] for r in records], dtype=float)
    wall = np.array([r["wall_s"] for r in records], dtype=float)
    return {
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "wall_s_mean": float(np.mean(wall)),
        "wall_s_std": float(np.std(wall)),
        "n_seeds": int(len(records)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-79 P2 3-way + PLL combined R^2 bench",
    )
    parser.add_argument(
        "--dataset",
        default="platinai_4cell",
        choices=["platinai_4cell", "metalcytotox_5metal"],
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--pll-k",
        type=int,
        default=PLL_K,
        help="PLL signature dim (default 16; PROJECTED 128 @ R20-P3)",
    )
    parser.add_argument(
        "--output",
        default="molmetal/reports/wf79_p2_3way_pll",
    )
    parser.add_argument("--data-csv", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    ds_dir = out_root / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load dataset ----
    if args.data_csv is None:
        if args.dataset == "platinai_4cell":
            args.data_csv = str(PROJ_ROOT / "molmetal" / "data" / "platinai_4cell.csv")
        else:
            args.data_csv = str(
                PROJ_ROOT / "molmetal" / "data" / "metalcytotox_5metal_multitask_v4.csv"
            )
    LOGGER.info(
        "Loading %s from %s (n=%d, seed=42)…",
        args.dataset,
        args.data_csv,
        args.n_samples,
    )
    if args.dataset == "platinai_4cell":
        smiles, y = _load_platinai_4cell(
            Path(args.data_csv),
            args.n_samples,
            seed=42,
        )
    else:
        smiles, y = _load_metalcytotox_5metal(
            Path(args.data_csv),
            args.n_samples,
            seed=42,
        )
    LOGGER.info(
        "Loaded %d valid rows (target mean=%.3f, std=%.3f).",
        len(smiles),
        float(np.mean(y)),
        float(np.std(y)),
    )

    # ---- 2. Extract three feature streams ----
    LOGGER.info("Extracting ASTSR 12-d features…")
    t0 = time.time()
    ast = _ast_features(smiles)
    LOGGER.info("ASTSR features shape=%s, wall=%.2fs", ast.shape, time.time() - t0)

    LOGGER.info("Extracting Multi-PH %d-bin features…", MPH_N_BINS)
    t0 = time.time()
    mph, mph_valid = _multiph_features(smiles, n_bins=MPH_N_BINS)
    LOGGER.info(
        "Multi-PH features shape=%s, valid=%d/%d, wall=%.2fs",
        mph.shape,
        int(mph_valid.sum()),
        len(smiles),
        time.time() - t0,
    )

    LOGGER.info("Extracting PLL %d-d features…", args.pll_k)
    t0 = time.time()
    pll, pll_valid = _pll_features(smiles, k=args.pll_k)
    LOGGER.info(
        "PLL features shape=%s, valid=%d/%d, wall=%.2fs",
        pll.shape,
        int(pll_valid.sum()),
        len(smiles),
        time.time() - t0,
    )

    # Combined validity mask.
    combined_valid = mph_valid & pll_valid
    LOGGER.info(
        "Combined valid mask: %d/%d samples",
        int(combined_valid.sum()),
        len(smiles),
    )

    # ---- 3. Build 5 method designs ----
    ast_s = ast[combined_valid]
    mph_s = mph[combined_valid]
    pll_s = pll[combined_valid]
    y_s = y[combined_valid]

    X_pll = pll_s
    X_ast = ast_s
    # 3-way = ASTSR (12) + Multi-PH + PLL (k)
    X_3way = np.concatenate([ast_s, mph_s, pll_s], axis=1)
    # 3way + PLL augmented = 3-way + PLL^2 augmentation.
    # ast_s (12-d) × pll_s (k-d) is broadcast-incompatible; instead
    # we add squared PLL terms and a polynomial-only ASTSR×PLL
    # cross-term after projection to matching dim.
    pll_sq = pll_s**2
    # Match dim by stacking ast_s (broadcast) + pll_s — only the
    # first 12 columns of pll_s so we can cross with ast_s.
    k_cross = min(ast_s.shape[1], pll_s.shape[1])
    ast_pll_cross = ast_s[:, :k_cross] * pll_s[:, :k_cross]
    X_3way_pll_aug = np.concatenate(
        [X_3way, pll_sq, ast_pll_cross],
        axis=1,
    )
    # Joint = ASTSR + Multi-PH (Wf-21 P5 reference)
    X_joint = np.concatenate([ast_s, mph_s], axis=1)

    methods = {
        "pll_only": X_pll,
        "ast_only": X_ast,
        "3way": X_3way,
        "3way_pll_aug": X_3way_pll_aug,
        "joint": X_joint,
    }

    # ---- 4. Per-method × per-seed fit + metrics ----
    per_seed: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    for method_name, X in methods.items():
        LOGGER.info("Method %s: feature_dim=%d", method_name, X.shape[1])
        for seed in args.seeds:
            try:
                rec = _split_and_fit(X, y_s, combined_valid, seed=int(seed))
                rec["seed"] = int(seed)
                per_seed[method_name].append(rec)
                LOGGER.info(
                    "  seed=%d  r2_test=%+.4f  wall=%.3fs",
                    seed,
                    rec["r2_test"],
                    rec["wall_s"],
                )
            except Exception as exc:
                LOGGER.warning("  seed=%d failed: %s", seed, exc)

    # ---- 5. Aggregate + lift over baselines ----
    summary: Dict[str, Dict[str, float]] = {}
    for method_name, records in per_seed.items():
        summary[method_name] = _aggregate(records)

    # Combined lift = 3way_pll_aug.r2_mean - 3way.r2_mean.
    if "3way" in summary and "3way_pll_aug" in summary:
        summary["delta_3way_pll_aug_vs_3way"] = {
            "r2_lift": float(
                summary["3way_pll_aug"]["r2_mean"] - summary["3way"]["r2_mean"]
            ),
            "r2_lift_pct": float(
                100.0
                * (summary["3way_pll_aug"]["r2_mean"] - summary["3way"]["r2_mean"])
                / max(1e-6, abs(summary["3way"]["r2_mean"]))
            ),
        }
    # 3way vs pll_only lift.
    if "3way" in summary and "pll_only" in summary:
        summary["delta_3way_vs_pll_only"] = {
            "r2_lift": float(
                summary["3way"]["r2_mean"] - summary["pll_only"]["r2_mean"]
            ),
        }

    # ---- 6. Persist outputs ----
    per_seed_path = ds_dir / "per_seed_metrics.json"
    with open(per_seed_path, "w") as fh:
        json.dump(
            {
                "config": {
                    "dataset": args.dataset,
                    "seeds": list(args.seeds),
                    "n_samples": int(args.n_samples),
                    "n_loaded": int(len(smiles)),
                    "n_valid": int(combined_valid.sum()),
                    "pll_k": int(args.pll_k),
                    "mph_n_bins": int(MPH_N_BINS),
                    "hilbert_dim": int(HILBERT_DIM),
                    "test_frac": float(TEST_FRAC),
                },
                "per_method": per_seed,
                "summary": summary,
            },
            fh,
            indent=2,
        )
    LOGGER.info("Wrote per_seed_metrics.json → %s", per_seed_path)

    summary_path = ds_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    LOGGER.info("Wrote summary.json → %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
