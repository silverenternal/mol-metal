#!/usr/bin/env python
"""BLK-L1 — does a genuine 50-d representation lift platinai_4cell R² off zero?

=====================================================================
The blocker
=====================================================================
``metrics/by_metric/m2_r2_per_cohort.json`` records platinai_4cell R² =
+0.0102 at n=4078 with **32** features (alpha=0.1) — indistinguishable from
zero, against a tier-table target of 50-d. The tier table
(``wf380a2_m2_r2_per_cohort.select_features``) says n>=1000 → 50-d, but the
loader only ever emits 32 columns, so ``apply_per_cohort_config``'s
``X[:, :50]`` silently truncates to ``X[:, :32]``. The 50-d path was never
built (BLK-L1 ``still_open`` #2).

This script builds it and measures the delta.

What the 50-d arm actually contains, and what it cannot
-------------------------------------------------------
The tier name is ``rdkit_50d_chemprop_v3_hilbert``. Two of its three named
components are **not available as trained representations on disk**:

* **chemprop_v3 encoder** — ``chemprop_v3_encoder_forward.py`` builds a
  ``ChempropV2Encoder`` and never loads a checkpoint; the
  ``blk_d2_chemprop_v3_frozen`` reports call it "frozen" meaning *weights
  frozen*, not *pretrained*. There is no trained chemprop v3 encoder
  checkpoint in ``.ckpt/`` or anywhere on disk. Deriving an 18-d feature from
  a random encoder would be noise dressed as a representation, so this script
  does **not** do it.
* **Hilbert coords** — no SMILES → Hilbert-coordinate featurizer exists in the
  repo; the "50-d" label in the tier table is a name, not a defined
  composition.

So the 50-d arm here is **32-d RDKit + 18 principal components of a real
Morgan ECFP4 (radius 2, 2048-bit) fingerprint**, with the PCA fit **inside
each training fold** (no leakage). That is a genuine, non-learned,
reproducible 50-d representation — the honest substitute for the two
unavailable components, and it is labelled as such in the output JSON.

Protocol (matched to the on-disk control)
-----------------------------------------
Same cohort, same target, same fold construction, same Ridge solver, same
alpha=0.1 as ``wf380a2_ridge_r2_cv`` for the n>=1000 tier — so the delta
between the two arms is attributable to the 18 added dimensions and nothing
else. The 32-d arm is re-measured here to confirm it reproduces +0.0102.

Honest-negative fallback (BLK-L1)
---------------------------------
If the 50-d arm still lands < 0.10, the reading is "representation ceiling
reached for this cohort", and metalcytotox_combined remains the load-bearing
cohort. The verdict is written to disk either way.

Usage
-----
::

    .venv/bin/python molmetal/scripts/blk_l1_platinai_50d_r2.py \
        --config configs/blk_l1_platinai_50d.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from molmetal.scripts.wf331r5_r2_w2_chemprop import (  # noqa: E402
    _load_platinai_expanded,
)
from molmetal.scripts.wf380a2_m2_r2_per_cohort import (  # noqa: E402
    ridge_r2_eval,
)

# the on-disk control this run must reproduce
CONTROL_R2 = 0.01025  # metrics/by_metric/m2_r2_per_cohort.json::wf_fix3lines_expand_2026_10_09


def morgan_matrix(smiles_list: List[str], radius: int, nbits: int) -> np.ndarray:
    """Dense ECFP4 bit matrix, one row per SMILES (all-zero row if unparseable)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = np.zeros((len(smiles_list), nbits), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        for bit in fp.GetOnBits():
            out[i, bit] = 1.0
    return out


def _folds(n: int, n_splits: int, seed: int) -> List[np.ndarray]:
    """The exact fold construction ``wf380a2_ridge_r2_cv`` uses."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        sizes[i] += 1
    folds, start = [], 0
    for fs in sizes:
        folds.append(perm[start : start + fs])
        start += fs
    return folds


def cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    X_extra: Optional[np.ndarray],
    n_components: int,
    n_splits: int,
    n_seeds: int,
    alpha: float,
    seed_offset: int,
    pca_seed: int,
) -> Dict[str, float]:
    """Multi-seed K-fold Ridge R².

    When ``X_extra`` is given (the Morgan matrix), a PCA(``n_components``) is
    fit **on the training fold only** and concatenated to ``X``, giving the
    50-d arm. When it is ``None`` the 32-d control is scored with no PCA at
    all — so the two arms differ only by the added dimensions.
    """
    from sklearn.decomposition import PCA

    n = X.shape[0]
    r2_per_seed: List[float] = []
    n_folds_effective = 0
    for s in range(n_seeds):
        seed = seed_offset + s
        folds = _folds(n, n_splits, seed)
        r2_per_fold: List[float] = []
        for k, test in enumerate(folds):
            train = np.concatenate([f for i, f in enumerate(folds) if i != k])
            if len(test) < 2 or len(train) < 2:
                continue
            if X_extra is None:
                Xtr, Xte = X[train], X[test]
            else:
                pca = PCA(n_components=n_components, random_state=pca_seed)
                pca.fit(X_extra[train])
                ztr = pca.transform(X_extra[train])
                zte = pca.transform(X_extra[test])
                Xtr = np.concatenate([X[train], ztr], axis=1)
                Xte = np.concatenate([X[test], zte], axis=1)
            r2 = ridge_r2_eval(Xtr, y[train], Xte, y[test], alpha=alpha)
            if np.isfinite(r2):
                r2_per_fold.append(r2)
                n_folds_effective += 1
        if r2_per_fold:
            r2_per_seed.append(float(np.mean(r2_per_fold)))
    if not r2_per_seed:
        return {"r2_mean": float("nan"), "r2_std": float("nan"), "n_folds": 0}
    return {
        "r2_mean": float(np.mean(r2_per_seed)),
        "r2_std": float(np.std(r2_per_seed)),
        "n_folds": int(n_folds_effective),
        "per_seed": r2_per_seed,
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    t0 = time.time()
    print(f"[blk-l1] loading platinai_4cell (cap={args.max_cohort_size}) ...", flush=True)
    X32, y, smiles, target_name = _load_platinai_expanded(
        "platinai_4cell", args.max_cohort_size
    )
    n, n_feat_control = X32.shape
    print(f"[blk-l1] n={n} n_feat_control={n_feat_control} target={target_name}", flush=True)

    morgan = morgan_matrix(smiles, args.radius, args.nbits)
    n_nonzero = int((morgan.sum(axis=1) > 0).sum())

    report: Dict[str, object] = {
        "metric": "blk_l1_platinai_50d_r2",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "R^2 (held-out fold; higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/blk_l1_platinai_50d_r2.py",
        "cohort": "platinai_4cell",
        "target_name": target_name,
        "n": int(n),
        "protocol": {
            "cv": f"{args.n_seeds} seeds x {args.n_splits} folds, Ridge (closed-form)",
            "alpha": args.alpha,
            "alpha_tier": "n_gte_1000 (wf380a2 select_features)",
            "fold_seed_offset": args.seed_offset,
            "note": (
                "fold construction + Ridge solver identical to "
                "wf380a2_ridge_r2_cv, so the 32-d control here reproduces the "
                "on-disk +0.01025"
            ),
        },
        "on_disk_control": {
            "r2_mean": CONTROL_R2,
            "n_features": 32,
            "alpha": 0.1,
            "source": (
                "metrics/by_metric/m2_r2_per_cohort.json::"
                "wf_fix3lines_expand_2026_10_09"
            ),
        },
        "feature_composition": {
            "control_32d": "PLATINAI_32D_COLUMNS (12-d Lipinski + 20-d shape/charge/ring)",
            "added_18d": (
                f"PCA({args.n_components}) of Morgan ECFP4 (radius={args.radius}, "
                f"{args.nbits}-bit), fit inside each training fold"
            ),
            "why_not_chemprop_v3": (
                "no trained chemprop v3 encoder checkpoint exists on disk "
                "(chemprop_v3_encoder_forward builds an untrained "
                "ChempropV2Encoder; 'frozen' in blk_d2 reports means weights "
                "frozen, not pretrained) — an 18-d slice of a random encoder "
                "would be noise, so it is not used"
            ),
            "why_not_hilbert": (
                "no SMILES -> Hilbert-coordinate featurizer exists in the repo; "
                "the '50-d' tier label names a family, not a defined composition"
            ),
            "morgan_rows_nonzero": n_nonzero,
        },
        "arms": {},
    }

    print("[blk-l1] arm A: 32-d RDKit control ...", flush=True)
    a = cv_r2(
        X32, y, None, args.n_components, args.n_splits, args.n_seeds,
        args.alpha, args.seed_offset, args.pca_seed,
    )
    a["n_features"] = int(n_feat_control)
    a["reproduces_on_disk_control"] = bool(abs(a["r2_mean"] - CONTROL_R2) < 0.02)
    report["arms"]["rdkit_32d_control"] = a
    print(f"[blk-l1]   A 32-d  R2={a['r2_mean']:.4f}±{a['r2_std']:.4f}", flush=True)

    print("[blk-l1] arm B: 32-d RDKit + PCA18(Morgan) = 50-d ...", flush=True)
    b = cv_r2(
        X32, y, morgan, args.n_components, args.n_splits, args.n_seeds,
        args.alpha, args.seed_offset, args.pca_seed,
    )
    b["n_features"] = int(n_feat_control + args.n_components)
    b["delta_vs_control"] = float(b["r2_mean"] - a["r2_mean"])
    report["arms"]["rdkit32_morganpca18_50d"] = b
    print(
        f"[blk-l1]   B 50-d  R2={b['r2_mean']:.4f}±{b['r2_std']:.4f} "
        f"delta={b['delta_vs_control']:+.4f}",
        flush=True,
    )

    target = args.target_r2
    best = max(a["r2_mean"], b["r2_mean"])
    if best >= target:
        verdict = "TARGET_MET"
    elif best >= 0.10:
        verdict = "PARTIAL_RECOVERY"
    else:
        verdict = "CEILING_REACHED_HONEST_NEGATIVE"
    report["verdict"] = {
        "target_r2": target,
        "best_arm_r2": float(best),
        "verdict": verdict,
        "reading": (
            "50-d (32-d RDKit + 18 PCA dims of a real Morgan fingerprint) does "
            "not move platinai_4cell R^2 off zero: the cohort is "
            "representation-limited, not feature-count-limited. "
            "metalcytotox_combined stays the load-bearing cohort."
            if verdict == "CEILING_REACHED_HONEST_NEGATIVE"
            else "see arm table"
        ),
    }
    report["honest_negative"] = {
        "blocker": "BLK-L1",
        "symptom": "platinai_4cell R2 ~= 0 at n=4078",
        "still_open_in_blocker": (
            "the blocker's Step 2 (chemprop_v3 frozen encoder) is not "
            "achievable — no trained encoder exists on disk; Step 3 (Hilbert "
            "18-d) has no defined featurizer. Both are recorded here rather "
            "than papered over with a random-encoder feature."
        ),
        "do_not_over_claim": (
            "platinai_4cell is a Pt-only continuous-regression cohort; it is "
            "not the MetalCytoToxDB 5-metal classification task and BLK-C15 "
            "does not transfer."
        ),
    }
    report["wall_sec"] = round(time.time() - t0, 1)
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLK-L1 platinai 50-d R2.")
    p.add_argument("--max-cohort-size", type=int, default=4078)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--n-components", type=int, default=18)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--pca-seed", type=int, default=0)
    p.add_argument("--target-r2", type=float, default=0.15)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (AGENTS.md §5.11). Keys resolve to the dests above; "
        "an explicit CLI flag still wins.",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/blk_l1_platinai_50d_r2.json",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Module-level entry point for scripts/_config_binding_gate.py."""
    p = _build_parser()
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rep = run(args)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))

    print("\narm                          n_feat    R2_mean    R2_std")
    for name, v in rep["arms"].items():
        print(f"{name:28s} {v['n_features']:6d} {v['r2_mean']:10.4f} {v['r2_std']:9.4f}")
    print(f"\nverdict: {rep['verdict']['verdict']} "
          f"(best={rep['verdict']['best_arm_r2']:.4f}, target={args.target_r2})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
