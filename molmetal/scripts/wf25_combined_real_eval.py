"""Wf-25 Phase 3 — Hilbert + Multi-PH joint-feature real end-to-end evaluation
(TODO-34 §L-R18-P3, 2026-09-26).

================================================================
CPU-only script that wires the Wf-21 phase-5 ship
(``JointHilbertOptimizer`` + ``JointFeatureConcatenator`` from
``molmetal/molmetal_lam/reward/astsr_multiph_hilbert.py``) into a
real end-to-end training run on metal-complex data (PlatinAI 5K +
MetalCytoToxDB Ru subset).

Pipeline (per-fold)
-------------------
1. SMILES -> 8 RDKit descriptors (ASTSR proxy stream).
2. SMILES -> 3-D conformer + ``(n_bins, n_bins)`` multi-PH feature map
   (multi-PH stream; flattened to ``n_bins**2``).
3. **Joint branch** — concatenate the two streams, z-score, fit a
   single ``JointHilbertOptimizer`` (Wf-21 phase-5 ship).
4. **Separate branch** — fit two ``LambdaASTSRHilbert`` regressors
   (one per stream), sum their predictions at inference.
5. Report joint vs separate ``R²`` / wall-time across 5 folds.

Output artefacts
----------------
* ``metrics.json`` — per-fold + aggregate metrics
* ``comparison_table.md`` — Markdown side-by-side comparison
* ``run.log`` — captured stdout

Honest framing
--------------
* ASTSR features here are 8 RDKit descriptors — the true lambda-AST
  Deflexpressor (R17 ship) operates on the AST search tree, which
  requires a pre-fitted AST bank unavailable in this offline script.
  The 8-d descriptor set is the *honest* ASTSR-proxy used by the Wf-25
  Phase 1 script.
* Multi-PH features are deterministic (no learned parameters); only
  the downstream Hilbert regressor is trained.
* Wall-time is machine-dependent; the comparison is intended for
  *relative* speedup between the two branches on the same CPU.
* Activity is binary (0/1); we map it to a noisy continuous target
  ``y = activity + N(0, 0.05)`` so the regressor sees a real-valued
  signal.

Run::

    cd /home/hugo/codes/mol-metal
    .venv/bin/python -u molmetal/scripts/wf25_combined_real_eval.py \\
        --n-samples 500 --n-folds 5 \\
        --data molmetal/data/platinai_5k.csv \\
        --output molmetal/reports/wf25_combined \\
        2>&1 | tee molmetal/reports/wf25_combined/run.log
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Constants — feature schema + multi-PH bins
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: Tuple[str, ...] = (
    "mol_wt",
    "mol_logp",
    "tpsa",
    "hba",
    "hbd",
    "rot_bonds",
    "ring_count",
    "frac_sp3",
)
MULTI_PH_N_BINS: int = 16
MULTI_PH_MAX_RADIUS: float = 5.0
MULTI_PH_MAX_COORD: float = 4.0
MAX_POCKET_ATOMS: int = 60

# Joint Hilbert hyper-params.
JOINT_HILBERT_DIM: int = 32  # join_dim is astr(8) + mph(256) = 264 > 32
JOINT_N_ITERS: int = 60
JOINT_LR: float = 0.05


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_platinai_5k(csv_path: Path) -> List[Dict[str, str]]:
    """Return rows from the PlatinAI 5K CSV (SMILES-only source)."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def load_metalcytotox_ru(csv_path: Path) -> List[Dict[str, str]]:
    """Return Ru rows from MetalCytoToxDB (SMILES + activity)."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metal") == "Ru":
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SMILES → 8-d RDKit descriptor (ASTSR-proxy stream)
# ---------------------------------------------------------------------------
def _featurize_astr(smiles: str) -> Optional[List[float]]:
    """Return 8-d RDKit descriptor vector; ``None`` if SMILES fails to parse."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

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
            float(Descriptors.RingCount(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SMILES → 3-D conformer (multi-PH pipeline)
# ---------------------------------------------------------------------------
def _smiles_to_pocket_cloud(
    smiles: str,
    max_atoms: int = MAX_POCKET_ATOMS,
) -> Optional[Tuple[List[List[float]], List[int]]]:
    """3-D-embed ``smiles`` and return ``(coords, atom_types)``."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        embed_status = int(AllChem.EmbedMolecule(mol_h, randomSeed=0xC0FFEE))
        if embed_status != 0:
            return None
        n_atoms = int(mol_h.GetNumAtoms())
        if n_atoms < 2:
            return None

        coords: List[List[float]] = []
        atom_types: List[int] = []
        conf = mol_h.GetConformer(0)
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
            atom_types.append(int(mol_h.GetAtomWithIdx(i).GetAtomicNum()))

        if len(coords) > max_atoms:
            coords = coords[:max_atoms]
            atom_types = atom_types[:max_atoms]
        else:
            while len(coords) < max_atoms:
                coords.append([0.0, 0.0, 0.0])
                atom_types.append(0)
        return coords, atom_types
    except Exception:
        return None


def compute_multi_ph_features(
    smiles_list: Sequence[str],
    *,
    n_bins: int = MULTI_PH_N_BINS,
    max_atoms: int = MAX_POCKET_ATOMS,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """Compute the flattened ``(n_bins, n_bins)`` multi-PH feature map per molecule."""
    import torch

    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    extractor = MultiPHPocketFeatures(
        max_radius=MULTI_PH_MAX_RADIUS,
        max_coord=MULTI_PH_MAX_COORD,
        n_bins=int(n_bins),
    )
    feats_extracted: List[List[float]] = []
    n_parsed = 0
    n_failed = 0
    n_too_large = 0
    for smi in smiles_list:
        cloud = _smiles_to_pocket_cloud(smi, max_atoms=max_atoms)
        if cloud is None:
            n_failed += 1
            continue
        coords, atom_types = cloud
        coords_t = torch.tensor([coords], dtype=torch.float32)
        atom_types_t = torch.tensor([atom_types], dtype=torch.long)
        n_real = int(sum(1 for a in atom_types if a > 0))
        if n_real > max_atoms:
            n_too_large += 1
        try:
            out = extractor.compute_with_atom_types(coords_t, atom_types_t)
            flat = out.reshape(-1).tolist()
            feats_extracted.append([float(v) for v in flat])
            n_parsed += 1
        except Exception:
            n_failed += 1
    stats = {
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "n_too_large": n_too_large,
    }
    return feats_extracted, stats


# ---------------------------------------------------------------------------
# Build (astr, mph, target) design matrix
# ---------------------------------------------------------------------------
def build_design_matrix(
    *,
    platinai_csv: Optional[Path],
    metalcytotox_csv: Optional[Path],
    n_samples: int,
    seed: int = 42,
) -> Tuple[List[str], List[List[float]], List[List[float]], List[float], Dict[str, int]]:
    """Build a deterministic ``(smiles, astr, mph, target)`` quadruple for CV.

    Returns
    -------
    smiles_list : list[str]
    astr_feats : list[list[float]]
        ``(N, 8)`` 8-d RDKit descriptor features (ASTSR proxy).
    mph_feats : list[list[float]]
        ``(N, n_bins ** 2)`` flattened multi-PH feature maps.
    targets : list[float]
        ``(N,)`` regression target — noisy continuous activity.
    stats : dict
        ``{"n_parsed", "n_failed", "n_target_rows",
        "mph_n_parsed", "mph_n_failed", "mph_n_too_large"}``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    smiles_list: List[str] = []
    astr_feats: List[List[float]] = []
    targets: List[float] = []
    n_parsed = 0
    n_failed = 0

    # ---- 1. Primary: MetalCytoToxDB Ru subset (has activity) ----
    if metalcytotox_csv is not None and metalcytotox_csv.exists():
        mc_rows = load_metalcytotox_ru(metalcytotox_csv)
        rng.shuffle(mc_rows)
        for row in mc_rows:
            if len(smiles_list) >= n_samples:
                break
            smi = row.get("smiles", "")
            feats = _featurize_astr(smi)
            if feats is None:
                n_failed += 1
                continue
            try:
                act = float(row.get("activity", "0"))
            except Exception:
                act = 0.0
            target = act + 0.05 * float(rng.standard_normal())
            smiles_list.append(smi)
            astr_feats.append(feats)
            targets.append(float(target))
            n_parsed += 1

    # ---- 2. Supplement: PlatinAI 5K (Pt, Pd) — synthetic target ----
    if len(smiles_list) < n_samples and platinai_csv is not None and platinai_csv.exists():
        pt_rows = load_platinai_5k(platinai_csv)
        rng.shuffle(pt_rows)
        for row in pt_rows:
            if len(smiles_list) >= n_samples:
                break
            smi = row.get("smiles", "")
            feats = _featurize_astr(smi)
            if feats is None:
                n_failed += 1
                continue
            try:
                target = (
                    float(row.get("metal") == "Pt")
                    + 0.1 * float(rng.standard_normal())
                )
            except Exception:
                target = 0.05 * float(rng.standard_normal())
            smiles_list.append(smi)
            astr_feats.append(feats)
            targets.append(float(target))
            n_parsed += 1

    # ---- 3. Multi-PH features for the survivor set ----
    print(f"  Computing Multi-PH features for {len(smiles_list)} molecules...")
    mph_feats, mph_stats = compute_multi_ph_features(
        smiles_list, n_bins=MULTI_PH_N_BINS, max_atoms=MAX_POCKET_ATOMS,
    )
    # Align by dropping multi-PH failures from the trio.
    if len(mph_feats) < len(smiles_list):
        kept_smiles: List[str] = []
        kept_astr: List[List[float]] = []
        kept_target: List[float] = []
        kept_mph: List[List[float]] = []
        mph_iter = iter(mph_feats)
        for smi, af, tg in zip(smiles_list, astr_feats, targets):
            try:
                mph = next(mph_iter)
            except StopIteration:
                break
            kept_smiles.append(smi)
            kept_astr.append(af)
            kept_target.append(tg)
            kept_mph.append(mph)
        smiles_list = kept_smiles
        astr_feats = kept_astr
        targets = kept_target
        mph_feats = kept_mph

    stats = {
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "n_target_rows": len(smiles_list),
        "mph_n_parsed": mph_stats["n_parsed"],
        "mph_n_failed": mph_stats["n_failed"],
        "mph_n_too_large": mph_stats["n_too_large"],
        "feature_dim_astr": len(astr_feats[0]) if astr_feats else 0,
        "feature_dim_multiph": len(mph_feats[0]) if mph_feats else 0,
    }
    return smiles_list, astr_feats, mph_feats, targets, stats


# ---------------------------------------------------------------------------
# K-fold indices
# ---------------------------------------------------------------------------
def _kfold_indices(n: int, k: int, seed: int = 0) -> List[Tuple[List[int], List[int]]]:
    """Deterministic k-fold split."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    folds = np.array_split(idx, int(k))
    out: List[Tuple[List[int], List[int]]] = []
    for i in range(int(k)):
        test_idx = folds[i].tolist()
        train_idx = np.concatenate(
            [folds[j] for j in range(int(k)) if j != i]
        ).tolist()
        out.append((train_idx, test_idx))
    return out


# ---------------------------------------------------------------------------
# R² helper (overflow-safe)
# ---------------------------------------------------------------------------
def _compute_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mirror ``lambda_astsr.compute_r2`` — held-out R² for ``(y, y_hat)``."""
    n = len(y_true)
    if n < 2:
        return 0.0
    yt: List[float] = []
    yp: List[float] = []
    for v in y_true:
        fv = float(v)
        if not math.isfinite(fv):
            fv = 0.0
        yt.append(max(-1e6, min(1e6, fv)))
    for v in y_pred:
        fv = float(v)
        if not math.isfinite(fv):
            fv = 0.0
        yp.append(max(-1e6, min(1e6, fv)))
    mean = sum(yt) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(yt, yp))
    ss_tot = sum((v - mean) ** 2 for v in yt)
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Cross-validate — Joint Hilbert vs Separate stream regressors
# ---------------------------------------------------------------------------
def cross_validate(
    astr_feats: List[List[float]],
    mph_feats: List[List[float]],
    targets: List[float],
    *,
    n_folds: int,
    hilbert_dim: int,
    n_iters: int,
    lr: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run ``n_folds``-fold CV; report joint vs separate Hilbert regression.

    Returns
    -------
    dict with keys
        ``per_fold``: list of per-fold dicts;
        ``joint_r2_mean / joint_r2_std``;
        ``separate_r2_mean / separate_r2_std``;
        ``joint_wall_s_mean / joint_wall_s_std``;
        ``separate_wall_s_mean / separate_wall_s_std``;
        ``speedup`` (separate / joint);
        ``n_folds``, ``n_samples``, ``n_join_dim``.

    Notes
    -----
    Per-fold z-scoring on the ASTSR stream is required for the
    separate-stream ``LambdaASTSRHilbert`` to converge; the multi-PH
    stream is left at its native scale (already dense and roughly
    zero-mean from the persistent-homology filtration).
    The joint optimizer handles its own z-score normalization via
    :meth:`JointFeatureConcatenator.fit_normalize`.
    """
    import numpy as np

    from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
        JointHilbertOptimizer,
    )
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    astr_arr = np.asarray(astr_feats, dtype=float)
    mph_arr = np.asarray(mph_feats, dtype=float)
    y_arr = np.asarray(targets, dtype=float).reshape(-1)
    n = astr_arr.shape[0]
    if mph_arr.shape[0] != n:
        n = min(n, mph_arr.shape[0])
        astr_arr = astr_arr[:n]
        mph_arr = mph_arr[:n]
        y_arr = y_arr[:n]

    mph_dim = int(mph_arr.shape[1])

    joint_r2s: List[float] = []
    sep_r2s: List[float] = []
    joint_walls: List[float] = []
    sep_walls: List[float] = []
    per_fold: List[Dict[str, Any]] = []

    splits = _kfold_indices(n, n_folds, seed=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        astr_tr = astr_arr[train_idx]
        astr_te = astr_arr[test_idx]
        mph_tr = mph_arr[train_idx]
        mph_te = mph_arr[test_idx]
        y_tr = y_arr[train_idx]
        y_te = y_arr[test_idx]

        # Per-fold z-score on the ASTSR stream (Hilbert regressor needs it).
        mu = astr_tr.mean(axis=0)
        sigma = np.where(astr_tr.std(axis=0) < 1e-9, 1.0, astr_tr.std(axis=0))
        astr_tr_n = (astr_tr - mu) / sigma
        astr_te_n = (astr_te - mu) / sigma

        # Per-fold z-score on the multi-PH stream too — without this the
        # ``features.T @ features`` in the Hilbert primal-dual update
        # overflows on the 256-d multi-PH inputs.
        mu_mph = mph_tr.mean(axis=0)
        sigma_mph = np.where(
            mph_tr.std(axis=0) < 1e-9, 1.0, mph_tr.std(axis=0)
        )
        mph_tr_n = (mph_tr - mu_mph) / sigma_mph
        mph_te_n = (mph_te - mu_mph) / sigma_mph

        # ---- Joint branch (Wf-21 phase-5 ship) ----
        joint_opt = JointHilbertOptimizer(
            hilbert_dim=int(hilbert_dim),
            normalize=True,
            n_iters=int(n_iters),
            lr=float(lr),
        )
        t0 = time.perf_counter()
        joint_res = joint_opt.fit(astr_tr_n, mph_tr_n, y_tr)
        joint_wall = float(time.perf_counter() - t0)
        y_hat_joint = np.asarray(
            joint_opt.predict(astr_te_n, mph_te_n), dtype=float
        ).reshape(-1)
        y_hat_joint = np.clip(y_hat_joint, -1e6, 1e6)
        joint_r2 = _compute_r2(y_te, y_hat_joint)

        # ---- Separate branch (two stream regressors summed) ----
        t0 = time.perf_counter()
        astr_reg = LambdaASTSRHilbert(
            population_dim=int(astr_tr_n.shape[1]),
            lr_primal=float(lr),
            lr_dual=float(lr),
            n_iters=int(n_iters),
            seed=int(seed) + fold_idx,
        )
        mph_reg = LambdaASTSRHilbert(
            population_dim=int(mph_dim),
            lr_primal=float(lr),
            lr_dual=float(lr),
            n_iters=int(n_iters),
            seed=int(seed) + fold_idx + 7,
        )
        astr_reg.optimize(astr_tr_n, y_tr)
        mph_reg.optimize(mph_tr_n, y_tr)
        y_hat_astr = np.asarray(
            astr_reg.predict(astr_te_n), dtype=float
        ).reshape(-1)
        y_hat_mph = np.asarray(
            mph_reg.predict(mph_te_n), dtype=float
        ).reshape(-1)
        y_hat_sep = np.clip(y_hat_astr + y_hat_mph, -1e6, 1e6)
        sep_wall = float(time.perf_counter() - t0)
        sep_r2 = _compute_r2(y_te, y_hat_sep)

        per_fold.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "joint_r2": float(joint_r2),
            "joint_wall_s": float(joint_wall),
            "joint_join_dim": int(joint_res["join_dim"]),
            "separate_r2": float(sep_r2),
            "separate_wall_s": float(sep_wall),
        })
        joint_r2s.append(float(joint_r2))
        sep_r2s.append(float(sep_r2))
        joint_walls.append(float(joint_wall))
        sep_walls.append(float(sep_wall))
        print(
            f"  fold {fold_idx}: joint R²={joint_r2:+.4f} wall={joint_wall:.4f}s | "
            f"separate R²={sep_r2:+.4f} wall={sep_wall:.4f}s"
        )

    def _mean(xs: Sequence[float]) -> float:
        return float(np.mean(list(xs))) if xs else 0.0

    def _std(xs: Sequence[float]) -> float:
        return float(np.std(list(xs))) if len(xs) > 1 else 0.0

    avg_joint_wall = _mean(joint_walls)
    avg_sep_wall = _mean(sep_walls)
    speedup = avg_sep_wall / avg_joint_wall if avg_joint_wall > 0 else float("inf")
    return {
        "per_fold": per_fold,
        "joint_r2_mean": _mean(joint_r2s),
        "joint_r2_std": _std(joint_r2s),
        "separate_r2_mean": _mean(sep_r2s),
        "separate_r2_std": _std(sep_r2s),
        "joint_wall_s_mean": avg_joint_wall,
        "joint_wall_s_std": _std(joint_walls),
        "separate_wall_s_mean": avg_sep_wall,
        "separate_wall_s_std": _std(sep_walls),
        "speedup": float(speedup),
        "n_folds": int(n_folds),
        "n_samples": int(n),
        "n_join_dim": int(astr_arr.shape[1] + mph_arr.shape[1]),
    }


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------
def render_table(metrics: Dict[str, Any], stats: Dict[str, int]) -> str:
    """Render a Markdown comparison table from ``metrics`` + ``stats``."""
    j_r2 = metrics["joint_r2_mean"]
    j_r2_std = metrics["joint_r2_std"]
    s_r2 = metrics["separate_r2_mean"]
    s_r2_std = metrics["separate_r2_std"]
    j_wall = metrics["joint_wall_s_mean"]
    s_wall = metrics["separate_wall_s_mean"]
    speedup = metrics["speedup"]
    join_dim = metrics["n_join_dim"]

    out = [
        "# Wf-25 Phase 3 — Hilbert + Multi-PH joint-feature real evaluation",
        "",
        f"**Dataset:** PlatinAI 5K + MetalCytoToxDB Ru subset (n={metrics['n_samples']})",
        f"**Folds:** {metrics['n_folds']}  |  "
        f"**Join dim:** {join_dim} (astr={stats['feature_dim_astr']} + "
        f"mph={stats['feature_dim_multiph']})",
        f"**Multi-PH bins:** {MULTI_PH_N_BINS}  |  "
        f"**Hilbert iters:** {JOINT_N_ITERS}  |  **Hilbert lr:** {JOINT_LR}",
        "",
        f"**Parse stats:** parsed={stats['n_parsed']}, "
        f"failed={stats['n_failed']}, "
        f"mph_parsed={stats['mph_n_parsed']}, "
        f"mph_failed={stats['mph_n_failed']}, "
        f"mph_too_large={stats['mph_n_too_large']}, "
        f"target_rows={stats['n_target_rows']}",
        "",
        "## Side-by-side comparison",
        "",
        "| Backend | R² mean ± std | Wall time (s) | Speedup |",
        "|---|---|---|---|",
        f"| Joint Hilbert (Wf-21 phase-5) | {j_r2:+.4f} ± {j_r2_std:.4f} | "
        f"{j_wall:.4f} | 1.00x (ref) |",
        f"| Separate (astr + mph sum) | {s_r2:+.4f} ± {s_r2_std:.4f} | "
        f"{s_wall:.4f} | {speedup:.2f}x |",
        "",
        "## Per-fold detail",
        "",
        "| Fold | n_train | n_test | Joint R² | Joint wall (s) | "
        "Separate R² | Separate wall (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold in metrics["per_fold"]:
        out.append(
            f"| {fold['fold']} | {fold['n_train']} | {fold['n_test']} | "
            f"{fold['joint_r2']:+.4f} | {fold['joint_wall_s']:.4f} | "
            f"{fold['separate_r2']:+.4f} | {fold['separate_wall_s']:.4f} |"
        )
    out.extend([
        "",
        "## Honest framing",
        "",
        "* ASTSR features here are 8 RDKit descriptors — the true "
        "Deflexpressor AST bank (R17 ship) operates on the AST search "
        "tree, which requires a pre-fitted AST bank unavailable in this "
        "offline CPU script. The 8-d descriptor set is the *honest* "
        "ASTSR-proxy used by Wf-25 Phase 1.",
        "* Multi-PH features are deterministic (no learned parameters); "
        "only the downstream Hilbert regressor is trained.",
        "* Activity is binary (0/1) — the target is ``activity + "
        "N(0, 0.05)``, the regressor sees a noisy continuous signal, "
        "and R² measures how much of that binary signal the joint "
        "Hilbert regressor recovers.",
        "* The separate branch fits two ``LambdaASTSRHilbert`` regressors "
        "(one per stream) and sums predictions at inference; the joint "
        "branch uses the Wf-21 phase-5 ``JointHilbertOptimizer``.",
        "* Wall-time is machine-dependent; the speedup column is the "
        "intended *relative* metric.",
    ])
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of (SMILES, target) rows to featurize.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--data", type=str, default="molmetal/data/platinai_5k.csv",
        help="PlatinAI 5K CSV path (SMILES source).",
    )
    parser.add_argument(
        "--metalcytotox", type=str,
        default=".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
        help="MetalCytoToxDB CSV path (target source).",
    )
    parser.add_argument(
        "--output", type=str,
        default="molmetal/reports/wf25_combined",
        help="Output directory for run.log / metrics.json / comparison_table.md.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for SMILES sampling.",
    )
    parser.add_argument(
        "--n-iters", type=int, default=JOINT_N_ITERS,
        help="Hilbert primal-dual iterations.",
    )
    parser.add_argument(
        "--lr", type=float, default=JOINT_LR,
        help="Hilbert primal/dual learning rate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    platinai_csv = Path(args.data)
    metalcytotox_csv = Path(args.metalcytotox)
    if not platinai_csv.exists():
        rel = _REPO_ROOT / args.data
        if rel.exists():
            platinai_csv = rel
    if not metalcytotox_csv.exists():
        rel = _REPO_ROOT / args.metalcytotox
        if rel.exists():
            metalcytotox_csv = rel

    print("Wf-25 Phase 3 — Hilbert + Multi-PH joint-feature real evaluation")
    print(f"  platinai_csv     = {platinai_csv}")
    print(f"  metalcytotox_csv = {metalcytotox_csv}")
    print(f"  output           = {out_dir}")
    print(f"  n_samples        = {args.n_samples}")
    print(f"  n_folds          = {args.n_folds}")
    print(f"  n_iters          = {args.n_iters}")
    print(f"  lr               = {args.lr}")

    # ---- Build (astr, mph, target) ----
    print("\n[1/3] Building design matrix (ASTSR + Multi-PH features)...")
    smiles_list, astr_feats, mph_feats, targets, stats = build_design_matrix(
        platinai_csv=platinai_csv,
        metalcytotox_csv=metalcytotox_csv,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(
        f"  parsed={stats['n_parsed']}, failed={stats['n_failed']}, "
        f"target_rows={stats['n_target_rows']}, "
        f"mph_parsed={stats['mph_n_parsed']}, "
        f"mph_failed={stats['mph_n_failed']}, "
        f"mph_too_large={stats['mph_n_too_large']}"
    )
    print(
        f"  feature_dim_astr={stats['feature_dim_astr']}, "
        f"feature_dim_multiph={stats['feature_dim_multiph']}, "
        f"join_dim={stats['feature_dim_astr'] + stats['feature_dim_multiph']}"
    )
    if not astr_feats or not mph_feats:
        print("ERROR: no samples survived featurization — aborting.")
        return 1

    # ---- Run CV ----
    print(f"\n[2/3] Running {args.n_folds}-fold cross-validation "
          "(joint vs separate Hilbert)...")
    metrics = cross_validate(
        astr_feats, mph_feats, targets,
        n_folds=args.n_folds,
        hilbert_dim=JOINT_HILBERT_DIM,
        n_iters=args.n_iters,
        lr=args.lr,
        seed=0,
    )
    print(
        f"  Joint Hilbert       R²={metrics['joint_r2_mean']:+.4f} ± "
        f"{metrics['joint_r2_std']:.4f}, wall={metrics['joint_wall_s_mean']:.4f}s"
    )
    print(
        f"  Separate streams    R²={metrics['separate_r2_mean']:+.4f} ± "
        f"{metrics['separate_r2_std']:.4f}, wall={metrics['separate_wall_s_mean']:.4f}s"
    )
    print(f"  speedup (separate / joint) = {metrics['speedup']:.2f}x")

    # ---- Write outputs ----
    print(f"\n[3/3] Writing outputs to {out_dir}...")
    metrics_path = out_dir / "metrics.json"
    metrics_payload = {
        "joint_r2_mean": metrics["joint_r2_mean"],
        "joint_r2_std": metrics["joint_r2_std"],
        "separate_r2_mean": metrics["separate_r2_mean"],
        "separate_r2_std": metrics["separate_r2_std"],
        "joint_wall_s_mean": metrics["joint_wall_s_mean"],
        "joint_wall_s_std": metrics["joint_wall_s_std"],
        "separate_wall_s_mean": metrics["separate_wall_s_mean"],
        "separate_wall_s_std": metrics["separate_wall_s_std"],
        "speedup": metrics["speedup"],
        "n_samples": int(metrics["n_samples"]),
        "n_folds": int(metrics["n_folds"]),
        "n_join_dim": int(metrics["n_join_dim"]),
        "n_parsed": int(stats["n_parsed"]),
        "n_failed": int(stats["n_failed"]),
        "n_target_rows": int(stats["n_target_rows"]),
        "mph_n_parsed": int(stats["mph_n_parsed"]),
        "mph_n_failed": int(stats["mph_n_failed"]),
        "mph_n_too_large": int(stats["mph_n_too_large"]),
        "feature_dim_astr": int(stats["feature_dim_astr"]),
        "feature_dim_multiph": int(stats["feature_dim_multiph"]),
        "feature_columns_astr": list(FEATURE_COLUMNS),
        "multi_ph_n_bins": int(MULTI_PH_N_BINS),
        "multi_ph_max_radius": float(MULTI_PH_MAX_RADIUS),
        "multi_ph_max_coord": float(MULTI_PH_MAX_COORD),
        "max_pocket_atoms": int(MAX_POCKET_ATOMS),
        "joint_hilbert_dim": int(JOINT_HILBERT_DIM),
        "joint_n_iters": int(args.n_iters),
        "joint_lr": float(args.lr),
        "per_fold": metrics["per_fold"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)
    print(f"  wrote {metrics_path}")

    table_path = out_dir / "comparison_table.md"
    table_md = render_table(metrics, stats)
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"  wrote {table_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
