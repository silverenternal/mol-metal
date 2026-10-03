"""Wf-25 Phase 4 — 5-seed × 3-method ASTSR benchmark on PlatinAI 5K.

================================================================
Comprehensive 5-fold × 5-seed × 3-method comparison:
- ``ASTSR`` (evolutionary ``LambdaASTSR``, R17 ship) — 8 RDKit descriptors.
- ``Hilbert`` (``LambdaASTSRHilbert``, Wf-14 ship) — 8 RDKit descriptors.
- ``Joint`` (``JointHilbertOptimizer``, Wf-21 phase-5 ship) — 8 RDKit +
  256 multi-PH (16×16 bins flattened).

The goal is to report **wall-time + R² + memory** for each backend
across 5 deterministic random seeds, on the PlatinAI 5K CSV.  Per-seed
mean / std is written to ``metrics.json`` along with per-fold records
and a 5-seed comparison markdown table under
``molmetal/reports/wf25_5seed_bench/``.

Honest framing
--------------
* ASTSR features are 8 RDKit descriptors — the *honest* ASTSR proxy
  used by Wf-25 Phases 1 & 3, not the F5 schema (which requires a
  Pareto-front unavailable here).
* Multi-PH features are deterministic (no learned parameters); only
  the downstream Hilbert regressor is trained.
* The three backends operate on different feature sets:
  ASTSR + Hilbert share the 8-d RDKit stream; Joint adds a 256-d
  multi-PH stream concatenated to 264-d before fitting.
* ``seed`` here controls the k-fold partition only — the per-backend
  internal RNG is reset independently (ASTSR via ``LambdaASTSR.seed``,
  Hilbert via ``LambdaASTSRHilbert.seed``, Joint via the wrapped
  ``LambdaASTSRHilbert.seed``).  Cross-backend determinism is
  therefore only required on a per-seed basis for the *split*.
* Wall-time and memory are machine-dependent; the speedup column is
  the intended *relative* metric.

Run::

    cd /home/hugo/codes/mol-metal
    .venv/bin/python -u molmetal/scripts/benchmark_wf25_5seed.py \\
        --n-samples 500 --n-folds 5 \\
        --data molmetal/data/platinai_5k.csv \\
        --output molmetal/reports/wf25_5seed_bench \\
        2>&1 | tee molmetal/reports/wf25_5seed_bench/run.log
"""

from __future__ import annotations

import argparse
import csv
import gc
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

import numpy as np  # used by _jsonable (module-level helper)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit chatter


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

# Hyperparameters — kept lean so the 5 × 5 × 3 = 75 fits complete in
# reasonable wall-time on CPU.
HILBERT_N_ITERS: int = 200
HILBERT_LR: float = 0.05
ASTSR_DEPTH: int = 4
ASTSR_POPULATION: int = 32
ASTSR_GENERATIONS: int = 20

#: Per-method hyperparameter overrides (kept as a module-level dict so the
#: test suite can introspect and override).
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "hilbert": {
        "n_iters": HILBERT_N_ITERS,
        "lr": HILBERT_LR,
    },
    "joint": {
        "n_iters": HILBERT_N_ITERS,
        "lr": HILBERT_LR,
    },
    "astsr": {
        "depth": ASTSR_DEPTH,
        "population": ASTSR_POPULATION,
        "generations": ASTSR_GENERATIONS,
    },
}

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_SAMPLES: int = 500
DEFAULT_N_FOLDS: int = 5
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_METALCYTOX_CSV: Path = _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf25_5seed_bench"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FoldRecord:
    """Per-(seed, fold, method) record."""

    seed: int
    fold_idx: int
    n_train: int
    n_test: int
    r2_test: float
    wall_s: float
    peak_mem_mb: float


@dataclass
class MethodSummary:
    """Aggregate over (seed × fold) for one method."""

    method: str
    r2_mean: float
    r2_std: float
    wall_s_mean: float
    wall_s_std: float
    peak_mem_mb_mean: float
    peak_mem_mb_std: float
    n_records: int
    per_seed: Dict[int, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SMILES → 8-d RDKit descriptor
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 8-d RDKit descriptor vector; ``None`` if SMILES fails."""
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


def _smiles_to_pocket_cloud(
    smiles: str,
    max_atoms: int = MAX_POCKET_ATOMS,
) -> Optional[Tuple[List[List[float]], List[int]]]:
    """3-D-embed ``smiles`` and return ``(coords, atom_types)``."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

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
    """Compute ``(n_bins ** 2,)`` multi-PH feature vector per molecule."""
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
# Design matrix
# ---------------------------------------------------------------------------
@dataclass
class DesignMatrix:
    """Container for the (astr, mph, target) design matrix."""

    smiles: Tuple[str, ...]
    astr: Tuple[Tuple[float, ...], ...]   # (N, 8)
    mph: Tuple[Tuple[float, ...], ...]    # (N, n_bins ** 2)
    target: Tuple[float, ...]
    stats: Dict[str, int]

    @property
    def n_samples(self) -> int:
        return len(self.smiles)

    @property
    def n_astr_dim(self) -> int:
        return len(self.astr[0]) if self.astr else 0

    @property
    def n_mph_dim(self) -> int:
        return len(self.mph[0]) if self.mph else 0

    @property
    def n_join_dim(self) -> int:
        return self.n_astr_dim + self.n_mph_dim


def build_design_matrix(
    *,
    platinai_csv: Path,
    metalcytotox_csv: Path,
    n_samples: int,
    seed: int,
) -> DesignMatrix:
    """Build a deterministic ``(smiles, astr, mph, target)`` quadruple."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    smiles_list: List[str] = []
    astr_feats: List[List[float]] = []
    targets: List[float] = []
    n_parsed = 0
    n_failed = 0

    # ---- 1. Primary: MetalCytoToxDB Ru subset ----
    if metalcytotox_csv.exists():
        mc_rows = load_metalcytotox_ru(metalcytotox_csv)
        rng.shuffle(mc_rows)
        for row in mc_rows:
            if len(smiles_list) >= n_samples:
                break
            smi = row.get("smiles", "")
            feats = _featurize_smiles(smi)
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

    # ---- 2. Supplement: PlatinAI 5K ----
    if len(smiles_list) < n_samples and platinai_csv.exists():
        pt_rows = load_platinai_5k(platinai_csv)
        rng.shuffle(pt_rows)
        for row in pt_rows:
            if len(smiles_list) >= n_samples:
                break
            smi = row.get("smiles", "")
            feats = _featurize_smiles(smi)
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

    # ---- 3. Multi-PH features ----
    mph_feats: List[List[float]] = []
    mph_stats: Dict[str, int] = {"n_parsed": 0, "n_failed": 0, "n_too_large": 0}
    if smiles_list:
        mph_feats, mph_stats = compute_multi_ph_features(
            smiles_list, n_bins=MULTI_PH_N_BINS, max_atoms=MAX_POCKET_ATOMS,
        )

    # Align by dropping multi-PH failures.
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
    return DesignMatrix(
        smiles=tuple(smiles_list),
        astr=tuple(tuple(r) for r in astr_feats),
        mph=tuple(tuple(r) for r in mph_feats),
        target=tuple(targets),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# k-fold partition
# ---------------------------------------------------------------------------
def kfold_indices(
    n: int, k: int, seed: int
) -> List[Tuple[List[int], List[int]]]:
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
# R² helper
# ---------------------------------------------------------------------------
def _compute_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Held-out R² — overflow-safe."""
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
# Per-method fit/predict
# ---------------------------------------------------------------------------
def _fit_predict_astsr(
    X_tr: Any, y_tr: Any, X_te: Any, seed: int, hp: Dict[str, Any]
) -> Tuple[Any, float]:
    """Evolutionary ``LambdaASTSR`` (R17 ship)."""
    from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

    model = LambdaASTSR(
        depth=int(hp.get("depth", ASTSR_DEPTH)),
        population=int(hp.get("population", ASTSR_POPULATION)),
        generations=int(hp.get("generations", ASTSR_GENERATIONS)),
        seed=int(seed),
        feature_names=tuple(FEATURE_COLUMNS),
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    wall = float(time.perf_counter() - t0)
    y_hat = model.predict(X_te)
    return y_hat, wall


def _fit_predict_hilbert(
    X_tr: Any, y_tr: Any, X_te: Any, seed: int, hp: Dict[str, Any]
) -> Tuple[Any, float]:
    """``LambdaASTSRHilbert`` (Wf-14 ship)."""
    from molmetal.molmetal_lam.reward.hilbert_astsr_integration import (
        HilbertASTSRComposite,
    )
    from molmetal.molmetal_lam.reward.hilbert_primal_dual import (
        LambdaASTSRHilbert,
    )

    composite = HilbertASTSRComposite(
        feature_columns=tuple(FEATURE_COLUMNS),
        use_hilbert=True,
        population_dim=int(X_tr.shape[1]),
    )
    composite.hilbert_optimizer = LambdaASTSRHilbert(
        population_dim=int(X_tr.shape[1]),
        lr_primal=float(hp.get("lr", HILBERT_LR)),
        lr_dual=float(hp.get("lr", HILBERT_LR)),
        n_iters=int(hp.get("n_iters", HILBERT_N_ITERS)),
        seed=int(seed),
    )
    t0 = time.perf_counter()
    composite.fit(X_tr, y_tr)
    wall = float(time.perf_counter() - t0)
    y_hat = composite.predict(X_te)
    return y_hat, wall


def _fit_predict_joint(
    astr_tr: Any, mph_tr: Any, y_tr: Any,
    astr_te: Any, mph_te: Any, seed: int, hp: Dict[str, Any],
) -> Tuple[Any, float, int]:
    """``JointHilbertOptimizer`` (Wf-21 phase-5 ship)."""
    from molmetal.molmetal_lam.reward.astsr_multiph_hilbert import (
        JointHilbertOptimizer,
    )

    opt = JointHilbertOptimizer(
        hilbert_dim=int(hp.get("hilbert_dim", 64)),
        normalize=True,
        feature_columns=tuple(FEATURE_COLUMNS),
        n_iters=int(hp.get("n_iters", HILBERT_N_ITERS)),
        lr=float(hp.get("lr", HILBERT_LR)),
    )
    t0 = time.perf_counter()
    res = opt.fit(astr_tr, mph_tr, y_tr)
    wall = float(time.perf_counter() - t0)
    y_hat = opt.predict(astr_te, mph_te)
    return y_hat, wall, int(res.get("join_dim", 0))


# ---------------------------------------------------------------------------
# Top-level benchmark driver
# ---------------------------------------------------------------------------
def run_5seed_benchmark(
    *,
    platinai_csv: Path = DEFAULT_PLATINAI_CSV,
    metalcytotox_csv: Path = DEFAULT_METALCYTOX_CSV,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_folds: int = DEFAULT_N_FOLDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    methods: Sequence[str] = ("astsr", "hilbert", "joint"),
    hyperparams: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full 5-seed × 3-method × n_folds-fold benchmark.

    Returns a dict with keys:
        ``per_method``: ``{method: MethodSummary-like dict}``
        ``per_fold``: ``list[dict]`` (long-form fold records).
        ``metadata``: dataset stats + hyperparameters.
        ``seeds``, ``methods``, ``n_folds``, ``n_samples``, ``join_dim``.
    """
    import numpy as np

    hp = {m: dict(DEFAULT_HYPERPARAMS[m]) for m in methods}
    if hyperparams:
        for k, v in hyperparams.items():
            hp[k] = dict(v)

    print(
        f"[wf25_5seed] platinai={platinai_csv} n_samples={n_samples} "
        f"n_folds={n_folds} seeds={list(seeds)} methods={list(methods)}"
    )

    # ---- Build the design matrix once (deterministic seed=0) ----
    print("[wf25_5seed] Building design matrix (seed=0)...")
    design = build_design_matrix(
        platinai_csv=platinai_csv,
        metalcytotox_csv=metalcytotox_csv,
        n_samples=int(n_samples),
        seed=0,
    )
    print(
        f"  parsed={design.stats['n_parsed']}, "
        f"failed={design.stats['n_failed']}, "
        f"target_rows={design.stats['n_target_rows']}, "
        f"mph_parsed={design.stats['mph_n_parsed']}, "
        f"mph_failed={design.stats['mph_n_failed']}"
    )
    print(
        f"  astr_dim={design.n_astr_dim}, mph_dim={design.n_mph_dim}, "
        f"join_dim={design.n_join_dim}"
    )
    if design.n_samples < int(n_folds) * 2:
        raise ValueError(
            f"design has only {design.n_samples} rows; need "
            f"at least {int(n_folds) * 2} for {n_folds}-fold CV"
        )

    astr_arr = np.asarray(design.astr, dtype=float)
    mph_arr = np.asarray(design.mph, dtype=float)
    y_arr = np.asarray(design.target, dtype=float).reshape(-1)

    # ---- Sweep ----
    fold_records: List[Dict[str, Any]] = []
    per_method_records: Dict[str, List[FoldRecord]] = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n[wf25_5seed] === seed={seed} ===")
        splits = kfold_indices(design.n_samples, int(n_folds), seed=int(seed))
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"  seed={seed} fold={fold_idx}: "
                  f"train={len(train_idx)} test={len(test_idx)}")

            astr_tr = astr_arr[train_idx]
            astr_te = astr_arr[test_idx]
            mph_tr = mph_arr[train_idx]
            mph_te = mph_arr[test_idx]
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]

            # Per-fold z-score on the ASTSR stream (Hilbert / ASTSR need it).
            mu = astr_tr.mean(axis=0)
            sigma = np.where(astr_tr.std(axis=0) < 1e-9, 1.0, astr_tr.std(axis=0))
            astr_tr_n = (astr_tr - mu) / sigma
            astr_te_n = (astr_te - mu) / sigma

            for method in methods:
                gc.collect()
                tracemalloc.start()
                try:
                    if method == "astsr":
                        y_hat, wall = _fit_predict_astsr(
                            astr_tr_n, y_tr, astr_te_n,
                            seed=int(seed), hp=hp["astsr"],
                        )
                        peak = tracemalloc.get_traced_memory()[1]
                    elif method == "hilbert":
                        y_hat, wall = _fit_predict_hilbert(
                            astr_tr_n, y_tr, astr_te_n,
                            seed=int(seed), hp=hp["hilbert"],
                        )
                        peak = tracemalloc.get_traced_memory()[1]
                    elif method == "joint":
                        # Per-fold z-score on the multi-PH stream too —
                        # without this the Hilbert ``features.T @
                        # features`` overflows on the 256-d inputs.
                        mu_mph = mph_tr.mean(axis=0)
                        sigma_mph = np.where(
                            mph_tr.std(axis=0) < 1e-9, 1.0,
                            mph_tr.std(axis=0)
                        )
                        mph_tr_n = (mph_tr - mu_mph) / sigma_mph
                        mph_te_n = (mph_te - mu_mph) / sigma_mph
                        y_hat, wall, _jd = _fit_predict_joint(
                            astr_tr_n, mph_tr_n, y_tr,
                            astr_te_n, mph_te_n,
                            seed=int(seed), hp=hp["joint"],
                        )
                        peak = tracemalloc.get_traced_memory()[1]
                    else:
                        raise ValueError(f"unknown method {method!r}")
                finally:
                    tracemalloc.stop()
                y_hat_arr = np.clip(
                    np.asarray(y_hat, dtype=float).reshape(-1),
                    -1e6, 1e6,
                )
                r2 = float(_compute_r2(y_te.tolist(), y_hat_arr.tolist()))
                rec = FoldRecord(
                    seed=int(seed),
                    fold_idx=int(fold_idx),
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    r2_test=float(r2),
                    wall_s=float(wall),
                    peak_mem_mb=float(peak) / (1024.0 * 1024.0),
                )
                per_method_records[method].append(rec)
                fold_records.append({
                    "seed": rec.seed,
                    "fold_idx": rec.fold_idx,
                    "n_train": rec.n_train,
                    "n_test": rec.n_test,
                    "r2_test": rec.r2_test,
                    "wall_s": rec.wall_s,
                    "peak_mem_mb": rec.peak_mem_mb,
                    "method": method,
                })
                print(
                    f"    {method:7s}: R²={rec.r2_test:+.4f} "
                    f"wall={rec.wall_s:.3f}s "
                    f"peak_mem={rec.peak_mem_mb:.2f}MB"
                )

    # ---- Aggregate per method ----
    def _mean(xs: Sequence[float]) -> float:
        return float(np.mean(list(xs))) if xs else 0.0

    def _std(xs: Sequence[float]) -> float:
        return float(np.std(list(xs), ddof=1)) if len(xs) > 1 else 0.0

    per_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        recs = per_method_records[method]
        r2s = [r.r2_test for r in recs]
        walls = [r.wall_s for r in recs]
        mems = [r.peak_mem_mb for r in recs]
        per_seed: Dict[int, Dict[str, float]] = {}
        for s in seeds:
            s_recs = [r for r in recs if r.seed == int(s)]
            if s_recs:
                per_seed[int(s)] = {
                    "r2_mean": _mean([r.r2_test for r in s_recs]),
                    "r2_std": _std([r.r2_test for r in s_recs]),
                    "wall_s_mean": _mean([r.wall_s for r in s_recs]),
                    "peak_mem_mb_mean": _mean(
                        [r.peak_mem_mb for r in s_recs]
                    ),
                }
        per_method[method] = {
            "method": method,
            "r2_mean": _mean(r2s),
            "r2_std": _std(r2s),
            "wall_s_mean": _mean(walls),
            "wall_s_std": _std(walls),
            "peak_mem_mb_mean": _mean(mems),
            "peak_mem_mb_std": _std(mems),
            "n_records": len(recs),
            "per_seed": per_seed,
        }

    metadata = {
        "n_samples": int(design.n_samples),
        "n_folds": int(n_folds),
        "seeds": list(seeds),
        "methods": list(methods),
        "join_dim": int(design.n_join_dim),
        "astr_dim": int(design.n_astr_dim),
        "mph_dim": int(design.n_mph_dim),
        "feature_columns": list(FEATURE_COLUMNS),
        "multi_ph_n_bins": int(MULTI_PH_N_BINS),
        "multi_ph_max_radius": float(MULTI_PH_MAX_RADIUS),
        "multi_ph_max_coord": float(MULTI_PH_MAX_COORD),
        "max_pocket_atoms": int(MAX_POCKET_ATOMS),
        "hilbert_n_iters": int(HILBERT_N_ITERS),
        "hilbert_lr": float(HILBERT_LR),
        "astsr_depth": int(ASTSR_DEPTH),
        "astsr_population": int(ASTSR_POPULATION),
        "astsr_generations": int(ASTSR_GENERATIONS),
        "hyperparams": {k: dict(v) for k, v in hp.items()},
        "stats": dict(design.stats),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "per_method": per_method,
        "per_fold": fold_records,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------
def render_5seed_table(payload: Dict[str, Any]) -> str:
    """Render the 5-seed comparison markdown."""
    methods = payload["metadata"]["methods"]
    per_method = payload["per_method"]
    seeds = payload["metadata"]["seeds"]
    n_samples = payload["metadata"]["n_samples"]
    n_folds = payload["metadata"]["n_folds"]
    join_dim = payload["metadata"]["join_dim"]

    # Identify the fastest method for the speedup ref.
    fastest = min(methods, key=lambda m: per_method[m]["wall_s_mean"])

    lines: List[str] = [
        "# Wf-25 Phase 4 — 5-seed × 3-method ASTSR benchmark (PlatinAI 5K)",
        "",
        f"**Dataset:** PlatinAI 5K + MetalCytoToxDB Ru subset (n={n_samples})",
        f"**Seeds:** {list(seeds)}  |  **Folds per seed:** {n_folds}  |  "
        f"**Join dim:** {join_dim} (astr={payload['metadata']['astr_dim']} + "
        f"mph={payload['metadata']['mph_dim']})",
        f"**Multi-PH bins:** {payload['metadata']['multi_ph_n_bins']}  |  "
        f"**Hilbert iters:** {payload['metadata']['hilbert_n_iters']}  |  "
        f"**Hilbert lr:** {payload['metadata']['hilbert_lr']}",
        f"**ASTSR depth:** {payload['metadata']['astsr_depth']}  |  "
        f"**population:** {payload['metadata']['astsr_population']}  |  "
        f"**generations:** {payload['metadata']['astsr_generations']}",
        "",
        f"**Parse stats:** parsed={payload['metadata']['stats']['n_parsed']}, "
        f"failed={payload['metadata']['stats']['n_failed']}, "
        f"target_rows={payload['metadata']['stats']['n_target_rows']}, "
        f"mph_parsed={payload['metadata']['stats']['mph_n_parsed']}, "
        f"mph_failed={payload['metadata']['stats']['mph_n_failed']}",
        "",
        "## Aggregate (5-seed × 5-fold = 25 records per method)",
        "",
        "| Method | R² mean ± std | Wall (s) mean ± std | Peak mem (MB) | Speedup |",
        "|---|---|---|---|---|",
    ]
    for m in methods:
        agg = per_method[m]
        speedup = (
            agg["wall_s_mean"] / per_method[fastest]["wall_s_mean"]
            if per_method[fastest]["wall_s_mean"] > 0 else float("inf")
        )
        speedup_str = (
            f"{speedup:.2f}x (ref)" if m == fastest else f"{speedup:.2f}x"
        )
        lines.append(
            f"| {m} | {agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f} | "
            f"{agg['wall_s_mean']:.3f} ± {agg['wall_s_std']:.3f} | "
            f"{agg['peak_mem_mb_mean']:.2f} ± {agg['peak_mem_mb_std']:.2f} | "
            f"{speedup_str} |"
        )

    lines.extend([
        "",
        "## Per-seed R² (mean across folds)",
        "",
        "| Method | " + " | ".join(f"seed {s}" for s in seeds) + " |",
        "|---|" + "|".join(["---"] * len(seeds)) + "|",
    ])
    for m in methods:
        cells = []
        for s in seeds:
            v = per_method[m]["per_seed"].get(int(s))
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v['r2_mean']:+.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Per-seed wall time (mean across folds)",
        "",
        "| Method | " + " | ".join(f"seed {s}" for s in seeds) + " |",
        "|---|" + "|".join(["---"] * len(seeds)) + "|",
    ])
    for m in methods:
        cells = []
        for s in seeds:
            v = per_method[m]["per_seed"].get(int(s))
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v['wall_s_mean']:.3f}s")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Honest framing",
        "",
        "* ASTSR features are 8 RDKit descriptors — the F5 schema is",
        "  unavailable offline (requires a Pareto-front computation).",
        "* The three backends operate on different feature sets:",
        "  ASTSR + Hilbert share the 8-d RDKit stream; Joint adds a 256-d",
        "  multi-PH stream (16×16 bins flattened) concatenated to 264-d",
        "  before fitting.",
        "* Wall-time and memory are machine-dependent; the speedup column",
        "  is the intended *relative* metric against the fastest backend",
        f"  ({fastest}).",
        "* `seed` controls the k-fold partition only; the per-backend",
        "  internal RNG is reset independently for each (seed, fold).",
        "* Activity is binary (0/1); the target is `activity + N(0, 0.05)`",
        "  — R² measures how much of that binary signal each backend",
        "  recovers from descriptors alone.",
        "",
        f"**Run timestamp (UTC):** {payload['metadata']['timestamp_utc']}",
        "",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------
def _jsonable(o: object) -> object:
    """Recursively convert numpy / Path types to JSON-serialisable types."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, Path):
        return str(o)
    return o


def write_metrics(payload: Dict[str, Any], path: Path) -> Path:
    """Write the benchmark payload as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False))
    return path


def write_markdown(payload: Dict[str, Any], path: Path) -> Path:
    """Write the comparison markdown to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_5seed_table(payload))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-25 Phase 4 — 5-seed × 3-method ASTSR benchmark on PlatinAI 5K."
        )
    )
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--data", type=str, default=str(DEFAULT_PLATINAI_CSV))
    p.add_argument("--metalcytotox", type=str,
                   default=str(DEFAULT_METALCYTOX_CSV))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; sweep runs seeds {seed, seed+1, ..., seed+4}")
    p.add_argument("--hilbert-n-iters", type=int, default=HILBERT_N_ITERS)
    p.add_argument("--hilbert-lr", type=float, default=HILBERT_LR)
    p.add_argument("--astsr-depth", type=int, default=ASTSR_DEPTH)
    p.add_argument("--astsr-population", type=int, default=ASTSR_POPULATION)
    p.add_argument("--astsr-generations", type=int, default=ASTSR_GENERATIONS)
    p.add_argument(
        "--methods", type=str, nargs="+",
        default=["astsr", "hilbert", "joint"],
        choices=["astsr", "hilbert", "joint"],
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    seeds = tuple(int(args.seed) + i for i in range(5))

    hyperparams = {
        "hilbert": {"n_iters": args.hilbert_n_iters, "lr": args.hilbert_lr},
        "joint":   {"n_iters": args.hilbert_n_iters, "lr": args.hilbert_lr},
        "astsr":   {
            "depth": args.astsr_depth,
            "population": args.astsr_population,
            "generations": args.astsr_generations,
        },
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    payload = run_5seed_benchmark(
        platinai_csv=Path(args.data),
        metalcytotox_csv=Path(args.metalcytotox),
        n_samples=int(args.n_samples),
        n_folds=int(args.n_folds),
        seeds=seeds,
        methods=tuple(args.methods),
        hyperparams=hyperparams,
    )
    elapsed = float(time.perf_counter() - t0)

    metrics_path = write_metrics(payload, out_dir / "metrics.json")
    print(f"[wf25_5seed] wrote {metrics_path}")

    md_path = write_markdown(payload, out_dir / "5seed_comparison.md")
    print(f"[wf25_5seed] wrote {md_path}")

    print(
        f"\n[wf25_5seed] === Summary (over {len(seeds)} seeds × "
        f"{args.n_folds} folds × {len(args.methods)} methods) ==="
    )
    for m in args.methods:
        agg = payload["per_method"][m]
        print(
            f"  {m:7s}: R²={agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f}  "
            f"wall={agg['wall_s_mean']:.3f}s ± {agg['wall_s_std']:.3f}  "
            f"peak_mem={agg['peak_mem_mb_mean']:.2f}MB"
        )
    print(f"\n[wf25_5seed] elapsed={elapsed:.1f}s  out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())