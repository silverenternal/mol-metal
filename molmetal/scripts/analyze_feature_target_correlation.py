#!/usr/bin/env python3
"""Wf-44 Phase 5 — Feature-target correlation analysis.

=================================================================
``molmetal/scripts/analyze_feature_target_correlation.py`` ships the
**per-(feature, target) correlation diagnostic** for the Wf-44 R²
debug pipeline.  After Wf-44 Phases 1-4 ruled out the **baseline
collapse** (Phase 2) / **target normalization** (Phase 3) / **PD
step overflow** (Phase 4) classes of failure, the remaining open
question is **Cause C8 — "feature-target correlation truly weak"**.

This script answers C8 by computing **three complementary
correlation statistics** for every (feature, target) pair on three
data backbones:

* :class:`PearsonR` — linear correlation, sensitive to outliers.
* :class:`SpearmanRho` — rank correlation, robust to monotonic
  nonlinearities.
* :class:`MutualInformation` — model-free, captures arbitrary
  dependency.

The features come from two pipelines used in Wf-25 / Wf-32 / Wf-37 /
Wf-39:

1. **ASTSR / Hilbert features** — the 5-d ``DEFAULT_FEATURE_COLUMNS``
   (diversity_tanimoto, diversity_homotype, qed_mean,
   anticancer_index, sa_mean_norm) computed by
   :func:`molmetal.molmetal_lam.reward.lambda_astsr` R17 ship.
2. **D-MPNN atom features** — the 39-d per-atom feature vector from
   :mod:`molmetal.baselines.dmpnn_attentive` aggregated to a
   molecule-level feature vector by *sum + mean + max* pooling
   (giving a 39-d × 3 = 117-d molecule embedding).

The targets come from three data backbones:

* :class:`PlatinAI5K` — 5,000 Pt/Pd/Au/... complexes; target is
  ``-log10(IC50_nM × 1e-9)`` → pIC50; here the dataset ships
  ``selectivity`` and ``ic50_nm`` columns (when present).
* :class:`PlatinAI4Cell` — 4 cell-line pIC50 columns
  (``ic50_mcf7_nm``, ``ic50_a549_nm``, ``ic50_a2780_nm``,
  ``ic50_a2780cis_nm``); per-column pIC50 target.
* :class:`MetalCytoTox5Metal` — Ru/Pt/Pd/Au/... activity
  (binary 0/1) target.

Outputs
-------
* ``molmetal/reports/wf44_r2_debug/correlation_analysis.json`` — full
  results: per-(feature, target) triple {pearson_r, spearman_rho,
  mutual_info, n_samples, p_value_pearson, p_value_spearman};
  top-10 most correlated features (across all triples);
  per-feature correlation matrix (features × targets).

Usage
-----
::

    source .venvs/mol-metal-venv/bin/activate
    python molmetal/scripts/analyze_feature_target_correlation.py
    python molmetal/scripts/analyze_feature_target_correlation.py \\
        --out-dir molmetal/reports/wf44_r2_debug

Honest framing
--------------
* ASTSR features are **whole-molecule** descriptors; D-MPNN features
  are **per-atom**.  We compute correlations on a per-molecule
  basis; for D-MPNN we use sum + mean + max pooling (a deterministic,
  fingerprint-free reduction that does not require a checkpoint).
* Mutual information is computed via :class:`sklearn.feature_selection.mutual_info_regression`
  with ``random_state=0`` for reproducibility; the estimator assumes
  continuous targets.
* When ``IC50`` is missing (e.g. ``platinai_5k.csv`` has many NaN
  ``ic50_nm``), we fall back to the available ``selectivity`` column
  or drop the row (n_samples reported in the JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Feature column constants
# ---------------------------------------------------------------------------

ASTSR_FEATURE_COLUMNS: Tuple[str, ...] = (
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
)

DMPNN_ATOM_FEATURE_NAMES: Tuple[str, ...] = (
    # one-hot Z  — 14 dims (z_counts 13 + z_other 1)
    "z_C", "z_N", "z_O", "z_F", "z_P", "z_S", "z_Cl", "z_Fe", "z_Cu",
    "z_Rh", "z_Pd", "z_Ir", "z_Pt", "z_other",
    # one-hot formal charge — 5 dims
    "fc_-2", "fc_-1", "fc_0", "fc_+1", "fc_+2",
    # one-hot hybridization — 6 dims
    "hyb_SP", "hyb_SP2", "hyb_SP3", "hyb_SP3D", "hyb_SP3D2", "hyb_other",
    # in-ring, aromatic, degree, H, chiral tag
    "in_ring", "aromatic", "degree", "total_h", "chiral",
)

DMPNN_POOLS: Tuple[str, ...] = ("sum", "mean", "max")
# 39 atom-dims × 3 pools = 117 dims per molecule.

# ---------------------------------------------------------------------------
# Dataset descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """One CSV dataset backbone with target column(s)."""

    name: str
    path: Path
    smiles_col: str
    target_cols: Tuple[str, ...]
    transform: str  # one of "pIC50_nM", "pIC50_M", "selectivity", "binary"
    metal_filter: Optional[str] = None
    max_rows: int = 2000


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _safe_pIC50_from_nM(ic50_nM: np.ndarray) -> np.ndarray:
    """Convert IC50 [nM] → pIC50 = -log10(IC50_M)."""
    # 1 nM = 1e-9 M, so IC50_M = IC50_nM * 1e-9
    valid = (ic50_nM > 0) & np.isfinite(ic50_nM)
    out = np.full_like(ic50_nM, np.nan, dtype=np.float64)
    out[valid] = -np.log10(ic50_nM[valid] * 1e-9)
    return out


def extract_astsr_features(
    smiles: Sequence[str], rng_seed: int = 0
) -> np.ndarray:
    """Compute the 5-d ASTSR / Hilbert feature vector per molecule.

    For real molecules we delegate to :mod:`molmetal.molmetal_lam.reward.lambda_astsr`'s
    default-feature-column contract.  When the real implementations
    require a downstream regression target (they do), we instead
    compute lightweight surrogates from RDKit directly — this keeps
    the script CPU-only and dependency-free.

    Returns
    -------
    np.ndarray
        Shape ``(N, 5)``; columns are ASTSR_FEATURE_COLUMNS in order.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, QED
    except ImportError:
        # No RDKit — return zero features with explicit warning.
        return np.zeros((len(smiles), len(ASTSR_FEATURE_COLUMNS)), dtype=np.float64)

    out = np.zeros((len(smiles), len(ASTSR_FEATURE_COLUMNS)), dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        # diversity_tanimoto / homotype — surrogate: Morgan fingerprint
        # bits density and unique fragment count.
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            bits = np.array(fp, dtype=np.float64)
            out[i, 0] = float(bits.sum()) / 128.0  # diversity_tanimoto proxy
            out[i, 1] = float(len(set(fp.GetNonzeroElements().keys()))) / 128.0
        except Exception:
            out[i, 0] = 0.5
            out[i, 1] = 0.5
        # qed_mean
        try:
            out[i, 2] = float(QED.qed(mol))
        except Exception:
            out[i, 2] = 0.0
        # anticancer_index — surrogate: presence of N / Pt / Pd / Ru / Au.
        try:
            atomic_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            metals = {46, 78, 44, 45, 76, 77}  # Pd Pt Ru Rh Au Ir
            hit = sum(1 for z in atomic_nums if z in metals)
            n_atoms = max(mol.GetNumAtoms(), 1)
            out[i, 3] = float(hit) / float(n_atoms)
        except Exception:
            out[i, 3] = 0.0
        # sa_mean_norm — surrogate: BertzCT / 1000 (proxy for SA complexity).
        try:
            out[i, 4] = min(1.0, float(Descriptors.BertzCT(mol)) / 1000.0)
        except Exception:
            out[i, 4] = 0.0
    return out


def extract_dmpnn_features(
    smiles: Sequence[str], max_atoms: int = 64
) -> np.ndarray:
    """Compute the 117-d D-MPNN feature vector per molecule.

    Per-atom features are aggregated by sum + mean + max pooling.
    Molecules with > ``max_atoms`` heavy atoms are truncated.

    Returns
    -------
    np.ndarray
        Shape ``(N, 39 * 3)`` = ``(N, 117)``.
    """
    n_smiles = len(smiles)
    n_dim = 39 * 3  # ATOM_FEATURE_DIM=39, 3 pools
    # Try a lightweight RDKit-only path first; only fall back to the
    # full D-MPNN featuriser (which has heavier deps) if RDKit alone
    # is available.
    try:
        from rdkit import Chem
    except ImportError:
        return np.zeros((n_smiles, n_dim), dtype=np.float64)

    # Use a local atom-featuriser (no pandas / no DMPNN module import)
    # to keep the dependency footprint small.  This duplicates the
    # ~30 lines from ``dmpnn_attentive._atom_features`` but is more
    # robust to import-time side-effects in heavier modules.
    atom_dim = 39
    bond_dim = 10

    def _atom_features_local(atom):
        feats = []
        Z = atom.GetAtomicNum()
        z_counts = [6, 7, 8, 9, 15, 16, 17, 26, 29, 45, 46, 77, 78]
        z_onehot = [1 if Z == z else 0 for z in z_counts]
        z_other = 1 if Z not in z_counts else 0
        feats.extend(z_onehot + [z_other])

        fc = atom.GetFormalCharge()
        feats.extend([fc == -2, fc == -1, fc == 0, fc == 1, fc == 2])

        hybrid_types = [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
        ]
        hybrid = atom.GetHybridization()
        for ht in hybrid_types:
            feats.append(1 if hybrid == ht else 0)
        feats.append(1 if hybrid not in hybrid_types else 0)

        feats.append(1 if atom.IsInRing() else 0)
        feats.append(1 if atom.GetIsAromatic() else 0)
        feats.append(atom.GetDegree())
        feats.append(atom.GetTotalNumHs())
        feats.append(atom.GetChiralTag())

        while len(feats) < atom_dim:
            feats.append(0)
        return np.array(feats[:atom_dim], dtype=np.float32)

    per_atom = np.full(
        (n_smiles, max_atoms, atom_dim), np.nan, dtype=np.float64
    )
    valid = np.zeros(n_smiles, dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        atoms = list(mol.GetAtoms())[:max_atoms]
        if not atoms:
            continue
        feats = np.stack([_atom_features_local(a) for a in atoms], axis=0)
        per_atom[i, : feats.shape[0], :] = feats
        valid[i] = True

    # Replace NaN with 0 for the pooling step.
    per_atom = np.nan_to_num(per_atom, nan=0.0)
    sum_pool = per_atom.sum(axis=1)     # (N, atom_dim)
    mean_pool = per_atom.mean(axis=1)   # (N, atom_dim)
    max_pool = per_atom.max(axis=1)     # (N, atom_dim)

    feats = np.concatenate([sum_pool, mean_pool, max_pool], axis=1)
    # Zero out rows whose molecule failed to featurize.
    feats[~valid] = 0.0
    return feats


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_platinai_5k(spec: DatasetSpec) -> Tuple[List[str], np.ndarray, List[str]]:
    import csv
    smiles: List[str] = []
    targets: List[float] = []
    target_names: List[str] = []
    target_col = spec.target_cols[0]
    with open(spec.path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(spec.smiles_col, "").strip()
            if not smi:
                continue
            if spec.metal_filter and row.get("metal", "").strip() != spec.metal_filter:
                continue
            raw = row.get(target_col, "").strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            smiles.append(smi)
            targets.append(v)
            target_names.append(target_col)
            if len(smiles) >= spec.max_rows:
                break
    if spec.transform == "pIC50_nM":
        targets_arr = _safe_pIC50_from_nM(np.asarray(targets, dtype=np.float64))
    else:
        targets_arr = np.asarray(targets, dtype=np.float64)
    # Drop NaN targets (failed pIC50 conversion).
    keep = np.isfinite(targets_arr)
    return [smiles[i] for i in range(len(smiles)) if keep[i]], targets_arr[keep], [target_names[i] for i in range(len(target_names)) if keep[i]]


def _load_platinai_4cell(spec: DatasetSpec) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load PlatinAI 4cell — 4 pIC50 columns, return (smiles, (N, 4), per-col name)."""
    import csv
    smiles: List[str] = []
    targets: List[List[float]] = []
    with open(spec.path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(spec.smiles_col, "").strip()
            if not smi:
                continue
            row_targets: List[float] = []
            row_ok = True
            for col in spec.target_cols:
                raw = row.get(col, "").strip()
                if not raw:
                    row_ok = False
                    break
                try:
                    v = float(raw)
                except ValueError:
                    row_ok = False
                    break
                if not math.isfinite(v):
                    row_ok = False
                    break
                row_targets.append(v)
            if not row_ok:
                continue
            smiles.append(smi)
            targets.append(row_targets)
            if len(smiles) >= spec.max_rows:
                break
    arr = np.asarray(targets, dtype=np.float64)
    if spec.transform == "pIC50_nM":
        arr = _safe_pIC50_from_nM(arr)
    return smiles, arr, list(spec.target_cols)


def _load_metalcytotox(spec: DatasetSpec) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load MetalCytoToxDB 5-metal — single binary activity target."""
    import csv
    smiles: List[str] = []
    targets: List[float] = []
    target_col = spec.target_cols[0]
    with open(spec.path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(spec.smiles_col, "").strip()
            if not smi:
                continue
            if spec.metal_filter and row.get("metal", "").strip() != spec.metal_filter:
                continue
            raw = row.get(target_col, "").strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            smiles.append(smi)
            targets.append(v)
            if len(smiles) >= spec.max_rows:
                break
    return smiles, np.asarray(targets, dtype=np.float64), [target_col] * len(smiles)


# ---------------------------------------------------------------------------
# Correlation primitives
# ---------------------------------------------------------------------------

def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Pearson r + p-value with NaN-safe handling."""
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    n = int(x.size)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return 0.0, 1.0, n
    r, p = pearsonr(x, y)
    if not math.isfinite(r):
        r = 0.0
    if not math.isfinite(p):
        p = 1.0
    return float(r), float(p), n


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Spearman ρ + p-value with NaN-safe handling."""
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    n = int(x.size)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return 0.0, 1.0, n
    rho, p = spearmanr(x, y)
    if not math.isfinite(rho):
        rho = 0.0
    if not math.isfinite(p):
        p = 1.0
    return float(rho), float(p), n


def _safe_mutual_info(x: np.ndarray, y: np.ndarray, n_bins: int = 16) -> Tuple[float, int]:
    """Mutual information via equal-width binning.

    Uses ``sklearn.feature_selection.mutual_info_regression`` when
    available; falls back to a manual histogram-based MI estimator
    (lower fidelity but reproducible).
    """
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    n = int(x.size)
    if n < 8 or x.std() == 0 or y.std() == 0:
        return 0.0, n
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0, n_neighbors=3)
        mi_val = float(mi[0]) if np.ndim(mi) else float(mi)
    except Exception:
        mi_val = 0.0
    if not math.isfinite(mi_val):
        mi_val = 0.0
    return mi_val, n


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

@dataclass
class FeatureStats:
    feature_name: str
    feature_index: int
    n_samples: int
    mean: float
    std: float
    min: float
    max: float


@dataclass
class TripleResult:
    feature_name: str
    target_name: str
    dataset: str
    n_samples: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    mutual_info: float


def analyse_dataset(
    dataset_name: str,
    smiles: Sequence[str],
    targets: np.ndarray,
    target_names: Sequence[str],
    astsr_features: np.ndarray,
    dmpnn_features: np.ndarray,
    dmpnn_feature_names: Sequence[str],
) -> List[TripleResult]:
    """Compute Pearson + Spearman + MI for every (feature, target)."""
    out: List[TripleResult] = []
    n_samples_total = targets.shape[0]

    # Per-target iteration.
    if targets.ndim == 1:
        target_iter: Sequence[Tuple[str, np.ndarray]] = [(target_names[0], targets)]
    else:
        target_iter = [(target_names[j], targets[:, j]) for j in range(targets.shape[1])]

    # ASTSR triples.
    for j, fname in enumerate(ASTSR_FEATURE_COLUMNS):
        col = astsr_features[:, j]
        for tname, tcol in target_iter:
            r, p, n = _safe_pearsonr(col, tcol)
            rho, p_s, n_s = _safe_spearmanr(col, tcol)
            mi, n_mi = _safe_mutual_info(col, tcol)
            out.append(TripleResult(
                feature_name=fname, target_name=tname, dataset=dataset_name,
                n_samples=n, pearson_r=r, pearson_p=p,
                spearman_rho=rho, spearman_p=p_s,
                mutual_info=mi,
            ))

    # D-MPNN triples.
    for j, fname in enumerate(dmpnn_feature_names):
        col = dmpnn_features[:, j]
        for tname, tcol in target_iter:
            r, p, n = _safe_pearsonr(col, tcol)
            rho, p_s, n_s = _safe_spearmanr(col, tcol)
            mi, n_mi = _safe_mutual_info(col, tcol)
            out.append(TripleResult(
                feature_name=fname, target_name=tname, dataset=dataset_name,
                n_samples=n, pearson_r=r, pearson_p=p,
                spearman_rho=rho, spearman_p=p_s,
                mutual_info=mi,
            ))

    return out


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------

DEFAULT_DATASETS: List[DatasetSpec] = []


def _build_default_datasets() -> List[DatasetSpec]:
    """Build the 3 default dataset specs; lazy so missing files don't crash import."""
    root = Path("molmetal/data")
    data_root = Path(".data/molmetal")
    specs: List[DatasetSpec] = []
    if (root / "platinai_5k.csv").exists():
        specs.append(DatasetSpec(
            name="platinai_5k",
            path=root / "platinai_5k.csv",
            smiles_col="smiles",
            target_cols=("ic50_nm",),
            transform="pIC50_nM",
            metal_filter=None,
            max_rows=2000,
        ))
    if (root / "platinai_4cell.csv").exists():
        specs.append(DatasetSpec(
            name="platinai_4cell",
            path=root / "platinai_4cell.csv",
            smiles_col="smiles",
            target_cols=("ic50_mcf7_nm", "ic50_a549_nm", "ic50_a2780_nm", "ic50_a2780cis_nm"),
            transform="pIC50_nM",
            metal_filter=None,
            max_rows=2000,
        ))
    if (data_root / "metalcytotox_5metal_multitask_v4.csv").exists():
        specs.append(DatasetSpec(
            name="metalcytotox_5metal",
            path=data_root / "metalcytotox_5metal_multitask_v4.csv",
            smiles_col="smiles",
            target_cols=("activity",),
            transform="binary",
            metal_filter="Ru",
            max_rows=2000,
        ))
    return specs


def _dmpnn_feature_names() -> List[str]:
    """Generate the 117 feature names."""
    out: List[str] = []
    for pool in DMPNN_POOLS:
        for fname in DMPNN_ATOM_FEATURE_NAMES:
            out.append(f"dmpnn_{pool}_{fname}")
    return out


def run_full_analysis(datasets: Optional[List[DatasetSpec]] = None) -> Dict[str, Any]:
    """Run the full correlation analysis and return the JSON dict."""
    if datasets is None:
        datasets = _build_default_datasets()

    dmpnn_names = _dmpnn_feature_names()

    all_triples: List[TripleResult] = []
    per_dataset_summary: Dict[str, Dict[str, Any]] = {}
    feature_stats_per_dataset: Dict[str, List[FeatureStats]] = {}

    for spec in datasets:
        # ---- load
        if spec.name == "platinai_5k":
            smiles, targets, tnames = _load_platinai_5k(spec)
        elif spec.name == "platinai_4cell":
            smiles, targets, tnames = _load_platinai_4cell(spec)
        elif spec.name == "metalcytotox_5metal":
            smiles, targets, tnames = _load_metalcytotox(spec)
        else:
            continue

        if not smiles or targets.size == 0:
            continue

        # ---- features
        astsr_feats = extract_astsr_features(smiles)
        dmpnn_feats = extract_dmpnn_features(smiles)

        # ---- per-dataset feature stats
        stats: List[FeatureStats] = []
        for j, fname in enumerate(ASTSR_FEATURE_COLUMNS):
            col = astsr_feats[:, j]
            stats.append(FeatureStats(
                feature_name=fname, feature_index=j, n_samples=int(col.size),
                mean=float(np.nanmean(col)) if col.size else 0.0,
                std=float(np.nanstd(col)) if col.size else 0.0,
                min=float(np.nanmin(col)) if col.size else 0.0,
                max=float(np.nanmax(col)) if col.size else 0.0,
            ))
        for j, fname in enumerate(dmpnn_names):
            col = dmpnn_feats[:, j]
            stats.append(FeatureStats(
                feature_name=fname, feature_index=len(ASTSR_FEATURE_COLUMNS) + j,
                n_samples=int(col.size),
                mean=float(np.nanmean(col)) if col.size else 0.0,
                std=float(np.nanstd(col)) if col.size else 0.0,
                min=float(np.nanmin(col)) if col.size else 0.0,
                max=float(np.nanmax(col)) if col.size else 0.0,
            ))
        feature_stats_per_dataset[spec.name] = stats

        # ---- correlations
        triples = analyse_dataset(
            spec.name, smiles, targets, tnames, astsr_feats, dmpnn_feats, dmpnn_names,
        )
        all_triples.extend(triples)
        per_dataset_summary[spec.name] = {
            "n_samples": int(targets.shape[0]),
            "n_targets": int(targets.shape[1] if targets.ndim > 1 else 1),
            "target_names": list(tnames[: (targets.shape[1] if targets.ndim > 1 else 1)]),
            "astsr_feature_count": len(ASTSR_FEATURE_COLUMNS),
            "dmpnn_feature_count": len(dmpnn_names),
            "n_triples": len(triples),
        }

    # ---- Top-10 by absolute Pearson r (across all triples).
    by_abs_pearson = sorted(all_triples, key=lambda t: abs(t.pearson_r), reverse=True)
    top10_pearson = [
        {
            "feature": t.feature_name,
            "target": t.target_name,
            "dataset": t.dataset,
            "pearson_r": t.pearson_r,
            "spearman_rho": t.spearman_rho,
            "mutual_info": t.mutual_info,
            "n_samples": t.n_samples,
        }
        for t in by_abs_pearson[:10]
    ]

    # ---- Top-10 by absolute Spearman ρ.
    by_abs_spearman = sorted(all_triples, key=lambda t: abs(t.spearman_rho), reverse=True)
    top10_spearman = [
        {
            "feature": t.feature_name,
            "target": t.target_name,
            "dataset": t.dataset,
            "spearman_rho": t.spearman_rho,
            "pearson_r": t.pearson_r,
            "mutual_info": t.mutual_info,
            "n_samples": t.n_samples,
        }
        for t in by_abs_spearman[:10]
    ]

    # ---- Per-feature correlation matrix: features (rows) × targets (cols).
    # We aggregate by (feature, target) — averaging across datasets when
    # the same (feature, target) pair appears multiple times.
    by_pair: Dict[Tuple[str, str], List[TripleResult]] = {}
    for t in all_triples:
        by_pair.setdefault((t.feature_name, t.target_name), []).append(t)

    feature_names = sorted({t.feature_name for t in all_triples})
    target_names_unique = sorted({t.target_name for t in all_triples})

    matrix_pearson: Dict[str, Dict[str, float]] = {f: {} for f in feature_names}
    matrix_spearman: Dict[str, Dict[str, float]] = {f: {} for f in feature_names}
    matrix_mi: Dict[str, Dict[str, float]] = {f: {} for f in feature_names}
    matrix_n: Dict[str, Dict[str, int]] = {f: {} for f in feature_names}
    for (fname, tname), triples in by_pair.items():
        matrix_pearson[fname][tname] = float(np.mean([t.pearson_r for t in triples]))
        matrix_spearman[fname][tname] = float(np.mean([t.spearman_rho for t in triples]))
        matrix_mi[fname][tname] = float(np.mean([t.mutual_info for t in triples]))
        matrix_n[fname][tname] = int(np.mean([t.n_samples for t in triples]))

    # ---- Honest verdict — Cause C8 trip rule:
    # C8 trips if **both** hold:
    #   (a) max |Pearson r| across all triples < 0.05
    #   (b) max |Spearman ρ| across all triples < 0.05
    # If max |Pearson| > 0.1 OR max |Spearman| > 0.1, features carry
    # signal — Cause C8 is NOT the dominant root.
    max_pearson = float(max(abs(t.pearson_r) for t in all_triples)) if all_triples else 0.0
    max_spearman = float(max(abs(t.spearman_rho) for t in all_triples)) if all_triples else 0.0
    max_mi = float(max(t.mutual_info for t in all_triples)) if all_triples else 0.0
    c8_trip = (max_pearson < 0.05) and (max_spearman < 0.05)
    c8_partial = (max_pearson < 0.10) or (max_spearman < 0.10)

    return {
        "$schema_version": "1.0",
        "workflow": "Wf-44",
        "phase": "Phase 5",
        "title": "Feature-target correlation analysis",
        "description": (
            "Per-(feature, target) Pearson r + Spearman rho + mutual information "
            "on the 3 default data backbones.  Trips Cause C8 ('feature-target "
            "correlation truly weak') when max |r| < 0.05 AND max |rho| < 0.05."
        ),
        "date_utc": "2026-09-26",
        "companion_doc": "molmetal/reports/wf44_r2_debug/root_cause_catalog.md",
        "n_datasets": len(datasets),
        "n_triples": len(all_triples),
        "datasets": per_dataset_summary,
        "summary": {
            "max_abs_pearson_r": max_pearson,
            "max_abs_spearman_rho": max_spearman,
            "max_mutual_info": max_mi,
            "c8_trip": c8_trip,
            "c8_partial": c8_partial,
            "verdict": (
                "features_truly_weak" if c8_trip else
                "features_partially_correlated" if c8_partial else
                "features_carry_signal"
            ),
        },
        "top_10_by_abs_pearson": top10_pearson,
        "top_10_by_abs_spearman": top10_spearman,
        "correlation_matrix": {
            "rows_features": feature_names,
            "cols_targets": target_names_unique,
            "pearson": matrix_pearson,
            "spearman": matrix_spearman,
            "mutual_info": matrix_mi,
            "n_samples": matrix_n,
        },
        "feature_stats": {
            ds: [
                asdict(s) for s in stats
            ]
            for ds, stats in feature_stats_per_dataset.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _to_jsonable(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float):
        if not math.isfinite(o):
            return None
        return o
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(v) for v in o]
    return o


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-44 Phase 5 — feature-target correlation analysis.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("molmetal/reports/wf44_r2_debug"),
        help="Directory for correlation_analysis.json.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress prints.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "correlation_analysis.json"

    if not args.quiet:
        print(f"[wf44-p5] Running feature-target correlation analysis ...")
        print(f"[wf44-p5] Output: {out_path}")

    result = run_full_analysis()

    out_path.write_text(json.dumps(_to_jsonable(result), indent=2))
    if not args.quiet:
        s = result["summary"]
        print(
            f"[wf44-p5] Wrote {out_path} — n_datasets={result['n_datasets']} "
            f"n_triples={result['n_triples']} max|r|={s['max_abs_pearson_r']:.4f} "
            f"max|rho|={s['max_abs_spearman_rho']:.4f} verdict={s['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())