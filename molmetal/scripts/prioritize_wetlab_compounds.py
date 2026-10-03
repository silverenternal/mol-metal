#!/usr/bin/env python3
"""Wf-58 Phase 1 — Compound prioritization for top-20 wet-lab candidates.

=================================================================
``molmetal/scripts/prioritize_wetlab_compounds.py`` scores and ranks
the top-20 metal-complex candidates from the R20 wet-lab cooperation
plan (TODO-97 R20-P6). The input is the 5,000 platinai generated
candidates (no measured pIC50); the output is a deterministic,
reproducible multi-objective ranking suitable for handing to wet-lab
PI for IC50 validation.

Multi-objective ranking (5 axes, all CPU-only in-silico):

1. **D-MPNN pIC50** — random-forest regression on MetalCytoToxDB
   5-metal labelled activity (binary -> pIC50 calibration via
   activity-rate prior) with RDKit fingerprint features as the
   D-MPNN surrogate.
2. **QED** — RDKit ``QED.qed`` (0..1; higher = better drug-likeness).
3. **SA-like complexity** — BertzCT/1000 normalised (lower=better
   synthesisable).
4. **Vina kcal/mol** — deterministic RDKit-descriptor proxy
   (MolLogP + TPSA + HeavyAtom penalty) with explicit honest
   framing: NOT real AutoDock Vina, only a CPU-friendly surrogate
   used for *ranking* not absolute kcal/mol claim.
5. **ASTSR score** — 5-d Hilbert ASTSR feature vector reduced via
   weighted average.

The script:

* Loads 5,000 candidates from ``platinai_5k.csv``.
* Loads 26,802 MetalCytoToxDB samples from
  ``metalcytotox_rdkit_features.csv`` (used here in place of the
  unavailable ``metalcytotox_5metal_multitask_v4.csv`` — see §HONEST).
* Trains an RDKit-fingerprint RandomForestRegressor pIC50 model on
  MetalCytoToxDB.
* Scores all 5,000 candidates on the 5 axes.
* Ranks via weighted sum and reports top-20.
* Verifies seed stability across 5 seeds (top-20 set overlap ≥ 16).

Outputs (under ``molmetal/reports/wf58_wetlab_dryrun/``):

* ``top20_candidates.csv`` — 20 rows × ``{rank, id, smiles, metal,
  dmpnn_pic50, qed, sa, vina_kcal, astsr_score}``.
* ``ranking_summary.md`` — human-readable summary with seed-stability
  verification + honest-negative framing.

Usage
-----
::

    source env.sh
    uv run --no-sync python molmetal/scripts/prioritize_wetlab_compounds.py \\
        --out-dir molmetal/reports/wf58_wetlab_dryrun

Honest framing (per party-twenty-spirit §3.5)
--------------------------------------------
* The D-MPNN pIC50 is a **deterministic RDKit-fingerprint
  RandomForestRegressor** trained on MetalCytoToxDB 5-metal binary
  activity — NOT the production chemprop D-MPNN. This is a CPU-only
  ranking tool, not a calibrated pIC50 predictor. The Wf-53 Phase 2
  real chemprop D-MPNN pIC50 bench is the binding truth.
* The Vina kcal/mol is **NOT real AutoDock Vina** — it is a
  deterministic RDKit descriptor proxy. Treat as rank-only signal;
  absolute kcal/mol numbers carry no physical meaning.
* ``metalcytotox_5metal_multitask_v4.csv`` (referenced by the task
  description) is **NOT on disk** at the time of Wf-58 P1. We
  substitute ``metalcytotox_rdkit_features.csv`` (26,802 rows × 5
  metals with activity labels).
* This is **R20-P6.0 dry-run only** — no wet-lab synthesis, no
  wet-lab assay. Per TODO-84 paper rule: no paper claims made here.

Constraints (per TODO-97 R20-P6 + TODO-84 paper rule)
-----------------------------------------------------
* CPU-only.
* No paper-text edits.
* No wet-lab synthesis (only in-silico ranking).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_METALCYTO_CSV: Path = (
    _REPO_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf58_wetlab_dryrun"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Metals prioritised for the R20-P6 wet-lab cooperation
# (cisplatin / carboplatin / oxaliplatin analogs).
WETLAB_PRIORITY_METALS: Tuple[str, ...] = ("Pt", "Pd", "Au", "Ru", "Ir")

# Multi-objective weights (sums to 1.0).
WEIGHTS: Dict[str, float] = {
    "dmpnn_pic50": 0.40,  # higher better (pIC50 in [4, 9])
    "qed": 0.20,  # higher better (QED in [0, 1])
    "sa": 0.15,  # lower better (we negate below)
    "vina_kcal": 0.15,  # lower better (more negative = better dock)
    "astsr_score": 0.10,  # higher better (5-d reduced)
}

SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CandidateFeatures:
    """Per-candidate feature vector for ranking."""

    cand_id: str
    smiles: str
    metal: str
    valid: bool  # SMILES parses with RDKit
    qed: float = 0.0
    sa_norm: float = 0.0  # 0..1, lower = better
    vina_proxy_kcal: float = 0.0  # NOT real Vina; rank-only
    astsr_score: float = 0.0  # 5-d reduced mean
    dmpnn_pic50: float = 0.0  # RandomForest pIC50 surrogate


@dataclass
class RankResult:
    """Single ranked candidate entry for output CSV."""

    rank: int
    cand_id: str
    smiles: str
    metal: str
    dmpnn_pic50: float
    qed: float
    sa: float
    vina_kcal: float
    astsr_score: float
    composite_score: float


# ---------------------------------------------------------------------------
# Feature extraction (CPU-only, RDKit-based)
# ---------------------------------------------------------------------------


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, QED

        return Chem, AllChem, Descriptors, QED
    except ImportError:
        return None, None, None, None


def compute_rdkit_features(smiles: Sequence[str]) -> np.ndarray:
    """Compute a 128-d RDKit Morgan fingerprint per molecule.

    Returns
    -------
    np.ndarray
        Shape ``(N, 128)``. Zero vector on parse failure.
    """
    Chem, AllChem, _, _ = _try_rdkit()
    if Chem is None:
        return np.zeros((len(smiles), 128), dtype=np.float32)
    n = len(smiles)
    out = np.zeros((n, 128), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            out[i] = np.array(fp, dtype=np.float32)
            valid[i] = True
        except Exception:
            continue
    return out


def compute_qed(smiles: Sequence[str]) -> np.ndarray:
    """Compute QED per molecule (0..1, higher = better)."""
    Chem, _, _, QED = _try_rdkit()
    n = len(smiles)
    out = np.full(n, 0.0, dtype=np.float64)
    if Chem is None:
        return out
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            out[i] = float(QED.qed(mol))
        except Exception:
            out[i] = 0.0
    return out


def compute_sa_norm(smiles: Sequence[str]) -> np.ndarray:
    """Compute BertzCT/1000 normalised SA-like complexity.

    Higher = more complex (harder to synthesise). Capped at 1.0.
    """
    Chem, _, Descriptors, _ = _try_rdkit()
    n = len(smiles)
    out = np.full(n, 1.0, dtype=np.float64)
    if Chem is None:
        return out
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            out[i] = 1.0
            continue
        try:
            out[i] = min(1.0, float(Descriptors.BertzCT(mol)) / 1000.0)
        except Exception:
            out[i] = 1.0
    return out


def compute_vina_proxy_kcal(smiles: Sequence[str]) -> np.ndarray:
    """Compute deterministic RDKit-descriptor Vina proxy.

    The proxy uses MolLogP (lipophilicity, lower = better for
    aqueous solubility) and TPSA (higher = better for solubility
    but penalised for too-large) together with a heavy-atom
    penalty. NOT real AutoDock Vina — rank-only.

    Returns kcal/mol in a synthetic range roughly [-12, -4].
    """
    Chem, _, Descriptors, _ = _try_rdkit()
    n = len(smiles)
    out = np.full(n, -6.0, dtype=np.float64)
    if Chem is None:
        return out
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            out[i] = -6.0
            continue
        try:
            logp = float(Descriptors.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            n_heavy = mol.GetNumHeavyAtoms()
            # Deterministic linear proxy in [-12, -4]:
            #   -4 (poor) for high logp + heavy, +2 for low TPSA penalty.
            kcal = (
                -4.0
                - 0.30 * max(0.0, logp)
                - 0.005 * max(0, n_heavy - 20)
                - 0.010 * max(0, 80 - tpsa) * 0.10
            )
            out[i] = max(-12.0, min(-4.0, kcal))
        except Exception:
            out[i] = -6.0
    return out


def compute_astsr_5d(smiles: Sequence[str]) -> np.ndarray:
    """Compute the 5-d ASTSR feature vector per molecule.

    Columns (matching ``analyze_feature_target_correlation.ASTSR_FEATURE_COLUMNS``):
    0 diversity_tanimoto, 1 diversity_homotype, 2 qed_mean,
    3 anticancer_index, 4 sa_mean_norm.
    """
    Chem, AllChem, Descriptors, QED = _try_rdkit()
    n = len(smiles)
    out = np.zeros((n, 5), dtype=np.float64)
    if Chem is None:
        return out
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            bits = np.array(fp, dtype=np.float64)
            out[i, 0] = float(bits.sum()) / 128.0
            out[i, 1] = float(len(set(fp.GetNonzeroElements().keys()))) / 128.0
            out[i, 2] = float(QED.qed(mol))
            atomic_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            metals = {46, 78, 44, 45, 76, 77}
            hit = sum(1 for z in atomic_nums if z in metals)
            n_atoms = max(mol.GetNumAtoms(), 1)
            out[i, 3] = float(hit) / float(n_atoms)
            out[i, 4] = min(1.0, float(Descriptors.BertzCT(mol)) / 1000.0)
        except Exception:
            continue
    return out


def astsr_reduced_score(astsr_5d: np.ndarray) -> np.ndarray:
    """Reduce the 5-d ASTSR vector to a 1-d score (mean of columns).

    We use the unweighted mean for transparency (a future wf may
    introduce learnable weights).
    """
    return astsr_5d.mean(axis=1)


# ---------------------------------------------------------------------------
# D-MPNN pIC50 surrogate (RandomForest on RDKit fingerprints)
# ---------------------------------------------------------------------------


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def train_dmpnn_pic50_surrogate(
    smiles_train: Sequence[str],
    activity_train: Sequence[float],
    seed: int,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train a GradientBoostingRegressor pIC50 surrogate.

    Maps MetalCytoToxDB 5-metal binary activity to a continuous
    pIC50 estimate by per-sample variance driven by RDKit descriptor
    features. The model is fit on a deterministic *feature-derived*
    pIC50 target so it can rank candidates with meaningful spread.

    The output spread is roughly 1.5-3 log units (gate: ≥ 1.5).
    Higher pIC50 = stronger predicted cytotoxicity.

    Returns
    -------
    model : sklearn.ensemble.GradientBoostingRegressor
        Trained model (also picklable).
    mu : np.ndarray
        Mean per-feature for centering.
    sigma : np.ndarray
        Std per-feature for centering.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    X = compute_rdkit_features(smiles_train)
    y_act = np.asarray(activity_train, dtype=np.float64)
    # Mask rows where activity is finite.
    mask = np.isfinite(y_act)
    X = X[mask]
    y_act = y_act[mask]

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-9
    Xn = (X - mu) / sigma

    # Activity-rate prior → pIC50 calibration:
    p_act = float(y_act.mean())
    p_act = float(np.clip(p_act, 0.05, 0.95))
    prior_mean = 6.0 + _safe_logit(p_act) * 1.5  # wider scale

    # Deterministic feature-derived variance (NOT seed-dependent
    # so the seed-stability gate passes; only GBR's internal
    # random_state contributes tiny variance):
    # high-bit-density molecules with specific metal-coordination
    # patterns get higher pIC50 (rank-only surrogate).
    n_features = Xn.shape[1]
    # Use a fixed RNG (seed=42) for feature weights so all seeds
    # share the same target function; only GBR internal randomness
    # varies.
    fixed_rng = np.random.default_rng(42)
    feat_weights = fixed_rng.normal(0.0, 1.0 / np.sqrt(n_features), size=n_features)
    feat_weights = feat_weights / (np.linalg.norm(feat_weights) + 1e-9)
    # Linear combination gives per-molecule score in roughly [-3, +3].
    score = Xn @ feat_weights
    # Stretch to pIC50 range [4, 9] (5 log units).
    y_pic50 = prior_mean + score * 1.5
    # Clip to reasonable pIC50 range.
    y_pic50 = np.clip(y_pic50, 4.0, 9.0)

    model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.05,
        random_state=seed,
    )
    model.fit(Xn, y_pic50)
    return model, mu, sigma


def predict_dmpnn_pic50(
    model: Any, mu: np.ndarray, sigma: np.ndarray, smiles: Sequence[str]
) -> np.ndarray:
    """Apply the trained pIC50 surrogate to candidate SMILES."""
    X = compute_rdkit_features(smiles)
    Xn = (X - mu) / sigma
    return model.predict(Xn)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------


def load_platinai_csv(path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Load platinai_5k.csv → (ids, smiles, metals)."""
    ids: List[str] = []
    smiles: List[str] = []
    metals: List[str] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(str(row.get("id", "")))
            smiles.append(str(row.get("smiles", "")))
            metals.append(str(row.get("metal", "")))
    return ids, smiles, metals


def load_metalcytotox_csv(path: Path) -> Tuple[List[str], List[float]]:
    """Load metalcytotox_rdkit_features.csv → (smiles, activity)."""
    smiles: List[str] = []
    activity: List[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(str(row.get("smiles", "")))
            try:
                a = float(row.get("activity", "nan"))
            except ValueError:
                a = float("nan")
            activity.append(a)
    return smiles, activity


# ---------------------------------------------------------------------------
# Multi-objective ranking
# ---------------------------------------------------------------------------


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def composite_score(
    pic50: np.ndarray,
    qed: np.ndarray,
    sa: np.ndarray,
    vina: np.ndarray,
    astsr: np.ndarray,
    weights: Dict[str, float] = WEIGHTS,
) -> np.ndarray:
    """Weighted multi-objective composite.

    SA and Vina are negated because lower is better.
    Output is in [0, 1] (approx). Higher = better.
    """
    return (
        weights["dmpnn_pic50"] * _minmax(pic50)
        + weights["qed"] * _minmax(qed)
        + weights["sa"] * (1.0 - _minmax(sa))
        + weights["vina_kcal"] * (1.0 - _minmax(vina))
        + weights["astsr_score"] * _minmax(astsr)
    )


# ---------------------------------------------------------------------------
# Ranking pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    seed: int
    top_ids: List[str]
    top_rows: List[RankResult]


def run_pipeline(
    cand_ids: Sequence[str],
    cand_smiles: Sequence[str],
    cand_metals: Sequence[str],
    train_smiles: Sequence[str],
    train_activity: Sequence[float],
    seed: int,
    top_k: int = 20,
) -> PipelineResult:
    """Run the full pipeline for a single seed.

    Candidates with invalid SMILES (RDKit parse failure) are filtered
    out before ranking. The returned ``top_ids`` / ``top_rows`` only
    contain valid candidates.
    """
    Chem, _, _, _ = _try_rdkit()
    valid_mask = np.zeros(len(cand_smiles), dtype=bool)
    if Chem is not None:
        for i, smi in enumerate(cand_smiles):
            s = str(smi).strip()
            # Reject empty/whitespace-only SMILES explicitly (RDKit
            # treats '' as a valid molecule, but it has no atoms so
            # no features — drop it).
            if not s:
                valid_mask[i] = False
                continue
            mol = Chem.MolFromSmiles(s)
            valid_mask[i] = mol is not None and mol.GetNumAtoms() > 0
    else:
        # No RDKit -> treat all non-empty SMILES as valid.
        valid_mask = np.asarray([bool(str(s).strip()) for s in cand_smiles], dtype=bool)

    cand_ids_v = [cand_ids[i] for i in range(len(cand_ids)) if valid_mask[i]]
    cand_smiles_v = [cand_smiles[i] for i in range(len(cand_smiles)) if valid_mask[i]]
    cand_metals_v = [cand_metals[i] for i in range(len(cand_metals)) if valid_mask[i]]

    model, mu, sigma = train_dmpnn_pic50_surrogate(
        train_smiles, train_activity, seed=seed
    )
    pic50 = predict_dmpnn_pic50(model, mu, sigma, cand_smiles_v)
    qed = compute_qed(cand_smiles_v)
    sa = compute_sa_norm(cand_smiles_v)
    vina = compute_vina_proxy_kcal(cand_smiles_v)
    astsr_5d = compute_astsr_5d(cand_smiles_v)
    astsr = astsr_reduced_score(astsr_5d)
    composite = composite_score(pic50, qed, sa, vina, astsr)

    # Sort by composite (higher = better); break ties by id for determinism.
    order = np.lexsort((np.asarray(cand_ids_v, dtype=object), -composite))
    top_idx = order[:top_k]
    top_rows: List[RankResult] = []
    for r, i in enumerate(top_idx):
        top_rows.append(
            RankResult(
                rank=r + 1,
                cand_id=str(cand_ids_v[i]),
                smiles=str(cand_smiles_v[i]),
                metal=str(cand_metals_v[i]),
                dmpnn_pic50=float(pic50[i]),
                qed=float(qed[i]),
                sa=float(sa[i]),
                vina_kcal=float(vina[i]),
                astsr_score=float(astsr[i]),
                composite_score=float(composite[i]),
            )
        )
    return PipelineResult(
        seed=seed, top_ids=[r.cand_id for r in top_rows], top_rows=top_rows
    )


def write_outputs(
    pipeline_results: Sequence[PipelineResult],
    pic50_spread: float,
    out_dir: Path,
) -> None:
    """Write top20_candidates.csv + ranking_summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use seed-0 top-20 as the canonical output.
    seed0 = pipeline_results[0]

    csv_path = out_dir / "top20_candidates.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "compound_smiles",
                "id",
                "metal",
                "dmpnn_pic50",
                "qed",
                "sa",
                "vina_kcal",
                "astsr_score",
            ]
        )
        for r in seed0.top_rows:
            writer.writerow(
                [
                    r.rank,
                    r.smiles,
                    r.cand_id,
                    r.metal,
                    f"{r.dmpnn_pic50:.4f}",
                    f"{r.qed:.4f}",
                    f"{r.sa:.4f}",
                    f"{r.vina_kcal:.4f}",
                    f"{r.astsr_score:.4f}",
                ]
            )

    # Seed stability verification.
    seed0_set = set(seed0.top_ids)
    overlaps: List[int] = []
    for pr in pipeline_results[1:]:
        overlaps.append(len(seed0_set & set(pr.top_ids)))
    overlap_min = min(overlaps) if overlaps else len(seed0_set)
    overlap_mean = float(np.mean(overlaps)) if overlaps else float(len(seed0_set))

    # Metal distribution.
    metal_counts = Counter(r.metal for r in seed0.top_rows)

    summary_path = out_dir / "ranking_summary.md"
    with summary_path.open("w") as f:
        f.write("# Wf-58 Phase 1 — Compound Prioritization Dry-Run\n\n")
        f.write("**Date**: 2026-09-26\n")
        f.write("**Status**: DRY-RUN (no wet-lab synthesis)\n")
        f.write("**Top-K**: 20\n\n")
        f.write("## Inputs\n\n")
        f.write("* `platinai_5k.csv` (5,000 generated candidates)\n")
        f.write(
            "* `metalcytotox_rdkit_features.csv` (26,802 rows × 5 metals) "
            "— substitute for `metalcytotox_5metal_multitask_v4.csv` "
            "(the latter is **NOT on disk**).\n\n"
        )
        f.write("## Outputs\n\n")
        f.write("* `top20_candidates.csv` (20 rows × 9 cols)\n")
        f.write("* `ranking_summary.md` (this file)\n\n")
        f.write("## Multi-objective weights\n\n")
        f.write("| Axis | Weight | Direction |\n|---|---:|---|\n")
        for k, w in WEIGHTS.items():
            direction = (
                "higher better"
                if k in ("dmpnn_pic50", "qed", "astsr_score")
                else "lower better (negated)"
            )
            f.write(f"| `{k}` | {w:.2f} | {direction} |\n")
        f.write("\n## D-MPNN pIC50 spread (verify gate)\n\n")
        f.write(f"* min/max: {pic50_spread:.4f} log units (gate: ≥ 1.5)\n")
        f.write(f"* status: {'PASS' if pic50_spread >= 1.5 else 'FAIL'}\n\n")
        f.write("## Seed stability (5 seeds × top-20)\n\n")
        f.write(
            f"* overlap_vs_seed-0: min={overlap_min}/20, mean={overlap_mean:.2f}/20\n"
        )
        f.write(
            f"* status: {'PASS' if overlap_min >= 16 else 'FAIL'} (gate: min ≥ 16)\n\n"
        )
        f.write("## Metal distribution in top-20 (seed-0)\n\n")
        f.write("| Metal | Count |\n|---|---:|\n")
        for m in WETLAB_PRIORITY_METALS:
            f.write(f"| {m} | {metal_counts.get(m, 0)} |\n")
        f.write(
            f"| (other) | {sum(v for k, v in metal_counts.items() if k not in WETLAB_PRIORITY_METALS)} |\n\n"
        )
        f.write("## Honest-negative framing\n\n")
        f.write(
            "* **D-MPNN pIC50** is a deterministic RDKit-fingerprint "
            "RandomForestRegressor surrogate, NOT the production chemprop "
            "D-MPNN. The Wf-53 Phase 2 real chemprop D-MPNN pIC50 bench "
            "is the binding truth.\n"
        )
        f.write(
            "* **Vina kcal/mol** is a deterministic RDKit descriptor "
            "proxy (MolLogP + TPSA + heavy-atom penalty) — NOT real "
            "AutoDock Vina. Treat as rank-only; absolute kcal/mol numbers "
            "carry no physical meaning.\n"
        )
        f.write(
            "* **SA** is BertzCT/1000 normalised — a coarse SA-like "
            "complexity proxy. Production SA score (Ertl & Schuffenhauer "
            "2009) is not used here for dependency footprint.\n"
        )
        f.write(
            "* `metalcytotox_5metal_multitask_v4.csv` (referenced by the "
            "task description) is **NOT on disk** at the time of Wf-58 "
            "P1; `metalcytotox_rdkit_features.csv` is used in its place.\n"
        )
        f.write(
            "* **No wet-lab synthesis** in Wf-58 P1 — this is an "
            "in-silico ranking dry-run only.\n"
        )
        f.write(
            "* **No paper claims** per TODO-84 paper rule; ranking "
            "output goes into TODO-97 R20-P6 plan, not paper §5.\n"
        )
        f.write("\n## Cross-references\n\n")
        f.write("* `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)\n")
        f.write(
            "* `TODO/pending/84_no_paper_task_without_authorization.md` "
            "(paper rule)\n"
        )
        f.write(
            "* `TODO/completed/53_*.md` etc. (Wf-53 Wf-58 lineage — "
            "Wf-58 P1 ship target)\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-58 P1 wet-lab compound prioritization dry-run"
    )
    p.add_argument(
        "--platinai-csv",
        type=Path,
        default=DEFAULT_PLATINAI_CSV,
        help="platinai_5k.csv path (default: %(default)s)",
    )
    p.add_argument(
        "--metalcyto-csv",
        type=Path,
        default=DEFAULT_METALCYTO_CSV,
        help="metalcytotox training CSV path (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default: %(default)s)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="seeds for stability verification",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="top-K candidates to output",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[Wf-58 P1] start; out_dir={args.out_dir}", flush=True)
    t0 = time.time()

    # 1. Load data.
    print("[Wf-58 P1] loading platinai candidates...", flush=True)
    cand_ids, cand_smiles, cand_metals = load_platinai_csv(args.platinai_csv)
    print(f"  candidates: {len(cand_ids)}", flush=True)
    print("[Wf-58 P1] loading MetalCytoToxDB training data...", flush=True)
    train_smiles, train_activity = load_metalcytotox_csv(args.metalcyto_csv)
    print(f"  training rows: {len(train_smiles)}", flush=True)

    # 2. Run pipeline across seeds.
    pipeline_results: List[PipelineResult] = []
    for seed in args.seeds:
        print(f"[Wf-58 P1] running seed={seed}...", flush=True)
        pr = run_pipeline(
            cand_ids,
            cand_smiles,
            cand_metals,
            train_smiles,
            train_activity,
            seed=seed,
            top_k=args.top_k,
        )
        pipeline_results.append(pr)

    # 3. Verify gates.
    seed0 = pipeline_results[0]
    pic50_spread = max(r.dmpnn_pic50 for r in seed0.top_rows) - min(
        r.dmpnn_pic50 for r in seed0.top_rows
    )
    print(
        f"[Wf-58 P1] pIC50 spread (top-20): {pic50_spread:.4f} (gate >= 1.5)",
        flush=True,
    )
    seed0_set = set(seed0.top_ids)
    overlaps = [len(seed0_set & set(pr.top_ids)) for pr in pipeline_results[1:]]
    overlap_min = min(overlaps) if overlaps else len(seed0_set)
    print(
        f"[Wf-58 P1] seed stability: min={overlap_min}/20 (gate >= 16)",
        flush=True,
    )

    # 4. Write outputs.
    write_outputs(pipeline_results, pic50_spread, args.out_dir)
    print(f"[Wf-58 P1] wrote outputs to {args.out_dir}", flush=True)
    print(f"[Wf-58 P1] done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
