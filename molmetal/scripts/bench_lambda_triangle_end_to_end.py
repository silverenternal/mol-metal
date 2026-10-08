"""Wf-39 Phase 4 — Lambda ASTSR ↔ Hilbert ↔ MCTS triangle end-to-end bench.

================================================================
Wf-39 L4: real-data triangle closure benchmark (TODO-83 §4.4 + Wf-39)
================================================================
This script extends the synthetic Wf-37 L8 :mod:`triangle_closure`
orchestrator into a **real-data** end-to-end benchmark:

* Input data: ``molmetal/data/platinai_5k.csv`` (5,000 metallodrug
  SMILES, 7 metals Pd/Ru/Pt/Ir/Rh/Au/Os).
* Features: 5 structural descriptors derived deterministically from
  each SMILES (length, heavy-atom count, branch count, charge count,
  metal index) — no RDKit dependency so the bench is robust on
  any host.
* Multi-seed: ``--seeds 5`` runs the full triangle closure
  ``len(seeds)`` times with disjoint RNG streams; per-stage wall
  time + success rate are reported **per seed** and **across seeds**.
* Scale: ``--n-trajectories 100`` × 5 seeds × PlatinAI 5K subset
  (= 500 row-bench cycles), per-stage.

What this benchmark measures
----------------------------
For each seed ``s ∈ seeds`` and each round ``r ∈ {0, ..., n_rounds-1}``
we run the triangle closure on ``n_trajectories`` PlatinAI rows
sampled with seed ``s`` and:

* record the per-stage wall time (ASTSR / Hilbert / MCTS)
* record the per-stage "success" flag (did the vertex produce a
  finite, well-defined output?)
* record the per-stage R² (Hilbert vs evolutionary) and variance
  reduction (MCTS)

Across seeds we aggregate:

* ``mean_per_stage_wall_s`` — average per-stage wall time
* ``per_stage_success_rate`` — fraction of stage-runs that succeeded
* ``convergence_across_seeds`` — mean ``R²`` lift Hilbert→ASTSR,
  averaged over seeds
* ``per_seed_summary`` — per-seed summary block for downstream audit

Inputs
------
* :mod:`molmetal.scripts.triangle_closure` (Wf-37 L8 ship) — the
  closure orchestrator; reused without modification.
* ``molmetal/data/platinai_5k.csv`` (5000 rows × 5 cols).

Outputs
-------
* ``molmetal/reports/wf39_triangle/triangle_bench.json`` — full
  per-seed × per-round benchmark.
* ``molmetal/reports/wf39_triangle/per_seed.csv`` — per-seed
  one-row-per-seed CSV (handy for spreadsheet audits).
* ``molmetal/reports/wf39_triangle/triangle_bench.md`` — human-
  readable summary.

Honest framing
--------------
* The PlatinAI 5K CSV has empty ``ic50_nm`` labels for all rows
  (the original PlatinAI predicted-activity sheets are not in
  this 5K subset).  The bench therefore uses a **deterministic
  synthetic target** constructed from the 5 structural features
  (``y = sum(weights[i] * X[:,i]) + 0.05 noise``) — this lets
  us report an honest ``R²`` between a known linear target and
  the triangle's fit/predict output.  The features are REAL
  PlatinAI-derived; only the target is synthetic.
* "Success" for the ASTSR / Hilbert vertices is ``True`` when
  the per-stage ``R²`` is finite and not NaN.  For MCTS, success
  is ``True`` when the variance reduction is finite.
* The bench is CPU-only and deterministic given the seeds.

CLI::

    # Default: 100 trajectories × 5 rounds × 5 seeds (= 25 round-runs)
    uv run --no-sync python molmetal/scripts/bench_lambda_triangle_end_to_end.py

    # Smoke (10 trajectories × 2 rounds × 2 seeds = 4 round-runs)
    uv run --no-sync python molmetal/scripts/bench_lambda_triangle_end_to_end.py \\
        --n-trajectories 10 --n-rounds 2 --seeds 0 1

Lit anchors
-----------
* Wf-37 L8 (TODO-83 §4.4) — base triangle closure.
* TODO-83 §4.4 + §6 — R20 Lambda algorithm enhancement plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_5k.csv"
DEFAULT_OUT_DIR: str = "molmetal/reports/wf39_triangle"

DEFAULT_N_TRAJECTORIES: int = 100
DEFAULT_N_ROUNDS: int = 5
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_ARMS: int = 4
DEFAULT_HILBERT_ITERS: int = 20

#: Canonical metal list for the one-hot feature.
METAL_LIST: Tuple[str, ...] = ("Pd", "Ru", "Pt", "Ir", "Rh", "Au", "Os")
METAL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(METAL_LIST)}

#: Feature names — five structural descriptors (no RDKit required).
FEATURE_NAMES: Tuple[str, ...] = (
    "smiles_length",
    "n_heavy_atoms",
    "n_branches",
    "n_charges",
    "metal_idx",
)


# ---------------------------------------------------------------------------
# Wf-330-T2 / Wf-A16 — MetalCytoToxDB 5-metal cohort registry
#
# Extends the bench with a second dataset family (the MetalCytoToxDB 5-metal
# cohort Ru/Ir/Rh/Os/Re) plus a 12-d RDKit-2D feature mode (F5_RDKIT_2D, from
# Wf-49 P1 / Wf-249).  The registry exposes the per-cohort metadata the
# Wf-330-T2 tests pin (feature mode, metal list/index, PROJECTED ledger).
# ---------------------------------------------------------------------------
DEFAULT_METALCYTO_5METAL_CSV: str = (
    ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
)

METALCYTO_5METAL_LIST: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
METALCYTO_5METAL_TO_IDX: Dict[str, int] = {
    m: i for i, m in enumerate(METALCYTO_5METAL_LIST)
}

#: PROJECTED per-metal ``mean_hilbert_r2`` ledger (skill mol-metal-triangle-closure
#: §4.3).  These are PROJECTED targets, NOT measured (AGENTS.md §5.2).
METALCYTO_5METAL_PROJECTED_R2: Dict[str, float] = {
    "Ru": 0.62,
    "Ir": 0.61,
    "Rh": 0.60,
    "Os": 0.58,
    "Re": 0.57,
}
METALCYTO_5METAL_PROJECTED_MEAN: float = 0.622

#: F5_RDKIT_2D — 12 RDKit 2-D descriptor names (Wf-49 P1 / Wf-249 ship surface).
F5_RDKIT_2D_FEATURE_NAMES: Tuple[str, ...] = (
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
)


def _f5_rdkit_2d_features(smiles: str) -> Optional[List[float]]:
    """Extract the 12-d F5_RDKIT_2D descriptor vector from *smiles*.

    Returns ``None`` when RDKit is unavailable or the SMILES fails to parse
    (callers fall back to the 5-d structural path).  The descriptor order
    matches :data:`F5_RDKIT_2D_FEATURE_NAMES`.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
        from rdkit.Chem import RDConfig  # noqa: F401
    except Exception:  # noqa: BLE001 - RDKit not installed ⇒ caller fallback
        return None
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        # BalabanJ comes from the RDKit Contrib SASA/GraphDescriptors module;
        # fall back to 0.0 when it is missing rather than failing the row.
        try:
            from rdkit.Chem import rdMolDescriptors as _rmd  # noqa: F401

            balabanj = float(Descriptors.BalabanJ(mol))
        except Exception:  # noqa: BLE001
            balabanj = 0.0
        return [
            float(Descriptors.MolWt(mol)),
            float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Crippen.MolMR(mol)),
            float(balabanj),
        ]
    except Exception:  # noqa: BLE001 - any descriptor failure ⇒ fallback
        return None


def _load_metalcytotox_5metal(csv_path: Path) -> List[Dict[str, str]]:
    """Load the MetalCytoToxDB 5-metal cohort CSV into a list of row dicts.

    Rows whose ``metal`` is outside :data:`METALCYTO_5METAL_LIST` or whose
    ``smiles`` is empty are dropped.  Uses the shared metal_cytotox loader so
    the v3/v4 column-synonym handling is applied uniformly.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MetalCytoToxDB 5-metal CSV not found: {csv_path} "
            f"(repo root = {_REPO_ROOT})"
        )
    try:
        from molmetal.data.metal_cytotox import load_metal_cytotox

        df = load_metal_cytotox(csv_path)
        rows: List[Dict[str, str]] = []
        for _, r in df.iterrows():
            metal = str(r.get("metal", "")).strip()
            smi = str(r.get("smiles", "")).strip()
            if not smi or metal not in METALCYTO_5METAL_LIST:
                continue
            rows.append({"smiles": smi, "metal": metal})
        return rows
    except Exception:  # noqa: BLE001 - fall back to a plain CSV read
        rows = []
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or row.get("SMILES_Ligands") or "").strip()
                metal = (row.get("metal") or row.get("Metal") or "").strip()
                if not smi or metal not in METALCYTO_5METAL_LIST:
                    continue
                rows.append({"smiles": smi, "metal": metal})
        return rows


# ---------------------------------------------------------------------------
# Data loading + feature extraction
# ---------------------------------------------------------------------------
def _smiles_features(smiles: str, metal: str) -> Tuple[float, ...]:
    """Deterministically extract 5 structural features from a SMILES.

    The features are intentionally simple (string-based) so the bench
    has no RDKit dependency.  They are deterministic, finite, and
    capture the main variation we want the triangle closure to
    recover: size, branching, charge density, and metal identity.

    Parameters
    ----------
    smiles : str
        The metallodrug SMILES (may be empty for malformed rows).
    metal : str
        One of :data:`METAL_LIST` (or any string — unknown metals map
        to ``metal_idx = -1`` which the bench tolerates).

    Returns
    -------
    tuple of float
        ``(smiles_length, n_heavy_atoms, n_branches, n_charges,
        metal_idx)`` — always finite, always 5-D.
    """
    if not isinstance(smiles, str):
        smiles = str(smiles) if smiles is not None else ""
    smiles = smiles.strip()
    # 1. Length (chars).
    smiles_length = float(len(smiles))
    # 2. Heavy atoms — uppercase / lowercase letters that look like
    #    organic atoms (we count A–Z and a–z excluding the bracket
    #    markers).  Simple heuristic: split by non-letter chars.
    n_heavy = 0
    for ch in smiles:
        if ch.isalpha():
            n_heavy += 1
    # 3. Branches — count "(" and ")".
    n_branches = float(smiles.count("(") + smiles.count(")"))
    # 4. Charges — count "+" and "-".
    n_charges = float(smiles.count("+") + smiles.count("-"))
    # 5. Metal index.
    metal_idx = float(METAL_TO_IDX.get(str(metal).strip(), -1))
    return (
        smiles_length,
        float(n_heavy),
        n_branches,
        n_charges,
        metal_idx,
    )


def _load_platinai_5k(csv_path: Path) -> List[Dict[str, str]]:
    """Load the PlatinAI 5K CSV into a list of dicts.

    Empty rows (no SMILES) and rows with empty ``smiles`` are
    silently dropped.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PlatinAI CSV not found: {csv_path} " f"(repo root = {_REPO_ROOT})"
        )
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            row["smiles"] = smi
            row.setdefault("metal", "")
            rows.append(row)
    return rows


#: Dataset registry consumed by the triangle bench + its Wf-330-T2 tests.
#: Defined AFTER both loaders so the ``loader`` entries resolve.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "platinai_5k": {
        "csv": DEFAULT_PLATINAI_CSV,
        "metal_list": METAL_LIST,
        "metal_to_idx": METAL_TO_IDX,
        "feature_mode": "f5_structural_5d",
        "feature_names": FEATURE_NAMES,
        "loader": _load_platinai_5k,
    },
    "metalcytotox_5metal": {
        "csv": DEFAULT_METALCYTO_5METAL_CSV,
        "metal_list": METALCYTO_5METAL_LIST,
        "metal_to_idx": METALCYTO_5METAL_TO_IDX,
        "feature_mode": "f5_rdkit_2d",
        "feature_names": F5_RDKIT_2D_FEATURE_NAMES,
        "loader": _load_metalcytotox_5metal,
        "projected_r2": METALCYTO_5METAL_PROJECTED_R2,
        "projected_mean": METALCYTO_5METAL_PROJECTED_MEAN,
    },
}


def _sample_rows(
    rows: Sequence[Dict[str, str]],
    n: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Deterministically sub-sample ``n`` rows from ``rows``."""
    if n <= 0:
        return []
    n = int(min(n, len(rows)))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(rows))[:n]
    return [rows[int(i)] for i in perm]


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column of ``X`` so Hilbert primal-dual is stable.

    Real-data features have wildly different scales (e.g. ``smiles_length``
    ranges 0–200 while ``metal_idx`` is 0–6).  The Hilbert primal-dual
    optimiser uses a fixed learning rate ``lr=0.05`` per primal/dual
    variable — without per-column normalisation the optimiser diverges
    on real-scale features (TODO-83 §0 honest framing: Wf-25 R²
    numerical bug).  This helper:

    * Subtracts the column mean
    * Divides by ``max(std, 1e-8)`` to avoid zero-variance columns

    Returns
    -------
    (X_std, mean, std)
        ``X_std`` is the standardised matrix, ``mean`` and ``std`` are
        the per-column statistics so callers can inverse-transform the
        prediction back to original scale.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0) if X.size > 0 else np.zeros(X.shape[1])
    std = np.std(X, axis=0) if X.size > 0 else np.ones(X.shape[1])
    std = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std
    return X_std, mean, std


def _build_design_matrix(
    sampled_rows: Sequence[Dict[str, str]],
    seed: int,
    feature_mode: str = "string5d",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic ``(X, y)`` design matrix from sampled rows.

    The synthetic target is drawn from the **standardised** features so
    the noise floor is comparable to the ASTSR / Hilbert fit targets.

    Parameters
    ----------
    feature_mode
        * ``"string5d"`` (default) — the legacy 5-d ``_smiles_features``
          string-heuristic matrix.
        * ``"f5_rdkit_2d"`` — the 12-d F5_RDKIT_2D RDKit descriptor
          matrix (:data:`F5_RDKIT_2D_FEATURE_NAMES`).  Rows whose SMILES
          cannot be parsed (RDKit missing / bad parse) fall back to a
          deterministic zero vector so the matrix shape stays stable.

    Returns
    -------
    (X, y, weights)
        ``X`` — ``(n, d)`` feature matrix (``d`` depends on
        ``feature_mode``).
        ``y`` — ``(n,)`` synthetic target drawn from a known linear
        combination of ``X`` plus small Gaussian noise.  The weights
        are returned so callers can audit the ground truth.
    """
    n = len(sampled_rows)
    if feature_mode == "f5_rdkit_2d":
        names: Tuple[str, ...] = F5_RDKIT_2D_FEATURE_NAMES
        X = np.zeros((n, len(names)), dtype=float)
        for i, row in enumerate(sampled_rows):
            feats = _f5_rdkit_2d_features(row.get("smiles", ""))
            if feats is not None:
                X[i, :] = feats
    elif feature_mode in ("string5d", "f5_structural_5d"):
        names = FEATURE_NAMES
        X = np.zeros((n, len(names)), dtype=float)
        for i, row in enumerate(sampled_rows):
            feats = _smiles_features(row.get("smiles", ""), row.get("metal", ""))
            X[i, :] = feats
    else:
        raise ValueError(
            f"unknown feature_mode {feature_mode!r}; expected "
            "'string5d' or 'f5_rdkit_2d'"
        )
    rng = np.random.default_rng(int(seed) + 17)
    weights = rng.standard_normal(X.shape[1])
    # Standardise X so the target lives in the same scale as the
    # features the ASTSR / Hilbert regressors see (post-standardisation).
    X_std, _, _ = _standardize(X)
    y = X_std @ weights + 0.05 * rng.standard_normal(n)
    return X, y, weights


# ---------------------------------------------------------------------------
# Stage runners (re-use the Wf-37 triangle closure primitives)
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """Telemetry for one stage run inside one round for one seed."""

    name: str
    wall_s: float = 0.0
    success: bool = False
    r2: float = float("nan")
    variance_reduction: float = float("nan")
    metadata: Dict[str, Any] = field(default_factory=dict)


def _run_one_seed(
    seed: int,
    n_trajectories: int,
    n_rounds: int,
    n_arms: int,
    hilbert_iters: int,
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    feature_names: Tuple[str, ...] = FEATURE_NAMES,
) -> Dict[str, Any]:
    """Run the full triangle closure for one seed.

    Reuses :class:`molmetal.scripts.triangle_closure.TriangleClosure`
    end-to-end so the contract matches Wf-37 L8 exactly.  Per-stage
    success is determined by finiteness of the per-stage metric.

    Feature standardisation
    -----------------------
    The Hilbert primal-dual optimiser uses fixed ``lr=0.05`` per
    primal/dual variable.  Real-data features span 4 orders of
    magnitude (``smiles_length`` ∈ [27, 137], ``n_charges`` ∈ [4, 14]),
    so the raw features cause the optimiser to diverge (TODO-83 §0:
    Wf-25 R² numerical bug).  We **standardise per round** so the
    optimiser sees zero-mean unit-variance inputs.  The ASTSR arm
    gets the original features (it is invariant to monotone feature
    transforms); the MCTS arm also gets the original features (its
    variance signal is computed on the column-mean projection).
    """
    # Local import keeps the module importable even if any of the
    # three upstream modules is broken.
    from molmetal.scripts.triangle_closure import TriangleClosure

    closure = TriangleClosure(
        n_trajectories=int(n_trajectories),
        n_rounds=int(n_rounds),
        n_features=int(X_seed.shape[1]),
        n_arms=int(n_arms),
        seed=int(seed),
        hilbert_iters=int(hilbert_iters),
    )
    # Build a copy so the orchestrator's internal feedback generator
    # doesn't mutate the caller's array.
    X = X_seed.copy()
    y = y_seed.copy()

    t_total_0 = time.perf_counter()
    per_round: List[Dict[str, Any]] = []
    for r in range(int(n_rounds)):
        round_seed = int(seed) + 100 * (r + 1)
        # Standardise X for the Hilbert vertex only.  ASTSR and MCTS
        # see the original features.
        X_std, _, _ = _standardize(X)
        astsr_v = closure._run_astsr_vertex(
            features=X,
            target=y,
            feature_names=feature_names,
            depth=3,
            population=12,
            generations=8,
            seed=round_seed,
        )
        hilbert_v = closure._run_hilbert_vertex(
            features=X_std,
            target=y,
            n_iters=int(hilbert_iters),
            seed=round_seed + 1,
        )
        mcts_v = closure._run_mcts_vertex(
            features=X,
            hilbert_r2=float(hilbert_v.r2),
            n_arms=int(n_arms),
            seed=round_seed + 2,
        )
        # Per-stage success flags.
        astsr_ok = bool(math.isfinite(astsr_v.r2))
        hilbert_ok = bool(math.isfinite(hilbert_v.r2))
        mcts_ok = bool(math.isfinite(mcts_v.variance_reduction))
        per_round.append(
            {
                "round": int(r),
                "seed": int(round_seed),
                "n_rows": int(X.shape[0]),
                "vertices": {
                    "astsr": {
                        "name": "astsr",
                        "wall_s": float(astsr_v.wall_s),
                        "r2": float(astsr_v.r2) if math.isfinite(astsr_v.r2) else 0.0,
                        "success": astsr_ok,
                        "metadata": dict(astsr_v.metadata),
                    },
                    "hilbert": {
                        "name": "hilbert",
                        "wall_s": float(hilbert_v.wall_s),
                        "r2": float(hilbert_v.r2)
                        if math.isfinite(hilbert_v.r2)
                        else 0.0,
                        "success": hilbert_ok,
                        "metadata": dict(hilbert_v.metadata),
                    },
                    "mcts": {
                        "name": "mcts",
                        "wall_s": float(mcts_v.wall_s),
                        "n_actions": int(mcts_v.n_actions),
                        "variance_reduction": float(mcts_v.variance_reduction),
                        "success": mcts_ok,
                        "metadata": dict(mcts_v.metadata),
                    },
                },
            }
        )
        # Closure feedback to next round.
        if r + 1 < int(n_rounds):
            X, y = closure._feedback_dataset(X, y, mcts_v, round_idx=int(r))

    total_wall = float(time.perf_counter() - t_total_0)

    # Aggregate per-stage metrics across rounds for this seed.
    astsr_walls = [r["vertices"]["astsr"]["wall_s"] for r in per_round]
    hilbert_walls = [r["vertices"]["hilbert"]["wall_s"] for r in per_round]
    mcts_walls = [r["vertices"]["mcts"]["wall_s"] for r in per_round]
    astsr_success = [r["vertices"]["astsr"]["success"] for r in per_round]
    hilbert_success = [r["vertices"]["hilbert"]["success"] for r in per_round]
    mcts_success = [r["vertices"]["mcts"]["success"] for r in per_round]
    return {
        "seed": int(seed),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "total_wall_s": float(total_wall),
        "per_round": per_round,
        "mean_astsr_wall_s": float(np.mean(astsr_walls)) if astsr_walls else 0.0,
        "mean_hilbert_wall_s": float(np.mean(hilbert_walls)) if hilbert_walls else 0.0,
        "mean_mcts_wall_s": float(np.mean(mcts_walls)) if mcts_walls else 0.0,
        "astsr_success_rate": float(np.mean(astsr_success)) if astsr_success else 0.0,
        "hilbert_success_rate": float(np.mean(hilbert_success))
        if hilbert_success
        else 0.0,
        "mcts_success_rate": float(np.mean(mcts_success)) if mcts_success else 0.0,
        "mean_hilbert_r2": float(
            np.mean([r["vertices"]["hilbert"]["r2"] for r in per_round])
        )
        if per_round
        else 0.0,
        "mean_evolutionary_r2": float(
            np.mean([r["vertices"]["astsr"]["r2"] for r in per_round])
        )
        if per_round
        else 0.0,
        "mean_variance_reduction": float(
            np.mean([r["vertices"]["mcts"]["variance_reduction"] for r in per_round])
        )
        if per_round
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_report(
    platinai_csv: str = DEFAULT_PLATINAI_CSV,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_rounds: int = DEFAULT_N_ROUNDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_arms: int = DEFAULT_N_ARMS,
    hilbert_iters: int = DEFAULT_HILBERT_ITERS,
    dataset: str = "platinai_5k",
) -> Dict[str, Any]:
    """Build the multi-seed end-to-end triangle bench report.

    Parameters
    ----------
    dataset
        * ``"platinai_5k"`` (default) — the PlatinAI 5K cohort with the
          legacy 5-d structural features.
        * ``"metalcytotox_5metal"`` — the MetalCytoToxDB 5-metal cohort
          (Ru/Ir/Rh/Os/Re) with the F5_RDKIT_2D 12-d feature mode.  This
          path additionally emits a ``per_metal_breakdown`` block with
          the MEASURED-vs-PROJECTED ledger per metal.
    """
    if dataset == "metalcytotox_5metal":
        return _build_metalcytotox_report(
            csv_path=Path(DEFAULT_METALCYTO_5METAL_CSV),
            n_trajectories=int(n_trajectories),
            n_rounds=int(n_rounds),
            seeds=seeds,
            n_arms=int(n_arms),
            hilbert_iters=int(hilbert_iters),
        )

    csv_path = Path(platinai_csv)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path
    rows = _load_platinai_5k(csv_path)
    if not rows:
        raise RuntimeError(f"PlatinAI CSV at {csv_path} is empty")

    # Pre-sample a deterministic per-seed design matrix per seed.
    per_seed_design: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for s in seeds:
        sampled = _sample_rows(rows, n=int(n_trajectories), seed=int(s))
        X, y, w = _build_design_matrix(sampled, seed=int(s))
        per_seed_design[int(s)] = (X, y, w)

    per_seed_results: List[Dict[str, Any]] = []
    for s in seeds:
        X, y, _ = per_seed_design[int(s)]
        per_seed_results.append(
            _run_one_seed(
                seed=int(s),
                n_trajectories=int(n_trajectories),
                n_rounds=int(n_rounds),
                n_arms=int(n_arms),
                hilbert_iters=int(hilbert_iters),
                X_seed=X,
                y_seed=y,
            )
        )

    # Cross-seed aggregate.
    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in per_seed_results]))

    def _std(key: str) -> float:
        return float(np.std([r[key] for r in per_seed_results]))

    summary = {
        "dataset": "platinai_5k",
        "feature_mode": "string5d",
        "n_seeds": int(len(seeds)),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "n_arms": int(n_arms),
        "hilbert_iters": int(hilbert_iters),
        "platinai_rows_loaded": int(len(rows)),
        "mean_astsr_wall_s": _mean("mean_astsr_wall_s"),
        "mean_hilbert_wall_s": _mean("mean_hilbert_wall_s"),
        "mean_mcts_wall_s": _mean("mean_mcts_wall_s"),
        "std_astsr_wall_s": _std("mean_astsr_wall_s"),
        "std_hilbert_wall_s": _std("mean_hilbert_wall_s"),
        "std_mcts_wall_s": _std("mean_mcts_wall_s"),
        "astsr_success_rate": _mean("astsr_success_rate"),
        "hilbert_success_rate": _mean("hilbert_success_rate"),
        "mcts_success_rate": _mean("mcts_success_rate"),
        "mean_hilbert_r2": _mean("mean_hilbert_r2"),
        "std_hilbert_r2": _std("mean_hilbert_r2"),
        "mean_evolutionary_r2": _mean("mean_evolutionary_r2"),
        "std_evolutionary_r2": _std("mean_evolutionary_r2"),
        "mean_variance_reduction": _mean("mean_variance_reduction"),
        "std_variance_reduction": _std("mean_variance_reduction"),
    }
    return {
        "feature_names": list(FEATURE_NAMES),
        "metal_list": list(METAL_LIST),
        "platinai_csv": str(csv_path),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed_results,
        "summary": summary,
    }


def _build_metalcytotox_report(
    csv_path: Path,
    n_trajectories: int,
    n_rounds: int,
    seeds: Sequence[int],
    n_arms: int,
    hilbert_iters: int,
) -> Dict[str, Any]:
    """Build the report for the MetalCytoToxDB 5-metal cohort.

    Runs the triangle closure per-metal (each metal's rows form their own
    design matrix under the F5_RDKIT_2D 12-d feature mode) and emits a
    ``per_metal_breakdown`` block with the honest MEASURED-vs-PROJECTED
    ledger.  MEASURED ``mean_hilbert_r2`` is compared against the
    PROJECTED per-metal target from the skill ledger.
    """
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path
    rows = _load_metalcytotox_5metal(csv_path)
    if not rows:
        raise RuntimeError(f"MetalCytoToxDB CSV at {csv_path} is empty")

    # Group rows by metal.
    by_metal: Dict[str, List[Dict[str, str]]] = {m: [] for m in METALCYTO_5METAL_LIST}
    for r in rows:
        m = r.get("metal", "")
        if m in by_metal:
            by_metal[m].append(r)

    per_metal_breakdown: Dict[str, Dict[str, Any]] = {}
    for metal in METALCYTO_5METAL_LIST:
        mrows = by_metal.get(metal, [])
        if not mrows:
            per_metal_breakdown[metal] = {
                "mean_hilbert_r2": 0.0,
                "projected_target": float(METALCYTO_5METAL_PROJECTED_R2[metal]),
                "MEASURED_vs_PROJECTED": "MEASURED=BELOW_TARGET",
                "n_rows": 0,
            }
            continue
        r2s: List[float] = []
        for s in seeds:
            sampled = _sample_rows(mrows, n=int(n_trajectories), seed=int(s))
            if not sampled:
                continue
            X, y, _ = _build_design_matrix(
                sampled, seed=int(s), feature_mode="f5_rdkit_2d"
            )
            res = _run_one_seed(
                seed=int(s),
                n_trajectories=int(n_trajectories),
                n_rounds=int(n_rounds),
                n_arms=int(n_arms),
                hilbert_iters=int(hilbert_iters),
                X_seed=X,
                y_seed=y,
                feature_names=F5_RDKIT_2D_FEATURE_NAMES,
            )
            r2s.append(float(res["mean_hilbert_r2"]))
        measured = float(np.mean(r2s)) if r2s else 0.0
        target = float(METALCYTO_5METAL_PROJECTED_R2[metal])
        verdict = (
            "MEASURED=ON_TARGET" if measured >= target else "MEASURED=BELOW_TARGET"
        )
        per_metal_breakdown[metal] = {
            "mean_hilbert_r2": measured,
            "projected_target": target,
            "MEASURED_vs_PROJECTED": verdict,
            "n_rows": int(len(mrows)),
        }

    cohort_measured = float(
        np.mean([v["mean_hilbert_r2"] for v in per_metal_breakdown.values()])
    )
    summary = {
        "dataset": "metalcytotox_5metal",
        "feature_mode": "f5_rdkit_2d",
        "n_seeds": int(len(seeds)),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "n_arms": int(n_arms),
        "hilbert_iters": int(hilbert_iters),
        "rows_loaded": int(len(rows)),
        "mean_hilbert_r2": cohort_measured,
        "projected_cohort_mean_hilbert_r2": float(METALCYTO_5METAL_PROJECTED_MEAN),
    }
    return {
        "feature_names": list(F5_RDKIT_2D_FEATURE_NAMES),
        "metal_list": list(METALCYTO_5METAL_LIST),
        "dataset": "metalcytotox_5metal",
        "feature_mode": "f5_rdkit_2d",
        "csv": str(csv_path),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "seeds": [int(s) for s in seeds],
        "per_metal_breakdown": per_metal_breakdown,
        "summary": summary,
    }


def write_report(
    out_dir: str,
    report: Dict[str, Any],
    *,
    also_markdown: bool = True,
    also_csv: bool = True,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Write the bench report to ``out_dir/triangle_bench.json`` etc."""
    out = Path(out_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "triangle_bench.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    csv_path: Optional[Path] = None
    if also_csv:
        csv_path = out / "per_seed.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "seed",
                    "n_trajectories",
                    "n_rounds",
                    "total_wall_s",
                    "mean_astsr_wall_s",
                    "mean_hilbert_wall_s",
                    "mean_mcts_wall_s",
                    "astsr_success_rate",
                    "hilbert_success_rate",
                    "mcts_success_rate",
                    "mean_hilbert_r2",
                    "mean_evolutionary_r2",
                    "mean_variance_reduction",
                ]
            )
            for r in report.get("per_seed", []):
                w.writerow(
                    [
                        r.get("seed", -1),
                        r.get("n_trajectories", -1),
                        r.get("n_rounds", -1),
                        f"{r.get('total_wall_s', 0.0):.4f}",
                        f"{r.get('mean_astsr_wall_s', 0.0):.4f}",
                        f"{r.get('mean_hilbert_wall_s', 0.0):.4f}",
                        f"{r.get('mean_mcts_wall_s', 0.0):.4f}",
                        f"{r.get('astsr_success_rate', 0.0):.4f}",
                        f"{r.get('hilbert_success_rate', 0.0):.4f}",
                        f"{r.get('mcts_success_rate', 0.0):.4f}",
                        f"{r.get('mean_hilbert_r2', 0.0):.4f}",
                        f"{r.get('mean_evolutionary_r2', 0.0):.4f}",
                        f"{r.get('mean_variance_reduction', 0.0):.4f}",
                    ]
                )

    md_path: Optional[Path] = None
    if also_markdown:
        md_path = out / "triangle_bench.md"
        summary = report.get("summary", {}) or {}
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# Wf-39 L4 triangle closure end-to-end benchmark\n\n")
            fh.write(
                f"n_seeds = {summary.get('n_seeds', '?')}, "
                f"n_trajectories = {summary.get('n_trajectories', '?')}, "
                f"n_rounds = {summary.get('n_rounds', '?')}, "
                f"n_arms = {summary.get('n_arms', '?')}, "
                f"hilbert_iters = {summary.get('hilbert_iters', '?')}\n\n"
            )
            fh.write(
                f"PlatinAI rows loaded: "
                f"{summary.get('platinai_rows_loaded', '?')}\n\n"
            )
            fh.write("## Per-stage wall time (mean ± std, s)\n\n")
            fh.write("| stage | mean wall s | std wall s |\n")
            fh.write("|-------|------------:|-----------:|\n")
            for stage_key, stage_label in (
                ("astsr", "ASTSR"),
                ("hilbert", "Hilbert"),
                ("mcts", "MCTS"),
            ):
                fh.write(
                    f"| {stage_label} "
                    f"| {summary.get(f'mean_{stage_key}_wall_s', 0.0):.4f} "
                    f"| {summary.get(f'std_{stage_key}_wall_s', 0.0):.4f} |\n"
                )
            fh.write("\n## Per-stage success rate (across seeds)\n\n")
            fh.write("| stage | success rate |\n")
            fh.write("|-------|-------------:|\n")
            for stage_key, stage_label in (
                ("astsr", "ASTSR"),
                ("hilbert", "Hilbert"),
                ("mcts", "MCTS"),
            ):
                fh.write(
                    f"| {stage_label} "
                    f"| {summary.get(f'{stage_key}_success_rate', 0.0):.4f} |\n"
                )
            fh.write("\n## Regression metrics (mean ± std)\n\n")
            fh.write("| metric | mean | std |\n")
            fh.write("|--------|-----:|----:|\n")
            for key, label in (
                ("hilbert_r2", "Hilbert R²"),
                ("evolutionary_r2", "evolutionary R²"),
                ("variance_reduction", "variance reduction"),
            ):
                fh.write(
                    f"| {label} "
                    f"| {summary.get(f'mean_{key}', 0.0):.4f} "
                    f"| {summary.get(f'std_{key}', 0.0):.4f} |\n"
                )
            fh.write("\n## Per-seed detail\n\n")
            fh.write("| seed | total wall | hilbert R² | evol R² | var red |\n")
            fh.write("|-----:|-----------:|-----------:|--------:|--------:|\n")
            for r in report.get("per_seed", []):
                fh.write(
                    f"| {r.get('seed', '?')} "
                    f"| {r.get('total_wall_s', 0.0):.3f} "
                    f"| {r.get('mean_hilbert_r2', 0.0):.4f} "
                    f"| {r.get('mean_evolutionary_r2', 0.0):.4f} "
                    f"| {r.get('mean_variance_reduction', 0.0):.4f} |\n"
                )
    return json_path, csv_path, md_path


# ---------------------------------------------------------------------------
# Smoke verification
# ---------------------------------------------------------------------------
def smoke_run(
    platinai_csv: str = DEFAULT_PLATINAI_CSV,
) -> bool:
    """CPU smoke that returns True iff the bench runs end-to-end."""
    try:
        report = build_report(
            platinai_csv=platinai_csv,
            n_trajectories=10,
            n_rounds=2,
            seeds=(0, 1),
            n_arms=4,
            hilbert_iters=10,
        )
    except Exception:
        return False
    if not report:
        return False
    summary = report.get("summary", {}) or {}
    if not isinstance(summary.get("mean_hilbert_wall_s"), float):
        return False
    if not isinstance(summary.get("mean_astsr_wall_s"), float):
        return False
    if not isinstance(summary.get("mean_mcts_wall_s"), float):
        return False
    if not isinstance(summary.get("astsr_success_rate"), float):
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point — return 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(
        description=(
            "Wf-39 L4 — Lambda ASTSR ↔ Hilbert ↔ MCTS triangle "
            "end-to-end bench on PlatinAI 5K (CPU-only)."
        ),
    )
    parser.add_argument(
        "--platinai-csv",
        type=str,
        default=DEFAULT_PLATINAI_CSV,
        help="PlatinAI CSV path (default: " + DEFAULT_PLATINAI_CSV + ").",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
        help=f"Rows per seed (default {DEFAULT_N_TRAJECTORIES}).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Closure rounds per seed (default {DEFAULT_N_ROUNDS}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"RNG seeds (default {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--n-arms",
        type=int,
        default=DEFAULT_N_ARMS,
        help=f"MCTS arm count (default {DEFAULT_N_ARMS}).",
    )
    parser.add_argument(
        "--hilbert-iters",
        type=int,
        default=DEFAULT_HILBERT_ITERS,
        help=f"Hilbert iterations per round " f"(default {DEFAULT_HILBERT_ITERS}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output dir (default {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the canonical small smoke and exit with 0/1.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-seed progress output.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        ok = smoke_run(platinai_csv=args.platinai_csv)
        print("Wf-39 L4 triangle e2e: smoke " f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    report = build_report(
        platinai_csv=args.platinai_csv,
        n_trajectories=int(args.n_trajectories),
        n_rounds=int(args.n_rounds),
        seeds=tuple(int(s) for s in args.seeds),
        n_arms=int(args.n_arms),
        hilbert_iters=int(args.hilbert_iters),
    )

    summary = report.get("summary", {}) or {}
    if not args.quiet:
        print(
            "Wf-39 L4 triangle e2e: "
            f"seeds={summary.get('n_seeds', 0)} "
            f"traj={summary.get('n_trajectories', 0)} "
            f"rounds={summary.get('n_rounds', 0)} "
            f"plat_rows={summary.get('platinai_rows_loaded', 0)}"
        )
        print(
            "  per-stage mean wall s: "
            f"ASTSR={summary.get('mean_astsr_wall_s', 0.0):.4f}  "
            f"Hilbert={summary.get('mean_hilbert_wall_s', 0.0):.4f}  "
            f"MCTS={summary.get('mean_mcts_wall_s', 0.0):.4f}"
        )
        print(
            "  per-stage success rate: "
            f"ASTSR={summary.get('astsr_success_rate', 0.0):.4f}  "
            f"Hilbert={summary.get('hilbert_success_rate', 0.0):.4f}  "
            f"MCTS={summary.get('mcts_success_rate', 0.0):.4f}"
        )
        print(
            "  regression metrics: "
            f"Hilbert R²={summary.get('mean_hilbert_r2', 0.0):.4f}  "
            f"evol R²={summary.get('mean_evolutionary_r2', 0.0):.4f}  "
            f"var_red={summary.get('mean_variance_reduction', 0.0):.4f}"
        )

    json_path, csv_path, md_path = write_report(
        out_dir=str(args.out_dir),
        report=report,
        also_markdown=True,
        also_csv=True,
    )
    print(
        f"Wrote {json_path}"
        + (f" + {csv_path}" if csv_path else "")
        + (f" + {md_path}" if md_path else "")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
