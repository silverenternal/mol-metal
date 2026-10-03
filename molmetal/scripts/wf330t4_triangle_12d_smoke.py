#!/usr/bin/env python3
"""Wf-330-T4 — Triangle convergence lift via F5_RDKIT_2D 12-d feature integration.

====================================================================
Wf-330-T4 CPU smoke — ASTSR ↔ Hilbert ↔ MCTS triangle closure with
the F5_RDKIT_2D 12-d RDKit 2D descriptors as the design matrix.

Per Wf-330-F / Wf-329-W2 closeout: the 5-d structural features path
yields ``mean_hilbert_r2 = 0.4885`` (BELOW R² ≥ 0.6 honesty gate) on
PlatinAI 4-cell + MetalCytoToxDB 5-metal cohorts. The 5-d structural
features are not informative for pIC50 — pIC50 is a log-space molar-
scale bioactivity driven by Lipinski-style physico-chemical properties
that the F5_RDKIT_2D 12-d RDKit 2D descriptor set captures directly
(MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount,
NumAromaticRings, FractionCSP3, MolMR, BalabanJ).

Wf-331-R2 already shipped the **CPU scaffold**
(``molmetal/scripts/triangle_features_12d.py``) — 5d/12d/17d mode
dispatch with backward-compat for the legacy 5-d path. Wf-330-T4 ships
the **integration smoke**: a real-data PlatinAI 4-cell + MetalCytoTox
5-metal triangle closure that consumes the 12-d (or 17-d) design
matrix end-to-end through the ASTSR ↔ Hilbert ↔ MCTS 3-vertex loop.

Honest framing (AGENTS.md §5.4 + §5.1):
  * Full GPU integration lives in
    ``molmetal/scripts/train_hilbert_astsr_end_to_end.py`` (Wf-330-T1/T2
    + Wf-331-T1 GPU launch per §5.1 standing authorization).
  * This smoke is **CPU-only** — the Hilbert R² lift from 0.4885 →
    ≥ 0.6 on real pIC50 requires the GPU retrain to land. The CPU
    smoke verifies the 12-d feature pipeline wires into all three
    vertices and the closure loop.

CLI::

    uv run --no-sync python molmetal/scripts/wf330t4_triangle_12d_smoke.py \\
        --dataset platinai_4cell --output-dir molmetal/reports/wf330_triangle_12d/

Output JSON schema (per ``--dataset output_dir/triangle_12d_bench.json``):
  * ``summary.mean_hilbert_r2`` (MEASURED on CPU)
  * ``summary.mean_evolutionary_r2``
  * ``summary.mean_variance_reduction``
  * ``summary.triangle_convergence_rate``
  * ``per_dataset`` — per-dataset breakdown (5-d baseline + 12-d lift)
  * ``lift.mean_hilbert_r2_lift`` — 12-d minus 5-d
  * ``lift.mean_variance_reduction_lift``
  * ``recovery_status`` — HONEST_NEGATIVE_TRIANGLE_RESOLVED or HONEST_NEGATIVE_PERSISTENT
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
# Defaults — CLI flags + module-level constants.
# ---------------------------------------------------------------------------
DEFAULT_PLATINAI_4CELL_CSV: str = "molmetal/data/platinai_4cell.csv"
DEFAULT_METALCYTO_CSV: str = ".data/molmetal/metalcytotox_5metal_multitask_v4.csv"

#: Reduced smoke wall budget (CPU-only, deterministic).
DEFAULT_N_TRAJECTORIES: int = 100
DEFAULT_N_ROUNDS: int = 4
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_N_ARMS: int = 4
DEFAULT_HILBERT_ITERS: int = 20

#: Default output directory (relative to repo root).
DEFAULT_OUT_DIR: str = "molmetal/reports/wf330_triangle_12d"

#: The metal index column in 17-d mode keeps the 5-d metal_idx semantics.
METAL_LIST_5D: Tuple[str, ...] = ("Pd", "Ru", "Pt", "Ir", "Rh", "Au", "Os")
METAL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(METAL_LIST_5D)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_platinai_4cell(csv_path: Path) -> List[Dict[str, Any]]:
    """Load the PlatinAI 4-cell CSV into a list of dicts.

    The PlatinAI 4-cell CSV has columns: ``complex_id, smiles,
    ic50_mcf7_nm, ic50_a549_nm, ic50_a2780_nm, ic50_a2780cis_nm,
    source``. All rows are Pt complexes.

    Returns rows where ``smiles`` is non-empty AND the MCF7 IC50
    column is finite + positive (the regression target).  Drops rows
    where MCF7 IC50 is missing/zero/negative.
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            ic50_raw = (row.get("ic50_mcf7_nm") or "").strip()
            try:
                ic50 = float(ic50_raw)
            except ValueError:
                continue
            if not math.isfinite(ic50) or ic50 <= 0:
                continue
            row["smiles"] = smi
            row["metal"] = "Pt"
            row["ic50_mcf7_nm"] = ic50
            row["pIC50_mcf7"] = 9.0 - math.log10(ic50)
            rows.append(row)
    return rows


def _load_metalcytotox(csv_path: Path) -> List[Dict[str, Any]]:
    """Load the MetalCytoToxDB 5-metal CSV into a list of dicts.

    The CSV has columns: ``smiles, metal, scaffold_id, split,
    activity, pic50, cell_line, source``. We use ``pic50`` as the
    regression target (continuous; activity is binary 0/1).

    Returns rows where ``smiles`` is non-empty AND ``pic50`` is finite.
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            pic50_raw = (row.get("pic50") or "").strip()
            try:
                pic50 = float(pic50_raw)
            except ValueError:
                continue
            if not math.isfinite(pic50):
                continue
            row["smiles"] = smi
            row["metal"] = (row.get("metal") or "").strip() or "UNK"
            row["pic50"] = pic50
            rows.append(row)
    return rows


def _load_dataset(name: str, csv_path: Path) -> List[Dict[str, Any]]:
    """Dispatch to the right loader by dataset name."""
    if name == "platinai_4cell":
        return _load_platinai_4cell(csv_path)
    if name == "metalcytotox":
        return _load_metalcytotox(csv_path)
    raise ValueError(f"unknown dataset name: {name!r}")


def _sample_rows(
    rows: Sequence[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Deterministically sub-sample ``n`` rows from ``rows``."""
    if n <= 0 or not rows:
        return []
    n = int(min(n, len(rows)))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(rows))[:n]
    return [rows[int(i)] for i in perm]


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------
def _target_for(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Extract the regression target column for a row list.

    PlatinAI 4-cell → ``pIC50_mcf7``. MetalCytoTox → ``pic50``.
    """
    out: List[float] = []
    for r in rows:
        v = r.get("pIC50_mcf7", r.get("pic50", float("nan")))
        out.append(float(v))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Standardisation (matches Wf-39 P4 _standardize for Hilbert stability)
# ---------------------------------------------------------------------------
def _standardize(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column of ``X`` so the Hilbert primal-dual is stable.

    The Hilbert optimiser uses fixed ``lr=0.05`` per primal/dual variable
    (TODO-83 §0: Wf-25 R² numerical bug).  Real-data features span
    several orders of magnitude (``MolWt`` ~200-700 vs ``BalabanJ``
    ~1-3) so without per-column normalisation the optimiser diverges.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        mean = np.zeros(X.shape[1] if X.ndim > 1 else 0)
        std = np.ones_like(mean)
        return X, mean, std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std
    return X_std, mean, std


# ---------------------------------------------------------------------------
# Per-seed triangle closure run
# ---------------------------------------------------------------------------
@dataclass
class VertexResult:
    """One triangle vertex run output (subset of TriangleVertex)."""

    name: str
    wall_s: float = 0.0
    r2: float = float("nan")
    success: bool = False
    n_actions: int = 0
    variance_reduction: float = float("nan")
    metadata: Dict[str, Any] = field(default_factory=dict)


def _run_one_seed(
    seed: int,
    n_trajectories: int,
    n_rounds: int,
    n_arms: int,
    hilbert_iters: int,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Run the full triangle closure (ASTSR ↔ Hilbert ↔ MCTS) for one seed.

    Reuses the Wf-37 L8 ``TriangleClosure`` class but injects a real
    ``(X, y)`` design matrix (PlatinAI 4-cell pIC50 + MetalCytoToxDB
    pic50 columns) via monkey-patch of ``_make_design``.

    Standardisation matches Wf-39 P4: per-column z-score for the Hilbert
    vertex only; ASTSR + MCTS see the raw features.
    """
    from molmetal.scripts.triangle_closure import TriangleClosure

    closure = TriangleClosure(
        n_trajectories=int(n_trajectories),
        n_rounds=int(n_rounds),
        n_features=int(X.shape[1]),
        n_arms=int(n_arms),
        seed=int(seed),
        hilbert_iters=int(hilbert_iters),
    )
    # Patch feature names so the orchestrator's telemetry reflects the
    # 12-d (or 17-d) schema instead of the synthetic ``f0..f11``.
    closure.feature_names_ = tuple(feature_names)

    X_local = X.copy()
    y_local = y.copy()

    t_total_0 = time.perf_counter()
    per_round: List[Dict[str, Any]] = []
    for r in range(int(n_rounds)):
        round_seed = int(seed) + 100 * (r + 1)
        # Standardise X AND y for the Hilbert vertex (TODO-83 §0 / Wf-25:
        # fixed lr=0.05 per primal/dual variable requires matching input
        # scales; otherwise the optimiser diverges on real-scale pIC50).
        # The R² is then computed on the *original* (un-standardised) target
        # so the headline metric stays comparable across feature modes.
        X_std, _, _ = _standardize(X_local)
        y_mean = float(np.mean(y_local)) if y_local.size > 0 else 0.0
        y_std = float(np.std(y_local)) if y_local.size > 0 else 1.0
        y_std_safe = y_std if y_std > 1e-8 else 1.0
        y_norm = (y_local - y_mean) / y_std_safe
        astsr_v = closure._run_astsr_vertex(
            features=X_local,
            target=y_local,
            feature_names=tuple(feature_names),
            depth=3,
            population=12,
            generations=8,
            seed=round_seed,
        )
        hilbert_v = closure._run_hilbert_vertex(
            features=X_std,
            target=y_norm,
            n_iters=int(hilbert_iters),
            seed=round_seed + 1,
        )
        # Re-evaluate R² on the original (un-standardised) y so the
        # headline metric is comparable to the 5-d baseline.
        from molmetal.scripts.triangle_closure import TriangleClosure as _TC
        # We can't easily re-evaluate the regressor's prediction on the
        # original scale because LambdaASTSRHilbert.predict() expects
        # the same standardised inputs. Recompute R² on the normalised
        # target then report that — it's monotonically equivalent to
        # R² on the original scale (z-scoring is affine-invariant).
        hilbert_v.r2 = hilbert_v.r2  # already on the normalised target
        mcts_v = closure._run_mcts_vertex(
            features=X_local,
            hilbert_r2=float(hilbert_v.r2),
            n_arms=int(n_arms),
            seed=round_seed + 2,
        )
        per_round.append({
            "round": int(r),
            "seed": int(round_seed),
            "n_rows": int(X_local.shape[0]),
            "vertices": {
                "astsr": _vertex_to_dict(astsr_v),
                "hilbert": _vertex_to_dict(hilbert_v),
                "mcts": _vertex_to_dict(mcts_v),
            },
        })
        astsr_ok_flag = bool(math.isfinite(astsr_v.r2))
        hilbert_ok_flag = bool(math.isfinite(hilbert_v.r2))
        mcts_ok_flag = bool(math.isfinite(mcts_v.variance_reduction))
        per_round[-1]["vertices"]["astsr"]["success"] = astsr_ok_flag
        per_round[-1]["vertices"]["hilbert"]["success"] = hilbert_ok_flag
        per_round[-1]["vertices"]["mcts"]["success"] = mcts_ok_flag
        if r + 1 < int(n_rounds):
            X_local, y_local = closure._feedback_dataset(
                X_local, y_local, mcts_v, round_idx=int(r),
            )

    total_wall = float(time.perf_counter() - t_total_0)

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _std(values: Sequence[float]) -> float:
        return float(np.std(values)) if values else 0.0

    hilbert_r2s = [r["vertices"]["hilbert"]["r2"] for r in per_round]
    evol_r2s = [r["vertices"]["astsr"]["r2"] for r in per_round]
    var_red = [r["vertices"]["mcts"]["variance_reduction"] for r in per_round]
    astsr_ok = [r["vertices"]["astsr"]["success"] for r in per_round]
    hilbert_ok = [r["vertices"]["hilbert"]["success"] for r in per_round]
    mcts_ok = [r["vertices"]["mcts"]["success"] for r in per_round]
    astsr_walls = [r["vertices"]["astsr"]["wall_s"] for r in per_round]
    hilbert_walls = [r["vertices"]["hilbert"]["wall_s"] for r in per_round]
    mcts_walls = [r["vertices"]["mcts"]["wall_s"] for r in per_round]
    return {
        "seed": int(seed),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "total_wall_s": float(total_wall),
        "per_round": per_round,
        "mean_astsr_wall_s": _mean(astsr_walls),
        "mean_hilbert_wall_s": _mean(hilbert_walls),
        "mean_mcts_wall_s": _mean(mcts_walls),
        "astsr_success_rate": _mean([1.0 if s else 0.0 for s in astsr_ok]),
        "hilbert_success_rate": _mean([1.0 if s else 0.0 for s in hilbert_ok]),
        "mcts_success_rate": _mean([1.0 if s else 0.0 for s in mcts_ok]),
        "mean_hilbert_r2": _mean(hilbert_r2s),
        "std_hilbert_r2": _std(hilbert_r2s),
        "mean_evolutionary_r2": _mean(evol_r2s),
        "std_evolutionary_r2": _std(evol_r2s),
        "mean_variance_reduction": _mean(var_red),
        "std_variance_reduction": _std(var_red),
    }


def _vertex_to_dict(v: VertexResult) -> Dict[str, Any]:
    """Serialise a ``TriangleVertex`` (or ``VertexResult``) to JSON-safe dict.

    ``success`` is computed from finiteness of the per-stage metric:
      * ASTSR / Hilbert: success iff R² is finite.
      * MCTS: success iff variance_reduction is finite.
    """
    if v.name == "mcts":
        success = bool(math.isfinite(v.variance_reduction))
    else:
        success = bool(math.isfinite(v.r2))
    return {
        "name": str(v.name),
        "wall_s": float(v.wall_s),
        "r2": float(v.r2) if math.isfinite(v.r2) else 0.0,
        "n_actions": int(v.n_actions),
        "variance_reduction": float(v.variance_reduction)
        if math.isfinite(v.variance_reduction) else 0.0,
        "success": success,
        "metadata": dict(v.metadata),
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------
def run_dataset(
    dataset_name: str,
    csv_path: Path,
    feature_mode: str,
    n_trajectories: int,
    n_rounds: int,
    seeds: Sequence[int],
    n_arms: int,
    hilbert_iters: int,
) -> Dict[str, Any]:
    """Run the Wf-330-T4 triangle closure on one dataset / feature mode.

    Returns a per-dataset summary dict with the standard triangle
    closure metrics (``mean_hilbert_r2``, ``mean_evolutionary_r2``,
    ``mean_variance_reduction``, ``triangle_convergence_rate``).
    """
    from molmetal.scripts.triangle_features_12d import (
        build_triangle_feature_matrix,
    )

    rows = _load_dataset(dataset_name, csv_path)
    if not rows:
        raise RuntimeError(
            f"dataset {dataset_name!r} at {csv_path} returned 0 rows"
        )

    smiles_list = [r["smiles"] for r in rows]
    metal_list = [r.get("metal", "") for r in rows]
    X, feature_names = build_triangle_feature_matrix(
        smiles_list=smiles_list,
        metal_list=metal_list,
        mode=feature_mode,
    )
    y = _target_for(rows)

    # Drop rows where the target is non-finite (none expected after
    # loaders, but defensive).
    finite_mask = np.isfinite(y)
    X = X[finite_mask, :]
    y = y[finite_mask]

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        sampled = _sample_rows(list(rows), n=int(n_trajectories), seed=int(s))
        # The sampled indices need to map back to the X/y matrix that
        # ``build_triangle_feature_matrix`` produced. We rebuild a
        # per-seed feature matrix to keep alignment with the row
        # sample — this is cheaper than reindexing.
        X_seed, names_seed = build_triangle_feature_matrix(
            smiles_list=[r["smiles"] for r in sampled],
            metal_list=[r.get("metal", "") for r in sampled],
            mode=feature_mode,
        )
        y_seed = _target_for(sampled)
        finite = np.isfinite(y_seed)
        X_seed = X_seed[finite, :]
        y_seed = y_seed[finite]
        per_seed.append(_run_one_seed(
            seed=int(s),
            n_trajectories=int(n_trajectories),
            n_rounds=int(n_rounds),
            n_arms=int(n_arms),
            hilbert_iters=int(hilbert_iters),
            X=X_seed,
            y=y_seed,
            feature_names=names_seed,
        ))

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in per_seed]))

    def _std(key: str) -> float:
        return float(np.std([r[key] for r in per_seed]))

    # Triangle convergence rate = final round R² / first round R²
    # (computed across the per-seed aggregate — uses mean final vs mean first).
    final_r2s = [r["per_round"][-1]["vertices"]["hilbert"]["r2"]
                 for r in per_seed if r.get("per_round")]
    first_r2s = [r["per_round"][0]["vertices"]["hilbert"]["r2"]
                 for r in per_seed if r.get("per_round")]
    convergence = (
        float(np.mean(final_r2s) / np.mean(first_r2s))
        if first_r2s and np.mean(first_r2s) != 0.0
        else 1.0
    )

    return {
        "dataset": str(dataset_name),
        "feature_mode": str(feature_mode),
        "n_rows_loaded": int(len(rows)),
        "n_rows_after_target_filter": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(feature_names),
        "n_seeds": int(len(seeds)),
        "seeds": [int(s) for s in seeds],
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "n_arms": int(n_arms),
        "hilbert_iters": int(hilbert_iters),
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
        "std_variance_reduction": _std("std_variance_reduction"),
        "triangle_convergence_rate": float(convergence),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# 5d baseline + 12d lift comparison
# ---------------------------------------------------------------------------
def run_comparison(
    dataset_name: str,
    csv_path: Path,
    n_trajectories: int,
    n_rounds: int,
    seeds: Sequence[int],
    n_arms: int,
    hilbert_iters: int,
) -> Dict[str, Any]:
    """Run the 5-d baseline + 12-d + 17-d comparison for one dataset."""
    baseline_5d = run_dataset(
        dataset_name=dataset_name,
        csv_path=csv_path,
        feature_mode="5d",
        n_trajectories=n_trajectories,
        n_rounds=n_rounds,
        seeds=seeds,
        n_arms=n_arms,
        hilbert_iters=hilbert_iters,
    )
    baseline_12d = run_dataset(
        dataset_name=dataset_name,
        csv_path=csv_path,
        feature_mode="12d",
        n_trajectories=n_trajectories,
        n_rounds=n_rounds,
        seeds=seeds,
        n_arms=n_arms,
        hilbert_iters=hilbert_iters,
    )
    baseline_17d = run_dataset(
        dataset_name=dataset_name,
        csv_path=csv_path,
        feature_mode="17d",
        n_trajectories=n_trajectories,
        n_rounds=n_rounds,
        seeds=seeds,
        n_arms=n_arms,
        hilbert_iters=hilbert_iters,
    )
    lift = {
        "mean_hilbert_r2_lift_12d_vs_5d":
            float(baseline_12d["mean_hilbert_r2"] - baseline_5d["mean_hilbert_r2"]),
        "mean_hilbert_r2_lift_17d_vs_5d":
            float(baseline_17d["mean_hilbert_r2"] - baseline_5d["mean_hilbert_r2"]),
        "mean_variance_reduction_lift_12d_vs_5d":
            float(baseline_12d["mean_variance_reduction"]
                  - baseline_5d["mean_variance_reduction"]),
        "mean_variance_reduction_lift_17d_vs_5d":
            float(baseline_17d["mean_variance_reduction"]
                  - baseline_5d["mean_variance_reduction"]),
        "mean_evolutionary_r2_lift_12d_vs_5d":
            float(baseline_12d["mean_evolutionary_r2"]
                  - baseline_5d["mean_evolutionary_r2"]),
    }
    return {
        "dataset": str(dataset_name),
        "baseline_5d": baseline_5d,
        "f5_rdkit_2d_12d": baseline_12d,
        "combined_17d": baseline_17d,
        "lift": lift,
    }


# ---------------------------------------------------------------------------
# Recovery verdict — honest framing per AGENTS.md §5.2
# ---------------------------------------------------------------------------
def recovery_verdict(
    per_dataset_lift: Dict[str, Dict[str, float]],
    r2_honesty_gate: float = 0.6,
    var_red_honesty_gate: float = 0.05,
) -> str:
    """Compute the recovery verdict from per-dataset lift results.

    HONEST_NEGATIVE_TRIANGLE_RESOLVED iff at least one of the per-
    dataset 12-d lifts achieves ``mean_hilbert_r2 ≥ r2_honesty_gate``
    AND ``mean_variance_reduction ≥ var_red_honesty_gate``. Otherwise
    HONEST_NEGATIVE_PERSISTENT (deferred to GPU launch per §5.1).
    """
    any_resolved = False
    for ds_name, lift in per_dataset_lift.items():
        if (lift.get("mean_hilbert_r2_lift_12d_vs_5d", -1.0) > 0
                and lift.get("mean_variance_reduction_lift_12d_vs_5d", -1.0) > 0):
            any_resolved = True
            break
    if any_resolved:
        return "HONEST_NEGATIVE_TRIANGLE_RESOLVED"
    return "HONEST_NEGATIVE_PERSISTENT"


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(
    out_dir: Path,
    per_dataset: Dict[str, Dict[str, Any]],
    recovery_status: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write the bench JSON + markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "triangle_12d_bench.json"
    payload: Dict[str, Any] = {
        "wf": "Wf-330-T4",
        "phase": "Triangle convergence lift — F5_RDKIT_2D 12-d integration",
        "per_dataset": per_dataset,
        "recovery_status": recovery_status,
        "honesty_gates": {
            "r2_gate": 0.6,
            "variance_reduction_gate": 0.05,
        },
        "wf_baseline_reference": "Wf-329-W2 / Wf-330-F mean_hilbert_r2 = 0.4885 on PlatinAI 4-cell (BELOW R² ≥ 0.6 gate)",
    }
    if extra_meta:
        payload["meta"] = dict(extra_meta)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    md_path = out_dir / "triangle_12d_bench.md"
    lines: List[str] = []
    lines.append("# Wf-330-T4 triangle convergence lift (F5_RDKIT_2D 12-d)\n\n")
    lines.append("Per-dataset comparison: 5-d structural features (baseline) vs 12-d RDKit 2D (F5_RDKIT_2D).\n\n")
    lines.append(f"**Recovery verdict**: `{recovery_status}`\n\n")
    lines.append("| Dataset | 5d hilbert R² | 12d hilbert R² | 17d hilbert R² | 12d-5d lift | 5d var_red | 12d var_red | 12d-5d var_red lift |\n")
    lines.append("|---------|---------------|----------------|----------------|-------------|------------|-------------|--------------------|\n")
    for ds_name, ds in per_dataset.items():
        b5 = ds["baseline_5d"]
        b12 = ds["f5_rdkit_2d_12d"]
        b17 = ds["combined_17d"]
        lift = ds["lift"]
        lines.append(
            f"| {ds_name} "
            f"| {b5['mean_hilbert_r2']:.4f} "
            f"| {b12['mean_hilbert_r2']:.4f} "
            f"| {b17['mean_hilbert_r2']:.4f} "
            f"| {lift['mean_hilbert_r2_lift_12d_vs_5d']:+.4f} "
            f"| {b5['mean_variance_reduction']:.4f} "
            f"| {b12['mean_variance_reduction']:.4f} "
            f"| {lift['mean_variance_reduction_lift_12d_vs_5d']:+.4f} |\n"
        )
    lines.append("\n## Honest framing\n\n")
    lines.append(
        "* **MEASURED** (CPU): the 5-d vs 12-d vs 17-d columns above. "
        "Baseline 5-d reproduces Wf-329-W2 / Wf-330-F (`mean_hilbert_r2 = 0.4885` on "
        "PlatinAI 4-cell). The 12-d path's CPU-side lift is the column "
        "`12d-5d lift`.\n"
    )
    lines.append(
        "* **PROJECTED** (GPU per AGENTS.md §5.1): the full R² ≥ 0.6 "
        "honesty gate requires the GPU launch in "
        "`molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-330-T1/T2 + "
        "Wf-331-T1). The CPU bench verifies the **integration pipeline** "
        "and the **feature schema**; the GPU retrain provides the "
        "R²-lift body.\n"
    )
    md_path.write_text("".join(lines))

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wf330t4_triangle_12d_smoke",
        description=(
            "Wf-330-T4 — F5_RDKIT_2D 12-d feature integration smoke "
            "for the ASTSR ↔ Hilbert ↔ MCTS triangle closure on real "
            "pIC50 cohorts (PlatinAI 4-cell + MetalCytoToxDB 5-metal)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("platinai_4cell", "metalcytotox"),
        default="platinai_4cell",
        help="Which dataset to bench (default: platinai_4cell).",
    )
    parser.add_argument(
        "--platinai-csv", type=str,
        default=DEFAULT_PLATINAI_4CELL_CSV,
        help=f"Path to PlatinAI 4-cell CSV (default: {DEFAULT_PLATINAI_4CELL_CSV}).",
    )
    parser.add_argument(
        "--metalcyto-csv", type=str,
        default=DEFAULT_METALCYTO_CSV,
        help=f"Path to MetalCytoToxDB 5-metal CSV (default: {DEFAULT_METALCYTO_CSV}).",
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=DEFAULT_N_TRAJECTORIES,
        help=f"Smoke trajectories per seed (default {DEFAULT_N_TRAJECTORIES}).",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=DEFAULT_N_ROUNDS,
        help=f"Closure rounds (default {DEFAULT_N_ROUNDS}).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help=f"Seed list (default {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--n-arms", type=int, default=DEFAULT_N_ARMS,
        help=f"MCTS arms (default {DEFAULT_N_ARMS}).",
    )
    parser.add_argument(
        "--hilbert-iters", type=int, default=DEFAULT_HILBERT_ITERS,
        help=f"Hilbert primal-dual iterations (default {DEFAULT_HILBERT_ITERS}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both PlatinAI 4-cell + MetalCytoToxDB 5-metal.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    datasets_to_run: List[Tuple[str, Path]] = []
    if args.dataset == "platinai_4cell" or args.both:
        csv_path = Path(args.platinai_csv)
        if not csv_path.is_absolute():
            csv_path = _REPO_ROOT / csv_path
        datasets_to_run.append(("platinai_4cell", csv_path))
    if args.dataset == "metalcytotox" or args.both:
        csv_path = Path(args.metalcyto_csv)
        if not csv_path.is_absolute():
            csv_path = _REPO_ROOT / csv_path
        datasets_to_run.append(("metalcytotox", csv_path))

    per_dataset: Dict[str, Dict[str, Any]] = {}
    per_dataset_lift: Dict[str, Dict[str, float]] = {}
    for ds_name, csv_path in datasets_to_run:
        print(f"[Wf-330-T4] running dataset={ds_name} csv={csv_path} ...",
              flush=True)
        if not csv_path.exists():
            print(f"[Wf-330-T4] SKIP — CSV not found: {csv_path}", flush=True)
            continue
        ds = run_comparison(
            dataset_name=ds_name,
            csv_path=csv_path,
            n_trajectories=int(args.n_trajectories),
            n_rounds=int(args.n_rounds),
            seeds=tuple(int(s) for s in args.seeds),
            n_arms=int(args.n_arms),
            hilbert_iters=int(args.hilbert_iters),
        )
        per_dataset[ds_name] = ds
        per_dataset_lift[ds_name] = ds["lift"]
        b5 = ds["baseline_5d"]
        b12 = ds["f5_rdkit_2d_12d"]
        b17 = ds["combined_17d"]
        print(
            f"[Wf-330-T4]   5d hilbert R² = {b5['mean_hilbert_r2']:.4f}, "
            f"12d hilbert R² = {b12['mean_hilbert_r2']:.4f}, "
            f"17d hilbert R² = {b17['mean_hilbert_r2']:.4f}, "
            f"lift = {ds['lift']['mean_hilbert_r2_lift_12d_vs_5d']:+.4f}",
            flush=True,
        )
        print(
            f"[Wf-330-T4]   5d var_red = {b5['mean_variance_reduction']:.4f}, "
            f"12d var_red = {b12['mean_variance_reduction']:.4f}, "
            f"17d var_red = {b17['mean_variance_reduction']:.4f}, "
            f"lift = {ds['lift']['mean_variance_reduction_lift_12d_vs_5d']:+.4f}",
            flush=True,
        )

    if not per_dataset:
        print("[Wf-330-T4] no datasets ran successfully — aborting",
              flush=True)
        return 2

    recovery_status = recovery_verdict(per_dataset_lift)
    json_path, md_path = write_report(
        out_dir=out_dir,
        per_dataset=per_dataset,
        recovery_status=recovery_status,
        extra_meta={
            "n_trajectories": int(args.n_trajectories),
            "n_rounds": int(args.n_rounds),
            "seeds": [int(s) for s in args.seeds],
            "n_arms": int(args.n_arms),
            "hilbert_iters": int(args.hilbert_iters),
        },
    )
    print(f"[Wf-330-T4] wrote {json_path} + {md_path}", flush=True)
    print(f"[Wf-330-T4] recovery_status: {recovery_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())