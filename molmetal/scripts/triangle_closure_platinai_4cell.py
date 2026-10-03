#!/usr/bin/env python3
"""Wf-330-T1 — Triangle closure end-to-end on PlatinAI 4-cell (per cell-line).

================================================================
Wf-330-T1: ASTSR ↔ Hilbert ↔ MCTS triangle closure on PlatinAI 4-cell
================================================================
This script is the **R21-P4 per-cell triangle closure ship** (TODO-95
cross-ref Wf-329-W2).  It extends the Wf-37 L8 (synthetic) + Wf-39 P4
(platinai_5k) triangle closure ships to **PlatinAI 4-cell** with **real
pIC50 labels per cell line**:

* Input data: ``molmetal/data/platinai_4cell.csv``
  (4078 rows × 4 cell lines: mcf7, a549, a2780, a2780cis).
* Features: **F5_RDKIT_2D 12-d RDKit 2D descriptor** per SMILES
  (Wf-49 P2 ship — see :mod:`molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell`).
* Target: ``y = -log10(IC50_µM × 1e-6)`` — real signed pIC50 per cell
  line, NOT a synthetic target.

The recovery target is the Wf-329-W2 honest-negative:

* Wf-329-W2 MEASURED (5-d SMILES-only features, platinai_5k synthetic target):
    per-cell mean hilbert R² = **-98.08**
    per-cell mean variance_reduction = **0.0097**
* Wf-330-T1 PROJECTED (12-d RDKit, platinai_4cell real pIC50):
    per-cell mean hilbert R² ≥ **0.6**
    per-cell mean variance_reduction ≥ **0.05**

Triangle semantics are unchanged from Wf-37 L8 (ASTSR → Hilbert → MCTS →
ASTSR closure); only the data source + feature dimension change.

Per-cell breakdown (matches Wf-329-W2 ``per_cell.csv`` schema):
    cell, n_seeds, mean_hilbert_r2, std_hilbert_r2,
    mean_evolutionary_r2, std_evolutionary_r2,
    mean_variance_reduction, std_variance_reduction

Honest framing
--------------
* Per-cell R² is computed on the *training matrix* (the
  ``TriangleClosure`` orchestrator does not split a hold-out — that
  ship happens in ``bench_hilbert_astr_phase2_platinai_4cell.py``).
  This is the same contract as Wf-329-W2.
* ``mean_variance_reduction`` measures ``1 - Var(MCTS Q) /
  Var(prior biases)``; values near zero mean the MCTS is not
  converging on the ASTSR/Hilbert signal.
* If 5-d → 12-d feature expansion lifts the per-cell mean:
  ``HONEST_NEGATIVE_RESOLVED``.
* If not: ``HONEST_NEGATIVE_PERSISTENT`` (deferred to GPU).

CLI::

    # Default: 100 trajectories × 5 rounds × 5 seeds × 4 cells
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/triangle_closure_platinai_4cell.py

    # Smoke (16 trajectories × 2 rounds × 2 seeds × 4 cells)
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/triangle_closure_platinai_4cell.py \\
        --n-trajectories 16 --n-rounds 2 --seeds 0 1

Lit anchors
-----------
* Wf-37 L8 (TODO-83 §4.4) — base triangle closure orchestrator.
* Wf-39 P4 — platinai_5k end-to-end multi-seed bench (5-d features).
* Wf-49 P2 — F5_RDKIT_2D 12-d RDKit 2D descriptor (this script's
  feature schema).
* Wf-329-W2 — prior honest-negative: per-cell R² = -98.08 +
  variance_reduction = 0.0097.
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
DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_4cell.csv"
DEFAULT_OUT_DIR: str = "molmetal/reports/wf330_triangle_platinai"
DEFAULT_N_TRAJECTORIES: int = 100
DEFAULT_N_ROUNDS: int = 5
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_ARMS: int = 4
DEFAULT_HILBERT_ITERS: int = 20

#: 12-d F5_RDKIT_2D feature schema (Wf-49 P2 ship).
FEATURE_NAMES: Tuple[str, ...] = (
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

#: PlatinAI 4-cell column map.
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
DEFAULT_CELL_LINES: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")

#: Wf-329-W2 baseline (per-cell aggregation from platinai_5k + 5-d features).
WF329_W2_BASELINE: Dict[str, float] = {
    "mean_hilbert_r2": -98.08,
    "mean_variance_reduction": 0.0097,
}

#: Wf-330-T1 honest-negative recovery targets (PROJECTED).
PROJECTED_HILBERT_R2_MIN: float = 0.6
PROJECTED_VAR_REDUCTION_MIN: float = 0.05


# ---------------------------------------------------------------------------
# 12-d RDKit featurizer — F5_RDKIT_2D (Wf-49 P2 contract).
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit 2D descriptor vector; ``None`` if SMILES fails.

    Schema matches ``FEATURE_NAMES`` byte-for-byte.  Reuses the
    contract from
    :mod:`molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell`.
    """
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
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.BalabanJ(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV to list-of-dicts (utf-8, tolerant to BOM)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


# ---------------------------------------------------------------------------
# Per-cell loader — F5_RDKIT_2D 12-d + real pIC50 labels.
# ---------------------------------------------------------------------------
@dataclass
class CellLineSpec:
    """Specification for one PlatinAI cell-line column."""

    cell_line: str
    target_col: str


def load_platinai_4cell(
    repo_root: Path,
    cell_line: str,
    n_max: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PlatinAI 4-cell for a single cell-line target column.

    Returns ``(X, y, smiles)``:

    * ``X`` is ``(n_samples, 12)`` float32 (F5_RDKIT_2D descriptor).
    * ``y`` is ``(n_samples,)`` float32 — log10(IC50_nM × 1e-9) =
      -log10(IC50_µM × 1e-6) (real pIC50, signed).
    * ``smiles`` is the list of SMILES (for debugging / cohort join).

    Rows with NaN/empty target or unparseable SMILES are dropped.
    """
    if cell_line not in PLATINAI_CELL_COLS:
        raise ValueError(
            f"Unknown cell line: {cell_line}; "
            f"supported: {sorted(PLATINAI_CELL_COLS)}"
        )
    csv_path = repo_root / DEFAULT_PLATINAI_CSV
    target_col = PLATINAI_CELL_COLS[cell_line]
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles(smi)
        if feats is None:
            continue
        raw = r.get(target_col)
        try:
            ic50_nm = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
            continue
        # Signed pIC50 = -log10(IC50_µM × 1e-6) = -log10(IC50_nM × 1e-9)
        pic50 = -math.log10(ic50_nm * 1e-9)
        if not math.isfinite(pic50):
            continue
        Xs.append(feats)
        Ys.append(float(pic50))
        smiles.append(smi)
        if len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(f"No usable rows for cell line {cell_line}")
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column of ``X`` so Hilbert primal-dual is stable.

    Reused from Wf-39 P4 — real-data features have wildly different
    scales (e.g. ``MolWt`` ~ 100–1000 vs ``FractionCSP3`` ∈ [0, 1])
    and the Hilbert primal-dual optimiser uses fixed ``lr=0.05``.
    Without per-column normalisation the optimiser diverges on
    real-scale features.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0) if X.size > 0 else np.zeros(X.shape[1])
    std = np.std(X, axis=0) if X.size > 0 else np.ones(X.shape[1])
    std = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std
    return X_std, mean, std


# ---------------------------------------------------------------------------
# Per-seed × per-cell orchestrator — reuses Wf-37 L8 + Wf-39 P4 contract.
# ---------------------------------------------------------------------------
@dataclass
class CellSeedResult:
    """Telemetry for one (cell, seed) triangle-closure run."""

    cell: str
    seed: int
    n_rows: int
    mean_hilbert_r2: float
    mean_evolutionary_r2: float
    mean_variance_reduction: float
    astsr_success_rate: float
    hilbert_success_rate: float
    mcts_success_rate: float
    total_wall_s: float
    per_round: List[Dict[str, Any]] = field(default_factory=list)


def _run_one_cell_one_seed(
    cell: str,
    seed: int,
    n_trajectories: int,
    n_rounds: int,
    n_arms: int,
    hilbert_iters: int,
    X: np.ndarray,
    y: np.ndarray,
) -> CellSeedResult:
    """Run the full triangle closure for one (cell, seed).

    Reuses :class:`molmetal.scripts.triangle_closure.TriangleClosure`
    end-to-end so the contract matches Wf-37 L8 + Wf-39 P4 exactly.
    """
    # Local import — keeps this script importable even if the
    # upstream orchestrator is broken.
    from molmetal.scripts.triangle_closure import TriangleClosure

    closure = TriangleClosure(
        n_trajectories=int(n_trajectories),
        n_rounds=int(n_rounds),
        n_features=int(X.shape[1]),
        n_arms=int(n_arms),
        seed=int(seed),
        hilbert_iters=int(hilbert_iters),
    )
    X = X.copy()
    y = y.copy()

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
            feature_names=FEATURE_NAMES,
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
                        "r2": float(astsr_v.r2)
                        if math.isfinite(astsr_v.r2) else 0.0,
                        "success": astsr_ok,
                        "metadata": dict(astsr_v.metadata),
                    },
                    "hilbert": {
                        "name": "hilbert",
                        "wall_s": float(hilbert_v.wall_s),
                        "r2": float(hilbert_v.r2)
                        if math.isfinite(hilbert_v.r2) else 0.0,
                        "success": hilbert_ok,
                        "metadata": dict(hilbert_v.metadata),
                    },
                    "mcts": {
                        "name": "mcts",
                        "wall_s": float(mcts_v.wall_s),
                        "n_actions": int(mcts_v.n_actions),
                        "variance_reduction": float(
                            mcts_v.variance_reduction
                        ),
                        "success": mcts_ok,
                        "metadata": dict(mcts_v.metadata),
                    },
                },
            }
        )
        if r + 1 < int(n_rounds):
            X, y = closure._feedback_dataset(X, y, mcts_v, round_idx=int(r))

    total_wall = float(time.perf_counter() - t_total_0)

    return CellSeedResult(
        cell=str(cell),
        seed=int(seed),
        n_rows=int(X.shape[0]),
        mean_hilbert_r2=float(np.mean(
            [r["vertices"]["hilbert"]["r2"] for r in per_round]
        )) if per_round else 0.0,
        mean_evolutionary_r2=float(np.mean(
            [r["vertices"]["astsr"]["r2"] for r in per_round]
        )) if per_round else 0.0,
        mean_variance_reduction=float(np.mean(
            [r["vertices"]["mcts"]["variance_reduction"]
             for r in per_round]
        )) if per_round else 0.0,
        astsr_success_rate=float(np.mean(
            [r["vertices"]["astsr"]["success"] for r in per_round]
        )) if per_round else 0.0,
        hilbert_success_rate=float(np.mean(
            [r["vertices"]["hilbert"]["success"] for r in per_round]
        )) if per_round else 0.0,
        mcts_success_rate=float(np.mean(
            [r["vertices"]["mcts"]["success"] for r in per_round]
        )) if per_round else 0.0,
        total_wall_s=float(total_wall),
        per_round=per_round,
    )


# ---------------------------------------------------------------------------
# Per-cell aggregation.
# ---------------------------------------------------------------------------
def _aggregate_per_cell(
    per_cell_seeds: List[CellSeedResult],
) -> Dict[str, Any]:
    """Aggregate per-cell mean/std across seeds.

    Schema matches Wf-329-W2 ``per_cell.csv`` exactly:
    cell, n_seeds, mean_hilbert_r2, std_hilbert_r2,
    mean_evolutionary_r2, std_evolutionary_r2,
    mean_variance_reduction, std_variance_reduction.
    """
    if not per_cell_seeds:
        return {
            "n_seeds": 0,
            "mean_hilbert_r2": 0.0,
            "std_hilbert_r2": 0.0,
            "mean_evolutionary_r2": 0.0,
            "std_evolutionary_r2": 0.0,
            "mean_variance_reduction": 0.0,
            "std_variance_reduction": 0.0,
        }
    h_r2 = np.asarray(
        [r.mean_hilbert_r2 for r in per_cell_seeds], dtype=float
    )
    e_r2 = np.asarray(
        [r.mean_evolutionary_r2 for r in per_cell_seeds], dtype=float
    )
    v_r = np.asarray(
        [r.mean_variance_reduction for r in per_cell_seeds], dtype=float
    )
    return {
        "n_seeds": int(len(per_cell_seeds)),
        "mean_hilbert_r2": float(np.mean(h_r2)),
        "std_hilbert_r2": float(np.std(h_r2)),
        "mean_evolutionary_r2": float(np.mean(e_r2)),
        "std_evolutionary_r2": float(np.std(e_r2)),
        "mean_variance_reduction": float(np.mean(v_r)),
        "std_variance_reduction": float(np.std(v_r)),
    }


# ---------------------------------------------------------------------------
# Report builder.
# ---------------------------------------------------------------------------
def build_report(
    platinai_csv: str = DEFAULT_PLATINAI_CSV,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_rounds: int = DEFAULT_N_ROUNDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_arms: int = DEFAULT_N_ARMS,
    hilbert_iters: int = DEFAULT_HILBERT_ITERS,
    cell_lines: Sequence[str] = DEFAULT_CELL_LINES,
) -> Dict[str, Any]:
    """Build the per-cell PlatinAI 4-cell triangle-closure report."""
    repo_root = _REPO_ROOT
    csv_path = repo_root / platinai_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PlatinAI 4-cell CSV not found: {csv_path}"
        )

    per_cell_records: Dict[str, List[CellSeedResult]] = {}
    per_cell_load_stats: Dict[str, Dict[str, int]] = {}
    for cell in cell_lines:
        X_cell, y_cell, smiles_cell = load_platinai_4cell(
            repo_root=repo_root,
            cell_line=str(cell),
            n_max=int(n_trajectories * max(len(seeds), 1) * 4 + 100),
        )
        per_cell_load_stats[str(cell)] = {
            "n_loaded": int(len(smiles_cell)),
            "target_col": PLATINAI_CELL_COLS.get(str(cell), ""),
        }
        per_cell_records[str(cell)] = []
        for s in seeds:
            # Deterministic per-cell × per-seed sub-sample
            n_sub = int(min(n_trajectories, X_cell.shape[0]))
            rng = np.random.default_rng(int(s) * 31 + hash(str(cell)) % 9973)
            idx = rng.permutation(X_cell.shape[0])[:n_sub]
            X_sub = X_cell[idx, :]
            y_sub = y_cell[idx]
            rec = _run_one_cell_one_seed(
                cell=str(cell),
                seed=int(s),
                n_trajectories=int(n_sub),
                n_rounds=int(n_rounds),
                n_arms=int(n_arms),
                hilbert_iters=int(hilbert_iters),
                X=X_sub,
                y=y_sub,
            )
            per_cell_records[str(cell)].append(rec)

    # Per-cell aggregation.
    per_cell_summary: Dict[str, Dict[str, Any]] = {}
    for cell in cell_lines:
        per_cell_summary[str(cell)] = _aggregate_per_cell(
            per_cell_records[str(cell)]
        )

    # Global aggregation: mean of per-cell means (Wf-329-W2 contract).
    if per_cell_summary:
        mean_hilbert = float(np.mean([
            per_cell_summary[c]["mean_hilbert_r2"]
            for c in cell_lines
        ]))
        mean_evol = float(np.mean([
            per_cell_summary[c]["mean_evolutionary_r2"]
            for c in cell_lines
        ]))
        mean_var_red = float(np.mean([
            per_cell_summary[c]["mean_variance_reduction"]
            for c in cell_lines
        ]))
    else:
        mean_hilbert = mean_evol = mean_var_red = 0.0

    # Honest-negative recovery verdict.
    hilbert_resolved = bool(
        mean_hilbert >= PROJECTED_HILBERT_R2_MIN
    )
    var_red_resolved = bool(
        mean_var_red >= PROJECTED_VAR_REDUCTION_MIN
    )
    if hilbert_resolved and var_red_resolved:
        recovery_status = "HONEST_NEGATIVE_RESOLVED"
    elif hilbert_resolved or var_red_resolved:
        recovery_status = "HONEST_NEGATIVE_PARTIAL"
    else:
        recovery_status = "HONEST_NEGATIVE_PERSISTENT"

    summary = {
        "n_seeds": int(len(seeds)),
        "n_cell_lines": int(len(cell_lines)),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "n_arms": int(n_arms),
        "hilbert_iters": int(hilbert_iters),
        "feature_schema": "F5_RDKIT_2D_12d",
        "target_schema": "real_signed_pIC50_per_cell",
        # Global metrics (mean of per-cell means — matches Wf-329-W2).
        "mean_hilbert_r2": float(mean_hilbert),
        "mean_evolutionary_r2": float(mean_evol),
        "mean_variance_reduction": float(mean_var_red),
        # Honest-negative framing.
        "wf329_w2_baseline": dict(WF329_W2_BASELINE),
        "projected_targets": {
            "mean_hilbert_r2_min": float(PROJECTED_HILBERT_R2_MIN),
            "mean_variance_reduction_min": float(
                PROJECTED_VAR_REDUCTION_MIN
            ),
        },
        "recovery_status": str(recovery_status),
        "hilbert_resolved": bool(hilbert_resolved),
        "varred_resolved": bool(var_red_resolved),
        "delta_hilbert_r2_vs_wf329w2": float(
            mean_hilbert - WF329_W2_BASELINE["mean_hilbert_r2"]
        ),
        "delta_varred_vs_wf329w2": float(
            mean_var_red - WF329_W2_BASELINE["mean_variance_reduction"]
        ),
        "load_stats": per_cell_load_stats,
    }

    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": int(len(FEATURE_NAMES)),
        "platinai_csv": str(csv_path),
        "cell_lines": [str(c) for c in cell_lines],
        "seeds": [int(s) for s in seeds],
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "per_cell": {
            str(cell): [
                {
                    "cell": r.cell,
                    "seed": r.seed,
                    "n_rows": r.n_rows,
                    "mean_hilbert_r2": r.mean_hilbert_r2,
                    "mean_evolutionary_r2": r.mean_evolutionary_r2,
                    "mean_variance_reduction": r.mean_variance_reduction,
                    "astsr_success_rate": r.astsr_success_rate,
                    "hilbert_success_rate": r.hilbert_success_rate,
                    "mcts_success_rate": r.mcts_success_rate,
                    "total_wall_s": r.total_wall_s,
                    "per_round": r.per_round,
                }
                for r in per_cell_records[str(cell)]
            ]
            for cell in cell_lines
        },
        "per_cell_summary": per_cell_summary,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Report writers.
# ---------------------------------------------------------------------------
def write_report(
    out_dir: str,
    report: Dict[str, Any],
    *,
    also_markdown: bool = True,
    also_csv: bool = True,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Write the bench report to ``out_dir/{triangle_bench.json,per_cell.csv,triangle_bench.md}``."""
    out = Path(out_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "triangle_bench.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    csv_path: Optional[Path] = None
    if also_csv:
        csv_path = out / "per_cell.csv"
        # Match Wf-329-W2 schema byte-for-byte.
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "cell", "n_seeds", "mean_hilbert_r2", "std_hilbert_r2",
                "mean_evolutionary_r2", "std_evolutionary_r2",
                "mean_variance_reduction", "std_variance_reduction",
            ])
            for cell, agg in report.get("per_cell_summary", {}).items():
                w.writerow([
                    cell,
                    agg.get("n_seeds", 0),
                    f"{agg.get('mean_hilbert_r2', 0.0):.4f}",
                    f"{agg.get('std_hilbert_r2', 0.0):.4f}",
                    f"{agg.get('mean_evolutionary_r2', 0.0):.4f}",
                    f"{agg.get('std_evolutionary_r2', 0.0):.4f}",
                    f"{agg.get('mean_variance_reduction', 0.0):.4f}",
                    f"{agg.get('std_variance_reduction', 0.0):.4f}",
                ])

    md_path: Optional[Path] = None
    if also_markdown:
        md_path = out / "triangle_bench.md"
        summary = report.get("summary", {}) or {}
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Wf-330-T1 triangle closure on PlatinAI 4-cell "
                "(F5_RDKIT_2D 12-d, real pIC50)\n\n"
            )
            fh.write(
                f"n_seeds = {summary.get('n_seeds', '?')}, "
                f"n_cell_lines = {summary.get('n_cell_lines', '?')}, "
                f"n_trajectories = {summary.get('n_trajectories', '?')}, "
                f"n_rounds = {summary.get('n_rounds', '?')}, "
                f"hilbert_iters = {summary.get('hilbert_iters', '?')}\n\n"
            )
            fh.write("## Global metrics (mean of per-cell means)\n\n")
            fh.write(
                f"| metric | value |\n"
                f"|--------|------:|\n"
                f"| mean_hilbert_r2 | "
                f"{summary.get('mean_hilbert_r2', 0.0):.4f} |\n"
                f"| mean_evolutionary_r2 | "
                f"{summary.get('mean_evolutionary_r2', 0.0):.4f} |\n"
                f"| mean_variance_reduction | "
                f"{summary.get('mean_variance_reduction', 0.0):.4f} |\n"
                f"| delta_hilbert_r2_vs_wf329w2 | "
                f"{summary.get('delta_hilbert_r2_vs_wf329w2', 0.0):+.4f} |\n"
                f"| delta_varred_vs_wf329w2 | "
                f"{summary.get('delta_varred_vs_wf329w2', 0.0):+.4f} |\n"
                f"| recovery_status | "
                f"{summary.get('recovery_status', '?')} |\n\n"
            )
            fh.write("## Per-cell breakdown\n\n")
            fh.write(
                "| cell | n_seeds | mean_hilbert_r2 | "
                "std_hilbert_r2 | mean_evolutionary_r2 | "
                "std_evolutionary_r2 | mean_variance_reduction | "
                "std_variance_reduction |\n"
                "|------|--------:|----------------:|---------------:"
                "|----------------------:|---------------------:"
                "|----------------------:|---------------------:|\n"
            )
            for cell, agg in report.get("per_cell_summary", {}).items():
                fh.write(
                    f"| {cell} | {agg.get('n_seeds', 0)} | "
                    f"{agg.get('mean_hilbert_r2', 0.0):.4f} | "
                    f"{agg.get('std_hilbert_r2', 0.0):.4f} | "
                    f"{agg.get('mean_evolutionary_r2', 0.0):.4f} | "
                    f"{agg.get('std_evolutionary_r2', 0.0):.4f} | "
                    f"{agg.get('mean_variance_reduction', 0.0):.4f} | "
                    f"{agg.get('std_variance_reduction', 0.0):.4f} |\n"
                )
            fh.write("\n## Honest-negative recovery verdict\n\n")
            fh.write(
                f"- Wf-329-W2 MEASURED baseline: "
                f"hilbert_r2={summary.get('wf329_w2_baseline', {}).get('mean_hilbert_r2', 0.0):+.4f}, "
                f"varred={summary.get('wf329_w2_baseline', {}).get('mean_variance_reduction', 0.0):+.4f}\n"
                f"- PROJECTED: hilbert_r2 ≥ "
                f"{summary.get('projected_targets', {}).get('mean_hilbert_r2_min', 0.0):.4f}, "
                f"varred ≥ "
                f"{summary.get('projected_targets', {}).get('mean_variance_reduction_min', 0.0):.4f}\n"
                f"- MEASURED: hilbert_r2 = "
                f"{summary.get('mean_hilbert_r2', 0.0):+.4f}, "
                f"varred = "
                f"{summary.get('mean_variance_reduction', 0.0):+.4f}\n"
                f"- RECOVERY_STATUS = **{summary.get('recovery_status', '?')}**\n"
            )

    return json_path, csv_path, md_path


# ---------------------------------------------------------------------------
# Smoke + CLI.
# ---------------------------------------------------------------------------
def smoke_run(
    n_trajectories: int = 16,
    n_rounds: int = 2,
    seeds: Sequence[int] = (0, 1),
) -> bool:
    """CPU smoke that returns ``True`` iff the per-cell closure runs end-to-end."""
    try:
        report = build_report(
            n_trajectories=int(n_trajectories),
            n_rounds=int(n_rounds),
            seeds=list(seeds),
        )
    except Exception:
        return False
    if not report:
        return False
    summary = report.get("summary", {}) or {}
    if not math.isfinite(summary.get("mean_hilbert_r2", float("nan"))):
        return False
    if not math.isfinite(summary.get("mean_variance_reduction", float("nan"))):
        return False
    # Per-cell summary must have all 4 cells.
    per_cell = report.get("per_cell_summary", {}) or {}
    for cell in DEFAULT_CELL_LINES:
        if cell not in per_cell:
            return False
        if not math.isfinite(
            per_cell[cell].get("mean_hilbert_r2", float("nan"))
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-330-T1 — Triangle closure end-to-end on PlatinAI 4-cell "
            "(F5_RDKIT_2D 12-d, real pIC50 per cell line)."
        )
    )
    parser.add_argument(
        "--platinai-csv", type=str, default=DEFAULT_PLATINAI_CSV,
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=DEFAULT_N_TRAJECTORIES,
    )
    parser.add_argument(
        "--n-rounds", type=int, default=DEFAULT_N_ROUNDS,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--n-arms", type=int, default=DEFAULT_N_ARMS,
    )
    parser.add_argument(
        "--hilbert-iters", type=int, default=DEFAULT_HILBERT_ITERS,
    )
    parser.add_argument(
        "--cell-lines", type=str, nargs="+",
        default=list(DEFAULT_CELL_LINES),
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--smoke", action="store_true",
    )
    parser.add_argument(
        "--quiet", action="store_true",
    )
    args = parser.parse_args()

    if args.smoke:
        ok = smoke_run()
        print(
            "Wf-330-T1 triangle closure on PlatinAI 4-cell: smoke "
            f"{'PASS' if ok else 'FAIL'}"
        )
        return 0 if ok else 1

    report = build_report(
        platinai_csv=str(args.platinai_csv),
        n_trajectories=int(args.n_trajectories),
        n_rounds=int(args.n_rounds),
        seeds=[int(s) for s in args.seeds],
        n_arms=int(args.n_arms),
        hilbert_iters=int(args.hilbert_iters),
        cell_lines=[str(c) for c in args.cell_lines],
    )

    if not args.quiet:
        summary = report.get("summary", {}) or {}
        print(
            "Wf-330-T1 triangle closure on PlatinAI 4-cell:\n"
            f"  feature_schema=F5_RDKIT_2D_12d "
            f"n_trajectories={args.n_trajectories} "
            f"n_rounds={args.n_rounds} seeds={args.seeds}\n"
            f"  mean_hilbert_r2={summary.get('mean_hilbert_r2', 0.0):+.4f}  "
            f"mean_evol_r2={summary.get('mean_evolutionary_r2', 0.0):+.4f}  "
            f"mean_varred={summary.get('mean_variance_reduction', 0.0):+.4f}\n"
            f"  delta_hilbert_r2_vs_wf329w2="
            f"{summary.get('delta_hilbert_r2_vs_wf329w2', 0.0):+.4f}  "
            f"delta_varred_vs_wf329w2="
            f"{summary.get('delta_varred_vs_wf329w2', 0.0):+.4f}\n"
            f"  RECOVERY_STATUS = {summary.get('recovery_status', '?')}"
        )
        per_cell = report.get("per_cell_summary", {}) or {}
        for cell in args.cell_lines:
            agg = per_cell.get(str(cell), {})
            print(
                f"  cell={cell}: hilbert R²="
                f"{agg.get('mean_hilbert_r2', 0.0):+.4f}  "
                f"evol R²="
                f"{agg.get('mean_evolutionary_r2', 0.0):+.4f}  "
                f"var_red="
                f"{agg.get('mean_variance_reduction', 0.0):+.4f}"
            )

    json_path, csv_path, md_path = write_report(
        out_dir=str(args.out_dir),
        report=report,
        also_markdown=True,
        also_csv=True,
    )
    print(
        f"Wrote {json_path}"
        + (f" + {csv_path}" if csv_path is not None else "")
        + (f" + {md_path}" if md_path is not None else "")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
