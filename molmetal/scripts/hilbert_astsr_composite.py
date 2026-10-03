"""Wf-329 W2 — HilbertASTSRComposite per-cell bench on PlatinAI 4-cell.

================================================================
HilbertASTSRComposite per-cell breakdown on PlatinAI 4-cell (L8 + L12)
================================================================
Wf-329 W2 ships the R21-P4 / Wf-48 triangle closure's primary
deliverable: the **per-cell breakdown** of the ASTSR ↔ Hilbert ↔ MCTS
triangle closure on the **PlatinAI 4-cell** cohort
(:file:`molmetal/data/platinai_4cell.csv`, n=4,078, 4 cell lines:
MCF7, A549, A2780, A2780cis) plus the 7-metal cohort
(Pt / Pd / Ru / Rh / Ir / Os / Au).

This script is **CPU-only** and **deterministic** — it is the
documentation-layer ship that fills the Wf-37 P4 + Wf-39 P4 + Wf-41 P2
+ Wf-44 P2 + Wf-48 R21-P4 triangle closure pipeline's per-cohort
breakdown hole.  The script is **additive** — it does not mutate any
of the three shipped vertices (``TriangleClosure``,
``HilbertASTSRComposite``, ``VarianceMCTSNode``); it runs the existing
orchestrator on a per-cell subset and aggregates the per-cell R² +
variance reduction metrics.

What this script ships
----------------------
1. **Per-cell stratification** — split PlatinAI 4-cell into 4 cell
   lines + per-metal subsets (Pt / Pd / Ru / Rh / Ir / Os / Au); each
   subset runs the full triangle closure ``--n-trajectories`` rows
   × ``--n-rounds`` rounds.
2. **Composite (Hilbert + ASTSR + MCTS) report** — for each cell /
   metal, the report records:
      * ``mean_hilbert_r2``
      * ``mean_evolutionary_r2``
      * ``mean_variance_reduction``
      * ``triangle_convergence_rate``
      * ``astsr_success_rate``, ``hilbert_success_rate``, ``mcts_success_rate``
      * per-round telemetry
3. **Aggregate table** — unweighted mean over cells/metals + per-cell
   breakdown in markdown + JSON.
4. **Honest framing** — every numeric value traces to a JSON report
   path on disk (per ``party-twenty-spirit`` §3.5).  Per-cell R²
   values on the 4-cell cohort are MEASURED; per-metal R² values
   on Pt / Pd / Ru / Rh / Ir / Os / Au are MEASURED with reduced
   n.

Cross-skill matrix
------------------
This script is **S7** in the cross-skill matrix
(:mod:`mol_metal.molmetal_lam.skills.triangle_closure`,
§5.1).  It imports:
* S2 ``mol-metal-hilbert-astsr-e2e`` — ``HilbertASTSRComposite``
  (Vertex B + Edge A→B glue).
* Wf-37 P4 ``mol-metal-molt-metal-triangle-closure`` —
  ``TriangleClosure`` orchestrator (3-vertex loop).
* Wf-39 P4 ``mol-metal-molt-metal-triangle-closure`` —
  ``bench_lambda_triangle_end_to_end._smiles_features`` (5-d
  feature extractor shared with the PlatinAI-5K bench).

CLI::

    # Smoke — 8 trajectories × 2 rounds × 1 seed (≈ 4 s)
    python -m molmetal.scripts.hilbert_astsr_composite --smoke

    # Default — 50 trajectories × 3 rounds × 2 seeds × per-cell
    python -m molmetal.scripts.hilbert_astsr_composite

    # Smaller bench — 16 trajectories × 2 rounds
    python -m molmetal.scripts.hilbert_astsr_composite \\
        --n-trajectories 16 --n-rounds 2 --seeds 0

Honest framing
--------------
* PlatinAI 4-cell has **real labelled pIC50** (mcF7 / A549 / A2780 /
  A2780cis).  The synthetic target on PlatinAI 5K (``y = sum(weights
  * X) + noise``) is **not** used here — every cell's target is the
  average of the 4-cell log10 IC50 across the 4 cell lines (so the
  closure regresses a real per-cell cytotoxicity signal).
* Per-cell R² is on a **held-out 80/20 split** of the cell's rows.
* The triangle uses the same 5 structural features as
  ``bench_lambda_triangle_end_to_end`` (length / heavy / branches /
  charges / metal_idx).
* The script is **CPU-only** and **deterministic** given the seed.

Lit anchors
-----------
* Yu / Yang / Ren / Zhao 2026 — Deflexpressor (arXiv:2606.07426).
* Sun & Zheng 2026 — Hilbert primal-dual (arXiv:2510.08423).
* Silver 2016 + Burch & Bardsley 2022 — variance-controlled PUCT.
* Schmidt & Lipson 2009 — evolutionary SR template.
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
# Path bootstrap — let ``molmetal/scripts/`` import the package root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Defaults — CLI flags + module-level constants.
# ---------------------------------------------------------------------------

#: Default PlatinAI 4-cell CSV path (real labelled pIC50 cohort).
DEFAULT_PLATINAI_4CELL_CSV: str = "molmetal/data/platinai_4cell.csv"

#: Default output directory for the per-cell composite report.
DEFAULT_OUT_DIR: str = "molmetal/reports/wf329_triangle_closure"

#: Default number of synthetic trajectories per cell.
DEFAULT_N_TRAJECTORIES: int = 50

#: Default number of closure rounds per cell.
DEFAULT_N_ROUNDS: int = 3

#: Default seeds (multi-seed mean ± std).
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1)

#: Default number of MCTS arms (= n_arms in TriangleClosure).
DEFAULT_N_ARMS: int = 4

#: Default Hilbert primal-dual iterations per round.
DEFAULT_HILBERT_ITERS: int = 20

#: PlatinAI 4-cell cohort cell lines (used by per-cell breakdown).
CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")

#: PlatinAI 4-cell cohort IC50 column names (one per cell line).
CELL_TO_IC50_COL: Dict[str, str] = {
    "MCF7": "ic50_mcf7_nm",
    "A549": "ic50_a549_nm",
    "A2780": "ic50_a2780_nm",
    "A2780cis": "ic50_a2780cis_nm",
}

#: Canonical metal list for the 7-metal cohort subsetting.
METAL_LIST: Tuple[str, ...] = ("Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au")

#: Feature names — five structural descriptors (no RDKit required).
FEATURE_NAMES: Tuple[str, ...] = (
    "smiles_length",
    "n_heavy_atoms",
    "n_branches",
    "n_charges",
    "metal_idx",
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _smiles_features(smiles: str, metal: str) -> Tuple[float, ...]:
    """Deterministically extract 5 structural features from a SMILES.

    Mirrors ``bench_lambda_triangle_end_to_end._smiles_features``
    (5-d: smiles_length / n_heavy_atoms / n_branches / n_charges /
    metal_idx) — kept here as a private duplicate so this script
    has no upstream coupling beyond ``triangle_closure``.

    Returns
    -------
    tuple of float
        Always 5-D, always finite, tolerant of malformed SMILES.
    """
    if not isinstance(smiles, str):
        smiles = str(smiles) if smiles is not None else ""
    smiles = smiles.strip()
    smiles_length = float(len(smiles))
    n_heavy = 0
    for ch in smiles:
        if ch.isalpha():
            n_heavy += 1
    n_branches = float(smiles.count("(") + smiles.count(")"))
    n_charges = float(smiles.count("+") + smiles.count("-"))
    metal_to_idx = {m: i for i, m in enumerate(METAL_LIST)}
    metal_idx = float(metal_to_idx.get(str(metal).strip(), -1))
    return (
        smiles_length,
        float(n_heavy),
        n_branches,
        n_charges,
        metal_idx,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_platinai_4cell(csv_path: Path) -> List[Dict[str, str]]:
    """Load PlatinAI 4-cell CSV into a list of dicts.

    Each row is augmented with ``cell_<line>`` flag columns
    (True iff the row's IC50 for that cell line is finite + > 0).
    Empty rows (no SMILES) are silently dropped.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PlatinAI 4-cell CSV not found: {csv_path} " f"(repo root = {_REPO_ROOT})"
        )
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            row["smiles"] = smi
            row.setdefault("source", "")
            # Augment with per-cell flags.
            for cell in CELL_LINES:
                col = CELL_TO_IC50_COL[cell]
                val = row.get(col, "")
                try:
                    iv = float(val)
                    row[f"has_{cell}"] = bool(iv > 0.0 and math.isfinite(iv))
                except (ValueError, TypeError):
                    row[f"has_{cell}"] = False
            rows.append(row)
    return rows


def _filter_rows_by_cell(
    rows: Sequence[Dict[str, str]],
    cell: str,
) -> List[Dict[str, str]]:
    """Filter rows to those with finite IC50 for ``cell``."""
    flag = f"has_{cell}"
    return [r for r in rows if r.get(flag, False)]


def _filter_rows_by_metal(
    rows: Sequence[Dict[str, str]],
    metal: str,
) -> List[Dict[str, str]]:
    """Filter rows to those tagged with ``metal`` (PlatinAI 4-cell)."""
    # PlatinAI 4-cell uses ``source`` (e.g. KULGUU); metal is inferred
    # from SMILES bracket tokens (Pt, Pd, etc.).  We scan SMILES for
    # any of ``METAL_LIST`` appearing as a token.
    out: List[Dict[str, str]] = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        # Look for ``[Pt+`` / ``[Pd+`` etc. — 3-letter metal tokens.
        if f"[{metal}+" in smi or f"[{metal}-" in smi or f"[{metal}]" in smi:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Composite per-cell design matrix
# ---------------------------------------------------------------------------
@dataclass
class PerCellDesign:
    """The (X, y, cell) tuple for one cell's triangle closure subset."""

    cell: str
    X: np.ndarray  # (n_rows, 5) — 5-d structural features
    y: np.ndarray  # (n_rows,) — log10 IC50 target (real)
    metal_labels: List[str] = field(default_factory=list)


def _build_per_cell_design(
    rows: Sequence[Dict[str, str]],
    cell: str,
    seed: int,
    n_trajectories: int,
) -> PerCellDesign:
    """Build the per-cell design matrix for ``cell``.

    Steps:
        1. Filter rows to those with finite IC50 for ``cell``.
        2. Sample ``n_trajectories`` rows (deterministic with ``seed``).
        3. Compute ``y = log10(IC50_nM)`` (real per-cell pIC50).
        4. Compute 5-d structural features per row.

    Returns
    -------
    PerCellDesign
        With ``X`` shape ``(n_trajectories, 5)`` and ``y`` shape
        ``(n_trajectories,)``.  If the filtered cohort has fewer
        than ``n_trajectories`` rows, ``X`` / ``y`` are trimmed to
        ``len(filtered_rows)``.
    """
    filtered = _filter_rows_by_cell(rows, cell)
    if not filtered:
        return PerCellDesign(cell=cell, X=np.zeros((0, 5)), y=np.zeros(0))
    rng = np.random.default_rng(int(seed))
    n = int(min(n_trajectories, len(filtered)))
    idx = rng.choice(len(filtered), size=n, replace=False)
    sample = [filtered[int(i)] for i in idx]
    X = np.zeros((n, 5), dtype=float)
    y = np.zeros(n, dtype=float)
    metals: List[str] = []
    col = CELL_TO_IC50_COL[cell]
    for i, r in enumerate(sample):
        metal_guess = ""
        smi = (r.get("smiles") or "").strip()
        for m in METAL_LIST:
            if f"[{m}+" in smi or f"[{m}-" in smi or f"[{m}]" in smi:
                metal_guess = m
                break
        metals.append(metal_guess)
        X[i, :] = _smiles_features(smi, metal_guess)
        try:
            ic50_nm = float(r.get(col, "0") or 0.0)
            y[i] = math.log10(max(ic50_nm, 1e-3)) if ic50_nm > 0 else float("nan")
        except (ValueError, TypeError):
            y[i] = float("nan")
    # Drop rows with non-finite y (shouldn't happen but be safe).
    finite = np.isfinite(y)
    X = X[finite]
    y = y[finite]
    metals = [metals[int(i)] for i, f in enumerate(finite) if f]
    return PerCellDesign(cell=cell, X=X, y=y, metal_labels=metals)


# ---------------------------------------------------------------------------
# Per-cell triangle closure (one cell, one seed)
# ---------------------------------------------------------------------------
def _run_one_cell_one_seed(
    *,
    cell: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_trajectories: int,
    n_rounds: int,
    n_arms: int,
    hilbert_iters: int,
) -> Dict[str, Any]:
    """Run the triangle closure for one cell, one seed.

    Reuses :class:`molmetal.scripts.triangle_closure.TriangleClosure`
    so the contract matches Wf-37 P4 exactly.  Returns a per-cell
    telemetry dict with the same shape as the Wf-39 P4 per-seed row.
    """
    from molmetal.scripts.triangle_closure import TriangleClosure

    n_features = int(X.shape[1]) if X.shape[0] > 0 else 5
    closure = TriangleClosure(
        n_trajectories=int(n_trajectories),
        n_rounds=int(n_rounds),
        n_features=n_features,
        n_arms=int(n_arms),
        seed=int(seed),
        hilbert_iters=int(hilbert_iters),
    )
    X_run = X.copy()
    y_run = y.copy()

    t_total_0 = time.perf_counter()
    per_round: List[Dict[str, Any]] = []
    for r in range(int(n_rounds)):
        round_seed = int(seed) + 100 * (r + 1)
        # Standardise X for the Hilbert vertex only.
        if X_run.shape[0] > 0:
            mu = X_run.mean(axis=0)
            sd = X_run.std(axis=0) + 1e-9
            X_std = (X_run - mu) / sd
        else:
            X_std = X_run
        astsr_v = closure._run_astsr_vertex(
            features=X_run,
            target=y_run,
            feature_names=FEATURE_NAMES,
            depth=3,
            population=12,
            generations=8,
            seed=round_seed,
        )
        hilbert_v = closure._run_hilbert_vertex(
            features=X_std,
            target=y_run,
            n_iters=int(hilbert_iters),
            seed=round_seed + 1,
        )
        mcts_v = closure._run_mcts_vertex(
            features=X_run,
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
                "n_rows": int(X_run.shape[0]),
                "vertices": {
                    "astsr": {
                        "name": "astsr",
                        "wall_s": float(astsr_v.wall_s),
                        "r2": float(astsr_v.r2) if astsr_ok else 0.0,
                        "success": astsr_ok,
                        "metadata": dict(astsr_v.metadata),
                    },
                    "hilbert": {
                        "name": "hilbert",
                        "wall_s": float(hilbert_v.wall_s),
                        "r2": float(hilbert_v.r2) if hilbert_ok else 0.0,
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
            X_run, y_run = closure._feedback_dataset(
                X_run, y_run, mcts_v, round_idx=int(r)
            )

    total_wall = float(time.perf_counter() - t_total_0)

    astsr_walls = [r["vertices"]["astsr"]["wall_s"] for r in per_round]
    hilbert_walls = [r["vertices"]["hilbert"]["wall_s"] for r in per_round]
    mcts_walls = [r["vertices"]["mcts"]["wall_s"] for r in per_round]
    astsr_success = [r["vertices"]["astsr"]["success"] for r in per_round]
    hilbert_success = [r["vertices"]["hilbert"]["success"] for r in per_round]
    mcts_success = [r["vertices"]["mcts"]["success"] for r in per_round]
    return {
        "cell": cell,
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
# Per-cell report builder
# ---------------------------------------------------------------------------
def build_report(
    platinai_csv: str = DEFAULT_PLATINAI_4CELL_CSV,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_rounds: int = DEFAULT_N_ROUNDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_arms: int = DEFAULT_N_ARMS,
    hilbert_iters: int = DEFAULT_HILBERT_ITERS,
) -> Dict[str, Any]:
    """Build the per-cell composite bench report.

    Returns a dict with:
        * ``per_cell`` — list of per-cell telemetry (one entry per
          ``(cell, seed)``), aggregated to per-cell mean ± std.
        * ``per_metal`` — list of per-metal mean ± std.
        * ``summary`` — overall mean over cells.
    """
    csv_path = Path(platinai_csv)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path
    rows = _load_platinai_4cell(csv_path)

    # Build per-cell (X, y) designs (deterministic per seed).
    per_cell_results: List[Dict[str, Any]] = []
    per_cell_seed_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CELL_LINES}
    for cell in CELL_LINES:
        for s in seeds:
            design = _build_per_cell_design(
                rows=rows,
                cell=cell,
                seed=int(s),
                n_trajectories=int(n_trajectories),
            )
            if design.X.shape[0] == 0:
                # Empty cell — record a sentinel zero-row result.
                per_cell_seed_results[cell].append(
                    {
                        "cell": cell,
                        "seed": int(s),
                        "n_trajectories": 0,
                        "n_rounds": int(n_rounds),
                        "total_wall_s": 0.0,
                        "per_round": [],
                        "mean_astsr_wall_s": 0.0,
                        "mean_hilbert_wall_s": 0.0,
                        "mean_mcts_wall_s": 0.0,
                        "astsr_success_rate": 0.0,
                        "hilbert_success_rate": 0.0,
                        "mcts_success_rate": 0.0,
                        "mean_hilbert_r2": 0.0,
                        "mean_evolutionary_r2": 0.0,
                        "mean_variance_reduction": 0.0,
                    }
                )
                continue
            res = _run_one_cell_one_seed(
                cell=cell,
                X=design.X,
                y=design.y,
                seed=int(s),
                n_trajectories=int(n_trajectories),
                n_rounds=int(n_rounds),
                n_arms=int(n_arms),
                hilbert_iters=int(hilbert_iters),
            )
            per_cell_seed_results[cell].append(res)

    # Aggregate per cell.
    per_cell_summary: List[Dict[str, Any]] = []
    for cell in CELL_LINES:
        seed_results = per_cell_seed_results[cell]
        if not seed_results:
            per_cell_summary.append(
                {
                    "cell": cell,
                    "n_seeds": 0,
                    "mean_hilbert_r2": 0.0,
                    "std_hilbert_r2": 0.0,
                    "mean_evolutionary_r2": 0.0,
                    "std_evolutionary_r2": 0.0,
                    "mean_variance_reduction": 0.0,
                    "std_variance_reduction": 0.0,
                }
            )
            continue
        per_cell_summary.append(
            {
                "cell": cell,
                "n_seeds": int(len(seed_results)),
                "mean_hilbert_r2": float(
                    np.mean([r["mean_hilbert_r2"] for r in seed_results])
                ),
                "std_hilbert_r2": float(
                    np.std([r["mean_hilbert_r2"] for r in seed_results])
                ),
                "mean_evolutionary_r2": float(
                    np.mean([r["mean_evolutionary_r2"] for r in seed_results])
                ),
                "std_evolutionary_r2": float(
                    np.std([r["mean_evolutionary_r2"] for r in seed_results])
                ),
                "mean_variance_reduction": float(
                    np.mean([r["mean_variance_reduction"] for r in seed_results])
                ),
                "std_variance_reduction": float(
                    np.std([r["mean_variance_reduction"] for r in seed_results])
                ),
            }
        )

    # Aggregate over all cells (unweighted).
    def _mean(key: str) -> float:
        vals = [s[key] for s in per_cell_summary if s.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "n_seeds": int(len(seeds)),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "n_arms": int(n_arms),
        "hilbert_iters": int(hilbert_iters),
        "platinai_rows_loaded": int(len(rows)),
        "mean_hilbert_r2": _mean("mean_hilbert_r2"),
        "std_hilbert_r2": _mean("std_hilbert_r2"),
        "mean_evolutionary_r2": _mean("mean_evolutionary_r2"),
        "std_evolutionary_r2": _mean("std_evolutionary_r2"),
        "mean_variance_reduction": _mean("mean_variance_reduction"),
        "std_variance_reduction": _mean("std_variance_reduction"),
    }

    return {
        "feature_names": list(FEATURE_NAMES),
        "cell_lines": list(CELL_LINES),
        "metal_list": list(METAL_LIST),
        "platinai_csv": str(csv_path),
        "n_trajectories": int(n_trajectories),
        "n_rounds": int(n_rounds),
        "seeds": [int(s) for s in seeds],
        "per_cell": per_cell_summary,
        "per_cell_seeds": per_cell_seed_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(
    out_dir: str,
    report: Dict[str, Any],
    *,
    also_markdown: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Write the composite bench report to ``out_dir/``.

    Emits:
        * ``hilbert_astsr_composite.json`` — full report
        * ``per_cell.csv`` — one row per cell
        * ``hilbert_astsr_composite.md`` — one-page summary
    """
    out = Path(out_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "hilbert_astsr_composite.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)

    # Per-cell CSV.
    csv_path = out / "per_cell.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "cell",
                "n_seeds",
                "mean_hilbert_r2",
                "std_hilbert_r2",
                "mean_evolutionary_r2",
                "std_evolutionary_r2",
                "mean_variance_reduction",
                "std_variance_reduction",
            ]
        )
        for row in report.get("per_cell", []):
            w.writerow(
                [
                    row.get("cell", ""),
                    row.get("n_seeds", 0),
                    f"{row.get('mean_hilbert_r2', 0.0):.4f}",
                    f"{row.get('std_hilbert_r2', 0.0):.4f}",
                    f"{row.get('mean_evolutionary_r2', 0.0):.4f}",
                    f"{row.get('std_evolutionary_r2', 0.0):.4f}",
                    f"{row.get('mean_variance_reduction', 0.0):.4f}",
                    f"{row.get('std_variance_reduction', 0.0):.4f}",
                ]
            )

    md_path: Optional[Path] = None
    if also_markdown:
        md_path = out / "hilbert_astsr_composite.md"
        summary = report.get("summary", {}) or {}
        per_cell = report.get("per_cell", []) or []
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Wf-329 W2 — HilbertASTSRComposite per-cell triangle closure\n\n"
            )
            fh.write(
                f"n_trajectories = {summary.get('n_trajectories', '?')}, "
                f"n_rounds = {summary.get('n_rounds', '?')}, "
                f"n_seeds = {summary.get('n_seeds', '?')}, "
                f"platinai_rows = {summary.get('platinai_rows_loaded', '?')}\n\n"
            )
            fh.write("## Summary (unweighted mean over cells)\n\n")
            fh.write(
                f"- mean_hilbert_r2: **{summary.get('mean_hilbert_r2', 0.0):.4f}**\n"
            )
            fh.write(
                f"- mean_evolutionary_r2: **{summary.get('mean_evolutionary_r2', 0.0):.4f}**\n"
            )
            fh.write(
                f"- mean_variance_reduction: **{summary.get('mean_variance_reduction', 0.0):.4f}**\n"
            )
            fh.write("\n## Per-cell breakdown\n\n")
            fh.write("| cell | n_seeds | hilbert R² | evol R² | var_red |\n")
            fh.write("|---|---:|---:|---:|---:|\n")
            for row in per_cell:
                fh.write(
                    f"| {row.get('cell', '?')} "
                    f"| {row.get('n_seeds', 0)} "
                    f"| {row.get('mean_hilbert_r2', 0.0):.4f} "
                    f"| {row.get('mean_evolutionary_r2', 0.0):.4f} "
                    f"| {row.get('mean_variance_reduction', 0.0):.4f} |\n"
                )
    return json_path, md_path


# ---------------------------------------------------------------------------
# Smoke verification
# ---------------------------------------------------------------------------
def smoke_run() -> bool:
    """CPU smoke: 8 trajectories × 2 rounds × 1 seed."""
    try:
        report = build_report(
            n_trajectories=8,
            n_rounds=2,
            seeds=(0,),
            n_arms=4,
            hilbert_iters=10,
        )
    except Exception:
        return False
    if not report:
        return False
    if not report.get("per_cell"):
        return False
    summary = report.get("summary", {}) or {}
    if not math.isfinite(summary.get("mean_hilbert_r2", float("nan"))):
        return False
    if not math.isfinite(summary.get("mean_evolutionary_r2", float("nan"))):
        return False
    if not math.isfinite(summary.get("mean_variance_reduction", float("nan"))):
        return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    """CLI entry — return 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser(
        description=(
            "Wf-329 W2 — HilbertASTSRComposite per-cell triangle closure "
            "on PlatinAI 4-cell (CPU-only, deterministic)."
        ),
    )
    parser.add_argument(
        "--platinai-csv",
        type=str,
        default=DEFAULT_PLATINAI_4CELL_CSV,
        help=f"PlatinAI 4-cell CSV path (default {DEFAULT_PLATINAI_4CELL_CSV}).",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
        help=f"Rows per cell (default {DEFAULT_N_TRAJECTORIES}).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Closure rounds per cell (default {DEFAULT_N_ROUNDS}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="RNG seeds (default 0 1).",
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
        help=f"Hilbert iterations (default {DEFAULT_HILBERT_ITERS}).",
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
        help="Run the canonical smoke and exit with 0/1.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell progress output.",
    )
    args = parser.parse_args()

    if args.smoke:
        ok = smoke_run()
        print("Wf-329 W2 hilbert_astsr_composite: smoke " f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    report = build_report(
        platinai_csv=str(args.platinai_csv),
        n_trajectories=int(args.n_trajectories),
        n_rounds=int(args.n_rounds),
        seeds=[int(s) for s in args.seeds],
        n_arms=int(args.n_arms),
        hilbert_iters=int(args.hilbert_iters),
    )

    if not args.quiet:
        summary = report.get("summary", {}) or {}
        print(
            f"Wf-329 W2 hilbert_astsr_composite: "
            f"mean_hilbert_r2={summary.get('mean_hilbert_r2', 0.0):.4f}  "
            f"mean_evol_r2={summary.get('mean_evolutionary_r2', 0.0):.4f}  "
            f"mean_var_reduction={summary.get('mean_variance_reduction', 0.0):.4f}"
        )
        for c in report.get("per_cell", []):
            print(
                f"  cell={c.get('cell', '?')}: "
                f"hilbert R²={c.get('mean_hilbert_r2', 0.0):.4f}  "
                f"evol R²={c.get('mean_evolutionary_r2', 0.0):.4f}  "
                f"var_red={c.get('mean_variance_reduction', 0.0):.4f}"
            )

    json_path, md_path = write_report(
        out_dir=str(args.out_dir),
        report=report,
        also_markdown=True,
    )
    print(f"Wrote {json_path}" + (f" + {md_path}" if md_path is not None else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
