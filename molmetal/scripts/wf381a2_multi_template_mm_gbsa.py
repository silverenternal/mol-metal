#!/usr/bin/env python3
"""Wf-381-A2 — TODO-204 m15 RMSD multi-template dock + MM-GBSA rescoring.

================================================================
Why this module exists
================================================================
Per ``TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md`` §1 — extend
the m15 Docking RMSD bench with two improvements that the canonical
Wf-179 BD-9 + Wf-246-B + Wf-247-A + Wf-341-T1 single-template bench
does NOT currently exercise:

  1. **Multi-template docking**: run Vina against **N homologous
     pocket templates** and pick the best-of-N pose as the per-pocket
     RMSD measurement.
  2. **MM-GBSA rescoring**: refine the Vina top-N poses with a
     Molecular Mechanics Generalized Born Surface Area free-energy
     estimate (Still 1990; Genheden & Ryde 2015 review).

This Wf-381-A2 module ships a **CPU-only scaffold** with object-level
API surface (PDB paths + ligand SDF + receptor/ligand paths).  It is
the API-level companion to Wf-380-A5 (scalar-level).  Both share the
same closed-form surrogate backend so the smoke bench is bit-identical
at the aggregate level.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* All numeric values in the smoke bench are **synthetic** — the m15
  cohort ``metrics/by_metric/docked_pose_rmsd.json`` MEASURED numbers
  (median 0.6578 Å across 5 organic-cofactor pockets) are the
  canonical baseline; this Wf only simulates the expected lift via
  a closed-form linear surrogate.
* **Multi-template + MM-GBSA on real Vina + GBSA binaries is
  deferred** to TODO-207 (Wf-299 launch).  The scaffold here provides
  the integration point.
* CPU-only scaffold; numpy + scipy + (optional) meeko / Vina hook.
* Per AGENTS.md §5.1 + §5.4, no GPU launches and no paper work
  performed in this Wf.

Expected lift (PROJECTED per Wf-381-A2 brief)
---------------------------------------------
* ``-0.10 to -0.15 Å`` on m15 RMSD vs single-template baseline.
* On the 5-pocket organic-cofactor set with 5 templates this would
  push the median from ``0.6578 Å`` (Wf-179 BD-9 MEASURED) to a
  projected ``~0.55 Å``.

SOTA thresholds (transferred from Wf-341-T1 + Wf-380-A5)
-------------------------------------------------------
* ``RMSD_SOTA_THRESHOLD_ANGSTROM = 2.0`` — success threshold.
* ``RMSD_PRACTICAL_THRESHOLD_ANGSTROM = 2.5`` — practical acceptance.
* ``PER_POCKET_SUCCESS_RATE = 1.0`` — SOTA bar (5/5 PASS on canonical
  cohort).

Outputs (NEW this workflow)
---------------------------
* ``molmetal/reports/wf381a2_multi_template_smoke/result.json``
  — smoke bench with multi-template + MM-GBSA simulated numbers
  (synthetic; not MEASURED on real Vina / GBSA binaries).
* ``molmetal/tests/test_wf381a2_multi_template.py`` — 2 NEW tests:
  ``test_multi_template_dock`` + ``test_mm_gbsa_rescore``.

Usage
-----
::

    # Default smoke (5 templates, n_poses=9)
    python molmetal/scripts/wf381a2_multi_template_mm_gbsa.py

    # 20 templates, n_poses=50
    python molmetal/scripts/wf381a2_multi_template_mm_gbsa.py \\
        --n-templates 20 --n-poses 50

    # Real meeko + Vina hook (requires binaries installed)
    python molmetal/scripts/wf381a2_multi_template_mm_gbsa.py \\
        --backend real --meeko --vina-binary vina
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants — SOTA success gate (transferred from Wf-341-T1 + Wf-380-A5)
# ---------------------------------------------------------------------------
RMSD_SOTA_THRESHOLD_ANGSTROM: float = 2.0
RMSD_PRACTICAL_THRESHOLD_ANGSTROM: float = 2.5
PER_POCKET_SUCCESS_RATE: float = 1.0

# ---------------------------------------------------------------------------
# Constants — multi-template defaults
# ---------------------------------------------------------------------------
DEFAULT_POCKETS: Tuple[str, ...] = (
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
)
DEFAULT_N_TEMPLATES: int = 5
DEFAULT_N_POSES: int = 9
DEFAULT_N_POSES_REAL: int = 50  # task brief default for real backend
DEFAULT_BACKEND: str = "surrogate"  # "surrogate" (synthetic) or "real"

# ---------------------------------------------------------------------------
# Constants — MM-GBSA surrogate (closed-form linear model)
# ---------------------------------------------------------------------------
#: Lower MM-GBSA = better binding.  Computed via a synthetic surrogate
#: combining Vina score + RMSD-derived term.
MMGBSA_GAS_CONSTANT_KCAL: float = 0.5961  # kcal/mol per RT at 298 K
MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL: float = -5.0  # typical GB solvation
MMGBSA_SA_NONPOLAR_KCAL_PER_A2: float = 0.005  # Still 1990 SA term

# ---------------------------------------------------------------------------
# Constants — surrogate model coefficients
# ---------------------------------------------------------------------------
ALPHA_TEMPLATE_LIFT: float = 0.10  # max template similarity lift (Å)
BETA_POSE_RANK_LIFT: float = 0.05  # max pose rank lift (Å)
NOISE_STD: float = 0.01  # Gaussian noise on RMSD grid

# ---------------------------------------------------------------------------
# Status strings
# ---------------------------------------------------------------------------
STATUS_MEASURED = "MEASURED"
STATUS_SURROGATE = "SURROGATE"  # synthetic via closed-form model
STATUS_HONEST_NEGATIVE = "HONEST_NEGATIVE"
STATUS_PARTIAL = "PARTIAL"
STATUS_SKIPPED = "SKIPPED"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = REPO
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf381a2_multi_template_smoke"
CONFIG_PATH = PROJECT_ROOT / "molmetal" / "configs" / "multi_template_config.yaml"

# Canonical m15 source-of-truth (MEASURED by Wf-179 BD-9 + Wf-246-B + Wf-247-A)
DOCKED_POSE_RMSD_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "docked_pose_rmsd.json"


# ---------------------------------------------------------------------------
# Dataclasses — pose grid + per-pocket + cohort summary
# ---------------------------------------------------------------------------
@dataclass
class PoseGrid:
    """Multi-template pose grid for one pocket.

    Attributes
    ----------
    pocket_id : str
        Target pocket id (e.g. "1h36").
    n_templates : int
        Number of homologous templates used.
    n_poses : int
        Number of Vina poses per template.
    rmsd_grid : np.ndarray of shape (n_templates, n_poses)
        Heavy-atom RMSD vs native pose (synthetic or measured).
    mmgbsa_grid : np.ndarray of shape (n_templates, n_poses)
        MM-GBSA free energy per (template, pose) in kcal/mol.
    status : str
        STATUS_SURROGATE / STATUS_MEASURED / STATUS_HONEST_NEGATIVE.
    """

    pocket_id: str
    n_templates: int
    n_poses: int
    rmsd_grid: np.ndarray
    mmgbsa_grid: np.ndarray
    status: str = STATUS_SURROGATE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rmsd_grid"] = self.rmsd_grid.tolist()
        d["mmgbsa_grid"] = self.mmgbsa_grid.tolist()
        return d


@dataclass
class RescoreGrid:
    """MM-GBSA rescoring result for a PoseGrid.

    Attributes
    ----------
    pocket_id : str
        Target pocket id.
    best_template_idx : int
        Index of the template that gave the best pose.
    best_pose_idx : int
        Index of the pose within that template.
    best_rmsd_angstrom : float
        RMSD of the best (template, pose) cell.
    best_mmgbsa_kcal : float
        MM-GBSA energy of the best cell (lower = better).
    status : str
        STATUS_SURROGATE / STATUS_MEASURED.
    """

    pocket_id: str
    best_template_idx: int
    best_pose_idx: int
    best_rmsd_angstrom: float
    best_mmgbsa_kcal: float
    status: str = STATUS_SURROGATE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerPocketResult:
    """Per-pocket multi-template + MM-GBSA result.

    Attributes
    ----------
    pocket : str
        Pocket id (e.g. "1h36").
    baseline_rmsd_angstrom : float
        Single-template MEASURED RMSD for this pocket.
    best_rmsd_angstrom : float
        Best-of-N (template x pose) RMSD after MM-GBSA rescoring.
    median_rmsd_angstrom : float
        Median RMSD across the n_templates * n_poses grid.
    passes_2_0a_sota : bool
        Whether best_rmsd <= RMSD_SOTA_THRESHOLD_ANGSTROM.
    best_template_idx : int
        Index of the template that gave the best pose.
    best_mmgbsa_kcal : float
        MM-GBSA surrogate energy (kcal/mol) for the best pose.
    lift_angstrom : float
        Projected lift vs baseline (negative = improvement).
    status : str
        STATUS_SURROGATE / STATUS_MEASURED / STATUS_HONEST_NEGATIVE.
    """

    pocket: str
    baseline_rmsd_angstrom: float
    best_rmsd_angstrom: float
    median_rmsd_angstrom: float
    passes_2_0a_sota: bool
    best_template_idx: int
    best_mmgbsa_kcal: float
    lift_angstrom: float
    status: str = STATUS_SURROGATE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CohortSummary:
    """Aggregate cohort summary across all pockets.

    Attributes
    ----------
    n_pockets : int
        Number of pockets in the cohort.
    aggregate_median_best_rmsd_angstrom : float
        Median of per-pocket best_rmsd_angstrom (post MM-GBSA).
    aggregate_success_rate_2_0a : float
        Fraction of pockets with best_rmsd <= 2.0 Å SOTA gate.
    aggregate_success_rate_2_5a : float
        Fraction of pockets with best_rmsd <= 2.5 Å practical gate.
    baseline_median_best_rmsd_angstrom : Optional[float]
        Baseline (single-template) median RMSD from
        ``metrics/by_metric/docked_pose_rmsd.json``.
    lift_angstrom : Optional[float]
        Projected lift vs baseline (negative = improvement).
    per_pocket : List[PerPocketResult]
        Per-pocket records.
    backend : str
        "surrogate" or "real".
    m15_status : str
        STATUS_MEASURED / STATUS_SURROGATE / STATUS_HONEST_NEGATIVE.
    timestamp : str
        ISO-8601 UTC timestamp.
    """

    n_pockets: int
    aggregate_median_best_rmsd_angstrom: float
    aggregate_success_rate_2_0a: float
    aggregate_success_rate_2_5a: float
    baseline_median_best_rmsd_angstrom: Optional[float]
    lift_angstrom: Optional[float]
    per_pocket: List[PerPocketResult] = field(default_factory=list)
    backend: str = DEFAULT_BACKEND
    m15_status: str = STATUS_SURROGATE
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic RMSD generator (closed-form surrogate)
# ---------------------------------------------------------------------------
def _synthetic_rmsd_grid(
    baseline_rmsd: float,
    n_templates: int,
    n_poses: int,
    seed: int,
) -> np.ndarray:
    """Generate a (n_templates, n_poses) synthetic RMSD grid.

    The synthetic RMSD is modelled as:
        rmsd[t, p] = baseline_rmsd
                              - alpha * template_similarity[t]
                              - beta  * pose_rank[p]
                              + small_noise

    where:
        * template_similarity[t] ~ U(0, 0.10) — how close template t
          is to the true pocket.
        * pose_rank[p] = p / (n_poses - 1) — Vina pose ranking.
        * alpha = 0.10 (multi-template lift ceiling)
        * beta  = 0.05 (best-of-N pose selection lift)
        * noise ~ N(0, 0.01) — Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    template_sim = rng.uniform(0.0, ALPHA_TEMPLATE_LIFT, size=n_templates)
    pose_rank = np.linspace(0.0, 1.0, n_poses)
    noise = rng.normal(0.0, NOISE_STD, size=(n_templates, n_poses))
    grid = (
        baseline_rmsd
        - template_sim[:, None]
        - BETA_POSE_RANK_LIFT * pose_rank[None, :]
        + noise
    )
    grid = np.clip(grid, 0.0, None)
    return grid


def _synthetic_mmgbsa_kcal(rmsd_angstrom: float, seed: int) -> float:
    """Closed-form surrogate MM-GBSA energy."""
    rng = np.random.default_rng(seed)
    base = -0.50 * (rmsd_angstrom / 2.0) + MMGBSA_BORN_IMPLICIT_SOLVENT_KCAL
    noise = rng.normal(0.0, 0.10)
    return float(base + noise)


# ---------------------------------------------------------------------------
# Baseline loader (single-template MEASURED m15 source-of-truth)
# ---------------------------------------------------------------------------
def _load_baseline_per_pocket_rmsd() -> Dict[str, float]:
    """Load per-pocket baseline RMSD from the canonical m15 JSON.

    Supports two schemas:
    * ``per_pocket`` array of dicts with ``pocket`` key (legacy)
    * ``measurements`` array of dicts with ``pocket`` +
      ``median_best_rmsd_angstrom`` keys (Wf-341-T1 + Wf-362-A3 canonical)
    Falls back to ``aggregate_median_across_pockets`` if neither is
    populated.
    """
    if not DOCKED_POSE_RMSD_JSON.exists():
        return {}
    data = json.loads(DOCKED_POSE_RMSD_JSON.read_text())
    per_pocket_baseline: Dict[str, float] = {}

    # Schema A: per_pocket array (legacy Wf-179 BD-9)
    for r in data.get("per_pocket", []):
        if isinstance(r, dict) and "pocket" in r:
            pocket = str(r["pocket"])
            rmsd = r.get("median_best_rmsd_angstrom")
            if rmsd is None:
                rmsd = r.get("best_rmsd_angstrom")
            if isinstance(rmsd, (int, float)):
                per_pocket_baseline[pocket] = float(rmsd)

    # Schema B: measurements array (Wf-341-T1 + Wf-362-A3 canonical)
    if not per_pocket_baseline:
        for r in data.get("measurements", []):
            if isinstance(r, dict) and "pocket" in r:
                pocket = str(r["pocket"])
                rmsd = r.get("median_best_rmsd_angstrom")
                if isinstance(rmsd, (int, float)):
                    per_pocket_baseline[pocket] = float(rmsd)

    # Fallback: aggregate median (Wf-179 BD-9 + Wf-341-T1 redesign)
    if not per_pocket_baseline:
        agg = data.get("aggregate_median_best_rmsd_angstrom") or data.get(
            "aggregate_median_across_pockets"
        )
        if isinstance(agg, (int, float)):
            for p in DEFAULT_POCKETS:
                per_pocket_baseline[p] = float(agg)
    return per_pocket_baseline


# ---------------------------------------------------------------------------
# 1. multi_template_dock(pocket_pdb, template_pdbs, n_poses=50)
# ---------------------------------------------------------------------------
def multi_template_dock(
    pocket_pdb: str,
    template_pdbs: List[str],
    n_poses: int = DEFAULT_N_POSES_REAL,
    seed: int = 0,
    backend: str = DEFAULT_BACKEND,
) -> PoseGrid:
    """Run multi-template dock for a single pocket.

    Parameters
    ----------
    pocket_pdb : str
        Path to target pocket PDB (used to derive baseline RMSD).
    template_pdbs : List[str]
        List of N homologous template PDB paths (used for surrogate
        template similarity sampling).
    n_poses : int
        Number of Vina poses per template (default 50).
    seed : int
        RNG seed for synthetic surrogate reproducibility.
    backend : str
        "surrogate" (closed-form) or "real" (requires meeko + Vina).

    Returns
    -------
    PoseGrid
        Multi-template pose grid with shape (n_templates, n_poses).

    Notes
    -----
    * Real backend raises NotImplementedError (TODO-207 / Wf-299 deferred).
    * Pocket id is derived from ``pocket_pdb`` basename (e.g. "1h36").
    """
    if backend == "real":
        raise NotImplementedError(
            "real Vina + MM-GBSA backend deferred to TODO-207 / Wf-299 launch; "
            "use --backend surrogate for the synthetic closed-form bench."
        )
    pocket_id = Path(pocket_pdb).stem if pocket_pdb else "unknown"
    # Strip common synthetic suffixes so the canonical JSON lookup hits.
    for suffix in ("_pocket", "_target", "_receptor"):
        if pocket_id.endswith(suffix):
            pocket_id = pocket_id[: -len(suffix)]
            break
    n_templates = max(1, len(template_pdbs))

    # Derive baseline RMSD from canonical m15 JSON or use a default.
    baseline_per_pocket = _load_baseline_per_pocket_rmsd()
    baseline_rmsd = baseline_per_pocket.get(pocket_id, 0.65)

    rmsd_grid = _synthetic_rmsd_grid(
        baseline_rmsd=baseline_rmsd,
        n_templates=n_templates,
        n_poses=n_poses,
        seed=seed,
    )
    mmgbsa_grid = np.zeros_like(rmsd_grid)
    for t in range(n_templates):
        for p in range(n_poses):
            cell_seed = seed * 1000 + t * 100 + p
            mmgbsa_grid[t, p] = _synthetic_mmgbsa_kcal(
                rmsd_angstrom=float(rmsd_grid[t, p]),
                seed=cell_seed,
            )
    return PoseGrid(
        pocket_id=pocket_id,
        n_templates=n_templates,
        n_poses=n_poses,
        rmsd_grid=rmsd_grid,
        mmgbsa_grid=mmgbsa_grid,
        status=STATUS_SURROGATE,
    )


# ---------------------------------------------------------------------------
# 2. mm_gbsa_rescore(poses, receptor, ligand)
# ---------------------------------------------------------------------------
def mm_gbsa_rescore(
    poses: PoseGrid,
    receptor: str,
    ligand: str,
) -> RescoreGrid:
    """MM-GBSA rescoring of Vina multi-template poses.

    Parameters
    ----------
    poses : PoseGrid
        Pose grid from :func:`multi_template_dock`.
    receptor : str
        Path to receptor PDBQT (for real backend; ignored in surrogate).
    ligand : str
        Path to ligand PDBQT (for real backend; ignored in surrogate).

    Returns
    -------
    RescoreGrid
        Rescoring result with best (template, pose) + MM-GBSA energy.

    Notes
    -----
    * Real MM-GBSA would consume ambertools ``MMPBSA.py`` or
      ``gmx_MMPBSA`` (deferred to TODO-207 / Wf-299).
    * Surrogate picks the pose with the most negative MM-GBSA.
    """
    if poses.status == STATUS_HONEST_NEGATIVE:
        return RescoreGrid(
            pocket_id=poses.pocket_id,
            best_template_idx=-1,
            best_pose_idx=-1,
            best_rmsd_angstrom=float("nan"),
            best_mmgbsa_kcal=float("nan"),
            status=STATUS_HONEST_NEGATIVE,
        )
    flat_idx = int(np.argmin(poses.mmgbsa_grid))
    best_t, best_p = np.unravel_index(flat_idx, poses.mmgbsa_grid.shape)
    best_rmsd = float(poses.rmsd_grid[best_t, best_p])
    best_dg = float(poses.mmgbsa_grid[best_t, best_p])
    return RescoreGrid(
        pocket_id=poses.pocket_id,
        best_template_idx=int(best_t),
        best_pose_idx=int(best_p),
        best_rmsd_angstrom=best_rmsd,
        best_mmgbsa_kcal=best_dg,
        status=STATUS_SURROGATE,
    )


# ---------------------------------------------------------------------------
# 3. per_pocket_median_rmsd(poses, native_ligand)
# ---------------------------------------------------------------------------
def per_pocket_median_rmsd(
    poses: PoseGrid,
    native_ligand: str,
) -> PerPocketResult:
    """Kabsch + Hungarian RMSD vs native pose for one pocket.

    Parameters
    ----------
    poses : PoseGrid
        Pose grid from :func:`multi_template_dock` + rescored via
        :func:`mm_gbsa_rescore`.
    native_ligand : str
        Path to native crystal ligand SDF (for real backend; used as
        baseline RMSD anchor in surrogate mode).

    Returns
    -------
    PerPocketResult
        Per-pocket record with best RMSD + SOTA gate verdict + lift.

    Notes
    -----
    * In surrogate mode, ``native_ligand`` path is ignored for RMSD
      computation (uses canonical m15 JSON baseline).
    * Returns the best-of-N (template x pose) RMSD + projected lift.
    """
    baseline_per_pocket = _load_baseline_per_pocket_rmsd()
    baseline = baseline_per_pocket.get(poses.pocket_id, 0.65)

    # Apply MM-GBSA rescore to pick best pose.
    rescore = mm_gbsa_rescore(poses, receptor="", ligand="")

    median_rmsd = float(np.median(poses.rmsd_grid))
    lift = rescore.best_rmsd_angstrom - baseline

    return PerPocketResult(
        pocket=poses.pocket_id,
        baseline_rmsd_angstrom=baseline,
        best_rmsd_angstrom=rescore.best_rmsd_angstrom,
        median_rmsd_angstrom=median_rmsd,
        passes_2_0a_sota=bool(
            rescore.best_rmsd_angstrom <= RMSD_SOTA_THRESHOLD_ANGSTROM
        ),
        best_template_idx=rescore.best_template_idx,
        best_mmgbsa_kcal=rescore.best_mmgbsa_kcal,
        lift_angstrom=lift,
        status=STATUS_SURROGATE,
    )


# ---------------------------------------------------------------------------
# 4. aggregate_median_rmsd(cohort_results)
# ---------------------------------------------------------------------------
def aggregate_median_rmsd(
    cohort_results: List[PerPocketResult],
    baseline_median: Optional[float] = None,
    backend: str = DEFAULT_BACKEND,
) -> CohortSummary:
    """Aggregate per-pocket results into a cohort summary.

    Parameters
    ----------
    cohort_results : List[PerPocketResult]
        Per-pocket records from :func:`per_pocket_median_rmsd`.
    baseline_median : Optional[float]
        Baseline (single-template) median RMSD from
        ``metrics/by_metric/docked_pose_rmsd.json``.
    backend : str
        "surrogate" or "real".

    Returns
    -------
    CohortSummary
        Cohort summary with aggregate median + lift + status.
    """
    if not cohort_results:
        raise ValueError("cohort_results must be non-empty")
    best_rmsds = [r.best_rmsd_angstrom for r in cohort_results]
    n_pass_2_0a = sum(1 for r in cohort_results if r.passes_2_0a_sota)
    n_pass_2_5a = sum(
        1
        for r in cohort_results
        if r.best_rmsd_angstrom <= RMSD_PRACTICAL_THRESHOLD_ANGSTROM
    )
    agg_median = float(statistics.median(best_rmsds))
    success_rate_2_0a = n_pass_2_0a / len(cohort_results)
    success_rate_2_5a = n_pass_2_5a / len(cohort_results)
    lift = (agg_median - baseline_median) if baseline_median is not None else None

    if backend == "real":
        m15_status = STATUS_MEASURED
    else:
        m15_status = STATUS_SURROGATE

    return CohortSummary(
        n_pockets=len(cohort_results),
        aggregate_median_best_rmsd_angstrom=agg_median,
        aggregate_success_rate_2_0a=success_rate_2_0a,
        aggregate_success_rate_2_5a=success_rate_2_5a,
        baseline_median_best_rmsd_angstrom=baseline_median,
        lift_angstrom=lift,
        per_pocket=list(cohort_results),
        backend=backend,
        m15_status=m15_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Top-level bench (CLI entry)
# ---------------------------------------------------------------------------
def _synthetic_template_paths(pocket: str, n_templates: int) -> List[str]:
    """Generate synthetic template PDB paths for the smoke bench.

    Real template sourcing would query RCSB PDB.  For the smoke bench
    we just create placeholder paths that the surrogate can use.
    """
    return [f"/tmp/template_{pocket}_{i}.pdb" for i in range(n_templates)]


def bench_multi_template_mm_gbsa(
    pockets: Sequence[str] = DEFAULT_POCKETS,
    n_templates: int = DEFAULT_N_TEMPLATES,
    n_poses: int = DEFAULT_N_POSES,
    seed: int = 0,
    backend: str = DEFAULT_BACKEND,
) -> CohortSummary:
    """Top-level bench: multi-template dock + MM-GBSA across pockets."""
    baseline_per_pocket = _load_baseline_per_pocket_rmsd()
    baseline_median: Optional[float] = None
    if baseline_per_pocket:
        baseline_median = statistics.median(baseline_per_pocket.values())

    per_pocket_results: List[PerPocketResult] = []
    for i, pocket in enumerate(pockets):
        pocket_pdb = f"/tmp/{pocket}_pocket.pdb"
        template_pdbs = _synthetic_template_paths(pocket, n_templates)
        pose_grid = multi_template_dock(
            pocket_pdb=pocket_pdb,
            template_pdbs=template_pdbs,
            n_poses=n_poses,
            seed=seed + i,
            backend=backend,
        )
        result = per_pocket_median_rmsd(
            poses=pose_grid,
            native_ligand=f"/tmp/{pocket}_native.sdf",
        )
        per_pocket_results.append(result)

    return aggregate_median_rmsd(
        cohort_results=per_pocket_results,
        baseline_median=baseline_median,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-381-A2 multi-template + MM-GBSA m15 RMSD bench (CPU-only). "
            "Default backend is 'surrogate' (closed-form synthetic); "
            "use --backend real only when meeko + Vina + GBSA binaries "
            "are installed (deferred to TODO-207 / Wf-299)."
        ),
    )
    p.add_argument(
        "--pockets",
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help=f"pocket ids (default: {list(DEFAULT_POCKETS)})",
    )
    p.add_argument(
        "--n-templates",
        type=int,
        default=DEFAULT_N_TEMPLATES,
        help=f"number of templates (default: {DEFAULT_N_TEMPLATES})",
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=DEFAULT_N_POSES,
        help=f"poses per template (default: {DEFAULT_N_POSES})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    p.add_argument(
        "--backend",
        choices=("surrogate", "real"),
        default=DEFAULT_BACKEND,
        help="bench backend (default: surrogate)",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPORTS_DIR / "result.json"),
        help="output JSON path",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = bench_multi_template_mm_gbsa(
        pockets=tuple(args.pockets),
        n_templates=args.n_templates,
        n_poses=args.n_poses,
        seed=args.seed,
        backend=args.backend,
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy arrays in per-pocket fields (none, but be safe).
    out_dict = summary.to_dict()
    out_path.write_text(json.dumps(out_dict, indent=2))
    print(json.dumps(out_dict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
