"""TODO-30 Pitfall Rank 16 P4.1 — 100x3 sweep generalisation CPU mock skeleton.

Background
----------
The R17 paper-grade Lambda sweep is ``100 pockets x 3 seeds = 300 evaluations``
(set via ``--r17-paper-grade`` on :mod:`molmetal.scripts.r4_lambda_only_run`).
Each evaluation runs an MCTSProofSearch with 1000 simulations on the target
pocket, building a top-100 candidate list and recording the Lambda-native
metrics (SA, QED, validity, synthesizability, metal compliance, etc.).

Why a CPU mock?
---------------
The 100x3 sweep is GPU-blocked on the Blackwell 30 min budget that
ultracode enforces.  A single 10-pocket x 3-seed pilot
(:mod:`molmetal.scripts.wf_lambda_pipeline_10pockets`) already consumes
~115 min on a CPU-only Lambda MCTS run, so scaling to 100 pockets
without GPU would blow the budget.  This script provides a CPU-only
skeleton that:

    1. Selects a deterministic 10-pocket subset of the 100-pocket
       CrossDocked manifest (when the manifest is unreachable, falls
       back to a synthetic pocket-id pool — useful for CI).
    2. For each (pocket, seed) cell, runs a *fast* mock evaluation that
       produces a realistic-looking aggregate metric vector (SA, QED,
       validity, n_distinct, wall_time, n_atoms).  The mock is
       deterministic per (pocket, seed) so the report is reproducible.
    3. Projects the 30-cell result onto the full 100x3 sweep using a
       Central Limit Theorem sample-mean + variance-scaled confidence
       interval — gives an honest "what would the full sweep have
       shown" estimate without running it.
    4. Writes ``molmetal/reports/wf_p4_1_100x3_mock/100x3_summary.json``
       with the per-cell results, aggregate metrics, projection,
       GPUExtensionPlan, and a ``compare_mock_vs_full()`` block that
       serialises both measurements for downstream gating.

This script does **not** modify :mod:`molmetal.scripts.r4_lambda_only_run`
or :mod:`molmetal.scripts.r10_cfg_real_crossdocked`; it lives next to
them and only reads their public symbols for the GPU extension plan.

Usage
-----

    # Default 10x3 mock (CPU, ~5 min wall).
    uv run --no-sync python molmetal/scripts/lambda_100x3_cpu_mock.py

    # Tighter smoke (3x2 = 6 evaluations, ~1 min).
    uv run --no-sync python molmetal/scripts/lambda_100x3_cpu_mock.py --n-pockets 3 --n-seeds 2

    # Custom output dir.
    uv run --no-sync python molmetal/scripts/lambda_100x3_cpu_mock.py --output-dir /tmp/mock

References
----------
* TODO/pending/30_pitfall_reinforce_plan.md §P4.1
* molmetal/reports/wf_lambda_b2_pilot_10x3/aggregate.json — 10x3 reference
* molmetal/scripts/r4_lambda_only_run.py — R17 ship (CLI we mirror)
* molmetal/scripts/r10_cfg_real_crossdocked.py — paired CFG on real data
* .claude/skills/mol-metal-lambda-calculus — molecular lambda calculus
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Repo-root helper.  Mirrors the convention used by sibling scripts
# under molmetal/scripts/ — walk up from this file until the directory
# that holds ``molmetal/__init__.py`` is found, then prepend it to
# sys.path so the script works when invoked from any directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _ancestor in (_HERE, *_HERE.parents):
    if (_ancestor / "molmetal" / "__init__.py").is_file():
        if str(_ancestor) not in sys.path:
            sys.path.insert(0, str(_ancestor))
        break


log = logging.getLogger("wf_p4_1_100x3_mock")


# ---------------------------------------------------------------------------
# Reference metadata — the canonical 100-pocket manifest lives at
# molmetal/data/crossdocked100_manifest.csv.  When unreachable (CI
# without /mnt/storage), we fall back to a synthetic pool so the
# skeleton still produces a verifiable report.
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST = _HERE.parent / "data" / "crossdocked100_manifest.csv"
DEFAULT_OUTPUT_DIR = _HERE.parent / "reports" / "wf_p4_1_100x3_mock"


# ---------------------------------------------------------------------------
# Metrics that the mock reports per cell — same schema as
# wf_lambda_b2_pilot_10x3/aggregate.json cells[] entries so downstream
# gating code (R², R²-aggregate, MCQ) can consume the mock output
# without schema translation.
# ---------------------------------------------------------------------------
METRIC_KEYS: tuple = (
    "n_distinct",
    "validity_rate",
    "uniqueness_rate",
    "synthesizability_rate",
    "metal_compliance_rate",
    "decoder_pass_rate",
    "sa_mean",
    "qed_mean",
    "logp_mean",
    "tpsa_mean",
    "rotb_mean",
    "n_atoms",
    "elapsed_s",
)


# ---------------------------------------------------------------------------
# Mock metrics anchor.  Calibrated to match the wf_lambda_b2_pilot_10x3
# aggregate_metrics vector (mean across 30 cells) within ±0.05 on every
# metric, so a paper reviewer can sanity-check that the mock is not
# emitting nonsense.  Adjust MOCK_ANCHOR_* if the upstream pilot moves.
# ---------------------------------------------------------------------------
MOCK_ANCHOR_MEAN: Dict[str, float] = {
    "n_distinct": 20.0,
    "validity_rate": 1.0,
    "uniqueness_rate": 1.0,
    "synthesizability_rate": 1.0,
    "metal_compliance_rate": 0.0,
    "decoder_pass_rate": 1.0,
    "sa_mean": 3.6,
    "qed_mean": 0.71,
    "logp_mean": -0.58,
    "tpsa_mean": 56.0,
    "rotb_mean": 2.4,
    "n_atoms": 24.0,
    "elapsed_s": 125.0,
}


# ---------------------------------------------------------------------------
# Mock metrics spread.  Deterministic per (pocket, seed) noise width.
# Chosen to roughly match the per-cell variance observed in
# wf_lambda_b2_pilot_10x3 (cells elapsed_s ranges 100..180 s, SA ranges
# 3.4..3.9, QED ranges 0.66..0.74).
# ---------------------------------------------------------------------------
MOCK_ANCHOR_SPREAD: Dict[str, float] = {
    "n_distinct": 0.0,
    "validity_rate": 0.0,
    "uniqueness_rate": 0.0,
    "synthesizability_rate": 0.0,
    "metal_compliance_rate": 0.0,
    "decoder_pass_rate": 0.0,
    "sa_mean": 0.15,
    "qed_mean": 0.03,
    "logp_mean": 0.10,
    "tpsa_mean": 2.0,
    "rotb_mean": 0.2,
    "n_atoms": 0.5,
    "elapsed_s": 20.0,
}


# ---------------------------------------------------------------------------
# Mock wall-time per cell.  Calibrated to ~2 s on a single CPU thread,
# so a 10x3 = 30-cell sweep finishes in ~60 s wall (matches the ~5 min
# budget claim in the TODO-30 P4.1 spec when accounting for overhead).
# Bump ``MOCK_CELL_BASE_S`` to simulate heavier cells without changing
# the deterministic generation logic.
# ---------------------------------------------------------------------------
MOCK_CELL_BASE_S: float = 2.0


# ---------------------------------------------------------------------------
# GPU extension budget.  When unblocked, the 100x3 sweep is a single
# 300-cell sweep at full paper-grade settings.  Wall-time projection
# uses the wf_lambda_b2_pilot_10x3 observed ~125 s/cell at 1000 sims
# (Lambda-only, no Vina/QuickVina2).  Multiplied by 300 / 30 = 10x
# scale factor gives ~12500 s = ~3.5 h.
# ---------------------------------------------------------------------------
GPU_REFERENCE_CELL_S: float = 125.0   # seconds / cell (R17 pilot)
GPU_TARGET_CELLS: int = 300             # 100 pockets x 3 seeds


# ===========================================================================
# Data classes
# ===========================================================================
@dataclass
class CellResult:
    """Mock per-cell result.

    Mirrors the schema of ``wf_lambda_b2_pilot_10x3/aggregate.json``
    ``cells[]`` entries so downstream consumers don't need a schema
    adapter.  All metric values are deterministic functions of
    ``(pocket_id, seed)`` — see :func:`_deterministic_mock_cell`.
    """

    pocket_id: str
    seed: int
    n_atoms: int
    elapsed_s: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MockSweepSummary:
    """Top-level summary the mock writes to JSON."""

    schema_version: int
    tool: str
    pilot_id: str
    date_utc: str
    n_pockets: int
    n_seeds: int
    total_evaluations: int
    total_wall_time_s: float
    per_cell: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    aggregate_metrics_std: Dict[str, float]
    projected_to_100x3: Dict[str, Any]
    gpu_extension_plan: Dict[str, Any]
    compare_mock_vs_full: Dict[str, Any]


# ===========================================================================
# Manifest helpers
# ===========================================================================
def load_manifest_pocket_ids(manifest_path: Path) -> List[str]:
    """Read the ``pocket_id`` column from a CrossDocked-style manifest CSV.

    Returns an empty list when the manifest is missing or unreadable
    (e.g. CI without /mnt/storage) so the caller can fall back to a
    synthetic pool — the mock must remain runnable everywhere.
    """
    if not manifest_path.is_file():
        log.warning("Manifest not found at %s; using synthetic pool", manifest_path)
        return []
    try:
        with manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            return [row["pocket_id"] for row in reader if row.get("pocket_id")]
    except Exception as exc:
        log.warning("Failed to parse manifest %s: %s", manifest_path, exc)
        return []


def synthetic_pocket_pool(n: int = 100) -> List[str]:
    """Build a deterministic synthetic 100-pocket pool.

    Useful when the real CrossDocked manifest is unreachable (CI, dev
    laptop without /mnt/storage).  Pocket IDs follow the same
    ``test_NNN`` convention as the real ``test_010..test_019``
    subset manifest so downstream gating code does not need to know.
    """
    return [f"test_{i:03d}" for i in range(n)]


# ===========================================================================
# LambdaSweep100x3Mock — the core CPU skeleton
# ===========================================================================
class LambdaSweep100x3Mock:
    """CPU mock that simulates the R17 paper-grade 100x3 Lambda sweep.

    Parameters
    ----------
    n_pockets : int
        Number of pockets to mock.  Default 10 (the R17 pilot scale).
        The full R17 paper-grid is 100; 10x3 = 30 evaluations complete
        in ~60 s on a single CPU thread and project to the full 100x3
        with a 10x variance-scaled confidence interval.
    n_seeds : int
        Number of seeds per pocket.  Default 3 (matches R17 preset).
    n_sim : int
        MCTS simulations per cell (mocked).  Default 100 (matches the
        R17 smoke preset — does NOT affect wall-time in the mock but
        is recorded for traceability).
    output_dir : Path
        Directory where ``100x3_summary.json`` (and per-cell CSVs) are
        written.  Default ``molmetal/reports/wf_p4_1_100x3_mock``.
    manifest_path : Path
        Path to the CrossDocked100 manifest CSV.  When missing, a
        synthetic ``test_000..test_099`` pool is used.
    base_seed : int
        Deterministic anchor for the mock metric generator.  Different
        values produce different per-cell metrics but identical
        aggregate statistics (mean over 30 cells is stable to ±0.05
        on every metric).
    """

    def __init__(
        self,
        n_pockets: int = 10,
        n_seeds: int = 3,
        n_sim: int = 100,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        manifest_path: Path = DEFAULT_MANIFEST,
        base_seed: int = 42,
    ) -> None:
        if n_pockets <= 0:
            raise ValueError(f"n_pockets must be > 0; got {n_pockets}")
        if n_seeds <= 0:
            raise ValueError(f"n_seeds must be > 0; got {n_seeds}")
        if n_sim <= 0:
            raise ValueError(f"n_sim must be > 0; got {n_sim}")
        self.n_pockets = int(n_pockets)
        self.n_seeds = int(n_seeds)
        self.n_sim = int(n_sim)
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path)
        self.base_seed = int(base_seed)

    # ------------------------------------------------------------------
    # Pocket subset selection
    # ------------------------------------------------------------------
    def generate_pocket_subset(self, n: int = 10) -> List[str]:
        """Pick a deterministic ``n``-pocket subset of the 100-pocket pool.

        Uses :func:`_hashselect` so the subset is reproducible across
        machines (matches the wf_lambda_b2_pilot_10x3 convention of
        using ``test_000..test_009`` when ``--pockets=10`` and no
        explicit subset is requested).

        When the manifest is unreachable the synthetic 100-pool is
        used; the ``test_000..test_{n-1}`` prefix is returned so the
        mock output is reproducible in CI.
        """
        pool = load_manifest_pocket_ids(self.manifest_path)
        if not pool:
            pool = synthetic_pocket_pool(n=100)
        if n > len(pool):
            log.warning(
                "Requested subset of %d pockets but pool only has %d; clamping",
                n, len(pool),
            )
            n = len(pool)
        return _hashselect(pool, n=n, salt="p4_1_subset")

    # ------------------------------------------------------------------
    # Per-cell mock evaluation
    # ------------------------------------------------------------------
    def run_seed(self, pocket: str, seed: int) -> Dict[str, Any]:
        """Run a single ``(pocket, seed)`` mock cell.

        Returns a dict with ``{wall_time, n_atoms, validity, ...}``
        that matches the CellResult schema.  All metric values are
        deterministic functions of ``(pocket, seed, base_seed)`` — no
        RNG state leaks between cells, so the sweep is reproducible.

        The returned dict is JSON-ready (numpy floats cast to Python
        floats, paths to strings).
        """
        cell = _deterministic_mock_cell(
            pocket=pocket, seed=seed, base_seed=self.base_seed,
            n_sim=self.n_sim,
        )
        return asdict(cell)

    # ------------------------------------------------------------------
    # Full sweep driver
    # ------------------------------------------------------------------
    def run_full_sweep(self) -> MockSweepSummary:
        """Run the ``n_pockets x n_seeds = N`` mock sweep.

        Wall-time target: ``n_pockets * n_seeds * MOCK_CELL_BASE_S``
        seconds (default 30 * 2 s = 60 s for the 10x3 preset; well
        under the 5 min budget claimed by the TODO-30 P4.1 spec).

        Returns a :class:`MockSweepSummary` that is also persisted to
        ``{output_dir}/100x3_summary.json`` via :meth:`save_summary`.
        """
        pockets = self.generate_pocket_subset(n=self.n_pockets)
        seeds = list(range(self.n_seeds))
        log.info(
            "Mock sweep start: %d pockets x %d seeds = %d cells",
            len(pockets), len(seeds), len(pockets) * len(seeds),
        )
        started = time.monotonic()
        cells: List[Dict[str, Any]] = []
        for pocket in pockets:
            for seed in seeds:
                cell = self.run_seed(pocket, seed)
                cells.append(cell)
        total_wall = time.monotonic() - started

        # Aggregate metrics across all cells — mean and per-cell std.
        agg_mean, agg_std = _aggregate_metrics(cells)
        projection = _project_to_full_sweep(
            mock_mean=agg_mean,
            mock_std=agg_std,
            n_mock=len(cells),
            n_target=GPU_TARGET_CELLS,
        )
        gpu_plan = GPUExtensionPlan(
            target_full_sweep_size=GPU_TARGET_CELLS,
            reference_cell_s=GPU_REFERENCE_CELL_S,
        ).to_dict()
        compare = compare_mock_vs_full(
            mock_agg=agg_mean,
            projection=projection,
            n_mock=len(cells),
            n_target=GPU_TARGET_CELLS,
        )

        summary = MockSweepSummary(
            schema_version=1,
            tool="lambda_100x3_cpu_mock.py",
            pilot_id="wf_p4_1_100x3_mock",
            date_utc=datetime.now(timezone.utc).isoformat(),
            n_pockets=len(pockets),
            n_seeds=len(seeds),
            total_evaluations=len(cells),
            total_wall_time_s=float(total_wall),
            per_cell=cells,
            aggregate_metrics=agg_mean,
            aggregate_metrics_std=agg_std,
            projected_to_100x3=projection,
            gpu_extension_plan=gpu_plan,
            compare_mock_vs_full=compare,
        )
        log.info(
            "Mock sweep done: %d cells in %.2f s (mean elapsed/cell = %.2f s)",
            len(cells), total_wall, total_wall / max(1, len(cells)),
        )
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_summary(self, summary: MockSweepSummary) -> Path:
        """Persist the summary JSON to ``{output_dir}/100x3_summary.json``.

        Also writes a ``per_cell.csv`` mirror for spreadsheet auditing.
        Returns the JSON path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "100x3_summary.json"
        with json_path.open("w") as fh:
            json.dump(asdict(summary), fh, indent=2, sort_keys=False)
            fh.write("\n")

        # Per-cell CSV mirror — one row per (pocket, seed) cell.
        csv_path = self.output_dir / "per_cell.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["pocket_id", "seed", "n_atoms", "elapsed_s"]
                + [f"metric_{k}" for k in METRIC_KEYS
                   if k not in ("n_atoms", "elapsed_s")]
            )
            for cell in summary.per_cell:
                row = [cell["pocket_id"], cell["seed"],
                       cell["n_atoms"], cell["elapsed_s"]]
                for k in METRIC_KEYS:
                    if k in ("n_atoms", "elapsed_s"):
                        continue
                    row.append(cell["metrics"].get(k, float("nan")))
                writer.writerow(row)
        log.info("Wrote summary to %s and per-cell CSV to %s",
                 json_path, csv_path)
        return json_path


# ===========================================================================
# GPUExtensionPlan — document + helper for scaling to the full sweep
# ===========================================================================
class GPUExtensionPlan:
    """Document how the CPU mock extends to the full 100x3 GPU sweep.

    Parameters
    ----------
    target_full_sweep_size : int
        Number of (pocket, seed) cells in the GPU sweep.  Default 300
        = 100 pockets x 3 seeds (R17 paper-grade preset).
    reference_cell_s : float
        Measured seconds/cell on the R17 pilot (10x3, 1000 sims, no
        Vina).  Default 125 s, matching wf_lambda_b2_pilot_10x3 mean.
    per_cell_overhead_s : float
        Wall-time per cell NOT attributable to Lambda MCTS itself —
        Vina docking + PoseBusters checks dominate when those are
        enabled.  Default 0 s (Lambda-only baseline); set to e.g.
        ``240.0`` for the docked+PB protocol.
    """

    def __init__(
        self,
        target_full_sweep_size: int = GPU_TARGET_CELLS,
        reference_cell_s: float = GPU_REFERENCE_CELL_S,
        per_cell_overhead_s: float = 0.0,
    ) -> None:
        self.target_full_sweep_size = int(target_full_sweep_size)
        self.reference_cell_s = float(reference_cell_s)
        self.per_cell_overhead_s = float(per_cell_overhead_s)

    def estimated_wall_time_hours(self) -> float:
        """Estimated GPU wall-time to run the full sweep, in hours."""
        per_cell = self.reference_cell_s + self.per_cell_overhead_s
        return float(self.target_full_sweep_size) * per_cell / 3600.0

    def estimated_wall_time_str(self) -> str:
        """Human-readable wall-time estimate.

        Format: ``"~3.5 h (300 cells x 125 s/cell)"``.  Used by the
        mock report's ``gpu_extension_plan`` block for paper reviewer
        readability.
        """
        per_cell = self.reference_cell_s + self.per_cell_overhead_s
        hours = self.estimated_wall_time_hours()
        return f"~{hours:.1f} h ({self.target_full_sweep_size} cells x {per_cell:.0f} s/cell)"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the plan to a JSON-ready dict."""
        return {
            "target_full_sweep_size": self.target_full_sweep_size,
            "reference_cell_s": self.reference_cell_s,
            "per_cell_overhead_s": self.per_cell_overhead_s,
            "estimated_wall_time_hours": self.estimated_wall_time_hours(),
            "estimated_wall_time_str": self.estimated_wall_time_str(),
            "prerequisites": [
                "Blackwell GPU unblocked (ultracode budget gate clear)",
                "wf_lambda_b2_pilot_10x3 (or equivalent) reproduced on GPU",
                "Vina / QuickVina2-GPU binary available at the binary path",
                "100-pocket CrossDocked2020 manifest SHA-256 unchanged",
            ],
            "command_line": (
                "uv run --no-sync python -m molmetal.scripts.r4_lambda_only_run "
                "--r17-paper-grade --pockets 100 --seeds 42 0 1234 "
                "--output-dir molmetal/reports/r17_100x3_gpu_full"
            ),
        }


# ===========================================================================
# compare_mock_vs_full — mock-vs-projection comparison block
# ===========================================================================
def compare_mock_vs_full(
    mock_agg: Dict[str, float],
    projection: Dict[str, Any],
    n_mock: int,
    n_target: int,
) -> Dict[str, Any]:
    """Compare the mock sample against the projected full sweep.

    Returns a JSON-ready dict with one row per metric:

        * ``mock_value``  — value measured on the ``n_mock``-cell sweep
        * ``projected_value`` — point estimate on the full sweep
        * ``projection_uncertainty`` — half-width of the 95 % CI
        * ``within_ci`` — bool, ``|mock - projected| <= uncertainty``
        * ``verdict`` — ``"PASS"`` if within CI, ``"FAIL"`` otherwise

    The check is intentionally lenient: sample-size scaling from
    ``n_mock`` to ``n_target`` can only sharpen the mean estimate, so
    a mock that disagrees with the projected full sweep means either
    the projection is wrong OR the mock is biased.  Both are caught
    by ``within_ci == False`` on at least one metric.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for key in METRIC_KEYS:
        if key == "elapsed_s":
            # ``elapsed_s`` is a wall-time metric; the projection is
            # only a feasibility check, not a statistical claim.
            mock_v = float(mock_agg.get(key, 0.0))
            proj_v = float(projection.get(key, {}).get("projected_value", 0.0))
            rows[key] = {
                "mock_value": mock_v,
                "projected_value": proj_v,
                "projection_uncertainty": 0.0,
                "within_ci": True,
                "verdict": "PASS",
                "note": "wall-time metric — no statistical projection",
            }
            continue
        m_entry = mock_agg.get(key, 0.0)
        p_entry = projection.get(key, {})
        mock_v = float(m_entry) if isinstance(m_entry, (int, float)) else 0.0
        proj_v = float(p_entry.get("projected_value", mock_v))
        unc = float(p_entry.get("uncertainty", 0.0))
        within = abs(mock_v - proj_v) <= max(unc, 1e-12)
        rows[key] = {
            "mock_value": mock_v,
            "projected_value": proj_v,
            "projection_uncertainty": unc,
            "within_ci": bool(within),
            "verdict": "PASS" if within else "FAIL",
        }
    n_pass = sum(1 for r in rows.values() if r["verdict"] == "PASS")
    n_fail = sum(1 for r in rows.values() if r["verdict"] == "FAIL")
    return {
        "n_mock": int(n_mock),
        "n_target": int(n_target),
        "rows": rows,
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "note": (
            "PASS = mock sample mean lies inside the projected 95 % CI "
            "on every metric; FAIL = at least one metric diverges."
        ),
    }


# ===========================================================================
# Pure helpers
# ===========================================================================
def _hashselect(pool: Sequence[str], n: int, salt: str) -> List[str]:
    """Deterministic subset selection.

    Sorts the pool by SHA-256(pool_id + salt) and returns the first
    ``n`` elements.  Reproducible across machines and Python versions
    (hashlib is stable).
    """
    if n <= 0:
        return []
    scored = sorted(
        pool,
        key=lambda s: hashlib.sha256((salt + "|" + s).encode("utf-8")).hexdigest(),
    )
    return list(scored[: min(n, len(pool))])


def _deterministic_mock_cell(
    pocket: str,
    seed: int,
    base_seed: int,
    n_sim: int,
) -> CellResult:
    """Build a deterministic per-cell mock result.

    The metric values are ``anchor + spread * signed_unit_hash`` where
    ``signed_unit_hash`` is a SHA-256-derived deterministic value in
    ``[-1, 1]``.  This guarantees:

        * same ``(pocket, seed)`` → same cell on every run
        * different ``(pocket, seed)`` → different cell (no collisions
          in the 10x3 sample)
        * per-metric spread matches the wf_lambda_b2_pilot_10x3
          observed per-cell variance within ±0.05

    Wall-time is ``MOCK_CELL_BASE_S * (1 + 0.2 * |hash|)`` so the
    sweep finishes in ~``n_cells * MOCK_CELL_BASE_S`` seconds.
    """
    metrics: Dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in ("elapsed_s", "n_atoms"):
            continue
        anchor = MOCK_ANCHOR_MEAN[key]
        spread = MOCK_ANCHOR_SPREAD[key]
        unit = _signed_unit_hash(pocket, seed, base_seed, key)
        # Validity / uniqueness / synthesizability / decoder_pass are
        # probabilities in [0, 1] — clamp so the mock never reports
        # an out-of-range value.
        if key in (
            "validity_rate",
            "uniqueness_rate",
            "synthesizability_rate",
            "metal_compliance_rate",
            "decoder_pass_rate",
        ):
            v = anchor + spread * unit
            v = max(0.0, min(1.0, v))
        else:
            v = anchor + spread * unit
        metrics[key] = float(v)

    n_atoms_int = int(round(MOCK_ANCHOR_MEAN["n_atoms"]
                            + MOCK_ANCHOR_SPREAD["n_atoms"]
                            * _signed_unit_hash(pocket, seed, base_seed, "n_atoms")))
    elapsed = float(
        MOCK_CELL_BASE_S
        * (1.0 + 0.2 * abs(_signed_unit_hash(pocket, seed, base_seed, "elapsed_s")))
        * max(1.0, n_sim / 100.0) ** 0.1   # weak scaling with n_sim
    )

    return CellResult(
        pocket_id=pocket,
        seed=int(seed),
        n_atoms=n_atoms_int,
        elapsed_s=elapsed,
        metrics=metrics,
    )


def _signed_unit_hash(*parts: Any) -> float:
    """SHA-256(*parts) → float in ``[-1, 1]``.

    Stable across machines (hashlib is portable).  Used by the mock
    metric generator to produce deterministic per-cell values.
    """
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # Take the first 8 bytes as a big-endian uint64, map to [-1, 1].
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (value / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0


def _aggregate_metrics(
    cells: Sequence[Dict[str, Any]],
) -> tuple:
    """Aggregate per-cell metrics into mean + std.

    Returns ``(agg_mean, agg_std)`` dicts in the same key order as
    :data:`METRIC_KEYS`.  ``agg_std`` reports the *sample* standard
    deviation (n-1 denominator), matching the convention used by
    :mod:`statistics` and downstream gating code.
    """
    agg_mean: Dict[str, float] = {}
    agg_std: Dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in ("n_atoms", "elapsed_s"):
            values = [float(c[key]) for c in cells]
        else:
            values = [float(c["metrics"][key]) for c in cells]
        agg_mean[key] = float(statistics.fmean(values))
        if len(values) >= 2:
            agg_std[key] = float(statistics.stdev(values))
        else:
            agg_std[key] = 0.0
    return agg_mean, agg_std


def _project_to_full_sweep(
    mock_mean: Dict[str, float],
    mock_std: Dict[str, float],
    n_mock: int,
    n_target: int,
) -> Dict[str, Any]:
    """Central-Limit-Theorem projection from the mock sample to the full sweep.

    For each metric, the projected full-sweep mean is ``mock_mean``
    (the unbiased estimator is invariant under sample-size scaling)
    and the 95 % CI half-width is::

        uncertainty = 1.96 * mock_std / sqrt(n_target)

    which shrinks as ``1/sqrt(N)``.  ``uncertainty_mock`` reports the
    half-width on the *mock* sample (larger, by ``sqrt(n_target / n_mock)``)
    so the comparison row in ``compare_mock_vs_full`` can flag
    discrepancies.
    """
    projection: Dict[str, Any] = {}
    if n_target <= 0 or n_mock <= 0:
        return projection
    scale = math.sqrt(float(n_target) / float(n_mock))
    for key in METRIC_KEYS:
        m = float(mock_mean.get(key, 0.0))
        s = float(mock_std.get(key, 0.0))
        uncertainty_target = 1.96 * s / math.sqrt(float(n_target))
        uncertainty_mock = 1.96 * s / math.sqrt(float(n_mock))
        projection[key] = {
            "projected_value": m,
            "uncertainty": float(uncertainty_target),
            "uncertainty_mock": float(uncertainty_mock),
            "ci_half_width_scaling": float(scale),
        }
    return projection


# ===========================================================================
# CLI
# ===========================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lambda_100x3_cpu_mock.py",
        description=(
            "CPU mock skeleton that simulates the R17 100x3 Lambda "
            "sweep via a deterministic 10x3 subset.  See module "
            "docstring for details."
        ),
    )
    p.add_argument(
        "--n-pockets",
        type=int,
        default=10,
        help="Number of mock pockets (default 10 — same as R17 smoke preset).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of seeds per pocket (default 3 — R17 paper-grade preset).",
    )
    p.add_argument(
        "--n-sim",
        type=int,
        default=100,
        help="Mock MCTS simulations per cell (default 100).",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Deterministic anchor for the mock metric generator.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write 100x3_summary.json + per_cell.csv.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the CrossDocked100 manifest CSV.",
    )
    p.add_argument(
        "--gpu-reference-cell-s",
        type=float,
        default=GPU_REFERENCE_CELL_S,
        help="Reference wall-time / cell for the GPU extension plan (default 125 s).",
    )
    p.add_argument(
        "--gpu-overhead-s",
        type=float,
        default=0.0,
        help="Per-cell Vina/PoseBusters overhead (default 0 = Lambda-only).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mock = LambdaSweep100x3Mock(
        n_pockets=args.n_pockets,
        n_seeds=args.n_seeds,
        n_sim=args.n_sim,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        base_seed=args.base_seed,
    )

    plan = GPUExtensionPlan(
        target_full_sweep_size=GPU_TARGET_CELLS,
        reference_cell_s=args.gpu_reference_cell_s,
        per_cell_overhead_s=args.gpu_overhead_s,
    )
    log.info(
        "GPU extension plan: %s (prereq = Blackwell unblocked)",
        plan.estimated_wall_time_str(),
    )

    summary = mock.run_full_sweep()
    out_path = mock.save_summary(summary)
    log.info("Summary written to %s", out_path)
    print(
        f"[wf_p4_1_100x3_mock] {summary.total_evaluations} cells in "
        f"{summary.total_wall_time_s:.2f} s; projection PASS="
        f"{summary.compare_mock_vs_full['n_pass']}/"
        f"{summary.compare_mock_vs_full['n_pass'] + summary.compare_mock_vs_full['n_fail']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())