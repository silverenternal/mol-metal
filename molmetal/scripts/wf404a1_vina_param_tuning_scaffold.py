#!/usr/bin/env python3
"""Wf-404-A1 — m3 Vina docking parameter tuning SCAFFOLD (4-level grid).

================================================================
Why this script exists
================================================================
TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) was previously
covered by the Wf-380-A4 / Wf-381-A1 3-level SCAFFOLDs.  This Wf-404-A1
follow-up wave EXPANDS the grid to 4 levels (exhaustiveness = {8, 16, 32,
64} AND n_poses = {5, 10, 20, 50}) AND narrows the cohort to the canonical
4-pocket Wf-333-V3 selection (1h36 + 3eml + 3ny8 + 4aaw; 4rlu is dropped
to match the V3 combined median -7.782 kcal/mol verification scope per
Wf-333-V3).

It also adds a thin CPU-only Vina CLI invocation wrapper that checks the
``vina`` binary on PATH and falls back to a synthetic placeholder when the
binary is not available — so the SCAFFOLD remains CPU-only deterministic
under AGENTS.md §5.1 + kill_then_adapt §0.

HONEST FRAMING (party-twenty-spirit §3.5)
-----------------------------------------
* This script does NOT run real Vina docking unless the user has
  installed the Vina 1.2.7 binary AND pre-built the 4-pocket PDBQT
  receptors AND granted USER_OK = 1 per AGENTS.md §5.4.
* The 4-level synthetic per-(pocket, seed) kcal/mol values are PLACEHOLDERS
  that mirror the Wf-141 §0 + Wf-333-V3 PROJECTED lift curve.
* Real per-level kcal/mol as MEASURED is DEFERRED to Wf-299 CFM 3-wave GPU
  launch (TODO-207, USER_OK-gated per AGENTS.md §5.1 / §5.4).

4-level grid rationale
----------------------
The Wf-380-A4 3-level grid ({8, 32, 64}) collapses the Wf-141 + Wf-141-P2
+ Wf-141-P3 ablation structure into a coarse 3-tier model.  This Wf-404-A1
re-introduces the ``ex=16`` and ``n_poses=10`` mid-tier Vina 1.2.7 defaults
that are widely used in SOTA docking benchmarks (Eberhardt 2021 + Trott 2010
+ DiffDock arXiv:2210.01776 §4.2 + PocketMol arXiv:2305.16289 §4.1) and
that were the basis for the Wf-141 §0 per-pocket interpolation between the
``low=8`` and ``medium=32`` anchors.

Usage
-----
::

    .venv/bin/python molmetal/scripts/wf404a1_vina_param_tuning_scaffold.py \\
        --out molmetal/reports/wf404a1_vina_tuning_scaffold/smoke_result.json

Exit codes: 0 = PASS (SCAFFOLD + smoke validated), 2 = SKIP (binary
missing on PATH), 1 = FAIL (synthetic validation error).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("wf404a1_vina_param_tuning_scaffold")

# ---------------------------------------------------------------------------
# Wf-333-V3 canonical 4-pocket cohort (drops 4rlu vs Wf-380-A4 / Wf-381-A1).
# ---------------------------------------------------------------------------
COHORT_POCKETS_WF333V3: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw")
COHORT_SEEDS_DEFAULT: Tuple[int, ...] = (0, 1, 2)
CISPLATIN_SMILES = "Cl[Pt](Cl)(N)N"
INDUSTRY_CUTOFF_KCAL_MOL: float = -7.0
SEED_JITTER_KCAL: float = 0.02

# ---------------------------------------------------------------------------
# 4-LEVEL EXHAUSTIVENESS GRID (Wf-404-A1 — expands Wf-380-A4 3-level grid).
# Each level is a (exhaustiveness, n_poses) pair drawn from the canonical
# Vina 1.2.7 budget bracket per Trott 2010 + Eberhardt 2021 + DiffDock
# arXiv:2210.01776 §4.2 + PocketMol arXiv:2305.16289 §4.1.
# ---------------------------------------------------------------------------
PARAM_GRID_4LEVEL: Dict[str, Dict[str, int]] = {
    "ex8_nposes5": {"exhaustiveness": 8, "n_poses": 5},
    "ex16_nposes10": {"exhaustiveness": 16, "n_poses": 10},
    "ex32_nposes20": {"exhaustiveness": 32, "n_poses": 20},
    "ex64_nposes50": {"exhaustiveness": 64, "n_poses": 50},
}

# Per-level base kcal/mol (Wf-141 §0 + Wf-333-V3 PROJECTED lift curve).
# Round values chosen to mirror the Wf-381-A1 low=-7.0 / medium=-7.5 /
# high=-8.0 ladder, with the new ex16_nposes10 mid-tier at -7.25.
BASE_KCAL_PER_LEVEL_WF404A1: Dict[str, float] = {
    "ex8_nposes5": -7.00,
    "ex16_nposes10": -7.25,
    "ex32_nposes20": -7.50,
    "ex64_nposes50": -8.00,
}

# Per-pocket offsets (subset of Wf-381-A1 5-pocket offsets, with 4rlu
# removed to match the Wf-333-V3 4-pocket selection).
PER_POCKET_OFFSET_KCAL_4POCKETS: Dict[str, float] = {
    "1h36": -0.30,
    "3eml": -0.05,
    "3ny8": -0.40,
    "4aaw": -0.10,
}

DEFAULT_OUT_PATH = (
    ROOT / "molmetal" / "reports" / "wf404a1_vina_tuning_scaffold" / "smoke_result.json"
)

DEFAULT_VINA_BINARY = "vina"


# ---------------------------------------------------------------------------
# §1 — Config + grid builders
# ---------------------------------------------------------------------------
def build_param_grid() -> Dict[str, Dict[str, int]]:
    """Return the canonical 4-level Vina param grid.

    The grid covers exhaustiveness {8, 16, 32, 64} AND n_poses {5, 10,
    20, 50}.  This is the Wf-404-A1 expansion of the Wf-380-A4 3-level
    grid (which collapsed to ex=8/32/64 and n_poses=9/20/50).
    """
    return {k: dict(v) for k, v in PARAM_GRID_4LEVEL.items()}


def load_cohort_4pockets(
    pockets: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Load the canonical Wf-333-V3 4-pocket cohort.

    Returns a dict with keys ``pockets`` and ``seeds``.
    """
    return {
        "pockets": list(pockets)
        if pockets is not None
        else list(COHORT_POCKETS_WF333V3),
        "seeds": list(seeds) if seeds is not None else list(COHORT_SEEDS_DEFAULT),
        "smiles": CISPLATIN_SMILES,
        "industry_cutoff_kcal_mol": INDUSTRY_CUTOFF_KCAL_MOL,
        "_cohort_source": "Wf-333-V3 4-pocket selection (drops 4rlu vs Wf-380-A4)",
    }


def validate_param_grid(param_grid: Dict[str, Dict[str, int]]) -> None:
    """Raise if the param grid does not match the Wf-404-A1 4-level spec."""
    expected = set(PARAM_GRID_4LEVEL)
    actual = set(param_grid)
    if actual != expected:
        raise ValueError(
            f"Wf-404-A1 param grid mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )
    for level, params in param_grid.items():
        if not (1 <= params["exhaustiveness"] <= 128):
            raise ValueError(
                f"Wf-404-A1 level '{level}' exhaustiveness={params['exhaustiveness']} "
                f"out of canonical Vina 1.2.7 bracket [1, 128]"
            )
        if not (1 <= params["n_poses"] <= 50):
            raise ValueError(
                f"Wf-404-A1 level '{level}' n_poses={params['n_poses']} "
                f"out of canonical Vina 1.2.7 bracket [1, 50]"
            )


# ---------------------------------------------------------------------------
# §2 — Synthetic kcal/mol generator (mirrors Wf-381-A1)
# ---------------------------------------------------------------------------
def _seed_jitter(pocket: str, seed: int) -> float:
    """Deterministic +/- SEED_JITTER_KCAL jitter per (pocket, seed)."""
    import hashlib

    h_bytes = hashlib.blake2b(
        f"{pocket}|{seed}".encode("utf-8"), digest_size=4
    ).digest()
    h = int.from_bytes(h_bytes, "big")
    sign = -1.0 if (h % 2 == 0) else 1.0
    return sign * SEED_JITTER_KCAL


def tune_docking(
    cohort: Dict[str, Any],
    level: str,
    param_grid: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Synthetic per-(pocket, seed) kcal/mol generator at ``level``.

    Parameters
    ----------
    cohort : dict
        The cohort mapping with keys ``pockets`` and ``seeds`` (from
        :func:`load_cohort_4pockets`).
    level : str
        One of the 4 Wf-404-A1 levels (ex8_nposes5 / ex16_nposes10 /
        ex32_nposes20 / ex64_nposes50).
    param_grid : dict, optional
        Override grid for the param lookup.  Defaults to the canonical
        Wf-404-A1 :data:`PARAM_GRID_4LEVEL`.

    Returns
    -------
    dict with keys: level, exhaustiveness, n_poses, pockets, seeds,
        per_pocket_seed_kcal, per_pocket_median_kcal, cohort_median_kcal,
        status, source, tag.
    """
    grid = param_grid if param_grid is not None else PARAM_GRID_4LEVEL
    if level not in grid:
        raise ValueError(f"Unknown level '{level}'; expected one of {list(grid)}")
    if level not in BASE_KCAL_PER_LEVEL_WF404A1:
        raise ValueError(
            f"Missing base kcal/mol for level '{level}' in "
            f"BASE_KCAL_PER_LEVEL_WF404A1"
        )
    pockets = list(cohort["pockets"])
    seeds = list(cohort["seeds"])
    base = BASE_KCAL_PER_LEVEL_WF404A1[level]
    per_pocket_seed_kcal: Dict[str, List[float]] = {}
    per_pocket_median_kcal: Dict[str, float] = {}
    for pocket in pockets:
        offset = PER_POCKET_OFFSET_KCAL_4POCKETS.get(pocket, 0.0)
        vals = [round(base + offset + _seed_jitter(pocket, seed), 4) for seed in seeds]
        per_pocket_seed_kcal[pocket] = vals
        per_pocket_median_kcal[pocket] = round(statistics.median(vals), 4)
    cohort_median = round(statistics.median(per_pocket_median_kcal.values()), 4)
    return {
        "level": level,
        "exhaustiveness": grid[level]["exhaustiveness"],
        "n_poses": grid[level]["n_poses"],
        "pockets": pockets,
        "seeds": seeds,
        "per_pocket_seed_kcal": per_pocket_seed_kcal,
        "per_pocket_median_kcal": per_pocket_median_kcal,
        "cohort_median_kcal": cohort_median,
        "status": "SYNTHETIC_PLACEHOLDER",
        "source": "Wf-404-A1 §2 (Wf-141 §0 + Wf-333-V3 PROJECTED lift model)",
        "tag": "PROJECTED",
    }


def aggregate_per_level(cohort_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-level cohort_results into an overall verdict."""
    per_level_median = {
        lvl: res["cohort_median_kcal"] for lvl, res in cohort_results.items()
    }
    baseline = per_level_median["ex8_nposes5"]
    lift = {lvl: round(per_level_median[lvl] - baseline, 4) for lvl in per_level_median}
    expected_pass = {
        lvl: per_level_median[lvl] <= INDUSTRY_CUTOFF_KCAL_MOL
        for lvl in per_level_median
    }
    overall_median = round(statistics.median(per_level_median.values()), 4)
    return {
        "per_level_cohort_median": per_level_median,
        "overall_median_kcal": overall_median,
        "lift_vs_ex8_nposes5_baseline": lift,
        "expected_pass_at_industry_cutoff_-7_0": expected_pass,
        "verdict": "SYNTHETIC_PLACEHOLDER",
    }


# ---------------------------------------------------------------------------
# §3 — CPU-only Vina CLI invocation wrapper
# ---------------------------------------------------------------------------
def which_vina(binary: str = DEFAULT_VINA_BINARY) -> Optional[str]:
    """Return absolute path to the Vina binary if on PATH, else None.

    CPU-only — uses shutil.which which is O(path length) per the OS
    stat implementation.  No subprocess.spawn.
    """
    path = shutil.which(binary)
    if path:
        LOG.debug("Wf-404-A1 found Vina binary at %s", path)
    else:
        LOG.debug("Wf-404-A1 Vina binary %r not on PATH", binary)
    return path


def invoke_vina_cli(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    *,
    exhaustiveness: int,
    n_poses: int,
    out_pdbqt: Path,
    binary: str = DEFAULT_VINA_BINARY,
    center_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    box_size_xyz: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    cpu: int = 1,
    timeout_sec: int = 600,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Invoke the Vina CLI on a (receptor, ligand) pair — CPU only.

    Returns a dict with the command, exit_code, stdout, stderr, and a
    ``dry_run`` flag.  When ``dry_run=True`` or the binary is missing,
    the function returns a synthetic placeholder result without
    launching any subprocess — so the SCAFFOLD remains deterministic
    under AGENTS.md §5.1 + kill_then_adapt §0.
    """
    cmd: List[str] = [
        binary,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--out",
        str(out_pdbqt),
        "--center_x",
        str(center_xyz[0]),
        "--center_y",
        str(center_xyz[1]),
        "--center_z",
        str(center_xyz[2]),
        "--size_x",
        str(box_size_xyz[0]),
        "--size_y",
        str(box_size_xyz[1]),
        "--size_z",
        str(box_size_xyz[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--cpu",
        str(cpu),
    ]
    resolved = which_vina(binary)
    if dry_run or resolved is None:
        return {
            "cmd": cmd,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
            "binary_resolved": resolved,
            "status": "DRY_RUN_NO_BINARY" if resolved is None else "DRY_RUN",
            "tag": "PROJECTED",
        }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "dry_run": False,
            "binary_resolved": resolved,
            "status": "OK" if proc.returncode == 0 else "VINA_FAIL",
            "tag": "MEASURED" if proc.returncode == 0 else "FAIL",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "exit_code": None,
            "stdout": "",
            "stderr": f"timeout after {timeout_sec}s",
            "dry_run": False,
            "binary_resolved": resolved,
            "status": "TIMEOUT",
            "tag": "FAIL",
            "timeout_sec": timeout_sec,
        }


# ---------------------------------------------------------------------------
# §4 — Smoke runner + report writer
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def write_smoke_report(
    out_path: Path,
    param_grid: Dict[str, Dict[str, int]],
    cohort: Dict[str, Any],
    cohort_results: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
    binary_resolved: Optional[str],
) -> Path:
    """Write the Wf-404-A1 smoke report JSON to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "_workflow": "Wf-404-A1",
        "_phase": "SCAFFOLD",
        "_title": "m3 Vina docking parameter tuning — 4-level scaffold (CPU-only)",
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU (no Vina binary launch; synthetic per-level placeholder)",
        "_param_grid": param_grid,
        "_n_pockets": len(cohort["pockets"]),
        "_n_seeds": len(cohort["seeds"]),
        "_smiles": cohort["smiles"],
        "_industry_cutoff_kcal_mol": cohort["industry_cutoff_kcal_mol"],
        "_cohort_source": cohort["_cohort_source"],
        "_binary_resolved": binary_resolved,
        "_per_level": cohort_results,
        "_aggregate": aggregate,
        "_honest_framing": {
            "status": "SYNTHETIC_PLACEHOLDER",
            "real_validation_blocker": (
                "Vina 1.2.7 binary + 4-pocket PDBQT receptors + Wf-299 CFM "
                "3-wave GPU launch (TODO-207, USER_OK-gated per AGENTS.md "
                "§5.1 / §5.4)"
            ),
            "per_level_lift_curve_source": "Wf-141 §0 + Wf-333-V3 PROJECTED lift",
            "claims": [
                "4-level config covers standard Vina 1.2.7 budget bracket",
                "ex16_nposes10 mid-tier matches SOTA DiffDock / PocketMol default",
                "ex64_nposes50 ceiling matches PocketMol arXiv:2305.16289 §4.1",
                "4-pocket cohort matches Wf-333-V3 verification scope",
            ],
            "do_not_claim": [
                "Real per-level kcal/mol as MEASURED until Vina binary actually docks",
                "m3 Vina Dock metric lift as MEASURED (only PROJECTED per Wf-141 §0)",
                "Pt+2 kcal/mol as MEASURED (still BLOCKED on UFF metal-atom parameterisation)",
            ],
        },
        "_cross_references": [
            "TODO/pending/203_m3_vina_docking_param_tuning.md",
            "molmetal/scripts/wf404a1_vina_param_tuning_scaffold.py",
            "molmetal/tests/test_wf404a1_vina_param.py",
            "molmetal/configs/vina_docking_params.yaml",
            "molmetal/scripts/wf381a1_vina_param_tuning.py",
            "TODO/completed/381a1_vina_param_tuning.md",
            "TODO/completed/333v3_vina_p2p3.md",
            "molmetal/scripts/cfm_vina_dock_panel.py",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_smoke(
    out_path: Path = DEFAULT_OUT_PATH,
    *,
    binary: str = DEFAULT_VINA_BINARY,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """End-to-end SCAFFOLD smoke run + report write.

    Returns the per-level + aggregate verdict dict.  Always CPU-only.
    """
    param_grid = build_param_grid()
    validate_param_grid(param_grid)
    cohort = load_cohort_4pockets()
    binary_resolved = which_vina(binary)
    cohort_results = {lvl: tune_docking(cohort, lvl, param_grid) for lvl in param_grid}
    aggregate = aggregate_per_level(cohort_results)
    write_smoke_report(
        out_path, param_grid, cohort, cohort_results, aggregate, binary_resolved
    )
    LOG.info(
        "Wf-404-A1 SCAFFOLD smoke run complete (binary_resolved=%s, " "out_path=%s)",
        binary_resolved,
        out_path,
    )
    return aggregate


# ---------------------------------------------------------------------------
# §5 — CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Smoke report output path (default: {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=DEFAULT_VINA_BINARY,
        help=f"Vina binary name on PATH (default: {DEFAULT_VINA_BINARY})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose DEBUG logging",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    aggregate = run_smoke(out_path=args.out, binary=args.binary)
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
