#!/usr/bin/env python3
"""Wf-141 v2 — cisplatin Vina exhaustiveness 8 → 32 lift.

================================================================
Why this script exists
================================================================
Wf-121 P3 measured real Vina 1.2.7 kcal/mol for cisplatin × 10 pockets
× 3 seeds at ``exhaustiveness=8`` and recorded baseline kcal/mol in
``molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json``.
The ex=8 values were very stable (per-pocket stddev ≈ 0.005–0.020
kcal/mol) and clustered between roughly -2.5 and -3.7 kcal/mol — far
above the -7 kcal/mol cutoff typically quoted for "good" metallodrug
poses (Trott & Olson 2010, AutoDock Vina documentation).

Wf-141 v2 lifts the same ``SMILES``/``receptor``/``docking_box``/
``seed`` set to ``exhaustiveness=32`` (Trott 2010's recommended
"production" setting for ligand-binding experiments), then runs
``exhaustiveness=64`` on two pockets (``1h36`` and ``4aaw``) as an
upper-bound reference.

Per-pocket deltas vs the ex=8 baseline are reported honestly —
Vina's stochastic search can occasionally produce a **regression**
(no-lift) at higher exhaustiveness, and we do NOT pretend otherwise
(see Honest Framing below).

Inputs
------
* ``molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json`` —
  baseline kcal_mol_per_seed at ex=8 (READ ONLY).
* ``molmetal/reports/wf121_cisplatin_real_vina/<pocket>/per_seed_metrics.json`` —
  baseline pocket geometry (receptor_pdbqt, docking_box_center).
* ``molmetal/reports/wf121_cisplatin_real_vina/<pocket>/cisplatin.pdbqt`` —
  already-generated meeko PDBQT for cisplatin (Wf-121 P1/P2 path).
* ``vina`` Python binding 1.2.7 + ``meeko`` 0.8.0 (mol-metal-venv).

Outputs
-------
* ``molmetal/reports/wf141_ex32_lift/per_pocket_summary.json`` —
  10 pockets × 3 seeds × {kcal_min, kcal_mean_top5, elapsed_s}.
* ``molmetal/reports/wf141_ex32_lift/upper_bound_ex64.json`` —
  2 pockets × 3 seeds at ex=64 (1h36, 4aaw).
* ``molmetal/reports/wf141_ex32_lift/<pocket>/per_seed_metrics.json`` —
  per-pocket per-seed Wf-141 v2 record (mirrors Wf-121 P3 schema).
* ``molmetal/reports/wf141_ex32_lift/synthesis.md`` —
  per-pocket delta vs ex=8 + honest framing.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* ``exhaustiveness=32`` is the well-known Trott 2010 lift, not a novel
  method — we use it because Wf-121 P3 baseline at ex=8 may have been
  under-sampled.
* cisplatin is a covalent binder — Vina does **not** include any
  covalent contribution in its scoring function, so the kcal/mol values
  here are the **non-covalent pose energy**, not the full reaction
  enthalpy. This is the same caveat as Wf-121 P3.
* Vina's stochastic search can occasionally fail to improve (or even
  regress) when exhaustiveness increases — we report those honestly
  as ``delta_vs_ex8 > 0`` (NO LIFT), not as silent success.
* ``ex=64`` is **not** an industry-standard protocol — it is a
  2-pocket upper-bound reference chosen because (a) both pockets
  have small ligand heavy-atom counts (595 and 401) so the run time
  stays tractable, and (b) they bracket the median ex=8 kcal/mol.

Usage
-----
::

    source env.sh
    python molmetal/scripts/bench_cisplatin_exhaustiveness_lift.py \\
        --out-dir molmetal/reports/wf141_ex32_lift
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf141_bench_exhaustiveness_lift")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "[Pt](N)(N)(Cl)Cl"
CISPLATIN_NAME = "cisplatin"

# 10 Wf-30 / Wf-121 P3 pockets — same cohort used at ex=8.
POCKETS: Tuple[str, ...] = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "4rn0",
    "1fmc",
    "1g9v",
    "2hnx",
)
SEEDS: Tuple[int, ...] = (0, 1, 2)

# Upper-bound pockets — small ligand-pocket heavy-atom count so ex=64
# stays tractable; both have intermediate ex=8 baseline values.
UPPER_BOUND_POCKETS: Tuple[str, ...] = ("1h36", "4aaw")

EXHAUSTIVENESS_PRIMARY = 32
EXHAUSTIVENESS_UPPER = 64
N_POSES = 5

WF121_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf141_ex32_lift"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SeedRun:
    """One (pocket, seed, exhaustiveness) Vina run."""

    pocket_id: str
    seed: int
    exhaustiveness: int
    kcal_min: Optional[float]  # top-pose kcal/mol
    kcal_mean_top5: Optional[float]  # mean kcal/mol across n_poses
    elapsed_s: float
    status: str  # "MEASURED" | "DOCK_FAILED"
    error: Optional[str] = None


@dataclass
class PocketSummary:
    """Per-pocket aggregate across the configured seeds."""

    pocket_id: str
    smiles: str
    exhaustiveness: int
    seeds: List[int]
    kcal_min_per_seed: List[Optional[float]]
    kcal_mean_top5_per_seed: List[Optional[float]]
    elapsed_s_per_seed: List[float]
    n_seeds_succeeded: int
    n_seeds_failed: int
    ex8_baseline_mean_kcal: Optional[float]  # from Wf-121 P3
    delta_vs_ex8_kcal: Optional[float]  # negative = lift
    status: str  # "MEASURED" | "MIXED" | "FAILED"
    engine: str = "vina"
    engine_version: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def _probe_vina_version() -> str:
    try:
        import vina  # noqa: F401

        return getattr(vina, "__version__", "unknown")
    except Exception:
        return "uninstalled"


# ---------------------------------------------------------------------------
# Baseline loading — Wf-121 P3
# ---------------------------------------------------------------------------
def _load_baseline(wf121_dir: Path) -> Dict[str, Dict[str, object]]:
    """Read the Wf-121 P3 baseline kcal_mol_per_seed + docking box per pocket.

    Returns a dict ``{pocket_id: {"kcal_mol_per_seed": [...], "docking_box_center": [..]}}``.
    """
    summary_path = wf121_dir / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Wf-121 P3 baseline missing: {summary_path} — cannot run Wf-141 v2 "
            f"without the ex=8 reference."
        )
    summary = json.loads(summary_path.read_text())
    baseline: Dict[str, Dict[str, object]] = {}
    for entry in summary["_per_pocket"]:
        pocket_id = entry["pocket_id"]
        # Box center — first try the per-seed JSON (which carries the
        # _pocket.docking_box_center); fall back to None.
        ps_path = wf121_dir / pocket_id / "per_seed_metrics.json"
        box_center: Optional[List[float]] = None
        n_heavy: Optional[int] = None
        receptor_pdbqt: Optional[str] = None
        if ps_path.exists():
            ps = json.loads(ps_path.read_text())
            pocket_block = ps.get("_pocket", {})
            box_center = pocket_block.get("docking_box_center")
            n_heavy = pocket_block.get("n_heavy_atoms")
            receptor_pdbqt = pocket_block.get("receptor_pdbqt")
        baseline[pocket_id] = {
            "kcal_mol_per_seed": entry["kcal_mol_per_seed"],
            "kcal_mol_mean_ex8": entry["kcal_mol_mean"],
            "docking_box_center": box_center,
            "n_heavy_atoms": n_heavy,
            "receptor_pdbqt": receptor_pdbqt,
        }
    return baseline


# ---------------------------------------------------------------------------
# Core: real Vina dock one (pocket, seed) at given exhaustiveness
# ---------------------------------------------------------------------------
def _dock_one(
    receptor_pdbqt: Path,
    lig_pdbqt: Path,
    center_xyz: List[float],
    seed: int,
    exhaustiveness: int,
    n_poses: int,
    box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
) -> Tuple[Optional[float], Optional[float], float]:
    """Run real Vina 1.2.7 (Python binding) at given exhaustiveness.

    Returns (kcal_min, kcal_mean_top_n, elapsed_s). kcal_min is the
    top pose's kcal/mol; kcal_mean_top_n is the mean across all
    n_poses. Returns ``None`` kcal fields if Vina raised an exception.
    """
    import vina  # type: ignore

    t0 = time.time()
    try:
        # Map uint32 seed -> nonzero int32 seed that Vina accepts.
        seed_eff = int(seed) % (2**31 - 1) or (2**31 - 1)
        v = vina.Vina(sf_name="vina", seed=seed_eff, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(
            center=[float(c) for c in center_xyz],
            box_size=list(box_size),
        )
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        energies = v.energies(n_poses=int(n_poses))
        kcal_min = float(energies[0, 0])
        kcal_mean = float(
            _mean([float(energies[i, 0]) for i in range(energies.shape[0])])
        )
        return kcal_min, kcal_mean, time.time() - t0
    except Exception as exc:  # noqa: BLE001 — honest failure path
        LOG.error(
            "Vina dock failed (seed=%d, ex=%d): %s\n%s",
            seed,
            exhaustiveness,
            exc,
            traceback.format_exc(),
        )
        return None, None, time.time() - t0


# ---------------------------------------------------------------------------
# Driver: run one pocket × all seeds at given exhaustiveness
# ---------------------------------------------------------------------------
def run_pocket_at_exhaustiveness(
    pocket_id: str,
    seeds: Tuple[int, ...],
    exhaustiveness: int,
    baseline: Dict[str, Dict[str, object]],
    wf121_dir: Path,
    out_dir: Path,
    n_poses: int,
    output_basename: str = "per_seed_metrics.json",
) -> PocketSummary:
    """Run all seeds for one pocket at given exhaustiveness; retry once on failure."""
    pocket_baseline = baseline[pocket_id]
    box_center = pocket_baseline["docking_box_center"]
    if box_center is None:
        raise RuntimeError(
            f"{pocket_id}: no docking_box_center in baseline — cannot run Wf-141 v2."
        )
    # Use the Wf-30 prep receptor (referenced from the Wf-121 P3
    # per-seed record) — same physical receptor, same SMILES, same
    # box, only exhaustiveness changes. Fall back to the wf121 dir if
    # the per-seed record lost its receptor path.
    receptor_str = pocket_baseline.get("receptor_pdbqt")
    if receptor_str:
        receptor_path = Path(receptor_str)
    else:
        receptor_path = wf121_dir / pocket_id / "receptor.pdbqt"
    if not receptor_path.exists():
        raise FileNotFoundError(
            f"{pocket_id}: receptor.pdbqt missing at {receptor_path}"
        )
    # Reuse the existing cisplatin.pdbqt (Wf-121 P1/P2 path).
    lig_pdbqt = wf121_dir / pocket_id / "cisplatin.pdbqt"
    if not lig_pdbqt.exists():
        raise FileNotFoundError(
            f"{pocket_id}: cisplatin.pdbqt missing at {lig_pdbqt} — re-run Wf-121 P3 first."
        )

    pocket_out = out_dir / pocket_id
    pocket_out.mkdir(parents=True, exist_ok=True)

    kcal_min_per_seed: List[Optional[float]] = []
    kcal_mean_per_seed: List[Optional[float]] = []
    elapsed_per_seed: List[float] = []
    per_seed_records: List[Dict[str, object]] = []
    n_failed = 0
    for seed in seeds:
        kcal_min, kcal_mean, elapsed = _dock_one(
            receptor_pdbqt=receptor_path,
            lig_pdbqt=lig_pdbqt,
            center_xyz=list(box_center),
            seed=seed,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
        )
        # One retry on failure (per workflow brief).
        if kcal_min is None:
            LOG.warning(
                "%s seed=%d ex=%d failed; retrying once",
                pocket_id,
                seed,
                exhaustiveness,
            )
            kcal_min, kcal_mean, elapsed = _dock_one(
                receptor_pdbqt=receptor_path,
                lig_pdbqt=lig_pdbqt,
                center_xyz=list(box_center),
                seed=seed,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
            )
        status = "MEASURED" if kcal_min is not None else "DOCK_FAILED"
        if kcal_min is None:
            n_failed += 1
        kcal_min_per_seed.append(kcal_min)
        kcal_mean_per_seed.append(kcal_mean)
        elapsed_per_seed.append(elapsed)
        per_seed_records.append(
            {
                "pocket_id": pocket_id,
                "seed": seed,
                "exhaustiveness": exhaustiveness,
                "kcal_min": kcal_min,
                "kcal_mean_top5": kcal_mean,
                "elapsed_s": elapsed,
                "status": status,
            }
        )
        LOG.info(
            "  %s seed=%d ex=%d -> kcal_min=%s kcal_mean_top5=%s (%.2fs)",
            pocket_id,
            seed,
            exhaustiveness,
            f"{kcal_min:.3f}" if kcal_min is not None else "FAILED",
            f"{kcal_mean:.3f}" if kcal_mean is not None else "FAILED",
            elapsed,
        )

    # Compute delta vs ex=8 baseline.
    ex8_mean = pocket_baseline["kcal_mol_mean_ex8"]
    succeeded_kcals = [k for k in kcal_min_per_seed if k is not None]
    new_mean = _mean(succeeded_kcals) if succeeded_kcals else None
    delta: Optional[float]
    if new_mean is not None and ex8_mean is not None:
        delta = float(new_mean - ex8_mean)
    else:
        delta = None

    n_succeeded = len(succeeded_kcals)
    if n_failed == 0:
        agg_status = "MEASURED"
    elif n_succeeded == 0:
        agg_status = "FAILED"
    else:
        agg_status = "MIXED"

    summary = PocketSummary(
        pocket_id=pocket_id,
        smiles=CISPLATIN_SMILES,
        exhaustiveness=int(exhaustiveness),
        seeds=list(seeds),
        kcal_min_per_seed=kcal_min_per_seed,
        kcal_mean_top5_per_seed=kcal_mean_per_seed,
        elapsed_s_per_seed=elapsed_per_seed,
        n_seeds_succeeded=n_succeeded,
        n_seeds_failed=n_failed,
        ex8_baseline_mean_kcal=float(ex8_mean) if ex8_mean is not None else None,
        delta_vs_ex8_kcal=delta,
        status=agg_status,
        engine="vina",
        engine_version=_probe_vina_version(),
    )

    # Write per-pocket per-seed record (mirror Wf-121 P3 schema).
    payload = {
        "_workflow": "Wf-141",
        "_phase": "v2",
        "_title": (
            f"Real Vina 1.2.7 dock cisplatin at ex={exhaustiveness} on {pocket_id} "
            f"(lift over Wf-121 P3 ex=8 baseline)"
        ),
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU-only (Vina Python binding) + GPU-0 Blackwell",
        "_engine": "vina",
        "_engine_version": _probe_vina_version(),
        "_smiles": CISPLATIN_SMILES,
        "_n_poses": int(n_poses),
        "_exhaustiveness": int(exhaustiveness),
        "_pocket": {
            "pocket_id": pocket_id,
            "receptor_pdbqt": str(receptor_path),
            "n_heavy_atoms": pocket_baseline.get("n_heavy_atoms"),
            "docking_box_center": list(box_center),
        },
        "_wf121_baseline_ex8": {
            "kcal_mol_per_seed": pocket_baseline["kcal_mol_per_seed"],
            "kcal_mol_mean": pocket_baseline["kcal_mol_mean_ex8"],
        },
        "_per_seed": per_seed_records,
        "_aggregate": asdict(summary),
    }
    (pocket_out / output_basename).write_text(json.dumps(payload, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Driver: full sweep at ex=32 (10 pockets) + ex=64 (2 pockets)
# ---------------------------------------------------------------------------
def run_full_lift(
    out_dir: Path,
    wf121_dir: Path,
    exhaustiveness_primary: int,
    exhaustiveness_upper: int,
    n_poses: int,
) -> Dict[str, object]:
    """Run the full Wf-141 v2 sweep and write summary JSON + synthesis.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline(wf121_dir)

    primary_summaries: Dict[str, PocketSummary] = {}
    for pocket_id in POCKETS:
        if pocket_id not in baseline:
            LOG.error("%s: missing from Wf-121 baseline — skipping", pocket_id)
            continue
        LOG.info(
            "Pocket %s — docking cisplatin at ex=%d × %d seeds...",
            pocket_id,
            exhaustiveness_primary,
            len(SEEDS),
        )
        primary_summaries[pocket_id] = run_pocket_at_exhaustiveness(
            pocket_id=pocket_id,
            seeds=SEEDS,
            exhaustiveness=exhaustiveness_primary,
            baseline=baseline,
            wf121_dir=wf121_dir,
            out_dir=out_dir,
            n_poses=n_poses,
        )

    upper_summaries: Dict[str, PocketSummary] = {}
    for pocket_id in UPPER_BOUND_POCKETS:
        if pocket_id not in baseline:
            LOG.error(
                "%s: missing from Wf-121 baseline — cannot run ex=%d upper-bound",
                pocket_id,
                exhaustiveness_upper,
            )
            continue
        LOG.info(
            "Upper-bound pocket %s — docking cisplatin at ex=%d × %d seeds...",
            pocket_id,
            exhaustiveness_upper,
            len(SEEDS),
        )
        upper_summaries[pocket_id] = run_pocket_at_exhaustiveness(
            pocket_id=pocket_id,
            seeds=SEEDS,
            exhaustiveness=exhaustiveness_upper,
            baseline=baseline,
            wf121_dir=wf121_dir,
            out_dir=out_dir,
            n_poses=n_poses,
            output_basename=f"per_seed_metrics_ex{exhaustiveness_upper}.json",
        )

    # ---- per_pocket_summary.json (10 pockets × ex=32) ----
    primary_payload = {
        ps.pocket_id: {
            "pocket_id": ps.pocket_id,
            "smiles": ps.smiles,
            "exhaustiveness": ps.exhaustiveness,
            "seeds": ps.seeds,
            "kcal_min_per_seed": ps.kcal_min_per_seed,
            "kcal_mean_top5_per_seed": ps.kcal_mean_top5_per_seed,
            "kcal_min_mean": _mean([k for k in ps.kcal_min_per_seed if k is not None])
            if any(k is not None for k in ps.kcal_min_per_seed)
            else None,
            "kcal_mean_top5_mean": _mean(
                [k for k in ps.kcal_mean_top5_per_seed if k is not None]
            )
            if any(k is not None for k in ps.kcal_mean_top5_per_seed)
            else None,
            "ex8_baseline_mean_kcal": ps.ex8_baseline_mean_kcal,
            "delta_vs_ex8_kcal": ps.delta_vs_ex8_kcal,
            "elapsed_s_total": float(_mean(ps.elapsed_s_per_seed)),
            "n_seeds_succeeded": ps.n_seeds_succeeded,
            "n_seeds_failed": ps.n_seeds_failed,
            "status": ps.status,
        }
        for ps in primary_summaries.values()
    }
    (out_dir / "per_pocket_summary.json").write_text(
        json.dumps(primary_payload, indent=2)
    )

    # ---- upper_bound_ex64.json (2 pockets × ex=64) ----
    upper_payload = {
        ps.pocket_id: {
            "pocket_id": ps.pocket_id,
            "smiles": ps.smiles,
            "exhaustiveness": ps.exhaustiveness,
            "seeds": ps.seeds,
            "kcal_min_per_seed": ps.kcal_min_per_seed,
            "kcal_mean_top5_per_seed": ps.kcal_mean_top5_per_seed,
            "kcal_min_mean": _mean([k for k in ps.kcal_min_per_seed if k is not None])
            if any(k is not None for k in ps.kcal_min_per_seed)
            else None,
            "kcal_mean_top5_mean": _mean(
                [k for k in ps.kcal_mean_top5_per_seed if k is not None]
            )
            if any(k is not None for k in ps.kcal_mean_top5_per_seed)
            else None,
            "ex8_baseline_mean_kcal": ps.ex8_baseline_mean_kcal,
            "delta_vs_ex8_kcal": ps.delta_vs_ex8_kcal,
            "n_seeds_succeeded": ps.n_seeds_succeeded,
            "n_seeds_failed": ps.n_seeds_failed,
            "status": ps.status,
        }
        for ps in upper_summaries.values()
    }
    (out_dir / "upper_bound_ex64.json").write_text(json.dumps(upper_payload, indent=2))

    # ---- synthesis.md ----
    md_lines: List[str] = []
    md_lines.append("# Wf-141 v2 — cisplatin Vina exhaustiveness 8 → 32 lift")
    md_lines.append("")
    md_lines.append(f"_Generated: {_utc_now_iso()}_  ")
    md_lines.append(f"_Hardware: CPU-only (Vina Python binding) + GPU-0 Blackwell_  ")
    md_lines.append(f"_Ligand SMILES_: `{CISPLATIN_SMILES}` (= cisplatin).  ")
    md_lines.append(f"_Engine_: Vina {_probe_vina_version()} (Python binding).  ")
    md_lines.append(
        f"_Pockets_: {list(POCKETS)}; _upper-bound pockets_: "
        f"{list(UPPER_BOUND_POCKETS)}."
    )
    md_lines.append(
        f"_Exhaustiveness_: primary={exhaustiveness_primary}, upper-bound="
        f"{exhaustiveness_upper}; _n_poses_: {n_poses}; _seeds_: {list(SEEDS)}."
    )
    md_lines.append("")
    md_lines.append("## Per-pocket kcal/mol (ex=32) vs Wf-121 P3 ex=8 baseline")
    md_lines.append("")
    md_lines.append(
        "| Pocket | ex=8 mean (baseline) | ex=32 mean (this run) | Δ (ex32−ex8) | "
        "n_seeds_ok | status |"
    )
    md_lines.append("|---|---|---|---|---|---|")
    for ps in primary_summaries.values():
        ex8 = ps.ex8_baseline_mean_kcal
        delta = ps.delta_vs_ex8_kcal
        new_mean = (
            _mean([k for k in ps.kcal_min_per_seed if k is not None])
            if any(k is not None for k in ps.kcal_min_per_seed)
            else None
        )
        md_lines.append(
            f"| {ps.pocket_id} | "
            f"{ex8:.3f} | "
            f"{(new_mean if new_mean is not None else float('nan')):.3f} | "
            f"{(delta if delta is not None else float('nan')):+.3f} | "
            f"{ps.n_seeds_succeeded}/{len(SEEDS)} | "
            f"{ps.status} |"
        )
    md_lines.append("")
    md_lines.append("## Upper-bound (ex=64) on 1h36 + 4aaw")
    md_lines.append("")
    md_lines.append(
        "| Pocket | ex=8 baseline | ex=32 mean | ex=64 mean | "
        "Δ(ex64−ex8) | Δ(ex64−ex32) |"
    )
    md_lines.append("|---|---|---|---|---|---|")
    for pocket_id in UPPER_BOUND_POCKETS:
        ps32 = primary_summaries.get(pocket_id)
        ps64 = upper_summaries.get(pocket_id)
        if ps32 is None or ps64 is None:
            continue
        ex8 = ps32.ex8_baseline_mean_kcal
        mean32 = _mean([k for k in ps32.kcal_min_per_seed if k is not None])
        mean64 = _mean([k for k in ps64.kcal_min_per_seed if k is not None])
        delta_64_8 = mean64 - ex8
        delta_64_32 = mean64 - mean32
        md_lines.append(
            f"| {pocket_id} | {ex8:.3f} | {mean32:.3f} | {mean64:.3f} | "
            f"{delta_64_8:+.3f} | {delta_64_32:+.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Honest framing (party-twenty-spirit §3.5)")
    md_lines.append("")
    md_lines.append(
        "* All kcal/mol values are **real Vina 1.2.7 output** — same "
        "SMILES, same receptor, same docking box, same seeds as "
        "Wf-121 P3; only `exhaustiveness` differs."
    )
    md_lines.append(
        "* `ex=32` is the well-known Trott 2010 lift (AutoDock Vina "
        "documentation: recommended production exhaustiveness), not a "
        "novel method. We use it because Wf-121 P3's ex=8 baseline may "
        "have been under-sampled."
    )
    md_lines.append(
        "* cisplatin is a **covalent binder** — Vina's scoring function "
        "does not include any covalent contribution. The kcal/mol here "
        "is the non-covalent pose energy, not the full reaction "
        "enthalpy. This caveat is identical to Wf-121 P3."
    )
    md_lines.append(
        "* Where `delta_vs_ex8 > 0` (NO LIFT, regression), we report "
        "the value honestly — Vina's stochastic search can occasionally "
        "fail to improve (or even regress) when exhaustiveness "
        "increases. We do not silently suppress negative-lift pockets."
    )
    md_lines.append(
        "* `ex=64` is **not** an industry-standard protocol — it is a "
        "2-pocket upper-bound reference chosen because (a) both "
        "pockets have small ligand heavy-atom counts (595 and 401) so "
        "the run time stays tractable, and (b) they bracket the "
        "median ex=8 kcal/mol."
    )
    md_lines.append(
        "* We do NOT claim ex=32 will hit the -7 kcal/mol cutoff on "
        "all 10 pockets, do NOT claim ex=64 is industry standard, and "
        "do NOT claim this is m3 paper §4 main lead."
    )
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append("")
    md_lines.append("* `per_pocket_summary.json` — 10 pockets × ex=32.")
    md_lines.append("* `upper_bound_ex64.json` — 2 pockets × ex=64.")
    md_lines.append("* `<pocket>/per_seed_metrics.json` — per-seed record per pocket.")
    (out_dir / "synthesis.md").write_text("\n".join(md_lines) + "\n")

    return {
        "_workflow": "Wf-141",
        "_phase": "v2",
        "_title": (
            f"Cisplatin Vina exhaustiveness lift: ex=8 → ex=32 on 10 pockets, "
            f"ex=64 upper-bound on 2 pockets"
        ),
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU-only (Vina Python binding) + GPU-0 Blackwell",
        "_engine": "vina",
        "_engine_version": _probe_vina_version(),
        "_smiles": CISPLATIN_SMILES,
        "_exhaustiveness_primary": int(exhaustiveness_primary),
        "_exhaustiveness_upper": int(exhaustiveness_upper),
        "_n_poses": int(n_poses),
        "_primary_pockets": list(primary_summaries.keys()),
        "_upper_pockets": list(upper_summaries.keys()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cisplatin_exhaustiveness_lift",
        description=(
            "Wf-141 v2 — lift Wf-121 P3 cisplatin × 10-pocket sweep from "
            "ex=8 to ex=32 (Trott 2010 production setting), plus an "
            "ex=64 upper-bound on 1h36 + 4aaw."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root (default: molmetal/reports/wf141_ex32_lift).",
    )
    p.add_argument(
        "--wf121-dir",
        type=Path,
        default=WF121_DIR,
        help="Wf-121 P3 baseline dir (default: molmetal/reports/wf121_cisplatin_real_vina).",
    )
    p.add_argument(
        "--exhaustiveness-primary",
        type=int,
        default=EXHAUSTIVENESS_PRIMARY,
        help="Primary exhaustiveness (default 32).",
    )
    p.add_argument(
        "--exhaustiveness-upper",
        type=int,
        default=EXHAUSTIVENESS_UPPER,
        help="Upper-bound exhaustiveness (default 64).",
    )
    p.add_argument(
        "--n-poses",
        type=int,
        default=N_POSES,
        help="Vina --num_modes (default 5).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Pre-flight
    if not args.wf121_dir.exists():
        LOG.error("Wf-121 P3 baseline dir not found: %s", args.wf121_dir)
        return 2
    if _probe_vina_version() == "uninstalled":
        LOG.error("vina Python binding not importable; aborting")
        return 1
    summary = run_full_lift(
        out_dir=args.out_dir,
        wf121_dir=args.wf121_dir,
        exhaustiveness_primary=int(args.exhaustiveness_primary),
        exhaustiveness_upper=int(args.exhaustiveness_upper),
        n_poses=int(args.n_poses),
    )
    LOG.info(
        "Lift sweep complete: %d primary pockets × ex=%d, %d upper-bound pockets × ex=%d. "
        "Outputs at %s.",
        len(summary["_primary_pockets"]),
        summary["_exhaustiveness_primary"],
        len(summary["_upper_pockets"]),
        summary["_exhaustiveness_upper"],
        args.out_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
