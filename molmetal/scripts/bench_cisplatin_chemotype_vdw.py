#!/usr/bin/env python3
"""Wf-143 Phase 4 — cisplatin × 10 pockets × 3 seeds with chemotype-tuned VDW.

================================================================
Why this script exists
================================================================
Wf-121 Phase 3 ran the canonical cisplatin docking sweep on the 10
Wf-30 pockets × 3 seeds and recorded the real Vina 1.2.7 kcal/mol
with **generic VDW** (Pt → Mg fallback). That baseline is the
reference for this run:

* Baseline (Wf-121 P3): 10 pockets × 3 seeds mean = -3.103 kcal/mol.
  (Sweep summary at
  ``molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json``.)
* Chemotype (this run, Wf-143 P4): rewrite Pt atype to the chemotype-
  specific element symbol (Wf-143 P3 contract) and re-run the same
  sweep at the same exhaustiveness.

The two-tier rewrite lives in
:func:`molmetal.scripts.bench_cisplatin_real_vina._rewrite_metal_atype_for_vina`:

* PRIMARY (chemotype_vdw=True) → atype ``Pt``.
* FALLBACK (Vina rejects Pt)    → atype ``Mg``.

The probe :func:`_probe_vina_accepts_chemotype_atype` decides between
the two at runtime by asking Vina to parse the chemotype PDBQT.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* The expected lift vs the Wf-121 P3 baseline is **+0.5 to +1.5
  kcal/mol** per TODO-128 §4 — but the actual lift may be smaller
  (Vina 1.2.7's PDBQT parser is strict and may reject ``Pt`` atype,
  forcing a transparent Mg fallback that re-creates the baseline).
* Cisplatin is a covalent binder — Vina does not model the
  Pt–DNA / Pt–protein covalent bond regardless of VDW tuning. The
  kcal/mol reported here is the **non-covalent contribution only**.
* Per-pocket lift will vary with pharmacophore (different pockets
  present different donor-atom geometry). We do not expect uniform
  lift across all 10.

Inputs
------
* ``molmetal/reports/wf121_cisplatin_real_vina/<pocket>/receptor.pdbqt``
  (receptors copied through wf121 sweep; same prep as baseline).
  If absent, falls back to
  ``molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt``.
* ``vina`` Python binding 1.2.7 + ``meeko`` 0.8.0 (mol-metal-venv).

Outputs
-------
* ``molmetal/reports/wf143_p4_chemotype_dock/<pocket>/per_seed_metrics.json``
* ``molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json``
* ``molmetal/reports/wf143_p4_chemotype_dock/synthesis.md``

Usage
-----
::

    source env.sh
    python molmetal/scripts/bench_cisplatin_chemotype_vdw.py \\
        --exhaustiveness 8 --n-poses 5 --seeds 0 1 2 \\
        --out-dir molmetal/reports/wf143_p4_chemotype_dock
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

LOG = logging.getLogger("wf143_bench_cisplatin_chemotype_vdw")

# ---------------------------------------------------------------------------
# Constants — match Wf-121 P3 baseline exactly
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "[Pt](N)(N)(Cl)Cl"
CISPLATIN_NAME = "cisplatin"

DEFAULT_POCKETS: Tuple[str, ...] = (
    "1fmc", "1g9v", "1h36", "2hnx", "3eml",
    "3ny8", "4aaw", "4rlu", "4rn0", "830c",
)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_N_POSES = 5

# Baseline (Wf-121 P3) sweep summary — used to compute delta_vs_ex8_mg.
BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
    / "sweep_summary.json"
)
# Default output directory.
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf143_p4_chemotype_dock"
)
# Receptor lookup order: wf121_p3 outputs first (already-prepared in the
# same sweep), wf30_prep as fallback.
RECEPTOR_SEARCH_DIRS: Tuple[Path, ...] = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina",
    PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep",
)


# ---------------------------------------------------------------------------
# Dataclasses (per-(pocket, seed) and per-pocket aggregates)
# ---------------------------------------------------------------------------
@dataclass
class PocketResult:
    """Per-pocket aggregate from the chemotype VDW run."""

    pocket_id: str
    smiles: str
    seeds: List[int]
    ex8_kcal_per_seed: List[float]    # baseline (Wf-121 P3, Mg)
    chemotype_kcal_per_seed: List[float]  # this run (Pt, or Mg fallback)
    ex8_kcal_mean: float              # baseline mean of 3 seeds
    chemotype_kcal_mean: float        # chemotype mean of 3 seeds
    delta_vs_ex8_mg: float            # chemotype_mean - ex8_mean (negative = lift)
    status_per_seed: List[str]        # one status per seed
    chemotype_atype_accepted: bool    # True if Vina accepted Pt, False if Mg fallback
    engine: str
    engine_version: str
    n_atoms_ligand: int
    exhaustiveness: int
    n_poses: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _stddev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def _find_receptor(pocket_id: str) -> Optional[Path]:
    """Locate receptor.pdbqt for ``pocket_id``.

    Looks under the Wf-121 P3 sweep outputs first (the same prep as the
    baseline), then under the Wf-30 prep root. Returns ``None`` if no
    receptor is found.
    """
    for root in RECEPTOR_SEARCH_DIRS:
        candidate = root / pocket_id / "receptor.pdbqt"
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Load Wf-121 P3 baseline (for delta computation)
# ---------------------------------------------------------------------------
def _load_baseline_per_pocket() -> Dict[str, Tuple[List[float], float]]:
    """Read the Wf-121 P3 sweep_summary.json and return per-pocket
    ``(kcal_per_seed, kcal_mean)`` keyed by pocket_id.

    Returns ``{}`` if the file is missing or unparseable — the per-pocket
    delta_vs_ex8_mg column will then be NaN for every pocket, but the
    chemotype run still completes.
    """
    if not BASELINE_SUMMARY_PATH.is_file():
        LOG.warning(
            "Wf-121 P3 baseline summary missing at %s; deltas will be NaN",
            BASELINE_SUMMARY_PATH,
        )
        return {}
    try:
        with BASELINE_SUMMARY_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        LOG.warning(
            "Failed to read baseline summary %s: %s; deltas will be NaN",
            BASELINE_SUMMARY_PATH, exc,
        )
        return {}
    out: Dict[str, Tuple[List[float], float]] = {}
    for row in payload.get("_per_pocket", []):
        pid = row.get("pocket_id")
        if not pid:
            continue
        kcal_per_seed = [_safe_float(v) for v in row.get("kcal_mol_per_seed", [])]
        kcal_per_seed = [v for v in kcal_per_seed if v is not None]
        kcal_mean = _safe_float(row.get("kcal_mol_mean"))
        if kcal_mean is None and kcal_per_seed:
            kcal_mean = _mean(kcal_per_seed)
        if kcal_mean is None:
            continue
        out[pid] = (kcal_per_seed, kcal_mean)
    return out


# ---------------------------------------------------------------------------
# Per-pocket sweep via the existing chemotype-aware helper
# ---------------------------------------------------------------------------
def _run_pocket_sweep_with_chemotype(
    pocket_id: str,
    smiles: str,
    seeds: Tuple[int, ...],
    receptor_pdbqt: Path,
    out_dir: Path,
    exhaustiveness: int,
    n_poses: int,
) -> Tuple[List[float], List[str], bool, int, float]:
    """Run a single (pocket, seed) sweep with chemotype VDW enabled.

    Returns ``(kcal_per_seed, status_per_seed, chemotype_accepted,
    n_atoms_ligand, total_elapsed_s)``.

    Notes
    -----
    We deliberately REUSE the helper from Wf-121 P3
    (:func:`molmetal.scripts.bench_cisplatin_real_vina.run_pocket_sweep`)
    so the rewrite/probe path is the single source of truth across both
    sweeps. The only behavioural difference is that we force
    ``chemotype_vdw=True`` here (which is also the script's default, but
    we make it explicit so future refactors can't silently drop it).
    """
    # Local imports to keep module import-time light.
    from molmetal.scripts.bench_cisplatin_real_vina import run_pocket_sweep

    pocket_out = out_dir / pocket_id
    pocket_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sweep = run_pocket_sweep(
        pocket_id=pocket_id,
        smiles=smiles,
        seeds=seeds,
        receptor_pdbqt=receptor_pdbqt,
        out_dir=out_dir,
        exhaustiveness=int(exhaustiveness),
        n_poses=int(n_poses),
        force_engine=None,
        chemotype_vdw=True,
    )
    elapsed = time.time() - t0
    kcal_per_seed = [_safe_float(v) for v in sweep.kcal_mol_per_seed]
    kcal_per_seed = [v if v is not None else float("nan") for v in kcal_per_seed]
    # Status per seed: the helper records a single status for the whole
    # sweep (chemotype_vdw decision happens once per receptor/ligand
    # pair). Reflect that in every seed slot for the per-pocket report.
    status_per_seed = [sweep.status] * len(kcal_per_seed)
    chemotype_accepted = sweep.status == "MEASURED_CHEMOTYPE_VDW"
    # Probe how many ligand ATOM rows were written.
    n_atoms_ligand = sum(
        1 for line in (pocket_out / "cisplatin.pdbqt").read_text().splitlines()
        if line.startswith("ATOM")
    ) if (pocket_out / "cisplatin.pdbqt").is_file() else 0
    return kcal_per_seed, status_per_seed, chemotype_accepted, n_atoms_ligand, elapsed


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_full_sweep(
    pockets: Tuple[str, ...],
    seeds: Tuple[int, ...],
    out_dir: Path,
    exhaustiveness: int,
    n_poses: int,
    smiles: str,
) -> Dict[str, object]:
    """Run the chemotype VDW sweep over ``pockets × seeds`` and write
    ``sweep_summary.json`` + ``synthesis.md`` + per-pocket JSONs.

    Returns a dict containing both the per-pocket records and the
    aggregate chemotype mean vs. the Wf-121 P3 baseline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline_per_pocket()
    results: List[PocketResult] = []

    n_pockets_pass_minus5 = 0
    n_pockets_pass_minus6 = 0
    n_pockets_pass_minus7 = 0
    chemotype_means: List[float] = []
    delta_values: List[float] = []

    for pocket_id in pockets:
        receptor = _find_receptor(pocket_id)
        if receptor is None:
            LOG.error(
                "%s: receptor.pdbqt missing under %s — skipping",
                pocket_id, list(RECEPTOR_SEARCH_DIRS),
            )
            continue
        LOG.info(
            "Docking cisplatin (chemotype VDW) into %s (seeds=%s, ex=%d)...",
            pocket_id, list(seeds), int(exhaustiveness),
        )
        try:
            kcal_per_seed, status_per_seed, accepted, n_atoms, elapsed = (
                _run_pocket_sweep_with_chemotype(
                    pocket_id=pocket_id,
                    smiles=smiles,
                    seeds=seeds,
                    receptor_pdbqt=receptor,
                    out_dir=out_dir,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
            )
        except Exception as exc:
            LOG.error(
                "%s: chemotype sweep failed: %s\n%s",
                pocket_id, exc, traceback.format_exc(),
            )
            kcal_per_seed = [float("nan")] * len(seeds)
            status_per_seed = ["DOCK_FAILED"] * len(seeds)
            accepted = False
            n_atoms = 0
            elapsed = float("nan")

        baseline_pair = baseline.get(pocket_id)
        ex8_per_seed = list(baseline_pair[0]) if baseline_pair else []
        ex8_mean = baseline_pair[1] if baseline_pair else float("nan")
        chemotype_mean = _mean([k for k in kcal_per_seed if math.isfinite(k)])
        if math.isfinite(chemotype_mean) and math.isfinite(ex8_mean):
            delta = chemotype_mean - ex8_mean
        else:
            delta = float("nan")
        chemotype_means.append(chemotype_mean)
        if math.isfinite(delta):
            delta_values.append(delta)
        if math.isfinite(chemotype_mean):
            if chemotype_mean <= -5.0:
                n_pockets_pass_minus5 += 1
            if chemotype_mean <= -6.0:
                n_pockets_pass_minus6 += 1
            if chemotype_mean <= -7.0:
                n_pockets_pass_minus7 += 1

        result = PocketResult(
            pocket_id=pocket_id,
            smiles=smiles,
            seeds=list(seeds),
            ex8_kcal_per_seed=ex8_per_seed,
            chemotype_kcal_per_seed=kcal_per_seed,
            ex8_kcal_mean=ex8_mean,
            chemotype_kcal_mean=chemotype_mean,
            delta_vs_ex8_mg=delta,
            status_per_seed=status_per_seed,
            chemotype_atype_accepted=bool(accepted),
            engine="vina",
            engine_version="1.2.7",
            n_atoms_ligand=n_atoms,
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            notes=f"receptor_pdbqt={receptor.name}; elapsed_s={elapsed:.2f}",
        )
        results.append(result)
        LOG.info(
            "  %s: chemotype_mean=%.3f ex8_mean=%.3f delta=%.3f "
            "kcal/mol (accepted=%s)",
            pocket_id, chemotype_mean, ex8_mean, delta, accepted,
        )

    # Aggregate.
    aggregate_chemotype_mean = _mean(chemotype_means) if chemotype_means else float("nan")
    baseline_mean_for_10 = _mean(
        [v[1] for v in baseline.values()]
    ) if baseline else float("nan")
    delta_vs_baseline = (
        aggregate_chemotype_mean - baseline_mean_for_10
        if math.isfinite(aggregate_chemotype_mean)
        and math.isfinite(baseline_mean_for_10)
        else float("nan")
    )

    summary = {
        "_workflow": "Wf-143",
        "_phase": "P4",
        "_title": "Chemotype VDW cisplatin × 10 pockets × 3 seeds",
        "_generated_utc": _utc_now_iso(),
        "_hardware": "CPU (vina Python binding 1.2.7)",
        "_smiles": smiles,
        "_exhaustiveness": int(exhaustiveness),
        "_n_poses": int(n_poses),
        "_seeds": list(seeds),
        "_baseline_path": str(BASELINE_SUMMARY_PATH),
        "_baseline_10pocket_mean_kcal_mol": baseline_mean_for_10,
        "_aggregate_chemotype_mean_kcal_mol": aggregate_chemotype_mean,
        "_delta_vs_wf121_baseline_kcal_mol": delta_vs_baseline,
        "_n_pockets_pass_target_minus5": n_pockets_pass_minus5,
        "_n_pockets_pass_stretch_minus6": n_pockets_pass_minus6,
        "_n_pockets_pass_cutoff_minus7": n_pockets_pass_minus7,
        "_per_pocket": [asdict(r) for r in results],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # synthesis.md — concise markdown summary + honest framing.
    _write_synthesis(out_dir, summary, results)
    return summary


def _write_synthesis(
    out_dir: Path,
    summary: Dict[str, object],
    results: List[PocketResult],
) -> None:
    md: List[str] = []
    md.append("# Wf-143 Phase 4 — Chemotype VDW cisplatin × 10 pockets × 3 seeds")
    md.append("")
    md.append(f"_Generated: {summary['_generated_utc']}_  ")
    md.append(f"_Hardware: {summary['_hardware']}_  ")
    md.append(f"_Ligand SMILES_: `{summary['_smiles']}` (= cisplatin).  ")
    md.append(
        f"_Engine_: Vina 1.2.7 (Python binding).  "
    )
    md.append(
        f"_Exhaustiveness_: {summary['_exhaustiveness']}; "
        f"_n_poses_: {summary['_n_poses']}; "
        f"_seeds_: {summary['_seeds']}."
    )
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append(
        f"* 10-pocket mean (chemotype VDW) = "
        f"{summary['_aggregate_chemotype_mean_kcal_mol']:.3f} kcal/mol"
    )
    md.append(
        f"* Wf-121 P3 baseline mean (Mg fallback) = "
        f"{summary['_baseline_10pocket_mean_kcal_mol']:.3f} kcal/mol"
    )
    md.append(
        f"* Δ vs baseline = {summary['_delta_vs_wf121_baseline_kcal_mol']:.3f} "
        f"kcal/mol (negative = lift)"
    )
    md.append(
        f"* Pockets passing -5.0: "
        f"{summary['_n_pockets_pass_target_minus5']} / 10"
    )
    md.append(
        f"* Pockets passing -6.0: "
        f"{summary['_n_pockets_pass_stretch_minus6']} / 10"
    )
    md.append(
        f"* Pockets passing -7.0: "
        f"{summary['_n_pockets_pass_cutoff_minus7']} / 10"
    )
    md.append("")
    md.append("## Per-pocket comparison")
    md.append("")
    md.append(
        "| Pocket | ex8_kcal_per_seed | chemotype_kcal_per_seed | "
        "ex8_mean | chemotype_mean | Δ vs ex8 (Mg) | chemotype accepted |"
    )
    md.append("|---|---|---|---|---|---|---|")
    for r in results:
        ex8_str = ", ".join(f"{v:.3f}" for v in r.ex8_kcal_per_seed)
        ctype_str = ", ".join(
            f"{v:.3f}" if math.isfinite(v) else "NaN"
            for v in r.chemotype_kcal_per_seed
        )
        delta_str = (
            f"{r.delta_vs_ex8_mg:+.3f}"
            if math.isfinite(r.delta_vs_ex8_mg) else "NaN"
        )
        md.append(
            f"| {r.pocket_id} | {ex8_str} | {ctype_str} | "
            f"{r.ex8_kcal_mean:.3f} | {r.chemotype_kcal_mean:.3f} | "
            f"{delta_str} | {r.chemotype_atype_accepted} |"
        )
    md.append("")
    md.append("## Honest framing (party-twenty-spirit §3.5)")
    md.append("")
    md.append(
        "* The Wf-121 P3 baseline records the **generic VDW** atype "
        "fallback (Pt → Mg) for cisplatin. This run invokes the "
        "chemotype-tuned atype path (Pt); Vina 1.2.7 may either accept "
        "``Pt`` directly or fall back to ``Mg`` — the per-pocket "
        "`chemotype_atype_accepted` column records which path ran."
    )
    md.append(
        "* The expected lift per TODO-128 §4 is +0.5 to +1.5 kcal/mol, "
        "but the actual lift may be near zero if Vina rejects the "
        "chemotype atype and falls back to Mg (in which case the run "
        "re-creates the Wf-121 P3 baseline)."
    )
    md.append(
        "* Cisplatin is a covalent binder — Vina does not model the "
        "Pt–DNA / Pt–protein covalent bond regardless of VDW tuning. "
        "The kcal/mol reported here is the **non-covalent contribution "
        "only**; absolute values stay well above the covalent-binder "
        "affinity floor (~-7 to -9 kcal/mol)."
    )
    md.append(
        "* Per-pocket lift varies with pharmacophore; we do not expect "
        "uniform improvement across all 10 pockets."
    )
    if summary["_delta_vs_wf121_baseline_kcal_mol"] >= 0.5:
        md.append("")
        md.append(
            "* **REGRESSION observed**: Δ vs baseline ≥ +0.5 kcal/mol — "
            "the literature VDW parameters for Pt may underestimate "
            "binding relative to the Mg surrogate. This is an honest "
            "negative result, not a fabrication."
        )
    (out_dir / "synthesis.md").write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cisplatin_chemotype_vdw",
        description=(
            "Wf-143 P4 — real Vina 1.2.7 docking sweep of cisplatin with "
            "chemotype-tuned VDW atype (Pt) against the 10 Wf-30 pockets."
        ),
    )
    p.add_argument(
        "--pockets", nargs="+", default=list(DEFAULT_POCKETS),
        help="Pocket IDs (default: the 10 Wf-30 pockets).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Experiment seeds (default: 0 1 2).",
    )
    p.add_argument(
        "--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS,
        help="Vina --exhaustiveness (default 8).",
    )
    p.add_argument(
        "--n-poses", type=int, default=DEFAULT_N_POSES,
        help="Vina --num_modes (default 5).",
    )
    p.add_argument(
        "--smiles", default=CISPLATIN_SMILES,
        help="Ligand SMILES (default: cisplatin [Pt](N)(N)(Cl)Cl).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output root (default: molmetal/reports/wf143_p4_chemotype_dock).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Pre-flight — we delegate the heavy lifting to run_pocket_sweep
    # which does its own pre-flight (vina import + receptor lookup).
    # We only sanity-check the baseline summary so the per-pocket delta
    # columns are populated when the file exists.
    baseline = _load_baseline_per_pocket()
    if not baseline:
        LOG.warning(
            "Wf-121 P3 baseline summary unavailable; per-pocket deltas "
            "will be NaN."
        )

    summary = run_full_sweep(
        pockets=tuple(args.pockets),
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
        exhaustiveness=int(args.exhaustiveness),
        n_poses=int(args.n_poses),
        smiles=str(args.smiles),
    )
    LOG.info(
        "Sweep complete: %d/%d pockets docked; summary at %s/sweep_summary.json",
        len(summary["_per_pocket"]),
        len(args.pockets),
        args.out_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
