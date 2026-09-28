#!/usr/bin/env python3
"""Wf-142 — per-pocket high_affinity lift via Vina exhaustiveness 4 → 32.

==========================================================================
Why this script exists
==========================================================================
Wf-126 P3 measured the per-pocket ``vina_high_affinity`` rate on the
Catherine closure-bounded cohort at ``exhaustiveness=4`` and
``n_poses=1`` (TODO-125 §0, Wf-126 P3 staging JSONs).

On pocket 4aaw the closure-bounded result was 0/5 = 0.0 rate
(``molmetal/reports/wf126_p3_vina_per_pocket/4aaw/vina_high_affinity.json``)
— far below the 0.30 uniform floor in TODO-125 §5. The historical
**unconstrained** baseline is 0.65 (n=20), so the closure filter
selectively killed high-affinity candidates on this pocket.

Wf-141 v2 demonstrated that bumping ``exhaustiveness`` from 8 → 32
delivers a real kcal/mol lift on cisplatin (Trott 2010 "production"
setting). Wf-142 applies the same lift to the per-pocket
``high_affinity`` rate by re-docking the Wf-140 closure-bounded cohort
(n=14 SMILES on 4aaw) at ``exhaustiveness=32`` + ``n_poses=5`` and
counting ``kcal_min < reference_kcal_mol`` (the same criterion as
Wf-126 baseline).

Inputs
------
* ``molmetal/reports/wf140_p1_sa_5pocket/4aaw/cohort_smiles.jsonl`` —
  Wf-140 closure-bounded cohort (n=14 SMILES).
* ``metrics/by_metric/vina_high_affinity_4aaw.json`` — Wf-126 baseline
  (box center, box side, reference_kcal_mol, historical baseline rate).
* ``molmetal/reports/wf30_pocket_prep/4aaw/receptor.pdbqt`` — receptor
  PDBQT (Wf-30 pocket prep).
* ``vina`` Python binding 1.2.7 + ``meeko`` 0.8.0 + ``rdkit``
  (mol-metal-venv).

Outputs
-------
* ``molmetal/reports/wf142_ex32_high_affinity/4aaw/per_smiles_results.json``
  — per-SMILES Vina kcal_min + high_affinity flag + aggregate stats.
* ``molmetal/reports/wf142_ex32_high_affinity/4aaw/run.log`` — wall-clock
  log.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* Reference = co-crystal ligand redocked kcal/mol is a **strong target**
  — small closure-bounded fragments may not reach it. We do not pretend
  every pocket lifts.
* Vina stochastic search can occasionally fail to improve (or regress)
  when exhaustiveness increases — honest reporting of ``rate=0`` is
  built in (no silent suppression).
* Wf-140 cohort is closure-bounded (Catherine 6/6 PASS) — small
  fragments, lower SA but lower binding potency than the unconstrained
  cohort.

Usage
-----
::

    molmetal/.venvs/mol-metal-venv/bin/python \\
        molmetal/scripts/bench_vina_high_affinity_ex32.py \\
        --pocket 4aaw
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf142_ex32_high_affinity")

# ---------------------------------------------------------------------------
# Constants — pocket + cohort + baseline
# ---------------------------------------------------------------------------
POCKETS: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
EXHAUSTIVENESS = 32
N_POSES = 5
SEED = 0

WF140_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf140_p1_sa_5pocket"
WF126_P3_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf126_p3_vina_per_pocket"
WF30_PREP_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf142_ex32_high_affinity"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _load_wf126_baseline(pocket_id: str) -> Dict[str, object]:
    """Read the Wf-126 baseline JSON for ``pocket_id``.

    Returns the ``value``, ``pocket``, and ``wf126_strengthen`` blocks.
    Raises ``FileNotFoundError`` if the JSON is missing.
    """
    path = (
        PROJECT_ROOT
        / "metrics"
        / "by_metric"
        / f"vina_high_affinity_{pocket_id}.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Wf-126 baseline JSON missing: {path}"
        )
    return json.loads(path.read_text())


def _find_receptor(pocket_id: str) -> Path:
    """Locate the staged receptor PDBQT for ``pocket_id``.

    Searches in priority order:
      1. Wf-126 P3 staging (if present)
      2. Wf-30 pocket prep
    """
    for staging in (WF126_P3_DIR, WF30_PREP_DIR):
        candidate = staging / pocket_id / "receptor.pdbqt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No staged receptor PDBQT for {pocket_id} (looked in "
        f"{WF126_P3_DIR / pocket_id} and {WF30_PREP_DIR / pocket_id})"
    )


def _receptor_centroid(pdbqt_path: Path) -> Tuple[float, float, float]:
    """Compute the geometric centroid of ATOM lines in a PDBQT file.

    This matches the Wf-126 P3 closure-bias path (see
    ``bench_vina_per_pocket_closure.py::_receptor_center``), which uses
    the centroid instead of the JSON's co_crystal_ligand_center_A so
    that the docking box actually covers the binding site. The JSON's
    center is the **co-crystal ligand center**, which can lie far from
    the receptor atom centroid after pocket-prep coordinate transforms
    (e.g. pocket 3eml: JSON center (-9.06, -7.14, 55.86), centroid
    (-5.66, -20.39, 21.83)).
    """
    coords: List[Tuple[float, float, float]] = []
    with pdbqt_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                coords.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except (ValueError, IndexError):
                continue
    if not coords:
        return (0.0, 0.0, 0.0)
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    return (cx, cy, cz)


def _load_cohort(pocket_id: str) -> List[Dict[str, object]]:
    """Load the Wf-140 closure-bounded cohort SMILES for ``pocket_id``.

    Returns a list of dicts (one per line). Skips blank lines.
    """
    cohort_path = WF140_DIR / pocket_id / "cohort_smiles.jsonl"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Wf-140 cohort missing: {cohort_path}"
        )
    cohort: List[Dict[str, object]] = []
    for line in cohort_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cohort.append(json.loads(line))
    return cohort


# ---------------------------------------------------------------------------
# Core: ligand PDBQT generation (RDKit + Meeko) + Vina dock
# ---------------------------------------------------------------------------
def _smiles_to_pdbqt_string(smiles: str, seed: int = SEED) -> Optional[str]:
    """Build a ligand PDBQT string from SMILES via RDKit + Meeko.

    Returns ``None`` if RDKit parse / Meeko generation fails.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        LOG.error("import failed: %s", exc)
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=int(seed))
        AllChem.MMFFOptimizeMolecule(mol_h)
        preparator = MoleculePreparation()
        for setup in preparator.prepare(mol_h):
            s, is_ok, _ = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return s
        return None
    except Exception as exc:  # noqa: BLE001
        LOG.error("PDBQT generation failed for %s: %s", smiles, exc)
        return None


def _dock_one(
    receptor_pdbqt: Path,
    pdbqt_string: str,
    center_xyz: List[float],
    box_size: Tuple[float, float, float],
    seed: int,
    exhaustiveness: int,
    n_poses: int,
) -> Tuple[Optional[float], Optional[float], float]:
    """Run real Vina 1.2.7 (Python binding) at given exhaustiveness.

    Returns ``(kcal_min, kcal_mean_top_n, elapsed_s)``; ``None`` kcal
    fields if Vina raised.
    """
    try:
        import vina  # type: ignore
    except Exception as exc:  # noqa: BLE001
        LOG.error("vina import failed: %s", exc)
        return None, None, 0.0

    seed_eff = int(seed) % (2**31 - 1) or (2**31 - 1)
    t0 = time.time()
    try:
        v = vina.Vina(sf_name="vina", seed=seed_eff, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_string(pdbqt_string)
        v.compute_vina_maps(center=[float(c) for c in center_xyz], box_size=list(box_size))
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        energies = v.energies(n_poses=int(n_poses))
        kcal_min = float(energies[0, 0])
        kcal_mean = float(
            _mean([float(energies[i, 0]) for i in range(energies.shape[0])])
        )
        return kcal_min, kcal_mean, time.time() - t0
    except Exception as exc:  # noqa: BLE001
        LOG.error("Vina dock failed: %s\n%s", exc, traceback.format_exc())
        return None, None, time.time() - t0


# ---------------------------------------------------------------------------
# Driver: full pocket run
# ---------------------------------------------------------------------------
def run_pocket(pocket_id: str, out_dir: Path) -> Dict[str, object]:
    """Re-dock the Wf-140 closure-bounded cohort at ex=32 + n_poses=5.

    Writes ``<out_dir>/<pocket_id>/per_smiles_results.json``.
    """
    out_pocket = out_dir / pocket_id
    out_pocket.mkdir(parents=True, exist_ok=True)

    baseline = _load_wf126_baseline(pocket_id)
    pocket_block = baseline.get("pocket", {})
    json_box_center = pocket_block.get("co_crystal_ligand_center_A")
    json_box_side = pocket_block.get("box_side_A")
    # Hard-code to 20 Å to match Wf-126 P3 — the JSON's box_side_A (28 Å)
    # is from the historical unconstrained run and exceeds the pocket10
    # subset's actual binding-site size.
    box_side = 20.0
    box_size = (box_side, box_side, box_side)
    value_block = baseline.get("value", {})
    ref_kcal = float(value_block.get("reference_kcal_mol"))

    if json_box_center is None:
        raise RuntimeError(
            f"{pocket_id}: baseline JSON has no co_crystal_ligand_center_A"
        )

    receptor = _find_receptor(pocket_id)
    # Use centroid-derived center (matches Wf-126 P3 path) so the box
    # actually covers the binding site. The JSON's co_crystal_ligand_center_A
    # is the co-crystal ligand position (can lie far from the pocket after
    # coordinate transforms — on 4aaw it's ~100 Å from the pocket10 subset).
    box_center_xyz = _receptor_centroid(receptor)
    LOG.info(
        "Box center: JSON=%s centroid=%s box_side=%.1f(json=%.1f) reference=%.3f",
        json_box_center, box_center_xyz, box_side,
        float(json_box_side) if json_box_side is not None else float("nan"),
        ref_kcal,
    )

    cohort = _load_cohort(pocket_id)

    # Wf-126 P3 closure-bounded baseline rate (n=5 docked at ex=4).
    # Used as the "delta_vs_baseline_ex4" anchor for honest framing.
    wf126_p3_path = WF126_P3_DIR / pocket_id / "vina_high_affinity.json"
    wf126_p3_aggregate = {}
    if wf126_p3_path.exists():
        wf126_p3_aggregate = json.loads(wf126_p3_path.read_text()).get(
            "aggregate", {}
        )
    baseline_rate_ex4 = float(
        wf126_p3_aggregate.get("high_affinity_rate_aggregated", 0.0) or 0.0
    )
    baseline_n_docked = int(
        wf126_p3_aggregate.get("n_docked_total", 0) or 0
    )

    per_smiles: List[Dict[str, object]] = []
    n_docked = 0
    n_high_affinity = 0
    kcal_mins: List[float] = []
    kcal_means_top5: List[float] = []
    elapsed_total = 0.0

    for i, item in enumerate(cohort):
        smiles = str(item.get("smiles", item.get("SMILES", "")))
        if not smiles:
            continue
        LOG.info("[%d/%d] %s ...", i + 1, len(cohort), smiles)
        pdbqt_string = _smiles_to_pdbqt_string(smiles, seed=SEED)
        if pdbqt_string is None:
            per_smiles.append({
                "smiles": smiles,
                "kcal_min": None,
                "kcal_mean_top5": None,
                "high_affinity": None,
                "error": "pdbqt_generation_failed",
                "wall_time_s": 0.0,
            })
            continue

        kcal_min, kcal_mean, elapsed = _dock_one(
            receptor_pdbqt=receptor,
            pdbqt_string=pdbqt_string,
            center_xyz=list(box_center_xyz),
            box_size=box_size,
            seed=SEED,
            exhaustiveness=EXHAUSTIVENESS,
            n_poses=N_POSES,
        )
        elapsed_total += elapsed
        if kcal_min is None:
            per_smiles.append({
                "smiles": smiles,
                "kcal_min": None,
                "kcal_mean_top5": None,
                "high_affinity": None,
                "error": "vina_dock_failed",
                "wall_time_s": elapsed,
            })
            continue

        n_docked += 1
        kcal_mins.append(kcal_min)
        kcal_means_top5.append(kcal_mean or kcal_min)
        high_aff = kcal_min < ref_kcal
        if high_aff:
            n_high_affinity += 1
        per_smiles.append({
            "smiles": smiles,
            "kcal_min": kcal_min,
            "kcal_mean_top5": kcal_mean,
            "high_affinity": high_aff,
            "wall_time_s": elapsed,
        })
        LOG.info(
            "  kcal_min=%.3f kcal_mean_top5=%.3f high_aff=%s (%.2fs)",
            kcal_min, kcal_mean or kcal_min, high_aff, elapsed,
        )

    rate = n_high_affinity / n_docked if n_docked > 0 else 0.0
    delta = rate - baseline_rate_ex4

    summary = {
        "_workflow": "Wf-142",
        "_phase": "P1",
        "_title": (
            f"Per-pocket high_affinity lift on Wf-140 closure-bounded "
            f"cohort at ex={EXHAUSTIVENESS}, n_poses={N_POSES}"
        ),
        "_pocket": pocket_id,
        "_reference_kcal_mol": ref_kcal,
        "_exhaustiveness": EXHAUSTIVENESS,
        "_n_poses": N_POSES,
        "_seed": SEED,
        "_box_center_A": list(box_center_xyz),
        "_json_box_center_A": list(json_box_center),
        "_box_size_A": list(box_size),
        "_receptor_pdbqt": str(receptor),
        "_wf126_baseline_ex4": {
            "rate": baseline_rate_ex4,
            "n_docked": baseline_n_docked,
            "source": str(wf126_p3_path) if wf126_p3_path.exists() else None,
        },
        "_historical_baseline_unconstrained": float(
            value_block.get("rate", 0.0)
        ),
        "_n_cohort_total": len(cohort),
        "_n_docked": n_docked,
        "_n_high_affinity": n_high_affinity,
        "_high_affinity_rate": rate,
        "_delta_vs_baseline_ex4": delta,
        "_kcal_min_overall": min(kcal_mins) if kcal_mins else None,
        "_kcal_mean_top5_avg": _mean(kcal_means_top5) if kcal_means_top5 else None,
        "_elapsed_s_total": elapsed_total,
        "_generated_utc": _utc_now_iso(),
        "_engine": "vina 1.2.7 (Python binding) + meeko 0.8.0 + rdkit",
        "_per_smiles": per_smiles,
    }
    out_path = out_pocket / "per_smiles_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    LOG.info(
        "%s: n_docked=%d, n_high_affinity=%d, rate=%.4f, "
        "delta_vs_baseline_ex4=%+.4f, kcal_min=%.3f (%.1fs total)",
        pocket_id, n_docked, n_high_affinity, rate, delta,
        summary["_kcal_min_overall"] if kcal_mins else float("nan"),
        elapsed_total,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_vina_high_affinity_ex32",
        description=(
            "Wf-142 — re-dock Wf-140 closure-bounded cohort on a pocket "
            "at ex=32 + n_poses=5 and compare high_affinity rate vs the "
            "Wf-126 P3 ex=4 closure-bounded baseline."
        ),
    )
    p.add_argument(
        "--pocket", default="4aaw", choices=POCKETS,
        help="Pocket to dock (default 4aaw).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output root (default: molmetal/reports/wf142_ex32_high_affinity).",
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
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_pocket(pocket_id=args.pocket, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Wf-142 failed for %s: %s\n%s", args.pocket, exc, traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))