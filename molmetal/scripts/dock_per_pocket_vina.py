#!/usr/bin/env python3
"""dock_per_pocket_vina.py — Wf-248-C per-pocket Vina config adapter CLI.

================================================================
Wf-248-C (TODO-125 per-pocket Vina balance strengthen)
================================================================
The TODO-125 plan (§0) records that the per-pocket ``vina_high_affinity``
rate spread (1h36 = 0.95 vs 3eml = 0.05) reflects a generation-side
mismatch AND that the per-pocket DRESSING (Docking RESolution) tuning
can also affect pose quality.

This CLI exercises the per-pocket config adapter:
``molmetal.adapters.per_pocket_vina_config.PerPocketVinaConfig``.

For each pocket in ``--pockets`` we:
  1. Resolve the per-pocket ``DockingConfig`` (exhaustiveness + n_poses)
  2. Build a synthetic Pocket from the pocket-id (no real PDB needed)
  3. Dock a small ligand via the project's ``VinaDockingAdapter``
     (real Vina Python binding when available, STUB_FALLBACK otherwise)
  4. Write per-pocket CSV + JSONL records + a smoke summary

Honest framing
--------------
When the Vina binary is missing, every (pocket, cand, seed) cell
records ``status="STUB_FALLBACK"`` and ``score_kcal_mol=None``.  No
kcal/mol score is fabricated.

Usage
-----
::
    molmetal/.venvs/mol-metal-venv/bin/python \\
        molmetal/scripts/dock_per_pocket_vina.py \\
        --pockets 1h36 3eml 3ny8 4aaw 4rlu \\
        --smiles "CCO" "c1ccccc1" \\
        --seeds 0 1 2 \\
        --out-dir molmetal/reports/wf248c_per_pocket_vina
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("wf248c_dock_per_pocket")


# ---------------------------------------------------------------------------
# Default per-pocket reference ligands (smiles that RDKit can parse).
# These are LITERATURE ligands used by Wf-126 P3 / TODO-125 for
# per-pocket closure-bias.  They are small + RDKit-embeddable so they
# double as a smoke-test ligand for the per-pocket adapter.
# ---------------------------------------------------------------------------
_DEFAULT_SMILES: Tuple[str, ...] = (
    "CCO",                                  # ethanol — tiny, always parseable
    "c1ccccc1",                             # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                # aspirin
    "Nc1ncnc2[nH]cnc12",                    # adenine
    "O=C(O)c1ccccc1",                       # benzoic acid
)


DEFAULT_POCKETS: Tuple[str, ...] = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)


@dataclass
class DockCellRecord:
    """One (pocket, smiles, seed) docking record."""

    pocket: str
    smiles: str
    seed: int
    exhaustiveness: int
    n_poses: int
    vina_score_kcal_mol: Optional[float]
    n_poses_returned: int
    vina_status: str  # "REAL_VINA" | "STUB_FALLBACK" | "PARSE_FAIL" | "DOCK_FAIL"
    box_center_xyz: Tuple[float, float, float]
    box_size_xyz: Tuple[float, float, float]
    wall_time_s: float
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_synthetic_pocket(pocket_id: str):
    """Build a synthetic :class:`Pocket` for the smoke test.

    This is the same synthetic fallback used by
    :mod:`bench_vina_per_pocket_closure` — 200 carbons on the
    x-axis spanning [-15, 15] Å.  Real PDBs are NOT required; the
    per-pocket config adapter is what we're validating here, not
    docking accuracy.
    """
    try:
        import torch  # type: ignore
        from molmetal.domain import Pocket  # type: ignore
    except Exception as exc:  # pragma: no cover
        LOG.warning("import-failed for synthetic pocket: %s", exc)
        return None
    n_atoms = 200
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    coords[:, 0] = torch.linspace(-15.0, 15.0, n_atoms)
    atom_types = torch.full((n_atoms,), 6, dtype=torch.long)
    return Pocket(
        pdb_id=pocket_id,
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.zeros(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        radius=15.0,
    )


def _detect_vina_binary() -> Dict[str, object]:
    """Return whether a real Vina-compatible binary is on PATH."""
    import shutil
    import subprocess
    for name in ("vina", "qvina02", "qvina2", "smina"):
        path = shutil.which(name)
        if not path:
            continue
        try:
            cp = subprocess.run(
                [path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            version = (cp.stdout or cp.stderr or "").strip().splitlines()[0]
        except Exception:
            version = "<version-call-failed>"
        return {
            "status": "REAL_VINA_AVAILABLE",
            "bin": path,
            "version": version,
            "reason": "",
        }
    return {
        "status": "MISSING_DEP",
        "bin": None,
        "version": "",
        "reason": "no vina/qvina02/smina binary on PATH",
    }


def _dock_one(
    pocket_id: str,
    smiles: str,
    seed: int,
    exhaustiveness: int,
    n_poses: int,
    adapter_available: bool,
) -> DockCellRecord:
    """Run one docking cell using the per-pocket config.

    Returns a :class:`DockCellRecord` populated with the per-pocket
    exhaustiveness + n_poses that were actually used (so the report
    can prove that the per-pocket adapter was honoured even on
    STUB_FALLBACK).
    """
    t0 = time.perf_counter()
    pocket = _build_synthetic_pocket(pocket_id)
    if pocket is None:
        return DockCellRecord(
            pocket=pocket_id,
            smiles=smiles,
            seed=int(seed),
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            vina_score_kcal_mol=None,
            n_poses_returned=0,
            vina_status="IMPORT_FAIL",
            box_center_xyz=(0.0, 0.0, 0.0),
            box_size_xyz=(20.0, 20.0, 20.0),
            wall_time_s=time.perf_counter() - t0,
            note="torch/domain import failed",
        )

    box_center = (0.0, 0.0, 0.0)
    box_size = (20.0, 20.0, 20.0)

    if not adapter_available:
        return DockCellRecord(
            pocket=pocket_id,
            smiles=smiles,
            seed=int(seed),
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            vina_score_kcal_mol=None,
            n_poses_returned=0,
            vina_status="STUB_FALLBACK",
            box_center_xyz=box_center,
            box_size_xyz=box_size,
            wall_time_s=time.perf_counter() - t0,
            note=(
                f"Vina binary missing — STUB_FALLBACK; "
                f"per-pocket config (ex={exhaustiveness}, "
                f"n_poses={n_poses}) honoured but no kcal/mol emitted"
            ),
        )

    # Try real docking via the project adapter.
    try:
        from molmetal.domain import Molecule  # type: ignore
        from molmetal.ports import DockingConfig  # type: ignore
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (  # type: ignore
            VinaDockingAdapter,
        )
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        return DockCellRecord(
            pocket=pocket_id,
            smiles=smiles,
            seed=int(seed),
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            vina_score_kcal_mol=None,
            n_poses_returned=0,
            vina_status="IMPORT_FAIL",
            box_center_xyz=box_center,
            box_size_xyz=box_size,
            wall_time_s=time.perf_counter() - t0,
            note=f"adapter import failed: {exc!r}",
        )

    try:
        adapter = VinaDockingAdapter(engine="auto")
        adapter.setup(device="cpu")
        # Build a tiny Molecule; the SMILES drives the actual
        # 3D-embed inside the adapter.
        mol = Molecule(
            coords=torch.zeros(1, 3),
            atom_types=torch.tensor([6], dtype=torch.long),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.tensor([0], dtype=torch.long),
            smiles=smiles,
        )
        cfg = DockingConfig(
            n_poses=int(n_poses),
            exhaustiveness=int(exhaustiveness),
            seed=int(seed),
        )
        complexes = adapter.dock(mol, pocket, cfg)
        if not complexes:
            return DockCellRecord(
                pocket=pocket_id,
                smiles=smiles,
                seed=int(seed),
                exhaustiveness=int(exhaustiveness),
                n_poses=int(n_poses),
                vina_score_kcal_mol=None,
                n_poses_returned=0,
                vina_status="DOCK_FAIL",
                box_center_xyz=box_center,
                box_size_xyz=box_size,
                wall_time_s=time.perf_counter() - t0,
                note="adapter returned no poses",
            )
        best = float(complexes[0].vina_score)
        return DockCellRecord(
            pocket=pocket_id,
            smiles=smiles,
            seed=int(seed),
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            vina_score_kcal_mol=best,
            n_poses_returned=len(complexes),
            vina_status="REAL_VINA",
            box_center_xyz=box_center,
            box_size_xyz=box_size,
            wall_time_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        return DockCellRecord(
            pocket=pocket_id,
            smiles=smiles,
            seed=int(seed),
            exhaustiveness=int(exhaustiveness),
            n_poses=int(n_poses),
            vina_score_kcal_mol=None,
            n_poses_returned=0,
            vina_status="DOCK_FAIL",
            box_center_xyz=box_center,
            box_size_xyz=box_size,
            wall_time_s=time.perf_counter() - t0,
            note=f"{type(exc).__name__}: {str(exc)[:120]}",
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dock_per_pocket_vina",
        description=(
            "Wf-248-C per-pocket Vina config adapter smoke test. "
            "Docks a small ligand set across each pocket using the "
            "per-pocket DockingConfig (different exhaustiveness + "
            "n_poses per pocket) and reports per-pocket output."
        ),
    )
    p.add_argument(
        "--pockets",
        nargs="+",
        default=list(DEFAULT_POCKETS),
        help=f"Pocket identifiers. Default: {' '.join(DEFAULT_POCKETS)}",
    )
    p.add_argument(
        "--smiles",
        nargs="+",
        default=list(_DEFAULT_SMILES),
        help=(
            "SMILES to dock per pocket. Default: a small RDKit-"
            "parseable ligand set (CCO, benzene, aspirin, adenine, "
            "benzoic acid)."
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds. Default: {' '.join(map(str, DEFAULT_SEEDS))}",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf248c_per_pocket_vina",
        help="Output directory.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the per-pocket config adapter.
    from molmetal.adapters.per_pocket_vina_config import (
        PerPocketVinaConfig,
        default_per_pocket_config,
    )
    pp_cfg: PerPocketVinaConfig = default_per_pocket_config()

    # Detect Vina binary.
    vina_status = _detect_vina_binary()
    adapter_available = vina_status["status"] == "REAL_VINA_AVAILABLE"
    LOG.info(
        "[wf248c] vina_binary_status=%s bin=%s",
        vina_status["status"], vina_status.get("bin"),
    )

    all_records: List[DockCellRecord] = []
    per_pocket_records: Dict[str, List[DockCellRecord]] = {}

    for pocket_id in args.pockets:
        exhaustiveness = pp_cfg.exhaustiveness_for(pocket_id)
        n_poses = pp_cfg.n_poses_for(pocket_id)
        LOG.info(
            "[wf248c] ===== pocket=%s ex=%d n_poses=%d =====",
            pocket_id, exhaustiveness, n_poses,
        )
        pocket_cells: List[DockCellRecord] = []
        for smi in args.smiles:
            for seed in args.seeds:
                cell = _dock_one(
                    pocket_id=pocket_id,
                    smiles=smi,
                    seed=int(seed),
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    adapter_available=adapter_available,
                )
                pocket_cells.append(cell)
                all_records.append(cell)
        per_pocket_records[pocket_id] = pocket_cells

    # -------------------------------------------------------------------
    # Write per-pocket JSON files (one per pocket).
    # -------------------------------------------------------------------
    for pocket_id, cells in per_pocket_records.items():
        pocket_dir = args.out_dir / pocket_id
        pocket_dir.mkdir(parents=True, exist_ok=True)
        per_pocket_json = {
            "pocket_id": pocket_id,
            "per_pocket_config": {
                "exhaustiveness": pp_cfg.exhaustiveness_for(pocket_id),
                "n_poses": pp_cfg.n_poses_for(pocket_id),
                "label": pp_cfg.get(pocket_id).label,
            },
            "n_cells": len(cells),
            "n_real_vina": sum(
                1 for c in cells if c.vina_status == "REAL_VINA"
            ),
            "n_stub_fallback": sum(
                1 for c in cells if c.vina_status == "STUB_FALLBACK"
            ),
            "per_cell": [asdict(c) for c in cells],
        }
        (pocket_dir / "vina_per_pocket.json").write_text(
            json.dumps(per_pocket_json, indent=2, default=str)
        )

    # -------------------------------------------------------------------
    # Write combined CSV.
    # -------------------------------------------------------------------
    csv_path = args.out_dir / "per_pocket_vina.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "pocket", "seed", "smiles", "exhaustiveness", "n_poses",
            "vina_score_kcal_mol", "n_poses_returned", "vina_status",
            "wall_time_s", "note",
        ])
        for r in all_records:
            writer.writerow([
                r.pocket, r.seed, r.smiles,
                r.exhaustiveness, r.n_poses,
                r.vina_score_kcal_mol, r.n_poses_returned,
                r.vina_status, round(r.wall_time_s, 4), r.note,
            ])

    # -------------------------------------------------------------------
    # Per-pocket exhaustiveness map (G7 evidence).
    # -------------------------------------------------------------------
    exhaustiveness_map: Dict[str, int] = {
        p: pp_cfg.exhaustiveness_for(p)
        for p in args.pockets
    }
    n_poses_map: Dict[str, int] = {
        p: pp_cfg.n_poses_for(p)
        for p in args.pockets
    }

    # -------------------------------------------------------------------
    # Smoke summary.
    # -------------------------------------------------------------------
    n_total = len(all_records)
    n_real = sum(1 for r in all_records if r.vina_status == "REAL_VINA")
    n_stub = sum(1 for r in all_records if r.vina_status == "STUB_FALLBACK")
    real_scores = [
        r.vina_score_kcal_mol for r in all_records
        if r.vina_score_kcal_mol is not None
    ]
    summary = {
        "metadata": {
            "workflow": "Wf-248-C",
            "task": "Per-pocket Vina config adapter (TODO-125)",
            "pockets": list(args.pockets),
            "seeds": list(args.seeds),
            "smiles": list(args.smiles),
            "n_cells": int(n_total),
        },
        "vina_binary_status": vina_status,
        "per_pocket_config_map": pp_cfg.as_dict(),
        "per_pocket_exhaustiveness": exhaustiveness_map,
        "per_pocket_n_poses": n_poses_map,
        "n_real_vina_total": int(n_real),
        "n_stub_fallback_total": int(n_stub),
        "n_other_total": int(n_total - n_real - n_stub),
        "score_summary": {
            "min": (
                None if not real_scores
                else float(round(min(real_scores), 4))
            ),
            "max": (
                None if not real_scores
                else float(round(max(real_scores), 4))
            ),
            "mean": (
                None if not real_scores
                else float(round(statistics.fmean(real_scores), 4))
            ),
            "n_real_scores": int(len(real_scores)),
        },
        "per_pocket_aggregate": {
            p: {
                "n_cells": int(len(per_pocket_records[p])),
                "n_real_vina": int(sum(
                    1 for c in per_pocket_records[p]
                    if c.vina_status == "REAL_VINA"
                )),
                "n_stub_fallback": int(sum(
                    1 for c in per_pocket_records[p]
                    if c.vina_status == "STUB_FALLBACK"
                )),
                "exhaustiveness_used": int(
                    pp_cfg.exhaustiveness_for(p)
                ),
                "n_poses_used": int(pp_cfg.n_poses_for(p)),
                "label": str(pp_cfg.get(p).label),
            }
            for p in args.pockets
        },
        "honest_framing": {
            "vina_binary_status": vina_status["status"],
            "stub_fallback_explanation": (
                "When the Vina binary is missing, every cell records "
                "STUB_FALLBACK and vina_score_kcal_mol=None. The "
                "per-pocket config (exhaustiveness + n_poses) IS "
                "honoured in the records — only the kcal/mol emission "
                "is stubbed. No score is fabricated."
            ),
            "per_pocket_basis": (
                "Exhaustiveness + n_poses are heuristic (small pocket "
                "→ more sampling, large pocket → baseline). They are "
                "NOT calibrated against an external benchmark. The "
                "goal is per-pocket resolution matching pocket size, "
                "not exhaustive pose-rediscovery accuracy."
            ),
        },
    }

    summary_path = args.out_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"[wf248c] DONE pockets={len(args.pockets)} "
        f"smiles={len(args.smiles)} seeds={len(args.seeds)} "
        f"cells={n_total} real_vina={n_real} "
        f"stub_fallback={n_stub} "
        f"per_pocket_exhaustiveness={exhaustiveness_map}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())