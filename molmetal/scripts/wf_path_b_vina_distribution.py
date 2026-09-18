"""WF-Path-B-GPU-Retrain — Phase 2 Vina distribution computation.

Honest measurement script.

Given:
  - Path-B GPU retrain produced 192 raw CFM coordinate samples
    (3 seeds × 2 pockets × 2 CFG × 16 samples).
  - n_decoded = 0 across all 192 samples (per wf_path_b_gpu_retrain/final.md).
  - Per the training-script protocol, the docking binary used during
    Path-B was `_fake_vina.sh` (exits 0; no real Vina call). The
    `physical` sub-directory of every cell is empty (verified by ls).

So a "per-cell weighted aggregation by per-pose Vina score" is
mathematically UNDEFINED — the numerator is the empty sum.

This script computes two honest measurements:

1. PRIMARY (the spec answer):
   - `vina_mean_real_kcal_mol` = NaN / undefined.
   - Reported as None in the JSON output, with the schema flag
     `vina_mean_real_undefined: true` and rationale.

2. DERIVED (redocked reference baseline per pocket):
   - For each of the 2 pockets (test_001, test_002) take the
     reference ligand (canonical SMILES from the SDF), redock it with
     the production VinaDockingAdapter (which auto-builds PDBQT
     receptors/ligands via meeko), and report the resulting score as
     the **pocket-baseline** binding affinity.
   - This is the score that any generated molecule must BEAT to claim
     a useful hit; it sets the y-axis of the Vina distribution that
     Round-13 will fill with real decoded + docked samples.
   - It is NOT a Vina-mean-from-CFM-samples. It is the upper-bound of
     the metric range.

The script then writes:

  molmetal/reports/wf_path_b_gpu_retrain/vina_distribution.json

with the schema block required by the task brief.

Run:
  PYTHONPATH=/home/hugo/codes/try_triton_on_rocm \
  uv run python molmetal/scripts/wf_path_b_vina_distribution.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
REPORT_ROOT = ROOT / "molmetal" / "reports" / "wf_path_b_gpu_retrain"
REPORT_JSON_IN = REPORT_ROOT / "cfm_10kstep" / "report.json"
MANIFEST = ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"
OUT_JSON = REPORT_ROOT / "vina_distribution.json"


def load_report() -> dict:
    with REPORT_JSON_IN.open() as f:
        return json.load(f)


def per_cell_status_aggregation(report: dict) -> dict:
    """Aggregate per-cell decode status counts; produce the canonical view."""
    cells = report["cells"]
    n_cells = len(cells)
    status_counts: dict[str, int] = {}
    n_decoded_total = 0
    n_docked_total = 0
    n_pb_pass_total = 0
    n_requested_total = 0

    for cell in cells:
        for k, v in cell.get("decode_status_counts", {}).items():
            status_counts[k] = status_counts.get(k, 0) + int(v)
        n_decoded_total += int(cell.get("n_decoded", 0))
        n_docked_total += int(cell.get("physical", {}).get("summary", {}).get("n_docked", 0))
        n_pb_pass_total += int(cell.get("physical", {}).get("summary", {}).get("n_pb_pass", 0))
        n_requested_total += int(cell.get("n_requested", 0))

    return {
        "n_cells": n_cells,
        "n_requested_total": n_requested_total,
        "n_decoded_total": n_decoded_total,
        "n_docked_total": n_docked_total,
        "n_pb_pass_total": n_pb_pass_total,
        "decode_status_counts": status_counts,
        "vina_mean_real_kcal_mol": None,
        "vina_mean_real_undefined": True,
        "vina_mean_real_undefined_reason": (
            "n_decoded_total=0 across 192 raw CFM samples; the per-pose "
            "Vina aggregation is an empty sum; kcal/mol units are "
            "undefined. The training script used _fake_vina.sh which "
            "exits 0 without producing a Vina score; no real docking "
            "happened during the Path-B GPU retrain. See "
            "molmetal/reports/wf_path_b_gpu_retrain/final.md §6 (honest "
            "limitations)."
        ),
    }


def redock_reference(pocket_id: str, ligand_sdf: Path, receptor_pdb: Path) -> dict:
    """Redock the canonical reference ligand into the pocket using VinaDockingAdapter."""
    try:
        from rdkit import Chem
        from molmetal.domain import Molecule, Pocket
        from molmetal.ports import DockingConfig
        from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
        import torch

        lig_mol = next((m for m in Chem.SDMolSupplier(str(ligand_sdf), removeHs=False) if m is not None), None)
        if lig_mol is None:
            return {
                "pocket_id": pocket_id,
                "vina_kcal_mol_redocked_reference": None,
                "engine": "VinaDockingAdapter",
                "error": "Could not parse reference ligand SDF",
                "ligand_sdf": str(ligand_sdf),
                "receptor_pdb": str(receptor_pdb),
            }

        center = lig_mol.GetConformer().GetPositions().mean(axis=0)
        center_t = torch.tensor(center, dtype=torch.float64)

        # Build Pocket via Pocket.from_pdb_file (matches production flow).
        # Pocket is frozen; we don't attach _pdb_path — the adapter will use
        # the round-trip fallback + mk_prepare_receptor.
        pocket = Pocket.from_pdb_file(str(receptor_pdb), center_t, radius=12.0)

        smiles = Chem.MolToSmiles(Chem.RemoveHs(lig_mol))
        molecule = Molecule.from_smiles(smiles, embed_3d=True)

        adapter = VinaDockingAdapter(engine="vina")
        cfg = DockingConfig(
            n_poses=1,
            exhaustiveness=4,
        )

        complexes = adapter.dock(molecule, pocket, cfg)
        if not complexes:
            return {
                "pocket_id": pocket_id,
                "vina_kcal_mol_redocked_reference": None,
                "engine": "VinaDockingAdapter",
                "error": "Dock returned no poses (receptor/ligand prep may have failed)",
                "ligand_sdf": str(ligand_sdf),
                "receptor_pdb": str(receptor_pdb),
            }
        # Filter to finite (non-NaN) scores; pick lowest (best).
        finite_complexes = [c for c in complexes if c.vina_score is not None and np.isfinite(c.vina_score)]
        if not finite_complexes:
            return {
                "pocket_id": pocket_id,
                "vina_kcal_mol_redocked_reference": None,
                "engine": "VinaDockingAdapter",
                "error": "All returned poses have non-finite vina_score",
                "ligand_sdf": str(ligand_sdf),
                "receptor_pdb": str(receptor_pdb),
                "n_poses_returned": len(complexes),
                "raw_scores": [c.vina_score for c in complexes],
            }
        best = min(finite_complexes, key=lambda c: c.vina_score)
        return {
            "pocket_id": pocket_id,
            "vina_kcal_mol_redocked_reference": float(best.vina_score),
            "engine": "VinaDockingAdapter (exhaustiveness=4, n_poses=1)",
            "ligand_sdf": str(ligand_sdf),
            "receptor_pdb": str(receptor_pdb),
            "n_poses_returned": len(complexes),
        }
    except Exception as exc:
        return {
            "pocket_id": pocket_id,
            "vina_kcal_mol_redocked_reference": None,
            "engine": "VinaDockingAdapter",
            "error": f"{type(exc).__name__}: {exc}",
            "ligand_sdf": str(ligand_sdf),
            "receptor_pdb": str(receptor_pdb),
        }


def aggregate_redocked_reference(pockets: dict[str, dict]) -> dict:
    """Compute the per-pocket redocked reference Vina baseline."""
    results = []
    for pocket_id, paths in pockets.items():
        rec = redock_reference(
            pocket_id=pocket_id,
            ligand_sdf=Path(paths["ligand_sdf"]),
            receptor_pdb=Path(paths["receptor_pdb"]),
        )
        if rec is not None:
            results.append(rec)
    scores = [
        r["vina_kcal_mol_redocked_reference"]
        for r in results
        if r.get("vina_kcal_mol_redocked_reference") is not None
    ]
    if not scores:
        return {
            "n_pockets_attempted": len(pockets),
            "n_pockets_succeeded": 0,
            "vina_mean_redocked_reference_kcal_mol": None,
            "vina_mean_std": None,
            "vina_below_minus8_count": 0,
            "high_affinity_rate_vs_minus8": None,
            "per_pocket": results,
        }
    arr = np.asarray(scores, dtype=float)
    below_minus8 = int((arr < -8.0).sum())
    return {
        "n_pockets_attempted": len(pockets),
        "n_pockets_succeeded": len(scores),
        "vina_mean_redocked_reference_kcal_mol": float(arr.mean()),
        "vina_mean_std": float(arr.std(ddof=0)),
        "vina_min_redocked_reference_kcal_mol": float(arr.min()),
        "vina_max_redocked_reference_kcal_mol": float(arr.max()),
        "vina_below_minus8_count": below_minus8,
        "high_affinity_rate_vs_minus8": float(below_minus8 / len(scores)),
        "per_pocket": results,
    }


def read_manifest_pockets() -> dict[str, dict]:
    wanted = ("test_001", "test_002")
    out: dict[str, dict] = {}
    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["pocket_id"] in wanted:
                out[row["pocket_id"]] = {
                    "receptor_pdb": row["receptor_path"],
                    "ligand_sdf": row["ligand_path"],
                }
    return out


def main() -> int:
    report = load_report()
    cell_agg = per_cell_status_aggregation(report)
    pockets = read_manifest_pockets()
    ref_agg = aggregate_redocked_reference(pockets)

    out = {
        "task": "WF-Path-B-GPU-Retrain Phase 2 — Vina distribution from real CFM samples",
        "date_utc": "2026-09-15",
        "scope": (
            "Honest measurement: aggregate Vina mean +/- std across the "
            "192 raw CFM samples from the 10000-step + h=64 + Path-B "
            "decoder-rework retrain (3 seeds x 2 pockets x 2 CFG x 16 "
            "samples)."
        ),
        "primary_measurement": cell_agg,
        "derived_reference_baseline": ref_agg,
        "schema_metrics": {
            "vina_mean_real_kcal_mol": None,
            "vina_mean_std": None,
            "n_decoded_aggregated": cell_agg["n_decoded_total"],
            "n_docked_aggregated": cell_agg["n_docked_total"],
            "n_pb_pass_aggregated": cell_agg["n_pb_pass_total"],
            "vina_mean_undefined_reason": cell_agg["vina_mean_real_undefined_reason"],
        },
        "decision_inputs": {
            "decode_ratio": (
                cell_agg["n_decoded_total"]
                / max(1, cell_agg["n_requested_total"])
            ),
            "all_status_counts": cell_agg["decode_status_counts"],
            "pocket_redocked_reference_mean_kcal_mol": (
                ref_agg["vina_mean_redocked_reference_kcal_mol"]
            ),
        },
        "honest_framing": (
            "The user-requested metric "
            "(vina_mean = sum(n_decoded * per-pose Vina) / sum(n_decoded)) "
            "is undefined because the denominator is 0 — every cell of the "
            "192-sample grid reported n_decoded=0. The per-pocket redocked "
            "reference ligand baseline is reported as a DERIVED measurement "
            "that bounds the Vina distribution y-axis that Round-13 will "
            "fill. It is NOT a Vina-mean-from-CFM-samples."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
