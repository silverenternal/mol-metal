"""Prepare an exact CrossDocked manifest receptor with strict Meeko chemistry.

No residue deletion, residue renaming, coordinate invention, charge-zeroing,
or synthetic receptor fallback is allowed. An optional real docking smoke
uses the prepared PDBQT in the adapter receptor cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def manifest_pair(manifest: Path, pocket_id: str) -> tuple[Path, Path]:
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["pocket_id"] == pocket_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {pocket_id} row; found {len(rows)}")
    paths = tuple(Path(rows[0][key]).expanduser().resolve() for key in ("receptor_path", "ligand_path"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _atoms(path: Path, pdbqt: bool) -> dict:
    """Read identities/coordinates; refuse duplicate atom identities."""
    records = {}
    ad_elements = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
                   "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
                   "P": "P", "F": "F", "Cl": "CL", "Br": "BR", "I": "I",
                   "H": "H", "HD": "H", "HS": "H"}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_type = line[77:].strip() if pdbqt else line[76:78].strip()
        if pdbqt:
            element = ad_elements.get(atom_type)
            if element is None:
                raise ValueError(f"Unvalidated AutoDock type: {atom_type}")
        else:
            element = atom_type.upper()
            if not element:
                raise ValueError("Original PDB must carry explicit elements")
        key = (line[21:22], line[22:27].strip(), line[17:20].strip(), line[12:16].strip())
        if key in records:
            raise ValueError(f"Duplicate atom identity {key}; resolve altloc explicitly")
        records[key] = {
            "element": element,
            "coords": [float(line[start:start + 8]) for start in (30, 38, 46)],
            "charge": float(line[70:76]) if pdbqt else None,
            "atom_type": atom_type,
        }
    if not records:
        raise ValueError(f"No atoms in {path}")
    return records


def audit_receptor(original: Path, prepared: Path) -> dict:
    before, after = _atoms(original, False), _atoms(prepared, True)
    before_heavy = {key: value for key, value in before.items() if value["element"] != "H"}
    after_heavy = {key: value for key, value in after.items() if value["element"] != "H"}
    if before_heavy.keys() != after_heavy.keys():
        raise ValueError("Preparation changed heavy-atom/residue identities")
    maximum = 0.0
    for key in before_heavy:
        if before_heavy[key]["element"] != after_heavy[key]["element"]:
            raise ValueError(f"Preparation changed element for {key}")
        delta = math.dist(before_heavy[key]["coords"], after_heavy[key]["coords"])
        if not math.isfinite(delta) or delta > 0.002:
            raise ValueError(f"Preparation moved {key} by {delta} Angstrom")
        maximum = max(maximum, delta)
    charges = [record["charge"] for record in after.values()]
    if not all(math.isfinite(charge) for charge in charges):
        raise ValueError("Non-finite prepared charge")
    if not any(abs(charge) > 1e-6 for charge in charges):
        raise ValueError("All prepared charges are zero; expected computed Gasteiger charges")
    return {
        "input_heavy_atoms": len(before_heavy), "output_heavy_atoms": len(after_heavy),
        "output_hydrogens": len(after) - len(after_heavy),
        "residue_count": len({key[:3] for key in before_heavy}),
        "heavy_atom_identities_preserved": True, "elements_preserved": True,
        "max_heavy_atom_displacement_A": maximum,
        "charge_model": "Meeko computed Gasteiger with template padding",
        "charge_min": min(charges), "charge_max": max(charges),
    }


def prepare_receptor(original: Path, output_dir: Path) -> dict:
    executable = shutil.which("mk_prepare_receptor.py")
    if executable is None:
        raise FileNotFoundError("mk_prepare_receptor.py unavailable; run with project uv environment")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "receptor"
    command = [sys.executable, executable, "--read_pdb", str(original),
               "-o", str(base), "-p", "-j", "--compute_charges", "--charge_model", "gasteiger"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    record = {"command": command, "returncode": proc.returncode,
              "stdout": proc.stdout, "stderr": proc.stderr,
              "pdbqt": str(base.with_suffix(".pdbqt"))}
    if proc.returncode != 0:
        record.update(passed=False, error="Strict Meeko preparation failed; no fallback applied")
        return record
    try:
        record["audit"] = audit_receptor(original, base.with_suffix(".pdbqt"))
        record["passed"] = True
    except (OSError, ValueError) as exc:
        record.update(passed=False, error=str(exc))
    return record


def dock_smoke(receptor: Path, prepared: Path, reference_ligand: Path, smiles: str) -> dict:
    import numpy as np
    import torch
    from rdkit import Chem
    from molmetal.domain import Molecule, Pocket
    from molmetal.ports import DockingConfig
    from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter

    reference = next((m for m in Chem.SDMolSupplier(str(reference_ligand), removeHs=False)
                      if m is not None), None)
    if reference is None:
        raise ValueError("Reference ligand SDF could not be parsed")
    heavy = [atom.GetIdx() for atom in reference.GetAtoms() if atom.GetAtomicNum() > 1]
    xyz = reference.GetConformer().GetPositions()[heavy]
    center = xyz.mean(axis=0)
    # Bounds symmetric about the heavy-atom centroid, plus 4 A each side.
    length = max(12.0, 2.0 * float(np.abs(xyz - center).max()) + 8.0)
    pocket = Pocket.from_pdb_file(receptor, torch.tensor(center), radius=length / 2)
    adapter = VinaDockingAdapter(engine="quickvina2", cpu_count=1, default_box_padding=0)
    adapter._receptor_pdbqt[pocket.pdb_id] = prepared
    config = DockingConfig(seed=42, exhaustiveness=1, n_poses=1)
    complexes = adapter.dock(Molecule.from_smiles(smiles), pocket, config)
    rows = [{"score_kcal_mol": item.vina_score,
             "smiles": item.molecule.smiles,
             "coordinates": item.molecule.coords.tolist(),
             "atom_numbers": item.molecule.atom_types.tolist(),
             "finite_coordinates": bool(torch.isfinite(item.molecule.coords).all())}
            for item in complexes]
    return {
        "passed": bool(rows) and all(row["score_kcal_mol"] is not None
                                     and math.isfinite(row["score_kcal_mol"])
                                     and row["finite_coordinates"] for row in rows),
        "scope": "N=1 smoke, not a pilot success criterion or engine-parity benchmark",
        "receptor_mode": "strict Meeko prepared original manifest PDB; no synthetic receptor fallback",
        "engine_binary": adapter._engine_binary, "smiles": smiles,
        "seed": 42, "exhaustiveness": 1, "cpu": 1, "n_poses": 1,
        "center_A": center.tolist(), "box_size_A": [length] * 3,
        "reference_heavy_atoms": len(heavy), "pocket_atoms": pocket.n_atoms,
        "complexes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "molmetal/data/crossdocked100_manifest.csv")
    parser.add_argument("--pocket", default="test_001")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "molmetal/reports/crossdocked_test001_receptor")
    parser.add_argument("--dock-smiles", help="Optionally run actual QuickVina public adapter dock()")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"pocket_id": args.pocket, "manifest": str(args.manifest), "passed": False}
    try:
        receptor, ligand = manifest_pair(args.manifest, args.pocket)
        report["sources"] = {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("receptor_pdb", receptor), ("reference_ligand_sdf", ligand))
        }
        report["preparation"] = prepare_receptor(receptor, args.out_dir)
        report["passed"] = report["preparation"]["passed"]
        if report["passed"] and args.dock_smiles:
            report["docking"] = dock_smoke(receptor, Path(report["preparation"]["pdbqt"]), ligand, args.dock_smiles)
            report["passed"] = report["docking"]["passed"]
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
