"""Explicit, separate computational reconstruction of test_005 ASP B101.

Run with ``uv run --with pdbfixer==1.12.0 -m molmetal.scripts.model_test005_missing_sidechain``.
This experiment is never selected by the default receptor preparation helper.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmetal/reports/test005_modeled_sidechain"
SEED = 42


def run(report: dict) -> None:
    import numpy as np
    import openmm as mm
    import pdbfixer
    from openmm import unit
    from pdbfixer import PDBFixer
    from molmetal.scripts.prepare_crossdocked_receptor import _atoms, manifest_pair
    from molmetal.scripts.receptor_preparation_for_evaluation import prepare_receptor_for_evaluation

    original, _ = manifest_pair(ROOT / "molmetal/data/crossdocked100_manifest.csv", "test_005")
    original_bytes = original.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    before = _atoms(original, False)
    report.update(source=str(original), source_sha256=original_hash,
                  pdbfixer_version=importlib.metadata.version("pdbfixer"), openmm_version=mm.__version__,
                  forcefield="PDBFixer 1.12.0 soft.xml: bonded geometry and soft repulsion, not a validated Amber protein energy",
                  forcefield_sha256=hashlib.sha256((Path(pdbfixer.__file__).parent / "soft.xml").read_bytes()).hexdigest(),
                  protocol="Add only ASP B101 CG/OD1/OD2; all existing atoms massless during PDBFixer minimization; retain every original PDB record verbatim",
                  native_missing_evidence="RCSB 4RN0 REMARK 470 ASP B 101 CG OD1 OD2",
                  computationally_modeled_not_observed=True, default_evaluation_protocol_modified=False)
    platform = mm.Platform.getPlatformByName("OpenCL")
    for name, value in {"Precision": "mixed", "OpenCLPlatformIndex": "0", "DeviceIndex": "0"}.items():
        platform.setPropertyDefaultValue(name, value)
    fixer = PDBFixer(filename=str(original), platform=platform)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    discovered = [(r.chain.id, r.id, r.name, [a.name for a in atoms])
                  for r, atoms in fixer.missingAtoms.items()]
    report["discovered_missing_atoms"] = discovered
    report["unapplied_missing_terminals"] = [(r.chain.id, r.id, r.name, names)
                                             for r, names in fixer.missingTerminals.items()]
    if discovered != [("B", "101", "ASP", ["CG", "OD1", "OD2"])]:
        raise ValueError(f"Unexpected missing atoms; refuse scope expansion: {discovered}")
    fixer.missingResidues = {}
    fixer.missingTerminals = {}
    # Save the template placement before the standard PDBFixer optimization.
    topology, positions, new_atoms, existing = fixer._addAtomsToTopology(True, True)
    initial = {a.name: list(positions[a.index].value_in_unit(unit.angstrom)) for a in new_atoms}
    movable = {a.index for a in new_atoms}
    existing_indices = {a.index for a in existing.values()}
    report["initial_template_coordinates_A"] = initial
    report["minimizations"] = []
    minimize = mm.LocalEnergyMinimizer.minimize

    def recorded_minimize(context, *args, **kwargs):
        system = context.getSystem()
        masses = [system.getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(system.getNumParticles())]
        actual_movable = {i for i, mass in enumerate(masses) if mass != 0}
        if actual_movable != movable or any(masses[i] != 0 for i in existing_indices):
            raise ValueError("Only the three modeled atoms may have nonzero mass")
        state0 = context.getState(getEnergy=True, getPositions=True)
        entry = {"selected_platform": context.getPlatform().getName(),
                 "properties": {name: platform.getPropertyValue(context, name)
                                for name in ("DeviceName", "OpenCLPlatformName", "DeviceIndex", "Precision")},
                 "particles": system.getNumParticles(), "frozen_original_particles": len(existing_indices),
                 "movable_particle_indices": sorted(actual_movable),
                 "energy_before_kj_mol": state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)}
        if entry["properties"]["DeviceName"] != "gfx1101" or "AMD" not in entry["properties"]["OpenCLPlatformName"]:
            raise RuntimeError("Required AMD gfx1101 OpenCL device not selected")
        result = minimize(context, *args, **kwargs)
        state1 = context.getState(getEnergy=True, getPositions=True)
        xyz0 = state0.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        xyz1 = state1.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        entry["energy_after_kj_mol"] = state1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        entry["max_frozen_displacement_A"] = float(np.linalg.norm(xyz1[list(existing_indices)] - xyz0[list(existing_indices)], axis=1).max())
        report["minimizations"].append(entry)
        if entry["max_frozen_displacement_A"] > 1e-6:
            raise ValueError("Frozen atom moved during optimization")
        if not math.isfinite(entry["energy_after_kj_mol"]):
            raise ValueError("Nonfinite modeled energy")
        return result

    mm.LocalEnergyMinimizer.minimize = recorded_minimize
    try:
        fixer.addMissingAtoms(seed=SEED)
    finally:
        mm.LocalEnergyMinimizer.minimize = minimize

    additions = []
    for atom in fixer.topology.atoms():
        residue = atom.residue
        key = (residue.chain.id, residue.id + residue.insertionCode.strip(), residue.name, atom.name)
        if key not in before:
            additions.append({"identity": list(key), "element": atom.element.symbol,
                              "coordinates_A": list(fixer.positions[atom.index].value_in_unit(unit.angstrom)),
                              "provenance": "PDBFixer residue template followed by frozen-environment OpenMM AMD OpenCL optimization; computational model"})
    expected = {("B", "101", "ASP", name) for name in ("CG", "OD1", "OD2")}
    if {tuple(a["identity"]) for a in additions} != expected:
        raise ValueError(f"Unexpected modeled atoms: {additions}")
    report["added_atoms"] = additions
    lines = original_bytes.decode().splitlines(keepends=True)
    last_target = max(i for i, line in enumerate(lines)
                      if line.startswith(("ATOM  ", "HETATM")) and line[21:22] == "B" and line[22:26].strip() == "101")
    serial = max(int(line[6:11]) for line in lines if line.startswith(("ATOM  ", "HETATM")))
    added_lines = []
    for index, atom in enumerate(additions, start=1):
        _, _, residue, name = atom["identity"]
        x, y, z = atom["coordinates_A"]
        # Occupancy 0.00 explicitly marks the absence of crystallographic observation.
        line = f"ATOM  {serial+index:5d} {name:^4s} {residue:3s} B 101    {x:8.3f}{y:8.3f}{z:8.3f}{0.0:6.2f}{0.0:6.2f}          {atom['element']:>2s}  \n"
        added_lines.append(line)
        atom["pdb_line"] = line.rstrip("\n")
    derived = OUT / "receptor_modeled_ASP_B101.pdb"
    derived.write_text("REMARK 999 COMPUTATIONALLY MODELED ASP B101 CG OD1 OD2; NOT OBSERVED\n"
                       + "".join(lines[:last_target+1] + added_lines + lines[last_target+1:]))
    after = _atoms(derived, False)
    assert set(after) - set(before) == expected
    assert all(after[key]["coords"] == value["coords"] and after[key]["element"] == value["element"]
               for key, value in before.items())
    assert original.read_bytes() == original_bytes
    assert all(line in derived.read_text().splitlines(keepends=True) for line in lines)
    report.update(derived_pdb=str(derived), derived_sha256=hashlib.sha256(derived.read_bytes()).hexdigest(),
                  source_unchanged=True, original_atom_records_preserved_verbatim=True,
                  original_atoms=len(before), new_atoms=len(additions), max_original_atom_displacement_A=0.0)
    target = {name: after[("B", "101", "ASP", name)]["coords"] for name in ("CB", "CG", "OD1", "OD2")}
    report["modeled_bond_lengths_A"] = {f"{a}-{b}": math.dist(target[a], target[b])
                                         for a, b in (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"))}
    others = [v["coords"] for key, v in before.items() if key[:3] != ("B", "101", "ASP") and v["element"] != "H"]
    report["minimum_added_to_other_residue_heavy_distance_A"] = min(math.dist(a["coordinates_A"], p) for a in additions for p in others)
    prepared = prepare_receptor_for_evaluation(derived, OUT / "preparation")
    report["derived_meeko_preparation"] = prepared
    report["passed"] = bool(prepared["passed"])
    report["limitation"] = "Single seed/template local sidechain model, no rotamer ensemble or experimental validation. The original strict test_005 remains failed and this is a separately labeled preparation protocol."


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "passed": False}
    try:
        run(report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "seed.json").write_text(json.dumps({"pdbfixer_seed": SEED}, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("passed", "error", "derived_pdb", "minimizations")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
