# test_005 separately modeled ASP B101 sidechain (2026-09-13)

## Goal

Test explicit reconstruction of the three missing sidechain atoms without moving or deleting any observed receptor atom. Native RCSB 4RN0 explicitly lists ASP B101 CG/OD1/OD2 as missing in REMARK 470; these coordinates cannot be recovered as crystallographic observations.

## Outcome

PASS for a single-template, seed-42 modeling/preparation experiment. PDBFixer 1.12.0 added only CG/OD1/OD2. OpenMM 8.6.1 used the real AMD OpenCL platform, device gfx1101, mixed precision. All 247 original atoms had zero mass; only the three new atoms were movable. PDBFixer's soft force field energy decreased from 171951.959497 to 171941.081528 kJ/mol, with exactly zero displacement of frozen atoms. Every original PDB record is retained verbatim and the original SHA256 is unchanged.

The derived PDB explicitly labels the atoms as computational models (REMARK 999, new-atom occupancy 0.00). Strict Meeko preparation passes: 250 heavy atoms, 45 polar hydrogens, 35 residues, zero heavy-atom displacement, computed Gasteiger charges −0.549 to +0.345. Original strict test_005 remains failed; the existing first-ten evaluation protocol is unchanged.

| Atom | Derived coordinate (Å) | Origin |
|---|---|---|
| ASP B101 CG | −21.497, −5.052, −7.334 | Template plus constrained optimization |
| ASP B101 OD1 | −20.642, −5.992, −7.262 | Template plus constrained optimization |
| ASP B101 OD2 | −21.645, −4.180, −6.455 | Template plus constrained optimization |

CG–OD1 and CG–OD2 are 1.273 and 1.247 Å; CB–CG is 1.621 Å. The closest added atom to another residue's heavy atom is 4.244 Å. Full precision coordinates, masses, energy/device evidence, hashes and raw preparation outputs are in `report.json` and `preparation/`.

## Caveats

PDBFixer uses `soft.xml` geometry and soft repulsion here, not a validated Amber protein energy or a binding energy. The large total includes the frozen, discontinuous pocket; only its local optimization decrease is reported. The CB–CG length is somewhat long, and this single template/seed has no rotamer-ensemble or experimental validation. Successful Meeko preparation establishes parameterization readiness, not sidechain correctness or docking validity. PDBFixer also suggested terminal GLY B302 OXT; that addition was explicitly disabled. No ligand docking was performed in this experiment.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=0 uv run --with pdbfixer==1.12.0 python -m molmetal.scripts.model_test005_missing_sidechain
```

PDBFixer is an isolated uv overlay; project dependency files and default receptor helpers are unchanged. The script verifies the exact missing-atom set, nonzero masses restricted to the three modeled atoms, original-record/coordinate preservation, actual AMD device and strict downstream preparation. `seed.json` records seed 42. Force field and source hashes are in `report.json`.
