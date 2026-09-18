# CrossDocked test_001 receptor preparation and physical smoke

The exact `test_001` manifest receptor was prepared successfully using
Meeko 0.8.0, then docked with the QuickVina 2 adapter. This is **one CCO
smoke**, not a successful biological candidate or a completed pilot.

```bash
HIP_VISIBLE_DEVICES=0 uv run python -m molmetal.scripts.prepare_crossdocked_receptor --pocket test_001 --dock-smiles CCO
uv run pytest -q molmetal/tests/test_prepare_crossdocked_receptor.py
```

The first command exited 0, with `passed: true`. The tests report **7 passed**.
Exact input paths, SHA-256 hashes, Meeko command/stdout/stderr, and the
resulting ligand coordinates are recorded in `report.json`. The prepared
receptor is `receptor.pdbqt`, with Meeko's full parameterized data retained
in `receptor.json`.

The pair is `GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0`, taken
from `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/`.
The PDB receptor and SDF reference ligand are exactly the row selected from
`molmetal/data/crossdocked100_manifest.csv`; no unrelated complete PDB,
pre-existing PARP1 cache, or synthetic receptor was substituted.

Strict preparation:

```text
mk_prepare_receptor.py --read_pdb <original manifest PDB> -o <output>/receptor -p -j --compute_charges --charge_model gasteiger
```

Meeko completed with no errors and no residue deletion or permissive flags.
It used residue templates and padding while computing Gasteiger charges.
No fallback was needed. The validation script compares every heavy atom's
chain, residue identifier, residue name, atom name, element and coordinates
against the original PDB. It rejects missing/renamed atoms, changed elements,
displacements over 0.002 Å, non-finite charges, and an all-zero charge output.
Unknown AutoDock atom types or ambiguous alternate atom identities are
rejected for explicit review instead of guessed.

| Property | Observed |
|---|---:|
| Input heavy atoms | 393 |
| Output heavy atoms | 393 |
| Original residues preserved | 53 |
| Added/retained output hydrogens | 93 polar hydrogens |
| Maximum heavy-atom coordinate displacement | 0.000 Å |
| Computed PDBQT charge range | −0.549 to +0.345 e |

Physical smoke used the ligand SDF's 31-heavy-atom centroid to define the
box center, with 4 Å padding about the greatest centroid-relative extent
and a cubic side of 21.650645 Å. All 393 prepared receptor heavy atoms were
passed to docking. The `Pocket` contains 197 atoms within the box half-side
sphere; it supplies the box and domain metadata, while the prepared receptor
cache supplies the full original pocket's PDBQT chemistry.

The public `VinaDockingAdapter.dock()` path ran QuickVina 2 with seed 42,
exhaustiveness 1, one CPU and one pose. It returned one nonempty `Complex`:
CCO with atomic numbers `[6, 6, 8]`, finite coordinates, and **−2.3 kcal/mol**.
The prepared cache bypasses the adapter's legacy receptor reconstruction,
while still exercising ligand preparation, the real engine, energy parsing,
pose reconstruction and domain-object creation.

This validates only the specified cropped CrossDocked receptor and a small
organic ligand. Gasteiger/template preparation is an explicit computational
model, not measured protonation chemistry. The test does not establish metal
parameterization, other pockets' preparation, receptor relaxation, or pilot
acceptance thresholds.
