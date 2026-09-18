# First 10 CrossDocked receptors: strict preparation and traceable resolution

Original strict run: **4/10 prepared**, with exact failure logs in
`../crossdocked_first10_preparation/report.json`. No original data or
preparation acceptance criteria were relaxed.

Traceable resolution run: **9/10 prepared**. Reproduce:

```bash
uv run python -m molmetal.scripts.prepare_crossdocked_batch --n-pockets 10 --resolve-conformers --out-dir molmetal/reports/crossdocked_first10_resolved
uv run pytest -q molmetal/tests/test_prepare_crossdocked_receptor.py molmetal/tests/test_receptor_preparation_for_evaluation.py
```

The batch exits 1 because `test_005` remains failed. Tests: **14 passed**.
The batch's `report.json` records every input path/hash, Meeko command,
stdout/stderr, raw strict failure, resolution operation and final audit.
Each successful entry exposes `pdbqt`, `audit` and `effective_pdb` in its
preparation result. PoseBusters must use that same effective PDB.

| Pocket | Initial strict result | Final result |
|---|---|---|
| test_000 | unresolved altloc A:44, A:90 | selected A, labels normalized; preserved all atom records |
| test_001 | prepared | unchanged strict success |
| test_002 | prepared | unchanged strict success |
| test_003 | prepared | unchanged strict success |
| test_004 | unresolved altloc A:246, A:247, A:371 | selected A, labels normalized; preserved all atom records |
| test_005 | missing ASP B:101 CG/OD1/OD2 | failed; genuine missing crystallographic atoms |
| test_006 | Meeko O/OXT names swapped, apparent 2.206905 Å displacement | derived PDBQT restores source names; zero physical displacement |
| test_007 | unresolved altloc A:176, A:204 | selected A, labels normalized; preserved all atom records |
| test_008 | prepared | unchanged strict success |
| test_009 | unresolved altloc A:110, A:251 | selected A, labels normalized; preserved all atom records |

`molmetal.scripts.receptor_preparation_for_evaluation` implements the
reusable `prepare_receptor_for_evaluation(original, output_dir)` helper.
It always attempts strict preparation first. Where needed it derives one
residue-consistent altloc using highest mean occupancy (tie: A first, then
lexicographic order), retains shared atoms, and clears the selected altloc
label only in the derived file. Every selected and excluded alternative
atom is recorded. A chosen alternative lacking an atom present in another
variant is rejected rather than silently deleting a heavy-atom identity.

In these four actual inputs, CrossDocked already retained only the A
variant. Exactly **zero alternative atom records were removed**. The
derived files merely normalize the remaining A labels for Meeko. Generic
multiple-variant selection is covered by tests but not falsely claimed to
have occurred in these data.

For test_006, Meeko exchanged only terminal ASN B:255 O and OXT names.
The two records retained the original coordinate set, element O, AutoDock
type OA and identical charge −0.538. The helper writes a new PDBQT with the
source names restored at their original coordinates, leaving every other
field unchanged. The final strict audit has 0.000 Å displacement. The raw
Meeko output and explicit mapping are preserved. Arbitrary displacements,
different atom types or different charges cannot enter this repair path.

The downloaded full RCSB `4RN0.pdb` explicitly states:

```text
REMARK 470     ASP B 101    CG   OD1  OD2
```

Those three side-chain atoms were absent in the deposited crystal itself.
The remaining native atoms align to the transformed CrossDocked coordinates
within rounding precision. The complete source therefore cannot provide
observed replacement coordinates. No side chain was invented. Source URLs,
hashes and alignment evidence are retained in
`../receptor_recovery_sources/report.json` together with downloaded 4RN0
and 1FMC originals.

These are receptor-readiness results, not docking-pilot outcomes. All 10
entries remain in the readiness report; the failed target must not be
silently omitted from pilot denominators.
