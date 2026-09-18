# MMFF94 Fix for PoseBusters Validity Adapter

Date: 2026-09-11
Author: MolMetal PoseBusters adapter refactor

## What changed

In `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`:

1. **3D optimisation now prefers MMFF94 over UFF.** The conformer
   embedded with `AllChem.ETKDGv3(randomSeed=42)` is first
   `MMFFOptimizeMolecule(maxIters=200)`-ed; only if MMFF returns a
   non-zero status (e.g. unsupported atom types such as `Cu`,
   metalloids, charged metals) do we fall back to
   `UFFOptimizeMolecule`. MMFF has substantially better parameters
   for drug-like organics (triazoles, aromatics, heterocycles), which
   is the regime this SBDD pipeline operates in.

2. **`_df_to_report` now filters by dtype.** Previously the method
   treated every column as a pass/fail check and fed it through
   `bool(v)`. For integer count columns (`num_h_added`,
   `number_clashes`, `number_short_outlier_bonds`, …) a value of 0
   was silently read as "failed", capping the apparent pass-rate at
   ~0 % even when every real chemistry check had passed. The fix
   restricts the check set to bool-dtype columns only and additionally
   skips the reference-file loading checks (`mol_true_loaded`,
   `mol_cond_loaded`, `mol_pred_loaded`) which are always False in a
   de novo generation context (we have no experimental pose).

## New pass-rates

Validated against 13 hand-drawn CuAAC/SPAAC triazole products and
the 12 standard click tiles from
`molmetal/molmetal_lam/tile_lib/click_tiles.py::STANDARD_12_TILES`.

| Set                  | # mols | Pass | Pass-rate |
|----------------------|-------:|-----:|----------:|
| 13 CuAAC/SPAAC       |     13 |   13 | 1.000     |
| 12 click tiles       |     12 |   12 | 1.000     |
| Combined (25)        |     25 |   25 | 1.000     |

### CuAAC/SPAAC products (13)

All drawn by hand, each containing the 1,2,3-triazole core:

```
Cn1cc(C)nn1                      # 1-methyl-4-methyl
Cc1cn(-c2ccccc2)nn1              # 1-phenyl-4-methyl
CCn1cc(CC)nn1                    # 1-ethyl-4-ethyl
OCc1cn(Cc2ccccc2)nn1             # 1-benzyl-4-(HO-CH2)
OCCCn1cc(-c2ccccc2)nn1           # 1-(3-OH-propyl)-4-phenyl
NCC1=CN(C)N=N1                   # 1-methyl-4-(H2N-CH2)
COC(=O)c1cn(CC)nn1               # 1-ethyl-4-(MeO-CO)
COCc1cn(-c2ccccc2)nn1            # 1-phenyl-4-(MeO-CH2)
CC(=O)c1cn(Cc2ccccc2)nn1         # 1-benzyl-4-acetyl
CC(=O)OCCc1cn(OCC)nn1            # 1-(Et-O)-4-(AcO-CH2-CH2)
OCCc1cn(NCCC)nn1                 # 1-(3-NH2-propyl)-4-(HO-CH2)
CCCCn1cc(CCC)nn1                 # 1-butyl-4-propyl
O=C(NCCc1ccccc1)c1cn(CC)nn1      # 1-ethyl-4-amide
```

### Click tiles (12)

```
CCN=[N+]=[N-]                    # ethyl azide
[N-]=[N+]=NCc1ccccc1             # benzyl azide
[N-]=[N+]=NCCOCCO                # 2-(2-azidoethoxy)ethanol
[N-]=[N+]=Nc1ccccc1              # phenyl azide
C#CC                              # propyne
C#CCc1ccccc1                      # 3-phenyl-1-propyne
C1#CCCCCCC1                      # cyclooctyne
C#CCN                             # propargylamine
CP                                # methanethiol (Staudinger reagent)
C1=CCC=C1                         # cyclopentadiene (Diels-Alder)
C=CC(C)=O                         # methyl vinyl ketone (Diels-Alder)
O=C1C=CC(=O)N1                    # maleimide (ThiolEne)
```

## Failure summary

After the fix, **every molecule in the benchmark passes all 14 real
PoseBusters checks**:

```
sanitization, inchi_convertible, all_atoms_connected, no_radicals,
bond_lengths, bond_angles, internal_steric_clash,
aromatic_ring_flatness, non-aromatic_ring_non-flatness,
double_bond_flatness, internal_energy, passes_valence_checks,
passes_kekulization, no_radicals_before_sanitization
```

The pre-fix failure pattern (`num_h_added`, `number_clashes`,
`number_outlier_angles`, …) was not a chemistry failure: it was the
adapter reading PoseBusters' integer count columns as boolean checks
and reporting `int(0) == False`. With the dtype filter applied, only
proper pass/fail checks contribute to `pass_rate`, and the
reference-file checks are no longer scored (they require an
experimental pose we never have in de novo generation).

## Validation

```
$ PYTHONPATH=. python -m pytest molmetal/tests/test_posebusters_adapter.py -v
...
7 passed
```

New `test_mmff94_used` parameterised case asserts ≥ 60 % of the 13
CuAAC/SPAAC products pass; with the MMFF94 fix the observed rate is
1.000 (13/13), comfortably above the threshold.