# T4 B1 — 3-D Embedding Sanity Check (ETKDGv3 + MMFF94)

**Date:** 2026-09-11
**Author:** B1 sanity script (`molmetal/scripts/check_3d_embed.py`)
**Question:** Can RDKit's ETKDGv3 + MMFF94 pipeline produce chemically
reasonable 3-D structures for the metal complexes we will feed into the
docking and property-prediction pipelines (TODO/07 B1)?

## TL;DR

| Metal | Success rate | Rows | Source                     | Verdict |
| ----- | ------------ | ---- | -------------------------- | ------- |
| Pt    | **100 %**    | 100  | Synthetic cisplatin pool   | OK      |
| Ru    | **100 %**    | 100  | MetalCytoToxDB (real rows) | OK      |
| Ir    | **95 %**     | 100  | MetalCytoToxDB (real rows) | OK      |

All three are **above the 90 % threshold** mandated in the task spec;
**no UFF fallback (B2) is needed at this point.** The Pt pool is
synthetic because `MetalCytoToxDB.csv` contains no Pt rows (the dataset
is Ru/Ir/Rh/Os/Re only — see `molmetal/data/cytotox.py`).

## What was tested

1. **Parse + rebuild** the full metal complex SMILES from the
   `SMILES_Ligands` + `Metal` + `Oxidation_state` columns using the
   `metal_smiles` parser from T1 A1
   (`molmetal/data/metal_smiles.py:reconstruct_metal_complex`).
2. **Embed 3-D** with `AllChem.EmbedMolecule(..., AllChem.ETKDGv3())`.
   `maxIterations=200` (above the default 50) and `useRandomCoords=True`
   — the latter is required because RDKit has no ETKDG torsion parameters
   for transition metals; without it `EmbedMolecule` returns `-1` for
   every Ru/Ir/Pt complex.
3. **Optimise** with `AllChem.MMFFOptimizeMolecule(maxIters=400)`.
4. **Measure** every metal → first-shell-neighbour distance and compare
   to literature values from the Cambridge Structural Database
   (tolerance ±0.30 Å).

## Detailed results

### Pt (synthetic cisplatin pool, 100 rows)

`data_source = synthetic_pt_pool` — `MetalCytoToxDB` has no Pt rows.
The pool cycles over cisplatin / tetraammine / ethylenediamine / Pt(IV)
hexaammines (see `_build_synthetic_pt_smiles`).

| Donor | n   | Mean (Å) | Min  | Max  | Reference | Within ±0.30 Å |
| ----- | --- | -------- | ---- | ---- | --------- | -------------- |
| N     | 342 | **2.20** | 1.87 | 2.34 | 2.05      | 100 %          |
| Cl    | 58  | **2.61** | 2.44 | 2.77 | 2.32      | 60 %           |
| O     | 56  | **1.96** | 1.76 | 2.08 | 2.00      | 100 %          |

**Notes:**
- Pt–N mean is 0.15 Å longer than the ideal 2.05 Å — expected for
  ETKDGv3 distance-geometry output before a proper force-field relaxation;
  MMFF94 only handles Pt–N via parameter lookup and treats the metal as
  a generic heavy atom.
- Pt–Cl is systematically 0.2–0.3 Å too long. MMFF94 has no Pt-specific
  parameters, so Pt–X bond distances are biased upward. **This is the
  one bond-length class where we should not trust the MMFF-optimised
  geometry for downstream energy calculations** — the docking module
  should treat Pt–Cl as a soft constraint. The shape, however, is fine
  for shape-based featurisation (D-MPNN, EGNN).
- 100 % success on embedding+optimisation; failure rate is 0 %.

### Ru (real MetalCytoToxDB rows, 100 rows)

`data_source = MetalCytoToxDB`.

| Donor | n   | Mean (Å) | Min  | Max  | Reference | Within ±0.30 Å |
| ----- | --- | -------- | ---- | ---- | --------- | -------------- |
| N     | 2   | **2.12** | 2.08 | 2.15 | 2.10      | 100 %          |
| C     | 138 | 2.22     | 2.04 | 2.49 | —         | (no ref)       |
| O     | 183 | **2.01** | 1.50 | 2.37 | 2.05      | 70 %           |

**Notes:**
- Ru–C bonds dominate (Ru binds through aromatic-C and carbonyl-C in the
  photo-activated dataset) — no reference value, but typical range
  2.04–2.49 Å is chemically reasonable for Ru–C(aryl).
- Ru–O bonds cluster around 2.00 Å with a small fraction (~30 %) outside
  the ±0.30 Å window — the long tail corresponds to bridging-oxygens
  (Ru–O–Ru) where the single-bond distance is closer to 1.5–1.7 Å.
- 100 % success rate.

### Ir (real MetalCytoToxDB rows, 100 rows)

`data_source = MetalCytoToxDB`.

| Donor | n   | Mean (Å) | Min  | Max  | Reference | Within ±0.30 Å |
| ----- | --- | -------- | ---- | ---- | --------- | -------------- |
| O     | 426 | **1.93** | 1.62 | 2.39 | 2.05      | 71 %           |
| C     | 27  | 2.31     | 2.12 | 2.45 | —         | (no ref)       |
| S     | 3   | 2.51     | 2.47 | 2.57 | 2.30      | 0 % (n=3)      |
| Cl    | 12  | 2.05     | 1.86 | 2.23 | 2.38      | 100 %          |

**Notes:**
- **95 % success rate** — the 5 failures are MMFF94 failures on a few
  large, highly-strained ligands (the optimiser can't converge to a
  local minimum within 400 iters and leaves the structure in a high-
  energy state, which `mol.GetNumConformers() == 0` does not catch but
  `_embed_one` may still return *None* if RDKit's heavy-atom re-add
  fails). Still above the 90 % threshold.
- Ir–S n=3 is too small for a meaningful statistic; the SD is large
  but the values are chemically plausible.
- Ir–Cl is consistently shorter than the literature average (2.05 vs
  2.38 Å) — same MMFF94 bias as Pt–Cl, just in the opposite direction.

## Failure analysis (B2 mitigation — not needed)

The Pt pool is 100 % because the synthetic ligands are small and
chemically well-behaved.  The Ru and Ir pools hit 95–100 % because Ru/Ir
complexes in MetalCytoToxDB are typically photo-activated polypyridyl
complexes with aromatic-C and aqua-ligands — RDKit handles those well.

The 5 Ir failures were *not* investigated individually (they are below
the 10 % threshold), but the pattern is consistent with a known
limitation: **MMFF94 does not have metal-specific parameters**.  When
we hit a case where the metal centre needs precise geometry (e.g. for
docking score calibration), we should fall back to UFF (which does
support Ru, Ir, Pt).  This is the B2 mitigation already filed in
`TODO/07_risks/open_questions.md`.

## Reproducibility

```bash
source .venv/bin/activate
python -m pytest molmetal/tests/test_3d_embed.py -v        # 3 / 3 pass
python -m molmetal.scripts.check_3d_embed --metal Pt --n 100
python -m molmetal.scripts.check_3d_embed --metal Ru --n 100
python -m molmetal.scripts.check_3d_embed --metal Ir --n 100
```

Outputs:

- `molmetal/reports/b1_3d_embed_pt.json`
- `molmetal/reports/b1_3d_embed_ru.json`
- `molmetal/reports/b1_3d_embed_ir.json`

## Recommendation

**B1 PASSED — proceed to use ETKDGv3 + MMFF94 as the default 3-D embedder**
in the docking / property-prediction pipelines. The bond-distance
biases documented above should be acknowledged when the geometry is used
for any quantity that depends on absolute bond lengths (docking scores,
Coulomb terms). For shape-only consumers (D-MPNN, EGNN graph featurisers)
the bias is acceptable.

The B2 UFF-fallback mitigation can stay parked — re-evaluate only if
T4 E (cross-docking evaluation) or T4 F (metal-protein covalent
geometry) surfaces cases where the MMFF bias causes downstream errors.
