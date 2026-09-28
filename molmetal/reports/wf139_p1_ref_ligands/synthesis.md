# Wf-139 Phase 1 — Crystal ligand SMILES extraction synthesis

**Date**: 2026-09-26T21:32:23.762299+00:00
**Status**: PASS
**Pockets extracted OK**: 6/6
**Pockets failed**: 0/6

## Inputs (per TODO-134 §1.3)

Per `molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt` (5/5 READY):

| Pocket | Source | Extraction kind |
|---|---|---|
| `830c` | `830c.pdb` | pdb_hetatm |
| `1h36` | `1h36_A_rec_1h36_r88_lig_tt_min_0.sdf` | sdf |
| `3eml` | `3eml_A_rec_3eml_zma_lig_tt_min_0.sdf` | sdf |
| `3ny8` | `3ny8_A_rec_3d4s_tim_lig_tt_min_0.sdf` | sdf |
| `4aaw` | `4aaw_A_rec_4ac3_r83_lig_tt_min_0.sdf` | sdf |
| `4rlu` | `4rlu_ligand.sdf` | sdf |

## Per-pocket result

| Pocket | OK | Heavy atoms | Formula | InChIKey | Notes |
|---|:---:|---:|---|---|---|
| `830c` | OK | 28 | `C19H36ClNO6S` | `NCYFNIIUPHKVCJ-UHFFFAOYSA-N` | MMP-13 hydroxamate (sulphone hydroxamate RS1) — co-crystal from mmp13_real/830c. |
| `1h36` | OK | 25 | `C21H22BrNO2` | `JAZMZJDLZUDIDG-NSCUHMNNSA-N` | SQHC_ALIAD R88 (squalene-hopene cyclase) CrossDocked SDF. |
| `3eml` | OK | 25 | `C16H15N7O2` | `PWTBZOIUWZOPFT-UHFFFAOYSA-N` | AA2AR_HUMAN ZMA (ZM241385, A2A antagonist) CrossDocked SDF. Note: CrossDocked sh |
| `3ny8` | OK | 21 | `C13H24N4O3S` | `BLJRIMJGRPQVNF-JTQLQIEISA-N` | ADRB2_HUMAN TIM (β2 adrenergic receptor ligand) CrossDocked SDF. |
| `4aaw` | OK | 31 | `C22H29N3O5S` | `LDJDHSINBFNMFS-UHFFFAOYSA-N` | GLMU_STRPN R83 (glutamate mutarotase) CrossDocked SDF. |
| `4rlu` | OK | 19 | `C15H12O4` | `DXDRHHKMWQZJHT-FPYGCLRLSA-N` | Trypsin HCC inhibitor from pdbbindpp-2020. SDF ships explicit Hs (V2000) which m |

## Verdict

- **6/6 SMILES extracted + canonical + valid**: YES
- **InChIKey consistency**: per-ligand unique InChIKey; no InChIKey collisions across pockets
- **Heavy atom count plausibility**: 19-56 heavy atoms (drug-like range)

## Decision

Phase 1 PASS → Wf-139 Phase 2 (Vina redocking benchmark) may launch next.
