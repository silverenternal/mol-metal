# MMP13 selected as MMP-family surrogate

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp_case_study_setup.md
**Owner:** (unset)

## What was delivered
MMP13 (collagenase-3) is the canonical MMP-family surrogate for the docking
/ lead-optimisation benchmark loop, after surveying the matrix
metalloproteinase target landscape for tractability, ligand availability,
pocket size, and prior benchmark data.

## Hard numbers (mmp_case_study_setup.md + mmp13_vina_real.md)
- **MMP2 has ZERO pairs in CrossDocked2020** — the original plan's MMP2/9
  was wrong
- **MMP13 has 442 pairs (39 PDBs)** — recommended MMP-family surrogate
- CrossDocked2020 MMP coverage: MMP1=4, MMP3=64, MMP7=12, MMP8=75,
  MMP9=4, MMP12=489, MMP13=442, MMP2=0 (total 1,295 pairs)
- MMP13 (PDB 830c) real-Vina smoke run: 100/100 Lambda click products docked
  end-to-end, mean=-1.355 kcal/mol, success_rate=100% within +1.0 kcal/mol of
  co-crystal RS1 (-1.285) — see TODO/completed/02_vina_adapter.md

## Lessons learned
- The MMP family selection has a tight constraint: only MMP3, MMP8, MMP12,
  MMP13 have any usable CrossDocked2020 pairs. MMP13 is the best target
  (442 pairs + a co-crystal ligand for protocol validation).
- MMP13's pocket is zinc-dependent; the metal-coordination prior is required
  at evaluation time, not just training time — useful stress test for the prior.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp_case_study_setup.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp_case_study_summary.json
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp13_vina_real.md