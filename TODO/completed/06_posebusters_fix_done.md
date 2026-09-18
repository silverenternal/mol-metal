# Fix PoseBusters 0/13 CuAAC pass-rate via MMFF94OptimizeMolecule

**Status:** done (round-8)
**Priority:** med
**Effort:** 1d
**Owner:** (unset)
**Depends on:** none
**Blockers:** none
**Created:** 2026-09-12

## Goal
PoseBusters currently returns 0/13 pass on the CuAAC click-chemistry smoke
set because the raw 1,2,3-triazole geometry produced by ETKDGv3+UFF does
not reach the MMFF94 minimum. Swap the in-pipeline 3D embed step from
ETKDGv3 + UFF fallback to ETKDGv3 + RDKit
`MMFF94OptimizeMolecule(mol, maxIters=200)` before PB evaluation.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py  (the 3D embed pre-pass)
- (also surface this fix in the design-loop generator's 3D path)

## Success criterion
CuAAC 13-molecule smoke set: PB pass rate goes from 0/13 to **≥10/13**.
Generic 50-molecule CrossDocked control set: PB pass rate does not
regress (within 2 pp of current baseline).
`lambda_vs_sbdd_paper_numbers.md` PoseBusters column updated with measured
Lambda pass-rate.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/b1_3d_embed_sanity.md  (failure mode doc)