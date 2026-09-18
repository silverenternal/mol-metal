# QVina / QuickVina2 swap behind vina_adapter.py flag

**Status:** pending
**Priority:** high
**Effort:** 0.5d
**Owner:** (unset)
**Depends on:** none
**Blockers:** qvina / quickvina2 binary install (apt or build from source)
**Created:** 2026-09-12

## Goal
Add a `--engine {vina|qvina|quickvina2}` flag to `vina_adapter.py` so the
docking backend can be swapped without changing call sites. QVina / QuickVina2
are the engines used by published SBDD baselines (Pocket2Mol, TargetDiff,
DiffSBDD); Vina 1.2.7 (current) produces comparable scores but at different
convergence, so any Lambda-vs-SOTA number has a systemic engine-mismatch
bias until this swap lands.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/vina_adapter.py
- (no new binaries needed in repo — qvina/quickvina2 install is host-level)

## Success criterion
`vina_adapter.py --engine quickvina2 --pocket MMP13 --ligand X` re-runs the
mmp13_vina_real.md experiment and produces a mean within ±0.3 kcal/mol of
the original Vina 1.2.7 mean (-1.355); `lambda_vs_sbdd_protocol_aligned.md`
P1 action item is marked DONE.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_protocol_aligned.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp13_vina_real.md