# Wire 204-tile ChEMBL/ZINC pool into MCTS proof search (L-3 PARTIAL → DONE)

**Status:** pending
**Priority:** high
**Effort:** 2d
**Owner:** (unset)
**Depends on:** 01 + 02
**Blockers:** none
**Created:** 2026-09-12

## Goal
Wire the validated 204-tile ChEMBL/ZINC reactive-handle fragment pool into
the MCTS proof-search expansion step. Currently the pool is validated and
ready (`lambda_round3_L3_fragment_library.md`) but expansion still uses the
hardcoded 12-tile library. Wire-up is explicitly gated on the L-1 binding
oracle being live (now done — see TODO/completed/06_round3_axes.md).

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py  (expansion calls / source / SMILES keys)
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tile_lib/click_tiles.py  (consume new pool)

## Success criterion
MCTS branching factor increases from 60 (5×12) to **1020 (5×204)**; embedding
failures stay below 5% (validated 1.9%); `lambda_round3_combined_report.md`
L-3 moves from PARTIAL to DONE; new test asserts branching_count=1020 and
embed_fail_rate<0.05.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L3_fragment_library.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md