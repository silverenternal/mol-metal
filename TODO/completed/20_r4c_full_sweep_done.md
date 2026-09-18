# Run R4-C 100-pocket CrossDocked sweep on MMP13

**Status:** pending
**Priority:** high
**Effort:** 2-3d
**Owner:** (unset)
**Depends on:** 04 (QVina swap) + 05 (REINVENT4 install)
**Blockers:** CrossDocked2020 subset pre-staged on RX 7800 XT scratch disk
**Created:** 2026-09-12

## Goal
Run the headline R4-C sweep: 100 CrossDocked pockets against MMP13 with
the full closed-loop pipeline (RewardAggregator reward + MCTS proof
search + QVina dock + REINVENT4 prior). Replaces the cite-only Lambda
Vina column in `lambda_vs_sbdd_paper_numbers.md` with measured numbers.

## Pilot (r4_c_pilot.md, n=1 pocket)
- sa_mean = 8.276, qed_mean = 0.390, lipinski_pass = 1.000, wall = 1.68 s
- Pipeline runs end-to-end on a single CrossDocked pocket

## File(s) to edit
- new: /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/scripts/r4_c_full_sweep.py
- generated: per-pocket results CSV

## Success criterion
Sweep completes 100 pockets within **a reasonable wall-clock budget** (to
be set based on the pilot's 1.68 s scaling — order ~3-5 hours on
RX 7800 XT, not 72 hours). Report CSV contains top-10 Vina score per
pocket and is reproducible from the script alone. Lambda's measured mean
Vina column lands in `lambda_vs_sbdd_paper_numbers.md` next to the cited
SOTA numbers.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_c_pilot.md  (n=1 pilot)
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r4_c_test.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_vs_sbdd_paper_numbers.md