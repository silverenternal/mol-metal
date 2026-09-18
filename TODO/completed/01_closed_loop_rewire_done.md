# Rewire closed_loop.py default reward to Vina+PoseBusters+AiZynth

**Status:** done (round-8)
**Priority:** high
**Effort:** 1d
**Owner:** (unset)
**Depends on:** none — all 3 adapters exist as MVP (see TODO/completed/02,03,04)
**Blockers:** none
**Created:** 2026-09-12

## Goal
Swap the placeholder reward in `closed_loop.py` so the default scaffold hopping
loop uses real Vina docking score + PoseBusters geometry checks + AiZynth
retrosynthesis availability as a weighted composite reward (the
`RewardAggregator` already exposes all three channels via
`r_vina` / `r_posebusters` / `r_retro` from proof_search.py:519-587).

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/orchestration/closed_loop.py

## Success criterion
`closed_loop.py --demo` produces reward values that decompose into
Vina / PoseBusters / AiZynth components; top-5 molecules by reward pass
PoseBusters at ≥80% rate on the 12-tile internal check (matches L7/L8
governance baseline SANITY=1.000, EMBED_SUCCESS=24/24).

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/mmp13_vina_real.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/h3_retrosynthesis_check.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/design_loop_phase0.md