# Wf-24 Phase 1 — Pending Cleanup Manifest

**Date:** 2026-09-26
**Agent:** Claude Code (ultracode)
**Workflow:** Wf-24 Phase 1 (move 16 SUPERSEDED `TODO/pending/` files → `TODO/04_history/`)
**Predecessor:** Wf-8 + Wf-11 (added the `> **STATUS: SUPERSEDED ...**` headers)

## Summary

All 16 files under `TODO/pending/` carried a `STATUS: SUPERSEDED 2026-09-25/26` header added by
Wf-8 / Wf-11, indicating the underlying plan had been resolved by R17 close-out
(commit `4561625`) or one of the 9 completed close-out workflows (Wf-4 … Wf-7). Per
Wf-24 Phase 1, this phase:

1. Verified each pending file already had a matching `_superseded_2026-09-26.md`
   copy under `TODO/04_history/` (sizes match the pending originals byte-for-byte —
   copies done by Wf-8 / Wf-11 ahead of the file delete).
2. Removed the 16 originals from `TODO/pending/` (`git rm -f` for 10 git-tracked
   files; plain `rm` for 6 untracked).
3. Left `TODO/04_history/` with the 16 superseded copies + the 8 pre-existing
   ship artefacts (24 entries total).

`TODO/pending/` is now empty (the directory is recreated and ready to receive
post-R18 ACTIVE plans).

## Files moved (16)

| # | Pending basename | History copy | Supersession source |
|---|------------------|--------------|---------------------|
| 1 | `14_full_100pocket_paper_r13.md` | `14_full_100pocket_paper_r13_superseded_2026-09-26.md` | R17 close-out (`4561625`) + Wf-5 weak→strong + Wf-7 Tier-2 |
| 2 | `19_user_decisions.md` | `19_user_decisions_superseded_2026-09-25.md` | R17 close-out (wt8 Reality Check, D6/D7/test_005 all shipped) |
| 3 | `21_lambda_model_coupling.md` | `21_lambda_model_coupling_superseded_2026-09-26.md` | Wf-4 close-out — `TODO/completed/57_lambda_cfm_coupling.md` |
| 4 | `22_data_gap_alignment_plan.md` | `22_data_gap_alignment_plan_superseded_2026-09-26.md` | R17 close-out (commit `4561625`) + paper §4.5 + §6 ship |
| 5 | `23_weak_to_strong_plan.md` | `23_weak_to_strong_plan_superseded_2026-09-26.md` | Wf-5 — `TODO/completed/58_weak_to_strong.md` + `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` |
| 6 | `24_cfm_architecture_redo_plan.md` | `24_cfm_architecture_redo_plan_superseded_2026-09-26.md` | Wf-6 — `TODO/completed/59_cfm_m3_m5_frontier.md` |
| 7 | `25_round14_lit_grounded_plan.md` | `25_round14_lit_grounded_plan_superseded_2026-09-26.md` | Wf-6 — paper §3.4 (24-paper frontier survey) ship |
| 8 | `26_round13_round14_complete_plan.md` | `26_round13_round14_complete_plan_superseded_2026-09-26.md` | R17 close-out + Wf-5 + Wf-7 |
| 9 | `28_round12_honest_negative_reframe.md` | `28_round12_honest_negative_reframe_superseded_2026-09-26.md` | R17 paper §4.5 + §6 — `paper/sections/04_5_honest_negative.tex` (731 lines) |
| 10 | `30_pitfall_reinforce_plan.md` | `30_pitfall_reinforce_plan_superseded_2026-09-26.md` | Wf-7 — `TODO/completed/60_pitfall_tier2.md` (Tier-1 5/5 + Tier-2 3/3) |
| 11 | `31_r17_ultracode_dispatch.md` | `31_r17_ultracode_dispatch_superseded_2026-09-26.md` | R17 close-out + 9 completed workflows (L/M/C tracks all SHIP) |
| 12 | `31_ultracode_trackA_flow_matching.md` | `31_ultracode_trackA_flow_matching_superseded_2026-09-26.md` | Track A A1–A4 SHIP — `TODO/completed/37_model_track_mr17.md` |
| 13 | `32_ultracode_trackB_lambda_calculus.md` | `32_ultracode_trackB_lambda_calculus_superseded_2026-09-26.md` | Track B B0–B5 SHIP — `TODO/completed/32_trackB_mlc_pll_wireup_done.md` + `32_late_phase.md` |
| 14 | `33_r17_fixes_blackwell_loader_paths.md` | `33_r17_fixes_blackwell_loader_paths_superseded_2026-09-26.md` | All 7 P1–P7 blockers SHIP — `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` |
| 15 | `34_parallel_ultracode_dispatch.md` | `34_parallel_ultracode_dispatch_superseded_2026-09-26.md` | All 6 UC-A/B/L/M/C/F tracks SHIP |
| 16 | `53_phase_a_reality_check.md` | `53_phase_a_reality_check_superseded_2026-09-26.md` | Wf-8 — `TODO/completed/53_phase_a_reality_check.md` + R18 SE(3) Schrödinger Bridge pivot |

## Completed reports referenced

- R17 final close-out: `TODO/completed/41_r17_final_closeout.md`
- R17 model track: `TODO/completed/37_model_track_mr17.md`
- Lambda × CFM coupling (Wf-4): `TODO/completed/57_lambda_cfm_coupling.md`
- Weak → strong audit (Wf-5): `TODO/completed/58_weak_to_strong.md`
- CFM M3/M5 frontier (Wf-6): `TODO/completed/59_cfm_m3_m5_frontier.md`
- Pitfall Tier-2 (Wf-7): `TODO/completed/60_pitfall_tier2.md`
- Phase-A Reality Check (Wf-8): `TODO/completed/53_phase_a_reality_check.md`
- Track B PLL wireup: `TODO/completed/32_trackB_mlc_pll_wireup_done.md`
- Track B B1 metal atoms 30: `TODO/completed/32_trackB_uc_b1_metal_atoms_30.md`
- Track B late phase: `TODO/completed/32_late_phase.md`
- Blackwell fixes: `TODO/completed/33_r17_fixes_blackwell_loader_paths.md`
- Zenodo wrapper: `TODO/completed/34_p7_zenodo_wrapper_done.md`
- PlatinAI 5K pretrain: `TODO/completed/35_platinai5k_coupling_pretrain.md`
- ASTSR swap (L-R17-B3): `TODO/completed/35_l_r17_b3_astsr_swap_done.md`
- Lambda phase-4 sweep: `TODO/completed/34_lambda_phase4_sweep.md`
- Lipman FM adapter: `TODO/completed/01_lipman_fm_adapter.md`
- Round-12 honest negatives: `TODO/completed/38_round12_honest_negatives.md`
- Round-14 complete plan: `TODO/completed/38_round14_complete_plan.md`

## Constraints honoured

- Only files with explicit `> **STATUS: SUPERSEDED ...**` header were moved (0 ACTIVE in `pending/`).
- All copies use the unified `_superseded_2026-09-26.md` suffix (19_user_decisions keeps the `2026-09-25` suffix from its original Wf-8 marker).
- No `.gitignore` rglobs were touched.
- `TODO/04_history/` is left untracked (consistent with `TODO/01_strategy/`, `TODO/02_metrics/`, `TODO/03_plans/`).

## Outcome

- `TODO/pending/` is now empty (directory preserved).
- `TODO/04_history/` holds 16 superseded files + 8 pre-existing ship artefacts (24 entries).
- Future ACTIVE TODO files (R18 Phase 1-3 follow-up) belong in `TODO/pending/` or `TODO/03_plans/`, not `TODO/04_history/`.

— Claude Code, Wf-24 Phase 1, 2026-09-26
