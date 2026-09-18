# WF-T26 — Paper §4 + §5 + §6 + §7 update with R15 + EV-3 + metallodrug data

**Date:** 2026-09-17
**Status:** SHIPPED (CPU-only, additive only, all cross-refs preserved)

## 1. Summary

5 additive edits to `paper/sections/04_evaluation.tex`, `05_ablation.tex`, `06_limitations.tex`, `07_future.tex` ship today's R15 + EV-3 + metallodrug + TODO-30 measurements. No existing content deleted; all cross-refs preserved; honest framing unchanged.

## 2. Edits shipped

**2.1 §4.5 diversity column — NEW EV-3 paragraph.** `sa_mean` = **3.0988 ± 0.0000** on `test_001` × 3 seeds at `--sa-weight 0.3` (362 s wall, CPU-only); `diversity_tanimoto` = **0.1366** (+28.3% vs R12 Path A baseline 0.1065); `diversity_homotype` = **0.045**. Honest framing: lift is basket-internal; pocket-invariance attractor (`pocket_invariance_pairwise_jaccard = 1.0`) remains; SA channel saturated on singleton Pt-trizole aryl family; TargetDiff range [2.65, 2.86] not reached (gap +0.239). Source: `molmetal/reports/wf_lift_ev3/final.md`.

**2.2 §4.6 PB column — NEW EV-2 honest INCOMPLETE update.** 15-cell production smoke at `n_sim=1000` honest **INCOMPLETE** — driver aborted at cell 3 (`Runtime code changed during experiment`); 2/15 cells measured (test_001 seed=42 + seed=0). PB adapter / Vina dispatcher inconsistency (0/21 vs 21/21) — data-pipeline issue. 0 cells promoted DESIGN→MEASURED. 2 follow-ups: (i) patch `runtime_fingerprints()` at `r4_c_full_sweep.py:152-161`; (ii) fix PB adapter to receive docked pose. Source: `molmetal/reports/wf_lift_ev2/final.md`.

**2.3 §5.8 P0 panel — NEW EV-3 + 4 Pt-specific proxies.** (1) `sa_mean` = 3.0988 (EV-3); (2) `diversity_tanimoto` = 0.1366 / `diversity_homotype` = 0.045 (EV-3, +28.3% vs R12); (3) 4 new Pt-specific proxies from `wf_t25_metallo_proxies`: `monodentate_cl_count`, `oxidation_state_distribution`, `coordination_number_mean`, `gs_ligand_evasion_proxy`; (4) `metal_compliance_truthful` (Fix-3 split) replaces legacy `metal_compliance_rate` as headline column.

**2.4 §6 limitations — NEW item (TODO-14 BUG-1 + BUG-2 closed).** BUG-1: `coupling_adapter` reshape 64→5 silent-fail patched (now raises ValueError on shape mismatch). BUG-2: `pocket_macro_inference` CWD-relative path made absolute. 15/15 tests pass on the regression triplet (test_coupling_adapter_bug1 + test_pocket_macro_bug2 + 3 pt-known tests); 4.7s CPU; part of `pytest full` smoke. CFM `decode_ratio=0` GPU-blocked still deferred to R16. 0 cells promoted DESIGN→MEASURED (engineering-bug close-out, not data-cell promotion).

**2.5 §7 future work — NEW TODO-30 Tier-1 + Tier-2 status.** **Tier-1 (5 items, all SHIPPED):** pareto.postprocess, metallodrug_vertical_coverage, D-MPNN baseline + LightGBM CLI + LOMO-CV + temporal holdout, 4 new Pt-specific proxies. **Tier-2 CPU (4 items, all SHIPPED):** P6.1+P6.2 reproducibility plumbing, P2.2 materialize_3d default flip. **Remaining Tier-2 DEFER (3 items, all GPU or new-deps blocked):** P1.2 FG-compat veto, P2.3 layered generation, P6.3 MMseqs2 fold-aware scaffold split.

## 3. Constraints respected

- [x] **CPU-only:** no GPU retrain executed; all edits are LaTeX prose only
- [x] **Additive only:** no existing content deleted; all cross-refs and label/ref keys preserved
- [x] **Honest framing:** EV-3 lift labelled basket-internal; EV-2 result labelled INCOMPLETE; BUG-1+BUG-2 close-out labelled engineering-only; TODO-30 defer items labelled GPU/new-deps blocked
- [x] **No silent promotions:** 0 cells promoted DESIGN→MEASURED across all 5 edits

## 4. Section line counts (post-edit)

- §4: 2698 lines (+73 vs 2625, +2.8%)
- §5: 1320 lines (+71 vs 1249, +5.7%)
- §6: 563 lines (+11 vs 552, +2.0%)
- §7: 428 lines (+64 vs 364, +17.6%)

## 5. Recommended next step

Re-run the §4.5 EV-3 measurement on the R12 Path A 10×3 + AlgoTune 3-cell panel (33 cells) once the structural pocket-invariance fix ships (Fix 2(a) MetalLigandExchange + Fix 1 soft-tiered prior, or `pocket_features.modify_root_prior` integration into `MCTSProofSearch`).

## 6. File paths

- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex`
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex`
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/06_limitations.tex`
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/07_future.tex`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t26_paper_update/final.md` (this file)
- Sources: `molmetal/reports/wf_lift_ev3/final.md`, `wf_lift_ev2/final.md`, `wf_t25_metallo_proxies/final.md`, `wf_t30_tier1_master/`, `wf_t30_tier2_cpu_master/`, `wf_metallodrug_vertical/MASTER.md`
