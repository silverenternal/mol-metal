# Mol-Metal — Full Wave History

Moved verbatim out of `AGENTS.md` on 2026-10-03 (Wf-A10).

## §0 Why this file exists

`AGENTS.md` had grown to **1,245,399 bytes / 897 lines**, of which the wave
tables below were ~1.20 MB (96%). Because project instructions are injected
into every subagent, that size made **every agent spawn fail** with
`Prompt is too long` (measured: two agents died in ~5 s with 0 tokens each).
No workflow could start, so the repo could not satisfy its own AGENTS.md §5.8
rule that prescribes workflows. The history is preserved here so AGENTS.md
stays small enough for agents to run.

**Nothing was deleted — only relocated.**

> **Compressed 2026-10-06** to one row per wave (what shipped + primary closeout
> file). Full text: `git log -p -- docs/WF_HISTORY.md`, or `TODO/completed/`.

## §1 Wave table — §6 block (through 2026-10-01)

| Period | Wave | Source |
|---|---|---|
| 2026-09-15 | R15 ship | `TODO/completed/` |
| 2026-09-16 | R15 close-out, paper §4 147 cells DESIGN→MEASURED | [`TODO/README.md`](TODO/README.md) |
| 2026-09-28 | Wf-256 ..268 ship | `TODO/completed/` |
| 2026-09-29 | Wf-296 SHIPPED | `327_synthesis.md` |
| 2026-09-29 | Wf-339 IMPROVEMENT_ENUM | `325d_m6_kappa_final.md` |
| 2026-09-29 | Wf-347 | `347a1_wf296_integration.md` |
| 2026-09-29 | Wf-344 | `351c2_agmd_index_sync.md` |
| 2026-09-29 | Wf-353 Wf-340 deeper integration + Wf-343c1 un-wired gates wave | `353c2_index_sync.md` |
| 2026-09-29 | Wf-356 final cleanup v10 | `356c1_final_verify.md` |
| 2026-09-29 | Wf-357 self-audit | `357_final_synthesis.md` |
| 2026-09-29 | Wf-358 | `357a1_preflight_audit.md` |
| 2026-09-29 | Wf-359 Tier-2 metric verification | `359c1_index_sync.md` |
| 2026-09-29 | Wf-360 ultracode residual + retrospectives wave | `360c1_index_sync.md` |
| 2026-09-29 | Wf-362 documentation gaps | `362_final_synthesis.md` |
| 2026-09-29 | Wf-363 GPU wrapper polish | `363_final_synthesis.md` |
| 2026-09-29 | Wf-364 | `364_final_synthesis.md` |
| 2026-09-29 | Wf-365 m6_kappa FINAL verdict | `365_final_synthesis.md` |
| 2026-09-29 | Wf-366 USER_OK gate gaps closure | `363b1_per_pair_tracking.md` |
| 2026-09-29 | Wf-367 pre-flight 2 FAIL fixes | `367_final_synthesis.md` |
| 2026-09-29 | Wf-368 | `368_final_synthesis.md` |
| 2026-09-29 → 2026-09-30 | Wf-369 final consolidation | `369_final_synthesis.md` |
| 2026-09-29 → 2026-09-30 | Wf-370 final consolidation v18 | `370_final_synthesis.md` |
| 2026-09-30 | Wf-371-B1 final cleanup v19 | `371_final_synthesis.md` |
| 2026-09-30 | Wf-373 retry | `373_final_synthesis.md` |
| 2026-09-30 | Wf-375 wetlab deprecation FAQ | `375_final_synthesis.md` |
| 2026-09-30 | Wf-374-B1 | `371a1_m6_kappa_degenerate.md` |
| 2026-09-30 | Wf-372 Zotero | `372_final_synthesis.md` |
| 2026-09-30 | Wf-376-B1 final ship | `375a4_m6_kappa_reframe.md` |
| 2026-09-30 | Wf-377-B1 final ship | `377_final_synthesis.md` |
| 2026-09-30 | Wf-378 OPEN CPU-only gaps closure | `378a1_gpu_ready_user_ok_gate.md` |
| 2026-09-29 → 2026-09-30 | Wf-378-B1 final ship | `378_final_synthesis.md` |
| 2026-09-29 → 2026-09-30 | Wf-379 GPU acceleration | `379_final_synthesis.md` |

| 2026-09-30 | Wf-384 | `384_final_synthesis.md` |

| 2026-09-30 | Wf-381 R20 plan CPU-only closure | `381_final_synthesis.md` |

| 2026-09-30 | Wf-382 TODO-94 Multi-PH | `381_final_synthesis.md` |

| 2026-09-30 | Wf-388 | `388_final_synthesis.md` |

| 2026-09-30 | Wf-398 | `398_final_synthesis.md` |

---

| 2026-09-30 | Wf-423 | `423_final_synthesis.md` |

| 2026-10-01 | Wf-424-A13 Paper §6 v4 ready-to-paste LaTeX (Limitations) sub-flow | `424a13_paper_section6_v4.md` |

| 2026-10-01 | Wf-430-B5 launch Wf-125 P1 CONTROLD PAIR retrain on cuda:0 Blackwell | `430b5_wf125_launch.md` |

| 2026-10-01 | Wf-430-B6 Wf-299 3- | `430b6_wf299_launch.md` |
| 2026-10-01 | Wf-432-A1 Wf-424-A12 wire-in actual fix verification (SCAFFOLD_ONLY →… | `432a1_wirein_fix.md` |

## §2 Wave table — appended block (2026-09-30 .. 2026-10-03)

These rows were appended after AGENTS.md §10.6 and had accumulated there.

| 2026-09-30 | Wf-383 WEAK_TO_STRONG_PLAN §4.3 audit | `383_final_synthesis.md` |

| 2026-09-30 | Wf-380-B1 trainer integration | `380_final_synthesis.md` |

| 2026-09-30 | Wf-385 | `385_final_synthesis.md` |

| 2026-09-30 | Wf-391 | `391_final_synthesis.md` |

| 2026-09-30 | Wf-392 | `392a1_multiph_pll_wire.md` |

| 2026-09-30 | Wf-393 | `393_final_synthesis.md` |

| 2026-09-30 | Wf-394 | `394_final_synthesis.md` |

| 2026-09-30 | Wf-395 | `395_final_synthesis.md` |
| 2026-09-30 | Wf-399 | `38_lit_honest_synthesis.md` |

*Wf-378-B1 docs-only sync row (INDEX §22.73 entry).* `378_final_synthesis.md`, `378a1_gpu_ready_user_ok_gate.md`, `378a2_g6_user_ok_parity.md`

*Wf-380-B1 docs-only sync row (INDEX §22.74 entry).* `380_final_synthesis.md`, `380a1_deep_integration_scaffolds.md`, `380a2_per_cohort_features.md`

*Wf-383-B1 docs-only sync row (INDEX §22.76 entry).* `383_final_synthesis.md`, `383a1_weak_to_strong_se3sb_audit.md`, `383a2_se3_sb_integration_plan.md`

*Wf-381-B1 docs-only sync row (INDEX §22.77 entry).* `381_final_synthesis.md`, `381a4_m6kappa_finalize.md`, `381a5_r20_plan_cpu_close.md`

*Wf-386-B1 docs-only sync row (INDEX §22.79 entry).* `386_final_synthesis.md`, `386a1_wf384_a3_overwrite_audit.md`, `386a2_wf378_a3_wrong_verdict_root_cause.md`

*Wf-385-B1 docs-only sync row (INDEX §22 entry).* `385_final_synthesis.md`, `385a1_wf380_cli_flag_audit.md`, `385a2_wf299_launch_wrapper_review.md`

*Wf-389-B1 docs-only sync row (INDEX §22.5 entry).* `385_final_synthesis.md`, `385b1_index_sync.md`, `389_final_synthesis.md`

*Wf-391-B1 docs-only sync row (INDEX §22.81 entry).* `391_final_synthesis.md`, `391a1_lambda_roadmap_refresh.md`, `391a2_improvement_enum_refresh.md`

*Wf-390-B1 docs-only sync row (INDEX §22.7 entry).* `390_final_synthesis.md`, `390a2_todo179_deeper_audit.md`, `390a3_se3_sb_wire.md`

*Wf-387-B1 docs-only sync row (INDEX §22.77 entry).* `387_final_synthesis.md`, `387a1_wf379_a1_gpu_aggregation_fix.md`, `387a2_balanced_rh_cohort.md`

*Wf-396-B1 docs-only sync row (INDEX §22.89 entry).* `396_decode_ratio_root_cause.md`, `396_final_synthesis.md`, `396a1_decode_ratio_regression_git_bisect.md`

*Wf-398-B1 docs-only sync row (INDEX §22.88 entry).* `398_final_synthesis.md`, `398a1_dmpnn_retrain.md`, `398a2_m2_r2_real.md`

| 2026-09-30 | Wf-407 | `407_final_synthesis.md` |

| 2026-09-30 | Wf-405 | `405_final_synthesis.md` |

| 2026-09-30 | Wf-408 | `408_final_synthesis.md` |

| 2026-09-30 | Wf-410 | `410_final_synthesis.md` |

| 2026-09-30 | Wf-411 | `411_final_synthesis.md` |

| 2026-09-30 | Wf-409 | `409_final_synthesis.md` |

| 2026-09-30 | Wf-413 | `413_final_synthesis.md` |

| 2026-09-30 | Wf-415 | `415_final_synthesis.md` |
| 2026-09-30 | Wf-416 | `125_controlled_pair_wrapper.md` |
| 2026-09-30 | Wf-418 | `418_final_synthesis.md` |
| 2026-09-30 | Wf-417 | `417_final_synthesis.md` |
| 2026-09-30 | Wf-419 | `419_final_synthesis.md` |
| 2026-09-30 | Wf-421 | `421_final_synthesis.md` |

| 2026-10-01 | Wf-422 | `422_final_synthesis.md` |

| 2026-10-01 | Wf-420 | `TODO/completed/` |

| 2026-10-01 | Wf-424-A4 atomic migration completion | `424_final_synthesis.md` |

| 2026-10-01 | Wf-424-A14 paper §3 v4 FINAL metric table update (Main Results) | `424a14_paper_section3_v4.md` |
| 2026-10-01 | Wf-424-A15 m6_kappa drop-promote paper-conversion | `424a15_m6_kappa_drop.md` |
| 2026-10-01 | Wf-424-A3 scaffold_conditioning shape mismatch fix | `424a3_scaffold_conditioning_fix.md` |
| 2026-10-01 | Wf-424-B1 | `424_final_synthesis.md` |

| 2026-10-01 | Wf-424-A1 Wf-420-A1 use_dann kwarg fix | `424a1_use_dann_fix.md` |

| 2026-10-01 | Wf-424-A7 Wf-380 SCAFFOLD-loss MEASURED retrain on cuda:0 Blackwell | `424a7_wf380_scaffold_retrain.md` |

| 2026-10-01 | Wf-424-A10 Wf-388 Wf-125 P1 CONTROLD PAIR retrain attempt on cuda:0 B… | `424a10_wf388_wf125_controlled_pair.md` |

| 2026-10-01 | Wf-424-A11 Wf-390-A1 m2 R² Path A 5-metal subset retrain on cuda:0 Bl… | `424a11_wf390_m2_r2_path_a.md` |

| 2026-10-01 | Wf-426-A2 wrapper infrastructure fix | `426a2_wrapper_infra.md` |

| 2026-10-01 | Wf-426-A3 post-fix retrain readiness infrastructure | `426a3_post_fix_retrain.md` |
| 2026-10-01 | Wf-426-B1 docs-only INDEX | `TODO/completed/` |
| 2026-10-01 | Wf-427-A4 B5 Wf-125 schema adapter | `427a4_b5_schema_adapter.md` |
| 2026-10-01 | Wf-427-B1 | `TODO/completed/` |
| 2026-10-01 | Wf-428-B1 docs sync | `425a3_kcal_sign_investigation.md` |
| 2026-10-01 | Wf-429-A1 Wf-426-A3 launcher line 137 path bug fix | `429a1_wf299_launcher_fix.md` |

| 2026-10-01 | Wf-429-C1 | `TODO/completed/` |

| 2026-10-01 | Wf-430-C1 docs sync | `TODO/completed/` |
| 2026-10-01 | Wf-431-B1 GPU utilization optimization wire-in | `431b1_gpu_utilization_fix.md` |

| 2026-10-01 | Wf-431 ultracode CPU-only diagnose + GPU utilization fix wave | see `TODO/completed/` |

| 2026-10-02 | Wf-432-A1 PRE_TRAINING_GATE_v2 wire-in | `432a1_pre_training_gate_v2.md` |

| 2026-10-02 | Wf-431-A2 Wf-3 8 d-block Vina sweep wrapper authoring | `TODO/completed/` |

| 2026-10-02 | Wf-A3 OmegaFold-gap-fill PHASE 2 integration | `443a3_omegafold_integration.md` |

| 2026-10-02 | Wf-444-A3 docs+code+index integration sync | see `TODO/completed/` |

| 2026-10-02 | Wf-424-A12 D-MPNN Wf-380 NEW CLI flag wire-in (args NameError fix) | `424a12_wirein_fix.md` |

| 2026-10-02 | Wf-201 | `201_m2_r2_per_cohort_wired.md` |

| 2026-10-02 | Wf-432-A2 TODO-202 m7 diversity | `202_m7_diversity_wired.md` |

| 2026-10-02 | Wf-A4 §5.7 50-step CPU smoke | `A4_wirein_verify.md` |
| 2026-10-02 | Wf-A4-final-sync non-doc task execution: Audit | `200_wf340_353_deep_integration.md` |
| 2026-10-03 | Wf-A5 follow-up N4 | `N4_resource_scaling.md` |
| 2026-10-03 | Wf-A5 follow-up N3 | `N3_checkpoint_resume.md` |

| 2026-10-03 | Wf-A5 follow-up N4 | `N4_resource_scaling.md` |

| 2026-10-03 | Wf-TUI runtime event emission | `TODO/completed/` |

| 2026-10-03 | Wf-A6 BLOCKER A | `A1_dmpnn_dynamic_metals.md` |

| 2026-10-03 | Wf-TUI P0 dashboard rewrite (this WFTUI invocation) | `WFTUI_tui_rewrite.md` |

| 2026-10-03 | Wf-A7 TUI deep integration closeout (CPU-only docs | `A7_tui_deep_integration_closeout.md` |

| 2026-10-03 | Wf-A6 A2 BLOCKER C fix | `A2_lr_scheduler_wired.md` |

| 2026-10-03 | Wf-A8 startup failure fix | `A8_startup_fix_closeout.md` |

| 2026-10-03 | Wf-A8 BLOCKER E + F fix | `A4_closeout.md` |

| 2026-10-03 | Wf-A8 PlatinAI 4-cell CSV loader | `A8_platinai_loader.md` |

| 2026-10-03 | Wf-A6a5 R5 + R6 wire-in verification | `A6a5_r5_r6_verify.md` |

| 2026-10-03 | Wf-A8 startup fix + 2 workflows resume + git sync | see `TODO/completed/` |

| 2026-10-03 | Wf-A8 YAML-unify pattern applied to 12 OTHER yaml files | `A8_other_yamls_unify.md` |

| 2026-10-03 | Wf-A8 startup failure fix | `A8_other_yamls_unify.md` |

| 2026-10-03 | Wf-A8c final verify | `A8c_yaml_3fix_closeout.md` |
