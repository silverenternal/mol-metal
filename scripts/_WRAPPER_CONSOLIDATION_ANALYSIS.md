# Wrapper Consolidation Analysis — Wf-A2

**Last updated:** 2026-10-02 (Wf-A2 wrapper test coverage analysis)

**Scope:** 23 =================================================================
 USER OK REQUIRED — kill_then_adapt standing directive (207 §1.1)
=================================================================

This script launches the TODO-119 P1.4 chemprop v3 LOMO Wave-A:
  Wave-A: 4 ablations × 1 seed × 3 ep smoke — ~2 hr on PRO 6000 cuda:0

Before running this script, the user MUST explicitly grant OK per
207 §1.1 (user 'kill then adapt' deferral pattern).

To grant OK and launch:
  USER_OK=1 bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh

To skip an ablation (cheaper smoke):
  USER_OK=1 SKIP_ABLATIONS=bond,closure bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh

To verify the launcher shape on CPU first:
  bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh --dry-run

ABORTING — no GPU work executed. wrappers (per ).

## Summary

| Trainers | Wrappers | Candidate for consolidation? |
|---|---|---|
|  | 4 (wf299 + wf310f + wf380 + wf394a1) | YES — single wrapper with --wave flag |
|  | 4 (wf310e_r20_p5 + wf388_a1 + wf390a1 + wf409a3) | YES — single wrapper with --metals + --coord-term flags |
|  | 3 (wf125_p1 + wf131_p3_deferred + wf131_p3_pt_4cells) | YES — single wrapper with --cohort + --metal-list flags |
|  | 1 | NO — standalone, different trainer () |
|  | 1 | NO — standalone |
|  | 1 | NO — standalone legacy trainer |
|  | 1 | NO — standalone |
| Ungrouped (g5/g6 G6G7 CFM residual) | 8 | NO — multi-purpose G5/G6 sweep scripts |

## Top 3 consolidation candidates (estimated LOC savings)

1. ** × 4 → 1 unified wrapper**
   - Current: wf299 (701 LOC) + wf310f (329 LOC) + wf380 (252 LOC) + wf394a1 (340 LOC) = **1,622 LOC**
   - Unified: single wrapper with  + cohort flag (~600 LOC)
   - **Savings: ~1,022 LOC (~63%)**
2. ** × 4 → 1 unified wrapper**
   - Current: wf310e_r20_p5 (473 LOC) + wf388_a1 (384 LOC) + wf390a1 (429 LOC) + wf409a3 (287 LOC) = **1,573 LOC**
   - Unified: single wrapper with  +  +  flags (~550 LOC)
   - **Savings: ~1,023 LOC (~65%)**
3. ** × 3 → 1 unified wrapper**
   - Current: wf125_p1 (405 LOC) + wf131_p3_deferred (322 LOC) + wf131_p3_pt_4cells (510 LOC) = **1,237 LOC**
   - Unified: single wrapper with  +  flags (~450 LOC)
   - **Savings: ~787 LOC (~64%)**

**Total estimated savings: ~2,832 LOC (~63% reduction)** across 11 → 3 consolidated wrappers.

## Consolidation constraints

- Each consolidated wrapper MUST preserve all 7 canonical markers (USER_OK + DRY_RUN + plan + pre_flight + PYTHONPATH + PY + --user-ok propagation) per Wf-A2 audit.
- Bit-for-bit backward compat with Wf-131 P1 + Wf-380 + Wf-299 wave architecture (Wf-415-A1 transparent-passthrough + Wf-408-A3 --wave 1/2/3 alias must work).
- USER_OK gate per Wf-294B + Wf-405-A4 + Wf-366-A1 canonical surface preserved.
- Pre-flight v3 PASS required per Wf-415-A3 + Wf-427-B1 cascade closures.

## Out of scope

- 8 ungrouped wrappers (launch_wf124_p3_g5/g6, launch_wf310b_g6_final, launch_wf310e_wf219_production_gpu, launch_wf310e_wf135_phase2_gpu, launch_wf308c_todo220_gpu, launch_wf333_vina_4step_gpu_bench, launch_wf119_p1_4_chemprop_v3_lomo, launch_wf394_lambda_cfm, launch_wf310f_frontier_math_gpu) — different trainer targets, low overlap; standalone.
- SCAFFOLD-tier wrappers (15 of 23) — these are incomplete per Wf-A2 audit and need first-pass SCAFFOLD → REAL conversion before consolidation is meaningful.

## Forward-looking recommendation

Consolidation deferred to Wf-450+ USER_OK-gated wave; estimated wall budget ~5 hr CPU + ~2 hr GPU integration test per wrapper group + ~30 min regression test re-run = ~22 hr total. Closure path = Wf-450-A1 CFM consolidation (~5 hr CPU + ~2 hr GPU) + Wf-450-A2 D-MPNN consolidation (~5 hr CPU + ~2 hr GPU) + Wf-450-A3 LOMO consolidation (~4 hr CPU + ~2 hr GPU) + Wf-450-A4 regression test re-run (~30 min) + Wf-450-A5 final docs sync (~5 min CPU).
