# Wf-275E — Wf-275 follow-up aggregation synthesis

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-275E ultracode aggregator
**Skill alignment:** `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED) + `mol-metal-paper-writing` §4 cross-ref contract + `mol-metal-cfm-training` (CFM backbone) + `mol-metal-gpu-execution` §6 (5-seed × 50-ep)

---

## §1 Scope

This synthesis aggregates the 4 upstream Wf-275 follow-up reports (Wf-275A P1.4 CPU pilot, Wf-275B DrugOOD ECFP4 adapter, Wf-275C m14 LEAD deployment precondition, Wf-275D paper §4 cross-ref audit + wave tracker) into a single follow-up summary, identifying remaining open issues + the next workflow hook.

## §2 Per-agent aggregate

| Workflow | Goal | Status | Key MEASURED output |
|---|---|:---:|---|
| **Wf-275A** | TODO-119 P1.4 CPU pilot: 4-ablation per-innovation protocol smoke | ✅ COMPLETE (honest-negative) | Synthetic 256-cloud cohort: all 4 configs hit decode_ratio=1.000 at epoch 0 (synthetic ceiling); pilot validates mechanics only, NOT per-lever attribution |
| **Wf-275B** | DrugOOD ECFP4 + scaffold-split adapter (TODO-130 G5) | ✅ COMPLETE (5/5 tests PASS) | n=95,236 rows × 9 cols CSV; 0 scaffold leakage train↔val↔test; 100.3 s single CPU; bridge for downstream Wf-253-D re-eval |
| **Wf-275C** | m14 LEAD deployment precondition formal spec | ✅ SHIPPED | YuelBond decoder MUST stay active (Δ=−1.000 if disabled); W₂ + closure-bounded are NOT preconditions (Δ=0.000 MEASURED); source-value discrepancy with TODO-207 §9.AF flagged |
| **Wf-275D** | paper §4 cross-ref audit + wave budget tracker update | ✅ SHIPPED | 207 §9 → 188 §4: 19/20 consistent (1 minor §9.4 cite-target inconsistency); 188 §4 → 207 §9: 6/8 explicit cite (2 minor implicit gaps); 20/20 entries PASS party-twenty-spirit §3.5 label discipline; 6 new `WF_269_G5_*` constants added to wave_budget_tracker.py |

## §3 Total work completed / failed / open

**Completed (4/4 agents):**
- Wf-275A pilot mechanics validated; flags wired correctly; JSON output schema confirmed.
- Wf-275B DrugOOD ECFP4 adapter + 5 tests PASS; CSV at `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv`.
- Wf-275C m14 LEAD precondition spec formalizes YuelBond load-bearing contract; supersedes scattered partial statements in TODO-188 §4.5.4, TODO-207 §9.AH, §9.AF, §9.6.3.
- Wf-275D cross-ref audit (19/20 + 6/8) + wave_budget_tracker.py constants block.

**Failed / honest-negative (3):**
- Wf-275A honest-negative: synthetic cohort does NOT differentiate ablation levers (4 configs all hit decode_ratio=1.000 ceiling); per-lever attribution requires real-cohort path (Wf-275B-Real-Cohort-CPU).
- Wf-275C source-value discrepancy: TODO-207 §9.AF cites `--no-wasserstein` Δ=−0.013 + `--no-closure-bounded` Δ=−0.008, which DO NOT match Wf-269C MEASURED JSON (Δ=0.000 + PROJECTED Δ=0.000 respectively); requires TODO-207 §9.AF follow-up edit per party-twenty-spirit §3.5.
- Wf-275D minor inconsistencies: 1 cite-target-section number (§9.4 says "§6" but 188 has no §6) + 2 implicit cite gaps (§4.1 m14 should cite 207 §9.6.3; §4.4 m7 should cite 207 §9.6.2); all content-aligned, paper-edit deferred per TODO-84 ACTIVE rule.

**Open issues (5):**
1. Wf-275A → Wf-275B-Real-Cohort-CPU not yet launched: 4 configs × 1 metal (Ru) × 1 seed × 3 epochs on `metalcytotox_5metal_multitask_v4.csv` via `--use-tanimoto-ood-filter` (TODO-98 protocol); ~30 min CPU. If decode_ratio separates (e.g. 0.05-0.30 range), proceed to GPU; otherwise skip pilot and launch 60-run GPU ablation with per-wave user-OK.
2. Wf-275B → Wf-275C+ rerun of Wf-253-D DrugOOD `lbap_refined_ic50_scaffold` ood_test verification with the Wf-275B ECFP4 + scaffold-split CSV; expect AUC ≥ 0.6707 (Wf-185 baseline) within seed-noise. Honest-negative expected if CSV-path AUC also lands at < 0.50 (implies Wf-268B root cause is in LOMO multi-task design, not featurization).
3. Wf-275C → TODO-119 P1.4 cross-cohort generalization: run Wf-269C ablation on PlatinAI 4-cell + DrugOOD + cross-metal cohorts (~5 hr GPU); target: confirm YuelBond load-bearing claim generalizes; secondary: measure Wasserstein + closure-bounded lift on smaller cohorts where Wf-269C saturation may not hold.
4. Wf-275C → TODO-207 §9.AF source-value fix: reconcile §9.AF `--no-wasserstein` Δ=−0.013 + `--no-closure-bounded` Δ=−0.008 with Wf-269C MEASURED Δ=0.000 + PROJECTED Δ=0.000.
5. Wf-275D → paper §4.1 + §4.4 cite-edit follow-up: add explicit 207 §9.6.3 (m14) + 207 §9.6.2 (m7) references; deferred per TODO-84 ACTIVE rule (no `paper/*.tex` writes without authorization).

## §4 Next follow-up: Wf-276 (real-cohort CFM ablation)

**Recommended workflow hook:** launch **Wf-276** combining Wf-275A's recommended real-cohort path + Wf-275C's TODO-119 P1.4 cross-cohort generalization in a single bounded GPU run.

**Scope (proposed):**
- **Wf-276A (CPU insurance):** 4 configs × 1 metal (Ru) × 1 seed × 3 epochs on `metalcytotox_5metal_multitask_v4.csv` via `--use-tanimoto-ood-filter` flag (~30 min CPU). If decode_ratio separates (≥ 0.10 spread between full and at least one ablation), proceed to Wf-276B.
- **Wf-276B (GPU ablation):** 4 configs × 5 metals (Ru/Ir/Os/Rh/Re) × 1 seed × 3 epochs on real cohort (PlatinAI 4-cell + DrugOOD + cross-metal) using production h=128 / n_layers=3 settings (~5 hr GPU per TODO-119 P1.4 re-estimate). Output: per-metal per-lever Δ matrix + cross-cohort YuelBond load-bearing confirmation.

**Decision points (defer to user OK):**
- Skip Wf-276A and proceed directly to Wf-276B (saves 30 min CPU; accepts risk of all-1.0 ceiling).
- Reduce Wf-276B scope to 1 metal × 1 cohort (saves ~4 hr; only confirms YuelBond load-bearing on Ru subset).
- Expand Wf-276B to 5-seed × 50-epoch per TODO-207 §9.AF (7.5 hr; full per-innovation attribution).

**Cross-references:**
- TODO-119 §3 (P1.4 cross-cohort generalization hook)
- TODO-130 G5 (Wf-275B adapter CSV ready for Wf-276B cohort expansion)
- TODO-207 §9.AF (source-value fix pending; Wf-276 honest-positive anchor will overwrite)
- TODO-84 ACTIVE (paper §4.1/§4.4 cite-edits deferred; Wf-276 does NOT touch paper/*.tex)
- Wf-275A pilot_report.md §"Recommendation for Wf-275B (next step)" (path 1 = Wf-276A)
- Wf-275C m14 LEAD deployment precondition §5.3 (TODO-119 P1.4 follow-up spec)
- Wf-275D wave_budget_tracker.py WF_269_G5_RU_SEED0_BEST_VAL_AUC_AT_EPOCH17 constant (calibration anchor for Wf-276B expected per-epoch convergence)

---

**End of synthesis report. Wf-275E ship status: ✅ AGGREGATED. 4/4 upstream reports synthesized; 3 honest-negatives flagged; 5 open issues cataloged; Wf-276 follow-up proposed with 3 decision-point variants.**