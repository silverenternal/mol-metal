# INDEX — TODO/03_plans/

**Last updated:** 2026-10-02 (per Wf-381-A4 + Wf-382-A4 + Wf-393-A1..A3 + Wf-410-A3 + Wf-391-A2 + Wf-391-A3 + Wf-393-A2)

Per-topic strategic plans — one line per R* plan + cross-link to R* file. Replaces ad-hoc READMEs scattered across `TODO/03_plans/`.

## R17 — CFM + multi-task anti-cancer metric suite

- [`R17_AUC_RECOVERY_PLAN.md`](R17_AUC_RECOVERY_PLAN.md) (~25k LOC) — Multi-task D-MPNN 5-metal retrain; m1_auc recovery ≥ 0.83; closed via Wf-131 P3 + Wf-379-A1 + Wf-413-A1; canonical 7/7 (metal, seed) aggregate `metrics/by_metric/metal_lomo_auc_wasserstein.json`

## R18 — Mathematical theorem integration

- [`R18_MATH_MIGRATION_PLAN.md`](R18_MATH_MIGRATION_PLAN.md) (~28k LOC) — Schrödinger Bridge + Transport Relaxation + Monge-Ampère + W₂ gradient flow integration; in-flight via Wf-383-A3
- [`R18_PLAN.md`](R18_PLAN.md) (~37k LOC) — R18 master plan; references Wf-379-A1 + Wf-380 + Wf-394

## R19 — Research extensions

- [`R19_PLAN.md`](R19_PLAN.md) (~67k LOC) — R19 master plan (Pre-Training P0/P1 + per-baseline protocol validation); closed via Wf-421-B1 + Wf-432 §12.5

## R20 — Big dataset full-retrain + Tier-2 reorientation

- [`R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`](R20_BIG_DATASET_FULL_RETRAIN_PLAN.md) (~18k LOC) — 6-phase plan for R20 full retrain on BindingDB 1.07M + PlatinAI 226K + DrugOOD 1.58M; in-flight per Wf-432 §12.5 + Wf-433 USER_OK-gated waves

## Tier-2 — Reorientation + drop-promote

- [`WEAK_TO_STRONG_PLAN.md`](WEAK_TO_STRONG_PLAN.md) (~31k LOC) — Weak-to-strong distillation + 22-metric × 5-category status table + m6_kappa supplementary sub-table per Wf-389-A1; m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates per Wf-424-A15 §3
- [`METRIC_LIFT_STRATEGY_v2.md`](METRIC_LIFT_STRATEGY_v2.md) (~41k LOC) — 11 sections + per-metric refresh + wall budget envelope + 2026-Q4 GPU retrain priority order (3-batch sequential + paper write batch = 4-batch total, ~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants) per Wf-391-A3
- [`MERGE_GATE.md`](MERGE_GATE.md) (~12k LOC) — Per-R* merge gate agreement

## Pre-retrain protocol-validation gates

- [`PRE_TRAINING_GATE_v2.md`](PRE_TRAINING_GATE_v2.md) (~39k LOC) — Layer 1 (8 Tier-1 must-pass) + Layer 2 (7 Tier-2 recommended) + Layer 3 (5 MOCK→REAL verification) + Layer 4 (3 honest-negative pre-closure) + Layer 5 (9 per-baseline protocol validation hooks = 3 hooks × 3 baselines) per Wf-432 §12.7.4; total NEW gates 23 + 117 baseline = 149 EXPECTED

## Improvement-tracking plans

- [`IMPROVEMENT_ENUM_2026_09_27.md`](IMPROVEMENT_ENUM_2026_09_27.md) (~22k LOC) — 4 → 12 tasks; 5 SHIPPED + 4 SCAFFOLD + 1 HONEST_NEGATIVE_FINAL + 2 PARTIAL per Wf-391-A2
- [`IMPROVEMENT_QUEUE_2026_09_27.md`](IMPROVEMENT_QUEUE_2026_09_27.md) (~18k LOC) — 4 PENDING → 32 tasks; 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY per Wf-391-A2

## Cross-links

- Top-level [../../REPRODUCE.md](../../REPRODUCE.md) — 1-command reproducibility
- [`../02_metrics/ARCHITECTURE.md`](../02_metrics/ARCHITECTURE.md) — 21 mechanisms + 4 phases
- [`../03_baselines/ARCHITECTURE.md`](../03_baselines/ARCHITECTURE.md) — 3 Tier-1 baselines + protocol alignment
- [`../INDEX.md`](../INDEX.md) — top-level TODO directory INDEX
- [`../../decisions.md`](../../decisions.md) — open architecture decisions
- [`../../risks.md`](../../risks.md) — open blockers / risks
- [`../../roadmap.md`](../../roadmap.md) — phase tracker
