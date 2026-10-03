# Wf-320G — MECHANISM_METRIC_MAPPING §13/§14 plan-level cross-walk (23 mechanisms)

**Date**: 2026-09-29
**Wf ID**: Wf-320G (per Wf-313E actual)
**Status**: SHIPPED (CPU-only documentation work)
**Author**: Claude (Wf-320G sub-agent)

---

## 1. Purpose

Per the user request relayed from the harness:
> 能不能基于刚才的洞察启动新的workflow做新的flow up,把暴露的问题全部解决

And the computed task brief (Wf-320G): extend `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` with 23 plan-level mechanisms (M22-M44 per the Wf-313E actual) cross-walking R18 P0-1..P0-4 + P1-1..P1-7 + 2 Wf architectural findings + R20-P1..P6 + 2 Wf findings + 2 R17 ship surfaces.

## 2. What was shipped

### 2.1 Documentation extension (`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`)

Added two new sections:

**§13 — Wf-320G R18/R20 plan-level mechanism registry (2026-09-29)**:
- §13.1 W1 (R18 math): M22-M32 (11 mechanisms)
- §13.2 W2 (Wf architectural findings): M33-M34 (2 mechanisms)
- §13.3 W3 (R20-P1..P6): M35-M40 (6 mechanisms)
- §13.4 W4 (Wf findings): M41-M42 (2 mechanisms)
- §13.5 W5 (R17 ship surfaces): M43-M44 (2 mechanisms)
- §13.6 cross-mapping summary table (M22-M44 × key metric)
- §13.7 failure-mode / regression-debug table

**§14 — Wf-320G plan-ID cross-reference (2026-09-29)**:
- §14.1 Plan-ID → M-ID cross-walk (every M22-M44 → origin plan / Wf / R17 Wf)
- §14.2 R18 / R19 / R20 plan section ↔ M-ID mapping
- §14.3 Honest framing (per `party-twenty-spirit` §3.5)

### 2.2 Test file (`molmetal/tests/test_wf320g_mechanism_metric_mappings.py`)

107 tests covering:
- 23 per-mechanism tests (`test_wf320g_mechanism_in_section_13` × 23)
- 23 mechanism-label tests (`test_wf320g_mechanism_label` × 23)
- 23 status-tag tests (`test_wf320g_mechanism_status` × 23)
- 23 required-fields tests (`test_wf320g_mechanism_required_fields` × 23)
- 5 wave-grouping tests (W1-W5)
- 4 §13 structural tests (§13.6, §13.7, §13.4 status, M40 DEPRECATED)
- 4 §14 structural tests (§14.1, §14.2, §14.3)
- 2 file-path invariant tests

**Test result**: 107 / 107 PASS (Wf-320G specific).
**Wf-313E regression**: 33 / 33 PASS (existing Wf-313E tests still pass).

### 2.3 Status distribution (honest framing per AGENTS.md §1.5 / `party-twenty-spirit` §3.5)

| Status | Count | Mechanism IDs |
|---|---|---|
| CPU-ONLY-SKELETON | 12 | M22, M23, M26, M27, M28, M29, M30, M31, M32, M35, M37, M39 |
| PLAN-ONLY | 2 | M24, M25 |
| PROJECTED @ R20 | 2 | M36, M38 |
| HONEST-NEGATIVE | 5 | M33, M34, M40, M41, M42 |
| MEASURED (R17 ship) | 2 | M43, M44 |
| **Total** | **23** | **M22-M44** |

## 3. Honest framing (per AGENTS.md §1.5 / `party-twenty-spirit` §3.5)

Per AGENTS.md §1.5 + `party-twenty-spirit` §3.5 0 容忍 fabricate:

1. **23 mechanisms are documentation work**, NOT new measurement. §13 extends §3 contribution matrix + §12 regression-debug surface to plan-level items.
2. **M22-M32 (R18 math, 11 mechanisms)**: 9 of 11 already shipped as CPU-only skeletons (§2 + §5 + §6); M24 (Monge-Ampère) and M25 (W₂ gradient flow) are PLAN-ONLY paper §3 theorem citations — not code, not metric uplift.
3. **M33-M34 (architectural findings, 2 mechanisms)**: HONEST-NEGATIVE preconditions. The R² ceiling (Wf-297B) and Vina covalent ceiling (Wf-143 + Wf-301B) are NOT metric uplifts; they document why certain metrics are bounded and what would be needed to lift them.
4. **M35-M40 (R20-P1..P6, 6 mechanisms)**: 3 of 6 are CPU-only skeletons; M36 (Hilbert-ASTSR R² lift) and M38 (3-way on MetalCytoToxDB) are PROJECTED @ R20 (GPU retrain gated per AGENTS.md §5.1 kill_then_adapt directive).
5. **M40 (Wet-lab IC50 validation) is DEPRECATED** per Wf-317 Tier-2; the wet-lab protocol skill `mol-metal-wetlab-protocol` replaces the R20-P6 mechanism entry.
6. **M41-M42 (Wf findings, 2 mechanisms)**: HONEST-NEGATIVE preconditions. They document why DrugOOD (Wf-301A BELOW_RANDOM) and Pt+2 Vina Dock (Wf-299B covalent blocker) are BLOCKED.
7. **M43-M44 (R17 ship, 2 mechanisms)**: MEASURED. Already shipped in §1 (M10 84-rule MLE + M11 18-rule + 4 NEW SMARTS); §13 cross-references them as plan-level R17 ship surfaces for completeness.

## 4. Why §13 instead of overwriting §1-§12

The existing `MECHANISM_METRIC_MAPPING.md` already contains:
- §1: M1-M21 (R17 close-out)
- §2: M22-M28 (R18 W1, Wf-1 + Wf-2 ship surface)
- §3: full 21-mechanism × 20-metric contribution matrix
- §4: failure-mode diagnostic
- §5: M29-M35 (R18 W2, Wf-10 + Wf-11 + Wf-12 + Wf-14 + Wf-15 ship surface)
- §6: M36-M44 (R18 W3, Wf-7 + Wf-11 + Wf-12 + Wf-1 ship surface) + M45 (Wf-318F) + M45b (Wf-319G)
- §7: ablation design
- §12: regression-debug table (Wf-313E, 2026-09-29)

The user's task brief lists M22-M44 with names that conflict with the existing M22-M44 (e.g., task M24 = Monge-Ampère, but existing M24 = Multi-PH). To avoid destroying existing Wf-1/Wf-2/Wf-7/Wf-10/Wf-11/Wf-12/Wf-14/Wf-15 ship work, Wf-320G:
1. **Preserves existing §1-§12 implementation-level entries** (M1-M45b).
2. **Adds §13 + §14 as a plan-level cross-walk** that registers the user's 23 mechanism names and maps each to its existing M-ID (or marks PLAN-ONLY).
3. **§13 uses status tags** (CPU-ONLY-SKELETON / PLAN-ONLY / PROJECTED @ R20 / HONEST-NEGATIVE / MEASURED) to make the implementation status explicit.

## 5. Cross-references

| Doc | Purpose |
|---|---|
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §13 | Wf-320G plan-level registry (this ship) |
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §14 | Wf-320G plan-ID cross-reference (this ship) |
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §12 | Wf-313E regression-debug table (already shipped) |
| `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` | R18 P0-1..P0-4 + P1-1..P1-7 plan IDs (M22-M32 source) |
| `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` | R20-P1..P6 plan IDs (M35-M40 source) |
| `molmetal/tests/test_wf320g_mechanism_metric_mappings.py` | 107 Wf-320G tests (this ship) |
| `molmetal/tests/test_wf313e_mechanism_metric_mappings.py` | 33 Wf-313E tests (already shipped) |
| `TODO/completed/220_wf220_temporal_split_acceptance.md` | Wf-220 honest framing precedent (R17 close-out style) |
| `mol-metal-mechanism-mapping` skill | §R18 mechanism cross-reference + §R20 Lambda algorithm reference |

## 7. Per-mechanism source citation

| M-ID | Source citation |
|---|---|
| M22 | R18 P0-1, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §2 (Mahmood 2026, arXiv:2603.14049) |
| M23 | R18 P0-2, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §2 (Jiang-Xu-Zhang 2026, arXiv:2602.08118) |
| M24 | R18 P0-3, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §2 (汪徐家 2026) |
| M25 | R18 P0-4, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §2 (薛小平 2026) |
| M26 | R18 P1-1, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (Kim & Jung 2026, *Adv. Appl. Math.* 174) |
| M27 | R18 P1-2, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (Kolbe & Mayr 2026, arXiv:2608.27269) |
| M28 | R18 P1-3, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (Henry-Labordere 2017 + Backhoff-Veraguas 2025) |
| M29 | R18 P1-4, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (Sun & Zheng 2026, 数学学报 53(4)) |
| M30 | R18 P1-5, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (数学学报 2026 fractional RD) |
| M31 | R18 P1-6, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (麻小南 2026 + Ma & Zhang 2008 Astérisque 318) |
| M32 | R18 P1-7, `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 (李思 2026) |
| M33 | Wf-297B R² architectural ceiling finding (planned `TODO/completed/297b_r2_ceiling.md`) |
| M34 | Wf-143 + Wf-301B Vina covalent mechanism ceiling finding (planned) |
| M35 | R20-P1, `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` (Wf-309C shipped) |
| M36 | R20-P2, `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (PROJECTED @ R20) |
| M37 | R20-P3, `TODO/pending/94_r20_p3_multiph_pll_feature.md` |
| M38 | R20-P4, `TODO/pending/95_r20_p4_3way_100pocket.md` (PROJECTED @ R20) |
| M39 | R20-P5, `TODO/pending/96_r20_p5_closure_catherine.md` |
| M40 | R20-P6, `TODO/pending/97_r20_p6_wetlab_ic50.md` (DEPRECATED per Wf-317 Tier-2) |
| M41 | Wf-301A DrugOOD BELOW_RANDOM fix finding (planned) |
| M42 | Wf-299B Pt+2 parameterisation covalent blocker finding (planned) |
| M43 | R17 ship: `TODO/completed/180_wf124_p3_remaining_metals_seeds.md` (84-rule MLE matrix = M10) |
| M44 | R17 ship: `TODO/completed/177_wf123_p3_m1m2_downstream_metric_lift.md` (18-rule + 4 NEW = M11) |

## 8. Pre-flight status

Per AGENTS.md §5.4 (TODO-207 pre-flight v2 25/25 PASS gate): this is documentation work, NOT a retrain / panel / ablation launch. The pre-flight gate is N/A for documentation work.

Per `party-twenty-spirit` §3.5 honest framing:
- ✅ Zero fabricated numbers
- ✅ All 23 mechanisms have source citations (§13.1-§13.5 + §14.1)
- ✅ All HONEST-NEGATIVE entries (M33, M34, M40, M41, M42) declare their precondition status
- ✅ DEPRECATED entry (M40) cites Wf-317 Tier-2 deprecation
- ✅ PROJECTED @ R20 entries (M36, M38) cite AGENTS.md §5.1 kill_then_adapt directive
- ✅ MEASURED entries (M43, M44) cite existing M10/M11 ship surface

## 9. Wf-320G ship summary

| Surface | Status |
|---|---|
| §13 plan-level registry (5 sub-sections) | SHIPPED |
| §13.6 cross-mapping summary | SHIPPED |
| §13.7 failure-mode diagnostic | SHIPPED (added 5 new rows for M31/M32/M35/M36/M40) |
| §14 plan-ID cross-reference (3 sub-sections) | SHIPPED |
| Test file `test_wf320g_mechanism_metric_mappings.py` (107 tests) | SHIPPED |
| 23 mechanism entries (M22-M44) | SHIPPED |
| Pre-flight v2 25/25 PASS gate | N/A (documentation work) |
| 0 GPU retrain shipped | YES (per AGENTS.md §5.1) |

---

**END OF Wf-320G SHIP REPORT** — Owner: Wf-320G sub-agent; cross-ref `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §13 + §14 + `molmetal/tests/test_wf320g_mechanism_metric_mappings.py`; last update 2026-09-29.
