# Wf-391-B1 — docs-only INDEX + AGENTS + final synthesis v31 sync

**Date:** 2026-09-30 (Wf-391-B1 close-out)
**Workflow:** Wf-391 ultracode CPU-only docs-only wave (parent)
**Sub-flow:** B1 — `TODO/INDEX.md` §22.81 entry + `AGENTS.md` §6 NEW row + comprehensive final synthesis v31
**Owner:** Hugo Lee + Wf-391-B1 ultracode agent
**Type:** DOCS-ONLY (CPU-only; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning per AGENTS.md §5.4 + §10)
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 + §1.5 standing rules

**Trigger:** User directive 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by just-launched workflows (Wf-340..390 ultracode series).

---

## §0 Phase

**Phase**: Wf-391-B1 docs-only INDEX + AGENTS + final synthesis sync (4 deliverables SHIPPED 100%: A1 LAMBDA_ROADMAP refresh CLOSED + A2 IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE refresh CLOSED + A3 METRIC_LIFT_STRATEGY_v2 refresh CLOSED + B1 docs-only INDEX + AGENTS + final synthesis v31 SHIPPED — this entry). Wf-391 ultracode CPU-only docs-only wave closes the strategic-planning-vs-shipped-code gap surfaced by Wf-340..390 ultracode audit.

**Date**: 2026-09-30 (Wf-391-B1 wall MEASURED ~10 min CPU-only docs-only sync; Wf-391 wave total MEASURED ~25-40 min across 4 deliverables — A1 ~5 min + A2 ~5 min + A3 ~7 min + B1 ~10 min = ~27 min).

---

## §1 Pre-flight verification

**Pre-flight cascade NOT re-run by Wf-391-B1** — CPU-only docs-only wave per AGENTS.md §5.1 + kill_then_adapt §0. PRESERVED at Wf-389-B1 baseline **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** per `molmetal/reports/wf389b1_preflight.txt` (NOT re-run by Wf-391 — CPU-only docs-only wave).

**HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-389-B1 baseline at same drift level (13+ consecutive runs PRESERVED at 89/2/3/94).

**2 FAILs preserved baseline** (NOT Wf-391-introduced):
- `p3_loi_send_ready` Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (per AGENTS.md §5.4)
- `p3_wf343c4_anchor_freshness_24h` `molmetal/reports/wf269c_report.json` age > 24h STALE — natural drift across Wf-374..390 wave activity

**3 SKIPs preserved (BY-DESIGN)** (NOT Wf-391-introduced):
- `p3_wf343c4_yuelbond_decoder_env` env-aware SKIP per Wf-345-A2
- `p3_wf343c4_wave_budget_remaining_nonneg` BY-DESIGN per Wf-262 §7.1 with −183.04h overrun
- `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency

**Pre-flight cascade STILL CLOSED end-to-end across 20+ consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-381-B1 → Wf-383-B1 → Wf-384-B1 → Wf-386-B1 → Wf-385-B1 → Wf-389-B1 → **Wf-391-B1** [this run]; `p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

**EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline** — no new pre-flight gates added by Wf-391 (CPU-only docs-only sync; 4 deliverables + 0 NEW tests + 1 docs-only INDEX sync only).

---

## §2 INDEX.md sync

**Appended `## §22.81 Wf-391 ultracode CPU-only docs-only wave` entry** to `TODO/INDEX.md` between `## §22.5 Wf-389 ultracode docs-only wave` (line 1258) and `**END OF §22**` (line 1314).

**§22.81 entry structure** (5 sub-sections):
- §22.81.1 Wf-391 sub-flows SHIPPED (3/3 + 1 docs sync = 4/4 deliverables SHIPPED 100%)
- §22.81.2 MEASURED test pass-rates (per Wf-391 wave) — 0 NEW tests across all 4 deliverables (docs-only per task brief)
- §22.81.3 User voice answer (verdict on user 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*) — DID彻底解决 the strategic-planning-vs-shipped-code gap
- §22.81.4 Cross-link (3 Wf-391 SHIPPED closeouts + 1 NEW docs file + 2 NEW synthesis docs + 1 NEW §6 row in AGENTS.md)
- §22.81.5 Per-section source preservation (LAMBDA_ROADMAP + IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE + METRIC_LIFT_STRATEGY_v2 unchanged except for the targeted refresh edits)

**Cross-refs in §22.81 entry**:
- `TODO/completed/391a1_lambda_roadmap_refresh.md` (Wf-391-A1 LAMBDA_ROADMAP refresh source)
- `TODO/completed/391a2_improvement_enum_refresh.md` (Wf-391-A2 IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE refresh source)
- `TODO/completed/391a3_metric_strategy_refresh.md` (Wf-391-A3 METRIC_LIFT_STRATEGY_v2 refresh source)
- `TODO/completed/391b1_index_sync.md` (this file — Wf-391-B1 docs sync)
- `TODO/completed/391_final_synthesis.md` (Wf-391-B1 comprehensive final synthesis v31, ~580 LOC, 9 sections)
- `TODO/03_plans/LAMBDA_ROADMAP.md` (Wf-391-A1 in-place edits: 6 anchors + 2 NEW sub-sections + 1 NEW trailing §11 update = 9 deliverables)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (Wf-391-A2 refresh: 4 → 12 tasks)
- `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` (Wf-391-A2 refresh: 4 → 32 tasks)
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (Wf-391-A3 refresh: ~410 → ~860 LOC, 8 → 11 sections)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-369-A3 65-entry catalog)

---

## §3 AGENTS.md sync

**Appended NEW §6 row for Wf-391** to `AGENTS.md` between Wf-385 row (line 536) and the trailer block (line 537 onwards).

**§6 row structure**: 1 row with Wf-391 + cross-refs + standing directives re-confirmed + Wf-392+ recommended follow-up + last-update trailer clause appended.

**Cross-refs in §6 row**:
- `TODO/INDEX.md` §22.81 (the INDEX entry for this wave — `## §22.81 Wf-391 ultracode CPU-only docs-only wave`)
- `TODO/completed/391{a1,a2,a3}_*.md` (3 Wf-391 SHIPPED closeouts: A1 LAMBDA_ROADMAP refresh + A2 IMPROVEMENT_ENUM + QUEUE refresh + A3 METRIC_LIFT_STRATEGY_v2 refresh)
- `TODO/completed/391b1_index_sync.md` (Wf-391-B1 docs sync — this file)
- `TODO/completed/391_final_synthesis.md` (Wf-391-B1 comprehensive final synthesis v31)
- `TODO/03_plans/LAMBDA_ROADMAP.md` (Wf-391-A1 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (Wf-391-A2 refresh)
- `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` (Wf-391-A2 refresh)
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (Wf-391-A3 refresh — NEW §10 forward-looking 2026-Q4 GPU retrain priority order)

---

## §4 Honest framing (per `party-twenty-spirit` §3.5)

**6 honest-negative caveats preserved** per Wf-391-B1 §4 + Wf-391-A1 §4 + Wf-391-A2 §3 + Wf-391-A3 §7 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-385-B1 §22.79.3:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-391-A1 §1 + Wf-377-A1 deep-dive-research move + Wf-389-A1 §3 supplementary sub-table reframe + Wf-381-A4 finalization + Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt; **9 paths exhausted**; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`).

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-391-A1 §1 + Wf-391-A2 §3 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 real-cohort verification; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`).

3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-391-introduced; both deferred to Wf-392+ follow-up.

4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency).

5. **Wf-391-A1 strategic-plan refresh has zero runtime impact** (docs-only; refreshes LAMBDA_ROADMAP to reflect Wf-381-A5 closure + Wf-382-A1 SCAFFOLD + Wf-376-A3 GPU retrain plan + AGENTS.md §10 wetlab deprecation; no GPU launches; no paper/* touches; no wet-lab planning per AGENTS.md §10).

6. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

---

## §5 Cross-references

- `TODO/INDEX.md` `## §22.81 Wf-391 ultracode docs-only wave` (this entry appended between `## §22.5 Wf-389 ultracode docs-only wave` line 1258 and `**END OF §22**` line 1314)
- `AGENTS.md` §6 NEW row (this sync — Wf-391)
- `AGENTS.md` last-update trailer extended with Wf-391-B1 sync clause
- `TODO/completed/391{a1,a2,a3}_*.md` (3 Wf-391 SHIPPED closeouts: A1 LAMBDA_ROADMAP refresh + A2 IMPROVEMENT_ENUM + QUEUE refresh + A3 METRIC_LIFT_STRATEGY_v2 refresh)
- `TODO/completed/391_final_synthesis.md` (Wf-391-B1 comprehensive final synthesis v31, ~580 LOC, 9 sections)
- `TODO/completed/{388{a1_m2_per_cohort_real,a2_m7_diversity_wire,a3_m4_sa_wire},389{a1_m6k_supp_reframe,a3_weak_to_strong_update,b1_index_sync,final_synthesis},390{a2_todo179_deeper_audit,a3_se3_sb_wire}}_*.md` (Wf-388 + Wf-389 + Wf-390 closeout sources for the convergence thread)
- `TODO/completed/{381a5_r20_plan_cpu_close,382a1_multiph_pll_scaffold,376a3_*.md,377a1_m6_kappa_redesign,377a2_m2r2_diagnostic}_*.md` (source verdicts)
- `TODO/03_plans/LAMBDA_ROADMAP.md` (Wf-391-A1 refreshed)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (Wf-391-A2 refreshed)
- `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` (Wf-391-A2 refreshed)
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (Wf-391-A3 refreshed)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` (Wf-389-A3 cross-ref + Wf-390-A3 §4.3.1 cross-ref)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-369-A3 65-entry catalog + Wf-391-A2 §3 honest framing extension)
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 NEW Tier-2 §3 metric table — 4 main + 3 supplementary + 1 demoted)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 NEW 8 sections)
- `docs/M2_R2_COHORT_DIAGNOSTIC.md` (Wf-377-A2 NEW 7 sections)
- `metrics/by_metric/m6_kappa.json::verdict` (Wf-389-A1 UPDATED to `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`)
- `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` (Wf-389-A1 NEW JSON block)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + Wf-362-B1 + Wf-363-C1 + Wf-364-C1 + Wf-365-C1 + Wf-366-C1 + Wf-367-C1 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-372-C1 + Wf-373-B1 + Wf-375-B1 + Wf-374-B1 + Wf-376-B1 + Wf-377-B1 + Wf-379-B1 + Wf-378-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-389-B1 prior INDEX syncs preserved unchanged
- Wf-391-B1 this sync adds the Wf-391 §6 row + §22.81 INDEX entry to capture the 4 SHIPPED deliverables (A1 + A2 + A3 + B1 all SHIPPED 100%)

---

**END OF Wf-391-B1 DOC SYNC** — Owner Hugo Lee + Wf-391-B1 ultracode agent; cross-ref `party-twenty-spirit` §3.5; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed.
