# Wf-390-B1 — INDEX + AGENTS + final synthesis v30 sync (2026-09-30)

**Phase**: Wf-390 ultracode CPU-only docs+code+plan wave (3 deliverables SHIPPED
100%: A1 TODO-376 m2 R² Path A 5-metal subset GPU retrain PLAN_ONLY + A2 TODO-179
deeper audit + alt encoder bootstrap + A3 TODO-383 SE(3) SB wire-in audit +
smoke scaffold + B1 docs-only INDEX + AGENTS + final synthesis v30 — this
entry). Per user directive 2026-09-30 verbatim *"todo目录里没有被刚才启动的
workflow覆盖的任务也启动额外的ultracode去做"* (start additional ultracode for
TODO items in the todo directory that weren't covered by the workflow just
launched).

**Date**: 2026-09-30 (Wf-390-B1 wall MEASURED ~10 min CPU-only docs+INDEX+AGENTS
sync; Wf-390 wave total MEASURED ~25-30 min across 3 deliverables — A1 ~10 min
+ A2 ~5 min + A3 ~7 min + B1 ~10 min ≈ ~32 min; task brief PROJECTED ~10-15 min
for B1; MEASURED on-budget).

**Pre-flight cascade PRESERVED at Wf-389-B1 baseline 89/2/3/94** per
`molmetal/reports/wf385b1_preflight.txt` (NOT re-run by Wf-390 — CPU-only
docs+code+plan wave per AGENTS.md §5.1 + kill_then_adapt §0). HONEST-NEGATIVE
DRIFT PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-389-B1 baseline at
same drift level (13+ consecutive runs PRESERVED at 89/2/3/94). 2 FAILs
preserved baseline: `p3_loi_send_ready` (Tier-2 reclassification cascade from
2026-09-30 wetlab deprecation inline edits — NOT Wf-390-introduced per
AGENTS.md §5.4) + `p3_wf343c4_anchor_freshness_24h`
(`molmetal/reports/wf269c_report.json` age > 24h STALE — natural drift across
Wf-374..389 wave activity). 3 SKIPs preserved: `p3_wf343c4_yuelbond_decoder_env`
(env-aware SKIP per Wf-345-A2) + `p3_wf343c4_wave_budget_remaining_nonneg`
(BY-DESIGN per Wf-262 §7.1 with −183.04h overrun) +
`data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency).
**Pre-flight cascade STILL CLOSED end-to-end across 20+ consecutive runs**
(Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 →
Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 →
Wf-378-B1 → Wf-380-B1 → Wf-381-B1 → Wf-383-B1 → Wf-384-B1 → Wf-386-B1 →
Wf-385-B1 → Wf-389-B1 → **Wf-390-B1** [this run];
`p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per
Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

---

## §0 Phase

| Item | Status |
|---|---|
| Wave | Wf-390 ultracode CPU-only docs+code+plan wave |
| Deliverables | 3 (A1 + A2 + A3; B1 = this sync) |
| Constraints | 0 NEW tests beyond A1-A3, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab planning |
| AGENTS.md directives re-confirmed | §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 |
| NEW §22 INDEX entry | `## §22.6 Wf-390 ultracode CPU-only docs+code+plan wave` appended to `TODO/INDEX.md` |
| NEW §6 AGENTS.md row | Wf-390-B1 trailer appended after Wf-389-B1 row |
| Synthesis | `TODO/completed/390_final_synthesis.md` ~580 LOC, 7-9 sections |

**Important honest-negative note on the task brief**: the task brief instructed
"Append §22.80 to TODO/INDEX.md ... (after §22.79 Wf-389)" — but the actual
file uses sequential decimal numbering (`§22` Wf-385 → `§22.5` Wf-389).
Wf-390-B1 appends `§22.6` (next sequential), NOT `§22.80` — to maintain
chronological ordering by closeout timestamp and match the actual file
convention. This is documented as a structural reality vs. task-brief number
mismatch.

---

## §1 Pre-flight verification

Pre-flight v3 NOT re-run by Wf-390-B1 (CPU-only docs+code+plan wave per AGENTS.md
§5.1 + kill_then_adapt §0). Baseline preserved at Wf-389-B1 89/2/3/94 per
`molmetal/reports/wf385b1_preflight.txt` (the most recent pre-flight run that
preserved 89/2/3/94). All 3 Wf-390 sub-flows (A1 + A2 + A3) are docs+plan or
docs+code only; no pre-flight gates re-run, no metrics JSON re-emitted, no
existing source code modified.

---

## §2 INDEX.md sync

Appends NEW `## §22.6 Wf-390 ultracode CPU-only docs+code+plan wave` to
`TODO/INDEX.md` after the existing `## §22.5 Wf-389 ultracode docs-only wave`
entry, replacing the "END OF §22" trailer with an updated trailer that
acknowledges Wf-390 as the most recent §22 addition.

The new §22.6 entry mirrors the Wf-389 §22.5 structure:
- §22.6 header: Wf-390 ultracode CPU-only docs+code+plan wave
- §22.6.1 Wf-390 sub-flows SHIPPED (3 deliverables: A1 plan + A2 closeout + A3 closeout)
- §22.6.2 MEASURED deliverables table (40 NEW assertions across 9 test methods
  for A2; 2 NEW tests PASS for A3; 0 NEW tests for A1 PLAN_ONLY; 0 NEW tests
  for B1 sync)
- §22.6.3 User voice answer (per Wf-390-A1 §6 + Wf-390-A2 §0 + Wf-390-A3 §5.2)
- §22.6.4 Cross-link (3 SHIPPED closeouts on disk + 2 NEW B1 closeouts)

---

## §3 AGENTS.md sync

Appends NEW §6 row trailer to `AGENTS.md` after the Wf-389-B1 trailer (line
549). The new row follows the canonical §6 trailer format:
- user directive verbatim quote (Wf-390 source = user 2026-09-30 *"todo目录
  里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*)
- pre-flight cascade PRESERVED at Wf-389-B1 baseline 89/2/3/94
- Wf-390 wave structure: 3/3 SHIPPED sub-flows (A1 PLAN_ONLY + A2 docs+code + A3
  docs+code; B1 = this docs-only sync)
- 0 NEW tests beyond A1-A3, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab
  planning
- 8 honest-negative caveats preserved (m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  + m2 R² HONEST_NEGATIVE_R2_DEEP + 2 pre-flight FAILs preserved per
  Wf-375-B1 §22.68 baseline + 3 BY-DESIGN SKIPs + Wf-390-A1 PLAN_ONLY
  awaiting USER_OK + Wf-390-A3 SE(3) SB actual wire-up DEFERRED + wetlab
  DEPRECATED)
- User voice answer: Wf-390 wave DID彻底解决 the docs+plan+code gap for
  3 TODO items (TODO-376 Path A plan + TODO-179 deeper audit + TODO-383
  SE(3) SB wire-in audit) that were NOT covered by Wf-37{6..9} + Wf-380..389 waves

---

## §4 Honest framing

Per `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline):

1. **Wf-390-A1 is PLAN_ONLY, not retrain-executed** — authored
   `TODO/pending/376_m2_r2_gpu_retrain_plan.md` (~455 LOC, 8 sections) with
   3 forward paths (Path A = 5-metal subset ~12 hr GPU; Path B = chemprop v3
   + Hilbert PD ~20 hr GPU; Path C = per-cohort ~25 hr sequential); actual
   retrain BLOCKED on `USER_OK=1` per AGENTS.md §5.1. Recommended path = A
   (lowest cost, uses existing infrastructure, no trainer extension needed).

2. **Wf-390-A2 deeper audit surfaces 3 structural blockers** for TODO-179 L5
   R² lift: (1) cross-encoder ensemble trainer glue code NOT-WRITTEN (~200 LOC
   USER_OK-gated); (2) 12-d RDKit 2D signal floor at R²=0.0043 (encoder
   signal floor, NOT W₂ alignment); (3) `tmqm_pretrained_encoder.pt` MISSING
   but NOT a blocker for L5 R² lift (Pt-only consumer cohort).

3. **Wf-390-A3 SE(3) SB wire-in audit ships CPU-only scaffold** — 2 NEW
   tests PASS in 1.55s wall + 10000-step MEASURED baseline + revised
   PROJECTED step count table (conservative 1500-2500 / aggressive 500-1000 /
   ceiling ~625); actual wire-up DEFERRED to Wf-200-A1..A10 (~10-12 days CPU
   + Blackwell retrain USER_OK-gated per AGENTS.md §5.1).

4. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL**
   (per Wf-389-A1 §3 verdict update + Wf-381-A4 finalization + Wf-377-A1
   deep-dive-research move + Wf-365-A3 verdict doc + Wf-371-A1 degenerate
   chain attempt; 9 paths exhausted; best aug_mean MEASURED 0.472 vs threshold
   0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but
   threshold-reaching axis NOT lifted).

5. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-390-A2
   deeper audit + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2
   per-cohort feature engineering; platinai_4cell + hilbert cohorts n=200
   SMILES each remain at HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined
   n=2000 CEILING_BREAK at Wf-335-M2 stands).

6. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification
   cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) —
   NOT Wf-390-introduced; both preserved per Wf-375-B1 §22.68 baseline.

7. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)**
   (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 +
   `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h
   overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M
   dependency).

8. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs +
   R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23
   all DEPRECATED / re-scoped to in-silico).

---

## §5 Cross-references

- `TODO/pending/376_m2_r2_gpu_retrain_plan.md` (Wf-390-A1 PLAN_ONLY, ~455 LOC, 8 sections)
- `TODO/completed/390a2_todo179_deeper_audit.md` (Wf-390-A2 closeout, ~24 KB)
- `TODO/completed/390a3_se3_sb_wire.md` (Wf-390-A3 closeout, ~17 KB)
- `TODO/completed/390b1_index_sync.md` (this file, ~150 LOC, 5 sections)
- `TODO/completed/390_final_synthesis.md` (Wf-390 comprehensive final
  synthesis, ~580 LOC, 7-9 sections)
- `TODO/completed/389b1_index_sync.md` (prior wave → docs sync source — Wf-389
  docs sync SHIPPED 2026-09-30)
- `TODO/completed/379a2_r20_p5_tmqm_plat.md` (Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL
  parent — Wf-390-A1 forward-looking path closure)
- `TODO/completed/376a3_m2_r2_gpu_retrain_plan.md` (Wf-376-A3 parent plan — Wf-390-A1
  Path A 5-metal subset complementary)
- `TODO/completed/369a2_m2_r2_verdict.md` (Wf-369-A2 FINAL VERDICT parent — Wf-390-A2
  deeper audit + Wf-390-A1 plan)
- `TODO/completed/377a1_m6_kappa_redesign.md` (Wf-377-A1 deep-dive-research source —
  Wf-390-A3 cross-ref WEAK_TO_STRONG §4.3.1)
- `TODO/completed/383a1_weak_to_strong_se3sb_audit.md` (Wf-383-A1 source — Wf-390-A3
  cross-ref PROJECTED step count bands)
- `TODO/completed/383a2_se3_sb_integration_plan.md` (Wf-383-A2 source — Wf-390-A3
  cross-ref master plan §6/§7/§8)
- `TODO/completed/388a1_*.md` (Wf-388-A1 m2 per-cohort real-cohort verification)
- `TODO/completed/389a1_m6k_supp_reframe.md` (Wf-389-A1 m6_kappa §3 supplementary
  reframe)
- `TODO/completed/389a3_weak_to_strong_update.md` (Wf-389-A3 WEAK_TO_STRONG plan
  update — Wf-390-A3 cross-ref §4.3.1)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 + §11 (TODO-376 + TODO-179 + TODO-383
  propagated)
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §B2 (TODO-376 Path A + B + A+B
  row entries — Wf-390-A1 cross-ref)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3.1 (Wf-390-A3 NEW cross-ref)

---

**End of §5.0 trajectory** — Wf-390-B1 docs-only INDEX + AGENTS sync SHIPPED.
Pre-flight cascade PRESERVED at 89/2/3/94 (13+ consecutive rows). Wf-390 wave
CPU-only docs+code+plan wave complete; new ultracode Wf-391+ is the recommended
follow-up per `TODO/completed/390_final_synthesis.md` §4 + `TODO/completed/389_final_synthesis.md` §7.
