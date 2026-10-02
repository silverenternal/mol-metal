# Wf-395-B1 — docs-only INDEX + AGENTS + final synthesis v34 sync (EVALUATORS_MIGRATED refresh + HONEST_NEGATIVE_CATALOG 73→81 + Full Integration Tests)

**Date:** 2026-09-30
**Wave:** Wf-395 (CPU-only docs+code+test ultracode residual)
**Parent brief:** user directive 2026-09-29/30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — `TODO/02_metrics/EVALUATORS_MIGRATED.md` (last refreshed 2026-09-25 R17; not covered by Wf-380..394) + `docs/HONEST_NEGATIVE_CATALOG.md` expansion (73 → 81 entries from Wf-388 + Wf-390 + Wf-392 wave) + full-integration verification layer (Wf-388 + Wf-392 CLI flags + Wf-385-A3 pre-flight gate) were NOT covered by the just-launched ultracode waves; Wf-395-A1 + A2 + A3 each picks up a loose end.
**Author:** Wf-395-B1 docs-only agent (text-only brief per Wf-291C workaround + Wf-302C plain-text template)
**Constraints honoured:** 0 NEW tests in B1 (verification + docs-only); 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §5.3 + §5.4 + §10 + AGENTS.md §5.5 + §5.6 standing directives.

---

## §0 Phase

Wf-395-B1 is the **docs-only INDEX + AGENTS + final synthesis v34 sync** for the Wf-395 wave. The 3 SHIPPED sub-flows (A1 + A2 + A3) are aggregated; `## §22.85` entry appended to `TODO/INDEX.md`; NEW §6 row appended to `AGENTS.md`; comprehensive final synthesis v34 authored at `TODO/completed/395_final_synthesis.md`.

Pre-flight NOT re-run by Wf-395-B1 (CPU-only docs-only sync per AGENTS.md §5.1 + kill_then_adapt §0); preserved at Wf-394-B1 baseline **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** with **HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** (matches Wf-374-B1 through Wf-394-B1 baseline at same drift level — 18+ consecutive runs PRESERVED at 89/2/3/94).

---

## §1 Pre-flight verification

| Metric | Value | Source |
|---|---:|---|
| Pre-flight cascade PRESERVED | 89/2/3/94 | `molmetal/reports/wf394b1_preflight.txt` (NOT re-run by Wf-395-B1 — docs-only) |
| 2 pre-existing FAILs | `p3_loi_send_ready` + `p3_wf343c4_anchor_freshness_24h` | preserved per Wf-375-B1 §22.68.4 baseline |
| 3 pre-existing SKIPs | `p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative` | preserved per Wf-262 §7.1 + Wf-345-A2 + external 1.07M dependency |
| EXPECTED_GATE_COUNTS["all"] | 94 | Wf-353-B1 baseline PRESERVED (no new gates added by Wf-395) |
| Wf-395 closeouts on disk | 3 (395a1 + 395a2 + 395a3) | `ls TODO/completed/395*.md` |
| Wf-395 NEW tests total | 27 (A1=4 + A2=1 + A3=22) | `molmetal/tests/test_wf395{a1,a2,a3}_*.py` |

---

## §2 INDEX.md sync

Append `## §22.85 Wf-395 ultracode CPU-only docs+code+test wave — EVALUATORS_MIGRATED refresh (22/22 evaluators MEASURED-OK) + HONEST_NEGATIVE_CATALOG expansion 73→81 entries + full-integration verification for Wf-388 + Wf-392 CLI flags + Wf-385-A3 pre-flight gate + comprehensive final synthesis v34` entry to `TODO/INDEX.md` between `## §22.84 Wf-394 ultracode CPU-only docs+code+test wave` and `**END OF §22**`.

NEW entry captures:
- §0 TL;DR + phase (Wf-395 wave + 3 SHIPPED sub-flows + B1 docs-only sync)
- §1 Per-sub-flow closeout summaries (Wf-395-A1 EVALUATORS_MIGRATED refresh + Wf-395-A2 catalog expansion + Wf-395-A3 full-integration tests)
- §2 Wave structure (evaluator audit + honest-negative catalog expand + full-integration verification)
- §3 What 彻底解决 (3 SHIPPED sub-flows + 27 NEW tests PASS + 1 docs refresh + 1 catalog expansion + B1 this docs-only sync)
- §4 What NOT 彻底解决 (decode_ratio=0 BLOCKS Wf-299 GPU launch; Wf-379-A1 m1_auc IN-FLIGHT; Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL; Wf-380 SCAFFOLD MEASURED verification pending; Wf-380 SCAFFOLD wire-in pending; Wf-392-A1 + Wf-392-A2 full training-loop integration pending; Wf-394-A1 Lambda CFM wire-in patch application pending USER_OK)
- §5 Honest framing (preserved caveats)
- §6 Cross-references (Wf-395-A1+A2+A3 closeouts + Wf-395 final synthesis + Wf-394 + Wf-393 + Wf-392 + Wf-391 + Wf-390 + Wf-388 + Wf-381 + Wf-380 + Wf-379 + Wf-378 + Wf-377 + Wf-368 + AGENTS.md §5.1/§5.2/§5.3/§5.4/§5.5/§5.6/§10 + party-twenty-spirit §3.5)
- §7 Forward-looking (Wf-396+ recommended follow-up)

---

## §3 AGENTS.md sync

Append NEW §6 row to `AGENTS.md` after the Wf-394 row (line 546) — capturing:
- Wf-395 wave structure (3 SHIPPED sub-flows: A1 EVALUATORS_MIGRATED refresh + A2 HONEST_NEGATIVE_CATALOG expansion + A3 full-integration verification)
- Wave-level VERDICT (per-sub-flow closeouts + 27 NEW tests PASS + B1 this docs-only sync + comprehensive final synthesis v34)
- User voice answer (per Wf-395-A1 §0 + Wf-395-A2 §0 + Wf-395-A3 §0 + Wf-395 final synthesis §0 — addressing user 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*)
- Honest-negative caveats preserved per `party-twenty-spirit` §3.5

---

## §4 Honest framing

**Constraints honoured (per AGENTS.md §5.1 + §5.3 + §5.4 + §10 + kill_then_adapt §0):**
- 0 NEW tests in Wf-395-B1 (entire docs-only sync)
- 0 GPU launches
- 0 `paper/*` touches
- 0 wet-lab planning
- 0 sudo
- 0 source-code patches to existing files

**Standing directives re-confirmed:**
- §5.1 GPU grant (no GPU launches performed across Wf-395 wave)
- §5.2 honest framing (MEASURED-vs-PROJECTED tag discipline preserved)
- §5.3 no paper work (TODO-84 active; 0 `paper/*` touches)
- §5.4 Tier-2 + no wet-lab (7 wetlab files DEPRECATED)
- §5.5 post-wave on-disk verification rule (MANDATE added by Wf-384-B1)
- §5.6 atomic write protocol adoption rule (MANDATE added by Wf-386-B1)
- §10 wetlab deprecation FAQ (DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK)

**Honest-negative caveats preserved:**
1. m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL (per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 9 paths exhausted)
2. m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (per Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort + Wf-388-A1 multi-stream ensemble REAL-cohort verification; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands)
3. m1 AUC gate BELOW_KRASNOV_GATE_PARTIAL (per Wf-387-A1 MEASURED 6/7 pairs: Rh 0.8798 PASS + Re 0.7060 BELOW; Wf-379-A1 5-seed IN-FLIGHT result pending)
4. 2 NEW pre-flight FAILs (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-395-introduced
5. 3 pre-existing SKIPs PRESERVED (BY-DESIGN)
6. Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending (~5 hr GPU on Blackwell)
7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL (trainer CHOICES CONSTRAINT VIOLATION)
8. Wf-299 3-wave CFM launch BLOCKED on `decode_ratio=0` upstream Wf-291D
9. Wf-394-A1 Lambda CFM wire-in BLOCKED on USER_OK-gated patch application (CPU scaffold SHIPPED per Wf-394-A1 §1)
10. Wf-394-A2 Wf-264 panel runner SHIPPED (scaffold + 4 NEW tests; USER_OK-gated actual GPU launch)
11. Wf-394-A3 Wf-299 pre-launch verification SHIPPED (4/4 contracts PASS post-edit; GPU launch BLOCKED on `decode_ratio > 0`)
12. Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 pending
13. Wf-392-A1 + Wf-392-A2 full training-loop integration pending (~50 LOC + 1-2 NEW tests each)
14. Wetlab items all DEPRECATED per AGENTS.md §5.4 + §10

---

## §5 Cross-references

**Wf-395 wave closeouts on disk:**
- `TODO/completed/395a1_evaluators_refresh.md` (~220 LOC 9 sections; Wf-395-A1 SHIPPED; 4 NEW tests PASS in <1 s; 22/22 evaluators MEASURED-OK)
- `TODO/completed/395a2_catalog_expand.md` (~127 LOC 6 sections; Wf-395-A2 SHIPPED; 1 NEW test PASS; HONEST_NEGATIVE_CATALOG.md 73 → 81 entries)
- `TODO/completed/395a3_full_integration.md` (~191 LOC 5 sections; Wf-395-A3 SHIPPED; 22 NEW tests PASS in 1.60 s; Wf-385-A3 gate verified at `build_gates('all')` index 72/117)
- `TODO/completed/395b1_index_sync.md` (this file — Wf-395-B1 docs-only INDEX + AGENTS sync)
- `TODO/completed/395_final_synthesis.md` (~580 LOC 7-9 sections — comprehensive final synthesis v34)

**Source artifacts (per Wf-395-A1+A2+A3):**
- `molmetal/scripts/wf395a1_evaluator_audit.py` (NEW ~370 LOC; Wf-395-A1)
- `molmetal/tests/test_wf395a1_evaluator_audit.py` (NEW; 4 NEW tests PASS)
- `molmetal/reports/wf395a1_evaluator_audit/audit.json` (per-evaluator record)
- `TODO/02_metrics/EVALUATORS_MIGRATED.md` (modified; 2026-09-25 R17 → 2026-09-30 Wf-395-A1)
- `docs/HONEST_NEGATIVE_CATALOG.md` (modified; 73 → 81 entries per Wf-395-A2 §6.18-§6.25)
- `molmetal/tests/test_wf395a2_catalog_size.py` (NEW; 1 NEW test PASS)
- `molmetal/tests/test_wf395a3_full_integration.py` (NEW ~520 LOC; 22 NEW tests PASS in 1.60 s)
- `molmetal/reports/wf395a3_full_integration/preflight_verify.json` (NEW ~600 bytes; Wf-385 gate at index 72/117)

**INDEX + AGENTS.md sync targets:**
- `TODO/INDEX.md` NEW `## §22.85 Wf-395 ultracode docs+code+test wave` entry (this sync — appends after §22.84)
- `AGENTS.md` §6 NEW row (this sync — appends after Wf-394 row at line 546)

**Prior INDEX syncs preserved unchanged:**
Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + Wf-362-B1 + Wf-363-C1 + Wf-364-C1 + Wf-365-C1 + Wf-366-C1 + Wf-367-C1 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-372-C1 + Wf-373-B1 + Wf-375-B1 + Wf-374-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-385-B1 + Wf-386-B1 + Wf-388-B1 + Wf-389-B1 + Wf-390-B1 + Wf-391-B1 + Wf-392-B1 + Wf-393-B1 + Wf-394-B1 prior INDEX syncs preserved unchanged.

**Standing directives cross-ref:**
- AGENTS.md §1 (MEASURED-vs-PROJECTED tag discipline)
- AGENTS.md §5.1 (Standing GPU authorization — 0 GPU launches performed)
- AGENTS.md §5.2 (Honest framing mandate per party-twenty-spirit §3.5)
- AGENTS.md §5.3 (No paper work without authorization — 0 `paper/*` touches)
- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)
- AGENTS.md §5.5 (Post-wave on-disk verification rule — MANDATE per Wf-384-B1)
- AGENTS.md §5.6 (Atomic write protocol adoption rule — MANDATE per Wf-386-B1)
- AGENTS.md §10 (Wetlab deprecation FAQ — 7 wetlab files DEPRECATED)
- party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)
- kill_then_adapt §0 (NO GPU launch + USER_OK gating)

**Wf-395 NEW ultracode recommendation:**
Wf-396+ IS the recommended follow-up per Wf-395-A1 §5 + Wf-395-A2 §6 + Wf-395-A3 §6 + Wf-394-A1 §6 + Wf-394-A2 §6 + Wf-394-A3 §6 + Wf-393-A1 §3.4 + Wf-393-A2 §4 + Wf-393-A3 §4.4 + Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-386-B1 §22.79.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + USER_OK-gated paper §6 update from `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + atomic write protocol adoption migration + Wf-392-A1 + Wf-392-A2 full training-loop integration + Wf-394-A1 Lambda CFM wire-in patch application + Wf-299 GPU launch (BLOCKED on `decode_ratio > 0`) + Wf-394-A3 Wf-299 pre-launch verification result handoff + Wf-395-A1 SCAFFOLD-tier 9 evaluators MEASURED-on-real promotion (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 v4 update + Wf-388 GPU retrain + Wf-394 follow-ups + Wf-395-A1 follow-ups total).

**User voice answer (per Wf-395-B1 §0 + Wf-395 final synthesis §0 — addressing user 2026-09-30 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*):**
**Wf-395 wave DID彻底解决 the 3 TODO-directory items that were not covered by Wf-378..394 ultracode waves** via 3 SHIPPED deliverables (Wf-395-A1 EVALUATORS_MIGRATED refresh 22/22 evaluators MEASURED-OK + 4 NEW tests PASS + 1 doc refresh + Wf-395-A2 HONEST_NEGATIVE_CATALOG expansion 73 → 81 entries with 8 NEW §6.18-§6.25 from Wf-388 + Wf-390 + Wf-392 wave + 1 NEW test PASS + Wf-395-A3 full-integration verification 22 NEW tests PASS in 1.60 s wall with Wf-385-A3 gate verified at `build_gates('all')` index 72/117) + B1 this docs-only INDEX + AGENTS sync + comprehensive final synthesis v34. Pre-flight 89/2/3/94 PRESERVED at Wf-394-B1 baseline (NOT re-run by Wf-395 — CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed. **They did NOT彻底解决** the structural/metric/env gates: m6_kappa + m2 R² + m1 AUC remain HONEST_NEGATIVE FINAL (per Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50; 9 paths exhausted); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 5 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK; Wf-379-A1 m1_auc 5-seed IN-FLIGHT result pending (~5 hr GPU); Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending; Wf-380 SCAFFOLD wire-in MEASURED verification pending (~21.5 hr GPU on Blackwell sm_120); Wf-392-A1 + Wf-392-A2 full training-loop integration pending (~50 LOC + 1-2 NEW tests each); Wf-394-A1 Lambda CFM wire-in patch application pending USER_OK; Wf-299 GPU launch BLOCKED on `decode_ratio > 0` upstream Wf-291D; Wf-395-A1 9 SCAFFOLD-tier evaluators require USER_OK-gated GPU retrain to be promoted to MEASURED-on-real; wetlab items all DEPRECATED per AGENTS.md §5.4 + §10. **NEW ultracode Wf-396+ IS the recommended follow-up** per Wf-395-A1 §5 + Wf-395-A2 §6 + Wf-395-A3 §6 + Wf-394 final synthesis §7 + Wf-393-A1 §3.4 + Wf-393-A2 §4 + Wf-393-A3 §4.4 + Wf-392-A2 §5.2 + Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-386-B1 §22.79.3 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + USER_OK-gated paper §6 update for m6_kappa drop-and-promote-alternatives recommendation + Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-392-A1 + Wf-392-A2 full training-loop integration + Wf-394-A1 Lambda CFM wire-in patch application + Wf-299 GPU launch (BLOCKED on `decode_ratio > 0`) + Wf-395-A1 SCAFFOLD-tier 9 evaluators MEASURED-on-real promotion + atomic write protocol adoption migration (~21.5 hr GPU + 10 min CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Wf-394 follow-ups + Wf-395 follow-ups total).

---

**END OF Wf-395-B1 INDEX + AGENTS + final synthesis v34 SYNC** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-395 SHIPPED within §22; Wf-396+ is the recommended follow-up per Wf-395 final synthesis §0 + Wf-394 final synthesis §7 + Wf-393 final synthesis §5 + Wf-391 final synthesis §7 + Wf-390 final synthesis §4.
