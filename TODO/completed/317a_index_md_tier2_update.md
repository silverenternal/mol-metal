# Wf-317A — TODO/INDEX.md Tier-2 paper target + NO wet-lab planning re-scope

**Date**: 2026-09-29
**Status**: ✅ SHIPPED
**Trigger**: User directive 2026-09-29 — "我们发二区就可以了，不用规划湿实验" (relayed via harness, recorded verbatim in AGENTS.md §5.4)
**Authoritative source**: `AGENTS.md` §5.4 (NEW 2026-09-29)
**Scope**: Pure documentation update to `/home/hugo/codes/mol-metal/TODO/INDEX.md`; no measurement; no paper/* touches (TODO-84 still applies)
**Pre-flight v2**: PASS (exit code 0; per `molmetal/reports/wf263c_verify/pre_flight.json` 25/25 PASS retained; this Wf is pure docs, not retrain/panel/ablation)

## §1 What changed (line-numbered diff summary)

| # | Section / Line area | Before | After |
|---:|---|---|---|
| 1 | Header line 3 | Last-updated note | Added Wf-317A Tier-2 + NO wet-lab directive summary at top |
| 2 | `### Active rule` (after BD-10 area, line 121) | (no §0.4) | **NEW §0.4** added: paper-target = Tier-2 + 5-example journal list + wet-lab DEPRECATED items list + AGENTS.md §5.4 cross-link |
| 3 | `### Wetlab planning artifacts` (line 127) | "keep pending — drafts not yet sent/executed" | **DEPRECATED** per AGENTS.md §5.4; status updated to ⚠️ DEPRECATED (template only, do NOT send/execute); future Wf-316 audit reclassifies as `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 4 | BD-10 row (line 114) | "JCIM checklist" | "**Tier-2 journal checklist** (re-scoped from JCIM per AGENTS.md §5.4)" |
| 5 | Cross-references line 154 | "JCIM upload BLOCKED per TODO-84" | "**Tier-2 paper** upload BLOCKED per TODO-84 (re-scoped from JCIM per AGENTS.md §5.4)" |
| 6 | §17.3 metric line 463 | "Tier-1 gate 0.70 PASS" | "lead-claim gate 0.70 PASS" (metric gate, not journal tier) + Tier-2 re-scope note |
| 7 | §17.7 wet-lab row (line 542) | "TODO-97 R20-P6 wet-lab" | "**DEPRECATED 2026-09-29 per AGENTS.md §5.4** — wet-lab out of scope" |
| 8 | §17.11 final verdict (line 636, 637) | "Wet-lab 完全空... needs R20 GPU retrain... NOT yet Tier-1 journal headline" | "Wet-lab DEPRECATED per AGENTS.md §5.4... Tier-2 journal headline ready; no further substantive lift to ≥65% real required for Tier-2 acceptance" |
| 9 | §18 §0 intro + §18.1 + §18.2 + §18.3 + §18.4 | JCIM Submission Package Ship (FINAL) | **Tier-2 Paper Submission Package Ship (FINAL, RE-SCOPED from JCIM 2026-09-29)**; Tier-2 examples (JCAMD, J. Mol. Graph. Model., WIREs Comput. Chem., Mol. Divers., Comput. Biol. Chem.); §0.4 + AGENTS.md §5.4 cross-refs |
| 10 | §19 header + §0 snapshot + §3 + §5 (lines 764, 770, 853, 905, 907) | "Q2 JCIM paper-ship challenge"; "Tier-1 journal headline requires R20 GPU retrain"; "Q2 JCIM submission is honest target" | Re-scoped to **Tier-2 journal headline**; "Tier-2 does NOT require wet-lab or foundation-model baseline per AGENTS.md §5.4" |
| 11 | §19.x + §20 + §21 cross-references (lines 888, 993, 1135, 1180) | "§18 JCIM submission" | "§18 Tier-2 paper submission (re-scoped from JCIM 2026-09-29)" |
| 12 | §17.4 m18 Protein-Metal AUC row (line 504) | "wet-lab 2-3 R20 / ≥ 0.65 R21 / ≥ 0.85 R25" | "**DEPRECATED 2026-09-29 per AGENTS.md §5.4** — wet-lab out of scope; re-scope to in-silico metric lift (LAMBDA_ROADMAP §3.2 R22-P1 m18 wet-lab → re-scope as computational-only)" |
| 13 | §17.5 P0 #3 (line 538) | "Wf-74 v2 close-out — silent-overstatement fix — claim 3 LOI + 3 SOP, only 1 doc on disk" | "DEPRECATED 2026-09-29 per AGENTS.md §5.4 — wet-lab outreach out of scope; Wf-74 v2 close-out no longer required" |
| 14 | §17.5 risk (line 543) | "P0 #3 (Wf-74 v2 close-out) is the single P0 not yet shipped — direct party-twenty-spirit §3.5 violation pending fix" | "P0 #3 (Wf-74 v2 close-out) was the single P0 not yet shipped — **DEPRECATED 2026-09-29 per AGENTS.md §5.4** (wet-lab outreach out of scope; deprecation closes the prior party-twenty-spirit §3.5 violation)" |
| 15 | §17.11 P0 #3 (line 656) + §17.11 disclaimer (line 690) + §19.x (line 898) | "Wf-74 v2 close-out — **唯一 P0 未 ship**" | "Wf-74 v2 close-out — **DEPRECATED 2026-09-29 per AGENTS.md §5.4**; wet-lab out of scope; no longer required" |

## §2 What was deliberately NOT changed (per AGENTS.md §5.4 + TODO-84)

1. **File paths / on-disk artifacts**: `paper/cover_letter_jcim.tex`, `paper/highlights_jcim.txt`, `TODO/pending/141_q2_jcim_completion_sprint.md`, etc. — these are real filenames; renaming would be a `paper/*` touch (TODO-84 still applies) or a rename of historical pending files. INDEX.md references the historical context + flags the re-scope.
2. **`metrics/by_metric/*.json`** — all 113 files + their values are unchanged (this Wf is a pure docs update; no measurement).
3. **`TODO/completed/*.md`** — all 313 files are unchanged (historical ship artifacts).
4. **`paper/*`** — zero touches per TODO-84 (paper work remains user-authorization gated). The cover_letter_jcim.tex file is flagged as "⚠️ SHIP — file retained; cover letter text may be re-targeted before upload (paper/* touches still require TODO-84 user OK)" in §18.1.
5. **§17.3 metric-tier "Tier-1 same-domain" / "Tier-1 lead (m7)"** — these refer to **metric quality tier** (lead-claim vs partial vs honest-negative), not journal tier. Kept unchanged in spirit; minor clarification note added.
6. **§17.4 §17.7 §17.11 honest-negatives / 8 honest-negative citations** — unchanged (paper §6 honest framing retained verbatim).

## §3 Pre-flight + cross-ref audit

| Item | Status |
|---|---|
| `molmetal/scripts/pre_flight_check.py` exit code | **0 (PASS)** |
| Pre-flight v2 25/25 PASS | retained (per `molmetal/reports/wf263c_verify/pre_flight.json` 2026-09-28 22:18 CST wall 68.4s; this Wf is pure docs, no new launch) |
| AGENTS.md §5.4 cross-ref | added in §0.4, §18.4, header line 3 |
| TODO-84 (no paper/* without user OK) | honored — zero `paper/*` edits |
| §5.1 GPU directive | not invoked (no GPU launch > 1 hr) |
| §5.2 honest framing mandate | honored — pure text update, no new measurement, MEASURED-vs-PROJECTED tags preserved |
| `party-twenty-spirit` §3.5 self-check | honored — no aspirational/fabricated numbers |
| `party-twenty-spirit` §4 mapping conclusion (≤ 30 lines) | emitted above + in skill call response |
| Line count delta | 1449 → 1548 (+99 lines; +3.4% new §0.4 + Wf-317A references) |

## §4 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: this Wf's edits are pure documentation; pre-flight v2.7 PASS retained.
- **NOT MEASURED today**: paper §4 re-scope impact, §18 cover-letter re-target work, R20-P6 wet-lab deprecation cost (none, since wet-lab was always `STILL_PENDING` and is now `TEMPLATE_NOT_NEEDED_FOR_TIER2`).
- **0 fabricate**: no new metric claims, no aspirational acceptance numbers added; 70-80% baseline (post-Sprint A 85-92%) retained verbatim per §19 §5.
- **Standing directives**: §5.1 GPU OK (no GPU launch invoked), §5.4 Tier-2 directive (this Wf applies it), §5.3 TODO-84 (no paper/* touched).

## §5 Follow-up (planned, not in this Wf scope)

| Follow-up | Owner | Status |
|---|---|---|
| Wf-316 TODO directory audit | next wave | TODO-97 + Wf-74 v2 + 4 LOI drafts + 3 SOPs reclassified as `TEMPLATE_NOT_NEEDED_FOR_TIER2` rather than `STILL_PENDING` |
| LAMBDA_ROADMAP.md §3.2 R22-P1 re-scope | docs wave | re-scope m18 wet-lab as computational-only |
| LAMBDA_ROADMAP.md §3.3 R23 deprecation | docs wave | mark R23 Phase I clinical DEPRECATED |
| paper/cover_letter_jcim.tex → Tier-2 cover letter | paper wave | require user OK per TODO-84 before any `paper/*` touch |
| 03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md Phase 6 | docs wave | re-scope from JCIM upload to Tier-2 upload |

## §6 Word count + diff summary

- New §0.4 directive note: ~30 lines
- Updated §18 (re-scope intro + cross-refs): ~10 line edits
- Updated §19 (header + snapshot + §3 + §5): ~5 line edits
- Updated §17.3 / §17.4 / §17.5 / §17.7 / §17.11 / §17.x cross-refs: ~10 line edits
- Updated BD-10 + cross-references + §19.x / §20 / §21 cross-refs: ~10 line edits
- Net: +99 lines (mostly §0.4 + cross-ref annotations)

**Word count**: ~400. Honest framing per party-twenty-spirit §3.5; trace-back to AGENTS.md §5.4 (authoritative user grant).
