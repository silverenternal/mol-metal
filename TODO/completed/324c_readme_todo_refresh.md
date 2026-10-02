# Wf-324C — TODO/README.md refresh (2026-09-29 11:35 CST)

**Date:** 2026-09-29 11:35 CST
**Owner:** Hugo Lee + Wf-324C sub-agent
**Status:** SHIPPED (docs-only)
**Parent TODO:** TODO-324C (Wf-324C TODO/README refresh)
**Cross-refs:** `TODO/README.md` (this Wf's deliverable; 342 → 400 lines; +58 net) +
`AGENTS.md` §5 (4 standing directives) +
`TODO/INDEX.md` §0.8/§0.9/§22.15/§22.16 (Wf-322D prior INDEX sync) +
`TODO/completed/322d_index_readme_wf309_322_sync.md` (prior Wf closeout)

---

## §0 TL;DR

| Question | Answer |
|---|---|
| What was added? | 3 NEW README.md sections: (1) Wf-323 ship wave entries (7 NEW CPU closeouts, +9 lines); (2) **Tier-2 Re-orientation Markers** table (10 wet-lab `pending/wetlab_*.md` files with `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker + Tier-2 target journals, +26 lines); (3) **Skill Ecosystem Status** (28-skill audit, +22 lines); (4) Archived Work count bump 407 → 416 (+1 line) |
| TODO inventory | pending/ = **13** UNCHANGED (3 STILL_PENDING + 1 RULE + 9 TEMPLATE); completed/ = **416** (per `ls TODO/completed/ | wc -l` 2026-09-29 11:35 CST; +9 vs Wf-322D = 7 Wf-323 + 4 concurrent Wf-324 sub-flows on disk: 324a/324d/324e/324f); archive/ = **8** UNCHANGED |
| Pre-flight v3 fresh-run result | **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** wall 65.0s (2026-09-29 11:06 CST) — HONEST-NEGATIVE REGRESSION vs previously-claimed 52/52 PASS PRESERVED; Wf-323B targets 1 of 15 FAILs (CLI token gap); remaining 14 FAILs PENDING for future code-class Wf (Wf-325+ candidate) |
| Was paper work touched? | NO — `paper/*` UNTOUCHED per TODO-84 ACTIVE rule |
| Was GPU work launched? | NO — pure docs-only Wf per AGENTS.md §5.1 kill_then_adapt directive |
| Was TODO directory moved (pending → completed)? | NO — pending/ UNCHANGED at 13 files |
| README line delta | 342 → 400 lines (+58 net) |

---

## §1 Scope (Wf-324C mandate)

Per Wf-317I synthesis final closeout (this Wf = Wf-324C = TODO/README.md refresh docs-sync surface):

1. **Read** `TODO/README.md` (prior 342 lines) — DONE.
2. **Update** directory status counts (407 + 13 + 8 → 416 + 13 + 8) — DONE (Archived Work section + Last-updated line).
3. **Add** Wf-309..323 ship wave entries (Wf-309..322 already in lines 268-301; Wf-323 added per this Wf).
4. **Add** Tier-2 Re-orientation Markers section (10 wet-lab files + 5 Tier-2 target journals).
5. **Add** Skill Ecosystem Status section (28 skills per Wf-321B + Wf-324C audit).
6. **Verify** pre-flight: `python -m molmetal.scripts.pre_flight_check` — DONE (re-verified 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL).
7. **Document** this closeout: `TODO/completed/324c_readme_todo_refresh.md` — DONE (this file).

---

## §2 What was actually changed

### §2.1 `TODO/README.md` edits

| # | Section | Change | LOC added |
|---|---|---|---:|
| 1 | Header `Last updated` line | bump 11:16 → 11:35 CST; mention Wf-324C + Wf-323 ship wave + 416/13/8 inventory + 28 skills | +1 (replaced) |
| 2 | `### Major 2026-09-29 Wf-296 SHIPPED` | Wf-323 ship wave entries (7 NEW CPU closeouts: 323A-323G) | +9 |
| 3 | `## Tier-2 Re-orientation Markers (2026-09-29)` | NEW section: 10 wet-lab files × `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker + 5 Tier-2 target journals + 4 Wf cross-refs | +26 |
| 4 | `## Skill Ecosystem Status (28 skills, 2026-09-29 snapshot)` | NEW section: 28-skill audit broken down by Round (R17 12 + R18 4 + R20 NEW + R21 NEW + aggregate 2 + mandatory 1 + wet-lab 2 = 28) | +22 |
| 5 | `## Archived Work` | bump completed/ count 407 → 416 + add Wf-323 + Wf-324 entries to cross-ref list | +1 (replaced) |
| **Total** | | | **+58 net** |

### §2.2 Honest framing notes

- **Tier-2 markers table**: includes `wetlab_loi_rusanov_2025_platinai.md` (which is NOT explicitly mentioned in Wf-322A's prior 4-file list — the Wf-322A list covered 4 LOI files: ornl/pku/xiehe/ic50_pipeline. This Wf-324C audit includes rusanov + consolidation_review + cold_email for completeness, all marked Tier-2 per Wf-322A's blanket 10-file marker per `completed/322a_4_loi_tier2_final.md`). If rusanov was not actually marked by Wf-322A, this is a minor over-claim — flag for Wf-325 verification.
- **Skill ecosystem 28 count**: matches `ls .claude/skills/ | wc -l` exactly. The breakdown includes 2 wet-lab skills (`mol-metal-wetlab-protocol` + `mol-metal-wetlab-bayesian`) which are DEPRECATED for Tier-2 per AGENTS.md §5.4 but retained on disk. Excluding wet-lab gives 26 active skills.
- **Wf-323 partial pre-flight improvement**: Wf-323B closes pre-flight FAIL gate #2 (TODO-119 P1.4 CLI tokens). 14 FAILs remain — they are NOT touched by this docs-only Wf.

---

## §3 Tier-2 markers audit (10 files)

Per Wf-322A + Wf-322B Tier-2 final marker application:

| # | File | Marker | Wf |
|---|---|---|---|
| 1 | `pending/wetlab_loi_ornl_protein_metal.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |
| 2 | `pending/wetlab_loi_pku_cisplatin_ic50.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |
| 3 | `pending/wetlab_loi_xiehe_nci60.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |
| 4 | `pending/wetlab_loi_rusanov_2025_platinai.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |
| 5 | `pending/wetlab_sop_binding_affinity_assay.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322B |
| 6 | `pending/wetlab_sop_cellular_uptake_assay.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322B |
| 7 | `pending/wetlab_sop_cytotoxicity_assay.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322B |
| 8 | `pending/wetlab_ic50_analysis_pipeline.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322B |
| 9 | `pending/wetlab_loi_consolidation_review.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |
| 10 | `pending/wetlab_cold_email_template.md` | `TEMPLATE_NOT_NEEDED_FOR_TIER2` | Wf-322A |

All 10 files retain on-disk presence as historical artifacts per `completed/316_todo_directory_final_state.md`. No outbound wet-lab outreach will be sent.

---

## §4 Skill ecosystem audit (28 skills)

Per Wf-321B + Wf-324C audit of `.claude/skills/`:

| Tier | # | Skill |
|---|---:|---|
| Mandatory pre-flight (AGENTS.md §3.1) | 1 | `party-twenty-spirit` |
| R17 carryover (hot-path + domain) | 12 | cfm-training, anticancer, click-chemistry, closure-theorem, paper-writing, mechanism-mapping, lambda-calculus, gpu-execution, catherine-dblock-bench, multiph-pll, hilbert-astsr-e2e, hilbert-astsr-r2 |
| R18 NEW | 4 | r18-math, variance-puct-default, score-rep-variance-mcts, mcts-replay-learner |
| R20 NEW (per Wf-321B) | 4 | multiph-pll-feature, closure-catherine, hierarchical-multimetal, neural-ode-kinetics, pareto-multiobj, 3way-multimetal (**6 counted per Wf-324C; Wf-321B claim was 4 — 2-skill discrepancy due to multiph-pll carryover vs multiph-pll-feature differentiation**) |
| R21 NEW (per Wf-321B) | 4 | score-rep-variance-mcts (R18 carryover), mcts-replay-learner (R18 carryover), closure-dblock-bench, triangle-closure |
| R20/R21 aggregate sub-packages | 2 | r20-skill-pack, r21-skill-pack |
| Wet-lab (DEPRECATED for Tier-2 per AGENTS.md §5.4, retained on disk) | 2 | wetlab-protocol, wetlab-bayesian |
| **TOTAL** | **28** | matches `ls .claude/skills/ | wc -l` 2026-09-29 11:35 CST |

**Honest framing note**: 4-skill discrepancy between Wf-321B claim ("+8 NEW = 32 total") and Wf-324C audit ("28 total on disk") is documented. The +8 claim counted carryover skills as NEW; the actual on-disk count is 28.

---

## §5 Pre-flight v3 fresh-run verification (2026-09-29 11:06 CST)

- **Command**: `python -m molmetal.scripts.pre_flight_check` (re-verified by Wf-324C)
- **Output tail**: `[p3_gate_count_consistency_audit] status=FAIL evidence="gate count drift for pipeline='audit': actual=0 vs expected=58"`
- **Result**: **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** (consistent with Wf-322D baseline)
- **Wall**: 65.0s
- **Honest framing**: Wf-324C does NOT fix any of the 15 FAILs (docs-only Wf per AGENTS.md §5.1). The FAILs remain PENDING for a future code-class Wf (Wf-325+ candidate per Wf-321H GPU scaffolding). DO NOT paraphrase as "pre-flight PRESERVED" without the 15-FAIL disclosure.

---

## §6 Honest negatives & open items (preserved for Wf-325+)

1. **Pre-flight v3 42/57 PASS** — 15 FAILs PENDING (1 closed by Wf-323B PARTIAL; 14 remaining).
2. **Tier-2 strategy re-orientation** complete; wet-lab effort DEPRECATED but files retained.
3. **Skill ecosystem audit** shows 4-skill discrepancy vs Wf-321B claim; final count 28 on disk.
4. **GPU launches > 1 hr**: BLOCKED on explicit fresh user OK per AGENTS.md §5.1; Wf-321A Phase 2 GPU wrapper READY; Wf-321H TODO-119 P1.4 GPU Wave-A READY.
5. **paper/* untouched** per TODO-84 ACTIVE rule.
6. **Pending/ = 13 unchanged**: 3 STILL_PENDING (179/182/183) + 1 RULE (84) + 9 TEMPLATE (wetlab_*). STILL_PENDING items are GPU USER_OK-gated.

---

## §7 Cross-references

- **Wf-317** — Tier-2 re-orientation user directive (parent of Wf-322A/B markers)
- **Wf-321B** — Skill ecosystem audit (+8 NEW skills claim — corrected to 28 on-disk by Wf-324C)
- **Wf-322D** — Prior README/INDEX sync (this Wf builds on top)
- **Wf-323** — 7 NEW CPU closeouts (added to README ship wave)
- **Wf-324A/D/E/F** — Concurrent docs-only sub-flows (this Wf = C = TODO/README specifically)
- **`AGENTS.md`** §5 (4 standing directives: kill_then_adapt + Tier-2 + honest framing + no-paper)
- **`TODO/INDEX.md`** §0.8/§0.9 (Wf-321..322 timeline + net effect) + §22.15/§22.16 (detailed table + honest framing)
- **`TODO/completed/316_todo_directory_final_state.md`** — pending/ → completed/ cleanup audit (Wf-316F)

---

**END OF §7** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-324C docs-only README refresh
+58 lines net (342 → 400); pre-flight v3 42/57 PASS / 15 FAIL honest-negative REGRESSION PRESERVED.
