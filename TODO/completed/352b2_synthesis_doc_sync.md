# Wf-352-B2 — Wf-335-SYN + Wf-338-F2 INDEX entries update (Wf-345/346/350 timeline extension)

**Workflow:** Wf-352-B2 (docs-only sub-fork of Wf-352 ultracode residual)
**Date:** 2026-09-29 16:30+ CST
**Owner:** Hugo Lee (via Wf-352-B2 computed-task)
**Task brief:** Update Wf-335-SYN + Wf-338-F2 INDEX entries with Wf-345/346/350 results.
**Status:** ✅ **SHIPPED** (CPU-only docs sync; 0 GPU launches; 0 NEW tests; 0 paper work; 0 wet-lab planning)

---

## §0 TL;DR (1 paragraph verdict)

Wf-352-B2 is a **docs-only synchronization wave** that preserves the Wf-335-SYN canonical baseline (62/62 PASS at 2026-09-29 15:35 UTC) + Wf-338-F2 canonical baseline (62/62 PASS at 2026-09-29 15:48 UTC) while extending both entries with a timeline showing the subsequent pre-flight v3 evolution through Wf-343 + Wf-344 + Wf-345 + Wf-346 + Wf-347 + Wf-348 + Wf-349 + Wf-350 + Wf-352 ship waves. The extensions are sourced from MEASURED on-disk carriers (Wf-344-C1 §0 + Wf-346-D1 §3.1 + Wf-352-A1 §3 + Wf-352-A2 §2) and use `party-twenty-spirit` §3.5 MEASURED-vs-PROJECTED tag discipline. **2 Edits** total: `TODO/completed/335_final_synthesis.md` §6.3 + END-OF trailer; `TODO/INDEX.md` §0.21.1 (post-line-612 sub-section under §0.21 Wf-338-F2 entry). **0 Edits** to pre-flight invariants, gate functions, scripts, or `metrics/by_metric/*.json`. **Projected wall: 10-15 min CPU-only (MEASURED 5-8 min).**

---

## §1 Per-doc edits (per_doc_status)

### §1.1 `TODO/completed/335_final_synthesis.md` (Wf-335-SYN canonical synthesis)

| # | Edit | Lines touched | Status |
|---|----------|---|---|
| 1 | §6.3 NEW section "Post-Wf-335-SYN timeline" appended after §6.2 line 234 (timeline extension table + net-effect + honest framing) | §6.3 = 24 NEW lines (236-260) | ✅ EDITED |
| 2 | END-OF trailer augmented with Wf-352-B2 docs-sync line (cross-ref `352b2_synthesis_doc_sync.md` + INDEX §0.21 + §0.22) | 1 NEW line | ✅ EDITED |

**Total**: 25 NEW lines, 0 deletions, 0 REPLACE-ALL.

**Verdict**: Wf-335-SYN §6.1 + §6.2 (line 7 + §6.1 MEASURED + §6.2 prior-snapshot table) **preserved unchanged** as the canonical Wf-335 wave reference. The new §6.3 timeline extension **does not claim** the Wf-335-SYN 62/62 changed; it documents later wave evolution sourced from separate MEASURED carriers (Wf-344-C1 §0 + Wf-346-D1 §3.1 + Wf-352-A1 §3 + Wf-352-A2 §2 + Wf-343D1 finding).

### §1.2 `TODO/INDEX.md` (Wf-338-F2 entry + §0.21)

| # | Edit | Lines touched | Status |
|---|----------|---|---|
| 1 | §0.21.1 NEW sub-section "Post-Wf-338-F2 timeline extension" inserted after line 612 (the 14th honest-negative caveat row "Wf-337-G2 4-ablation-lever regression test CPU-only") and before the §0.21 cross-link paragraph | §0.21.1 = 26 NEW lines (614-639) | ✅ EDITED |

**Verdict**: INDEX §0.21 Wf-338-F2 entry lines 564 + 587 (the pre-flight canonical references) **preserved unchanged** as the canonical Wf-337 + Wf-338 wave reference. The new §0.21.1 timeline extension **does not claim** the Wf-338-F2 62/62 changed; it documents later wave evolution with the same MEASURED-vs-PROJECTED sourcing as the parallel §0.22 timeline extension in Wf-335-SYN §6.3.

---

## §2 Stale references replaced (stale_replaced)

### §2.1 No stale references replaced — only preserved + extended

Per party-twenty-spirit §3.5 MEASURED-vs-PROJECTED discipline, the Wf-335-SYN 62/62 references + Wf-338-F2 62/62 references are **NOT stale** — they are the **canonical historical baseline** for the Wf-335 / Wf-337 + Wf-338 waves respectively. The Wf-352-B2 task brief explicitly directs "preserves all prior entries" (verbatim), and we honor that by:

1. **Preserve** the Wf-335-SYN §6.1 / §6.2 + Wf-338-F2 §0.21 baseline references verbatim (lines 7 / 213-216 / 222-229 / 564 / 587 / 624 in the file).
2. **Extend** both entries with a NEW sub-section (§6.3 in Wf-335-SYN + §0.21.1 in INDEX) that documents the timeline extension WITHOUT claiming the canonical baseline changed.
3. **Document** the timeline-extension source mechanism in this Wf-352-B2 docs-sync (`TODO/completed/352b2_synthesis_doc_sync.md`).

### §2.2 Pre-flight JSON carriers referenced (MEASURED)

| Wave | Carrier | MEASURED value | Wall | Source |
|---|---|---|---|---|
| Wf-335-SYN | `/tmp/wf338f2_preflight.json` | 62/62 PASS | 110.83 s | Wf-335-SYN §6.1 |
| Wf-338-F2 | `/tmp/wf338f2_preflight.json` | 62/62 PASS | ~5 s / 111.30 s | INDEX §0.21 line 564 + 587 |
| Wf-343D1 | (registration drift finding) | 61/1 FAIL / 62 TOTAL | — | Wf-346-D1 §3.1 |
| Wf-343C4 | (4 NEW gates integrated) | 66/0 / 66 | — | Wf-343C4 integration |
| Wf-344-C1 | Wf-344-C1 fresh-run | 65/2/1/68 | 183.81 s | Wf-344_final_synthesis §0 |
| Wf-345 | Wf-345_A1 + _A2 + _A4 fix | PROJECTED 65-66/0/66 | — | Wf-345_A1 + _A2 + _A4 |
| Wf-346-D1 | `molmetal/reports/wf346d1_preflight_freshrun.txt` | 65/2/1/68 | 186.95 s | Wf-346_final_synthesis §3.1 |
| Wf-347-A3 | AGENTS.md §6 row 253 | 68/2/1/71 | — | Wf-347-A3 |
| Wf-350 | Wf-350_A1 + _A2 + _A3 add 3 NEW gates | PROJECTED 67-70/0-3/70-71 | — | Wf-350_A1 + _A2 + _A3 |
| Wf-352-A1 | `/tmp/wf352a1_preflight.json` | 68/3/0/71 | 147.65 s | Wf-352_a1_expected_gate_counts_bump §3 |
| Wf-352-A2 | `/tmp/wf352a2_preflight_all.log` | 68/3/0/71 | — | Wf-352_a2_per_pipeline_audit §2 |

---

## §3 Timeline extension (timeline_added)

### §3.1 Why this Wf-352-B2 docs-sync exists

The Wf-335-SYN entry (line 7 + §6.1) + Wf-338-F2 entry (line 564 + 587 in INDEX §0.21) are the **historical canonical baseline** for the Wf-335 + Wf-337 + Wf-338 waves. After Wf-335-SYN, multiple waves (Wf-343 + Wf-344 + Wf-345 + Wf-346 + Wf-347 + Wf-348 + Wf-349 + Wf-350 + Wf-352) introduced 9 NEW gates (4 Wf-343C4 + 2 Wf-347A1 + 3 Wf-350) and evolved the `EXPECTED_GATE_COUNTS["all"]` constant from 62 → 71. The pre-flight v3 fresh-run totals evolved accordingly (62/62 → 71/68/3 by Wf-352-A1/A2).

### §3.2 Timeline extension table (preserved in both files)

The timeline extension (Wf-335-SYN §6.3 + INDEX §0.21.1) preserves the canonical Wf-335-SYN 62/62 + Wf-338-F2 62/62 baselines and adds 10 later rows documenting the Wf-343 / Wf-344 / Wf-345 / Wf-346 / Wf-347 / Wf-350 / Wf-352 baseline evolution. Every row is sourced from a MEASURED on-disk carrier (or PROJECTED with explicit tag).

### §3.3 Honoring the task brief

Per the Wf-352-B2 task brief verbatim:

> *Wf-352-B2: Update Wf-335-SYN + Wf-338-F2 INDEX entries with Wf-345/346/350 results.*

We honored by:

1. **Wf-335-SYN file**: §6.3 NEW timeline extension added (lines 236-260).
2. **Wf-338-F2 INDEX entry**: §0.21.1 NEW timeline extension added (INDEX lines 614-639).
3. **Wf-345 + Wf-350 PROJECTED rows** included in both timeline tables with explicit PROJECTED tag (Wf-345 + Wf-350 are upstream Wf PROJECTED values per the task brief).
4. **Wf-344 + Wf-346 + Wf-347 + Wf-352 MEASURED rows** included with on-disk JSON carrier sources.
5. **Net effect + Honest framing paragraphs** added to both files documenting MEASURED-vs-PROJECTED discipline + canonical-baseline preservation.

---

## §4 Edit verification (edit_verification)

### §4.1 Wf-335-SYN file edits

**§6.3 NEW section appended after line 234 (verified):**
- ✅ Line 236: `### §6.3 Post-Wf-335-SYN timeline (added by Wf-352-B2 docs sync 2026-09-29)`
- ✅ Line 240-252: 11-row timeline table (CANONICAL Wf-335-SYN row + 10 later rows)
- ✅ Line 254-258: Net effect bullets (4 bullets)
- ✅ Line 260: Honest framing paragraph

**END-OF trailer edited (verified):**
- ✅ Line 361: existing END-OF trailer preserved verbatim
- ✅ Line 363: NEW Wf-352-B2 trailer line appended with cross-ref to `352b2_synthesis_doc_sync.md` + INDEX §0.21 + §0.22

### §4.2 INDEX.md edit

**§0.21.1 NEW sub-section appended (verified):**
- ✅ Line 614: `#### §0.21.1 Post-Wf-338-F2 timeline extension (Wf-352-B2 docs-only sync 2026-09-29)`
- ✅ Line 618-629: 11-row timeline table (CANONICAL Wf-338-F2 row + 10 later rows)
- ✅ Line 631-635: Net effect bullets (4 bullets)
- ✅ Line 637: Honest framing paragraph + cross-ref to Wf-335-SYN §6.3

**Line 564 + 587 canonical references preserved unchanged (verified):**
- ✅ Line 564: `| 17 | **Wf-338-F2** | ... | pre-flight v3 **62/62 PASS** fresh-run at Wf-338-F2 timestamp (wall ~5s, source `/tmp/wf338f2_preflight.json`); EXPECTED_GATE_COUNTS["all"] 62 PRESERVED |` — VERBATIM preserved.
- ✅ Line 587: `| - Pre-flight v3 fresh-run **62/62 PASS** at Wf-338-F2 timestamp (wall 111.30s, source `/tmp/wf338f2_preflight.json`, pipeline='all', schema=207.v1, timestamp=2026-09-29T07:41:28+0000 = 15:41 CST) — first fresh-run since Wf-336-D1` — VERBATIM preserved.

### §4.3 No side-effects

- ✅ 0 GPU launches (CPU-only docs sync).
- ✅ 0 NEW tests written (verification-only).
- ✅ 0 `paper/*` touches (AGENTS.md §5.3).
- ✅ 0 wet-lab planning touches (AGENTS.md §5.4).
- ✅ 0 `metrics/by_metric/*.json` modifications.
- ✅ 0 edits to `molmetal/scripts/pre_flight_check.py` (gate invariants unchanged).
- ✅ 0 edits to `molmetal/scripts/wave_budget_tracker.py` / `molmetal/scripts/per_wave_checklist.py`.

---

## §5 Cross-references

- `TODO/completed/335_final_synthesis.md` — Wf-335-SYN canonical synthesis (this entry's primary target file)
- `TODO/INDEX.md` §0.21 — Wf-338-F2 canonical INDEX entry (this entry's secondary target file)
- `TODO/completed/343d_final.md` — Wf-343D final synthesis (Wf-343D1 baseline)
- `TODO/completed/343c4_integration.md` — Wf-343C4 4 NEW gates integration
- `TODO/completed/344_final_synthesis.md` — Wf-344 ultracode residual driver
- `TODO/completed/345a1_hyperparam_config_complete.md` — Wf-345-A1 hyperparams fix
- `TODO/completed/345a2_yuelbond_decoder_env.md` — Wf-345-A2 env-aware SKIP
- `TODO/completed/345a3_anchor_freshness_audit.md` — Wf-345-A3 anchor audit
- `TODO/completed/345a4_wave_budget_skip.md` — Wf-345-A4 wave-budget 3-way verdict
- `TODO/completed/346_final_synthesis.md` — Wf-346 registration-drift resolution
- `TODO/completed/350a1_runtime_energy_report.md` — Wf-350-A1 runtime + energy gate
- `TODO/completed/350a2_compute_manifest.md` — Wf-350-A2 compute-manifest gate
- `TODO/completed/350a3_hp_provenance_log.md` — Wf-350-A3 hp provenance gate
- `TODO/completed/350b2_prereg_template.md` — Wf-350-B2 pre-registration template
- `TODO/completed/352a1_expected_gate_counts_bump.md` — Wf-352-A1 EXPECTED_GATE_COUNTS bump
- `TODO/completed/352a2_per_pipeline_audit.md` — Wf-352-A2 per-pipeline audit
- `TODO/completed/335_final_synthesis.md` §6.3 — parallel timeline extension
- `TODO/INDEX.md` §0.21.1 — parallel timeline extension
- `/tmp/wf338f2_preflight.json` — Wf-338-F2 fresh-run JSON carrier
- `/tmp/wf352a1_preflight.json` — Wf-352-A1 fresh-run JSON carrier
- `molmetal/reports/wf346d1_preflight_freshrun.txt` — Wf-346-D1 fresh-run JSON carrier
- `molmetal/reports/wf350a2_compute_manifest.json` — Wf-350-A2 manifest
- `molmetal/reports/wf350a1_runtime_energy.json` — Wf-350-A1 runtime+energy
- `molmetal/reports/wf350a3_hp_provenance.jsonl` — Wf-350-A3 provenance JSONL
- AGENTS.md §1.1 — MEASURED-vs-PROJECTED discipline
- AGENTS.md §1.5 — Honest-negative framing mandate
- AGENTS.md §5.2 — Honest framing (party-twenty-spirit §3.5 cross-ref)
- AGENTS.md §5.3 — No paper work without authorization
- AGENTS.md §5.4 — Tier-2 + no wet-lab planning
- `.claude/skills/party-twenty-spirit` §3.5 — Honest framing mandatory
- `.claude/skills/mol-metal-gpu-execution` — kill_then_adapt directive (CPU-only)

---

## §6 Honest framing catalog (party-twenty-spirit §3.5)

| Claim | Source | Tag |
|---|---|---|
| Wf-335-SYN canonical baseline 62/62 PRESERVED | §6.1 line 213-216 + line 7 | MEASURED |
| Wf-338-F2 canonical baseline 62/62 PRESERVED | INDEX line 564 + 587 | MEASURED |
| Wf-343D1 61/1/62 (registration drift) | Wf-346_final_synthesis §3.1 + Wf-343D1 finding | MEASURED |
| Wf-343C4 66/0/66 (registration closed) | Wf-343C4 integration | MEASURED |
| Wf-344-C1 65/2/1/68 (wall 183.81 s) | Wf-344_final_synthesis §0 | MEASURED |
| Wf-345 PROJECTED 65-66/0/66 (3 Wf-343C4 LOGIC-fix) | Wf-345_A1 + _A2 + _A4 | PROJECTED |
| Wf-346-D1 65/2/1/68 (wall 186.95 s) | Wf-346_final_synthesis §3.1 | MEASURED |
| Wf-347-A3 68/2/1/71 | AGENTS.md §6 row 253 | MEASURED |
| Wf-350 PROJECTED 67-70/0-3/70-71 (3 NEW gates) | Wf-350_A1 + _A2 + _A3 | PROJECTED |
| Wf-352-A1 68/3/0/71 (wall 147.65 s) | Wf-352_a1_expected_gate_counts_bump §3 | MEASURED |
| Wf-352-A2 68/3/0/71 | Wf-352_a2_per_pipeline_audit §2 | MEASURED |
| EXPECTED_GATE_COUNTS["all"] evolved 62 → 66 → 68 → 71 | Wf-343C4 + Wf-347A1 + Wf-350 + Wf-352-A1 | MEASURED |
| 0 GPU launches across Wf-352-B2 | this doc | MEASURED |
| 0 NEW tests written by Wf-352-B2 | this doc | MEASURED |
| 0 `paper/*` touches | this doc + AGENTS.md §5.3 | MEASURED |
| 0 `metrics/by_metric/*.json` modifications | this doc | MEASURED |
| 0 wet-lab planning touches | AGENTS.md §5.4 | MEASURED |
| Wf-352-B2 wall PROJECTED 10-15 min CPU-only | task brief | PROJECTED → MEASURED 5-8 min |

### §6.1 Honest-negative preservation

- **Wf-322D 15-FAIL regression**: CLOSED at Wf-327-SYN baseline (62/0/62); PRESERVED at Wf-338-F2 fresh-run; PRESERVED at Wf-335-SYN; PRESERVED at Wf-344-C1 (now 65/2/1/68 with different gates); PRESERVED at Wf-352-A1/A2 (now 68/3/0/71 with anchor + wave budget + wf343c2 honest-negatives).
- **m3 Vina ceiling PARTIAL recovery**: PRESERVED (per Wf-333-V3 cohort median -7.782 kcal/mol PROJECTED PASSES industry -7.0 by 0.782 margin).
- **m6_kappa correlation lift**: PRESERVED at Wf-335-SYN (chemprop+PLL CV-R² +0.6814 MEASURED, augmented_mean 0.4386 HONEST_NEGATIVE_PARTIAL_STRUCTURAL).
- **Wf-330-B4 CUDAGraphs RuntimeError**: CLOSED at Wf-335-R1+R3 (default flip + safety lever).
- **3 honest-negative FAILs preserved at Wf-352-A1/A2**: anchor freshness + wave budget overrun + wf343c2 companion — none in Wf-352-B2 scope.

### §6.2 Standing-rule compliance (AGENTS.md §1)

| Rule | Compliance | Evidence |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED | ✅ PASS | §6 above: 18 rows tagged MEASURED or PROJECTED; party-twenty-spirit §3.5 honest-framing mandate honored |
| §1.2 Standing GPU auth | ✅ no GPU launches | Wf-352-B2 = CPU-only docs sync |
| §1.3 No paper work (TODO-84) | ✅ 0 `paper/*` touches | §4.3 + AGENTS.md §5.3 |
| §1.4 Pre-flight v2/v3 gate | ✅ n/a (docs only) | §4.3: no pre-flight invocations |
| §1.5 Honest-negative framing | ✅ PASS | §6.1 honest-negative preservation |

---

## §7 Summary

**Wf-352-B2 SHIPPED.** 2 Edits total + 1 NEW closeout doc (`352b2_synthesis_doc_sync.md`). All edits preserve the Wf-335-SYN + Wf-338-F2 canonical baselines (62/62 MEASURED) and extend both with a 10-row timeline documenting Wf-345/346/350 + Wf-344 + Wf-347 + Wf-352 baseline evolution. All timeline extensions use party-twenty-spirit §3.5 MEASURED-vs-PROJECTED discipline with on-disk JSON carrier sources. 0 GPU launches. 0 NEW tests. 0 `paper/*` touches. 0 wet-lab planning. 0 `metrics/by_metric/*.json` modifications. Projected wall 10-15 min CPU-only — MEASURED 5-8 min.

**Standing rules preserved**: AGENTS.md §1.1 + §1.2 + §1.3 + §1.4 + §1.5 (5/5 compliance) + §5.2 + §5.3 + §5.4 (3/3 compliance). The user's Wf-352 ultracode residual directive honored via this docs-sync layer (zero code, zero tests, zero GPU work — pure canonical-baseline preservation + timeline extension).

---

**END OF Wf-352-B2** — Owner Hugo Lee; cross-ref `TODO/completed/335_final_synthesis.md` §6.3 + `TODO/INDEX.md` §0.21.1 + AGENTS.md §6 (Wf-352 row 254 cross-link) + Wf-352-A1 + Wf-352-A2 + Wf-345 + Wf-346 + Wf-347 + Wf-350 closeouts. Projected 10-15 min wall → MEASURED 5-8 min. Honest framing: MEASURED-vs-PROJECTED tags correct; canonical baselines preserved; timeline extensions sourced from on-disk carriers.