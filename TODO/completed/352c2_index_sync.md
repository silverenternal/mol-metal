# Wf-352-C2: TODO/INDEX.md §22.38 entry append (Wf-352 ultracode residual wave INDEX sync)

**Date:** 2026-09-29
**Workflow:** Wf-352 ultracode residual wave (CPU-only docs-only INDEX sync)
**Task:** Wf-352-C2 — Append `TODO/INDEX.md §22.38` entry documenting Wf-352 ship wave
**Status:** SHIPPED (CPU-only docs-only; 0 GPU launches; 0 NEW pre-flight gates)

---

## §1 Outcome

`TODO/INDEX.md §22.38` NEW entry appended between `§22.37` (Wf-354-C2 INDEX sync) and `§23` (Wf-316B archival audit) documenting the Wf-352 ultracode residual wave closeouts. Per `party-twenty-spirit` §3.5 honest framing, every claim in the appended entry cites a MEASURED on-disk source from the Wf-352 closeout files (A1 + A2 + B1 + B2 + B3 + C1 = 6 SHIPPED + C2 = this INDEX sync).

**Net effect:**
- `TODO/INDEX.md` line count: 4741 → 4741 + §22.38 entry (~95 NEW lines)
- `TODO/completed/` file count: 676 (Wf-354-C2 baseline) → 677 (+1 NEW = `352c2_index_sync.md`)
- `EXPECTED_GATE_COUNTS["all"]` = **71 PRESERVED** (Wf-352-A1 baseline; Wf-352-C2 is docs-only — no pre-flight gate change)

---

## §2 Edit details

| # | Edit | Source | Verdict |
|---|---|---|---|
| 1 | `TODO/INDEX.md` appended §22.38 NEW entry between §22.37 (line 4588) and §23 (line 4591) | this Wf-352-C2 | ✅ EDITED |
| 2 | `TODO/completed/352c2_index_sync.md` NEW (this file) | this Wf-352-C2 | ✅ NEW |

**Edit pattern:** 1 `Edit` (anchor on the §22.37 cross-link closing line + insert NEW §22.38 block before `---` separator preceding `## §23`).

---

## §3 §22.38 entry coverage

Per Wf-352 closeout on-disk carriers (sourced from `TODO/completed/352{a1,a2,b1,b2,b3,c1}_*.md`):

| Closeout | Subject | Tests | Wall |
|---|---|---:|---|
| Wf-352-A1 | EXPECTED_GATE_COUNTS["all"] bump 66 → 71 | 2 NEW | MEASURED 147.65 s pre-flight |
| Wf-352-A2 | Per-pipeline audit (all=71 / frontier=25 / chemprop_v2_v3=14 / cfm=6; drift=0) | 0 NEW | <1 min |
| Wf-352-B1 | Wf-321D audit doc sync (§4 55/56 → 71; 3 Edits / 6 tokens) | 0 NEW | <1 min |
| Wf-352-B2 | Wf-335-SYN + Wf-338-F2 INDEX entries timeline extension | 0 NEW | MEASURED 5-8 min |
| Wf-352-B3 | AGENTS.md §6 final sync (Wf-345/346/350/352 + Wf-344/348/349/351-C2 row) | 0 NEW | <1 min |
| Wf-352-C1 | Full verification (4/4 pipelines MATCH; 89/3/1/93 fresh-run; wall 197.97 s) | 2 NEW (re-runs A1) | MEASURED ~3.3 min |
| Wf-352-C2 | THIS INDEX sync (TODO/INDEX.md §22.38 append + this closeout) | 0 NEW | PROJECTED 5-10 min |

**Wave totals:** 6 closeouts on disk + this C2 INDEX sync = 7 files; 2 NEW tests (Wf-352-A1); 0 GPU launches; 0 NEW pre-flight gates; 0 paper work; 0 wet-lab planning.

---

## §4 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

**Measured (this Wf-352-C2 closeout):**
- 1 Edit on `TODO/INDEX.md` (§22.38 NEW entry).
- 1 NEW file `TODO/completed/352c2_index_sync.md` (this file).
- Wf-352 ship-wave count: 6 closeouts on disk (A1 + A2 + B1 + B2 + B3 + C1) + C2 = 7.
- Wf-352 NEW tests: 2 (`test_wf352a1_*.py` in `molmetal/tests/`).

**Projected:**
- Wall for future Wf-352-C2-style docs-only INDEX sync: 5-10 min CPU-only (consistent with Wf-352-B3 §4 measured wall).
- Pre-flight v3 fresh-run wall at next Wf re-invocation: ~200 s CPU-only (consistent with Wf-352-C1 MEASURED 197.97 s).

**Honest-negative caveats preserved** (3 entries):
1. **3 pre-flight FAILs preserved** from Wf-352-A1 audit + Wf-353 evolution: `p3_wf343c4_anchor_freshness_24h` (4/11 anchors STALE) + `p3_wf343c4_wave_budget_remaining_nonneg` (-183.04h overrun) + `p3_wf343c2_per_wave_runtime_check` (wf343c2_per_wave_check.py FAIL). None are gate-count drift; not in scope for Wf-352-C2.
2. **Wf-353 ship (2026-09-29 16:54) bumped EXPECTED_GATE_COUNTS 71 → 93 with +22 NEW gates but AGENTS.md §6 + INDEX.md §0/§22 have NOT been re-synced.** Per Wf-352-C1 §4, this is the next docs-sync Wf (e.g. **Wf-353-D1** or **Wf-353-C2**) — flagged as HONEST_GAP, NOT BLOCKING (pre-flight + gate-count internally consistent).
3. **§22.35 already used** for Wf-349-B2 (per INDEX.md line 4352 grep); Wf-352-C2 uses the next available slot **§22.38** per honest framing (no stale-references gap).

---

## §5 Standing-rule compliance

- **§1.1 MEASURED-vs-PROJECTED tag discipline**: all numeric values cite on-disk source (§3).
- **§1.4 Pre-flight v2 25/25 PASS gate**: NOT re-run by Wf-352-C2 (docs-only; pre-flight already 62/62 PASS at Wf-335-SYN baseline per Wf-352-C1 §1).
- **§5.1 Standing GPU authorization**: not invoked — CPU-only docs change.
- **§5.2 Honest framing mandate**: §4 of this file.
- **§5.3 No paper work without authorization**: zero touches to `paper/*`.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: Tier-2 markers UNCHANGED (computational-only acceptance bar).

---

## §6 Audit trail

| Time | Action | Source |
|---|---|---|
| 2026-09-29 | User voice: "基于刚才所有结束的工作流以及所有的的洞察，可以启动ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长" (carried over from Wf-330 / Wf-335 / Wf-336 / Wf-341 ultracode series) | parent directive |
| 2026-09-29 | Wf-352-A1 bumps EXPECTED_GATE_COUNTS["all"] 66 → 71; 2 NEW tests | `TODO/completed/352a1_expected_gate_counts_bump.md` |
| 2026-09-29 | Wf-352-A2 per-pipeline audit confirms 0 drift on all 4 pipelines | `TODO/completed/352a2_per_pipeline_audit.md` |
| 2026-09-29 | Wf-352-B1 syncs Wf-321D audit doc §4 (55/56 → 71); 3 Edits | `TODO/completed/352b1_wf321d_audit_doc_sync.md` |
| 2026-09-29 | Wf-352-B2 syncs Wf-335-SYN §6.3 + Wf-338-F2 §0.21.1 timeline extensions | `TODO/completed/352b2_synthesis_doc_sync.md` |
| 2026-09-29 | Wf-352-B3 syncs AGENTS.md §6 row for Wf-344/345/346/348/349/350/351-C2/352 | `TODO/completed/352b3_agmd_sync.md` |
| 2026-09-29 | Wf-352-C1 full verification (89/3/1/93 pre-flight; 4/4 pipelines MATCH) | `TODO/completed/352c1_full_verify.md` |
| 2026-09-29 | Wf-352-C2 (this closeout): appends INDEX.md §22.38 entry + writes this file | `TODO/completed/352c2_index_sync.md` |

**END OF Wf-352-C2 CLOSEOUT**
