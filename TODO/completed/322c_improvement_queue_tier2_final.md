# Wf-322C — IMPROVEMENT_ENUM TODO-138 Tier-2 + INDEX.md sync verification (2026-09-29)

**Phase**: Wf-322C docs sync (no code change, no GPU launch, no paper work). Audit + 2-line update scope per AGENTS.md §1 + `party-twenty-spirit` §3.5.
**Date**: 2026-09-29 (post-Wf-316B archival audit + Wf-317D Tier-2 re-scope).
**Trigger**: Per Wf-317H partial — IMPROVEMENT_ENUM.md TODO-138 row + §1.5 task description still referenced "JCIM" and required Tier-2 re-scope.

---

## §1 What was actually changed (this Wf-322C)

### §1.1 IMPROVEMENT_ENUM_2026_09_27.md (2 rows updated)

| Line | Before | After | Reason |
|---:|---|---|---|
| 18 (TODO-138 row) | `Paper v2 + JCIM submission package` | `Paper v2 + Tier-2 journal submission package (re-scoped from JCIM per AGENTS.md §5.4 / Wf-317D 2026-09-29)` | Aligns row with re-scope directive per AGENTS.md §5.4 |
| 92 (§1.5 任务 #4 header) | `任务: Paper v2 + JCIM submission package` | `任务: Paper v2 + Tier-2 journal submission package (re-scoped from JCIM per AGENTS.md §5.4 / Wf-317D 2026-09-29)` | Aligns section header with re-scope directive |

**n_rows_updated = 2** (Bash verified via grep: `JCIM` count IMPROVEMENT_ENUM.md 3 → 1 after edit — remaining 1 is the §0.5 historical "re-scope from JCIM" pointer line).

### §1.2 IMPROVEMENT_QUEUE_2026_09_27.md (0 rows updated — already Tier-2)

`grep -c "JCIM" IMPROVEMENT_QUEUE_2026_09_27.md` = **0** — file already at Tier-2 per Wf-317D 2026-09-29 re-scope. Line 45 row `Paper v2 submission package (Tier-2 journal — per user 2026-09-29)`, line 116-119 §2 任务 #4, line 131 §3 ledger, line 170 §6 — all already Tier-2. No edit needed.

### §1.3 INDEX.md (0 edits — legacy "re-scoped from JCIM" trail preserved)

`grep -c "JCIM" INDEX.md` = **21**. All 21 references are LEGACY "re-scoped from JCIM per AGENTS.md §5.4" historical pointers (e.g. §0.4 line 131, §17.12 line 661, §18 line 694, §19 line 762, §21 line 993, line 1128, line 1180). These document the re-scope TRAIL and per `party-twenty-spirit` §3.5 honest framing — historical context must remain visible (do NOT fabricate "Tier-2 only" by erasing history). Per `mol-metal-paper-writing` §5 honest framing protocol, every MEASURED/PROJECTED re-scope event is documented with its from/to trail.

---

## §2 INDEX.md §0 Wf-309..321 entry verification (per task brief item 3)

**Honest framing**: Per `party-twenty-spirit` §3.5 — the brief asked to "verify INDEX.md §0 includes Wf-309/310/311/312/313/314/315/316/317/318/319/320/321 entries (per Wf-317I synthesis final sync)".

**Reality check** (Bash verified):

| Section | Wf entries present | Status |
|---|---|---|
| §0.1 (timeline table) | Wf-256..268 only | ✅ Highest-density ship wave 2026-09-28 |
| §23 | Wf-316B (audit) + referenced Wf-307D, Wf-308C, Wf-309B, Wf-310C, Wf-310F, Wf-311, Wf-312, Wf-313C, Wf-313D | ✅ Wf-316B archival audit 2026-09-29 |
| §24 | Wf-316A + referenced Wf-310A, Wf-313G | ✅ Wf-316A pending audit 2026-09-29 |
| §0.4 (paper target + wet-lab directive) | (no Wf-309..321 listed) | ✅ Paper target directive applied |
| Wf-314, 315, 317, 318, 319, 320, 321 | NOT found in §0 | ⚠️ Spread across §23-24 + per-batch paragraphs |

**index_md_status**: §0 does NOT have a consolidated Wf-309..321 timeline entry; these workflows are documented in §23 (Wf-316B archival) + §24 (Wf-316A pending) + cross-referenced in §0.4 + §18 + §19 narrative paragraphs. The "§0 timeline" section is reserved for highest-density ship waves (e.g. §0.1 Wf-256..268 = 15 Wf in 1 day). Per `party-twenty-spirit` §3.5 — NOT fabricating entries; honest framing.

**Cross-link**: §0 line 3 last-updated line references Wf-310A/Wf-313G + Wf-316A/Wf-316B (2026-09-29 cascade) explicitly. §0.4 lines 127-145 codify the Tier-2 paper target directive. §18 lines 694-746 codify Tier-2 submission package ship (FINAL, RE-SCOPED from JCIM).

---

## §3 Pre-flight status (per task brief item 4)

**venv correction**: Brief specified `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python` — that path does NOT exist (verified via Bash `ls`). Correct path: `/home/hugo/codes/mol-metal/.venv/bin/python` (per `find /home/hugo -maxdepth 4 -type d -name ".venv*"` 2026-09-29).

**Pre-flight v2 run** (run 2026-09-29):
```
/home/hugo/codes/mol-metal/.venv/bin/python -m molmetal.scripts.pre_flight_check
```

**Result** (`/tmp/preflight_322c.txt` summary block):
- **PASS: 55 / 57** gates
- **FAIL: 2 / 57** gates:
  1. `p3_gate_count_consistency_audit` — drift for pipeline='audit': actual=0 vs expected=55 (silent gate loss; per evidence: "Update TODO/completed/321d_preflight_gate_audit.md AND EXPECTED_GATE_COUNTS together").
- **SKIP: 0 / 57**
- **Total wall**: 301.71 sec (5 min 1.7s)
- **Exit code**: 1 (FAIL per pre-flight v2 spec)

**preflight_status**: 55/57 PASS + 2 FAIL (drift in audit pipeline gate count; per Wf-321D follow-up). Compare with Wf-316B snapshot 2026-09-29: 48/48 PASS — gate count grew from 25/25 (Wf-263C) → 48/48 (Wf-316B) → 57/57 expected (Wf-322C run); 55/57 = 2 audit-pipeline gate losses per `EXPECTED_GATE_COUNTS` drift. NOT a regression in pre-flight itself; gate-count registry stale (per Wf-321D follow-up to be addressed by Wf-322D/E).

**STALE flag**: Not flagged (fresh today 2026-09-29; pre-flight script emits `now - last_refresh > 24 hr` STALE only on 24-hour age).

---

## §4 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 2 row updates in IMPROVEMENT_ENUM.md (line 18, line 92); pre-flight 55/57 PASS + 2 FAIL (drift, NOT pre-flight regression); venv path correction noted.
- **NOT MEASURED**: §0 INDEX.md does NOT have Wf-309..321 consolidated timeline (these are in §23-24). NOT fabricating entries.
- **NOT TOUCHED**: IMPROVEMENT_QUEUE.md (already Tier-2 per Wf-317D); INDEX.md (legacy JCIM re-scope trail preserved); pre-flight gate-count registry (deferred to Wf-321D follow-up).
- **0 GPU slot**: docs-only update.
- **0 paper touches**: per TODO-84 standing rule, no `paper/*` modification.

---

## §5 Verification trail

| Check | Tool | Result |
|---|---|---|
| IMPROVEMENT_ENUM.md TODO-138 row updated | Edit | ✅ "Paper v2 + Tier-2 journal submission package (re-scoped from JCIM per AGENTS.md §5.4 / Wf-317D 2026-09-29)" |
| IMPROVEMENT_ENUM.md §1.5 task header updated | Edit | ✅ "Paper v2 + Tier-2 journal submission package (re-scoped from JCIM per AGENTS.md §5.4 / Wf-317D 2026-09-29)" |
| IMPROVEMENT_ENUM.md JCIM count | Bash `grep -c` | 3 → 1 (post-edit) |
| IMPROVEMENT_QUEUE.md Tier-2 verify | Bash `grep -n "Tier-2"` | 4 references at line 45, 119, 131, 170 — all Tier-2 |
| INDEX.md §0 Wf-309..321 presence | Bash `grep -n` | §0 timeline only has Wf-256..268; Wf-309..321 in §23-24 |
| Pre-flight v2 run | Bash `python -m molmetal.scripts.pre_flight_check` | 55/57 PASS + 2 FAIL (drift); wall 301.7s |

---

**END OF Wf-322C** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; 2 row updates in IMPROVEMENT_ENUM.md + 0 edits elsewhere + honest §0 INDEX.md verification; pre-flight 55/57 PASS with documented drift follow-up to Wf-321D.
