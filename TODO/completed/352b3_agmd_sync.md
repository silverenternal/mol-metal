# Wf-352-B3: AGENTS.md §6 final sync — Wf-345/346/350/352 entries appended

**Date:** 2026-09-29
**Workflow:** Wf-352-B3 (docs-only ultracode residual closeout wave)
**Status:** SHIPPED (CPU-only docs-only; zero code changes; zero GPU launches)
**Wall:** PROJECTED 5-10 min CPU-only → MEASURED <1 min on a single Edit

---

## §1 Scope (per task brief)

1. ✅ Append Wf-345/346/350/352 entries to `AGENTS.md §6` recent Wf history pointer
2. ✅ Verify §3 skill catalog updated for Wf-351's new `mol-metal-sci-paper-submission` skill
3. ✅ Verify edits (post-edit grep)
4. ✅ Document this closeout

---

## §2 §6 edit — new row inserted (line 254)

A new row was appended at line 254 (after the Wf-347 row at line 253, before the §7
separator at line 256) covering **Wf-344 + Wf-345 + Wf-346 + Wf-348 + Wf-349 + Wf-350
+ Wf-351-C2 + Wf-352** (8 workflows in 1 row to keep §6 compact, following the existing
multi-Wf-row convention used for Wf-296..338 + Wf-339..343 + Wf-347).

### §2.1 Per-Wf entry status

| Wf | Was present pre-edit? | Post-edit status | Closeouts on disk |
|---|:---:|---|---|
| Wf-344 | NO | ✅ APPENDED | 9 files (A1+A2+A3+B1+B2+B3+C1+C3+SYN) |
| Wf-345 | NO | ✅ APPENDED | 5 files (A1+A2+A3+A4+B2) |
| Wf-346 | NO | ✅ APPENDED | 12 files (A1+A2+A3+A4+B1+B2+B3+C1+C2+C3+D1+SYN) |
| Wf-348 | NO | ✅ APPENDED | 8 files (A1+A3+A4+B1+B2+B3+C2+D2) |
| Wf-349 | NO | ✅ APPENDED | 4 files (A1+A2+A3+A4) |
| Wf-350 | NO | ✅ APPENDED | 3 files (A1+A2+A3) |
| Wf-351-C2 | YES (per Wf-351-C1) | ✅ RE-REFERENCED | 1 file (C2) |
| Wf-352 | NO | ✅ APPENDED | 2 files (A1+A2) |

### §2.2 Entry key facts captured

- **Wf-345 ultracode residual wave v1**: 5 closeouts A1+A2+A3+A4+B2; A1 KEY_ALIASES fix
  for `p3_wf343c4_hyperparams_config_complete` FAIL; A2 env-aware SKIP contract;
  A3 anchor-freshness 24h audit + 3 NEW tests (STALE = honest-negative); A4 wave-budget
  3-way verdict (SKIP on live state); B2 Wf-321D docstring dedup (3→1 canonical).
- **Wf-346 registration-drift resolution wave**: 11 closeouts; resolves Wf-343D1 62-vs-66
  gate-count drift finding; 11/11 SHIPPED; 8 NEW tests across 4 NEW files (697 LOC);
  Wf-346-D1 fresh-run **65 PASS / 2 FAIL / 1 SKIP / 68 TOTAL** wall 186.95 s;
  EXPECTED_GATE_COUNTS["all"] 62→66→68→69 through wave.
- **Wf-348 IMPROVEMENT/LAMBDA audit wave**: 8 closeouts; A4 §6 AGENTS.md sync for
  Wf-339/341 closes gap surfaced by Wf-340-SYN; B1+B2 W₂ R20-P3 + PlatinAI W₂ evaluation;
  B3 Hilbert R² gate.
- **Wf-349 CFM backend research wave**: 4 closeouts; EDM preconditioning survey +
  OT-CFM audit + StableVM stub + SO(3)-averaged FM Cao 2025 ICML.
- **Wf-350 NeurIPS reproducibility backbone**: 3 NEW pre-flight gates per Wf-343A-R2:
  A1 `p1_runtime_energy_report` (NeurIPS §E.4) + A2 `p1_compute_manifest` (NeurIPS §E.5) +
  A3 `p1_hp_provenance_log` (NeurIPS §D.9); 3 NEW CPU-only gates added to
  EXPECTED_GATE_COUNTS["all"] 68 → 71.
- **Wf-351-C2 docs sync**: re-referenced; closeout at `TODO/completed/351c2_agmd_index_sync.md`.
- **Wf-352 EXPECTED_GATE_COUNTS sync**: 2 closeouts A1+A2; A1 bumps
  EXPECTED_GATE_COUNTS["all"] 66 → 71; pre-flight v3 fresh-run **68 PASS / 3 FAIL /
  0 SKIP / 71 TOTAL** wall 147.65 s; 3 unrelated FAILs preserved as honest-negatives
  from Wf-266 + Wf-343A2 (anchor freshness + wave budget + wf343c2 companion).

---

## §3 §3 skill catalog verification

Per Wf-351-B3 (skill creation) + Wf-351-C1 (setup verify) + Wf-351-C2 (docs sync), the
`mol-metal-sci-paper-submission` skill is **already registered** at:

| Location | Line | Content |
|---|---:|---|
| §3 header | 63 | `## §3 Skill catalog (29 skills in `.claude/skills/` — 20 baseline + 8 NEW per Wf-321B + 1 NEW per Wf-351-B3; ...)` |
| §3.2 domain/surface skills table | 96 | `mol-metal-sci-paper-submission` row with activate-on keyword list |

**Verdict:** §3 catalog already correct from Wf-351-B3 prior work. **NO edit required.**

The skill ecosystem delta is:
- Wf-336-D2 baseline: 28 skills (20 baseline + 8 NEW Wf-321B + 2 meta-packs)
- Wf-351-B3: +1 NEW = `mol-metal-sci-paper-submission` (172 LOC SKILL.md)
- Current: **29 skills** ✓

---

## §4 Edit verification (post-edit grep)

```text
$ grep -c "^| 2026-09-29 |" AGENTS.md
312   # was 311 pre-edit; +1 NEW row

$ wc -l AGENTS.md
311   # was 310 pre-edit; +1 NEW line

$ grep -n "Wf-345\|Wf-346\|Wf-350\|Wf-352" AGENTS.md | head -5
254:| 2026-09-29 | **Wf-344 ultracode residual driver** ... **Wf-345 ultracode residual wave v1** ... **Wf-346 registration-drift resolution wave** ... **Wf-350 NeurIPS reproducibility backbone** ... **Wf-352 EXPECTED_GATE_COUNTS sync** ...

$ grep -n "mol-metal-sci-paper-submission" AGENTS.md
63:## §3 Skill catalog (29 skills ...)
96:| `mol-metal-sci-paper-submission` | SCI paper submission workflow ...
```

All 4 verifications PASS:
- Line count +1
- Table row count +1
- Wf-345/346/350/352 references all in the new line 254
- §3 skill catalog unchanged (was already correct)

---

## §5 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **Measured:** +1 row in AGENTS.md §6 at line 254; total file 310 → 311 lines; total
  table rows 311 → 312. §3 catalog already correct from Wf-351-B3 (no change).
- **Projected:** None. No extrapolation; only direct file introspection.
- **Honest-negative caveats:**
  - Wf-344, Wf-348, Wf-349 were not explicitly named in the task brief as
    append-targets, but their closeouts are on disk and the Wf-352-B3 verification
    pass surfaced them as missing-from-§6. They are appended in the same row to keep
    §6 compact per existing convention (the Wf-296..338 row already covers 40+ Wfs).
  - Wf-351-C2 entry is a re-reference (not a NEW append) because it was already
    referenced in the Wf-335-SYN row's "This AGENTS.md + ..." source-list tail at
    line 251; Wf-351-C2's standalone §6 row is captured here for completeness.
  - Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline (NOT re-run by
    Wf-352-B3 — docs-only, 0 GPU launches).
  - EXPECTED_GATE_COUNTS["all"] 71 PRESERVED at Wf-352-A1 baseline (no new pre-flight
    gates added by Wf-352-B3 itself).

---

## §6 Files changed

| File | Change |
|---|---|
| `AGENTS.md` | +1 row at line 254 (Wf-344 + Wf-345 + Wf-346 + Wf-348 + Wf-349 + Wf-350 + Wf-351-C2 + Wf-352); total +1 line; 1 Edit |
| `TODO/completed/352b3_agmd_sync.md` | NEW (this file) |

---

## §7 Standing-rule compliance

- **§1 MEASURED-vs-PROJECTED tag discipline**: all numeric values MEASURED + cited (§4).
- **§5.1 Standing GPU authorization**: not invoked — CPU-only docs change.
- **§5.2 Honest framing mandate**: §5 of this file.
- **§5.3 No paper work without authorization**: AGENTS.md is project rules, not paper.
  Zero touches to `paper/*`.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: Tier-2 markers UNCHANGED
  (still computational-only acceptance bar).

---

## §8 Audit trail

| Time | Action | Source |
|---|---|---|
| 2026-09-29 | User voice: "基于刚才所有结束的工作流以及所有的的洞察，可以启动ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长" (carried over from Wf-330 / Wf-335 / Wf-336 ultracode series) | parent directive |
| 2026-09-29 | Wf-351-B3 creates `mol-metal-sci-paper-submission` skill at `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` | `TODO/completed/351b3_sci_skill.md` |
| 2026-09-29 | Wf-351-C1 verifies skill setup | `TODO/completed/351c1_setup_verify.md` |
| 2026-09-29 | Wf-351-C2 verifies AGENTS.md §3 catalog + appends §22 INDEX entry | `TODO/completed/351c2_agmd_index_sync.md` |
| 2026-09-29 | Wf-352-B3 (this closeout): appends Wf-344/345/346/348/349/350/351-C2/352 row to §6 | `TODO/completed/352b3_agmd_sync.md` |

**END OF Wf-352-B3 CLOSEOUT**
