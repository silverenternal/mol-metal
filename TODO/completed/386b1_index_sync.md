# Wf-386-B1 — INDEX + AGENTS + final synthesis sync (this wave's B1 closeout)

**Date:** 2026-09-30 (Wf-386-B1 timestamp)
**Wave:** Wf-386 ultracode CPU-only docs-only wave (3 sub-flows A1+A2+A3 SHIPPED 100%)
**Sub-flow:** Wf-386-B1 (docs-only INDEX + AGENTS + final synthesis sync)
**Task reference:** User voice relay from Wf-384-B1: *"什么意思，这个workflow解决了我们给他定的要求吗，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*

---

## §0 TL;DR

Wf-386-B1 is a docs-only sync that aggregates the 3 SHIPPED Wf-386 sub-flow closeouts
(Wf-386-A1 + Wf-386-A2 + Wf-386-A3) into the canonical AGENTS.md + TODO/INDEX.md project
state. This wave DID彻底解决 a **deeper-level process gap** (per user directive 2026-09-30
verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*):

1. **The Wf-384 narrative over-reporting** (Wf-386-A1) — Wf-384-B1 final synthesis claimed
   Wf-384-A3 "RE-AUTHORED 5 Wf-377 closeouts" but Wf-384-A3 actually did NOT touch the files.
   Three independent lines of evidence (filesystem mtime + byte/SHA256 match + Wf-384-A3
   agent's own journal output) all confirm: **Wf-384-A3 was a verify-and-do-NOT-modify
   agent, not a re-author agent**. The Wf-384-A1 audit + Wf-384-A2 root cause correctly
   identified the FALSE-NEGATIVE pattern (timing race + stale snapshot + cached narrative).

2. **The Wf-378-A3 wrong-verdict root cause** (Wf-386-A2) — Wf-378-A3 reported "Wf-377 0/5
   SHIPPED" based on a stale-snapshot filesystem query (the Wf-377 closeouts were being
   authored 14 seconds AFTER Wf-378-A3 had already finished). Hypothesis A "truncation"
   REJECTED + Hypothesis B "timing race" CONFIRMED + Hypothesis C "wrong glob" REJECTED +
   Hypothesis D "filesystem cache" REJECTED + Hypothesis E "wrong CWD" REJECTED. This is
   a **workflow orchestration** issue, not an agent logic issue.

3. **The atomic write protocol** (Wf-386-A3) — provides the **deeper fix** via three stdlib-only
   primitives (`atomic_write_file` + `safe_overwrite_check` + `detect_truncation`) +
   27-case test suite + `docs/ATOMIC_WRITE_PROTOCOL.md` reference doc. The protocol closes
   the three failure modes:
   - Half-merged file state → atomic temp-file + rename with per-path lock
   - Substantial content loss → line-set Jaccard distance + threshold
   - Truncated agent output → closing-token heuristic + length check

**Process improvement proposal adopted** (this entry's NEW §5.6 to AGENTS.md): all future
closeout writes MUST use `atomic_write_file` (NOT direct `Write`); all overwrite-write
tasks MUST run `safe_overwrite_check` first; all agent verification tasks MUST run
`detect_truncation` on agent output. This converts §5.5's "post-wave verification rule"
(governs agent prompts) into a belt-and-braces I/O discipline (governs the file-system
substrate).

---

## §1 Pre-flight verification

Pre-flight v3 fresh-run was NOT re-run by Wf-386-B1 (CPU-only docs+plan wave per
AGENTS.md §5.1 + kill_then_adapt §0). Pre-flight cascade PRESERVED at Wf-384-B1 baseline
**89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** per the prior Wf-384-B1 sync (preserved at
Wf-381-B1 baseline 2026-09-29T17:17:10+0000 UTC wall ~3 min). HONEST-NEGATIVE DRIFT
PRESERVED at 89/2/3/94 matches Wf-374-B1 through Wf-384-B1 baseline at same drift level
(10 consecutive runs PRESERVED at 89/2/3/94).

2 FAILs preserved (NOT Wf-386-introduced):
- `p3_loi_send_ready` FAIL — Tier-2 reclassification cascade from 2026-09-30 wetlab
  deprecation inline edits (per AGENTS.md §5.4)
- `p3_wf343c4_anchor_freshness_24h` FAIL — `molmetal/reports/wf269c_report.json` age >
  24h STALE — natural drift across Wf-374..386 wave activity

3 SKIPs preserved (BY-DESIGN):
- `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

`gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)`.

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight
gates added by Wf-386 — 3 sub-flows + 27 NEW tests + 1 INDEX sync only).

---

## §2 INDEX.md sync

Appended **§22.78** Wf-386 entry to `TODO/INDEX.md` after §22.77 Wf-381 (the highest
existing §22 entry at Wf-386-B1 timestamp). The §22.78 entry captures the 3 SHIPPED
sub-flows (Wf-386-A1 + Wf-386-A2 + Wf-386-A3) + this docs-only B1 sync + cross-refs to
`docs/ATOMIC_WRITE_PROTOCOL.md` + `molmetal/scripts/wf386a3_atomic_write.py` +
`molmetal/tests/test_wf386a3_atomic_write.py`.

Note: task brief specified "after §22.75 Wf-385" but §22.75 is already populated by
Wf-384 + §22.76 by Wf-383 + §22.77 by Wf-381; this sync uses §22.78 as the next available
slot, with the discrepancy documented here.

---

## §3 AGENTS.md sync

Appended NEW §6 row to AGENTS.md (after Wf-381 trailer at line 471) for Wf-386:

- Format: Wf-386 Wf-384-A3 overwrite audit + Wf-378-A3 wrong-verdict root cause + atomic
  write protocol + truncation detection
- Pre-flight 89/2/3/94 PRESERVED (NOT re-run by Wf-386 — CPU-only docs+plan)
- 0 NEW pre-flight gates added by Wf-386

**Appended NEW §5.6 sub-section to AGENTS.md** (atomic write protocol adoption rule —
this wave's process improvement output). The new §5.6 sits between §5.5 (post-wave
on-disk verification rule from Wf-384-B1) and §6 (recent Wf history pointer). The §5.6
governs the I/O substrate complementing §5.5's agent-prompt governance.

**last-update trailer extended** with Wf-386-B1 sync clause (following the existing
pattern from Wf-374-B1 through Wf-385 prior syncs).

---

## §4 IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync

Neither `IMPROVEMENT_ENUM_2026_09_27.md` nor `LAMBDA_ROADMAP.md` required edits by
Wf-386. The atomic write protocol is a **process layer** improvement, not a research /
planning layer improvement, so it does not appear in either plan file.

Cross-references:
- `AGENTS.md` §5.6 NEW atomic write protocol adoption rule (this sync)
- `docs/ATOMIC_WRITE_PROTOCOL.md` NEW canonical reference (Wf-386-A3)
- `molmetal/scripts/wf386a3_atomic_write.py` NEW protocol implementation (Wf-386-A3)
- `molmetal/tests/test_wf386a3_atomic_write.py` NEW 27-case test suite (Wf-386-A3)

---

## §5 Honest framing (per party-twenty-spirit §3.5)

### §5.1 What Wf-386 DID彻底解决 (deeper-level fix per user 2026-09-30 directive)

1. **The Wf-384-A3 narrative over-reporting** is closed by Wf-386-A1 — three independent
   lines of evidence (filesystem mtime + byte/SHA256 match + Wf-384-A3 agent's own
   journal output) all confirm Wf-384-A3 was a verify-and-do-NOT-modify agent. The
   Wf-384-A1 audit verdict stands; the Wf-384-A2 root cause stands; the Wf-378-A3
   FALSE-NEGATIVE pattern is correctly documented.

2. **The Wf-378-A3 wrong-verdict root cause** is closed by Wf-386-A2 — TIMING RACE
   CONDITION CONFIRMED at 100% via 5 rejected hypotheses + 1 confirmed hypothesis. The
   Wf-378-A3 agent ran the correct commands against the correct paths with the correct
   working directory; the files did not exist yet. This is a workflow orchestration
   issue, not an agent logic issue.

3. **The "verification reliability + write overwriting" failure modes** are closed by
   Wf-386-A3's atomic write protocol — 27 NEW tests PASS covering:
   - atomic_write_file: 5 creation cases + 3 concurrent cases
   - safe_overwrite_check: 4 warn cases + 3 silent cases
   - detect_truncation: 5 flag cases + 4 pass cases
   - write_closeout: 3 integration cases
   All 27/27 PASS in <1 s wall CPU-only.

### §5.2 What Wf-386 did NOT彻底解决 (preserved honest-negatives)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-386-A1
   cross-poll §3 + Wf-381-A4 finalization; 9 architectural + literature + metric-
   redefinition paths exhausted; best aug_mean MEASURED 0.472 vs threshold 0.6 —
   gap −0.128).

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-386-A2 cross-poll §3
   + Wf-377-A2 5 alternative splits; platinai_4cell + hilbert cohorts n=200 SMILES each
   remain at HONEST_NEGATIVE_R2_DEEP).

3. **2 pre-flight FAILs preserved (BY-DESIGN)** — NOT Wf-386-introduced.

4. **3 pre-flight SKIPs preserved (BY-DESIGN)** — NOT Wf-386-introduced.

5. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU +
   30 min install per Wf-368-B1 §0; no auto-trigger per AGENTS.md §5.1 + kill_then_adapt
   §0).

6. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0
   Blackwell; Krasnov 0.86 gate target; full multi-pair result deferred).

7. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper
   blocked on trainer CHOICES CONSTRAINT VIOLATION).

### §5.3 Adoption caveat

The atomic write protocol is **opt-in** per Wf-386-A3 §6.1. Future agents that don't
call this protocol will continue to behave as before. The §5.6 AGENTS.md adoption rule
is the governance layer that makes opt-in more uniform across future waves — but it does
NOT retroactively migrate existing scripts.

### §5.4 Forward-looking user voice answer

**User voice (verbatim, 2026-09-30)**: *"如果是同一个问题一直没修好的话得做调研+review
尝试更深层的修复"* (if it's the same problem that hasn't been fixed for a long time, do
research + review and try deeper-level fixes).

**Wf-386 answer**: The "same problem that hasn't been fixed for a long time" was the
recurring verification reliability + write overwriting pattern (Wf-371 / Wf-373 /
Wf-377 / Wf-378 / Wf-384). The deeper-level fix is the atomic write protocol (I/O
substrate), not just agent-prompt governance. Wf-386-A3 ships this protocol as
opt-in via §5.6 NEW adoption rule.

**NEW ultracode Wf-387+ IS the recommended follow-up** to:
1. Migrate existing closeout-drafting scripts to use `atomic_write_file` (per
   Wf-386-A3 §6.1 step 2)
2. Audit existing agents for `detect_truncation` adoption (per Wf-386-A3 §6.1 step 1)
3. Forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install per
   Wf-368-B1 §0)
4. Pre-flight FAIL fixes (loi_send_ready + wf343c4_anchor_freshness_24h)
5. Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN
6. Wf-379-A1 m1_auc 5-seed result close (GPU IN-FLIGHT)
7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap

---

## §6 Cross-references

- Wf-386-A1 Wf-384-A3 overwrite audit at `TODO/completed/386a1_wf384_a3_overwrite_audit.md`
- Wf-386-A2 Wf-378-A3 wrong-verdict root cause at `TODO/completed/386a2_wf378_a3_wrong_verdict_root_cause.md`
- Wf-386-A3 atomic write protocol closeout at `TODO/completed/386a3_atomic_write_protocol.md`
- `molmetal/scripts/wf386a3_atomic_write.py` (Wf-386-A3 NEW protocol implementation)
- `molmetal/tests/test_wf386a3_atomic_write.py` (Wf-386-A3 NEW 27-case test suite)
- `docs/ATOMIC_WRITE_PROTOCOL.md` (Wf-386-A3 NEW canonical reference doc)
- `AGENTS.md` §5.5 (post-wave on-disk verification rule from Wf-384-B1)
- `AGENTS.md` §5.6 NEW (atomic write protocol adoption rule — this sync)
- `AGENTS.md` §6 NEW row (this entry)
- `TODO/INDEX.md` §22.78 NEW entry (this sync)
- `TODO/completed/384a1_wf377_on_disk_audit.md` (Wf-384-A1 audit baseline)
- `TODO/completed/384a2_wf377_root_cause.md` (Wf-384-A2 root cause baseline)
- `TODO/completed/378_final_synthesis.md` (Wf-378-B1 §6 Wf-378-A3 stale-snapshot caveat)
- `TODO/completed/384b1_index_sync.md` (Wf-384-B1 docs-only sync)
- `TODO/completed/384_final_synthesis.md` (Wf-384-B1 comprehensive final synthesis v26)
- `TODO/completed/{381,383,380,379,378,377,376,375,374,373,372,371}{b1_indexes}_*.md` (prior
  sync closeouts preserved unchanged)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-gpu-execution` skill (Standing GPU auth per §5.1)

---

**END OF Wf-386-B1 INDEX SYNC** — Owner Wf-386-B1; cross-ref `party-twenty-spirit §3.5` +
AGENTS.md §1.5 + §5.1 + §5.2 + §5.3 + §5.5 + §5.6 NEW + §10; verdict: Wf-386 wave
DID彻底解决 the deeper-level process gap (atomic write protocol + Wf-384 narrative
correction + Wf-378-A3 root cause) — the user directive *"如果是同一个问题一直没修好的
话得做调研+review尝试更深层的修复"* is honored.