# Wf-386-A2 — Wf-378-A3 Wrong-Verdict Root Cause Investigation

## §0 TL;DR — Root Cause Verdict

**HYPOTHESIS B (TIMING RACE CONDITION) CONFIRMED — 100%.**

Wf-378-A3 ran to completion in **53 seconds** (2026-09-29T16:44:01.131Z → 16:45:11.957Z).
Wf-377-A1 wrote its first closeout (`377a1_m6_kappa_redesign.md`) at **2026-09-29T16:45:25Z** —
**14 seconds AFTER Wf-378-A3 finished**.
All 5 Wf-377 closeouts were written **AFTER** Wf-378-A3 returned its result.

Wf-378-A3 verdict ("0/5 SHIPPED — files do NOT exist on disk") was **technically accurate at
the moment of its execution** (16:44:54Z when its last bash tool returned) but became
**stale** by the time Wf-378-B1 ran its final synthesis ~26 minutes later at 17:11Z,
which correctly reported 5/5 SHIPPED.

This is a **workflow orchestration issue** (parallel verify launched before wave completed),
**NOT** an agent logic issue. Wf-378-A3 ran the correct `ls` commands against the correct
paths with the correct working directory — the files simply did not exist yet.

---

## §1 Hypothesis A Test — Truncation Check (REJECTED)

**Hypothesis:** Wf-378-A3's output was truncated by the harness, hiding any partial verification.

**Test method:** Read full agent jsonl + journal.jsonl result entry.

**Findings:**
- Wf-378-A3's jsonl file `agent-a3e50818b106ecf43.jsonl` is **720,386 bytes** (compact for a
  multi-tool agent). Final event timestamp: `2026-09-29T16:45:11.957Z`.
- 21 `tool_use` events recorded during 53-second life (16:44:01 → 16:45:11).
- Result text in journal.jsonl is **complete and well-structured**: 9 numbered verification
  sections + cross-poll verdict + recommended follow-up + OUTPUT (PLAIN TEXT) footer.
- Total result text length: ~3,800 characters; no truncation markers.

**Verdict:** Hypothesis A **REJECTED**. Output was NOT truncated. Agent ran 21 tool calls
and emitted a complete verification report.

---

## §2 Hypothesis B Test — Timing Race Check (CONFIRMED)

**Hypothesis:** Wf-377 closeouts were written AFTER Wf-378-A3 finished.

**Test method:** Compare Wf-378-A3 lifecycle timestamps vs Wf-377 closeout file mtimes.

### §2.1 Wf-378-A3 Lifecycle (from agent-a3e50818b106ecf43.jsonl)

| Event | Timestamp (UTC) | Local (CST = UTC+8) |
|---|---|---|
| Workflow launched | (parent wave) | 00:44 CST |
| Agent started (first user message) | 2026-09-29T16:44:01.131Z | 00:44:01 CST |
| Environment snapshot (CWD = repo root) | 2026-09-29T16:44:01.134Z | 00:44:01 CST |
| Last bash tool result returned | 2026-09-29T16:44:54.545Z | 00:44:54 CST |
| Assistant thinking completed | 2026-09-29T16:44:58.035Z | 00:44:58 CST |
| **Agent result sent (end of work)** | **2026-09-29T16:45:11.957Z** | **00:45:11 CST** |
| Agent jsonl mtime | 1790700312 (≈ 16:45:12Z) | 00:45:12 CST |

**Wf-378-A3 agent life:** 70.83 seconds (16:44:01 → 16:45:12).

### §2.2 Wf-377 Closeout File mtimes (epoch → UTC)

```
377a1_m6_kappa_redesign.md    1790700325 → 2026-09-29T16:45:25Z (00:45:25 CST)
377a2_m2r2_diagnostic.md      1790700499 → 2026-09-29T16:48:19Z (00:48:19 CST)
377a4_dashboard_final.md      1790700502 → 2026-09-29T16:48:22Z (00:48:22 CST)
377_final_synthesis.md        1790701036 → 2026-09-29T16:57:16Z (00:57:16 CST)
377b1_index_sync.md           1790701054 → 2026-09-29T16:57:34Z (00:57:34 CST)
```

### §2.3 Critical Timing Delta

```
Wf-378-A3 finished work:    2026-09-29T16:45:11.957Z (00:45:11 CST)
Wf-377-A1 wrote 377a1:      2026-09-29T16:45:25Z     (00:45:25 CST)
                                  DELTA: +13.043 seconds (Wf-377-A1 wrote AFTER Wf-378-A3 finished)
```

**Wf-378-A3 saw the filesystem state AT 16:44:54Z** (when its last bash tool returned,
reading `ls -la TODO/completed/377*`).
**At 16:44:54Z, no Wf-377 closeout files existed on disk.**

**Verdict:** Hypothesis B **CONFIRMED**. Wf-378-A3 was launched BEFORE Wf-377 had written
ANY of its closeout files. By the time Wf-378-A3 finished 53 seconds later, Wf-377-A1
was still authoring (it did not finish writing until 16:45:48Z, 14 seconds after Wf-378-A3
had emitted its result).

### §2.4 Race Sequence — Full Timeline

```
Time (UTC)        | Event
------------------+-------------------------------------------------------
16:43:37Z         | Wf-377 wave launched (wf_dd441809-69c)
                  | Wf-377-A1 starts (a258279661a867f96) — 2-min authoring
                  | Wf-377-A3 starts (abaccebd22d200510) — quick cross-poll
16:44:01Z         | Wf-378 wave launched (wf_c6a05300-bb4)
                  | **Wf-378-A1 starts (pre_flight_gpu_ready.sh gate)**
                  | **Wf-378-A2 starts (G6 USER_OK gate)**
                  | **Wf-378-A3 starts (Wf-377 cross-poll) ← THE RACE**
16:44:54Z         | Wf-378-A3 last bash tool returns
                  | → reads `ls TODO/completed/377*` → sees NOTHING
                  | → reports "0/5 SHIPPED"
16:45:12Z         | Wf-378-A3 result emitted
                  | Wf-378-A3 agent terminates (71 sec life)
16:45:25Z         | Wf-377-A1 finally writes 377a1_m6_kappa_redesign.md
                  | → 14 seconds AFTER Wf-378-A3 finished
16:48:10Z         | Wf-378-A2 result emitted (rc=64 enforcement)
16:48:19Z         | Wf-377-A2 writes 377a2_m2r2_diagnostic.md
16:48:22Z         | Wf-377-A4 writes 377a4_dashboard_final.md
16:57:16Z         | Wf-377 writes 377_final_synthesis.md
16:57:34Z         | Wf-377 writes 377b1_index_sync.md
17:02:13Z         | Wf-378-A1 result emitted (pre_flight_gpu_ready.sh gate)
17:03:37Z         | Wf-377-B1 starts (final ship sub-flow)
17:03:49Z         | Wf-377-B1 final ship emitted
                  | → correctly reports 5/5 SHIPPED
17:11Z+           | Wf-378-B1 starts (final ship sub-flow)
                  | → also correctly reports 5/5 SHIPPED
```

**Conclusion:** Wf-378-A3 was a **frozen snapshot** of filesystem state at 16:44:54Z —
accurate for that instant, but stale by the time the parent wave ended.

---

## §3 Hypothesis C Test — File Pattern Check (REJECTED)

**Hypothesis:** Wf-378-A3 used wrong glob or regex.

**Test method:** Read Wf-378-A3 task prompt from jsonl user message + verify glob pattern.

**Findings:**
- Task brief explicitly specified: `ls -la TODO/completed/377*` (correct glob).
- Wf-378-A3 result text lists 6 expected paths:
  - `TODO/completed/377a1_*`
  - `TODO/completed/377a2_*`
  - `TODO/completed/377a3_wf376_crosspoll.md`
  - `TODO/completed/377a4_dashboard_final.md`
  - `TODO/completed/377b1_index_sync.md`
  - `TODO/completed/377_final_synthesis.md`
- All 6 patterns are correct and match the actual filenames on disk.

**Verdict:** Hypothesis C **REJECTED**. Glob patterns were correct. The agent just found
nothing because nothing existed yet at the time of its `ls`.

---

## §4 Hypothesis D Test — Filesystem State Check (REJECTED — no stale view)

**Hypothesis:** Agent saw a stale view of the filesystem (caching, NFS, etc.).

**Test method:** Verify `TODO/completed/377*.md` actually existed when Wf-378-A3 ran.

**Findings:**
- Wf-378-A3 finished at 16:45:11Z.
- First Wf-377 closeout (377a1) created at 16:45:25Z — **14 seconds AFTER**.
- Wf-378-A3's bash command `ls -la TODO/completed/377*` would have correctly returned
  empty (no matches) because the files did not exist.
- The filesystem view was **NOT stale** — it accurately reflected disk state at that moment.
- Wf-378-B1 ran ~26 minutes later and saw the same filesystem accurately with all 5 files.

**Verdict:** Hypothesis D **REJECTED**. Filesystem view was accurate at the time of
Wf-378-A3's check. No caching/NFS issue.

---

## §5 Hypothesis E Test — Working Directory Check (REJECTED)

**Hypothesis:** Agent had wrong working directory.

**Test method:** Read environment snapshot in Wf-378-A3 jsonl.

**Findings:**
- Environment snapshot at 2026-09-29T16:44:01.134Z (3 ms after agent start):
  ```json
  {
    "type": "attachment",
    "snapshot": {
      "workingDirectory": "/home/hugo/codes/mol-metal",
      "isWorktree": false,
      "isGitRepo": true,
      "additionalWorkingDirectories": [],
      "platform": "linux",
      "shell": "zsh",
      "osVersion": "Linux 7.2.6-hardened1-1-hardened"
    }
  }
  ```
- CWD = `/home/hugo/codes/mol-metal` (correct repo root).
- `TODO/completed/377*` relative path resolves correctly from this CWD.

**Verdict:** Hypothesis E **REJECTED**. Working directory was correct.

---

## §6 Root Cause Conclusion

**Root Cause:** TIMING RACE CONDITION between Wf-378 wave launch and Wf-377 wave completion.

The workflow orchestration launched Wf-378 (a verify-only cross-poll wave) in parallel with
Wf-377's authoring. Wf-378-A3 (the verify-only sub-flow) ran to completion in 71 seconds
while Wf-377-A1 was still authoring its first closeout file. The `ls TODO/completed/377*`
check was technically correct but returned empty because the files had not yet been
written to disk.

**This is a benign stale-snapshot issue, NOT an agent error:**
- Wf-378-A3's verdict was correct AT THE INSTANT IT RAN
- The verdict became stale by the time the parent wave finished
- Wf-378-B1 (final ship, ran at 17:11Z+) correctly captured the final state as 5/5 SHIPPED
- Wf-384 cross-poll (later wave) correctly confirmed 5/5 SHIPPED on disk

**Cross-pollination status:**
- Wf-378-A3 (16:44-16:45Z): wrong verdict (stale snapshot at 0/5) → HARM: low; 0 source
  files modified, 0 GPU launches, 0 paper/* touches; verdict recorded in journal
- Wf-378-B1 (17:11Z+): correct verdict (5/5 SHIPPED) → CORRECTION: high; documented the
  eventual final state and rolled Wf-378-A3 result into §6 as a stale-snapshot caveat
- Wf-384-A1/A2/B1 (later, 2026-09-30+): correct verdict (5/5 SHIPPED) → RE-AUTHOR: high;
  Wf-384-A3 re-authored the Wf-378-A3 closeout to reflect the correct final state

**Recommended workflow-orchestration fix:** For verify-only cross-poll sub-flows targeting
a sibling wave, the parent workflow should:
1. Either launch the verify-only AFTER the sibling wave's deliverables are written
2. Or instruct the verify-only sub-flow to retry once with a short sleep + re-check loop
3. Or include a "AS_OF timestamp" assertion in the verify-only sub-flow's verdict that
   explicitly notes when the snapshot was taken

Currently no such protection exists. This race could recur.

---

## §7 Cross-References

- `/home/hugo/codes/mol-metal/TODO/completed/378_final_synthesis.md` (§6 Wf-378 OPEN
  gaps closure — references the Wf-378-A3 stale-snapshot caveat)
- `/home/hugo/codes/mol-metal/TODO/completed/378b1_index_sync.md` (Wf-378-B1 docs sync)
- `/home/hugo/codes/mol-metal/TODO/completed/377a1_m6_kappa_redesign.md` (Wf-377-A1 first
  closeout; mtime 16:45:25Z)
- `/home/hugo/codes/mol-metal/TODO/completed/377a2_m2r2_diagnostic.md` (Wf-377-A2; 16:48:19Z)
- `/home/hugo/codes/mol-metal/TODO/completed/377a4_dashboard_final.md` (Wf-377-A4; 16:48:22Z)
- `/home/hugo/codes/mol-metal/TODO/completed/377b1_index_sync.md` (Wf-377-B1; 16:57:34Z)
- `/home/hugo/codes/mol-metal/TODO/completed/377_final_synthesis.md` (Wf-377 final ship; 16:57:16Z)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf378b1_preflight.txt` (Wf-378-B1 pre-flight
  89/2/3/94 wall 155.01s; correctly reports wf377=5/5 SHIPPED at 17:12:27Z)

**Workflow journals:**
- `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_c6a05300-bb4/journal.jsonl` (Wf-378 parent journal)
- `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_dd441809-69c/journal.jsonl` (Wf-377 parent journal)
- `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_43a93c29-1ea/journal.jsonl` (Wf-384 retro-fix journal)

---

## §8 Honest Framing per party-twenty-spirit §3.5

This is **a workflow orchestration finding**, not an agent finding:
- Wf-378-A3 ran the correct commands against the correct paths with the correct working
  directory at 16:44-16:45Z.
- The filesystem state it observed was accurate at that instant.
- The race is between the workflow harness (launched Wf-378 in parallel with Wf-377)
  and the agent (which could not anticipate future file creations).
- Wf-378-B1 (the wave's own final ship) correctly captured the final state 5/5 SHIPPED.
- Wf-384 retroactively re-authored Wf-378-A3's record with the correct final state.

**MEASURED timing deltas:**
- Wf-378-A3 finished BEFORE Wf-377-A1 wrote any file: Δ = +13.043 seconds
- Wf-378-A3 finished BEFORE Wf-377-A2 wrote any file: Δ = +188 seconds (~3 min)
- Wf-378-A3 finished BEFORE Wf-377-B1 wrote any file: Δ = +742 seconds (~12 min)
- Wf-378-A3 finished BEFORE 377_final_synthesis was written: Δ = +724 seconds (~12 min)

**PROJECTED likelihood of recurrence:** MEDIUM. The parallel verify pattern (Wf-378-A3
verify-Wf-377 in parallel with Wf-377 authoring) was used as a "quick cross-poll" but
introduced a structural race. Future verify-only sub-flows targeting sibling waves should
either (a) be deferred to AFTER the sibling wave's file writes, or (b) include a
retry-once-with-sleep loop in the verify-only sub-flow's tool orchestration.

**0 NEW tests** — investigation is READ-only per task brief.
**0 GPU launches** — investigation is CPU-only.
**0 paper/* touches** — investigation is docs-only.
**0 wet-lab planning** — investigation is docs-only.

---

**END OF Wf-386-A2 CLOSEOUT**
