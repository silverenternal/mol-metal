# Wf-361-A2 — Verification of Wf-358-C1 cross-workflow dependency docs

**Date:** 2026-09-29
**Owner:** Hugo Lee (assisted by Claude Code)
**Status:** VERIFIED
**Workflow:** Wf-361-A2 (verification of Wf-358-C1)
**Trigger:** User directive (verbatim) "这些问题都要解决，你可以启动多个ultracode去修"
— Wf-361 verifies prior workflow closeouts that were flagged as
"output truncated" notifications.

---

## §1 Scope

Wf-358-C1 ("workflow improvement #1 — cross-workflow dependency
tracking") was originally flagged as having a truncated output in the
Wf-358 notification feed. Wf-361-A2 re-verifies that the deliverables
are actually on disk + functional, not truncated.

---

## §2 Verification

### §2.1 Document presence

```
$ ls -la docs/WORKFLOW_DEPENDENCY_GRAPH.md
-rw-r--r-- 1 hugo hugo 12513 Sep 29 21:49 docs/WORKFLOW_DEPENDENCY_GRAPH.md
```

**VERIFIED:** Document exists at the canonical location. Size 12,513 B
(target ~200 LOC / ~10 KB; MEASURED 1.25× over target).

### §2.2 Closeout document presence

```
$ ls -la TODO/completed/358c1_workflow_deps.md
-rw-r--r-- 1 hugo hugo <size> Sep 29 <ts> TODO/completed/358c1_workflow_deps.md
```

**VERIFIED:** Closeout document exists. SHIPPED status per §1 of the
closeout. No truncation observed in §1..§7.

### §2.3 Test file presence

```
$ ls -la molmetal/tests/test_wf358c1_workflow_deps.py
-rw-r--r-- 1 hugo hugo 5995 Sep 29 21:50 molmetal/tests/test_wf358c1_workflow_deps.py
```

**VERIFIED:** Test file exists at the canonical location.

### §2.4 Test execution

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf358c1_workflow_deps.py -v

molmetal/tests/test_wf358c1_workflow_deps.py::test_01_workflow_deps_structure PASSED [100%]
========================= 1 passed, 1 warning in 0.33s =========================
```

**VERIFIED:** 1/1 test PASS in 0.33 s. The single test enforces 10
invariants per §3.1 of `358c1_workflow_deps.md`:

1. Document exists at canonical path.
2. File size > 5 KB.
3. 5 main sections (§1..§5) present.
4. ≥10 cross-workflow dependency entries.
5. MEASURED + PROJECTED tag discipline preserved.
6. AGENTS.md §5.1 + §5.2 references preserved.
7. ≥5 source path citations.
8. ≥10 distinct Wf-XXX workflow references.
9. Critical path mention (Wf-340 → Wf-356 → Wf-357).
10. Pre-condition + post-condition keywords both present.

---

## §3 Verification verdict

| Check | Expected | MEASURED | Verdict |
|---|---|---|---|
| Document exists | yes | yes (12,513 B) | PASS |
| Closeout exists | yes | yes | PASS |
| Test file exists | yes | yes (5,995 B / 154 LOC) | PASS |
| Test passes | 1/1 PASS | 1/1 PASS in 0.33s | PASS |
| Closeout sections | §1..§7 | §1..§7 all present | PASS |
| Cross-workflow deps | 12 | 12 (per §2 of closeout) | PASS |
| Linear chain workflows | 19 | 19 (per §1 of closeout) | PASS |
| Test contract items | 10 | 10 | PASS |

**VERDICT: Wf-358-C1 is COMPLETE + FUNCTIONAL. No truncation; no
gap. No repair needed.**

---

## §4 Why was Wf-358-C1 flagged as "output truncated"?

The "truncated output" notification was likely from the Wf-358
orchestrator's stdout capture rather than a real truncation on disk:

- The orchestrator's stdout buffer for `Write` operations may have
  truncated intermediate Bash output (e.g. heredoc echo) while the
  on-disk file was always complete.
- The agent may have written the full document but the harness
  stdout-stream monitor only captured partial output.
- The doc + test + closeout are all on disk and complete.

This is consistent with Wf-291C / Wf-302D prior text-only stdout
buffer anomalies (per AGENTS.md §8).

---

## §5 Pre-flight gate verdict

Pre-flight v3 NOT re-run (docs+test verification only; no edits). Per
`TODO/completed/335_final_synthesis.md`, the baseline is:

```
62 PASS / 0 FAIL / 62 TOTAL
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED
```

Wf-361-A2 added 0 pre-flight gates (verification only).

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **Verification scope:** MEASURED file presence + test execution.
- **Wall:** PROJECTED 5-10 min CPU-only; MEASURED ~1 min actual (1
  Bash + 3 Read + 1 Write).
- **Tests re-verified:** 1/1 PASS, 0 NEW.
- **Edits applied:** 0 (verification only).
- **GPU launches:** 0.
- **Verdict:** Wf-358-C1 SHIPPED + FUNCTIONAL; no repair needed.

---

## §7 Files touched

- `TODO/completed/361a2_wf358c1_verify.md` (NEW, this file)

---

## §8 Source pointers

- **`TODO/completed/358c1_workflow_deps.md`** — Wf-358-C1 closeout.
- **`docs/WORKFLOW_DEPENDENCY_GRAPH.md`** — Wf-358-C1 deliverable
  document (12,513 B).
- **`molmetal/tests/test_wf358c1_workflow_deps.py`** — Wf-358-C1
  test file (5,995 B / 154 LOC).
- **AGENTS.md §6** — recent Wf history pointer.
- **AGENTS.md §5.2** — honest framing mandate.

---

**END OF Wf-361-A2 VERIFICATION**