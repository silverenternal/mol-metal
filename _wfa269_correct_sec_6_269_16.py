#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Two dated, additive corrections to the freshly published §6.269.16.

WHY THESE TWO, AND WHY ONLY THESE TWO
-------------------------------------
Every other MEASURED claim in §6.269.16 was re-verified against disk at
publication and came back EXACT:

  §6.269.16.2  the §5.9 gate        -> 31 wrappers scanned, REAL 31 / DUPLICATE 0   EXACT
  §6.269.16.2  the regression suite -> 23 passed                                        EXACT
  §6.269.16.3  the post-repair state-> `bash -n` OK, gate 31 REAL / 0 DUPLICATE         EXACT
  §6.269.16.4  the moved launcher   -> `git log --all` for molmetal/scripts/... is EMPTY,
                                       the archive path was added in bc29aa90           EXACT
  §6.269.16.4  the todo_ref         -> TODO-pending/ does not exist,
                                       TODO/pending/179_wf123_p3_w2_pic50_alignment.md does EXACT
  §6.269.16.5  the allow-list cap   -> `assert len(ALLOWLIST) <= 10` (line 688)        EXACT
  §6.269.16.5  the regression suite -> 8 passed                                         EXACT
  §6.269.16.7  the ruff measurement -> 2083 findings; PTH123 795 / PTH118 331 /
                                       PTH120 257 / PTH100 248                          EXACT
  §6.269.16.9  the surface guard    -> 15 files, 170 collected, 9 untracked             EXACT
  §6.269.16.1  the wave-2 journal   -> 20 started / 9 result / 11 failed;
                                       repair:G2 (ac4c903a071f64818) has an EMPTY payload EXACT

TWO claims are wrong, and BOTH have the same root cause: **the wave's ledger
was written before the wave finished.**

  1. §6.269.16.1 claims wave-5 produced "ELEVEN result rows for TWELVE started
     agents".  The FINAL journal reads **13 started / 13 result**, every
     payload non-empty (3 512 - 6 145 B).  The measurement was taken at 19:24
     while verify:306b was still running (its transcript was still growing) and
     a 13th agent (the critic) was started afterwards.
  2. §6.269.16.6 claims a "2:2 SPLIT" over "four verdicts".  The journal holds
     **SIX** verifiers, and the split is **THREE CONFIRMED : THREE REFUTED**:
     verify:306a REFUTED, verify:t179 REFUTED, verify:todo220 REFUTED,
     verify:wf74 CONFIRMED, verify:wf264 CONFIRMED, verify:306b CONFIRMED.
     verify:306b is not mentioned anywhere in the entry.

⇒ THE ENTRY'S OWN THESIS IS STRENGTHENED, NOT WEAKENED: the rule it states
("a `started` row is not a verdict") was vindicated twice — the mid-wave
ledger was wrong about the agent COUNT and wrong about the verdict SPLIT.

Both corrections are ADDITIVE + DATED.  Every superseded rendering — the
heading, the table, the split sentence — is preserved VERBATIM inside the
correction block (AGENTS.md §5.5 rule #6).

THE VERIFICATION IS A **SUBSTRING** TEST, NOT A LINE-SUBSEQUENCE TEST.  A
preserved rendering inside a blockquote gains a ">     " prefix, so it is no
longer byte-equal as a LINE; asserting line-equality would have rejected a
correct preservation.  The invariant that matters is: **no original line's
text disappears from the file.**
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402

CATALOG = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"

# ---------------------------------------------------------------- §6.269.16.1
OLD_H_161 = (
    '#### §6.269.16.1 — "spawned ≠ ran": wave-5 produced ELEVEN result rows for'
    " TWELVE started agents"
)
NEW_H_161 = (
    '#### §6.269.16.1 — "spawned ≠ ran": wave-5\'s ledger claimed ELEVEN result rows'
    " for TWELVE started agents — its FINAL journal reads THIRTEEN for THIRTEEN"
)
OLD_TBL_161 = (
    "| journal | started | result | failed |\n"
    "|---|---|---|---|\n"
    "| wave-5 `wf_b05d1b2b-1ca` | **12** | **11** | 0 |\n"
    "| wave-2 `wf_a315d33f-e34` | **20** | **9** | **11** |\n"
)
NEW_TBL_161 = (
    "| journal | started | result | failed | empty payloads |\n"
    "|---|---|---|---|---|\n"
    "| wave-5 `wf_b05d1b2b-1ca` | **13** | **13** | **0** | **0** |\n"
    "| wave-2 `wf_a315d33f-e34` | **20** | **9** | **11** | **1** (`repair:G2`) |\n"
)

CORR_161 = """
> **CORRECTION (2026-11-02, the main thread, on publication of this entry) — THE LEDGER WAS WRONG ABOUT THE AGENT COUNT, EXACTLY AS THE RULE BELOW PREDICTED.**
> **SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:**
>     #### §6.269.16.1 — "spawned ≠ ran": wave-5 produced ELEVEN result rows for TWELVE started agents
>
>     | journal | started | result | failed |
>     |---|---|---|---|
>     | wave-5 `wf_b05d1b2b-1ca` | **12** | **11** | 0 |
>     | wave-2 `wf_a315d33f-e34` | **20** | **9** | **11** |
>
> **MEASURED at publication (2026-11-02) — the wave-5 journal re-read row-by-row, `…/subagents/workflows/wf_b05d1b2b-1ca/journal.jsonl`:**
>
> | journal | started | result | failed | empty payloads |
> |---|---|---|---|---|
> | wave-5 `wf_b05d1b2b-1ca` | **13** | **13** | **0** | **0** |
> | wave-2 `wf_a315d33f-e34` | **20** | **9** | **11** | **1** (`repair:G2`) |
>
> ⇒ **THE WAVE-5 ROW IS NOW 13/13, NOT 12/11.** All thirteen agents carry a `result` row and **every payload is non-empty** (MEASURED: 3 512 – 6 145 B). The "**12** started / **11** result" reading was taken at **19:24**, while `verify:306b` was still running — its transcript was still growing (4 042 397 B at 19:24) and finished at **19:27** (4 202 293 B) — and a **thirteenth** agent (`critic`) was started **after** the ledger was written down. The table above is therefore a **mid-wave snapshot**, not the wave's final state.
> ⇒ **THE WAVE-2 ROW IS UNCHANGED AND EXACT**: 20 started / 9 result / 11 failed, with the one empty payload on `repair:G2` (`ac4c903a071f64818`) — MEASURED, re-confirmed at publication.
> ⇒ **⇒ THE RULE BELOW IS NOW DEMONSTRATED TWICE OVER.** The mid-wave ledger was wrong about the agent **COUNT** (§6.269.16.1) **and** wrong about the verdict **SPLIT** (§6.269.16.6) — both for the same reason: it was written before the wave finished. A counter that stops at `started` rows was wrong twice; a counter that counts **`result` rows whose payload is non-empty** is right.
""".rstrip("\n")

# ---------------------------------------------------------------- §6.269.16.6
OLD_H_166 = (
    "#### §6.269.16.6 — THE TWO CONFIRMED VERDICTS, AND WHAT THE 2:2 SPLIT MEANS"
)
NEW_H_166 = (
    "#### §6.269.16.6 — THE **THREE** CONFIRMED VERDICTS, AND WHAT THE **3:3** SPLIT"
    " MEANS"
)
OLD_SPLIT_SENTENCE = (
    "Two of the wave's four verdicts say the repair is right; two say it is not."
)
SPLIT_MARKER = (
    " **⟨CORRECTED 2026-11-02 (the main thread, on publication of this entry) — the"
    " split is **THREE CONFIRMED : THREE REFUTED** over **SIX** verifiers, not 2:2 over"
    " four; the MEASURED table is in §6.269.16.6.1 below. The sentence above is"
    " PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.⟩**"
)

CORR_166 = """
#### §6.269.16.6.1 — THE SPLIT IS **3:3 OVER SIX VERIFIERS**, NOT 2:2 OVER FOUR (the main thread, 2026-11-02, on publication)

> **CORRECTION (2026-11-02, the main thread, on publication of this entry) — THE SPLIT WAS MIS-COUNTED, AND ONE VERDICT WAS OMITTED ENTIRELY.**
> **SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:**
>     #### §6.269.16.6 — THE TWO CONFIRMED VERDICTS, AND WHAT THE 2:2 SPLIT MEANS
>
> **MEASURED at publication (2026-11-02) — every `verify:*` result payload in the wave-5 journal, with its leading verdict token:**

| verifier | agentId | verdict | payload bytes |
|---|---|---|---|
| `verify:306a` | `a2b27ac64f7b18d9d` | **REFUTED** | 4 706 |
| `verify:t179` | `a72f59095a2a199b9` | **REFUTED** | 5 638 |
| `verify:todo220` | `a6160e45511184a2a` | **REFUTED** | 5 621 |
| `verify:wf74` | `ae3834b131949588f` | **CONFIRMED** | 4 207 |
| `verify:wf264` | `a6448d70d55ec9535` | **CONFIRMED** | 4 833 |
| `verify:306b` | `a30bac9c7643e5e89` | **CONFIRMED** | 4 081 |

> ⇒ **THE TRUE SPLIT IS THREE CONFIRMED : THREE REFUTED, OVER SIX VERIFIERS.** The entry said "**four** verdicts" and a "**2:2**" split, and it **never mentioned `verify:306b`** — the sixth verifier, whose verdict is CONFIRMED. The five verdicts the entry does discuss (306a REFUTED, t179 REFUTED, todo220 REFUTED, wf74 CONFIRMED, wf264 CONFIRMED) are themselves a **3:2**, so the "2:2" figure did not follow even from the entry's own list.
> ⇒ **AND ONE AGENT ID IN §6.269.16.5 IS TRUNCATED BY ONE CHARACTER.** The entry gives `verify:306a`'s agentId as `a2b27ac64f7b18d9`; MEASURED, the full id is **`a2b27ac64f7b18d9d`**. The other two ids the entry quotes — `a72f59095a2a199b9` for `verify:t179` (§6.269.16.4) and `a6160e45511184a2a` for `verify:todo220` (§6.269.16.3) — are **exact**, as is `a30bac9c7643e5e89` for `verify:306b` (§6.269.16.1).
> ⇒ **⇒ THE ENTRY'S THESIS IS STRENGTHENED, NOT WEAKENED.** *"A wave in which every verifier confirms its repair has verified nothing"* was true: **half** the wave said no. And the mis-count has the same root cause as §6.269.16.1 — **the ledger was written mid-wave**, before `verify:306b` had returned (§6.269.16.1's correction) — so the ledger's *verdict tally* was as stale as its *agent tally*. **A ledger written from a live wave is a snapshot; a ledger written from a closed journal is a record.**
""".rstrip("\n")


def main() -> int:
    original = CATALOG.read_text(encoding="utf-8")

    for anchor in (OLD_H_161, OLD_TBL_161, OLD_H_166, OLD_SPLIT_SENTENCE):
        assert original.count(anchor) == 1, f"anchor missing/ambiguous: {anchor[:60]!r}"

    # --- §6.269.16.1 ---------------------------------------------------------
    # INSERT the correction block FIRST (anchored on the original heading), THEN
    # tense-correct the heading.  The heading is the earlier occurrence, so the
    # `.replace(..., 1)` below hits it and not the copy quoted inside the block.
    out = original.replace(OLD_H_161 + "\n\n", OLD_H_161 + "\n\n" + CORR_161 + "\n\n", 1)
    assert out.count(OLD_H_161) == 2, "the superseded heading copy did not land"
    out = out.replace(OLD_H_161, NEW_H_161, 1)
    out = out.replace(OLD_TBL_161, NEW_TBL_161, 1)

    # --- §6.269.16.6 ---------------------------------------------------------
    out = out.replace(
        OLD_H_166 + "\n\n",
        OLD_H_166 + "\n\n" + CORR_166 + "\n\n",
        1,
    )
    assert out.count(OLD_H_166) == 2, "the superseded §6.269.16.6 heading did not land"
    out = out.replace(OLD_H_166, NEW_H_166, 1)

    # The marker goes at the END OF THE LINE, not immediately after the
    # sentence.  The sentence is the MIDDLE of a paragraph line, so inserting
    # there would split the line and the original line's text would no longer
    # exist contiguously anywhere in the file — which is exactly what the
    # "nothing lost" invariant below checks for.  Appending at the line end
    # keeps the whole original line as a verbatim PREFIX.
    lines = out.split("\n")
    hits = [i for i, ln in enumerate(lines) if OLD_SPLIT_SENTENCE in ln]
    assert len(hits) == 1, f"the split sentence is on {len(hits)} lines, expected 1"
    lines[hits[0]] = lines[hits[0]] + SPLIT_MARKER
    out = "\n".join(lines)

    # --- verification: NO original line's text may disappear ----------------
    old_lines = original.splitlines()
    new_text = out
    lost = [ln for ln in old_lines if ln.strip() and ln.strip() not in new_text]
    assert not lost, f"{len(lost)} original line(s) lost: {lost[:3]}"
    assert out.count(OLD_SPLIT_SENTENCE) == 1, "the superseded split sentence must survive"
    assert SPLIT_MARKER in out
    assert CORR_161 in out and CORR_166 in out
    assert "| wave-5 `wf_b05d1b2b-1ca` | **13** | **13** | **0** | **0** |" in out
    assert "| wave-2 `wf_a315d33f-e34` | **20** | **9** | **11** | **1** (`repair:G2`) |" in out
    assert out.endswith("\n")

    atomic_write_file(CATALOG, out, encoding="utf-8")

    print(f"CATALOG {len(original)} -> {len(out)} chars (+{len(out) - len(original)})")
    print("  §6.269.16.1  heading tense-corrected; original heading + table preserved verbatim in-block")
    print("  §6.269.16.6  heading tense-corrected; §6.269.16.6.1 verdict table; inline dated marker")
    print(f"  no original line's text lost: True ({len(old_lines)} original lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
