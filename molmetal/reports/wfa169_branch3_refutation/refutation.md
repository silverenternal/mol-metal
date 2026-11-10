# Wf-A169 branch 3 — refutation (adversarial review)

**Verdict: REFUTED (partial).** CPU-only, 0 GPU (`CUDA_VISIBLE_DEVICES=""` on every
call), single venv `/home/hugo/codes/mol-metal/.venv/bin/python`, no `os.walk`,
`pre_flight_check.py` READ-ONLY, nothing written outside this directory.

**What is NOT refuted.** Every number the branch reports reproduces from disk:
live counts `all=129 launch=131 frontier=25 chemprop_v2_v3=18 cfm=8` + `counts_match=True`;
the census counters `24 / 9 / 15 / 4 / 9 / 6 / 5 / 14 / 4` and the 230 stale prose rows;
the 15 stale pin sites in 4 files; the 25-vs-4 split of the 29 `test_wf433a1` failures;
the `wfa159` stale-artefact diagnosis; `17 passed` on its own test file; the patch not
applied to the tree; the 24 gates really unregistered (24 of 32 missing from
`build_gates('all')`, re-derived).

**What IS refuted — the deliverable.**

1. **The patch no longer applies.** `git apply --check -p1 .../patch_resync_stale_pins.diff`
   -> **rc=1** on the live tree: the `test_wf433a1_gate_v2_wire.py` hunk fails because a
   concurrent wave (Wf-A172, 12:46:38, +115/-14) rewrote that file 17 min after the branch
   finished. `git apply` is atomic, so the patch applies **nothing** — including the 3 hunks
   that are still valid. Claim #5 ("`git apply --check -p1` = PASS") was true at 12:29 and
   is false now; the branch diagnosed this exact failure mode for `test_wf350c1` (item 10)
   without generalising it into a rebase step for its own patch.
2. **The repair mechanism is measurably inferior.** Wf-A172 moved the same assertions onto the
   SSOT-vs-runtime invariant (`EXPECTED_GATE_COUNTS['all'] == len(build_gates('all'))`), green on
   **both** the worktree (129) and HEAD (126). Branch 3 re-pins to the **uncommitted** worktree's
   129/131. On a clean-HEAD pre-flight (HEAD: `{all:126, launch:128, cfm:7}`) branch 3's 3
   still-valid hunks leave the **same 17** non-registration failures as not applying them —
   the 3 count-pin tests are RED on HEAD under 127 and under 129 alike.
3. **No apply-order precondition, and the wave's own README assigns this to branch 3.**
   `patch_wire_24_gates.diff` (same wave, same report dir) also applies cleanly (rc=0 against
   the live pre-flight sha `ae878bcd…`). With both applied: `build_gates('all')`=153 while
   `EXPECTED_GATE_COUNTS` stays 129 -> the pre-flight's **own** drift gate
   `p3_gate_count_consistency_audit` returns **FAIL** ("actual=153 vs expected=129
   [+DRIFT (silent gate bloat)]"), registered at `pre_flight_check.py:15667-15676`.
   `README.md:60-62` says the pins must go to 153/155 and that the witness test's pins
   "belong to the count-pin branch"; branch 3 bumps neither the pins nor the constant.
4. **The pre-flight was never run.** Measured now on the live tree: **117 PASS / 1 FAIL /
   11 SKIP / 0 WARN / 0 TIMEOUT / 129 total, rc=1**. The 1 FAIL is `p3_ssot_path_reference_check`
   (Wf-A131 guard: 5 of 17 assertions failing). The pre-flight is **RED today**, against
   AGENTS.md §1.4's PASS gate and §2's "118 PASS / 0 FAIL".

**Census defects (both harmless to the verdict).** `executable_pin_count` 24 / `ok_pin_count` 9
include 3 doc-derived pins (really 21 executable in 8 files + 3 doc-derived); `floor_guard_count`
6 undercounts — at least 7 further executable floor guards exist in 5 files
(`test_wf323c:263 >=57`, `test_wf346a1:173 >=63` behind `POST_REGISTRATION_ALL_FLOOR`,
`test_wf346a2:132 >=63`, `test_wf346a4:119 >=63`, `test_wf350a3:320 >=69`,
`test_wf353b1:244/:250 >=93`), all satisfied. The artifact is a 12:31 snapshot with no
as-of stamp: since 12:46 the tree reports 10 stale sites / 18 executable pins, not 15 / 24.

**Surviving concerns.** The patch needs a rebase (or the SSOT-vs-runtime form) before it can be
applied at all; its header must state the apply-order precondition and the required
`EXPECTED_GATE_COUNTS` bump to 153/155 if the wiring lands; no branch in this wave owns
`EXPECTED_GATE_COUNTS`; the live pre-flight is RED now; `floor_guard_count` and
`executable_pin_count` overstate/understate coverage; `scripts/gpu_pipeline_master.sh:149`
still logs a stale 66 (live 129) — outside this branch's ownership.

Full check-by-check evidence with file:line citations: `refutation.json`.
