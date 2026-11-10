# Wf-A166 Branch 2 — REFUTATION (adversarial review)

**VERDICT: NOT REFUTED.** Every load-bearing number in the branch report was re-derived
independently from disk and matched EXACTLY. One supporting artifact (the blast-radius list)
is INCOMPLETE — that is the surviving concern, not a refutation.

## Checks

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | branch's own claims re-derived | PASS | 418 rows / 396 metrics / 396 latest / 22 superseded / 6 shas / 398 paths / 416 shas; live validator run BYTE-IDENTICAL to the published audit (0.90 s); 0/396 touched; 385/11 sha-vs-disk; 388/7/1 content-at-commit; 975414bc exists but unreachable from `--all` (reflog only) |
| 2 | regression test bites | PASS | 11 passed; mutation of reachability -> `test_negative_control_unreachable_commit_is_flagged` FAILS; mutation of commit types -> `test_negative_control_missing_commit_is_flagged` FAILS; mutation of the touched-axis -> `test_negative_control_untouched_source_is_flagged` FAILS |
| 3 | validator not vacuous | PASS | bad row (sha 000..0) -> exactly 1 `commit_unresolvable`; good row (sha = the commit that authored the file) -> 0 strict-axis violations; cross-check vs `uncommitted_provenance` = 418/396/416/398/0 open, 9 superseded |
| 4 | blast radius COMPLETE | **FAIL** | 2 hard pins missed: `test_wf433a1_gate_v2_wire.py:108` (`EXPECTED_GATE_COUNTS["all"] == 127`) and `:120` (`len(set(names)) == 127`); `test_wf350c1_gpu_env_probe.py:334` (`EXPECTED_GATE_COUNTS["all"] == 94`) |
| 5 | patch is safe, not applied | PASS | `git apply --check` PASS for both diffs (scratch repo AND the current 11:54 file), `git apply` + py_compile PASS; working tree still 129/131 with no `p3_run_ledger_git_provenance`; ledger sha256/mtime unchanged |
| 6 | would it block launches? | PASS | baseline `--pipeline all` EXIT 0 (118 PASS / 0 FAIL / 11 SKIP / 129); STRICT patch EXIT 1 (1 FAIL, OPEN_metrics=396); RECOMMENDED patch EXIT 1 (OPEN_metrics=8); + `WF_RUN_LEDGER_GIT_PROVENANCE=0` -> SKIP |
| 7 | semantic diagnosis | PASS | `run_ledger.py:201-202` sets `git_sha = git rev-parse HEAD`; the provenance-TRUE rule independently gives exactly the 8 named metrics |

## Surviving concerns

1. **Blast radius incomplete** — the two files above are ALREADY RED (test_wf433a1: 29 failed /
   11 passed; test_wf350c1: 3 failed), so the branch's claim #9 ("pre-existing drift = exactly
   2 files: test_wfa128, test_wfa127") undercounts it — at least 4 files. Their pins must move
   to 130/132 in the same edit.
2. **Report says "17-entry", artifact has 20** (B1..B20) — MEASURED-vs-PROJECTED mismatch.
3. **Wrong date on the 8 open metrics** — they are the 2026-10-20 rows (sha `e0d473cc7bae`)
   plus row #408 (2026-10-21); the 2026-10-08 rows are the SUPERSEDED predecessors.
4. **"Unrepairable" overstatement** — appending a correction row per metric repairs the strict
   rule (396 appends).
5. `test_wf415a3_preflight_fail_audit.py:151` (assertion message) and `:131-138` (docstring)
   still say 129; `test_wfa159_timeout_verdict_census.py:33` docstring says 129/131.
6. `pre_flight_check.py` mtime moved 10:16 -> 11:54 during the review (a concurrent agent);
   the branch's substantive claim (patch NOT applied) still holds.

Detail: `molmetal/reports/wfa166_b2_refutation/refutation.json`
