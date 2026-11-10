# Wf-A166 Branch 1 -- REFUTATION (adversarial review)

**VERDICT: NOT REFUTED.** Every headline number in the branch report re-derives EXACTLY under a
third, independent method. The regression test passes 5/5 and bites on 8/8 mutations; the
validator is non-vacuous; the blast radius is complete for this branch's scope; the branch did
not touch the ledger and ships no patch. ONE concern is material to the branch's proposed repair
column; two are minor.

MEASURED 2026-11-01. CPU-only, 0 GPU. READ-ONLY on the ledger. Scratch under `/tmp/refute166`,
`/tmp/sc166`.

## Checks

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | branch's own counts re-derived | PASS | 418 rows / 396 metrics / 398 paths / 6 shas; A 387 (31 stale) / B 409 / C 418 / D 1 (line 23) / E 417 (line 408 = 975414bc unreachable) / F 384 = d86c879a (91.9%) / G data 30-30-30, config 6 present, 5 resolvable, 0 match; 0 missing sources; 398 paths / 754 blobs, resolvers AGREE. `git show --numstat f55f063e` = 386 lines added in ONE commit (ledger 22 -> 408 lines), 384 of them d86c879a. The 9 never-committed shas match NO blob among all 17,490 objects in the DB. A==B 0/54, A==C 0/418, B==C 0 |
| 2 | regression test bites | PASS | 5 passed / 0.66 s CPU-only; 8/8 mutations bite (axes A,B,C,D,E forced True; recoverability forced 0; resolver DISAGREE suppressed; distinct_git_sha forced to 1). Mutant re-bound in BOTH `sys.modules` and the package attribute, with `REPO_ROOT`/`LEDGER_PATH`/`BY_METRIC_DIR`/`PROTOCOLS_PATH` repointed, so git did not fail rc=128 for the wrong reason |
| 3 | validator not vacuous | PASS | synthetic GOOD row (real on-disk sha, git_sha=HEAD) -> A/B/C/E all True, NOT flagged; synthetic BAD row (`b`*64, `0`*40) -> all False and `rows_with_NO_recoverable_producing_commit == 1`. Shipped artifact self-consistent: len(rows)=418, every axis_count in [0,418], every row has a distinct 1-based `line` |
| 4 | blast radius COMPLETE | PASS | `wfa156_text_scan.py` (not PATH grep): `registry.py` + `registry/*.yaml` ledger hits = **0**; `pre_flight_check.py` git_sha hits = **0** (both ledger gates, :9814 and :9920, check `source_json_sha256`). The only three git_sha READ sites repo-wide are echoes: `run_ledger.py:547`, `run_ledger.py:874`, `wfa156_ledger_provenance_guard.py:576`. All 30 column-contract line citations verified against the working tree. Census script writes ONLY its own output dir |
| 5 | patch safe + not applied | PASS | This branch ships **no patch** (`wfa166_run_ledger_census/` = census.json + census.md only) and applied none: ledger `git diff --numstat` = `10 0` (the prior wave's 10 rows), ledger mtime 2026-10-31 09:54:13 and `run_ledger.py` mtime 2026-10-31 09:54:05 (the PRIOR wave's +411-line Wf-A123 FIX 7 block) both PREDATE the census artifacts at 11:35-11:36. For completeness the SIBLING branch's two diffs were applied on a scratch copy: `git apply --check` PASS for both, `git apply` + `ast.parse` PASS for both |
| 6 | would it block? | PASS | This branch adds NO gate, so it blocks nothing and changes no gate-count pin. The number it DOES establish: a gate requiring "the recorded git_sha touched this row's own source_json" would **FAIL 417/418 rows (99.8%)**, PASS 1/418 -- and that single pass is a superseded row (latest-rows-only D = 0/418). For the branch's own proposed repair: 409/418 rows get a holder under its method, but only **404/418 get a UNIQUE** one (see concern 1). Axis-B-style gate: 9/418 fail, all superseded, OPEN = 0 |

## Surviving concerns

1. **MATERIAL -- "exactly ONE holding commit" is an artifact of git history simplification.**
   `wfa166_run_ledger_census.py:379` enumerates holders with `git rev-list --all -- <path>`, which
   prunes TREESAME merges. Merge `8289a7f54726` ("Merge Wf-A17 Track A natural-charge 9-layer
   wire-in", 2026-10-13 20:26:52, parents `0e232defe64e` / `85cc90102b59`) HOLDS the same blob as
   its first parent but is pruned. Re-derived with `--full-history`, the holder-count distribution
   over the 418 rows changes from `{0: 9, 1: 409}` to `{0: 9, 1: 404, 2: 5}`. The 5 affected rows
   are **L54** `cfm_cross_dataset_transfer`, **L57** `cfm_decode_real`, **L68**
   `cfm_internal_diversity_intdiv1`, **L74** `cfm_loss_term_liveness`, **L310** `platinai_4cell_auc`.
   For **all 5** the "latest holder" flips to `8289a7f54726`, so the value the proposed new column
   would carry DIFFERS between the two methods.
   *Scope:* no axis count changes (A..G identical under both; A==C stays 0/418; blob set identical
   -- 754 blobs, 0 new under `--full-history`); the verdict that `git_sha` is not a provenance
   pointer is unaffected. But the prose "UNIQUE for 409/418 rows (exactly one holding commit each)"
   is overstated by 5 rows, and the recommended `source_json_commit` column is **ambiguous for
   5/418 rows** unless the holder definition is pinned (state `--full-history` explicitly, or name
   the column to say the merge is excluded).
   Reproduce: `git rev-list --all -- metrics/by_metric/cfm_decode_real.json` -> 1 commit;
   add `--full-history` -> 2.
2. **MINOR -- report prose "54/417 rows"** uses 417 (the axis-E reachable count) where the row
   denominator is 418. Independently measured: 54 rows carry `provenance.git_commit` (54 distinct
   paths, 54/54 with a reachable ledger `git_sha`). The SHIPPED artifacts are correct
   (`rows_with_a_candidate_B = 54` against the 418 denominator, census.md line 76) -- only the
   report prose is off.
3. **MINOR -- the D-hit and the 9 never-committed rows are all SUPERSEDED.** Line 23 is superseded
   by line 410; each of the 9 never-committed rows has a later row for its metric_id, so OPEN = 0.
   `census.json` carries `is_latest_row_for_metric` per row (False at line 23, True at line 408), so
   the artifact supports the distinction -- but the report prose presents "D = 1 of 418" and
   "9 of 418 unrecoverable" without saying **0 CURRENT rows** are affected, which over-reads severity.

## Explicitly NOT found

No fabricated number; no off-by-one in the denominator; no vacuous validator; no vacuous
regression test; no ledger mutation by this branch; no path writes beyond the census output dir;
and NO validator of `git_sha` anywhere in the repo (which corroborates rather than refutes the
branch's central claim).

Detail: `molmetal/reports/wfa166_run_ledger_census_refutation/refutation.json`
