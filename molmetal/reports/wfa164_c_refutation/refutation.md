# Wf-A164 -- adversarial refutation of branch C

**VERDICT: NOT REFUTED**

Branch C: `molmetal/scripts/wfa156_bond_threshold_frontier.py` (the EF-TALFM bond-threshold
measurement tool). 0 GPU, CPU-only, single venv.

## Checks

| # | check | result |
|---|---|---|
| 1 | the branch's own claims | PASS (one partial overstatement) |
| 2 | the regression test really bites | PASS |
| 3 | the guard is not vacuous | PASS |
| 4 | nothing broke | PASS |
| 5 | scope discipline | PASS |
| 6 | extra: Wf-A163 census no longer arms this file | PASS |

1. **The six claimed literals are gone and the replacements are COMPUTED.** The pre-repair
   baseline is on disk, not reconstructed:
   `molmetal/reports/wfa156_bond_threshold_frontier/report.json` carries the note verbatim
   (`529/2112 ... 2082/2112 ... 30/2112 ... 0/2112`), the caveat's `2,112-row`, the default pool
   path and `every one of our pool rows carries a metal (metal_ratio 1.0)`; the Wf-A163 census
   agrees (`wfa163_narrative_truth_scan/report.md:187`, L1008-L1010). Repaired: note is an
   f-string over machine fields (`wfa156_bond_threshold_frontier.py` L1389-1397) and reads
   `3/6 ... 5/5 ... 0/5 ... 1/6` on a 6-row synthetic pool and
   `1078/12669 ... 9612/12668 ... 3056/12668 ... 0/12669` on the 12,669-row c7 pool; the c7
   caveat reads `12668 of our 12669 pool rows carry a metal` (machine: `rows_with_metal_frac`
   = 0.9999210671718368 -- the old claim was FALSE for c7). The master.log quotes and the
   run-dir listing are now read from disk (`read_run_log_excerpt` L1070, `read_run_dir_listing`
   L1110) and match the real files exactly.
2. **The test bites.** `pytest molmetal/tests/test_wfa164_bond_threshold_narrative.py` -> 10
   passed (`CUDA_VISIBLE_DEVICES=""`). Independent negative control: a pytest plugin that
   restores the census-recorded pre-repair behaviour makes 3 of the 10 FAIL
   (`test_synthetic_pool_artifact_has_no_stale_default_pool_literal`,
   `..._carries_its_own_numbers`, `test_copied_pool_under_a_different_directory_name`).
3. **The guard fires.** A fresh unbacked number (`424242`) -> `pass=False`,
   `unbacked_numbers={'424242': ...}`. A foreign pool path -> `pool_path_scan pass=False`.
   End-to-end: `main()` -> rc=1, `report.json` `status='FAIL'` on disk, `report.md` and
   `per_row.*` not written.
4. **Nothing broke.** Default pool -> rc=0, PASS, 46 prose numbers / 0 unbacked / 0 foreign.
   Independent strict diff vs the pre-repair artefact: 1,266 measurement leaves compared,
   **0 changed, 0 removed**; markdown 313 -> 314 lines with **27** changed diff-body lines
   (the branch's exact figure). The ORIGINAL tool test
   `molmetal/tests/test_wfa156_bond_threshold_frontier.py` still passes 11/11.
5. **Scope.** `find` over `metrics registry paper docs TODO configs
   molmetal/reports/run_ledger.jsonl -newermt "2026-11-01 11:30"` -> **zero files**. C's
   footprint: the tool, its own test, its own report dir.

Branch numbers traced to disk: AST re-enumeration gives 1,944 lines / 1,053 string literals /
186 digit-bearing (exact); `"92.7"` -> 0 hits in the fetched 1,234,045-byte HTML while
`"92.98"` -> 3 and `"94.33"` -> 5; the census tool-layer scan now reports 2 armed scripts and
`wfa156_bond_threshold_frontier.py` is NOT among them (was 3 sites at L1008-L1010).

## Surviving concerns

1. **Moderate -- the guard checks BACKING, not CORRECTNESS.** A false note
   (`6/6 rows with one fragment` when the truth is `3/6`) PASSES the guard, because any of the
   ~1,211 machine numbers or 89 machine strings counts as backing. The tool's docstring L105-106
   ("The class is stopped here") overstates what the guard does.
2. **Low -- a FAIL run leaves a stale `report.md`** in a reused out-dir (report.md is written only
   on the PASS branch, L1888-1890); the branch's test never checks its absence.
3. **Low -- `render_markdown` L1557** still asserts `directory holds no coordinate dump`
   unconditionally: a hard-coded pool-specific prose claim outside `NARRATIVE_PATHS`. True for
   all 12 pools on disk, so latent.
4. **Low -- the `read_run_log_excerpt` docstring (L1074-L1075)** still transcribes `518,276
   params, cutoff=3.5 A` / `threshold=0.500` (verified accurate against the real master.log,
   not emitted into any artefact), contradicting the branch's own L362-364 policy.
5. **Low -- `FRONTIER_BAND "92.7-94 %"`** left in place; report.md L44 still calls it "the
   92.7-94 % frontier band" without labelling it a restatement.
6. **Low -- the branch's `1,453 lines pre-repair` denominator is the one number not traceable to
   disk** (the script is untracked and was overwritten in place; no pre-repair copy or .pyc).

Details: `refutation.json` (same directory). Scratch/negative-control sources:
`/tmp/wfa164_refute/{prepatch.py,guard_probe.py,probe_c.py}`.
