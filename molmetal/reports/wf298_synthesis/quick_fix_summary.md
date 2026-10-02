# Wf-298E — Quick Fix Synthesis

**Date**: 2026-09-29
**Parent**: Wf-291C pipeline-regression mitigation
**Owner**: Hugo Lee + workflow harness

---

## Per-agent summary

| Agent | Task | Status | Key result |
|---|---|---|---|
| **Wf-298A** | Fix `'elapsed is not defined'` in 3 launch wrappers | **HONEST-NEGATIVE** — bug does not exist | `bash -n` SYNTAX OK on all 3 (`g4`/`g5`/`g6`); `g5 --dry-run` enumerates 15 (metal, seed) pairs and exits 0. The described bug pattern was computed against the wrong file surface (Python helper, not bash). No edits applied. |
| **Wf-298B** | Multi-root patch for `wf278a_g4_aggregate.py` | **COMPLETED** | `~90 LOC` patch adds `--input-dirs` (multi-root, last-wins-by-mtime dedup) + `--expected-pairs default=15`; **3/3 tests pass** in new `test_wf278a_multi_root.py`. Cascade G4+G5 yields 5-metal `test_auc=0.7243 ± 0.0460` (n=15, vs legacy 14). |
| **Wf-298C** | Re-emit stale-read guard tests (Wf-285D) | **COMPLETED** | `molmetal/tests/test_stale_read_guard.py` shipped with **8/8 tests pass** (4 named + 4 parametrised regex cases); wall-time 1.17 s CPU-only. Closes the *testable* slice of Wf-285D HN-1. Python reference impl (`parse_epoch_line` + `wait_until_epoch_reached`) bundled in the test file. |
| **Wf-298D** | `StructuredOutput` schema-shape harness detector | **PARTIAL** | `molmetal/reports/wf298d_harness_detector/wrapper.py` DELIVERED. `HarnessOutputExtractor.detect_schema_shape()` catches **4/4 regression-class payloads** (Wf-270A/271D/272A/273B) with stable reason codes (`properties_is_schema_object` / `hybrid_empty_string_properties` / `nested_object_with_schema_type_keyword`). 7/8 strict pass rate (1 contrived name+type false-positive). The closed-source `claude-code` ELF binary is NOT user-modifiable — only the post-extraction layer is fixable from userland. |

## Critical remaining

1. **Wf-291H** — `read_anchor_json(path, max_age_sec=..., required_version=...)` Python helper still MISSING on disk. Wf-298A redirected to this; Wf-298C §4 marked it DEFERRED. Closes the real Wf-271B + Wf-287A m14 anchor re-use risk pattern that motivated Wf-285D / Wf-291B.
2. **Bash `wait_until_epoch_reached()` in `scripts/launch_wf124_p3_g5.sh`** — still MISSING (Wf-298C §4 DEFERRED). Wf-298C ships only the Python reference + tests, not the bash wrapper.
3. **Wf-298E** — prompt-text-only task briefs (Wf-291C §5.4 option a). Would sidestep the `StructuredOutput` regression class entirely. Not started.
4. **Wf-298F** — retry-with-explicit-correction wrapper (Wf-291C §5.4 option b). Requires harness-level hooks; not started.
5. **Closed-source prompt-pressure layer** — the ELF binary's ` You MUST call the  tool` injection string still fires on first response; Wf-298D only fixes the *post-extraction* layer.

## Next priorities (8-12 lines)

1. Land **Wf-291H** (~45 min CPU): ship `molmetal/scripts/read_anchor_json.py` + `molmetal/tests/test_read_anchor_json.py` (3-4 tests: fresh / stale / version-mismatch / missing-path). Closes Wf-271B + Wf-287A risk and unblocks downstream m14 / cfm_decode_real / per_metal_pass consumers.
2. Add `wait_until_epoch_reached()` bash function to `scripts/launch_wf124_p3_g5.sh` (~30 min). Re-target the Wf-298C Python reference at it via `subprocess` in a follow-up test file; keep the current 8/8 contract tests as a regression suite.
3. Re-dispatch **Wf-298E** (text-only task briefs): refactor the parent workflow's task-brief generator to emit schemas as plain-text prompts instead of `StructuredOutput` wrappers. Cost: each agent loses 1 round of structured-validation retries; benefit: bypasses the regression class entirely.
4. Re-dispatch **Wf-298F** (retry-with-explicit-correction): wrap the `StructuredOutput` call site in a retry helper that, on schema-mismatch, re-emits the prompt with an explicit "respond with VALUES, not schema" instruction. Wf-298D wrapper is the post-extraction safety net; Wf-298F is the upstream prompt-side mitigation.
5. Wire **Wf-298D wrapper** into the Wf-XXXE synthesis slot: each upstream agent's `result` string runs through `HarnessOutputExtractor.detect_schema_shape()` first; on `True`, the agent's 1-line free-text summary becomes the recovered payload (Wf-274/280/282/284 workaround path).
6. After 1-2 weeks, audit `molmetal/reports/wf298_synthesis/` for `strict_pass_rate` regression (target: 8/8 with the contrived edge case re-classified as a non-issue once real-data noise dominates).

**End of Wf-298E synthesis.**