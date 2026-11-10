# Wf-A125 — ADVERSARIAL VERIFICATION REPORT

Verifier: independent probe agent (READ-ONLY on source; this report is the only file written).
Host: CPU-only session, every python/pytest invocation prefixed `CUDA_VISIBLE_DEVICES=""`.
HEAD: `1b9470a0`. Design plan `.claude/wf/wfa125_DESIGN.md` sha256 `a44bca9a0503214d9c232b68bb7538e59c439a1297f068e917ef35c9826392c9` (VERIFIED).

Every number below was MEASURED on this machine during this verification. Probes are in
`/tmp/wfa125_probe/` (outside the repo). "revert-caught?" = would the wave's own test for that
gap FAIL if the defect were re-introduced? Answered by EMPIRICALLY re-introducing the defect
(monkeypatching the guard back to its pre-wave behaviour) wherever possible.

---

## G1 — repo-root CPU lock covers the top-level `tests/` tree

**Verdict: PARTIALLY-FIXED** (mechanism works on this host; the fix CANNOT SHIP).

MEASURED probes:
1. `PYTHONPATH=/tmp/wfa125_probe CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -p envprobe
   --collect-only -q tests/test_fused_ops.py` → the injected plugin printed `PROBE_CVD=''`
   (the caller asked for GPU 0; the root conftest overrode it) and `mol-metal: CPU lock ON
   (CUDA_VISIBLE_DEVICES='')`. The root conftest IS an ancestor of `tests/`, so the lock reaches it.
2. `MOLMETAL_TEST_ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=0 ... --collect-only tests/test_fused_ops.py`
   → `mol-metal: CPU lock OFF (MOLMETAL_TEST_ALLOW_GPU=1, CUDA_VISIBLE_DEVICES='0')`. The opt-out
   preserves the caller's value.
3. **SHIPPING BLOCKER (MEASURED):** `git check-ignore -v conftest.py` → `.gitignore:78:/conftest.py
   conftest.py`; `git cat-file -e HEAD:conftest.py` → `fatal: path 'conftest.py' exists on disk, but
   not in HEAD`. The root conftest is untracked AND ignored. A fresh clone has NO root conftest and
   the `tests/` tree runs unlocked. `.gitignore` was already dirty before the wave (the diff is a
   Wf-A114 cluster-D block, not a wave edit), so the wave did not introduce it — but it left the
   blocker in place.

Residual: **the G1 fix is a working-tree-only artifact.** Same root cause as G9.

Revert-caught? **YES** for the defect (`molmetal/tests/test_wfa125_root_conftest_gpu_lock.py`,
8 tests, all fail if the root conftest is deleted or its lock line is removed) / **NO** for the
shipping gap (no test asserts the root conftest is *tracked*; `git ls-files` never sees it).

---

## G2 — curated `PUBLISHED_FILE_PATHS` (registry SSOT + ledger SSOT)

**Verdict: VERIFIED-FIXED** (as scoped by the plan; one acknowledged residual).

MEASURED probes:
1. `PUBLISHED_FILE_PATHS` = `registry/metrics.yaml`, `registry/datasets.yaml`,
   `registry/protocols.yaml`, `molmetal/reports/run_ledger.jsonl` (4 entries, each with
   `authorised_writers`).
2. END-TO-END: a pytest plugin rewrote `registry/metrics.yaml` with IDENTICAL bytes (in place,
   then via `os.replace`) inside a session over `molmetal/tests/`. The session-scoped guard fired
   at teardown: `PUBLISHED TREE WRITE GUARD ... - registry/metrics.yaml (touched)`, identical
   before/after sha256, and **pytest exit code 1**. The registry file's bytes are unchanged
   (sha256 `f39d901f…` before and after).
3. Unit: a curated file with no authorised writer → 1 violation; with
   `molmetal/tests/test_registry_generate_import_path.py` active → 0 violations.
4. No false positives: `test_registry_generate_import_path.py` + `test_registry.py` → 12 passed,
   session clean. `test_metric_citability_rule.py` → 86 passed, clean (its writes go to `tmp_path`).

Residual (plan open question 1, acknowledged): `registry/narrative_claims.yaml` and
`registry/file_provenance.json` exist on disk and are **NOT** in the guarded set (0 hits in
`molmetal/tests/conftest.py`).

Revert-caught? **YES** — EMPIRICAL: emptying `PUBLISHED_FILE_PATHS` made 3 of the wave's own tests
fail (`test_the_curated_list_covers_the_registry_and_the_ledger`,
`test_a_curated_file_write_is_reported_when_the_writer_is_unauthorised`,
`test_the_ledger_is_guarded_against_every_writer`).

---

## G3 — stat quadruple + the `touched` diff kind

**Verdict: VERIFIED-FIXED.**

MEASURED probes:
1. The write-then-restore above was reported with `kind == "touched"` and identical sha256 — the
   exact case a sha-only diff cannot see. Session exit code 1.
2. `diff_published_trees` on a synthetic same-sha / moved-stat pair → `kind == "touched"`.
3. Reliability of the mtime signal on this host: 200 rapid in-place rewrites of identical bytes →
   **0/200** left `mtime_ns` unchanged (btrfs). `os.replace` additionally installs a new inode.

Residual: the `touched` signal is stat-based; a filesystem with coarse mtime granularity AND an
in-place rewrite would be invisible. Not observed on this host.

Revert-caught? **YES** — EMPIRICAL: reverting `_stat_entry` to the pre-wave sha-only form made
`test_an_identical_byte_rewrite_is_reported_as_touched` (AssertionError `[]`) and
`test_an_in_place_identical_rewrite_is_reported_as_touched` (KeyError) fail.

---

## G4 — AUTHORISATION vs CAPABILITY RECORD

**Verdict: VERIFIED-FIXED.**

MEASURED probes:
1. The shipped declaration file's 3 keys are all `molmetal/scripts/*.py` →
   `split_declarations` yields **0 authorisations / 3 capability records**; `load_capability_records()`
   returns 3 (previously they were silently inert).
2. ADVERSARIAL: a declaration keyed by a SCRIPT that declares a published path, with that script
   passed as the *active test module* → still **1 violation**. The capability record cannot
   authorise.
3. POSITIVE: a declaration keyed by `molmetal/tests/test_x.py` that ran → **0 violations**.

Revert-caught? **YES** — EMPIRICAL: replacing `is_test_module_key` with `lambda k: True` made
`test_a_capability_record_cannot_authorise_a_write` and
`test_the_declaration_file_splits_into_zero_authorisations` fail.

---

## G5 — new pre-flight gate `p3_run_ledger_provenance`

**Verdict: VERIFIED-FIXED.**

MEASURED probes:
1. REGISTERED: `build_gates("all")` → 127 gates, contains `p3_run_ledger_provenance`;
   `build_gates("launch")` → 129, contains it. `EXPECTED_GATE_COUNTS["all"] == 127` matches.
2. OBSERVABLE: `--gates p3_run_ledger_provenance,p3_run_ledger_staleness --json` →
   `p3_run_ledger_provenance PASS | ledger_rows=418 metrics=396 OPEN_never_committed=0
   superseded_never_committed=9`; `summary = {pass:2, fail:0, skip:0, warn:0, total:2}`.
3. BLOCKS (sandbox, tmp git repo + sandbox ledger): a row whose sha matches neither disk nor any
   commit → **FAIL** (`OPEN_never_committed=1`). On-disk sha → PASS. Committed sha → PASS.
   Superseded (earlier row uncommitted, later row committed) → PASS (counted, not blocking).
4. A FAIL gate makes the pre-flight exit **1** (`if n_fail > 0: return 1`, pre_flight_check.py:15616).

Revert-caught? **YES** — the wave's tests assert registration in every pipeline, the updated gate
counts, and the sandbox FAIL.

---

## G6 — `registry.py repoint <metric_id> [--apply]`

**Verdict: VERIFIED-FIXED** (one cosmetic residual).

MEASURED probes (all on a SANDBOX COPY of `registry/metrics.yaml`, never the real file):
1. Dry run (default): `changed=True, applied=False`, and the file was **byte-identical** afterwards.
2. `--apply`: the diff against the pristine registry is **2 lines in 1 row**
   (`source_json_sha256:` + `last_refreshed:`); all 578 metric ids preserved; the YAML re-parses;
   only `wf32_pb_valid` differs between the parsed docs.
3. Refuses an unknown metric id (`KeyError`).
4. REAL registry: `wf32_pb_valid` row sha = disk sha = `d6eec79e…` MATCH; `cfm_run_census` row sha =
   disk sha = `e85c529e…` MATCH. `registry.py check` → `REGISTRY: PASS — 578 current, 0 stale,
   0 missing`.

Residual (cosmetic): `registry/metrics.yaml:2` `generated_at: '2026-10-30T18:17:10+00:00'` was not
bumped by `repoint`, so the header stamp **predates** the two re-pointed rows' `last_refreshed`
(`2026-10-31T01:51:31Z` / `01:53:40Z`). `check` still PASSes.

Revert-caught? **YES** — the wave's tests cover dry-run, single-line change, no-op on a current
row, unknown-id refusal, missing-source refusal, and two-sha-line refusal.

---

## G7 — `metrics/by_metric/wf32_pb_valid.json` is self-describing

**Verdict: VERIFIED-FIXED but NOT DURABLE.**

MEASURED probes:
1. The artifact's `spec_target` is **byte-equal** to `molmetal/reports/wf32_m8_summary.json`'s
   `spec_target` (10 pockets × 5 candidates × 2 modes = 100 cells; method "se3sb (Wf-31
   backbone)"; 60–180 min). Verbatim, not re-derived.
2. The artifact records **1** pocket, **1** candidate, **2** evaluations, both `status="skipped"`
   ("posebusters not installed").
3. The old claim is gone: `grep -c "5/5 pockets PASS" metrics/by_metric/wf32_pb_valid.json` → **0**.
   The new `honest_framing` states the 100-cell sweep was **NEVER RUN** and forbids citing the
   artifact as a PoseBusters validity result.

NEW DEFECT (durability): `molmetal/scripts/bench_pb_valid.py` builds the artifact payload
(line ~835, `"recipe": "wf32_pb_valid"`) and contains **0** occurrences of `spec_target` and no
`wf432a1_pb_validity_deferred_block`. A future run of that script overwrites the artifact without
the G7 additions → the repair is silently lost and the registry row goes stale again.

Revert-caught? **YES** for the artifact as it stands (the wave's tests assert the verbatim
`spec_target`, the absence of a measured-validity claim, and the absence of a multi-pocket pass
claim) / **NO** for durability (no test regenerates through `bench_pb_valid.py`).

---

## G8 — the census distinguishes two SKIP causes

**Verdict: VERIFIED-FIXED** (the wave's numbers are MEASURED-correct; the DESIGN's are not).

INDEPENDENT RECOMPUTATION (my own probe, not the wave's code): for all 27 SKIP runs I read each
run's `per_epoch_metrics.json` (CSV fallback) and asked whether ANY epoch row carries
`gen_single_fragment_ratio`. Cross-tab against the artifact's own `basis`:
`{('field_absent','field_absent'): 11, ('denominator_empty','denominator_empty'): 16}`,
**0 mismatches**. The published artifact carries exactly 11 `field_absent` / 16
`denominator_empty` (27 FAIL, 5 PASS across the 59 runs) and its `changelog.wf_a125` records the
split and why it differs from the plan.

**THE DESIGN'S 9/18 IS REFUTED.** The plan's numbers came from an offline `events.jsonl` read
(which the plan's own fix text forbids); the census's normalised per-epoch rows give 11/16.

NEW DEFECT: `molmetal/scripts/wf_cfm_run_census.py:423` and `:427` — the docstring of the very
function that was repaired still says *"MEASURED 2026-10-31: 9 of the 27 SKIP runs"* and
*"18 of the 27 SKIP runs"*. Those are the refuted plan numbers and they contradict both the
artifact and the artifact's own changelog.

Revert-caught? **YES** — the wave's tests assert both bases are produced, that the published
census carries both, and that the changelog records the split.

---

## G9 — the GPU skip is no longer silent

**Verdict: PARTIALLY-FIXED** (works on this host; same shipping blocker as G1).

MEASURED probes:
1. `pytest -q molmetal/tests/test_rocm_egnn.py molmetal/tests/test_wf25_gpu_smoke.py` →
   `mol-metal GPU contract: 9 GPU-required test(s) did not run (CPU-only session; set
   MOLMETAL_TEST_ALLOW_GPU=1 to enable them)` and pytest's own `9 skipped` → **N matches**.
2. `pytest -q tests/test_cpu_tensor_rejection.py` → contract line **N=4**, pytest's own
   `4 skipped` → **N matches** (the top-level `tests/` tree is covered).
3. Exactly ONE contract line per session; the exit code is never touched.
4. `grep -rn pytest_terminal_summary` — before the wave: 0 hits repo-wide; now: the root
   `conftest.py:180`.

Residual: the whole mechanism lives in the gitignored root conftest → **does not ship** (see G1).

Revert-caught? **YES** for the defect (the wave's test greps for the contract line, checks it is
machine-readable, that N equals pytest's own skip count, and that the exit code is unchanged) /
**NO** for shipping.

---

## G10 — the guard has regression coverage

**Verdict: PARTIALLY-FIXED** (coverage exists and is strictly stronger; **HEAD still has none**).

MEASURED:
1. `molmetal/tests/test_published_tree_guard.py` exists (6 440 B, 8 tests) and is **UNTRACKED**:
   `git cat-file -e HEAD:molmetal/tests/test_published_tree_guard.py` →
   `fatal: … exists on disk, but not in 'HEAD'`. The plan's claim is confirmed.
2. The wave's replacement `molmetal/tests/test_wfa125_published_tree_guard.py` (17 tests) is also
   **UNTRACKED** (`??`). It carries the original three cases PLUS the G2/G3/G4 cases.
3. All wave files together: `pytest molmetal/tests/test_wfa125_*.py
   molmetal/tests/test_published_tree_guard.py` → **67 passed** (36.45 s, CPU-only).
4. `molmetal/tests/test_run_ledger.py` → **14 passed** — including
   `test_preflight_gate_enforced_by_default`, which was a KNOWN FAILURE at HEAD (F12). G11 fixed
   it honestly (the gate now returns PASS/FAIL, never WARN).

Residual: the wave does not commit, so a clean checkout of HEAD still has **zero** guard coverage.

Revert-caught? **YES** — every wave test imports the guard helpers from
`molmetal.tests.conftest`; removing or renaming one breaks the import (collection error).

---

## G11 — WARN is a first-class status; the staleness gate splits PASS/FAIL

**Verdict: VERIFIED-FIXED** for the mechanism — but the wave **left the repo RED**.

MEASURED probes:
1. Synthetic in-process gates (no file written): `summary = {pass:1, fail:1, skip:1, warn:1,
   total:4}` → `pass+fail+skip+warn == total` **TRUE**; `pass+fail+skip == total` **FALSE**
   (the pre-wave arithmetic is now correctly broken). Exit codes: WARN alone → **0** (advisory,
   does not block); FAIL → **1** (blocks); SKIP + `--strict` → **2**.
2. `gate_run_ledger_staleness` in a sandbox with one stale row → **FAIL**
   (`current=0 reformatted=0 stale=1 no_run=0 stale_metrics=m_stale`).
3. Evidence cap: with 300 `no_run` metrics the evidence is **224 chars** (was ~6 000), listing the
   stale ids first, then `... and N more (full list: run_ledger.py status)`.
4. The F11b reproduction now closes: `--gates p3_run_ledger_staleness,p3_protocol_alignment_yaml_
   target_in_baseline_range` → `summary {pass:2, fail:0, skip:0, warn:0, total:2}`, exit 0.

Residual / honest-negative: **no registered gate emits WARN in practice**, so `warn` is 0 in every
default run. The 11 WARN return sites live in `gate_p3_protocol_recommended_seed_count_3`,
`…_per_cell_breakdown`, `…_5_fold_cv_compatibility_50x`, `…_krasnov_paper_table_4_replication`
(**defined but NEVER registered** — `--gates` rejects those names) and in
`gate_p3_protocol_alignment_yaml_*` / `gate_p3_honest_neg_closure_m1_re_fold` (registered, but
PASS today). The counting machinery is real and correct; it is currently unexercised in
production. The wave's own test `test_the_summary_counts_warn_and_the_identity_holds` pins
`summary["warn"] == 0`, so it does not cover a WARN>0 case.

NEW DEFECT (repo left RED): `molmetal/tests/test_wf415a3_preflight_fail_audit.py:78` asserts
`preflight_summary["total"] == 117`; the wave added a gate → **MEASURED RED**:
`AssertionError: pre-flight total gate count drift: expected 117 … got 127` (`1 failed, 5 passed
in 156.12s`). The wave named this as a "REQUIRED OUT-OF-SET EDIT" in its handoff but did not
apply it. This is the only other test in the repo that pins the gate count (the other `117` hits
are unrelated Wf-335P2 GPU sweep counts).

NEW DEFECT (latent): `molmetal/tests/test_pre_flight_check.py:107-109` asserts
`s["pass"] + s["fail"] + s["skip"] == len(data["gates"])` — the pre-wave identity, now contradicted
by the wave's own `pass+fail+skip+warn == total`; it will fail the moment any gate WARNs. And
`:30` `ALLOWED_STATUS = {"PASS","FAIL","SKIP"}` lacks `"WARN"`, so a WARN gate would fail the
status check. Both are green today only because `warn == 0`.

Revert-caught? **YES** for the `warn` key + the identity (removing `n_warn` fails the wave's
`assert "warn" in summary`) / **NO** for a live WARN>0 case (the wave's test pins `warn == 0`).

---

## VERDICT TABLE

| gap | verdict | revert-caught? | one-line probe |
|---|---|---|---|
| G1 | PARTIALLY-FIXED | YES (defect) / NO (shipping) | root conftest blanks CVD for `tests/` even when the caller asks for GPU 0 — but `.gitignore:78` ignores it, `git cat-file -e HEAD:conftest.py` → not in HEAD |
| G2 | VERIFIED-FIXED | YES (empirical) | write-then-restore of `registry/metrics.yaml` in a live session → guard fires, exit 1; 4 curated paths present |
| G3 | VERIFIED-FIXED | YES (empirical) | identical-bytes rewrite reported `kind == "touched"`; sha-only revert fails 2 of the wave's tests |
| G4 | VERIFIED-FIXED | YES (empirical) | 0 authorisations / 3 capability records; a script key as active module still yields a violation |
| G5 | VERIFIED-FIXED | YES | gate registered in `all` (127) and `launch` (129); sandbox OPEN row → FAIL; FAIL → exit 1 |
| G6 | VERIFIED-FIXED | YES | sandbox dry-run leaves bytes identical; `--apply` = 2 lines in 1 of 578 rows; both real rows match disk |
| G7 | VERIFIED-FIXED (not durable) | YES (artifact) / NO (durability) | `spec_target` byte-equal to the M8 summary; "5/5 pockets PASS" gone; `bench_pb_valid.py` writes neither |
| G8 | VERIFIED-FIXED | YES | independent recomputation over all 27 SKIP runs: 11/16, 0 mismatches; the plan's 9/18 is refuted |
| G9 | PARTIALLY-FIXED | YES (defect) / NO (shipping) | contract line N=9 vs pytest's 9 skipped, N=4 in `tests/` — but lives in the gitignored conftest |
| G10 | PARTIALLY-FIXED | YES | coverage exists (17+8 tests, 67 green) but BOTH files are `??`; HEAD has zero guard coverage |
| G11 | VERIFIED-FIXED | YES (key/identity) / NO (live WARN) | synthetic `{pass:1,fail:1,skip:1,warn:1,total:4}`; stale → FAIL; evidence 224 chars; repo left with 1 RED test |

## NEW DEFECTS INTRODUCED BY THE WAVE

1. `molmetal/tests/test_wf415a3_preflight_fail_audit.py:78` — **RED**: `assert 127 == 117`
   (MEASURED `1 failed, 5 passed in 156.12s`). Named by the wave as a required out-of-set edit,
   never applied.
2. `molmetal/tests/test_pre_flight_check.py:107-109` — latent: identity `pass+fail+skip ==
   len(gates)` contradicts the wave's new `pass+fail+skip+warn == total`; `:30` `ALLOWED_STATUS`
   lacks `"WARN"`.
3. `molmetal/scripts/wf_cfm_run_census.py:423` and `:427` — docstring still carries the REFUTED
   9/18 split, contradicting the artifact's 11/16.
4. `molmetal/scripts/bench_pb_valid.py` (~line 835 payload) — writes the artifact with no
   `spec_target` and no deferral block → the G7 repair is silently dropped on regeneration.
5. `.gitignore:78` (`/conftest.py`) — not introduced by the wave, but the wave left it in place, so
   the ENTIRE G1+G9 mechanism cannot ship. (Pre-existing line; the wave's own handoff flagged it.)
6. `conftest.py:52`, `molmetal/tests/test_wfa125_preflight_warn_contract.py:36`,
   `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py:36` — 3 new ruff 0.6.9 F401 findings
   (ruff is a manual-stage hook, so not commit-blocking).
7. `registry/metrics.yaml:2` — `registry.py repoint` does not bump `generated_at`; the header
   stamp predates two of its own rows (cosmetic; `check` still PASSes).
8. Wave footprint: 8 modified + 6 untracked (`??`) + 1 gitignored. Nothing is committed.

## HEADLINE CLAIMS THAT SURVIVE

- **"G11 is now visible"** — SURVIVES. WARN is counted (`summary["warn"]`), the identity
  `pass+fail+skip+warn == total` closes, and the staleness gate now returns FAIL on a stale row
  (was WARN) → a launch is blocked (exit 1). Advisory WARN still does not block (exit 0) — that is
  the intended contract. Caveat: no registered gate emits WARN today, so the counter reads 0 in
  every default run.
- **"G5 — the launch gate now calls `uncommitted_provenance`"** — SURVIVES. Registered in `all` and
  `launch`, observable in the JSON report, FAILs on an OPEN never-committed row (sandbox-verified).
- **"G3 — a write-then-restore is now visible"** — SURVIVES. Verified end-to-end in a live session
  (`kind == "touched"`, exit 1).
- **"G6 — `registry.py repoint` exists and is surgical"** — SURVIVES. 2 lines in 1 of 578 rows.
- **"G7 — the artifact is self-describing"** — SURVIVES as an artifact; **DOES NOT SURVIVE as a
  durable repair** (`bench_pb_valid.py` regenerates it without the additions).
- **"G8 — two distinct SKIP causes, 11/16"** — SURVIVES, and the numbers are CORRECT. The DESIGN's
  9/18 is refuted by independent recomputation.
- **"G1/G9 — the CPU lock covers the `tests/` tree and the GPU skip is announced"** — SURVIVES ON
  THIS MACHINE ONLY. It does not survive as a shipped fix: `.gitignore:78` keeps the root conftest
  out of the repository, so a fresh clone still runs `tests/` unlocked and silently.
- **"G10 — the guard now has regression coverage"** — SURVIVES IN THE WORKING TREE ONLY; HEAD has
  none.

## HONEST-NEGATIVE / RESIDUAL LEDGER

1. G1 + G9 cannot ship until `.gitignore:78` is removed or negated (one line, outside both sets).
2. G7 is not durable until `bench_pb_valid.py` emits `spec_target` + the deferral block.
3. G11's `warn` counter is currently unreachable in production (all registered WARN-capable gates
   PASS today; four WARN-emitting gate functions are defined but never registered).
4. G2 does not guard `registry/narrative_claims.yaml` / `registry/file_provenance.json`.
5. The wave left one RED test (G11) and one latent contradiction (G11) in the repo.
