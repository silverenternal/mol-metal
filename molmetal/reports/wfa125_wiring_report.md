# Wf-A125 — Wiring verification report

Verifier: Wf-A125 wiring agent (read-only; the ONLY file written is this one).
Date: 2026-10-31. Machine: CPU-only session, `CUDA_VISIBLE_DEVICES=""` on every
python/pytest invocation. HEAD = `1b9470a0`. Working tree carries the wave's
edits UNCOMMITTED (see W9).

Every number below is MEASURED on this machine in this session. Command lines
are reproduced verbatim. Exit codes are the process exit code of the command
shown.

---

## W1 — The new test surface

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wfa125_*.py -q -p no:cacheprovider
```

```
...........................................................              [100%]
mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')
mol-metal GPU contract: 0 GPU-required test(s) did not run (CPU-only session; set MOLMETAL_TEST_ALLOW_GPU=1 to enable them)
59 passed, 1 warning in 27.46s
```

EXIT = **0**. MEASURED twice (30.05 s then 27.46 s), 59 passed / 0 failed /
0 skipped both runs. 5 files matched the glob:
`test_wfa125_gpu_contract.py`, `test_wfa125_preflight_warn_contract.py`,
`test_wfa125_published_tree_guard.py`, `test_wfa125_registry_sha_refresh.py`,
`test_wfa125_root_conftest_gpu_lock.py`.

Cross-check: SET A reported 14 passed over its 2 files, SET B reported 45 over
its 3 files; 14 + 45 = 59 = the measured total. The two implementers' reports
and the wave's actual surface agree.

---

## W2 — The regression surface the wave touched

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_promote_metrics.py molmetal/tests/test_run_ledger.py molmetal/tests/test_wf32_phase3_pb_valid.py molmetal/tests/test_registry.py -q -p no:cacheprovider
```

```
......s...........................................                       [100%]
49 passed, 1 skipped, 1 warning in 58.41s
```

EXIT = **0**.

### State of `test_run_ledger.py::test_preflight_gate_enforced_by_default`

MEASURED **FIXED** (it was known-FAILING at HEAD per the wave brief, F12):

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_run_ledger.py::test_preflight_gate_enforced_by_default -q -p no:cacheprovider
1 passed, 1 warning in 2.01s
```

EXIT = 0. The contradiction was resolved honestly, not by weakening the test:
the assertion is `r.status in ("PASS","FAIL")`, and G11 changed
`gate_run_ledger_staleness` from returning WARN to returning **PASS on a clean
ledger / FAIL on a stale row**. The gate's own docstring claim ("enforces by
DEFAULT") is now true. W5 below shows the same gate returning PASS with a
bounded evidence string.

---

## W3 — The published tree is byte-identical before and after W1+W2

```
find metrics/by_metric metrics/paper_tables molmetal/metrics/by_metric -type f | sort | xargs sha256sum > BEFORE.txt
# ... W1 + W2 ...
find metrics/by_metric metrics/paper_tables molmetal/metrics/by_metric -type f | sort | xargs sha256sum > AFTER.txt
diff BEFORE.txt AFTER.txt
```

- file COUNT: **612** in both snapshots (identical count).
- diff: **EMPTY** (`diff` exit 0, 0 lines of output).
- `git status --porcelain -- metrics/` before vs after W1+W2: **identical**
  (34 lines, `diff` empty). No published metric was overwritten by the runs.

The snapshot was taken AFTER the wave's own artifact edits (G7/G8), so this
proves the *test runs* mutate nothing; it does not and cannot assert the
wave's own edits to `wf32_pb_valid.json` / `cfm_run_census.json`.

---

## W4 — `registry.py check`

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/registry.py check
```

```
[PASS] datasets: 35/35 datasets verified | ACTIVE 19, DO-NOT-TRAIN 1, REFERENCE 10, SUPERSEDED 3, UNUSED 2 | ...
[PASS] protocols: 8/8 protocol locks verified | 8/8 protocol ids resolve to a metric row
[PASS] metrics: 578 current, 0 stale, 0 missing | 11 paper-line (8 quotable, 3 headline-grade) | ...
REGISTRY: PASS
```

EXIT = **0**. Real numbers: **578 current / 0 stale / 0 missing**; 35/35
datasets; 8/8 protocols. Matches SET B's report exactly.

G6 objective independently verified: the two re-pointed rows' shas equal the
on-disk artifacts.
- `registry/metrics.yaml:15811` `wf32_pb_valid.source_json_sha256 =
  d6eec79ee513e8427939b79096fd462b62b70d2aa648a04cf0e4338f951b6127`
  == `sha256sum metrics/by_metric/wf32_pb_valid.json`
- `registry/metrics.yaml:3167` `cfm_run_census.source_json_sha256 =
  e85c529e49e034f121e2a1a16878878239cb986139a23b0836bdc463292e4cf9`
  == `sha256sum metrics/by_metric/cfm_run_census.json`

---

## W5 — The new pre-flight gate (G5) actually fires

CLI surface:

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m molmetal.scripts.run_ledger provenance --all
```
```
ledger provenance: rows=418 metrics=396 distinct_source_paths=398 OPEN_never_committed=0 superseded_never_committed=9
```
EXIT = 0. (Baseline at HEAD was rows=417; the wave appended 1 row.)

The pre-flight's OWN view — the gate is not merely registered, its status is
observable:

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/pre_flight_check.py \
  --gates p3_run_ledger_provenance,p3_run_ledger_staleness --json /tmp/pf.json
```

```
summary: {'pass': 2, 'fail': 0, 'skip': 0, 'warn': 0, 'total': 2, 'total_wall_sec': 1.5813}

p3_run_ledger_provenance   -> PASS
  evidence: p3_run_ledger_provenance: ledger_rows=418 metrics=396 OPEN_never_committed=0 superseded_never_committed=9

p3_run_ledger_staleness    -> PASS
  evidence: p3_run_ledger_staleness: current=385 reformatted=11 stale=0 no_run=182
            non_current_ids=m7_diversity,m21_nci60_9cell,... and 168 more (full list: run_ledger.py status)
```

EXIT = 0. **Gate name = `p3_run_ledger_provenance`, status = PASS, evidence as
quoted above.**

Two secondary facts MEASURED:
- `p3_run_ledger_staleness` returns **PASS** today, not WARN — G11's change.
- Its evidence is now **bounded** (the "... and 168 more (full list:
  run_ledger.py status)" tail replaces the old 6 kB single-line enumeration of
  all 182 `no_run` metrics).

Non-vacuity (the PASS is not vacuous): the wave's own
`molmetal/tests/test_wfa125_preflight_warn_contract.py` contains
`test_the_provenance_gate_fails_on_a_never_committed_row` (asserts FAIL +
`OPEN_never_committed=1`), `test_the_provenance_gate_fails_on_a_broken_ledger`,
`test_the_staleness_gate_fails_on_a_stale_row`, and
`test_the_provenance_gate_is_registered_in_every_pipeline`. All passed in W1.

---

## W6 — The WARN roll-up (G11) is visible and the arithmetic closes

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all --json /tmp/pf_all.json
```

```
summary: {'pass': 113, 'fail': 4, 'skip': 10, 'warn': 0, 'total': 127, 'total_wall_sec': 84.0047}
EXIT = 1
```

- `summary.pass`  = 113
- `summary.fail`  = 4
- `summary.skip`  = 10
- `summary.warn`  = **0** (the field EXISTS — it is in the summary dict)
- `summary.total` = 127
- `pass+fail+skip` = **127 == total** → the arithmetic closes in the
  no-WARN case.
- The 4 FAILs are `p1_cfm_smoke_decoded`, `p1_cfm_catherine_30metals`,
  `p1_gpu_env_probe`, `p3_wf385_scaffold_loss_wire_in_verified` — none of them
  is a gate this wave added or edited; `p1_gpu_env_probe` FAILs as a direct
  consequence of the CPU-only lock.

### The WARN branch itself, exercised

`warn` is 0 in the `all` pipeline, so the above does not by itself prove the
WARN branch counts. I injected exactly one synthetic WARN gate into the
pre-flight's own runner list (in-process only; no source file modified, no
file on disk written) and re-ran the full `all` pipeline:

```
SUMMARY {"pass": 112, "fail": 4, "skip": 10, "warn": 1, "total": 127, "total_wall_sec": 88.3774}
IDENTITY pass+fail+skip+warn==total -> True
IDENTITY pass+fail+skip==total     -> False
WARN gates: ['p3_honest_neg_closure_m2_r2']
MAIN_RC 1
```

MEASURED conclusion: with a WARN present, `pass+fail+skip == 126 != 127` (the
old inconsistency, now surfaced rather than hidden) and
`pass+fail+skip+warn == 127 == total` (the new identity). The exit code is
**unchanged** by a WARN (still 1, from the 4 pre-existing FAILs) — the WARN is
advisory, exactly as the roll-up's own stderr line states.

---

## W7 / W8 — GPU and paper

```
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -i mol-metal
```
→ **no output** (grep exit 1). The full `--query-compute-apps` listing is
empty — no compute process at all. PASS. No watchdog was started.

```
git status --porcelain -- paper/
```
→ **0 lines**. PASS (AGENTS.md §5.3 / TODO-84).

---

## W9 — Changed paths

```
git status --porcelain | wc -l
```
→ **234** changed paths in the whole working tree. The pre-wave tree already
carried a large uncommitted backlog (Wf-A110..A126 artifacts, `results/`,
`molmetal/reports/`, `docs/`, `TODO/`), so the raw 234 is NOT the wave's
footprint.

### The wave's own footprint — reconstructed from file mtimes

Every wave-touched path carries an mtime inside the 2026-10-31 09:49–09:56
window; the surrounding backlog does not. MEASURED per-path:

| # | path | mtime (2026-10-31) | git status |
|---|---|---|---|
| 1 | `conftest.py` (repo root, NEW) | 09:50:45 | **NO GIT STATUS LINE — GITIGNORED** |
| 2 | `molmetal/tests/conftest.py` | 09:49:20 | ` M` |
| 3 | `molmetal/scripts/pre_flight_check.py` | 09:56:43 | ` M` |
| 4 | `molmetal/scripts/run_ledger.py` | 09:54:05 | ` M` |
| 5 | `molmetal/scripts/registry.py` | 09:50:31 | ` M` |
| 6 | `molmetal/scripts/wf_cfm_run_census.py` | 09:53:24 | ` M` |
| 7 | `metrics/by_metric/wf32_pb_valid.json` | 09:51:18 | ` M` |
| 8 | `metrics/by_metric/cfm_run_census.json` | 09:53:28 | ` M` |
| 9 | `molmetal/reports/run_ledger.jsonl` | 09:54:13 | ` M` |
| 10 | `molmetal/tests/test_wfa125_root_conftest_gpu_lock.py` (NEW) | — | `??` |
| 11 | `molmetal/tests/test_wfa125_gpu_contract.py` (NEW) | — | `??` |
| 12 | `molmetal/tests/test_wfa125_published_tree_guard.py` (NEW) | — | `??` |
| 13 | `molmetal/tests/test_wfa125_preflight_warn_contract.py` (NEW) | — | `??` |
| 14 | `molmetal/tests/test_wfa125_registry_sha_refresh.py` (NEW) | — | `??` |

**14 wave-touched paths; 13 visible to git, 1 (the root `conftest.py`) invisible
because it is gitignored.** 13 of the 14 are UNCOMMITTED — the wave ships
nothing to a fresh clone until it is committed.

Adjacent file `registry/metrics.yaml` is ` M` (dirty vs HEAD, 4012 changed
lines) but is **PRE-EXISTING**: its `generated_at` is
`2026-10-30T18:17:10+00:00` (= 2026-10-31 02:17 +0800), i.e. a wholesale
`registry.py generate` from an earlier wave that predates this wave's 05:23
start. The wave's contribution to it is the 2 sha re-point lines verified in
W4. W1 and W2 did not change it: sha256 of `registry/{metrics,datasets,protocols}.yaml`
measured before and after W1 → identical.

Note: because `registry.py repoint` is a surgical single-line edit, it does
**not** bump `generated_at`. The registry's `generated_at` therefore predates
the content of two of its rows. `registry.py check` still reports PASS
(578 current / 0 stale / 0 missing), so this is a provenance-stamp wrinkle,
not a staleness defect.

---

## W10 — Ruff on the changed .py files (pinned pre-commit env)

```
RUFF=$(ls -d /home/hugo/.cache/pre-commit/*/py_env*/bin/ruff | head -1)
# /home/hugo/.cache/pre-commit/repo3n9semkz/py_env-python3.12/bin/ruff
"$RUFF" --version
ruff 0.6.9
```

```
"$RUFF" check conftest.py molmetal/tests/conftest.py \
  molmetal/tests/test_wfa125_root_conftest_gpu_lock.py \
  molmetal/tests/test_wfa125_gpu_contract.py \
  molmetal/tests/test_wfa125_published_tree_guard.py \
  molmetal/tests/test_wfa125_preflight_warn_contract.py \
  molmetal/tests/test_wfa125_registry_sha_refresh.py \
  molmetal/scripts/pre_flight_check.py molmetal/scripts/run_ledger.py \
  molmetal/scripts/registry.py molmetal/scripts/wf_cfm_run_census.py
```

EXIT = **1**, **11 findings**:

```
conftest.py:52:20: F401 [*] `typing.Dict` imported but unused
molmetal/scripts/pre_flight_check.py:3449:71: F841 Local variable `exc` ...
molmetal/scripts/pre_flight_check.py:9363:5: F841 Local variable `elapsed` ...
molmetal/scripts/pre_flight_check.py:10123:5: F841 Local variable `PER_METAL_RANGE` ...
molmetal/scripts/pre_flight_check.py:10125:5: F841 Local variable `DRUGOOD_RANGE` ...
molmetal/scripts/pre_flight_check.py:11172:5: F841 Local variable `failures` ...
molmetal/scripts/pre_flight_check.py:11702:34: E741 Ambiguous variable name: `l`
molmetal/scripts/registry.py:1694:5: F841 Local variable `value_missing` ...
molmetal/scripts/wf_cfm_run_census.py:312:5: F401 [*] `cfm_cohort_recovery.COHORT_SOURCE_CONFIG` imported but unused
molmetal/tests/test_wfa125_preflight_warn_contract.py:36:8: F401 [*] `os` imported but unused
molmetal/tests/test_wfa125_root_conftest_gpu_lock.py:36:8: F401 [*] `pytest` imported but unused
```

Attribution, MEASURED by re-running the same pinned ruff on the HEAD versions
of the wave-modified files (`git show HEAD:<path>` + `ruff check --config
pyproject.toml`):

- **7 findings are PRE-EXISTING at HEAD**: the 6 in `pre_flight_check.py`
  (identical rules, at HEAD line numbers 3449/9256/10016/10018/11065/11595)
  and the 1 in `registry.py` (`value_missing`, same line 1694).
- **3 findings are in the wave's NEW files** and are therefore introduced by
  the wave: `conftest.py:52` (`typing.Dict`), `test_wfa125_preflight_warn_contract.py:36`
  (`os`), `test_wfa125_root_conftest_gpu_lock.py:36` (`pytest`).
- **1 finding** (`wf_cfm_run_census.py:312`, `COHORT_SOURCE_CONFIG`) is NOT
  attributable to this wave: it is a dead import from an earlier wave's
  untracked module (`molmetal/scripts/cfm_cohort_recovery.py`, mtime
  04:23:45, defines the name; the sibling import `recover_cohort` IS used at
  line 2225). Wf-A125's census edit is at the SKIP-reason site
  (line ~423), not the import block.

Context that bounds how much this matters: `ruff` is a **manual-stage**
pre-commit hook (`stages: [manual]` in `.pre-commit-config.yaml`), so it does
not block a commit; and `pyproject.toml`'s `[tool.ruff.lint]` deliberately
leaves F401/F841/E741 VISIBLE as real debt (the repo's own comment records
`ruff check molmetal` -> 1,423 findings at the pin). The 3 new-file F401s
still add debt the repo's stated policy wants surfaced, not hidden.

---

## FAILURES / OPEN ITEMS

1. **CRITICAL — G1's repo-root CPU lock CANNOT SHIP.** The root `conftest.py`
   is gitignored: `git check-ignore -v conftest.py` →
   `.gitignore:78:/conftest.py	conftest.py`; `git ls-files --error-unmatch
   conftest.py` → error (not tracked); `git cat-file -e HEAD:conftest.py` →
   `fatal: path 'conftest.py' exists on disk, but not in 'HEAD'`. It
   therefore does not appear in `git status` at all (row 1 of W9 is blank).
   MEASURED proof that it is the ONLY CPU lock for the top-level `tests/` tree:
   ```
   pytest --collect-only -q tests/test_fused_ops.py                    -> "mol-metal: CPU lock ON (CUDA_VISIBLE_DEVICES='')"
   pytest --collect-only -q --confcutdir=tests tests/test_fused_ops.py -> (no CPU lock line at all)
   ```
   So on a fresh clone the 5 `tests/` files listed in F2
   (`test_fused_ops.py`, `test_new_kernels.py`,
   `test_triton_fused_residual_wired.py`, `test_cpu_tensor_rejection.py`,
   `test_featurizer_cache_versioning.py`) run with **no CPU lock**, and
   `MOLMETAL_TEST_ALLOW_GPU` / the `gpu` marker machinery does not reach
   them either. The fix is one line in `.gitignore` (drop `/conftest.py`, or
   negate it with `!/conftest.py`) — `.gitignore` is outside the wave's sets,
   so it needs an owner.

2. **RED, still red — the required out-of-set edit was NOT made.**
   `molmetal/tests/test_wf415a3_preflight_fail_audit.py::test_preflight_total_count`
   asserts `preflight_summary["total"] == 117`; the measured total is 127:
   ```
   E       assert 127 == 117
   1 failed, 2 passed, 1 warning in 44.31s
   ```
   (EXIT = 1.) This is the pin the wave had to update (117 -> 127) because it
   added a 127th gate. It is a one-token edit that was flagged as
   out-of-set by SET B and never performed.

3. **LATENT, green only by luck — `molmetal/tests/test_pre_flight_check.py`
   does not know about WARN.** MEASURED line contents:
   - `:28` `REQUIRED_SUMMARY_KEYS = {"pass", "fail", "skip", "total_wall_sec"}`
     — no `"warn"`.
   - `:30` `ALLOWED_STATUS = {"PASS", "FAIL", "SKIP"}` — no `"WARN"`, so
     `:101` `assert gate["status"] in ALLOWED_STATUS` will FAIL the moment
     any gate emits WARN.
   - `:107` `assert s["pass"] + s["fail"] + s["skip"] == len(data["gates"])`
     — no `s["warn"]`, so this identity will FAIL the moment a WARN gate
     runs (that is precisely the arithmetic the wave repaired).
   The file is **green today** (`3 passed in 92.57s`, EXIT 0) only because
   `warn == 0` in the `all` pipeline. W6's injected-WARN probe shows the
   report shape that will break it: `warn: 1`.

4. **3 new ruff F401 findings in the wave's own new files** (see W10). Not
   commit-blocking (ruff is a manual-stage hook) but new debt under the
   repo's stated "leave it visible" policy.

5. **G7's repair is not durable against a regeneration.**
   `molmetal/scripts/bench_pb_valid.py:144-145` writes
   `metrics/by_metric/wf32_pb_valid.json` (the published artifact) and does
   NOT emit `spec_target` or the `wf432a1_pb_validity_deferred_block`. The
   artifact on disk currently HAS both (MEASURED: top keys include
   `spec_target` and `wf432a1_pb_validity_deferred_block`), but a future run
   of `bench_pb_valid.py` would drop them and re-stale the registry row.

6. **Everything the wave produced is UNCOMMITTED.** 13 of the 14 wave-touched
   paths are working-tree-only (8 ` M`, 5 `??`); the 14th (`conftest.py`) is
   gitignored. The 5 new test files are `??` — they enter the repo only at
   the next commit. Until then the wave's regression coverage exists nowhere
   but this working tree.

7. **Unresolved SET A handoff (out of scope, real).**
   `triton_kernels/fused_residual_add.py:286` does
   `torch.add(x, residual, alpha=alpha)`, which scales the SECOND operand.
   SET A measured `torch.add(ones, 2*ones, alpha=0.5) == 2.0` where
   `0.5*x + 1.0*r == 2.5`. The G1 CPU lock EXPOSES this (the test now runs on
   CPU and fails). I did not re-run that specific parity test — reported as
   SET A's measurement, not re-verified by me.
   Also pre-existing and NOT introduced by the wave:
   `pytest --collect-only -q tests molmetal` exits 3 (INTERNAL ERROR) because
   `molmetal/scripts/wf229_smoke_test.py` calls `sys.exit(0)` at import.

8. **The registry's `generated_at` no longer describes its content** (2 rows
   re-pointed after the stamp was written; see W9 note). `registry.py check`
   is PASS, so this is cosmetic provenance, not staleness.

---

## SUMMARY

Wired and MEASURED: the 5-file new test surface (59 passed), the 4-file
regression surface (49 passed / 1 skipped), the 612-file published-tree
immutability across W1+W2, `registry.py check` (578 current / 0 stale /
0 missing), the new `p3_run_ledger_provenance` gate (PASS, observable, and
proven non-vacuous by its own tests), the WARN roll-up (identity closes at
127 == 127 with and without a WARN present), the GPU-empty and paper-clean
invariants, and the G6 sha re-point (both shas match disk).

Not wired: G1's repo-root CPU lock is gitignored and cannot ship (failure 1),
and the 117->127 pin the wave required was never updated (failure 2, still
RED). Two further items are latent-but-green-today (failure 3) and
non-durable (failure 5).
