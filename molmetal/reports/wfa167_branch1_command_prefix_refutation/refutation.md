# Wf-A167 branch 1 — adversarial refutation

**VERDICT: NOT REFUTED** (tag: MEASURED; 0 GPU, CPU-only review)

Branch under review: 1 — the live command-prefix script repair.
Reviewer re-derived every census from `git show HEAD:` and the live tree, never from the
branch's own list.

| # | Check | Result |
|---|---|---|
| 1 | Census re-derived independently | PASS |
| 2 | Three-way semantics (unset / "" / "1") at all 20 sites | PASS 20/20 |
| 3 | Regression test passes + bites under external mutation | PASS |
| 4 | Nothing over-repaired (pure lock-preserving rewrite) | PASS |
| 5 | Archived-script policy | PASS (one supporting sentence FALSE) |
| 6 | Patch applies to a scratch copy; tree not broken | PASS |
| 7 | No GPU allocated | PASS |

## Evidence highlights

- **Census (re-derived).** Own classifier over the 113 tracked `*.sh`: LIVE prefix defeaters
  post-repair = **0**; ARCHIVED = **11** (matches the branch). From `git show HEAD:` the same
  19 files held **20** (wf308c x2, rest x1) — the branch's list exactly. 0 untracked and 52
  ignored `*.sh` also scanned: 0 defeaters, so the `git ls-files` scope has no blind spot.
  Wf-A165's 32 sites: **32/32** reproduce byte-for-byte in the branch's pre-repair
  `census.json`; live-set diff (file,line) = 20 == 20, both directions empty.
- **Three-way (my own bash harness, pre-repair value from HEAD).** 20/20: unset -> identical to
  the pre-repair default; ambient `""` -> EMPTY; ambient `"1"` -> `1`. `${VAR-default}` (not
  `:-`) is the right form and is `set -u`-safe.
- **Test bites.** `test_wfa167_command_prefix_scripts.py` 11 passed under
  `CUDA_VISIBLE_DEVICES=""`. External mutation against HEAD content with the branch's own
  scanner: 20 detected pre-repair -> 0 post-repair. Sibling set (8 files): 106 passed,
  2 skipped, 0 failed.
- **No over-repair.** Full diff read hunk by hunk: every hunk is
  `<token>` -> `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-<D>}"` + one `# Wf-A167:` WHY
  comment. Modes HEAD 17x100755+2x100644 -> now 17x755+2x644. `bash -n` 19/19.
  `proof.json` `source_now` matches disk byte-for-byte 20/20.
- **Archived policy.** No live file's command word is followed by a `TODO/archive/` path
  (independent grep) -> the archived copies are never executed; leaving them alone is correct.
  Their old live paths are missing from disk AND `git ls-files` -> dead refs. `TODO/archive/`
  untouched (`git status --porcelain TODO/archive/` empty).
- **Patch.** `git archive HEAD` -> scratch `git init` -> `git apply --check` on the branch-1-only
  diff: all 19 patches check clean.
- **GPU.** Watchdog log still exactly 24 lines, all `pid=700241 ... 550 MiB`, mtime Oct 31
  17:52 — no new allocation.

## Surviving concerns (do not refute the repair)

1. **S1 (medium)** — 11 archived prefix defeaters remain under `TODO/archive/wf437_simplify/`
   (listed in `refutation.json`). Correctly out of scope, but any restore must re-apply the
   lock-preserving form.
2. **S2 (medium)** — 8 dead refs to archived launchers' old live paths: `scripts/gpu_pipeline_master.sh:209,224,231,237,246,261`,
   `scripts/gpu_smoke_when_vram_free.sh:211`, `scripts/run_wf264_cfm_property_panel.sh:71`.
   The branch listed 4 of the 6 in `gpu_pipeline_master.sh` and neither of the other two files.
   Pre-existing, not introduced here; but 6 GPU-master jobs would fail if run today.
3. **S3 (low)** — `scripts/launch_wf308c_todo220_gpu.sh:125` argv-position assignment still
   present (broken invocation, not a lock defeater). Branch's out-of-scope finding (1) confirmed.
4. **S4 (low)** — `nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}"` still at
   `scripts/launch_wf299_cfm_main.sh:560` and `scripts/launch_wf310f_frontier_math_gpu.sh:276`
   (post-repair line numbers; the branch cited the pre-repair 559/275). Finding (2) confirmed.
5. **S5 (low)** — "167 occurrences" is a classified-record count (167 records / 129 unique
   lines), not a literal token count (210 now, ~170 pre-repair). Coverage is complete.
6. **S6 (low)** — "the only live code naming an archived .sh is
   `test_wfa131_repo_path_reference_guard.py:155`" is FALSE; 6+ other live test files name
   archived launchers (all skip-guarded / non-executing), so the policy conclusion stands but
   the sentence should be corrected.
