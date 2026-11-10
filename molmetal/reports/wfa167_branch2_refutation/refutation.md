# Wf-A167 branch 2 — adversarial refutation

VERDICT: **NOT REFUTED**

Branch under review: the independent command-prefix census
(`molmetal/scripts/wfa167_command_prefix_guard.py`) + the guard-extension SPEC
(`molmetal/reports/wfa167_command_prefix_guard/patch_guard_extension.diff`).

0 GPU (every call prefixed `CUDA_VISIBLE_DEVICES=""`), single venv
(`/home/hugo/codes/mol-metal/.venv/bin/python`), read-only: no repo file was
modified by this review.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | Census re-derived independently | PASS |
| 2 | Three-way semantics (unset / `""` / `"1"`) | PASS |
| 3 | Regression test really bites | PASS |
| 4 | Nothing over-repaired | PASS |
| 5 | Archived-script policy | PASS |
| 6 | Patch would not break the tree | PASS |
| 7 | No GPU used | PASS |

### 1 — census re-derived (PASS)
- `git ls-files -z -- '*.sh'` = **113** (unique). Independent token count of
  `CUDA_VISIBLE_DEVICES` over those files = **210** = `census.counts.sites_total`.
  No token missed.
- Independent line-start regex found 50 assignment sites, all a **subset** of the
  branch's 61; the 11 extra are exactly the `if`/`if !`-prefixed (10) plus the
  single `setsid nohup` arg-position site.
- Re-running the guard reproduces `census.json` byte-for-byte (counts, denominator,
  self_check, all 210 sites). All 113 `snapshot.per_file_sha256` match disk;
  `snapshot.git_head` == HEAD (`1b9470a0`).
- `git ls-files --others --exclude-standard -- '*.sh'` = 0 → tracked scope is the
  whole live `.sh` universe.

### 2 — three-way semantics (PASS)
All 32 `COMMAND_PREFIX` sites extracted from disk and **executed** under bash with
ambient unset / `""` / `"1"`:
- 21 `ALREADY_LOCK_PRESERVING`: unset → the script's own default, `""` → `''`
  (empty, lock preserved), `"1"` → `'1'`. **0 mismatches.**
- 11 `REAL_DEFEATER`: the value replaces the ambient `""` → genuine defeats, all
  inside `TODO/archive/wf437_simplify/`.
- All 60 env-assignment sites executed: 0 whose behaviour contradicts the census.
- The one `ARG_POSITION` site reproduces the claim exactly:
  `nohup: failed to run command 'CUDA_VISIBLE_DEVICES=0'`.

### 3 — the test bites (PASS)
`pytest molmetal/tests/test_wfa167_command_prefix_guard.py` → **19 passed, 0 failed**.
Mutation test on a scratch copy — 5 mutants, each caught:
no-colon self-ref treated as unmask (5 failures) · arg-position mislabelled
`COMMAND_PREFIX` (3) · reserved-word handling removed (3) · `export` builtin
handling removed (2) · census count hardcoded (1). Baseline restored to 19 passed.

### 4 — nothing over-repaired (PASS)
- The 19 live prefix-repaired files' diff is a pure rewrite + one `Wf-A167:` comment.
  Executed unset-ambient value equals the pre-repair default at all 20 live sites.
- Disjointness re-derived: 42 modified `scripts/` files = 22 export-repaired +
  19 prefix-repaired + `scripts/INDEX.md`; overlap **0 files**.
- Diff vs branch A's list (`molmetal/reports/wfa165_shell_launchers/proof.json`,
  sha256 `0b73f35c…`): file level identical (27 files / 32 sites); line level
  20 shifted (+1; +2 for the 2nd `launch_wf308c` site), 12 unchanged.

### 5 — archived-script policy (PASS)
The 7 archived launchers carrying the 11 defeat sites have **no live twin** and are
referenced only in prose/doc/test text. `results/cfm_full_retrain/launch_loop.sh`
is referenced by no live script.

### 6 — the patch would not break the tree (PASS)
`git apply --check` rc=0 on a scratch copy, `git apply` rc=0, patched module imports.
Patched vs unpatched guard run on the real repo root: `findings`,
`findings_by_mechanism`, `findings_total`, `findings_defeat_ambient_empty_mask`,
`mechanisms`, `per_file`, `self_check`, `denominator`, `unresolved_env_sites*`,
`git_head` all **byte-identical**; only `generated` (timestamp) differs. New section
matches the crosscheck (61 sites / 32 / 28 / 1 / 11 defeats / 113-113-0).
**NOT applied to the working tree**: `sha256(wfa165_gpu_lock_guard.py)` =
`2112191c…` = the claimed sha (the file is untracked, so the sha is the only proof).
Two-implementation agreement re-derived: A167 set == patched-A165 set (61/61, both
directions empty; command-prefix 32/32).

### 7 — no GPU (PASS)
Watchdog log `…/tasks/bpzsdj6aj.output`: still **24 lines**, all
`pid=700241, 550 MiB`, **0 new lines**, mtime `2026-10-31 17:52`.
`nvidia-smi --query-compute-apps` → empty; GPU0/GPU1 `memory.used` = 2 MiB each.

## Surviving concerns

- **C1 (low)** — Report prose says "Live command-prefix class = 21" / "21 live
  sites"; disk truth is **20 live sites in 19 live files** (the 21st,
  `results/cfm_full_retrain/launch_loop.sh:10`, is `ARCHIVED`-classed). Branch A's
  own `proof.json` says `in_scripts_live_launchers=20`. `census.json` is exact;
  only the prose is off by one.
- **C2 (low)** — `crosscheck_a165.json` →
  `step2_a165_structural_blind_spot.arg_position_sites` holds **50** coordinates of
  which exactly **one** is an `ARG_POSITION` site; the 50 are exactly
  (61 assignment sites − 11 archived defeaters), and **43** of them have the
  assignment at the physical line start (A165 can see them). The adjacent
  `sites_in_this_repo_a165_cannot_see: "none beyond the arg-position site below"`
  is contradicted by the branch's own list.
- **C3 (medium)** — Scope is tracked `*.sh`. 449 `.sh` files live in gitignored
  agent worktrees (`.claude/worktrees/*/`), **72** of them non-archived launchers
  still carrying the pre-repair hard override. Nothing in the main tree references
  `.claude/worktrees` (0 grep hits), so not a live defect of the main checkout.
- **C4 (low)** — "Two independent implementations" is true at the code level
  (no shared code; 61/61 and 32/32 identical) but both were authored by branch 2 —
  a second opinion, not a second author.
- **C5 (medium)** — `scripts/launch_wf308c_todo220_gpu.sh:125` — the arg-position
  assignment makes the GPU probe (lines 122-129) fail with
  `nohup: failed to run command 'CUDA_VISIBLE_DEVICES=0'` (reproduced), so the live
  launcher always exits 5. Correctly **not** counted as a lock defeat, but it is a
  real live launcher bug left as a note only. Pre-existing (the repair touched
  lines 179/206 only).
- **C6 (informational, not charged to branch 2)** —
  `molmetal/scripts/pre_flight_check.py` (a forbidden-write target for every
  branch) carries a 2429-line working-tree diff, mtime `2026-11-01 11:54:50` —
  inside this wave but outside branch 2's artifact window (11:56–11:58) and outside
  its declared file list; several branches ran concurrently, so it cannot be
  attributed from mtime alone. Flagged for the orchestrator.

## Reproduction

```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa167_command_prefix_guard.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa167_command_prefix_guard.py --out /tmp/a167.json --quiet
sha256sum molmetal/scripts/wfa165_gpu_lock_guard.py
wc -l /tmp/claude-1001/-home-hugo-codes-mol-metal/0f647790-b0f6-4b29-8009-1cef00206cd3/tasks/bpzsdj6aj.output
```
