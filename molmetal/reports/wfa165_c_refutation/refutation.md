# Wf-A165 — refutation of branch C (repo-wide GPU lock-defeater guard)

Refuter: adversarial branch. 0 GPU (`CUDA_VISIBLE_DEVICES=""` on every call), single venv
(`/home/hugo/codes/mol-metal/.venv/bin/python`), no `os.walk`, every number traced to a path on disk.
Verdict: **REFUTED** (the census NUMBERS are not refuted; the claim "All deliverables are in place
and verified" is).

| # | Check | Result |
|---|---|---|
| C1a | denominator reproduces (2987 = 2874 .py + 113 .sh; 9550 tracked) | PASS |
| C1b | per-mechanism counts reproduce (M1 35 / M2 0 / M3 27 / M4 27; 89 findings; 54 defeats) | PASS |
| C1c | report claim #10: Wf-A163 numbers as quoted | **FAIL** |
| C1d | report claim #4: repo-wide `${CUDA_VISIBLE_DEVICES:-...}` scan | **FAIL** |
| C2a | `pytest molmetal/tests/test_wfa165_gpu_lock_guard.py` | PASS (15 passed) |
| C2b | negative control on the REAL pre-repair files (`git show HEAD:` copies) | PASS (4 findings, defeats=yes) |
| C2c | mutation test — does the test bite? | PASS (2 mutants, 3 tests FAIL each) |
| C3a | default device preserved (env.sh / setup_molmetal_env.sh / the `=1` launcher / pa15) | PASS |
| C3b | cited file:line pairs trace to the real files | PARTIAL (69/89 at t0; 20 stale 8 min later) |
| C3c | guard robustness on `env CUDA_VISIBLE_DEVICES=<v> <cmd>` | **FAIL — crash** |
| C3d | no torch, no write outside `--out` | PASS |
| C4 | post-repair census + artefact anchor | number PASS / anchor **FAIL** |
| C5 | scope discipline (4 paths, all allowed) | PASS |
| C6 | undeclared blind spots (`unset`, `env -u`) | **FAIL (minor)** |

## C3c — the blocking defect (reproduce)

`molmetal/scripts/wfa165_gpu_lock_guard.py:187` defines `_SHELL_ENV_CMD_RE` for the
`env CUDA_VISIBLE_DEVICES=<v> <cmd>` form and lines 329-332 / 345-346 handle it — but the handler
then calls `m.group("mods")` (line 335) on a regex that has no `mods` group:

    scan_shell_text('env CUDA_VISIBLE_DEVICES=0 python x.py')
    -> IndexError: no such group   (molmetal/scripts/wfa165_gpu_lock_guard.py:335)

End-to-end (CLI): a temp git repo with one `.sh` holding that line -> traceback, **EXIT=1**,
stdout 0 bytes, `--out` artefact NOT written. `scan_repository` has no per-file try/except
(line 1117). Exit 1 is indistinguishable from "findings exist -> gate the launch".
0 tracked/untracked files use the form today, so the census NUMBERS are unaffected (latent defect).

## C4 — post-repair census (headline)

* 03:40:55Z (branch C's artefact): **89 findings / 54 defeats**; M1 35, M2 0, M3 27, M4 27; 72 files.
* 03:48:36Z (my re-run, after a concurrent Wf-A167 branch landed): **69 findings / 34 defeats**;
  M1 15, M2 0, M3 27, M4 27; 53 files.
* Working-tree anchor for my number: `git diff --binary | sha256sum` =
  `7a243b8337e54a283a20e1c77472b329ed157efc1dd07e600b8aa085711d45f4` (03:48:19Z).
* The artefact's `git_head` + `generated` do **not** anchor the snapshot (comparability_breaks #3
  says they do): the git_head was identical across both censuses while the counts moved 89 -> 69.

Survivor M1 (12 unmask + 3 forces_blank) file:line list is in `refutation.json` check C4.

## C1c / C1d — report numbers that do not trace to disk

* Wf-A163 artefact on disk: `lock_defeaters=32`, `lock_forcers=43`, `files_enumerated_in_scope=6325`,
  `files_read=3005`, `hits_by_kind={274,293,1377,75,11}`. The report quotes 54 / 41 / 6311 / 2993 /
  {272,293,1306,70,11}. Branch C's own `mechanism_index.json` embeds the CORRECT numbers in
  `wf_a163_census_verification.artifact_summary` while its `note` repeats the wrong ones.
* `${CUDA_VISIBLE_DEVICES:-...}` is in **5** tracked `.sh` files, not 1 (M2 = 0 is still correct —
  the rest are non-assignment reads). Residual, pre-existing:
  `scripts/launch_dmpnn_clean5v3_loso.sh:148` prints `CUDA_VISIBLE_DEVICES: 1` under the lock and
  when unset (the script then exports 0).

Full evidence: `refutation.json` (same directory).
