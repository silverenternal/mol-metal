# Wf-A170 branch 2 — adversarial refutation

**VERDICT: NOT REFUTED.**  Every load-bearing count in the branch report reproduced EXACTLY from disk.
Five concerns survive (two substantive), none of which contradicts a published number.

Artefacts: `molmetal/reports/wfa170_census_freshness_refutation/refutation.json` (full evidence).
CPU-only (`CUDA_VISIBLE_DEVICES=""`), single venv, `metrics/` byte-identical before/after every pytest run
(`025a37efadd2df79…`, 38 lines). 0 GPU. `os.walk` not used by me.

## Checks

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | census artefact REGENERATED, not hand-patched | PASS | producer re-run -> JSON-equal to shipped minus `generated_at`; `target_sha256`=live sha `ae878bcd…`; `target_lines`=17608 |
| C2 | old pin matched NEITHER HEAD nor worktree | PASS | HEAD `4c648e48237b…` (15487 L); old `56ac01199c82…` != worktree; old state RECOVERABLE at `/tmp/wfa159/pre_flight_check.py` |
| C3 | "only `line` moved on 5 sites (+62/+143/+158/+158/+172)" | PASS | producer re-run vs the 17436-line predecessor: exactly 5 sites changed, ONLY field `line`, deltas **[62,143,158,158,172]**, 0 other field changed, summary identical (35/35/0) |
| C4 | regression test bites | PASS | 15/15 + 20/20 PASS; M1 always-FRESH -> negative control FAILS; M2 always-STALE -> positive control FAILS; forbidden wfa159 test untouched (mtime 01:52) |
| C5 | not vacuous | PASS | 1 skip call, env-guarded and never firing; GPU probe is a real subprocess probe, not a skip |
| C6 | guard counts re-derived | PASS | 1823 pins / 160 STALE / 1656 FRESH / 7 MISSING / 138 UNPAIRED / 39 artefacts; stale SET byte-identical to a fresh run; independent re-hash: 160/160 stale confirmed, 0 false positives; 398/398 Wf-A166 cross-check reproduced exactly |
| C7 | patch applies, moves no PINNED count | PASS | `git apply --check` rc=0, applies cleanly (82 L), py_compile PASS, AST diff: only `main` changed, `EXPECTED_GATE_COUNTS` byte-identical {all 129, launch 131} |
| C8 | no scope creep; metrics/ untouched | PASS | branch footprint = 5 new files + 1 regenerated artefact; metrics/ sha256 identical before/after |

## Surviving concerns

- **C-A (substantive)** — neither headline finding is protected by a LIVE test: M4 (generalized pairing disabled), M5 (primary scope disabled) and M6 (targets never resolve) each leave the suite at **15/15 PASSED**. The 160-stale finding is TRUE but deletable.
- **C-B (substantive)** — the patch's claim "wrapped in try/except, so a missing or broken guard can never break a pre-flight run" is **false for the import**: the `from molmetal.scripts import wfa170_census_freshness_guard` is module-level, outside the try/except. REPRODUCED — with the guard absent, importing the patched pre-flight raises an uncaught `ImportError`. The guard file is currently untracked.
- **C-C (minor)** — the guard's `_write` hand-rolls a fixed `{path}.tmp` + `os.replace` instead of §5.6's `wf386a3_atomic_write.atomic_write_file` (no fsync, no per-path lock, no unlink-on-failure) — a concurrency hazard under parallel runs.
- **C-D (minor)** — 4 of the 15 tests read `report.json`, which `.gitignore:158` ignores; they FAIL on a clean checkout.
- **C-E (minor)** — `n_json_artefacts_scanned` is a moving target (3993 -> 3994 -> 3998) under parallel branches; the 1823/160/1656/7/138/39 numbers were stable across 3 runs. Guard runtime MEASURED 18.2 s (report says ~15 s).
- **C-F (note)** — the guard's `os.walk` is scoped to `molmetal/reports` (282968 files, 2.3 GB), never reads checkpoint content, 18.2 s MEASURED.
