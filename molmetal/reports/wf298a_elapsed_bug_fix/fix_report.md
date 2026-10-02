# Wf-298A — Stale-Read Guard `elapsed` Bug Fix Report

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** HONEST-NEGATIVE — described bug pattern does not exist in target files

## 1. Scope of task

Task asked to fix `'elapsed is not defined'` in three launch wrappers:

- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh` (116 LOC)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` (156 LOC)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh` (195 LOC)

…by ensuring `local start_s elapsed last_epoch` is declared before a
`while true; do … elapsed=$(( $(date +%s) - start_s )) … done`
stale-read guard loop.

## 2. Findings

**No stale-read guard function exists in any of the three wrappers.**

Direct evidence:

```text
$ grep -n "while\|elapsed\|max_wait\|stale\|guard\|poll" \
    scripts/launch_wf124_p3_g4.sh \
    scripts/launch_wf124_p3_g5.sh \
    scripts/launch_wf124_p3_g6.sh
launch_wf124_p3_g6.sh:37:#   --epochs 50 --batch-size 256 --hidden 300 --depth 3 + use-l2-guard
launch_wf124_p3_g6.sh:47:  --use-l2-guard
launch_wf124_p3_g6.sh:119:# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
launch_wf124_p3_g4.sh:32:  --use-l2-guard
launch_wf124_p3_g4.sh:62:# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
launch_wf124_p3_g5.sh:33:# Per Wf-263C task spec: --epochs 50 --batch-size 256 --hidden 300 --depth 3 + use-l2-guard
launch_wf124_p3_g5.sh:42:  --use-l2-guard
launch_wf124_p3_g5.sh:81:# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
```

Only matches are `--use-l2-guard` (D-MPNN CLI flag, unrelated) and a
"Wave-tool guards" comment about pre-flight / per-wave-checklist / budget-tracker
Python invocations. There is **no `while` loop, no `elapsed` variable, no
`max_wait_s` parameter, and no stale-read guard function** in any of the three
files. Each wrapper is a straight-line bash script that:

1. Calls the three Python guard scripts (`pre_flight_check`,
   `per_wave_checklist`, `wave_budget_tracker`) once each.
2. Runs `run_pair` / `run_tuple` for every (metal, seed) / (config, metal, seed)
   tuple back-to-back (no polling).
3. Aggregates `per_seed_metrics_seed*.json` into a `training_summary.json`.

## 3. Cross-ref: where the actual stale-read guard work lives

Per the Wf-285 / Wf-291 synthesis reports, the actual missing deliverable is
a **Python** helper:

- `read_anchor_json(path, max_age_sec=..., required_version=...)`

…intended for `m14_anchor.json`, `cfm_decode_real.json`, and
`per_metal_pass.json` consumers (see
`molmetal/reports/wf285_synthesis/followup_summary.md` §1.4 + §P2 and
`molmetal/reports/wf291_synthesis/followup_summary.md` §MISSING-DELIVERABLE
HN-1 + HN-4).

That work was queued as **Wf-291H** (~45 min CPU) per
`molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` §7.4 — it has
**not** landed on disk. The bug template referenced by Wf-298A appears to
have been computed against the wrong file surface (Python helper code, not
the three `launch_wf124_p3_g*.sh` bash wrappers).

## 4. What was actually checked

| Check                                          | Result      |
| ---------------------------------------------- | ----------- |
| `bash -n scripts/launch_wf124_p3_g5.sh`        | SYNTAX OK   |
| `bash -n scripts/launch_wf124_p3_g4.sh`        | SYNTAX OK   |
| `bash -n scripts/launch_wf124_p3_g6.sh`        | SYNTAX OK   |
| `bash scripts/launch_wf124_p3_g5.sh --dry-run` | DRY-RUN OK  |
|   (15 (metal, seed) pairs enumerated, last line "DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.") |

The wrappers are healthy. No edits applied.

## 5. Honest framing (party-twenty-spirit §3.5)

- **n_wrappers_fixed: 0** — no `elapsed` bug exists in the three files.
- **n_wrappers_modified: 0** — refuse to fabricate a stale-read guard into
  the launch wrappers, because (a) the described bug is absent, (b) the three
  files are launch-only bash (no JSON anchor reads), and (c) inserting a
  phantom polling loop would be a regression risk on the live Wf-124 P3 sweep.
- **Recommended next action:** re-dispatch **Wf-291H** to land
  `read_anchor_json(path, max_age_sec=..., required_version=...)` + 2-3 tests
  under `molmetal/tests/test_stale_read_guard.py`, exactly as scoped by
  `wf285_synthesis/followup_summary.md` §P2. That closes the real
  Wf-271B + Wf-287A m14 anchor re-use risk pattern that motivated Wf-285D /
  Wf-291B in the first place.

## 6. Output summary (PLAIN TEXT)

- n_wrappers_fixed = 0 (described `elapsed is not defined` bug does not exist in any of the 3 launch wrappers)
- syntax_check = PASS (all 3 files: `bash -n` returns 0; "SYNTAX OK" emitted)
- dry_run_verdict = PASS (`launch_wf124_p3_g5.sh --dry-run` exits 0 with "DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.")

**End of Wf-298A report. No code changes applied.**
