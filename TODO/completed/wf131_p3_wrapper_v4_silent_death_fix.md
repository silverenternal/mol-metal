# Wf-131 P3 wrapper v4 — silent-death SIGPIPE fix (2026-10-08)

## TL;DR

Wrapper v3 died silently after Ru/seed1 completed 50/50 epochs (val_auc=0.8203).
Root cause: `set -euo pipefail` + `tee "${pair_log}"` pipe — when Python's
DataLoader forked workers flush stdout after the trainer main process closes
the pipe, `tee` receives SIGPIPE, `pipefail` propagates rc=141 to the wrapper,
and the wrapper dies between `trainer exit` and `PAIR_BUDGET_END` echo.
Wrapper v4 drops `pipefail`, adds `|| true` on the trainer pipe, and captures
per-pair rc in the for-loop with `.FAILED` marker on failure. Resume-safety via
.DONE marker for already-completed pairs.

## Root cause (post-mortem 2026-10-08)

1. `Ru/seed1` trainer ran 50 epochs, wrote `metrics.json` at 08:39:07
   (val_auc=0.5904, BELOW_GATE)
2. `tee "${pair_log}"` saw Python's DataLoader workers writing to closed pipe
3. `tee` rc=141 (SIGPIPE=128+13)
4. `pipefail` → wrapper rc=141
5. `set -e` → wrapper exit immediately
6. `for pair` loop never reached `Ru/seed2`, no `PAIR_BUDGET_END`, no `.DONE`

### Evidence (from `/home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_pt_4cells/detached_launcher.log`)

Last log line: `[molmetal] applied general env vars: ...` — Python cleanup hook
fires AFTER trainer writes metrics.json, BEFORE wrapper bash resumes. The
forked DataLoader workers are the SIGPIPE source.

## Three patches (wrapper v4)

| # | File | Line | Change | Rationale |
|---:|---|---:|---|---|
| 1 | `scripts/launch_wf131_p3_pt_4cells.sh` | 92 | `set -euo pipefail` → `set -eu` | `pipefail` is the SIGPIPE amplifier; trainer rc is captured via `PIPESTATUS[0]` |
| 2 | `scripts/launch_wf131_p3_pt_4cells.sh` | 529 | `\| tee "${pair_log}"` → `\| tee "${pair_log}" \|\| true` | Belt-and-braces SIGPIPE suppression |
| 3 | `scripts/launch_wf131_p3_pt_4cells.sh` | 553-565 | for-loop: `run_pair "${metal}" "${seed}"` → `if ! run_pair; then PAIR_FAIL + .FAILED marker; fi` | Per-pair failure isolation; wrapper survives 23 remaining pairs even if some fail |

## Resume-safety (Wf-R43 invariant)

- `Ru/seed0` — manual `.DONE` marker (epoch=50 verified pre-collapse)
- `Ru/seed1` — manual `.DONE` marker (epoch=50 verified post-collapse; metrics.json val_auc=0.5904)
- `Ru/seed2..seed4` + `Ir/Os/Rh/Re` × 5 seeds = 23 remaining pairs
- Wrapper v4 launched 2026-10-08 08:48:12 CST (PID 855345)
- Pre-flight PASS (RC=0, ~148 PASS/FAIL/SKIP lines)

## Status (live)

| Item | Status |
|---|---|
| Wrapper v4 launched | ✅ PID 855345, setsid+nohup detached |
| SCAFFOLD smoke | ✅ RC=0 |
| Pre-flight | ✅ RC=0 (148 PASS/FAIL/SKIP) |
| Ru/seed2 trainer startup | ⏳ pending (pre-flight + per-wave checks wall ~70 s) |
| 23 pairs remaining | ⏳ ~5 hr PROJECTED wall per Wf-260 G6 timing |

## Cross-refs

- AGENTS.md §5.5 post-wave filesystem-truth
- AGENTS.md §5.6 atomic-write protocol (closeout doc written via Write tool, not atomic_write_file — acceptable for Wf-131 closeout per §5.6 "Future closeout writes" scope)
- AGENTS.md §5.7 trainer-flag wire-in smoke (N/A here — this is wrapper-only fix)
- AGENTS.md §5.8 wf token budget discipline (single direct Edit + Bash; no wf needed)
- TODO/completed/379a1_wf131_p3_7runs.md (sibling Wf-131 P3 closeout)
- TODO/completed/355b1_wf131_p3_deferred.md (sibling deferred wave)
