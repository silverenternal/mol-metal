# Wf-337-G1 — G6 55 PENDING + 28 FAILED tuple recovery

**Status:** SHIPPED (CPU-only; recovery launcher READY, real launch USER_OK-gated)
**Date:** 2026-09-29 15:14 CST
**Owner:** Claude Code (Wf-337-G1 subagent of Wf-337 ultracode)
**Skill:** `mol-metal-gpu-execution` (multi-GPU + launch script Wf-328-W2 + Wf-335 fixes context)

---

## §1 TL;DR

| Item | Status |
|---|---|
| `scripts/recover_g6_failed_tuples.sh` written + chmod +x | MEASURED |
| `--dry-run` enumerates manifest with 2 OK + 28 FAILED + 1 RUNNING + 29 PENDING = 60 | MEASURED |
| Recovery launch wires all Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes | MEASURED |
| `molmetal/tests/test_wf337g1_g6_recovery.py` (5 tests) PASS | MEASURED |
| Pre-flight v3 62/62 PASS gate inherited (no fresh retrain / panel launched) | MEASURED |

**Net verdict: RECOVERY LAUNCHER READY.** Real launch is
`USER_OK=1 bash scripts/recover_g6_failed_tuples.sh` per AGENTS.md §5.1
kill_then_adapt directive. Recovery will skip 2 OK + 1 RUNNING tuples
and re-launch the remaining 57 tuples with all Wf-328 + Wf-335 + Wf-296C
fixes wired into COMMON_ARGS / CONFIRMED_FLAGS.

---

## §2 G6 in-flight status (MEASURED, 2026-09-29 15:13 CST)

`results/wf124_p3_per_innovation_ablation_g6/master.log` snapshot at Wf-337-G1
task time (66 lines, last entry `=== 15:03:52 START no_kendall/Ru/seed=0 ===`):

| Bucket | Count | Tuples | Source |
|---|---:|---|---|
| OK | 2 | no_multi_task/Ru/seed=0, no_multi_task/Ru/seed=1 | training_summary.json status=OK |
| FAILED | 28 | Ru/2 + Ir 0-2 + Os 0-2 + Rh 0-2 + Re 0-2 from no_multi_task + all 15 from no_wass | training_summary.json status=ALL_SEEDS_FAILED (CUDAGraphs RuntimeError per Wf-330-B4 §4.1) |
| RUNNING | 1 | no_kendall/Ru/seed=0 (started 15:03:52) | run.log mtime < 30 min, no summary yet |
| PENDING | 29 | rest of no_kendall (14) + all of no_dann (15) | no run.log, no training_summary.json |
| TOTAL | 60 | 4 cfgs × 5 metals × 3 seeds | confirmed by manifest parser |

**TO_RECOVER (FAILED + PENDING) = 57 tuples.**

Source paths:
- `results/wf124_p3_per_innovation_ablation_g6/master.log` (66 lines, mtime 15:03:52)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed0/training_summary.json:40` (status=OK)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/run.log` (CUDAGraphs RuntimeError)
- `results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv` (NEW; emitted by `--dry-run`)

---

## §3 What changed (Wf-337-G1 ship surface)

### §3.1 NEW: `scripts/recover_g6_failed_tuples.sh` (196 LOC bash)

CPU-only state detector + USER_OK-gated recovery launcher.  Two modes:

```bash
bash scripts/recover_g6_failed_tuples.sh --dry-run
  # Scans results/, emits manifest, NO GPU work.
USER_OK=1 bash scripts/recover_g6_failed_tuples.sh
  # Re-launches FAILED + PENDING tuples with all Wf-328 + Wf-335 + Wf-296C fixes.
```

Key design points:

1. **Status detection** — Python helper scans each `(cfg, metal, seed)` triple:
   - If `training_summary.json` exists, recursively locate the leaf `"status"` key
     (the field is nested inside `data["seeds"][0]["status"]`, not at the top level).
   - If no summary but `run.log` mtime < 30 min → RUNNING (skip, leave alone).
   - If no summary but `run.log` mtime ≥ 30 min → FAILED (stale, no progress).
   - If no summary and no run.log → PENDING (never started).
2. **Manifest output** — writes `results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv`
   with columns `cfg<TAB>metal<TAB>seed<TAB>status<TAB>reason` (60 rows + header).
3. **Skip logic in run_tuple()** — same heuristics, so real launch does not
   re-run OK or RUNNING tuples. Same recovery_set / SKIP semantics.
4. **USER_OK gate (AGENTS.md §5.1 kill_then_adapt)** — `--dry-run` exits
   BEFORE the gate; real launch requires `USER_OK=1` env var.
5. **All fixes wired** — `--cpu-num-workers 8`, `--no-cuda-graphs`,
   `--no-kendall`/`--no-dann` per-config, torch.compile mode default
   `max-autotune-no-cudagraphs` (trainer default per Wf-335-R1).

### §3.2 NEW: `molmetal/tests/test_wf337g1_g6_recovery.py` (5 tests)

| # | Test | What it gates |
|---|---|---|
| 1 | `test_recovery_script_identifies_failed_and_pending` | Core recovery classification: 60 tuples total, OK + FAILED + RUNNING + PENDING, TO_RECOVER ≥ 50 |
| 2 | `test_recovery_launch_includes_all_wf328_wf335_fixes` | Wf-328-W2 + Wf-335-R1 + Wf-335-R3 + Wf-296C fixes all wired in `scripts/recover_g6_failed_tuples.sh` |
| 3 | `test_no_cudagraphs_flag_present` | Trainer `--no-cuda-graphs` / `--cudagraphs` flags defined; `cudagraphs_disabled` safety lever wired in main() |
| 4 | `test_backward_compat_with_dry_run` | `--dry-run` exits 0, prints expected headers, does NOT require USER_OK (backward compat) |
| 5 | `test_trainer_torch_compile_mode_default_is_max_autotune_no_cudagraphs` | Belt-and-suspenders: trainer function default + argparse default both `max-autotune-no-cudagraphs` |

All 5 tests are CPU-only (< 5 s wall, no torch.cuda, no real launches).

### §3.3 Test result (MEASURED 2026-09-29 15:14 CST)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf337g1_g6_recovery.py -v

molmetal/tests/test_wf337g1_g6_recovery.py::test_recovery_script_identifies_failed_and_pending PASSED
molmetal/tests/test_wf337g1_g6_recovery.py::test_recovery_launch_includes_all_wf328_wf335_fixes PASSED
molmetal/tests/test_wf337g1_g6_recovery.py::test_no_cudagraphs_flag_present PASSED
molmetal/tests/test_wf337g1_g6_recovery.py::test_backward_compat_with_dry_run PASSED
molmetal/tests/test_wf337g1_g6_recovery.py::test_trainer_torch_compile_mode_default_is_max_autotune_no_cudagraphs PASSED

========== 5 passed, 1 warning in 0.30s ==========
```

### §3.4 Dry-run result (MEASURED 2026-09-29 15:14 CST)

```bash
$ bash /home/hugo/codes/mol-metal/scripts/recover_g6_failed_tuples.sh --dry-run

=== Wf-337-G1 G6 RECOVERY START Tue Sep 29 03:14:49 PM CST 2026 ===
OUT_BASE: results/wf124_p3_per_innovation_ablation_g6
RECOVERY MANIFEST: results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv
  OK:        2
  FAILED:   28
  RUNNING:   1
  PENDING:  29
  TOTAL:    60  (expected 60)
  TO_RECOVER (FAILED + PENDING): 57

Recovery launch will skip OK + RUNNING tuples and re-launch the rest with
Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes wired (see COMMON_ARGS banner).

=== DRY-RUN OK: 1 ===
No GPU work performed. Set USER_OK=1 for real launch:
$ USER_OK=1 bash scripts/recover_g6_failed_tuples.sh
```

---

## §4 Wall-time projection (PROJECTED, AGENTS.md §5.2 honest framing)

| Path | Wall | Source |
|---|---:|---|
| `--dry-run` (CPU-only enumeration) | < 5 s | MEASURED |
| Real launch (57 tuples sequential on RTX 5090) | ~11.4 hr | PROJECTED (12 min/tuple based on Ru seed=3 timing in Wf-124 P3) |

Real launch is BLOCKED on AGENTS.md §5.1 kill_then_adapt directive —
USER_OK required. Per the user's standing GPU grant §5.1, the user can
authorize the launch with `USER_OK=1` and the script will run without
per-launch user OK once authorized.

The `--dry-run` path itself is not GPU-bound and was executed
successfully without USER_OK (verified by test 4).

---

## §5 Honest framing (party-twenty-spirit §3.5)

| Claim | Status | Source |
|---|---|---|
| 2 OK tuples (no_multi_task/Ru seed=0 + seed=1) | **MEASURED** | training_summary.json status=OK at L40 |
| 28 FAILED tuples (Ru/2 + Ir 0-2 + Os 0-2 + Rh 0-2 + Re 0-2 from no_multi_task + all 15 from no_wass) | **MEASURED** | training_summary.json status=ALL_SEEDS_FAILED + Wf-330-B4 §4.1 |
| 1 RUNNING tuple (no_kendall/Ru/seed=0) | **MEASURED** | master.log L66 (START 15:03:52, no END yet) |
| 29 PENDING tuples (rest of no_kendall + all of no_dann) | **MEASURED** | no run.log, no training_summary.json |
| CUDAGraphs RuntimeError root cause | **MEASURED** | Wf-330-B4 §4.1 + Wf-335-R1 §3 |
| `scripts/recover_g6_failed_tuples.sh` written + executable | **MEASURED** | `ls -la` 2026-09-29 |
| `scripts/recover_g6_failed_tuples.sh --dry-run` exits 0 + emits manifest | **MEASURED** | run 2026-09-29 15:14 CST |
| 5/5 NEW tests PASS | **MEASURED** | pytest output 2026-09-29 15:14 CST |
| `--cpu-num-workers 8` wired in recovery script COMMON_ARGS | **MEASURED** | recovery script L78 + test 2 |
| `--no-cuda-graphs` wired in recovery script COMMON_ARGS | **MEASURED** | recovery script L88 + test 3 |
| `--no-kendall` / `--no-dann` wired in CONFIRMED_FLAGS | **MEASURED** | recovery script L93-95 + test 2 |
| torch.compile mode default = `max-autotune-no-cudagraphs` | **MEASURED** | trainer L1121 + L2028 + test 5 |
| Recovery wall ~11.4 hr (57 tuples × ~12 min) | **PROJECTED** | based on Ru seed=3 + Wf-124 P3 baseline; BLOCKED on §5.1 to MEASURE |
| The recovery launch eliminates the CUDAGraphs regression | **PROJECTED** | Wf-335-R1 MEASURED on RTX PRO 6000 + Wf-330-B4 §4.3; needs to be MEASURED on RTX 5090 G6 path |

---

## §6 What Wf-337-G1 did NOT do

- **Did NOT launch the GPU recovery sweep.** Real launch is USER_OK-gated.
  The user must authorize `USER_OK=1 bash scripts/recover_g6_failed_tuples.sh`
  per AGENTS.md §5.1 kill_then_adapt directive.
- **Did NOT touch `paper/*` or paper §1-§7.** Per TODO-84 + AGENTS.md §5.3.
- **Did NOT modify the in-flight master.log or training_summary.json files.**
  Recovery launch will overwrite run.log for FAILED tuples (existing ckpts are
  preserved since `--out-dir` is the same per-tuple directory).
- **Did NOT modify `scripts/launch_wf124_p3_g6.sh`.** That script already
  has all the Wf-328-W2 + Wf-335 + Wf-296C fixes wired (verified by tests 2
  and 5 of `test_wf328w2_launcher_num_workers.py` and `test_wf335r2_wireup_reverify.py`).
  The recovery script is a separate launcher that classifies state first.

---

## §7 Cross-references

- AGENTS.md §5.1 — standing GPU grant + kill_then_adapt directive
- AGENTS.md §5.2 — MEASURED-vs-PROJECTED honest framing
- `TODO/completed/328w2_launcher_num_workers_fix.md` — Wf-328-W2 ship (--cpu-num-workers 8)
- `TODO/completed/335r1_no_cudagraphs_fix.md` — Wf-335-R1 (torch.compile mode default switch)
- `TODO/completed/335r3_g6_no_cudagraphs.md` — Wf-335-R3 (--no-cuda-graphs safety lever)
- `TODO/completed/330b4_regression_smoke.md` — Wf-330-B4 (CUDAGraphs regression MEASURED)
- `TODO/completed/296c_wf296_wireup.md` — Wf-296C (--no-kendall / --no-dann per-config levers)
- `molmetal/scripts/recover_g6_failed_tuples.sh` — NEW recovery script (Wf-337-G1)
- `molmetal/tests/test_wf337g1_g6_recovery.py` — NEW 5-test gate (Wf-337-G1)
- `results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv` — NEW manifest (Wf-337-G1)
- `results/wf124_p3_per_innovation_ablation_g6/master.log` — G6 in-flight state

---

**END OF Wf-337-G1 doc** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; cross-ref Wf-330-B4 + Wf-328-W2 + Wf-335-R1+R3 + Wf-296C; last update 2026-09-29 15:14 CST.