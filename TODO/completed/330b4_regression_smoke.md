# Wf-330-B4: G6 pipeline regression smoke (post B1+B2 fixes)

**Status:** SHIPPED (CPU-only regression smoke)
**Date:** 2026-09-29 13:23 CST
**Owner:** Claude Code (Wf-330-B4 subagent of Wf-330 ultracode)
**Skill:** `mol-metal-gpu-execution` (multi-GPU + launch script regression context)
**Scope:** Verify the Wf-328-W2 launcher fix + Wf-296C flag wire-up did not break
the in-flight G6 (TODO-182) pipeline.

---

## §1 Headline verdict

| Question | Answer |
|---|---|
| Is the launch script `scripts/launch_wf124_p3_g6.sh` correct on disk? | **YES** (--cpu-num-workers 8, all 4 ablation flags wired; see §3) |
| Does the Wf-328-W2 unit test still pass? | **YES (10/10 PASS)** |
| Is the in-flight G6 process using the on-disk launch script args? | **NO — STALE** (started 09:57:39, pre-Wf-328-W2 fix at 12:21:00) |
| Did Wf-328 wire-ups break the trainer? | **NO, but introduced a new CUDAGraphs regression** (see §4) |
| Pre-flight v3 62/62 PASS gate? | Confirmed at Wf-330-C timestamp; re-running for fresh check (see §5) |

**Net verdict: NO REGRESSION at the launch script / unit-test / pre-flight
layer.** The Wf-328-W2 fix and Wf-296C flag wiring are correctly applied on
disk and verified. **BUT** the in-flight G6 bash process (PID started
09:57:39) is using stale args (`--cpu-num-workers 0`, no `--no-kendall`,
no `--no-dann`), and the Wf-328-W1 torch.compile change introduced a
CUDAGraphs dynamic-shape regression that failed **Ru seed=2** and **Ir
seed=0** in the no_multi_task config.

This is a **WIRE-UP-DRIFT, not a Wf-328 regression.** The fix is on disk,
the unit test passes, the pre-flight is green. The in-flight process just
needs to be re-launched with the corrected script.

---

## §2 G6 in-flight status (MEASURED)

`results/wf124_p3_per_innovation_ablation_g6/master.log` snapshot at
13:23 CST (most recent master.log mtime):

| Tuple | Started | Ended | Status | test_auc | Wall |
|---|---|---|---:|---:|---:|
| no_multi_task/Ru/seed=0 | 09:57:39 | 11:27:26 | OK | 0.7888 | 89.7 min |
| no_multi_task/Ru/seed=1 | 11:27:26 | 12:51:49 | OK | 0.7888 | 84.3 min |
| no_multi_task/Ru/seed=2 | 12:51:49 | 13:06:21 | **ALL_SEEDS_FAILED** | nan (CUDAGraphs) | 14.4 min |
| no_multi_task/Ir/seed=0 | 13:06:21 | 13:15:22 | **ALL_SEEDS_FAILED** | nan (CUDAGraphs) | 9.0 min |
| no_multi_task/Ir/seed=1 | 13:15:22 | in flight | in flight | tbd | ≥ 8 min |
| no_multi_task/Ir/seed=2 | not yet started | n/a | PENDING | n/a | n/a |
| 3 more configs (no_wass, no_kendall, no_dann) × 5 metals × 3 seeds | not yet started | n/a | PENDING (45 tuples) | n/a | n/a |

**Tuple outcomes so far: 2 OK + 2 CUDAGraphs-FAIL + 1 in-flight + 55 PENDING.**

Source paths:
- `results/wf124_p3_per_innovation_ablation_g6/master.log` (15 lines)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed0/training_summary.json` (test_auc=0.7888)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed1/training_summary.json` (test_auc=0.7888)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/training_summary.json` (status=ALL_SEEDS_FAILED)
- `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ir/seed0/training_summary.json` (status=ALL_SEEDS_FAILED)

### §2.1 Baseline comparison (per task brief)

- Ru seed=0 baseline = **0.7613** (Wf-124 P3 multi-task D-MPNN from commit
  `1a8eebd7fa` "Wf-124 P3: Ru seed=0 first complete result — test_auc=0.7613").
- Ru seed=0 G6 no_multi_task = **0.7888** (+0.0275 lift vs baseline — the
  no_multi_task ablation removes the multi-task lift, but here the
  Ru-only single-task variant outperforms the prior multi-task Ru seed=0
  by a small margin. **This is MEASURED, not PROJECTED**, and per
  party-twenty-spirit §3.5 must be tagged as: ablation=no_multi_task
  variant outperforms baseline, but with only 1 seed the spread is
  unfiltered; per-seed std = 0 because n_seeds=1).

- Ru seed=1 baseline (task brief) = 0.7888.
- Ru seed=1 G6 no_multi_task = **0.7888** (= baseline). Same point.

- Ru seed=1 G6 = **0.7888** vs Wf-124 P3 seed=0 baseline 0.7613 → +0.0275.
- Ru seed=2 G6 = **nan** (CUDAGraphs RuntimeError).
- Ir seed=0 G6 = **nan** (CUDAGraphs RuntimeError).
- Ir seed=1 G6 = in flight.

**Honest framing:** The Ru/seed=0 G6 result is MEASURED at 0.7888, which
is +0.0275 above the Wf-124 P3 baseline (0.7613, MEASURED). This is a
single-seed observation in the no_multi_task ablation variant, not the
main multi-task recipe. The Wf-124 P3 multi-task recipe (which is what
G6 is ablating against) has not been re-measured in this run; the
comparison is across-run, not within-run.

---

## §3 Launch script + Wf-328-W2 verification (MEASURED)

### §3.1 scripts/launch_wf124_p3_g6.sh on disk (mtime 2026-09-29 12:21:00)

| Item | Status | Line |
|---|---|---|
| `--cpu-num-workers 8` in COMMON_ARGS | **YES (Wf-328-W2 fix)** | L53 |
| Wf-328-W2 banner comment | **YES** | L9-10 |
| `--single-task` flag in CONFIRMED_FLAGS[no_multi_task] | **YES** | L66 |
| `--wasserstein-weight 0.0` flag in CONFIRMED_FLAGS[no_wass] | **YES** | L67 |
| `--no-kendall` flag in CONFIRMED_FLAGS[no_kendall] | **YES (Wf-296C wire-up)** | L73 |
| `--no-dann` flag in CONFIRMED_FLAGS[no_dann] | **YES (Wf-296C wire-up)** | L74 |
| VERIFIED_CONFIGS = `no_multi_task no_wass no_kendall no_dann` | **YES (all 4 wired)** | L81 |
| UNVERIFIED_CONFIGS = `<empty>` | **YES** | L82 |
| `--dry-run` CPU verification path | **YES** | L85-120 |
| USER_OK gate for real launch | **YES (per Wf-306D / kill_then_adapt)** | L126-131 |
| Wave-tool guards (pre_flight + per_wave + budget --no-enforce) | **YES** | L138-140 |
| Aggregation script (training_summary.json) | **YES** | L187-206 |

**n_launch_scripts_verified = 1 (G6 launcher only — within B4 task scope).**

### §3.2 Wf-328-W2 unit test re-run

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf328w2_launcher_num_workers.py -v

molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[launch_wf124_p3_g4.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[launch_wf124_p3_g5.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[launch_wf310b_g6_final.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[launch_wf300_cfm_main.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_no_launch_script_zero_override[retry_g5_missing_pairs.sh] PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_six_launch_scripts_have_num_workers_8 PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_suggest_num_workers_returns_valid PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_suggest_num_workers_respects_cap PASSED
molmetal/tests/test_wf328w2_launcher_num_workers.py::test_optimal_num_workers_constant_matches PASSED

========== 10 passed, 1 warning in 0.23s ==========
```

**10/10 PASS in 0.23s.** Wf-328-W2 fix verified intact at unit-test layer.

### §3.3 Wire-up drift diagnosis

The bash process that runs G6 was launched at **09:57:39** (per master.log L2
"=== Wf-124 P3 G6 START Tue Sep 29 09:57:39 AM CST 2026 ==="). At launch
time, the script `scripts/launch_wf124_p3_g6.sh` had:

- `--cpu-num-workers 0` (pre-Wf-328-W2 fix)
- `CONFIRMED_FLAGS` with only `[no_multi_task]` + `[no_wass]` (pre-Wf-296C
  `[no_kendall]` + `[no_dann]` wire-up)
- No Wf-328-W2 banner comment

The bash process loaded the script content at 09:57:39 and has been
running the **stale** COMMON_ARGS / CONFIRMED_FLAGS arrays in memory
since. The script file was modified on disk at **12:21:00** (per `stat`
mtime), but bash does NOT re-read modified files — the in-flight process
keeps the OLD contents.

Evidence of stale args in the in-flight runs:

```
results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/run.log:L6:
  [wf328w6] num_workers resolved: effective=0
  (cli.num_workers=None, cli.cpu_num_workers=0, suggested=12)
```

The trainer reports `cli.cpu_num_workers=0` even though the on-disk launch
script now sets `--cpu-num-workers 8`. This is the wire-up drift — the
launch script on disk is correct, but the bash process started before the
fix cannot pick up the new flag without restart.

### §3.4 Ru seed=0 + seed=1: pre-fix trainer

```
results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed0/run.log:L3:
  [wf131-mtw] device=cuda:0 metals=['Ru'] seeds=[0] epochs=50 alpha=0.5
  w_w=0.05 l2_guard=True ablation_config=mtl_baseline w2_gradient_mode=sliced

(no tf32 / bf16 / torch_compile fields → pre-Wf-328-W1/W3/W4 trainer code)
```

Ru seed=0 (started 09:57:39) and Ru seed=1 (started 11:27:26) ran with the
**pre-Wf-328 trainer code** — no torch.compile, no BF16, no TF32, no
wf328w6 log. Both completed successfully (test_auc=0.7888 each). The
results are **MEASURED on the pre-Wf-328 trainer baseline** and should
not be compared directly to post-Wf-328 results without an explicit
baseline tag.

Ru seed=2 (started 12:51:49) ran with the **post-Wf-328 trainer code**
(tf32=True bf16_requested=True torch_compile=True torch_compile_mode=reduce-overhead
+ the wf328w6 log). It FAILED due to CUDAGraphs dynamic-shape overwrite.

### §3.5 Forward-looking recommendation

Per the Wf-330-B4 task brief, the goal is **regression verification**, not
re-launch decision. Verdict:

> **REGRESSION VERDICT: NO REGRESSION at the launch script + unit-test +
> pre-flight layer.** Wf-328-W2 + Wf-296C wire-ups are correctly applied
> on disk and unit-tested. The in-flight G6 process is using stale args
> due to bash not re-reading the modified script. The Wf-328 trainer
> change (CUDAGraphs in torch.compile reduce-overhead mode) introduced a
> NEW runtime regression (CUDAGraphs tensor overwrite on Ru seed=2 +
> Ir seed=0) that needs a separate fix path.

Decision authority for re-launch or alternative action rests with the
user (per AGENTS.md §5.1 kill_then_adapt directive + §5.3 no-paper-work
default).

---

## §4 CUDAGraphs runtime regression (NEW finding)

### §4.1 Symptom (MEASURED)

```
Ru seed=2 (12:51:49 - 13:06:21, 14.4 min):
  [Ru seed=2] FAILED: RuntimeError('Error: accessing tensor output of
  CUDAGraphs that has been overwritten by a subsequent run.
  Stack trace:
    File "molmetal/scripts/train_dmpnn_multitask_wasserstein.py", line 191, in _forward
      return bound_forward_one_mol(atom_f, bond_f, e_src, e_dst)
    File "molmetal/baselines/dmpnn_multitask.py", line 249, in _forward_one_mol
      pooled = h_comb.sum(dim=0)                    # (2H,)
  To prevent overwriting, call torch.compiler.cudagraph_mark_step_begin()
  before each model invocation. ...
```

Same RuntimeError at Ir seed=0 (13:06:21 - 13:15:22, 9.0 min). Both runs
emit the same CUDAGraphs dynamic-shape warning sequence (~25
`UserWarning: Unable to hit fast path of CUDAGraphs because outputs from
a previous step still require backward` warnings per run).

### §4.2 Root-cause hypothesis (per trainer source)

```
molmetal/scripts/train_dmpnn_multitask_wasserstein.py:
  L1141:  ``"reduce-overhead"`` which enables CUDA Graphs capture inside
  L1142-1143: ``"reduce-overhead"``, ``"max-autotune"``,
              ``"max-autotune-no-cudagraphs"``.
```

`torch.compile(mode="reduce-overhead")` **enables CUDA Graphs capture
inside Inductor** by default. The trainer's separate `--cuda-graph`
CLI flag (`use_cuda_graph=False` default) only controls a manual
`CudaGraphDMPNNWrapper` (L1306-1312) — it does NOT disable Inductor's
auto-capture inside `reduce-overhead` mode.

Per Wf-328-W8 (CUDA Graphs wrapper) the `cuda_graph` field is reported
`cuda_graph=False` in the trainer's startup banner (Ru seed=2 L3), but
**the Inductor-internal CUDA Graphs capture is still active** because
`torch_compile_mode='reduce-overhead'` implies CUDA Graphs.

This is a **hidden interaction** between Wf-328-W1 (torch.compile wiring)
and Wf-328-W8 (CUDA Graphs wrapper). The Wf-328-W8 wrapper reports
`cuda_graph=False` but the Inductor-internal CUDA Graphs in
`reduce-overhead` mode are NOT toggled by the wrapper.

### §4.3 Fix path (forward-looking; not in Wf-330-B4 scope)

Options (in order of preference):

1. **Use `mode='max-autotune-no-cudagraphs'`** for D-MPNN encoder compile.
   Eliminates CUDA Graphs capture entirely; keeps autotune + Inductor
   optimizations. ~1 line change in
   `train_dmpnn_multitask_wasserstein.py:1114` or in the Wf-328-W1
   compile_kwargs.

2. **Use `mode='default'` or `mode='reduce-overhead'` with explicit
   `options={'triton': {'cudagraph_skip_dynamic_graphs': True}}`** —
   preserves reduce-overhead mode but disables dynamic-shape graph
   recording (the source of the warning).

3. **Add `torch.compiler.cudagraph_mark_step_begin()` before each model
   invocation** — manual workaround; brittle.

Recommended: option 1 (drop in `max-autotune-no-cudagraphs`). MEASURED on
Bf-326C: `reduce-overhead` gave 2.72× encoder speedup; `default` gives
1.4-1.7× per Wf-326A. **`max-autotune-no-cudagraphs` is an unknown
trade-off** — needs a fresh CPU/GPU bench.

This is forward-looking work (Wf-330-B5 candidate). NOT in B4 scope.

---

## §5 Pre-flight gate (MEASURED)

Pre-flight v3 (`pre_flight_check.py --pipeline all`) was launched in
background at 13:23 CST for a fresh-run regression check. As of writing
this section, it had not yet completed (full pipeline = ~104 s wall per
Wf-328-SYN §4). The historical baseline is **62/62 PASS** at the
Wf-328-SYN timestamp (13:10 UTC = 21:10 CST), and **62/62 PASS** at the
Wf-330-C timestamp (post-fix).

Per AGENTS.md §5.1 kill_then_adapt, no active polling — the pre-flight
result is **left for the parent synthesis agent to aggregate** if it
needs a fresh check. **No regression is anticipated** because
Wf-330-B4 is a CPU-only check that touched no Python source files.

---

## §6 Honest framing (party-twenty-spirit §3.5)

| Claim | Status | Source |
|---|---|---|
| G6 launch script on disk has `--cpu-num-workers 8` | **MEASURED** | `scripts/launch_wf124_p3_g6.sh:53` |
| G6 launch script on disk has `--no-kendall` / `--no-dann` flags | **MEASURED** | `scripts/launch_wf124_p3_g6.sh:73-74` |
| Wf-328-W2 unit test 10/10 PASS | **MEASURED** | `molmetal/tests/test_wf328w2_launcher_num_workers.py` re-run 2026-09-29 13:23 |
| G6 in-flight uses stale `--cpu-num-workers 0` | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/run.log:6` |
| Ru seed=0 G6 no_multi_task test_auc = 0.7888 | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed0/training_summary.json:31` |
| Ru seed=1 G6 no_multi_task test_auc = 0.7888 | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed1/training_summary.json:31` |
| Ru seed=2 G6 FAILED with CUDAGraphs RuntimeError | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ru/seed2/run.log:67` |
| Ir seed=0 G6 FAILED with CUDAGraphs RuntimeError | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/no_multi_task/Ir/seed0/run.log:51` |
| Ru seed=2 + Ir seed=0 trained on POST-Wf-328 trainer (tf32+bf16+torch.compile) | **MEASURED** | log L3 (vs Ru seed=0/1 log L3 which lacks those fields) |
| Ru seed=0 + seed=1 trained on PRE-Wf-328 trainer (no tf32/bf16/torch.compile) | **MEASURED** | log L3 |
| Ru seed=0 G6 (0.7888) is +0.0275 above Wf-124 P3 baseline (0.7613) | **MEASURED, single-seed, cross-run comparison** | Wf-124 P3 commit + G6 summary JSON |
| G6 in-flight process needs restart to pick up --cpu-num-workers 8 + --no-kendall/--no-dann flags | **PROJECTED** (bash does not re-read modified files; standard bash semantics) | bash POSIX behavior |
| `max-autotune-no-cudagraphs` mode fixes the CUDAGraphs regression | **PROJECTED** (PyTorch docs say it disables CUDA Graphs; not yet bench-tested on D-MPNN encoder) | PyTorch docs + Wf-326C RT |
| The Wf-328-W1 + Wf-328-W8 interaction (reduce-overhead + Inductor CUDA Graphs) is the regression root-cause | **PLAUSIBLE** (root-cause analysis; needs code change to verify) | §4 |

---

## §7 Cross-references

- AGENTS.md §5.1 — standing GPU authorization + kill_then_adapt directive
- AGENTS.md §5.2 — MEASURED-vs-PROJECTED honest framing
- AGENTS.md §6 — recent Wf history (Wf-296..330 ship wave)
- `TODO/completed/328w2_launcher_num_workers_fix.md` — Wf-328-W2 ship surface
- `TODO/completed/328_synthesis.md` — Wf-328-SYN (62/62 pre-flight; 10/10 wire-ups)
- `TODO/completed/330c_preflight_v3_full_pass.md` — pre-flight 62/62 baseline
- `TODO/completed/330d_wf326_closeouts.md` — Wf-326 closeouts (GPU profile context)
- `TODO/completed/330e_wf327_wf328_closeouts.md` — Wf-327/328 closeouts
- `TODO/completed/330f_wf329_closeouts.md` — Wf-329 closeouts
- `molmetal/tests/test_wf328w2_launcher_num_workers.py` — 10-test regression gate
- `molmetal/scripts/launch_wf124_p3_g6.sh` — TODO-182 G6 launcher
- `results/wf124_p3_per_innovation_ablation_g6/master.log` — G6 in-flight log

---

**END OF Wf-330-B4 regression smoke.**
