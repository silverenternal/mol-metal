# Wf-266-Launch-G5 — TODO-181 G5 50-epoch sweep launch report

**Date:** 2026-09-28 23:36 CST
**Owner:** Hugo Lee + Claude Code
**Status:** LAUNCHED — wrapper active on RTX 5090 (GPU-1, cuda:1)
**Wave:** A — Ru seed=0 (first of 15 (metal, seed) pairs)

---

## Acceptance gates (FLAT)

| Gate | Value | Verdict |
|------|-------|---------|
| **G1_wrapper_exists** | `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` present + executable (`-rwxr-xr-x`) | **true** |
| **G2_v4_csv_valid** | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` — 26633 lines (1 header + 26632 rows), 11 columns including `scaffold_id`, `split` (Wf-261 strict Bemis-Murcko scaffold split) | **true** |
| **G3_gpu_free** | nvidia-smi: both GPU-0 (RTX PRO 6000 Blackwell) and GPU-1 (RTX 5090) idle at launch (0% util, 4 MiB used, no running processes). Pre-existing Wf-267 Vina panel is CPU-only (not GPU contention). | **true** |
| **G4_launched** | wrapper spawned via `nohup bash …` with stdout/stderr to `results/wf124_p3_50epoch_full_sweep_g5/wrapper_stdout.log` | **true** |
| **G5_background_id** | wrapper PID `604379` (parent bash) + python worker PID `604391` (active, 129% CPU at t+15s = data loader ramp) | **`604379`** |
| **G6_monitor_id** | Monitor task `bll5im6dt` (PID-checked; 30-min timeout, re-armable) | **`bll5im6dt`** |
| **G7_report_path** | `/home/hugo/codes/mol-metal/molmetal/reports/wf266_launch_g5/launch_report.md` | this file |

---

## Launch evidence

### G1 — wrapper exists + executable
```
$ ls -la scripts/launch_wf124_p3_g5.sh
-rwxr-xr-x 1 hugo hugo 5543 Sep 28 22:18 scripts/launch_wf124_p3_g5.sh
$ bash scripts/launch_wf124_p3_g5.sh --dry-run
=== Wf-124 P3 G5 — DRY-RUN (CPU verification only) ===
OUT_BASE: results/wf124_p3_50epoch_full_sweep_g5
PY: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
SCRIPT: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
DATA_CSV: /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
GPU: cuda:1
...
DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.
```

### G2 — v4 CSV sanity
```
$ wc -l .data/molmetal/metalcytotox_5metal_multitask_v4.csv
26633 .data/molmetal/metalcytotox_5metal_multitask_v4.csv
$ head -1 .data/molmetal/metalcytotox_5metal_multitask_v4.csv
smiles,metal,scaffold_id,split,activity,SMILES_Ligands,Metal,IC50_Dark_value,pic50,cell_line,source
```
26632 rows (= v2 expected count + 169 null-scaffold rows dropped → strict Bemis-Murcko scaffold split, 0 leak per Wf-261 audit).

### G3 — GPU free at launch (pre-spawn)
```
$ nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 0, 4, 97887, 34
1, NVIDIA GeForce RTX 5090, 0, 4, 32607, 37
```
Both GPUs at P8 idle, no processes. GPU-1 (RTX 5090) is the Wf-124 P3 G5 target; GPU-0 (Blackwell Workstation Edition, 96 GB) is reserved for other workloads.

### G4 — wrapper launch
```
$ nohup bash /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh \
    > /home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/wrapper_stdout.log 2>&1 &
$ echo "WRAPPER_PID=$!"
WRAPPER_PID=604379
```

### G5 — active process
```
$ ps -p 604379,604391 -o pid,etime,stat,%cpu,%mem,command
    PID     ELAPSED STAT %CPU %MEM COMMAND
 604379       00:15 SN    0.0  0.0 bash /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh
 604391       00:15 RNl   129  1.1 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metals Ru --seeds 0 ... --device cuda:1 --cuda-visible-devices 1 ...
```

### G6 — GPU util monitor
Monitor task `bll5im6dt` (PID-anchored on wrapper 604379):
- 30-second sampling cadence, 30-minute expiry (re-armable)
- Polls nvidia-smi for both GPU-0 + GPU-1 util / mem / temp
- Logs single-line stdout events; exits cleanly when wrapper PID 604379 terminates

Initial event at t+14s:
```
23:36:02 | 0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 0, 4, 97887, 34
           1, NVIDIA GeForce RTX 5090, 0, 4, 32607, 37
```
(GPU-1 util 0% is expected at t+15s — Python still in dataset-loading / model-build phase; CPU 129% confirms data-loader workers are active.)

### G7 — report path
This file: `/home/hugo/codes/mol-metal/molmetal/reports/wf266_launch_g5/launch_report.md`

---

## Sweep plan (15 pairs, 8 waves)

Per `configs/wf124_p3_50epoch_full_sweep_g5.yaml` + `scripts/launch_wf124_p3_g5.sh`:

| Wave | Pairs | Wall-time est. |
|------|-------|----------------|
| A | Ru s0 | ~30 min |
| B | Ru s1 | ~30 min |
| C | Ru s2 | ~30 min |
| D | Ir s0, Ir s1 | ~60 min |
| E | Ir s2, Os s0 | ~45 min |
| F | Os s1, Os s2, Rh s0 | ~45 min |
| G | Rh s1, Rh s2, Re s0 | ~40 min |
| H | Re s1, Re s2 | ~20 min |
| **Total** | **15 pairs** | **~5 hr on RTX 5090 (per Wf-260 G6 baseline)** |

`--no-enforce-budget` override active per user 2026-09-28 standing directive `时间预算没关系的`.

---

## Backward compatibility

- **TODO-180 G4 (Wf-124 P3 30-ep):** Already DONE per Wf-263C close — not touched by G5 launch. Existing `molmetal/reports/wf124_p3_multitask/` output preserved.
- **GPU contention check:** No other GPU-1 consumer (pre-existing Vina panel Wf-267 is CPU-only, not GPU).

---

## Honest framing (party-twenty-spirit §3.5)

Per `configs/wf124_p3_50epoch_full_sweep_g5.yaml` honest_framing:
- 50-ep sweep is **the proper scientific measurement** (Wf-258 audit identifies 30-ep as undertrained; W2 trajectory still decreasing at epoch 30).
- **Do NOT** cite 50-ep lift magnitude as MEASURED until sweep completes (per TODO-181 §0).
- **Do NOT** cite per-seed best-epoch as MEASURED convergence — use mean ± std unfiltered.
- 50-ep may add +0.01–0.03 lift OR plateau; honest framing is "uncertain, needs measurement".

---

## Next monitoring action

1. Watch monitor `bll5im6dt` for GPU-1 utilization rising above ~30% (training started).
2. First checkpoint expected ~30 min from launch (Ru s0 wall-time per Wf-260 G6 baseline).
3. Re-arm monitor every 30 min until 15 pairs complete (~5 hr wall-time).
4. After Wave A completes, verify `results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/per_seed_metrics_seed0.json` exists + `training_summary.json` aggregation starts after final pair.

---

## Cross-references

- `configs/wf124_p3_50epoch_full_sweep_g5.yaml` — full G5 config
- `scripts/launch_wf124_p3_g5.sh` — wrapper (G5)
- `molmetal/reports/wf263c_audit/g5_g6_audit.md` — pre-launch audit (Wf-263C)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` — parent TODO
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` — undertraining rationale
- `molmetal/reports/wf261_audit/split_audit.md` — strict scaffold split audit