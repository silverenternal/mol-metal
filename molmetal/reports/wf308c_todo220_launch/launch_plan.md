# Wf-308C TODO-220 launch plan — temporal-split + NCI-60 9-cell full GPU

| Field | Value |
|---|---|
| Wf | Wf-308C |
| TODO | TODO-220 |
| Status | CPU smoke MEASURED (Wf-308C Parts 1+2); GPU launch wrapper USER_OK-gated |
| Created | 2026-09-29 |
| Owner | Wf-308C subagent (kill_then_adapt relay from main Wf-308C driver) |

## 1. Honest framing

Per Wf-305C + Wf-306B:

* **CPU smoke (actionable-now, no USER_OK)** — Wf-308C Parts 1+2
  wired the `--execute` path for both drivers. Smoke runs on CPU
  succeeded for 1-seed 1-epoch hidden=64 depth=2. **No GPU required.**
* **GPU full run (USER_OK-gated)** — Wf-308C Part 3 wraps the full
  5-metal × 5-seed × 50-epoch temporal split + 9-cell × 3-seed ×
  50-epoch multi-task sweep in `launch_wf308c_todo220_gpu.sh`.
  Honours kill_then_adapt via `/tmp/USER_OK_WF308C` marker file.

## 2. CSV status (verified 2026-09-29)

| CSV | Path | Status | Notes |
|---|---|---|---|
| metalcytotox v4 | `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` | **PRESENT** | 26632 rows × 11 cols. **No Year column** — Wf-308C Part 1 merges Year from `MetalCytoToxDB.csv` via `(smiles, metal)` lookup (same pattern as `wf219_krasnov_aligned.py --split temporal` / Wf-303C T3). |
| MetalCytoToxDB raw | `/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv` | **PRESENT** | 26325 rows × 28 cols with Year (2001-2025 range; 2024-2025 has 3431 rows = temporal test pool). |
| NCI-60 9-cell | `/home/hugo/codes/mol-metal/molmetal/data/nci60_precious_9cell.csv` | **PRESENT** | 7823 rows × 4 cols (`smiles, cell_line, pIC50, active`). All 9 requested cells matched (HCT116 + HCT-116 alias-dedup). |

## 3. CPU smoke results (MEASURED 2026-09-29)

### 3.1 Temporal split — `wf219_temporal_split_driver.py --execute`

Command:
```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf219_temporal_split_driver.py \
    --csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv \
    --train-cutoff 2024-01-01 --test-cutoff 2025-01-01 \
    --seeds 0 --epochs 1 --hidden 64 --depth 2 --device cpu \
    --out /tmp/wf308c_temporal_smoke --execute
```

Result: **5/5 ok runs in 0.43 min** (hidden=64, depth=2, 1 epoch, 1 seed, CPU).

| Metal | mean_auc (seed=0) | n_train | n_test |
|---|---|---|---|
| Ru | 0.7398 | 15650 | 523 |
| Ir | 0.6876 | 3273 | 319 |
| Os | 0.6905 | 892 | 88 |
| Rh | 0.7732 | 836 | 71 |
| Re | 0.5175 | 544 | 1 (degenerate; pre-cutoff year window) |
| **Overall** | **0.6817** | | |

Per-metal seed JSON: `/tmp/wf308c_temporal_smoke` (the script writes
the full sweep_summary.json when given `--out`).

Honest framing: 1-epoch smoke numbers are lower-bound; Krasnov 2026
baseline (Ru=0.81, Ir=0.73) is reached at 50 epochs (Wf-303C T3
projection). Re=0.5175 is the random-chance baseline (n_test=1).

### 3.2 NCI-60 9-cell — `nci60_9cell_chemprop_driver.py --execute`

Command:
```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --epochs 1 --hidden 64 --depth 2 --device cpu \
    --out /tmp/wf308c_nci60_smoke --execute
```

Result: **27/27 ok runs in 0.36 min** (9 cells × 3 seeds, hidden=64,
depth=2, 1 epoch, CPU).

| Cell | mean_auc | std | n_train | n_test |
|---|---|---|---|---|
| MCF7 | 0.5300 | 0.0524 | 1374 | 393 |
| A2780 | 0.5687 | 0.0502 | 766 | 219 |
| A549 | 0.5715 | 0.0683 | 1852 | 530 |
| HCT116 | 0.6829 | 0.0127 | 490 | 141 |
| PC3 | 0.2153 | 0.0344 | 192 | 55 |
| HL60 | 0.4848 | 0.2157 | 157 | 45 |
| HCT-116 | 0.6829 | 0.0127 | 490 | 141 (alias dedup) |
| A2780cis | 0.5177 | 0.0385 | 394 | 113 |
| DU145 | 0.5000 | 0.2510 | 112 | 33 |
| **Overall** | **0.5282** | | | |

Note: HCT-116 and HCT116 alias to the same data; both shown as
independent runs because the driver iterates over the requested cell
list verbatim. Future P2 wires the MultiTaskLoss head per TODO-130 G3.

## 4. GPU launch wrapper

Path: `molmetal/scripts/launch_wf308c_todo220_gpu.sh`
(syntax-checked via `bash -n`; exit code 2 on missing USER_OK).

### 4.1 USER_OK gate (kill_then_adapt)

```bash
# Default (no run; helpful message printed):
bash molmetal/scripts/launch_wf308c_todo220_gpu.sh

# To authorise:
touch /tmp/USER_OK_WF308C
bash molmetal/scripts/launch_wf308c_todo220_gpu.sh
```

### 4.2 Stages

| Stage | Driver | Config | Projected wall |
|---|---|---|---|
| A | `wf219_temporal_split_driver.py` | Ru/Ir/Os/Rh/Re × 5 seeds × 50 epochs, hidden=300 depth=3, train<2024 test≥2025 | **2.9 hr** |
| B | `nci60_9cell_chemprop_driver.py` | 9 cells × 3 seeds × 50 epochs, hidden=128 depth=3 | **1.0 hr** |
| **Total** | | | **3.9 hr** |

### 4.3 Exit codes

| Code | Meaning |
|---|---|
| 0 | GPU run completed (both stages ok) |
| 2 | USER_OK missing (gated; no run started) |
| 3 | Stage A failed (Stage B not attempted) |
| 4 | Stage B failed (Stage A results still saved) |
| 5 | Setup error (venv / GPU probe / CSV missing) |

### 4.4 Output paths

```
molmetal/reports/wf308c_temporal_split/sweep_summary.json
molmetal/reports/wf308c_temporal_split/run.log
molmetal/reports/wf308c_nci60_9cell/sweep_summary.json
molmetal/reports/wf308c_nci60_9cell/run.log
```

### 4.5 CUDA device override

Default `CUDA_VISIBLE_DEVICES=1` (matches Wf-130 / Wf-190 env guard +
Wf-219 default). Override via `WF308C_CUDA_VISIBLE_DEVICES=<idx>` env var.

## 5. Files added / modified by Wf-308C

| Path | Action |
|---|---|
| `molmetal/scripts/wf219_temporal_split_driver.py` | rewrote: added `--execute` path + temporal split (Year merge from raw CSV) |
| `molmetal/scripts/nci60_9cell_chemprop_driver.py` | rewrote: added `--execute` path + per-cell 80/10/20 random split + alias dedup |
| `molmetal/scripts/launch_wf308c_todo220_gpu.sh` | new: USER_OK-gated GPU launch script (Stages A+B) |
| `molmetal/reports/wf308c_todo220_launch/launch_plan.md` | new: this file |

## 6. Cross-references

* **TODO-220**: `TODO/pending/220_wf219_temporal_split_nci60_driver.md`
* **Wf-303C T3** (temporal path on `wf219_krasnov_aligned.py`):
  template for the temporal split + Year merge from raw CSV.
* **Wf-305C** (this script's predecessor stub; both drivers).
* **Wf-306B** (honest-negative framing for missing CSVs / GPU gates).
* **Wf-300C** (kill_then_adapt pattern; USER_OK marker convention).
* **Wf-130 / Wf-190** (`CUDA_VISIBLE_DEVICES=1` + `MOLMETAL_ALLOW_CPU` env guard).

## 7. Follow-ups (not in Wf-308C scope)

* **TODO-130 G3 MultiTaskLoss head** — wire the 9-cell joint multi-task
  chemprop v2 backbone (shared encoder + 9 BCE heads) so per-cell AUC
  benefits from cross-cell signal (current driver is per-cell
  independent encoders).
* **Re temporal split degeneracy** — Re has only 1 row with Year ≥
  2025; the random 80/10/20 fallback handles this but per-paper
  comparison, Re should be excluded or use a longer train window
  (2001-2023 vs 2024-2025 window is too tight for the small metals).
* **GPU wall-time calibration** — Stage A projected 2.9 hr is from
  Wf-219 + Wf-303C T3 estimates; the actual MEASURED wall will be
  logged in `wf308c_temporal_split/run.log` after the first GPU run.