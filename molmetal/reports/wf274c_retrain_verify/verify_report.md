# Wf-274C: In-Flight Retrain v3-Fallback Verification Report

**Date**: 2026-09-29
**Wf**: Wf-274C (verify in-flight retrain uses v3-fallback per Wf-271B finding)
**Owner**: subagent (Claude Code)
**Method**: Bash read-only inspection of launch scripts + active PIDs + log files
**Scope**: G5 (PID 609543), G4 launch wrapper, Wf-273A background tasks, Wf-269C CFM ablation

---

## §0 TL;DR — Verdict Matrix (FLAT schema)

| Slot | Status | Script used | --chemprop-mode flag | Verdict |
|------|--------|-------------|---------------------|---------|
| **G5** (PID 609543) | IN-FLIGHT (epoch 33/50, val_auc 0.7556) | `train_dmpnn_multitask_wasserstein.py` (D-MPNN multitask W₂) | NOT APPLICABLE — flag does not exist in this script | **SAFE — by-virtue-of-unrelated** |
| **G4** launch wrapper | ALREADY COMPLETED (22:23:21 Sep 28) | `train_dmpnn_multitask_wasserstein.py` (same as G5) | NOT APPLICABLE — flag does not exist in this script | **SAFE — already finished, no in-flight exposure** |
| **Wf-273A** background bash task | **DOES NOT EXIST** — only Wf-273C (mode-matrix report) + Wf-273D (file-audit report) are present; no Wf-273A launch wrapper or background bash invocation found | n/a | n/a | **NOT APPLICABLE — no such in-flight task** |
| **Wf-269C** CFM ablation script | CPU-only test; no in-flight retrain | `wf269c_cfm_real_ablation.py` (CFM, NOT chemprop) | NOT APPLICABLE — script contains ZERO `chemprop` references | **NOT APPLICABLE — CFM, not chemprop** |

**Overall verdict**: **SAFE** — none of the in-flight retrain slots are exposed to the Wf-271B
`--chemprop-mode v3` risk vector. Both G4 and G5 use the D-MPNN multitask Wasserstein trainer,
which has no `--chemprop-mode` flag at all. Wf-273A does not exist; Wf-269C is CFM-only.

---

## §1 Evidence (per-slot)

### §1.1 G5 — PID 609543 (Ru seed=0, epoch 33/50 val_auc 0.7556)

**`/proc/609543/cmdline` (NUL-delimited, space-joined):**

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python
molmetal/scripts/train_dmpnn_multitask_wasserstein.py
--metals Ru --seeds 0
--csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
--data-csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
--out-dir results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0
--ckpt-dir results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ckpts
--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5
--wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline
--w2-gradient-mode sliced --n-mc 256 --device cuda:0
--cuda-visible-devices 1 --cpu-num-workers 0
--data-csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
```

- **No `--chemprop-mode` flag present.**
- **No `--enhanced-head` flag present.**
- **No `--metal-attention` flag present.**
- Script is `train_dmpnn_multitask_wasserstein.py` (D-MPNN multitask Wasserstein), NOT chemprop.

**Launch script `scripts/launch_wf124_p3_g5.sh`:**
- `COMMON_ARGS` (lines 38-52): `--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3
   --alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline
   --w2-gradient-mode sliced --n-mc 256 --device cuda:0 --cuda-visible-devices 1
   --cpu-num-workers 0 --data-csv ${DATA_CSV}`. **No chemprop-mode flag.**

**Conclusion**: G5 is a D-MPNN multitask W₂ run, not a chemprop run. The Wf-271B
`--chemprop-mode v3` regression does not apply.

---

### §1.2 G4 — launch wrapper `scripts/launch_wf124_p3_g4.sh`

**G4 status**: **Already COMPLETED** — `results/wf124_p3_remaining_metals_g4/master.log`
ends with `=== Wave B DONE Mon Sep 28 10:23:21 PM CST 2026 === === ALL TODO-180 G4 PAIRS DONE
Mon Sep 28 10:23:21 PM CST 2026 ===`. All 14 pairs (Ru s1/s2 + Ir s0/s1/s2 + Os s0/s1/s2 +
Rh s0/s1/s2 + Re s0/s1/s2) finished before the Wf-271B finding was even raised.

**Launch script `scripts/launch_wf124_p3_g4.sh`:**
- `COMMON_ARGS` (lines 38-54): `--epochs 30 --batch-size 256 --lr 1e-3 --alpha 0.5
   --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline
   --w2-gradient-mode sliced --n-mc 256 --hidden 300 --depth 3 --device cuda:0
   --cuda-visible-devices 1 --cpu-num-workers 0 --data-csv ${DATA_CSV}`.
   **No chemprop-mode flag.**

- Script used is `train_dmpnn_multitask_wasserstein.py` (same as G5). D-MPNN, not chemprop.

**Conclusion**: G4 had no in-flight exposure to the Wf-271B chemprop-mode risk vector. All
14 pairs were D-MPNN multitask W₂, all completed before the Wf-271B finding was raised.

---

### §1.3 Wf-273A background bash task — **DOES NOT EXIST**

**Search results:**
- `grep -rn "wf273a\|wf-273a\|Wf-273A"` against `scripts/` + `TODO/`: **0 hits**.
- `ls /tmp/wf273a_*.log`: **no matches** (zsh: `(eval):1: no matches found`).
- `ls molmetal/reports/ | grep wf273`: only `wf273c_mode_matrix/` (read-only mode-matrix
  report) + `wf273d_file_audit/` (read-only file-audit report) exist. **No `wf273a_*` directory**.
- `ps -ef | grep -E "wf273a|chemprop_v2_v3"`:
  - PID 641583: `train_chemprop_v2_v3.py --help` (just an argparse `--help` probe, no training).
  - No Wf-273A process or background bash task found.

**Conclusion**: The task brief's premise — "Wf-273A G4 pairs: launched as background bash task"
— does not match the on-disk state. There is no Wf-273A launch wrapper, no background bash
invocation, and no `wf273a_*.log` artifacts. Only Wf-273C (mode-matrix report) and Wf-273D
(file-audit report) exist as completed read-only reports.

---

### §1.4 Wf-269C CFM ablation script — NOT chemprop

**Script path**: `molmetal/scripts/wf269c_cfm_real_ablation.py`.

**Search**: `grep -rn "chemprop" molmetal/scripts/wf269c_cfm_real_ablation.py` → **0 hits**.
Zero references to chemprop, chemprop-mode, BinaryClassificationFFN, or any chemprop concept.

**Script nature**:
- CFM (Conditional Flow Matching) ablation — YuelBond + Wasserstein + closure-bounded innovations.
- `train_cfm_yuelbond_wasserstein.CFMJointModel` + `YuelBondBondDecoder` + `WassersteinConfig`.
- CPU-only (`--device cpu`).
- Smoke-fast (hidden_dim=32, n_layers=2, 5 epochs per config).

**Conclusion**: Wf-269C is a CFM ablation, not a chemprop run. The Wf-271B `--chemprop-mode v3`
risk vector does not apply to Wf-269C.

---

## §2 Wf-271B Default in `train_chemprop_v2_v3.py`

For completeness — the `train_chemprop_v2_v3.py` script default:

```
molmetal/scripts/train_chemprop_v2_v3.py:1523-1525
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder"),
        default="v3-fallback",
        ...
```

Default is `v3-fallback` (forces chemprop v2 `BinaryClassificationFFN` regardless of
`--enhanced-head` / `--metal-attention`; matches Wf-185 paper-grade baseline). The Wf-271B
finding is that any retrain launched against this script WITHOUT an explicit `--chemprop-mode`
override automatically gets `v3-fallback` — the safe mode.

**No currently-running process** uses `train_chemprop_v2_v3.py` (PID 641583 is just `--help`).

---

## §3 Verdict

**In-flight retrain is SAFE from the Wf-271B chemprop-mode risk vector.**

| Slot | Status | Conclusion |
|------|--------|-----------|
| G5 | IN-FLIGHT (Ru seed=0, epoch 33/50 val_auc 0.7556) | SAFE — D-MPNN, not chemprop; no `--chemprop-mode` flag |
| G4 | COMPLETED (22:23:21 Sep 28) | SAFE — already finished; was D-MPNN; no exposure |
| Wf-273A | DOES NOT EXIST | NOT APPLICABLE |
| Wf-269C | CFM ablation, no retrain | NOT APPLICABLE |

No retrain slot requires chemprop-mode-override intervention. No RISK exposure detected.

---

## §4 Audit Trail

| Step | Command | Result |
|------|---------|--------|
| 1 | `cat /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` | read; no `--chemprop-mode` flag in `COMMON_ARGS` |
| 2 | `cat /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh` | read; no `--chemprop-mode` flag in `COMMON_ARGS` |
| 3 | `cat /proc/609543/cmdline | tr '\0' ' '` | read; `train_dmpnn_multitask_wasserstein.py`, no chemprop flags |
| 4 | `grep -n "chemprop_mode" train_dmpnn_multitask_wasserstein.py` | empty; D-MPNN script has no `--chemprop-mode` flag |
| 5 | `grep -rn "wf273a\|wf-273a\|Wf-273A" scripts/ TODO/` | empty; no Wf-273A in codebase |
| 6 | `ls /tmp/wf273a_*.log` | no matches |
| 7 | `ls molmetal/reports/ | grep wf273` | only `wf273c_mode_matrix/` + `wf273d_file_audit/` |
| 8 | `ps -ef | grep -E "wf273a|chemprop_v2_v3"` | PID 641583 = `train_chemprop_v2_v3.py --help` only |
| 9 | `grep -rn "chemprop" molmetal/scripts/wf269c_cfm_real_ablation.py` | empty; pure CFM, zero chemprop refs |
| 10 | `cat results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/run.log \| tail -30` | epoch 33/50 val_auc 0.7556, training healthy |
| 11 | `cat results/wf124_p3_remaining_metals_g4/master.log` | ends at 22:23:21 Sep 28; all 14 pairs DONE |

---

**End of Wf-274C verify report.**
