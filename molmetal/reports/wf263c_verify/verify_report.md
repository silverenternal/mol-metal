# Wf-263C Verify Report

**Date:** 2026-09-28 (post-Wf-258 user OK)
**Owner:** Hugo Lee
**Task:** Wf-263C-Verify — Run --dry-run mode on both wrappers + verify configs parse.

---

## Acceptance Gates (FLAT schema)

| Gate | Status | Notes |
|------|--------|-------|
| **G1_g5_syntax_ok** | PASS | `bash -n scripts/launch_wf124_p3_g5.sh` exit=0 |
| **G2_g6_syntax_ok** | PASS | `bash -n scripts/launch_wf124_p3_g6.sh` exit=0 |
| **G3_dry_run_works** | PASS | Both wrappers' `--dry-run` modes succeed; CSV + config paths verified |
| **G4_pre_flight_25_25** | PASS | 25/25 gates PASS, 0 FAIL, 0 SKIP (wall 68.4s) |
| **G5_verify_report_path** | PASS | This file: `molmetal/reports/wf263c_verify/verify_report.md` |

---

## Step 1: bash syntax check

```
$ bash -n /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh
G5_SYNTAX_OK  (exit=0)

$ bash -n /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh
G6_SYNTAX_OK  (exit=0)
```

Both wrappers pass `bash -n`. No syntax errors.

---

## Step 2: --dry-run mode (G5 + G6)

### G5 dry-run (`scripts/launch_wf124_p3_g5.sh --dry-run`)

- **OUT_BASE:** `results/wf124_p3_50epoch_full_sweep_g5`
- **PY:** `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`
- **SCRIPT:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- **DATA_CSV:** `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26633 rows, 11 cols, OK)
- **GPU:** `cuda:1`
- **COMMON_ARGS:** `--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline --w2-gradient-mode sliced --n-mc 256 --device cuda:1 --cuda-visible-devices 1 --cpu-num-workers 0 --data-csv <path>`
- **15 (metal, seed) pairs PLANNED:** Ru/Ir/Os/Rh/Re × {0,1,2}
- **Exit:** 0 (DRY-RUN OK, no GPU work executed)
- Full log: `molmetal/reports/wf263c_verify/g5_dry_run.log`

### G6 dry-run (`scripts/launch_wf124_p3_g6.sh --dry-run`)

- **OUT_BASE:** `results/wf124_p3_per_innovation_ablation_g6`
- **PY:** `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`
- **SCRIPT:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- **DATA_CSV:** `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26633 rows, OK)
- **GPU:** `cuda:1`
- **COMMON_ARGS:** `--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --w2-gradient-mode sliced --n-mc 256 --use-l2-guard --device cuda:1 --cuda-visible-devices 1 --cpu-num-workers 0 --data-csv <path>`
- **CONFIRMED configs (real launch eligible):** `no_multi_task`, `no_wass` (30 runs)
- **UNVERIFIED configs (skip-if-real):** `no_kendall`, `no_dann` (30 runs skipped pending flag verification)
- **60 (config, metal, seed) tuples PLANNED** (30 confirmed + 30 unverified-skip)
- **Exit:** 0 (DRY-RUN OK, no GPU work executed)
- Full log: `molmetal/reports/wf263c_verify/g6_dry_run.log`

### Config YAML parse check

```
OK: configs/wf124_p3_50epoch_full_sweep_g5.yaml
    (top-level keys: ['dry_run', 'user_ok_required', 'data', 'model', 'protocol'])
OK: configs/wf124_p3_per_innovation_ablation_g6.yaml
    (top-level keys: ['dry_run', 'user_ok_required', 'data', 'model', 'protocol'])
    entry.pre_launch_checklist:
      - 'Verify --no-kendall flag exists in train_dmpnn_multitask_wasserstein.py'
      - 'Verify --no-dann flag exists or fallback to --dann-domain-mode=off'
      - 'Confirm --single-task replaces --no-multi-task (already confirmed)'
```

All wf124 configs parse OK.

---

## Step 3: pre_flight_check.py 25/25 PASS

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py \
    --pipeline all --json molmetal/reports/wf263c_verify/pre_flight.json

summary: {pass: 25, fail: 0, skip: 0, total: 25, total_wall_sec: 68.3842}
schema_version: 207.v1
v2_doc_reference: TODO/completed/207_unified_pre_flight_checklist_v2.md
```

### Per-gate breakdown

| # | gate | pipeline | status | wall_sec |
|---|------|----------|--------|----------|
| 1 | p1_chemprop_indices_fix | chemprop_v2_v3 | PASS | 0.0002 |
| 2 | p1_chemprop_kendall_optimizer | chemprop_v2_v3 | PASS | 0.0001 |
| 3 | p1_chemprop_dann_per_metal | chemprop_v2_v3 | PASS | 0.0001 |
| 4 | p1_chemprop_nan_guard | chemprop_v2_v3 | PASS | 0.0001 |
| 5 | p1_chemprop_grad_clip | chemprop_v2_v3 | PASS | 0.0001 |
| 6 | p1_chemprop_epoch_avg | chemprop_v2_v3 | PASS | 0.0001 |
| 7 | p1_chemprop_ablation_flags | chemprop_v2_v3 | PASS | 4.9248 |
| 8 | p1_chemprop_seeds_min | chemprop_v2_v3 | PASS | 4.9174 |
| 9 | p1_chemprop_master_wass_flag | chemprop_v2_v3 | PASS | 4.8854 |
| 10 | p1_chemprop_master_cath_flag | chemprop_v2_v3 | PASS | 4.8854 |
| 11 | p1_chemprop_click_aug_flag | chemprop_v2_v3 | PASS | 4.9774 |
| 12 | p1_chemprop_smoke | chemprop_v2_v3 | PASS | 36.9381 |
| 13 | p1_cfm_script_exists | cfm | PASS | 0.0001 |
| 14 | p1_cfm_config_exists | cfm | PASS | 0.0 |
| 15 | p1_cfm_decoder_exists | cfm | PASS | 0.0 |
| 16 | p1_cfm_wasserstein_loss_exists | cfm | PASS | 0.0 |
| 17 | p1_cfm_smoke_decoded | cfm | PASS | 0.0001 |
| 18 | p1_cfm_catherine_30metals | cfm | PASS | 0.0003 |
| 19 | p2_data_metalcytotox_exists | data | PASS | 0.0033 |
| 20 | p2_data_metallo_drugs_exists | data | PASS | 0.0001 |
| 21 | p2_tests_pass | tests | PASS | 6.3644 |
| 22 | p2_vina_installed | data | PASS | 0.0002 |
| 23 | p0_config_schema_verify | config | PASS | 0.0527 |
| 24 | p0_csv_hash_verify | data | PASS | 0.0137 |
| 25 | p0_scaffold_zero_leak | data | PASS | 0.4199 |

**Backward compat verified:** `p2_tests_pass` PASS = pytest 24 passed, 1 warning in 3.54s.

Full JSON: `molmetal/reports/wf263c_verify/pre_flight.json`

---

## Honest Framing Note

- G5/G6 dry-run modes verify paths + flags ONLY; no GPU work executed in Wf-263C.
- G6 has 30 of 60 tuples flagged "UNVERIFIED-skip-if-real" (no_kendall + no_dann). These tuples require a `--no-kendall` / `--no-dann` CLI flag verification step BEFORE any real launch — per the g6 yaml `entry.pre_launch_checklist`.
- Real launch of either wrapper is BLOCKED on user OK + GPU availability + no competing workflow contention (per TODO-181/182 §3 + Wf-241 + Wf-260 timing).
- This verify step does NOT measure or report per-lever attribution; that requires the actual 60-run sweep (TODO-182 §4).
- Path used: `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (note: the task spec referenced `cades` — that is a typo; the actual path is `codes`). The wrappers themselves use the correct path.

---

## Files Generated

- `molmetal/reports/wf263c_verify/verify_report.md` (this file)
- `molmetal/reports/wf263c_verify/pre_flight.json` (full pre-flight JSON)
- `molmetal/reports/wf263c_verify/pre_flight_stdout.log` (pre-flight stdout)
- `molmetal/reports/wf263c_verify/g5_dry_run.log` (g5 dry-run output)
- `molmetal/reports/wf263c_verify/g6_dry_run.log` (g6 dry-run output)
- `molmetal/reports/wf263c_verify/yaml_parse.log` (YAML config parse check)