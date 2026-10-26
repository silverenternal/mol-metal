#!/bin/bash
# Wf-310A — Retry 2 missing G5 pairs (Ru seed=2, Ir seed=0)
# Background launcher. Each pair runs on a different GPU for parallel exec.
# Both pairs were killed mid-training by the kill_then_adapt scheduler
# during the original G5 sweep (TODO-181) on 2026-09-29 03:11:13 CST.
#
# GPU mapping:
#   - Ru seed=2 → CUDA_VISIBLE_DEVICES=1, --device cuda:0 (RTX 5090 — heavy Ru cohort)
#   - Ir seed=0 → CUDA_VISIBLE_DEVICES=0, --device cuda:0 (RTX PRO 6000 — medium Ir cohort)
#
# Wall-time estimate:
#   - Ru seed=2: ~30 min for 50 epochs on RTX 5090
#   - Ir seed=0: ~30 min for 50 epochs on RTX PRO 6000
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Usage:
#   bash scripts/retry_g5_missing_pairs.sh         # sequential (default)
#   bash scripts/retry_g5_missing_pairs.sh --parallel   # parallel on 2 GPUs

set -e
cd /home/hugo/codes/mol-metal

OUT_BASE="results/wf124_p3_50epoch_full_sweep_g5"
LOG="${OUT_BASE}/master.log"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
# The trainer already accepts --config and ends parse_args() in
# apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
# yaml is the base and the per-call flags below override it.
# Precedence: explicit CLI flag > YAML > argparse default.
DMPNN_CONFIG="${DMPNN_CONFIG:-configs/dmpnn.yaml}"
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

COMMON_ARGS=(
  --epochs 50
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --ablation-config mtl_baseline
  --w2-gradient-mode sliced
  --n-mc 256
  --cpu-num-workers 8
  --data-csv "${DATA_CSV}"
  # Wf-335-R3's `--no-cuda-graphs` was REMOVED here on 2026-10-27: the trainer's
  # argparse does not declare it, so this retry wrapper — the one path that
  # exists to recover the Wf-330-B4 failures — died rc=2 before launching
  # anything. Graphs only exist behind `--torch-compile`, which is not passed.
)

run_pair() {
  local metal=$1
  local seed=$2
  local gpu=$3
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"
  echo "=== $(date '+%H:%M:%S') Wf-310A RETRY START ${metal} seed=${seed} on GPU ${gpu} ===" | tee -a "${LOG}"
  # `setsid --wait`: the retry runs in its own session so it survives the
  # invoking shell (BLK-I1a), and `$rc` below still reports the trainer's
  # status rather than setsid's. Bare `setsid` may fork and return 0.
  CUDA_VISIBLE_DEVICES=${gpu} setsid --wait ${PY} ${SCRIPT} \
    --config "${DMPNN_CONFIG}" \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
    --device cuda:0 \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}
  echo "=== $(date '+%H:%M:%S') Wf-310A RETRY END ${metal} seed=${seed} (rc=${rc}) on GPU ${gpu} ===" | tee -a "${LOG}"
  return "$rc"
}

PARALLEL="${1:-}"

# ---- Wf-356-A1: wf343c3 data quality check (loader-agnostic — all 5 R20
#      loaders internally: load_binding_db, load_nci60, load_drugood,
#      metalcytotox_v4, platinai_4cell). Does NOT accept --data-csv; only
#      --out-dir. Use launcher-specific out-dir so each launcher's report
#      is preserved. Inserted as the first wave-tool guard before any
#      run_pair() call per Wf-347-B3 retry. ----
echo "=== wf343c3 data quality check START $(date) ===" | tee -a "${LOG}"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/retry_g5_missing_pairs_data_quality || {
  echo "FAIL: wf343c3_data_quality_check"
  exit 1
}
echo "=== wf343c3 data quality check PASS $(date) ===" | tee -a "${LOG}"

if [[ "${PARALLEL}" == "--parallel" ]]; then
  echo "=== Wf-310A PARALLEL retry on 2 GPUs ===" | tee -a "${LOG}"
  run_pair "Ru" "2" "1" &
  PID_RU=$!
  run_pair "Ir" "0" "0" &
  PID_IR=$!
  wait "${PID_RU}" || echo "Ru seed=2 failed (PID ${PID_RU})"
  wait "${PID_IR}" || echo "Ir seed=0 failed (PID ${PID_IR})"
  echo "=== Wf-310A PARALLEL retry DONE ===" | tee -a "${LOG}"
else
  echo "=== Wf-310A SEQUENTIAL retry ===" | tee -a "${LOG}"
  # Heavy Ru first, then Ir
  run_pair "Ru" "2" "1"
  run_pair "Ir" "0" "1"
  echo "=== Wf-310A SEQUENTIAL retry DONE ===" | tee -a "${LOG}"
fi
