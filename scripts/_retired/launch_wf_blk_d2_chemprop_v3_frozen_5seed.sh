#!/bin/bash
# Wf-BLK-D2 Step 2 5-seed retest: chemprop_v3 frozen encoder + trainable MLP head
# on bace, CPU-only (chemprop_v3 frozen encoder is fast enough on CPU; keeps GPU
# 0 free for the parallel BLK-D1 AttentiveFP 5-seed run).
#
# Rationale: BLK-D2 Step 2 wire-in is SHIPPED (per
# TODO/completed/blk_d2_step_2_chemprop_v3_frozen_head_2026_10_09.md);
# the 1-epoch §5.7 smoke at bace/seed0 reached roc_auc=0.4997 (random
# floor — expected for 1 epoch on 1060 train rows with a frozen
# encoder). The 5-seed 50-epoch retest is the lift measurement that
# closes the BLK-D2 Step 2 loop. Comparison floor is the dmpnn 5-seed
# baseline (mean=0.7505 ± 0.0456).
#
# Per AGENTS.md §5.1 (CPU-only, no GPU auth needed) + §5.4 no wet-lab +
# §5.5 filesystem-truth + §5.6 atomic write + BLK-I1 setsid+nohup +
# §5.7 wire-in smoke ALREADY PASSED + §5.9 single-venv (.venv/bin/python).
#
# No pipefail (per Wf-FIX-3LINES 2026-10-08 pipefix lesson).
set -e

# AGENTS.md §5.9 single-venv. Absolute, not `.venv/bin/python`: the relative form
# only resolves when cwd happens to be the repo root.
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# BLK-I2 (2026-10-09): pre-flight gate per AGENTS.md §1.4.
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_LOG="$LOG_DIR/preflight_$(basename "$0" .sh)_$TS.log"
if ! PYTHONPATH=/home/hugo/codes/mol-metal "$PY"     molmetal/scripts/pre_flight_check.py     --pipeline launch     >"$PREFLIGHT_LOG" 2>&1; then
  echo "FAIL: pre_flight_check (BLK-I2 gate); see $PREFLIGHT_LOG" >&2
  exit 1
fi
echo "[BLK-I2] pre_flight_check PASS (log: $PREFLIGHT_LOG)"

cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/blk_d2_chemprop_v3_frozen_5seed_2026_10_09
LOG=logs/blk_d2_chemprop_v3_frozen_5seed_2026_10_09.log
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect the entire .sh stdout/stderr to LOG so wrapper output
# survives parent-shell exit; trainer is also detached via setsid+nohup so
# its process group outlives the parent bash.
exec >"$LOG" 2>&1

# CPU-only: keep GPU 0 free for the parallel BLK-D1 AttentiveFP 5-seed run.
# chemprop_v3 frozen encoder + 32-unit MLP head is fast enough on CPU at
# batch-size 32.
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# BLK-I1: detach trainer from parent process group + background + wait for rc
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no cli参数"): every
# hyperparameter below moved into configs/blk_d2_chemprop_v3_frozen_5seed_2026_10_09.yaml as a faithful transcription
# of what this command line used to pass. `--config` support was added to
# train_chemprop_v3_frozen_dmpnn_head.py on the same date — it had no YAML surface before, which is why these
# nine-to-eleven flags could only be set here.
setsid nohup "$PY" molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py \
  --config configs/blk_d2_chemprop_v3_frozen_5seed_2026_10_09.yaml \
  --out-dir "$OUT_DIR" \
  >/dev/null 2>&1 &
PID=$!
wait $PID
rc=$?

echo "[blk_d2_chemprop_v3_frozen_5seed] trainer rc=$rc"
exit $rc
