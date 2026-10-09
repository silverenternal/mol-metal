#!/bin/bash
# Wf-fix3lines bace 5-seed GPU re-run.
# Rationale: the 2026-10-08 bace rescue ran on CPU while the baseline ran on GPU;
# ALIGNED_COMPARISON.md states "a clean GPU re-run with a same-device control is
# the confirmation still owed". This run is that confirmation on the canonical
# baseline protocol (class-weight on, no regularisation, hidden 300 / depth 3 /
# batch 64 / lr 1e-3) so the number is same-protocol comparable to
# metrics/by_metric/moleculenet_auc_dmpnn_5seed.json.
#
# AGENTS.md section 5.9 single-venv: .venv/bin/python only. No pipefail (SIGPIPE lesson).
# AGENTS.md section 5.6 atomic write + BLK-I1 2026-10-09: setsid+nohup + redirect
# to logs/<script>.log so the trainer survives parent shell exit (per
# Wf-FIX-3LINES 2-of-3 ABORTED root cause, commit 6d1c796d89).
set -e

# BLK-I2 (2026-10-09): pre-flight gate per AGENTS.md §1.4.
# Per `todo.json` BLK-I2 fix_steps: every trainer-launching
# scripts/launch_*.sh runs `pre_flight_check.py` first; rc != 0
# aborts before the trainer is invoked. Captures stdout to
# logs/preflight_<script>_<ts>.log so the gate evidence is on disk.
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_LOG="$LOG_DIR/preflight_$(basename "$0" .sh)_$TS.log"
if ! PYTHONPATH=/home/hugo/codes/mol-metal PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"     molmetal/scripts/pre_flight_check.py     --pipeline launch     >"$PREFLIGHT_LOG" 2>&1; then
  echo "FAIL: pre_flight_check (BLK-I2 gate); see $PREFLIGHT_LOG" >&2
  exit 1
fi
echo "[BLK-I2] pre_flight_check PASS (log: $PREFLIGHT_LOG)"

cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09
LOG=logs/wf_fix3lines_bace_50ep.log
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect the entire .sh stdout/stderr to LOG so wrapper output
# survives parent-shell exit; trainer is also detached via setsid+nohup so
# its process group outlives the parent bash.
exec >"$LOG" 2>&1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# BLK-I1: detach trainer from parent process group + background + wait for rc
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no cli参数"): every
# hyperparameter below moved into configs/wf_fix3lines_bace_50ep_2026_10_09.yaml as a faithful transcription
# of what this command line used to pass. `--config` support was added to
# train_moleculenet.py on the same date — it had no YAML surface before, which is why these
# nine-to-eleven flags could only be set here.
setsid nohup /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/train_moleculenet.py \
  --config configs/wf_fix3lines_bace_50ep_2026_10_09.yaml \
  --out-dir "$OUT_DIR" \
  >/dev/null 2>&1 &
PID=$!
wait $PID
rc=$?

echo "[wf_fix3lines_bace] trainer rc=$rc"
exit $rc
