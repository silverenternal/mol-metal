#!/bin/bash
# Wf-BLK-D1 Step 4: bace 5-seed retest on AttentiveFP backbone (cuda:0).
#
# Rationale: the BLK-D1 fix_steps Step 2-3 wire-in is SHIPPED (per
# TODO/completed/blk_d1_step_2a_attentivefp_2026_10_09.md); the 5-seed
# 50-epoch retest on the new backbone is the lift measurement that closes
# the BLK-D1 loop. Comparison floor is the dmpnn 5-seed baseline
# (mean=0.7505 ± 0.0456, on disk at
# molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/metrics.json).
#
# Per AGENTS.md §5.1 standing GPU auth (no per-launch user OK); §5.4 no
# wet-lab; §5.5 filesystem-truth (--out-metric must point at a fresh
# per-run metrics.json, NEVER at metrics/by_metric/m* — per TODO-181
# guard); §5.6 atomic write + BLK-I1 setsid+nohup; §5.7 §5.7 wire-in
# smoke ALREADY PASSED in BLK-D1 Step 3; §5.9 single-venv (.venv/bin/python).
#
# No pipefail (per Wf-FIX-3LINES 2026-10-08 pipefix lesson: tee flushing
# DataLoader workers after trainer exit silently dies under pipefail).
set -e

# AGENTS.md §5.9 single-venv. Absolute, not `.venv/bin/python`: the relative form
# only resolves when cwd happens to be the repo root, so the same wrapper run
# from anywhere else silently picks up the wrong (or no) interpreter.
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# BLK-I2 (2026-10-09): pre-flight gate per AGENTS.md §1.4.
# Per `todo.json` BLK-I2 fix_steps: every trainer-launching
# scripts/launch_*.sh runs `pre_flight_check.py` first; rc != 0
# aborts before the trainer is invoked. Captures stdout to
# logs/preflight_<script>_<ts>.log so the gate evidence is on disk.
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

OUT_DIR=molmetal/reports/blk_d1_attentivefp_5seed_2026_10_09
LOG=logs/blk_d1_attentivefp_5seed_2026_10_09.log
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect the entire .sh stdout/stderr to LOG so wrapper output
# survives parent-shell exit; trainer is also detached via setsid+nohup so
# its process group outlives the parent bash.
exec >"$LOG" 2>&1

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# BLK-I1: detach trainer from parent process group + background + wait for rc
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no cli参数"): every
# hyperparameter below moved into configs/blk_d1_attentivefp_5seed_2026_10_09.yaml as a faithful transcription
# of what this command line used to pass. `--config` support was added to
# train_moleculenet.py on the same date — it had no YAML surface before, which is why these
# nine-to-eleven flags could only be set here.
setsid nohup "$PY" molmetal/scripts/train_moleculenet.py \
  --config configs/blk_d1_attentivefp_5seed_2026_10_09.yaml \
  --out-dir "$OUT_DIR" \
  >/dev/null 2>&1 &
PID=$!
wait $PID
rc=$?

echo "[blk_d1_attentivefp_5seed] trainer rc=$rc"
exit $rc
