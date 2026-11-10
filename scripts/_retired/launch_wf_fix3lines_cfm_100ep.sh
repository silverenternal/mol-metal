#!/bin/bash
# Wf-fix3lines Wave 3: CFM bond head 100-epoch GPU run.
#
# Builds on BLK-C1 (20-ep) which verified BLK-I1 launcher fix on CFM and confirmed
# mode collapse at 20-ep (best decode_ratio 0.135). Per Wf-124 P2 synthesis.md honest
# framing: "the full 100-epoch retrain would lift further". This 100-ep run is to
# confirm whether the 22x bond_f1 lift (0.036 → 0.803) co-occurs with a decode_ratio
# lift past the 0.8 gate, or whether 20-ep mode collapse is structural.
#
# Two corrections vs the brief, both verified against disk first:
#
# 1. NO --out-metric. `grep -n "out.metric" molmetal/scripts/train_cfm_
#    yuelbond_wasserstein.py` returns nothing, and passing the flag makes
#    argparse exit 2 with "unrecognized arguments". This trainer writes
#    metrics.json into --out-dir itself; it has no --out-metric dest.
#
# 2. --tmqm-dir IS passed. Without it the trainer falls to
#    build_synthetic_pt_click_clouds (one jittered square-planar Pt layout,
#    same 5 atoms every run). MEASURED on a CPU probe: the synthetic path
#    emits `dataset=synthetic` + the cfm-bond-target WARNING that targets are
#    SYNTHESISED, and scores bond_n_scored=0 / bond_f1=0.0 — a perfect
#    decode_ratio there means nothing. With --tmqm-dir
#    .data/molmetal/tmQM (324 MB, has tmQM_X*.BO.gz Wiberg blocks) the same
#    probe reports dataset=tmqm and bond_n_scored=24 / bond_f1=0.477. This is
#    the whole point of a bond-head run, so the corpus flag is not optional.
#
# 3. --early-stop-patience 999. MEASURED on the first launch (43 s wall):
#    the trainer defaulted to --early-stop-metric bond_exact_type with
#    patience 3 and stopped at epoch 4 ("best bond_exact_type=0.0052 @
#    epoch 1"), so a 20-epoch run never happened. The flag is raised so the
#    requested 20 epochs actually execute; the gate is still recorded in the
#    done-event.
#
# AGENTS.md section 5.9 single-venv: .venv/bin/python only.
# No `set -o pipefail` (see the SIGPIPE lesson — DataLoader workers flush
# after trainer exit and kill the wrapper silently). No `| tee` here at all;
# the harness captures the background log.
# AGENTS.md section 5.6 atomic write + BLK-I1 2026-10-09: setsid+nohup + redirect
# to logs/<script>.log so the trainer survives parent shell exit (per
# Wf-FIX-3LINES CFM bond head epoch 1/20 ABORTED root cause, commit 6d1c796d89).
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

OUT_DIR=molmetal/reports/cfm_bond_head_100ep_2026_10_09
LOG=logs/wf_fix3lines_cfm_100ep.log
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect the entire .sh stdout/stderr to LOG; trainer detached
exec >"$LOG" 2>&1

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# BLK-I1: detach trainer from parent process group + background + wait for rc
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no cli参数"): every
# hyperparameter below moved into configs/cfm_bond_head_100ep_2026_10_09.yaml as a faithful transcription
# of what this command line used to pass. The wrapper is now env + detach +
# redirect only. Note that cfm_bond_head_100ep_2026_10_09.yaml records use_fm_objective: false — the
# original command never set it, so the velocity field had no flow-matching
# target; see configs/cfm_blk_c5_repair_2026_10_09.yaml for the measured
# gradient census behind that note.
setsid nohup /home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --config configs/cfm_bond_head_100ep_2026_10_09.yaml \
  --out-dir "$OUT_DIR" \
  >/dev/null 2>&1 &
PID=$!
wait $PID
rc=$?

echo "[wf_fix3lines_cfm] trainer rc=$rc"
exit $rc
