#!/usr/bin/env bash
# 2026-10-05 — CFM re-train on tmQM all6 (after Wf-432 centering fix).
#
# Why
# ---
# The 2026-10-04 Wf-432 audit found tmQM coordinates were never centered
# before being fed to the EGNN velocity field: per-complex centroid ~11 A
# from origin.  The fix (EDM zero-CoM centering in tmqm_clouds.py) is in,
# but the epoch-9 checkpoint was trained on uncentered data, so it is
# now miscalibrated for the new target distribution.  Re-training is the
# only way to recover generation quality.
#
# Scope
# -----
#   - configs/cfm_tmqm_all6.yaml (the canonical all6 config)
#   - 20 epochs, default seed
#   - GPU 0 only (RTX PRO 6000 Blackwell, 98 GB VRAM, only 25 GB used)
#   - One output dir; one metrics JSON
#   - Wall-time target: ~4-5 hr
#
# Honest framing
# --------------
# This is a CLEAN retrain with the centering fix in.  Prior measurements
# (gen_validity 0/128, etc.) reflect the OLD pipeline.  If the new run
# still has gen_validity near 0, the centering fix is not the root cause
# and we will need to find another.  Both outcomes are MEASURED and
# honest-reported.
#
# Usage
# -----
#   bash scripts/launch_cfm_tmqm_all6_retrain.sh --dry-run
#   bash scripts/launch_cfm_tmqm_all6_retrain.sh
#
# Constraint: AGENTS.md §5.1 (standing GPU authorization) + §5.7
# (50-step smoke already done for any flag changes; the trainer has no
# flag changes — pure re-train on the same code path).

# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# 2026-10-08 pipefix: drop pipefail (SIGPIPE from forked DataLoader workers
# flushing stdout after trainer exit killed the wrapper silently via `tee`).
# Capture trainer rc via ${PIPESTATUS[0]}; suppress tee SIGPIPE with `|| true`.
#
# Wf-A269 wave-10 (2026-11-02): the `|| true` half above is what made the rc
# read DEAD — `|| true` runs AFTER the failing pipeline and CLOBBERS
# PIPESTATUS to (0), so the "trainer exited rc != 0 -> exit" branch could never
# fire and this wrapper exited 0 after a crashed trainer.  The repaired shape
# is at the launch site below (scripts/_WRAPPER_TEMPLATE.sh:190-193).

# BLK-I2 (2026-10-09): pre-flight gate per AGENTS.md §1.4.
# Per `todo.json` BLK-I2 fix_steps: every trainer-launching
# scripts/launch_*.sh runs `pre_flight_check.py` first; rc != 0
# aborts before the trainer is invoked. Captures stdout to
# logs/preflight_<script>_<ts>.log so the gate evidence is on disk.
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_LOG="$LOG_DIR/preflight_$(basename "$0" .sh)_$TS.log"
if ! PYTHONPATH=/home/hugo/codes/mol-metal ${PY:-.venv/bin/python}     molmetal/scripts/pre_flight_check.py     --pipeline launch     >"$PREFLIGHT_LOG" 2>&1; then
  echo "FAIL: pre_flight_check (BLK-I2 gate); see $PREFLIGHT_LOG" >&2
  exit 1
fi
echo "[BLK-I2] pre_flight_check PASS (log: $PREFLIGHT_LOG)"

set -uo pipefail

PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
# 2026-10-05 (Round-3) — the output directory moved.  The 2026-10-05 Round-2
# run in `cfm_tmqm_all6_retrain/` is the MEASURED baseline this run is being
# compared against, and `--out-dir` is also where the trainer writes
# `per_epoch_metrics.json` + `events.jsonl`; pointing a second run at the
# same directory overwrites the baseline in place.  Override with
# REPORTS_DIR=... to go back.
REPORTS_DIR="${REPORTS_DIR:-${PROJECT_ROOT}/molmetal/reports/cfm_round3_retrain/run}"
LOG_OUT="${REPORTS_DIR}/train.log"
METRIC_OUT="${REPORTS_DIR}/per_epoch_metrics.json"
CFG="${PROJECT_ROOT}/configs/cfm.yaml"

mkdir -p "${REPORTS_DIR}"

EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-0}"
# Round-3 D1.  0.0 is the pre-D1 objective and is NOT what this run measures,
# so the launcher carries a non-zero default: re-running this wrapper has to
# reproduce the Round-3 run, not silently reproduce Round-2.  `--assemble-loss-weight
# 0` on the command line still restores the old objective.
ASSEMBLE_LOSS_WEIGHT="${ASSEMBLE_LOSS_WEIGHT:-0.05}"
# Round-3 gate hygiene: the run must start cold.  `--resume-from auto` picks
# up whatever checkpoint is in the config's `ckpt_dir` and then reports its
# metrics as this run's, which is how a 0-epoch "run" ends up claiming a
# decode_ratio it never measured.
RESUME_FROM="${RESUME_FROM:-none}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --assemble-loss-weight) ASSEMBLE_LOSS_WEIGHT="$2"; shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Round-3 D2 and D3 need no flag here: both are config keys the trainer
# already reads (`connectivity_prior: "on"`, `gen_eval_n: 64`,
# `gen_eval_seeds: "0 1 2"` in configs/cfm_tmqm_all6.yaml).  Passing them
# on the command line as well would only be a second place for them to
# disagree.
CMD=("${PY}" "${PROJECT_ROOT}/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
     --config "${CFG}"
     --epochs "${EPOCHS}"
     --seed "${SEED}"
     --device cuda:0
     --assemble-loss-weight "${ASSEMBLE_LOSS_WEIGHT}"
     --resume-from "${RESUME_FROM}"
     --out-dir "${REPORTS_DIR}")

echo
echo "=== cfm_tmqm_all6_retrain launcher ==="
echo "  PY:           ${PY}"
echo "  REPORTS_DIR:  ${REPORTS_DIR}"
echo "  CFG:          ${CFG}"
echo "  EPOCHS:       ${EPOCHS}"
echo "  SEED:         ${SEED}"
echo "  ASSEMBLE_LOSS_WEIGHT: ${ASSEMBLE_LOSS_WEIGHT}  (Round-3 D1)"
echo "  RESUME_FROM:  ${RESUME_FROM}"
echo "  LOG_OUT:      ${LOG_OUT}"
echo "  DRY_RUN:      ${DRY_RUN}"
echo "  CMD: ${CMD[*]}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "DRY RUN — not launching."
    exit 0
fi


echo "Starting at $(date -Iseconds)"
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   setsid nohup "${CMD[@]}" 2>&1 | tee "${LOG_OUT}" || true   # 2026-10-08 pipefix
#   RC=${PIPESTATUS[0]}                              # 2026-10-08 pipefix
#
# Canonical shape (scripts/_WRAPPER_TEMPLATE.sh:190-193): pipefail off for
# this ONE pipeline (a trainer whose forked DataLoader workers still hold the
# pipe open takes SIGPIPE and would report 141 instead of its own status — the
# tee stays, it is the only diagnostic); `setsid --wait` (bare setsid may fork
# and return 0 before the trainer has done anything, which would make
# PIPESTATUS[0] report success for a run that had not started — and setsid
# gives the trainer its own session, which already covers nohup's SIGHUP
# immunity); and the rc read BETWEEN the two `set` lines, while PIPESTATUS
# still holds the pipeline's statuses.
set +o pipefail
setsid --wait "${CMD[@]}" 2>&1 | tee "${LOG_OUT}"
RC=${PIPESTATUS[0]}
set -o pipefail
echo "Done at $(date -Iseconds)"
if [ "${RC}" -ne 0 ]; then
    echo "FAIL: trainer exited rc=${RC}" >&2
    exit "${RC}"
fi

echo
echo "=== Result ==="
ls -la "${REPORTS_DIR}/"
