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

set -euo pipefail

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
CFG="${PROJECT_ROOT}/configs/cfm_tmqm_all6.yaml"

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
"${CMD[@]}" 2>&1 | tee "${LOG_OUT}"
echo "Done at $(date -Iseconds)"

echo
echo "=== Result ==="
ls -la "${REPORTS_DIR}/"
