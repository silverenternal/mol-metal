#!/usr/bin/env bash
# 2026-10-06 — Round 7 CFM re-train on tmQM all6 with F5+F6 fixes applied.
#
# Why
# ---
# Round 5/6 ship surface (D6 retrain, 2026-10-04 to 10-05):
#   - Round-2/D6 best decode_ratio=0.402 was actually the *untrained* model —
#     velocity field never received a CFM target because `use_fm_objective`
#     was missing from the config. F4-F6 audits identified three framework
#     bugs in CFM training; F5 + F6 are now fixed (config + trainer).
#   - F5: `use_fm_objective: true` in configs/cfm_tmqm_all6.yaml +
#     `fm_vel_head` gated on `not use_egnn_velocity`. EGNN field now has a
#     real regression target instead of zero.
#   - F6: `--fm-t-per-cloud` flag, default 1 (per-cloud t-distribution
#     instead of per-atom t-collapse). FM MSE -23% in smoke.
#   - F4 deferred: Wasserstein reference positions hardcoded Pt
#     geometry (build_reference_positions() at L3925); only an issue if W2
#     term is on, which it is not (--wasserstein-weight 0).
#
# This script re-trains on the fixed config so we can measure whether
# gen_validity moves off the 0/128 baseline and decode_ratio climbs.
#
# Scope
# -----
#   - configs/cfm.yaml (F5+F6 already in as defaults; cfm_tmqm_all6.yaml
#     deleted 2026-10-07 per user "配置和当前项目真实环境对齐")
#   - 20 epochs, default seed
#   - GPU 0 (Blackwell, 86 GB free — RTX 5090 used by D-MPNN retrain)
#   - New output dir; does NOT overwrite cfm_round3_retrain/run (D6 baseline)
#   - Wall-time target: ~4-5 hr GPU 0
#
# Honest framing
# --------------
# D6 baseline (cfm_round3_retrain/run, 2026-10-04): best decode_ratio=0.402,
# best gen_validity=0.9115 @ epoch 18, fm_term=0. Round 7 expected:
#   - fm_term > 0 (F5 fix verified by §5.7 smoke; F5 wire-in is REQUIRED
#     for the velocity field to learn anything)
#   - gen_validity stays near D6's 0.5-0.9 range (the encoder path was OK
#     even when the field was untrained)
#   - decode_ratio improves *iff* the velocity field now produces
#     trajectories that the YuelBond decoder can bond-decide coherently.
#     If gen_validity stays flat, the F5 fix is not enough — fall back to
#     midpoint solver + larger W2 term.
# Both outcomes are MEASURED and honestly reported.
#
# Usage
# -----
#   bash scripts/launch_cfm_round7_f5f6_retrain.sh --dry-run
#   bash scripts/launch_cfm_round7_f5f6_retrain.sh
#
# Constraint: AGENTS.md §5.1 (standing GPU authorization) + §5.7
# (50-step smoke already passed for both F5+F6 flag changes; this is the
# full retrain).

set -euo pipefail

PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
# 2026-10-06 (Round 7) — new dir; do NOT overwrite cfm_round3_retrain/run.
REPORTS_DIR="${REPORTS_DIR:-${PROJECT_ROOT}/molmetal/reports/cfm_round7_f5f6_retrain/run}"
LOG_OUT="${REPORTS_DIR}/train.log"
CFG="${PROJECT_ROOT}/configs/cfm.yaml"

mkdir -p "${REPORTS_DIR}"

EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-0}"
ASSEMBLE_LOSS_WEIGHT="${ASSEMBLE_LOSS_WEIGHT:-0.05}"  # D1 default
RESUME_FROM="${RESUME_FROM:-none}"
GPU="${GPU:-0}"  # physical GPU index (0=Blackwell; 1=RTX 5090)
DEVICE="cuda:0"  # visible index inside the trainer
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --assemble-loss-weight) ASSEMBLE_LOSS_WEIGHT="$2"; shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

CMD=("${PY}" "${PROJECT_ROOT}/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
     --config "${CFG}"
     --epochs "${EPOCHS}"
     --seed "${SEED}"
     --device "${DEVICE}"
     --assemble-loss-weight "${ASSEMBLE_LOSS_WEIGHT}"
     --resume-from "${RESUME_FROM}"
     --out-dir "${REPORTS_DIR}")

# Set CUDA_VISIBLE_DEVICES to the physical GPU index (env var wins over
# the trainer's `ensure_cuda_visible_devices_default` per Wf-A9 semantics).
# We use --device cuda:0 because torch sees only 1 GPU after filtering.
export CUDA_VISIBLE_DEVICES="${GPU}"
echo "  CUDA_VISIBLE_DEVICES (physical GPU): ${CUDA_VISIBLE_DEVICES}"
echo "  DEVICE (visible index):              ${DEVICE}"

echo
echo "=== cfm_round7_f5f6_retrain launcher (Round 7 F5+F6 fix) ==="
echo "  PY:           ${PY}"
echo "  REPORTS_DIR:  ${REPORTS_DIR}"
echo "  CFG:          ${CFG}"
echo "  EPOCHS:       ${EPOCHS}"
echo "  SEED:         ${SEED}"
echo "  ASSEMBLE_LOSS_WEIGHT: ${ASSEMBLE_LOSS_WEIGHT}  (Round-3 D1)"
echo "  RESUME_FROM:  ${RESUME_FROM}"
echo "  DEVICE:       ${DEVICE}"
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
