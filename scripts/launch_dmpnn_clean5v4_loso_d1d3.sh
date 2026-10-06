#!/usr/bin/env bash
# 2026-10-06 — Round 7 D-MPNN LOMO re-measurement WITH D1 (GroupShuffleSplit) + D3 (Rh pos_weight=none).
#
# Why
# ---
# Round 5 E2 root-cause fixed val/test mismatch (Rh val was in-distribution,
# test was true LOMO) but still used row-level halve — 108 Rh real rows /
# 51 unique ligands → 48-63% of test rows share an exact ligand with val.
# val_AUC=0.785 selected on memorised ligands that were also in test, and
# test_AUC=0.386 reported below-chance as a sign error, not noise.
#
# Round 7 fixes (this script):
#  D1: GroupShuffleSplit on canonical stripped ligand groups, not rows.
#  D3: For Rh fold (held_out=Rh + --rh-loso-validation on), default
#      pos_weight_mode=none and balanced_sampler=False to remove the
#      train-prior push that was inverting the ranking on Rh.
#
# Scope
# -----
#   - 5 metals (Ru / Ir / Os / Rh / Re), 5 seeds each, 30 epochs each
#   - GPU 1 (RTX 5090) — Blackwell on GPU 0 stays free for CFM F5+F6 retrain
#   - --rh-loso-validation ON (D1 + D3 path)
#   - --metal-strip-mode retain (F3 wf-7b fix; strip is no-op on metal_lomo.csv)
#   - Wasserstein W2 weight 0.0 (E1 default; opt-in via WASSERSTEIN_REAL=1)
#   - New output dir; does NOT overwrite v3 baseline
#   - Wall-time target: ~3-5 hr GPU 1
#
# Honest framing
# --------------
# v3 baseline (2026-10-06, this dir's predecessor): overall 0.5435 BELOW 0.7 gate,
# Rh 0.3380 BELOW_CHANCE. v4 expected: Rh 0.45-0.60 (D1 closes leak, D3 removes
# sign error); overall 0.55-0.65. Both outcomes are MEASURED and honestly reported.
#
# Usage
# -----
#   bash scripts/launch_dmpnn_clean5v4_loso_d1d3.sh --dry-run
#   bash scripts/launch_dmpnn_clean5v4_loso_d1d3.sh
#
# Constraint: party-twenty-spirit §3.5; AGENTS.md §5.1 (USER_OK GPU) + §5.5
# (filesystem truth) + §5.7 (smoke) + §5.9 (single venv).

set -euo pipefail

PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
# 2026-10-06 (Round 7): new output dir; do not overwrite v3 baseline.
REPORTS_DIR="${REPORTS_DIR:-${PROJECT_ROOT}/molmetal/reports/dmpnn_clean5v4_loso_d1d3}"
LOG_OUT="${REPORTS_DIR}/train.log"
METRIC_OUT="${REPORTS_DIR}/metrics.json"

mkdir -p "${REPORTS_DIR}"

EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METALS="${METALS:-Ru Ir Os Rh Re}"
RH_LOSO="${RH_LOSO:-1}"  # D1 + D3 require this
WASSERSTEIN_REAL="${WASSERSTEIN_REAL:-0}"
RH_OVERSAMPLE="${RH_OVERSAMPLE:-500}"
WASSERSTEIN_WEIGHT="${WASSERSTEIN_WEIGHT:-0.0}"
METAL_STRIP_MODE="${METAL_STRIP_MODE:-retain}"  # F3 wf-7b fix default
# Round 7 (2026-10-06): the trainer's `ensure_cuda_visible_devices_default`
# honours the env var set by us, so torch-visible index is always 0.
# We expose --gpu to pick the physical GPU (--device is visible-index).
GPU="${GPU:-1}"  # physical GPU index (1=RTX 5090; 0=Blackwell)
DEVICE="cuda:0"  # visible index inside the trainer
# Wf-R10-A3: expose the CSV + exposure-time window so the Rh-block widening
# is reachable from the launcher.  Defaults preserve the Round-7/8 behaviour.
CSV="${CSV:-/home/hugo/codes/mol-metal/molmetal/data/metal_lomo.csv}"
TIME_THRESHOLD="${TIME_THRESHOLD:-24.0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --metal-strip-mode) METAL_STRIP_MODE="$2"; shift 2 ;;
        --no-rh-loso) RH_LOSO=0; shift ;;
        --wasserstein-real) WASSERSTEIN_REAL=1; shift ;;
        --wasserstein-weight) WASSERSTEIN_WEIGHT="$2"; shift 2 ;;
        --rh-oversample) RH_OVERSAMPLE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

CMD=("${PY}" "${PROJECT_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"
     --metals ${METALS}
     --seeds ${SEEDS}
     --epochs "${EPOCHS}"
     --wasserstein-weight "${WASSERSTEIN_WEIGHT}"
     --rh-oversample "${RH_OVERSAMPLE}"
     --metal-strip-mode "${METAL_STRIP_MODE}"
     --device "${DEVICE}"
     --out-dir "${REPORTS_DIR}"
     --csv "${CSV}"
     --time-threshold "${TIME_THRESHOLD}"
     --out-metric "${METRIC_OUT}")

if [[ ${RH_LOSO} -eq 1 ]]; then
    CMD+=(--rh-loso-validation)
fi

if [[ ${WASSERSTEIN_REAL} -eq 1 ]]; then
    CMD+=(--wasserstein-real)
fi

# Set CUDA_VISIBLE_DEVICES to the physical GPU index (env var wins over
# the trainer's `ensure_cuda_visible_devices_default` per Wf-A9 semantics).
# We use --device cuda:0 because torch sees only 1 GPU after filtering.
export CUDA_VISIBLE_DEVICES="${GPU}"
echo "  CUDA_VISIBLE_DEVICES (physical GPU): ${CUDA_VISIBLE_DEVICES}"
echo "  DEVICE (visible index):              ${DEVICE}"

echo "=== dmpnn_clean5v4_loso_d1d3 launcher (Round 7 D1+D3 fix) ==="
echo "  PY:                   ${PY}"
echo "  REPORTS_DIR:          ${REPORTS_DIR}"
echo "  METRIC_OUT:           ${METRIC_OUT}"
echo "  LOG_OUT:              ${LOG_OUT}"
echo "  EPOCHS:               ${EPOCHS}"
echo "  SEEDS:                ${SEEDS}"
echo "  METALS:               ${METALS}"
echo "  RH_LOSO (D1+D3):      ${RH_LOSO}"
echo "  WASSERSTEIN_REAL:     ${WASSERSTEIN_REAL}"
echo "  WASSERSTEIN_WEIGHT:   ${WASSERSTEIN_WEIGHT}"
echo "  RH_OVERSAMPLE:        ${RH_OVERSAMPLE}"
echo "  METAL_STRIP_MODE:     ${METAL_STRIP_MODE}"
echo "  DEVICE:               ${DEVICE}"
echo "  CSV:                  ${CSV}"
echo "  TIME_THRESHOLD:       ${TIME_THRESHOLD}"
echo "  DRY_RUN:              ${DRY_RUN}"
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
if [[ -f "${METRIC_OUT}" ]]; then
    echo "metric written: ${METRIC_OUT}"
    ls -la "${METRIC_OUT}"
else
    echo "WARN: metric not written; check ${LOG_OUT}"
fi
