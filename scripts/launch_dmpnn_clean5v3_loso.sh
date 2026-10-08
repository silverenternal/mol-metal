#!/usr/bin/env bash
# 2026-10-06 — Round 5 D-MPNN LOMO re-measurement WITH E2 --rh-loso-validation fix.
#
# Why
# ---
# Round 4 E2 root-cause: WassersteinLOMOSplitter drew val from train_pool[~synth]
# (other metals only), so val_auc never saw held-out-metal rows. With Rh held
# out, val=0.785 was in-distribution on Ru/Ir/Re/Os while test=0.386 was the
# true LOMO transfer. Val/test measured DIFFERENT distributions, so val-based
# model selection was structurally wrong.
#
# The E2 fix adds --rh-loso-validation: when Rh is held out, split the 108 real
# Rh rows 50/50 into val and test, so both folds now sample the same target
# domain. This makes val_auc an honest early-stopping signal for test.
#
# Scope
# -----
#   - 5 metals (Ru / Ir / Os / Rh / Re), 5 seeds each, 30 epochs each
#   - GPU 1 (RTX 5090, 29 GB free) — D6 retrain owns GPU 0
#   - --rh-loso-validation ON for Rh fold; other folds unchanged
#   - Wasserstein W2 (real, opt-in via --wasserstein-real) + Rh oversample
#   - One output dir; one metric JSON
#   - Wall-time target: ~3 hr GPU 1
#
# Honest framing
# --------------
# v2 baseline (2026-10-05): Rh 0.386 ± 0.020 BELOW_CHANCE; overall 0.5574 BELOW
# 0.7 gate. v3 expected: Rh 0.50-0.62 (RF reference 0.626); overall 0.62-0.68.
# Both outcomes are MEASURED and honestly reported.
#
# Usage
# -----
#   bash scripts/launch_dmpnn_clean5v3_loso.sh --dry-run
#   bash scripts/launch_dmpnn_clean5v3_loso.sh
#
# Constraint: party-twenty-spirit §3.5; AGENTS.md §5.1 (USER_OK GPU) + §5.5
# (filesystem truth) + §5.7 (smoke) + §5.9 (single venv).

set -euo pipefail

PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
# 2026-10-06 (Round 5): new output dir; do not overwrite v2 baseline.
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/dmpnn_clean5v3_loso"
METRIC_OUT="${REPORTS_DIR}/metrics.json"
LOG_OUT="${REPORTS_DIR}/train.log"
SYNTHESIS_OUT="${REPORTS_DIR}/synthesis.md"

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (per AGENTS.md §5.1 kill_then_adapt — GPU launch > 1 hr)
# This wrapper runs on GPU 1 (RTX 5090) for ~3 hr; USER_OK=1 must be present.
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: this wrapper runs ~3 hr on GPU 1.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s              # actual 3-hr GPU launch\n' "$0" >&2
    exit 64
fi

# Defaults
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METALS="${METALS:-Ru Ir Os Rh Re}"
MODEL_SELECT="${MODEL_SELECT:-last_epoch}"
RH_LOSO="${RH_LOSO:-1}"  # E2 fix ON by default for v3
WASSERSTEIN_REAL="${WASSERSTEIN_REAL:-0}"  # opt-in via env; default off
RH_OVERSAMPLE="${RH_OVERSAMPLE:-500}"
WASSERSTEIN_WEIGHT="${WASSERSTEIN_WEIGHT:-0.0}"  # E1 default is 0.0 (legacy no-op)
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --model-select) MODEL_SELECT="$2"; shift 2 ;;
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
     --model-select "${MODEL_SELECT}"
     --out-dir "${REPORTS_DIR}"
     --out-metric "${METRIC_OUT}")

# E2 fix: only attach --rh-loso-validation when RH_LOSO=1
if [[ ${RH_LOSO} -eq 1 ]]; then
    CMD+=(--rh-loso-validation)
fi

# E1 fix: opt-in to real W2 gradient
if [[ ${WASSERSTEIN_REAL} -eq 1 ]]; then
    CMD+=(--wasserstein-real)
fi

# Use GPU 1 (D6 retrain owns GPU 0)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=1
fi

echo "=== dmpnn_clean5v3_loso launcher (Round 5, E2 fix) ==="
echo "  PY:                   ${PY}"
echo "  REPORTS_DIR:          ${REPORTS_DIR}"
echo "  METRIC_OUT:           ${METRIC_OUT}"
echo "  LOG_OUT:              ${LOG_OUT}"
echo "  EPOCHS:               ${EPOCHS}"
echo "  SEEDS:                ${SEEDS}"
echo "  METALS:               ${METALS}"
echo "  MODEL_SELECT:         ${MODEL_SELECT}"
echo "  RH_LOSO (E2 fix):     ${RH_LOSO}"
echo "  WASSERSTEIN_REAL:     ${WASSERSTEIN_REAL}"
echo "  WASSERSTEIN_WEIGHT:   ${WASSERSTEIN_WEIGHT}"
echo "  RH_OVERSAMPLE:        ${RH_OVERSAMPLE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-1}"
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
