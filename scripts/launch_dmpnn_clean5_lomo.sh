#!/usr/bin/env bash
# 2026-10-05 — Clean 5-metal D-MPNN LOMO re-measurement (Tier-2 headline).
#
# Why
# ---
# Per Wf-432..435 audit: the existing metal_lomo_auc_wasserstein.json has
# 6 disclosure fields flagging it as USABLE_AS_HEADLINE: false because the
# underlying measurements were taken before the Wf-385 6.15x D-MPNN
# vectorization fix + Wf-405-A2 Triton-W2 hook were applied.  To get a
# Tier-2-shippable headline we need a CLEAN re-measurement on the new
# code path with all the recent fixes in place.
#
# Scope
# -----
#   - 5 metals (Ru / Ir / Os / Rh / Re), 5 seeds each, 30 epochs each
#   - CPU-only (D-MPNN on RTX 5090 saturates; CPU is faster wall-time)
#   - Wasserstein W2 + Rh oversample + L2 guard (full Wf-125 P2 stack)
#   - One output dir; one metric JSON
#   - Wall-time target: ~1.4 hr CPU
#
# Honest framing
# --------------
# The previous headline (m19 LOMO 0.535, Rh 0.368) is honest-negative:
# measured on the old code path.  This script writes a NEW metric to
# molmetal/reports/dmpnn_clean5_lomo/metrics.json with provenance so the
# paper can cite whichever is closer to ground truth.
#
# Usage
# -----
#   bash scripts/launch_dmpnn_clean5_lomo.sh --dry-run    # show plan
#   bash scripts/launch_dmpnn_clean5_lomo.sh              # actual launch
#
# Constraint: party-twenty-spirit §3.5; AGENTS.md §5.5 (filesystem truth).

# 2026-10-08 pipefix: drop pipefail (SIGPIPE from forked DataLoader workers
# flushing stdout after trainer exit killed the wrapper silently via `tee`).
# Capture trainer rc via ${PIPESTATUS[0]}; suppress tee SIGPIPE with `|| true`.
set -uo pipefail

PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/dmpnn_clean5_lomo"
METRIC_OUT="${REPORTS_DIR}/metrics.json"
LOG_OUT="${REPORTS_DIR}/train.log"

mkdir -p "${REPORTS_DIR}"

# Defaults
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METALS="${METALS:-Ru Ir Os Rh Re}"
MODEL_SELECT="${MODEL_SELECT:-last_epoch}"
# P-A fix (Wf-R40): was hard-coded `--rh-oversample 500`, which re-injects
# the fabricated-label Rh cohort on every run and breaks the real-only
# re-measurement the Wf-432..435 audit requires.  The trainer's default
# (train_metal_lomo_wasserstein.py:1831) is `default=0` since Wf-R10,
# so leaving RH_OVERSAMPLE unset = real-only fold.  Set explicitly via
# env var to reproduce the historical wf125_p1 / wf131_p2 path.
RH_OVERSAMPLE="${RH_OVERSAMPLE:-0}"
DRY_RUN=0
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --quick) QUICK=1; EPOCHS=5; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --model-select) MODEL_SELECT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

CMD=("${PY}" "${PROJECT_ROOT}/molmetal/scripts/train_metal_lomo_wasserstein.py"
     --metals ${METALS}
     --seeds ${SEEDS}
     --epochs "${EPOCHS}"
     --wasserstein-weight 0.05
     --rh-oversample "${RH_OVERSAMPLE}"
     --model-select "${MODEL_SELECT}"
     --out-dir "${REPORTS_DIR}"
     --out-metric "${METRIC_OUT}")

if [[ ${QUICK} -eq 1 ]]; then
    CMD+=(--quick)
fi

echo "=== dmpnn_clean5_lomo launcher ==="
echo "  PY:           ${PY}"
echo "  REPORTS_DIR:  ${REPORTS_DIR}"
echo "  METRIC_OUT:   ${METRIC_OUT}"
echo "  LOG_OUT:      ${LOG_OUT}"
echo "  EPOCHS:       ${EPOCHS}"
echo "  SEEDS:        ${SEEDS}"
echo "  METALS:       ${METALS}"
echo "  MODEL_SELECT: ${MODEL_SELECT}"
echo "  RH_OVERSAMPLE:${RH_OVERSAMPLE}"
echo "  DRY_RUN:      ${DRY_RUN}"
echo "  CMD: ${CMD[*]}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "DRY RUN — not launching."
    exit 0
fi

echo "Starting at $(date -Iseconds)"
"${CMD[@]}" 2>&1 | tee "${LOG_OUT}" || true   # 2026-10-08 pipefix
RC=${PIPESTATUS[0]}                              # 2026-10-08 pipefix
echo "Done at $(date -Iseconds)"
if [ "${RC}" -ne 0 ]; then
    echo "FAIL: trainer exited rc=${RC}" >&2
    exit "${RC}"
fi

echo
echo "=== Result ==="
if [[ -f "${METRIC_OUT}" ]]; then
    echo "metric written: ${METRIC_OUT}"
    ls -la "${METRIC_OUT}"
else
    echo "WARN: metric not written; check ${LOG_OUT}"
fi
