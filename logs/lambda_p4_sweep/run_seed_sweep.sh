#!/usr/bin/env bash
# Run the Lambda p4 sweep for one seed and move the JSONs into
# logs/lambda_p4_sweep/seed_<seed_index>/.
#
# Usage: run_seed_sweep.sh <seed_value> <seed_index> [pockets] [n_sims] [top_k]
set -euo pipefail
cd "$(dirname "$0")/../.."

SEED_VAL="${1:-42}"
SEED_IDX="${2:-0}"
POCKETS="${3:-100}"
N_SIMS="${4:-20}"
TOP_K="${5:-5}"

LOG_DIR="logs/lambda_p4_sweep"
SEED_DIR="${LOG_DIR}/seed_${SEED_IDX}"
TMP_OUT_DIR="_lambda_p4_seed_${SEED_IDX}"

mkdir -p "${LOG_DIR}" "${SEED_DIR}"

echo "[lambda_p4] seed=${SEED_VAL} idx=${SEED_IDX} pockets=${POCKETS} n_sims=${N_SIMS} top_k=${TOP_K}"
source ./env.sh
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets "${POCKETS}" \
    --seeds "${SEED_VAL}" \
    --n-simulations "${N_SIMS}" \
    --n-top-k "${TOP_K}" \
    --output-dir "${TMP_OUT_DIR}" \
    --quiet 2>&1 | tail -20

# Move generated report.json / summary.md into the per-seed dir
SRC_DIR="molmetal/reports/wf_lambda1_${TMP_OUT_DIR}"
if [[ ! -f "${SRC_DIR}/report.json" ]]; then
    echo "[lambda_p4] ERROR: missing ${SRC_DIR}/report.json" >&2
    exit 2
fi
cp -f "${SRC_DIR}/report.json" "${SEED_DIR}/report.json"
cp -f "${SRC_DIR}/seeds.json" "${SEED_DIR}/seeds.json" 2>/dev/null || true
cp -f "${SRC_DIR}/summary.md" "${SEED_DIR}/summary.md"

# Clean up the wf_lambda1_<tmp> scaffolding to keep the reports tree tidy
rm -rf "${SRC_DIR}"

echo "[lambda_p4] wrote ${SEED_DIR}/report.json"
