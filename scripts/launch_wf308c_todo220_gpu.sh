#!/usr/bin/env bash
# launch_wf308c_todo220_gpu.sh — USER_OK-aware full-GPU launcher for
# TODO-220 temporal-split + NCI-60 9-cell drivers (Wf-308C Part 3).
#
# ============================================================================
# Wf-308C PART 3 — FULL --execute GPU LAUNCH WRAPPER
# ============================================================================
#
# This wrapper gates the full GPU run behind a USER_OK marker file so the
# expensive 5-metal × 5-seed × 50-epoch temporal sweep + 9-cell multi-task
# sweep do NOT start without an explicit user opt-in (kill_then_adapt
# convention per Wf-300C).
#
# Stage A — temporal-split (Wf-219 Krasnov exact protocol)
#   * 5 metals (Ru/Ir/Os/Rh/Re) × 5 seeds × 50 epochs
#   * hidden=300, depth=3 (chemprop v2 production)
#   * Wall: ~2.9 hr on RTX 5090 (projected from Wf-219 + Wf-303C T3 timing)
#
# Stage B — NCI-60 9-cell multi-task
#   * 9 cells × 3 seeds × 50 epochs (per-cell single-task for now;
#     MultiTaskLoss head is a TODO-130 G3 follow-up)
#   * hidden=128, depth=3 (TODO-130 G3 production default)
#   * Wall: ~1.0 hr on RTX 5090 (projected)
#
# Total wall: 3.9 hr GPU.
#
# USAGE:
#   # 1. CPU smoke first (no USER_OK needed; already covered by Wf-208C
#   #    smoke; this script skips the smoke if env var WF208C_SMOKE_DONE=1).
#   # 2. To authorise the GPU run:
#   touch /tmp/USER_OK_WF308C
#   bash molmetal/scripts/launch_wf308c_todo220_gpu.sh
#   # 3. To explicitly skip GPU (default behaviour):
#   unset WF308C_RUN_GPU   # (the default is "no" without USER_OK file)
#
# Exit codes:
#   0 = GPU run completed (all stages ok)
#   2 = USER_OK missing (gated; no run started)
#   3 = Stage A failed (Stage B not attempted)
#   4 = Stage B failed (Stage A results still saved)
#   5 = Setup error (venv / GPU probe failed)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root from script location so the wrapper works regardless
# of the caller's CWD.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PY="/home/hugo/codes/mol-metal/.venv/bin/python"
USER_OK_FILE="/tmp/USER_OK_WF308C"

TEMPORAL_OUT_DIR="molmetal/reports/wf308c_temporal_split"
NCI60_OUT_DIR="molmetal/reports/wf308c_nci60_9cell"
TEMPORAL_SUMMARY="${TEMPORAL_OUT_DIR}/sweep_summary.json"
NCI60_SUMMARY="${NCI60_OUT_DIR}/sweep_summary.json"

TEMPORAL_CSV=".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
NCI60_CSV="molmetal/data/nci60_precious_9cell.csv"

echo "[wf308c-launch] project_root=${PROJECT_ROOT}"
echo "[wf308c-launch] venv=${VENV_PY}"
echo "[wf308c-launch] USER_OK_FILE=${USER_OK_FILE}"

# ---------------------------------------------------------------------------
# 1. Pre-flight: venv + GPU + CSV existence
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[wf308c-launch] ERROR: venv python missing at ${VENV_PY}" >&2
    exit 5
fi
if [[ ! -f "${TEMPORAL_CSV}" ]]; then
    echo "[wf308c-launch] ERROR: temporal CSV missing at ${TEMPORAL_CSV}" >&2
    echo "[wf308c-launch] Per Wf-306B honest-negative: do not run --execute GPU until the CSV is in place." >&2
    exit 5
fi
if [[ ! -f "${NCI60_CSV}" ]]; then
    echo "[wf308c-launch] ERROR: NCI-60 CSV missing at ${NCI60_CSV}" >&2
    exit 5
fi

# ---------------------------------------------------------------------------
# 2. USER_OK gate (kill_then_adapt convention per Wf-300C)
# ---------------------------------------------------------------------------
if [[ ! -f "${USER_OK_FILE}" ]]; then
    cat <<EOF
[wf308c-launch] USER_OK GATE NOT PASSED.

To authorise the full GPU run, create the marker file and re-run:

    touch ${USER_OK_FILE}
    bash molmetal/scripts/launch_wf308c_todo220_gpu.sh

Then monitor via:
    tail -f ${TEMPORAL_OUT_DIR}/run.log
    tail -f ${NCI60_OUT_DIR}/run.log

Total wall: ~3.9 hr GPU (Stage A 2.9 hr + Stage B 1.0 hr).
Honest framing per Wf-306B: CPU smoke (Wf-308C Parts 1+2) is actionable
now and ran successfully; the GPU run is the MEASURED-vs-PROJECTED
promotion step that requires user opt-in.
EOF
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. GPU probe (per Wf-130 + Wf-190 env guard)
# ---------------------------------------------------------------------------
echo "[wf308c-launch] GPU probe ..."
CUDA_PROBE_OUT=$(
    CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-1}" \
    MOLMETAL_ALLOW_CPU=0 \
    "${VENV_PY}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
) || true
echo "[wf308c-launch] CUDA probe: ${CUDA_PROBE_OUT}"
if ! echo "${CUDA_PROBE_OUT}" | grep -q "^True"; then
    echo "[wf308c-launch] ERROR: CUDA not available on this host; aborting GPU run." >&2
    echo "[wf308c-launch] Use the CPU smoke runs instead (see Wf-308C Parts 1+2)." >&2
    exit 5
fi

mkdir -p "${TEMPORAL_OUT_DIR}" "${NCI60_OUT_DIR}"

# ---------------------------------------------------------------------------
# 4. Stage A — Temporal split, 5-metal × 5-seed × 50-epoch (Krasnov 2026)
# ---------------------------------------------------------------------------
echo "[wf308c-launch] === Stage A: temporal-split 5-metal × 5-seed × 50-epoch ==="
if ! CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-1}" \
     "${VENV_PY}" molmetal/scripts/wf219_temporal_split_driver.py \
        --csv "${TEMPORAL_CSV}" \
        --train-cutoff 2024-01-01 \
        --test-cutoff 2025-01-01 \
        --metals Ru Ir Os Rh Re \
        --seeds 0 1 2 3 4 \
        --epochs 50 \
        --hidden 300 \
        --depth 3 \
        --device cuda \
        --num-workers 4 \
        --time-budget-s 9000 \
        --out "${TEMPORAL_SUMMARY}" \
        --execute \
        2>&1 | tee "${TEMPORAL_OUT_DIR}/run.log"; then
    echo "[wf308c-launch] ERROR: Stage A failed; see ${TEMPORAL_OUT_DIR}/run.log" >&2
    exit 3
fi
echo "[wf308c-launch] Stage A OK. Summary: ${TEMPORAL_SUMMARY}"

# ---------------------------------------------------------------------------
# 5. Stage B — NCI-60 9-cell, 9 × 3-seed × 50-epoch (TODO-130 G3 prod)
# ---------------------------------------------------------------------------
echo "[wf308c-launch] === Stage B: NCI-60 9-cell × 3-seed × 50-epoch ==="
if ! CUDA_VISIBLE_DEVICES="${WF308C_CUDA_VISIBLE_DEVICES:-1}" \
     "${VENV_PY}" molmetal/scripts/nci60_9cell_chemprop_driver.py \
        --csv "${NCI60_CSV}" \
        --cells MCF7 A2780 A549 HCT116 PC3 HL60 HCT-116 A2780cis DU145 \
        --seeds 0 1 2 \
        --epochs 50 \
        --hidden 128 \
        --depth 3 \
        --device cuda \
        --num-workers 4 \
        --time-budget-s 3600 \
        --out "${NCI60_SUMMARY}" \
        --execute \
        2>&1 | tee "${NCI60_OUT_DIR}/run.log"; then
    echo "[wf308c-launch] ERROR: Stage B failed; see ${NCI60_OUT_DIR}/run.log" >&2
    exit 4
fi
echo "[wf308c-launch] Stage B OK. Summary: ${NCI60_SUMMARY}"

# ---------------------------------------------------------------------------
# 6. Wrap-up
# ---------------------------------------------------------------------------
echo "[wf308c-launch] === Wf-308C GPU run COMPLETE ==="
echo "[wf308c-launch] Temporal summary: ${TEMPORAL_SUMMARY}"
echo "[wf308c-launch] NCI-60 summary:   ${NCI60_SUMMARY}"
echo "[wf308c-launch] Run logs:"
echo "[wf308c-launch]   ${TEMPORAL_OUT_DIR}/run.log"
echo "[wf308c-launch]   ${NCI60_OUT_DIR}/run.log"
exit 0
