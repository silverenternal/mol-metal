#!/usr/bin/env bash
# Wf-310C Part 4 — TODO-220 Driver 1+2 full GPU launch wrapper.
#
# ============================================================================
# Wf-310C PART 4 — TODO-220 GPU WRAPPER FINAL
# ============================================================================
#
# This wrapper is the Wf-310C sibling to the Wf-308C launch_wf308c_todo220_gpu.sh
# wrapper. It uses the same USER_OK-file gate convention but additionally
# runs the Wf-259 pre_flight_check.py first (per the user's Wf-310C spec) so
# the env is verified before the 3.9 hr GPU sweep starts.
#
# Stage A — temporal-split (Wf-219 Krasnov exact protocol)
#   * 5 metals (Ru/Ir/Os/Rh/Re) × 5 seeds × 50 epochs
#   * hidden=300, depth=3 (chemprop v2 production)
#   * Wall: ~2.9 hr on RTX 5090 (projected from Wf-219 + Wf-303C T3 timing)
#
# Stage B — NCI-60 9-cell multi-task
#   * 9 cells × 3 seeds × 50 epochs (per-cell single-task; MultiTaskLoss head
#     is a TODO-130 G3 follow-up)
#   * hidden=128, depth=3 (TODO-130 G3 production default)
#   * Wall: ~1.0 hr on RTX 5090 (projected)
#
# Total wall: 3.9 hr GPU.
#
# USAGE:
#   # 1. CPU smoke (already covered by Wf-310C Parts 1+2 — confirmed working)
#   # 2. To authorise the GPU run:
#   touch /tmp/USER_OK_WF310C_TODO220
#   bash molmetal/scripts/launch_wf310c_todo220_gpu.sh
#
# Exit codes:
#   0 = GPU run completed (all stages ok)
#   2 = USER_OK missing (gated; no run started)
#   3 = Pre-flight check failed (env / GPU / CSV / pre_flight_check.py FAILed)
#   4 = Stage A failed (Stage B not attempted)
#   5 = Stage B failed (Stage A results still saved)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root from script location so the wrapper works regardless
# of the caller's CWD.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
USER_OK_FILE="/tmp/USER_OK_WF310C_TODO220"

TEMPORAL_OUT_DIR="molmetal/reports/wf310c_temporal_split"
NCI60_OUT_DIR="molmetal/reports/wf310c_nci60_9cell"
TEMPORAL_SUMMARY="${TEMPORAL_OUT_DIR}/sweep_summary.json"
NCI60_SUMMARY="${NCI60_OUT_DIR}/sweep_summary.json"
PRE_FLIGHT_OUT="molmetal/reports/wf310c_todo220_launch/pre_flight.json"
PLAN_JSON="molmetal/reports/wf310c_todo220_launch/plan.json"

TEMPORAL_CSV=".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
NCI60_CSV="molmetal/data/nci60_precious_9cell.csv"

echo "[wf310c-todo220] project_root=${PROJECT_ROOT}"
echo "[wf310c-todo220] venv=${VENV_PY}"
echo "[wf310c-todo220] USER_OK_FILE=${USER_OK_FILE}"

mkdir -p "${TEMPORAL_OUT_DIR}" "${NCI60_OUT_DIR}" "$(dirname "${PLAN_JSON}")"

# ---------------------------------------------------------------------------
# 1. Pre-flight: venv + trainers + CSVs + pre_flight_check.py
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[wf310c-todo220] ERROR: venv python missing at ${VENV_PY}" >&2
    exit 3
fi
if [[ ! -f "${TEMPORAL_CSV}" ]]; then
    echo "[wf310c-todo220] ERROR: temporal CSV missing at ${TEMPORAL_CSV}" >&2
    exit 3
fi
if [[ ! -f "${NCI60_CSV}" ]]; then
    echo "[wf310c-todo220] ERROR: NCI-60 CSV missing at ${NCI60_CSV}" >&2
    exit 3
fi
if [[ ! -f "molmetal/scripts/pre_flight_check.py" ]]; then
    echo "[wf310c-todo220] ERROR: pre_flight_check.py missing" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 2. USER_OK gate (kill_then_adapt convention per Wf-295D + Wf-306B)
# ---------------------------------------------------------------------------
if [[ ! -f "${USER_OK_FILE}" ]]; then
    cat <<EOF
[wf310c-todo220] USER_OK GATE NOT PASSED.

To authorise the full TODO-220 GPU run, create the marker file and re-run:

    touch ${USER_OK_FILE}
    bash molmetal/scripts/launch_wf310c_todo220_gpu.sh

Then monitor via:
    tail -f ${TEMPORAL_OUT_DIR}/run.log
    tail -f ${NCI60_OUT_DIR}/run.log

Total wall: ~3.9 hr GPU (Stage A 2.9 hr + Stage B 1.0 hr).

CPU smoke status (Wf-310C Parts 1+2 already verified 2026-09-29):
  * Part 1 (temporal-split): 5/5 ok runs @ 0.42 min on CPU (1-seed 1-epoch hidden=64)
  * Part 2 (NCI-60 9-cell):   9/9 ok runs @ 0.09 min on CPU (1-seed 1-epoch hidden=64)
EOF
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. Wf-259 pre_flight_check.py — auto-verifies chemprop v2 + CFM env gates
#    per the v2 pre-flight checklist (TODO-207). Runs before GPU launch so
#    any FAILed env gate is surfaced BEFORE the 3.9 hr GPU wall is committed.
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === pre_flight_check.py (Wf-259) — chemprop_v2_v3 pipeline ==="
PRE_FLIGHT_RC=0
"${VENV_PY}" molmetal/scripts/pre_flight_check.py \
    --pipeline chemprop_v2_v3 \
    --json "${PRE_FLIGHT_OUT}" \
    --verbose \
    2>&1 | tee "$(dirname "${PRE_FLIGHT_OUT}")/pre_flight.log" || PRE_FLIGHT_RC=$?

if [ "${PRE_FLIGHT_RC}" -ne 0 ]; then
    echo "[wf310c-todo220] ERROR: pre_flight_check.py FAILED (rc=${PRE_FLIGHT_RC}); see ${PRE_FLIGHT_OUT}" >&2
    echo "[wf310c-todo220] Per Wf-259 contract: do NOT proceed with GPU run until FAILs are fixed." >&2
    exit 3
fi
ok_preflight() { printf '%s[OK]%s %s\n' '\033[0;32m' '\033[0m' "$*"; }
ok_preflight "pre_flight_check.py PASSED → ${PRE_FLIGHT_OUT}"

# ---------------------------------------------------------------------------
# 4. GPU probe (per Wf-130 + Wf-190 env guard)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] GPU probe ..."
CUDA_PROBE_OUT=$(
    CUDA_VISIBLE_DEVICES="${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}" \
    MOLMETAL_ALLOW_CPU=0 \
    "${VENV_PY}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
) || true
echo "[wf310c-todo220] CUDA probe: ${CUDA_PROBE_OUT}"
if ! echo "${CUDA_PROBE_OUT}" | grep -q "^True"; then
    echo "[wf310c-todo220] ERROR: CUDA not available on this host; aborting GPU run." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 5. Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED
# label discipline (party-twenty-spirit §3.5).
# ---------------------------------------------------------------------------
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310c_todo220_gpu.sh",\n'
    printf '  "wf_origin": "Wf-310C Part 4",\n'
    printf '  "sibling": "launch_wf308c_todo220_gpu.sh (Wf-308C Part 3)",\n'
    printf '  "todo_ref": "TODO-pending/220_wf219_temporal_split_nci60_driver.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "pre_flight_check": "PASSED (Wf-259 chemprop_v2_v3 pipeline; JSON: %s)",\n' "${PRE_FLIGHT_OUT}"
    printf '  "stage_a": {\n'
    printf '    "name": "temporal-split (Wf-219 Krasnov exact)",\n'
    printf '    "metals": ["Ru","Ir","Os","Rh","Re"],\n'
    printf '    "seeds": [0,1,2,3,4],\n'
    printf '    "epochs": 50,\n'
    printf '    "hidden": 300,\n'
    printf '    "depth": 3,\n'
    printf '    "wall_hr": 2.9,\n'
    printf '    "wall_label": "PROJECTED from Wf-219 + Wf-303C T3 timing"\n'
    printf '  },\n'
    printf '  "stage_b": {\n'
    printf '    "name": "NCI-60 9-cell multi-task",\n'
    printf '    "cells": ["MCF7","A2780","A549","HCT116","PC3","HL60","A2780cis","DU145","SKOV3"],\n'
    printf '    "seeds": [0,1,2],\n'
    printf '    "epochs": 50,\n'
    printf '    "hidden": 128,\n'
    printf '    "depth": 3,\n'
    printf '    "wall_hr": 1.0,\n'
    printf '    "wall_label": "PROJECTED from per-cell 50-epoch chemprop v2 timing"\n'
    printf '  },\n'
    printf '  "total_wall_hr": 3.9,\n'
    printf '  "gpu": "cuda:1",\n'
    printf '  "honest_framing": "Wf-310C Parts 1+2 CPU smoke verified 2026-09-29 (5/5 temporal + 9/9 NCI-60 ok runs). GPU run is the MEASURED-vs-PROJECTED promotion step that requires USER_OK."\n'
    printf '}\n'
} > "${PLAN_JSON}"
echo "[wf310c-todo220] plan emitted → ${PLAN_JSON}"

# ---------------------------------------------------------------------------
# 6. Stage A — Temporal split, 5-metal × 5-seed × 50-epoch (Krasnov 2026)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === Stage A: temporal-split 5-metal × 5-seed × 50-epoch ==="
start_ts_a=$(date +%s)
if ! CUDA_VISIBLE_DEVICES="${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}" \
     "${VENV_PY}" molmetal/scripts/wf219_temporal_split_driver.py \
        --csv "${TEMPORAL_CSV}" \
        --train-cutoff 2023-01-01 \
        --test-cutoff 2024-01-01 \
        --metals Ru Ir Os Rh Re \
        --seeds 0 1 2 3 4 \
        --epochs 50 \
        --hidden 300 \
        --depth 3 \
        --device cuda \
        --num-workers 4 \
        --time-budget-s 10800 \
        --out "${TEMPORAL_SUMMARY}" \
        --execute \
        2>&1 | tee "${TEMPORAL_OUT_DIR}/run.log"; then
    echo "[wf310c-todo220] ERROR: Stage A failed; see ${TEMPORAL_OUT_DIR}/run.log" >&2
    exit 4
fi
elapsed_a=$(( $(date +%s) - start_ts_a ))
elapsed_a_min=$(( elapsed_a / 60 ))
echo "[wf310c-todo220] Stage A OK. elapsed=${elapsed_a_min} min. Summary: ${TEMPORAL_SUMMARY}"

# ---------------------------------------------------------------------------
# 7. Stage B — NCI-60 9-cell, 9 × 3-seed × 50-epoch (TODO-130 G3 prod)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === Stage B: NCI-60 9-cell × 3-seed × 50-epoch ==="
start_ts_b=$(date +%s)
if ! CUDA_VISIBLE_DEVICES="${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}" \
     "${VENV_PY}" molmetal/scripts/nci60_9cell_chemprop_driver.py \
        --csv "${NCI60_CSV}" \
        --cells MCF7 A2780 A549 HCT116 PC3 HL60 A2780cis DU145 SKOV3 \
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
    echo "[wf310c-todo220] ERROR: Stage B failed; see ${NCI60_OUT_DIR}/run.log" >&2
    exit 5
fi
elapsed_b=$(( $(date +%s) - start_ts_b ))
elapsed_b_min=$(( elapsed_b / 60 ))
echo "[wf310c-todo220] Stage B OK. elapsed=${elapsed_b_min} min. Summary: ${NCI60_SUMMARY}"

# ---------------------------------------------------------------------------
# 8. Wrap-up
# ---------------------------------------------------------------------------
elapsed_total_min=$(( (elapsed_a + elapsed_b) / 60 ))
echo "[wf310c-todo220] === Wf-310C TODO-220 GPU run COMPLETE ==="
echo "[wf310c-todo220] Stage A elapsed: ${elapsed_a_min} min (PROJECTED 174 min ≈ 2.9 hr)"
echo "[wf310c-todo220] Stage B elapsed: ${elapsed_b_min} min (PROJECTED 60 min ≈ 1.0 hr)"
echo "[wf310c-todo220] Total elapsed:   ${elapsed_total_min} min (PROJECTED 234 min ≈ 3.9 hr)"
echo "[wf310c-todo220] Temporal summary: ${TEMPORAL_SUMMARY}"
echo "[wf310c-todo220] NCI-60 summary:   ${NCI60_SUMMARY}"
echo "[wf310c-todo220] Run logs:"
echo "[wf310c-todo220]   ${TEMPORAL_OUT_DIR}/run.log"
echo "[wf310c-todo220]   ${NCI60_OUT_DIR}/run.log"
exit 0
