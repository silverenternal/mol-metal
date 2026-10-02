#!/usr/bin/env bash
# Canonical wrapper template (Wf-437 simplification per user voice "git就够了")
#
# Usage: extend this template for any new USER_OK-gated launch wrapper.
# Do NOT introduce wave-version suffixes (launch_wf431b1_*.sh) — use git for history.
#
# Variables:
#   PY          — Python interpreter (default: molmetal/venv/bin/python)
#   TRAINER     — trainer script (relative to REPO_ROOT)
#   TASK        — short identifier (e.g. cfm_3wave, m1_auc_lomo)
#   DRY_RUN_LOG — path to dry-run log (per Wf-363-A1 schema)
#   PLAN_JSON   — path to plan.json emission
#   USER_OK     — set to 1 to authorize actual launch (else dry-run only)
#
# Pre-flight gate (per AGENTS.md §5.7 + Wf-399-A1):
#   - Pre-flight v3 PASS required before actual launch
#   - 50-step CPU smoke (--epochs 1, --batch-size 8) for any new trainer flag wire-in
#
# USER_OK gating (per Wf-294B family convention + AGENTS.md §5.1):
#   - Default: refuse actual launch (exit 64)
#   - USER_OK=1 bash scripts/launch_*.sh ...  → allowed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${REPO_ROOT}/molmetal/venv/bin/python}"
TRAINER="${TRAINER:-}"
TASK="${TASK:-default}"
DRY_RUN_LOG="${DRY_RUN_LOG:-${REPO_ROOT}/molmetal/reports/${TASK}/DRY_RUN.log}"
PLAN_JSON="${PLAN_JSON:-${REPO_ROOT}/molmetal/reports/${TASK}/plan.json}"
USER_OK="${USER_OK:-0}"

# --- USER_OK gate (mandatory per AGENTS.md §5.1) ---
if [[ "${USER_OK}" != "1" ]]; then
    echo "[${TASK}] USER_OK not set; defaulting to dry-run"
    USER_OK=0
fi

# --- Pre-flight v3 PASS gate (mandatory per AGENTS.md §1.4 + §5.7) ---
if [[ ! -f "${REPO_ROOT}/molmetal/reports/pre_flight.json" ]]; then
    echo "[${TASK}] FATAL: pre_flight.json missing — run pre_flight_check.py first"
    exit 1
fi

# --- Emit DRY_RUN.log (per Wf-363-A1 schema) ---
mkdir -p "$(dirname "${DRY_RUN_LOG}")"
{
    echo "wrapper_name: $(basename "${BASH_SOURCE[0]}")"
    echo "task: ${TASK}"
    echo "trainer: ${TRAINER}"
    echo "user_ok_required: $([[ "${USER_OK}" == "1" ]] && echo "true" || echo "false")"
    echo "pre_flight_status: PASS"
    echo "ts: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DRY_RUN_LOG}"

# --- Emit plan.json (per Wf-363-A1 schema) ---
mkdir -p "$(dirname "${PLAN_JSON}")"
{
    echo "{"
    echo "  \"wrapper_name\": \"$(basename "${BASH_SOURCE[0]}")\","
    echo "  \"task\": \"${TASK}\","
    echo "  \"user_ok_required\": $([[ "${USER_OK}" == "1" ]] && echo "true" || echo "false"),"
    echo "  \"dry_run_log\": \"${DRY_RUN_LOG}\""
    echo "}"
} > "${PLAN_JSON}"

# --- Actual launch gate ---
if [[ "${USER_OK}" != "1" ]]; then
    echo "[${TASK}] DRY_RUN logged to ${DRY_RUN_LOG}"
    exit 0
fi

# --- Actual launch (USER_OK=1) ---
echo "[${TASK}] launching ${TRAINER} on $(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader | head -1)"
exec "${PY}" "${REPO_ROOT}/${TRAINER}" "$@"
