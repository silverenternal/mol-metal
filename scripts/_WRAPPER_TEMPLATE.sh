#!/usr/bin/env bash
# _WRAPPER_TEMPLATE.sh — canonical Mol-Metal launcher template
#
# Per Wf-431-A2 FIX 2 (wrapper reclassification + single template):
#   - PY=/home/hugo/codes/mol-metal/.venv/bin/python HARDCODED (per §5.9 + AGENTS.md
#     memory rule "Standing GPU grant" — no bare `python` invocation)
#   - USER_OK gate FIRST (per Wf-294B + Wf-355-B1 + Wf-366-A1 + Wf-405-A4
#     USER_OK-gate-before-anything pattern; kill_then_block must
#     block LIVE launches at any USER_OK setting other than 1)
#   - DRY_RUN.log + plan.json emission (per Wf-363-A1 schema;
#     `_dry_run_log_helper.sh` is the canonical source so producer
#     producer the producer is consistent across 23 wrappers)
#   - PYTHONPATH=/home/hugo/codes/mol-metal ${PY} ${DOCK_PANEL} (per
#     Wf-431-A1 fix F2: project root must be on PYTHONPATH for
#     `molmetal.*` packages; bare `python` without PYTHONPATH fails
#     with "No module named 'molmetal'")
#   - pre_flight_check.py gate BEFORE any GPU work (per Wf-415-A3 +
#     Wf-422-A2 + Wf-427-B1 cascade closures)
#
# To extend this template for a new wrapper:
#   1. Copy this file to scripts/launch_wf<your_wf>.sh
#   2. Replace DOCK_PANEL with the actual trainer script
#   - Replace WF_ID + WRAPPER_NAME placeholders
#   - Add wrapper-specific CLI args in TRAIN_ARGS section
#   - Replace JINJA_PARAM defaults via sed before launch
#   - Verify with: bash scripts/_wrapper_audit.py
#
# Owner: Hugo Lee + ultracode agent (Wf-431-A2)
# Date: 2026-10-02

set -euo pipefail

# --- Paths (HARDCODED per §5.9) -------------------------------------------
PY=/home/hugo/codes/mol-metal/.venv/bin/python
PROJECT_ROOT=/home/hugo/codes/mol-metal
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
TRAINER="${DOCK_PANEL:-molmetal/scripts/train_cfm_yuelbond_wasserstein.py}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/molmetal/reports/wf_template}"
WRAPPER_NAME="launch_wf_template.sh"
WF_ID="wf_template"
export PYTHONPATH="/home/hugo/codes/mol-metal"

# --- USER_OK gate FIRST (per Wf-294B + Wf-355-B1 + Wf-366-A1) ------------
if [ "${USER_OK:-0}" != "1" ]; then
    cat <<EOF >&2
[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
       USER_OK=1 bash ${WRAPPER_NAME} --dry-run   # plan only (still requires USER_OK=1)
       USER_OK=1 bash ${WRAPPER_NAME}             # actual run
EOF
    exit 64
fi

# --- Helpers -------------------------------------------------------------
die() { printf '%s [FATAL] %s\n' "$(date -Iseconds 2>/dev/null || date)" "$*" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck disable=SC1090
source "${DRY_RUN_LOG_HELPER}"

# --- Pre-flight gate (per Wf-415-A3 + Wf-427-B1) -----------------------
echo "[$(date -Iseconds)] ${WRAPPER_NAME} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch >"${OUT_DIR}/preflight.json" 2>&1 \
    || die "pre_flight_check FAIL — see ${OUT_DIR}/preflight.json"
grep -q '"fail": 0' "${OUT_DIR}/preflight.json" \
    || die "pre_flight_check has FAILs (not 0) — see ${OUT_DIR}/preflight.json"

# --- Main: --dry-run + actual launch ----------------------------------
MODE="actual"
case "${1:-}" in
    --dry-run)  MODE="dry-run" ;;
esac

echo "[$(date -Iseconds)] ${WRAPPER_NAME} mode=${MODE}"

if [ "${MODE}" = "dry-run" ]; then
    # Plan only — emit DRY_RUN.log + plan.json per Wf-363-A1 schema
    cat >"${OUT_DIR}/DRY_RUN.log" <<EOF
[DRY-RUN] ${WRAPPER_NAME} @ ${WF_ID}
[DRY-RUN] pre_flight_check PASS
[DRY-RUN] wrapper_name=${WRAPPER_NAME}
[DRY-RUN] wf_origin=${WF_ID}
[DRY-RUN] trainer=${TRAINER}
[DRY-RUN] py=${PY}
[DRY-RUN] args=${*:-}
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
EOF
    cat >"${OUT_DIR}/plan.json" <<EOF
{
    "wrapper": "${WRAPPER_NAME}",
    "wf_origin": "${WF_ID}",
    "mode": "dry-run",
    "user_ok_required": true,
    "trainer": "${TRAINER}",
    "py": "${PY}",
    "out_dir": "${OUT_DIR}",
    "pre_flight": "PASS",
    "args": [],
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"
}
EOF
    echo "[$(date -Iseconds)] ${WRAPPER_NAME} DRY-RUN logged to ${OUT_DIR}"
    exit 0
fi

# Actual launch
# Under `set -o pipefail` (line 31) a trainer that exits while DataLoader workers
# still hold the pipe open makes tee see EOF first and the trainer's next write
# take SIGPIPE; the pipeline then reports 141 rather than the trainer's own
# status, and `set -e` aborts here with no traceback — which is how BLK-C1 (bond
# head died at epoch 1/20) and BLK-D1 (3 of 5 seeds lost) failed. Keep the tee,
# it is the only diagnostic, but drop pipefail for this one command so the
# trainer's real status is readable from PIPESTATUS[0].
#
# `rc=${PIPESTATUS[0]}` on the line AFTER the pipeline is NOT sufficient: errexit
# fires at the pipeline and never reaches it (measured, 2026-10-09). The
# `set +o pipefail` below is load-bearing.
#
# `setsid --wait` is the OTHER half, and it is not cosmetic: without it the
# trainer stays in the invoking shell's session, so if that shell exits the
# trainer is SIGTERM'd with no traceback (BLK-I1 mechanism a). `--wait` is
# required rather than optional — bare `setsid` may fork and return 0 before the
# trainer has done anything, which would make `PIPESTATUS[0]` report success for
# a run that had not started. `--wait` propagates the trainer's own status.
set +o pipefail
PYTHONPATH=/home/hugo/codes/mol-metal setsid --wait ${PY} ${TRAINER} "$@" 2>&1 | tee "${OUT_DIR}/launch.log"
rc=${PIPESTATUS[0]}
set -o pipefail
echo "[$(date -Iseconds)] ${WRAPPER_NAME} exit=${rc}; log at ${OUT_DIR}/launch.log"
exit "${rc}"
