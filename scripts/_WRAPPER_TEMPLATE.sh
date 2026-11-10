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
#     Wf-422-A2 + Wf-427-B1 cascade closures).  The gate RUNS on BOTH
#     paths; only the ABORT is path-conditional: fail-closed on the
#     launch path, report-only on the --dry-run path, whose plan.json +
#     DRY_RUN.log RECORD the pre-flight's verdict.
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
# Revised: 2026-11-02 — the pre-flight gate now RUNS on the --dry-run path
#   and its verdict is RECORDED in DRY_RUN.log + plan.json; the fail-closed
#   ABORT is kept on the launch path only.  Rationale: a dry run that dies at
#   the gate can never publish the plan that says "the pre-flight failed",
#   and that plan is the only artefact an operator can read to learn WHY a
#   launch would be refused.  The two pre-flight `die` MESSAGES are preserved
#   byte-identical from the pre-fix template (AGENTS.md §5.5 rule #6), as is
#   the SECOND die's whole statement; the FIRST die's GUARD moved from the
#   command's own `||` to an explicit test of the CAPTURED exit status,
#   because that status must now be RECORDED instead of aborting.
#   (Wf-A269 wave-10: a check that cannot fail is the defect; the mirror image
#   is a verdict that can never be read.)

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

# --- Main: --dry-run + actual launch ----------------------------------
# The mode is parsed BEFORE the pre-flight gate on purpose.  The dry run's
# job is to PRODUCE a plan that REPORTS the pre-flight's verdict — including
# the verdict "FAIL".  That is only possible if the gate is allowed to RUN and
# report on the dry-run path, so the mode has to be known before the gate
# runs.  (Wf-A269 wave-10: a check that cannot fail is the defect; the mirror
# image is a check whose verdict can never be READ.)
MODE="actual"
case "${1:-}" in
    --dry-run)  MODE="dry-run" ;;
esac

# --- Pre-flight gate (per Wf-415-A3 + Wf-427-B1) -----------------------
# The gate RUNS on BOTH paths: a dry run that skipped it would publish a plan
# claiming a verdict nobody measured.  Only the ABORT is conditional —
# fail-closed on the path that can actually launch a GPU, report-only on the
# dry-run path.  The two `die` MESSAGES below are byte-identical to the
# pre-fix template's (AGENTS.md §5.5 rule #6); they are simply reached only
# on the launch path.
#
# Wf-A269 wave-17 (2026-11-03) — FOUR repairs to this block.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# the pre-fix block captured the pre-flight's CONSOLE STREAM (stdout + stderr
# merged by `2>&1`) into a file NAMED `preflight.json`, then read the verdict
# out of it with `grep -q '"fail": 0'`.  Two defects lived in that shape:
#   (a) the file was not JSON at all — the gate log lines came first and the
#       payload the tool prints at the END was the last top-level `{...}` in
#       it (MEASURED on the published report: 190,245 B / 1,826 lines, first
#       `{` at line 270, `json.loads` => "Expecting value: line 1 column 2");
#   (b) the `grep` predicate was TIMEOUT-BLIND — it could not see a TIMEOUT
#       at all, so a report with 1 TIMEOUT and 0 FAILs MATCHED it and the
#       launch went ahead although the measurement was never obtained
#       (Wf-A135: a gate that timed out did not obtain its measurement).
#  (1) `--json PATH` writes a REAL JSON document (the tool writes it itself,
#      atomically, via AGENTS.md §5.6); the console transcript goes to a
#      sibling .log, so the report path stays pure JSON.
#  (2) the verdict comes from the repo's own consumer
#      (molmetal/scripts/wf_launch_gate_verdict.py), which is FAIL-CLOSED:
#      no summary / zero gates / a stale stamp / a foreign revision / an
#      edited payload / a TIMEOUT => FAIL, and an unreadable report => SKIP
#      (never PASS).
#  (3) the previous run's report is removed first, so a crashed or killed
#      pre-flight cannot leave a PASS report behind for the consumer to read
#      ("stale is treated as missing").
#  (4) the COUNTS are read from that pure-JSON report, not grepped out of a
#      transcript; an unreadable report yields "unknown", never 0.
PREFLIGHT_JSON="${OUT_DIR}/preflight.json"
PREFLIGHT_LOG="${OUT_DIR}/preflight.log"
mkdir -p "${OUT_DIR}"
rm -f "${PREFLIGHT_JSON}"
echo "[$(date -Iseconds)] ${WRAPPER_NAME} pre-flight check begin"
PREFLIGHT_RC=0
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch --json "${PREFLIGHT_JSON}" >"${PREFLIGHT_LOG}" 2>&1 \
    || PREFLIGHT_RC=$?

# The gate's own verdict, read from the report the gate just wrote.
PREFLIGHT_FAIL_COUNT="$(${PY} -c 'import json,sys;d=json.load(open(sys.argv[1],encoding="utf-8"));s=d.get("summary");print(int(s["fail"]) if isinstance(s,dict) and "fail" in s else "unknown")' "${PREFLIGHT_JSON}" 2>/dev/null || echo unknown)"
PREFLIGHT_FAIL_COUNT_JSON="null"
if [ "${PREFLIGHT_FAIL_COUNT}" != "unknown" ]; then
    PREFLIGHT_FAIL_COUNT_JSON="${PREFLIGHT_FAIL_COUNT}"
fi
PREFLIGHT_VERDICT="FAIL"
PREFLIGHT_VERDICT_LINE="the pre-flight report is missing or unreadable: ${PREFLIGHT_JSON}"
if [ -f "${PREFLIGHT_JSON}" ]; then
    PREFLIGHT_VERDICT_LINE="$(PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf_launch_gate_verdict.py preflight --report "${PREFLIGHT_JSON}" 2>&1 || true)"
    PREFLIGHT_VERDICT="${PREFLIGHT_VERDICT_LINE%% *}"
fi
echo "[$(date -Iseconds)] ${WRAPPER_NAME} pre-flight verdict=${PREFLIGHT_VERDICT} fail_count=${PREFLIGHT_FAIL_COUNT} rc=${PREFLIGHT_RC}"
# The reason is printed on its own line, so the two `die` MESSAGES below can
# stay byte-identical to the pre-fix template's (AGENTS.md §5.5 rule #6)
# while still telling the operator WHY the gate refused.
if [ "${PREFLIGHT_VERDICT}" != "PASS" ]; then
    printf '%s [WARN] pre_flight_check detail: %s\n' \
        "$(date -Iseconds 2>/dev/null || date)" "${PREFLIGHT_VERDICT_LINE}" >&2
fi

# Fail-closed ONLY on the path that can actually launch a GPU.  On the
# dry-run path the verdict is RECORDED (plan.json + DRY_RUN.log) and the dry
# run continues, so the operator can read a plan that says WHY the launch
# would be refused.
if [ "${MODE}" != "dry-run" ]; then
    [ "${PREFLIGHT_RC}" -eq 0 ] || die "pre_flight_check FAIL — see ${PREFLIGHT_JSON}"
    [ "${PREFLIGHT_VERDICT}" = "PASS" ] \
        || die "pre_flight_check has FAILs (not 0) — see ${PREFLIGHT_JSON}"
elif [ "${PREFLIGHT_VERDICT}" != "PASS" ]; then
    printf '%s [WARN] pre_flight_check %s (fail_count=%s) — the dry run CONTINUES and records the verdict; the REAL launch path would abort here. See %s\n' \
        "$(date -Iseconds 2>/dev/null || date)" "${PREFLIGHT_VERDICT}" "${PREFLIGHT_FAIL_COUNT}" "${PREFLIGHT_JSON}" >&2
fi

echo "[$(date -Iseconds)] ${WRAPPER_NAME} mode=${MODE}"

if [ "${MODE}" = "dry-run" ]; then
    # Plan only — emit DRY_RUN.log + plan.json per Wf-363-A1 schema
    cat >"${OUT_DIR}/DRY_RUN.log" <<EOF
[DRY-RUN] ${WRAPPER_NAME} @ ${WF_ID}
[DRY-RUN] pre_flight_check ${PREFLIGHT_VERDICT}
[DRY-RUN] pre_flight_fail_count=${PREFLIGHT_FAIL_COUNT}
[DRY-RUN] pre_flight_rc=${PREFLIGHT_RC}
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
    "pre_flight": "${PREFLIGHT_VERDICT}",
    "pre_flight_fail_count": ${PREFLIGHT_FAIL_COUNT_JSON},
    "pre_flight_rc": ${PREFLIGHT_RC},
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
