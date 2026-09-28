#!/usr/bin/env bash
# WF-IMPROVEMENT-BACKLOG I2 — Evaluator subsystem smoke (14 evaluators × 4–5 test molecules).
#
# Goal
# ====
# Verify that every evaluator used by the paper §4 evaluation suite is
# importable + functional on this device.  Runs the canonical
# ``eval_all_evaluators.py`` script and parses its ``SUMMARY: 14/14``
# line to drive the PASS/FAIL decision.
#
# Why this matters
# ----------------
# 14 evaluators are involved in the §4 metrics: SA, QED, Vina adapter,
# Vina binding, Meeko, PoseBusters, Bio.PDB (PLIP replacement), OpenBabel,
# ProDy, plus 4 reference-drug sanity checks.  If any one regresses (a
# missing import, a binding failure on this device's Python env) the
# entire §4 table collapses.  This smoke catches the regression in <30 s.
#
# What it does
# ------------
# 1. Run ``eval_all_evaluators.py`` via the shared venv (mirrors env.sh).
# 2. Parse the trailing ``SUMMARY: X/14 evaluators PASS`` line.
# 3. PASS iff X == 14.
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/smoke/smoke_evaluators.sh
#
# Exit code
# ---------
# 0 — all 14 evaluators PASS.
# 1 — fewer than 14 evaluators PASS, or the runner crashed.
#
# Wall-time budget
# ----------------
# 14 import + sanity checks ≈ 5-15 s on CPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Locate the shared venv (mirrors env.sh convention).
SHARED_VENV="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
if [[ -x "${SHARED_VENV}" ]]; then
    PY="${SHARED_VENV}"
else
    PY="${PYTHON:-python}"
fi

echo "[smoke_evaluators] Launching eval_all_evaluators.py" >&2
echo "[smoke_evaluators] (14 evaluators: SA / QED / Vina / Meeko / PoseBusters / Bio.PDB / OpenBabel / ProDy / ...)" >&2

T_START=$(date +%s)
LOG="$(mktemp /tmp/molmetal_smoke_evaluators.XXXXXX.log)"
trap 'rm -f "${LOG}"' EXIT

set +e
"${PY}" -u molmetal/scripts/eval_all_evaluators.py > "${LOG}" 2>&1
RC=$?
set -e
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

if [[ ${RC} -ne 0 ]]; then
    echo "[smoke_evaluators] FAIL runner_crashed rc=${RC} after ${ELAPSED}s" >&2
    echo "[smoke_evaluators] tail of log:" >&2
    tail -30 "${LOG}" >&2 || true
    exit 1
fi

# Parse the canonical summary line: ``SUMMARY: 14/14 evaluators PASS``.
SUMMARY="$(grep -E '^SUMMARY: [0-9]+/[0-9]+ evaluators PASS' "${LOG}" | tail -1 || true)"
if [[ -z "${SUMMARY}" ]]; then
    echo "[smoke_evaluators] FAIL no_summary: SUMMARY line not produced" >&2
    echo "[smoke_evaluators] tail of log:" >&2
    tail -20 "${LOG}" >&2 || true
    exit 1
fi

# Extract the X/Y pair and the post-summary verdict.
PASS_COUNT="$(echo "${SUMMARY}" | awk '{print $2}' | cut -d/ -f1)"
TOTAL_COUNT="$(echo "${SUMMARY}" | awk '{print $2}' | cut -d/ -f2)"
VERDICT_LINE="$(tail -3 "${LOG}" | grep -E 'All evaluators ready|evaluators need attention' || true)"

echo "[smoke_evaluators] summary=${SUMMARY} wall=${ELAPSED}s" >&2

# PASS iff 14/14 evaluators PASS (per IMPROVEMENT_BACKLOG §I2 spec).
if [[ "${PASS_COUNT}" == "14" && "${TOTAL_COUNT}" == "14" ]]; then
    echo "EVALUATORS_SMOKE PASS ${PASS_COUNT}/${TOTAL_COUNT} evaluators PASS wall=${ELAPSED}s"
    exit 0
fi
echo "EVALUATORS_SMOKE FAIL ${PASS_COUNT}/${TOTAL_COUNT} evaluators PASS (expected 14/14)" >&2
echo "${VERDICT_LINE}" >&2
exit 1
