#!/usr/bin/env bash
# WF-IMPROVEMENT-BACKLOG I2 — Run all 4 subsystem smoke scripts sequentially.
#
# Goal
# ====
# Single entry point that runs every subsystem smoke (CFM, Lambda,
# evaluators, pitfalls) sequentially and reports a final aggregate
# verdict.  Designed for the dispatch harness / ultracode scheduler.
#
# What it does
# ------------
# 1. Run ``smoke_cfm.sh``       — CFM 200-step train + 4 decode probes.
# 2. Run ``smoke_lambda.sh``    — 10-pocket × 1-seed Lambda MCTS pilot.
# 3. Run ``smoke_evaluators.sh`` — all 14 evaluators PASS check.
# 4. Run ``smoke_pitfalls.sh``  — all 7 R12-R17 pitfall guards PASS.
#
# Each sub-smoke has a 120-180 s per-script timeout (mirrored from the
# IMPROVEMENT_BACKLOG §I2 wall-time budget).  Total wall-time budget is
# ~3 min on CPU — well under the 2 min per-script + sequencing slack.
#
# Output
# ------
# Each line is ``<smoke_id> PASS|FAIL <details>``.  The script then prints
# one final ``ALL_SMOKE TOTAL_PASS=X/4 TOTAL_FAIL=Y/4 wall=<s>s`` line.
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/smoke/all.sh
#
# Exit code
# ---------
# 0 — all 4 smokes PASS.
# 1 — at least one smoke FAILed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Per-script timeout (seconds).  Each smoke already enforces a hard ceiling
# on its own; this is just a backstop so a runaway sub-smoke cannot
# lock the dispatch harness indefinitely.
PER_SMOKE_TIMEOUT="${PER_SMOKE_TIMEOUT:-180}"

declare -a SMOKE_NAMES=("cfm" "lambda" "evaluators" "pitfalls")
declare -a SMOKE_RESULTS=()
declare -a SMOKE_WALLS=()

T_TOTAL_START=$(date +%s)

for name in "${SMOKE_NAMES[@]}"; do
    script="${HERE}/smoke_${name}.sh"
    if [[ ! -x "${script}" ]]; then
        SMOKE_RESULTS+=("FAIL no_exec")
        SMOKE_WALLS+=("0")
        echo "${name} FAIL no_exec (script missing or not executable at ${script})"
        continue
    fi
    T_SUB_START=$(date +%s)
    echo "==================================================================="
    echo "[all.sh] running smoke_${name}.sh (timeout=${PER_SMOKE_TIMEOUT}s)"
    echo "==================================================================="
    set +e
    timeout "${PER_SMOKE_TIMEOUT}" bash "${script}"
    RC=$?
    set -e
    T_SUB_END=$(date +%s)
    ELAPSED=$((T_SUB_END - T_SUB_START))
    if [[ ${RC} -eq 0 ]]; then
        SMOKE_RESULTS+=("PASS")
        SMOKE_WALLS+=("${ELAPSED}")
        echo "[all.sh] smoke_${name}.sh -> PASS in ${ELAPSED}s"
    else
        SMOKE_RESULTS+=("FAIL rc=${RC}")
        SMOKE_WALLS+=("${ELAPSED}")
        echo "[all.sh] smoke_${name}.sh -> FAIL rc=${RC} in ${ELAPSED}s"
    fi
done

T_TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((T_TOTAL_END - T_TOTAL_START))

# Tally.
N_PASS=0
N_FAIL=0
for r in "${SMOKE_RESULTS[@]}"; do
    if [[ "${r}" == "PASS" ]]; then
        N_PASS=$((N_PASS + 1))
    else
        N_FAIL=$((N_FAIL + 1))
    fi
done

echo "==================================================================="
echo "ALL_SMOKE SUMMARY"
echo "==================================================================="
for i in "${!SMOKE_NAMES[@]}"; do
    printf "  %-12s %-15s %4ss\n" "${SMOKE_NAMES[$i]}" "${SMOKE_RESULTS[$i]}" "${SMOKE_WALLS[$i]}"
done
echo "-------------------------------------------------------------------"
echo "ALL_SMOKE TOTAL_PASS=${N_PASS}/4 TOTAL_FAIL=${N_FAIL}/4 wall=${TOTAL_ELAPSED}s"
echo "==================================================================="

if [[ ${N_PASS} -eq 4 ]]; then
    exit 0
fi
exit 1
