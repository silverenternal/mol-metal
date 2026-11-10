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

# ---------------------------------------------------------------------------
# CPU LOCK (AGENTS.md §5.1 + the repo's standing 0-GPU session contract)
#
# MEASURED 2026-11-04: running this script WITHOUT the lock allocated
# 694 MiB of GPU memory per process, while the smoke itself ran on CPU.
# The cause is the IMPORT CHAIN, not the smoke: it reaches
# ``torch.cuda.get_device_capability()``
# (molmetal/adapters/flow_matching_lipman/amp.py:187), and
# ``torch.cuda.is_available()`` initialises the CUDA driver (``cuInit``),
# which creates the context.  This is the SAME class Wf-A162 repaired
# inside the pre-flight (molmetal/scripts/pre_flight_check.py::_smoke_device)
# -- but these smoke scripts carried no lock of their own, so a caller who
# forgot it silently paid the allocation (this repo's own session did,
# twice, before the lock existed).
#
# These smokes are CPU-BY-DESIGN (each script's own header budgets the run
# as "<2 min on CPU"), so the lock is the DEFAULT rather than a caller
# obligation: an unset ``CUDA_VISIBLE_DEVICES`` becomes "" (no device
# visible), and an explicit value is PRESERVED so a deliberate GPU smoke
# is still possible.  ``MOLMETAL_SMOKE_DEVICE=cpu`` is the repo's parity
# kill-switch (honoured by ``metallo_drug_smoke_retrain.py::_resolve_device``).
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export MOLMETAL_SMOKE_DEVICE="${MOLMETAL_SMOKE_DEVICE:-cpu}"

# Locate the canonical repo-local venv (AGENTS.md §5.9 single-venv rule).
SHARED_VENV="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${SHARED_VENV}" ]]; then
    PY="${SHARED_VENV}"
elif [[ -n "${PYTHON-}" && -x "${PYTHON}" ]]; then
    # An EXPLICIT override is honoured; a BARE `python` is not (AGENTS.md
    # §5.9: the smoke must run under the repo's single venv, never under an
    # ambient interpreter that may be another project's -- `kanzi_venv`,
    # `CadButEaas`, or a system python with a different torch).
    PY="${PYTHON}"
else
    echo "[smoke] FATAL: no interpreter." >&2
    echo "[smoke]   expected the repo venv: ${SHARED_VENV}" >&2
    echo "[smoke]   (missing or not executable) -- AGENTS.md §5.9 forbids a" >&2
    # Wf-A269 wave-26 (2026-11-04) — THE MESSAGE MENTIONS AN INTERPRETER
    # PATH, AND THE REPO'S OWN §5.9 GATE READS MENTIONS AS USE.
    # `scripts/_wrapper_audit.py::interpreter_census` adjudicates a LITERAL
    # absolute interpreter path wherever it appears on a CODE line (comments
    # are stripped, but a path inside an `echo` ARGUMENT is not a comment).
    # The message below therefore tripped the gate:
    #   VIOLATION molmetal/scripts/smoke/<smoke>.sh:<line> [/abs/path/to/python]
    #   -- §5.9 forbids an interpreter outside the repo venv
    # MEASURED 2026-11-04: 4 files / 4 violations, all from THIS message.
    # The gate is CORRECT as documented ("the rule is LITERAL by construction:
    # an interpreter path is adjudicated when it is written out in full") — the
    # defect was the MESSAGE, not the gate.  => the message now names the
    # override VARIABLE without writing out a literal absolute path, and the
    # gate returns to 0 violations.  The gate's own imprecision (it cannot
    # tell an interpreter INVOCATION from an interpreter MENTION) is
    # disclosed separately, not silently reworded away.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   echo "[smoke]   bare \`python\`.  Set PYTHON=/abs/path/to/python to" >&2
    echo "[smoke]   bare \`python\`.  Set PYTHON=<absolute python path> to" >&2
    echo "[smoke]   override explicitly, or create the repo venv." >&2
    exit 1
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
