#!/usr/bin/env bash
# WF-IMPROVEMENT-BACKLOG I2 — Pitfall-guard subsystem smoke (7 R12-R17 guards).
#
# Goal
# ====
# Verify every CI guard rail for the 7 infrastructure-level pitfalls
# that bit the project between Round 12 and Round 17 is still in place.
# This is the regression-detection canary for the R17 close-out.
#
# Why this matters
# ----------------
# Each pitfall guard is a parametrized pytest test that fires (FAILS)
# when the *current* code state no longer protects against the
# corresponding regression.  If a future refactor reverts one of the
# fixes, the guard must catch it before the regression ships.
#
# What it does
# ------------
# 1. Run ``wf_pitfall_guard/check_all.py`` (the canonical
#    ``--format human`` invocation, default rc-based contract).
# 2. Parse the human-readable table to count ``PASS`` entries.
# 3. Parse the trailing ``PASS=N FAIL=M SKIP=K`` summary line.
# 4. PASS iff PASS == 7 (no FAIL, no SKIP-on-the-good-state).
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/smoke/smoke_pitfalls.sh
#
# Exit code
# ---------
# 0 — all 7 guards PASS.
# 1 — fewer than 7 guards PASS, or a guard FAILED, or the runner
#     crashed.
#
# Wall-time budget
# ----------------
# 7 parametrized pytest guards × ~5 s each ≈ 30-60 s on CPU.

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

echo "[smoke_pitfalls] Launching wf_pitfall_guard/check_all.py" >&2
echo "[smoke_pitfalls] (7 R12-R17 guard rails: P1 OOV atoms / P2 SFN shim / P3 MetalLigandExchange / P4 BF16 clamp / P5 SAFETY_MAX / P6 reference clone / P7 ModelWrapper kwonly)" >&2

T_START=$(date +%s)
LOG="$(mktemp /tmp/molmetal_smoke_pitfalls.XXXXXX.log)"
trap 'rm -f "${LOG}"' EXIT

# The check_all.py runner uses pytest under the hood.  We prefer
# ``pytest`` (lands in the shared venv when uv has set PATH) but fall
# back to ``python -m pytest`` for plain-Python environments.
set +e
"${PY}" -u molmetal/scripts/wf_pitfall_guard/check_all.py > "${LOG}" 2>&1
RC=$?
set -e
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

if [[ ${RC} -ne 0 ]]; then
    # Even on rc != 0 we want to inspect the table — a single FAIL should
    # still be reported as FAIL with the count, not as a runner crash.
    SUMMARY="$(grep -E '^PASS=[0-9]+ FAIL=[0-9]+ SKIP=[0-9]+' "${LOG}" | tail -1 || true)"
    if [[ -z "${SUMMARY}" ]]; then
        echo "[smoke_pitfalls] FAIL runner_crashed rc=${RC} after ${ELAPSED}s" >&2
        echo "[smoke_pitfalls] tail of log:" >&2
        tail -30 "${LOG}" >&2 || true
        exit 1
    fi
fi

# Parse the trailing PASS= FAIL= SKIP= line emitted by check_all._format_table.
SUMMARY="$(grep -E '^PASS=[0-9]+ FAIL=[0-9]+ SKIP=[0-9]+' "${LOG}" | tail -1 || true)"
if [[ -z "${SUMMARY}" ]]; then
    echo "[smoke_pitfalls] FAIL no_summary: PASS/FAIL/SKIP line not produced" >&2
    echo "[smoke_pitfalls] tail of log:" >&2
    tail -30 "${LOG}" >&2 || true
    exit 1
fi

PASS_COUNT="$(echo "${SUMMARY}" | sed -nE 's/^PASS=([0-9]+).*/\1/p')"
FAIL_COUNT="$(echo "${SUMMARY}" | sed -nE 's/^PASS=[0-9]+ FAIL=([0-9]+).*/\1/p')"
SKIP_COUNT="$(echo "${SUMMARY}" | sed -nE 's/^PASS=[0-9]+ FAIL=[0-9]+ SKIP=([0-9]+).*/\1/p')"

echo "[smoke_pitfalls] summary=${SUMMARY} rc=${RC} wall=${ELAPSED}s" >&2

# PASS iff PASS == 7 (per IMPROVEMENT_BACKLOG §I2 spec).
if [[ "${PASS_COUNT}" == "7" && "${FAIL_COUNT}" == "0" ]]; then
    echo "PITFALLS_SMOKE PASS ${PASS_COUNT}/7 guards PASS (FAIL=${FAIL_COUNT}, SKIP=${SKIP_COUNT}) wall=${ELAPSED}s"
    exit 0
fi
echo "PITFALLS_SMOKE FAIL ${PASS_COUNT}/7 guards PASS (FAIL=${FAIL_COUNT}, SKIP=${SKIP_COUNT})" >&2
echo "[smoke_pitfalls] tail of log:" >&2
tail -30 "${LOG}" >&2 || true
exit 1
