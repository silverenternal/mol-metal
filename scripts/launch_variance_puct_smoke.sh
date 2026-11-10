#!/bin/bash
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# launch_variance_puct_smoke.sh — AGENTS.md §5.7 50-step CPU smoke for the
# L3 VariancePUCT wire-in (R20 FIX 2).
#
# This is a CPU-only regression gate (per §5.7 rule 2), NOT a GPU retrain.
# It re-runnable: safe to invoke repeatedly, each run gets its own timestamped
# log under logs/variance_puct_smoke/.
#
# Usage:
#   USER_OK=1 bash scripts/launch_variance_puct_smoke.sh
#
# Exit codes: 0 = PASS, 1 = FAIL, 64 = missing USER_OK.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# 2026-10-08 pipefix: drop pipefail from `set -e` (SIGPIPE from forked
# DataLoader workers flushing stdout after trainer exit killed the wrapper
# silently via `tee`). rc captured below via ${PIPESTATUS[0]}.
#
# Wf-A269 wave-10 (2026-11-02): the rc capture was DEAD.  `|| true` runs
# AFTER the failing pipeline and CLOBBERS PIPESTATUS to (0), so this
# AGENTS.md §5.7 smoke gate could never FAIL even though its producer
# (scripts/variance_puct_smoke_invocation.py:150) exits non-zero on any
# failed check.  The repaired shape is at the call site below
# (scripts/_WRAPPER_TEMPLATE.sh:190-193).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# §5.9 single-venv rule: the canonical Mol-Metal venv, never bare python.
PY="$REPO_ROOT/.venv/bin/python"
SMOKE="$REPO_ROOT/scripts/variance_puct_smoke_invocation.py"
LOG_DIR="$REPO_ROOT/logs/variance_puct_smoke"

if [[ "${USER_OK:-0}" != "1" ]]; then
    echo "USER_OK gate (§5.7 / AGENTS.md): set USER_OK=1 to run the CPU smoke." >&2
    exit 64
fi

if [[ ! -x "$PY" ]]; then
    echo "FAIL: §5.9 single-venv python not found/executable: $PY" >&2
    exit 1
fi

if [[ ! -f "$SMOKE" ]]; then
    echo "FAIL: smoke invocation script not on disk: $SMOKE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${STAMP}.log"

cd "$REPO_ROOT"
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   WFA2_CONFIG_STRICT=1 "$PY" "$SMOKE" 2>&1 | tee "$LOG" || true   # 2026-10-08 pipefix
#   RC="${PIPESTATUS[0]}"
#
# Canonical shape (scripts/_WRAPPER_TEMPLATE.sh:190-193): pipefail off for
# this ONE pipeline (a trainer killed by SIGPIPE from a forked worker would
# otherwise report 141 instead of its own status) and the rc read BETWEEN the
# two `set` lines, while PIPESTATUS still holds the pipeline's statuses — so
# the FAIL branch below is reachable for a non-zero trainer exit.
set +e
set +o pipefail
WFA2_CONFIG_STRICT=1 "$PY" "$SMOKE" 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
set -o pipefail
set -e

if [[ "$RC" -ne 0 ]]; then
    echo "FAIL: variance PUCT smoke exited $RC (see $LOG)" >&2
    exit "$RC"
fi

echo "PASS: variance PUCT smoke exit 0; log written to $LOG"
exit 0