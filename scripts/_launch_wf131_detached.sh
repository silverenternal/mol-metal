#!/usr/bin/env bash
# _launch_wf131_detached.sh — detached runner for launch_wf131_p3_pt_4cells.sh
#
# Per Wf-R43 substantive fix: the wrapper bash itself died after 1 pair when
# launched as a Claude Code background bash (SIGHUP from parent tmux pane).
# The right pattern is `setsid nohup` + redirect to log + new session, so the
# wrapper is fully detached from any parent process group.
#
# Usage:
#   USER_OK=1 bash scripts/_launch_wf131_detached.sh --pair "MCF-7 0"   # debug pair
#   USER_OK=1 bash scripts/_launch_wf131_detached.sh                       # all 20 pairs
#   USER_OK=1 bash scripts/_launch_wf131_detached.sh --dry-run            # dry-run plan only
#
# After launching, monitor:
#   tail -f /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_pt_4cells/master.log
#   ls /home/hugo/codes/mol-metal/molmetal/reports/wf131_p3_pt_4cells/Pt/*/seed*/.DONE
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPPER="$REPO_ROOT/scripts/launch_wf131_p3_pt_4cells.sh"
LOG_DIR="$REPO_ROOT/molmetal/reports/wf131_p3_pt_4cells"
DETACHED_LOG="${LOG_DIR}/detached_launcher.log"

mkdir -p "${LOG_DIR}"

if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    exit 64
fi

if [ ! -x "${WRAPPER}" ]; then
    printf '%s[FATAL]%s wrapper missing or not executable: %s\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" "${WRAPPER}" >&2
    exit 5
fi

# Canonical invocation per Wf-R43: setsid (new session) + nohup (ignore SIGHUP)
# + </dev/null (no stdin tie) + tee to detached_launcher.log (live observability).
#
# The wrapper itself also writes to master.log via its own tee chain.
# Both logs are independent and the wrapper's master.log is the canonical record.
printf '[%s] launching wrapper detached\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${DETACHED_LOG}"
setsid nohup bash "${WRAPPER}" "$@" </dev/null >>"${DETACHED_LOG}" 2>&1 &
WRAPPER_PID=$!
disown

printf '%s[OK]%s wrapper launched detached, PID=%s\n' \
    "$(printf '\033[0;32m')" "$(printf '\033[0m')" "${WRAPPER_PID}" >&2
printf '       monitor: tail -f %s/master.log\n' "${LOG_DIR}" >&2
printf '       monitor: tail -f %s\n' "${DETACHED_LOG}" >&2
printf '       resume-safety: %ss-mol-metal/molmetal/reports/wf131_p3_pt_4cells/Pt/*/seed*/.DONE files\n' "${REPO_ROOT}" >&2