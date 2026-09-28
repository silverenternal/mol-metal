#!/usr/bin/env bash
# check_no_stale_venv.sh — fail if mol-metal/.venv exists.
#
# Mol-Metal routes uv to the shared /home/hugo/codes/CadButEaas/.venv-vlm via
# UV_PROJECT_ENVIRONMENT (set in env.sh). A stale mol-metal/.venv will shadow
# that shared env when `uv run` auto-detects it. This guard fails loudly so
# CI / pre-commit catches regressions.
#
# Wire it up:
#   * pre-commit:   - repo: local; hooks: [{id: no-stale-venv, entry: bash molmetal/scripts/check_no_stale_venv.sh}]
#   * CI gate:      bash molmetal/scripts/check_no_stale_venv.sh
#   * manual:       bash molmetal/scripts/check_no_stale_venv.sh
#
# Always exits 0 when no stale venv is present; exits 1 (with diagnostics)
# when mol-metal/.venv is found.

set -euo pipefail

# Resolve repo root from this script's location (molmetal/scripts/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STALE_VENV="$REPO_ROOT/.venv"

if [ -d "$STALE_VENV" ]; then
    echo "FAIL: stale venv detected at $STALE_VENV" >&2
    echo "  mol-metal routes uv to the shared CadButEaas env via" >&2
    echo "  UV_PROJECT_ENVIRONMENT (see env.sh). A local .venv here will" >&2
    echo "  shadow it and break the CUDA/torch build assumptions." >&2
    echo "  Remove it: rm -rf $STALE_VENV" >&2
    exit 1
fi

# Also fail if the parent shell leaked a VIRTUAL_ENV that points at this
# repo's .venv — defensive against the exact bug this guard exists for.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" = "$STALE_VENV" ]; then
    echo "FAIL: VIRTUAL_ENV=$VIRTUAL_ENV points at the stale local .venv." >&2
    echo "  Run: unset VIRTUAL_ENV && source env.sh" >&2
    exit 1
fi

echo "OK: no stale mol-metal/.venv (VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>})"
exit 0