#!/usr/bin/env bash
# check_no_stale_venv.sh — assert the repo is pointed at the CANONICAL venv.
#
# AGENTS.md §5.9 (single-venv rule): the canonical interpreter is the
# repo-local /home/hugo/codes/mol-metal/.venv (derived here from this
# script's location as $REPO_ROOT/.venv). The legacy CadButEaas / kanzi
# venvs are FORBIDDEN. This guard FAILS when UV_PROJECT_ENVIRONMENT or
# VIRTUAL_ENV points at any interpreter other than the canonical .venv.
#
# Wire it up:
#   * pre-commit:   - repo: local; hooks: [{id: no-stale-venv, entry: bash molmetal/scripts/check_no_stale_venv.sh}]
#   * CI gate:      bash molmetal/scripts/check_no_stale_venv.sh
#   * manual:       bash molmetal/scripts/check_no_stale_venv.sh
#
# Contract: exits 0 (PASS) when the repo is canonical — both vars unset is
# canonical too, since uv then defaults to the repo-local .venv. Exits 1
# (FAIL, diagnostics on stderr) when a var resolves anywhere else.
# NEVER suggests deleting .venv — .venv IS canonical.

set -euo pipefail

# Resolve repo root from this script's location (molmetal/scripts/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CANONICAL_VENV="$REPO_ROOT/.venv"

RC=0
for VAR in UV_PROJECT_ENVIRONMENT VIRTUAL_ENV; do
    VALUE="${!VAR:-}"
    [ -n "$VALUE" ] || continue
    RESOLVED="$(realpath -m "$VALUE")"
    if [ "$RESOLVED" != "$CANONICAL_VENV" ]; then
        echo "FAIL: $VAR=$VALUE resolves to $RESOLVED, not the canonical venv $CANONICAL_VENV." >&2
        echo "  AGENTS.md §5.9: Mol-Metal uses the single repo-local .venv;" >&2
        echo "  the legacy CadButEaas / kanzi venvs are forbidden." >&2
        echo "  Fix: unset $VAR && source env.sh" >&2
        RC=1
    fi
done

if [ ! -x "$CANONICAL_VENV/bin/python" ]; then
    echo "WARN: canonical venv python missing at $CANONICAL_VENV/bin/python (run: uv sync)" >&2
fi

if [ "$RC" -eq 0 ]; then
    echo "OK: venv points at canonical $CANONICAL_VENV (UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-<unset>}, VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>})"
fi
exit "$RC"
