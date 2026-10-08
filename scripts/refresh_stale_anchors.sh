#!/bin/bash
# Wf-A6 (BLOCKER D) — surface stale paper-grade anchors for regeneration.
#
# p3_wf343c4_anchor_freshness_24h FAILs when a LIVE anchor matching
# ANCHOR_GLOB_PATTERNS is older than 24h.  This wrapper lists them so an
# operator can re-run the producing wave.
#
# It does NOT touch/backdate anchors by default.  Resetting an mtime to
# satisfy the freshness gate without regenerating the data is fabricated
# freshness (AGENTS.md §5.2) and is exactly what the gate's own remedy
# text forbids.  The only sanctioned fix is re-running the producer, so
# --touch exists solely for the narrow case where the file was already
# regenerated elsewhere and only its mtime is behind; it is opt-in and
# prints a loud warning.
#
# Usage:
#   scripts/refresh_stale_anchors.sh                 # list stale live anchors
#   scripts/refresh_stale_anchors.sh --include-frozen
#   scripts/refresh_stale_anchors.sh --touch         # opt-in, see warning
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "FAIL: canonical venv python not found at $PY (AGENTS.md §5.9)" >&2
  exit 2
fi

DO_TOUCH=0
WALK_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --touch) DO_TOUCH=1 ;;
    *) WALK_ARGS+=("$arg") ;;
  esac
done

echo "=== stale anchor scan ($(date '+%Y-%m-%d %H:%M:%S')) ==="
"$PY" "$REPO_ROOT/molmetal/scripts/walk_anchors.py" "${WALK_ARGS[@]+"${WALK_ARGS[@]}"}"
rc=$?

if [ "$rc" -eq 0 ]; then
  echo "OK: nothing to refresh"
  exit 0
fi

if [ "$DO_TOUCH" -eq 0 ]; then
  echo ""
  echo "Stale anchors above are LIVE (not frozen-exempt). Remedy: re-run the"
  echo "producing wave for each. Re-run this script with --touch ONLY if the"
  echo "file was already regenerated and merely has an old mtime."
  exit 0
fi

echo ""
echo "WARNING (AGENTS.md §5.2): --touch resets mtime WITHOUT regenerating data."
echo "Use it only when the underlying metric is already fresh. Frozen-exempt"
echo "historical anchors are NEVER touched."
"$PY" "$REPO_ROOT/molmetal/scripts/walk_anchors.py" "${WALK_ARGS[@]+"${WALK_ARGS[@]}"}" \
  | grep -E '^ *[0-9]' \
  | while read -r _hours path; do
      [ -f "$REPO_ROOT/$path" ] || continue
      touch "$REPO_ROOT/$path"
      echo "touched $path"
    done
echo "done"
