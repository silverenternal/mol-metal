#!/usr/bin/env bash
# scripts/_pre_flight_check.sh
# Root-level pre-flight check (bash mirror of molmetal/scripts/pre_flight_check.py).
# Per AGENTS.md §1.4: every launch wrapper MUST PASS this before any retrain / panel / ablation.
# Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP
#
# Usage:
#   bash scripts/_pre_flight_check.sh           # full check
#   bash scripts/_pre_flight_check.sh --fast   # skip pytest collect-only
#
# This script is intentionally minimal — it checks structural pre-conditions
# (Python interpreter, project importability, pre-flight gate, pytest collect).
# Detailed pre-flight gates live in molmetal/scripts/pre_flight_check.py
# (the canonical Python implementation per Wf-207 unified pre-flight checklist).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[pre-flight] project_root=$PROJECT_ROOT"
echo "[pre-flight] timestamp=$(date -Iseconds)"

# 1) Python interpreter reachable
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[pre-flight][FAIL] python interpreter $PYTHON not found on PATH"
  exit 1
fi
echo "[pre-flight][OK] python=$($PYTHON --version 2>&1 | head -1)"

# 2) Project venv reachable (if .venv exists)
if [[ -d .venv ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    echo "[pre-flight][FAIL] .venv/bin/python missing"
    exit 1
  fi
  echo "[pre-flight][OK] venv=.venv"
else
  echo "[pre-flight][SKIP] no .venv present"
fi

# 3) molmetal package importable
if "$PYTHON" -c "import molmetal" >/dev/null 2>&1; then
  echo "[pre-flight][OK] import molmetal"
else
  echo "[pre-flight][FAIL] cannot import molmetal"
  exit 1
fi

# 4) Pre-flight gate script present
if [[ ! -f molmetal/scripts/pre_flight_check.py ]]; then
  echo "[pre-flight][FAIL] molmetal/scripts/pre_flight_check.py missing"
  exit 1
fi
echo "[pre-flight][OK] pre_flight_check.py present"

# 5) Pytest collection OK (skipped if --fast)
if [[ "${1:-}" != "--fast" ]]; then
  if timeout 30 "$PYTHON" -m pytest --collect-only -q tests molmetal >/dev/null 2>&1; then
    echo "[pre-flight][OK] pytest collect-only"
  else
    echo "[pre-flight][FAIL] pytest collect-only failed"
    exit 1
  fi
fi

echo "[pre-flight] PASS"
exit 0
