#!/usr/bin/env bash
# Paper-A reproducibility driver — thin wrapper around scripts/reproduce_paper_a.py.
#
# Regenerates the CPU-reproducible Paper-A metrics into a scratch dir and diffs
# them against the on-disk artifacts. Never writes to metrics/by_metric/.
# See paper_a/repro/README.md for the manifest and the cost classes.
#
# Usage:
#   scripts/reproduce_paper_a.sh                # CPU subset
#   scripts/reproduce_paper_a.sh --dry-run      # print the plan
#   scripts/reproduce_paper_a.sh --include-gpu-long
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "FAIL: expected venv interpreter at ${PY} (AGENTS.md §5.9 single-venv)" >&2
  exit 2
fi

export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-${REPO_ROOT}/.data/molmetal}"
export MOLMETAL_TMQM_DIR="${MOLMETAL_TMQM_DIR:-${REPO_ROOT}/.data/molmetal/tmQM}"

cd "${REPO_ROOT}"
exec "${PY}" scripts/reproduce_paper_a.py "$@"
