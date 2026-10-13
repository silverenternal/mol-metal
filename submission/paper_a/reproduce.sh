#!/usr/bin/env bash
# Paper-A reproducibility package — CPU subset driver.
#
# Regenerates the CPU-reproducible Paper-A metrics into ./.repro_out and diffs
# each against the shipped artifact in metrics/by_metric/. Never overwrites a
# shipped artifact. Exit code 0 iff every executed entry matches.
#
# Usage:
#   ./reproduce.sh                 # run the CPU subset
#   ./reproduce.sh --dry-run       # print the plan, run nothing
#   ./reproduce.sh --only pa3_chemprop_lomo_probe_cohort
#   ./reproduce.sh --include-gpu-long   # also the GPU entries (hours)
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Interpreter: honour $PYTHON, else a local .venv, else python3.
if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif [[ -x "${PKG_ROOT}/.venv/bin/python" ]]; then
  PY="${PKG_ROOT}/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  echo "FAIL: no python interpreter (set \$PYTHON or create ./.venv)" >&2
  exit 2
fi

# Raw data defaults (see data/README.md).  Both are overridable.
export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-${PKG_ROOT}/.data/molmetal}"
export MOLMETAL_TMQM_DIR="${MOLMETAL_TMQM_DIR:-${PKG_ROOT}/.data/molmetal/tmQM}"

cd "${PKG_ROOT}"
exec "${PY}" scripts/reproduce_paper_a.py \
  --folds-dir "${PKG_ROOT}/folds" \
  --out-dir "${PKG_ROOT}/.repro_out" \
  "$@"
