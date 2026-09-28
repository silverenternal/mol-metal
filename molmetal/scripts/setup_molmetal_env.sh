#!/usr/bin/env bash
# setup_molmetal_env.sh — single-command env activator for Mol-Metal.
#
# Wf-23 Phase 3 (2026-09-26): drop-in replacement for the manual
# `source env.sh && uv run …` sequence so R18 retrain jobs can run without
# first activating the parent shell. Mirrors env.sh verbatim for the env
# vars that are load-bearing for the R18 modules, then exec's the user
# command under the shared .venv-vlm Python so torch/CUDA imports work.
#
# Usage:
#   bash molmetal/scripts/setup_molmetal_env.sh python -c "import molmetal"
#   bash molmetal/scripts/setup_molmetal_env.sh python -m pytest molmetal/tests/test_env_setup.py -v
#   bash molmetal/scripts/setup_molmetal_env.sh ./molmetal/scripts/train_ru_multitask.py --help
#
# What it does (idempotent, safe to chain with env.sh):
#   1. Resolves the repo root + symlinked venv (molmetal/.venvs/mol-metal-venv).
#   2. Adds the repo root to PYTHONPATH so `import molmetal` resolves.
#   3. Mirrors env.sh: PYTORCH_CUDA_ALLOC_CONF, MOLMETAL_CFM_AMP[_DTYPE],
#      MOLMETAL_EGNN_CHECKPOINT, TOKENIZERS_PARALLELISM, TRANSFORMERS_VERBOSITY,
#      MOLMETAL_DATA_DIR, MOLMETAL_TMQM_DIR. CUDA_VISIBLE_DEVICES defaults to 0;
#      export CUDA_VISIBLE_DEVICES=1 before invocation to use the RTX 5090.
#   4. Execs the shared venv's Python (or whatever interpreter is passed
#      as $1) so torch/rdkit/openbabel imports resolve from
#      /home/hugo/codes/CadButEaas/.venv-vlm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_SYMLINK="$REPO_ROOT/molmetal/.venvs/mol-metal-venv"
SHARED_VENV="/home/hugo/codes/CadButEaas/.venv-vlm"

if [ ! -e "$VENV_SYMLINK" ]; then
    echo "ERROR: $VENV_SYMLINK missing." >&2
    echo "  Run: ln -snf $SHARED_VENV $VENV_SYMLINK" >&2
    exit 2
fi
if [ ! -x "$SHARED_VENV/bin/python" ]; then
    echo "ERROR: $SHARED_VENV/bin/python not executable." >&2
    exit 2
fi

# --- PYTHONPATH: prepend repo root so `import molmetal` works.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# --- Mol-Metal defaults (mirror env.sh; export only if not already set so
#     callers can still override per-job, e.g. CUDA_VISIBLE_DEVICES=1).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MOLMETAL_CFM_AMP="${MOLMETAL_CFM_AMP:-1}"
export MOLMETAL_CFM_AMP_DTYPE="${MOLMETAL_CFM_AMP_DTYPE:-bfloat16}"
export MOLMETAL_EGNN_CHECKPOINT="${MOLMETAL_EGNN_CHECKPOINT:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-$REPO_ROOT/.data/molmetal}"
export MOLMETAL_TMQM_DIR="${MOLMETAL_TMQM_DIR:-$REPO_ROOT/.data/molmetal/tmQM}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VIRTUAL_ENV="$SHARED_VENV"

# --- Pick interpreter: $1 if it looks like a python-ish binary, else shared venv.
EXEC_PY="${SHARED_VENV}/bin/python"
if [ "${1:-}" = "python" ] || [ "${1:-}" = "python3" ]; then
    shift
fi

exec "$EXEC_PY" "$@"
