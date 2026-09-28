#!/usr/bin/env bash
# WF-R17-Ablation-Runner — bash driver for the 11 R17 mechanism ablations
# declared in ``TODO/MECHANISM_METRIC_MAPPING.md`` §5.
#
# Why this script exists
# =======================
# The R17 paper requires an ablation table that shows which mechanism is
# responsible for which metric movement.  §5 enumerates 11 ablations,
# the metric that is expected to regress, and the workflow that owns the
# experiment.
#
# This bash driver is a thin wrapper around ``ablation_runner.py`` — it
# sources the shared ``env.sh`` (so torch / rdkit / openbabel /
# posebusters are importable), forwards the per-ablation toggles into
# the Python runner, and prints a single-line status at the end so the
# dispatch harness can grep for it.
#
# What it does
# ------------
# 1. Sources ``env.sh`` so the runner inherits MOLMETAL_DATA_DIR,
#    UV_PROJECT_ENVIRONMENT, MOLMETAL_CFM_AMP, etc.
# 2. Calls ``uv run --no-sync python molmetal/scripts/ablation_runner.py``
#    with the supplied CLI args (steps, decode-samples, --only).
# 3. Streams stdout/stderr to the console for live progress.
#
# Outputs (all under ``molmetal/reports/wf_ablation_runner/``)
# ------------------------------------------------------------
# * ``<ablation_id>.json`` — per-ablation metrics + status.
# * ``aggregate.csv``      — 11-row × 9-col matrix for paper §6.
# * ``final.md``           — human-readable verdict + table.
#
# Wall-budget on GPU: ~30 min (200 steps × 5 toggleable ablations ×
#                       ~6 min each).
# Wall-budget on CPU: ~3 h (10× slower; use --steps 50 for fast demo).
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     source env.sh
#     bash molmetal/scripts/run_ablation_smoke.sh
#
# Or to run a subset (faster smoke):
#     bash molmetal/scripts/run_ablation_smoke.sh --steps 50 --only M13,M14,M15
#
# Exit code
# ---------
# 0 — every toggleable ablation completed (rc=0); design-only ablations
#     are documented but not re-run.
# 1 — at least one toggleable ablation subprocess failed; see
#     ``final.md`` for the per-ablation status table.

set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve repo root and source env.sh (paths / CUDA / shared venv).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_SH="${REPO_ROOT}/env.sh"
if [[ ! -f "${ENV_SH}" ]]; then
    echo "[run_ablation_smoke] FATAL: env.sh not found at ${ENV_SH}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${ENV_SH}"
echo "[run_ablation_smoke] REPO_ROOT=${REPO_ROOT}" >&2
echo "[run_ablation_smoke] python=$(command -v python || true)" >&2
echo "[run_ablation_smoke] uv=$(command -v uv || true)" >&2

# ---------------------------------------------------------------------------
# 1. Verify the harness script + training CSV exist.
# ---------------------------------------------------------------------------
ABLATION_RUNNER="${SCRIPT_DIR}/ablation_runner.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/metallo_drug_smoke_retrain.py"
TRAIN_CSV="${REPO_ROOT}/molmetal/data/metallo_drugs_500_train.csv"
OUT_DIR="${REPO_ROOT}/molmetal/reports/wf_ablation_runner"

for path in "${ABLATION_RUNNER}" "${TRAIN_SCRIPT}" "${TRAIN_CSV}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[run_ablation_smoke] FATAL: missing ${path}" >&2
        exit 1
    fi
done
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 2. Forward CLI args to the Python runner.  Anything after
#    ``--`` is passed verbatim; common flags are exposed below for
#    convenience.
# ---------------------------------------------------------------------------
STEPS="${STEPS:-200}"
DECODE_SAMPLES="${DECODE_SAMPLES:-16}"
ONLY="${ONLY:-}"  # e.g. "M13,M14,M15" for a fast subset
TIMEOUT_S="${TIMEOUT_S:-3600}"

PYTHON_ARGS=(
    "${ABLATION_RUNNER}"
    "--steps" "${STEPS}"
    "--decode-samples" "${DECODE_SAMPLES}"
    "--out-dir" "${OUT_DIR}"
    "--csv" "${TRAIN_CSV}"
    "--timeout-s" "${TIMEOUT_S}"
)
if [[ -n "${ONLY}" ]]; then
    PYTHON_ARGS+=("--only" "${ONLY}")
fi

# Allow extra flags (e.g. ``bash run_ablation_smoke.sh --steps 100``).
for arg in "$@"; do
    PYTHON_ARGS+=("${arg}")
done

echo "[run_ablation_smoke] === ablation_runner.py invocation ===" >&2
echo "[run_ablation_smoke] uv run --no-sync python ${PYTHON_ARGS[*]}" >&2

# ---------------------------------------------------------------------------
# 3. Run the ablation harness.  We use ``uv run --no-sync`` (mirrors
#    ``metallo_drug_smoke_retrain.py`` docstring) so we land in the
#    shared CadButEaas .venv-vlm that has torch 2.14.0+cu130, rdkit,
#    openbabel, etc.  UV_PROJECT_ENVIRONMENT is already wired by
#    env.sh; ``--no-sync`` skips uv's dependency resolver because
#    we've already installed everything in the shared env.
# ---------------------------------------------------------------------------
T_START=$(date +%s)
uv run --no-sync python "${PYTHON_ARGS[@]}"
RC=$?
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

if [[ ${RC} -ne 0 ]]; then
    echo "[run_ablation_smoke] FAIL rc=${RC} after ${ELAPSED}s." >&2
    echo "[run_ablation_smoke] See ${OUT_DIR}/final.md for per-ablation status." >&2
    exit "${RC}"
fi
echo "[run_ablation_smoke] DONE rc=0 in ${ELAPSED}s." >&2
echo "[run_ablation_smoke] Outputs:" >&2
echo "[run_ablation_smoke]   - ${OUT_DIR}/aggregate.csv" >&2
echo "[run_ablation_smoke]   - ${OUT_DIR}/final.md" >&2
echo "[run_ablation_smoke]   - ${OUT_DIR}/<AblationID>.json (11 files)" >&2
exit 0
