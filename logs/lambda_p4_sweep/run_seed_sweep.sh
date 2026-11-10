#!/usr/bin/env bash
# Run the Lambda p4 sweep for one seed and move the JSONs into
# logs/lambda_p4_sweep/seed_<seed_index>/.
#
# Usage: run_seed_sweep.sh <seed_value> <seed_index> [pockets] [n_sims] [top_k]
#
# HYPERPARAMETERS (2026-11-04, AGENTS.md §5.11)
# ------------------------------------------------
# The sweep's protocol values (pockets / seeds / n-simulations / n-top-k) live
# in ``configs/lambda_p4_sweep.yaml``; the command line carries ``--config``
# only.  Before that file existed, the five values were hardcoded on this
# command line, and the sweep was invisible to scripts/_yaml_only_gate.py —
# that gate's trainer predicate is the regex ``train_[a-z0-9_]+\.py``, and
# this sweep runs ``r4_lambda_only_run.py``, which does not match.
#
# What stays on the command line, and why:
#   * ``--output-dir`` — a RUN SELECTOR (which run), not a hyperparameter:
#     the sweep writes one report per seed, so a static YAML value would be
#     wrong for 2 of the 3 seeds.  Same class as ``--out-dir``, which §5.11's
#     allowlist permits.
#   * the positional overrides (seed / pockets / n-sims / top-k) — passed
#     ONLY when the caller supplies them, and an explicit CLI flag outranks
#     the YAML (§5.11 precedence: explicit CLI > YAML > argparse default).
#   * ``--quiet`` — run-control, not a hyperparameter.
#
# The interpreter is the canonical repo venv (AGENTS.md §5.9 single-venv rule):
# ``uv run python`` was replaced by ``.venv/bin/python`` because uv may try to
# sync (network) and because §5.9 names that interpreter explicitly.
set -euo pipefail
cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
CONFIG="${REPO_ROOT}/configs/lambda_p4_sweep.yaml"

SEED_VAL="${1:-}"
SEED_IDX="${2:-0}"
POCKETS="${3:-}"
N_SIMS="${4:-}"
TOP_K="${5:-}"

LOG_DIR="logs/lambda_p4_sweep"
SEED_DIR="${LOG_DIR}/seed_${SEED_IDX}"
TMP_OUT_DIR="_lambda_p4_seed_${SEED_IDX}"

mkdir -p "${LOG_DIR}" "${SEED_DIR}"

# Explicit overrides only — when the caller supplies none, the YAML applies.
EXTRA=()
if [[ -n "${SEED_VAL}" ]]; then EXTRA+=(--seeds "${SEED_VAL}"); fi
if [[ -n "${POCKETS}" ]]; then EXTRA+=(--pockets "${POCKETS}"); fi
if [[ -n "${N_SIMS}" ]]; then EXTRA+=(--n-simulations "${N_SIMS}"); fi
if [[ -n "${TOP_K}" ]]; then EXTRA+=(--n-top-k "${TOP_K}"); fi

if [[ -x "${PY}" ]]; then
    EFFECTIVE="$("${PY}" -c 'import sys, yaml; p = (yaml.safe_load(open(sys.argv[1])) or {}).get("protocol", {}); print("pockets=%s seeds=%s n_sims=%s top_k=%s" % (p.get("pockets"), p.get("seeds"), p.get("n_simulations"), p.get("n_top_k")))' "${CONFIG}")"
else
    EFFECTIVE="pockets=? seeds=? n_sims=? top_k=? (canonical venv missing at ${PY})"
fi

echo "[lambda_p4] ${EFFECTIVE} idx=${SEED_IDX} config=${CONFIG}"
echo "[lambda_p4] explicit overrides: ${EXTRA[*]-<none: every value comes from the YAML>}"
source ./env.sh
"${PY}" molmetal/scripts/r4_lambda_only_run.py \
    --config "${CONFIG}" \
    --output-dir "${TMP_OUT_DIR}" \
    --quiet \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    2>&1 | tail -20

# Move generated report.json / summary.md into the per-seed dir
SRC_DIR="molmetal/reports/wf_lambda1_${TMP_OUT_DIR}"
if [[ ! -f "${SRC_DIR}/report.json" ]]; then
    echo "[lambda_p4] ERROR: missing ${SRC_DIR}/report.json" >&2
    exit 2
fi
cp -f "${SRC_DIR}/report.json" "${SEED_DIR}/report.json"
cp -f "${SRC_DIR}/seeds.json" "${SEED_DIR}/seeds.json" 2>/dev/null || true
cp -f "${SRC_DIR}/summary.md" "${SEED_DIR}/summary.md"

# Clean up the wf_lambda1_<tmp> scaffolding to keep the reports tree tidy
rm -rf "${SRC_DIR}"

echo "[lambda_p4] wrote ${SEED_DIR}/report.json"
