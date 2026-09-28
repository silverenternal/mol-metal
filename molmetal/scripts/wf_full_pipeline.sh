#!/bin/bash
# wf_full_pipeline.sh — end-to-end R17 demo
#
# Runs the full Mol-Metal pipeline (evaluation → pitfall-guards → train
# smoke → 10-pocket pilot → aggregate → diagnose) on CPU in <10 min.
#
# Each section is independently callable: a section that fails does NOT
# abort the pipeline (we log and continue) so the harness still produces
# a complete log.  Section [6/6] always runs to surface the Tier-1
# summary even if upstream sections degraded.
#
# Usage
# =====
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/wf_full_pipeline.sh 2>&1 | tee /tmp/full_pipeline_test.log
#
# Exit codes
# ==========
#     0   — every section succeeded.
#     1   — at least one section failed (but the run completed end-to-end).
#     124 — a section exceeded its per-section wall-clock budget.
#
# Honest framing
# ==============
# This is the F3 backlog item: a *demonstration* that the full pipeline
# is wired correctly on this host.  Wall-time, decode ratio and pilot
# diversity are MEASURED numbers; nothing in this script is mocked.
set -o pipefail

# Resolve repo root (one level above molmetal/scripts/)
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Per-section wall-clock budget (CPU-only demo). ---
SEC_BUDGET_EVAL=120      # [1/6] eval all evaluators
SEC_BUDGET_PITFALL=180   # [2/6] pitfall guards (7 pytest runs)
SEC_BUDGET_TRAIN=300     # [3/6] 200-step CFM smoke (small model on CPU)
SEC_BUDGET_PILOT=180     # [4/6] 10-pocket × 1-seed pilot
SEC_BUDGET_AGG=120       # [5/6] aggregate metrics
SEC_BUDGET_DIAG=120      # [6/6] diagnose Tier-1 metric m14

PY_BIN="${PY_BIN:-python}"
export PYTHONUNBUFFERED=1

# Source the project env if present (env.sh wires UV_PROJECT_ENVIRONMENT
# to the shared venv — see env.sh header).  When env.sh is absent (e.g.
# CI) we fall back to whatever python is on PATH.
if [[ -f "${REPO_ROOT}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/env.sh"
fi

# --- Helpers ---------------------------------------------------------------
START_TS=$(date +%s)
FAILED_SECTIONS=()

run_section () {
    local title="$1"
    local budget_sec="$2"
    shift 2
    local sec_idx="$1"
    local sec_total="$2"
    shift 2
    echo
    echo "============================================================"
    echo "[${sec_idx}/${sec_total}] ${title}"
    echo "  budget: ${budget_sec}s"
    echo "  cmd:    $*"
    echo "============================================================"
    local t0
    t0=$(date +%s)
    # `timeout` returns 124 on wall-clock hit; we still want to continue.
    timeout --foreground "${budget_sec}" "$@"
    local rc=$?
    local t1
    t1=$(date +%s)
    local dt=$((t1 - t0))
    if [[ ${rc} -eq 0 ]]; then
        echo "[${sec_idx}/${sec_total}] OK   (${dt}s)"
    elif [[ ${rc} -eq 124 ]]; then
        echo "[${sec_idx}/${sec_total}] TIMEOUT after ${dt}s (budget ${budget_sec}s)"
        FAILED_SECTIONS+=("${sec_idx}:${title}:TIMEOUT")
    else
        echo "[${sec_idx}/${sec_total}] FAIL rc=${rc} (${dt}s)"
        FAILED_SECTIONS+=("${sec_idx}:${title}:rc=${rc}")
    fi
}

# --- Section [1/6] — Eval all evaluators ----------------------------------
run_section "Eval all evaluators (sanity)" \
    "${SEC_BUDGET_EVAL}" "1" "6" \
    ${PY_BIN} -u molmetal/scripts/eval_all_evaluators.py

# --- Section [2/6] — Run pitfall guards ------------------------------------
run_section "Run pitfall guards (7 guards)" \
    "${SEC_BUDGET_PITFALL}" "2" "6" \
    ${PY_BIN} -u molmetal/scripts/wf_pitfall_guard/check_all.py

# --- Section [3/6] — Train 200-step R17 smoke (1 seed) ---------------------
SMOKE_OUT="/tmp/full_pipeline_smoke"
mkdir -p "${SMOKE_OUT}"
run_section "Train 200-step CFM smoke (1 seed, hidden_dim=64)" \
    "${SEC_BUDGET_TRAIN}" "3" "6" \
    ${PY_BIN} -u molmetal/scripts/metallo_drug_smoke_retrain.py \
        --steps 200 --batch-size 4 --seed 0 \
        --hidden-dim 64 --n-layers 2 \
        --lr-warmup-steps 20 \
        --decode-smoke-every 100 \
        --out-dir "${SMOKE_OUT}"

# --- Section [4/6] — Run 10-pocket × 1-seed pilot --------------------------
PILOT_OUT="${REPO_ROOT}/full_pipeline_smoke"
mkdir -p "${PILOT_OUT}"
run_section "10-pocket × 1-seed Lambda pilot (100 sims each)" \
    "${SEC_BUDGET_PILOT}" "4" "6" \
    ${PY_BIN} -u molmetal/scripts/r4_lambda_only_run.py \
        --metal-seed cisplatin --pockets 10 --seeds 0 \
        --n-simulations 100 --click-rules auto-pt-strict \
        --output-dir full_pipeline_smoke

# --- Section [5/6] — Compute aggregate metrics ----------------------------
run_section "Compute aggregate metrics (re-eval)" \
    "${SEC_BUDGET_AGG}" "5" "6" \
    ${PY_BIN} -u molmetal/scripts/eval_all_evaluators.py

# --- Section [6/6] — Generate Tier-1 summary ------------------------------
run_section "Generate Tier-1 summary (m14 diagnostic)" \
    "${SEC_BUDGET_DIAG}" "6" "6" \
    ${PY_BIN} -u molmetal/scripts/diagnose_metric.py --metric m14

# --- Summary ---------------------------------------------------------------
END_TS=$(date +%s)
TOTAL_DT=$((END_TS - START_TS))
echo
echo "============================================================"
echo "wf_full_pipeline.sh — DONE"
echo "  total wall-time : ${TOTAL_DT}s ($(date -u -d "@${TOTAL_DT}" +%H:%M:%S 2>/dev/null || echo "${TOTAL_DT}s"))"
echo "  sections run    : 6"
echo "  sections failed : ${#FAILED_SECTIONS[@]}"
if (( ${#FAILED_SECTIONS[@]} > 0 )); then
    for f in "${FAILED_SECTIONS[@]}"; do
        echo "    - ${f}"
    done
    echo "ALL DONE (with ${#FAILED_SECTIONS[@]} failure(s))"
    exit 1
fi
echo "ALL DONE"
exit 0
