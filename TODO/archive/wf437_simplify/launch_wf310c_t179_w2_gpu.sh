#!/usr/bin/env bash
# Wf-310C Part 3 — TODO-179 W₂ pIC50 alignment GPU retrain wrapper.
#
# ============================================================================
# Wf-310C PART 3 — TODO-179 W₂ ALIGNMENT ON PLATINAI 4-CELL PROBE
# ============================================================================
#
# This wrapper gates the W₂ pIC50 alignment retrain behind a USER_OK marker
# file so the expensive 5-seed × 50-epoch Hilbert+ASTSR+W₂ joint sweep on the
# PlatinAI 4-cell probe (4078 Pt complexes × 4 cell lines = ~16k rows) does
# NOT start without an explicit user opt-in (per kill_then_adapt convention,
# Wf-295D + Wf-306B).
#
# Trainer
# -------
#   molmetal/scripts/train_hilbert_astsr_end_to_end.py
#   --wass-weight 0.05 --ref-distribution platinai_4cell  (TODO-179 §3.1)
#
# Scope (5 seeds × 50 epochs × PlatinAI 4-cell probe)
# ----------------------------------------------------
#   seed 0  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 1  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 2  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 3  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 4  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   total wall budget: ~4.5 hr on RTX 5090 cuda:1 (PROJECTED; 5 × 50 min
#   with 10% slack = 275 min ≈ 4.6 hr; budgeted 4.5 hr to fit a single GPU-1
#   reservation slot).
#
# Per-cell breakdown
# ------------------
# A2780 / A2780cis / MCF7 / A549 are the 4 cells in PlatinAI. Per-cell R² is
# the primary honest metric (no aggregation hiding per-cell noise, per
# party-twenty-spirit §3.5).
#
# Honest framing (per Wf-127 §Cause 2 + TODO-179 §6)
# --------------------------------------------------
# * MEASURED baseline: L5 R² = +0.163 (Wf-37 P1 bug fix; NOT algorithmic lift).
# * PROJECTED ceiling: 0.40-0.60 from W₂ alone (12-d RDKit signal floor).
# * NOT MEASURED: 0.85+ — that would require F5_RDKIT_2D 32-d expansion
#   which is a separate follow-up (TODO-179 §8).
#
# USAGE:
#   # 1. CPU smoke (already covered by Wf-246-A; this wrapper skips smoke).
#   # 2. To authorise the GPU run:
#   touch /tmp/USER_OK_WF310C_T179
#   bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh
#   # 3. To explicitly skip GPU (default behaviour):
#   unset WF310C_RUN_GPU   # (the default is "no" without USER_OK file)
#
# Exit codes:
#   0 = GPU run completed (all seeds ok)
#   2 = USER_OK missing (gated; no run started)
#   3 = Setup error (venv / GPU probe / CSV missing)
#   4 = GPU run failed (partial results still in run.log)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root from script location so the wrapper works regardless
# of the caller's CWD.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
USER_OK_FILE="/tmp/USER_OK_WF310C_T179"

TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_hilbert_astsr_end_to_end.py"
PLATINAI_CSV="molmetal/data/platinai_4cell.csv"
OUT_DIR="molmetal/reports/wf310c_t179_w2_wrapper"
RUN_LOG="${OUT_DIR}/run.log"
SUMMARY_JSON="${OUT_DIR}/sweep_summary.json"
PLAN_JSON="${OUT_DIR}/plan.json"

EPOCHS=50
SEEDS=(0 1 2 3 4)
WALL_BUDGET_MIN=270   # 4.5 hr wall budget (PROJECTED from Wf-37 + Wf-246-A 1-seed timings)
GPU="cuda:1"         # RTX 5090

echo "[wf310c-t179] project_root=${PROJECT_ROOT}"
echo "[wf310c-t179] venv=${VENV_PY}"
echo "[wf310c-t179] USER_OK_FILE=${USER_OK_FILE}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Pre-flight: venv + trainer + CSV existence
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[wf310c-t179] ERROR: venv python missing at ${VENV_PY}" >&2
    exit 3
fi
if [[ ! -f "${TRAINER}" ]]; then
    echo "[wf310c-t179] ERROR: trainer missing at ${TRAINER}" >&2
    exit 3
fi
if [[ ! -f "${PLATINAI_CSV}" ]]; then
    echo "[wf310c-t179] ERROR: PlatinAI 4-cell CSV missing at ${PLATINAI_CSV}" >&2
    echo "[wf310c-t179] Per Wf-246-A honest-negative: do not run GPU W₂ until CSV is in place." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 2. USER_OK gate (kill_then_adapt convention per Wf-295D + Wf-306B)
# ---------------------------------------------------------------------------
if [[ ! -f "${USER_OK_FILE}" ]]; then
    cat <<EOF
[wf310c-t179] USER_OK GATE NOT PASSED.

To authorise the W₂ alignment GPU retrain, create the marker file and re-run:

    touch ${USER_OK_FILE}
    bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh

Then monitor via:
    tail -f ${RUN_LOG}

Total wall: ~4.5 hr GPU (5 seeds × 50 epochs × Hilbert+ASTSR+W₂ joint loss).
Honest framing per TODO-179 §6:
  * MEASURED baseline R² = +0.163 (Wf-37 P1 bug fix; NOT algorithmic lift).
  * PROJECTED ceiling 0.40-0.60 from W₂ alone (12-d RDKit signal floor).
  * Honest per-cell breakdown will be emitted (no aggregation hiding noise).
EOF
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. GPU probe (per Wf-130 + Wf-190 env guard)
# ---------------------------------------------------------------------------
echo "[wf310c-t179] GPU probe ..."
CUDA_PROBE_OUT=$(
    CUDA_VISIBLE_DEVICES="${WF310C_T179_CUDA_VISIBLE_DEVICES:-1}" \
    MOLMETAL_ALLOW_CPU=0 \
    "${VENV_PY}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
) || true
echo "[wf310c-t179] CUDA probe: ${CUDA_PROBE_OUT}"
if ! echo "${CUDA_PROBE_OUT}" | grep -q "^True"; then
    echo "[wf310c-t179] ERROR: CUDA not available on this host; aborting GPU run." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 4. Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED
# label discipline (party-twenty-spirit §3.5).
# ---------------------------------------------------------------------------
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310c_t179_w2_gpu.sh",\n'
    printf '  "wf_origin": "Wf-310C",\n'
    printf '  "todo_ref": "TODO-pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}")"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "wall_budget_label": "PROJECTED (5 seeds × 50 epochs × ~50 min/seed from Wf-37 1-seed timings + Wf-246-A W2 wire-in overhead; 10%% slack)",\n'
    printf '  "wass_weight": 0.05,\n'
    printf '  "ref_distribution": "platinai_4cell",\n'
    printf '  "per_cell_cap": 32,\n'
    printf '  "out_dir": "%s",\n' "${OUT_DIR}"
    printf '  "status": "READY_TO_RUN",\n'
    printf '  "honest_framing": "MEASURED L5 R2=+0.163 (Wf-37 P1 bug fix; NOT algorithmic lift); PROJECTED ceiling 0.40-0.60 from W2 alone (12-d RDKit signal floor per Wf-127 Cause 2)"\n'
    printf '}\n'
} > "${PLAN_JSON}"
echo "[wf310c-t179] plan emitted → ${PLAN_JSON}"

# ---------------------------------------------------------------------------
# 5. Actual GPU run — sequential 5-seed sweep, ~4.5 hr total wall
# ---------------------------------------------------------------------------
echo "[wf310c-t179] === W2 alignment GPU retrain START (5 seeds × 50 epochs) ==="

start_ts=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    echo "[wf310c-t179] === seed=${SEED} Hilbert+ASTSR+W2 50 epochs on ${GPU} ==="
    if CUDA_VISIBLE_DEVICES="${WF310C_T179_CUDA_VISIBLE_DEVICES:-1}" \
         "${VENV_PY}" "${TRAINER}" \
            --dataset platinai \
            --seeds "${SEED}" \
            --epochs "${EPOCHS}" \
            --n-samples 500 \
            --n-folds 5 \
            --wass-weight 0.05 \
            --ref-distribution platinai_4cell \
            --per-cell-cap 32 \
            --platinai-csv "${PLATINAI_CSV}" \
            --output "${OUT_DIR}/seed${SEED}" \
            >> "${RUN_LOG}" 2>&1; then
        echo "[wf310c-t179] [OK] seed=${SEED} completed"
    else
        echo "[wf310c-t179] [FATAL] seed=${SEED} failed (see ${RUN_LOG})" >&2
        exit 4
    fi
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_min=$(( elapsed_sec / 60 ))
echo "[wf310c-t179] === W2 alignment GPU retrain DONE — elapsed=${elapsed_min} min (budget=${WALL_BUDGET_MIN} min) ==="

if [ "${elapsed_min}" -gt "${WALL_BUDGET_MIN}" ]; then
    echo "[wf310c-t179] [WARN] elapsed (${elapsed_min} min) exceeded PROJECTED wall budget (${WALL_BUDGET_MIN} min); flag as MEASURED-over-PROJECTED"
fi

# ---------------------------------------------------------------------------
# 6. Aggregate per-seed summaries
# ---------------------------------------------------------------------------
echo "[wf310c-t179] === Aggregating per-seed results ==="
{
    printf '{\n'
    printf '  "wf_origin": "Wf-310C",\n'
    printf '  "todo_ref": "TODO-pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    printf '  "elapsed_min": %d,\n' "${elapsed_min}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "n_seeds_completed": %d,\n' "${#SEEDS[@]}"
    printf '  "per_seed_dirs": [%s],\n' "$(IFS=,; for s in "${SEEDS[@]}"; do printf '"%s/seed%s"' "${OUT_DIR}" "${s}"; done | sed 's/,/, /g')"
    printf '  "honest_framing": "MEASURED baseline L5 R2=+0.163 (Wf-37 P1 bug fix; NOT algorithmic lift); PROJECTED ceiling 0.40-0.60 from W2 alone (12-d RDKit signal floor). See TODO-179 §6 for honest projection framing."\n'
    printf '}\n'
} > "${SUMMARY_JSON}"
echo "[wf310c-t179] [OK] summary → ${SUMMARY_JSON}"

echo "[wf310c-t179] === Wf-310C TODO-179 W2 GPU retrain COMPLETE ==="
echo "[wf310c-t179] Run log: ${RUN_LOG}"
echo "[wf310c-t179] Per-seed dirs: ${OUT_DIR}/seed{0,1,2,3,4}"
echo "[wf310c-t179] Sweep summary: ${SUMMARY_JSON}"
exit 0
