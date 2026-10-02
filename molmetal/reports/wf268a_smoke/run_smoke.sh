#!/usr/bin/env bash
# Wf-268A-Smoke runner: 4 ablation configs x 5 epochs each.
#
# Honors the task spec:
#   - hidden_dim=32, n_layers=2, 100 samples (CPU-only)
#   - 5 epochs each
#   - Flags: full / --no-yuelbond / --no-wasserstein / --no-closure-bounded
#   - Decode_ratio must be measured + non-NaN for each config.
#
# CONSTRAINT: USE /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
# (the task spec path "/home/hugo/cades/CadButEaas/.venv-vlm/bin/python"
# does not exist; the actual venv lives at /home/hugo/codes/CadButEaas/.
# This was verified via find /home/hugo -name ".venv-vlm" -type d.)
set -uo pipefail

PYTHON="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
SCRIPT="/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
REPORT_ROOT="/home/hugo/codes/mol-metal/molmetal/reports/wf268a_smoke"

mkdir -p "${REPORT_ROOT}"

cd /home/hugo/codes/mol-metal

# Sanity: venv path
if [[ ! -x "${PYTHON}" ]]; then
    echo "FATAL: python venv not found at ${PYTHON}" >&2
    exit 2
fi

# Common flags (CPU + 100 samples + 5 epochs + small model)
COMMON=(
    --epochs 5
    --n-train 100
    --n-val 100
    --hidden-dim 32
    --n-layers 2
    --batch-size 32
    --device cpu
    --cuda-visible-devices ""
    --patience 10
    --seed 0
    --out-dir
)

# 4 configs.  Full has all default flags (no --no-*).
# Each run writes per_epoch_metrics.json under its own out-dir.
run_config () {
    local label="$1"; shift
    local out="${REPORT_ROOT}/${label}"
    mkdir -p "${out}"
    echo "================================================================"
    echo "[wf268a-smoke] ${label} : out=${out}"
    echo "================================================================"
    "${PYTHON}" -u "${SCRIPT}" \
        "${COMMON[@]}" "${out}" \
        "$@" \
        2>&1 | tee "${out}/run.log"
    local rc="${PIPESTATUS[0]}"
    echo "[wf268a-smoke] ${label} exit_code=${rc}"
    return ${rc}
}

# Full — default flags (yuelbond on, wasserstein on, closure on)
run_config "config_full" || echo "[wf268a-smoke] config_full FAILED rc=$?"

# --no-yuelbond
run_config "config_no_yuelbond" --no-yuelbond || echo "[wf268a-smoke] config_no_yuelbond FAILED rc=$?"

# --no-wasserstein
run_config "config_no_wasserstein" --no-wasserstein || echo "[wf268a-smoke] config_no_wasserstein FAILED rc=$?"

# --no-closure-bounded
run_config "config_no_closure_bounded" --no-closure-bounded || echo "[wf268a-smoke] config_no_closure_bounded FAILED rc=$?"

echo "================================================================"
echo "[wf268a-smoke] All 4 runs complete."
echo "================================================================"