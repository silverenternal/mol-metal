#!/bin/bash
# Wf-193 — Run 14 (metal, seed) pairs at 50 epochs, hidden=64, depth=2, CPU
# Output to molmetal/reports/wf193_wf131_p3_extended/

set -uo pipefail

PY=/home/hugo/codes/mol-metal/.venv/bin/python
OUT_DIR=/home/hugo/codes/mol-metal/molmetal/reports/wf193_wf131_p3_extended
PROJ_ROOT=/home/hugo/codes/mol-metal
LOG_DIR=${OUT_DIR}/logs

mkdir -p "${LOG_DIR}"

cd "${PROJ_ROOT}"

# Pair list: Ir×3 + Os×3 + Rh×3 + Re×5 = 14
PAIRS=(
  "Ir 1"
  "Ir 2"
  "Ir 3"
  "Os 1"
  "Os 2"
  "Os 3"
  "Rh 1"
  "Rh 2"
  "Rh 3"
  "Re 0"
  "Re 1"
  "Re 2"
  "Re 3"
  "Re 4"
)

# Start wall-time timer
WF_T0=$(date +%s)

for entry in "${PAIRS[@]}"; do
  read -r metal seed <<< "${entry}"
  T_PAIR=$(date +%s)
  echo "===== Wf-193 :: fold=${metal} seed=${seed}  epochs=50 hidden=64 depth=2 cpu ====="
  ${PY} molmetal/scripts/train_metal_lomo_wasserstein.py \
    --metals "${metal}" --seeds "${seed}" \
    --epochs 50 --hidden 64 --depth 2 \
    --wasserstein-weight 0.05 --rh-oversample 500 \
    --out-dir "${OUT_DIR}" \
    --device cpu \
    --batch-size 64 --lr 1e-3 \
    > "${LOG_DIR}/${metal}_seed${seed}.log" 2>&1
  EC=$?
  T_PAIR_END=$(date +%s)
  PAIR_MIN=$(( (T_PAIR_END - T_PAIR) / 60 ))
  ELAPSED_MIN=$(( (T_PAIR_END - WF_T0) / 60 ))
  echo "[wf193] fold=${metal} seed=${seed} ec=${EC} pair_min=${PAIR_MIN} cum_min=${ELAPSED_MIN}"
  if [ ${EC} -ne 0 ]; then
    echo "[wf193] ERROR pair failed; continuing"
  fi
done

WF_T1=$(date +%s)
WF_TOTAL_MIN=$(( (WF_T1 - WF_T0) / 60 ))
echo "[wf193] DONE total_min=${WF_TOTAL_MIN}"