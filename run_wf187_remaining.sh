#!/bin/bash
# Wf-187 — Wf-131 P3 7/25 remaining runs (Rh seed3/4 + Re seed0-4)
# Uses the Wf-187 m19 fix: --oversample-positives + --pos-weight 10
# --quick sets epochs=2 (time budget: ~5min per run)

set -e
cd /home/hugo/codes/mol-metal
PY=/home/hugo/codes/CadButEaas/.venv-vlm/bin/python
OUT=molmetal/reports/wf187_p3_rh_oversample_fix
LOG=$OUT/run.log

mkdir -p $OUT
echo "[wf187] start $(date +%H:%M:%S)" > $LOG

# Rh seed3
echo "[wf187] Rh seed3 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Rh --seeds 3 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Rh_seed3 --quick 2>&1 | tail -10 | tee -a $LOG

# Rh seed4
echo "[wf187] Rh seed4 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Rh --seeds 4 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Rh_seed4 --quick 2>&1 | tail -10 | tee -a $LOG

# Re seed0
echo "[wf187] Re seed0 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Re --seeds 0 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Re_seed0 --quick 2>&1 | tail -10 | tee -a $LOG

# Re seed1
echo "[wf187] Re seed1 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Re --seeds 1 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Re_seed1 --quick 2>&1 | tail -10 | tee -a $LOG

# Re seed2
echo "[wf187] Re seed2 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Re --seeds 2 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Re_seed2 --quick 2>&1 | tail -10 | tee -a $LOG

# Re seed3
echo "[wf187] Re seed3 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Re --seeds 3 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Re_seed3 --quick 2>&1 | tail -10 | tee -a $LOG

# Re seed4
echo "[wf187] Re seed4 $(date +%H:%M:%S)" | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --model v3 --metals Re --seeds 4 --epochs 2 \
  --device cpu --no-bf16 \
  --wasserstein-weight 0 \
  --oversample-positives --pos-weight 10 \
  --batch-size 64 \
  --out-dir $OUT/Re_seed4 --quick 2>&1 | tail -10 | tee -a $LOG

echo "[wf187] all done $(date +%H:%M:%S)" | tee -a $LOG