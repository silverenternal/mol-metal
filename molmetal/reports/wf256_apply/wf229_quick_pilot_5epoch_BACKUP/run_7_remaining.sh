#!/bin/bash
# Wf-254-A: Run the 7 remaining per-lever ablation runs
# - cfg010_seed2 (D-only)
# - cfg001_seed0/1/2 (F-only)
# - cfg111_seed0/1/2 (K+D+F)
set -u
cd /home/hugo/codes/mol-metal
PY=/home/hugo/codes/mol-metal/.venv/bin/python
OUT_DIR=/home/hugo/codes/mol-metal/molmetal/reports/wf229_ablation/quick_pilot/per_config
LOG_DIR=/home/hugo/codes/mol-metal/molmetal/reports/wf229_ablation/quick_pilot/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

# CPU-only env (matches the F4-AblationCPU design)
export MOLMETAL_ALLOW_CPU=1
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# 1) cfg010_seed2 (D-only)
echo "[wf254-a] cfg010_seed2 start $(date +%H:%M:%S)"
$PY molmetal/scripts/wf229_run_one.py \
  --code 010 --epochs 1 --seed 2 \
  --out "$OUT_DIR/cfg010_seed2.json" \
  --use-dann \
  2>&1 | tee "$LOG_DIR/cfg010_seed2.log"
echo "[wf254-a] cfg010_seed2 done $(date +%H:%M:%S)"

# 2) cfg001 seed 0/1/2 (F-only)
for S in 0 1 2; do
  echo "[wf254-a] cfg001_seed$S start $(date +%H:%M:%S)"
  $PY molmetal/scripts/wf229_run_one.py \
    --code 001 --epochs 1 --seed $S \
    --out "$OUT_DIR/cfg001_seed$S.json" \
    --data-config four-domain \
    2>&1 | tee "$LOG_DIR/cfg001_seed$S.log"
  echo "[wf254-a] cfg001_seed$S done $(date +%H:%M:%S)"
done

# 3) cfg111 seed 0/1/2 (K+D+F)
for S in 0 1 2; do
  echo "[wf254-a] cfg111_seed$S start $(date +%H:%M:%S)"
  $PY molmetal/scripts/wf229_run_one.py \
    --code 111 --epochs 1 --seed $S \
    --out "$OUT_DIR/cfg111_seed$S.json" \
    --use-kendall-loss --use-dann \
    --data-config four-domain \
    2>&1 | tee "$LOG_DIR/cfg111_seed$S.log"
  echo "[wf254-a] cfg111_seed$S done $(date +%H:%M:%S)"
done

echo "[wf254-a] ALL 7 REMAINING RUNS COMPLETE"
