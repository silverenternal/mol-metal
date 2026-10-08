#!/bin/bash
# Wf-fix3lines bace 5-seed GPU re-run.
# Rationale: the 2026-10-08 bace rescue ran on CPU while the baseline ran on GPU;
# ALIGNED_COMPARISON.md states "a clean GPU re-run with a same-device control is
# the confirmation still owed". This run is that confirmation on the canonical
# baseline protocol (class-weight on, no regularisation, hidden 300 / depth 3 /
# batch 64 / lr 1e-3) so the number is same-protocol comparable to
# metrics/by_metric/moleculenet_auc_dmpnn_5seed.json.
#
# AGENTS.md §5.9 single-venv: .venv/bin/python only. No pipefail (SIGPIPE lesson).
set -e
cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09
LOG=logs/wf_fix3lines_bace_50ep.log
mkdir -p "$OUT_DIR" logs

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

.venv/bin/python molmetal/scripts/train_moleculenet.py \
  --tasks bace \
  --backbone dmpnn \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden 300 \
  --depth 3 \
  --seeds 0 1 2 3 4 \
  --out-dir "$OUT_DIR" \
  --out-metric "$OUT_DIR/metrics.json" \
  --out-report "$OUT_DIR/report.md"
rc=$?

echo "[wf_fix3lines_bace] trainer rc=$rc"
exit $rc