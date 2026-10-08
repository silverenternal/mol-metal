#!/bin/bash
# Wf-fix3lines Wave 2: CFM bond head 20-epoch GPU production run.
#
# Two corrections vs the brief, both verified against disk first:
#
# 1. NO --out-metric. `grep -n 'out.metric' molmetal/scripts/train_cfm_
#    yuelbond_wasserstein.py` returns nothing, and passing the flag makes
#    argparse exit 2 with "unrecognized arguments". This trainer writes
#    metrics.json into --out-dir itself; it has no --out-metric dest.
#
# 2. --tmqm-dir IS passed. Without it the trainer falls to
#    build_synthetic_pt_click_clouds (one jittered square-planar Pt layout,
#    same 5 atoms every run). MEASURED on a CPU probe: the synthetic path
#    emits `dataset=synthetic` + the cfm-bond-target WARNING that targets are
#    SYNTHESISED, and scores bond_n_scored=0 / bond_f1=0.0 — a perfect
#    decode_ratio there means nothing. With --tmqm-dir
#    .data/molmetal/tmQM (324 MB, has tmQM_X*.BO.gz Wiberg blocks) the same
#    probe reports dataset=tmqm and bond_n_scored=24 / bond_f1=0.477. This is
#    the whole point of a bond-head run, so the corpus flag is not optional.
#
# 3. --early-stop-patience 999. MEASURED on the first launch (43 s wall):
#    the trainer defaulted to --early-stop-metric bond_exact_type with
#    patience 3 and stopped at epoch 4 ("best bond_exact_type=0.0052 @
#    epoch 1"), so a 20-epoch run never happened. The flag is raised so the
#    requested 20 epochs actually execute; the gate is still recorded in the
#    done-event.
#
# AGENTS.md §5.9 single-venv: .venv/bin/python only.
# No `set -o pipefail` (see the SIGPIPE lesson — DataLoader workers flush
# after trainer exit and kill the wrapper silently). No `| tee` here at all;
# the harness captures the background log.
set -e
cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/cfm_bond_head_20ep_2026_10_09
mkdir -p "$OUT_DIR" logs

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

.venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 128 \
  --n-layers 3 \
  --device cuda:0 \
  --tmqm-dir .data/molmetal/tmQM \
  --early-stop-patience 999 \
  --out-dir "$OUT_DIR"
rc=$?

echo "[wf_fix3lines_cfm] trainer rc=$rc"
exit $rc