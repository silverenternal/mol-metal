#!/bin/bash
# Wf-fix3lines Wave 4: CFM bond head 100-epoch GPU run WITH m5 QED wire-in.
#
# Builds on BLK-C1 (20-ep) + BLK-C2 (100-ep). BLK-C2 confirmed mode collapse
# plateau at ~0.55 (best decode_ratio 0.500, far below 0.8 gate). This run adds
# the m5 QED wire-in (CFM decode -> property panel) so the 100-epoch trained
# model is evaluated not just on bond_f1/decode_ratio but on the real generated
# SMILES via SA + QED + diversity.
#
# Three NEW flags vs BLK-C2:
#   --use-egnn-velocity      (req'd for --gen-eval-n to actually trigger)
#   --gen-eval-n 64          (per-seed sample count for gen eval; up from 8)
#   --save-generated-smiles  (persist <out_dir>/generated_smiles.csv for panel)
#
# Wire-in VERIFIED on CPU smoke (5-ep, 69 SMILES generated, panel m5 QED
# = 0.240 ± 0.173 across 15 last-epoch molecules — see
# molmetal/reports/cfm_smoke_wire_5ep/property_panel.json). The 100-epoch
# scale is the same as BLK-C2 (743.6 s wall); the gen-eval overhead
# (8 -> 64 samples × 100 ep) is small vs bond-head training cost.
#
# AGENTS.md section 5.9 single-venv: .venv/bin/python only.
# AGENTS.md section 5.6 atomic write + BLK-I1 2026-10-09: setsid+nohup + redirect.
PY="/home/hugo/codes/mol-metal/.venv/bin/python"
set -e
cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/cfm_bond_head_100ep_qed_2026_10_09
LOG=logs/wf_fix3lines_cfm_100ep_qed.log
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect the entire .sh stdout/stderr to LOG; trainer detached
exec >"$LOG" 2>&1

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# BLK-I1: detach trainer from parent process group + background + wait for rc
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no cli参数"): every
# hyperparameter below moved into configs/cfm_bond_head_100ep_qed_2026_10_09.yaml as a faithful transcription
# of what this command line used to pass. The wrapper is now env + detach +
# redirect only. Note that cfm_bond_head_100ep_qed_2026_10_09.yaml records use_fm_objective: false — the
# original command never set it, so the velocity field had no flow-matching
# target; see configs/cfm_blk_c5_repair_2026_10_09.yaml for the measured
# gradient census behind that note.
setsid nohup "$PY" molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --config configs/cfm_bond_head_100ep_qed_2026_10_09.yaml \
  --out-dir "$OUT_DIR" \
  >/dev/null 2>&1 &
PID=$!
wait $PID
rc=$?

echo "[wf_fix3lines_cfm_qed] trainer rc=$rc"
exit $rc
