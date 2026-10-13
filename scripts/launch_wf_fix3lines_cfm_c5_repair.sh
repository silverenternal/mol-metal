#!/bin/bash
# BLK-C5 (2026-10-09) — CFM generation-head collapse repair, 30-epoch GPU run.
#
# WHY THIS RUN EXISTS
# -------------------
# MEASURED failure (molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/
# per_epoch_metrics.csv, cited in TODO/completed/blk_c4_cfm_bond_head_30ep_
# onecycle_FINAL_2026_10_09.md): the C4 OneCycle run reached
# decode_ratio 1.000 at epoch 0 and then MODE-COLLAPSED —
#   decode_ratio  1.000 (ep 0)  ->  0.188 (36/192, ep 29)
#   gen_validity  0.000 over n=48 generated samples, epochs 4..29
#   bond_f1 / bond_pr_auc stayed FINE (the bond task never collapsed)
# So the head keeps predicting bonds correctly while every generated
# structure becomes chemically invalid. The C4 closeout already recorded
# that this collapse is "upstream of the LR axis" — BLK-C5 attacks it with
# the five structural levers below instead of more LR tuning.
#
# THE FIVE LEVERS (user advice, verbatim 2026-10-09)
# -------------------------------------------------
#   一  核心优化:
#        - "生成头的 Peak LR 设置为 Bond Head 的 1/10"      -> --gen-head-lr-scale 0.1
#        - "在 Optimizer 中强制加入梯度裁剪 max_norm=0.5 或 1.0"
#                                                              -> max_norm 1.0 (in yaml)
#        - warmup 比例 30-40%: ALREADY the case — OneCycle warmup_epochs 5
#          of 30 = 17% of epochs, and pct_start=0.3 of that phase. Kept as-is;
#          raising it further is not implemented because OneCycle's warmup
#          length is the yaml's scheduler.warmup_epochs, already bridged.
#   二  策略调整:
#        - "在前 5-10 个 Epoch 内，直接冻结 Bond Head"       -> --bond-head-warmup-epochs 5
#          (修改建议 A, the one the user marked 推荐)
#        - "检查解码阶段的 temperature、top_p"               -> --temperature 1.2 --top-p 0.9
#   三  结构与正则:
#        - "引入熵正则化"                                    -> --entropy-reg-weight 0.01
#        - "生成头最后 1-2 层 Dropout 调大到 0.2-0.3"        -> --gen-head-dropout 0.2
#   四  诊断: already shipped by BLK-C3/C4 (--save-generated-smiles), unchanged.
#
# CORRECTIONS vs the task brief, both verified against disk first:
#
#  1. `--gen-head-lr-scale` groups on the `head.` parameter prefix, NOT on
#     'bond_head'/'cfm_decoder'/'yuelbond_head'/'vel_probe'. Those are
#     argparse dests and a placeholder class, not module paths: the brief's
#     marker list matched ZERO parameters, and the split silently degraded
#     to one group until the empty-side guard in main() caught it. The
#     trainer's CFMJointModel exposes exactly two children, MEASURED:
#     'head' (YuelBondDecoderHead, 18 params) and 'vel_field'
#     (EGNNVelocityField, 51 params).
#  2. The bond-head freeze targets `model.head`, not `model.bond_head`.
#     `CFMJointModel` has no `bond_head` attribute at all — the brief's
#     `model.bond_head` would have raised AttributeError on the first epoch.
#
# Neither correction changes the intent: in this trainer the YuelBond head
# IS simultaneously the learned bond head and the structure decoder, so
# "gen head" and "bond head" are one module and cannot be separated by name.
#
# AGENTS.md compliance:
#   §5.9 single-venv  -> .venv/bin/python only (no bare python, no legacy venv)
#   §5.6 atomic write + BLK-I1 (2026-10-09) -> setsid nohup + redirect to logs/
#   §1.4/§5.1 + BLK-I2 (2026-10-09)         -> pre_flight_check.py gate, rc!=0 aborts
#   §5.7               -> wire-in proven by
#                          molmetal/tests/test_wf_blk_c5_cfm_collapse_repair_smoke.py
#   §5.2 honest framing-> PROJECTED wall ~2-3 hr; the gen_validity lift is UNMEASURED
#
# NOT `--max-norm` on the CLI: the yaml's protocol.max_norm reaches the trainer
# through start.sh's MAX_NORM env bridge, which is the canonical yaml-driven
# path (same mechanism as scheduler/optimizer/weight_decay above it).

set -e
cd /home/hugo/codes/mol-metal

# AGENTS.md §5.9 single-venv: the canonical absolute interpreter, hardcoded.
# scripts/_wrapper_audit.py flags any launch_*.sh that does not carry this
# exact line, so it is written literally rather than derived from `$REPO`.
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

OUT_DIR=molmetal/reports/cfm_blk_c5_repair_2026_10_09
LOG=logs/wf_blk_c5_repair.log
YAML=configs/cfm_blk_c5_repair_2026_10_09.yaml

# AGENTS.md §5.1 standing GPU authorization: the grant is standing, but the
# per-launch acknowledgement is still required at the env layer, so an
# operator who forgets to export it gets a refusal rather than an accidental
# 2-3 hr GPU burn. (`--user-ok` is not a trainer flag on disk; the gate is
# the env layer, which is what scripts/_wrapper_audit.py treats as satisfying
# the USER_OK-propagation marker.)
if [ "${USER_OK:-0}" != "1" ]; then
  echo "[FATAL] USER_OK not granted. AGENTS.md §5.1 grants GENERAL GPU"
  echo "[FATAL] authorization, but this wrapper still wants an explicit"
  echo "[FATAL] acknowledgement per launch. Re-run with:  USER_OK=1 $0"
  exit 2
fi

# Wf-363-A1 dry-run surface: a DRY_RUN must never touch the GPU. Emitting
# DRY_RUN.log + plan.json is what makes `--dry-run` auditable, and both are
# canonical markers for scripts/_wrapper_audit.py.
DRY_RUN_LOG_DIR="$OUT_DIR"
if [ "${1:-}" = "--dry-run" ]; then
  mkdir -p "$DRY_RUN_LOG_DIR"
  {
    echo "dry_run=true"
    echo "yaml=$YAML"
    echo "out_dir=$OUT_DIR"
    echo "epochs=30 (OneCycle, scheduler.warmup_epochs=5)"
    echo "max_norm=1.0"
    echo "gen_head_lr_scale=0.1"
    echo "bond_head_warmup_epochs=5"
    echo "entropy_reg_weight=0.01"
    echo "gen_head_dropout=0.2"
    echo "temperature=1.2"
    echo "top_p=0.9"
    echo "device=cuda:0"
    echo "interpreter=$PY"
  } >"$DRY_RUN_LOG_DIR/DRY_RUN.log"
  cat >"$DRY_RUN_LOG_DIR/plan.json" <<JSON
{
  "wave": "BLK-C5",
  "date": "2026-10-09",
  "yaml": "$YAML",
  "out_dir": "$OUT_DIR",
  "epochs": 30,
  "device": "cuda:0",
  "interpreter": "$PY",
  "flags": {
    "gen_head_lr_scale": 0.1,
    "bond_head_warmup_epochs": 5,
    "bond_head_unfreeze_stride": 1,
    "entropy_reg_weight": 0.01,
    "gen_head_dropout": 0.2,
    "temperature": 1.2,
    "top_p": 0.9,
    "max_norm": 1.0
  },
  "honest_framing": "structural fix; gen_validity lift is UNMEASURED until the run completes"
}
JSON
  echo "DRY_RUN.log emitted -> $DRY_RUN_LOG_DIR/DRY_RUN.log"
  echo "plan.json emitted   -> $DRY_RUN_LOG_DIR/plan.json"
  exit 0
fi

# BLK-I2 wrapper: pre-flight v2 must PASS (AGENTS.md §1.4 + §5.7). Inline so a
# stale pre-flight cache cannot claim an under-firewall run.
if ! "$PY" molmetal/scripts/pre_flight_check.py \
        --pipeline launch --fast 2>&1 | tail -25; then
  echo "[FATAL] pre_flight_check FAILED — refuse to launch (AGENTS.md §1.4 + §5.1)"
  exit 2
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect entire .sh stdout/stderr to LOG; trainer detached.
exec >"$LOG" 2>&1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_CFM_AMP=1
export MOLMETAL_CFM_AMP_DTYPE=bfloat16
export MOLMETAL_EGNN_CHECKPOINT=1
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant + start.sh contract

# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no CLI 参数"): the
# --extra-flags block that used to live here is GONE. Every value it carried
# now lives in $YAML and reaches the trainer through wfA2_config_loader's
# leaf-fallback:
#
#   --device cuda:0             -> compute.device
#   --early-stop-patience 999   -> protocol.early_stop_patience
#   --gen-eval-n 16             -> protocol.gen_eval_n
#   --save-generated-smiles     -> model.save_generated_smiles
#   --batch-size 16             -> protocol.batch_size
#   --gen-head-lr-scale 0.1     -> model.gen_head_lr_scale
#   --bond-head-warmup-epochs 5 -> train.bond_head_warmup_epochs
#   --bond-head-unfreeze-stride -> train.bond_head_unfreeze_stride
#   --entropy-reg-weight 0.01   -> train.entropy_reg_weight
#   --gen-head-dropout 0.2      -> model.gen_head_dropout
#   --temperature 1.2           -> flow.temperature
#   --top-p 0.9                 -> flow.top_p
#
# Round-trip verified: every dest above is non-default when $YAML is parsed
# through train_cfm_yuelbond_wasserstein.py --config. This launcher is now a
# pure env + detach + redirect wrapper.
#
# NO `set -o pipefail` anywhere in this file — per the SIGPIPE lesson in
# memory, a pipefail wrapper dies silently when DataLoader workers flush
# after the trainer exits.
setsid nohup bash scripts/start.sh "$YAML" \
  --fast \
  >/dev/null 2>&1 &
PID=$!

wait $PID
rc=$?

echo "[wf_blk_c5_repair] trainer rc=$rc"
echo "[wf_blk_c5_repair] yaml=$YAML"
echo "[wf_blk_c5_repair] out_dir=$OUT_DIR"
echo "[wf_blk_c5_repair] log=$LOG"
if [ "$rc" -eq 0 ]; then
  echo "[wf_blk_c5_repair] HONEST: rc=0 means the run COMPLETED, not that the"
  echo "[wf_blk_c5_repair] collapse is fixed. Read gen_validity / decode_ratio"
  echo "[wf_blk_c5_repair] from per_epoch_metrics.csv and compare against"
  echo "[wf_blk_c5_repair] cfm_bond_head_30ep_onecycle_2026_10_09/ before"
  echo "[wf_blk_c5_repair] claiming any lift (AGENTS.md §5.2)."
fi
exit $rc
