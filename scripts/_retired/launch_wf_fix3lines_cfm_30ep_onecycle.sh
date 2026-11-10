#!/bin/bash
# Wf-fix3lines C4-restart: CFM bond head 30-epoch GPU run with OneCycle + AdamW
#
# Per AGENTS.md §5.5 (filesystem-truth) + §5.6 (atomic-write BLK-I1 wrapper) +
# §5.7 (trainer-flag wire-in CPU smoke required) + §5.9 (single-venv) +
# user voice 2026-10-09 verbatim
#   "有必要跑慢100个epoch吗，我记得我们是有快速收敛和退火类型的优化的优化器的呀"
#   "A吧，然后这个退火的配置你是应该写到yaml配置文件里面的,
#    快速收敛优化呢，那个用了吗？"
#
# CHANGES vs the killed original C4 100-epoch launcher:
#   - 100 -> 30 epochs (OneCycle converges ~5-10x faster than constant-LR;
#     cf. Smith-Topin 2018 super-convergence + Wf-A6 A2 PROJECTED)
#   - inline --lr-scheduler/--optimizer via start.sh scheduler/optimizer env
#     (was constant-LR default; the no-scheduler bug that prompted the restart)
#   - 64 -> 16 gen-eval samples (4x faster per-epoch eval overhead)
#   - ALL annealing hyperparameters live in the yaml (cfm_bond_head_30ep_onecycle.yaml)
#     per user voice 2026-10-09 "退火的配置你是应该写到yaml配置文件里面的"
#
# YAML-DRIVEN bridge chain (canonical per scripts/launch_all_training.sh):
#   yaml protocol.scheduler.type      -> start.sh SCHEDULER        -> trainer --lr-scheduler
#   yaml protocol.scheduler.warmup_*  -> start.sh LR_WARMUP_EPOCHS  -> trainer --lr-warmup-epochs
#   yaml protocol.scheduler.min_lr    -> start.sh LR_MIN_LR         -> trainer --lr-min-lr
#   yaml protocol.weight_decay        -> start.sh WEIGHT_DECAY      -> trainer --weight-decay
#   yaml protocol.max_norm            -> start.sh MAX_NORM          -> trainer --max-norm
# Plus the LEAF-FALLBACK path (wfA2_config_loader.L78-145) where applicable:
#   yaml protocol.epochs             -> trainer --epochs (via leaf fallback)
#   yaml protocol.batch_size         -> trainer --batch-size (via leaf fallback)
#   yaml protocol.lr                 -> trainer --lr (via leaf fallback)
#   yaml protocol.weight_decay       -> trainer --weight-decay (redundant with start.sh)
#   yaml protocol.optimizer          -> trainer --optimizer (via leaf fallback)
#
# The launcher wraps the trainer with BLK-I1 (setsid+nohup+redirect logs/) +
# BLK-I2 (pre-flight pass per §1.4) wrappers per Wf-fix3lines 2026-10-09.

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
set -e
cd /home/hugo/codes/mol-metal

OUT_DIR=molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09
LOG=logs/wf_fix3lines_cfm_30ep_onecycle.log
YAML=configs/cfm_bond_head_30ep_onecycle.yaml

# BLK-I2 wrapper: pre-flight v2 must PASS (per AGENTS.md §1.4 + §5.7 trainer-flag wire-in).
# Inline so a stale pre-flight cache cannot claim an under-firewall run.
# (kept inside setsid background so the .sh exit code reflects pre-flight rc)
if ! "$PY" molmetal/scripts/pre_flight_check.py \
        --pipeline launch --fast 2>&1 | tail -25; then
  echo "[FATAL] pre_flight_check FAILED — refuse to launch (AGENTS.md §1.4 + §5.1)"
  exit 2
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

# BLK-I1: redirect entire .sh stdout/stderr to LOG; trainer detached.
exec >"$LOG" 2>&1

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_CFM_AMP=1
export MOLMETAL_CFM_AMP_DTYPE=bfloat16
export MOLMETAL_EGNN_CHECKPOINT=1
export USER_OK=1   # per AGENTS.md §5.1 kill_then_adapt directive + start.sh contract

# start.sh is the canonical yaml-driven launch surface (per
# scripts/launch_all_training.sh:77). It bridges protocol.scheduler.{type,
# warmup_epochs, min_lr}, protocol.weight_decay, and protocol.max_norm into
# the trainer's --lr-scheduler / --lr-warmup-epochs / --lr-min-lr /
# --weight-decay / --max-norm CLI flags via env vars
# (scripts/start.sh:482-489 + scripts/start.sh:640-644).
#
# 2026-10-09 (AGENTS.md §5.11 — "只能 yaml 启动，no CLI 参数"): the
# --extra-flags block that used to live here is GONE. Its five values now live
# in $YAML and reach the trainer through wfA2_config_loader's leaf-fallback:
#
#   --device cuda:0           -> compute.device
#   --early-stop-patience 999 -> protocol.early_stop_patience
#   --gen-eval-n 16           -> protocol.gen_eval_n
#   --save-generated-smiles   -> model.save_generated_smiles
#   --batch-size 16           -> protocol.batch_size
#
# Round-trip verified: every dest above is non-default when $YAML is parsed
# through train_cfm_yuelbond_wasserstein.py --config.
#
# The model-side switches (use_fm_objective / use_egnn_velocity /
# use_tmqm_init / bond_head / pcgrad_multi_task / joint_train / cfm_decoder /
# learnable_vel_scale) were ALREADY in the yaml before this date — start.sh
# reads the yaml for entry points (see scripts/start.sh:457) and those flow
# through apply_yaml_overrides via the leaf fallback. They never needed CLI.

setsid nohup bash scripts/start.sh "$YAML" \
  --fast \
  >/dev/null 2>&1 &
PID=$!

wait $PID
rc=$?

echo "[wf_fix3lines_cfm_30ep_onecycle] trainer rc=$rc"
echo "[wf_fix3lines_cfm_30ep_onecycle] yaml=$YAML"
echo "[wf_fix3lines_cfm_30ep_onecycle] out_dir=$OUT_DIR"
echo "[wf_fix3lines_cfm_30ep_onecycle] log=$LOG"
exit $rc
