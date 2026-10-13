#!/usr/bin/env bash
# BLK-C3 CONTROL arm — use_fm_objective=false, otherwise byte-identical to
# configs/cfm_fm_ab_2026_10_09.yaml. Both arms run on the LR-isolation-fixed
# trainer (fix landed 20:33), so the A/B differs in exactly one variable.
# The pre-16:36 baseline in cfm_blk_c5_repair differs in TWO (fm + isolation)
# and must not be used as the comparison for this question.
#
# WHY THIS FILE DELEGATES TO scripts/start.sh (and not to the trainer):
# `protocol.scheduler.*` in the YAML is read by NOTHING except start.sh, which
# turns it into the trainer's --lr-scheduler / --lr-warmup-epochs / --lr-min-lr
# CLI flags (scripts/start.sh:482-505, 640-661). The trainer does not read
# those env vars itself and its --lr-scheduler default is "none". So a launcher
# that calls the trainer directly with only --config SILENTLY RUNS CONSTANT-LR.
#
# That is not hypothetical: an earlier version of this file did exactly that
# and trained with no scheduler. Verified by dry-run that the corrected path
# emits `--lr-scheduler "onecycle"`. See also the note in
# scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh:15 — "the no-scheduler bug
# that prompted the restart".
#
# AGENTS.md §5.11  no hyperparameter appears on this command line; start.sh
#                 reads them all from $YAML.
# AGENTS.md §5.9   single venv (start.sh hardcodes the absolute .venv path).
# BLK-I1          setsid --wait so $? is start.sh's status, not setsid's.
#                 Without --wait setsid may fork and return 0 immediately.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

YAML="configs/cfm_fm_control_2026_10_09.yaml"
OUT="molmetal/reports/cfm_fm_control_2026_10_09"
LOG="$OUT/master.log"

mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_CFM_AMP=1
export MOLMETAL_CFM_AMP_DTYPE=bfloat16
export MOLMETAL_EGNN_CHECKPOINT=1
export USER_OK=1   # AGENTS.md §5.1 standing grant + start.sh contract

echo "[cfm_fm_ctrl] start $(date -Iseconds)  yaml=$YAML"
echo "[cfm_fm_ctrl] scheduler comes from start.sh; verify it in the dry-run output:"
echo "[cfm_fm_ctrl]   USER_OK=1 bash scripts/start.sh $YAML --fast --dry-run | grep lr-scheduler"

# pipefail dropped for this one command so PIPESTATUS[0] reads start.sh's real
# status; under errexit the capture below would never run (measured 2026-10-09).
set +o pipefail
setsid --wait bash scripts/start.sh "$YAML" --fast 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -o pipefail

echo "[cfm_fm_ctrl] exit=$rc  log=$LOG"
if [ "$rc" -eq 0 ]; then
  echo "[cfm_fm_ctrl] Read the verdict from $OUT/metrics.json — NOT from rc."
  echo "[cfm_fm_ctrl] gen_validity is the number under test; the fm=false baseline"
  echo "[cfm_fm_ctrl] collapsed 0.854 -> 0.062."
fi
exit "$rc"