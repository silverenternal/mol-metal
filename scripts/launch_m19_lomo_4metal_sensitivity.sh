#!/usr/bin/env bash
# BLK-D2 Step 3 — m19 LOMO drop-Rh sensitivity analysis.
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config and NOTHING else.
# Every hyperparameter lives in configs/m19_lomo_4metal_sensitivity.yaml. The
# protocol is a verbatim replication of scripts/launch_dmpnn_clean5_lomo.sh —
# which produced the 0.5575 +/- 0.0949 headline — with `metals` as the single
# changed variable. That one-variable guarantee is asserted by diffing the
# resolved argparse namespaces, not by reading this file; see the yaml header.
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when DataLoader workers flush after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/m19_lomo_4metal_sensitivity.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/m19_lomo_4metal_sensitivity"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[m19_4metal][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[m19_4metal] DRY RUN — not launching"
  echo "[m19_4metal]   yaml:    $YAML"
  echo "[m19_4metal]   command: $PY $REPO_ROOT/molmetal/scripts/train_metal_lomo_wasserstein.py --config $YAML --out-dir $OUT_DIR"
  echo "[m19_4metal]   note: --out-metric is NOT passed; the yaml's output.out_metric"
  echo "[m19_4metal]         points inside the report dir, so the 5-metal headline in"
  echo "[m19_4metal]         metrics/by_metric/ cannot be overwritten by this arm."
  exit 0
fi

# Pre-flight must PASS before the launch (AGENTS.md §1.4 + §5.1). Both §5.11
# gates live in the launch pipeline, so this also re-asserts yaml-only launch
# and the config<->trainer binding at launch time.
if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
        --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
  echo "[m19_4metal][FATAL] pre_flight_check FAILED — refusing to launch" >&2
  echo "[m19_4metal]         see $OUT_DIR/preflight.log" >&2
  exit 2
fi
echo "[m19_4metal] pre-flight PASS ($OUT_DIR/preflight.log)"

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBIZITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant

echo "[m19_4metal] starting $(date -Iseconds)"
# --wait is load-bearing: without it setsid may fork and return 0 immediately,
# so rc=$? below would be setsid's code, not the trainer's — and the wrapper
# would report "rc=0" while the run is still going. That falsifies line 69.
setsid --wait nohup "$PY" "$REPO_ROOT/molmetal/scripts/train_metal_lomo_wasserstein.py" \
  --config "$YAML" \
  --out-dir "$OUT_DIR" \
  >"$LOG" 2>&1
rc=$?
echo "[m19_4metal] finished $(date -Iseconds) rc=$rc"
echo "[m19_4metal] log=$LOG"
echo "[m19_4metal] HONEST: rc=0 means the run COMPLETED. The drop-Rh verdict is"
echo "[m19_4metal]         value / value_std / per_fold_auc_mean in $OUT_DIR/metrics.json."
exit $rc
