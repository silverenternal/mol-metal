#!/usr/bin/env bash
# PA3 — chemprop v2 official D-MPNN cross-metal LOMO on the Paper-A cohort.
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config + --out-dir and
# NOTHING else.  Every hyperparameter lives in
# configs/pa3_chemprop_lomo_v1.yaml.  Scanned by scripts/_yaml_only_gate.py.
#
# What this measures (BLK-PA3): the cross-metal (LOMO) collapse of Paper A's
# Claim 1, re-measured with the official chemprop v2 message-passing backbone
# (d_h=300 / depth=3) on the SAME cohort and SAME ligand-disjoint split as the
# Morgan-FP + XGBoost probe in metrics/by_metric/metal_activity_transfer.json.
# If the collapse reproduces, it is the metal axis and not a weak backbone.
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when a worker flushes after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/pa3_chemprop_lomo_v1.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/pa3_chemprop_lomo"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[pa3][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[pa3] DRY RUN — not launching"
  echo "[pa3]   yaml:    $YAML"
  echo "[pa3]   command: $PY $REPO_ROOT/molmetal/scripts/train_chemprop_lomo_paper_cohort.py --config $YAML --out-dir $OUT_DIR"
  echo "[pa3]   note: the metrics JSON path comes from output.out_json in the yaml"
  echo "[pa3]         (molmetal/reports/pa3_chemprop_lomo/metrics.json), so the"
  echo "[pa3]         canonical metrics/by_metric/ copies are written separately."
  exit 0
fi

# Pre-flight must PASS before the launch (AGENTS.md §1.4 + §5.1).  Both §5.11
# gates live in the launch pipeline, so this also re-asserts yaml-only launch
# and the config<->trainer binding at launch time.
if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
        --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
  echo "[pa3][FATAL] pre_flight_check FAILED — refusing to launch" >&2
  echo "[pa3]         see $OUT_DIR/preflight.log" >&2
  exit 2
fi
echo "[pa3] pre-flight PASS ($OUT_DIR/preflight.log)"

# AGENTS.md §5.1 standing GPU grant — check free VRAM first.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pa3] GPU state:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR="$REPO_ROOT/.data/molmetal"
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant

echo "[pa3] starting $(date -Iseconds)"
# --wait is load-bearing: without it setsid may fork and return 0 immediately,
# so rc=$? below would be setsid's code, not the trainer's.
setsid --wait nohup "$PY" "$REPO_ROOT/molmetal/scripts/train_chemprop_lomo_paper_cohort.py" \
  --config "$YAML" \
  --out-dir "$OUT_DIR" \
  >"$LOG" 2>&1
rc=$?
echo "[pa3] finished $(date -Iseconds) rc=$rc"
echo "[pa3] log=$LOG"
echo "[pa3] HONEST: rc=0 means the run COMPLETED. The verdict is"
echo "[pa3]         per_fold[*].mean_auc / mean_auc_across_metals in"
echo "[pa3]         $OUT_DIR/metrics.json — compare it against the XGB probe in"
echo "[pa3]         metrics/by_metric/metal_activity_transfer.json."
exit $rc
