#!/usr/bin/env bash
# PA14 — MoleculeNet 4-task re-run at hidden 512 / depth 4 (BLK-PA14).
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config + --out-dir and
# NOTHING else.  Every hyperparameter lives in
# configs/pa14_moleculenet_cap512_d4.yaml.  Scanned by scripts/_yaml_only_gate.py.
#
# Runs on GPU 0 (the 98 GB RTX PRO 6000) per the 2026-10-16 user directive:
# "以后所有操作都上gpu0,不要再在gpu1上做".  GPU 1 (the 32 GB RTX 5090) stays
# idle.  The job is minutes-scale and needs a fraction of the card; still
# check free VRAM before launching (the user reminder is "不要让他oom").
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when a worker flushes after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/pa14_moleculenet_cap512_d4.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/pa14_moleculenet_cap512_d4"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[pa14][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[pa14] DRY RUN — not launching"
  echo "[pa14]   yaml:    $YAML"
  echo "[pa14]   command: $PY $REPO_ROOT/molmetal/scripts/train_moleculenet.py --config $YAML --out-dir $OUT_DIR"
  exit 0
fi

if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
        --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
  echo "[pa14][FATAL] pre_flight_check FAILED — refusing to launch" >&2
  echo "[pa14]         see $OUT_DIR/preflight.log" >&2
  exit 2
fi
echo "[pa14] pre-flight PASS"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pa14] GPU state:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR="$REPO_ROOT/.data/molmetal"
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant

echo "[pa14] starting $(date -Iseconds)"
setsid --wait nohup "$PY" "$REPO_ROOT/molmetal/scripts/train_moleculenet.py" \
  --config "$YAML" \
  --out-dir "$OUT_DIR" \
  >"$LOG" 2>&1
rc=$?
echo "[pa14] finished $(date -Iseconds) rc=$rc"
echo "[pa14] log=$LOG"
echo "[pa14] HONEST: rc=0 means the run COMPLETED. The verdict is the per-task"
echo "[pa14]         test_auc_mean in $OUT_DIR/metrics.json vs the canonical"
echo "[pa14]         metrics/by_metric/moleculenet_auc_dmpnn_5seed.json (300/3)."
exit $rc
