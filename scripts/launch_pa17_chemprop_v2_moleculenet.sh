#!/usr/bin/env bash
# PA17 — chemprop v2 (official D-MPNN) on MoleculeNet under this repo's split.
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config + --out-dir and
# NOTHING else.  Every hyperparameter lives in
# configs/pa17_chemprop_v2_moleculenet.yaml.  Scanned by scripts/_yaml_only_gate.py.
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when a worker flushes after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/pa17_chemprop_v2_moleculenet.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/pa17_chemprop_v2_moleculenet"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[pa17][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[pa17] DRY RUN — not launching"
  echo "[pa17]   yaml:    $YAML"
  echo "[pa17]   command: $PY $REPO_ROOT/molmetal/scripts/train_chemprop_v2_moleculenet.py --config $YAML --out-dir $OUT_DIR"
  exit 0
fi

if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  # Explicit, loud, and recorded.  Only legitimate when the failing gate is
  # orthogonal to the code this run exercises — name it in
  # PA17_SKIP_PREFLIGHT_REASON so the log carries the justification.
  echo "[pa17][WARN] SKIP_PREFLIGHT=1 — pre-flight WAIVED by the caller." >&2
  echo "[pa17][WARN] reason: ${PA17_SKIP_PREFLIGHT_REASON:-<none given>}" >&2
  echo "[pa17][WARN] last pre-flight log (if any): $OUT_DIR/preflight.log" >&2
else
  if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
          --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
    echo "[pa17][FATAL] pre_flight_check FAILED — refusing to launch" >&2
    echo "[pa17]         see $OUT_DIR/preflight.log" >&2
    exit 2
  fi
  echo "[pa17] pre-flight PASS"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pa17] GPU state:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

export CUDA_VISIBLE_DEVICES="${PA17_GPU:-0}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR="$REPO_ROOT/.data/molmetal"
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant

echo "[pa17] starting $(date -Iseconds)"
setsid --wait nohup "$PY" "$REPO_ROOT/molmetal/scripts/train_chemprop_v2_moleculenet.py" \
  --config "$YAML" \
  --out-dir "$OUT_DIR" \
  >"$LOG" 2>&1
rc=$?
echo "[pa17] finished $(date -Iseconds) rc=$rc"
echo "[pa17] log=$LOG"
echo "[pa17] HONEST: rc=0 means the run COMPLETED. The verdict is each task's"
echo "[pa17]         test_auc_mean vs the literature constant carried in"
echo "[pa17]         metrics/by_metric/moleculenet_auc.json (aligned split key)."
echo "[pa17]         The pre-2026-10-19 constant lived in"
echo "[pa17]         moleculenet_auc_dmpnn_5seed.json, which is on the OLD key —"
echo "[pa17]         do not cite it against this run."
exit $rc
