#!/usr/bin/env bash
# PA17b — Morgan-FP + MLP on MoleculeNet under the aligned split key.
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config + --out-dir and
# NOTHING else.  Every hyperparameter lives in
# configs/pa17_morgan_mlp_moleculenet.yaml.  Scanned by scripts/_yaml_only_gate.py.
#
# GPU 0 only (2026-10-16 user directive: "以后所有操作都上gpu0,不要再在gpu1上做").
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when a worker flushes after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/pa17_morgan_mlp_moleculenet.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/pa17_morgan_mlp_moleculenet"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[pa17b][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[pa17b] DRY RUN — not launching"
  echo "[pa17b]   yaml:    $YAML"
  echo "[pa17b]   command: $PY $REPO_ROOT/molmetal/scripts/train_moleculenet.py --config $YAML --out-dir $OUT_DIR"
  exit 0
fi

if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  # Explicit, loud, and recorded.  Only legitimate when the failing gate is
  # orthogonal to the code this run exercises — name it in
  # PA17B_SKIP_PREFLIGHT_REASON so the log carries the justification.
  echo "[pa17b][WARN] SKIP_PREFLIGHT=1 — pre-flight WAIVED by the caller." >&2
  echo "[pa17b][WARN] reason: ${PA17B_SKIP_PREFLIGHT_REASON:-<none given>}" >&2
  echo "[pa17b][WARN] last pre-flight log (if any): $OUT_DIR/preflight.log" >&2
else
  if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
          --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
    echo "[pa17b][FATAL] pre_flight_check FAILED — refusing to launch" >&2
    echo "[pa17b]         see $OUT_DIR/preflight.log" >&2
    exit 2
  fi
  echo "[pa17b] pre-flight PASS"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pa17b] GPU state:"
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

echo "[pa17b] starting $(date -Iseconds)"
setsid --wait nohup "$PY" "$REPO_ROOT/molmetal/scripts/train_moleculenet.py" \
  --config "$YAML" \
  --out-dir "$OUT_DIR" \
  >"$LOG" 2>&1
rc=$?
echo "[pa17b] finished $(date -Iseconds) rc=$rc"
echo "[pa17b] log=$LOG"
echo "[pa17b] HONEST: rc=0 means the run COMPLETED. The numbers are in"
echo "[pa17b]         $OUT_DIR/metrics.json; compare them only against the"
echo "[pa17b]         aligned D-MPNN run, never against the pre-alignment column."
exit $rc
