#!/usr/bin/env bash
# PA15 — metal-conditioned cross-metal LOMO ablation (BLK-PA15).
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config + --out-dir and
# NOTHING else.  Every hyperparameter lives in the per-arm YAML under
# configs/pa15_metalcond_lomo_*.yaml.  Scanned by scripts/_yaml_only_gate.py.
#
# Three arms, identical cohort / LOMO split / seeds, differing only in the metal
# input to the pooled head:
#   none  — metal-agnostic PA3 scorer (baseline)
#   token — learnable metal embedding (held-out row is UNTRAINED in LOMO)
#   desc  — fixed periodic-table descriptor vector through a shared projection
#           (the only arm whose held-out representation is known at test time)
#
# The question (BLK-PA15): does explicit metal conditioning recover any of the
# cross-metal transfer, i.e. can Paper A offer a method and not only a benchmark?
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when a worker flushes after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
TRAINER="$REPO_ROOT/molmetal/scripts/train_chemprop_lomo_paper_cohort.py"

ARMS=(none token desc)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

for ARM in "${ARMS[@]}"; do
  YAML="$REPO_ROOT/configs/pa15_metalcond_lomo_${ARM}.yaml"
  if [[ ! -f "$YAML" ]]; then
    echo "[pa15][FATAL] config missing: $YAML" >&2
    exit 1
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[pa15] DRY RUN — not launching"
  for ARM in "${ARMS[@]}"; do
    YAML="$REPO_ROOT/configs/pa15_metalcond_lomo_${ARM}.yaml"
    OUT_DIR="$REPO_ROOT/molmetal/reports/pa15_metalcond_lomo_${ARM}"
    echo "[pa15]   [$ARM] $PY $TRAINER --config $YAML --out-dir $OUT_DIR"
  done
  echo "[pa15]   note: the metrics JSON path comes from output.out_json in each yaml"
  exit 0
fi

# Pre-flight must PASS before the launch (AGENTS.md §1.4 + §5.1).  Both §5.11
# gates live in the launch pipeline, so this also re-asserts yaml-only launch
# and the config<->trainer binding at launch time.
mkdir -p "$REPO_ROOT/molmetal/reports/pa15_metalcond_lomo_none"
if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
        --pipeline launch >"$REPO_ROOT/molmetal/reports/pa15_metalcond_lomo_none/preflight.log" 2>&1; then
  echo "[pa15][FATAL] pre_flight_check FAILED — refusing to launch" >&2
  echo "[pa15]         see molmetal/reports/pa15_metalcond_lomo_none/preflight.log" >&2
  exit 2
fi
echo "[pa15] pre-flight PASS"

# AGENTS.md §5.1 standing GPU grant — check free VRAM first.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pa15] GPU state:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR="$REPO_ROOT/.data/molmetal"
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant

RC=0
for ARM in "${ARMS[@]}"; do
  YAML="$REPO_ROOT/configs/pa15_metalcond_lomo_${ARM}.yaml"
  OUT_DIR="$REPO_ROOT/molmetal/reports/pa15_metalcond_lomo_${ARM}"
  LOG="$OUT_DIR/train.log"
  mkdir -p "$OUT_DIR"
  echo "[pa15] arm=$ARM starting $(date -Iseconds)"
  # --wait is load-bearing: without it setsid may fork and return 0 immediately.
  setsid --wait nohup "$PY" "$TRAINER" \
    --config "$YAML" \
    --out-dir "$OUT_DIR" \
    >"$LOG" 2>&1
  rc=$?
  echo "[pa15] arm=$ARM finished $(date -Iseconds) rc=$rc (log=$LOG)"
  if [[ $rc -ne 0 ]]; then
    RC=$rc
  fi
done

echo "[pa15] all arms done; aggregate rc=$RC"
echo "[pa15] HONEST: rc=0 means the runs COMPLETED. The verdict is the paired"
echo "[pa15]         per-(metal,seed) AUC difference vs the none arm — see each"
echo "[pa15]         molmetal/reports/pa15_metalcond_lomo_*/metrics.json."
exit $RC
