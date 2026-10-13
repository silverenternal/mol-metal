#!/usr/bin/env bash
# BLK-C-PERF: settle pad_groups with McNemar on paired per-cloud outcomes.
#
# ONE ARM PER PROCESS is the point of this launcher, not an implementation
# detail. The first attempt ran both arms in one process and died of CUDA OOM
# at 04:57:04 on the 31 GB card: the grouped arm had finished but still held
# 30.65 GB when the padded arm started. Process exit releases VRAM for
# certain; an in-process `del` + `empty_cache()` can be skipped by any early
# return or exception.
#
# AGENTS.md 5.9: .venv/bin/python only.
# AGENTS.md 5.11: this wrapper carries NO hyperparameter. The cohort size and
# the seed live on the script's argparse defaults and are passed explicitly
# below only so they appear in the log.
#
#   bash scripts/launch_cfm_pad_groups_mcnemar.sh --dry-run   # print, launch nothing
#   bash scripts/launch_cfm_pad_groups_mcnemar.sh             # run both arms, sequentially
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"                      # 5.9
SCRIPT="$REPO/scripts/cfm_pad_groups_mcnemar.py"
OUT="$REPO/molmetal/reports/cfm_pad_groups_mcnemar"
LOG_DIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
LOG="$LOG_DIR/cfm_pad_groups_mcnemar.log"

# 192 clouds is the cohort the CRASHED run used and the one the surviving
# grouped arm completed (n_out 37, decoded 37, 1892.7s). Keeping it means the
# new grouped arm is comparable with that number rather than being a second,
# different measurement.
N_CLOUDS=192
SEED=0

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

for f in "$PY" "$SCRIPT"; do
  [[ -e "$f" ]] || { echo "FAIL: missing $f" >&2; exit 1; }
done

if [[ "$DRY" == "1" ]]; then
  echo "DRY RUN -- nothing launched"
  echo "  script  : $SCRIPT"
  echo "  out     : $OUT"
  echo "  log     : $LOG"
  echo "  arms    : grouped then padded, ONE PROCESS EACH (the OOM fix)"
  echo "  n_clouds: $N_CLOUDS   seed: $SEED"
  echo "  test    : McNemar exact two-sided on per-cloud pass vectors"
  echo "  GPU     : 0 (standing directive 只用gpu0吧)"
  exit 0
fi

mkdir -p "$OUT" "$(dirname "$LOG")"

{
  echo "=== CFM pad_groups McNemar A/B, $(date -Is) ==="
  echo "cohort: $N_CLOUDS clouds, seed $SEED, midpoint/50, one arm per process"
  for ARM in grouped padded; do
    echo "--- arm $ARM ---"
    # Each invocation is a separate process: on exit, the VRAM goes back.
    # No pipefail on any pipe (memory: wrapper-pipefail-sigpipe-silent-death).
    "$PY" "$SCRIPT" \
      --arm "$ARM" \
      --seed "$SEED" \
      --n-clouds "$N_CLOUDS" \
      --device cuda:0 \
      --out "$OUT/arm_${ARM}.json" 2>&1 || echo "ARM $ARM FAILED rc=$?"
    echo "--- arm $ARM done ---"
  done
  echo "--- aggregating ---"
  # 2026-10-12 FIX. `|| echo ...FAILED` swallowed the failure and the script
  # went on to print "SWEEP DONE" regardless -- so an arm that died of OOM
  # still produced a completion banner. This is the third time today a CFM
  # launcher has reported success while the thing it was gating had not
  # happened (the aromatic sweep printed DONE twice with all arms void; the
  # REPO-path bug printed DONE with both arms dead in under a second).
  #
  # The aggregate is the ONLY step that knows whether both arms exist and have
  # the same length, so it is made the gate rather than an echo.
  if ! "$PY" "$SCRIPT" --aggregate 2>&1; then
    echo "FAIL: aggregate failed -- at least one arm is missing or "
    echo "      misaligned. NOT a completed sweep. See the traceback above."
    exit 1
  fi
  echo "SWEEP DONE $(date -Is)"
} > "$LOG" 2>&1

echo "launched detached; log: $LOG"
nohup tail -f "$LOG" >/dev/null 2>&1 &