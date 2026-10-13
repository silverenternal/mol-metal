#!/usr/bin/env bash
# CFM gen-collapse A/B arm — `use_fm_objective: true`.
#
# AGENTS.md §5.11 (2026-10-09): this wrapper passes --config and NOTHING else.
# Every hyperparameter lives in configs/cfm_gen_collapse_fm.yaml.
#
# SINGLE-VARIABLE A/B. The only functional difference from the completed run's
# config (configs/cfm_blk_c5_repair_2026_10_09.yaml) is `model.use_fm_objective:
# false -> true`. Verified by round-trip, not by reading this file: parse_args()
# on both yamls yields exactly one differing dest out of 31 checked, with the
# other 30 identical.
#
# NOT LAUNCHED. This wrapper is written and dry-run-checked only; whether it
# runs is the user's call (AGENTS.md §5.1 / §9).
#
# HONEST FRAMING (AGENTS.md §5.2). rc=0 means the run COMPLETED. It does NOT
# mean the collapse is fixed. Both arms are n_seeds=1, so a single 30-epoch
# run cannot separate `use_fm_objective` from run-to-run variance. Do not
# describe this arm as a lift until both arms sit on disk at the same epoch
# cadence AND a multi-seed sweep exists.
#
# ⚠️ pct_start CAVEAT (MEASURED 2026-10-09, verification pass): this yaml
# declares `protocol.scheduler.pct_start: 0.35`, but scripts/start.sh does not
# list `--lr-pct-start` in its `_EXTRA_CANDIDATES` probe, so the flag is dropped
# from the emitted command and the trainer keeps its hardcoded 0.3. The baseline
# arm is identically inert, so the A/B stays single-variable — but neither run
# had a 35% warmup. Fixing that gap would change BOTH arms; it is not done here.
#
# NO `set -o pipefail`: per the SIGPIPE lesson, a pipefail wrapper dies silently
# when DataLoader workers flush after the trainer exits.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"          # AGENTS.md §5.9 single-venv
YAML="$REPO_ROOT/configs/cfm_gen_collapse_fm.yaml"
OUT_DIR="$REPO_ROOT/molmetal/reports/cfm_gen_collapse_fm"
LOG="$OUT_DIR/train.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

if [[ ! -f "$YAML" ]]; then
  echo "[cfm_gen_collapse_fm][FATAL] config missing: $YAML" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[cfm_gen_collapse_fm] DRY RUN — not launching"
  echo "[cfm_gen_collapse_fm]   yaml:    $YAML"
  echo "[cfm_gen_collapse_fm]   command: $PY $REPO_ROOT/molmetal/scripts/train_cfm_yuelbond_wasserstein.py --config $YAML --out-dir $OUT_DIR"
  echo "[cfm_gen_collapse_fm]   note: no trainer flag is passed here. The scheduler"
  echo "[cfm_gen_collapse_fm]         (--lr-scheduler/--lr-warmup-epochs/--lr-min-lr/"
  echo "[cfm_gen_collapse_fm]         --max-norm) and the events path are bridged by"
  echo "[cfm_gen_collapse_fm]         scripts/start.sh from the SAME yaml, which is why"
  echo "[cfm_gen_collapse_fm]         this wrapper calls start.sh rather than the trainer."
  echo "[cfm_gen_collapse_fm]         Calling the trainer directly would silently run"
  echo "[cfm_gen_collapse_fm]         lr_scheduler=none and break the A/B."
  echo "[cfm_gen_collapse_fm]   A/B: single variable vs cfm_blk_c5_repair_2026_10_09 —"
  echo "[cfm_gen_collapse_fm]         model.use_fm_objective false -> true. NOT a lift."
  exit 0
fi

# Pre-flight must PASS before the launch (AGENTS.md §1.4 + §5.1). Both §5.11
# gates live in the launch pipeline, so this also re-asserts yaml-only launch
# and the config<->trainer binding at launch time.
if ! "$PY" "$REPO_ROOT/molmetal/scripts/pre_flight_check.py" \
        --pipeline launch >"$OUT_DIR/preflight.log" 2>&1; then
  echo "[cfm_gen_collapse_fm][FATAL] pre_flight_check FAILED — refusing to launch" >&2
  echo "[cfm_gen_collapse_fm]         see $OUT_DIR/preflight.log" >&2
  exit 2
fi
echo "[cfm_gen_collapse_fm] pre-flight PASS ($OUT_DIR/preflight.log)"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER_OK=1   # AGENTS.md §5.1 standing GPU grant + start.sh contract

echo "[cfm_gen_collapse_fm] starting $(date -Iseconds)"
# --wait is load-bearing: without it setsid may fork and return 0 immediately,
# so rc=$? below would be setsid's code, not the trainer's — and the wrapper
# would report "rc=0" while the run is still going.
setsid --wait nohup bash "$REPO_ROOT/scripts/start.sh" "$YAML" \
  >"$LOG" 2>&1
rc=$?
echo "[cfm_gen_collapse_fm] finished $(date -Iseconds) rc=$rc"
echo "[cfm_gen_collapse_fm] log=$LOG"
echo "[cfm_gen_collapse_fm] HONEST: rc=0 means the run COMPLETED, not that the"
echo "[cfm_gen_collapse_fm]         collapse is fixed. Read gen_validity /"
echo "[cfm_gen_collapse_fm]         decode_ratio from $OUT_DIR/per_epoch_metrics.csv"
echo "[cfm_gen_collapse_fm]         and compare against"
echo "[cfm_gen_collapse_fm]         molmetal/reports/cfm_blk_c5_repair_2026_10_09/"
echo "[cfm_gen_collapse_fm]         at the SAME epoch before claiming anything"
echo "[cfm_gen_collapse_fm]         (AGENTS.md §5.2). n_seeds=1 on both arms."
exit $rc
