# Wf-302B Guard Report — bash `wait_until_epoch_reached()` in launch_wf124_p3_g5.sh

**Date:** 2026-09-29
**Workflow:** Wf-302B
**Context:** Per Wf-298E critical remaining #2 — bash `wait_until_epoch_reached()` still MISSING in `scripts/launch_wf124_p3_g5.sh`
**Script path:** `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh`

## 1. Function Added (verbatim)

```bash
# ---- Helper: poll pair log until target epoch is reached (Wf-302B guard) ----
# Refs: Wf-285D + Wf-291B — elapsed=0 MUST be initialised BEFORE the loop
# Refs: Wf-298E critical remaining #2 — bash wait_until_epoch_reached was MISSING
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-1200}"
  local start_s elapsed last_epoch
  start_s=$(date +%s)
  elapsed=0
  while true; do
    last_epoch=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$pair_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "$last_epoch" ] && [ "$last_epoch" -ge "$target_epoch" ]; then
      echo "Pair complete: last_epoch=$last_epoch >= target=$target_epoch"
      return 0
    fi
    elapsed=$(( $(date +%s) - start_s ))
    if [ "$elapsed" -ge "$max_wait_s" ]; then
      echo "TIMEOUT: last=$last_epoch elapsed=$((elapsed))s"
      return 1
    fi
    sleep 10
  done
}
```

## 2. Wire-In Point

The helper is **wired into `run_pair()`** (the per-pair runner) so every one of the 15 (metal, seed) pairs automatically gets the guard:

```bash
run_pair() {
  local metal=$1
  local seed=$2
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"
  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}
  echo "=== $(date '+%H:%M:%S') END ${metal} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "$rc" -ne 0 ]; then
    echo "FAIL: ${metal} seed=${seed} exited rc=${rc}"
    return "$rc"
  fi
  # Wf-302B guard: verify the pair actually reached target epoch (50)
  wait_until_epoch_reached "${pair_log}" 50 1200 || { echo "FAIL: ${metal} seed=${seed} did not reach epoch 50"; return 1; }
}
```

## 3. Critical Bug Fix: `elapsed=0` BEFORE Loop

Line `elapsed=0` is now set BEFORE the `while true; do` loop — exactly as required by Wf-285D + Wf-291B bug fixes. This prevents the "unbound variable" / stale-elapsed class of bugs.

## 4. Syntax Check

```
$ bash -n scripts/launch_wf124_p3_g5.sh && echo SYNTAX_OK
SYNTAX_OK
```

## 5. Dry-Run Verdict

```
$ bash scripts/launch_wf124_p3_g5.sh --dry-run 2>&1 | tail -10
  Os seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Os/seed1/
  Os seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Os/seed2/
  Rh seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed0/
  Rh seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed1/
  Rh seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Rh/seed2/
  Re seed=0  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed0/
  Re seed=1  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed1/
  Re seed=2  ->  results/wf124_p3_50epoch_full_sweep_g5/Re/seed2/

DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.
```

**Verdict:** DRY-RUN OK — all 15 (metal, seed) pairs enumerated correctly, no GPU execution triggered, script returns exit 0.

## 6. Checklist

| Item | Status |
|---|---|
| function_added | YES — `wait_until_epoch_reached` defined before main loop |
| syntax_check | PASS — `bash -n` returns `SYNTAX_OK` |
| dry_run_verdict | PASS — 15 pairs PLANNED, exit 0 |
| elapsed_init_present | YES — `elapsed=0` initialised BEFORE `while true; do` (fixes Wf-285D + Wf-291B) |
| wire_after_each_pair | YES — helper called at end of `run_pair()`, covering all 15 pairs |
| target_epoch | 50 (matches `--epochs 50` in COMMON_ARGS) |
| default_max_wait_s | 1200s (20 min, overrideable via 3rd arg) |
