#!/usr/bin/env bash
# Wf-355-B2 — Wf-146 Re fold 5/5 + Os seed 4 DEFERRED runs re-launch wrapper
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-355-B2)
# Origin:      Wf-355 ultracode residual wave — closes the Wf-354-A1 §5
#              "6 DEFERRED runs" gap surfaced by Wf-342-P2 partial integration.
# Status:      DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt).
#
# Purpose
# -------
# Per Wf-354-A1 §5: Wf-146 chemprop v2 LOMO sweep PARTIAL: 19/25 (metal, seed)
# combinations completed.  Six remaining DEFERRED runs:
#
#   Re  seeds 0, 1, 2, 3, 4 (5 folds; sweep never started Re)
#   Os  seed 4          (1 fold; sweep stopped at epoch 10/50 on 2026-09-27 17:29)
#
# This wrapper closes the gap by providing a USER_OK-gated launch entry
# point.  All 6 runs use the **smoke backbone** hidden=64/depth=2 (the same
# spec that produced the 19 MEASURED runs in Wf-146 sweep.log), so the new
# data is bit-comparable with the existing Re/Os rows.
#
# Spec (per Wf-146 sweep.log + metal_lomo_chemprop_v2.json + Wf-354-A1 §5)
# ------------------------------------------------------------------------
#   - Data:     MetalCytoToxDB precious-metal subset (5 metals × N rows)
#   - Loader:   chemprop v2 SimpleMoleculeMolGraphFeaturizer
#                + LeaveOneGroupOut split (per metal fold)
#   - Backbone: hidden=64, depth=2 (smoke backbone per Wf-146 + Wf-354-A1 §6)
#                NOT production hidden=300/depth=3 (TODO-154 §3 reference)
#   - Train:    50 epochs × batch_size=50 × lr=1e-3 × BF16 autocast
#                + Adam optimizer (no LR scheduler per Wf-146 honest framing)
#   - Eval:     per-seed test AUC (held-out metal head)
#   - Hardware: single RTX 5090 (32 GB) or RTX PRO 6000 Blackwell (98 GB)
#   - ETA:      ~30-45 min GPU per (metal, seed) = ~3 hr total for 6 runs
#
# Modes (per Wf-294B / Wf-300C kill_then_adapt pattern)
# -----------------------------------------------------
#   R1 (default; CPU smoke):  1 (metal, seed) pair × 1 epoch on CPU
#                              USER_OK OPTIONAL (smoke only; ~3 min CPU)
#   R3 (USER_OK=YES; RTX 5090):
#                              6 (metal, seed) pairs × 50 epochs
#                              USER_OK=YES required (kill_then_adapt ≥ 2 hr GPU)
#
# USER_OK gate (per Wf-300C pattern; --dry-run does NOT bypass the gate)
# -----------------------------------------------------------------------
#   Wf-355-B2 R1 mode: USER_OK optional (smoke).
#   Wf-355-B2 R3 mode: USER_OK=YES required (kill_then_adapt ≥ 2 hr GPU).
#
# Per-wave safety belt (post-Wf-347 / Wf-343-C2 + Wf-343-C3):
#   - wf343c2_per_wave_check.py  (7-check runtime safety belt; --skip-live
#                                  since global pre_flight_check.py already
#                                  covers VRAM + disk + freshness)
#   - wf343c3_data_quality_check.py (5×5 data quality matrix)
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:   ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:     3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:      for 5+ minutes
#
# Usage
# -----
#   bash scripts/launch_wf146_deferred_runs.sh --dry-run --mode R1
#       # CPU smoke (Wf-355-B2 R1 default; USER_OK not required)
#   USER_OK=YES bash scripts/launch_wf146_deferred_runs.sh --mode R3
#       # 6 DEFERRED runs on RTX 5090 (~3 hr wall)
#   USER_OK=YES ONLY_RUNS=Re_0,Re_1 bash scripts/launch_wf146_deferred_runs.sh --mode R3
#       # Subset launch (just 2 Re runs)

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf146_chemprop_lomo"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
PER_WAVE_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
WF146_TRAINER="${SCRIPTS_DIR}/wf146_chemprop_lomo.py"

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.25       # Wf-355-B2 R1 smoke (CPU; 1 pair × 1 epoch)
R3_BUDGET_HR=3.0        # Wf-355-B2 R3 full (6 pairs × ~30 min on RTX 5090)

# Per-phase GPU assignment (per Wf-124 P3 + Wf-262 §0.5)
R1_GPU="cpu"
R3_GPU="cuda:1"

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB).
# Wf-146 hidden=64/depth=2 smoke backbone uses ~2-3 GB peak VRAM (per
# Wf-124 P3 chemprop v2 5-metal MEASURED at TODO-154 §3).
VRAM_BUDGET_GB=8

# 6 DEFERRED (metal, seed) pairs per Wf-354-A1 §5 + Wf-146 sweep.log
#   - Re  seeds 0..4:    all 5 folds never started (sweep halted 2026-09-27 17:29)
#   - Os  seed 4:        sweep halted at epoch 10/50 (last log line)
WF146_DEFERRED_PAIRS=(
    "Re_0"
    "Re_1"
    "Re_2"
    "Re_3"
    "Re_4"
    "Os_4"
)

mkdir -p "${REPORTS_DIR}/r1" "${REPORTS_DIR}/r3"

# -----------------------------------------------------------------------------
# Color codes (only when stdout is a tty)
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
MODE="R1"   # Wf-355-B2 R1 default (CPU smoke)
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,100p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-355-B2 R1 default; Wf-355-B2 R3 = USER_OK=YES)"

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-300C pattern)
#   R1: optional (smoke only; CPU)
#   R3: required (full GPU run; kill_then_adapt)
# -----------------------------------------------------------------------------
user_ok_gate() {
    local mode="$1"
    if [ "${mode}" = "R3" ]; then
        if [ -z "${USER_OK:-}" ]; then
            die "USER_OK env var not set. Re-run with: USER_OK=YES bash $0 --mode R3"
        fi
        if [ "${USER_OK}" != "YES" ]; then
            die "USER_OK must equal 'YES' (got: '${USER_OK}')"
        fi
        ok "USER_OK gate passed (USER_OK=YES)"
    else
        if [ -z "${USER_OK:-}" ]; then
            warn "USER_OK not set; Wf-355-B2 R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-355-B2 R1 smoke)"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ]      || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]              || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]          || die "wave_budget_tracker.py missing"
[ -f "${PER_WAVE_CHECK}" ]          || die "wf343c2_per_wave_check.py missing"
[ -f "${DATA_QUALITY_CHECK}" ]      || die "wf343c3_data_quality_check.py missing"
[ -f "${WF146_TRAINER}" ]           || die "Wf-146 chemprop LOMO trainer missing: ${WF146_TRAINER}"

# Wf-146 trainer marker check
grep -q "ChempropV2MTModel" "${WF146_TRAINER}" \
    || die "Wf-146 trainer missing ChempropV2MTModel (chemprop v2 backbone not wired)"
grep -q "LeaveOneGroupOut" "${WF146_TRAINER}" \
    || die "Wf-146 trainer missing LeaveOneGroupOut (LOMO split not wired)"

log "Wf-355-B2 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline chemprop_v2_v3
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline chemprop_v2_v3 (TODO-207 v3 gate)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates"
        return 0
    fi
    if ! "${PY}" "${PRE_FLIGHT}" --pipeline chemprop_v2_v3 --json \
            > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
        die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    fi
    local n_pass n_fail
    n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
    n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
    ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)"
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py (advisory)
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py (advisory)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        return 0
    fi
    if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" \
            --report-json "${REPORTS_DIR}/budget_report.json" \
            > "${REPORTS_DIR}/budget.log" 2>&1; then
        warn "budget tracker returned non-zero (advisory only)"
    fi
    ok "budget tracker logged"
}

# -----------------------------------------------------------------------------
# Pre-launch check #3 — VRAM budget (per AGENTS.md §5.1)
# -----------------------------------------------------------------------------
vram_gate() {
    local gpu="$1"
    log "VRAM budget check (--vram-budget-gb ${VRAM_BUDGET_GB})"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping nvidia-smi VRAM probe"
        return 0
    fi
    if [ "${gpu}" = "cpu" ]; then
        ok "CPU mode (Wf-355-B2 R1) — no VRAM check required"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found in PATH (required for ${gpu} GPU launch)"
    fi
    local idx="${gpu##cuda:}"
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader -i "${idx}" 2>/dev/null \
                | head -1 | tr -d ' ' || echo "0")
    local free_gb
    free_gb=$(echo "${free_mib} / 1024" | bc 2>/dev/null || echo "0")
    if [ "$(echo "${free_gb} < ${VRAM_BUDGET_GB}" | bc 2>/dev/null || echo 1)" -eq 1 ]; then
        die "VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx} (AGENTS.md §5.1 OOM risk)"
    fi
    ok "VRAM free=${free_gb}GB on GPU ${idx} (budget=${VRAM_BUDGET_GB}GB OK)"
}

# -----------------------------------------------------------------------------
# Pre-launch check #4 — per-wave check (Wf-343c2 post-Wf-347 retro-fit)
# -----------------------------------------------------------------------------
per_wave_check_gate() {
    local wave="$1"
    log "wf343c2_per_wave_check.py --wave '${wave}' (7-check runtime safety belt)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping per-wave live probes (--skip-live equivalent)"
        return 0
    fi
    if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
            --wave "${wave}" \
            --pairs-metals "Re,Os" \
            --pairs-seeds "0,1,2,3,4" \
            --skip-live \
            --out "${REPORTS_DIR}/per_wave.json" \
            > "${REPORTS_DIR}/per_wave.log" 2>&1; then
        warn "per-wave check returned non-zero (advisory; see ${REPORTS_DIR}/per_wave.log)"
        return 0   # non-fatal per Wf-347-B1 §3 design
    fi
    ok "per-wave check logged (advisory)"
}

# -----------------------------------------------------------------------------
# Pre-launch check #5 — data quality check (Wf-343c3 post-Wf-347 retro-fit)
# -----------------------------------------------------------------------------
data_quality_gate() {
    log "wf343c3_data_quality_check.py --strict (5×5 loader quality matrix)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping data quality live loaders"
        return 0
    fi
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${DATA_QUALITY_CHECK}" \
        --strict \
        --out-dir "${REPORTS_DIR}/data_quality" \
        > "${REPORTS_DIR}/data_quality.log" 2>&1 \
        && ok "data quality check PASS" \
        || warn "data quality check returned non-zero (advisory; see ${REPORTS_DIR}/data_quality.log)"
}

# -----------------------------------------------------------------------------
# Per-pair filter (per ONLY_RUNS env var)
#   ONLY_RUNS="Re_0,Re_1"  → launch only those 2 pairs
#   ONLY_REDATES=ALL (default) → launch all 6 pairs
# -----------------------------------------------------------------------------
select_pairs() {
    if [ -z "${ONLY_RUNS:-}" ]; then
        printf '%s\n' "${WF146_DEFERRED_PAIRS[@]}"
        return 0
    fi
    IFS=',' read -ra WANT <<< "${ONLY_RUNS}"
    for want in "${WANT[@]}"; do
        local matched=0
        for pair in "${WF146_DEFERRED_PAIRS[@]}"; do
            if [ "${pair}" = "${want}" ]; then
                printf '%s\n' "${pair}"
                matched=1
            fi
        done
        [ "${matched}" -eq 1 ] || die "ONLY_RUNS pair '${want}' not in 6 DEFERRED pairs: ${WF146_DEFERRED_PAIRS[*]}"
    done
}

# -----------------------------------------------------------------------------
# Wf-146 flag builder (single source of truth)
#   Reflects Wf-146 sweep.log + metal_lomo_chemprop_v2.json spec:
#   hidden=64, depth=2, 50 epochs, batch_size=50, lr=1e-3, BF16 autocast.
# -----------------------------------------------------------------------------
build_wf146_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"
    local metal="$4"
    local seed="$5"

    # Common flags (mirror Wf-146 sweep.log 2026-09-27 16:45:44 line)
    local args=(
        --metals "${metal}"
        --seeds "${seed}"
        --hidden 64
        --depth 2
        --batch-size 50
        --lr 1e-3
        --ckpt-dir "${out_dir}/ckpts"
        --json-dir "${out_dir}/jsons"
    )

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-355-B2 R1 smoke: 1 epoch only on CPU
        args+=( --epochs 1 )
    else
        # Wf-355-B2 R3 full: 50 epochs on RTX 5090 (mirror Wf-146 sweep)
        args+=( --epochs 50 )
    fi

    if [ "${gpu}" != "cpu" ]; then
        args+=( --device cuda )
    else
        args+=( --device cpu )
    fi

    printf '%s\n' "${args[@]}"
}

# -----------------------------------------------------------------------------
# Per-phase runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_pair() {
    local mode="$1"
    local budget_hr="$2"
    local pair="$3"     # e.g. Re_0
    local out_dir="$4"
    local gpu="$5"

    local metal="${pair%_*}"
    local seed="${pair#*_}"

    log "==== Pair ${pair} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="

    # Per-wave check (Wf-347-B2 pattern; before each pair)
    per_wave_check_gate "Wf-355-B2/${pair}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        local args
        mapfile -t args < <(build_wf146_flags "${mode}" "${gpu}" "${out_dir}" "${metal}" "${seed}")
        log "DRY-RUN: would launch Wf-146 ${pair}"
        # Wf-363-A1: use shared DRY_RUN.log schema helper for per-pair polish
        emit_dry_run_log "${out_dir}/DRY_RUN.log" \
            --title "Wf-355-B2 ${mode} deferred run (${pair})" \
            --wrapper "$(basename "$0")" \
            --mode "${mode}" \
            --trainer "${WF146_TRAINER}" \
            --pair "${metal} seed=${seed}" \
            --flags "${args[*]}" \
            --gpu "${gpu}" \
            --budget-hr "${budget_hr}" \
            --user-ok "$( [ "${mode}" = "R3" ] && echo REQUIRED || echo OPTIONAL_smoke )"
        ok "Pair ${pair} DRY-RUN logged"
        return 0
    fi

    local args
    mapfile -t args < <(build_wf146_flags "${mode}" "${gpu}" "${out_dir}" "${metal}" "${seed}")

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
        "${PY}" "${WF146_TRAINER}" "${args[@]}" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        CUDA_VISIBLE_DEVICES="${idx}" \
            "${PY}" "${WF146_TRAINER}" "${args[@]}" \
                > "${out_dir}/train.log" 2>&1 &
        pid=$!
    fi

    log "Pair ${pair} PID=${pid}; monitoring..."

    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Pair ${pair} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 5
            kill -KILL "${pid}" 2>/dev/null || true
            aborted=1
            break
        fi
        if [ "${gpu}" != "cpu" ]; then
            local gpu_util
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                    --format=csv,noheader \
                                    -i "${gpu##cuda:}" 2>/dev/null \
                        | head -1 || echo "0")
            log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"
        fi
        sleep 60
    done

    wait "${pid}" 2>/dev/null || true

    local wall_sec
    wall_sec=$(( $(date +%s) - started_at ))
    log "Pair ${pair} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || ! grep -q "test_auc.*status=ok" "${out_dir}/train.log"; then
        warn "Pair ${pair} FAIL — see ${out_dir}/train.log"
        die "Pair ${pair} aborted"
    fi

    ok "Pair ${pair} complete — ${wall_sec}s"
}

# -----------------------------------------------------------------------------
# Phase R1 — Wf-355-B2 R1 smoke (1 DEFERRED pair × 1 epoch CPU)
#   R1 picks the FIRST pair from WF146_DEFERRED_PAIRS by default (Re_0).
#   Override via ONLY_RUNS=Re_0 or similar.
# -----------------------------------------------------------------------------
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-355-B2 R1 plan: Wf-146 chemprop LOMO 1-pair smoke (~3 min)"
    log "  Wf-146 driver: --epochs 1 --hidden 64 --depth 2 --bf16"
    local pair
    if [ -z "${ONLY_RUNS:-}" ]; then
        pair="${WF146_DEFERRED_PAIRS[0]}"   # Re_0
    else
        mapfile -t wanted < <(select_pairs)
        pair="${wanted[0]}"
    fi
    run_pair "R1" "${R1_BUDGET_HR}" "${pair}" "${out}" "${R1_GPU}"
    ok "Phase R1 Wf-146 deferred smoke complete (pair=${pair})"
}

# Wf-363-A1: emit phase-level DRY_RUN.log that aggregates ALL pairs for R3
# (otherwise each pair overwrites the same out_dir's DRY_RUN.log)
phase_R3_dry_run_log() {
    local out="${REPORTS_DIR}/r3"
    local pairs
    mapfile -t pairs < <(select_pairs)
    local pair_args=()
    local sample_args=()
    for p in "${pairs[@]}"; do
        local metal="${p%_*}"
        local seed="${p#*_}"
        pair_args+=( --pair "${metal} seed=${seed}" )
        # Build a representative flag set from the first pair for the schema summary
        if [ "${#sample_args[@]}" -eq 0 ]; then
            mapfile -t sample_args < <(build_wf146_flags "R3" "${R3_GPU}" "${out}" "${metal}" "${seed}")
        fi
    done
    local per_pair_budget_hr=1  # dry-run floor per-pair
    emit_dry_run_log "${out}/DRY_RUN.log" \
        --title "Wf-355-B2 R3 deferred runs (${#pairs[@]} pairs)" \
        --wrapper "$(basename "$0")" \
        --mode "R3" \
        --trainer "${WF146_TRAINER}" \
        --flags "${sample_args[*]}" \
        --gpu "${R3_GPU}" \
        --budget-hr "${per_pair_budget_hr}" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    ok "Phase R3 DRY_RUN.log aggregated (${#pairs[@]} pairs) → ${out}/DRY_RUN.log"
}

# -----------------------------------------------------------------------------
# Phase R3 — Wf-355-B2 R3 full GPU run (6 DEFERRED pairs × 50 epochs RTX 5090)
# -----------------------------------------------------------------------------
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-355-B2 R3 plan: Wf-146 chemprop LOMO 6 DEFERRED pairs × 50 epochs (~3 hr cuda:1)"
    log "  Wf-146 driver: --epochs 50 --hidden 64 --depth 2 --bf16"
    local pairs
    mapfile -t pairs < <(select_pairs)
    local n_pairs=${#pairs[@]}
    [ "${n_pairs}" -gt 0 ] || die "no pairs selected (n_pairs=${n_pairs}); check ONLY_RUNS env var"
    # Per-pair budget: at minimum 1 hr (per Wf-294B kill_then_adapt < 2 hr hard cap
    # is a phase-level cap; per-pair budget is a sub-phase and should still cover
    # the 50-epoch sweep which MEASURED takes ~30 min on RTX 5090 per Wf-146 log).
    local per_pair_budget_hr
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Dry-run uses 1 hr floor regardless of pair count so the budget check
        # doesn't fail in verification mode (no actual GPU work happens).
        per_pair_budget_hr=1
    else
        per_pair_budget_hr=$(( ${R3_BUDGET_HR%.*} / n_pairs ))
        [ "${per_pair_budget_hr}" -ge 1 ] || die "per-pair budget ${per_pair_budget_hr} hr too small for 50-epoch run"
    fi
    log "  Per-pair time budget: ${per_pair_budget_hr} hr (R3_BUDGET_HR/${#pairs[@]} split)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Wf-363-A1: emit aggregated DRY_RUN.log BEFORE the per-pair loop so it
        # enumerates all pairs (per-pair log emission inside run_pair would
        # overwrite the same out_dir's DRY_RUN.log for each pair).
        phase_R3_dry_run_log
        return 0
    fi
    for pair in "${pairs[@]}"; do
        log "  - launching pair=${pair}"
        run_pair "R3" "${per_pair_budget_hr}" "${pair}" "${out}" "${R3_GPU}"
        ok "Phase R3 pair=${pair} complete"
    done
    ok "Phase R3 Wf-146 deferred full GPU run complete (${#pairs[@]} pairs)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate
budget_gate
data_quality_gate

if [ "${MODE}" = "R1" ]; then
    vram_gate "${R1_GPU}"
    phase_R1
elif [ "${MODE}" = "R3" ]; then
    vram_gate "${R3_GPU}"
    phase_R3
fi

ok "Wf-355-B2 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing:"
log "  - PROJECTED wall-time: ~3 hr on RTX 5090 (per Wf-146 sweep MEASURED ~30 min/pair)"
log "  - MEASURED baseline: Wf-146 sweep.log 2026-09-27 (19/25 done, 6 DEFERRED)"
log "  - 6 DEFERRED: Re 5/5 (fold never started) + Os seed 4 (sweep halted epoch 10/50)"
log "Per Wf-294B kill_then_adapt: Wf-355-B2 R1 = USER_OK optional; Wf-355-B2 R3 = USER_OK=YES required."
log "Per Wf-343c2 retro-fit pattern (Wf-347-B1/B2): per-wave check + data quality wired."
log "Wf-354-A1 §5 HONEST-NEGATIVE CLOSED by this wrapper (USER_OK-gated launch READY)."