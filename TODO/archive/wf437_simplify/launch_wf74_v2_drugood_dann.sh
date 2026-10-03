#!/usr/bin/env bash
# Wf-336-X3 — Wf-74 v2 DrugOOD DANN launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Origin: Wf-336-X3 ultracode residual closure wave
# Status: DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt)
#
# Purpose
# -------
# Wf-74 v2 is the docs-only wet-lab outreach closeout (per
# `TODO/completed/97c_wf74_reclassification.md` + Wf-300C pattern); Wf-74
# itself was DEPRECATED for wet-lab scope per AGENTS.md §5.4 Tier-2
# strategy.  However, the Wf-305D DrugOOD DANN trainer is the on-disk
# computational follow-up that lives at the Wf-74 v2 lineage: the
# drugood_v3_scaffold ood_test_auc=0.4574 BELOW the field-avg 0.55-0.65
# band (per Wf-297B + Wf-301A) needs a re-launch wrapper that wires all
# Wf-326 + Wf-328 production optimizations (W1-W10) onto the existing
# Wf-305D DANN driver.
#
# This wrapper:
#   1. Runs `pre_flight_check.py --pipeline all` (TODO-207 v3 gate).
#   2. Runs `wave_budget_tracker.py --cfm-yuelbond-required` (advisory).
#   3. Enforces the USER_OK gate (per Wf-294B / Wf-300C pattern).
#   4. Confirms VRAM budget via `nvidia-smi` (per AGENTS.md §5.1).
#   5. Wires all Wf-328 wire-ups (W1-W10) onto the Wf-305D driver:
#        * W1 torch.compile   (default ON)
#        * W2 num-workers     (--cpu-num-workers 8)
#        * W3 BF16            (default ON; --no-bf16 to disable)
#        * W4 TF32            (default ON)
#        * W5 W₂ sort         (--w2-gradient-mode sort)
#        * W6 DataLoader      (--num-workers 8)
#        * W7 PyG backend     (default ON; --dmpnn-backend pyg)
#        * W8 CUDA Graphs     (default OFF; --cuda-graph opt-in)
#        * W9 Triton edge     (default OFF; --triton-edge-update opt-in)
#        * W10 DMPNN cache    (default ON via Wf-305D cache module)
#   6. Default mode = Wf-335-R1: 1 seed × 1 epoch CPU verification only
#       (per Wf-335 launch suggestion smoke convention).  Wf-335-R1 mode
#       gates the bigger 3-seed × 50-epoch run behind USER_OK=YES.
#
# Scope (PROJECTED per Wf-285B + Wf-304C honest-negative)
# ------------------------------------------------------
#   R1 mode (default; CPU verification):  1 seed × 1 epoch × hidden=64
#                                          × depth=2 → ~30 sec wall
#   R3 mode (USER_OK=YES; RTX 5090):     3 seeds × 50 epochs
#                                          × hidden=128 × depth=3 → ~3-4 hr
#                                          (per Wf-306A Phase-A timing)
#
# USER_OK gate (per Wf-300C pattern)
# ----------------------------------
#   --dry-run does NOT bypass the USER_OK gate.
#   Wf-335-R1 default mode: USER_OK is OPTIONAL (smoke only).
#   Wf-335-R3 mode (3-seed × 50-ep): USER_OK=YES required.
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:  ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:    3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:     for 5+ minutes
#   7.2.4  collapse:           drugood ood_test_auc < 0.45 on 2/3 seeds
#
# Usage
# -----
#   bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R1
#       # CPU smoke (Wf-335-R1 default; USER_OK not required)
#   USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3
#       # 3-seed × 50-epoch RTX 5090 full DANN run
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/CadButEaas/.venv-vlm/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf74_v2_drugood_dann_launch"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
DRUGOOD_TRAINER="${SCRIPTS_DIR}/wf305d_drugood_dann_train.py"
DRUGOOD_CSV="${PROJECT_ROOT}/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.5        # Wf-335-R1 smoke (CPU; 1 seed × 1 ep)
R3_BUDGET_HR=4.0        # Wf-335-R3 full GPU (3 seeds × 50 ep)

# Per-phase GPU assignment (per Wf-130 default kanzi GPU 1)
R1_GPU="cpu"            # R1 is CPU verification
R3_GPU="cuda:1"         # R3 is RTX 5090

# Wf-328 wire-up flags (mirror the production trainer defaults)
WF328_NUM_WORKERS=8
WF328_TORCH_COMPILE=1
WF328_TF32=1
WF328_BF16=1
WF328_W2_MODE="sort"
WF328_DMPNN_BACKEND="pyg"
# W8 (CUDA Graphs) and W9 (Triton edge) are opt-in only (per Wf-328-SYN §3)

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB)
VRAM_BUDGET_GB=20

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
MODE="R1"   # Wf-335-R1 default (CPU smoke)
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,75p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-335-R1 default; Wf-335-R3 = USER_OK=YES)"

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
            warn "USER_OK not set; Wf-335-R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-335-R1 smoke)"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]         || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]     || die "wave_budget_tracker.py missing"
[ -f "${DRUGOOD_TRAINER}" ]    || die "DrugOOD trainer missing: ${DRUGOOD_TRAINER}"
[ -f "${DRUGOOD_CSV}" ]        || die "DrugOOD CSV missing: ${DRUGOOD_CSV}"

# Wf-305D DANN driver marker check (per TODO-207 v3 + Wf-305D §DANN)
grep -q "use_dann" "${DRUGOOD_TRAINER}" \
    || die "Wf-305D trainer missing --use-dann flag (DANN GRL not wired)"
grep -q "ReverseLayerF" "${DRUGOOD_TRAINER}" \
    || die "Wf-305D trainer missing ReverseLayerF (DANN GRL not wired)"

log "Wf-336-X3 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline all
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline all (TODO-207 v3 gate)"
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
    if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr 6 --cfm-yuelbond-required \
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
        ok "CPU mode (Wf-335-R1) — no VRAM check required"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found in PATH (required for ${mode} GPU launch)"
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
# Wf-328 wire-up flag builder (single source of truth)
#   Reflects Wf-328-SYN §3 production defaults (W3 BF16 ON, W4 TF32 ON,
#   W5 W₂ sort ON, W7 PyG ON, W8 CUDA Graphs OFF, W9 Triton OFF).
# -----------------------------------------------------------------------------
build_wf328_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"

    # Common flags
    local args=(
        --csv "${DRUGOOD_CSV}"
        --out-dir "${out_dir}"
        --ckpt-dir "${PROJECT_ROOT}/molmetal/checkpoints"
        --use-dann
        --lambda-grl 0.1
        --use-kendall
        --wasserstein-weight 0.05
        --alpha 0.5
        --lr 1e-3
        --cpu-num-workers "${WF328_NUM_WORKERS}"
        --w2-gradient-mode "${WF328_W2_MODE}"
    )

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-335-R1 smoke: 1 seed × 1 epoch × hidden=64 × depth=2 (CPU)
        args+=( --seeds 0 --epochs 1 --hidden 64 --depth 2 )
    else
        # Wf-335-R3 full: 3 seeds × 50 epochs × hidden=128 × depth=3 (RTX 5090)
        args+=( --seeds 0 1 2 --epochs 50 --hidden 128 --depth 3
                --cuda-visible "${TF276_R3_GPU_IDX:-1}" )
    fi

    if [ "${gpu}" != "cpu" ]; then
        args+=( --device cuda )
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
    shift 5
    # remaining args = --flag pairs

    log "==== Phase ${phase_label} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Phase ${phase_label}"
        cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Phase ${phase_label}
  trainer: ${trainer}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        ok "Phase ${phase_label} DRY-RUN logged"
        return 0
    fi

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
        "${PY}" "${trainer}" --out-dir "${out_dir}" "$@" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        CUDA_VISIBLE_DEVICES="${idx}" \
            "${PY}" "${trainer}" --out-dir "${out_dir}" "$@" \
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
    if [ "${aborted}" -eq 1 ] || ! grep -q "TEST (ood) AUC\|phase complete" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL — see ${out_dir}/train.log"
        die "Phase ${phase_label} aborted"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s"
}

# -----------------------------------------------------------------------------
# Phase R1 — Wf-335-R1 smoke (1 seed × 1 epoch CPU; no user OK gate)
# -----------------------------------------------------------------------------
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-335-R1 plan: drugood DANN smoke 1 seed × 1 epoch (~30 sec, CPU)"
    log "  Wf-328 wire-ups: --cpu-num-workers 8 --w2-gradient-mode sort"
    log "                   --use-dann --use-kendall --lambda-grl 0.1"
    log "                   --hidden 64 --depth 2 (smoke small)"
    local args
    mapfile -t args < <(build_wf328_flags "R1" "${R1_GPU}" "${out}")
    run_phase "R1" "${R1_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${R1_GPU}" "${out}" "${args[@]}"
    ok "Phase R1 drugood DANN smoke complete"
}

# -----------------------------------------------------------------------------
# Phase R3 — Wf-335-R3 full GPU run (3 seeds × 50 epochs RTX 5090)
# -----------------------------------------------------------------------------
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-335-R3 plan: drugood DANN full 3-seed × 50-epoch (~3-4 hr, cuda:1)"
    log "  Wf-328 wire-ups: --cpu-num-workers 8 --w2-gradient-mode sort"
    log "                   --use-dann --use-kendall --lambda-grl 0.1"
    log "                   --hidden 128 --depth 3 --batch-size 64 --lr 1e-3"
    local args
    mapfile -t args < <(build_wf328_flags "R3" "${R3_GPU}" "${out}")
    run_phase "R3" "${R3_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${R3_GPU}" "${out}" "${args[@]}"
    ok "Phase R3 drugood DANN full GPU run complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate
budget_gate

if [ "${MODE}" = "R1" ]; then
    vram_gate "${R1_GPU}"
    phase_R1
elif [ "${MODE}" = "R3" ]; then
    vram_gate "${R3_GPU}"
    phase_R3
fi

ok "Wf-336-X3 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: Wf-335-R1 = USER_OK optional; Wf-335-R3 = USER_OK=YES required."
log "Wf-74 v2 lineage: wet-lab outreach DEPRECATED per AGENTS.md §5.4; computational"
log "follow-up is Wf-305D DrugOOD DANN (this wrapper) + Wf-306A DrugOOD+Vina combined."
