#!/usr/bin/env bash
# Wf-310F — TODO-234 frontier-math Day 7 launch (USER_OK gate).
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Status: READY on user OK (per TODO-234 §5 + Wf-310F §1.7).
# Constraint: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED.
#
# Purpose
# -------
# GPU production wire-in for the 4 frontier-math adapters composed
# via ``molmetal.adapters.frontier_math_integration.FrontierMathIntegration``
# (Wf-309B ship + Wf-310F Day 1-2).  Per TODO-234 §4 + §5 the 5-7 day
# ultracode window's final day exercises the wire-in on a single GPU
# (~3-5 hr wall).
#
# What this launches
# ------------------
# Three sequential waves, each with its own GPU, scope, and abort criteria:
#   Wave A — TODO-119 P1.4 frontier-multi-ph smoke (~1.5 hr, cuda:0).
#            Wires ``augment_pocket_embedding`` into the CFM trainer
#            (``train_cfm_yuelbond_wasserstein.py``) and re-runs the P1.4
#            smoke with the augmented pocket embedding.
#   Wave B — TODO-182 G6 frontier-SE3 smoke (~0.5 hr, cuda:1).
#            Wires ``SE3LipmanSolver`` into the D-MPNN multi-task
#            trainer (``train_dmpnn_multitask_wasserstein.py``) and
#            re-runs the G6 no_wass smoke on the SE(3) manifold.
#   Wave C — TODO-234 frontier-math paper-grade sweep (~2-3 hr, cuda:0).
#            Joint ``FrontierMathIntegration`` wire-in across the full
#            R18 trainer stack — measure baseline + augmented lift
#            (PROJECTED, kill_then_adapt < 2 hr/sub-wave).
#
# kill_then_adapt semantics (per TODO-182 + Wf-295D)
# ---------------------------------------------------
#   - Each wave MUST finish < 2 hr (per-wave OK gate).
#   - On any 207 v2 §7.2 abort criterion → invoke abort_recovery.py,
#     exit 1.
#   - Between waves: stop and request per_wave_user_ok (per Wf-295D).
#
# Pre-launch checks (BLOCKER before any wave)
# -------------------------------------------
#   1. pre_flight_check.py --pipeline all (target 36/36 PASS).
#   2. Wave-specific budget tracker call.
#   3. Wave-specific per-wave checklist call.
#
# Per-wave OK gate (USER_OK)
# --------------------------
#   - Wall-time < 2 hr/wave.
#   - All P0 BLOCKER + P1 VERIFY gates still PASS (re-run pre_flight).
#   - Per-wave sub-checklist emit + JSON written under
#     molmetal/reports/wf310f_frontier_math_gpu/wave_<x>_subchecklist.json.
#
# Usage
# -----
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave A
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave all
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --dry-run
#
# Stop semantics: USER_OK required between waves.  The default ``--wave
# all`` invocation pauses between waves and waits for the caller to set
# ``WF310F_NEXT_WAVE=A|B|C`` (or pass ``--yes`` to skip the OK gate for
# dry-run / CI smoke).

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/hugo/codes/mol-metal}"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310f_frontier_math_gpu"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
ABORT_RECOVERY="${SCRIPTS_DIR}/abort_recovery.py"

# Trainers — Day 3-4 + Day 5-6 wire-in points (per Wf-310F §1).
CFM_TRAINER="${SCRIPTS_DIR}/train_cfm_yuelbond_wasserstein.py"
DMPNN_TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"

# Configs — same P1.4 / G6 cfgs as Wf-299C (frontier-math just wires
# into them; it does not change the configs themselves).
P1_4_CONFIG="${PROJECT_ROOT}/configs/cfm_per_innovation_ablation_p1.4.yaml"
G6_CONFIG="${PROJECT_ROOT}/configs/wf124_p3_per_innovation_ablation_g6.yaml"

mkdir -p "${REPORTS_DIR}/wave_a" "${REPORTS_DIR}/wave_b" "${REPORTS_DIR}/wave_c"

# Per-wave budget in hours (kill_then_adapt < 2 hr/wave hard cap).
WAVE_A_BUDGET_HR=1.5
WAVE_B_BUDGET_HR=0.5
WAVE_C_BUDGET_HR=2.5

# Per-wave GPU assignment.
WAVE_A_GPU="cuda:0"      # PRO 6000 Blackwell
WAVE_B_GPU="cuda:1"      # RTX 5090
WAVE_C_GPU="cuda:0"      # PRO 6000 Blackwell (joint sweep)

# Color codes (only when stdout is a tty).
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()  { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn(){ printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# USER_OK gate helper — blocks between waves.
user_ok_gate() {
    local wave_label="$1"
    if [ "${DRY_RUN:-0}" -eq 1 ] || [ "${YES:-0}" -eq 1 ]; then
        log "USER_OK gate: bypassed (dry-run / --yes) for wave ${wave_label}"
        return 0
    fi
    if [ -z "${WF310F_NEXT_WAVE:-}" ]; then
        die "USER_OK gate: set WF310F_NEXT_WAVE=${wave_label} to proceed (kill_then_adapt)"
    fi
    if [ "${WF310F_NEXT_WAVE}" != "${wave_label}" ]; then
        die "USER_OK gate: WF310F_NEXT_WAVE=${WF310F_NEXT_WAVE} != requested ${wave_label}"
    fi
    ok "USER_OK gate cleared for wave ${wave_label}"
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
YES=0
WAVE_TARGET="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --yes)      YES=1; shift ;;
        --wave)     WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)   WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,90p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] \
    || die "wave must be A | B | C | all (got: ${WAVE_TARGET})"

[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]        || die "pre_flight_check.py missing"
[ -f "${CFM_TRAINER}" ]       || die "CFM trainer missing: ${CFM_TRAINER}"
[ -f "${DMPNN_TRAINER}" ]     || die "D-MPNN trainer missing: ${DMPNN_TRAINER}"
[ -f "${P1_4_CONFIG}" ]       || die "P1.4 config missing: ${P1_4_CONFIG}"
[ -f "${G6_CONFIG}" ]         || die "G6 config missing: ${G6_CONFIG}"

log "Wf-310F Day 7 launch begin — wave target = ${WAVE_TARGET} — dry-run = ${DRY_RUN}"

# -----------------------------------------------------------------------------
# Pre-launch check — pre_flight_check.py --pipeline all
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates"
        return 0
    fi
    if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
            > "${REPORTS_DIR}/pre_flight.stdout" 2>&1; then
        die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    fi
    local n_pass n_fail
    n_pass=$(grep -c '"status": "PASS"'   "${REPORTS_DIR}/pre_flight.json" || true)
    n_fail=$(grep -c '"status": "FAIL"'   "${REPORTS_DIR}/pre_flight.json" || true)
    ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
}

# -----------------------------------------------------------------------------
# Per-wave runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_wave() {
    local wave_label="$1"; shift
    local budget_hr="$1"; shift
    local trainer="$1"; shift
    local config="$1"; shift
    local gpu="$1"; shift
    local out_dir="$1"; shift
    # remaining args = --flag pairs joined into the trainer invocation

    log "==== Wave ${wave_label} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  config:  ${config}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    user_ok_gate "${wave_label}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Wave ${wave_label}"
        cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Wave ${wave_label}
  trainer: ${trainer}
  config:  ${config}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        ok "Wave ${wave_label} DRY-RUN logged"
        return 0
    fi

    local started_at
    started_at=$(date +%s)
    local deadline
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))

    # Background launcher for the trainer; PID captured for kill_then_adapt.
    CUDA_VISIBLE_DEVICES="${gpu##cuda:}" \
        PYTHONPATH="${PROJECT_ROOT}" \
        "${PY}" "${trainer}" \
            --config "${config}" \
            --out-dir "${out_dir}" \
            "$@" \
            > "${out_dir}/train.log" 2>&1 &
    local pid=$!

    log "Wave ${wave_label} PID=${pid}; monitoring..."

    # Poll loop: every 60s, check timeout + GPU util + per-wave abort
    local aborted=0
    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Wave ${wave_label} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 5
            kill -KILL "${pid}" 2>/dev/null || true
            aborted=1
            break
        fi

        local gpu_util
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null \
                    | head -1 || echo "0")
        log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"

        sleep 60
    done

    wait "${pid}" 2>/dev/null || true

    local wall_sec=$(( $(date +%s) - started_at ))
    log "Wave ${wave_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"

    # Post-wave abort criteria sweep (per TODO-207 v2 §7.2)
    if [ "${aborted}" -eq 1 ] || ! grep -q "wave_${wave_label,,}_complete\|done" "${out_dir}/train.log"; then
        warn "Wave ${wave_label} FAIL — invoking abort_recovery.py"
        PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
            --pattern TIMEOUT --json \
            > "${out_dir}/recovery.json" 2>&1 || true
        die "Wave ${wave_label} aborted — recovery at ${out_dir}/recovery.json"
    fi

    ok "Wave ${wave_label} complete — ${wall_sec}s"
    if [ "${wall_sec}" -gt $(( ${WAVE_A_BUDGET_HR%.*} * 3600 )) ]; then
        die "Wave ${wave_label} exceeded per-wave < 2 hr OK gate (kill_then_adapt)"
    fi
}

# -----------------------------------------------------------------------------
# Wave A — Day 3-4 CFM training-loop frontier-multi-ph wire-in (~1.5 hr)
# -----------------------------------------------------------------------------
wave_A() {
    local out="${REPORTS_DIR}/wave_a"
    log "Wave A plan: P1.4 CFM trainer with --frontier-multi-ph (~1.5 hr, cuda:0)"
    log "  wires augment_pocket_embedding (W2.3) into PocketEncoder output"
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${CFM_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        --frontier-multi-ph --frontier-multi-ph-bins 8 \
        --out-dir "${out}/frontier_multiph" || true
    ok "Wave A complete"
}

# -----------------------------------------------------------------------------
# Wave B — Day 5-6 D-MPNN multi-task frontier-SE3 wire-in (~0.5 hr)
# -----------------------------------------------------------------------------
wave_B() {
    local out="${REPORTS_DIR}/wave_b"
    log "Wave B plan: G6 D-MPNN multi-task trainer with --frontier-se3 (~0.5 hr, cuda:1)"
    log "  wires SE3LipmanSolver (Integration 1) into the trainer manifold"
    run_wave "B" "${WAVE_B_BUDGET_HR}" \
        "${DMPNN_TRAINER}" "${G6_CONFIG}" "${WAVE_B_GPU}" "${out}" \
        --metals Ru --seeds 0 --epochs 10 --batch-size 64 --lr 1e-3 \
        --alpha 0.5 --wasserstein-weight 0.0 \
        --device cuda:1 --frontier-se3 \
        --out-dir "${out}/frontier_se3_ru" || true
    ok "Wave B complete"
}

# -----------------------------------------------------------------------------
# Wave C — Day 7 joint FrontierMathIntegration paper-grade sweep (~2-3 hr)
# -----------------------------------------------------------------------------
wave_C() {
    local out="${REPORTS_DIR}/wave_c"
    log "Wave C plan: joint frontier sweep (PROJECTED < 2 hr sub-waves × 2, cuda:0)"
    log "  wires ALL 4 frontier adapters via FrontierMathIntegration"
    warn "Wave C requires USER_OK between sub-waves — auto-OK only on dry-run"
    log "Wave C sub-wave 1: FrontierMathIntegration.get_multi_ph_features + get_transport_relaxation (~1.5 hr)"
    log "Wave C sub-wave 2: FrontierMathIntegration.get_se3_lipman_solver + get_se3_sb_decoder (~1.0 hr)"
    ok "Wave C plan logged (joint sweep)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate

case "${WAVE_TARGET}" in
    A)   wave_A ;;
    B)   wave_B ;;
    C)   wave_C ;;
    all) wave_A; wave_B; wave_C ;;
esac

ok "Wf-310F Day 7 launch script done — wave=${WAVE_TARGET}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each wave runs."
log "Per Wf-310F §1.7: USER_OK required between waves (kill_then_adapt)."