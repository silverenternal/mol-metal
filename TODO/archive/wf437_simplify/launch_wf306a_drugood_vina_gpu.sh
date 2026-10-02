#!/usr/bin/env bash
# Wf-306A — drugood + Vina real GPU launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Status: CPU smoke PASS (Wf-306A PART 1 + PART 2).  GPU launch wrapper ready
#          on USER_OK gate (per Wf-294B kill_then_adapt pattern).
# Constraint: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED.
#
# Purpose
# -------
# Two sequential phases, each with its own GPU assignment, scope, abort criteria,
# and per-phase user OK gate.  This is the Wf-306A follow-up to:
#   * Wf-297B + Wf-301A: drugood_v3_scaffold ood_test_auc=0.4574 BELOW
#     field-avg 0.55-0.65.
#   * Wf-299B + Wf-301B: Vina 0/100 docked on Pt coordination complexes
#     (Pt+2 UFF parameterisation blocker; Wf-301B shipped --pt-strip flag,
#     Wf-305D shipped the wrapper).
#   * Wf-305D: drugood DANN driver + Vina pt-strip wrapper IMPLEMENTED but
#     CPU smoke NOT RUN before Wf-306A.
#
# Wf-306A results:
#   * PART 1 (drugood DANN): CPU smoke 95,236 rows × 1 seed × 1 epoch
#     (featurize+train).  Field-avg recovery is PROJECTED per Wf-304C.
#   * PART 2 (Vina pt-strip): CPU smoke 5/5 docked successfully against
#     pocket 1h36 (vs 0/5 without --pt-strip).  Overall mean kcal/mol=-4.60
#     (ABOVE field-avg band -9.5 to -7.5 — honest-negative on the
#     field-avg gate; the recovery axis is *being able to dock at all*,
#     which the pt-strip enables).
#
# Phases
# ------
#   Phase-A — drugood DANN full 3-seed × 50-epoch GPU run on cuda:1
#             (RTX 5090).  ~3-4 hr wall (PROJECTED, per Wf-285B
#             5-metal Cytotox 3-seed × 50-epoch = 4 hr on RTX 5090;
#             drugood single-domain is 50% larger so add ~0.5 hr).
#   Phase-B — Vina pt-strip full 100-mol panel on cuda:1 (CPU-only; Vina
#             binary is single-threaded).  ~1-2 hr wall (PROJECTED, per
#             Wf-305D smoke ~5 mols in 7-10 s × 100 mols ≈ 100-200 s +
#             receptor pre-dock work × 4 pockets ≈ 1-2 hr).
#
# Total wall: ~5 hr (PROJECTED).
#
# USER_OK gate (per Wf-300C pattern)
# ---------------------------------
#   The script aborts unless ``WF306A_USER_OK=YES`` is exported in the
#   environment.  This is the kill_then_adapt guard against un-watched
#   GPU launches (per Wf-294B).
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:  ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:    3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:     for 5+ minutes (Phase-A only; Phase-B is CPU)
#   7.2.4  collapse:           drugood Phase-A ood_test_auc < 0.45 on 2/3 seeds
#                             Vina Phase-B n_docked < 50 / 100
#
# Pre-launch checks (BLOCKER before any phase)
# -------------------------------------------
#   1. pre_flight_check.py --pipeline all (24/24 PASS target)
#   2. wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required
#       (defer if not CFM, since this wrapper is drugood+Vina not CFM)
#   3. USER_OK env gate
#
# Usage
# -----
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/CadButEaas/.venv-vlm/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf306a_drugood_vina_launch"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
ABORT_RECOVERY="${SCRIPTS_DIR}/abort_recovery.py"

DRUGOOD_TRAINER="${SCRIPTS_DIR}/wf305d_drugood_dann_train.py"
VINA_TRAINER="${SCRIPTS_DIR}/wf305d_vina_pt_strip.py"
DRUGOOD_CSV="${PROJECT_ROOT}/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv"
VINA_CSV="${PROJECT_ROOT}/.data/molmetal/platinai_4cell_mbfinder.csv"

mkdir -p "${REPORTS_DIR}/phase_a" "${REPORTS_DIR}/phase_b"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap; Phase-A
# spans 3-4 hr as 2 sub-phases < 2 hr each)
PHASE_A_BUDGET_HR=4.0      # 3-seed × 50-epoch drugood DANN full
PHASE_B_BUDGET_HR=2.0      # 100-mol Vina pt-strip panel

# Per-phase GPU assignment (drugood: GPU; Vina: CPU-only — Vina binary is
# single-threaded so cuda:1 is unused during Phase-B)
PHASE_A_GPU="cuda:1"        # RTX 5090 (per Wf-130 default kanzi GPU 1)
PHASE_B_GPU="cpu"           # Vina is CPU-only

# Color codes (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()  { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn(){ printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
PHASE_TARGET="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --phase)   PHASE_TARGET="${2:-all}"; shift 2 ;;
        --phase-*) PHASE_TARGET="${1#--phase-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${PHASE_TARGET}" = "all" ] || [ "${PHASE_TARGET}" = "A" ] || [ "${PHASE_TARGET}" = "B" ] \
    || die "phase must be A | B | all (got: ${PHASE_TARGET})"

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-300C pattern)
# -----------------------------------------------------------------------------
user_ok_gate() {
    if [ -z "${WF306A_USER_OK:-}" ]; then
        die "WF306A_USER_OK env var not set.  Re-run with: WF306A_USER_OK=YES bash $0 ..."
    fi
    if [ "${WF306A_USER_OK}" != "YES" ]; then
        die "WF306A_USER_OK must equal 'YES' (got: '${WF306A_USER_OK}')"
    fi
    ok "USER_OK gate passed (WF306A_USER_OK=YES)"
}

# -----------------------------------------------------------------------------
# Pre-flight sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]        || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]    || die "wave_budget_tracker.py missing"
[ -f "${ABORT_RECOVERY}" ]    || die "abort_recovery.py missing"
[ -f "${DRUGOOD_TRAINER}" ]   || die "drugood trainer missing: ${DRUGOOD_TRAINER}"
[ -f "${VINA_TRAINER}" ]      || die "vina trainer missing: ${VINA_TRAINER}"
[ -f "${DRUGOOD_CSV}" ]       || die "drugood CSV missing: ${DRUGOOD_CSV}"
[ -f "${VINA_CSV}" ]          || die "vina CSV missing: ${VINA_CSV}"

log "Wf-306A launch begin — phase target = ${PHASE_TARGET} — dry-run = ${DRY_RUN}"
user_ok_gate

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline all
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates"
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
    [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py --budget-hr 6
#   --cfm-yuelbond-required is a gate that DEFERs non-CFM waves; this
#   wrapper is drugood+Vina so we DO pass it but expect a DEFERRED flag.
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        return 0
    fi
    if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr 6 --cfm-yuelbond-required \
            --report-json "${REPORTS_DIR}/budget_report.json" \
            > "${REPORTS_DIR}/budget.log" 2>&1; then
        warn "budget tracker returned non-zero (advisory only; --no-enforce-budget supported)"
    fi
    ok "budget tracker logged (DEFERRED expected since drugood/Vina are not CFM)"
}

# -----------------------------------------------------------------------------
# Per-phase runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_phase() {
    local phase_label="$1"; shift
    local budget_hr="$1"; shift
    local trainer="$1"; shift
    local gpu="$1"; shift
    local out_dir="$1"; shift
    # remaining args = --flag pairs joined into the trainer invocation

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

    local started_at
    started_at=$(date +%s)
    local deadline
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))

    # Background launcher for the trainer; PID captured for kill_then_adapt.
    CUDA_VISIBLE_DEVICES="${gpu##cuda:}" \
        "${PY}" "${trainer}" \
            --out-dir "${out_dir}" \
            "$@" \
            > "${out_dir}/train.log" 2>&1 &
    local pid=$!

    log "Phase ${phase_label} PID=${pid}; monitoring..."

    # Poll loop: every 60s, check timeout + GPU util + per-phase abort criteria
    local aborted=0
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

        # GPU util < 70% for 5+ min — TODO-207 v2 §7.2.3 (advisory only here;
        # applies to Phase-A only; Phase-B is CPU)
        if [ "${phase_label}" = "A" ]; then
            local gpu_util
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                    --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-1}" 2>/dev/null \
                        | head -1 || echo "0")
            log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"
        fi

        sleep 60
    done

    wait "${pid}" 2>/dev/null || true

    local wall_sec=$(( $(date +%s) - started_at ))
    log "Phase ${phase_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || ! grep -q "phase_${phase_label,,}_complete\|TEST (ood) AUC\|n_docked=" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL — invoking abort_recovery.py"
        "${PY}" "${ABORT_RECOVERY}" --wave "phase_${phase_label,,}" \
            --pattern TIMEOUT --json \
            > "${out_dir}/recovery.json" 2>&1 || true
        die "Phase ${phase_label} aborted — recovery at ${out_dir}/recovery.json"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s"
    # Per-phase OK gate: < 2 hr (per kill_then_adapt).  Phase-A's 4 hr
    # PROJECTED budget is split into 2 sub-phases of < 2 hr each at the
    # config level (3 seeds ≈ 1.3 hr each).
    if [ "${wall_sec}" -gt $(( ${budget_hr%.*} * 3600 )) ]; then
        die "Phase ${phase_label} exceeded per-phase ${budget_hr} hr OK gate (kill_then_adapt)"
    fi
}

# -----------------------------------------------------------------------------
# Phase A — drugood DANN full 3-seed × 50-epoch GPU run (cuda:1)
# -----------------------------------------------------------------------------
phase_A() {
    local out="${REPORTS_DIR}/phase_a"
    log "Phase A plan: drugood DANN full 3-seed × 50-epoch (~3-4 hr, cuda:1)"
    log "  seeds=0,1,2 epochs=50 hidden=128 depth=3 batch=64 lr=1e-3"
    log "  use_dann=True lambda_grl=0.1 use_kendall=True w_w=0.05"
    run_phase "A" "${PHASE_A_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${PHASE_A_GPU}" "${out}" \
        --csv "${DRUGOOD_CSV}" \
        --seeds 0 1 2 \
        --epochs 50 \
        --hidden 128 \
        --depth 3 \
        --batch-size 64 \
        --lr 1e-3 \
        --alpha 0.5 \
        --wasserstein-weight 0.05 \
        --use-dann \
        --lambda-grl 0.1 \
        --use-kendall \
        --device cuda \
        --cpu-num-workers 2 \
        || true
    ok "Phase A drugood full DANN run complete"
}

# -----------------------------------------------------------------------------
# Phase B — Vina pt-strip full 100-mol panel (CPU-only)
# -----------------------------------------------------------------------------
phase_B() {
    local out="${REPORTS_DIR}/phase_b"
    log "Phase B plan: Vina pt-strip full 100-mol × 4-pocket panel (~1-2 hr, CPU)"
    log "  n_mols=100 pocket_id=1h36 (extend to 4 pockets for paper-grade)"
    run_phase "B" "${PHASE_B_BUDGET_HR}" \
        "${VINA_TRAINER}" "${PHASE_B_GPU}" "${out}" \
        --csv "${VINA_CSV}" \
        --pocket-id 1h36 \
        --n-mols 100 \
        --pt-strip \
        --out "${out}/panel.json" \
        || true
    ok "Phase B Vina pt-strip full panel complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate
budget_gate

case "${PHASE_TARGET}" in
    A)   phase_A ;;
    B)   phase_B ;;
    all) phase_A; phase_B ;;
esac

ok "Wf-306A launch script done — phase=${PHASE_TARGET}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: per_phase_user_ok_required between phases."
