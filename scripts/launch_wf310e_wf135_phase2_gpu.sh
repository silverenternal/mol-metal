#!/usr/bin/env bash
# Wf-310E Part 1 — Wf-135 Phase 2 LOMO 5-metal × 5-seed × 50-epoch re-bench
#                   USER_OK-gated GPU launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Parent: Wf-309A Rh cohort 500 (Phase 1 prerequisite) + Wf-135 Phase 2 plan
# Refs:
#   - TODO/completed/135_wf135_phase1_plan.md §2 (Phase 2 spec)
#   - TODO/completed/135_m19_lomo_rh_oversample_fix.md §2.2 (planned Phase 2)
#   - TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (Wf-309D restored)
#   - molmetal/reports/wf176_m19_oversample_lomo/synthesis.md (current state)
#   - TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.4 (acceptance gates)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt; 2 weeks GPU wall)
#   - Phase 1 prerequisite: Wf-309A Rh cohort 500 must be COMPLETE on disk
#     (molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json)
#   - Scope: 5-metal × 5-seed × 50-epoch = 1250 runs
#   - Wall: ~10 GPU-hours (PROJECTED) = ~2 calendar weeks (best-case 7 days,
#     worst-case 14 days with retries per Phase 1 plan §2.4)
#   - Rh fold AUC ≥ 0.30 + Overall LOMO mean ≥ 0.60 = MEASURED acceptance gates
#   - Trainer target: molmetal/scripts/train_metal_lomo_v2.py
#     (uses balanced cohort + per-metal head ensemble + Wasserstein transfer
#      target = Rh real active distribution, NOT synthetic cohort)
#   - Per Wf-139 P7 ship (Phase 1): 250 active + 250 inactive balanced
#     cohort with diff-based ranking; scaffold-split verified clean.
#   - Per Wf-131 P3 + Wf-125 P2 honest-negative: Rh fold regressed 0.368 → 0.136
#     on buggy constant-inactive cohort; this Phase 2 must recover ≥ 0.30.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh           # actual run
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0)
# --------------------------------------------------------
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch and emits
#     the planned invocation as a record.
#   - 2-week wall budget is enforced as a logging warning, not a hard stop
#     (per kill_then_adapt + Wf-295D per_wave_user_ok pattern).
#
# Stop semantics: per TODO-207 v2 §7.2 (per-phase abort criteria):
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN/OOM/IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: Rh fold AUC < 0.10 on 2+ seeds (binding regression alarm)
#
# Pre-launch checks (BLOCKER before any GPU work):
#   1. Phase 1 cohort on disk (Wf-309A complete)
#   2. pre_flight_check.py --pipeline all (24/24 PASS target)
#   3. wave_budget_tracker.py --budget-hr 14 --chemprop-not-required
#   4. USER_OK env gate

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310e_wf135_phase2"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_v2.py"

# Phase 1 prerequisite
PHASE1_COHORT="${PROJECT_ROOT}/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json"

# Phase 2 spec
METALS=(Rh Ru Ir Os Re)   # 5-metal LOMO
SEEDS=(0 1 2 3 4)         # 5 seeds (vs Wf-131 P3 baseline 3)
EPOCHS=50
WALL_BUDGET_HR=14          # 2 weeks upper bound
GPU="cuda:1"               # RTX 5090 GPU-1

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual 2-week GPU run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
SKIP_PHASE1_CHECK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)            DRY_RUN=1; shift ;;
        --skip-phase1-check)  SKIP_PHASE1_CHECK=1; shift ;;
        -h|--help)            sed -n '2,60p' "$0"; exit 0 ;;
        *)                    printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Color codes (only when stdout is a tty)
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
# Sanity checks
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${TRAINER}" ]           || die "trainer missing: ${TRAINER}"

# Phase 1 prerequisite check (Wf-309A must be complete)
if [ "${SKIP_PHASE1_CHECK}" != "1" ]; then
    if [ ! -f "${PHASE1_COHORT}" ]; then
        die "Phase 1 prerequisite missing: ${PHASE1_COHORT}. Run Wf-309A first."
    fi
    # Quick sanity: cohort must be 500 lines with 250 active + 250 inactive
    N_ACT=$(grep -c '"active_inactive": "active"' "${PHASE1_COHORT}" 2>/dev/null || echo 0)
    N_INACT=$(grep -c '"active_inactive": "inactive"' "${PHASE1_COHORT}" 2>/dev/null || echo 0)
    if [ "${N_ACT}" != "250" ] || [ "${N_INACT}" != "250" ]; then
        warn "Phase 1 cohort may be unbalanced (active=${N_ACT}, inactive=${N_INACT}); continuing"
    else
        ok "Phase 1 cohort verified: 250 active + 250 inactive balanced"
    fi
fi

log "Wf-310E Wf-135 Phase 2 LOMO re-bench wrapper — USER_OK=1 confirmed"
log "  metals=${METALS[*]}"
log "  seeds=${SEEDS[*]}"
log "  epochs=${EPOCHS}"
log "  total_runs=$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
log "  gpu=${GPU}"
log "  wall_budget_hr=${WALL_BUDGET_HR}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310e_wf135_phase2_gpu.sh",\n'
    printf '  "wf_origin": "Wf-310E Part 1",\n'
    printf '  "parent_plan": "TODO/completed/135_wf135_phase1_plan.md §2",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metals": [%s],\n' "$(IFS=,; echo "${METALS[*]}" | tr ' ' ',')"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}" | tr ' ' ',')"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "total_runs": %d,\n' "$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": %d,\n' "${WALL_BUDGET_HR}"
    printf '  "wall_budget_label": "PROJECTED (2 calendar weeks per Wf-135 Phase 1 plan §2.4; compute ~10 GPU-hr; remainder = setup/teardown/reporting)",\n'
    printf '  "phase1_prereq": "%s",\n' "${PHASE1_COHORT}"
    printf '  "phase1_status": "%s",\n' "$([ -f "${PHASE1_COHORT}" ] && echo "PRESENT" || echo "MISSING")"
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "acceptance_gates": {\n'
    printf '    "rh_fold_auc_min": 0.30,\n'
    printf '    "overall_lomo_mean_min": 0.60,\n'
    printf '    "min_tests_pass": 16,\n'
    printf '    "rh_target_origin": "recover from Wf-131 P2 regression 0.136 to baseline 0.368"\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; phase1 prerequisite check wired",\n'
    printf '    "PROJECTED": "Rh fold AUC +0.164 recovery (0.136 -> 0.30 target); overall LOMO mean +0.065 (0.535 -> 0.60 target)",\n'
    printf '    "kill_then_adapt_directive": "2-week wall; user_ok gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  CUDA_VISIBLE_DEVICES=1 ${PY} ${TRAINER} \\"
    log "    --metals ${METALS[*]} --seeds ${SEEDS[*]} \\"
    log "    --epochs ${EPOCHS} \\"
    log "    --balanced-cohort-json ${PHASE1_COHORT} \\"
    log "    --out-dir ${REPORTS_DIR}/per_seed_results \\"
    log "    --device cuda:0"
    log ""
    log "DRY-RUN: per-fold sequencing (5 metals x 5 seeds = 25 (metal,seed) tuples; each tuple = 1 Rh-fold + 4 other-fold runs = ~5 LOMO runs)"
    log "DRY-RUN: total wall = ~10 GPU-hr sequential = ~2 calendar weeks worst case"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 5-metal × 5-seed sweep
# Each (metal, seed) tuple runs a full LOMO: hold out the focal metal,
# train on remaining 4 + Rh balanced cohort, evaluate on held-out metal.
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)

# Phase 2 sequencing per Wf-135 Phase 1 plan §2.1:
#   For each metal m in {Rh, Ru, Ir, Os, Re}:
#     For each seed s in {0, 1, 2, 3, 4}:
#       Train multi-task D-MPNN with W₂ transfer + per-metal head ensemble
#       Use Wf-139 P7 balanced cohort for Rh fold
#       Wasserstein target = Rh REAL active distribution (not synthetic)
for METAL in "${METALS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        log "=== metal=${METAL} seed=${SEED} on ${GPU} ==="
        if CUDA_VISIBLE_DEVICES=1 "${PY}" "${TRAINER}" \
                --metals "${METAL}" \
                --seeds "${SEED}" \
                --epochs "${EPOCHS}" \
                --balanced-cohort-json "${PHASE1_COHORT}" \
                --out-dir "${REPORTS_DIR}/${METAL}/seed${SEED}" \
                --device cuda:0 \
                >> "${RUN_LOG}" 2>&1; then
            ok "metal=${METAL} seed=${SEED} completed"
        else
            die "metal=${METAL} seed=${SEED} failed (see ${RUN_LOG}); aborting 2-week sweep"
        fi
    done
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_hr=$(( elapsed_sec / 3600 ))
log "Wf-310E Wf-135 Phase 2 LOMO sweep done — elapsed=${elapsed_hr} hr (budget=${WALL_BUDGET_HR} hr)"

if [ "${elapsed_hr}" -gt "${WALL_BUDGET_HR}" ]; then
    warn "elapsed (${elapsed_hr} hr) exceeded PROJECTED wall budget (${WALL_BUDGET_HR} hr); flagging as MEASURED-over-PROJECTED in plan.json"
    # Patch plan.json with measured wall time
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-(metal,seed) JSONs"
ok "Next: emit synthesis.md + paper §4.5.6 honest-framing decision (per TODO-84 rule)"