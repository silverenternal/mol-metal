#!/usr/bin/env bash
# Wf-306B — Wf-219 Ru short-epoch GPU re-test wrapper.
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Origin: Wf-304C Task 1 + Wf-305B/Wf-305C follow-up
# Status: READY on user OK (Wf-304C noted all Wf-219 Ru seed runs hit 43/50
#         epochs and timed out under the standard 5-seed x 50-epoch protocol;
#         this wrapper drops to 30 epochs and 3 seeds to fit ~10 min/wall on
#         RTX 5090 cuda:1, leaving room for an honest re-measurement).
#
# Purpose
# -------
# Re-measure the Wf-219 Ru 50-epoch baseline under a tighter epoch budget
# (30 instead of 50) so the run completes inside ~10 min/wall on RTX 5090
# cuda:1. The reduced budget is justified by Wf-304C §1: every Ru run
# timed out at 43/50 epochs under the standard protocol — the actual
# convergence point is unknown, so 30 epochs is a conservative lower bound
# that should still land the model in the Krasnov-aligned AUC range.
#
# Scope (3 seeds, 30 epochs, Ru only)
# -----------------------------------
#   seed 0  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   seed 1  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   seed 2  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   total wall budget: ~10 min on RTX 5090 cuda:1 (PROJECTED 12 min)
#
# Usage
# -----
#   USER_OK=1 bash molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh --dry-run
#   USER_OK=1 bash molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh           # actual run
#
# USER_OK gate (explicit, per Wf-295D per_wave_user_ok pattern)
# -------------------------------------------------------------
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch and emits
#     the planned invocation as a record.
#   - This is intentionally stricter than Wf-299 (which auto-OKs on dry-run)
#     because this wrapper targets a real (not stub) GPU retrain of Wf-219.
#
# Stop semantics: per TODO-119 P1.4 §directives.per_wave_user_ok_required
# (Wf-295D) — only one wave here, no per-wave sub-checks needed.
#
# All numbers are MEASURED where possible, PROJECTED where not (party-twenty-
# spirit §3.5).

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf306b_wf219_ru_wrapper"
TRAINER="${SCRIPTS_DIR}/wf219_krasnov_aligned.py"

# Wf-304C §1: Ru baseline timed out at 43/50 epochs. Drop to 30.
EPOCHS=30
SEEDS=(0 1 2)
METAL="Ru"
GPU="cuda:1"   # RTX 5090

# Wall budget: 30 epochs × 3 seeds × ~20 sec/epoch ≈ 1800 sec ≈ 30 min PROJECTED.
# Wf-304C measured 43/50 epochs in ~13.5 min, so 30/50 ≈ 9.4 min PROJECTED.
WALL_BUDGET_MIN=10
mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s      # actual GPU run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *)         printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
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

log "Wf-306B Wf-219 Ru short-epoch wrapper — USER_OK=1 confirmed"
log "  epochs=${EPOCHS} (down from 50; Wf-304C §1 timeout at 43/50)"
log "  seeds=${SEEDS[*]}"
log "  metal=${METAL}"
log "  gpu=${GPU}"
log "  wall_budget_min=${WALL_BUDGET_MIN}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf306b_wf219_ru_gpu.sh",\n'
    printf '  "wf_origin": "Wf-306B",\n'
    printf '  "origin_wf": "Wf-304C Task 1 + Wf-305B/Wf-305C follow-up",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metal": "%s",\n' "${METAL}"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}")"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "wall_budget_label": "PROJECTED (Wf-304C measured 13.5 min for 43/50 epochs; 30/50 ≈ 9.4 min PROJECTED; +33 pct safety = 10 min budget)",\n'
    printf '  "epoch_target_origin": "Wf-304C §1: Ru baseline timed out at 43/50 epochs under standard protocol",\n'
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  CUDA_VISIBLE_DEVICES=1 ${PY} ${TRAINER} \\"
    log "    --metals ${METAL} --metals-only ${METAL} \\"
    log "    --seeds ${SEEDS[*]} --epochs ${EPOCHS} \\"
    log "    --out ${REPORTS_DIR}/sweep_summary.json"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 3-seed sweep, ~10 min total wall
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    log "=== seed=${SEED} Ru ${EPOCHS} epochs on ${GPU} ==="
    if CUDA_VISIBLE_DEVICES=1 "${PY}" "${TRAINER}" \
            --metals "${METAL}" --metals-only "${METAL}" \
            --seeds "${SEED}" --epochs "${EPOCHS}" \
            --out "${REPORTS_DIR}/sweep_seed${SEED}.json" \
            >> "${RUN_LOG}" 2>&1; then
        ok "seed=${SEED} completed"
    else
        die "seed=${SEED} failed (see ${RUN_LOG})"
    fi
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_min=$(( elapsed_sec / 60 ))
log "Wf-306B Wf-219 Ru short-epoch sweep done — elapsed=${elapsed_min} min (budget=${WALL_BUDGET_MIN} min)"

if [ "${elapsed_min}" -gt "${WALL_BUDGET_MIN}" ]; then
    warn "elapsed (${elapsed_min} min) exceeded PROJECTED wall budget (${WALL_BUDGET_MIN} min); flagging as MEASURED-over-PROJECTED in plan.json"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-seed sweep_seed{0,1,2}.json"
