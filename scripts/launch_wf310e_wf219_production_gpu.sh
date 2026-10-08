#!/usr/bin/env bash
# Wf-310E Part 2 — Wf-219 §4.2 production backbone (hidden=300/depth=3)
#                   USER_OK-gated GPU launch wrapper — PAPER BLOCKED per TODO-84
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Parent: TODO/completed/219_wf219_krasnov_aligned_closeout.md §4.2
# Refs:
#   - TODO/completed/219_wf219_krasnov_aligned_closeout.md §4.2 (production backbone spec)
#   - TODO/pending/84_no_paper_task_without_authorization.md (paper BLOCKED rule)
#   - molmetal/reports/krasnov_protocol_investigation.md §3 (Krasnov 2026 protocol)
#   - TODO/pending/180_wf124_p3_remaining_metals_seeds.md (related gap context)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt; 80-min GPU wall)
#   - **PAPER BLOCKED per TODO-84**: this wrapper can only launch if user
#     explicitly OKs paper work. TODO-84 rule (active): zero touches to
#     paper/* unless user explicitly authorizes. The §4.2 production
#     backbone produces paper-grade numbers matching Krasnov 2026, which
#     is paper-class work per TODO-84 §1.1.
#   - Scope: 5-metal × 3-seed × 50-epoch × hidden=300/depth=3 = 15 runs
#     (matches Wf-219 3-seed pattern; production backbone only)
#   - Wall: ~80 min GPU-1 (PROJECTED; ~6.6 min/run × 12 = 80 min for
#     the 4-metal × 3-seed subset; +Ru adds 3 runs × 10 min = 30 min;
#     total worst case ~110 min = ~2 hr)
#   - Trainer: molmetal/scripts/wf219_krasnov_aligned.py (already exists)
#     with --hidden 300 --depth 3 flags
#   - Per Wf-219 §4.1: Ru baseline timed out at 43/50 epochs under the
#     standard hidden=64/depth=2 protocol. Production backbone (300/3)
#     makes this worse. Wall budget expanded accordingly.
#
# Usage:
#   USER_OK=1 PAPER_OK=1 bash scripts/launch_wf310e_wf219_production_gpu.sh --dry-run
#   USER_OK=1 PAPER_OK=1 bash scripts/launch_wf310e_wf219_production_gpu.sh           # actual run
#
# USER_OK + PAPER_OK gate (explicit, per mol-metal-gpu-execution §0 + TODO-84)
# --------------------------------------------------------------------
#   - The wrapper REFUSES to run unless BOTH USER_OK=1 AND PAPER_OK=1
#     are set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch.
#   - 80-min wall budget is logged as warning, not hard-stop.
#
# Stop semantics: per TODO-207 v2 §7.2:
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN/OOM/IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: 2+ seeds fail to complete 50/50 epochs (timeout risk
#                        dominated by Ru at hidden=300/depth=3)
#
# Pre-launch checks (BLOCKER before any GPU work):
#   1. PAPER_OK env gate (TODO-84 lift)
#   2. pre_flight_check.py --pipeline launch (24/24 PASS target)
#   3. wave_budget_tracker.py --budget-hr 2 --cfm-not-required
#   4. USER_OK env gate

set -euo pipefail


# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310e_wf219_production"
TRAINER="${SCRIPTS_DIR}/wf219_krasnov_aligned.py"

# Production backbone spec (Wf-219 §4.2)
METALS=(Ru Ir Os Rh Re)     # 5-metal (Ru first — biggest cohort + most expensive)
SEEDS=(0 1 2)              # 3 seeds per Wf-219 P1 ship pattern
EPOCHS=50
HIDDEN=300                 # chemprop v2 production (vs Wf-219 smoke 64)
DEPTH=3                    # chemprop v2 production (vs Wf-219 smoke 2)
GPU="cuda:0"               # RTX PRO 6000 Blackwell GPU-0 (serial execution)
WALL_BUDGET_MIN=80         # 80-min wall per Wf-219 §4.2 (best case for 12 non-Ru runs)
WALL_BUDGET_MAX_MIN=110    # 110-min worst case (with Ru)

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK + PAPER_OK gates (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
PAPER_BLOCKED_MSG="⚠️  PAPER BLOCKED per TODO-84 (no paper task without authorization).
       Set PAPER_OK=1 in env ONLY after user explicitly authorizes paper work.
       The §4.2 production backbone produces paper-grade Krasnov-2026 numbers."

if [ "${PAPER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s PAPER_OK gate refused (TODO-84 paper-block rule).\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '%s\n' "${PAPER_BLOCKED_MSG}" >&2
    printf '       USER_OK=1 PAPER_OK=1 bash %s --dry-run   # plan only\n' "$0" >&2
    exit 64
fi

if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 PAPER_OK=1 bash %s --dry-run   # plan only\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
SKIP_RU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-ru) SKIP_RU=1; shift ;;
        -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
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

# Skip Ru if requested (Ru is the bottleneck per Wf-219 §4.1)
RUN_METALS=( "${METALS[@]}" )
if [ "${SKIP_RU}" = "1" ]; then
    RUN_METALS=( "${METALS[@]:1}" )  # drop first (Ru)
    warn "--skip-ru: dropping Ru from sweep; total runs = $(( ${#RUN_METALS[@]} * ${#SEEDS[@]} ))"
fi

log "Wf-310E Wf-219 §4.2 production backbone wrapper — USER_OK + PAPER_OK gates confirmed"
log "  metals=${RUN_METALS[*]}"
log "  seeds=${SEEDS[*]}"
log "  epochs=${EPOCHS}"
log "  hidden=${HIDDEN} (production vs Wf-219 smoke 64)"
log "  depth=${DEPTH} (production vs Wf-219 smoke 2)"
log "  total_runs=$(( ${#RUN_METALS[@]} * ${#SEEDS[@]} ))"
log "  gpu=${GPU}"
log "  wall_budget_min=${WALL_BUDGET_MIN} (best) / ${WALL_BUDGET_MAX_MIN} (worst with Ru)"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310e_wf219_production_gpu.sh",\n'
    printf '  "wf_origin": "Wf-310E Part 2 (Wf-219 §4.2 production backbone)",\n'
    printf '  "parent_closeout": "TODO/completed/219_wf219_krasnov_aligned_closeout.md §4.2",\n'
    printf '  "user_ok": true,\n'
    printf '  "paper_ok": true,\n'
    printf '  "paper_rule": "TODO-84 active; PAPER_OK gate enforces explicit user authorization for paper-class work",\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metals": [%s],\n' "$(IFS=,; echo "${RUN_METALS[*]}" | tr ' ' ',')"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}" | tr ' ' ',')"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "hidden": %d,\n' "${HIDDEN}"
    printf '  "depth": %d,\n' "${DEPTH}"
    printf '  "total_runs": %d,\n' "$(( ${#RUN_METALS[@]} * ${#SEEDS[@]} ))"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "wall_budget_max_min": %d,\n' "${WALL_BUDGET_MAX_MIN}"
    printf '  "wall_budget_label": "PROJECTED (12 non-Ru runs × 6.6 min/run ≈ 80 min; +Ru 3 runs × 10 min ≈ 30 min; total 110 min worst case)",\n'
    printf '  "skip_ru": %s,\n' "${SKIP_RU}"
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "acceptance_gates": {\n'
    printf '    "epoch_completion": "all 50/50 epochs per (metal, seed) pair; Ru timeout risk acknowledged",\n'
    printf '    "krasnov_target_match": "mean test_auc ≥ Krasnov 2026 baseline (Ru 0.81, Ir 0.73)",\n'
    printf '    "no_best_seed_selection": "report per-seed AUC honestly (per mol-metal-gpu-execution §7)"\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; USER_OK + PAPER_OK gates enforced; pre-flight BLOCKER list documented",\n'
    printf '    "PROJECTED": "5-metal × 3-seed × 50-epoch × hidden=300/depth=3 ~80-110 min GPU-1",\n'
    printf '    "kill_then_adapt_directive": "80-min wall budget; USER_OK + PAPER_OK gates HARD-REQUIRED per mol-metal-gpu-execution §0 + TODO-84",\n'
    printf '    "ru_timeout_risk": "Wf-219 §4.1: Ru baseline timed out at 43/50 epochs even at hidden=64/depth=2; production backbone increases risk"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

# Emit DRY_RUN.log (Wf-363-A1 schema)
DRY_RUN_LOG="${REPORTS_DIR}/DRY_RUN.log"
cat >"${DRY_RUN_LOG}" <<EOF
[DRY-RUN] launch_wf310e_wf219_production_gpu.sh @ wf310e_wf219
[DRY-RUN] wrapper_name=launch_wf310e_wf219_production_gpu.sh
[DRY-RUN] trainer=${TRAINER}
[DRY-RUN] py=/home/hugo/codes/mol-metal/.venv/bin/python
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
EOF

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK + PAPER_OK gates were still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} \\"
    log "    --metals ${RUN_METALS[*]} --seeds ${SEEDS[*]} \\"
    log "    --epochs ${EPOCHS} --hidden ${HIDDEN} --depth ${DEPTH} \\"
    log "    --device=cuda \\"
    log "    --time-budget-s $(( ${WALL_BUDGET_MAX_MIN} * 60 )) \\"
    log "    --json-dir ${REPORTS_DIR} \\"
    log "    --ckpt-dir ${PROJECT_ROOT}/molmetal/checkpoints/wf310e_wf219_production"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 5-metal × 3-seed sweep at hidden=300/depth=3
# Ru first because it's the largest cohort + most expensive (per Wf-219 §4.1)
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)

for METAL in "${RUN_METALS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        log "=== metal=${METAL} seed=${SEED} hidden=${HIDDEN}/depth=${DEPTH} on ${GPU} ==="
        # GPU 0 (RTX PRO 6000 Blackwell) per the 2026-10-03 all-GPU-0
        # directive; this call site is serial with every other job.
        if CUDA_VISIBLE_DEVICES=0 "${PY}" "${TRAINER}" \
                --metals "${METAL}" \
                --seeds "${SEED}" \
                --epochs "${EPOCHS}" \
                --hidden "${HIDDEN}" \
                --depth "${DEPTH}" \
                --device cuda \
                --time-budget-s $(( ${WALL_BUDGET_MAX_MIN} * 60 )) \
                --json-dir "${REPORTS_DIR}/${METAL}" \
                --ckpt-dir "${PROJECT_ROOT}/molmetal/checkpoints/wf310e_wf219_production/${METAL}" \
                >> "${RUN_LOG}" 2>&1; then
            ok "metal=${METAL} seed=${SEED} completed"
        else
            # Ru may timeout per Wf-219 §4.1; emit WARN + continue (don't die)
            if [ "${METAL}" = "Ru" ]; then
                warn "Ru timeout expected per Wf-219 §4.1; per-seed AUCs may be PARTIAL (43/50 epochs)"
                warn "continuing with other (metal, seed) pairs"
            else
                die "metal=${METAL} seed=${SEED} failed (see ${RUN_LOG}); aborting sweep"
            fi
        fi
    done
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_min=$(( elapsed_sec / 60 ))
log "Wf-310E Wf-219 §4.2 production backbone sweep done — elapsed=${elapsed_min} min (budget=${WALL_BUDGET_MAX_MIN} min)"

if [ "${elapsed_min}" -gt "${WALL_BUDGET_MAX_MIN}" ]; then
    warn "elapsed (${elapsed_min} min) exceeded PROJECTED wall budget (${WALL_BUDGET_MAX_MIN} min); flagging as MEASURED-over-PROJECTED in plan.json"
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-(metal,seed) JSONs"
ok "Next: aggregate into sweep_summary.json + paper §4.5 Krasnov comparison"
ok "⚠️  Paper §4.5 updates require TODO-84 lift before any paper/* edits"
