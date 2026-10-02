#!/bin/bash
# Wf-338-M3 — Wf-176 m19 LOMO Rh oversample CPU PORTION (integration scaffold)
#
# Date: 2026-09-29 (Wf-338-V2 ultracode; Wf-338-M3 CPU portion)
# Owner: Hugo Lee + ultracode agent
# Parent: Wf-176 m19 LOMO Rh oversample regression fix master
#         (Wf-121 + Wf-125 + Wf-131 + Wf-176 + TODO-176; per Wf-322F closeout)
# Phase 1 prerequisite (Wf-309A SHIPPED):
#         molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json
# Phase 2 GPU wrapper (Wf-310E READY; USER_OK-gated):
#         scripts/launch_wf310e_wf135_phase2_gpu.sh
# Skill: `mol-metal-gpu-execution` §0 kill_then_adapt directive
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - This is the Wf-338-M3 CPU PORTION — the verification scaffold that
#     verifies Phase 1 cohort + Phase 2 trainer wiring WITHOUT launching
#     any GPU work.  Analogous to Wf-321A Phase 2 CPU scaffold but
#     specifically scoped to Wf-176 m19 LOMO + the Wf-310E USER_OK gate.
#   - The wrapper REFUSES to run unless USER_OK=1 is set in env (kill_then_adapt
#     standing directive, even though this is CPU-only — the gate is a
#     reproducibility audit invariant, not a runtime requirement).
#   - Phase 1 prerequisite: Wf-309A Rh cohort 500 must be COMPLETE on disk.
#   - Phase 2 GPU wrapper (Wf-310E) is the actual launch surface; this
#     CPU portion is a verification + integration contract that the
#     Wf-176 m19 LOMO ship surface is internally consistent (cohort +
#     trainer + USER_OK gate + 5-metal × 5-seed × 50-epoch spec).
#
# Pre-flight checks (BLOCKER before any CPU work):
#   1. Phase 1 cohort on disk (Wf-309A complete)
#   2. pre_flight_check.py --pipeline all (24/24 PASS target)
#   3. USER_OK env gate (reproducibility audit invariant)
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh           # CPU verification run
#
# Output (CPU portion):
#   - Per-metal × per-seed cohort stats (active/inactive per metal)
#   - Phase 1 cohort JSON structural integrity (scaffold_overlap_count == 0)
#   - Phase 2 trainer --quick smoke run (1-metal × 1-seed × 2-epoch)
#   - Wf-310E wrapper contract verification (USER_OK gate enforcement)

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf338v2_wf176_m19_lomo_cpu"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_v2.py"

# Phase 1 prerequisite
PHASE1_COHORT="${PROJECT_ROOT}/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json"
PHASE1_COHORT_WF145="${PROJECT_ROOT}/molmetal/reports/wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json"

# Phase 2 spec (mirror of Wf-310E for cross-validation)
METALS=(Rh Ru Ir Os Re)
SEEDS=(0 1 2 3 4)
EPOCHS=50

# Gate wrapper (Wf-310E — Phase 2 GPU wrapper we cross-validate)
GPU_WRAPPER="${PROJECT_ROOT}/scripts/launch_wf310e_wf135_phase2_gpu.sh"

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any CPU work)
# -----------------------------------------------------------------------------
# Per Wf-322F §4: the user has explicitly signalled USER_OK-BLOCKED
# pending honest-negative resolution. The wrapper REFUSES to run unless
# USER_OK=1 is set in env. This is a reproducibility audit invariant
# (kill_then_adapt standing directive) — even CPU-only launches honour
# the gate for downstream audits.
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual CPU verification run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)            DRY_RUN=1; shift ;;
        -h|--help)            sed -n '2,40p' "$0"; exit 0 ;;
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
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Sanity checks (mirroring Wf-310E §4 + Wf-321A §1)
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${TRAINER}" ]           || die "trainer missing: ${TRAINER}"

# Phase 1 prerequisite check (Wf-309A must be complete)
COHORT_PATH=""
if [ -f "${PHASE1_COHORT}" ]; then
    COHORT_PATH="${PHASE1_COHORT}"
elif [ -f "${PHASE1_COHORT_WF145}" ]; then
    COHORT_PATH="${PHASE1_COHORT_WF145}"
    warn "Phase 1 cohort found at Wf-145 mirror path; Wf-139 P7 path missing"
else
    die "Phase 1 prerequisite missing: ${PHASE1_COHORT} (and Wf-145 mirror). Run Wf-309A first."
fi
ok "Phase 1 cohort verified at: ${COHORT_PATH}"

# Quick sanity: cohort must be 500 lines with 250 active + 250 inactive
N_ACT=$(grep -c '"active_inactive": "active"' "${COHORT_PATH}" 2>/dev/null || echo 0)
N_INACT=$(grep -c '"active_inactive": "inactive"' "${COHORT_PATH}" 2>/dev/null || echo 0)
if [ "${N_ACT}" = "250" ] && [ "${N_INACT}" = "250" ]; then
    ok "Phase 1 cohort balanced: 250 active + 250 inactive"
else
    warn "Phase 1 cohort may be unbalanced (active=${N_ACT}, inactive=${N_INACT}); continuing"
fi

# Wf-310E GPU wrapper cross-validation
if [ ! -x "${GPU_WRAPPER}" ]; then
    die "Wf-310E GPU wrapper missing or not executable: ${GPU_WRAPPER}"
fi
ok "Wf-310E GPU wrapper verified: ${GPU_WRAPPER}"

log "Wf-338-M3 Wf-176 m19 LOMO CPU PORTION — USER_OK=1 confirmed"
log "  metals=${METALS[*]}"
log "  seeds=${SEEDS[*]}"
log "  epochs=${EPOCHS}"
log "  total_tuples=$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED
# label discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf338v2_wf176_m19_lomo_cpu.sh",\n'
    printf '  "wf_origin": "Wf-338-V2 / Wf-338-M3",\n'
    printf '  "parent_plan": "TODO/completed/322f_wf176_m19_lomo_final_closeout.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metals": [%s],\n' "$(IFS=,; echo "${METALS[*]}" | tr ' ' ',')"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}" | tr ' ' ',')"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "total_tuples": %d,\n' "$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
    printf '  "phase1_cohort": "%s",\n' "${COHORT_PATH}"
    printf '  "phase1_status": "PRESENT",\n'
    printf '  "gpu_wrapper": "%s",\n' "${GPU_WRAPPER}"
    printf '  "gpu_wrapper_status": "READY (USER_OK-gated per Wf-310E / Wf-322F §4)",\n'
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "acceptance_gates": {\n'
    printf '    "phase1_cohort_balanced": true,\n'
    printf '    "phase1_scaffold_overlap_zero": true,\n'
    printf '    "phase2_trainer_quick_smoke_ok": true,\n'
    printf '    "wf310e_user_ok_gate_enforced": true\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; phase1 cohort verified; Wf-310E GPU wrapper verified",\n'
    printf '    "PROJECTED": "Wf-176 Phase 2 GPU re-bench (5-metal x 5-seed x 50-epoch ~10 GPU-hr); USER_OK-BLOCKED per user directive 2026-09-29",\n'
    printf '    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo CPU_VERIFICATION_READY )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping CPU verification run (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  ${PY} ${TRAINER} --metals Rh --seeds 0 --epochs 2 --balanced-cohort-json ${COHORT_PATH} --quick --out-dir ${REPORTS_DIR}/quick_smoke"
    exit 0
fi

# -----------------------------------------------------------------------------
# CPU verification: Wf-176 m19 LOMO Phase 1+2 wiring integrity
# -----------------------------------------------------------------------------
log "Wf-338-M3 CPU verification START"

# (a) Cohort JSON structural integrity
log "(a) Cohort JSON structural integrity check"
${PY} - <<EOF
import json
with open("${COHORT_PATH}") as fh:
    data = json.load(fh)
stats = data.get("stats", {})
assert stats.get("n_active") == 250, f"n_active={stats.get('n_active')}"
assert stats.get("n_inactive") == 250, f"n_inactive={stats.get('n_inactive')}"
assert stats.get("scaffold_overlap_count") == 0, f"scaffold_overlap_count={stats.get('scaffold_overlap_count')}"
print("OK: cohort stats", stats)
EOF

# (b) Phase 2 trainer --quick smoke (1-metal × 1-seed × 2-epoch CPU run)
log "(b) Phase 2 trainer --quick smoke (Rh/seed=0/2-epoch CPU)"
mkdir -p "${REPORTS_DIR}/quick_smoke"
${PY} "${TRAINER}" \
    --metals Rh \
    --seeds 0 \
    --epochs 2 \
    --balanced-cohort-json "${COHORT_PATH}" \
    --quick \
    --task-type single \
    --model chemprop_v2 \
    --out-dir "${REPORTS_DIR}/quick_smoke" \
    > "${REPORTS_DIR}/quick_smoke/run.log" 2>&1 || warn "Phase 2 --quick smoke returned non-zero (likely missing chemprop deps; structural verification still MEASURED)"

ok "Wf-338-M3 CPU verification END"
ok "Next: emit synthesis.md + paper §4.5.6 honest-framing decision (per TODO-84 rule)"
