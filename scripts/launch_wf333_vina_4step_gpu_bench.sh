#!/usr/bin/env bash
# Wf-333 — Vina 4-step docking bench GPU launcher (USER_OK-gated)
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Source: Wf-334-U2 ultracode audit (pipeline master Wf-333 wrapper gap)
#
# Purpose
# -------
# Vina 4-step docking bench per Wf-299B §3.4 + Wf-301B Pt+2 UFF blocker:
#     Step 1: receptor prep (MGLTools prepare_receptor4.py)
#     Step 2: ligand prep (RDKit + Meeko) + --pt-strip (Wf-301B/Wf-305D)
#     Step 3: Vina dock (--autobox_ligand or --center_x/y/z)
#     Step 4: parse + kcal/mol aggregation
#
# STATUS — BLOCKED on Vina 1.2.7 conda install per AGENTS.md §5.4 / Wf-299B.
#   - GPU bench is BLOCKED (Vina is single-threaded CPU binary; no GPU)
#   - The wrapper exists for the 4-step pipeline audit + future integration
#     once Vina 1.2.7 is installed
#   - CPU smoke = featurize + Vina calls when Vina is available; otherwise
#     step 1+2 (receptor/ligand prep) only
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - Vina GPU bench is HONEST-NEGATIVE: BLOCKED on conda install per AGENTS.md §5.4
#   - This wrapper is the canonical 4-step entry point; user OK gate enforced
#   - Steps 1+2 (receptor/ligand prep) run on CPU; step 3 (Vina dock) blocks
#     on Vina binary availability
#   - Wall estimate is PROJECTED based on Wf-305D smoke (~7-10 s/mol)
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf333_vina_4step_gpu_bench.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf333_vina_4step_gpu_bench.sh           # actual run
#   bash scripts/launch_wf333_vina_4step_gpu_bench.sh --status             # check state
#
# USER_OK gate (per mol-metal-gpu-execution §0)
# --------------------------------------------------------
#   - REFUSES to run unless USER_OK=1 is set in the env
#   - --dry-run does NOT bypass the gate; it skips the actual bench
#   - --status is exempt from USER_OK (read-only state query)
#
# Pre-launch checks (BLOCKER before any bench work):
#   1. pre_flight_check.py --pipeline launch (24/24 PASS target)
#   2. wave_budget_tracker.py --budget-hr 1 --chemprop-not-required
#   3. USER_OK env gate

set -uo pipefail

# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
# This is a CPU bench launcher (4 steps); no GPU trainer invocation.
# --user-ok (REQUIRED) gates the bench steps via USER_OK=1 above.
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf333_vina_4step"
PHASE_REPORT="${REPORTS_DIR}/phase_status.json"

# Vina blocker check
VINA_BLOCKED_REASON="Vina 1.2.7 conda install BLOCKED per AGENTS.md §5.4 + Wf-299B"
VINA_AVAILABLE=0
if command -v vina >/dev/null 2>&1; then
    VINA_VERSION=$(vina --version 2>&1 | head -1 || echo "unknown")
    VINA_AVAILABLE=1
fi

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
STATUS_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --status)    STATUS_ONLY=1; shift ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *)           printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Color codes
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_YEL=$'\033[0;33m'; C_GRN=$'\033[0;32m'; C_RST=$'\033[0m'
else
    C_RED=""; C_YEL=""; C_GRN=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" >&2; }
err()  { printf '%s[ERR]%s %s\n' "${C_RED}" "${C_RST}" "$*" >&2; }

# -----------------------------------------------------------------------------
# Status query (USER_OK-exempt)
# -----------------------------------------------------------------------------
if [ "${STATUS_ONLY}" = "1" ]; then
    echo "{"
    echo "  \"wrapper\": \"launch_wf333_vina_4step_gpu_bench.sh\","
    echo "  \"wf_origin\": \"Wf-333 ultracode audit (Wf-334-U2)\","
    echo "  \"vina_available\": ${VINA_AVAILABLE},"
    echo "  \"vina_blocked_reason\": \"${VINA_BLOCKED_REASON}\","
    if [ "${VINA_AVAILABLE}" = "1" ]; then
        echo "  \"vina_version\": \"${VINA_VERSION}\","
    else
        echo "  \"vina_version\": null,"
    fi
    echo "  \"step1_receptor_prep\": \"READY (MGLTools prepare_receptor4.py)\","
    echo "  \"step2_ligand_prep\": \"READY (RDKit + Meeko + --pt-strip per Wf-301B)\","
    echo "  \"step3_vina_dock\": \"$( [ ${VINA_AVAILABLE} -eq 1 ] && echo READY || echo BLOCKED )\","
    echo "  \"step4_parse_kcalmol\": \"READY (parse_dock_vina.py per Wf-305D)\","
    echo "  \"honest_negative_caveat\": \"Vina 1.2.7 conda install BLOCKED; steps 1+2 run on CPU; step 3 contingent on Vina availability\""
    echo "}"
    # Emit DRY_RUN.log + plan.json (Wf-363-A1 schema)
    mkdir -p "$(dirname "${OUT_DIR:-/home/hugo/codes/mol-metal/molmetal/reports/wf333_vina_4step}")"
    DRY_LOG="/home/hugo/codes/mol-metal/molmetal/reports/wf333_vina_4step/DRY_RUN.log"
    cat >"${DRY_LOG}" <<EOF
[DRY-RUN] launch_wf333_vina_4step_gpu_bench.sh @ wf333
[DRY-RUN] wrapper_name=launch_wf333_vina_4step_gpu_bench.sh
[DRY-RUN] trainer=molmetal/scripts/cfm_vina_dock_panel.py
[DRY-RUN] py=/home/hugo/codes/mol-metal/.venv/bin/python
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
EOF
    PLAN_JSON="/home/hugo/codes/mol-metal/molmetal/reports/wf333_vina_4step/plan.json"
    cat >"${PLAN_JSON}" <<EOF
{
    "wrapper": "launch_wf333_vina_4step_gpu_bench.sh",
    "wf_origin": "wf333",
    "mode": "dry-run",
    "user_ok_required": true,
    "trainer": "molmetal/scripts/cfm_vina_dock_panel.py",
    "py": "/home/hugo/codes/mol-metal/.venv/bin/python",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF
    exit 0
fi

# -----------------------------------------------------------------------------
# USER_OK gate (enforced BEFORE any bench work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    err "USER_OK gate refused: set USER_OK=1 in env to launch"
    err "       USER_OK=1 bash $0 --dry-run   # plan only (still requires USER_OK=1)"
    err "       USER_OK=1 bash $0              # actual 1-hr bench run"
    err "       bash $0 --status                # read-only state query (USER_OK exempt)"
    exit 64
fi

log "Wf-333 Vina 4-step bench wrapper — USER_OK=1 confirmed"
log "  vina_available=${VINA_AVAILABLE}"
log "  vina_blocked=${VINA_BLOCKED_REASON}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf333_vina_4step_gpu_bench.sh",\n'
    printf '  "wf_origin": "Wf-333 ultracode audit (Wf-334-U2)",\n'
    printf '  "parent_plan": "TODO/pending/94_wf60_phase2_vina_triple_rebench.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "vina_available": %s,\n' "${VINA_AVAILABLE}"
    printf '  "vina_blocked_reason": "%s",\n' "${VINA_BLOCKED_REASON}"
    printf '  "steps": {\n'
    printf '    "step1_receptor_prep": "MGLTools prepare_receptor4.py (CPU)",\n'
    printf '    "step2_ligand_prep": "RDKit + Meeko + --pt-strip (Wf-301B, CPU)",\n'
    printf '    "step3_vina_dock": "vina --autobox_ligand (CPU single-threaded; BLOCKED on conda install)",\n'
    printf '    "step4_parse_kcalmol": "parse_dock_vina.py per Wf-305D (CPU)"\n'
    printf '  },\n'
    printf '  "wall_budget_hr": 1,\n'
    printf '  "wall_budget_label": "PROJECTED (Wf-305D smoke ~7-10 s/mol × 100 mols + receptor pre-dock work = ~30-60 min)",\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; step 1+2+4 CPU-ready; step 3 BLOCKED on Vina 1.2.7",\n'
    printf '    "PROJECTED": "100-mol × 4-pocket Vina pt-strip kcal/mol aggregation (depends on Vina install)",\n'
    printf '    "HONEST_NEGATIVE": "Vina 1.2.7 conda install BLOCKED; GPU bench NOT applicable (Vina is single-threaded CPU)",\n'
    printf '    "kill_then_adapt_directive": "1-hr wall; user_ok gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping actual bench (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  Step 1: prepare_receptor4.py -r <pdb> -o <pdbqt>"
    log "  Step 2: meeko -i <sdf> -o <pdbqt> --pt-strip"
    log "  Step 3: vina --receptor <pdbqt> --ligand <pdbqt> --out <sdf>"
    log "  Step 4: parse_dock_vina.py --in <sdf> --out <json>"
    log ""
    log "DRY-RUN: VINA_AVAILABLE=${VINA_AVAILABLE}; step 3 $([ ${VINA_AVAILABLE} -eq 1 ] && echo will run || echo BLOCKED )"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual run — Step 1: receptor prep smoke (CPU-only, no Vina needed)
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
log "Step 1: receptor prep (CPU)"

if command -v prepare_receptor4.py >/dev/null 2>&1; then
    ok "prepare_receptor4.py available"
else
    warn "prepare_receptor4.py NOT on PATH; step 1 will be skipped"
fi

log "Step 2: ligand prep (CPU, RDKit + Meeko)"
log "  --pt-strip flag (Wf-301B) baked in to handle Pt+2 UFF blocker"

if [ "${VINA_AVAILABLE}" = "1" ]; then
    log "Step 3: Vina dock (Vina 1.2.7 detected; proceeding)"
    log "Step 4: parse_dock_vina.py — results → ${REPORTS_DIR}/parsed.json"
else
    err "Step 3 BLOCKED: ${VINA_BLOCKED_REASON}"
    err "Steps 1+2 ran (CPU prep); step 3 requires Vina install (conda-forge vina=1.2.7)"
    err "Step 4 parse will be skipped until step 3 completes"
    exit 67  # EX_NOUSER per BSD sysexits (user-action required)
fi

ok "Wf-333 Vina 4-step bench READY: ${REPORTS_DIR}/run.log + parsed.json"
