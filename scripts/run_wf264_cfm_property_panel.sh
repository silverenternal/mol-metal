#!/usr/bin/env bash
# Wf-338-F1 — Wf-264 CFM property panel run-once launcher
#
# =====================================================================
# Wf-338-F1 (2026-09-29) — Run-once launcher for Wf-264 CFM property panel
# =====================================================================
#
# Owner: Hugo Lee + ultracode agent
# Date: 2026-09-29
# Source: Wf-338-F1 ultracode residual-gap closure wave
#        (user directive 2026-09-29 verbatim "好像有些之前发现的问题没有被覆盖到,
#         可以额外启动新的ultracode吗?" honored at the Wf-338 layer —
#         provides a thin run-once launcher that enforces the user OK
#         gate + pre-flight check + exit-code contract for batch
#         scheduling (e.g. cron, pipeline orchestrator)).
#
# PURPOSE
# -------
# The existing scripts/launch_wf264_cfm_property_panel.sh is the FULL
# launcher (with --dry-run gate + USER_OK gate + pre-flight + real run).
# This wrapper is the THIN run-once entry point for batch scheduling:
#   - It REFUSES --dry-run (run-once is real-only; if you want a
#     dry-run, use the full launcher).
#   - It enforces USER_OK=1 unconditionally (kill_then_adapt standing
#     directive).
#   - It invokes pre_flight_check.py --pipeline launch (TODO-207 §1 gate).
#   - It surfaces a deterministic exit code (0=PASS, 1=FAIL, 2=SKIP,
#     3=USER_OK_GATE_DENIED) for pipeline orchestration.
#
# Per AGENTS.md §5.1 standing GPU authorization ("显存够就放心大胆用"):
#   - The wrapper is USER_OK-gated (kill_then_adapt standing directive)
#   - The actual property panel is CPU-only (RDKit SA + QED on
#     pre-existing SMILES); no GPU launches, no torch.compile / BF16 /
#     TF32 wiring.  Those Wf-328 flags apply to the D-MPNN Wasserstein
#     trainer (Wf-334-U1 wired --dmpnn-backend there), NOT to this
#     post-hoc property panel.
#   - Wf-328-W2 --num-workers 8 is documented but NOT wired: the panel
#     script does not use a DataLoader; SMILES are processed sequentially
#     in a single process.  This is honest-negative per §5.2.
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - MEASURED (2026-09-28): Wf-264 panel ran successfully on the
#     Wf-188 MetalCytoToxDB ligand population (n=26,801); SA mean=3.289,
#     QED mean=0.280, per molmetal/reports/wf264_panel/panel_report.md.
#   - MEASURED: Vina sub-panel MEASURED 0/100 docked ok on 1h36 (meeko
#     0.8.0 metal-atom limitation, honest-negative per Wf-264 §2).
#   - PROJECTED wall: ~10 min CPU on 1 worker (RDKit SA + QED compute).
#   - HONEST_NEGATIVE_VINA: Vina sub-panel is BLOCKED on meeko+Pt
#     parameterisation per AGENTS.md §5.4 + Wf-299B.
#
# USAGE
# -----
#   USER_OK=1 bash scripts/run_wf264_cfm_property_panel.sh          # actual run-once
#   bash scripts/run_wf264_cfm_property_panel.sh --status           # read-only state query
#
# EXIT CODES
# ----------
#   0 = PASS (panel ran successfully)
#   1 = FAIL (pre-flight FAIL, or panel raised exception)
#   2 = SKIP (panel deferred / blocked)
#   3 = USER_OK_GATE_DENIED (USER_OK=1 not set)

set -uo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
LAUNCHER="${PROJECT_ROOT}/scripts/launch_wf264_cfm_property_panel.sh"
OUT_BASE="${PROJECT_ROOT}/results/wf264_cfm_property_panel"
LOG_DIR="${PROJECT_ROOT}/molmetal/reports/wf338f1_wf264_run"
WRAPPER_LOG="${LOG_DIR}/wrapper.log"
STATE_FILE="${LOG_DIR}/state.json"

mkdir -p "${OUT_BASE}" "${LOG_DIR}"

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --status)    STATUS_ONLY=1; shift ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *)           printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] [wf338f1_run_wf264] $*" | tee -a "${WRAPPER_LOG}"
}

# -----------------------------------------------------------------------------
# Status query (USER_OK-exempt)
# -----------------------------------------------------------------------------
if [ "${STATUS_ONLY}" = "1" ]; then
  echo "{"
  echo "  \"wrapper\": \"run_wf264_cfm_property_panel.sh\","
  echo "  \"wf_origin\": \"Wf-338-F1 ultracode residual-gap closure\","
  echo "  \"full_launcher\": \"${LAUNCHER}\","
  echo "  \"out_base\": \"${OUT_BASE}\","
  echo "  \"user_ok_required\": true,"
  echo "  \"dry_run_supported\": false,"
  echo "  \"honest_negative_caveat\": \"Vina sub-panel is BLOCKED on meeko+Pt per AGENTS.md §5.4 + Wf-299B\""
  echo "}"
  exit 0
fi

# -----------------------------------------------------------------------------
# USER_OK gate (enforced BEFORE any panel work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
  log "USER_OK_GATE_DENIED — set USER_OK=1 to authorise real launch"
  log "  Re-run with: USER_OK=1 bash $0"
  log "  --status query is exempt (no gate)"
  exit 3
fi

log "Wf-338-F1 Wf-264 run-once launcher — USER_OK=1 confirmed"
log "  full_launcher=${LAUNCHER}"
log "  out_base=${OUT_BASE}"

# -----------------------------------------------------------------------------
# Pre-flight check (TODO-207 §1 gate)
# -----------------------------------------------------------------------------
log "Running pre_flight_check.py --pipeline launch"
if ! "${PY}" "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all >/dev/null 2>&1; then
  log "PRE_FLIGHT FAIL — aborting (TODO-207 §1.4 standing rule)"
  cat > "${STATE_FILE}" <<EOF
{
  "wf338f1_run_wf264_state": {
    "date": "$(date -Iseconds)",
    "user_ok": true,
    "preflight": "FAIL",
    "panel_rc": null,
    "exit_code": 1
  }
}
EOF
  exit 1
fi
log "PRE_FLIGHT PASS"

# -----------------------------------------------------------------------------
# Actual run — invoke the full launcher (which itself enforces USER_OK +
# pre-flight; we re-enforce here for orchestration determinism).
# -----------------------------------------------------------------------------
PANEL_LOG="${LOG_DIR}/panel.log"
log "Invoking ${LAUNCHER}"
log "  panel log: ${PANEL_LOG}"

USER_OK=1 bash "${LAUNCHER}" > "${PANEL_LOG}" 2>&1
panel_rc=$?
log "  panel_rc=${panel_rc}"

# Persist state
cat > "${STATE_FILE}" <<EOF
{
  "wf338f1_run_wf264_state": {
    "date": "$(date -Iseconds)",
    "user_ok": true,
    "preflight": "PASS",
    "panel_rc": ${panel_rc},
    "exit_code": ${panel_rc},
    "out_base": "${OUT_BASE}"
  }
}
EOF
log "State persisted to ${STATE_FILE}"
log "=== Wf-338-F1 Wf-264 run-once launcher COMPLETE (rc=$panel_rc) ==="

exit "${panel_rc}"
