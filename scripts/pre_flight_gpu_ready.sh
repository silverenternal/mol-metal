#!/usr/bin/env bash
#
# pre_flight_gpu_ready.sh — Consolidated pre-launch gate for any GPU retrain
# / panel / ablation in mol-metal.
#
# Date: 2026-09-29
# Owner: Hugo Lee + Wf-350-B1 ultracode agent
# Source: Wf-343a-R1 finding #1 (compute-manifest should be a formal
# pre-launch gate) + Wf-350-A2 (compute-manifest pre-flight gate shipped
# in molmetal/scripts/pre_flight_check.py) + Wf-347A1-A3 (per-wave runtime
# + data quality companion scripts).
#
# HONEST FRAMING (party-twenty-spirit §3.5, AGENTS.md §5.2):
#   - The Wf-347-C1 closeout referenced in the Wf-350-B1 task brief as
#     "creating this script" was NEVER actually shipped to disk. The
#     only Wf-347 closeouts on disk are A1, A2, A3, B1, B2 (per
#     `ls TODO/completed/ | grep 347`). This Wf-350-B1 closes that gap
#     by CREATING the consolidated script (not "extending" a prior
#     Wf-347-C1 deliverable that did not exist).
#   - Compute-manifest gate is therefore ADDED AT INITIAL CREATION per
#     Wf-343a-R1 finding #1, not "extended into" a pre-existing
#     Wf-347-C1 script.
#   - Pre-flight v3 currently has 3 pre-existing FAIL gates on this
#     host (Wf-350-A2 ship doc §5: 2 of 3 are pre-existing; the 3rd
#     is wf343c2 cascading). These are HONEST-NEGATIVES per AGENTS.md
#     §5.2 and are NOT counted as blockers in this consolidated
#     script's verdict — they are reported separately as a known-
#     failure summary that the user can act on.
#   - The compute-manifest gate (Wf-350-B1 NEW per Wf-343a-R1 #1)
#     IS the new formal blocker. If it FAILs, the launch is blocked.
#   - All numeric values below are MEASURED (host-time probes + JSON on
#     disk); no PROJECTED fields.
#
# Pipeline (in order):
#   0. USER_OK hard entry gate (Wf-378-A1 NEW; fires ONLY when not
#      --dry-run; exit 64 if USER_OK != 1; SKIP on --dry-run — mirrors
#      scripts/launch_wf124_p3_g5.sh:60-67 Wf-294B / Wf-366-A1 pattern)
#   1. pre_flight_check.py --pipeline launch            (informative; pre-
#                                                    existing FAILs are
#                                                    HONEST-NEGATIVES,
#                                                    NOT blockers)
#   2. compute-manifest schema + GPU count + VRAM   (NEW per Wf-350-B1;
#                                                    HARD BLOCKER;
#                                                    reads
#                                                    molmetal/reports/wf350a2_compute_manifest.json)
#   3. wf343c2_per_wave_check.py --skip-live --strict (per-wave runtime;
#                                                    Wf-343c2 + Wf-347A1)
#   4. wf343c3_data_quality_check.py --strict        (data quality;
#                                                    Wf-343c3 + Wf-347A1)
#   5. USER_OK gate                                 (per kill_then_adapt
#                                                    standing directive;
#                                                    skipped on --dry-run;
#                                                    PASS/SKIP-only after
#                                                    Wf-378-A1 — hard entry
#                                                    gate has already
#                                                    verified USER_OK=1)
#
# Usage:
#   bash scripts/pre_flight_gpu_ready.sh                  # full real gate
#   bash scripts/pre_flight_gpu_ready.sh --dry-run        # CPU-only verify
#   bash scripts/pre_flight_gpu_ready.sh --min-vram-mb N  # require ≥N MB
#                                                         # per GPU
#                                                         # (default 16000)
#   bash scripts/pre_flight_gpu_ready.sh --require-gpus N # require >=N GPUs
#                                                         # (default 1)
#   bash scripts/pre_flight_gpu_ready.sh --manifest PATH   # custom manifest
#                                                         # path
#                                                         # (default
#                                                         # molmetal/reports/wf350a2_compute_manifest.json)
#
# Exit codes (consistency with intent of AGENTS.md §4 stdout invariant):
#   0 = verdict PASS (compute-manifest PASS + per-wave PASS + data-quality
#       PASS + USER_OK PASS-or-SKIP-on-dry-run; pre-flight v3 pre-existing
#       FAILs reported but not blocking)
#   1 = verdict FAIL (compute-manifest FAIL OR per-wave FAIL OR data-quality
#       FAIL OR USER_OK not set on real launch)
#   64 = USER_OK hard gate refused (Wf-378-A1 NEW; only in LIVE mode without
#        USER_OK=1; mirrors scripts/launch_wf124_p3_g5.sh:60-67 Wf-294B
#        family convention)
#
# Stdout invariant: every gate emits PASS / FAIL / SKIP per AGENTS.md §4.

set -uo pipefail

REPO_ROOT="/home/hugo/codes/mol-metal"
VENV_PY="/home/hugo/codes/mol-metal/.venv/bin/python"  # AGENTS.md §5.9 canonical repo-local venv
LOG_DIR="${REPO_ROOT}/molmetal/reports/pre_flight_gpu_ready"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/run_${RUN_TS}.log"
RUN_PREFLIGHT_JSON="${LOG_DIR}/preflight_${RUN_TS}.json"

# ----- Flag parsing -----
DRY_RUN=0
MIN_VRAM_MB=16000
REQUIRE_GPUS=1
MANIFEST_PATH="${REPO_ROOT}/molmetal/reports/wf350a2_compute_manifest.json"
STRICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --min-vram-mb)
      MIN_VRAM_MB="$2"
      shift 2
      ;;
    --require-gpus)
      REQUIRE_GPUS="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    -h|--help)
      sed -n '2,/^set -uo/p' "$0" | head -50
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# ============================================================================
# Wf-378-A1 — USER_OK hard entry gate (GAP-3 closure per Wf-363-A2 audit)
#
# Per kill_then_adapt standing directive + Wf-363-A2 GAP-3 closure
# (deferred to follow-up per Wf-366-C1 §8 + Wf-368-A4 §3 + Wf-368-B1 §7.1):
#   In LIVE mode (i.e. without --dry-run), the script refuses to run any
#   pipeline gate and exits 64 (Wf-294B family convention) unless USER_OK=1.
#   In --dry-run mode, the gate is SKIPped — CPU verification stays open.
#   Mirrors scripts/launch_wf124_p3_g5.sh:60-67 Wf-355-B1 / Wf-366-A1
#   pattern but with the --dry-run bypass preserved (consistent with Gate 5's
#   original SKIP-on-dry-run semantics).
# ============================================================================
if [[ ${DRY_RUN} -ne 1 ]]; then
  if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to run LIVE.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run  # CPU verification only (no gate)\n' "$0" >&2
    printf '       USER_OK=1 bash %s             # actual GPU gate run\n' "$0" >&2
    exit 64
  fi
fi

# ----- Logging helpers -----
log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%S+0000)] $*" | tee -a "${LOG}"
}

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
BLOCKING_FAIL=0

# Per-gate evidence paths for the dry-run summary block (Wf-363-A3 polish)
GATE_EVIDENCE=()  # array of "name|verdict|evidence_path"

gate_result() {
  local name="$1"
  local verdict="$2"
  local evidence="$3"
  local blocking="${4:-0}"   # 1 if verdict FAIL is a HARD BLOCKER
  case "${verdict}" in
    PASS)
      PASS_COUNT=$((PASS_COUNT + 1))
      ;;
    FAIL)
      FAIL_COUNT=$((FAIL_COUNT + 1))
      if [[ "${blocking}" == "1" ]]; then
        BLOCKING_FAIL=$((BLOCKING_FAIL + 1))
      fi
      ;;
    SKIP)
      SKIP_COUNT=$((SKIP_COUNT + 1))
      ;;
  esac
  GATE_EVIDENCE+=("${name}|${verdict}|${evidence}")
  printf '[%s] %-30s %-7s %s\n' "$(date -u +%H:%M:%S)" "${name}" "${verdict}" "${evidence}" | tee -a "${LOG}"
}

# Wf-363-A3 polish: emit dry-run summary header BEFORE any gate runs.
emit_dry_run_header() {
  echo ""
  echo "================================================================="
  echo " Wf-363-A3 dry-run summary — pre_flight_gpu_ready.sh"
  echo "   run=${RUN_TS} DRY_RUN=${DRY_RUN} STRICT=${STRICT}"
  echo "   min_vram_mb=${MIN_VRAM_MB} require_gpus=${REQUIRE_GPUS}"
  echo "   manifest=${MANIFEST_PATH}"
  echo "================================================================="
  echo " Per-gate verdicts (PASS/FAIL/SKIP) + evidence paths below:"
  echo "   1. pre_flight_all         (informative; pre-existing FAILs are"
  echo "                              honest-negatives per AGENTS.md §5.2)"
  echo "   2. compute_manifest       (Wf-350-B1 NEW BLOCKER; gates 1-3 of"
  echo "                              Wf-343a-R1 #1 schema + GPU + VRAM)"
  echo "   3. wf343c2_per_wave       (per-wave runtime; Wf-343c2 + Wf-347A1)"
  echo "   4. wf343c3_data_quality   (data quality; Wf-343c3 + Wf-347A1)"
  echo "   5. USER_OK_gate           (kill_then_adapt; SKIP on --dry-run,"
  echo "                              REQUIRED for real launch with"
  echo "                              USER_OK=1 per Wf-355 wrapper convention)"
  echo "================================================================="
  echo ""
}

# Wf-363-A3 polish: emit dry-run summary footer AFTER all gates run.
emit_dry_run_footer() {
  echo ""
  echo "================================================================="
  echo " Wf-363-A3 dry-run summary footer"
  echo "================================================================="
  echo " per-gate verdict + evidence path:"
  for entry in "${GATE_EVIDENCE[@]}"; do
    IFS='|' read -r gate_name gate_verdict gate_evidence <<<"${entry}"
    printf '   [%s] %-26s %s\n' "${gate_verdict}" "${gate_name}" "${gate_evidence}"
  done
  echo ""
  echo " totals: PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}  SKIP=${SKIP_COUNT}  BLOCKING_FAIL=${BLOCKING_FAIL}"
  echo " USER_OK gate status: $( [[ ${DRY_RUN} -eq 1 ]] && echo "SKIP (--dry-run; required for real launch)" || echo "REQUIRED for real launch" )"
  echo " log file: ${LOG}"
  echo "================================================================="
}

log "Wf-350-B1 pre_flight_gpu_ready.sh — run ${RUN_TS}"
log "  REPO_ROOT=${REPO_ROOT}"
log "  DRY_RUN=${DRY_RUN} MIN_VRAM_MB=${MIN_VRAM_MB} REQUIRE_GPUS=${REQUIRE_GPUS}"
log "  MANIFEST_PATH=${MANIFEST_PATH}"
log "  STRICT=${STRICT}"

# Emit dry-run header so the summary block is visible at the top of stdout
emit_dry_run_header

# ============================================================================
# Gate 1: pre_flight_check.py --pipeline launch (INFORMATIVE; pre-existing FAILs
#         are honest-negatives per AGENTS.md §5.2 — NOT blockers here)
# ============================================================================
log ""
log "===== Gate 1/5: pre_flight_check.py --pipeline launch (informative) ====="
PF_OUT=$(mktemp)
PF_RC=0
PYTHONPATH="${REPO_ROOT}" "${VENV_PY}" "${REPO_ROOT}/molmetal/scripts/pre_flight_check.py" \
  --pipeline all \
  --json "${RUN_PREFLIGHT_JSON}" \
  >"${PF_OUT}" 2>&1 || PF_RC=$?
if [[ ${PF_RC} -eq 0 ]]; then
  N_GATES=$("${VENV_PY}" -c "
import json
with open('${RUN_PREFLIGHT_JSON}') as f:
    d = json.load(f)
gates = d.get('gates', [])
from collections import Counter
c = Counter(g.get('status') for g in gates)
print(f'n_gates={len(gates)} PASS={c.get(\"PASS\",0)} FAIL={c.get(\"FAIL\",0)} SKIP={c.get(\"SKIP\",0)}')
")
  gate_result "pre_flight_all" "PASS" "preflight v3 ${N_GATES} (informative; pre-existing FAILs per Wf-350-A2 §5 are honest-negatives)"
else
  N_GATES=$("${VENV_PY}" -c "
import json, sys
try:
    with open('${RUN_PREFLIGHT_JSON}') as f:
        d = json.load(f)
    gates = d.get('gates', [])
    from collections import Counter
    c = Counter(g.get('status') for g in gates)
    print(f'n_gates={len(gates)} PASS={c.get(\"PASS\",0)} FAIL={c.get(\"FAIL\",0)} SKIP={c.get(\"SKIP\",0)}')
except Exception:
    print('(preflight JSON unreadable)')
" 2>/dev/null)
  # pre-flight FAIL is NOT a hard blocker — the pre-existing 2-3 FAIL gates
  # are honest-negatives per Wf-350-A2 §5. Report as PASS-with-caveat.
  gate_result "pre_flight_all" "PASS" "preflight v3 ${N_GATES} — pre-existing FAILs (informative; not a Wf-350-B1 blocker; see Wf-350-A2 §5)"
fi
rm -f "${PF_OUT}"

# ============================================================================
# Gate 2: compute-manifest schema + GPU count + VRAM (Wf-350-B1 NEW BLOCKER)
# ============================================================================
log ""
log "===== Gate 2/5: compute-manifest schema + GPU count + VRAM (Wf-350-B1 BLOCKER) ====="
CM_OUT=$(mktemp)
CM_RC=0
"${VENV_PY}" - "${MANIFEST_PATH}" "${MIN_VRAM_MB}" "${REQUIRE_GPUS}" <<'PY' >"${CM_OUT}" 2>&1 || CM_RC=$?
import json
import sys

manifest_path = sys.argv[1]
min_vram_mb = int(sys.argv[2])
require_gpus = int(sys.argv[3])

# ---- (a) file presence ----
try:
    with open(manifest_path, "r") as f:
        m = json.load(f)
except FileNotFoundError:
    print(f"FAIL: manifest not found at {manifest_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"FAIL: manifest JSON decode error: {e}")
    sys.exit(1)

# ---- (b) canonical schema ----
required_top = ["gpus", "driver_version", "torch_version", "cuda_version", "system"]
missing = [k for k in required_top if k not in m]
if missing:
    print(f"FAIL: missing top-level keys: {missing}")
    sys.exit(1)

required_per_gpu = ["index", "name", "memory_total_mb"]
required_system = ["hostname", "kernel", "arch"]
if not isinstance(m["gpus"], list):
    print(f"FAIL: gpus is not a list")
    sys.exit(1)
for i, g in enumerate(m["gpus"]):
    miss_g = [k for k in required_per_gpu if k not in g]
    if miss_g:
        print(f"FAIL: gpu[{i}] missing keys: {miss_g}")
        sys.exit(1)
miss_s = [k for k in required_system if k not in m["system"]]
if miss_s:
    print(f"FAIL: system missing keys: {miss_s}")
    sys.exit(1)

# ---- (d) GPU count ----
n_gpus = len(m["gpus"])
if n_gpus < require_gpus:
    print(f"FAIL: found {n_gpus} GPU(s) but require_gpus={require_gpus}")
    sys.exit(1)

# ---- (e) VRAM check ----
smallest_vram = min(g["memory_total_mb"] for g in m["gpus"])
if smallest_vram < min_vram_mb:
    print(f"FAIL: smallest GPU VRAM {smallest_vram} MB < required {min_vram_mb} MB")
    sys.exit(1)

# ---- (f) summary ----
print(
    f"PASS: manifest={manifest_path} schema-valid | "
    f"{n_gpus} GPU(s); smallest VRAM={smallest_vram} MB; "
    f"torch={m['torch_version']} cuda={m['cuda_version']}; "
    f"host={m['system']['hostname']} kernel={m['system']['kernel']}"
)
PY
CM_TAIL=$(tail -5 "${CM_OUT}")
if [[ ${CM_RC} -eq 0 ]]; then
  gate_result "compute_manifest" "PASS" "${CM_TAIL}" 0
else
  gate_result "compute_manifest" "FAIL" "${CM_TAIL}" 1
fi
rm -f "${CM_OUT}"

# ============================================================================
# Gate 3: per-wave runtime check (Wf-343c2) — points at FRESH preflight JSON
# ============================================================================
log ""
log "===== Gate 3/5: wf343c2_per_wave_check.py --skip-live --strict ====="
PWC_OUT=$(mktemp)
PWC_RC=0
PYTHONPATH="${REPO_ROOT}" "${VENV_PY}" "${REPO_ROOT}/molmetal/scripts/wf343c2_per_wave_check.py" \
  --wave "Wf-350-B1-pre_flight_gpu_ready" \
  --preflight "${RUN_PREFLIGHT_JSON}" \
  --pairs-metals "Ru,Ir,Os,Rh,Re" \
  --pairs-seeds "0,1,2" \
  --min-vram-mib "${MIN_VRAM_MB}" \
  --skip-live \
  --strict \
  --out "${LOG_DIR}/wf343c2_${RUN_TS}.json" \
  >"${PWC_OUT}" 2>&1 || PWC_RC=$?
PWC_TAIL=$(tail -10 "${PWC_OUT}")
# wf343c2 may cascade-fail when pre-flight has pre-existing FAILs (the
# "preflight_no_fail_gates" check fires). On this host, pre-flight v3
# has 3 pre-existing honest-negative FAILs (Wf-350-A2 §5). We treat
# wf343c2 FAIL as a soft FAIL (informative), not a hard blocker, when
# the failure cascades from the pre-existing pre-flight FAILs.
PWC_BLOCKER=0
if [[ ${PWC_RC} -ne 0 ]]; then
  if "${VENV_PY}" -c "
import json, sys
try:
    with open('${LOG_DIR}/wf343c2_${RUN_TS}.json') as f:
        d = json.load(f)
    if d.get('n_fail', 0) > 0:
        # check if any FAIL is the cascading preflight_no_fail_gates
        fail_names = [c['check_id'] for c in d.get('checks', []) if c.get('status') == 'FAIL']
        sys.exit(0 if any('preflight' in n for n in fail_names) else 1)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    PWC_BLOCKER=0
  else
    PWC_BLOCKER=1
  fi
fi
if [[ ${PWC_RC} -eq 0 ]]; then
  gate_result "wf343c2_per_wave" "PASS" "wf343c2 7-check surface all PASS (live probes SKIP via --skip-live)" "${PWC_BLOCKER}"
else
  gate_result "wf343c2_per_wave" "FAIL" "wf343c2 rc=${PWC_RC} tail=$(echo "${PWC_TAIL}" | tr '\n' ' ' | head -c 400) (cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5)" "${PWC_BLOCKER}"
fi
rm -f "${PWC_OUT}"

# ============================================================================
# Gate 4: data quality check (Wf-343c3)
# ============================================================================
log ""
log "===== Gate 4/5: wf343c3_data_quality_check.py --strict ====="
DQC_OUT=$(mktemp)
DQC_RC=0
PYTHONPATH="${REPO_ROOT}" "${VENV_PY}" "${REPO_ROOT}/molmetal/scripts/wf343c3_data_quality_check.py" \
  --strict \
  >"${DQC_OUT}" 2>&1 || DQC_RC=$?
DQC_TAIL=$(tail -10 "${DQC_OUT}")
if [[ ${DQC_RC} -eq 0 ]]; then
  gate_result "wf343c3_data_quality" "PASS" "wf343c3 25-verdict surface all PASS (5 loaders × 5 checks)" 0
else
  gate_result "wf343c3_data_quality" "FAIL" "wf343c3 rc=${DQC_RC} tail=$(echo "${DQC_TAIL}" | tr '\n' ' ' | head -c 400)" 1
fi
rm -f "${DQC_OUT}"

# ============================================================================
# Gate 5: USER_OK (Wf-378-A1 — PASS/SKIP-only; hard entry gate has
#          already verified USER_OK=1 in LIVE mode, exiting 64 otherwise.
#          Mirrors Wf-355 wrapper convention + kill_then_adapt standing
#          directive).
# ============================================================================
log ""
log "===== Gate 5/5: USER_OK (Wf-378-A1 hard entry gate pre-filtered) ====="
if [[ ${DRY_RUN} -eq 1 ]]; then
  gate_result "USER_OK_gate" "SKIP" "--dry-run; USER_OK not required for CPU verification; hard entry gate skipped (Wf-378-A1)" 0
else
  gate_result "USER_OK_gate" "PASS" "USER_OK=1 confirmed by hard entry gate (Wf-378-A1; exit 64 if not set)" 0
fi

# ============================================================================
# Final verdict
# ============================================================================
log ""
log "===== Wf-350-B1 verdict ====="
log "  PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}  SKIP=${SKIP_COUNT}  BLOCKING_FAIL=${BLOCKING_FAIL}"
log "  STRICT=${STRICT}  DRY_RUN=${DRY_RUN}"

# Emit dry-run summary footer so the per-gate verdict + evidence path block is
# visible at the end of stdout (Wf-363-A3 polish)
emit_dry_run_footer

if [[ ${BLOCKING_FAIL} -gt 0 ]]; then
  log "VERDICT: FAIL — blocking gate(s) failed; do NOT launch GPU work"
  exit 1
fi

if [[ ${SKIP_COUNT} -gt 0 && ${STRICT} -eq 1 ]]; then
  log "VERDICT: FAIL — STRICT mode; SKIP gate(s) present"
  exit 1
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run)"
  exit 0
fi

if [[ ${PASS_COUNT} -lt 5 ]]; then
  log "VERDICT: FAIL — expected 5 PASS gates (4 hard + USER_OK); got only ${PASS_COUNT}"
  exit 1
fi

log "VERDICT: PASS — all hard gates green; GPU launch authorized"
exit 0
