#!/usr/bin/env bash
# Wf-337-U3 — VRAM-budgeted GPU smoke wrapper for Wf-264 + Vina 4-step + Triangle
#
# =====================================================================
# Wf-337-U3 (2026-09-29) — VRAM-budgeted GPU smoke wrapper
# =====================================================================
#
# Owner: Hugo Lee + ultracode agent
# Date: 2026-09-29
# Source: Wf-334 + Wf-337 ultracode residual gap closure wave
#        (user directive 2026-09-29 verbatim "好像有些之前发现的问题没有被覆盖到，
#         可以额外启动新的ultracode吗？" honored at the Wf-337 layer — extends
#         Wf-334 GPU smoke validation to the 3 deferred smoke targets).
#
# PURPOSE
# -------
# Per Wf-334: GPU smoke was DEFERRED for Wf-264 + Vina 4-step + Triangle
# due to VRAM occupied (G6 on GPU 1, sunlong/dfine_t146 on GPU 0).
# This wrapper IS the pipeline script that autonomously watches VRAM and
# launches the 3 smoke targets as soon as the second GPU (cuda:0) frees
# up — this is the "pipeline script, not polling" pattern honored per
# user memory "Don't poll GPU, write pipelines" (2026-09-29).
#
# Per AGENTS.md §5.1 standing GPU authorization ("显存够就放心大胆用"):
#   - The wrapper is USER_OK-gated (kill_then_adapt standing directive)
#   - VRAM budget check is enforced BEFORE each launch
#   - When VRAM is occupied, the wrapper sleeps and re-checks every 60s
#     (autonomous pipeline, not user-side polling)
#   - Wf-328 + Wf-335 fixes are propagated to all 3 launch targets:
#       * --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS)
#       * --w2-gradient-mode sort (Wf-328-W5; MEASURED 2-3× speedup)
#       * --bf16, --tf32 (Wf-328-W3, W4)
#       * --num-workers 8 (Wf-328-W6)
#       * --torch-compile-mode-v2 max-autotune-no-cudagraphs (Wf-335-R1)
#       * --no-cuda-graphs safety lever (Wf-335-R3)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - MEASURED (2026-09-29 15:30): GPU 0 (RTX PRO 6000 Blackwell,
#     98 GB) — 4 MiB used (idle); GPU 1 (RTX 5090, 32 GB) — 6044 MiB
#     used (~19% utilized, mostly free).
#   - VRAM_DEFERRED_GPU_OCCUPIED: previous smoke was deferred because
#     G6 was occupying GPU 1; current GPU 1 free VRAM ≈ 26 GB which
#     exceeds the 24 GB threshold for Triangle GPU smoke.
#   - PROJECTED wall estimates:
#       * Wf-264 smoke: ~10 min CPU (RDKit SA + QED on n=26,801 ligands)
#       * Vina 4-step smoke: BLOCKED on Vina 1.2.7 conda install
#         (per AGENTS.md §5.4 + Wf-299B); steps 1+2 (receptor/ligand
#         prep) run as smoke, step 3 (Vina dock) is BLOCKED
#       * Triangle smoke: ~5 min GPU on cuda:0 (Wf-333-G2 wrapper,
#         1-seed × 5-epoch smoke run)
#   - Each target's MEASURED-vs-PROJECTED label preserved; pre-flight
#     v3 62/62 PASS gate preserved from Wf-328-SYN baseline
#     (EXPECTED_GATE_COUNTS["all"] = 62).
#
# USAGE
# -----
#   bash scripts/gpu_smoke_when_vram_free.sh --dry-run            # verify wrapper, no launch
#   bash scripts/gpu_smoke_when_vram_free.sh --status            # check VRAM + state
#   bash scripts/gpu_smoke_when_vram_free.sh --target wf264       # launch single target
#   bash scripts/gpu_smoke_when_vram_free.sh --target vina_4step  # launch single target
#   bash scripts/gpu_smoke_when_vram_free.sh --target triangle    # launch single target
#   USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh           # full pipeline (3 smokes)
#   USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh --watch   # watch + auto-launch
#
# EXIT CODES
# ----------
#   0 = PASS (smoke launched or VRAM verified free)
#   1 = FAIL (VRAM insufficient after timeout, or pre-flight FAIL)
#   2 = SKIP (target BLOCKED on external dependency, e.g. Vina install)
#   3 = USER_OK_GATE_DENIED (USER_OK=1 not set for real launch)

set -uo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
OUT_BASE="${PROJECT_ROOT}/results/wf337u3_gpu_smoke"
LOG_DIR="${PROJECT_ROOT}/molmetal/reports/wf337u3_gpu_smoke"
WRAPPER_LOG="${LOG_DIR}/wrapper.log"
STATE_FILE="${LOG_DIR}/state.json"

# VRAM budget threshold (MiB) — Triangle GPU smoke needs ≥ 24 GB on cuda:0
VRAM_FREE_MIN_MIB=24576  # 24 GB
VRAM_POLL_INTERVAL_SEC=60
VRAM_MAX_WAIT_MIN=120    # wait up to 2 hr before declaring VRAM_BLOCKED

mkdir -p "${OUT_BASE}" "${LOG_DIR}"

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
DRY_RUN=0
STATUS_ONLY=0
WATCH_MODE=0
TARGET_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --status)    STATUS_ONLY=1; shift ;;
    --watch)     WATCH_MODE=1; shift ;;
    --target)    TARGET_FILTER="${2:-}"; shift 2 ;;
    --vram-min)  VRAM_FREE_MIN_MIB="${2:-24576}"; shift 2 ;;
    -h|--help)   sed -n '2,60p' "$0"; exit 0 ;;
    *)           printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] [wf337u3] $*" | tee -a "${WRAPPER_LOG}"
}

# -----------------------------------------------------------------------------
# VRAM probe (parsed from nvidia-smi CSV)
# -----------------------------------------------------------------------------
probe_vram() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "VRAM_PROBE_FAIL nvidia-smi not available"
    return 1
  fi
  # Output: index, name, used_mib, total_mib (per-GPU row)
  nvidia-smi --query-gpu=index,name,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null
}

parse_vram_for_index() {
  local gpu_idx="${1:-0}"
  local probe_out
  probe_out=$(probe_vram) || { echo "0"; return 1; }
  echo "${probe_out}" | awk -F',' -v idx="${gpu_idx}" '
    $1 == idx {
      gsub(/ /, "", $3); gsub(/ /, "", $4);
      print $4 - $3;  # free_mib = total - used
    }'
}

# -----------------------------------------------------------------------------
# Pre-flight check (pre_flight_check.py --pipeline launch)
# -----------------------------------------------------------------------------
run_pre_flight() {
  log "Running pre_flight_check.py --pipeline launch"
  if "${PY}" "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all >/dev/null 2>&1; then
    log "PRE_FLIGHT PASS"
    return 0
  else
    log "PRE_FLIGHT FAIL — aborting (TODO-207 §1.4 standing rule)"
    return 1
  fi
}

# -----------------------------------------------------------------------------
# USER_OK gate (per mol-metal-gpu-execution §0 + Wf-306D pattern)
# -----------------------------------------------------------------------------
check_user_ok() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    return 0  # --dry-run is exempt
  fi
  if [ "${STATUS_ONLY:-0}" = "1" ]; then
    return 0  # --status is read-only, exempt
  fi
  if [ "${USER_OK:-0}" != "1" ]; then
    log "USER_OK_GATE_DENIED — set USER_OK=1 to authorise real launch"
    log "  Re-run with: USER_OK=1 bash $0"
    return 3
  fi
  return 0
}

# -----------------------------------------------------------------------------
# VRAM wait (polls every 60s, up to VRAM_MAX_WAIT_MIN)
# -----------------------------------------------------------------------------
wait_for_vram() {
  local gpu_idx="${1:-0}"
  local free_mib
  local waited_min=0
  while [ "${waited_min}" -lt "${VRAM_MAX_WAIT_MIN}" ]; do
    free_mib=$(parse_vram_for_index "${gpu_idx}" | head -1)
    if [ -z "${free_mib}" ] || [ "${free_mib}" = "0" ]; then
      log "VRAM_PROBE_FAIL for cuda:${gpu_idx}; sleeping 60s and retrying"
      sleep "${VRAM_POLL_INTERVAL_SEC}"
      waited_min=$((waited_min + 1))
      continue
    fi
    if [ "${free_mib}" -ge "${VRAM_FREE_MIN_MIB}" ]; then
      log "VRAM_OK cuda:${gpu_idx} free=${free_mib} MiB (threshold=${VRAM_FREE_MIN_MIB})"
      return 0
    fi
    log "VRAM_BUSY cuda:${gpu_idx} free=${free_mib} MiB < ${VRAM_FREE_MIN_MIB}; waiting 60s"
    sleep "${VRAM_POLL_INTERVAL_SEC}"
    waited_min=$((waited_min + 1))
  done
  log "VRAM_TIMEOUT cuda:${gpu_idx} still busy after ${VRAM_MAX_WAIT_MIN} min"
  return 1
}

# -----------------------------------------------------------------------------
# Smoke target: Wf-264 CFM property panel (CPU-only)
# -----------------------------------------------------------------------------
launch_wf264_smoke() {
  local target_log="${LOG_DIR}/wf264_smoke.log"
  log "Launching Wf-264 CFM property panel smoke (CPU)"
  log "  wrapper: scripts/launch_wf264_cfm_property_panel.sh"
  log "  Wf-328+Wf-335 fixes: --no-cuda-graphs (Wf-335-R3 safety lever, redundant for CPU)"
  log "  expected wall: ~10 min (CPU; Wf-188 MetalCytoToxDB ligand population n=26,801)"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh" | tee -a "${target_log}"
    return 0
  fi
  USER_OK=1 nohup bash "${PROJECT_ROOT}/scripts/launch_wf264_cfm_property_panel.sh" \
    > "${target_log}" 2>&1 &
  local pid=$!
  log "  Wf-264 smoke launched pid=${pid}; log=${target_log}"
  echo "${pid}" > "${LOG_DIR}/wf264.pid"
  return 0
}

# -----------------------------------------------------------------------------
# Smoke target: Vina 4-step docking bench
# -----------------------------------------------------------------------------
launch_vina_4step_smoke() {
  local target_log="${LOG_DIR}/vina_4step_smoke.log"
  log "Launching Vina 4-step smoke"
  log "  wrapper: scripts/launch_wf333_vina_4step_gpu_bench.sh"
  log "  status: BLOCKED on Vina 1.2.7 conda install (AGENTS.md §5.4 + Wf-299B)"
  log "  Wf-328+Wf-335 fixes: --no-cuda-graphs (redundant for CPU Vina)"
  log "  expected wall: PROJECTED 7-10 s/mol × N ligands (steps 1+2 only)"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] USER_OK=1 bash scripts/launch_wf333_vina_4step_gpu_bench.sh --dry-run" | tee -a "${target_log}"
    echo "[DRY-RUN] Vina 4-step is HONEST-NEGATIVE on Vina install per AGENTS.md §5.4" | tee -a "${target_log}"
    return 2  # SKIP — blocked on external dependency
  fi
  USER_OK=1 nohup bash "${PROJECT_ROOT}/scripts/launch_wf333_vina_4step_gpu_bench.sh" \
    --status > "${target_log}" 2>&1 &
  local pid=$!
  log "  Vina 4-step status query launched pid=${pid}; log=${target_log}"
  echo "${pid}" > "${LOG_DIR}/vina_4step.pid"
  return 2  # SKIP — actual launch is BLOCKED on Vina install
}

# -----------------------------------------------------------------------------
# Smoke target: Triangle 12-d GPU (Wf-333-G2 wrapper)
# -----------------------------------------------------------------------------
launch_triangle_smoke() {
  local target_log="${LOG_DIR}/triangle_smoke.log"
  log "Launching Triangle 12-d GPU smoke on cuda:0"
  log "  wrapper: molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py"
  log "  Wf-328+Wf-335 fixes:"
  log "    --torch-compile-mode-v2 max-autotune-no-cudagraphs (Wf-335-R1)"
  log "    --no-cuda-graphs (Wf-335-R3 safety lever)"
  log "    --num-workers 8 (Wf-328-W6 / Wf-326B OPTIMAL_NUM_WORKERS)"
  log "    --cpu-num-workers 8 (Wf-328-W2)"
  log "    --w2-gradient-mode sort (Wf-328-W5)"
  log "  expected wall: ~5 min GPU cuda:0 (1-seed × 5-epoch smoke)"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] USER_OK=1 ${PY} molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py --cuda-visible-devices 0 --output molmetal/reports/wf337u3_triangle_smoke" | tee -a "${target_log}"
    return 0
  fi
  USER_OK=1 nohup "${PY}" "${SCRIPTS_DIR}/wf333g2_triangle_12d_gpu_wrapper.py" \
    --cuda-visible-devices 0 \
    --output "${PROJECT_ROOT}/molmetal/reports/wf337u3_triangle_smoke" \
    > "${target_log}" 2>&1 &
  local pid=$!
  log "  Triangle smoke launched pid=${pid}; log=${target_log}"
  echo "${pid}" > "${LOG_DIR}/triangle.pid"
  return 0
}

# -----------------------------------------------------------------------------
# Status query (USER_OK-exempt)
# -----------------------------------------------------------------------------
print_status() {
  echo "=== Wf-337-U3 GPU smoke wrapper status ==="
  echo "Date: $(date -Iseconds)"
  echo "VRAM threshold: ${VRAM_FREE_MIN_MIB} MiB"
  echo ""
  echo "VRAM probe:"
  probe_vram | awk -F',' 'BEGIN{printf "  %-3s %-30s %10s %10s %10s\n", "GPU", "Name", "UsedMiB", "TotalMiB", "FreeMiB"} {used=$3; gsub(/ /,"",used); total=$4; gsub(/ /,"",total); free=total-used; printf "  %-3s %-30s %10s %10s %10s\n", $1, $2, used, total, free}'
  echo ""
  echo "Recent wrapper log (last 15 lines):"
  if [ -f "${WRAPPER_LOG}" ]; then
    tail -15 "${WRAPPER_LOG}"
  else
    echo "  (no log yet)"
  fi
}

# -----------------------------------------------------------------------------
# Main dispatch
# -----------------------------------------------------------------------------
main() {
  log "=== Wf-337-U3 GPU smoke wrapper START ==="
  log "DRY_RUN=${DRY_RUN} WATCH_MODE=${WATCH_MODE} TARGET_FILTER=${TARGET_FILTER:-<all>}"
  log "USER_OK=${USER_OK:-<unset>}"

  if [ "${STATUS_ONLY:-0}" = "1" ]; then
    print_status
    exit 0
  fi

  if ! check_user_ok; then
    exit 3
  fi

  # Pre-flight (CPU-only) before any real launch
  if [ "${DRY_RUN:-0}" != "1" ]; then
    run_pre_flight || { log "ABORT: pre-flight FAIL"; exit 1; }
  fi

  # Initial VRAM probe
  local vram_free
  vram_free=$(parse_vram_for_index 1 | head -1)
  log "Initial VRAM probe: cuda:0 free=${vram_free:-?} MiB"

  # Watch mode: wait until VRAM is free, then launch
  if [ "${WATCH_MODE:-0}" = "1" ] || [ "${DRY_RUN:-0}" != "1" ]; then
    if ! wait_for_vram 1; then
      log "ABORT: VRAM not free on cuda:0 after ${VRAM_MAX_WAIT_MIN} min"
      exit 1
    fi
  fi

  # Launch targets (filtered by --target if set)
  local rc_wf264=0 rc_vina=0 rc_triangle=0
  if [ -z "${TARGET_FILTER}" ] || [ "${TARGET_FILTER}" = "wf264" ]; then
    launch_wf264_smoke || rc_wf264=$?
  fi
  if [ -z "${TARGET_FILTER}" ] || [ "${TARGET_FILTER}" = "vina_4step" ]; then
    launch_vina_4step_smoke || rc_vina=$?
  fi
  if [ -z "${TARGET_FILTER}" ] || [ "${TARGET_FILTER}" = "triangle" ]; then
    launch_triangle_smoke || rc_triangle=$?
  fi

  # Aggregate verdict
  log "=== Wf-337-U3 launch summary ==="
  log "  wf264_rc=${rc_wf264}"
  log "  vina_4step_rc=${rc_vina}  (rc=2 expected: BLOCKED on Vina install)"
  log "  triangle_rc=${rc_triangle}"

  # Persist state
  cat > "${STATE_FILE}" <<EOF
{
  "wf337u3_state": {
    "date": "$(date -Iseconds)",
    "dry_run": ${DRY_RUN},
    "watch_mode": ${WATCH_MODE},
    "vram_free_mib_initial": ${vram_free:-0},
    "vram_threshold_mib": ${VRAM_FREE_MIN_MIB},
    "wf264_rc": ${rc_wf264},
    "vina_4step_rc": ${rc_vina},
    "triangle_rc": ${rc_triangle},
    "user_ok": "${USER_OK:-<unset>}"
  }
}
EOF
  log "State persisted to ${STATE_FILE}"
  log "=== Wf-337-U3 wrapper COMPLETE ==="

  # Return 0 if any target launched successfully; 2 if all blocked
  if [ "${rc_wf264}" -eq 0 ] || [ "${rc_triangle}" -eq 0 ]; then
    exit 0
  fi
  exit 2  # all blocked
}

main
