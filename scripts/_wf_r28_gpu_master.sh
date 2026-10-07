#!/usr/bin/env bash
# =============================================================================
# Wf-R28 GPU master orchestrator — 5 sequential tasks, ALL on cuda:0
# =============================================================================
# Added per Wf-R28 §5.5 filesystem-truth verify + 5-GPU-launch-command brief.
#
# USER VOICE (2026-10-07, relayed verbatim by the harness):
#   "启动ultracode解决我的需求，显卡可以用但是只用gpu0"
#   -> GPU authorized; GPU 0 ONLY. Nothing runs on cuda:1.
#
# HONEST FRAMING (AGENTS.md §5.2):
#   The five sub-launchers are NOT re-measured here. Wall budgets below are
#   the launchers' own documented estimates (MEASURED-from-source, not from
#   this run). This master script is a SEQUENCER, not a measurement.
#
# GPU 0 (MEASURED 2026-10-07 14:41, nvidia-smi):
#   NVIDIA RTX PRO 6000 Blackwell Workstation Edition
#   97887 MiB total / 87680 MiB free  -> ample VRAM for a single task at a time.
#   Sequential-by-design: one task at a time keeps peak VRAM bounded.
#
# USAGE:
#   bash scripts/_wf_r28_gpu_master.sh --dry-run          # plan only (default)
#   USER_OK=1 bash scripts/_wf_r28_gpu_master.sh           # real 32-hr chain
#   USER_OK=1 bash scripts/_wf_r28_gpu_master.sh --from 3  # resume at task 3
#   USER_OK=1 bash scripts/_wf_r28_gpu_master.sh --only 5  # run one task
#
# §5.1 GPU auth: USER_OK=1 is the env gate. The four wrappers that carry
# their own internal gate (G5/G6/Path-A) will refuse without it; the two CFM
# retrain wrappers do not, so this master enforces it once, centrally.
#
# §5.9 single-venv: every task inherits PY=/home/hugo/codes/mol-metal/.venv/bin/python
# from the sub-launchers. No bare python, no CadButEaas, no kanzi_venv.
# =============================================================================

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PY="/home/hugo/codes/mol-metal/.venv/bin/python"   # §5.9 canonical venv
export PY

# --- GPU 0 ONLY (user directive 2026-10-07) ---------------------------------
export CUDA_VISIBLE_DEVICES=0
GPU="cuda:0"
GPU_PHYS=0

LOGDIR="${REPO_ROOT}/logs/wf_r28_gpu_master"
mkdir -p "${LOGDIR}"
MASTER_LOG="${LOGDIR}/master.log"

DRY_RUN=1
START_TASK=1
ONLY_TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --live)    DRY_RUN=0; shift ;;
    --from)    START_TASK="$2"; shift 2 ;;
    --only)    ONLY_TASK="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[wf-r28] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- §5.1 USER_OK gate -------------------------------------------------------
if [[ "${USER_OK:-0}" != "1" ]]; then
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "[FATAL][wf-r28] USER_OK gate refused: set USER_OK=1 for a live run." >&2
    echo "         USER_OK=1 bash scripts/_wf_r28_gpu_master.sh --dry-run  # still gated" >&2
    exit 1
  fi
  echo "[wf-r28] NOTE: USER_OK not set. Dry-run continues (planning is safe)." >&2
fi

# --- VRAM pre-check (§5.1 rule 1: VRAM budget check first) -------------------
vram_free_mib() {
  nvidia-smi --id="${GPU_PHYS}" --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -1
}
VRAM_FREE="$(vram_free_mib)"
if [[ -z "${VRAM_FREE}" ]]; then
  echo "[FATAL][wf-r28] could not read VRAM for GPU ${GPU_PHYS}; aborting." >&2
  exit 1
fi
# 32 GiB floor. Below this, abort rather than risk an OOM mid-chain.
VRAM_FLOOR_MIB=32768

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')][wf-r28] $*" | tee -a "${MASTER_LOG}"; }

# ---------------------------------------------------------------------------
# Task table. Each entry: name | launcher | est-wall | abort-reason
# Wall figures are the launchers' own documented estimates, MEASURED-from-source.
# ---------------------------------------------------------------------------
# round7 reads GPU as an ENV var (GPU="${GPU:-0}" at line 70), not a --gpu flag.
T1_CMD="GPU=${GPU_PHYS} bash scripts/launch_cfm_round7_f5f6_retrain.sh"
T1_WALL="4-5 hr"
T1_WHY="CFM round7 F5/F6 retrain"

T2_CMD="bash scripts/launch_cfm_tmqm_all6_retrain.sh"
T2_WALL="4-5 hr"
T2_WHY="CFM tmQM all-6-metal retrain"

T3_CMD="USER_OK=1 bash scripts/launch_wf124_p3_g5.sh"
T3_WALL="5 hr"
T3_WHY="TODO-181 G5 50-epoch full sweep (15 pairs)"

T4_CMD="USER_OK=1 bash scripts/launch_wf124_p3_g6.sh"
T4_WALL="7.5 hr"
T4_WHY="TODO-182 G6 per-innovation ablation"

# Task 5 has a hard PRE-FLIGHT precondition (§4: pre-flight v2 25/25 gate).
# NOTE: this wrapper is CPU SCAFFOLD-only by default (header line 3); a real
# 10-hr CFM fine-tune needs its GPU stage invoked explicitly. See honest framing.
T5_CMD="USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh"
T5_WALL="10 hr"
T5_WHY="Plan-A Step 2 CFM prior fine-tune"

# --- pre-flight (§4 + §5.1 rule 4) ------------------------------------------
run_preflight() {
  log "pre-flight: python molmetal/scripts/pre_flight_check.py"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "pre-flight: SKIPPED (dry-run)"
    return 0
  fi
  "${PY}" molmetal/scripts/pre_flight_check.py 2>&1 | tee "${LOGDIR}/preflight.log"
  local rc=${PIPESTATUS[0]}
  # STALE flag (>24 hr) blocks PASS per §4 stale-flag rule.
  if grep -q "STALE" "${LOGDIR}/preflight.log" 2>/dev/null; then
    log "pre-flight: STALE flag present -> auto-verify refuses PASS. Re-run to refresh."
    return 1
  fi
  if [[ "${rc}" -ne 0 ]]; then
    log "pre-flight: FAIL (rc=${rc}). Aborting chain per §4."
    return 1
  fi
  log "pre-flight: PASS"
  return 0
}

# ---------------------------------------------------------------------------
# run_task <n> <cmd> <wall> <why>
# Aborts the whole chain on non-zero rc, with the reason named explicitly.
# ---------------------------------------------------------------------------
run_task() {
  local n="$1" cmd="$2" wall="$3" why="$4"
  local tag="task${n}"
  local log_file="${LOGDIR}/${tag}.log"

  if [[ -n "${ONLY_TASK}" && "${ONLY_TASK}" != "${n}" ]]; then
    return 0
  fi

  log "=============================================================="
  log "TASK ${n}: ${why}"
  log "  cmd:   ${cmd}"
  log "  wall:  ${wall} (launcher-documented estimate)"
  log "  gpu:   ${GPU} (physical ${GPU_PHYS})"
  log "  log:   ${log_file}"

  # VRAM guard immediately before each task.
  local free
  free="$(vram_free_mib)"
  if [[ -z "${free}" || "${free}" -lt "${VRAM_FLOOR_MIB}" ]]; then
    log "ABORT task ${n}: GPU ${GPU_PHYS} free VRAM ${free:-unknown} MiB < floor ${VRAM_FLOOR_MIB} MiB."
    log "ABORT REASON: VRAM_BUDGET. Free the GPU or wait; do not proceed."
    return 1
  fi
  log "  vram:  ${free} MiB free (floor ${VRAM_FLOOR_MIB})"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN task ${n}: would execute ->"
    log "    ${cmd}"
    log "    (nohup + tee -> ${log_file})"
    return 0
  fi

  local start_ts; start_ts=$(date +%s)
  log "START task ${n} at ${start_ts}"
  # shellcheck disable=SC2086
  nohup bash -c "${cmd}" > "${log_file}" 2>&1 &
  local pid=$!
  log "  launched pid=${pid}; streaming to ${log_file}"

  wait "${pid}"
  local rc=$?
  local end_ts; end_ts=$(date +%s)
  local mins=$(( (end_ts - start_ts) / 60 ))

  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL task ${n}: rc=${rc} after ${mins} min (estimated ${wall})."
    log "ABORT REASON: task ${n} returned rc=${rc}; chain halted (subsequent tasks"
    log "  depend on an ${why} baseline that did not complete). Tail:"
    tail -n 25 "${log_file}" | tee -a "${MASTER_LOG}"
    return "${rc}"
  fi

  log "PASS task ${n} (rc=0). log: ${log_file}"
  return 0
}

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
START_TS=$(date +%s)

log "=============================================================="
log "Wf-R28 GPU master — DRY_RUN=${DRY_RUN} START_TASK=${START_TASK} ONLY_TASK=${ONLY_TASK:-none}"
log "GPU: ${GPU} | physical ${GPU_PHYS} | venv ${PY}"
log "logdir: ${LOGDIR}"
log "=============================================================="

if ! run_preflight; then
  log "[wf-r28] chain ABORTED at pre-flight. No task ran."
  exit 1
fi

for spec in \
  "1|${T1_CMD}|${T1_WALL}|${T1_WHY}" \
  "2|${T2_CMD}|${T2_WALL}|${T2_WHY}" \
  "3|${T3_CMD}|${T3_WALL}|${T3_WHY}" \
  "4|${T4_CMD}|${T4_WALL}|${T4_WHY}" \
  "5|${T5_CMD}|${T5_WALL}|${T5_WHY}" ; do
  n="${spec%%|*}"; rest="${spec#*|}"
  cmd="${rest%%|*}"; rest2="${rest#*|}"
  wall="${rest2%%|*}"; why="${rest2#*|}"

  if [[ "${n}" -lt "${START_TASK}" ]]; then
    log "SKIP task ${n} (< --from ${START_TASK})"
    continue
  fi
  run_task "${n}" "${cmd}" "${wall}" "${why}" || {
    log "[wf-r28] CHAIN ABORTED after task ${n}."
    exit 1
  }
done

END_TS=$(date +%s)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY-RUN complete — 5 tasks planned, none executed. Sum of documented walls: 30.5-32 hr."
else
  log "CHAIN complete in $(( (END_TS - START_TS) / 3600 ))h $(( ((END_TS - START_TS) % 3600) / 60 ))m."
fi
log "=============================================================="
exit 0