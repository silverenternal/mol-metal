#!/usr/bin/env bash
# Item 4 GPU chain — chemprop_v3 P1.4 → cfm_round7 F5/F6
# Triggers automatically when wrapper v4 (Wf-131 P3 5-metal LOMO) exits.
#
# Why this exists (2026-10-08):
#   Wrapper v4 is currently running 25 (metal, seed) pairs sequentially on
#   GPU 0. Both Item 4 launchers (chemprop_v3 LOMO + cfm_round7 retrain) need
#   exclusive GPU 0 access. Rather than launch them concurrently and risk
#   VRAM contention + DataLoader SIGPIPE collisions, this chain wrapper
#   blocks until wrapper v4 exits, then runs Item 4 sequentially.
#
# Wall PROJECTION (per wrapper headers + Wf-260 G6 timing):
#   - chemprop_v3 P1.4:  ~2.0 hr (Wave-A: 4 ablations × 1 seed × 3 ep smoke)
#   - cfm_round7 F5/F6:  ~1.5 hr (20 epochs, single seed, GPU 0)
#   - TOTAL sequential:  ~3.5 hr GPU 0
#
# USER_OK gate: per AGENTS.md §5.1 standing GPU grant (2026-09-29 verbatim
# "显存够就放心大胆用"), no per-launch user OK needed. Each child wrapper
# has its own USER_OK gate inside, which we satisfy with USER_OK=1 inherited.
#
# Honest framing (party-twenty-spirit §3.5):
#   - PROJECTED wall times; actual may vary 1.5-2x depending on data shape.
#   - chemprop_v3 P1.4 smoke is a 3-epoch ablation, NOT a full 50-ep retrain.
#     Full retrain deferred per Wf-300C Wave-A plan.
#   - cfm_round7 is a retrain of cfm_round3_retrain/run with F5+F6 fixes
#     applied; expected outcome: fm_term > 0, decode_ratio may improve.
#   - GPU 0 currently held by wrapper v4 (~20 min remaining). VRAM check
#     before each launch ensures no OOM risk.
#
# Per Wf-347-B1 + Wf-431-B1: each child wrapper runs its own pre-flight +
# per_wave + wave_budget gates. This script does NOT bypass any of them.
#
# Usage:
#   nohup bash scripts/launch_item4_chain_post_wf131.sh \
#     > /home/hugo/codes/mol-metal/molmetal/reports/item4_chain.log 2>&1 &
#
# Cross-refs:
#   - scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh (chemprop_v3 launcher)
#   - scripts/launch_cfm_round7_f5f6_retrain.sh (cfm_round7 launcher)
#   - scripts/launch_wf131_p3_pt_4cells.sh (wrapper v4; the trigger)
#   - TODO/completed/wf131_p3_wrapper_v4_silent_death_fix.md

set -eu
# NB: pipefail intentionally DISABLED per Wf-131 P3 post-mortem 2026-10-08
# (Python DataLoader SIGPIPE on ${PY} trainer | tee log pipes).

REPO_ROOT="/home/hugo/codes/mol-metal"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/molmetal/reports/item4_chain"
mkdir -p "${LOG_DIR}"
CHAIN_LOG="${LOG_DIR}/chain.log"

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "${CHAIN_LOG}"
}

# -----------------------------------------------------------------------------
# Stage 0 — Wait for wrapper v4 (Wf-131 P3 5-metal LOMO) to exit
# -----------------------------------------------------------------------------
WRAPPER_PATTERN="launch_wf131_p3_pt_4cells.sh"
log "=== Item 4 chain START — waiting for wrapper v4 (${WRAPPER_PATTERN}) to exit ==="
while pgrep -af "${WRAPPER_PATTERN}" >/dev/null 2>&1; do
  done_n=$(find "${REPO_ROOT}/molmetal/reports/wf131_p3_pt_4cells" -name ".DONE" 2>/dev/null | wc -l)
  log "  wrapper v4 still alive — .DONE=${done_n}/25 — sleeping 60 s"
  sleep 60
done
log "=== wrapper v4 EXITED — proceeding to Item 4 chain ==="

# -----------------------------------------------------------------------------
# Stage 1 — VRAM check (RTX PRO 6000 Blackwell sm_120: 98 GB total)
# -----------------------------------------------------------------------------
log "Stage 1: VRAM check (must have >= 4 GB free on GPU 0)"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "  GPU 0 free VRAM: ${FREE_MIB} MiB"
if [[ ${FREE_MIB} -lt 4096 ]]; then
  log "FATAL: GPU 0 free VRAM ${FREE_MIB} < 4096 MiB; aborting chain (OOM risk)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Stage 2 — chemprop_v3 P1.4 LOMO (~2.0 hr GPU 0)
# -----------------------------------------------------------------------------
log "=== Stage 2: chemprop_v3 P1.4 LOMO START ==="
USER_OK=1 nohup bash "${REPO_ROOT}/scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh" \
  > "${LOG_DIR}/wf119_p14_chemprop_v3.log" 2>&1
RC_CHEMPROP=$?
log "=== Stage 2 DONE: chemprop_v3 rc=${RC_CHEMPROP} ==="

if [[ ${RC_CHEMPROP} -ne 0 ]]; then
  log "WARN: chemprop_v3 exited rc=${RC_CHEMPROP}; skipping cfm_round7 to avoid stacking regressions"
  exit ${RC_CHEMPROP}
fi

# -----------------------------------------------------------------------------
# Stage 3 — VRAM re-check + cfm_round7 F5/F6 retrain (~1.5 hr GPU 0)
# -----------------------------------------------------------------------------
log "Stage 3: VRAM re-check before cfm_round7"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "  GPU 0 free VRAM: ${FREE_MIB} MiB"
if [[ ${FREE_MIB} -lt 4096 ]]; then
  log "FATAL: GPU 0 free VRAM ${FREE_MIB} < 4096 MiB; aborting chain before cfm_round7"
  exit 1
fi

log "=== Stage 3: cfm_round7 F5/F6 retrain START ==="
USER_OK=1 nohup bash "${REPO_ROOT}/scripts/launch_cfm_round7_f5f6_retrain.sh" \
  > "${LOG_DIR}/cfm_round7_f5f6.log" 2>&1
RC_CFM=$?
log "=== Stage 3 DONE: cfm_round7 rc=${RC_CFM} ==="

log "=== Item 4 chain COMPLETE (chemprop_v3 rc=${RC_CHEMPROP}, cfm_round7 rc=${RC_CFM}) ==="
exit $(( RC_CHEMPROP + RC_CFM ))
