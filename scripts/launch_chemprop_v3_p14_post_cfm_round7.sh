#!/usr/bin/env bash
# Stage 4 (post-cfm_round7) — chemprop_v3 P1.4 LOMO re-run with fixed wrapper
#
# Why this exists (2026-10-08):
#   The Item 4 chain (launch_item4_chain_post_wf131.sh) Stage 2 ran
#   chemprop_v3 P1.4 LOMO at 09:31 CST but exited with a FALSE rc=0 because
#   of two wrapper bugs:
#     (A) ABLATION_BASE_ARGS included `--user-ok "REQUIRED"` — a literal that
#         train_chemprop_v2_v3.py has no flag for, causing argparse error rc=2.
#     (B) `${PY} trainer | tee log || { exit 1 }` catches tee's rc, not the
#         trainer's; with pipefail disabled (to avoid SIGPIPE), the rc=2 was
#         masked and the wrapper moved on as if all 4 ablations had run.
#   Result: results/wf119_p1_4_chemprop_v3_lomo/{4 ablations}/seed0/ contains
#   only the argparse error in train.log — no metrics.json, no checkpoints.
#
#   Both bugs are now fixed in scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh
#   (see git diff for 2026-10-08). This Stage 4 wrapper waits for the
#   cfm_round7 wrapper to exit (which currently occupies GPU 0), then re-runs
#   the fixed chemprop_v3 P1.4 wrapper.
#
# Wall PROJECTION:
#   - chemprop_v3 P1.4: ~2.0 hr (Wave-A: 4 ablations × 1 seed × 3 ep smoke)
#   - cfm_round7 still in flight (PID 1164804, started 09:31 CST, ~2 hr)
#
# Usage:
#   nohup bash scripts/launch_chemprop_v3_p14_post_cfm_round7.sh \
#     > /home/hugo/codes/mol-metal/molmetal/reports/item4_chain/stage4_chemprop_v3.log 2>&1 &
#
# Cross-refs:
#   - scripts/launch_item4_chain_post_wf131.sh (the chain that already ran
#     Stage 2 with the bug + Stage 3 cfm_round7 currently in flight)
#   - scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh (the FIXED wrapper)
#   - scripts/launch_cfm_round7_f5f6_retrain.sh (the cfm_round7 wrapper to wait for)
#   - TODO/completed/wf_realdata_gapfix_v1_m8pb_ship.md (sibling fix)

set -eu
# NB: pipefail intentionally DISABLED per Wf-131 P3 post-mortem 2026-10-08

REPO_ROOT="/home/hugo/codes/mol-metal"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/molmetal/reports/item4_chain"
mkdir -p "${LOG_DIR}"
STAGE4_LOG="${LOG_DIR}/stage4_chemprop_v3.log"

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "${STAGE4_LOG}"
}

# -----------------------------------------------------------------------------
# Stage 4.0 — Wait for cfm_round7 wrapper to exit
# -----------------------------------------------------------------------------
CFM_WRAPPER_PATTERN="launch_cfm_round7_f5f6_retrain.sh"
log "=== Stage 4 START — waiting for cfm_round7 wrapper (${CFM_WRAPPER_PATTERN}) to exit ==="
while pgrep -af "${CFM_WRAPPER_PATTERN}" >/dev/null 2>&1; do
  # Prefer per_epoch_metrics.json (end-of-train summary); fall back to train.log last epoch
  # Wf-A269 wave-11 (2026-11-02) - REPAIR (FINDING C2-7): the stage-4 epoch
  # probe read run/metrics.json, which does NOT exist on disk, and its pattern
  # '"epoch":[0-9]+' does not match the sibling that does exist,
  # run/per_epoch_metrics.json, which writes '"epoch": N' WITH the JSON indent
  # space -- so the probe could never match.  MEASURED 2026-11-02 on disk: the
  # unspaced pattern matches 0 lines of per_epoch_metrics.json, the spaced one
  # matches 6; run/metrics.json does not exist at all.
  # The run artifacts under molmetal/reports/**/ are git-ignored
  # (.gitignore:146 molmetal/reports/**/train.log, .gitignore:158
  # molmetal/reports/**/*.json), so on a clean checkout BOTH files are absent
  # and the probe degrades to the train.log fallback below rather than to "?".
  # RESIDUAL (reported to the wave owner, 2026-11-02, NOT changed here): the
  # `tail -1` below selects the LAST "epoch": N key in the document, and in
  # this artifact that is best_bond.epoch = 0 while the last epoch TRAINED was
  # 3 -- so the log line prints the best epoch, not the last one (the
  # train.log fallback reports 3).  Left as-is on purpose: changing the epoch
  # SELECTION is outside this repair, whose finding was the dead PATTERN.
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   # Prefer metrics.json (end-of-train summary); fall back to train.log last epoch
  #   epoch_n=$(grep -oE '"epoch":[0-9]+' "${REPO_ROOT}/molmetal/reports/cfm_round7_f5f6_retrain/run/metrics.json" 2>/dev/null | tail -1)
  epoch_n=$(grep -oE '"epoch":[[:space:]]*[0-9]+' "${REPO_ROOT}/molmetal/reports/cfm_round7_f5f6_retrain/run/per_epoch_metrics.json" 2>/dev/null | tail -1)
  if [[ -z "${epoch_n}" ]]; then
    epoch_n=$(grep -oE '\[cfm-bond\] epoch=[0-9]+' "${REPO_ROOT}/molmetal/reports/cfm_round7_f5f6_retrain/run/train.log" 2>/dev/null | tail -1 | sed 's/.*epoch=//')
  fi
  epoch_n=${epoch_n:-"?"}
  log "  cfm_round7 still alive — last_epoch=${epoch_n}/20 — sleeping 60 s"
  sleep 60
done
log "=== cfm_round7 wrapper EXITED — proceeding to chemprop_v3 P1.4 re-run ==="

# -----------------------------------------------------------------------------
# Stage 4.1 — VRAM check
# -----------------------------------------------------------------------------
log "Stage 4.1: VRAM check (must have >= 4 GB free on GPU 0)"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "  GPU 0 free VRAM: ${FREE_MIB} MiB"
if [[ ${FREE_MIB} -lt 4096 ]]; then
  log "FATAL: GPU 0 free VRAM ${FREE_MIB} < 4096 MiB; aborting Stage 4 (OOM risk)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Stage 4.2 — chemprop_v3 P1.4 LOMO re-run (with FIXED wrapper)
# -----------------------------------------------------------------------------
log "=== Stage 4.2: chemprop_v3 P1.4 LOMO RE-RUN START (fixed wrapper) ==="
# Clear stale broken outputs from the false-positive Stage 2 so the wrapper's
# resume-safety / aggregation doesn't see half-trained dirs
log "  Clearing stale broken outputs from previous false-positive Stage 2"
for ablation in baseline v3_fix_only min_epochs_only lomo_init_scale_only; do
  ablation_dir="${REPO_ROOT}/results/wf119_p1_4_chemprop_v3_lomo/${ablation}/seed0"
  if [[ -d "${ablation_dir}" ]] && [[ ! -s "${ablation_dir}/metrics.json" ]]; then
    log "    rm -rf ${ablation_dir} (no metrics.json from broken Stage 2)"
    rm -rf "${ablation_dir}"
  fi
done

USER_OK=1 nohup bash "${REPO_ROOT}/scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh" \
  > "${LOG_DIR}/stage4_chemprop_v3_v2.log" 2>&1
RC_CHEMPROP=$?
log "=== Stage 4.2 DONE: chemprop_v3 P1.4 rc=${RC_CHEMPROP} ==="

log "=== Stage 4 COMPLETE (chemprop_v3 rc=${RC_CHEMPROP}) ==="
exit ${RC_CHEMPROP}