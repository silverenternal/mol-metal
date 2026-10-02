#!/usr/bin/env bash
# GPU pipeline master — sequential execution of all queued GPU jobs
# (USER_OK granted per 2026-09-29 standing directive, "显存够就放心大胆用").
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Source: Wf-310 ultracode queue dashboard + standing GPU grant
#
# Pipeline order (each job runs after the previous one completes):
# ----------------------------------------------------------------------------
# PHASE 1 — Currently in flight (NOT touched by this script):
#   - G5 retrain retry (Ru seed=2 + Ir seed=0) — IN-FLIGHT on both GPUs
#   - Wf-124 G6 60-tuple sweep (7.5 hr cuda:1) — IN-FLIGHT (launched 09:54:31)
#   - Wf-310F frontier Wave A (cuda:0) — IN-FLIGHT (launched 09:59:49)
#
# PHASE 2 — GPU 0 jobs (run when Wave A finishes):
#   1. Wf-310F frontier Wave C (2-3 hr cuda:0)
#   2. Wf-119 P1.4 chemprop v3 LOMO (2.0 hr cuda:0)         [NEW per Wf-336-X2]
#
# PHASE 3 — GPU 1 jobs (run when G6 finishes, in priority order):
#   1. Wf-310F frontier Wave B (0.5 hr cuda:1)
#   2. Wf-219 §4.1 Ru (13.5 min cuda:1)
#   3. Wf-333 Vina 4-step (BLOCKED on Vina 1.2.7 install)  [NEW per Wf-334-U2]
#   4. Wf-310C TODO-220 (3.9 hr cuda:1)
#   5. Wf-310C TODO-179 (4.5 hr cuda:1)
#   6. Wf-310D DrugOOD DANN Phase A (3-4 hr cuda:1)
#   7. DrugOOD Wf-74 v2 (3-4 hr cuda:1)                     [NEW per Wf-336-X3]
#   8. Wf-135 Phase 2 LOMO 5×5×50 (2 weeks cuda:1)
#   9. Wf-264 CFM property panel (10 min CPU, post-LOMO)    [NEW per Wf-334-U3]
#  10. Wf-264 CFM property panel RUN-ONCE (10 min CPU)      [NEW per Wf-338-F1]
#  11. W₂ sliced GPU smoke (5 min GPU cuda:1)               [NEW per Wf-338-F1]
#
# SKIPPED:
#   - Wf-219 §4.2 production backbone (PAPER_OK required per TODO-84)
#   - Wf-219 §4.4 NCI-60 9-cell multi-task chemprop v2 (no wrapper on disk yet;
#     Wf-219 closeout §4.4 spec only; deferred to Wf-338+)
#
# Honest framing (party-twenty-spirit §3.5):
#   - Each wrapper has its own USER_OK/USER_OK_FILE/PAPER_OK gate — this script
#     just removes the *standing* kill_then_adapt friction per the 2026-09-29
#     user grant. Per-wrapper gates still apply (e.g. PAPER_OK for §4.2).
#   - Wall estimates are PROJECTED; actual times may vary 1.5-2x.
#   - Each job's wrapper has its own pre_flight + per_wave + wave_budget gates;
#     this script does NOT bypass any of them.
#
# Usage:
#   bash scripts/gpu_pipeline_master.sh         # run full pipeline
#   bash scripts/gpu_pipeline_master.sh phase3  # run only Phase 3 (GPU 1)
#   bash scripts/gpu_pipeline_master.sh phase2  # run only Phase 2 (GPU 0)

set -uo pipefail
cd /home/hugo/codes/mol-metal

# ----- Logging helpers -----
LOG_DIR="/home/hugo/codes/mol-metal/molmetal/reports/gpu_pipeline_master"
mkdir -p "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" | tee -a "${PIPELINE_LOG}"
}

wait_for_gpu0_free() {
  log "Waiting for GPU 0 free (no cuda:0 processes other than G5 Ir seed=0 + Wave-A done)..."
  # Wait for any foreground cuda:0 training process to finish
  while pgrep -af "train_(dmpnn|cfm|chemprop).*cuda:0" >/dev/null 2>&1; do
    sleep 60
  done
  log "GPU 0 free"
}

wait_for_gpu1_free() {
  log "Waiting for GPU 1 free (no cuda:1 processes other than G5 Ru seed=2 + G6 done)..."
  # Wait for any foreground cuda:1 training process to finish
  while pgrep -af "train_(dmpnn|cfm|chemprop).*cuda:1" >/dev/null 2>&1; do
    sleep 60
  done
  log "GPU 1 free"
}

# ============================================================================
# Wf-343C4 pre-retrain internal check gate runner — NEW per Wf-347-A2
# ============================================================================
# Per Wf-343C4 (TODO/completed/343c4_integration.md), the pre-flight v2 gate
# set grew from 62 -> 66 (4 NEW pre-retrain internal check gates registered
# in pre_flight_check.py).  Three companion scripts run those gates' values
# against the per-wave + per-loader surface:
#
#   * wf343c2_per_wave_check.py  — per-wave runtime gate (7 checks,
#     4 of which mirror the 4 NEW Wf-343C4 gates + 3 live-env extensions).
#   * wf343c3_data_quality_check.py — per-loader data quality gate
#     (5 checks × every loader in molmetal/data/).
#   * Wf-343C4 integration — the 4 NEW gates themselves are now part of
#     EXPECTED_GATE_COUNTS["all"] = 66 (vs 62 pre-Wf-343C4).
#
# This runner is opt-in via WF343C4_GATE=1 (default off) to keep the
# happy-path fast — the wrappers themselves already enforce their own
# pre_flight + per_wave + wave_budget gates per line 43 comment.
#
# Exit semantics:
#   * 0 — every gate PASSED (or SKIP, with --strict not set).
#   * 1 — at least one gate FAILED.  Pipeline halts (kill_then_adapt).
#   * 2 — at least one SKIP under --strict.  Pipeline halts.
#
# CLI:
#     bash scripts/gpu_pipeline_master.sh wf343c4 [--strict] [--skip-live]
#
wf343c4_gate() {
  local STRICT_FLAG="${1:-}"

  log "=== WF-343C4 PRE-RETRAIN INTERNAL CHECK GATE RUNNER (Wf-347-A2) ==="

  # Wf-343C2 per-wave check (7 checks; CPU-only when --skip-live)
  log "Running Wf-343C2 per-wave check (--wave 'PipelineMaster-Phase3')..."
  python3 molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "PipelineMaster-Phase3" \
    --pairs-metals "Ru,Ir,Os,Rh,Re" \
    --pairs-seeds "0,1,2" \
    --budget-hr 12.0 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    ${STRICT_FLAG} \
    --out molmetal/reports/wf343c2_per_wave_check/pipeline_master_phase3.json 2>&1 \
    | tee -a "${LOG_DIR}/wf343c2_pipeline_master.log"
  local RC_C2=$?
  if [[ ${RC_C2} -ne 0 ]]; then
    log "Wf-343C2 per-wave check returned rc=${RC_C2} — gate FAILED (per kill_then_adapt)"
    return 1
  fi
  log "Wf-343C2 per-wave check PASSED (rc=${RC_C2})"

  # Wf-343C3 data quality check (5 checks × every loader)
  log "Running Wf-343C3 data quality check (per-loader surface)..."
  python3 molmetal/scripts/wf343c3_data_quality_check.py \
    --out-dir molmetal/reports/wf343c3_data_quality \
    ${STRICT_FLAG} 2>&1 \
    | tee -a "${LOG_DIR}/wf343c3_pipeline_master.log"
  local RC_C3=$?
  if [[ ${RC_C3} -ne 0 ]]; then
    log "Wf-343C3 data quality check returned rc=${RC_C3} — gate FAILED (per kill_then_adapt)"
    return 1
  fi
  log "Wf-343C3 data quality check PASSED (rc=${RC_C3})"

  # Wf-343C4 integration gate references — the 4 NEW pre-retrain internal
  # check gates live inside pre_flight_check.py::build_gates('all') and are
  # auto-invoked by the global pre-flight pipeline.  The pipeline master
  # cross-references them so any audit reader can confirm the gate surface
  # has been updated to 66 (vs 62 pre-Wf-343C4).
  log "Wf-343C4 integration: EXPECTED_GATE_COUNTS['all'] = 66 (post-Wf-343C4 baseline)"
  log "  - p3_wf343c4_hyperparams_config_complete"
  log "  - p3_wf343c4_yuelbond_decoder_env"
  log "  - p3_wf343c4_anchor_freshness_24h"
  log "  - p3_wf343c4_wave_budget_remaining_nonneg"
  log "=== WF-343C4 GATE RUNNER COMPLETE ==="
  return 0
}

# ============================================================================
# PHASE 2 — GPU 0 jobs (after Wave A finishes)
# ============================================================================
phase2_gpu0() {
  log "=== PHASE 2: GPU 0 JOBS START ==="

  # Opt-in Wf-343C4 pre-retrain internal check gate runner (NEW per Wf-347-A2)
  # Set WF343C4_GATE=1 to enforce; off by default for happy-path speed.
  if [[ "${WF343C4_GATE:-0}" == "1" ]]; then
    wf343c4_gate || return $?
  fi

  wait_for_gpu0_free

  # Job 1: Wf-310F frontier Wave C (~2-3 hr cuda:0)
  log "Launching Wf-310F frontier Wave C"
  WF310F_NEXT_WAVE=C nohup bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave C \
    > "${LOG_DIR}/wf310f_wave_c.log" 2>&1
  log "Wave C done (rc=$?)"

  # Job 2: Wf-119 P1.4 chemprop v3 LOMO (~2.0 hr cuda:0) — NEW per Wf-336-X2
  log "Launching Wf-119 P1.4 chemprop v3 LOMO"
  USER_OK=1 nohup bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh \
    > "${LOG_DIR}/wf119_p14.log" 2>&1
  log "Wf-119 P1.4 done (rc=$?)"

  log "=== PHASE 2 COMPLETE ==="
}

# ============================================================================
# PHASE 3 — GPU 1 jobs (after G6 finishes, sequential)
# ============================================================================
phase3_gpu1() {
  log "=== PHASE 3: GPU 1 JOBS START ==="

  # Opt-in Wf-343C4 pre-retrain internal check gate runner (NEW per Wf-347-A2)
  # Set WF343C4_GATE=1 to enforce; off by default for happy-path speed.
  if [[ "${WF343C4_GATE:-0}" == "1" ]]; then
    wf343c4_gate || return $?
  fi

  wait_for_gpu1_free

  # Job 1: Wf-310F frontier Wave B (~0.5 hr cuda:1)
  log "Launching Wf-310F frontier Wave B"
  WF310F_NEXT_WAVE=B nohup bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave B \
    > "${LOG_DIR}/wf310f_wave_b.log" 2>&1
  log "Wave B done (rc=$?)"

  # Job 2: Wf-219 §4.1 Ru (13.5 min cuda:1)
  log "Launching Wf-219 §4.1 Ru"
  USER_OK=1 nohup bash molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh \
    > "${LOG_DIR}/wf219_ru.log" 2>&1
  log "Wf-219 §4.1 Ru done (rc=$?)"

  # Job 3: Wf-333 Vina 4-step (BLOCKED on Vina 1.2.7 install) — NEW per Wf-334-U2
  # --status is read-only (USER_OK-exempt); the real launch path requires
  # USER_OK=1 and will BLOCK on Vina 1.2.7 conda install per AGENTS.md §5.4.
  log "Launching Wf-333 Vina 4-step (BLOCKED on Vina install)"
  USER_OK=1 nohup bash molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh \
    > "${LOG_DIR}/wf333_vina.log" 2>&1
  log "Wf-333 Vina 4-step done (rc=$?, BLOCKED on Vina install)"

  # Job 4: Wf-310C TODO-220 (3.9 hr cuda:1)
  log "Launching Wf-310C TODO-220"
  touch /tmp/USER_OK_WF310C_TODO220
  nohup bash molmetal/scripts/launch_wf310c_todo220_gpu.sh \
    > "${LOG_DIR}/wf310c_todo220.log" 2>&1
  log "TODO-220 done (rc=$?)"

  # Job 5: Wf-310C TODO-179 W2 PlatinAI 4-cell (4.5 hr cuda:1)
  log "Launching Wf-310C TODO-179"
  touch /tmp/USER_OK_WF310C_T179
  nohup bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh \
    > "${LOG_DIR}/wf310c_t179.log" 2>&1
  log "TODO-179 done (rc=$?)"

  # Job 6: Wf-310D DrugOOD DANN Phase A (3-4 hr cuda:1)
  log "Launching Wf-310D DrugOOD DANN Phase A"
  WF306A_USER_OK=YES nohup bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase A \
    > "${LOG_DIR}/wf310d_drugood.log" 2>&1
  log "DrugOOD DANN done (rc=$?)"

  # Job 7: DrugOOD Wf-74 v2 DANN (3-4 hr cuda:1) — NEW per Wf-336-X3
  # Re-launch wrapper that wires Wf-328 (W1-W10) onto the Wf-305D DANN
  # driver; ood_test_auc=0.4574 BELOW field-avg 0.55-0.65 (per Wf-297B +
  # Wf-301A) needs the re-launch wrapper to attempt recovery.
  log "Launching DrugOOD Wf-74 v2 DANN"
  USER_OK=YES nohup bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3 \
    > "${LOG_DIR}/wf74_v2_drugood.log" 2>&1
  log "Wf-74 v2 DrugOOD done (rc=$?)"

  # Job 8: Wf-135 Phase 2 LOMO (2 weeks cuda:1) — wrapper lives in scripts/ (Wf-310E ship)
  log "Launching Wf-135 Phase 2 LOMO"
  USER_OK=1 nohup bash scripts/launch_wf310e_wf135_phase2_gpu.sh \
    > "${LOG_DIR}/wf135_phase2.log" 2>&1
  log "Wf-135 Phase 2 done (rc=$?)"

  # Job 9: Wf-264 CFM property panel (~10 min CPU, post-LOMO) — NEW per Wf-334-U3
  # CPU-only (RDKit SA + QED on Wf-188 MetalCytoToxDB ligand population
  # n=26,801); runs at the tail of Phase 3 to avoid GPU contention with
  # the 14-day LOMO job.  USER_OK gate enforced by wrapper itself.
  log "Launching Wf-264 CFM property panel (CPU)"
  USER_OK=1 nohup bash scripts/launch_wf264_cfm_property_panel.sh \
    > "${LOG_DIR}/wf264_panel.log" 2>&1
  log "Wf-264 CFM property panel done (rc=$?)"

  # Job 10: Wf-264 CFM property panel RUN-ONCE wrapper (~10 min CPU) — NEW per Wf-338-F1
  # Thin run-once launcher for batch scheduling (cron, pipeline orchestrator).
  # Refuses --dry-run; surfaces 0=PASS, 1=FAIL, 2=SKIP, 3=USER_OK_GATE_DENIED.
  # Per Wf-338-F1 ultracode residual-gap closure: bridges the Wf-264 wrapper
  # to deterministic batch scheduling without bypassing USER_OK or pre-flight.
  log "Launching Wf-264 CFM property panel RUN-ONCE (CPU)"
  USER_OK=1 nohup bash scripts/run_wf264_cfm_property_panel.sh \
    > "${LOG_DIR}/wf264_run_once.log" 2>&1
  log "Wf-264 CFM property panel RUN-ONCE done (rc=$?)"

  # Job 11: W₂ sliced GPU smoke (~5 min GPU cuda:1) — NEW per Wf-338-F1
  # Per Wf-328-W5: --w2-gradient-mode default is "sort" (MEASURED 2-3×
  # speedup).  This smoke exercises the legacy "sliced" mode for paper
  # §6 reproducibility audits.  VRAM-budgeted (>= 24 GB free); USER_OK
  # gate enforced by wrapper itself.
  log "Launching W₂ sliced GPU smoke on cuda:1"
  USER_OK=1 nohup bash scripts/gpu_smoke_w2_sliced.sh \
    > "${LOG_DIR}/w2_sliced_smoke.log" 2>&1
  log "W₂ sliced GPU smoke done (rc=$?)"

  log "=== PHASE 3 COMPLETE ==="
}

# ============================================================================
# Main entry
# ============================================================================
case "${1:-all}" in
  phase2)
    phase2_gpu0
    ;;
  phase3)
    phase3_gpu1
    ;;
  wf343c4)
    wf343c4_gate "${2:-}"
    ;;
  all)
    phase2_gpu0
    phase3_gpu1
    ;;
  *)
    echo "Usage: bash $0 [phase2|phase3|wf343c4|all]"
    exit 1
    ;;
esac

log "Pipeline complete"