#!/bin/bash
# scripts/launch_all_training.sh
# Per user voice 2026-10-03: sequential GPU training pipeline on cuda:0
# All training config is yaml-only (per AGENTS.md 5.5 + user directive)
# Pre-flight validation per AGENTS.md 1.4 must PASS before each launch
#
# Usage:
#   USER_OK=1 DRY_RUN=true bash scripts/launch_all_training.sh  # pre-validate only
#   USER_OK=1 bash scripts/launch_all_training.sh              # actual launch
#
# Pre-conditions:
#   - USER_OK=1 must be set
#   - cuda:0 must be free
#   - yaml files in PRIORITY_ORDER below must reference Tier-1 benchmark datasets

set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ "${USER_OK:-0}" != "1" ]]; then
  echo "[FATAL] USER_OK=1 required per AGENTS.md 5.1" >&2
  exit 1
fi

# PRIORITY_ORDER (per Wf-A8 alignment matrix)
P0_LAUNCHES=(
  "configs/cfm_per_innovation_ablation_p1.4.yaml"     # Wf-299 Wave A (m14 + m17 + m3)
  "configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml"  # Wf-125 CONTROLD PAIR (m1 + m21)
  # m2_r2_per_cohort.yaml is a per-cohort FEATURE config (not a trainer launch config) per Wf-A8c closeout note; do NOT launch
)
P1_LAUNCHES=(
  "configs/cfm_full_retrain.yaml"                       # Wf-380 SCAFFOLD-loss (m7 + m4 + m15)
  # lambda_click_physical_smoke.yaml is a SCORING config (not trainer launch config); do NOT launch
)

run_pre_flight_only() {
  local yaml="$1"
  echo "=== [$(date +%H:%M:%S)] PRE-FLIGHT (dry-run) for ${yaml} ==="
  if ! USER_OK=1 bash scripts/start.sh "${yaml}" --fast --dry-run 2>&1 | tail -15; then
    echo "[FAIL] PRE-FLIGHT FAILED for ${yaml} - SKIP per kill_then_adapt 0"
    return 1
  fi
  echo "[OK] PRE-FLIGHT PASSED for ${yaml}"
}

run_one() {
  local yaml="$1"
  local mode="$2"
  if [[ "${mode}" == "pre-flight-only" ]]; then
    run_pre_flight_only "${yaml}"
    return $?
  fi
  echo "=== [$(date +%H:%M:%S)] LAUNCH ${yaml} on cuda:0 (TUI dashboard auto-spawned; see tmux pane) ==="
  # No --fast flag: this runs pre-flight + actual trainer + spawns TUI dashboard (per AGENTS.md 5.5; Wf-TUI)
  USER_OK=1 bash scripts/start.sh "${yaml}" 2>&1 | tail -50
}

if [[ "${DRY_RUN:-true}" == "true" ]]; then
  echo "=== DRY_RUN=true: pre-flight validation only (NO actual launch) ==="
  for yaml in "${P0_LAUNCHES[@]}" "${P1_LAUNCHES[@]}"; do
    run_one "${yaml}" pre-flight-only
  done
else
  echo "=== ACTUAL LAUNCH: SERIAL on cuda:0 ==="
  echo "=== Estimated wall time: ~41 hr (P0 ~17 hr + P1 ~15 hr) ==="
  for yaml in "${P0_LAUNCHES[@]}" "${P1_LAUNCHES[@]}"; do
    run_one "${yaml}" actual
    sleep 30
  done
fi

echo "=== ALL TRAINING DONE ==="
echo "Next: collect results from molmetal/reports/<yaml-name>/master.log"