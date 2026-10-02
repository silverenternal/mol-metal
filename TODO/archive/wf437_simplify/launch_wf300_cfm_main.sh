#!/bin/bash
# Wf-300C — TODO-119 P1.4 actual GPU launch sequence (3 waves)
#   Wave-A: P1.4 smoke  (4 ablations × 1 seed × 3 ep) — 2.0 hr on PRO 6000 cuda:0
#   Wave-B: G6 C12 smoke (no_wass × Ru × seed=0) — 10 min on RTX 5090 cuda:1
#   Wave-C: G6 full sweep (4 cfgs × 5 metals × 3 seeds = 60 runs) — 7.5 hr on RTX 5090 cuda:1
#   Total ~9.5 hr sequential.
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Parent: Wf-299C launch script (in flight) + Wf-295D launch orchestration plan
# Refs:
#   - molmetal/reports/wf295d_launch_orchestration/plan.md §1 (3-wave sequence)
#   - molmetal/reports/wf299d_cross_cohort/plan.md §5 (P1.4 sequencing)
#   - TODO/completed/207_unified_pre_flight_checklist_v2.md §1.1 (user OK gate)
#   - configs/cfm_per_innovation_ablation_p1.4.yaml (Wave-A config)
#   - configs/wf124_p3_per_innovation_ablation_g6.yaml (Wave-B/C config)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt standing directive — 207 §1.1)
#   - per_wave_user_ok_required: true (P1.4 yaml §directives + Wf-295D §4.8)
#   - Total MEASURED = 9.5 hr sequential (Wf-295D §1 + §7)
#   - target_test_auc: 0.86 in G6 yaml is PROJECTED, NOT MEASURED
#   - 13 h user budget with --no-enforce-budget (per §7.1.1) — budget is tracked but not enforced
#
# Per Wf-328-W2: Wave-B --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core
# host) — replaces Wf-326B-reverted --cpu-num-workers 0 override. Wave-A CFM trainer
# does not consume --cpu-num-workers (CFM uses EGNN graph loader); Wave-C defers to
# launch_wf124_p3_g6.sh which is also updated by Wf-328-W2.
#
# Usage:
#   bash scripts/launch_wf300_cfm_main.sh --dry-run         # CPU verification only
#   USER_OK=1 bash scripts/launch_wf300_cfm_main.sh         # REAL launch (after user grants)
#   USER_OK=1 SKIP_WAVES=B,C bash scripts/launch_wf300_cfm_main.sh  # Wave-A only

set -e
cd /home/hugo/codes/mol-metal

PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
export PYTHONPATH="/home/hugo/codes/mol-metal:${PYTHONPATH:-}"

# ---- USER OK GATE (P0 BLOCKER per 207 §1.1 + kill_then_adapt standing directive) ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-300C — DRY-RUN (CPU verification only; no GPU execution) ==="
  echo ""
  echo "USER OK REQUIRED for real launch. Set USER_OK=1 after user grants per 207 §1.1."
  echo ""
else
  if [[ "${USER_OK:-}" != "1" ]]; then
    echo "================================================================="
    echo " USER OK REQUIRED — kill_then_adapt standing directive (207 §1.1)"
    echo "================================================================="
    echo ""
    echo "This script launches 3 sequential GPU waves totalling ~9.5 hr:"
    echo "  Wave-A: P1.4 smoke       — 2.0 hr on PRO 6000 cuda:0"
    echo "  Wave-B: G6 C12 no_wass   — 10 min on RTX 5090 cuda:1"
    echo "  Wave-C: G6 full sweep    — 7.5 hr on RTX 5090 cuda:1"
    echo ""
    echo "Before running this script, the user MUST explicitly grant OK per"
    echo "207 §1.1 (user 'kill then adapt' deferral pattern)."
    echo ""
    echo "To grant OK and launch:"
    echo "  USER_OK=1 bash scripts/launch_wf300_cfm_main.sh"
    echo ""
    echo "To launch only Wave-A (cheapest, 2 hr):"
    echo "  USER_OK=1 SKIP_WAVES=B,C bash scripts/launch_wf300_cfm_main.sh"
    echo ""
    echo "ABORTING — no GPU work executed."
    exit 1
  fi
  echo "=== USER OK GRANTED (USER_OK=1) — proceeding with real launch ==="
  echo "Timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"
fi

# ---- Per-wave OK gate selector ----
SKIP_WAVES="${SKIP_WAVES:-}"  # e.g. "C" or "B,C" to skip
want_wave() {
  local w=$1
  [[ -z "${SKIP_WAVES}" ]] && return 0
  [[ ",${SKIP_WAVES}," == *",${w},"* ]] && return 1
  return 0
}

# ---- Wave tool guards (per Wf-295D §4 + Wf-262 / Wf-296D wiring) ----
echo ""
echo "=== WAVE TOOL GUARDS ==="
echo "pre_flight_check.py:"
${PY} molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 | tail -3 || { echo "FAIL: pre_flight_check"; exit 1; }
echo ""
echo "wf343c3_data_quality_check.py (Wf-356-A1; loader-agnostic — all 5 R20 loaders internally):"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf300_cfm_main_data_quality 2>&1 | tail -10 || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
echo ""
echo "wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required:"
${PY} molmetal/scripts/wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required 2>&1 | tail -5 || { echo "FAIL: wave_budget_tracker"; exit 1; }

# ============================================================================
# Wave-A: P1.4 smoke (TODO-119) — 4 ablation configs × 1 seed × 3 epochs
# ============================================================================
WAVE_A_BASE="results/cfm_per_innovation_ablation_p1.4"
WAVE_A_LOG="${WAVE_A_BASE}/master.log"
WAVE_A_SCRIPT="molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
WAVE_A_GPU="cuda:0"  # PRO 6000 Blackwell
WAVE_A_WARM_CKPT="molmetal/checkpoints/cfm_yuelbond_wasserstein.pt"

if want_wave "A"; then
  echo ""
  echo "================================================================="
  echo " Wave-A: TODO-119 P1.4 smoke — 4 ablations × 1 seed × 3 epochs"
  echo "================================================================="
  echo "GPU: PRO 6000 cuda:0 (CUDA_VISIBLE_DEVICES=0)"
  echo "Wall-est: 2.0 hr (4 sub-waves × 0.5 hr each)"
  echo "Warm-start: ${WAVE_A_WARM_CKPT}"
  echo ""

  if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY-RUN: skipping real Wave-A launch"
    echo "Sub-waves planned: a_full_stack, b_yuelbond_only, c_w2_only, d_catherine_only"
    exit 0
  fi

  mkdir -p "${WAVE_A_BASE}"
  echo "=== Wave-A START $(date) ===" | tee "${WAVE_A_LOG}"

  # Per P1.4 yaml §waves + §ablation_configs
  for cfg in a_full_stack b_yuelbond_only c_w2_only d_catherine_only; do
    case "${cfg}" in
      a_full_stack)
        FLAGS="--use-yuelbond --wasserstein-weight 0.1 --closure-bounded"
        ;;
      b_yuelbond_only)
        FLAGS="--use-yuelbond --no-wasserstein --no-closure-bounded"
        ;;
      c_w2_only)
        FLAGS="--no-yuelbond --wasserstein-weight 0.1 --no-closure-bounded"
        ;;
      d_catherine_only)
        FLAGS="--no-yuelbond --no-wasserstein --closure-bounded"
        ;;
    esac
    SUB_LOG="${WAVE_A_BASE}/${cfg}/run.log"
    mkdir -p "${WAVE_A_BASE}/${cfg}/ckpts"
    echo "" | tee -a "${WAVE_A_LOG}"
    echo "=== Wave-A ${cfg} START $(date '+%H:%M:%S') ===" | tee -a "${WAVE_A_LOG}"
    CUDA_VISIBLE_DEVICES=0 ${PY} ${WAVE_A_SCRIPT} \
      --epochs 3 \
      --seed 0 \
      --pocket-id 1h36 \
      --device "${WAVE_A_GPU}" \
      --cuda-visible-devices 0 \
      --out-dir "${WAVE_A_BASE}/${cfg}" \
      --ckpt-dir "${WAVE_A_BASE}/${cfg}/ckpts" \
      --target-decode-ratio 0.5 \
      ${FLAGS} 2>&1 | tee "${SUB_LOG}"
    echo "=== Wave-A ${cfg} END $(date '+%H:%M:%S') ===" | tee -a "${WAVE_A_LOG}"
  done

  echo "=== Wave-A ALL 4 ABLATIONS DONE $(date) ===" | tee -a "${WAVE_A_LOG}"

  # Per-wave OK gate (per Wf-295D §4)
  echo "" | tee -a "${WAVE_A_LOG}"
  echo "=== Wave-A per_wave_checklist ===" | tee -a "${WAVE_A_LOG}"
  ${PY} molmetal/scripts/per_wave_checklist.py --wave "Wave-A Wf-300 P1.4 smoke" \
    --config "configs/cfm_per_innovation_ablation_p1.4.yaml" \
    --out "molmetal/reports/wf300c_launch_sequence/wave_a_checklist.json" \
    --out-md "molmetal/reports/wf300c_launch_sequence/wave_a_checklist.md" 2>&1 | tee -a "${WAVE_A_LOG}" || \
    { echo "FAIL: Wave-A per_wave_checklist"; exit 1; }
else
  echo ""
  echo "=== Wave-A SKIPPED (SKIP_WAVES=${SKIP_WAVES}) ==="
fi

# ============================================================================
# Wave-B: G6 C12 no_wass smoke (TODO-182) — 1 config × 1 metal × 1 seed
# ============================================================================
if want_wave "B"; then
  echo ""
  echo "================================================================="
  echo " Wave-B: G6 C12 no_wass smoke — 1 cfg × Ru × seed=0"
  echo "================================================================="
  echo "GPU: RTX 5090 cuda:1 (CUDA_VISIBLE_DEVICES=1)"
  echo "Wall-est: 10 min"
  echo ""

  if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY-RUN: skipping real Wave-B launch"
    exit 0
  fi

  # Wave-B reuses g6 launcher for the C12 single tuple
  WAVE_B_BASE="results/wf124_p3_per_innovation_ablation_g6/no_wass/Ru/seed0"
  mkdir -p "${WAVE_B_BASE}/ckpts"
  WAVE_B_LOG="${WAVE_B_BASE}/run.log"
  DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

  echo "=== Wave-B START $(date) ===" | tee "${WAVE_B_LOG}"
  CUDA_VISIBLE_DEVICES=1 ${PY} molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
    --metals Ru \
    --seeds 0 \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${WAVE_B_BASE}" \
    --ckpt-dir "${WAVE_B_BASE}/ckpts" \
    --epochs 50 \
    --batch-size 256 \
    --hidden 300 \
    --depth 3 \
    --lr 1e-3 \
    --alpha 0.5 \
    --wasserstein-weight 0.0 \
    --use-l2-guard \
    --ablation-config mtl_baseline \
    --w2-gradient-mode sliced \
    --n-mc 256 \
    --device cuda:0 \
    --cuda-visible-devices 1 \
    --cpu-num-workers 8 \
    --data-csv "${DATA_CSV}" \
    # Wf-335-R3 — CUDA Graphs hard-disable safety lever (default ON
    # in trainer; explicit here for paper §6 audits + Wf-330-B4
    # RuntimeError prevention).
    --no-cuda-graphs 2>&1 | tee -a "${WAVE_B_LOG}"
  echo "=== Wave-B END $(date) ===" | tee -a "${WAVE_B_LOG}"

  echo "=== Wave-B per_wave_checklist ===" | tee -a "${WAVE_B_LOG}"
  ${PY} molmetal/scripts/per_wave_checklist.py --wave "Wave-B Wf-300 G6 C12 no_wass smoke" \
    --config "configs/wf124_p3_per_innovation_ablation_g6.yaml" \
    --out "molmetal/reports/wf300c_launch_sequence/wave_b_checklist.json" \
    --out-md "molmetal/reports/wf300c_launch_sequence/wave_b_checklist.md" 2>&1 | tee -a "${WAVE_B_LOG}" || \
    { echo "FAIL: Wave-B per_wave_checklist"; exit 1; }
else
  echo ""
  echo "=== Wave-B SKIPPED (SKIP_WAVES=${SKIP_WAVES}) ==="
fi

# ============================================================================
# Wave-C: G6 full sweep (TODO-182) — 4 configs × 5 metals × 3 seeds = 60 runs
# ============================================================================
if want_wave "C"; then
  echo ""
  echo "================================================================="
  echo " Wave-C: G6 full sweep — 4 cfgs × 5 metals × 3 seeds = 60 runs"
  echo "================================================================="
  echo "GPU: RTX 5090 cuda:1 (CUDA_VISIBLE_DEVICES=1)"
  echo "Wall-est: 7.5 hr (per Wf-295D §1 + Wf-292B §1 MEASURED)"
  echo ""

  if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY-RUN: skipping real Wave-C launch (60 runs); use scripts/launch_wf124_p3_g6.sh"
    exit 0
  fi

  # Wave-C defers to the existing g6 launch wrapper (Wf-296C verified all flags)
  bash scripts/launch_wf124_p3_g6.sh 2>&1 | tee "results/wf124_p3_per_innovation_ablation_g6/master.log"

  echo "=== Wave-C per_wave_checklist ===" | tee -a "results/wf124_p3_per_innovation_ablation_g6/master.log"
  ${PY} molmetal/scripts/per_wave_checklist.py --wave "Wave-C Wf-300 G6 full sweep" \
    --config "configs/wf124_p3_per_innovation_ablation_g6.yaml" \
    --out "molmetal/reports/wf300c_launch_sequence/wave_c_checklist.json" \
    --out-md "molmetal/reports/wf300c_launch_sequence/wave_c_checklist.md" 2>&1 | tee -a "results/wf124_p3_per_innovation_ablation_g6/master.log" || \
    { echo "FAIL: Wave-C per_wave_checklist"; exit 1; }
else
  echo ""
  echo "=== Wave-C SKIPPED (SKIP_WAVES=${SKIP_WAVES}) ==="
fi

echo ""
echo "=== ALL REQUESTED WAVES DONE $(date) ==="
echo "Per-wave checklists: molmetal/reports/wf300c_launch_sequence/wave_{a,b,c}_checklist.{json,md}"
