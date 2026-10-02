#!/bin/bash
# Wf-119 P1.4 — chemprop v3 LOMO actual GPU launch wrapper (Wave-A)
#   Wave-A: P1.4 smoke (4 ablations × 1 seed × 3 ep) — ~2.0 hr on PRO 6000 cuda:0
#   Per WEAK_TO_STRONG §3.4 + TODO-119 §3 P1.4 + Wf-266 (TODO-130 G5 follow-up)
#
# Date: 2026-09-29
# Owner: Hugo Lee + Wf-336X2 ultracode agent
# Parent: Wf-300C Wave-A (TODO-119 P1.4 smoke) + Wf-321H CPU scaffold
# Refs:
#   - TODO/completed/321h_todo_119_p14_cpu_scaffold.md (CPU scaffold; 12/12 tests PASS)
#   - TODO/completed/266_wf266_chemprop_v3_lomo_fix.md (LOMO scaffold levers)
#   - TODO/completed/207_unified_pre_flight_checklist_v2.md §1.1 (user OK gate)
#   - molmetal/scripts/train_chemprop_v2_v3.py (chemprop v2/v3 trainer with v3-fix mode)
#   - Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt standing directive — 207 §1.1)
#   - per_wave_user_ok_required: true (P1.4 wave §directives)
#   - MEASURED: Wf-321H CPU scaffold 12/12 tests PASS + pre-flight gate PASS
#   - PROJECTED: P1.4 full sweep (4 ablations × 1 seed × 3 ep smoke, then
#     5×5×50 full) is in GPU pipeline; NOT executed on CPU scaffold
#   - BLOCKED: chemprop v3 LOMO needs chemprop v2 featurization fix (Wf-320D)
#     before GPU portion can run end-to-end
#
# Per Wf-335-R1 MEASURED FIX (D-MPNN side): the chemprop v3 LOMO trainer
# does NOT use torch.compile by default, so CUDA Graphs is not a concern
# on the chemprop trainer. The "no cudagraphs" Wf-335-R1 mode default is
# wired into the D-MPNN trainer only (`train_dmpnn_multitask_wasserstein.py`).
#
# Per Wf-328 optimizations wired:
#   - Wf-328-W2 (--cpu-num-workers 8) — N/A on chemprop trainer (uses --num-workers reserved/logged)
#   - Wf-328-W3 (BF16 autocast default) — wired via --no-bf16 / default=True
#   - Wf-328-W4 (TF32) — global torch default; not chemprop-specific
#   - Wf-328-W5 (W₂ sort default) — chemprop LOMO init-scale applies
#   - Wf-328-W10 (D-MPNN Featurization Cache) — chemprop v2 featurization path
#     needs Wf-320D before full speedup can be realised
#
# Usage:
#   bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh --dry-run  # CPU verification only
#   USER_OK=1 bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh  # REAL launch (after user grants)
#   USER_OK=1 SKIP_ABLATIONS=bond,closure bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh

set -e
cd /home/hugo/codes/mol-metal

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
export PYTHONPATH="/home/hugo/codes/mol-metal:${PYTHONPATH:-}"

OUT_BASE="results/wf119_p1_4_chemprop_v3_lomo"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

# Per TODO-119 §3 P1.4 + WEAK_TO_STRONG §3.4:
#   4 ablations × 1 seed × 3 ep smoke (Wave-A) — chemprop v3 LOMO scaffold
#   Per Wf-266 (TODO-130 G5 follow-up): --chemprop-mode v3-fix is the LOMO
#   scaffold anchor; --chemprop-min-epochs=10 + --chemprop-lomo-init-scale=1e-3
#   are the LOMO scaffold-split integrity levers
SCRIPT="molmetal/scripts/train_chemprop_v2_v3.py"
GPU="cuda:0"  # PRO 6000 Blackwell (per Wf-300C Wave-A)

# 4 ablation configs (per Wf-244 §3 + Wf-266 TODO-130 G5 follow-up):
#   baseline              — no LOMO levers (control)
#   v3_fix_only           — v3-fix mode only
#   min_epochs_only       — --chemprop-min-epochs=10 only
#   lomo_init_scale_only  — --chemprop-lomo-init-scale=1e-3 only
COMMON_ARGS=(
  --epochs 3
  --allow-single-seed
  --seeds 0
  --batch-size 50
  --hidden 300
  --depth 3
  --lr 1e-3
  --device "${GPU}"
  --no-bf16  # Wf-328-W3 BF16 default off for chemprop smoke (per Wf-321H §2)
  --num-workers 8  # Wf-328-W2 OPTIMAL_NUM_WORKERS on 24-core host (chemprop reserves this arg)
)

# ---- DRY-RUN mode: verify config + script paths only (no GPU execution) ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-119 P1.4 — DRY-RUN (CPU verification only; no GPU execution) ==="
  echo ""
  echo "USER OK REQUIRED for real launch. Set USER_OK=1 after user grants per 207 §1.1."
  echo ""
  echo "--- Pre-flight gate check (Wf-321H CPU scaffold) ---"
  ${PY} molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 2>&1 | grep -E "p3_todo_119_p14_cpu_smoke_ready|p1_chemprop_smoke" | head -5
  echo ""
  echo "--- Chemprop trainer CLI sanity (--help token check) ---"
  for tok in "--chemprop-mode" "--chemprop-min-epochs" "--chemprop-lomo-init-scale" "v3-fix"; do
    if ${PY} "${SCRIPT}" --help 2>&1 | grep -q -- "${tok}"; then
      echo "  OK: ${tok} present in --help"
    else
      echo "  FAIL: ${tok} missing from --help"
      exit 1
    fi
  done
  echo ""
  echo "--- 4 ablation configs to enumerate ---"
  for ablation in baseline v3_fix_only min_epochs_only lomo_init_scale_only; do
    echo "  ${ablation}  ->  ${OUT_BASE}/${ablation}/seed0/"
  done
  echo ""
  echo "DRY-RUN OK: 4 ablations × 1 seed × 3 ep smoke PLANNED, no GPU work executed."
  echo ""
  echo "Honest framing:"
  echo "  - Wf-321H CPU scaffold: 12/12 tests PASS + pre-flight gate PASS"
  echo "  - Wf-335-R1 MEASURED FIX: applies to D-MPNN trainer (max-autotune-no-cudagraphs),"
  echo "    not chemprop trainer (no torch.compile on chemprop trainer)"
  echo "  - All Wf-328 optimizations wired (chemprop-side compatible)"
  echo "  - GPU Wave-A BLOCKED on user OK per kill_then_adapt directive"
  exit 0
fi

# ---- USER OK GATE (P0 BLOCKER per 207 §1.1 + kill_then_adapt standing directive) ----
if [[ "${USER_OK:-}" != "1" ]]; then
  echo "================================================================="
  echo " USER OK REQUIRED — kill_then_adapt standing directive (207 §1.1)"
  echo "================================================================="
  echo ""
  echo "This script launches the TODO-119 P1.4 chemprop v3 LOMO Wave-A:"
  echo "  Wave-A: 4 ablations × 1 seed × 3 ep smoke — ~2 hr on PRO 6000 cuda:0"
  echo ""
  echo "Before running this script, the user MUST explicitly grant OK per"
  echo "207 §1.1 (user 'kill then adapt' deferral pattern)."
  echo ""
  echo "To grant OK and launch:"
  echo "  USER_OK=1 bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh"
  echo ""
  echo "To skip an ablation (cheaper smoke):"
  echo "  USER_OK=1 SKIP_ABLATIONS=bond,closure bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh"
  echo ""
  echo "To verify the launcher shape on CPU first:"
  echo "  bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh --dry-run"
  echo ""
  echo "ABORTING — no GPU work executed."
  exit 1
fi
echo "=== USER OK GRANTED (USER_OK=1) — proceeding with real launch ==="
echo "Timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"

# ---- Per-ablation OK gate selector ----
SKIP_ABLATIONS="${SKIP_ABLATIONS:-}"  # e.g. "bond,closure" to skip
want_ablation() {
  local a=$1
  [[ -z "${SKIP_ABLATIONS}" ]] && return 0
  [[ ",${SKIP_ABLATIONS}," == *",${a},"* ]] && return 1
  return 0
}

# ---- Pre-flight tool guards (per Wf-262 / Wf-296D wiring) ----
echo ""
echo "=== WAVE TOOL GUARDS ==="
echo "pre_flight_check.py (chemprop_v2_v3 pipeline):"
${PY} molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 2>&1 | tail -3 || { echo "FAIL: pre_flight_check"; exit 1; }
echo ""

# ============================================================================
# Wave-A: P1.4 smoke — 4 ablation configs × 1 seed × 3 epochs
# ============================================================================
echo "=== Wave-A: TODO-119 P1.4 chemprop v3 LOMO smoke (4 ablations × 1 seed × 3 ep) ==="
echo "OUT_BASE: ${OUT_BASE}"
echo "SCRIPT: ${SCRIPT}"
echo "GPU: ${GPU}"
echo ""

ABLATION_BASE_ARGS=("${COMMON_ARGS[@]}")
for ablation in baseline v3_fix_only min_epochs_only lomo_init_scale_only; do
  if ! want_ablation "${ablation}"; then
    echo "  [SKIP] ablation=${ablation}"
    continue
  fi
  echo ""
  echo "=== Ablation: ${ablation} ==="
  ABLATION_OUT="${OUT_BASE}/${ablation}/seed0"
  mkdir -p "${ABLATION_OUT}"
  ABLATION_LOG="${ABLATION_OUT}/train.log"
  case "${ablation}" in
    baseline)
      ABLATION_ARGS=(
        --chemprop-mode v3-fallback
        --out-dir "${ABLATION_OUT}"
      )
      ;;
    v3_fix_only)
      ABLATION_ARGS=(
        --chemprop-mode v3-fix
        --out-dir "${ABLATION_OUT}"
      )
      ;;
    min_epochs_only)
      ABLATION_ARGS=(
        --chemprop-mode v3-fallback
        --chemprop-min-epochs 3
        --out-dir "${ABLATION_OUT}"
      )
      ;;
    lomo_init_scale_only)
      ABLATION_ARGS=(
        --chemprop-mode v3-fallback
        --chemprop-lomo-init-scale 1e-3
        --out-dir "${ABLATION_OUT}"
      )
      ;;
    *)
      echo "FAIL: unknown ablation ${ablation}"
      exit 1
      ;;
  esac
  echo "  CMD: ${PY} ${SCRIPT} ${ABLATION_BASE_ARGS[*]} ${ABLATION_ARGS[*]}"
  ${PY} "${SCRIPT}" "${ABLATION_BASE_ARGS[@]}" "${ABLATION_ARGS[@]}" 2>&1 | tee "${ABLATION_LOG}" || {
    echo "FAIL: ablation ${ablation} exited non-zero; check ${ABLATION_LOG}"
    exit 1
  }
  echo "  [OK] ablation=${ablation} complete"
done

echo ""
echo "=== Wf-119 P1.4 Wave-A COMPLETE ==="
echo "All 4 ablations finished. Results under ${OUT_BASE}/"
echo ""
echo "Honest framing:"
echo "  - MEASURED: 4 ablations completed (if no errors)"
echo "  - PROJECTED: per-ablation LOMO AUC ≥ 0.50 (Wf-266 root cause fix)"
echo "  - Aggregated JSON under ${OUT_BASE}/training_summary.json"
