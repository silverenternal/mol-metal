#!/usr/bin/env bash
# Wf-414-A1 — Wf-390-A1 Path A 5-metal subset m2 R² GPU retrain launch wrapper
#                  (USER_OK-gated SCAFFOLD wrapper; CPU SCAFFOLD-only by default)
#
# Date:        2026-09-30
# Owner:       Hugo Lee + ultracode agent (Wf-414-A1)
# Parent:      Wf-390-A1 §2 Path A (TODO/pending/376_m2_r2_gpu_retrain_plan.md)
#              + Wf-411-A2 HONEST-NEGATIVE ("wrapper MISSING on disk")
# Trainer:     molmetal/scripts/train_dmpnn_multitask_wasserstein.py
#              (canonical D-MPNN multi-task wasserstein trainer; FIVE_METALS
#               constraint at line 141 + argparse choices at line 881 — Path A
#               works WITHIN this constraint by using the 5-metal subset natively)
#
# Refs:
#   - TODO/pending/376_m2_r2_gpu_retrain_plan.md §2 Path A (Wf-390-A1 plan)
#   - TODO/completed/367b1_m2_r2_recovery.md (Wf-367-B1 multi-stream Ridge ensemble)
#   - TODO/completed/390a1_m2_r2_path_a_smoke.md (Wf-390-A1 SCAFFOLD smoke)
#   - TODO/completed/390a2_todo179_deeper_audit.md (Wf-390-A2 alternative encoder audit)
#   - TODO/completed/390a3_se3_sb_wire.md (Wf-390-A3 SE(3) SB wire-in audit)
#   - docs/M2_R2_FINAL_VERDICT.md (Wf-369-A2 FINAL VERDICT 320 LOC)
#   - scripts/_dry_run_log_helper.sh (Wf-363-A1 per-pair DRY_RUN.log schema)
#   - molmetal/scripts/wf390a1_path_a_smoke.py (Wf-390-A1 SCAFFOLD smoke helper)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#   - Wf-335-R1+R3 (--no-cuda-graphs safety lever for Blackwell sm_120)
#   - Wf-380-B1 + Wf-388-A1 + Wf-405-A2 (--per-cohort-features flag addition)
#
# HONEST FRAMING (party-twenty-spirit §3.5 + AGENTS.md §5.2):
#   - user_ok_required: true (kill_then_adapt; 5-metal × 3-seed × 50-ep × ~25 min
#     wall per pair PROJECTED = ~12.5 hr GPU wall PROJECTED per Wf-390-A1 §2.1)
#   - Scope: 15 (metal, seed) pairs (5 metals × 3 seeds × 50 epochs) on the
#     metalcytotox_combined cohort (n=2000) for the per-cohort config axis;
#     this corresponds to sub-launch A.1 in Wf-390-A1 §2.2 (the n=2000 cohort).
#   - Wall: ~12 hr GPU (PROJECTED) per Wf-260 G6 timing on RTX 5090 / Blackwell sm_120
#   - Per-cohort config axis: --per-cohort-features ON (per Wf-388-A1 + Wf-405-A2)
#     emits per-cohort tier (n_lt_300 / n_lt_1000 / n_gte_1000) + features + alpha
#     to stdout at training start; backward-compat when --per-cohort-features OFF
#   - --chemprop-v3-cache: this flag exists on train_chemprop_v2_v3.py per Wf-392-A2
#     but NOT on train_dmpnn_multitask_wasserstein.py. The task brief listed it
#     alongside the D-MPNN trainer; we honor the brief by including it conditional
#     on the script variable — see COMMON_ARGS assembly below for the runtime guard.
#   - n=200 cohort ceiling preserved: Rh (n=1200) + Re (n=800) per-metal singlet
#     subsweeps PROJECTED R² remains near 0 or negative per Wf-316A 12-d RDKit
#     signal floor + Wf-367-B1 multi-stream overfit. The 15-pair sweep targets
#     metalcytotox_combined (n=2000) CEILING_BREAK axis only.
#   - BUGGY Wf-125 P1 cohort structural note: this wrapper uses
#     metalcytotox_5metal_multitask_v4.csv (Wf-261 strict scaffold split, 0 leak)
#     NOT the BUGGY Wf-125 P1 cohort — per Wf-387-A2 honest-negative, the balanced
#     Rh cohort would be a separate 3-run CONTROLD-PAIR retrain if needed.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run
#     # CPU verification only (USER_OK gate still required)
#   USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh
#     # actual GPU launch (~12 hr wall PROJECTED)
#   USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --pair "Ru 0"
#     # subset override (single pair for debugging)
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0 + Wf-294B family):
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch.
#   - Exit code 64 on USER_OK gate refusal (Wf-294B convention).
#
# Stop semantics: per TODO-207 v2 §7.2 (per-phase abort criteria):
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN/OOM/IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: per-cohort R² drop > 0.20 vs Wf-335-M2 CEILING_BREAK baseline

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf390a1_path_a"
TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"
# 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
# The trainer already accepts --config and ends parse_args() in
# apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
# yaml is the base and the per-call flags below override it.
# Precedence: explicit CLI flag > YAML > argparse default.
# Absolute: this wrapper has no top-level `cd`, so a relative path would
# resolve against the caller's cwd.
DMPNN_CONFIG="${DMPNN_CONFIG:-${PROJECT_ROOT}/configs/dmpnn.yaml}"
# Wf-261 strict scaffold split (0 leak) + v4 cohort
DATA_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# Path A 5-metal subset (matches trainer FIVE_METALS exactly per Wf-390-A1 §1.1)
# Tuple: (Ru, Ir, Rh, Os, Re) — matches train_dmpnn_multitask_wasserstein.py:141
PATH_A_METALS=(Ru Ir Rh Os Re)
# 3 seeds × 5 metals × 50 epochs = 15 (metal, seed) pairs per the task brief
PATH_A_SEEDS=(0 1 2)

# Default pairs: all 15 (metal, seed) tuples per task brief
DEFAULT_PAIRS=(
  "Ru 0" "Ru 1"
  "Ir 0" "Ir 1"
  "Rh 0" "Rh 1"
  "Os 0" "Os 1"
  "Re 0" "Re 1"
  "Ru 2"
  "Ir 2"
  "Rh 2"
  "Os 2"
  "Re 2"
)

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-294B family convention; exit 64 on refusal)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # CPU plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~12 hr GPU run (15 pairs)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --pair "Ru 0"  # subset override (debug pair)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# DRY-RUN log helper (Wf-363-A1 canonical schema)
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# -----------------------------------------------------------------------------
# Flag parsing
# -----------------------------------------------------------------------------
DRY_RUN=0
PAIR_OVERRIDE=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --pair)
      shift_next=1  # next arg is the pair string
      ;;
    --pair=*)
      PAIR_OVERRIDE+=("${arg#--pair=}")
      ;;
  esac
done
# After flag-scan, capture positional --pair <str> if not already set
if [[ ${#PAIR_OVERRIDE[@]} -eq 0 ]]; then
  prev_arg=""
  for i in "$@"; do
    if [[ "${prev_arg}" == "--pair" ]]; then
      PAIR_OVERRIDE+=("$i")
    fi
    prev_arg="$i"
  done
fi

# If --pair not supplied, default to all 15 pairs
if [[ ${#PAIR_OVERRIDE[@]} -eq 0 ]]; then
  for p in "${DEFAULT_PAIRS[@]}"; do
    PAIR_OVERRIDE+=("$p")
  done
fi

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
OUT_BASE="${REPORTS_DIR}"
mkdir -p "${OUT_BASE}"
LOG="${OUT_BASE}/master.log"

# -----------------------------------------------------------------------------
# Common args (per Wf-380-B1 + Wf-388-A1 + Wf-405-A2 + Wf-328 series)
# -----------------------------------------------------------------------------
# Per TODO-181 §3 + Wf-260 G6 timing: --epochs 50 --batch-size 256 --hidden 300
#   --depth 3 + --use-l2-guard + --ablation-config mtl_baseline
# Per Wf-328-W5: --w2-gradient-mode sort (REPLACES legacy sliced mode)
# Per Wf-405-A2 + Wf-388-A1: --per-cohort-features (emit per-cohort tier
#   n_lt_300 / n_lt_1000 / n_gte_1000 + features + alpha at training start;
#   backward-compat OFF preserves bit-for-bit Wf-131 P1 + Wf-251-F4 baseline)
COMMON_ARGS=(
  --epochs 50
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --ablation-config mtl_baseline
  --w2-gradient-mode sort
  --n-mc 256
  --device "cuda:0"
  --cuda-visible-devices 0
  --cpu-num-workers 8
  --data-csv "${DATA_CSV}"
  --per-cohort-features
)

# --chemprop-v3-cache: per Wf-392-A2, this flag exists on train_chemprop_v2_v3.py
# but NOT on train_dmpnn_multitask_wasserstein.py. The task brief listed it for
# the D-MPNN trainer; we honor the brief LITERALLY by appending it conditionally
# on the trainer script path. If a future user retargets this wrapper to chemprop
# v3 (e.g. Path B per Wf-390-A1 §0), the flag will pass through cleanly.
if [[ "${TRAINER}" == *"chemprop_v2_v3"* ]]; then
    COMMON_ARGS+=( --chemprop-v3-cache )
fi

# -----------------------------------------------------------------------------
# Pre-launch SCAFFOLD smoke (CPU-only; Wf-390-A1 §1.2 SCAFFOLD/APPLIED pattern)
# -----------------------------------------------------------------------------
# Wf-390-A1 SCAFFOLD mode: verify trainer CLI accepts the 5-metal subset
# + emits per-cohort config JSON without GPU launch. This is the SCAFFOLD
# phase; APPLIED phase requires USER_OK=1 + VRAM check + wave_budget_tracker.
echo "=== Wf-390-A1 Path A SCAFFOLD smoke (CPU-only) ===" | tee -a "${LOG}"
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf390a1_path_a_smoke.py" \
    --out-dir "${OUT_BASE}/scaffold_smoke" 2>&1 | tee -a "${LOG}"
SMOKE_RC=${PIPESTATUS[0]}
echo "SCAFFOLD_SMOKE_RC=${SMOKE_RC}" | tee -a "${LOG}"
if [ "${SMOKE_RC}" -ne 0 ]; then
    echo "FATAL: SCAFFOLD smoke failed; aborting before any GPU launch." | tee -a "${LOG}"
    exit 1
fi

# -----------------------------------------------------------------------------
# DRY-RUN mode: emit DRY_RUN.log + plan.json + enumerate pairs (no GPU work)
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== Wf-390-A1 Path A — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ===" | tee -a "${LOG}"
    echo "OUT_BASE: ${OUT_BASE}" | tee -a "${LOG}"
    echo "PY: ${PY}" | tee -a "${LOG}"
    echo "SCRIPT: ${TRAINER}" | tee -a "${LOG}"
    echo "DATA_CSV: ${DATA_CSV}" | tee -a "${LOG}"
    echo "GPU: cuda:0 (RTX PRO 6000 Blackwell sm_120)" | tee -a "${LOG}"
    echo "PAIRS (${#PAIR_OVERRIDE[@]}): ${PAIR_OVERRIDE[*]}" | tee -a "${LOG}"
    echo "COMMON_ARGS: ${COMMON_ARGS[*]}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # CSV sanity check
    test -f "${DATA_CSV}" && echo "OK: CSV exists (${DATA_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
    n_rows=$(wc -l < "${DATA_CSV}")
    echo "n_rows (incl header): ${n_rows}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # Per-pair breakdown
    echo "--- ${#PAIR_OVERRIDE[@]} (metal, seed) pairs to enumerate ---" | tee -a "${LOG}"
    for pair in "${PAIR_OVERRIDE[@]}"; do
        read -r metal seed <<< "${pair}"
        echo "  ${metal} seed=${seed}  ->  ${OUT_BASE}/${metal}/seed${seed}/" | tee -a "${LOG}"
    done
    echo "" | tee -a "${LOG}"

    # Wf-363-A1 — emit structured DRY_RUN.log
    pair_args=()
    for pair in "${PAIR_OVERRIDE[@]}"; do
        pair_args+=( --pair "${pair}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-390-A1 Path A — ${#PAIR_OVERRIDE[@]} (metal, seed) pairs" \
        --wrapper "$(basename "$0")" \
        --mode "PathA_5metal_subset_50epoch" \
        --trainer "${TRAINER}" \
        --flags "${COMMON_ARGS[*]}" \
        --gpu "cuda:0" \
        --budget-hr "12" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log" | tee -a "${LOG}"

    # Wf-363-A1 — emit plan.json (consistent with Wf-355 pattern)
    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_wf390a1_m2_r2_path_a.sh",\n'
        printf '  "wf_origin": "Wf-414-A1 (USER_OK gate + SCAFFOLD smoke wire-in)",\n'
        printf '  "parent_plan": "TODO/completed/376_m2_r2_gpu_retrain_plan.md",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "metals": ["Ru", "Ir", "Rh", "Os", "Re"],\n'
        printf '  "seeds": [0, 1, 2],\n'
        printf '  "epochs": 50,\n'
        printf '  "n_pairs": %d,\n' "${#PAIR_OVERRIDE[@]}"
        printf '  "gpu": "cuda:0",\n'
        printf '  "wall_budget_hr": 12,\n'
        printf '  "wall_budget_label": "PROJECTED (~12 hr GPU sequential per Wf-390-A1 §2.1 + Wf-260 G6 timing; Ru heaviest ~25 min/seed x 3 + Ir ~25 min x 3 + Os ~20 min x 3 + Rh ~15 min x 3 + Re ~10 min x 3 + ~10% overhead)",\n'
        printf '  "trainer": "%s",\n' "${TRAINER}"
        printf '  "data_csv": "%s",\n' "${DATA_CSV}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "per_cohort_features": true,\n'
        printf '  "chemprop_v3_cache_conditional": "appended only when TRAINER=*chemprop_v2_v3*",\n'
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "USER_OK gate enforced; SCAFFOLD smoke passed (Wf-390-A1 §1.2); 15 pairs enumerated; DRY_RUN.log + plan.json emitted; 0 GPU launches",\n'
        printf '    "PROJECTED": "5-metal x 3-seed x 50-epoch sequential on metalcytotox_combined (n=2000) CEILING_BREAK axis; ~12 hr wall on RTX 5090 / Blackwell sm_120 per Wf-260 G6 timing",\n'
        printf '    "kill_then_adapt_directive": "12-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
        printf '    "n_200_cohort_ceiling_preserved": "Rh (n=1200) + Re (n=800) per-metal singlet subsweeps PROJECTED R² remains near 0 or negative per Wf-316A 12-d RDKit signal floor + Wf-367-B1 multi-stream overfit",\n'
        printf '    "wf390a1_scaffold_smoke_status": "SCAFFOLD phase PASSED (CLI argparse + FIVE_METALS subset match + Wf-379-A2 negative test); APPLIED phase USER_OK-gated",\n'
        printf '    "wf411a2_honest_negative": "wrapper MISSING on disk before Wf-414-A1; now SHIPPED at scripts/launch_wf390a1_m2_r2_path_a.sh per Wf-414-A1 SCAFFOLD closure"\n'
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    echo "plan.json emitted -> ${PLAN_JSON}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"
    echo "DRY-RUN OK: ${#PAIR_OVERRIDE[@]} (metal, seed) pairs PLANNED, no GPU work executed." | tee -a "${LOG}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Pre-launch wave-tool guards (Wf-343C2 + Wf-343C3 per Wf-347-B1 + Wf-347-B2)
# -----------------------------------------------------------------------------
# Wf-347-A1/A2: pre-flight + wf343c3 data quality check wired into all
# launchers. --out-dir only (NO --data-csv per Wf-356-A1 retry-bug fix).
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all \
    || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf343c3_data_quality_check.py" \
    --out-dir "${REPORTS_DIR}/wf343c3_data_quality" \
    || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/per_wave_checklist.py" \
    --wave "Wf-390-A1-PathA" \
    || { echo "FAIL: per_wave_checklist"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wave_budget_tracker.py" \
    || { echo "FAIL: wave_budget_tracker"; exit 1; }

# -----------------------------------------------------------------------------
# Helper: poll pair log until target epoch is reached (Wf-302B guard)
# -----------------------------------------------------------------------------
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-3600}"  # 60 min cap per pair (PROJECTED ~25 min wall)
  local start_s elapsed last_epoch
  start_s=$(date +%s)
  elapsed=0
  while true; do
    last_epoch=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$pair_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "${last_epoch}" ] && [ "${last_epoch}" -ge "${target_epoch}" ]; then
      echo "Pair complete: last_epoch=${last_epoch} >= target=${target_epoch}"
      return 0
    fi
    elapsed=$(( $(date +%s) - start_s ))
    if [ "${elapsed}" -ge "${max_wait_s}" ]; then
      echo "TIMEOUT: last=${last_epoch} elapsed=$((elapsed))s"
      return 1
    fi
    sleep 10
  done
}

# -----------------------------------------------------------------------------
# Real launch: per-pair runner
# -----------------------------------------------------------------------------
run_pair() {
  local metal=$1
  local seed=$2
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  # (Wf-343c2 7-check surface; --skip-live for CPU-only dry-runs)
  PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf343c2_per_wave_check.py" \
    --wave "Wf-390-A1-PathA-${metal}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --budget-hr 0.5 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" 2>&1
  local rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK ${metal} seed=${seed} rc=${rc_c2} log=${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" | tee -a "${LOG}"
  # Diagnostic-only: do NOT return 1 on rc!=0 (per kill_then_adapt; see
  # Wf-347-B1 §5 comment in launch_wf124_p3_g5.sh).

  # Wf-347-B1: per-pair budget tracking
  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START ${metal} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=0 ${PY} "${TRAINER}" \
    --config "${DMPNN_CONFIG}" \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}

  local pair_end_s pair_elapsed_s
  pair_end_s=$(date +%s)
  pair_elapsed_s=$(( pair_end_s - pair_start_s ))
  echo "PAIR_BUDGET_END ${metal} seed=${seed} elapsed_s=${pair_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') END ${metal} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "${rc}" -ne 0 ]; then
    echo "FAIL: ${metal} seed=${seed} exited rc=${rc}"
    return "${rc}"
  fi
  # Wf-302B guard: verify the pair actually reached target epoch (50)
  wait_until_epoch_reached "${pair_log}" 50 3600 || { echo "FAIL: ${metal} seed=${seed} did not reach epoch 50"; return 1; }
}

echo "=== Wf-390-A1 Path A START $(date) ===" | tee -a "${LOG}"
echo "${#PAIR_OVERRIDE[@]} pairs (5 metals x 3 seeds x 50 ep on metalcytotox_combined n=2000) -- USER_OK-gated per AGENTS.md §5.1 + Wf-294B family" | tee -a "${LOG}"

# Iterate PAIR_OVERRIDE sequentially
for pair in "${PAIR_OVERRIDE[@]}"; do
  read -r metal seed <<< "${pair}"
  run_pair "${metal}" "${seed}"
done

echo "=== ALL Wf-390-A1 PATH A PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation: collect per_seed_metrics_seed*.json into training_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for metal_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not metal_dir.is_dir() or metal_dir.name == 'logs' or metal_dir.name.startswith('scaffold_'):
        continue
    for seed_dir in sorted(metal_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        for p in seed_dir.glob('per_seed_metrics_seed*.json'):
            runs.append({'metal': metal_dir.name, 'seed': seed_dir.name.replace('seed', ''), 'metrics_path': str(p)})
out = {
    'wf390a1_patha_total_pairs': len(runs),
    'n_pairs_planned': ${#PAIR_OVERRIDE[@]},
    'runs': runs,
}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (planned ${#PAIR_OVERRIDE[@]})')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"
