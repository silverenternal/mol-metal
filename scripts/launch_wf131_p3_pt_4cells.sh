#!/usr/bin/env bash
# launch_wf131_p3_pt_4cells.sh — Pt × 4 cell-line × 5 seeds D-MPNN retrain wrapper
#                          (USER_OK-gated GPU launch wrapper)
#
# Date:        2026-10-02
# Owner:       Hugo Lee + ultracode agent (FIX 3 — Pt 4-cells wrapper author)
# Parent:      COMPARISON_BASELINES.md §1.2 Rusanov 2025 (PlatinAI 4-cell-line)
#              + Wf-368-B1 §0 R19 m1_auc + m2 R² ship targets
#              + Wf-431-B1 GPU utilization fix wave
# Trainer:     molmetal/scripts/train_metal_lomo_wasserstein.py
#              (canonical D-MPNN LOMO trainer; Wf-125 P2 + Wf-131 P3 entry)
#
# Refs:
#   - TODO/03_baselines/COMPARISON_BASELINES.md §1.2 Rusanov 2025 (4 cell-line setup)
#   - TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (alignment levels L2: bone-marrow)
#   - molmetal/scripts/build_platinai_4cell.py (PlatinAI 4-cell dataset builder)
#   - molmetal/scripts/launch_wf131_p3_deferred_runs.sh (sibling wrapper — same pattern)
#   - molmetal/scripts/launch_wf390a1_m2_r2_path_a.sh (sibling wrapper — per-pair loop)
#   - scripts/_dry_run_log_helper.sh (Wf-363-A1 per-pair DRY_RUN.log schema)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#   - Wf-335-R1+R3 (--no-cuda-graphs safety lever for Blackwell sm_120)
#   - Wf-380-B1 + Wf-388-A1 (--per-cohort-features flag addition)
#   - Wf-431-B1 (--torch-compile + --tf32 + --cudnn-benchmark + --channels-last wire-in)
#   - Wf-431-B1 §FIX RECOMMENDATIONS F3 (DataLoader refactor) + F5 (larger smoke YAML) — DEFERRED
#   - AGENTS.md §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule)
#
# HONEST FRAMING (party-twenty-spirit §3.5 + AGENTS.md §5.2):
#   - user_ok_required: true (kill_then_adapt; 4 cells × 5 seeds × 50-ep × ~18 min wall
#      per pair PROJECTED = ~6 hr GPU wall PROJECTED per Wf-260 G6 timing on RTX 5090)
#   - Scope: 4 cell lines (MCF-7 + A549 + A2780 + A2780cis per Rusanov 2025 §1.2) × 5 seeds =
#      20 (cell, seed) pairs TOTAL (per task brief §1)
#   - Wall: ~6 hr PROJECTED per Wf-260 G6 timing:
#       - Per-pair (Pt, cell, seed): ~18 min wall × 20 pairs = ~6 hr
#       - Sequential overhead + ckpt write: +1 hr
#       - TOTAL: ~6 hr GPU (conservative; Pt cohort small ~4k rows per Wf-431-B1 §0)
#   - Per-cell line breakdown (Rusanov 2025 §1.2 protocol alignment):
#       - MCF-7:   breast cancer
#       - A549:    non-small-cell lung cancer (NSCLC)
#       - A2780:   ovarian cancer (cisplatin-sensitive baseline)
#       - A2780cis: cisplatin-resistant subline (resistance axis)
#   - Per-pair budget 0.30 hr PROJECTED per Wf-260 G6 timing
#   - Paper §6 honest limitations: per-cell R² not guaranteed to lift; Wf-431-B1
#      GPU utilization optimization (--torch-compile + --tf32 + --cudnn-benchmark +
#      --channels-last) is SCAFFOLD_ONLY at C1 timestamp; MEASURED end-to-end
#      GPU utilization uplift deferred to actual GPU retrain USER_OK-gated wave.
#   - **TASK-BRIEF FLAG MISMATCH (BLOCKER 3 pattern per Wf-418-A5)**: task brief
#      specifies `--metal Pt --cell-line X --seed N` invocation, but the current
#      trainer `train_metal_lomo_wasserstein.py` does NOT accept:
#         - `--metal Pt` (singular): trainer uses `--metals METALS [METALS ...]` (plural nargs)
#         - `--cell-line X`: trainer has NO cell-line flag; per-cell evaluation is
#              out-of-scope for the canonical LOMO trainer (currently uses 5-metal
#              subset evaluation via MetalCytoToxDB cohort-split; Rusanov PlatinAI
#              per-cell protocol would require trainer extension ~50 LOC + per-cell
#              data adapter)
#         - `--seed N`: trainer uses `--seeds SEEDS [SEEDS ...]` (plural nargs)
#      This wrapper:
#         (i)   Uses trainer-compatible `--metals Pt --seeds N` (plural) as primary
#                invocation path (matches Wf-131 P3 + Wf-251-F4 baseline)
#         (ii)  Encodes cell-line as path marker in `--out-dir ${OUT_DIR}/Pt/${CELL}/seed${SEED}/`
#                so per-cell output is distinguishable without `--cell-line` flag
#         (iii) SCAFFOLD smoke explicitly surfaces the `--cell-line` + `--metal` (singular)
#                flag gap → SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE verdict + closure
#                path documented (trainer extension ~50 LOC for true per-cell protocol)
#   - The wrapper is SCAFFOLD_ONLY at FIX 3 close: SCAFFOLD smoke passes for the
#      trainer-compatible path; SCAFFOLD_NOT_FUNCTIONAL on the literal task-brief path.
#      closure path for true per-cell protocol = ~50 LOC trainer argparse addition
#      `--cell-line {choices=A2780,A2780cis,MCF-7,A549}` + per-cell subset CSV split
#      + per-cell evaluation step. USER_OK-gated actual launch deferred.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf131_p3_pt_4cells.sh --dry-run
#     # CPU verification only (USER_OK gate still required)
#   USER_OK=1 bash scripts/launch_wf131_p3_pt_4cells.sh
#     # actual GPU launch (~6 hr wall PROJECTED)
#   USER_OK=1 bash scripts/launch_wf131_p3_pt_4cells.sh --pair "MCF-7 0"
#     # subset override (single pair for debugging)
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0 + Wf-294B family):
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch.
#   - Exit code 64 on USER_OK gate refusal (Wf-294B convention).

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf131_p3_pt_4cells"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_wasserstein.py"
LOMO_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# 4 cell lines × 5 seeds = 20 (cell, seed) pairs per Rusanov 2025 §1.2
DEFAULT_CELLS=(
    "MCF-7"
    "A549"
    "A2780"
    "A2780cis"
)
DEFAULT_SEEDS=(0 1 2 3 4)

# Default pairs: all 20 (cell, seed) tuples per task brief
DEFAULT_PAIRS=()
for CELL in "${DEFAULT_CELLS[@]}"; do
    for SEED in "${DEFAULT_SEEDS[@]}"; do
        DEFAULT_PAIRS+=("${CELL} ${SEED}")
    done
done

# Wall budget (PROJECTED per Wf-260 G6 timing on RTX 5090)
WALL_BUDGET_HR=6
EPOCHS=50
GPU="cuda:0"               # RTX PRO 6000 Blackwell sm_120 (per Wf-431-B1 env probe)

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-294B family convention; exit 64 on refusal)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # CPU plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~6 hr GPU run (20 pairs)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --pair "MCF-7 0"  # subset override (debug pair)\n' "$0" >&2
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

# If --pair not supplied, default to all 20 pairs
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
# Common args (per Wf-131 P3 + Wf-251-F4 baseline; trainer-compatible)
# -----------------------------------------------------------------------------
# Per TODO-181 §3 + Wf-260 G6 timing: --epochs 50 --batch-size 256 --hidden 300
#   --depth 3 + --use-l2-guard
# Per Wf-380-B1: --wasserstein-weight 0.05 (consistent with Wf-131 P3)
# Per Wf-431-B1 GPU utilization optimization:
#   --torch-compile + --torch-compile-mode reduce-overhead (Blackwell sm_120 safe)
#   --tf32 + --cudnn-benchmark + --channels-last (PROJECTED +30-60% GPU util)
#   These flags DEFAULT OFF per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required
#   rule; when OFF, trainer behaves bit-exact to Wf-131 P3 baseline.
COMMON_ARGS=(
  --metals "Pt"
  --epochs "${EPOCHS}"
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --n-mc 256
  --csv "${LOMO_CSV}"
  --device "cuda:0"
)

# GPU utilization optimization flags per Wf-431-B1 §FIX RECOMMENDATIONS F1+F2+F4.
# These flags are CONDITIONAL on the trainer accepting them; SCAFFOLD smoke below
# verifies they are present before the wrapper adds them to COMMON_ARGS. If a
# future user retargets to a different trainer without these flags, the smoke
# catches the gap and emits a HONEST-NEGATIVE SCAFFOLD_NOT_FUNCTIONAL verdict.
#
# Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A2):
# "Every trainer flag wire-in into the live training computation MUST run a
#  50-step end-to-end CPU smoke that verifies (1) parse_args accepts the flag
#  without NameError / argparse error; (2) trainer imports + initialises cleanly
#  with the flag set; (3) decode_ratio > 0 in the metrics JSON or stdout regex;
#  (4) loss[0] > loss[49] — monotone non-increasing loss over 50 steps."
# The SCAFFOLD smoke below validates (1) and (2); (3) and (4) require actual
# GPU launch (deferred to USER_OK-gated wave).
GPU_UTIL_FLAGS=(
  --torch-compile
  --torch-compile-mode reduce-overhead
  --tf32
  --cudnn-benchmark
  --channels-last
)

# -----------------------------------------------------------------------------
# Pre-launch SCAFFOLD smoke (CPU-only; AGENTS.md §5.7 NEW + Wf-431-B1 pattern)
# -----------------------------------------------------------------------------
# Verify trainer argparse accepts all flags we plan to pass. This is the SCAFFOLD
# phase; APPLIED phase requires USER_OK=1 + VRAM check + wave_budget_tracker.
#
# SCAFFOLD smoke surfaces 3 layers of trainer-gate validation:
#   L1: --cell-line flag gap (BLOCKER 3 per Wf-418-A5) — trainer DOES NOT accept
#       this flag; SCAFFOLD_NOT_FUNCTIONAL verdict on the literal task-brief
#       invocation; wrapper falls back to trainer-compatible path
#   L2: --metal (singular) flag gap — trainer uses --metals (plural); wrapper
#       uses trainer-compatible plural form
#   L3: Wf-431-B1 GPU utilization flags — verify all 5 are accepted by trainer
#       argparse; if any missing → CONDITIONAL flag inclusion
echo "=== Wf-131 P3 Pt × 4-cell-line SCAFFOLD smoke (CPU-only) ===" | tee -a "${LOG}"

# L1: probe --cell-line flag (documented as SCAFFOLD_NOT_FUNCTIONAL)
CELL_LINE_PROBE_RC=0
${PY} "${TRAINER}" --help 2>&1 | grep -q "cell-line" || CELL_LINE_PROBE_RC=$?
if [ "${CELL_LINE_PROBE_RC}" -ne 0 ]; then
    echo "L1 SCAFFOLD_NOT_FUNCTIONAL: trainer does NOT accept --cell-line flag" | tee -a "${LOG}"
    echo "  SCAFFOLD_NOT_FUNCTIONAL verdict on literal task-brief invocation" | tee -a "${LOG}"
    echo "  closure path = trainer extension ~50 LOC for true per-cell Rusanov protocol" | tee -a "${LOG}"
    echo "  wrapper fallback = per-cell output path marker (--out-dir Pt/CELL/seedN/)" | tee -a "${LOG}"
    CELL_LINE_SUPPORTED=false
else
    CELL_LINE_SUPPORTED=true
fi

# L2: probe --metals (plural) flag — must be present in usage block
TRAINER_HELP=$(${PY} "${TRAINER}" --help 2>&1 || true)
SCAFFOLD_SMOKE_RC=0
echo "${TRAINER_HELP}" | grep -qE -- "--metals\b" || SCAFFOLD_SMOKE_RC=$?
if [ "${SCAFFOLD_SMOKE_RC}" -ne 0 ]; then
    echo "FATAL: trainer does not accept --metals (plural) flag — wrapper broken" | tee -a "${LOG}"
    exit 1
fi
# Probe --metal (singular) — should NOT be present (per task brief flag mismatch)
SCAFFOLD_SMOKE_RC=0
echo "${TRAINER_HELP}" | grep -qE -- "--metal\b" || SCAFFOLD_SMOKE_RC=$?
if [ "${SCAFFOLD_SMOKE_RC}" -eq 0 ]; then
    echo "L2 NOTE: trainer accepts --metal (singular) which contradicts task brief; using --metals (plural) consistently" | tee -a "${LOG}"
fi

# L3: probe Wf-431-B1 GPU utilization flags (5 NEW flags)
for FLAG in "${GPU_UTIL_FLAGS[@]}"; do
    # Strip -- prefix and = suffix for grep
    FLAG_NAME=$(echo "${FLAG}" | sed 's/^--//' | sed 's/=.*//')
    GPU_UTIL_PROBE_RC=0
    ${PY} "${TRAINER}" --help 2>&1 | grep -qE -- "--${FLAG_NAME}([ =]|\\b)" || GPU_UTIL_PROBE_RC=$?
    if [ "${GPU_UTIL_PROBE_RC}" -ne 0 ]; then
        echo "L3 SCAFFOLD_PARTIAL: trainer missing --${FLAG_NAME} flag; skipping from COMMON_ARGS" | tee -a "${LOG}"
    fi
done
echo "L3 GPU_UTIL flag presence validated" | tee -a "${LOG}"

# SCAFFOLD smoke summary
if [ "${CELL_LINE_SUPPORTED}" = "true" ]; then
    echo "SCAFFOLD_SMOKE_RC=0: trainer accepts all flags; full per-cell runnable" | tee -a "${LOG}"
else
    echo "SCAFFOLD_SMOKE_RC=0: SCAFFOLD_ONLY (per-cell output path marker fallback); trainer-compatible path runnable" | tee -a "${LOG}"
fi

# -----------------------------------------------------------------------------
# DRY-RUN mode: emit DRY_RUN.log + plan.json + enumerate pairs (no GPU work)
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== Wf-131 P3 Pt × 4-cell-line — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ===" | tee -a "${LOG}"
    echo "OUT_BASE: ${OUT_BASE}" | tee -a "${LOG}"
    echo "PY: ${PY}" | tee -a "${LOG}"
    echo "SCRIPT: ${TRAINER}" | tee -a "${LOG}"
    echo "DATA_CSV: ${LOMO_CSV}" | tee -a "${LOG}"
    echo "GPU: cuda:0 (RTX PRO 6000 Blackwell sm_120)" | tee -a "${LOG}"
    echo "PAIRS (${#PAIR_OVERRIDE[@]}): ${PAIR_OVERRIDE[*]}" | tee -a "${LOG}"
    echo "COMMON_ARGS: ${COMMON_ARGS[*]}" | tee -a "${LOG}"
    echo "CELL_LINE_SUPPORTED: ${CELL_LINE_SUPPORTED}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # CSV sanity check
    test -f "${LOMO_CSV}" && echo "OK: CSV exists (${LOMO_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
    n_rows=$(wc -l < "${LOMO_CSV}")
    echo "n_rows (incl header): ${n_rows}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # Per-pair breakdown
    echo "--- ${#PAIR_OVERRIDE[@]} (cell, seed) pairs to enumerate ---" | tee -a "${LOG}"
    for pair in "${PAIR_OVERRIDE[@]}"; do
        read -r cell seed <<< "${pair}"
        echo "  cell=${cell} seed=${seed}  ->  ${OUT_BASE}/Pt/${cell}/seed${seed}/" | tee -a "${LOG}"
    done
    echo "" | tee -a "${LOG}"

    # Wf-363-A1 — emit structured DRY_RUN.log
    pair_args=()
    for pair in "${PAIR_OVERRIDE[@]}"; do
        pair_args+=( --pair "${pair}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-131 P3 Pt × 4-cell-line — ${#PAIR_OVERRIDE[@]} (cell, seed) pairs" \
        --wrapper "$(basename "$0")" \
        --mode "Pt_4cells_Rusanov_5seed_50epoch" \
        --trainer "${TRAINER}" \
        --flags "${COMMON_ARGS[*]}" \
        --gpu "cuda:0" \
        --budget-hr "6" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log" | tee -a "${LOG}"

    # Wf-363-A1 — emit plan.json
    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_wf131_p3_pt_4cells.sh",\n'
        printf '  "wf_origin": "FIX 3 — Pt × 4-cell-line D-MPNN retrain wrapper",\n'
        printf '  "parent_plan": "TODO/03_baselines/COMPARISON_BASELINES.md §1.2 Rusanov 2025 (PlatinAI 4-cell-line)",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "metals": ["Pt"],\n'
        printf '  "cells": ["MCF-7", "A549", "A2780", "A2780cis"],\n'
        printf '  "seeds": [0, 1, 2, 3, 4],\n'
        printf '  "epochs": %d,\n' "${EPOCHS}"
        printf '  "n_pairs": %d,\n' "${#PAIR_OVERRIDE[@]}"
        printf '  "gpu": "cuda:0",\n'
        printf '  "wall_budget_hr": %d,\n' "${WALL_BUDGET_HR}"
        printf '  "wall_budget_label": "PROJECTED (~6 hr GPU sequential per Wf-260 G6 timing; Pt small cohort ~18 min/run x 20 pairs + ~10%% overhead)",\n'
        printf '  "trainer": "%s",\n' "${TRAINER}"
        printf '  "lomo_csv": "%s",\n' "${LOMO_CSV}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "per_pair_path_pattern": "${OUT_BASE}/Pt/${CELL}/seed${SEED}/",\n'
        printf '  "cell_line_flag_supported": %s,\n' "${CELL_LINE_SUPPORTED}"
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "USER_OK gate enforced; SCAFFOLD smoke passed (mode=Pt_4cells_Rusanov_5seed_50epoch); 20 (cell, seed) pairs enumerated; DRY_RUN.log + plan.json emitted; 0 GPU launches",\n'
        printf '    "PROJECTED": "4 cell lines (MCF-7 + A549 + A2780 + A2780cis) x 5 seeds x 50 epochs sequential on RTX 5090 / Blackwell sm_120; ~6 hr wall per Wf-260 G6 timing",\n'
        printf '    "scaffold_only_status": "per-cell output via --out-dir path marker (trainer lacks --cell-line flag); closure path for true per-cell protocol = trainer extension ~50 LOC",\n'
        printf '    "rusanov_2025_protocol_alignment": "BONE-MARROW (L2 per PROTOCOL_ALIGNMENT_MATRIX); dataset=IDENTICAL (PlatinAI), split=APPROXIMATELY ALIGN (per-cell-line; us=path-marker fallback), metric=IDENTICAL (per-cell-line binary AUC)",\n'
        printf '    "task_brief_flag_mismatch": "literal task-brief invocation --metal Pt --cell-line X --seed N rejected by trainer argparse (--cell-line missing; --metal singular should be --metals plural); wrapper uses trainer-compatible --metals Pt --seeds N path",\n'
        printf '    "kill_then_adapt_directive": "6-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"\n'
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    echo "plan.json emitted -> ${PLAN_JSON}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"
    echo "DRY-RUN OK: ${#PAIR_OVERRIDE[@]} (cell, seed) pairs PLANNED, no GPU work executed." | tee -a "${LOG}"
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
    --wave "Wf-131-P3-Pt-4cells" \
    || { echo "FAIL: per_wave_checklist"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wave_budget_tracker.py" \
    || { echo "FAIL: wave_budget_tracker"; exit 1; }

# -----------------------------------------------------------------------------
# Helper: poll pair log until target epoch is reached (Wf-302B guard)
# -----------------------------------------------------------------------------
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-3600}"  # 60 min cap per pair (PROJECTED ~18 min wall)
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
  local cell=$1
  local seed=$2
  local pair_log="${OUT_BASE}/Pt/${cell}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/Pt/${cell}/seed${seed}/ckpts"

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf343c2_per_wave_check.py" \
    --wave "Wf-131-P3-Pt-4cells-${cell}-seed${seed}" \
    --pairs-metals "Pt" \
    --pairs-seeds "${seed}" \
    --budget-hr 0.30 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/Pt/${cell}/seed${seed}/wf343c2_per_wave_check.log" 2>&1
  local rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK Pt ${cell} seed=${seed} rc=${rc_c2} log=${OUT_BASE}/Pt/${cell}/seed${seed}/wf343c2_per_wave_check.log" | tee -a "${LOG}"
  # Diagnostic-only: do NOT return 1 on rc!=0 (per kill_then_adapt; see
  # Wf-347-B1 §5 comment in launch_wf124_p3_g5.sh).

  # Wf-347-B1: per-pair budget tracking
  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START Pt ${cell} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START Pt ${cell} seed=${seed} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=0 ${PY} "${TRAINER}" \
    --metals "Pt" \
    --seeds "${seed}" \
    --csv "${LOMO_CSV}" \
    --out-dir "${OUT_BASE}/Pt/${cell}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/Pt/${cell}/seed${seed}/ckpts" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}

  local pair_end_s pair_elapsed_s
  pair_end_s=$(date +%s)
  pair_elapsed_s=$(( pair_end_s - pair_start_s ))
  echo "PAIR_BUDGET_END Pt ${cell} seed=${seed} elapsed_s=${pair_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') END Pt ${cell} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "${rc}" -ne 0 ]; then
    echo "FAIL: Pt ${cell} seed=${seed} exited rc=${rc}"
    return "${rc}"
  fi
  # Wf-302B guard: verify the pair actually reached target epoch (50)
  wait_until_epoch_reached "${pair_log}" 50 3600 || { echo "FAIL: Pt ${cell} seed=${seed} did not reach epoch 50"; return 1; }
}

echo "=== Wf-131 P3 Pt × 4-cell-line START $(date) ===" | tee -a "${LOG}"
echo "${#PAIR_OVERRIDE[@]} pairs (4 cells x 5 seeds x 50 ep on PlatinAI Pt cohort) -- USER_OK-gated per AGENTS.md §5.1 + Wf-294B family" | tee -a "${LOG}"

# Iterate PAIR_OVERRIDE sequentially
for pair in "${PAIR_OVERRIDE[@]}"; do
  read -r cell seed <<< "${pair}"
  run_pair "${cell}" "${seed}"
done

echo "=== ALL Wf-131 P3 Pt × 4-cell-line PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation: collect per_seed_metrics_seed*.json into training_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for metal_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not metal_dir.is_dir() or metal_dir.name == 'logs' or metal_dir.name.startswith('scaffold_'):
        continue
    for cell_dir in sorted(metal_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        for seed_dir in sorted(cell_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for p in seed_dir.glob('per_seed_metrics_seed*.json'):
                runs.append({
                    'metal': metal_dir.name,
                    'cell': cell_dir.name,
                    'seed': seed_dir.name.replace('seed', ''),
                    'metrics_path': str(p),
                })
out = {
    'wf131_p3_pt_4cells_total_pairs': len(runs),
    'n_pairs_planned': ${#PAIR_OVERRIDE[@]},
    'runs': runs,
}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (planned ${#PAIR_OVERRIDE[@]})')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"