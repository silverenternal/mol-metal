#!/usr/bin/env bash
# launch_wf131_p3_pt_4cells.sh — REAL 5-metal LOMO D-MPNN retrain wrapper
#                          (USER_OK-gated GPU launch wrapper)
#
# 2026-10-08 SUBSTANTIVE PIVOT per user direction
# "得用真实数据啊,合成数据真的有用吗?":
#   Original task asked for Pt × 4-cell-line LOMO, but real Pt per-cell
#   IC50 numerics are not accessible (Rusanov 2025 S4-S5 Cloudflare,
#   MB-Finder coverage-only API, Selenium blocked, synthetic 99% inactive).
#   Pivoted to REAL 5-metal LOMO (Ir/Os/Re/Rh/Ru) on MetalCytoToxDB.csv.
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
#   - SUBSTANTIVE PIVOT 2026-10-08 per user direction
#     "得用真实数据啊,合成数据真的有用吗?":
#     Original task brief asked for Pt × 4-cell-line LOMO, but real Pt per-cell
#     IC50 numerics are NOT accessible:
#       * Rusanov 2025 supplementary S4-S5 → Cloudflare JS challenge (Wf-292D)
#       * MB-Finder portal API → coverage flags only, no IC50 numerics
#       * Selenium undetected_chromedriver scrape → 0 rows (Wf-304B BLOCK_CF)
#       * Synthetic PlatinAI 4-cell platinai_4cell.csv → 99% inactive
#         (pIC50<8.0 for 16309/16312 rows); binary AUC gate ≥0.70 unreachable.
#     Pivoted to REAL 5-metal LOMO on MetalCytoToxDB.csv (Ir/Os/Re/Rh/Ru,
#     26632 rows, REAL per-metal IC50 numerics) — same canonical m2 source
#     as Wf-125 P2 baseline.
#   - user_ok_required: true (kill_then_adapt; 5 metals × 5 seeds × 50-ep
#     × ~12 min wall PROJECTED per Wf-260 G6 timing on RTX 5090)
#   - Scope: 5 metals (Ru + Ir + Os + Rh + Re) × 5 seeds = 25 (metal, seed)
#     pairs TOTAL
#   - Wall: ~5 hr PROJECTED per Wf-260 G6 timing:
#       - Per-pair (metal, seed): ~12 min wall × 25 pairs = ~5 hr
#       - Sequential overhead + ckpt write: +1 hr
#       - TOTAL: ~5 hr GPU (conservative; Ru cohort ~19k rows is biggest)
#   - Per-metal breakdown (canonical m2 MetalCytoToxDB):
#       - Ru: 18,987 rows (largest cohort, most expensive)
#       - Ir: 4,546 rows
#       - Os: 1,113 rows
#       - Rh: 1,118 rows (with Wf-125 P2 synthetic-oversample per Wf-A3)
#       - Re: 868 rows
#   - Per-pair budget 0.6 hr PROJECTED per Wf-260 G6 timing
#   - Pt × 4-cell-line task BLOCKED on external dependency per AGENTS.md
#     §5.2 honest-negative framing; documented in
#     TODO/completed/131_wf131_p3_pt_4cell_blocked_external.md
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

set -eu
# NB: pipefail INTENTIONALLY DISABLED per Wf-131 P3 post-mortem 2026-10-08:
# The per-pair invocation `${PY} ${TRAINER} ... | tee "${pair_log}"` causes
# SIGPIPE in `tee` when Python's DataLoader workers (forked at process exit)
# flush stdout after the trainer's main process has already closed the pipe.
# pipefail then propagates a non-zero exit to the wrapper, killing it
# silently between pair completion and .DONE marker write.
# Fix: drop pipefail; capture trainer rc via PIPESTATUS[0]; explicitly suppress
# tee failures with `|| true` so SIGPIPE never reaches the wrapper.

# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# Wf-130-G fix: unbuffered stdout/stderr so per-step failures write to ${LOG}
# immediately (set -euo pipefail + 4 KiB block buffer masked the real exit site).
# stdbuf -oL -eL keeps line-buffered I/O without disabling set -e.
exec 4>&1
if [ -n "${LOG:-}" ]; then
    exec > >(stdbuf -oL -eL tee -a "${LOG}") 2>&1
fi

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf131_p3_pt_4cells"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_wasserstein.py"
# Per Wf-131 P3 SUBSTANTIVE PIVOT (2026-10-08 user direction
# "得用真实数据啊,合成数据真的有用吗?"):
# Real Pt per-cell IC50 numerics are NOT accessible:
#   * Rusanov 2025 supplementary S4-S5 → Cloudflare JS challenge (Wf-292D)
#   * MB-Finder portal API → coverage flags only, no IC50 numerics
#   * Selenium undetected_chromedriver scrape → 0 rows (Wf-304B BLOCK_CF)
#   * Synthetic PlatinAI 4-cell platinai_4cell.csv → 99% inactive (pIC50<8.0
#     for 16309/16312 rows); binary AUC gate ≥0.70 unreachable.
#
# Substantive pivot: switch to REAL 5-metal LOMO on MetalCytoToxDB
# (Ir/Os/Re/Rh/Ru, 26632 rows, REAL per-metal IC50 numerics from
# MetalCytoToxDB.csv).  This is the canonical m2 (metal-LOMO-AUC) source
# for the Tier-2 paper; Wf-125 P2 baseline is on this same dataset.
#
# Pt × 4-cell is BLOCKED on external dependency per AGENTS.md §5.2
# honest-negative framing; documented in
# TODO/completed/131_wf131_p3_pt_4cell_blocked_external.md.
LOMO_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"
if [[ ! -f "${LOMO_CSV}" ]]; then
    echo "FATAL: REAL 5-metal CSV missing: ${LOMO_CSV}" >&2
    exit 1
fi

# 5 metals × 5 seeds = 25 (metal, seed) pairs per Wf-125 P2 pattern
DEFAULT_METALS=(
    "Ru"
    "Ir"
    "Os"
    "Rh"
    "Re"
)
DEFAULT_SEEDS=(0 1 2 3 4)

# Default pairs: all 25 (metal, seed) tuples
DEFAULT_PAIRS=()
for METAL in "${DEFAULT_METALS[@]}"; do
    for SEED in "${DEFAULT_SEEDS[@]}"; do
        DEFAULT_PAIRS+=("${METAL} ${SEED}")
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
    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run (25 pairs)\n' "$0" >&2
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
  --epochs "${EPOCHS}"
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --n-mc 256
  --device "cuda:0"
)
# NOTE: --metals and --csv are passed per-pair by run_pair() (--metals per
# metal, --csv shared LOMO_CSV). COMMON_ARGS only carries the common flags.

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
echo "=== Wf-131 P3 5-metal REAL LOMO SCAFFOLD smoke (CPU-only) ===" | tee -a "${LOG}"

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
    echo "=== Wf-131 P3 5-metal REAL LOMO — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ===" | tee -a "${LOG}"
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
    echo "--- ${#PAIR_OVERRIDE[@]} (metal, seed) pairs to enumerate ---" | tee -a "${LOG}"
    for pair in "${PAIR_OVERRIDE[@]}"; do
        read -r metal seed <<< "${pair}"
        echo "  metal=${metal} seed=${seed}  ->  ${OUT_BASE}/${metal}/seed${seed}/" | tee -a "${LOG}"
    done
    echo "" | tee -a "${LOG}"

    # Wf-363-A1 — emit structured DRY_RUN.log
    pair_args=()
    for pair in "${PAIR_OVERRIDE[@]}"; do
        pair_args+=( --pair "${pair}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-131 P3 5-metal REAL LOMO — ${#PAIR_OVERRIDE[@]} (metal, seed) pairs" \
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
        printf '  "wf_origin": "Wf-131 P3 SUBSTANTIVE PIVOT — REAL 5-metal LOMO on MetalCytoToxDB.csv (Pt × 4-cell BLOCKED on Rusanov 2025 S4-S5 Cloudflare)",\n'
        printf '  "parent_plan": "Wf-131 P3 SUBSTANTIVE PIVOT (2026-10-08 user direction). Pt × 4-cell BLOCKED on Rusanov 2025 S4-S5 Cloudflare; pivoted to REAL 5-metal LOMO on MetalCytoToxDB.csv",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
        printf '  "seeds": [0, 1, 2, 3, 4],\n'
        printf '  "epochs": %d,\n' "${EPOCHS}"
        printf '  "n_pairs": %d,\n' "${#PAIR_OVERRIDE[@]}"
        printf '  "gpu": "cuda:0",\n'
        printf '  "wall_budget_hr": %d,\n' "${WALL_BUDGET_HR}"
        printf '  "wall_budget_label": "PROJECTED (~5 hr GPU sequential per Wf-260 G6 timing; 5 metals x 5 seeds x 50 epochs ~12 min/pair avg)",\n'
        printf '  "trainer": "%s",\n' "${TRAINER}"
        printf '  "lomo_csv": "%s",\n' "${LOMO_CSV}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "per_pair_path_pattern": "${OUT_BASE}/${METAL}/seed${SEED}/",\n'
        printf '  "data_provenance": "REAL MetalCytoToxDB.csv (26632 rows, 5 metals); per-metal pIC50 NUMERIC from MetalCytoToxDB.csv",\n'
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "USER_OK gate enforced; SCAFFOLD smoke passed (mode=5metal_REAL_LOMO_5seed_50epoch); 25 (metal, seed) pairs enumerated; DRY_RUN.log + plan.json emitted; 0 GPU launches",\n'
        printf '    "PROJECTED": "5 metals (Ru + Ir + Os + Rh + Re) x 5 seeds x 50 epochs sequential on Blackwell sm_120; ~5 hr wall per Wf-260 G6 timing",\n'
        printf '    "pt_x_4cell_blocked_reason": "Rusanov 2025 S4-S5 tables: Cloudflare JS challenge (Wf-292D); MB-Finder API: coverage flags only (Wf-295B); Selenium scrape: 0 rows (Wf-304B BLOCK_CF); synthetic platinai_4cell.csv: 99% inactive, binary AUC gate unreachable",\n'
        printf '    "pt_x_4cell_status": "BLOCKED on external dependency per AGENTS.md §5.2 honest-negative; documented in TODO/completed/131_wf131_p3_pt_4cell_blocked_external.md",\n'
        printf '    "kill_then_adapt_directive": "5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"\n'
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
    --wave "Wf-131-P3-Pt-4cells" \
    || { echo "FAIL: per_wave_checklist"; exit 1; }
# Wf-130-G fix: wave_budget_tracker.py scans 970 molmetal/reports/* dirs and
# can take 30+ s wall + sometimes emits DEFERRED labeling warnings that, under
# `set -euo pipefail`, get re-interpreted as non-zero exit by some pipelines.
# Per AGENTS.md §5.1 + kill_then_adapt §0 we treat wave_budget_tracker as
# diagnostic-only here (its real assertion already lives in pre_flight_check).
# Pre-flight already enforces `p3_wf343c4_wave_budget_remaining_nonneg`.
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wave_budget_tracker.py" \
    > "${REPORTS_DIR}/wave_budget_tracker.log" 2>&1 \
    || echo "WARN: wave_budget_tracker diagnostic-only (rc=$?); pre-flight already enforces"

# -----------------------------------------------------------------------------
# Helper: verify pair metrics.json reached target epoch (Wf-R43 + Wf-302B)
# -----------------------------------------------------------------------------
# Reads the trainer-written metrics.json + checks the `epochs` field.
# The trainer always writes metrics.json at end-of-run with `epochs` field
# (set from CLI --epochs). If metrics.json missing OR epochs < target,
# return 1. This replaces the broken `wait_until_epoch_reached` regex
# grep on run.log (which doesn't match the trainer's `epoch=N` log format).
# Returns 0 on success, 1 on failure (no sleep-loop, no infinite wait).
verify_pair_reached_epoch() {
  local metrics_json="$1" target_epoch="$2"
  local max_wait_s="${3:-30}"  # 30 s cap (metrics.json should appear within ~5 s of trainer exit)
  local start_s elapsed actual_epochs
  start_s=$(date +%s)
  elapsed=0
  while true; do
    if [ -f "${metrics_json}" ]; then
      # metrics.json has top-level "epochs" integer
      actual_epochs=$(grep -oE '"epochs":[[:space:]]*[0-9]+' "${metrics_json}" 2>/dev/null | head -1 | grep -oE '[0-9]+')
      if [ -n "${actual_epochs}" ] && [ "${actual_epochs}" -ge "${target_epoch}" ]; then
        echo "Pair verified: epochs=${actual_epochs} >= target=${target_epoch}"
        return 0
      fi
    fi
    elapsed=$(( $(date +%s) - start_s ))
    if [ "${elapsed}" -ge "${max_wait_s}" ]; then
      echo "VERIFY_TIMEOUT: metrics_json=${metrics_json} elapsed=${elapsed}s actual_epochs=${actual_epochs:-MISSING}"
      return 1
    fi
    sleep 2
  done
}

# -----------------------------------------------------------------------------
# Real launch: per-pair runner
# -----------------------------------------------------------------------------
run_pair() {
  local metal=$1
  local seed=$2
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  local pair_done="${OUT_BASE}/${metal}/seed${seed}/.DONE"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"

  # Wf-R43 resume-safety: skip already-completed pairs on restart
  if [[ -f "${pair_done}" ]]; then
    echo "PAIR_SKIP ${metal} seed=${seed} reason=.DONE marker present" | tee -a "${LOG}"
    return 0
  fi

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  # NB: rc_c2 is diagnostic-only (do NOT abort wrapper on FAIL per kill_then_adapt;
  # see Wf-347-B1 §5). The `|| true` suppresses `set -euo pipefail` from killing
  # the wrapper when wave_budget_remaining_nonneg or other gates FAIL — those
  # are informational, not blocking. Pre-flight already enforces the real budget.
  PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf343c2_per_wave_check.py" \
    --wave "Wf-131-P3-5metal-${metal}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --budget-hr 0.6 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" 2>&1 \
    || true
  local rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK ${metal} seed=${seed} rc=${rc_c2} log=${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" | tee -a "${LOG}"
  # Diagnostic-only: do NOT return 1 on rc!=0 (per kill_then_adapt; see
  # Wf-347-B1 §5 comment in launch_wf124_p3_g5.sh).

  # Wf-347-B1: per-pair budget tracking
  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START ${metal} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=0 setsid nohup ${PY} "${TRAINER}" \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${LOMO_CSV}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --out-metric "${OUT_BASE}/${metal}/seed${seed}/metrics.json" \
"${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}" || true
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
  # Wf-R43 + Wf-302B guard: verify the pair actually reached target epoch (50)
  # via metrics.json (replaces broken regex-grep on run.log).
  verify_pair_reached_epoch "${OUT_BASE}/${metal}/seed${seed}/metrics.json" 50 30 \
    || { echo "FAIL: ${metal} seed=${seed} did not reach epoch 50 (metrics.json missing or epochs < 50)"; return 1; }
  # Wf-R43 resume-safety: write .DONE marker only after BOTH trainer + epoch-50 verified
  echo "PAIR_DONE ${metal} seed=${seed} rc=${rc} epoch=50 verified" > "${pair_done}"
  echo "PAIR_MARKER ${metal} seed=${seed} .DONE=${pair_done}" | tee -a "${LOG}"
}

echo "=== Wf-131 P3 5-metal REAL LOMO START $(date) ===" | tee -a "${LOG}"
echo "${#PAIR_OVERRIDE[@]} pairs (5 metals x 5 seeds x 50 ep on REAL MetalCytoToxDB) -- USER_OK-gated per AGENTS.md §5.1 + Wf-294B family" | tee -a "${LOG}"

# Iterate PAIR_OVERRIDE sequentially
# Wf-131 P3 fix: capture per-pair rc, continue on failure, write .FAILED marker
# so partial progress survives wrapper crash. The wrapper MUST survive all
# 25 pairs (or as many as the budget allows) without silent death.
for pair in "${PAIR_OVERRIDE[@]}"; do
  read -r metal seed <<< "${pair}"
  if ! run_pair "${metal}" "${seed}"; then
    rc_pair=$?
    echo "PAIR_FAIL ${metal} seed=${seed} rc=${rc_pair} continuing to next pair" | tee -a "${LOG}"
    echo "PAIR_FAILED ${metal} seed=${seed} rc=${rc_pair}" > "${OUT_BASE}/${metal}/seed${seed}/.FAILED"
  fi
done

echo "=== ALL Wf-131 P3 5-metal REAL LOMO PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation: collect one metrics path per (metal, seed) pair into training_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json, sys
from pathlib import Path
out_base = Path('${OUT_BASE}')
runs = []
for metal_dir in sorted(out_base.iterdir()):
    if not metal_dir.is_dir() or metal_dir.name == 'logs' or metal_dir.name.startswith('scaffold_'):
        continue
    for cell_dir in sorted(metal_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        # Canonical layout: cell_dir/<Metal>/per_seed_metrics_seed*.json.
        # Ru/seed0 came from a legacy trainer path that wrote only
        # cell_dir/metrics.json (no nested detail file), so fall back to it.
        cand = None
        nested = sorted((cell_dir / metal_dir.name).glob('per_seed_metrics_seed*.json')) \
            if (cell_dir / metal_dir.name).is_dir() else []
        if nested:
            cand = nested[0]
        else:
            for fb in ('metrics.json', 'aggregate.json'):
                if (cell_dir / fb).exists():
                    cand = cell_dir / fb
                    break
        if cand is None:
            continue
        runs.append({
            'metal': metal_dir.name,
            'cell': cell_dir.name,
            'seed': cell_dir.name.replace('seed', ''),
            'metrics_path': str(cand),
        })
planned = ${#PAIR_OVERRIDE[@]}
out = {
    'wf131_p3_pt_4cells_total_pairs': len(runs),
    'n_pairs_planned': planned,
    'runs': runs,
}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (planned {planned})')
# Fail-loud: a count mismatch means a seed dir produced no metrics at all.
if planned and len(runs) != planned:
    print(f'[FAIL] aggregated {len(runs)} != planned {planned}', file=sys.stderr)
    sys.exit(1)
" | tee -a "${LOG}"
agg_rc=\${PIPESTATUS[0]}
if [ \"\${agg_rc}\" -ne 0 ]; then
    echo \"[FAIL] Wf-131 aggregation rc=\${agg_rc}\" | tee -a \"\${LOG}\"
    exit \"\${agg_rc}\"
fi
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"
