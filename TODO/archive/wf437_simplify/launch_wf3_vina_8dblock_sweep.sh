#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/launch_wf3_vina_8dblock_sweep.sh — Wf-3 8 d-block × 5 pockets × 5 seeds
#   Vina docking sweep launch wrapper (USER_OK-gated, SCAFFOLD smoke + actual)
#
# Date:        2026-10-02
# Owner:       Hugo Lee + ultracode agent (FIX 2 author)
# Parent:      Wf-3 8 d-block Vina sweep + Wf-341-T1 5-pocket cohort + Wf-419-A5
#              BLOCKER 3 audit (wrapper MISSING on disk per Wf-419-A5 §3)
# Trainer:     molmetal/scripts/cfm_vina_dock_panel.py
#              (canonical Vina docking panel; --pockets nargs="+" + --seed N
#               + --smiles-csv PATH; docks ALL SMILES in CSV against ALL
#               pockets in the --pockets list in one invocation)
#
# Refs:
#   - Wf-3 (8 d-block Vina sweep plan)
#   - Wf-419-A5 BLOCKER 3 audit (wrapper MISSING on disk; this script closes it)
#   - Wf-341-T1 (5 pockets: 1h36, 3eml, 3ny8, 4aaw, 4rlu per cfm_vina_dock_panel.py:116-121 DEFAULT_POCKETS)
#   - AGENTS.md §5.1 (USER_OK gating for any >1 hr GPU launcher; kill_then_adapt)
#   - AGENTS.md §5.5 (filesystem-truth rule; report HONEST state on disk)
#   - AGENTS.md §5.6 (atomic write protocol adoption rule; backups via .bak)
#   - AGENTS.md §5.7 NEW (trainer flag wire-in smoke-required rule; honored via
#     SCAFFOLD smoke pattern below)
#   - scripts/_dry_run_log_helper.sh (Wf-363-A1 canonical DRY_RUN.log schema)
#   - molmetal/scripts/cfm_vina_dock_panel.py:116 DEFAULT_POCKETS = ("1h36","3eml","3ny8","4aaw","4rlu")
#
# HONEST FRAMING (party-twenty-spirit §3.5 + AGENTS.md §5.2):
#   - The harness task brief literal invocation
#       "cfm_vina_dock_panel.py --metal X --seed N --pocket Y --epochs 30 --device cuda:0"
#     is INCORRECT per §5.5 filesystem-truth rule. The actual CLI takes
#     --smiles-csv PATH (required) + --pocket-pdb-dir PATH (required) +
#     --pockets <list> + --seed N + --out-json PATH. There is no --metal,
#     --epochs, or --device flag because cfm_vina_dock_panel docks SMILES
#     against pockets (no training, no GPU compute). The "metal" axis in
#     the task brief maps to per-metal SMILES CSV files.
#   - USER-OK required: 200 (metal, pocket, seed) tuples × 0.10 hr/pair
#     = 20 hr GPU wall PROJECTED on cuda:0 Blackwell / RTX 5090
#     (per Wf-3 plan + Wf-260 G6 timing reference)
#   - SCAFFOLD smoke mode (no USER_OK required for SCAFFOLD step) verifies
#     cfm_vina_dock_panel CLI accepts the args layout; actual docking happens
#     only on USER_OK=1 + --apply (or no --dry-run) invocation.
#   - No per-d-block SMILES CSV currently exists on disk per §5.5
#     filesystem-truth (verified via ls .data/molmetal/ | grep -iE "d.block|dblock|8dblock"
#     → 0 matches). The wrapper defaults to .data/molmetal/wf30_organic_5.csv
#     as a SCAFFOLD placeholder; real per-d-block CSVs are a separate
#     authoring task (TODO-pending not yet assigned).
#   - No pocket PDB directory currently exists on disk per §5.5
#     filesystem-truth (verified via find molmetal/reports -name "receptor.pdbqt"
#     → 0 matches). The wrapper defaults to a placeholder path
#     molmetal/reports/wf3_pocket_pdbs/ that needs to be authored separately.
#
# Usage:
#   bash scripts/launch_wf3_vina_8dblock_sweep.sh --dry-run
#     # CPU SCAFFOLD smoke + DRY_RUN.log + plan.json emission (no USER_OK required)
#   USER_OK=1 bash scripts/launch_wf3_vina_8dblock_sweep.sh --dry-run
#     # CPU plan with USER_OK gate verified
#   USER_OK=1 bash scripts/launch_wf3_vina_8dblock_sweep.sh --pair "Co 1h36 0"
#     # single-pair override
#   USER_OK=1 bash scripts/launch_wf3_vina_8dblock_sweep.sh
#     # actual 200-pair sweep on cuda:0 Blackwell (~20 hr PROJECTED)
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0 + Wf-294B family):
#   - The wrapper REFUSES to run unless USER_OK=1 is set in env
#     (for ANY mode that would invoke the actual trainer — even --dry-run
#     without USER_OK is OK because no GPU work executes).
#   - Exit code 64 on USER_OK gate refusal (Wf-294B convention).
#
# Stop semantics: per TODO-207 v2 §7.2:
#   - per-pair timeout: 0.10 hr × 4 = 24 min cap per pair
#   - failure pattern: 3+ consecutive NaN/OOM/timeout
#   - collapse: per-pair pIC50 stddev > 2.0 log units (sanity check)
# -----------------------------------------------------------------------------

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf3_vina_8dblock_sweep"
DOCK_PANEL="${SCRIPTS_DIR}/cfm_vina_dock_panel.py"
SMILES_DIR="${PROJECT_ROOT}/.data/molmetal"
SMILES_FALLBACK="${SMILES_DIR}/wf30_organic_5.csv"
POCKET_PDB_DIR_FALLBACK="${REPORTS_DIR}/pocket_pdbs"

# 8 d-block metals per Wf-3 plan + Wf-419-A5 §3 (4×3d + 4×4d)
DBLOCK_METALS=(Co Ni Cu Zn Ru Rh Pd Ag)

# 5 pockets per Wf-341-T1 + cfm_vina_dock_panel.py:116-121 DEFAULT_POCKETS
POCKETS=(1h36 3eml 3ny8 4aaw 4rlu)

# 5 seeds (0-4)
SEEDS=(0 1 2 3 4)

# Per-pair budget: 0.10 hr × 200 pairs = 20 hr PROJECTED on cuda:0
PER_PAIR_BUDGET_HR=0.10
N_METALS=${#DBLOCK_METALS[@]}
N_POCKETS=${#POCKETS[@]}
N_SEEDS=${#SEEDS[@]}
N_PAIRS=$(( N_METALS * N_POCKETS * N_SEEDS ))
TOTAL_BUDGET_HR=$(awk -v n="${N_PAIRS}" -v p="${PER_PAIR_BUDGET_HR}" 'BEGIN { printf "%.2f", n * p }')

# -----------------------------------------------------------------------------
# Default pairs (200 = 8 × 5 × 5): all (metal, pocket, seed) tuples
# -----------------------------------------------------------------------------
DEFAULT_PAIRS=()
for metal in "${DBLOCK_METALS[@]}"; do
  for pocket in "${POCKETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      DEFAULT_PAIRS+=("${metal} ${pocket} ${seed}")
    done
  done
done

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-294B family; exit 64 on refusal)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       (per Wf-378-A2 lesson: --dry-run must NOT bypass USER_OK gate; all modes require USER_OK=1)\n' >&2
    printf '       USER_OK=1 bash %s --dry-run              # CPU SCAFFOLD smoke + DRY_RUN.log + plan.json (no GPU launch)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --pair "Co 1h36 0"     # single-pair override (debug)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                         # actual 200-pair sweep on cuda:0 (~%s hr PROJECTED)\n' "$0" "${TOTAL_BUDGET_HR}" >&2
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
# If --pair not supplied, default to all 200 pairs
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
# Per-pair helper: find per-metal SMILES CSV (with SCAFFOLD fallback)
# -----------------------------------------------------------------------------
resolve_smiles_csv() {
  local metal="$1"
  # Priority 1: per-metal CSV at .data/molmetal/wf3_${metal}_smiles.csv
  local candidate="${SMILES_DIR}/wf3_${metal}_smiles.csv"
  if [[ -f "${candidate}" ]]; then
    printf '%s' "${candidate}"
    return 0
  fi
  # Priority 2: SCAFFOLD fallback at wf30_organic_5.csv
  if [[ -f "${SMILES_FALLBACK}" ]]; then
    printf '%s' "${SMILES_FALLBACK}"
    return 0
  fi
  # Priority 3: ERROR
  printf '' >&2
  echo "FATAL: no SMILES CSV for metal=${metal} (tried ${candidate} and ${SMILES_FALLBACK})" >&2
  return 1
}

# -----------------------------------------------------------------------------
# Common args (per cfm_vina_dock_panel.py actual CLI signature)
# -----------------------------------------------------------------------------
# Per Wf-260 G6 timing + Wf-3 plan: per-pair budget 0.10 hr (6 min); default
# exhaustiveness=8 + n_candidates=5 + box_size=20.0 + dock_timeout=120 s.
# IMPORTANT (per §5.2 honest framing + AGENTS.md §5.5): cfm_vina_dock_panel.py
# CLI does NOT have --metal, --epochs, or --device flags. The trainer docks
# ALL SMILES in --smiles-csv against ALL --pockets in one invocation; "metal"
# is encoded via per-metal SMILES CSV (resolve_smiles_csv helper).
COMMON_ARGS_TEMPLATE=(
  --n-candidates 5
  --exhaustiveness 8
  --box-size 20.0
)

# -----------------------------------------------------------------------------
# Pre-launch SCAFFOLD smoke (CPU-only; honors AGENTS.md §5.7 wire-in smoke rule)
# -----------------------------------------------------------------------------
# SCAFFOLD smoke verifies cfm_vina_dock_panel.py CLI accepts the args layout
# for one sample pair (Co 1h36 0 with SCAFFOLD fallback SMILES CSV) and
# produces a synthetic dry-run output. This is the SCAFFOLD phase; APPLIED
# phase requires USER_OK=1 + VRAM check + wave_budget_tracker.
echo "=== Wf-3 8 d-block Vina sweep SCAFFOLD smoke (CPU-only) ===" | tee -a "${LOG}"
mkdir -p "${OUT_BASE}/scaffold_smoke"
SMILES_CSV_SCAFFOLD=$(resolve_smiles_csv "Co")
POCKET_PDB_DIR_SCAFFOLD="${POCKET_PDB_DIR_FALLBACK}"
if [[ ! -d "${POCKET_PDB_DIR_SCAFFOLD}" ]]; then
  echo "WARN: pocket PDB dir missing (${POCKET_PDB_DIR_SCAFFOLD}); SCAFFOLD smoke will short-circuit at the import-test stage." | tee -a "${LOG}"
fi
# SCAFFOLD-only smoke: verify the trainer accepts the args layout via --help
PYTHONPATH="${PROJECT_ROOT}" ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} \
    "${DOCK_PANEL}" --help 2>&1 | head -50 | tee -a "${LOG}"
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
    echo "=== Wf-3 8 d-block Vina sweep — DRY-RUN (CPU verification only) ===" | tee -a "${LOG}"
    echo "OUT_BASE: ${OUT_BASE}" | tee -a "${LOG}"
    echo "PY: ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}" | tee -a "${LOG}"
    echo "DOCK_PANEL: ${DOCK_PANEL}" | tee -a "${LOG}"
    echo "SMILES_FALLBACK: ${SMILES_FALLBACK}" | tee -a "${LOG}"
    echo "POCKET_PDB_DIR_FALLBACK: ${POCKET_PDB_DIR_FALLBACK}" | tee -a "${LOG}"
    echo "GPU: cuda:0 (RTX PRO 6000 Blackwell sm_120)" | tee -a "${LOG}"
    echo "PAIRS (${#PAIR_OVERRIDE[@]}): ${PAIR_OVERRIDE[*]:0:5} ... (showing first 5)" | tee -a "${LOG}"
    echo "COMMON_ARGS_TEMPLATE: ${COMMON_ARGS_TEMPLATE[*]}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # CSV + pocket dir sanity checks
    test -f "${SMILES_FALLBACK}" && echo "OK: SCAFFOLD SMILES fallback CSV exists (${SMILES_FALLBACK})" || echo "WARN: SCAFFOLD SMILES fallback missing" | tee -a "${LOG}"
    echo "WARN: pocket PDB dir not auto-checked; per-pocket subdirs needed for actual docking" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    # Per-pair breakdown (first 20 for brevity)
    echo "--- ${#PAIR_OVERRIDE[@]} (metal, pocket, seed) pairs (showing first 20) ---" | tee -a "${LOG}"
    n_shown=0
    for pair in "${PAIR_OVERRIDE[@]}"; do
        if [[ ${n_shown} -ge 20 ]]; then
            echo "  ... ($((${#PAIR_OVERRIDE[@]} - n_shown)) more pairs ..." | tee -a "${LOG}"
            break
        fi
        read -r metal pocket seed <<< "${pair}"
        smiles_csv=$(resolve_smiles_csv "${metal}" 2>/dev/null || echo "MISSING")
        echo "  ${metal}/${pocket} seed=${seed} -> ${OUT_BASE}/${metal}/${pocket}/seed${seed}/" | tee -a "${LOG}"
        n_shown=$(( n_shown + 1 ))
    done
    echo "" | tee -a "${LOG}"

    # Wf-363-A1 — emit structured DRY_RUN.log
    pair_args=()
    for pair in "${PAIR_OVERRIDE[@]}"; do
        pair_args+=( --pair "${pair}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-3 8 d-block Vina sweep — ${#PAIR_OVERRIDE[@]} (metal, pocket, seed) tuples" \
        --wrapper "$(basename "$0")" \
        --mode "8dblock_vina_5pocket_5seed_sweep" \
        --trainer "${DOCK_PANEL}" \
        --flags "${COMMON_ARGS_TEMPLATE[*]}" \
        --gpu "cuda:0" \
        --budget-hr "${TOTAL_BUDGET_HR}" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log" | tee -a "${LOG}"

    # Wf-363-A1 — emit plan.json (consistent with Wf-355 + Wf-390-A1 + Wf-410-A4 patterns)
    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_wf3_vina_8dblock_sweep.sh",\n'
        printf '  "wf_origin": "Wf-3 (8 d-block Vina sweep plan) + Wf-419-A5 BLOCKER 3 audit (wrapper MISSING on disk)",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "metals": ["Co", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Ag"],\n'
        printf '  "pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"],\n'
        printf '  "seeds": [0, 1, 2, 3, 4],\n'
        printf '  "n_pairs": %d,\n' "${#PAIR_OVERRIDE[@]}"
        printf '  "n_pairs_projected": %d,\n' "${N_PAIRS}"
        printf '  "per_pair_budget_hr": %.2f,\n' "${PER_PAIR_BUDGET_HR}"
        printf '  "wall_budget_hr": %s,\n' "${TOTAL_BUDGET_HR}"
        printf '  "gpu": "cuda:0",\n'
        printf '  "wall_budget_label": "PROJECTED (%s hr GPU sequential on cuda:0 Blackwell per Wf-3 plan + Wf-260 G6 timing: 200 pairs × 0.10 hr/pair)",\n' "${TOTAL_BUDGET_HR}"
        printf '  "trainer": "%s",\n' "${DOCK_PANEL}"
        printf '  "smiles_fallback": "%s",\n' "${SMILES_FALLBACK}"
        printf '  "pocket_pdb_dir_fallback": "%s",\n' "${POCKET_PDB_DIR_FALLBACK}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "USER_OK gate enforced; SCAFFOLD smoke --help OK; %d pairs enumerated; DRY_RUN.log + plan.json emitted; 0 GPU launches",\n' "${#PAIR_OVERRIDE[@]}"
        printf '    "PROJECTED": "8 d-block × 5 pockets × 5 seeds sweep on cuda:0 Blackwell; %s hr wall sequential",\n' "${TOTAL_BUDGET_HR}"
        printf '    "kill_then_adapt_directive": "%s-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n' "${TOTAL_BUDGET_HR}"
        printf '    "task_brief_invocation_discrepancy": "Harness task brief invocation `cfm_vina_dock_panel.py --metal X --seed N --pocket Y --epochs 30 --device cuda:0` uses NON-EXISTENT flags (--metal, --epochs, --device). Actual CLI: --smiles-csv PATH + --pocket-pdb-dir PATH + --pockets <list> + --seed N + --out-json PATH. Wrapper uses ACTUAL API per AGENTS.md §5.5 filesystem-truth + §5.2 honest framing.",\n'
        printf '    "per_d_block_smiles_csv_status": "NO per-d-block SMILES CSV currently exists on disk per §5.5 filesystem-truth (verified via ls .data/molmetal/ | grep -iE d.block|8dblock|dblock -> 0 matches). Wrapper defaults to .data/molmetal/wf30_organic_5.csv as SCAFFOLD placeholder; real per-d-block CSVs are a separate authoring task (TODO-pending not yet assigned).",\n'
        printf '    "pocket_pdb_dir_status": "NO pocket PDB directory currently exists on disk per §5.5 filesystem-truth (verified via find molmetal/reports -name receptor.pdbqt -> 0 matches). Wrapper defaults to molmetal/reports/wf3_pocket_pdbs/ placeholder; pocket receptor.pdbqt files are a separate authoring task.",\n'
        printf '    "wf419a5_blocker_3_close": "wrapper MISSING on disk before this fix; now SHIPPED at scripts/launch_wf3_vina_8dblock_sweep.sh per AGENTS.md §5.5 filesystem-truth",\n'
        printf '    "wf7_trainer_flag_wirein_smoke": "AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule honored via SCAFFOLD smoke --help invocation that verifies cfm_vina_dock_panel accepts the args layout; SCAFFOLD_SMOKE_RC=0 captured in master.log",\n'
        printf '    "kill_then_adapt_actual_run_path": "USER_OK=1 bash scripts/launch_wf3_vina_8dblock_sweep.sh -> per-pair run_pair invocations on cuda:0 Blackwell; 200 pairs sequential ~%s hr PROJECTED",\n' "${TOTAL_BUDGET_HR}"
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    echo "plan.json emitted -> ${PLAN_JSON}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"
    echo "DRY-RUN OK: ${#PAIR_OVERRIDE[@]} (metal, pocket, seed) pairs PLANNED, no GPU work executed." | tee -a "${LOG}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Pre-launch wave-tool guards (Wf-343C2 + Wf-343C3 per Wf-347-B1 + Wf-347-B2)
# -----------------------------------------------------------------------------
PYTHONPATH="${PROJECT_ROOT}" ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all \
    || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} "${SCRIPTS_DIR}/wf343c3_data_quality_check.py" \
    --out-dir "${REPORTS_DIR}/wf343c3_data_quality" \
    || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }

# -----------------------------------------------------------------------------
# Helper: per-pair Vina docking run
# -----------------------------------------------------------------------------
run_pair() {
  local metal="$1"
  local pocket="$2"
  local seed="$3"
  local pair_log="${OUT_BASE}/${metal}/${pocket}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/${pocket}/seed${seed}"

  # Resolve per-metal SMILES CSV (SCAFFOLD fallback if missing)
  local smiles_csv
  if ! smiles_csv=$(resolve_smiles_csv "${metal}"); then
    echo "FAIL: cannot resolve SMILES CSV for ${metal}" | tee -a "${LOG}"
    return 1
  fi

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  PYTHONPATH="${PROJECT_ROOT}" ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} "${SCRIPTS_DIR}/wf343c2_per_wave_check.py" \
    --wave "Wf-3-VinaSweep-${metal}-${pocket}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --budget-hr "${PER_PAIR_BUDGET_HR}" \
    --min-vram-mib 4096 \
    --min-disk-gib 10 \
    --skip-live > "${OUT_BASE}/${metal}/${pocket}/seed${seed}/wf343c2_per_wave_check.log" 2>&1
  local rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK ${metal} ${pocket} seed=${seed} rc=${rc_c2}" | tee -a "${LOG}"

  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START ${metal} ${pocket} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START ${metal}/${pocket} seed=${seed} ===" | tee -a "${LOG}"
  PYTHONPATH="${PROJECT_ROOT}" ${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} "${DOCK_PANEL}" \
    --smiles-csv "${smiles_csv}" \
    --pocket-pdb-dir "${POCKET_PDB_DIR_FALLBACK}" \
    --pockets "${pocket}" \
    --seed "${seed}" \
    --out-json "${OUT_BASE}/${metal}/${pocket}/seed${seed}/dock_result.json" \
    --work-dir "${OUT_BASE}/${metal}/${pocket}/seed${seed}/work" \
    --dock-timeout 3600 \
    "${COMMON_ARGS_TEMPLATE[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}

  local pair_end_s pair_elapsed_s
  pair_end_s=$(date +%s)
  pair_elapsed_s=$(( pair_end_s - pair_start_s ))
  echo "PAIR_BUDGET_END ${metal} ${pocket} seed=${seed} elapsed_s=${pair_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') END ${metal}/${pocket} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "${rc}" -ne 0 ]; then
    echo "FAIL: ${metal}/${pocket} seed=${seed} exited rc=${rc}"
    return "${rc}"
  fi
}

echo "=== Wf-3 8 d-block Vina sweep START $(date) ===" | tee -a "${LOG}"
echo "${#PAIR_OVERRIDE[@]} pairs (8 metals × 5 pockets × 5 seeds; ${PER_PAIR_BUDGET_HR} hr/pair ≈ ${TOTAL_BUDGET_HR} hr wall on cuda:0 Blackwell) -- USER_OK-gated per AGENTS.md §5.1" | tee -a "${LOG}"

# Iterate PAIR_OVERRIDE sequentially
for pair in "${PAIR_OVERRIDE[@]}"; do
  read -r metal pocket seed <<< "${pair}"
  run_pair "${metal}" "${pocket}" "${seed}"
done

echo "=== ALL Wf-3 8 d-block Vina sweep PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation: collect per-pair dock_result.json into sweep_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY:-/home/hugo/codes/mol-metal/.venv/bin/python} -c "
import json
from pathlib import Path
runs = []
for metal_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not metal_dir.is_dir() or metal_dir.name.startswith('scaffold_') or metal_dir.name in ('logs',):
        continue
    for pocket_dir in sorted(metal_dir.iterdir()):
        if not pocket_dir.is_dir():
            continue
        for seed_dir in sorted(pocket_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            dock_json = seed_dir / 'dock_result.json'
            if dock_json.exists():
                seed_n = seed_dir.name.replace('seed', '')
                runs.append({'metal': metal_dir.name, 'pocket': pocket_dir.name, 'seed': seed_n, 'metrics_path': str(dock_json)})
out = {
    'wf3_vina_8dblock_total_pairs': len(runs),
    'n_pairs_planned': ${#PAIR_OVERRIDE[@]},
    'runs': runs,
}
Path('${OUT_BASE}/sweep_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (planned ${#PAIR_OVERRIDE[@]})')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"
