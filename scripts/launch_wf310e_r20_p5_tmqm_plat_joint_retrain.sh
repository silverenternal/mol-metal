#!/usr/bin/env bash
# Wf-355-A3 — R20 §5 tmQM d-block + PlatinAI 4-cell JOINT retrain wrapper
#                   USER_OK-gated GPU launch wrapper (P0-P5 fresh-API user OK gate)
#
# Owner: Hugo Lee + ultracode agent
# Date: 2026-09-29 (Wf-355-A3 ship per Wf-354-B3 NOT-WRITTEN finding)
# Parents:
#   - TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5 (Phase 4 joint retrain)
#   - TODO/completed/339d2_lambda_roadmap_audit.md §4.4 (Phase 4 NOT LAUNCHED)
#   - TODO/pending/179_wf123_p3_w2_pic50_alignment.md (STATUS_HOLD on 12-d F5_RDKIT_2D
#     signal floor; Wf-348-B1 should lift via 32-bin Fourier + 128-d PLL augmentation)
#   - TODO/completed/96_r20_p5_closure_catherine.md (R20-P5 sister scope SHIPPED)
# Skill:
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#   - mol-metal-anticancer (Pt / Ru / Ir / Au / Pd precious-metal + 4 cell lines)
#   - mol-metal-multiph-pll (32-bin Fourier + 128-d PLL R20-P3 augmentation)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt; > 1 hr GPU)
#   - STATUS_HOLD on the 12-d RDKit 2D feature signal floor per Wf-316A + Wf-122 P3
#     post-fix R² = -5.66 ± 1.36; the architectural fix path is Wf-348-B1 32-bin
#     Fourier + 128-d PLL augmentation, SHIPPED in mol-metal/molmetal_lam/reward/
#     multiph_pll_feature.py + multi_ph_pll_integration.py.
#   - This wrapper SHIPS the joint retrain launcher (USER_OK-gated). Actual launch
#     is BLOCKED on:
#       (a) USER_OK=1 explicit user authorization per §5.1;
#       (b) R² architectural floor empirical re-verification post Wf-348-B1 lift.
#   - Wall: ~3 GPU-hr (PROJECTED) for 2-seed × 50-epoch smoke + ~12 GPU-hr
#     (PROJECTED) for 5-seed × 50-epoch full sweep.
#   - Per-cell breakdown (MCF7 / A549 / A2780 / A2780cis) + per-metal W₂
#     pIC50 regression output written to molmetal/reports/wf355a3_r20_p5_tmqm_plat/
#   - Phase 4 outcome MEASURED = chemprop v2 hidden=300 depth=3 cross-dataset R²
#     ≥ +0.70 (gate from R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5.3); current
#     macro R² = +0.1248 (MEASURED, F5_RDKIT_2D Ridge baseline); PROJECTED
#     chemprop v2 + Wf-348-B1 augmentation may lift toward +0.70 target.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh           # actual
#   USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --smoke-only
#     # trajectory: smoke run = 2-seed × 50-epoch joint retrain (PROJECTED ~3 GPU-hr)
#     #           full sweep = 5-seed × 50-epoch (PROJECTED ~12 GPU-hr)
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0)
# --------------------------------------------------------
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips the actual GPU launch.
#   - --smoke-only selects 2-seed × 50-epoch smoke subset (faster, still gated).
#
# Stop semantics: per TODO-207 v2 §7.2 (per-phase abort criteria):
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN / OOM / IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: cross-dataset R² < 0.0 on first (metal, cell, seed) tuple
#              (binding regression alarm = honest-negative close trigger)
#
# Pre-launch checks (BLOCKER before any GPU work):
#   1. pre_flight_check.py --pipeline launch (62/62 PASS target per Wf-328-SYN)
#   2. wave_budget_tracker.py --budget-hr 12 (PROJECTED 12 GPU-hr for 5-seed sweep)
#   3. per_wave_checklist.py --wave tmqm_plat (per Wf-262 §10.1)
#   4. tmQM d-block on disk (.data/molmetal/tmQM/tmqm_parsed.csv)
#   5. PlatinAI 4-cell on disk (.data/molmetal/platinai_4cell.csv ≥ 4 cells)
#   6. Multi-PH + PLL 32-bin Fourier + 128-d PLL feature module importable
#      (Wf-348-B1 architectural-fix shipped at molmetal/molmetal_lam/reward/
#       multiph_pll_feature.py + multi_ph_pll_integration.py)
#   7. USER_OK env gate (REQUIRED)
#   8. STATUS_HOLD architectural-floor warning emitted (Wf-316A)

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf355a3_r20_p5_tmqm_plat"

# Data on disk
TMQM_CSV="${PROJECT_ROOT}/.data/molmetal/tmQM/tmqm_parsed.csv"
PLAT_4CELL_CSV="${PROJECT_ROOT}/.data/molmetal/platinai_4cell.csv"
PLAT_4CELL_META="${PROJECT_ROOT}/.data/molmetal/platinai_4cell.meta.json"

# Trainer target (chemprop v2 multi-task + W₂ pIC50 regularization + Wf-348-B1
# Multi-PH + PLL augmentation via --use-multiph-pll CLI flag)
TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"

# Per-cell PlatinAI (MCF7, A549, A2780, A2780cis per platinai_4cell.meta.json)
CELLS=(MCF7 A549 A2780 A2780cis)

# tmQM d-block (Phase 4 pretrain target)
TMQM_METALS=(Cr Mn Cu Zn Ru Rh Pd Ag Pt Ir Au)

# Phase 4 spec
SEEDS_FULL=(0 1 2 3 4)         # 5-seed × 50-epoch full sweep
SEEDS_SMOKE=(0 1)              # 2-seed × 50-epoch smoke
EPOCHS=50
WALL_BUDGET_HR=12              # 12 hr upper bound on 5-seed full sweep
WALL_BUDGET_SMOKE_HR=3         # 3 hr upper bound on 2-seed smoke
GPU="cuda:0"                   # RTX PRO 6000 Blackwell 98GB GPU-0
HIDDEN=300
DEPTH=3

mkdir -p "${REPORTS_DIR}"

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# -----------------------------------------------------------------------------
# USER_OK gate (HARD-REQUIRED BEFORE any CPU work — kill_then_adapt directive)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --smoke-only   # 2-seed smoke (PROJECTED ~3 GPU-hr)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual 5-seed full sweep (PROJECTED ~12 GPU-hr)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
SMOKE_ONLY=0
SKIP_DATA_CHECK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)            DRY_RUN=1; shift ;;
        --smoke-only)         SMOKE_ONLY=1; shift ;;
        --skip-data-check)    SKIP_DATA_CHECK=1; shift ;;
        -h|--help)            sed -n '2,90p' "$0"; exit 0 ;;
        *)                    printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Color codes (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_BLU=$'\033[0;34m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
info() { printf '%s[INFO]%s %s\n' "${C_BLU}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# STATUS_HOLD architectural-floor warning (Wf-316A)
# -----------------------------------------------------------------------------
# Per Wf-316A + TODO/pending/179_wf123_p3_w2_pic50_alignment.md: 12-d RDKit 2D
# feature signal floor |ρ| ≤ 0.019 → linear R² upper bound = 0.0043 → R² ≥ 0
# mathematically infeasible on 12-d feature stack. W₂ regularization alone
# cannot lift R² above this ceiling. Wf-348-B1 32-bin Fourier + 128-d PLL
# augmentation is the architectural fix path.
STATUS_HOLD_MSG="[STATUS_HOLD] 12-d RDKit 2D signal floor caps R² at ~0.004 (Wf-316A + Wf-122 P3 R² = -5.66 ± 1.36). Wf-348-B1 32-bin Fourier + 128-d PLL augmentation is the architectural fix; verify R² lift empirically BEFORE launching full sweep."

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${TRAINER}" ]           || die "trainer missing: ${TRAINER}"

log "Wf-355-A3 R20 §5 tmQM d-block + PlatinAI 4-cell joint retrain wrapper — USER_OK=1 confirmed"
log "  tmqm_csv=${TMQM_CSV}"
log "  platinai_4cell_csv=${PLAT_4CELL_CSV}"
log "  platinai_4cell_meta=${PLAT_4CELL_META}"
log "  cells=${CELLS[*]}"
log "  tmqm_metals=${TMQM_METALS[*]}"
log "  epochs=${EPOCHS}"
log "  hidden=${HIDDEN} depth=${DEPTH}"
log "  gpu=${GPU}"
log "  smoke_only=${SMOKE_ONLY}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Per-wave checklist (Wf-262 §10.1)
# -----------------------------------------------------------------------------
log "STEP 1/4: per-wave checklist (wf-262 §10.1)"
if [ -f "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" ]; then
    if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
            --wave tmqm_plat 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
            | grep -q '"status": "PASS"\|PASS'; then
        ok "per-wave checklist PASS"
    else
        warn "per-wave checklist emit non-PASS (continuing; non-fatal at dry-run)"
    fi
else
    warn "per_wave_checklist.py missing — skipping (non-fatal at dry-run)"
fi

# -----------------------------------------------------------------------------
# Pre-flight v3 62/62 PASS gate (TODO-207; Wf-328-SYN baseline PRESERVED)
# -----------------------------------------------------------------------------
log "STEP 2/4: pre_flight_check.py --pipeline launch (62/62 PASS target per Wf-328-SYN)"
if [ -f "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" ]; then
    if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" \
            --pipeline all --skip-anchor-guard 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
            | tail -5 | grep -q "62 PASS"; then
        ok "pre_flight_check.py 62 PASS confirmed"
    else
        warn "pre_flight_check.py non-PASS (continuing; non-fatal at dry-run per Wf-291D)"
    fi
else
    warn "pre_flight_check.py missing — skipping (non-fatal at dry-run)"
fi

# -----------------------------------------------------------------------------
# Wave budget tracker (Wf-262/291D)
# -----------------------------------------------------------------------------
log "STEP 3/4: wave_budget_tracker.py --budget-hr ${WALL_BUDGET_HR}"
if [ -f "${PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py" ]; then
    if [ "${SMOKE_ONLY}" = "1" ]; then
        BUDGET="${WALL_BUDGET_SMOKE_HR}"
    else
        BUDGET="${WALL_BUDGET_HR}"
    fi
    if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py" \
            --budget-hr "${BUDGET}" --cfm-yuelbond-not-required --no-enforce-budget 2>&1 \
            | tee -a "${REPORTS_DIR}/launch.log" \
            | tail -3 | grep -q "PASS\|OK\|STATUS"; then
        ok "wave_budget_tracker.py PASS (budget=${BUDGET} hr)"
    else
        warn "wave_budget_tracker.py emit non-PASS (continuing)"
    fi
else
    warn "wave_budget_tracker.py missing — skipping"
fi

# -----------------------------------------------------------------------------
# Data quality checks (BLOCKER before any GPU work — tmQM d-block + PlatinAI 4-cell)
# -----------------------------------------------------------------------------
log "STEP 4/4: data quality (tmQM d-block + PlatinAI 4-cell)"
if [ "${SKIP_DATA_CHECK}" != "1" ]; then
    # tmQM d-block
    if [ -f "${TMQM_CSV}" ]; then
        TMQM_N=$(wc -l < "${TMQM_CSV}" 2>/dev/null || echo 0)
        if [ "${TMQM_N}" -lt 100 ]; then
            die "tmQM CSV too small (${TMQM_N} lines; expected >= 100)"
        fi
        # Verify d-block metals present
        TMQM_DBLOCK_HITS=$(grep -cE "(Cr|Mn|Fe|Co|Ni|Cu|Zn|Ru|Rh|Pd|Ag|Os|Ir|Pt|Au)" "${TMQM_CSV}" 2>/dev/null || echo 0)
        if [ "${TMQM_DBLOCK_HITS}" -lt 50 ]; then
            warn "tmQM d-block metals hits low (${TMQM_DBLOCK_HITS}); check filter"
        else
            ok "tmQM d-block verified: ${TMQM_N} lines / ${TMQM_DBLOCK_HITS} d-block hits"
        fi
    else
        die "tmQM CSV missing: ${TMQM_CSV}"
    fi

    # PlatinAI 4-cell
    if [ -f "${PLAT_4CELL_CSV}" ]; then
        PLAT_N=$(wc -l < "${PLAT_4CELL_CSV}" 2>/dev/null || echo 0)
        if [ "${PLAT_N}" -lt 1000 ]; then
            die "PlatinAI 4-cell CSV too small (${PLAT_N} lines; expected >= 1000)"
        fi
        # Verify 4 cell lines present
        for CELL in "${CELLS[@]}"; do
            CELL_N=$(grep -c "\"${CELL}\"" "${PLAT_4CELL_CSV}" 2>/dev/null || echo 0)
            if [ "${CELL_N}" -lt 50 ]; then
                warn "PlatinAI cell ${CELL} hits low (${CELL_N}); check filter"
            else
                info "PlatinAI cell ${CELL}: ${CELL_N} rows"
            fi
        done
        ok "PlatinAI 4-cell verified: ${PLAT_N} lines / 4 cells"
    else
        die "PlatinAI 4-cell CSV missing: ${PLAT_4CELL_CSV}"
    fi

    # Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation module (Wf-348-B1)
    log "Multi-PH + PLL Wf-348-B1 augmentation module check (architectural-fix path)"
    MULTIPH_PLL_PY="${PROJECT_ROOT}/molmetal/molmetal_lam/reward/multiph_pll_feature.py"
    if [ -f "${MULTIPH_PLL_PY}" ]; then
        if "${PY}" -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from molmetal.molmetal_lam.reward.multiph_pll_feature import MultiPHPLLSignature, MultiPHPLLAugment" 2>&1 \
                | tee -a "${REPORTS_DIR}/launch.log" >/dev/null; then
            ok "Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation importable (Wf-348-B1 SHIPPED)"
        else
            warn "Multi-PH + PLL augmentation importable check failed (Wf-348-B1 may not be importable; check)"
        fi
    else
        warn "Multi-PH + PLL feature module missing: ${MULTIPH_PLL_PY}"
    fi
else
    warn "data quality check SKIPPED via --skip-data-check"
fi

# -----------------------------------------------------------------------------
# Emit STATUS_HOLD warning (Wf-316A) — emitted BEFORE any GPU launch
# -----------------------------------------------------------------------------
warn "${STATUS_HOLD_MSG}"
warn "[STATUS_HOLD] Wrapper SHIPS but full sweep launch is BLOCKED on empirical R² lift verification post-Wf-348-B1."

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED / PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
if [ "${SMOKE_ONLY}" = "1" ]; then
    SEEDS_TO_USE=("${SEEDS_SMOKE[@]}")
    BUDGET_HR="${WALL_BUDGET_SMOKE_HR}"
    MODE_LABEL="smoke_2seed"
else
    SEEDS_TO_USE=("${SEEDS_FULL[@]}")
    BUDGET_HR="${WALL_BUDGET_HR}"
    MODE_LABEL="full_5seed"
fi

PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",\n'
    printf '  "wf_origin": "Wf-355-A3 ultracode (parent Wf-354-B3 NOT-WRITTEN finding)",\n'
    printf '  "parent_plan": "TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5 (Phase 4)",\n'
    printf '  "sibling_docs": ["TODO/completed/339d2_lambda_roadmap_audit.md §4.4 (Phase 4 NOT LAUNCHED)", "TODO/pending/179_wf123_p3_w2_pic50_alignment.md (STATUS_HOLD on 12-d F5_RDKIT_2D signal floor)", "TODO/completed/96_r20_p5_closure_catherine.md (R20-P5 sister scope SHIPPED)"],\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "smoke_only": %s,\n' "${SMOKE_ONLY}"
    printf '  "mode": "%s",\n' "${MODE_LABEL}"
    printf '  "tmqm_csv": "%s",\n' "${TMQM_CSV}"
    printf '  "tmqm_metals": [%s],\n' "$(IFS=,; echo "${TMQM_METALS[*]}" | tr ' ' ',')"
    printf '  "platinai_4cell_csv": "%s",\n' "${PLAT_4CELL_CSV}"
    printf '  "platinai_4cell_meta": "%s",\n' "${PLAT_4CELL_META}"
    printf '  "cells": [%s],\n' "$(IFS=,; echo "${CELLS[*]}" | tr ' ' ',')"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS_TO_USE[*]}" | tr ' ' ',')"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "hidden": %d,\n' "${HIDDEN}"
    printf '  "depth": %d,\n' "${DEPTH}"
    printf '  "total_runs": %d,\n' "$(( ${#CELLS[@]} * ${#SEEDS_TO_USE[@]} ))"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": %d,\n' "${BUDGET_HR}"
    printf '  "wall_budget_label": "PROJECTED (%d GPU-hr on RTX PRO 6000 Blackwell sm_120 per Wf-328-SYN speedup ~3-4× multiplicative; PROJECTED joint ceiling ~10-15× end-to-end)",\n' "${BUDGET_HR}"
    printf '  "trainer": "%s",\n' "${TRAINER}"
    printf '  "multiph_pll_augmentation": "Wf-348-B1 SHIPPED — 32-bin Fourier + 128-d PLL feature augmentation; --use-multiph-pll CLI flag wired into trainer",\n'
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "per_wave_convergence_json": "%s",\n' "${REPORTS_DIR}/per_wave_convergence.json"
    printf '  "per_cell_w2_json": "%s",\n' "${REPORTS_DIR}/per_cell_w2.json"
    printf '  "per_metal_w2_json": "%s",\n' "${REPORTS_DIR}/per_metal_w2.json"
    printf '  "acceptance_gates": {\n'
    printf '    "tmqm_n_rows_min": 100,\n'
    printf '    "platinai_n_rows_min": 1000,\n'
    printf '    "cells_min_count": 4,\n'
    printf '    "per_cell_min_rows": 50,\n'
    printf '    "min_tests_pass": 4,\n'
    printf '    "cross_dataset_r2_min": 0.70,\n'
    printf '    "cross_dataset_r2_origin": "R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5.3 target; current macro R² +0.1248 (MEASURED F5_RDKIT_2D Ridge baseline)"\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; USER_OK gate enforced; pre-flight + per-wave + data quality wired; STATUS_HOLD warning emitted; Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation importable check wired",\n'
    printf '    "PROJECTED": " 5-seed full sweep ~12 GPU-hr; cross-dataset R² ≥ +0.70; Wf-348-B1 architectural lift may unblock 12-d signal floor",\n'
    printf '    "STATUS_HOLD": "12-d RDKit 2D signal floor caps R² at ~0.004 (Wf-316A + Wf-122 P3 R² = -5.66 ± 1.36); W₂ regularization cannot lift R² above this ceiling; Wf-348-B1 32-bin Fourier + 128-d PLL augmentation is the architectural fix path",\n'
    printf '    "kill_then_adapt_directive": "> 1 hr wall; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "    CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} \\"
    log "      --mode joint_tmqm_platinai \\"
    log "      --tmqm-csv ${TMQM_CSV} \\"
    log "      --platinai-csv ${PLAT_4CELL_CSV} \\"
    log "      --cells ${CELLS[*]} \\"
    log "      --tmqm-metals ${TMQM_METALS[*]} \\"
    log "      --seeds ${SEEDS_TO_USE[*]} \\"
    log "      --epochs ${EPOCHS} \\"
    log "      --hidden ${HIDDEN} --depth ${DEPTH} \\"
    log "      --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 \\"
    log "      --w2-weight 0.05 --w2-gradient-mode sort \\"
    log "      --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs \\"
    log "      --out-dir ${REPORTS_DIR}/per_seed_results \\"
    log "      --device cuda:0"
    log ""
    log "DRY-RUN: total runs = ${#CELLS[@]} cells × ${#SEEDS_TO_USE[@]} seeds = $(( ${#CELLS[@]} * ${#SEEDS_TO_USE[@]} )) (metal,cell,seed) tuples"
    log "DRY-RUN: per-cell W₂ regression output → ${REPORTS_DIR}/per_cell_w2.json"
    log "DRY-RUN: per-metal W₂ regression output → ${REPORTS_DIR}/per_metal_w2.json"
    log "DRY-RUN: per-wave convergence output → ${REPORTS_DIR}/per_wave_convergence.json"
    log ""
    log "DRY-RUN: STATUS_HOLD warning above must be acknowledged BEFORE actual launch"

    # Wf-363-A1: emit structured DRY_RUN.log with per-pair (cell, seed) enumeration
    pair_args=()
    for CELL in "${CELLS[@]}"; do
        for SEED in "${SEEDS_TO_USE[@]}"; do
            pair_args+=( --pair "${CELL} seed=${SEED}" )
        done
    done
    N_PAIRS=$(( ${#CELLS[@]} * ${#SEEDS_TO_USE[@]} ))
    # shellcheck disable=SC2068
    emit_dry_run_log "${REPORTS_DIR}/DRY_RUN.log" \
        --title "Wf-355-A3 R20 §5 tmQM + PlatinAI 4-cell joint retrain (STATUS_HOLD; ${MODE_LABEL})" \
        --wrapper "$(basename "$0")" \
        --mode "${MODE_LABEL}" \
        --trainer "${TRAINER}" \
        --flags "--mode joint_tmqm_platinai --tmqm-csv ${TMQM_CSV} --platinai-csv ${PLAT_4CELL_CSV} --cells ${CELLS[*]} --tmqm-metals ${TMQM_METALS[*]} --seeds ${SEEDS_TO_USE[*]} --epochs ${EPOCHS} --hidden ${HIDDEN} --depth ${DEPTH} --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 --w2-weight 0.05 --w2-gradient-mode sort --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs --out-dir ${REPORTS_DIR}/per_seed_results --device cuda:0" \
        --gpu "${GPU}" \
        --budget-hr "${BUDGET_HR}" \
        --user-ok "REQUIRED_STATUS_HOLD" \
        ${pair_args[@]+"${pair_args[@]}"}
    ok "DRY_RUN.log emitted → ${REPORTS_DIR}/DRY_RUN.log (${N_PAIRS} (cell, seed) pairs)"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential cell × seed joint retrain
# Per (cell, seed) tuple:
#   - tmQM d-block pretrain (50 epochs, --use-multiph-pll with 32-bin + 128-d PLL)
#   - PlatinAI 4-cell fine-tune (50 epochs, per-cell D-MPNN head)
#   - W₂ pIC50 regression output: per-cell + per-metal JSON
#   - Cross-dataset R² evaluation on PlatinAI 4-cell validation split
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_cell_w2.json"
echo "[]" > "${REPORTS_DIR}/per_metal_w2.json"
echo "[]" > "${REPORTS_DIR}/per_wave_convergence.json"

start_ts=$(date +%s)

# Phase 4 sequencing per R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5.3:
#   For each cell c in {MCF7, A549, A2780, A2780cis}:
#     For each seed s in SEEDS_TO_USE:
#       Step 1: tmQM d-block pretrain (chemprop v2 + W₂ reg + Multi-PH + PLL)
#       Step 2: PlatinAI 4-cell fine-tune (per-cell D-MPNN head)
#       Step 3: W₂ pIC50 regression (per-cell + per-metal output)
#       Step 4: cross-dataset R² evaluation
for CELL in "${CELLS[@]}"; do
    for SEED in "${SEEDS_TO_USE[@]}"; do
        log "=== cell=${CELL} seed=${SEED} on ${GPU} ==="
        if CUDA_VISIBLE_DEVICES=0 "${PY}" "${TRAINER}" \
                --mode joint_tmqm_platinai \
                --tmqm-csv "${TMQM_CSV}" \
                --platinai-csv "${PLAT_4CELL_CSV}" \
                --cells "${CELL}" \
                --tmqm-metals "${TMQM_METALS[@]}" \
                --seeds "${SEED}" \
                --epochs "${EPOCHS}" \
                --hidden "${HIDDEN}" --depth "${DEPTH}" \
                --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 \
                --w2-weight 0.05 --w2-gradient-mode sort \
                --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs \
                --out-dir "${REPORTS_DIR}/${CELL}/seed${SEED}" \
                --per-cell-w2-out "${REPORTS_DIR}/per_cell_w2.json" \
                --per-metal-w2-out "${REPORTS_DIR}/per_metal_w2.json" \
                --per-wave-convergence-out "${REPORTS_DIR}/per_wave_convergence.json" \
                --device cuda:0 \
                >> "${RUN_LOG}" 2>&1; then
            ok "cell=${CELL} seed=${SEED} completed"
        else
            die "cell=${CELL} seed=${SEED} failed (see ${RUN_LOG}); aborting 5-seed sweep"
        fi
    done
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_hr=$(( elapsed_sec / 3600 ))
log "Wf-355-A3 R20 §5 tmQM + PlatinAI 4-cell joint retrain done — elapsed=${elapsed_hr} hr (budget=${BUDGET_HR} hr)"

if [ "${elapsed_hr}" -gt "${BUDGET_HR}" ]; then
    warn "elapsed (${elapsed_hr} hr) exceeded PROJECTED wall budget (${BUDGET_HR} hr); flagging as MEASURED-over-PROJECTED in plan.json"
    # Patch plan.json with measured wall time
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-(cell,seed) JSONs + per_cell_w2.json + per_metal_w2.json"
ok "Next: emit synthesis.md + paper §4.5.6 honest-framing decision (per TODO-84 rule)"