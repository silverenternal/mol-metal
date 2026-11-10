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
#       (b) R² architectural floor empirical re-verification post Wf-348-B1 lift;
#       (c) THE TRAINER INTERFACE DOES NOT EXIST — the `joint_tmqm_platinai`
#           mode and the --use-multiph-pll / --per-cell-w2-out /
#           --per-metal-w2-out / --per-wave-convergence-out flags are NOT
#           declared by train_dmpnn_multitask_wasserstein.py.  MEASURED
#           2026-11-02 by molmetal/scripts/wf_argv_capability_check.py: 12 of
#           the 20 argv flags this launcher passes are undeclared (see
#           molmetal/reports/wf355a3_r20_p5_tmqm_plat/argv_capability.txt).
#           That count is NOT a literal in the plan record: the plan.json this
#           wrapper emits PARSES the same report at run time, so the number it
#           publishes is derived from the checker, not asserted here.  The
#           figure quoted in this header is a MEASURED snapshot of that report
#           on the date above, cited for the reader.
#   - Wall: ~3 GPU-hr (PROJECTED) for 2-seed × 50-epoch smoke + ~12 GPU-hr
#     (PROJECTED) for 5-seed × 50-epoch full sweep.
#   - Per-cell breakdown (MCF7 / A549 / A2780 / A2780cis) + per-metal W₂
#     pIC50 regression output written to molmetal/reports/wf355a3_r20_p5_tmqm_plat/
#   - Phase 4 outcome: NOT RUN — there is NO MEASURED Phase 4 result for this
#     wrapper.  MEASURED 2026-11-02: TODO/completed/339d2_lambda_roadmap_audit.md:166
#     records "§5 Phase 4 … ⏸️ NOT LAUNCHED (USER_OK-gated)".  The acceptance
#     GATE is chemprop v2 hidden=300 depth=3 cross-dataset R² ≥ +0.70
#     (R20_BIG_DATASET_FULL_RETRAIN_PLAN.md:192) — that gate has never run, so
#     the R² ≥ +0.70 figure is a PROJECTED TARGET, not a measurement.
#   - The R² reading that DOES exist is a CROSS-DATASET macro figure and it is
#     NOT this launcher's cohort.  MEASURED 2026-11-02 by reading
#     metrics/by_metric/tmqm_combined_r2.json:
#       * cross_dataset_macro_r2_mean = 0.12475146687654443 — a macro over 38
#         cohorts in 3 datasets (platinai_4cell 4 + metalcytotox_5metal 5 +
#         tmqm_dblock 29; source artifact
#         molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json);
#       * the PlatinAI-4-cell cohort THIS launcher trains on has its own
#         per-dataset macro R² = 0.0034405282483466974 (same JSON,
#         cross_dataset.per_dataset_macro_r2_mean.platinai_4cell).
#     It is also NOT an "F5_RDKIT_2D Ridge baseline" reading.  PROJECTED:
#     chemprop v2 + Wf-348-B1 augmentation may lift toward the +0.70 target.
#
# Wf-A269 wave-11 (2026-11-02) - REPAIR (DEFECT C): the header used to assert
# "Phase 4 outcome MEASURED = … current macro R² = +0.1248 (MEASURED,
# F5_RDKIT_2D Ridge baseline)" with no cohort label, and described a gate that
# NEVER RAN as a MEASURED outcome — while the plan.json in the SAME file
# (acceptance_gates.cross_dataset_r2_origin) already carried the correct
# disambiguation.  The two renderings of the same file contradicted each
# other.  The header now says what the plan.json says, and the never-run
# Phase 4 outcome is labelled NOT RUN / PROJECTED.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
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


# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

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
# 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
# The trainer already accepts --config and ends parse_args() in
# apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
# yaml is the base and the per-call flags below override it.
# Precedence: explicit CLI flag > YAML > argparse default.
# Absolute: this wrapper has no top-level `cd`, so a relative path would
# resolve against the caller's cwd.
DMPNN_CONFIG="${DMPNN_CONFIG:-${PROJECT_ROOT}/configs/dmpnn.yaml}"

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
# Wf-A269 wave-10 (2026-11-02): the SESSION cap (AGENTS.md §5.1 + Wf-294B
# kill_then_adapt).  This is the number the wave's projected wall is measured
# AGAINST — the wave budget above is what the wave is EXPECTED to cost, and
# the two are not the same quantity.
SESSION_CAP_HR=10
GPU="cuda:0"                   # RTX PRO 6000 Blackwell 98GB GPU-0
HIDDEN=300
DEPTH=3

mkdir -p "${REPORTS_DIR}"

# Die helper — MOVED UP to before its first call (Wf-A269 repair 2, 2026-11-02).
# Bash executes a script line-by-line, so a function definition that appears
# LATER in the file is not yet in scope at an earlier call site — the call then
# dies with "die: command not found" (rc=127) and the intended FATAL message
# + rc=1 are both lost. ${C_RED:-}/${C_RST:-} keep the early call site safe
# (C_RED is initialised further down); `mkdir -p` keeps the FATAL message
# deliverable when the reports dir does not exist yet.
die()  { mkdir -p "${REPORTS_DIR}" 2>/dev/null || true; printf '%s[FATAL]%s %s\n' "${C_RED:-}" "${C_RST:-}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

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
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
info() { printf '%s[INFO]%s %s\n' "${C_BLU}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# Wf-A269 wave-9 (2026-11-02): JSON array emitters.  The plan record used to
# build its arrays with `printf '[%s]' "$(IFS=,; echo "${A[*]}" | tr ' ' ',')`,
# which produced UNQUOTED tokens - `"cells": [MCF7,A549,A2780,A2780cis]` - so
# plan.json was not parseable by ANY JSON reader (MEASURED: json.load raised
# "Expecting value: line 11 column 19").  Nothing caught it because the only
# guard on this file was a substring match on the raw text, and no code
# consumer parses it (only TODO docs reference it).
json_array() {
    local _out="" _e
    for _e in "$@"; do
        _out="${_out}\"${_e}\","
    done
    printf '%s' "${_out%,}"
}
json_num_array() {
    local _out="" _e
    for _e in "$@"; do
        _out="${_out}${_e},"
    done
    printf '%s' "${_out%,}"
}
# Wf-A269 wave-10 (2026-11-02): JSON string escaper for a FREE-FORM detail
# string (the gate verdicts quote a reason that comes from a report file, so
# it can contain quotes, backslashes or newlines).  Without this, one quote in
# a reason would make plan.json unparseable — the same defect the array
# emitters above had.
json_escape() {
    local s="${1-}"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/ }"
    s="${s//$'\r'/ }"
    s="${s//$'\t'/ }"
    printf '%s' "${s}"
}

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

# -----------------------------------------------------------------------------
# Trainer capability gate (2026-10-27)
# -----------------------------------------------------------------------------
# This launcher builds an argv the trainer does not parse.  MEASURED against
# train_dmpnn_multitask_wasserstein.parse_args, the following are NOT declared:
#
#   --mode --tmqm-csv --platinai-csv --cells --tmqm-metals --use-multiph-pll
#   --multiph-fourier-bins --pll-signature-dim --w2-weight --per-cell-w2-out
#   --per-metal-w2-out --per-wave-convergence-out
#
# i.e. the whole `joint_tmqm_platinai` interface this script drives has never
# existed in the trainer: grep for `joint_tmqm_platinai` or `per_cell_w2_out`
# across molmetal/ returns nothing.  Before this gate the sweep died per tuple
# with argparse rc=2, so a 4-cell x 5-seed run reported five failures and no
# diagnosis.  Now it refuses once, with the list.
#
# Removing the gate is not the fix; implementing the interface is (that is a
# feature, not a launcher repair, and it is not part of the 2026-10-27 pass).
#
# Placed after the helper definitions: the first version of this gate sat at
# the top of the file and called `die`, which is defined further down, so the
# failure path itself failed with "die: command not found" and the launcher
# carried on.
CAPABILITY_CHECK="${SCRIPTS_DIR}/wf_argv_capability_check.py"
LAUNCH_ARGV_FLAGS="--mode --tmqm-csv --platinai-csv --cells --tmqm-metals \
--seeds --epochs --hidden --depth --use-multiph-pll --multiph-fourier-bins \
--pll-signature-dim --w2-weight --w2-gradient-mode --cpu-num-workers \
--out-dir --per-cell-w2-out --per-metal-w2-out --per-wave-convergence-out \
--device"
# Wf-A269 wave-9 (2026-11-02) - REPAIR: the gate is scoped to the LAUNCH path.
# It used to run BEFORE the DRY_RUN block and called die(), so a --dry-run could
# never reach its own DRY_RUN.log emission: the wrapper exited 1 and the ONLY
# DRY_RUN.log left on disk stayed the stale 2026-10-27 full-sweep artefact
# (budget 12 hr, mode full_5seed, 20 pairs).  That stale file is what
# molmetal/tests/test_wf363a1_dry_run_log.py::test_budget_within_session_cap
# read, so the cap assertion failed on a budget the test never asked the
# wrapper to produce.  A dry run launches NOTHING, and the gate's own stated
# purpose ("refusing to launch tuples that can only exit rc=2") does not apply
# to it - so the gate now RECORDS the verdict and the dry run proceeds to emit
# a plan that states it.  On the real launch path the gate is UNCHANGED: it
# still die()s before a single trainer process starts.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     if ! "${PY}" "${CAPABILITY_CHECK}" --trainer "${TRAINER}" \
#             --flags "${LAUNCH_ARGV_FLAGS}" > "${REPORTS_DIR}/argv_capability.txt" 2>&1; then
#         cat "${REPORTS_DIR}/argv_capability.txt" >&2
#         die "trainer does not declare this launcher's argv (see ${REPORTS_DIR}/argv_capability.txt). The joint_tmqm_platinai interface is not implemented in ${TRAINER}; refusing to launch tuples that can only exit rc=2."
#     fi
CAPABILITY_FILE="${REPORTS_DIR}/argv_capability.txt"
#
# Wf-A269 wave-11 (2026-11-02) - REPAIR (DEFECT A): the gate mapped EVERY
# non-zero exit of the checker to the single status BLOCKED, and the plan
# record then asserted a HARDCODED flag count ("12").  MEASURED: the checker
# has four exits — 0 = PASS, 1 = trainer-not-found OR N-flags-missing,
# 2 = the trainer has no importable parse_args.  Exit 2 means UNVERIFIED
# ("we could not measure"), which is NOT the same as "we measured a defect",
# so it now gets its own status.  Every count below is PARSED out of the
# report the checker wrote (${CAPABILITY_FILE}); no number here is a literal.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     CAPABILITY_STATUS="UNVERIFIED"
#     if [ -f "${CAPABILITY_CHECK}" ]; then
#         if "${PY}" "${CAPABILITY_CHECK}" --trainer "${TRAINER}" \
#                 --flags "${LAUNCH_ARGV_FLAGS}" > "${REPORTS_DIR}/argv_capability.txt" 2>&1; then
#             CAPABILITY_STATUS="OK"
#             ok "trainer declares every flag this launcher passes"
#         else
#             cat "${REPORTS_DIR}/argv_capability.txt" >&2
#             CAPABILITY_STATUS="BLOCKED"
#             if [ "${DRY_RUN}" = "1" ]; then
#                 warn "trainer does NOT declare this launcher's argv (see ${REPORTS_DIR}/argv_capability.txt)."
#                 warn "DRY-RUN only: the joint_tmqm_platinai interface is not implemented in ${TRAINER}, so the plan below CANNOT be launched as written."
#                 warn "The real launch path is unchanged and still fails closed before any trainer process starts."
#             else
#                 die "trainer does not declare this launcher's argv (see ${REPORTS_DIR}/argv_capability.txt). The joint_tmqm_platinai interface is not implemented in ${TRAINER}; refusing to launch tuples that can only exit rc=2."
#             fi
#         fi
#     else
#         warn "capability checker missing: ${CAPABILITY_CHECK}; argv not verified"
#     fi
CAPABILITY_STATUS="UNVERIFIED"
CAPABILITY_RC=""
CAPABILITY_MISSING_COUNT=""
CAPABILITY_TOTAL_COUNT=""
CAPABILITY_MISSING_LINES=""
CAPABILITY_COUNT_TEXT=""
CAPABILITY_COUNT_PHRASE=""
CAPABILITY_COUNT_CLAUSE="no flag count was produced — the capability was NOT measured"

# Parse the counts out of the checker's OWN report.  Its last summary line is
# either "PASS — all N flag(s) are declared ..." or
# "FAIL — N of M flag(s) are not declared ...", and every undeclared flag is
# also listed on its own "MISSING <flag>" line.
capability_derive_counts() {
    local summary="" pair=""
    CAPABILITY_MISSING_COUNT=""
    CAPABILITY_TOTAL_COUNT=""
    CAPABILITY_COUNT_TEXT=""
    [ -f "${CAPABILITY_FILE}" ] || return 0
    summary="$(grep -E 'flag\(s\)' "${CAPABILITY_FILE}" 2>/dev/null | tail -n 1 || true)"
    [ -n "${summary}" ] || return 0
    pair="$(printf '%s\n' "${summary}" | grep -oE '[0-9]+ of [0-9]+ flag' | head -n 1 || true)"
    if [ -n "${pair}" ]; then
        CAPABILITY_MISSING_COUNT="${pair%% of *}"
        CAPABILITY_TOTAL_COUNT="${pair##* of }"
        CAPABILITY_TOTAL_COUNT="${CAPABILITY_TOTAL_COUNT%% flag*}"
    else
        pair="$(printf '%s\n' "${summary}" | grep -oE 'all [0-9]+ flag' | head -n 1 || true)"
        if [ -n "${pair}" ]; then
            CAPABILITY_MISSING_COUNT="0"
            CAPABILITY_TOTAL_COUNT="${pair#all }"
            CAPABILITY_TOTAL_COUNT="${CAPABILITY_TOTAL_COUNT%% flag*}"
        fi
    fi
    if [ -n "${CAPABILITY_MISSING_COUNT}" ] && [ -n "${CAPABILITY_TOTAL_COUNT}" ]; then
        CAPABILITY_COUNT_TEXT="${CAPABILITY_MISSING_COUNT} of ${CAPABILITY_TOTAL_COUNT}"
    fi
}

if [ -f "${CAPABILITY_CHECK}" ]; then
    CAPABILITY_RC=0
    "${PY}" "${CAPABILITY_CHECK}" --trainer "${TRAINER}" \
        --flags "${LAUNCH_ARGV_FLAGS}" > "${CAPABILITY_FILE}" 2>&1 || CAPABILITY_RC=$?
    case "${CAPABILITY_RC}" in
        0) CAPABILITY_STATUS="OK" ;;
        1) CAPABILITY_STATUS="BLOCKED" ;;
        # 2 = the trainer has no importable parse_args => UNVERIFIED, not a
        # measured defect.  Any other exit code is likewise "not measured".
        *) CAPABILITY_STATUS="UNVERIFIED" ;;
    esac
    capability_derive_counts
    # Cross-check the parsed count against the checker's own MISSING lines: if
    # the summary line and the flag list disagree, the report is not trusted
    # and the capability degrades to UNVERIFIED rather than asserting a number.
    if [ -n "${CAPABILITY_MISSING_COUNT}" ]; then
        CAPABILITY_MISSING_LINES="$(grep -c '^MISSING ' "${CAPABILITY_FILE}" 2>/dev/null || true)"
        if [ "${CAPABILITY_MISSING_LINES}" != "${CAPABILITY_MISSING_COUNT}" ]; then
            warn "argv-capability cross-check FAILED: the summary says ${CAPABILITY_MISSING_COUNT} undeclared flag(s) but the report lists ${CAPABILITY_MISSING_LINES} MISSING line(s); treating the capability as UNVERIFIED."
            CAPABILITY_STATUS="UNVERIFIED"
            CAPABILITY_MISSING_COUNT=""
            CAPABILITY_TOTAL_COUNT=""
            CAPABILITY_COUNT_TEXT=""
        fi
    fi
    # A PASS whose report carries no parsable count cannot be corroborated:
    # do not assert "0 of N undeclared" from a report we could not read.
    if [ "${CAPABILITY_STATUS}" = "OK" ] && [ -z "${CAPABILITY_TOTAL_COUNT}" ]; then
        warn "argv-capability report carries no parsable flag count although the checker exited 0; treating the capability as UNVERIFIED rather than asserting a count."
        CAPABILITY_STATUS="UNVERIFIED"
    fi
    case "${CAPABILITY_STATUS}" in
        OK)
            ok "trainer declares every flag this launcher passes (MEASURED: 0 of ${CAPABILITY_TOTAL_COUNT} undeclared; see ${CAPABILITY_FILE})"
            CAPABILITY_COUNT_CLAUSE="0 of ${CAPABILITY_TOTAL_COUNT} of the launcher's argv flags are undeclared (MEASURED)"
            CAPABILITY_COUNT_PHRASE="declares every flag this launcher passes"
            ;;
        BLOCKED)
            cat "${CAPABILITY_FILE}" >&2
            if [ -n "${CAPABILITY_COUNT_TEXT}" ]; then
                CAPABILITY_COUNT_CLAUSE="${CAPABILITY_COUNT_TEXT} of the launcher's argv flags are NOT declared by the trainer (MEASURED)"
                CAPABILITY_COUNT_PHRASE="does not declare ${CAPABILITY_COUNT_TEXT} of this launcher's argv flags"
            else
                CAPABILITY_COUNT_CLAUSE="the checker's report carries no parsable flag count — the trainer does NOT accept this launcher's argv (MEASURED from the checker's report, but no per-flag count could be derived)"
                CAPABILITY_COUNT_PHRASE="does not accept this launcher's argv (the checker's report carries no parsable flag count)"
            fi
            if [ "${DRY_RUN}" = "1" ]; then
                warn "trainer ${CAPABILITY_COUNT_PHRASE} (see ${CAPABILITY_FILE})."
                warn "DRY-RUN only: the joint_tmqm_platinai interface is not implemented in ${TRAINER}, so the plan below CANNOT be launched as written."
                warn "The real launch path is unchanged and still fails closed before any trainer process starts."
            else
                die "trainer ${CAPABILITY_COUNT_PHRASE} (see ${CAPABILITY_FILE}). The joint_tmqm_platinai interface is not implemented in ${TRAINER}; refusing to launch tuples that can only exit rc=2."
            fi
            ;;
        *)
            cat "${CAPABILITY_FILE}" >&2
            CAPABILITY_COUNT_CLAUSE="no flag count was produced — the capability was NOT measured (the checker exited ${CAPABILITY_RC})"
            CAPABILITY_COUNT_PHRASE="could not be verified (the checker exited ${CAPABILITY_RC})"
            if [ "${DRY_RUN}" = "1" ]; then
                warn "argv capability UNVERIFIED (the checker exited ${CAPABILITY_RC}; see ${CAPABILITY_FILE}): the trainer's parse_args could not be exercised, so this is NOT a measured defect."
                warn "DRY-RUN only: the plan below records the capability as UNVERIFIED, and the real launch path still refuses before any trainer process starts."
            else
                die "argv capability UNVERIFIED (the checker exited ${CAPABILITY_RC}; see ${CAPABILITY_FILE}): the trainer's parse_args could not be exercised, so this launcher's argv could not be verified. Refusing to launch."
            fi
            ;;
    esac
else
    warn "capability checker missing: ${CAPABILITY_CHECK}; argv not verified"
    CAPABILITY_STATUS="UNVERIFIED"
    CAPABILITY_COUNT_CLAUSE="no flag count was produced — the capability was NOT measured (the checker itself is missing)"
    CAPABILITY_COUNT_PHRASE="could not be verified (the checker itself is missing)"
fi

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
# Pre-launch gates — Wf-A269 wave-10 REPAIR (2026-11-02)
# -----------------------------------------------------------------------------
# Wf-A269 wave-10 (2026-11-02) — REPAIR: all three pre-launch gates below were
# DEAD GATES — checks that could not fail:
#
#   * the pre-flight was called with `--skip-anchor-guard`, a flag the tool
#     does NOT define -> argparse exit 2, no gate ever ran;
#   * its success branch grepped the LAST 5 stdout lines for the literal
#     "62 PASS" — a Wf-328-SYN baseline that no longer matches the gate count
#     — while the report's `summary` block sits at the TOP of the JSON, so
#     `tail -5` could never see it even if the tool had run;
#   * the per-wave checklist was grepped for '"status": "PASS"\|PASS' on
#     stdout, but the tool prints NOTHING on stdout: its verdict is in the
#     JSON it writes via --out;
#   * the wave budget tracker was called with `--cfm-yuelbond-not-required`,
#     a flag the tool does NOT define (only `--cfm-yuelbond-required`
#     exists) -> argparse exit 1, and the tool prints NOTHING on stdout
#     either: its verdict is in the JSON it writes via --report-json.
#
# Because every broken flag made its tool exit INSTANTLY, the dry-run looked
# clean and the wrapper looked ready to launch although it had never run a
# pre-flight at all.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
#             --wave tmqm_plat 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
#             | grep -q '"status": "PASS"\|PASS'; then
#     if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" \
#             --pipeline all --skip-anchor-guard 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
#             | tail -5 | grep -q "62 PASS"; then
#     if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py" \
#             --budget-hr "${BUDGET}" --cfm-yuelbond-not-required --no-enforce-budget 2>&1 \
#             | tee -a "${REPORTS_DIR}/launch.log" \
#             | tail -3 | grep -q "PASS\|OK\|STATUS"; then
#
# The repair: every gate calls its tool with flags the tool DEFINES, reads the
# verdict out of that tool's own JSON report through the shared checker
# (molmetal/scripts/wf_launch_gate_verdict.py), and records the verdict in the
# plan record + the DRY_RUN.log.  A dry-run never prints a verdict it did not
# measure: the live gates are SKIPPED there and their verdict stays
# UNVERIFIED.  The sibling launcher
# (scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh) skips the live
# gates in dry-run for the same reason — the real pre-flight takes ~400 s on
# this repo (MEASURED), which is longer than the dry-run test budgets allow.

GATE_VERDICT_CHECKER="${PROJECT_ROOT}/molmetal/scripts/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

PRE_FLIGHT_JSON="${REPORTS_DIR}/pre_flight.json"
CHECKLIST_JSON="${REPORTS_DIR}/per_wave_checklist.json"
BUDGET_JSON="${REPORTS_DIR}/wave_budget.json"

# UNVERIFIED until a gate has actually measured something (AGENTS.md §5.2:
# a number that was not measured must not be presented as one).
PRE_FLIGHT_VERDICT="UNVERIFIED"
CHECKLIST_VERDICT="UNVERIFIED"
BUDGET_VERDICT="UNVERIFIED"
PRE_FLIGHT_DETAIL="not run"
CHECKLIST_DETAIL="not run"
BUDGET_DETAIL="not run"

# The wave's grid + projected wall, needed by the checklist and the budget
# tracker.  (SEEDS_TO_USE is derived again further down for the plan record.)
if [ "${SMOKE_ONLY}" = "1" ]; then
    GATE_SEEDS=("${SEEDS_SMOKE[@]}")
    BUDGET="${WALL_BUDGET_SMOKE_HR}"
else
    GATE_SEEDS=("${SEEDS_FULL[@]}")
    BUDGET="${WALL_BUDGET_HR}"
fi

# Run the shared checker and echo back its verdict (PASS / FAIL / SKIP).
# The reason is written next to the gate's report so the plan can quote it.
run_gate_check() {
    local kind="$1" report="$2" wave="${3:-}" rc=0 out=""
    if [ -n "${wave}" ]; then
        out="$("${PY}" "${GATE_VERDICT_CHECKER}" "${kind}" --report "${report}" \
            --wave "${wave}" 2>&1)" || rc=$?
    else
        out="$("${PY}" "${GATE_VERDICT_CHECKER}" "${kind}" --report "${report}" 2>&1)" || rc=$?
    fi
    printf '%s\n' "${out}" > "${REPORTS_DIR}/gate_${kind}.verdict.txt"
    case "${rc}" in
        0) printf 'PASS' ;;
        1) printf 'FAIL' ;;
        *) printf 'SKIP' ;;
    esac
}

gate_detail() {
    head -1 "${REPORTS_DIR}/gate_$1.verdict.txt" 2>/dev/null || printf ''
}

# -----------------------------------------------------------------------------
# STEP 1/4 — pre-flight v3 gate (TODO-207; AGENTS.md §1.4 + §4)
# -----------------------------------------------------------------------------
log "STEP 1/4: pre_flight_check.py --pipeline all (TODO-207 v3 gate)"
if [ "${DRY_RUN}" = "1" ]; then
    warn "DRY-RUN: skipping the live pre-flight (plan only — verdict stays UNVERIFIED)"
    PRE_FLIGHT_DETAIL="not run: dry-run skips the live pre-flight (the real run takes ~400 s, MEASURED)"
elif [ -f "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" ]; then
    # The report is written BEFORE the tool returns its exit code, so the
    # verdict can be read from it even when the tool exits 1 (FAIL gates).
    # Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A crashed
    # or killed pre-flight must not leave the last run's PASS report on disk
    # for the consumer to read — "stale is treated as missing".
    rm -f "${PRE_FLIGHT_JSON}"
    "${PY}" "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" \
        --pipeline all --json "${PRE_FLIGHT_JSON}" \
        >> "${REPORTS_DIR}/launch.log" 2>&1 || true
    PRE_FLIGHT_VERDICT="$(run_gate_check preflight "${PRE_FLIGHT_JSON}")"
    PRE_FLIGHT_DETAIL="$(gate_detail preflight)"
    if [ "${PRE_FLIGHT_VERDICT}" = "PASS" ]; then
        ok "pre_flight_check.py PASS (${PRE_FLIGHT_DETAIL})"
    else
        warn "pre_flight_check.py ${PRE_FLIGHT_VERDICT} (${PRE_FLIGHT_DETAIL})"
        # AGENTS.md §4: the pre-flight PASS gate blocks a retrain.  A gate that
        # cannot fail is the defect this wave is repairing, so a FAIL here
        # refuses the launch instead of warning past it.
        die "pre_flight_check ${PRE_FLIGHT_VERDICT} — refusing to launch (report: ${PRE_FLIGHT_JSON})"
    fi
else
    warn "pre_flight_check.py missing — skipping (non-fatal at dry-run)"
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: this branch used the verdict
    # token "SKIPPED", so the plan record carried a FOURTH vocabulary value
    # (PASS / FAIL / SKIPPED / UNVERIFIED) that no consumer and no gate
    # checker knows.  AGENTS.md §5.2: a gate that was not verified says
    # UNVERIFIED, never PASS.  The dry-run branch above already uses
    # UNVERIFIED for the same condition, so the two branches disagreed about
    # what "not measured" looks like.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     PRE_FLIGHT_VERDICT="SKIPPED"
    PRE_FLIGHT_VERDICT="UNVERIFIED"
    PRE_FLIGHT_DETAIL="not run: pre_flight_check.py is missing from the tree"
fi

# -----------------------------------------------------------------------------
# STEP 2/4 — per-wave checklist (Wf-262 §10.1)
# -----------------------------------------------------------------------------
log "STEP 2/4: per-wave checklist (wf-262 §10.1)"
if [ "${DRY_RUN}" = "1" ]; then
    warn "DRY-RUN: skipping the per-wave checklist (its input is the skipped pre-flight — verdict stays UNVERIFIED)"
    CHECKLIST_DETAIL="not run: dry-run skips the checklist (it reads the pre-flight report)"
elif [ -f "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" ]; then
    # The tool prints NOTHING on stdout: its verdict is the JSON it writes
    # via --out.  --preflight points it at THIS run's pre-flight report.
    "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
        --wave tmqm_plat \
        --preflight "${PRE_FLIGHT_JSON}" \
        --out "${CHECKLIST_JSON}" \
        --out-md "${REPORTS_DIR}/per_wave_checklist.md" \
        --pairs-metals "$(IFS=,; echo "${CELLS[*]}")" \
        --pairs-seeds "$(IFS=,; echo "${GATE_SEEDS[*]}")" \
        >> "${REPORTS_DIR}/launch.log" 2>&1 || true
    CHECKLIST_VERDICT="$(run_gate_check checklist "${CHECKLIST_JSON}")"
    CHECKLIST_DETAIL="$(gate_detail checklist)"
    if [ "${CHECKLIST_VERDICT}" = "PASS" ]; then
        ok "per-wave checklist PASS (${CHECKLIST_DETAIL})"
    else
        warn "per-wave checklist ${CHECKLIST_VERDICT} (${CHECKLIST_DETAIL})"
    fi
else
    warn "per_wave_checklist.py missing — skipping (non-fatal at dry-run)"
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: same vocabulary defect as the
    # pre-flight branch above — "SKIPPED" was a fourth verdict token.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     CHECKLIST_VERDICT="SKIPPED"
    CHECKLIST_VERDICT="UNVERIFIED"
    CHECKLIST_DETAIL="not run: per_wave_checklist.py is missing from the tree"
fi

# -----------------------------------------------------------------------------
# STEP 3/4 — wave budget tracker (Wf-262/291D)
# -----------------------------------------------------------------------------
log "STEP 3/4: wave_budget_tracker.py --budget-hr ${SESSION_CAP_HR} (session cap; wave projects ${BUDGET} hr)"
if [ -f "${PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py" ]; then
    # The tool prints NOTHING on stdout and its exit code is ALWAYS 0 on a
    # successful run: the verdict is in the JSON it writes via --report-json.
    # --waves names THIS wave with its projected wall, so the tracker measures
    # this launch instead of auto-discovering eight unrelated waves.
    "${PY}" "${PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py" \
        --budget-hr "${SESSION_CAP_HR}" \
        --waves "tmqm_plat:${BUDGET}" \
        --report-json "${BUDGET_JSON}" \
        >> "${REPORTS_DIR}/launch.log" 2>&1 || true
    BUDGET_VERDICT="$(run_gate_check budget "${BUDGET_JSON}" tmqm_plat)"
    BUDGET_DETAIL="$(gate_detail budget)"
    if [ "${BUDGET_VERDICT}" = "PASS" ]; then
        ok "wave_budget_tracker.py PASS (${BUDGET_DETAIL})"
    else
        warn "wave_budget_tracker.py ${BUDGET_VERDICT} (${BUDGET_DETAIL})"
    fi
else
    warn "wave_budget_tracker.py missing — skipping"
    BUDGET_VERDICT="SKIPPED"
    BUDGET_DETAIL="not run: wave_budget_tracker.py is missing from the tree"
fi

# -----------------------------------------------------------------------------
# Data quality checks (BLOCKER before any GPU work — tmQM d-block + PlatinAI 4-cell)
# -----------------------------------------------------------------------------
log "STEP 4/4: data quality (tmQM d-block + PlatinAI 4-cell)"
# Wf-A269 wave-9 (2026-11-02): the 4-cell check verdict is recorded in the
# plan JSON, so it must carry a value even when the data check is skipped.
PLAT_4CELL_STATUS="SKIPPED"
if [ "${SKIP_DATA_CHECK}" != "1" ]; then
    # tmQM d-block
    if [ -f "${TMQM_CSV}" ]; then
        TMQM_N=$(wc -l < "${TMQM_CSV}" 2>/dev/null || echo 0)
        if [ "${TMQM_N}" -lt 100 ]; then
            die "tmQM CSV too small (${TMQM_N} lines; expected >= 100)"
        fi
        # Verify d-block metals present
        # Wf-A269 wave-9 (2026-11-02) - REPAIR: `grep -c` prints "0" AND exits 1
        # when nothing matches, so the `|| echo 0` fallback appended a SECOND
        # "0" and the captured value became "0\n0".  `[ "0\n0" -lt 50 ]` then
        # raised "integer expression expected" and returned FALSE, so the LOW
        # branch was unreachable and a broken d-block filter printed as a
        # PASS.  `|| true` keeps grep's own "0" and discards only the exit
        # status; the `:-0` default covers a file that vanished between the
        # `-f` test above and this line.
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     TMQM_DBLOCK_HITS=$(grep -cE "(Cr|Mn|Fe|Co|Ni|Cu|Zn|Ru|Rh|Pd|Ag|Os|Ir|Pt|Au)" "${TMQM_CSV}" 2>/dev/null || echo 0)
        TMQM_DBLOCK_HITS=$(grep -cE "(Cr|Mn|Fe|Co|Ni|Cu|Zn|Ru|Rh|Pd|Ag|Os|Ir|Pt|Au)" "${TMQM_CSV}" 2>/dev/null || true)
        TMQM_DBLOCK_HITS=${TMQM_DBLOCK_HITS:-0}
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
        # Wf-A269 wave-9 (2026-11-02) - REPAIR: the 4-cell check was a ZOMBIE
        # GATE - it could not fail.  It grepped for a quoted UPPERCASE cell
        # name ("MCF7") in a WIDE csv whose columns are lowercase
        # (pIC50_mcf7 / pIC50_a549 / pIC50_a2780 / pIC50_a2780cis), so no row
        # can ever contain that string and the count was always 0; and
        # because `grep -c` prints "0" AND exits 1 on no match, the
        # `|| echo 0` fallback appended a SECOND "0", making the captured
        # value "0\n0".  The test `[ "0\n0" -lt 50 ]` then errored ("integer
        # expression expected") and returned FALSE, so the LOW branch was
        # unreachable, control fell through to the PASS branch, and the
        # wrapper printed "4-cell verified" for a check that verified
        # NOTHING about the 4 cells.
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     for CELL in "${CELLS[@]}"; do
        #         CELL_N=$(grep -c "\"${CELL}\"" "${PLAT_4CELL_CSV}" 2>/dev/null || echo 0)
        #         if [ "${CELL_N}" -lt 50 ]; then
        #             warn "PlatinAI cell ${CELL} hits low (${CELL_N}); check filter"
        #         else
        #             info "PlatinAI cell ${CELL}: ${CELL_N} rows"
        #         fi
        #     done
        #     ok "PlatinAI 4-cell verified: ${PLAT_N} lines / 4 cells"
        # The replacement asks the CSV with a real CSV parser, uses the
        # dataset builder's own "active" definition (pIC50 >= 5.0, i.e.
        # IC50 <= 10 uM; build_platinai_4cell_csv.py:345), and cross-checks
        # the counts against PLAT_4CELL_META - which was DEFINED at line 87
        # and never READ before this repair.
        PLAT_4CELL_CHECK="${SCRIPTS_DIR}/wf_platinai_4cell_check.py"
        PLAT_4CELL_REPORT="${REPORTS_DIR}/platinai_4cell_check.txt"
        PLAT_4CELL_STATUS="UNVERIFIED"
        if [ -f "${PLAT_4CELL_CHECK}" ]; then
            PLAT_4CELL_RC=0
            "${PY}" "${PLAT_4CELL_CHECK}" \
                --csv "${PLAT_4CELL_CSV}" \
                --meta "${PLAT_4CELL_META}" \
                --min-active 50 > "${PLAT_4CELL_REPORT}" 2>&1 || PLAT_4CELL_RC=$?
            sed 's/^/    /' "${PLAT_4CELL_REPORT}" | tee -a "${REPORTS_DIR}/launch.log" >/dev/null
            if [ "${PLAT_4CELL_RC}" -eq 0 ]; then
                PLAT_4CELL_STATUS="PASS"
                ok "PlatinAI 4-cell verified: ${PLAT_N} lines / 4 cells (active rows per cell line MEASURED; see ${PLAT_4CELL_REPORT})"
                # A low active count is the severity the dead branch intended:
                # a warning, not a block.
                while IFS= read -r _plat_warn; do
                    warn "PlatinAI ${_plat_warn#*FINDING [WARN]: }"
                done < <(grep -F "FINDING [WARN]" "${PLAT_4CELL_REPORT}" || true)
            else
                PLAT_4CELL_STATUS="FAIL"
                die "PlatinAI 4-cell check FAILED (rc=${PLAT_4CELL_RC}; see ${PLAT_4CELL_REPORT}); refusing to launch on data that does not carry the 4 cell lines."
            fi
        else
            die "PlatinAI 4-cell checker missing: ${PLAT_4CELL_CHECK}"
        fi
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

# Wf-A269 wave-9 (2026-11-02) - REPAIR: the plan record claimed "wrapper ready"
# and "READY_TO_RUN" unconditionally, and its multiph_pll_augmentation line
# claimed the `--use-multiph-pll` CLI flag was "wired into trainer".  MEASURED:
# the trainer declares 70 argparse flags (71 parser actions minus the
# auto-added help action) and `--use-multiph-pll` is NOT one of them (nor
# does the string occur anywhere under molmetal/ or configs/).  The strings
# below are derived from the MEASURED gate verdict instead of being asserted.
#
# Wf-A269 wave-11 (2026-11-02) - REPAIR (DEFECT A): the flag count in the
# BLOCKED string was a SHELL LITERAL ("12"), not derived from the gate's own
# report, so the plan kept claiming 12 even if the trainer's interface
# changed.  The count and the verdict are now taken from the capability gate's
# MEASURED output (${CAPABILITY_FILE}); the string is assembled from those
# variables.  The UNVERIFIED case is a third branch of its own: "the checker
# could not measure" is not "the checker measured a defect".
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     PLAN_MEASURED_TODAY="wrapper NOT launchable as written: the trainer does not declare 12 of this launcher's argv flags (MEASURED via molmetal/scripts/wf_argv_capability_check.py; see argv_capability.txt); the joint_tmqm_platinai interface was never implemented in the trainer; USER_OK gate enforced; pre-flight + per-wave + data quality wired; STATUS_HOLD warning emitted; Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation module importable (the CLI flag that would reach the trainer does not exist)"
#     PLAN_STATUS="BLOCKED_CAPABILITY"
if [ "${CAPABILITY_STATUS}" = "OK" ]; then
    PLAN_MEASURED_TODAY="wrapper ready; USER_OK gate enforced; pre-flight + per-wave + data quality wired; STATUS_HOLD warning emitted; the trainer declares every flag this launcher passes (MEASURED: 0 of ${CAPABILITY_TOTAL_COUNT} undeclared, via molmetal/scripts/wf_argv_capability_check.py; see argv_capability.txt)"
    PLAN_STATUS="READY_TO_RUN"
elif [ "${CAPABILITY_STATUS}" = "BLOCKED" ]; then
    PLAN_MEASURED_TODAY="wrapper NOT launchable as written: the trainer ${CAPABILITY_COUNT_PHRASE} (MEASURED via molmetal/scripts/wf_argv_capability_check.py; see argv_capability.txt); the joint_tmqm_platinai interface was never implemented in the trainer; USER_OK gate enforced; pre-flight + per-wave + data quality wired; STATUS_HOLD warning emitted; Multi-PH + PLL 32-bin Fourier + 128-d PLL augmentation module importable (the CLI flag that would reach the trainer does not exist)"
    PLAN_STATUS="BLOCKED_CAPABILITY"
else
    PLAN_MEASURED_TODAY="wrapper NOT launchable as written: the argv capability is UNVERIFIED, NOT a measured defect (the checker exited ${CAPABILITY_RC} and produced no flag count; see argv_capability.txt); the trainer's parse_args could not be exercised, so this launcher's argv could not be verified; USER_OK gate enforced; pre-flight + per-wave + data quality wired; STATUS_HOLD warning emitted"
    PLAN_STATUS="BLOCKED_CAPABILITY_UNVERIFIED"
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
    printf '  "tmqm_metals": [%s],\n' "$(json_array "${TMQM_METALS[@]}")"
    printf '  "platinai_4cell_csv": "%s",\n' "${PLAT_4CELL_CSV}"
    printf '  "platinai_4cell_meta": "%s",\n' "${PLAT_4CELL_META}"
    # Wf-A269 wave-9 (2026-11-02): the MEASURED verdict of the 4-cell data
    # check.  Before the repair this check was a zombie gate that could not
    # fail and always printed "verified".
    printf '  "platinai_4cell_check": "%s",\n' "${PLAT_4CELL_STATUS}"
    printf '  "platinai_4cell_check_detail": "verdict of molmetal/scripts/wf_platinai_4cell_check.py over the 4 cell-line columns (pIC50_mcf7 / pIC50_a549 / pIC50_a2780 / pIC50_a2780cis), active = pIC50 >= 5.0 (IC50 <= 10 uM), cross-checked against platinai_4cell.meta.json n_active_per_line within the round(x,4) boundary band; PASS = the CSV carries all 4 cell lines, FAIL = a declared column is absent / a label is not numeric / the meta and the CSV disagree, SKIPPED = the data check was skipped",\n'
    printf '  "cells": [%s],\n' "$(json_array "${CELLS[@]}")"
    printf '  "seeds": [%s],\n' "$(json_num_array "${SEEDS_TO_USE[@]}")"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "hidden": %d,\n' "${HIDDEN}"
    printf '  "depth": %d,\n' "${DEPTH}"
    printf '  "total_runs": %d,\n' "$(( ${#CELLS[@]} * ${#SEEDS_TO_USE[@]} ))"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": %d,\n' "${BUDGET_HR}"
    printf '  "wall_budget_label": "PROJECTED (%d GPU-hr on RTX PRO 6000 Blackwell sm_120 per Wf-328-SYN speedup ~3-4× multiplicative; PROJECTED joint ceiling ~10-15× end-to-end)",\n' "${BUDGET_HR}"
    printf '  "trainer": "%s",\n' "${TRAINER}"
    # Wf-A269 wave-11 (2026-11-02) - REPAIR (DEFECT A): the capability field
    # and its counts are DERIVED from the report the checker wrote
    # (${CAPABILITY_FILE}); no count here is a literal.  "capability" keeps the
    # bare verdict token (the same vocabulary the DRY_RUN.log's `capability:`
    # field uses — OK / BLOCKED / UNVERIFIED, asserted by
    # molmetal/tests/test_wf363a1_dry_run_log.py::VALID_CAPABILITY), and the
    # derived NUMBERS live in the two sibling fields below (null = the checker
    # produced no count, e.g. its exit-2 "no importable parse_args" case).
    printf '  "capability": "%s",\n' "${CAPABILITY_STATUS}"
    printf '  "capability_flags_undeclared": %s,\n' "${CAPABILITY_MISSING_COUNT:-null}"
    printf '  "capability_flags_total": %s,\n' "${CAPABILITY_TOTAL_COUNT:-null}"
    printf '  "capability_detail": "argv-capability gate verdict (MEASURED, derived from %s): %s; OK = the trainer declares every flag this launcher passes; BLOCKED = it does not, and the real launch path refuses before any trainer process starts; UNVERIFIED = the checker could not exercise the trainer parse_args (exit 2), which is NOT a measured defect",\n' "${CAPABILITY_FILE}" "${CAPABILITY_COUNT_CLAUSE}"
    # Wf-A269 wave-10 (2026-11-02): the MEASURED verdict of each pre-launch
    # gate.  Before the repair these three gates could not fail (see the
    # block comment above), so the plan said nothing about them and the
    # wrapper looked launchable without ever having run a pre-flight.
    # UNVERIFIED means "not measured" (the dry-run skips the live gates) —
    # it is deliberately NOT spelled PASS.
    printf '  "preflight_verdict": "%s",\n' "${PRE_FLIGHT_VERDICT}"
    printf '  "preflight_verdict_detail": "%s",\n' "$(json_escape "${PRE_FLIGHT_DETAIL}")"
    printf '  "preflight_report": "%s",\n' "${PRE_FLIGHT_JSON}"
    printf '  "checklist_verdict": "%s",\n' "${CHECKLIST_VERDICT}"
    printf '  "checklist_verdict_detail": "%s",\n' "$(json_escape "${CHECKLIST_DETAIL}")"
    printf '  "checklist_report": "%s",\n' "${CHECKLIST_JSON}"
    printf '  "budget_verdict": "%s",\n' "${BUDGET_VERDICT}"
    printf '  "budget_verdict_detail": "%s",\n' "$(json_escape "${BUDGET_DETAIL}")"
    printf '  "budget_report": "%s",\n' "${BUDGET_JSON}"
    printf '  "session_cap_hr": %d,\n' "${SESSION_CAP_HR}"
    printf '  "multiph_pll_augmentation": "Wf-348-B1 SHIPPED — 32-bin Fourier + 128-d PLL feature augmentation module (molmetal/molmetal_lam/reward/multiph_pll_feature.py) IS importable and the wrapper checks that at launch; but the --use-multiph-pll CLI flag that would carry it into the trainer is NOT declared by the trainer (MEASURED: absent from its 70 argparse flags and from every file under molmetal/ and configs/), so this augmentation is NOT reachable from this launcher",\n'
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
    # Wf-A269 wave-9 (2026-11-02) - REPAIR: the +0.1248 was labelled "F5_RDKIT_2D
    # Ridge baseline" and left un-scoped, which is the flattering direction:
    # it is the CROSS-DATASET macro R² over 38 cohorts in 3 datasets
    # (platinai_4cell 4 + metalcytotox_5metal 5 + tmqm_dblock 29), NOT the
    # PlatinAI-4-cell cohort this launcher trains on.  The 4-cell cohort's own
    # per-dataset macro R² is 0.00344.  MEASURED source:
    # metrics/by_metric/tmqm_combined_r2.json::cross_dataset_macro_r2_mean
    # = 0.12475146687654443, source artifact
    # molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json.
    printf '    "cross_dataset_r2_origin": "R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §5.3 target R² >= 0.70. CURRENT READING (MEASURED) = +0.12475 cross-dataset macro R² over 38 cohorts in 3 datasets (platinai_4cell 4 cohorts + metalcytotox_5metal 5 + tmqm_dblock 29) — NOT the PlatinAI-4-cell cohort this launcher trains on, whose own per-dataset macro R² is 0.00344. Source: metrics/by_metric/tmqm_combined_r2.json::cross_dataset_macro_r2_mean (0.12475146687654443) + molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json. It is NOT an F5_RDKIT_2D-Ridge-only baseline."\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "%s",\n' "${PLAN_MEASURED_TODAY}"
    printf '    "PROJECTED": " 5-seed full sweep ~12 GPU-hr; cross-dataset R² ≥ +0.70; Wf-348-B1 architectural lift may unblock 12-d signal floor",\n'
    # Wf-A269 wave-9 (2026-11-02) - REPAIR: the 0.004 was stated as a cap
    # ("caps R² at ~0.004") with no source, and the artifact it was attributed
    # to (molmetal/reports/wf127_l5_hilbert_diagnostic/) DOES NOT EXIST on
    # disk (MEASURED: `ls -d` → no such path).  The 0.0043 is a DERIVED
    # UPPER BOUND, not a measured R²: every F5_RDKIT_2D feature has
    # |rho| <= 0.019 on PlatinAI pIC50 (Wf-49 §Cause 2 audit), so a linear
    # model caps at 0.019^2 * 12 = 0.0043.  Source:
    # TODO/03_plans/LAMBDA_ROADMAP.md:296 + TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md.
    printf '    "STATUS_HOLD": "12-d RDKit 2D signal floor: DERIVED UPPER BOUND (not a measured R²) R² <= 0.019^2 x 12 = 0.0043, from |rho| <= 0.019 per F5_RDKIT_2D feature on PlatinAI pIC50 (Wf-49 Cause 2 audit; source TODO/03_plans/LAMBDA_ROADMAP.md:296 + TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md). MEASURED companion: Wf-122 P3 combined R² = -5.66 +/- 1.36 and Wf-129 P3 Multi-PH + PLL guarded R² = -0.0780 +/- 0.0148. NOTE: molmetal/reports/wf127_l5_hilbert_diagnostic/ does NOT exist on disk (MEASURED 2026-11-02), so it is NOT a source for any number here. W₂ regularization cannot lift R² above the derived ceiling; Wf-348-B1 32-bin Fourier + 128-d PLL augmentation is the architectural fix path",\n'
    printf '    "kill_then_adapt_directive": "> 1 hr wall; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    # Wf-A269 wave-11 (2026-11-02) - REPAIR (DEFECT B): the dry-run path
    # FORCED "PLAN_ONLY", overriding PLAN_STATUS="BLOCKED_CAPABILITY" — so a
    # plan that CANNOT be launched reported PLAN_ONLY, and the dashboard's
    # BLOCKED arm (scripts/gpu_launch_dashboard.sh `case BLOCKED*`) was DEAD
    # CODE on every dry-run plan.  PLAN_ONLY is now emitted only when the plan
    # really is launchable (capability OK); a blocked plan keeps its real
    # status so the dashboard reports BLOCKED.  The dashboard's existing
    # PLAN_ONLY / READY_TO_RUN / STALE arms are untouched.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo "${PLAN_STATUS}" )"
    if [ "${DRY_RUN}" = "1" ] && [ "${PLAN_STATUS}" = "READY_TO_RUN" ]; then
        PLAN_JSON_STATUS="PLAN_ONLY"
    else
        PLAN_JSON_STATUS="${PLAN_STATUS}"
    fi
    printf '  "status": "%s"\n' "${PLAN_JSON_STATUS}"
    printf '}\n'
} > "${PLAN_JSON}"
# Wf-A269 wave-9 (2026-11-02): the plan record was single-path, so a smoke
# dry-run DESTROYED the full-sweep plan (mode full_5seed, 12 hr, 20 pairs).
# §5.5 rule #6 requires the superseded rendering to SURVIVE, so a MODE-SPECIFIC
# durable copy is written beside the canonical one.  The canonical path is
# kept because the dashboard and the aggregators read it by exact name.
PLAN_JSON_MODE="${PLAN_JSON%.json}.${MODE_LABEL}.json"
cp -f "${PLAN_JSON}" "${PLAN_JSON_MODE}"
ok "plan emitted → ${PLAN_JSON}"
ok "plan mode record → ${PLAN_JSON_MODE} (durable; not overwritten by another mode)"

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
    log "      --cpu-num-workers 8 \\"
    log "      --out-dir ${REPORTS_DIR}/per_seed_results \\"
    log "      --device=cuda:0"
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
    # Wf-A269 wave-9 (2026-11-02) - REPAIR: the 10 hr session cap (AGENTS.md
    # §5.1 + Wf-294B kill_then_adapt) bounds a single LAUNCH.  This wrapper
    # launches N_PAIRS tuples SEQUENTIALLY, one trainer process per tuple, so
    # the per-LAUNCH wall is BUDGET_HR / N_PAIRS while the sweep TOTAL is
    # BUDGET_HR.  The PRODUCER is the only party that knows both numbers, so
    # the producer DECLARES the wall policy and the per-pair budget here and
    # the schema test VERIFIES the declaration (it does not divide, and it
    # does not trust a free-text label).  Previously the consumer divided
    # 12 hr / 20 pairs itself and gated the exception on the log's own
    # self-signed `user_ok:` field.
    PER_PAIR_BUDGET_HR="$(awk -v b="${BUDGET_HR}" -v n="${N_PAIRS}" 'BEGIN{ if (n>0) printf "%.4f", b/n; else printf "0.0000" }')"
    # Wf-A269 wave-9 (2026-11-02) - REPAIR: the DRY_RUN.log path was
    # single-path, so a smoke dry-run OVERWROTE the full-sweep record (mode
    # full_5seed, budget 12 hr, 20 pairs) with the smoke one (budget 3 hr, 8
    # pairs).  Two dry runs of the same wrapper with different modes left
    # exactly ONE record, and which one survived depended on run order.
    # §5.5 rule #6 (append-only + dated) requires the superseded rendering to
    # SURVIVE.  The canonical path is kept (the aggregators and the dashboard
    # read it by exact name) and a MODE-SPECIFIC durable record is written
    # beside it, so no mode's record can destroy another's.
    DRY_RUN_LOG_CANON="${REPORTS_DIR}/DRY_RUN.log"
    DRY_RUN_LOG_MODE="${REPORTS_DIR}/DRY_RUN.${MODE_LABEL}.log"
    for _dry_log_path in "${DRY_RUN_LOG_CANON}" "${DRY_RUN_LOG_MODE}"; do
        # shellcheck disable=SC2068
        emit_dry_run_log "${_dry_log_path}" \
            --title "Wf-355-A3 R20 §5 tmQM + PlatinAI 4-cell joint retrain (STATUS_HOLD; ${MODE_LABEL})" \
            --wrapper "$(basename "$0")" \
            --mode "${MODE_LABEL}" \
            --trainer "${TRAINER}" \
            --flags "--mode joint_tmqm_platinai --tmqm-csv ${TMQM_CSV} --platinai-csv ${PLAT_4CELL_CSV} --cells ${CELLS[*]} --tmqm-metals ${TMQM_METALS[*]} --seeds ${SEEDS_TO_USE[*]} --epochs ${EPOCHS} --hidden ${HIDDEN} --depth ${DEPTH} --use-multiph-pll --multiph-fourier-bins 32 --pll-signature-dim 128 --w2-weight 0.05 --w2-gradient-mode sort --cpu-num-workers 8 --out-dir ${REPORTS_DIR}/per_seed_results --device=cuda:0" \
            --gpu "${GPU}" \
            --budget-hr "${BUDGET_HR}" \
            --budget-label "PROJECTED" \
            --wall-policy "whole_sweep_sequential" \
            --per-pair-budget-hr "${PER_PAIR_BUDGET_HR}" \
            --user-ok "REQUIRED_STATUS_HOLD" \
            --capability "${CAPABILITY_STATUS}" \
            --gate-verdict "preflight=${PRE_FLIGHT_VERDICT} checklist=${CHECKLIST_VERDICT} budget=${BUDGET_VERDICT}" \
            ${pair_args[@]+"${pair_args[@]}"}
    done
    ok "DRY_RUN.log emitted → ${DRY_RUN_LOG_CANON} (${N_PAIRS} (cell, seed) pairs; per-pair budget ${PER_PAIR_BUDGET_HR} hr)"
    ok "DRY_RUN.log mode record → ${DRY_RUN_LOG_MODE} (durable; not overwritten by another mode)"
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
        # `setsid --wait`: each (cell, seed) run gets its own session so it
        # survives the invoking shell, and the `if` still tests the trainer's
        # own exit status. Bare `setsid` may fork and return 0 immediately,
        # which would make the failure branch unreachable.
        # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
        if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait "${PY}" "${TRAINER}" \
                --config "${DMPNN_CONFIG}" \
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
                --cpu-num-workers 8 \
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
