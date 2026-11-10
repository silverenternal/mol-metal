#!/usr/bin/env bash
# Wf-355-B1 — Wf-131 P3 7 DEFERRED (metal, seed) runs re-launch wrapper
#                 USER_OK-gated GPU launch wrapper
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-355-B1 ultracode residual wave)
# Parent:      Wf-354-A1 §5 HONEST-NEGATIVE ("Wf-131 P3 Re-launch wrapper
#              needed (USER_OK-gated for >1hr GPU per AGENTS.md §5.1)")
# Trainers:    molmetal/scripts/train_metal_lomo_wasserstein.py
#              (48958 bytes — Wf-125 P2 + Wf-131 P3 canonical entry; per
#               Wf-354-A1 §3.2 the D-MPNN LOMO logic is NOT in
#               train_dmpnn_multitask_wasserstein.py; the train_metal_lomo*
#               family is the canonical LOMO entry point).
#
# Refs:
#   - TODO/completed/354a1_wf342_p2_lomo_integration.md §1.2 + §5 (per-run
#     DEFERRED status)
#   - metrics/by_metric/metal_lomo_auc.json `wf131_p3_lomo_partial_update`
#     (MEASURED 18/25 = 72% complete on BUGGY Wf-125 P1 cohort)
#   - TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.4 (acceptance gates)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#   - Wf-335-R1+R3 (--no-cuda-graphs safety lever for Blackwell sm_120)
#   - Wf-328-W2 (--cpu-num-workers 8 — NOT supported by train_metal_lomo*; removed Wf-379-A1)
#   - Wf-328-W5 (--w2-gradient-mode sort — NOT supported by train_metal_lomo*; removed Wf-379-A1)
#
# HONEST FRAMING, Refs, Usage, USER_OK gate, stop semantics and the pre-launch
# checklist are preserved VERBATIM in configs/wf131_p3_deferred_runs.yaml
# (header, "WRAPPER HEADER — VERBATIM") and in
# molmetal/reports/wfa269_yaml_migration/launch_wf131_p3_deferred_runs.before.sh
# (the whole pre-migration file, header included). Summary of that block:
#   - USER_OK=1 required; --dry-run does NOT bypass it; --runs "Rh 3 Rh 4 Re 0"
#     picks a subset of the 7 runs.
#   - Scope: 7 DEFERRED (metal, seed) runs on the BUGGY Wf-125 P1 cohort
#     (Rh 3, Rh 4, Re 0..4); ~5 hr GPU wall PROJECTED (Wf-260 G6 timing).
#   - MEASURED (BUGGY cohort): Rh fold 3/5 seeds, mean AUC 0.6874 ± 0.3584
#     (seed0=0.274 outlier); Re fold DEFERRED 0/5.
#   - Stop semantics (TODO-207 v2 §7.2): ckpt older than 2x projected wall, 3+
#     consecutive NaN/OOM/IndexError, GPU util < 70% for 5+ min, Rh fold AUC
#     < 0.10 on 2+ seeds.
#   - Pre-launch: pre_flight_check.py --pipeline launch, wave_budget_tracker.py
#     --budget-hr 5, USER_OK env gate, trainer + cohort + CSV existence.

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
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf131_p3_deferred_runs"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_wasserstein.py"
# AGENTS.md §5.11 ("no cli参数"): the trainer is launched from a YAML config and
# nothing else — every CLI hyperparameter now lives in BASE_CONFIG (whose header
# carries the flag -> dest map + the verbatim pre-migration argv).
BASE_CONFIG="${PROJECT_ROOT}/configs/wf131_p3_deferred_runs.yaml"
PAIR_CONFIG_HELPER="${PROJECT_ROOT}/scripts/_wf_yaml_pair_config.py"
# The config is authoritative; these copies exist only for the existence
# check and are checked against it by --expect.
LOMO_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"
RH_COHORT_JSON="${PROJECT_ROOT}/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json"

# 7 DEFERRED runs (per Wf-354-A1 §5 HONEST-NEGATIVE): 14 array elements = 7
# "metal seed" pairs, same convention as --runs "Rh 3 Rh 4 Re 0".
DEFAULT_RUNS=("Rh" 3 "Rh" 4 "Re" 0 "Re" 1 "Re" 2 "Re" 3 "Re" 4)

# Wall budget (PROJECTED per Wf-260 G6 timing)
WALL_BUDGET_HR=5
# Epochs + device: AUTHORITATIVE in BASE_CONFIG (checked by --expect).
EPOCHS=50
GPU="cuda:0"               # RTX PRO 6000 Blackwell GPU-0 (serial execution)

mkdir -p "${REPORTS_DIR}"

# Die helper — MOVED UP before its first call (Wf-A269 repair 2, 2026-11-02).
die()  { mkdir -p "${REPORTS_DIR}" 2>/dev/null || true; printf '%s[FATAL]%s %s\n' "${C_RED:-}" "${C_RST:-}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run\n' "$0" >&2
    printf '       USER_OK=1 bash %s --runs "Rh 3 Re 0"   # subset\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --runs)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                RUNS+=("$1")
                shift
            done
            ;;
        -h|--help)  sed -n '2,75p' "$0"; exit 0 ;;
        *)          printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Default to all 7 DEFERRED runs if --runs not specified
if [ "${#RUNS[@]}" -eq 0 ]; then
    RUNS=("${DEFAULT_RUNS[@]}")
fi

# Validate runs are space-separated (metal seed) pairs
N_RUNS=$((${#RUNS[@]} / 2))
if [ "$((${#RUNS[@]} % 2))" -ne 0 ]; then
    printf '%s[FATAL]%s --runs takes an even number of args (metal seed pairs); got %d\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" "${#RUNS[@]}" >&2
    exit 65
fi

# Color codes (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
ok()   { printf '[%s][OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ]   || die "PY not found: ${PY}"
[ -f "${TRAINER}" ]              || die "trainer missing: ${TRAINER}"
[ -f "${BASE_CONFIG}" ]          || die "base config missing: ${BASE_CONFIG}"
[ -f "${PAIR_CONFIG_HELPER}" ]   || die "pair-config helper missing: ${PAIR_CONFIG_HELPER}"
[ -f "${LOMO_CSV}" ]             || die "LOMO CSV missing: ${LOMO_CSV}"
[ -f "${RH_COHORT_JSON}" ]       || die "Rh cohort JSON missing: ${RH_COHORT_JSON}"

log "Wf-355-B1 Wf-131 P3 DEFERRED runs re-launch wrapper — USER_OK=1 confirmed"
log "  total_runs=${N_RUNS} (default 7)"
log "  gpu=${GPU}"
log "  epochs=${EPOCHS}"
log "  wall_budget_hr=${WALL_BUDGET_HR}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run)
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf131_p3_deferred_runs.sh",\n'
    printf '  "wf_origin": "Wf-355-B1 (ultracode residual wave)",\n'
    printf '  "parent_plan": "TODO/completed/354a1_wf342_p2_lomo_integration.md §5",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metals": ['
    first=1
    for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
        if [ "${first}" -eq 1 ]; then first=0; else printf ', '; fi
        printf '"%s"' "${RUNS[$i]}"
    done
    printf '],\n'
    printf '  "seeds": ['
    first=1
    for ((i = 1; i < ${#RUNS[@]}; i += 2)); do
        if [ "${first}" -eq 1 ]; then first=0; else printf ', '; fi
        printf '%s' "${RUNS[$i]}"
    done
    printf '],\n'
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "total_runs": %d,\n' "${N_RUNS}"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": %d,\n' "${WALL_BUDGET_HR}"
    printf '  "wall_budget_label": "PROJECTED (~5 hr GPU sequential per Wf-260 G6 timing; Rh small cohort ~15 min/run + Re smallest ~30 min/run)",\n'
    printf '  "trainer": "%s",\n' "${TRAINER}"
    printf '  "lomo_csv": "%s",\n' "${LOMO_CSV}"
    printf '  "rh_cohort_json": "%s",\n' "${RH_COHORT_JSON}"
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; 7 DEFERRED runs enumerated from Wf-354-A1 §5",\n'
    printf '    "PROJECTED": "Rh seeds 3+4 may clarify if seed0=0.274 outlier is fluke or structural; Re fold 5/5 will establish per-metal lift baseline on BUGGY cohort",\n'
    printf '    "rh_status_carryover": "BUGGY Wf-125 P1 cohort (constant-inactive labels, see TODO-135 §0); Rh fold MEASURED 3/5 = 0.6874 ± 0.3584 with seed0=0.274 outlier",\n'
    printf '    "re_status_carryover": "Re fold DEFERRED (0/5) — Re cohort smallest (~800 rows); per-seed AUC variance may be high",\n'
    printf '    "kill_then_adapt_directive": "5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation per (metal, seed) tuple would be:"
    log "  ${PY} ${PAIR_CONFIG_HELPER} --base ${BASE_CONFIG} --out <derived> --metal <METAL> --seed <SEED>"
    log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} --config <derived per-pair config> --out-dir ${REPORTS_DIR}/<METAL>/seed<SEED>"
    log ""
    log "DRY-RUN: per-run sequencing (${N_RUNS} (metal, seed) tuples):"
    for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
        log "  ${RUNS[$i]} seed=${RUNS[$((i+1))]}  ->  ${REPORTS_DIR}/${RUNS[$i]}/seed${RUNS[$((i+1))]}/"
    done
    log ""
    log "DRY-RUN: total wall = ~5 hr GPU sequential = ~5 calendar hours worst case"

    # Wf-363-A1: emit structured DRY_RUN.log with per-pair enumeration
    # Build per-pair arg list for the schema helper
    pair_args=()
    for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
        pair_args+=( --pair "${RUNS[$i]} seed=${RUNS[$((i+1))]}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${REPORTS_DIR}/DRY_RUN.log" \
        --title "Wf-355-B1 Wf-131 P3 DEFERRED runs (USER_OK-gated)" \
        --wrapper "$(basename "$0")" \
        --mode "DEFERRED_${N_RUNS}_runs" \
        --trainer "${TRAINER}" \
        --flags "--config <derived per-pair config> --out-dir ${REPORTS_DIR}/<METAL>/seed<SEED>" \
        --gpu "${GPU}" \
        --budget-hr "${WALL_BUDGET_HR}" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    ok "DRY_RUN.log emitted → ${REPORTS_DIR}/DRY_RUN.log"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 7-run sweep. Each (metal, seed) tuple runs a
# full LOMO: hold out the focal metal, train on the remaining 4 + the Wf-125 P1
# Rh synthetic cohort, evaluate on the held-out metal.
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)

# Per Wf-131 P3 partial update (metal_lomo_auc.json:280-285):
#   - Training stopped at Rh seed=3 epoch=23 (2026-09-27 09:05:40)
#   - Re not started (0/5 seeds)
#   - This re-launch CONTINUES from those DEFERRED positions
for ((i = 0; i < ${#RUNS[@]}; i += 2)); do
    METAL="${RUNS[$i]}"
    SEED="${RUNS[$((i+1))]}"
    log "=== metal=${METAL} seed=${SEED} on ${GPU} ==="
    # GPU 0 (RTX PRO 6000 Blackwell) per the 2026-10-03 all-GPU-0
    # directive; this call site is serial with every other job.
    # `setsid --wait` puts each run in its own session so the sweep survives
    # the invoking shell, and still makes the `if` test the TRAINER's status:
    # bare `setsid` may fork and return 0 before the run starts, which would
    # turn `die` into an unreachable branch.
    # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
    # AGENTS.md §5.11 — YAML-only: the per-pair config carries every CLI value.
    DERIVED_CONFIG="${REPORTS_DIR}/configs/${METAL}_seed${SEED}.yaml"
    if ! "${PY}" "${PAIR_CONFIG_HELPER}" \
            --base "${BASE_CONFIG}" --out "${DERIVED_CONFIG}" \
            --metal "${METAL}" --seed "${SEED}" \
            --expect "protocol.epochs=${EPOCHS}" --expect "compute.device=${GPU}" \
            --expect "csv=${LOMO_CSV}" --expect "data.rh_cohort_json=${RH_COHORT_JSON}" \
            --set "output.out_dir=${REPORTS_DIR}/${METAL}/seed${SEED}" --set "output.out_metric=${REPORTS_DIR}/${METAL}/seed${SEED}/metrics.json" \
            > "${REPORTS_DIR}/config_derive.log" 2>&1; then
        die "config derivation failed for ${METAL}/seed${SEED} (see ${REPORTS_DIR}/config_derive.log); aborting sweep"
    fi

    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    # (2026-11-03, AGENTS.md §5.11 YAML-only migration. Every line below is the
    #  ORIGINAL line byte-for-byte (same flags, same order, same literal values,
    #  same indentation) after the "# " comment prefix, so the migration can be
    #  audited against the exact text it replaced. Every flag it carried now lives
    #  in configs/wf131_p3_deferred_runs.yaml; the wrapper passes only --config and
    #  --out-dir.
    # pre-migration trainer invocation, its DRY-RUN rendering and its DRY_RUN.log
    # --flags string live byte-for-byte in all three of:
    # 1) molmetal/reports/wfa269_yaml_migration/launch_wf131_p3_deferred_runs.before.sh  (the whole pre-migration file)
    # 2) configs/wf131_p3_deferred_runs.yaml  (header)
    # 3) molmetal/reports/wfa269_yaml_migration/launch_wf131_p3_deferred_runs.json  (old_argv)
    # #
    # [SUPERSEDED 1 -- the pre-migration trainer invocation, verbatim]
    #     if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait "${PY}" "${TRAINER}" \
    #             --metals "${METAL}" \
    #             --seeds "${SEED}" \
    #             --epochs "${EPOCHS}" \
    #             --wasserstein-weight 0.05 \
    #             --rh-oversample 500 \
    #             --rh-cohort-json "${RH_COHORT_JSON}" \
    #             --csv "${LOMO_CSV}" \
    #             --use-l2-guard \
    #             --out-dir "${REPORTS_DIR}/${METAL}/seed${SEED}" \
    #             --out-metric "${REPORTS_DIR}/${METAL}/seed${SEED}/metrics.json" \
    #             --device cuda:0 \
    #             >> "${RUN_LOG}" 2>&1; then
    # #
    # [SUPERSEDED 2 -- the pre-migration DRY-RUN rendering, verbatim]
    #     log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} \\"
    #     log "    --metals <METAL> --seeds <SEED> \\"
    #     log "    --epochs ${EPOCHS} \\"
    #     log "    --wasserstein-weight 0.05 \\"
    #     log "    --rh-oversample 500 \\"
    #     log "    --rh-cohort-json ${RH_COHORT_JSON} \\"
    #     log "    --csv ${LOMO_CSV} \\"
    #     log "    --use-l2-guard \\"
    #     log "    --device=cuda:0"
    # #
    # [SUPERSEDED 3 -- the pre-migration DRY_RUN.log --flags string, verbatim]
    #         --flags "--metals <METAL> --seeds <SEED> --epochs ${EPOCHS} --wasserstein-weight 0.05 --rh-oversample 500 --rh-cohort-json ${RH_COHORT_JSON} --csv ${LOMO_CSV} --use-l2-guard --device=cuda:0" \
    if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait "${PY}" "${TRAINER}" \
            --config "${DERIVED_CONFIG}" \
            --out-dir "${REPORTS_DIR}/${METAL}/seed${SEED}" \
            >> "${RUN_LOG}" 2>&1; then
        ok "metal=${METAL} seed=${SEED} completed"
    else
        die "metal=${METAL} seed=${SEED} failed (see ${RUN_LOG}); aborting sweep"
    fi
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_hr=$(( elapsed_sec / 3600 ))
log "Wf-355-B1 Wf-131 P3 DEFERRED runs sweep done — elapsed=${elapsed_hr} hr (budget=${WALL_BUDGET_HR} hr)"

if [ "${elapsed_hr}" -gt "${WALL_BUDGET_HR}" ]; then
    warn "elapsed (${elapsed_hr} hr) exceeded PROJECTED wall budget (${WALL_BUDGET_HR} hr); flagging as MEASURED-over-PROJECTED in plan.json"
    # Patch plan.json with measured wall time
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-(metal,seed) JSONs"
ok "Next: append per-(metal,seed) AUCs to metrics/by_metric/metal_lomo_auc.json (wf131_p3_lomo_partial_update block)"
ok "Next: re-emit paper §4.5.6 honest-framing decision (per TODO-84 rule)"
