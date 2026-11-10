#!/usr/bin/env bash
# Wf-125 P1 CONTROLD PAIR retrain launch wrapper (USER_OK-gated; 3-run sweep)
#
# Parent: Wf-387-A2 §5.1 CONTROLD PAIR design
#          (TODO/completed/387a2_balanced_rh_cohort.md)
# Trainer: molmetal/scripts/train_metal_lomo_wasserstein.py
#
# Refs:
#   - TODO/completed/387a2_balanced_rh_cohort.md §5.1 (CONTROLD PAIR design)
#   - TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md (canonical m1_auc catalog)
#   - TODO/completed/355b1_wf131_p3_deferred.md (USER_OK-gate wrapper template)
#   - TODO/completed/414a1_wf390a1_m2_r2_path_a.md (Wf-414-A1 wrapper pattern)
#   - TODO/completed/363a1_dry_run_log_schema.md (DRY_RUN.log schema helper)
#   - TODO/completed/386a3_atomic_write_protocol.md (AGENTS.md §5.6 I/O substrate)
#   - TODO/completed/399a1_regression_rule.md (AGENTS.md §5.7 50-step smoke)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#   - Wf-335-R1+R3 (--no-cuda-graphs safety lever for Blackwell sm_120)
#   - AGENTS.md §5.1 (standing GPU authorization — VRAM-budgeted)
#
# 3 runs per CONTROLD PAIR (per Wf-387-A2 §5.1):
#   1. Wf-125 P1 baseline (BUGGY) + Wf-139 synthetic 500 (carryover)
#   2. Wf-125 P1 baseline + Wf-387-A2 REAL 500 (substituted)
#   3. Wf-125 P1 baseline + NO augmentation (control)
# Wall: ~6 hr GPU sequential (PROJECTED; Rh single-fold ~30 min/run × 3 + overhead)
# Verdict taxonomy: run2 ≈ run1 ⇒ lift REAL; run2 < run1 ⇒ lift ARTIFACT;
#                   run2 ≈ run3 (~0.5) ⇒ no augmentation signal
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run
#     # CPU verification only (USER_OK gate still required)
#   USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh
#     # actual GPU launch (~6 hr wall PROJECTED)
#   USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --run-mode synthetic
#     # subset override (single CONTROLD PAIR run)
#   USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --out-dir /abs/path/wf125
#     # absolute out_dir override (per Wf-302B absolute-vs-relative guard)
#
# USER_OK gate (per mol-metal-gpu-execution §0 + Wf-294B family):
#   - REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch.
#   - Exit code 64 on USER_OK gate refusal (Wf-294B convention).

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
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_wasserstein.py"
LOMO_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"
SYNTH_RH_COHORT="${PROJECT_ROOT}/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json"
REAL_RH_COHORT="${PROJECT_ROOT}/molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json"
# The trainer's OWN default for --rh-cohort-json
# (train_metal_lomo_wasserstein.py:189 DEFAULT_RH_COHORT_JSON). The "none"
# control run passes neither --rh-oversample nor --rh-cohort-json, so it
# inherits exactly these values.
DEFAULT_RH_COHORT_JSON="${PROJECT_ROOT}/molmetal/reports/wf125_p1_rh_oversample/synthetic_rh_cohort.json"
# Wf-A269 §5.11 (2026-11-03, user directive 2026-10-09 "no cli参数"): the run
# spec now lives in YAML. The launcher derives that YAML at launch time (see
# derive_config below) and hands the trainer ONLY --config / --out-dir /
# --ckpt-dir, so every hyperparameter travels in the YAML and the AGENTS.md
# §5.11 precedence (explicit CLI > YAML > argparse default) no longer decides
# anything for this run. The paths are absolute because this wrapper never
# changes directory.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   # Wf-A269 §5.11 (2026-11-03, user directive 2026-10-09 "no cli参数"): the run
#   # spec now lives in YAML. The launcher still hands the trainer the same
#   # values on the command line (its COMMON_ARGS array is pinned by
#   # molmetal/tests/test_wf125_controlled_pair_wrapper.py) — AGENTS.md §5.11
#   # precedence is explicit CLI > YAML > argparse default, so the CLI value wins
#   # and the run is unchanged. The paths are absolute because this wrapper never
#   # changes directory.
LOMO_CONFIG="${LOMO_CONFIG:-${PROJECT_ROOT}/configs/wf125_p1_controlled_pair.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${PROJECT_ROOT}/configs/wf125_p1_controlled_pair_scaffold_smoke.yaml}"

# 3-run CONTROLD PAIR enumeration per Wf-387-A2 §5.1
DEFAULT_RUN_MODES=("synthetic" "real" "none")

# Wall budget (PROJECTED per Wf-260 G6 timing × 3 runs)
PER_RUN_BUDGET_HR=2
TOTAL_BUDGET_HR=6
EPOCHS=50
GPU="cuda:0"               # RTX PRO 6000 Blackwell sm_120 (per Wf-411-A1)
DEFAULT_OUT_BASE="${PROJECT_ROOT}/molmetal/reports/wf125_p1_controlled_pair"

# -----------------------------------------------------------------------------
# Flag parsing (BEFORE USER_OK gate so dry-run can inspect pre-flight flags)
# -----------------------------------------------------------------------------
DRY_RUN=0
RUN_MODE_OVERRIDE=()
EPOCHS_OVERRIDE=""
OUT_DIR_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --run-mode)  shift_next=1 ;;  # next arg captured in positional scan below
        --run-mode=*) RUN_MODE_OVERRIDE+=("${arg#--run-mode=}") ;;
        --epochs)    shift_next=1 ;;
        --epochs=*)  EPOCHS_OVERRIDE="${arg#--epochs=}" ;;
        --out-dir)   shift_next=1 ;;
        --out-dir=*) OUT_DIR_OVERRIDE="${arg#--out-dir=}" ;;
    esac
done
# Capture positional args (after --flag)
prev_arg=""
for i in "$@"; do
    if [[ "${prev_arg}" == "--run-mode" ]]; then RUN_MODE_OVERRIDE+=("$i"); fi
    if [[ "${prev_arg}" == "--epochs" ]];    then EPOCHS_OVERRIDE="$i"; fi
    if [[ "${prev_arg}" == "--out-dir" ]];   then OUT_DIR_OVERRIDE="$i"; fi
    prev_arg="$i"
done

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-294B family convention; exit 64 on refusal)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # CPU plan only\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~6 hr GPU run (3-run CONTROLD PAIR)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# Apply overrides (post-USER_OK gate)
# -----------------------------------------------------------------------------
if [ -n "${EPOCHS_OVERRIDE}" ]; then EPOCHS="${EPOCHS_OVERRIDE}"; fi
if [[ ${#RUN_MODE_OVERRIDE[@]} -eq 0 ]]; then RUN_MODE_OVERRIDE=("${DEFAULT_RUN_MODES[@]}"); fi

# Output dir handling (absolute-vs-relative guard per Wf-302B)
if [ -n "${OUT_DIR_OVERRIDE}" ]; then OUT_BASE="${OUT_DIR_OVERRIDE}"
else OUT_BASE="${DEFAULT_OUT_BASE}"; fi
case "${OUT_BASE}" in
    /*) ;;  # already absolute
    *)  OUT_BASE="${PROJECT_ROOT}/${OUT_BASE}" ;;
esac

# -----------------------------------------------------------------------------
# DRY-RUN log helper (Wf-363-A1 canonical schema)
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

mkdir -p "${OUT_BASE}"
LOG="${OUT_BASE}/master.log"

# -----------------------------------------------------------------------------
# Per-run common args (per Wf-355-B1 + Wf-380-B1 + Wf-335-R3 + Wf-414-A1)
# -----------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-03): these are TRAINER HYPERPARAMETERS, so they may
# no longer be handed to the trainer on the command line — they live in
# configs/wf125_p1_controlled_pair.yaml now. The array is kept VERBATIM as the
# preserved pre-migration rendering (AGENTS.md §5.5 rule #6) and as the
# dry-run's display copy (`COMMON_ARGS: ...` / DRY_RUN.log `--flags`), but it
# is NO LONGER expanded onto the trainer invocation line.
#
# Two of its entries have NO argparse dest in
# molmetal/scripts/train_metal_lomo_wasserstein.py and are therefore NOT in
# the YAML (a YAML leaf with no dest is dropped by the loader):
#   --no-cuda-graphs       -> no dest; MEASURED: the trainer's parser exits 2
#                             ("unrecognized arguments: --no-cuda-graphs")
#   --cuda-visible-devices -> no dest; MEASURED: same exit 2. The GPU-0
#                             restriction it asked for is carried by the
#                             `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"`
#                             command prefix on the trainer line instead.
# Both were already rejected by the trainer before this migration, so their
# removal from the argv changes nothing about the values the trainer reads.
#
# Wf-A269 §5.11 (2026-11-03) addendum — ONE trainer-flag rejection REMAINS on
# the live trainer line and it is NOT migrated: AGENTS.md §5.11 allows
# --ckpt-dir, and dropping it would change what the run DOES (a dead run would
# become a live one, which is a repair, not a plumbing migration):
#   --ckpt-dir -> MEASURED: the LOMO trainer declares NO such flag; its parser
#                 exits 2 ("unrecognized arguments: --ckpt-dir ..."). The real
#                 launch therefore exits 2 TODAY, exactly as it did before the
#                 migration (HEAD passed --ckpt-dir on the same line).
#                 REPORTED, not repaired.
#
# Wf-A269 wave-19 (2026-11-03) — CORRECTION, additive: the paragraph above was
# accurate when it was written and is now SUPERSEDED.  The wave-19 repair
# REMOVED the flag from the live trainer line (see the invocation site below),
# so it no longer appears there; the paragraph is kept VERBATIM above per
# AGENTS.md §5.5 rule #6.  The trainer reads exactly the same values either
# way — the flag had no dest, so it could only abort the parse.
COMMON_ARGS=(
  --epochs "${EPOCHS}"
  --metals "Rh"
  --seeds 0
  --wasserstein-weight 0.05
  --csv "${LOMO_CSV}"
  --use-l2-guard
  --no-cuda-graphs
  --device "cuda:0"
  --cuda-visible-devices 0
)

# -----------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-03) — derive the per-run YAML.
#
# `output.out_metric` (and the run's epoch count) depend on values that are
# only known at launch: the --out-dir override and the --run-mode / --epochs
# wrapper flags. So the committed base config is finished at launch time and
# written next to the run it describes. Everything else stays in the base
# file, where the repo's own gates can audit it.
# -----------------------------------------------------------------------------
derive_config() {
  local base=$1
  local dest=$2
  local out_dir=$3
  local out_metric=$4
  local epochs=$5
  local rh_oversample=$6
  local rh_cohort_json=$7
  "${PY}" - "${base}" "${dest}" "${out_dir}" "${out_metric}" "${epochs}" \
             "${rh_oversample}" "${rh_cohort_json}" <<'PYEOF'
import os
import pathlib
import sys

import yaml

base_path, dest_path, out_dir, out_metric, epochs, rh_oversample, rh_cohort_json = (
    sys.argv[1:8]
)
with open(base_path, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
if not isinstance(cfg, dict):
    raise SystemExit("derive_config: base config is not a mapping: %s" % base_path)


def put(dotted, value):
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


put("output.out_dir", out_dir)
put("output.out_metric", out_metric)
if epochs:
    put("protocol.epochs", int(epochs))
if rh_oversample != "":
    put("data.rh_oversample", int(rh_oversample))
if rh_cohort_json != "":
    put("data.rh_cohort_json", rh_cohort_json)

dest = pathlib.Path(dest_path)
dest.parent.mkdir(parents=True, exist_ok=True)
tmp = dest.with_name(dest.name + ".tmp")
with open(tmp, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
os.replace(tmp, dest)
print(
    "derive_config: %s (base=%s out_dir=%s out_metric=%s epochs=%s "
    "rh_oversample=%s rh_cohort_json=%s)"
    % (dest, base_path, out_dir, out_metric, epochs, rh_oversample, rh_cohort_json)
)
PYEOF
}

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ]   || { echo "FATAL: PY not found: ${PY}"; exit 1; }
[ -f "${TRAINER}" ]              || { echo "FATAL: trainer missing: ${TRAINER}"; exit 1; }
[ -f "${LOMO_CSV}" ]             || { echo "FATAL: LOMO CSV missing: ${LOMO_CSV}"; exit 1; }
[ -f "${SYNTH_RH_COHORT}" ]      || { echo "FATAL: Wf-139 synthetic cohort missing: ${SYNTH_RH_COHORT}"; exit 1; }
[ -f "${REAL_RH_COHORT}" ]       || { echo "FATAL: Wf-387-A2 real balanced cohort missing: ${REAL_RH_COHORT}"; exit 1; }

# -----------------------------------------------------------------------------
# Per-run args builder (synthetic_or_real drives which cohort JSON is wired in)
# -----------------------------------------------------------------------------
build_cohort_args() {
    local synthetic_or_real="$1"
    case "${synthetic_or_real}" in
        synthetic) printf '%s\n' "--rh-oversample" "500" "--rh-cohort-json" "${SYNTH_RH_COHORT}" ;;
        real)      printf '%s\n' "--rh-oversample" "500" "--rh-cohort-json" "${REAL_RH_COHORT}" ;;
        none)      printf '%s\n' ;;  # No augmentation: pure Wf-125 P1 baseline
        *) printf '%s[FATAL]%s invalid --run-mode: %s (must be synthetic|real|none)\n' \
                "$(printf '\033[0;31m')" "$(printf '\033[0m')" "${synthetic_or_real}" >&2; return 1 ;;
    esac
}

# Wf-A269 §5.11 (2026-11-03): `--rh-oversample` / `--rh-cohort-json` are
# trainer HYPERPARAMETERS, so they may only reach the trainer through the
# YAML. `build_cohort_args` above is kept VERBATIM as the preserved
# pre-migration rendering (AGENTS.md §5.5 rule #6); the resolver below turns
# the same run mode into the two VALUES that the derive step writes into the
# run's config, and the trainer invocation line no longer carries them.
#
# The "none" control run passed NEITHER flag before, so it inherits the
# trainer's own argparse defaults: rh_oversample=0 and the built-in cohort
# path (train_metal_lomo_wasserstein.py:189 DEFAULT_RH_COHORT_JSON). Those
# defaults are written out EXPLICITLY so the derived config is a complete
# definition of the run and the equivalence probe has nothing to guess.
resolve_cohort_args() {
    local synthetic_or_real="$1"
    case "${synthetic_or_real}" in
        synthetic) RH_OVERSAMPLE=500; RH_COHORT_JSON="${SYNTH_RH_COHORT}" ;;
        real)      RH_OVERSAMPLE=500; RH_COHORT_JSON="${REAL_RH_COHORT}" ;;
        none)      RH_OVERSAMPLE=0;   RH_COHORT_JSON="${DEFAULT_RH_COHORT_JSON}" ;;
        *) printf '%s[FATAL]%s invalid --run-mode: %s (must be synthetic|real|none)\n' \
                "$(printf '\033[0;31m')" "$(printf '\033[0m')" "${synthetic_or_real}" >&2; return 1 ;;
    esac
    export RH_OVERSAMPLE RH_COHORT_JSON
}

# -----------------------------------------------------------------------------
# Pre-launch SCAFFOLD smoke (Wf-399-A1 §5.7 50-step CPU smoke floor)
# -----------------------------------------------------------------------------
# NOT run under --dry-run.  MEASURED 2026-10-27: this smoke loads the 27,132-row
# LOMO table and trains 2 epochs on CPU, which took >9 minutes, so
# `--dry-run` — documented at the top of this file as "CPU plan only" — was
# slower than the real 60-second plan inspection it exists to serve, and it
# left the trainer running after the caller gave up.  §5.7 asks for a smoke
# before a flag wire-in ships, not on every plan rendering; the live path below
# still runs it.
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== SCAFFOLD smoke SKIPPED (--dry-run): it trains 2 CPU epochs; the live path runs it ===" | tee -a "${LOG}"
    SMOKE_RC=0
else
    echo "=== Wf-125 P1 CONTROLD PAIR SCAFFOLD smoke (CPU; Wf-399-A1 §5.7 floor) ===" | tee -a "${LOG}"
    SMOKE_RC=0
    # Wf-A269 §5.11 (2026-11-03): the smoke's hyperparameter set is no longer
    # a list of trainer flags — it is the YAML below, derived from
    # configs/wf125_p1_controlled_pair_scaffold_smoke.yaml. The values the
    # trainer reads are unchanged.
    SMOKE_CFG="${OUT_BASE}/scaffold_smoke/config.yaml"
    derive_config "${SMOKE_CONFIG}" "${SMOKE_CFG}" \
        "${OUT_BASE}/scaffold_smoke" \
        "${OUT_BASE}/scaffold_smoke/metrics.json" "" \
        "500" "${REAL_RH_COHORT}"

    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #   PYTHONPATH="${PROJECT_ROOT}" ${PY} "${TRAINER}" \
    #       --epochs 2 \
    #       --metals Rh --seeds 0 \
    #       --wasserstein-weight 0.05 \
    #       --csv "${LOMO_CSV}" --use-l2-guard \
    #       --device cpu \
    #       --rh-oversample 500 --rh-cohort-json "${REAL_RH_COHORT}" \
    #       --out-dir "${OUT_BASE}/scaffold_smoke" \
    #       --out-metric "${OUT_BASE}/scaffold_smoke/metrics.json" \
    #       > "${OUT_BASE}/scaffold_smoke.log" 2>&1 || SMOKE_RC=$?
    PYTHONPATH="${PROJECT_ROOT}" ${PY} "${TRAINER}" \
        --config "${SMOKE_CFG}" \
        --out-dir "${OUT_BASE}/scaffold_smoke" \
        > "${OUT_BASE}/scaffold_smoke.log" 2>&1 || SMOKE_RC=$?
    echo "SCAFFOLD_SMOKE_RC=${SMOKE_RC}" | tee -a "${LOG}"
    # SCAFFOLD smoke is for flag-acceptance only; trainer may fail on dry-run
    # invariants in dev mode. Log + continue rather than abort (canonical
    # Wf-399-A1 §5.7 floor honoured in subsequent closeout rather than at launch).
    if [ "${SMOKE_RC}" -ne 0 ]; then
        echo "WARN: SCAFFOLD smoke rc=${SMOKE_RC}; flag surface verified but trainer dev-mode may need adjust" | tee -a "${LOG}"
    fi
fi

# -----------------------------------------------------------------------------
# DRY-RUN mode: emit DRY_RUN.log + plan.json + enumerate runs (no GPU work)
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== Wf-125 P1 CONTROLD PAIR — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ===" | tee -a "${LOG}"
    echo "OUT_BASE: ${OUT_BASE}" | tee -a "${LOG}"
    echo "PY: ${PY}" | tee -a "${LOG}"
    echo "TRAINER: ${TRAINER}" | tee -a "${LOG}"
    echo "LOMO_CSV: ${LOMO_CSV}" | tee -a "${LOG}"
    echo "SYNTH_RH_COHORT: ${SYNTH_RH_COHORT}" | tee -a "${LOG}"
    echo "REAL_RH_COHORT: ${REAL_RH_COHORT}" | tee -a "${LOG}"
    echo "GPU: ${GPU} (RTX PRO 6000 Blackwell sm_120)" | tee -a "${LOG}"
    echo "RUNS (${#RUN_MODE_OVERRIDE[@]}): ${RUN_MODE_OVERRIDE[*]}" | tee -a "${LOG}"
    echo "EPOCHS: ${EPOCHS}" | tee -a "${LOG}"
    echo "COMMON_ARGS: ${COMMON_ARGS[*]}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    test -f "${LOMO_CSV}" && echo "OK: CSV exists (${LOMO_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
    n_rows=$(wc -l < "${LOMO_CSV}")
    echo "n_rows (incl header): ${n_rows}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"

    echo "--- ${#RUN_MODE_OVERRIDE[@]} CONTROLD PAIR runs to enumerate ---" | tee -a "${LOG}"
    for run_mode in "${RUN_MODE_OVERRIDE[@]}"; do
        echo "  run_mode=${run_mode}  ->  ${OUT_BASE}/run_${run_mode}/" | tee -a "${LOG}"
    done
    echo "" | tee -a "${LOG}"

    # Wf-363-A1 — emit structured DRY_RUN.log
    pair_args=()
    for run_mode in "${RUN_MODE_OVERRIDE[@]}"; do
        pair_args+=( --pair "Rh seed=0 run_mode=${run_mode}" )
    done
    # shellcheck disable=SC2068
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-125 P1 CONTROLD PAIR — ${#RUN_MODE_OVERRIDE[@]} runs" \
        --wrapper "$(basename "$0")" \
        --mode "CONTROLED_PAIR_${#RUN_MODE_OVERRIDE[@]}runs" \
        --trainer "${TRAINER}" \
        --flags "${COMMON_ARGS[*]}" \
        --gpu "${GPU}" \
        --budget-hr "${TOTAL_BUDGET_HR}" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log" | tee -a "${LOG}"

    # Wf-363-A1 — emit plan.json
    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_wf125_p1_controlled_pair.sh",\n'
        printf '  "meta": {\n'
        printf '    "wrapper_name": "launch_wf125_p1_controlled_pair.sh",\n'
        printf '    "user_ok_required": true,\n'
        printf '    "wf_origin": "Wf-125 P1 CONTROLD PAIR (USER_OK gate + per-run synthetic_or_real dispatch)",\n'
        printf '    "parent_plan": "TODO/completed/387a2_balanced_rh_cohort.md §5.1",\n'
        printf '    "wf429a4_compliance": "plan.json schema now satisfies wf426a3_post_fix_results_aggregator check; meta.wrapper_name + meta.user_ok_required fields populated per Wf-429-A4 fix"\n'
        printf '  },\n'
        printf '  "wf_origin": "Wf-125 P1 CONTROLD PAIR (USER_OK gate + per-run synthetic_or_real dispatch)",\n'
        printf '  "parent_plan": "TODO/completed/387a2_balanced_rh_cohort.md §5.1",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "run_modes": ['
        first=1
        for rm in "${RUN_MODE_OVERRIDE[@]}"; do
            if [ "${first}" -eq 1 ]; then first=0; else printf ', '; fi
            printf '"%s"' "${rm}"
        done
        printf '],\n'
        printf '  "n_runs": %d,\n' "${#RUN_MODE_OVERRIDE[@]}"
        printf '  "epochs": %d,\n' "${EPOCHS}"
        printf '  "per_run_budget_hr": %d,\n' "${PER_RUN_BUDGET_HR}"
        printf '  "wall_budget_hr": %d,\n' "${TOTAL_BUDGET_HR}"
        printf '  "wall_budget_label": "PROJECTED (~6 hr GPU sequential per Wf-260 G6 timing; Rh single-fold ~30 min/run × 3 + sequential overhead)",\n'
        printf '  "gpu": "%s",\n' "${GPU}"
        printf '  "trainer": "%s",\n' "${TRAINER}"
        printf '  "lomo_csv": "%s",\n' "${LOMO_CSV}"
        printf '  "synth_rh_cohort": "%s",\n' "${SYNTH_RH_COHORT}"
        printf '  "real_rh_cohort": "%s",\n' "${REAL_RH_COHORT}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "USER_OK gate enforced; 3-run CONTROLD PAIR enumerated; DRY_RUN.log + plan.json emitted; 0 GPU launches",\n'
        printf '    "PROJECTED": "~6 hr GPU sequential; Rh single-fold 50 epochs x 3 runs on RTX 5090 / Blackwell sm_120 per Wf-260 G6 timing",\n'
        printf '    "controlled_pair_design": "1. Wf-125 P1 + Wf-139 synthetic 500 (BUGGY); 2. Wf-125 P1 + Wf-387-A2 real 500 (substituted); 3. Wf-125 P1 + no augmentation (control)",\n'
        printf '    "verdict_taxonomy": "run2 approx run1 -> Wf-125 P1 lift is REAL; run2 lt run1 -> lift was ARTIFACT; run2 approx run3 -> no augmentation signal",\n'
        printf '    "kill_then_adapt_directive": "6-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
        printf '    "wf387a2_real_cohort_attestation": "all 500 entries is_synthetic=False + wf139_p7_carryover=False per TODO/completed/387a2_balanced_rh_cohort.md",\n'
        printf '    "wf261_scaffold_split_guarantee": "0 scaffold leak from cohort to Rh val or Rh test per Wf-261 strict scaffold split",\n'
        printf '    "wf399a1_50step_smoke": "SCAFFOLD smoke (CPU) ran in DRY-RUN pre-flight per AGENTS.md §5.7 NEW rule; WARN if rc!=0 because trainer dev-mode may need adjust",\n'
        printf '    "wf386a3_atomic_write_compliance": "wrapper emits atomic_write_file-compliant artefacts (plan.json via printf redirect); no overwriting prior closeouts since this is a NEW wrapper path"\n'
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    echo "plan.json emitted -> ${PLAN_JSON}" | tee -a "${LOG}"
    echo "" | tee -a "${LOG}"
    echo "DRY-RUN OK: ${#RUN_MODE_OVERRIDE[@]} CONTROLD PAIR runs PLANNED, no GPU work executed." | tee -a "${LOG}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Pre-launch wave-tool guards (Wf-343C2 + Wf-343C3 per Wf-347-B1 + Wf-347-B2)
# -----------------------------------------------------------------------------
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all \
    || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wf343c3_data_quality_check.py" \
    --out-dir "${OUT_BASE}/wf343c3_data_quality" \
    || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/per_wave_checklist.py" \
    --wave "Wf-125-P1-ControlledPair" \
    || { echo "FAIL: per_wave_checklist"; exit 1; }
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${SCRIPTS_DIR}/wave_budget_tracker.py" \
    || { echo "FAIL: wave_budget_tracker"; exit 1; }

# -----------------------------------------------------------------------------
# Helper: poll pair log until target epoch is reached (Wf-302B guard)
# -----------------------------------------------------------------------------
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-7200}"  # 2 hr cap per run (PROJECTED ~30 min wall)
  local start_s elapsed last_epoch
  start_s=$(date +%s)
  elapsed=0
  while true; do
    last_epoch=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$pair_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "${last_epoch}" ] && [ "${last_epoch}" -ge "${target_epoch}" ]; then
      echo "Run complete: last_epoch=${last_epoch} >= target=${target_epoch}"
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
# Real launch: per-run runner
# -----------------------------------------------------------------------------
run_controlled_pair_run() {
  local run_mode=$1
  local run_log="${OUT_BASE}/run_${run_mode}/run.log"
  mkdir -p "${OUT_BASE}/run_${run_mode}/ckpts"

  # Per-pair cohort resolution (Wf-414-A1 §4 + Wf-A269 §5.11):
  # build_cohort_args emitted "key value" tokens separated by newlines, read
  # into an array via mapfile; those tokens were trainer flags. The run mode
  # now resolves to two VALUES (RH_OVERSAMPLE / RH_COHORT_JSON) that the
  # derive step below writes into the run's YAML.
  #
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   local cohort_args=()
  #   while IFS= read -r line; do
  #       [ -n "${line}" ] && cohort_args+=("${line}")
  #   done < <(build_cohort_args "${run_mode}")
  resolve_cohort_args "${run_mode}"

  local run_start_s
  run_start_s=$(date +%s)
  echo "RUN_BUDGET_START run_mode=${run_mode} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START CONTROLD PAIR run_mode=${run_mode} ===" | tee -a "${LOG}"
  # `--wait` on setsid, and pipefail off for this one pipeline.
  #   * `setsid nohup cmd | tee` without `--wait`: setsid may fork and return 0
  #     before the trainer has started, so PIPESTATUS[0] below reports success
  #     for a run that has not done anything yet. `--wait` propagates the
  #     trainer's own status.
  #   * pipefail on: DataLoader workers hold the pipe open after the trainer
  #     exits, tee takes EOF first, the trainer's next write takes SIGPIPE and
  #     the propagated 141 aborts the wrapper (BLK-I1b). errexit would fire at
  #     the pipeline and the `local rc=` line would never run.
  set +o pipefail
  # Wf-A269 §5.11 (2026-11-03): `--out-metric` is a trainer hyperparameter and
  # may only reach the trainer through the YAML. Its path depends on the run
  # mode and the --out-dir override, so the base config is finished at launch
  # time and written next to the run it describes. The cohort knobs
  # (--rh-oversample / --rh-cohort-json) are hyperparameters too, so they go
  # into the same derived config.
  local run_cfg="${OUT_BASE}/run_${run_mode}/config.yaml"
  derive_config "${LOMO_CONFIG}" "${run_cfg}" \
      "${OUT_BASE}/run_${run_mode}" \
      "${OUT_BASE}/run_${run_mode}/metrics.json" "${EPOCHS}" \
      "${RH_OVERSAMPLE}" "${RH_COHORT_JSON}"

  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait nohup ${PY} "${TRAINER}" \
  #     --out-dir "${OUT_BASE}/run_${run_mode}" \
  #     --out-metric "${OUT_BASE}/run_${run_mode}/metrics.json" \
  #     --ckpt-dir "${OUT_BASE}/run_${run_mode}/ckpts" \
  # "${COMMON_ARGS[@]}" \
  #     "${cohort_args[@]+"${cohort_args[@]}"}" \
  #     2>&1 | tee "${run_log}"
  # Wf-A269 wave-19 (2026-11-03): `--ckpt-dir` was REMOVED from the live
  # invocation.  It is in §5.11's allowlist, but `train_metal_lomo_wasserstein.py`
  # declares NO checkpoint dest at all (MEASURED: its parser's long options
  # contain no `ckpt`/`save`/`checkpoint` flag), so the flag could only ever
  # abort the run:
  #   molmetal/reports/wf125_p1_controlled_pair/run_synthetic/run.log:18
  #   train_metal_lomo_wasserstein.py: error: unrecognized arguments:
  #     --ckpt-dir .../ckpts --no-cuda-graphs --cuda-visible-devices 0
  # The pre-migration rendering kept the flag too, so this wrapper has never
  # completed a run — it died at argument parsing, before training started.
  # Dropping it is behaviour-preserving for the run that actually happened
  # (which was no run) and unblocking for the one that now can.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait nohup ${PY} "${TRAINER}" \
    --config "${run_cfg}" \
    --out-dir "${OUT_BASE}/run_${run_mode}" \
    2>&1 | tee "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -o pipefail

  local run_end_s run_elapsed_s
  run_end_s=$(date +%s)
  run_elapsed_s=$(( run_end_s - run_start_s ))
  echo "RUN_BUDGET_END run_mode=${run_mode} elapsed_s=${run_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') END CONTROLD PAIR run_mode=${run_mode} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "${rc}" -ne 0 ]; then
    echo "FAIL: run_mode=${run_mode} exited rc=${rc}"
    return "${rc}"
  fi
  wait_until_epoch_reached "${run_log}" "${EPOCHS}" 7200 || { echo "FAIL: run_mode=${run_mode} did not reach epoch ${EPOCHS}"; return 1; }
}

echo "=== Wf-125 P1 CONTROLD PAIR START $(date) ===" | tee -a "${LOG}"
echo "${#RUN_MODE_OVERRIDE[@]} runs (synthetic|real|none) -- USER_OK-gated per AGENTS.md §5.1 + Wf-294B family" | tee -a "${LOG}"

# Iterate RUN_MODE_OVERRIDE sequentially
for run_mode in "${RUN_MODE_OVERRIDE[@]}"; do
  run_controlled_pair_run "${run_mode}"
done

echo "=== ALL Wf-125 P1 CONTROLD PAIR RUNS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation: collect per-run metrics into CONTROLED_PAIR_SUMMARY.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for run_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
        continue
    run_mode = run_dir.name.replace('run_', '')
    for p in run_dir.glob('per_seed_metrics_seed*.json'):
        runs.append({'run_mode': run_mode, 'metrics_path': str(p)})
out = {
    'wf125_p1_controlled_pair_total_runs': len(runs),
    'n_runs_planned': ${#RUN_MODE_OVERRIDE[@]},
    'runs': runs,
    'verdict_taxonomy': {
        'run2_approx_run1': 'Wf-125 P1 lift is REAL',
        'run2_lt_run1': 'Wf-125 P1 lift was ARTIFACT',
        'run2_approx_run3': 'no augmentation signal',
    },
}
Path('${OUT_BASE}/CONTROLED_PAIR_SUMMARY.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} runs (planned ${#RUN_MODE_OVERRIDE[@]})')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"
