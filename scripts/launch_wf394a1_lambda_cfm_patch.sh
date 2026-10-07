#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/launch_wf394a1_lambda_cfm_patch.sh
#
# Wf-394-A1 Lambda × CFM coupling patch wrapper (USER_OK-gated, GPU launch
# deferred). Direct invocation of train_cfm_yuelbond_wasserstein.py —
# NOT delegating to Wf-299. 4 ablation configs × 1 seed × N epochs on
# cuda:0 (PRO 6000 Blackwell) for the TODO-21 Lambda × CFM coupling
# retrain surface (per Wf-329-W6 + Wf-333-G1 + Wf-394-A1 + Wf-406-A2 +
# Wf-410-A2 + Wf-414-A2 + Wf-418-A2 patches).
#
# Sister wrapper to scripts/launch_wf394_lambda_cfm.sh (the latter
# DELEGATES to Wf-299; this one calls the trainer directly per the
# Wf-428-A2 retrain readiness audit forward-looking fix at §3.4:
# "Wf-394 wrapper direct invocation — close the Wf-299-delegate gap").
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf394a1_lambda_cfm_patch.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf394a1_lambda_cfm_patch.sh --epochs 3
#   USER_OK=1 bash scripts/launch_wf394a1_lambda_cfm_patch.sh --epochs 50 \
#       --lambda-cfm-coupling-weight 0.2 --lambda-cfm-coupling-samples 100
#
# Exit codes:
#   0   PASS (--dry-run emits DRY_RUN.log + plan.json; LIVE runs 4 configs)
#   1   FAIL (filesystem or trainer call missing)
#   64  USER_OK gate refused (USER_OK env not set to "1")
#
# Required: USER_OK=1 in env (explicit per AGENTS.md §5.1 + Wf-294B +
# Wf-355-B1 + Wf-366-A1 + Wf-378-A2 + Wf-413-A3 pattern). The wrapper
# REFUSES to run unless USER_OK=1 is set, including for --dry-run.
# --dry-run must NOT bypass the USER_OK gate per Wf-378-A2 G6 lesson.
#
# Wf-415-A1 push — true EXTRA_FLAGS accumulator captures any flag not in
# the wrapper's known surface (--dry-run, --epochs, --seed, -h, --help)
# and passes it verbatim to the underlying trainer.
#
# Wf-429-A4 plan.json meta-block schema (per wf426a3_post_fix_results_aggregator
# line 14 contract) — `meta.wrapper_name` + `meta.user_ok_required` fields
# are populated so the wrapper passes Wf-426-A3 readiness audit step 4.
# -----------------------------------------------------------------------------
set -euo pipefail

# Locate repo root + canonical CFM trainer (direct invocation, no delegation
# to Wf-299 launcher; per Wf-428-A2 retrain readiness §3.4 forward-look fix).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
CFM_TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"

# 2026-10-07 (Wf-R29 Tier-3 migration) — every hyperparameter now comes from
# configs/cfm.yaml, matching launch_cfm_tmqm_all6_retrain.sh (the canonical CFM
# template).  Before this edit the wrapper passed NO `--config`, so every value
# came from the argparse DEFAULT — and the defaults are not equivalent to the
# yaml.  MEASURED delta (probe of `parse_args()` under the two inputs):
#
#   key                 argparse default -> cfm.yaml
#   use_fm_objective    False            -> True     (velocity target was ZERO)
#   use_egnn_velocity   False            -> True     (MLP placeholder -> EGNN)
#   fm_noise_sigma      1.0              -> 2.485    (measured cohort std)
#   lr_scheduler        "none"           -> "cosine"
#   hidden_dim          64               -> 128
#   yuelbond_vocab      "pt"             -> "all6"
#   n_steps / method    100 / "euler"    -> 50 / "midpoint"
#   gen_eval_n          None (eval OFF)  -> 256
#   tmqm_dir            None (synthetic) -> .data/molmetal/tmQM
#
# The first two are the load-bearing ones: with `use_fm_objective=False` the
# flow was trained against a zero target, so every number this wrapper had
# produced measured an untrained model.  configs/cfm.yaml:39-45 records the
# same conclusion for the Round-2 cohort.
#
# Only RUN-CONTROL args stay on the command line: --epochs/--seed (per-run
# budget), --pocket-id/--device/--out-dir (placement), plus the ablation and
# Lambda-coupling tuples, which are the experiment's independent variable.
#
# NOTE on the ablation tuples: `--no-wasserstein` sets `wasserstein_weight=0.0`
# while the argparse default is 0.1, so the loader's "CLI beats YAML" rule
# keeps the 0.0 (verified by probe, not assumed).  `--use-yuelbond` is a
# no-op against the default True but cfm.yaml does not declare use_yuelbond
# either, so it stays consistent.
CFG="${CFG:-${PROJECT_ROOT}/configs/cfm.yaml}"

# Source the DRY_RUN.log helper (Wf-363-A1 schema) for consistent
# per-wrapper DRY_RUN.log + plan.json emission.
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
if [ ! -f "${DRY_RUN_LOG_HELPER}" ]; then
    printf '%s[FATAL]%s DRY_RUN log helper missing: %s\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" \
        "${DRY_RUN_LOG_HELPER}" >&2
    exit 1
fi
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Die helper
die() {
    printf '%s[FATAL]%s %s\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" \
        "$*" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# CLI arg parsing — USER_OK gate + DRY_RUN + EXTRA_FLAGS accumulator
# -----------------------------------------------------------------------------
DRY_RUN=0
EPOCHS=3
SEED=0
POCKET_ID="pocket_a"
EXTRA_FLAGS=""

# Default Lambda × CFM coupling flags (Wf-406-A2 canonical store_true +
# float + int shape on trainer's parse_args). These are the trainer-side
# flag names per molmetal/scripts/train_cfm_yuelbond_wasserstein.py
# lines 1978-2007 (`--lambda-cfm-coupling`, `--lambda-cfm-coupling-weight`,
# `--lambda-cfm-coupling-samples`). The `--lambda-cfm-coupling` flag is
# the BooleanOptionalAction gate; both --lambda-cfm-coupling-weight > 0
# AND --lambda-cfm-mean-weight > 0 must hold for the coupling to activate
# per trainer line 2547-2575 (loss composition conditional).
LAMBDA_CFM_DEFAULT_FLAGS="--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200"

# Wf-415-A1 — true EXTRA_FLAGS accumulator pattern (no hard-coded flag
# allow-list; any flag not in the wrapper's known surface is transparently
# passthroughed to the trainer). Per Wf-415-A1 §0: forbidden to use
# simplified solutions — this MUST remain a true passthrough accumulator.
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --epochs)
            EPOCHS="${2:-3}"
            shift 2
            ;;
        --seed)
            SEED="${2:-0}"
            shift 2
            ;;
        --pocket-id)
            POCKET_ID="${2:-pocket_a}"
            shift 2
            ;;
        -h|--help)
            printf 'Wf-394-A1 Lambda × CFM coupling patch wrapper (USER_OK-gated)\n'
            printf 'Usage: USER_OK=1 %s [--dry-run|--epochs N] [--seed N] [--pocket-id ID] [extra-trainer-flags...]\n' "$0"
            printf '  Default Lambda × CFM coupling flags: %s\n' "${LAMBDA_CFM_DEFAULT_FLAGS}"
            printf '  Direct invocation of train_cfm_yuelbond_wasserstein.py (NOT delegating to Wf-299).\n'
            printf '  4 ablation configs × 1 seed × --epochs N each (full_stack + yuelbond_only + w2_only + catherine_only).\n'
            printf '  Per-ablation-config budget 1 hr × 4 configs = 4 hr total PROJECTED on cuda:0 Blackwell.\n'
            printf '  EXTRA_FLAGS: any flag not in the wrapper surface is transparently\n'
            printf '  passthroughed to the trainer (Wf-415-A1 contract).\n'
            exit 0
            ;;
        --)
            # explicit "--" terminator: pass everything after as opaque args.
            shift
            while [ $# -gt 0 ]; do
                EXTRA_FLAGS="${EXTRA_FLAGS} $1"
                shift
            done
            ;;
        *)
            # Catch-all: transparent passthrough. Preserves `--flag value`
            # and `--flag=value` syntax verbatim.
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            if [ $# -gt 1 ] && [ "${2#--}" = "$2" ]; then
                EXTRA_FLAGS="${EXTRA_FLAGS} $2"
                shift 2
            else
                shift 1
            fi
            ;;
    esac
done

# -----------------------------------------------------------------------------
# USER_OK gate — explicit, enforced BEFORE --dry-run block per Wf-378-A2
# G6 lesson (--dry-run must NOT bypass USER_OK gate).
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" >&2
    printf '       USER_OK=1 bash %s --dry-run           # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --epochs 50          # full 4-config retrain (~4 hr cuda:0)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# Pre-flight check: CFM trainer must exist on disk before any launch.
# Per Wf-428-A2 retrain readiness §1.5: trainer file existence is a
# BLOCKER for SCAFFOLD_ONLY → READY transition.
# -----------------------------------------------------------------------------
if [ ! -f "${CFM_TRAINER}" ]; then
    die "CFM trainer missing on disk: ${CFM_TRAINER}"
fi

# Wf-R29 Tier-3 migration: the yaml is now the sole hyperparameter source, so a
# missing/typo'd CFG would silently fall back to argparse defaults — the exact
# silent-lie class this migration exists to remove.  Fail loudly instead.
if [ ! -f "${CFG}" ]; then
    die "CFM config missing on disk: ${CFG}"
fi

# Default Lambda × CFM coupling flags when caller did not override via
# EXTRA_FLAGS. We honour caller-supplied EXTRA_FLAGS first, then prepend
# our defaults, so explicit overrides always win.
if [ "${EXTRA_FLAGS#*--lambda-cfm-coupling}" = "${EXTRA_FLAGS}" ]; then
    EFFECTIVE_FLAGS="${LAMBDA_CFM_DEFAULT_FLAGS}${EXTRA_FLAGS:+ }${EXTRA_FLAGS}"
else
    EFFECTIVE_FLAGS="${EXTRA_FLAGS}"
fi

# OUT_DIR default — Wf-363-A1 DRY_RUN.log + plan.json emission path.
OUT_DIR_DEFAULT="${PROJECT_ROOT}/molmetal/reports/wf394a1_lambda_cfm_patch"
OUT_DIR="${OUT_DIR_DEFAULT}"

# -----------------------------------------------------------------------------
# Per-ablation-config enumeration (handy of pairs for --dry-run DRY_RUN.log
# emission). 4 ablation configs per Wf-299 P1.4 §ablation_configs:
#   full_stack       — no negation flags (uses --use-yuelbond)
#   yuelbond_only     — --no-wasserstein --no-closure-bounded (YuelBond only)
#   w2_only          — --no-yuelbond --no-closure-bounded (W2 only)
#   catherine_only   — --no-yuelbond --no-wasserstein (Catherine-only-like)
# Per Wf-299 launcher wave_A function lines 588-618.
# -----------------------------------------------------------------------------
N_PAIRS=4
PER_CONFIG_BUDGET_HR=1.0  # kill_then_adapt < 2 hr/wave cap; 1 hr × 4 = 4 hr
TOTAL_BUDGET_HR=$(awk -v n="${N_PAIRS}" -v h="${PER_CONFIG_BUDGET_HR}" 'BEGIN { printf "%.2f", n*h }')

# Per-ablation-config flag tuples
declare -A ABLATION_FLAGS
ABLATION_FLAGS[full_stack]="--use-yuelbond"
ABLATION_FLAGS[yuelbond_only]="--use-yuelbond --no-wasserstein --no-closure-bounded"
ABLATION_FLAGS[w2_only]="--no-yuelbond --no-closure-bounded"
ABLATION_FLAGS[catherine_only]="--no-yuelbond --no-wasserstein"

PAIRS=()
for key in full_stack yuelbond_only w2_only catherine_only; do
    PAIRS+=("${key}_seed${SEED}")
done

# -----------------------------------------------------------------------------
# Emit DRY_RUN.log per Wf-363-A1 schema (out_dir absolute vs relative handling).
# POSIX-atomic via temp-file + mv -f (same semantics as atomic_write_file
# os.replace on Linux per AGENTS.md §5.6).
# -----------------------------------------------------------------------------
case "${OUT_DIR}" in
    /*) LOG_DIR="${OUT_DIR}" ;;
    *)  LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}" ;;
esac
mkdir -p "${LOG_DIR}"

FLAGS_DISPLAY="(${LAMBDA_CFM_DEFAULT_FLAGS} ${EFFECTIVE_FLAGS#${LAMBDA_CFM_DEFAULT_FLAGS}})"

# Build the pair_args array (each --pair as a discrete argv element so the
# helper sees --pair as a flag, not a glob). Per Wf-409-A3 wrapper fix.
PAIR_ARGS=()
for pair in "${PAIRS[@]}"; do
    PAIR_ARGS+=( --pair "${pair}" )
done

emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
    --title "Wf-394-A1 Lambda × CFM coupling patch retrain (4 ablation configs × 1 seed × ${EPOCHS} epochs)" \
    --wrapper "scripts/launch_wf394a1_lambda_cfm_patch.sh" \
    --mode "R1_dry_run" \
    --trainer "${CFM_TRAINER}" \
    --subset "TODO-21 Lambda × CFM coupling (Wf-394-A1 wire-in)" \
    --flags "${FLAGS_DISPLAY}" \
    --gpu "cuda:0 (PRO 6000 Blackwell)" \
    --budget-hr "${TOTAL_BUDGET_HR}" \
    --user-ok "REQUIRED" \
    "${PAIR_ARGS[@]}"

# Append the Wf-415-A1 transparent-passthrough line per Wf-299 wrapper §0.
# The trainer's argparse will reject any unknown flag at its own parse_args
# boundary, with a clear error pointing to the trainer's own --help output.
cat >> "${LOG_DIR}/DRY_RUN.log" <<EOF
  extra_flags: ${EFFECTIVE_FLAGS#${LAMBDA_CFM_DEFAULT_FLAGS} }
  epochs_per_config: ${EPOCHS}
  seed: ${SEED}
  pocket_id: ${POCKET_ID}
  per_config_budget_hr: ${PER_CONFIG_BUDGET_HR}
  ablation_configs: full_stack | yuelbond_only | w2_only | catherine_only
  direct_invocation: true  # NOT delegating to Wf-299
EOF

# -----------------------------------------------------------------------------
# Emit plan.json per Wf-431-A4 fix + Wf-429-A4 meta-block schema.
# POSIX-atomic via temp-file + mv -f per AGENTS.md §5.6.
# -----------------------------------------------------------------------------
PLAN_JSON="${LOG_DIR}/plan.json"
cat > "${PLAN_JSON}.tmp" <<EOF
{
  "wrapper": "scripts/launch_wf394a1_lambda_cfm_patch.sh",
  "meta": {
    "wrapper_name": "scripts/launch_wf394a1_lambda_cfm_patch.sh",
    "user_ok_required": true,
    "wf_origin": "Wf-394-A1 Lambda × CFM coupling patch wrapper (direct invocation, USER_OK-gated)",
    "parent_plan": "TODO/03_baselines/COMPARISON_BASELINES.md §6 + Wf-394-A1 closure path (HONEST_NEGATIVE_CLOSURE_PLAN.md was Oct 2 docs-only planning doc; removed 2026-10-05 as superseded)",
    "wf_origin_patches": ["Wf-329-W6", "Wf-333-G1", "Wf-394-A1", "Wf-406-A2", "Wf-410-A2", "Wf-414-A2", "Wf-418-A2", "Wf-428-A2 §3.4 forward-look"],
    "wf429a4_compliance": "plan.json schema satisfies wf426a3_post_fix_results_aggregator check; meta.wrapper_name + meta.user_ok_required fields populated per Wf-429-A4 fix"
  },
  "user_ok": true,
  "dry_run": ${DRY_RUN},
  "config": "${CFG}",
  "epochs_per_config": ${EPOCHS},
  "seed": ${SEED},
  "pocket_id": "${POCKET_ID}",
  "n_pairs": ${N_PAIRS},
  "pair_budget_hr": ${PER_CONFIG_BUDGET_HR},
  "total_budget_hr": ${TOTAL_BUDGET_HR},
  "gpu_assignment": {
    "default": "cuda:0 (PRO 6000 Blackwell)",
    "note": "wrapper default per FIX_SCHEMA spec; user may override via EXTRA_FLAGS"
  },
  "default_flags": "${LAMBDA_CFM_DEFAULT_FLAGS}",
  "effective_flags": "${EFFECTIVE_FLAGS}",
  "ablation_configs": {
    "full_stack": "${ABLATION_FLAGS[full_stack]}",
    "yuelbond_only": "${ABLATION_FLAGS[yuelbond_only]}",
    "w2_only": "${ABLATION_FLAGS[w2_only]}",
    "catherine_only": "${ABLATION_FLAGS[catherine_only]}"
  },
  "transparent_passthrough": true,
  "honest_framing": {
    "wrapper_authored_only": true,
    "gpu_launch_blocked_on_user_ok": true,
    "actual_gpu_run_requires_user_ok_1": true,
    "per_config_budget_hr": ${PER_CONFIG_BUDGET_HR},
    "total_budget_hr": "${TOTAL_BUDGET_HR}",
    "PROJECTED_vs_MEASURED": "PROJECTED; MEASURED after actual launch",
    "wf394_lambda_cfm_upstream_blocker": "decode_ratio > 0 per Wf-291D + Wf-299"
  },
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"

# -----------------------------------------------------------------------------
# Branch: --dry-run vs actual launch.
# In --dry-run mode: emit logs + exit 0 (no trainer invocation, no GPU work).
# In LIVE mode: invoke the trainer directly (NOT delegating to Wf-299)
# for each of 4 ablation configs sequentially.
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" = "1" ]; then
    printf '[Wf-394-A1] R1 dry-run mode: wrapper scaffold exists; '
    printf 'CFM trainer exists on disk at %s; ' "${CFM_TRAINER}"
    printf 'USER_OK gate honored; pre-flight path validated; '
    printf 'would launch 4 ablation configs × 1 seed × %d epochs on cuda:0; ' "${EPOCHS}"
    printf 'per-config budget %s hr; total budget %s hr; ' \
        "${PER_CONFIG_BUDGET_HR}" "${TOTAL_BUDGET_HR}"
    printf 'DRY_RUN.log: %s/DRY_RUN.log; plan.json: %s/plan.json\n' \
        "${LOG_DIR}" "${LOG_DIR}"
    exit 0
fi

# LIVE mode: invoke CFM trainer directly (NOT delegating to Wf-299) for
# each of 4 ablation configs sequentially on cuda:0. Per Wf-299 wave_A
# pattern (lines 588-618) but with our Lambda × CFM coupling flags.
printf '[Wf-394-A1] launching 4 ablation configs × 1 seed × %d epochs on cuda:0\n' "${EPOCHS}"

for key in full_stack yuelbond_only w2_only catherine_only; do
    ABLATION_FLAG_TUPLE="${ABLATION_FLAGS[${key}]}"
    CONFIG_OUT_DIR="${LOG_DIR}/${key}"
    mkdir -p "${CONFIG_OUT_DIR}"
    printf '[Wf-394-A1] === Config: %s (flags: %s) ===\n' \
        "${key}" "${ABLATION_FLAG_TUPLE}"
    CUDA_VISIBLE_DEVICES=0 \
        "${PY}" "${CFM_TRAINER}" \
            --config "${CFG}" \
            --epochs "${EPOCHS}" \
            --seed "${SEED}" \
            --pocket-id "${POCKET_ID}" \
            --device cuda:0 \
            --out-dir "${CONFIG_OUT_DIR}" \
            ${EFFECTIVE_FLAGS} \
            ${ABLATION_FLAG_TUPLE} \
            > "${CONFIG_OUT_DIR}/train.log" 2>&1 || {
                printf '[Wf-394-A1][WARN] config %s exited non-zero; continuing\n' "${key}" >&2
            }
    printf '[Wf-394-A1] === Config: %s complete ===\n' "${key}"
done

printf '[Wf-394-A1] all 4 ablation configs complete\n'
