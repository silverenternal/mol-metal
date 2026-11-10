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
#   USER_OK=1 bash scripts/launch_wf394a1_lambda_cfm_patch.sh
#
# §5.11 YAML-ONLY LAUNCH (2026-11-03, Wf-A269 cohort c6_cfm) — the trainer
# receives `--config` and `--out-dir` and NOTHING else.  Every
# hyperparameter this wrapper used to pass on the command line is a key in
# ONE YAML per ablation arm:
#
#   configs/wfa269_c6_cfm_full_stack.yaml
#   configs/wfa269_c6_cfm_yuelbond_only.yaml
#   configs/wfa269_c6_cfm_w2_only.yaml
#   configs/wfa269_c6_cfm_catherine_only.yaml
#
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may pass only --config /
# --out-dir / --ckpt-dir / --help to a trainer.  The epoch count, the seed,
# the pocket id, the device, the Lambda-coupling tuple and the per-arm
# ablation tuple are all hyperparameters, so they moved into those four
# files.  To change one, EDIT THE YAML — the wrapper REFUSES a command-line
# value that disagrees with the YAML (loudly, never silently).
#
# Exit codes:
#   0   PASS (--dry-run emits DRY_RUN.log + plan.json; LIVE runs 4 configs)
#   1   FAIL (filesystem or trainer call missing, or a CLI value that
#        disagrees with the YAML)
#   64  USER_OK gate refused (USER_OK env not set to "1")
#
# Required: USER_OK=1 in env (explicit per AGENTS.md §5.1 + Wf-294B +
# Wf-355-B1 + Wf-366-A1 + Wf-378-A2 + Wf-413-A3 pattern). The wrapper
# REFUSES to run unless USER_OK=1 is set, including for --dry-run.
# --dry-run must NOT bypass the USER_OK gate per Wf-378-A2 G6 lesson.
#
# Wf-A269 cohort c6_cfm (2026-11-03) — the Wf-415-A1 EXTRA_FLAGS
# passthrough accumulator is RETIRED in this wrapper: a trainer
# hyperparameter may not reach the trainer through the command line
# (AGENTS.md §5.11), and the ablation tuples are hyperparameters, not
# run-control.  An unknown flag is now a FATAL error naming the YAML to
# edit, instead of a silent command-line hyperparameter.
#
# Wf-429-A4 plan.json meta-block schema (per wf426a3_post_fix_results_aggregator
# line 14 contract) — `meta.wrapper_name` + `meta.user_ok_required` fields
# are populated so the wrapper passes Wf-426-A3 readiness audit step 4.
# -----------------------------------------------------------------------------
set -euo pipefail


# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# Die helper — DEFINED BEFORE ANY CALL (Wf-A269 repair 2, 2026-11-02).
# Bash executes a script line-by-line, so a function definition that appears
# LATER in the file is not yet in scope at an earlier call site. The
# pre-flight gate below calls die() at top level, so this definition MUST
# precede it — otherwise the call dies with "die: command not found" (rc=127)
# and the intended FATAL message + rc=1 are both lost.
die() {
    printf '%s[FATAL]%s %s\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" \
        "$*" >&2
    exit 1
}

# --- Pre-flight gate (Wf-415-A3 + Wf-427-B1) ------------------------------
# Wf-A269 wave-16 (2026-11-03) — THREE repairs to this block:
#
#  (1) DISTINCT report path.  This wrapper used to write the SAME
#      molmetal/reports/wf_preflight/preflight.json as
#      launch_wf380_scaffold_loss.sh and
#      launch_wf409a3_metal_coord_retrain.sh, so two concurrent launches
#      destroyed each other's evidence (last writer wins on a shared file).
#  (2) `--json PATH` — the pre-flight writes a PURE-JSON report; the console
#      transcript goes to a sibling .log.  The old `> "${PREFLIGHT_OUT}"
#      2>&1` merged the gate LOGS (stderr) and the PAYLOAD (stdout) into ONE
#      file: its first 269 lines were "[pre_flight_check] INFO ..." lines, so
#      json.load raised "Expecting value: line 1 column 2" — the "JSON
#      report" was not JSON, and no machine could read the gate verdict.
#  (3) the verdict comes from the repo's own consumer
#      (molmetal/scripts/wf_launch_gate_verdict.py), which checks the FAIL
#      count AND the TIMEOUT count AND the report's freshness.  The old
#      `grep -q '"fail": 0'` was TIMEOUT-blind: a report with one TIMEOUT
#      and zero FAILs passed it, although Wf-A135 says a TIMEOUT means the
#      measurement was NOT obtained.
PREFLIGHT_OUT="${PROJECT_ROOT:-/home/hugo/codes/mol-metal}/molmetal/reports/wf_preflight/preflight_wf394a1_lambda_cfm_patch.json"
PREFLIGHT_LOG="${PREFLIGHT_OUT%.json}.log"
mkdir -p "$(dirname "${PREFLIGHT_OUT}")"
echo "[$(date -Iseconds)] ${BASH_SOURCE[0]} pre-flight check begin"
# Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A crashed
# or killed pre-flight must not leave the last run's PASS report on disk
# for the consumer to read — "stale is treated as missing".
rm -f "${PREFLIGHT_OUT}"
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    "${PROJECT_ROOT:-/home/hugo/codes/mol-metal}/molmetal/scripts/pre_flight_check.py" --pipeline launch \
    --json "${PREFLIGHT_OUT}" \
    >"${PREFLIGHT_LOG}" 2>&1 \
    || die "pre_flight_check FAIL — see ${PREFLIGHT_LOG}"
pf_verdict="$(PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    "${PROJECT_ROOT:-/home/hugo/codes/mol-metal}/molmetal/scripts/wf_launch_gate_verdict.py" \
    preflight --report "${PREFLIGHT_OUT}" 2>&1)" \
    || die "pre_flight_check verdict is not PASS — ${pf_verdict} (see ${PREFLIGHT_OUT})"
echo "[$(date -Iseconds)] pre-flight PASS — ${pf_verdict}"

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
#
# 2026-11-03 (AGENTS.md §5.11, Wf-A269 cohort c6_cfm) — the paragraph above
# describes the state BEFORE the §5.11 migration; it is PRESERVED VERBATIM
# per AGENTS.md §5.5 rule #6.  Today the trainer receives --config and
# --out-dir ONLY: --epochs / --seed / --pocket-id / --device, the
# Lambda-coupling tuple AND the per-arm ablation tuple are all keys in the
# four YAMLs named below (the old command line is kept as a comment at the
# launch site).  One YAML per arm, on disk, NOT generated at launch time.
#
# The four arm configs are copies of configs/cfm.yaml with only the
# command-line values folded in, so every OTHER hyperparameter of the old run
# (hidden_dim, batch_size, lr, n_train, fm_noise_sigma, lr_scheduler,
# tmqm_dir, ...) is byte-identical to the pre-migration run.
CFG_DIR="${CFG_DIR:-${PROJECT_ROOT}/configs}"
declare -A ABLATION_CONFIG
ABLATION_CONFIG[full_stack]="${CFG_DIR}/wfa269_c6_cfm_full_stack.yaml"
ABLATION_CONFIG[yuelbond_only]="${CFG_DIR}/wfa269_c6_cfm_yuelbond_only.yaml"
ABLATION_CONFIG[w2_only]="${CFG_DIR}/wfa269_c6_cfm_w2_only.yaml"
ABLATION_CONFIG[catherine_only]="${CFG_DIR}/wfa269_c6_cfm_catherine_only.yaml"

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

# -----------------------------------------------------------------------------
# CLI arg parsing — USER_OK gate + DRY_RUN + run-control selectors
# -----------------------------------------------------------------------------
DRY_RUN=0
EPOCHS=3
SEED=0
POCKET_ID="pocket_a"

# The Lambda × CFM coupling tuple the wrapper used to pass on the command
# line (`--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1
# --lambda-cfm-coupling-samples 200`).  These are the trainer-side flag
# names per molmetal/scripts/train_cfm_yuelbond_wasserstein.py lines
# 1978-2007 (`--lambda-cfm-coupling`, `--lambda-cfm-coupling-weight`,
# `--lambda-cfm-coupling-samples`).  The `--lambda-cfm-coupling` flag is
# the BooleanOptionalAction gate; both --lambda-cfm-coupling-weight > 0
# AND --lambda-cfm-mean-weight > 0 must hold for the coupling to activate
# per trainer line 2547-2575 (loss composition conditional).
#
# 2026-11-03 (§5.11): the tuple is CARRIED BY THE YAML now
# (train.lambda_cfm_coupling / train.lambda_cfm_coupling_weight /
# train.lambda_cfm_coupling_samples in each of the four arm configs).  The
# string is kept here because DRY_RUN.log and plan.json report it.
LAMBDA_CFM_DEFAULT_FLAGS="--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200"

# Wf-A269 cohort c6_cfm (2026-11-03) — the Wf-415-A1 true EXTRA_FLAGS
# accumulator is RETIRED here.  It was a "transparent passthrough" of ANY
# flag to the trainer; AGENTS.md §5.11 says a trainer hyperparameter may
# only come from the YAML.  A flag this wrapper does not know is therefore a
# FATAL error that names the file to edit, never a silent hyperparameter.
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
            printf 'Usage: USER_OK=1 %s [--dry-run] [--epochs N] [--seed N] [--pocket-id ID]\n' "$0"
            printf '  The trainer is launched with --config + --out-dir ONLY (AGENTS.md §5.11).\n'
            printf '  Hyperparameters live in one YAML per ablation arm:\n'
            printf '    configs/wfa269_c6_cfm_full_stack.yaml\n'
            printf '    configs/wfa269_c6_cfm_yuelbond_only.yaml\n'
            printf '    configs/wfa269_c6_cfm_w2_only.yaml\n'
            printf '    configs/wfa269_c6_cfm_catherine_only.yaml\n'
            printf '  --epochs / --seed / --pocket-id are CHECKED against those files and\n'
            printf '  REFUSED when they disagree; edit the YAML to change them.\n'
            printf '  Default Lambda × CFM coupling flags (now YAML keys): %s\n' "${LAMBDA_CFM_DEFAULT_FLAGS}"
            printf '  Direct invocation of train_cfm_yuelbond_wasserstein.py (NOT delegating to Wf-299).\n'
            printf '  4 ablation configs × 1 seed × N epochs (full_stack + yuelbond_only + w2_only + catherine_only).\n'
            printf '  Per-ablation-config budget 1 hr × 4 configs = 4 hr total PROJECTED on cuda:0 Blackwell.\n'
            exit 0
            ;;
        --)
            # explicit "--" terminator: opaque args.  Under §5.11 they cannot
            # reach the trainer, so they are refused rather than dropped.
            die "-- <args> is refused: AGENTS.md §5.11 lets the trainer receive --config/--out-dir only; put the hyperparameter in ${ABLATION_CONFIG[full_stack]} (or the arm's own YAML)."
            ;;
        *)
            # Unknown flag: under §5.11 it may not become a trainer
            # hyperparameter.  Name the file to edit instead of passing it on.
            die "unknown flag '$1' is refused: AGENTS.md §5.11 lets the trainer receive --config/--out-dir only; put the hyperparameter in the arm's YAML (${ABLATION_CONFIG[full_stack]} and its three siblings)."
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
    printf '       USER_OK=1 bash %s                     # full 4-config retrain (~4 hr cuda:0)\n' "$0" >&2
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
# missing/typo'd config would silently fall back to argparse defaults — the exact
# silent-lie class this migration exists to remove.  Fail loudly instead.
# One config per ablation arm; all four must exist before ANY arm launches.
for _key in full_stack yuelbond_only w2_only catherine_only; do
    if [ ! -f "${ABLATION_CONFIG[${_key}]}" ]; then
        die "CFM config missing on disk for arm '${_key}': ${ABLATION_CONFIG[${_key}]}"
    fi
done

# -----------------------------------------------------------------------------
# §5.11 run-control reconciliation (2026-11-03, Wf-A269 cohort c6_cfm)
# -----------------------------------------------------------------------------
# --epochs / --seed / --pocket-id are HYPERPARAMETERS, so §5.11 forbids them
# on the trainer's command line: they live in the YAML now.  The wrapper
# still ACCEPTS them (its CLI surface is unchanged) but it can no longer
# forward them, so a value that DISAGREES with the YAML is REFUSED loudly
# rather than silently ignored.  The values are read straight out of the
# four arm configs — the same files the trainer will read — so there is no
# second copy of the hyperparameters to drift.
RUN_CONTROL_TSV="$("${PY}" - \
    "full_stack=${ABLATION_CONFIG[full_stack]}" \
    "yuelbond_only=${ABLATION_CONFIG[yuelbond_only]}" \
    "w2_only=${ABLATION_CONFIG[w2_only]}" \
    "catherine_only=${ABLATION_CONFIG[catherine_only]}" <<'PYEOF'
import sys

import yaml


def leaf(cfg, section, key):
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        return ""
    value = sec.get(key)
    return "" if value is None else str(value)


for spec in sys.argv[1:]:
    arm, path = spec.split("=", 1)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    print(
        "\t".join(
            [
                arm,
                leaf(cfg, "train", "epochs"),
                leaf(cfg, "protocol", "seed"),
                leaf(cfg, "protocol", "pocket_id"),
            ]
        )
    )
PYEOF
)" || die "could not read the run-control values out of the arm configs"

while IFS=$'\t' read -r _arm _y_epochs _y_seed _y_pocket; do
    [ -n "${_arm}" ] || continue
    if [ "${_y_epochs}" != "${EPOCHS}" ]; then
        die "--epochs ${EPOCHS} disagrees with the YAML (train.epochs=${_y_epochs:-<absent>}) in ${ABLATION_CONFIG[${_arm}]} — per AGENTS.md §5.11 the epoch count lives in the YAML: edit that file and re-run WITHOUT --epochs."
    fi
    if [ "${_y_seed}" != "${SEED}" ]; then
        die "--seed ${SEED} disagrees with the YAML (protocol.seed=${_y_seed:-<absent>}) in ${ABLATION_CONFIG[${_arm}]} — per AGENTS.md §5.11 the seed lives in the YAML: edit that file and re-run WITHOUT --seed."
    fi
    if [ "${_y_pocket}" != "${POCKET_ID}" ]; then
        die "--pocket-id ${POCKET_ID} disagrees with the YAML (protocol.pocket_id=${_y_pocket:-<absent>}) in ${ABLATION_CONFIG[${_arm}]} — per AGENTS.md §5.11 the pocket id lives in the YAML: edit that file and re-run WITHOUT --pocket-id."
    fi
done <<< "${RUN_CONTROL_TSV}"

# The Lambda × CFM coupling tuple is YAML-carried now; the variable is kept
# for the DRY_RUN.log / plan.json report only (it is NOT passed to the
# trainer any more).
EFFECTIVE_FLAGS="${LAMBDA_CFM_DEFAULT_FLAGS}"

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
#
# 2026-11-03 (§5.11): these tuples are the OLD command-line spelling of the
# per-arm YAML keys (use_yuelbond / wasserstein_weight / closure_bounded).
# They are kept here for the DRY_RUN.log + plan.json report; the trainer
# receives them from the arm's YAML.
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
# 2026-11-03 (§5.11): there is no command-line passthrough any more — the
# trainer receives --config + --out-dir only — so this block reports WHERE
# the flags live instead of claiming they are passed through.
cat >> "${LOG_DIR}/DRY_RUN.log" <<EOF
  extra_flags: (none — AGENTS.md §5.11: the trainer receives --config/--out-dir only)
  flags_location: yaml (train.lambda_cfm_coupling* + per-arm use_yuelbond/wasserstein_weight/closure_bounded)
  arm_configs: full_stack=${ABLATION_CONFIG[full_stack]} | yuelbond_only=${ABLATION_CONFIG[yuelbond_only]} | w2_only=${ABLATION_CONFIG[w2_only]} | catherine_only=${ABLATION_CONFIG[catherine_only]}
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
# 2026-11-03 (§5.11): `config` used to name ONE file; the launch now uses
# one YAML per ablation arm, so it carries the map.  No consumer reads this
# key (molmetal/scripts/gpu_launch_aggregator.py reads `status` /
# `total_runs`; molmetal/scripts/wf426a3_post_fix_results_aggregator.py
# reads the `meta` block) — MEASURED by grep, 2026-11-03.
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
  "config": {
    "full_stack": "${ABLATION_CONFIG[full_stack]}",
    "yuelbond_only": "${ABLATION_CONFIG[yuelbond_only]}",
    "w2_only": "${ABLATION_CONFIG[w2_only]}",
    "catherine_only": "${ABLATION_CONFIG[catherine_only]}"
  },
  "epochs_per_config": ${EPOCHS},
  "seed": ${SEED},
  "pocket_id": "${POCKET_ID}",
  "n_pairs": ${N_PAIRS},
  "pair_budget_hr": ${PER_CONFIG_BUDGET_HR},
  "total_budget_hr": ${TOTAL_BUDGET_HR},
  "gpu_assignment": {
    "default": "cuda:0 (PRO 6000 Blackwell)",
    "note": "wrapper default per FIX_SCHEMA spec; the device is YAML-carried (compute.device)"
  },
  "default_flags": "${LAMBDA_CFM_DEFAULT_FLAGS}",
  "effective_flags": "${EFFECTIVE_FLAGS}",
  "ablation_configs": {
    "full_stack": "${ABLATION_FLAGS[full_stack]}",
    "yuelbond_only": "${ABLATION_FLAGS[yuelbond_only]}",
    "w2_only": "${ABLATION_FLAGS[w2_only]}",
    "catherine_only": "${ABLATION_FLAGS[catherine_only]}"
  },
  "transparent_passthrough": false,
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
    printf 'trainer receives --config + --out-dir only (AGENTS.md §5.11); '
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
    ARM_CONFIG="${ABLATION_CONFIG[${key}]}"
    CONFIG_OUT_DIR="${LOG_DIR}/${key}"
    mkdir -p "${CONFIG_OUT_DIR}"
    printf '[Wf-394-A1] === Config: %s (config: %s; yaml-carried flags: %s) ===\n' \
        "${key}" "${ARM_CONFIG}" "${ABLATION_FLAG_TUPLE}"
    # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6
    # (the pre-§5.11 command line, 2026-11-03):
    #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" \
    #   setsid nohup         "${PY}" "${CFM_TRAINER}" \
    #       --config "${CFG}" \
    #       --epochs "${EPOCHS}" \
    #       --seed "${SEED}" \
    #       --pocket-id "${POCKET_ID}" \
    #       --device cuda:0 \
    #       --out-dir "${CONFIG_OUT_DIR}" \
    #       ${EFFECTIVE_FLAGS} \
    #       ${ABLATION_FLAG_TUPLE} \
    #       > "${CONFIG_OUT_DIR}/train.log" 2>&1 || {
    #
    # 2026-11-03 (AGENTS.md §5.11): --epochs / --seed / --pocket-id / `--device`
    # / ${EFFECTIVE_FLAGS} / ${ABLATION_FLAG_TUPLE} are HYPERPARAMETERS, so
    # they are keys of the arm's YAML now (train.epochs, protocol.seed,
    # protocol.pocket_id, compute.device, train.lambda_cfm_coupling*,
    # flow.use_yuelbond / eval.wasserstein_weight / flow.closure_bounded).
    # --config and --out-dir stay: they select WHICH config and WHERE the
    # run writes, never WHAT it trains.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" \
setsid nohup         "${PY}" "${CFM_TRAINER}" \
            --config "${ARM_CONFIG}" \
            --out-dir "${CONFIG_OUT_DIR}" \
            > "${CONFIG_OUT_DIR}/train.log" 2>&1 || {
                printf '[Wf-394-A1][WARN] config %s exited non-zero; continuing\n' "${key}" >&2
            }
    printf '[Wf-394-A1] === Config: %s complete ===\n' "${key}"
done

printf '[Wf-394-A1] all 4 ablation configs complete\n'
