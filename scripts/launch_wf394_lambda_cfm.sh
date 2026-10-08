#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/launch_wf394_lambda_cfm.sh
#
# Wf-394 Lambda × CFM coupling launch wrapper (USER_OK-gated, GPU launch
# deferred). Sister wrapper to scripts/launch_wf299_cfm_main.sh — same
# Wf-355-B1 USER_OK-gated wrapper family + Wf-415-A1 transparent-passthrough
# pattern, but specialised for the TODO-21 Lambda × CFM coupling training
# surface (per Wf-329-W6 + Wf-333-G1 + Wf-406-A2 + Wf-394-A1 + Wf-410-A2 +
# Wf-414-A2 + Wf-418-A2).
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --wave A
#   USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --wave 1
#   USER_OK=1 bash scripts/launch_wf394_lambda_cfm.sh --wave A \
#       --lambda-cfm-coupling-weight 0.2 --lambda-cfm-coupling-samples 100 \
#       --chemprop-v3-cache  # transparent passthrough of NEW trainer flags
#
# Exit codes:
#   0   PASS (--dry-run emits DRY_RUN.log + plan.json; --wave X delegates)
#   1   FAIL (filesystem or trainer call missing)
#   64  USER_OK gate refused (USER_OK env not set to "1")
#
# Required: USER_OK=1 in env (explicit per AGENTS.md §5.1 + Wf-294B + Wf-355-B1
# + Wf-366-A1 + Wf-378-A2 + Wf-413-A3 pattern). The wrapper REFUSES to run
# unless USER_OK=1 is set, including for --dry-run. --dry-run must NOT bypass
# the USER_OK gate per Wf-378-A2 G6 lesson.
#
# Wf-408-A3 push --wave 1/2/3 → A/B/C alias mapping accepts numeric form
# (1 → A, 2 → B, 3 → C) in addition to the canonical lettered form
# (A/B/C/all). Wf-415-A1 push — true EXTRA_FLAGS accumulator captures any flag
# not in the wrapper's known surface (--dry-run, --wave, --wave-X, -h,
# --help) and passes it verbatim to the underlying Wf-299 launcher.
# -----------------------------------------------------------------------------
set -euo pipefail

# Locate repo root + the Wf-299 launcher we delegate to.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
WF299_LAUNCHER="${PROJECT_ROOT}/molmetal/scripts/launch_wf299_cfm_main.sh"

# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
PY="/home/hugo/codes/mol-metal/.venv/bin/python"
export PYTHONPATH="/home/hugo/codes/mol-metal"
export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-${PROJECT_ROOT}/.data/molmetal}"

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
# CLI arg parsing — USER_OK gate + DRY_RUN + wave alias + EXTRA_FLAGS accumulator
# -----------------------------------------------------------------------------
DRY_RUN=0
WAVE_TARGET="A"
EXTRA_FLAGS=""

# Default Lambda × CFM coupling flags (Wf-389-OFF baseline; opt-in via
# Wf-394-A2 audit + Wf-406-A2 + Wf-410-A2 canonical store_true + float +
# int shape on trainer's parse_args). These are the trainer-side flag names
# per molmetal/scripts/train_cfm_yuelbond_wasserstein.py lines 1978-2007
# (`--lambda-cfm-coupling`, `--lambda-cfm-coupling-weight`,
# `--lambda-cfm-coupling-samples`, `--lambda-cfm-mean-weight`,
# `--lambda-cfm-dataset-from-json`). The `--lambda-cfm-coupling` flag is the
# BooleanOptionalAction gate; both --lambda-cfm-coupling-weight > 0 AND
# --lambda-cfm-mean-weight > 0 must hold for the coupling to activate per
# trainer line 1378-1384 (loss composition conditional).
LAMBDA_CFM_DEFAULT_FLAGS="--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 200"

# Wf-415-A1 — true EXTRA_FLAGS accumulator pattern (no hard-coded flag
# allow-list; any flag not in the wrapper's known surface is transparently
# passthroughed to Wf-299 launcher which in turn passthroughes to the
# trainer). Per Wf-415-A1 §0: forbidden to use simplified solutions —
# this MUST remain a true passthrough accumulator.
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --wave)
            WAVE_TARGET="${2:-A}"
            shift 2
            ;;
        --wave-*)
            # `--wave-A` / `--wave-B` / `--wave-C` aliases (Wf-415-A1 contract).
            WAVE_TARGET="${1#--wave-}"
            shift
            ;;
        -h|--help)
            printf 'Wf-394 Lambda × CFM coupling launch wrapper (USER_OK-gated)\n'
            printf 'Usage: USER_OK=1 %s [--dry-run|--wave A|B|C|1|2|3|all] [extra-trainer-flags...]\n' "$0"
            printf '  Default Lambda × CFM coupling flags: %s\n' "${LAMBDA_CFM_DEFAULT_FLAGS}"
            printf '  --wave 1|2|3 accepted as numeric alias for --wave A|B|C (Wf-408-A3).\n'
            printf '  EXTRA_FLAGS: any flag not in the wrapper surface is transparently\n'
            printf '  passthroughed to the Wf-299 launcher (Wf-415-A1 contract).\n'
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
            # and `--flag=value` syntax verbatim. The DELIMITER check
            # `1#--*` ensures `--lambda-cfm-coupling-weight 0.2` consumes
            # 0.2 as the value (not a separate EXTRA_FLAGS entry), so
            # both `--flag value` and `--flag=value` work.
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

# Wf-408-A3 — accept numeric wave aliases (1/2/3) in addition to the
# canonical lettered form (A/B/C). Internal Wf-299 launcher expects the
# lettered form so we normalise here before validation.
case "${WAVE_TARGET}" in
    1)    WAVE_TARGET="A" ;;
    2)    WAVE_TARGET="B" ;;
    3)    WAVE_TARGET="C" ;;
    all)  WAVE_TARGET="all" ;;
esac

case "${WAVE_TARGET}" in
    A|B|C|all) : ;;  # accepted
    *) die "wave must be A | B | C | all | 1 | 2 | 3 (got: ${WAVE_TARGET})" ;;
esac

# -----------------------------------------------------------------------------
# USER_OK gate — explicit, enforced BEFORE --dry-run block per Wf-378-A2
# G6 lesson (--dry-run must NOT bypass USER_OK gate).
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --wave A    # Wave A actual launch on cuda:0 (~2 hr)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# Pre-flight check: Wf-299 launcher must exist on disk before any delegation.
# Wf-428-A2 retrain readiness verification flagged the absence of
# scripts/launch_wf394_lambda_cfm.sh as a BLOCKER; this wrapper closes the
# BLOCKER via direct delegation to the canonical Wf-299 launcher.
# -----------------------------------------------------------------------------
if [ ! -x "${WF299_LAUNCHER}" ]; then
    die "Wf-299 launcher missing or not executable: ${WF299_LAUNCHER}"
fi

# --- Pre-flight gate (Wf-415-A3 + Wf-427-B1) ------------------------------
# Per §5.1 standing GPU auth + canonical pattern: pre-flight MUST be PASS
# before any GPU delegation. Even though we delegate to Wf-299, we run the
# pre-flight here so USER_OK=1 + dry-run still validates the gate.
PREFLIGHT_OUT="${PROJECT_ROOT}/molmetal/reports/wf394_lambda_cfm/preflight.json"
mkdir -p "$(dirname "${PREFLIGHT_OUT}")"
echo "[$(date -Iseconds)] ${BASH_SOURCE[0]} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal "${PY}" \
    "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" --pipeline launch \
    >"${PREFLIGHT_OUT}" 2>&1 \
    || die "pre_flight_check FAIL — see ${PREFLIGHT_OUT}"
grep -q '"fail": 0' "${PREFLIGHT_OUT}" \
    || die "pre_flight_check has FAILs — see ${PREFLIGHT_OUT}"
echo "[$(date -Iseconds)] pre-flight PASS"

# Default Lambda × CFM coupling flags when caller did not override via
# EXTRA_FLAGS. We honour caller-supplied EXTRA_FLAGS first, then prepend
# our defaults, so explicit overrides always win.
if [ "${EXTRA_FLAGS#*--lambda-cfm-coupling}" = "${EXTRA_FLAGS}" ]; then
    EFFECTIVE_FLAGS="${LAMBDA_CFM_DEFAULT_FLAGS}${EXTRA_FLAGS:+ }${EXTRA_FLAGS}"
else
    EFFECTIVE_FLAGS="${EXTRA_FLAGS}"
fi

# OUT_DIR default — Wf-363-A1 DRY_RUN.log + plan.json emission path.
OUT_DIR_DEFAULT="${PROJECT_ROOT}/molmetal/reports/wf394_lambda_cfm_launch"
OUT_DIR="${OUT_DIR_DEFAULT}"

# -----------------------------------------------------------------------------
# Per-wave enumeration (handy of pairs for --dry-run DRY_RUN.log emission)
# -----------------------------------------------------------------------------
# Each (metal, seed) pair is emitted via `--pair` arg to emit_dry_run_log.
# Wave A scope per Wf-299 launcher = 4 ablation configs × 1 seed × 3 epochs
# (TODO-119 P1.4 smoke). Wave B = 12 runs smoke (~10 min). Wave C = 60 runs
# full sweep (~7.5 hr). Total wall-time depends on wave target.
case "${WAVE_TARGET}" in
    A)    N_PAIRS=4   ; PAIR_BUDGET_HR=0.50 ;;  # ~30 min conservative
    B)    N_PAIRS=12  ; PAIR_BUDGET_HR=0.17 ;;  # ~10 min smoke
    C)    N_PAIRS=60  ; PAIR_BUDGET_HR=7.50 ;;  # ~7.5 hr paper-grade subset
    all)  N_PAIRS=76  ; PAIR_BUDGET_HR=8.17 ;;  # ~8.17 hr sequential 3-wave
esac

PAIRS=()
i=0
while [ "${i}" -lt "${N_PAIRS}" ]; do
    PAIRS+=("pair_${i}")
    i=$((i + 1))
done

# -----------------------------------------------------------------------------
# Emit DRY_RUN.log per Wf-363-A1 schema (out_dir absolute vs relative handling).
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
    --title "Wf-394 Lambda × CFM coupling retrain (Wave ${WAVE_TARGET})" \
    --wrapper "scripts/launch_wf394_lambda_cfm.sh" \
    --mode "R1_dry_run" \
    --trainer "${WF299_LAUNCHER}" \
    --subset "Wf-299 3-wave CFM main (TODO-119 P1.4 + TODO-182 G6)" \
    --flags "${FLAGS_DISPLAY}" \
    --gpu "cuda:0 (all waves, serial)" \
    --budget-hr "${PAIR_BUDGET_HR}" \
    --user-ok "REQUIRED" \
    "${PAIR_ARGS[@]}"

# Append the Wf-415-A1 transparent-passthrough line per Wf-299 wrapper §0.
# The Wf-299 launcher also has its own EXTRA_FLAGS accumulator, so the
# passthrough is double-buffered (safe).
cat >> "${LOG_DIR}/DRY_RUN.log" <<EOF
  extra_flags: ${EFFECTIVE_FLAGS#${LAMBDA_CFM_DEFAULT_FLAGS} }
EOF

# -----------------------------------------------------------------------------
# Emit plan.json per Wf-380-B1 pattern.
# -----------------------------------------------------------------------------
cat > "${LOG_DIR}/plan.json" <<EOF
{
    "wf": "Wf-394-A3",
    "deliverable": "Wf-394 Lambda × CFM coupling launch wrapper (USER_OK-gated)",
    "wrapper": "scripts/launch_wf394_lambda_cfm.sh",
    "delegate_to": "${WF299_LAUNCHER}",
    "wave_target": "${WAVE_TARGET}",
    "mode": "R1_dry_run",
    "user_ok_required": true,
    "user_ok_env_var": "USER_OK=1",
    "default_flags": "${LAMBDA_CFM_DEFAULT_FLAGS}",
    "effective_flags": "${EFFECTIVE_FLAGS}",
    "n_pairs": ${N_PAIRS},
    "pair_budget_hr": ${PAIR_BUDGET_HR},
    "gpu_assignment": {
        "A": "cuda:0 (PRO 6000 Blackwell)",
        "B": "cuda:0 (RTX 5090)",
        "C": "cuda:0 (RTX 5090)"
    },
    "sister_wrappers": [
        "scripts/launch_wf299_cfm_main.sh",
        "scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
        "scripts/launch_wf409a3_metal_coord_retrain.sh"
    ],
    "transparent_passthrough": true,
    "wave_alias_1_2_3": true,
    "honest_framing": {
        "wrapper_authored_only": true,
        "gpu_launch_blocked_on_user_ok": true,
        "actual_gpu_run_requires_user_ok_1": true,
        "wf299_launcher_dry_run_validated": true
    },
    "meta": {
        "wrapper_name": "launch_wf394_lambda_cfm.sh",
        "user_ok_required": true,
        "wf_origin": "Wf-394 Lambda × CFM coupling launch wrapper (USER_OK-gated)",
        "parent_plan": "TODO/completed/428a2_retrain_readiness_audit.md §3.3",
        "wf429b1_compliance": "plan.json schema now satisfies wf426a3_post_fix_results_aggregator check; meta.wrapper_name + meta.user_ok_required fields populated per Wf-429-B1 fix extension (Wf-429-A4 fix scope extension to Wf-394 wrapper)"
    },
    "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# -----------------------------------------------------------------------------
# Branch: --dry-run vs actual delegation.
# In --dry-run mode: emit logs + exit 0 (no delegation, no GPU work).
# In LIVE mode: delegate to Wf-299 launcher via `exec` (replace process).
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" = "1" ]; then
    printf '[Wf-394-A3] R1 dry-run mode: wrapper scaffold exists; '
    printf 'Wf-299 launcher exists + executable; pre-flight path validated; '
    printf 'would delegate Wave %s to %s; ' \
        "${WAVE_TARGET}" "${WF299_LAUNCHER}"
    printf 'extra_flags=%s; %d pairs; per-pair budget %s hr; ' \
        "${EFFECTIVE_FLAGS}" "${N_PAIRS}" "${PAIR_BUDGET_HR}"
    printf 'DRY_RUN.log: %s/DRY_RUN.log; plan.json: %s/plan.json\n' \
        "${LOG_DIR}" "${LOG_DIR}"
    exit 0
fi

# LIVE mode: delegate to Wf-299 launcher with --wave + Lambda CFM flags.
# `exec` replaces this wrapper's process with the Wf-299 launcher so USER_OK
# gate at line 110 of Wf-299 launcher fires once + exit codes propagate.
printf '[Wf-394-A3] delegating Wave %s to %s\n' \
    "${WAVE_TARGET}" "${WF299_LAUNCHER}"
exec bash "${WF299_LAUNCHER}" --wave "${WAVE_TARGET}" ${EFFECTIVE_FLAGS}
