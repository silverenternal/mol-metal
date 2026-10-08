#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/launch_wf380_scaffold_loss.sh — Wf-380 SCAFFOLD-loss MEASURED retrain wrapper
#
# Wf-429-A2 (2026-10-01) — author this wrapper per Wf-428-A2 readiness
# verification (wf380_scaffold_loss candidate BLOCKED because this wrapper
# MISSING on disk per AGENTS.md §5.5 filesystem-truth rule).
#
# Canonical SCAFFOLD-flag retrain path per Wf-424-A7 §9: this wrapper is a
# thin canonical surface that DELEGATES to the Wf-299 launcher (which has
# Wf-415-A1 transparent-passthrough + Wf-408-A3 `--wave 1/2/3` alias mapping
# already SHIPPED at launcher lines 210-251 + 256-269). 22+ SCAFFOLD flags
# (Wf-340 + Wf-353 + Wf-380 + Wf-388 + Wf-392 + Wf-405 + Wf-408 + Wf-409 +
# Wf-410) transparent-passthrough via Wf-415-A1 case-statement catch-all.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf380_scaffold_loss.sh --dry-run
#     # CPU verification only; emits DRY_RUN.log + plan.json
#   USER_OK=1 bash scripts/launch_wf380_scaffold_loss.sh
#     # actual ~10 hr GPU run on cuda:0 Blackwell (per Wf-260 G6 timing)
#   USER_OK=1 bash scripts/launch_wf380_scaffold_loss.sh --wave A \
#     --use-ot-cfm --stable-vm --so3-average --log-normal-t \
#     --triton-ot --shortcut-distill --consistency-weight 1.0 \
#     --per-cohort-features --scaffold-conditioning
#     # Wave A SCAFFOLD-full ablation; all flags passed via EXTRA_FLAGS
#     # accumulator through Wf-299 wrapper to trainer.
#
# Standing directives honored:
# - AGENTS.md §5.1 USER_OK gating: hard-gate BEFORE --dry-run per Wf-294B
#   family + Wf-378-A2 block-order lesson (closes Wf-363-A2 + Wf-365-B2
#   HONEST-NEGATIVE finding where --dry-run bypassed USER_OK gate).
# - AGENTS.md §5.5 filesystem-truth rule: each SHIPPED file verified via ls
#   -la before authoring this wrapper.
# - AGENTS.md §5.6 atomic write protocol: this wrapper authored via
#   atomic_write_file from molmetal/scripts/wf386a3_atomic_write.py.
# - AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule
#   (Wf-399-A1): pre-flight `p3_wf385_scaffold_loss_wire_in_verified` MUST
#   PASS before any actual GPU launch (caller responsibility; wrapper
#   only emits plan + delegates to Wf-299).
# - mol-metal-gpu-execution §0 kill_then_adapt directive: NO auto-launch;
#   USER_OK=1 required.
#
# Wall-time estimate (per TODO-181 §3 + Wf-260 G6 + Wf-424-A7):
#   - Wave A full SCAFFOLD (4 ablation configs × 30 min conservative):
#     ~2 hr on RTX PRO 6000 Blackwell GPU-0
#   - Wave B smoke (2 configs): ~10 min
#   - Wave C full sweep (single config × 30 min): ~30 min
#   - Total: ~3 hr on Blackwell GPU-0 (per Wf-424-A7 §14 PROJECTED)
# -----------------------------------------------------------------------------

set -euo pipefail


# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# --- Pre-flight gate (Wf-415-A3 + Wf-427-B1) ------------------------------
PREFLIGHT_OUT="${PROJECT_ROOT:-/home/hugo/codes/mol-metal}/molmetal/reports/wf_preflight/preflight.json"
mkdir -p "$(dirname "${PREFLIGHT_OUT}")"
echo "[$(date -Iseconds)] ${BASH_SOURCE[0]} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    "${PROJECT_ROOT:-/home/hugo/codes/mol-metal}/molmetal/scripts/pre_flight_check.py" --pipeline launch \
    >"${PREFLIGHT_OUT}" 2>&1 \
    || die "pre_flight_check FAIL — see ${PREFLIGHT_OUT}"
grep -q '"fail": 0' "${PREFLIGHT_OUT}" \
    || die "pre_flight_check has FAILs — see ${PREFLIGHT_OUT}"
echo "[$(date -Iseconds)] pre-flight PASS"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Source the shared DRY_RUN.log helper (consistent per-pair schema per
# Wf-363-A1). Verified presence per §5.5.
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] \
    || { printf '[FATAL] DRY_RUN log helper missing: %s\n' "${DRY_RUN_LOG_HELPER}" >&2; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Canonical Wf-299 launcher (SCAFFOLD-flag retrain surface per Wf-424-A7 §9).
WF299_LAUNCHER="${REPO_ROOT}/molmetal/scripts/launch_wf299_cfm_main.sh"
[ -x "${WF299_LAUNCHER}" ] \
    || { printf '[FATAL] Wf-299 launcher missing or non-executable: %s\n' "${WF299_LAUNCHER}" >&2; exit 1; }

# --- Wf-294B family USER_OK gate (BEFORE --dry-run block per Wf-378-A2) ---
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run        # plan + DRY_RUN.log only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --wave A [scaffold_flags...]   # actual ~3 hr GPU run\n' "$0" >&2
    printf '       AGENTS.md §5.1 + kill_then_adapt §0: NO auto-launch permitted.\n' >&2
    exit 64
fi

# --- CLI parser with EXTRA_FLAGS accumulator (Wf-415-A1 pattern) + --wave alias ---
DRY_RUN=0
WAVE_TARGET="all"  # default to "all" matches Wf-299 wrapper convention
PREREG=""
EXTRA_FLAGS=""
MODE="full_stack"
ABORT_ON_FAIL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --wave)
            WAVE_TARGET="${2:-all}"
            shift 2
            ;;
        --wave-*)
            WAVE_TARGET="${1#--wave-}"
            shift
            ;;
        --prereg)
            PREREG="${2:-}"
            shift 2
            ;;
        --prereg=*)
            PREREG="${1#--prereg=}"
            shift
            ;;
        --mode)
            MODE="${2:-full_stack}"
            shift 2
            ;;
        --abort-on-fail)
            ABORT_ON_FAIL=1
            shift
            ;;
        -h|--help)
            sed -n '2,/^# ---/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --)
            # Explicit "--" terminator: pass everything after as opaque args
            # (Wf-415-A1 §1.1 contract). nargs="+" equivalent for per-pair
            # argv expansion (e.g. `--metals Ru Ir Rh` unquoted).
            shift
            while [ $# -gt 0 ]; do
                EXTRA_FLAGS="${EXTRA_FLAGS} $1"
                shift
            done
            ;;
        *)
            # Wf-415-A1 catch-all: transparent passthrough to Wf-299 wrapper.
            # Future waves can introduce NEW SCAFFOLD flags without re-editing
            # this wrapper; Wf-299 wrapper's own EXTRA_FLAGS accumulator
            # forwards to the trainer via `${EXTRA_FLAGS:-}` at run_wave().
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            shift
            ;;
    esac
done

# --wave 1/2/3 alias support (Wf-408-A3 contract mirrored at wrapper layer
# for early-rejection of invalid wave aliases BEFORE delegation).
case "${WAVE_TARGET}" in
    1) WAVE_TARGET="A" ;;
    2) WAVE_TARGET="B" ;;
    3) WAVE_TARGET="C" ;;
esac

case "${WAVE_TARGET}" in
    A|B|C|all) ;;
    *) printf '[FATAL] --wave must be A | B | C | all | 1 | 2 | 3 (got: %s)\n' "${WAVE_TARGET}" >&2; exit 1 ;;
esac

OUT_BASE="${REPO_ROOT}/molmetal/reports/wf380_scaffold_loss"
LOG="${OUT_BASE}/master.log"
GPU="cuda:0"  # Wf-424-A7 §14 RTX PRO 6000 Blackwell GPU-0
WALL_BUDGET_HR="3.0"  # Wf-424-A7 §14 PROJECTED ~3 hr Blackwell
PY="/home/hugo/codes/mol-metal/.venv/bin/python"
TRAINER="${REPO_ROOT}/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"

mkdir -p "${OUT_BASE}"

# --- DRY-RUN path: CPU-only plan + DRY_RUN.log + plan.json emission ---
if [ "${DRY_RUN}" = "1" ]; then
    printf '%s[Wf-429-A2]%s Wf-380 SCAFFOLD-loss wrapper DRY-RUN (USER_OK=1 confirmed)\n' \
        "$(printf '\033[0;36m')" "$(printf '\033[0m')" | tee -a "${LOG}"
    printf '  wrapper:    %s\n' "$0" | tee -a "${LOG}"
    printf '  wf299:      %s\n' "${WF299_LAUNCHER}" | tee -a "${LOG}"
    printf '  trainer:    %s\n' "${TRAINER}" | tee -a "${LOG}"
    printf '  wave:       %s\n' "${WAVE_TARGET}" | tee -a "${LOG}"
    printf '  mode:       %s\n' "${MODE}" | tee -a "${LOG}"
    printf '  gpu:        %s\n' "${GPU}" | tee -a "${LOG}"
    printf '  budget:     %s hr\n' "${WALL_BUDGET_HR}" | tee -a "${LOG}"
    printf '  extra_flags: %s\n' "${EXTRA_FLAGS:-<none>}" | tee -a "${LOG}"
    printf '  ts:         %s\n' "$(date -Iseconds)" | tee -a "${LOG}"
    printf '  prereg:     %s\n' "${PREREG:-<none>}" | tee -a "${LOG}"
    printf '  user_ok:    1\n' | tee -a "${LOG}"

    # Per Wf-363-A1 schema: emit DRY_RUN.log via shared helper.
    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "Wf-429-A2 Wf-380 SCAFFOLD-loss wrapper DRY-RUN" \
        --wrapper "$(basename "$0")" \
        --mode "${MODE}" \
        --trainer "${TRAINER}" \
        --flags "--wave ${WAVE_TARGET} ${EXTRA_FLAGS:-}" \
        --gpu "${GPU}" \
        --budget-hr "${WALL_BUDGET_HR}" \
        --user-ok "REQUIRED" \
        --pair "Wave ${WAVE_TARGET} SCAFFOLD-loss retrain"
    printf 'DRY_RUN.log emitted -> %s\n' "${OUT_BASE}/DRY_RUN.log" | tee -a "${LOG}"

    # plan.json per Wf-378-A2 schema (consistent with Wf-355 pattern).
    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_wf380_scaffold_loss.sh",\n'
        printf '  "wf_origin": "Wf-429-A2 (Wf-380 BLOCKED closure per Wf-428-A2 §9.2 + Wf-426-A3 launcher contract)",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "wave_target": "%s",\n' "${WAVE_TARGET}"
        printf '  "mode": "%s",\n' "${MODE}"
        printf '  "ablation_configs": ["full_stack", "yuelbond_only", "w2_only", "catherine_only"],\n'
        printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
        printf '  "seeds": [0, 1, 2],\n'
        printf '  "epochs": 30,\n'
        printf '  "gpu": "%s",\n' "${GPU}"
        printf '  "wall_budget_hr": %s,\n' "${WALL_BUDGET_HR}"
        printf '  "wall_budget_label": "PROJECTED (~3 hr Blackwell GPU-0 per Wf-424-A7 §14)",\n'
        printf '  "trainer": "%s",\n' "${TRAINER}"
        printf '  "wf299_delegate": "%s",\n' "${WF299_LAUNCHER}"
        printf '  "extra_flags": "%s",\n' "${EXTRA_FLAGS:-}"
        printf '  "prereg": "%s",\n' "${PREREG:-}"
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "wrapper ready; USER_OK gate enforced BEFORE --dry-run per Wf-378-A2 block-order lesson; WAVE_TARGET validated with Wf-408-A3 1/2/3 alias; EXTRA_FLAGS captured for Wf-415-A1 transparent passthrough to Wf-299 wrapper",\n'
        printf '    "PROJECTED": "actual GPU launch ~3 hr Blackwell per Wf-424-A7 §14 + Wf-260 G6 timing",\n'
        printf '    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1; AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule MUST be satisfied (pre-flight p3_wf385_scaffold_loss_wire_in_verified PASS) before actual GPU launch",\n'
        printf '    "wf429_a2_status": "Wf-429-A2 closes Wf-428-A2 §9.2 BLOCKED candidate (wf380_scaffold_loss wrapper missing on disk); canonical SCAFFOLD-flag retrain surface is the Wf-299 launcher with Wf-415-A1 transparent-passthrough + Wf-408-A3 alias mapping per Wf-424-A7 §9; this wrapper provides a stable canonical surface + USER_OK gate + DRY_RUN.log + plan.json emission for the Wf-380 SCAFFOLD-loss retrain"\n'
        printf '  },\n'
        printf '  "meta": {\n'
        printf '    "wrapper_name": "launch_wf380_scaffold_loss.sh",\n'
        printf '    "user_ok_required": true,\n'
        printf '    "wf_origin": "Wf-380 SCAFFOLD-loss retrain (USER_OK gate + 22+ SCAFFOLD flag transparent-passthrough via Wf-415-A1)",\n'
        printf '    "parent_plan": "TODO/completed/428a2_retrain_readiness_audit.md §3.2",\n'
        printf '    "wf429b1_compliance": "plan.json schema now satisfies wf426a3_post_fix_results_aggregator check; meta.wrapper_name + meta.user_ok_required fields populated per Wf-429-B1 fix extension (Wf-429-A4 fix scope extension to Wf-380 wrapper)"\n'
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    printf 'plan.json emitted -> %s\n' "${PLAN_JSON}" | tee -a "${LOG}"

    printf '\n[OK] DRY-RUN complete: 0 GPU launches; %s ready for actual launch.\n' "$(basename "$0")" | tee -a "${LOG}"
    printf '     USER_OK=1 bash %s --wave A [scaffold_flags...]  # actual GPU launch\n' "$0" | tee -a "${LOG}"
    exit 0
fi

# --- Real-launch path: DELEGATE to Wf-299 wrapper (Wf-415-A1 + Wf-408-A3 surface) ---
printf '%s[Wf-429-A2]%s USER_OK=1 set; DELEGATING to Wf-299 wrapper (Wf-415-A1 transparent-passthrough + Wf-408-A3 alias mapping)\n' \
    "$(printf '\033[0;32m')" "$(printf '\033[0m')" | tee -a "${LOG}"
printf '  wave: %s | extra_flags: %s\n' "${WAVE_TARGET}" "${EXTRA_FLAGS:-<none>}" | tee -a "${LOG}"

# Wf-429-A2 core: delegate to canonical Wf-299 launcher. The Wf-299 launcher
# handles all canonical CFM NEW_FLAGS + per-pair argv expansion + Wf-415-A1
# transparent-passthrough of any remaining user-passed flags. This wrapper
# adds: (a) USER_OK gate (Wf-294B + Wf-378-A2 block-order lesson),
# (b) Wf-408-A3 1/2/3 → A/B/C alias normalization at wrapper layer for
# early-rejection of bad aliases, (c) DRY_RUN.log + plan.json emission per
# Wf-363-A1 + Wf-378-A2 schemas, (d) canonical surface name for Wf-380
# SCAFFOLD-loss retrain per Wf-428-A2 §9.2 + Wf-426-A3 launcher contract.
exec "${WF299_LAUNCHER}" --wave "${WAVE_TARGET}" --user-ok \
    ${ABORT_ON_FAIL:+--abort-on-fail} \
    ${EXTRA_FLAGS:-}
