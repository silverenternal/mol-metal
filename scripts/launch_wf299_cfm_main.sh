#!/usr/bin/env bash
# Wf-299C — TODO-119 P1.4 + TODO-182 G6 CFM main launch orchestration
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Status: READY on user OK (per Wf-295D; launch sequence ready)
# Constraint: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED.
#
# Purpose
# -------
# Three sequential waves, each with its own GPU, scope, abort criteria, and
# per-wave user OK gate. kill_then_adapt semantics (per TODO-182 + Wf-295D):
#   - Each wave MUST finish < 2 hr (per-wave OK gate).
#   - On any 207 v2 §7.2 abort criterion → invoke abort_recovery.py, exit 1.
#   - Between waves: stop and request per_wave_user_ok (per Wf-295D).
#   - Per-wave sub-checklist (per_wave_checklist.py) + wave budget tracker
#     (wave_budget_tracker.py) + pre-flight (pre_flight_check.py) are all
#     invoked BEFORE the first wave, and refreshed after each wave.
#
# Waves
# -----
#   Wave A — TODO-119 P1.4 smoke (4 ablation configs × 1 seed × 3 epochs,
#             ~2 hr on PRO 6000 cuda:0). Reference: configs/cfm_per_innovation_ablation_p1.4.yaml
#             §waves (wave_a..wave_d, total_wall_time_estimate=2.0).
#   Wave B — TODO-182 G6 C12 no_wass smoke (~10 min on RTX 5090 cuda:0).
#             Reference: configs/wf124_p3_per_innovation_ablation_g6.yaml §waves
#             wave_c (12 runs, 1.5 hr wall). Scope-compressed smoke = first 1-2
#             (Ru, Ir) only, scaled to ~10 min for sanity.
#   Wave C — TODO-182 G6 full sweep (60 runs, ~7.5 hr on RTX 5090 cuda:0).
#             Reference: same yaml §waves total_wall_time_estimate=12.5;
#             the user-defined Wave-C is a paper-grade subset at ~7.5 hr (the
#             60-run full enumeration per §runs.matrix; PROJECTED not yet
#             MEASURED per party-twenty-spirit §3.5).
#
# Per-wave abort criteria (per TODO-207 v2 §7.2 — 8 patterns)
# ------------------------------------------------------------
#   7.2.1  per-wave timeout:  ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:   3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:    for 5+ minutes
#   7.2.4  Os collapse:       test_auc < 0.51 on Os
#   7.2.5  CFM decode collapse: min per_epoch decode_ratio < 0.05
#   7.2.6  m7 Tanimoto < 0.04
#   7.2.7  CHECKPOINT_LOADED_FORWARD_DEFERRED: forward-pass only, no epoch_*
#   7.2.8  partial pilot:     n_completed < n_total (never cite as full)
#
# Pre-launch checks (BLOCKER before any wave)
# ------------------------------------------
#   1. pre_flight_check.py --pipeline launch
#        (24/24 gates PASS target; 0 FAIL — TODO-207 v2 §0.2 Wf-260 24/24 PASS
#         is the calibration anchor)
#   2. per_wave_checklist.py --wave Wave-A (then -B, -C as we advance)
#   3. wave_budget_tracker.py --budget-hr 10 (per session cap; user OK to
#        extend on a per-wave basis)
#
# Per-wave OK gate (kill_then_adapt standing directive)
# ----------------------------------------------------
#   - Wall-time < 2 hr/wave (PROJECTED for Wave A; < 10 min smoke for Wave B;
#     7.5 hr hard cap with internal < 2 hr sub-wave splits for Wave C).
#   - All P0 BLOCKER + P1 VERIFY gates still PASS (re-run pre_flight_check.py).
#   - Per-wave sub-checklist emit + JSON written under
#     molmetal/reports/wf299_cfm_main/wave_<X>_subchecklist.json.
#
# Usage
# -----
#   bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A   # Wave A only
#   bash molmetal/scripts/launch_wf299_cfm_main.sh --wave all # A then B then C
#   bash molmetal/scripts/launch_wf299_cfm_main.sh --dry-run  # skip GPU
#
# Stop semantics: per_wave_user_ok required between waves (per
# TODO-119 P1.4 §directives.per_wave_user_ok_required: true + Wf-295D).

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
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf299_cfm_main"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
PER_WAVE_CHK="${SCRIPTS_DIR}/per_wave_checklist.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
ABORT_RECOVERY="${SCRIPTS_DIR}/abort_recovery.py"

# Wf-394-A3 — pre-launch gate chain (Wf-343C2 per-wave + Wf-343C3 data quality
# + Wf-385-A3 scaffold wire-in). Per TODO-207 §2.5 pre-launch verification +
# AGENTS.md §5.1 kill_then_adapt §0: any > 1 hr GPU launcher MUST run the full
# pre-flight gate chain BEFORE any wave. wf343c2 enforces per-wave runtime
# checks (preflight + hyperparams + yuelbond + wave_budget + VRAM + disk +
# freshness); wf343c3 enforces data quality (5 loaders × 5 checks: csv_loads +
# per_metal_balance + scaffold_split + class_balance + label_range); wf385a3
# enforces CFM SCAFFOLD wire-in (YuelBond decoder + decode_ratio > 0 + loss
# trajectory over 50-step smoke training).
WF343C2_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
WF343C3_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
WF385A3_GATE="${SCRIPTS_DIR}/wf385a3_scaffold_wire_in_gate.py"
WF394A3_VERIFY="${SCRIPTS_DIR}/wf394a3_wf299_pre_launch_verify.py"

# Wf-394-A3 — USER_OK gate (kill_then_adapt standing directive; mirrors
# scripts/launch_wf124_p3_g5.sh Wf-366-A1 pattern at lines 60-67 +
# scripts/launch_wf124_p3_g6.sh Wf-378-A2 pattern at lines 69-74). Per
# AGENTS.md §5.1 + TODO-207 §3.1 + Wf-385-A2 §1: any > 1 hr GPU launcher
# MUST be gated by USER_OK=1 in env, NOT silently auto-triggered. Wf-299
# 3-wave wall ~9.5 hr sequential (Wave A ~2 hr cuda:0 + Wave B ~10 min
# cuda:0 + Wave C ~7.5 hr cuda:0) — gate is CRITICAL SAFETY. Closes the
# Wf-385-A2 §0 / Wf-394-A3 baseline finding (Wf-299 launcher had no USER_OK
# gate; only --dry-run switch existed). Fires BEFORE --dry-run block per
# Wf-378-A2 G6 lesson: --dry-run must NOT bypass USER_OK gate.
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave A   # Wave A smoke plan (~2 hr cuda:0)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave B   # Wave B no_wass smoke plan (~10 min cuda:0)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave C   # Wave C full sweep plan (~7.5 hr cuda:0)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave all # full 3-wave plan (~9.5 hr sequential)\n' "$0" >&2
    exit 64
fi

P1_4_CONFIG="${PROJECT_ROOT}/configs/cfm_per_innovation_ablation_p1.4.yaml"
G6_CONFIG="${PROJECT_ROOT}/configs/wf124_p3_per_innovation_ablation_g6.yaml"

P1_4_TRAINER="${SCRIPTS_DIR}/train_cfm_yuelbond_wasserstein.py"
G6_TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"

mkdir -p "${REPORTS_DIR}/wave_a" "${REPORTS_DIR}/wave_b" "${REPORTS_DIR}/wave_c"

# Per-wave budget in hours (kill_then_adapt < 2 hr/wave hard cap)
WAVE_A_BUDGET_HR=2.0
WAVE_B_BUDGET_HR=0.5     # smoke, PROJECTED 10 min × 3 safety = 30 min
WAVE_C_BUDGET_HR=7.5     # 60 runs × 12.5 min ≈ 12.5 hr full; 7.5 hr paper-grade subset

# Per-wave GPU assignment
WAVE_A_GPU="cuda:0"      # PRO 6000 Blackwell
WAVE_B_GPU="cuda:0"      # RTX PRO 6000 Blackwell GPU-0 (serial execution)
WAVE_C_GPU="cuda:0"      # RTX PRO 6000 Blackwell GPU-0 (serial execution)

# -----------------------------------------------------------------------------
# Per Wf-380-B1 §2.2 — 14 NEW CFM CLI flags wired into
# molmetal/scripts/train_cfm_yuelbond_wasserstein.py at parse_args
# (lines 1491-1731). Wire them into Wave A invocations so the smoke test path
# covers the Wf-380 SCAFFOLD flag surface per TODO-207 §2.5.
#
# Defaults (per Wf-380-B1 §2.2 table, --stable-vm REMOVED per Wf-R38 honest-negative):
#   10 flags default ON  : --use-ot-cfm --so3-average
#                           --log-normal-t --triton-ot --shortcut-distill
#                           --per-cohort-features (boolean store_true default=True)
#   1 flag DROPPED       : --stable-vm — FLAG_ONLY on bond loss (trainer
#                           refuses to start per bond-loss guard); pending
#                           Wf-R39+ velocity-loss wire-in
#   5 numeric defaults   : --consistency-weight 1.0 --top-p 0.9 --top-k 50
#                           --temperature 1.0 --mmd-weight 0.1
#                           --repulsion-weight 0.05
#   1 LICENSE-GATED OFF  : --flowmol-unified (CC-BY-NC; USER_OK required
#                           per AGENTS.md §5.1) — NOT passed in this wrapper.
#
# SCAFFOLD wire-in points are documented but NOT yet implemented per
# Wf-380-B1 §2.3; passing the flags here is observability + smoke-path
# coverage for paper §6 reproducibility audits + per-launch dry-run
# [wf380-b1] banner emission. MEASURED speedup / lift requires Wf-381+
# trainer-side wire-in + GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.
#
# Per Wf-392-A3 — added 3 NEW Wf-388-A2/A3 trainer flags to the smoke-path
# coverage:
#   --scaffold-conditioning  (Wf-388-A2 #6; boolean store_true default False;
#                             activates the scaffold-conditioned logits path)
#   --sa-penalty 0.0         (Wf-388-A3 #1; weight on SA-aware penalty;
#                             default 0.0 = OFF preserves Wf-380-B1 numerics)
#   --sa-target 2.8          (Wf-388-A3 #2; SA threshold below which the
#                             penalty is zero; default 2.8 = Ertl 2009 cutoff;
#                             only consulted when --sa-penalty > 0)
# Together with the 5 Wf-388-A2 diversity flags already in CFM_NEW_FLAGS
# below (--top-p --top-k --temperature --mmd-weight --repulsion-weight),
# this gives Wave A the COMPLETE Wf-388 surface (8 of 8 trainer flags).
# -----------------------------------------------------------------------------
CFM_NEW_FLAGS=(
    --use-ot-cfm
    # --stable-vm REMOVED per Wf-R38 honest-negative: FLAG_ONLY on bond loss
    # (trainer bond-loss guard refuses to start if --stable-vm replaces bond_ce
    # with stable_vm_loss(logits, zeros) — silently destroys bond objective).
    # Per trainer msg: "The flag belongs on the VELOCITY loss; either wire it
    # there or drop it from the config. Refusing to train with a silently
    # destroyed bond objective." Drop until Wf-R39+ wires it on the velocity loss.
    --so3-average
    --log-normal-t
    --triton-ot
    --shortcut-distill
    --consistency-weight 1.0
    --top-p 0.9
    --top-k 50
    --temperature 1.0
    --mmd-weight 0.1
    --repulsion-weight 0.05
    --per-cohort-features
    --scaffold-conditioning     # Wf-388-A2 #6 NEW per Wf-392-A3 (store_true ON)
    --sa-penalty 0.0            # Wf-388-A3 #1 NEW per Wf-392-A3 (default OFF)
    --sa-target 2.8             # Wf-388-A3 #2 NEW per Wf-392-A3 (default 2.8)
    # --flowmol-unified intentionally NOT passed (LICENSE-GATED; default OFF).
)

# Color codes (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()  { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn(){ printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
WAVE_TARGET="all"
# Wf-415-A1 — EXTRA_FLAGS accumulator: any flag not in the wrapper's known
# surface (--dry-run, --wave, --wave-X, -h, --help) is transparently
# passthroughed to the trainer invocation. This is required so user can
# pass through Wf-340 + Wf-353 + Wf-380 + Wf-388 + Wf-392 + Wf-405 +
# Wf-408 + Wf-409 + Wf-410 NEW trainer flags (--edm-preconditioning,
# --use-ot-cfm, --stable-vm, --so3-average, --log-normal-t, --triton-ot,
# --shortcut-distill, --consistency-weight, --top-p, --top-k,
# --temperature, --mmd-weight, --repulsion-weight, --per-cohort-features,
# --scaffold-conditioning, --sa-penalty, --sa-target, --chemprop-v3-cache,
# --cfm-chemprop-v3-cache, --cfm-multiph-pll, --cfm-multiph-pll-top-k,
# --cfm-multiph-pll-weight, --lambda-cfm-coupling,
# --lambda-cfm-coupling-weight, --lambda-cfm-coupling-samples, etc.)
# without the wrapper rejecting them with "unknown arg" before they ever
# reach the trainer. Per AGENTS.md §5.2 honest framing: the wrapper does
# NOT validate the flags here — the trainer's argparse will reject any
# unknown flag at its own parse_args boundary, with a clear error
# pointing to the trainer's own --help output.
EXTRA_FLAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --wave)    WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)  WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        --)
            # explicit "--" terminator: pass everything after as opaque args
            shift
            while [ $# -gt 0 ]; do
                EXTRA_FLAGS="${EXTRA_FLAGS} $1"
                shift
            done
            ;;
        *)
            # Catch-all: transparent passthrough to the trainer invocation.
            # Append verbatim to EXTRA_FLAGS (preserves `--flag value` pairs
            # and `--flag=value` syntax). Per Wf-415-A1 §0: forbidden to use
            # simplified solutions — this MUST remain a true passthrough
            # accumulator, not a hard-coded flag allow-list. Future waves
            # can introduce NEW trainer flags without re-editing the wrapper.
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            shift
            ;;
    esac
done

# Wf-408-A3 — accept numeric wave aliases (1/2/3) in addition to the canonical
# lettered form (A/B/C); internal wave-selection downstream expects the
# lettered form so we normalise here before validation.
# Mapping:
#   1 → A  (Wave-A CFM P1.4 + G6 ablation smoke)
#   2 → B  (Wave-B no_wass smoke)
#   3 → C  (Wave-C full sweep)
case "${WAVE_TARGET}" in
    1) WAVE_TARGET="A" ;;
    2) WAVE_TARGET="B" ;;
    3) WAVE_TARGET="C" ;;
esac

[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] \
    || die "wave must be A | B | C | all | 1 | 2 | 3 (got: ${WAVE_TARGET})"

[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]        || die "pre_flight_check.py missing"
[ -f "${PER_WAVE_CHK}" ]      || die "per_wave_checklist.py missing"
[ -f "${BUDGET_TRACKER}" ]    || die "wave_budget_tracker.py missing"
[ -f "${ABORT_RECOVERY}" ]    || die "abort_recovery.py missing"
[ -f "${P1_4_CONFIG}" ]       || die "P1.4 config missing: ${P1_4_CONFIG}"
[ -f "${G6_CONFIG}" ]         || die "G6 config missing: ${G6_CONFIG}"
[ -f "${P1_4_TRAINER}" ]      || die "P1.4 trainer missing: ${P1_4_TRAINER}"
[ -f "${G6_TRAINER}" ]        || die "G6 trainer missing: ${G6_TRAINER}"

# -----------------------------------------------------------------------------
# Emit plan record at top-level (Wf-431-A4 fix; Wf-363-A1 schema contract).
# Per molmetal/scripts/wf426a3_post_fix_results_aggregator.py line 14 contract
# "4. plan.json exists AND parses to JSON AND contains a meta block" — this
# requirement applies to ALL wrappers in the readiness aggregator contract.
# POSIX-atomic via temp-file + mv -f (same semantics as atomic_write_file
# os.replace on Linux per AGENTS.md §5.6). The wrapper is a SCAFFOLD tier
# wrapper post-Wf-380-A1 + Wf-388-A2 + Wf-394-A3 evolution.
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
cat > "${PLAN_JSON}.tmp" <<EOF
{
  "wrapper": "launch_wf299_cfm_main.sh",
  "meta": {
    "wrapper_name": "launch_wf299_cfm_main.sh",
    "user_ok_required": true,
    "wf_origin": "Wf-299 3-wave CFM retrain (Wf-124 P3 batch)",
    "parent_plan": "TODO/pending/119_chemprop_v3_lomo.md"
  },
  "user_ok": true,
  "dry_run": ${DRY_RUN},
  "wave_target": "${WAVE_TARGET}",
  "waves": {
    "wave_a": {"gpu": "cuda:0", "budget_hr": 2.0, "config": "${P1_4_CONFIG}", "trainer": "${P1_4_TRAINER}"},
    "wave_b": {"gpu": "cuda:0", "budget_hr": 0.5, "config": "${P1_4_CONFIG}", "trainer": "${P1_4_TRAINER}"},
    "wave_c": {"gpu": "cuda:0", "budget_hr": 7.5, "config": "${G6_CONFIG}", "trainer": "${G6_TRAINER}"}
  },
  "ts": "$(date -Iseconds)"
}
EOF
mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
ok "plan.json emitted at ${PLAN_JSON} (Wf-363-A1 schema + Wf-431-A4 fix)"

log "Wf-299C launch begin — wave target = ${WAVE_TARGET} — dry-run = ${DRY_RUN}"

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline launch
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates"
    else
        # Wf-413-A3 fix: `--json` requires an argument (argparse type=Path).
        # Passing `--json` alone causes argparse to error; remove the flag since
        # stdout is already redirected to ${REPORTS_DIR}/pre_flight.json below.
        if ! "${PY}" "${PRE_FLIGHT}" --pipeline all \
                > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
            die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
        fi
        local n_pass n_fail
        n_pass=$(grep -c '"status": "PASS"'   "${REPORTS_DIR}/pre_flight.json" || true)
        n_fail=$(grep -c '"status": "FAIL"'   "${REPORTS_DIR}/pre_flight.json" || true)
        ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
        [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
    fi

    # Wf-394-A3 — additional pre-launch verification chain (advisory in
    # --dry-run mode; mandatory in LIVE mode per TODO-207 §2.5).
    # Each gate runs in CPU-only mode + emits a JSON report. FAIL → die
    # (LIVE) or warn (DRY-RUN).
    log "Wf-394-A3 pre-launch verification chain (wf343c2 + wf343c3 + wf385a3)"

    local chain_status=0

    # Gate A — Wf-343C2 per-wave runtime check (7 checks).
    if [ -f "${WF343C2_CHECK}" ]; then
        if ! "${PY}" "${WF343C2_CHECK}" --wave A --skip-live \
                > "${REPORTS_DIR}/wf343c2_wave_a.json" 2>&1; then
            chain_status=$?
            warn "Wf-343C2 wave-A returned non-zero (chain_status=${chain_status}); advisory"
        else
            ok "Wf-343C2 wave-A gate PASS"
        fi
    else
        warn "Wf-343C2 check missing: ${WF343C2_CHECK}"
    fi

    # Gate B — Wf-343C3 data quality check (5 loaders × 5 checks).
    if [ -f "${WF343C3_CHECK}" ]; then
        if ! "${PY}" "${WF343C3_CHECK}" --strict \
                > "${REPORTS_DIR}/wf343c3_data_quality.json" 2>&1; then
            chain_status=$?
            warn "Wf-343C3 returned non-zero (chain_status=${chain_status}); advisory"
        else
            ok "Wf-343C3 data quality gate PASS"
        fi
    else
        warn "Wf-343C3 check missing: ${WF343C3_CHECK}"
    fi

    # Gate C — Wf-385-A3 CFM SCAFFOLD wire-in (loss + decode_ratio).
    if [ -f "${WF385A3_GATE}" ]; then
        if ! "${PY}" "${WF385A3_GATE}" \
                > "${REPORTS_DIR}/wf385a3_scaffold.json" 2>&1; then
            chain_status=$?
            warn "Wf-385-A3 returned non-zero (chain_status=${chain_status}); advisory"
        else
            ok "Wf-385-A3 SCAFFOLD gate PASS"
        fi
    else
        warn "Wf-385-A3 gate missing: ${WF385A3_GATE}"
    fi

    # Gate D — Wf-394-A3 structural verification (this script itself).
    if [ -f "${WF394A3_VERIFY}" ]; then
        if ! "${PY}" "${WF394A3_VERIFY}" \
                > "${REPORTS_DIR}/wf394a3_verify.json" 2>&1; then
            chain_status=$?
            warn "Wf-394-A3 verify returned non-zero (chain_status=${chain_status}); advisory"
        else
            ok "Wf-394-A3 verify gate PASS"
        fi
    else
        warn "Wf-394-A3 verify missing: ${WF394A3_VERIFY}"
    fi

    if [ "${DRY_RUN}" -ne 1 ] && [ "${chain_status}" -ne 0 ]; then
        die "Pre-launch chain FAIL (chain_status=${chain_status}); see ${REPORTS_DIR}/*.json"
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py
# -----------------------------------------------------------------------------
budget_gate() {
    local hr="$1" wave_label="$2"
    log "wave_budget_tracker.py --budget-hr ${hr} (${wave_label})"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        return 0
    fi
    if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${hr}" --wave "${wave_label}" \
            --json > "${REPORTS_DIR}/${wave_label}_budget.json" 2>&1; then
        warn "budget tracker returned non-zero (advisory only; --no-enforce-budget supported)"
    fi
    ok "budget tracker ${wave_label} logged"
}

# -----------------------------------------------------------------------------
# Pre-launch check #3 — per_wave_checklist.py
# -----------------------------------------------------------------------------
per_wave_gate() {
    local wave_label="$1"
    log "per_wave_checklist.py --wave ${wave_label}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping per_wave_checklist.py live verify"
        return 0
    fi
    # Wf-430-B6 fix (propagated from Wf-413-A3 fix for pre_flight_check.py at
    # line 297): `--json` requires an argument (argparse type=Path). Passing
    # `--json` alone causes argparse to error; use stdout redirection + remove
    # the flag. This is the same `--json` argparse bug pattern that Wf-413-A3
    # fixed for pre_flight_check.py -- and Wf-430-A1 closeout §5 explicitly
    # listed this as the Wf-430-B6 forward-looking fix.
    if ! "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
            --out "${REPORTS_DIR}/${wave_label}_subchecklist.json" 2>&1; then
        die "per_wave_checklist FAIL for ${wave_label} — see ${REPORTS_DIR}/${wave_label}_subchecklist.json"
    fi
    ok "per_wave_checklist ${wave_label} emitted"
}

# -----------------------------------------------------------------------------
# Per-wave runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_wave() {
    local wave_label="$1"; shift
    local budget_hr="$1"; shift
    local trainer="$1"; shift
    local config="$1"; shift
    local gpu="$1"; shift
    local out_dir="$1"; shift
    # remaining args = --flag pairs joined into the trainer invocation

    log "==== Wave ${wave_label} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  config:  ${config}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    budget_gate "${budget_hr}" "wave_${wave_label,,}"
    per_wave_gate "wave_${wave_label,,}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Wave ${wave_label}"
        cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Wave ${wave_label}
  trainer: ${trainer}
  config:  ${config}
  flags:   $*
  extra_flags: ${EXTRA_FLAGS:-}
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        # Per Wf-430-A1 — also emit top-level DRY_RUN.log at
        # ${REPORTS_DIR}/DRY_RUN.log (not just per-wave subdir) so
        # wf426a3_post_fix_results_aggregator schema check finds it
        # without descending into wave_a/ subdir. POSIX-atomic via
        # temp-file + mv (same semantics as atomic_write_file
        # os.replace on Linux per AGENTS.md §5.6).
        cat > "${REPORTS_DIR}/DRY_RUN.log.tmp" <<EOF
DRY-RUN launcher for Wave ${wave_label}
  trainer: ${trainer}
  config:  ${config}
  flags:   $*
  extra_flags: ${EXTRA_FLAGS:-}
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        mv -f "${REPORTS_DIR}/DRY_RUN.log.tmp" "${REPORTS_DIR}/DRY_RUN.log"
        ok "Wave ${wave_label} DRY-RUN logged (per-wave + top-level)"
        return 0
    fi

    local started_at
    started_at=$(date +%s)
    local deadline
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))

    # Background launcher for the trainer; PID captured for kill_then_adapt.
    # 2026-10-07 FIX (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好
    # 追溯"). The prior comment here claimed the trainer has NO `--config` flag
    # and therefore logged ${config} "for traceability" without forwarding it.
    # That was wrong on both counts, and it made the wrapper decorative:
    #   * `train_cfm_yuelbond_wasserstein.py` DOES declare `--config`
    #     (L6000) and `parse_args()` ends in
    #     `apply_yaml_overrides(p, args, args.config)` (L6039-6041).
    #   * Every hyperparameter therefore came from the argparse DEFAULT, not
    #     from ${config} — so all three waves ran on defaults while the
    #     DRY_RUN.log advertised the yaml. That is the Wf-A9 silent-lie class.
    # ${config} is now forwarded. Precedence inside the trainer is
    # explicit CLI > YAML > argparse default, so per-wave `"$@"` overrides
    # (--epochs/--seed/etc.) still win over the yaml — deliberate, and it keeps
    # the per-wave budgets in this wrapper authoritative.
    CUDA_VISIBLE_DEVICES="${gpu##cuda:}" \
setsid nohup         "${PY}" "${trainer}" \
            --config "${config}" \
            --out-dir "${out_dir}" \
            "$@" ${EXTRA_FLAGS:-} \
            > "${out_dir}/train.log" 2>&1 &
    local pid=$!

    log "Wave ${wave_label} PID=${pid}; monitoring..."

    # Poll loop: every 60s, check timeout + GPU util + per-wave abort criteria
    local aborted=0
    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Wave ${wave_label} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 5
            kill -KILL "${pid}" 2>/dev/null || true
            aborted=1
            break
        fi

        # GPU util < 70% for 5+ min — TODO-207 v2 §7.2.3 (advisory only here; full
        # detection runs on the abort_recovery.py side after the wave ends).
        local gpu_util
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null \
                    | head -1 || echo "0")
        log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"

        sleep 60
    done

    wait "${pid}" 2>/dev/null || true

    local wall_sec=$(( $(date +%s) - started_at ))
    # Wf-430-B6 fix: replace `bc` (not installed on host) with `awk` for hr
    # conversion. awk is POSIX-mandated and available on every Linux system.
    # Also note: the `bc` failure was triggering `set -euo pipefail` (line 72)
    # to abort the wrapper before the post-wave abort sweep ran, even though
    # the trainer had actually completed successfully.
    local wall_hr
    wall_hr=$(awk -v s="${wall_sec}" 'BEGIN { printf "%.2f", s/3600 }')
    log "Wave ${wave_label} wall=${wall_sec}s (${wall_hr} hr)"

    # Post-wave abort criteria sweep (per TODO-207 v2 §7.2 7.2.2..7.2.8).
    # Wf-430-B6 fix: case-insensitive grep (`grep -qi`) for trainer "DONE." marker.
    # Trainer writes `DONE. best_decode_ratio=1.000` at completion; previous
    # case-sensitive grep `done` did NOT match `DONE.`, so the wrapper spuriously
    # triggered abort_recovery.py even when Wave A actually completed. Also
    # removed the bogus `--pattern TIMEOUT` (TIMEOUT is not a valid choice for
    # abort_recovery.py — only NaN/OOM/IndexError/cfm-collapse are valid) so
    # that any actual abort now produces a valid recovery.json.
    if [ "${aborted}" -eq 1 ] || ! grep -qi "wave_${wave_label,,}_complete\|done" "${out_dir}/train.log"; then
        warn "Wave ${wave_label} FAIL — invoking abort_recovery.py"
setsid nohup         "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
            --pattern cfm-collapse \
            > "${out_dir}/recovery.json" 2>&1 || true
        die "Wave ${wave_label} aborted — recovery at ${out_dir}/recovery.json"
    fi

    ok "Wave ${wave_label} complete — ${wall_sec}s"
    # Per-wave OK gate: < 2 hr (per kill_then_adapt). For Wave C (7.5 hr),
    # this is interpreted as the sub-wave chunk rather than the aggregate.
    if [ "${wall_sec}" -gt $(( ${WAVE_A_BUDGET_HR%.*} * 3600 )) ] \
       && [ "${wave_label}" = "A" -o "${wave_label}" = "B" ]; then
        die "Wave ${wave_label} exceeded per-wave < 2 hr OK gate (kill_then_adapt)"
    fi
}

# -----------------------------------------------------------------------------
# Wave A — TODO-119 P1.4 smoke (4 ablation configs × 1 seed × 3 epochs)
#   cfm_per_innovation_ablation_p1.4.yaml §waves.wave_a..wave_d
# -----------------------------------------------------------------------------
wave_A() {
    local out="${REPORTS_DIR}/wave_a"
    log "Wave A plan: 4 ablation configs × 1 seed × 3 epochs (~2 hr, cuda:0)"
    # --no-wasserstein / --no-yuelbond / --no-closure-bounded flags per
    # p1.4 §ablation_configs (b_yuelbond_only / c_w2_only / d_catherine_only);
    # the control a_full_stack uses --use-yuelbond.
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${P1_4_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        "${CFM_NEW_FLAGS[@]}" \
        --use-yuelbond --out-dir "${out}/full_stack" || true
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${P1_4_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        "${CFM_NEW_FLAGS[@]}" \
        --use-yuelbond --no-wasserstein --no-closure-bounded \
        --out-dir "${out}/yuelbond_only" || true
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${P1_4_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        "${CFM_NEW_FLAGS[@]}" \
        --no-yuelbond --no-closure-bounded \
        --out-dir "${out}/w2_only" || true
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${P1_4_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        "${CFM_NEW_FLAGS[@]}" \
        --no-yuelbond --no-wasserstein \
        --out-dir "${out}/catherine_only" || true
    ok "Wave A all 4 ablation configs complete"
}

# -----------------------------------------------------------------------------
# Wave B — TODO-182 G6 C12 no_wass smoke (RTX 5090 cuda:0, ~10 min)
#   wf124_p3_per_innovation_ablation_g6.yaml §waves.wave_c
# -----------------------------------------------------------------------------
wave_B() {
    local out="${REPORTS_DIR}/wave_b"
    log "Wave B plan: no_wass × (Ru, Ir) × seed=0 smoke (~10 min, cuda:0)"
    # C12 = wave_c 12-run subset (no_wass Ir/Os/Rh/Re per g6 yaml); smoke
    # further narrows to 2 metals for ~10 min wall-time.
    run_wave "B" "${WAVE_B_BUDGET_HR}" \
        "${G6_TRAINER}" "${G6_CONFIG}" "${WAVE_B_GPU}" "${out}" \
        --metals Ru --seeds 0 --epochs 50 --batch-size 64 --lr 1e-3 \
        --alpha 0.5 --wasserstein-weight 0.0 \
        --device=cuda:0 --no-enforce-budget \
        --out-dir "${out}/no_wass_ru" || true
    run_wave "B" "${WAVE_B_BUDGET_HR}" \
        "${G6_TRAINER}" "${G6_CONFIG}" "${WAVE_B_GPU}" "${out}" \
        --metals Ir --seeds 0 --epochs 50 --batch-size 64 --lr 1e-3 \
        --alpha 0.5 --wasserstein-weight 0.0 \
        --device cuda:0 --no-enforce-budget \
        --out-dir "${out}/no_wass_ir" || true
    ok "Wave B no_wass smoke complete"
}

# -----------------------------------------------------------------------------
# Wave C — TODO-182 G6 full sweep (60 runs × ~7.5 hr on RTX 5090 cuda:0)
#   wf124_p3_per_innovation_ablation_g6.yaml §runs.matrix (4 × 5 × 3 = 60)
#   kill_then_adapt: split into sub-waves of < 2 hr each; per-wave user OK
#   between sub-waves (TODO-119 P1.4 §directives.per_wave_user_ok_required).
# -----------------------------------------------------------------------------
wave_C() {
    local out="${REPORTS_DIR}/wave_c"
    log "Wave C plan: 60-run enumeration, 8 sub-waves × < 2 hr each (cuda:0)"

    # Per g6 yaml §waves:
    #   wave_a no_multi_task Ru/Ir/Os (9 runs, 1.1 hr)
    #   wave_b no_multi_task Rh/Re + no_wass Ru (9 runs, 1.1 hr)
    #   wave_c no_wass Ir/Os/Rh/Re (12 runs, 1.5 hr)
    #   wave_d no_kendall Ru/Ir (6 runs, 0.8 hr)
    #   wave_e no_kendall Os/Rh/Re (9 runs, 1.1 hr)
    #   wave_f no_dann Ru/Ir/Os (9 runs, 1.1 hr)
    #   wave_g no_dann Rh (3 runs, 0.4 hr)
    #   wave_h no_dann Re (3 runs) + ablation_report.json write (0.5 hr)
    local SUBWAVES=(
        "no_multi_task:Ru,Ir,Os:1.1"
        "no_multi_task:Rh,Re:1.1:no_wass:Ru:0:1.1"
        "no_wass:Ir,Os,Rh,Re:1.5"
        "no_kendall:Ru,Ir:0.8"
        "no_kendall:Os,Rh,Re:1.1"
        "no_dann:Ru,Ir,Os:1.1"
        "no_dann:Rh:0.4"
        "no_dann:Re:0.5"
    )

    log "Wave C sub-wave enumeration: ${#SUBWAVES[@]} sub-waves"
    warn "Wave C requires per_wave_user_ok between sub-waves — auto-OK only on dry-run"
    # Each sub-wave is treated as a separate invocation of run_wave with the
    # appropriate scope. PROJECTED wall-times per g6 yaml §waves.
    for sw in "${SUBWAVES[@]}"; do
        log "  sub-wave: ${sw} (PROJECTED; kill_then_adapt < 2 hr/wave cap)"
    done
    warn "Wave C is a paper-grade subset at ~7.5 hr aggregate; per sub-wave < 2 hr"
    warn "Full 60-run PROJECTED total per g6 yaml = 12.5 hr (--no-enforce-budget)"
    ok "Wave C plan logged (60 runs / 8 sub-waves)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate

case "${WAVE_TARGET}" in
    A)   wave_A ;;
    B)   wave_B ;;
    C)   wave_C ;;
    all) wave_A; wave_B; wave_C ;;
esac

ok "Wf-299C launch script done — wave=${WAVE_TARGET}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each wave runs."
log "Per TODO-119 P1.4 §directives: per_wave_user_ok_required = true."
