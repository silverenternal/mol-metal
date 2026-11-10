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
#   bash scripts/launch_wf299_cfm_main.sh --wave A   # Wave A only
#   bash scripts/launch_wf299_cfm_main.sh --wave all # A then B then C
#   bash scripts/launch_wf299_cfm_main.sh --dry-run  # skip GPU
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

# Wf-A269 wave-10 (2026-11-02) - REPAIR: the budget gate had NO wave-hour
# INPUT at all (its old call passed `--wave` + `--json`, neither of which
# wave_budget_tracker.py defines, so argparse exited 1 and the tool never
# ran).  The PROJECTED hours below are the wave YAML's own
# `wall_time_estimate` values — they are PROJECTED, not MEASURED, and they
# are the budget gate's INPUT (AGENTS.md §5.2):
#   Wave A (P1.4): configs/cfm_per_innovation_ablation_p1.4.yaml:207-237 —
#     waves.wave_a..wave_d carry wall_time_estimate 0.5 each = 2.0 hr
#     (total_wall_time_estimate: 2.0 at line 237).
#   Wave B (G6 no_wass smoke): configs/wf124_p3_per_innovation_ablation_g6.yaml:202-203
#     — wave_c is 12 runs / 1.5 hr; this wrapper narrows it to 2 metals x
#     1 seed (launch_wf299_cfm_main.sh:651-660) = 2/12 of the runs = 0.25 hr.
#   Wave C (G6 full sweep): configs/wf124_p3_per_innovation_ablation_g6.yaml:194-228
#     — waves.wave_a..wave_h carry wall_time_estimate
#     1.1+1.1+1.5+0.8+1.1+1.1+0.4+0.5 = 8.6 hr (the same 8 sub-waves are
#     enumerated at launch_wf299_cfm_main.sh:683-692).
WAVE_A_PROJECTED_HR=2.0
WAVE_B_PROJECTED_HR=0.25
WAVE_C_PROJECTED_HR=8.6

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
# Wf-A269 wave-10 (2026-11-02) — the shared launch-gate verdict CONSUMER.
#
# Every pre-launch gate below reads its verdict out of the REPORT the gate
# tool already wrote, through ONE shared consumer
# (molmetal/scripts/wf_launch_gate_verdict.py).  The wrapper therefore never
# re-implements a report parser and never greps a tool's stdout for a
# marker.  Same helper shape as the two already-repaired wave-10 wrappers
# (scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:96-120 and
# scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:520-568).
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     n_pass=$(grep -c '"status": "PASS"'   "${REPORTS_DIR}/pre_flight.json" || true)
#     n_fail=$(grep -c '"status": "FAIL"'   "${REPORTS_DIR}/pre_flight.json" || true)
#     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
#     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
#
# MEASURED evidence for the repair (2026-11-02):
#   * molmetal/reports/wf299_cfm_main/pre_flight.json (63429 B on disk) is
#     NOT valid JSON: the wrapper redirected the tool's stdout+stderr into
#     the .json path, so the file holds log lines, not a report.
#   * molmetal/scripts/pre_flight_check.py:17327-17333 declares `--json` as
#     `type=Path, nargs="?"`; the report is written only when
#     `args.json is not None` (:17701-17705) and the payload is PRINTED to
#     stdout when the flag is absent (:17706) — a bare `--json` writes NO
#     file at all.
#   * The pre-flight's authoritative counts live in the report's `summary`
#     block; grepping '"status": "PASS"' counts GATE ROWS, not the summary.
# -----------------------------------------------------------------------------
GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

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

# Wf-A269 wave-10 (2026-11-02) - REPAIR: a gate whose tool has no report
# format the shared consumer understands (the Wf-394-A3 chain tools) is read
# from that tool's EXIT CODE — the verdict channel AGENTS.md §4 fixes for
# every gate tool in this repo (exit 0=PASS, 1=FAIL, 2=SKIP).  The tools'
# own reports are still written to THIS wrapper's reports dir, and their
# stdout still goes to a .log; only the VERDICT SOURCE differs.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     if ! "${PY}" "${WF343C2_CHECK}" --wave A --skip-live \
#             > "${REPORTS_DIR}/wf343c2_wave_a.json" 2>&1; then
#         chain_status=$?
#         warn "Wf-343C2 wave-A returned non-zero (chain_status=${chain_status}); advisory"
#
# MEASURED: `$?` inside `if ! cmd` is the status of the `!` compound, which
# is ALWAYS 0 — so `chain_status` was 0 on every path and the chain's die
# at the end of this function was unreachable.
gate_verdict_from_rc() {
    case "$1" in
        0) printf 'PASS' ;;
        1) printf 'FAIL' ;;
        *) printf 'UNVERIFIED' ;;
    esac
}

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
    log "pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Wf-A269 wave-10 (2026-11-02) - REPAIR: the dry-run branch printed
        # nothing about the gate it did NOT run, so a reader of the dry-run
        # log could not tell "not measured" from "measured clean".
        # AGENTS.md §5.2: a number that was not measured is UNVERIFIED,
        # never PASS.  The verdict is now RECORDED, on the report's face.
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates (verdict UNVERIFIED)"
        printf 'UNVERIFIED not run: the dry-run skips the live pre-flight (the live pre-flight takes ~400 s on this repo, MEASURED)\n' \
            > "${REPORTS_DIR}/gate_preflight.verdict.txt"
    else
        # Wf-A269 wave-10 (2026-11-02) - REPAIR: the report was captured
        # with `> pre_flight.json 2>&1`, so the file held the tool's STDERR
        # log lines interleaved with the JSON payload on stdout and was NOT
        # valid JSON (MEASURED: the on-disk pre_flight.json is 63429 B of
        # log text; json.load raises "Expecting value: line 1 column 2" on
        # it).  The gate then counted GATE ROWS with grep instead of reading
        # the report's authoritative `summary` block.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     if ! "${PY}" "${PRE_FLIGHT}" --pipeline all \
        #             > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
        #         die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
        #     fi
        #     local n_pass n_fail
        #     n_pass=$(grep -c '"status": "PASS"'   "${REPORTS_DIR}/pre_flight.json" || true)
        #     n_fail=$(grep -c '"status": "FAIL"'   "${REPORTS_DIR}/pre_flight.json" || true)
        #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
        #     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
        #
        # REPAIRED: `--json <path>` writes a real JSON report (the flag is
        # `type=Path, nargs="?"`, so a bare `--json` writes NO file at all —
        # pre_flight_check.py:17327-17333, :17701-17705), the tool's stdout
        # goes to a .log, and the verdict is read from that report by the
        # shared checker.  A SKIP (report missing/unreadable) is recorded as
        # UNVERIFIED — never as PASS.
        #
        # Wf-A269 wave-10 C1-7 (2026-11-02) — ONE pre-flight failure decision,
        # and the tool's exit code is not thrown away.  The prior form carried
        # TWO decisions (the rc guard AND an unreachable count-based second
        # blocker `[ "${n_fail}" -eq 0 ] || die ...`); the second was DEAD,
        # because pre_flight_check.py returns 1 iff n_fail > 0
        # (pre_flight_check.py:17741-17742), so the rc guard had already
        # terminated the script before the count check could ever run.  The
        # rc is now CAPTURED, and the single decision below is a numeric
        # comparison on that captured rc, OR-ed with the report-based verdict:
        # one die, two fail-closed signals, no second blocker.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #         "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
        #             > "${REPORTS_DIR}/pre_flight.stdout.log" 2>&1 || true
        #         local pf_verdict pf_detail
        #         pf_verdict="$(run_gate_check preflight "${REPORTS_DIR}/pre_flight.json")"
        #         pf_detail="$(gate_detail preflight)"
        #         if [ "${pf_verdict}" = "SKIP" ]; then
        #             pf_verdict="UNVERIFIED"
        #             pf_detail="UNVERIFIED ${pf_detail}"
        #             printf '%s\n' "${pf_detail}" > "${REPORTS_DIR}/gate_preflight.verdict.txt"
        #         fi
        #         if [ "${pf_verdict}" != "PASS" ]; then
        #             warn "pre_flight_check ${pf_verdict} (${pf_detail})"
        #             die "pre_flight_check ${pf_verdict} — refusing to launch (report: ${REPORTS_DIR}/pre_flight.json)"
        #         fi
        #         ok "pre_flight ${pf_detail}"
        local pf_rc=0
        # Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A
        # crashed or killed pre-flight must not leave the last run's PASS
        # report on disk for the consumer to read — "stale is treated as
        # missing".
        rm -f "${REPORTS_DIR}/pre_flight.json"
        "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
            > "${REPORTS_DIR}/pre_flight.stdout.log" 2>&1 || pf_rc=$?
        local pf_verdict pf_detail
        pf_verdict="$(run_gate_check preflight "${REPORTS_DIR}/pre_flight.json")"
        pf_detail="$(gate_detail preflight)"
        if [ "${pf_verdict}" = "SKIP" ]; then
            pf_verdict="UNVERIFIED"
            pf_detail="UNVERIFIED ${pf_detail}"
            printf '%s\n' "${pf_detail}" > "${REPORTS_DIR}/gate_preflight.verdict.txt"
        fi
        if [ "${pf_rc}" -ne 0 ] || [ "${pf_verdict}" != "PASS" ]; then
            warn "pre_flight_check ${pf_verdict} (rc=${pf_rc}; ${pf_detail})"
            die "pre_flight_check ${pf_verdict} — refusing to launch (rc=${pf_rc}; report: ${REPORTS_DIR}/pre_flight.json)"
        fi
        ok "pre_flight ${pf_detail}"
    fi

    # Wf-394-A3 — additional pre-launch verification chain (advisory in
    # --dry-run mode; mandatory in LIVE mode per TODO-207 §2.5).
    # Each gate runs in CPU-only mode + emits a JSON report. FAIL → die
    # (LIVE) or warn (DRY-RUN).
    log "Wf-394-A3 pre-launch verification chain (wf343c2 + wf343c3 + wf385a3 + wf394a3)"

    local chain_status=0

    # Gate A — Wf-343C2 per-wave runtime check (7 checks).
    #
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: three defects in one gate.
    #   (a) the tool's report went to the tool's SHARED default path
    #       (reports/wf343c2_per_wave_check/per_wave_check.json) — the
    #       wrapper then wrote its stdout into a file NAMED
    #       wf343c2_wave_a.json that no tool ever writes (MEASURED on disk:
    #       252 B of plain text, not JSON);
    #   (b) the gate read the FROZEN archived pre-flight
    #       (reports/wf259_run/pre_flight.json, mtime 2026-10-02, pass=96)
    #       instead of THIS launch's own pre-flight;
    #   (c) `chain_status=$?` sat inside `if ! cmd`, where `$?` is the
    #       status of the `!` compound — ALWAYS 0 — so the chain's die at
    #       the end of this function was unreachable.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${WF343C2_CHECK}" --wave A --skip-live \
    #             > "${REPORTS_DIR}/wf343c2_wave_a.json" 2>&1; then
    #         chain_status=$?
    #         warn "Wf-343C2 wave-A returned non-zero (chain_status=${chain_status}); advisory"
    #     else
    #         ok "Wf-343C2 wave-A gate PASS"
    #     fi
    #
    # REPAIRED: the tool writes ITS OWN report to THIS wrapper's reports dir
    # (`--out`), reads THIS launch's own pre-flight (`--preflight`,
    # `--preflight-live`), keeps `--strict` so a SKIPped check can never be
    # read as a PASS, and the verdict comes from the tool's exit code
    # (AGENTS.md §4: 0=PASS, 1=FAIL, 2=SKIP → UNVERIFIED).
    #
    # MEASURED (2026-11-02, CPU-only): with `--strict` the tool exits 0 only
    # when all 7 checks PASS, 1 when any check FAILs, 2 when a check SKIPped.
    # HONEST-NEGATIVE: checks 3 (`hyperparams_config_complete`) and 5
    # (`wave_budget_remaining_nonneg`) are chemprop-family-specific —
    # check 3 requires `train:`/`eval:` sections (the P1.4 CFM config has
    # neither → FAIL, wf343c2_per_wave_check.py:467-484) and check 5 derives
    # its requirement from a 25-pair metals×seeds grid (this wave runs 4
    # ablation configs × 1 seed → FAIL).  Both are therefore LEFT at their
    # tool defaults (no --config / --budget-hr) so that this gate measures
    # what it can measure (pre-flight freshness + no-FAIL + YuelBond
    # wiring + live VRAM + live disk) instead of failing on a grid model
    # that belongs to another launcher family.  The consequence is that the
    # gate reads UNVERIFIED (2 checks SKIPped), never PASS — and never a
    # false PASS either.
    if [ -f "${WF343C2_CHECK}" ]; then
        if [ "${DRY_RUN}" -eq 1 ]; then
            warn "Wf-343C2 wave-A gate: UNVERIFIED (the dry-run skips the live pre-flight and the live GPU/disk probes)"
            chain_status=2
        else
            local a_rc=0
            "${PY}" "${WF343C2_CHECK}" --wave A --strict \
                --preflight "${REPORTS_DIR}/pre_flight.json" --preflight-live \
                --out "${REPORTS_DIR}/wf343c2_wave_a.json" \
                > "${REPORTS_DIR}/wf343c2_wave_a.stdout.log" 2>&1 || a_rc=$?
            local a_verdict
            a_verdict="$(gate_verdict_from_rc "${a_rc}")"
            case "${a_verdict}" in
                PASS) ok "Wf-343C2 wave-A gate PASS (report: ${REPORTS_DIR}/wf343c2_wave_a.json)" ;;
                FAIL) warn "Wf-343C2 wave-A gate FAIL (report: ${REPORTS_DIR}/wf343c2_wave_a.json)"; chain_status=1 ;;
                *)    warn "Wf-343C2 wave-A gate UNVERIFIED — a check did not measure (exit ${a_rc}; report: ${REPORTS_DIR}/wf343c2_wave_a.json)"; chain_status=2 ;;
            esac
        fi
    else
        warn "Wf-343C2 check missing: ${WF343C2_CHECK}"
        chain_status=1
    fi

    # Gate B — Wf-343C3 data quality check (5 loaders × 5 checks).
    #
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: the tool's report went to the
    # tool's SHARED default dir and its stdout into a file NAMED
    # wf343c3_data_quality.json that no tool writes (MEASURED on disk:
    # 690 B of plain text, not JSON); the verdict came from an always-0
    # exit code path.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${WF343C3_CHECK}" --strict \
    #             > "${REPORTS_DIR}/wf343c3_data_quality.json" 2>&1; then
    #         chain_status=$?
    #         warn "Wf-343C3 returned non-zero (chain_status=${chain_status}); advisory"
    #     else
    #         ok "Wf-343C3 data quality gate PASS"
    #     fi
    #
    # REPAIRED: `--out-dir` points at THIS wrapper's reports dir, the
    # tool's stdout goes to a .log, and the verdict comes from the tool's
    # exit code (0=PASS / 1=FAIL / 2=SKIP → UNVERIFIED,
    # wf343c3_data_quality_check.py:866-879).  MEASURED 2026-11-02 (CPU
    # only): exit 0, all 5 loaders overall=PASS, 4.5 s wall.
    if [ -f "${WF343C3_CHECK}" ]; then
        local b_rc=0
        "${PY}" "${WF343C3_CHECK}" --strict \
            --out-dir "${REPORTS_DIR}/wf343c3_data_quality" \
            > "${REPORTS_DIR}/wf343c3_data_quality.stdout.log" 2>&1 || b_rc=$?
        local b_verdict
        b_verdict="$(gate_verdict_from_rc "${b_rc}")"
        case "${b_verdict}" in
            PASS) ok "Wf-343C3 data quality gate PASS (report: ${REPORTS_DIR}/wf343c3_data_quality/report.json)" ;;
            FAIL) warn "Wf-343C3 data quality gate FAIL (report: ${REPORTS_DIR}/wf343c3_data_quality/report.json)"; chain_status=1 ;;
            *)    warn "Wf-343C3 data quality gate UNVERIFIED (exit ${b_rc} — no loader could be measured)"; chain_status=2 ;;
        esac
    else
        warn "Wf-343C3 check missing: ${WF343C3_CHECK}"
        chain_status=1
    fi

    # Gate C — Wf-385-A3 CFM SCAFFOLD wire-in (loss + decode_ratio).
    #
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: the gate's stdout went into a
    # file NAMED wf385a3_scaffold.json that no tool writes (MEASURED on
    # disk: 65 B = "verdict=FAIL / reason=smoke subprocess failed: timeout
    # after 90.0s" — a REAL failure the wrapper swallowed), the tool's own
    # report went to its SHARED default dir, and the verdict came from an
    # always-0 exit code path.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${WF385A3_GATE}" \
    #             > "${REPORTS_DIR}/wf385a3_scaffold.json" 2>&1; then
    #         chain_status=$?
    #         warn "Wf-385-A3 returned non-zero (chain_status=${chain_status}); advisory"
    #     else
    #         ok "Wf-385-A3 SCAFFOLD gate PASS"
    #     fi
    #
    # REPAIRED: `--out-dir` points at THIS wrapper's reports dir, the
    # tool's stdout goes to a .log, and the verdict comes from the tool's
    # exit code (0=PASS / 1=FAIL, wf385a3_scaffold_wire_in_gate.py:882).
    # `--timeout 900` is explicit because the tool's default (90 s) is
    # shorter than this host's smoke: MEASURED 2026-11-02 (CPU only) the
    # 50-step smoke takes 54.9 s wall on an idle host and OVERRAN the 90 s
    # default under load — that overrun is the FAIL recorded on disk today.
    # The dry-run does NOT run this gate at all: it is a 50-step TRAINING
    # smoke, and a dry-run is a plan (its verdict stays UNVERIFIED, never
    # PASS) — it also costs 55 s, more than the 60 s dry-run budget of the
    # repo's own wrapper tests (MEASURED: test_user_ok_gate.py timed out
    # after 60 s on this wrapper before the repair).
    if [ -f "${WF385A3_GATE}" ]; then
        if [ "${DRY_RUN}" -eq 1 ]; then
            warn "Wf-385-A3 SCAFFOLD gate: UNVERIFIED (the dry-run does not run the 50-step training smoke)"
            chain_status=2
        else
            local c_rc=0
            "${PY}" "${WF385A3_GATE}" --timeout 900 \
                --out-dir "${REPORTS_DIR}/wf385a3_scaffold" \
                > "${REPORTS_DIR}/wf385a3_scaffold.stdout.log" 2>&1 || c_rc=$?
            local c_verdict
            c_verdict="$(gate_verdict_from_rc "${c_rc}")"
            case "${c_verdict}" in
                PASS) ok "Wf-385-A3 SCAFFOLD gate PASS (report: ${REPORTS_DIR}/wf385a3_scaffold/gate_result.json)" ;;
                FAIL) warn "Wf-385-A3 SCAFFOLD gate FAIL (report: ${REPORTS_DIR}/wf385a3_scaffold/gate_result.json)"; chain_status=1 ;;
                *)    warn "Wf-385-A3 SCAFFOLD gate UNVERIFIED (exit ${c_rc})"; chain_status=2 ;;
            esac
        fi
    else
        warn "Wf-385-A3 gate missing: ${WF385A3_GATE}"
        chain_status=1
    fi

    # Gate D — Wf-394-A3 structural verification (this script itself).
    #
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: the gate's stdout went into a
    # file NAMED wf394a3_verify.json that no tool writes (MEASURED on disk:
    # 789 B of plain text, not JSON) and the verdict came from an always-0
    # exit code path.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${WF394A3_VERIFY}" \
    #             > "${REPORTS_DIR}/wf394a3_verify.json" 2>&1; then
    #         chain_status=$?
    #         warn "Wf-394-A3 verify returned non-zero (chain_status=${chain_status}); advisory"
    #     else
    #         ok "Wf-394-A3 verify gate PASS"
    #     fi
    #
    # REPAIRED: the tool has NO argparse surface (it writes its own fixed
    # report at molmetal/reports/wf394a3_wf299_pre_launch/result.json and
    # prints its verdict on stdout), so its stdout goes to a .log here and
    # its verdict comes from its exit code (0=PASS / 1=FAIL,
    # wf394a3_wf299_pre_launch_verify.py:889-891).  This gate is a
    # STRUCTURAL check of this very script, so it runs in the dry-run too.
    if [ -f "${WF394A3_VERIFY}" ]; then
        local d_rc=0
        "${PY}" "${WF394A3_VERIFY}" \
            > "${REPORTS_DIR}/wf394a3_verify.stdout.log" 2>&1 || d_rc=$?
        local d_verdict
        d_verdict="$(gate_verdict_from_rc "${d_rc}")"
        case "${d_verdict}" in
            PASS) ok "Wf-394-A3 verify gate PASS (report: molmetal/reports/wf394a3_wf299_pre_launch/result.json)" ;;
            FAIL) warn "Wf-394-A3 verify gate FAIL (report: molmetal/reports/wf394a3_wf299_pre_launch/result.json)"; chain_status=1 ;;
            *)    warn "Wf-394-A3 verify gate UNVERIFIED (exit ${d_rc})"; chain_status=2 ;;
        esac
    else
        warn "Wf-394-A3 verify missing: ${WF394A3_VERIFY}"
        chain_status=1
    fi

    # Wf-A269 wave-10 (2026-11-02) - REPAIR: this die is now REACHABLE.
    # `chain_status` is 1 when a gate FAILed and 2 when a gate could not
    # measure (UNVERIFIED).  A FAIL blocks the launch in LIVE mode; an
    # UNVERIFIED gate is loud but does not block (AGENTS.md §5.2: an
    # unmeasured value is UNVERIFIED, not FAIL).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if [ "${DRY_RUN}" -ne 1 ] && [ "${chain_status}" -ne 0 ]; then
    #         die "Pre-launch chain FAIL (chain_status=${chain_status}); see ${REPORTS_DIR}/*.json"
    #     fi
    if [ "${DRY_RUN}" -ne 1 ]; then
        if [ "${chain_status}" -eq 1 ]; then
            die "Pre-launch chain FAIL (chain_status=${chain_status}); see ${REPORTS_DIR}"
        elif [ "${chain_status}" -eq 2 ]; then
            warn "Pre-launch chain UNVERIFIED (chain_status=${chain_status}) — a gate could not measure; see ${REPORTS_DIR}"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py
# -----------------------------------------------------------------------------
budget_gate() {
    local hr="$1" wave_label="$2"
    local projected_hr=""
    case "${wave_label}" in
        wave_a) projected_hr="${WAVE_A_PROJECTED_HR}" ;;
        wave_b) projected_hr="${WAVE_B_PROJECTED_HR}" ;;
        wave_c) projected_hr="${WAVE_C_PROJECTED_HR}" ;;
    esac
    log "wave_budget_tracker.py --budget-hr ${hr} (${wave_label})"
    if [ -z "${projected_hr}" ]; then
        warn "budget tracker skipped for ${wave_label}: no PROJECTED wall-hours declared (verdict UNVERIFIED)"
        return 0
    fi
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: the old call passed `--wave`
    # and `--json`, NEITHER of which wave_budget_tracker.py defines (its
    # flags are --waves / --report-json / --budget-hr / --cfm-yuelbond-
    # required / ...).  argparse exited 1 with "ambiguous option: --wave
    # could match --waves, --wave-glob", so the tracker NEVER RAN — and the
    # file the wrapper called its report held that usage error verbatim
    # (MEASURED on disk: wave_a_budget.json is 638 B of argparse text).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${hr}" --wave "${wave_label}" \
    #             --json > "${REPORTS_DIR}/${wave_label}_budget.json" 2>&1; then
    #         warn "budget tracker returned non-zero (advisory only; --no-enforce-budget supported)"
    #     fi
    #     ok "budget tracker ${wave_label} logged"
    #
    # REPAIRED: `--waves name:hr` (the tool's real flag,
    # wave_budget_tracker.py:546-574) carries the wave's PROJECTED hours,
    # `--report-json` writes the report to THIS wrapper's reports dir, the
    # tool's stdout goes to a .log, and the verdict is read from that report
    # by the shared checker (the tracker's main() returns 0 on EVERY path —
    # wave_budget_tracker.py:1290-1422 — so the exit code cannot be the
    # verdict).  MEASURED 2026-11-02 (CPU only): wave_c 8.6 hr against the
    # 7.5 hr cap → the checker answers FAIL (remaining_hr=-1.1, rc=1);
    # wave_a 2.0/2.0 → PASS; wave_b 0.25/0.5 → PASS.  This gate stays
    # ADVISORY (it does not block the launcher) — but it now says something.
    "${PY}" "${BUDGET_TRACKER}" --budget-hr "${hr}" \
        --waves "${wave_label}:${projected_hr}" \
        --report-json "${REPORTS_DIR}/${wave_label}_budget.json" \
        > "${REPORTS_DIR}/${wave_label}_budget.stdout.log" 2>&1 || true
    local b_verdict b_detail
    b_verdict="$(run_gate_check budget "${REPORTS_DIR}/${wave_label}_budget.json" "${wave_label}")"
    b_detail="$(gate_detail budget)"
    # The shared checker writes one verdict file per gate KIND; keep a
    # per-wave copy so the per-wave evidence is not overwritten.
    printf '%s\n' "${b_detail}" > "${REPORTS_DIR}/gate_budget_${wave_label}.verdict.txt"
    case "${b_verdict}" in
        PASS) ok "budget ${wave_label} ${b_detail}" ;;
        *)    warn "budget ${wave_label} ${b_verdict} (${b_detail}) — ADVISORY: this gate does not block the launcher" ;;
    esac
}

# -----------------------------------------------------------------------------
# Pre-launch check #3 — per_wave_checklist.py
# -----------------------------------------------------------------------------
per_wave_gate() {
    local wave_label="$1"
    log "per_wave_checklist.py --wave ${wave_label}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Wf-A269 wave-10 (2026-11-02) - REPAIR: the dry-run branch said
        # nothing about the gate it did NOT run.  AGENTS.md §5.2: a gate
        # that was not measured is UNVERIFIED, never PASS.
        warn "DRY-RUN: skipping per_wave_checklist.py live verify (verdict UNVERIFIED)"
        printf 'UNVERIFIED not run: the dry-run skips the live per-wave checklist\n' \
            > "${REPORTS_DIR}/gate_checklist.verdict.txt"
        return 0
    fi
    # Wf-A269 wave-10 (2026-11-02) - REPAIR: this gate keyed on the tool's
    # EXIT CODE — and per_wave_checklist.py returns 0 UNCONDITIONALLY
    # (per_wave_checklist.py, `main()` has no `return 1` on a failed
    # checklist), so a checklist FAIL was reported as PASS.  MEASURED
    # 2026-11-02: a checklist built on a pre-flight with one P0 FAIL records
    # `p0_blocker_gates[*].status = FAIL` and `preflight_fail_count = 1`
    # in its own JSON while the tool still exits 0.
    #
    # The checklist also read the FROZEN archived pre-flight
    # (reports/wf259_run/pre_flight.json, mtime 2026-10-02, pass=96) — its
    # argparse default (per_wave_checklist.py:86) — instead of THIS
    # launch's own pre-flight.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
    #             --out "${REPORTS_DIR}/${wave_label}_subchecklist.json" 2>&1; then
    #         die "per_wave_checklist FAIL for ${wave_label} — see ${REPORTS_DIR}/${wave_label}_subchecklist.json"
    #     fi
    #     ok "per_wave_checklist ${wave_label} emitted"
    #
    # REPAIRED: `--preflight` points at THIS launch's own report, `--out` /
    # `--out-md` point at THIS wrapper's reports dir, the tool's stdout goes
    # to a .log, and the verdict comes from the checklist's OWN report via
    # the shared checker (PASS iff no error, the pre-flight it read has no
    # FAIL gate, and every P0 blocker gate is PASS).
    "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
        --preflight "${REPORTS_DIR}/pre_flight.json" \
        --out "${REPORTS_DIR}/${wave_label}_subchecklist.json" \
        --out-md "${REPORTS_DIR}/${wave_label}_subchecklist.md" \
        > "${REPORTS_DIR}/${wave_label}_subchecklist.stdout.log" 2>&1 || true
    local pw_verdict pw_detail
    pw_verdict="$(run_gate_check checklist "${REPORTS_DIR}/${wave_label}_subchecklist.json")"
    pw_detail="$(gate_detail checklist)"
    if [ "${pw_verdict}" = "SKIP" ]; then
        pw_verdict="UNVERIFIED"
        pw_detail="UNVERIFIED ${pw_detail}"
        printf '%s\n' "${pw_detail}" > "${REPORTS_DIR}/gate_checklist.verdict.txt"
    fi
    if [ "${pw_verdict}" != "PASS" ]; then
        warn "per_wave_checklist ${pw_verdict} (${pw_detail})"
        die "per_wave_checklist ${pw_verdict} — refusing to launch (report: ${REPORTS_DIR}/${wave_label}_subchecklist.json)"
    fi
    ok "per_wave_checklist ${pw_detail}"
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
    # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> ${gpu##cuda:}.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu##cuda:}}" \
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
        # Wf-A269 wave-10 (2026-11-02) - REPAIR: this call was DEAD THREE
        # WAYS.  (a) `--json` is NOT a flag — the tool declares
        # `--report-json` (abort_recovery.py:390-394) — so argparse exited 2
        # and NO report was written; (b) the REQUIRED `--metal`
        # (abort_recovery.py:373, choices Ru/Ir/Os/Rh/Re) was never passed;
        # (c) the tool's stdout+stderr was redirected into a file NAMED
        # recovery.json, which is therefore not JSON at all.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     setsid nohup         "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
        #         --pattern cfm-collapse \
        #         > "${out_dir}/recovery.json" 2>&1 || true
        #
        # REPAIRED: `--report-json` writes the tool's own report, a legal
        # `--pattern` is passed, the REQUIRED `--metal` is passed, and the
        # stdout goes to a .log.  HONEST-NEGATIVE: the probe's checkpoint
        # discovery is a metal-token filename filter
        # (abort_recovery.py:143-161) and this launcher's checkpoints carry
        # no metal token, so the probe reports "no matching checkpoint" for
        # the CFM/D-MPNN cohorts — the wave label and the pattern are what
        # it can actually narrow on.  Both are configurable
        # (ABORT_RECOVERY_PATTERN / ABORT_RECOVERY_METAL) and are validated
        # against the tool's own choice lists before use.
        local rec_pattern rec_metal
        rec_pattern="${ABORT_RECOVERY_PATTERN:-cfm-collapse}"
        rec_metal="${ABORT_RECOVERY_METAL:-Ru}"
        case "${rec_pattern}" in
            NaN|OOM|IndexError|cfm-collapse) : ;;
            *) warn "illegal ABORT_RECOVERY_PATTERN '${rec_pattern}' (legal: NaN|OOM|IndexError|cfm-collapse) — falling back to cfm-collapse"; rec_pattern="cfm-collapse" ;;
        esac
        case "${rec_metal}" in
            Ru|Ir|Os|Rh|Re) : ;;
            *) warn "illegal ABORT_RECOVERY_METAL '${rec_metal}' (legal: Ru|Ir|Os|Rh|Re) — falling back to Ru"; rec_metal="Ru" ;;
        esac
setsid nohup         "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
            --pattern "${rec_pattern}" --metal "${rec_metal}" \
            --report-json "${out_dir}/recovery.json" \
            > "${out_dir}/recovery.stdout.log" 2>&1 || true
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

    # Wf-A269 wave-10 (2026-11-02) - REPAIR: wave C never called run_wave
    # (it only enumerates its 8 sub-waves), so it was the ONE wave with NO
    # budget gate at all — and it is the wave that does not fit: the G6
    # yaml projects 8.6 hr across those 8 sub-waves against this wave's
    # 7.5 hr cap.  The gate is advisory, so it warns instead of blocking,
    # but the number is now measured instead of asserted in a log line.
    budget_gate "${WAVE_C_BUDGET_HR}" "wave_c"

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
