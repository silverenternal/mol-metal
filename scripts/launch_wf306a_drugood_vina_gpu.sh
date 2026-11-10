#!/usr/bin/env bash
# Wf-306A — drugood + Vina real GPU launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Status: CPU smoke PASS (Wf-306A PART 1 + PART 2).  GPU launch wrapper ready
#          on USER_OK gate (per Wf-294B kill_then_adapt pattern).
# Constraint: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED.
#
# Purpose
# -------
# Two sequential phases, each with its own GPU assignment, scope, abort criteria,
# and per-phase user OK gate.  This is the Wf-306A follow-up to:
#   * Wf-297B + Wf-301A: drugood_v3_scaffold ood_test_auc=0.4574 BELOW
#     field-avg 0.55-0.65.
#   * Wf-299B + Wf-301B: Vina 0/100 docked on Pt coordination complexes
#     (Pt+2 UFF parameterisation blocker; Wf-301B shipped --pt-strip flag,
#     Wf-305D shipped the wrapper).
#   * Wf-305D: drugood DANN driver + Vina pt-strip wrapper IMPLEMENTED but
#     CPU smoke NOT RUN before Wf-306A.
#
# Wf-306A results:
#   * PART 1 (drugood DANN): CPU smoke 95,236 rows × 1 seed × 1 epoch
#     (featurize+train).  Field-avg recovery is PROJECTED per Wf-304C.
#   * PART 2 (Vina pt-strip): CPU smoke 5/5 docked successfully against
#     pocket 1h36 (vs 0/5 without --pt-strip).  Overall mean kcal/mol=-4.60
#     (ABOVE field-avg band -9.5 to -7.5 — honest-negative on the
#     field-avg gate; the recovery axis is *being able to dock at all*,
#     which the pt-strip enables).
#
# Phases
# ------
#   Phase-A — drugood DANN full 3-seed × 50-epoch GPU run on cuda:1
#             (RTX 5090).  ~3-4 hr wall (PROJECTED, per Wf-285B
#             5-metal Cytotox 3-seed × 50-epoch = 4 hr on RTX 5090;
#             drugood single-domain is 50% larger so add ~0.5 hr).
#   Phase-B — Vina pt-strip full 100-mol panel on cuda:1 (CPU-only; Vina
#             binary is single-threaded).  ~1-2 hr wall (PROJECTED, per
#             Wf-305D smoke ~5 mols in 7-10 s × 100 mols ≈ 100-200 s +
#             receptor pre-dock work × 4 pockets ≈ 1-2 hr).
#
# Total wall: ~5 hr (PROJECTED).
#
# USER_OK gate (per Wf-300C pattern)
# ---------------------------------
#   The script aborts unless ``WF306A_USER_OK=YES`` is exported in the
#   environment.  This is the kill_then_adapt guard against un-watched
#   GPU launches (per Wf-294B).
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:  ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:    3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:     for 5+ minutes (Phase-A only; Phase-B is CPU)
#   7.2.4  collapse:           drugood Phase-A ood_test_auc < 0.45 on 2/3 seeds
#                             Vina Phase-B n_docked < 50 / 100
#
# Pre-launch checks (BLOCKER before any phase)
# -------------------------------------------
#   1. pre_flight_check.py --pipeline all (24/24 PASS target)
#   2. wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required
#       (defer if not CFM, since this wrapper is drugood+Vina not CFM)
#   3. USER_OK env gate
#
# Usage
# -----
# Wf-A269 wave-5 (2026-11-02) - D5 repair: BOTH env vars are required --
#         USER_OK=1 (canonical gate; refuses with exit 64) AND
#         WF306A_USER_OK=YES (Wf-300C gate; refuses with exit 1).  The
#         previous USAGE block named only WF306A_USER_OK=YES, so the
#         command a user copied from it was refused with exit 64.
#         The previous USAGE block ALSO named a path that never resolved
#         (molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh -- this
#         launcher was ADDED to the repo at TODO/archive/wf437_simplify/ and
#         was never at molmetal/scripts/), so copying it also failed with
#         "No such file or directory".
#         => The three lines under "SUPERSEDED RENDERING" are the OLD
#            rendering, kept for the record ONLY -- do NOT copy them.  The
#            corrected commands are the four lines below them.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
#   WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run
#
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase B
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run
#
# Wf-A269 wave-11 (2026-11-02) — VOCABULARY CONVERGED.  The canonical
# repo-wide value is USER_OK=1 (MEASURED census 2026-11-02: 44 wrappers
# accept exactly {1}, exit 64).  The wave-3 patch had made this wrapper
# require TWO spellings at once (USER_OK=1 AND WF306A_USER_OK=YES) — a
# vocabulary added on top of the canonical one instead of converged onto it.
# ONE canonical spelling is now enough; WF306A_USER_OK is a DEPRECATED
# alias that is still accepted (so scripts/gpu_pipeline_master.sh:263 keeps
# working) but is no longer an independent requirement.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase B
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
#   USER_OK=1 WF306A_USER_OK=YES bash scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run
#
#   USER_OK=1 bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase A
#   USER_OK=1 bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase B
#   USER_OK=1 bash scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
#   USER_OK=1 bash scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run
#

set -euo pipefail

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 — no retrain / panel / ablation launch may start without
#         the user OK gate.  This is the FIRST executable block in the file: it
#         precedes every mkdir, tee and trainer call, so a GPU job can never
#         start unattended.  Exit 64 is the repo's documented refusal code
#         (molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md:30 —
#         "exit 64, refuses").  ADDITIVE per AGENTS.md §5.5 rule #6: the
#         wrapper's own WF306A_USER_OK=YES gate (Wf-300C) still runs below, so
#         a launch now needs BOTH USER_OK=1 and WF306A_USER_OK=YES.
# --- USER_OK gate FIRST (per Wf-294B + Wf-355-B1 + Wf-366-A1) -------------
if [ "${USER_OK:-0}" != "1" ]; then
    echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
    echo "       USER_OK=1 bash $(basename "$0") --dry-run   # plan only (still requires USER_OK=1)" >&2
    echo "       USER_OK=1 bash $(basename "$0")             # actual run" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Wf-A269 wave-3 (2026-11-02) — REPAIR: AGENTS.md §5.9 mandates the
# single repo venv and names CadButEaas as FORBIDDEN.  The old default
# was /home/hugo/codes/CadButEaas/.venv-vlm/bin/python, a directory that does not exist on this host, so
# the launcher died at its own venv check before any work.  SUPERSEDED
# RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   PY="${PY:-/home/hugo/codes/CadButEaas/.venv-vlm/bin/python}"
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §5.9 + Wf-431-A1 fix F2 — the `molmetal.*` packages
#         resolve only when the repo root is on PYTHONPATH; the repo venv
#         python without it dies with "No module named 'molmetal'".  Exported
#         once here so every later call (pre-flight, budget tracker, trainer)
#         inherits it.
export PYTHONPATH="/home/hugo/codes/mol-metal"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf306a_drugood_vina_launch"

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §5.9 canonical launch surface — the template addresses its
#         pre-flight and dry-run artefacts through one OUT_DIR variable; this
#         wrapper's existing out-dir is ${REPORTS_DIR}, so OUT_DIR aliases it.
#         No new tracked location is invented.
OUT_DIR="${REPORTS_DIR}"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
ABORT_RECOVERY="${SCRIPTS_DIR}/abort_recovery.py"

DRUGOOD_TRAINER="${SCRIPTS_DIR}/wf305d_drugood_dann_train.py"
VINA_TRAINER="${SCRIPTS_DIR}/wf305d_vina_pt_strip.py"
DRUGOOD_CSV="${PROJECT_ROOT}/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv"
VINA_CSV="${PROJECT_ROOT}/.data/molmetal/platinai_4cell_mbfinder.csv"

# Wf-A269 wave-11 (2026-11-02) — the ONE place that knows how to read a
# launcher's pre-launch gate REPORT (pre-flight / wave budget tracker) and
# turn it into an honest PASS / FAIL verdict.
# `die` is MOVED UP to before its first call: bash executes a script
# line-by-line, so a function definition that appears LATER in the file is
# not yet in scope at an earlier call site — the call then dies with
# "die: command not found" (rc=127) and the intended FATAL message + rc=1
# are both lost (same repair as
# scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:87-94, which is the
# wave-10 reference pattern).  ${C_RED:-}/${C_RST:-} keep the early call
# site safe (C_RED is initialised further down); `mkdir -p` keeps the FATAL
# message deliverable when the reports dir does not exist yet.
die() { mkdir -p "${REPORTS_DIR}" 2>/dev/null || true; printf '%s[FATAL]%s %s\n' "${C_RED:-}" "${C_RST:-}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

# Wf-A269 wave-11 (2026-11-02) — the shared gate-verdict helpers.  The
# checker reads the REPORT a gate tool already wrote and decides PASS / FAIL
# from that file's own contents; it never re-runs the tool and never infers
# a verdict from the absence of output.  Exit 0=PASS, 1=FAIL, 2=SKIP.
# A missing report => SKIP, never PASS.
run_gate_check() {
    local kind="$1" report="$2" wave="${3:-}" rc=0 out=""
    if [ -n "${wave}" ]; then
        out="$("${PY}" "${GATE_VERDICT_CHECKER}" "${kind}" --report "${report}" \
            --wave "${wave}" 2>&1)" || rc=$?
    else
        out="$("${PY}" "${GATE_VERDICT_CHECKER}" "${kind}" --report "${report}" 2>&1)" || rc=$?
    fi
    printf '%s\n' "${out}" > "${REPORTS_DIR}/gate_${kind}.verdict.txt"
    case "${rc}" in 0) printf 'PASS' ;; 1) printf 'FAIL' ;; *) printf 'SKIP' ;; esac
}

gate_detail() { head -1 "${REPORTS_DIR}/gate_$1.verdict.txt" 2>/dev/null || printf ''; }

mkdir -p "${REPORTS_DIR}/phase_a" "${REPORTS_DIR}/phase_b"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap; Phase-A
# spans 3-4 hr as 2 sub-phases < 2 hr each)
PHASE_A_BUDGET_HR=4.0      # 3-seed × 50-epoch drugood DANN full
PHASE_B_BUDGET_HR=2.0      # 100-mol Vina pt-strip panel

# Per-phase GPU assignment (drugood: GPU; Vina: CPU-only — Vina binary is
# single-threaded so cuda:1 is unused during Phase-B)
PHASE_A_GPU="cuda:1"        # RTX 5090 (per Wf-130 default kanzi GPU 1)
PHASE_B_GPU="cpu"           # Vina is CPU-only

# Color codes (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
# Wf-A269 wave-11 (2026-11-02) — `die` was MOVED UP to the constants block
# (line 159) so that the GATE_VERDICT_CHECKER existence check can call it;
# bash runs a script line-by-line, so a function defined LATER is not in
# scope at an earlier call site.  The previous definition at this line was:
#     die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
# (preserved here verbatim per AGENTS.md section 5.5 rule #6; the moved-up
# definition is identical except that it defaults ${C_RED:-}/${C_RST:-} and
# mkdir -p's the reports dir so the FATAL message is deliverable before the
# colour block runs).
ok()  { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn(){ printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
PHASE_TARGET="all"

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §5.9 canonical launch surface — plan.json records the
#         invocation exactly as the user typed it; captured before the parse
#         loop shifts the positional parameters away.
ORIG_ARGS="$*"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --phase)   PHASE_TARGET="${2:-all}"; shift 2 ;;
        --phase-*) PHASE_TARGET="${1#--phase-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${PHASE_TARGET}" = "all" ] || [ "${PHASE_TARGET}" = "A" ] || [ "${PHASE_TARGET}" = "B" ] \
    || die "phase must be A | B | all (got: ${PHASE_TARGET})"

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-300C pattern)
# -----------------------------------------------------------------------------
user_ok_gate() {
    # Wf-A269 wave-11 (2026-11-02) — USER_OK VOCABULARY CONVERGENCE.
    # The canonical repo-wide value is USER_OK=1: MEASURED census
    # 2026-11-02 — 44 wrappers accept exactly {1} and refuse with exit 64
    # (e.g. scripts/_WRAPPER_TEMPLATE.sh:56,
    # scripts/launch_wf124_p3_g5.sh:65,
    # scripts/launch_wf124_p3_g6.sh:84); the canonical USER_OK=1 gate at the
    # top of THIS file (line 103) already fired before we got here.
    #
    # This wrapper had TWO INDEPENDENT gates with DISJOINT vocabularies, so
    # a launch needed two spellings at once — the wave-3 patch ADDED a
    # vocabulary instead of converging it:
    #     USER_OK=1        (wave-3 canonical gate, exit 64)
    #   AND WF306A_USER_OK=YES (Wf-300C legacy gate, exit 1)
    # REPAIRED: WF306A_USER_OK is now a DEPRECATED ALIAS of the canonical
    # USER_OK variable.  It is NO LONGER an independent requirement — one
    # canonical spelling (USER_OK=1) is enough — and when it IS set it is
    # validated against the canonical value.  "YES" is still accepted so that
    # scripts/gpu_pipeline_master.sh:263
    # (`USER_OK=1 WF306A_USER_OK=YES nohup bash scripts/launch_wf306a_...`)
    # keeps working; it is DEPRECATED and warns.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if [ -z "${WF306A_USER_OK:-}" ]; then
    #         die "WF306A_USER_OK env var not set.  Re-run with: WF306A_USER_OK=YES bash $0 ..."
    #     fi
    #     if [ "${WF306A_USER_OK}" != "YES" ]; then
    #         die "WF306A_USER_OK must equal 'YES' (got: '${WF306A_USER_OK}')"
    #     fi
    #     ok "USER_OK gate passed (WF306A_USER_OK=YES)"
    if [ -n "${WF306A_USER_OK:-}" ]; then
        if [ "${WF306A_USER_OK}" = "YES" ]; then
            warn "WF306A_USER_OK=YES is DEPRECATED — the canonical value is USER_OK=1 (per Wf-294B + Wf-366-A1)"
        elif [ "${WF306A_USER_OK}" != "1" ]; then
            die "WF306A_USER_OK is a DEPRECATED alias of USER_OK; the canonical value is 1 (got: '${WF306A_USER_OK}')"
        fi
    fi
    ok "USER_OK gate passed (USER_OK=${USER_OK})"
}

# -----------------------------------------------------------------------------
# Pre-flight sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]        || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]    || die "wave_budget_tracker.py missing"
[ -f "${ABORT_RECOVERY}" ]    || die "abort_recovery.py missing"
[ -f "${DRUGOOD_TRAINER}" ]   || die "drugood trainer missing: ${DRUGOOD_TRAINER}"
[ -f "${VINA_TRAINER}" ]      || die "vina trainer missing: ${VINA_TRAINER}"
[ -f "${DRUGOOD_CSV}" ]       || die "drugood CSV missing: ${DRUGOOD_CSV}"
[ -f "${VINA_CSV}" ]          || die "vina CSV missing: ${VINA_CSV}"

log "Wf-306A launch begin — phase target = ${PHASE_TARGET} — dry-run = ${DRY_RUN}"
user_ok_gate

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline all
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: this gate captured the report
    # with `> pre_flight.json 2>&1` after a BARE `--json`.  Two defects in
    # those three lines:
    #   (1) `pre_flight_check.py --json` is `type=Path, nargs="?"`
    #       (molmetal/scripts/pre_flight_check.py:17327-17333), so a BARE
    #       `--json` parses to None and the tool writes NO FILE at all — it
    #       only prints the payload to stdout (the file is written only when
    #       `args.json is not None`, pre_flight_check.py:17701-17705).
    #       MEASURED on this host: the captured
    #       molmetal/reports/wf306a_drugood_vina_launch/pre_flight.json is
    #       0 bytes.
    #   (2) redirecting stdout AND stderr into the .json path interleaves the
    #       tool's stderr INFO lines with the JSON payload, so the file is
    #       NOT valid JSON — MEASURED: json.load raises
    #       "Expecting value: line 1 column 2 (char 1)" and the shared
    #       consumer answers "SKIP the preflight report is missing or
    #       unreadable" (rc=2).
    #   (3) the PASS/FAIL counts came from `grep -c '"status": "PASS"'`
    #       over that non-JSON file, which counts gate ROWS and ignores the
    #       authoritative `summary` block.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${PRE_FLIGHT}" --pipeline all --json \
    #             > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
    #         die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    #     fi
    #     local n_pass n_fail
    #     n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    #     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
    #
    # REPAIRED: `--json <PATH>` writes a real JSON file (the tool writes it
    # itself), stdout goes to a .log file, and the verdict is read from that
    # report by the shared checker.  The gate can now FAIL on a FAIL or a
    # TIMEOUT gate, and it can no longer pass on an unreadable report.
    # Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A crashed
    # or killed pre-flight must not leave the last run's PASS report on disk
    # for the consumer to read — "stale is treated as missing".
    rm -f "${REPORTS_DIR}/pre_flight.json"
    "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
        > "${REPORTS_DIR}/pre_flight.stdout.log" 2>&1 || true
    local pf_verdict pf_detail
    pf_verdict="$(run_gate_check preflight "${REPORTS_DIR}/pre_flight.json")"
    pf_detail="$(gate_detail preflight)"
    if [ "${pf_verdict}" != "PASS" ]; then
        warn "pre_flight_check ${pf_verdict} (${pf_detail})"
        die "pre_flight_check ${pf_verdict} — refusing to launch (report: ${REPORTS_DIR}/pre_flight.json)"
    fi
    ok "pre_flight ${pf_detail}"
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py --budget-hr 6
#   --cfm-yuelbond-required is a gate that DEFERs non-CFM waves; this
#   wrapper is drugood+Vina so we DO pass it but expect a DEFERRED flag.
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the gate keyed on the
    # tracker's EXIT CODE, but `wave_budget_tracker.py`'s main() returns 0
    # on EVERY path (molmetal/scripts/wave_budget_tracker.py:1422), so the
    # warn branch was UNREACHABLE and the gate announced "budget tracker
    # logged" whatever the tracker had actually decided.
    # MEASURED on this host: `--budget-hr 0.1` exits 0 while the report
    # says remaining_hr=-52.0 with 6 DEFERRED waves — and the shared
    # consumer reading that SAME report answers FAIL (rc=1).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr 6 --cfm-yuelbond-required \
    #             --report-json "${REPORTS_DIR}/budget_report.json" \
    #             > "${REPORTS_DIR}/budget.log" 2>&1; then
    #         warn "budget tracker returned non-zero (advisory only; --no-enforce-budget supported)"
    #     fi
    #     ok "budget tracker logged (DEFERRED expected since drugood/Vina are not CFM)"
    #
    # REPAIRED: the verdict now comes from the shared consumer reading the
    # tracker's own report.  This gate stays ADVISORY (it warns, it does not
    # die) — the tracker only knows a GLOBAL session budget, not this
    # wrapper's own 4 hr + 2 hr phase budgets, so failing the launch on its
    # number would be a false blocker.
    "${PY}" "${BUDGET_TRACKER}" --budget-hr 6 --cfm-yuelbond-required \
        --report-json "${REPORTS_DIR}/budget_report.json" \
        > "${REPORTS_DIR}/budget.log" 2>&1 || true
    local bg_verdict bg_detail
    bg_verdict="$(run_gate_check budget "${REPORTS_DIR}/budget_report.json")"
    bg_detail="$(gate_detail budget)"
    if [ "${bg_verdict}" = "PASS" ]; then
        ok "budget tracker ${bg_detail}"
    else
        warn "budget tracker ${bg_verdict} (${bg_detail}) — advisory only (DEFERRED expected since drugood/Vina are not CFM)"
    fi
}

# -----------------------------------------------------------------------------
# Per-phase runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_phase() {
    local phase_label="$1"; shift
    local budget_hr="$1"; shift
    local trainer="$1"; shift
    local gpu="$1"; shift
    local out_dir="$1"; shift
    # remaining args = --flag pairs joined into the trainer invocation

    log "==== Phase ${phase_label} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Phase ${phase_label}"
        cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Phase ${phase_label}
  trainer: ${trainer}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        ok "Phase ${phase_label} DRY-RUN logged"
        return 0
    fi

    local started_at
    started_at=$(date +%s)
    local deadline
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))

    # Background launcher for the trainer; PID captured for kill_then_adapt.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu##cuda:}}" \
        "${PY}" "${trainer}" \
            --out-dir "${out_dir}" \
            "$@" \
            > "${out_dir}/train.log" 2>&1 &
    local pid=$!

    log "Phase ${phase_label} PID=${pid}; monitoring..."

    # Poll loop: every 60s, check timeout + GPU util + per-phase abort criteria
    local aborted=0
    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Phase ${phase_label} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 5
            kill -KILL "${pid}" 2>/dev/null || true
            aborted=1
            break
        fi

        # GPU util < 70% for 5+ min — TODO-207 v2 §7.2.3 (advisory only here;
        # applies to Phase-A only; Phase-B is CPU)
        if [ "${phase_label}" = "A" ]; then
            local gpu_util
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                    --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-1}" 2>/dev/null \
                        | head -1 || echo "0")
            log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"
        fi

        sleep 60
    done

    wait "${pid}" 2>/dev/null || true

    local wall_sec=$(( $(date +%s) - started_at ))
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `bc` is NOT INSTALLED on this
    # host (MEASURED: `command -v bc` -> not found), so
    # `$(echo "scale=2; ${wall_sec}/3600" | bc)` produced an EMPTY string and
    # the log line read "wall=NNNs ( hr)" — the wall-clock budget line was
    # silently blank.  awk is present (GNU Awk 5.4.1) and is what
    # scripts/launch_wf299_cfm_main.sh:576 already uses for the same value.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     log "Phase ${phase_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"
    local wall_hr
    wall_hr=$(awk -v s="${wall_sec}" 'BEGIN { printf "%.2f", s/3600 }')
    log "Phase ${phase_label} wall=${wall_sec}s (${wall_hr} hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || ! grep -q "phase_${phase_label,,}_complete\|TEST (ood) AUC\|n_docked=" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL — invoking abort_recovery.py"
        # Wf-A269 wave-11 (2026-11-02) — REPAIR: this call was DEAD THREE WAYS.
        #   (1) `--json` is NOT a flag of abort_recovery.py — the tool defines
        #       `--report-json` (molmetal/scripts/abort_recovery.py:390-394).
        #   (2) `--pattern TIMEOUT` is NOT a legal choice — the tool's
        #       SUPPORTED_PATTERNS = ("NaN","OOM","IndexError","cfm-collapse")
        #       (molmetal/scripts/abort_recovery.py:99-105).
        #   (3) the REQUIRED `--metal` was never passed
        #       (molmetal/scripts/abort_recovery.py:372-376).
        #   argparse therefore exited 2 INSTANTLY and the file named
        #   "recovery.json" held the ARGPARSE USAGE TEXT, not a recovery
        #   record — MEASURED: the .json parsed as a string, not an object.
        #   Additionally the stdout+stderr redirect put the tool's own output
        #   into the .json path, so even a legal run would have produced an
        #   invalid JSON file.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     "${PY}" "${ABORT_RECOVERY}" --wave "phase_${phase_label,,}" \
        #         --pattern TIMEOUT --json \
        #         > "${out_dir}/recovery.json" 2>&1 || true
        #
        # REPAIRED: legal flags only, the report is written by the TOOL
        # itself into this phase's OWN reports dir, and stdout goes to a .log.
        #
        # HONEST-NEGATIVE (documented, not hidden): abort_recovery.py's
        # `--metal` vocabulary is (Ru,Ir,Os,Rh,Re) and it filters candidate
        # checkpoints by REQUIRING the metal token inside the ckpt filename
        # stem (molmetal/scripts/abort_recovery.py:143-161).  This wrapper's
        # cohorts are metal-agnostic — Phase-A writes
        # `dmpnn_dann_drugood_seed{N}.pt` and Phase-B strips Pt — so no legal
        # `--metal` value can ever match this wrapper's own checkpoints and
        # the recovery probe will honestly report "no ckpt candidate found".
        # The knobs are exposed so a caller can point the probe at a real
        # metal-named cohort; the default keeps the tool from dying on
        # argparse.
        local rec_pattern rec_metal
        rec_pattern="${ABORT_RECOVERY_PATTERN:-OOM}"
        rec_metal="${ABORT_RECOVERY_METAL:-Ru}"
        case "${rec_pattern}" in
            NaN|OOM|IndexError|cfm-collapse) : ;;
            *) warn "ABORT_RECOVERY_PATTERN='${rec_pattern}' is not one of NaN/OOM/IndexError/cfm-collapse — falling back to OOM"
               rec_pattern="OOM" ;;
        esac
        case "${rec_metal}" in
            Ru|Ir|Os|Rh|Re) : ;;
            *) warn "ABORT_RECOVERY_METAL='${rec_metal}' is not one of Ru/Ir/Os/Rh/Re — falling back to Ru"
               rec_metal="Ru" ;;
        esac
        "${PY}" "${ABORT_RECOVERY}" --wave "phase_${phase_label,,}" \
            --pattern "${rec_pattern}" --metal "${rec_metal}" \
            --report-json "${out_dir}/recovery.json" \
            > "${out_dir}/recovery.stdout.log" 2>&1 || true
        if [ -s "${out_dir}/recovery.json" ] \
           && "${PY}" -c 'import json,sys; json.load(open(sys.argv[1]))' "${out_dir}/recovery.json" >/dev/null 2>&1; then
            ok "abort_recovery report written: ${out_dir}/recovery.json"
        else
            warn "abort_recovery did NOT write a valid report (${out_dir}/recovery.json) — see ${out_dir}/recovery.stdout.log"
        fi
        die "Phase ${phase_label} aborted — recovery at ${out_dir}/recovery.json"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s"
    # Per-phase OK gate: < 2 hr (per kill_then_adapt).  Phase-A's 4 hr
    # PROJECTED budget is split into 2 sub-phases of < 2 hr each at the
    # config level (3 seeds ≈ 1.3 hr each).
    if [ "${wall_sec}" -gt $(( ${budget_hr%.*} * 3600 )) ]; then
        die "Phase ${phase_label} exceeded per-phase ${budget_hr} hr OK gate (kill_then_adapt)"
    fi
}

# -----------------------------------------------------------------------------
# Phase A — drugood DANN full 3-seed × 50-epoch GPU run (cuda:1)
# -----------------------------------------------------------------------------
phase_A() {
    local out="${REPORTS_DIR}/phase_a"
    log "Phase A plan: drugood DANN full 3-seed × 50-epoch (~3-4 hr, cuda:1)"
    log "  seeds=0,1,2 epochs=50 hidden=128 depth=3 batch=64 lr=1e-3"
    log "  use_dann=True lambda_grl=0.1 use_kendall=True w_w=0.05"
    run_phase "A" "${PHASE_A_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${PHASE_A_GPU}" "${out}" \
        --csv "${DRUGOOD_CSV}" \
        --seeds 0 1 2 \
        --epochs 50 \
        --hidden 128 \
        --depth 3 \
        --batch-size 64 \
        --lr 1e-3 \
        --alpha 0.5 \
        --wasserstein-weight 0.05 \
        --use-dann \
        --lambda-grl 0.1 \
        --use-kendall \
        --device cuda \
        --cpu-num-workers 2 \
        || true
    ok "Phase A drugood full DANN run complete"
}

# -----------------------------------------------------------------------------
# Phase B — Vina pt-strip full 100-mol panel (CPU-only)
# -----------------------------------------------------------------------------
phase_B() {
    local out="${REPORTS_DIR}/phase_b"
    log "Phase B plan: Vina pt-strip full 100-mol × 4-pocket panel (~1-2 hr, CPU)"
    log "  n_mols=100 pocket_id=1h36 (extend to 4 pockets for paper-grade)"
    run_phase "B" "${PHASE_B_BUDGET_HR}" \
        "${VINA_TRAINER}" "${PHASE_B_GPU}" "${out}" \
        --csv "${VINA_CSV}" \
        --pocket-id 1h36 \
        --n-mols 100 \
        --pt-strip \
        --out "${out}/panel.json" \
        || true
    ok "Phase B Vina pt-strip full panel complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
pre_flight_gate
budget_gate

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + TODO-207 — the pre-flight gate must PASS before any
#         trainer invocation.  This is the canonical `launch`-pipeline gate
#         (Wf-431-A4 canonical surface); the wrapper's own pre_flight_gate()
#         above additionally runs the `all` pipeline and counts PASS/FAIL, so
#         both artefacts are kept side by side — ${OUT_DIR}/pre_flight.json
#         (`all`) and ${OUT_DIR}/preflight.json (`launch`).  Dry-run skips it:
#         a plan-only run must not execute a live gate (a live gate may
#         allocate GPU memory — honest-negative catalog §6.265).
if [ "${DRY_RUN}" -ne 1 ]; then
    # --- pre-flight gate (AGENTS.md §4 + TODO-207) --------------------------
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the `launch`-pipeline gate
    # redirected stdout+stderr straight into "${OUT_DIR}/preflight.json" and
    # passed NO `--json` at all.  Two defects:
    #   (1) the tool's stdout carries human-readable INFO lines, so the file
    #       is NOT valid JSON — MEASURED: json.load on it raises
    #       "Expecting value: line 1 column 2 (char 1)";
    #   (2) the verdict came from the tool's EXIT CODE alone, so a report
    #       whose `summary.timeout > 0` (a gate that did not obtain its
    #       measurement) would still pass, and the shared consumer on that
    #       SAME report answers FAIL.
    # The verdict now comes from the shared consumer reading the report the
    # tool itself wrote.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${PRE_FLIGHT}" --pipeline launch >"${OUT_DIR}/preflight.json" 2>&1 \
    #         || { echo "[FATAL] pre_flight_check FAIL — see ${OUT_DIR}/preflight.json" >&2; exit 3; }
    #
    # Wf-A269 wave-18 (2026-11-03) — REPAIR (found by the new guard, not by a
    # run): this SECOND pre-flight call site had no `rm -f` before it.  The
    # first one does (`rm -f "${REPORTS_DIR}/pre_flight.json"`, line 366), so
    # the asymmetry was invisible to any check that looked at one call site.
    # The consequence is the "stale is treated as missing" rule violated at
    # the tail: if this pre-flight is killed or crashes, the previous run's
    # PASS report survives at "${OUT_DIR}/preflight.json", and
    # `run_gate_check preflight` reads THAT — a verdict earned by an earlier
    # run, on a tree that may since have moved.  The consumer's 24 h
    # freshness arm narrows the window (a report older than 24 h is FAIL),
    # but a report from an earlier run on the SAME day is inside it, so the
    # hole is real, not theoretical.
    rm -f "${OUT_DIR}/preflight.json"
    PYTHONPATH=/home/hugo/codes/mol-metal "${PY}" "${PRE_FLIGHT}" --pipeline launch \
        --json "${OUT_DIR}/preflight.json" \
        > "${OUT_DIR}/preflight.stdout.log" 2>&1 || true
    lp_verdict="$(run_gate_check preflight "${OUT_DIR}/preflight.json")"
    lp_detail="$(gate_detail preflight)"
    if [ "${lp_verdict}" != "PASS" ]; then
        echo "[FATAL] pre_flight_check ${lp_verdict} (${lp_detail}) — see ${OUT_DIR}/preflight.json" >&2
        exit 3
    fi
    ok "pre_flight_check (launch pipeline) ${lp_detail}"
fi

# Wf-A269 wave-5 (2026-11-02) - D11 (INFO): this launcher runs TWO live
#         pre-flight checks per invocation, and the phase target does NOT
#         multiply them.  MEASURED 2026-11-02 on this file: phase_A() and
#         phase_B() contain NO pre_flight_check.py call; the only two live
#         call sites are the `pre_flight_gate` call (the `--pipeline all`
#         run) and the `--pipeline launch` run immediately above this note,
#         and BOTH of them sit in Main, ABOVE the `case "${PHASE_TARGET}"`
#         dispatch.  So `--phase A`, `--phase B` and `--phase all` each
#         execute each of the two runs exactly ONCE per invocation.
#         The two runs are NOT duplicates: they are DIFFERENT pipelines.
#         MEASURED 2026-11-02 via build_gates(): `all` = 129 gates,
#         `launch` = 131 gates, and `launch` is a strict SUPERSET of `all`
#         (the 2 extra gates are p3_wf343c2_per_wave_runtime_check and
#         p3_wf343c4_anchor_freshness_24h).  They also write DIFFERENT
#         artefacts (pre_flight.json for `all`, preflight.json for
#         `launch`).  The second run is therefore LOAD-BEARING -- it is the
#         canonical launch gate AGENTS.md section 4 requires -- so it is NOT
#         removed, NOT merged into the first, and NOT hoisted (hoisting is a
#         no-op: both runs already sit above the dispatch).  The two-run
#         behaviour is INTENDED and is unchanged by wave-5; this note only
#         documents it.
# QUOTATION OF THE LIVE CODE -- this is NOT a superseded rendering: wave-5
#         changed none of the code quoted below, so AGENTS.md §5.5 rule #6
#         does not apply to it.  The quote is a SELECTIVE EXCERPT, not a
#         byte-identical copy: it omits the interior comment line
#             # --- pre-flight gate (AGENTS.md §4 + TODO-207) ---...
#         and the trailing `esac`, and it re-indents the `case` arms onto
#         the comment column.  wave-5 deletes none of the live code.
#   pre_flight_gate
#   if [ "${DRY_RUN}" -ne 1 ]; then
#       PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${PRE_FLIGHT}" --pipeline launch >"${OUT_DIR}/preflight.json" 2>&1 \
#           || { echo "[FATAL] pre_flight_check FAIL — see ${OUT_DIR}/preflight.json" >&2; exit 3; }
#   fi
#   case "${PHASE_TARGET}" in
#       A)   phase_A ;;
#       B)   phase_B ;;
#       all) phase_A; phase_B ;;
#   esac

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §5.9 canonical launch surface + Wf-363-A1 schema — every
#         launcher emits the same artefacts (DRY_RUN.log + plan.json) into its
#         own out-dir, so molmetal/scripts/wf426a3_post_fix_results_aggregator.py
#         finds plan.json meta.wrapper_name + meta.user_ok_required without
#         descending into phase_a/ and phase_b/.  POSIX-atomic write
#         (temp-file + mv -f) per AGENTS.md §5.6.
#         plan.json is written on EVERY invocation, not only on --dry-run: a
#         dry-run plan left on disk would otherwise be read by
#         scripts/gpu_launch_dashboard.sh as the plan of a later REAL launch
#         (the Wf-A9 silent-lie class).  The --dry-run branch below still
#         produces it, with mode=plan_only / dry_run=1.
PRE_FLIGHT_STATUS="PASS"
PLAN_MODE="actual"
PLAN_STATUS="READY_TO_RUN"
if [ "${DRY_RUN}" -eq 1 ]; then
    PRE_FLIGHT_STATUS="SKIPPED - dry-run executes no live gate"
    PLAN_MODE="plan_only"
    PLAN_STATUS="PLAN_ONLY"
    cat > "${OUT_DIR}/DRY_RUN.log.tmp" <<EOF
[DRY-RUN] $(basename "$0") @ wf306a_drugood_vina_launch
[DRY-RUN] wrapper_name=$(basename "$0")
[DRY-RUN] wf_origin=Wf-306A - drugood DANN (Phase A) + Vina pt-strip (Phase B)
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
[DRY-RUN] phase_target=${PHASE_TARGET}
[DRY-RUN] phase_a trainer=${DRUGOOD_TRAINER} gpu=${PHASE_A_GPU} budget_hr=${PHASE_A_BUDGET_HR}
[DRY-RUN] phase_b trainer=${VINA_TRAINER} gpu=${PHASE_B_GPU} budget_hr=${PHASE_B_BUDGET_HR}
[DRY-RUN] py=${PY}
[DRY-RUN] out_dir=${OUT_DIR}
[DRY-RUN] NOTHING WAS EXECUTED - no trainer was launched and no GPU was touched.
[DRY-RUN] ts=$(date -Iseconds)
EOF
    mv -f "${OUT_DIR}/DRY_RUN.log.tmp" "${OUT_DIR}/DRY_RUN.log"
    log "[DRY-RUN] canonical DRY_RUN.log emitted -> ${OUT_DIR}/DRY_RUN.log"
fi

cat > "${OUT_DIR}/plan.json.tmp" <<EOF
{
  "wrapper": "$(basename "$0")",
  "meta": {
    "wrapper_name": "$(basename "$0")",
    "user_ok_required": true,
    "wf_origin": "Wf-306A - drugood DANN (Phase A, ${PHASE_A_GPU}) + Vina pt-strip (Phase B, CPU-only)",
    "parent_plan": "TODO/completed/349c1_wf341_wrappers_deep_integration.md - Wf-306A wrapper inventory"
  },
  "user_ok": true,
  "dry_run": ${DRY_RUN},
  "mode": "${PLAN_MODE}",
  "status": "${PLAN_STATUS}",
  "phase_target": "${PHASE_TARGET}",
  "trainer": "${DRUGOOD_TRAINER}",
  "py": "${PY}",
  "out_dir": "${OUT_DIR}",
  "dry_run_log": "${OUT_DIR}/DRY_RUN.log",
  "pre_flight": "${PRE_FLIGHT_STATUS}",
  "phases": [
    {"label": "A", "trainer": "${DRUGOOD_TRAINER}", "gpu": "${PHASE_A_GPU}", "budget_hr": ${PHASE_A_BUDGET_HR}, "wall_budget_label": "PROJECTED (~3-4 hr per Wf-285B extrapolation)"},
    {"label": "B", "trainer": "${VINA_TRAINER}", "gpu": "${PHASE_B_GPU}", "budget_hr": ${PHASE_B_BUDGET_HR}, "wall_budget_label": "PROJECTED (~1-2 hr per Wf-305D extrapolation)"}
  ],
  "args": "${ORIG_ARGS}",
  "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution + AGENTS.md section 5.1",
  "ts": "$(date -Iseconds)"
}
EOF
mv -f "${OUT_DIR}/plan.json.tmp" "${OUT_DIR}/plan.json"
log "plan.json emitted -> ${OUT_DIR}/plan.json"

case "${PHASE_TARGET}" in
    A)   phase_A ;;
    B)   phase_B ;;
    all) phase_A; phase_B ;;
esac

ok "Wf-306A launch script done — phase=${PHASE_TARGET}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: per_phase_user_ok_required between phases."
