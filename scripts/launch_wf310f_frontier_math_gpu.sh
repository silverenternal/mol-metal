#!/usr/bin/env bash
# Wf-310F — TODO-234 frontier-math Day 7 launch (USER_OK gate).
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Status: READY on user OK (per TODO-234 §5 + Wf-310F §1.7).
# Constraint: party-twenty-spirit §3.5 — every number MEASURED or honest-PROJECTED.
#
# Purpose
# -------
# GPU production wire-in for the 4 frontier-math adapters composed
# via ``molmetal.adapters.frontier_math_integration.FrontierMathIntegration``
# (Wf-309B ship + Wf-310F Day 1-2).  Per TODO-234 §4 + §5 the 5-7 day
# ultracode window's final day exercises the wire-in on a single GPU
# (~3-5 hr wall).
#
# What this launches
# ------------------
# Three sequential waves, each with its own GPU, scope, and abort criteria:
#   Wave A — TODO-119 P1.4 frontier-multi-ph smoke (~1.5 hr, cuda:0).
#            Wires ``augment_pocket_embedding`` into the CFM trainer
#            (``train_cfm_yuelbond_wasserstein.py``) and re-runs the P1.4
#            smoke with the augmented pocket embedding.
#   Wave B — TODO-182 G6 frontier-SE3 smoke (~0.5 hr, cuda:0).
#            Wires ``SE3LipmanSolver`` into the D-MPNN multi-task
#            trainer (``train_dmpnn_multitask_wasserstein.py``) and
#            re-runs the G6 no_wass smoke on the SE(3) manifold.
#   Wave C — TODO-234 frontier-math paper-grade sweep (~2-3 hr, cuda:0).
#            Joint ``FrontierMathIntegration`` wire-in across the full
#            R18 trainer stack — measure baseline + augmented lift
#            (PROJECTED, kill_then_adapt < 2 hr/sub-wave).
#
# kill_then_adapt semantics (per TODO-182 + Wf-295D)
# ---------------------------------------------------
#   - Each wave MUST finish < 2 hr (per-wave OK gate).
#   - On any 207 v2 §7.2 abort criterion → invoke abort_recovery.py,
#     exit 1.
#   - Between waves: stop and request per_wave_user_ok (per Wf-295D).
#
# Pre-launch checks (BLOCKER before any wave)
# -------------------------------------------
#   1. pre_flight_check.py --pipeline all --json <this wrapper's own report>;
#      the verdict is read by molmetal/scripts/wf_launch_gate_verdict.py
#      (a TIMEOUT counts as FAIL per Wf-A135).  Wf-A269 wave-11 (2026-11-02):
#      the old "target 36/36 PASS" literal was a STALE target — the `all`
#      pipeline carries 129 gates (AGENTS.md §1 rule #4) — and the verdict
#      was re-derived by grepping the report instead of being read.
#   2. Wave-specific budget tracker call.   (Wf-A269 wave-11: the header
#      PROMISED this and NO code called it — a declared-but-dead gate.)
#   3. Wave-specific per-wave checklist call. (Wf-A269 wave-11: same — the
#      header promised it, no code called it, and no
#      wave_<x>_subchecklist.json was ever written.)
#
# Per-wave OK gate (USER_OK)
# --------------------------
#   - Wall-time < 2 hr/wave.
#   - All P0 BLOCKER + P1 VERIFY gates still PASS (re-run pre_flight).
#   - Per-wave sub-checklist emit + JSON written under
#     molmetal/reports/wf310f_frontier_math_gpu/wave_<x>_subchecklist.json
#     (x = a|b|c, lower case, matching the wave_<x>/ output dirs).
#
# Usage
# -----
#   bash scripts/launch_wf310f_frontier_math_gpu.sh --wave A
#   bash scripts/launch_wf310f_frontier_math_gpu.sh --wave all
#   bash scripts/launch_wf310f_frontier_math_gpu.sh --dry-run
#
# Wf-A269 wave-19 (2026-11-04): the USAGE block above named a path this
# launcher NEVER occupied -- `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh`
# does not exist (MEASURED: 0 matches under molmetal/scripts/; the launcher is
# tracked at scripts/launch_wf310f_frontier_math_gpu.sh and at no other
# location).  The block was NEVER re-worded: `git log --all -S 'bash
# scripts/launch_wf310f'` returns NOTHING, so the original rendering is the
# dead one and it was a LIVE defect, not a superseded rendering.  Every other
# live wrapper words it `bash scripts/launch_*.sh`.
# The superseded rendering is preserved VERBATIM per AGENTS.md §5.5 rule #6:
#
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave A
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --wave all
#   bash molmetal/scripts/launch_wf310f_frontier_math_gpu.sh --dry-run
#
# Stop semantics: USER_OK required between waves.  The default ``--wave
# all`` invocation pauses between waves and waits for the caller to set
# ``WF310F_NEXT_WAVE=A|B|C`` (or pass ``--yes`` to skip the OK gate for
# dry-run / CI smoke).

set -euo pipefail

# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/hugo/codes/mol-metal}"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310f_frontier_math_gpu"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
ABORT_RECOVERY="${SCRIPTS_DIR}/abort_recovery.py"
# Wf-A269 wave-11 (2026-11-02): the two per-wave gates the header PROMISED
# but no code called.  Both are now invoked (see wave_gate()).
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
PER_WAVE_CHECKLIST="${SCRIPTS_DIR}/per_wave_checklist.py"

# Trainers — Day 3-4 + Day 5-6 wire-in points (per Wf-310F §1).
CFM_TRAINER="${SCRIPTS_DIR}/train_cfm_yuelbond_wasserstein.py"
DMPNN_TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"

# Configs — same P1.4 / G6 cfgs as Wf-299C (frontier-math just wires
# into them; it does not change the configs themselves).
P1_4_CONFIG="${PROJECT_ROOT}/configs/cfm_per_innovation_ablation_p1.4.yaml"
G6_CONFIG="${PROJECT_ROOT}/configs/wf124_p3_per_innovation_ablation_g6.yaml"

mkdir -p "${REPORTS_DIR}/wave_a" "${REPORTS_DIR}/wave_b" "${REPORTS_DIR}/wave_c"

# Per-wave budget in hours (kill_then_adapt < 2 hr/wave hard cap).
WAVE_A_BUDGET_HR=1.5
WAVE_B_BUDGET_HR=0.5
WAVE_C_BUDGET_HR=2.5

# Per-wave GPU assignment.
WAVE_A_GPU="cuda:0"      # PRO 6000 Blackwell
WAVE_B_GPU="cuda:0"      # RTX PRO 6000 Blackwell GPU-0 (serial execution)
WAVE_C_GPU="cuda:0"      # PRO 6000 Blackwell (joint sweep)

# kill_then_adapt session cap (per this file's header, line ~35: "Each wave
# MUST finish < 2 hr (per-wave OK gate)"; the constants block above repeats
# "kill_then_adapt < 2 hr/wave hard cap").
# Wf-A269 wave-11 (2026-11-02): the budget gate measures each wave against
# THIS cap, not against the wave's own WAVE_*_BUDGET_HR constant, so it is
# NOT a tautology and can actually FAIL — wave C's own 2.5 h constant does
# not fit a 2 h cap, and the gate now says so out loud instead of hiding it.
KILL_THEN_ADAPT_CAP_HR=2.0

# Color codes (only when stdout is a tty).
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()  { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn(){ printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# Wf-A269 wave-11 (2026-11-02): the ONE place that knows how to read a
# launcher's pre-launch gate REPORT (pre-flight / per-wave checklist / wave
# budget tracker) and turn it into an honest PASS / FAIL / SKIP verdict.
# MEASURED evidence: molmetal/scripts/wf_launch_gate_verdict.py:1-241 —
#   sub-commands `preflight` | `checklist` | `budget`; exit 0=PASS, 1=FAIL,
#   2=SKIP (a missing/unreadable report is SKIP, never PASS);
#   `preflight` = PASS iff summary.fail==0 AND summary.timeout==0 (a TIMEOUT
#   is a FAIL per Wf-A135); `checklist` = PASS iff no `error`,
#   preflight_fail_count==0 and every P0 blocker gate reads PASS;
#   `budget` = PASS iff remaining_hr>=0 and the wave is not in deferred_waves.
# Mirrors scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:96-100 exactly.
GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

# run_gate_check / gate_detail — the shared verdict reader.  Mirrors
# scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:153-169 verbatim in shape.
# The wrapper NEVER re-implements report parsing and NEVER greps a tool's
# stdout for a verdict token: it asks the shared consumer, which reads the
# REPORT the tool wrote.
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

# USER_OK gate helper — blocks between waves.
user_ok_gate() {
    local wave_label="$1"
    if [ "${DRY_RUN:-0}" -eq 1 ] || [ "${YES:-0}" -eq 1 ]; then
        log "USER_OK gate: bypassed (dry-run / --yes) for wave ${wave_label}"
        return 0
    fi
    if [ -z "${WF310F_NEXT_WAVE:-}" ]; then
        die "USER_OK gate: set WF310F_NEXT_WAVE=${wave_label} to proceed (kill_then_adapt)"
    fi
    if [ "${WF310F_NEXT_WAVE}" != "${wave_label}" ]; then
        die "USER_OK gate: WF310F_NEXT_WAVE=${WF310F_NEXT_WAVE} != requested ${wave_label}"
    fi
    ok "USER_OK gate cleared for wave ${wave_label}"
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
YES=0
WAVE_TARGET="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --yes)      YES=1; shift ;;
        --wave)     WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)   WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,90p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ] \
    || die "wave must be A | B | C | all (got: ${WAVE_TARGET})"

[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]        || die "pre_flight_check.py missing"
[ -f "${CFM_TRAINER}" ]       || die "CFM trainer missing: ${CFM_TRAINER}"
[ -f "${DMPNN_TRAINER}" ]     || die "D-MPNN trainer missing: ${DMPNN_TRAINER}"
[ -f "${P1_4_CONFIG}" ]       || die "P1.4 config missing: ${P1_4_CONFIG}"
[ -f "${G6_CONFIG}" ]         || die "G6 config missing: ${G6_CONFIG}"

log "Wf-310F Day 7 launch begin — wave target = ${WAVE_TARGET} — dry-run = ${DRY_RUN}"

# --- USER_OK gate (canonical §5.9 + Wf-294B pattern) -----------------------
# Required by AGENTS.md §5.1 + Wf-294B family. The wave-level WF310F_NEXT_WAVE
# gate is per-wave; this USER_OK=1 gate is the launch-level authorization.
if [ "${USER_OK:-0}" != "1" ] && [ -z "${WF310F_NEXT_WAVE:-}" ]; then
    die "USER_OK gate refused: set USER_OK=1 in env (or WF310F_NEXT_WAVE=<wave>) to launch"
fi
ok "USER_OK gate cleared"

# Emit plan.json (Wf-363-A1 schema)
mkdir -p "${REPORTS_DIR}"
cat >"${REPORTS_DIR}/plan.json" <<EOF
{
    "wrapper": "launch_wf310f_frontier_math_gpu.sh",
    "wf_origin": "wf310f",
    "mode": "wave_${WAVE_TARGET}",
    "dry_run": ${DRY_RUN},
    "user_ok_required": true,
    "trainer": "${CFM_TRAINER}",
    "py": "${PY}",
    "out_dir": "${REPORTS_DIR}",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF

# -----------------------------------------------------------------------------
# Pre-launch check — pre_flight_check.py --pipeline launch
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch  (TODO-207 v2 §1 + §3 + §4)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight_check.py GPU/live gates (verdict UNVERIFIED, not PASS)"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the verdict was re-derived by
    # grepping the report, which cannot see a TIMEOUT, and the tool's stdout
    # was captured into a file whose name does not say "log".
    # MEASURED evidence:
    #   * molmetal/scripts/pre_flight_check.py:17360-17366 — `--json` takes a
    #     PATH (`type=Path, nargs="?"`); the report is written only
    #     `if args.json is not None` (:17734-17736).  The old call DID pass a
    #     path, so the artefact was valid JSON — this half was LIVE.
    #   * :17743-17751 — a TIMEOUT is a FAIL under --strict (Wf-A135); the
    #     old grep counted only '"status": "FAIL"' and would have said PASS
    #     on a report full of TIMEOUTs.
    #   * molmetal/scripts/wf_launch_gate_verdict.py:96-114 — the shared
    #     consumer's `preflight` verdict is PASS iff fail==0 AND timeout==0.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PRE_FLIGHT}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" \
    #             > "${REPORTS_DIR}/pre_flight.stdout" 2>&1; then
    #         die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    #     fi
    #     local n_pass n_fail
    #     n_pass=$(grep -c '"status": "PASS"'   "${REPORTS_DIR}/pre_flight.json" || true)
    #     n_fail=$(grep -c '"status": "FAIL"'   "${REPORTS_DIR}/pre_flight.json" || true)
    #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    #     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v2 §1)"
    #
    # REPAIRED: the tool's stdout goes to a .log file (never into the .json
    # path) and the verdict comes from the shared consumer, so a TIMEOUT is
    # a FAIL and a missing/unreadable report is a SKIP (never a PASS).
    # The gate stays HARD: a non-PASS verdict refuses the launch.
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
# Per-wave pre-launch gates #2 + #3 (Wf-A269 wave-11, 2026-11-02)
#   wave_gate <label> <budget_hr> <wave_dir> <config-or-empty>
#     #2 wave_budget_tracker.py   — ADVISORY (AGENTS.md §4: "gate: PLANNED only")
#     #3 per_wave_checklist.py    — HARD (the header calls it a BLOCKER)
# -----------------------------------------------------------------------------
wave_gate() {
    local wave_label="$1" budget_hr="$2" wave_dir="$3" config="$4"
    local wave_name="wave_${wave_label,,}"
    log "==== Wave ${wave_label} pre-launch gates (budget tracker + per-wave checklist) ===="

    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: wave ${wave_label} budget + checklist gates SKIPPED — verdict UNVERIFIED, not PASS"
        return 0
    fi

    # ---- gate #2: wave-specific budget tracker (advisory) --------------------
    # Wf-A269 wave-11: MEASURED evidence —
    #   * molmetal/scripts/wave_budget_tracker.py:1422 — main() returns 0 on
    #     EVERY path, so its exit code can never carry a verdict; the verdict
    #     lives ONLY in the report it writes (:1370-1399: remaining_hr,
    #     deferred_waves, honest_framing at the top level).
    #   * molmetal/scripts/wf_launch_gate_verdict.py:151-173 — `budget` is
    #     PASS iff remaining_hr >= 0 and the wave is not in deferred_waves.
    #   * The wave is named "wave_<label>" (no "cfm"), so the tracker's
    #     cfm-without-yuelbond deferral rule
    #     (wave_budget_tracker.py:865-871) does not fire.
    local bt_rc=0 bg_verdict bg_detail
    "${PY}" "${BUDGET_TRACKER}" \
        --budget-hr "${KILL_THEN_ADAPT_CAP_HR}" \
        --waves "${wave_name}:${budget_hr}" \
        --report-json "${wave_dir}/budget_report.json" \
        > "${wave_dir}/budget.log" 2>&1 || bt_rc=$?
    bg_verdict="$(run_gate_check budget "${wave_dir}/budget_report.json" "${wave_name}")"
    bg_detail="$(gate_detail budget)"
    printf '%s %s\n' "${bg_verdict}" "${bg_detail}" > "${wave_dir}/budget.verdict.txt"
    if [ "${bg_verdict}" = "PASS" ]; then
        ok "wave ${wave_label} budget ${bg_verdict} (${bg_detail})"
    else
        warn "wave ${wave_label} budget ${bg_verdict} (${bg_detail}) — advisory only (rc=${bt_rc}; see ${wave_dir}/budget.log)"
    fi

    # ---- gate #3: wave-specific per-wave checklist (hard) -------------------
    # Wf-A269 wave-11: this is the gate the header promised and no code ever
    # called; no wave_<x>_subchecklist.json was ever written.  It now runs
    # against THIS launch's own pre-flight report (never the frozen
    # molmetal/reports/wf259_run/pre_flight.json default at
    # molmetal/scripts/per_wave_checklist.py:86) and its verdict is read by
    # the shared consumer, which understands this exact report schema
    # (preflight_fail_count / p0_blocker_gates / error).
    local cl_rc=0 cl_verdict cl_detail
    local cl_args=(--wave "${wave_name}")
    if [ -n "${config}" ]; then
        cl_args+=(--config "${config}")
    fi
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECKLIST}" "${cl_args[@]}" \
        --preflight "${REPORTS_DIR}/pre_flight.json" \
        --out "${REPORTS_DIR}/${wave_name}_subchecklist.json" \
        --out-md "${REPORTS_DIR}/${wave_name}_subchecklist.md" \
        --time-budget-hr "${budget_hr}" \
        > "${wave_dir}/checklist.log" 2>&1 || cl_rc=$?
    cl_verdict="$(run_gate_check checklist "${REPORTS_DIR}/${wave_name}_subchecklist.json")"
    cl_detail="$(gate_detail checklist)"
    printf '%s %s\n' "${cl_verdict}" "${cl_detail}" > "${wave_dir}/checklist.verdict.txt"
    if [ "${cl_verdict}" != "PASS" ]; then
        warn "wave ${wave_label} checklist ${cl_verdict} (${cl_detail})"
        die "wave ${wave_label} per-wave checklist ${cl_verdict} — refusing to launch (report: ${REPORTS_DIR}/${wave_name}_subchecklist.json)"
    fi
    ok "wave ${wave_label} checklist ${cl_verdict} (${cl_detail})"
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

    user_ok_gate "${wave_label}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Wave ${wave_label}"
        cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Wave ${wave_label}
  trainer: ${trainer}
  config:  ${config}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
        ok "Wave ${wave_label} DRY-RUN logged"
        return 0
    fi

    local started_at
    started_at=$(date +%s)
    local deadline
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `${budget_hr%.*}` TRUNCATED the
    # budget to a whole number of hours.  MEASURED: wave B's budget is 0.5 h
    # => `${budget_hr%.*}` == "0" => deadline == started_at, so the first
    # poll tick (`now >= deadline`) killed the trainer IMMEDIATELY; wave A's
    # 1.5 h budget became a 1 h timeout and wave C's 2.5 h became 2 h.  The
    # timeout gate therefore never measured what it claimed.  awk is present
    # (/usr/bin/awk) and converts the decimal budget exactly.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    local budget_sec
    budget_sec=$(awk -v h="${budget_hr}" 'BEGIN { printf "%d", h*3600 }')
    deadline=$(( started_at + budget_sec ))

    # Background launcher for the trainer; PID captured for kill_then_adapt.
    # Bug fix 2026-10-08: removed --user-ok "REQUIRED" literal (G6/chemprop
    # trainers have no --user-ok flag; passing it caused argparse rc=2
    # backgrounded so no error visible). USER_OK gating is enforced by the
    # outer wrapper env check, NOT a CLI flag (per chemprop_v3 P1.4 wrapper
    # post-mortem same date).
    # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> ${gpu##cuda:}.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu##cuda:}}" \
        PYTHONPATH=/home/hugo/codes/mol-metal \
setsid nohup         "${PY}" "${trainer}" \
            --config "${config}" \
            --out-dir "${out_dir}" \
            "$@" \
            > "${out_dir}/train.log" 2>&1 &
    local pid=$!

    log "Wave ${wave_label} PID=${pid}; monitoring..."

    # Poll loop: every 60s, check timeout + GPU util + per-wave abort
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

        local gpu_util
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null \
                    | head -1 || echo "0")
        log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"

        sleep 60
    done

    # Capture the trainer's REAL exit status. The previous `wait ... || true`
    # discarded it, so a trainer that died instantly on an unknown flag
    # (argparse rc=2) still let the wave return 0 and print "complete".
    local trainer_rc=0
    wait "${pid}" 2>/dev/null || trainer_rc=$?

    local wall_sec=$(( $(date +%s) - started_at ))
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `bc` is NOT INSTALLED on this host
    # (MEASURED 2026-11-02: `command -v bc` -> rc=1, no binary), so the old
    # sub-shell produced an EMPTY string and the line read "wall=123s ( hr)".
    # awk is present (/usr/bin/awk); same pattern as
    # scripts/launch_wf306a_drugood_vina_gpu.sh:574-576.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     log "Wave ${wave_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr) rc=${trainer_rc}"
    local wall_hr
    wall_hr=$(awk -v s="${wall_sec}" 'BEGIN { printf "%.2f", s/3600 }')
    log "Wave ${wave_label} wall=${wall_sec}s (${wall_hr} hr) rc=${trainer_rc}"

    if [ "${trainer_rc}" -ne 0 ]; then
        warn "Wave ${wave_label} trainer exited rc=${trainer_rc} — see ${out_dir}/train.log"
        # rc=2 is argparse's "unrecognized arguments", i.e. the wave asked for a
        # flag the trainer does not implement. That is a BLOCKED wave, not a
        # completed one; say so instead of letting ok "complete" follow.
        if [ "${trainer_rc}" -eq 2 ]; then
            warn "Wave ${wave_label} BLOCKED: the trainer rejected an argument (rc=2)."
            warn "  Check the flags in this wave against: ${trainer} --help"
        fi
        return "${trainer_rc}"
    fi

    # Post-wave abort criteria sweep (per TODO-207 v2 §7.2)
    if [ "${aborted}" -eq 1 ] || ! grep -q "wave_${wave_label,,}_complete\|done" "${out_dir}/train.log"; then
        warn "Wave ${wave_label} FAIL — invoking abort_recovery.py"
        # Wf-A269 wave-11 (2026-11-02) — REPAIR: the abort_recovery call was
        # DEAD THREE WAYS.  MEASURED evidence
        # (molmetal/scripts/abort_recovery.py):
        #   * :366-371 — `--pattern` has choices=("NaN","OOM","IndexError",
        #     "cfm-collapse") (SUPPORTED_PATTERNS :99-105).  "TIMEOUT" is NOT
        #     one of them => argparse exited rc=2 before doing any work.
        #   * :372-377 — `--metal` is REQUIRED with choices
        #     ("Ru","Ir","Os","Rh","Re") (SUPPORTED_METALS :105).  The old
        #     call omitted it => argparse exited rc=2 again.
        #   * there is NO `--json` flag (the report flag is `--report-json`,
        #     :390, type=Path) => a third argparse error; and the stdout
        #     redirect put the tool's LOG into recovery.json, which is not a
        #     JSON report at all.  (DEFAULT_REPORT_JSON is a SHARED default
        #     path, so the report flag must always be given explicitly.)
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
        #         --pattern TIMEOUT --json \
        #         > "${out_dir}/recovery.json" 2>&1 || true
        #
        # REPAIRED: a legal --pattern, the REQUIRED --metal, --report-json
        # pointed at THIS wave's own path, the tool's stdout captured into a
        # .log file (never into the .json path), and the tool's REAL exit
        # code captured (`cmd; rc=$?` — an `if ! cmd` would force $?=0).
        # Mirrors the repaired block in
        # scripts/launch_wf306a_drugood_vina_gpu.sh:505-570.
        # HONEST NEGATIVE: abort_recovery's --metal filter only matches
        # checkpoints whose FILENAME STEM carries the metal token
        # (molmetal/scripts/abort_recovery.py:143-161), so a wave whose
        # checkpoints are not metal-named recovers nothing.
        local rec_pattern rec_metal rec_rc=0
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
        PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${ABORT_RECOVERY}" --wave "wave_${wave_label,,}" \
            --pattern "${rec_pattern}" --metal "${rec_metal}" \
            --report-json "${out_dir}/recovery.json" \
            > "${out_dir}/recovery.stdout.log" 2>&1 || rec_rc=$?
        if [ -s "${out_dir}/recovery.json" ] \
           && "${PY}" -c 'import json,sys; json.load(open(sys.argv[1]))' "${out_dir}/recovery.json" >/dev/null 2>&1; then
            ok "abort_recovery report written: ${out_dir}/recovery.json (rc=${rec_rc})"
        else
            warn "abort_recovery did NOT write a valid report (${out_dir}/recovery.json; rc=${rec_rc}) — see ${out_dir}/recovery.stdout.log"
        fi
        die "Wave ${wave_label} aborted — recovery at ${out_dir}/recovery.json"
    fi

    ok "Wave ${wave_label} complete — ${wall_sec}s"
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the per-wave OK gate compared
    # every wave against WAVE_A_BUDGET_HR, i.e. wave B (0.5 h cap) and wave C
    # (2.5 h cap) were both graded against wave A's 1.5 h.  It now uses the
    # wave's OWN budget (the same `budget_sec` the deadline above uses).
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if [ "${wall_sec}" -gt $(( ${WAVE_A_BUDGET_HR%.*} * 3600 )) ]; then
    #         die "Wave ${wave_label} exceeded per-wave < 2 hr OK gate (kill_then_adapt)"
    #     fi
    if [ "${wall_sec}" -gt "${budget_sec}" ]; then
        die "Wave ${wave_label} exceeded its own per-wave budget (${budget_hr} hr = ${budget_sec}s) — kill_then_adapt"
    fi
}

# -----------------------------------------------------------------------------
# Wave A — Day 3-4 CFM training-loop frontier-multi-ph wire-in (~1.5 hr)
# -----------------------------------------------------------------------------
wave_A() {
    local out="${REPORTS_DIR}/wave_a"
    # Wf-A269 wave-11 (2026-11-02): the per-wave gates (#2 budget tracker,
    # #3 per-wave checklist) run HERE, BEFORE the NOT_IMPLEMENTED guard.
    # Inside run_wave() they could never execute for waves A/B, because the
    # guard returns 3 first — a gate that can never run is a dead gate, and
    # the header promised both of these checks for every wave.
    wave_gate "A" "${WAVE_A_BUDGET_HR}" "${out}" "${P1_4_CONFIG}"
    # --frontier-multi-ph / --frontier-multi-ph-bins are NOT implemented in
    # ${CFM_TRAINER} (verified: neither the flag string nor the dest
    # frontier_multi_ph appears in the file). Passing them makes argparse exit
    # rc=2. Guard here so the wave reports BLOCKED rather than launching into
    # a guaranteed failure.
    if ! "${PY}" "${CFM_TRAINER}" --help 2>&1 | grep -q -- "--frontier-multi-ph"; then
        warn "Wave A BLOCKED: ${CFM_TRAINER} has no --frontier-multi-ph."
        warn "  augment_pocket_embedding (W2.3) was never wired into the trainer."
        warn "  This wave is NOT_IMPLEMENTED, not pending GPU time."
        return 3
    fi
    log "Wave A plan: P1.4 CFM trainer with --frontier-multi-ph (~1.5 hr, cuda:0)"
    run_wave "A" "${WAVE_A_BUDGET_HR}" \
        "${CFM_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
        --epochs 3 --seed 0 --pocket-id pocket_a \
        --frontier-multi-ph --frontier-multi-ph-bins 8 \
        --out-dir "${out}/frontier_multiph" \
        && ok "Wave A complete" \
        || warn "Wave A did not complete (see above)"
}

# -----------------------------------------------------------------------------
# Wave B — Day 5-6 D-MPNN multi-task frontier-SE3 wire-in (~0.5 hr)
# -----------------------------------------------------------------------------
wave_B() {
    local out="${REPORTS_DIR}/wave_b"
    # Wf-A269 wave-11 (2026-11-02): per-wave gates run BEFORE the guard (see
    # wave_A for the reason).
    wave_gate "B" "${WAVE_B_BUDGET_HR}" "${out}" "${G6_CONFIG}"
    # Same guard as wave A: --frontier-se3 is not implemented in
    # ${DMPNN_TRAINER} (no "--frontier" string, no frontier_se3 dest).
    if ! "${PY}" "${DMPNN_TRAINER}" --help 2>&1 | grep -q -- "--frontier-se3"; then
        warn "Wave B BLOCKED: ${DMPNN_TRAINER} has no --frontier-se3."
        warn "  SE3LipmanSolver (Integration 1) was never wired into the trainer."
        warn "  This wave is NOT_IMPLEMENTED, not pending GPU time."
        return 3
    fi
    log "Wave B plan: G6 D-MPNN multi-task trainer with --frontier-se3 (~0.5 hr, cuda:0)"
    run_wave "B" "${WAVE_B_BUDGET_HR}" \
        "${DMPNN_TRAINER}" "${G6_CONFIG}" "${WAVE_B_GPU}" "${out}" \
        --metals Ru --seeds 0 --epochs 10 --batch-size 64 --lr 1e-3 \
        --alpha 0.5 --wasserstein-weight 0.0 \
        --device cuda:0 --frontier-se3 \
        --out-dir "${out}/frontier_se3_ru" \
        && ok "Wave B complete" \
        || warn "Wave B did not complete (see above)"
}

# -----------------------------------------------------------------------------
# Wave C — Day 7 joint FrontierMathIntegration paper-grade sweep (~2-3 hr)
# -----------------------------------------------------------------------------
wave_C() {
    local out="${REPORTS_DIR}/wave_c"
    # Wf-A269 wave-11 (2026-11-02): per-wave gates run here too (no --config:
    # wave C is a joint sweep over the whole trainer stack, so the checklist's
    # config-schema check is skipped — it is never a PASS by omission, the
    # P0/P1 partition of THIS launch's pre-flight is still graded).
    wave_gate "C" "${WAVE_C_BUDGET_HR}" "${out}" ""
    log "Wave C plan: joint frontier sweep (PROJECTED < 2 hr sub-waves × 2, cuda:0)"
    log "  wires ALL 4 frontier adapters via FrontierMathIntegration"
    warn "Wave C requires USER_OK between sub-waves — auto-OK only on dry-run"
    log "Wave C sub-wave 1: FrontierMathIntegration.get_multi_ph_features + get_transport_relaxation (~1.5 hr)"
    log "Wave C sub-wave 2: FrontierMathIntegration.get_se3_lipman_solver + get_se3_sb_decoder (~1.0 hr)"
    ok "Wave C plan logged (joint sweep)"
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

ok "Wf-310F Day 7 launch script done — wave=${WAVE_TARGET}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each wave runs."
log "Per Wf-310F §1.7: USER_OK required between waves (kill_then_adapt)."
