#!/usr/bin/env bash
# Wf-355-B2 — Wf-146 Re fold 5/5 + Os seed 4 DEFERRED runs re-launch wrapper
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-355-B2)
# Origin:      Wf-355 ultracode residual wave — closes the Wf-354-A1 §5
#              "6 DEFERRED runs" gap surfaced by Wf-342-P2 partial integration.
# Status:      DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt).
#
# Purpose
# -------
# Per Wf-354-A1 §5: Wf-146 chemprop v2 LOMO sweep PARTIAL: 19/25 (metal, seed)
# combinations completed.  Six remaining DEFERRED runs:
#   Re  seeds 0, 1, 2, 3, 4 (5 folds; sweep never started Re)
#   Os  seed 4          (1 fold; sweep stopped at epoch 10/50 on 2026-09-27 17:29)
# This wrapper closes the gap with a USER_OK-gated launch entry point.  All 6
# runs use the **smoke backbone** hidden=64/depth=2 (the same spec that produced
# the 19 MEASURED runs in Wf-146 sweep.log), so the new data is bit-comparable
# with the existing Re/Os rows.
#
# Spec (per Wf-146 sweep.log + metal_lomo_chemprop_v2.json + Wf-354-A1 §5)
#   - Data:     MetalCytoToxDB precious-metal subset (5 metals × N rows)
#   - Loader:   chemprop v2 SimpleMoleculeMolGraphFeaturizer + LeaveOneGroupOut
#                split (per metal fold)
#   - Backbone: hidden=64, depth=2 (smoke backbone per Wf-146 + Wf-354-A1 §6)
#                NOT production hidden=300/depth=3 (TODO-154 §3 reference)
#   - Train:    50 epochs × batch_size=50 × lr=1e-3 × BF16 autocast + Adam
#                optimizer (no LR scheduler per Wf-146 honest framing)
#   - Eval:     per-seed test AUC (held-out metal head)
#   - Hardware: single RTX 5090 (32 GB) or RTX PRO 6000 Blackwell (98 GB)
#   - ETA:      ~30-45 min GPU per (metal, seed) = ~3 hr total for 6 runs
#
# Modes / USER_OK gate (per Wf-294B / Wf-300C kill_then_adapt pattern;
# --dry-run does NOT bypass the gate):
#   R1 (default; CPU smoke): 1 (metal, seed) pair × 1 epoch on CPU, USER_OK
#     OPTIONAL (smoke only; ~3 min CPU)
#   R3 (USER_OK=YES; RTX 5090): 6 (metal, seed) pairs × 50 epochs, USER_OK=YES
#     required (kill_then_adapt ≥ 2 hr GPU)
#
# Per-wave safety belt (post-Wf-347 / Wf-343-C2 + Wf-343-C3):
#   - wf343c2_per_wave_check.py (7-check runtime safety belt; --skip-live since
#     the global pre_flight_check.py already covers VRAM + disk + freshness)
#   - wf343c3_data_quality_check.py (5×5 data quality matrix)
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
#   7.2.1  per-phase timeout:   ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:     3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:      for 5+ minutes
#   Wf-A269 wave-11 (2026-11-02) — HONEST-NEGATIVE: only 7.2.1 is implemented
#   (in run_pair); 7.2.2 and 7.2.3 are NOT — the GPU utilisation is only
#   LOGGED, never acted on.
#
# Usage
#   bash scripts/launch_wf146_deferred_runs.sh --dry-run --mode R1
#       # CPU smoke (Wf-355-B2 R1 default; USER_OK not required)
#   USER_OK=YES bash scripts/launch_wf146_deferred_runs.sh --mode R3
#       # 6 DEFERRED runs on RTX 5090 (~3 hr wall)
#   USER_OK=YES ONLY_RUNS=Re_0,Re_1 bash scripts/launch_wf146_deferred_runs.sh --mode R3
#       # Subset launch (just 2 Re runs)

set -euo pipefail


# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
export PYTHONPATH="/home/hugo/codes/mol-metal"
PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# --- Constants ---
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf146_chemprop_lomo"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
PER_WAVE_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
WF146_TRAINER="${SCRIPTS_DIR}/wf146_chemprop_lomo.py"

# Die helper — MOVED UP to before its first call (Wf-A269 repair 2, 2026-11-02).
# Bash executes a script line-by-line: a function defined LATER is not in scope
# at an earlier call site, so the call died with "die: command not found"
# (rc=127) and the intended FATAL message + rc=1 were both lost.
# ${C_RED:-}/${C_RST:-} keep the early call site safe (C_RED is initialised
# further down); `mkdir -p` keeps the FATAL message deliverable.
die()  { mkdir -p "${REPORTS_DIR}" 2>/dev/null || true; printf '%s[FATAL]%s %s\n' "${C_RED:-}" "${C_RST:-}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# --- Wf-A269 wave-11 (2026-11-02) — the shared gate-verdict consumer --------
# The ONE place that knows how to read a pre-flight / budget report is
# molmetal/scripts/wf_launch_gate_verdict.py (the Wf-A269 wave-10 consumer).
# Every gate below calls its tool with REAL flags and reads the verdict out of
# the tool's own report through that consumer: no gate greps stdout for a
# marker and no gate keys on an exit code that is 0 by construction.  The
# MEASURED evidence for each repair sits in the SUPERSEDED block above the
# repaired code (AGENTS.md §5.5 rule #6).
GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

# This launch's OWN report paths — never a tool's shared default, never a path
# another wrapper overwrites.
PRE_FLIGHT_JSON="${REPORTS_DIR}/pre_flight.json"
BUDGET_JSON="${REPORTS_DIR}/budget_report.json"
PER_WAVE_DIR="${REPORTS_DIR}/per_wave"
DATA_QUALITY_DIR="${REPORTS_DIR}/data_quality"

# UNVERIFIED until a gate has measured something (AGENTS.md §5.2; the Wf-A269
# wave-10 rule "a gate that was SKIPPED must read UNVERIFIED, never PASS").
# The verdict vocabulary is exactly PASS / FAIL / UNVERIFIED.
PRE_FLIGHT_VERDICT="UNVERIFIED"; BUDGET_VERDICT="UNVERIFIED"
PER_WAVE_VERDICT="UNVERIFIED"; DATA_QUALITY_VERDICT="UNVERIFIED"
PRE_FLIGHT_DETAIL="not run"; BUDGET_DETAIL="not run"
PER_WAVE_DETAIL="not run"; DATA_QUALITY_DETAIL="not run"

# Run the shared checker; echo PASS / FAIL / SKIP and persist its reason.
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

# The checker prints SKIP for a MISSING/unreadable report = "the measurement
# was not obtained" -> recorded as UNVERIFIED, never PASS.
gate_verdict() {
    local raw; raw="$(run_gate_check "$1" "$2" "${3:-}")"
    if [ "${raw}" = "SKIP" ]; then printf 'UNVERIFIED'; else printf '%s' "${raw}"; fi
}

# The launch's gate verdicts PERSIST to this launch's own reports dir
# (gate_<kind>.verdict.txt, written by run_gate_check) and are summarised on
# stdout at the end of the run (AGENTS.md §5.2: a gate that did not measure
# anything must never read PASS).

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.25       # Wf-355-B2 R1 smoke (CPU; 1 pair × 1 epoch)
# Wf-A269 wave-11 — REPAIR: the R3 (non-dry-run) path was DEAD.  phase_R3
# splits the phase budget per pair (per_pair_budget_hr = R3_BUDGET_HR /
# n_pairs) and then refuses to launch when the split is < 1 hr.  MEASURED
# arithmetic with the declared 3.0 hr envelope: ${R3_BUDGET_HR%.*} = 3,
# n_pairs = 6, 3 / 6 = 0 (integer division) -> [ 0 -ge 1 ] is FALSE ->
# die "per-pair budget 0 hr too small for 50-epoch run".  The R3 launch could
# therefore NEVER start: the check could only ever fire, never pass.
# REPAIRED with an arithmetic that matches the sweep the wrapper actually
# runs: 6 pairs x 1 hr per-pair sub-phase = a 6 hr declared ENVELOPE, each
# sub-phase inside the Wf-294B kill_then_adapt < 2 hr per-LAUNCH cap (the cap
# bounds one launch; this is 6 sequential launches).  The PROJECTED sweep
# total is ~3 hr (6 pairs x the MEASURED ~30 min per pair in the Wf-146
# sweep.log 2026-09-27); 6 hr is the ENVELOPE (per-pair sub-phase ceiling),
# not the projection.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# R3_BUDGET_HR=3.0        # Wf-355-B2 R3 full (6 pairs × ~30 min on RTX 5090)
R3_BUDGET_HR=6.0        # Wf-355-B2 R3 full (6 pairs x 1 hr sub-phase ceiling)

# Per-phase GPU assignment (per Wf-124 P3 + Wf-262 §0.5)
R1_GPU="cpu"
R3_GPU="cuda:0"

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB).
# Wf-146 hidden=64/depth=2 smoke backbone uses ~2-3 GB peak VRAM (per
# Wf-124 P3 chemprop v2 5-metal MEASURED at TODO-154 §3).
VRAM_BUDGET_GB=8

# 6 DEFERRED (metal, seed) pairs per Wf-354-A1 §5 + Wf-146 sweep.log
#   - Re  seeds 0..4:    all 5 folds never started (sweep halted 2026-09-27 17:29)
#   - Os  seed 4:        sweep halted at epoch 10/50 (last log line)
WF146_DEFERRED_PAIRS=(
    "Re_0"
    "Re_1"
    "Re_2"
    "Re_3"
    "Re_4"
    "Os_4"
)

mkdir -p "${REPORTS_DIR}/r1" "${REPORTS_DIR}/r3"

# --- Color codes (only when stdout is a tty) ---
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# --- CLI ---
DRY_RUN=0
MODE="R1"   # Wf-355-B2 R1 default (CPU smoke)
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,100p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-355-B2 R1 default; Wf-355-B2 R3 = USER_OK=YES)"

# --- USER_OK gate (per Wf-300C pattern) ---
#   R1: optional (smoke only; CPU)
#   R3: required (full GPU run; kill_then_adapt)
user_ok_gate() {
    local mode="$1"
    if [ "${mode}" = "R3" ]; then
        if [ -z "${USER_OK:-}" ]; then
            die "USER_OK env var not set. Re-run with: USER_OK=1 bash $0 --mode R3 (USER_OK=YES is accepted as a legacy alias)"
        fi
        if [ "${USER_OK}" != "YES" ] && [ "${USER_OK:-0}" != "1" ]; then
            die "USER_OK must equal '1' (canonical repo-wide spelling) or 'YES' (legacy alias) (got: '${USER_OK}')"
        fi
        ok "USER_OK gate passed (USER_OK=${USER_OK})"
    else
        if [ -z "${USER_OK:-}" ]; then
            warn "USER_OK not set; Wf-355-B2 R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-355-B2 R1 smoke)"
        fi
    fi
}

# --- Pre-launch sanity ---
[ -x "$(command -v "${PY}")" ]      || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]              || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]          || die "wave_budget_tracker.py missing"
[ -f "${PER_WAVE_CHECK}" ]          || die "wf343c2_per_wave_check.py missing"
[ -f "${DATA_QUALITY_CHECK}" ]      || die "wf343c3_data_quality_check.py missing"
[ -f "${WF146_TRAINER}" ]           || die "Wf-146 chemprop LOMO trainer missing: ${WF146_TRAINER}"

# Wf-146 trainer marker check
grep -q "ChempropV2MTModel" "${WF146_TRAINER}" \
    || die "Wf-146 trainer missing ChempropV2MTModel (chemprop v2 backbone not wired)"
grep -q "LeaveOneGroupOut" "${WF146_TRAINER}" \
    || die "Wf-146 trainer missing LeaveOneGroupOut (LOMO split not wired)"

log "Wf-355-B2 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# Emit plan.json (Wf-363-A1 schema)
mkdir -p "${REPORTS_DIR:-/home/hugo/codes/mol-metal/molmetal/reports/wf146_deferred_runs}"
cat >"${REPORTS_DIR:-/home/hugo/codes/mol-metal/molmetal/reports/wf146_deferred_runs}/plan.json" <<EOF
{
    "wrapper": "launch_wf146_deferred_runs.sh",
    "wf_origin": "wf146",
    "mode": "${MODE}",
    "dry_run": ${DRY_RUN},
    "user_ok_required": true,
    "trainer": "molmetal/scripts/train_chemprop_v2_v3.py",
    "py": "/home/hugo/codes/mol-metal/.venv/bin/python",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF

# --- Pre-launch check #1 — pre_flight_check.py --pipeline launch ---
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch (TODO-207 v3 gate; canonical launch surface)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates"
        PRE_FLIGHT_DETAIL="not run: dry-run skips the live pre-flight"
        return 0
    fi
    # Wf-A269 wave-11 — REPAIR 1 (the --json ARTEFACT DEFECT): the report was
    # captured with `> pre_flight.json 2>&1`, so the file held the tool's
    # stdout banner + stderr, MEASURED `json.load` -> JSONDecodeError
    # "Expecting value: line 1 column 2 (char 1)".  `--json` is
    # `type=Path, nargs="?"` (pre_flight_check.py:17280) -> a BARE `--json`
    # parses to None, writes NO file (:17654) and prints the payload to
    # stdout (:17659); `--json <PATH>` makes the tool write the file itself.
    #
    # REPAIR 2 (the pipeline surface): `--pipeline chemprop_v2_v3` is 18 of
    # the 131 launch gates.  MEASURED by live import of the pre-flight's own
    # build_gates: set("chemprop_v2_v3") <= set("launch") and set("all") <=
    # set("launch"), gates LOST: NONE — the launch surface STRENGTHENS the
    # gate.  MEASURED on THIS tree (CPU-only, 423 s): summary = {pass: 114,
    # fail: 4, skip: 11, timeout: 2, total: 131}; the shared consumer answers
    # FAIL (rc=1) on that report, so the gate CAN fail (and on this tree does:
    # p3_run_ledger_provenance, p3_ssot_path_reference_check,
    # p3_wf343c4_wave_budget_remaining_nonneg, p3_yaml_only_launch_check).
    #
    # REPAIR 3 (the exit-code trap): the verdict now comes from the report the
    # tool wrote, read by the shared consumer.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${PRE_FLIGHT}" --pipeline chemprop_v2_v3 --json \
    #             > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
    #         die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    #     fi
    #     local n_pass n_fail
    #     n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    #     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)"
    # Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A crashed
    # or killed pre-flight must not leave the last run's PASS report on disk
    # for the consumer to read — "stale is treated as missing".
    rm -f "${PRE_FLIGHT_JSON}"
    "${PY}" "${PRE_FLIGHT}" --pipeline launch --json "${PRE_FLIGHT_JSON}" \
        > "${REPORTS_DIR}/pre_flight.stdout.log" 2>&1 || true
    PRE_FLIGHT_VERDICT="$(gate_verdict preflight "${PRE_FLIGHT_JSON}")"
    PRE_FLIGHT_DETAIL="$(gate_detail preflight)"
    if [ "${PRE_FLIGHT_VERDICT}" = "PASS" ]; then
        ok "pre_flight ${PRE_FLIGHT_DETAIL}"
    else
        warn "pre_flight_check ${PRE_FLIGHT_VERDICT} (${PRE_FLIGHT_DETAIL})"
        # AGENTS.md §4: the pre-flight PASS gate blocks a launch.  A gate that
        # cannot fail is the defect this wave repairs, so FAIL *and*
        # UNVERIFIED (missing/unreadable report) both refuse the launch.
        die "pre_flight_check ${PRE_FLIGHT_VERDICT} — refusing to launch (report: ${PRE_FLIGHT_JSON})"
    fi
}

# --- Pre-launch check #2 — wave_budget_tracker.py (advisory) ---
budget_gate() {
    log "wave_budget_tracker.py (advisory; wave=wf146_deferred, budget=${R3_BUDGET_HR} hr)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        BUDGET_DETAIL="not run: dry-run skips the budget tracker"
        return 0
    fi
    # Wf-A269 wave-11 — REPAIR: the gate keyed on the tracker's EXIT CODE, and
    # that code is 0 on EVERY path (wave_budget_tracker.py:1422).  MEASURED:
    # with `--budget-hr 3` the tracker exits 0 while its OWN report says
    # remaining_hr=-49.1 with 5 DEFERRED waves, and the shared consumer on
    # that SAME report answers FAIL (rc=1) — the old "budget tracker logged"
    # branch could never fail.  The tracker is also given THIS launch's wave
    # explicitly (`--waves`): without it it AUTO-DISCOVERS the repo's whole
    # wave plan (36 waves, MEASURED) and reports about the whole repo, not
    # about this launch.  With `--waves` MEASURED: remaining_hr=0.0, 0
    # DEFERRED.  The gate stays ADVISORY (Wf-355-B2 design): a FAIL warns.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" \
    #             --report-json "${REPORTS_DIR}/budget_report.json" \
    #             > "${REPORTS_DIR}/budget.log" 2>&1; then
    #         warn "budget tracker returned non-zero (advisory only)"
    #     fi
    #     ok "budget tracker logged"
    "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" \
        --waves "wf146_deferred:${R3_BUDGET_HR}" \
        --report-json "${BUDGET_JSON}" \
        > "${REPORTS_DIR}/budget.log" 2>&1 || true
    BUDGET_VERDICT="$(gate_verdict budget "${BUDGET_JSON}" "wf146_deferred")"
    BUDGET_DETAIL="$(gate_detail budget)"
    if [ "${BUDGET_VERDICT}" = "PASS" ]; then
        ok "budget tracker ${BUDGET_DETAIL}"
    else
        warn "budget tracker ${BUDGET_VERDICT} (${BUDGET_DETAIL}) — advisory only (Wf-146 deferred runs are not CFM)"
    fi
}

# --- Pre-launch check #3 — VRAM budget (per AGENTS.md §5.1) ---
vram_gate() {
    local gpu="$1"
    log "VRAM budget check (--vram-budget-gb ${VRAM_BUDGET_GB})"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping nvidia-smi VRAM probe"
        return 0
    fi
    if [ "${gpu}" = "cpu" ]; then
        ok "CPU mode (Wf-355-B2 R1) — no VRAM check required"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found in PATH (required for ${gpu} GPU launch)"
    fi
    local idx="${gpu##cuda:}"
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader -i "${idx}" 2>/dev/null \
                | head -1 | tr -d ' ' || echo "0")
    # Wf-A269 wave-11 — REPAIR: the arithmetic used `bc`, and `bc` is NOT
    # INSTALLED on this host (MEASURED: `command -v bc` -> nothing), so every
    # `bc` call fell back to its `|| echo` arm: free_gb became "0", the
    # comparison became "1" and the gate ALWAYS died with "VRAM free=0GB <
    # budget=8GB" even on a GPU with 32 GB free.  awk is present and does
    # the same arithmetic exactly (also for the wall clock in run_pair).
    case "${free_mib}" in
        ''|*[!0-9]*) free_mib=0 ;;
    esac
    local free_gb
    free_gb=$(awk -v m="${free_mib}" 'BEGIN{printf "%.2f", m/1024}')
    if awk -v f="${free_gb}" -v b="${VRAM_BUDGET_GB}" 'BEGIN{exit !(f+0 < b+0)}'; then
        die "VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx} (AGENTS.md §5.1 OOM risk)"
    fi
    ok "VRAM free=${free_gb}GB on GPU ${idx} (budget=${VRAM_BUDGET_GB}GB OK)"
}

# --- Pre-launch check #4 — per-wave check (Wf-343c2 post-Wf-347 retro-fit) ---
per_wave_check_gate() {
    local wave="$1"
    local safe_wave="${wave//\//_}"
    log "wf343c2_per_wave_check.py --wave '${wave}' (7-check runtime safety belt)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping per-wave live probes (--skip-live equivalent)"
        PER_WAVE_DETAIL="not run: dry-run skips the per-wave safety belt"
        return 0
    fi
    # Wf-A269 wave-11 — REPAIR: the safety belt read the tool's FROZEN
    # ARCHIVED pre-flight, not this launch's own report:
    # `wf343c2_per_wave_check.py` defaults `--preflight` to
    # DEFAULT_PREFLIGHT_JSON = reports/wf259_run/pre_flight.json
    # (molmetal/scripts/wf343c2_per_wave_check.py:133).  MEASURED: with no
    # `--preflight` it reported "preflight upstream FAIL=0/98" from that
    # archive (mtime 2026-10-02) while this tree's live launch pre-flight
    # has FAIL gates — the belt could not see the repo's real blockers.
    # `--preflight` + `--preflight-live` point it at THIS launch's own
    # report under the 24 h mtime contract
    # (molmetal/scripts/wf343c2_per_wave_check.py:962-975).
    #
    # The verdict is the tool's EXIT CODE, which IS load-bearing here:
    # `main()` returns 1 iff a check FAILED
    # (molmetal/scripts/wf343c2_per_wave_check.py:1169-1173) — it is NOT one
    # of the always-0 tools.  The tool's summary line is quoted as EVIDENCE
    # in the detail; the verdict never branches on that text.  The report
    # path is per-wave so one pair's report cannot overwrite the previous
    # pair's (AGENTS.md §5.5 rule #6).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
    #             --wave "${wave}" \
    #             --pairs-metals "Re,Os" \
    #             --pairs-seeds "0,1,2,3,4" \
    #             --skip-live \
    #             --out "${REPORTS_DIR}/per_wave.json" \
    #             > "${REPORTS_DIR}/per_wave.log" 2>&1; then
    #         warn "per-wave check returned non-zero (advisory; see ${REPORTS_DIR}/per_wave.log)"
    #         return 0   # non-fatal per Wf-347-B1 §3 design
    #     fi
    #     ok "per-wave check logged (advisory)"
    local rc=0
    mkdir -p "${PER_WAVE_DIR}"
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
        --wave "${wave}" \
        --preflight "${PRE_FLIGHT_JSON}" \
        --preflight-live \
        --pairs-metals "Re,Os" \
        --pairs-seeds "0,1,2,3,4" \
        --skip-live \
        --out "${PER_WAVE_DIR}/per_wave_${safe_wave}.json" \
        > "${REPORTS_DIR}/per_wave.log" 2>&1 || rc=$?
    case "${rc}" in
        0) PER_WAVE_VERDICT="PASS" ;;
        1) PER_WAVE_VERDICT="FAIL" ;;
        *) PER_WAVE_VERDICT="UNVERIFIED" ;;
    esac
    PER_WAVE_DETAIL="wf343c2_per_wave_check exit=${rc} [$(tail -1 "${REPORTS_DIR}/per_wave.log" 2>/dev/null || printf '')]"
    if [ "${PER_WAVE_VERDICT}" = "PASS" ]; then
        ok "per-wave check ${PER_WAVE_DETAIL} (advisory)"
    else
        warn "per-wave check ${PER_WAVE_VERDICT} — ${PER_WAVE_DETAIL} (advisory; see ${REPORTS_DIR}/per_wave.log)"
    fi
    return 0   # non-fatal per Wf-347-B1 §3 design
}

# --- Pre-launch check #5 — data quality check (Wf-343c3 post-Wf-347 retro-fit) ---
data_quality_gate() {
    log "wf343c3_data_quality_check.py --strict (5×5 loader quality matrix)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping data quality live loaders"
        DATA_QUALITY_DETAIL="not run: dry-run skips the data quality matrix"
        return 0
    fi
    # Wf-A269 wave-11 — REPAIR: the old `cmd && ok || warn` discarded the
    # tool's exit code, so the message could not say WHICH outcome it was.
    # `wf343c3_data_quality_check.py` returns 1 on a FAILing loader, 1 on
    # `--strict` with any SKIPping loader and 2 when EVERY loader SKIPPED
    # (i.e. nothing was verified)
    # (molmetal/scripts/wf343c3_data_quality_check.py:876-891), so the exit
    # code IS load-bearing here.  rc=2 is recorded as UNVERIFIED, never PASS:
    # all-SKIP means no loader was actually checked.  The report goes to THIS
    # launch's own out-dir (never the tool's shared default) and the tool's
    # stdout goes to a .log file, never a .json path.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${DATA_QUALITY_CHECK}" \
    #         --strict \
    #         --out-dir "${REPORTS_DIR}/data_quality" \
    #         > "${REPORTS_DIR}/data_quality.log" 2>&1 \
    #         && ok "data quality check PASS" \
    #         || warn "data quality check returned non-zero (advisory; see ${REPORTS_DIR}/data_quality.log)"
    local rc=0
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${DATA_QUALITY_CHECK}" \
        --strict \
        --out-dir "${DATA_QUALITY_DIR}" \
        > "${REPORTS_DIR}/data_quality.log" 2>&1 || rc=$?
    case "${rc}" in
        0) DATA_QUALITY_VERDICT="PASS" ;;
        1) DATA_QUALITY_VERDICT="FAIL" ;;
        *) DATA_QUALITY_VERDICT="UNVERIFIED" ;;
    esac
    DATA_QUALITY_DETAIL="wf343c3_data_quality_check exit=${rc} [$(tail -1 "${REPORTS_DIR}/data_quality.log" 2>/dev/null || printf '')]"
    if [ "${DATA_QUALITY_VERDICT}" = "PASS" ]; then
        ok "data quality check ${DATA_QUALITY_DETAIL}"
    else
        warn "data quality check ${DATA_QUALITY_VERDICT} — ${DATA_QUALITY_DETAIL} (advisory; see ${REPORTS_DIR}/data_quality.log)"
    fi
    return 0
}

# --- Per-pair filter (per ONLY_RUNS env var) ---
#   ONLY_RUNS="Re_0,Re_1"  → launch only those 2 pairs
#   ONLY_REDATES=ALL (default) → launch all 6 pairs
select_pairs() {
    if [ -z "${ONLY_RUNS:-}" ]; then
        printf '%s\n' "${WF146_DEFERRED_PAIRS[@]}"
        return 0
    fi
    IFS=',' read -ra WANT <<< "${ONLY_RUNS}"
    for want in "${WANT[@]}"; do
        local matched=0
        for pair in "${WF146_DEFERRED_PAIRS[@]}"; do
            if [ "${pair}" = "${want}" ]; then
                printf '%s\n' "${pair}"
                matched=1
            fi
        done
        [ "${matched}" -eq 1 ] || die "ONLY_RUNS pair '${want}' not in 6 DEFERRED pairs: ${WF146_DEFERRED_PAIRS[*]}"
    done
}

# --- Wf-146 flag builder (single source of truth) ---
#   Reflects Wf-146 sweep.log + metal_lomo_chemprop_v2.json spec:
#   hidden=64, depth=2, 50 epochs, batch_size=50, lr=1e-3, BF16 autocast.
build_wf146_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"
    local metal="$4"
    local seed="$5"

    # Common flags (mirror Wf-146 sweep.log 2026-09-27 16:45:44 line)
    local args=(
        --metals "${metal}"
        --seeds "${seed}"
        --hidden 64
        --depth 2
        --batch-size 50
        --lr 1e-3
        --ckpt-dir "${out_dir}/ckpts"
        --json-dir "${out_dir}/jsons"
    )

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-355-B2 R1 smoke: 1 epoch only on CPU
        args+=( --epochs 1 )
    else
        # Wf-355-B2 R3 full: 50 epochs on RTX 5090 (mirror Wf-146 sweep)
        args+=( --epochs 50 )
    fi

    if [ "${gpu}" != "cpu" ]; then
        args+=( --device=cuda )
    else
        args+=( --device cpu )
    fi

    printf '%s\n' "${args[@]}"
}

# --- Per-phase runner with abort criteria + kill_then_adapt semantics ---
run_pair() {
    local mode="$1"
    local budget_hr="$2"
    local pair="$3"     # e.g. Re_0
    local out_dir="$4"
    local gpu="$5"

    local metal="${pair%_*}"
    local seed="${pair#*_}"

    log "==== Pair ${pair} begin (budget=${budget_hr} hr, gpu=${gpu}) ===="

    # Per-wave check (Wf-347-B2 pattern; before each pair)
    per_wave_check_gate "Wf-355-B2/${pair}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        local args
        mapfile -t args < <(build_wf146_flags "${mode}" "${gpu}" "${out_dir}" "${metal}" "${seed}")
        log "DRY-RUN: would launch Wf-146 ${pair}"
        # Wf-363-A1: use shared DRY_RUN.log schema helper for per-pair polish
        emit_dry_run_log "${out_dir}/DRY_RUN.log" \
            --title "Wf-355-B2 ${mode} deferred run (${pair})" \
            --wrapper "$(basename "$0")" \
            --mode "${mode}" \
            --trainer "${WF146_TRAINER}" \
            --pair "${metal} seed=${seed}" \
            --flags "${args[*]}" \
            --gpu "${gpu}" \
            --budget-hr "${budget_hr}" \
            --gate-verdict "preflight=${PRE_FLIGHT_VERDICT} budget=${BUDGET_VERDICT} per_wave_check=${PER_WAVE_VERDICT} data_quality=${DATA_QUALITY_VERDICT}" \
            --user-ok "$( [ "${mode}" = "R3" ] && echo REQUIRED || echo OPTIONAL_smoke )"
        ok "Pair ${pair} DRY-RUN logged"
        return 0
    fi

    local args
    mapfile -t args < <(build_wf146_flags "${mode}" "${gpu}" "${out_dir}" "${metal}" "${seed}")

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
setsid nohup         "${PY}" "${WF146_TRAINER}" "${args[@]}" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> ${idx}.
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${idx}}" \
setsid nohup             "${PY}" "${WF146_TRAINER}" "${args[@]}" \
                > "${out_dir}/train.log" 2>&1 &
        pid=$!
    fi

    log "Pair ${pair} PID=${pid}; monitoring..."

    while kill -0 "${pid}" 2>/dev/null; do
        local now
        now=$(date +%s)
        if [ "${now}" -ge "${deadline}" ]; then
            warn "Pair ${pair} exceeded budget ${budget_hr} hr (kill_then_adapt §7.2.1)"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 5
            kill -KILL "${pid}" 2>/dev/null || true
            aborted=1
            break
        fi
        if [ "${gpu}" != "cpu" ]; then
            local gpu_util
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu \
                                    --format=csv,noheader \
                                    -i "${gpu##cuda:}" 2>/dev/null \
                        | head -1 || echo "0")
            log "  tick now=${now} pid=${pid} gpu_util=${gpu_util}%"
        fi
        sleep 60
    done

    # Wf-A269 wave-11 — REPAIR: the trainer's exit code was discarded by
    # `wait ... || true`, so a trainer that CRASHED was indistinguishable from
    # one that finished.  It is now captured and participates in the abort
    # sweep.  (`bc` is NOT installed on this host — MEASURED — so the
    # wall-clock arithmetic below used `bc` too and rendered an EMPTY hour
    # figure; awk does the same arithmetic.)  The log marker itself is LIVE:
    # "test_auc.*status=ok" is produced by the trainer
    # (molmetal/scripts/wf146_chemprop_lomo.py:493-494, logger ->
    # StreamHandler -> stderr -> train.log).
    local train_rc=0
    wait "${pid}" 2>/dev/null || train_rc=$?

    local wall_sec
    wall_sec=$(( $(date +%s) - started_at ))
    log "Pair ${pair} wall=${wall_sec}s ($(awk -v s="${wall_sec}" 'BEGIN{printf "%.2f", s/3600}') hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || [ "${train_rc}" -ne 0 ] \
            || ! grep -q "test_auc.*status=ok" "${out_dir}/train.log"; then
        warn "Pair ${pair} FAIL (trainer exit=${train_rc}) — see ${out_dir}/train.log"
        die "Pair ${pair} aborted"
    fi

    ok "Pair ${pair} complete — ${wall_sec}s (trainer exit=0)"
}

# --- Phase R1 — Wf-355-B2 R1 smoke (1 DEFERRED pair × 1 epoch CPU) ---
#   R1 picks the FIRST pair from WF146_DEFERRED_PAIRS by default (Re_0).
#   Override via ONLY_RUNS=Re_0 or similar.
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-355-B2 R1 plan: Wf-146 chemprop LOMO 1-pair smoke (~3 min)"
    log "  Wf-146 driver: --epochs 1 --hidden 64 --depth 2 --bf16"
    local pair
    if [ -z "${ONLY_RUNS:-}" ]; then
        pair="${WF146_DEFERRED_PAIRS[0]}"   # Re_0
    else
        mapfile -t wanted < <(select_pairs)
        pair="${wanted[0]}"
    fi
    run_pair "R1" "${R1_BUDGET_HR}" "${pair}" "${out}" "${R1_GPU}"
    ok "Phase R1 Wf-146 deferred smoke complete (pair=${pair})"
}

# Wf-363-A1: emit phase-level DRY_RUN.log that aggregates ALL pairs for R3
# (otherwise each pair overwrites the same out_dir's DRY_RUN.log)
phase_R3_dry_run_log() {
    local out="${REPORTS_DIR}/r3"
    local pairs
    mapfile -t pairs < <(select_pairs)
    local pair_args=()
    local sample_args=()
    for p in "${pairs[@]}"; do
        local metal="${p%_*}"
        local seed="${p#*_}"
        pair_args+=( --pair "${metal} seed=${seed}" )
        # Build a representative flag set from the first pair for the schema summary
        if [ "${#sample_args[@]}" -eq 0 ]; then
            mapfile -t sample_args < <(build_wf146_flags "R3" "${R3_GPU}" "${out}" "${metal}" "${seed}")
        fi
    done
    local per_pair_budget_hr=1  # dry-run floor per-pair
    # Wf-A269 wave-11 (2026-11-02) — the record declared the PER-PAIR budget
    # (1 hr) for a phase that launches ${#pairs[@]} SEQUENTIAL runs, so a
    # reader could take it as "the whole R3 phase costs 1 hr" (AGENTS.md §5.2
    # honest framing).  The record now declares the SWEEP TOTAL (computed
    # here, by the only party that knows the pair count) together with the
    # wall policy and the per-pair budget that implies.
    local sweep_total_hr=$(( per_pair_budget_hr * ${#pairs[@]} ))
    local wall_policy="single_launch"
    if [ "${#pairs[@]}" -ge 2 ]; then
        wall_policy="whole_sweep_sequential"
    fi
    emit_dry_run_log "${out}/DRY_RUN.log" \
        --title "Wf-355-B2 R3 deferred runs (${#pairs[@]} pairs)" \
        --wrapper "$(basename "$0")" \
        --mode "R3" \
        --trainer "${WF146_TRAINER}" \
        --flags "${sample_args[*]}" \
        --gpu "${R3_GPU}" \
        --budget-hr "${sweep_total_hr}" \
        --budget-label "PROJECTED" \
        --wall-policy "${wall_policy}" \
        --per-pair-budget-hr "${per_pair_budget_hr}" \
        --gate-verdict "preflight=${PRE_FLIGHT_VERDICT} budget=${BUDGET_VERDICT} per_wave_check=${PER_WAVE_VERDICT} data_quality=${DATA_QUALITY_VERDICT}" \
        --user-ok "REQUIRED" \
        ${pair_args[@]+"${pair_args[@]}"}
    ok "Phase R3 DRY_RUN.log aggregated (${#pairs[@]} pairs) → ${out}/DRY_RUN.log"
}

# --- Phase R3 — Wf-355-B2 R3 full GPU run (6 DEFERRED pairs × 50 epochs RTX 5090) ---
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-355-B2 R3 plan: Wf-146 chemprop LOMO 6 DEFERRED pairs × 50 epochs (~3 hr cuda:0)"
    log "  Wf-146 driver: --epochs 50 --hidden 64 --depth 2 --bf16"
    local pairs
    mapfile -t pairs < <(select_pairs)
    local n_pairs=${#pairs[@]}
    [ "${n_pairs}" -gt 0 ] || die "no pairs selected (n_pairs=${n_pairs}); check ONLY_RUNS env var"
    # Per-pair budget: at minimum 1 hr (per Wf-294B kill_then_adapt < 2 hr hard cap
    # is a phase-level cap; per-pair budget is a sub-phase and should still cover
    # the 50-epoch sweep which MEASURED takes ~30 min on RTX 5090 per Wf-146 log).
    local per_pair_budget_hr
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Dry-run uses 1 hr floor regardless of pair count so the budget check
        # doesn't fail in verification mode (no actual GPU work happens).
        per_pair_budget_hr=1
    else
        per_pair_budget_hr=$(( ${R3_BUDGET_HR%.*} / n_pairs ))
        [ "${per_pair_budget_hr}" -ge 1 ] || die "per-pair budget ${per_pair_budget_hr} hr too small for 50-epoch run"
    fi
    log "  Per-pair time budget: ${per_pair_budget_hr} hr (R3_BUDGET_HR/${#pairs[@]} split)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        # Wf-363-A1: emit aggregated DRY_RUN.log BEFORE the per-pair loop so it
        # enumerates all pairs (per-pair log emission inside run_pair would
        # overwrite the same out_dir's DRY_RUN.log for each pair).
        phase_R3_dry_run_log
        return 0
    fi
    for pair in "${pairs[@]}"; do
        log "  - launching pair=${pair}"
        run_pair "R3" "${per_pair_budget_hr}" "${pair}" "${out}" "${R3_GPU}"
        ok "Phase R3 pair=${pair} complete"
    done
    ok "Phase R3 Wf-146 deferred full GPU run complete (${#pairs[@]} pairs)"
}

# --- Main ---
pre_flight_gate
budget_gate
data_quality_gate

if [ "${MODE}" = "R1" ]; then
    vram_gate "${R1_GPU}"
    phase_R1
elif [ "${MODE}" = "R3" ]; then
    vram_gate "${R3_GPU}"
    phase_R3
fi

ok "Wf-355-B2 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Gate verdicts (PASS/FAIL/UNVERIFIED) — persisted in ${REPORTS_DIR}/gate_<kind>.verdict.txt:"
log "  preflight=${PRE_FLIGHT_VERDICT}  budget=${BUDGET_VERDICT}  per_wave_check=${PER_WAVE_VERDICT}  data_quality=${DATA_QUALITY_VERDICT}"
log "Honest framing:"
log "  - PROJECTED wall-time: ~3 hr on RTX 5090 (6 pairs x the Wf-146 sweep MEASURED ~30 min/pair); declared envelope ${R3_BUDGET_HR} hr (6 pairs x 1 hr per-pair sub-phase ceiling)"
log "  - MEASURED baseline: Wf-146 sweep.log 2026-09-27 (19/25 done, 6 DEFERRED)"
log "  - 6 DEFERRED: Re 5/5 (fold never started) + Os seed 4 (sweep halted epoch 10/50)"
log "Per Wf-294B kill_then_adapt: Wf-355-B2 R1 = USER_OK optional; Wf-355-B2 R3 = USER_OK=1 required (USER_OK=YES accepted as a legacy alias)."
log "Per Wf-343c2 retro-fit pattern (Wf-347-B1/B2): per-wave check + data quality wired."
log "Wf-354-A1 §5 HONEST-NEGATIVE CLOSED by this wrapper (USER_OK-gated launch READY)."
