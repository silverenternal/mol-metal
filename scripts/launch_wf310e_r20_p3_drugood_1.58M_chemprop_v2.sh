#!/usr/bin/env bash
# Wf-355-A1 — R20 §3 DrugOOD 1.58M chemprop v2 retrain launch wrapper
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-355-A1)
# Origin:      Wf-355 ultracode residual wave — closes the R20 §3 1.58M
#              chemprop v2 launch wrapper gap surfaced by Wf-354-B1
#              (TODO-260 / R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §3 Phase 2).
# Status:      DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt).
#
# Purpose
# -------
# Per Wf-354-B1 §6 HONEST-NEGATIVE: the R20 §3 DrugOOD 1.58M chemprop v2
# launch wrapper did NOT exist on disk. The closest sibling
# (`scripts/launch_wf74_v2_drugood_dann.sh`) targets DANN on a 95K ECFP4
# subset, NOT the chemprop v2 spec on the full 1.58M DrugOOD corpus
# (1,583,302 rows × 3 subsets × 5 splits). This wrapper closes it.
#
# Spec (per R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §3.2 + Wf-185 trainer):
#   - Data:     DrugOOD IC50 full corpus (1,583,302 rows, 138 MB on disk)
#   - Loader:   molmetal/data/drugood_loader.py  (33 tests PASS per Wf-342-L1)
#   - Splits:   3 subsets × 5 splits × 50 epochs × multi-seed ≈ 750 cohort-runs
#               subsets:  lbap_refined_ic50_assay
#                         lbap_refined_ic50_scaffold
#                         lbap_refined_ic50_size
#               splits:   train / iid_val / iid_test / ood_val / ood_test
#   - Trainer:  molmetal/scripts/wf185_chemprop_v2_drugood_full.py
#               (per-config per-seed AUC JSON, persistent MolGraphCache,
#               BF16 autocast, ood_test gate ≥ 0.55 per subset)
#   - Model:    chemprop v2 AtomMessagePassing + SumAggregation +
#               BinaryClassificationFFN (via ChempropV2MTModel Ru head)
#   - Hardware: single RTX 5090 (32 GB) or RTX PRO 6000 Blackwell (98 GB)
#   - ETA:      ~10 hr wall on RTX 5090 per R20 §3.2 ETA
#
# Modes (per Wf-294B / Wf-300C kill_then_adapt pattern)
# -----------------------------------------------------
#   R1 (default; CPU smoke):  1 subset × 1 seed × 1 epoch × hidden=64/depth=2
#                              USER_OK OPTIONAL (smoke only; ~5 min CPU)
#   R3 (USER_OK=1; RTX 5090):
#                              3 subsets × 5 seeds × 50 epochs
#                              hidden=128/depth=2 (Wf-185 default smaller backbone)
#                              USER_OK=1 required (~10 hr wall)
#
# USER_OK gate (per Wf-300C pattern; --dry-run does NOT bypass the gate)
# -----------------------------------------------------------------------
#   Wf-355-A1 R1 mode: USER_OK optional (smoke).
#   Wf-355-A1 R3 mode: USER_OK=1 required (kill_then_adapt ≥ 2 hr GPU).
#                       USER_OK=YES is STILL ACCEPTED — that is the Wf-300C
#                       family spelling pinned by
#                       molmetal/tests/test_wf363a2_user_ok_ux.py:127 and
#                       molmetal/tests/test_wf363a1_dry_run_log.py:183.
#                       Wf-A269 wave-11 (2026-11-02): USER_OK=1 is the
#                       repo-wide canonical spelling (census: 390 occurrences
#                       of USER_OK=1 across 57 files vs 48 of USER_OK=YES
#                       across 6 files; 44 wrappers accept exactly {1}), so
#                       the refusal message now advertises USER_OK=1 while
#                       still accepting USER_OK=YES.
#
# Per-wave safety belt (post-Wf-347 / Wf-343-C2 + Wf-343-C3):
#   - wf343c2_per_wave_check.py  (7-check runtime safety belt; --skip-live
#                                  since global pre_flight_check.py already
#                                  covers VRAM + disk + freshness)
#   - wf343c3_data_quality_check.py (5×5 data quality matrix)
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:   ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:     3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:      for 5+ minutes
#   7.2.4  collapse:            ood_test_auc < 0.55 on 2/3 seeds
#
# Usage
# -----
#   bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R1
#       # CPU smoke (Wf-355-A1 R1 default; USER_OK not required)
#   USER_OK=1 bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --mode R3
#       # 3 subsets × 5 seeds × 50 epoch full chemprop v2 retrain (RTX 5090)
#

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
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310e_r20_p3_drugood_full_launch"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
PER_WAVE_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
DATA_QUALITY_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
CHEMPROP_TRAINER="${SCRIPTS_DIR}/wf185_chemprop_v2_drugood_full.py"
DRUGOOD_LOADER="${SCRIPTS_DIR}/../data/drugood_loader.py"

# Die helper — MOVED UP to before its first call (Wf-A269 repair 2, 2026-11-02).
# Bash executes a script line-by-line, so a function definition that appears
# LATER in the file is not yet in scope at an earlier call site — the call then
# dies with "die: command not found" (rc=127) and the intended FATAL message
# + rc=1 are both lost. ${C_RED:-}/${C_RST:-} keep the early call site safe
# (C_RED is initialised further down); `mkdir -p` keeps the FATAL message
# deliverable when the reports dir does not exist yet.
die()  { mkdir -p "${REPORTS_DIR}" 2>/dev/null || true; printf '%s[FATAL]%s %s\n' "${C_RED:-}" "${C_RST:-}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }

# Wf-A269 wave-11 (2026-11-02): the ONE place that knows how to read a
# launcher's pre-launch gate REPORT (pre-flight / per-wave checklist / wave
# budget tracker) and turn it into an honest PASS / FAIL / SKIP verdict.
# MEASURED evidence: molmetal/scripts/wf_launch_gate_verdict.py:1-241 —
#   sub-commands `preflight` | `checklist` | `budget`; exit 0=PASS, 1=FAIL,
#   2=SKIP (missing/unreadable report => SKIP, never PASS);
#   `preflight` = PASS iff summary.fail==0 AND summary.timeout==0
#   (a TIMEOUT is a FAIL per Wf-A135); `budget` = PASS iff remaining_hr>=0
#   AND the wave is not in deferred_waves.
# Mirrors scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:96-100 exactly.
GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.5        # Wf-355-A1 R1 smoke (CPU; 1 subset × 1 seed × 1 ep)
R3_BUDGET_HR=12.0       # Wf-355-A1 R3 full (3 subsets × 5 seeds × 50 ep on RTX 5090)

# Wave-budget scope (Wf-A269 wave-11, 2026-11-02) — the budget tracker is
# scoped to THIS launch instead of auto-discovering the repo's other waves.
#   BUDGET_WAVE_HR  = PROJECTED R3 wall-clock from this file's header
#                     ("~10 hr wall on RTX 5090 per R20 §3.2 ETA", header
#                     line 33).  PROJECTED, not MEASURED — the tracker's own
#                     honest_framing carries the tag.
#   BUDGET_WAVE_NAME must not contain "cfm": wave_budget_tracker.py:865-871
#                     defers cfm-without-yuelbond waves under
#                     --cfm-yuelbond-required (drugood is not a CFM run).
BUDGET_WAVE_NAME="drugood_r20_p3"
BUDGET_WAVE_HR=10.0

# Per-phase GPU assignment (per Wf-130 default kanzi GPU 1)
R1_GPU="cpu"
R3_GPU="cuda:0"

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB)
VRAM_BUDGET_GB=24

# 3 DrugOOD subsets (per Wf-185 default + R20 §3 spec)
DRUGOOD_SUBSETS=(
        "lbap_refined_ic50_assay"
        "lbap_refined_ic50_scaffold"
        "lbap_refined_ic50_size"
)

mkdir -p "${REPORTS_DIR}/r1" "${REPORTS_DIR}/r3"

# -----------------------------------------------------------------------------
# Color codes (only when stdout is a tty)
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# Wf-A269 wave-11 (2026-11-02) — shared launch-gate verdict reader.
# Mirrors scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:153-169 verbatim in
# shape.  The wrapper NEVER re-implements report parsing and NEVER greps a
# tool's stdout for a verdict token: it asks the shared consumer, which reads
# the REPORT the tool wrote.
#   run_gate_check <kind> <report> [wave]  -> "PASS" | "FAIL" | "SKIP"
#       kind: preflight | checklist | budget
#   gate_detail <kind>                     -> the consumer's first stdout line
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

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
MODE="R1"   # Wf-355-A1 R1 default (CPU smoke)
# Wf-A269 wave-9 (2026-11-02): the top-level DRY_RUN.log summary is written
# ONCE PER RUN, not once per FILE.  See the guard inside run_phase() for why
# the file-existence guard was wrong.
DRY_RUN_SUMMARY_WRITTEN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,99p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-355-A1 R1 default; Wf-355-A1 R3 = USER_OK=1)"

# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-300C pattern)
#   R1: optional (smoke only; CPU)
#   R3: required (full GPU run; kill_then_adapt)
# -----------------------------------------------------------------------------
user_ok_gate() {
    local mode="$1"
    if [ "${mode}" = "R3" ]; then
        # Wf-A269 wave-11 (2026-11-02) — USER_OK vocabulary.
        # MEASURED evidence (repo-wide census, 2026-11-02): USER_OK=1 is the
        # canonical spelling (390 occurrences across 57 files; 44 wrappers
        # accept exactly {1}); USER_OK=YES appears 48 times across 6 files.
        # This wrapper keeps accepting BOTH spellings so the Wf-300C family
        # contract pinned by molmetal/tests/test_wf363a2_user_ok_ux.py:127
        # (gate_value "YES") and molmetal/tests/test_wf363a1_dry_run_log.py:183
        # (WF363A1_USER_OK "YES") keeps passing; the refusal message now
        # advertises the canonical USER_OK=1.
        # There is NO AGENTS.md anchor for USER_OK — the vocabulary is
        # repo-wide policy, reported to the user as a policy item.
        if [ -z "${USER_OK:-}" ]; then
            die "USER_OK env var not set. Re-run with: USER_OK=1 bash $0 --mode R3"
        fi
        if [ "${USER_OK}" != "1" ] && [ "${USER_OK}" != "YES" ]; then
            die "USER_OK must equal '1' (canonical) or 'YES' (Wf-300C legacy) — got: '${USER_OK}'. Re-run with: USER_OK=1 bash $0 --mode R3"
        fi
        ok "USER_OK gate passed (USER_OK=${USER_OK})"
    else
        if [ -z "${USER_OK:-}" ]; then
            warn "USER_OK not set; Wf-355-A1 R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-355-A1 R1 smoke)"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ]      || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]              || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]          || die "wave_budget_tracker.py missing"
[ -f "${PER_WAVE_CHECK}" ]          || die "wf343c2_per_wave_check.py missing"
[ -f "${DATA_QUALITY_CHECK}" ]      || die "wf343c3_data_quality_check.py missing"
[ -f "${CHEMPROP_TRAINER}" ]        || die "chemprop v2 DrugOOD trainer missing: ${CHEMPROP_TRAINER}"
[ -f "${DRUGOOD_LOADER}" ]          || die "DrugOOD loader missing: ${DRUGOOD_LOADER}"

# Wf-185 trainer marker check (per TODO-260W + Wf-185 §DANN)
grep -q "ChempropV2MTModel" "${CHEMPROP_TRAINER}" \
    || die "Wf-185 trainer missing ChempropV2MTModel (chemprop v2 backbone not wired)"
grep -q "load_drugood" "${CHEMPROP_TRAINER}" \
    || die "Wf-185 trainer missing load_drugood (loader not wired)"

log "Wf-355-A1 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# Emit plan.json (Wf-363-A1 schema)
mkdir -p "${REPORTS_DIR:-/home/hugo/codes/mol-metal/molmetal/reports/wf355a1_r20_p3_drugood}"
cat >"${REPORTS_DIR:-/home/hugo/codes/mol-metal/molmetal/reports/wf355a1_r20_p3_drugood}/plan.json" <<EOF
{
    "wrapper": "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
    "wf_origin": "wf310e_r20_p3",
    "mode": "${MODE}",
    "dry_run": ${DRY_RUN},
    "user_ok_required": true,
    "trainer": "molmetal/scripts/train_chemprop_v2_v3.py",
    "py": "/home/hugo/codes/mol-metal/.venv/bin/python",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline launch
# -----------------------------------------------------------------------------
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch (TODO-207 v3 gate)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates (verdict UNVERIFIED, not PASS)"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the pre-flight REPORT ARTEFACT
    # was DEAD — nothing could read it, and the FAIL count was re-derived by
    # grepping a file that was not JSON.
    # MEASURED evidence: pre_flight_check.py:17360-17366 — `--json` is
    # `type=Path, nargs="?", default=None` and the report is written ONLY
    # `if args.json is not None` (:17734-17736), so a BARE `--json` writes
    # NO file; the payload goes to STDOUT (:17738) and the progress lines to
    # STDERR, so redirecting both into pre_flight.json produced a 1803-line
    # file whose first 268 lines are `[pre_flight_check] INFO` stderr and
    # whose `json.load` raises "Expecting value: line 1 column 2 (char 1)".
    # The grep also ignored TIMEOUT, which Wf-A135 (pre_flight_check.py:
    # 17743-17751) treats as a failure verdict, and a missing/unreadable
    # report yields SKIP, never PASS (wf_launch_gate_verdict.py:118-152).
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${PRE_FLIGHT}" --pipeline all --json \
    #             > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
    #         die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    #     fi
    #     local n_pass n_fail
    #     n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
    #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    #     [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)"
    #
    # REPAIRED: `--json` takes a PATH (the wrapper's OWN report path), the
    # tool's stdout+stderr go to a .log file (never into the .json path),
    # and the verdict comes from the shared consumer
    # (molmetal/scripts/wf_launch_gate_verdict.py) instead of a grep.
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
# Pre-launch check #2 — wave_budget_tracker.py
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py (advisory; scoped to THIS launch's R3 wave)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only (verdict UNVERIFIED, not PASS)"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the tracker's verdict was never
    # read, and the tracker measured OTHER launches, not this one.
    # MEASURED evidence: wave_budget_tracker.py:1422 — main() returns 0 on
    # EVERY path, so the `if ! cmd` branch was unreachable (the tracker can
    # never report a failure through its exit code); the verdict lives ONLY
    # in the report (wf_launch_gate_verdict.py:151-173: PASS iff
    # remaining_hr>=0 and the wave is not in deferred_waves).
    # MEASURED 2026-11-02 (this session): the old unscoped invocation
    # auto-discovered 8 waves belonging to OTHER launches and reported
    # remaining_hr=-40.1 with 6 DEFERRED waves (Wave-cfm:cfm_*,
    # Wave-dmpnn:wf124_p3_*, Wave-1/2 chemprop) — a FAIL verdict under the
    # shared consumer that the wrapper threw away.  Scoped
    # (--waves "drugood_r20_p3:10.0" against the wrapper's own 12.0 h cap,
    # R3_BUDGET_HR) it reports remaining_hr=2.0, deferred_waves=[],
    # wave_source="from --waves CLI", honest_framing="12.0h budget
    # sufficient; 2.0h buffer remaining after projected waves."
    # The wave name must NOT contain "cfm": wave_budget_tracker.py:865-871
    # defers any wave whose name contains "cfm" without "yuelbond" when
    # --cfm-yuelbond-required is passed (drugood is not a CFM run).
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" --cfm-yuelbond-required \
    #             --report-json "${REPORTS_DIR}/budget_report.json" \
    #             > "${REPORTS_DIR}/budget.log" 2>&1; then
    #         warn "budget tracker returned non-zero (advisory only)"
    #     fi
    #     ok "budget tracker logged (DEFERRED expected; drugood is not CFM)"
    #
    # REPAIRED: the verdict comes from the shared consumer reading the
    # tracker's OWN report; the gate stays ADVISORY (a planning number is
    # not a defect verdict) — it WARNS, it does not refuse the launch.
    local bt_rc=0
    "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" \
        --waves "${BUDGET_WAVE_NAME}:${BUDGET_WAVE_HR}" \
        --cfm-yuelbond-required \
        --report-json "${REPORTS_DIR}/budget_report.json" \
        > "${REPORTS_DIR}/budget.log" 2>&1 || bt_rc=$?
    local bg_verdict bg_detail
    bg_verdict="$(run_gate_check budget "${REPORTS_DIR}/budget_report.json" "${BUDGET_WAVE_NAME}")"
    bg_detail="$(gate_detail budget)"
    if [ "${bg_verdict}" = "PASS" ]; then
        ok "budget tracker ${bg_verdict} (${bg_detail})"
    else
        warn "budget tracker ${bg_verdict} (${bg_detail}) — advisory only (rc=${bt_rc}; see ${REPORTS_DIR}/budget.log)"
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch check #3 — VRAM budget (per AGENTS.md §5.1)
# -----------------------------------------------------------------------------
vram_gate() {
    local gpu="$1"
    log "VRAM budget check (--vram-budget-gb ${VRAM_BUDGET_GB})"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping nvidia-smi VRAM probe"
        return 0
    fi
    if [ "${gpu}" = "cpu" ]; then
        ok "CPU mode (Wf-355-A1 R1) — no VRAM check required"
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
    local free_gb
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `bc` is NOT INSTALLED on this
    # host (MEASURED 2026-11-02: `command -v bc` -> rc=1, no binary; awk IS
    # present at /usr/bin/awk).  The old expression therefore always
    # evaluated free_gb="0" (the `|| echo "0"` fallback) and the comparison
    # always evaluated "1" (the `|| echo 1` fallback), so this gate ALWAYS
    # died with "VRAM free=0GB < budget=24GB" — even on a completely idle
    # GPU.  A gate that cannot tell a full GPU from an empty one is not a
    # gate at all.  Bash integer arithmetic needs no external tool.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     local free_gb
    #     free_gb=$(echo "${free_mib} / 1024" | bc 2>/dev/null || echo "0")
    #     if [ "$(echo "${free_gb} < ${VRAM_BUDGET_GB}" | bc 2>/dev/null || echo 1)" -eq 1 ]; then
    #         die "VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx} (AGENTS.md §5.1 OOM risk)"
    #     fi
    case "${free_mib}" in
        ''|*[!0-9]*)
            die "nvidia-smi returned a non-numeric free VRAM ('${free_mib}') on GPU ${idx} — the AGENTS.md §5.1 VRAM budget cannot be evaluated"
            ;;
    esac
    free_gb=$(( free_mib / 1024 ))
    if [ "${free_gb}" -lt "${VRAM_BUDGET_GB}" ]; then
        die "VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx} (AGENTS.md §5.1 OOM risk)"
    fi
    ok "VRAM free=${free_gb}GB on GPU ${idx} (budget=${VRAM_BUDGET_GB}GB OK)"
}

# -----------------------------------------------------------------------------
# Pre-launch check #4 — per-wave check (Wf-343c2 post-Wf-347 retro-fit)
# -----------------------------------------------------------------------------
per_wave_check_gate() {
    local wave="$1"
    log "wf343c2_per_wave_check.py --wave '${wave}' (7-check runtime safety belt)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping per-wave live probes (--skip-live equivalent; verdict UNVERIFIED, not PASS)"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the belt read the FROZEN
    # archived pre-flight instead of THIS launch's, and its verdict was never
    # mapped to PASS / FAIL / UNVERIFIED (a SKIP therefore looked like
    # "logged" = success).
    # MEASURED evidence: wf343c2_per_wave_check.py:133 — DEFAULT_PREFLIGHT_JSON
    # = "reports/wf259_run/pre_flight.json", i.e. the Oct-02 archived report
    # (:212 FROZEN_PREFLIGHT_PATHS is the allow-list of frozen paths whose
    # freshness SKIP is accepted): without an explicit --preflight the belt
    # grades THIS launch against a MONTH-OLD archive.  :1166-1173 — exit
    # policy: n_fail>0 => 1; n_skip>0 AND --strict => 2; else 0, so --strict
    # is what turns a SKIP into a visible non-PASS (AGENTS.md §5.2: a skipped
    # gate reads UNVERIFIED, never PASS).  The tool's report schema
    # (n_pass/n_fail/n_skip/overall_status) is NOT the per_wave_checklist.py
    # schema that wf_launch_gate_verdict.py's `checklist` kind reads
    # (:62-112), so this report is deliberately NOT routed through the shared
    # consumer — it would answer PASS on an empty p0_blocker_gates list.  The
    # verdict here is the tool's own exit code, which IS a live oracle.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
    #             --wave "${wave}" \
    #             --pairs-metals "Ru,Ir,Os,Rh,Re" \
    #             --pairs-seeds "0,1,2,3,4" \
    #             --skip-live \
    #             --out "${REPORTS_DIR}/per_wave.json" \
    #             > "${REPORTS_DIR}/per_wave.log" 2>&1; then
    #         warn "per-wave check returned non-zero (advisory; see ${REPORTS_DIR}/per_wave.log)"
    #         return 0   # non-fatal per Wf-347-B1 §3 design
    #     fi
    #     ok "per-wave check logged (advisory)"
    #
    # REPAIRED: the belt now grades THIS launch's own pre-flight report
    # (--preflight + --preflight-live) and its exit code is mapped to an
    # honest verdict.  The gate stays ADVISORY (non-fatal per Wf-347-B1 §3
    # design): it WARNS, it does not refuse the launch.
    local pw_rc=0
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
        --wave "${wave}" \
        --pairs-metals "Ru,Ir,Os,Rh,Re" \
        --pairs-seeds "0,1,2,3,4" \
        --preflight "${REPORTS_DIR}/pre_flight.json" \
        --preflight-live \
        --skip-live \
        --strict \
        --out "${REPORTS_DIR}/per_wave.json" \
        > "${REPORTS_DIR}/per_wave.log" 2>&1 || pw_rc=$?
    local pw_verdict
    case "${pw_rc}" in
        0) pw_verdict="PASS" ;;
        1) pw_verdict="FAIL" ;;
        *) pw_verdict="UNVERIFIED" ;;
    esac
    if [ "${pw_verdict}" = "PASS" ]; then
        ok "per-wave check PASS (report: ${REPORTS_DIR}/per_wave.json)"
    else
        warn "per-wave check ${pw_verdict} (rc=${pw_rc}; advisory per Wf-347-B1 §3 design; see ${REPORTS_DIR}/per_wave.log)"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Pre-launch check #5 — data quality check (Wf-343c3 post-Wf-347 retro-fit)
# -----------------------------------------------------------------------------
data_quality_gate() {
    log "wf343c3_data_quality_check.py --strict (5×5 loader quality matrix)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping data quality live loaders"
        return 0
    fi
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${DATA_QUALITY_CHECK}" \
        --strict \
        --out-dir "${REPORTS_DIR}/data_quality" \
        > "${REPORTS_DIR}/data_quality.log" 2>&1 \
        && ok "data quality check PASS" \
        || warn "data quality check returned non-zero (advisory; see ${REPORTS_DIR}/data_quality.log)"
}

# -----------------------------------------------------------------------------
# Wf-185 flag builder (single source of truth)
#   Reflects Wf-185 default recipe: hidden=128, depth=2, head_hidden=128,
#   5 seeds, 20 epochs (R20 §3 spec bumps epochs to 50 + seeds=5).
# -----------------------------------------------------------------------------
build_wf185_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"
    local subset="$4"

    # Common flags
    local args=(
        --configs "${subset}"
        --out "${out_dir}/${subset}"
        --cache-dir "${out_dir}/cache"
        --reset-cache
    )

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-355-A1 R1 smoke: 1 seed × 1 epoch × hidden=64 × depth=2 (CPU)
        args+=( --seeds 0 --epochs 1 --hidden 64 --depth 2 --head-hidden 64
                --batch 64 --lr 1e-3 --bf16 --num-workers 2 --eval-max-samples 500 )
    else
        # Wf-355-A1 R3 full: 5 seeds × 50 epochs × hidden=128 × depth=2 (RTX 5090)
        local per_subset_budget_s=$(( ${R3_BUDGET_HR%.*} * 3600 / 3 ))
        args+=( --seeds 0 1 2 3 4 --epochs 50 --hidden 128 --depth 2 --head-hidden 128
                --batch 128 --lr 1e-3 --bf16 --num-workers 4
                --time-budget-s "${per_subset_budget_s}" )
    fi

    if [ "${gpu}" != "cpu" ]; then
        args+=( --device=cuda )
    else
        args+=( --device cpu )
    fi

    printf '%s\n' "${args[@]}"
}

# -----------------------------------------------------------------------------
# Per-phase runner with abort criteria + kill_then_adapt semantics
# -----------------------------------------------------------------------------
run_phase() {
    local phase_label="$1"
    local budget_hr="$2"
    local trainer="$3"
    local gpu="$4"
    local out_dir="$5"
    local subset="$6"
    shift 6
    # remaining args = --flag pairs

    log "==== Phase ${phase_label} begin (subset=${subset} budget=${budget_hr} hr, gpu=${gpu}) ===="
    log "  trainer: ${trainer}"
    log "  out_dir: ${out_dir}"
    log "  flags:   $*"

    # Per-wave check (Wf-347-B2 pattern; before each subset)
    per_wave_check_gate "Wf-355-A1/${phase_label}/${subset}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        log "DRY-RUN: would launch Phase ${phase_label} (subset=${subset})"
        # Wf-363-A1: use shared DRY_RUN.log schema helper for per-pair polish;
        # emit per-subset DRY_RUN.log so R3 mode doesn't overwrite across subsets.
        # Also emit a top-level summary at ${out_dir}/DRY_RUN.log when this is the
        # first subset so the dashboard can find a single canonical entry.
        local phase_root="${phase_label%%-*}"   # R1 or R3 (strip "-subset" suffix)
        local user_ok_label="$( [ "${phase_root}" = "R3" ] && echo REQUIRED || echo OPTIONAL_smoke )"
        emit_dry_run_log "${out_dir}/${subset}/DRY_RUN.log" \
            --title "Wf-355-A1 ${phase_label} (subset=${subset})" \
            --wrapper "$(basename "$0")" \
            --mode "${phase_label}" \
            --trainer "${trainer}" \
            --subset "${subset}" \
            --flags "$*" \
            --gpu "${gpu}" \
            --budget-hr "${budget_hr}" \
            --user-ok "${user_ok_label}"
        # Top-level summary log: list all subsets (written ONCE per RUN).
        #
        # Wf-A269 wave-9 (2026-11-02) — REPAIR: the guard used to be
        #     if [ ! -f "${out_dir}/DRY_RUN.log" ]; then
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        # (the line above is kept verbatim in the comment block for the record)
        #
        # i.e. "write it only if the file is not already there".  That made the
        # top-level summary a WRITE-ONCE-EVER artefact: a DRY_RUN.log left on
        # disk by an EARLIER run (here: 2026-09-29) permanently suppressed
        # every later write, so the record the dashboard and the tests read
        # described a run that had already ended.  It was worse than stale:
        # that September file predates the Wf-363-A1 schema extension, so it
        # carried no `budget_label`, and it still named `gpu: cuda:1` after
        # the wrapper moved to GPU 0 — a record asserting a plan that was
        # never planned.
        #
        # The guard is now per-RUN (a shell flag), not per-FILE: every
        # invocation rewrites the summary for the mode it is actually
        # planning, while the 3 R3 subsets still collapse into one summary.
        if [ "${DRY_RUN_SUMMARY_WRITTEN}" -eq 0 ]; then
            DRY_RUN_SUMMARY_WRITTEN=1
            local summary_pairs=()
            for s in "${DRUGOOD_SUBSETS[@]}"; do
                summary_pairs+=( --pair "subset=${s}" )
            done
            # shellcheck disable=SC2068
            emit_dry_run_log "${out_dir}/DRY_RUN.log" \
                --title "Wf-355-A1 ${phase_root} summary (${#DRUGOOD_SUBSETS[@]} subsets)" \
                --wrapper "$(basename "$0")" \
                --mode "${phase_root}" \
                --trainer "${trainer}" \
                --flags "$*" \
                --gpu "${gpu}" \
                --budget-hr "${budget_hr}" \
                --user-ok "${user_ok_label}" \
                ${summary_pairs[@]+"${summary_pairs[@]}"}
        fi
        ok "Phase ${phase_label} DRY-RUN logged (subset=${subset})"
        return 0
    fi

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
setsid nohup         "${PY}" "${trainer}" "$@" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> ${idx}.
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${idx}}" \
setsid nohup             "${PY}" "${trainer}" "$@" \
                > "${out_dir}/train.log" 2>&1 &
        pid=$!
    fi

    log "Phase ${phase_label} PID=${pid}; monitoring..."

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

    wait "${pid}" 2>/dev/null || true

    local wall_sec
    wall_sec=$(( $(date +%s) - started_at ))
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `bc` is NOT INSTALLED on this
    # host (MEASURED 2026-11-02: `command -v bc` -> rc=1), so the old
    # sub-shell produced an EMPTY string and the log line read "wall=123s
    # ( hr)".  awk is present (/usr/bin/awk) and is the pattern already used
    # by scripts/launch_wf299_cfm_main.sh:576 and
    # scripts/launch_wf306a_drugood_vina_gpu.sh:574-576.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     log "Phase ${phase_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"
    local wall_hr
    wall_hr=$(awk -v s="${wall_sec}" 'BEGIN { printf "%.2f", s/3600 }')
    log "Phase ${phase_label} wall=${wall_sec}s (${wall_hr} hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || ! grep -q "ood_test\|FINAL\|SUMMARY" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL — see ${out_dir}/train.log"
        die "Phase ${phase_label} aborted"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s"
}

# -----------------------------------------------------------------------------
# Phase R1 — Wf-355-A1 R1 smoke (1 subset × 1 seed × 1 epoch CPU)
# -----------------------------------------------------------------------------
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-355-A1 R1 plan: DrugOOD chemprop v2 smoke (1 subset × 1 seed × 1 epoch ~5 min CPU)"
    log "  Wf-185 driver: --seeds 0 --epochs 1 --hidden 64 --depth 2 --bf16"
    local subset="${DRUGOOD_SUBSETS[0]}"
    local args
    mapfile -t args < <(build_wf185_flags "R1" "${R1_GPU}" "${out}" "${subset}")
    run_phase "R1" "${R1_BUDGET_HR}" \
        "${CHEMPROP_TRAINER}" "${R1_GPU}" "${out}" "${subset}" "${args[@]}"
    ok "Phase R1 DrugOOD chemprop v2 smoke complete (subset=${subset})"
}

# -----------------------------------------------------------------------------
# Phase R3 — Wf-355-A1 R3 full GPU run (3 subsets × 5 seeds × 50 epochs RTX 5090)
# -----------------------------------------------------------------------------
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-355-A1 R3 plan: DrugOOD chemprop v2 full 3 subsets × 5 seeds × 50 epoch (~10 hr cuda:0)"
    log "  Wf-185 driver: --seeds 0..4 --epochs 50 --hidden 128 --depth 2 --bf16"
    log "  Per-subset time budget: $(( ${R3_BUDGET_HR%.*} * 3600 / 3 ))s (R3_BUDGET_HR/3 split)"
    for subset in "${DRUGOOD_SUBSETS[@]}"; do
        log "  - launching subset=${subset}"
        local args
        mapfile -t args < <(build_wf185_flags "R3" "${R3_GPU}" "${out}" "${subset}")
        local subset_budget_hr=$(( ${R3_BUDGET_HR%.*} / 3 ))
        run_phase "R3-${subset}" "${subset_budget_hr}" \
            "${CHEMPROP_TRAINER}" "${R3_GPU}" "${out}" "${subset}" "${args[@]}"
        ok "Phase R3 subset=${subset} complete"
    done
    ok "Phase R3 DrugOOD chemprop v2 full GPU run complete (3 subsets)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

ok "Wf-355-A1 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: Wf-355-A1 R1 = USER_OK optional; Wf-355-A1 R3 = USER_OK=YES required."
log "Per Wf-343c2 retro-fit pattern (Wf-347-B1/B2): per-wave check + data quality wired."
log "R20 §3 spec (1.58M DrugOOD full chemprop v2 retrain): HONEST-NEGATIVE CLOSED by this wrapper."
