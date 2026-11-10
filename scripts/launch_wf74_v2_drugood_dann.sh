#!/usr/bin/env bash
# Wf-336-X3 — Wf-74 v2 DrugOOD DANN launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Origin: Wf-336-X3 ultracode residual closure wave
# Status: DRY-RUN READY; USER_OK gate enforced (per Wf-294B kill_then_adapt)
#
# Purpose
# -------
# Wf-74 v2 is the docs-only wet-lab outreach closeout (per
# `TODO/completed/97c_wf74_reclassification.md` + Wf-300C pattern); Wf-74
# itself was DEPRECATED for wet-lab scope per AGENTS.md §5.4 Tier-2
# strategy.  However, the Wf-305D DrugOOD DANN trainer is the on-disk
# computational follow-up that lives at the Wf-74 v2 lineage: the
# drugood_v3_scaffold ood_test_auc=0.4574 BELOW the field-avg 0.55-0.65
# band (per Wf-297B + Wf-301A) needs a re-launch wrapper that wires all
# Wf-326 + Wf-328 production optimizations (W1-W10) onto the existing
# Wf-305D DANN driver.
#
# This wrapper:
#   1. Runs `pre_flight_check.py --pipeline all` (TODO-207 v3 gate).
#   2. Runs `wave_budget_tracker.py --cfm-yuelbond-required` (advisory).
#   3. Enforces the USER_OK gate (per Wf-294B / Wf-300C pattern).
#   4. Confirms VRAM budget via `nvidia-smi` (per AGENTS.md §5.1).
#   5. Wires all Wf-328 wire-ups (W1-W10) onto the Wf-305D driver:
#        * W1 torch.compile   (default ON)
#        * W2 num-workers     (--cpu-num-workers 8)
#        * W3 BF16            (default ON; --no-bf16 to disable)
#        * W4 TF32            (default ON)
#        * W5 W₂ sort         (--w2-gradient-mode sort)
#        * W6 DataLoader      (--num-workers 8)
#        * W7 PyG backend     (default ON; --dmpnn-backend pyg)
#        * W8 CUDA Graphs     (default OFF; --cuda-graph opt-in)
#        * W9 Triton edge     (default OFF; --triton-edge-update opt-in)
#        * W10 DMPNN cache    (default ON via Wf-305D cache module)
#   6. Default mode = Wf-335-R1: 1 seed × 1 epoch CPU verification only
#       (per Wf-335 launch suggestion smoke convention).  Wf-335-R1 mode
#       gates the bigger 3-seed × 50-epoch run behind USER_OK=YES.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
#   1. Runs `pre_flight_check.py --pipeline all` (TODO-207 v3 gate).
# Wf-A269 wave-5 (2026-11-02) - D9 [LOW]: item 1's `--pipeline all` spelling
#   is the SUPERSEDED one.  The pre-flight now runs ONCE per launcher
#   invocation on the canonical `--pipeline launch` surface, which is a
#   strict SUPERSET of `--pipeline all` (MEASURED 2026-11-02 by live import
#   of the pre-flight's own build_gates: len("all") == 129,
#   len("launch") == 131, gates lost by the merge = NONE).  The canonical
#   call is hoisted ABOVE the phase dispatch — see the pre-launch gate block
#   (the pre-flight gate function) and the note inside run_phase().
#
# Wf-A269 wave-11 (2026-11-02) — additive corrections to items 1, 3 and 5
# above (the SUPERSEDED renderings are preserved verbatim above):
#   * item 1 — the pre-flight gate's REPORT was captured with a BARE
#     `--json` redirected into the .json path, so the file held the tool's
#     stdout banner instead of JSON (MEASURED: json.load -> JSONDecodeError
#     "Expecting value: line 1 column 2 (char 1)"), and the FAIL-count grep
#     over that non-JSON file measured 0 hits — the blocking branch was
#     unreachable.  The gate now writes a real report with `--json <PATH>`
#     and reads the verdict from it through the shared consumer
#     (molmetal/scripts/wf_launch_gate_verdict.py).
#   * item 3 — the wrapper was SELF-CONTRADICTORY about USER_OK: its own
#     top-level gate accepts {1, YES} and its refusal message tells the user
#     to set USER_OK=1, while user_ok_gate() accepted ONLY "YES".  R3 now
#     accepts both spellings; USER_OK=1 is canonical repo-wide (MEASURED
#     census: 390 occurrences / 57 files vs 48 / 6 for USER_OK=YES).
#   * item 5 — W5 (`--w2-gradient-mode sort`) is NOT wireable on this driver:
#     MEASURED 2026-11-02, the Wf-305D trainer's own parse_args() answers
#     "unrecognized arguments: --w2-gradient-mode sort" -> SystemExit(2), so
#     the flag made BOTH phases die at argument parsing.  The flag belongs to
#     the D-MPNN multitask trainer
#     (molmetal/scripts/train_dmpnn_multitask_wasserstein.py:3220); the
#     Wf-305D driver hardcodes w2_gradient_mode="sliced"
#     (molmetal/scripts/wf305d_drugood_dann_train.py:347).  The wrapper no
#     longer passes a flag the tool does not define — see build_wf328_flags().
#
# Scope (PROJECTED per Wf-285B + Wf-304C honest-negative)
# ------------------------------------------------------
#   R1 mode (default; CPU verification):  1 seed × 1 epoch × hidden=64
#                                          × depth=2 → ~30 sec wall
#   R3 mode (USER_OK=YES; RTX 5090):     3 seeds × 50 epochs
#                                          × hidden=128 × depth=3 → ~3-4 hr
#                                          (per Wf-306A Phase-A timing)
#
# USER_OK gate (per Wf-300C pattern)
# ----------------------------------
#   --dry-run does NOT bypass the USER_OK gate.
#   Wf-335-R1 default mode: USER_OK is OPTIONAL (smoke only).
#   Wf-335-R3 mode (3-seed × 50-ep): USER_OK=YES required.
#
# Per-phase abort criteria (per TODO-207 v2 §7.2)
# ------------------------------------------------
#   7.2.1  per-phase timeout:  ckpt older than 2× projected wall-time
#   7.2.2  failure pattern:    3+ consecutive NaN/OOM/IndexError
#   7.2.3  GPU util < 70%:     for 5+ minutes
#   7.2.4  collapse:           drugood ood_test_auc < 0.45 on 2/3 seeds
#
# Usage
# -----
#   bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R1
#       # CPU smoke (Wf-335-R1 default; USER_OK not required)
#   USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3
#       # 3-seed × 50-epoch RTX 5090 full DANN run
#

set -euo pipefail

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
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason per block: Wf-431-A1 fix F2 — `molmetal.*` imports need the repo
# root on PYTHONPATH; a bare `.venv/bin/python` without it dies with
# "No module named 'molmetal'".  The canonical template exports it once,
# right after the interpreter variable.
export PYTHONPATH="/home/hugo/codes/mol-metal"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf74_v2_drugood_dann_launch"
PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"
BUDGET_TRACKER="${SCRIPTS_DIR}/wave_budget_tracker.py"
DRUGOOD_TRAINER="${SCRIPTS_DIR}/wf305d_drugood_dann_train.py"
DRUGOOD_CSV="${PROJECT_ROOT}/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv"

# Per-phase budget in hours (kill_then_adapt < 2 hr/phase hard cap)
R1_BUDGET_HR=0.5        # Wf-335-R1 smoke (CPU; 1 seed × 1 ep)
R3_BUDGET_HR=4.0        # Wf-335-R3 full GPU (3 seeds × 50 ep)

# Per-phase GPU assignment (per Wf-130 default kanzi GPU 1)
R1_GPU="cpu"            # R1 is CPU verification
R3_GPU="cuda:1"         # R3 is RTX 5090

# Wf-328 wire-up flags (mirror the production trainer defaults)
WF328_NUM_WORKERS=8
WF328_TORCH_COMPILE=1
WF328_TF32=1
WF328_BF16=1
WF328_W2_MODE="sort"
# Wf-A269 wave-11 (2026-11-02) — the constant is KEPT (it is the Wf-328 W5
# wire-up ledger value and molmetal/tests/test_wf341p3_wf74_user_ok.py
# asserts its presence) but it is NOT passed to the Wf-305D driver: that
# trainer has no `--w2-gradient-mode` flag.  MEASURED 2026-11-02: the
# trainer's own parse_args() answers "unrecognized arguments:
# --w2-gradient-mode sort" -> SystemExit(2).  See build_wf328_flags().
WF328_DMPNN_BACKEND="pyg"
# W8 (CUDA Graphs) and W9 (Triton edge) are opt-in only (per Wf-328-SYN §3)

# VRAM budget (per AGENTS.md §5.1 — RTX 5090 32GB or RTX PRO 6000 98GB)
VRAM_BUDGET_GB=20

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason per block: AGENTS.md §4 forbids a launch with no user OK — the
# canonical USER_OK gate (Wf-294B + Wf-355-B1 + Wf-366-A1) must be the FIRST
# side-effecting block, before any mkdir / tee / trainer call, or a GPU job
# can start with no user OK at all.
#
# This wrapper's historical spelling is USER_OK=YES (Wf-300C pattern, still
# enforced by the Wf-335-R3 branch further down), so BOTH spellings pass this
# gate; every other value — including unset — is refused with exit 64 (the
# repo's documented refusal code, per
# molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md:30).
if [ "${USER_OK:-0}" != "1" ] && [ "${USER_OK:-0}" != "YES" ]; then
    echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
    echo "       USER_OK=1 bash $(basename "$0") --dry-run   # plan only (still requires USER_OK=1)" >&2
    echo "       USER_OK=1 bash $(basename "$0")             # actual run" >&2
    exit 64
fi

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
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; exit 1; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" | tee -a "${REPORTS_DIR}/launch.log" >&2; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
MODE="R1"   # Wf-335-R1 default (CPU smoke)
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --mode)      MODE="${2:-R1}"; shift 2 ;;
        --mode-*)    MODE="${1#--mode-}"; shift ;;
        -h|--help)
            sed -n '2,75p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "${MODE}" = "R1" ] || [ "${MODE}" = "R3" ] \
    || die "mode must be R1 | R3 (got: ${MODE}; Wf-335-R1 default; Wf-335-R3 = USER_OK=YES)"

# -----------------------------------------------------------------------------
# Wf-A269 wave-11 (2026-11-02) — the shared gate-verdict consumer.
#
# WHY (every number below MEASURED on this host, CPU-only, 2026-11-02):
#   * `pre_flight_check.py --json` is `type=Path, nargs="?"`
#     (molmetal/scripts/pre_flight_check.py:17280-17286), so a BARE `--json`
#     writes NO file and prints the payload to stdout.  This wrapper captured
#     that stdout with `> preflight.json 2>&1`, so the "JSON" file held the
#     molmetal import banner on line 1 and was NOT valid JSON.  MEASURED
#     `json.load` on that file -> JSONDecodeError "Expecting value: line 1
#     column 2 (char 1)".  The gate then counted `"status": "FAIL"` in a
#     NON-JSON file: MEASURED 0 hits, so `n_fail` was 0 and the "FAIL count >
#     0" branch was unreachable — a FAILing pre-flight could not block.
#   * `wave_budget_tracker.py` returns 0 on EVERY path
#     (molmetal/scripts/wave_budget_tracker.py:1422).  MEASURED: with
#     `--budget-hr 1 --waves "wf74_drugood_dann:4.0"` it STILL exits 0 while
#     its own report says remaining_hr=-3.0, and the shared consumer on that
#     SAME report answers FAIL (rc=1).  A gate keyed on that exit code CANNOT
#     FAIL.
#   * `bc` is NOT INSTALLED on this host (MEASURED: `command -v bc` -> no
#     output), so every `bc` expression fell back to its `|| echo` arm: the
#     VRAM comparison always evaluated to "1" and the gate ALWAYS died with
#     "VRAM free=0GB < budget=20GB", even on a GPU with 32 GB free.  The
#     success branch was unreachable.
#   * `vram_gate()` referenced the UNBOUND variable `${mode}` (only `${MODE}`
#     and the local `${gpu}` are ever assigned), so under `set -u` the
#     "nvidia-smi not found" branch aborted with "mode: unbound variable"
#     instead of its intended diagnostic.
#
# The ONE place that knows how to read those reports is
# molmetal/scripts/wf_launch_gate_verdict.py (the Wf-A269 wave-10 consumer).
# Every gate below calls its tool with REAL flags and reads the verdict out
# of the tool's own report through that consumer; no gate greps stdout for a
# marker and no gate keys on an exit code that is 0 by construction.
# -----------------------------------------------------------------------------
GATE_VERDICT_CHECKER="${SCRIPTS_DIR}/wf_launch_gate_verdict.py"
[ -f "${GATE_VERDICT_CHECKER}" ] || die "launch-gate verdict checker missing: ${GATE_VERDICT_CHECKER}"

# This launch's OWN report paths — never a tool's shared default, never a path
# another wrapper overwrites.
PRE_FLIGHT_JSON="${REPORTS_DIR}/preflight.json"
BUDGET_JSON="${REPORTS_DIR}/budget_report.json"

# UNVERIFIED until a gate has actually measured something (AGENTS.md §5.2: a
# number that was not measured must never be presented as one; and the
# Wf-A269 wave-10 rule "a gate that was SKIPPED must read UNVERIFIED, never
# PASS").  The wrapper's verdict vocabulary is exactly PASS / FAIL / UNVERIFIED.
PRE_FLIGHT_VERDICT="UNVERIFIED"
BUDGET_VERDICT="UNVERIFIED"
PRE_FLIGHT_DETAIL="not run"
BUDGET_DETAIL="not run"

# Run the shared checker and echo back its verdict (PASS / FAIL / SKIP).
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

# The shared checker prints SKIP for a MISSING or UNREADABLE report.  For
# this wrapper that condition means "the measurement was not obtained", so it
# is recorded as UNVERIFIED — never PASS.
gate_verdict() {
    local kind="$1" report="$2" wave="${3:-}" raw=""
    raw="$(run_gate_check "${kind}" "${report}" "${wave}")"
    if [ "${raw}" = "SKIP" ]; then printf 'UNVERIFIED'; else printf '%s' "${raw}"; fi
}

# Wf-A269 wave-11 (2026-11-02) — the launch's gate verdicts are PERSISTED to
# this launch's own reports dir, ONE FILE PER GATE, written by run_gate_check
# as soon as that gate has run: ${REPORTS_DIR}/gate_<kind>.verdict.txt.  A
# launch that dies in a later gate therefore still leaves the EARLIER
# verdicts on disk.  (The wave-11 repair removed a shell-embedded python
# JSON writer: it duplicated the record format without parsing any report,
# so the wrapper now has exactly ONE record format — the one the reference
# wrapper scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh uses.)
# -----------------------------------------------------------------------------
# USER_OK gate (per Wf-300C pattern)
#   R1: optional (smoke only; CPU)
#   R3: required (full GPU run; kill_then_adapt)
# -----------------------------------------------------------------------------
user_ok_gate() {
    local mode="$1"
    if [ "${mode}" = "R3" ]; then
        if [ -z "${USER_OK:-}" ]; then
            die "USER_OK env var not set. Re-run with: USER_OK=1 bash $0 --mode R3 (USER_OK=YES is accepted as a legacy alias)"
        fi
        # Wf-A269 wave-11 (2026-11-02) — REPAIR: this wrapper was SELF-
        # CONTRADICTORY.  Its own top-level USER_OK gate (this file, above)
        # accepts {1, YES} and its refusal message tells the user to set
        # USER_OK=1, but this gate accepted ONLY "YES" — so the canonical
        # spelling the wrapper itself asks for was REFUSED here.  MEASURED
        # repo-wide census: USER_OK=1 appears 390 times in 57 files vs
        # USER_OK=YES 48 times in 6 files, so USER_OK=1 is the canonical
        # spelling and USER_OK=YES is kept as a legacy alias.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     if [ "${USER_OK}" != "YES" ]; then
        #         die "USER_OK must equal 'YES' (got: '${USER_OK}')"
        #     fi
        #     ok "USER_OK gate passed (USER_OK=YES)"
        if [ "${USER_OK}" != "YES" ] && [ "${USER_OK:-0}" != "1" ]; then
            die "USER_OK must equal '1' (canonical repo-wide spelling) or 'YES' (legacy alias) (got: '${USER_OK}')"
        fi
        ok "USER_OK gate passed (USER_OK=${USER_OK})"
    else
        if [ -z "${USER_OK:-}" ]; then
            warn "USER_OK not set; Wf-335-R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-335-R1 smoke)"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Pre-launch sanity
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${PRE_FLIGHT}" ]         || die "pre_flight_check.py missing"
[ -f "${BUDGET_TRACKER}" ]     || die "wave_budget_tracker.py missing"
[ -f "${DRUGOOD_TRAINER}" ]    || die "DrugOOD trainer missing: ${DRUGOOD_TRAINER}"
[ -f "${DRUGOOD_CSV}" ]        || die "DrugOOD CSV missing: ${DRUGOOD_CSV}"

# Wf-305D DANN driver marker check (per TODO-207 v3 + Wf-305D §DANN)
grep -q "use_dann" "${DRUGOOD_TRAINER}" \
    || die "Wf-305D trainer missing --use-dann flag (DANN GRL not wired)"
grep -q "ReverseLayerF" "${DRUGOOD_TRAINER}" \
    || die "Wf-305D trainer missing ReverseLayerF (DANN GRL not wired)"

log "Wf-336-X3 launch begin — mode=${MODE} dry-run=${DRY_RUN}"
user_ok_gate "${MODE}"

# -----------------------------------------------------------------------------
# Pre-launch check #1 — pre_flight_check.py --pipeline launch
# -----------------------------------------------------------------------------
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
#   # -----------------------------------------------------------------------------
#   # Pre-launch check #1 — pre_flight_check.py --pipeline all
#   # -----------------------------------------------------------------------------
#   pre_flight_gate() {
#       log "pre_flight_check.py --pipeline all (TODO-207 v3 gate)"
#       if [ "${DRY_RUN}" -eq 1 ]; then
#           warn "DRY-RUN: skipping pre_flight GPU/live gates"
#           return 0
#       fi
#       if ! "${PY}" "${PRE_FLIGHT}" --pipeline all --json \
#               > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
#           die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
#       fi
#       local n_pass n_fail
#       n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
#       n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
#       ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
#       [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)"
#   }
#
# Wf-A269 wave-5 (2026-11-02) - D9 [LOW] repair: the canonical pre-flight ran
# TWICE per launcher invocation -- once HERE on `--pipeline all` and once
# inside run_phase() on `--pipeline launch`.  It now runs ONCE, HERE, above
# the phase dispatch, on the canonical `--pipeline launch` surface.
# `--pipeline launch` is a strict SUPERSET of `--pipeline all` — MEASURED
# 2026-11-02 by live import of the pre-flight's own build_gates:
# len(build_gates("all")) == 129, len(build_gates("launch")) == 131, the
# gates lost by the merge are NONE, and the 2 gates ADDED by `launch` are
# p3_wf343c2_per_wave_runtime_check + p3_wf343c4_anchor_freshness_24h.
# So the merge is a STRENGTHENING: the 129 shared gates now run ONCE
# instead of twice, and nothing that used to run stops running.
# The FAIL exit code moves from this wrapper's `die` (1) to the canonical
# launch-surface code (3) — the same code the superseded in-phase call used;
# a FAILing pre-flight still BLOCKS the launch either way.
pre_flight_gate() {
    log "pre_flight_check.py --pipeline launch (TODO-207 v3 gate; canonical launch surface)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates"
        PRE_FLIGHT_DETAIL="not run: dry-run skips the live pre-flight"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR (the --json ARTEFACT DEFECT).
    # The report was captured with a BARE `--json` redirected into the .json
    # path.  `--json` is `type=Path, nargs="?"`
    # (molmetal/scripts/pre_flight_check.py:17280-17286), so a bare `--json`
    # parses to None, the tool writes NO file (:17654) and prints the payload
    # to stdout (:17659).  MEASURED 2026-11-02: `json.load` on the captured
    # file -> JSONDecodeError "Expecting value: line 1 column 2 (char 1)".
    # The gate then counted `"status": "FAIL"` in that NON-JSON file —
    # MEASURED 0 hits — so `n_fail` was 0 and the "FAIL count > 0" branch was
    # UNREACHABLE: a FAILing pre-flight could not block the launch.
    #
    # REPAIRED: the tool writes a real JSON report to THIS launch's own path
    # (--json <PATH>); its stdout+stderr go to a .log file (never into the
    # .json path); the verdict is read from the report by the shared checker.
    # MEASURED 2026-11-02 on this tree (CPU-only, 423 s wall):
    # `pre_flight_check.py --pipeline launch --json` measured
    # summary = {pass: 114, fail: 4, skip: 11, timeout: 2, total: 131} and
    # the shared consumer on that SAME report answers FAIL (rc=1); on a
    # MISSING report it answers SKIP (rc=2) -> UNVERIFIED -> also refused.
    # Both branches refuse the launch with exit 3 — the gate can FAIL.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! PYTHONPATH=/home/hugo/codes/mol-metal "${PY}" "${PRE_FLIGHT}" --pipeline launch --json \
    #             > "${REPORTS_DIR}/preflight.json" 2>&1; then
    #         printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "pre_flight_check FAIL — see ${REPORTS_DIR}/preflight.json" \
    #             | tee -a "${REPORTS_DIR}/launch.log" >&2
    #         exit 3
    #     fi
    #     local n_pass n_fail
    #     n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/preflight.json" || true)
    #     n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/preflight.json" || true)
    #     ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    #     if [ "${n_fail}" -ne 0 ]; then
    #         printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)" \
    #             | tee -a "${REPORTS_DIR}/launch.log" >&2
    #         exit 3
    #     fi
    # Wf-A269 wave-17 (2026-11-03): rm the PREVIOUS report first.  A crashed
    # or killed pre-flight must not leave the last run's PASS report on disk
    # for the consumer to read — "stale is treated as missing".
    rm -f "${PRE_FLIGHT_JSON}"
    PYTHONPATH=/home/hugo/codes/mol-metal "${PY}" "${PRE_FLIGHT}" --pipeline launch \
        --json "${PRE_FLIGHT_JSON}" \
        > "${REPORTS_DIR}/preflight.stdout.log" 2>&1 || true
    PRE_FLIGHT_VERDICT="$(gate_verdict preflight "${PRE_FLIGHT_JSON}")"
    PRE_FLIGHT_DETAIL="$(gate_detail preflight)"
    if [ "${PRE_FLIGHT_VERDICT}" = "PASS" ]; then
        ok "pre_flight ${PRE_FLIGHT_DETAIL}"
        return 0
    fi
    printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "pre_flight_check ${PRE_FLIGHT_VERDICT} — see ${PRE_FLIGHT_JSON} (${PRE_FLIGHT_DETAIL})" \
        | tee -a "${REPORTS_DIR}/launch.log" >&2
    exit 3
}

# -----------------------------------------------------------------------------
# Pre-launch check #2 — wave_budget_tracker.py
# -----------------------------------------------------------------------------
budget_gate() {
    log "wave_budget_tracker.py (advisory; wave=wf74_drugood_dann, budget=${R3_BUDGET_HR} hr)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: budget tracker advisory only"
        BUDGET_DETAIL="not run: dry-run skips the budget tracker"
        return 0
    fi
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the gate keyed on the tracker's
    # EXIT CODE, and that code is 0 on EVERY path
    # (molmetal/scripts/wave_budget_tracker.py:1422), so the "returned
    # non-zero" branch was UNREACHABLE and the gate always took its success
    # branch.  MEASURED 2026-11-02: with `--budget-hr 1 --waves
    # "wf74_drugood_dann:4.0"` the tracker STILL exits 0 while its own report
    # says remaining_hr=-3.0, and the shared consumer on that SAME report
    # answers FAIL (rc=1) — so the gate CAN fail once it reads the report.
    #
    # The tracker is also given THIS launch's wave explicitly (`--waves`),
    # because without it the tracker AUTO-DISCOVERS the repo's whole wave
    # plan (MEASURED 36 waves for the wf146 sibling) and reports a number
    # about the whole repo rather than about this launch.  The old hardcoded
    # `--budget-hr 6` was likewise a DIFFERENT phase's budget; this wrapper's
    # own declared phase budget is ${R3_BUDGET_HR} hr.
    # MEASURED with `--waves "wf74_drugood_dann:4.0"`: remaining_hr=0.0, 0
    # DEFERRED, the consumer answers PASS (rc=0).  `--cfm-yuelbond-required`
    # is kept: it applies to CFM waves only, and MEASURED 0 DEFERRED for this
    # (non-CFM) wave — the old log line claiming it "will defer" was wrong.
    #
    # This gate stays ADVISORY (per the Wf-336-X3 design): a FAIL warns, it
    # does not block.  It is NOT dead — it now carries a verdict.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     if ! "${PY}" "${BUDGET_TRACKER}" --budget-hr 6 --cfm-yuelbond-required \
    #             --report-json "${REPORTS_DIR}/budget_report.json" \
    #             > "${REPORTS_DIR}/budget.log" 2>&1; then
    #         warn "budget tracker returned non-zero (advisory only)"
    #     fi
    #     ok "budget tracker logged (DEFERRED expected; drugood is not CFM)"
    #
    # REPAIRED: the verdict is read from the tracker's own report through the
    # shared checker; the tool's stdout/stderr go to a .log file.
    "${PY}" "${BUDGET_TRACKER}" --budget-hr "${R3_BUDGET_HR}" \
        --cfm-yuelbond-required \
        --waves "wf74_drugood_dann:${R3_BUDGET_HR}" \
        --report-json "${BUDGET_JSON}" \
        > "${REPORTS_DIR}/budget.log" 2>&1 || true
    BUDGET_VERDICT="$(gate_verdict budget "${BUDGET_JSON}" "wf74_drugood_dann")"
    BUDGET_DETAIL="$(gate_detail budget)"
    if [ "${BUDGET_VERDICT}" = "PASS" ]; then
        ok "budget tracker ${BUDGET_DETAIL}"
    else
        warn "budget tracker ${BUDGET_VERDICT} (${BUDGET_DETAIL}) — advisory only (drugood DANN is not a CFM wave)"
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
        ok "CPU mode (Wf-335-R1) — no VRAM check required"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        # Wf-A269 wave-11 (2026-11-02) — REPAIR: this branch referenced the
        # UNBOUND variable ${mode} (only ${MODE} and the local ${gpu} are ever
        # assigned), so under `set -u` the script aborted with "mode: unbound
        # variable" and the intended diagnostic was never printed.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #     die "nvidia-smi not found in PATH (required for ${mode} GPU launch)"
        die "nvidia-smi not found in PATH (required for ${gpu} GPU launch)"
    fi
    local idx="${gpu##cuda:}"
    local free_mib
    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the VRAM arithmetic used `bc`,
    # and `bc` is NOT INSTALLED on this host (MEASURED: `command -v bc` ->
    # nothing).  Every `bc` expression therefore fell back to its `|| echo`
    # arm: free_gb became "0" and the comparison became "1", so the gate
    # ALWAYS died with "VRAM free=0GB < budget=20GB" — even on a GPU with
    # 32 GB free.  The success branch was UNREACHABLE.  awk is present on this
    # host and does the same arithmetic exactly.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     free_mib=$(nvidia-smi --query-gpu=memory.free \
    #                        --format=csv,noheader -i "${idx}" 2>/dev/null \
    #                 | head -1 | tr -d ' ' || echo "0")
    #     local free_gb
    #     free_gb=$(echo "${free_mib} / 1024" | bc 2>/dev/null || echo "0")
    #     if [ "$(echo "${free_gb} < ${VRAM_BUDGET_GB}" | bc 2>/dev/null || echo 1)" -eq 1 ]; then
    #         die "VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx} (AGENTS.md §5.1 OOM risk)"
    #     fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
                       --format=csv,noheader -i "${idx}" 2>/dev/null \
                | head -1 | tr -d ' ' || true)"
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

# -----------------------------------------------------------------------------
# Wf-328 wire-up flag builder (single source of truth)
#   Reflects Wf-328-SYN §3 production defaults (W3 BF16 ON, W4 TF32 ON,
#   W5 W₂ sort ON, W7 PyG ON, W8 CUDA Graphs OFF, W9 Triton OFF).
# -----------------------------------------------------------------------------
build_wf328_flags() {
    local mode="$1"
    local gpu="$2"
    local out_dir="$3"

    # Common flags
    local args=(
        --csv "${DRUGOOD_CSV}"
        --out-dir "${out_dir}"
        --ckpt-dir "${PROJECT_ROOT}/molmetal/checkpoints"
        --use-dann
        --lambda-grl 0.1
        --use-kendall
        --wasserstein-weight 0.05
        --alpha 0.5
        --lr 1e-3
        --cpu-num-workers "${WF328_NUM_WORKERS}"
    )

    # Wf-A269 wave-11 (2026-11-02) — REPAIR: `--w2-gradient-mode` is NOT a
    # flag of the Wf-305D DrugOOD DANN driver.  MEASURED 2026-11-02 by
    # calling the trainer's own parse_args() with THIS wrapper's flag list
    # (both R1 and R3):
    #   "unrecognized arguments: --w2-gradient-mode sort" -> SystemExit(2)
    # argparse exits 2 BEFORE any training work, so BOTH phases died at
    # argument parsing and the trainer never started — the "TEST (ood) AUC"
    # marker could never be produced, so the post-phase grep could never
    # succeed.  The flag belongs to the D-MPNN multitask trainer
    # (molmetal/scripts/train_dmpnn_multitask_wasserstein.py:3220), not to
    # this driver, which hardcodes w2_gradient_mode="sliced"
    # (molmetal/scripts/wf305d_drugood_dann_train.py:347).
    # HONEST FRAMING: the Wf-328 W5 wire-up (W₂ sort) is NOT available on this
    # driver; the wrapper does NOT pass a flag the tool does not define.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #         --w2-gradient-mode "${WF328_W2_MODE}"

    # Mode-specific overrides
    if [ "${mode}" = "R1" ]; then
        # Wf-335-R1 smoke: 1 seed × 1 epoch × hidden=64 × depth=2 (CPU)
        args+=( --seeds 0 --epochs 1 --hidden 64 --depth 2 )
    else
        # Wf-335-R3 full: 3 seeds × 50 epochs × hidden=128 × depth=3 (RTX 5090)
        args+=( --seeds 0 1 2 --epochs 50 --hidden 128 --depth 3
                --cuda-visible-devices "${TF276_R3_GPU_IDX:-1}" )
    fi

    if [ "${gpu}" != "cpu" ]; then
        args+=( --device cuda )
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
    shift 5
    # remaining args = --flag pairs

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

    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
    #   # Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
    #   # Reason per block: AGENTS.md §4 + TODO-207 — the pre-flight PASS gate
    #   # must run BEFORE the trainer invocation; the canonical form writes
    #   # ${OUT_DIR}/preflight.json and hard-fails (exit 3) when the gate FAILs.
    #   # The wrapper's pre-existing `--pipeline all` gate (pre_flight_gate())
    #   # is left untouched — it is the stricter superset (AGENTS.md §5.5 rule #6:
    #   # additive only, no existing line reworded).
    #   PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${PRE_FLIGHT}" --pipeline launch >"${REPORTS_DIR}/preflight.json" 2>&1 \
    #       || { echo "[FATAL] pre_flight_check FAIL — see ${REPORTS_DIR}/preflight.json" >&2; exit 3; }
    #
    # Wf-A269 wave-5 (2026-11-02) - D9 [LOW] repair: the canonical pre-flight
    # is HOISTED out of run_phase() into the single pre-launch call
    # (pre_flight_gate(), above the phase dispatch), so it runs ONCE per
    # launcher invocation instead of once per phase.
    #
    # WHY the in-phase placement was NOT load-bearing (the brief's PROVE step):
    #   1. This launcher runs exactly ONE phase per invocation — the mode
    #      dispatch is an if/elif, so there is no phase B whose pre-flight
    #      could ever see a phase-A artifact.  MEASURED 2026-11-02 with a
    #      `bash -x` dry-run trace: run_phase executes 1 time per invocation.
    #   2. No pre-flight gate reads anything this launcher writes between the
    #      old call site and the phase.  MEASURED 2026-11-02 by grepping the
    #      pre-flight for the launcher's own outputs (budget_report.json /
    #      budget.log / launch.log / plan.json): the only budget-report gate
    #      reads the repo-side wf262 wave-budget report, NOT this launcher's
    #      own out-dir, so moving the call earlier changes no verdict.
    #   3. The launch-only gates do not depend on launcher state either:
    #      p3_wf343c4_anchor_freshness_24h reads anchor-file mtimes against a
    #      24 h TTL (a wall clock, and the pre-flight itself already runs for
    #      minutes), and p3_wf343c2_per_wave_runtime_check SKIPs loudly
    #      because this launcher declares no wave (unchanged by the hoist).
    # The gate is STRICTLY STRONGER after the hoist: one `--pipeline launch`
    # call (131 gates) replaces `--pipeline all` (129 gates) + `--pipeline
    # launch` (131 gates), so the 129 shared gates run ONCE instead of twice
    # and the 2 launch-only gates are no longer separated from the rest.

    local started_at deadline pid aborted
    started_at=$(date +%s)
    deadline=$(( started_at + ${budget_hr%.*} * 3600 ))
    aborted=0

    if [ "${gpu}" = "cpu" ]; then
        "${PY}" "${trainer}" --out-dir "${out_dir}" "$@" \
            > "${out_dir}/train.log" 2>&1 &
        pid=$!
    else
        local idx="${gpu##cuda:}"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${idx}}" \
            "${PY}" "${trainer}" --out-dir "${out_dir}" "$@" \
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

    # Wf-A269 wave-11 (2026-11-02) — REPAIR: the trainer's exit code was
    # discarded by `wait ... || true`, so a trainer that CRASHED was
    # indistinguishable from one that finished.  It is now captured and
    # participates in the abort sweep.  (`bc` is NOT installed on this host —
    # MEASURED — so the wall-clock arithmetic below used `bc` too and
    # rendered an EMPTY hour figure; awk does the same arithmetic.)
    #
    # The marker set was repaired: "phase complete" has NO producer anywhere
    # in the repo (MEASURED 2026-11-02 by `git grep -na "phase complete"`:
    # the only hits are this wrapper's own grep and a SUPERSEDED comment in
    # scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh:597-606), so the
    # success branch could be satisfied by a marker that is never emitted.
    # The LIVE marker is "TEST (ood) AUC", produced by the trainer itself
    # (molmetal/scripts/wf305d_drugood_dann_train.py:812).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     wait "${pid}" 2>/dev/null || true
    #     local wall_sec
    #     wall_sec=$(( $(date +%s) - started_at ))
    #     log "Phase ${phase_label} wall=${wall_sec}s ($(echo "scale=2; ${wall_sec}/3600" | bc) hr)"
    #     # Post-phase abort criteria sweep
    #     if [ "${aborted}" -eq 1 ] || ! grep -q "TEST (ood) AUC\|phase complete" "${out_dir}/train.log"; then
    #         warn "Phase ${phase_label} FAIL — see ${out_dir}/train.log"
    #         die "Phase ${phase_label} aborted"
    #     fi
    local train_rc=0
    wait "${pid}" 2>/dev/null || train_rc=$?

    local wall_sec
    wall_sec=$(( $(date +%s) - started_at ))
    log "Phase ${phase_label} wall=${wall_sec}s ($(awk -v s="${wall_sec}" 'BEGIN{printf "%.2f", s/3600}') hr)"

    # Post-phase abort criteria sweep
    if [ "${aborted}" -eq 1 ] || [ "${train_rc}" -ne 0 ] \
            || ! grep -q "TEST (ood) AUC" "${out_dir}/train.log"; then
        warn "Phase ${phase_label} FAIL (trainer exit=${train_rc}) — see ${out_dir}/train.log"
        die "Phase ${phase_label} aborted"
    fi

    ok "Phase ${phase_label} complete — ${wall_sec}s (trainer exit=0)"
}

# -----------------------------------------------------------------------------
# Phase R1 — Wf-335-R1 smoke (1 seed × 1 epoch CPU; no user OK gate)
# -----------------------------------------------------------------------------
phase_R1() {
    local out="${REPORTS_DIR}/r1"
    log "Wf-335-R1 plan: drugood DANN smoke 1 seed × 1 epoch (~30 sec, CPU)"
    # Wf-A269 wave-11 (2026-11-02) — the W5 line was removed from the plan
    # log because the flag is NOT passed to this driver: MEASURED 2026-11-02,
    # the Wf-305D trainer's own parse_args() answers "unrecognized arguments:
    # --w2-gradient-mode sort" -> SystemExit(2) (see build_wf328_flags()).
    # The log now states exactly what is on the command line.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     log "  Wf-328 wire-ups: --cpu-num-workers 8 --w2-gradient-mode sort"
    log "  Wf-328 wire-ups: --cpu-num-workers 8 (W5 W₂ sort NOT available on this driver)"
    log "                   --use-dann --use-kendall --lambda-grl 0.1"
    log "                   --hidden 64 --depth 2 (smoke small)"
    local args
    mapfile -t args < <(build_wf328_flags "R1" "${R1_GPU}" "${out}")
    run_phase "R1" "${R1_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${R1_GPU}" "${out}" "${args[@]}"
    ok "Phase R1 drugood DANN smoke complete"
}

# -----------------------------------------------------------------------------
# Phase R3 — Wf-335-R3 full GPU run (3 seeds × 50 epochs RTX 5090)
# -----------------------------------------------------------------------------
phase_R3() {
    local out="${REPORTS_DIR}/r3"
    log "Wf-335-R3 plan: drugood DANN full 3-seed × 50-epoch (~3-4 hr, cuda:1)"
    # Wf-A269 wave-11 (2026-11-02) — same correction as phase_R1 (the W5
    # flag is not accepted by this driver; see build_wf328_flags()).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     log "  Wf-328 wire-ups: --cpu-num-workers 8 --w2-gradient-mode sort"
    log "  Wf-328 wire-ups: --cpu-num-workers 8 (W5 W₂ sort NOT available on this driver)"
    log "                   --use-dann --use-kendall --lambda-grl 0.1"
    log "                   --hidden 128 --depth 3 --batch-size 64 --lr 1e-3"
    local args
    mapfile -t args < <(build_wf328_flags "R3" "${R3_GPU}" "${out}")
    run_phase "R3" "${R3_BUDGET_HR}" \
        "${DRUGOOD_TRAINER}" "${R3_GPU}" "${out}" "${args[@]}"
    ok "Phase R3 drugood DANN full GPU run complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# Wf-A269 wave-5 (2026-11-02) - D9 [LOW]: this is the launcher's ONLY
# pre-flight invocation.  The canonical `--pipeline launch` call used to live
# inside run_phase(); it is HOISTED here (above the phase dispatch) so the
# pre-flight runs ONCE per invocation instead of once per phase.  It stays
# ABOVE wave_budget_tracker.py (the repo's kill_then_adapt ordering
# invariant: the pre-flight must precede the budget tracker) and above the
# trainer invocation.
pre_flight_gate

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason per block: Wf-363-A1 canonical dry-run surface — a dry run must emit a
# DRY_RUN.log + plan.json artifact pair under the wrapper's own out-dir,
# recording what WOULD run and that nothing was executed.  The existing
# per-phase DRY_RUN.log (${REPORTS_DIR}/<phase>/DRY_RUN.log) is still emitted
# further down; both are read-only artifacts — no trainer, no GPU.
if [ "${DRY_RUN}" -eq 1 ]; then
    # Wf-A269 wave-11 (2026-11-02) — the plan line no longer claims
    # "Wf-328 wire-ups W1-W10": W5 (W₂ sort) is NOT wireable on the
    # Wf-305D driver (MEASURED argparse rejection; see build_wf328_flags()).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    # [DRY-RUN] plan that WOULD run: mode=${MODE} — R1 = 1 seed x 1 epoch CPU smoke (0.5 hr budget); R3 = 3 seeds x 50 epochs on cuda:1 (4.0 hr budget); Wf-328 wire-ups W1-W10
    cat > "${REPORTS_DIR}/DRY_RUN.log" <<EOF
[DRY-RUN] launch_wf74_v2_drugood_dann.sh @ wf74_v2_drugood_dann
[DRY-RUN] wrapper_name=launch_wf74_v2_drugood_dann.sh
[DRY-RUN] wf_origin=wf74_v2_drugood_dann (Wf-336-X3 / Wf-305D DrugOOD DANN driver)
[DRY-RUN] trainer=${DRUGOOD_TRAINER}
[DRY-RUN] py=${PY}
[DRY-RUN] mode=${MODE}
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
[DRY-RUN] plan that WOULD run: mode=${MODE} — R1 = 1 seed x 1 epoch CPU smoke (0.5 hr budget); R3 = 3 seeds x 50 epochs on cuda:1 (4.0 hr budget); Wf-328 wire-ups W1-W10 except W5 (W₂ sort is NOT available on the Wf-305D driver — MEASURED argparse rejection)
[DRY-RUN] NOTHING WAS EXECUTED — no trainer was invoked, no GPU was touched.
EOF
    cat > "${REPORTS_DIR}/plan.json.tmp" <<EOF
{
  "wrapper": "launch_wf74_v2_drugood_dann.sh",
  "meta": {
    "wrapper_name": "launch_wf74_v2_drugood_dann.sh",
    "wf_origin": "wf74_v2_drugood_dann",
    "user_ok_required": true
  },
  "user_ok": true,
  "dry_run": true,
  "mode": "dry-run",
  "trainer": "${DRUGOOD_TRAINER}",
  "py": "${PY}",
  "out_dir": "${REPORTS_DIR}",
  "pre_flight": "UNVERIFIED (dry-run: live gates deferred per this wrapper's dry-run convention; a SKIPPED gate reads UNVERIFIED, never PASS)",
  "args": [],
  "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",
  "ts": "$(date -Iseconds)"
}
EOF
    mv -f "${REPORTS_DIR}/plan.json.tmp" "${REPORTS_DIR}/plan.json"
    echo "[DRY-RUN] launch_wf74_v2_drugood_dann.sh plan written to ${REPORTS_DIR}/DRY_RUN.log + ${REPORTS_DIR}/plan.json — nothing executed"
fi

budget_gate

if [ "${MODE}" = "R1" ]; then
    vram_gate "${R1_GPU}"
    phase_R1
elif [ "${MODE}" = "R3" ]; then
    vram_gate "${R3_GPU}"
    phase_R3
fi

ok "Wf-336-X3 launch script done — mode=${MODE}"
log "Reports: ${REPORTS_DIR}"
log "Gate verdicts (PASS/FAIL/UNVERIFIED) — persisted in ${REPORTS_DIR}/gate_<kind>.verdict.txt:"
log "  preflight=${PRE_FLIGHT_VERDICT}  budget=${BUDGET_VERDICT}"
log "Honest framing: PROJECTED wall-times; MEASURED after each phase runs."
log "Per Wf-294B kill_then_adapt: Wf-335-R1 = USER_OK optional; Wf-335-R3 = USER_OK=1 (or the legacy alias USER_OK=YES) required."
log "Wf-74 v2 lineage: wet-lab outreach DEPRECATED per AGENTS.md §5.4; computational"
log "follow-up is Wf-305D DrugOOD DANN (this wrapper) + Wf-306A DrugOOD+Vina combined."
