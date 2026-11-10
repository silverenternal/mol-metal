#!/usr/bin/env bash
# Wf-426-A3 post-fix retrain readiness launcher.
#
# Iterates 6 retrain candidates (Wf-380 SCAFFOLD-loss, Wf-394 Lambda CFM,
# Wf-246-C metal-coordination, Wf-125 CONTROLD PAIR, Wf-390 Path A,
# Wf-299 3-wave CFM) and validates each is ready for USER_OK-gated actual
# launch. Honors AGENTS.md §5.1 (USER_OK gating, no auto-launch),
# §5.5 (filesystem-truth), §5.6 (atomic write protocol), §5.7 NEW
# (50-step CPU smoke before any trainer flag wire-in), §1.4 (pre-flight
# v3 117/0/3/117 PASS gate).
#
# Author: Wf-426-A3 — post-fix retrain readiness wave.
#
# Usage:
#   USER_OK=1 bash molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh [--all]
#       # Iterate all 6 retrain candidates; dry-run + readiness check.
#   USER_OK=1 bash molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh wf380
#       # Single retrain: Wf-380 SCAFFOLD-loss.
#   USER_OK=1 bash molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh wf299
#       # Single retrain: Wf-299 3-wave CFM.
#
# Wall budget: ~5 min CPU-only dry-run readiness loop per candidate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRE_FLIGHT="${REPO_ROOT}/molmetal/scripts/pre_flight_check.py"
AGGREGATOR="${REPO_ROOT}/molmetal/scripts/wf426a3_post_fix_results_aggregator.py"
READINESS_JSON="${REPO_ROOT}/molmetal/reports/wf426a3_retrain_readiness.json"
DASHBOARD_JSON="${REPO_ROOT}/molmetal/reports/gpu_launch_dashboard.json"

# Wf-A269 wave-25 (2026-11-04) — AGENTS.md §5.9 (single-venv) REPAIR.
#
# THE DEFECT (MEASURED 2026-11-04): every python invocation in this launcher
# was BARE `python`.  On this host the ambient PATH resolves `python` to
# /usr/bin/python = Python 3.14.7, which has NO torch
# (`/usr/bin/python -c "import torch"` => ModuleNotFoundError) — i.e. NOT the
# project venv (which is Python 3.12.13).  CONSEQUENCE (MEASURED):
# `USER_OK=1 bash molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh`
# exits rc=1 after 47.5 s at the pre-flight, so the readiness JSON
# (molmetal/reports/wf426a3_retrain_readiness.json) is NEVER produced — the
# file does not exist on disk.
# FIX: the canonical `PY="/home/hugo/codes/mol-metal/.venv/bin/python"`
# (AGENTS.md §5.9 + docs/STANDING_DIRECTIVES.md §F; the pattern already used
# by scripts/launch_wf*.sh) + a fail-closed existence check (exit 66).
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   PRE_FLIGHT_OUTPUT="$(python "${PRE_FLIGHT}" --pipeline all --json 2>&1)"
#   PASS_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('pass', d.get('summary', {}).get('pass', 0)))" 2>/dev/null || echo "0")"
#   FAIL_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('fail', d.get('summary', {}).get('fail', 0)))" 2>/dev/null || echo "0")"
#   SKIP_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('skip', d.get('summary', {}).get('skip', 0)))" 2>/dev/null || echo "0")"
#   TOTAL_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('total', d.get('summary', {}).get('total', 0)))" 2>/dev/null || echo "0")"
#   python - <<EOF > "${json_path}"
# Wf-A269 wave-29 (2026-11-06) — THE §5.9 REPAIR ITSELF WAS HOST-BOUND.
#
# THE DEFECT (MEASURED 2026-11-06): the wave-25 FIX above hard-codes the
# ABSOLUTE venv path.  That literal is the host-bound-path class
# (`molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py`): it is valid only
# on the machine that wrote it, so on any OTHER clone `PY` resolves to a
# NON-EXISTENT file and the fail-closed check below exits 66 — the launcher
# is correct on exactly one box.  It is also NOT REQUIRED to be absolute:
# AGENTS.md §5.9's scope is `scripts/launch_*.sh`, and this launcher lives in
# `molmetal/scripts/` — OUT of that scope AND out of
# `scripts/_wrapper_audit.py`'s `launch_wf*.sh` glob.  Meanwhile
# docs/STANDING_DIRECTIVES.md §F's CANONICAL pattern is the PORTABLE one
# (`"$REPO_ROOT/.venv/bin/python"`), and `REPO_ROOT` is already defined 28
# lines above this block.  ⇒ the wave-25 comment's citation of §F ("the
# pattern already used by scripts/launch_wf*.sh") cited the ABSOLUTE
# spelling, which is NOT §F's canonical pattern; that sentence is preserved
# verbatim above and is superseded by this one.
#
# THE REPAIR: `PY="${REPO_ROOT}/.venv/bin/python"` — byte-identical behaviour
# on this host (same interpreter, same venv) and correct on every other
# clone.  The fail-closed existence check (exit 66) is UNCHANGED.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   PY="/home/hugo/codes/mol-metal/.venv/bin/python"
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "[FATAL] the canonical venv python is MISSING at ${PY} — AGENTS.md §5.9 requires it." >&2
    exit 66
fi

# ---------------------------------------------------------------------------
# Standing directives — fail-fast on contract violation.
# ---------------------------------------------------------------------------
if [[ -z "${USER_OK:-}" || "${USER_OK}" != "1" ]]; then
    echo "[FATAL] USER_OK=${USER_OK:-unset} — per AGENTS.md §5.1, USER_OK=1 required." >&2
    exit 64
fi
if [[ ! -f "${PRE_FLIGHT}" ]]; then
    echo "[FATAL] pre_flight_check.py MISSING on disk per AGENTS.md §5.5." >&2
    exit 65
fi

# ---------------------------------------------------------------------------
# Pre-flight v3 PASS gate per AGENTS.md §1.4.
# ---------------------------------------------------------------------------
echo "[INFO] Running pre-flight v3 fresh-run check (~2 min wall)..."

# Wf-A269 wave-25 (2026-11-04) — THE FAIL GATE WAS DEAD.  THREE defects, all
# MEASURED, all in this block:
#
# (a) THE rc CHECK WAS UNREACHABLE.  `VAR="$(cmd)"` propagates cmd's exit
#     status, and `set -e` aborts the script on a non-zero status — so the
#     `PRE_FLIGHT_RC=$?` line could NEVER run, and the launcher's own
#     documented tolerance ("expected 0 or 2 SKIP-only") could never fire:
#     a SKIP-only pre-flight (rc 2) killed the launcher HERE instead of
#     reaching the FAIL_COUNT gate below.  MEASURED (bash repro):
#     `set -euo pipefail; OUT="$(bash -c 'exit 2')"; RC=$?` exits 2 at the
#     assignment and never reaches `RC=$?`; the same script without `set -e`
#     reaches it.
#     FIX: capture the rc explicitly with `|| PRE_FLIGHT_RC=$?` (a command on
#     the left of `||` is not subject to errexit).
#
# (b) THE COUNTS WERE NEVER PARSED.  The pre-flight's stdout is NOT pure
#     JSON: importing molmetal prints a banner FIRST —
#     "[molmetal] applied env vars: TOKENIZERS_PARALLELISM=false, ..." —
#     so `json.load(sys.stdin)` raised, the `|| echo "0"` fallback fired, and
#     ALL FOUR COUNTS BECAME "0".  CONSEQUENCE: `FAIL_COUNT` was ALWAYS 0, so
#     `if [[ "${FAIL_COUNT}" -gt 0 ]]` could NEVER fire — the §1.4 FAIL gate
#     was a DEAD GATE (the repo's own §6.269.17 class).
#     MEASURED 2026-11-04: the live `all` pipeline reports 120 PASS / 3 FAIL
#     / 10 SKIP / 133 TOTAL — i.e. the gate SHOULD have aborted, and could
#     not.  FIX: read the report from the FILE the pre-flight writes
#     (`--json PATH`), and FAIL CLOSED if the counts cannot be read.
#
# (c) The pre-flight's rc 1 means "at least one gate FAILed" — the abort
#     below is therefore reached with the real reason.  rc 2 = SKIP-only,
#     which the launcher tolerates and continues (as its own help text says).
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
PRE_FLIGHT_JSON="${WORK_DIR}/pre_flight_all.json"
PRE_FLIGHT_RC=0
PRE_FLIGHT_OUTPUT="$("${PY}" "${PRE_FLIGHT}" --pipeline all --json "${PRE_FLIGHT_JSON}" 2>&1)" || PRE_FLIGHT_RC=$?
# (c) THE PRE-FLIGHT'S EXIT-CODE CONTRACT (MEASURED from
#     `molmetal/scripts/pre_flight_check.py::main`, 2026-11-04):
#         return 2  ->  --strict AND (SKIP + TIMEOUT) > 0
#         return 1  ->  at least one gate FAILed, OR a gate TIMED OUT
#         return 0  ->  otherwise (a SKIP alone still exits 0, BY DESIGN)
#     ⇒ rc 1 IS the ordinary "a gate failed" signal, and rc 2 is a FAILURE
#     signal too (not the "SKIP-only" the launcher's old text claimed).
#
# (d) Wf-A269 wave-26 (2026-11-04) — THE §1.4 FAIL GATE WAS STILL UNREACHABLE
#     AFTER THE wave-25 REPAIR.  The wave-25 rc check aborted on ANY rc
#     other than 0 or 2 — and rc 1 is exactly the "a gate FAILed" case, so
#     the script exited at the rc check and NEVER REACHED the
#     `if [[ "${FAIL_COUNT}" -gt 0 ]]` gate below.  ⇒ the gate that wave-25
#     set out to revive was, mechanically, STILL DEAD (the repo's own
#     §6.269.17 class, second costume).
#     MEASURED 2026-11-04 (end-to-end, CPU-only): the launcher aborted with
#     rc=1 at the rc check while the live pre-flight reported
#     120 PASS / 3 FAIL / 10 SKIP / 133 TOTAL — i.e. it never printed the
#     FAIL count it had just measured.
#     REPAIR: the rc check now rejects only rc values OUTSIDE the
#     pre-flight's documented {0, 1, 2} (those mean the pre-flight itself
#     crashed).  rc 1 and rc 2 flow on to the FAIL-count gate, which is the
#     §1.4 authority: it aborts with the MEASURED count.  A non-zero rc with
#     ZERO FAIL gates (a TIMEOUT, or a strict-mode SKIP) is also aborted —
#     a failure the count alone does not explain must not pass silently.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   if [[ "${PRE_FLIGHT_RC}" -ne 0 && "${PRE_FLIGHT_RC}" -ne 2 ]]; then
#       echo "[FATAL] pre-flight v3 fresh-run rc=${PRE_FLIGHT_RC} (expected 0 or 2 SKIP-only)." >&2
#       echo "${PRE_FLIGHT_OUTPUT}" | tail -20 >&2
#       exit 1
#   fi
if [[ "${PRE_FLIGHT_RC}" -ne 0 && "${PRE_FLIGHT_RC}" -ne 1 && "${PRE_FLIGHT_RC}" -ne 2 ]]; then
    echo "[FATAL] pre-flight v3 fresh-run rc=${PRE_FLIGHT_RC} — outside the pre-flight's documented {0,1,2} (it crashed)." >&2
    echo "${PRE_FLIGHT_OUTPUT}" | tail -20 >&2
    exit 1
fi
# FAIL CLOSED (Wf-A269 wave-25): a pre-flight whose counts cannot be read is
# NOT a passing pre-flight.  The extractor prints "pass fail skip total" on
# one line, or exits non-zero; the `|| _counts=""` + the emptiness test below
# turn any extraction failure into an ABORT, never into a silent "0".
_PRE_FLIGHT_COUNTS="$("${PY}" - "${PRE_FLIGHT_JSON}" <<'PYEOF_COUNT'
import json
import sys

path = sys.argv[1]
try:
    raw = open(path, encoding="utf-8", errors="replace").read()
except OSError as exc:
    print("PARSE_ERROR: cannot read %s: %s" % (path, exc), file=sys.stderr)
    raise SystemExit(1)
doc = None
try:
    doc = json.loads(raw)
except json.JSONDecodeError:
    # The report file may carry a banner ahead of the JSON object; retry from
    # the first '{'.  If THAT fails, the extraction fails (fail-closed).
    i = raw.find("{")
    if i >= 0:
        try:
            doc = json.loads(raw[i:])
        except json.JSONDecodeError as exc:
            print("PARSE_ERROR: %s" % exc, file=sys.stderr)
            raise SystemExit(1)
if doc is None:
    print("PARSE_ERROR: no JSON object in %s" % path, file=sys.stderr)
    raise SystemExit(1)
summary = doc.get("summary") or {}


def pick(key):
    value = doc.get(key)
    if value is None:
        value = summary.get(key)
    if value is None:
        print("PARSE_ERROR: no %r key in %s" % (key, path), file=sys.stderr)
        raise SystemExit(1)
    return int(value)


print(pick("pass"), pick("fail"), pick("skip"), pick("total"))
PYEOF_COUNT
)" || _PRE_FLIGHT_COUNTS=""
if [[ -z "${_PRE_FLIGHT_COUNTS}" ]]; then
    echo "[FATAL] could not read the pre-flight counts from ${PRE_FLIGHT_JSON} — the §1.4 gate is FAIL-CLOSED (a pre-flight whose counts cannot be read is not a passing pre-flight)." >&2
    echo "${PRE_FLIGHT_OUTPUT}" | tail -20 >&2
    exit 67
fi
read -r PASS_COUNT FAIL_COUNT SKIP_COUNT TOTAL_COUNT <<< "${_PRE_FLIGHT_COUNTS}"
echo "[INFO] Pre-flight: ${PASS_COUNT} PASS / ${FAIL_COUNT} FAIL / ${SKIP_COUNT} SKIP / ${TOTAL_COUNT} TOTAL"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    echo "[FATAL] Pre-flight has ${FAIL_COUNT} FAIL gate(s) — abort per AGENTS.md §1.4." >&2
    exit 2
fi
# A non-zero pre-flight rc with ZERO FAIL gates = a TIMEOUT, or a
# strict-mode SKIP.  Neither is a failure the FAIL-count gate can see, so
# it is adjudicated here — otherwise a TIMEOUT would pass silently.
if [[ "${PRE_FLIGHT_RC}" -ne 0 ]]; then
    echo "[FATAL] pre-flight rc=${PRE_FLIGHT_RC} with 0 FAIL gate(s) (a TIMEOUT, or a strict-mode SKIP) — abort per AGENTS.md §1.4." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 6 retrain candidates — wrapper check + DRY_RUN.log + plan.json schema.
# Wf-299 wrapper does NOT exist on disk per AGENTS.md §5.5 — that
# candidate will be flagged HONEST_NEGATIVE_BLOCKED at wrapper check.
# ---------------------------------------------------------------------------
CANDIDATES_DIR="$(mktemp -d)"
trap "rm -rf ${WORK_DIR} ${CANDIDATES_DIR}" EXIT

write_candidate_record() {
    local name="$1"
    local script="$2"
    local report_dir="$3"
    local notes="$4"
    local json_path="${CANDIDATES_DIR}/${name}.json"
    "${PY}" - <<EOF > "${json_path}"
import json
record = {
    "name": "${name}",
    "script": "${script}",
    "script_exists": __import__("os").path.isfile("${script}"),
    "report_dir": "${report_dir}",
    "report_dir_exists": __import__("os").path.isdir("${report_dir}"),
    "dry_run_log_exists": __import__("os").path.isfile("${report_dir}/DRY_RUN.log"),
    "plan_json_exists": __import__("os").path.isfile("${report_dir}/plan.json"),
    "notes": "${notes}",
}
print(json.dumps(record, indent=2))
EOF
    cat "${json_path}"
}

# Candidate 1: Wf-380 SCAFFOLD-loss (CUDA trainer flag wire-in surface).
WF380_SCRIPT="${REPO_ROOT}/scripts/launch_wf380_scaffold_loss.sh"
WF380_REPORT="${REPO_ROOT}/molmetal/reports/wf380_scaffold_loss_retrain"
write_candidate_record "wf380_scaffold_loss" \
    "${WF380_SCRIPT}" \
    "${WF380_REPORT}" \
    "Wf-380 SCAFFOLD-loss trainer wire-in — 14 CFM + 3 DMPNN NEW CLI flags; Wf-388-A2/A3 + Wf-405-A2 + Wf-408-A2 wire-in series; requires §5.7 50-step CPU smoke per AGENTS.md §5.7."

# Candidate 2: Wf-394 Lambda CFM coupling (trainer wire-in surface).
WF394_SCRIPT="${REPO_ROOT}/scripts/launch_wf394_lambda_cfm.sh"
WF394_REPORT="${REPO_ROOT}/molmetal/reports/wf394_lambda_cfm_retrain"
write_candidate_record "wf394_lambda_cfm" \
    "${WF394_SCRIPT}" \
    "${WF394_REPORT}" \
    "Wf-394 Lambda × CFM coupling trainer wire-in — 3 NEW CLI flags (--lambda-cfm-coupling + --lambda-cfm-coupling-weight + --lambda-cfm-coupling-samples); Wf-406-A2 + Wf-410-A2 + Wf-414-A2 + Wf-418-A2 patches; requires §5.7 50-step CPU smoke."

# Candidate 3: Wf-246-C metal-coordination term retrain.
WF246C_SCRIPT="${REPO_ROOT}/scripts/launch_wf409a3_metal_coord_retrain.sh"
WF246C_REPORT="${REPO_ROOT}/molmetal/reports/wf409a3_metal_coord_retrain"
write_candidate_record "wf246c_metal_coord" \
    "${WF246C_SCRIPT}" \
    "${WF246C_REPORT}" \
    "Wf-246-C metal-coordination term retrain — kcal sign fix verified at metal_coord_term.py:635; Wf-409-A3 SCAFFOLD script + launch wrapper SHIPPED per §5.5; wrapper does NOT accept --metals / --seeds literal CLI; corrected-interface dry-run required."

# Candidate 4: Wf-125 CONTROLD PAIR (augmented-vs-real smoke cohort).
WF125_SCRIPT="${REPO_ROOT}/scripts/launch_wf125_p1_controlled_pair.sh"
WF125_REPORT="${REPO_ROOT}/molmetal/reports/wf125_p1_controlled_pair"
write_candidate_record "wf125_controlled_pair" \
    "${WF125_SCRIPT}" \
    "${WF125_REPORT}" \
    "Wf-125 P1 CONTROLD PAIR retrain — 5-metal × 5-seed × 5-fold CV; Wf-387-A2 balanced 500 cohort; Wf-388-A1 per-cohort config axis; wrapper does NOT accept --balanced-cohort flag (BLOCKER 3 per Wf-419-A5); corrected-interface dry-run required."

# Candidate 5: Wf-390-A1 m2 R² Path A 5-metal subset retrain.
WF390_SCRIPT="${REPO_ROOT}/scripts/launch_wf390a1_m2_r2_path_a.sh"
WF390_REPORT="${REPO_ROOT}/molmetal/reports/wf390a1_m2_r2_path_a"
write_candidate_record "wf390_m2_r2_path_a" \
    "${WF390_SCRIPT}" \
    "${WF390_REPORT}" \
    "Wf-390-A1 m2 R² Path A 5-metal subset (Ru/Ir/Rh/Os/Re) × 3-seed × 50-epoch retrain — Wf-405-A2 wire-in INCOMPLETE per Wf-424-A11 §5.7 smoke FAIL (NameError on args at train_dmpnn_multitask_wasserstein.py:800); awaiting Wf-424-A12 wire-in fix."

# Candidate 6: Wf-299 3-wave CFM (Wave A + Wave B + Wave C).
WF299_SCRIPT="${REPO_ROOT}/scripts/launch_wf299_cfm_main.sh"
WF299_REPORT="${REPO_ROOT}/molmetal/reports/wf299_cfm_main"

# Wf-A269 wave-28 (2026-11-06) — the wrapper's byte count is RE-MEASURED AT
# RUN TIME, not quoted from a literal.
#
# THE DEFECT (MEASURED 2026-11-06): the notes string below carried a
# hardcoded `35072 B`.  That was a TRUE reading of the file on 2026-11-02,
# but it had gone STALE — the wrapper on disk is 61231 B (MEASURED
# 2026-11-06: stat -c %s scripts/launch_wf299_cfm_main.sh).  A published
# number that no longer describes the file it names is the §6.269.38 class,
# and this one was in a MACHINE-consumed `notes` field.
# FIX: measure the size and the mode here, so the number cannot drift again.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   WF299_BYTES="35072"   # the hardcoded literal this replaces (2026-11-02)
WF299_BYTES="$(stat -c %s "${WF299_SCRIPT}" 2>/dev/null || echo NOT_MEASURED)"
WF299_MODE="$(stat -c %A "${WF299_SCRIPT}" 2>/dev/null || echo NOT_MEASURED)"
write_candidate_record "wf299_cfm_3wave" \
    "${WF299_SCRIPT}" \
    "${WF299_REPORT}" \
    "Wf-299 3-wave CFM main (Wave A yuelbond_only + Wave B W₂_only + Wave C catherine_only) — Wf-415-A1 transparent-passthrough + Wf-408-A3 --wave alias; WRAPPER FOUND on disk at scripts/launch_wf299_cfm_main.sh per AGENTS.md §5.5 filesystem-truth (${WF299_BYTES} B; executable) || 2026-11-06 CORRECTION (Wf-A269 wave-28) — the byte count in THIS clause is RE-MEASURED AT RUN TIME by this launcher (stat -c %s scripts/launch_wf299_cfm_main.sh = ${WF299_BYTES} B, mode ${WF299_MODE}); it was a hardcoded literal (35072 B) and had gone STALE — MEASURED 2026-11-06: the file on disk is 61231 B. || 2026-11-02 CORRECTION (Wf-A269 wave-3 adjudication) — the byte count above is re-measured on disk (stat -c %s scripts/launch_wf299_cfm_main.sh = 35072 B, mode -rwxr-xr-x). SUPERSEDED RENDERINGS PRESERVED VERBATIM per AGENTS.md §5.5 rule #6: (i) the pre-wave-3 rendering — WRAPPER FOUND on disk at molmetal/scripts/launch_wf299_cfm_main.sh per AGENTS.md §5.5 filesystem-truth (29179 B; executable) — was a TRUE MEASURED reading of the file as it stood on 2026-09-30 17:39 (primary ls -la in TODO/completed/424a6_wf299_wave_a_cuda0.md:16 and 8 sibling closeouts), but it went STALE when Wf-439 moved and extended the wrapper: the file was never COMMITTED under molmetal/scripts/ (git log --all -- molmetal/scripts/launch_wf299_cfm_main.sh returns nothing); the same bytes survive in the committed backup molmetal/scripts/launch_wf299_cfm_main.sh.bak.wf430a1 (29179 B, added at bc29aa90); and the live wrapper own size history is 33578 (bc29aa90) / 33587 / 33588 / 33666 / 34713 / 34920 / 34889 / 34915 (HEAD) / 35072 (this tree) — 29179 B is not a size the live path ever had. || 2026-11-06 CORRECTION (Wf-A269 wave-28) — the entry 35072 (this tree) was a TRUE reading on 2026-11-02 and has since gone STALE; the live size is ${WF299_BYTES} B, re-measured at run time by this launcher (MEASURED 2026-11-06: 61231 B). (ii) the intermediate wave-3 rendering — (35099 B; executable) — carried the CORRECT path but a byte count that traces to NO revision of any file (it overstates the disk by 27 B); it is preserved here only as the immediately-superseded value.; DRY_RUN.log found in wave_a/ + wave_b/ subdirs (not at top-level per readiness contract); BLOCKED per readiness-contract step 3 (DRY_RUN.log expected at top-level, see Wf-430 misfix); wrapper has Wf-415-A1 transparent-passthrough + Wf-408-A3 --wave alias; pre-flight v3 (${PASS_COUNT} PASS / ${FAIL_COUNT} FAIL / ${SKIP_COUNT} SKIP / ${TOTAL_COUNT} TOTAL — MEASURED by this launcher's own pre_flight_check.py --pipeline all --json run; the superseded Wf-424-A1 baseline 114/0/3/117 is preserved VERBATIM per AGENTS.md §5.5 rule #6); closure path = Wf-430-A1 align DRY_RUN.log top-level + Wf-425-A1 actual launch USER_OK-gated."

# ---------------------------------------------------------------------------
# Aggregate via Python aggregator (Wf-426-A3 deliverable #2).
# ---------------------------------------------------------------------------
echo "[INFO] Aggregating readiness across 6 retrain candidates..."
# Wf-A269 wave-25 (2026-11-04): the launcher MEASURED the pre-flight four
# lines above (PASS_COUNT / FAIL_COUNT / SKIP_COUNT / TOTAL_COUNT, from the
# live `pre_flight_check.py --pipeline all --json` run), printed them, and
# gated on FAIL_COUNT — and then DISCARDED them: the aggregator was invoked
# WITHOUT --pre-flight-pass/--pre-flight-total/--pre-flight-skip, so every
# published pre-flight number fell back to the aggregator's hardcoded
# Wf-353-B1/Wf-424-B1 baseline (114 PASS / 117 TOTAL / 3 SKIP) while the
# live `all` pipeline carries 133 gates (MEASURED 2026-11-04: pre_flight_check
# .EXPECTED_GATE_COUNTS["all"] == 133 == len(build_gates("all"))).
# ⇒ the readiness JSON published a HISTORICAL BASELINE as if it were this
# run's measurement.  FIX: pass the measured counts.  The baseline constants
# are PRESERVED VERBATIM in the aggregator (they are still its documented
# fallback, and the aggregator now labels which source it used).
# Wf-A269 wave-26 (2026-11-04) — TWO MORE BARE `python` SITES, found by this
# wave's OWN new test (`test_the_launcher_never_invokes_a_bare_python`, which
# MEASURED them).  The wave-25 patch had swept the PRE-FLIGHT and the
# candidate-record sites but MISSED these two — and a first grep for bare
# `python` MISSED them too, because the pattern it used required a
# preceding character and these sites sit at the START of a line.
# MEASURED (before this edit): 2 command-position bare `python` invocations
# remained, at the aggregator call and at the summary-table heredoc.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   python "${AGGREGATOR}" --candidates-dir "${CANDIDATES_DIR}" --output "${READINESS_JSON}" \
#   python - <<EOF
"${PY}" "${AGGREGATOR}" --candidates-dir "${CANDIDATES_DIR}" --output "${READINESS_JSON}" \
    --pre-flight-pass "${PASS_COUNT}" --pre-flight-total "${TOTAL_COUNT}" --pre-flight-skip "${SKIP_COUNT}"
echo "[INFO] Readiness JSON written to ${READINESS_JSON}"

# ---------------------------------------------------------------------------
# Display summary table — non-blocking, readable.
# ---------------------------------------------------------------------------
"${PY}" - <<EOF
import json, os
with open("${READINESS_JSON}") as f:
    data = json.load(f)
print()
print("=" * 72)
print("Wf-426-A3 RETRAIN READINESS SUMMARY (post-Wf-425 fix series)")
print("=" * 72)
for cand in data["candidates"]:
    status = cand["verdict"]
    mark = "READY" if status == "READY" else "BLOCKED" if status == "BLOCKED" else "WARN"
    print(f"  [{mark:7s}] {cand['name']:32s} {cand['reason']}")
print()
print(f"  pre_flight: {data['pre_flight_pass']}/{data['pre_flight_total']} PASS ({data['pre_flight_pass_pct']:.1f}%)")
print(f"  total candidates: {data['total_candidates']}")
print(f"  ready: {data['ready_count']}    blocked: {data['blocked_count']}    warn: {data['warn_count']}")
print("=" * 72)
EOF

echo ""
echo "[INFO] Wf-426-A3 retrain readiness launcher COMPLETED."
echo "[INFO] Actual GPU launches require explicit operator invocation;"
echo "[INFO] this launcher DOES NOT auto-launch per AGENTS.md §5.1."
echo "[INFO] To launch a single retrain, invoke the wrapper directly with"
echo "[INFO] USER_OK=1 + a per-launch USER_OK waiver, e.g.:"
echo "[INFO]   USER_OK=1 bash scripts/launch_wf390a1_m2_r2_path_a.sh --dry-run"
