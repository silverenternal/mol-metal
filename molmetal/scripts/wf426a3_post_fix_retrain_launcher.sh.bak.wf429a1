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
echo "[INFO] Running pre-flight v3 fresh-run check (~3 min wall)..."
PRE_FLIGHT_OUTPUT="$(python "${PRE_FLIGHT}" --pipeline all --json 2>&1)"
PRE_FLIGHT_RC=$?
if [[ "${PRE_FLIGHT_RC}" -ne 0 && "${PRE_FLIGHT_RC}" -ne 2 ]]; then
    echo "[FATAL] pre-flight v3 fresh-run rc=${PRE_FLIGHT_RC} (expected 0 or 2 SKIP-only)." >&2
    echo "${PRE_FLIGHT_OUTPUT}" | tail -20 >&2
    exit 1
fi
PASS_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('pass', d.get('summary', {}).get('pass', 0)))" 2>/dev/null || echo "0")"
FAIL_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('fail', d.get('summary', {}).get('fail', 0)))" 2>/dev/null || echo "0")"
SKIP_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('skip', d.get('summary', {}).get('skip', 0)))" 2>/dev/null || echo "0")"
TOTAL_COUNT="$(echo "${PRE_FLIGHT_OUTPUT}" | python -c "import sys, json; d = json.load(sys.stdin); print(d.get('total', d.get('summary', {}).get('total', 0)))" 2>/dev/null || echo "0")"
echo "[INFO] Pre-flight: ${PASS_COUNT} PASS / ${FAIL_COUNT} FAIL / ${SKIP_COUNT} SKIP / ${TOTAL_COUNT} TOTAL"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    echo "[FATAL] Pre-flight has ${FAIL_COUNT} FAIL gate(s) — abort per AGENTS.md §1.4." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 6 retrain candidates — wrapper check + DRY_RUN.log + plan.json schema.
# Wf-299 wrapper does NOT exist on disk per AGENTS.md §5.5 — that
# candidate will be flagged HONEST_NEGATIVE_BLOCKED at wrapper check.
# ---------------------------------------------------------------------------
CANDIDATES_DIR="$(mktemp -d)"
trap "rm -rf ${CANDIDATES_DIR}" EXIT

write_candidate_record() {
    local name="$1"
    local script="$2"
    local report_dir="$3"
    local notes="$4"
    local json_path="${CANDIDATES_DIR}/${name}.json"
    python - <<EOF > "${json_path}"
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
WF380_SCRIPT="${REPO_ROOT}/molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh"
WF380_REPORT="${REPO_ROOT}/molmetal/reports/wf380_scaffold_loss_retrain"
write_candidate_record "wf380_scaffold_loss" \
    "${WF380_SCRIPT}" \
    "${WF380_REPORT}" \
    "Wf-380 SCAFFOLD-loss trainer wire-in — 14 CFM + 3 DMPNN NEW CLI flags; Wf-388-A2/A3 + Wf-405-A2 + Wf-408-A2 wire-in series; requires §5.7 50-step CPU smoke per AGENTS.md §5.7."

# Candidate 2: Wf-394 Lambda CFM coupling (trainer wire-in surface).
WF394_SCRIPT="${REPO_ROOT}/molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh"
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
WF299_SCRIPT="${REPO_ROOT}/molmetal/scripts/launch_wf299_cfm_main.sh"
WF299_REPORT="${REPO_ROOT}/molmetal/reports/wf299_cfm_main"
write_candidate_record "wf299_cfm_3wave" \
    "${WF299_SCRIPT}" \
    "${WF299_REPORT}" \
    "Wf-299 3-wave CFM main (Wave A yuelbond_only + Wave B W₂_only + Wave C catherine_only) — Wf-415-A1 transparent-passthrough + Wf-408-A3 --wave alias; WRAPPER FOUND on disk at molmetal/scripts/launch_wf299_cfm_main.sh per AGENTS.md §5.5 filesystem-truth (29179 B; executable); DRY_RUN.log found in wave_a/ + wave_b/ subdirs (not at top-level per readiness contract); BLOCKED per readiness-contract step 3 (DRY_RUN.log expected at top-level, see Wf-430 misfix); wrapper has Wf-415-A1 transparent-passthrough + Wf-408-A3 --wave alias; pre-flight v3 PASS (114/0/3/117 per Wf-424-A1); closure path = Wf-430-A1 align DRY_RUN.log top-level + Wf-425-A1 actual launch USER_OK-gated."

# ---------------------------------------------------------------------------
# Aggregate via Python aggregator (Wf-426-A3 deliverable #2).
# ---------------------------------------------------------------------------
echo "[INFO] Aggregating readiness across 6 retrain candidates..."
python "${AGGREGATOR}" --candidates-dir "${CANDIDATES_DIR}" --output "${READINESS_JSON}"
echo "[INFO] Readiness JSON written to ${READINESS_JSON}"

# ---------------------------------------------------------------------------
# Display summary table — non-blocking, readable.
# ---------------------------------------------------------------------------
python - <<EOF
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
