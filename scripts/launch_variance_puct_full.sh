#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# launch_variance_puct_full.sh — L3 full A/B run for R20 FIX 2 variance-cPUct
# Owner: Hugo Lee
# Per AGENTS.md §5.10 rule 6 (Apache-2.0 header) + §5.9 (single-venv rule).
#
# WHAT THIS LAUNCHES
#   molmetal/scripts/bench_mcts_search_quality.py — the R20 FIX 1 quantitative
#   harness — run with --cpuct-mode variance (the R20 FIX 2 arm) or
#   --cpuct-mode constant (the legacy control arm). The A/B delta on
#   success_rate / mean_depth / mean_wall is the variance-cPUct evidence.
#
# Wf-R26 note: variance-cPUct was FLAG_ONLY until Wf-R26 repaired the
# `node.value_var` read (proof_search.py). This wrapper is the measurement
# surface for that repair; it does NOT touch proof_search.py.
#
# DEVIATION FROM THE G5/G6 CONVENTIONS — READ THIS FIRST
#   G5/G6 wrap GPU trainers. This bench is CPU-ONLY. It has NO --device,
#   NO --cuda-0, NO --no-cuda-graphs, NO --epochs, NO --seeds, NO --output-dir,
#   and NO --user-ok flag. Passing any of those makes argparse exit rc=2 —
#   exactly the Wf-r28 defect that killed all 15 G5 pairs with
#   "unrecognized arguments" before a single epoch ran. Every flag this
#   wrapper passes below was verified against
#   `bench_mcts_search_quality.py`'s argparse via grep before being written.
#   The USER_OK gate itself is preserved verbatim (it is a shell-level env
#   check, not a trainer flag) — see below.
#
# DEGENERACY GUARD (MEASURED, not assumed)
#   The bench REFUSES to emit JSON when mean_depth_reached < min(4, --max-depth)
#   or n_success == 0 (bench_mcts_search_quality.py:582-613). Measured on this
#   host 2026-10-07:
#     - --max-depth 4 --n-simulations 200  -> mean_depth_reached=3.50 -> BENCH_DEGENERATE
#     - --max-depth 5 --n-simulations 400  -> mean_depth_reached=2.00 -> BENCH_DEGENERATE
#     - --max-depth 3 --n-simulations 100  -> mean_depth_reached=3.75 -> OK (20/20)
#   So this wrapper pins --max-depth 3, which lowers the floor to min(4,3)=3
#   and clears the guard on the MEASURED configuration. Raising --max-depth to
#   4 or 5 WITHOUT raising --n-simulations reproduces BENCH_DEGENERATE and
#   emits no JSON. The wrapper treats a degenerate verdict as a FAIL.
#
# WALL BUDGET (MEASURED 2026-10-07 on this 24-core host, CPU-only):
#   -n-targets 10 -n-seeds 2 -n-simulations 100 -> 20 runs in 230.2 s wall
#   (mean_wall 14.388 s/run) => ~11.5 s per run amortised.
#   Defaults below are 20 targets x 2 seeds = 40 runs per arm => PROJECTED
#   ~16 min for the two-arm A/B. That is well inside the ~1 hr envelope; the
#   budget is recorded as 1.0 hr in plan.json.
#
# USAGE
#   USER_OK=1 bash scripts/launch_variance_puct_full.sh --dry-run
#   USER_OK=1 bash scripts/launch_variance_puct_full.sh
#   USER_OK=1 WF35_PROTOCOL_LOCK=1 bash scripts/launch_variance_puct_full.sh
#   USER_OK=1 bash scripts/launch_variance_puct_full.sh --mode constant
#
# AGENTS.md §5.4: no wet-lab work is planned, requested, or implied here.

# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# 2026-10-08 pipefix: drop pipefail (SIGPIPE from forked DataLoader workers
# flushing stdout after trainer exit killed the wrapper silently via `tee`).
# Trainer rc captured via ${PIPESTATUS[0]}; tee SIGPIPE suppressed with `|| true`.
#
# Wf-A269 wave-10 (2026-11-02): the `|| true` half above is what made the rc
# read DEAD — `|| true` runs AFTER the failing pipeline and CLOBBERS
# PIPESTATUS to (0), so the "arm exited rc != 0" branch could never fire and
# this wrapper exited 0 after a crashed trainer.  The repaired shape is at the
# launch site below (scripts/_WRAPPER_TEMPLATE.sh:190-193).
set -uo pipefail
cd /home/hugo/codes/mol-metal

# ---- USER_OK gate (AGENTS.md §5.1 kill_then_adapt; mirrors G5 lines 61-68) ----
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual A/B run (CPU-only)\n' "$0" >&2
    exit 64
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# ---- optional WF35_PROTOCOL_LOCK opt-in gate (Wf-347-A1/A2/B1/B2 + verify-r36) ----
# pre_flight_check.py emits SKIP for the protocol-lock gates unless
# WF35_PROTOCOL_LOCK=1 is in the environment (pre_flight_check.py:7616). The
# lock file configs/locks/protocol_lock.yaml IS present on disk. For a
# paper-grade A/B this wrapper surfaces (but does not force) the opt-in so the
# measurement inherits the locked protocol rather than the ad-hoc one.
PROTOCOL_LOCK="SKIP (unset)"
if [ "${WF35_PROTOCOL_LOCK:-0}" = "1" ]; then
    [ -f "${REPO_ROOT}/configs/locks/protocol_lock.yaml" ] \
        || { echo "FAIL: WF35_PROTOCOL_LOCK=1 but configs/locks/protocol_lock.yaml missing"; exit 1; }
    PROTOCOL_LOCK="ENFORCED (WF35_PROTOCOL_LOCK=1)"
    echo "PROTOCOL_LOCK: ENFORCED via configs/locks/protocol_lock.yaml"
else
    echo "PROTOCOL_LOCK: SKIP — set WF35_PROTOCOL_LOCK=1 to enforce configs/locks/protocol_lock.yaml"
fi

# ---- arg parse (--dry-run, --mode constant|variance) ----
DRY_RUN=0
MODE="variance"
if [ "$#" -gt 0 ]; then
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dry-run)   DRY_RUN=1 ;;
            --mode)      shift; MODE="${1:-variance}" ;;
            --mode=*)    MODE="${1#--mode=}" ;;
            *) echo "WARN: unrecognised arg '$1' ignored" ;;
        esac
        shift
    done
fi
case "${MODE}" in
    variance|constant) : ;;
    *) echo "FAIL: --mode must be 'variance' or 'constant', got '${MODE}'"; exit 1 ;;
esac

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/bench_mcts_search_quality.py"
OUT_BASE="results/variance_puct_full"
LOG_DIR="logs/variance_puct_full"
mkdir -p "${OUT_BASE}" "${LOG_DIR}"
LOG="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# CPU-only bench (bench_mcts_search_quality.py has no --device flag; the only
# torch call in the file pins device="cpu" at line 893). Scale below is the
# MEASURED non-degenerate configuration from the header block.
N_TARGETS="${N_TARGETS:-20}"
N_SEEDS="${N_SEEDS:-2}"
N_SIMULATIONS="${N_SIMULATIONS:-100}"
MAX_DEPTH="${MAX_DEPTH:-3}"
CPUCT_ALPHA="${CPUCT_ALPHA:-0.8}"

COMMON_ARGS=(
    --n-targets "${N_TARGETS}"
    --n-seeds "${N_SEEDS}"
    --n-simulations "${N_SIMULATIONS}"
    --max-depth "${MAX_DEPTH}"
    --top-k 10
    --cpuct-variance-alpha "${CPUCT_ALPHA}"
    --cpuct-mode "${MODE}"
    --out-dir "${OUT_BASE}/${MODE}"
)

# ---- DRY-RUN mode: enumerate the plan, no bench execution ----
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== variance-cPUct L3 full A/B — DRY-RUN (USER_OK=1 confirmed) ==="
    echo "PY: ${PY}"
    echo "SCRIPT: ${SCRIPT}"
    echo "MODE: ${MODE}"
    echo "OUT_BASE: ${OUT_BASE}"
    echo "LOG: ${LOG}"
    echo "N_TARGETS=${N_TARGETS} N_SEEDS=${N_SEEDS} N_SIMULATIONS=${N_SIMULATIONS} MAX_DEPTH=${MAX_DEPTH} CPUCT_ALPHA=${CPUCT_ALPHA}"
    echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
    echo "PROTOCOL_LOCK: ${PROTOCOL_LOCK}"
    echo ""
    echo "--- bench argparse contract (every flag below verified in ${SCRIPT}) ---"
    PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${SCRIPT}" --help >/dev/null 2>&1 \
        && echo "OK: bench --help resolves (import chain + argparse intact)" \
        || { echo "FAIL: bench --help failed"; exit 1; }
    echo ""
    echo "--- A/B arms to execute ---"
    for arm in variance constant; do
        echo "  arm=${arm}  n_runs=$((N_TARGETS * N_SEEDS))  ->  ${OUT_BASE}/${arm}/search_quality_bench.json"
    done
    echo ""

    emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
        --title "variance-cPUct L3 full A/B (variance vs constant)" \
        --wrapper "$(basename "$0")" \
        --mode "${MODE}" \
        --trainer "${SCRIPT}" \
        --flags "--n-targets ${N_TARGETS} --n-seeds ${N_SEEDS} --n-simulations ${N_SIMULATIONS} --max-depth ${MAX_DEPTH} --top-k 10 --cpuct-variance-alpha ${CPUCT_ALPHA} --cpuct-mode ${MODE} --out-dir ${OUT_BASE}/${MODE}" \
        --gpu "cpu-only (bench has no --device flag)" \
        --budget-hr "1" \
        --user-ok "REQUIRED"
    echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log"

    PLAN_JSON="${OUT_BASE}/plan.json"
    {
        printf '{\n'
        printf '  "wrapper": "launch_variance_puct_full.sh",\n'
        printf '  "wf_origin": "L3 full A/B for R20 FIX 2 variance-cPUct (Wf-R26 repair)",\n'
        printf '  "user_ok": true,\n'
        printf '  "dry_run": true,\n'
        printf '  "mode": "%s",\n' "${MODE}"
        printf '  "arms": ["variance", "constant"],\n'
        printf '  "n_targets": %s,\n' "${N_TARGETS}"
        printf '  "n_seeds": %s,\n' "${N_SEEDS}"
        printf '  "n_simulations": %s,\n' "${N_SIMULATIONS}"
        printf '  "max_depth": %s,\n' "${MAX_DEPTH}"
        printf '  "cpuct_variance_alpha": %s,\n' "${CPUCT_ALPHA}"
        printf '  "n_runs_per_arm": %s,\n' "$((N_TARGETS * N_SEEDS))"
        printf '  "device": "cpu-only",\n'
        printf '  "wall_budget_hr": 1.0,\n'
        printf '  "wall_budget_label": "PROJECTED (~16 min total for both arms). MEASURED basis: 10 targets x 2 seeds x 100 sims x depth 3 = 20 runs in 230.2 s wall (mean_wall 14.388 s/run) on this 24-core host, 2026-10-07.",\n'
        printf '  "bench": "%s",\n' "${SCRIPT}"
        printf '  "out_base": "%s",\n' "${OUT_BASE}"
        printf '  "protocol_lock": "%s",\n' "${PROTOCOL_LOCK}"
        printf '  "honest_framing": {\n'
        printf '    "MEASURED_today": "wrapper ready; USER_OK gate enforced; bench argparse verified; both arms enumerated; DRY_RUN.log + plan.json emitted; degeneracy guard calibrated on this host",\n'
        printf '    "DEGENERACY_RISK": "--max-depth 4/5 at these n_simulations returns BENCH_DEGENERATE and emits NO JSON (MEASURED: mean_depth_reached 3.50 at depth 4/sims 200; 2.00 at depth 5/sims 400). Wrapper pins --max-depth 3 to clear the min(4, max_depth) floor and treats a degenerate verdict as FAIL.",\n'
        printf '    "NO_CLAIM": "This wrapper runs the A/B; it makes NO claim about whether variance-cPUct wins. No variance-vs-constant delta has been measured at this scale yet.",\n'
        printf '    "CPU_ONLY": "bench_mcts_search_quality.py exposes no --device/--cuda flag; the only torch call pins device=cpu. This is NOT a GPU launch and consumes no VRAM.",\n'
        printf '    "wf_R26": "variance-cPUct was FLAG_ONLY until Wf-R26 repaired the node.value_var read in proof_search.py; this A/B is its measurement surface"\n'
        printf '  },\n'
        printf '  "status": "PLAN_ONLY"\n'
        printf '}\n'
    } > "${PLAN_JSON}"
    echo "plan.json emitted -> ${PLAN_JSON}"
    echo ""
    echo "DRY-RUN OK: 2 arms (variance, constant) x ${N_TARGETS} targets x ${N_SEEDS} seeds PLANNED; no bench work executed."
    exit 0
fi

# ---- Pre-launch gates (real run only) ----
echo "=== variance-cPUct L3 full A/B START $(date) ===" | tee -a "${LOG}"
echo "MODE=${MODE} PROTOCOL_LOCK=${PROTOCOL_LOCK} ARGS=${COMMON_ARGS[*]}" | tee -a "${LOG}"

# AGENTS.md §4 pre-flight v2 gate. --pipeline launch = 'all' + pre-retrain gates.
# L3 is CPU-only VariancePUCT bench (no metal cohort); declare a virtual
# 'L3' metal so the wf343c2 grid check resolves a non-empty (metal, seed)
# grid (n_pairs=1*${N_SEEDS} -> REQUIRED_HR=1*${N_SEEDS}*0.7*1.25).
setsid nohup PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py \
    --pipeline launch \
    --declared-wave "L3 variance-cPUct full A/B (${MODE} arm)" \
    --pairs-metals L3 \
    --pairs-seeds "${N_SEEDS}" \
    --budget-hr 1 \
    2>&1 | tee -a "${LOG}" || { echo "FAIL: pre_flight_check"; exit 1; }

# AGENTS.md §5.1 wf343c2 per-wave runtime gate. Diagnostic-only (does NOT
# abort): the G5 launcher made the same call at lines 335-357 — the 2 known
# project-wide pre-flight FAILs (CFM Wasserstein loss file MISSING + Vina not
# installed) are pre-existing and unrelated to a CPU MCTS bench. The JSON
# evidence is captured for post-run audit.
rc_c2=0
setsid nohup PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "L3-variance-puct-${MODE}" \
    --pairs-metals L3 \
    --pairs-seeds "${N_SEEDS}" \
    --budget-hr 1 \
    --skip-live > "${OUT_BASE}/wf343c2_${MODE}.log" 2>&1 || rc_c2=$?
echo "PER_WAVE_CHECK ${MODE} rc=${rc_c2} log=${OUT_BASE}/wf343c2_${MODE}.log" | tee -a "${LOG}"
echo "(diagnostic-only; rc!=0 does not abort the run)"

# ---- Real run: the selected arm ----
run_arm() {
    local arm="$1"
    echo "=== $(date '+%H:%M:%S') START arm=${arm} ===" | tee -a "${LOG}"
    local arm_start_s
    arm_start_s=$(date +%s)
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #   setsid nohup     PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${SCRIPT}" \
    #       --n-targets "${N_TARGETS}" \
    #       --n-seeds "${N_SEEDS}" \
    #       --n-simulations "${N_SIMULATIONS}" \
    #       --max-depth "${MAX_DEPTH}" \
    #       --top-k 10 \
    #       --cpuct-variance-alpha "${CPUCT_ALPHA}" \
    #       --cpuct-mode "${arm}" \
    #       --out-dir "${OUT_BASE}/${arm}" 2>&1 | tee "${OUT_BASE}/${arm}_run.log" || true   # 2026-10-08 pipefix
    #   local rc=${PIPESTATUS[0]}
    #
    # Canonical shape (scripts/_WRAPPER_TEMPLATE.sh:190-193): pipefail off for
    # this ONE pipeline (a trainer whose forked DataLoader workers still hold
    # the pipe open takes SIGPIPE and would report 141 instead of its own
    # status — the tee stays, it is the only diagnostic); `setsid --wait`
    # (bare setsid may fork and return 0 before the trainer has done anything,
    # which would make PIPESTATUS[0] report success for a run that had not
    # started — and setsid gives the trainer its own session, which already
    # covers nohup's SIGHUP immunity); and the rc read BETWEEN the two `set`
    # lines, while PIPESTATUS still holds the pipeline's statuses.
    #
    # The env assignment moved in FRONT of setsid (template form): `setsid`
    # does NOT accept `VAR=value` arguments — it execs the first argument as
    # the program, so the old form died with
    #   "nohup: failed to run command 'PYTHONPATH=/home/hugo/codes/mol-metal'"
    # and rc=127, which the `|| true` then reported as a SUCCESS (rc=0).
    set +o pipefail
    PYTHONPATH=/home/hugo/codes/mol-metal setsid --wait ${PY} "${SCRIPT}" \
        --n-targets "${N_TARGETS}" \
        --n-seeds "${N_SEEDS}" \
        --n-simulations "${N_SIMULATIONS}" \
        --max-depth "${MAX_DEPTH}" \
        --top-k 10 \
        --cpuct-variance-alpha "${CPUCT_ALPHA}" \
        --cpuct-mode "${arm}" \
        --out-dir "${OUT_BASE}/${arm}" 2>&1 | tee "${OUT_BASE}/${arm}_run.log"
    local rc=${PIPESTATUS[0]}
    set -o pipefail
    local arm_elapsed_s=$(($(date +%s) - arm_start_s))
    echo "ARM_BUDGET_END ${arm} elapsed_s=${arm_elapsed_s} rc=${rc}" | tee -a "${LOG}"
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: arm ${arm} exited rc=${rc}" | tee -a "${LOG}"
        return "$rc"
    fi
    # Degeneracy guard: the bench exits 0 but emits NO JSON when the search
    # fails to clear min(4, max_depth). Treat a missing JSON as a FAIL so a
    # degenerate run can never be mistaken for a measured result.
    local json="${OUT_BASE}/${arm}/search_quality_bench.json"
    if [ ! -f "${json}" ]; then
        echo "FAIL: arm ${arm} emitted no JSON at ${json} — bench verdict was BENCH_DEGENERATE" | tee -a "${LOG}"
        return 1
    fi
    ${PY} -c "
import json
d = json.load(open('${json}'))
print('ARM_SUMMARY ${arm} verdict=%s success_rate=%s mean_depth=%s mean_wall=%s' % (
    d.get('verdict'), d.get('success_rate'), d.get('mean_depth'), d.get('mean_wall')))
" | tee -a "${LOG}"
    return 0
}

run_arm "${MODE}"

# ---- A/B comparison stub when both arms are present ----
# Wf-R38 fix: bench JSON nests aggregate metrics under d['agg']; previous
# code read d.get('success_rate') etc. at top level which returned None.
# Read from agg + map field names to match the stdout line summary.
if [ -f "${OUT_BASE}/variance/search_quality_bench.json" ] && [ -f "${OUT_BASE}/constant/search_quality_bench.json" ]; then
    ${PY} -c "
import json
from pathlib import Path
out = {}
for arm in ('variance', 'constant'):
    p = Path('${OUT_BASE}') / arm / 'search_quality_bench.json'
    d = json.load(open(p))
    a = d.get('agg', {})
    out[arm] = {
        'verdict': d.get('verdict'),
        'success_rate': a.get('success_rate'),
        'mean_depth': a.get('mean_depth_reached'),
        'mean_n_distinct': a.get('mean_n_distinct'),
        'mean_wall_s': a.get('mean_wall_time_s'),
        'wall_total_s': a.get('total_wall_time_s'),
        'tt_cache_hit_rate': a.get('tt_cache_hit_rate'),
    }
Path('${OUT_BASE}/ab_comparison.json').write_text(json.dumps(out, indent=2))
# raw per-arm metrics only — NO delta claim, NO paper-grade verdict.
print('ab_comparison.json written (raw per-arm metrics; NO delta claim)')
"
else
    echo "NOTE: only the ${MODE} arm has JSON; run the other arm to populate ${OUT_BASE}/ab_comparison.json"
fi | tee -a "${LOG}"

echo "=== variance-cPUct L3 full A/B DONE $(date) ===" | tee -a "${LOG}"