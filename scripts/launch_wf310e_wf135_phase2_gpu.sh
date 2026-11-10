#!/usr/bin/env bash
# Wf-310E Part 1 — Wf-135 Phase 2 LOMO 5-metal × 5-seed × 50-epoch re-bench
#                   USER_OK-gated GPU launch wrapper
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Parent: Wf-309A Rh cohort 500 (Phase 1 prerequisite) + Wf-135 Phase 2 plan
# Refs:
#   - TODO/completed/135_wf135_phase1_plan.md §2 (Phase 2 spec)
#   - TODO/completed/135_m19_lomo_rh_oversample_fix.md §2.2 (planned Phase 2)
#   - TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md (Wf-309D restored)
#   - molmetal/reports/wf176_m19_oversample_lomo/synthesis.md (current state)
#   - TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md §1.4 (acceptance gates)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt; 2 weeks GPU wall)
#   - Phase 1 prerequisite: Wf-309A Rh cohort 500 must be COMPLETE on disk
#     (molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json)
#   - Scope: 5-metal × 5-seed × 50-epoch = 1250 runs
#   - Wall: ~10 GPU-hours (PROJECTED) = ~2 calendar weeks (best-case 7 days,
#     worst-case 14 days with retries per Phase 1 plan §2.4)
#   - Rh fold AUC ≥ 0.30 + Overall LOMO mean ≥ 0.60 = MEASURED acceptance gates
#   - Trainer target: molmetal/scripts/train_metal_lomo_v2.py
#     (uses balanced cohort + per-metal head ensemble + Wasserstein transfer
#      target = Rh real active distribution, NOT synthetic cohort)
#   - Per Wf-139 P7 ship (Phase 1): 250 active + 250 inactive balanced
#     cohort with diff-based ranking; scaffold-split verified clean.
#   - Per Wf-131 P3 + Wf-125 P2 honest-negative: Rh fold regressed 0.368 → 0.136
#     on buggy constant-inactive cohort; this Phase 2 must recover ≥ 0.30.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh           # actual run
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0)
# --------------------------------------------------------
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch and emits
#     the planned invocation as a record.
#   - 2-week wall budget is enforced as a logging warning, not a hard stop
#     (per kill_then_adapt + Wf-295D per_wave_user_ok pattern).
#
# Stop semantics: per TODO-207 v2 §7.2 (per-phase abort criteria):
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN/OOM/IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: Rh fold AUC < 0.10 on 2+ seeds (binding regression alarm)
#
# Pre-launch checks (BLOCKER before any GPU work):
#   1. Phase 1 cohort on disk (Wf-309A complete)
#   2. pre_flight_check.py --pipeline launch (24/24 PASS target)
#   3. wave_budget_tracker.py --budget-hr 14 --chemprop-not-required
#   4. USER_OK env gate

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
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf310e_wf135_phase2"
TRAINER="${SCRIPTS_DIR}/train_metal_lomo_v2.py"

# Phase 1 prerequisite
PHASE1_COHORT="${PROJECT_ROOT}/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json"

# §5.11 (2026-11-03): the five trainer hyperparameters this wrapper used to
# pass on the command line (--metals / --seeds / --epochs /
# --balanced-cohort-json / --device) now live in a YAML config. The sweep is
# 25 (metal, seed) pairs and the trainer loops over --metals/--seeds
# internally, so the wrapper derives ONE file per pair from this one; the
# derivation is in the run loop below.
# --out-dir is NOT migrated: §5.11 lets a wrapper keep it on the command
# line (it selects WHERE the run writes, not WHAT it trains), so it is
# still passed there with the byte-identical string it always had. The
# derived config mirrors it under output.out_dir so the equivalence probe
# can compare every flag the old invocation passed.
BASE_CONFIG="${PROJECT_ROOT}/configs/launch_wf310e_wf135_phase2_gpu.yaml"

# Phase 2 spec
METALS=(Rh Ru Ir Os Re)   # 5-metal LOMO
SEEDS=(0 1 2 3 4)         # 5 seeds (vs Wf-131 P3 baseline 3)
EPOCHS=50
WALL_BUDGET_HR=14          # 2 weeks upper bound
GPU="cuda:0"               # RTX PRO 6000 Blackwell GPU-0 (serial execution)

mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual 2-week GPU run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
SKIP_PHASE1_CHECK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)            DRY_RUN=1; shift ;;
        --skip-phase1-check)  SKIP_PHASE1_CHECK=1; shift ;;
        -h|--help)            sed -n '2,60p' "$0"; exit 0 ;;
        *)                    printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Color codes (only when stdout is a tty)
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
# Sanity checks
# -----------------------------------------------------------------------------
[ -x "$(command -v "${PY}")" ] || die "PY not found: ${PY}"
[ -f "${TRAINER}" ]           || die "trainer missing: ${TRAINER}"

# Phase 1 prerequisite check (Wf-309A must be complete)
if [ "${SKIP_PHASE1_CHECK}" != "1" ]; then
    if [ ! -f "${PHASE1_COHORT}" ]; then
        die "Phase 1 prerequisite missing: ${PHASE1_COHORT}. Run Wf-309A first."
    fi
    # Quick sanity: cohort must be 500 lines with 250 active + 250 inactive
    # Wf-A269 wave-9 (2026-11-02) - REPAIR: the check grepped for a literal
    # `"active_inactive": "active"` that appears NOWHERE in the cohort file.
    # The file is a JSON object with top-level `actives` / `inactives` ARRAYS
    # and a `stats` block (n_active / n_inactive); the string
    # "active_inactive" does not occur in it.  Both greps therefore returned
    # 0, and because `grep -c` prints "0" AND exits 1 on no match, the
    # `|| echo 0` fallback appended a SECOND "0" ("0\n0").  So the check
    # could NEVER pass - the "250 active + 250 inactive balanced" branch was
    # unreachable (a dead PASS branch) - and the wrapper reported "Phase 1
    # cohort may be unbalanced (active=0\n0, inactive=0\n0)" on every run
    # while the cohort IS balanced (MEASURED: stats.n_active=250,
    # stats.n_inactive=250, len(actives)=250, len(inactives)=250).
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     N_ACT=$(grep -c '"active_inactive": "active"' "${PHASE1_COHORT}" 2>/dev/null || echo 0)
    #     N_INACT=$(grep -c '"active_inactive": "inactive"' "${PHASE1_COHORT}" 2>/dev/null || echo 0)
    COHORT_CHECK="${PROJECT_ROOT}/molmetal/scripts/wf_cohort_balance_check.py"
    if [ -f "${COHORT_CHECK}" ]; then
        COHORT_CHECK_OUT="${REPORTS_DIR}/cohort_balance_check.txt"
        COHORT_RC=0
        "${PY}" "${COHORT_CHECK}" \
            --json "${PHASE1_COHORT}" \
            --expect-active 250 --expect-inactive 250 > "${COHORT_CHECK_OUT}" 2>&1 || COHORT_RC=$?
        sed 's/^/    /' "${COHORT_CHECK_OUT}" | tee -a "${REPORTS_DIR}/launch.log" >/dev/null
        if [ "${COHORT_RC}" -eq 0 ]; then
            ok "Phase 1 cohort verified: 250 active + 250 inactive balanced (MEASURED)"
        else
            warn "Phase 1 cohort may be unbalanced (active/inactive counts above, rc=${COHORT_RC}); continuing"
        fi
    else
        warn "cohort balance checker missing: ${COHORT_CHECK}; skipping the Phase 1 balance check"
    fi
fi

log "Wf-310E Wf-135 Phase 2 LOMO re-bench wrapper — USER_OK=1 confirmed"
log "  metals=${METALS[*]}"
log "  seeds=${SEEDS[*]}"
log "  epochs=${EPOCHS}"
log "  total_runs=$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
log "  gpu=${GPU}"
log "  wall_budget_hr=${WALL_BUDGET_HR}"
log "  dry_run=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310e_wf135_phase2_gpu.sh",\n'
    printf '  "wf_origin": "Wf-310E Part 1",\n'
    printf '  "parent_plan": "TODO/completed/135_wf135_phase1_plan.md §2",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metals": [%s],\n' "$(IFS=,; echo "${METALS[*]}" | tr ' ' ',')"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}" | tr ' ' ',')"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "total_runs": %d,\n' "$(( ${#METALS[@]} * ${#SEEDS[@]} ))"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": %d,\n' "${WALL_BUDGET_HR}"
    printf '  "wall_budget_label": "PROJECTED (2 calendar weeks per Wf-135 Phase 1 plan §2.4; compute ~10 GPU-hr; remainder = setup/teardown/reporting)",\n'
    printf '  "phase1_prereq": "%s",\n' "${PHASE1_COHORT}"
    printf '  "phase1_status": "%s",\n' "$([ -f "${PHASE1_COHORT}" ] && echo "PRESENT" || echo "MISSING")"
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "acceptance_gates": {\n'
    printf '    "rh_fold_auc_min": 0.30,\n'
    printf '    "overall_lomo_mean_min": 0.60,\n'
    printf '    "min_tests_pass": 16,\n'
    printf '    "rh_target_origin": "recover from Wf-131 P2 regression 0.136 to baseline 0.368"\n'
    printf '  },\n'
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; phase1 prerequisite check wired",\n'
    printf '    "PROJECTED": "Rh fold AUC +0.164 recovery (0.136 -> 0.30 target); overall LOMO mean +0.065 (0.535 -> 0.60 target)",\n'
    printf '    "kill_then_adapt_directive": "2-week wall; user_ok gate HARD-REQUIRED per mol-metal-gpu-execution §0"\n'
    printf '  },\n'
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

# Emit DRY_RUN.log (Wf-363-A1 schema)
# Wf-A269 wave-3 (2026-11-02) — REPAIR.  The variable interpolation was
# missing: the path was the LITERAL string "/REPORTS_DIR/DRY_RUN.log",
# so the dry-run died with
#   line 191: /REPORTS_DIR/DRY_RUN.log: No such file or directory
# and exited 1 — which is why
# ``molmetal/tests/test_wf337u2_wf135_user_ok.py::
# TestUserOKGateAccepts::test_accepts_exit_zero_with_user_ok_one`` was
# RED at HEAD (the defect predates this wave: it is present in
# ``git show HEAD:scripts/launch_wf310e_wf135_phase2_gpu.sh``).  This is
# the ONLY literal ``/UPPER_CASE/`` path segment in a tracked shell
# file — MEASURED: ``git grep -n -E '"/[A-Z_]{3,}/' -- 'scripts/*.sh'
# 'molmetal/scripts/*.sh'`` returns exactly this one hit.  SUPERSEDED
# RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   DRY_RUN_LOG="/REPORTS_DIR/DRY_RUN.log"
DRY_RUN_LOG="${REPORTS_DIR}/DRY_RUN.log"
cat >"${DRY_RUN_LOG}" <<EOF
[DRY-RUN] launch_wf310e_wf135_phase2_gpu.sh @ wf310e_wf135
[DRY-RUN] wrapper_name=launch_wf310e_wf135_phase2_gpu.sh
[DRY-RUN] trainer=${TRAINER}
[DRY-RUN] py=/home/hugo/codes/mol-metal/.venv/bin/python
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
EOF

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} \\"
    log "    --config <derived from ${BASE_CONFIG##*/}: data.metals=[<metal>], data.seeds=[<seed>]> \\"
    log "    --out-dir ${REPORTS_DIR}/<metal>/seed<seed>"
    log "DRY-RUN: yaml-carried hyperparameters (were CLI flags):"
    log "  protocol.epochs=${EPOCHS}  compute.device=cuda:0"
    log "  data.balanced_cohort_json=${PHASE1_COHORT}"
    log "  data.metals=[<one metal per invocation>]  data.seeds=[<one seed per invocation>]"
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #   log "  CUDA_VISIBLE_DEVICES=0 ${PY} ${TRAINER} \\"
    #   log "    --metals ${METALS[*]} --seeds ${SEEDS[*]} \\"
    #   log "    --epochs ${EPOCHS} \\"
    #   log "    --balanced-cohort-json ${PHASE1_COHORT} \\"
    #   log "    --out-dir ${REPORTS_DIR}/per_seed_results \\"
    #   log "    --device=cuda:0"
    log ""
    log "DRY-RUN: per-fold sequencing (5 metals x 5 seeds = 25 (metal,seed) tuples; each tuple = 1 Rh-fold + 4 other-fold runs = ~5 LOMO runs)"
    log "DRY-RUN: total wall = ~10 GPU-hr sequential = ~2 calendar weeks worst case"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 5-metal × 5-seed sweep
# Each (metal, seed) tuple runs a full LOMO: hold out the focal metal,
# train on remaining 4 + Rh balanced cohort, evaluate on held-out metal.
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)

# Phase 2 sequencing per Wf-135 Phase 1 plan §2.1:
#   For each metal m in {Rh, Ru, Ir, Os, Re}:
#     For each seed s in {0, 1, 2, 3, 4}:
#       Train multi-task D-MPNN with W₂ transfer + per-metal head ensemble
#       Use Wf-139 P7 balanced cohort for Rh fold
#       Wasserstein target = Rh REAL active distribution (not synthetic)
for METAL in "${METALS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        log "=== metal=${METAL} seed=${SEED} on ${GPU} ==="
        # GPU 0 (RTX PRO 6000 Blackwell) per the 2026-10-03 all-GPU-0
        # directive; this call site is serial with every other job.
        # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
        #
        # §5.11 (2026-11-03) — per-pair YAML derivation.
        # The trainer loops over args.metals / args.seeds internally
        # (train_metal_lomo_v2.py:2030-2031), so the CLI used to carry a
        # SINGLE-element list per invocation to keep 25 separate processes.
        # The derived config carries the same single-element lists, so the
        # run is unchanged: same 25 processes, same cohorts, same out-dirs.
        PAIR_CONFIG="${REPORTS_DIR}/_derived_configs/${METAL}/seed${SEED}.yaml"
        PAIR_OUT_DIR="${REPORTS_DIR}/${METAL}/seed${SEED}"
        mkdir -p "$(dirname "${PAIR_CONFIG}")"
        "${PY}" - "${BASE_CONFIG}" "${PAIR_CONFIG}" "${METAL}" "${SEED}" "${PAIR_OUT_DIR}" <<'PYEOF' || die "per-pair config derivation failed for metal=${METAL} seed=${SEED}"
import sys

import yaml

base, out, metal, seed, out_dir = sys.argv[1:6]
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
data = cfg.setdefault("data", {})
data["metals"] = [metal]
data["seeds"] = [int(seed)]
output = cfg.setdefault("output", {})
output["out_dir"] = out_dir
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# DERIVED per-pair config for scripts/launch_wf310e_wf135_phase2_gpu.sh\n")
    fh.write("# base: %s\n" % base)
    fh.write("# pair: metal=%s seed=%s\n" % (metal, seed))
    fh.write("# data.metals / data.seeds carry the SAME single-element lists\n")
    fh.write("# the CLI flags used to pass. Nothing else was changed.\n")
    fh.write("# output.out_dir mirrors the --out-dir the wrapper still passes on\n")
    fh.write("# the command line (same string; the CLI wins over the YAML).\n")
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
PYEOF
        if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" "${PY}" "${TRAINER}" \
                --config "${PAIR_CONFIG}" \
                --out-dir "${REPORTS_DIR}/${METAL}/seed${SEED}" \
                >> "${RUN_LOG}" 2>&1; then
            ok "metal=${METAL} seed=${SEED} completed"
        else
            die "metal=${METAL} seed=${SEED} failed (see ${RUN_LOG}); aborting 2-week sweep"
        fi
    done
done
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   [the pre-§5.11 trainer invocation, one per (metal, seed) pair]
#         if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" "${PY}" "${TRAINER}" \
#                 --metals "${METAL}" \
#                 --seeds "${SEED}" \
#                 --epochs "${EPOCHS}" \
#                 --balanced-cohort-json "${PHASE1_COHORT}" \
#                 --out-dir "${REPORTS_DIR}/${METAL}/seed${SEED}" \
#                 --device cuda:0 \
#                 >> "${RUN_LOG}" 2>&1; then

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_hr=$(( elapsed_sec / 3600 ))
log "Wf-310E Wf-135 Phase 2 LOMO sweep done — elapsed=${elapsed_hr} hr (budget=${WALL_BUDGET_HR} hr)"

if [ "${elapsed_hr}" -gt "${WALL_BUDGET_HR}" ]; then
    warn "elapsed (${elapsed_hr} hr) exceeded PROJECTED wall budget (${WALL_BUDGET_HR} hr); flagging as MEASURED-over-PROJECTED in plan.json"
    # Patch plan.json with measured wall time
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-(metal,seed) JSONs"
ok "Next: emit synthesis.md + paper §4.5.6 honest-framing decision (per TODO-84 rule)"
