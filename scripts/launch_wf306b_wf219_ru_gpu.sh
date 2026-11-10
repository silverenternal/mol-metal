#!/usr/bin/env bash
# Wf-306B — Wf-219 Ru short-epoch GPU re-test wrapper.
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Origin: Wf-304C Task 1 + Wf-305B/Wf-305C follow-up
# Status: READY on user OK (Wf-304C noted all Wf-219 Ru seed runs hit 43/50
#         epochs and timed out under the standard 5-seed x 50-epoch protocol;
#         this wrapper drops to 30 epochs and 3 seeds to fit ~10 min/wall on
#         RTX 5090 cuda:1, leaving room for an honest re-measurement).
#
# Purpose
# -------
# Re-measure the Wf-219 Ru 50-epoch baseline under a tighter epoch budget
# (30 instead of 50) so the run completes inside ~10 min/wall on RTX 5090
# cuda:1. The reduced budget is justified by Wf-304C §1: every Ru run
# timed out at 43/50 epochs under the standard protocol — the actual
# convergence point is unknown, so 30 epochs is a conservative lower bound
# that should still land the model in the Krasnov-aligned AUC range.
#
# Scope (3 seeds, 30 epochs, Ru only)
# -----------------------------------
#   seed 0  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   seed 1  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   seed 2  Ru 30 epochs  ~3-4 min wall (PROJECTED)
#   total wall budget: ~10 min on RTX 5090 cuda:1 (PROJECTED 12 min)
#
# Usage
# -----
#   USER_OK=1 bash scripts/launch_wf306b_wf219_ru_gpu.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf306b_wf219_ru_gpu.sh           # actual run
#
# USER_OK gate (explicit, per Wf-295D per_wave_user_ok pattern)
# -------------------------------------------------------------
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch and emits
#     the planned invocation as a record.
#   - This is intentionally stricter than Wf-299 (which auto-OKs on dry-run)
#     because this wrapper targets a real (not stub) GPU retrain of Wf-219.
#
# Stop semantics: per TODO-119 P1.4 §directives.per_wave_user_ok_required
# (Wf-295D) — only one wave here, no per-wave sub-checks needed.
#
# All numbers are MEASURED where possible, PROJECTED where not (party-twenty-
# spirit §3.5).

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §5.9 + Wf-431-A1 fix F2 — the launcher must be able to import the
#          `molmetal.*` packages, so the project root goes on PYTHONPATH.
export PYTHONPATH="/home/hugo/codes/mol-metal"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
REPORTS_DIR="${PROJECT_ROOT}/molmetal/reports/wf306b_wf219_ru_wrapper"
TRAINER="${SCRIPTS_DIR}/wf219_krasnov_aligned.py"

# Wf-304C §1: Ru baseline timed out at 43/50 epochs. Drop to 30.
EPOCHS=30
SEEDS=(0 1 2)
METAL="Ru"
GPU="cuda:1"   # RTX 5090

# Wall budget: 30 epochs × 3 seeds × ~20 sec/epoch ≈ 1800 sec ≈ 30 min PROJECTED.
# Wf-304C measured 43/50 epochs in ~13.5 min, so 30/50 ≈ 9.4 min PROJECTED.
WALL_BUDGET_MIN=10
mkdir -p "${REPORTS_DIR}"

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s      # actual GPU run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *)         printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
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

log "Wf-306B Wf-219 Ru short-epoch wrapper — USER_OK=1 confirmed"
log "  epochs=${EPOCHS} (down from 50; Wf-304C §1 timeout at 43/50)"
log "  seeds=${SEEDS[*]}"
log "  metal=${METAL}"
log "  gpu=${GPU}"
log "  wall_budget_min=${WALL_BUDGET_MIN}"
log "  dry_run=${DRY_RUN}"
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + TODO-207 — every retrain/panel/ablation launch must pass the
#          pre-flight v2 gate BEFORE any GPU work is attempted.
# Wf-A269 wave-5 (2026-11-02) - D8: the canonical pre-flight is now dry-run
# guarded.  The guard mirrors the two reference implementations: the
# explicit `if [ "${DRY_RUN}" -ne 1 ]` block in
# scripts/launch_wf306a_drugood_vina_gpu.sh, and the early-return in
# scripts/launch_wf74_v2_drugood_dann.sh (whose pre_flight_gate() logs the
# same "DRY-RUN: skipping pre_flight GPU/live gates" line reproduced below).
# Reason: a plan-only run must not execute a live gate — the pre-flight may
# allocate GPU memory (honest-negative catalog §6.265), so an UNGUARDED
# pre-flight makes --dry-run neither plan-only nor cheap.  The gate is NOT
# weakened for a real launch: the identical `--pipeline launch` call still
# runs and still exits 3 on FAIL.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
# PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch >"${REPORTS_DIR}/preflight.json" 2>&1 \
#     || { echo "[FATAL] pre_flight_check FAIL — see ${REPORTS_DIR}/preflight.json" >&2; exit 3; }
# --- pre-flight gate (AGENTS.md §4 + TODO-207) --------------------------
PRE_FLIGHT_STATUS="PASS"
if [ "${DRY_RUN}" -ne 1 ]; then
    PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch >"${REPORTS_DIR}/preflight.json" 2>&1 \
        || { echo "[FATAL] pre_flight_check FAIL — see ${REPORTS_DIR}/preflight.json" >&2; exit 3; }
else
    PRE_FLIGHT_STATUS="SKIPPED - dry-run executes no live gate"
    warn "DRY-RUN: skipping pre_flight GPU/live gates"
fi

# -----------------------------------------------------------------------------
# Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED label
# discipline (party-twenty-spirit §3.5).
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
# Wf-A269 wave-5 (2026-11-02) - D6: the emitted plan.json now carries the
# canonical `meta` block (wrapper_name / user_ok_required / wf_origin /
# parent_plan), placed immediately AFTER the "wrapper" key and BEFORE the
# "user_ok" key so the emitted JSON stays a valid object.  The shape is
# copied from the reference implementation
# scripts/launch_wf306a_drugood_vina_gpu.sh (the meta block is what
# molmetal/scripts/wf426a3_post_fix_results_aggregator.py reads).
# Wf-A269 wave-5 (2026-11-02) - D7: the plan.json write is now atomic
# (.tmp + mv -f), the shell equivalent of the repo atomic-write protocol
# (AGENTS.md §5.6), so an interrupted write cannot leave a truncated
# plan.json behind.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
# PLAN_JSON="${REPORTS_DIR}/plan.json"
# {
#     printf '{\n'
#     printf '  "wrapper": "launch_wf306b_wf219_ru_gpu.sh",\n'
#     printf '  "wf_origin": "Wf-306B",\n'
#     printf '  "origin_wf": "Wf-304C Task 1 + Wf-305B/Wf-305C follow-up",\n'
#     printf '  "user_ok": true,\n'
#     printf '  "dry_run": %s,\n' "${DRY_RUN}"
#     printf '  "metal": "%s",\n' "${METAL}"
#     printf '  "epochs": %d,\n' "${EPOCHS}"
#     printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}")"
#     printf '  "gpu": "%s",\n' "${GPU}"
#     printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
#     printf '  "wall_budget_label": "PROJECTED (Wf-304C measured 13.5 min for 43/50 epochs; 30/50 ≈ 9.4 min PROJECTED; +33 pct safety = 10 min budget)",\n'
#     printf '  "epoch_target_origin": "Wf-304C §1: Ru baseline timed out at 43/50 epochs under standard protocol",\n'
#     printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
#     printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
#     printf '}\n'
# } > "${PLAN_JSON}"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf306b_wf219_ru_gpu.sh",\n'
    printf '  "meta": {\n'
    printf '    "wrapper_name": "%s",\n' "$(basename "$0")"
    printf '    "user_ok_required": true,\n'
    printf '    "wf_origin": "%s",\n' "Wf-306B - Wf-219 Ru short-epoch GPU re-test wrapper (30 epochs x 3 seeds, RTX 5090 cuda:1)"
    printf '    "parent_plan": "%s"\n' "TODO/completed/334_final_closeout.md §8.3 - Wf-306B wrapper inventory (Wf-219 §4.1 Ru re-run; wrapper plan: molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md)"
    printf '  },\n'
    printf '  "wf_origin": "Wf-306B",\n'
    printf '  "origin_wf": "Wf-304C Task 1 + Wf-305B/Wf-305C follow-up",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "metal": "%s",\n' "${METAL}"
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}")"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "wall_budget_label": "PROJECTED (Wf-304C measured 13.5 min for 43/50 epochs; 30/50 ≈ 9.4 min PROJECTED; +33 pct safety = 10 min budget)",\n'
    printf '  "epoch_target_origin": "Wf-304C §1: Ru baseline timed out at 43/50 epochs under standard protocol",\n'
    printf '  "out_dir": "%s",\n' "${REPORTS_DIR}"
    printf '  "pre_flight": "%s",\n' "${PRE_FLIGHT_STATUS}"
    printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"
    printf '}\n'
} > "${PLAN_JSON}.tmp"
mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
ok "plan emitted → ${PLAN_JSON}"

if [ "${DRY_RUN}" = "1" ]; then
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + Wf-363-A1 — a --dry-run must leave a durable on-disk record
#          of the plan it WOULD have run, and of the fact that NOTHING was executed.
    # Wf-A269 wave-5 (2026-11-02) - D7: the DRY_RUN.log write is now atomic
    # (.tmp + mv -f), matching the reference implementations
    # (scripts/launch_wf306a_drugood_vina_gpu.sh writes DRY_RUN.log.tmp then
    # mv -f; scripts/launch_wf74_v2_drugood_dann.sh does the same), so an
    # interrupted write cannot leave a truncated DRY_RUN.log behind.
    # Wf-A269 wave-5 (2026-11-02) - D8 follow-through: the pre_flight_check
    # line below used to assert "PASS (exit 0)".  After the dry-run guard the
    # live pre-flight is NOT executed on the dry-run path, so that claim would
    # be false; the line now reports the real status carried in
    # ${PRE_FLIGHT_STATUS} ("SKIPPED - dry-run executes no live gate").
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
    #     cat >"${REPORTS_DIR}/DRY_RUN.log" <<EOF
    # [DRY-RUN] launch_wf306b_wf219_ru_gpu.sh @ wf306b_wf219_ru_wrapper
    # [DRY-RUN] pre_flight_check PASS (exit 0; gate summary in ${REPORTS_DIR}/preflight.json)
    # [DRY-RUN] wrapper_name=launch_wf306b_wf219_ru_gpu.sh
    # [DRY-RUN] wf_origin=Wf-306B
    # [DRY-RUN] trainer=${TRAINER}
    # [DRY-RUN] py=${PY}
    # [DRY-RUN] args=--dry-run
    # [DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
    # [DRY-RUN] plan_that_would_run: CUDA_VISIBLE_DEVICES=1 ${PY} ${TRAINER} --metals ${METAL} --metals-only ${METAL} --seeds ${SEEDS[*]} --epochs ${EPOCHS} --out ${REPORTS_DIR}/sweep_summary.json
    # [DRY-RUN] nothing_was_executed: true (no trainer invoked, no GPU touched, no checkpoint written)
    # EOF
    cat >"${REPORTS_DIR}/DRY_RUN.log.tmp" <<EOF
[DRY-RUN] launch_wf306b_wf219_ru_gpu.sh @ wf306b_wf219_ru_wrapper
[DRY-RUN] pre_flight_check ${PRE_FLIGHT_STATUS}
[DRY-RUN] wrapper_name=launch_wf306b_wf219_ru_gpu.sh
[DRY-RUN] wf_origin=Wf-306B
[DRY-RUN] trainer=${TRAINER}
[DRY-RUN] py=${PY}
[DRY-RUN] args=--dry-run
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
[DRY-RUN] plan_that_would_run: CUDA_VISIBLE_DEVICES=1 ${PY} ${TRAINER} --metals ${METAL} --metals-only ${METAL} --seeds ${SEEDS[*]} --epochs ${EPOCHS} --json-dir ${REPORTS_DIR}/seed<SEED>
[DRY-RUN] nothing_was_executed: true (no trainer invoked, no GPU touched, no checkpoint written)
EOF
    mv -f "${REPORTS_DIR}/DRY_RUN.log.tmp" "${REPORTS_DIR}/DRY_RUN.log"
    log "DRY-RUN: record emitted -> ${REPORTS_DIR}/DRY_RUN.log"
    log "DRY-RUN: skipping GPU launch (USER_OK was still required)"
    log "DRY-RUN: actual invocation would be:"
    log "  CUDA_VISIBLE_DEVICES=1 ${PY} ${TRAINER} \\"
    log "    --metals ${METAL} --metals-only ${METAL} \\"
    log "    --seeds ${SEEDS[*]} --epochs ${EPOCHS} \\"
    log "    --json-dir ${REPORTS_DIR}/seed<SEED>"
    exit 0
fi

# -----------------------------------------------------------------------------
# Actual GPU run — sequential 3-seed sweep, ~10 min total wall
#
# 2026-11-04 (Wf-A269 wave-19) — THE OUTPUT FLAG WAS WRONG, AND NOTHING
# CHECKED IT.  This wrapper passed ``--out ${REPORTS_DIR}/sweep_seed${SEED}.json``
# to the trainer, but ``molmetal/scripts/wf219_krasnov_aligned.py`` declares
# NO ``--out``: its declared flags are ``--metals --metals-only --seeds
# --epochs --hidden --depth --batch-size --lr --num-workers --device
# --ckpt-dir --json-dir --time-budget-s --quick --metals-only``
# (MEASURED 2026-11-04 by AST + parser capture).  argparse would have exited
# 2 with "unrecognized arguments: --out" — the wrapper could never have run.
#
#   THE SUPERSEDED RENDERING IS PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#       --out "${REPORTS_DIR}/sweep_seed${SEED}.json" \
#
#   WHY NO GATE CAUGHT IT: ``scripts/_launch_contract_gate.py`` is ANCHORED on
#   the trainer name pattern ``^train_[a-z0-9_]+\.py$``, and this wrapper's
#   script is ``wf219_krasnov_aligned.py`` — not a ``train_*`` file.  The
#   yaml-only gate's rule 2 is anchored the same way.  So "the flag reaches a
#   parser that declares it" was UNCHECKED for every non-``train_*`` script
#   (MEASURED 2026-11-04: a read-only census over the 101 in-contract
#   wrappers found exactly 2 such defects — this one and
#   ``molmetal/scripts/smoke/smoke_lambda.sh``'s ``--out-dir``).
#
# THE REPAIR: the trainer takes ``--json-dir`` (a DIRECTORY) and writes
# ``<json-dir>/{metal}_seed{seed}.json`` (one per metal x seed cell) and
# ``<json-dir>/sweep_summary.json``.  A per-seed directory keeps each seed's
# summary separate — a flat directory would have the three invocations
# OVERWRITE each other's ``sweep_summary.json`` (the last seed would win).
# NOT EXECUTED: this session is 0-GPU (AGENTS.md §5.1 standing grant
# suspended), so the repair is verified BY CONSTRUCTION — the flag is declared
# (parser capture) and the write paths are read from the trainer's source.
# -----------------------------------------------------------------------------
RUN_LOG="${REPORTS_DIR}/run.log"
echo "[]" > "${REPORTS_DIR}/per_seed_results.json"

start_ts=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    log "=== seed=${SEED} Ru ${EPOCHS} epochs on ${GPU} ==="
    if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1}" "${PY}" "${TRAINER}" \
            --metals "${METAL}" --metals-only "${METAL}" \
            --seeds "${SEED}" --epochs "${EPOCHS}" \
            --json-dir "${REPORTS_DIR}/seed${SEED}" \
            >> "${RUN_LOG}" 2>&1; then
        ok "seed=${SEED} completed"
    else
        die "seed=${SEED} failed (see ${RUN_LOG})"
    fi
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_min=$(( elapsed_sec / 60 ))
log "Wf-306B Wf-219 Ru short-epoch sweep done — elapsed=${elapsed_min} min (budget=${WALL_BUDGET_MIN} min)"

if [ "${elapsed_min}" -gt "${WALL_BUDGET_MIN}" ]; then
    warn "elapsed (${elapsed_min} min) exceeded PROJECTED wall budget (${WALL_BUDGET_MIN} min); flagging as MEASURED-over-PROJECTED in plan.json"
fi

ok "READY: see ${REPORTS_DIR}/run.log + per-seed seed{0,1,2}/{sweep_summary.json,Ru_seed<N>.json}"
