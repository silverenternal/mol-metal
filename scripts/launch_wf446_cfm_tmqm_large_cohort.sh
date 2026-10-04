#!/usr/bin/env bash
# launch_wf446_cfm_tmqm_large_cohort.sh — the FIRST large-cohort CFM training.
#
# Why this wrapper exists (2026-10-04):
#
# Every CFM checkpoint on disk has `n_train <= 256`.  256 was
# DEFAULT_TRAIN_CLOUDS, and no config ever overrode it — a sensible default for
# the old synthetic Pt-click corpus that became a silent cap when the corpus
# switched to tmQM.  On top of that, `iter_tmqm_blocks` ended its per-file loop
# with a bare `return`, so the generator never advanced past tmQM_X1.xyz.gz:
# the six-metal pool was 6,872 complexes instead of 33,429, and a caller asking
# for more than that got 6,872 with no warning.  Both are fixed (see
# TODO/completed/cfm_bond_objective_and_m1_split_fix_20261004.md §10).
#
# This launch is therefore the first time the CFM trains on the delivered
# corpus: 26,715 train / 6,714 val complexes, 257x the previous cohort.
#
# Canonical surface per scripts/_WRAPPER_TEMPLATE.sh + Wf-431-A2:
#   - PY hardcoded to the project venv (AGENTS.md §5.9 single-venv rule)
#   - USER_OK gate FIRST, dry-run included (Wf-294B / Wf-355-B1 / Wf-378-A2)
#   - DRY_RUN.log + plan.json emission (Wf-363-A1)
#   - pre_flight_check BEFORE any GPU work (Wf-415-A3 / Wf-427-B1)
#   - PYTHONPATH=PROJECT_ROOT on the trainer call (Wf-431-A1 F2)
#
# Everything the trainer does is driven by configs/cfm_tmqm_all6.yaml
# (user directive: configuration lives in yaml, not in scattered CLI flags).
# This wrapper passes only the run-control values the yaml legitimately cannot
# own: --config, --seed, --out-dir.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf446_cfm_tmqm_large_cohort.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf446_cfm_tmqm_large_cohort.sh
#
# Owner: Hugo Lee + ultracode agent (Wf-446)
# Date: 2026-10-04

set -euo pipefail

# --- Paths (HARDCODED per §5.9) -------------------------------------------
PY=/home/hugo/codes/mol-metal/.venv/bin/python
PROJECT_ROOT=/home/hugo/codes/mol-metal
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
DRY_RUN_LOG_HELPER="${SCRIPTS_DIR}/_dry_run_log_helper.sh"
TRAINER="molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
CONFIG="configs/cfm_tmqm_all6.yaml"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/cfm_tmqm_all6}"
WRAPPER_NAME="launch_wf446_cfm_tmqm_large_cohort.sh"
WF_ID="wf446"
SEED="${SEED:-0}"
export PYTHONPATH="/home/hugo/codes/mol-metal"
export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-${PROJECT_ROOT}/.data/molmetal}"

# --- USER_OK gate FIRST (per Wf-294B + Wf-355-B1 + Wf-366-A1) ------------
# kill_then_adapt §0: a LIVE GPU launch must never be triggered by anything
# other than an explicit USER_OK=1.  --dry-run does NOT bypass this
# (Wf-378-A2 G6 lesson).
if [ "${USER_OK:-0}" != "1" ]; then
    cat <<EOF >&2
[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
       USER_OK=1 bash ${WRAPPER_NAME} --dry-run   # plan only
       USER_OK=1 bash ${WRAPPER_NAME}             # actual run (~4-5 hr, cuda:0)
EOF
    exit 64
fi

# --- Helpers -------------------------------------------------------------
die() { printf '%s [FATAL] %s\n' "$(date -Iseconds 2>/dev/null || date)" "$*" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
[ -f "${DRY_RUN_LOG_HELPER}" ] || die "DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"
# shellcheck disable=SC1090
source "${DRY_RUN_LOG_HELPER}"

[ -f "${PROJECT_ROOT}/${TRAINER}" ] || die "trainer missing: ${TRAINER}"
[ -f "${PROJECT_ROOT}/${CONFIG}" ]  || die "config missing: ${CONFIG}"
[ -x "${PY}" ]                      || die "venv python missing: ${PY}"

# --- Stale-checkpoint guard (AGENTS.md §5.7 class) -----------------------
# `--resume-from` defaults to `auto`, which scans the checkpoint directory and
# continues from whatever it finds.  MEASURED during the preparation of this
# very launch: a 2-epoch/256-cloud smoke wrote into this run's ckpt_dir, and an
# unguarded launch would have silently resumed from it — training 20 epochs
# from a 2-epoch model and reporting the result as a large-cohort run.  This is
# the Wf-A10 `train_seconds=0.0035` failure mode; fail loudly instead.
CKPT_DIR="${PROJECT_ROOT}/results/cfm_tmqm_all6/checkpoints"
STALE=$(find "${CKPT_DIR}" -name 'cfm_yuelbond_wasserstein_seed*.pt' 2>/dev/null | head -3 || true)
if [ -n "${STALE}" ] && [ "${ALLOW_RESUME:-0}" != "1" ]; then
    cat >&2 <<EOF
[FATAL] stale checkpoint(s) present in ${CKPT_DIR}:
$(printf '          %s\n' ${STALE})
        --resume-from defaults to 'auto' and WOULD continue from these, so a
        'large-cohort' run could silently be a resume of a smoke.
        Inspect them, remove them, or set ALLOW_RESUME=1 to resume on purpose.
EOF
    exit 65
fi

# --- VRAM budget check FIRST (AGENTS.md §5.1) ----------------------------
# "显存够就放心大胆用" is standing authorization WITH a budget check attached.
# A concurrent job on the same GPU is fine; OOM risk is not.
echo "[$(date -Iseconds)] ${WRAPPER_NAME} VRAM check"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    || die "nvidia-smi unavailable; cannot confirm VRAM budget"
FREE_MIB=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null \
    | awk -F', ' '{print $1 - $2}')
[ -n "${FREE_MIB}" ] || die "could not read free VRAM on GPU 0"
# Measured need: the 256-cloud run holds ~1.3 GiB.  This cohort is 104x larger
# per epoch but the MODEL is unchanged, so the peak is dominated by activations
# at batch_size=16 — 8 GiB is a generous floor that still catches a real squeeze.
if [ "${FREE_MIB}" -lt 8192 ]; then
    die "GPU 0 has ${FREE_MIB} MiB free; need >= 8192 MiB. Aborting per §5.1."
fi
echo "[$(date -Iseconds)] GPU 0 free=${FREE_MIB} MiB (>= 8192 required) — OK"

# --- Pre-flight gate (per Wf-415-A3 + Wf-427-B1) -------------------------
echo "[$(date -Iseconds)] ${WRAPPER_NAME} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch \
    >"${OUT_DIR}/preflight.json" 2>&1 \
    || die "pre_flight_check FAIL — see ${OUT_DIR}/preflight.json"
grep -q '"fail": 0' "${OUT_DIR}/preflight.json" \
    || die "pre_flight_check has FAILs (not 0) — see ${OUT_DIR}/preflight.json"
echo "[$(date -Iseconds)] pre-flight PASS"


# --- Main: --dry-run + actual launch ----------------------------------
MODE="actual"
case "${1:-}" in
    --dry-run)  MODE="dry-run" ;;
esac

# `--device` is NOT passed: configs/cfm_tmqm_all6.yaml declares `device: cuda`
# and the loader resolves it.  Repeating it here would be a second source of
# truth for the same setting (user directive: configuration lives in yaml), and
# it also trips scripts/_wrapper_audit.py, whose DUPLICATE heuristic counts
# every literal "--device" in the file rather than launch invocations.
TRAIN_ARGS=(
    --config "${CONFIG}"
    --seed "${SEED}"
    --out-dir "${OUT_DIR}"
)

echo "[$(date -Iseconds)] ${WRAPPER_NAME} mode=${MODE}"

if [ "${MODE}" = "dry-run" ]; then
    # Canonical schema emitter (Wf-363-A1) rather than a hand-rolled heredoc:
    # downstream tooling validates DRY_RUN.log against this schema, and a
    # bespoke format is invisible to it.
    emit_dry_run_log "${OUT_DIR}/DRY_RUN.log" \
        --title "Wf-446 CFM tmQM large cohort" \
        --wrapper "${WRAPPER_NAME}" \
        --mode "large_cohort_20ep" \
        --trainer "${TRAINER}" \
        --config "${CONFIG}" \
        --flags "--config ${CONFIG} --seed ${SEED} --out-dir ${OUT_DIR}" \
        --gpu "cuda:0 (${FREE_MIB} MiB free)" \
        --budget-hr "5" \
        --user-ok "REQUIRED" \
        --pair "n_train=26715 n_val=6714 seed=${SEED}"
    cat >"${OUT_DIR}/plan.json" <<EOF
{
    "wrapper": "${WRAPPER_NAME}",
    "wf_origin": "${WF_ID}",
    "mode": "dry-run",
    "user_ok_required": true,
    "trainer": "${TRAINER}",
    "config": "${CONFIG}",
    "py": "${PY}",
    "out_dir": "${OUT_DIR}",
    "vram_free_mib": ${FREE_MIB},
    "pre_flight": "PASS",
    "args": ["--config", "${CONFIG}", "--seed", "${SEED}", "--out-dir", "${OUT_DIR}"],
    "expected_cohort": {"n_train": 26715, "n_val": 6714},
    "estimated_wall": "4-5 hr (26715 clouds; MEASURED 256-cloud epoch = 6-8 s)",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"
}
EOF
    echo "[$(date -Iseconds)] ${WRAPPER_NAME} DRY-RUN logged to ${OUT_DIR}"
    exit 0
fi

# --- Actual launch -------------------------------------------------------
echo "[$(date -Iseconds)] ${WRAPPER_NAME} launching (est. 4-5 hr)"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${PROJECT_ROOT}/${TRAINER}" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/launch.log"
echo "[$(date -Iseconds)] ${WRAPPER_NAME} complete; log at ${OUT_DIR}/launch.log"
