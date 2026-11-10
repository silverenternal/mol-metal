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

# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# 2026-10-08 pipefix: drop pipefail (SIGPIPE from forked DataLoader workers
# flushing stdout after trainer exit killed the wrapper silently via `tee`).
# Capture trainer rc via ${PIPESTATUS[0]}; suppress tee SIGPIPE with `|| true`.
#
# Wf-A269 wave-10 (2026-11-02): the `|| true` half above is what made the rc
# read DEAD — `|| true` runs AFTER the failing pipeline and CLOBBERS
# PIPESTATUS to (0), so the "trainer exited rc != 0 -> exit" branch could never
# fire and this wrapper exited 0 after a crashed trainer.  The repaired shape
# is at the launch site below (scripts/_WRAPPER_TEMPLATE.sh:190-193).
set -uo pipefail

# --- Paths (HARDCODED per §5.9) -------------------------------------------
PY=/home/hugo/codes/mol-metal/.venv/bin/python
PROJECT_ROOT=/home/hugo/codes/mol-metal
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
DRY_RUN_LOG_HELPER="${SCRIPTS_DIR}/_dry_run_log_helper.sh"
TRAINER="molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
CONFIG="configs/cfm.yaml"
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

# -----------------------------------------------------------------------------
# §5.11 (2026-11-03) — per-run config derivation
# -----------------------------------------------------------------------------
# The trainer used to be invoked as `--config <base> --seed ${SEED} --out-dir
# <out>`.  `--seed` is a HYPERPARAMETER (§5.11: only --config / --out-dir /
# --ckpt-dir / --help may appear on a trainer invocation line), so it moves
# into the YAML.
#
# The base config ALREADY declares the seed — `protocol.seed: 0` in
# configs/cfm.yaml — and `protocol.seed` is a LIVE key: MEASURED 2026-11-03
# with scripts/_yaml_equivalence_probe.py,
#     --old-argv-json '["--seed","3"]'  ->  DIFF seed: old=3 (int) new=0 (int)
# i.e. the config's value (0) reaches the `seed` dest and the probe sees the
# CLI value override it.  `protocol` is one of the loader's leaf-fallback
# sections (wfA2_config_loader.py:78), so the key resolves rather than being
# dropped with a warning.
#
# The derived config is written with the RUN's seed so `SEED=3 bash <wrapper>`
# still selects seed 3 — which is what the old `--seed "${SEED}"` did.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   TRAIN_ARGS=(
#       --config "${CONFIG}"
#       --seed "${SEED}"
#       --out-dir "${OUT_DIR}"
#   )
DERIVED_CONFIG="${OUT_DIR}/_derived_configs/cfm_seed${SEED}.yaml"
mkdir -p "$(dirname "${DERIVED_CONFIG}")"
${PY} - "${PROJECT_ROOT}/${CONFIG}" "${DERIVED_CONFIG}" "${SEED}" <<'PYEOF' \
    || die "per-run config derivation failed for seed=${SEED}"
import sys

import yaml

base, out, seed = sys.argv[1:4]
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
# `protocol.seed` is the key that reaches the `seed` dest on this parser
# (see the probe evidence above).  Nothing else in the base config changes.
cfg.setdefault("protocol", {})["seed"] = int(seed)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# DERIVED per-run config for scripts/launch_wf446_cfm_tmqm_large_cohort.sh\n")
    fh.write("# base: %s\n" % base)
    fh.write("# seed: %s\n" % seed)
    fh.write("# Only `protocol.seed` differs from the base; it carries the value\n")
    fh.write("# the old `--seed ${SEED}` command-line flag used to pass.\n")
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
PYEOF
[ -f "${DERIVED_CONFIG}" ] || die "derived config missing: ${DERIVED_CONFIG}"

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
# Wf-A269 wave-17 (2026-11-03) — FOUR repairs to this block.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# the pre-fix block ran the pre-flight under a BARE `setsid nohup` and then
# read the verdict out of a file named `preflight.json` with
# `grep -q '"fail": 0'`.  Two independent defects lived in that shape:
#  (a) bare `setsid` MAY fork and return 0 immediately, so the `|| die`
#      below saw SETSID's exit code rather than the pre-flight's — and the
#      `grep` then read a file the pre-flight had not finished writing.  A
#      gate whose verdict is read from a half-written file is a race, not a
#      gate.  `setsid --wait` is the repo's own established fix for this
#      (see scripts/launch_cfm_frag_ab.sh:148-153).
#  (b) the `grep` predicate was TIMEOUT-BLIND (a report with 1 TIMEOUT and
#      0 FAILs matched it) and could not see a TIMEOUT at all, so a gate
#      whose measurement was NOT obtained counted as a PASS (Wf-A135).
#  (1) `setsid --wait` so the wrapper waits for the pre-flight it gated on.
#  (2) `--json PATH` writes a REAL JSON document; the console transcript
#      goes to a sibling .log, so the report path stays pure JSON.
#  (3) the verdict comes from the repo's own FAIL-CLOSED consumer
#      (molmetal/scripts/wf_launch_gate_verdict.py).
#  (4) the previous run's report is removed first ("stale is treated as
#      missing"), so a crashed pre-flight leaves no PASS report behind.
PREFLIGHT_OUT="${OUT_DIR}/preflight.json"
PREFLIGHT_LOG="${OUT_DIR}/preflight.log"
mkdir -p "${OUT_DIR}"
rm -f "${PREFLIGHT_OUT}"
echo "[$(date -Iseconds)] ${WRAPPER_NAME} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal setsid --wait nohup ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch \
    --json "${PREFLIGHT_OUT}" \
    >"${PREFLIGHT_LOG}" 2>&1 \
    || die "pre_flight_check FAIL — see ${PREFLIGHT_LOG}"
pf_verdict="$(PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf_launch_gate_verdict.py \
    preflight --report "${PREFLIGHT_OUT}" 2>&1)" \
    || die "pre_flight_check verdict is not PASS — ${pf_verdict} (see ${PREFLIGHT_OUT})"
echo "[$(date -Iseconds)] pre-flight PASS — ${pf_verdict}"


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
    --config "${DERIVED_CONFIG}"
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
        --flags "--config ${DERIVED_CONFIG} --out-dir ${OUT_DIR}" \
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
    "args": ["--config", "${DERIVED_CONFIG}", "--out-dir", "${OUT_DIR}"],
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
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   setsid nohup PYTHONPATH=/home/hugo/codes/mol-metal ${PY} "${PROJECT_ROOT}/${TRAINER}" "${TRAIN_ARGS[@]}" \
#       2>&1 | tee "${OUT_DIR}/launch.log" || true   # 2026-10-08 pipefix
#   TRAIN_RC=${PIPESTATUS[0]}                        # 2026-10-08 pipefix
#
# Canonical shape (scripts/_WRAPPER_TEMPLATE.sh:190-193): pipefail off for
# this ONE pipeline (a trainer whose forked DataLoader workers still hold the
# pipe open takes SIGPIPE and would report 141 instead of its own status — the
# tee stays, it is the only diagnostic); `setsid --wait` (bare setsid may fork
# and return 0 before the trainer has done anything, which would make
# PIPESTATUS[0] report success for a run that had not started — and setsid
# gives the trainer its own session, which already covers nohup's SIGHUP
# immunity); and the rc read BETWEEN the two `set` lines, while PIPESTATUS
# still holds the pipeline's statuses.
#
# The env assignment moved in FRONT of setsid (template form): `setsid` does
# NOT accept `VAR=value` arguments — it execs the first argument as the
# program, so the old form died with
#   "nohup: failed to run command 'PYTHONPATH=/home/hugo/codes/mol-metal'"
# and rc=127, which the `|| true` then reported as a SUCCESS (rc=0).
set +o pipefail
PYTHONPATH=/home/hugo/codes/mol-metal setsid --wait ${PY} "${PROJECT_ROOT}/${TRAINER}" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/launch.log"
TRAIN_RC=${PIPESTATUS[0]}
set -o pipefail
echo "[$(date -Iseconds)] ${WRAPPER_NAME} complete; log at ${OUT_DIR}/launch.log"
if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "FAIL: trainer exited rc=${TRAIN_RC}" >&2
    exit "${TRAIN_RC}"
fi
