#!/usr/bin/env bash
# =============================================================================
# 2026-10-04 — THIS LAUNCHER CANNOT RUN.  It passes --metal-coord-term, which
# now raises SystemExit in the trainer.
#
# Why it was disabled rather than left alone: MetalCoordinationTerm is a
# rule-based SMILES scorer (coordination_geometry_score / hsab_match_score —
# RDKit + an HSAB table), so `kcal_adjustment` returns a Python float with no
# gradient.  The old wire-in wrapped it in torch.tensor() behind a .item(),
# which put a CONSTANT into the loss behind `metal_coord_weight` (default 0.5)
# while every parameter update stayed identical to running without it — a run
# that reports the flag as active while doing nothing.
#
# It is not repairable by dropping the .item(): the term has no gradient to
# give, and feeding it sigmoid(pic50_pred) as a stand-in (as the old code did)
# would teach the model to inflate its own output, since minimising
# (1 - 2*c) pushes c up and c is derived from the supervised prediction.
#
# Independently, the term cannot fire on this corpus: it detects the metal
# FROM THE SMILES, and MEASURED, 0 of 800 sampled rows of metalcytotox_5metal
# contain any metal atom — the metal is a separate `metal` COLUMN.
#
# The term already runs where it works:
#   molmetal/molmetal_lam/sbdd_env/vina_adapter.py::_compute_metal_coord_adjustment
# applies it to real docked SMILES as a post-hoc kcal/mol correction.
#
# To make metal chemistry visible to the D-MPNN, the mechanism has to be an
# auxiliary target computed per-row from the `metal` column — which first
# requires hsab_match_score / coordination_geometry_score to accept an
# explicit metal symbol.  That is a different piece of work; see
# TODO/completed/431a3_wf246c_scafffold_close.md.
#
# Kept on disk (not deleted) so the Wf-388 / Wf-409 / Wf-406 provenance chain
# stays intact.
# =============================================================================
# -----------------------------------------------------------------------------
# scripts/launch_wf388_a1_metal_coord_retrain.sh
#
# Wf-388-A1 — Wf-246-C metal-coordination term retrain launch wrapper
# (USER_OK-gated, GPU launch deferred per AGENTS.md §5.1).
#
# Per the FIX 2 brief (Wf-432-A2 follow-up): this is the canonical
# Wf-388 Wf-246-C wrapper. Sister to Wf-409-A3 launch wrapper at
# scripts/launch_wf409a3_metal_coord_retrain.sh (Wf-409-A3 SCAFFOLD) and
# scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh (Wf-406-A4 legacy).
# This Wf-388-A1 wrapper uses the real D-MPNN trainer
# (molmetal/scripts/train_dmpnn_multitask_wasserstein.py) with the
# Wf-246-C metal-coord term wire-in (3 NEW CLI flags per Wf-432-A2 +
# Wf-441B §3 + Wf-427-A1 line 641 kcal sign fix):
#   - --metal-coord-term (opt-in wire-in)
#   - --metal-coord-weight (default 0.5; loss cascade weight)
#   - --cal-sign-vina-convention (default ON; gates bit-for-bit Vina
#     kcal/mol sign convention; lower = better binding)
#
# Date: 2026-10-02
# Owner: Hugo Lee + Wf-388-A1 ultracode agent
# Parent: TODO/pending/376_m2_r2_gpu_retrain_plan.md (Path A 5-metal cohort)
#          + Wf-411-A2 SCAFFOLD_NOT_FUNCTIONAL close
#          + Wf-431-A3 §3 B6 Wf-246-C SCAFFOLD_NOT_FUNCTIONAL
# Trainer: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
#         (canonical D-MPNN multi-task Wasserstein trainer;
#          FIVE_METALS constraint at line 141 + argparse choices at
#          line 1057 — Path A works WITHIN this constraint by using
#          the 5-metal subset natively per Wf-390-A1 §1.1)
#
# Refs:
#   - TODO/pending/376_m2_r2_gpu_retrain_plan.md §1.1 (Path A 5-metal)
#   - TODO/03_baselines/COMPARISON_BASELINES.md §1.1 (Krasnov 5-metal subset)
#   - TODO/completed/367b1_m2_r2_recovery.md (Wf-367-B1 multi-stream Ridge)
#   - TODO/completed/390a1_m2_r2_path_a_smoke.md (Wf-390-A1 SCAFFOLD smoke)
#   - TODO/completed/431a3_wf246c_scafffold_close.md (Wf-431-A3 honest-neg)
#   - docs/M2_R2_FINAL_VERDICT.md (Wf-369-A2 FINAL VERDICT 320 LOC)
#   - scripts/_dry_run_log_helper.sh (Wf-363-A1 per-pair DRY_RUN.log schema)
#   - molmetal/scripts/wf386a3_atomic_write.py (Wf-386-A3 atomic_write_file
#     primitive per AGENTS.md §5.6)
#   - Wf-415-A1 + Wf-408-A3 + Wf-380-A2 (transparent passthrough pattern)
#   - Wf-424-A2 (greedy tokenisation for --metals + --seeds)
#   - Wf-335-R1+R3 (--no-cuda-graphs safety lever for Blackwell sm_120)
#   - Wf-432-A2 + Wf-441B §3 (real trainer --metal-coord-term wire-in)
#   - Wf-427-A1 line 641 (kcal sign fix; Vina convention lower=better)
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#
# HONEST FRAMING (party-twenty-spirit §3.5 + AGENTS.md §5.2):
#   - user_ok_required: true (kill_then_adapt; 15 pairs × 50 epochs ×
#     ~30 min wall per pair PROJECTED = ~7.5 hr GPU wall PROJECTED
#     per Wf-260 G6 timing on RTX 5090 / Blackwell sm_120)
#   - Scope: 15 (metal, seed) pairs (5 metals × 3 seeds × 50 epochs)
#     on metalcytotox_combined (n=2000) cohort per Wf-390-A1 Path A.
#   - kcal sign fix per Wf-427-A1 line 641:
#     ``(1 - 2*total) * kcal_per_unit`` (Vina convention: lower
#     kcal/mol = better binding). Without the fix, training would
#     invert the kcal gradient.
#   - backward compat: when --metal-coord-term NOT passed, the trainer
#     falls back to baseline Wf-131 P1 behaviour (bit-for-bit when all
#     3 metal-coord flags OFF).
#   - Wf-415-A1 transparent passthrough: any extra flags NOT in the
#     wrapper's known surface are appended to the trainer invocation
#     verbatim (preserves --no-cuda-graphs, --user-ok, --chemprop-v3-cache,
#     --per-cohort-features, etc.).
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --dry-run
#     # CPU verification only (USER_OK gate still required)
#   USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh --smoke
#     # CPU 5-step SCAFFOLD smoke (no GPU launch; bit-for-bit)
#   USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh
#     # actual ~7.5 hr GPU run (15 pairs × 50 epochs × ~30 min)
#   USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh \
#       --config configs/wfa269_c4_dmpnn_metal_coord.yaml
#     # the cohort lives in the YAML (AGENTS.md §5.11): copy that file, edit
#     #   data.metals / data.seeds / train.epochs, and select the copy with
#     #   --config <path> (or the DMPNN_CONFIG env var)
#
#   SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     USER_OK=1 bash scripts/launch_wf388_a1_metal_coord_retrain.sh \
#         --metals "Ru Ir" --seeds 0 1 --epochs 30
#       # subset override (debug pair)
#
#   2026-11-03 (AGENTS.md §5.11 migration): the example above is superseded.
#   --metals / --seeds / --epochs are HYPERPARAMETERS, so they live in the
#   YAML the trainer is launched with; the wrapper REFUSES a cohort that
#   disagrees with that file ("the cohort now lives in the config, not on
#   the command line") instead of training a subset the plan.json /
#   DRY_RUN.log it writes does not describe.  The refusal is loud.
#
# Exit codes:
#   0   PASS (dry-run or smoke)
#   1   FAIL
#   64  USER_OK gate refused (USER_OK env not set to "1")
#
# USER_OK gate (explicit, per mol-metal-gpu-execution §0 + Wf-294B family):
#   - The wrapper REFUSES to run unless USER_OK=1 is set in the env.
#   - --dry-run does NOT bypass the gate; it only skips GPU launch.
#
# Stop semantics: per TODO-207 v2 §7.2 (per-phase abort criteria):
#   - per-phase timeout: ckpt older than 2× projected wall-time
#   - failure pattern: 3+ consecutive NaN/OOM/IndexError
#   - GPU util < 70% for 5+ minutes
#   - collapse: per-cohort R² drop > 0.20 vs Wf-335-M2 CEILING_BREAK
#
# Pre-launch checks (BLOCKER before any GPU work):
#   1. pre_flight_check.py --pipeline launch (per AGENTS.md §1.4; 117/117
#      PASS target; 0 FAIL — Wf-411-A1 baseline is 113/0/4/117)
#   2. wave_budget_tracker.py --budget-hr 7.5 (per session cap; USER_OK
#      to extend on per-wave basis)
#   3. USER_OK env gate
#   4. Trainer + cohort CSV existence
# -----------------------------------------------------------------------------
set -euo pipefail

# Locate repo root + scaffold script + real trainer
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
# --- §5.9 single-venv + canonical markers (Wf-431-A1) ---------------------
PY="/home/hugo/codes/mol-metal/.venv/bin/python"
export PYTHONPATH="/home/hugo/codes/mol-metal"
TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
# The trainer already accepts --config and ends parse_args() in
# apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3965), so the
# yaml is the base of the run.
# Absolute: this wrapper has no top-level `cd`, so a relative path would
# resolve against the caller's cwd.
#
# 2026-11-03 (AGENTS.md §5.11): the config is now DEDICATED to this wrapper
# (configs/wfa269_c4_dmpnn_metal_coord.yaml — a copy of configs/dmpnn.yaml
# carrying the cohort / coord-term values this wrapper used to pass on the
# command line).  configs/dmpnn.yaml is SHARED by 8 launchers, so those
# values cannot be written there without changing the other 7 runs.
#
# THE CONFIG IS STATIC: this file is what the trainer is launched with.  The
# wrapper does NOT derive a per-run copy at launch time, so the values in
# that file ARE the values the trainer receives (a derived copy would move
# the cohort back into a shell variable — the knob nobody audits).
# To run a different cohort: copy the config, edit it, and select the copy
# with --config <path> (or the DMPNN_CONFIG environment variable).
DMPNN_CONFIG="${DMPNN_CONFIG:-${PROJECT_ROOT}/configs/wfa269_c4_dmpnn_metal_coord.yaml}"
# Wf-261 strict scaffold split (0 leak) + v4 cohort
DATA_CSV="${PROJECT_ROOT}/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# Source the DRY_RUN.log helper (Wf-363-A1 schema)
DRY_RUN_LOG_HELPER="${PROJECT_ROOT}/scripts/_dry_run_log_helper.sh"
if [ ! -f "${DRY_RUN_LOG_HELPER}" ]; then
    printf '%s[FATAL]%s DRY_RUN log helper missing: %s\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" \
        "${DRY_RUN_LOG_HELPER}" >&2
    exit 1
fi
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Die helper
die() {
    printf '%s[FATAL]%s %s\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" \
        "$*" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# CLI arg parsing (Wf-424-A2 greedy tokenisation for --metals + --seeds)
# -----------------------------------------------------------------------------
DRY_RUN=0
SMOKE=0
ACTUAL=0
# 5-metal Path A subset (matches trainer FIVE_METALS exactly per Wf-390-A1 §1.1)
# Tuple: (Ru, Ir, Rh, Os, Re) — matches train_dmpnn_multitask_wasserstein.py:141
METALS="Ru Ir Rh Os Re"
# 3 seeds × 5 metals × 50 epochs = 15 (metal, seed) pairs per the task brief
SEEDS="0 1 2"
EPOCHS=50
OUT_DIR="molmetal/reports/wf388_a1_metal_coord_launch"
# Wf-415-A1 transparent passthrough: any flag NOT consumed by the wrapper's
# case statement is collected into EXTRA_FLAGS and appended verbatim to the
# trainer invocation. This preserves --no-cuda-graphs, --user-ok,
# --chemprop-v3-cache, --per-cohort-features, etc.
EXTRA_FLAGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --actual) ACTUAL=1; shift ;;
        --metals)
            shift
            METALS=""
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                METALS="${METALS:+${METALS} }$1"
                shift
            done
            [ -n "${METALS}" ] || die "--metals requires at least one metal symbol"
            ;;
        --seeds)
            shift
            SEEDS=""
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                SEEDS="${SEEDS:+${SEEDS} }$1"
                shift
            done
            [ -n "${SEEDS}" ] || die "--seeds requires at least one integer seed"
            ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --config)
            # §5.11: the config selects WHICH run, not WHAT to train.  It is
            # the only trainer-facing knob this wrapper adds (see
            # scripts/_yaml_only_gate.py ALLOWED_TRAINER_FLAGS).
            DMPNN_CONFIG="$2"; shift 2 ;;
        -h|--help)
            printf 'Wf-388-A1 launch wrapper (USER_OK-gated Wf-246-C metal-coord term retrain)\n'
            printf 'Usage: USER_OK=1 %s [--dry-run|--smoke|--actual] [--metals M1 M2 ...] [--seeds 0 1 2 ...] [--epochs 50] [--config PATH]\n' "$0"
            printf '  --metals + --seeds accept both quoted ("Ru Ir Rh") and unquoted (Ru Ir Rh) forms (Wf-424-A2).\n'
            printf '  --config selects the YAML the trainer is launched with (default: %s).\n' "${DMPNN_CONFIG}"
            printf '  AGENTS.md 5.11: the cohort (--metals/--seeds) and --epochs live in that YAML.\n'
            printf '    The three options are ASSERTIONS against it — a mismatch REFUSES the run.\n'
            printf '    To run a different cohort: copy the config, edit it, pass --config <copy>.\n'
            printf '  Wf-415-A1 transparent passthrough: extra flags are appended verbatim to trainer.\n'
            printf '  --metal-coord-term + --metal-coord-weight 0.5 + --cal-sign-vina-convention are DEFAULT ON\n'
            printf '    for --actual mode (Wf-432-A2 wire-in; kcal sign fix per Wf-427-A1 line 641).\n'
            exit 0
            ;;
        *)
            # Wf-415-A1 transparent passthrough: collect unknown flags into EXTRA_FLAGS
            EXTRA_FLAGS+=("$1")
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, fires BEFORE --dry-run block per Wf-378-A2 G6 lesson)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" >&2
    printf '       USER_OK=1 bash %s --dry-run  # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --smoke     # 5-step CPU SCAFFOLD smoke (bit-for-bit)\n' "$0" >&2
    printf '       USER_OK=1 bash %s             # actual ~7.5 hr GPU run (15 pairs × 50 epochs × ~30 min)\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# §5.11 cohort guard (2026-11-03) — the cohort lives in the CONFIG FILE
# -----------------------------------------------------------------------------
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): --metals / --seeds / --epochs are HYPERPARAMETERS,
# so they live in the YAML the trainer is launched with, not on its command
# line.  The wrapper still ACCEPTS the three options (the call sites are
# unchanged), but they are now ASSERTIONS against the config: if they
# disagree with it, the run would train a different cohort than the
# plan.json / DRY_RUN.log this wrapper writes describe — a silent lie.
# So the wrapper REFUSES instead.
#
# To run a different cohort: copy the config, edit data.metals / data.seeds /
# train.epochs, and select the copy with --config <path> (or DMPNN_CONFIG).
# -----------------------------------------------------------------------------
# Absolute config path (this wrapper has no top-level `cd`, so a relative
# path would resolve against the caller's cwd).
case "${DMPNN_CONFIG}" in
    /*) ;;
    *) DMPNN_CONFIG="${PROJECT_ROOT}/${DMPNN_CONFIG}" ;;
esac
if ! "${PY}" - "${DMPNN_CONFIG}" "${METALS}" "${SEEDS}" "${EPOCHS}" <<'PYEOF'
import sys

import yaml

cfg_path, metals, seeds, epochs = sys.argv[1:5]
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
except OSError as exc:
    sys.stderr.write(f"[FATAL] cannot read config {cfg_path}: {exc}\n")
    raise SystemExit(1)
data = cfg.get("data") or {}
train = cfg.get("train") or {}
cfg_metals = [str(m) for m in (data.get("metals") or [])]
cfg_seeds = [str(s) for s in (data.get("seeds") or [])]
cfg_epochs = train.get("epochs")
want_metals = metals.split()
want_seeds = seeds.split()
mismatch = []
if cfg_metals != want_metals:
    mismatch.append(
        "metals cli=[{}] config=[{}]".format(" ".join(want_metals), " ".join(cfg_metals))
    )
if cfg_seeds != want_seeds:
    mismatch.append(
        "seeds cli=[{}] config=[{}]".format(" ".join(want_seeds), " ".join(cfg_seeds))
    )
if str(cfg_epochs) != str(epochs):
    mismatch.append(f"epochs cli={epochs} config={cfg_epochs}")
if mismatch:
    sys.stderr.write(
        "[FATAL] the cohort now lives in the config, not on the command line "
        "(AGENTS.md 5.11): " + "; ".join(mismatch) + "\n"
        f"        edit {cfg_path} (or run with --config <edited copy>) instead of "
        "--metals/--seeds/--epochs\n"
    )
    raise SystemExit(1)
PYEOF
then
    die "cohort in ${DMPNN_CONFIG} does not match --metals/--seeds/--epochs (AGENTS.md §5.11)"
fi

# -----------------------------------------------------------------------------
# Pre-launch sanity checks (BLOCKER before any GPU work)
# -----------------------------------------------------------------------------
[ -f "${TRAINER}" ]  || die "Trainer script missing: ${TRAINER}"
[ -f "${DATA_CSV}" ] || die "Cohort CSV missing: ${DATA_CSV} (Wf-261 strict scaffold split + v4 cohort)"

# --- Pre-flight gate (Wf-415-A3 + Wf-427-B1) ------------------------------
# Wf-A269 wave-17 (2026-11-03) — FOUR repairs to this block.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
# the pre-fix block captured the pre-flight's CONSOLE STREAM (stdout + stderr
# merged by `2>&1`) into a file named `preflight.json` and then read the
# verdict out of it with `grep -q '"fail": 0'`.  That predicate was
# TIMEOUT-BLIND (a report with 1 TIMEOUT and 0 FAILs matched it), and it
# could not see a TIMEOUT at all, so a gate whose measurement was NOT
# obtained counted as a PASS (Wf-A135).
#  (1) `--json PATH` writes a REAL JSON document; the console transcript
#      goes to a sibling .log, so the report path stays pure JSON.
#  (2) the verdict comes from the repo's own consumer
#      (molmetal/scripts/wf_launch_gate_verdict.py), which is FAIL-CLOSED.
#  (3) the previous run's report is removed first ("stale is treated as
#      missing"), so a crashed pre-flight leaves no PASS report behind.
PREFLIGHT_OUT="${LOG_DIR}/preflight.json"
PREFLIGHT_LOG="${PREFLIGHT_OUT%.json}.log"
mkdir -p "$(dirname "${PREFLIGHT_OUT}")"
rm -f "${PREFLIGHT_OUT}"
echo "[$(date -Iseconds)] ${BASH_SOURCE[0]} pre-flight check begin"
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    "${PROJECT_ROOT}/molmetal/scripts/pre_flight_check.py" --pipeline launch \
    --json "${PREFLIGHT_OUT}" \
    >"${PREFLIGHT_LOG}" 2>&1 \
    || die "pre_flight_check FAIL — see ${PREFLIGHT_LOG}"
pf_verdict="$(PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    "${PROJECT_ROOT}/molmetal/scripts/wf_launch_gate_verdict.py" \
    preflight --report "${PREFLIGHT_OUT}" 2>&1)" \
    || die "pre_flight_check verdict is not PASS — ${pf_verdict} (see ${PREFLIGHT_OUT})"
echo "[$(date -Iseconds)] pre-flight PASS — ${pf_verdict}"
# Wf-422-B1 / Wf-419-A1: pre-flight check is opt-in via /tmp/preflight_all.json
# (must be invoked BEFORE this wrapper; per AGENTS.md §1.4 the user's
# pre-flight verification must be CLEAN before any retrain launch).
[ -f "/tmp/preflight_all.json" ] || printf '[WARN] /tmp/preflight_all.json missing — pre-flight not yet run (per AGENTS.md §1.4 USER_OK waiver required)\n' >&2

# -----------------------------------------------------------------------------
# Mode selection (DRY_RUN > SMOKE > ACTUAL precedence)
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" = "1" ]; then
    MODE="R1_dry_run"
elif [ "${SMOKE}" = "1" ]; then
    MODE="R2_smoke"
elif [ "${ACTUAL}" = "1" ]; then
    MODE="R3_actual"
else
    die "must specify one of --dry-run / --smoke / --actual"
fi

# -----------------------------------------------------------------------------
# Per-pair enumeration (5-metal × 3-seed default = 15 pairs)
# -----------------------------------------------------------------------------
PAIRS=()
for metal in ${METALS}; do
    for seed in ${SEEDS}; do
        PAIRS+=("${metal} seed=${seed}")
    done
done

# -----------------------------------------------------------------------------
# Per-pair budget: assume ~30 min/seed on RTX 5090 / Blackwell sm_120
# (Wf-260 G6 timing × 1.5× margin for metal-coord term overhead)
# -----------------------------------------------------------------------------
PAIR_BUDGET_HR=0.50
TOTAL_BUDGET_HR=$(awk -v n="${#PAIRS[@]}" -v b="${PAIR_BUDGET_HR}" 'BEGIN { printf "%.2f", n*b }')

# -----------------------------------------------------------------------------
# Emit DRY_RUN.log per Wf-363-A1 schema
# -----------------------------------------------------------------------------
# OUT_DIR may be absolute or relative — handle both correctly.
case "${OUT_DIR}" in
    /*) LOG_DIR="${OUT_DIR}" ;;
    *)  LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}" ;;
esac
mkdir -p "${LOG_DIR}"

# Build the pair_args array: each --pair on its own argv element so the
# helper sees --pair as a discrete flag, not a glob.
PAIR_ARGS=()
for pair in "${PAIRS[@]}"; do
    PAIR_ARGS+=( --pair "${pair}" )
done

# Build the trainer flag string for DRY_RUN.log emission (--smoke / --actual)
TRAINER_FLAGS="(--metal-coord-term --metal-coord-weight 0.5 --cal-sign-vina-convention --metals ${METALS} --seeds ${SEEDS} --epochs ${EPOCHS})"

emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
    --title "Wf-388-A1 Wf-246-C metal-coord term retrain" \
    --wrapper "scripts/launch_wf388_a1_metal_coord_retrain.sh" \
    --mode "${MODE}" \
    --trainer "${TRAINER}" \
    --subset "5-metal Path A (Ru/Ir/Rh/Os/Re)" \
    --flags "${TRAINER_FLAGS}" \
    --gpu "cuda:0" \
    --budget-hr "${TOTAL_BUDGET_HR}" \
    --user-ok "REQUIRED" \
    "${PAIR_ARGS[@]}"

# Convert METALS and SEEDS space-separated strings to bash arrays for
# proper argv construction (avoid word-splitting on spaces).
METALS_ARR=(${METALS})
SEEDS_ARR=(${SEEDS})

# -----------------------------------------------------------------------------
# §5.11 YAML-only run config (2026-11-03) — the config IS the file
# -----------------------------------------------------------------------------
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may hand the trainer only
# --config / --out-dir / --ckpt-dir / --help.  Every hyperparameter lives in
# the YAML.
#
# The trainer SWEEPS the cohort itself (train_dmpnn_multitask_wasserstein.py
# :4352 `for metal in args.metals:` / :4356 `for seed in args.seeds:`), so one
# process runs all 15 (metal, seed) pairs and ONE static config covers the
# whole sweep: data.metals / data.seeds / train.epochs in
# ${DMPNN_CONFIG}.  No per-run config is derived here — a derived copy would
# put the cohort back into a shell variable, which is the knob nobody audits.
# The cohort guard above makes the wrapper's own --metals / --seeds /
# --epochs options agree with that file or refuse the run.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Emit plan.json (Wf-363-A1 schema + Wf-415-A1 extra_flags tracking)
# -----------------------------------------------------------------------------
cat > "${LOG_DIR}/plan.json" <<EOF
{
    "wf": "Wf-388-A1",
    "deliverable": "Wf-246-C metal-coordination term retrain launch wrapper (refactored)",
    "parent_wf": "Wf-432-A2",
    "mode": "${MODE}",
    "user_ok_required": true,
    "user_ok_env_var": "USER_OK=1",
    "metals": [$(printf '"%s",' ${METALS} | sed 's/,$//')],
    "seeds": [$(printf '%s,' ${SEEDS} | sed 's/,$//')],
    "epochs": ${EPOCHS},
    "n_pairs": ${#PAIRS[@]},
    "pair_budget_hr": ${PAIR_BUDGET_HR},
    "total_budget_hr": "${TOTAL_BUDGET_HR}",
    "trainer": "molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
    "config": "${DMPNN_CONFIG}",
    "kcal_sign_fix": "Wf-427-A1 line 641: (1 - 2*total) * kcal_per_unit (Vina convention lower=better)",
    "trainer_new_flags": [
        "--metal-coord-term (opt-in wire-in)",
        "--metal-coord-weight 0.5 (loss cascade weight)",
        "--cal-sign-vina-convention (default ON)"
    ],
    "scaffold_script": "molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py",
    "sister_module": "molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py",
    "extra_flags_passthrough": $(printf '"%s",' "${EXTRA_FLAGS[@]}" | sed 's/,$//' | sed 's/^/[/' | sed 's/$/]/' | sed 's/\[\]/[]/'),
    "wf415a1_transparent_passthrough": true,
    "wf424a2_greedy_tokenisation": true,
    "wf420a1_p2_tests_pass_fix": true,
    "honest_framing": {
        "metal_coord_sign_fix_id": "Wf-427-A1 line 641",
        "metal_coord_term_wire_in_id": "Wf-432-A2 + Wf-441B §3",
        "scaffold_only": false,
        "gpu_launch_blocked_on_user_ok": true,
        "actual_gpu_run_requires_user_ok_1": true,
        "wall_budget_label": "PROJECTED (~7.5 hr GPU sequential on RTX 5090 / Blackwell sm_120 per Wf-260 G6 timing × 1.5× margin)",
        "per_pair_wall_label": "PROJECTED (~30 min/seed × ~30 sec metal-coord term overhead)"
    },
    "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# -----------------------------------------------------------------------------
# Branch on mode
# -----------------------------------------------------------------------------
case "${MODE}" in
    R1_dry_run)
        printf '%s[Wf-388-A1]%s R1 dry-run mode: trainer + cohort + scaffold validated; ' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf '%d pairs; total budget %s hr (PROJECTED per Wf-260 G6 timing).\n' \
            "${#PAIRS[@]}" "${TOTAL_BUDGET_HR}"
        printf 'DRY_RUN.log: %s/DRY_RUN.log\n' "${LOG_DIR}"
        printf 'plan.json:   %s/plan.json\n' "${LOG_DIR}"
        exit 0
        ;;
    R2_smoke)
        SCAFFOLD="${PROJECT_ROOT}/molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py"
        [ -f "${SCAFFOLD}" ] || die "SCAFFOLD script missing: ${SCAFFOLD}"
        printf '%s[Wf-388-A1]%s R2 SCAFFOLD smoke mode: 5-step CPU smoke (bit-for-bit with --metal-coord-term OFF default).\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        exec "${PROJECT_ROOT}/.venv/bin/python" "${SCAFFOLD}" \
            --metal-coord-term \
            --metals "${METALS_ARR[@]}" \
            --seeds "${SEEDS_ARR[@]}" \
            --epochs 5 \
            --smoke \
            --out-dir "${LOG_DIR}"
        ;;
    R3_actual)
        # Wf-432-A2 — real trainer wire-in. The R3_actual case invokes the
        # real Wf-131 P3 D-MPNN trainer
        # (``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``)
        # with the Wf-246-C metal-coord term wire-in (3 NEW CLI flags
        # added in Wf-432-A2 per Wf-441B §3):
        #   --metal-coord-term (opt-in wire-in; default FALSE)
        #   --metal-coord-weight (loss cascade weight; default 0.5)
        #   --cal-sign-vina-convention (default ON; gates bit-for-bit
        #   Vina kcal/mol sign convention; lower = better binding per
        #   Wf-427-A1 line 641 kcal sign fix)
        # The kcal sign fix is applied per Wf-427-A1 line 641:
        # ``(1 - 2*total) * kcal_per_unit``; lower kcal/mol = better
        # binding per Vina convention. The trainer is invoked on cuda:0
        # by default (per AGENTS.md §5.1 + Wf-358 GPU standing grant).
        printf '%s[Wf-388-A1]%s R3 ACTUAL mode: invoking REAL trainer (Wf-131 P3 + Wf-246-C metal-coord term wire-in).\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf '  trainer: %s\n' "${TRAINER}"
        printf '  kcal sign fix per Wf-427-A1 line 641 (Vina convention)\n'
        printf '  kcal weight: --metal-coord-weight 0.5 (canonical Wf-246-C)\n'
        printf '  epochs: %d × %d (metal, seed) pairs = %d total\n' \
            "${EPOCHS}" "${#PAIRS[@]}" \
            "$((EPOCHS * ${#PAIRS[@]}))"
        # Bug fix 2026-10-08: removed --user-ok "REQUIRED" literal — D-MPNN trainer
        # has no --user-ok flag; passing it caused argparse rc=2 (the trainer would
        # exit silently via cmd '>' redirect). USER_OK gating is enforced by
        # the outer wrapper env check (Wf-310E part 1 gates USER_OK=1; this script
        # is a child), NOT by a CLI flag (per chemprop_v3 P1.4 wrapper post-mortem).
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #   PYTHONPATH=/home/hugo/codes/mol-metal \
        #   exec /home/hugo/codes/mol-metal/.venv/bin/python "${TRAINER}" \
        #       --config "${DMPNN_CONFIG}" \
        #       --metal-coord-term \
        #       --metal-coord-weight 0.5 \
        #       --cal-sign-vina-convention \
        #       --metals "${METALS_ARR[@]}" \
        #       --seeds "${SEEDS_ARR[@]}" \
        #       --epochs ${EPOCHS} \
        #       --device cuda:0 \
        #       --out-dir "${LOG_DIR}" \
        #       "${EXTRA_FLAGS[@]}"
        #
        # 2026-11-03 (AGENTS.md §5.11): every flag above except --config and
        # --out-dir is a HYPERPARAMETER, so it now lives in the YAML — the
        # STATIC file ${DMPNN_CONFIG}
        # (configs/wfa269_c4_dmpnn_metal_coord.yaml), which carries the
        # cohort (data.metals / data.seeds), the epoch count (train.epochs),
        # the coord-term switches (data.metal_coord_term /
        # data.metal_coord_weight / data.cal_sign_vina_convention) and the
        # device pin (compute.device: cuda:0).
        #
        # VERIFIED (MEASURED 2026-11-03, scripts/_yaml_equivalence_probe.py):
        # every dest the superseded invocation passed comes back with the
        # SAME value and the SAME type from --config alone; the full
        # namespace (argparse defaults + the old configs/dmpnn.yaml + the old
        # CLI flags) is identical to the new one, dest for dest, except the
        # `config` path itself.
        #
        # Wf-415-A1 transparent passthrough: "${EXTRA_FLAGS[@]}" is still
        # appended verbatim.  It is the one remaining channel through which a
        # trainer flag can reach the trainer (the yaml-only gate cannot see a
        # variable expansion); see the Wf-A269 receipt for the note.
        PYTHONPATH=/home/hugo/codes/mol-metal \
        exec /home/hugo/codes/mol-metal/.venv/bin/python "${TRAINER}" \
            --config "${DMPNN_CONFIG}" \
            --out-dir "${LOG_DIR}" \
            "${EXTRA_FLAGS[@]}"
        ;;
esac
