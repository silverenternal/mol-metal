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
# scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh
#
# Wf-406-A4 — Wf-246-C metal-coordination term retrain launch wrapper
# (USER_OK-gated, GPU launch deferred).
#
# Usage:
#   USER_OK=1 bash scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh --dry-run
#   USER_OK=1 bash scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh --smoke
#   USER_OK=1 bash scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh \
#   USER_OK=1 bash scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh \
#
# Exit codes:
#   0   PASS (smoke or --dry-run)
#   1   FAIL
#   64  USER_OK gate refused (USER_OK env not set to "1")
#
# Required: USER_OK=1 in env (explicit per AGENTS.md §5.1 + Wf-355-B1 pattern).
# The wrapper REFUSES to run unless USER_OK=1 is set, including for --dry-run.
# -----------------------------------------------------------------------------
set -euo pipefail

# Locate repo root + scaffold script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
SCAFFOLD="${PROJECT_ROOT}/molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py"
# 2026-10-08 §5.9: canonical single-venv interpreter (bare python3 lacks torch/pandas).
PY="${PROJECT_ROOT}/.venv/bin/python"

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
# CLI arg parsing
# -----------------------------------------------------------------------------
DRY_RUN=0
SMOKE=0
ACTUAL=0
METALS="Ru Ir Rh Os Re"
SEEDS="0 1 2"
EPOCHS=50
OUT_DIR="molmetal/reports/wf406a4_metal_coord_launch"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --actual) ACTUAL=1; shift ;;
        # 2026-11-04 (Wf-A269 script-contract repair — adjacent finding in the
        # same wrapper): the wrapper's OWN help text documents
        # `[--metals M1 M2] [--seeds 0 1 2]`, and the repo's own regression
        # test drives it that way (`--metals Ru Ir Rh --seeds 0`,
        # molmetal/tests/test_wf406a4_metal_coord_scaffold.py), but
        # `--metals) METALS="$2"; shift 2 ;;` consumed ONE argv token — so the
        # SECOND metal became the next FLAG and the wrapper died with
        # "[FATAL] unknown flag: Ir" (rc 1) BEFORE the scaffold was reached:
        # the R2/R3 branches were unreachable in the documented spelling and
        # the repo's own test was RED. `--metals`/`--seeds` now consume every
        # following NON-FLAG token, so BOTH spellings work:
        #     --metals "Ru Ir Rh"   (one quoted token — the old behaviour)
        #     --metals Ru Ir Rh     (three argv tokens — the documented form)
        # A missing value is still REFUSED (`die`), never silently empty.
        #
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #   --metals) METALS="$2"; shift 2 ;;
        #   --seeds) SEEDS="$2"; shift 2 ;;
        --metals) shift; METALS=""; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do METALS="${METALS:+$METALS }$1"; shift; done; [ -n "$METALS" ] || die "--metals needs at least one metal";;
        --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do SEEDS="${SEEDS:+$SEEDS }$1"; shift; done; [ -n "$SEEDS" ] || die "--seeds needs at least one seed";;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            printf 'Wf-406-A4 launch wrapper (USER_OK-gated)\n'
            printf 'Usage: USER_OK=1 %s [--dry-run|--smoke|--actual] [--metals M1 M2] [--seeds 0 1 2] [--epochs 50]\n' "$0"
            exit 0
            ;;
        *) die "unknown flag: $1" ;;
    esac
done

# -----------------------------------------------------------------------------
# USER_OK gate (explicit, enforced BEFORE any GPU work)
# -----------------------------------------------------------------------------
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --smoke      # 5-step CPU SCAFFOLD smoke\n' "$0" >&2
    printf '       USER_OK=1 bash %s --actual     # actual ~5 hr GPU run\n' "$0" >&2
    exit 64
fi

# -----------------------------------------------------------------------------
# Scaffold script existence check
# -----------------------------------------------------------------------------
if [ ! -f "${SCAFFOLD}" ]; then
    die "SCAFFOLD script missing: ${SCAFFOLD}"
fi

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
# Per-pair budget: assume ~20 min/seed on RTX 5090 = ~5 hr for 15 pairs
# -----------------------------------------------------------------------------
PAIR_BUDGET_HR=0.33
TOTAL_BUDGET_HR=$(awk -v n="${#PAIRS[@]}" -v b="${PAIR_BUDGET_HR}" 'BEGIN { printf "%.2f", n*b }')

# -----------------------------------------------------------------------------
# Emit DRY_RUN.log per Wf-363-A1 schema
# -----------------------------------------------------------------------------
# 2026-11-04 (Wf-A269 script-contract repair — adjacent findings, same
# wrapper). Two more defects stood between this wrapper and a runnable
# launch, both PROVEN by driving it on CPU:
#
# (B) `--out-dir` was ALWAYS concatenated onto PROJECT_ROOT, so an ABSOLUTE
#     out-dir ("/tmp/...") became "<repo>//tmp/..." and the run's artefacts
#     (DRY_RUN.log / plan.json / result.json) landed somewhere the caller
#     never named. A relative out-dir stays repo-relative (unchanged); an
#     absolute one is honoured as given.
#
# (C) `--pair` was passed ONCE with an array expansion
#     (`--pair "${PAIRS[@]}"`), but `scripts/_dry_run_log_helper.sh` parses
#     `--pair` as a SINGLE-VALUE option (the helper's own usage example
#     repeats it: `--pair "Re seed=0" --pair "Os seed=4"`). With more than
#     one pair the second label fell through to the helper's `*)` branch:
#     "[FATAL] emit_dry_run_log: unknown arg Ir seed=0" (rc 1) ⇒ with
#     `set -e` the wrapper DIED at the DRY_RUN.log step — i.e. with its
#     DEFAULT cohort (5 metals x 3 seeds = 15 pairs) it never reached the
#     scaffold at all. `--pair` is now repeated once per pair.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}"
#   ...
#   emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
#       ... \
#       --pair "${PAIRS[@]}"
case "${OUT_DIR}" in
    /*) LOG_DIR="${OUT_DIR}" ;;
    *)  LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}" ;;
esac
mkdir -p "${LOG_DIR}"

FLAGS="(--metal-coord-term --metals ${METALS} --seeds ${SEEDS} --epochs ${EPOCHS})"

# One `--pair` per enumerated pair (the helper's single-value option).
PAIR_ARGS=()
for p in "${PAIRS[@]}"; do
    PAIR_ARGS+=(--pair "${p}")
done

emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
    --title "Wf-406-A4 Wf-246-C metal-coord retrain" \
    --wrapper "scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh" \
    --mode "${MODE}" \
    --trainer "${SCAFFOLD}" \
    --subset "5-metal Path A (Ru/Ir/Rh/Os/Re)" \
    --flags "${FLAGS}" \
    --gpu "cpu" \
    --budget-hr "${TOTAL_BUDGET_HR}" \
    --user-ok "REQUIRED" \
    "${PAIR_ARGS[@]}"

# Convert METALS and SEEDS space-separated strings to bash arrays for
# proper argv construction (avoid word-splitting on spaces).
METALS_ARR=(${METALS})
SEEDS_ARR=(${SEEDS})

# -----------------------------------------------------------------------------
# Emit plan.json
# -----------------------------------------------------------------------------
cat > "${LOG_DIR}/plan.json" <<EOF
{
    "wf": "Wf-406-A4",
    "deliverable": "Wf-246-C metal-coordination term retrain launch wrapper",
    "mode": "${MODE}",
    "user_ok_required": true,
    "user_ok_env_var": "USER_OK=1",
    "metals": [$(printf '"%s",' ${METALS} | sed 's/,$//')],
    "seeds": [$(printf '%s,' ${SEEDS} | sed 's/,$//')],
    "epochs": ${EPOCHS},
    "n_pairs": ${#PAIRS[@]},
    "pair_budget_hr": ${PAIR_BUDGET_HR},
    "total_budget_hr": ${TOTAL_BUDGET_HR},
    "scaffold_script": "molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py",
    "honest_framing": {
        "metal_coord_sign_fix_id": "Wf-333-V1 line 635",
        "scaffold_only": true,
        "gpu_launch_blocked_on_user_ok": true,
        "actual_gpu_run_requires_user_ok_1": true
    },
    "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# -----------------------------------------------------------------------------
# Branch on mode
# -----------------------------------------------------------------------------
case "${MODE}" in
    R1_dry_run)
        printf '%s[Wf-406-A4]%s R1 dry-run mode: SCAFFOLD script exists; '
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf 'pre-flight path validated; %d pairs; total budget %s hr.\n' \
            "${#PAIRS[@]}" "${TOTAL_BUDGET_HR}"
        printf 'DRY_RUN.log: %s/DRY_RUN.log\n' "${LOG_DIR}"
        printf 'plan.json:   %s/plan.json\n' "${LOG_DIR}"
        exit 0
        ;;
    R2_smoke)
        printf '%s[Wf-406-A4]%s R2 SCAFFOLD smoke mode: 5-step CPU smoke.\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        exec "${PY}" "${SCAFFOLD}" \
            --metal-coord-term \
            --metals "${METALS_ARR[@]}" \
            --seeds "${SEEDS_ARR[@]}" \
            --epochs 5 \
            --smoke \
            --out-dir "${LOG_DIR}"
        ;;
    R3_actual)
        printf '%s[Wf-406-A4]%s R3 ACTUAL mode: would launch full GPU run.\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf '  ACTUAL GPU launch BLOCKED per AGENTS.md §5.1 + kill_then_adapt §0\n'
        printf '  Use Wf-355-B1 USER_OK-gated wrapper pattern for actual run.\n'
        printf '  Falling back to R2 SCAFFOLD smoke for now.\n'
        exec "${PY}" "${SCAFFOLD}" \
            --metal-coord-term \
            --metals "${METALS_ARR[@]}" \
            --seeds "${SEEDS_ARR[@]}" \
            --epochs ${EPOCHS} \
            --no-smoke \
            --out-dir "${LOG_DIR}"
        ;;
esac
