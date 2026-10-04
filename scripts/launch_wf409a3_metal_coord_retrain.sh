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
# scripts/launch_wf409a3_metal_coord_retrain.sh
#
# Wf-409-A3 — Wf-246-C metal-coordination term retrain launch wrapper
# (USER_OK-gated, GPU launch deferred).
#
# Sister to Wf-406-A4 launch wrapper at
# scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh (216 LOC); this
# module is the Wf-409-A3 mirror with identical surface so the
# Wf-355-B1 USER_OK-gated wrapper family can wrap either module
# interchangeably.
#
# Usage:
#   USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --dry-run
#   USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --smoke
#   USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh \
#       --metals "Ru Ir Rh" --seeds 0 1 2 --epochs 50
#   USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh \
#       --metals "Ru Ir Rh Os Re" --seeds 0 1 2 3 4 --epochs 50 --actual
#
# Wf-424-A2 fix: `--metals` + `--seeds` accept both quoted-string form
# ("Ru Ir Rh") AND unquoted multi-token form (Ru Ir Rh --seeds 0 1 2);
# greedy tokenisation captures all tokens up to next `--*` flag.
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

# Locate repo root + scaffold script + real trainer
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
SCAFFOLD="${PROJECT_ROOT}/molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py"
TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# Wf-432-A2 — Wf-246-C trainer (real D-MPNN trainer with the
# MetalCoordinationTerm wire-in via --metal-coord-term).  R3_actual
# invokes the real trainer; R2_smoke still invokes the SCAFFOLD for
# the 5-step CPU safety net (per Wf-431-A3 §3.2.3).

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
OUT_DIR="molmetal/reports/wf409a3_metal_coord_launch"

# Wf-424-A2 fix: greedy tokenisation pattern (nargs="+" equivalent) for
# `--metals` and `--seeds`. Captures all tokens until the next `--*` flag
# so both `--metals "Ru Ir Rh"` (quoted single string) and
# `--metals Ru Ir Rh` (unquoted multi-token) work. Matches the SCAFFOLD
# argparse `nargs="+"` contract for these two flags.
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
        -h|--help)
            printf 'Wf-409-A3 launch wrapper (USER_OK-gated, Wf-424-A2 greedy tokenisation)\n'
            printf 'Usage: USER_OK=1 %s [--dry-run|--smoke|--actual] [--metals M1 M2 ...] [--seeds 0 1 2 ...] [--epochs 50]\n' "$0"
            printf '  --metals + --seeds accept both quoted ("Ru Ir Rh") and unquoted (Ru Ir Rh) forms.\n'
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
if [ ! -f "${TRAINER}" ]; then
    die "Trainer script missing: ${TRAINER}"
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
# OUT_DIR may be absolute or relative — handle both correctly.
case "${OUT_DIR}" in
    /*) LOG_DIR="${OUT_DIR}" ;;
    *)  LOG_DIR="${PROJECT_ROOT}/${OUT_DIR}" ;;
esac
mkdir -p "${LOG_DIR}"

FLAGS="(--metal-coord-term --metals ${METALS} --seeds ${SEEDS} --epochs ${EPOCHS})"

# Build the pair_args array: each --pair on its own argv element so the
# helper sees --pair as a discrete flag, not a glob.
PAIR_ARGS=()
for pair in "${PAIRS[@]}"; do
    PAIR_ARGS+=( --pair "${pair}" )
done

emit_dry_run_log "${LOG_DIR}/DRY_RUN.log" \
    --title "Wf-409-A3 Wf-246-C metal-coord retrain" \
    --wrapper "scripts/launch_wf409a3_metal_coord_retrain.sh" \
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
    "wf": "Wf-409-A3",
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
    "scaffold_script": "molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py",
    "sister_module": "molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py",
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
        printf '%s[Wf-409-A3]%s R1 dry-run mode: SCAFFOLD script exists; '
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf 'pre-flight path validated; %d pairs; total budget %s hr.\n' \
            "${#PAIRS[@]}" "${TOTAL_BUDGET_HR}"
        printf 'DRY_RUN.log: %s/DRY_RUN.log\n' "${LOG_DIR}"
        printf 'plan.json:   %s/plan.json\n' "${LOG_DIR}"
        exit 0
        ;;
    R2_smoke)
        printf '%s[Wf-409-A3]%s R2 SCAFFOLD smoke mode: 5-step CPU smoke.\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        exec python3 "${SCAFFOLD}" \
            --metal-coord-term \
            --metals "${METALS_ARR[@]}" \
            --seeds "${SEEDS_ARR[@]}" \
            --epochs 5 \
            --smoke \
            --out-dir "${LOG_DIR}"
        ;;
    R3_actual)
        # Wf-432-A2 — Fix-B: real trainer wire-in replaces the Wf-431-B3
        # SCAFFOLD-fallback.  The R3_actual case now invokes the real
        # Wf-131 P3 D-MPNN trainer (``train_dmpnn_multitask_wasserstein.py``)
        # with the Wf-246-C metal-coord term wire-in (3 new CLI flags:
        # --metal-coord-term + --metal-coord-weight + --cal-sign-vina-convention
        # added in Wf-432-A2 per Wf-441B §3 + Wf-427-A1 line 641).
        # The kcal sign fix is applied per Wf-427-A1 line 641
        # (1-2*total)*kcal_per_unit; lower kcal/mol = better binding
        # per Vina convention.  The trainer is invoked on cuda:0 by
        # default (per AGENTS.md §5.1 + Wf-358 GPU standing grant).
        # Wall budget ~3 hr for the canonical 5-metal × 5-seed × 50-epoch
        # Wf-131 P3 protocol per Wf-390-A1 Path A.
        printf '%s[Wf-409-A3]%s R3 ACTUAL mode: invoking REAL trainer (Wf-131 P3 + Wf-246-C metal-coord term wire-in).\n' \
            "$(tput bold 2>/dev/null || true)" "$(tput sgr0 2>/dev/null || true)"
        printf '  trainer: %s\n' "${TRAINER}"
        printf '  kcal sign fix per Wf-427-A1 line 641 (Vina convention)\n'
        printf '  kcal weight: --metal-coord-weight 0.5 (canonical Wf-246-C)\n'
        printf '  epochs: %d × %d (metal, seed) pairs = %d total\n' \
            "${EPOCHS}" "${#PAIRS[@]}" \
            "$((EPOCHS * ${#PAIRS[@]}))"
        PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        exec .venv/bin/python "${TRAINER}" \
            --metal-coord-term \
            --metal-coord-weight 0.5 \
            --cal-sign-vina-convention \
            --metals "${METALS_ARR[@]}" \
            --seeds "${SEEDS_ARR[@]}" \
            --epochs ${EPOCHS} \
            --out-dir "${LOG_DIR}"
        ;;
esac
