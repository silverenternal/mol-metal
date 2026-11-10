#!/usr/bin/env bash
# wf_reproduce_R17.sh — single-command reproducibility driver for R17 outputs.
#
# TODO IMPROVEMENT_BACKLOG extension item G1 (2026-09-25):
# verify that one command + seed=0 can reproduce all key R17 outputs.
#
# Sections
# --------
#   1. eval_all_evaluators.py               — expect 14/14 PASS
#   2. wf_pitfall_guard/check_all.py        — expect 7/7 PASS
#   3. 200-step R17 CFM smoke               — seed=0, hidden-dim=64, n-layers=2
#   4. r4_lambda_only_run.py 10pocketsx1seed — n-simulations=100, metal-seed cisplatin
#   5. VinaDockingAdapter instantiation smoke
#   6. SA + QED on 3 test SMILES
#   7. pitfall guards --json output (machine-readable)
#   8. checkpoint reproducibility: 100-step smoke twice, same seed
#
# Exit code: 0 on success (final line: ALL DONE); 1 if any section fails.
# Wall budget: ~25 min on CPU (the CFM smoke dominates); <30 min target.
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     source env.sh
#     bash molmetal/scripts/wf_reproduce_R17.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/env.sh"
echo "[wf_reproduce_R17] REPO_ROOT=${REPO_ROOT}"
T_START=$(date +%s)

_section_fail() {
    echo "[wf_reproduce_R17] FAIL section=$1 rc=$2" >&2
    exit 1
}

# --- Section 1 — eval_all_evaluators.py (expect 14/14 PASS) -------------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 1 — eval_all_evaluators.py (expect 14/14)"
echo "================================================================"
uv run --no-sync python molmetal/scripts/eval_all_evaluators.py || _section_fail 1 $?

# --- Section 2 — wf_pitfall_guard/check_all.py (expect 7/7 PASS) --------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 2 — pitfall guards (expect 7/7 PASS)"
echo "================================================================"
uv run --no-sync python molmetal/scripts/wf_pitfall_guard/check_all.py || _section_fail 2 $?

# --- Section 3 — 200-step R17 CFM smoke (seed=0, hidden-dim=64, n-layers=2)
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 3 — 200-step CFM smoke"
echo "================================================================"
REPRO_SMOKE_DIR="${REPO_ROOT}/molmetal/reports/wf_reproduce_R17_smoke"
mkdir -p "${REPRO_SMOKE_DIR}"
uv run --no-sync python molmetal/scripts/metallo_drug_smoke_retrain.py \
    --smoke --seed 0 --hidden-dim 64 --n-layers 2 \
    --out-dir "${REPRO_SMOKE_DIR}" || _section_fail 3 $?
[[ -f "${REPRO_SMOKE_DIR}/final.json" ]] || _section_fail 3 2

# --- Section 4 — r4_lambda_only_run 10 pockets x 1 seed x nsim=100 -----
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 4 — r4_lambda_only_run 10pockets x 1seed x nsim=100"
echo "================================================================"
mkdir -p /tmp/repro_lambda
uv run --no-sync python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seed 0 --n-simulations 100 --metal-seed cisplatin \
    --output-dir reproduce_R17_lambda || _section_fail 4 $?

# --- Section 5 — VinaDockingAdapter instantiation ----------------------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 5 — VinaDockingAdapter instantiation"
echo "================================================================"
uv run --no-sync python -c "
from molmetal.molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
a = VinaDockingAdapter()
print(f'VinaDockingAdapter instantiated: {type(a).__name__}')
print('OK')
" || _section_fail 5 $?

# --- Section 6 — SA + QED on 3 test SMILES -----------------------------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 6 — SA + QED on 3 test SMILES"
echo "================================================================"
uv run --no-sync python -c "
from molmetal.molmetal_lam.sbdd_env.sa_score import sa_score_ertl
from molmetal.molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles
TEST = {
    'cisplatin': '[NH3][Pt]([NH3])([Cl])[Cl]',
    'aspirin':   'CC(=O)Oc1ccccc1C(=O)O',
    'taxol':     'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C8=CC=CC=C8)OC(=O)C9=CC=CC=C9)OC(=O)C',
}
for name, smi in TEST.items():
    sa = sa_score_ertl(smi); qed = qed_from_smiles(smi)
    assert sa == sa and qed == qed, f'NaN metric for {name}'
    print(f'  {name:10s} SA={sa:.3f}  QED={qed:.3f}')
print('OK')
" || _section_fail 6 $?

# --- Section 7 — pitfall guards --json output --------------------------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 7 — pitfall guards --json output"
echo "================================================================"
uv run --no-sync python molmetal/scripts/wf_pitfall_guard/check_all.py --json \
    > /tmp/wf_reproduce_R17_pitfalls.json || _section_fail 7 $?

# --- Section 8 — checkpoint reproducibility: 2x100-step smoke -----------
echo "================================================================"
echo "[wf_reproduce_R17] SECTION 8 — checkpoint reproducibility (2x100 steps)"
echo "================================================================"
REPRO_CKPT_DIR="${REPO_ROOT}/molmetal/reports/wf_reproduce_R17_ckpt"
mkdir -p "${REPRO_CKPT_DIR}/runA" "${REPRO_CKPT_DIR}/runB"
for run in A B; do
    uv run --no-sync python molmetal/scripts/metallo_drug_smoke_retrain.py \
        --steps 100 --batch-size 4 --decode-smoke-every 100 \
        --seed 0 --hidden-dim 64 --n-layers 2 \
        --out-dir "${REPRO_CKPT_DIR}/run${run}" || _section_fail 8 $?
done
LOSS_A=$(uv run --no-sync python -c "
import json; print(float(json.load(open('${REPRO_CKPT_DIR}/runA/final.json'))['final_loss']))")
LOSS_B=$(uv run --no-sync python -c "
import json; print(float(json.load(open('${REPRO_CKPT_DIR}/runB/final.json'))['final_loss']))")
# 2026-11-09 REPAIR (§5.9, additive + dated; the line above is the
# SUPERSEDED rendering, preserved VERBATIM): this invocation was a BARE
# `python3`, i.e. the SYSTEM interpreter.  AGENTS.md §5.9 forbids a bare
# python, and the §5.9 gate could not SEE it -- scripts/_wrapper_audit.py's
# _ABS_PY_RE / _REL_PY_RE both require a "/", so a bare `python3` word is
# invisible to the census.  MEASURED over the 113 tracked `*.sh`: this was
# the LAST bare invocation (BARE-INVOCATION 1 / WRAPPED 0 / MENTION 50).
# It is now routed through the SAME interpreter this file already uses for
# all ten of its other python invocations -- `uv run --no-sync python`,
# which env.sh pins to the repo-local canonical venv (env.sh:39 exports
# UV_PROJECT_ENVIRONMENT="$CANONICAL_VENV").  The snippet is stdlib-only,
# so its computation is unchanged.
uv run --no-sync python -c "
a = float('${LOSS_A}'); b = float('${LOSS_B}')
diff = abs(a - b); print(f'final_loss delta: {diff:.6e}')
assert diff < 1e-4, f'final_loss delta {diff} exceeds 1e-4 — ckpt not reproducible!'
print('OK — checkpoint reproduction within float32 epsilon')
" || _section_fail 8 $?

# --- Final marker -------------------------------------------------------
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo "================================================================"
echo "[wf_reproduce_R17] ALL DONE — 8/8 sections PASS in ${ELAPSED}s"
echo "[wf_reproduce_R17]   Section 3: ${REPRO_SMOKE_DIR}/final.json"
echo "[wf_reproduce_R17]   Section 4: molmetal/reports/wf_lambda1_reproduce_R17_lambda/"
echo "[wf_reproduce_R17]   Section 7: /tmp/wf_reproduce_R17_pitfalls.json"
echo "[wf_reproduce_R17]   Section 8: ${REPRO_CKPT_DIR}/runA/ + runB/"
echo "================================================================"
echo "ALL DONE"
