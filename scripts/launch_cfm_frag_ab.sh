#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# BLK-C11 single-variable A/B: does penalising disconnection reduce it?
#
# RENAMED from launch_cfm_genval_ab.sh. The soft-valence (gen_validity_weight)
# variant was dropped on 2026-10-10, not because the term is wrong but because
# the problem it was aimed at was a crashed metric: the 0/2146 "invalid valence"
# that motivated it turned out to be an AttributeError swallowed by a bare
# `except`, and the real value is 0.8495. See
# metrics/by_metric/cfm_decode_chemistry_validity.json (RETRACTION) and
# todo.json BLK-C11.
#
# WHAT THIS IS FOR
# ----------------
# Three real CFM runs generated 2146 molecules between them and NOT ONE has a
# valid valence (metrics/by_metric/cfm_decode_chemistry_validity.json). The
# differentiable term that could penalise that — valence_log_barrier, wired as
# soft_valence_term — sat at weight 0.0 in every config on disk. This runs the
# two arms that make that attributable:
#
#   control   configs/cfm_frag_control.yaml    assemble_loss_weight 0.0
#   assemble  configs/cfm_frag_assemble.yaml   assemble_loss_weight 1.0
#                                         assemble_formulation escape_time
#                                         assemble_hinge_tau    0.30
#
# The two configs differ in EXACTLY the three assemble keys, which travel
# together as one treatment. That is asserted before launch — see below.
#
# WHY TWO ARMS AND NOT "JUST TURN IT ON"
# -------------------------------------
# The previous attempt at this question (cfm_fm_ab vs cfm_fm_control) changed
# the flow-matching objective AND the LR isolation together, so when collapse
# still appeared at epoch 18 nobody could say whether anything had been fixed.
# One variable per run is the whole discipline here.
#
# AND WHY NOT SIMPLY TURNING ON THE EXISTING TERM
# ----------------------------------------------
# Because it would train nothing. Measured at the bond logits, the shipped
# product aggregation's gradient is ~5e-09 in every case — a diagnostic, not an
# objective. The treatment is therefore formulation + hinge + weight together;
# the hinge is what lets the treatment have a live gradient without violating
# the pinned "connected graphs get no gradient" property.
#
# §5.11: this wrapper passes --config and nothing else. device comes from the
# config (compute.device: auto -> cuda:0 when available), so no trainer flag is
# smuggled in on the command line.
# §5.9: single venv, absolute interpreter path.
# GPU: standing directive "只用gpu0吧".
#
# USAGE:  bash scripts/launch_cfm_frag_ab.sh [--dry-run]

set -uo pipefail

REPO_ROOT="/home/hugo/codes/mol-metal"
PY="${REPO_ROOT}/.venv/bin/python"
TRAINER="molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
PANEL="molmetal/scripts/cfm_property_panel.py"

CONTROL_CFG="configs/cfm_frag_control.yaml"
TREAT_CFG="configs/cfm_frag_assemble.yaml"
CONTROL_DIR="molmetal/reports/cfm_frag_control"
TREAT_DIR="molmetal/reports/cfm_frag_assemble"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

cd "${REPO_ROOT}" || exit 1

# ---------------------------------------------------------------- guard ----
# The A/B is only worth running if the configs really do differ by one key.
# A silent edit to either file would turn this into an unattributable run, and
# an unattributable run is exactly what already happened once.
echo "[frag-ab] checking the two arms differ in exactly one hyperparameter"
"${PY}" - <<'GUARD'
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "_t", "molmetal/scripts/train_cfm_yuelbond_wasserstein.py")
m = importlib.util.module_from_spec(spec); sys.modules["_t"] = m
spec.loader.exec_module(m)
KEYS = ("epochs","batch_size","lr","n_steps","method","gen_eval_n","gen_eval_seeds",
        "save_generated_smiles","early_stop_patience","bond_head_warmup_epochs",
        "entropy_reg_weight","mmd_weight","repulsion_weight","fm_weight",
        "gen_validity_weight","tmqm_metals","tmqm_dir","bond_decision",
        "connectivity_prior","resume_from")
# The three keys that ARE allowed to differ — they are one treatment, not three
# variables. They must be in the COMPARED set, not merely excluded from it:
# an earlier version listed them here but diffed only over KEYS, so `diff` was
# structurally always empty and the guard passed for the wrong reason.
TREATMENT = {"assemble_loss_weight", "assemble_formulation", "assemble_hinge_tau"}
vals = {}
for cfg in ("configs/cfm_frag_control.yaml","configs/cfm_frag_assemble.yaml"):
    sys.argv = ["t","--config",cfg]
    a = m.parse_args()
    vals[cfg] = {k: getattr(a,k) for k in KEYS}
    for k in TREATMENT:
        vals[cfg][k] = getattr(a, k, "<absent>")
c = vals["configs/cfm_frag_control.yaml"]; s = vals["configs/cfm_frag_assemble.yaml"]
diff = sorted(k for k in set(c) | TREATMENT
              if c.get(k, "<absent>") != s.get(k, "<absent>"))
print(f"[frag-ab] differing keys: {diff}")
assert set(diff) == TREATMENT, (
    f"A/B ABORTED: the arms differ in {diff}, expected exactly "
    f"{sorted(TREATMENT)}. The run would not be attributable.")
assert float(s["assemble_loss_weight"]) > 0, "the treatment arm has the term OFF"
assert s["assemble_hinge_tau"] > 0, (
    "the treatment arm has no dead zone — that would charge the head for "
    "uncertainty and break the pinned connected-graph property")
print("[frag-ab] guard PASS")
GUARD
[[ $? -ne 0 ]] && { echo "[frag-ab] FAIL: arm-divergence guard"; exit 1; }

if [[ "${DRY}" -eq 1 ]]; then
  echo "[frag-ab] DRY RUN — would execute, in order:"
  echo "  1+2) ${PY} ${TRAINER} --config ${CONTROL_CFG}  &  ... --config ${TREAT_CFG}  (CONCURRENT on GPU 0)"
  
  echo "  3) ${PY} ${PANEL} --smiles-csv <each>/generated_smiles.csv --decode-chemistry-validity"
  echo "[frag-ab] wall: UNKNOWN to better than a factor of ~2. The only epoch"
  echo "           timed on this config was the FIRST one (259 s), which carries"
  echo "           CUDA init and kernel compilation; the historical collapse runs"
  echo "           showed ~130-160 s/epoch steady state at gen_eval_n=16 / batch 16,"
  echo "           while this config uses gen_eval_n=32 x 4 seeds. Both arms run"
  echo "           concurrently on GPU 0."
  exit 0
fi

# ----------------------------------------------------------------- run -----
# Both arms run CONCURRENTLY on GPU 0. They are independent runs writing to
# their own out_dir, so there is no shared state; and the job is far too small
# to need the card to itself — MEASURED: one arm peaks at ~704 MiB of the
# 97340 MiB free, and the model is 518,276 params, so the run is bound by data
# loading and kernel-launch overhead rather than by arithmetic. Running them
# sequentially would leave the GPU mostly idle for hours.
#
# Both stay on GPU 0 per the standing "只用gpu0吧" directive. Concurrency does
# NOT weaken the A/B: the two arms do not share a process, a seed stream or a
# checkpoint directory, and neither reads the other's output.
run_arm () {
  local cfg="$1" dir="$2" tag="$3"
  echo "[frag-ab] === ${tag} start $(date -Is)"
  # §5.6: trainer writes its own artefacts; we only redirect the log.
  "${PY}" "${TRAINER}" --config "${cfg}" \
    > "${dir%/}.log" 2>&1
  local rc=$?
  echo "[frag-ab] === ${tag} exit ${rc} $(date -Is)"
  return ${rc}
}

mkdir -p "${CONTROL_DIR}" "${TREAT_DIR}"
run_arm "${CONTROL_CFG}" "${CONTROL_DIR}" CONTROL &
PID_CONTROL=$!
run_arm "${TREAT_CFG}" "${TREAT_DIR}" TREATMENT &
PID_SOFTVAL=$!
echo "[frag-ab] both arms launched: control pid ${PID_CONTROL}, assemble-term pid ${PID_TREAT}"

wait "${PID_CONTROL}"; RC_CONTROL=$?
wait "${PID_TREAT}"; RC_SOFTVAL=$?
[[ ${RC_CONTROL} -ne 0 ]] && echo "[frag-ab] control arm FAILED (rc=${RC_CONTROL})"
[[ ${RC_TREAT} -ne 0 ]] && echo "[frag-ab] assemble-term arm FAILED (rc=${RC_TREAT})"
[[ ${RC_CONTROL} -ne 0 && ${RC_TREAT} -ne 0 ]] && {
  echo "[frag-ab] BOTH arms failed — the comparison is void, do not score."; }

# ---------------------------------------------------------------- score ----
# The primary metric is chemistry_validity_pass_rate over the molecules each
# arm actually decoded. It reads 0.000 for every historical run, so it is the
# one number that can move off the floor.
echo "[frag-ab] === scoring decoded molecules"
for d in "${CONTROL_DIR}" "${TREAT_DIR}"; do
  if [[ -s "${d}/generated_smiles.csv" ]]; then
    "${PY}" "${PANEL}" \
      --smiles-csv "${d}/generated_smiles.csv" \
      --decode-chemistry-validity \
      --out-json "${d}/property_panel.json" || echo "[frag-ab] panel failed for ${d}"
  else
    echo "[frag-ab] SKIP panel for ${d}: no generated_smiles.csv"
  fi
done

echo "[frag-ab] === DONE $(date -Is)"
echo "[frag-ab] next: compare chemistry_validity_pass_rate in the two"
echo "             property_panel.json files, and report bond_f1 alongside it."
echo "             A single bond_f1 number is the known misleading report —"
echo "             see docs/00_MODEL_NARRATIVE.md §5."
