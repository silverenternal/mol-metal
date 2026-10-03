#!/bin/bash
# Wf-264 — CFM generated molecule SA + QED property panel launch wrapper
# Wf-394-A2 — extend with diversity-aware eval + SA-aware runner + DRY_RUN.log
#
# Owner: Hugo Lee
# Date: 2026-09-29 (Wf-334-U3 ultracode; closes Wf-298 §1.3 "launch script not yet written")
# Extended: 2026-09-30 (Wf-394-A2 ultracode)
# Parent: TODO/pending/wf264_cfm_property_panel.md (implicit per INDEX.md §22.7
#         "Property panel status: NOT STARTED at Wf-265-Collect time (no launch
#         script on disk)") + Wf-264 audit ship 2026-09-28 (paper §4 90 claims,
#         5 §3.5 violations fixed per Wf-311A).
# Driver: molmetal/scripts/cfm_property_panel.py (SA score Ertl & Schuffenhauer
#         2009 + QED Bickerton 2012 + per-metal + global aggregates)
# Runner: molmetal/scripts/wf394a2_property_panel_runner.py (Wf-394-A2 NEW —
#         composes SA+QED, SA-aware penalty, diversity pass on the cohort)
#
# Per AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing:
#   - The panel script is CPU-only (RDKit SA + QED + scaffold extraction +
#     Morgan fingerprint Tanimoto computation).  NO GPU launches, NO
#     torch.compile / BF16 / TF32 wiring.  Those Wf-328 flags apply to
#     the D-MPNN Wasserstein trainer, NOT to this post-hoc property panel.
#   - Wf-328-W2 --num-workers 8 is documented but NOT wired: the panel
#     script does not use a DataLoader; SMILES are processed sequentially
#     in a single process.  This is honest-negative per §5.2.
#   - Per AGENTS.md §5.1 (standing GPU auth) + §5.2 (MEASURED vs
#     PROJECTED tag discipline), this launcher requires:
#       (a) pre_flight_check.py --pipeline all PASS
#       (b) USER_OK gate (kill_then_adapt standing directive; even CPU
#           launches honour the gate for reproducibility audits).
#
# Honest framing (party-twenty-spirit §3.5):
#   - The Wf-264 panel IS MEASURED on the Wf-188 MetalCytoToxDB ligand
#     population (n=26,801) per molmetal/reports/wf264_panel/panel_report.md
#     (2026-09-28 run).  This launch script makes that run RE-PRODUCIBLE
#     end-to-end (CSV existence check + pre-flight + USER_OK gate + dry-run).
#   - Wf-394-A2 ADDS: diversity-aware eval (--diversity-aware-eval) + SA-aware
#     penalty (--sa-aware) flags that wire into the Wf-394-A2 runner script.
#     These flags are MEASURED-only; no TRAIN-time mutation is performed.
#     Wall budget ~30 s additional for n=10k cohort (PROJECTED).
#   - Vina sub-panel: MEASURED 0/100 docked ok on 1h36 (meeko 0.8.0
#     metal-atom limitation, honest-negative per Wf-264 §2).
#   - Generated SMILES note: Wf-188 did NOT retain individual generated
#     SMILES, so the panel ran on the CFM-input ligand population as the
#     closest MEASURED analogue. See Wf-264 panel_report.md §4.
#
# Usage:
#   bash scripts/launch_wf264_cfm_property_panel.sh                 # real launch
#   bash scripts/launch_wf264_cfm_property_panel.sh --dry-run       # verification only
#   USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh       # user-authorised launch
#   # Wf-394-A2 NEW flags (env-var wire-in):
#   USER_OK=1 DIVERSITY_AWARE_EVAL=1 bash scripts/launch_wf264_cfm_property_panel.sh
#   USER_OK=1 SA_AWARE=1 bash scripts/launch_wf264_cfm_property_panel.sh
#   USER_OK=1 DIVERSITY_AWARE_EVAL=1 SA_AWARE=1 bash scripts/launch_wf264_cfm_property_panel.sh

set -e
cd /home/hugo/codes/mol-metal

OUT_BASE="results/wf264_cfm_property_panel"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema)
DRY_RUN_LOG_HELPER="$(dirname "$0")/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck disable=SC1091
source "${DRY_RUN_LOG_HELPER}"

# Per AGENTS.md §5.1 standing GPU auth + Wf-330-B2 §1 venv convention:
# the Mol-Metal venv is /home/hugo/codes/mol-metal/.venv/bin/python
# (NOT /home/hugo/cades/CadButEaas/.venv-vlm/bin/python which is the
# unrelated CadButEaas VLM venv).
PY="/home/hugo/codes/mol-metal/.venv/bin/python"
RUNNER="molmetal/scripts/wf394a2_property_panel_runner.py"

# Per Wf-264 panel_report.md §1 + §5 — inputs:
#   - SMILES CSV:  Wf-188 MetalCytoToxDB ligand population (26,801 rows)
#   - Output dir:   results/wf264_cfm_property_panel/{aggregate,per_molecule,per_metal,per_pocket}.json
SMILES_CSV="/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv"

# Per AGENTS.md §5.2 — Wf-328 flags NOT wired (honest-negative):
#   --torch-compile        NA (RDKit compute, no torch.compile path)
#   --bf16                 NA (RDKit compute, no autocast path)
#   --tf32                 NA (RDKit compute, no matmul path)
#   --w2-gradient-mode     NA (no W₂ loss in property panel)
#   --num-workers          NA (sequential SMILES loop, no DataLoader)

# Wf-394-A2 NEW: opt-in diversity + SA-aware flags via env-var wire-in.
# Default off to preserve Wf-264 backward compatibility.
DIVERSITY_AWARE_EVAL="${DIVERSITY_AWARE_EVAL:-0}"
SA_AWARE="${SA_AWARE:-0}"
DIVERSITY_SAMPLE_SIZE="${DIVERSITY_SAMPLE_SIZE:-1000}"
DIVERSITY_SEED="${DIVERSITY_SEED:-42}"
SA_TARGET="${SA_TARGET:-2.8}"
SA_WEIGHT="${SA_WEIGHT:-0.3}"

RUNNER_ARGS=(
  --smiles-csv "${SMILES_CSV}"
  --out-json "${OUT_BASE}/aggregate.json"
)
if [[ "${DIVERSITY_AWARE_EVAL}" == "1" ]]; then
  RUNNER_ARGS+=(--diversity-aware-eval --diversity-sample-size "${DIVERSITY_SAMPLE_SIZE}" --diversity-seed "${DIVERSITY_SEED}")
fi
if [[ "${SA_AWARE}" == "1" ]]; then
  RUNNER_ARGS+=(--sa-aware --sa-target "${SA_TARGET}" --sa-weight "${SA_WEIGHT}")
fi

# ---- DRY-RUN mode: verify config + CSV paths + emit DRY_RUN.log + plan.json ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-264 / Wf-394-A2 CFM property panel — DRY-RUN (CPU verification only) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "RUNNER: ${RUNNER}"
  echo "SMILES_CSV: ${SMILES_CSV}"
  echo "RUNNER_ARGS: ${RUNNER_ARGS[*]}"
  echo ""
  echo "--- CSV sanity check ---"
  test -f "${SMILES_CSV}" && echo "OK: SMILES CSV exists (${SMILES_CSV})" || { echo "FAIL: SMILES CSV missing"; exit 1; }
  n_rows=$(wc -l < "${SMILES_CSV}")
  echo "n_rows (incl header): ${n_rows}"
  echo ""
  echo "--- Pre-flight check (Wf-262 §10.1 + AGENTS.md §5.1) ---"
  # Dry-run: invoke pre-flight for SURFACE visibility but do NOT exit on
  # FAIL (per AGENTS.md §6 + Wf-324D pre-flight v3 state: 42 PASS / 15
  # FAIL / 57 TOTAL HONEST-NEGATIVE REGRESSION — 15 known FAILs are
  # project state, not a bug in this launcher).  The real-launch path
  # below DOES gate exit on pre-flight FAIL.
  ${PY} molmetal/scripts/pre_flight_check.py --pipeline all || echo "PREFLIGHT_FAIL_ACK (dry-run: surface only; real launch gates)"
  echo ""
  echo "--- Wf-328 optimization flags: NA for this panel ---"
  echo "Per AGENTS.md §5.2 honest framing: --torch-compile / --bf16 / --tf32"
  echo "/ --w2-gradient-mode / --num-workers do NOT apply to the property"
  echo "panel (RDKit SA + QED compute on pre-existing SMILES; no torch"
  echo "matmul, no DataLoader, no W₂ loss).  Documented in launcher header."
  echo ""
  echo "--- Wf-394-A2 NEW flags (env-var wire-in) ---"
  echo "DIVERSITY_AWARE_EVAL=${DIVERSITY_AWARE_EVAL} (env-var; default 0)"
  echo "SA_AWARE=${SA_AWARE} (env-var; default 0)"
  echo "DIVERSITY_SAMPLE_SIZE=${DIVERSITY_SAMPLE_SIZE}"
  echo "DIVERSITY_SEED=${DIVERSITY_SEED}"
  echo "SA_TARGET=${SA_TARGET}"
  echo "SA_WEIGHT=${SA_WEIGHT}"
  echo ""
  echo "--- Wf-264 panel MEASURED 2026-09-28 (per panel_report.md) ---"
  echo "  SA global: mean = 3.289, std = 0.851, p50 = 3.16"
  echo "  QED global: mean = 0.280, std = 0.185, p50 = 0.22"
  echo "  Vina: 0/100 docked ok on 1h36 (meeko metal-atom limit; honest-negative)"
  echo ""
  echo "--- Wf-363-A1: emit DRY_RUN.log via helper ---"
  emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
    --title "Wf-394-A2 Wf-264 CFM property panel (USER_OK-gated)" \
    --wrapper "launch_wf264_cfm_property_panel.sh" \
    --mode "DRY_RUN" \
    --trainer "${RUNNER}" \
    --subset "metalcytotox_rdkit_features" \
    --flags "(--smiles-csv ${SMILES_CSV} --diversity-aware-eval ${DIVERSITY_AWARE_EVAL} --sa-aware ${SA_AWARE})" \
    --gpu "cpu" \
    --budget-hr "0.5" \
    --user-ok "REQUIRED_REAL_LAUNCH" \
    --pair "metalcytotox_rdkit_features 26801 molecules"
  echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log"

  echo "--- Wf-363-A1: emit plan.json ---"
  PLAN_JSON="${OUT_BASE}/plan.json"
  {
    printf '{\n'
    printf '  "launcher": "launch_wf264_cfm_property_panel.sh",\n'
    printf '  "wave": "Wf-394-A2",\n'
    printf '  "mode": "DRY_RUN",\n'
    printf '  "smiles_csv": "%s",\n' "${SMILES_CSV}"
    printf '  "n_rows_smiles_csv": %d,\n' "${n_rows}"
    printf '  "out_json": "%s/aggregate.json",\n' "${OUT_BASE}"
    printf '  "runner": "%s",\n' "${RUNNER}"
    printf '  "diversity_aware_eval": %s,\n' "${DIVERSITY_AWARE_EVAL}"
    printf '  "sa_aware": %s,\n' "${SA_AWARE}"
    printf '  "diversity_sample_size": %s,\n' "${DIVERSITY_SAMPLE_SIZE}"
    printf '  "diversity_seed": %s,\n' "${DIVERSITY_SEED}"
    printf '  "sa_target": %s,\n' "${SA_TARGET}"
    printf '  "sa_weight": %s,\n' "${SA_WEIGHT}"
    printf '  "wall_budget_hr": 0.5,\n'
    printf '  "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1 (panel is CPU-only but reproducibility audit gate still enforced)",\n'
    printf '  "MEASURED_today": "wrapper ready; USER_OK gate enforced; DRY_RUN.log + plan.json emitted",\n'
    printf '  "ts": "%s"\n' "$(date -Iseconds)"
    printf '}\n'
  } > "${PLAN_JSON}"
  echo "plan.json emitted -> ${PLAN_JSON}"

  echo ""
  echo "DRY-RUN OK: 1 (csv, panel) tuple PLANNED, no GPU work executed."
  exit 0
fi

# USER_OK gate (Wf-306D — Wf-300C pattern: user explicit approval required
# for any wall-time launch; per kill_then_adapt standing directive, never
# auto-trigger wall-time without user OK).  Set USER_OK=1 to authorise the
# real launch; --dry-run path already exited above so it is unaffected.
if [[ -z "${USER_OK:-}" ]]; then
  echo "USER OK REQUIRED — kill_then_adapt standing directive."
  echo "Wf-264 CFM property panel re-run wall ~10 min on CPU (RDKit SA + QED)."
  echo "Re-run with:  USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh"
  echo ""
  echo "Wf-394-A2 NEW flags (env-var wire-in):"
  echo "  USER_OK=1 DIVERSITY_AWARE_EVAL=1 bash scripts/launch_wf264_cfm_property_panel.sh"
  echo "  USER_OK=1 SA_AWARE=1 bash scripts/launch_wf264_cfm_property_panel.sh"
  exit 1
fi
echo "[Wf-334-U3 / Wf-394-A2] USER_OK set — proceeding with Wf-264 panel re-run (CPU ~10 min)" | tee -a "${LOG}"

# Wave-tool guards (Wf-283C / Wf-262 §10.1) — pre-flight PASS required
# before any panel launch (TODO-207 v2 25/25 PASS gate).
${PY} molmetal/scripts/pre_flight_check.py --pipeline all || { echo "FAIL: pre_flight_check"; exit 1; }

# ---- REAL launch: Wf-264 panel run (Wf-394-A2 runner) ----
echo "=== Wf-264 / Wf-394-A2 CFM property panel START $(date) ===" | tee -a "${LOG}"
echo "SMILES CSV: ${SMILES_CSV}" | tee -a "${LOG}"
echo "OUT_JSON: ${OUT_BASE}/aggregate.json" | tee -a "${LOG}"
echo "DIVERSITY_AWARE_EVAL=${DIVERSITY_AWARE_EVAL} SA_AWARE=${SA_AWARE}" | tee -a "${LOG}"

${PY} ${RUNNER} "${RUNNER_ARGS[@]}" 2>&1 | tee -a "${LOG}"

echo "=== Wf-264 / Wf-394-A2 CFM property panel END $(date) ===" | tee -a "${LOG}"

# Per Wf-264 panel_report.md §5 — also emit per_molecule / per_metal /
# per_pocket JSON files. The current `cfm_property_panel.py` writes only
# the aggregate.json; the per_molecule / per_metal splits are derived in
# `panel_report.md` §5.  Future extension: have the panel script emit
# all three files directly (TODO for Wf-335+).
echo "=== Aggregation notice ===" | tee -a "${LOG}"
echo "panel_report.md equivalent: derive per_molecule / per_metal JSON from" | tee -a "${LOG}"
echo "${OUT_BASE}/aggregate.json (current panel script emits aggregate only)." | tee -a "${LOG}"
