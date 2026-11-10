#!/usr/bin/env bash
# =============================================================================
# RETIRED 2026-11-04 — Wf-268A-Smoke runner: 4 ablation configs x 5 epochs each.
#
# WHY THIS FILE IS RETIRED RATHER THAN REPAIRED
# ---------------------------------------------
# 1. THE RUN RECORD IS PRESERVED, AND MUST NOT BE OVERWRITTEN.  This script
#    produced the artifacts that sit beside it:
#      molmetal/reports/wf268a_smoke/config_full/{run.log,
#        per_epoch_metrics.json, per_epoch_metrics.csv, wasserstein_decay.json,
#        synthesis.md}
#      molmetal/reports/wf268a_smoke/config_no_yuelbond/{...}
#      molmetal/reports/wf268a_smoke/config_no_wasserstein/{...}
#      molmetal/reports/wf268a_smoke/config_no_closure_bounded/{...}
#      molmetal/reports/wf268a_smoke/aggregate.json + aggregate.py
#      molmetal/reports/wf268a_smoke/smoke_runner.log
#    Those files are the evidence; this script is how they were made.
# 2. IT CANNOT RUN ON THIS HOST.  The interpreter it names,
#      /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
#    does not exist — MEASURED 2026-11-04: `ls -d /home/hugo/codes/CadButEaas`
#    => "No such file or directory" (the whole CadButEaas tree is gone), so
#    the script exits 2 at the venv check at the bottom, before any training.
# 3. REPAIRING IT WOULD CONTAMINATE THE RECORD.  Pointing it at the canonical
#    .venv would make it runnable — but it would then write NEW numbers
#    (a different torch/rdkit) over the preserved per_epoch_metrics.json of a
#    2026-09 run, under the same filenames.  The recorded run's environment no
#    longer exists, so its numbers are not reproducible; a new run belongs in a
#    NEW output directory, never on top of a preserved metric record.
# 4. IT IS NOT A LAUNCH SURFACE.  It lives under molmetal/reports/ (a
#    run-record tree), not scripts/launch_*.sh, which is what AGENTS.md §5.11
#    governs.  scripts/_yaml_only_gate.py keeps it in its EXCLUDED list, and
#    the excluded-file residual census continues to disclose its 10 trainer
#    flags on every run: retiring it did NOT make those flags disappear from
#    the gate's disclosure (they are kept verbatim below, live, as the
#    provenance of the preserved artifacts — the run.log files do NOT record
#    the command line).
#
# WHAT CHANGED ON 2026-11-04: this banner and the explicit early exit below.
# The historical invocation is KEPT VERBATIM and is now unreachable.
# =============================================================================
echo "[wf268a-smoke] RETIRED 2026-11-04: this is a preserved RUN RECORD." >&2
echo "[wf268a-smoke] The artifacts it produced are kept next to it and must not be overwritten." >&2
echo "[wf268a-smoke] It cannot run: the interpreter it names" >&2
echo "[wf268a-smoke] (/home/hugo/codes/CadButEaas/.venv-vlm/bin/python) no longer exists on this host." >&2
echo "[wf268a-smoke] See the retirement banner at the top of this file for the full rationale." >&2
exit 2

# ---------------------------------------------------------------------------
# HISTORICAL BODY — kept verbatim as the provenance of the preserved artifacts
# above.  Unreachable since 2026-11-04 (the retirement exits 2 above).
# ---------------------------------------------------------------------------
# Honors the task spec:
#   - hidden_dim=32, n_layers=2, 100 samples (CPU-only)
#   - 5 epochs each
#   - Flags: full / --no-yuelbond / --no-wasserstein / --no-closure-bounded
#   - Decode_ratio must be measured + non-NaN for each config.
#
# CONSTRAINT: USE /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
# (the task spec path "/home/hugo/cades/CadButEaas/.venv-vlm/bin/python"
# does not exist; the actual venv lived at /home/hugo/codes/CadButEaas/ —
# that tree was REMOVED from this host after the 2026-09 run; MEASURED
# 2026-11-04: `ls -d /home/hugo/codes/CadButEaas` => No such file or directory.)
set -uo pipefail

PYTHON="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
SCRIPT="/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
REPORT_ROOT="/home/hugo/codes/mol-metal/molmetal/reports/wf268a_smoke"

mkdir -p "${REPORT_ROOT}"

cd /home/hugo/codes/mol-metal

# Sanity: venv path
if [[ ! -x "${PYTHON}" ]]; then
    echo "FATAL: python venv not found at ${PYTHON}" >&2
    exit 2
fi

# Common flags (CPU + 100 samples + 5 epochs + small model)
COMMON=(
    --epochs 5
    --n-train 100
    --n-val 100
    --hidden-dim 32
    --n-layers 2
    --batch-size 32
    --device cpu
    --cuda-visible-devices ""
    --patience 10
    --seed 0
    --out-dir
)

# 4 configs.  Full has all default flags (no --no-*).
# Each run writes per_epoch_metrics.json under its own out-dir.
run_config () {
    local label="$1"; shift
    local out="${REPORT_ROOT}/${label}"
    mkdir -p "${out}"
    echo "================================================================"
    echo "[wf268a-smoke] ${label} : out=${out}"
    echo "================================================================"
    "${PYTHON}" -u "${SCRIPT}" \
        "${COMMON[@]}" "${out}" \
        "$@" \
        2>&1 | tee "${out}/run.log"
    local rc="${PIPESTATUS[0]}"
    echo "[wf268a-smoke] ${label} exit_code=${rc}"
    return ${rc}
}

# Full — default flags (yuelbond on, wasserstein on, closure on)
run_config "config_full" || echo "[wf268a-smoke] config_full FAILED rc=$?"

# --no-yuelbond
run_config "config_no_yuelbond" --no-yuelbond || echo "[wf268a-smoke] config_no_yuelbond FAILED rc=$?"

# --no-wasserstein
run_config "config_no_wasserstein" --no-wasserstein || echo "[wf268a-smoke] config_no_wasserstein FAILED rc=$?"

# --no-closure-bounded
run_config "config_no_closure_bounded" --no-closure-bounded || echo "[wf268a-smoke] config_no_closure_bounded FAILED rc=$?"

echo "================================================================"
echo "[wf268a-smoke] All 4 runs complete."
echo "================================================================"
