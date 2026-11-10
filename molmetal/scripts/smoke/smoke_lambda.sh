#!/usr/bin/env bash
# WF-IMPROVEMENT-BACKLOG I2 — Lambda MCTS subsystem smoke (10-pocket × 1-seed pilot).
#
# Goal
# ====
# Verify the Lambda-only MCTSProofSearch can run a 10-pocket × 1-seed
# sweep (10 cells, ~10 minutes budget) and produce at least one distinct
# candidate per pocket.  This is the cheapest end-to-end Lambda smoke that
# exercises the click-rule library, the metal seed default (cisplatin),
# the persistent search tree, and the report writer.
#
# Why --smoke preset?
# -------------------
# ``r4_lambda_only_run.py --smoke`` resolves to the
# :data:`_R17_SMOKE_PRESET` constants (``pockets=10, seeds=[42],
# n_simulations=20``, manifest ``test_010_019_subset_manifest.csv``).  This
# is the same preset used by ``tests/test_smoke_flag.py`` — a known-good
# configuration that completes in <2 min on CPU.
#
# What it does
# ------------
# 1. Run ``r4_lambda_only_run.py --smoke --quiet --out-dir <tmp>``.
# 2. Parse ``<tmp>/report.json`` and tally ``n_distinct`` per pocket.
# 3. PASS if every pocket reports ``n_distinct >= 1`` AND the total
#    distinct count across pockets is >= 1 (one of the canonical
#    success criteria from TODO-32 §L-R17-B2 verify gate).
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/smoke/smoke_lambda.sh
#
# Exit code
# ---------
# 0 — 10/10 cells reported n_distinct >= 1.
# 1 — any cell reported n_distinct == 0, or the runner crashed.
#
# Wall-time budget
# ----------------
# 10 cells × 20 sims × n_top_k=20 ≈ 60-90 s on CPU (per existing
# IMPROVEMENT_BACKLOG measurements).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# CPU LOCK (AGENTS.md §5.1 + the repo's standing 0-GPU session contract)
#
# MEASURED 2026-11-04: running this script WITHOUT the lock allocated
# 694 MiB of GPU memory per process, while the smoke itself ran on CPU.
# The cause is the IMPORT CHAIN, not the smoke: it reaches
# ``torch.cuda.get_device_capability()``
# (molmetal/adapters/flow_matching_lipman/amp.py:187), and
# ``torch.cuda.is_available()`` initialises the CUDA driver (``cuInit``),
# which creates the context.  This is the SAME class Wf-A162 repaired
# inside the pre-flight (molmetal/scripts/pre_flight_check.py::_smoke_device)
# -- but these smoke scripts carried no lock of their own, so a caller who
# forgot it silently paid the allocation (this repo's own session did,
# twice, before the lock existed).
#
# These smokes are CPU-BY-DESIGN (each script's own header budgets the run
# as "<2 min on CPU"), so the lock is the DEFAULT rather than a caller
# obligation: an unset ``CUDA_VISIBLE_DEVICES`` becomes "" (no device
# visible), and an explicit value is PRESERVED so a deliberate GPU smoke
# is still possible.  ``MOLMETAL_SMOKE_DEVICE=cpu`` is the repo's parity
# kill-switch (honoured by ``metallo_drug_smoke_retrain.py::_resolve_device``).
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export MOLMETAL_SMOKE_DEVICE="${MOLMETAL_SMOKE_DEVICE:-cpu}"

# Locate the canonical repo-local venv (AGENTS.md §5.9 single-venv rule).
SHARED_VENV="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${SHARED_VENV}" ]]; then
    PY="${SHARED_VENV}"
elif [[ -n "${PYTHON-}" && -x "${PYTHON}" ]]; then
    # An EXPLICIT override is honoured; a BARE `python` is not (AGENTS.md
    # §5.9: the smoke must run under the repo's single venv, never under an
    # ambient interpreter that may be another project's -- `kanzi_venv`,
    # `CadButEaas`, or a system python with a different torch).
    PY="${PYTHON}"
else
    echo "[smoke] FATAL: no interpreter." >&2
    echo "[smoke]   expected the repo venv: ${SHARED_VENV}" >&2
    echo "[smoke]   (missing or not executable) -- AGENTS.md §5.9 forbids a" >&2
    # Wf-A269 wave-26 (2026-11-04) — THE MESSAGE MENTIONS AN INTERPRETER
    # PATH, AND THE REPO'S OWN §5.9 GATE READS MENTIONS AS USE.
    # `scripts/_wrapper_audit.py::interpreter_census` adjudicates a LITERAL
    # absolute interpreter path wherever it appears on a CODE line (comments
    # are stripped, but a path inside an `echo` ARGUMENT is not a comment).
    # The message below therefore tripped the gate:
    #   VIOLATION molmetal/scripts/smoke/<smoke>.sh:<line> [/abs/path/to/python]
    #   -- §5.9 forbids an interpreter outside the repo venv
    # MEASURED 2026-11-04: 4 files / 4 violations, all from THIS message.
    # The gate is CORRECT as documented ("the rule is LITERAL by construction:
    # an interpreter path is adjudicated when it is written out in full") — the
    # defect was the MESSAGE, not the gate.  => the message now names the
    # override VARIABLE without writing out a literal absolute path, and the
    # gate returns to 0 violations.  The gate's own imprecision (it cannot
    # tell an interpreter INVOCATION from an interpreter MENTION) is
    # disclosed separately, not silently reworded away.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   echo "[smoke]   bare \`python\`.  Set PYTHON=/abs/path/to/python to" >&2
    echo "[smoke]   bare \`python\`.  Set PYTHON=<absolute python path> to" >&2
    echo "[smoke]   override explicitly, or create the repo venv." >&2
    exit 1
fi

# Use a stable per-run temp dir so re-runs don't collide.
TMP_DIR="$(mktemp -d /tmp/molmetal_smoke_lambda.XXXXXX)"
# The runner writes its report under molmetal/reports/wf_lambda1_<output-dir>/,
# NOT under the temp dir (MEASURED 2026-11-04: the runner's own code does
#   out = _REPO_ROOT / "molmetal" / "reports" / f"wf_lambda1_{args.output_dir}"
# and then writes report.json + summary.md + seeds.json there).  The
# pre-repair smoke looked for ${TMP_DIR}/report.json, which the runner never
# writes, so the smoke could not have passed even with a valid command line.
OUT_NAME="smoke_lambda_$$"
OUT_DIR="${REPO_ROOT}/molmetal/reports/wf_lambda1_${OUT_NAME}"
trap 'rm -rf "${TMP_DIR}" "${OUT_DIR}"' EXIT

echo "[smoke_lambda] Launching r4_lambda_only_run.py --config configs/lambda_smoke.yaml --output-dir ${OUT_NAME}" >&2
echo "[smoke_lambda] (10 pockets × 1 seed × 20 sims; <2 min on CPU)" >&2

T_START=$(date +%s)
set +e
"${PY}" -u molmetal/scripts/r4_lambda_only_run.py \
    --config configs/lambda_smoke.yaml \
    --output-dir "${OUT_NAME}" \
    > "${TMP_DIR}/stdout.log" 2> "${TMP_DIR}/stderr.log"
RC=$?
set -e
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

if [[ ${RC} -ne 0 ]]; then
    echo "[smoke_lambda] FAIL runner_crashed rc=${RC} after ${ELAPSED}s" >&2
    echo "[smoke_lambda] stderr (last 20 lines):" >&2
    tail -20 "${TMP_DIR}/stderr.log" >&2 || true
    exit 1
fi

REPORT_JSON="${OUT_DIR}/report.json"
if [[ ! -f "${REPORT_JSON}" ]]; then
    echo "[smoke_lambda] FAIL no_report: ${REPORT_JSON} not produced" >&2
    tail -20 "${TMP_DIR}/stderr.log" >&2 || true
    exit 1
fi

# Parse the report JSON with a small Python one-liner (tolerant of jq absence).
read -r N_CELLS N_DISTINCT_TOTAL N_DISTINCT_MIN <<<"$(
    "${PY}" - "${REPORT_JSON}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as fh:
    d = json.load(fh)
cells = d.get("cells", [])
n_cells = len(cells)
n_distinct_total = 0
n_distinct_min = float("inf")
for c in cells:
    n = int(c.get("n_distinct", 0) or 0)
    n_distinct_total += n
    n_distinct_min = min(n_distinct_min, n)
if n_cells == 0:
    n_distinct_min = 0
print(f"{n_cells} {n_distinct_total} {int(n_distinct_min)}")
PYEOF
)"

echo "[smoke_lambda] cells=${N_CELLS} total_distinct=${N_DISTINCT_TOTAL} min_distinct=${N_DISTINCT_MIN} wall=${ELAPSED}s" >&2

# PASS criteria (per IMPROVEMENT_BACKLOG §I2):
#   1. n_cells == 10 (the --smoke preset)
#   2. n_distinct_total >= 1 (at least one distinct candidate produced)
#   3. n_distinct_min >= 1 (every pocket produced >= 1 candidate)
if [[ "${N_CELLS}" -eq 10 && "${N_DISTINCT_TOTAL}" -ge 1 && "${N_DISTINCT_MIN}" -ge 1 ]]; then
    echo "LAMBDA_SMOKE PASS n_cells=${N_CELLS} n_distinct=${N_DISTINCT_TOTAL} wall=${ELAPSED}s"
    exit 0
fi
echo "LAMBDA_SMOKE FAIL n_cells=${N_CELLS} n_distinct=${N_DISTINCT_TOTAL} (min=${N_DISTINCT_MIN})" >&2
exit 1
