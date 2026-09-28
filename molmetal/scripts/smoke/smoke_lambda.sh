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

# Locate the shared venv (mirrors env.sh convention).
SHARED_VENV="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
if [[ -x "${SHARED_VENV}" ]]; then
    PY="${SHARED_VENV}"
else
    PY="${PYTHON:-python}"
fi

# Use a stable per-run temp dir so re-runs don't collide.
TMP_DIR="$(mktemp -d /tmp/molmetal_smoke_lambda.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "[smoke_lambda] Launching r4_lambda_only_run.py --smoke --quiet --out-dir ${TMP_DIR}" >&2
echo "[smoke_lambda] (10 pockets × 1 seed × 20 sims; <2 min on CPU)" >&2

T_START=$(date +%s)
set +e
"${PY}" -u molmetal/scripts/r4_lambda_only_run.py \
    --smoke \
    --quiet \
    --output-dir "smoke_lambda_$$" \
    --out-dir "${TMP_DIR}" \
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

REPORT_JSON="${TMP_DIR}/report.json"
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
