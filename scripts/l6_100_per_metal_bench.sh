#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/l6_100_per_metal_bench.sh — L6 closure bench at the R20-P5 spec volume
#
# Wf-R38 Phase P-C (2026-10-08) — sample-size reconciliation per
# TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md §4.1.
#
# The gap this closes. `LAMBDA_ROADMAP.md:49` states the L6 gate as
# "closure 30 metal × 100+ molecule reachability". The measured Wf-37 P2 run
# used 2244 samples = 30 d-block metals × ~75 samples each, i.e. BELOW spec
# on the volume axis (molmetal/reports/wf67_p4_2_closure_r20/synthesis.md:8,53).
# This wrapper re-runs the same bench at 100 samples/metal = 3000 samples so
# the reported number is protocol-clean.
#
# HONEST FRAMING — READ BEFORE QUOTING ANY OUTPUT (AGENTS.md §5.2):
# Raising 75 → 100 samples/metal is a VOLUME fix. It does NOT address the
# reachability ceiling. The measured value is 0.5891 against a gate of ≥ 0.95
# (gap −0.3609), which is already recorded as HONEST_NEGATIVE_STRUCTURAL at
# `LAMBDA_ROADMAP.md:274` (CPU SCAFFOLD SHIPPED, Wf-381-A5 §2.3, TODO-96
# closed CPU-side). A volume shortfall far smaller than the 0.36 gap cannot
# explain it. This script does NOT re-frame the gate as achievable; it only
# removes the sample-count confound so the ceiling is measured on-spec.
#
# CPU-only (no GPU, no USER_OK gate — see TODO/03_plans/…_RE_PLAN §Phase P-C
# row: "NO — CPU-only bench"). Default is --dry-run; pass --run to execute.
#
# Usage:
#   bash scripts/l6_100_per_metal_bench.sh              # dry-run, prints the plan
#   bash scripts/l6_100_per_metal_bench.sh --run        # actual 30 × 100 bench
#   N_SAMPLES=250 bash scripts/l6_100_per_metal_bench.sh --run   # sensitivity run
#
# Outputs (written by the bench itself):
#   molmetal/reports/wf_r38_l6_100_per_metal/
#
# AGENTS.md §5.9 — single-venv rule: the canonical project venv, never bare
# `python` / python3 / CadButEaas / kanzi_venv.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
BENCH="molmetal/scripts/bench_closure_dblock_r20.py"
OUT_DIR="molmetal/reports/wf_r38_l6_100_per_metal"
N_SAMPLES="${N_SAMPLES:-100}"
N_METALS="${N_METALS:-30}"

if [[ ! -x "$PY" ]]; then
  echo "FAIL: canonical venv python not found at $PY" >&2
  echo "      AGENTS.md §5.9 single-venv rule — do not substitute a bare python." >&2
  exit 1
fi
if [[ ! -f "$BENCH" ]]; then
  echo "FAIL: L6 bench not found at $BENCH" >&2
  exit 1
fi

CMD=("$PY" -u "$BENCH"
     "--n-metals" "$N_METALS"
     "--n-samples" "$N_SAMPLES"
     "--output-dir" "$OUT_DIR")

RUN=0
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --dry-run) RUN=0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

echo "L6 closure re-bench — Wf-R38 Phase P-C"
echo "  metals=$N_METALS  samples_per_metal=$N_SAMPLES  total=$((N_METALS * N_SAMPLES))"
echo "  out_dir=$OUT_DIR"
echo "  honest framing: volume fix only; the 0.5891 -> 0.95 reachability gap"
echo "                  stays HONEST_NEGATIVE_STRUCTURAL (LAMBDA_ROADMAP.md:274)."
echo "  cmd: ${CMD[*]}"

if [[ "$RUN" -ne 1 ]]; then
  echo "DRY-RUN (default). Pass --run to execute."
  exit 0
fi

mkdir -p "$OUT_DIR"
exec "${CMD[@]}"
