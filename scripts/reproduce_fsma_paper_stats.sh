#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce metrics/by_metric/fsma_paper_stats.json
#
# The consolidated FSMA statistics used by paper_a/ (head-only K-curve,
# C2 matched controls, C3 sampler sweep, C4 scope sweep, recovery-to-ceiling,
# per-family Holm-Bonferroni).  CPU-only and deterministic given the report
# inputs; the reports themselves are GPU-produced by
# molmetal/scripts/train_chemprop_lomo_paper_cohort.py --config <cfg>.
#
# AGENTS.md §5.9 single-venv: use the canonical interpreter, no bare python.
# AGENTS.md §5.5: never overwrite the on-disk artifact — write to a scratch
# path and diff (the caller, scripts/reproduce_paper_a.py, does the diff).
#
# Usage:
#   scripts/reproduce_fsma_paper_stats.sh                 # -> /tmp/fsma_paper_stats.repro.json
#   scripts/reproduce_fsma_paper_stats.sh OUT.json        # -> OUT.json
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="${1:-/tmp/fsma_paper_stats.repro.json}"

cd "$REPO"

exec "$PY" molmetal/scripts/fsma_synthesize.py \
  --run head_k5=molmetal/reports/fsma_c4_head_k5 \
  --run head_k10=molmetal/reports/fsma_c4_head_k10 \
  --run head_k20=molmetal/reports/fsma_c4_head_k20 \
  --run head_k40=molmetal/reports/fsma_c4_head_k40 \
  --run head_k80=molmetal/reports/fsma_c4_head_k80 \
  --run full_k5=molmetal/reports/l3_kshot_pf_k5 \
  --run full_k10=molmetal/reports/l3_kshot_pf_k10 \
  --run full_k20=molmetal/reports/l3_kshot_pf_k20 \
  --run full_k40=molmetal/reports/l3_kshot_pf_k40 \
  --run full_k80=molmetal/reports/l3_kshot_pf_k80 \
  --run c2m_k20=molmetal/reports/fsma_c2_matched_k20 \
  --run c2m_k40=molmetal/reports/fsma_c2_matched_k40 \
  --run c3_random=molmetal/reports/fsma_c3_random_k20 \
  --run c3_strat=molmetal/reports/fsma_c3_stratified_k20 \
  --run c3_scaffold=molmetal/reports/fsma_c3_scaffold_diverse_k20 \
  --run c3_activity=molmetal/reports/fsma_c3_activity_scaffold_k20 \
  --run c4_head_metal_k20=molmetal/reports/fsma_c4_head_metal_k20 \
  --family C1_kshot_curve_headonly=head_k5,head_k10,head_k20,head_k40,head_k80 \
  --family C1b_kshot_curve_fullscope=full_k5,full_k10,full_k20,full_k40,full_k80 \
  --family C2_controls_matched=c2m_k20,c2m_k40 \
  --family C3_support_sampler=c3_random,c3_strat,c3_scaffold,c3_activity \
  --family C4_adapt_scope=full_k5,full_k10,full_k20,full_k40,full_k80,head_k5,head_k10,head_k20,head_k40,head_k80,c4_head_metal_k20 \
  --xgb-json metrics/by_metric/fsma_xgb_warmstart_kshot.json \
  --ceiling-run molmetal/reports/l3_within_pf_ceiling \
  --out-metric "$OUT"
