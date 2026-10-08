# LANE D — TODO-180/181/182 disposition (2026-10-08)

**Status**: 2 CLOSED (180, 181) + 1 PARTIAL remains GPU-blocked (182)
**Method**: disk verification per AGENTS.md §5.5 (filesystem is truth) — every
claim below traces to a file on disk, not to a prior narrative.

---

## §1 TODO-180 (G4 — remaining metals/seeds)

**Verdict: CLOSED.** The pending gap file is superseded.

Evidence:
- `molmetal/reports/wf278a_g4_aggregation_fix/fix_report.md` — G4 14 pairs
  MEASURED COMPLETE; aggregator path-mismatch fixed (0 → 14 pairs; 5-metal
  aggregate test_auc 0.7209 ± 0.0458).
- `molmetal/reports/wf124_p3_g4_data_quality/report.json` + `summary.md` —
  wf343c3 5-loader × 5-check data-quality gate PASS 5/5.

`TODO/pending/180_wf124_p3_remaining_metals_seeds.md` is the *original gap
analysis*; its work is done and its numbers are superseded by the G5 sweep
below. Kept as a historical artifact.

## §2 TODO-181 (G5 — 50-epoch full sweep)

**Verdict: CLOSED.** `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md`
documents the Wf-310A re-aggregation (15/15 pairs after retry; the original run
aggregated 0 pairs due to a path + schema mismatch).

The 5-metal REAL LOMO full sweep now also exists as the wf131 P3 run
(`TODO/completed/wf131_p3_5metal_lomo_25pairs_complete.md`): 25/25 pairs
trained 50/50 epochs on REAL MetalCytoToxDB (26,632 rows) — this is the
current G5-equivalent surface. `TODO/pending/181_*.md` is the original
gap analysis; kept as a historical artifact.

## §3 TODO-182 (G6 — per-innovation ablation) — REAL GAP REMAINS

**Verdict: PARTIAL / GPU-BLOCKED — NOT closed.**

Disk state is unambiguous:
- `molmetal/reports/wf124_p3_ablation/ablation_report.json` exists but covers
  **only 1 metal (Ru) × 1 seed (seed0) × 5 epochs × 4 configs**
  (baseline / wasserstein / kendall / dann). It is a smoke-scale ablation.
- The full matrix the pending file specifies (5 metals × 3 seeds × 4 configs ×
  50 epochs = 60 runs) has **not** been run. No report on disk covers it.

This is an honest-negative, not a bookkeeping item: the paper §3 attribution
claim ("which of the 3 innovations carries the lift") is only supported at
5-epoch/Ru-smoke scale today. Full-matrix ablation is >1 hr GPU per AGENTS.md
§5.1 kill_then_adapt threshold → requires an explicit fresh user OK, and it is
**not** required for the Tier-2 computational-only submission scope (§5.4).

Decision: keep `TODO/pending/182_wf124_p3_per_innovation_ablation.md`, but its
status is corrected to `PARTIAL_5EPOCH_RU_SMOKE_ONLY — full 60-run matrix
GPU-BLOCKED (needs USER_OK, not Tier-2-gating)`.

## §4 Follow-on defect fixed by this lane

The wf131 G5-equivalent wrapper `scripts/launch_wf131_p3_pt_4cells.sh`
aggregated only 24/25 pairs: its aggregation loop globbed
`cell_dir/<seed_dir>/per_seed_metrics_seed*.json`, but Ru/seed0 came from a
legacy trainer path that wrote only `cell_dir/metrics.json` (no nested detail
file), so the glob silently dropped it. Real bug, real fix:

- Aggregation now prefers `cell_dir/<Metal>/per_seed_metrics_seed*.json` and
  falls back to `cell_dir/metrics.json` / `aggregate.json`.
- Added a fail-loud gate: aggregated count ≠ planned count → exit 1 (per §5.7
  degenerate-run class — a silent under-count is a fabricated-green surface).
- Regenerated `molmetal/reports/wf131_p3_pt_4cells/training_summary.json`:
  `wf131_p3_pt_4cells_total_pairs` now reads **25** (was 24), planned 25.
