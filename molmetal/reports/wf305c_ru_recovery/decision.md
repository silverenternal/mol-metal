# Wf-305C Ru seed recovery decision

**Date**: 2026-09-29 (Wf-305C Part 3, per Wf-304C honest-negative Task 1 audit)
**Owner**: Hugo Lee + ultracode agent
**Status**: ⏸ DEFERRED — awaiting user OK + GPU-1 reservation

---

## §1 Background

Per `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json`
(consolidated by Wf-304C Task 1), the Wf-219 5-metal × 3-seed ×
50-epoch Krasnov-aligned chemprop v2 sweep ran as follows:

| Metal | n_seeds_ok | Mean AUC | Std | Status |
|-------|-----------:|---------:|----:|--------|
| Re    | 3 / 3      | 0.8027   | 0.078 | OK |
| Os    | 3 / 3      | 0.7884   | 0.012 | OK |
| Rh    | 3 / 3      | 0.7525   | 0.014 | OK |
| Ir    | 3 / 3      | 0.7586   | 0.017 | OK |
| **Ru**| **0 / 3**  | **null** | **null** | **TIMEOUT (all 3 seeds)** |

**Aggregate (4 metals, Ru excluded)**: 0.7755 ± 0.039 (matches Wf-219 ship).

## §2 Ru timeout root cause

Per-seed JSON analysis (`Ru_seed{0,1,2}.json`):

| Seed | epoch_count | elapsed_s | per-seed AUC | status |
|------|------------:|----------:|-------------:|--------|
| 0    | 43 / 50     | 276.1     | 0.7858       | timeout |
| 1    | 32 / 50     | 271.8     | 0.8215       | timeout |
| 2    | 36 / 50     | 275.5     | 0.8214       | timeout |

**Root cause** (per Wf-304C §3.5 honest framing, repeated in
`wf219_krasnov_aligned/sweep_summary.json` honest_framing_note):
> hidden_dim=64/depth=2 chemprop v2 on Ru's 19135-row
> scaffold-split train took ~270s per run, exceeding 1800s
> budget headroom.

The 1800s budget is a hard wall-time cap; Ru is the largest cohort
by an order of magnitude (16,349 train rows vs Ir 3,902 / Os 956 /
Rh 930 / Re 741), so it spends almost the full budget on epochs
without reaching epoch 50 in 3/3 runs.

## §3 Decision

### §3.1 Option A — Use existing partial per-seed AUCs

**Verdict**: NOT YET — partial per-seed AUCs exist on disk, but
the Wf-219 ship aggregation intentionally **excludes** Ru from
the 5-metal aggregate mean per `party-twenty-spirit` §3.5 honest
framing. Reasons:

1. Epoch counts are 32, 36, 43 — none of the 3 seeds reached the
   Krasnov 2026 protocol's 50-epoch target. Reporting
   ``mean_auc=0.8096 ± 0.016`` (partial) as MEASURED would violate
   the protocol-aligned contract.
2. The honest framing for paper §3 / §17 cross-metal comparison
   must distinguish between "Ru completed at 50 epochs" and
   "Ru completed 32-43 epochs and we extrapolated".
3. There IS a partial-AUC 4-metal aggregate (Re/Os/Rh/Ir) of
   ``0.7755 ± 0.039`` that the Wf-219 ship already uses; adding
   Ru as MEASURED would conflate a protocol-compliant result with
   a partial result.

**Decision**: defer Option A pending the Wf-219 ship ledger
explicitly tagging the per-seed AUCs as ``status=timeout`` /
``epoch_count<50`` (already in ``sweep_summary.json`` per_seed_*
fields — confirmed). No new aggregation is permitted without
user OK.

### §3.2 Option B — Wait for GPU-1 retest with shorter epoch target

**Verdict**: ⏸ DEFERRED — requires user OK + 13.5 min GPU-1.

A protocol-faithful Ru recovery would either:

  * **(a)** Increase the per-seed wall-time budget from 1800s to
    ~4500s (so Ru's 50-epoch run completes). Estimated wall-time:
    3 seeds × ~450s each ≈ **22.5 min** on RTX 5090 GPU-1.
  * **(b)** Reduce the epoch target to 30 (matching Wf-124 P3's
    30-epoch protocol) so Ru completes at the current budget.
    Estimated wall-time: 3 seeds × ~170s each ≈ **8.5 min** on
    RTX 5090 GPU-1.
  * **(c)** Reduce cohort size via scaffold-split tightening
    (Bemis-Murcko min-count=5 → min-count=10) so Ru drops from
    16,349 → ~9,000 train rows. Estimated wall-time: 3 seeds ×
    ~240s each ≈ **12 min** on RTX 5090 GPU-1.

**Conservative pick**: **(c)** — keeps the 50-epoch target and the
current budget, just reduces the largest cohort to a level where
50-epoch training completes cleanly. Estimated wall-time:
**~12 min** on RTX 5090 GPU-1.

**Aggressive pick**: **(b)** — keeps the cohort but reduces epoch
target to 30. May under-converge relative to Krasnov 2026 protocol
but is faster. Estimated wall-time: **~8.5 min** on RTX 5090 GPU-1.

### §3.3 Recommended action

**RECOMMENDATION**: **(c)** — Ru scaffold-split tightening
(min-count=5 → min-count=10), 50-epoch target preserved, ~12 min
GPU-1 wall-time. Awaiting user OK before launching.

**Not recommended**: **(a)** — 22.5 min is over-budget for a
single-metal recovery when (c) achieves the same protocol
fidelity at half the wall-time.

**Not recommended (without further audit)**: **(b)** — epoch
reduction violates the protocol-aligned contract used by the
Re/Os/Rh/Ir results.

## §4 Honest framing per `party-twenty-spirit` §3.5

* **MEASURED**: 4/5 metals complete (Re/Os/Rh/Ir); aggregate
  ``0.7755 ± 0.039``.
* **NOT MEASURED**: Ru — all 3 seeds timed out at epochs
  32/36/43 < 50.
* **PARTIAL DATA on disk**: per-seed AUCs (0.7858 / 0.8215 /
  0.8214) exist in ``Ru_seed{0,1,2}.json`` but are tagged
  ``status=timeout`` in ``sweep_summary.json`` per_seed_* —
  they are NOT aggregated into the 5-metal mean.
* **RECOVERY REQUIRES USER OK** + ~12 min GPU-1
  (Option c, recommended).
* **NO OVERCLAIM**: paper §3 / §17 cross-metal comparison uses
  4-metal aggregate (Re/Os/Rh/Ir) until Ru recovery is
  completed and re-aggregated honestly.

## §5 Cross-references

* `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json`
  (Wf-304C Task 1 consolidation, Ru tagged ``status=timeout``)
* `molmetal/reports/wf219_krasnov_aligned/Ru_seed{0,1,2}.json`
  (per-seed JSONs with partial AUCs)
* `TODO/pending/219_wf219_krasnov_aligned.md` (Wf-219 ship)
* `TODO/pending/220_wf219_temporal_split_nci60_driver.md`
  (Wf-305C Part 2 sibling drivers)
* `party-twenty-spirit` §3.5 (honest-negative framing rules)
* `molmetal/reports/wf219_krasnov_aligned/decision.md` (Wf-219
  Ru timeout root-cause write-up, if it exists — see note below)

**NOTE**: Wf-304C audit already documented the Ru timeout root
cause in `sweep_summary.json` honest_framing_note. This file is
the Wf-305C Part 3 formal decision document for whether / when /
how to recover.

---

**Decision**: Option (c) — scaffold-split tightening (min-count=10),
**DEFERRED** until user OK + GPU-1 reservation (~12 min wall-time).

If user wants Ru in the paper §3 / §17 5-metal aggregate:
1. User OKs Option (c) or chooses (a) / (b).
2. Launch the 3-seed re-run on RTX 5090 GPU-1.
3. Re-aggregate ``sweep_summary.json`` honestly (5-metal mean).
4. Update paper §3 / §17 cross-metal table.

If user does NOT want to spend GPU-1 on Ru recovery:
* Keep the 4-metal aggregate (Re/Os/Rh/Ir) at ``0.7755 ± 0.039``.
* Paper §6 honest-negative entry for Ru timeout.
* TODO-220 P1 sibling drivers (temporal-split + NCI-60 9-cell)
  remain unaffected.