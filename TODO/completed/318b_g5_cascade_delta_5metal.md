# Wf-318B — TODO-181 G5 cascade delta extended to all 5 metals

**Date**: 2026-09-29 (Wf-318B ship)
**Status**: ✅ SHIPPED — CPU-only aggregation, no GPU launch required
**Parent**: `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` §4 (cascade comparison)
**Sibling**: `TODO/completed/310a_wf310a_g5_aggregation.md` (Wf-310A re-aggregation)
**Trigger**: Wf-310A Follow-up #3 + Wf-314B analysis gap — cascade delta table only had Ru populated

---

## §1 What this workflow delivers

Wf-310A's closeout (TODO-181) included a cascade delta 30ep→50ep for **only Ru**
because that was the only metal where G4 and G5 had overlapping seeds at the
time of closeout. Wf-314B's gap analysis identified the 4 remaining metals
(Ir/Os/Rh/Re) as needing the same delta computation.

Wf-318B closes that gap by:

1. **Reading** the G4 30-ep 5-metal × 3-seed baseline from
   `results/wf124_p3_remaining_metals_g4/training_summary.json` (MEASURED).
2. **Reading** the G5 50-ep per-metal aggregates from
   `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` (13/15 MEASURED) +
   the **Ir/seed=0 retry** at
   `results/wf124_p3_50epoch_full_sweep_g5/Ir/seed0/training_summary.json`
   (MEASURED test_auc=0.6888).
3. **Computing** per-metal cascade delta (G5_mean − G4_mean) with delta_pct,
   std_pooled (unpaired two-sample formula), and t_stat.
4. **Writing** the new metric JSON
   `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` (6,309 bytes).
5. **Adding** 5 new CLAIM_CATALOG entries to `molmetal/scripts/paper_dashboard.py`
   (3 MEASURED + 2 HONEST_NEGATIVE).
6. **Updating** `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md`
   with §2.3 — the full 5-metal cascade table.

---

## §2 MEASURED cascade delta (all 5 metals)

Source: `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json::per_metal_table`

| Metal | G4 30-ep test_auc | G5 50-ep test_auc | Δ (lift) | t_stat | n_overlap | Status |
|-------|-------------------|-------------------|----------|--------|-----------|--------|
| **Ru** | 0.7561 ± 0.0112 | 0.7860 ± 0.0205 | **+0.0300 (+3.96%)** | 2.10 | 2/3 (Ru/seed=2 in-flight) | MEASURED lift |
| **Ir** | 0.6702 ± 0.0056 | 0.6969 ± 0.0150 | **+0.0266 (+3.98%)** | 2.88 | 3/3 (full, retry incorporated) | MEASURED lift |
| **Os** | 0.7025 ± 0.0205 | 0.7276 ± 0.0159 | **+0.0250 (+3.56%)** | 1.67 | 3/3 (full) | MEASURED lift |
| **Rh** | 0.7227 ± 0.0540 | 0.6798 ± 0.0165 | **−0.0428 (−5.92%)** | −1.29 | 3/3 (full) | HONEST_NEGATIVE regression |
| **Re** | 0.7700 ± 0.0244 | 0.7329 ± 0.0433 | **−0.0371 (−4.82%)** | −1.10 | 3/3 (full) | HONEST_NEGATIVE regression |

5-metal aggregate delta: **−0.0011** (essentially flat; large metals Ru/Ir/Os
lift +0.025 to +0.030 each, small metals Rh/Re regress −0.04 each, partially
cancel out at aggregate).

---

## §3 Honest framing per party-twenty-spirit §3.5

- **3 metals LIFT (Ru/Ir/Os)** at the +20 epoch doubling:
  - Ru: +0.0300 (n=2-seed overlap; Ru/seed=2 still in-flight)
  - Ir: +0.0266 (n=3-seed overlap; Wf-310A retry at 0.6888 incorporated)
  - Os: +0.0250 (n=3-seed overlap; full coverage)
- **2 metals REGRESS (Rh/Re)** at the +20 epoch doubling:
  - Rh: −0.0428 (n=3-seed overlap; full coverage — honest-negative)
  - Re: −0.0371 (n=3-seed overlap; full coverage — honest-negative)
- **No fabrication**: every value traces to JSON on disk. Unpaired two-sample
  t-test formula `delta / std_pooled × sqrt(n_g4 × n_g5 / (n_g4 + n_g5))` —
  std_pooled combines G4 std + G5 std (NOT paired t-test because G4 and G5 use
  different random seeds in the per-seed splits).
- **Hypothesis for the Rh/Re regression** (for the Wf-182 G6 ablation):
  Both are small-cohort (Rh ~696 rows train, Re ~692 rows train). The W₂
  regulariser (alpha=0.5 + w_w=0.05) may over-smooth on small cohort at 50
  epochs (overfitting). TODO-182 G6 per-innovation ablation will attribute
  this to either W₂ contribution or L2 guard over-tightening.

---

## §4 Files written / updated

| Path | Type | Size |
|---|---|---|
| `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` | NEW metric JSON | 6,309 bytes |
| `molmetal/scripts/compute_cascade_delta_5metal.py` | NEW script (CPU-only) | ~5 KB |
| `molmetal/scripts/paper_dashboard.py` | +5 CLAIM_CATALOG entries | 76 total entries (was 71) |
| `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` | +§2.3 cascade table | (sub-section added) |
| `TODO/completed/318b_g5_cascade_delta_5metal.md` | NEW (this file) | — |

---

## §5 Pre-flight verification

`pre_flight_check.py --pipeline frontier --json /tmp/wf318b_preflight.json`:
- **PASS=20 FAIL=0 SKIP=0 TOTAL=20 wall=21.7s** — ALL 20 FRONTIER GATES PASS
- Includes: p3_frontier_math_wire_in (4/4 importable),
  p3_frontier_math_day_3_5_integration_tests (8 passed, 1 warning),
  p3_frontier_math_paper_refs_present (all 5 files), p3_smoke_fail_on_missing_audit
  (5 passed), p3_loi_send_ready (4/4 LOI), p3_nci60_9cell_per_cell_aggregate
  (9-cell + HCT116 + PC3 honest-negative), p3_qed_score_metric_present
  (HONEST_NEGATIVE=true; mean=0.2297 below Wf-139 P6 0.5 by −0.270).

The full `--pipeline all` pipeline hit a 240s timeout (the Wf-319B/C P3
sub-checks include slow gates — out of scope for Wf-318B's CPU-only cascade
delta computation). Frontier is the canonical subset per Wf-306C + Wf-259.

---

## §6 Follow-ups

1. **Wf-318C** (next): re-run `compute_cascade_delta_5metal.py` after
   Ru/seed=2 retry lands (epoch 27/50 in-flight as of 2026-09-29). Ru row
   should update from n=2 to n=3 overlap.
2. **TODO-182 G6 per-innovation ablation**: 5×3×50=75 pairs to attribute the
   Rh/Re regression to W₂ vs L2-guard vs baseline (Wf-310F follow-up).
3. **Wf-309A early-stopping** may be preferable for small-cohort metals
   (Rh/Os/Re < 1.2k train rows) — review once cascade delta reproduces with
   shorter horizon.
4. **TODO-219 §3 temporal-split driver** (Wf-310D): direct comparison vs
   Krasnov 2026's temporal split for the lifted metals (Ru/Ir/Os).

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: All 5 cascade deltas
are MEASURED from G4 + G5 MEASURED training outputs (no projection, no
fabrication). 2 honest-negative regressions (Rh, Re) are reported with the
HONEST_NEGATIVE badge in `paper_dashboard.py` CLAIM_CATALOG rather than
papered over. The 5-metal aggregate delta (−0.0011) is reported unfiltered.
Ru/seed=2 retry in-flight means the Ru row is partial n=2 (will refresh
post-Wf-318C).
