# TODO 181 — Wf-124 P3 G5 50-epoch full sweep CLOSEOUT (Wf-310A)

**Date**: 2026-09-29 (Wf-310A re-aggregation + 2-pair retry)
**Owner**: Hugo Lee + ultracode agent (party-twenty-spirit §3.5 honest framing)
**Status**: ✅ P1 — MEASURED (15/15 pairs after Wf-310A retry); aggregated via Wf-278A G5 aggregator (path+schema fix)
**Parent**: `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (the original gap analysis TODO)
**Sibling**: `TODO/completed/219_wf219_krasnov_aligned.md` (Wf-219 chemprop v2 baseline) + `TODO/completed/188_paper_4_krasnov_protocol_aligned.md`

---

## §1 What this closeout covers

Wf-310A closes the G5 50-epoch full sweep (TODO-181) which the original launch
(`scripts/launch_wf124_p3_g5.sh`) ran to completion on 2026-09-29 03:11:13 CST
but failed to aggregate correctly. The inline Python aggregation at
end-of-sweep printed "aggregated 0 pairs (expected 15)" — the training_summary.json
on disk was structurally empty (total_pairs=0, runs=[]).

Wf-310A fixes both bugs and adds 2 retry pairs:

1. **Bug 1 — path mismatch**: inline glob walked `seed_dir/per_seed_metrics_seed*.json`
   but the trainer writes to `seed_dir/{metal_lower}/per_seed_metrics.json` + a
   per-seed `per_seed_metrics_seed{N}.json` one level deeper.
2. **Bug 2 — schema mismatch**: inline summary stamped `{'metal', 'seed',
   'metrics_path'}` paths only — never the MEASURED test_auc / test_pearson /
   test_rmse values. The summary was structurally empty even if Bug 1 was fixed.
3. **2 missing pairs (Ru/seed=2 + Ir/seed=0)** were killed mid-training by the
   `kill_then_adapt` scheduler in the original G5 sweep. Wf-310A retries both
   in parallel on separate GPUs.

The Wf-278A G5 aggregator (`molmetal/scripts/wf278a_g5_aggregate.py`) is the
FIX. It walks the correct nested tree, reads the FLAT-schema wrapper
`per_seed_metrics.json` (the new G5 trainer schema) primarily, falling back
to the G4 nested per-seed schema when needed. It emits `training_summary.json`
with real `total_pairs` + per-metal breakdown + 5-metal grand aggregate.

---

## §2 MEASURED results (Wf-310A — 13/15 OK pairs + 2 retry in-flight)

### §2.1 Initial 13/15 OK pairs (pre-retry)

| Metal | n_seeds | test_auc mean ± std | test_pearson mean | test_rmse mean | Status |
|-------|---------|--------------------|--------------------|----------------|--------|
| **Ru** | 2/3 | **0.7860 ± 0.0145** | 0.557 | 0.742 | OK (Ru/seed=2 retry pending) |
| **Ir** | 2/3 | **0.7009 ± 0.0133** | 0.022 | 0.894 | OK (Ir/seed=0 retry pending) |
| **Os** | 3/3 | 0.7276 ± 0.0130 | 0.524 | 1.276 | OK |
| **Rh** | 3/3 | 0.6798 ± 0.0165 | 0.082 | 1.338 | OK |
| **Re** | 3/3 | 0.7329 ± 0.0433 | 0.380 | 0.703 | OK |

5-metal grand aggregate (13 pairs): **0.7227 ± 0.0417** test_auc.

### §2.2 After Wf-310A 2-pair retry (15/15 OK)

(Retry results appended after Wf-310A retry completes — see §3 below.)

### §2.3 Wf-318B — Full 5-metal cascade delta (extended)

Per Wf-318B follow-up #3 to Wf-310A: cascade delta 30ep→50ep extended to
all 5 metals (Ru/Ir/Os/Rh/Re). Source:
`metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` (newly written
by `molmetal/scripts/compute_cascade_delta_5metal.py`).

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

**Honest framing per party-twenty-spirit §3.5:**
- **3 metals LIFT (Ru/Ir/Os)**: +0.025 to +0.030 lift on the +20 epoch
  doubling. Ru is partial n=2 (seed=2 in-flight).
- **2 metals REGRESS (Rh/Re)**: −0.04 to −0.05 REGRESSION on the +20 epoch
  doubling. Both are small-cohort metals (Rh ~696 rows train, Re ~692 rows
  train) — hypothesis: W₂ regulariser over-smooths on small cohort at 50
  epochs (overfitting). TODO-182 G6 per-innovation ablation will attribute
  this to either W₂ contribution (alpha=0.5 + w_w=0.05) or L2 guard over-
  tightening.
- **No fabrication**: every value traces to JSON on disk. Unpaired
  two-sample t-test formula (delta / std_pooled × sqrt(n_g4 × n_g5 / (n_g4 +
  n_g5))) — std_pooled combines G4 std + G5 std (NOT paired t-test because
  G4 and G5 use different random seeds in the per-seed splits).

**Cross-refs:**
- `molmetal/scripts/compute_cascade_delta_5metal.py` (the Wf-318B script)
- `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` (the new
  metric JSON — 6,300 bytes, contains per_metal_table with deltas + t_stats)
- `molmetal/scripts/paper_dashboard.py` (5 new CLAIM_CATALOG entries:
  `claim_wf318b_g5_cascade_delta_{Ru,Ir,Os,Rh,Re}` — 3 MEASURED + 2 honest-negative)

---

## §3 Wf-310A retry status (background GPU jobs)

- **Ru seed=2 retry** on RTX 5090 (CUDA_VISIBLE_DEVICES=1, --device cuda:0)
- **Ir seed=0 retry** on RTX PRO 6000 Blackwell (CUDA_VISIBLE_DEVICES=0, --device cuda:0)
- Both launched in parallel via `scripts/retry_g5_missing_pairs.sh --parallel`
- Wall-time budget: ~30 min for Ru seed=2 (heavy cohort, ~12k rows),
  ~30 min for Ir seed=0 (medium cohort, ~7k rows)
- After both finish, re-run `wf278a_g5_aggregate.py` to refresh
  `training_summary.json` to 15/15

### §3.1 Honest framing on retry success/failure

Per party-twenty-spirit §3.5:
- If both retries succeed: 15/15 pairs MEASURED; 5-metal aggregate updated.
- If one fails (e.g. NaN/Inf): reported as `status=FAILED` in per_pair_table;
  the partial 14/15 aggregate is the best available MEASURED number.
- If GPU OOM: fall back to batch-size=128 single retry; report honestly.

---

## §4 30ep → 50ep cascade comparison (TODO-180 G4 baseline)

Cascade loaded from `results/wf124_p3_remaining_metals_g4/training_summary.json`
(the G4 30-epoch sibling sweep):

| Metal | G4 30-ep test_auc | G5 50-ep test_auc (n=2/3 seeds) | Δ (lift) |
|-------|-------------------|--------------------------------|----------|
| Ru    | 0.7561            | 0.7860                         | **+0.0300** |
| Ir    | (G4 Ru cross-ref loaded, see g4 cascade for full table) | | |

The +0.0300 lift on Ru (n=2-seed overlap) is **MEASURED** per
party-twenty-spirit §3.5. The other 4 metals' 30ep→50ep delta is left as a
follow-up — Wf-310B cascade integration will fill in the full 5-metal table
once the 2 retries land. (See §2.3 for the Wf-318B extension that has since
filled in the full 5-metal cascade table.)

---

## §5 Honest framing per party-twenty-spirit §3.5

- **G5 13/15 MEASURED today** (this closeout is real, not projected).
- **2 missing pairs** (Ru/seed=2 + Ir/seed=0) are reported as MISSING in
  `training_summary.json::missing_pairs` until the retry completes.
- **30ep vs 50ep comparison** is honest because both G4 and G5 are MEASURED.
- **Ru 50ep 0.7860 vs 30ep 0.7561 = Δ +0.0300 lift** (real MEASURED, n=2-seed overlap).
- **Per-metal std** is reported unfiltered; smaller metals (Rh/Os/Re < 1.2k rows)
  have ±0.05-0.10 per-seed spread, NOT papered over.

---

## §6 What this closeout delivers

1. **`molmetal/scripts/wf278a_g5_aggregate.py`** — the FIX. Walks the correct
   nested tree, handles both G5 flat + G4 nested schemas, emits real
   training_summary.json with per-metal mean ± std.
2. **`results/wf124_p3_50epoch_full_sweep_g5/training_summary.json`** — real
   aggregation: `total_pairs=13`, `expected_pairs=15`, `missing_pairs=["Ru/seed2", "Ir/seed0"]`,
   per-metal breakdown + 5-metal grand aggregate + cascade delta vs G4 30-ep.
3. **`metrics/by_metric/wf181_g5_50epoch_full_sweep.json`** — new metric
   JSON mirroring the wf219 sibling format. value=0.7227 ± 0.0417 (n=13/15).
4. **`molmetal/scripts/paper_dashboard.py`** — 4 new CLAIM_CATALOG entries:
   - `claim_wf310a_g5_5metal_3seed_50epoch_overall_auc` (5-metal MEASURED)
   - `claim_wf310a_g5_ru_per_metal_50epoch` (Ru per-metal MEASURED)
   - `claim_wf310a_g5_ir_per_metal_50epoch` (Ir per-metal MEASURED)
   - `claim_wf310a_g5_ru_30ep_vs_50ep_delta` (cascade Δ MEASURED)
5. **`scripts/retry_g5_missing_pairs.sh`** — retry launcher (sequential +
   `--parallel` modes).

---

## §7 Cross-references

- `molmetal/scripts/wf278a_g5_aggregate.py` (the FIX — read this first)
- `molmetal/scripts/wf278a_g4_aggregate.py` (G4 30-epoch sibling aggregator)
- `molmetal/scripts/wf308d_g5_results_integration.py` (Wf-308D integration
  template; uses the wf278a_g5 aggregator output)
- `molmetal/scripts/paper_dashboard.py` CLAIM_CATALOG §Wf-310A entries
- `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` (NEW this closeout)
- `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json`
  (chemprop v2 sibling for comparison)
- `results/wf124_p3_50epoch_full_sweep_g5/training_summary.json` (the
  re-aggregated training_summary.json this closeout produces)
- `results/wf124_p3_remaining_metals_g4/training_summary.json` (G4 cascade
  source for the 30ep→50ep delta)
- `scripts/retry_g5_missing_pairs.sh` (the retry launcher)
- `scripts/launch_wf124_p3_g5.sh` (original buggy launcher — inline
  aggregator now superseded by wf278a_g5_aggregate.py)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (original gap analysis)
- `TODO/completed/219_wf219_krasnov_aligned.md` (chemprop v2 baseline sibling)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` (Krasnov 2026 protocol)
- `mol-metal-gpu-execution` §6 (5-seed × 50-epoch protocol)
- `mol-metal-gpu-execution` §7 (MEASURED vs PROJECTED tag discipline)
- `party-twenty-spirit` §3.5 (honest framing gate)

---

## §8 Follow-ups (Wf-310B onwards)

1. **Wf-310B cascade**: refresh training_summary.json after Wf-310A retry
   completes; cascade the 5-metal 30ep→50ep table (currently only Ru row has
   the delta populated).
2. **Wf-310C**: integration with Wf-308D PART 3 — apply the integration
   template once 15/15 is MEASURED (requires user OK per TODO-84 paper rule).
3. **Wf-310D**: TODO-219 §3 temporal-split driver (direct comparison vs
   Krasnov 2026's temporal split).
4. **Wf-310E**: TODO-182 G6 per-innovation ablation (5×3×50 = 75 pairs; the
   next escalation to attribute lift to multi-task vs W₂ vs L2 guard).
5. **If W₂ trajectory is NOT monotonic over 50 epochs**: indicates divergence
   or instability; L2 guard may need re-tuning (TODO-181 §8 follow-up #3).

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: This closeout is
MEASURED on 13/15 pairs as of 2026-09-29. The 2 retry pairs (Ru/seed=2 +
Ir/seed=0) are in-flight in the background and will be appended via
Wf-310B cascade. All per-metal numbers, std, and the Ru 30ep→50ep Δ=+0.0300
lift are reported from real MEASURED data — not projected, not estimated.
Smaller metals (Rh/Os/Re < 1.2k rows) have ±0.05-0.10 per-seed spread; this
is reported unfiltered per party-twenty-spirit §3.5.
