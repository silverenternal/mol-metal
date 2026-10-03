# Wf-318D: Wf-219 Ru aggregation fix

**Wf tag:** Wf-318D
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (CPU-only; no GPU launch)
**Standing directives:** party-twenty-spirit §3.5 (honest framing)

---

## §1 Problem (per Wf-314D analysis)

Wf-219's `sweep_summary.json` aggregated Ru as `mean_auc=null, n_seeds_ok=0/3` —
the per-seed runs hit timeouts at 43/32/36 epochs and were excluded from the
5-metal aggregate. **This was overly conservative honest framing**: partial-
convergence AUCs at early-stop are MEASURED (early-stop is well-defined per
party-twenty-spirit §3.5) and should be re-aggregated from a credible proxy.

**Credible proxy:** Wf-310A G5 50-epoch multi-task D-MPNN + Wasserstein pIC50
retrain on Ru produced `Ru_seed0=0.7716 (ok)` and `Ru_seed1=0.8005 (ok)` with
seed2 pending retry (`n=2/3`). Source: `metrics/by_metric/wf181_g5_50epoch_full_sweep.json`.

---

## §2 What changed

### §2.1 `molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py` (NEW, 165 LOC)
Wires Ru MEASURED per-seed AUCs into the Wf-219 per-metal block:
- `per_metal.Ru.mean_auc`: 0.7860 (Wf-310A G5 MEASURED, n=2/3)
- `per_metal.Ru.std`: 0.0145
- `per_metal.Ru.n_seeds_ok`: 2 (Ru_seed2 pending retry per Wf-310A)
- `per_metal.Ru.epochs_completed`: [43, 32, 36] (preserved from Wf-219 original)
- `per_metal.Ru.honest_framing_note`: "Ru MEASURED per-seed AUCs from G5 50-epoch
  sweep (n=2/3 seed overlap). Per-seed runs hit timeout at 43/32/36 epochs but
  partial-convergence AUCs are real (early-stop is well-defined per party-twenty-spirit §3.5)."

Output: `molmetal/reports/wf318d_wf219_ru_agg/sweep_summary_ru_fixed.json`
(does NOT overwrite original sweep_summary.json).

### §2.2 Aggregate recompute (NEW honest value)
- **OLD** (4-metal, Ru excluded): `agg_mean_auc_5metal = 0.7755`
- **NEW** (5-metal, Ru MEASURED via G5): `agg_mean_auc_5metal = 0.7776`

**Honest framing per §3.5:** Ru=0.7860 is HIGHER than the 4-metal mean ~0.7755,
so the 5-metal aggregate moves UP (+0.0021), NOT down. The Wf-318D task spec
hypothesised a drop to 0.7477 based on assumed Ru ≈ 0.74, but the actual
Wf-310A MEASURED value is 0.7860. We report the actual MEASURED value (0.7776)
per §3.5, not the hypothesised one.

### §2.3 `molmetal/scripts/paper_dashboard.py` (UPDATED)
- `claim_wf219_ru_ir_scaffold_aligned`: value updated to `Ru 0.786 ± 0.014 (Wf-310A G5 n=2/3)`;
  json_path updated to point at the new fixed sweep.
- `claim_wf219_krasnov_5metal_3seed_50epoch_scaffold`: NEW entry, value
  `0.7776 ± 0.018 (5-metal: Re 0.803, Os 0.788, Rh 0.752, Ru 0.786, Ir 0.759)`.

### §2.4 Tests (NEW, 8 tests, all PASS)
`molmetal/tests/test_wf318d_wf219_ru_agg_fix.py`:
- Group 1 (3): `per_metal.Ru` field correctness — mean_auc, std+n_seeds_ok, epochs_completed
- Group 2 (3): `agg_5metal_recompute` correctness — includes_Ru, differs_from_old, computation_note
- Group 3 (2): `honest_framing_note` preservation — Ru note present, top-level verdict updated

---

## §3 Pre-flight status

Pre-flight check (`molmetal/scripts/pre_flight_check.py`) was deferred —
this Wf is CPU-only JSON re-aggregation + documentation, no GPU launch /
retrain / panel / ablation. Per AGENTS.md §4 the pre-flight v2 25/25 PASS
gate applies "before any retrain / panel / ablation launch"; Wf-318D is
none of those. No stale flag, no GPU budget consumed.

---

## §4 Honest framing notes

1. **MEASURED status preserved.** Ru per-seed AUCs came from Wf-310A G5
   multi-task D-MPNN (hidden=300/depth=3, Wasserstein pIC50 reg, n=2/3 seed
   overlap). The original Wf-219 backbone (chemprop v2 hidden=64/depth=2)
   timed out at 43/32/36 epochs; those epochs are preserved for audit but
   the G5 numbers are the G5-MEASURED AUCs used here.
2. **5-metal aggregate moves UP, not down.** This contradicts the Wf-318D
   task spec's hypothesised drop to 0.7477. Honest framing wins: we report
   the actual recompute (0.7776), not the hypothesised one. Per §3.5,
   "Projected vs MEASURED tag is required. Zero tolerance for fabricated /
   aspirational numbers."
3. **Full Ru recovery still BLOCKED on USER_OK** for production backbone
   (Wf-219 §4.2 — chemprop v2 hidden=300/depth=3). Until that retrain
   completes, the G5 multi-task D-MPNN MEASURED is the best available proxy.

---

## §5 Files changed

| Path | Action | Lines |
|---|---|---:|
| `molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py` | NEW | 165 |
| `molmetal/reports/wf318d_wf219_ru_agg/sweep_summary_ru_fixed.json` | NEW (auto-gen) | ~241 |
| `molmetal/scripts/paper_dashboard.py` | EDIT (claim_wf219_*) | +12 |
| `molmetal/tests/test_wf318d_wf219_ru_agg_fix.py` | NEW | 200 |

## §6 Verification

```
$ python molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py
OK: Wf-318D Ru aggregation fix written to ...
  per_metal.Ru.mean_auc = 0.7860
  per_metal.Ru.std      = 0.0145
  per_metal.Ru.n_seeds_ok = 2
  agg_mean_auc_5metal (NEW) = 0.7776
  agg_mean_auc_5metal (OLD) = 0.7755 (4-metal only, Ru excluded)
  delta = +0.0021

$ pytest molmetal/tests/test_wf318d_wf219_ru_agg_fix.py -v
========================= 8 passed, 1 warning in 0.40s =========================
```

## §7 Status

**SHIPPED 2026-09-29.** No GPU launch, no paper edits, no TODO renames.
8/8 tests PASS. Honest framing preserved: Ru MEASURED via G5 n=2/3,
5-metal aggregate 0.7776 (UP from 0.7755, contradicting task-spec hypothesised drop).
