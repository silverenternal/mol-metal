# Wf-359-A2 — m1 multi-seed evaluation scaffold (CPU-only; deferred GPU launch)

**Date**:        2026-09-29
**Workflow**:    Wf-359-A2 (ultracode residual wave)
**Status**:      SHIPPED — scaffold + 4 NEW tests PASS (CPU-only; 0 GPU launches)
**Owner**:       Hugo Lee + ultracode agent (per AGENTS.md §1.2 + §5.1 + §5.2)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per Wf-357-B3 §4 ("m1 Ru AUC at 0.7613 (1-seed) is the canonical Wf-124
P3 anchor and is HONEST_NEGATIVE for the Krasnov ≥ 0.86 multi-seed gate")
+ Wf-355-B1 (USER_OK-gated Wf-131 P3 7 DEFERRED-runs wrapper), the m1 Ru
AUC metric needs a **multi-seed aggregation scaffold** so the
USER_OK-gated Wf-355-B1 launch wrapper can hand off its outputs and
produce the paper-grade multi-seed mean ± std the moment the 7
DEFERRED runs (Rh 3+4 + Re 0-4) complete.

**Wf-359-A2 closes this gap with**:

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Scaffold script** | `molmetal/scripts/eval_m1_multi_seed.py` | 314 LOC | SHIPPED; CPU-only aggregation READY |
| **Test file** | `molmetal/tests/test_wf359a2_m1_multi_seed.py` | 289 LOC (4 tests) | 4/4 PASS (0.33 s wall, CPU-only) |
| **Aggregate output** | `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` | EMITTED on scaffold run | 5-metal schema |

**Tests**: 4/4 PASS
- `test_wf359a2_load_per_seed_metrics` — scaffold loads per-seed metrics from wf131_p3_lomo_full/ fallback (≥3 seeds for at least one metal)
- `test_wf359a2_aggregate_mean_std` — mean=0.70 / std=√0.005 MEASURED on synthetic 5-seed input
- `test_wf359a2_honest_negative_1seed` — 1-seed input emits HONEST_NEGATIVE_1SEED (no silent 5-seed faking)
- `test_wf359a2_integration_with_wf131_p3_wrapper` — output JSON schema compatible with Wf-131 P3 + Wf-355-B1 wrapper; Re=NOT_MEASURED honestly

---

## §1 The m1 multi-seed gap (per Wf-357-B3 §4)

### §1.1 m1 Ru AUC current state

| Axis | MEASURED | Source |
|---|---:|---|
| **Wf-124 P3 Ru seed=0 test_auc (30-epoch, multi-task + Wasserstein + L2 guard)** | **0.7613** | `metrics/by_metric/metalcytotox_ru_auc.json` `value=0.7613038529802696` |
| Single-task 5-epoch baseline | 0.7123 | same JSON `baseline_value=0.7122770919067215` |
| Lift vs baseline | +0.0490 | same JSON `lift_vs_baseline` |
| Gap to Krasnov ≥ 0.86 gate | -0.0987 | same JSON `gap_to_gate` |
| Hit gate | false | same JSON `hit_gate=false` |
| n_seeds | 1 (seed=0) | same JSON `n_seeds=1` + `per_seed_values=[0.7613...]` |
| Verdict | MEASURED_PARTIAL_LIFT (gate NOT YET HIT) | same JSON `verdict` field |

**Honest framing**: The Wf-124 P3 30-epoch Ru seed=0 lifts AUC from 0.7123 → 0.7613 (+0.049). The Krasnov 2026 ≥ 0.86 gate is **NOT YET HIT** (gap remaining = -0.099). The single-seed reading is honest-multi-seed-pending — when the Wf-355-B1 wrapper re-runs the 7 DEFERRED (metal, seed) runs, the aggregate mean ± std will close the multi-seed gap.

### §1.2 The 7 DEFERRED runs (per Wf-354-A1 §5 + Wf-355-B1 §1.2)

| # | Metal | Seed | DEFERRED reason |
|:---:|:---:|:---:|---|
| 1 | Rh | 3 | training stopped at epoch 23 (last log 2026-09-27 09:05:40) |
| 2 | Rh | 4 | never started |
| 3 | Re | 0 | never started |
| 4 | Re | 1 | never started |
| 5 | Re | 2 | never started |
| 6 | Re | 3 | never started |
| 7 | Re | 4 | never started |

Per Wf-355-B1 §1.3: the **Wf-355-B1 wrapper** (`scripts/launch_wf131_p3_deferred_runs.sh`) is USER_OK-gated and ready to launch the 7 runs (5 hr GPU, sequential). When USER_OK=1 is set, the wrapper runs `train_metal_lomo_wasserstein.py` (the Wf-131 P3 canonical entry per Wf-354-A1 §3.2) for each (metal, seed) pair, writes per-seed JSON to `molmetal/reports/wf131_p3_deferred_runs/{metal}/per_seed_metrics_seed{N}.json`, and exits.

The Wf-359-A2 scaffold AGGREGATES these per-seed JSONs as soon as they land on disk.

---

## §2 Scaffold design

### §2.1 Path resolution

| Source | Path | Status |
|---|---|---|
| Primary (Wf-355-B1 wrapper output) | `molmetal/reports/wf131_p3_deferred_runs/` | Empty today (awaiting USER_OK launch) |
| Fallback (Wf-131 P3 anchor on disk) | `molmetal/reports/wf131_p3_lomo_full/` | 18/25 seeds present (Ru 5/5 + Ir 5/5 + Os 5/5 + Rh 3/5 + Re 0/5) |
| Output (aggregate) | `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` | EMITTED on scaffold run |

The scaffold tries the **primary** dir first (Wf-355-B1 wrapper's expected output). If a metal's per-seed JSONs are not there, it falls back to the **fallback** dir (the Wf-131 P3 anchor on disk today). This way the scaffold is **forward-compatible** — when the Wf-355-B1 wrapper completes, the next scaffold run will automatically pick up the new per-seed JSONs from the primary dir.

### §2.2 Per-seed JSON schema (Wassertein + LOMO)

Two on-disk conventions are supported:

1. **Aggregate form** (`per_seed_metrics.json`): a single JSON file containing `per_seed: [{seed, test_auc, val_auc, n_train, n_test, test_pic50_mae}, ...]` plus `n_seeds_total`, `n_seeds_trained_now`, `test_auc_mean`, `test_auc_std`.
2. **Per-seed files** (`per_seed_metrics_seedN.json`): one JSON per seed with `fold`, `seed`, `test_auc`, `val_auc`, `n_train`, `n_test`, `test_pic50_mae`, `wasserstein_loss_history`.

The scaffold tries (1) first, falls back to (2). Both shapes are in the Wf-124 P3 + Wf-131 P3 codebase per `train_dmpnn_multitask_wasserstein.py` per-seed-write logic.

### §2.3 Honest-negative contract (per Wf-357-B3 + Wf-355-B1)

The scaffold emits **explicit verdict strings** keyed on `n_seeds_used`:

| n_seeds | verdict | semantics |
|:---:|---|---|
| 0 | `NOT_MEASURED` | No per-seed data on disk; emit empty block (mean=None, std=None) |
| 1 | `HONEST_NEGATIVE_1SEED` | Single-seed reading; emit the 1-seed value with explicit `n_seeds_used=1` tag; **no fake std** |
| 2 | `PARTIAL_2SEED` | 2-seed mean; population std on 2-sample pair |
| ≥3 | `MEASURED` | mean ± population std |

This tag discipline preserves `party-twenty-spirit` §3.5 honest framing — the scaffold NEVER silently promotes 1-seed data to a 5-seed mean.

### §2.4 Output schema (aggregate.json)

```json
{
  "schema": "wf359a2_m1_multi_seed_scaffold.v1",
  "wf_origin": "Wf-359-A2 (ultracode residual wave; Wf-357-B3 multi-seed gap)",
  "parent": "Wf-355-B1 wrapper + Wf-131 P3 anchor + Wf-357-B3 verification",
  "primary_dir": ".../wf131_p3_deferred_runs",
  "fallback_dir": ".../wf131_p3_lomo_full",
  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],
  "per_metal": {
    "Ru": {
      "metal": "Ru",
      "n_seeds_used": 5,
      "seeds_used": [0, 1, 2, 3, 4],
      "per_seed_aucs": [0.5911, 0.6185, 0.5881, 0.5917, 0.5810],
      "test_auc_mean": 0.5941,
      "test_auc_std": 0.0143,
      "source_path": ".../wf131_p3_lomo_full/Ru/per_seed_metrics.json",
      "verdict": "MEASURED",
      "baseline_krasnov_2026": 0.7123,
      "gate_target": 0.86,
      "lift_vs_baseline": -0.1182,
      "gap_to_gate": -0.2659,
      "hit_gate": false
    },
    ...
    "Re": {
      "metal": "Re",
      "n_seeds_used": 0,
      "seeds_used": [],
      "per_seed_aucs": [],
      "test_auc_mean": null,
      "test_auc_std": null,
      "source_path": "<missing: .../wf131_p3_lomo_full/Re>",
      "verdict": "NOT_MEASURED"
    }
  },
  "overall_mean": 0.6283,
  "overall_std": 0.0383,
  "n_metals_with_data": 4,
  "honest_framing": {
    "MEASURED_today": "CPU-only aggregation scaffold; no GPU launches",
    "PROJECTED": "When Wf-355-B1 USER_OK launch completes, re-running this script will populate per_metal Rh + Re from wf131_p3_deferred_runs/ rather than fallback LOMO-full data",
    "honest_negative_note": "1-seed-only readings emit HONEST_NEGATIVE_1SEED verdict with n_seeds_used=1 to preserve MEASURED-vs-PROJECTED tag discipline",
    "kill_then_adapt_directive": "CPU-only; no retrain / panel / ablation"
  },
  "verdict": "READY_FOR_USER_OK_LAUNCH"
}
```

**Schema contract**:
- `per_metal.Ru.baseline_krasnov_2026` = 0.7123 (m1 Ru-specific; **only present for Ru**)
- `per_metal.Ru.gate_target` = 0.86 (m1 Ru-specific; **only present for Ru**)
- `per_metal.{metal}.lift_vs_baseline` and `gap_to_gate` and `hit_gate` are **only present for Ru** and **only when n_seeds_used >= 1**

The schema is intentionally extensible — future waves (Wf-360+) can add `per_metal.{metal}.metal_krasnov_baseline` for Ir/Os/Rh/Re if those gates are formalized.

---

## §3 Test inventory (4/4 PASS; CPU-only)

### §3.1 test_wf359a2_load_per_seed_metrics

| Aspect | Value |
|---|---|
| LOC | ~50 |
| What it tests | Primary + fallback dirs exist; Ru block has ≥3 seeds; per_seed_aucs all in [0,1] |
| Honest framing | Real on-disk data; no synthetic mocks |

### §3.2 test_wf359a2_aggregate_mean_std

| Aspect | Value |
|---|---|
| LOC | ~45 |
| What it tests | Synthetic 5-seed input [0.60, 0.65, 0.70, 0.75, 0.80] → mean=0.70, std=√0.005 ≈ 0.07071; verdict=MEASURED; m1 Ru-specific lift/gate fields present |
| Honest framing | Pure arithmetic verification; no IO |

### §3.3 test_wf359a2_honest_negative_1seed

| Aspect | Value |
|---|---|
| LOC | ~30 |
| What it tests | Synthetic 1-seed input [0.7613] → verdict=HONEST_NEGATIVE_1SEED; test_auc_std=0.0 (no fake std); m1 Ru-specific lift computed correctly |
| Honest framing | Verifies the scaffold NEVER silently promotes 1-seed to 5-seed |

### §3.4 test_wf359a2_integration_with_wf131_p3_wrapper

| Aspect | Value |
|---|---|
| LOC | ~75 |
| What it tests | Primary dir exists (Wf-355-B1 wrapper output dir); aggregate JSON has all required keys; 5-metals default present; Ru/Ir/Os MEASURED (≥3 seeds); Rh MEASURED (3 seeds); **Re NOT_MEASURED (0 seeds, honestly)** |
| Honest framing | Re 0/5 DEFERRED case is reported honestly (no fake aggregation); the integration test would FAIL if the scaffold silently faked Re data |

---

## §4 Honest framing per `party-twenty-spirit` §3.5

### §4.1 What this scaffold MEASURED today

1. ✅ Scaffold `molmetal/scripts/eval_m1_multi_seed.py` SHIPPED (314 LOC, `python -c "import ..."` clean)
2. ✅ 4 NEW tests PASS (CPU-only, 0.33 s wall)
3. ✅ Aggregate emitted to `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` (5-metal schema)
4. ✅ Ru: 5 seeds MEASURED (0.5941 ± 0.0128)
5. ✅ Ir: 5 seeds MEASURED (0.6242 ± 0.0165)
6. ✅ Os: 5 seeds MEASURED (0.6075 ± 0.0263)
7. ✅ Rh: 3 seeds MEASURED (0.6874 ± 0.2926; high std reflects Wf-125 P1 BUGGY cohort per Wf-355-B1 §1.3)
8. ✅ Re: 0 seeds NOT_MEASURED (honest; 0/5 DEFERRED per Wf-354-A1 §5)
9. ✅ Zero GPU launches performed (CPU-only verification)
10. ✅ No `paper/*` touches (TODO-84 honored)

### §4.2 What this scaffold does NOT claim

1. ❌ Does NOT claim the Krasnov 2026 ≥ 0.86 gate is HIT — Ru 5-seed mean 0.5941 is BELOW 0.7123 baseline (which itself is the Wf-131 P1 baseline, NOT the Wf-124 P3 1-seed 0.7613 anchor)
   - **Critical honest framing**: the Wf-131 P3 5-seed LOMO means (Ru 0.5941, Ir 0.6242, Os 0.6075) are LOWER than the Wf-124 P3 1-seed Ru 0.7613 because they are on DIFFERENT cohort splits:
     - Wf-131 P3 = 5-metal LOMO split (5-fold CV, leave-one-metal-out)
     - Wf-124 P3 = Ru-only held-out test set (different cohort)
   - The Krasnov gate refers to Wf-124 P3-style test AUC, NOT Wf-131 P3 LOMO mean
   - The scaffold's `lift_vs_baseline` and `gap_to_gate` fields are computed on the LOMO-fallback values; they are NOT directly comparable to the Wf-124 P3 anchor
2. ❌ Does NOT claim the Wf-355-B1 wrapper has been launched — USER_OK gate remains HARD-REQUIRED per AGENTS.md §5.1 + Wf-294B / Wf-300C pattern
3. ❌ Does NOT claim the 7 DEFERRED runs are COMPLETE — they remain DEFERRED until USER_OK=1 launches the Wf-355-B1 wrapper
4. ❌ Does NOT modify `metrics/by_metric/metal_lomo_auc.json` or `metalcytotox_ru_auc.json` — output is placed in `molmetal/reports/wf359a2_m1_multi_seed_scaffold/` and is operator-promoted manually after honest-negative review
5. ❌ Does NOT claim the Rh high-std reading is resolved — Wf-355-B1 §1.3 notes that even after re-running Rh seeds 3+4, the result will still be on the BUGGY Wf-125 P1 cohort, so the per-seed variance may not clear the paper §4 acceptance gate

### §4.3 Cross-metric honest framing

The scaffold's `lift_vs_baseline` and `gap_to_gate` fields use `baseline_krasnov_2026=0.7123` (the Wf-124 P3 single-task 5-epoch baseline per `metrics/by_metric/metalcytotox_ru_auc.json:baseline_value`). This is the canonical m1 baseline.

**However**, the Wf-131 P3 LOMO 5-seed mean (Ru 0.5941) is BELOW this baseline because LOMO is a different evaluation protocol. The honest framing is:

- m1 Wf-124 P3 1-seed (30-epoch multi-task) = **0.7613** (canonical Ru test_auc reading)
- m1 Wf-131 P1 3-seed × 10-epoch (multi-task) = **0.7352 ± 0.0023** (canonical Ru multi-seed mean reading)
- m1 Wf-131 P3 5-seed LOMO mean (multi-task + Wasserstein + L2 guard) = **0.5941** (LOMO protocol; NOT directly comparable to above)

The scaffold's `per_metal.Ru` block reports the LOMO 5-seed mean (0.5941) since that's what's on disk in `wf131_p3_lomo_full/Ru/`. When the Wf-355-B1 wrapper completes, the new per-seed JSONs from `wf131_p3_deferred_runs/Ru/` (if any Rh/Re seeds complete that affect Ru — they don't, since Ru is COMPLETE) will not change the Ru block; the Rh + Re blocks will be populated instead.

**The Krasnov 2026 ≥ 0.86 gate verdict remains MEASURED_PARTIAL_LIFT (gate NOT YET HIT)** — the Wf-124 P3 1-seed 0.7613 is the canonical anchor and requires the TODO-181 G5 launch (5-seed × 50-epoch) to fully resolve.

---

## §5 Files touched

| Path | Action | LOC |
|---|---|---:|
| `molmetal/scripts/eval_m1_multi_seed.py` | NEW | 314 |
| `molmetal/tests/test_wf359a2_m1_multi_seed.py` | NEW | 289 (4 tests) |
| `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` | EMITTED on scaffold run | ~85 lines (5-metal schema) |
| `TODO/completed/359a2_m1_multi_seed_scaffold.md` | NEW (this doc) | ~330 |

**No source files were modified** beyond the NEW scaffold + tests + this doc + the emitted aggregate JSON.

---

## §6 Cross-references

- **Wf-357-B3** — `TODO/completed/357b3_tier2_honest_neg_metrics.md` §4 (m1 Ru AUC 0.7613 1-seed anchor; multi-seed gap)
- **Wf-355-B1** — `TODO/completed/355b1_wf131_p3_deferred.md` §1.1, §1.2, §2 (USER_OK-gated Wf-131 P3 7 DEFERRED-runs wrapper)
- **Wf-354-A1** — `TODO/completed/354a1_wf342_p2_lomo_integration.md` §1.2 + §5 (parent; 7 DEFERRED runs origin)
- **Wf-131 P3 partial update** — `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update` block
- **Wf-124 P3 anchor** — `metrics/by_metric/metalcytotox_ru_auc.json` `value=0.7613` (1-seed, MEASURED_PARTIAL_LIFT)
- **Wf-260 G6 timing** — per-metal wall-time estimate source
- **Wf-294B + Wf-300C** — USER_OK gate pattern reference
- **Wf-326G §3.4 / Wf-357-D1** — scaffold test pattern reference (synthetic cohort + per-seed JSON loading)
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.1 standing GPU authorization + §5.2 honest framing mandate + §5.3 no paper work + §5.4 Tier-2 markers UNCHANGED
- **`party-twenty-spirit`** §3.5 (MEASURED-vs-PROJECTED tag discipline + 0-tolerance honest framing)

---

## §7 What happens after USER_OK (per AGENTS.md §5.1)

When the user clears the multi-seed resolution gate and approves the Wf-355-B1 launch:

1. **Launch command**:
   ```bash
   USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh
   ```
2. **Wall**: ~5 hr GPU sequential (PROJECTED per Wf-260 G6 timing)
3. **Outputs**: per-(metal, seed) AUCs at `molmetal/reports/wf131_p3_deferred_runs/{metal}/per_seed_metrics_seed{N}.json`
4. **Aggregation** (this Wf-359-A2 scaffold):
   ```bash
   PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/eval_m1_multi_seed.py
   ```
5. **Result**: `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` re-emitted with new Rh seeds 3+4 + Re seeds 0-4 populated (n_seeds_used=5 for Rh+Re); overall_mean recomputed
6. **Operator review** (TODO-84 gated): the operator reviews the new aggregate against `metrics/by_metric/metal_lomo_auc.json` and decides whether to promote any of the new per-metal readings into `metrics/by_metric/*.json` (per party-twenty-spirit §3.5 honest framing)

---

**END of Wf-359-A2** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; SHIP (CPU-only verification; 4 NEW tests; 0 GPU launches; 1 NEW scaffold; 1 NEW test file; 1 NEW aggregate JSON; 1 NEW doc; 5 honest-negative caveats preserved per §4.2).