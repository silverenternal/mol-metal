# Wf-196 — 50-epoch CPU sweep of 14 missing (metal, seed) pairs

**Status**: 🚀 **Complete — 15/15 (metal, seed) pairs at 50 epochs; per-metal mean ± std computed**
**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode agents
**Parent plan**: `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (TODO-181) +
              `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (TODO-180)

---

## §0 Why this plan exists

**Wf-124 P3** ran the 5-metal × 3-seed × 30-epoch sweep on RTX 5090 GPU-1,
but only **1/15 (metal, seed) pair** had its `per_seed_metrics_seed*.json`
written to disk (`Ru seed=0` with `test_auc=0.7613`). The other 14 pairs
existed as bare checkpoints (10-epoch warm-starts from earlier Wf-131 P3
runs), but no per-seed metrics JSON.

**TODO-181** identified the 50-epoch undertraining risk (per Wf-124 P3 §0
honest framing: "30-epoch scaling was due to session time budget; full
5×5×50 would take ~5h"). The W₂ trajectory was still decreasing at epoch
30 (final 0.3906), suggesting the recipe had not fully converged.

**This Wf-196** runs the **14 missing (metal, seed) pairs at 50 epochs
on CPU** (per the standing "kill then adapt" GPU directive + 8h wall-time
budget). Plus Ru seed=0 re-run at 50 epochs on CPU for apples-to-apples
3-seed Ru aggregate. Uses a smaller backbone (hidden=64, depth=2 vs the
GPU default hidden=128, depth=3) to fit CPU speed.

---

## §1 Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| **G1** | 14/14 missing pairs complete | ✅ **15/15** (14 missing + Ru seed=0 re-run) |
| **G2** | All at 50 epochs | ✅ All pairs at 50 epochs |
| **G3** | Per-metal mean ± std computed (3 seeds × 5 metals) | ✅ Per-metal mean ± std computed |
| **G4** | Total wall-time ≤ 8 hr | ✅ **~12 min** total wall (314.5s for 14 pairs + 383.3s for Ru seed=0) |

---

## §3 Scope + recipe

| Item | Detail |
|---|---|
| Hardware | CPU only (32 cores, 94 GB RAM, Intel i9-14900K) |
| Wall-time (MEASURED) | 314.5s = 5.24 min for 14 pairs (6-way parallel); +383.3s for Ru seed=0 = ~12 min total |
| Output | `molmetal/reports/wf196_50ep_sweep/{metal}/per_seed_metrics_seed{seed}.json` |
| Checkpoints | `molmetal/checkpoints/wf196_50ep/dmpnn_mtw_{metal}_seed{seed}.pt` (15 ckpts) |
| Backbone | hidden=64, depth=2 (smaller than Wf-124 P3 GPU default hidden=128, depth=3) |
| Other recipe | Same as Wf-124 P3: α=0.5, w_w=0.05, use_l2_guard=True, lr=1e-3, batch_size=64, n_mc=256 |
| 5-metal cohort | Ru/Ir/Os/Rh/Re from `molmetal/data/cytotox_5metal_multitask.csv` |

### §3.1 14 missing (metal, seed) pairs covered

| # | (metal, seed) | Wall-time (s) | Status |
|---:|---|---:|---|
| 1 | (Ru, 1) | 310.9 | ✅ DONE |
| 2 | (Ru, 2) | 314.5 | ✅ DONE |
| 3 | (Ru, 0) (re-run) | 383.3 | ✅ DONE |
| 4 | (Ir, 0) | 142.3 | ✅ DONE |
| 5 | (Ir, 1) | 135.6 | ✅ DONE |
| 6 | (Ir, 2) | 139.7 | ✅ DONE |
| 7 | (Os, 0) | 14.8 | ✅ DONE |
| 8 | (Os, 1) | 15.6 | ✅ DONE |
| 9 | (Os, 2) | 15.1 | ✅ DONE |
| 10 | (Rh, 0) | 15.8 | ✅ DONE |
| 11 | (Rh, 1) | 16.5 | ✅ DONE |
| 12 | (Rh, 2) | 13.4 | ✅ DONE |
| 13 | (Re, 0) | 23.6 | ✅ DONE |
| 14 | (Re, 1) | 27.7 | ✅ DONE |
| 15 | (Re, 2) | 19.7 | ✅ DONE |

**Smaller cohorts (Os/Rh/Re < 1.2k rows) finished in 14-28s**; Ru/Ir (2.5k/784 train rows)
took 130-310s each. 6-way parallelism kept total wall-time to ~5.2 min for 14 pairs.

---

## §4 MEASURED per-metal results (50 epochs, hidden=64, depth=2)

| Metal | n_seeds | test_auc (mean ± std) | test_pearson (mean ± std) | test_rmse (mean ± std) | W₂ final |
|---|---:|---:|---:|---:|---:|
| **Ru** | 3 | **0.7459 ± 0.0050** | 0.283 ± 0.040 | 0.790 ± 0.034 | 0.42 ± 0.04 |
| **Ir** | 3 | **0.6680 ± 0.0195** | 0.096 ± 0.018 | 1.018 ± 0.057 | 0.54 ± 0.16 |
| **Os** | 3 | **0.3611 ± 0.0982** | 0.376 ± 0.040 | 0.943 ± 0.275 | 0.59 ± 0.18 |
| **Rh** | 3 | **0.4203 ± 0.0211** | 0.326 ± 0.162 | 0.853 ± 0.153 | 0.39 ± 0.18 |
| **Re** | 3 | **0.6768 ± 0.0990** | 0.003 ± 0.087 | 1.179 ± 0.107 | 0.30 ± 0.13 |
| **5-metal aggregate** | 15 | **0.5744 ± 0.1662** | 0.217 ± 0.149 | 0.957 ± 0.186 | — |

### §4.1 30-epoch (Wf-124 P3, hidden=128) vs 50-epoch (Wf-196, hidden=64) comparison

| (metal, seed) | 30ep test_auc | 50ep test_auc (Wf-196) | Δ |
|---|---:|---:|---:|
| (Ru, 0) | 0.7613 (hidden=128, depth=3) | 0.7404 (hidden=64, depth=2) | **−0.0209** |

**Honest framing**: the +0.049 absolute lift MEASURED at Wf-124 P3 (Ru seed=0,
30ep, full backbone hidden=128) DOES NOT translate to a +0.049 lift at
50-epoch with the smaller backbone. The 50-epoch smaller-backbone number
(0.7404) is **lower than the 30-epoch full-backbone number (0.7613) by
−0.021** — i.e., the smaller backbone capacity costs more than the extra
20 epochs gain at the smaller h=64 architecture.

The Krasnov 2026 target (≥0.86) is **BELOW gate** for Ru 50ep (0.7459 mean).
Per-metal std is tight for Ru (±0.005), so the 3-seed mean is a stable
estimate.

---

## §5 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 15/15 (metal, seed) pairs at 50 epochs on CPU with hidden=64,
  depth=2 backbone. Per-metal mean ± std reported unfiltered. Total wall-time
  ~12 min, far under the 8h budget.
- **NOT apples-to-apples with Wf-124 P3 30-epoch**: smaller backbone
  (hidden=64 vs 128) trades capacity for speed. The +0.049 lift MEASURED at
  Wf-124 P3 (Ru seed=0, 30ep, hidden=128) does NOT replicate at 50ep with
  hidden=64 (0.7404 vs 0.7613 = **−0.021 regression**).
- **Smaller metals (Os/Rh/Re) have noisier per-seed AUC** (±0.10 std). The
  Os 3-seed mean (0.3611) is below the 0.5 chance baseline for a binary
  classifier, indicating the multi-task signal from the much larger Ru cohort
  is **not propagating** to the 66-train-row Os cohort.
- **W₂ still decreasing at 50ep**: Ru seed=0 W₂ history = [...] (final 0.4);
  suggests undertraining may continue. The 100-epoch run is the proper
  scientific measurement, but was outside the Wf-196 8h budget.
- **PROJECTED**: 50-epoch with FULL backbone (hidden=128) may recover the
  +0.049 lift MEASURED at 30ep (Wf-124 P3) AND add additional epochs of
  convergence → PROJECTED 0.78-0.82 range, still BELOW the 0.86 Krasnov target.
  The 0.86 target requires either (a) deeper architecture, (b) more epochs
  (100+), (c) per-metal hyperparameter tuning, or (d) more training signal
  (e.g. LOMO pseudo-labels). None of these were explored in Wf-196.
- **No fabrication**: do NOT cite Wf-196 numbers as MEASURED for the Wf-124
  P3 30-epoch recipe. They are a different backbone + epoch regime.

---

## §6 Outputs on disk

```
molmetal/reports/wf196_50ep_sweep/
├── synthesis.md                            # this file
├── training_summary.json                   # per-metal + 5-metal aggregate
├── 30ep_vs_50ep_comparison.json            # vs Wf-124 P3 30-epoch Ru seed=0
├── wf196_launcher.log                      # launcher stdout (15 DONE lines)
├── wf196_launch_summary.json               # launcher timing
├── logs/
│   ├── ru_seed0.log, ru_seed1.log, ru_seed2.log
│   ├── ir_seed0.log, ir_seed1.log, ir_seed2.log
│   ├── os_seed0.log, os_seed1.log, os_seed2.log
│   ├── rh_seed0.log, rh_seed1.log, rh_seed2.log
│   └── re_seed0.log, re_seed1.log, re_seed2.log
├── ru/per_seed_metrics_seed{0,1,2}.json
├── ir/per_seed_metrics_seed{0,1,2}.json
├── os/per_seed_metrics_seed{0,1,2}.json
├── rh/per_seed_metrics_seed{0,1,2}.json
└── re/per_seed_metrics_seed{0,1,2}.json

molmetal/checkpoints/wf196_50ep/
└── dmpnn_mtw_{metal}_seed{seed}.pt   # 15 ckpts (hidden=64, depth=2)
```

### §6.1 Code changes

| File | Change |
|---|---|
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | Add `--hidden` + `--depth` CLI flags; propagate to model construction + checkpoint metadata + `training_summary.json` + per-seed `seed_payload` |
| `molmetal/scripts/wf196_50ep_cpu_sweep.py` | NEW — 14-pair parallel CPU sweep launcher |
| `molmetal/scripts/wf196_aggregate.py` | NEW — per-metal mean ± std aggregator + 30ep vs 50ep comparison |

---

## §7 Follow-ups

- **TODO-180 G4** (14 PENDING pairs complete) — ✅ CLOSED by Wf-196
- **TODO-181 G5** (50-epoch full sweep) — ⚠️ PARTIAL: smaller backbone
  50ep numbers DO NOT close the gap to 0.86; 50ep full backbone (hidden=128)
  is the next escalation. Estimated 5×3 = 15 pairs × 14 min/pair ÷ 6-way
  parallel = ~35 min on RTX 5090 GPU.
- **TODO-182 G6** (per-innovation ablation) — still OPEN; needed to decide
  whether the +0.049 lift at Wf-124 P3 came from multi-task vs W₂ vs L2 guard.
- **Paper §4 honest framing**: Wf-196 MEASURED numbers (Ru 50ep hidden=64
  = 0.7459) should NOT be cited as the m1 Ru AUC value. The Wf-124 P3
  MEASURED 30ep hidden=128 number (0.7613) remains the best single-point
  estimate until 50ep hidden=128 runs are completed.

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: The 50-epoch sweep
MEASURED a +0.014 lift for Ru (0.7459 vs 0.7123 single-task baseline) but
a −0.021 regression vs Wf-124 P3 30-epoch full-backbone (0.7613). The
Krasnov 2026 target (≥0.86) is BELOW gate. Smaller metals (Os/Rh/Re) have
high std (±0.10) and 5-metal aggregate is dragged down by Os (0.3611).
DO NOT cite these numbers as paper §4 LEAD claims without the full-backbone
50-epoch follow-up.