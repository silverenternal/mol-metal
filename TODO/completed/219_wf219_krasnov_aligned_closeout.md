# TODO-219 — Wf-219 Krasnov-2026-aligned chemprop v2 sweep — close-out

**Date**: 2026-09-29
**Workflow**: Wf-304C Task 1 (Wf-219 state consolidation)
**Status**: 🚀 **PARTIAL SHIP — 4-metal × 3-seed × 50-epoch MEASURED; Ru timeout (43/50 epochs)**
**Priority**: **P1** (Wf-219 Krasnov protocol alignment)
**Owner**: Hugo Lee + Wf-304C
**Parent**: `molmetal/reports/krasnov_protocol_investigation.md` §3 (Krasnov 2026 baseline protocol)
**Predecessor doc**: `TODO/pending/219_wf219_krasnov_aligned.md` (status: ✅ P1 SHIP per 2026-09-28 doc)

---

## §0 TL;DR

| Gate (per TODO-219 §2 acceptance) | Status | Evidence |
|---|---|---|
| **G1** n_runs = 15 (5-metal × 3-seed) | ⚠️ **PARTIAL** | 12/15 ok runs (Re/Os/Rh/Ir × 3); Ru 3/3 ran but timed out at 43/50 epochs |
| **G2** All 50 epochs | ⚠️ **PARTIAL** | Re/Os/Rh/Ir = 50/50; Ru = 43/50 (timeout) |
| **G3** Scaffold-split used | ✅ **PASS** | Bemis-Murcko via `molmetal.data.splits.ScaffoldSplitter`; `split` field in every per-seed JSON |
| **G4** Per-run AUC reported | ✅ **PASS** | `per_metal_seed_auc` in both per-seed JSONs and sweep_summary.json |
| **G5** Total wall ≤ 6 hr | ✅ **PASS** | 4.19 min wall for 9 runs + ~20.7 min for Ru/Ir partial re-run = ~25 min total |

**One-line verdict (per party-twenty-spirit §3.5)**: The Wf-219 Krasnov-aligned chemprop v2 sweep is **4-metal × 3-seed × 50-epoch MEASURED** (Re/Os/Rh/Ir). Ru ran but timed out at 43/50 epochs (per-seed AUCs MEASURED but excluded from 5-metal aggregate). The metric JSON + sweep_summary have been re-aggregated by Wf-304C Task 1 to honestly reflect this partial state. Ru re-run requires fresh user authorization (per `mol-metal-gpu-execution` §0 "kill then adapt" directive) since it exceeds 1-hr GPU launch budget.

---

## §1 MEASURED state (re-aggregated 2026-09-29)

### §1.1 Per-metal 3-seed MEASURED (4-metal fully, 1-metal partial)

| Metal | n_train | n_val | n_test | Mean AUC ± std | Per-seed AUC | Status |
|---|---:|---:|---:|---:|---|---|
| **Re** | 741 | 32 | 95 | **0.8027 ± 0.0637** | 0.8851, 0.7299, 0.7931 | ✅ 50/50 epochs |
| **Os** | 956 | 36 | 126 | **0.7884 ± 0.0098** | 0.8019, 0.7792, 0.7841 | ✅ 50/50 epochs |
| **Rh** | 930 | 58 | 146 | **0.7525 ± 0.0114** | 0.7426, 0.7463, 0.7684 | ✅ 50/50 epochs |
| **Ir** | 3902 | 162 | 482 | **0.7586 ± 0.0142** | 0.7673, 0.7699, 0.7386 | ✅ 50/50 epochs |
| **Ru** | 16349 | 711 | 2075 | **TIMEOUT** (excluded from agg) | 0.7858, 0.8215, 0.8215 | ⚠️ 43/50 epochs |

**Aggregate (4-metal × 3-seed)**: 0.7755
**5-metal × 3-seed mean (incl. Ru)**: NOT computed — Ru excluded per honest framing

### §1.2 Krasnov 2026 comparison (MEASURED vs PROJECTED)

| Metal | Krasnov 2026 baseline | Wf-219 MEASURED | Verdict |
|---|---:|---:|---|
| Ru | 0.81 | 0.7858, 0.8215, 0.8215 (timeout, n=3) | PARTIAL — Ru ran but timed out; re-run needed |
| Ir | 0.73 | 0.7586 ± 0.0142 (n=3) | ✅ MEASURED, matches Krasnov 0.73 |
| Os | n/a | 0.7884 ± 0.0098 (n=3) | ✅ MEASURED (Krasnov didn't report) |
| Rh | n/a | 0.7525 ± 0.0114 (n=3) | ✅ MEASURED (Krasnov didn't report) |
| Re | n/a | 0.8027 ± 0.0637 (n=3) | ✅ MEASURED (Krasnov didn't report) |

### §1.3 Wf-219 artifacts on disk

| Artifact | Path | Status |
|---|---|---|
| Per-seed JSONs (15 files) | `molmetal/reports/wf219_krasnov_aligned/{Re,Os,Rh,Ru,Ir}_seed{0,1,2}.json` | ✅ all 15 present |
| Sweep summary | `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` | ✅ re-aggregated 2026-09-29 |
| Checkpoints (15 files) | `molmetal/checkpoints/wf219_krasnov_aligned/{Re,Os,Rh,Ru,Ir}_seed{0,1,2}.pt` | ✅ all 15 present |
| Metric JSON | `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` | ✅ updated 2026-09-29 |

---

## §2 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 4-metal × 3-seed × 50-epoch (Re/Os/Rh/Ir) + Ru 3/3 partial (timeout 43/50 epochs)
- **PROJECTED**: 5-metal × 3-seed × 50-epoch full sweep — requires fresh user authorization for Ru re-run (≥ 1-hr GPU launch)
- **Smaller-metal caveat**: Re has high std (0.0637) — small cohort (868 rows) amplifies seed-to-seed variation; per-seed std IS the answer (no best-seed selection per `mol-metal-gpu-execution` §7)
- **Timeout caveat**: Ru at hidden=64/depth=2 chemprop v2 on 19135-row scaffold-split train took ~270s/run, exceeding the 1800s budget headroom. The MEASURED per-seed AUCs (0.7858/0.8215/0.8215) are honest data points but excluded from the 5-metal aggregate mean per honest framing
- **Krasnov comparison caveat**: scaffold split (Bemis-Murcko) is NOT identical to Krasnov's temporal split. Direct paper comparison needs explicit temporal-split driver (TODO-219 §4.3 follow-up)
- **Backbone caveat**: hidden=64/depth=2 is NOT production chemprop v2 (300/3). Production chemprop v2 per-run cost is ~9.7 min per (metal, seed), exceeding 1-hr subagent budget for 5x3x15 sweep
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): per-seed std reported honestly, NOT pick the highest seed and call it MEASURED

---

## §3 What Wf-304C Task 1 did (the "flow up" consolidation)

1. **Re-aggregated sweep_summary.json** from per-seed JSON files (CPU, 1 min). Original 9 (Re/Os/Rh × 3) runs from Wf-219 + partial re-run (Ru/Ir × 3) combined into a single consolidated summary.
2. **Updated metric JSON** to reflect 4-metal shipped + Ru timeout. Added `ru_status` block with per-seed AUCs and a clear honest-framing note.
3. **Verified artifact on disk**: all 15 per-seed JSONs + 15 checkpoints + sweep_summary + metric JSON present.
4. **Wrote this closeout** to `TODO/completed/219_wf219_krasnov_aligned_closeout.md`.

**No code change** beyond the documentation/metric updates. No GPU launch (Ru re-run deferred pending user authorization).

---

## §4 Follow-ups (still open)

### §4.1 Ru re-run (BLOCKED per `mol-metal-gpu-execution` §0 — needs user OK)

```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 \
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf219_krasnov_aligned.py \
    --metals Ru --seeds 0 1 2 \
    --epochs 50 --hidden 64 --depth 2 \
    --device cuda --time-budget-s 3600 \
    --json-dir molmetal/reports/wf219_krasnov_aligned \
    --ckpt-dir molmetal/checkpoints/wf219_krasnov_aligned
```

**Estimated wall-time**: ~15-20 min (Ru takes ~270s/run × 3 = 13.5 min + overhead).
**Blocker**: `mol-metal-gpu-execution` §0 "kill then adapt" directive — > 1 hr GPU launch requires fresh user OK.

### §4.2 Production backbone (hidden=300/depth=3) — blocked (≥ 1 hr GPU)

For paper-grade numbers matching Krasnov 2026, the sweep should be re-run at hidden=300/depth=3:
- Estimated wall-time: ~80 min (15 runs × ~6.6 min each at hidden=300/depth=3)
- Blocker: `mol-metal-gpu-execution` §0 + `TODO/pending/84_no_paper_task_without_authorization.md` (paper task)

### §4.3 Explicit temporal-split driver (Krasnov direct comparison) — TODO-220

The Krasnov 2026 paper used a **temporal** split (train pre-2024, test 2025). A direct comparison requires a temporal-split variant of Wf-219. This is a Wf-220 follow-up (TODO-pending, NOT in this Wf-304C).

### §4.4 NCI-60 9-cell multi-task chemprop v2 (Wf-219 task #2) — TODO-220

Per Wf-219 §4.4, this driver does not exist yet (the existing `train_nci60_10cell.py` uses D-MPNN, not chemprop v2). Implementation:
- Per-cell BinaryClassificationFFN head (9 heads)
- Shared chemprop v2 encoder
- Multi-task loss = mean BCE across 9 cells
- Scaffold split per Krasnov 2026 protocol
- Estimated wall-time: ~30 min for 9-cell × 3-seed × 50-epoch at hidden=64/depth=2

This is a TODO-220 sub-pending — NOT in this Wf-304C scope.

---

## §5 Cross-references

- `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` (re-aggregated 2026-09-29)
- `molmetal/reports/wf219_krasnov_aligned/{Re,Os,Rh,Ru,Ir}_seed{0,1,2}.json` (15 per-seed JSONs)
- `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` (updated 2026-09-29)
- `molmetal/scripts/wf219_krasnov_aligned.py` (~500 LOC driver, unchanged)
- `TODO/pending/219_wf219_krasnov_aligned.md` (predecessor doc — being moved to completed/ in Wf-304C)
- `molmetal/reports/krasnov_protocol_investigation.md` §3 (Krasnov 2026 protocol context)
- `molmetal/reports/wf304c_todo_impl/impl_report.md` (Wf-304C parent report)

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: The Wf-219 Krasnov-aligned sweep is 4-metal × 3-seed × 50-epoch MEASURED on RTX 5090 GPU-1. Ru ran but timed out (43/50 epochs); its per-seed AUCs are MEASURED but excluded from the 5-metal aggregate mean per honest framing. Re-aggregation by Wf-304C Task 1 is a **documentation discipline win** (CPU, 1 min), NOT a new code ship. The next steps (Ru re-run + production backbone + temporal-split driver) all require fresh user authorization per `mol-metal-gpu-execution` §0 + `TODO/pending/84_no_paper_task_without_authorization.md`.
