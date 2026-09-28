# Wf-145 P2 — LOMO 5-metal x 5-seed x 50-epoch full re-bench (v2 driver)

**Date**: 2026-09-27
**Status**: ⚙️ **PARTIAL SHIP — v2 driver + 18/18 tests PASS, full GPU sweep DEFERRED**
**Owner**: Hugo Lee + Claude Code (ultracode)
**Parent**: TODO-135 §5 Phase 2 + Wf-145 P1 BALANCED Rh cohort

---

## §1 What shipped (this workflow run)

| Artifact | Status | Path |
|---|---|---|
| v2 trainer driver (~860 LOC) | ✅ NEW | `molmetal/scripts/train_metal_lomo_v2.py` |
| v2 test suite (18 tests) | ✅ NEW — 18/18 PASS | `molmetal/tests/test_metal_lomo_v2.py` |
| CLI smoke (CPU, 1 metal × 1 seed × 2 epochs) | ✅ PASS | `/tmp/wf145_smoke` (test run) |
| Per-fold output dir | ✅ EMPTY (sweep DEFERRED) | `molmetal/reports/wf145_p2_lomo_v2/{metal}/per_seed_metrics_seed{0..4}.json` |
| Aggregate JSON | ✅ EMPTY | `molmetal/reports/wf145_p2_lomo_v2/aggregate.json` |
| `metrics/by_metric/metal_lomo_auc_v2.json` | ✅ NEW (DEFERRED_PARTIAL_RUN) | `metrics/by_metric/metal_lomo_auc_v2.json` |
| `metrics/by_metric/metal_lomo_auc.json` | ⚙️ Refreshed — append-only entry | `metrics/by_metric/metal_lomo_auc.json` |
| v2 synthesis.md (this file) | ✅ NEW | `molmetal/reports/wf145_p2_lomo_v2/synthesis.md` |

## §2 What was deferred (and why)

The task spec asks for **25 GPU trainings × 50 epochs × D-MPNN-MT** (≈ 4 hours
wall-clock on RTX 5090).  Two blockers:

1. **GPU-1 (RTX 5090) is currently occupied** by PID 649993 (Wf-131 P3
   continuation, started 05:13 with `--metals Ru Ir Os Rh Re --seeds 0 1 2 3 4`),
   running the Wf-125 P1 BUGGY constant-inactive cohort sweep.  It has been
   on Os/Rh/Re since ~06:22 and is expected to finish in 1-3 hours.
2. **Session budget for this subagent**: sub-hour.  Running 25 sequential
   50-epoch trainings would block the next workflow for ~4 hours.

The v2 driver + test suite is SHIPPED so the next session can launch
the sweep without re-implementing the loader / runner / aggregator.

## §3 v2 driver spec — what it does differently from Wf-125 P2

| Aspect | Wf-125 P2 (`train_metal_lomo_wasserstein.py`) | Wf-145 P2 v2 (`train_metal_lomo_v2.py`) |
|---|---|---|
| Cohort | `wf125_p1_rh_oversample/synthetic_rh_cohort.json` (constant-INACTIVE labels) | `wf145_p1_rh_active_cohort/synthetic_rh_pos_neg.json` (BALANCED 250 active + 250 inactive) |
| W₂ target | pairwise per-metal embeddings (5 metal marginals) | real Rh ACTIVE in train pool + per-metal marginals |
| Hilbert L2 guard | opt-in via `--use-l2-guard` | **default ON** (`max_norm=5.0`) |
| ASTR scale clip | opt-in via `--use-astr-scale-guard` | always ON (`[-10, 10]`) |
| Default epochs | 30 | 50 |
| Per-metal head | implicit via per-metal embedding hook | explicit per-metal head ensemble (5 heads, 1 per metal) |

## §4 MEASURED data on disk (reference — from Wf-131 P3, BUGGY cohort)

| Fold | Seeds MEASURED | test_auc (mean ± std) |
|---|---|---|
| Ru | 5/5 (seeds 0-4) | **0.5941 ± 0.0146** (vs R17 0.5706 = **+0.0235** lift) |
| Ir | 3/5 (seeds 0-2) | **0.6209 ± 0.0114** (vs R17 0.6176 = +0.0033 lift) |
| Os | 0/5 | (deferred in Wf-131 P3) |
| Rh | 0/5 | (deferred — bug-stricken fold; expect Rh to regress again with BUGGY cohort) |
| Re | 0/5 | (deferred in Wf-131 P3) |

Wf-131 P3 used the Wf-125 P1 BUGGY constant-inactive cohort — Rh was
0.136 in the 6-epoch smoke (see `metal_lomo_auc_wasserstein.json`).  The
v2 driver replaces that cohort with the Wf-145 P1 BALANCED cohort, but
the v2 sweep itself was not launched in this session (see §2).

## §5 Targets (per task spec, NOT yet MEASURED on v2)

| Target | Value | Source for delta |
|---|---|---|
| Rh LOMO AUC 5-seed mean | ≥ **0.50** | vs R17 0.368 / Wf-131 P2 regression 0.136 |
| Overall LOMO 5-seed mean | ≥ **0.60** | vs R17 baseline 0.5346 |

These remain **PROJECTED** — the v2 sweep is DEFERRED.  Honest framing
per party-twenty-spirit §3.5 below.

## §6 Honest framing (party-twenty-spirit §3.5)

**承认**:
1. **Rh n=60 train is small** — fundamental sample size ceiling; the
   balanced cohort is only 500 SMILES of training-time augmentation, not
   real wet-lab data.
2. **5-seed mean may not be statistically robust** at small n — the
   90 % CI on a 5-seed mean of an AUC is wide.
3. **Multi-task backbone cross-metal share** — signal leak between metals
   via the shared D-MPNN trunk.
4. **Wasserstein target = Rh real ACTIVE distribution** — but the real
   active distribution may not be known well (only 413 real actives in
   MetalCytoToxDB Rh train).
5. **50 epochs may be insufficient** for full convergence.
6. **v2 sweep DEFERRED in this session** — only the driver + tests shipped.

**不假装**:
1. 不假装 Rh AUC ≥ 0.50 guaranteed — 50% chance fail per TODO-135 §5
   honest framing.
2. 不假装 overall LOMO ≥ 0.60 guaranteed — depends on per-metal
   performance.
3. 不假装 balanced labels = real binding — only wet-lab validation.

## §7 Next-session launch instructions

```bash
cd /home/hugo/codes/mol-metal
# Wait for PID 649993 to finish first, then launch on GPU-1:
CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_metal_lomo_v2.py \
    --metals Rh Ru Ir Os Re --seeds 0 1 2 3 4 --epochs 50 \
    --device cuda \
    --out-dir molmetal/reports/wf145_p2_lomo_v2
```

Expected wall-clock: ~4 hours on RTX 5090 (25 trainings × ~10 min each).

## §8 Cross-references

- `TODO/pending/135_*.md` (TODO-135) — parent planning
- `molmetal/reports/wf145_p1_rh_active_cohort/synthesis.md` — Wf-145 P1 PASS
- `molmetal/reports/wf131_p3_lomo_full/Ru/per_seed_metrics.json` — 5-seed Ru fold
- `molmetal/scripts/train_metal_lomo_wasserstein.py` — Wf-125 P2 (predecessor)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` — M1 wasserstein primitive
- `metrics/by_metric/metal_lomo_auc.json` — R17 baseline (canonical)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` — Wf-125 P2 smoke (buggy)
- `TODO/completed/121_m19_lomo_strengthen.md` — closeout for predecessor Wf-131 P3