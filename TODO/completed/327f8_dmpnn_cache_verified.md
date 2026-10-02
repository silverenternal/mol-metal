# Wf-327-F8 — D-MPNN Featurization Cache (Wf-326F Follow-up, VERIFIED)

**Owner:** Hugo Lee
**Date:** 2026-09-29
**Verdict:** **VERIFIED** — cache already shipped as Wf-328-W10; Wf-326F
target met via earlier ship. **4/4 tests PASS** (1.95s for the original
Wf-328-W10 suite; 13.65s for the Wf-327-F8 audit mega-bench).

---

## §0 Status check (per task)

> 1. Verify Wf-326F status: read TODO/completed/326f_dmpnn_featurization_cache.md

**Finding:** No `TODO/completed/326f_*.md` exists. Wf-326F never landed as a
standalone TODO file. However, the underlying engineering **was already
shipped** under the Wf-328 umbrella as `Wf-328-W10`:

| Artefact | Path | Status |
|---|---|---|
| Cache implementation | `molmetal/models/dmpnn_cache.py::DMPNNFeaturizationCache` (436 LOC) | SHIPPED (Wf-328-W10) |
| Original tests | `molmetal/tests/test_wf328w10_dmpnn_cache.py` (4 tests) | 4/4 PASS |
| Wf-327-F8 audit tests | `molmetal/tests/test_wf327f8_dmpnn_cache.py` (4 tests) | 4/4 PASS |
| Smoke runner (referenced in docstring) | `molmetal/scripts/smoke_wf328w10_dmpnn_cache.py` | REFERENCED (not yet shipped as a file) |

> 2. If Wf-326F already implemented: confirm + skip; otherwise execute:

**Confirmed.** Wf-326F cache is implemented (shipped as Wf-328-W10). Skip
the implementation step; run the Wf-327-F8 audit tests to lock in the
behavioural contract for future regressions.

---

## §1 Test results (Wf-327-F8 audit suite)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
      -m pytest molmetal/tests/test_wf327f8_dmpnn_cache.py -v

molmetal/tests/test_wf327f8_dmpnn_cache.py::test_wf327f8_cache_hit_miss PASSED         [ 25%]
molmetal/tests/test_wf327f8_dmpnn_cache.py::test_wf327f8_per_epoch_reset PASSED        [ 50%]
molmetal/tests/test_wf327f8_dmpnn_cache.py::test_wf327f8_integration_dmpnn_trainer PASSED [ 75%]
molmetal/tests/test_wf327f8_dmpnn_cache.py::test_wf327f8_speedup_megabench PASSED      [100%]

============================== 4 passed in 13.65s ===============================
```

Original Wf-328-W10 suite also passes:
```
molmetal/tests/test_wf328w10_dmpnn_cache.py .... 4 passed in 1.95s
```

---

## §2 Megabench measurements (Wf-327-F8 §4)

The Wf-327-F8 `test_wf327f8_speedup_megabench` test simulates the
production G6 workload shape at small scale:

| Param | Value |
|---|---|
| Metal cohorts | 5 (5 × 50 = 250 SMILES) |
| Seeds | 5 |
| Epochs | 10 (scaled down from 50 for test runtime) |
| Batch size | 16 |
| SMILES tier-1 cache | 10 unique SMILES (round-robin from FIXTURE_SMILES) |
| Batch tier-2 cache | 100 unique batch keys |

**MEASURED throughput (CPU-only featurization wall time):**

| Path | Wall (ms) |
|---:|---:|
| No cache (re-featurise every epoch) | 5894.3 ms |
| Cached (tier-1 SMILES + tier-2 batch) | 81.8 ms |
| **Speedup ratio (CPU-only micro-bench)** | **72.06×** |

**MEASURED hit rates:**
- Tier-2 (batch) hit rate: **0.9000** (90%)
- Tier-1 (SMILES) hit rate: **0.9992** (99.92%)

**Result saved to:** `molmetal/reports/wf327f8_dmpnn_cache/result.json`.

---

## §3 Honest framing (per party-twenty-spirit §3.5)

> Wf-326A: featurization is MINOR (CPU collate per batch ~7ms vs
> forward/backward ~2500ms). Cache savings PROJECTED 2-5% on the G6
> full sweep.

The 72× CPU-only speedup is **misleading**. In production training,
the forward+backward pass (~2500ms) dominates featurization (~7ms).
The cache eliminates the 7ms but cannot accelerate the 2500ms.
**Real-world wall-time savings are projected at 2-5% on the G6 full
sweep** (50 epochs × 75 batches × 7ms ≈ 26s saved per (metal, seed)).

The dominant D-MPNN bottleneck — the per-molecule Python loop in
`DirectedMPNN.forward` (line 326) — is **NOT** addressed by this
cache. That fix is **Wf-328-W7 PyG port**, which eliminates the
Python loop and is the actual 5-15× ROI win (per Wf-326G §5 P0 #1).

This Wf-327-F8 audit is therefore correctly framed as a **MODEST,
incremental** optimisation — not a flagship change.

---

## §4 Pre-flight + wave-budget posture

* **Pre-flight v2 (TODO-207):** N/A — this is a CPU-only audit /
  test-creation Wf, not a retrain / panel / ablation launch.
  Per AGENTS.md §4 the pre-flight gate is required "before any
  retrain / panel / ablation launch." No GPU launches are scheduled.
* **Wave-budget tracker:** N/A — same reason. No G6 contention
  risk from this Wf.

---

## §5 Summary

* Wf-326F target = "D-MPNN featurization cache" — **already shipped
  as Wf-328-W10**.
* Wf-327-F8 audit = **4/4 NEW tests pass**, megabench measures
  72× CPU-only speedup with 0.90 batch hit rate and 0.9992 SMILES
  hit rate on the 5×5×10 simulated workload.
* Real-world wall-time savings on the G6 full sweep: **2-5%
  PROJECTED** (not MEASURED on G6 — would require G6 launch).
* The dominant D-MPNN bottleneck fix is **Wf-328-W7 (PyG port)**,
  not this cache.
* No paper claims, no GPU launches, no TODO-84 paper touches.
* All numbers cited trace to
  `molmetal/reports/wf327f8_dmpnn_cache/result.json` (MEASURED)
  or `TODO/completed/326g_missed_optimizations_plan.md` §3
  (PROJECTED).

---

**END OF Wf-327-F8 TODO** — Owner Hugo Lee; cross-ref
`party-twenty-spirit §3.5`; last update 2026-09-29.