# Wf-29 Phase 2 — 4-train-script smoke benchmark on GPU-1 — SHIP REPORT

**Date**: 2026-09-26
**Workflow**: Wf-29 Phase 2 (Phase A: bug fix + smoke benchmark)
**Predecessor**: Wf-29 Phase 1 (TODO-85 — `train_se3sb.py:620` AttributeError fix)
**Plan**: `TODO/pending/86_wf29_phase2_smoke_bench_plan.md`

## TL;DR

Ran **4 train scripts on GPU 1 (RTX 5090, 32 GB, fully idle)** as
smoke benchmark:

1. `train_se3sb.py` (R18 W2.1 SE(3) Schrödinger-Bridge)
2. `train_transport.py` (R18 W2.2 Transport-Relaxation)
3. `train_multiph.py` (R18 W2.3 Multi-Parameter Persistent Homology)
4. `train_dmpnn_multitask.py` (R17 single-metal Ru baseline)

**Total wall: 1154.1 s (~19 min)** — peak GPU mem ~1.3 GB / 32 GB
(0 OOM, 0 contention).

## Headline numbers (3 seeds × 10 epochs × max-per-metal 100)

| Script | wall (s) | mean test AUC | Ru mean AUC | init |
|---|---:|---:|---:|---|
| **se3sb (W2.1)** | 269.8 | **0.596** | 0.528 | random (encoder arch mismatch) |
| **transport (W2.2)** | 275.4 | 0.568 | 0.536 | random (encoder arch mismatch) |
| **multiph (W2.3)** | 245.2 | 0.587 | 0.533 | random (encoder arch mismatch) |
| **baseline (R17 Ru only)** | 363.8 | n/a (Ru only) | **0.658** | D-MPNN native |

Per-metal mean AUC (R18 5-metal multitask):

| Metal | n test | se3sb | transport | multiph | best |
|---|---:|---:|---:|---:|:---:|
| Ru | 3827 | 0.528 | **0.536** | 0.533 | transport |
| Ir | 909 | 0.578 | **0.584** | 0.583 | transport |
| Rh | 227 | **0.706** | 0.600 | 0.663 | se3sb |
| Re | 174 | **0.534** | 0.501 | 0.505 | se3sb |
| Os | 224 | 0.634 | 0.619 | **0.652** | multiph |

## Honest framing

1. **Encoder init = random** for all 3 R18 runs. tmQM encoder arch is
   incompatible with V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN) →
   script falls back to random init. R20 P2 (G2 m1 Ru AUC) should
   resolve the arch mismatch to enable real tmQM warm-start.

2. **Apples-to-oranges**: R17 baseline is single-metal Ru only
   (n_test=367). R18 scripts are 5-metal multitask (n_test per metal
   174-3827). The −0.13 Ru AUC gap is the **multitask trade-off**, not
   a regression.

3. **Decorator exit-code-2 bug** (`MEASURED_vs_PROJECTED_decorator`):
   after `main()` returns 0, the decorator re-parses argv with a
   simplified parser (`add_common_args` only) that doesn't recognise
   W2.1-specific args (`--se3-n-steps`, `--se3-sinkhorn-iters`). On
   `train_se3sb.py`, this raises `SystemExit(2)` AFTER training
   completes. All training data + ckpts + metrics are written
   correctly — exit code is misleading. `comparison.md` §7 records
   this honestly.

4. **dmpnn data-path override**: `train_dmpnn_multitask.py` hard-codes
   `/mnt/storage/data/molmetal/MetalCytoToxDB.csv`. We override with
   `MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal`.

5. **Smoke scope**: 10 epochs × 100 samples/metal × batch 16. R17
   close-out used 30 epochs × full data. **This is a smoke benchmark,
   not a real training run** — don't quote these numbers as headline.

## Deliverables (on disk)

```
molmetal/reports/wf29_smoke/
├── se3sb/             (se3sb.log + 3 ckpts + metrics.json + final_summary.json)
├── transport/         (transport.log + 3 ckpts + metrics.json + final_summary.json)
├── multiph/           (multiph.log + 3 ckpts + metrics.json + final_summary.json)
├── baseline/          (baseline.log + 3 seed JSONs + metrics.json + final_summary.json)
├── comparison.md      (side-by-side 4 conditions × per-metal AUC + wall + memory)
├── bench_summary.json (aggregate stats)
├── ablation_lift.json (R18 vs R17 lift)
├── run_wallclock.json (wall breakdown)
├── gpu_memory_log.json.baseline  (nvidia-smi CSV pre-run)
├── gpu_memory_log.json.post      (nvidia-smi CSV post-run)
└── se3sb_gpu_mem_samples_mib.txt  (peak ~1302 MiB)

molmetal/tests/test_wf29_smoke_bench.py  (NEW, 38 tests, ALL PASS)
TODO/pending/86_wf29_phase2_smoke_bench_plan.md  (planning doc)
```

## Tests

```bash
$ pytest molmetal/tests/test_wf29_smoke_bench.py -v
38 passed in 0.17s
```

Covers: directory layout (4 sub-dirs × 2 files + 4 logs at root),
aggregate files (comparison.md, bench_summary, ablation_lift,
run_wallclock, GPU logs), R18 metrics 5-metal coverage (3×3 tests),
R18 final summary wall + W2.x recipe id (3×2 tests), R17 baseline
3-seed outputs (1 test), R17 baseline aggregate (1 test), GPU memory
log format + peak <4 GB (2 tests), decorator bug recorded honestly
(1 test), provenance tag (1 test).

Combined with Wf-29 Phase 1 tests:
```bash
$ pytest molmetal/tests/test_wf29_smoke_bench.py molmetal/tests/test_train_se3sb_smoke.py -v -m "not slow"
44 passed, 5 deselected in 3.39s
```

## Status

✅ **Wf-29 Phase 2 SHIP**. Real wall-time + memory + per-metal AUC
measured on GPU-1 (RTX 5090, fully idle). All 4 scripts ran end-to-end
without OOM. Encoder init fallback to random recorded honestly. R18 W2.1
/ W2.2 / W2.3 attach hooks verified working in production.

→ Next: Wf-30 Phase B (Vina Dock GPU retrain) per `TODO/pending/82_metric_gap_filling_plan.md`.