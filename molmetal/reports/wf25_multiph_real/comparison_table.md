# Wf-25 Phase 2 — Multi-PH real end-to-end evaluation

**Dataset:** PlatinAI 5K + MetalCytoToxDB 5-metal multi-task (n=499)
**Folds:** 5  |  **Multi-PH bins:** 16 (dim=256)  |  **RDKit dim:** 8
**Metals:** Ru, Ir, Rh, Os, Re

**Parse stats:** parsed=500, failed=0, mph_parsed=499, mph_failed=1, mph_too_large=0, target_rows=499, ru_subset=362

## Side-by-side comparison

| Backend | Acc mean ± std | Macro-F1 mean ± std | Train wall (s) |
|---|---|---|---|
| Multi-PH (W2.3) | 0.639 ± 0.068 | 0.362 ± 0.119 | 0.802 |
| RDKit descriptors (baseline) | 0.595 ± 0.149 | 0.168 ± 0.019 | 0.612 |

## Per-fold detail

| Fold | n_train | n_test | RDKit acc | RDKit F1 | Multi-PH acc | Multi-PH F1 |
|---|---|---|---|---|---|---|
| 0 | 399 | 100 | 0.500 | 0.201 | 0.670 | 0.263 |
| 1 | 399 | 100 | 0.350 | 0.144 | 0.700 | 0.542 |
| 2 | 399 | 100 | 0.680 | 0.162 | 0.520 | 0.229 |
| 3 | 399 | 100 | 0.750 | 0.171 | 0.610 | 0.319 |
| 4 | 400 | 99 | 0.697 | 0.164 | 0.697 | 0.459 |

## Honest framing

* The Multi-PH branch uses a flattened ``(n_bins, n_bins)`` feature map — no learnable parameters in the feature extractor; the only trained parameters are in the downstream MLP head.
* The RDKit-descriptor baseline uses 8 RDKit descriptors (parallel to the Wf-25 Phase 1 schema); both branches use the same MLP topology and same training schedule.
* The metal-coordination classifier targets 5 classes (``Ru, Ir, Rh, Os, Re``); the Ru subset dominates the dataset (~71 %), so macro-F1 surfaces minority-class behaviour more honestly than accuracy.
* Wall-time is **machine-dependent**; the comparison is intended for *relative* speedup between the two branches on the same CPU.
