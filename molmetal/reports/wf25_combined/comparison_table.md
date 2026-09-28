# Wf-25 Phase 3 — Hilbert + Multi-PH joint-feature real evaluation

**Dataset:** PlatinAI 5K + MetalCytoToxDB Ru subset (n=500)
**Folds:** 5  |  **Join dim:** 264 (astr=8 + mph=256)
**Multi-PH bins:** 16  |  **Hilbert iters:** 60  |  **Hilbert lr:** 0.05

**Parse stats:** parsed=500, failed=0, mph_parsed=500, mph_failed=0, mph_too_large=0, target_rows=500

## Side-by-side comparison

| Backend | R² mean ± std | Wall time (s) | Speedup |
|---|---|---|---|
| Joint Hilbert (Wf-21 phase-5) | -0.3736 ± 0.0808 | 0.0050 | 1.00x (ref) |
| Separate (astr + mph sum) | -0.4250 ± 0.1326 | 0.0059 | 1.17x |

## Per-fold detail

| Fold | n_train | n_test | Joint R² | Joint wall (s) | Separate R² | Separate wall (s) |
|---|---|---|---|---|---|---|
| 0 | 400 | 100 | -0.3197 | 0.0058 | -0.3883 | 0.0058 |
| 1 | 400 | 100 | -0.2875 | 0.0052 | -0.2428 | 0.0053 |
| 2 | 400 | 100 | -0.4339 | 0.0042 | -0.5595 | 0.0056 |
| 3 | 400 | 100 | -0.3252 | 0.0042 | -0.3414 | 0.0065 |
| 4 | 400 | 100 | -0.5016 | 0.0058 | -0.5931 | 0.0064 |

## Honest framing

* ASTSR features here are 8 RDKit descriptors — the true Deflexpressor AST bank (R17 ship) operates on the AST search tree, which requires a pre-fitted AST bank unavailable in this offline CPU script. The 8-d descriptor set is the *honest* ASTSR-proxy used by Wf-25 Phase 1.
* Multi-PH features are deterministic (no learned parameters); only the downstream Hilbert regressor is trained.
* Activity is binary (0/1) — the target is ``activity + N(0, 0.05)``, the regressor sees a noisy continuous signal, and R² measures how much of that binary signal the joint Hilbert regressor recovers.
* The separate branch fits two ``LambdaASTSRHilbert`` regressors (one per stream) and sums predictions at inference; the joint branch uses the Wf-21 phase-5 ``JointHilbertOptimizer``.
* Wall-time is machine-dependent; the speedup column is the intended *relative* metric.
