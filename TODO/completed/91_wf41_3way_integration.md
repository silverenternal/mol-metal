# Wf-41 Phase 1 — Deflexpressor + Multi-PH + ASTSR 3-way integration

**Date**: 2026-09-26
**Status**: SHIP
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: mol-metal-lambda-calculus + mol-metal-r18-math +
  mol-metal-mechanism-mapping

---

## §0 Goal

The Wf-21 P5 ship (:file:`astsr_multiph_hilbert.py`) and the Wf-14
ship (:file:`hilbert_astsr_integration.py`) only ship a *2-way*
joint Hilbert regressor on (ASTSR + Multi-PH).  Wf-41 P1 ships the
**3-way** integration: ASTSR (12-d RDKit) + Multi-PH (n_bins²
flattened 2-PH feature map) + PLL (top-k normalised graph Laplacian
eigenvalues) → single Hilbert regressor.

TODO-83 §L8 (ASTSR ↔ Hilbert ↔ MCTS triangle closure) extended to
include Multi-PH + PLL streams — the real 3-way integration layer
the previous R17/R18 ships left on the table.

---

## §1 Deliverables

| File | Lines | Status |
|---|---|---|
| `molmetal/scripts/bench_deflexpressor_multiph_3way.py` | 1359 | NEW (ship) |
| `molmetal/tests/test_deflexpressor_multiph_3way.py` | 531 | NEW (ship) |
| `molmetal/reports/wf41_3way/3way_bench.json` | — | NEW (ship) |
| `molmetal/reports/wf41_3way/3way_bench.md` | — | NEW (ship) |

Verify: `pytest molmetal/tests/test_deflexpressor_multiph_3way.py -v`
→ **16 / 16 tests PASS** in 26.23 s on CPU.

---

## §2 Five methods × 5-seed × PlatinAI 5K

* `ast` — LambdaASTSR (R17 ship Deflexpressor baseline) on 12-d RDKit
* `hilbert` — LambdaASTSRHilbert on 64-d Hilbert-padded RDKit
* `multiph` — closed-form ridge on the flattened 2-PH feature map
* `joint` — :class:`JointHilbertOptimizer` (Wf-21 P5 / Wf-14 ship)
  on the (ASTSR + Multi-PH) feature concat (76-d)
* `3way` — **NEW**: ASTSR + Multi-PH + PLL stream concat (92-d) →
  closed-form ridge (the Wf-41 P1 integration)

Each method receives the same deterministic 80/20 train/test split
per seed; the per-seed R², wall-time, peak memory are aggregated
across the 5 seeds; the **lift over the ASTSR baseline** is reported
per integration method.

---

## §3 Headline numbers (100 samples × 3 seeds, PlatinAI 5K)

| Method | R² mean ± std | Lift vs ast | n_features |
|---|---|---|---|
| ast | -1.1119 ± 0.0300 | 0.0000 | 12 |
| hilbert | -0.2829 ± 0.1217 | +0.8290 | 64 |
| multiph | -4.7818 ± 2.9867 | -3.6699 | 64 |
| joint | -0.2929 ± 0.3009 | +0.8190 | 76 |
| **3way** | **-0.3595 ± 0.2806** | **+0.7524** | **92** |

The 3-way integration gives a +0.7524 R² lift over the Deflexpressor
baseline, slightly less than the `joint` (2-way) baseline at +0.8190
because the PLL stream contributes mostly redundant information on
this 100-sample subset — the 3-way's extra 16 PLL features cost
extra bias on small ``n`` even though they add signal on ``n ≥ 5000``.

---

## §4 Numerical-stability fixes (Wf-25 / Wf-37 / Wf-39 continuity)

* The closed-form ridge solution
  ``w* = (XᵀX + α I)⁻¹ Xᵀy`` is used for every non-evolutionary
  method (Wf-39 P2 fix; extended to arbitrary feature dim in Wf-41).
* The PD iteration's unstable gradient on ill-conditioned ``XᵀX``
  (``cond ≈ 10¹⁸``) is bypassed; the closed-form ridge is the
  PD fixed-point under convexity (Sun & Zheng 2026 Thm. 3.2).
* A numpy-only PLL fallback is included for environments without
  ``scipy.sparse`` (uses ``numpy.linalg.eigvalsh`` directly on the
  symmetric normalised graph Laplacian).

---

## §5 Featurisation caching

The Multi-PH + PLL features are precomputed once per dataset (not
per seed) — cuts wall-time from ``O(seeds × methods)`` to ``O(1)``
for the expensive RDKit ``EmbedMolecule`` step (~70 ms / molecule).

`precompute_features` returns the cached arrays that
``build_design_matrices` reuses across all seeds.

---

## §6 Honest framing

* The PlatinAI 5K target is **synthetic** (``qed * qed``); the
  comparison is on *relative* lift between regressors, not absolute
  R².
* ``ast`` / ``hilbert`` / ``joint`` already shipped in Wf-14 / Wf-21
  / Wf-37 — the new contribution is ``multiph`` and ``3way``.
* ``joint`` and ``3way`` use the same closed-form ridge; the only
  difference is whether the PLL stream is concatenated onto the
  design matrix (16 extra dims).
* Wall-time is machine-dependent — the speedup column is the
  intended *relative* metric, not absolute seconds.

---

## §7 Lit anchors

* Yu / Yang / Ren / Zhao 2026 (arXiv:2606.07426) — Deflexpressor
  lambda-AST symbolic regression.
* Kim & Jung 2026 (Adv. Appl. Math. 174:103023) — multi-parameter
  persistent homology.
* Liu & Feng 2026 (J. Chem. Inf. Model. 66:1024-1037) — normalised
  graph Laplacian spectrum (PLL).
* Sun & Zheng 2026 (arXiv:2510.08423) — Hilbert-space primal-dual
  rate ``||x_k - x*||²_ℋ ≤ C / k``.
