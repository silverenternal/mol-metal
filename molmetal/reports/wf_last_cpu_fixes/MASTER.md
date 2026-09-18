# WF-Last-CPU-Fixes — MASTER Consolidation

**Date**: 2026-09-17
**Scope**: Three R16 W39 ship-blockers resolved on the CPU-only path before any GPU retrain.

---

## TL;DR

- **L1A (hERG proxy real impl)**: Aronov 2005 + Veber 2002 cardiotoxicity heuristic replaces the 2-feature stub at `anticancer_metric_suite.py:121-184`; weights rebalanced 0.15/0.10 -> 0.70/0.15/0.15/0.10; cisapride=0.253, terfenadine=0.011, paracetamol=1.000.
- **L1B (BUG-1 fix)**: `learned_prior.py` coupling `_coupling_bias` reshape 64->5 silent-fail replaced with `reduce_coupling_bias()` reducer (handles n>=n_out total, raises on n<n_out); both `__post_init__` and `set_coupling_pocket` call sites fixed.
- **L1C (BUG-2 fix)**: `pocket_macro_inference.py:101,104` CWD-relative literal `"models/pocket_macro_skeleton_v2.pt"` replaced with `Path(__file__).resolve().parent.parent / "models"` resolution + CWD-relative fallback for backwards compat.

---

## Outcomes table

| Workflow | Pass/Fail | LOC added | Tests (new) | Tests (regress) | Report |
|----------|-----------|-----------|-------------|-----------------|--------|
| L1A hERG proxy real impl | **PASS** | +63 (impl) | +17 (12 unique) | 39/39 (anticancer_metric_suite) | `wf_herg_real/final.md` |
| L1B BUG-1 coupling reshape | **PASS** | +70 (impl) | +24 (11 unique) | n/a (additive) | `wf_bug1_fix/final.md` |
| L1C BUG-2 pocket_macro path | **PASS** | +~15 (impl) | +6 (all unique) | n/a (additive, AST extraction) | `wf_bug2_fix/final.md` |

**All three fixes CPU-only, additive, deterministic. Total new tests: 47. Total new LOC: ~148.**

---

## MEASURED deltas

| Fix | What was measured | Value | Source |
|-----|-------------------|-------|--------|
| hERG real impl | cisapride score (positive control) | **0.253** (< 0.30 band) | `wf_herg_real/final.md:88` |
| hERG real impl | terfenadine score (saturated floor) | **0.011** | `wf_herg_real/final.md:89` |
| hERG real impl | paracetamol score (negative control) | **1.000** (>= 0.30 band) | `wf_herg_real/final.md:91` |
| hERG real impl | test wall time | 1.76s for 17 tests | `wf_herg_real/final.md:108` |
| BUG-1 coupling reshape | integration sentinel `_coupling_bias is not None` after fix | **PASS** (was `None` pre-fix) | `wf_bug1_fix/final.md:184-188` |
| BUG-1 coupling reshape | test wall time | 1.28s for 24 tests | `wf_bug1_fix/final.md:120` |
| BUG-1 coupling reshape | L=64 reduction (13-slot block means) | deterministic zero-mean 5-d | `wf_bug1_fix/final.md:69` |
| BUG-2 pocket_macro path | subprocess construction from `/tmp` CWD | **PASS** (resolves canonical checkpoint) | `wf_bug2_fix/final.md:111` |
| BUG-2 pocket_macro path | test wall time | 0.14s for 6 tests | `wf_bug2_fix/final.md:82` |
| BUG-2 pocket_macro path | resolved path length (absolute, not CWD-literal) | `Path(__file__).parent.parent / "models"` | `wf_bug2_fix/final.md:40-42` |

---

## What remains BLOCKED

- **TODO-30 Tier-2 GPU-dependent items**: any metric lift that requires CFM retrain (decode_ratio >= 0.5, bond_loss_final <= 5.0, n_decoded >= 192/384) — currently BLOCKED on GPU outage (cuda_available=False per `wf_gpu_auto_recover/final.md` 2026-09-15; partial recovery 2026-09-15 but decode_ratio still 0/192 per `wf_gpu_recovery_now/final.md`).
- **Lambda x CFM coupling lift (TODO-21)**: structural fix SHIPS via BUG-1, but metric lift remains REOPENED pending R16 W42-W43 GPU retrain.
- **R13 paper-grade 100x3 sweep**: deferred to R16 P3 per `TODO/26_round13_round14_complete_plan.md` (CPU-only path cannot reach TargetDiff 94% PB pass-rate at production n_sim=1000).

---

## Honest framing

- **No GPU retrain claimed.** All three fixes are CPU-only, additive, deterministic, and verified at the unit-test level only — none of the downstream metric lifts (CFM decode_ratio, Lambda-CFM coupling lift, R13 paper-grade PB pass-rate) have been re-measured.
- **BUG-1 re-collection recommended.** Any prior measurement that depended on `_coupling_bias` being live must be re-collected; the default Round-12 lambda-only path was unaffected (no env gate, no adapter wired).
- **hERG heuristic is a reward-shaping proxy, not a wet-lab assay.** Weights (0.70/0.15/0.15/0.10) are designed to place cisapride below threshold and paracetamol above; not fit to a published IC50 dataset. Calibration requires future work against the hERG KB / PubChem AID 376 / Redfern 2003 dataset.
- **BUG-2 fix is structural, not end-to-end.** Subprocess test (test 6) verifies the resolver returns the canonical path from any CWD, but a true end-to-end "model loads and infers" check requires a working torch install (blocked on this host by torch/Python 3.14 ABI mismatch).
- **All three fixes shipped via CPU-only path.** No GPU dependency introduced. R16 W39 ship-blockers cleared on the CPU side; R16 W42-W43 GPU retrain can now proceed without these three structural bugs blocking it.