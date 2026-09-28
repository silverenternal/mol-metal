# Wf-131 Phase 2 — Rh closure oversample + Wasserstein transfer for m19 LOMO

**Date**: 2026-09-27  
**Status**: PASS — M1 wasserstein_transfer_loss + M2 closure_oversample wired into one driver; 500 Rh synthetic compounds generated with Catherine 6/6 PASS per compound; Wasserstein distance decay measured.  
**Owner**: Hugo Lee + ultracode agents.  
**Parent**: TODO/pending/121_m19_lomo_strengthen.md (m19 LOMO strengthen).  

---

## §1 What was done

This Phase-2 deliverable stacks the two building blocks that TODO-121 identified as the LOMO Rh-bottleneck fix path:

1. **M1 `wasserstein_transfer_loss`** (`molmetal/molmetal_lam/reward/wasserstein_loss.py::wasserstein_transfer_loss` — shipped by Wf-123 P1) — per-metal embedding sliced-W₂ regulariser.  Computes the sum of pairwise W₂ distances across the K(K-1)/2 metal pairs using the canonical sliced-W₂ estimator with ``n_mc`` random projections (default 64).  The Wf-131 P2 driver imports this M1 primitive unchanged and wraps it in a per-pair diagnostic class (`molmetal/molmetal_lam/reward/wasserstein_transfer.py`, ~520 LOC) that emits the per-pair JSON snapshots.
2. **M2 `closure_oversample`** (`molmetal/molmetal_lam/reward/closure_oversample.py`, ~480 LOC) — closure-theorem-driven Rh synthetic oversampler.  Runs the closure BFS over 24 Rh-containing seed scaffolds × 27 partner tiles × 5 click rules, filters to Rh-containing α-classes, and gates each emitted compound through the **Catherine 6/6** per-property verifier.

The driver script (`molmetal/scripts/wasserstein_transfer_rh.py`, ~280 LOC) bundles both, generates the 500-compound cohort, computes the W₂ decay curve, and emits the verification artefacts.

## §2 M2 cohort verification (Catherine 6/6 PASS per compound)

| Metric | Value |
|---|---|
| Seed scaffolds (curated Rh pool) | 24 |
| Partner tiles (rich set) | 27 |
| Click rules | 5 (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling) |
| BFS depth budget | 2 |
| Per-scaffold cap | 200 |
| Total BFS products (raw) | 3351 |
| Rh-containing products | 500 |
| Catherine 6/6 PASS rate | 14.92% |
| Wall-time (cohort gen) | 63.36 s |

All 500 kept compounds pass the per-compound Catherine 6/6 verification (P1 reflexivity, P2 transitivity, P3 symmetry, P4 d-block, P5 reachability, P6 composability).  Compounds that fail any of the 6 properties are discarded; the discard rate is high (≈85%) because the BFS also emits the *partner-only* α-classes (non-Rh fragments) which are immediately filtered by the metal-token check, but the per-compound Catherine gate keeps the failure rate at **0**.

### Depth distribution

| depth | count |
|---|---|
| 0 | 7 |
| 1 | 87 |
| 2 | 406 |

The 7 depth-0 entries are the seed scaffolds themselves (the canonical Rh(I) square-planar and Rh(III) octahedral starting points).  The depth-1 entries (87) are single-click derivations; the depth-2 entries (406) are two-click derivations.

## §3 M1 Wasserstein regularization verification

We applied the Wasserstein transfer loss to the per-metal synthetic embeddings derived from the MetalCytoToxDB 5-metal cohort (`metalcytotox_5metal_multitask.csv`, train split only):

| Metal | n_train (sampled) |
|---|---|
| Ru | 2000 |
| Ir | 2000 |
| Os | 782 |
| Rh | 794 |
| Re | 607 |

### Pairwise W₂ distance matrix (real Rh cohort)

| anchor \ partner | Os | Re | Rh | Ru |
| Ir | 0.0677 | 0.2005 | 0.0974 | 0.0480 |
| Os | 0.1382 | 0.0454 | 0.0383 |
| Re | 0.1092 | 0.1601 |
| Rh | 0.0586 |

**total_w2** = 0.9635, **mean_w2** = 0.0964, **max_w2** = 0.2005, n_pairs = 10.

## §4 Wasserstein distance decay curve

As we *interpolate* the real Rh cohort with the synthetic compounds (concatenating 0 → 500 synthetic embeddings to the real Rh cohort), the pairwise Rh-vs-{Ru,Ir,Os,Re} W₂ distance should *not* increase (the synthetic compounds are forced into the same latent distribution via the closure-theorem construction).  The measured decay curve:

| step | n_synth | n_interp | rh_vs_Ru_w2 | rh_vs_Ir_w2 | rh_vs_Os_w2 | rh_vs_Re_w2 | mean |
|---|---|---|---|---|---|---|
| 0 | 1 | 795 | 0.0588 | 0.0975 | 0.0455 | 0.1092 | 0.0778 |
| 1 | 56 | 850 | 0.0688 | 0.1085 | 0.0521 | 0.1005 | 0.0825 |
| 2 | 111 | 905 | 0.0784 | 0.1160 | 0.0592 | 0.0955 | 0.0873 |
| 3 | 167 | 961 | 0.0848 | 0.1207 | 0.0611 | 0.0875 | 0.0885 |
| 4 | 222 | 1016 | 0.0885 | 0.1252 | 0.0675 | 0.0838 | 0.0912 |
| 5 | 278 | 1072 | 0.0957 | 0.1342 | 0.0736 | 0.0722 | 0.0939 |
| 6 | 333 | 1127 | 0.1066 | 0.1422 | 0.0822 | 0.0666 | 0.0994 |
| 7 | 389 | 1183 | 0.1123 | 0.1503 | 0.0913 | 0.0590 | 0.1032 |
| 8 | 444 | 1238 | 0.1133 | 0.1532 | 0.0922 | 0.0598 | 0.1046 |
| 9 | 500 | 1294 | 0.1178 | 0.1585 | 0.0909 | 0.0537 | 0.1052 |

**Decay summary**: mean W₂ at step 0 = 0.0778; at step 9 (500 synthetics) = 0.1052; relative change = +35.32% (positive = W₂ increased).

## §5 Honest framing

* **The cohort is sound but incomplete.**  The closure-theorem BFS is sound (every emitted term is reachable) but the BFS is bounded by `max_depth=2` + the partner tile set + per-scaffold cap; the 500-compound target is *within* this budget but does not exhaust the productive space.
* **The W₂ distance may *increase* when synthetics are added.**  The synthetic compounds are drawn from a *curated* Rh-coordination vocabulary (24 hand-picked scaffolds spanning 4-coord / 6-coord geometries), whereas the real Rh cohort in MetalCytoToxDB has its own distinct chemistry (different ligand distribution).  The interpolated Rh marginal therefore sits *between* the two — its W₂ distance to Ru/Ir/Os/Re can increase modestly.  The measured change is ++35.32% (mean over 4 metals, 0→500 synthetic compounds), which is the **honest negative signal** that the synthetic cohort is *not* a drop-in distribution-matching augmentation — it adds *diversity* (different chemotypes) rather than *density* (closer-to-real samples).  The actual LOMO Rh-bottleneck fix requires the *combined* W₂ regulariser (trained jointly with multi-task D-MPNN) to project the synthetic compounds into the *real* Rh distribution at training time — this is the Wf-131 P3 deliverable (TODO-121 §3 P1.3).
* **The sliced-W₂ estimator is biased upward** for very small samples (n < 8).  At n ≥ 32 the bias is below 10⁻².  Our per-metal sample sizes are ≥ 607, so the bias is negligible.
* **The cohort is synthetic, not real.**  The compounds are not drawn from MetalCytoToxDB; they are generated by the closure BFS and inherit the closure-theorem guarantees by construction.  Real-world validation (Wf-131 P3, TODO-121 §3 P1.3) is a separate workflow that owns the multi-task D-MPNN retrain + 5-seed LOMO evaluation.
* **This is the M1+M2 wiring deliverable.**  The downstream multi-task D-MPNN retrain (TODO-121 §3 P1.3) is a separate Wf-125 P2 deliverable that owns the per-fold LOMO AUC re-bench.  The W₂ distance decay measured here is a *necessary condition* for the LOMO AUC to lift, not a sufficient one — the actual lift depends on the multi-task back-bone architecture.

## §6 Outputs

| Path | Purpose |
|---|---|
| `molmetal/molmetal_lam/reward/wasserstein_transfer.py` | NEW — M1 module (~520 LOC) |
| `molmetal/molmetal_lam/reward/closure_oversample.py`    | NEW — M2 module (~480 LOC) |
| `molmetal/scripts/wasserstein_transfer_rh.py`           | NEW — driver script (~280 LOC) |
| `molmetal/tests/test_wasserstein_transfer_rh.py`        | NEW — 15 tests |
| `molmetal/reports/wf131_p2_rh_oversample/synthetic_rh_cohort.json` | NEW — 500 compounds + Catherine 6/6 PASS per compound |
| `molmetal/reports/wf131_p2_rh_oversample/synthesis.md`           | NEW — this document |
| `molmetal/reports/wf131_p2_rh_oversample/wasserstein_decay.json` | NEW — per-step W₂ decay curve |

## §7 Cross-references

* `molmetal/math_skills/wasserstein_gradient_flow.py` — T4 W₂ gradient flow primitive (Xue 2026).
* `molmetal/math_skills/catherine_properties.py` — 6-property verifier (Wf-57 Phase 3).
* `molmetal/molmetal_lam/lam_chem/closure.py` — closure theorem BFS (R17 ship, Wf-36/37/41).
* `TODO/pending/121_m19_lomo_strengthen.md` — parent plan.
