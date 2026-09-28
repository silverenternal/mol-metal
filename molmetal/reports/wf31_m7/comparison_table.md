# Wf-31 Phase 3 — Multi-PH feature extraction + diversity bench

**Dataset:** PlatinAI 5K (n=500, parsed Multi-PH=380, parsed diversity=500)
**Multi-PH bins:** 16 (dim_multi_ph=256, dim_single_ph=16)
**Pocket budget:** max_atoms=60, max_radius=5.0, max_coord=4.0

## Expressivity (mean pairwise cosine distance)

| Feature family | Mean 1 - cos(x, y) |
|---|---|
| Multi-PH (2-parameter) | 0.242 |
| Single-PH (1-D Betti marginal) | 0.120 |
| Delta (multi - single) | +0.122 (PASS) |

## Chemical diversity (Tanimoto distance)

| Metric | Value |
|---|---|
| Mean pairwise Tanimoto distance | 0.904 |
| m7 floor (R18 PLAN §3) | 0.750 |
| Meets m7 floor | PASS |

## Wall-time

* Multi-PH extraction: **164.74 s** for 500 molecules (parsed: 380).

## Honest framing

* The **single-PH baseline** is the marginal sum over the coordination-distance axis — a canonical reduction of the 2-PH feature map to a 1-D Betti curve.  It is *not* a separate extraction; it is the same feature map collapsed along one axis.
* The **expressivity** metric is the mean pairwise cosine distance (1 - cos).  It is a coarse proxy for how well the feature family spreads samples in the embedding space; a higher value means more distinguishable features.
* The **diversity** metric is the mean pairwise Tanimoto *distance* on Morgan-ECFP4 fingerprints — a separate chemistry measurement that does not involve the Multi-PH extractor.
* The m7 floor ``DIVERSITY_FLOOR = 0.75`` is the R18 PLAN §3 success criterion; the bench does *not* enforce it as a fail-condition (the bench is descriptive, not gating).
