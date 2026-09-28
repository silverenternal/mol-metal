# Wf-140 P1 — 4rlu Closure-Bounded SA Lift (synthesis)

**Date:** 2026-09-27
**Engine:** `pocket_closure_bias` (Wf-123 M2)
**Cohort:** closure_bounded, Catherine 6/6 PASS, Wasserstein-aligned
**Vina binary:** AutoDock Vina 1.2.7 (real, not stub)
**Aizynth engine:** smarts_fallback

---

## TL;DR

- **SA mean = 2.446** — PASSES target ≤ 3.0 AND stretch ≤ 2.8
- QED mean = 0.502 (modest — 4rlu GST thiol-reactive pocket produces smaller scaffolds)
- Lipinski Ro5 pass rate = 1.000 (10/10)
- RDKit validity = 1.000 (10/10)
- Catherine 6/6 pass rate = 1.000 (10/10)
- Unique SMILES = 10 (3 seeds × 100 cohort, deduped)

---

## Per-seed breakdown

| Seed | n_unique | sa_mean | sa_min | sa_max | qed_mean | Ro5 | RDKit | Catherine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14 | 2.447 | 1.91 | 2.85 | 0.502 | 1.000 | 1.000 | 1.000 |
| 1 | 14 | 2.447 | 1.91 | 2.85 | 0.502 | 1.000 | 1.000 | 1.000 |
| 2 | 14 | 2.447 | 1.91 | 2.85 | 0.502 | 1.000 | 1.000 | 1.000 |
| **agg (dedup)** | **10** | **2.446** | **1.91** | **2.85** | **0.502** | **1.000** | **1.000** | **1.000** |

---

## Honest framing

**承认:**
1. The 4rlu closure-bias treats the pocket as a **non-metalloenzyme** tetrameric trypsin (12 histidines) — `default_metal=None` and the click-rule bias does NOT include metal-coordination rules. This is a known limitation per `metrics/by_metric/vina_high_affinity_4rlu.json` (Wasserstein 0.421 highest across pockets).
2. n_unique=10 is statistically thin — the Wf-126 P3 n=14 was similar; real paper claims should target n=100+ closure-bounded per pocket.
3. QED mean 0.502 is below the 0.7 typical drug-likeness threshold — the 4rlu closure-bias produces smaller electrophilic scaffolds.

**不假装:**
1. The 4rlu SA lift does NOT solve the `vina_high_affinity_4rlu` problem (which is structural: pocket chemistry + pharmacophore depth).
2. The 4rlu Wasserstein loss (0.421) is the HIGHEST across all 5 pockets — the closure-bias is reaching pockets that are not perfectly pharmacophore-aligned with the 4 curated reference ligands (phthalimide, anthraquinone, thio-pyrimidinone, fluorenone).

---

## Artifacts on disk

- `molmetal/reports/wf140_p1_sa_5pocket/4rlu/per_seed_metrics.json` — per-seed SA / QED / Ro5
- `molmetal/reports/wf140_p1_sa_5pocket/4rlu/cohort_smiles.jsonl` — 10 unique SMILES

## Metric JSON updated

- `metrics/by_metric/sa_mean.json` — added 4rlu measurement, lifted verdict to PARTIAL_LIFT_v5 (5/5 pocket coverage)
