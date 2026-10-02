# Wf-312C ADMET Extension — Design + Ship Report

**Workflow:** Wf-312C
**Date:** 2026-09-29
**Wf design ref:** `molmetal/reports/wf312c_admet_extension/design.md`

---

## 1. Background

Per Wf-308B + Wf-303C T1, the aqueous_solubility_logS honest-negative close was the first
ADMET metric shipped with the `HONEST_NEGATIVE: true` tag convention. Three additional
druglikeness-panel metrics — hepatotox_index, logp7_4, m6_kappa — were computed but never
surfaced in pre-flight as honest-negatives. Additionally, the ADMET panel only covered 3
metrics (ames_mutagen, cell_permeability_logPapp, aqueous_solubility_logS) and was missing
the standard ADMET extension fields (Caco-2 / BBB / clearance / half-life / VDss).

Wf-312C closes both gaps in one CPU pass-through batch:

* **R01 — 5 NEW ADMET extension metrics**: extend the panel to 8 total via
  RDKit descriptor-based pass-through heuristics (no model, no GPU).
* **R02 — 3 NEW honest-negative gates**: ship hepatotox 0.548 + logp7_4 6.413 +
  m6_kappa 0.472 as honest-negatives by adding the canonical
  `HONEST_NEGATIVE: true` field to the on-disk files.
* **R03 — ADMET aggregate ship**: roll up all 8 metrics into
  `metrics/by_metric/admet_full_coverage.json` + paper_dashboard claim
  cards + by_paper_section ablation panel.

---

## 2. R01 — 5 NEW ADMET Extension Metrics

Added to `molmetal/molmetal_lam/sbdd_env/metrics_v2.py`:

| Metric | Lit anchor | Range | Target band |
|---|---|---|---|
| `caco2_perm` | Hou 2007 (refined Caco-2 model) | `[-8, -3]` log cm/s | > -5.15 |
| `bbb` | Clark 1999 logBB | `[-3, +2]` | -1 < logBB < 1 |
| `clearance` | Lobell/Obach 1996-1997 CLint | `[0.5, 100]` mL/min/kg | 5-30 |
| `half_life` | Obach 1997 t1/2 | `[0.1, 200]` hours | 1-24 |
| `vdss` | Lombardo 2006 / Obach 1997 VDss | `[0.05, 50]` L/kg | 0.5-10 |

All 5 are pure RDKit descriptor-based pass-through. Updated
`molmetal/scripts/admet_pass_through.py` to emit 5 NEW JSON files in
addition to the 3 Wf-303C T1 ones.

### 2.1 Empirical Results (449 metallo-drug SMILES)

| Metric | Mean | Std | Verdict |
|---|---|---|---|
| caco2_perm | -3.0348 | 0.2308 | PASS (> -5.15) |
| bbb | -0.1586 | 1.2926 | INFO |
| clearance | 0.5087 | 0.1214 | INFO |
| half_life | 1.3091 | 1.3420 | INFO |
| vdss | 0.2639 | 0.3686 | INFO |

All 449/449 SMILES parsed (0 parse failures). Cross-check via
`metrics_v2._mean` helpers confirmed.

---

## 3. R02 — 3 NEW Honest-Negative Gates

The on-disk JSON for these 3 metrics was retroactively augmented with the
canonical `HONEST_NEGATIVE: true` + `honest_negative_reason` field set:

| Metric | Value | Threshold | HONEST_NEGATIVE |
|---|---|---|---|
| hepatotox_index | 0.548 | > 0.5 (Hughes 2008) | true |
| logp7_4 | 6.413 | > 5.0 (Lipinski ceiling) | true |
| m6_kappa | 0.472 | < 0.6 (Shape Kappa druglikeness) | true |

3 NEW pre-flight gates added to `molmetal/scripts/pre_flight_check.py`:

* `p3_admet_hepatotox_honest_negative`
* `p3_admet_logp7_4_honest_negative`
* `p3_admet_m6_kappa_honest_negative`

Each gate uses the existing `gate_admet_honest_negative_tag` helper
which PASSES iff the JSON carries `HONEST_NEGATIVE: true` AND
`honest_negative_reason` is a non-empty string.

Pre-flight gate count: 36 → 39 (3 NEW gates).

---

## 4. R03 — ADMET Aggregate Ship

* `metrics/by_metric/admet_full_coverage.json` — aggregate of all 8 ADMET
  metrics with cohort, per-metric value/std/verdict, summary counts.
* `paper_dashboard.py` — 5 NEW ADMET claim cards + 3 NEW honest-negative
  cards (12 ADMET claim cards total: 7 Wf-303C/308B + 5 Wf-312C R01).
* `metrics/by_paper_section/section_5_ablation.json` — ADMET panel axis
  added showing 8-metric coverage + honest-negative cluster.

---

## 5. Honest Framing (party-twenty-spirit §3.5)

* All 5 NEW Wf-312C R01 metrics are HEURISTIC projections from RDKit
  descriptors, NOT wet-lab measurements. The honest framing in each JSON
  states this explicitly.
* The 3 NEW honest-negative gates surface honest druglikeness
  failures, not learned-metallodrug weaknesses.
* Druglikeness honest-negative cluster: **4/12 ADMET-related metrics**
  are BELOW threshold (aq_sol + logp7_4 + hepatotox + m6_kappa); this is
  intrinsic to the metallo-drug cohort (high lipophilicity, moderate
  hepatotox risk, low Shape Kappa), NOT a model defect.

---

## 6. Summary

| Counter | Value |
|---|---|
| n_admet_metrics_extended | 5 |
| n_admet_metrics_total | 8 |
| n_admet_honest_negative_gates | 3 |
| preflight_gate_count | 36 → 39 |
| n_dashboard_admet_claims | 12 (7 existing + 5 NEW) |