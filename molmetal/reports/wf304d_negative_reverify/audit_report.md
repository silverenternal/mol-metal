# Wf-304D — HONEST-NEGATIVE Re-Audit (Post Wf-301 Fixes)

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** HONEST-NEGATIVE re-audit per Wf-297B field-avg targets §4
**Scope:** Verify whether Wf-301 fixes (a: drugood_dann_smoke, b: vina_pt_strip, c: platinai_stratified) lifted the CFM main-line field-avg gaps.

---

## §1 — Method

Re-check each CFM main-line metric vs Wf-297B field-avg band:

| Metric | Field-avg band (Wf-297B §4) | Source for current value |
|---|---|---|
| `drugood_v3_scaffold` | ≥ 0.55–0.65 | `metrics/by_metric/drugood_v3_scaffold.json` (Wf-253D MEASURED) + Wf-301a (NOT YET RUN) |
| `cfm_real_vina_dock` | -7.5 to -9.5 kcal/mol | `molmetal/reports/wf301b_vina_pt_strip/vina.json` (72 docked ligands, 1h36 pocket, MEASURED) |
| `m14_lead_decode_ratio` | 0.85–0.95 | `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` (n=26,801, 5-metal stratified) |
| `cfm_decode_real` | 0.85–0.95 | `metrics/by_metric/cfm_decode_real.json` (Wf-271B) + Wf-188 m14 anchor SUPERSEDED |

---

## §2 — Per-Metric Findings

### §2.1 — `drugood_v3_scaffold` (target ≥ 0.55–0.65)

| Item | Value |
|---|---|
| **Current value (MEASURED)** | 0.4544 ± 0.0327 (aggregate, n=8 (metal,seed) pairs across 3 metals; Wf-253D) |
| **Per-metal MEASURED** | Ru: 0.4905±0.0312 (n=4); Ir: 0.4615±0.0283 (n=3); Os: 0.4113 (n=1) |
| **Wf-253D verdict** | `BELOW_RANDOM (aggregate 0.4544)` |
| **Wf-301a (drugood_dann_smoke)** | **NOT YET RUN** — `molmetal/reports/wf301a_drugood_dann_smoke/` is empty (no result.json, no checkpoint) |
| **Target ≥ 0.55** | NOT MET |
| **Verdict** | **NOT MET** (BELOW_RANDOM; Wf-301a was supposed to ship a DANN smoke + multi-task pretrain per Wf-300A §3 but the directory contains zero artifacts) |

### §2.2 — `cfm_real_vina_dock` (target -7.5 to -9.5 kcal/mol)

| Item | Value |
|---|---|
| **Current value (MEASURED)** | mean = **-5.0125 kcal/mol** (n=72 real_docked, n=28 PDBQT_FAIL/PARSE_FAIL out of 100 inputs) |
| **Range** | min -9.333, max -2.442, best_score_overall -9.333 |
| **Pocket** | 1h36 (Pt-stripped, exhaustiveness=4, n_poses=3, box_size=20.0) |
| **Wf-301b (vina_pt_strip)** | COMPLETED — see `molmetal/reports/wf301b_vina_pt_strip/vina.json` |
| **Target -7.5 to -9.5** | **MEAN NOT MET** (-5.0125 is above field-avg band; only the BEST pose (-9.333) and a small tail (-6.5 to -7.4) reach into band) |
| **Verdict** | **NOT MET** (mean -5.0 is ~2.5 kcal/mol weaker than field-avg lower bound; Wf-301b delivered 72/100 = 72% Pt-strip success rate but the docked energies are still shallow) |

### §2.3 — `m14_lead_decode_ratio` (target 0.85–0.95)

| Item | Value |
|---|---|
| **Current value (MEASURED)** | **0.9958** (Wf-188, n=26,801 MetalCytoToxDB m14 full cohort, 5-metal stratified Ru/Ir/Os/Rh/Re, cuda, jitter=0.05, closure_bounded=True) |
| **Per-metal MEASURED** | Ru: 0.9948, Ir: 0.9974, Os: 1.000, Rh: 1.000, Re: 1.000 |
| **Target 0.85–0.95** | **MET** (above band by +0.046 to +0.146) |
| **Verdict** | **MET** (above field-avg band; m14 LEAD anchor already shipped) |

### §2.4 — `cfm_decode_real` (target 0.85–0.95)

| Item | Value |
|---|---|
| **Current value (MEASURED, headline)** | **0.9645** (`metrics/by_metric/cfm_decode_real.json`, n=818, 1 seed, CPU, PlatinAI pt_subset, Wf-271B 2026-09-29) |
| **Superseding anchor** | **0.9958** (`molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json`, n=26,801, 5-metal stratified, cuda) — `verdict: SUPERSEDED metrics/by_metric/cfm_decode_real.json 0.9645` |
| **Target 0.85–0.95** | **MET** (both 0.9645 and 0.9958 above band) |
| **Verdict** | **MET** (already PASS, with methodology caveat: Wf-188 anchor is the paper-grade value; the Wf-271B 0.9645 is a smaller-cohort methodology-mismatch re-measurement per its own `regression_disclosure` block) |

---

## §3 — Output Summary (PLAIN TEXT)

```
n_metrics = 4
n_field_avg_met = 2 / 4

per_metric_current_target_verdict:
  drugood_v3_scaffold         current=0.4544   target=0.55-0.65   verdict=NOT MET  (BELOW_RANDOM; Wf-301a directory empty)
  cfm_real_vina_dock          current=-5.0125  target=-7.5 to -9.5 kcal/mol  verdict=NOT MET  (mean above band; only best pose -9.333 hits band)
  m14_lead_decode_ratio       current=0.9958   target=0.85-0.95   verdict=MET      (above band; Wf-188 anchor 5-metal stratified)
  cfm_decode_real             current=0.9645   target=0.85-0.95   verdict=MET      (Wf-188 anchor 0.9958 supersedes; methodology caveat noted)
```

---

## §4 — Honest Framing (party-twenty-spirit §3.5)

**Bottom line**: Wf-301 closed **2/4** field-avg targets (m14 LEAD and cfm_decode_real were already PASS, no Wf-301 work needed) and **2/4** remain honest-negatives:

1. **`drugood_v3_scaffold` (NOT MET)** — Wf-301a was scoped to ship a DANN smoke + multi-task DrugOOD pretrain per Wf-300A §3, but the report directory `molmetal/reports/wf301a_drugood_dann_smoke/` contains zero artifacts. The Wf-300A remediation plan (`remediation_plan.md` §3.1-3.2) extends `--dataset drugood` and `DomainAdversarialHead` into `train_dmpnn_multitask_wasserstein.py` — these extensions have not landed on disk; current MEASURED value remains the Wf-253D BELOW_RANDOM 0.4544.

2. **`cfm_real_vina_dock` (NOT MET)** — Wf-301b shipped (72/100 Pt-strip + dock complete on 1h36), but the docked mean kcal/mol is -5.0 — ~2.5 kcal/mol above the field-avg lower bound -7.5. The best pose reaches -9.333 (in band), but the median pose is shallow (-4.0 to -5.0). Honest framing: Pt-strip is wired but the docking funnel still drops enough affinity to break field-avg band on the mean.

**What still needs to ship**:
- Wf-301a DANN + multi-task DrugOOD pretrain (per Wf-300A §3) — ~3-4 hr RTX 5090; output 25 ckpts + `aggregate_drugood_v3_scaffold.json`; pass criterion: ≥ 3/5 metals with mean AUC ≥ 0.55.
- Wf-301 vina affinity lift — 1h36 pose-cluster re-rank + pose-energy funnel on top-10 (current: best=-9.333 hits band, median -4.0 misses). Honest-negative REMAINS until mean kcal/mol lands in band.

**End of Wf-304D audit report.**