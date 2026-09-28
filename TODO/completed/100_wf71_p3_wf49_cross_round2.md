# TODO-100 — Wf-71 Phase 3 SHIP (Wf-49 cross-dataset retrain round 2)

**Date**: 2026-09-26
**Workflow**: Wf-71 Phase 3 (Wf-49 cross-dataset retrain round 2 — extended per-cohort breakdown)
**Owner**: Hugo Lee + Wf-71 ultracode agent
**Parent**: `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 Phase 1+2) + `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 retrain)
**Status**: **SHIP** — 18/18 tests PASS + cross-dataset macro R^2 = +0.1248 (target_met = YES)
**Constraint**: paper-gate (TODO-84) honored -- no paper work touched

---

## §0 TL;DR

Wf-49 Phase 1+2 already SHIP the **RDKit 2D descriptor extractor + ASTSR F5 → F5_RDKIT_2D swap** (see `TODO/completed/91_wf49_r2_root_fix.md`). The natural follow-up is a per-cohort Ridge baseline retrain with honest per-cohort breakdown. Wf-71 P3 delivers exactly that:

| Dataset | Cohorts | Macro R^2 | Pearson r |
|---|---:|---:|---:|
| PlatinAI 4-cell (4 cell-lines) | 4 | +0.0034 ± 0.019 | +0.04 |
| MetalCytoToxDB 5-metal (Ru/Ir/Rh/Os/Re) | 5 | +0.1275 ± 0.028 | +0.36 |
| tmQM d-block (29 d-block metals) | 29 | +0.2433 ± 0.097 | n/a (R^2) |
| **Cross-dataset macro R^2** | n/a | **+0.1248 ± 0.098** | n/a |

**Verify criterion** (the Wf-71 P3 task contract):

> "F5_RDKIT_2D R^2 ≥ 0 mean across 3 datasets with per-cohort breakdown (or honest per-cohort breakdown)"

**target_met = YES** -- recorded in
`molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json`.

## §1 Deliverables (on disk)

| File | Status | Path |
|---|---|---|
| Bench script | SHIP (~620 LOC) | `molmetal/scripts/bench_wf49_cross_dataset_round2.py` |
| Test file | SHIP (18 tests PASS) | `molmetal/tests/test_wf49_cross_dataset_round2.py` |
| PlatinAI 4-cell per-cohort JSON | SHIP | `molmetal/reports/wf71_p3_wf49_cross_round2/platinai_4cell/per_cohort_metrics.json` |
| MetalCytoToxDB 5-metal per-cohort JSON | SHIP | `molmetal/reports/wf71_p3_wf49_cross_round2/metalcytotox_5metal/per_cohort_metrics.json` |
| tmQM d-block per-cohort JSON | SHIP | `molmetal/reports/wf71_p3_wf49_cross_round2/tmqm_dblock/per_cohort_metrics.json` (29 cohorts) |
| Cross-dataset summary | SHIP | `molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json` |
| Synthesis (close-out) | SHIP | `molmetal/reports/wf71_p3_wf49_cross_round2/synthesis.md` |
| This TODO entry | SHIP | `TODO/completed/100_wf71_p3_wf49_cross_round2.md` |

## §2 Honest per-cohort breakdown (party-twenty-spirit §3.5)

### PlatinAI 4-cell (per cell-line)
- MCF7: R² = -0.007; A549: R² = -0.009; A2780: R² = -0.008; A2780cis: R² = **+0.037** (Pearson r = +0.20)
- **Honest interpretation**: 3 of 4 cohorts at predict-the-mean level; only A2780cis (cisplatin-resistant) shows positive signal. This matches the Wf-49 Phase 1+2 finding that PlatinAI's RDKit 2D correlations per-cell-line are weak (|ρ| < 0.02). The macro R² = +0.003 reflects the A2780cis-only lift.

### MetalCytoToxDB 5-metal (per metal)
- Ru n=19135: R²=+0.116; Ir n=4546: R²=+0.084; Rh n=1134: R²=+0.125; Os n=1118: R²=+0.145; Re n=868: R²=**+0.168**
- **All 5 cohorts positive**, macro R² = +0.128, macro Pearson r = +0.36. Strong signal -- MolLogP / MolMR / HeavyAtoms are the dominant RDKit 2D drivers (consistent with Wf-49 Phase 1+2 univariate findings).

### tmQM d-block (per metal, n up to 2000/metal)
- Strongest: Mn (+0.51) / Cr (+0.42) / Zn (+0.38) / Cd (+0.36)
- Weakest: Ta (+0.07) / Nb (+0.08) / Rh (+0.11) / Hg (+0.14)
- **All 29 cohorts positive** -- macro R² = +0.243
- Honest interpretation: ligand-field-driven electronic properties are predictable from 12-d RDKit 2D for d-block transition metals but underfit highly-charged multi-coordination complexes (Ta/Nb).

## §3 What this bench is and isn't

### What it is
- A **predict-the-mean-beating baseline test** for F5_RDKIT_2D at the per-cohort level (5-seed × 80/20 train/test splits, n=4078 / 26801 / 100733 respectively)
- Honest per-cohort R² + Pearson r reporting
- 18 test cases verifying data loaders, fit, JSON shape, cross-dataset aggregation, CLI
- CPU-only (~3 minutes wall-clock)
- Backward-compatible with the Wf-49 Phase 1+2 features

### What it isn't
- **Not a state-of-the-art R² claim** -- this is a Ridge baseline on 12-d descriptors; it's a floor, not a ceiling.
- **Not the ASTSR + Hilbert retrain** (R20-P2 / TODO-93 deferred to GPU-0 Blackwell)
- **Not real-data wet-lab** (R20-P6 / TODO-97, separate 6-month outreach)

## §4 Cross-references

* `molmetal/scripts/bench_wf49_cross_dataset_round2.py` -- bench (~620 LOC)
* `molmetal/tests/test_wf49_cross_dataset_round2.py` -- tests (18 PASS)
* `molmetal/reports/wf71_p3_wf49_cross_round2/{platinai_4cell,metalcytotox_5metal,tmqm_dblock}/per_cohort_metrics.json` -- per-cohort JSONs
* `molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json` -- `target_met: true`
* `molmetal/reports/wf71_p3_wf49_cross_round2/synthesis.md` -- full close-out writeup
* `TODO/completed/91_wf49_r2_root_fix.md` -- Wf-49 Phase 1+2
* `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` -- R20-P2 follow-up

## §5 Honest framing (party-twenty-spirit §3.5)

We do not inflate R² by selecting best cohort / best seed / best feature subset. All 38 cohorts (4 + 5 + 29) reported with full per-seed breakdown. Negative R² (predict-the-mean-beating) reported as such. Cross-dataset macro R² = +0.124 is the macro of per-cohort means -- not a cherry-picked headline.

PlatinAI 4-cell's weak performance (macro R² = +0.003) is honest and matches Wf-49 Phase 1+2 univariate findings. This is the **expected** result for a 12-d Ridge baseline on cell-line pIC50, and provides the honest floor against which R20-P2 Hilbert+ASTSR joint retrain must demonstrate lift.

No fabricated numbers. No best-seed picking. All seeds (0..4), all cohorts, all 3 datasets reported.
