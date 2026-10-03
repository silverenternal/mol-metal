# TODO-97 — R20-P6 Wet-lab IC50 validation (ORNL / PKU / 协和)

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (wet-lab outreach, 6 months)**
**Priority**: **MEDIUM** (m18/m19/m20 wet-lab cooperation — gates paper §5 final claim)
**Owner**: Hugo Lee + ultracode agents + wet-lab PI (TBD)
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/82_metric_gap_filling.md` §5
**Metric gap(s)**: m18 (Protein-metal binding AUC) + m19 (LOMO 5-fold) + m20 (PlatinAI 4-cell resistance)
**Skill alignment**: `mol-metal-anticancer` + `mol-metal-mechanism-mapping`

---

## §0 Why this plan exists

The Mol-Metal framework's **anticancer claim** is anchored on:
- m18 Protein-metal binding AUC 0.7393 ± 0.009 (D-MPNN, 3 seeds × 100 ep, scaffold split, n=5,257 × 209 cells × 5 metals)
- m19 LOMO 5-fold 0.535 ± 0.110 (Ru 0.571 / Ir 0.618 / Os 0.611 / **Rh 0.368** / Re 0.506; 3 seeds × 30 ep)
- m20 PlatinAI 3-cell resistance 0.546 (5 seeds × 100 ep, scaffold_70_10_20, tmQM init)

These are **CPU-computed on retrospective datasets**. **R20-P6 initiates wet-lab cooperation** with ORNL / PKU / 协和 (3 candidate PIs) for prospective IC50 validation on top-20 generated candidate metal complexes (cisplatin / carboplatin / oxaliplatin analogs). Wet-lab results gate paper §5 final claim.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | wet-lab (cell culture + MTT / SRB / CellTiter-Glo assay) |
| Wall-time | 6 months (compound synthesis 1 mo + cell culture 2 mo + IC50 assay 2 mo + analysis 1 mo) |
| Output | `molmetal/reports/wf55_r20_p6/wetlab_ic50_{orf,orf_oxaliplatin,...}.json` (per-compound IC50 ± std) |
| Tests | ~5 wet-lab validation reports |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Wet-lab coordination | mol-metal-anticancer | mol-metal-mechanism-mapping |
| Compound prioritization | mol-metal-click-chemistry | mol-metal-anticancer |
| IC50 analysis pipeline | mol-metal-anticancer | mol-metal-lambda-calculus |

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P6.1** Wet-lab PI outreach + compound synthesis | 1 mo wet-lab | LOI signed + 20 compounds synthesized | LOI signed + 20/20 compounds QC-pass |
| **P6.2** Cell culture + IC50 assay (MCF7 + A549 + A2780 + A2780cis + 5 more NCI-60 cells) | 3 mo wet-lab | wetlab_ic50_*.csv (per-compound × per-cell IC50 ± std) | full panel IC50 measured |
| **P6.3** In-silico vs wet-lab agreement | 1 mo CPU | `molmetal/reports/wf55_r20_p6/silico_vs_wetlab.json` (Pearson r ≥ 0.7) | Pearson r ≥ 0.7 |
| **P6.4** Paper §5 final claim | 1 mo doc | `paper/sections/05_wetlab_validation.tex` (after TODO-84 authorization) | shipped after auth |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = wet-lab IC50 + in-silico vs wet-lab agreement JSON
- PROJECTED = Pearson r ≥ 0.7 between D-MPNN prediction and wet-lab IC50

## §5 Cross-references

- `TODO/completed/82_metric_gap_filling.md` §5 (R20-P1..P7)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate; P6.4 needs auth)
- `TODO/completed/82c_m1m2m7_retrain.md` (Wf-31 m1/m2 deferred to R20-P2/P3)
- `TODO/completed/82d_m16m19m8_retrain.md` (Wf-32 m16/m19/m8 deferred)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 R20-P6
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `mol-metal-wetlab-protocol`)
