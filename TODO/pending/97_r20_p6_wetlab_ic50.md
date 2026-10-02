# TODO-97: R20-P6 Wet-lab IC50 validation (ORNL/PKU/协和)

**Status**: **DEPRECATED** per AGENTS.md §10 + §3.2 R22-P1 + §3.3 R23 (2026-09-29/30 user directives)
**Lambda gap**: m18 / m19 / m20 (per LAMBDA_ROADMAP §3.2/§3.3)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §8 (DEPRECATED scope)

## §1 What

Originally R20-P6 wet-lab IC50 validation (ORNL/PKU/协和). **DEPRECATED** per AGENTS.md §5.4 + §10 + user directive 2026-09-29 *"我们发二区就可以了，不用规划湿实验"* + 2026-09-30 reinforcement *"搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"*.

## §2 Why

Tier-2 acceptance bar is computational-only (per AGENTS.md §5.4). Wet-lab validation is 6-12 month external-dependency that does NOT gate Tier-2 acceptance.

## §3 How

**Re-scoped to in-silico alternatives** (NOT a wet-lab plan):

1. **In-silico AlphaFold-Multimer docking** for protein-metal binding affinity (replaces wet-lab SPR/BLI)
2. **In-silico MetalSite feature extraction** from PDB structures (replaces wet-lab crystallography)
3. **Cite literature values** from peer papers for any reviewer-rebuttal experimental data request (per AGENTS.md §10.4)

All 4 LOI drafts (`wetlab_loi_{rusanov_2025_platinai,pku_cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md`) + 3 SOPs (`wetlab_sop_{binding_affinity_assay,cellular_uptake_assay,cytotoxicity_assay}.md`) + `wetlab_ic50_analysis_pipeline.md` RETAINED as **historical artifacts** in `TODO/pending/` per AGENTS.md §10.5 retention policy — do NOT execute / do NOT send.

## §4 Pre-conditions

**NO live pre-conditions.** This file exists only as a re-scope marker per AGENTS.md §10.6 + 2026-09-29/30 user directives.

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: 4 LOIs + 3 SOPs + 1 IC50 pipeline file EXIST on disk in `TODO/pending/` per Wf-381 + Wf-358 audit (template reference, NOT executed)
- **PROJECTED**: In-silico AlphaFold-Multimer + MetalSite feature extraction (if USER_OK-gated launch — `TODO/completed/234_wf234_frontier_math_audit_integration.md`)
- **HONEST-NEGATIVE**: NO actual wet-lab work — full deprecation per AGENTS.md §10 + 2026-09-29/30 user directives

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §8 (R20-P6 spec DEPRECATED)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §3.2 R22-P1 DEPRECATION NOTICE + §3.3 R23 ALL DEPRECATED
- `AGENTS.md` §5.4 (Tier-2 + no wet-lab) + §10 (Wetlab deprecation FAQ)
- `TODO/pending/wetlab_loi_*.md` (4 historical LOI templates, do NOT send)
- `TODO/pending/wetlab_sop_*.md` (3 historical SOP templates, do NOT execute)
- `TODO/pending/wetlab_cold_email_template.md` (template, do NOT send)
