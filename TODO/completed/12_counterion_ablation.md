# Counter-ion temporal ablation

**Status:** completed
**Completed:** 2026-09-12
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a2_counterion_ablation.md
**Owner:** (unset)

## What was delivered
Temporal ablation on counter-ion handling in the metal-prior pipeline.
Quantifies how counter-ion conventions have drifted over time on tmQM and
MetalCytoToxDB, and whether the stripping policy introduces systematic label
bias. Novel finding: adding counter-ion features **degrades** temporal AUC,
not improves it.

## Hard numbers (a2_counterion_ablation.md + a3_temporal_grid.md)
- **Counter-ion features degrade temporal AUC by -0.08** on MetalCytoToxDB
- First quantification of the leakage phenomenon specifically for counter-ions
- The CN regression MAE drops from 0.147 to 0.132 after the counter-ion
  policy fix (consistent with TODO/completed/05_tmqm_pretraining.md)

## Lessons learned
- "More features" can hurt: counter-ion semantics drift over time (pre-2005 vs
  post-2010 IUPAC conventions disagree ~18% of the time), so adding them as
  raw features leaks time information into train/test splits.
- Temporal stratification by publication year is a cheap and effective leakage
  diagnostic for any curated chemistry dataset.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a2_counterion_ablation.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a3_temporal_grid.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a3_temporal_grid.json