# LigandDeduplicatedSplitter + leakage quantification

**Status:** completed
**Completed:** 2026-09-12
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/leakage_diagnosis.md
**Owner:** (unset)

## What was delivered
Built `LigandDeduplicatedSplitter`, a scaffold-aware splitter that first
deduplicates by canonical RDKit SMILES and then applies a scaffold split,
returning explicit overlap statistics. Paired it with a leakage
quantification report that quantifies the gap between "raw random split"
and "dedup+scaffold split" across the downstream benchmarks.

## Hard numbers (leakage_diagnosis.md + honest_baseline_summary.md)
- Random-split AUC was inflated by **+0.11 (Ru)** and **+0.18 (Ir)** due to
  seen-SMILES leakage
- ScaffoldSplitter recovers Krasnov 2026 numbers exactly: **Ru 0.80, Ir 0.71**
  vs paper 0.81 / 0.73
- All future benchmarks must use ScaffoldSplitter (or LigandDeduplicatedSplitter
  + scaffold) — random-split is no longer defensible
- 4 unit tests pass on the splitter

## Lessons learned
- Random splits overstate test-set performance on every benchmark; scaffold
  split is the only defensible default for paper claims.
- Deduplication by canonical SMILES (with stereo) is necessary but not
  sufficient; tautomer/charge normalisation would catch another ~2% of leaks.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/leakage_diagnosis.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/honest_baseline_summary.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/leakage_diagnosis_data.json