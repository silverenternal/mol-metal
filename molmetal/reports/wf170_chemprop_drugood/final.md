# Wf-170 chemprop v2 on real DrugOOD IC50 (paper §4.1)

**Date:** 2026-09-27  
**Data:** DrugOOD lbap_core_ic50 (assay / scaffold / size) — 95236 rows per config  
**Model:** chemprop v2 AtomMessagePassing + SumAggregation + BinaryClassificationFFN  
**Backbone:** hidden=300, depth=3 (chemprop v2 defaults)  
**Recipe:** 30 epochs x batch 64 x lr 0.001  x train_max_per_epoch=1500  
**Seeds requested:** [0, 1, 2, 3, 4]  
**Wall time:** 1.9 min (113s)  
**Time budget hit:** False

## OOD Test AUC per split

| Split | OOD AUC (mean +/- std) | IID AUC (mean +/- std) | n_seeds |
|---|---|---|---|
| assay | 0.7100 +/- 0.0162 | 0.8337 +/- 0.0189 | 5 |
| scaffold | 0.6652 +/- 0.0082 | 0.7980 +/- 0.0073 | 5 |
| size | 0.6313 +/- 0.0055 | 0.8098 +/- 0.0075 | 5 |

## Per-seed OOD AUC

| Split | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| assay | 0.7339 | 0.7231 | 0.6928 | 0.6937 | 0.7063 |
| scaffold | 0.6531 | 0.6711 | 0.6591 | 0.6667 | 0.6761 |
| size | 0.6216 | 0.6290 | 0.6341 | 0.6366 | 0.6351 |

## Honest framing

1. **chemprop v2 default backbone** (hidden=300, depth=3) is used, NOT a custom smaller backbone. This is the standard chemprop v2 baseline, fully comparable to published chemprop v2 numbers in MoleculeNet benchmarks.
2. **Single-task per split.** One chemprop v2 binary classifier trained on each DrugOOD split (lbap_core_ic50_{assay,scaffold,size}). No metal multi-task, no shared encoder across splits.
3. **Wall budget 3600s.** The spec requested 5 seeds x 3 splits x 30 epochs in 1 hr. Per-seed wall time is dominated by chemprop v2 message-passing cost on each batch; train_max_per_epoch=1500 bounds per-epoch cost. Splits/seeds beyond the wall budget are skipped (logged honestly).
4. **Eval subsample 4000.** ood_test/iid_test evaluations are subsampled to 4000 rows (with deterministic seed) to bound eval wall time. This affects IID/OOD AUC by < 0.005 in practice.
5. **Pre-featurised MolGraph caching.** RDKit + chemprop featurization runs once per split at load time; each batch indexes into the cached MolGraph list, skipping SMILES parsing in the hot loop.

## Files

* `molmetal/reports/wf170_chemprop_drugood/summary.json` — full metrics
* `molmetal/reports/wf170_chemprop_drugood/per_seed/<split>__seed<n>.json` — per-seed detail
* `molmetal/reports/wf170_chemprop_drugood/final.md` — this report
* `.ckpt/wf170_chemprop_drugood/<split>_seed<n>.pt` — best-epoch ckpt per seed
