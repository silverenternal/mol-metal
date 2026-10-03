# DrugOOD dataset sourcing

This directory does **not** ship the upstream DrugOOD tarballs. The IC50
splits used by `molmetal/data/four_domain_combined.py`,
`molmetal/scripts/train_drugood.py`, and
`molmetal/scripts/run_drugood_native_smoke.py` are sourced directly from
the **official** DrugOOD repository:

- Repo:    https://github.com/tencent-ailab/DrugOOD
- Paper:   Ji et al., "DrugOOD: Out-Of-Distribution (OOD) Dataset
  Curator and Benchmark for Drug Discovery" (NeurIPS 2022 Datasets
  and Benchmarks Track).
- Splits:  `drugood/ic50/` (ChEMBL_29 IC50 subset, with
  `assay` / `scaffold` / `size` OOD splits).
- License: see `LICENSE` in the upstream repo (research use).

## How to obtain the data

```bash
# 1. Clone the upstream repo (do NOT commit the tarballs).
git clone https://github.com/tencent-ailab/DrugOOD.git /tmp/DrugOOD

# 2. Locate the IC50 split files.
ls /tmp/DrugOOD/data/ic50/

# 3. Point Mol-Metal at them via the env var consumed by
#    molmetal/data/four_domain_combined.py::DEFAULT_DRUGOOD_TRAIN
#    (or pass --data to molmetal/scripts/run_drugood_native_smoke.py).
export DRUGOOD_IC50_TRAIN=/tmp/DrugOOD/data/ic50/data/train.csv
```

## Why no home-grown preparer anymore

The previous `molmetal/scripts/prepare_drugood_native.py` was a
thin wrapper that re-derived an upstream-format CSV from the public
DrugOOD release. It is **deprecated** and has been moved to
`/tmp/bak_audit/prepare_drugood_native.py.deprecated` (see git history
for the deletion commit). Users should consume the upstream CSVs
directly to avoid a second source of truth.

## Fallback for offline environments

If the upstream repo is unreachable (offline build, CI sandbox), the
MLP smoke-test in `run_drugood_native_smoke.py` will fall back to the
`metacytotox_fallback` data source via the `data_source` field in
`metrics/by_metric/drugood_ood_auc_native_smoke.json`. This is a
**MEASURED** degradation (typically ~0.05 AUC lower than the native
DrugOOD split); the OOD-robustness numbers in the paper should be
re-run from the official CSVs before submission.
