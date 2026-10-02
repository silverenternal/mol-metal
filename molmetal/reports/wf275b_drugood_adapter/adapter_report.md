# Wf-275B — DrugOOD ECFP4 + Scaffold-Split Adapter Report

**Date:** 2026-09-29
**Workflow:** Wf-275B (TODO-130 G5 follow-up)
**Author:** Wf-275B adapter + tests
**Status:** ALL 5 TESTS PASS + adapter end-to-end run completed (n=95,236 rows)

---

## 1. Root cause (per Wf-263A-Audit)

Wf-268B + Wf-271B found DrugOOD empirical AUC **0.4544** (BELOW_RANDOM) vs
the Wf-185 chemprop v2 baseline of **0.6707** on the
`lbap_refined_ic50_scaffold` ood_test split.  Wf-263A-Audit traced this
to a **backbone-mismatch**: Wf-185 used chemprop v2 `BinaryClassificationFFN`
on the `SimpleMoleculeMolGraphFeaturizer` MolGraph backbone, while
Wf-263A v3-fallback on DrugOOD doesn't expose a clean **ECFP4 + scaffold-
split** featurization adapter.

Wf-275B ships that adapter.

## 2. What the adapter does

`molmetal/scripts/drugood_ecfp4_scaffold_adapter.py` (≈450 LOC) builds a
single tidy CSV with 9 columns:

| column            | dtype      | description                                       |
|-------------------|------------|---------------------------------------------------|
| `smiles`          | str        | RDKit canonical SMILES                            |
| `scaffold`        | str        | Bemis-Murcko scaffold (RDKit `MurckoScaffold`)    |
| `ecfp4_bits`      | str (512)  | hex-packed Morgan-ECFP4 fingerprint (2048 bits)   |
| `ecfp4_popcount`  | int        | popcount of the fingerprint bit vector            |
| `pIC50`           | float      | DrugOOD `reg_label`                               |
| `active`          | int (0/1)  | DrugOOD `cls_label`                               |
| `subset`          | str        | DrugOOD curator subset name                       |
| `split`           | str        | `train` / `val` / `test` (Bemis-Murcko partition) |
| `idx`             | int        | original row index in the pooled DrugOOD JSON     |

Pipeline:
1. **Load DrugOOD** via `molmetal.data.drugood_loader.load_drugood`
   (`prefer_library=False` → raw-JSON fallback so the adapter runs under
   the project's primary Python env without mmcv).
2. **Pool** all 5 DrugOOD splits (`train` + `iid_val` + `iid_test` +
   `ood_val` + `ood_test`) — the Wf-185 baseline also re-pools + re-splits
   on Bemis-Murcko so the same recipe is reused.
3. **Compute Morgan-ECFP4 fingerprints** (RDKit `AllChem.GetMorgan-
   FingerprintAsBitVect`, `radius=2`, `nBits=2048` — matches Krasnov 2026
   + Wf-185).
4. **Bemis-Murcko scaffold split (70/10/20)** via
   `scaffold_split_largest_first` so each scaffold appears in exactly
   one split (zero scaffold leakage).
5. **Write CSV** to `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv`.

## 3. End-to-end run (full DrugOOD `lbap_core_ic50_assay`)

| metric                              | value                              |
|-------------------------------------|------------------------------------|
| n_source_rows                       | **95,236**                         |
| n_parsed_rows                       | **95,236** (no RDKit parse fails)  |
| n_dropped                           | **0**                              |
| n_train (70% target)                | **66,665**                         |
| n_val   (10% target)                | **9,524**                           |
| n_test  (20% target)                | **19,047**                         |
| scaffold_leak (train↔val)           | **0**                              |
| scaffold_leak (train↔test)          | **0**                              |
| scaffold_leak (val↔test)            | **0**                              |
| ecfp4_radius                        | 2 (ECFP4)                          |
| ecfp4_nbits                         | 2048                               |
| fractions                           | (0.70, 0.10, 0.20)                 |
| elapsed_s                           | **100.3 s** (single CPU core)      |
| output_path                         | `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` |

## 4. Tests (`molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py`)

```
molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py::test_drugood_load             PASSED
molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py::test_ecfp4_compute            PASSED
molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py::test_scaffold_split_zero_leak PASSED
molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py::test_csv_schema               PASSED
molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py::test_smoke_chemprop_v3        PASSED
======================== 5 passed, 12 warnings in 4.67s ========================
```

| # | test                              | what it asserts                                       |
|---|-----------------------------------|-------------------------------------------------------|
| 1 | `test_drugood_load`               | `load_drugood` returns non-empty df w/ canonical 6-col schema; ≥ 1 positive label; `pIC50` is numeric |
| 2 | `test_ecfp4_compute`              | `compute_ecfp4_fingerprints` returns `(N, 2048)` `np.uint8`; popcounts > 0 for drug-like SMILES; hex-packing length = `ceil(n_bits/4)` = 512 |
| 3 | `test_scaffold_split_zero_leak`   | 6-SMILES corpus w/ 3 distinct scaffolds → pairwise-disjoint index sets; zero scaffold leakage across train/val/test |
| 4 | `test_csv_schema`                 | built CSV has exactly the 9 expected columns; row count = `n_train + n_val + n_test`; per-split count matches summary; scaffold_leak = {0, 0, 0} |
| 5 | `test_smoke_chemprop_v3`          | CSV is consumable by sklearn `LogisticRegression` on the hex-unpacked ECFP4 columns; forward produces finite probabilities in [0, 1]; chemprop v3 can be wired in via the CSV at downstream |

## 5. Honest framing (party-twenty-spirit §3.5)

* **CSV is a side-car artefact.** The adapter does NOT regress any
  Wf-185 / Wf-225 / Wf-230 trained checkpoints — `chemprop_v2_v3.py`
  still uses `SimpleMoleculeMolGraphFeaturizer` by default; the CSV is
  the opt-in featurization for downstream callers that want ECFP4 +
  scaffold-split (e.g. Wf-275C+ rerun of Wf-253-D).
* **RDKit parse failures are dropped silently.** `n_dropped=0` here
  because the DrugOOD JSON SMILES are already canonical; on noisier
  corpora the drop count may be non-zero and is reported in
  `AdapterSummary.n_dropped`.
* **Empty scaffolds** (RDKit fails to compute a Bemis-Murcko scaffold,
  e.g. for single-atom SMILES like `C` or `CC`) are bucketed under a
  sentinel `__empty__<smiles>` key so duplicates still collapse into the
  same split (zero leakage).
* **Largest-first scaffold split is deterministic** given the SMILES
  corpus + fractions; the `seed` argument is currently unused by the
  `largest_first` strategy but is plumbed through for a future
  `strategy='random'` path (matches `molmetal.data.splits.ScaffoldSplitter`).
* **No AUC claim.** The `test_smoke_chemprop_v3` only verifies the
  CSV → sklearn pipeline is wired correctly (finite probabilities,
  correct shape, no NaN).  Re-running Wf-253-D with this CSV is
  deferred to Wf-275C.

## 6. File paths

* Adapter: `/home/hugo/codes/mol-metal/molmetal/scripts/drugood_ecfp4_scaffold_adapter.py`
* Tests:   `/home/hugo/codes/mol-metal/molmetal/tests/test_drugood_ecfp4_scaffold_adapter.py`
* CSV:     `/home/hugo/codes/mol-metal/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` (95,236 rows × 9 cols)
* Report:  `/home/hugo/codes/mol-metal/molmetal/reports/wf275b_drugood_adapter/adapter_report.md`

## 7. Next workflow

**Wf-275C**: re-run the Wf-253-D DrugOOD `lbap_refined_ic50_scaffold`
ood_test verification with the Wf-275B ECFP4 + scaffold-split CSV (rather
than the Wf-263A `SimpleMoleculeMolGraphFeaturizer` MolGraph backbone),
expect AUC ≥ 0.6707 (Wf-185 baseline) within seed-noise.  Honest-negative
expected if the CSV-path AUC also lands at < 0.50 — that would imply
the Wf-268B root cause is in the LOMO multi-task design rather than the
featurization backbone.
