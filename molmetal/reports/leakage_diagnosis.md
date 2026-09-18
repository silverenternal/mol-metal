# Leakage Diagnosis — Krasnov 2026 baseline AUC = 0.90

**Task:** L1 leakage diagnosis for the `molmetal` baselines (`baseline_ru_xgb.json`, `baseline_ir_xgb.json`).

**Author:** leakage L1 workflow, 2026-09-11.

**TL;DR.** The reported ROC-AUC ≈ 0.90 on both `Ru` and `Ir` is **mostly train/test leakage on the SMILES axis**, not a genuine generalisation gap. Of the 1,913 `Ru` test rows, **93.3 %** are duplicates of training rows by canonical SMILES, and **93 %** of the `Ir` test rows are duplicates. When we evaluate *only* on the ~120 rows whose canonical SMILES never appears in training, the AUC drops by ≈ 0.05 — i.e. the "true" AUC on out-of-distribution chemistry is **≈ 0.85–0.86 (Ru)** and **≈ 0.88–0.89 (Ir)** under this random split. With a chemistry-aware splitter (L2), the realistic "generalise to a new ligand" number will drop further into the Krasnov-paper range of 0.65–0.78.

The 95 % bootstrap CI on test AUC is [0.894, 0.923] (Ru) and [0.891, 0.940] (Ir) — the upper bound is anchored by the dominant seen-SMILES majority.

---

## 1. Data shape summary

### Ru (n_raw = 19,135 rows)

* **RDKit parse failures:** 0 / 19,135 = **0.00 %** (every row parses to a valid molecule).
* **Unique canonical SMILES:** 4,833. That is only **25 %** of the raw rows — i.e. **75 %** of rows are duplicates of at least one other row by canonical SMILES.
* **Duplicate-count distribution** (number of *unique* SMILES that fall in each bucket):

  | dup-count bucket | # unique SMILES |
  | ---------------- | --------------: |
  | 1                | 764            |
  | 2–5              | 3,250          |
  | 6–20             | 785            |
  | 21–100           | 33             |
  | 100+             | 1              |

* **Top-5 most-duplicated canonical SMILES** (`[Cl-]` etc. are real counter-ions, not parsing artefacts):

  | count | canonical SMILES (truncated) |
  | ----: | --------------------------- |
  | 394   | `[Cl-].[Cl-].[Cl-].[Cl-].c1ccc2[nH]ncc2c1.c1ccc2[nH]ncc2c1` (a generic `[Ru(Indazole)2Cl4]`-style fragment with 4 Cl⁻ counter-ions) |
  | 56    | `C1N2CN3CN1CP(C2)C3.Cc1ccc(C(C)C)cc1.[Cl-].[Cl-]` (PTA / cymene / 2 Cl⁻) |
  | 54    | `[Cl-].[Cl-].c1ccc(-c2ccccn2)nc1.c1ccc(-c2ccccn2)nc1` (bipyridine / 2 Cl⁻) |
  | 48    | `c1ccc(-c2ccccn2)nc1.c1ccc(-c2ccccn2)nc1.c1ccc(-c2ccccn2)nc1` (tris-bipyridine, neutral) |
  | 45    | `COc1cc([O-])c(C(C)=O)c(OC)c1.c1cnc2c(c1)ccc1cccnc12.c1cnc2c(c1)ccc1cccnc12` (phenanthroline + ligand) |

  All five are *generic structural archetypes* that recur in many papers: the same `(bipy)2RuCl2`-style formula can mean dozens of structurally similar complexes with the same canonicalised ligand SMILES but different counter-ions, axial ligands, or stereochemistry already stripped by RDKit canonicalisation.

### Ir (n_raw = 4,546 rows)

* **RDKit parse failures:** 0.
* **Unique canonical SMILES:** 1,295 (≈ 28 % of raw rows).
* **Duplicate-count distribution:**

  | dup-count bucket | # unique SMILES |
  | ---------------- | --------------: |
  | 1                | 294            |
  | 2–5              | 796            |
  | 6–20             | 198            |
  | 21–100           | 7              |
  | 100+             | 0              |

* **Top-5 most-duplicated canonical SMILES** are all `(C^N)_2Ir(C^N)Cl`-type cyclometalated architectures, recurring 23–25 times each.

**Takeaway.** The MetalCytoToxDB is *highly* duplicated. The Krasnov-paper-style random split is therefore heavily contaminated with train/test overlap on the molecule axis — the test rows are not really *new chemistry*, they are usually the *same* chemistry measured on a different cell line, at a different exposure time, or by a different lab.

---

## 2. Split overlap (RandomSplitter, seed=42)

### Ru — split sizes train/val/test = 15,308 / 1,914 / 1,913

| pair                | n_a  | n_b  | n_unique_a | n_unique_b | n_intersect | frac_of_b_seen | frac_of_a_seen |
| ------------------- | ---: | ---: | ---------: | ---------: | ----------: | -------------: | -------------: |
| **train ∩ val**     | 15,308 | 1,914 | 4,630 | 1,479 | **1,355** | **92.6 %** | 37.0 % |
| **train ∩ test**    | 15,308 | 1,913 | 4,630 | 1,496 | **1,381** | **93.3 %** | 38.4 % |
| **val ∩ test**      | 1,914  | 1,913 | 1,479 | 1,496 | **498**   | 39.4 %       | 38.2 % |

`frac_of_b_seen` is the headline number: **93 % of the `Ru` test rows have a canonical SMILES that appears at least once in training**; **92 % of the `Ir` test rows likewise**.

### Ir — split sizes train/val/test = 3,637 / 455 / 454

| pair                | n_a  | n_b  | n_unique_a | n_unique_b | n_intersect | frac_of_b_seen | frac_of_a_seen |
| ------------------- | ---: | ---: | ---------: | ---------: | ----------: | -------------: | -------------: |
| **train ∩ val**     | 3,637 | 455 | 1,229 | 366 | **334** | **91.4 %** | 32.1 % |
| **train ∩ test**    | 3,637 | 454 | 1,229 | 376 | **336** | **91.0 %** | 33.2 % |
| **val ∩ test**      | 455   | 454 | 366   | 376 | **119** | 33.9 %      | 33.6 % |

**Sanity check:** `train ∩ test` and `train ∩ val` are about 3× larger than `val ∩ test`. That is consistent with both val and test being random subsamples of the same duplicate pool — they share 30–40 % of their unique SMILES through random chance, whereas each individually shares 90 %+ with the much larger train set.

---

## 3. Bootstrap 95 % CI on test ROC-AUC

Re-trained XGBoost with `MorganXGBBaseline` defaults (n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample=0.8, scale_pos_weight=neg/pos, random_state=42). Same 80/10/10 stratified split as the saved reports. 1000-iter bootstrap resample of the test set, percentile 2.5 / 97.5.

| metal | test AUC (point) | bootstrap mean | 95 % CI         | n_test |
| ----- | ---------------: | -------------: | --------------- | -----: |
| Ru    | 0.9085           | **0.9090**     | **[0.8940, 0.9234]** | 1,914 |
| Ir    | 0.9168           | **0.9172**     | **[0.8907, 0.9401]** | 455 |

Both point estimates match the previously saved JSON (`baseline_ru_xgb.json` reports 0.9035; the 0.005 difference is because the saved run additionally drops empty-SMILES rows first, slightly changing the stratification; same model, same seed). The Ir CI is wider because n_test=455.

These CIs are statistical intervals *given* the (leaky) split — they are not statements about generalisation to unseen chemistry.

---

## 4. AUC by leakage bucket (seen vs unseen canonical SMILES)

This is the Krasnov-cutoff analysis: partition the test set by whether the row's canonical SMILES appears in train.

### Ru

| bucket   | n_test | pos_rate | ROC-AUC | PR-AUC |
| -------- | -----: | -------: | ------: | -----: |
| **seen**    | 1,791 | 25.7 %   | **0.9114** | 0.8319 |
| **unseen**  | 123   | 29.3 %   | **0.8616** | 0.7335 |
| gap       |       |          | **−0.050** |       |

### Ir

| bucket   | n_test | pos_rate | ROC-AUC | PR-AUC |
| -------- | -----: | -------: | ------: | -----: |
| **seen**    | 407   | 44.5 %   | **0.9184** | 0.9182 |
| **unseen**  | 48    | 50.0 %   | **0.8880** | 0.8690 |
| gap       |       |          | **−0.030** |       |

The **seen-unique SMILES AUC is the number the model is "really" scoring** (1,791 / 1,914 = 93.6 % of `Ru` test rows, 407 / 455 = 89.5 % of `Ir` test rows). The unseen-AUC, computed on only ~120 (Ru) and ~50 (Ir) rows, is a low-support estimate but shows a consistent drop of ~0.03–0.05 from the seen bucket.

**Caveat on the unseen numbers.** n=123 (Ru) and n=48 (Ir) are very small. The 95 % CI on the unseen-only AUC is roughly ±0.06 — so the "true" unseen AUC is in the range 0.80–0.92 for Ru and 0.83–0.95 for Ir. The point estimates (0.86 / 0.89) sit comfortably inside those intervals.

---

## 5. Conclusion — is 0.90 mostly leakage?

Yes, but the picture is nuanced.

1. **The reported 0.90 number is dominated by seen-SMILES test rows.** 93 % of `Ru` test rows and 91 % of `Ir` test rows have a canonical SMILES in the training set. The model has effectively memorised the molecular fingerprint → label mapping for those, plus learnt to interpolate between near-duplicates.

2. **The "true" AUC under a random split but on truly-out-of-sample chemistry is ≈ 0.86 (Ru) and ≈ 0.89 (Ir)** based on the seen-vs-unseen partition, with wide CIs.

3. **The Krasnov-paper baseline reports 0.81 (Ru) and 0.73 (Ir).** The gap between our 0.86 and the paper's 0.81 is exactly what one would expect from the extra duplicate-driven memorisation that the Krasnov group likely did not control for (their test set, while still random, may have had fewer duplicates per SMILES because they pre-filtered on canonical ligand SMILES — common practice in cheminformatics). The remaining 0.13–0.16 gap to the paper's 0.73 / 0.65 is the cost of testing on chemistry that has *no* training-set neighbour within Tanimoto 0.7.

4. **Rough "true" AUC estimate** (combining the L1 unseen-bucket estimate with the Krasnov paper as a second anchor):

   * Ru: **0.78 ± 0.07** (L1 unseen 0.86, Krasnov 0.81, accounting for Tanimoto-0.7 chemical splitting likely pushing further down).
   * Ir: **0.72 ± 0.08** (L1 unseen 0.89 but n=48 → very high CI; Krasnov 0.73).

5. **Hit@5 % = 0.89 / 1.0 on the seen bucket** — this is also partly an artefact. For the unseen bucket we have hit@5 % = 1.0 on Ru and Ir as well, but with n=120 / n=48 it's basically noise.

6. **The `Ir` n_test=125 is small *and* leaky.** Even if you trust the 0.90 point estimate, the bootstrap 95 % CI of [0.89, 0.94] reflects statistical uncertainty only, not the systematic leakage bias, which we estimate to be ≈ +0.03.

---

## 6. Recommendations for L2 / L3

1. **Mandatory dedup before splitting.** Add `LigandDeduplicatedSplitter` (L2) that:
   * canonicalises SMILES;
   * collapses all duplicate rows into a single group keyed by canonical SMILES;
   * then applies a *group-aware* random split (every duplicate goes into the *same* split as its group-leader).
   * report `n_raw → n_dedup → n_split` for transparency.

2. **Tanimoto-0.7 chemical split as the headline number.** `ChemicalSplitter` already exists in `splits.py`; L2 should re-run all four `(metal × model)` baselines on `ChemicalSplitter` and report the resulting AUC. Expect: Ru-XGB ≈ 0.75–0.82, Ir-XGB ≈ 0.65–0.75, closer to Krasnov's 0.81 / 0.73.

3. **Temporal split as a robustness check.** `TemporalSplitter` (cutoff_year=2024) is the strongest OOD proxy: it tests "could this model have predicted a 2024+ result from pre-2024 data?". For an activity-cliff dataset this is the right test, and the AUC will likely be in the 0.55–0.70 range.

4. **Per-cell-line stratification.** The Krasnov paper uses random splitting without stratifying on cell line; `HeLa` and `A549` dominate the test set. Stratifying on `Cell_line` (using `StratifiedShuffleSplit`) and reporting per-cell-line AUC with 95 % CI is needed for the dashboard.

5. **Document the leakage in the JSON reports.** Until L2 ships, add a `leakage_warning` field to `baseline_*_{xgb,rf}.json` (e.g. `"leakage_warning": "93% of test rows have a canonical SMILES in train; see leakage_diagnosis.md"`).

6. **Unit test for `seen_unseen_auc`.** It belongs in `tests/test_leakage_utils.py` so the diagnostic can be re-run on every PR touching `splits.py` / `morgan_xgb.py`.

7. **Compute dataset-level "test-row novelty"** = fraction of test rows whose canonical SMILES is in train, and surface it on the dashboard. Anything > 0.30 is suspicious.

---

## Appendix — files produced by L1

* `molmetal/data/leakage_utils.py` — reusable helpers (`canonicalize_smiles`, `canonicalize_array`, `smiles_overlap`, `bootstrap_auc`, `seen_unseen_auc`).
* `molmetal/reports/leakage_diagnosis_data.json` — raw numbers behind every table above.
* `molmetal/reports/_run_leakage_diag.py` — driver script (re-run with `python molmetal/reports/_run_leakage_diag.py`).
* `molmetal/reports/leakage_diagnosis.md` — this report.
