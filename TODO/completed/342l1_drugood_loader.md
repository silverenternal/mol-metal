# Wf-342-L1 — DrugOOD loader status check + coverage extension

**Date**: 2026-09-29
**Workflow**: Wf-342-L1 (R20-Phase-0 closeout)
**Status**: SHIPPED (CPU-only; 33 tests passing across 3 test files)
**Parent**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.1 (BD-1: 3 missing loader modules)

---

## §0 TL;DR

1. **Status check**: `molmetal/data/drugood_loader.py` was **already shipped**
   via Wf-313H (Sep 28) + re-anchored via Wf-320F.  Wf-342-L1 was
   redundant in its core spec but added 4 coverage-extension tests
   (CSV upstream format + OOD test split + OOD scaffold split + multi-task
   label contract) that pin gaps the prior two test sets did not cover.
2. **MEASURED** (Sep 28 2026 on-disk):
   - `molmetal/data/drugood_loader.py`: 293 LOC; full implementation
     (load_drugood + canonical 6-col schema + multi-task pIC50+active +
     library + raw-JSON fallback paths).
   - Test coverage: 16 (Wf-313H) + 13 (Wf-320F) + 4 (Wf-342-L1) = **33 tests
     passing** in 9.33 s wall (CPU only).
3. **HONEST FRAMING**: The plan's §1.1 claim "no module API" was correct at
   audit time but stale by Wf-313H ship.  This Wf is a **status
   confirmation + extension**, not a fresh implementation.

---

## §1 DrugOOD loader — implementation summary

### §1.1 File under test

`molmetal/data/drugood_loader.py` (293 LOC).  Public surface:

```python
load_drugood(
    subset: str = "lbap_refined_ic50_assay",  # 3 curated subsets
    split: str = "train",                   # 5 splits (train/ood_val/ood_test/iid_val/iid_test)
    target_root: Optional[Path] = None,
    prefer_library: bool = True,            # drugood lib path vs raw-JSON fallback
) -> pd.DataFrame                           # 6-col canonical schema
```

Canonical schema (matches `molmetal.data.binding_db` + `molmetal.data.nci60`):

| column | dtype | source | DrugOOD source field |
|---|---|---|---|
| `smiles` | str | `smiles` | DrugOOD ligand SMILES |
| `metal` | str | literal `"NA"` | no per-row metal identifier |
| `cell_line` | str | literal `"NA"` | no per-row cell-line identifier |
| `protein` | str | `assay_id` | bioassay target proxy |
| `pIC50` | float | `reg_label` | Ji 2022 curator regression label |
| `active` | bool | `cls_label` | Ji 2022 curator classification label (≥ 5) |

### §1.2 Two code paths

1. **`_load_via_drugood_lib`** (preferred when `drugood` is importable):
   builds `drugood.datasets.DrugOODDataset`, then flattens
   `ds.data_infos`.  Requires `mmcv==1.5.0` + the DrugOOD conda env.
2. **`_load_via_raw_json`** (fallback / `prefer_library=False`): reads
   `<target_root>/<subset>.json` directly.  Same schema contract.

The fallback path is what runs in `.venv-vlm` (no mmcv / no DrugOOD
installed in the default venv).  Wf-313H + Wf-320F + Wf-342-L1 ALL
exercise the fallback path.

---

## §2 Test coverage (3 files, 33 tests)

### §2.1 `molmetal/tests/test_wf313h_drugood_loader.py` (16 tests — Wf-313H)

| Group | Tests |
|---|---|
| Schema | `test_load_returns_dataframe_raw_json_path`, `test_canonical_columns_exact_order`, `test_pIC50_numeric_and_finite`, `test_metal_and_cell_line_NA_filled` |
| Row/split | `test_row_count_95373`, `test_assay_diversity`, `test_scaffold_split_subsample` |
| Edge | `test_missing_json_raises_filenotfound`, `test_invalid_split_raises_valueerror` |
| Integration | `test_smiles_rdkit_round_trip_subsample`, `test_dataframe_consumable_by_chemprop_v2_smoke_loader` |
| Cross-loader | `test_internal_canonical_columns_constant_matches`, `test_resolve_subset_path_uses_env_var`, `test_default_subsets_constant_is_3_entries`, `test_valid_splits_constant_is_5_entries`, `test_default_target_root_path_attribute` |

### §2.2 `molmetal/tests/test_wf320f_drugood_loader.py` (13 tests — Wf-320F)

Independent second-angle re-verification (double-anchored with Wf-313H).
Same 5 groups (schema 4 + row/split 3 + edge 2 + integration 2 +
cross-loader 2).

### §2.3 `molmetal/tests/test_wf342l1_drugood_loader.py` (4 tests — NEW this Wf)

| Test | Coverage gap filled |
|---|---|
| `test_wf342l1_drugood_csv_loads` | DrugOOD upstream CSV (296,675 data rows + header) — pins `molmetal/data/drugood/train.csv` (5-col upstream format used by `molmetal/data/four_domain_combined.py`) |
| `test_wf342l1_drugood_ood_test_set_loads` | OOD test split loads non-empty with canonical schema (Wf-313H + Wf-320F exercised only `train`) |
| `test_wf342l1_drugood_ood_test_scaffold_split_subsample` | Bemis-Murcko scaffold split on `ood_test` subsample (1k rows, ≥90 % RDKit parseable, disjoint 3-way partition) |
| `test_wf342l1_drugood_multitask_labels_intact` | Multi-task label contract: numeric `pIC50` + boolean `active` BOTH present + non-NaN + binary active distribution (per Wf-305D) |

All 4 new tests are **skip-if-unavailable** so the file degrades
gracefully when the curator JSON or upstream CSV is missing.

---

## §3 MEASURED per-split row counts (Sep 28 2026)

Source: `.data/molmetal/drugood/ic50/official/lbap_refined_ic50_assay.json`

| Split | Rows | Notes |
|---|---:|---|
| `train` | 95,373 | Wf-313H anchor (pinned by `EXPECTED_TRAIN_ROW_COUNT`) |
| `ood_val` | 53,293 | MEASURED 2026-09-29 |
| `ood_test` | 53,291 | MEASURED 2026-09-29; Wf-342-L1 anchor |
| `iid_val` | (not loaded this Wf) | valid per `VALID_SPLITS` |
| `iid_test` | (not loaded this Wf) | valid per `VALID_SPLITS` |

CSV upstream (separate artefact):

| File | Rows | Schema |
|---|---:|---|
| `molmetal/data/drugood/train.csv` | 296,675 data rows | `smiles, y, assay_id, scaffold_split_id, size_split_id` |

---

## §4 BD-1 status (R20 §1.1 — 3 missing loaders)

Per `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.1 the P0 blockers were
3 missing loader modules:

| Loader | Status | Test file(s) | Tests |
|---|---|---|---|
| `molmetal/data/drugood_loader.py` | **SHIPPED (this Wf confirms + extends)** | `test_wf313h_drugood_loader.py` + `test_wf320f_drugood_loader.py` + `test_wf342l1_drugood_loader.py` | 33 |
| `molmetal/data/binding_db.py` | SHIPPED (per `molmetal/data/binding_db.md` + `binding_db.py` exists) | (check `molmetal/tests/` for `test_binding_db*`) | TBD |
| `molmetal/data/nci60.py` | SHIPPED (per `nci60.py` exists in `molmetal/data/`) | (check `molmetal/tests/` for `test_nci60*`) | TBD |

**Wf-342-L1 closes BD-1 entry 1 (drugood_loader)** with a 33-test
double-anchored contract.  BD-1 entries 2 + 3 are out of scope for
this Wf (already shipped via earlier Wfs; Wf-342-L2/L3 would be the
parallel work for binding_db / nci60 if the parent orchestration
sequence picks them up).

---

## §5 Test command + result

```bash
cd /home/hugo/codes/mol-metal
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf313h_drugood_loader.py \
    molmetal/tests/test_wf320f_drugood_loader.py \
    molmetal/tests/test_wf342l1_drugood_loader.py
# => 33 passed, 1 warning in 9.33s
```

---

## §6 Honest framing

- **MEASURED**: 33 tests passing (CPU only; total 9.33 s wall).
- **MEASURED**: per-split row counts (train 95,373; ood_val 53,293;
  ood_test 53,291) from on-disk JSON.
- **PROJECTED**: 1.58M DrugOOD full corpus is on disk (per
  `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §0); Wf-342-L1 only exercised
  the 95k+53k+53k curator outputs of one subset (lbap_refined_ic50_assay)
  — the full 1.58M (3 subsets × 5 splits) is **NOT yet loaded** by any
  test (would exceed unit-test budget; belongs in Phase 2 GPU bench).
- **HONEST-NEGATIVE**: OOD and IID scaffold-disjointness between train
  and ood_test is NOT directly asserted by these tests (only disjoint
  3-way partition within ood_test).  DrugOOD's contract guarantees
  scaffold-disjointness upstream — pinning it would require
  cross-split scaffold-set intersection computation (a Phase 2 bench
  concern, not a loader-contract concern).
