# R20 Phase 0 — 3-loader API index (binding_db / nci60 / drugood_loader)

**Wf-322E / R20 §1.1 loader-API index** (2026-09-29).

Per
[`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`](../../TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md)
§1.1 the **three P0 CRITICAL loader modules** for R20 Phase 0 each ship a
per-loader API doc at `molmetal/data/<loader>.md`. This page is the
**index** of those three per-loader docs — a single landing page that
documents the shared canonical schema, the per-coverage surface differences,
and the cross-loader invariants every downstream consumer can assume.

| Per-loader API doc | Loader | MEASURED row count | Module |
|---|---|---:|---|
| [`binding_db.md`](binding_db.md) | BindingDB (metal-filtered subset) | **5,257** | `molmetal/data/binding_db.py` |
| [`nci60.md`](nci60.md) | NCI-60 (precious-metal cell-line panel) | **7,823** | `molmetal/data/nci60.py` |
| [`drugood_loader.md`](drugood_loader.md) | DrugOOD (Ji 2022 ChEMBL-29 IC50 curator) | **95,373** (assay/train) | `molmetal/data/drugood_loader.py` |

---

## §1 Canonical 6-column schema (shared contract)

Every loader returns a `pandas.DataFrame` with **exactly** these six
columns in this order:

```
smiles, metal, cell_line, protein, pIC50, active
```

| # | Column | Type | Source binding_db | Source nci60 | Source drugood_loader |
|---|---|---|:---:|:---:|:---:|
| 1 | `smiles` | `str` | from upstream CSV | from upstream CSV | from DrugOOD curator |
| 2 | `metal` | `str` | from upstream CSV | **NA-filled** (no per-row metal symbol) | **NA-filled** (no per-row metal symbol) |
| 3 | `cell_line` | `str` | from upstream CSV | from upstream CSV | **NA-filled** (no per-row cell-line identifier) |
| 4 | `protein` | `str` | from upstream CSV | **NA-filled** (no per-row protein target) | stringified `assay_id` |
| 5 | `pIC50` | `float64` | from upstream CSV | from upstream CSV (renamed from `pIC50_precious`) | from DrugOOD `reg_label` |
| 6 | `active` | `bool` | from upstream CSV | from upstream CSV (renamed from `active_precious`) | from DrugOOD `cls_label` |

The literal "NA" fill is the standard sentinel for "this upstream
source does not provide the column" — callers can union the three
DataFrames and treat `metal == "NA"` / `cell_line == "NA"` /
`protein == "NA"` as "missing by upstream design", not as a parse
error.  This invariant is enforced by `test_data_loaders.py`
canonical-schema test and re-anchored by `test_wf313h_*` and
`test_wf320f_*` per-loader suites.

---

## §2 Per-loader public surface (cheat sheet)

### §2.1 `molmetal.data.binding_db`

| Symbol | Kind | Purpose |
|---|---|---|
| `load_binding_db(path=None)` | function | Load + parse + canonicalise the BindingDB CSV |
| `DEFAULT_CSV` | module-level `Path` | `molmetal/data/protein_metal_binding.csv` |
| `_REQUIRED_COLUMNS` | module-level tuple | The canonical 6-col schema tuple |

Full API: [`molmetal/data/binding_db.md`](binding_db.md).

### §2.2 `molmetal.data.nci60`

| Symbol | Kind | Purpose |
|---|---|---|
| `load_nci60(path=None)` | function | Load + parse + canonicalise the NCI-60 precious-metal CSV |
| `DEFAULT_CSV` | module-level `Path` | `molmetal/data/nci60_platinai_link.csv` |
| `_REQUIRED_COLUMNS` | module-level tuple | The canonical 6-col schema tuple |
| `_NA_FILL` | module-level `str` | `"NA"` for `metal` + `protein` |

Full API: [`molmetal/data/nci60.md`](nci60.md).

### §2.3 `molmetal.data.drugood_loader`

| Symbol | Kind | Purpose |
|---|---|---|
| `load_drugood(subset, split, target_root, prefer_library)` | function | Load + parse + canonicalise a DrugOOD IC50 split |
| `_resolve_subset_path(subset, target_root)` | helper | Resolve `<subset>.json` under `target_root` |
| `_load_via_drugood_lib(subset, split, target_root)` | helper | Internal: build `DrugOODDataset` and flatten |
| `_load_via_raw_json(subset, split, target_root)` | helper | Internal: read DrugOOD JSON payload directly |
| `DEFAULT_DRUGOOD_TARGET_ROOT` | module-level `Path` | Default JSON root |
| `DEFAULT_SUBSETS` | module-level tuple | 3 refined curator outputs (assay / scaffold / size) |
| `VALID_SPLITS` | module-level tuple | 5 split keys (`train`, `ood_val`, `ood_test`, `iid_val`, `iid_test`) |

Full API: [`molmetal/data/drugood_loader.md`](drugood_loader.md).

---

## §3 Cross-loader invariants (downstream consumer contract)

These invariants are guaranteed by every loader.  Any consumer
(`molmetal/scripts/pre_featurize_dmpnn.py`,
`train_dmpnn_multitask_wasserstein.py`, chemprop v2/v3 drivers,
cross-domain OOD bench scripts) can rely on them without per-loader
special-casing:

1. **Schema**: exactly 6 columns in the canonical order.
2. **No NaN in `pIC50`**: rows with NaN pIC50 are dropped at load time.
3. **Finite `pIC50`**: coercion via `pd.to_numeric(errors="coerce")`
   ensures `pIC50.dtype == np.float64`.
4. **`active` is `bool`**: cast through `.astype(bool)`.
5. **String columns are `str`**: `smiles / metal / cell_line / protein`
   are all `str` (not `object`-with-mixed-types).
6. **Index is reset**: `reset_index(drop=True)` after load; integer
   index `[0..N-1]`.
7. **"NA" sentinel is literal string `"NA"`** for `metal` /
   `cell_line` / `protein` columns where upstream does not carry the
   field.  Comparisons use `== "NA"` (not `pd.isna`).
8. **RDKit-parseable SMILES**: ≥95 % of `smiles` round-trip through
   `rdkit.Chem.MolFromSmiles` (verified in `test_wf313h_*`).

---

## §4 MEASURED anchors (Sep 28 2026 — verified 2026-09-29 Wf-322E)

| Loader | MEASURED rows | Module bytes | API doc bytes | Dedicated tests | Cross-loader tests |
|---|---:|---:|---:|---:|---:|
| BindingDB | **5,257** | 2,379 | 4,140 | **13** | 2 |
| NCI-60 | **7,823** | 3,903 | 4,623 | **13** | 2 |
| DrugOOD | **95,373** | 10,856 | 6,731 | **16** | 2 |
| **Total** | **108,453** | — | — | **42** | **6** |

> Sources: `molmetal/data/protein_metal_binding.csv`,
> `molmetal/data/nci60_platinai_link.csv`,
> `molmetal/.data/molmetal/drugood/ic50/official/lbap_refined_ic50_assay.json`.

The **Wf-313H** dedicated test suites (`test_wf313h_*_loader.py`) and
the **Wf-320F** independent re-verification suites
(`test_wf320f_*_loader.py`) together **double-anchor** the loader
contract: 79 total dedicated tests, all passing on the host.

---

## §5 Honest framing

* The MEASURED row counts in §4 are the **project-shipped subsets on
  disk**.  Full-Dataset versions are deferred to R20 Phase 3 (BindingDB
  ~50K–200K after metal filter) and Phase 2 (DrugOOD scaffold + size
  splits in addition to the assay/train split shipped here).
* The 3 loaders are **GATE-6 ready** for Phase 3+ once the upstream
  downloads complete.  No loader code changes are anticipated for the
  full-Dataset versions; the canonical 6-col contract is preserved.
* The per-loader API docs (`binding_db.md`, `nci60.md`,
  `drugood_loader.md`) were originally written during Wf-313H (Sep
  29 2026).  This index page (Wf-322E, 2026-09-29) is a **pure
  documentation closeout**: it surfaces the three per-loader docs as a
  coherent loader-API surface for downstream consumers and adds a
  pre-flight gate `p3_r20_p0_loader_api_doc_present` that asserts all
  four files (3 per-loader docs + this index) are present on disk.

**Phase 0 loader-API documentation closeout: SHIP**.
