# `molmetal.data.drugood_loader` — API doc

**Wf-313H / R20 Phase 0 loader closeout** (2026-09-29).

API documentation for `molmetal/data/drugood_loader.py` — the canonical-6-col
loader for the **DrugOOD (Ji et al. NeurIPS 2022) ChEMBL-29 IC50 subset**
(`lbap_refined_ic50_assay` train split = **95,373 rows**; scaffold + size
splits also exposed). For the canonical 6-col schema shared with
`load_binding_db` and `load_nci60`, see
`TODO/completed/313_r20_phase0_loaders_closeout.md` §3.

---

## §1 Public API

| Symbol | Kind | Purpose |
|---|---|---|
| `load_drugood(subset, split, target_root, prefer_library)` | function | Load + parse + canonicalise a DrugOOD IC50 split |
| `_resolve_subset_path(subset, target_root)` | helper | Resolve `<subset>.json` under `target_root` (or env-var override) |
| `_load_via_drugood_lib(subset, split, target_root)` | helper | Internal: build `drugood.datasets.DrugOODDataset` and flatten |
| `_load_via_raw_json(subset, split, target_root)` | helper | Internal: read DrugOOD JSON payload directly |
| `DEFAULT_DRUGOOD_TARGET_ROOT` | module-level `Path` | Default JSON root (`molmetal/.data/molmetal/drugood/ic50/official`) |
| `DEFAULT_SUBSETS` | module-level tuple | 3 refined curator outputs (assay / scaffold / size) |
| `VALID_SPLITS` | module-level tuple | 5 split keys (`train`, `ood_val`, `ood_test`, `iid_val`, `iid_test`) |
| `_CANONICAL_COLUMNS` | module-level tuple | The canonical 6-col schema tuple |
| `_NA_FILL` | module-level `str` | Literal string (`"NA"`) used to fill `metal` + `cell_line` columns |

---

## §2 `load_drugood(subset='lbap_refined_ic50_assay', split='train', target_root=None, prefer_library=True) -> pandas.DataFrame`

### Parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `subset` | `str` | `'lbap_refined_ic50_assay'` | One of `DEFAULT_SUBSETS` (warns but proceeds otherwise) |
| `split` | `str` | `'train'` | One of `VALID_SPLITS` (raises `ValueError` otherwise) |
| `target_root` | `Path | None` | `None` (use env-var → default) | Directory containing the curator's `<subset>.json` outputs |
| `prefer_library` | `bool` | `True` | When `True`, build upstream `drugood.datasets.DrugOODDataset`; on `ImportError`, falls back to raw JSON read |

### Env-var override

The default `target_root` is resolved via the `DRUGOOD_IC50_TARGET_ROOT`
environment variable if set (overriding `DEFAULT_DRUGOOD_TARGET_ROOT`).

### Returns

A `pandas.DataFrame` with **exactly** these 6 columns in this order:

```
smiles, metal, cell_line, protein, pIC50, active
```

| Column | Type | Description |
|---|---|---|
| `smiles` | `str` | Canonical SMILES (from upstream DrugOOD curator) |
| `metal` | `str` | Always `"NA"` (DrugOOD carries no per-row metal symbol) |
| `cell_line` | `str` | Always `"NA"` (DrugOOD carries no per-row cell-line identifier) |
| `protein` | `str` | Stringified `assay_id` (DrugOOD uses `assay_id` as the bioassay target proxy) |
| `pIC50` | `float` | `reg_label` cast to float (DrugOOD's IC50, −log10(molar)); rows with NaN pIC50 are dropped |
| `active` | `bool` | `cls_label` cast to bool (binary activity label at curator threshold, default pIC50 ≥ 5) |

### Raises

| Exception | When |
|---|---|
| `FileNotFoundError` | `target_root/<subset>.json` does not exist on disk |
| `ValueError` | `split` is not in `VALID_SPLITS`, or the JSON payload is missing the requested split key |
| `ImportError` | `prefer_library=True` AND `drugood` is not importable (rare; usually falls back to raw JSON automatically) |

### Side effects

* Calls `_resolve_subset_path(subset, target_root)` to locate `<subset>.json`.
* If `prefer_library=True` and `drugood` is importable: builds a
  `DrugOODDataset` and flattens `data_infos` into rows.  If
  `ImportError` occurs (e.g. running under a non-DrugOOD Python env),
  the loader **falls back to raw JSON read** automatically — the public
  API stays the same.
* If `prefer_library=False`: reads the JSON payload directly via
  `json.load(open(...))` (no `drugood` library import).
* Coerces `pIC50` to numeric via `pd.to_numeric(errors="coerce")`; rows with NaN pIC50 are dropped.
* Resets the row index (`reset_index(drop=True)`).

### Sample usage

```python
from molmetal.data.drugood_loader import load_drugood

# Raw JSON path (works without the drugood conda env installed)
df = load_drugood(
    subset="lbap_refined_ic50_assay",
    split="train",
    prefer_library=False,
)
print(df.columns.tolist())
# ['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
print(len(df))
# 95373
print(df["protein"].nunique())
# ≥100 unique assay ids

# Library path (requires mmcv + drugood installed in this env)
df2 = load_drugood()  # default args: subset=lbap_refined_ic50_assay, split=train, prefer_library=True
```

### Downstream consumers

* `molmetal/scripts/pre_featurize_dmpnn.py` — domain-id 1 (DrugOOD)
  multi-task featurisation.
* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN
  multi-task Wasserstein pIC50 regression.
* `molmetal/scripts/run_drugood_native_smoke.py` — Wf-296 native
  DrugOOD smoke test that exercises the raw-JSON fallback path.
* `molmetal/tests/test_drugood_3split.py` — 3-split cross-domain OOD
  benchmarks (assay / scaffold / size).

---

## §3 MEASURED anchors (Sep 28 2026)

| Property | Value | Source |
|---|---|---|
| Train row count (`lbap_refined_ic50_assay`) | **95,373** | `molmetal/.data/molmetal/drugood/ic50/official/lbap_refined_ic50_assay.json` (Ji 2022 official curator) |
| Default subset | **`lbap_refined_ic50_assay`** | `drugood_loader.py::load_drugood` default |
| Default split | **`train`** | same |
| `DEFAULT_SUBSETS` | **`(lbap_refined_ic50_assay, lbap_refined_ic50_scaffold, lbap_refined_ic50_size)`** | 3 refined curator outputs |
| `VALID_SPLITS` | **`(train, ood_val, ood_test, iid_val, iid_test)`** | 5 split keys |
| NA-fill columns | `metal`, `cell_line` (literal `"NA"`) | `drugood_loader.py::_NA_FILL` |
| Schema | **canonical 6-col** (`smiles, metal, cell_line, protein, pIC50, active`) | `drugood_loader.py::_CANONICAL_COLUMNS` |
| Tests | **16 dedicated tests** (this loader) + 1 in `test_data_loaders.py` + 1 cross-loader test = **18 tests** | `test_wf313h_drugood_loader.py` |

Honest framing: the **raw-JSON fallback path** is the **production
code path** used by every test + every smoke run on this host (the
`drugood` conda env is **NOT** the default Python). The library path
(`prefer_library=True`) is **OPTIONAL** and only used when callers
explicitly activate the DrugOOD conda env (`/home/hugo/.conda_envs/
drugood/bin/python`). Both paths converge on the **same canonical
schema** — the GATE-6 contract is preserved across both code paths.
