# `molmetal.data.binding_db` — API doc

**Wf-313H / R20 Phase 0 loader closeout** (2026-09-29).

API documentation for `molmetal/data/binding_db.py` — the canonical-6-col
loader for the **5,257-row metal-filtered BindingDB subset** (`Ru / Ir /
Re / Os / Rh` × small set of protein targets). For the canonical 6-col
schema shared with `load_nci60` and `load_drugood`, see
`TODO/completed/313_r20_phase0_loaders_closeout.md` §3.

---

## §1 Public API

| Symbol | Kind | Purpose |
|---|---|---|
| `load_binding_db(path=None)` | function | Load + parse + canonicalise the BindingDB CSV |
| `DEFAULT_CSV` | module-level `Path` | Default path (`molmetal/data/protein_metal_binding.csv`) |
| `_REQUIRED_COLUMNS` | module-level tuple | The canonical 6-col schema tuple |

---

## §2 `load_binding_db(path=None) -> pandas.DataFrame`

### Parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | `Path | str | None` | `None` (i.e. use `DEFAULT_CSV`) | Optional explicit CSV path. |

### Returns

A `pandas.DataFrame` with **exactly** these 6 columns in this order:

```
smiles, metal, cell_line, protein, pIC50, active
```

| Column | Type | Description |
|---|---|---|
| `smiles` | `str` | Canonical SMILES (from upstream BindingDB) |
| `metal` | `str` | D-block metal symbol; one of `Ru / Ir / Re / Os / Rh` |
| `cell_line` | `str` | Cell-line identifier (e.g. `MCF-7`, `HCT-116`) |
| `protein` | `str` | Protein target identifier (cell-line proxy here; same as `cell_line` for the project-shipped CSV) |
| `pIC50` | `float` | −log10(IC50 [M]); rows with NaN pIC50 are dropped |
| `active` | `bool` | Binary activity label |

### Raises

| Exception | When |
|---|---|
| `FileNotFoundError` | `path` (or `DEFAULT_CSV`) does not exist on disk |
| `ValueError` | CSV is missing any of the required columns (`smiles, metal, cell_line, protein, pIC50, active`) |

### Side effects

* Reads the CSV via `pandas.read_csv(..., low_memory=False)`.
* Coerces `pIC50` to numeric via `pd.to_numeric(errors="coerce")`; rows with NaN pIC50 are dropped.
* Coerces `active` to `bool` and the four string columns to `str`.
* Resets the row index (`reset_index(drop=True)`).

### Sample usage

```python
from molmetal.data.binding_db import load_binding_db

df = load_binding_db()
print(df.columns.tolist())
# ['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
print(len(df))
# 5257
print(df["metal"].value_counts().to_dict())
# {'Ru': ~1100, 'Ir': ~1100, 'Re': ~1000, 'Os': ~1000, 'Rh': ~1000}
print(df[df["metal"] == "Ru"]["pIC50"].describe())
# count ~1100, mean ~5.5, std ~1.5, ...

# Custom CSV path
from pathlib import Path
df2 = load_binding_db(path=Path("/tmp/my_bindingdb.csv"))
```

### Downstream consumers

* `molmetal/scripts/pre_featurize_dmpnn.py` — domain-id 3 (BindingDB)
  multi-task featurisation.
* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN
  multi-task Wasserstein pIC50 regression with per-metal task heads.
* Any chemprop v2 driver reading the canonical 6-col schema via
  `molmetal/data/__init__.py::load_binding_db`.

---

## §3 MEASURED anchors (Sep 28 2026)

| Property | Value | Source |
|---|---|---|
| Row count | **5,257** | `molmetal/data/protein_metal_binding.csv` |
| Metal set | **`{Ru, Ir, Re, Os, Rh}`** (5 d-block) | same |
| Schema | **canonical 6-col** (`smiles, metal, cell_line, protein, pIC50, active`) | `binding_db.py::_REQUIRED_COLUMNS` |
| Coercion | `pIC50 → float64`, `active → bool`, string cols → `str` | `binding_db.py::load_binding_db` |
| Tests | **13 dedicated tests** (this loader) + 1 in `test_data_loaders.py` + 1 cross-loader test = **15 tests** | `test_wf313h_binding_db_loader.py` |

Honest framing: the **5,257-row subset** is the project-shipped artefact
on disk; the **full BindingDB** is **NOT on disk** (per
`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.2 P0 BLOCKER —
download TSV.gz from `bindingdb.org`, filter to metal-binding entries,
expect ~50K-200K rows after metal filter). The loader is **GATE-6 ready**
to consume the full BindingDB once the TSV.gz download completes (see
Phase 3 of the R20 plan).
