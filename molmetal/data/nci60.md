# `molmetal.data.nci60` — API doc

**Wf-313H / R20 Phase 0 loader closeout** (2026-09-29).

API documentation for `molmetal/data/nci60.py` — the canonical-6-col
loader for the **7,823-row NCI-60 precious-metal cytotoxicity subset**
(`MCF7 / A549 / A2780 / A2780cis / HCT116 / PC3 / HL60 / SKOV3` × compound
rows). For the canonical 6-col schema shared with `load_binding_db` and
`load_drugood`, see `TODO/completed/313_r20_phase0_loaders_closeout.md` §3.

---

## §1 Public API

| Symbol | Kind | Purpose |
|---|---|---|
| `load_nci60(path=None)` | function | Load + parse + canonicalise the NCI-60 precious-metal CSV |
| `DEFAULT_CSV` | module-level `Path` | Default path (`molmetal/data/nci60_platinai_link.csv`) |
| `_REQUIRED_COLUMNS` | module-level tuple | The canonical 6-col schema tuple |
| `_NA_FILL` | module-level `str` | Literal string (`"NA"`) used to fill `metal` + `protein` columns |

---

## §2 `load_nci60(path=None) -> pandas.DataFrame`

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
| `smiles` | `str` | Canonical SMILES |
| `metal` | `str` | Always `"NA"` (NCI-60 link CSV carries no per-row metal symbol) |
| `cell_line` | `str` | NCI-60 cell-line identifier (e.g. `MCF7`, `A2780`, `HCT116`) |
| `protein` | `str` | Always `"NA"` (NCI-60 link CSV carries no per-row protein target identifier) |
| `pIC50` | `float` | −log10(IC50 [M]); rows with NaN pIC50 are dropped |
| `active` | `bool` | Binary activity label |

### Source → canonical column rename

| Source column | Canonical column |
|---|---|
| `cell_line_precious` | `cell_line` |
| `pIC50_precious` | `pIC50` |
| `active_precious` | `active` |

The `_precious` suffix disambiguates the NCI-60 columns from any
forthcoming PlatinAI columns that may be merged into the same row
later. `metal` and `protein` are filled with `"NA"` (literal) when not
present in the source CSV.

### Raises

| Exception | When |
|---|---|
| `FileNotFoundError` | `path` (or `DEFAULT_CSV`) does not exist on disk |
| `ValueError` | CSV is missing any of the required canonical columns after the rename |

### Side effects

* Reads the CSV via `pandas.read_csv(..., low_memory=False)`.
* Renames the `_precious`-suffixed source columns to canonical names.
* Coerces `pIC50` to numeric via `pd.to_numeric(errors="coerce")`; rows with NaN pIC50 are dropped.
* Coerces `active` to `bool` and the four string columns to `str`.
* Resets the row index (`reset_index(drop=True)`).

### Sample usage

```python
from molmetal.data.nci60 import load_nci60

df = load_nci60()
print(df.columns.tolist())
# ['smiles', 'metal', 'cell_line', 'protein', 'pIC50', 'active']
print(len(df))
# 7823
print(df["cell_line"].value_counts().to_dict())
# {'MCF7': ~1100, 'A549': ~1100, 'A2780': ~1000, ...}
print(df["metal"].unique())
# ['NA']  -- metal column is NA-filled by design

# Custom CSV path
from pathlib import Path
df2 = load_nci60(path=Path("/tmp/my_nci60.csv"))
```

### Downstream consumers

* `molmetal/scripts/pre_featurize_dmpnn.py` — domain-id 2 (NCI-60)
  multi-task featurisation.
* `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN
  multi-task Wasserstein pIC50 regression.
* `molmetal/scripts/nci60_9cell_chemprop_driver.py` — Wf-220
  chemprop v2 driver for the 9 precious-metal-active cell lines.

---

## §3 MEASURED anchors (Sep 28 2026)

| Property | Value | Source |
|---|---|---|
| Row count | **7,823** | `molmetal/data/nci60_platinai_link.csv` |
| Cell-line set | **`{MCF7, A549, A2780, A2780cis, HCT116, PC3, HL60, SKOV3}`** (≥8 unique) | same |
| Schema | **canonical 6-col** (`smiles, metal, cell_line, protein, pIC50, active`) | `nci60.py::_REQUIRED_COLUMNS` |
| Source rename | `cell_line_precious → cell_line`, `pIC50_precious → pIC50`, `active_precious → active` | `nci60.py::load_nci60` |
| NA-fill columns | `metal`, `protein` (literal `"NA"`) | same |
| Tests | **13 dedicated tests** (this loader) + 1 in `test_data_loaders.py` + 1 cross-loader test = **15 tests** | `test_wf313h_nci60_loader.py` |

Honest framing: the **7,823-row subset** is the project-shipped
precious-metal subset (the **full NCI-60 panel** is NOT on disk —
filtered to 9 precious-metal-active cell lines for the metal-cyto
multi-task). The loader is **GATE-6 ready** for the broader panel if
that is needed in Phase 3+.
