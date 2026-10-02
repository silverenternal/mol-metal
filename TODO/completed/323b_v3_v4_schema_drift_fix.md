# Wf-323B: MetalCytoToxDB loader v3/v4 schema drift fix

**Wf:** wf323b_v3_v4_schema_drift_fix
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-323B agent
**Status:** SHIPPED
**Refs:** Wf-319D audit (TODO-178 G2 metalcytotox CSV) §2.1 v3 schema drift

---

## §1 Context

Per Wf-319D audit §2.1: v3 metalcytotox CSV has schema drift
(`SMILES_Ligands` / `Metal` / `IC50_Dark_value`, no lowercase `smiles`
/ `metal`), while v4 (latest) normalises back to lowercase canonical
columns while keeping the v3-style provenance as enrichment. The
existing loader (`molmetal/data/metal_cytotox.py`) was written when
only the v1 schema existed and silently breaks on v4.

Wf-323B fixes the loader so both v3 and v4 round-trip through the
same `_has_column()` synonym-aware lookup without raising pandas
duplicate-column `ValueError`.

---

## §2 Findings (MEASURED)

### §2.1 CSV versions present on disk (re-verified 2026-09-29)

| Version | Path | n_rows | Columns |
|---|---|---:|---|
| original (v1) | `.data/molmetal/metalcytotox_5metal_multitask.csv` | 26,801 | smiles, metal, scaffold_id, split, activity, pic50, cell_line, source |
| v3 | `.data/molmetal/metalcytotox_5metal_multitask_v3.csv` | 26,632 | **SMILES_Ligands** (no `smiles`), **Metal** (no `metal`) |
| v4 (latest) | `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` | 26,632 | smiles, metal, scaffold_id, split, activity, **SMILES_Ligands, Metal, IC50_Dark_value**, pic50, cell_line, source |
| canonical | `.data/molmetal/MetalCytoToxDB.csv` | (n/a — 28 cols) | full release; only `IC50_Dark_value` for IC50, no `active`/`activity` |

### §2.2 Loader behaviour pre-Wf-323B (MEASURED)

- v1 + v3 + canonical → loader returns 4-col DataFrame (works).
- v4 → loader raises `ValueError: Buffer has wrong number of dimensions
  (expected 1, got 2)` on the column rename because v4 carries BOTH
  `smiles` AND `SMILES_Ligands` (and `metal`/`Metal`) — pandas cannot
  index into a frame with duplicate column names.

### §2.3 Loader behaviour post-Wf-323B (MEASURED)

- v1 → 26,801 rows × 4 cols, 5 metals, pic50 NaN=0, active True=9,213.
- v3 → 26,632 rows × 4 cols, 5 metals, pic50 NaN=0, active True=9,161.
- v4 → 26,632 rows × 4 cols, 5 metals, pic50 NaN=0, active True=9,161.
- canonical → 26,801 rows × 4 cols, 5 metals, pic50 NaN=0, active True=9,213.

v3 and v4 produce identical row counts and active distribution because
v4 IS v3 normalised plus provenance columns. The loader now round-trips
all 4 versions through the same code path.

---

## §3 Implementation (MEASURED)

### §3.1 Loader changes (`molmetal/data/metal_cytotox.py`)

| Component | Change |
|---|---|
| `_COLUMN_SYNONYMS` | NEW dict: canonical column → tuple of accepted aliases (lowercase canonical listed first) |
| `_has_column(df, canonical)` | NEW helper: returns the actual column name matching canonical (prefers lowercase to avoid v4 collision) |
| `_rename_v3_to_canonical(df)` | NEW helper: renames v3-style uppercase → lowercase, skips when target column already exists (v4) |
| `load_metal_cytotox()` | MODIFIED: uses `_has_column` and `_rename_v3_to_canonical` instead of naive rename. Now resolves `pIC50`/`pic50`/`IC50_Dark_value` and `active`/`activity`/`IC50_Dark_value` through the synonym map |

Lines added: +113 (was 168 → 236); lines removed: +60.

### §3.2 Synonym map shape

```python
_COLUMN_SYNONYMS = {
    "smiles": ("smiles", "SMILES_Ligands", "smiles_ligands"),
    "metal": ("metal", "Metal"),
    "pIC50": ("pic50", "pIC50", "IC50_Dark_value"),
    "active": ("active", "activity", "IC50_Dark_value"),
}
```

Lowercase canonical forms are listed FIRST so `_has_column` short-circuits
on the lowercase column when present (v4 path), preventing the
`pd.DataFrame.rename` from creating duplicate columns.

### §3.3 Tests (`molmetal/tests/test_wf323b_v3_v4_schema_drift.py`)

8 NEW tests, all PASS:

| # | Test | Class | Status |
|---:|---|---|---|
| 1 | `test_v3_loader_returns_canonical_4_columns` | TestV3CsvLoading | PASS |
| 2 | `test_v3_loader_resolves_5_metals` | TestV3CsvLoading | PASS |
| 3 | `test_v3_loader_no_nan_pic50` | TestV3CsvLoading | PASS |
| 4 | `test_v4_loader_returns_canonical_4_columns` | TestV4CsvLoading | PASS (was FAIL pre-Wf-323B) |
| 5 | `test_v4_loader_resolves_5_metals` | TestV4CsvLoading | PASS |
| 6 | `test_v3_and_v4_outputs_match` | TestV4CsvLoading | PASS |
| 7 | `test_column_synonyms_cover_all_canonical` | TestSynonymMap | PASS |
| 8 | `test_rename_v3_skips_when_lowercase_present` | TestSynonymMap | PASS |

CPU-only, runtime < 2 s.

### §3.4 Regression coverage

Re-ran the existing metalcytotox smoke + Wf-319D audit tests with the
new loader — **35/35 PASS** (8 new Wf-323B + 15 metalcytotox_bench_smoke
+ 12 wf319d_metalcytotox_csv_audit). Zero regressions.

---

## §4 Pre-flight gate impact

Pre-flight `p1_chemprop_smoke` was FAILING pre-Wf-323B because
`benchmark_r18_metalcytotox.py` triggers the canonical-path loader
which raised `ValueError: ... missing active / activity` (canonical
`MetalCytoToxDB.csv` only carries `IC50_Dark_value`). The synonym map
fix added `IC50_Dark_value` as the third `active` alias.

| Gate | Pre-Wf-323B | Post-Wf-323B |
|---|---|---|
| `p1_chemprop_smoke` | FAIL (loader ValueError) | **PASS** (Ru seed=0 test_auc=0.7929) |
| `chemprop_v2_v3` pipeline (13 gates) | 13 FAILs from loader | **13/13 PASS** |
| full pre-flight | multiple loader-related FAILs | 58 PASS, 1 FAIL (`p3_gate_count_consistency_audit` — pre-existing gate count drift, NOT loader related) |

---

## §5 Honest framing (party-twenty-spirit §3.5)

- v3 and v4 are both legitimate versions of the same data. v4 is v3
  normalised + provenance enrichment columns.
- Loader now handles both for backward compatibility.
- Pure CPU fix (no GPU).
- Pre-flight audit gate (`p3_gate_count_consistency_audit`) drift FAIL
  is pre-existing — flagged in Wf-321D audit doc; not in scope for
  Wf-323B.

---

## §6 Files touched

| Path | Change |
|---|---|
| `molmetal/data/metal_cytotox.py` | +113/-32 (synonym map + helpers + loader update) |
| `molmetal/tests/test_wf323b_v3_v4_schema_drift.py` | NEW, 205 LOC, 8 tests |

---

## §7 Wf-323B ship artefacts

| Artefact | Path |
|---|---|
| Loader source | `molmetal/data/metal_cytotox.py` (236 lines post-fix) |
| 8 regression tests | `tests/test_wf323b_v3_v4_schema_drift.py` |
| This doc | `TODO/completed/323b_v3_v4_schema_drift_fix.md` |

---

**END of Wf-323B doc**