# Wf-344A1: pin metal_cytotox loader import path + canonical 4-column contract

**Wf:** wf344a1_metal_cytotox_import
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-344A1 agent
**Status:** SHIPPED
**Refs:** Wf-338-A3 finding 1 (pre-flight `p1_chemprop_smoke` FAIL on
`metal_cytotox.py:217`); Wf-323B loader rewrite (synonym-aware +
derive-active fallback); Wf-327F2 schema drift invariants (4-test
verification layer)

---

## §1 Context

Per Wf-338-A3 finding 1: pre-flight gate `p1_chemprop_smoke` (TODO-207
§4.2 P1) FAILed because `train_chemprop_v2_v3 --quick` raised at
`molmetal/data/metal_cytotox.py:217`:

```
ValueError: /home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv
is missing active / activity; present columns: ['smiles', 'Counterion',
..., 'IC50_Dark_value', ..., 'metal', ...]
```

The CSV at `.data/molmetal/MetalCytoToxDB.csv` is the **canonical**
MetalCytoToxDB release (24 columns, no `active` / `activity` lowercase
column, only `IC50_Dark_value` for IC50).  Per Wf-323B ship surface
(`TODO/completed/323b_v3_v4_schema_drift_fix.md`), the loader was
rewritten with synonym-aware lookup so the `active` synonym chain
`("active", "activity", "IC50_Dark_value")` falls back to the
`IC50_Dark_value` derive path via `_derive_active(df)`.  Wf-327F2
added 4 verification tests.

**Wf-344A1** adds an additive 2-test layer that pins the **import
path** + **canonical 4-column contract** specifically for the
MetalCytoToxDB.csv the Wf-338-A3 finding flagged.

---

## §2 Findings (MEASURED)

### §2.1 Current loader behaviour on canonical CSV (MEASURED 2026-09-29)

| Check | MEASURED value | Source |
|---|---|---|
| Module import | OK (no raise) | `python -c "from molmetal.data.metal_cytotox import load_metal_cytotox"` |
| `load_metal_cytotox()` rows | 26,801 | direct invocation |
| Column order | `[smiles, metal, pIC50, active]` | direct invocation |
| `smiles` dtype | `object` (str-typed) | `df.dtypes` |
| `metal` dtype | `object` (str-typed) | `df.dtypes` |
| `pIC50` dtype | `float32` | `df.dtypes` |
| `active` dtype | `bool` | `df.dtypes` |
| `pIC50` NaN count | 0 | `df['pIC50'].notna().all()` |
| `active` value distribution | True=9,213 / False=17,588 | `df["active"].value_counts()` |
| Metal coverage | Ru=19,135 / Ir=4,546 / Rh=1,134 / Os=1,118 / Re=868 | `df["metal"].value_counts()` |

**Verdict:** loader is MEASURED-correct on the canonical CSV.  Wf-338-A3
finding 1 is closed at the loader layer.  The original Wf-338-A3 gate
FAIL was based on a stale snapshot; Wf-323B's synonym-aware fallback +
derive path resolved it.

### §2.2 Current pre-flight gate status (MEASURED 2026-09-29)

| Gate | Status | Reason |
|---|---|---|
| `p1_chemprop_smoke` | **FAIL** (out of Wf-344A1 scope) | `ModuleNotFoundError: No module named 'chemprop'` (env-side, NOT loader-side).  `train_chemprop_v2_v3.py` line 127 imports `from chemprop.data import BatchMolGraph`; the venv at `/home/hugo/codes/mol-metal/.venv/bin/python` does not have chemprop installed. |

The current `p1_chemprop_smoke` FAIL is a separate env-side blocker
(chemprop not in venv); Wf-344A1 pins only the loader-side regression.

---

## §3 Implementation (MEASURED)

### §3.1 New test file (`molmetal/tests/test_wf344a1_metal_cytotox_import.py`)

2 NEW tests; CPU-only, < 5 s total, no GPU / network / external state:

1. **`TestMetalCytotoxImportPath::test_module_import_is_error_free`** —
   import `molmetal.data.metal_cytotox` via `importlib.import_module`,
   access `load_metal_cytotox` callable, re-import for idempotence.
   Regression target: any future top-level ImportError / NameError /
   AttributeError that would cascade into
   `train_chemprop_v2_v3 --quick` and FAIL `p1_chemprop_smoke`.

2. **`TestMetalCytotoxCanonicalContract::test_load_metal_cytotox_csv_round_trip`** —
   load `.data/molmetal/MetalCytoToxDB.csv` (the canonical CSV Wf-338-A3
   flagged) and assert:
   - column order exactly `[smiles, metal, pIC50, active]`
   - non-empty (n_rows > 0)
   - `smiles`/`metal` are string-dtype, `pIC50` numeric, `active` bool
   - `pIC50` has no NaN
   - `active` is mixed True/False (proves 10µM derive threshold is firing)

### §3.2 Test run output (MEASURED)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 2 items

molmetal/tests/test_wf344a1_metal_cytotox_import.py::TestMetalCytotoxImportPath::test_module_import_is_error_free PASSED [ 50%]
molmetal/tests/test_wf344a1_metal_cytotox_import.py::TestMetalCytotoxCanonicalContract::test_load_metal_cytotox_csv_round_trip PASSED [100%]

============================== 2 passed in 0.55s ===============================
```

Both tests **PASS**.  CPU-only, total runtime 0.55 s.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: loader is import-error-free on canonical CSV
  (`molmetal.data.metal_cytotox.load_metal_cytotox()` returns
  26,801 rows × `[smiles, metal, pIC50, active]` with bool active +
  float32 pIC50 + zero NaN).  The Wf-338-A3 finding is closed at the
  loader layer.
- **MEASURED**: 2 NEW tests in `test_wf344a1_metal_cytotox_import.py`
  PASS in 0.55 s.
- **NOT-MEASURED / out-of-scope**: `p1_chemprop_smoke` pre-flight gate
  itself is currently FAIL on `ModuleNotFoundError: No module named
  'chemprop'` — env-side issue, not a metal_cytotox regression.
  Wf-344A1's 2 tests pin the loader side; the gate-side env install
  is a separate TODO (likely TODO-371 / Wf-345A).

---

## §5 Files added / modified

| Path | Action | Purpose |
|---|---|---|
| `molmetal/tests/test_wf344a1_metal_cytotox_import.py` | NEW | 2 NEW tests for import path + canonical 4-col contract |
| `TODO/completed/344a1_metal_cytotox_import.md` | NEW (this file) | Wf-344A1 ship document |

No source code modified — Wf-323B's synonym-aware + derive path
resolved the regression at the loader layer; Wf-344A1 is a verification
+ documentation layer only.

---

## §6 Cross-refs

- Wf-338-A3 finding 1 (originated this task) — pre-flight
  `p1_chemprop_smoke` gate FAIL evidence: `metal_cytotox.py:217`
  `ValueError: ... missing active / activity`.
- Wf-323B (`TODO/completed/323b_v3_v4_schema_drift_fix.md`) — shipped
  the synonym-aware + derive-active fallback that Wf-344A1 verifies.
- Wf-327F2 (`molmetal/tests/test_wf327f2_v3_v4_schema.py`) — 4
  existing schema-drift tests (complementary; Wf-344A1 adds 2 import-
  path tests).
- TODO-207 §4.2 P1 — pre-flight `p1_chemprop_smoke` gate definition.
- AGENTS.md §1 rule 4 — pre-flight v2 25/25 PASS gate.
- AGENTS.md §1 rule 1 — MEASURED-vs-PROJECTED tag discipline.
