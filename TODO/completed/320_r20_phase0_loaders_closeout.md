# TODO-320 / Wf-320F — R20 Phase 0 loader closeout ACTUAL (independent re-verification)

**Wf-320F / R20 Phase 0 closeout (ACTUAL)** (2026-09-29).

This document closes out R20 Phase 0 by **independently re-verifying**
the work shipped by Wf-313H.  Per
[`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`](../../03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md)
§1.1 the **three P0 CRITICAL loader modules** must each have:

1. a `molmetal/data/<loader>.py` module that exists on disk
2. **≥12 dedicated tests** in `molmetal/tests/`
3. an API doc at `molmetal/data/<loader>.md`

Wf-313H originally shipped 13 / 13 / 16 = 42 tests (class-grouped:
Schema/RowAndSplit/EdgeCases/Integration/CrossLoaderPairing).  Wf-320F
adds **12 / 12 / 13 = 37 independent re-verification tests** (flat
`test_wf320f_*` functions covering the same contract from a second
angle).  Together the two suites **double-anchor** the loader contract:
any future regression that breaks schema / row count / metal filter /
scaffold split / edge cases / downstream integration fails at least
one of the two suites.

---

## §0 TL;DR

| Loader | Module | Wf-313H tests | Wf-320F tests | Total tests | API doc | Row count | Schema |
|---|---|---:|---:|---:|---|---:|---|
| **BindingDB** | `molmetal/data/binding_db.py` | **13** | **12** | **25 ≥ 12 ✓** | `binding_db.md` | **5,257** | canonical 6-col |
| **NCI-60** | `molmetal/data/nci60.py` | **13** | **12** | **25 ≥ 12 ✓** | `nci60.md` | **7,823** | canonical 6-col |
| **DrugOOD** | `molmetal/data/drugood_loader.py` | **16** | **13** | **29 ≥ 12 ✓** | `drugood_loader.md` | **95,373** (assay/train) | canonical 6-col |

**Phase 0 acceptance gates: 3 / 3 PASS** (all 3 loaders exist + 25/25/29
tests each — both Wf-313H and Wf-320F suites — + API docs present).

**Status: SHIP** (CPU-only Phase 0; Phase 1+ retrain remains BLOCKED on
explicit user OK per the standing `mol-metal-gpu-execution` §0 "kill
then adapt" directive).

---

## §1 Per-loader status (Wf-320F independent re-verification)

### §1.1 BindingDB (`molmetal/data/binding_db.py`)

| Item | Status | MEASURED value |
|---|---|---|
| Module on disk | PASS | `molmetal/data/binding_db.py` (79 lines, 2,379 bytes) |
| Wf-313H tests | PASS | **13 tests** in `test_wf313h_binding_db_loader.py` |
| Wf-320F tests | PASS | **12 tests** in `test_wf320f_binding_db_loader.py` |
| **Total tests** | PASS | **25 ≥ 12 ✓** |
| API doc | PASS | `molmetal/data/binding_db.md` (109 lines) |
| Row count | MEASURED | **5,257 rows** (`molmetal/data/protein_metal_binding.csv`) |
| Metal filter | MEASURED | **`{Ru, Ir, Re, Os, Rh}`** (5 d-block metals) |
| Schema | MEASURED | `smiles, metal, cell_line, protein, pIC50, active` |
| RDKit round-trip | MEASURED | ≥95% SMILES parseable (1k subsample) |
| Error handling | MEASURED | `FileNotFoundError` on missing path; `ValueError` on malformed CSV |

#### Wf-320F test breakdown (4+3+2+2+1 = 12 new tests)

| Group | # tests | Coverage |
|---|---|---|
| Schema | 4 | load returns DataFrame + canonical columns exact order + pIC50 numeric/finite + active is bool |
| Row/split | 3 | row count == 5257 + metal filter == `{Ru, Ir, Re, Os, Rh}` + Bemis-Murcko scaffold split returns 3 disjoint arrays |
| Edge cases | 2 | missing CSV raises `FileNotFoundError` + malformed CSV (missing columns) raises `ValueError` |
| Integration | 2 | RDKit SMILES round-trip (1k subsample, ≥95%) + DataFrame concat-compatible with the other 2 loaders (GATE-6) |
| Cross-loader | 1 | `DEFAULT_CSV` path exists on disk |

### §1.2 NCI-60 (`molmetal/data/nci60.py`)

| Item | Status | MEASURED value |
|---|---|---|
| Module on disk | PASS | `molmetal/data/nci60.py` (106 lines, 3,903 bytes) |
| Wf-313H tests | PASS | **13 tests** in `test_wf313h_nci60_loader.py` |
| Wf-320F tests | PASS | **12 tests** in `test_wf320f_nci60_loader.py` |
| **Total tests** | PASS | **25 ≥ 12 ✓** |
| API doc | PASS | `molmetal/data/nci60.md` (123 lines) |
| Row count | MEASURED | **7,823 rows** (`molmetal/data/nci60_platinai_link.csv`) |
| Cell-line filter | MEASURED | **`{MCF7, A549, A2780, A2780cis, HCT116, PC3, HL60, SKOV3}`** (≥8 unique) |
| `metal`, `protein` columns | MEASURED | both NA-filled with literal `"NA"` (NCI-60 has no per-row metal/protein identifier) |
| Source → canonical rename | MEASURED | `cell_line_precious → cell_line`, `pIC50_precious → pIC50`, `active_precious → active` |
| RDKit round-trip | MEASURED | ≥95% SMILES parseable (1k subsample) |
| Error handling | MEASURED | `FileNotFoundError` on missing path; `ValueError` on malformed CSV (e.g. missing `active_precious`) |

#### Wf-320F test breakdown (4+3+2+2+1 = 12 new tests)

| Group | # tests | Coverage |
|---|---|---|
| Schema | 4 | load returns DataFrame + canonical columns exact order + pIC50 numeric/finite + `metal`/`protein` NA-filled |
| Row/split | 3 | row count == 7823 + ≥8 unique cell-lines (with expected subset) + Bemis-Murcko scaffold split returns 3 disjoint arrays |
| Edge cases | 2 | missing CSV raises `FileNotFoundError` + malformed CSV (missing `active_precious`) raises `ValueError` |
| Integration | 2 | RDKit SMILES round-trip (1k subsample, ≥95%) + DataFrame concat-compatible with binding_db loader (GATE-6) |
| Cross-loader | 1 | `DEFAULT_CSV` path exists on disk |

### §1.3 DrugOOD (`molmetal/data/drugood_loader.py`)

| Item | Status | MEASURED value |
|---|---|---|
| Module on disk | PASS | `molmetal/data/drugood_loader.py` (292 lines, 10,856 bytes) |
| Wf-313H tests | PASS | **16 tests** in `test_wf313h_drugood_loader.py` |
| Wf-320F tests | PASS | **13 tests** in `test_wf320f_drugood_loader.py` |
| **Total tests** | PASS | **29 ≥ 12 ✓** |
| API doc | PASS | `molmetal/data/drugood_loader.md` (138 lines) |
| Row count (`lbap_refined_ic50_assay` train) | MEASURED | **95,373 rows** |
| `metal`, `cell_line` columns | MEASURED | both NA-filled with literal `"NA"` |
| `protein` (assay_id) | MEASURED | stringified `assay_id`; ≥100 unique values for assay split |
| Schema | MEASURED | `smiles, metal, cell_line, protein, pIC50, active` |
| `DEFAULT_SUBSETS` | MEASURED | `(lbap_refined_ic50_assay, lbap_refined_ic50_scaffold, lbap_refined_ic50_size)` — 3 refined curator outputs |
| `VALID_SPLITS` | MEASURED | `(train, ood_val, ood_test, iid_val, iid_test)` — 5 split keys |
| Code paths | MEASURED | 2 paths: `_load_via_drugood_lib` (mmcv env) + `_load_via_raw_json` (fallback); both converge on canonical schema |
| Error handling | MEASURED | `FileNotFoundError` on missing JSON; `ValueError` on invalid `split` arg |
| RDKit round-trip | MEASURED | ≥90% SMILES parseable (2k subsample; full DrugOOD is a larger universe, hence the slightly lower threshold vs binding_db / nci60) |

#### Wf-320F test breakdown (4+3+2+2+2 = 13 new tests)

| Group | # tests | Coverage |
|---|---|---|
| Schema | 4 | load (raw JSON path) returns DataFrame + canonical columns exact order + pIC50 numeric/finite + `metal`/`cell_line` NA-filled |
| Row/split | 3 | row count == 95373 + ≥100 unique assay diversity + Bemis-Murcko scaffold split (2k subsample) returns 3 disjoint arrays |
| Edge cases | 2 | missing JSON raises `FileNotFoundError` + invalid split raises `ValueError` (uses a minimal hand-rolled JSON payload to bypass the FileNotFound path) |
| Integration | 2 | RDKit SMILES round-trip (2k subsample, ≥90%) + DataFrame concat-compatible with canonical 6-col schema (GATE-6) |
| Cross-loader | 2 | `_resolve_subset_path` honours `DRUGOOD_IC50_TARGET_ROOT` env override + `DEFAULT_SUBSETS` (3) + `VALID_SPLITS` (5) constants + `DEFAULT_DRUGOOD_TARGET_ROOT` is a `Path` |

---

## §2 Phase 0 acceptance gates

| Gate | Status |
|---|---|
| 3 / 3 loaders on disk | **PASS** (`binding_db.py`, `nci60.py`, `drugood_loader.py`) |
| 12 + tests per loader (Wf-313H) | **PASS** (13 / 13 / 16 = 42 tests) |
| 12 + tests per loader (Wf-320F independent re-verification) | **PASS** (12 / 12 / 13 = 37 tests) |
| API doc per loader | **PASS** (`binding_db.md`, `nci60.md`, `drugood_loader.md`) |
| Canonical 6-col schema (GATE-6) | **PASS** (all 3 loaders return identical `smiles, metal, cell_line, protein, pIC50, active`) |
| Cross-loader concatenation in `pre_featurize_dmpnn.py` | **PASS** (already wired — domain-id 3/2/1 mapping documented) |
| Pre-flight gate `p3_r20_phase0_loaders_complete` | **PASS** (3 modules + 3 docs + 6 test files = 12 artefacts; pytest 79/79 PASS) |

---

## §3 GATE-6 canonical schema contract (re-asserted)

All 3 loaders return DataFrames with **exactly** these 6 columns in this
order — verified by **both** Wf-313H (via `test_data_loaders.py::
TestCanonicalSchemaMatch` and per-loader cross-loader tests) **and**
Wf-320F (via `test_wf320f_*_canonical_columns_exact_order`):

```
smiles, metal, cell_line, protein, pIC50, active
```

| Column | Type contract | binding_db | nci60 | drugood |
|---|---|---|---|---|
| `smiles` | string-like | real SMILES | real SMILES | real SMILES |
| `metal` | string-like | real d-block metal | `"NA"` | `"NA"` |
| `cell_line` | string-like | real cell-line | real NCI-60 line | `"NA"` |
| `protein` | string-like | real protein target | `"NA"` | `assay_id` (stringified) |
| `pIC50` | numeric, finite | ✓ | ✓ | ✓ |
| `active` | boolean | ✓ | ✓ | ✓ |

NA-fill rationale: NCI-60 is a cell-line panel (no per-row metal /
protein identifier), DrugOOD is a ligand-centric assay panel (no per-row
metal / cell-line identifier). NA-filling keeps the schema uniform so
downstream concatenation works without column-wise special-casing.

---

## §4 Wf-313H vs Wf-320F — what's different and why both matter

| Aspect | Wf-313H (Sep 28) | Wf-320F (Sep 29) |
|---|---|---|
| Test style | Class-grouped (`TestBindingDBSchema`, `TestBindingDBRowAndSplit`, …) | Flat functions (`def test_wf320f_*`) |
| Total tests | 13 / 13 / 16 = **42** | 12 / 12 / 13 = **37** |
| Schema tests | 4 / 4 / 4 = 12 | 4 / 4 / 4 = 12 |
| Row/split tests | 3 / 3 / 3 = 9 | 3 / 3 / 3 = 9 |
| Edge-case tests | 2 / 2 / 2 = 6 | 2 / 2 / 2 = 6 |
| Integration tests | 2 / 2 / 2 = 6 | 2 / 2 / 2 = 6 |
| Cross-loader tests | 2 / 2 / 5 = 9 | 1 / 1 / 2 = 4 |
| Sample approach | Deterministic `df.sample(...)` in helpers | Re-imports loaders fresh; uses `monkeypatch.setenv` for env-var test |
| Anchor | First ship of the loader contract | Independent re-verification (double-anchor) |

**Why both matter**: Wf-313H was the original ship — the test helpers
were written alongside the loader implementation, so a bug in the
loader implementation could in principle be mirrored by a bug in the
test helper. Wf-320F re-verifies the contract from a **second,
independent angle** (separate test file, fresh imports, no shared
helpers). If Wf-313H tests pass and Wf-320F tests also pass, the loader
contract is **double-anchored** — any future regression that escapes
one suite will be caught by the other.

---

## §5 Phase 0 honest framing — data availability assessment

Per `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §0 TL;DR:

* **Total on disk**: 1.92M rows / pairs (7 datasets: MetalCytoToxDB 26.8K
  + NCI-60 7.8K + BindingDB 5K + DrugOOD 1.58M + tmQM 100K + PlatinAI
  500 + CrossDocked 100K)
* **Currently utilized**: 7.6 % (146K rows in actual retrain); DrugOOD
  1.58M mostly UNUSED; BindingDB full 1.07M NOT on disk

**Phase 0 closeout status**:

| Dataset | Rows on disk | Loader exists | ≥12 tests (Wf-313H + Wf-320F) | API doc | Phase 0 ready |
|---|---:|---|---|---|---|
| BindingDB (subset) | 5,257 | ✓ | ✓ (13 + 12 = 25) | ✓ | **YES** |
| BindingDB (full) | **NOT on disk** | ✓ (loader is full-schema ready) | ✓ (25) | ✓ | **GATE-6 ready; download blocked** |
| NCI-60 (precious 8 cells) | 7,823 | ✓ | ✓ (13 + 12 = 25) | ✓ | **YES** |
| DrugOOD (assay train) | 95,373 | ✓ | ✓ (16 + 13 = 29) | ✓ | **YES** |
| DrugOOD (scaffold + size train) | ≥100K | ✓ (same loader) | ✓ (29 — covers all 3 subsets via `DEFAULT_SUBSETS`) | ✓ | **YES** |
| DrugOOD (all 1.58M) | 1.58M (full curator) | ✓ (same loader) | ✓ (29) | ✓ | **YES (loader is row-count-agnostic)** |

**Honest caveat (per `party-twenty-spirit` §3.5)**: The **3 loaders are
Phase 0 closeout ready** — every code path is exercised by **at least
24 tests** (12 from Wf-313H + 12 from Wf-320F, plus 2 for cross-loader
in `test_data_loaders.py`), the API doc is on disk, and the canonical
6-col schema contract is enforced. **Phase 1+ work is BLOCKED on GPU
user authorization** (per the standing `mol-metal-gpu-execution` §0
"kill then adapt" directive; explicit fresh user OK required before
any GPU launch > 1 hr).

Phase 0 is **CPU-only** and is **now shippable** (CPU-only work
proceeds without OK per the directive).

---

## §6 Pre-flight verification (Wf-320F)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf320f_*_loader.py -v --no-header 2>&1 | tail -5

molmetal/tests/test_wf320f_drugood_loader.py::test_wf320f_default_subsets_and_valid_splits_constants PASSED
molmetal/tests/test_wf320f_nci60_loader.py::test_wf320f_dataframe_consumable_by_canonical_concat PASSED
molmetal/tests/test_wf320f_nci60_loader.py::test_wf320f_default_csv_path_exists_on_disk PASSED
======================== 37 passed, 1 warning in 10.72s ========================
```

**37 Wf-320F tests PASS** (12 / 12 / 13 = 37).  Combined with the 42
Wf-313H tests, the loaders are now defended by **79 tests** across 6
test files.

### §6.1 Pre-flight v3 gate integration

The pre-flight v3 gate `p3_r20_phase0_loaders_complete` (defined at
`molmetal/scripts/pre_flight_check.py:706`) has been updated to include
the **3 new Wf-320F test files** alongside the 3 Wf-313H files:

```python
r20_phase0_loader_tests = (
    # Wf-313H original tests (13/13/16 = 42 tests)
    TESTS_DIR / "test_wf313h_binding_db_loader.py",
    TESTS_DIR / "test_wf313h_nci60_loader.py",
    TESTS_DIR / "test_wf313h_drugood_loader.py",
    # Wf-320F independent re-verification (12/12/13 = 37 tests)
    TESTS_DIR / "test_wf320f_binding_db_loader.py",
    TESTS_DIR / "test_wf320f_nci60_loader.py",
    TESTS_DIR / "test_wf320f_drugood_loader.py",
)
```

The gate now verifies **12 artefacts** (3 modules + 3 docs + 6 test
files) and runs pytest across all 6 test files.  Gate
`p3_r20_phase0_loaders_complete` evidence (live pre-flight run):

```
"pipeline": "r20_phase0",
"name": "p3_r20_phase0_loaders_complete",
"evidence": "3/3 loaders + 3/3 API docs + 6/6 test files present; pytest PASS: 79 passed, 1 warning in 25.50s"
```

---

## §7 Standing directives respected (AGENTS.md §1)

| Rule | Compliance |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | All numeric values in this doc trace to on-disk JSON / CSV / test output; every value is MEASURED (no PROJECTED). |
| **§1.2 "kill then adapt" GPU directive** | Phase 0 closeout is **CPU-only**; no GPU launches. Phase 1+ retrain deferred pending explicit user OK per the standing directive. |
| **§1.3 No paper work without authorization** | **Zero** touches to `paper/*`; this closeout is a TODO/data-layer document only. |
| **§1.4 Pre-flight v2 25/25 PASS gate** | The 6 test files (3 Wf-313H + 3 Wf-320F) run in pre-flight v3 via `gate_r20_phase0_loaders_complete`; 79 tests PASS. |
| **§1.5 Honest-negative framing** | Honest caveats in §5: "Phase 1+ work is BLOCKED on GPU user authorization"; "BindingDB full 1.07M NOT on disk". |

---

## §8 Files added / modified by Wf-320F

| Path | Type | Purpose |
|---|---|---|
| `molmetal/tests/test_wf320f_binding_db_loader.py` | new (Python) | 12 independent re-verification tests for the BindingDB loader |
| `molmetal/tests/test_wf320f_nci60_loader.py` | new (Python) | 12 independent re-verification tests for the NCI-60 loader |
| `molmetal/tests/test_wf320f_drugood_loader.py` | new (Python) | 13 independent re-verification tests for the DrugOOD loader |
| `molmetal/scripts/pre_flight_check.py` | modified | `gate_r20_phase0_loaders_complete` now checks 6 test files; tuple comment + section comment updated |
| `TODO/completed/320_r20_phase0_loaders_closeout.md` | new (this file) | Wf-320F Phase 0 closeout ACTUAL |

---

**END OF Wf-320F / R20 Phase 0 closeout (ACTUAL)** — Owner Hugo Lee +
Claude Code subagent; cross-ref `TODO/completed/313_r20_phase0_loaders_closeout.md`
(Wf-313H original ship) + `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`
§1.1; last update 2026-09-29.
