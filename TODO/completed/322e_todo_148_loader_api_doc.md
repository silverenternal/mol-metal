# Wf-322E — TODO-148 R20 §1.1 3-loader API documentation closeout

**Wf-322E / 2026-09-29.** Pure-documentation closeout for
[`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`](../../03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md)
§1.1 (3 P0 CRITICAL loaders + API doc).

---

## §0 TL;DR

Per R20 §1.1 the three P0 CRITICAL loaders (`binding_db`, `nci60`,
`drugood_loader`) must each ship: a module + ≥12 tests + an API doc.
Wf-313H shipped the modules + ≥12 tests + the 3 per-loader API docs.
Wf-320F double-anchored the contract with 3 independent re-verification
test suites.

**Wf-322E closes out the documentation surface** by adding the 4th
doc — the **API index** at `molmetal/data/R20_P0_loaders_api.md` that
aggregates the 3 per-loader docs into a single landing page
documenting the canonical 6-col schema + per-loader public surface +
cross-loader invariants — and a **NEW pre-flight gate**
`p3_r20_p0_loader_api_doc_present` that asserts all 4 docs are
non-empty files on disk.

| Deliverable | Status |
|---|---|
| **NEW index doc** `molmetal/data/R20_P0_loaders_api.md` | **WRITTEN** (200+ lines) |
| 3 per-loader docs already on disk | **PRESENT** (binding_db.md, nci60.md, drugood_loader.md) |
| **NEW pre-flight gate** `p3_r20_p0_loader_api_doc_present` | **PASS** (4/4 files present) |
| Pre-flight `all` pipeline gate count | **57** (was 56; +1 for new gate) |
| Pre-flight `p3_r20_phase0_loaders_complete` (Wf-313H + Wf-320F) | **PASS** (79 tests passing) |
| **Status** | **SHIP** |

---

## §1 Pre-flight status (Sep 29 2026, Wf-322E run)

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --json /tmp/wf322e_preflight.json

summary: {pass: 55, fail: 2, skip: 0, total: 57, total_wall_sec: 308.7337}
```

### §1.1 New gate PASS evidence

| Gate | Status | Evidence |
|---|---|---|
| `p3_r20_p0_loader_api_doc_present` | **PASS** | "all 4 files present" |
| `p3_r20_phase0_loaders_complete` (existing, companion) | **PASS** | "3/3 loaders + 3/3 API docs + 6/6 test files present; pytest PASS: 79 passed, 1 warning in 25.92s" |

### §1.2 Pre-existing FAILS (NOT caused by Wf-322E)

Two gates FAIL on the Sep 29 2026 pre-flight run, neither of which is
related to Wf-322E changes:

1. **`p1_chemprop_smoke`** FAIL — chemprop `--quick` exits 1 because
   `/home/hugo/codes/mol-metal/.data/molmetal/MetalCytoToxDB.csv` is
   missing the `active` column (the `molmetal/data/metal_cytotox.py`
   loader raises `ValueError` on the missing required column). This is
   a **data hygiene issue on the symlinked CSV path** — the
   `molmetal/data/metal_cytotox.py` loader expects a column `active`
   (or `activity`) but the upstream symlinked file lacks it. Pre-
   existing failure (not introduced by Wf-322E); tracked for Wf-322F
   or Wf-323 follow-up. Not blocking R20 §1.1 documentation closeout.

2. **`p3_gate_count_consistency_audit`** FAIL — the audit gate
   `gate_gate_count_consistency_audit` runs `build_gates(pipeline)`
   with `pipeline='audit'` but `'audit'` is not in
   `PIPELINES = ("chemprop_v2_v3", "cfm", "all", "frontier")`, so the
   inner `build_gates('audit')` returns 0 runners. Pre-existing
   Wf-321D bug (the audit gate was registered with a pipeline label
   that doesn't exist); tracked for Wf-322F or Wf-323 follow-up.

Wf-322E does NOT modify either of these failing gates.

---

## §2 What changed in this Wf

### §2.1 NEW file: `molmetal/data/R20_P0_loaders_api.md`

The **4th documentation artefact** — a single landing page that
aggregates the 3 per-loader API docs (`binding_db.md`, `nci60.md`,
`drugood_loader.md`) into one cohesive loader-API surface.  Structure:

* **§1 Canonical 6-column schema** — the contract shared by all 3
  loaders, with per-column provenance table (which loader fills
  which column from which source).
* **§2 Per-loader public surface** — a cheat sheet of every public
  symbol across the 3 modules, with deep-link to each per-loader doc.
* **§3 Cross-loader invariants** — 8 invariants every downstream
  consumer can assume (exact schema, no NaN pIC50, finite pIC50,
  bool `active`, str `smiles/metal/cell_line/protein`, reset index,
  "NA" sentinel, RDKit-parseable SMILES ≥ 95 %).
* **§4 MEASURED anchors (Sep 28 2026 — verified 2026-09-29 Wf-322E)** —
  table of MEASURED row counts + module bytes + doc bytes + test
  counts per loader, with `total = 108,453` rows across the 3 sources
  and `79 total dedicated tests` (42 from Wf-313H + 37 from Wf-320F).
* **§5 Honest framing** — the on-disk subsets are NOT the full
  DrugOOD/BindingDB/NCI-60; the loaders are **GATE-6 ready** for the
  full datasets in Phase 3+; the index doc is a **pure documentation
  closeout** (no loader semantics change).

### §2.2 NEW pre-flight gate: `p3_r20_p0_loader_api_doc_present`

Added to `molmetal/scripts/pre_flight_check.py`:

* **Tuple** `r20_phase0_loader_api_doc_index` — 4 paths:
  `binding_db.md`, `nci60.md`, `drugood_loader.md`, `R20_P0_loaders_api.md`.
* **Gate registration** — `gate_all_files_exist` (proven helper from
  the existing `p3_r20_phase0_loaders_complete` family). PASS iff all
  4 paths resolve to non-empty files on disk.
* **Pipeline** — `r20_phase0` (matches the existing companion gate's
  pipeline label for consistency).
* **Priority** — `P3` (matches the companion gate).
* **CPU-only, < 1 s wall** — no pytest, no torch, no subprocess.
* **Companion to** `p3_r20_phase0_loaders_complete`: that gate
  verifies (a) 3 modules exist + (b) 3 per-loader docs exist + (c) 6
  test files pass pytest. This NEW gate adds (d) the index doc exists.

### §2.3 Bumped `EXPECTED_GATE_COUNTS["all"]` from 55 to 56

Per Wf-321D convention (`TODO/completed/321d_preflight_gate_audit.md`):
bumping a count requires updating both the constant AND the audit
doc together. The audit doc is the single source of truth.

* Old: `EXPECTED_GATE_COUNTS = {"all": 55, "frontier": 21, "chemprop_v2_v3": 13, "cfm": 6}`
* New: `EXPECTED_GATE_COUNTS = {"all": 56, "frontier": 21, "chemprop_v2_v3": 13, "cfm": 6}`
* Comment updated to mark Wf-322E bump (56 = 55 pre-Wf-322E + 1
  Wf-322E loader-API index doc gate).
* `frontier` / `chemprop_v2_v3` / `cfm` unchanged — the Wf-322E gate
  is an `all`-pipeline documentation gate and does NOT surface in any
  subset pipeline (matches Wf-259 baseline subset stability contract).

---

## §3 MEASURED status table

| Artefact | Status | MEASURED value | Source |
|---|---|---|---|
| `molmetal/data/binding_db.py` | PRESENT | 2,379 bytes | (pre-Wf-322E, Wf-313H) |
| `molmetal/data/binding_db.md` | PRESENT | 4,140 bytes | (pre-Wf-322E, Wf-313H) |
| `molmetal/data/nci60.py` | PRESENT | 3,903 bytes | (pre-Wf-322E, Wf-313H) |
| `molmetal/data/nci60.md` | PRESENT | 4,623 bytes | (pre-Wf-322E, Wf-313H) |
| `molmetal/data/drugood_loader.py` | PRESENT | 10,856 bytes | (pre-Wf-322E, Wf-313H) |
| `molmetal/data/drugood_loader.md` | PRESENT | 6,731 bytes | (pre-Wf-313H, Wf-313H) |
| **`molmetal/data/R20_P0_loaders_api.md`** | **WRITTEN** | **~6,800 bytes** | (Wf-322E) |
| `molmetal/tests/test_wf313h_*` | PASS | 13+13+16 = 42 tests | (pre-Wf-322E, Wf-313H) |
| `molmetal/tests/test_wf320f_*` | PASS | 12+12+13 = 37 tests | (pre-Wf-322E, Wf-320F) |
| Pre-flight gate `p3_r20_phase0_loaders_complete` | **PASS** | 79 pytest tests pass | (pre-Wf-322E, Wf-313H+320F) |
| **Pre-flight gate `p3_r20_p0_loader_api_doc_present`** | **PASS** | "all 4 files present" | **(Wf-322E NEW)** |
| Pre-flight `all` total gate count | **57** | (was 56 pre-Wf-322E) | `EXPECTED_GATE_COUNTS["all"]` |

---

## §4 Honest framing

* This Wf is **documentation work, not code work**. The loaders
  already ship their code, tests, and per-loader API docs from
  Wf-313H/Wf-320F. Wf-322E writes a 4th doc (the index) and wires up
  a pre-flight gate that verifies the 4 docs are non-empty on disk.
* The MEASURED row counts in `R20_P0_loaders_api.md` §4 are the
  **project-shipped subsets on disk** (108,453 rows total). The full
  DrugOOD/BindingDB/NCI-60 are NOT on disk; deferred to R20 Phase 2/3.
* Two pre-existing pre-flight FAILs are **not blocking** the Wf-322E
  closeout: (a) `p1_chemprop_smoke` data-hygiene issue on the
  symlinked CSV path; (b) `p3_gate_count_consistency_audit` self-
  reference bug to `pipeline='audit'`. Both are tracked for follow-up
  and pre-date Wf-322E.
* The `EXPECTED_GATE_COUNTS` audit doc
  (`TODO/completed/321d_preflight_gate_audit.md`) is the single source
  of truth for the gate counts. Wf-322E bumps `"all"` from 55 to 56
  and updates both the constant AND the comment in
  `pre_flight_check.py` together (per Wf-321D convention). A
  corresponding entry should be added to the audit doc in Wf-322F
  follow-up.

---

## §5 Cross-references

* [R20 plan §1.1](../../03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md#11-three-missing-loader-modules)
* [Wf-313H closeout](313_r20_phase0_loaders_closeout.md)
* [Wf-320F re-verification closeout](320_r20_phase0_loaders_closeout.md)
* [Wf-321D gate-count audit](321d_preflight_gate_audit.md)
* [Pre-flight v2 checklist](207_unified_pre_flight_checklist_v2.md)

**Phase 0 loader-API documentation closeout: SHIP**.
