# WF-Lambda-Boost Phase 4 — pt_click_compat matrix update + version marker

**Date:** 2026-09-16
**Owner:** Lambda boost agent
**Phase:** 4 of 4
**Goal:** Pin the F2(a) invariant that `strict_Pt_II` is COMPATIBLE
on both `metal_ligand_exchange` and `aqua_exchange` (the canonical
Lippard 1995 + Reedijk 1987 coordination chemistry), add a
`COMPAT_MATRIX_VERSION` marker, and expose a `strict_pt_ii_allows_metal_coordination()`
helper for unit tests + audit reports.

---

## 1. Honest framing (read first)

* **What was attempted:** the F2(a) rows were already present in
  `pt_click_compat.py` from the WF-Lambda-Fix-FullPath-v2 work
  (2026-09-15).  Phase 4 ADDS:
  1. `COMPAT_MATRIX_VERSION` constant (`"wf-lambda-boost-2026-09-16"`)
  2. `strict_pt_ii_allows_metal_coordination()` helper function
* **What was measured:** 3/3 unit tests pass on CPU.
  * The helper returns `True` iff `strict_Pt_II` row has
    `metal_ligand_exchange == "compatible"` AND
    `aqua_exchange == "compatible"`.
  * The matrix version marker carries the WF-Lambda-Boost + 2026-09-16
    provenance string.
* **What is NOT changed:** the COMPAT_MATRIX dict itself is
  unchanged (the F2(a) rows were already correct).

---

## 2. Files changed

| File | Edit | Lines |
|------|------|-------|
| `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | Add `COMPAT_MATRIX_VERSION` + `strict_pt_ii_allows_metal_coordination()` helper | ~25 |
| `molmetal/tests/test_wf_lambda_boost.py` | NEW: 3 tests for `TestPtClickCompatMatrixUpdate` | ~30 |

---

## 3. The strict_Pt_II row (unchanged from WF-Lambda-Fix-FullPath-v2)

```python
COMPAT_MATRIX["strict_Pt_II"] = {
    "CuAAC":                  "compatible",
    "SPAAC":                  "compatible",
    "ThiolEne":               "incompatible",
    "Suzuki":                 "marginal",
    "AmideCoupling":          "incompatible",
    "metal_ligand_exchange":  "compatible",   # F2(a) Pt-Cl + NH3
    "aqua_exchange":          "compatible",   # F2(a) Pt-Cl + H2O
}
```

The two F2(a) rows are the canonical Lippard 1995 + Reedijk 1987
coordination chemistry — Pt_II square-planar undergoes *both* NH3
substitution (MetalLigandExchange) and H2O substitution (AquaExchange)
as part of its standard reactivity.  These rules are NOT click rules
— they are the *coordination* rules that allow the Lambda search to
build different Pt(II) ligand spheres while keeping
`metal_compliance == 1.0`.

---

## 4. New helper: strict_pt_ii_allows_metal_coordination()

```python
def strict_pt_ii_allows_metal_coordination() -> bool:
    """WF-Lambda-Boost Phase 4 helper.

    Returns ``True`` iff the ``strict_Pt_II`` row has
    ``metal_ligand_exchange == "compatible"`` AND
    ``aqua_exchange == "compatible"``.
    """
    row = COMPAT_MATRIX.get("strict_Pt_II", {})
    return (
        row.get("metal_ligand_exchange") == "compatible"
        and row.get("aqua_exchange") == "compatible"
    )
```

This helper is the **F2(a) invariant pin**.  An accidental future
tightening of the strict_Pt_II row to `"incompatible"` on either
metal-coordination column would fail the unit test suite.

---

## 5. Test coverage (3 tests)

| Test | Asserts |
|------|---------|
| `test_strict_pt_ii_allows_metal_coordination_helper` | `strict_pt_ii_allows_metal_coordination() is True` |
| `test_strict_pt_ii_row_has_metal_coordination_compatible` | Direct dict check on `COMPAT_MATRIX["strict_Pt_II"]` |
| `test_compat_matrix_version_marker` | `COMPAT_MATRIX_VERSION` contains "wf-lambda-boost" + "2026-09-16" |

---

## 6. Honest limitations + follow-ups

* **The matrix is hand-curated, not learned.**  Future work: fit
  the COMPAT_MATRIX on a tmQM reaction-outcome classifier (TODO-29
  Phase-3L follow-up).  Until then, the verdicts are chemistry
  knowledge, not ML predictions.
* **`metal_ligand_exchange` + `aqua_exchange` rows on `Pt_IV`,
  `Pt_II_chelating`, `labile_metal`, `unknown` scaffolds are all
  COMPATIBLE** — this is intentional (these scaffolds permit
  coordination chemistry by definition).  The strict_Pt_II row
  is the only one where the verdict is non-trivial.
* **`strict_pt_ii_allows_metal_coordination()` is a one-liner.**
  It is exposed for unit-test pinning + audit-report rendering.
  Production callers should consult `default_compatible_rules()`
  for the canonical rule list rather than this boolean.
