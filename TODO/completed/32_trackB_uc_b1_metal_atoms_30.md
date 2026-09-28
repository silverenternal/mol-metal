# TODO-32 UC-B1 — METAL_ATOMS 30-metal extension — DONE 2026-09-25

**Supersedes TODO-32 in `pending/`** (`32_ultracode_trackB_lambda_calculus.md` §B1 atoms-table expansion).

**Status:** ✅ DONE — `METAL_ATOMS` extended from 6 to 30 entries,
mirroring the tmQM 30-metal subset in `PRIMITIVE_ATOMS._TMQM_METAL_DATA`.
All 30 entries satisfy `arity = valence + lone_pairs` and carry a
canonical coordination geometry.  69 tests green
(`molmetal/tests/test_lambda_combinators_expansion.py`).

---

## 1. What was done

Extended the `METAL_ATOMS` dict in
`molmetal/molmetal_lam/atoms/combinators.py` from 6 hand-curated
metals (Pt_II / Ru_II / Zn_II / Ir_III / Cu_II / Au_III) to 30
d-block transition metals, matching the tmQM coordination-chemistry
subset.  Each new entry uses the convention
`valence=0, lone_pairs=coordination_number` (d-block centres accept
dative bonds from lone-pair donors; the arity equals the metal's
canonical coordination number) with the geometry drawn from
{square_planar, octahedral, tetrahedral, linear}.

| Block | New METAL_ATOMS keys |
|---|---|
| 3d (8 new) | `Sc_III, Ti_IV, V_V, Cr_III, Mn_II, Fe_II, Co_II, Ni_II` |
| 4d (8 new) | `Y_III, Zr_IV, Nb_V, Mo_VI, Tc_VII, Rh_III, Pd_II, Ag_I, Cd_II` |
| 5d + La (7 new) | `La_III, Hf_IV, Ta_V, W_VI, Re_VII, Os_III, Hg_II` |

Total METAL_ATOMS: **30 entries** (was 6, added 24).

The 6 R16-contract metals keep their original semantics
(`valence=2, lone_pairs=2, geometry=square_planar` for Pt_II / Cu_II /
Au_III / Zn_II; `valence=0, lone_pairs=6, geometry=octahedral` for
Ru_II; `valence=3, lone_pairs=3, geometry=octahedral` for Ir_III) so
no existing test that depends on the 6-metal semantics regressed.

`_EXPECTED_METAL_CN` was extended in parallel to cover all 30
entries; the `METAL_GEOMETRY_OK` sanity metric (`1.0` healthy)
now reports `30/30` instead of `6/6`.

---

## 2. Tests added

In `molmetal/tests/test_lambda_combinators_expansion.py`:

* `test_metal_atoms_has_at_least_30_entries` — contract test for
  the R17 UC-B1 minimum size.
* `test_metal_atoms_keys_match_expected_set` — verifies the full
  30-key set is present.
* `test_metal_atoms_arity_matches_coordination_number` —
  `@pytest.mark.parametrize` over all 30 symbols; verifies
  `arity == valence + lone_pairs`, `is_metal == True`, and
  `geometry ∈ VALID_GEOMETRIES`.
* `test_metal_atoms_arity_table` — `@pytest.mark.parametrize` over
  the 30 symbols × expected arity pairs (4 / 6 / 2).

Combined with the 8 pre-existing tests in the file
(`test_30_transition_metals_in_primitive_atoms`,
`test_halogens_present`, `test_metal_geometries_are_canonical`,
`test_arity_equals_valence_plus_lone_pairs`,
`test_cisplatin_still_parses`,
`test_carboplatin_and_oxaliplatin_parse`,
`test_metal_symbols_includes_all_30`, etc.) the file now has
**69 passing tests** (was 9).

```
$ uv run --no-sync python -m pytest molmetal/tests/test_lambda_combinators_expansion.py -v --no-header
============================= 69 passed, 1 warning in 1.68s ==============================
```

---

## 3. Cross-references

* `TODO/pending/32_ultracode_trackB_lambda_calculus.md` §B1 — UC-B1
  "30-metal `METAL_ATOMS` 表扩展" — DONE.
* `TODO/pending/34_parallel_ultracode_dispatch.md` §2.2 — Track B-剩余
  UC-B1 (3h CPU budget; shipped in ~15 min via direct edit + parametrize).
* `TODO/completed/32_trackB_mlc_pll_wireup_done.md` — Track B
  PARTIAL entry that ships PLL + β-reducer + BioLM adapter; this
  B1 entry now upgrades the atom-table to 30, leaving the B2 reward
  coupling and B3 5-new-click-rules still as the remaining Track B
  work.

---

## 4. Files changed

| File | Change | LOC delta |
|---|---|---|
| `molmetal/molmetal_lam/atoms/combinators.py` | METAL_ATOMS +24 entries; _EXPECTED_METAL_CN +24 entries | +130 |
| `molmetal/tests/test_lambda_combinators_expansion.py` | 4 new test funcs + 60 parametrize cases | +95 |

---

End of report.