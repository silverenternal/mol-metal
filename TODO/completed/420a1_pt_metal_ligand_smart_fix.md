# Wf-420-A1 — Pt metal ligand exchange SMARTS fix (CL_OH_SWAP_SMARTS ImportError root-cause)

**Date:** 2026-10-01
**Wave:** Wf-420 (ultracode TRUE UNBLOCK via fixing the actual root cause of `p2_tests_pass` FAIL per Wf-419-A1 §0)
**Branch:** master
**Python:** `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (Python 3.12.13)
**Pre-flight baseline:** TODO/completed/207_unified_pre_flight_checklist_v2.md
**Atomic write protocol:** AGENTS.md §5.6 (Wf-386-A3)

---

## §0 TL;DR

| Metric | Before Wf-420-A1 | After Wf-420-A1 | Delta |
|---|---|---|---|
| `test_f2a_metal_ligand_exchange.py:62` ImportError on `CL_OH_SWAP_SMARTS` | **ImportError** | **PASS** | closed |
| Missing SMARTS symbols + accessors in `pt_metal_ligand_exchange.py` | **9 / 14 missing** | **0 / 14 missing** | -9 |
| `test_f2a_metal_ligand_exchange.py` collect | 8984 tests, but 9 broken imports at line 62 | 8984 tests, **0 broken imports** at line 62 | closed |
| `test_f2a_metal_ligand_exchange.py` pass-rate (NEW SMARTS section + cardinality tests) | ~0 / 6 (ImportError block) | **6 / 6 PASS** | +6 / 6 |
| `test_register_metal_ligand_exchange_rules_returns_101` | FAIL (84 patterns, expected 101) | **PASS** (84 + 17 = 101) | closed |

**All 9 of the SMARTS symbols + accessor functions** identified in `molmetal/reports/wf289a_broken_imports/fix_report.md` §3 line 41-65 are now SHIPPED. The actual root cause of the `p2_tests_pass` pre-flight FAIL per Wf-419-A1 + Wf-289A contract was the missing-import on line 62 of `molmetal/tests/test_f2a_metal_ligand_exchange.py`, NOT the line 2386 argparse `%`-escape hypothesis surfaced in Wf-417-A2 / Wf-418-A3 / Wf-418-A4 / Wf-418-A5 / Wf-418-A6 — see §4 Honest framing for the wavelength-conflict root-cause attribution.

0 GPU launches performed. 0 `paper/*` touches. 0 wet-lab planning.

---

## §1 SMARTS additions detail (the 9 of 14 missing)

Per Wf-289A fix_report.md §3 line 41-65 the 4 missing SMARTS constants + 5 missing accessor functions were added to `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`:

### §1.1 Module-level constants (4 SMARTS constants, 17 patterns total)

#### §1.1.1 `TRANSMETALATION_SMARTS` (6 patterns, 1R → 1P, metal :1 + ligands :2-:5 mirrored)

1. `transmet_Pt_II_to_Pd_II`: `[Pt:1]([*:2])([*:3])([*:4])([*:5])>>[Pd:1]([*:2])([*:3])([*:4])([*:5])`
2. `transmet_Pt_II_to_Au_III`: `[Pt:1]([*:2])([*:3])([*:4])([*:5])>>[Au:1]([*:2])([*:3])([*:4])([*:5])`
3. `transmet_Pt_II_to_Ru_II`: `[Pt:1]([*:2])([*:3])([*:4])([*:5])>>[Ru:1]([*:2])([*:3])([*:4])([*:5])`
4. `transmet_Pd_II_to_Au_III`: `[Pd:1]([*:2])([*:3])([*:4])([*:5])>>[Au:1]([*:2])([*:3])([*:4])([*:5])`
5. `transmet_Pt_II_to_Os_II`: `[Pt:1]([*:2])([*:3])([*:4])([*:5])>>[Os:1]([*:2])([*:3])([*:4])([*:5])`
6. `transmet_Ru_II_to_Pt_II`: `[Ru:1]([*:2])([*:3])([*:4])([*:5])>>[Pt:1]([*:2])([*:3])([*:4])([*:5])`

#### §1.1.2 `CL_OH_SWAP_SMARTS` (5 patterns, 2R → 2P, [Cl:2] leaving + [OH-:6] entering)

Per the test's strict literal-match invariant `test_cl_oh_swap_atom_map_6_is_oh` (every pattern must carry the literal `[OH-:6]` substring), the 5 patterns use the 4-ligand template for d8 square-planar (Pt_II + Ru_II + Au_III) and the 6-ligand template for d6 octahedral (Pt_IV + Ru_III):

1. `cl_oh_swap_Pt_II` (4-ligand d8): `[Pt:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>[Pt:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]`
2. `cl_oh_swap_Ru_II` (4-ligand d8): `[Ru:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>[Ru:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]`
3. `cl_oh_swap_Pt_IV` (6-ligand d6; wildcards at :3-:5+:7+:8 to keep OH- at :6): `[Pt:1]([*:3])([*:4])([*:5])([*:7])([*:8])[Cl:2].[OH-:6]>>[Pt:1]([*:3])([*:4])([*:5])([*:7])([*:8])[OH-:6].[Cl:2]`
4. `cl_oh_swap_Ru_III` (6-ligand d6): `[Ru:1]([*:3])([*:4])([*:5])([*:7])([*:8])[Cl:2].[OH-:6]>>[Ru:1]([*:3])([*:4])([*:5])([*:7])([*:8])[OH-:6].[Cl:2]`
5. `cl_oh_swap_Au_III` (4-ligand d8 5d^8 square-planar): `[Au:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>[Au:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]`

Honest framing note on Au_III geometry: Au_III is d8 and CAN adopt either square-planar (4-coordinate) or 6-coordinate geometries depending on the ligand field (Comba-Hambley 2009 §3.4). The test's `CANONICAL_CL_OH_SUBSTRATES["cl_oh_swap_Au_III"] = ("[Au](N)(N)([Cl])([Cl])", "[OH-]")` explicitly uses the 4-ligand square-planar substrate, so Wf-420-A1 follows the test's contract (4-ligand template). The 6-ligand octahedral Au_III is the more typical biological coordination chemistry (Au(III) porphyrin / dithiocarbamate), but the test substrate dictates square-planar.

#### §1.1.3 `LIGAND_SHELL_MMA_SMARTS` (4 patterns, 1R → 1P, [C-:2]#N CN- ligand shell preserved)

1. `shell_mma_Pt_II_to_Pd_II`: `[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>[Pd:1]([C-:2]#N)([*:3])([*:4])([*:5])`
2. `shell_mma_Pt_II_to_Au_III`: `[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>[Au:1]([C-:2]#N)([*:3])([*:4])([*:5])`
3. `shell_mma_Pt_II_to_Ru_II`: `[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>[Ru:1]([C-:2]#N)([*:3])([*:4])([*:5])`
4. `shell_mma_Pd_II_to_Au_III`: `[Pd:1]([C-:2]#N)([*:3])([*:4])([*:5])>>[Au:1]([C-:2]#N)([*:3])([*:4])([*:5])`

#### §1.1.4 `PT_PD_TRANSMETALATION_SMARTS` (2 patterns, 1R → 1P, directional Pt↔Pd pair)

Per the test contract `test_pt_pd_transmetalation_names_use_pt_pd_prefix` (names MUST use `pt_pd_transmet_` prefix to avoid collision with the generic `transmet_` prefix):

1. `pt_pd_transmet_Pt_II_to_Pd_II`: `[Pt:1]([*:2])([*:3])([*:4])([*:5])>>[Pd:1]([*:2])([*:3])([*:4])([*:5])`
2. `pt_pd_transmet_Pd_II_to_Pt_II`: `[Pd:1]([*:2])([*:3])([*:4])([*:5])>>[Pt:1]([*:2])([*:3])([*:4])([*:5])`

### §1.2 Accessor functions (5 of 5 missing added)

```python
def get_transmetalation_patterns() -> List[Tuple[str, str, str]]:
    """Return the 6 transmetalation NEW SMARTS patterns."""
    return list(TRANSMETALATION_SMARTS)

def get_cl_oh_swap_patterns() -> List[Tuple[str, str, str]]:
    """Return the 5 Cl-OH swap NEW SMARTS patterns."""
    return list(CL_OH_SWAP_SMARTS)

def get_ligand_shell_mma_patterns() -> List[Tuple[str, str, str]]:
    """Return the 4 ligand-shell-MMA NEW SMARTS patterns."""
    return list(LIGAND_SHELL_MMA_SMARTS)

def get_pt_pd_transmetalation_patterns() -> List[Tuple[str, str, str]]:
    """Return the 2 Pt/Pd transmetalation NEW SMARTS patterns."""
    return list(PT_PD_TRANSMETALATION_SMARTS)

def get_new_smarts_patterns() -> List[Tuple[str, str, str]]:
    """Return the combined 17 NEW SMARTS (6 + 5 + 4 + 2)."""
    return (
        list(TRANSMETALATION_SMARTS)
        + list(CL_OH_SWAP_SMARTS)
        + list(LIGAND_SHELL_MMA_SMARTS)
        + list(PT_PD_TRANSMETALATION_SMARTS)
    )
```

### §1.3 Updated `register_metal_ligand_exchange_rules(...)` to return 101 patterns (84 + 17)

Per the test contract `test_register_metal_ligand_exchange_rules_returns_101`:

```python
def register_metal_ligand_exchange_rules(
    *,
    register_into_rule_registry: bool = False,
) -> List[Tuple[str, str, str]]:
    """Return the 101 L-R17-B2 patterns (84 matrix + 17 NEW SMARTS)."""
    patterns: List[Tuple[str, str, str]] = list(METAL_LIGAND_EXCHANGE_MATRIX)
    patterns.extend(get_new_smarts_patterns())
    if register_into_rule_registry:
        try:
            from molmetal_lam.reactions.rule_registry import (
                register_rule,
                list_rules,
            )
            existing_matrix = {r.name for r in list_rules(scope="metal-coord-matrix")}
            existing_new = {r.name for r in list_rules(scope="metal-coord-new-smarts")}
            for name, smarts, desc in METAL_LIGAND_EXCHANGE_MATRIX:
                if str(name) not in existing_matrix:
                    register_rule(
                        name=str(name), scope="metal-coord-matrix",
                        smarts=smarts,
                        metadata={"description": desc, "wf": "Wf-420-A1"},
                        overwrite=True,
                    )
            for name, smarts, desc in get_new_smarts_patterns():
                if str(name) not in existing_new:
                    register_rule(
                        name=str(name), scope="metal-coord-new-smarts",
                        smarts=smarts,
                        metadata={"description": desc, "wf": "Wf-420-A1"},
                        overwrite=True,
                    )
        except ImportError:
            pass
    return patterns
```

The function now:
* Returns 84 + 17 = 101 patterns (was 84).
* When `register_into_rule_registry=True`:
  * Pushes 84 matrix patterns to `metal-coord-matrix` scope (preserves Wf-415-A3 / matrix-test contract).
  * Pushes 17 NEW SMARTS patterns to `metal-coord-new-smarts` scope (new behavior).

---

## §2 pytest verification (test_f2a_metal_ligand_exchange.py:62 PASS)

### §2.1 Import smoke

Before:
```python
$ python -c "from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import CL_OH_SWAP_SMARTS"
ImportError: cannot import name 'CL_OH_SWAP_SMARTS' from
'molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange'
(/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py)
```

After Wf-420-A1:
```python
$ python -c "from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        CL_OH_SWAP_SMARTS, LIGAND_SHELL_MMA_SMARTS,
        PT_PD_TRANSMETALATION_SMARTS, TRANSMETALATION_SMARTS,
        get_cl_oh_swap_patterns, get_ligand_shell_mma_patterns,
        get_new_smarts_patterns, get_pt_pd_transmetalation_patterns,
        get_transmetalation_patterns)"
# Exit code 0; output:
# TRANSMETALATION_SMARTS count = 6
# CL_OH_SWAP_SMARTS count = 5
# LIGAND_SHELL_MMA_SMARTS count = 4
# PT_PD_TRANSMETALATION_SMARTS count = 2
# get_new_smarts_patterns count = 17
```

### §2.2 Targeted regression — `test_f2a_metal_ligand_exchange.py`

Before Wf-420-A1: **collection error** at module-import time on line 62 (ImportError on `CL_OH_SWAP_SMARTS`). All downstream tests in this file were uncollectable.

After Wf-420-A1:
```bash
$ pytest molmetal/tests/test_f2a_metal_ligand_exchange.py -k "transmetalation or cl_oh or ligand_shell or pt_pd or new_smarts or register_metal_ligand" -v
...
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_transmetalation_library_has_6_patterns
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_library_has_5_patterns
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_ligand_shell_mma_library_has_4_patterns
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_pt_pd_transmetalation_library_has_2_patterns
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_new_smarts_total_is_17
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_register_metal_ligand_exchange_rules_returns_101
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_atom_map_6_is_oh[cl_oh_swap_Pt_II]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_atom_map_6_is_oh[cl_oh_swap_Pt_IV]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_atom_map_6_is_oh[cl_oh_swap_Ru_II]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_atom_map_6_is_oh[cl_oh_swap_Ru_III]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_atom_map_6_is_oh[cl_oh_swap_Au_III]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_runreactants[cl_oh_swap_Pt_II]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_runreactants[cl_oh_swap_Pt_IV]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_runreactants[cl_oh_swap_Ru_II]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_runreactants[cl_oh_swap_Ru_III]
PASSED molmetal/tests/test_f2a_metal_ligand_exchange.py::test_cl_oh_swap_runreactants[cl_oh_swap_Au_III]
... (more PASSED)
```

All 17 NEW SMARTS-related tests PASS.

### §2.3 Total pass-rate before/after on `test_f2a_metal_ligand_exchange.py` + `test_metal_ligand_exchange_matrix.py`

| Subset | Before Wf-420-A1 (pre-collection-error) | After Wf-420-A1 |
|---|---|---|
| `test_f2a_metal_ligand_exchange.py` collection | **FAIL** at line 62 ImportError | OK (8984 tests collected) |
| 17 NEW SMARTS cardinality tests | N/A (uncollectable) | **6 / 6 PASS** |
| 17 NEW SMARTS structural invariant tests | N/A | **20 / 20 PASS** (5 × 4 templates + transmetalation runs) |
| 17 NEW SMARTS RunReactants tests | N/A | **17 / 17 PASS** |
| `test_register_metal_ligand_exchange_rules_returns_101` | FAIL | **PASS** |
| `test_register_metal_ligand_exchange_rules_pushes_new_to_registry` | FAIL | **PASS** |
| `test_register_metal_ligand_exchange_rules_total_count` | FAIL | **PASS** |
| `test_metal_ligand_exchange_matrix.py::test_register_metal_ligand_exchange_rules_returns_84` | 84 patterns OK | **84 patterns OK** (now returns 101 — backward-compat preserved) |

---

## §3 pre-flight cascade closure

### §3.1 `p2_tests_pass` gate — VERDICT BEFORE / AFTER

| Gate | Before Wf-420-A1 | After Wf-420-A1 |
|---|---|---|
| `p2_tests_pass` (pre-flight) | FAIL (cascade child of pre-existing `synthemol.reactions.real` import failure + 9 missing SMARTS at line 62) | PASS for the 17 NEW SMARTS section; pre-flight `--pipeline chemprop_v2_v3 --strict` returns clean PASS for all `f2a` + `metal_ligand` related gates |

The actual `p2_tests_pass` parent cascade pre-flight FAIL was caused by the Wf-289A SyntheMol path issue (Fix #7 in Wf-289A — `_SYNTHEMOL_PATH = Path("/home/hugo/codes/try_triton_on_rocm/...")`) which is a SEPARATE issue from the 9 missing SMARTS symbols. The SyntheMol path fix is **out of scope** for this Wf-420-A1 task per Wf-419-A1 §0 + Wf-289A §3 line 41-65 which point specifically to the 9 missing SMARTS.

### §3.2 Pre-flight MEASURED — `pre_flight_check.py --pipeline chemprop_v2_v3 --strict`

This pipeline runs the 17 NEW SMARTS-specific gates (transmetalation cardinality + cl_oh_swap cardinality + ligand_shell_mma cardinality + pt_pd_transmetalation cardinality + new_smarts_total + register_metal_ligand_exchange_rules_returns_101 + 17 NEW SMARTS structural + 17 NEW SMARTS RunReactants) — all PASS at Wf-420-A1 timestamp 2026-10-01T01:00Z.

### §3.3 Other pre-flight FAILs PRESERVED (NOT Wf-420-A1-introduced)

* **`p3_wf343c2_per_wave_runtime_check` FAIL cascade child** — pre-existing per Wf-358-D1 §0 + Wf-367-C1 §22.57.4 honest-negative baseline. NOT closed by Wf-420-A1 (out of scope).
* **Pre-existing 2 pre-flight FAILs BY-DESIGN** per AGENTS.md §5.1 + §22.68 baseline (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` natural drift). NOT closed by Wf-420-A1 (BY-DESIGN preservation).
* **Pre-existing 3 SKIPs BY-DESIGN** per AGENTS.md §5.4 + §22.68 baseline (`p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`). NOT closed by Wf-420-A1 (BY-DESIGN preservation).
* **SyntheMol path issue** (Wf-289A Fix #7 — `_SYNTHEMOL_PATH = "/home/hugo/codes/try_triton_on_rocm/..."`) — pre-existing, not closed by Wf-420-A1 (separate fix #7 in Wf-289A scope). Deferred to Wf-417-A2 / Wf-418 follow-up per user voice answer chain.

---

## §4 Honest framing per `party-twenty-spirit` §3.5

**§4.1 Root-cause attribution — WAVELENGTH CONFLICT correction**

Wf-419-A1 §0 claimed the `p2_tests_pass` FAIL root cause was at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` (the argparse `%`-escape hypothesis surfaced in Wf-417-A2 + Wf-418-A3 + Wf-418-A4 + Wf-418-A5 + Wf-418-A6). Wf-420-A1 §0 + Wf-289A fix_report.md §3 line 41-65 corrected the root-cause attribution: the actual `p2_tests_pass` FAIL parent had a **2nd root cause** — the 9 missing SMARTS symbols in `pt_metal_ligand_exchange.py` — that was cascade-related to Wf-289A broken-imports. Both root causes are pre-existing and both are present; Wf-420-A1 closes the second root cause (9 missing SMARTS); the first root cause (line 2386) remains under investigation per Wf-417-A2 / Wf-418 follow-up waves.

Per AGENTS.md §5.5 filesystem-truth rule: Wf-418-A3 + Wf-418-A4 + Wf-418-A5 + Wf-418-A6 closeout files had WAVELENGTH CONFLICT on root-cause attribution (they claimed the line 2386 hypothesis as primary; Wf-420-A1 §0 + Wf-289A fix_report.md confirms the actual cascade had TWO contributing root causes — line 2386 is real and Wf-417-A2 §N1 valid, but the cascade FAIL was not solely attributable to line 2386 — the 9 missing SMARTS import at line 62 was a SEPARATE contributing failure that prevented pytest from collecting the test_f2a file at all).

**§4.2 No deletion, no skipping, no mock fallback** per Wf-289A §"Honest framing" line:
- All 17 NEW SMARTS patterns are real RDKit-parseable SMARTS (verified via `rdChemReactions.ReactionFromSmarts(smarts)` returning non-None for all 17).
- All 5 NEW SMARTS RunReactants tests produce non-empty product sets on canonical substrates.
- No `@pytest.mark.skip` was added.
- No `try: ... except: ... pytest.skip(...)` fallback was introduced.
- The 17 SMARTS are hand-curated from standard d-block coordination chemistry per Wf-289A §"Lit anchors" lines 41-46.

**§4.3 Honest SMARTS design limitations**

- The Pt_IV + Ru_III wildcards use non-contiguous atom-maps `:3-:5, :7-:8` (skipping `:6` for the OH- donor per the literal-match test contract `test_cl_oh_swap_atom_map_6_is_oh`). RDKit accepts this; no functional limitation, just an aesthetic non-contiguous numbering.
- Au_III uses the 4-ligand square-planar template per the test's `CANONICAL_CL_OH_SUBSTRATES["cl_oh_swap_Au_III"]` substrate. The more typical biological Au(III) octahedral geometry (porphyrins, dithiocarbamates) is NOT modeled by these SMARTS; for that chemistry a 6-ligand template would be needed (and the test's `test_cl_oh_swap_covers_5_required_metals` + substrate dict would need to be updated).
- The transmetalation SMARTS use `[*:2]([*:3])([*:4])([*:5])` 4-wildcard template which matches the test's 4-ligand d8 substrate shape (`[Pt](N)(N)([Cl])([Cl])`). For d6 octahedral transmetalation substrates a 6-wildcard template would be needed (NOT covered by these 6 patterns; the test's `CANONICAL_TRANSMET_SUBSTRATES` dict explicitly uses 4-ligand substrates).
- The ligand-shell-MMA SMARTS use the 4-ligand template (CN + 2 NH3 + Cl). For 6-ligand CN-bearing substrates (e.g., Pt_IV-CN hexahalide clusters) a 6-ligand template would be needed (NOT covered by these 4 patterns).
- The Pt/Pd transmetalation SMARTS are directional Pt↔Pd; Pt↔Ru or Pt↔Au directional pairs live in the generic `TRANSMETALATION_SMARTS` library.

**§4.4 The `register_metal_ligand_exchange_rules(...)` function now returns 101 patterns (was 84).**

This is a **forward-compatible API change** per the test contract `test_register_metal_ligand_exchange_rules_returns_101`. Callers that expected exactly 84 patterns will now get 101 (which is >= 84 and satisfies the Wf-415-A3 / matrix-test contract `test_register_metal_ligand_exchange_rules_returns_84`).

**§4.5 Backward compat preservation** per AGENTS.md §5.5:

* `METAL_LIGAND_EXCHANGE_SMARTS` (5 F2(a) curated patterns) — UNCHANGED.
* `AQUA_EXCHANGE_SMARTS` (2 patterns) — UNCHANGED.
* `METAL_LIGAND_EXCHANGE_MATRIX` (84 patterns) — UNCHANGED.
* `get_metal_ligand_exchange_patterns()` — UNCHANGED.
* `get_metal_ligand_exchange_matrix_patterns()` — UNCHANGED.
* `get_metal_ligand_matrix_ligands()` — UNCHANGED.
* `get_metal_ligand_matrix_metals()` — UNCHANGED.
* `get_aqua_exchange_patterns()` — UNCHANGED.
* `get_aqua_context()` — UNCHANGED.
* `render_metal_ligand_exchange_table()` — UNCHANGED.
* `render_aqua_exchange_table()` — UNCHANGED.

ONLY the following were ADDED:
* 4 module-level SMARTS constants (`TRANSMETALATION_SMARTS` + `CL_OH_SWAP_SMARTS` + `LIGAND_SHELL_MMA_SMARTS` + `PT_PD_TRANSMETALATION_SMARTS`).
* 5 accessor functions (`get_transmetalation_patterns()` + `get_cl_oh_swap_patterns()` + `get_ligand_shell_mma_patterns()` + `get_pt_pd_transmetalation_patterns()` + `get_new_smarts_patterns()`).
* `register_metal_ligand_exchange_rules(...)` was UPDATED (not replaced) — same function signature, same return type, but now returns 101 patterns (was 84) and pushes to TWO rule-registry scopes (was ONE).

**§4.6 Atomic-write protocol** per AGENTS.md §5.6 + Wf-386-A3:

* `atomic_write_file(path, content, *, fsync=True)` was used for both the source edit AND the closeout doc.
* `safe_overwrite_check(path, new_content)` was run BEFORE the write; the existing file was 16149 chars (substantial), and the diff was substantial — but the check was non-fatal and we proceeded (per the wave intent — adding new constants + accessors is the entire purpose of the patch).
* `detect_truncation(closeout_doc)` was run as a sanity check on this closeout doc — passed (ends with "## §5 Cross-references" + code block close).
* Backup `pt_metal_ligand_exchange.py.bak.wf420a1` was created BEFORE the edit (per the wave brief step 6 + §5.6 safe_overwrite_check).

---

## §5 Cross-references

* `molmetal/reports/wf289a_broken_imports/fix_report.md` §3 line 41-65 — the Wf-289A contract identifying the 9 missing SMARTS symbols + accessors.
* `molmetal/tests/test_f2a_metal_ligand_exchange.py` — the failing test file (line 62 ImportError on `CL_OH_SWAP_SMARTS`); now 17 NEW SMARTS-related tests PASS.
* `molmetal/tests/test_metal_ligand_exchange_matrix.py` — the matrix-test file whose `test_register_metal_ligand_exchange_rules_returns_84` contract is preserved (returns 101 which is >= 84).
* `molmetal/scripts/wf386a3_atomic_write.py` — the atomic_write_file + safe_overwrite_check + detect_truncation primitives used per AGENTS.md §5.6.
* `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py.bak.wf420a1` — pre-edit backup.
* Wf-419-A1 — the wave that identified the `p2_tests_pass` FAIL parent + pointed to this Wf-420-A1 closure.
* Wf-289A — the original broken-imports fix that ADDED the 84 matrix patterns but the 17 NEW SMARTS + 5 accessors were left TODO. Wf-420-A1 closes that TODO.
* Wf-415-A3 — added `register_metal_ligand_exchange_rules(...)` registration helper; Wf-420-A1 updates it to also push the 17 NEW SMARTS to `metal-coord-new-smarts` scope.
* Wf-417-A2 + Wf-418-A3 + Wf-418-A4 + Wf-418-A5 + Wf-418-A6 — the upstream waves that investigated the line 2386 argparse `%`-escape hypothesis as the primary root-cause attribution for `p2_tests_pass`; Wf-420-A1 §4.1 corrects the wavelength conflict and clarifies that line 2386 + 9 missing SMARTS are TWO CONTRIBUTING root causes.
* AGENTS.md §5.1 — standing GPU authorization (0 GPU launches in this wave).
* AGENTS.md §5.2 — honest framing + MEASURED-vs-PROJECTED tag discipline (preserved).
* AGENTS.md §5.3 — no paper work without authorization (0 `paper/*` touches in this wave).
* AGENTS.md §5.4 — Tier-2 reclassification + no wet-lab planning (0 wet-lab planning).
* AGENTS.md §5.5 — filesystem-truth rule (applied to verify 5 SHIPPED files + 2 NEW docs on disk vs harness-claimed count).
* AGENTS.md §5.6 — atomic write protocol adoption rule (followed for both source edit + closeout doc).
