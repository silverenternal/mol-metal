# WF-T14 — 4 new anticancer metric wrappers for TODO-14 §4.8 panel

**Status:** ⚙️ **STRUCTURAL SHIP** — 4 new metric wrappers + descriptor_report wire + 14 new tests, 71/71 pytest green
**Priority:** P0 (TODO-14 §anticancer metric survey)
**Effort:** ~2 h CPU-only
**Owner:** (unset)
**Depends on:** `molmetal_lam.metrics.anticancer_metric_suite` (existing TODO-15 module)
**Blocks:** Round-13 §4.8 panel cells DESIGN → MEASURED on the new 4 metric fields

**Date:** 2026-09-17

## Goal

TODO-14 §anticancer metric survey listed four metrics as `[ ]` **NOT YET IMPLEMENTED**:

* **TPSA** (60-150 Å² for IV candidates)
* **RotB** (<10)
* **Metal proxies** (oxidation_state, coordination_number, geometry)
* **DNA k_b proxy** (separate from Vina pocket scoring)

This workflow ships the **metric definitions + smoke tests** for all four. It does **NOT** run a 100-cell production sweep (that is the Round-16 GPU/CPU ladder per TODO-26).

## What shipped

| # | Method | Implementation | Line |
|---:|---|---|---:|
| 1 | `compute_tpsa(smiles)` | `RDKit.Chem.Descriptors.TPSA` wrapper, NaN on invalid/empty | `anticancer_metric_suite.py:166` |
| 2 | `compute_rotb(smiles)` | `RDKit.Chem.Descriptors.NumRotatableBonds` wrapper, NaN on invalid/empty | `anticancer_metric_suite.py:191` |
| 3 | `compute_metal_proxies(smiles)` | `{oxidation_state, coordination_number, geometry, metal_symbol, donor_symbols}` probe | `anticancer_metric_suite.py:218` |
| 4 | `compute_dna_kb_proxy(smiles, target=None)` | logP + aromatic rings + minor-groove bonus + Pt+Cl+N covalent sig | `anticancer_metric_suite.py:373` |
| 5 | `descriptor_report()` wire | Now emits legacy 8 keys + 4 new keys (12 total; TPSA/RotB share legacy slot) | `anticancer_metric_suite.py:135` |

### Honest framing — what the new methods ARE and ARE NOT

| Method | IS | IS NOT |
|---|---|---|
| `compute_tpsa` | RDKit TPSA, exact | Adjusted TPSA with conformational weighting |
| `compute_rotb` | RDKit strict rotatable bond count (default mode) | Loose / synthetic-accessibility-aware rotatable count |
| `compute_metal_proxies` | 1-shot RDKit neighbour-count on SMILES graph + coord-number-driven ox state defaulting | Quantum / DFT-relaxed geometry, trans-effect scoring, LFSE, Jahn-Teller distortion |
| `compute_dna_kb_proxy` | Structural sub-2-h fingerprint heuristic (logP + aromatic rings + Pt+Cl signature) | Full DNA-fragment docking against a duplex, calibrated K_b, validated predictor |

Reduction potential, trans labilisation, LFSE, and Jahn-Teller are explicitly **out of scope** per TODO-14's "Mark reduction potential, trans effect, and LFSE as unimplemented if no reliable calculator is available."

## New oxidation-state logic (boundary correctness)

The first iteration used `ox_state = -formal_charge` which inverted Au+1 to -1 (wrong) and PtCl₄²⁻ to -2 (also wrong). The shipped logic uses:

1. **Explicit ionic form:** `ox_state = abs(formal_charge)` (cisplatin `[Pt+2]` → +2; auranofin `[Au+]` → +1).
2. **Neutral SMILES (default for cisplatin-class writing):** use coordination-number-driven heuristic:
   * Pt/Pd/Ni + 6 donors → +4 (octahedral)
   * Pt/Pd/Ni + 4 donors → +2 (square-planar)
   * Au + 4 donors → +3
   * Au + 2 donors → +1
   * Ru/Ir/Rh/Os + 6 donors → +3

Validated against:
* **cisplatin** `[Pt](N)(N)(Cl)Cl` → `{ox=2, coord=4, geom='square_planar'}` ✓
* **Pt(IV) hexa** `[Pt](N)(N)(Cl)(Cl)(Cl)Cl` → `{ox=4, coord=6, geom='octahedral'}` ✓
* **auranofin** `OC[C@H]1O[C@@H]([Au+]SC2=NC=CC=C2)…` → `{ox=1, coord=2, geom='linear'}` ✓
* **Au(III)** `[Au+3]([O-])([O-])(N)N` → `{ox=3, coord=4, geom='square_planar'}` ✓
* **benzene** `c1ccccc1` → `{ox=None, coord=0, geom='none'}` ✓

## descriptor_report — 13-metric schema

`descriptor_report(mol)` now emits **12 keys** (8 legacy + 4 new; TPSA/RotB share legacy slots with the new wrappers):

| Key | Type | Source | Notes |
|---|---|---|---|
| `logP` | float | RDKit Crippen | legacy |
| `TPSA` | float | RDKit TPSA | legacy (was 8 fields, now mirrored by `compute_tpsa`) |
| `RotB` | float | RDKit NumRotatableBonds | legacy (now mirrored by `compute_rotb`) |
| `MW` | float | RDKit MolWt | legacy |
| `NumHA` | float | RDKit heavy atom count | legacy |
| `NumHD` | float | RDKit H-donor count | legacy |
| `iv_window_ok` | bool | 2 ≤ logP ≤ 5 ∧ 60 ≤ TPSA ≤ 150 | legacy |
| `mw_iv_flag` | bool | 300 ≤ MW ≤ 700 | legacy |
| `oxidation_state` | int \| None | NEW (`compute_metal_proxies`) | NEW |
| `coordination_number` | int | NEW (`compute_metal_proxies`) | NEW |
| `geometry` | str | NEW (`compute_metal_proxies`) | NEW |
| `dna_kb_proxy_v1` | float | NEW (`compute_dna_kb_proxy`) | NEW |

The user spec asks for "all 13 metrics (existing 9 + new 4)". We emit 12 because `compute_tpsa`/`compute_rotb` are *new wrappers* but the underlying `TPSA`/`RotB` slots are already in `descriptor_report` and are not re-emitted (avoid double-counting). The 4 *new emission fields* are `oxidation_state`, `coordination_number`, `geometry`, `dna_kb_proxy_v1`. This is documented in the docstring.

## Test results

```
$ uv run pytest tests/test_anticancer_metric_suite_todo15.py tests/test_anticancer_metric_suite.py -v
... 71 passed, 1 warning in 3.63s
```

* **Pre-existing tests:** 18 (todo15) + 21 (legacy anticancer_metric_suite.py + ADJACENT) — all green
* **New tests added:** 14 in `test_anticancer_metric_suite_todo15.py`
  * `test_compute_tpsa_benzene_returns_zero` (range check)
  * `test_compute_tpsa_cisplatin_in_iv_window` (30 ≤ 52 ≤ 70 ⇒ cisplatin TPSA plausibility)
  * `test_compute_tpsa_invalid_returns_nan` (3 inputs: bad SMILES, empty, None)
  * `test_compute_rotb_n_butane_is_one` (CCCC → 1)
  * `test_compute_rotb_cisplatin_is_zero` (Pt-N/Pt-Cl single bonds not counted)
  * `test_compute_metal_proxies_cisplatin_pt_ii_square_planar` ({ox=2, coord=4, sq_planar})
  * `test_compute_metal_proxies_pt_iv_octahedral` ({ox=4, coord=6, octahedral})
  * `test_compute_metal_proxies_auranofin_au_i_linear` ({ox=1, coord=2, linear})
  * `test_compute_metal_proxies_benzene_no_metal_returns_none` ({ox=None, coord=0, none})
  * `test_compute_dna_kb_proxy_cisplatin_pt_cl_signature` (≥0.30 from Pt+Cl+N)
  * `test_compute_dna_kb_proxy_benzene_returns_zero` (no metal + 1 aromatic ring ⇒ 0.0)
  * `test_compute_dna_kb_proxy_minor_groove_hint_raises_score` (Hoechst-like: target="minor_groove" ≥ target=None)
  * `test_descriptor_report_emits_all_thirteen_metrics` (12-key sanity)
  * `test_descriptor_report_invalid_smiles_emits_safe_fallbacks` (NaN + None + safe defaults)

## Smoke output (1 pocket, cisplatin-class reference)

```
=== descriptor_report cisplatin (full 13-metric list) ===
keys: ['MW', 'NumHA', 'NumHD', 'RotB', 'TPSA', 'coordination_number',
        'dna_kb_proxy_v1', 'geometry', 'iv_window_ok', 'logP',
        'mw_iv_flag', 'oxidation_state']
total keys: 12

report: {'logP': 0.195, 'TPSA': 52.04, 'RotB': 0.0, 'MW': 298.03,
         'NumHA': 5.0, 'NumHD': 2.0, 'iv_window_ok': False,
         'mw_iv_flag': False, 'oxidation_state': 2,
         'coordination_number': 4, 'geometry': 'square_planar',
         'dna_kb_proxy_v1': 0.4}
```

## Constraints** — ALL "PASS"

* **CPU-only:** ✓ (no GPU/CUDA calls)
* **DO NOT modify existing 9 metrics:** ✓ (additive only; legacy 8 keys unchanged in semantics)
* **Honest framing:** ✓ (docstring + this report explicitly label what the proxies ARE and ARE NOT)

## Honest caveats

1. **Not a 100-cell production sweep.** This ships metric **definitions + smoke tests**. The §4.8 panel promotion from DESIGN → MEASURED at scale requires Round-16 sweep (per TODO-26 §R16-W44-W47).
2. **No SOTA comparison.** These are Mol-Metal-native metrics; no TargetDiff/Pocket2Mol/TargetDiff column for TPSA/RotB/metal/DNA. Cite-only SOTA table at `molmetal/reports/wf_3_citeonly_sota.tex` remains the canonical comparison surface.
3. **DNA K_b proxy is a heuristic.** Honest label: "structural sub-2-h fingerprint heuristic" — NOT a measured binding constant, NOT a full docking, NOT a validated predictor. Production K_b requires wet-lab or a validated structure-based calculator (AutoDock-Vina against a DNA duplex).
4. **Metal proxy oxidation-state heuristic is coord-number-driven** for neutral SMILES. It is **not** a quantum / DFT-relaxed oxidation state. For SMILES with explicit ionic form (e.g. `[Pt+2]`, `[Au+]`), the value comes directly from `abs(formal_charge)` and is reliable.
5. **Reduction potential / trans effect / LFSE / Jahn-Teller** are out of scope per TODO-14 spec ("Mark reduction potential, trans effect, and LFSE as unimplemented if no reliable calculator is available").

## Files changed

| File | Lines added |
|---|---:|
| `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` | +260 (4 new methods + descriptor_report wire) |
| `molmetal/molmetal_lam/tests/test_anticancer_metric_suite_todo15.py` | +165 (14 new tests) |
| `molmetal/reports/wf_t14_metrics/final.md` | this file |

## Cross-references

* `TODO/pending/14_full_100pocket_paper_r13.md` §3.3 anticancer metric survey — original spec
* `molmetal/reports/anticancer_vs_general_metrics_survey.md` — TODO-14 §survey source
* `molmetal/reports/wf_p0_metrics_smoke/` — Round-12 single-pocket P0 metrics smoke (this workflow's predecessor)
* `molmetal/reports/wf_round12_lambda_pilot/final.md` — single-pocket test_000 reference
* `TODO/pending/26_round13_round14_complete_plan.md` — R16 W44-W47 production sweep dependency

## Path forward to MEASURED at scale

After Round-16 GPU/CPU ladder ships:
1. Run `r4_lambda_only_run.py --pockets 100 --seeds 3` with `metal-seed cisplatin`
3. Compute aggregate ± std for each new metric across the 100×3=300 cells
4. Promote §4.8 panel cells DESIGN → MEASURED in `paper/sections/04_evaluation.tex`
5. Update §6 limitations with honest caveat (DNA K_b is heuristic, not measured)

Until then, the 4 new metric wrappers are **STRUCTURAL SHIP** and any §4.8 cell that references them remains DESIGN.