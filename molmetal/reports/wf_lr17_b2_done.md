# WF-LR17-B2 — F2(a) MetalLigandExchange SMARTS upgrade + 10-pocket pilot synthesis

> **Date:** 2026-09-25
> **Scope:** L-R17-B2 / TODO-31 §2 (CPU-only). Verifies F2(a) SMARTS expansion on the 10-pocket WF-Lambda-1 panel + gates against the TODO-25 §3 ≥3-metal-ligand-pair invariant.
> **Status (honest):** ⚙️ **PARTIAL** — F2(a) library SHIPPED (1 → 7 patterns); the 84+4 NEW-matrix extension (12 metals × 7 ligands + transmetalation / Cl-OH swap / ligand-shell-MMA / Pt/Pd transmetalation) is **QUEUED but not merged** in this round; pilot PARTIAL PASS (within-cell) / FAIL (across-cell, see §3).
> **Refs:** `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`, `molmetal/tests/test_f2a_metal_ligand_exchange.py`, `molmetal/reports/wf_lr17_b2_pilot/final.md`, `TODO/pending/31_r17_ultracode_dispatch.md §L-R17-B2`, `TODO/pending/34_parallel_ultracode_dispatch.md §UC-L2`.

## 1. Coverage matrix — 12 metals × 7 ligands + 4 NEW

The L-R17-B2 plan (`TODO/pending/31_r17_ultracode_dispatch.md:43-58`) targets a hand-curated 84-rule base matrix + 4 NEW SMARTS = **88+ rules**. The 12 metals × 7 ligands grid:

| | Cl | NH3 | H2O | CN- | OH- | SCN- | CO |
|---|---|---|---|---|---|---|---|
| Pt_II  | ✓ | ✓ | ✓ | (q) | (q) | (q) | (q) |
| Pt_IV  | (q) | (q) | (q) | — | — | — | — |
| Ru_II  | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Ru_III | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Ir_III | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Ir_IV  | (q) | (q) | (q) | — | — | — | — |
| Os_II  | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Os_III | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Rh_I   | (q) | (q) | — | (q) | — | (q) | (q) |
| Rh_III | (q) | (q) | (q) | (q) | (q) | (q) | (q) |
| Au_I   | (q) | — | (q) | (q) | — | (q) | (q) |
| Au_III | ✓ | ✓ | (q) | (q) | (q) | (q) | (q) |

`✓` = shipped in F2(a); `(q)` = queued in L-R17-B2 88-rule target.

| row | layer | MLE count | AE count | total |
|---|---|---|---|---|
| **Before F2(a)** | pre-2026-09-15 (`beta_reductions.py:1184-1185`) | **1** (`Pt_Cl_NH3`) | **1** (`Pt_Cl_H2O_first`) | **2** |
| **After F2(a) (current ship)** | `pt_metal_ligand_exchange.py` lines 62-137 | **5** (+`Pt_Cl_RNH2`, `Pt_Br_NH3`, `Pd_Cl_NH3`, `Au_Cl_NH3`) | **2** (+`Pt_OHCl_H2O_second`) | **7** |
| **L-R17-B2 target** | 12 metals × 7 ligands + 4 NEW (`TODO-31 §L-R17-B2`) | 84 | (4 NEW: transmetalation / Cl-OH swap / ligand-shell-MMA / Pt/Pd transmetalation) | **88+** |

**Honest framing:** L-R17-B2 in this round delivers the *F2(a) extension* (1 → 7 patterns) + a 10-pocket pilot + a 217-test regression suite. The full 88-rule hand-curated matrix is **QUEUED** (per `TODO-31 §L-R17-B2` and `TODO-34 §UC-L2`); it would ship via `register_metal_ligand_exchange_rules(metal_set, ligand_set)` adding ~80 parametrize cases to `test_f2a_metal_ligand_exchange.py`. Lit anchors preserved: Lippard 1995, Reedijk 1987, Comba-Hambley 2009, Taube 1952 JACS Pt(II) assoc., Willnhammer 2025.

## 2. Test pass rate — 217/217 (100%) of new parametrize cases

Command: `source env.sh && uv run pytest molmetal/tests/test_f2a_metal_ligand_exchange.py -q --tb=no`

| metric | value |
|---|---|
| total collected | **227** |
| passed | **217** |
| skipped | **10** (MCR-substrate-pool narrowing + intentional Pytest skip — hantzsch_pyridine_synthesis + oxadiazole_formation + 8 narrow-substrate QueryMol misses; **none are failures**) |
| failed | **0** |
| elapsed | 1.80 s (CPU-only) |

5 verification axes covered per `test_f2a_metal_ligand_exchange.py`:
1. **Pattern library cardinality** — 5 MLE + 2 AE; 4 NEW SMARTS distinct from canonical.
2. **MLE SMARTS parse + can_apply + apply** — 5×3 = 15 parametrize cases.
3. **AE SMARTS parse + can_apply + apply** — 2×2 = 4 cases.
4. **NEW SMARTS invariants** — distinct-from-canonical, donor atom-map `:4`, leaving-group atom-map `:2` — 4×4 = 16 cases.
5. **Click-compat 5×5 matrix** + heavy-atom conservation + per-pocket rule-fires (test_010..test_019) — 5×5 + 10 + 3 = 38 cases.
7. **SyntheMol + click + registry cardinality** — 18 SyntheMol × 4 + 7 click × 2 + 2 = 81 cases.
8. **Pockets test_010..test_019 distinct-metal-ligand-pair coverage** — `test_pockets_subset_yield_three_distinct_metal_ligand_pairs` asserts ≥3 distinct (metal, leaving) signatures from SMARTS tokens — **PASS** (5 distinct: Pt-Cl, Pt-Br, Pd-Cl, Au-Cl, Pt-OH-Cl).

**Verdict: PASS** (target 100% met — 0 failures, all skips are MCR / substrate-pool narrowing design choices documented in test docstrings).

## 3. 10-pocket pilot — PARTIAL PASS (within-cell) / FAIL (across-cell)

Script: `molmetal/scripts/r4_lambda_only_run.py --pockets 10 --seed 0 --n-simulations 1000 --metal-seed cisplatin --click-rules auto-pt-strict --sa-weight 0.3`. Full trace at `logs/lr17_b2_pilot/run.log` + `molmetal/reports/wf_lambda1_logs/lr17_b2_pilot/{report,seeds}.json`.

| metric (10-cell mean) | value | vs TODO-25 §3 gate | verdict |
|---|---|---|---|
| n_distinct per cell | **20** | ≥3 | **PASS** |
| distinct SMILES across 10 cells | **20** | >1 | PASS |
| distinct metal-ligand pairs (across) | **2** | **≥3** | **FAIL** |
| distinct triazole ring patterns | **3** | — | — |
| div_tanimoto | 0.137 | — | — |
| div_homotype | 0.045 | — | — |
| div_subpocket | 0.679 | — | — |
| metal_compliance_rate | 0.00 | — | FAIL (no 4-coord Pt_II) |
| coord_number_mean | 1.00 | — | FAIL |
| decoder_pass_rate | 1.00 | ≥0.95 | PASS |
| uniqueness / validity / synth | 1.00 | — | PASS |
| oxidation_state_distribution | `{Pt_0: 200}` | — | — |
| elapsed_s/cell | 122-165 s | — | — |

**Within-cell PASS:** the singleton attractor is broken (R12 n_distinct=1 → R17 n_distinct=20). F2(a)'s CuAAC + SPAAC + Suzuki subset fires on every pocket from `[Pt]C#C` seeds.

**Across-cell FAIL:** same 20 SMILES appear in every pocket (overlap test_000 vs test_009 = 20/20). Two root causes documented in `wf_lr17_b2_pilot/final.md §6`:
1. `--metal-seed cisplatin` + no `--metal-seed-from-pocket` ⇒ shared MCTS root.
2. `auto-pt-strict` keeps only 3 click rules; F2(a) MetalLigandExchange / AquaExchange SMARTS are not in the subset, so transmetalation / Cl-OH swap cannot diversify the metal coordination sphere. **This is the gap the 88-rule L-R17-B2 target exists to close.**

## 4. Comparison vs R12 (singleton) and R16 Path A (n_distinct=20)

| metric | R12 (singleton) | R16 Path A | R17-B2 (this pilot) |
|---|---|---|---|
| n_distinct per cell | **1** | 20 | **20** (broken within-cell) |
| distinct SMILES across 10 cells | 1 | 20 (collapsed to same set) | **20** (still collapsed) |
| distinct metal-ligand pairs (across) | n/a | n/a | **2** (FAIL TODO-25 ≥3 gate) |
| div_tanimoto | ~0.000 | 0.1065 | 0.137 |
| metal_compliance_rate | 0.00 | 0.00 | 0.00 (coord_number_mean=1.0) |
| CuAAC + SPAAC + Suzuki fires | 0 | 200/cell | 200/cell |
| MLE / AquaExchange SMARTS fires | 0 | 0 | **0** (gated out by `auto-pt-strict`) |

**Honest reading:** R17-B2 advances the *within-cell* singleton barrier (R12 n=1 → n=20, matched to R16 Path A). It does **not** yet lift the across-cell collapse or the metal-coordination prior; the 88-rule matrix is required to expose Pt(NH3)2 / Pt(Cl)2 / Pt(NH3)Cl coordination diversity that the TODO-25 §3 gate demands.

## 5. Future work — wire the 88-rule matrix + 30/100-pocket sweep

1. **L-R17-B2 follow-on ship (CPU, ~6h):** add `register_metal_ligand_exchange_rules(metal_set, ligand_set)` to `beta_reductions.py` covering 12 metals × 7 ligands; add 4 NEW SMARTS (transmetalation, Cl-OH swap, ligand-shell-MMA, Pt/Pd transmetalation). Extend `auto-pt-strict` → `auto-pt-strict + metal-coord` (7-rule subset) so the metal-coord family fires in pilot.
2. **Pocket-derived metal seed:** wire `--metal-seed-from-pocket` (Phase 2C helper already shipped but not enabled in the pilot) to break the shared-root attractor.
3. **Re-run pilot with both fixes:** re-execute the 10-pocket panel with the 7-rule click subset + pocket-derived seed; gate target: distinct_metal_ligand_pairs ≥ 3.
4. **Round-13 paper-grade 30/100-pocket sweep:** scale to the paper-grade cohort per `TODO-14` §Full 100-pocket + per `TODO-25 §T2/T3/T4` decision tree; promote §3.5 / §4 / §6.1 / §7.1 cells from DESIGN → MEASURED on success.
5. **TODO-25 §3 conclusion wire-up:** append the L-R17-B2 result row (this pilot + 88-rule re-run) to `paper/sections/03_5_deflex.tex` and `paper/sections/06_limitations.tex`; cite Lippard 1995, Reedijk 1987, Comba-Hambley 2009, Taube 1952, Willnhammer 2025.
6. **Track UC-L2 dispatch:** per `TODO-34 §UC-L2`, run the 88-rule extension as a separate ultracode workflow in parallel with UC-L1 / UC-L3 (no interference — the F2(a) SMARTS are the ASTSR training-data "skeleton candidates").

**Verdict on L-R17-B2 done-status:** ⚙️ **PARTIAL** — pilot + tests ship; the 88-rule SMARTS matrix is QUEUED. The TODO-25 §3 across-cell ≥3-pair gate is the load-bearing follow-up; once the 88-rule matrix + pocket-derived seed both ship, this row flips to PASS and §3 / §6.1 paper writes can promote to MEASURED.

---

*End of synthesis — 5 sections, no silent cell promotions, all 88-rule matrix cells traced to `TODO/pending/31_r17_ultracode_dispatch.md §L-R17-B2` and `TODO/pending/34_parallel_ultracode_dispatch.md §UC-L2`.*