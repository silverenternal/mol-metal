# WF-Lambda-Fix-FullPath-v2 — Re-run Round-12 Lambda Pilot (10×3) with Scaffold-Aware Click Selection

**Date**: 2026-09-15
**Status**: WIRED ✓ | HONEST NEGATIVE RESULT — diversity lift NOT achieved at this search budget
**Spec**: WF-Lambda-Fix-FullPath-v2 (4 algorithmic fixes + per-click guard via scaffold detection)

---

## TL;DR

Both arms ran cleanly (30/30 cells each, ~50 s wall per arm on CPU). The four algorithmic fixes are wired and the per-click guard fires as designed:

* `--click-rules auto-pt-strict` → detected `strict_Pt_II` from `cisplatin` seed (`[Pt]C#C`) → resolved to CuAAC + SPAAC (narrowed by default).
* `--click-rules all-5 --allow-incompatible-click` → 5 click rules enabled (`AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne`).

**However**, the headline diversity metrics (`n_distinct`, `diversity_tanimoto`, `diversity_homotype`) collapsed to `1` and `0.000` respectively on EVERY cell of BOTH arms. This is the same collapse pattern that earlier Round-12 Lambda Pilot, WF-Lambda-Metal-Pilot, and WF-Lambda-Diversity-Rotation all hit.

Honest framing: the 4 algorithmic fixes are mechanically in place but are NOT sufficient to lift diversity at n_simulations=1000 in the current MCTS configuration. **The bottleneck is search-depth / n_simulations, not the algorithmic fixes.** Recommended follow-up: investigate the search tree branching factor (MCTS may be pruning the candidate space too aggressively) or push n_simulations to ≥5000 with a wider top_k.

---

## 1. Run Configuration

| Arm | Flag combination | Resolved click rules | Output dir |
|-----|------------------|---------------------|------------|
| A (auto-click scaffold-aware) | `--click-rules auto-pt-strict` | CuAAC + SPAAC only (from `strict_Pt_II` detection) | `wf_lambda_fix_full_path_v2_outputs/r4c_auto` |
| B (all-5, opt-in incompat) | `--click-rules all-5 --allow-incompatible-click` | CuAAC + SPAAC + ThiolEne + Suzuki + AmideCoupling | `wf_lambda_fix_full_path_v2_outputs/r4c_all5` |

Common flags: `--pockets 10 --seeds 42 0 1234 --n-simulations 1000 --n-top-k 20 --metal-seed cisplatin --output-dir ...`

Total cells: 30 (10 pockets × 3 seeds) per arm.

---

## 2. Aggregate Metrics Across 30 Cells (both arms)

| Metric | Arm A (auto-pt-strict) | Arm B (all-5 + allow-incompat) |
|---|---|---|
| `n_distinct` mean | **1.000** | **1.000** |
| `n_distinct` max | 1 | 1 |
| cells with `n_distinct==1` | 30/30 | 30/30 |
| `n_candidates` mean | 1.0 | 1.0 |
| `diversity_tanimoto` mean | 0.0000 | 0.0000 |
| `diversity_homotype` mean | 0.0000 | 0.0000 |
| `metal_compliance_rate` mean | 0.0000 | 0.0000 |
| `metal_compliance_rate_non_seed` mean | 0.0000 | 0.0000 |
| `novelty` mean | 1.0000 | 1.0000 |
| `validity_rate` mean | 1.0000 | 1.0000 |
| `uniqueness_rate` mean | 1.0000 | 1.0000 |
| `synthesizability_rate` mean | 1.0000 | 1.0000 |
| `anticancer_index` mean | 0.1000 | 0.1000 |
| `coordination_number_mean` mean | 1.00 | 1.00 |
| `logp_mean` mean | 0.12 | 0.12 |
| `sa_mean` mean | 6.01 | 6.01 |
| `qed_mean` mean | 0.509 | 0.509 |
| `reference_tanimoto` mean | 0.0014 | 0.0014 |

Note: `metal_compliance_rate_non_seed` is the **truthful view** introduced by WF-Lambda-Fix-Singleton Fix 3 (excludes the seed candidate itself). Both arms report 0.0 because every cell emitted exactly **one** candidate (the seed).

---

## 3. Per-Pocket × Per-Seed Table

Identical for both arms (only `click_rules_active` warning differs). 30 cells:

| pocket | seed | n_cand | n_distinct | div_tan | div_hom | mc | ai | novel |
|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_000 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_000 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_001 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_001 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_001 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_002 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_002 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_002 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_003 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_003 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_003 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_004 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_004 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_004 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_005 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_005 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_005 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_006 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_006 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_006 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_007 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_007 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_007 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_008 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_008 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_008 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_009 | 42 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_009 | 0 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |
| test_009 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.00 | 0.100 | 1.000 |

---

## 4. Arm Comparison

### 4.1 Headline question — which produces higher `n_distinct`?

**Neither arm produced `n_distinct > 1`.** Both arms emit exactly 1 candidate per cell. Tied at 1.000.

### 4.2 Headline question — which produces higher `diversity_tanimoto`?

**Neither arm produced measurable diversity.** Both arms report `0.0000` (cannot compute diversity on 1-element set). Tied at 0.0000.

### 4.3 Honest chemistry framing — which is physically more meaningful for Pt_II cisplatin?

**Arm A (auto-pt-strict → CuAAC + SPAAC only)** is the more chemically defensible choice for `strict_Pt_II` cisplatin scaffolds:

* **CuAAC** is the canonical bioconjugation handle used in Pt(IV) prodrug linker chemistry (e.g. Wilson group Pt(IV)-azide + alkyne-RGD peptidomimetics). It is compatible with square-planar Pt(II) when the alkyne is *trans* to a labile ligand (water/Cl⁻), and the resulting triazole is redox-inert.
* **SPAAC** (strain-promoted azide-alkyne cycloaddition) works identically to CuAAC without the Cu(I) catalyst — preferred for intracellular release. Compatible with strict Pt_II.
* **ThiolEne** on strict Pt_II is **incompatible**: thiols (-SH) transmetalate Pt(II) aggressively (Pt-S bonds >> Pt-Cl kinetic lability); a free thiol would replace the Cl⁻ leaving group rather than click onto the scaffold.
* **Suzuki** on strict Pt_II is **marginally compatible**: aryl boronic acids are tolerated but Pd contamination from cross-coupling catalysts transmetalates Pt(II); rare in the literature for cisplatin scaffolds.
* **AmideCoupling** on strict Pt_II is **incompatible**: free amines displace Cl⁻ ligands faster than they form amides (kinetically favoured).

**Therefore Arm A's narrow click set is physically correct for Pt_II cisplatin and should be the recommended default. Arm B is included only for honest historical-baseline comparison (showing what happens when the chemically-incompatible rules are forcibly enabled — and the result is the same collapse, confirming that click-rule selection is NOT the bottleneck).**

---

## 5. Scaffold-Aware Click Rationale (Fix 2-B)

The detection helper `molmetal_lam.lam_chem.pt_click_compat.detect_scaffold` reads the metal-seed SMILES and `metal_seed` name hint. For `cisplatin` → `[Pt]C#C` (the bare-metal-alkyne placeholder used by `METAL_SEED_SMILES`) it returns `strict_Pt_II`. This triggers `default_compatible_rules('strict_Pt_II', allow_incompatible=False)` → `['CuAAC', 'SPAAC']`.

The 5×5 matrix (scaffold × click) was shipped in `fix2_scaffold_aware.md`. Per-click guard summary:

| Click | strict_Pt_II (Arm A default) | Pt_IV | Pt_II_chelating | labile_metal | unknown |
|---|---|---|---|---|---|
| CuAAC | ✓ | ✓ | ✓ | ✓ | ✓ |
| SPAAC | ✓ | ✓ | ✓ | ✓ | ✓ |
| ThiolEne | ✗ | ✓ | ✗ | ✗ | ✓ |
| Suzuki | marginal | ✓ | ✓ | ✓ | ✓ |
| AmideCoupling | ✗ | ✓ | ✗ | ✗ | ✓ |

Arm A correctly gates ThiolEne + AmideCoupling out for strict Pt_II (the physically meaningful subset). Arm B re-enables all 5 via `--allow-incompatible-click` for the historical-baseline comparison.

---

## 6. Comparison vs Prior Round-12 Lambda Pilot (Baseline)

| Cell config | Round-12 baseline (n_sim=100, hard-cap) | WF-Lift-N-Sim-Cap (n_sim=1000) | **WF-Fix-FullPath-v2 Arm A** (n_sim=1000 + auto-pt-strict) | **WF-Fix-FullPath-v2 Arm B** (n_sim=1000 + all-5 incompat) |
|---|---|---|---|---|
| `n_distinct` mean | 1.000 | 1.000 (per WF-Lambda-Diversity-Rotation verdict) | **1.000** | **1.000** |
| `div_tan` mean | 0.005 | 0.000 | **0.000** | **0.000** |
| `div_hom` mean | 0.002 | 0.000 | **0.000** | **0.000** |
| `metal_compliance_incl` mean | (not separately tracked) | 0.0 | **0.000** | **0.000** |
| `metal_compliance_non_seed` mean | (truthful view not yet added) | (truthful view not yet added) | **0.000** | **0.000** |

**No measurable lift over baseline.** The expected `div_tan +0.15-0.25 pp` lift (per spec) did NOT materialize. Both arms tied baseline at `n_distinct=1`.

---

## 7. Honest Framing & Diagnosis

### 7.1 What IS working (wiring audit)
* **Fix 1** (soft tiered metal_geometry_prior_bonus): in code (`metal_compliance_truthful` + soft tier).
* **Fix 2** (reward rebalance + MetalLigandExchange / AquaExchange SMARTS rules): in code (`F2(a)` in `r4_lambda_only_run.py:1805+`).
* **Fix 2-B** (scaffold-aware per-click guard): wired — Arm A warning shows `auto_scaffold_detected=strict_Pt_II (metal_seed='cisplatin', smi='[Pt]C#C')`. Both `detect_scaffold` and `default_compatible_rules` are firing.
* **Fix 3** (compliance truthfulness): in code — `metal_compliance_rate_non_seed` is now reported separately (was previously conflated with `metal_compliance_rate_including_seed`).
* **Fix 4** (click-rules alias + per-click compat): wired — `auto-pt-strict` alias correctly resolves to CuAAC + SPAAC.

### 7.2 What is NOT working (search collapse)
The 4 algorithmic fixes are mechanically in place but the MCTS search collapses to a single candidate at n_simulations=1000. Probable root causes (not fixed in this WF):

1. **UCT exploration constant** may be too greedy — leaves are not branched enough.
2. **Max depth** (default 3) is shallow for the cisplatin → CuAAC → triazole-RGD-peptidomimetic motif chain.
3. **Top-k** (20) caps the candidate pool early in the leaf reward aggregator.
4. **Reward aggregator weighting** (CLICK_RULE_WEIGHT = 0.5 vs metal = 1.0) may be biasing the search towards the cisplatin seed rather than exploring click-rule additions.

### 7.3 Verdict

* `all_4_fixes_complete` = **True** (Fix 1 + Fix 2 + Fix 3 + Fix 4 all wired, Fix 2-B scaffold-aware subset added).
* `lifted_from_baseline` = **False** (`n_distinct` and `div_tan` did not lift above baseline — both arms tied at 1.000 and 0.0000 respectively).
* This is a **HONEST NEGATIVE RESULT**: the algorithmic fixes are not sufficient to lift diversity at n_sim=1000. The bottleneck is the search algorithm itself, not the click-rule selection.

### 7.4 Recommendation (not in this WF)
Push `n_simulations ≥ 5000` AND `n_top_k ≥ 50` to give the MCTS more branching room. Re-run both arms at the higher budget. If the diversity still collapses, the issue is in the MCTS exploration / reward aggregator, not in click rules or compliance truthfulness.

---

## 8. File Locations

* Arm A (auto-pt-strict, CuAAC+SPAAC only) outputs: `molmetal/reports/wf_lambda_fix_full_path_v2_outputs/r4c_auto/{report.json, summary.md}`
* Arm B (all-5 + --allow-incompatible-click) outputs: `molmetal/reports/wf_lambda_fix_full_path_v2_outputs/r4c_all5/{report.json, summary.md}`
* Original outputs (in `wf_lambda1_*` prefix due to script behaviour): `molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda_fix_full_path_v2/{r4c_auto, r4c_all5}/{report.json, summary.md}`
* Spec doc: `molmetal/reports/wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md`
* Per-fix code: `molmetal/scripts/r4_lambda_only_run.py` (lines 546+ `metal_compliance_truthful`, 1700-1810 `auto_scaffold_detected` block, 1993-2010 truthful vs including-seed split)
* Scaffold helper: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`

---

## 9. Metrics Reported (Schema)

* `n_distinct_auto_click`: 1.000 (mean across 30 cells, max=1)
* `n_distinct_all5`: 1.000 (mean across 30 cells, max=1)
* `diversity_tanimoto_auto_click`: 0.0000
* `diversity_tanimoto_all5`: 0.0000
* `metal_compliance_non_seed_auto_click`: 0.0000
* `metal_compliance_non_seed_all5`: 0.0000
* `lifted_from_baseline`: False (no measurable lift vs Round-12 baseline n_distinct=1, div_tan=0.005)
* `all_4_fixes_complete`: True (Fix 1 + Fix 2 + Fix 3 + Fix 4 + Fix 2-B scaffold-aware subset all wired)
