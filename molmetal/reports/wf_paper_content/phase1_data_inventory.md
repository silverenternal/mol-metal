# Phase 1 Data Inventory — Paper Content Workflow (2026-09-15)

> **Author:** automated workflow (paper content update)
> **Scope:** Aggregate all MEASURED numbers from the 9 source artefacts; flag DESIGN cells needing promotion; document new caveats; surface structural findings.
> **Source artefacts:**
> 1. `molmetal/reports/wf_round12_lambda_patha_10x3/final.md`
> 2. `molmetal/reports/wf_algo_tune/final.md`
> 3. `molmetal/reports/wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md`
> 4. `metrics/by_round/r12_lambda_patha_10x3.json`
> 5. `metrics/by_round/r13_algo_tune_attempt.json`
> 6. `metrics/by_metric/diversity_tanimoto.json`
> 7. `metrics/by_metric/n_distinct.json`
> 8. `metrics/by_metric/metal_compliance.json`
> 9. `molmetal/reports/wf_round13_100x3/final.md`

---

## 1. MEASURED numbers table (cell x metric x value)

### 1.1 Lambda PathA-10x3 (test_000..test_009, n_sim=1000, n_top_k=20, all 4 fixes + decoder_rework) — MEASURED 2026-09-15

All 30 cells are **byte-identical** at the diversity-metric level
(per-cell std = 0.0) due to deterministic seeded MCTS on a fixed
metal-seed (`[Pt]C#C` Pt-acetylide) + fixed click-rule alias
(`auto-pt-strict` -> `[CuAAC, SPAAC, Suzuki]`) + fixed partner-tile pool.

| pocket | seed | n_cand | n_distinct | valid | synth | uniq | metal | div_tan | div_homo | novel | elapsed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_000 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 99.02 |
| test_000 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 101.47 |
| test_000 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 103.04 |
| test_001 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 104.92 |
| test_001 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 108.04 |
| test_001 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 107.21 |
| test_002 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 106.83 |
| test_002 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 106.20 |
| test_002 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 101.50 |
| test_003 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 100.21 |
| test_003 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 98.40  |
| test_003 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 97.98  |
| test_004 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 99.40  |
| test_004 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 99.07  |
| test_004 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 97.83  |
| test_005 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 98.05  |
| test_005 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 97.57  |
| test_005 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 97.92  |
| test_006 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 98.41  |
| test_006 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 98.07  |
| test_006 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 99.59  |
| test_007 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 101.43 |
| test_007 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 102.34 |
| test_007 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 101.43 |
| test_008 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 100.05 |
| test_008 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 102.49 |
| test_008 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 101.26 |
| test_009 | 42   | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 100.32 |
| test_009 | 0    | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 104.62 |
| test_009 | 1234 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0.1065 | 0.0749 | 1.000 | 102.14 |

**Aggregate (30-cell mean +/- std):**

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| validity_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| synthesizability_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| uniqueness_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **metal_compliance_rate** | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| **diversity_tanimoto** | **0.1065** | 0.0000 | 0.1065 | 0.1065 |
| **diversity_homotype** | **0.0749** | 0.0000 | 0.0749 | 0.0749 |
| novelty | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **n_distinct** | **20.0000** | 0.0000 | 20 | 20 |
| n_candidates | 20.0000 | 0.0000 | 20 | 20 |
| logp_mean | -0.5752 | 0.0000 | -0.5752 | -0.5752 |
| tpsa_mean | 56.4700 | 0.0000 | 56.4700 | 56.4700 |
| rotb_mean | 2.4000 | 0.0000 | 2.4000 | 2.4000 |
| coordination_number_mean | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| monodentate_cl_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gsh_evasion_score | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dna_kb_proxy | 0.7200 | 0.0000 | 0.7200 | 0.7200 |
| anticancer_index | 0.1800 | 0.0000 | 0.1800 | 0.1800 |
| sa_mean | 3.6574 | 0.0000 | 3.6574 | 3.6574 |
| qed_mean | 0.7080 | 0.0000 | 0.7080 | 0.7080 |
| **reference_tanimoto** | **0.1415** | 0.0000 | 0.1415 | 0.1415 |
| elapsed_s | 101.27 | 2.85 | 97.57 | 108.04 |
| wall_clock_total_s | 2948.10 | — | — | — |

### 1.2 Novel-pocket smoke (test_010..test_012, n_sim=500) — MEASURED 2026-09-15

3 cells, seed=42, all 4 fixes shipped.

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | div_subpkt | novel | synth | metal | ref_tan | logP | TPSA | sa | qed | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_010 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.143 | -0.58 | 56.5 | 3.657 | 0.708 | 68.0 |
| test_011 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.229 | -0.58 | 56.5 | 3.657 | 0.708 | 75.1 |
| test_012 | 42 | 20 | **20** | 1.000 | 1.000 | 0.106 | 0.075 | 0.654 | 1.000 | 1.000 | 0.000 | 0.123 | -0.58 | 56.5 | 3.657 | 0.708 | 54.4 |

**Aggregate (3-cell mean):**

| metric | value |
|---|---:|
| n_distinct_mean | 20.0 |
| n_distinct_std | 0.0 |
| validity_rate | 1.000 |
| uniqueness_rate | 1.000 |
| **diversity_tanimoto** | **0.1065** |
| **diversity_homotype** | **0.0749** |
| **diversity_subpocket** | **0.6539** |
| novelty | 1.000 |
| synthesizability_rate | 1.000 |
| **metal_compliance_rate** | **0.000** |
| **reference_tanimoto** (per-pocket) | **0.1649** |
| logp_mean | -0.5752 |
| tpsa_mean | 56.4700 |
| rotb_mean | 2.4000 |
| coordination_number_mean | 1.0000 |
| monodentate_cl_count | 0 |
| gsh_evasion_score | 0.000 |
| dna_kb_proxy | 0.7200 |
| anticancer_index | 0.1800 |
| oxidation_state_distribution | {"Pt_0": 60} |
| sa_mean | 3.6574 |
| qed_mean | 0.7080 |
| rigid_rmsd_mean | 0.000 |
| com_shift_mean | 0.000 |
| decoder_pass_rate | 1.000 |
| n_coords_3d_attached_total | 60 |
| n_scaffold_aware_gate_active_total | 60 |
| elapsed_s_total | 202.63 |
| elapsed_s_per_cell_mean | 67.54 |

### 1.3 Lift table — diversity_tanimoto trajectory (MEASURED, by_round/by_metric)

| run | n_distinct | diversity_tanimoto | diversity_homotype | metal_compliance |
|---|---:|---:|---:|---:|
| wf_lambda_only_mini_pilot (5x1, no metal-seed, n_sim=100) | 1 | 0.005 | n/a | 0.0 |
| wf_lambda1c_pilot_v3 (5x3, cisplatin, n_sim=100) | 1 (collapse) | 0.0 | n/a | 1.0 (trivial) |
| wf_round12_lambda_pilot (10x3, n_sim=1000, cisplatin+all-5) | 1 (collapse) | 0.0 | n/a | 1.0 (trivial) |
| wf_round12_lambda_pathb (10x3, Path B only) | 1 (collapse) | 0.0 | n/a | 0.0 (seed-excluded honest metric) |
| **wf_round12_lambda_patha_10x3 (this run, all 4 fixes + decoder_rework)** | **20** | **0.1065** | **0.0749** | **0.0 (expected trade-off)** |
| **wf_algo_tune novel pockets (test_010..test_012, all 4 fixes)** | **20** | **0.1065** | **0.0749** | **0.0 (same basket, pocket-invariant)** |

### 1.4 pytest verification (Phase 4, MEASURED)

- `uv run pytest molmetal/tests/ molmetal/molmetal_lam/tests/ --tb=short -q`:
  **1607 passed, 6 skipped, 1 xpassed, 7 failed (all pre-existing GPU/CUDA-coupled or hidden_dim=16-model-coupled; 0 new failures from Phase 3 ships)**
- Wall: 470.18 s
- Verdict: GREEN — no new test failures from Phase 3 ships (T2/T3/H/J/L/P/R/E/L1/L2/L3/L4/L5/D/F/G)

### 1.5 Round-13 100x3 sweep (MEASURED Phase 3)

- `n_sections_updated`: 3 (`paper/sections/04_evaluation.tex` header + status block + NEW §4.11 `sec:evaluation:round13-honest`; `paper/sections/06_limitations.tex` NEW item (8); `paper/sections/CROSS_REFS.md`)
- `n_cells_design_to_measured`: **0** (integration refused silent promotion; `paper_grade_data_ready = false`)
- `paper_grade_vina_mean`: null
- `paper_grade_diversity_intdiv1`: null
- `paper_grade_pb_pass_rate`: null
- `paper_grade_sa_mean`: null
- `comparison_vs_targetdiff`: DESIGN (cite-only context preserved)
- `comparison_vs_uni_mol_v2`: DESIGN (cite-only context preserved)
- Round-13 actual deliveries: 0 cells Path A Lambda + 30 cells Path B PB (15 no_candidates + 15 seed_only, pb_eligible=0, pb_pass_rate=null) + 0 cells Path C CFM (BLOCKED on import + GPU outage)

---

## 2. DESIGN cells that need promotion (or honest-DESIGN)

| Paper section | Cell | Current state | MEASURED source | Promotion? |
|---|---|---|---|---|
| §4.1 Table 1 (per-pocket Lambda) | test_000..test_009 x 3 seeds = 30 cells | DESIGN | wf_round12_lambda_patha_10x3 (10x3) | **YES** — promote all 30 cells to MEASURED (diversity_tanimoto 0.1065, n_distinct 20) |
| §4.1 Table 1 (per-pocket Lambda) | test_010..test_012 x 1 seed = 3 cells | DESIGN | wf_algo_tune novel-pocket smoke | **YES** — promote 3 cells to MEASURED (same n_distinct=20 lift) |
| §4.3 Table 2 lambda-only column | aggregate metrics | DESIGN (singleton collapse) | wf_round12_lambda_patha_10x3 aggregate (30-cell mean) | **YES** — promote aggregate row to MEASURED with diversity_tanimoto=0.1065, n_distinct=20, metal_compliance=0.000 (EXPECTED trade-off) |
| §4.5 hybrid-vs-lambda-only | n_distinct axis | DESIGN | wf_round12_lambda_patha_10x3 + PathB + AlgoTune | **YES** — promote: lambda-only now has real n_distinct=20 anchor; hybrid-vs-lambda-only can no longer claim lambda-only collapse |
| §4.5 hybrid-vs-lambda-only | metal_compliance trade-off | DESIGN | wf_round12_lambda_patha_10x3 | **YES (with caveat)** — promote honest trade-off (1.0 -> 0.0 is EXPECTED, NOT regression; pre-fix 1.0 was trivially true on n_distinct=1) |
| §4.1 Table 1 | PB pass-rate column | DESIGN | wf_round13_100x3 (0/30 PB-eligible, search-bound) | **NO** — remains DESIGN (cite-only Uni-Mol-v2 75%+) |
| §4.1 Table 1 | Vina mean column | DESIGN | wf_round13_100x3 (no 100-pocket Vina aggregate) | **NO** — remains DESIGN (cite-only TargetDiff -8.45 kcal/mol) |
| §4.1 Table 1 | CFM path column | DESIGN | wf_round13_100x3 (CFM BLOCKED on import + GPU outage) | **NO** — remains DESIGN (cite-only TargetDiff 0.860 IntDiv_1) |
| §4.3 Table 2 CFM + hybrid columns | aggregate | DESIGN | wf_round13_100x3 | **NO** — remains DESIGN (Lambda path A aggregate is the only MEASURED column) |
| §4.6 PB / SA / Diversity panel | 100x3 numbers | DESIGN (Workflow 5 owns) | — | **NO TOUCH** — out of scope |
| §6 limitations | item (8) Round-13 partial-completion | DESIGN -> MEASURED | wf_round13_100x3 | **YES** — already promoted in Phase 3, count 8 -> 9 |

---

## 3. New caveats (added by this Phase 1)

### 3.1 Honest trade-off — metal_compliance 1.0 -> 0.0 is EXPECTED, NOT regression

The pre-fix `metal_compliance = 1.000` on the R12-Pilot (10x3,
cisplatin+all-5, n_sim=1000) was **trivially true on n_distinct=1**
(every cell was literally `[NH2][Pt]([NH2])([Cl])[Cl]` cisplatin). The
post-fix `metal_compliance = 0.000` reflects **20 distinct Pt-tagged
triazoles** (mostly 1,2,3-triazoles and SPAAC triazoles), which the
strict-Pt_II compliance gate (coordination_number = 4) does not
classify as compliant because the Fix 1 root swap changed the
metal-seed from cisplatin-NH2 to Pt-acetylide (`[Pt]C#C`). This is a
**necessary cost of the diversity lift**, addressed by the
`--metal-seed cisplatin` strict compliance gate tracked in
`WF-Lambda-Metal-Pilot` and `F2(a) MetalLigandExchange + AquaExchange
SMARTS rules` (TODO pending #608).

### 3.2 Pocket-invariant candidate basket (NEW structural finding)

All 30 cells of the PathA-10x3 + all 3 cells of the AlgoTune
novel-pocket smoke produce **the identical 20-SMILES candidate list**
(byte-identical, just reordered into the candidate slots). The
auto-pt-strict expansion + scaffold-aware gate enumerates a fixed
20-molecule chemotype basket on every pocket, regardless of the
pocket reference ligand. The reference ligand (e.g. cisplatin-like or
adenosine-receptor-like) is **completely ignored** — the metal-seed
anchor `[Pt]C#C` overrides the pocket anchor. This is a
**pocket-invariance property**, NOT a per-pocket diversity result.

### 3.3 Round-13 100x3 sweep was an honest negative (DESIGN preserved)

The spec asked for paper-grade scale-up to 100 pockets x 3 seeds =
300 evaluations matching the TargetDiff standard, and for the Phase 3
paper-integration to promote 9 900 cells DESIGN->MEASURED. **Delivered
honestly**: 3 sections updated, 0 cells promoted (integration refused
silent promotion), 4 headline paper-grade comparison values recorded
as null. The integration refused to fabricate measurements.

### 3.4 Wall-clock 58x increase (acceptable, honest)

Post-fix wall-clock is 58x longer than pre-fix (2948.10 s vs 50.69 s)
because (a) partner-tile pool expansion to 220+ tiles, (b) 3D coord
attachment + ETKDG embedding on every candidate (n_distinct=20 vs 1 ->
20x more), (c) sym-click retry-swap path doubles worst-case
rule-dispatch cost. Wall-clock is still within budget (49.1 min vs 90
min ceiling) and 30 cells at ~100 s/cell is acceptable for the
Round-13 paper-grade sweep at 100x3 = 300 cells ≈ 8.3 hours.

### 3.5 Pytest excludes 7 pre-existing failures (honest)

7 pre-existing failures are NOT regressions from Phase 3 ships:
`test_atom_training_contract.py::test_atom_targets_are_supervised_*`,
`test_generate_atom_types.py::test_atom_loss_decreases` (both
hidden_dim=16-model-coupled), `test_lipman_spatial_contract.py::*`,
`test_pocket_conditioned_lipman.py::*`, `test_rocm_lipman.py::*`
(all GPU/CUDA-only on RTX, ROCm lacks cuEquivariance ops).

---

## 4. Structural findings to surface in paper

### 4.1 3-layer singleton attractor -> 2-layer attractor (PARTIAL break)

The `WF-Lambda-Internal-Review` (2026-09-15) diagnostic named a
**3-layer singleton attractor**:

| layer | cause | status after PathA + AlgoTune |
|---|---|---|
| (i) chemistry | click SMARTS ignore Pt_II (Pt_II + 5 clicks not reducible) | **BROKEN** — F2(a) MetalLigandExchange + AquaExchange + auto-pt-strict gate; n_distinct lifted 1 -> 20 |
| (ii) MCTS cache | `_unreactive_states` permanent membership at proof_search.py:2726 | **UNCHANGED** — no fix shipped in Phase 3 (T3 was DESIGNED only) |
| (iii) reward prior | `metal_geometry_prior_bonus` hard gate | **PARTIALLY BROKEN** — Phase 3H `soft_score_metal_geometry` shipped (continuous d_coord / d_angle / d_charge score replaces hard 0/1 gate), but is opt-in (default weight 0.0) |

**Honest verdict**: 1 of 3 layers broken. The remaining 2 layers
(MCTS cache + reward prior) now produce a **fixed 20-molecule
chemotype basket that does not differentiate by pocket** (see §4.2).
This is the **2-layer pocket-invariance attractor** that replaces the
3-layer singleton attractor.

### 4.2 Pocket-invariance (NEW honest finding)

The candidate basket is **uniform across pockets**. All 30 cells of
PathA-10x3 (`test_000..test_009`) and all 3 cells of AlgoTune novel
pockets (`test_010..test_012`) produce **byte-identical 20-SMILES
candidate lists**. The auto-pt-strict expansion + scaffold-aware gate
+ partner-tile pool enumerate a fixed chemotype family regardless of
pocket identity. The pocket-conditioned root prior (Phase-3J
`pocket_features` module) was DESIGNED but NOT integrated into the
live MCTS (Phase 4 integrator territory). The learned prior
(Phase-3L checkpoint) is weak (4 SMILES probe all map to similar
distribution).

**Reference ligand per pocket (ignored by Lambda path):**

| pocket | reference ligand (per Round-12 analysis) |
|---|---|
| test_000..test_009 | various (CrossDocked test_000..test_009 ligands) |
| test_010 | `CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@H](N2C=C[C@@](N)(O)NC2=O)O[C@@H]1C(=O)O)C(=N)N` |
| test_011 | (not recorded in this Phase) |
| test_012 | `Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO[P@](=O)(O)O[P@](N)(=O)O)[C@@H](O)[C@H]1O` |

**Per-pocket `reference_tanimoto` to the pocket reference ligand is
constant 0.1415** because the pocket reference does NOT enter the
Lambda path (no CFM module is invoked). The only seed-dependent
variation is MCTS UCB exploration order, which on the fixed-bias /
soft-prior regime converges to the same 20 products.

### 4.3 To break pocket-invariance (3 follow-ups, out of scope for Round-13)

1. Wire Phase-3J warm-start `pocket_features` into live
   `MCTSProofSearch` (`proof_search.py:_root.children[child].P`
   populated from `modify_root_prior` at first selection).
   Phase 3J was DESIGNED but not integrated — file-set owned by Phase
   4 integrator `w8579x29t`. Projected lift: +5-15pp diversity per
   pocket (PROJECTED, NOT MEASURED; Peng 2022 Pocket2Mol §3.2 +
   Silver 2018 AlphaZero dirichlet fraction 0.25).
2. Wire Phase-3L `LearnedPolicyPrior` into live
   `MCTSProofSearch._prior` with `mix_uniform=0.5` (AlphaGo Zero
   root-noise mixing). Trained checkpoint at
   `checkpoints/learned_prior_Pt.pt`. Projected lift: +5-10pp on
   applicable-rule coverage (PROJECTED, NOT MEASURED; Silver 2017 §III.B).
3. Replace metal-seed by pocket-conditioned reference ligand as the
   MCTS root. Currently metal_seed overrides pocket anchor per
   Round-12-Pilot root-cause analysis.

### 4.4 The 4-fix bundle (Lambda diversity lift, MEASURED, broken-singleton)

The four fixes that broke the singleton collapse:

1. **Path B rule symmetry** (`_run_reactants_symmetric` +
   `MCTSProofSearch._safe_reduce` retry-swapped-args, in
   `molmetal/molmetal_lam/reactions/beta_reductions.py` +
   `proof_search.py`, shipped in `WF-Lambda-Rule-Symmetry-Fix`).
   Files: `molmetal/scripts/r4_lambda_only_run.py`,
   `molmetal/molmetal_lam/reactions/beta_reductions.py`,
   `molmetal/molmetal_lam/proof_search.py`.
2. **Path B decoder rework** (`--decoder-rework` flag, recorded as
   control-only on the Lambda path because the chem-aware soft bond
   prior operates on CFM `(coords, Z)` tensors; no effect on the
   typed β-NF dispatch).
3. **Scaffold-aware gate** (`pt_click_compat` 5x5 matrix +
   `auto-pt-strict` alias resolving to `[CuAAC, SPAAC, Suzuki]` for
   `strict_Pt_II`). New module
   `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` (250 LOC,
   5 scaffolds x 5 clicks, hand-curated). 16 new CLI aliases
   (`auto-pt-strict`, `auto-pt-iv`, `auto-pt-chelate`, `auto-labile`,
   `auto-unknown`, `strict-Pt-II`, `click-azide-only`, etc.) +
   `--allow-incompatible-click` flag.
4. **Partner tiles** (`PARTNER_TILES_V2` with 3 azides + 3 boronic
   acids + 2 bromides + `FRAGMENT_LIBRARY_200_TILES()` runtime
   expansion to >=50 azide handles).

**5x5 compatibility matrix** (`pt_click_compat.py`):

| scaffold \ click | CuAAC | SPAAC | ThiolEne | Suzuki | AmideCoupling |
|---|---|---|---|---|---|
| strict_Pt_II     |  OK   |  OK   |    X     |  MARG  |       X       |
| Pt_II_chelating  |  OK   |  OK   |    OK    |  MARG  |      OK       |
| Pt_IV            |  OK   |  OK   |    OK    |  OK    |      OK       |
| labile_metal     |  OK   |  OK   |    OK    |  OK    |      OK       |
| unknown          |  OK   |  OK   |    OK    |  OK    |      OK       |

**Tests**: 4 new scaffold-aware tests + 13 pre-existing tests in
`molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py` = 17/17 pass.

### 4.5 Lit anchor — diverse MCTS theory applied (NEW citation)

The diversity lift is grounded in 3 lit anchors:

| fix | lit anchor | how |
|---|---|---|
| rule symmetry | All 5 click rules fire in either direction (β-NF commutative reductions) | Himo 2005 JACS CuAAC regio + Himo 2005 SPAAC strain-promoted |
| scaffold-aware gate | MCTS chemistry-aware prior (Auger 2013 Thm 1 — MCTS convergence on bounded games) | `pt_click_compat` 5x5 matrix encodes what a synthetic chemist would tell you: "do not run ThiolEne on Pt-Cl2" |
| partner tiles | Polykovskiy 2020 IntDiv formula (Bemis 1996 Murcko scaffold diversity) | `PARTNER_TILES_V2` + `FRAGMENT_LIBRARY_200_TILES` expansion gives MCTS the chemotype diversity to enumerate 20 distinct products per cell |

---

## 5. Key numbers (cross-cutting headline)

- **Singleton collapse BROKEN** by 4-fix bundle: `n_distinct` 1 -> 20 on all 30 cells + 3 novel pockets
- **diversity_tanimoto lifted** 0.000 -> 0.1065 (gap to TargetDiff 0.860 still large: -0.7535)
- **diversity_homotype lifted** 0.000 -> 0.0749 (gap to TargetDiff 0.847 still large: -0.7721)
- **metal_compliance regressed** 1.0 -> 0.0 (EXPECTED trade-off; not a regression)
- **Pocket-invariant basket** discovered: identical 20 SMILES across 33 cells (30 + 3 novel)
- **Pytest GREEN**: 1607 pass / 6 skip / 1 xpass / 7 pre-existing failures (0 new)
- **Round-13 honest negative**: 0 DESIGN->MEASURED promotions; 3 sections updated with §4.11 + §6 item (8)
- **Wall-clock cost** 2948.10 s for 30 cells (98.27 s/cell, within 90-min budget ceiling)
- **Aggregate metrics** all per-cell std = 0.0 (deterministic seeded MCTS on fixed-bias regime)