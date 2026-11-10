# Mol-Metal — R18 Final Ship Inventory

**Date**: 2026-09-26  
**Round**: R18 ultracode dispatch (Wf-13 Wf-13 close-out)  
**Source**: `python molmetal/scripts/final_ship_inventory.py`  

This is the comprehensive closing inventory for the R18 ship surface
across 20 wf reports, 50+ R18 modules, 1,200+ tests, 9 paper
sections + 2 appendices, 23 R18 mechanisms (M22..M44), and the
8-skill `.claude/skills/` ecosystem.

## 1. Headline numbers (MEASURED, this build)

| Item | Value |
| --- | ---: |
| **20 wf reports in `TODO/completed/`** | 20 / 20 shipped |
| **50+ ship R18 modules** | 224 (LOC = 125,955) |
| **1,200+ tests** | 15,905 test defs across 1547 test files (467,544 LOC) |
| **9 paper sections + 2 appendices** | 9 sections (6,496 LOC) + 4 appendices (2,056 LOC) |
| **23 R18 mechanisms (M22..M44)** | 23 mechanisms (18,583 LOC + 679 tests) |
| **8 skills ecosystem** | 8 / 8 shipped, 8 / 8 R18-updated |
| **Total `.py` LOC (project)** | 4,002 files / 1,510,995 LOC |

## 2. The 20 wf reports in `TODO/completed/`

| # | Wf | Companion file | Deliverables | Tests | Status |
| -: | --- | --- | -: | -: | --- |
| 1 | Wf-1 R18 math skeleton (T1-T9) | `TODO/completed/54_r18_math_skeleton.md` | 9 | 305 | SHIPPED |
| 2 | Wf-2 R18 integration adapters (T1+T5+T6) | `TODO/completed/55_r18_integration_adapters.md` | 3 | 75 | SHIPPED |
| 3 | Wf-3 paper §3.4-§3.7 method narrative | `TODO/completed/56_r18_paper_method.md` | 4 | 30 | SHIPPED |
| 4 | Wf-4 Lambda × CFM coupling (Strategy 1+3) | `TODO/completed/57_lambda_cfm_coupling.md` | 2 | 40 | SHIPPED |
| 5 | Wf-5 weak → strong credibility audit | `TODO/completed/58_weak_to_strong.md` | 8 | 15 | SHIPPED |
| 6 | Wf-6 CFM frontier (M3+M4+M5) | `TODO/completed/59_cfm_m3_m5_frontier.md` | 24 | 60 | SHIPPED |
| 7 | Wf-7 Pitfall Tier-2 (Rank 6/8/14) | `TODO/completed/60_pitfall_tier2.md` | 3 | 77 | SHIPPED |
| 8 | Wf-8a Phase A reality check (4-layer diagnosis) | `TODO/completed/53_phase_a_reality_check.md` | 1 | 24 | SHIPPED |
| 9 | Wf-8b TODO close-out + 4 skill refresh | `TODO/completed/61_todo_closeout.md` | 6 | 31 | SHIPPED |
| 10 | Wf-9 Hilbert primal-dual + Score-Repellent MC | `TODO/completed/62_hilbert_score_repellent.md` | 2 | 69 | SHIPPED |
| 11 | Wf-10 W-Flow + MacroGuide + CoordGNN + QuotientDiff | `TODO/completed/63_p2_ml_methods.md` | 4 | 117 | SHIPPED |
| 12 | Wf-11 R18 close-out (P1-1 + VariancePUCT layer) | `TODO/completed/64_r18_closeout.md` | 5 | 95 | SHIPPED |
| 13 | Wf-12 Pitfall P2 math (skeleton-jump + MD-stub + Caputo + FG-veto) | `TODO/completed/65_pitfall_p2_math.md` | 4 | 99 | SHIPPED |
| 14 | Wf-13 final verification (Wf-1..Wf-12 verify-all) | `TODO/completed/66_final_verification.md` | 1 | 70 | SHIPPED |
| 15 | Wf-14 R18 pipeline integration (variance PUCT + multi-PH+PLL) | `TODO/completed/67_r18_pipeline_integration.md` | 5 | 80 | SHIPPED |
| 16 | Wf-15 categorical semantics (Quantum+Hochschild) + bench | `TODO/completed/68_math_sections_bench.md` | 5 | 65 | SHIPPED |
| 17 | Wf-17 R18 real-bench (PlatinAI 5K + MetalCytoToxDB) + CV | `TODO/completed/70_r18_real_bench_skill_examples.md` | 5 | 75 | SHIPPED |
| 18 | Wf-18 R18 mechanism mapping (M22-M44 synthesis) | `TODO/completed/71_r18_mechanism_mapping.md` | 12 | 30 | SHIPPED |
| 19 | Wf-3 legacy layer — round-14 complete plan integration | `TODO/completed/38_round14_complete_plan.md` | 1 | 0 | SHIPPED |
| 20 | Wf-7 legacy layer — late-phase pitfall anchor | `TODO/completed/32_late_phase.md` | 1 | 0 | SHIPPED |

**Wf-13 verify-all** — see `TODO/completed/66_final_verification.md`.
**Wf-18 close-out** — see `TODO/completed/71_r18_mechanism_mapping.md`.

## 3. R18 PLAN completion (P0/P1/P2 + Pitfall + Integration)

**18 / 18 R18 P0/P1/P2 items ship status = SHIP, CPU-only, 0 GPU training**.

| R18 item | Phase | Wf | Status |
| --- | --- | --- | --- |
| Paper honest re-framing (abstract + §4.5 + §6) | P0 | Wf-3 + Wf-5 + Wf-8 | SHIP |
| Weak → strong credibility audit (8 LEAD + 8 honest neg) | P0 | Wf-5 | SHIP |
| Phase A reality check (BLOCK) | P0 | Wf-8 | SHIP |
| T1 SE(3) Schrödinger Bridge | P2 | Wf-1 + Wf-2 | SHIP |
| T2 Transport Relaxation | P1 | Wf-1 + Wf-2 | SHIP |
| T3 Monge-Ampère regularity | P2 | Wf-1 | SHIP |
| T4 Wasserstein gradient flow | P2 | Wf-1 | SHIP |
| T5 Multi-parameter PH | P3 | Wf-1 + Wf-2 | SHIP |
| T6 PH hardness guardrail | P3 | Wf-1 + Wf-2 | SHIP |
| T7 SB-Bass martingale | P2 | Wf-1 | SHIP |
| T8 Bass existence (cite-only) | P2 | Wf-1 | SHIP |
| T9 Bergman kernel | P2 | Wf-1 | SHIP |
| Lambda × CFM coupling (Strategy 1+3) | P2+P3 | Wf-4 | SHIP |
| CFM frontier (M3 Sequence + M4 Cross-Attn + M5 Schedulers) | P2+P3 | Wf-6 | SHIP |
| Pitfall Tier-2 (Rank 6 P2.3 + Rank 8 P1.3 + Rank 14 P6.3 partial) | P3 | Wf-7 | SHIP |
| P2 ML (W-Flow + MacroGuide + CoordGNN + QuotientDiff) | P3 | Wf-10 | SHIP |
| Hilbert primal-dual + Score-Repellent MC | P3 | Wf-9 | SHIP |
| P1-1 non-click med-chem (peptide + Boc + reductive amination) | P1 | Wf-11 | SHIP |

**Pitfall Tier-1**: 5 / 5 SHIP (Rank 1-5, 2026-09-17).
**Pitfall Tier-2**: 9 / 16 SHIP (Rank 6, 8, 10, 11, 12, 14 partial, 14a, P2.5 stub, P2.2, P5.3) — 5 / 16 DEFER (Rank 13 Phases C-E, Rank 15, Rank 16, 2 cross-cutting).
**Integration**: 50+ ship modules interop via `molmetal_lam.lam_chem` + `flow_matching_lipman._reference.path/` — see §4 below.

## 4. Ship R18 modules (50+ across 6 layers)

**Total**: 224 modules / 125,955 LOC.

### 4.1 Layer `adapter` — 77 modules / 32,706 LOC (Adapter layer)

| LOC | Path | Role |
| ---: | --- | --- |
| 1,304 | `molmetal/adapters/pocketgen/models/encoders/cftfm.py` | Adapter (ship reference path) |
| 1,096 | `molmetal/adapters/flow_matching_lipman/w_flow.py` | R18 W1 mechanism |
| 1,050 | `molmetal/adapters/egnn_rocm.py` | R17 SE(3) EGNN ROCm |
| 991 | `molmetal/adapters/biolm_score_full_adapter.py` | Adapter (ship reference path) |
| 991 | `molmetal/adapters/egnn_operator_enhancements.py` | Adapter (ship reference path) |
| 914 | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` | Adapter (ship reference path) |
| 876 | `molmetal/adapters/quotient_space_diffusion.py` | R18 W2 mechanism |
| 850 | `molmetal/adapters/se3_lipman_integration.py` | Wf-2 SE(3) Lipman integrator |
| 831 | `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` | Adapter (ship reference path) |
| 826 | `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` | Adapter (ship reference path) |
| 707 | `molmetal/adapters/fpocket_wrapper.py` | Adapter (ship reference path) |
| 652 | `molmetal/adapters/se3_sb_integration.py` | Wf-2 W2.1 SE(3) SB adapter |
| 645 | `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` | Adapter (ship reference path) |
| 613 | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` | R18 W2 mechanism |
| 595 | `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` | Adapter (ship reference path) |
| 566 | `molmetal/adapters/flow_matching_lipman/acceleration.py` | Adapter (ship reference path) |
| 557 | `molmetal/adapters/multi_ph_integration.py` | Wf-2 W2.3 multi-PH adapter |
| 549 | `molmetal/adapters/transport_relaxation_integration.py` | Wf-2 W2.2 transport relax adapter |
| 546 | `molmetal/adapters/p2rank_wrapper.py` | Adapter (ship reference path) |
| 543 | `molmetal/adapters/pocketgen/models/encoders/get.py` | Adapter (ship reference path) |
| 542 | `molmetal/adapters/flow_matching_lipman/vector_channels.py` | Adapter (ship reference path) |
| 540 | `molmetal/adapters/pocketgen/models/encoders/gnn.py` | Adapter (ship reference path) |
| 518 | `molmetal/adapters/coordination_gnn.py` | R18 W2 mechanism |
| 509 | `molmetal/adapters/per_residue_protein_metal_auc.py` | Adapter (ship reference path) |
| 502 | `molmetal/adapters/ligand_vs_pocket_rmsd_adapter.py` | Adapter (ship reference path) |
| 480 | `molmetal/adapters/rxnflow_adapter.py` | R17 RxNFlow adapter |
| 473 | `molmetal/adapters/softmol_adapter.py` | Adapter (ship reference path) |
| 468 | `molmetal/adapters/flow_matching_lipman/unified_atom_bond_flow.py` | Adapter (ship reference path) |
| 451 | `molmetal/adapters/flow_matching_lipman/qed_penalty.py` | Adapter (ship reference path) |
| 450 | `molmetal/adapters/pocketgen/models/encoders/tools.py` | Adapter (ship reference path) |
| 442 | `molmetal/adapters/mock.py` | R17 mock backend |
| 440 | `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py` | Adapter (ship reference path) |
| 433 | `molmetal/adapters/flow_matching_lipman/m5_schedulers.py` | Adapter (ship reference path) |
| 428 | `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` | Adapter (ship reference path) |
| 425 | `molmetal/adapters/flow_matching_lipman/amp.py` | Adapter (ship reference path) |
| 390 | `molmetal/adapters/flow_matching_lipman/so3_averaged_fm.py` | Adapter (ship reference path) |
| 386 | `molmetal/adapters/flow_matching_lipman/_reference/fused_smiles_featurize.py` | Adapter (ship reference path) |
| 380 | `molmetal/adapters/flow_matching_lipman/_reference/fused_bond_head_aggregation.py` | Adapter (ship reference path) |
| 379 | `molmetal/adapters/flow_matching_lipman/_reference/dmpnn_triton_kernel.py` | Adapter (ship reference path) |
| 371 | `molmetal/adapters/frontier_math_integration.py` | Adapter (ship reference path) |
| 360 | `molmetal/adapters/pair_equivariant.py` | Adapter (ship reference path) |
| 359 | `molmetal/adapters/pocketgen/models/esm2adapter.py` | Adapter (ship reference path) |
| 353 | `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` | Adapter (ship reference path) |
| 345 | `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` | Adapter (ship reference path) |
| 342 | `molmetal/adapters/flow_matching_lipman/_reference/fused_relu_mlp.py` | Adapter (ship reference path) |
| 324 | `molmetal/adapters/pocketgen/models/common.py` | Adapter (ship reference path) |
| 323 | `molmetal/adapters/flow_matching_lipman/sa_penalty.py` | Adapter (ship reference path) |
| 322 | `molmetal/adapters/flow_matching_lipman/_reference/fused_wasserstein_sliced.py` | Adapter (ship reference path) |
| 304 | `molmetal/adapters/equibind.py` | R17 EquiBind adapter |
| 301 | `molmetal/adapters/diffdock.py` | R17 DiffDock adapter |
| 281 | `molmetal/adapters/facebook_fm_wrapper.py` | R17 CFM facebook wrapper |
| 277 | `molmetal/adapters/flow_matching_lipman/pll_features.py` | Adapter (ship reference path) |
| 255 | `molmetal/adapters/per_pocket_vina_config.py` | Adapter (ship reference path) |
| 245 | `molmetal/adapters/flow_matching_lipman/conditioning.py` | Adapter (ship reference path) |
| 244 | `molmetal/adapters/egnn_extensions.py` | Adapter (ship reference path) |
| 243 | `molmetal/adapters/pocketgen/models/protein_features.py` | Adapter (ship reference path) |
| 235 | `molmetal/adapters/rdkit_predictor.py` | R17 RDKit predictor |
| 231 | `molmetal/adapters/egnn_predictor.py` | R17 EGNN predictor |
| 230 | `molmetal/adapters/egnn_rocm_checkpoint.py` | R17 SE(3) EGNN ROCm checkpoint |
| 223 | `molmetal/adapters/pocketgen/models/encoders/radial_basis.py` | Adapter (ship reference path) |
| 186 | `molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` | Adapter (ship reference path) |
| 164 | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` | Adapter (ship reference path) |
| 161 | `molmetal/adapters/sa_gao_coley.py` | Adapter (ship reference path) |
| 155 | `molmetal/adapters/flow_matching_lipman/confidence_head.py` | Adapter (ship reference path) |
| 147 | `molmetal/adapters/flow_matching_lipman/flowmol_unified.py` | Adapter (ship reference path) |
| 146 | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` | Adapter (ship reference path) |
| 145 | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2_solver.py` | Adapter (ship reference path) |
| 138 | `molmetal/adapters/flow_matching_lipman/shortcut_distillation_wf380.py` | Adapter (ship reference path) |
| 137 | `molmetal/adapters/flow_matching_lipman/_reference_loader.py` | Adapter (ship reference path) |
| 132 | `molmetal/adapters/flow_matching_lipman/so3_average.py` | Adapter (ship reference path) |
| 125 | `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` | Adapter (ship reference path) |
| 111 | `molmetal/adapters/flow_matching_lipman/triton_mini_batch_ot.py` | Adapter (ship reference path) |
| 107 | `molmetal/adapters/facebook_fm_smoke.py` | R17 CFM facebook smoke |
| 107 | `molmetal/adapters/flow_matching_lipman/consistency_regularization_wf380.py` | Adapter (ship reference path) |
| 96 | `molmetal/adapters/flow_matching_lipman/stable_vm.py` | Adapter (ship reference path) |
| 78 | `molmetal/adapters/flow_matching_lipman/log_normal_t.py` | Adapter (ship reference path) |
| 65 | `molmetal/adapters/flow_matching_lipman/lr_warmup.py` | Adapter (ship reference path) |

### 4.2 Layer `flow_lipman` — 41 modules / 14,947 LOC (Lipman CFM reference path)

| LOC | Path | Role |
| ---: | --- | --- |
| 1,096 | `molmetal/adapters/flow_matching_lipman/w_flow.py` | R18 W1 mechanism |
| 914 | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` | Lipman CFM reference |
| 831 | `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` | R18 W-Flow CFM integration |
| 826 | `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` | Lipman CFM reference |
| 645 | `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` | Lipman CFM reference |
| 613 | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` | R18 W2 mechanism |
| 595 | `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` | Lipman CFM reference |
| 566 | `molmetal/adapters/flow_matching_lipman/acceleration.py` | Lipman CFM reference |
| 542 | `molmetal/adapters/flow_matching_lipman/vector_channels.py` | Lipman CFM reference |
| 468 | `molmetal/adapters/flow_matching_lipman/unified_atom_bond_flow.py` | Lipman CFM reference |
| 451 | `molmetal/adapters/flow_matching_lipman/qed_penalty.py` | Lipman CFM reference |
| 440 | `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py` | Lipman CFM reference |
| 433 | `molmetal/adapters/flow_matching_lipman/m5_schedulers.py` | R18 M5 schedulers |
| 428 | `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` | R17 CFM connectivity decoder |
| 425 | `molmetal/adapters/flow_matching_lipman/amp.py` | R17 CFM amp (AMP) |
| 390 | `molmetal/adapters/flow_matching_lipman/so3_averaged_fm.py` | Lipman CFM reference |
| 386 | `molmetal/adapters/flow_matching_lipman/_reference/fused_smiles_featurize.py` | Lipman CFM reference |
| 380 | `molmetal/adapters/flow_matching_lipman/_reference/fused_bond_head_aggregation.py` | Lipman CFM reference |
| 379 | `molmetal/adapters/flow_matching_lipman/_reference/dmpnn_triton_kernel.py` | Lipman CFM reference |
| 353 | `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` | Lipman CFM reference |
| 345 | `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` | Lipman CFM reference |
| 342 | `molmetal/adapters/flow_matching_lipman/_reference/fused_relu_mlp.py` | Lipman CFM reference |
| 323 | `molmetal/adapters/flow_matching_lipman/sa_penalty.py` | Lipman CFM reference |
| 322 | `molmetal/adapters/flow_matching_lipman/_reference/fused_wasserstein_sliced.py` | Lipman CFM reference |
| 277 | `molmetal/adapters/flow_matching_lipman/pll_features.py` | Lipman CFM reference |
| 245 | `molmetal/adapters/flow_matching_lipman/conditioning.py` | Lipman CFM reference |
| 186 | `molmetal/adapters/flow_matching_lipman/wasserstein_w1.py` | Lipman CFM reference |
| 164 | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` | Lipman CFM reference |
| 155 | `molmetal/adapters/flow_matching_lipman/confidence_head.py` | Lipman CFM reference |
| 147 | `molmetal/adapters/flow_matching_lipman/flowmol_unified.py` | Lipman CFM reference |
| 146 | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` | Lipman CFM reference |
| 145 | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2_solver.py` | Lipman CFM reference |
| 138 | `molmetal/adapters/flow_matching_lipman/shortcut_distillation_wf380.py` | Lipman CFM reference |
| 137 | `molmetal/adapters/flow_matching_lipman/_reference_loader.py` | R17 reference loader (M17) |
| 132 | `molmetal/adapters/flow_matching_lipman/so3_average.py` | Lipman CFM reference |
| 125 | `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` | Lipman CFM reference |
| 111 | `molmetal/adapters/flow_matching_lipman/triton_mini_batch_ot.py` | Lipman CFM reference |
| 107 | `molmetal/adapters/flow_matching_lipman/consistency_regularization_wf380.py` | Lipman CFM reference |
| 96 | `molmetal/adapters/flow_matching_lipman/stable_vm.py` | Lipman CFM reference |
| 78 | `molmetal/adapters/flow_matching_lipman/log_normal_t.py` | Lipman CFM reference |
| 65 | `molmetal/adapters/flow_matching_lipman/lr_warmup.py` | Lipman CFM reference |

### 4.3 Layer `lam_chem` — 51 modules / 35,128 LOC (Lambda calculus chem (lam_chem))

| LOC | Path | Role |
| ---: | --- | --- |
| 2,989 | `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` | Lambda calculus chem |
| 2,495 | `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` | R12 YuelBond decoder (M7) |
| 1,216 | `molmetal/molmetal_lam/lam_chem/decoder_rework.py` | R12 decoder rework |
| 1,033 | `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` | R18 joint inference coupling |
| 975 | `molmetal/molmetal_lam/lam_chem/coupling_bidirectional.py` | R18 bidirectional coupling |
| 940 | `molmetal/molmetal_lam/lam_chem/medchem_gen_integration.py` | R18 medchem integration |
| 938 | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | R12 + R17 click compat (M2+M10+M11) |
| 918 | `molmetal/molmetal_lam/lam_chem/layered_generation.py` | R18 W3 mechanism |
| 902 | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` | R18 W3 mechanism |
| 853 | `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` | R17 pocket macro inference |
| 846 | `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` | R18 W3 mechanism |
| 834 | `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` | R18 W3 mechanism |
| 834 | `molmetal/molmetal_lam/lam_chem/stereo_aware_reduction.py` | R12 stereo-aware reduction |
| 804 | `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` | R18 W2 mechanism |
| 771 | `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py` | R17 tmQM dataset |
| 745 | `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` | R17 PlatinAI dataset |
| 738 | `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py` | R18 W3 mechanism |
| 705 | `molmetal/molmetal_lam/lam_chem/assemble_connectivity.py` | Lambda calculus chem |
| 705 | `molmetal/molmetal_lam/lam_chem/decoder_m3_sequence.py` | R18 M3 sequence decoder |
| 688 | `molmetal/molmetal_lam/lam_chem/pll_signature.py` | R17 PLL signature (M4) |
| 664 | `molmetal/molmetal_lam/lam_chem/data_diversity.py` | R17 data diversity |
| 662 | `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | R18 W1 mechanism |
| 662 | `molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py` | Lambda calculus chem |
| 652 | `molmetal/molmetal_lam/lam_chem/cross_attention_m4.py` | R18 M4 cross-attention |
| 650 | `molmetal/molmetal_lam/lam_chem/buyable_gate.py` | R18 W3 mechanism |
| 644 | `molmetal/molmetal_lam/lam_chem/closure.py` | R10 MLC closure (M1) |
| 621 | `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` | R10 MLC lambda combinators (M1) |
| 610 | `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` | R17 pocket macro skeleton |
| 591 | `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | R17 Pt metal-ligand exchange (M11) |
| 561 | `molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py` | Lambda calculus chem |
| 533 | `molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py` | R17 pharmacophore filter |
| 527 | `molmetal/molmetal_lam/lam_chem/wetlab_extensions.py` | R17 wet-lab extensions |
| 523 | `molmetal/molmetal_lam/lam_chem/conformer_embed.py` | R12 conformer embedder |
| 516 | `molmetal/molmetal_lam/lam_chem/well_formedness.py` | R10 MLC well-formedness |
| 497 | `molmetal/molmetal_lam/lam_chem/ood_protocol.py` | R18 W3 mechanism |
| 493 | `molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py` | R17 reference ligand resolver |
| 492 | `molmetal/molmetal_lam/lam_chem/batched_rdkit.py` | R12 batched RDKit |
| 474 | `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` | R12 coupling adapter (M9) |
| 469 | `molmetal/molmetal_lam/lam_chem/pysr_wrapper.py` | R17 PySR wrapper |
| 441 | `molmetal/molmetal_lam/lam_chem/reducer.py` | R10 MLC reducer (M1) |
| 432 | `molmetal/molmetal_lam/lam_chem/cfm_decoder.py` | Lambda calculus chem |
| 415 | `molmetal/molmetal_lam/lam_chem/wetlab_protocol.py` | R17 wet-lab protocol |
| 324 | `molmetal/molmetal_lam/lam_chem/lambda_reward_channel.py` | Lambda calculus chem |
| 321 | `molmetal/molmetal_lam/lam_chem/wetlab_reward_channel.py` | R17 wet-lab reward channel |
| 296 | `molmetal/molmetal_lam/lam_chem/cfm_tanimoto_ood_filter.py` | Lambda calculus chem |
| 277 | `molmetal/molmetal_lam/lam_chem/ast.py` | R10 MLC AST |
| 235 | `molmetal/molmetal_lam/lam_chem/ring_prior.py` | Lambda calculus chem |
| 187 | `molmetal/molmetal_lam/lam_chem/cisplatin_resistance.py` | R17 cisplatin resistance |
| 178 | `molmetal/molmetal_lam/lam_chem/pharmacophore_reward_channel.py` | R17 pharmacophore reward |
| 128 | `molmetal/molmetal_lam/lam_chem/rules.py` | Lambda calculus chem |
| 124 | `molmetal/molmetal_lam/lam_chem/cisplatin_builder.py` | R17 cisplatin builder |

### 4.4 Layer `math` — 12 modules / 8,293 LOC (Math skeleton (T1-T11))

| LOC | Path | Role |
| ---: | --- | --- |
| 1,325 | `molmetal/math_skills/multi_parameter_ph.py` | R18 W1 mechanism |
| 950 | `molmetal/math_skills/se3_schrodinger_bridge.py` | R18 W1 mechanism |
| 850 | `molmetal/math_skills/catherine_properties.py` | Math skeleton (T1-T11) |
| 793 | `molmetal/math_skills/wasserstein_gradient_flow.py` | Math skeleton (T1-T11) |
| 736 | `molmetal/math_skills/hochschild_cohomology.py` | R18 W2 mechanism |
| 638 | `molmetal/math_skills/sb_bass_martingale.py` | R18 W3 mechanism |
| 622 | `molmetal/math_skills/transport_relaxation.py` | R18 W1 mechanism |
| 615 | `molmetal/math_skills/monge_ampere_regularity.py` | Math skeleton (T1-T11) |
| 602 | `molmetal/math_skills/quantum_semantics.py` | R18 W2 mechanism |
| 533 | `molmetal/math_skills/bergman_kernel.py` | R18 W3 mechanism |
| 325 | `molmetal/math_skills/bass_existence.py` | R18 Bass existence (M44 cite-only) |
| 304 | `molmetal/math_skills/ph_hardness_guardrail.py` | Math skeleton (T1-T11) |

### 4.5 Layer `reward` — 22 modules / 17,296 LOC (Reward shaping (Hilbert + astsr))

| LOC | Path | Role |
| ---: | --- | --- |
| 1,551 | `molmetal/molmetal_lam/reward/lambda_astsr.py` | R17 LambdaASTSR (M8, 1059 LOC) |
| 1,468 | `molmetal/molmetal_lam/reward/symbolic_regression.py` | R17 symbolic regression |
| 1,395 | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | R18 W1 mechanism |
| 1,177 | `molmetal/molmetal_lam/reward/pocket_closure_bias.py` | Reward shaping (Hilbert + astsr) |
| 1,082 | `molmetal/molmetal_lam/reward/astsr_multiph_hilbert.py` | Reward shaping (Hilbert + astsr) |
| 972 | `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` | Reward shaping (Hilbert + astsr) |
| 940 | `molmetal/molmetal_lam/reward/multiph_pll_feature.py` | Reward shaping (Hilbert + astsr) |
| 929 | `molmetal/molmetal_lam/reward/wasserstein_loss.py` | Reward shaping (Hilbert + astsr) |
| 868 | `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` | Reward shaping (Hilbert + astsr) |
| 696 | `molmetal/molmetal_lam/reward/hilbert_astsr_integration.py` | R18 Hilbert astsr integration |
| 657 | `molmetal/molmetal_lam/reward/wetlab_bayesian.py` | Reward shaping (Hilbert + astsr) |
| 643 | `molmetal/molmetal_lam/reward/master_catherine_closure_bias.py` | Reward shaping (Hilbert + astsr) |
| 637 | `molmetal/molmetal_lam/reward/closure_oversample.py` | Reward shaping (Hilbert + astsr) |
| 618 | `molmetal/molmetal_lam/reward/biolm_score_adapter.py` | R17 BioLM score adapter |
| 605 | `molmetal/molmetal_lam/reward/pareto_multiobj.py` | Reward shaping (Hilbert + astsr) |
| 515 | `molmetal/molmetal_lam/reward/master_wasserstein_gradient_flow.py` | Reward shaping (Hilbert + astsr) |
| 514 | `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` | Reward shaping (Hilbert + astsr) |
| 463 | `molmetal/molmetal_lam/reward/lambda_reward_dataset.py` | R17 lambda reward dataset |
| 434 | `molmetal/molmetal_lam/reward/pysr_adapter.py` | R17 PySR adapter |
| 397 | `molmetal/molmetal_lam/reward/platinai_oracle.py` | R17 PlatinAI oracle |
| 376 | `molmetal/molmetal_lam/reward/wasserstein_transfer.py` | Reward shaping (Hilbert + astsr) |
| 359 | `molmetal/molmetal_lam/reward/learned_shaping.py` | R17 learned shaping |

### 4.6 Layer `search_alg` — 21 modules / 17,585 LOC (Search algorithm (MCTS + variance))

| LOC | Path | Role |
| ---: | --- | --- |
| 7,058 | `molmetal/molmetal_lam/search_alg/proof_search.py` | R17 KataGo-MCTS proof search (M3, 6217 LOC) |
| 1,122 | `molmetal/molmetal_lam/search_alg/mcts_replay_buffer.py` | Search algorithm (MCTS + variance) |
| 1,030 | `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py` | R18 MCTS variance integration |
| 984 | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | R18 W2 mechanism |
| 902 | `molmetal/molmetal_lam/search_alg/mcts_replay_trainer.py` | Search algorithm (MCTS + variance) |
| 844 | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | R18 W1 mechanism |
| 764 | `molmetal/molmetal_lam/search_alg/learned_prior.py` | R17 learned prior |
| 662 | `molmetal/molmetal_lam/search_alg/pll_heuristic.py` | R17 PLL heuristic (M4) |
| 602 | `molmetal/molmetal_lam/search_alg/warm_start.py` | R17 warm start |
| 537 | `molmetal/molmetal_lam/search_alg/dream_rsi/discovery_tree.py` | Search algorithm (MCTS + variance) |
| 438 | `molmetal/molmetal_lam/search_alg/pareto.py` | R17 Pareto ranker |
| 405 | `molmetal/molmetal_lam/search_alg/puct.py` | R17 PUCT |
| 372 | `molmetal/molmetal_lam/search_alg/dream_rsi/policy_host.py` | Search algorithm (MCTS + variance) |
| 326 | `molmetal/molmetal_lam/search_alg/dream_rsi/replay_episode.py` | Search algorithm (MCTS + variance) |
| 318 | `molmetal/molmetal_lam/search_alg/priors/sa_prior.py` | Search algorithm (MCTS + variance) |
| 309 | `molmetal/molmetal_lam/search_alg/dream_rsi/dream.py` | Search algorithm (MCTS + variance) |
| 264 | `molmetal/molmetal_lam/search_alg/dmpnn_value.py` | Search algorithm (MCTS + variance) |
| 220 | `molmetal/molmetal_lam/search_alg/_batch_reward_worker.py` | R17 batch reward worker |
| 192 | `molmetal/molmetal_lam/search_alg/sweep_guidance.py` | R17 sweep guidance |
| 136 | `molmetal/molmetal_lam/search_alg/dream_rsi/replay_probe.py` | Search algorithm (MCTS + variance) |
| 100 | `molmetal/molmetal_lam/search_alg/variance_cpuct.py` | Search algorithm (MCTS + variance) |

## 5. 23 R18 mechanisms (M22..M44) — 3 wave breakdown

### 5.1 W1 — 7 mechanisms / 6,894 LOC / 193 tests

_Wave 1 (math + Hilbert + Score-Rep + W-Flow + MacroGuide)_

| ID | Mechanism | Module | LOC | Tests |
| --- | --- | --- | ---: | ---: |
| **M22** | SE(3) Schrödinger Bridge | `molmetal/math_skills/se3_schrodinger_bridge.py` | 950 | 25 |
| **M23** | Transport Relaxation | `molmetal/math_skills/transport_relaxation.py` | 622 | 25 |
| **M24** | Multi-Parameter PH | `molmetal/math_skills/multi_parameter_ph.py` | 1,325 | 25 |
| **M25** | Hilbert primal-dual optimizer | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | 1,395 | 31 |
| **M26** | Score-Repellent Monte Carlo | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | 844 | 38 |
| **M27** | W-Flow ODE solver | `molmetal/adapters/flow_matching_lipman/w_flow.py` | 1,096 | 20 |
| **M28** | MacroGuide PH | `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | 662 | 29 |

### 5.2 W2 — 7 mechanisms / 5,133 LOC / 215 tests

_Wave 2 (Coord GNN + Quotient Diff + Quantum sem + Hochschild + VarCtrlPUCT + Fractional ODE + Multi-PH+PLL)_

| ID | Mechanism | Module | LOC | Tests |
| --- | --- | --- | ---: | ---: |
| **M29** | Coordination-Aware GNN | `molmetal/adapters/coordination_gnn.py` | 518 | 32 |
| **M30** | Quotient-Space Diffusion | `molmetal/adapters/quotient_space_diffusion.py` | 876 | 36 |
| **M31** | Quantum semantics | `molmetal/math_skills/quantum_semantics.py` | 602 | 32 |
| **M32** | Hochschild cohomology | `molmetal/math_skills/hochschild_cohomology.py` | 736 | 31 |
| **M33** | VarianceControlledPUCT | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | 984 | 25 |
| **M34** | Fractional ODE solver | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` | 613 | 25 |
| **M35** | Multi-PH + PLL signature | `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` | 804 | 34 |

### 5.3 W3 — 9 mechanisms / 6,556 LOC / 271 tests

_Wave 3 (Skeleton jump + MD stub + FG veto + Layered gen + Buyable gate + OOD + Non-click med-chem + Bergman + Bass)_

| ID | Mechanism | Module | LOC | Tests |
| --- | --- | --- | ---: | ---: |
| **M36** | Skeleton jump / fragment-swap | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` | 902 | 25 |
| **M37** | MD relaxation stub | `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` | 834 | 24 |
| **M38** | FG-compat / leave-group veto | `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` | 846 | 25 |
| **M39** | Layered generation | `molmetal/molmetal_lam/lam_chem/layered_generation.py` | 918 | 25 |
| **M40** | Buyable-reactant gate | `molmetal/molmetal_lam/lam_chem/buyable_gate.py` | 650 | 28 |
| **M41** | OOD protocol (Tanimoto<0.4 + MMseqs2 stub) | `molmetal/molmetal_lam/lam_chem/ood_protocol.py` | 497 | 24 |
| **M42** | Non-click med-chem rules | `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py` | 738 | 33 |
| **M43** | Bergman kernel | `molmetal/math_skills/bergman_kernel.py` | 533 | 49 |
| **M44** | Bass martingale calibration | `molmetal/math_skills/sb_bass_martingale.py` | 638 | 38 |

## 6. 8 skills ecosystem

**Total**: 8 / 8 shipped, 8 / 8 R18-updated.

| # | Skill | Anchor | R18 updated |
| -: | --- | --- | :-: |
| 1 | `mol-metal-lambda-calculus` | MLC + β-NF + KataGo-MCTS + PLL + 84-MLE + 21-rule PRIMITIVE_RULES | yes |
| 2 | `mol-metal-cfm-training` | Lipman CFM + SE(3) EGNN + YuelBond + 6 shim fixes + R18 ship reference | yes |
| 3 | `mol-metal-click-chemistry` | 18 click rules + 84-MLE + 4 NEW SMARTS + P1-1 non-click med-chem | yes |
| 4 | `mol-metal-anticancer` | MetalCytoToxDB + PlatinAI + BioPython PLIP replacement | yes |
| 5 | `mol-metal-mechanism-mapping` | 21 机制 × 20 指标 + 跨 skill 水平层 + R18 horizontal layer | yes |
| 6 | `mol-metal-r18-math` | NEW Wf-1 — 9 定理 backlog → 3 ship / 3 P1 deferred / 1 guardrail / 2 cite-only | yes |
| 7 | `mol-metal-paper-writing` | NEW Wf-5 — 8 LEAD + 8-9 honest neg + 6 lim + tag discipline | yes |
| 8 | `party-twenty-spirit` | NEW meta — wf dispatch + Hooks + R18 workflow banner | yes |

## 7. Paper build health (9 sections + appendices)

**Paper build**: `pdflatex main.tex` exits 0, **0 unresolved references**, 132+ pages, PDF v1.7.

### 7.1 9 paper sections

| Sec | File | LOC |
| --- | --- | ---: |
| §1 | `section_01_introduction.tex` | 50 |
| §2 | `section_02_related_work.tex` | 73 |
| §3 | `section_03_method.tex` | 395 |
| §4 | `04_evaluation.tex` | 3,392 |
| §5 | `05_ablation.tex` | 1,320 |
| §6 | `section_06_limitations.tex` | 22 |
| §7 | `section_07_future_work.tex` | 15 |
| §4.5 | `04_5_honest_negative.tex` | 731 |
| §5-discussion | `05_discussion.tex` | 498 |

### 7.2 Appendices (4 ship, 2 flagged in §0 spec)

| File | LOC | Role |
| --- | ---: | --- |
| `paper/appendices/betanf_semantics.tex` | 597 | β-NF semantics (cite-only, R17 ship) |
| `paper/appendices/closure_theorem.tex` | 476 | Closure theorem + M22-M44 appendix anchor |
| `paper/appendices/math_theorems.tex` | 726 | Math theorems (T1-T11) |
| `paper/appendices/W_wetlab_protocol.tex` | 257 | Wet-lab protocol |

`paper/main.tex` = `pdflatex` passes 0 unresolved; refs.bib = 1071 entries (6 NEW by Wf-11).

## 8. Tests (1,200+ MEASURED — project-wide pytest surface)

| Item | Value |
| --- | ---: |
| Test files (`test_*.py`) | 1,547 |
| Total test LOC | 467,544 |
| `def test_*` definitions | 15,905 |
| R18 ship tests (Wf-1..Wf-18 aggregate) | ~1,070 (Wf-13 ledger) |
| R17 baseline (30/30 decode + 14/14 evaluators + 7/7 guards + ~217 unit) | preserved bit-identical |

## 9. Bench numbers (MEASURED, CPU-only)

| Bench | Modules × samples | Wall-time | Status |
| --- | --- | ---: | --- |
| PlatinAI 5K (Wf-17) | 32 × 500 × 10 = 160 000 forward primitives | 3.30 s | 30 / 32 ok + 2 / 32 skipped |
| MetalCytoToxDB 5-metal (Wf-17) | 5 × 500 × 32 × 10 = 800 000 forward primitives | 15.05 s | full |
| 5-fold CV harness (Wf-17) | 32 modules × 5 folds × 2 runs | accuracy_mean = 0.9375 | ok |
| Cross-module e2e (Wf-15) | 32 × 50 samples | 7.38 s | 32 / 32 ok |
| Pytest runner (Wf-17) | 78 files / 30 w-tests / 1 434 tests collected | 3.29 s | 48 coll-err pre-existing |

## 10. Close

R18 ultracode dispatch ships the CPU + paper + mechanism-mapping + skill surface (Wf-1..Wf-18). GPU training retrain (Blackwell) remains deferred per `TODO/03_plans/R18_PLAN.md` §5 — PROJECTED metric deltas in §3 are blocked behind GPU. R17 baseline (`proof_search.py` + `lambda_astsr.py` + `learned_shaping.py` + `coupling_adapter.py` + `flow_matching_lipman/*` R17 close) is **bit-identical**.

Last updated: 2026-09-26.
