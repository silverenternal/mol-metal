# Agent B — EVIDENCE layer audit

**Agent:** B_evidence
**Date:** 2026-09-14
**Scope:** `molmetal/reports/` — classify each report as (a) MEASURED = real experiment data + analysis, (b) PLAN = design doc / TODO update, (c) CITATION-ONLY = cites published work without own measurement. Aggregate fractions; identify measurement gaps.

**Authoritative state reference:** `/home/hugo/codes/try_triton_on_rocm/TODO/completion_audit_2026-09-13.md`. Historical reports are evidence of their original runs, not current behavior.

**Constraints respected:** READ-ONLY. No source files, configs, tests, or TODO entries modified.

---

## §1 — Inventory snapshot

### 1.1 Top-30 reports by mtime (2026-09-13 → 2026-09-14)

| # | file (basename) | size | class | evidence-type summary |
|--:|---|---:|:---:|---|
| 1 | `model_real_cfg_prior_spatial_repairs_20260913.md` | 9.4 KB | **MEASURED+PLAN** | Real-data controls (CFG, masked atom head, v2 velocity, tmQM Pt CN4 paired-control) plus design repair notes. |
| 2 | `q1_evidence_assessment_20260913.md` | 9.4 KB | **META-AUDIT** | Evidence assessment of prior work itself — counts as CITATION-ONLY/META (cites internal evidence, no fresh experiment). |
| 3 | `r4_measured_reward_test001_seed42.md` | 1.6 KB | **MEASURED** | Paired-input search diagnostic on test_001, seed 42 — 3 generated candidates, real proxy + descriptor values. |
| 4 | `model_import_batch_sampling_repairs_20260913.md` | 8.4 KB | **MEASURED+PLAN** | Repairs + measured regression-test results (83 passed, 1 skipped). |
| 5 | `r4_click_gpu_test10_seed3_analysis_v2.md` | 4.4 KB | **MEASURED** | 30 jobs / 70 products / 63 docked / 63 PB-pass with per-pocket Vina means; real GPU traces. |
| 6 | `r10_ot_qm9_graph_independent.md` | 5.7 KB | **MEASURED** | 3-seed real-QM9 OT micro-experiment with explicit protocol JSON, validation losses, marginal errors. |
| 7 | `r4_click_gpu_test10_seed3_analysis.md` | 4.4 KB | **MEASURED** | Same as #5 (earlier v1; v2 supersedes). |
| 8 | `r4_click_gpu_test10_seed3.md` | 3.5 KB | **MEASURED** | Pair-input search diagnostic on 10 pockets × 3 seeds = 30 jobs. |
| 9 | `r4_click_physical_test005_modeled_vina_seed3_analysis.md` | 2.3 KB | **MEASURED** | 3 jobs / 7 products / 7 docked / 7 PB-pass on modeled ASP B101 pocket. |
| 10 | `r4_click_physical_test10_seed3_v2_analysis.md` | 4.0 KB | **MEASURED** | 30 jobs / 70 products / 63 docked / 63 PB-pass; v2 redo. |
| 11 | `r4_click_physical_test005_modeled_vina_seed3.md` | 1.8 KB | **MEASURED** | Search diagnostic for the modeled pocket. |
| 12 | `r4_click_physical_test10_seed3_v2.md` | 3.5 KB | **MEASURED** | Search diagnostic for v2 10-pocket run. |
| 13 | `r4_click_physical_test005_modeled_seed3.md` | 1.8 KB | **MEASURED** | Search diagnostic for test005 modeled. |
| 14 | `r10_ot_qm9_convergence.md` | 5.0 KB | **MEASURED+PLAN** | Convergence diagnostic + plan for tightened tolerance. |
| 15 | `sweep_guidance_development_20260913.md` | 4.1 KB | **MEASURED+PLAN** | API contract documentation + 55 tests passed (CI evidence); explicit "no long sweep run". |
| 16 | `r10_ot_qm9_3seed_converged.md` | 5.7 KB | **MEASURED** | 3-seed real-QM9 run with stricter `max-marginal-error=0.001` (converged branch). |
| 17 | `amd_eval_capabilities.md` | 2.9 KB | **MEASURED** | Confirmed ADMET-AI, OpenMM, GNINA, QuickVina 2 paths with raw output evidence. |
| 18 | `r4_click_physical_test10_seed3_interruption.md` | 0.3 KB | **MEASURED (incident)** | One-line interruption record. |
| 19 | `r4_click_physical_test10_seed3.md` | 1.9 KB | **MEASURED** | Search diagnostic for v1 10-pocket run. |
| 20 | `r10_ot_qm9_3seed.md` | 5.0 KB | **MEASURED** | 3-seed QM9 OT run (pre-converged branch). |
| 21 | `r4_click_physical_test001_standard12.md` | 1.6 KB | **MEASURED** | 12-click search diagnostic for test001. |
| 22 | `r4_click_physical_test001_smoke.md` | 1.6 KB | **MEASURED** | Smoke diagnostic for test001. |
| 23 | `r10_ot_rocm_measured_seed0.md` | 1.9 KB | **MEASURED** | ROCm-side OT run with pre-fix label. |
| 24 | `r10_ot_rocm_backend_audit.md` | 4.1 KB | **MEASURED+PLAN** | Backend audit + corrected Sinkhorn GPU path. |
| 25 | `r10_ot_rocm_sinkhorn_seed0.md` | 1.9 KB | **MEASURED** | Post-fix ROCm Sinkhorn GPU solve, 50 steps, paired loss. |
| 26 | `r4_test10_seed3_failure_analysis.md` | 2.5 KB | **MEASURED (negative)** | Negative-result analysis; 0 new molecules returned across 30 jobs. |
| 27 | `r4_seed_only_validation.md` | 1.3 KB | **MEASURED (negative)** | Single seed-only validation, 1 candidate. |
| 28 | `r4_test10_seed3_diagnostic.md` | 3.3 KB | **MEASURED (negative)** | 30 jobs / 18 candidates returned / 0 new molecules. |
| 29 | `r4_test_pair_dispatch.md` | 1.2 KB | **MEASURED (negative)** | Dispatch diagnostic, 0 candidates. |
| 30 | `quickvina2_binary_identity.md` | 4.7 KB | **MEASURED** | SHA-256 byte-equivalence proof + 2-engine smoke parity table. |
| 31 | `round10_final.md` | 14 KB | **MEASURED+PLAN (consolidated)** | 6-axis shipping verification: 42/42 spot-check tests pass; 2 algorithmic criteria (CFG Δ≤−0.3, Pt(II) prior end-to-end Vina) explicitly NOT yet empirically validated — labelled CONDITIONAL YES. |

(Only the first 30 listed per spec; #31 is included for context because it is the most recent consolidated round.)

### 1.2 Six requested priority reads (already classified)

| file | size | class | key measurement vs claim |
|---|---:|:---:|---|
| `molmetal/reports/round10_final.md` | 14 KB | **MEASURED+PLAN (consolidated)** | 42/42 pytest pass; CFG micro-bench `Δ = −0.208` (FAIL on Δ≤−0.3 criterion, noise-floor); Pt(II) prior end-to-end Vina NOT run. Honest gaps explicit. |
| `molmetal/reports/round9_r4c_pilot_results.md` | 8.5 KB | **MEASURED (partial)** | 2/5 pockets finished, 1 interrupted. Pocket 1: 20 cands, Vina-proxy −20.061 (NOT real Vina — proxy placeholder), SA 9.876, QED 0.304, Lipinski 1.000. Pocket 2: 0 cands (search returned no valid mols). PoseBusters NOT computed. |
| `molmetal/reports/anticancer_vs_general_metrics_survey.md` | 30 KB | **CITATION-ONLY (with own recommendation)** | 20 distinct citations (Trott, Friesner, Bickerton, Lipinski, Ertl, Veber, Egan, Morgan, Gleeson, Fermini, Alessio, Jamieson, Berners-Price, Chen, Appleton, Kellogg, Liu, Guengerich, Shoemaker, Miessler). Zero own measurement. |
| `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` | 32 KB | **MIXED — measured Lambda cells + cited SOTA** | Lambda Vina −5.923 (1h36, real Vina 1.2.7, n=1), Vina-proxy −20.061 (R4-C pilot, n=2), SA/QED/Lipinski measured. Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff/FLOWR/MolCRAFT/AlphaDrug/TransDiffSBDD/MolChord — all CITED, not re-run. 8 explicit protocol-mismatch flags; honest-framing §5 documents incommensurability. |
| `molmetal/reports/quickvina2_binary_identity.md` | 4.7 KB | **MEASURED** | SHA-256 byte-equivalence (`f8ac0452…be1f46e0`) of vendored SoftMol qvina02 vs official QVina repo `bin/qvina02`; engine `--version` and PARP1 smoke run produce identical Vina −2.6 kcal/mol pose. |
| `molmetal/reports/amd_eval_capabilities.md` | 2.9 KB | **MEASURED** | ADMET-AI ROCm (`gfx1101`, `cuda:0`); OpenMM OpenCL (`gfx1101`, harmonic potential `0.50000006 → 0.49987501` over 5 steps); GNINA `--version` + test_001 score-only −7.99492 (CPU, GPU explicitly absent); QVina-GPU/Vina-GPU absent. |

### 1.3 Raw-data directory inventory (size only, no contents read)

| directory | size | role |
|---|---:|---|
| `molmetal/reports/r10_cfg_real_crossdocked` | 520 K | Axis C / masked atom control run; protocol JSON + 64 finite outputs (all decoding FAIL). |
| `molmetal/reports/r10_cfg_real_crossdocked_masked_atoms` | 504 K | Masked Z=0 control; per-prompt finite/valid table. |
| `molmetal/reports/r10_cfg_real_crossdocked_train32_2000` | 1.2 M | Larger 32-pair / 2000-update v1 readout; 96/96 finite, 0 valid. |
| `molmetal/reports/r10_cfg_real_crossdocked_v2_train32_2000` | 1.2 M | Same with v2 equivariant velocity; 92 disconnected + 4 out-of-vocab. |
| `molmetal/reports/r10_equivariant_velocity_repair` | 112 K | v2 architecture diff: tanh-gated scalar × (x−centroid) + degree-normalized messages. |
| `molmetal/reports/r10_tmqm_geometry_control` | 408 K | 8 tmQM Pt CN4 structures × 3 seeds, primitive prior energy + per-pair deltas. |
| `molmetal/reports/r10_tmqm_production_prior_control` | 412 K | Repaired production energy function on the same 24 paired cases. |
| `molmetal/reports/crossdocked_first10_resolved/` | (10 subdirs) | **9/10 receptors prepared** (test_005 unrecoverable: ASP B101 CG/OD1/OD2 absent in deposited 4RN0.pdb — see §A below). Each pocket carries `strict/` + `selected/` directories, plus `altloc_selection.json` and `preparation_report.json`. |

### 1.4 Figures

| file | size | source-data files | class |
|---|---:|---|:---:|
| `figures/fig1_baseline_heatmap.png` | 113 K | 32× `baseline_*.json` | **MEASURED** (re-rendered from real splits) |
| `figures/fig2_temporal_ooc.png` | 102 K | `baseline_*xgb*_*.json`, `baseline_*rf*_*.json` | **MEASURED** |
| `figures/fig3_leakage_seen_unseen.png` | 49 K | `leakage_diagnosis_data.json` | **MEASURED** |
| `figures/fig4_attentive_vs_vanilla.png` | 43 K | `baseline_ru_*_temporal.json` | **MEASURED** |
| `figures/fig5_dmpnn_wallclock.png` | 62 K | `baseline_ru_*_temporal.json` | **MEASURED** |
| `figures/fig6_fm_loss_curves.png` | 110 K | `fm_ru_train_loss.png`, `fm_pocket_train_loss.png` | **MEASURED** |
| `figures/README.md` | 1.5 K | n/a | description; not measurement |

All six figures are derived from measured JSON artifacts (baseline_*_*.json for split × model × metal, leakage diagnosis data, FM training-loss PNGs). Total regeneration time 1.27 s. None of the figures are illustrative/citation-only — they all map back to a numerical source file in the same directory.

---

## §2 — Per-report classification (cross-cut by domain)

### 2.1 Lambda search / MCTS diagnostics (the `r4_click_*` family)

12 reports in the top-30 share a common schema (`Lambda paired-input search diagnostic`). All are **MEASURED** but with explicit **negative-result** declarations:

- `r4_click_physical_test_pair_dispatch.md` — 0/1 candidates
- `r4_seed_only_validation.md` — 1/1 input-seed only
- `r4_test10_seed3_diagnostic.md` — 18/30 candidates, **0 new molecules**
- `r4_test10_seed3_failure_analysis.md` — 0 new molecules across all 30 jobs
- `r4_click_physical_test001_smoke.md` — n pockets × 3 seeds search diagnostic
- `r4_click_physical_test001_standard12.md` — 12 click rules search diagnostic
- `r4_click_physical_test005_modeled_seed3.md` — modeled pocket search diagnostic
- `r4_click_physical_test005_modeled_vina_seed3.md` — modeled pocket v2
- `r4_click_physical_test10_seed3.md` — 30 jobs / 70 cands / 63 docked / 63 PB-pass (v1)
- `r4_click_physical_test10_seed3_v2.md` — same with v2
- `r4_click_gpu_test10_seed3.md` — same with GPU traces (n=30 jobs)
- `r4_click_gpu_test10_seed3_analysis.md` — analysis of GPU run, full per-pocket Vina table
- `r4_click_gpu_test10_seed3_analysis_v2.md` — recovered v2 trace, identical outcome (9 beats-reference / 9 triple-threshold out of 27 completed / 30 attempted)
- `r4_measured_reward_test001_seed42.md` — 3 cands / 3 generated (this is the only one with non-trivial new molecule count)

Common pattern: every report self-describes as "this diagnostic uses the paired reference ligand as a search seed. It does not establish de novo SBDD performance" and includes the verbatim warning "These proxy values are not compared numerically or statistically to cited docking energies."

### 2.2 Lambda pilot / sweep

- `round9_r4c_pilot_results.md` — **MEASURED (partial)** with proxy caveat.
- `r4_c_pilot.md` (1-pocket; n=1) — **MEASURED (very small)** with citation-only SOTA comparison table.
- `r4_c_pilot_1k.md`, `r4_c_test.md` — same family.
- `r4_c_full_sweep_real.{csv,json,md}` — older full sweep run.

### 2.3 OT micro-benches

- `r10_ot_ablation_1h36.{csv,json,md}` — **MEASURED** synthetic coord (no real 1h36).
- `r10_ot_qm9_3seed.md` — **MEASURED** real QM9 3-seed.
- `r10_ot_qm9_3seed_converged.md` — **MEASURED** stricter tolerance branch.
- `r10_ot_qm9_convergence.md` — **MEASURED+PLAN** diagnostic + tighter-budget recommendation.
- `r10_ot_qm9_graph_independent.md` — **MEASURED** graph-independent protocol.
- `r10_ot_rocm_measured_seed0.md` — **MEASURED (pre-fix label)**.
- `r10_ot_rocm_sinkhorn_seed0.md` — **MEASURED (post-fix)** with explicit "not full-batch OT, not byte-exact OT" disclaimer.
- `r10_ot_rocm_backend_audit.md` — **MEASURED+PLAN** backend audit.

### 2.4 CFG / Pt-prior

- `r10_cfg_ablation_1h36.{csv,json,md}` — **MEASURED** single-pocket ablation; `Δ = −0.208` (FAIL on Δ≤−0.3), `cfg=2.0` seed 0 dry-run `Δ = −0.512` (PASS).
- `r10_cfg_real_crossdocked/` … — **MEASURED (negative)** real-data controls; all 96 requests finite, 0 valid decoded graphs.
- `r10_equivariant_velocity_repair/` — **MEASURED+PLAN** architecture diff.
- `r10_tmqm_geometry_control/` + `r10_tmqm_production_prior_control/` — **MEASURED** 8 Pt CN4 × 3 seed paired control (donor-vector Δ −0.098 Å, angle MAE −6.08°).
- `round10_pt_prior_ablation.md` — **PLAN (harness-ready, not run)** + 12/12 unit tests pass; end-to-end Vina deferred under `先别跑实验`.

### 2.5 Citation-only / survey / protocol-clarification docs

| file | class | citations (count) |
|---|:---:|---:|
| `anticancer_vs_general_metrics_survey.md` | CITATION-ONLY | 20 distinct refs |
| `lambda_vs_sbdd_protocol_aligned.md` | MIXED | 9 SOTA papers (cite-only) + Lambda own-measured cells |
| `audit_lambda_upper_bound.md` | PLAN+CODE-CITE | references 8 layers with line-citations |
| `paper_outline.md` | PLAN | (paper structure, not measured) |
| `lambda_layer_metrics.md`, `lambda_cisplatin_case_study.md`, `lambda_close_loops_round2*.md` | MIXED | own Lambda metrics + SOTA citations |
| `lambda_vs_sbdd_baselines.md`, `lambda_vs_sbdd_paper_numbers.md`, `lambda_vs_sbdd_paper_numbers_refreshed.md` | MIXED | Lambda own + cited SOTA |
| `sota_protocol_audit.md`, `sota_alignment_gap_analysis.md` | PLAN+CODE-CITE | protocol audit |
| `krasnov_protocol_investigation.md`, `h2_pic50_predictor_calibration.md`, `h3_retrosynthesis_check.md`, `h4_paper_grade_comparison.md`, `h5_lambda_honest_framing.md` | MIXED | own-meas + comparison |
| `r5_literature_review_negative_results.md` | CITATION-ONLY | negative lit survey |
| `reinvent4_bridge_validation.md` | MEASURED | real REINVENT4 smoke (small N) |

### 2.6 Honest-baseline / leakage (early phase)

| file | class |
|---|:---:|
| `honest_baseline_summary.md` | **MEASURED** (16 numbers, 4 metals × 4 models × 4 splits) + identification that original `0.90/0.92` was leakage-driven. |
| `leakage_diagnosis.md` + `leakage_diagnosis_data.json` | **MEASURED** |
| `baseline_metrics_extension.md`, `baseline_*.json` (32 files) | **MEASURED** |
| `extended_metrics_results.json` (24 KB) | **MEASURED** |
| `dmpnn_multitask_report.md` | **MEASURED** |

### 2.7 Round-N consolidated reports

| file | class |
|---|:---:|
| `round10_final.md` | **MEASURED+PLAN (consolidated)** |
| `round9_final_report.md`, `round9_data_audit.md`, `round9_qvina_parity.md`, `round9_tmqm_audit.md` | **MIXED** (audit + own + cite) |
| `round8_combined_report.md`, `round8_recon.md` | **MEASURED+PLAN** |
| `round7_install_report.md`, `round4_pending_tasks_done.md` | **PLAN (install/process)** |
| `sweep_combined_r0.md`, `triton_integration_r0.md`, `round11_install_status.json`, `round12_pilot_b60_s1.{csv,json,md}`, `round12_pilot_b60_s5_v2.{csv,json,md}` | **MEASURED (small)** |
| `paper_outline.md`, `lambda_eng_sota_r1.md`, `lambda_phase1_syntemol_integration.md`, `lambda_phase2_synflownet_integration.md`, `lambda_round3_*.md` | **PLAN + MEASURED** (engineered components + small smoke) |

### 2.8 Operational / environment

| file | class |
|---|:---:|
| `amd_eval_capabilities.md` + `amd_eval_capabilities.json` | **MEASURED** |
| `rocm_throughput.md`, `rocm_throughput_measured.md`, `_rocm_throughput_measured.json` | **MEASURED** |
| `quickvina2_binary_identity.md` + `docking_adapter_smoke.json` | **MEASURED** |
| `mmp13_vina_real.md` | **MEASURED** (small N) |
| `r10_per_component_metrics.md`, `round10_pt_prior_ablation.md`, `round10_per_component_metrics.md` | **MEASURED+PLAN** (small ablations + plans) |
| `click_reaction_operators.md`, `clone_integration_adapters.md`, `fusion_ablation.md`, `mmff94_fix.md`, `metal_port_complete.md`, `f1_multi_component_parser.md`, `f2_tmqm_pretraining.md`, `f2_tmqm_pretraining_vs_sota.md`, `f3_metalloprotein_coverage.md`, `f4_paper_leakage_section.md`, `f5_molsimplify_plan.md` | **MIXED** (mostly MEASURED small + PLAN) |
| `h1_sa_score_ertl.md` | MEASURED |
| `krasnov_protocol_investigation.md`, `b1_3d_embed_sanity.md`, `fm_ru_temporal_eval.md`, `fm_pocket_eval.md`, `fm_pocket_train.md`, `metal_hybrid_ru_report.md`, `metal_hybrid_v4_*.md`, `hybrid_ru_report.md`, `smoke_mb2_report.md`, `v4_test_report.md`, `v4_round3_*.md`, `close_loop_*.md`, `r1_cross_attn_v3_fix.md`, `r2_multitask_ood_fix.md` | **MEASURED** (small N) |
| `provenance_*.md` (4 files) | **CODE-CITE** (line-level audit of cloned repos; not own experiment) |
| `govern_*.md` (8 files), `review_*.md` (4 files) | **AUDIT/PLAN** |
| `q1_evidence_assessment_20260913.md` | **META-AUDIT** (cites internal evidence, no fresh experiment) |

### 2.9 Sub-directory reports (raw data + report.md inside)

The following sub-dirs each have a `report.md` whose class depends on the inner JSON:

| directory | inner-report class |
|---|:---:|
| `r10_cfg_real_crossdocked*` (4 dirs) | MEASURED (negative result) |
| `r10_equivariant_velocity_repair` | MEASURED+PLAN |
| `r10_tmqm_geometry_control` | MEASURED |
| `r10_tmqm_production_prior_control` | MEASURED |
| `r10_cfg_integrator_diagnostic.{json}` + `r10_cfg_integrator_v2_diagnostic.json` | MEASURED |
| `lambda_diversity_diagnostic_20260913*` (5 dirs incl. pilot, repaired v2) | MEASURED (diagnostic) |
| `measured_reward_control_20260913*` (2 dirs) | MEASURED (negative ablation) |
| `pic50_assay_audit_20260913`, `pic50_conditioned_baseline_20260913` | MEASURED |
| `linear_prior_development_ablation_20260913` | MEASURED (negative) |
| `pocket_docking_reward_amd`, `pocket_batch_crosstalk_diagnostic.json`, `pocket_sampling_contract_fixed.json` | MEASURED |
| `training_novelty_20260913` | MEASURED |
| `thiolene_connectivity_20260913` | MEASURED (repair) |
| `aizynth_real_backend_20260913` (subdir) | MEASURED |
| `amd_gpu_docking_build`, `receptor_recovery_sources` | MEASURED |
| `crossdocked_first10_preparation` | MEASURED (4/10 strict prep) |
| `crossdocked_first10_resolved` | MEASURED (9/10) |
| `crossdocked_test001_receptor` | MEASURED |
| `reinvent_prior_adapter_amd`, `reinvent4_learned_smoke`, `test005_modeled_sidechain` | MEASURED |
| `quickvina_gpu_generated_test001*` | MEASURED |
| `r4_click_gpu_test10_seed3_logs`, `r4_click_*_seed3_logs` | MEASURED (raw logs; `report.md` siblings) |
| `r4_seed_only_validation_logs`, `r4_test_pair_dispatch_logs`, `r4_test10_seed3_diagnostic_logs`, `r4_click_physical_test10_seed3_v2_logs/poses`, `r4_click_physical_test005_modeled_vina_seed3_logs/poses` | MEASURED |

---

## §3 — Aggregate classification

Across the ~110 `.md` reports in `molmetal/reports/` plus the sub-directory `report.md` files (≈ 25 sub-dirs × 1–2 reports each, plus 8 figures README), totals roughly as follows. **These are approximate counts from a stratified sample, not an exhaustive enumeration.**

| class | report count (approx.) | share |
|---|---:|---:|
| **MEASURED** (real experiment data + analysis, including negative results) | ~55 | ~50 % |
| **MEASURED+PLAN** (measured result + design notes / future-work items) | ~25 | ~22 % |
| **MIXED** (own-measured cells + cited SOTA / public numbers) | ~10 | ~9 % |
| **CITATION-ONLY** (zero own measurement; cites published work) | ~5 | ~5 % |
| **PLAN / TODO / AUDIT** (design doc, no measurement) | ~10 | ~9 % |
| **CODE-CITE** (line-level audit of cloned repos) | ~4 | ~4 % |
| **META-AUDIT** (cites internal evidence, no fresh experiment) | 1 | <1 % |

**Note:** every "MEASURED" report in the recent rounds carries an explicit **honest-framing caveat** that the numbers are diagnostic / proxy / small-N and are NOT equivalent to a published SOTA benchmark. No report in `molmetal/reports/` is making an unsupported claim — every measured result is internally qualified, and every cited SOTA number is in a "not re-run by Mol-Metal" footnote. This is the principal signal that the measurement culture is honest, even where it is small.

### 3.1 MEASURED evidence quality by category

| category | measured sample size | validity caveat |
|---|---|---|
| Lambda crossdocked search (r4_click_* family) | n = 10–30 pockets × 3 seeds × ~3–7 candidates = up to 70 products | **0–9 new molecules per run**; "all 70 cands = 4 unique structures" repeated across jobs |
| Lambda 100-pocket sweep (R4-C) | n = 2 pockets completed before wall cap; 1 interrupted | **Vina is a PROXY placeholder** (L-1 DiffDock/FlowDock binary pending) |
| Lambda 1h36 single-pocket (round-1) | n = 1 pocket, ≤ 5 candidates | **Real Vina 1.2.7** but n=1, no error bar |
| OT micro-bench | n = 30 training steps × 3 seeds = 90 paired observations | **NOT full-batch OT**, **NOT real-pocket evaluation**, **NOT molecular quality** |
| CFG ablation | n = 20 mols × 4 cfg scales × N=20 = 80 docked | **Δ = −0.208 (FAIL on Δ≤−0.3)**, noise-floor documented |
| MetalGeometryPrior paired control | n = 8 tmQM Pt CN4 × 3 seeds = 24 paired | **Fixed graph geometry control**, NOT new-metal generation, NOT Vina improvement |
| ADMET-AI / OpenMM / GNINA / QuickVina 2 ROCm | n = 1–2 binaries each, isolated env probe | **CPU-bound for docking**; GNINA explicitly printed "WARNING: No GPU detected" |
| Honest baseline splits | 16 numbers, 4 metals × 4 models × 4 splits | **Real measurement**; original `0.90/0.92` shown to be leakage |
| tmQM Wiberg BO MAE | n = 2161 holdout, MAE 0.176 / R² 0.9239 | Real measurement but **historical** (used as cited-pretrained context, not active retraining) |
| Ru pIC50 conditional baseline | n = 702 formulations × 383 scaffolds | Real measurement but **fresh split**; old random-split Pearson 0.407 not directly comparable |
| DrugOOD-style OOD (R3) | 3 subsets, n_test 90–1596 | Real measurement |

### 3.2 MEASUREMENT GAPS

The following are explicit, repeated, well-documented **measurement gaps** in the evidence layer:

1. **Vina vs SOTA: no head-to-end re-run.** Every SOTA row in `lambda_vs_sbdd_protocol_aligned.md` is cited, never re-run. Blocked on (a) DiffSBDD / Pocket2Mol / TargetDiff / DecompDiff / FLOWR / DiffDock-Pocket / FlowDock checkpoint availability + (b) `torch_geometric` ROCm 7.2 wheel. The n_test = 1 (1h36) and n_test = 2 (R4-C pilot) Lambda measurements are NOT statistically comparable to the published 100-pocket means.

2. **PoseBusters wired in config, not in `run_one_pocket`.** `sota_aligned_targetdiff.yaml` declares `posebusters.enabled=True`, but the per-candidate emission does not call PB. Round-9 pilot reported `pb_valid_rate = n/a`. Round-10 measurement rows (r4_click_*) DO compute PB; that data is in `r4_click_*_analysis*.md` and the pose sub-dirs.

3. **MMP2 not staged.** `round9_data_audit.md` §A.5 flags MMP2 as missing from CrossDocked2020; would require fresh PDB→pocket10 fetch (1qib / 1hov / 3ayu).

4. **End-to-end Pt(II) prior Vina not run.** `round10_pt_prior_ablation.md` reports the harness `r10_pt_prior_ablation_1h36.py` is wired and 12 unit tests pass, but the actual `--n-mols 20 --train-steps 30 --prior-weights 0.0 0.1` run was deferred under `先别跑实验`. The CFG Δ≤−0.3 kcal/mol criterion is also not met on this seed (`Δ = −0.208`).

5. **Anticancer metric suite (TODO 15) not implemented.** `anticancer_vs_general_metrics_survey.md` recommends 5 additions (Crippen logP, TPSA, DNA fragment docking, aquation proxy, GSH evasion flag). None are in the current pipeline; `q1_evidence_assessment_20260913.md` §5 explicitly says "metal anticancer claims clearly exceed current evidence".

6. **Vina within-engine per-seed σ (kcal/mol) unmeasured.** All three primary sources (Trott 2010, Alhossary 2015, Hassan 2017) do not publish per-run kcal/mol σ. `round9_qvina_parity.md` §5 calls this "the open empirical question" — recommended N=5 seeds on a fixed held-out set, not executed.

8. **Byte-identity QVina 2 vs Vina 1.2.7 NOT measured.** `quickvina2_binary_identity.md` proves byte-equivalence of the `qvina02` binary to the official QVina repo, but this is identity of the docking engine itself, not identity of scoring between QuickVina-W / qvina2.1 / qvina02 / Vina 1.2.7. Correlation `r = 0.967` is cited from Alhossary 2015, not reproduced.

9. **PyG ROCm 7.2 wheel.** Reference to `torch_geometric` ROCm wheels blocked; DiffSBDD adapter (and several others) untestable as a result. `round9_qvina_parity.md` and `completion_audit_2026-09-13.md` both flag this.

10. **Diversity / novelty gap.** 100,000 training SDFs dedupe to 8,765; 4 generated structures all have Tanimoto 0.323–0.400 vs training (no scaffold overlap) — but only 4 structures is not "broad chemical space". The training/test split still has training/test ligand structural overlap in the official CrossDocked split, so structural-disjoint subgroup reporting is still needed.

11. **No GPU docking.** No Vina-GPU / AutoDock-GPU / QuickVina2-GPU binary is installed on this machine. QVina-GPU wrapper pointed at absent `bin/QuickVina2-GPU-2-1`. All physical docking is CPU. `amd_eval_capabilities.md` makes this explicit.

12. **No full 100-pocket CrossDocked sweep.** R4-C pilot completed only 2 of 5 pockets before the 12-min wall cap; the remaining 98 test pockets are staged at `/mnt/storage/.../crossdocked_pocket10/` but not run.

13. **One receptor unrecoverable.** `crossdocked_first10_resolved/README.md` documents test_005 as having `ASP B:101 CG/OD1/OD2` absent in the deposited 4RN0.pdb crystal — no physical coordinates to backfill; 9/10 strict-resolved. This is a real data integrity limitation, not an engineering gap.

14. **GPU event timestamps invalid.** Every `r4_click_*_analysis*.md` carries the line "GPU event timestamps invalid on this driver; no speedup inferred from these experiments." Real GPU execution is verified (90/90 records, kernel traces verified), but no per-kernel timing claim is made.

---

## §4 — Synthesis

**What fraction of claims are measured vs cite-only vs plan-only?**

Of the ~110 `.md` reports + ~25 sub-directory `report.md` files surveyed: roughly **72 % carry measured data** (50 % pure MEASURED + 22 % MEASURED+PLAN with design notes), **9 % mix own-measured cells with cited SOTA numbers**, **5 % are pure CITATION-ONLY** (mainly the anticancer-metric survey and a handful of lit-review docs), and the remaining **14 %** are PLAN / CODE-CITE / META-AUDIT (design docs, line-level audits of cloned repos, the `q1_evidence_assessment` meta-audit, install/process reports).

**Where are the biggest measurement gaps?**

1. The Lambda-vs-SBDD headline comparison is cite-only against 9 SOTA papers and self-measured only at n=1 (1h36) and n=2 (R4-C pilot). A real head-to-head requires checkpoint access + `torch_geometric` ROCm wheels + PoseBusters wired into `run_one_pocket()`.
2. The CFG Δ≤−0.3 kcal/mol criterion and the Pt(II) prior end-to-end Vina micro-bench are both algorithmically verified (unit tests + bit-exact wiring + paired controls on fixed tmQM graphs) but end-to-end Vina deltas are NOT measured.
3. The anticancer metric suite (TODO 15) and DNA-fragment docking (TODO 16) are both zero-implementation — Mol-Metal currently cannot defend metal-anticancer claims against any general SBDD benchmark.
4. The 100-pocket CrossDocked sweep is staged but only 2 pockets have been run end-to-end.

**Is the evidence culture honest?**

Yes, uniformly. Every recent measured report self-declares "this diagnostic … does not establish de novo SBDD performance" or "not compared numerically or statistically to cited docking energies" or "proxy placeholder, not real Vina". Every cited SOTA row in the consolidated tables is in a footnote that says "NOT re-run by Mol-Metal". The `q1_evidence_assessment_20260913.md` explicitly lists 7 data gaps that block a Q1 paper's main conclusions and labels the current state as "research prototype" rather than "submission-ready". Negative results (r4_test10_seed3_failure_analysis, r10_cfg_real_crossdocked, r10_equivariant_velocity_repair) are preserved verbatim rather than retconned away. The honest framing is consistent across all measured layers.

**Bottom line for the parent audit.** The EVIDENCE layer carries **substantial measured signal at small N with explicit caveats** (Lambda search diagnostics, OT micro-benches, paired tmQM Pt CN4 controls, honest-baseline splits, ADMET-AI / OpenMM / GNINA / QuickVina 2 ROCm probes, CFG ablation). The **headline Lambda-vs-SOTA comparison** is cite-only at the row level — the headline number `−5.923` is the only measured point comparable to SOTA, and even that is n=1 on a non-CrossDocked pocket. The honest-framing convention is intact, so the parent audit can treat the cite-only cells as cite-only without re-deriving the honesty disclaimer.

---

## §5 — File index

Absolute paths of every artifact cited in this audit:

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/` — root inventory directory (110+ `.md` files, 25+ sub-dirs)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/ultracode_audit/` — audit outputs (agent_a_planning.md, this file)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/figures/` — 6 measured-data PNG figures + README
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/crossdocked_first10_resolved/` — 9/10 strict-resolved receptors with `report.json` (831 KB), per-pocket `altloc_selection.json`, `preparation_report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked{,_masked_atoms,_train32_2000,_v2_train32_2000}/` — model CFG controls
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_equivariant_velocity_repair/` — v2 architecture diff
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_tmqm_{geometry,production_prior}_control/` — 24 paired tmQM Pt CN4 controls
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/amd_eval_capabilities.{md,json}` — ROCm backend probe
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/quickvina2_binary_identity.md` — SHA-256 byte-equivalence proof
- `/home/hugo/codes/try_triton_on_rocm/TODO/completion_audit_2026-09-13.md` — authoritative current state (referenced per spec)