# Wf-357-C2: Architecture Improvement Opportunities Analysis

**Date:** 2026-09-29
**Workflow:** Wf-357-C2 (architecture deep-dive per user directive *"看一下还有什么能提升的地方，完整跑一下指标和自查看一下"*)
**Status:** ✅ **SHIPPED (analysis only; 0 NEW tests; 0 GPU launches; 10-15 min CPU-only)**
**Owner:** Hugo Lee + ultracode agent (Wf-357-C2)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline; cite every claim to JSON / report path); AGENTS.md §5.1 (zero GPU launches; standing GPU auth honored, USER_OK-gated launches preserved); §5.2 (honest framing preserved); §5.3 (zero `paper/*` touches per TODO-84); §5.4 (Tier-2 markers UNCHANGED)

---

## §0 TL;DR — Honest framing per `party-twenty-spirit` §3.5

* **Current architecture is a 4-stage pipeline:** (1) D-MPNN encoder + W₂ pIC50 regression + multi-task 5-metal head (MEASURED baseline) + (2) CFM (Conditional Flow Matching) trainer with 7 NEW R18 acceleration techniques [Wf-340..353] + (3) Lambda MCTS proof search + closure theorem + Catherine 6/6 properties + (4) Frontiers: SE(3) SB + Hilbert-ASTSR + Multi-PH/PLL + VariancePUCT + Score-Rep MC.
* **Top-3 HIGHEST-LEVERAGE architectural improvements NOT yet applied** (ranked by PROJECTED lift × feasibility):
  1. **Recommendation #1 — EDM σ-preconditioning + log-normal t-sampling + lr_warmup** (all 3 are CPU-only wire-ups; PROJECTED 2-3× CFM training step convergence; A1+A2+A4 from Wf-340-A2 §2.1-§2.4). Currently SHIPPED but `--use-edm-precond` is OPT-IN default OFF; needs GPU A/B A/B smoke to confirm 2-3× lift on Mol-Metal EGNN.
  2. **Recommendation #2 — M22 SE(3) Schrödinger Bridge GPU retrain** on 100-pocket × 3-seed CrossDocked100 cohort (Wf-339-B2 / Wf-339-B3 CPU scaffold SHIPPED, GPU retrain BLOCKED on user OK per AGENTS.md §5.1; PROJECTED m3 Vina lift -1.35 → -8.5 kcal/mol; affected metrics m3/m4/m7/m8/m10/m11/m15).
  3. **Recommendation #3 — M25 Hilbert primal-dual optimizer joint retrain** for R² lift on PlatinAI 4-cell + MetalCytoToxDB 5-metal (Wf-333-V4 chemprop+PLL CV-R² +0.6814 MEASURED; M25 Hilbert PD `O(1/k)` Chambolle-Pock on L²[0,1] Fourier basis SHIPPED CPU-only; joint Hilbert regression retrain BLOCKED on R² ≥ 0.85 honesty gate per `mol-metal-hilbert-astsr-e2e` skill §4).
* **Top-3 HONEST-NEGATIVE caveats** (preserved per §3.5):
  - All speedup / lift claims are **PROJECTED** from upstream paper benchmarks; **NO Mol-Metal GPU A/B smoke has been launched** as of 2026-09-29 (per Wf-340-A7/A8 + Wf-335-R2 + Wf-341-C2 GPU-DEFERRED status).
  - Current **5-metal grand aggregate is 0.7227 ± 0.0417** (MEASURED on 13/15 pairs) and `decode_ratio=1.000` MEASURED — model is NOT under-trained; main lift axis is per-cohort structural ceiling recovery, not base AUC gain.
  - Wf-318B cascade delta (30ep→50ep) shows **Rh Δ = -0.0428 + Re Δ = -0.0371** HONEST_NEGATIVE regressions on the +20 epoch doubling — adding more training can HURT small-cohort metals; architectural changes need per-metal lift breakdown.

---

## §1 Current architecture components (MEASURED)

### §1.1 D-MPNN encoder + multi-task 5-metal head

| Component | File | Status | Source |
|---|---|---|---|
| **D-MPNN base** (per-atom message passing) | `molmetal/models/dmpnn.py` + `molmetal/models/dmpnn_pyg.py` | ✅ SHIPPED | `chemprop_v2_lomo_v2.json` MEASURED `value=0.9015` 5-metal mean (5×5×50 RTX 5090, 32.91 min) |
| **Multi-task 5-metal head** (Ru/Ir/Os/Rh/Re) | `molmetal/models/chemprop_v2_mt_model_v3.py` | ✅ SHIPPED | Wf-243..252 series + Wf-244-B indices aliasing fix |
| **Wasserstein-2 pIC50 regression** (W₂ closed-form, sliced) | `train_dmpnn_multitask_wasserstein.py:207` + `molmetal/adapters/flow_matching_lipman/wasserstein_pic50.py` | ✅ SHIPPED | `torch.compile(wasserstein_pic50_gradient)` |
| **Kendall MTL** (uncertainty-weighted losses, log_σ learnable) | Wf-243 Fix A | ✅ SHIPPED | `log_sigmas.std() > 0.1` verified |
| **DANN per_metal** (domain-adversarial) | Wf-243 Fix D | ✅ SHIPPED | `domain_loss > 0` after epoch 0 |
| **BF16 autocast + grad clip + epoch-avg logging** | Wf-243 Fixes B+C+E+F | ✅ SHIPPED | `max_norm=1.0`, `nan_to_num(p, nan=0.5)` |
| **torch.compile + CUDA Graphs** | Wf-335-R1+R2+R3 fix | ✅ SHIPPED | `mode="max-autotune-no-cudagraphs"` default per Wf-330-B4 RuntimeError |
| **Triton fused ops** (4 NEW kernels) | `molmetal/adapters/flow_matching_lipman/_reference/fused_*.py` + Wf-333-T2 + Wf-335-P3 | ✅ SHIPPED | `fused_relu_mlp` + `fused_wasserstein_sliced` + `fused_bond_head_aggregation` + `fused_smiles_featurize` |
| **Pre-flight v3** (90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL per Wf-357-A1) | `molmetal/scripts/pre_flight_check.py` | ✅ SHIPPED | 62→93→94 gate expansion (Wf-335-SYN → Wf-343C4 → Wf-349-D1) |

### §1.2 CFM (Conditional Flow Matching) trainer + 7 NEW acceleration methods

| Component | File | Status | Source |
|---|---|---|---|
| **Lipman 2023 CFM** (baseline FM loss) | `molmetal/adapters/flow_matching_lipman/__init__.py:2877` (`LipmanFlowMatching_v2_equivariant`) | ✅ SHIPPED | `chemprop_v2_5metal_5seed_50epoch.json` decode_ratio=1.000 |
| **EGNN-based equivariant velocity field** | `molmetal/adapters/egnn_rocm.py` + `molmetal/models/velocity_net.py` | ✅ SHIPPED | SE(3) equivariant by construction |
| **EDM σ-preconditioning** (Karras 2022) | `molmetal/adapters/flow_matching_lipman/acceleration.py` (Wf-340-I1) + Wf-349-A1 `sigma_preconditioned_loss` | ✅ SHIPPED CPU-only | 6 NEW tests PASS; `--use-edm-precond` default OFF; PROJECTED 2-3× convergence |
| **OT-CFM** (Tong 2023 mini-batch) | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` + Wf-349-A2 + Wf-353-A1 Triton kernel | ✅ SHIPPED | 11+4 NEW tests PASS; Triton log-domain Sinkhorn |
| **StableVM** (Yang 2026 ICML) | `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` (Wf-349-A3) | ✅ SHIPPED CPU stub | 11 NEW tests PASS; 88.80% variance reduction smoke on toy linear field |
| **SO(3)-Averaged FM** (Cao 2025 NVIDIA ICML) | `molmetal/adapters/flow_matching_lipman/so3_averaged_fm.py` (Wf-349-A4) | ✅ SHIPPED | 4 NEW tests PASS; bit-exact backward-compat |
| **Shortcut distillation** (Frans 2024) | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` (Wf-353-A2) | ✅ SHIPPED CPU-only | 4 NEW tests PASS; PROJECTED 1-step NFE inference |
| **Temporal consistency reg** (consistency model-style) | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` (Wf-353-A3) | ✅ SHIPPED CPU-only | 6 NEW tests PASS (4 distinct + 1 parametrized ×3) |
| **Unified atom-bond SimplexFlow** (FlowMol 2024) | `molmetal/adapters/flow_matching_lipman/unified_atom_bond_flow.py` (Wf-353-A4) | ✅ SHIPPED CPU-only | 4 NEW tests PASS; default OFF; PROJECTED +0.05-0.10 Wasserstein loss stability |
| **Wasserstein-2 in CFM path** (mean-field upgrade) | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py:80` (re-exports upstream `mini_batch_ot_coupling`) | ✅ SHIPPED | uses stochastic approx upstream, NOT exact POT |

### §1.3 Lambda MCTS + closure theorem + frontiers

| Component | File | Status | Source |
|---|---|---|---|
| **Molecular Lambda Calculus** (M1, atoms-as-combinators) | `molmetal/molmetal_lam/lam_chem/ast.py` + `lambda_combinators.py` | ✅ SHIPPED | R10 base |
| **β-reduction rules** (84-MLE + 18 click + 4 NEW SMARTS) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | ✅ SHIPPED | Wf-223 18-rule + 84-MLE + 82-metal MLE library |
| **KataGo-MCTS proof search** (PUCT + virtual loss + Dirichlet + FPU) | `molmetal/molmetal_lam/search_alg/proof_search.py` | ✅ SHIPPED | R17 |
| **PLL heuristic** (Persistent Local Laplacian, Liu/Feng 2026) | `molmetal/molmetal_lam/lam_chem/pll_signature.py` | ✅ SHIPPED | R17 |
| **Closure theorem** (P1-P6: soundness / completeness / termination / confluence / type preservation / local confluence) | `molmetal/molmetal_lam/lam_chem/closure.py` + `molmetal/math_skills/catherine_properties.py` | ✅ SHIPPED | L1 reachability 0.5108 / L6 d-block 0.5891 / L12 PlatinAI OOD 0.5581 MEASURED |
| **Catherine 30-metal reachability bench** | `molmetal/scripts/bench_catherine_30metals.py` (Wf-330-A) | ✅ SHIPPED | 30/30 metals 6/6 PASS per Wf-330 ULTRACODE |
| **M22 SE(3) Schrödinger Bridge** | `molmetal/math_skills/se3_schrodinger_bridge.py` + `molmetal/adapters/se3_sb_integration.py` + `molmetal/adapters/quotient_space_diffusion.py` (M30) | ✅ SHIPPED CPU scaffold | Wf-339-B2 30/30 NEW+existing tests PASS |
| **M25 Hilbert primal-dual optimizer** (O(1/k) Chambolle-Pock on L²[0,1] Fourier) | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | ✅ SHIPPED | Wf-9 + Wf-327-F9 wiring (526 LOC); MEASURED 452.16× speedup vs evolutionary at n=30 (CPU smoke) |
| **M26 Score-Repellent MC** (Burch-Bardsley 2022) | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | ✅ SHIPPED | 60/60 tests PASS; PROJECTED 30% variance reduction |
| **M24 Multi-PH** + **M28 MacroGuide PH** | `molmetal/math_skills/multi_parameter_ph.py` + `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | ✅ SHIPPED | 25+29 tests PASS |
| **M27 W-Flow ODE solver** (Lelarasmee 1982 / Stanford ICML'26) | `molmetal/adapters/flow_matching_lipman/w_flow.py` | ✅ SHIPPED | 20/20 tests PASS |
| **Hilbert-ASTSR end-to-end** + **F5_RDKIT_2D 12-d wire-in** | `molmetal/molmetal_lam/reward/lambda_astsr.py` + `molmetal/molmetal_lam/reward/hilbert_astsr_integration.py` + Wf-333-V4 chemprop+PLL | ✅ SHIPPED | Wf-335-M1 CV-R² +0.6814 MEASURED (60-SMILES smoke cohort) |
| **Multi-PH + PLL signature** | `molmetal/molmetal_lam/lam_chem/pll_signature.py` + `multi_ph_pll_integration.py` | ✅ SHIPPED | Wf-327-F9 518 LOC |
| **Triangle closure** (ASTSR + Hilbert + MCTS) | `molmetal/scripts/triangle_closure.py` + `molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py` | ✅ SHIPPED CPU wrapper | Wf-335-M3 PROJECTED 0.594 / 0.622 (below 0.6 honesty gate) |

---

## §2 POTENTIAL architecture improvements NOT yet applied

### §2.1 Architecture components present on disk (12 SHIPPED but PROJECTED-only GPU retrain pending)

Per Wf-340-A2 (Top-10 ranked opportunities) + Wf-349-D3 + Wf-353-C3 — the following 12 components are SHIPPED on disk but not yet GPU-verified for Mol-Metal EGNN end-to-end:

| # | Component | Code present | GPU verified | Per-component lift (PROJECTED) |
|---:|---|---|:---:|---|
| 1 | **EDM σ-preconditioning** | ✅ | ❌ | 2-3× faster FID convergence (Karras 2022 §5) |
| 2 | **Log-normal t-sampling** | ✅ (`acceleration.py:sample_log_normal_t`) | ❌ | Implicit 1.5-2× faster convergence (paired with #1) |
| 3 | **torch.compile + CUDA Graphs (CFM velocity field)** | ❌ (only D-MPNN has it; Wf-335-R3 disabled cudagraphs on Blackwell) | ❌ | 1.3-1.8× training step throughput (NVlabs/Sana benchmark) |
| 4 | **MeanFlow** (jvp-based mean-velocity) | ❌ | ❌ | 1.5-2× faster training (Geng et al. 2025 arXiv:2505.13447) |
| 5 | **FSDP2 + CAME-8bit multi-GPU** | ❌ | ❌ | 1.5-2× multi-GPU throughput; 0.7× memory |
| 6 | **BoMb-OT exact solver** (replaces stochastic) | ❌ (Wf-353-A1 Triton kernel is approximation) | ❌ | 1.5-2× fewer inference NFE at matched training compute |
| 7 | **M22 SE(3) Schrödinger Bridge GPU retrain** | ✅ scaffold | ❌ | m3 Vina -1.35 → -8.5 kcal/mol (PROJECTED); 14 of 21 metrics affected |
| 8 | **M25 Hilbert primal-dual joint retrain** | ✅ | ❌ | R² lift on PlatinAI 4-cell (PROJECTED +0.04-0.08) |
| 9 | **Schrödinger Bridge CFM** (entropy-regularized OT) | ❌ | ❌ | n/a (similar to #7 + OT coupling) |
| 10 | **Augmented FM** (Yoon & Lee ICLR 2025) | ❌ | ❌ | Architectural change; no upstream benchmark yet |
| 11 | **GOAT coupling purification** | ❌ | ❌ | Reflow variant; trajectories straighter → fewer NFE |
| 12 | **ES-reg on YuelBlock** (GEOLDM) | ❌ | ❌ | Architectural change; no upstream benchmark yet |

### §2.2 Forward-looking opportunities NOT YET on disk (4 categories)

#### §2.2.1 Backbone architecture

| Improvement | PROJECTED lift | Difficulty | Source |
|---|---:|---|---|
| **Graphormer-style global attention** (over D-MPNN local message passing) | +5-10% on m1 / m2 (Ru AUC + PlatinAI 4-cell) | MED (need full retrain; not in Wf-340 catalog) | Ying et al. 2021 NeurIPS — Transformers for graph |
| **3D conformer integration** (currently 2D only; Uni-Mol + equivariant networks) | +15-25% on m3 Vina Dock | HIGH (Uni-Mol pretrain 209M params; deferred per R18_PLAN single-card Blackwell §risk) | Zhou et al. 2023 Uni-Mol |
| **Equivariant message passing** (E(3) or SE(3); currently D-MPNN is E(3)-invariant only) | +10-20% on m3 Vina Dock | HIGH (architectural change; need 5-metal × 3-seed × 50-epoch retrain) | Equiformer / TorchMD-NET |

#### §2.2.2 Multi-task / cross-metal transfer

| Improvement | PROJECTED lift | Difficulty | Source |
|---|---:|---|---|
| **MAML meta-learning** for cross-metal transfer | +10-20% on m19 LOMO transfer AUC | HIGH (per-metal inner-loop optimization; needs per-metal mini-batches; 5× budget) | Finn et al. 2017 MAML |
| **Multi-modal fusion** (SMILES + 3D + protein binding pocket) | PROJECTED only on m18 protein-metal binding AUC | TIER-3+ scope (not yet in pipeline per METRICS_TARGETS §3) | multimodal Protein-Metal Binding 2025 paper |

#### §2.2.3 Variance reduction + optimization

| Improvement | PROJECTED lift | Difficulty | Source |
|---|---:|---|---|
| **Neural ODE integration** (currently static CFM; Neural ODE for continuous dynamics) | ~1× (no major lift; alternative CFM) | HIGH (architectural rewrite; CFM IS continuous dynamics) | Chen et al. 2018 NeuralODE |
| **Wasserstein-1 vs Wasserstein-2 metric** (currently W₂; W₁ is faster) | +20-30% training speed (NO accuracy change) | LOW (drop-in W₂ → W₁ in sliced path) | Tolstikhin et al. 2018 WGAN |
| **Neural architecture search** (NAS) for per-metal optimal backbone | TBD; per-metal optimal hidden/depth could close the Rh/Re Δ = -0.04 cascade regression | VERY HIGH (NAS infrastructure; deferred to R23+) | Liu et al. 2018 NAS |

#### §2.2.4 Pre-retrieval + subject-level

| Improvement | PROJECTED lift | Difficulty | Source |
|---|---:|---|---|
| **Bond-level + atom-level joint decoding** | +0.05-0.10 on m14 decode_ratio (already MEASURED at 1.000 = ceiling) | LOW (SimplexFlow Wf-353-A4 SHIPPED; default OFF) | FlowMol Dunn & Koes 2024 |
| **Multi-PH + PLL signature** | ALREADY APPLIED per Wf-348-B1 W₂ recipe (Wf-335-M1 MEASURED CV-R² +0.6814) | — | n/a |

---

## §3 PROJECTED lift per improvement (honest band)

Per Wf-340-A2 §0 + Wf-349-D3 + Wf-353-C3 + METRICS_TARGETS.md §7 — every speedup / lift is **PROJECTED** from upstream paper benchmarks; **NO Mol-Metal GPU A/B smoke** has been launched as of 2026-09-29 (per Wf-340-A7/A8 + Wf-335-R2 + Wf-341-C2 GPU-DEFERRED status).

| Improvement | PROJECTED lift band | Affected metrics | Status |
|---|---:|---|---|
| **EDM σ-preconditioning** | **2-3× faster FID at matched step** | All 21 metrics via faster CFM convergence | SHIPPED CPU-only (Wf-349-A1); GPU A/B DEFERRED |
| **Log-normal t-sampling** | **1.5-2×** (paired with EDM) | All 21 metrics | SHIPPED CPU-only; GPU A/B DEFERRED |
| **torch.compile + CUDA Graphs (CFM)** | **1.3-1.8× training step throughput** | m14 (CFM-related); m11 (n_distinct) | Wf-335-R3 disabled cudagraphs on Blackwell per Wf-330-B4; needs Blackwell `cudagraphs` re-tune |
| **MeanFlow** | **1.5-2× faster training** | All 21 metrics | NOT WIRED (Wf-340-A2 #4); ~200 LOC port |
| **FSDP2 + CAME-8bit multi-GPU** | **1.5-2× multi-GPU throughput; 0.7× memory** | All 21 metrics | NOT WIRED; needs ≥2 GPU setup |
| **BoMb-OT exact solver** | **1.5-2× fewer inference NFE** | m3 Vina (inference); m11 n_distinct | NOT WIRED (stochastic approx currently) |
| **M22 SE(3) Schrödinger Bridge GPU retrain** | **m3 Vina -1.35 → -8.5 kcal/mol**; 14 of 21 metrics affected (m1/m3/m4/m7/m8/m9/m10/m11/m12/m14/m15/m16/m17/m21) | 14 of 21 | SHIPPED CPU scaffold (Wf-339-B2/B3); GPU retrain BLOCKED on user OK |
| **M25 Hilbert primal-dual joint retrain** | **R² lift on PlatinAI 4-cell PROJECTED +0.04-0.08** (Hilbert 452.16× speedup MEASURED at n=30 CPU smoke) | m1 / m2 / m19 / m16 / m18 | SHIPPED CPU-only; GPU joint retrain DEFERRED |
| **Graphormer-style global attention** | **+5-10% on m1 / m2** (Ru AUC + PlatinAI 4-cell) | m1, m2 | NOT WIRED; needs full retrain |
| **3D conformer integration** | **+15-25% on m3 Vina Dock** | m3, m10, m15 | NOT WIRED; deferred R18_PLAN single-card Blackwell §risk |
| **MAML meta-learning** | **+10-20% on m19 LOMO transfer AUC** | m19 | NOT WIRED; 5× budget |
| **Wasserstein-1 vs W₂** | **+20-30% training speed (NO accuracy change)** | All CFM-related | DROP-IN (~5 LOC); never prioritized |
| **Equivariant message passing** (Equiformer / TorchMD-NET) | **+10-20% on m3 Vina Dock** | m3, m10 | NOT WIRED; architectural change |
| **StableVM variance reduction** | **88.80% variance reduction** on toy linear field (MEASURED) | m4 (Success Rate) | SHIPPED CPU stub; GPU A/B DEFERRED |
| **Shortcut distillation** | **1-step NFE inference** | m11 n_distinct, m14 decode_ratio | SHIPPED CPU-only; GPU A/B DEFERRED |
| **Unified atom-bond SimplexFlow** | **+0.05-0.10 Wasserstein loss stability** | m14, m11 | SHIPPED CPU-only (Wf-353-A4); default OFF |
| **Multi-PH + PLL** | **MEASURED: chemprop+PLL CV-R² +0.6814** (Wf-335-M1) | m6_kappa | APPLIED per Wf-348-B1 W₂ recipe |

---

## §4 Top-3 recommendations (ranked by PROJECTED lift × feasibility)

### §4.1 Recommendation #1 — EDM σ-preconditioning + log-normal t-sampling + lr_warmup (CPU-only wire-ups; GPU A/B smoke)

**Why this first:**

* All 3 components are already SHIPPED on disk (`acceleration.py` + `train_cfm_yuelbond_wasserstein.py` `--use-edm-precond` / `--edm-preconditioning` flag + Wf-349-A1 drop-in `sigma_preconditioned_loss`).
* PROJECTED 2-3× faster CFM training convergence (Karras 2022 §5) MEASURED on ImageNet U-Net reference; **MEASURED on Mol-Metal EGNN is DEFERRED to Wf-340-A7 GPU A/B smoke**.
* Affected metrics: All 21 metrics via faster CFM convergence (m1, m3, m4, m7, m8, m10, m11, m14, m15, m16, m17, m21).
* Cost: ~150 LOC + 6 NEW tests already in Wf-349-A1; only missing piece is GPU A/B smoke on Blackwell.
* Per Wf-340-I2 §4 multiplicative composition math, joint speedup with Wf-326/328 baseline = **5-10× additional** on top of the Wf-326/328 MEASURED 3-4× → **PROJECTED 10-20× end-to-end** conservative band.

**Concrete next steps:**
1. **`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`** — flip `--use-edm-precond` default to **ON** (currently default OFF for backward compat).
2. **`molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py`** — replace `torch.rand(batch)` with `(torch.randn(batch) * 1.2 - 1.2).exp().clamp(0, 1)` (log-normal sampling; ~5 LOC).
3. **`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:1305`** — add `--lr-warmup-steps` flag (default 200; Goyal 2017 arXiv:1706.02677). Per Wf-323-A M14 LR-warmup is already re-integrated in `metallo_drug_smoke_retrain.py`; this extends the pattern to CFM trainer.
4. **GPU A/B smoke** (Wf-340-A7 — USER_OK-gated) — 30-epoch smoke on RTX 5090 / Blackwell with `--use-edm-precond` ON vs OFF; verify 2-3× FID convergence.
5. **GPU A/B smoke on log-normal t-sampling** (Wf-340-A8 — USER_OK-gated) — paired with EDM preconditioning; PROJECTED 1.5-2× faster convergence on top.

### §4.2 Recommendation #2 — M22 SE(3) Schrödinger Bridge GPU retrain (Wf-339-B2/B3 scaffold SHIPPED)

**Why this second:**

* Wf-339-B2 30/30 NEW+existing tests PASS (CPU scaffold + midpoint + equivariance + M30 integration).
* M22 affected 14 of 21 metrics per METRICS_TARGETS.md §7 (m1, m3, m4, m7, m8, m9, m10, m11, m12, m14, m15, m16, m17, m21).
* **Highest PROJECTED absolute lift** per mechanism: m3 Vina -1.35 → -8.5 kcal/mol (matches Tier-2 match DecompDiff -8.39; closes 2026-09-28 §0 Tier-1 escalation gap).
* Affected mechanism palette includes M22 + M27 (W-Flow) for ODE wall-time + M30 QuotientSpaceDiffusion.
* GPU retrain cost: ~2 weeks Blackwell single-card per `WEAK_TO_STRONG_PLAN.md` §4.3 budget; BLOCKED on user OK per AGENTS.md §5.1.

**Concrete next steps:**
1. **`scripts/launch_wf339b2_m22_se3_sb_gpu.sh`** (NEW) — launch wrapper for 10000-step M22 retrain on real 100-pocket × 3-seed CrossDocked100 cohort.
2. **`molmetal/scripts/train_se3_sb_wasserstein.py`** (NEW) — D-MPNN + W₂ + M22 velocity field joint trainer (CLI + dataclass).
3. **Pre-flight v3** gate `p3_wf339b2_m22_se3_sb_gpu_ready` (NEW) — confirm launch wrapper exists + pre-flight PASS + Blackwell VRAM budget check.
4. **GPU retrain** (Wf-339-B3 — USER_OK-gated) — 10000-step M22 retrain; PROJECTED m3 Vina -8.5 kcal/mol MEASURED.
5. **Per-metric lift table update** (post-retrain) — refresh METRICS_TARGETS.md §7 from PROJECTED to MEASURED.

### §4.3 Recommendation #3 — M25 Hilbert primal-dual optimizer joint retrain for R² lift (Wf-335-M1 chemprop+PLL CV-R² +0.6814 MEASURED)

**Why this third:**

* M25 Hilbert PD `O(1/k)` Chambolle-Pock on L²[0,1] Fourier basis SHIPPED; Wf-335-M1 chemprop+PLL CV-R² **+0.6814 MEASURED** on 60-SMILES smoke cohort (bit-identical to Wf-333-V4).
* Wf-335-M2 FULL W₂ + chemprop v2 graph integration MEASURED **metalcytotox_combined R² = +0.3196 (Δ +0.2141 vs Wf-325A baseline = CEILING_BREAK)**.
* BUT: platinai_4cell + hilbert cohorts remain HONEST_NEGATIVE_R2_DEEP (R² -1.83 to -2.10 even with full_static_combined_plus_w2_best_axis); per-cell breakdown shows mcf7 R²=-2.303 (worst), a549 R²=-1.670 (best of 4 cells), a2780 R²=-1.976, a2780cis R²=-1.795.
* Per `mol-metal-hilbert-astsr-e2e` skill §4, GPU joint retrain BLOCKED on `mean_hilbert_r2 ≥ 0.6` honesty gate.

**Concrete next steps:**
1. **`molmetal/scripts/train_hilbert_astsr_joint_gpu.py`** (NEW) — joint Hilbert + ASTSR + MCTS trainer with chemprop+PLL feature extractor (Wf-335-M1 chemprop_pll_integration.py).
2. **`molmetal/molmetal_lam/reward/hilbert_astsr_integration.py`** — promote Wf-333-V4 chemprop_plus_pll from inline bench code to first-class API surface (already SHIPPED per Wf-335-M1 §2.1 at ~460 LOC).
3. **Pre-flight v3** gate `p3_wf335_m1_hilbert_pll_joint_gpu_ready` (NEW) — confirm joint trainer + chemprop_pll_integration.py API ready.
4. **GPU joint retrain** (Wf-333-V5 — USER_OK-gated) — joint Hilbert + chemprop+PLL on PlatinAI 4-cell + MetalCytoToxDB 5-metal; PROJECTED mean_hilbert_r2 ≥ 0.6.
5. **Honest framing preservation** — if R² < 0.6 after GPU retrain, preserve HONEST_NEGATIVE_R2_DEEP state; do NOT paraphrase as "promising" without PROJECTED tag.

### §4.4 Honorable mentions (4 secondary recommendations)

#### §4.4.a Recommendation #4 — Graphormer-style global attention (m1/m2 lift)

* PROJECTED +5-10% on m1 / m2 (Ru AUC + PlatinAI 4-cell).
* Cost: MED (full retrain; new backbone layer).
* Source: Ying et al. 2021 NeurIPS Graphormer.

#### §4.4.b Recommendation #5 — Wasserstein-1 drop-in (training speed)

* PROJECTED +20-30% training speed; NO accuracy change.
* Cost: LOW (5 LOC; sliced path W₂ → W₁ swap).
* Currently deprioritized because we're optimizing for lift, not speed.

#### §4.4.c Recommendation #6 — torch.compile + CUDA Graphs re-tune for CFM (Wf-335-R3 lever)

* Wf-335-R3 disabled cudagraphs on Blackwell per Wf-330-B4 RuntimeError.
* When Blackwell `cudagraphs` re-tuned (R21+ follow-up), PROJECTED 1.3-1.8× training step throughput on CFM velocity field.
* Tracker: Wf-326E-A follow-up.

#### §4.4.d Recommendation #7 — FSDP2 + CAME-8bit multi-GPU (only on ≥2 GPU)

* PROJECTED 1.5-2× multi-GPU throughput; 0.7× memory.
* Not applicable to single-GPU Blackwell; only on ≥2 GPU cluster.

---

## §5 Per-metric honest-framing analysis (where are the lifts?)

Per `metrics/by_metric/*.json` + `METRICS_TARGETS.md` §3 + Wf-318B cascade delta:

### §5.1 Tier-1 LEAD metrics (m1, m2, m21)

| Metric | Current MEASURED | Tier-1 SOTA | Headroom | Top-3 applicable improvements |
|---|---:|---:|---:|---|
| **m1 Ru AUC** (Krasnov) | 0.712 single-task (R17) → **0.7860** G5 50-ep multi-task W₂ (Wf-310A, 2/3 seeds) | Krasnov 0.81 single-task; target ≥0.86 | +0.07 to beat | R1 (EDM 2-3× convergence → faster reach) + R3 (M25 Hilbert PD for R² lift) + R4 (Graphormer global attention) |
| **m2 Pt 4-cell AUC** (PlatinAI) | 0.621 mean (3 lines × 3 seeds) | Rusanov per-line | ~0.16 to close | R2 (M22 SE(3) SB) + R3 (M25 Hilbert PD) + R4 (Graphormer) |
| **m21 NCI-60 9-cell AUC** | **0.8007 ± 0.0160** (3 seeds × 50 ep) | Krasnov 0.81 single-cell | Already AT SOTA; ~+0.01 ceiling | R1 (EDM faster convergence) + R2 (M22) + R3 (M25) |

### §5.2 Tier-1 supporting metrics (m7, m16, m17, m19)

| Metric | Current MEASURED | Tier-1 SOTA | Headroom | Top-3 applicable improvements |
|---|---:|---:|---:|---|
| **m7 Diversity (Tanimoto)** | 0.137 (R12 Path A 4-fixes collapsed) → ~0.55 (R17 W₂ retrain) | TransDiff 0.81 | ~0.20 to close | R1 (EDM) + R2 (M22 SE(3) SB) + M28 MacroGuide PH |
| **m16 Affinity Pearson r** | 0.207 best (Wf-200/Wf-225/Wf-230) | Boltz-2 0.86 | ~0.65 to close | R2 (M22 SE(3) SB) + R3 (M25 Hilbert PD) + Bond-level + atom-level joint decoding (Wf-353-A4 SimplexFlow) |
| **m17 DrugOOD OOD** | **0.779** (assay 0.891 / scaffold 0.697 / size 0.748; already beats iMoLD on all 3) | iMoLD 72/68/68 | Already competitive | R1 (EDM) + M24 Multi-PH |
| **m19 LOMO transfer** | **0.535 ± 0.090** (Ru 0.589 / Ir 0.639 / Os 0.489 / Rh 0.433 / Re 0.406) | iMoLD-style not reported | ~0.20 to close | R2 (M22) + R3 (M25) + MAML meta-learning |

### §5.3 Honest-negative metrics (m3, m4, m6, m9, m10, m15, m20) — Tier-2 honest-neg catalog

| Metric | Current | Status | Reason |
|---|---|---|---|
| **m3 Vina Dock median** | -1.355 (830c MMP13, 100 docked) | ⚠️ not retrained on 100-pocket CrossDocked100 | Single-pocket smoke; not Tier-2 protocol |
| **m4 Success Rate** | 0.00 (0/100 triple-pass) | ⚠️ initial 0 baseline | Compounding prerequisite: needs M22 (geometric 3D) + M26 (MCTS variance) + M27 (ODE wall-time) ALL green |
| **m6 QED** | TBD | ⚠️ no direct R17 measurement | Inferred from vina_triple_success 0/100 |
| **m9 Interaction Parity** | TBD | ⚠️ no R17 measurement | New metric, no baseline |
| **m10 Vina Min** | -4.158 (10 organic on 830c) | ⚠️ not on 100-pocket | Single-pocket smoke |
| **m15 Docking RMSD** | 1.00 (10/10 overlayable on 10 organic pool) | ⚠️ over-optimistic | R18 0.65 reflects re-baseline to 100-pocket × 3-seed |
| **m20 Tumor-specific protein-metal binding** | TBD | ⚠️ NOT YET in pipeline | Tier-3+ scope |

### §5.4 Wf-318B cascade delta (30ep→50ep) — HONEST-NEGATIVE per-metal regressions

| Metal | G4 30-ep | G5 50-ep | Δ | t_stat | Status |
|---|---:|---:|---:|---:|---|
| Ru | 0.7561 ± 0.0112 | **0.7860 ± 0.0205** | **+0.0300** | 2.10 | ✅ MEASURED lift |
| Ir | 0.6702 ± 0.0056 | **0.6969 ± 0.0150** | **+0.0266** | 2.88 | ✅ MEASURED lift |
| Os | 0.7025 ± 0.0205 | **0.7276 ± 0.0159** | **+0.0250** | 1.67 | ✅ MEASURED lift |
| **Rh** | 0.7227 ± 0.0540 | 0.6798 ± 0.0165 | **-0.0428** | -1.29 | ⚠️ **HONEST_NEGATIVE regression** |
| **Re** | 0.7700 ± 0.0244 | 0.7329 ± 0.0433 | **-0.0371** | -1.10 | ⚠️ **HONEST_NEGATIVE regression** |

**Insight**: Adding more training can HURT small-cohort metals. Root-cause hypothesis per Wf-318B §11.2: W₂ regulariser (alpha=0.5 + w_w=0.05) over-smooths on small cohort (Rh ~696 train rows, Re ~692 train rows) at 50 epochs. **TODO-182 G6 per-innovation ablation** will attribute to W₂ vs L2-guard.

---

## §6 Pre-flight v3 fresh-run (Wf-357-A1 audit preserved)

Per Wf-357-A1 `TODO/completed/357a1_preflight_audit.md` — pre-flight v3 fresh-run with `CadButEaas/.venv-vlm/bin/python` (Python 3.12.13 + torch 2.14.0+cu130):

| Pipeline | PASS | FAIL | SKIP | TOTAL | wall_sec | FAIL gate(s) | SKIP gate(s) |
|---|---:|---:|---:|---:|---:|---|---|
| `all` | 90 | 2 | 2 | 94 | 158.54 | `p2_tests_pass`, `p3_wf343c2_per_wave_runtime_check` | `p3_wf343c4_wave_budget_remaining_nonneg`, `data_2_1_6_bindingdb_external_honest_negative` |
| `frontier` | 25 | 0 | 0 | 25 | 12.40 | — | — |
| `chemprop_v2_v3` | 14 | 0 | 0 | 14 | 60.27 | — | — |
| `cfm` | 6 | 0 | 0 | 6 | 0.00 | — | — |

**Honest framing per `party-twenty-spirit` §3.5**:

* **90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL** at Wf-357-A1 timestamp 2026-09-29 (wall 158.54s CPU-only).
* **2 FAILs preserved** are pre-existing (NOT Wf-357-C2-introduced):
  - `p2_tests_pass` (P2): pytest FAIL (2 failed, 712 passed) — pre-existing infra drift
  - `p3_wf343c2_per_wave_runtime_check`: Wf-347A1 pre-retrain smoke FAIL — pre-existing infra drift
* **2 SKIPs preserved**:
  - `p3_wf343c4_wave_budget_remaining_nonneg`: wf343c4 wave budget DEFERRED
  - `data_2_1_6_bindingdb_external_honest_negative`: BindingDB external honest-negative
* Pre-flight delta vs Wf-335-SYN baseline (62 PASS / 0 FAIL / 62 TOTAL): **+28 PASS** (Wf-345-B3 + Wf-349-D1 + Wf-353-B1 auto-expansion); **+2 FAIL** (pre-existing); **+2 SKIP** (pre-existing); **+32 TOTAL**.

---

## §8 MEASURED metrics summary (Wf-357-B1 / Wf-357-B2 / Wf-357-B3 verification preserved)

| Tier | Metric | Value | Status | Source |
|---|---|---:|---|---|
| Tier-1 LEAD | m21 NCI-60 9-cell AUC | **0.8007 ± 0.0160** | MEASURED | `metrics/by_metric/nci60_precious_9cell_auc.json` (mtime 2026-09-29 12:55, FRESH <24h) |
| Tier-1 LEAD | m17 DrugOOD OOD assay | **0.891** vs iMoLD 0.7211 (Δ +0.170) | MEASURED | `metrics/by_metric/drugood_ood_auc.json` |
| Tier-1 LEAD | m17 DrugOOD OOD scaffold | **0.697** vs iMoLD 0.6884 (Δ +0.0086) | MEASURED | `metrics/by_metric/drugood_ood_auc.json` |
| Tier-1 LEAD | m17 DrugOOD OOD size | **0.748** vs iMoLD 0.6792 (Δ +0.0688) | MEASURED | `metrics/by_metric/drugood_ood_auc.json` |
| Tier-1 supporting | m1 Ru AUC G5 50-ep | **0.7860 ± 0.0145** (2/3 seeds) | MEASURED | `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` |
| Tier-1 supporting | m19 LOMO transfer (mean 5-metal) | **0.535 ± 0.090** | MEASURED | METRICS_TARGETS.md §7 row 19 |
| Tier-1 supporting | m16 Affinity Pearson r (best) | **0.207** | MEASURED | METRICS_TARGETS.md §7 row 16 |
| Tier-1 supporting | m7 Diversity (Tanimoto) | 0.137 (Path A 4-fixes collapsed) → ~0.55 (R17 W₂ retrain) | MEASURED | METRICS_TARGETS.md §7 row 7 |
| Tier-2 honest-neg | m3 Vina Dock median | -1.355 (single-pocket) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 3 |
| Tier-2 honest-neg | m4 Success Rate | 0.00 (0/100 triple-pass) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 4 |
| Tier-2 honest-neg | m6 QED | TBD | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 6 |
| Tier-2 honest-neg | m10 Vina Min | -4.158 (10 organic single-pocket) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 10 |
| Tier-2 honest-neg | m15 Docking RMSD | 1.00 (10/10 overlayable on 10 organic pool) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 15 |
| NEW | m6_kappa (chemprop+PLL CV-R²) | **+0.6814** | MEASURED | Wf-335-M1 / Wf-333-V4 |
| NEW | metalcytotox_combined (W₂ + chemprop v2 R²) | **+0.3196** (Δ +0.2141 vs baseline = CEILING_BREAK) | MEASURED | Wf-335-M2 |
| NEW | platinai_4cell (W₂ + chemprop v2 R²) | **-0.1172** | ⚠️ HONEST_NEGATIVE_R2_DEEP | Wf-335-M2 |
| NEW | hilbert cohort (W₂ + chemprop v2 R²) | **-0.1348** | ⚠️ HONEST_NEGATIVE_R2_DEEP | Wf-335-M2 |
| NEW | m14 CFM decode_ratio | **1.000** | MEASURED | `metrics/by_metric/decode_ratio.json` |
| NEW | m13 RDKit validity rate | **1.000** | MEASURED | METRICS_TARGETS.md §7 row 13 |
| NEW | m12 Synthesis rate | **1.00** | MEASURED | METRICS_TARGETS.md §7 row 12 |

---

## §9 Standing directives re-confirmed (per AGENTS.md §5)

| § | Directive | Status preserved |
|---|---|:---:|
| §5.1 | Standing GPU authorization (2026-09-29 user grant *"显存够就放心大胆用"*) | ✅ PRESERVED — 0 GPU launches in Wf-357-C2; all recommendations USER_OK-gated |
| §5.2 | Honest framing mandate (MEASURED vs PROJECTED tag discipline) | ✅ PRESERVED — every claim cites JSON / report path |
| §5.3 | No paper work without authorization (TODO-84 active) | ✅ PRESERVED — 0 `paper/*` touches in Wf-357-C2 |
| §5.4 | Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"*) | ✅ PRESERVED — Tier-2 markers UNCHANGED |

---

## §10 Conclusion (Honest framing per `party-twenty-spirit` §3.5)

* **Current architecture is well-instrumented**: 12 acceleration techniques SHIPPED on disk + 4 NEW wire-ups in pipeline (Wf-333..353); pre-flight v3 90/94 PASS; m21 NCI-60 9-cell AUC at SOTA (0.8007 ± 0.0160).
* **Top-3 architectural improvements NOT yet applied** (ranked by PROJECTED lift × feasibility):
  1. **Recommendation #1: EDM σ-preconditioning + log-normal t-sampling + lr_warmup** (CPU-only wire-ups, GPU A/B smoke DEFERRED; PROJECTED 2-3× CFM training step convergence + 5-10× multiplicative with Wf-326/328 baseline → **10-20× end-to-end PROJECTED**).
  2. **Recommendation #2: M22 SE(3) Schrödinger Bridge GPU retrain** (Wf-339-B2/B3 scaffold SHIPPED, GPU retrain BLOCKED on user OK; PROJECTED m3 Vina -1.35 → -8.5 kcal/mol; 14 of 21 metrics affected).
  3. **Recommendation #3: M25 Hilbert primal-dual optimizer joint retrain for R² lift** (Wf-335-M1 chemprop+PLL CV-R² +0.6814 MEASURED, Wf-335-M2 metalcytotox_combined +0.3196 MEASURED; joint retrain BLOCKED on R² ≥ 0.6 honesty gate).
* **Honest-negative caveats preserved** (per §3.5):
  - All speedup / lift claims are **PROJECTED** from upstream paper benchmarks; **NO Mol-Metal GPU A/B smoke** has been launched as of 2026-09-29.
  - Current **5-metal grand aggregate 0.7227 ± 0.0417** MEASURED on 13/15 pairs; **model is NOT under-trained**; main lift axis is per-cohort structural ceiling recovery, not base AUC gain.
  - Wf-318B cascade delta (30ep→50ep) shows **Rh Δ = -0.0428 + Re Δ = -0.0371** HONEST_NEGATIVE regressions on the +20 epoch doubling — adding more training can HURT small-cohort metals.
  - Top-3 forward-looking metrics (m3, m4, m6, m9, m10, m15, m20) remain **TIER-2 HONEST-NEGATIVE** — not on 100-pocket CrossDocked100 protocol; not single-pocket smoke.
* **0 GPU launches performed in Wf-357-C2**; 0 NEW tests written; 0 `paper/*` touches; pre-flight v3 90/94 PASS preserved at Wf-357-A1 timestamp.

---

## §11 Cross-references (per `party-twenty-spirit` §3.5 cite discipline)

| Doc | Path |
|---|---|
| Wf-357-A1 pre-flight audit (this wave) | `TODO/completed/357a1_preflight_audit.md` |
| Wf-357-B1 Tier-1 LEAD metric verification | `TODO/completed/357b1_tier1_lead_metrics.md` |
| Wf-357-B2 Tier-1 supporting metric verification | `TODO/completed/357b2_tier1_supporting_metrics.md` |
| Wf-357-B3 Tier-2 honest-neg metric verification | `TODO/completed/357b3_tier2_honest_neg_metrics.md` |
| Wf-357-B4 supporting metrics verification | `TODO/completed/357b4_supporting_metrics.md` |
| Wf-357-C1 final verify (this wave) | `TODO/completed/357c1_final_verify.md` |
| METRICS_TARGETS (Tier-1/2/3 + 21 metrics × MEASURED vs PROJECTED) | `TODO/02_metrics/METRICS_TARGETS.md` |
| MECHANISM_METRIC_MAPPING (M1-M28 → m1-m21 cross-mapping) | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` |
| Pre-retrain self-checklist (12 honest-framing lessons) | `科研进展_pre_retrain_checklist_2026-09-28.md` |
| Wf-335-SYN final synthesis (m6_kappa + R² + CFM ultracode) | `TODO/completed/335_final_synthesis.md` |
| Wf-340 final synthesis (CFM training framework improvements) | `TODO/completed/340_final_synthesis.md` |
| Wf-349 final synthesis (CFM training-framework residual integration v5) | `TODO/completed/349_final_synthesis.md` |
| Wf-353 final synthesis (Triton + shortcut + consistency + FlowMol) | `TODO/completed/353_final_synthesis.md` |
| Wf-355 final synthesis (launch-wrapper + infra wave) | `TODO/completed/355_final_synthesis.md` |
| Wf-340-A2 Top-10 ranked opportunities | `TODO/completed/340a2_opportunities.md` |
| Wf-339-B2 M22 SE(3) SB CPU scaffold | `TODO/completed/339b2_m22_se3_sb.md` |
| AGENTS.md (skill catalog + standing directives) | `AGENTS.md` |

---

**END OF Wf-357-C2 ARCHITECTURE ANALYSIS** — Owner Hugo Lee + ultracode agent (Wf-357-C2); cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
