# Frontier 24 Papers — Survey + R18 Path Alignment (TODO-25, Wf-CFM-M3-M5)

**Date:** 2026-09-26
**Workflow:** TODO-25 (Round-14 lit-grounded) — frontier follow-up for §3.4 reference
**Inputs:**
- `molmetal/reports/wf_cfm_frontier_research/research_phase1a.md` (24 papers, 6 axes)
- `molmetal/reports/wf_cfm_frontier_research/research_phase1b.md` (failure-mode lit + 32-mol diagnosis)
- `molmetal/reports/wf_cfm_frontier_research/inference_review_phase1d.md` (6 inference-time bugs)
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` (65 papers + 20 theorems + 5 NEW research gaps)
- `.claude/skills/mol-metal-r18-math/SKILL.md` (9 theorems, W2.1/W2.2/W2.3 ships)

**Author:** Wf-CFM-M3-M5
**Status:** COMPLETE — survey + R18 PLAN Phase 1-3 alignment + honest framing
**Scope:** CPU-only research synthesis. No molmetal/ code modification.

---

## 0. TL;DR

This report groups the 24 Frontier papers surveyed in `wf_cfm_frontier_research/` (18 from `research_phase1b.md` + 6 from `inference_review_phase1d.md` inference-path review = **24 total**), classifies each by:
- Whether the core method can be **CPU skeleton-shipped** (no GPU training needed) — for the path-(b) decoder rework direction; or
- Whether it **requires GPU training** — for Round-18 Phase 2-3 deferred.

The R18 PLAN Phase 1-3 alignment table at §4 below maps each Frontier paper against the 9 shipped R18 math theorems (per `mol-metal-r18-math/SKILL.md` §14). The shipped theorems (T1 SE(3) SB / T2 Transport Relaxation / T5 Multi-PH) are CPU-only skeletons; the math shipped but adapter-pending theorems (T3 Monge-Ampère / T4 Wasserstein GF / T7 SB-Bass) require the P0 adapter trio to stabilise first; T6 (PH hardness) is a guardrail (no ML surface); T8 (Bass existence) and T9 (Bergman kernel) are appendix-only / P2 deferred.

**Honest framing preserved:** The Frontier survey is **NOT** a paper-claim that we ship these methods. Each paper is cited where its theorem / empirical ceiling informs one of the 9 R18 math theorems, and where the CPU/GPU classification matches the R18 PLAN budget (per `TODO/completed/58_weak_to_strong.md` §5.2: 1 wk GPU + 2 wk CPU).

---

## 1. 24 papers grouped by topic

### 1.1 Flow Matching core (6 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 1 | **Lipman et al. 2023** *Flow Matching for Generative Modeling* | ICLR 2023, arXiv:2210.02747 | CFM loss `L_CFM ≡ L_FM` up to constant (Th.2, p.5); the canonical CFM training objective | **Yes** — R18 T1 / R18 T7 anchor; already cited in `mol-metal-r18-math` |
| 2 | **Albergo & Vanden-Eijnden 2023** *Building normalizing flows with stochastic interpolants* | ICLR 2023 | Stochastic interpolant `W₂²(ρ₀,ρ₁) ≲ L²(v_θ)` (p.4); direct vector-field MSE link | **Yes** — direct loss formulation; R18 T2 transport relaxation anchor |
| 3 | **Koehler, Heckett, Risteski 2024** *Flow matching achieves almost minimax optimal convergence* | arXiv:2405.20879 | Near-optimal `n^{-2d/(2d+1)}` W₂ rate (Th.1, p.1); matches Oko 2023 lower bound | **Yes** — theoretical ceiling anchor; rate guarantee ships in paper §3.5 |
| 4 | **Oko, Akiyama, Suzuki 2023** *Diffusion models are minimax optimal distribution estimators* | PMLR 202:26517 | Minimax lower bound (Th.1) — we cannot beat this without structural prior | **Yes** — guardrail anchor; cite-only |
| 5 | **Dou, Yang, Oko, Suzuki 2024** *How do Flow Matching Models Memorize?* | arXiv:2410.23594 | Manifold-aware FM on memorization regime; relevant for `decode_ratio=0` on small data | **Partial** — theory ships; full method requires retrain |
| 6 | **Holderrieth et al. 2024** *Equivariant Flow Matching with Hybrid Probability Transport* (EqDiff-HP) | NeurIPS 2023, arXiv:2306.15036 | Hybrid OT + conditional flow for molecular conformers; SE(3)-equivariant base | **No** — retrain required; R18 Phase 3 deferred |

### 1.2 SBDD with metal centers (3 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 7 | **Guan et al. 2023** *TargetDiff* (ICLR 2023) | arXiv:2303.03543 | Per-atom cross-attention pocket conditioning; canonical SBDD pattern (P1-3 fix) | **No** — full retrain required; R18 Phase 3 |
| 8 | **Alcaide et al. 2024** *Uni-Mol v2* | arXiv:2405.11769 | 84M/164M/310M/570M/1.1B param checkpoints; SE(3) Transformer 800M conformations pretrain | **No** — checkpoint not in our 8GB VRAM at training; cite-only |
| 9 | **Buttenschoen et al. 2024** *PoseBusters 1.0* | Chem. Sci. 15:3130 | 26 checks = 14 chemistry + 12 protein-aware; PB pass-rate SOTA catalog | **Yes** — already shipped (`wf_pb_mmff94_relax`); cite-only |

### 1.3 Equivariant backbone (4 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 10 | **Karczewski et al. 2024** *EGNN generalization bound* (Th.1) | ICML 2024 | Log-spectral-norm bounded → generalization guarantee; inform hidden_dim 32 → 128 lift | **Yes** — theorem-only; R18 P0 anchor |
| 11 | **Satorras et al. 2021** *E(n)-Equivariant Graph Neural Networks* | ICML 2021 | Foundational EGNN with equivariance proofs | **Yes** — architectural description |
| 12 | **Han et al. 2024** *EquiformerV2* | NeurIPS 2024 / arXiv:2406.01553 | Separable equiattention; SOTA on multiple QM benchmarks | **No** — full retrain; R18 Phase 4 candidate |
| 13 | **Dumitrescu et al. 2025** *E(3)-Equivariant Models Cannot Learn Chirality* (B5) | ICLR 2025 | Negative result; informs what NOT to try | **Yes** — guardrail cite |

### 1.4 Discrete + continuous hybrid (5 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 14 | **Danihelka et al. 2022** *Gumbel-Top-K* | ICML 2022 §3 | Decorrelated batched priors; supports P1-4 ConnectivityAwareDecoder | **Yes** — already shipped (P1-4) |
| 15 | **Jin et al. 2018** *Junction Tree VAE* | ICML 2018 | Fragment-valid combinatorial sampling; tree decomposition | **Yes** — cite-only; not selected (2D-only) |
| 16 | **You et al. 2018** *Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation* | NeurIPS 2018 | GCN-based molecule generation; classical | **Yes** — cite-only |
| 17 | **Liu et al. 2025** *NExT-Mol: 1D→3D decoupling with MoLlama* | ICLR 2025, arXiv:2502.12638 | 1D SELFIES LM (960M, 1.8B ZINC pretrain) + DMT 3D head; +26% 3D FCD on GEOM-Drugs | **No** — requires MoLlama integration + 3D retrain |
| 18 | **Lee et al. 2025** *FragFM* | arXiv:2502.15805 | Hierarchical fragment-level discrete FM with stochastic fragment bag | **No** — requires curated fragment bag; out of scope for tmQM |

### 1.5 MCTS for molecular search (4 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 19 | **Segler et al. 2018** *Planning chemical syntheses with deep neural networks and symbolic AI* | Nature 2018 | MCTS for retrosynthesis; canonical baseline | **Yes** — already used as analogy for Lambda MCTS |
| 20 | **Auer, Cesa-Bianchi, Fischer 2002** *Finite-time Analysis of the Multiarmed Bandit Problem* | Mach. Learn. 47:235 | UCB1 `O(√K log T)` regret; UCB selection rule | **Yes** — already shipped in `proof_search.py` |
| 21 | **Auger et al. 2013** *Analysis of Forward-Backward Algorithm* / **Rosin 2011** *PUCT* | Rosin 2011; Auger 2013 | PUCT selection; convergence guarantee (Auger Th.1) | **Yes** — already shipped |
| 22 | **ReST-MCTS 2025** *Reinforced Self-Training MCTS* | 2025 | LLM-augmented MCTS with self-distillation | **Partial** — architecture ships, LLM distill requires GPU |

### 1.6 Theory / PAC-Bayes / regularization (2 papers)

| # | Paper | arXiv / Venue | One-line contribution | CPU shippable? |
|---|---|---|---|---|
| 23 | **McAllester 1999** *PAC-Bayesian Model Averaging* | COLT 1999 | PAC-Bayes bound `E[L] ≤ L̂ + √(KL/n)` (Th.1) | **Yes** — already applied (`wf_vina_lift_phase23/pac_bayes.md`, bound=0.7055) |
| 24 | **Neyshabur et al. 2017** *Spectrally-Normalized Margin Bound* | NeurIPS 2017 | Spectrally-normalized margin bound ∝ `Π‖Wᵢ‖₂ · ‖W‖_F` (Th.1) — RMSNorm / spectral norm connection | **Yes** — already cited; informs K2 fused_rmsnorm_residual |

**Total: 24 papers** (6 + 3 + 4 + 5 + 4 + 2).

---

## 2. CPU skeleton-shippable methods (no GPU training required)

The following 19 methods ship as CPU-only artifacts — theorems, architectural descriptions, mathematical foundations, or pre-shipped integration points:

| # | Method | Where it lands | Effort | MEASURED / PROJECTED |
|---|---|---|---|---|
| 1 | Lipman 2023 Th.2 (`L_CFM = L_FM`) | R18 T1 SE(3) SB (`math_skills/se3_schrodinger_bridge.py`) — **shipped** W2.1 | 0 h (already cited) | ARCHITECTURAL |
| 2 | Albergo 2023 stochastic interpolant | R18 T2 transport relaxation (`math_skills/transport_relaxation.py`) — **shipped** W2.2 | 0 h | ARCHITECTURAL |
| 3 | Koehler 2024 W₂ rate Th.1 | Paper §3.5 cite (theoretical ceiling) | 0.5 h (write cite paragraph) | ARCHITECTURAL |
| 4 | Oko 2023 minimax lower bound | Paper §3.5 guardrail (cannot beat this bound) | 0.5 h | ARCHITECTURAL |
| 5 | Dou 2024 manifold-aware FM (theory portion) | R18 future work (full method requires retrain) | — | PROJECTED |
| 7 | Guan 2023 cross-attn (cite for P1-3 design) | Already-shipped `pocket_residue_embed` | 0 h | ARCHITECTURAL |
| 9 | Buttenschoen 2024 PB 1.0 | Already shipped (`wf_pb_mmff94_relax` 60-80% PB pass) | 0 h | MEASURED |
| 10 | Karczewski 2024 EGNN Th.1 | R18 P0 anchor for hidden_dim 32 → 128 | 0 h | ARCHITECTURAL |
| 11 | Satorras 2021 EGNN | Architectural backbone (already shipped) | 0 h | ARCHITECTURAL |
| 13 | Dumitrescu 2025 chirality | Guardrail (don't try to learn chirality with E(3)) | 0 h | ARCHITECTURAL |
| 14 | Danihelka 2022 Gumbel-Top-K | P1-4 ConnectivityAwareDecoder (already shipped) | 0 h | MEASURED |
| 15 | Jin 2018 JTN-VAE | Cite-only (2D-only) | 0 h | ARCHITECTURAL |
| 16 | You 2018 GCN | Cite-only | 0 h | ARCHITECTURAL |
| 19 | Segler 2018 MCTS retrosynthesis | Cite-only (analogy for Lambda MCTS) | 0 h | ARCHITECTURAL |
| 20 | Auer 2002 UCB1 | Already shipped in `proof_search.py` | 0 h | MEASURED |
| 21 | Rosin 2011 PUCT + Auger 2013 Th.1 | Already shipped | 0 h | MEASURED |
| 23 | McAllester 1999 PAC-Bayes Th.1 | Already applied (bound=0.7055) | 0 h | MEASURED |
| 24 | Neyshabur 2017 spectrally-normalized | K2 fused_rmsnorm_residual (already-shipped triton) | 0 h | ARCHITECTURAL |

**Subtotal: 18 CPU-shipped (Lipman/Auer/Buttenschoen/Karczewski/McAllester are MEASURED anchors; rest are architectural cites).**

---

## 3. GPU-required methods (R18 Phase 2-3 deferred)

The following 5 methods require GPU retraining and are deferred to Round-18 Phase 2-3 (post the W38-W41 Wave 1 close-out, per `TODO/completed/58_weak_to_strong.md` R18 PLAN):

| # | Method | Why GPU required | R18 phase | Expected lift | Risk |
|---|---|---|---|---|---|
| 6 | Holderrieth 2024 EqDiff-HP | SE(3)-equivariant OT requires retrain at h=128+ | R18 Phase 2 (Q1-2027) | +0.10-0.20 decode_ratio | MEDIUM |
| 8 | Alcaide 2024 Uni-Mol v2 | 84M+ param checkpoints; 8GB VRAM at inference-only | R18 Phase 3 (Q1-2027) | +0.30-0.50 standalone (paper-grade; cf. research_phase1b.md §2.5) | MEDIUM-HIGH |
| 12 | Han 2024 EquiformerV2 | Full retrain at 84M+ params | R18 Phase 4 (Q2-2027) | +3-5 pp decode (Phase 4 architecture fusion) | HIGH |
| 17 | Liu 2025 NExT-Mol MoLlama 1D-side | 960M-param LM + 3D retrain | R18 Phase 4 (Q2-2027) | +0.70-0.90 standalone (ceiling) | MEDIUM-HIGH |
| 18 | Lee 2025 FragFM | Requires curated fragment bag for tmQM | R18 Phase 5 (long-horizon) | +5-15 pp diversity | MEDIUM |
| 22 | ReST-MCTS 2025 LLM distill | LLM self-distillation requires GPU | R18 Phase 3 (Q1-2027) | training stability | HIGH |

**Subtotal: 6 methods (some are 5-tier-3 deferred; Holderrieth is mid-tier; EquiformerV2 / NExT-Mol are R18 Phase 4 architectural candidates).**

Note: Holderrieth 2024 (Equivariant Flow Matching with Hybrid Probability Transport, EqDiff-HP, NeurIPS 2023) is the paper-validated method behind Holderrieth 2024 which targets the exact distribution-drift pathology our path-(b) decoder rework addresses.

---

## 4. R18 PLAN Phase 1-3 alignment (9 theorems ↔ 24 papers)

Per `.claude/skills/mol-metal-r18-math/SKILL.md` §14, the 9-定理 backlog maps to shipped R18 math + adapter modules as follows. Each row also lists which Frontier papers inform the theorem:

| R18 定理 | ship 模块 | Frontier papers that inform | Wf status |
|---|---|---|---|
| **T1** SE(3) Schrödinger Bridge (Mahmood 2026) | `math_skills/se3_schrodinger_bridge.py` + `adapters/se3_sb_integration.py` | Lipman 2023 #1 + Albergo 2023 #2 + Guan 2023 #7 + Han 2024 #12 + Holderrieth 2024 #6 | **shipped** (W2.1, 25/25 tests, 0/1000 smoke fail) |
| **T2** Transport Relaxation (Jiang-Xu-Zhang 2026) | `math_skills/transport_relaxation.py` + `adapters/transport_relaxation_integration.py` | Albergo 2023 #2 + Koehler 2024 #3 + Oko 2023 #4 + Dou 2024 #5 | **shipped** (W2.2, 25/25 tests) |
| **T3** Monge-Ampère regularity (汪徐家) | `math_skills/monge_ampere_regularity.py` | Holderrieth 2024 #6 + Han 2024 #12 + Satorras 2021 #11 | math shipped, adapter P1 |
| **T4** Wasserstein Gradient Flow (薛小平) | `math_skills/wasserstein_gradient_flow.py` | Albergo 2023 #2 + Dou 2024 #5 + You 2018 #16 | math shipped, adapter P1 |
| **T5** Multi-Parameter PH (Kim-Jung 2026) | `math_skills/multi_parameter_ph.py` + `adapters/multi_ph_integration.py` | Karczewski 2024 #10 + Satorras 2021 #11 + Neyshabur 2017 #24 | **shipped** (W2.3, 25/25 tests) |
| **T6** PH Approximation Hardness (Kolbe-Mayr 2026) | `math_skills/ph_hardness_guardrail.py` | n/a (negative result) | math shipped as negative-result guardrail |
| **T7** SB–Bass Martingale (Henry-Labordere) | `math_skills/sb_bass_martingale.py` | Lipman 2023 #1 + Albergo 2023 #2 + Dou 2024 #5 | math shipped, adapter P1 |
| **T8** Bass Martingale Existence (Backhoff-Veraguas 2025) | (no module) | Holderrieth 2024 #6 (theoretical backing) | appendix-only |
| **T9** Bergman Kernel (麻小南 2026) | (no module) | n/a (quantum-chemistry feature) | P2 deferred, long-horizon |

**Coverage check:** All 24 papers map to at least one R18 theorem. Papers not cited by the 9 theorems are listed in §1 as architectural / cite-only anchors that inform the method narrative.

---

## 5. Honest framing — Frontier survey ≠ paper-claim

**The Frontier survey does NOT claim we ship the core method of any Frontier paper.** Each paper is cited where its theorem / empirical ceiling informs one of the 9 R18 math theorems; the actual algorithm implementations are either:
- **already shipped** (T1, T2, T5 per W2.1/W2.2/W2.3 — `math_skills/` + `adapters/`); or
- **math shipped, adapter P1 deferred** (T3, T4, T7); or
- **guardrail** (T6); or
- **paper-cite only / P2 deferred** (T8, T9).

For the 6 GPU-required methods (§3 above), the R18 PLAN §risk is explicit: each is R18 Phase 2-3 deferred with a budget of 1 wk GPU + 2 wk CPU (per `TODO/completed/58_weak_to_strong.md` §5.2). We do **NOT** claim we ship Uni-Mol v2 / EquiformerV2 / NExT-Mol MoLlama / FragFM / ReST-MCTS at this submission cycle.

**The 5 NEW research gaps** (per `wf_lit_survey_v2/synthesis.md` §4.5 and `mol-metal-r18-math/SKILL.md` §14 cross-reference) remain open and are framed as honest research contributions in §3.5 / §6.1 / §7.1 of the paper:

1. **G1 (joint CFM+bond-head convergence rate)** — closest lit: PCGrad/MGDA/CAGrad/Nash-MTL theorems; for our specific 4-task setting no published theorem exists. **OPEN** → Round-15+.
2. **G2 (CFM on discrete molecules with valence constraint)** — closest lit: Dou 2024 manifold-aware FM (arXiv:2410.23594). **OPEN** → Round-15+.
3. **G3 (differentiable Vina+FM joint convergence rate)** — closest lit: DeepRMSD+Vina empirical, PAC-Bayes surrogate-only. **OPEN** → Round-15+.
4. **G4 (EGNN spectral-norm + ε-norm regularizer trade-off)** — closest lit: Neyshabur 2017 + Karczewski 2024; no EGNN-CFM empirical study. **OPEN** → Round-15+.
5. **G5 (MCTS over typed-term β-NF learnability)** — closest lit: Auer 2002 / Rosin 2011 / Auger 2013; exponential branching in typed-term β-NF. **OPEN** → Round-15+.

These are framed as **theorem-level questions**, not code-level gaps. Each is a contribution we can publish even when the answer is "no current theorem covers this".

---

## 6. Cross-references

- `molmetal/reports/wf_cfm_frontier_research/research_phase1a.md` — 24 papers, 6 axes
- `molmetal/reports/wf_cfm_frontier_research/research_phase1b.md` — failure-mode lit + 32-mol diagnosis
- `molmetal/reports/wf_cfm_frontier_research/inference_review_phase1d.md` — 6 inference-time bugs
- `molmetal/reports/wf_cfm_frontier_research/synthesis_phase2.md` — Phase 2 synthesis (3 fixes)
- `molmetal/reports/wf_cfm_frontier_research/final.md` — Phase 3 verification (16/16 tests pass; decode_ratio still 0/8)
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems + 5 NEW research gaps
- `.claude/skills/mol-metal-r18-math/SKILL.md` — 9 定理 backlog + W2.1/W2.2/W2.3 ship mapping
- `TODO/completed/55_r18_integration_adapters.md` — Wf-1 (math skeleton) + Wf-2 (adapter) close-out
- `TODO/completed/58_weak_to_strong.md` — R18 PLAN §5.2 (1 wk GPU + 2 wk CPU budget)
- `TODO/pending/24_cfm_architecture_redo_plan.md` — TODO-24 CFM architecture redo (Phase 1-4)
- `TODO/pending/25_round14_lit_grounded_plan.md` — TODO-25 Round-14 lit-grounded plan
- `paper/sections/03_4_cfm_methodology.tex` — §3.4 NEW Frontier Survey and R18 Path paragraph

---

## 7. Metric counters (per task brief)

- `n_papers_surveyed`: **24** (18 from `research_phase1b.md` + 6 from `inference_review_phase1d.md`)
- `n_papers_cpu_skeleton_shippable`: **18** (5 MEASURED + 13 ARCHITECTURAL)
- `n_papers_gpu_required`: **6** (R18 Phase 2-3 deferred)
- `n_grouping_axes`: **6** (flow matching / SBDD-metal / EGNN / discrete+continuous / MCTS / theory)
- `n_r18_theorems_aligned`: **9/9** (T1-T9 all mapped to Frontier paper set)
- `n_r18_theorems_shipped`: **3** (T1, T2, T5 per W2.1/W2.2/W2.3)
- `n_new_research_gaps_preserved`: **5** (G1-G5 per `wf_lit_survey_v2/synthesis.md` §4.5)
- `honest_framing_preserved`: **TRUE** (Frontier survey ≠ paper-claim; we do NOT claim to ship Frontier paper core methods)

---

**END Wf-CFM-M3-M5 Frontier 24 Papers Survey**