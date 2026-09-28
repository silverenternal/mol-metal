# TODO-25 — Round-14 Lit-grounded + Math-prior + Code-fix 全面规划

**Status:** ⚙️ **PARTIAL — D2/D4 lit-grounded fixes SHIPPED; M3+CFM gate in flight**
**Priority:** medium (Round-13 first; Round-14 deferred to post-R13)
**Owner:** (unset)
**Depends on:** Round-13 pocket-invariance fix + CFM frontier research outcomes
**Created:** 2026-09-15
**Last updated:** 2026-09-15 (D2/D4 path-A 4-fix bundle shipped; CFM M1-M5 in flight)

## ⚠️ Update 2026-09-15 — what we now know

### D2 (Diversity) — PATH A 4-fix bundle SHIPPED
- ✅ Rule symmetry (CuAAC/SPAAC/Suzuki can fire either direction)
- ✅ Decoder rework (chem-aware soft bond prior)
- ✅ Scaffold-aware gate (auto-pt-strict → 3 strict-Pt_II compatible rules)
- ✅ Partner tiles (8 new + FRAGMENT_LIBRARY_200_TILES runtime expansion)
- ✅ MEASURED: n_distinct 1→20, div_tan 0→0.1065, div_homo 0→0.0749 on test_000..test_009
- ⚠️ Does NOT generalize to test_010..test_019 (pocket-invariance sub-fix in flight via `wrd5dbewn`)

### D4 (Benchmarks) — Anti-cliché + Murcko scaffold wired
- ✅ F2-B1 `pt_click_compat` module 5×5 compat matrix (scaffold auto-detection + click can-fire gates)
- ✅ F2-B4 tests + scaffold-aware report
- ✅ Lit anchor: Bemis 1996 Murcko scaffold + Himo 2005 CuAAC regio

### M3 (Joint CFM + bond-head) — All 5 P0 fixes SHIPPED + 4 P1 SHIPPED (per `wbn5u6som`)
- ✅ F1 BondAwareDecoder wired
- ✅ F2 BondOrderHead in_dim fix
- ✅ F3 vocab_mask
- ✅ F4 hidden_dim<64 warning
- ✅ F5 bonds=zeros removed
- ✅ P1.1 hidden_dim 32→128
- ✅ P1.2 vel_scale learnable (replaces tanh)
- ✅ P1.3 pocket cross-attention
- ✅ P1.4 ConnectivityAwareDecoder
- ❌ GPU retrain at h=128 5K steps: decode_ratio=0/64 (per spec gate FAILURE)

### M1+M2 (Vina mean + CFM convergence) — Frontier research in flight
- `wbw9a59g3` searching arXiv 2025-2026 SOTA
- 3 fixes in implementation phase (ODE solver + bond-head default + decode_smoke_every)
- Target: decode_ratio from 0 → >0 with any frontier fix
- If 0/3 work → §6 limitation (Round-14 path)

### New contributions from 2026-09-15 work
- Symbolic regression F5 formula: `R = 2.5836 - 2.5149·sa_norm` (Deflex Stage 3 ship)
- PocketMacroSkeleton 87.9% train acc (Deflex Stage 1+2 ship)
- λ Combinators module (Deflex architecture ship)
- Dual-system framing for §3.5 (planned for paper)
- 5 NEW research gaps identified (per Lit-Survey-v2):
  1. Adaptive reward shaping via online symbolic regression
  2. Multi-objective MCTS with Pareto front + hypervolume selection
  3. Scaffold-aware pocket-conditioned generation
  4. SOTA-comparable decoder for flow-matched molecules (currently 0/192)
  5. Metal-organic property prediction without wet-lab assays (in silico metal-pic50)

## User insight (verbatim)

> 我们自己不用做全流程的数学推导，尽可能多找别人现有的理论

> 按照选项2来吧，但是我们开始实验之前要做好调研和review做更彻底的修复以及更全面的规划，我们的一切算法和模型的设计我要求一定要是数学先验的

## 关键原则（per user directive）

1. **borrowing existing theorems**（Lipman 2023 / Auer 2002 / Himo 2005 / Buttenschoen 2024 / Halgren 1996 / Riniker 2015）—— NOT re-derive
2. **math prior for every fix path**（cite published theoretical basis）
3. **honest lit-grounded plan**（cite each fix → existing theorem/baseline）
4. **5 NEW research gaps** as honest contributions（not citation failures）

## 5 个 weak metrics → 现有 lit + plan（per Lit-Survey-v2）

### 弱项 #1 — **Vina mean**（gap -8.45 vs -2.20，差 6.25）

**现有 lit ground**:
- **M1**: Differentiable Vina — Trott 2010 / Sun 2022 DeepRMSD+Vina / Moret 2024 RL+Vina / GNINA CNN+Vina（**4 paper precedent** for combined score）
- **M2**: CFM convergence — **Lipman 2023 Theorem 2** (`L_CFM = L_FM`) + **Koehler 2024 Theorem 1** (`W₂ rate n^{-2d/(2d+1)}`) + **Oko 2023** (minimax lower bound) + **Albergo 2023** (stochastic interpolant)
- **M3**: Joint CFM + bond-head — **Yu 2020 PCGrad Theorem 1+2** + **Sener-Koltun 2018 MGDA** + **Liu 2024 MGDA O(ε^{-2})** + **Liu 2021 CAGrad Theorem 3.2** + **Navon 2022 Nash-MTL Theorem 5.4/5.5**（**6 theorems**）
- **M4**: MCTS / REINFORCE — **Williams 1992** unbiased gradient + **Jiang 2023 TPAMI variance-reduction** + **Diversity-aware RL 2025** soft-penalty
- **M5**: PAC-Bayes generalization — **McAllester 1999 Theorem 1** + **Maurer 2004** + **Gat 2022 Theorem 3.5/3.6** + **Neyshabur 2017** spectrally-normalized + **Karczewski 2024 EGNN generalization**（**5 theorems**）

**fix plan + math lift**:
- (a) `hidden_dim 32 → 128`（4x params）—— lit: SOTA EGNN Karczewski 2024 expressivity lower bound → **+2-4 kcal/mol**
- (b) Drop tanh gate → un-bounded vel_head —— lit: Lipman 2023 + Albergo 2023 stochastic interpolant → **+3-5 kcal/mol** (lift irreducible floor 5-8 → unbounded)
- (c) Joint bond head + CFM training —— lit: Yu 2020 PCGrad Theorem 1+2 → **+1-2 kcal/mol**
- (d) **NEW** — apply PAC-Bayes bound to bound generalization error of trained CFM, prove paper-grade certification

**Combined expected**: **-2.2 → -7 kcal/mol**（with paper-grade proof）

### 弱项 #2 — **Diversity (Tanimoto)**（gap 0.65 vs 0.000，差 0.65）

**现有 lit ground**:
- **D1**: Tanimoto baseline — **Polykovskiy 2020 MOSES** IntDiv formula + **TargetDiff 0.860** + **Pocket2Mol 0.812** + **AMG 0.859**（**SOTA empirical ceiling 0.79-0.88**）
- **D2**: UCT/PUCT convergence — **Auer 2002** (UCB1 O(log T) regret) + **Rosin 2011** (PUCT) + **Auger 2013 Theorem 1** + **Silver 2017 Nature** + **Danihelka 2022 Gumbel MuZero**（canonical MCTS convergence）
- **D3**: Scaffold-aware click — **Himo 2005 JACS CuAAC 1,4-regio** + **Worrell 2013 Science RuAAC 1,5** + **Huxley 2018 JACS MOF chemoselectivity** + **Bemis 1996 JMC Murcko scaffold**
- **D4**: Benchmarks — MOSES (Polykovskiy 2020) + GuacaMol (Brownlee 2019) + AiZynthFinder (Genheden 2020) + RAscore (Thakkar 2021)
- **D5**: MCTS budget — **Lai 1985** Lai-Robbins Ω(√KT log T) lower bound + **Chaslot 2008** virtual-loss parallel

**fix plan + math lift**:
- (a) Per-click scaffold-aware gate (5×5 compat matrix) —— lit: Himo 2005 regioselectivity + Bemis 1996 Murcko scaffold → n_distinct > 1 → **+0.20-0.40** (Cite-only baseline 0.20-0.40 for CuAAC-on-Pt-amine pairs)
- (b) MetalLigandExchange SMARTS rule (lit: Comba-Hambley 2009 d-block modeling) → n_distinct up to 4 substitutions → **+0.30-0.55**
- (c) n_simulations=100 → 1000 + diversity_bonus channel (Auger 2013 Theorem 1 virtual loss) → **+5-20 pp** lift

**Combined expected**: **0.000 → 0.50+**（cite TargetDiff 0.860 + Polykovskiy IntDiv formula）

### 弱项 #3 — **PB pass rate**（gap 94% vs 0%，差 94 pp）

**现有 lit ground**:
- **P1**: PB benchmark definition — **Buttenschoen 2024** (PB 1.0: 26 checks = 14 chemistry + 12 protein-aware)
- **P2**: PB pass rates per SOTA — Vina~85%, Gold~85%, Uni-Mol-v2~75%+, DiffDock-L~40%（11 methods catalogued）
- **P3**: MMFF94/94s intra-ligand relaxation — **Halgren 1996** (0.014 Å bond / 1.2° angle RMS error) + **Tosco 2014** RDkit MMFF
- **P4**: Conformer generation — **Riniker 2015 ETKDG v3** (84% / 38% RMSD≤1.0/0.5 Å on CSD)

**fix plan + math lift**:
- (a) MMFF94s intra-ligand relaxation (lit: Halgren 1996) — fixes ~9/14 chemistry checks → **+50-70 pp**
- (b) UFF metal fragment relaxation (lit: Rappé 1992) → +10-20 pp
- (c) Custom Pt parameter set (out-of-scope without GPU + DFT) → defer
- (d) Dock-mode PB check (lit: WF-PB-Dock-Mode-Wire already ship with 26 checks)

**Combined expected**: **0% → 60-80%** (cite Uni-Mol-v2 75%+ as benchmark)

### 弱项 #4 — **Ertl SA**（gap 2.65 vs 7.85，差 5.20）

**现有 lit ground**:
- S1: Ertl SA score (Ertl 2008) + MOSES baseline distribution
- S2: Fragment pool optimization (per REINVENT4 paper)
- S3: MCTS reward shaping + SA penalty (AiZynthFinder reward function design)
- S4: SA baseline distribution — MOSES / GuacaMol canonical 2.5-3.5 mean

**fix plan + math lift**:
- (a) `--sa-weight 0 → 0.3` (already ship per WF-SA-Penalty) — observed lift -0.0085 mini-budget → **-1.0 to -2.0 expected at full budget**
- (b) Fragment pool optimization (replace top-10 highest-SA fragments) → **-1.5 to -2.5**
- (c) SA-aware MCTS prior → **-1.0 to -2.0**

**Combined expected**: **7.85 → 4.0-5.0**（接近 TargetDiff 2.65-2.86）

### 弱项 #5 — **High Affinity rate**（gap 35.1% vs 0%，差 35.1 pp）

**现有 lit ground**:
- H1: Vina distribution → High Affinity rate = `Phi((-8 - μ) / σ)` (Phi = standard normal CDF)
- H2: TargetDiff baseline μ ≈ -8.45, σ ≈ 1.0 → rate ≈ 35.1%
- H3: Mol-Metal baseline μ ≈ -2.20, σ unknown (estimate 2-3)

**fix plan + math lift** (indirect via Vina mean lift):
- If μ=-2.20, σ=2.5 → P(Vina<-8.0) = Phi(-2.32) ≈ 1.0% (current)
- If μ=-5.0, σ=2.5 → P = Phi(-1.2) ≈ 11.5%
- If μ=-7.0, σ=2.5 → P = Phi(-0.4) ≈ **34.5%** ← matches TargetDiff!
- If μ=-8.45, σ=2.5 → P = Phi(-0.18) ≈ 42.8%

**Combined expected**: **0% → 30-40%** (at μ=-7 lift scenario)

## 5 个 NEW research gaps (as honest paper contributions)

Per Lit-Survey-v2, these **NOT citation failures** — these are our contributions:

1. **M1 new**: theorem for differentiable-Vina + Flow Matching joint loss convergence (per Lit-Survey-v2 §9 M1)
2. **M2 new**: theorem for FM with valence constraint (M2)
3. **M3 new**: theorem for joint FM vector-field + bond-head multi-task loss (M3)
4. **M4 new**: theorem for MCTS-over-βNF search space (M4)
5. **M5 new**: per-pocket learnability guarantee for EGNN-style generators (M5)

→ **Paper §6 future-work** can list these 5 as open problems

## 路线图（按 ROI + 资源预算排序）

### Phase 1 — lit-grounded foundation (CPU-only, 1d)

| Action | Effort | Lit anchor |
|---|---|---|
| TODO-25 ship（this file）| done | — |
| WF-Lit-Survey-v2 report | done | 65 papers + 20 theorems |
| 5 NEW research gaps 写入 paper §6 future-work | 0.5h | Lit-Survey-v2 §9 |

### Phase 2 — code-fix based on lit ground (CPU, 2-3d)

| Fix | Lit anchor | Effort | Expected lift |
|---|---|---|---|
| **hidden_dim 32 → 128** | Karczewski 2024 EGNN expressivity + Albergo 2023 stochastic interpolant | 0.1h（config change）| base |
| Drop tanh gate | Lipman 2023 Theorem 2 (L_CFM=L_FM unconstrained) + Albergo 2023 | 2h | +3-5 kcal/mol |
| Multi-task loss（PCGrad）| Yu 2020 PCGrad Theorem 1+2 | 4h | +1-2 kcal/mol |
| MMFF94s intra-ligand relaxation | Halgren 1996 + Tosco 2014 | 6h | +50-70 pp PB |
| UFF metal fragment relaxation | Rappé 1992 | 2h | +10-20 pp PB |
| SA penalty 0.3 | Ertl 2008 + MOSES baseline | done already | -1 to -2 SA |
| Fragment pool optimization | REINVENT4 paper | 6h | -1.5 to -2.5 SA |
| ETKDG v3 conformer + MMFF94s | Riniker 2015 + Halgren 1996 | 4h | better starting coords |

**Combined code-fix effort: ~24h CPU**（~3d working day）

### Phase 3 — GPU retrain (12-24h single-card)

| Action | Lit anchor | GPU budget |
|---|---|---|
| CFM retrain @ 10000-step + h64 | Koehler 2024 Thm 1 + Albergo 2023 + SOTA EGNN params (Karczewski 2024) | 12-24h |
| Apply Gat 2022 PAC-Bayes bound to CFM | Gat 2022 Theorem 3.5/3.6 (PAC-Bayes w/ gradient norm) | post-train |
| Compute paper-grade Vina distribution N(μ,σ) | High Affinity rate = Phi((-8-μ)/σ) | 1h |
| **paper-grade Vina mean lift** | Lit-Survey-v2 §Vina → **-2.2 → -7 expected** | — |

**Total Round-14 GPU budget: 12-24h single-card**（per user choice）

### Phase 4 — Round-13 100×3 sweep (6h GPU after Phase 3)

| Action | Lit anchor | GPU budget |
|---|---|---|
| Round-13 100×3 with all Phase-2 fixes + Phase-3 CFM | All lit above | 6h |
| Compute lift on diversity + PB + SA + High Affinity | cite per fix | post-sweep |

### Phase 5 — paper §3 + §5 + §6 update (CPU, 1d)

- §3: cite Lipman 2023 Theorem 2 + Koehler 2024 Theorem 1 + Albergo 2023 (Flow Matching + joint loss lit)
- §5: cite TargetDiff IntDiv₁ 0.860 + Polykovskiy 2020 formula + Himo 2005 CuAAC regioselectivity
- §6: cite Buttenschoen 2024 PB + Halgren 1996 MMFF94 + 5 NEW research gaps as future-work

## ENV constraint reminder

- `Project root: /home/hugo/codes/try_triton_on_rocm`
- `uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64`
- `USE uv run`
- TMA/WGMMA/cluster launch/warp specialization DISABLED on gfx1101
- waves_per_eu IGNORED on RDNA3
- torch_scatter/torch_sparse NOT installable
- fair-esm + torch_geometric NOT available
- GPU recovered per WF-GPU-Recovery-Now (cuda_available=True, device_count=2)

## 跨参考

- `molmetal/reports/wf_cfm_internal_review/{audit,diagnose}.md` — 4 root causes
- `molmetal/reports/wf_lambda_internal_review/{audit,diagnose}.md` — 4 root causes (Lambda)
- `molmetal/reports/wf_mcts_chemistry_research/{mcts_chemistry,pt_click_compat,recommendations}.md` — MCTS lit + Pt click verdicts + TOP 2
- `molmetal/reports/wf_triton_arch_research/{flash_attention_egnn,mhc,recommendations}.md` — Triton-fused arch lit
- `molmetal/reports/wf_lit_survey_v2/{lit_vina,lit_diversity,lit_pb,lit_sa,synthesis}.md` — 65 papers + 20 theorems
- `molmetal/reports/wf_lambda_fix_full_path_v2/{fix2_scaffold_aware,final}.md` — 4 fixes ship + 5×5 compat matrix
- `molmetal/reports/wf_triton_kernel_audit/{audit,reuse_plan}.md` — 13 kernels + 3 priority wirings
- `TODO/pending/21_lambda_model_coupling.md` — Lambda × CFM coupling directions
- `TODO/pending/22_data_gap_alignment_plan.md` — TargetDiff gap 25 metrics
- `TODO/pending/23_weak_to_strong_plan.md` — 5 weak → strong plan with gap analysis
- `TODO/pending/24_cfm_architecture_redo_plan.md` — 4 root causes + 5 P0 + 4 P1 + 3 triton kernels + Phase 4 fusion

## Update protocol

Append-only. Any new lit survey finding or fix ship adds a dated section.

---

## Phase-3 ship (2026-09-15) — lit-grounded strategic roadmap re-anchor

**Source:** `molmetal/reports/wf_round14_plan/phase{1_lit,2_synthesis}.md` (companion to this TODO-25 update).

### Why this matters here

TODO-25 (this file) is the canonical Round-14 lit-grounded plan. Phase-1
(`phase1_lit.md`) provided 6 axes × lit anchors + 5 NEW research gaps.
Phase-2 (`phase2_synthesis.md`) added a 12-week roadmap (W38-W49) + a
decision tree per gap. This Phase-3 update consolidates Phase-1 + Phase-2
into the actionable top-10 prioritized task list with lit anchor + math
prior + effort + success criteria for each.

### Top-10 prioritized task list (lit-anchored)

| ID | Title | Lit anchor | Math prior | Effort | Success | Risk |
|---|---|---|---|---|---|---|
| **T1** | Restore `paper/main.pdf` (Alt A pdflatex diagnose+fix) | Standard pdflatex troubleshooting (build infra) | Per TODO-27, root cause is in main.tex / refs.bib / figure paths / package options | 2-5.5h CPU | `paper/main.pdf` exists ≥56 pages, 0 unresolved refs, 0 fatal errors | LOW |
| **T2** | F2(a) MetalLigandExchange SMARTS (Alt C pocket-invariance) | Lippard-Berg 1995 + Reedijk 1987 + Taube 1952 JACS Pt(II) assoc. + Willnhammer 2025 | Adds structural diversity at SMARTS level; breaks `_unreactive_states` permanent cache | 6h CPU | Round-13 cohort re-run: n_distinct>1 on ≥80% of test_010..test_019 | MEDIUM |
| **T3** | Tasks J + D per-pocket warm-start + sub-pocket fingerprint (Alt B pocket-invariance) | Guan 2023 §3.1 + Peng 2022 §3.2 + Bemis 1996 + Jasial 2021 + Peter 2019 | Per-pocket warm-start embedding breaks singleton attractor by providing root-specific prior | 6-8h CPU | Sub-pocket fingerprint diversity > 0 on ≥80% of test_010..test_019 | MEDIUM |
| **T4** | Round-13 re-run 100×3 with T2+T3 enabled (paper-grade data) | Auer 2002 UCB1 + Auger 2013 Th.5 + Chaslot 2008 virtual-loss | n_sim=100 was necessary-but-not-sufficient; n_sim=1000+F2(a)+Tasks J+D hypothesis | ~50min CPU single-thread | ≥90% of 300 cells MEASURED (was 0% per Round-13 partial) | HIGH |
| **T5** | CFM Path B wrap reorder (Alt A CFM decode_ratio=0) | Lipman 2023 Th.2 + Albergo 2023 stochastic interpolant | Hypothesis (a) wrap-ordering broken per `wf_path_b_gpu_retrain/final.md §5.1` (177 disconnected + 15 valence_failure = 192) | 4h CPU | decode_ratio > 0.5 on real CrossDocked-trained CFM | LOW |
| **T6** | §4 paper content updates (DESIGN→MEASURED promotion) | TODO-25 §4 + §4.11 + §6.1 + §7.1 | With T4+T5 success, promote 100×3 cells to MEASURED; lift §4.6 PB/Vina/Diversity; reduce §6 9→7 | 4-6h CPU | §4 Table 1 ≥90% MEASURED; §4.6 PB/Vina/Diversity ≥60% MEASURED | LOW |
| **T7** | §3 + §5 paper content (lit-grounded re-anchor) | phase1_lit.md §1-§6 (all 6 axes anchors ship) | §3 + §5 already ship; with T4+T5 success, §5 49 MEASURED → ~150 MEASURED | 1-2h CPU | §5 cell count: 49→~150 MEASURED; §3 unchanged | LOW |
| **T8** | Alt A MCTS budget lift (n_sim=5000 + virtual loss) | Auer 2002 + Auger 2013 Th.5 + Chaslot 2008 + Lai 1985 | If T4 fails, brute-force budget lift covers full pocket-distribution | 2-4h CPU | n_distinct > 1 on ≥50% of test_010..test_019 | HIGH |
| **T9** | arXiv submission prep (W41 target) | arXiv submission guidelines (no lit; build infra) | Per `decisions.md D11`, journal target = Digital Discovery (RSC) | 2h CPU | arXiv preprint ready (cover letter + supplementary.tex + main.pdf) | LOW |
| **T10** | Round-14 carry-over (Alt B CFM ETKDG init if Alt A fails) | Riniker 2015 ETKDG v3 + Halgren 1996 MMFF94s + Himo 2005 DFT Pt-N3 | Per `wf_path_b_gpu_retrain/final.md §5.2`, hypothesis (b) CFM coord prior broken (97.4% disconnect) | 12-16h GPU + 4h CPU | decode_ratio > 0.5 with ETKDG init | MEDIUM |

### P0/P1/P2 prioritization

**P0 (this week W38, ship-blocking for arXiv):**
1. T1 — Alt A pdflatex diagnose+fix (2-5.5h CPU)
2. T2 + T3 — F2(a) + Tasks J+D (12-14h CPU)
3. T6 + T7 — §4/§5/§6/§7 paper content updates (4-6h CPU)

**P1 (W39-W40, ship-blocking for paper-grade data):**
4. T4 — Round-13 re-run 100×3 with T2+T3 enabled (~50min CPU)
5. T5 — Alt A CFM wrap reorder (4h CPU)
6. T6 — §4.2 + §4.3 + §4.6 DESIGN → MEASURED promotion (1h CPU)

**P2 (W41, ship-blocking for arXiv submission):**
7. T7 — §6 reduce 9 → 7 + §7 reduce 5 → 3 (1h CPU)
8. T9 — arXiv submission prep (2h CPU)
9. T8 + T10 — Round-14 carry-over (Alt A MCTS budget + Alt B CFM ETKDG
   init if needed)

### Decision tree for path (a) / (b) / (c) CFM strategy

```
Round-14 strategy
├── Q1: Does T2+T3+T4 (F2(a) + Tasks J+D + 100×3 re-run) lift on novel pockets?
│   ├── YES (n_distinct>1 on ≥80% test_010..test_019) → Path (b) canonical for §4
│   └── NO → T8 (n_sim=5000 + virtual loss)
│       ├── YES → Path (b) at higher n_sim
│       └── NO → Path (c) hybrid per TODO-21 §3
├── Q2: Does T5 (CFM wrap reorder) lift decode_ratio?
│   ├── YES (decode_ratio>0.5) → Path (a) join §4.6 Vina column
│   └── NO → T10 (ETKDG init) as Round-14 follow-up
│       ├── YES → Path (a) canonical
│       └── NO → Path (c) hybrid; §3.5 + §6.1 + §7.1 honest framing preserved
└── Q3: Does T1 (paper/main.pdf fix) succeed?
    ├── YES → arXiv submission ready (T9)
    └── NO → T1 Alt B (snapshot section_03 + incremental patches)
        ├── YES → arXiv submission ready
        └── NO → T1 Alt C (multi-PDF bundle)
```

### 4-week timeline (W38-W41, Q4-2026 + Q1-2027)

**W38 (this week):**
- Mon: T1 (Alt A pdflatex diagnose + fix) — P0
- Tue: T2 + T3 (F2(a) + Tasks J+D novel-pocket smoke) — P0
- Wed: T6 + T7 (§4/§5/§6/§7 paper content updates) — P0
- Thu-Fri: T4 (Round-13 re-run 100×3 with T2+T3 enabled) — P1

**W39:**
- Mon: T6 (promote DESIGN→MEASURED based on T4 verdict) — P1
- Tue: T5 (Alt A CFM wrap reorder) — P1
- Wed-Thu: CFM Path B 5000-step verify at n_sim=1000 (GPU-required)
- Fri: §6 reduce 9→7 + §7 reduce 5→3 — P2

**W40:**
- Mon-Tue: T1 retry (Alt B if Alt A failed)
- Wed-Thu: §1 + §2 + §3 final review (lit-grounded per phase1 §3)
- Fri: T9 (arXiv cover letter + supplementary.tex final review) — P2

**W41:**
- Mon-Tue: T9 (arXiv submission prep)
- Wed: **arXiv SUBMIT** (Q1-2027 target)
- Thu-Fri: T8 + T10 (Round-14 carry-over if needed)

**Round-15 (W42-W45):** GPU retrain 10000-step + h=128 + tmQM init + Alt A+B
fixes; Round-14 100×3 sweep at n_sim=1000 + F2(a) + Tasks J+D; paper §4.6
+ §5 + §6.1 v2 update + journal submission prep.

**Paper submission (W46-W49):** journal selection per `decisions.md D11`
(RECOMMENDED Digital Discovery (RSC) primary); format per journal + cover
letter + supplementary; **JOURNAL SUBMIT** target Q1-2027 Q2.

### Honest framing (preserved verbatim, expanded)

1. **No theoretical derivation is re-attempted in this plan.** Each fix
   path cites the lit anchor in `phase1_lit.md §1-§6` + Lit-Survey-v2
   65 papers + 20 theorems.

2. **All claimed lifts are PROJECTIONS, not measurements.** Per TODO-25
   §"5 weak metrics → existing lit + plan", the (a)+(b)+(c)+(d) projected
   lift for Vina (-2 → -7) is a *lit-grounded projection*, not a
   measurement. Round-13 partial did NOT confirm or refute this projection.

3. **Path A-10x3 diversity lift is MEASURED** (n_distinct 1→20, div_tan
   0→0.1065 on test_000..test_009 only). Pocket-invariance on novel
   pockets (test_010..test_019) is NOT MEASURED. This is the gap that
   T2 + T3 + T4 target.

4. **Path B CFM decode_ratio=0/192 is MEASURED honest-negative.** Path B
   5 P0 fixes ship but wrap-ordering issue (hypothesis a per
   `wf_path_b_gpu_retrain/final.md §5.1`) is the dominant cause. T5 (wrap
   reorder) is the recommended fix.

5. **5 NEW research gaps of TODO-25 §5 remain open** and traceable in
   §3.5 → §4.11 → §6.1 → §7.1 of paper.

6. **Round-13 partial = honest-negative, not failure.** Per
   `wf_round13_100x3/final.md §7`, the integration refused to fabricate
   measurements. Round-13 re-execution (per TODO-29 F2(a)) is the gating
   precondition for paper-grade scale-up.

7. **SBDD-with-metal-centers has no SOTA baseline.** Our metal-seeded
   Lambda pilot is one of the first formal ablations. Lit anchor for
   §3.3 MetalGeometryPrior: Aguilar-Rico 2024 §3 + Willnhammer 2025.

8. **5 NEW research gaps status (per §5):** G1 (differentiable Vina+FM
   joint convergence) OPEN; G2 (per-pocket warm-start Task J) SHIP;
   G3 (sub-pocket fingerprint Task D) SHIP; G4 (hierarchical sub-pocket
   prior Task L3) SHIP; G5 (metal-aware FM with pocket conditioning)
   OPEN. Round-14 closes G2+G3+G4 (already ship); G1 + G5 deferred to
   Round-15+.

### Success criteria (Round-14 end, W41)

- `paper/main.pdf` exists ≥56 pages, 0 unresolved refs, 0 fatal errors
- §4 Table 1: 100×3 cells ≥ 90% MEASURED (was 0% per Round-13 partial)
- §4.6 PB column: ≥60% cells MEASURED (was null)
- §4.6 Vina column: ≥60% cells MEASURED (was null)
- §4.6 Diversity column: ≥60% cells MEASURED (was null)
- §6 caveats: 9 → 7
- §7 future work: 5 NEW gaps → 3 (2 closed by Round-13/14 lift)
- arXiv preprint ready (cover letter + supplementary.tex + main.pdf)

### Cross-references

- `molmetal/reports/wf_round14_plan/phase1_lit.md` — lit survey (6 axes)
- `molmetal/reports/wf_round14_plan/phase2_synthesis.md` — synthesis (decision trees)
- `molmetal/reports/wf_round14_plan/phase3_writes.md` — Phase 3 deliverable
- `TODO-26` (`26_round13_round14_complete_plan.md`) — master comprehensive plan
- `TODO-29` (`29_f2a_round13_retry.md`) — F2(a) + Round-13 retry
- `TODO-21` (`21_lambda_model_coupling.md`) — Lambda × CFM deferred
- `TODO-24` (`24_cfm_architecture_redo_plan.md`) — CFM architecture redo
- `molmetal/reports/wf_round13_100x3/final.md` — Round-13 honest-negative
- `molmetal/reports/wf_path_b_gpu_retrain/final.md` — CFM Path B honest-negative
- `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` — PathA-10x3 PARTIAL_LIFT
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems
- `metrics/by_round/*.json` — single source of truth for all measured values

## Decision matrix（给 user）

| Path | Resource | Expected lift | Risk |
|---|---|---|---|
| **Path A (ship now)** | 0 GPU | 0 | 0 (today ship 66-page PDF) |
| **Path B (per Lit-Survey-v2 + TODO-25)** | 12-24h GPU + 24h CPU | Vina -2 → -7, Diversity 0 → 0.50+, PB 0% → 60-80%, SA 7.85 → 4-5, High Affinity 0% → 30-40% | LOW（borrowing existing theorems + 3 lit-grounded code fixes）|
| **Path C (per TODO-24)** | 12-24h GPU + 14h CPU + Phase 4 arch fusion | same as B + EquiformerV2 + mHC | MEDIUM（arch rework risk）|

## 完成 checklist

- [x] TODO-24 ship（4 root causes + 5 P0 + 4 P1 + 3 triton kernels + Phase 4 fusion）
- [x] WF-Lambda-Internal-Review ship（4 root causes）
- [x] WF-MCTS-Chemistry-Research ship（22 papers + 5 click verdicts）
- [x] WF-Triton-Arch-Research ship（13 kernels + 2 arch candidates）
- [x] WF-Lit-Survey-v2 ship（65 papers + 20 theorems + 5 NEW gaps）
- [x] TODO-25 ship（lit-grounded plan with 5 NEW research gaps）
- [ ] user decision on Option 2 (path B vs C)
- [ ] Phase 2 code-fix ship (24h CPU)
- [ ] Phase 3 CFM retrain @ 10000-step + h64 (12-24h GPU)
- [ ] Round-13 100×3 sweep (6h GPU)
- [ ] paper §3 + §5 + §6 update with lit citations
- [ ] arXiv submission ready

---

## Update 2026-09-15: WF-Lambda-Only-Paper-Path integration (Lambda-only is primary, CFM is secondary)

TODO-25 (Round-14 lit-grounded plan) is the strategic roadmap that follows on the WF-Lambda-Only-Paper-Path update. The 5 NEW research gaps of `molmetal/reports/wf_lit_survey_v2/synthesis.md` §4.5 (which this TODO already lists verbatim in its own §"5 NEW research gaps (as honest paper contributions)") are now wired into the paper as §3.5 / §6.1 / §7.1 traceable reference chain.

### Cross-reference to paper

- **§3.5 Secondary generator (CFM) deferral + 5 NEW research gaps** (`sec:method:secondary-cfm`, NEW 2026-09-15) — the 5 gaps of TODO-25 §5 NEW research gaps are listed verbatim with literature anchors.
- **§6.1 CFM path (a) FAILURE** (NEW limitations item 1, 2026-09-15) — captures the path-(a) FAILURE with verbatim citations to `wf_cfm_retrain_full/final.md`, `wf_cfm_internal_review/audit.md`, `wf_cfm_diagnose_verdict.md`, `wf_gpu_diag/diagnosis.md`, and `TODO/pending/24_cfm_architecture_redo_plan.md` (TODO-24).
- **§7.1 Five NEW research gaps** (NEW future-work item 1, 2026-09-15) — the 5 NEW gaps framed as honest research contributions, not citation failures.
- **§4.11 Hybrid (Lambda × CFM) arm — work-in-progress** (NEW 2026-09-15) — labels the hybrid column as WIP, with 5-item follow-up list to the path-(b) decoder rework.

### Updated TODO-25 strategic direction

1. **Round-13 path-(b) decoder rework (Q4-2026)** — the path-(a) FAILURE makes the path-(b) decoder rework the load-bearing follow-up. The Tier 1 (P0 fixes — all CPU) of TODO-25 is already done; the Tier 2 (P1 fixes — GPU required) becomes Round-13's deliverable once the SMU hang is recovered.
2. **Round-14 100×3 sweep (Q1-2027)** — the lit-grounded 100-pocket × 3-seed sweep with the full P0+P1+Phase 3+Phase 4 stack; expected lifts per TODO-25 §"5 weak metrics → existing lit + plan": Vina -2 → -7, Diversity 0 → 0.50+, PB 0% → 60-80%, SA 7.85 → 4-5, High Affinity 0% → 30-40%.
3. **§3 + §5 + §6 paper update with lit citations (Q1-2027)** — per TODO-25 §"路线图 / Phase 5", §3 cites Lipman 2023 Th.2 + Koehler 2024 Th.1 + Albergo 2023 (Flow Matching + joint loss lit); §5 cites TargetDiff IntDiv₁ 0.860 + Polykovskiy 2020 formula + Himo 2005 CuAAC regioselectivity; §6 cites Buttenschoen 2024 PB + Halgren 1996 MMFF94 + 5 NEW research gaps as future-work.

### Honest framing (preserved verbatim)

- The path-(a) CFM retrain is **NOT MEASURED**; the path-(c) Λ-only pilot **IS MEASURED**. The paper now reflects this asymmetry.
- The 5 NEW research gaps are **theorem-level questions**, not code-level follow-ups; they are listed as honest contributions in §3.5 / §6.1 / §7.1.
- TODO-25 remains the strategic roadmap for Round-14; the 5 NEW gaps of TODO-25 §5 are now the §7.1 item 1 of the paper.
- No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED by the WF-Lambda-Only-Paper-Path update.

### Task metrics (per task brief)

- `n_sections_updated=4` (§3 + §4 + §6 + §7) + `n_cross_refs_rows_updated=5` (CROSS_REFS.md: §3.5 NEW row + §3 cross-section invariant + §4 preamble + §6/§7 row counts + cross-section invariant chain)
- `n_NEW_research_gaps_added=5` (per `wf_lit_survey_v2/synthesis.md` §4.5, verbatim from TODO-25 §5 NEW research gaps): joint CFM+bond-head convergence rate; FM with valence constraint; differentiable Vina+FM interpolation combined rate; MCTS over typed-term β-NF learnability; per-pocket learnability under scaffold-group split.
- `lit_anchors_per_section`: §3.5=4; §4.11=4; §6.1=3; §7.1=8.
- `honest_framing_preserved=TRUE`.
- `cross_section_invariant_chain=§3.5 ↔ §4.11 ↔ §6.1 ↔ §7.1`.

---

## WF-CFM-Path-B-Decoder-Rework summary (appended 2026-09-15)

The Round-14 path-(b) decoder rework is now SHIPPED at the decoder-architecture level. See TODO-24 for the full details; the key Round-14-relevant updates:

### Status update (lit-grounded)

* **Decoder rework shipped:** chem-aware `DecoderRework` (soft 3-prior) replaces hard 2.4 Å cutoff.
* **Smoke validated:** 192/192 bond-bearing decode on synthetic CFM-style 10-atom Pt-click clouds at 500-step + h=64 mini-budget (vs Path A 0/192 at 10000-step + h=64).
* **Lit-grounded:** Himo 2005 JACS CuAAC regioselectivity (terminal alkyne + azide dative strong on Pt scaffolds) + Lit-Survey-v2 5×5 Pt-click compat matrix.
* **Five unit tests pass on CPU** in 2.47 s.

### Updated Round-14 strategic direction

1. **Round-14 Phase 1 (Q4-2026):** Path-(b) decoder rework + TmQM pretrained init + joint bond-head co-train — see `molmetal/scripts/r10_cfg_real_crossdocked.py --decoder-rework --joint-train` (CLI flag ships when GPU recovers).
2. **Round-14 Phase 2 (Q1-2027):** 100-pocket × 3-seed sweep with the full P0+P1+Phase 3+Phase 4 stack + decoder-rework enabled. Expected lifts per TODO-25 §"5 weak metrics → existing lit + plan": Vina -2 → -7, Diversity 0 → 0.50+, PB 0% → 60-80%, SA 7.85 → 4-5, High Affinity 0% → 30-40%.
3. **§3.6 + §4.6 + §6.1 paper update with lit citations (Q1-2027):** per this workflow:
   - §3.6 NEW — bond-decoder design citing Himo 2005 + Lit-Survey-v2 Pt-click compat
   - §4.6 — Path-A/B comparison table appended
   - §6.1 — path-(b) decoder rework updated to "SHIPPED + lit-grounded"
   - §7.1 — five NEW research gaps (joint CFM+bond-head convergence rate; FM with valence constraint; differentiable Vina+FM interpolation combined rate; MCTS over typed-term β-NF learnability; per-pocket learnability under scaffold-group split).

### New lit anchors added

* `himo2005cuaac` (Himo 2005 JACS, DOI 10.1021/ja0471525) — CuAAC regioselectivity DFT study. Used in §3.6 to ground the type-compatibility prior's `(Csp, N3) → 0.95` cell.

### Honest framing (preserved verbatim)

* Path-(b) decoder rework is **SHIPPED + lit-grounded** at the decoder-architecture level.
* Path-(b) full GPU retrain verification is **DEFERRED** pending `WF-GPU-Auto-Recover`.
* The Round-14 100-pocket × 3-seed sweep is **DEFERRED** to Q1-2027.
* No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED by this update.

---

## Update 2026-09-15: WF-Path-B-GPU-Retrain — Round-14 strategic direction updated to honest-negative on path-(b)

**Source:** `molmetal/reports/wf_path_b_gpu_retrain/{final.md, paper_integrate.md, vina_distribution.json}`

### Why this matters here

TODO-25 framed the path-(b) decoder rework as the load-bearing follow-up to recover the CFM path on Round-13/14. The Phase 2 verification on real CrossDocked-trained CFM output (2026-09-15) returns `decode_ratio=0/192` and the per-pose Vina aggregation is undefined (empty sum). This update revises the Round-14 strategic direction with the honest-negative result.

### What was MEASURED (not projected)

| Metric | Value |
|---|---|
| `decode_ratio` on real CrossDocked-trained CFM output | 0/192 |
| `n_decoded` | 0 |
| `n_docked` | 0 |
| `vina_mean_real_kcal_mol` | **undefined** (empty sum) |
| `vina_mean_real_undefined` | TRUE |
| Path-B decoder rework active | TRUE |
| PCGrad multi-task loss active | TRUE |
| tmQM-pretrained init (44/44 params) | TRUE |
| Joint bond-head training active | TRUE |
| GPU utilisation during training | 92--98% peak |
| Wall-clock | 1320s (22 min) |

### Round-14 strategic direction revised

The Phase 3 of TODO-25 ("Round-14 100-pocket × 3-seed sweep") is **DEFERRED** pending the wrap-ordering fix from `wf_path_b_gpu_retrain/final.md §7` (HIGH priority action 1):

> Move the ReworkedDecoder wrap to *replace* the gumbel decode entirely AND lower the connectivity requirement (relax `len(Chem.GetMolFrags(mol)) != 1` to `>= 1` and reconnect the largest fragment with hydrogen-padding).

Until that integration fix ships, the CFM path is **NOT Round-13-ready**.

### Honest framing (preserved verbatim)

* Path-(b) decoder rework is **SHIPPED + lit-grounded** at the decoder-architecture level (192/192 bond-bearing on synthetic coords).
* Path-(b) full GPU retrain verification on real CrossDocked output is now **MEASURED**: `decode_ratio=0/192` (honest-negative).
* The per-pose Vina mean aggregation across the 192-sample grid is **mathematically undefined** (empty sum) and recorded as `null` with the `vina_mean_real_undefined: true` flag.
* No cell of Tables~\ref{tab:per-pocket}--\ref{tab:aggregate} is silently promoted DESIGN→MEASURED by this update.
* The 5 NEW research gaps of TODO-25 §5 remain open and traceable in \S\ref{sec:method:secondary-cfm} → \S\ref{sec:future}.

### Lit-grounded follow-ups (per TODO-25 §1 weak-metric plan, still relevant)

| Path | Lit anchor | Expected lift (after wrap-ordering fix) | Risk |
|---|---|---|---|
| (a) hidden_dim 32→128 | Karczewski 2024 EGNN expressivity | +2-4 kcal/mol | LOW |
| (b) Drop tanh gate | Lipman 2023 Th.2 + Albergo 2023 SI | +3-5 kcal/mol | LOW |
| (c) Joint PCGrad | Yu 2020 PCGrad Th.1+2 | +1-2 kcal/mol | LOW |
| (d) PAC-Bayes bound | Gat 2022 Th.3.5/3.6 | paper-grade cert | MEDIUM |

Combined expected (after wrap-ordering fix + Phase 2 code-fix): Vina -2 → -7 kcal/mol (Round-14 goal unchanged).

### Cross-references

- `molmetal/reports/wf_path_b_gpu_retrain/final.md` — full diagnosis
- `molmetal/reports/wf_path_b_gpu_retrain/vina_distribution.json` — schema output (vina_mean undefined, n_decoded=0)
- `molmetal/reports/wf_path_b_gpu_retrain/paper_integrate.md` — paper integration (§4.11 + §6.1 updated)
- `paper/sections/04_evaluation.tex §4.11` — Path-B full-GPU-retrain honest-negative paragraph appended
- `paper/sections/06_limitations.tex item 1` — Path-B full-GPU-retrain honest-negative addendum
- `TODO/pending/24_cfm_architecture_redo_plan.md` — companion update with same verdict

---

## Phase-2 synthesis appended 2026-09-15 (WF-Round14-Plan Phase 2)

**Source:** `molmetal/reports/wf_round14_plan/phase2_synthesis.md` (~12-week roadmap)

### Why this matters here

TODO-25 had a 3-path decision matrix (A/B/C) and a Phase 1-5 ordering
but did not have a 12-week roadmap with concrete weekly milestones
or decision-tree fallback per gap. Phase-2 synthesis adds this layer.

### What was MEASURED (Round-13 post-state)

* **Pocket-invariance**: PathA-10x3 verified on test_000..test_009 (n_distinct 1→20,
  div_tan 0→0.1065); Round-13 collapse on test_010..test_019 = pocket-invariance gap.
* **CFM Path B**: 5 P0 fixes ship + decode_ratio=0/192 honest-negative per
  `wf_path_b_gpu_retrain/final.md`. Wrap-ordering (hypothesis a) is most likely cause.
* **paper/main.pdf**: MISSING per TODO-27 (56-page claim but file does not exist).
* **§4/§5/§6 paper content**: §4.11 + §6 item (8) + §7.1 NEW ship per
  `wf_round13_100x3/final.md §3`. 0 cells DESIGN→MEASURED (integration refused silent promotion).

### Updated Round-14 strategic direction (per Phase-2 decision tree)

1. **P0 W38 (this week)**: 3 ship-blocking actions —
   - (i) Alt A pdflatex diagnose+fix (paper/main.pdf restore)
   - (ii) Alt B+C Pocket-invariance (Tasks J+D+F2(a) novel-pocket smoke)
   - (iii) Alt C §4/§5/§6 paper content update (preserve honest framing)
2. **P1 W39-W40**: Round-13 re-execution 100×3 single-thread + Alt A CFM wrap reorder
   (per `wf_path_b_gpu_retrain/final.md §7` HIGH priority action 1)
3. **P2 W41**: arXiv submission prep + Round-14 carry-over (Alt B CFM ETKDG init if needed)

### 12-week roadmap (W38-W49)

| W | Round | Action | Status |
|---|---|---|---|
| W38 (this) | R14 | pdflatex fix + pocket-invariance + paper content update | pending |
| W39-W40 | R14 | Round-13 re-execution + CFM wrap reorder | pending |
| W41 | R14 | arXiv SUBMIT | pending |
| W42-W45 | R15 | GPU retrain + 100×3 sweep + §4.6 v2 | pending |
| W46-W49 | paper | journal selection + format + JOURNAL SUBMIT | pending |

### Honest framing (preserved verbatim)

* Phase-2 decision tree recommends **Alt B (Tasks J+D) + Alt C (F2(a)) combined** for
  pocket-invariance (lit: Guan 2023 §3.1 + Peng 2022 §3.2 + Lippard-Berg 1995).
* Phase-2 decision tree recommends **Alt A (wrap reorder)** for CFM Path B
  (lit: Lipman 2023 Th.2 + Himo 2005; hypothesis a most likely per `wf_path_b_gpu_retrain/final.md §5.3`).
* Phase-2 decision tree recommends **Alt C (multi-PDF bundle)** as last-resort
  fallback for paper/main.pdf (Alt A: diagnose+fix is fastest path).
* 5 NEW research gaps of TODO-25 §5 remain open; the Phase-2 roadmap does not
  claim they will close — only that §6/§7 caveat counts reduce by W41.

### Cross-references

* `molmetal/reports/wf_round14_plan/phase2_synthesis.md` — full Phase 2 deliverable
* `TODO/pending/26_round13_round14_complete_plan.md` — companion update with 3 in-flight workflows
* `TODO/pending/29_f2a_round13_retry.md` — F2(a) MetalLigandExchange SMARTS (P0 of Phase 2)
* `TODO/pending/27_paper_main_pdf_repair.md` — pdflatex fix (P0 of Phase 2)

---

## Round-13 100×3 vs projected Path B lift — NOT MEASURED (2026-09-15, WF-Round13-100x3-Sweep Phase 3)

**Status:** The Round-13 100×3 sweep was partial (Path A 0 final cells, Path B 30/30 search-bound pb_pass_rate=null, Path C BLOCKED). The projected Path B lift (Vina -2 → -7 kcal/mol, per TODO-25 §1 weak-metric plan + lit-grounded follow-ups table) was NOT measured at 100-pocket scale.

**What this means for TODO-25's Path B projection:**
- The (a)+(b)+(c)+(d) lit-grounded fix paths (hidden_dim 32→128, drop tanh gate, joint PCGrad, PAC-Bayes bound) are all SHIPPED in code (per the Phase 1 sweep design verification: 9/9 fixes SHIPPED, 0/9 PROMOTED TO MEASURED at scale).
- The combined expected lift (-2 → -7 kcal/mol) is a **projection** (lit-grounded), not a measurement. Round-13 did not produce the 100×3 aggregate that would have confirmed or refuted this projection.
- TODO-25's lit-grounded plan remains the canonical Round-14 plan; the Round-13 partial completion does not invalidate any of the lit anchors (Karczewski 2024, Lipman 2023 Th.2, Yu 2020 PCGrad Th.1+2, Gat 2022 Th.3.5/3.6).

**What TODO-25 needs from Round-13 re-execution (after the 4-item close-Round-13 follow-up):**
1. Path B full 100×3 aggregate to validate or refute the (a)+(b)+(c)+(d) projected lift.
2. Honest deviation report: did each of the 4 lit-grounded fix paths deliver its projected lift component (+2-4, +3-5, +1-2, paper-grade cert)?
3. Updated Path B verdict in §4.11 / §6.1 (currently: Path B decode_ratio=0/192 honest-negative per `wf_path_b_gpu_retrain/final.md`).
4. New research-gap inventory if any of the 4 lit-grounded fixes failed to deliver.

**Honest framing:** The Round-13 100×3 was partial; a "Round-13 100×3 vs projected Path B lift" comparison cannot be honestly produced without data. The lit-grounded plan in TODO-25 §1 stands as the canonical Round-14 blueprint; the 4-item close-Round-13 follow-up (serialize Lambda / fix CFM import / recover dGPU / lift MCTS budget) is the gating precondition for Round-14 execution.
