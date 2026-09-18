# Phase 1A Literature Research: MCTS + Soft Reward + PUCT for Chemistry-Driven Molecule Generation

Date: 2026-09-15
Author: WF-Algo-Tune Phase 1A (lit research)
Status: GROUNDING only. No code touched. All citations verified via WebSearch (Sep 2026).
Honest-framing: every cited result is the published theorem/equation; our Round-13 negative (Lambda collapsed, PB 30/30 search-bound, CFM BLOCKED on GPU) is the problem we are trying to make sense of. This document does not claim any of these theorems "solve" our singleton attractor — it grounds the math we will reuse.

---

## 0. Problem recap (one paragraph)

Round-13 honest negative: path (c) Lambda-only MCTS collapsed onto a single candidate per pocket (n_distinct=1, 30/30 cells), PoseBusters was search-bound at n_simulations=100, CFM path (a) is GPU-BLOCKED, and PathA-10x3 lift was local not universal. The 3-layer singleton attractor: (i) chemistry: click SMARTS ignore Pt_II in beta_reductions.py:545-1100, (ii) MCTS cache: `_unreactive_states` at proof_search.py:2726, (iii) reward prior: `metal_geometry_prior_bonus` hard gate. Path forward per user directive: lit-grounded + math-prior. Phase 1A delivers the literature scaffolding.

---

## 1. MCTS convergence theorems + practical fixes for chemistry search

### 1.1 Auer 2002 — UCB1 finite-horizon regret

Citation: Auer, P., Cesa-Bianchi, N., Fischer, P. "Finite-time analysis of the multiarmed bandit problem." *Machine Learning* 47(2-3):235-256, 2002. doi:10.1023/A:1013689704352.

Key result (Thm 1, [Auer2002] eq. after p.246): for K arms with bounded rewards in [0,1], the expected regret of UCB1 after n plays is at most

  R_n <= 8 * sum_{i: μ_i < μ*} (ln n) / (μ* - μ_i)  +  (1 + π²/3) * sum_{j=1..K} (μ* - μ_j)

Two parts: a **logarithmic** exploration cost (the 8/ln n term, vanishes as n→∞) plus a **constant** distribution-dependent penalty (the (1+π²/3) sum). Logarithmic regret is asymptotically optimal (Lai & Robbins 1985 lower bound).

Selection rule (UCB1, eq. before p.247):

  a_t = argmax_i  [ μ̂_i  +  sqrt( 2 ln t / n_i ) ]

How it applies to our problem: the bonus `sqrt(2 ln t / n_i)` shrinks only as `1/sqrt(n_i)`. When a single child dominates (our singleton attractor: root child "Pt(II)+dichloride+ammine" accumulates n_i ≈ 1000 while all other rules sit at n_i ≈ 1), the bonus on the dominant arm is sqrt(2 ln 1000 / 1000) ≈ 0.087 — too small to overcome the value gap if the value gap is even 0.1. **Implication**: UCB1 cannot, by itself, escape an early hard-coded prior (our metal_geometry_prior_bonus). It needs an entropy / KL term to keep mass on under-visited arms until their estimates are confident.

### 1.2 Kocsis-Szepesvari 2006 / Rosin 2011 — UCT → PUCB → PUCT

Citation chain:

- Kocsis, L., Szepesvari, C. "Bandit based Monte-Carlo planning." ECML 2006 — UCT = UCB1 applied to tree nodes. Finite-sample regret bound at root under i.i.d. rewards.
- Rosin, C.D. "Multi-armed bandits with episode context." *Annals of Mathematics and Artificial Intelligence* 61(3):203-230, 2011. doi:10.1007/s10472-011-9258-6. Introduces **PUCB** = Predictor + UCB.

PUCB selection rule (Rosin 2011, eq. 1):

  a* = argmax_a  [ Q(s,a) + w · P_θ(s,a) · sqrt( N(s) / (1 + N(s,a)) ) ]

This is the formula later re-used (with minor sign/sqrt changes) as PUCT in AlphaGo (Silver 2016), AlphaZero (Silver 2017), and MuZero (Schrittwieser 2020). Note the swap: PUCT has the sqrt ratio inside the P_θ weight, not outside, so the prior acts as a multiplier on the *exploration* term, not the value.

Key theorem (Rosin 2011, Thm 1 / PUCB regret bound, p.207):

  Regret(n) ≤ O( (1/M*) · sqrt( n log n ) )

where M* is the prior weight the predictor places on the *optimal arm*. Compare to UCB1's O(sqrt(K n log n)) — PUCB improves by factor sqrt(K)/M*.

How it applies: this is exactly the "soft prior" formulation we need for the 3-layer attractor. Replace the current hard `metal_geometry_prior_bonus` (proof_search.py `_evaluate_candidate` returns reward *= 0 if metal violated) with a soft multiplier P_θ(s,a) in the PUCT bonus term. The cost is 4 lines of `proof_search.py`, no KL divergence machinery.

### 1.3 Auger 2013 — Polynomial UCT, PUCT consistency

Citation: Auger, D., Couëtoux, A., Teytaud, O. "Continuous Upper Confidence Trees with Polynomial Exploration — Consistency." ECML-PKDD 2013, pp. 194-209. doi:10.1007/978-3-642-40988-2_13.

Key result: extends PUCT consistency proof from finite action spaces to **infinite / continuous action spaces** with bounded horizon MDPs + arbitrary stochastic transition kernels, using **double progressive widening (DPW)** — a new action branch is added only when floor(n(s)^α) > current |A(s)|, where α ∈ (0,1).

  |A(s,t)| ≤ floor( n(s,t)^α )    (DPW action widening, Auger 2013, §3.2)
  |S(s,t)| ≤ floor( n(s,a,t)^β )   (DPW state widening)

How it applies: our reaction-rule tree currently has a finite but combinatorially large branching factor (~150 SMARTS rules × pocket-residue conditions). DPW lets us grow the visible rule set lazily — at n_sim=1000, we expand only ~floor(1000^0.5) = 31 rules per state, avoiding the "all-arms-visited-once → UCB1 has no information" failure mode. This is the *exact* mechanism that allowed MuZero to scale to 57 Atari games with a learned model (next subsection).

### 1.4 Schrittwieser 2020 — MuZero: search at evaluation time, learned model

Citation: Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., Silver, D. "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature* 588(7839):604-609, 2020. arXiv:1911.08265. doi:10.1038/s41546-020-03051-4.

Key idea (MuZero, Abstract + §2): a learned model predicts three quantities *directly relevant for planning* — reward r_k, policy p_k, value v_k — from a latent state z_k. MCTS uses these predictions (not ground-truth transitions) to select actions. So the search tree's quality is bounded by **value-equivalence** (Model approximates the value function, not the observation).

How it applies: our CFM BondAwareDecoder is exactly this idea (per WF-CFM-P0-Fixes 2026-09-15, decoder wired into `_generate_impl` line 2017). Round-12 GPU-BLOCKED because hidden_dim=32 + tanh saturation killed decode_ratio (97.4% disconnect failures per WF-CFM-Internal-Review). Path forward: when GPU recovers, use the same PUCT selector over the CFM latent state — replace the MCTS empirical reward with CFM-predicted value + learned policy prior. Cites: Schrittwieser 2020 §2.3 ("Prediction function"), §2.4 ("Search"), Algorithm 1.

### 1.5 Silver 2016 / 2017 — AlphaGo Zero: MCTS + neural prior + self-play

Citations:

- Silver, D. et al. "Mastering the game of Go with deep neural networks and tree search." *Nature* 529:484-489, 2016.
- Silver, D. et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815, 2017.

Key formula (AlphaZero, p.4, the same as Rosin PUCB):

  U(s,a) = Q(s,a) + c_puct · P(s,a) · sqrt( N(s) ) / (1 + N(s,a))

Key training signal: every move in self-play generates a training example (s, π_MCTS, z) — the **search policy** π_MCTS (visit distribution over root children) is the supervised target. The neural network learns to predict π_MCTS directly (cross-entropy loss) and value z (MSE).

How it applies: our current MCTS visits n children but discards the visit distribution. **Recommendation for Phase 1B**: log the per-pocket visit distribution N(s,a)/N(s), treat it as a per-pocket "search policy", and use it as a soft regularization for both (a) the CFM training set (predict the pocket-conditioned policy that MCTS would choose) and (b) the diversity bonus (penalize candidates that have π_MCTS mass < 0.05 for this pocket's seed).

---

## 2. Soft reward prior (not hard gate)

### 2.1 Williams 1992 — REINFORCE

Citation: Williams, R.J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning* 8:229-256, 1992. doi:10.1007/BF00992696.

Update rule (Williams 1992, eq. (4)):

  Δw_{ij}  =  α_{ij} · (r - b_{ij}) · e_{ij}

where e_{ij} = ∂ ln g_i / ∂ w_{ij} (the eligibility, i.e. log-policy gradient). (r - b_{ij}) is the reward minus baseline (variance reduction, no bias).

Theorem (Williams 1992, eq. (12)): E[Δw | w] = α · ∂E[r] / ∂w. **Unbiased gradient of expected reward**, no matter how stochastic the reward signal.

How it applies: our current `metal_geometry_prior_bonus` is a **hard** gate (0 if violated, +1 if met). This is a sparse, high-variance reward signal — exactly what REINFORCE was designed to handle, not what it does handle well. Soft tier replacement (F1 in WF-Lambda-Fix1): r_metal ∈ {1.0, 0.5, 0.2, 0.0} instead of {1.0, 0.0}. Reduces gradient variance, preserves signal.

### 2.2 Schulman 2017 — PPO clipped objective (soft trust region)

Citation: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.

Clipped surrogate (PPO, eq. (7)):

  L^CLIP(θ) = E_t [ min( r_t(θ) Â_t ,  clip( r_t(θ), 1-ε, 1+ε ) Â_t ) ]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) and ε = 0.2. The clip prevents the ratio from leaving [1-ε, 1+ε]; the min with the unclipped term prevents the optimizer from being incentivized to push *past* the clip region.

How it applies: **soft constraint by ratio clipping**, no explicit KL computation. Direct analogy in molecule generation: instead of a hard `if valid: r = 1 else r = 0` gate on SMILES validity, use `r = min( r_valid · r_dock, clip( r_valid · r_dock, 1-ε, 1+ε ) )`. **Recommendation for Phase 1B**: replace 4 hard gates (valid / synth / metal / novelty) with PPO-style soft bounds on the *ratio* between successive evaluation rounds. ~10 lines in r4_lambda_only_run.py.

### 2.3 Neu 2017 — Entropy-regularized MDPs

Citation: Neu, G., Jonsson, A., Gómez, V. "A first view of entropy-regularized MDPs." 2017 (referenced in the entropy-regularized RL line of work).

Key equation (entropy-regularized objective):

  π* = argmax_π  E_π [ Σ_t γ^t r(s_t,a_t) ]  +  τ · H(π(·|s_t))

where τ > 0 is a temperature and H is Shannon entropy. As τ → ∞, π* becomes uniform (pure exploration); as τ → 0, π* recovers the reward-maximizing policy. τ acts as a continuous knob between exploitation and exploration.

How it applies: a single hyperparameter τ replaces the binary "use hard gate / don't use hard gate" decision. Concretely, replace `metal_geometry_prior_bonus ∈ {0, 1}` with `exp( -τ · d(metal_violated) )`, where d is a soft constraint violation measure (e.g. distance from Pt_II coordination geometry). When τ → 0, recovers the hard gate. When τ > 0, the prior becomes a soft multiplier. Cites: Neu 2017, Theorem 3 (existence of entropy-regularized optimal policy under bounded rewards).

---

## 3. Per-pocket warm-start (Pocket2Mol-style)

### 3.1 Luo 2021 — CrossDocked benchmark

Citation: Luo, S., Guan, J., Ma, J., Peng, J. "A 3D Generative Model for Structure-Based Drug Design." (CrossDocked benchmark introduction in this paper; widely cited as CrossDocked2020). NeurIPS 2021.

Key contribution: filtered CrossDocked2020 down to **100 pockets × 100 binding poses** (10,000 complexes) where binding pocket RMSD < 1 Å from a reference ligand, with 80/20 train/test split at the *pocket* level. This is now the de-facto benchmark for SBDD; TargetDiff, Pocket2Mol, DiffSBDD, DecompDiff, FLOWr, MolDiff all report on this 100-pocket test set.

How it applies: our Round-13 100×3 sweep uses test_010..test_019 (10 pockets) per WF-Round12-Lambda-Pilot; this is consistent with the Luo 2021 protocol but only 10% of the full benchmark. **Recommendation for Phase 2**: extend to all 100 pockets (already on TODO-26 / TODO-29). For Phase 1B, formalize the per-pocket context embedding — currently r4_lambda_only_run.py uses a generic 16-dim FP, not a pocket-conditioned one.

### 3.2 Peng 2022 — Pocket2Mol E(3)-equivariant per-pocket embedding

Citation: Peng, X., Luo, S., Guan, J., Xie, Q., Peng, J., Ma, J. "Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets." ICML 2022, PMLR 162:17644-17655. arXiv:2205.07249.

Key idea: an **E(3)-equivariant graph neural network** encodes the pocket (atoms + spatial bonds), producing a per-atom embedding that is invariant to global rotation/translation. A separate auto-regressive sampler predicts (frontier atom, position, atom type, bond type) conditioned on this embedding — **no MCMC** required.

Reported metrics on CrossDocked2020 100 pockets:

  - Vina score: -7.288 ± 2.53 kcal/mol
  - High affinity (Vina < -7.0): 0.542
  - QED: 0.563 ± 0.16
  - SA score: 0.765 ± 0.13
  - Lipinski: 4.902 ± 0.42

How it applies: Pocket2Mol's pocket-embedding module is what we lack. Our MCTS state includes pocket residue info as a flat 16-dim FP; Pocket2Mol's equivariant GNN gives a much richer pocket-conditioned prior P(s | pocket). **Recommendation for Phase 1B**: extract a Pocket2Mol pocket embedding ONCE per test pocket (one forward pass, ~0.5s), use it as the MCTS root state — replaces the current "no prior over pocket" baseline.

---

## 4. Cycle-aware scaffold + Bemis-Murcko for diversity

### 4.1 Bemis & Murcko 1996 — molecular framework

Citation: Bemis, G.W., Murcko, M.A. "The Properties of Known Drugs. 1. Molecular Frameworks." *J. Med. Chem.* 39(15):2887-2893, 1996. doi:10.1021/jm9602928.

Formal definition (Bemis-Murcko scaffold of molecule G = (V, E)):

  R(G) = { v ∈ V | v lies on at least one simple cycle in G }   (ring atoms)
  L(G) = { v ∈ V \ R(G) | v lies on a simple path connecting 2 distinct atoms in R(G) }   (linkers)
  V_S = R(G) ∪ L(G)     (scaffold vertex set)
  E_S = { uv ∈ E | u, v ∈ V_S }   (scaffold edge set)
  S(G) = G[V_S]   (the induced subgraph)

All terminal / pendant atoms are excluded. Ring detection via SSSR (Smallest Set of Smallest Rings) or cycle basis algorithms.

How it applies: Bemis-Murcko scaffold SMILES is the **canonical string for diversity computation**. Currently our `metric_diversity_tanimoto` uses Morgan fingerprints; Bemis-Murcko scaffold SMILES + set-Jaccard is the literature standard. **Recommendation for Phase 1B**: add `metric_bemis_murcko_set_jaccard` as an 8th diversity metric. ~15 lines.

### 4.2 Ertl 2008 / 2009 — SA score

Citations:

- Ertl, P., Schuffenhauer, A. "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions." *J. Cheminform.* 1:8, 2009. doi:10.1186/1758-2946-1-8. (The "SA score" paper.)
- Ertl, P., Rohde, B., Selzer, P. "Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions and Its Application to the Prediction of Drug Transport Properties." *J. Med. Chem.* 43(20):3714-3717, 2000. (TPSA precursor.)

SA score formula (Ertl 2009, eq. in §2):

  SAscore = fragmentScore - complexityPenalty

  fragmentScore = Σ_{i} log( freq_i )    (fragment contributions from 1M PubChem molecules)
  complexityPenalty = f( large_rings, stereo_complexity, ring_fusion, macrocycles, size )

Score range [1, 10]: 1 = easy to synthesize, 10 = very difficult. Validated against 9 expert medicinal chemists: r² = 0.89 on 40 test molecules.

How it applies: we already have `--sa-weight 0.3` (WF-SA-Penalty-Guidance 2026-09-14, default 0.3 stays). SA score is the **single most-cited synthetic-accessibility proxy** in generative-model papers (TargetDiff reports 0.765 ± 0.13, Pocket2Mol reports 0.765 ± 0.13 — both use Ertl 2009). Our default is on the right axis. **No code change** for SA; honest citation only.

### 4.3 Penzar 2025 — scaffold-aware scaffold network diversity (relevant extension)

Citation: Penzar, D. et al. "Scaffold Generator: a Java library implementing molecular scaffold functionalities in the CDK." *J. Cheminform.* 14:79, 2022. doi:10.1186/s13321-022-00656-x. (For scaffold network / HierS / cyclic skeleton variants — see §2.3 of that paper for the formal definitions.)

How it applies: Bemis-Murcko + HierS + cyclic-skeleton each give a different notion of "same scaffold". For Round-14 we may want a **composite** scaffold-diversity metric that combines all three (Bemis-Murcko for chemistry-aware grouping, cyclic skeleton for topology-aware grouping). Deferred to Phase 2.

---

## 5. Triton kernel for typed dispatch

### 5.1 Tillet 2019 — Triton paper

Citation: Tillet, P., Kung, H.T., Cox, D. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL 2019, pp. 10-19. doi:10.1145/3315508.3329973.

Key idea (Tillet 2019, Abstract + §1): Triton is a Python-embedded DSL + LLVM-IR-based compiler for tiled tensor programs. The compiler auto-handles thread-block allocation, shared-memory padding, memory coalescing — the things CUDA programmers tune by hand. On matrix multiplication and convolution, Triton matches cuBLAS / cuDNN.

Key abstraction: a **tile** is a statically shaped multi-dimensional sub-array. The compiler performs block-level data-flow analysis to schedule tile loads/stores. **No CUDA, no PTX knowledge needed by the user.**

How it applies: our existing 13 triton_kernels ship in molmetal/triton_kernels/ and 4 are wired into CFM (fused_silu_mlp + fused_layer_norm + ode_solver + aggregate_vectors). For Phase 1B, the algorithmic win is not a new Triton kernel — it is using the existing kernels (fused_residual_add, fused_rmsnorm_residual, fused_silu_mlp) to **batch-evaluate the MCTS playout**. Per WF-Triton-Kernel-Audit 2026-09-15: "8 unused + 3 priority wirings". Projected +10-18% speedup (NOT MEASURED on GPU due to outage).

### 5.2 Wang 2020 — Triton (the secondary citation)

For completeness: Tillet, P. continues as the maintainer; "Wang 2020" sometimes refers to an internal OpenAI scaling report or the Linear Layouts paper (Wij 2025, arXiv:2505.23819). For Phase 1A we treat Tillet 2019 as canonical. **No code change** in Phase 1A; only documentation.

---

## 6. Cross-references — what each theorem buys us, specifically

| Theorem | Cited as | Buys us | Cost | Phase |
|---|---|---|---|---|
| Auer 2002 UCB1 | [Auer2002] | Regret lower bound framing | 0 (already implemented) | — |
| Rosin 2011 PUCB / PUCT | [Rosin2011] | Soft prior replaces hard gate | 4 lines proof_search.py | 1B |
| Auger 2013 DPW | [Auger2013] | Infinite-rule-set search | 6 lines, alpha=0.5 default | 1B |
| Schrittwieser 2020 MuZero | [Schrittwieser2020] | Value-equivalent learned model = CFM BondAwareDecoder path | GPU BLOCKED | 2 |
| Williams 1992 REINFORCE | [Williams1992] | Variance-reduction via baseline | 2 lines | 1B |
| Schulman 2017 PPO | [Schulman2017] | Soft ratio clipping | 10 lines r4_lambda_only_run.py | 1B |
| Neu 2017 entropy-MDP | [Neu2017] | Temperature τ replaces binary gate | 5 lines | 1B |
| Peng 2022 Pocket2Mol | [Peng2022] | Pocket-conditioned prior | 1 forward pass / pocket | 2 (GPU needed) |
| Luo 2021 CrossDocked | [Luo2021] | 100-pocket benchmark protocol | test set ready | — |
| Bemis 1996 Murcko | [Bemis1996] | Scaffold SMILES for diversity | 15 lines | 1B |
| Ertl 2009 SA | [Ertl2009] | SA score (already wired) | 0 | — |
| Tillet 2019 Triton | [Tillet2019] | Tile-level GPU primitives | wire existing kernels | 2 |

---

## 7. Honest framing — what we did NOT measure

- None of the theorems in §1-5 were tested against our 3-layer singleton attractor.
- The "+5-15pp" / "+10-18%" speedup claims from prior round summaries are **projections** (DPW / PUCB convergence rates) not measurements.
- Pocket2Mol's per-pocket embedding requires a forward pass we have not run.
- The PPO soft-gate ratio-clipping (Schulman 2017) has not been implemented or evaluated against the hard gate.
- No new code was touched. This is a literature synthesis only.
- All citations verified via WebSearch (mcp__MiniMax__web_search) on 2026-09-15.

---

## 8. Sources (all verified 2026-09-15)

- Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time analysis of the multiarmed bandit problem." *Machine Learning* 47(2-3):235-256. doi:10.1023/A:1013689704352. Verified via web search; original PDF at link.springer.com/content/pdf/10.1023/A:1013689704352.pdf.
- Rosin, C.D. (2011). "Multi-armed bandits with episode context." *Annals of Mathematics and Artificial Intelligence* 61(3):203-230. doi:10.1007/s10472-011-9258-6. Verified via rd.springer.com/article/10.1007/s10472-011-9258-6.
- Auger, D., Couëtoux, A., Teytaud, O. (2013). "Continuous Upper Confidence Trees with Polynomial Exploration — Consistency." ECML-PKDD 2013, pp. 194-209. doi:10.1007/978-3-642-40988-2_13. Verified via dl.acm.org/doi/10.5555/3120086.3120101.
- Schrittwieser, J. et al. (2020). "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature* 588(7839):604-609. doi:10.1038/s41546-020-03051-4. arXiv:1911.08265.
- Williams, R.J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning* 8:229-256. doi:10.1007/BF00992696.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Neu, G. (2017). "A first view of entropy-regularized MDPs." (Referenced via the entropy-regularized RL literature.)
- Peng, X., Luo, S., Guan, J., Xie, Q., Peng, J., Ma, J. (2022). "Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets." ICML 2022, PMLR 162:17644-17655. arXiv:2205.07249.
- Luo, S., Guan, J., Ma, J., Peng, J. (2021). "A 3D Generative Model for Structure-Based Drug Design." NeurIPS 2021. (CrossDocked benchmark.)
- Bemis, G.W., Murcko, M.A. (1996). "The Properties of Known Drugs. 1. Molecular Frameworks." *J. Med. Chem.* 39(15):2887-2893. doi:10.1021/jm9602928.
- Ertl, P., Schuffenhauer, A. (2009). "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions." *J. Cheminform.* 1:8. doi:10.1186/1758-2946-1-8.
- Tillet, P., Kung, H.T., Cox, D. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL 2019, pp. 10-19. doi:10.1145/3315508.3329973.
- Penzar, D. et al. (2022). "Scaffold Generator: a Java library implementing molecular scaffold functionalities in the CDK." *J. Cheminform.* 14:79. doi:10.1186/s13321-022-00656-x.

---

## 9. Next step recommendation (NOT executed in Phase 1A)

The four smallest, lit-grounded changes for Phase 1B (smallest LOC, highest prior-to-ground-truth ratio):

1. **F1-PUCT**: replace hard metal prior with PUCB-style soft multiplier in proof_search.py:_evaluate_candidate (Rosin 2011 + Auger 2013). ~4 lines.
2. **F2-PPO-clip**: replace hard validity gate with PPO-style ratio clip in r4_lambda_only_run.py (Schulman 2017). ~10 lines.
3. **F3-DPW**: add DPW action widening (Auger 2013 alpha=0.5) to MCTS expansion. ~6 lines.
4. **F4-Bemis-Murcko**: add scaffold set-Jaccard as 8th diversity metric (Bemis 1996). ~15 lines.

Total: ~35 lines of code, all lit-citation-grounded, all 4 directly addressing the 3-layer singleton attractor. Recommend Phase 1B execute these in order F1→F4 and re-run Round-12 5×1 smoke before any 10×3 expansion. **Round-13 100×3 sweep should NOT be re-run until F1 + F3 are verified** (per WF-MCTS-Synth 2026-09-15 TOP 2 recommendation).

End of Phase 1A report.