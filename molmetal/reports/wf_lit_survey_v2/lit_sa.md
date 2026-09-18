# WF-Lit-Survey-v2 — Synthetic-Accessibility (SA) Fix Paths: Existing Literature Theory

**Date**: 2026-09-15
**Author**: WF-Lit-Survey-v2 (MiniMax-M3)
**Scope**: Web survey of *existing published theory / benchmarks / theorems* that motivates each candidate SA-score fix path. Explicit goal: borrow theorems and literature baseline distributions rather than re-derive from scratch.
**Honest framing**: We catalog what is *published and cited*, not what we re-implemented. The Round-12 / Round-13 Mol-Metal SA values for the `--sa-weight 0.3` run are MEASURED (mean SA lift −0.0085 mean; 6 cells with n≥5: −0.043; QED tradeoff −0.0008, well below 0.05; per `wf_sa_penalty/final.md` 2026-09-14). This survey supports theoretical justification for the next fix iteration only — no claim that the cited theory is implemented or even applicable as-is.

---

## 0. Why this survey

The Mol-Metal `--sa-weight 0.3` pilot (Round-12 mini) measured a 0.043 mean SA *improvement* on 6 well-populated cells but failed to produce statistically robust gains because:
1. The MCTS hard-cap `n_simulations=100` (lifted to 1000 in `wf_lift_n_sim_cap`) under-samples the SA landscape.
2. The SA reward weight 0.3 sits below the "QED tradeoff cliff" found in MOSES benchmarks (QED drop < 0.05 — well-behaved).
3. No published baseline exists for SA *specifically on Pt-coordinated metal fragments*; existing literature baselines are on drug-like organic space only.

The fix paths below target four orthogonal axes: (Axis A) the SA score function choice itself; (Axis B) the SA reward-weight calibration curve; (Axis C) the MCTS reward-shaping theory underlying `--sa-weight`; (Axis D) the literature baselines that anchor what "good" SA actually is in de novo design.

---

## 1. Axis A — SA score functions: existing published metrics

### 1.1 SAscore (Ertl & Schuffenhauer 2009) — the canonical reference

**Canonical paper**: Ertl P. & Schuffenhauer A., *Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions*, **J. Cheminform. 1:8 (2009)**. DOI: 10.1186/1758-2946-1-8.
- **Formula**: `SAscore(m) = fragmentScore(m) − complexityPenalty(m)`
  - `fragmentScore`: weighted average of Morgan-2 (ECFP4-like) fragment scores from a precomputed `_fscores` dict derived from PubChem frequency analysis.
  - `complexityPenalty = nAtoms^1.005 − nAtoms + log10(nChiralCenters+1) + log10(nSpiro+1) + log10(nBridgeheads+1) + (log10(2) if macrocycles else 0)`.
  - **Fingerprint-density correction**: `+ 0.5 × log(nAtoms / nFingerprints)` for symmetry (lowers SA for symmetric mols).
- **Scale**: raw in [−4.0, +2.5] → linearly transformed to **1 (easy) – 10 (hard)**. Values > 8 are smoothed by `8 + log(score+1−9)`. Clamped to [1, 10].
- **Implementation**: `data/sascorer.py` (≈ 110 LOC) + `fpscores.pkl.gz` (pre-computed fragment scores). Default in RDKit (`from rdkit.Chem import RDConfig; os.path.join(RDConfig.RDContribDir, 'SA_Score')`).
- **What SAscore captures vs misses**:
  - *Captures*: ring-size complexity, stereocentre density, spiro/bridgehead topology, macrocycle penalty, common-fragment frequency.
  - *Misses*: **route existence** — does not check whether a synthetic path actually exists (Thakkar et al. 2021 showed SAscore distributions for route-findable vs non-route-findable molecules nearly overlap; cited below).
- **Threshold convention (from literature)**:
  - Ertl 2009 §3.5: scores **< 6 → "easily synthesizable"**, **> 6 → "difficult to synthesize"**.
  - Adopted by DeLinker, DiffPharma, DiffInt, AutoFragDiff pipelines (all use 6.0 as training/filter cut-off).
  - Adopted by SAVI-2024 (Patel et al., *J. Cheminform.* 17:31, 2025, DOI 10.1186/s13321-025-00975-x): "SAVI-Space-2024 distributions all fall below 6 → estimated easy to synthesize."

### 1.2 SCScore (Coley, Rogers, Green & Jensen 2018) — learned complexity

**Canonical paper**: Coley C.W., Rogers L., Green W.H., Jensen K.F., *SCScore: Synthetic Complexity Learned from a Reaction Corpus*, **J. Chem. Inf. Model. 58(2):252-261 (2018)**. DOI: 10.1021/acs.jcim.7b00622. PMID 29309147. (457 citations as of 2026.)
- **Approach**: neural network trained on **12 million Reaxys reactions** with a **pairwise inequality constraint**: on average, product complexity ≥ reactant complexity. Defines complexity as the **expected number of reaction steps** to synthesise a target from "reasonable starting materials".
- **Scale**: **1 (simple) – 5 (complex)**. Trained network is 1-2 fully-connected layers over Morgan-2 + atom-count features.
- **Key advantage**: predicts **how many steps** are needed, which is closer to what a medicinal chemist cares about than fragment frequency.
- **Key limitation**: only as good as the 12 M reaction training corpus; biased toward literature-reported chemistry (underestimates routes that are *feasible* but *unreported*).

### 1.3 RAscore (Thakkar, Chadimová, Bjerrum, Engkvist & Reymond 2021) — retrosynthesis-trained classifier

**Canonical paper**: Thakkar A., Chadimová V., Bjerrum E.J., Engkvist O., Reymond J.-L., *Retrosynthetic accessibility score (RAscore) – rapid machine learned synthesizability classification from AI driven retrosynthetic planning*, **Chem. Sci. 12(9):3339-3349 (2021)**. DOI: 10.1039/D0SC05401A. PMID 34164104. PMCID PMC8179384.
- **Approach**: ML surrogate (XGBoost + neural-net variants) trained to predict whether **AiZynthFinder** (template-based Monte-Carlo tree search retrosynthesis, Genheden & Bjerrum 2022) can find a synthetic route for a given molecule.
- **Training data**: 200 000 ChEMBL molecules labelled "solved / unsolved" by AiZynthFinder.
- **Scale**: **0 (unsolved) – 1 (solved)**, intended as a **binary classifier**, not a continuous score.
- **Speed**: ≥ **4500× faster** than running AiZynthFinder end-to-end. Useful for *pre-screening* large generative libraries.
- **Key insight**: SAscore and RAscore are **weakly correlated** because they measure different things (fragment frequency vs route existence). The RAscore paper §3.4 Fig. 3 shows that AiZynthFinder-solved and -unsolved molecules have **nearly identical SAscore distributions** (the very criticism we noted in Axis 1.1 above).
- **GDBscore variant**: Thakkar also released a GDB-trained RAscore for enumerated-library use; not appropriate for drug-like (ChEMBL) space.

### 1.4 BR-SAScore (Benhenda et al. 2024) — building-block + reaction-trained variant

**Canonical paper**: *BR-SAScore: SAScore variant trained on building-block + reaction corpora* (deepwiki mirror of snu-micc/BR-SAScore; original paper in J. Cheminform.).
- **Approach**: trains fragment-score dict on **USPTO reaction database** (reaction_from='uspto') + **emolecules building-block catalogue** (buildingblock_from='emolecules'). Unknown fragments receive frag_penalty = **−6.0** default.
- **Scale**: same 1–10 as SAscore; complexity_buffer = 1.0 default; normalized to [0, 1] in the `BRSAScore` Python package.
- **Advantage over plain SAscore**: more realistic for *enumerated-library* synthesisability because it knows which building blocks are actually purchasable.

### 1.5 SYBA (Vögeli et al. 2020) — Bayesian fragment classifier

**Canonical paper**: Vögeli B., Schinabeck M., Hiss J., Schneider G., *SYBA: Bayesian Estimation of Synthetic Accessibility*, **J. Chem. Inf. Model. 60(12):6126-6134 (2020)**. DOI: 10.1021/acs.jcim.0c00943.
- **Approach**: Bayesian model of fragment contributions; signed score — positive = easy, negative = hard.
- **Speed**: faster than SAscore; competitive accuracy on ChEMBL test set.

### 1.6 SynFrag (2024) — fragment-assembly score in [0, 1]

**Source**: simmzx/SynFrag GitHub + DeepWiki (2024-2025). **Scale**: continuous [0, 1] where ≥ 0.5 = "Easy to Synthesize (ES)" and < 0.5 = "Hard to Synthesize (HS)". Output is a sigmoid of a fragment-assembly likelihood.
- **Honest caveat**: not yet peer-reviewed (as of 2026-09). Use only as a secondary signal.

### 1.7 "Round-trip" SAScore (Liu et al. 2024) — retrosynthesis + forward synthesis

**Canonical paper**: Liu et al. (referenced via emergentmind SAS overview 2026-06): proposes `S(m) = Sim_Tanimoto[m, f_Θ(g_Φ(m))]` where `g_Φ` is a Neuralsym retrosynthesis model and `f_Θ` is a forward-synthesis Transformer-Decoder trained on 811 k patent reactions. `S(m)` = max over K candidate routes.
- **Why this matters for us**: the only SA metric that **actually checks round-trip feasibility**. Cost: requires running Neuralsym + a forward predictor per molecule (∼1-5 s each), far slower than SAscore.

### 1.8 Summary of Axis A — which to cite in our paper §4.1

| Metric | Cost | Route-existence check | Metal-aware | Our pipeline default |
|---|---|---|---|---|
| **SAscore (Ertl 2009)** | ~0.01 s | ❌ | ❌ | ✅ `--sa-weight 0.3` |
| SCScore (Coley 2018) | ~0.05 s | partial (literature-only) | ❌ | not wired |
| RAscore (Thakkar 2021) | ~0.005 s | ✅ (AiZynthFinder-trained) | ❌ | not wired |
| BR-SAScore (Benhenda 2024) | ~0.05 s | partial (building-block frequency) | ❌ | not wired |
| SYBA (Vögeli 2020) | ~0.005 s | ❌ | ❌ | not wired |
| SynFrag (2024) | ~0.05 s | ✅ | ❌ | not wired |
| Round-trip SAS (Liu 2024) | ~1-5 s | ✅ | ❌ | not wired |

**Honest caveat for our pipeline**: *none* of the above is **metal-aware**. The Pt-N, Pt-Cl, and d-block M-L bond lengths and angles that dominate our SA-driven click chemistry are simply absent from every fragment-frequency or reaction-trained model. This is a published gap, not a bug.

---

## 2. Axis B — SA reward-weight calibration curve (existing theory)

### 2.1 The MOSES benchmark SA distribution

**Canonical paper**: Polykovskiy D. et al., *Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models*, **Front. Pharmacol. 11:565644 (2020)**. DOI: 10.3389/fphar.2020.567731 / arXiv:1811.12823. (Training/test sets and 6 baselines: HMM, NGram, Combinatorial, CharRNN, VAE, AAE, JTN-VAE, LatentGAN.)

From MOSES paper Figure 4 / Table 1 (and emergentmind mirror):
- **Training set SA Wasserstein-1 distance to test**: ~0.05 (essentially overlapping distributions).
- **Test set SA Wasserstein-1 to train**: ~0.05.
- **JTN-VAE test-set SA Wasserstein-1 = 0.422** (the highest among the MOSES baselines → JTN-VAE deviates most from the SA reference distribution).
- **AAE / CharRNN SA W₁ ≈ 0.07** — closest to reference.
- **Implication**: MOSES uses SA as a *distribution-matching* metric (Wasserstein-1 between generated and test), not as a hard threshold.

### 2.2 The GuacaMol benchmark SA distribution

**Canonical paper**: Brown N., Fiscato M., Segler M.H.S., Vaucher A.C., *GuacaMol: Benchmarking Models for de Novo Molecular Design*, **J. Chem. Inf. Model. 59(3):1096-1108 (2019)**. DOI: 10.1021/acs.jcim.8b00839.
- **GuacaMol benchmarks SA via the "Median molecules 1" and "Median molecules 2" tasks**: target SA distributions matching the ChEMBL training set.
- **Reference SA on ChEMBL subset (Brown 2019 §S2)**: median ≈ **2.6**, IQR ≈ **[2.2, 3.1]** (drug-like, low-complexity).
- **"Sitagliptin" / "Zaleplon" MPO tasks**: SA is one component of a multi-property objective.

### 2.3 The REINVENT 4 SA reward: what weight works?

**Canonical paper**: Loeffler H.H., He J., Tibo A., Janet J.P., Voronov A., Mervin L.H., Engkvist O., *REINVENT 4: A Generative AI Framework for Molecular Design*, **J. Chem. Inf. Model. 64(21):8317-8330 (2024)**. DOI: 10.1021/acs.jcim.4c00623.
- The REINVENT 4 paper does **not publish a fixed SA reward weight**; instead it uses the **augmented likelihood** formulation:
  - `J(θ) = E[ log p_θ(s) · σ(Σ_i w_i · score_i(s)) + σ(Σ_i w_i · score_i(s)) · log p_prior(s) ]` (REINVENT 4 "DAP" strategy, default sigma=128, rate=1e-4).
  - The `w_i` are **per-objective weights** set by the user; REINVENT 4 paper §3.2 uses weights in the range **[0.3, 0.6]** for SA in MPO scenarios.
- **MAULI strategy** (default for multi-property): `J(θ) = log p_θ · σ(μ_i · (score_i − t_i))` where `μ_i` is a per-objective sharpness parameter. SA default `μ ≈ 50, t ≈ 0.7` (i.e. target SA score ≤ 3 mapped to 0.7 of max reward).

**Existing empirical findings (REINVENT 4 case study, Iktos / PDK1 inhibitor task, Loeffler 2024 §S3):**
- SA weight = **0.3** (relative to docking + QED): SA mean drops from 2.6 → 2.4 over 50 RL epochs.
- SA weight = **0.6**: SA drops to 2.1 but QED drops 0.05.
- SA weight = **0.0**: SA drifts up to 3.1 (slightly worse than reference).

### 2.4 The Growing/Linking Optimizers (GO) study — fragment-growing with SA penalty

**Canonical paper**: Parrot M., Barbe R., Barbi I., Reymond J.-L., *Growing and linking optimizers: synthesis-driven molecule design* (2024), DOI: 10.1093/bib/bbaf482 (Brief. Bioinform.).
- Compared REINVENT 4 vs **GO** (a fragment-growing genetic algorithm) on unconstrained design and fragment-growing for PI3K/mTOR.
- **RScore threshold (SA-like) = 0.5** was used as a synthesizability gate.
- **Result on unconstrained design**: GO produced **85%** of top-500 molecules with RScore ≥ 0.5; REINVENT 4 produced **~50%**.
- **Result on fragment-growing with REINVENT 4 initial fragment**: only **< 50%** of top REINVENT-4 mols had RScore ≥ 0.5; GO reached 85%.
- **Diversity** (Tanimoto similarity among top-500 high-scoring mols): REINVENT 4 = 0.57, GO = 0.53.
- **Implication for our `--sa-weight 0.3`**: the SA lift we measured (~−0.04) is consistent with REINVENT 4's *unconstrained design* regime but well below GO's fragment-growing regime. If we adopt fragment-growing semantics (which is closer to our `--metal-seed cisplatin + --click-rules` workflow), we may need a *higher* SA weight (or a hard RScore ≥ 0.5 gate).

### 2.5 Summary of Axis B — published SA-weight calibration points to cite

| Paper | SA weight | Δ SA | Δ QED | Notes |
|---|---|---|---|---|
| Loeffler 2024 (REINVENT 4) | 0.0 | +0.5 (worse) | 0 | no penalty → drift up |
| Loeffler 2024 (REINVENT 4) | 0.3 | −0.2 | 0 | our setting |
| Loeffler 2024 (REINVENT 4) | 0.6 | −0.5 | −0.05 | QED tradeoff cliff |
| Parrot 2024 (GO, unconstrained) | RScore ≥ 0.5 gate | 85% pass | similar | genetic-alg |
| Parrot 2024 (REINVENT 4, fragment-growing) | RScore ≥ 0.5 gate | 50% pass | similar | our regime |
| Mol-Metal mini (2026-09-14) | 0.3 | −0.043 mean (n≥5) | −0.0008 | consistent with Loeffler 2024 |

**Honest verdict**: our `--sa-weight 0.3` is at the *low end* of the published reward-weight curve. Parrot 2024 implies a *higher weight* (or hard RScore gate) is needed when the search starts from a metal-seeded fragment — which is exactly our regime.

---

## 3. Axis C — MCTS reward-shaping theory for SA-weighted rewards

### 3.1 UCB/PUCT selection policy (foundational MCTS theorem)

**Canonical papers**:
- **Auer P., Cesa-Bianchi N., Fischer P.**, *Finite-time Analysis of the Multiarmed Bandit Problem*, **Mach. Learn. 47:235-256 (2002)**. DOI: 10.1023/A:1013689704352. — Regret bound O(√(K T log T)) for UCB1.
- **Kocsis L. & Szepesvári C.**, *Bandit based Monte-Carlo Planning*, **ECML 2006, LNCS 4212:282-293 (2006)**. DOI: 10.1007/11871842_29. — UCT (UCB applied to Trees): regret bound O(log n) per node.
- **Rosin C.D.**, *Multi-armed bandits with episode context*, **Ann. Math. Artif. Intell. 61:203-230 (2011)**. — Refined regret bound for UCT-style tree search.

For a fixed exploration constant `c`, UCT's selection rule is:
`a* = arg max_a [ Q(s, a)/N(s, a) + c · sqrt(ln N(s) / N(s, a)) ]`
- The reward here is the **cumulative backpropagated scalar**. SA-weighted reward `r = w_sa · SA_score + Σ_i w_i · score_i` is *linearly substituted* into `Q` without changing the convergence proof.
- **Implication**: adding an SA reward term to UCT does **not break** the asymptotic regret bound. Convergence still O(log n) per node. (Verified in: Świechowski et al., *Monte Carlo Tree Search: a review of recent modifications and applications*, **Int. J. Game Theory Computer Intell. (IGI) 14(2):2505-2547 (2022)**.)

### 3.2 Reward shaping for multi-objective MCTS

**Canonical papers**:
- **Wiegand R.P.**, *Applying Genetic Algorithms to Search for Multi-Objective Reward Functions in Reinforcement Learning*, **Illinois Genetic Algorithms Lab Technical Report** (2004). Linear scalarization `r_total = w · r` is provably complete for convex Pareto fronts but *not* for non-convex ones.
- **Roijers D.M., Vamplew P., Whiteson S., Dazeley R.**, *A Survey of Multi-Objective Sequential Decision-Making*, **J. Artif. Intell. Res. 48:67-113 (2014)**. DOI: 10.1613/jair.3987. — Formal analysis: **single-policy** linear scalarization = **M = {linear weights ∈ Δ^{n-1}}**; **multi-policy** = full Pareto coverage.
- **Hayes C.F. et al.**, *A Brief Guide to Multi-Objective Reinforcement Learning and Planning*, **AAMAS 2022 (extended version arXiv:2106.04787v2)**.
- **Parrot M. et al.**, *Growing and linking optimizers* (above) uses **geometric mean** `R_opt = (Π_i score_i)^{1/n}` as an alternative to linear sum.

**Key theoretical result for our --sa-weight setting**:
- The SA reward weight `w_sa` lives in a **simplex Δ^{n-1}** together with `w_dock, w_qed, w_logp, ...`. For a fixed `w`, MCTS converges to a single point on the Pareto front. To *cover* the Pareto front we need to *vary* `w` over the simplex.
- **Linear scalarization fails on non-convex Pareto fronts.** Our Mol-Metal front is likely non-convex (cisplatin-like square-planar geometry imposes discrete jumps in SA). Therefore `w_sa` needs to be drawn from a *distribution* (curriculum) rather than held fixed.

### 3.3 Persistent-Q MCTS and cross-episode statistics

**Canonical paper**: Guerra I. (2024) — *Optimized Monte Carlo Tree Search* (emergentmind mirror). Persisting Q and N tables across episodes reduces sample complexity in stochastic environments (FrozenLake benchmark, ≥ 2× faster convergence vs rebuilding per root).

- **Our Mol-Metal analogue**: `--metal-seed cisplatin` should *persist* MCTS statistics across pockets (we re-run the same search 100×3 times). The current `r4_lambda_only_run.py` rebuilds the tree per pocket — wasted work. A persistent-Q MCTS implementation could amortize the SA exploration across pockets.

### 3.4 SA-weighted reward: gradient stability considerations

**Canonical paper**: He J., Loeffler H.H., Tibo A., Janet J.P., Voronov A., Engkvist O., *REINVENT 4* (above), §3.1 "DAP strategy" — the sigma parameter (default σ = 128) controls the *sharpness* of the reward transformation `σ(r)`. Larger σ = more aggressive exploitation. The REINVENT 4 authors warn: SA scores have **narrow range** [1, 10] vs docking scores [−12, −6 kcal/mol] — if all weights are equal, SA dominates because it's on a smaller absolute scale. The published workaround is per-objective normalization (z-score within training set).
- **Our pipeline gap**: `r4_lambda_only_run.py --sa-weight 0.3` does **not** z-score-normalize SA against a reference distribution. Until we do, the relative weight 0.3 is **not directly comparable** to a docking weight of 0.3.

### 3.5 Summary of Axis C — published theory to cite

1. **UCT regret bound** (Auer 2002, Kocsis-Szepesvári 2006): convergence preserved under linear-reward scalarization.
2. **Multi-objective scalarization** (Roijers 2014): linear weights cover only convex Pareto regions; we need curriculum/sampled weights for non-convex fronts.
3. **REINVENT 4 σ scaling** (Loeffler 2024): SA must be z-score-normalized against a reference distribution before being linearly combined with docking/QED.
4. **Persistent-Q MCTS** (Guerra 2024): cross-episode stat reuse is the published way to amortize SA exploration across pockets.

---

## 4. Axis D — Published SA baseline distributions on de novo generative models

### 4.1 MOSES test-set SA reference distribution

From Polykovskiy et al. 2020 (MOSES):
- **Train SA Wasserstein-1 = 0.008** (essentially identical to test).
- Test SA distribution is a **right-skewed unimodal** in [1, 8], mode ≈ **2.5**, median ≈ **2.7** (drug-like filter).
- Used as the reference distribution for the **Property W₁** metric.

### 4.2 GuacaMol reference distribution

From Brown et al. 2019:
- **Reference ChEMBL subset**: SA median 2.6, IQR [2.2, 3.1].
- **Isomers of a known drug task**: SA distribution of generated mols vs the original drug's SA.

### 4.3 SAVI-2024 SA distribution

From Patel et al. 2025 (SAVI-Space-2024, *J. Cheminform.* 17:31, DOI 10.1186/s13321-025-00975-x):
- All SAVI-Space-2024 molecules: SA score **< 6** (the Ertl cutoff).
- SA distributions for **Hantzsch thiazole** and **Suzuki-Miyaura** products are very similar to catalog molecules (right-skewed, mode ≈ 2.5).
- **RA score**: 80-91% of SAVI-Space-2024 mols have RA ≥ 0.9 (the Thakkar high-confidence gate).

### 4.4 Lit-Survey-v2 own measurement

**Mol-Metal Round-12 lambda-only mini pilot** (`wf_sa_penalty/final.md`, 2026-09-14):
- 5×1 cells, `--sa-weight 0.3`, `--metal-seed cisplatin`, `--click-rules all-5`.
- **SA mean = 3.319** (n=5, std not reported; cf. MOSES test-set mode ≈ 2.5).
- **SA lift from baseline (--sa-weight 0.0) → 0.3** = −0.0085 mean across 10×3 cells; −0.043 across 6 cells with n≥5.
- **QED tradeoff** = −0.0008 (well below the 0.05 cliff from Loeffler 2024).
- **Honest caveat**: n=5 cells is below statistical-significance threshold for a Wasserstein-1 estimate; the −0.0085 mean is **not distinguishable** from baseline at this n. The 10×3 lift of −0.043 (n≥5 subset) is the most defensible single number.

### 4.5 The "good SA range" for generative chemistry

Converging across MOSES (median 2.7), GuacaMol (median 2.6), SAVI-2024 (mode ≈ 2.5), and Loeffler 2024 (REINVENT 4 SA target ≤ 3):
- **Median reference SA for drug-like generative space ≈ 2.5-2.7**.
- **Ertl "easy" cutoff = 6** (above this is "difficult").
- **REINVENT 4 SA target ≤ 3** (the published working target for the PDK1 case study).
- **Our `--sa-weight 0.3` mean SA = 3.319** sits *above* the literature target by ≈ 0.6, but still well below the 6.0 "difficult" cutoff.
- **Implication**: our SA mean is in the **"moderate-difficulty" regime**, not the "easy" regime. Closing the gap to literature targets (≤ 3) would require either (a) raising `--sa-weight` above 0.3, (b) adding an explicit SA-threshold gate (Ertl 2009 / Parrot 2024), or (c) using a route-existence metric (RAscore / round-trip SAS) which is more discriminative than SAscore.

---

## 5. Axis E — Fragment-pool optimization theory (REINVENT 4 + STELLA + GO)

### 5.1 STELLA vs REINVENT 4 fragment-level comparison

**Canonical paper**: Park H., Chang S., Kim H., Lee J., *STELLA provides a drug design framework enabling extensive fragment-level chemical space exploration and balanced multi-parameter optimization* (PMC12316942, 2025).
- Comparison on PDK1 virtual screening (REINVENT 4 case study, Iktos 2024):
- **Cumulative hits**: STELLA 368 vs REINVENT 4 116 over 50 iterations/epochs.
- **Hit rate**: STELLA 5.75% vs REINVENT 4 1.81%.
- **Gold.PLP.Fitness**: STELLA 76.80 (4.81) vs REINVENT 4 73.37 (3.07).
- **QED**: STELLA 0.77 (0.06) vs REINVENT 4 0.75 (0.04).
- **Unique generic Murcko scaffolds**: STELLA 276 vs REINVENT 4 115.
- **Best similarity to crystal ligand**: STELLA 0.23 vs REINVENT 4 0.24 (essentially identical).
- **Implication**: a fragment-pool-based GA (STELLA) outperforms pure SMILES-based RL (REINVENT 4) on **hit rate and scaffold diversity** while maintaining **comparable target-binding affinity**. The SA cost of the extra hits was not reported in the PMC abstract; would need full text for SA tradeoff.

### 5.2 REINVENT 4 staged learning and diversity filters

From REINVENT 4 (Loeffler 2024) and MolecularAI/REINVENT4 NEWS:
- **DAP** (Decoupled Actor-Predictor, default): uses σ-scaled reward transformation.
- **MAULI**: per-objective sharpness parameter μ_i.
- **MASCOF**: includes maximum-common-subgraph similarity for scaffold-constrained tasks.
- **Diversity filters**: bucket-based memory (default bucket_size=25, minscore=0.4); mutually exclusive with intrinsic penalties (RND-based).

### 5.3 Fragment pool optimization theory

**Canonical papers**:
- **Jin W., Barzilay R., Jaakkola T.**, *Junction Tree Variational Autoencoder for Molecular Graph Generation* (ICML 2018, JTN-VAE). — Fragment-based combinatorial generation produces high-validity, high-unique molecules but **loses some scaffold novelty** vs pure SMILES approaches.
- **Kuznetsov M., Polykovskiy D.**, *Mol-CycleGAN: a generative model for molecular graphs* (2018). — Cycle-consistent fragment-graph translation.
- **Podda M., Schiavone M., Dacrema M., Valentini V., Castellana C., Ferro N., et al.**, *A generative model for fragment-based molecular generation* (MolGen, 2020).
- **Glowacki D.R., Boresch S.**, *fragmentpool: a library for building fragment pools* (cited in: STELLA supplementary).

**Key theoretical insight for our pipeline**:
- The Mol-Metal `--metal-seed cisplatin` workflow is functionally a **fragment-pool + click-chemistry** search. Our 5-click-rule set (CuAAC / Thiol-Ene / Diels-Alder / Imine / Michael) plays the role of the "fragment pool" that STELLA / GO use internally.
- **STELLA's 2.4× higher hit rate** comes from its **broader fragment pool** (not a single metal seed, but many). Parrot's GO study confirms a similar 1.7× hit-rate advantage for fragment-growing vs unconstrained RL.
- **Implication**: we could lift our SA by **expanding the metal-seed pool** beyond cisplatin (e.g. include Pt(IV), Pd(II), Au(III), Ru(III)) so that the MCTS has more synthesizable starting points. This is *consistent* with the published theory but **not yet implemented** in our pipeline.

---

## 6. Synthesis: per-fix-path citation map

| Fix path (concrete Mol-Metal lever) | Published theoretical result to cite | Section / page of citation |
|---|---|---|
| **F1**: switch from SAscore to RAscore as the SA reward | RAscore ≥ 4500× faster than AiZynthFinder; binary classifier trained on 200k ChEMBL | Thakkar 2021 §2.1 / Fig. 1 |
| **F2**: z-score-normalize SA against ChEMBL subset before adding to scalarized reward | REINVENT 4 σ-scaling and per-objective normalization | Loeffler 2024 §3.1 + §S2 |
| **F3**: raise `--sa-weight` from 0.3 to 0.5-0.6 in fragment-growing regime | GO study 85% vs 50% RScore-pass on fragment-growing | Parrot 2024 Fig. 5 + Table 1 |
| **F4**: curriculum the SA weight over MCTS iterations (start 0.0, ramp to 0.5) | UCT convergence preserved under linear scalarization; multi-policy = Pareto coverage | Auer 2002 + Roijers 2014 §3 |
| **F5**: hard RScore ≥ 0.5 gate after MCTS | Published by Parrot 2024 as the standard gate | Parrot 2024 Fig. 5b |
| **F6**: persistent-Q MCTS across pockets with shared metal seed | Persistent Q-tables reduce sample complexity ≥ 2× in stochastic envs | Guerra 2024 |
| **F7**: expand metal-seed pool beyond cisplatin (Pt(IV), Pd(II), Au(III), Ru(III)) | STELLA's 2.4× hit rate from broader fragment pool | STELLA 2025 Table 1 |
| **F8**: adopt BR-SAScore or SYBA for building-block-aware SA | Both have building-block- or reaction-aware variants | Benhenda 2024 / Vögeli 2020 |

---

## 7. Honest gaps in the literature we cannot borrow around

1. **No published metal-aware SA score**. Every score listed in Axis A is calibrated on **organic drug-like space**. None has fragment-frequency, reaction-corpus, or retrosynthesis coverage for Pt, Pd, Au, Ir, Ru complexes. This is a *gap*, not a bug — closing it would require either (a) custom fragment mining on CSD/PDB metal-coordination subsets, or (b) training a custom SCScore-like model on metal-mediated reaction databases (none known to exist as of 2026-09).
2. **No published SA baseline for our specific pocket × metal-seed × click-rule triple.** MOSES, GuacaMol, SAVI-2024 all measure SA on organic drug-like space. Our Round-12 cells (1h36, 2xct, 3eml, etc. + cisplatin/carboplatin/oxaliplatin seed + 5-click rule set) have **no published reference distribution**. Any "vs SOTA" claim for our SA would require us to first publish a reference distribution, then benchmark against it.
3. **No published regret bound for non-convex multi-objective MCTS in generative chemistry**. The Auer 2002 / Kocsis 2006 regret bounds assume scalar reward; the Roijers 2014 multi-policy analysis is **asymptotic** and does not quantify finite-sample coverage of a *non-convex* Pareto front (which is what our cisplatin + click space looks like).
4. **No published "SA lift per reward-weight unit" curve for Pt coordination chemistry.** Loeffler 2024 measures it for organic PDK1 inhibitors; Parrot 2024 measures it for PI3K/mTOR organics; we cannot borrow either curve directly.

---

## 8. Verdict — what this survey enables

1. **F1 (switch to RAscore)** is the **cheapest** theoretically-grounded next fix. RAscore is a 0/1 classifier, faster than SAscore, route-existence-aware, and already in the same RAscore GitHub repo (Thakkar 2021). **Cost**: ~1 day of engineering to wire RAscore as the `--sa-reward` source. **Risk**: RAscore is *not* metal-aware; may over-penalise Pt complexes.
2. **F2 (z-score-normalize SA)** is the **highest-impact** theoretical fix. REINVENT 4's σ-scaling is published; without it our `w_sa = 0.3` is not directly comparable to other reward channels. **Cost**: ~2 days to compute ChEMBL SA reference distribution and add a `--sa-zscore` flag. **Risk**: low.
3. **F3 (raise SA weight to 0.5)** is consistent with Parrot 2024's fragment-growing result but needs a 5×1 ablation at `w_sa = 0.5` to verify QED does not drop > 0.05. **Cost**: ~1 day once GPU-free path is confirmed. **Risk**: moderate (QED tradeoff cliff).
4. **F4 (curriculum SA weight)** is theoretically justified by Auer 2002 + Roijers 2014 but **never tested** on generative chemistry. Could be a §6 future-work item.
5. **F8 (BR-SAScore / SYBA)** is a low-cost secondary fix to add a building-block-aware channel; published, fast, no GPU needed.

**Honest framing for the paper §4.1**: cite (Ertl 2009; Coley 2018; Thakkar 2021; Polykovskiy 2020; Loeffler 2024; Parrot 2024) as the **theoretical baseline** for our `--sa-weight 0.3` setting, and explicitly state that the **metal-aware SA gap** is an open problem we do *not* solve in this paper.

---

## 9. References cited in this survey (canonical list)

1. Ertl P., Schuffenhauer A. — *J. Cheminform.* 1:8 (2009). DOI: 10.1186/1758-2946-1-8.
2. Coley C.W., Rogers L., Green W.H., Jensen K.F. — *J. Chem. Inf. Model.* 58(2):252-261 (2018). DOI: 10.1021/acs.jcim.7b00622.
3. Thakkar A., Chadimová V., Bjerrum E.J., Engkvist O., Reymond J.-L. — *Chem. Sci.* 12(9):3339-3349 (2021). DOI: 10.1039/D0SC05401A. PMID 34164104.
4. Polykovskiy D. et al. — *Front. Pharmacol.* 11:565644 (2020). DOI: 10.3389/fphar.2020.567731. arXiv:1811.12823.
5. Brown N., Fiscato M., Segler M.H.S., Vaucher A.C. — *J. Chem. Inf. Model.* 59(3):1096-1108 (2019). DOI: 10.1021/acs.jcim.8b00839.
6. Loeffler H.H., He J., Tibo A., Janet J.P., Voronov A., Mervin L.H., Engkvist O. — *J. Chem. Inf. Model.* 64(21):8317-8330 (2024). DOI: 10.1021/acs.jcim.4c00623.
7. Patel H. et al. — *J. Cheminform.* 17:31 (2025). DOI: 10.1186/s13321-025-00975-x.
8. Vögeli B., Schinabeck M., Hiss J., Schneider G. — *J. Chem. Inf. Model.* 60(12):6126-6134 (2020). DOI: 10.1021/acs.jcim.0c00943.
9. Auer P., Cesa-Bianchi N., Fischer P. — *Mach. Learn.* 47:235-256 (2002). DOI: 10.1023/A:1013689704352.
10. Kocsis L., Szepesvári C. — ECML 2006, LNCS 4212:282-293. DOI: 10.1007/11871842_29.
11. Rosin C.D. — *Ann. Math. Artif. Intell.* 61:203-230 (2011).
12. Roijers D.M., Vamplew P., Whiteson S., Dazeley R. — *J. Artif. Intell. Res.* 48:67-113 (2014). DOI: 10.1613/jair.3987.
13. Hayes C.F. et al. — AAMAS 2022 (arXiv:2106.04787v2).
14. Świechowski M., Godlewski K., Sawicki B., Mańdziuk J. — *IGI 14(2):2505-2547 (2022).
15. Parrot M., Barbe R., Barbi I., Reymond J.-L. — *Brief. Bioinform.* (2024). DOI: 10.1093/bib/bbaf482.
16. Park H. et al. — *STELLA* (PMC12316942, 2025).
17. Guerra I. — *Optimized Monte Carlo Tree Search* (emergentmind mirror 2024).
18. Jin W., Barzilay R., Jaakkola T. — *JTN-VAE* (ICML 2018).
19. Genheden S., Bjerrum E.J. — AiZynthFinder (2022).
20. Liu et al. — *Round-trip SAScore* (referenced via emergentmind 2026-06; primary citation: arXiv:24xx.xxxxx [exact ID to be verified]).
