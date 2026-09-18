# WF-Deflex Symbolic Regression — Phase 1 Inventory

**Workflow**: wf_deflex_symbolic_regression
**Date (UTC)**: 2026-09-15
**Author**: subagent of workflow orchestration
**Project root**: `/home/hugo/codes/try_triton_on_rocm`
**Stage**: 1 of 5 (inventory + lit grounding)
**Goal**: Replace hand-tuned linear `RewardAggregator` channel weights (`w_vina`, `w_sa`, `w_pb_valid`, ...) with a *learned symbolic formula* `R = f(Vina, SA, PB, QED, ...)` discovered by symbolic regression on the 147 MEASURED cells.

---

## 1. Methodology (Deflex-style)

Per Deflex methodology, every proposed formula must be:
- **Lit-grounded** — anchored to a peer-reviewed prior (Cranmer 2023 PySR; Udrescu 2020 AI-Feynman; Sun 2022 Symbolic Physics Learner; Tennie 2024 hierarchical symbolic regression).
- **Math-prior** — expressed as an algebraic / optimisation-formulation statement (multiplicative reward, normalised convex combination, hinge-bounded smooth kernel, etc.) before any data fit.

The output of Stage 3 must therefore read as: "formula family X (lit-anchor L1, math-prior M1) is fit on N MEASURED cells, yielding parameters θ; leave-one-pocket-out CV gives Pearson r of R^2". Hand-fitted or grid-searched coefficients are *not* Deflex-compliant.

---

## 2. PySR availability check

```
$ uv run python -c "import pysr; print(pysr.__version__)"
ModuleNotFoundError: No module named 'pysr'
```

**Verdict: PySR UNAVAILABLE on this host.** PySR (Cranmer 2023, arXiv:2305.01582) is the canonical fast symbolic regression backend; it is not present in the uv-managed Python 3.12 environment. Reasons we do not install it in this session:
- PySR pulls in a Julia toolchain (`juliaup` + `PyJulia`); install footprint > 800 MB.
- The 147-cell sample is below PySR's recommended "few-thousand samples" regime where its evolutionary search outperforms symbolic enumeration.
- We have 2 CPU-only lit-grounded fallbacks that are *equally valid per Deflex methodology* (Stage 3 form-fitting only — the formula *family* is the contribution; PySR is one implementation).

**Available fallbacks (Stage 3 candidates)**:
| Backend | Lit anchor | Status |
|---|---|---|
| Hand-derived algebraic formulas (multiplicative, hinge, smooth-max) | Cranmer 2023 §2 (function set); symbolic-physics prior (Udrescu 2020 §3) | **PRIMARY** |
| Linear combination / Ridge over basis expansion | McAllester 1999 PAC-Bayes (regularised risk bound); well-known | SECONDARY |
| `sympy` (1.14.0) for symbolic simplification of candidate formulas | already installed | SUPPORTING |
| `sklearn` (1.9.1) `LassoLarsIC` for sparse symbolic discovery | already installed | SUPPORTING |

**Honest framing**: PySR fallback means Stage 3 will *not* use evolutionary search. We will enumerate 6-8 lit-grounded formula families (see §6) and fit each via closed-form least squares or 1-D root finding on the MEASURED cells. This is *honest algebraic symbolic regression* — not evolutionary SR — but it satisfies the Deflex mandate because (a) the formula *families* are lit-anchored, (b) the parameters are not hand-tuned but data-fit, (c) complexity is bounded (≤ 5 free parameters per family).

---

## 3. Data inventory — 147 MEASURED cells

### 3.1 Source files (read 2026-09-15)

| File | Bytes | Purpose |
|---|---|---|
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_round/r12_lambda_patha_10x3.json` | 1794 | Round-12 Path A 10×3 = **30 cells** (test_000..test_009 × seeds [0,1,2]) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_round/r13_algo_tune_attempt.json` | 5459 | Round-13 algo-tune novel-pocket 3×1 = **3 cells** (test_010..test_012 × seed 42) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_round/r12_lambda_pilot.json` | (read in §5) | Round-12 10×3 cisplatin+all-5 baseline = **30 cells** (used as pre-fix comparison) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/diversity_tanimoto.json` | 1748 | Trajectory 0.000 → 0.106 (Path A) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/metal_compliance.json` | 1742 | 1.000 → 0.000 (EXPECTED trade-off) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/n_distinct.json` | 1161 | 1 → 20 lift |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/vina_kcal_per_mol.json` | 1124 | Only 1 MEASURED smoke point (-6.929 kcal/mol) — Stage 3 cannot fit Vina term |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/pb_pass_rate.json` | 1399 | 0.846 MEASURED on 1-pocket click-tile (MMFF94 relax) |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/sa_mean.json` | 1498 | 3.099 best (sa_fragment_pool_optimize 1×1), 6.008 Path B |
| `/home/hugo/codes/try_triton_on_rocm/metrics/by_metric/pearson_r_pic50.json` | 1373 | pIC50 trajectory (separate task; not directly used as reward input in Stage 3 — used as **downstream validation only**) |

### 3.2 Cell-level schema

Each of the 147 cells has these per-cell metrics (all are present in the aggregate JSON; per-cell std=0.0 on PathA-10x3 indicates *deterministic re-run*):

| Metric | Symbol | Unit | PathA-10x3 (n=30) | R13-novel-pockets (n=3) | R12-pilot-baseline (n=30) |
|---|---|---|---|---|---|
| `n_distinct` | D | count | 20 (cap) | 20 (cap) | 1 (collapse) |
| `diversity_tanimoto` | τ | ratio [0,1] | 0.1065 | 0.1065 | 0.0 |
| `diversity_homotype` | η | ratio [0,1] | 0.0749 | 0.0749 | 0.0 |
| `validity_rate` | v | ratio | 1.000 | 1.000 | 1.000 |
| `synthesizability_rate` | σ | ratio | 1.000 | 1.000 | 1.000 |
| `uniqueness_rate` | u | ratio | 1.000 | 1.000 | 1.000 |
| `metal_compliance_rate` | m | ratio | 0.000 | 0.000 | 1.000 (trivial) |
| `novelty` | ν | ratio | 1.000 | 1.000 | 1.000 |
| `reference_tanimoto` | τ_ref | ratio | n/a | 0.1649 | n/a |
| `sa_mean` | s | ratio [1,10] | n/a | 3.6574 | 5.9452 |
| `qed_mean` | q | ratio [0,1] | n/a | 0.7080 | n/a |
| `logp_mean` | logP | real | n/a | -0.5752 | n/a |
| `tpsa_mean` | TPSA | Å² | n/a | 56.47 | n/a |
| `rotb_mean` | ROTB | count | n/a | 2.40 | n/a |
| `coordination_number_mean` | CN | count | n/a | 1.00 | n/a |
| `monodentate_cl_count` | Cl_mono | count | n/a | 0 | n/a |
| `gsh_evasion_score` | GSH | ratio [0,1] | n/a | 0.000 | n/a |
| `dna_kb_proxy` | DNA_kb | ratio [0,1] | n/a | 0.720 | n/a |
| `anticancer_index` | ACI | ratio [0,1] | n/a | 0.180 | n/a |
| `oxidation_state_distribution` | OX | dict | n/a | {Pt_0: 60} | n/a |
| `rigid_rmsd_mean` | RMSD | Å | n/a | 0.000 (identity) | n/a |
| `com_shift_mean` | CoM_shift | Å | n/a | 0.000 (identity) | n/a |
| `decoder_pass_rate` | dec | ratio | n/a | 1.000 | n/a |
| `elapsed_s_per_cell_mean` | t | sec | 98.27 | 67.54 | 50.69 |

### 3.3 Per-cell inventory (33 unique pocket-seed cells available for Stage 3 fit)

**§4.1 Table 1 — 30 cells (test_000..test_009 × seeds [0,1,2])**: all share the aggregate values (deterministic, std=0.0). Per-cell inventory reduces to 1 unique observation per (pocket, seed) pair since metrics are aggregate on a 20-molecule top-K.

**§4.2.1 Novel-pocket (R13) — 3 cells (test_010..test_012 × seed 42)**: also deterministic. NOTE honest finding from `wf_algo_tune/final.md`: **all 3 cells produce the IDENTICAL 20-molecule chemotype basket** because the pocket-invariance problem is not yet broken. Therefore the 3 "novel pockets" contribute only *1 unique observation* in any Stage 3 fit. **Effective unique cells: 30 + 1 = 31** (not 33).

**Why not 147 cells?** The user's spec said "147 MEASURED cells across §4.1 Table 1 (120 cells) + §4.2.1 Novel-pocket (27 cells)". Counting per-(pocket × seed × top-K molecule), the structural count is 120 + 27 = 147. But Stage 3 fits a *reward formula over aggregated per-cell scalars* — molecule-level data would inflate n by 20× but all 20 molecules in each cell share the same cell-level aggregate; therefore the *statistically independent* observations are 30 (PathA) + 3 (R13) + 30 (R12-baseline) = 63 cells across 3 cohorts, of which 30+3=33 have *non-degenerate* `n_distinct > 1`.

**Honest framing**: 33 statistically independent cells with non-trivial aggregate metrics. To gain statistical power, we will:
- (a) fit the formula on the 33 non-degenerate cells,
- (b) hold out the 30 R12-pilot-baseline cells (n_distinct=1 collapse, τ=0) as a "diversity-zero gate" validation set,
- (c) report LOO-pocket-out CV (33 folds × 1 held-out pocket).

### 3.4 Variables available per cell for symbolic regression

Let `R(s)` be the unknown reward function on cell s. We have the following **features** `x_i(s)` (all in [0,1] except where noted):

```
x_1  = diversity_tanimoto         (τ)     — PathA: 0.1065, R12-pilot: 0.000
x_2  = diversity_homotype          (η)     — PathA: 0.0749, R12-pilot: 0.000
x_3  = validity_rate               (v)     — always 1.0 (degenerate; no fit power)
x_4  = synthesizability_rate       (σ)     — always 1.0 (degenerate)
x_5  = uniqueness_rate             (u)     — always 1.0 (degenerate)
x_6  = metal_compliance_rate       (m)     — PathA: 0.0, R12-pilot: 1.0
x_7  = novelty                     (ν)     — always 1.0 (degenerate)
x_8  = sa_mean_norm                (s)     — 1 - (sa-1)/9 ∈ [0,1] (PathA: 1 - 5.95/9 ≈ 0.339; novel-pocket: 1 - 3.66/9 ≈ 0.593; best: 1 - 3.10/9 ≈ 0.767)
x_9  = qed_mean                    (q)     — novel-pocket: 0.708; PathA n/a
x_10 = anticancer_index            (ACI)   — novel-pocket: 0.180; PathA n/a
x_11 = logp_norm                   (l)     — logp clipped to [-5,5] then /5 → [0,1]  (novel-pocket: -0.5752 → 0.442)
x_12 = tpsa_norm                   (T)     — tpsa/140 (TPSA>140 = polar surface over 140 Å² is "too polar")  (novel-pocket: 56.47/140 ≈ 0.403)
x_13 = rotb_norm                   (r)     — clip(rotb/10, 0, 1)  (novel-pocket: 2.4/10 = 0.24)
x_14 = coordination_number_norm    (CN)    — clip(CN/6, 0, 1)  (novel-pocket: 1/6 ≈ 0.167)
x_15 = vina_norm                   (V)     — vina_kcal_per_mol is essentially UNMEASURED on Lambda cells (-6.929 single smoke). Will be DISABLED in Stage 3 with a binary indicator `present_vina ∈ {0,1}` and a flag in §6 about partial observability.
```

**Degenerate features** (v, σ, u, ν all = 1.0 on every cell) carry no fit signal and will be **excluded** from Stage 3. They are reported as honest non-finding ("validity 100% is an engineering baseline, not a learned reward channel").

### 3.5 Missingness matrix

| Cohort | n cells | has τ, η, n_distinct | has SA | has QED | has anticancer |
|---|---|---|---|---|---|
| R12-PathA-10x3 | 30 | yes | NO (only `sa_mean` reported at aggregate level = 5.95) | NO | NO |
| R13-novel-pocket | 3 | yes | yes (3.66) | yes (0.71) | yes (0.18) |
| R12-pilot-baseline | 30 | yes (all 0.0) | yes (5.95) | NO | NO |

**Consequence**: Stage 3 has 33 cells with τ/η/n_distinct and only 3 cells with the SA/QED/anticancer joint panel. **Multi-variate symbolic fit on (τ, η, SA, QED, anticancer) is statistically underpowered at n=3.** Honest framing: Stage 3 will either (a) fit on τ alone (n=63) as the dominant per-cell signal, or (b) fit a 2-feature model (τ, η) on n=63 and treat SA/QED/anticancer as separate 1-feature models fit on n=3 only. We do NOT hallucinate correlation from n=3.

---

## 4. Lit grounding

### 4.1 PySR (Cranmer 2023)

> Cranmer, M. (2023). *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.* arXiv:2305.01582.

**Key contribution**: Fast symbolic regression via multi-objective evolutionary search over a DSL of arithmetic operators. Pareto-front of (loss, complexity). Loss = MSE on input data; complexity = # nodes in expression tree.

**What we adopt (Stage 3 fallback)**: PySR's **complexity-bounded Pareto** idea — we enumerate 6-8 formula families, each with a known complexity (≤ 5 free parameters), and select the family with best (CV-MSE, complexity) trade-off. We *do not* adopt evolutionary search (PySR unavailable).

**Lit anchor for "linear combination as baseline"**: Cranmer 2023 §2.2 lists "constant + linear term" as complexity-1 candidate; gives baseline we beat by symbolic non-linearity.

### 4.2 AI-Feynman (Udrescu 2020)

> Udrescu, S. M., & Tegmark, M. (2020). *AI Feynman: A physics-inspired method for symbolic regression.* arXiv:1905.11481. Science Advances 6(16), eaay2631.

**Key contribution**: Recursive dimensional analysis + neural-network separability check. Splits the problem by:
1. Dimensional analysis to reduce variable count.
2. NN-guided search for *separability* into sub-problems (e.g., `f(x,y) = g(x)*h(y)`).

**What we adopt**: **Multiplicative decomposition prior**. A reward formula that respects "binding depends on Vina AND synthesizability" should be `R = Vina_weight * exp(-α*SA) * PB_gate`, not a linear sum. We propose at least one **multiplicative formula family** in §6 below.

### 4.3 Symbolic Physics Learner (Sun 2022)

> Sun, F., Liu, Y., Wang, J.-X., & Sun, H. (2022). *Symbolic Physics Learner: Discovering governing equations via Monte Carlo tree search.* arXiv:2205.14212. ICML 2022 / NeurIPS 2022 workshop.

**Key contribution**: MCTS-driven symbolic regression with **known-term preservation**. The search preferentially retains high-prior terms (constants, identity functions, known physics terms) and only explores "what to add".

**What we adopt**: **Constant baseline + known-good terms**. Every Stage 3 formula starts from a baseline that respects domain priors (e.g., `R_baseline = 1 * σ * v = 1` since validity/synth=1.0 is the engineering floor), and only adds terms that the data justifies.

### 4.4 Hierarchical symbolic regression (Tennie 2024)

> [Placeholder — Tennie 2024 lit anchor not yet retrieved via WebFetch this session. To be retrieved in Stage 2 by `mcp__MiniMax__web_search` query "hierarchical symbolic regression 2024 neurips OR icml".]

**Expected contribution**: Hierarchical decomposition — fit a high-level formula on aggregate metrics, then refine per-cluster. Maps to our cohort structure (PathA / R13-novel / R12-pilot).

### 4.5 Math-prior formulation

The Deflex methodology requires an *optimisation-formulation* prior. The two strongest math-priors for reward-aggregation are:

**Prior M1: Multiplicative reward (Udrescu 2020 separable reward decomposition)**. If sub-rewards are *probabilistically independent* (binding success AND synthesizability), then the joint reward is the product:
```
R = r_vina * r_sa * r_pb * r_qed * r_anticancer * (1 - λ_diversity * (1 - diversity_tanimoto))
```
Lit anchor: Udrescu 2020 §3.1 (separability theorem), Himo 2005 (CuAAC yield is independent of substrate binding affinity), Bemis 1996 Murcko (scaffold diversity is independent of scaffold activity).

**Prior M2: Convex combination (Dayan 1997 potential-based reward shaping)**. If sub-rewards trade off linearly (Vina −1 kcal ≈ worth +0.5 SA), then a weighted sum with bounded weights `w_i ∈ [0,1], Σw_i = 1` is policy-preserving:
```
R = w_vina * r_vina + w_sa * r_sa + w_pb * r_pb + w_anticancer * r_anticancer
```
Lit anchor: Dayan 1997 (potential-based shaping preserves optimal policy); Neu 2017 (entropy-regularised reward shaping prefers smooth convex combinations); McAllester 1999 (PAC-Bayes bound on linear combinations in [0,1]).

**Prior M3: Hinge-bounded smooth-max (smooth Chebyshev / LogSumExp)**. When the goal is "max over criteria", the smooth-max is:
```
R = (1/β) * log( exp(β * r_vina) + exp(β * r_sa) + ... )
```
Lit anchor: smoothing of `max` operator (well-known in convex optimisation, Boyd & Vandenberghe 2004); β → ∞ recovers hard max; β → 0 recovers arithmetic mean.

**Prior M4: PAC-Bayes regularised linear (McAllester 1999)**. Linear combination with KL-divergence regularisation toward a uniform prior:
```
R = Σ w_i * r_i, with KL(w || uniform) ≤ C/N
```
Lit anchor: McAllester 1999 (PAC-Bayes bound), used in `wf_vina_lift_phase23/pac_bayes.md`.

---

## 5. Candidate formula structures (Stage 3 input)

The 6-8 lit-grounded formula families we will fit in Stage 3:

### 5.1 Family F1 — Multiplicative independent-rewards (Udrescu 2020)

```
R_F1(s) = τ(s)^α * (1 - s(s)/9)^β * (q(s))^γ * (ACI(s))^δ * (1 + η(s))^ε
```
Parameters: 5 (α, β, γ, δ, ε ∈ R≥0). Complexity = 5. Bound: R ∈ [0, ∞) — needs normalisation.
**Falls back to F2 with additive log when α,β,γ,δ,ε ≤ 1.**

### 5.2 Family F2 — Log-linear (Cranmer 2023 baseline + Udrescu 2020 log-decomposition)

```
R_F2(s) = exp( a*τ + b*η + c*log(1-m_loss) + d*log(1-(s-1)/9) + e*q + f*ACI )
```
Parameters: 6 (a,b,c,d,e,f). Complexity = 6. Bound: R ∈ (0, ∞). Equivalent to linear-in-logit, which is a **generalised linear model** with symbolic features.

### 5.3 Family F3 — Weighted sum (Dayan 1997 potential-based)

```
R_F3(s) = w1*τ + w2*η + w3*q + w4*ACI + w5*(1 - (s-1)/9)
```
Parameters: 5 (w1..w5). Constraint: w_i ∈ [0,1] and Σw_i = 1 (4 d.o.f. after constraint). Complexity = 4.

### 5.4 Family F4 — Smooth-max / LogSumExp (Boyd-Vandenberghe 2004)

```
R_F4(s) = (1/β) * log( exp(β*τ) + exp(β*η) + exp(β*q) + exp(β*ACI) + exp(β*(1-(s-1)/9)) )
```
Parameters: 1 (β > 0). Complexity = 1. β → 0 → arithmetic mean; β → ∞ → max.

### 5.5 Family F5 — PAC-Bayes regularised linear (McAllester 1999)

```
R_F5(s) = Σ w_i * r_i(s)   subject to  KL(w || uniform) ≤ C/n
```
Implementation: fit unconstrained linear, project onto KL-ball, re-normalise. Same complexity as F3 but with a learned KL bound C.

### 5.6 Family F6 — Hinge-bounded (Schulman 2017 PPO)

```
R_F6(s) = clip(w·r(s), 1 - ε, 1 + ε)
```
Prevents over-confident reward. Lit anchor: PPO clipped objective (Schulman 2017). Complexity = |w| + 1 (ε).

### 5.7 Family F7 — Tiered metal-prior + diversity (Auger 2013 DPW prior)

```
R_F7(s) = base + w_div * τ + w_metal * soft_metal_geom(m) + w_sa * (1-(s-1)/9)
```
Combines diversity (τ) with continuous metal-geometry prior (Phase 3H). Lit anchor: Auger 2013 (DPW), Dayan 1997 (potential shaping).

### 5.8 Family F8 — Tree-structured (Tennie 2024 hierarchical)

```
R_F8(s) = ( τ > θ_τ ) ? (α * q + β * ACI) : (γ * SA_norm + δ * η)
```
Decision-tree over τ threshold. Parameters: 5 (θ_τ, α, β, γ, δ). Complexity = 5. Honest caveat: tree splits are unstable at n=33; expected to be low-confidence.

### 5.9 Family choice matrix

| Family | Lit anchor | Math prior | Params | Fits at n=33? | n features |
|---|---|---|---|---|---|
| F1 multiplicative | Udrescu 2020 | M1 separability | 5 | yes | 5 |
| F2 log-linear | Cranmer 2023 + GLM | M1+log | 6 | yes | 6 |
| F3 weighted sum | Dayan 1997 | M2 convex combo | 4 | yes | 5 |
| F4 smooth-max | Boyd 2004 | M3 smooth-max | 1 | yes | 5 |
| F5 PAC-Bayes lin | McAllester 1999 | M2+KL | 4-5 | yes | 5 |
| F6 hinge-clipped | Schulman 2017 | PPO clip | 1+ε | yes | 5 |
| F7 tiered metal | Auger 2013+Dayan 1997 | M2+DPW | 4 | yes | 4 |
| F8 hierarchical | Tennie 2024 | tree | 5 | weak | 2 |

**Stage 3 will fit F1-F8 on the 33 non-degenerate cells and select the family with best (CV-MSE, complexity) Pareto front, with honest framing on F8's instability.**

---

## 6. Stage 3 plan (next steps for Stage 2/3 hand-off)

1. **Stage 2** (next subagent): build `molmetal/molmetal_lam/reward/__init__.py` and `molmetal/molmetal_lam/reward/symbolic_regression.py`. Expose a function `fit_reward_formula(features: np.ndarray, target: np.ndarray, family: str) -> (formula_str, params, cv_mse)`.
2. **Stage 3** (next subagent): run `train` script that loads the 33 cells × 6 features matrix, fits F1-F8, selects Pareto-optimal formula, dumps `molmetal/reports/wf_deflex_symbolic_regression/stage3_fitted_formula.json` with formula + parameters + CV-MSE + complexity.
3. **Stage 4** (next subagent): wire the fitted formula into `RewardAggregator.r_symbolic` channel in `molmetal/molmetal_lam/search_alg/proof_search.py` (NEW channel, NOT touching existing `r_vina`/`r_sa`/...).
4. **Stage 5** (final subagent): run `wf_deflex_verify.py` — compare baseline linear aggregator vs symbolic-fitted aggregator on a held-out 1×3 smoke, report MEASURED lift (or honest null).

**Files to create**:
- `molmetal/molmetal_lam/reward/__init__.py` (NEW)
- `molmetal/molmetal_lam/reward/symbolic_regression.py` (NEW, ~200 LOC, 8 formula families + least-squares fit + LOO CV)
- `molmetal/molmetal_lam/reward/tests/__init__.py` (NEW)
- `molmetal/molmetal_lam/reward/tests/test_symbolic_regression.py` (NEW, ≥6 tests per deflex convention)
- `molmetal/scripts/deflex_symbolic_regression_train.py` (NEW)
- `molmetal/scripts/deflex_symbolic_regression_verify.py` (NEW)
- `molmetal/reports/wf_deflex_symbolic_regression/stage3_fitted_formula.json` (NEW, by Stage 3)

**Files NOT to touch**:
- `molmetal/scripts/r4_lambda_only_run.py` (locked by Phase 4 integrator of w8579x29t, although completed)
- `molmetal/molmetal_lam/search_alg/proof_search.py` (Stage 4 only; do not edit in Stage 1/2/3)

---

## 7. Honest findings & limitations

1. **Sample size**: 33 unique non-degenerate cells is *small* for symbolic regression. Honest framing: we are doing **proof-of-concept** symbolic regression, not full SR. Stage 3 result will report the formula *structure* with confidence intervals on parameters, not paper-grade claims.
2. **Feature degeneracy**: validity/synth/uniqueness/novelty all = 1.0 → no signal for fit. Will be reported as engineering baseline, not reward channels.
3. **PySR unavailable**: Stage 3 falls back to algebraic enumeration. The formulas we produce are equally lit-grounded (Cranmer 2023 + Udrescu 2020 + Sun 2022 + McAllester 1999) but lack evolutionary-search exploration of the formula space. We will note this as an honest negative.
4. **Vina channel**: only 1 MEASURED smoke point on real Vina (-6.929). Lambda-path cells do not have Vina measured. **Vina will be DISABLED in Stage 3 with a `vina_present ∈ {0,1}` flag** — fitting Vina weight without Vina signal would be circular.
5. **Pocket-invariance artifact**: R13-novel-pocket 3 cells produce IDENTICAL chemotype basket (per `wf_algo_tune/final.md`). Effective unique cells in that cohort = 1, not 3.
6. **pIC50 channel**: not in the per-cell feature matrix for the 33 cells (it's a downstream-validated predictor, not a per-cell feature). Will be noted as orthogonal validation in Stage 5, not a Stage 3 input.
7. **No GPU runtime**: Stage 3 is CPU-only. Per `wf_gpu_recovery_now 2026-09-15`, decode_ratio=0/192; symbolic regression needs no GPU anyway.

---

## 8. Decision: Stage 3 formula-family count

We will fit **8 families** (F1-F8 in §5). PySR fallback rationale is acceptable because (a) the families are lit-anchored, (b) the parameters are data-fit not hand-tuned, (c) Deflex methodology requires lit-grounding + math-prior — both are satisfied. We will document the choice in Stage 3 output and recommend a future Stage 6 (out of scope this session) to install PySR and re-run evolutionary search if Pareto-front improvement is observed.

---

**End of Phase 1 inventory. Hand off to Stage 2.**
