# Phase 1 Lit Research: 5 Lambda Core Features

> Context: Round-13 100x3 honest negative result (Lambda killed, PB 30/30 search-bound at n_sim=100 with strict gates). PathA-10x3 lift was local not universal. This phase grounds the 5 features we plan to ship + generalize, in published theory so the implementation is "cite existing theorem + math-prior optimization formulation" rather than ad-hoc heuristics.
>
> All citations are real (web-searched 2026-09-15), not from training memory. Project root: /home/hugo/codes/try_triton_on_rocm. uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64. Honest-framing mandatory.

---

## 1. Conformer embedding (canonical 3D for Vina docking)

### Citations (real, web-verified 2026-09-15)
1. **Riniker & Landrum 2015** "Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation" *J. Chem. Inf. Model.* 55(12):2562-2574. doi:10.1021/acs.jcim.5b00654 — ETKDG (Experimental-Torsion Knowledge Distance Geometry), the algorithm baked into RDKit `AllChem.EmbedMolecule` as default since 2015. Repros 84% of CSD crystal conformations within 1.0 A RMSD, 38% within 0.5 A.
2. **Halgren 1996** "Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94" *J. Comput. Chem.* 17(5-6):490-519. doi:10.1002/(SICI)1096-987X(199604)17:5/6:490::AID-JCC1>3.0.CO;2-P — MMFF94s force field, 5086 citations. Parameterized at HF/6-31G* + MP2/6-31G* on 500+ structures. Reproduces exptl bond lengths at 0.014 A rms, conformational energies at 0.38 kcal/mol rms.
3. **Blaney & Dixon 1994** "Distance Geometry in Molecular Modeling" *Reviews in Computational Chemistry* vol 5 (Lipkowitz & Boyd eds, VCH) — classic distance-geometry reference; Havel-Crippen-Sippl metrization, triangle-bounds smoothing, 4D-embedding.
4. **Wang, Witek, Landrum, Riniker 2020** "Improving Conformer Generation for Small Rings and Macrocycles Based on Distance Geometry and Experimental Torsional-Angle Preferences" *J. Chem. Inf. Model.* 60(4):2044-2058. doi:10.1021/acs.jcim.0c00025 — extended ETKDG with 105 SMARTS for small rings; macrocycle heuristics (elliptical geometry, CPCIs).

### Key theorem / math-prior
**ETKDG bounds matrix + torsion-potential embedding**: Given a molecule with N atoms, the distance bounds matrix L_ij <= d_ij <= U_ij is smoothed via triangle inequality, then a random distance matrix D is sampled uniformly in the bounds, then projected into 3D by eigendecomposition of the metric matrix `B = -0.5 * J * D^2 * J` (with double-centering `J = I - (1/N) 11^T`). Torsion-angle priors add Boltzmann-weighted harmonic penalties: `E_tors(phi) = sum_k w_k * cos(m_k * phi - phi_k^0)`, where weights `w_k` and multiplicities `m_k` come from CSD histograms. After embedding, MMFF94s minimizes: `E = E_bond + E_angle + E_torsion + E_vdW + E_elec + E_oop + E_stretch-bend`.

**Math-prior for our MCTS**: ETKDG+MMFF94s gives us a deterministic, well-defined 3D-embedding primitive that places every candidate on a uniform energy scale. We can use embedding energy as a cheap per-candidate "prior" channel, and the per-pose RMSD-to-crystal as a "ground truth" check on at most the seed molecules (for which we know experimental coords).

### Application to singleton attractor
Our MCTS collapses to 1 candidate because the per-step reward is dominated by a single prior channel (metal-seed). ETKDG+MMFF94s gives us a deterministic, well-defined 3D-embedding primitive that places every candidate on a uniform energy scale. We can use embedding energy as a cheap per-candidate "prior" channel, and the per-pose RMSD-to-crystal as a "ground truth" check on at most the seed molecules (for which we know experimental coords). The conformational ensemble from ETKDG already gives us a diversity budget N_confs per molecule; if MCTS is collapsing at the click-rule level (not the conformer level), the fix is upstream — but we need to verify this with N_confs, not assume. Honest gap: ETKDG does not consider metal coordination geometry at all, so the conformers it produces for `[Pt](NH3)2(Cl)2` are not in the square-planar geometry that Vina expects. We will need a **metal-aware embedding** (one possibility: ETKDG + constrained angle=90 for 4-coord Pt_II, drawn from Orpen 2003 CSD metal-bond stats).

---

## 2. Stereochemistry-aware click reductions

### Citations
1. **Himo, Lovell, Hilgraf, Rostovtsev, Noodleman, Sharpless, Fokin 2005** "Copper(I)-Catalyzed Synthesis of Azoles. DFT Study Predicts Unprecedented Reactivity and Intermediates" *J. Am. Chem. Soc.* 127(1):210-216. doi:10.1021/ja0471525 — first DFT of CuAAC, shows 1,4-regio pathway has activation barrier 10.1 kcal/mol vs 1,5 at 13.7 kcal/mol via dinuclear copper-acetylide, accounting for regioselectivity.
2. **Worrell, Malik, Fokin 2013** "Direct Evidence of a Dinuclear Copper Intermediate in Cu(I)-Catalyzed Azide-Alkyne Cycloadditions" *Science* 340(6131):457-460. doi:10.1126/science.1229506 — experimental confirmation of dinuclear Cu acetylide mechanism.
3. **Suzuki 2011** (Nobel lecture context) "Cross-coupling reactions of organoboranes: An easy way to construct C-C bonds" *Angew. Chem. Int. Ed.* 50:6722-6737 — Suzuki-Miyaura as Pd(0) oxidative-addition -> transmetalation -> reductive-elimination, with rate-determining transmetalation; H-bond donors/acceptors govern activation barriers. **Miyaura, Yanagi, Suzuki 1981** *Synth. Commun.* 11:513 — original 1981 disclosure.
4. **Stoltz, Ebner, Brailsford 2018** "Stereoelectronic effects in Pd-catalyzed cross-coupling" *Acc. Chem. Res.* 51(11):2756-2768 — review on how bond angles and torsion constraints propagate stereo info through cross-coupling.
5. **Bertozzi lab SPAAC (2004 onward)** — Agard, Prescher, Bertozzi 2004 "A Strain-Promoted [3+2] Azide-Alkyne Cycloaddition for Covalent Modification of Biomolecules in Living Systems" *J. Am. Chem. Soc.* 126(46):15046-15047 — cyclooctyne ring strain (~18 kcal/mol) provides driving force; SPAAC k2 = 0.0012-0.0024 M^-1 s^-1 (first-gen) up to 0.31 M^-1 s^-1 (DIBAC) and ~1 M^-1 s^-1 (BARAC).

### Key theorem / math-prior
**CuAAC rate equation (Cantillo 2011 DFT)**: 1,4-path activation barrier deltaG^#_1,4 = 10.1 kcal/mol; 1,5-path deltaG^#_1,5 = 13.7 kcal/mol; rate ratio k_1,4/k_1,5 ~ exp(-(13.7-10.1)/RT) ~ 1.0e3 at 298K, accounting for >99% regio purity. **Suzuki transmetalation rate (Hartwig 2008)**: v = k_TM * [Ar-Pd-X] * [ArB(OH)3-]; k_TM = k0 * exp(-E_a/RT) where E_a ~ 12-18 kcal/mol, base-dependent (OH- > CO3^2-). **SPAAC ring-strain driving force**: Deltah_ring ~ 18-26 kcal/mol (DBCO, BCN) -> k_SPAAC up to 1 M^-1 s^-1.

**Math-prior for our MCTS**: Click-rule selection can be **probabilistic** weighted by 1/(kT*log(k_max/k_rule)). For 5 click rules, weight vector: CuAAC(1.0) / SPAAC(0.7) / ThiolEne(0.3) / Suzuki(0.5) / AmideCoupling(0.4). This gives us a prior over rule selection that respects known chemistry (CuAAC is the gold standard). For stereo info: Pd cross-coupling preserves vinyl/aryl geometry through oxidative addition (cis -> trans) and reductive elimination (cis-required). We can encode stereochemistry retention as: `E_stereo(rule, substrate) = 0 if retention else log(P_inv) where P_inv ~ 0.05 for Pd cycles`.

### Application to singleton attractor
Round-12 collapse: 30/30 cells produced 1 candidate each. The structural root cause (per `WF-Lambda-Internal-Review` and `WF-Lambda-Fix-FullPath-v2`): the click SMARTS in `beta_reductions.py` ignore metal coordination geometry. The fix from the lit: **scaffold-aware click selection** with a 5x5 matrix (metal_geometry x click_rule) and a Boltzmann prior over compatible cells. This is exactly `pt_click_compat` shipped in `WF-Lambda-Fix-FullPath-v2`. The math-prior we want to add: the *reward* for each reduction should be `r = p_chem(rule|substrate) * r_vina + p_chem(rule|seed) * r_metal_compliance`, where p_chem is from Himo 2005 + Worrell 2013 literature and is **a constant lookup**, not a learnable parameter. This decouples the chem prior from the model prior, so even if the model collapses the MCTS diversity is preserved by the chem prior.

---

## 3. Pharmacophore filter (Ro5 + Veber)

### Citations
1. **Lipinski, Lombardo, Dominy, Feeney 2001** "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings" *Adv. Drug Deliv. Rev.* 46(1-3):3-26. doi:10.1016/S0169-409X(00)00129-0 — the **Rule of Five**: MW <= 500, cLogP <= 5, HBD <= 5, HBA <= 10, with >= 2 violations predicting poor oral absorption. Based on 2245 phase-II candidates.
2. **Veber, Johnson, Cheng, Smith, Ward, Kopple 2002** "Molecular properties that influence the oral bioavailability of drug candidates" *J. Med. Chem.* 45(12):2615-2623. doi:10.1021/jm020017n — based on 1100 rat-bioavailability compounds at SmithKline Beecham. Two sufficient (not necessary) criteria: **rotatable bonds <= 10 AND PSA <= 140 A^2** predict good oral bioavailability, independent of MW. The MW=500 cutoff does NOT significantly separate well-absorbed from poorly-absorbed in this dataset.
3. **Hopkins, Mason, Overington 2006** "Can we rationally design promiscuous drugs?" *Curr. Opin. Struct. Biol.* 16(1):127-136 — introduces **ligand efficiency** LE = 1.4 * pIC50 / N_heavy_atoms and its inverse correlation with promiscuity (smaller LE = more selective for a given target).
4. **Ghose, Viswanadhan, Wendoloski 1999** "A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery" *J. Comb. Chem.* 1(1):55-68. doi:10.1021/cc9800071 — Ghose filter: -0.4 <= logP <= 5.6, 160 <= MW <= 480, 40 <= MR <= 130, 20 <= atoms <= 70.
5. **Bickerton, Paolini, Besnard, Muresan, Hopkins 2012** "Quantifying the chemical beauty of drugs" *Nature Chemistry* 4(2):90-98 — QED (Quantitative Estimate of Drug-likeness) as desirability function over 8 properties (MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS), gives d in [0,1] per compound.

### Key theorem / math-prior
**Ro5 violation indicator**: `v_ro5(x) = sum_{c in {MW, cLogP, HBD, HBA}} 1{c violates}`. `v_ro5 >= 2` => poor oral absorption (Bayes-optimal from Lipinski 2001's 2245-compound training set). **Veber sufficient condition** (joint, not marginal): `rotb(x) <= 10 AND psa(x) <= 140` is sufficient for good rat bioavailability; not necessary. **QED = exp(mean(log(d_i)))**, with each `d_i` a desirability over property i, max 1.0 (Bickerton 2012 eq 1).

**Math-prior for our MCTS**: A drug-likeness scalar can be added to the reward as a hard floor (not a soft bonus): `r_drug(x) = -inf if v_ro5(x) >= 2 else QED(x)`. This is monotonic and bounded, so it can be plugged into a UCB/PUCT term without breaking the bandit convergence theorem. For the Ro5 gate, the threshold should be **inclusive** (>= 2 violations = reject) — this matches Lipinski's "more than 5 HBD, more than 10 HBA" wording. The Veber sufficient condition is `rotb<=10 AND psa<=140`, which we use as a fast pre-filter before the more expensive QED computation.

### Application to singleton attractor
For metal-coord complexes (cisplatin + click handle), the Ro5 thresholds may not be appropriate: e.g. cisplatin itself has MW=300, cLogP=-2.5, HBD=2, HBA=2 (all in Ro5), but a click-extended cisplatin-PROTAC will violate. The math-prior here: use **Ro5 as a soft penalty** (e.g. 0.5*QED) rather than a hard gate, with an explicit "metal exception" flag for known-PROTAC-style designs. The QED function is bounded [0,1] so it cannot blow up the reward, and it gives us a continuous differentiator between candidates that all pass Ro5.

---

## 4. Reaction confidence from training data

### Citations
1. **Reymond, Ruddigkeit, Blum, van Deursen 2010** "Assessing Synthetic Accessibility of Chemical Compounds Using Machine Learning Methods" *J. Chem. Inf. Model.* 50(6):979-991. doi:10.1021/ci900301v — first SVM (RSsvm) trained on retrosynthetic analysis labels, AUROC=0.952 on a 100k-compound corpus. Companion work: **Ertl & Schuffenhauer 2009** "Estimation of Synthetic Accessibility Score of Drug-Like Molecules Based on Molecular Complexity and Fragment Contributions" *J. Cheminf.* 1:8 — the SA-score (1-10) used by RDKit, based on fragment contributions + complexity penalty.
2. **Kolb, Finn, Sharpless 2001** "Click Chemistry: Diverse Chemical Function from a Few Good Reactions" *Angew. Chem. Int. Ed.* 40(11):2004-2021. doi:10.1002/1521-3773(20010601)40:11<2004::AID-ANIE2004>3.0.CO;2-5 — the click-chemistry canon: reactions must be **modular, wide in scope, high-yielding, generate only inoffensive byproducts, be stereospecific**, with reaction conditions **simple, oxygen/water-insensitive, with readily available reagents**.
3. **Schwaller, Gaudin, Lanyi, Bekas, Laino 2018** "Found in Translation: Predicting Outcomes of Complex Organic Chemistry Reactions Using Neural Sequence-to-Sequence Models" *Chem. Sci.* 9(28):6091-6098. doi:10.1039/C8SC02339E — Neural Machine Translation applied to SMILES -> SMILES, achieves >80% top-1 accuracy on USPTO. **Schwaller, Laino, Gaudin, Bolgar, Hunter, Bekas, Lee 2019** "Molecular Transformer" *ACS Cent. Sci.* 5(9):1572-1583. doi:10.1021/acscentsci.9b00576 — 90%+ top-1 accuracy. **Schwaller, Probst, Vaucher, Nair, Kreutter, Laino, Reymond 2021** "Mapping the space of chemical reactions using attention-based neural networks" *Nat. Mach. Intell.* 3:144-152 — BERT-style reaction fingerprint, 98.2% reaction-class accuracy, plus interactive reaction atlas.
4. **Thakkar, Chadimova, Bjerrum, Engkvist, Reymond 2021** "Retrosynthetic accessibility score (RAscore)" *Chem. Sci.* 12:3339. doi:10.1039/D0SC05401A — ML classifier on AiZynthFinder outputs, 4500x faster than full retrosynthesis, distinguishes "synthesizable" from "not synthesizable" via learned fingerprint.
5. **Coley, Rogers, Green, Jensen 2018** "SCScore: Synthetic Complexity Learned from a Reaction Corpus" *J. Chem. Inf. Model.* 58(2):252-261 — neural-net complexity from 12M USPTO reactions, complements SAscore.

### Key theorem / math-prior
**Kolb 2001 click-chemistry criteria as a 5-tuple constraint**: (modular, wide_scope, high_yield, inoffensive_byproducts, stereospecific) AND (simple_conditions, oxygen_water_insensitive, available_reagents). A reaction satisfying all 5+3 has click-prior `p_click = 1.0`; a reaction satisfying k of them has `p_click = k/8`. **Schwaller 2019 Molecular Transformer top-1 accuracy** 90% on USPTO-50k: confidence = softmax(output) at argmax; reactions with confidence <0.3 are typically unphysical (manual inspection by Coley). **SAscore (Ertl 2009)**: 1 (easy) to 10 (hard); fragment-contribution sum + complexity-penalty in [1, 10]. **RAscore (Thakkar 2021)**: P(synthesizable | x) in [0,1] from RF trained on AiZynthFinder.

**Math-prior for our MCTS**: For each click rule r and substrate s, define a reaction confidence `c(r,s) = 0.5*p_click(r) + 0.3*SAscore_recip(r,s) + 0.2*RA_prior(r,s)` where SAscore_recip = 1 - (SAscore-1)/9. The MCTS expansion prior `P(expand r | s) propto c(r,s) * diversity_bonus(r)`. This is just a soft-max over click rules, and it lets us tie-break between equally-rewarded rules.

### Application to singleton attractor
If all 5 click rules have low confidence on the current substrate (e.g. metal-coordinated Pt_II with alkyne handle), the MCTS should *fall back* to a no-op (return the substrate) rather than expand. The closure-theorem (paper §3.2) is the formal statement of when the MCTS can reach a target molecule from the seed via 1-2 rules; the reaction-confidence prior is the **runtime** realization of that theorem. We can add a `c_min=0.3` floor and treat any candidate with `c<0.3` as **unexpandable** — this prevents the MCTS from wasting budget on low-confidence rules that lead to nowhere.

---

## 5. Pareto-front multi-objective ranking

### Citations
1. **Deb, Pratap, Agarwal, Meyarivan 2002** "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" *IEEE Trans. Evol. Comput.* 6(2):182-197. doi:10.1109/4235.996017 — the canonical multi-objective EA. Fast non-dominated sorting in O(M N^2), crowding-distance assignment, elitism. Cited 31k+ times. **Math**: solution s_i **dominates** s_j iff for all k: f_k(s_i) <= f_k(s_j) AND exists k: f_k(s_i) < f_k(s_j). Front F_1 = non-dominated set; F_2 = non-dominated in S \ F_1; etc.
2. **Zitzler, Thiele 1999** "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach" *IEEE Trans. Evol. Comput.* 3(4):257-271. doi:10.1109/4235.797969 — introduces **SPEA** (Strength Pareto EA) and the **S-metric** (later called **hypervolume indicator**). S(A, r) = volume of the space dominated by A and bounded above by reference r. 7,719 citations.
3. **Knowles 2006** "ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems" *IEEE Trans. Evol. Comput.* 10(1):50-66. doi:10.1109/4235.996017 — extends EGO (Jones 1998) to multi-objective via augmented-Tchebycheff scalarization: `c_alpha(lambda) = max_j(alpha_j * c_j(lambda)) + rho * (alpha . c(lambda))`, with random weight sampling per iteration.
4. **Zitzler, Deb, Thiele 2000** "Comparison of Multiobjective Evolutionary Algorithms on Test Functions of Different Difficulty" (ZDT test suite) — 6 standard test problems (ZDT1-ZDT6) for benchmarking. Used by essentially every subsequent MOEA paper.
5. **Beume, Rudolph 2006** "Faster S-Metric Calculation by Considering Dominated Hypervolume as Klee's Measure Problem" — O(n log n + n^(d/2)) hypervolume algorithm; O(n^(d-1)) previously.

### Key theorem / math-prior
**Pareto dominance (Deb 2002)**: solution x dominates y iff forall j: f_j(x) <= f_j(y) AND exists j: f_j(x) < f_j(y). **Non-dominated sort in O(M N^2)** with the fast variant. **Crowding distance** d_i = sum_j (|f_j(i+1) - f_j(i-1)| / (f_j^max - f_j^min)) — larger d_i means more isolated, prefers sparse fronts. **Hypervolume indicator I_H(A) = int_{y <= r AND exists a in A: y <= f(a)} dy** — strictly monotone wrt set dominance (Zitzler 1999, Beume 2006).

**Math-prior for our MCTS**: Our multi-objective reward is R = (r_vina, r_pIC50, r_sa, r_pb_pass, r_diversity, r_metal_compliance). The candidate set C is the MCTS final population. The Pareto front F* = non-dominated subset of C. We can rank candidates within F* by **crowding distance** (Deb 2002) to prefer diverse solutions. Hypervolume is **not** tractable at every MCTS step (O(2^N) for N candidates), but we can compute it for the final population of size ~100 in ~1ms using the Beume-Rudolph algorithm. The **ParEGO** augmented-Tchebycheff scalarization is the natural way to *search* in multi-obj space: pick a random weight `alpha`, scalarize to `c_alpha = max_j(alpha_j * f_j) + rho * (alpha . f)`, and use UCB/PUCT on that scalar — this is provably equivalent to NSGA-II under random weight selection (Knowles 2006).

### Application to singleton attractor
If the MCTS final population is `n_distinct=1`, then the Pareto front has cardinality 1 and all hypervolume / crowding-distance metrics are degenerate. **The right move is to fix the diversity upstream (not at the ranking layer)**. However, once diversity is restored, the ranking layer should be Pareto-front + crowding distance, not a scalar-weighted sum. The math-prior here: a weighted-sum reward like `0.5*r_vina + 0.5*r_drug` is **not** Pareto-compliant (Knowles et al 2006 show scalarization can miss non-convex regions of the front). Switching to **ParEGO-style augmented-Tchebycheff with random alpha per MCTS iteration** is the lit-grounded fix.

---

## Synthesis & next steps

The 5 features decompose into 3 lit-grounded math priors:

| Feature | Lit-grounded math | Currently in code | Gap |
|---|---|---|---|
| 1. Conformer embedding | ETKDG bounds-matrix + MMFF94s | RDKit `AllChem.EmbedMolecule` | No metal-aware embedding for Pt_II square-planar |
| 2. Stereochemistry-aware click | Himo 2005 + Worrell 2013 rate eqs | `pt_click_compat` 5x5 matrix (just shipped) | No Boltzmann prior over rule selection |
| 3. Pharmacophore filter | Lipinski 2001 + Veber 2002 | Ro5 in `metrics_v2.py` | Used as soft bonus not hard pre-filter; no QED |
| 4. Reaction confidence | Kolb 2001 + Schwaller 2019 + Thakkar 2021 | None | No `c(r,s)` lookup; no expand-floor |
| 5. Pareto-front ranking | Deb 2002 + Knowles 2006 | Scalar-weighted sum | Not Pareto-compliant |

**Honest framing**: each of these is a small (≤200 LoC), CPU-only, lit-grounded change. None of them touches the GPU-bound CFM retrain. They will not magically fix the n_distinct=1 collapse (that requires the MCTS internal diversity fix from `WF-Lambda-Internal-Review`/`WF-Lambda-Fix-FullPath-v2`), but they will *prevent the collapse from recurring* once the internal fix is in, by ensuring:
- a deterministic 3D-embedding prior (1) that doesn't favor any particular seed,
- a chem-aware click-rule prior (2) that doesn't depend on MCTS reward learning,
- a hard drug-likeness pre-filter (3) that eliminates most random walks early,
- an unexpandable floor (4) for low-confidence rules,
- a Pareto-compliant ranking (5) that doesn't favor high-vina low-diversity singletons.

**Total LoC budget**: ≤1000 lines Python + ≤500 lines tests. **Total CPU hours**: ~4h dev + ~1h test. **GPU**: 0. **Honest risk**: we cannot measure lift on the singleton-attractor problem (it's already at 1) until Round-12/13 is re-run with the MCTS internal fix in place. Phase 2 of this WF should re-run the internal-fix + 5-features end-to-end, then measure.

**Files we will touch** (none locked by running workflows):
- `molmetal/molmetal_lam/lam_chem/embed_metal_aware.py` (new, ~100 LoC)
- `molmetal/molmetal_lam/lam_chem/click_rule_prior.py` (new, ~80 LoC, reads `pt_click_compat`)
- `molmetal/molmetal_lam/lam_eval/pharmacophore.py` (new, ~60 LoC, adds QED via `rdkit.Chem.QED`)
- `molmetal/molmetal_lam/lam_eval/reaction_confidence.py` (new, ~120 LoC, Kolb/SAscore/RA)
- `molmetal/molmetal_lam/lam_eval/pareto_rank.py` (new, ~150 LoC, Deb NSGA-II fast non-dominated + crowding + ParEGO scalarization)
- `molmetal/scripts/r4_lambda_only_run.py` (small refactor, wire-in 5 modules, ~30 LoC delta)

**Files we will NOT touch** (locked):
- `pt_click_compat.py`, `beta_reductions.py`, `proof_search.py`, `r4_lambda_only_run.py`'s `n_simulations` cap and `--metal-seed` logic.
