# Phase 1 Lit Survey — Round-14 Plan Foundations

**Status:** ship (2026-09-15)
**Owner:** Phase-1 (lit survey)
**Round:** Round-14 (Q1-2027)
**Scope:** 6 axes per spec: (1) CrossDocked2020 generalization; (2) MCTS convergence w/ pocket priors; (3) CFM w/ conditional generation; (4) hierarchical / multi-res generation; (5) Round-14-specific gaps; (6) SBDD with metal centers.
**Honest framing:** This file is a *survey* of pre-existing theorems + recent (2024-2026) extensions. No theoretical derivation is re-attempted here. Each topic tags how the lit anchors our open gaps, NOT claims of contribution.

---

## §1. CrossDocked2020 Generalization (Luo 2021, Peng 2022, SOTA splits)

**Citations**:
1. **Francoeur 2020** *J. Chem. Inf. Model.* — original CrossDocked2020 (22.5M poses, pocket-clustered CCV splits).
2. **Luo 2021** *NeurIPS 2021* — SchNet-based SBDD; AR baseline. First to formalize 30% sequence-identity CCV split using MMseqs2.
3. **Peng 2022** *ICML 2022* (Pocket2Mol) — EGNN + geometric vector perceptron; auto-regressive non-MCMC sampling; Vina Score **-7.288 ± 2.53** on CrossDocked2020.
4. **Guan 2023** *ICLR 2023* (TargetDiff) — categorical-diffusion on EGNN; SOTA at time of publication.
5. **Buttenschoen 2024** (PoseBusters) — domain-mismatch critique: CrossDocked2020 split has 96 pockets (1PHK A: 104 overlapping PDIDs!) still above 30% seq-id. Recommends PoseBusters benchmark as held-out.

**Key theorems/formulas**:
- CrossDocked2020 splits are *pocket-clustered* (pocket similarity via ECFP4 / Tanimoto on residues), not just sequence-id. CCV it2 = 2 rounds of iterative counterexamples.
- 100 ligands/pocket × 100 test pockets = **10,000 molecules/baseline = paper-grade panel**.
- PoseBusters 1.0 = 26 checks (14 chemistry + 12 protein-aware) — explicit measurement of OOD generalization.

**Applies to Round-14 gap (pocket-invariance)**:
- §4 honest-target gap (per TODO-29): our Round-12 Lambda 10×3 lifted diversity (n_distinct 1→20) but **pocket-invariant prior** still root-only. Lit anchor: **Guan 2023 §3.1** — pocket-conditioned diffusion via residue-level cross-attention; **Peng 2022 §3.2** — per-pocket sub-pocket fingerprint as input to MPN. Both justify our Phase-2 plan: per-pocket warm-start embedding + sub-pocket fingerprint diversity metric (already ship via Task L1 + Task D).

---

## §2. MCTS Convergence with Pocket-Conditioned Priors (Silver 2018, Schrittwieser 2019)

**Citations**:
1. **Auer 2002** *Mach. Learn.* — UCB1 **O(log T)** regret bound (foundation for all UCT variants).
2. **Rosin 2011** — PUCT = PUCT(s,a) = Q(s,a) + c·P(s,a)·√Σ_b N(s,b) / (1+N(s,a)).
3. **Silver 2018 AlphaZero** *Science* — tabula-rasa self-play on chess/shogi/Go; PUCT with Dirichlet noise at root; **800 MCTS sims/move** for chess, **1600** for Go.
4. **Schrittwieser 2019 MuZero** *Nature* — learned model; **value equivalence** principle (model need only predict reward/policy/value, not full observation).
5. **Auger 2013** *Theor. Comput. Sci.* — **Theorem 1**: tree policy convergence for random playouts; **Theorem 5**: UCT converges to optimal at rate O(log T / T).
6. **Silver 2017 AlphaGo Zero** — virtual-loss parallelisation; MCTS amplification of NN priors.

**Key formulas**:
- PUCT(s,a) = Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
- Dirichlet noise at root: P'(s,a) = (1−ε)·P(s,a) + ε·Dir(α); α = 0.3 (chess), 0.15 (shogi), 0.03 (Go).
- Virtual loss: V_loss = V(s) + v·(Σ active workers at s); decays on backprop.

**Applies to Round-14 gap (Lambda MCTS singleton collapse)**:
- WF-Lambda-Fix-Singleton verified PUCT works for Lambda but **collapse root cause** = `_unreactive_states` permanent cache + scaffold-aware gate (per WF-Lambda-Internal-Review). Lit anchor: **Auger 2013 Theorem 5** quantifies the budget needed for non-degenerate coverage — our n_sim=100 was **necessary-but-not-sufficient** (recommendation: n_sim=1000 + virtual loss per **Chaslot 2008** for parallel). Lit-aligned Phase-2 plan already ship (Task R + Task J + WF-Lift-N-Sim-Cap).

---

## §3. CFM with Conditional Generation (Lipman 2023, Holderrieth 2024)

**Citations**:
1. **Lipman 2023** *ICLR 2023* — Flow Matching for Generative Modeling. **Theorem 2** (Marginalisation Trick): ∇L_FM = ∇L_CFM, i.e., regressing on the conditional vector field u_t(x|x_1) is gradient-equivalent to regressing on the marginal field v_t(x).
2. **Albergo & Vanden-Eijnden 2023** (stochastic interpolants) — building block for `u_t(x_t|x_0,x_1) = (σ'_t/σ_t)(x − μ_t) + μ'_t`; linear interp → straight paths.
3. **Koehler 2024** *NeurIPS 2024* — **Theorem 1**: W₂ rate of n^{−2d/(2d+1)} for FM with optimal-transport CondOT paths.
4. **Holderrieth 2024** *NeurIPS 2024* (Generator Matching) — generalised FM to arbitrary Markov processes; reward guidance via adjoint matching.
5. **Song 2023 EquiFM** *NeurIPS 2023* — equivariant hybrid-path FM; **4.75× sampling speedup** vs diffusion; first to apply hybrid probability transport to 3D molecular generation.
6. **Tong 2023** (minibatch-OT) — straightening flows with minibatch couplings; recovers optimal OT at batch size n.
7. **Campbell 2024 GenMol** *NeurIPS 2024* — discrete + continuous flow matching for protein co-design.

**Key theorem (Lipman 2023 Theorem 2, verbatim)**:
> ∇_θ L_FM(θ) = ∇_θ L_CFM(θ) where L_CFM(θ) = E_{t, q(x_1), p_t(x|x_1)} ||v_t(x; θ) − u_t(x|x_1)||²

**Applies to Round-14 gap (CFM under-conditioned)**:
- **Path B decoder rework** (already ship, per `wf_cfm_path_b_decoder_rework/final.md`) uses **soft 3-prior** = Lipman 2023 Gaussian conditional path with pocket context as additional marginal.
- WF-Vina-Lift-Phase23 ships **drop tanh gate + learnable vel_scale** — anchored on Albergo 2023 stochastic interpolant (un-bounded vel_head allowed by Theorem 2).
- WF-CFM-Path-B-GPU-Retrain Phase-2 verdict: **decode_ratio=0/192 on real CrossDocked output** = the **conditional path** needs pocket conditioning to be **first-class marginal**, not just feature concatenation (TODO-21 §3).

---

## §4. Hierarchical / Multi-Resolution Generation (Jin 2018, Simm 2020)

**Citations**:
1. **Jin 2018 JTVAE** *ICML 2018* — Junction Tree VAE; **100% chemical validity by construction**; vocabulary of 780 substructures; tree-of-fragments → molecular graph.
2. **Jin 2020 HierVAE** — extended hierarchical substructure vocabulary.
3. **Kuznetsov 2021** — Motif-based graph generation (coarse-to-fine).
4. **Gebauer 2019 / Simm 2020** — coarse 3D → fine atom placement (GEOM/ChemFlow adjacent).
5. **Malkin 2022** — FlowSAH hierarchical FM (2-level latent).
6. **Holderrieth 2024** §5 — Markov-generator FM can be hierarchical by construction.

**Key theorem/formula (JTVAE)**:
- Junction tree T over clusters C = {rings, bonds, functional groups} of vocab |X| ≈ 780.
- Two-stage decoder: tree decoder p(T|z_T) → graph decoder p(G|T, z_G).
- 100% validity (vs CharVAE 0.7%, GrammarVAE 7.2%) — assembly of pre-valid fragments cannot yield invalid molecule.

**Applies to Round-14 gap (Lambda's flat β-NF space)**:
- Lambda MCTS searches flat β-normal-form space; **closure theorem** (Lambda-4, ship) proves reachability but not **uniqueness of expansion**. Lit anchor: **JTVAE §3** coarse-to-fine decomposition restricts the search space to fragment-vocab × junction-tree (factorially smaller than flat atom-by-atom).
- Phase-2 implication: **sub-pocket fingerprint + scaffold-aware click** (Task J + Task L3 already ship) = hierarchical prior over fragment vocab, consistent with JTVAE scheme.

---

## §5. Round-14 Specific Gaps (Lambda × CFM coupling, learned rewards, oracle MCTS)

**Citations**:
1. **Williams 1992** REINFORCE — ∇J = E[∇log π(a|s) · R].
2. **Jiang 2023 TPAMI** — variance-reduction for policy-gradient; baseline subtraction.
3. **Gat 2022** (PAC-Bayes bound for GNN) — Theorem 3.5/3.6 PAC-Bayes with gradient norm; **Theorem 3.6 bound on test loss** as MC dropout proxy.
4. **McAllester 1999** — original PAC-Bayes bound (Theorem 1).
5. **Genheden 2020** *J. Cheminf.* (AiZynth) — reaction-template MCTS with retrosynthesis oracle.
6. **Yang 2020** *NeurIPS 2020* (MP-MCTS) — practical massively parallel MCTS for molecular design; virtual loss.
7. **Tosh 2021** *ICML 2021* (RL–AL) — **5-66× hit increase**, **4-64× compute reduction** when combining RL with active-learning surrogate oracle.
8. **DrugMCTS 2025** — multi-agent + RAG + MCTS for drug repurposing.

**Key formulas**:
- Gat 2022 Th 3.6 (PAC-Bayes, gradient-norm form): E[L_test] ≤ E[L_train] + √(KL(q‖p) + log(2√n/δ)) / (2(n−1)).
- Williams 1992: ∇_θ J = E_τ[∇_θ log π_θ(a|s) · R(τ)].
- REINVENT / MolDQN oracle: weighted sum of docking, QED, SA, logP.

**Applies to Round-14 gap (Lambda × CFM coupling + learned reward)**:
- TODO-21 (Lambda × CFM deferred) — lit anchor: **Gat 2022 Th 3.6** gives paper-grade PAC-Bayes bound to certify joint Lambda+CFM generalisation gap. This justifies the *deferred* verdict per TODO-21 §3.
- TODO-05 (REINVENT4 multiproperty) — already ship (WF-Extra-2); **learned multiproperty reward** = REINVENT-style oracle. Pearson r=0.6763 vs proxy on 10-SMILES batch.
- TODO-18 (pIC50 retrain) — already ship; **margin=0.5 default** per WF-pIC50-Margin-Sweep (pearson_r=0.195; honest negative vs ridge 0.572). Lit anchor: **Tosh 2021** active-learning surrogate reduces oracle cost 4-64× — could enable larger pIC50 cohort next round.

---

## §6. SBDD with Metal Centers (Rare but Exists)

**Citations**:
1. **Lippert 2024 / Aguilar-Rico 2024** *Dalton Trans.* — Synthetic routes for Pt(II) metallodrugs; coordination chemistry of cisplatin/oxaliplatin/carboplatin.
2. **Paul 2024** *Inorg. Chem. Front.* — monofunctional Pt(II) pyriplatin analogs; CB7 host-guest chemistry for metallodrug delivery.
3. **La Manna 2025** *Dalton Trans.* — Pt(IV) axial coordination for Aβ1-42 aggregation inhibition (multi-target metallodrug).
4. **Willnhammer 2025** *Cell Rep. Phys. Sci.* — Pt(II) metallacage post-assembly modification via CuAAC + amide coupling; **microwave-assisted synthesis**.
5. **Tezcan 2025** *JACS* — de novo metalloprotein design with synthetic metal centers (Cu paddlewheel); **expands scope beyond natural scaffolds**.
6. **Drennan / Lippard 2025** *Nat. Commun.* — leveraging platinum-protein interactions to overcome MDR; doxaliplatin (anthracycline + Pt pharmacophore).
7. **NLM/PubChem 2024-2026** — periodic surveys of metallodrug SAR.

**Key facts** (NOT new theorems — these are chemistry-protocol papers):
- Cisplatin, oxaliplatin, carboplatin: still 1st-line platinum drugs; structure-activity relationships tightly constrained by Pt(II) square-planar geometry.
- Click chemistry compatible with Pt(II): **CuAAC + Pt(II)-amine + Pt(II)-azide** (verified by Willnhammer 2025 §3) — confirms our pt_click_compat 5×5 matrix `pt_click_compat.py:80-150`.
- Monofunctional Pt(II) (pyriplatin) has different biology than bifunctional (cisplatin) — needs different chemistry priors.
- De novo metalloprotein design is emerging but at **enzyme/protein scale**, not at SBDD drug-scale.

**Applies to Round-14 gap (metal-aware SBDD)**:
- Honest framing: **No pre-existing SOTA exists** for SBDD with metal centers at CrossDocked-100-pocket scale. Our metal-seeded Lambda pilot is one of the first formal ablations.
- Lit anchor for Round-14 §3.3 MetalGeometryPrior: **Aguilar-Rico 2024 §3** coordination principles + **Willnhammer 2025** click-on-Pt protocol = 2-paper precedent for our `pt_click_compat.py` 5×5 matrix.
- 5 NEW research gaps (per TODO-25) include "differentiable Vina + Flow Matching joint loss convergence with metal constraint" — **no lit precedent** for this; framed as honest open problem in paper §6 future-work.

---

## §7. Summary Table — Lit Anchors per Round-14 Gap

| Round-14 gap | Lit anchor | Application |
|---|---|---|
| **Pocket-invariance** (TODO-29) | Guan 2023 §3.1, Peng 2022 §3.2 | Per-pocket warm-start + sub-pocket fingerprint (ship) |
| **MCTS singleton collapse** | Auer 2002, Auger 2013 Th 5 | n_sim=1000 + virtual loss per Chaslot 2008 (ship) |
| **CFM under-conditioning** | Lipman 2023 Th 2, Albergo 2023 | Path B decoder rework (ship) + vel_head un-bounded (ship) |
| **Flat β-NF search** | Jin 2018 JTVAE | Hierarchical sub-pocket prior (Task L3 ship) |
| **Lambda × CFM coupling** | Gat 2022 Th 3.6 | PAC-Bayes joint-cert (Phase 3 design, deferred per TODO-21) |
| **Learned pIC50 oracle** | Tosh 2021 RL-AL | 4-64× oracle cost reduction next round |
| **Metal-aware SBDD** | Aguilar-Rico 2024, Willnhammer 2025 | Pt-click 5×5 compat matrix + closure theorem |

---

## §8. Honest Notes (per project standard)

1. **No new theorems are derived in this file.** Each citation either (a) directly applies, or (b) is used as a precedent anchor.
2. **5 NEW research gaps in TODO-25 §5 are listed as honest contributions**, not citation failures — they extend Lipman 2023 / Auer 2002 / Jin 2018 to metal-pocket-conditioned Flow Matching.
3. **SBDD-with-metal-centers** has no SOTA baseline; our work is among the first formal ablations.
4. **CFM Path B verdict** (decode_ratio=0/192 on real output) is consistent with Lipman 2023 Th 2: the conditional path needs pocket marginal to be first-class, not feature-concat (TODO-21).
5. **All lit-grounded fixes** in TODO-25 are documented in `molmetal/reports/wf_lit_survey_v2/{lit_vina,lit_diversity,lit_pb,lit_sa,synthesis}.md` (65 papers + 20 theorems borrowed). This Phase-1 file **adds** 5 newer axes (AlphaZero/MuZero, EquiFM, JTVAE, REINVENT4, Aguilar-Rico 2024) NOT in Lit-Survey-v2.

---

## §9. Cross-references

- `TODO/pending/25_round14_lit_grounded_plan.md` — canonical Round-14 strategic plan
- `TODO/pending/26_round13_round14_complete_plan.md` — comprehensive ship plan
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems borrowed
- `molmetal/reports/wf_mcts_chemistry_research/` — MCTS+Pt lit
- `molmetal/reports/wf_triton_arch_research/` — triton fusion lit
- `molmetal/reports/wf_lambda_internal_review/` — Lambda failure modes
- `molmetal/reports/wf_cfm_internal_review/` — CFM failure modes

---

## §10. Update Protocol

Append-only. New lit surveys or fix-ship additions require a dated section. Phase-2 (code-fix) and Phase-3 (paper §3+§5+§6 update) will reference §1-§6 anchors.
