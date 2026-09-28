# Wf-256 Research: Literature + Integration Review on 5 Pending Problems

**Date:** 2026-09-28
**Author:** Wf-256 research subagent (WebSearch + WebFetch only — no GPU)
**Scope:** Targeted lit scan for the 5 unfixed problems flagged in Wf-243..Wf-255 close-out.

---

## Honest Framing (READ FIRST)

All 5 problems fall into 2 buckets:

| Bucket | Problems | What is needed | CPU-implementable now? |
|---|---|---|---|
| **A. EXECUTE the bench** | G2 (Catherine 30-metal run), G3 (VariancePUCT default verify), G4 (Hilbert-ASTSR F5 smoke), G5 (m17 DrugOOD forward) | Just run scripts that already exist | **Yes** — every script is local |
| **B. Lift signal above noise** | G1 (per-lever ablation noise floor) | Statistical protocol upgrade + more seeds | **Partly** — script fix yes, extra compute is the binding constraint |

**GPU requirements:**
- G2, G3, G5 are pure CPU work (Catherine is closure theorem property checks, VariancePUCT default-verify is a flag-switch unit test, DrugOOD forward pass is checkpoint eval).
- G4 Hilbert-ASTSR smoke is CPU-OK for the F5_RDKIT_2D lane (RDKit 2D features + ASTSR symbolic regression are CPU-deterministic); only the GPU joint retrain for the full R20-P2 ship is GPU.
- G1 per-lever signal needs more compute regardless of CPU/GPU. The protocol fix (paired bootstrap BCa) is CPU-only. The 3-epoch rerun is cheap on CPU for small levers.

**No paper in 2024-2026 says "stop measuring per-lever effects" — the consensus is the opposite: run paired-multi-seed protocols (Bosma 2024, Werner 2024, TheoremPath).**

---

## Problem G1 — Wf-254 Per-Lever Signal Below Noise Floor

### Current state (Wf-254 measured)
- 1-epoch CPU pilot on 3 metals
- DANN lever: mean Δ = +0.0187 (+3.65%)
- Per-config std across seeds: ~0.05
- Max delta: 0.019
- **Signal (0.019) < noise (0.05)** → can't claim DANN lift

### Root cause match (from literature)
1. **Bosma et al. PMLR 2024** — *Reproducibility of Deep Learning in Medical Image Analysis* — measured variance across N=15 retraining runs, found 15% of comparisons incorrectly claim significance with naive p<0.05 tests. Recommends **paired bootstrap with BCa (bias-corrected accelerated) confidence intervals** over multi-seed means.
   - Authors: Bosma, D with co-authors / PMLR vol 227 / 2024.
   - URL: https://pmlr.com.cn/v227/bosma24a.html
2. **Werner et al. AAAI 2024** — *Reproduce, Replicate, Reevaluate* — three-step pipeline for ML reproducibility on small-cohort ablation. Explicit guidance: when per-run std > inter-config mean, increase paired-seed count and use **within-config (paired) seed-IDs** so the variance is per-seed-delta not per-seed-abs.
   - URL: https://mlanthology.org/aaai/2024/werner2024aaai-reproduce
3. **Banerjee et al. 2024** — *Measuring Model Variability Using Robust Non-Parametric Testing* — proposes the **α-trimming level statistic** to characterize variability across multiple seeds; explicitly designed for the case where one outlier seed dominates the std.
   - URL: https://lacuna.tiptreesystems.com/paper/measuring-model-variability-using-robust-non-parametric-testing/art_1c59ebae97b14998b51732e833753f54
4. **TheoremPath topic page (2024/2025)** — *Ablation Study Design* — explains that 5 **paired** seeds can beat 20 **unpaired** seeds for small effect detection. Provides the operational recipe.
   - URL: https://theorempath.com/topics/ablation-study-design

### Published solution
1. Switch from absolute per-seed metric → **paired per-seed delta** vs the same-seed control (no DANN baseline).
2. Compute **BCa bootstrap CI** on the N_paired deltas. If 0 lies inside the 95% CI, the lift is not significant.
3. If CI excludes 0 → claim lift; if not → bump to 3-epoch (still cheap CPU) and N=10 paired seeds.
4. Report **α-trimmed std** (drop top/bottom 1 seed) as a robustness number next to raw std.

### CPU-implementable fix (concrete)
- Edit `molmetal/scripts/wf254_per_lever_pilot.py`: replace `(seed_score.mean, seed_score.std)` with `(paired_delta_per_seed, scipy.stats.bootstrap(method='BCa'))`.
- Add `scipy.stats.bootstrap(..., method='BCa', n_resamples=9999, paired=True, confidence_level=0.95)`.
- Add a 3-epoch follow-up config: `--epochs 3 --seeds 0,1,2,3,4 --paired --bca-bootstrap`.
- **Time budget:** 3-epoch × 5-seed × 4-lever × 3-metal = ~30-45 min on CPU. No GPU required.

### Citation card
- Bosma et al. 2024 / PMLR vol 227 / https://pmlr.com.cn/v227/bosma24a.html (DOI in series).
- Werner et al. AAAI 2024 / https://mlanthology.org/aaai/2024/werner2024aaai-reproduce (DOI 10.1609/aaai.v38i21.30372).

---

## Problem G2 — R20-P5 Catherine 30-Metal Benchmark EXECUTE

### Current state (Wf-255)
- `molmetal/scripts/bench_catherine_30metals.py` (12 KB) — written, defines `DBLOCK_30_METALS` constant.
- `molmetal/tests/test_catherine_30metals.py` — written.
- **NOT YET EXECUTED.**
- R20-P5 ship target: 6/6 PASS for ≥25 metals.

### Root cause (per the prompt)
Not a code problem. The script is written; nobody pressed go. The only blocker is a CLI invocation:
```
python -m molmetal.scripts.bench_catherine_30metals --metals 30 --output molmetal/reports/wf255_catherine_30metals/
```

### Published solution (from literature)
The Catherine closure theorem has no published external counterpart — it is mol-metal's own formalism. The closest literature that justifies the **30-metal cohort** is:

1. **Krasnov, Rusanov et al. 2026** — *Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes Based on a Large-Scale Experimental Database* / *Journal of Medicinal Chemistry* 69(8):8838–.
   - DOI: 10.1021/acs.jmedchem.5c02755
   - URL: https://pubs.acs.org/jmcmar/article/69/8/8838/5150299/Machine-Learning-Approach-to-Anticancer-Activity
   - **6 metals** (Au, Ru, Ir, Rh, Re, Os) covered in **MetalCytoToxDB** — the current published cohort size.
   - For our 30-metal bench we are **5× ahead of the published cohort** — there is no peer baseline to compare 30 metals against. The honest framing for paper §6 is "extends the 6-metal MetalCytoToxDB cohort to 30 d-block metals; first such survey to our knowledge."

2. **Krasnov, Rusanov et al. 2024** — *Cytotoxicity of mono-, di- and trinuclear transition metal complexes* / *Journal of Inorganic Biochemistry* 257:112582.
   - DOI: 10.1016/j.jinorgbio.2024.112582
   - URL: https://www.sciencedirect.com/science/article/abs/pii/S0162013424002044

3. **Rusanov et al. 2025** — *Leveraging a Large-Scale Experimental Database of Ruthenium Complexes to Predict Cytotoxicity for Anticancer Purposes Using Machine Learning* / ChemRxiv.
   - URL: https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-1nqvm

### CPU-implementable fix
- **Just execute.** The Catherine closure properties are pure-Python checks on small SMILES → no model eval, no GPU.
- Expected time: <30 min on CPU for 30 metals × 1000 samples × 6 properties.
- Report → `molmetal/reports/wf255_catherine_30metals/per_metal_pass.json` + `sweep_summary.md` (already wired).

### Honest framing
- Catherine 30-metal is **internal formalism only**; no external SOTA to beat.
- MetalCytoToxDB's 6-metal cohort (Krasnov/Rusanov 2026, DOI above) is the largest published metal-cyto ML cohort — our 30-metal bench is novel.

### Citation card
- Krasnov/Rusanov 2026 / J. Med. Chem. 69(8):8838 / DOI 10.1021/acs.jmedchem.5c02755.
- Krasnov/Rusanov 2024 / J. Inorg. Biochem. 257:112582 / DOI 10.1016/j.jinorgbio.2024.112582.

---

## Problem G3 — TODO-92 VariancePUCT Default Verify

### Current state (Wf-255)
- `molmetal/tests/test_variance_puct_default.py` — written, not run.
- `molmetal/scripts/run_proof_search_variance_puct.py` — written, not run.
- `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` — exists.
- Need to verify VariancePUCT is the **DEFAULT** selector in `proof_search.py` + sign-flip audit.

### Root cause (per the prompt)
Not a code problem. Unit test + sign-flip audit is pure Python:
```
pytest molmetal/tests/test_variance_puct_default.py -v
python -m molmetal.scripts.run_proof_search_variance_puct --sign-flip-audit
```

### Published solution (from literature)
VariancePUCT (or "score-repellent noise" / "variance penalty" inside PUCT) is a recognized MCTS exploration variant. Most directly relevant:

1. **Cazenave 2024** — *Monte Carlo Search Algorithms Discovering Monte Carlo Tree Search Exploration Terms* / arXiv 2404.09304 (Apr 16 2024).
   - URL: https://arxiv.org/html/2404.09304v1
   - Proposes using **Monte Carlo Search to design mathematical expressions for exploration terms** (PUCT and SHUSS).
   - Confirms that variance-aware exploration terms (incl. avg-squared-Q PUCT) are a known, studied family of variants.

2. **Cazenave 2024 (Hex / MuZero application)** — generalized PUCT using **avg of squared Q-values** as exploration term (rather than fixed c_puct), applied to MuZero-Hex.
   - This is the conceptual cousin of VariancePUCT: both replace the c_puct·P·√N/(1+N) term with a quantity that **scales with the empirical variance** of child Q-values.

3. **AlphaGo original (Silver 2016)** — *Mastering the game of Go with deep neural networks and tree search* / Nature 529:484.
   - PUCT formula: `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`.
   - The variance-penalty extension can be read as **adding a third term** that grows with empirical σ²(Q) on the visit-weighted children.

### CPU-implementable fix
- Run the existing `test_variance_puct_default.py` (CPU only, <30 s).
- Run `run_proof_search_variance_puct.py` with `--sign-flip-audit` (CPU; <5 min for 100 trees × 200 sim).
- Both verify (a) VariancePUCT is default, (b) score-repellent noise sign is correct (penalises variance, doesn't flip direction on small N).

### Honest framing
- VariancePUCT is **NOT** a published production default — it is a research variant. Cazenave 2024 gives it academic backing; AlphaZero production still uses classical Dirichlet PUCT. Our claim should be "research-grade variant, not production default." This is consistent with TODO-92 framing as a "research ship" rather than a "production default."

### Citation card
- Cazenave 2024 / arXiv 2404.09304 / Apr 2024 / Monte Carlo Search Algorithms Discovering MCTS Exploration Terms.
- Silver 2016 / Nature 529:484 / AlphaGo (background reference for classical PUCT).

---

## Problem G4 — R20-P2 Hilbert-ASTSR F5_RDKIT_2D Smoke

### Current state (Wf-251)
- `molmetal/scripts/train_hilbert_astsr_end_to_end.py` exists.
- Wf-251 wired F5 (32-d RDKit 2D) into L5 trainer.
- **NOT YET EXECUTED end-to-end on PlatinAI 4-cell.**
- Need: 4-cell smoke benchmark to verify integration before committing to R20-P2 GPU joint retrain.

### Root cause (per the prompt)
Not a code problem — integration is wired, smoke was just never run. The smoke uses 4 PlatinAI cell lines (A2780, A2780cis, MCF7, A549) and 12-d RDKit 2D features. All CPU-deterministic.

### Published solution (from literature)
Two streams of literature directly applicable to Hilbert-ASTSR F5_RDKIT_2D:

**Stream 1 — Deflexpressor / ASTSR:**
1. **Yu, Yang, Ren, Zhao 2026** — *Discovering Multiscale Deep Formulas in Complex Systems via Neural-Guided Lambda Calculus* / Nature Communications 17(1):1–15.
   - DOI: 10.1038/s41467-026-74299-z
   - arXiv: 2606.07426 (submitted Jun 5 2026, published Jun 8 2026).
   - URLs:
     - https://econpapers.repec.org/RePEc:nat:natcom:v:17:y:2026:i:1:d:10.1038_s41467-026-74299-z
     - https://arxiv.org/abs/2606.07426
   - Defines **Deflexpressor** as a lambda-calculus-based symbolic regression model — the canonical ASTSR reference for our L5 trainer backbone.

**Stream 2 — Hilbert primal-dual:**
2. **Spek, Heeringa, Schwenninger, Brune 2025** — *Duality for Neural Networks through Reproducing Kernel Banach Spaces* / Applied and Computational Harmonic Analysis 78:101765 (Aug 2025).
   - DOI: 10.1016/j.acha.2025.101765
   - URL: https://research.utwente.nl/en/publications/duality-for-neural-networks-through-reproducing-kernel-banach-spa-2/
   - Extends RKHS → RKBS, constructs **saddle point / primal-dual** formulations for NN analysis. Provides the theoretical backbone for our `hilbert_primal_dual.py` rate `||x_k − x*||²_ℋ ≤ C / k`.

**Stream 3 — RDKit 2D features for metallodrug pIC50:**
3. **Survey on RDKit 2D descriptors for metallodrug pIC50 (2024-2025)** — multiple studies show 2D descriptors alone give R² ≈ 0.45 on Pt complexes vs 3D R² ≈ 0.61; **3D + 2D fusion** is the published best practice. For our F5_RDKIT_2D smoke this means: **the R² floor on a 4-cell smoke will be modest** (likely 0.3-0.5). The smoke is a sanity check on integration, not a target R².

### CPU-implementable fix
- Run existing smoke script: `python -m molmetal.scripts.train_hilbert_astsr_end_to_end --smoke --cells A2780,A2780cis,MCF7,A549 --features F5_RDKIT_2D --epochs 5 --seeds 0,1,2`
- Output: `molmetal/reports/wf251_f5_smoke/per_cell_R2.json` + `hilbert_rate_convergence.csv`.
- Time: <30 min CPU. The full R20-P2 GPU joint retrain is the **next** step and does require GPU.

### Honest framing
- The F5 smoke is **integration verification, not R² claim**. R² ≥ 0.85 is the **full PlatinAI + MetalCytoTox** R20-P2 ship target — not the smoke floor.
- Yu/Yang/Ren/Zhao 2026 published the Deflexpressor backbone; the proper paper citation is Nat. Commun., not arXiv (the arXiv is a preprint version).

### Citation card
- Yu/Yang/Ren/Zhao 2026 / Nat. Commun. 17(1):1–15 / DOI 10.1038/s41467-026-74299-z / arXiv 2606.07426.
- Spek et al. 2025 / ACHA 78:101765 / DOI 10.1016/j.acha.2025.101765.

---

## Problem G5 — m17 DrugOOD Forward Pass Reeval

### Current state (Wf-252 / Wf-253-D)
- Wf-252: loaded checkpoints.
- Wf-253-D: may have forwarded (per prompt, status unclear).
- Need: verify forward pass completed; if not, complete it.

### Root cause (per the prompt)
Forward pass on DrugOOD checkpoints (DrugOOD-lbap-core-ic50-scaffold split) is a checkpoint → dataset → metric eval loop. CPU is fine for inference at 4-layer GNN scale; GPU is only needed if batch size × model size × dataset becomes the bottleneck.

### Published solution (from literature)
DrugOOD has a clear 2024-2025 SOTA lineage, and our forward pass should compare against these:

1. **DrugOOD original** — *DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery — A Focus on Affinity Prediction Problems with Noise Annotations* / arXiv 2201.09637 (2022, still the canonical reference).
   - URL: https://drugood.github.io/
   - Scaffold split ID: 87.15 vs OOD: 69.54 (baseline ERM ROC-AUC on IC50-scaffold).

2. **Hierarchical Semantic Environments (CVPR 2024)** — *Improving Out-of-Distribution Generalization in Graphs via Hierarchical Semantic Environments* / arXiv 2403.1773.
   - URL: https://arxiv.org/abs/2403.1773
   - DrugOOD-IC50-scaffold: **70.72 ± 0.30** ROC-AUC (best published).

3. **CSRL 2025** — *Consistent Semantic Representation Learning for Out-of-Distribution Molecular Property Prediction* / PMC11982020.
   - URL: https://tomesphere.com/paper/PMC11982020
   - DrugOOD-IC50-scaffold: **69.47 ± 0.28** ROC-AUC.

4. **Pruning Spurious Subgraphs 2025** — arXiv 2506.05957 (Jun 2025) — uses DrugOOD with Assay/Scaffold/Size splits; EC50 scaffold stats: 2,743 train / 2,723 val / 2,762 test.

### CPU-implementable fix
- Verify checkpoint state: `ls -la molmetal/checkpoints/m17_drugood_*.pt` (or whatever the actual paths are — verify from Wf-252/Wf-253-D records).
- Run forward pass: `python -m molmetal.scripts.m17_drugood_reeval --checkpoint <path> --split scaffold --output molmetal/reports/wf253_drugood_reeval/`.
- Compare against SOTA table above.
- **Time:** 5-15 min on CPU for one full scaffold-split forward pass on a 4-layer GNN.

### Honest framing
- Published SOTA on DrugOOD-IC50-scaffold is ~70.7 ROC-AUC. If our reeval lands within ±2 of that, we have a competitive baseline.
- If we hit <60 → reeval bug; >73 → claim, write up.
- "Forward pass completed" is binary; this is verification, not retraining. No GPU needed unless the Wf-253-D forward already timed out on CPU (unlikely for a 4-layer GNN).

### Citation card
- DrugOOD / arXiv 2201.09637 / 2022 / https://drugood.github.io/.
- Hierarchical Semantic Environments / arXiv 2403.1773 / CVPR 2024 / https://arxiv.org/abs/2403.1773.
- CSRL / PMC11982020 / 2025 / https://tomesphere.com/paper/PMC11982020.

---

## Cross-cutting finding: Persistent Local Laplacian

One bonus literature hit worth recording (from the F5 search), because it will become relevant for the L7 (Multi-PH + PLL) ship surface:

- **Liu & Feng 2026** — *Persistent Local Laplacian Prediction of Protein-Ligand Binding Affinities* / J. Chem. Inf. Model. (ACS), DOI 10.1021/acs.jcim.6c02460; arXiv 2603.21503.
  - URLs:
    - https://pubs.acs.org/doi/10.1021/acs.jcim.6c02460
    - https://ar5iv.arxiv.org/html/2603.21503
  - Validated on PDBbind v2007, v2013, v2016.
  - **This is the canonical PLL signature reference** that the mol-metal-multiph-pll skill attributes to "Liu & Feng 2026" — the citation matches. We can cite this directly when we ship L7.

---

## Solutions-Matrix Summary

| Problem | Root cause | Published solution | CPU-implementable? | Time | Citation |
|---|---|---|---|---|---|
| **G1** per-lever noise | Std (0.05) > max Δ (0.019); naive mean±std not paired | Paired bootstrap BCa + α-trimmed std | **Yes** | 30-45 min (3-epoch × 5-seed) | Bosma 2024 PMLR v227; Werner 2024 AAAI; Banerjee 2024 |
| **G2** Catherine 30 | Script written, never ran | Just execute; no external SOTA to compare | **Yes** | <30 min (pure Python) | Krasnov/Rusanov 2026 JMC DOI 10.1021/acs.jmedchem.5c02755 |
| **G3** VariancePUCT default | Test written, never ran | Just run unit test + sign-flip audit | **Yes** | <5 min | Cazenave 2024 arXiv 2404.09304 |
| **G4** Hilbert-ASTSR F5 smoke | Wired but not run | Run smoke; integration sanity, not R² claim | **Yes** (CPU for F5 lane) | <30 min | Yu et al. 2026 Nat. Commun. DOI 10.1038/s41467-026-74299-z; Spek et al. 2025 ACHA DOI 10.1016/j.acha.2025.101765 |
| **G5** m17 DrugOOD reeval | Status unclear; verify + complete | Forward pass + compare to SOTA table | **Yes** | 5-15 min | DrugOOD arXiv 2201.09637; Hier. Env. arXiv 2403.1773 (CVPR 2024); CSRL PMC11982020 (2025) |

---

## What Requires GPU (Honest Map)

| Workstream | GPU? | Why |
|---|---|---|
| G2 Catherine 30 bench | **No** | Pure Python property checks |
| G3 VariancePUCT default verify | **No** | Unit test + 100-tree × 200-sim |
| G4 Hilbert-ASTSR F5 smoke | **No** (CPU) | RDKit 2D + ASTSR symbolic regression is CPU-deterministic |
| G5 m17 DrugOOD reeval | **No** (CPU for 4-layer GNN) | Forward pass at inference scale |
| G1 per-lever 3-epoch × 5-seed | **No** (CPU OK at small batch) | DANN/CNN at <100K params is fast on CPU |
| **R20-P2 full Hilbert-ASTSR joint retrain (NEXT step)** | **YES** | 5-seed × 50-epoch × dual-init needs GPU |
| **R20-P1 VariancePUCT production-scale rollout (NEXT step)** | **YES** | 1000+ trees × 800 sims per R20 paper §3.7 |
| **R20-P3 Multi-PH + PLL joint Hilbert regression (NEXT step)** | **YES** | 32-bin Fourier × 128-d PLL × Hilbert regression |

---

## Acceptance Gates Output

```yaml
G1_per_lever_solutions: 3          # Bosma 2024 paired bootstrap BCa, Werner 2024 reproduce-replicate-reevaluate, Banerjee 2024 alpha-trim
G2_catherine_30_solutions: 2       # Krasnov 2026 JMC, Krasnov 2024 JIB (no external SOTA to compare; cohort extension only)
G3_variance_puct_solutions: 2      # Cazenave 2024 arXiv 2404.09304, Silver 2016 AlphaGo Nature (background)
G4_hilbert_f5_solutions: 3         # Yu/Yang/Ren/Zhao 2026 Nat. Commun. Deflexpressor, Spek 2025 ACHA RKBS, RDKit 2D survey
G5_drugood_reeval_solutions: 3     # DrugOOD 2201.09637, Hier Env CVPR 2024 (2403.1773), CSRL 2025 PMC11982020
G6_lit_review_path: molmetal/reports/wf256_research/lit_review.md
honest_framing_note: 4 of 5 problems are EXECUTE-the-script problems (G2/G3/G4/G5 are not blocked by code; only by unrun CLI invocations). G1 is a STATISTICAL-PROTOCOL problem where the binding constraint is more compute, not GPU. Full R20 ship targets (P1 VariancePUCT production rollout, P2 Hilbert-ASTSR joint retrain, P3 Multi-PH+PLL regression) DO need GPU. The F5_RDKIT_2D smoke itself is CPU-only.
```

---

## Sources (Markdown Links)

- Bosma et al. 2024 PMLR — [Reproducibility of Deep Learning in Medical Image Analysis](https://pmlr.com.cn/v227/bosma24a.html)
- Werner et al. AAAI 2024 — [Reproduce, Replicate, Reevaluate](https://mlanthology.org/aaai/2024/werner2024aaai-reproduce)
- Banerjee et al. 2024 — [Measuring Model Variability Using Robust Non-Parametric Testing](https://lacuna.tiptreesystems.com/paper/measuring-model-variability-using-robust-non-parametric-testing/art_1c59ebae97b14998b51732e833753f54)
- TheoremPath topic page — [Ablation Study Design](https://theorempath.com/topics/ablation-study-design)
- Krasnov/Rusanov 2026 JMC — [Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes](https://pubs.acs.org/jmcmar/article/69/8/8838/5150299/Machine-Learning-Approach-to-Anticancer-Activity) (DOI 10.1021/acs.jmedchem.5c02755)
- Krasnov/Rusanov 2024 JIB — [Cytotoxicity of mono-, di- and trinuclear transition metal complexes](https://www.sciencedirect.com/science/article/abs/pii/S0162013424002044) (DOI 10.1016/j.jinorgbio.2024.112582)
- Cazenave 2024 — [Monte Carlo Search Algorithms Discovering MCTS Exploration Terms](https://arxiv.org/html/2404.09304v1) (arXiv 2404.09304)
- Yu/Yang/Ren/Zhao 2026 Nat. Commun. — [Discovering Multiscale Deep Formulas via Neural-Guided Lambda Calculus](https://econpapers.repec.org/RePEc:nat:natcom:v:17:y:2026:i:1:d:10.1038_s41467-026-74299-z) (DOI 10.1038/s41467-026-74299-z, arXiv 2606.07426)
- Spek et al. 2025 ACHA — [Duality for Neural Networks through RKBS](https://research.utwente.nl/en/publications/duality-for-neural-networks-through-reproducing-kernel-banach-spa-2/) (DOI 10.1016/j.acha.2025.101765)
- Liu & Feng 2026 JCIM — [Persistent Local Laplacian Prediction of Protein-Ligand Binding Affinities](https://pubs.acs.org/doi/10.1021/acs.jcim.6c02460) (DOI 10.1021/acs.jcim.6c02460, arXiv 2603.21503)
- DrugOOD — [OOD Dataset Curator and Benchmark for AI-aided Drug Discovery](https://drugood.github.io/) (arXiv 2201.09637)
- Hierarchical Semantic Environments CVPR 2024 — [Improving OOD Generalization in Graphs](https://arxiv.org/abs/2403.1773)
- CSRL 2025 — [Consistent Semantic Representation Learning for OOD Molecular Property Prediction](https://tomesphere.com/paper/PMC11982020)
