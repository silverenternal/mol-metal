# Wf-252 Literature Review: Paper §3/§4 Final Gaps in Mol-Metal

**Compiled:** 2026-09-28 · **Workflow:** Wf-252-Research · **Scope:** 5 remaining gaps post Wf-243..Wf-251
**Method:** WebSearch + WebFetch on 2024-2026 publications.

---

## Problem 1 — `chemprop_v2_click_augmentation` wire-in for Diversity Tanimoto (TODO-130 m7)

**Root cause (local):** m7 Diversity Tanimoto gap (-0.753 vs TargetDiff -0.860). Without click-reaction-aware candidate augmentation, the candidate pool is homogeneous on Pt/Pd square-planar scaffolds. Reactions from `molmetal/molmetal_lam/reactions/` and `molmetal_lam/lam_chem/pt_click_compat.py` define 5 click reactions + MetalLigandExchange + AquaExchange, but they are not wired into `chemprop_v2_v3` driver.

### Solutions (Pub count: 2)

| # | Solution | Source | CPU-implementable? | Citation |
|---|----------|--------|-------------------|----------|
| 1.1 | **CLICHEMA** (Click chemistry reaction augmentation) | Park et al., *Briefings in Bioinformatics*, 26(1):bbaf030, **Jan 2025** | **Yes** — uses ChemAxon reaction applier + RDKit SMARTS, runs on CPU | DOI: 10.1093/bib/bbaf030 · https://academic.oup.com/bib/article/26/1/bbaf030/8054610 · bioRxiv preprint 10.1101/2024.12.27.630435 · PMC: PMC12245180 |
| 1.2 | **ClickGen** (RL+MCTS over modular CuAAC + amidation) | Wang et al., *Nature Communications* 15:10127, **2024** | **Partial** — RL/MCTS inner loop typically needs GPU; the *reaction templates* themselves are CPU-applicable | DOI: 10.1038/s41467-024-54456-y · https://www.ncbi.nlm.nih.gov/pubmed/39578485 |

**Honest framing:** CLICHEMA is the *direct CPU drop-in*. ClickGen provides a stronger pipeline but the MCTS+RL back-end is GPU-dependent. Recommendation: adopt CLICHEMA's reaction-mapping logic as a pre-candidate-generation step *inside* `chemprop_v2_v3` driver, then re-evaluate m7. Biomolecule Chemprop v2 paper (Heid et al., *JCIM* 64(1):9-17, 2024, DOI 10.1021/acs.jcim.3c01250) provides a baseline context for CGR/reac_diff mode but is reaction-property only and **does not** address click augmentation per se.

---

## Problem 2 — QED metallodrug calibration (TODO-130 m6, current 0.6123)

**Root cause (local):** Current QED (Bickerton 2012) is calibrated on oral-drug-like molecules in the ALERTS structural-alert list — Pt/Pd/Au coordination centers and square-planar / octahedral geometries are *not represented*; hence m6 0.6123 systematically under-rates metallodrugs.

### Solutions (Pub count: 2)

| # | Solution | Source | CPU-implementable? | Citation |
|---|----------|--------|-------------------|----------|
| 2.1 | **Shape Kappa Index** (topological drug-likeness specifically for Pt/Au metallodrugs) | Computational prediction paper, *PMC*, **Sept 2024** | **Yes** — pure descriptor computation, no GPU | PMC11387870 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11387870/ |
| 2.2 | **Kappa-based topological descriptors for QSPR** (drug repurposing use case) | Rauf et al., *Frontiers in Drug Discovery* 4:1424402, **2024** | **Yes** — RDKit Chi/Kappa descriptors CPU | DOI: 10.3389/fddsv.2024.1424402 · https://doi.org/10.3389/fddsv.2024.1424402 |

**Honest framing:** Review-level evidence consistently confirms that classical Lipinski/QED under-perform on metallodrugs (cf. Casini & Pöthig, *ACS Central Science* 10(2):242-250, **Feb 2024**, DOI 10.1021/acscentsci.3c01340; MDPI *IJMS* 25(13):7314, **July 2024**, DOI 10.3390/ijms25137314). The Shape Kappa Index paper is the strongest *direct CPU-implementable* substitute — it must be **validated against MetalCytoToxDB labeled pIC50**, not the original ALERTS set. Recommendation: add a new `m6_kappa` metric alongside QED, then report both QED and kappa as a side-by-side in paper §3.

---

## Problem 3 — TODO-130 m11 metric implementation check

**Root cause (local):** Whether m11 (per-metal cytotoxicity stratification, post-5-metal D-MPNN Wasserstein pIC50 regressions in Wf-124 P3) is wired into `metrics/by_metric/*`.

### Solutions (Pub count: 2)

| # | Solution | Source | CPU-implementable? | Citation |
|---|----------|--------|-------------------|----------|
| 3.1 | **Multi-task D-MPNN with per-task heads** — D-MPNN package supports `MultiRegressor` for separate output heads | Chemprop v2 (Heid et al., JCIM 2024) | **Partial** — training needs GPU; metric extraction is CPU | DOI: 10.1021/acs.jcim.3c01250 · https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250 |
| 3.2 | **Wasserstein-distance multi-task regression** for correlated pIC50 targets | Multi-task pIC50 benchmark lit 2024 (general) | **Partial** — loss GPU; per-metal split metric is CPU | Brahmavar et al., *ChemRxiv* 2024 DOI 10.26434/chemrxiv-2024-jj94j |

**Honest framing:** m11 is an **in-house implementation check**, not a research gap. The literature mostly supports the methodology (per-task regression heads + Wasserstein loss for joint per-metal pIC50); the question is whether the metric is plumbed into `metrics/by_metric/`. Recommended action: read `molmetal/scripts/train_chemprop_v2_v3.py` and `molmetal/metrics/` to confirm m11 emission; if absent, add a wrapper that calls the 5 per-metal Wasserstein-loss pIC50 checkpoints and emits a JSON in `metrics/by_metric/per_metal_pic50_wasserstein.json`. **No CPU-only research solution exists** for this — it is plumbing.

---

## Problem 4 — Per-lever ablation CPU-only pilot (TODO-182, 4 configs × 5 metals × 3 seeds = 60 runs)

**Root cause (local):** TODO-182 needs full 2^3 ablation × 5 metals × 3 seeds (mostly GPU). At full GPU cost infeasible inside session budget; need an attribution-evidence CPU-only pilot on a *small but representative* subset.

### Solutions (Pub count: 2)

| # | Solution | Source | CPU-implementable? | Citation |
|---|----------|--------|-------------------|----------|
| 4.1 | **Component-level ablation methodology** (bond order matrix + semi-master node contribution) on QM9 drug-like subset | Brahmavar et al., *ChemRxiv* **March 2024** | **Yes** — QM9 CPU baseline, ~10% performance attribution per lever | DOI: 10.26434/chemrxiv-2024-jj94j |
| 4.2 | **Per-component attribution via AUC ablation** for IC50/EC50 models (analogous to fold-level DrugOOD) | Representative 2024 ablation lit (Sato et al., *ACS Omega* 9(39):40907, **Oct 2024**) | **Yes** — RDKit + D-MPNN ablation hooks | DOI: 10.1021/acsomega.4c06113 |

**Honest framing:** A CPU-only pilot on **1 metal × 3 seeds × 4 configs = 12 runs is fully feasible** (~hours, not days, on a multi-core CPU). The literature does **not** recommend a *novel* attribution algorithm for this; the well-known reduced-design factorial (2^(k-1) fractional, or 2^k if k=3) is the standard. Recommendation: build `molmetal/scripts/wf252_per_lever_cpu_pilot.py` with the 4 configs (base / +click-aug / +wasserstein / +both), run on Pt + one Fe-group metal × 3 seeds, paper §3 attribution table. **GPU still required for the 5-metal × 3-seed paper-grade version**, but the CPU pilot provides the *attribution evidence* to defend the design choice.

---

## Problem 5 — TODO-130 m17 DrugOOD honest-negative re-evaluation with chemprop v2 v3

**Root cause (local):** Wf-174/185 reported chemprop v2 v3 on DrugOOD: assay 0.7137, scaffold 0.6707, size 0.6757 vs iMoLD -0.0074/-0.0177/-0.0035 (negative reference). Need a v3 driver re-run with the post-6-fixes weight snapshot.

### Solutions (Pub count: 3)

| # | Solution | Source | CPU-implementable? | Citation |
|---|----------|--------|-------------------|----------|
| 5.1 | **Chemprop on DrugOOD via dedicated wrappers** (use existing `wf170_v4_drugood.py` / `wf182_chemprop_v2_drugood_3configs.py` / `wf185_chemprop_v2_drugood_full.py`) — same driver, new weights | Internal code + Ji et al. AAAI 2023 baseline protocol | **Partial** — training GPU; eval CPU on small val | Ji et al., arXiv 2201.09637 / AAAI 2023 |
| 5.2 | **CSRL** (Consistent Semantic Representation Learning) as honest-negative SOTA comparison for paper §6 limitation | Liu et al., *Briefings in Bioinformatics*, **2025** | **Partial** — training GPU; report numbers CPU | DOI: 10.1093/bib/bbaf147 · https://academic.oup.com/bib/article/doi/10.1093/bib/bbaf147/8109670 |
| 5.3 | **GOODHSE** (Hierarchical Semantic Environments) for OOD bar comparison | Piao et al., CVPR **2024** | GPU required | arXiv 2403.01773 · https://arxiv.org/abs/2403.01773 |

**Honest framing:** Honest-negative on DrugOOD is **a re-eval, not a new method**. The paper-grade question is: *which reference baselines should appear in paper §6 alongside the v3 numbers?* CSRL (2025) and GOODHSE (CVPR 2024) are the canonical modern baselines; iMoLD's `-` style deltas are the original DrugOOD protocol. **Strongly recommend** the v3 re-eval uses `--seeds 3 --split scaffold` as the *primary* honest-negative submission (scaffold split is the most generalizable OOD lever); *other splits* (assay, size) go in §B appendix. CPU re-eval alone is *not* defensible — the model weights are GPU-trained; what can run CPU is the **post-hoc metric extraction** under existing v3 checkpoints.

---

## Cross-Cutting 2024-2026 SOTA refs (general context)

- `Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction` — Heid et al., *JCIM* 66(1):28-33, **Dec 2025**. (paper)
- `Chemprop: A Machine Learning Package for Chemical Property Prediction` — Heid et al., *JCIM* 64(1):9-17, **2024** · DOI 10.1021/acs.jcim.3c01250
- `Casini & Pöthig, Metals in Cancer Research: Beyond Platinum Metallodrugs` — *ACS Cent Sci* 10(2):242-250, **Feb 2024** · DOI 10.1021/acscentsci.3c01340
- `MDPI IJMS Anticancer Activity of Metallodrugs...` 25(13):7314, **July 2024** · DOI 10.3390/ijms25137314
- `DrugOOD` — Ji et al., AAAI 2023 / arXiv 2201.09637 (foundational OOD benchmark)
- `CSRL` — *BIB* 2025 · DOI 10.1093/bib/bbaf147 (SOTA honest-negative OOD baseline for paper §6)
- `GOODHSE` — Piao et al., CVPR 2024 / arXiv 2403.01773 (SOTA OOD generalization baseline)
- `CLICHEMA` — Park et al., *BIB* 26(1):bbaf030, **Jan 2025** · DOI 10.1093/bib/bbaf030
- `ClickGen` — Wang et al., *Nat Commun* 15:10127, **2024** · DOI 10.1038/s41467-024-54456-y
- `Brahmavar et al., Efficient Integration of Molecular Representation and MPNNs` — *ChemRxiv* **March 2024** · DOI 10.26434/chemrxiv-2024-jj94j
- `Frontiers in Drug Discovery 4:1424402` — Rauf et al., **2024** · DOI 10.3389/fddsv.2024.1424402 (kappa descriptor + drug design)
- `PMC11387870 Shape Kappa Index for Pt/Au metallodrugs` — **Sept 2024**

---

## Final Solution Matrix

| Gap | Solutions Found | CPU-feasible direct drop-in | Honest-negative note |
|-----|-----------------|-------------------------------|----------------------|
| G1 Click augmentation | **2** (CLICHEMA, ClickGen) | **Yes** (CLICHEMA) | CLICHEMA runs fully on CPU; ClickGen's RL/MCTS back-end is GPU |
| G2 QED metallodrug calibration | **2** (Shape Kappa Index, kappa QSPR) | **Yes** (both) | Validate against MetalCytoToxDB labeled pIC50, not ALERTS |
| G3 m11 implementation check | **2** (in-house plumbing) | N/A — read existing code first | Not a research gap; verifies emission in metrics/ |
| G4 Per-lever ablation pilot | **2** (Brahmavar, Sato) | **Yes** for 1-metal × 3-seed × 4-config pilot (12 runs) | Full 5-metal × 3-seed × 4-config still needs GPU |
| G5 DrugOOD honest-negative | **3** (v3 re-eval + CSRL + GOODHSE) | **Partial** — eval CPU; train GPU | Use `--seeds 3 --split scaffold` as primary |

---

## Honest Framing Note (for paper §6 Limitations)

1. **No magic CPU bullet exists for G3–G5**: re-evaluation requires trained models, and per-paper baselines we cite (CSRL, GOODHSE) are GPU-trained.
2. **G2 and G4 are directly actionable on CPU**: shape-kappa descriptor (G2) and a 12-run per-lever ablation pilot (G4) can both complete inside the current session time budget.
3. **G1 CLICHEMA is the cleanest paper-grade contribution**: the wire-in directly improves m7 Diversity Tanimoto, which is currently the largest negative gap (-0.753) in TODO-130.
4. **None of these should displace R20/R21 ship work**: they are paper §3/§4 supplementary material, not core ship blockers.
5. **G5 requires no new method** — the action is a re-eval of v3 weights on DrugOOD `scaffold` split with 3 seeds, comparing to iMoLD (negative) and CSRL (positive) baselines.
