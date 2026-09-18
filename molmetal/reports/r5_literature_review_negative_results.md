# R5 Literature Review: Negative Results in Molecular ML (2022-2026)

A growing body of 2022-2026 work documents that molecular ML progress is regularly overstated. We summarize seven representative papers.

## Papers

1. **Gao et al. 2022/2024 (PMO)** — *Sample Efficiency Matters*, NeurIPS Datasets & Benchmarks. Under a 10K-oracle budget, most "SOTA" generative algorithms fail to beat their predecessors on 23 PMO tasks; trivial oracles (QED, penalized logP) reward shortcuts and do not distinguish methods. arXiv:2206.12411.

2. **Tripp 2024 (Diagnosing BO for molecules)** — arXiv:2406.07709. A correctly-tuned Bayesian-optimization baseline (priors, kernels, acquisition) becomes PMO's best method; the prior poor BO performance was largely a hyperparameter artifact, not a real ranking of methods.

3. **Wang et al. 2023 (Nature Comm.)** — *Limitations of representation learning*. Training 62,820 models on MoleculeNet shows RF/XGBoost on fixed fingerprints match or beat graph/Transformer/SMILES representation learners on most tasks; deep methods only become competitive when training data exceed ~1k examples, and even then scaffold-split performance collapses. doi:10.1038/s41467-023-41967-3.

4. **Volkov et al. 2024 (JCIM)** — *From Proteins to Ligands: Decoding DTA*. On KBA/Davis kinase sets, protein encodings (AlphaFold2, ESM-1b, Pconsc4 vs random) produce no statistically significant difference in CI/Pearson/Spearman/RMSE, while ligand perturbations do — i.e. affinity models lean on ligand shortcut features. PMID:37983381.

5. **Tossou et al. 2024 (JCIM)** — *Real-World Molecular OOD*. With deployment-style splits, MOOD scoring models lose up to 60% performance and 40% calibration, exposing a large train/test gap that random splits hide.

6. **Ektefaie et al. 2024 (Sci. Adv. / arXiv:2402.581982)** — *Spectra*. Across 18 molecular datasets (incl. protein-ligand binding) and 19 SOTA models, performance steadily falls with decreasing train/test overlap — no model is uniformly best.

7. **Montoya et al. 2025 (Chem. Sci.)** — *DNA-encoded library false negatives*. Screening data systematically labels active compounds as inactive due to barcode chemistry; ML models then appear to learn predictive chemistry while actually memorizing recurring fragments. doi:10.1039/d5sc00844a.

## Common lessons

- **Splits and oracles drive the headline.** Random splits, trivial oracles (QED, logP), and single-seed reporting inflate gains; scaffold/MOOD/temporal splits and 10K-budget protocols erase most of them.
- **Baselines are underrated.** Properly-tuned RF, XGBoost, BO, and even simple nearest-neighbour baselines routinely match "SOTA" deep models on small molecular datasets.
- **Shortcuts over science.** Models lean on ligand substructure or annotation artifacts (barcodes, recurring fragments, drug-mean memorization) rather than target chemistry.
- **Data quality > architecture.** Negatives are missing or biased (PU learning, DEL artefacts, pseudoreplication); no learner fixes upstream noise.
- **Validation is the lever.** Pre-registered splits, calibration, statistical significance testing, and blind challenges (CACHE, D3R) repeatedly change the ranking of methods.