# Provenance: DiffDock-L / FLOWR / FlowDock

## TL;DR — DiffDock-L (Corso 2024)
ICLR 2024 paper "Deep Confident Steps to New Pockets: Strategies for Docking Generalization" is the actual DiffDock-L paper (arXiv 2403.05784 = prior submission; ICLR 2024 PDF). Original DiffDock (ICLR 2022, arXiv 2210.01776) reports **38.2% top-1 RMSD<2Å on PDBBind time-split (n=363)** with **40 samples** + 20 inference steps. DiffDock-L with **10 samples / 20 steps** lifts this to **43.0% top-1** (median RMSD 2.8Å) on the same PDBBind split, and **22.6% on DockGen-full, 27.6% DockGen-clusters** (unseen ECOD-domain pockets). **PoseBusters set: 50% top-1 RMSD<2Å** vs DiffDock 38% / RFAA 42%. Trained on PDBBind pre-2019 (~17k) + Binding MOAD (+20k) + van der Mer synthetic pockets; SE(3) tensor-field network ~30M params (up from 20M). Runtime ~25 s/complex on A100.

## TL;DR — FLOWR (Cremer 2024)
**WARNING on arXiv ID**: 2404.02819 does NOT resolve to FLOWR. FLOWR's true arXiv is **2504.10564** (Nat Comp Sci 2026, v3 Apr 2025). The ID 2404.02819 is closest to PILOT (arXiv 2405.14925, Chem Sci 2024). FLOWR is the flow-matching follow-up to PILOT. On **CrossDocked2020 (100 pockets × 100 ligands)**: Vina mean **−6.29±1.56**, Vina_min **−6.48±1.45**, PB-validity **0.92±0.22**, strain 87.83±74.30, inference **12.05±8.01 s**. On the curated **SPINDR** (PLINDER-derived, 35,666 complexes): Mean Vina **−6.93±0.92**, Vina_min **−7.22±0.92**, PB-validity **0.88±0.21** (rises to 0.95 after MMFF94s relax), RDKit-validity **0.94±0.24**, inference ~70× faster than PILOT at 20 steps. Key protocol deviation from PILOT: **pocket encoder is computed once** (single forward), vs PILOT/TargetDiff/DiffSBDD recomputing each denoise step.

## TL;DR — FlowDock (Cremer 2024)
**WARNING on arXiv ID**: 2403.04777 does NOT resolve to FlowDock. FlowDock's actual arXiv is **2412.10966** (Oxford Bioinformatics ISMB 2025). On **PoseBusters (n=308)** with apo inputs: **51% blind RMSD<2Å** after energy minimization, vs AF3-single-seq 41% / NeuralPLexer 41% / Chai-1 56%. On **DockGen-E (n=122)**: ties with Chai-1-single-seq; best ~33% success. **PDBBind 2020 (n=363) affinity**: Pearson **0.705**, Spearman **0.674** (beats DynamicBind 0.665/0.634). **Runtime on Astex 25%: 39.34 s, GPU 25.61 GB, CPU 11.98 GB** — 2nd-lowest runtime among DL methods; 3× faster than DynamicBind; trained 1 week on 4×H100 (32× less GPU-hr than AF3/Chai-1). CASP16 ligand category: top-5 affinity across 140 complexes. **Important caveat** (Pith referee report): Eq.(2) regresses v_θ to x1 (endpoint), not (x1−x0) — sampling update Eq.5 is NOT a true CFM Euler step; apo prior is ESMFold surrogate (TM-score ≥0.7, RMSD <5Å filter), not unbound ensemble.

## 5-line core numbers per model

### DiffDock-L (arXiv 2403.05784 / ICLR 2024, "Deep Confident Steps")
- Dataset / split: **PDBBind time-split 2019 / 363 test**; train ~17k pre-2019 + Binding MOAD 20k + vdM synthetic
- Metric & n_test: **RMSD<2Å top-1 = 43.0%** (PDBBind 363); **22.6% DockGen-full**; **50% PoseBusters (308)**
- NFE / sampling: **10 samples × 20 inference steps = 200 NFE** (vs DiffDock 40 samples → 800 NFE for 38.2%)
- Median RMSD: **2.8 Å** (PDBBind), 4.3 Å (DockGen-full); runtime **~25 s/complex A100**
- Protocol delta vs Lambda 1h36-only: DiffDock-L is **rigid-receptor, holo-train**, no confidence-bootstrap inference (training-only on DiffDock-S); pose-confidence ranking **after** sampling; **no per-step pocket resampling** like Lambda's pocket-encoder flow

### FLOWR (arXiv 2504.10564 / Nat Comp Sci 2026)
- Dataset / split: **SPINDR (PLINDER-derived, 35,666 complexes, no leakage)** primary; CrossDocked2020 secondary (100 pockets × 100 ligands)
- Metric & n_test: **Mean Vina −6.93±0.92** (SPINDR 225 targets × 100 lig); **PB-valid 0.88→0.95 (MMFF)**, RDKit-valid 0.94
- NFE / sampling: **flow matching, 20–100 Euler steps**; default 100; pocket encoder **1× forward** (vs PILOT/DiffSBDD recompute)
- Inference: **12.05 s** (CrossDocked) → ~70× faster than PILOT at 20 steps; **SPINDR test: ~similar**
- Protocol delta vs Lambda 1h36-only: FLOWR = **de novo ligand generation**, not pose prediction; continuous 3D-coord flow + discrete atom-type/bond flow; no Lambda-style 1h36-only restriction; uses **equivariant optimal transport** CFM

### FlowDock (arXiv 2412.10966 / Oxford Bioinformatics ISMB 2025)
- Dataset / split: **PDBBind-E** (PDBBind 2020 + Binding MOAD + ESMFold apo priors, TM≥0.7 & RMSD<5Å); test on **PoseBusters 308, DockGen-E 122, PDBBind 363, Astex 25%**
- Metric & n_test: **51% PB RMSD<2Å** (308); **Pearson 0.705 / Spearman 0.674** affinity on PDBBind 363
- NFE / sampling: **40 ODE steps** (variance-diminishing update Eq.5); single-pass per step
- Runtime: **39.34 s/complex** (Astex), **GPU 25.61 GB**, **CPU 11.98 GB** on A100 80GB; train 4×H100 × 1 week
- Protocol delta vs Lambda 1h36-only: FlowDock = **apo→holo mapping with sidechain flex** (TM-score/RMSD filter on training pairs); hybrid **structure + affinity**; CASP16-validated; equation (2) is NOT vanilla CFM target — endpoint predictor with custom integrator