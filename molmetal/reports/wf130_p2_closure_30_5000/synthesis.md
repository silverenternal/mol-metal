# Wf-130 Phase 2 — Closure reachability 30 × 5000 expansion

Lifts Wf-67 P2 (0.5891 on 30 × ~75 unique) + Wf-68 P2 (0.5744 on 25 × ~180 unique) toward the gate target **≥ 0.85** on **30 metals × 5000 samples each = 150000 requested**, sourced from the full tmQM parsed CSV.  Metals with fewer unique SMILES are sampled **with replacement** (seeded).

## TL;DR

- **Macro reachability** (mean of per-metal rates, equal-weight): **0.5589** ± 0.1526 (std-of-means across metals)  
- **Micro reachability** (sum closure_ok / sum n_samples, weighted by sample count): 0.5589  
- **Overall gate pass (≥ 0.85)**: False  
- **Metals at gate**: 0/30  
- **Metals below honest-negative threshold (0.50)**: 10/30  
- **Metals sampled with replacement**: 25/30  
- **N samples (overall)**: 150000  
- **N unique SMILES (overall)**: 97397  
- **N repeats (overall)**: 70164  
- **Total wall**: 1043.83 s  

## Baselines

- **Wf-67 P2 baseline**: 0.5891 (2244 unique samples × 30 metals).
- **Wf-68 P2 baseline**: 0.5744 (4497 unique samples × 25 metals).
- **Wf-130 P2 measured (macro)**: 0.5589  (30 metals × 5000 samples each, with-replacement where required).
- **Delta vs Wf-67**: -0.0302
- **Delta vs Wf-68**: -0.0155

## Per-metal breakdown

| Metal | N | unique | repeats | closure_ok | reachability | gate |
|-------|---|--------|---------|------------|--------------|------|
| Sc | 5000 | 457 | 4543 | 2072 | 0.4144 | False |
| Ti | 5000 | 3098 | 1902 | 2927 | 0.5854 | False |
| V | 5000 | 1325 | 3675 | 898 | 0.1796 | False |
| Cr | 5000 | 1998 | 3002 | 3505 | 0.7010 | False |
| Mn | 5000 | 1956 | 3044 | 3440 | 0.6880 | False |
| Fe | 5000 | 4921 | 79 | 3464 | 0.6928 | False |
| Co | 5000 | 3161 | 1839 | 3033 | 0.6066 | False |
| Ni | 5000 | 9706 | 0 | 3460 | 0.6920 | False |
| Cu | 5000 | 3323 | 1677 | 3839 | 0.7678 | False |
| Zn | 5000 | 6808 | 0 | 4090 | 0.8180 | False |
| Y | 5000 | 810 | 4190 | 2560 | 0.5120 | False |
| Zr | 5000 | 2614 | 2386 | 2404 | 0.4808 | False |
| Nb | 5000 | 512 | 4488 | 1699 | 0.3398 | False |
| Mo | 5000 | 3794 | 1206 | 1701 | 0.3402 | False |
| Tc | 5000 | 275 | 4725 | 2177 | 0.4354 | False |
| Ru | 5000 | 8734 | 0 | 3260 | 0.6520 | False |
| Rh | 5000 | 4151 | 849 | 3081 | 0.6162 | False |
| Pd | 5000 | 9839 | 0 | 2540 | 0.5080 | False |
| Ag | 5000 | 1538 | 3462 | 3021 | 0.6042 | False |
| Cd | 5000 | 1621 | 3379 | 3833 | 0.7666 | False |
| La | 5000 | 114 | 4886 | 4000 | 0.8000 | False |
| Hf | 5000 | 564 | 4436 | 2540 | 0.5080 | False |
| Ta | 5000 | 739 | 4261 | 2202 | 0.4404 | False |
| W | 5000 | 3156 | 1844 | 2596 | 0.5192 | False |
| Re | 5000 | 3895 | 1105 | 2808 | 0.5616 | False |
| Os | 5000 | 1204 | 3796 | 2379 | 0.4758 | False |
| Ir | 5000 | 4551 | 449 | 1795 | 0.3590 | False |
| Pt | 5000 | 7474 | 0 | 2674 | 0.5348 | False |
| Au | 5000 | 3925 | 1075 | 2418 | 0.4836 | False |
| Hg | 5000 | 1134 | 3866 | 3416 | 0.6832 | False |

## Honest framing

- **CPU-only.** Closure verifier is pure-functional; no GPU dependency.
- **Deterministic.** Same flags reproduce every per-sample metric; with-replacement RNG is seeded (`sample_seed=42`).
- **Real data.** Full tmQM parsed CSV (108K records, Balcells & Skjelstad 2020); the Wf-28 d-block subset was a 5K sample drawn from this CSV.
- **Sampling strategy.** Each metal receives exactly 5000 samples.  Metals with 5000+ unique SMILES take the first 5000 (deterministic).  Metals with fewer unique SMILES (notably Tc, La, Sc, Nb, Hf, Ta) are sampled with replacement (seeded) so every metal has the same sample budget — this is the standard honest sampling strategy; per-metal JSON files carry both `n_unique` and `n_repeats` so the with-replacement behaviour is transparent.
- **Macro vs micro reachability.** Macro = equal-weight mean of per-metal rates (this is the gate definition).  Micro = sum(closure_ok)/sum(n_samples) (weighted by sample count, so well-sampled metals like Pd/Ni dominate). Both are reported.
- **Honest-negative threshold.** Any metal with reachability < 0.50 is flagged. At-risk metals for honest-negative framing: Tc (only 283 unique SMILES), La (only 116 unique — lanthanide), Sc (459), Nb (516), Hf (572), Ta (740).  Per-metal rates for these are high-variance due to sample scarcity even after with-replacement sampling.
- **Why is the 0.85 gate not always reachable?** Structural reasons documented in TODO-83 / Wf-67 P2: the closure-theorem valence-BNF predicate conservatively rejects tmQM seeds with dative-bond notation `[M+]<-X` (under-saturates the metal in RDKit valence counting).  Lifting reachability to 0.95+ requires a chemistry-aware SMILES parser or closure-theorem relaxations; both are out-of-scope for the Wf-130 P2 measurement pass.

## Honest-negative metals

The following metals achieved reachability below 0.50 — honest framing: 
- **Sc**: n=5000 unique=457 repeats=4543 reachability=0.4144
- **V**: n=5000 unique=1325 repeats=3675 reachability=0.1796
- **Zr**: n=5000 unique=2614 repeats=2386 reachability=0.4808
- **Nb**: n=5000 unique=512 repeats=4488 reachability=0.3398
- **Mo**: n=5000 unique=3794 repeats=1206 reachability=0.3402
- **Tc**: n=5000 unique=275 repeats=4725 reachability=0.4354
- **Ta**: n=5000 unique=739 repeats=4261 reachability=0.4404
- **Os**: n=5000 unique=1204 repeats=3796 reachability=0.4758
- **Ir**: n=5000 unique=4551 repeats=449 reachability=0.3590
- **Au**: n=5000 unique=3925 repeats=1075 reachability=0.4836

Reasons vary by metal: Tc and La have < 300 unique SMILES (sample scarcity); V/Cr/Mn can be valence-BNF rejected on early-row 3d samples (under-saturated); Mo/Tc have high oxidation-state variance in tmQM.
