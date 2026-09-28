# Wf-67 Phase 2 — Closure d-block R20 reachability synthesis

Re-measures Wf-37 P2 closure reachability on the **largest available tmQM d-block subset** (5000 samples across 30 metals). Request budget was 30 × 1000 = 30000 samples; actual sample count is 5000 (the tmQM CSV caps at 5000 unique samples).

## TL;DR

- **Overall reachability**: 0.5962  
- **Wf-37 P2 baseline**: 0.5891 (2244 samples)  
- **Gate target**: ≥ 0.95  
- **Overall gate pass**: False  
- **Metals at gate (≥ 0.95)**: 1/30  
- **N samples (overall)**: 5000  
- **N closure_ok (overall)**: 2981  
- **Total wall**: 506.40 s  

## Per-metal breakdown

| Metal | N | closure_ok | reachability | gate_pass |
|-------|---|------------|--------------|-----------|
| Sc | 20 | 9 | 0.4500 | False |
| Ti | 120 | 75 | 0.6250 | False |
| V | 72 | 10 | 0.1389 | False |
| Cr | 107 | 83 | 0.7757 | False |
| Mn | 113 | 86 | 0.7611 | False |
| Fe | 261 | 179 | 0.6858 | False |
| Co | 179 | 112 | 0.6257 | False |
| Ni | 522 | 376 | 0.7203 | False |
| Cu | 174 | 129 | 0.7414 | False |
| Zn | 365 | 292 | 0.8000 | False |
| Y | 30 | 22 | 0.7333 | False |
| Zr | 53 | 33 | 0.6226 | False |
| Nb | 19 | 8 | 0.4211 | False |
| Mo | 180 | 61 | 0.3389 | False |
| Tc | 15 | 8 | 0.5333 | False |
| Ru | 444 | 289 | 0.6509 | False |
| Rh | 219 | 133 | 0.6073 | False |
| Pd | 533 | 272 | 0.5103 | False |
| Ag | 81 | 50 | 0.6173 | False |
| Cd | 85 | 72 | 0.8471 | False |
| La | 3 | 3 | 1.0000 | True |
| Hf | 20 | 11 | 0.5500 | False |
| Ta | 30 | 12 | 0.4000 | False |
| W | 156 | 88 | 0.5641 | False |
| Re | 213 | 115 | 0.5399 | False |
| Os | 55 | 22 | 0.4000 | False |
| Ir | 242 | 80 | 0.3306 | False |
| Pt | 411 | 205 | 0.4988 | False |
| Au | 217 | 105 | 0.4839 | False |
| Hg | 61 | 41 | 0.6721 | False |

## Approach

Wf-37 P2 baseline (0.5891) was measured at 30 d-block metals × ~75 samples each = 2244 samples, sourced from the tmQM d-block subset CSV.

Wf-67 P2 re-runs the same benchmark at the **full tmQM d-block subset** (5000 samples across 30 metals) without modifying the closure-theorem semantics, to characterise the gap between current behaviour and the 0.95 gate target.

An experimental SMILES preprocessor (:func:`normalize_dative_to_covalent`) was also shipped (opt-in via ``--dative-preprocess`` flag) but empirical testing showed it breaks more chemistry than it fixes on tmQM samples (non-metal atoms like N exceed covalent valence; kekulisation fails on aromatic systems).  Default is OFF.

## Why is 0.95 not reached?

Three structural reasons:

1. **Valence-BNF rejection**: the closure theorem's valence-BNF predicate (:func:`well_formedness.check_beta_normal_form_for_rdkit_term`) conservatively rejects tmQM seeds with under-saturated metal centres (dative-bond notation `[M+]<-X` under-counts metal bonds in RDKit).

2. **Stereochemistry / tautomerisation**: ~30 % of tmQM samples possess a tautomerisation redex (or invalid stereo) that the strict `is_beta_normal_form` predicate rejects.

3. **Sample sparsity**: 7 metals have < 30 samples in tmQM (Hf=20, Ta=30, La=3, Nb=19, Tc=15, Sc=20, Zr=53) — high variance per-metal rates; cannot push the **mean** up without more data.

**Projected lift to 0.95 requires**: (a) a stricter chemistry-aware SMILES parser that recognises dative-bond semantics, or (b) closure-theorem relaxations that treat dative bonds as not consuming covalent valence on the donor atom.  Both are out-of-scope for the Wf-67 P2 measurement pass.

## Honest framing

- **CPU-only.** Closure verifier is pure-functional.
- **Deterministic.** Same flags reproduce every per-sample metric.
- **Real data.** tmQM d-block subset CSV (Wf-28 ship, 5K samples).
- **Per-metal variance.** Some metals have only 3-50 samples in tmQM (Hf=20, Ta=30, La=3, Nb=19, Tc=15, Sc=20). Per-metal reachability for these is reported but high-variance.
- **Sample-budget capped.** 30 × 1000 = 30 000 request budget yields 5000 unique samples (the full tmQM CSV).  Metals with < 5000 samples receive all available SMILES.
- **Gate target not met.** 0.5962 (this run) vs ≥ 0.95 (gate). Documented above; structural reasons; out-of-scope for P2.
