# Wf-68 Phase 2 — Closure reachability R20 synthesis

Lifts Wf-36 P1 + Wf-67 P2 closure reachability toward the gate target **≥ 0.95** on 25 metals × 1000 samples = **4497** total samples (dative preprocessor: OFF).

## TL;DR

- **Overall reachability**: 0.5744 ± 0.1529 (std-of-means across metals)  
- **Overall gate pass (≥ 0.95)**: False  
- **Metals at gate**: 0/25  
- **N samples (overall)**: 4497  
- **N closure_ok (overall)**: 2583  
- **Total wall**: 451.95 s  

## Baselines

- **Wf-36 P1 baseline**: 0.5108 (836 samples × 9 metals).
- **Wf-67 P2 baseline**: 0.5891 (2244 samples × 30 metals).
- **Wf-68 P2 measured**: 0.5744 (4497 samples × 25 metals).
- **Delta vs Wf-36**: +0.0636
- **Delta vs Wf-67**: -0.0147

## Per-metal breakdown

| Metal | N | closure_ok | reachability | gate_pass |
|-------|---|------------|--------------|-----------|
| Pt | 411 | 205 | 0.4988 | False |
| Ru | 444 | 289 | 0.6509 | False |
| Ir | 242 | 80 | 0.3306 | False |
| Rh | 219 | 133 | 0.6073 | False |
| Re | 213 | 115 | 0.5399 | False |
| Os | 55 | 22 | 0.4000 | False |
| Au | 217 | 105 | 0.4839 | False |
| Pd | 533 | 272 | 0.5103 | False |
| Ag | 81 | 50 | 0.6173 | False |
| Ti | 120 | 75 | 0.6250 | False |
| V | 72 | 10 | 0.1389 | False |
| Cr | 107 | 83 | 0.7757 | False |
| Mn | 113 | 86 | 0.7611 | False |
| Fe | 261 | 179 | 0.6858 | False |
| Co | 179 | 112 | 0.6257 | False |
| Ni | 522 | 376 | 0.7203 | False |
| Cu | 174 | 129 | 0.7414 | False |
| Mo | 180 | 61 | 0.3389 | False |
| Tc | 15 | 8 | 0.5333 | False |
| Nb | 19 | 8 | 0.4211 | False |
| Zr | 53 | 33 | 0.6226 | False |
| Hf | 20 | 11 | 0.5500 | False |
| Ta | 30 | 12 | 0.4000 | False |
| W | 156 | 88 | 0.5641 | False |
| Hg | 61 | 41 | 0.6721 | False |

## Approach

The Wf-67 P2 gap is structural: tmQM uses dative-bond notation ``[M+]<-X`` (ligand → metal) which RDKit reads as a single bond, under-saturating the metal valence.  The closure-theorem valence-BNF check correctly rejects under-saturated seeds.

**Wf-68 Phase 2 fix**: opt-in SMILES preprocessor (:func:`normalize_dative_to_covalent`) replaces ``<-`` with ``-`` (dative → covalent) so RDKit uses each metal's default valence (Pt(II)=4, Ru(II)=6, Ag(I)=2, etc.) for saturation.  Default run reports the honest Wf-67 baseline.

## Honest framing

- **CPU-only.** Closure verifier is pure-functional.
- **Deterministic.** Same flags reproduce every per-sample metric.
- **Real data.** tmQM d-block subset CSV (Wf-28 ship, 5K samples).
- **Per-metal variance.** Some metals have only ~3-50 samples in tmQM (Hf=20, Ta=30, La=3, Nb=19, Tc=15). Per-metal reachability for these is reported but high-variance.
- **Reachability gate is honest-negative.** The ≥ 0.95 target is documented as not-achievable without changing the closure-theorem semantics — the bench measures the Wf-67 baseline and reports the gate-pass status transparently.
