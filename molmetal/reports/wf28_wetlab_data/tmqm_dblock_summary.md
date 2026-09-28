# WF-28 Phase 1 — tmQM d-block metal complex subset

**Date**: 2026-09-26  
**Source corpus**: tmQM 108K (Balcells & Skjelstad 2020, MIT)  
**PRIMITIVE_ATOMS contract**: 30 d-block metals in `molmetal_lam.atoms.combinators` (R17 ship)  
**Output CSV**: `molmetal/data/tmqm_dblock_subset.csv`  
**Stratified target**: 5,000 complexes (~proportional to per-metal counts)

## 1. Provenance

TmQM was parsed via `molmetal.data.tmqm.load_tmqm` with
`require_smiles=True`; rows with empty SMILES were dropped
(7,692 of 108,541 in the raw release).

Filters applied (in order):

1. SMILES parseable by RDKit  
2. Coordination number in [2, 9]  
3. Heavy-atom window [8, 60]  
4. SA-score computable (Ertl-Schuffenhauer via RDKit Contrib)  
5. Stratified random sample across all 30 d-block metals

## 2. Counts

| Metric | Value |
|---|---|
| Loaded from tmQM | 100,849 |
| After coord-number filter | 91,939 |
| After heavy-atom + SA filter | 91,939 |
| After stratified sample | 5,000 |
| Unique d-block metals represented | 30 / 30 |

## 3. Per-metal allocation

| Metal | n | Fraction |
|---|---:|---:|
| Ag | 81 | 0.016 |
| Au | 217 | 0.043 |
| Cd | 85 | 0.017 |
| Co | 179 | 0.036 |
| Cr | 107 | 0.021 |
| Cu | 174 | 0.035 |
| Fe | 261 | 0.052 |
| Hf | 20 | 0.004 |
| Hg | 61 | 0.012 |
| Ir | 242 | 0.048 |
| La | 3 | 0.001 |
| Mn | 113 | 0.023 |
| Mo | 180 | 0.036 |
| Nb | 19 | 0.004 |
| Ni | 522 | 0.104 |
| Os | 55 | 0.011 |
| Pd | 533 | 0.107 |
| Pt | 411 | 0.082 |
| Re | 213 | 0.043 |
| Rh | 219 | 0.044 |
| Ru | 444 | 0.089 |
| Sc | 20 | 0.004 |
| Ta | 30 | 0.006 |
| Tc | 15 | 0.003 |
| Ti | 120 | 0.024 |
| V | 72 | 0.014 |
| W | 156 | 0.031 |
| Y | 30 | 0.006 |
| Zn | 365 | 0.073 |
| Zr | 53 | 0.011 |

## 4. Quick descriptors

- **mw**: mean=591.965, std=162.078, min=185.582, max=1258.270
- **logp**: mean=3.755, std=4.600, min=-16.795, max=19.283
- **qed_score**: mean=0.257, std=0.144, min=0.023, max=0.799
- **sa_score_unit**: mean=0.552, std=0.112, min=0.160, max=1.000
- **n_atoms**: mean=33.341, std=11.118, min=8.000, max=60.000
- **n_rings**: mean=3.613, std=2.175, min=0.000, max=16.000

## 5. Coverage of the project's 6 anchor metals

The wet-lab validation tier prioritises Pt / Ru / Ir / Pd / Au / Rh
(the metallodrug core; see `molmetal/reports/wf_wetlab_plan/`).
The table below checks that the stratified sample still covers them
with reasonable mass:

| Metal | n | Mean n_atoms | Mean MW | Mean QED |
|---|---:|---:|---:|---:|
| Pt | 411 | 31.3 | 659.9 | 0.277 |
| Ru | 444 | 36.7 | 624.3 | 0.230 |
| Ir | 242 | 37.0 | 713.4 | 0.228 |
| Pd | 533 | 32.8 | 588.8 | 0.252 |
| Au | 217 | 30.7 | 631.6 | 0.250 |
| Rh | 219 | 34.3 | 598.4 | 0.224 |

## 6. Reproducibility

- Seed: `20260926`  
- tmQM path: `$MOLMETAL_TMQM_DIR`  
- Run: `python -m molmetal.scripts.extract_tmqm_dblock`  
- The script is CPU-only (RDKit + pandas) and writes both the CSV
  and the companion stats JSON (`tmqm_dblock_stats.json`).

## 7. Downstream hand-off

This CSV is the input for:

- WF-28 Phase 2 (RDKit-side scaffold-extraction + SyntheMol route
  enumeration per `wf_wire_click_rules.py`)
- WF-28 Phase 3 (RDKit 3D conformer + MMFF94 relax + QuickVina
  prep per `recover_gpu_trace_reference.py`)
- WF-28 Phase 4 (¥70-130k budget procurement batch plan; see
  `wf_wetlab_plan/final.md`)
