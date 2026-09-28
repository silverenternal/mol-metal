# Wf-143 Phase 4 — Chemotype VDW cisplatin × 10 pockets × 3 seeds

_Generated: 2026-09-26T22:16:45.413830+00:00_  
_Hardware: CPU (vina Python binding 1.2.7)_  
_Ligand SMILES_: `[Pt](N)(N)(Cl)Cl` (= cisplatin).  
_Engine_: Vina 1.2.7 (Python binding).  
_Exhaustiveness_: 8; _n_poses_: 5; _seeds_: [0, 1, 2].

## Headline numbers

* 10-pocket mean (chemotype VDW) = -3.155 kcal/mol
* Wf-121 P3 baseline mean (Mg fallback) = -3.155 kcal/mol
* Δ vs baseline = 0.000 kcal/mol (negative = lift)
* Pockets passing -5.0: 0 / 10
* Pockets passing -6.0: 0 / 10
* Pockets passing -7.0: 0 / 10

## Per-pocket comparison

| Pocket | ex8_kcal_per_seed | chemotype_kcal_per_seed | ex8_mean | chemotype_mean | Δ vs ex8 (Mg) | chemotype accepted |
|---|---|---|---|---|---|---|
| 1fmc | -3.131, -3.131, -3.138 | -3.131, -3.131, -3.138 | -3.133 | -3.133 | +0.000 | False |
| 1g9v | -3.057, -3.039, -3.040 | -3.057, -3.039, -3.040 | -3.045 | -3.045 | +0.000 | False |
| 1h36 | -3.597, -3.575, -3.589 | -3.597, -3.575, -3.589 | -3.587 | -3.587 | +0.000 | False |
| 2hnx | -3.387, -3.376, -3.376 | -3.387, -3.376, -3.376 | -3.380 | -3.380 | +0.000 | False |
| 3eml | -2.826, -2.827, -2.827 | -2.826, -2.827, -2.827 | -2.827 | -2.827 | +0.000 | False |
| 3ny8 | -3.721, -3.716, -3.734 | -3.721, -3.716, -3.734 | -3.724 | -3.724 | +0.000 | False |
| 4aaw | -3.287, -3.275, -3.283 | -3.287, -3.275, -3.283 | -3.282 | -3.282 | +0.000 | False |
| 4rlu | -2.995, -2.981, -2.974 | -2.995, -2.981, -2.974 | -2.983 | -2.983 | +0.000 | False |
| 4rn0 | -2.548, -2.547, -2.549 | -2.548, -2.547, -2.549 | -2.548 | -2.548 | +0.000 | False |
| 830c | -3.033, -3.067, -3.030 | -3.033, -3.067, -3.030 | -3.043 | -3.043 | +0.000 | False |

## Honest framing (party-twenty-spirit §3.5)

* The Wf-121 P3 baseline records the **generic VDW** atype fallback (Pt → Mg) for cisplatin. This run invokes the chemotype-tuned atype path (Pt); Vina 1.2.7 may either accept ``Pt`` directly or fall back to ``Mg`` — the per-pocket `chemotype_atype_accepted` column records which path ran.
* The expected lift per TODO-128 §4 is +0.5 to +1.5 kcal/mol, but the actual lift may be near zero if Vina rejects the chemotype atype and falls back to Mg (in which case the run re-creates the Wf-121 P3 baseline).
* Cisplatin is a covalent binder — Vina does not model the Pt–DNA / Pt–protein covalent bond regardless of VDW tuning. The kcal/mol reported here is the **non-covalent contribution only**; absolute values stay well above the covalent-binder affinity floor (~-7 to -9 kcal/mol).
* Per-pocket lift varies with pharmacophore; we do not expect uniform improvement across all 10 pockets.
