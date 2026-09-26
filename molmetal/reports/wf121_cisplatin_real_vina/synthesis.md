# Wf-121 Phase 3 — Real Vina dock on cisplatin × 10 pockets

_Generated: 2026-09-26T19:21:01.409939+00:00_  
_Hardware: CPU (Vina Python binding) + GPU-0 Blackwell_  
_Ligand SMILES_: `[Pt](N)(N)(Cl)Cl` (= cisplatin).  
_Engine_: Vina 1.2.7 (Python binding preferred, CLI fallback).  
_Exhaustiveness_: 8; _n_poses_: 5; _seeds_: [0, 1, 2].

## Per-pocket real kcal/mol (best pose)

| Pocket | kcal/mol min | median | mean | stddev | n_seeds | status | boltz_affinity pIC50 |
|---|---|---|---|---|---|---|---|
| 830c | -3.067 | -3.033 | -3.043 | 0.017 | 3 | MEASURED | -0.132 |
| 1h36 | -3.597 | -3.589 | -3.587 | 0.009 | 3 | MEASURED | -0.132 |
| 3eml | -2.827 | -2.827 | -2.827 | 0.000 | 3 | MEASURED | -0.132 |
| 3ny8 | -3.734 | -3.721 | -3.724 | 0.008 | 3 | MEASURED | -0.132 |
| 4aaw | -3.287 | -3.283 | -3.282 | 0.005 | 3 | MEASURED | -0.132 |
| 4rlu | -2.995 | -2.981 | -2.983 | 0.009 | 3 | MEASURED | -0.132 |
| 4rn0 | -2.549 | -2.548 | -2.548 | 0.001 | 3 | MEASURED | -0.132 |
| 1fmc | -3.138 | -3.131 | -3.133 | 0.003 | 3 | MEASURED | -0.132 |
| 1g9v | -3.057 | -3.040 | -3.045 | 0.008 | 3 | MEASURED | -0.132 |
| 2hnx | -3.387 | -3.376 | -3.380 | 0.005 | 3 | MEASURED | -0.132 |

## Honest framing (party-twenty-spirit §3.5)

* All kcal/mol values are **real Vina 1.2.7 output** — not the Wf-30 hash-derived stub floor (-9.6574) and not a placeholder.
* The Pt atom is typed with the generic AutoDock VDW atype `M` (Wf-121 P1 SMARTS rules). The kcal/mol is therefore a **real** Vina measurement with a documented generic-atom caveat — disposition `MEASURED_GENERIC_ATOM` (i.e. *not* mock, *not* full-fidelity chemotype).
* The boltz_affinity column is the **D-MPNN pIC50 fallback** (Wf-120 R20 ship, MetalCytoToxDB-trained Attentive D-MPNN). It is included as parallel evidence — Vina and boltz_affinity are different scoring functions with different units (kcal/mol vs. pIC50), so they are not directly comparable.
* Each (pocket, seed) record lives in `molmetal/reports/wf121_cisplatin_real_vina/<pocket>/per_seed_metrics.json` with kcal/mol + RMSD-lb/ub + Vina pose count + ligand atom count.

## Files

* `sweep_summary.json` — machine-readable per-pocket aggregate.
* `<pocket>/per_seed_metrics.json` — per-seed raw Vina record.
* `<pocket>/cisplatin.pdbqt` — meeko-emitted cisplatin PDBQT (generic `M` atype for Pt).
