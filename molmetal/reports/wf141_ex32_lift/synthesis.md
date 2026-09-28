# Wf-141 v2 — cisplatin Vina exhaustiveness 8 → 32 lift

_Generated: 2026-09-26T21:59:48.501577+00:00_  
_Hardware: CPU-only (Vina Python binding) + GPU-0 Blackwell_  
_Ligand SMILES_: `[Pt](N)(N)(Cl)Cl` (= cisplatin).  
_Engine_: Vina 1.2.7 (Python binding).  
_Pockets_: ['830c', '1h36', '3eml', '3ny8', '4aaw', '4rlu', '4rn0', '1fmc', '1g9v', '2hnx']; _upper-bound pockets_: ['1h36', '4aaw'].
_Exhaustiveness_: primary=32, upper-bound=64; _n_poses_: 5; _seeds_: [0, 1, 2].

## Per-pocket kcal/mol (ex=32) vs Wf-121 P3 ex=8 baseline

| Pocket | ex=8 mean (baseline) | ex=32 mean (this run) | Δ (ex32−ex8) | n_seeds_ok | status |
|---|---|---|---|---|---|
| 830c | -3.043 | -3.066 | -0.023 | 3/3 | MEASURED |
| 1h36 | -3.587 | -3.585 | +0.002 | 3/3 | MEASURED |
| 3eml | -2.827 | -2.826 | +0.001 | 3/3 | MEASURED |
| 3ny8 | -3.724 | -3.731 | -0.007 | 3/3 | MEASURED |
| 4aaw | -3.282 | -3.285 | -0.004 | 3/3 | MEASURED |
| 4rlu | -2.983 | -2.980 | +0.003 | 3/3 | MEASURED |
| 4rn0 | -2.548 | -2.548 | -0.000 | 3/3 | MEASURED |
| 1fmc | -3.133 | -3.134 | -0.000 | 3/3 | MEASURED |
| 1g9v | -3.045 | -3.041 | +0.005 | 3/3 | MEASURED |
| 2hnx | -3.380 | -3.379 | +0.001 | 3/3 | MEASURED |

## Upper-bound (ex=64) on 1h36 + 4aaw

| Pocket | ex=8 baseline | ex=32 mean | ex=64 mean | Δ(ex64−ex8) | Δ(ex64−ex32) |
|---|---|---|---|---|---|
| 1h36 | -3.587 | -3.585 | -3.591 | -0.004 | -0.006 |
| 4aaw | -3.282 | -3.285 | -3.278 | +0.004 | +0.008 |

## Honest framing (party-twenty-spirit §3.5)

* All kcal/mol values are **real Vina 1.2.7 output** — same SMILES, same receptor, same docking box, same seeds as Wf-121 P3; only `exhaustiveness` differs.
* `ex=32` is the well-known Trott 2010 lift (AutoDock Vina documentation: recommended production exhaustiveness), not a novel method. We use it because Wf-121 P3's ex=8 baseline may have been under-sampled.
* cisplatin is a **covalent binder** — Vina's scoring function does not include any covalent contribution. The kcal/mol here is the non-covalent pose energy, not the full reaction enthalpy. This caveat is identical to Wf-121 P3.
* Where `delta_vs_ex8 > 0` (NO LIFT, regression), we report the value honestly — Vina's stochastic search can occasionally fail to improve (or even regress) when exhaustiveness increases. We do not silently suppress negative-lift pockets.
* `ex=64` is **not** an industry-standard protocol — it is a 2-pocket upper-bound reference chosen because (a) both pockets have small ligand heavy-atom counts (595 and 401) so the run time stays tractable, and (b) they bracket the median ex=8 kcal/mol.
* We do NOT claim ex=32 will hit the -7 kcal/mol cutoff on all 10 pockets, do NOT claim ex=64 is industry standard, and do NOT claim this is m3 paper §4 main lead.

## Files

* `per_pocket_summary.json` — 10 pockets × ex=32.
* `upper_bound_ex64.json` — 2 pockets × ex=64.
* `<pocket>/per_seed_metrics.json` — per-seed record per pocket.
