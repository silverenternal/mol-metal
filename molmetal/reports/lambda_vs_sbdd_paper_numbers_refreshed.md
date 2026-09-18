# Refreshed Lambda-vs-SBDD Paper Numbers (round 9)

**Date:** 2026-09-13  
**Scope:** cite-only SOTA rows versus measured Lambda pilot values.  
**Provenance:** `lambda_vs_sbdd_paper_numbers.md`, `lambda_vs_sbdd_protocol_aligned.md`, `round9_r4c_pilot_results.md`, `r4_c_full_sweep_real.md`, `sota_papers_2025_2026.md`.

## §1 Pocket-mean Vina comparison

All SOTA values are cited paper means; Lambda values are measured on the stated pocket. The 1433B value is a proxy placeholder and must not be interpreted as AutoDock Vina.

| STATUS | Method (arXiv) | Pocket / protocol | Vina mean (kcal/mol) | n |
|---|---|---|---:|---:|
| CITED | Pocket2Mol (arXiv:2205.07249) | CrossDocked2020, 100 pockets | `-7.07` | 100 |
| CITED | TargetDiff (arXiv:2303.03543) | CrossDocked2020, 100 pockets | `-8.45` | 100 |
| CITED | DiffSBDD (arXiv:2210.13695) | CrossDocked2020, 100 pockets | `-7.62` | 100 |
| CITED | DecompDiff (arXiv:2303.10120) | CrossDocked2020, 100 pockets | `-8.39` | 100 |
| CITED | FLOWR (arXiv:2504.10564) | SPINDR/PLINDER curated set | `-6.93` | 100 |
| CITED | TransDiffSBDD (arXiv:2025 preprint) | CrossDocked2020 | `-9.37` | 100 |
| CITED | MolCRAFT (arXiv:2024 preprint) | CrossDocked2020 | `-9.25` | 100 |
| CITED | AlphaDrug (arXiv:2024 preprint) | published benchmark | `-9.77` | published |
| CITED | MolChord (arXiv:2025 preprint) | CrossDocked2020 | `-7.62` | published |
| MEASURED | Lambda (round-9, 1h36) | real AutoDock Vina 1.2.7, exh=16 | `-5.923` | 1 pocket |
| MEASURED | Lambda (round-9, 1433B) | Vina-proxy placeholder | `-20.061` | 20 candidates |

**Sources:** protocol-aligned §1/§2; round9 pilot results; paper-numbers report.

## §2 SA / QED / Lipinski

| STATUS | Method (arXiv) | SA mean | QED mean | Lipinski pass |
|---|---|---:|---:|---:|
| CITED | Pocket2Mol (arXiv:2205.07249) | `2.51` | `0.55` | — |
| CITED | TargetDiff (arXiv:2303.03543) | `2.65` | `0.48` | — |
| CITED | DiffSBDD (arXiv:2210.13695) | `2.81` | — | — |
| CITED | DecompDiff (arXiv:2303.10120) | `2.71` | — | — |
| CITED | FLOWR (arXiv:2504.10564) | `2.86` | — | — |
| MEASURED | Lambda 1h36 | `1.870` | `0.548` | `1.000` |
| MEASURED | Lambda 1433B pilot | `9.876` | `0.304` | `1.000` |
| MEASURED | Lambda R4-C sweep (1h36 + 830c) | `7.854` | `0.857` | `1.000` |

**Sources:** round9_r4c_pilot_results.md; r4_c_full_sweep_real.md; protocol-aligned §3.2.

## §3 PoseBusters pass rate

| STATUS | Method (arXiv) | PoseBusters pass rate | Scope |
|---|---|---:|---|
| CITED | Pocket2Mol (arXiv:2205.07249) | not reported in cited table | paper metric unavailable |
| CITED | TargetDiff (arXiv:2303.03543) | not reported | paper metric unavailable |
| CITED | DiffSBDD (arXiv:2210.13695) | not reported | paper metric unavailable |
| CITED | DecompDiff (arXiv:2303.10120) | not reported | paper metric unavailable |
| CITED | FLOWR (arXiv:2504.10564) | `94%` PB-valid | PB-valid, not docking success |
| MEASURED | Lambda (prior click-tile report) | `12/12 = 100%` | pre-pilot 12 click tiles |
| MEASURED | Lambda round-9 pilot | n/a | PoseBusters not wired in pilot |

**Source:** lambda_vs_sbdd_paper_numbers.md §3; protocol-aligned §4.1.

## §4 Triple-threshold success rate

Definition: **Vina < co-crystal ∧ SA < 4 ∧ QED > 0.5**.

| STATUS | Method (arXiv) | Published / measured success |
|---|---|---:|
| CITED | Pocket2Mol (arXiv:2205.07249) | `49.8%` |
| CITED | TargetDiff (arXiv:2303.03543) | `35.1%` relaxed / `10.5%` strict |
| CITED | DiffSBDD (arXiv:2210.13695) | `24.6%` |
| CITED | DecompDiff (arXiv:2303.10120) | `39.0%` / `24.5%` |
| CITED | FLOWR (arXiv:2504.10564) | `94%` PB-valid only |
| CITED | TransDiffSBDD (arXiv:2025 preprint) | `83.9%` |
| CITED | MolCRAFT (arXiv:2024 preprint) | `36.1%` |
| CITED | MolChord (arXiv:2025 preprint) | `33.2%` |
| MEASURED | Lambda 1433B pilot | `0.00` |

Lambda 1433B fails the conjunction because SA `9.876 ≫ 4`; its Vina value is also proxy-only.

## §5 7 protocol-mismatch flags

The following table is reproduced verbatim from `lambda_vs_sbdd_protocol_aligned.md` §2.

| # | Flag | Why it blocks direct comparison |
|---:|---|---|
| 1 | n_test = 1 vs 100 (or 363) | Every Group A and Group B paper reports a population mean over 100–363 held-out pockets. Lambda’s 1h36 run is a single-case value with no error bar. |
| 2 | Pocket corpus mismatch | Lambda’s pocket (PDB 1h36 chain A, HEM-bound) is not in the standard CrossDocked2020 100-pocket test split; Vina kcal/mol is pocket-dependent. |
| 3 | SA-score implementation likely aligned but not byte-verified | Lambda uses RDKit Contrib sascorer.py; provenance audits do not pin the exact file for every cited paper, so byte-equivalence is unverified. |
| 4 | Pocket2Mol / TargetDiff / DiffSBDD were NOT run by Lambda | Group A numbers are cited from the original papers, not reproduced checkpoints. |
| 5 | FLOWR’s “94%” is PoseBusters-valid only, not docking success | PoseBusters tests conformer geometry; it does not test whether a molecule binds the pocket better than the reference ligand. |
| 6 | NFE accounting differs | Group A papers report diffusion NFE; Lambda is MCTS and its simulation budget is not commutable with diffusion-NFE. |
| 7 | Docking engine differences | TargetDiff uses QVina, Pocket2Mol uses QuickVina2, and Lambda uses AutoDock Vina 1.2.x; scoring/search versions can drift. |

**Source:** lambda_vs_sbdd_protocol_aligned.md §2 (flags 1–7).

## §6 Per-pocket table

Delta is Lambda minus the cited Pocket2Mol Vina mean (`-7.07`) when a numeric comparison is meaningful.

| pocket_id | Lambda Vina | cite-only SOTA Vina | delta | n_molecules |
|---|---:|---:|---:|---:|
| 1h36 | `-5.923` real Vina | `-7.07` Pocket2Mol | `+1.147` | 1-pocket pool |
| 1433B | `-20.061` proxy | `-7.07` Pocket2Mol | n/a (proxy) | 20 |
| 830c | `-14.637` proxy top-1 | `-7.07` Pocket2Mol | n/a (proxy) | 5 |

**Sources:** round9_r4c_pilot_results.md; r4_c_full_sweep_real.md (830c row).

## §7 Per-seed standard deviation

| Quantity | Value | Status / provenance |
|---|---:|---|
| Round-9 Lambda seeds | `1` | measured pilot; flag as single seed |
| Within-engine per-seed Vina sigma | `unmeasured` | round9_qvina_parity.md; no published σ |
| Recommended follow-up | `N=5 seeds` | round9_qvina_parity.md recommendation |

No multi-seed error bar is reported for the pilot.

## §8 Honest caveats

- ROCm 7.2 Lambda measurements run on AMD hardware, while cited SOTA was produced in CUDA environments; hardware and software stacks differ.
- SOTA checkpoints and headline metrics are cite-only; Mol-Metal did not rerun those models.
- tmQM pretraining reaches R² = `0.986` in the tracked work, but there is no head-to-head Lambda-vs-SOTA evaluation.
- PoseBusters uses an MMFF94 geometry limitation in the current validation path; PB-valid is not docking success.
- Sample sizes are small for Lambda (one pocket, 20 pilot candidates, or five per R4-C pocket) versus published 100-pocket aggregates.
