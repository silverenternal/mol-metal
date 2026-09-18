# R4-C full sweep — Lambda vs cited SOTA (1/1 pockets)

- **n_pockets_total:** 1
- **n_pockets_ok:** 0
- **n_pockets_fail:** 1
- **n_candidates_total:** None
- **mean candidates/pocket:** 0.00
- **lipinski pass rate:** 0.000
- **SA mean (1-10, lower=easier):** nan
- **QED mean (0-1, higher=better):** nan
- **Vina-proxy top-1 mean:** nan
- **wall seconds/pocket:** 0.00

## Method

- MCTSProofSearch with n_simulations=1, max_depth=3
- L-3 204-tile library (FRAGMENT_LIBRARY_200_TILES)
- LIPINSKI predicate
- Vina proxy placeholder (L-1 oracle pending; see TODO/pending/decisions.md D4)

## Lambda vs **cited** SOTA — same protocol (CrossDocked100, n=100)

| method | year | Vina (kcal/mol) | success_rate | status | source |
|---|---:|---:|---:|---|---|
| **Lambda (R4-C, measured)** | 2026 | nan (proxy) | 0.000 | **MEASURED** (0/1 pockets) | this sweep |
| Pocket2Mol | 2022 | -7.07 | 0.244 | **CITED** | Peng 2022 (CrossDocked100, n=100) |
| TargetDiff | 2023 | -8.45 | 0.105 | **CITED** | Guan 2023 (CrossDocked100, n=100) |
| DiffSBDD | 2024 | -7.62 | 0.246 | **CITED** | Qin 2024 (CrossDocked100, n=100) |
| DecompDiff | 2024 | -8.39 | 0.245 | **CITED** | DecompDiff paper (CrossDocked100) |
| FLOWR | 2024 | -6.93 | — | **CITED** | FLOWR paper (CrossDocked100) |
| MolCRAFT | 2024 | -9.25 | 0.361 | **CITED** | MolCRAFT paper (CrossDocked100, n=100) |
| AlphaDrug | 2025 | -9.77 | — | **CITED** | AlphaDrug paper (CrossDocked100) |
| TransDiffSBDD | 2025 | -9.37 | 0.839 | **CITED** | TransDiffSBDD paper (CrossDocked100) |
| MolChord | 2025 | -7.62 | 0.332 | **CITED** | MolChord paper (CrossDocked100) |

## Honest framing
- **Lambda row is MEASURED by us**, but uses a Vina *proxy* placeholder (TODO/pending/decisions.md D4). Replace with real Vina once L-1 oracle is live.
- **SOTA rows are CITED** from each paper; not re-run by us. Strict head-to-head is gated on DiffSBDD ckpt access + torch_geometric ROCm 7.2 wheels (TODO/pending/risks.md R1).
- **Vina 1.2.7 vs published-protocol QVina mismatch** inflates Lambda numbers relative to SOTA baselines — see TODO/pending/decisions.md D7.
- Sample sizes: Lambda = 1 (this sweep); SOTA = 100 per paper (CrossDocked100 standard).

## Top-1 SMILES per pocket (sample)

| pocket_id | status | top1_smiles | top1_sa | top1_qed | lipinski | vina_proxy |
|---|---|---|---|---|---|---|
| 1433B_HUMAN_1_240_pep_0 | no_candidates | `` | — | — | — | — |

## Provenance

- Lambda side: `molmetal/scripts/lambda_100pocket_sweep.py` (L-3 204-tile, LIPINSKI)
- Vina proxy: TODO/pending/decisions.md D4 (gated on L-1 oracle)
- SOTA citations: `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` + `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` + `molmetal/reports/h4_paper_grade_comparison.md`
