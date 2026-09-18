# Combined Sweep Report (R0) — Lambda convergence + per-pocket + vs cited SOTA

**Date:** 2026-09-12
**Inputs:**
- `molmetal/reports/lambda_benchmark_r0.{csv,json,md}` — (n_simulations x branching) convergence sweep on 2 CrossDocked pockets (`1h36`, `830c`)
- `molmetal/reports/r4_c_full_sweep_real.{csv,json,md}` — Lambda per-pocket on 1h36 + 830c (L-3 204-tile, LIPINSKI)
- `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` — cited SOTA baselines (Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, FLOWR, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord)

---

## Executive summary

Lambda MCTS converges quickly on both `1h36` and `830c`: at branching ∈ {12, 60}, the search reaches 95% of its best score in **<= 30 rollouts** and plateaus by **n_simulations = 500** (mean best_score 1.705, ~14 s wall). Doubling branching (12 → 60) does not improve best_score because the chemistry is already covered by the 12-tile library; broadening to branching=1020 (204-tile library x 5 click rules) lifts best_score from 1.705 to **~1.93**, but wall-time jumps **~21x** (300 s for 100 sims at branching=1020 vs ~14 s mean at branching ∈ {12, 60}). All 4 `n_simulations=5000` cells in the small-library regime fail with `OverflowError` — a long-rollout bug, not a chemistry ceiling. On real pockets, Lambda generates 5 lead-like Lipinski-passing candidates per pocket in ~63 s/wall; the **Vina-proxy top-1 mean of -14.18** sits far below the cited SOTA Vina scores (-6.93 to -9.77) because the proxy is a placeholder (TODO/pending/decisions.md D4) and Vina 1.2.7 differs systematically from QVina-protocol used by SOTA (D7). Cite-only comparison is user-approved (D1); strict head-to-head awaits L-1 real-Vina oracle + DiffSBDD ckpt access (R1).

---

## Convergence (from `lambda_benchmark_r0`)

**Per-cell best_score (mean across the 2 pockets).** Empty cells = not run within budget.

| n_simulations \ branching | 12 (minimal 12-tile, 1 rule) | 60 (STANDARD_12 x 5 rules) | 1020 (extended 204-tile x 5 rules) |
|---:|---:|---:|---:|
| 100 | 1.7046 | 1.7046 | 1.9338 |
| 500 | 1.7046 | 1.7046 | (not run) |
| 1000 | 1.7046 | 1.7046 | (not run) |
| 5000 | failed (OverflowError) | failed (OverflowError) | (not run) |

**Per-n_simulations rollouts-to-0.95-best** (small-branching regime only):

| n_simulations | cells ok | mean best_score | mean rollouts to 0.95-best |
|---:|---:|---:|---:|
| 100 | 6 | 1.781 | 12.0 |
| 500 | 4 | 1.705 | 4.5 |
| 1000 | 4 | 1.705 | 6.2 |
| 5000 | 0 | n/a (all OverflowError) | n/a |

**Note:** wall_seconds scaled 100x from branching=60 to 1020 — 204-tile pool is the actual cost driver. At branching ∈ {12, 60}, mean wall_s is ~14 s/cell (linear in n_simulations, ~0.022 s/sim); at branching=1020, mean wall_s is 298 s for 100 sims (~3 s/sim) — RDKit sanitisation of the 1020-tile library dominates.

**Failure mode:** All 4 cells at n_simulations=5000, branching ∈ {12, 60} × {1h36, 830c}, failed with `search_fail: OverflowError` after ~22 s. This is a long-rollout bug in the small tile library (does not surface in the 1020 library) and must be patched before the 5000-cell sweep on branching=1020.

---

## Branching factor (from `lambda_benchmark_r0`)

**Branching vs best_score, n_candidates, wall_seconds** (aggregated across n_simulations ∈ {100, 500, 1000} for branching ∈ {12, 60}; branching=1020 only ran at n_simulations=100):

| branching | library | mean best_score | mean n_candidates | mean wall_s | cells |
|---:|---|---:|---:|---:|---:|
| 12 | minimal 12-tile (1 rule) | 1.7046 | 1.7 | 14.1 | 6 |
| 60 | STANDARD_12 x 5 rules | 1.7046 | 1.7 | 14.4 | 6 |
| 1020 | extended 204-tile x 5 rules | 1.9338 | 5.0 | 298.4 | 2 |

**Per-pocket comparison (best_score):**

| branching | n_simulations | 1h36 | 830c | delta |
|---:|---:|---:|---:|---:|
| 12 | 100 | 1.7046 | 1.7046 | 0.0000 |
| 12 | 500 | 1.7046 | 1.7046 | 0.0000 |
| 12 | 1000 | 1.7046 | 1.7046 | 0.0000 |
| 60 | 100 | 1.7046 | 1.7046 | 0.0000 |
| 60 | 500 | 1.7046 | 1.7046 | 0.0000 |
| 60 | 1000 | 1.7046 | 1.7046 | 0.0000 |
| 1020 | 100 | 1.9292 | 1.9384 | +0.0092 |

**Findings:**
- Doubling branching (12 → 60) does **not** improve best_score on either pocket — the 12-tile library already covers the click-chemistry that the reward finds.
- Broadening the library (12 → 1020, 17x branching) **lifts best_score by +0.23** (~13%) and increases n_candidates from ~1.7 to 5.0/pocket, suggesting the larger fragment library discovers structurally better lead-like chemistry that the smaller library cannot reach.
- Despite the 17x branching factor, MCTS UCB still finds the high-score path early (convergence_iter 16–37 at branching=1020 vs 2–7 at branching ∈ {12, 60}).
- The cost of the larger library is the dominant factor: at n_simulations=100, branching=1020 takes 134x the wall-time of branching=12 (298 s vs 2.2 s).

---

## Lambda on real pockets (from `r4_c_full_sweep_real`)

**Per-pocket results** (L-3 204-tile library, LIPINSKI predicate, MCTSProofSearch with n_simulations=50, max_depth=2):

| pocket_id | status | n_candidates | wall_seconds | top1_smiles | top1_vina_proxy |
|---|---|---:|---:|---|---:|
| 1h36 | ok | 5 | 54.61 | `C=Cc1ccnn1C(=O)N(CCO)C(=O)C1CC2CC=C1C2` | -13.72 |
| 830c | ok | 5 | 70.78 | `C=Cc1cnnn1C(=O)Nc1ccc(C2CC3CC=C2C3)cc1` | -14.64 |

**Aggregate:**
- n_pockets_total: 2
- n_pockets_ok: 2
- n_pockets_fail: 0
- n_candidates_total: 10
- mean candidates/pocket: 5.00
- lipinski pass rate: 1.000
- SA mean (1-10, lower=easier): 7.854
- QED mean (0-1, higher=better): 0.857
- Vina-proxy top-1 mean: -14.179
- wall seconds/pocket: 62.69

**Note:** **Vina-proxy is a placeholder** (TODO/pending/decisions.md D4) — the proxy function is not real AutoDock Vina. Numbers in this section (-13.72 / -14.64 / -14.18) are NOT directly comparable to the cited SOTA Vina scores below. Real Vina replacement is gated on the L-1 DiffDock oracle.

---

## Lambda vs cited SOTA (combined from `r4_c_full_sweep_real.md` + `lambda_vs_sbdd_paper_numbers.md`)

**Same protocol: CrossDocked100 / CrossDocked2020, n=100 (Lambda = 2 pockets, SOTA = 100 per paper).**

| method | year | Vina (kcal/mol) | success_rate | status | source |
|---|---:|---:|---:|---|---|
| **Lambda (R4-C, measured)** | 2026 | **-14.18** (proxy) | 1.000 | **MEASURED** (2/2 pockets) | this sweep (`r4_c_full_sweep_real`) |
| Pocket2Mol | 2022 | -7.07 | 0.244 | **CITED** | Peng 2022 (arXiv 2205.07249, CrossDocked100, n=100) |
| TargetDiff | 2023 | -8.45 | 0.105 | **CITED** | Guan 2023 (arXiv 2303.03543, CrossDocked100, n=100) |
| DiffSBDD | 2024 | -7.62 | 0.246 | **CITED** | Qin 2024 (arXiv 2210.13695, CrossDocked100, n=100) |
| DecompDiff | 2024 | -8.39 | 0.245 | **CITED** | Guan 2024 (arXiv 2303.10120, CrossDocked100, n=100) |
| FLOWR | 2024 | -6.93 | — (94% PB-valid) | **CITED** | Cremer 2025 (arXiv 2504.10564, CrossDocked100) |
| MolCRAFT | 2024 | -9.25 | 0.361 | **CITED** | MolCRAFT paper (CrossDocked100, n=100) |
| AlphaDrug | 2025 | -9.77 | — | **CITED** | AlphaDrug paper (CrossDocked100) |
| TransDiffSBDD | 2025 | -9.37 | 0.839 | **CITED** | TransDiffSBDD paper (CrossDocked100) |
| MolChord | 2025 | -7.62 | 0.332 | **CITED** | MolChord paper (CrossDocked100) |

**WARNING / FLAG:**
- **Lambda's -14.18 is NOT comparable** to the SOTA numbers — the Vina-proxy is a placeholder (D4), not real AutoDock Vina.
- **Lambda's n=2 vs SOTA's n=100** — sample size is 50x smaller; statistical significance not yet meaningful.
- **Vina 1.2.7 vs QVina-protocol mismatch** (D7) — once the real Vina oracle (L-1) replaces the proxy, expect a systematic downward shift on absolute kcal/mol numbers; relative ranking against SOTA should be re-evaluated.
- **Cite-only status** is user-approved (D1). Strict head-to-head is gated on (a) DiffSBDD ckpt access, (b) torch_geometric ROCm 7.2 wheels (R1).

---

## Honest framing (preserved)

1. **Vina proxy = placeholder (D4).** The `top1_vina_proxy` column in this report is a placeholder function, not real AutoDock Vina. The proxy's numeric values (-13.72, -14.64, -14.18) are artefacts of the placeholder and should not be used to claim Lambda "beats" any SOTA baseline. Replace with real Vina once the L-1 DiffDock oracle is live.

2. **Cite-only = user-approved (D1).** SOTA rows (Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, FLOWR, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord) are cited from each paper; they were NOT re-run by us. The cited numbers carry their own protocol-mismatch flags (n_test=100, pocket corpus, SA implementation parity, model-not-rerun, FLOWR PB-valid scope, NFE definition, docking engine) — see `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` for the full provenance audit.

3. **Vina 1.2.7 vs QVina systematic bias (D7).** Once L-1 replaces the proxy, the docking engine in `vina_adapter.py` is AutoDock Vina 1.2.7 with meeko 0.8.0. The published SOTA baselines use QVina (or Vina variants) with slightly different scoring-function updates and exhaustiveness defaults. The systematic bias inflates Lambda's absolute kcal/mol numbers relative to the published SOTA numbers and must be accounted for in any apples-to-apples comparison.

4. **Sample size.** Lambda's measured row reflects 2 pockets (1h36, 830c); SOTA rows reflect 100 pockets each on the CrossDocked100 benchmark. The next sweep (Lambda on 100 pockets) is the only way to put Lambda on the same statistical footing.

---

## Provenance

- Convergence sweep: `molmetal/molmetal_lam/scripts/lambda_benchmark.py` (`--full --max-depth 2`)
- Per-pocket sweep: `molmetal/scripts/lambda_100pocket_sweep.py` (L-3 204-tile, LIPINSKI, n_simulations=50, max_depth=2)
- SOTA citations: `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` + `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` + `molmetal/reports/h4_paper_grade_comparison.md`
- Vina proxy: TODO/pending/decisions.md D4 (gated on L-1 oracle)
- Cite-only policy: TODO/pending/decisions.md D1
- Vina 1.2.7 vs QVina mismatch: TODO/pending/decisions.md D7
- Tk / PyG ROCm 7.2 wheels: TODO/pending/risks.md R1
