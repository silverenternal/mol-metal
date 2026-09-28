# WF-Lambda Phase-4 vs R16 Baseline — 100-pocket × 3-seed Aggregate Comparison
Date: 2026-09-25. Verifier: claude-code. Phase-4 source: `logs/lambda_p4_sweep/aggregate.json` (300 cells, 18-rule click+SyntheMol registry, n_sim=20, metal_seed=None, sa_weight=0.0).

## (1) Per-metric table
| Metric | Phase-4 (n=300, mean ± std) | R16 baseline (per TODO-14 + `wf_r16_*`) | Target | Hit (P4 vs R16) |
|---|---:|---:|---:|---|
| n_distinct | **1.00 ± 0.00** | 20.0 (R13 30cell) / 1.0 (PB 15cell baseline) | ≥ 3 | **MISS / LOSE** (1 vs 20) |
| sa_mean | **6.00 ± 0.00** | 3.66 (R13 30cell) | < 3.5 | **MISS / LOSE** (+2.34) |
| qed_mean | **0.621 ± 0.000** | 0.708 (R13 30cell) | ≥ 0.55 | **PASS / LOSE** (-0.087) |
| div_tanimoto | **0.00 ± 0.00** | 0.1065 (R13 30cell) | ≥ 0.6 | **MISS / LOSE** (-0.107) |
| synth_rate | **1.00 ± 0.00** | 1.00 (R13 30cell) | ≥ 0.95 | **PASS / TIE** |
| validity_rate | **1.00 ± 0.00** | 1.00 (R13 30cell) | == 1.0 | **PASS / TIE** |
| pb_pass_rate | **NULL** (0 docked / 300 cells) | 0.80 strict (12/15, R16 PB smoke) | ≥ 0.6 | **N/A / LOSE** by construction (search-bound) |
| candidates/cell | `[Cl][Pt][Cl]` × 1 | 20-SMILES set (R16 30cell) | ≥ 3 distinct | **MISS / LOSE** |

NOTE: `std=0.00` reflects 3 seeds (42/0/1234) producing bit-identical outputs at n_sim=20 with metal_seed=None — see `aggregate.csv`; per-seed dispersion is single-candidate determinism, not true variance.

## (2) Wins / losses
- **Wins (3/8):** synth_rate (tie), validity_rate (tie), qed_mean (passes target but loses to R16).
- **Losses (5/8):** n_distinct, sa_mean, div_tanimoto, pb_pass_rate, candidates/cell. The 18-rule sweep **regressed** every diversity metric vs R16 30-cell because the n_sim=20 budget + metal_seed=None collapses to the singleton `[Cl][Pt][Cl]` seed on 100/100 pockets.

## (3) Verdict
**NO.** The 18-rule 100-pocket × 3-seed sweep did **not** beat R16 Path A by ≥3 metal-ligand pairs. Phase-4 produced **0 metal-ligand pairs** (candidate set = `{[Cl][Pt][Cl]}` × 300 cells; metal_compliance_rate=0.0 across the cohort). R16 Path A also collapsed to n_distinct=1, so neither arm advanced the singleton-attractor ceiling; the 18-rule registry expanded branching 3.6× but n_sim=20 cut the search depth below the n_distinct≥3 threshold. **Cause:** spec mismatch — phase-4 ran without `--metal-seed cisplatin` AND with `--sa-weight 0.0`, both of which R12/R16 found load-bearing for the diversity lift.

## (4) Recommendation for Phase-5 (3-axis SOTA benchmark)
1. **Restore the load-bearing flags:** `--metal-seed cisplatin --sa-weight 0.3 --n-simulations 1000` (R12/R16 winning config).
2. **Lift n_simulations to ≥ 1000** before measuring n_distinct; n_sim=20 is a structural floor that prevents the 18-rule expansion from compounding.
3. **Per-pocket reference resolver** (sub-fix C) + Deflex v2 macro features must be active — current phase-4 leaves both off, so the candidate set never varies per pocket.
4. **Phase-5 targets:** only after step 1-3 lift `n_distinct ≥ 3` on ≥80% of test_010..test_019 (R16-W44 commit), add the 3-axis SOTA bars (TransDiffSBDD Vina, Pocket2Mol diversity, SyntheMol synth).
5. **Honest framing carries forward:** phase-4 is **NEGATIVE-RESULT MEASURED** (registry wires correctly; metric lift requires budget + flag stack, not registry extension).
