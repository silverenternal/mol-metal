# WF-Lambda-Fix-Singleton — Round-12 Re-Pilot after 3 Fixes

> **Goal:** re-run the Round-12 Lambda Pilot at `n_simulations=1000` after
> shipping the 3 algorithmic fixes (F1 soft tiered metal prior, F2 reward
> rebalance, F3 truthful metal_compliance) and measure whether the singleton
> collapse lifts.
> **Honest framing:** the 3 fixes shipped, the harness is healthier, but
> `n_distinct` **did not** lift above 1. See §2 for the post-fix diagnosis
> and §6 for honest summary.

## 1. Configuration (MEASURED)

| field | value |
|---|---|
| CLI | `uv run python molmetal/scripts/r4_lambda_only_run.py --pockets 10 --seeds 42 0 1234 --n-simulations 1000 --n-top-k 20 --metal-seed cisplatin --click-rules all-5 --output-dir wf_lambda_fix_singleton/r4c` |
| n_pockets | 10 (test_000 … test_009) |
| n_seeds | 3 (42, 0, 1234) |
| n_simulations / cell | **1000** (SAFETY_MAX=10000; no clamp applied) |
| n_top_k / cell | 20 |
| n_cells | 30 |
| metal_seed | `cisplatin` → `[Pt]C#C` (Fix-1 pre-saturated-removed seed) |
| click_rules | all-5 (AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne) |
| sa_weight | 0.0 (default; λ-only) |
| reward weighting | F2 active: `w_click=0.5`, `w_metal=0.5`, `valid=1.0`, `syn=1.0`, `aeq=1.0` |
| metal prior | F1 active: soft tiered `1.0/0.5/0.2/0.0` (no longer hard gate) |
| metal_compliance metric | F3 active: `metal_compliance_truthful` (seed-excluded) |
| total wall-clock | **43.56 s** (1.45 s/cell, well under 30-min budget) |
| Python | 3.12 (uv-managed) |
| ROCm / Triton | 7.2 / 3.8.0 / gfx1101 wave64 |

Note: the script auto-prepends `wf_lambda1_` to the `--output-dir` value, so
the actual on-disk dir is `wf_lambda1_wf_lambda_fix_singleton/r4c/`. We
copy the artefacts below under the spec-required name
`molmetal/reports/wf_lambda_fix_singleton/`.

## 2. Honest finding — singleton collapse NOT lifted

**Every cell still collapsed to 1 candidate.** The unique emitted SMILES
across all 30 cells is exactly one:

```
C#[C][Pt]
```

This is `[Pt]C#C` (the **new** cisplatin seed from Fix 1 — bare metal with
one terminal alkyne handle), **not** the prior baseline's
`[NH2][Pt]([NH2])([Cl])[Cl]` (the pre-Fix-1 cisplatin). The fix changed
which seed is used; it did **not** break the collapse. All 30 cells share
`n_candidates=1, n_distinct=1`, byte-identical candidates, std=0 across
every metric column.

| metric | WF-Round12-Lambda-Pilot baseline (pre-fix, n_sim=1000) | this run (post-fix, n_sim=1000) | delta |
|---|---|---|---|
| emitted SMILES | `[NH2][Pt]([NH2])([Cl])[Cl]` | `C#[C][Pt]` | **seed changed** (Fix-1 consequence) |
| n_distinct | 1 | 1 | 0 |
| diversity_tanimoto | 0.000 | 0.000 | 0 |
| diversity_homotype | 0.000 | 0.000 | 0 |
| metal_compliance_rate | **1.000** (trivially true) | **0.000** (truthful, F3 working) | **−1.000** (F3 truthfulness restored) |
| anticancer_index | 0.425 | 0.100 | −0.325 (single different molecule) |
| qed_mean | 0.671 | 0.509 | −0.162 |
| coordination_number_mean | 4.000 | 1.000 | −3.000 (seed is undercoordinated) |
| monodentate_cl_count | 2 | 0 | −2 (no Cl in new seed) |
| n_cells | 30 | 30 | unchanged |
| wall-clock | 50.69 s | 43.56 s | −7.13 s (1.45 s/cell vs 1.69 s/cell) |

**Honest interpretation of the regression in metal_compliance_rate** (1.000
→ 0.000): this is **the F3 fix doing its job** — the prior metric was
trivially-true-by-construction (every cell returned the seed cisplatin, so
"contains Pt" was always 1.0). The new metric excludes the seed itself
(per `metal_compliance_truthful` in `r4_lambda_only_run.py:521-562`), so
the only candidate (`C#[C][Pt]`) — which IS the seed — is now correctly
excluded and the cell reports 0.000. The truthfulness is a metric-design
win, not a chemistry regression.

**Honest interpretation of singleton persistence** (n_distinct=1 across
all 30 cells): the 3 fixes are **necessary but not sufficient**. F1
softens the reward surface (the seed `[Pt]C#C` has bare-Pt coord=0, which
now scores 0.2 instead of 0.0); F2 rebalances click vs metal weights
(0.5/0.5 instead of 1.0/1.0); F3 fixes the metric. But the **structural
F2 problem — 5 organic click rules × bare-Pt_alkyne seed** — remains.
The MCTS still finds zero reducible children (CuAAC needs azide+alkyne
in the seed; SPAAC same; ThiolEne needs alkene+thiol; Suzuki needs
boronic acid+halide; AmideCoupling needs COOH+NH2). The seed has only
one warhead (terminal alkyne) — none of the 5 rules can apply.

For n_distinct to lift above 1, **F2(a) — new SMARTS rules
(`MetalLigandExchange`, `AquaExchange`)** is required. That fix is
in-progress (task #608) and was NOT shipped before this re-pilot.

## 3. Per-pocket per-seed table (MEASURED)

| pocket | seed | n_cand | n_distinct | valid | synth | uniq | metal | div_tan | div_homo | novel | logp | tpsa | rotb | coord | cl | gsh | dna | anticancer | sa | qed | com_shift | rigid_rmsd | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.49 |
| test_000 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.40 |
| test_000 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.55 |
| test_001 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.39 |
| test_001 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.46 |
| test_001 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.39 |
| test_002 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.41 |
| test_002 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.40 |
| test_002 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.39 |
| test_003 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.46 |
| test_003 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.41 |
| test_003 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.51 |
| test_004 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.42 |
| test_004 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.42 |
| test_004 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.42 |
| test_005 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.51 |
| test_005 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.51 |
| test_005 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.49 |
| test_006 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.41 |
| test_006 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.43 |
| test_006 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.42 |
| test_007 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.50 |
| test_007 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.49 |
| test_007 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.51 |
| test_008 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.46 |
| test_008 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.46 |
| test_008 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.46 |
| test_009 | 42   | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.45 |
| test_009 | 0    | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.49 |
| test_009 | 1234 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.124 | 0.0 | 0.0 | 1.0 | 0 | 0.000 | 0.400 | 0.100 | 6.008 | 0.509 | 0.000 | 0.000 | 1.43 |

> `oxidation_state_distribution` per cell = `{Pt_0=1}` (single candidate → single Pt_0 entry, since the seed is bare Pt with one C#C, not yet coordinated).
>
> `warnings` per cell: `sa_weight=0.000`, `click_rules_active=['AmideCoupling', 'CuAAC', 'SPAAC', 'Suzuki', 'ThiolEne']`, `metal_seed_active=cisplatin smi=[Pt]C#C`.

## 4. Per-pocket aggregate (mean over 3 seeds)

| pocket | n_cand | valid | synth | uniq | metal | div_tan | div_homo | novel | anticancer | qed | sa | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.481±0.061 |
| test_001 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.413±0.034 |
| test_002 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.402±0.009 |
| test_003 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.460±0.041 |
| test_004 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.421±0.000 |
| test_005 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.503±0.009 |
| test_006 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.421±0.008 |
| test_007 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.500±0.005 |
| test_008 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.461±0.000 |
| test_009 | 1.00±0.00 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.100 | 0.509 | 6.008 | 1.456±0.025 |

> Note: std=0.000 across all metric columns because every seed produces
> byte-identical candidates — the seed `[Pt]C#C` is the only thing the
> MCTS emits per cell. The non-zero std on wall-clock (mean 1.45 s/cell
> ± 0.038 s across 30 cells) is pure Python timing jitter, not chemistry
> variation. **n_distinct_mean across pockets = 1.00** (was 1.00 in
> baseline; no lift).

## 5. Cross-pocket × cross-seed aggregate (30-cell mean ± std)

| metric | mean | std | min | max |
|---|---|---|---|---|
| validity_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| synthesizability_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| uniqueness_rate | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **metal_compliance_rate** | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| diversity_tanimoto | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| diversity_homotype | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| novelty | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| logp_mean | 0.1239 | 0.0000 | 0.1239 | 0.1239 |
| tpsa_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rotb_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| coordination_number_mean | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| monodentate_cl_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gsh_evasion_score | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dna_kb_proxy | 0.4000 | 0.0000 | 0.4000 | 0.4000 |
| anticancer_index | 0.1000 | 0.0000 | 0.1000 | 0.1000 |
| sa_mean | 6.0081 | 0.0000 | 6.0081 | 6.0081 |
| qed_mean | 0.5087 | 0.0000 | 0.5087 | 0.5087 |
| com_shift_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rigid_rmsd_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| n_distinct | 1.0000 | 0.0000 | 1 | 1 |
| elapsed_s_total | 43.56 s | — | — | — |

**Pool of unique SMILES across all 30 cells:** exactly **1** (`C#[C][Pt]`).
**n_distinct_mean across pockets = 1.00.**

## 6. Honest framing summary

> The 3 algorithmic fixes (F1 soft tiered metal prior, F2 reward
> rebalance w_click=w_metal=0.5, F3 truthful metal_compliance) shipped
> to `molmetal/scripts/r4_lambda_only_run.py` and the harness is healthier:
> the SAFETY_MAX cap lifted to 10000, the bare-Pt seed unblocks metal
> expansion potential, the truthful metric no longer conflates "seed has
> metal" with "MCTS generated new metal-bearing molecules". The
> re-pilot runs deterministically in 43.56 s for 30 cells, well under the
> 30-min budget, and produces a single SMILES `C#[C][Pt]` per cell —
> **n_distinct=1 across all cells, identical to baseline**. The singleton
> collapse is **NOT lifted** by F1+F2+F3 alone. The structural F2(a)
> problem remains: 5 organic click rules (CuAAC/SPAAC/ThiolEne/Suzuki/
> AmideCoupling) are chemically inapplicable to a bare Pt_alkyne seed,
> so the MCTS finds zero reducible children, the cache marks the seed
> as terminal, and only the seed itself is emitted. To break the
> collapse, **F2(a) `MetalLigandExchange` + `AquaExchange` SMARTS rules
> (in-progress, task #608) must ship before the next re-pilot**. The
> post-fix metrics DO show F3's truthfulness restored (metal_compliance
> 1.000 → 0.000 honest), and F2's reward rebalance is in place — these
> are honest wins but do not show up as diversity lift because n_distinct
> is mathematically 0-trivially-locked at 1. The Round-12 scientific
> claim ("λ-only MCTS can produce diverse Pt(II) complexes") **remains
> unproven**; this re-pilot demonstrates that the 3 reward/metric fixes
> alone are insufficient and motivates the structural F2(a) fix.

## 7. Spec acceptance criteria

| criterion | spec | this run | status |
|---|---|---|---|
| `n_distinct_after > 1` | required | 1 | **NOT MET** — singleton collapse not lifted by F1+F2+F3 alone |
| `diversity_tanimoto_after > 0.000` | required | 0.000 | NOT MET — degenerate by n_distinct=1 |
| `metal_compliance_after` reflects truthful (non-seed) | required | 0.000 | **MET** (F3 working: seed-excluded, post-fix honest) |
| `lifted_from_baseline` | positive | **0** (n_distinct unchanged) | NOT MET — see §2 for F2(a) dependency |
| `honest_framing` | mandatory | §6 above | MET — collapse diagnosis + F2(a) dependency + truthfulness all documented |
| n_pockets | 10 | 10 | MET |
| n_seeds | 3 | 3 | MET |
| n_cells_total | 30 | 30 | MET |
| wall_clock_total | ≤30 min | 43.56 s | MET (40× under budget) |
| per_pocket_validity_mean | ≥0.99 | 1.000 | MET |
| per_pocket_synthesizability_mean | ≥0.99 | 1.000 | MET |
| per_pocket_metal_compliance_mean (truthful) | reported | 0.000 | MET (truthful) |
| per_pocket_diversity_tanimoto_mean | reported | 0.000 | MET (degenerate) |
| per_pocket_diversity_homotype_mean | reported | 0.000 | MET (degenerate) |
| per_pocket_novelty_mean | ≥0.95 | 1.000 | MET |
| per_pocket_anticancer_index_mean | reported | 0.100 | MET (single-molecule) |
| all 21 metrics populated | required | yes | MET (incl. wall_clock + nfe + oxidation_state_distribution) |

## 8. Files

- Script: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`
- Per-cell JSON + summary (auto-renamed by harness): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lambda_fix_singleton/r4c/report.json` + `summary.md`
- This report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda_fix_singleton/final.md`
- Baseline (pre-fix): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_round12_lambda_pilot/final.md`
- Fix sources: `wf_lambda_internal_review/diagnose.md §2` (F1 + F3), `wf_mcts_chemistry_research/recommendations.md` (F2 ranking + evidence)

**Wall-clock budget used:** 43.56 s of 30-min allowance.
**Lift achieved:** Δ `n_distinct` = **0** (vs target > 1); Δ `div_tan` = **0.000**; F3 truthfulness restored (metal_compliance 1.000 → 0.000 honest).
**Next step (out of scope, task #608 in-progress):** ship `MetalLigandExchange` + `AquaExchange` SMARTS rules (F2(a)), then re-run with same CLI to verify `n_distinct ≥ 2` lift.
