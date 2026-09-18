# WF-Lambda-Diversity-Rotation — final report

**Date:** 2026-09-14
**Goal:** lift diversity_tanimoto from 0.005 (singleton collapse at n_sim=100) toward 0.10–0.20 by:
(a) increasing n-simulations from 100 to 1000
(b) rotating metal seeds across {cisplatin, ru_arene, ir_cp_star}
(c) keeping click-rules all-5 enabled
**Hypothesis (paper §5 ablation):** rotating metal seeds + 10× more sim budget yields non-degenerate diversity panel.

---

## 1. Configuration

| Parameter            | Value                                                |
|----------------------|------------------------------------------------------|
| n_pockets            | 5 (test_000 … test_004)                              |
| n_seeds              | 1 (seed 42)                                          |
| n_metal_seeds        | 3 (cisplatin, ru_arene, ir_cp_star)                  |
| n_simulations (CLI)  | **1000** (but hard-capped at 100 — see §2)           |
| n_top_k              | 20                                                   |
| click_rules          | all-5 (active: AmideCoupling/CuAAC/SPAAC/Suzuki/ThiolEne; "amide coupling"+"amide_coupling"+"cuaac"+"spaac"+"suzuki"+"thiol-ene" alias list appended) |
| prior_enabled        | True                                                 |
| manifest             | `molmetal/data/crossdocked100_manifest.csv`          |

Script: `molmetal/scripts/r4_lambda_only_run.py` v (round-4 orchestrator).
Output dirs: `molmetal/reports/wf_lambda_div_rotation/{cisplatin,ru_arene,ir_cp_star}/`.

---

## 2. CRITICAL FINDING — silent n_simulations hard-cap

`r4_lambda_only_run.py:1466-1467` contains:

```python
if args.n_simulations > 100:
    args.n_simulations = 100  # hard cap per spec
```

Every invocation in this workflow, despite the CLI `--n-simulations 1000`, was silently **capped at 100**. Every cell's `n_simulations` field in the per-cell JSON confirms this (100 everywhere; never 1000). The diversity-lift hypothesis that depends on 10× more simulation budget is therefore **not testable with the current harness** without lifting the cap.

This is a known harness-level invariant (the hard-cap was deliberately added "per spec" in round-4). For this run, the actual budget used is `n_sim=100` per cell × 5 pockets × 3 metal seeds = 1500 evals total. Wall-clock: ~5 s / 5 cells / metal seed = 16 s total across all 3 seeds (far under the 30 min budget).

---

## 3. Per-metal-seed aggregate table

| metal_seed    | n_cells | validity | uniqueness | synth | metal_compliance | diversity_tanimoto | diversity_homotype | novelty | ref_tanimoto | logP | TPSA | rotB | CN | monodCl |
|---------------|---------|----------|------------|-------|------------------|--------------------|--------------------|---------|--------------|------|------|------|----|---------|
| cisplatin     | 5       | 1.000    | 1.000      | 1.000 | **1.000**        | **0.000**          | **0.000**          | 1.000   | 0.0120       | 0.20 | 52.0 | 0.0  | 4  | 10      |
| ru_arene      | 5       | 1.000    | 1.000      | 1.000 | **1.000**        | **0.000**          | **0.000**          | 1.000   | 0.0903       | 1.92 | 52.0 | 2.0  | 6  | 10      |
| ir_cp_star    | 5       | 1.000    | 1.000      | 1.000 | **0.000**        | **0.000**          | **0.000**          | 1.000   | 0.0478       | 2.52 | 52.0 | 1.0  | 4  | 5       |
| **mean**      | —       | 1.000    | 1.000      | 1.000 | **0.667**        | **0.000**          | **0.000**          | 1.000   | 0.0500       | 1.55 | 52.0 | 1.0  | 4.7| —       |

`mean` is the cross-seed aggregate over 15 cells (5 cells × 3 seeds).

**Per-pocket reproducibility check:** for each metal seed, all 5 pockets produced **byte-identical** candidates lists:
- cisplatin: `[NH2][Pt]([NH2])([Cl])[Cl]` × 5
- ru_arene: `[NH2][Ru]([NH2])([Cl])([Cl])([c]1ccccc1)[c]1ccccc1` × 5
- ir_cp_star: `CC1=C(C)[CH]([Ir]([NH2])([NH2])[Cl])C(C)=C1C` × 5

`n_distinct=1, n_candidates=1` in **every** cell. **The Lambda-only harness returns the metal seed verbatim** (the only candidate it surfaces per pocket is the seed itself, with minor canonicalisation: `[NH3]→[NH2]` for cisplatin, kekulé-aromatic `[c]` for ru_arene's phenyls).

---

## 4. Singleton collapse diagnosis

| Metric             | Baseline (WF-Lambda-Metal-Pilot, cisplatin-only, n_sim=100) | Rotation (this run, 3 seeds, n_sim=100) | Lift |
|--------------------|-------------------------------------------------------------|-----------------------------------------|------|
| diversity_tanimoto | 0.000                                                       | 0.000                                   | Δ = 0.000 |
| diversity_homotype | 0.000                                                       | 0.000                                   | Δ = 0.000 |
| metal_compliance   | 1.000 (cisplatin only)                                      | 0.667 (3-seed mean; ir_cp_star = 0.0)  | **Δ = -0.333** (regresses) |
| novelty            | 1.000                                                       | 1.000                                   | Δ = 0.000 |
| validity           | 1.000                                                       | 1.000                                   | Δ = 0.000 |
| synthesizability   | 1.000                                                       | 1.000                                   | Δ = 0.000 |

**Honest interpretation:** the diversity goal of this workflow (lift from 0.005 → 0.10–0.20) **was not achieved**. The Lambda-only harness continues to exhibit singleton collapse: every (pocket, metal-seed) cell returns exactly 1 candidate (= the seed SMILES with canonicalisation-only edits). Rotating across 3 metal seeds does produce 3 distinct seeds (cisplatin, ru_arene, ir_cp_star), which gives a **cross-seed** diversity signal — but the harness's per-cell `diversity_tanimoto` metric is undefined for n_distinct=1 (trivially 0), so it cannot surface the cross-seed lift numerically.

The metal_compliance regression (1.0 → 0.667) is real: ir_cp_star's `CC1=C(C)[CH]([Ir]([NH2])([NH2])[Cl])C(C)=C1C` fails the metal-compliance predicate (only 1 monodentate Cl, not the required ≥2; oxidation-state distribution shows `Ir_0` × 5, but the rubric expects Ir(III) or higher with 2 leaving groups for "compliance" under the lambda-only reward spec). This is a useful negative finding — the iridium Cp* scaffold is **not** lambda-only-compliant without the click-rules engine filling in the second Cl.

---

## 5. Cross-seed aggregation

**Pool of unique SMILES across all 15 cells:** exactly 3 (the three seed scaffolds).
- `[NH2][Pt]([NH2])([Cl])[Cl]` (cisplatin canonicalised)
- `[NH2][Ru]([NH2])([Cl])([Cl])([c]1ccccc1)[c]1ccccc1` (ru_arene canonicalised)
- `CC1=C(C)[CH]([Ir]([NH2])([NH2])[Cl])C(C)=C1C` (ir_cp_star canonicalised)

Cross-seed pairwise Tanimoto distances (Morgan radius=2, 2048-bit, RDKit): all three seeds are Tanimoto-dissimilar (≤0.05 between any pair). So if the harness were asked to compute diversity over the **pool** of 3 seeds, the diversity_tanimoto would be ~0.95+. But the per-cell metric is what's reported.

---

## 6. Lift vs WF-Lambda-Metal-Pilot baseline

The baseline (`wf_lambda_metal_pilot/cisplatin`, n_sim=100) reported `diversity_tanimoto=0.000, diversity_homotype=0.000, metal_compliance_rate=1.000`. This run's cisplatin arm reproduces the baseline **byte-for-byte** (same cells, same SMILES, same n_distinct=1). The cross-seed rotation adds 2 more seeds (ru_arene, ir_cp_star) but does not change the per-cell diversity_tanimoto because:

1. **Harness-level cap:** n_sim=100 (capped from CLI n_sim=1000).
2. **Generator-level singleton emission:** the Lambda-only generator emits exactly 1 candidate per cell (the seed itself).
3. **Diversity metric undefined for n=1:** Tanimoto over a set of size 1 is trivially 0; the metric does not signal cross-seed diversity.

**Verdict on hypothesis:** ✗ Rejected. Rotating metal seeds across {cisplatin, ru_arene, ir_cp_star} while keeping Lambda-only mode does **not** lift diversity_tanimoto above 0.000. The paper §5 ablation table should label this row as "singleton collapse, n_distinct=1 across all cells; cross-seed pool size = 3 distinct scaffolds."

---

## 7. What's needed to actually lift diversity

To genuinely reach 0.10–0.20 diversity_tanimoto, the harness needs **at least one** of:

1. **Lift the n_simulations hard-cap** in `r4_lambda_only_run.py:1466-1467` so the CLI flag actually takes effect, then re-run. At n_sim=1000 the MCTS rollout has 10× more budget to mutate the seed and surface diverse children.
2. **Switch generator off Lambda-only** — the CFM/CFG joint generator (with `bond_pattern_mask`) emits n_top_k=20 distinct candidates per cell (see R10 evidence). Lambda-only mode by design returns the seed verbatim; that's a property of the generator, not a bug.
3. **Activate the click-rules engine properly** — the warnings show `click_rules_filter_emptied: requested=['all-5']; falling back to all-5`, meaning the `--click-rules all-5` filter is not gating anything (the filter name doesn't match the registered names). The "all-5" alias needs to be wired in `click_rules_active` to actually constrain the Lambda generator.

Until one of these is done, the WF-Lambda-Diversity-Rotation cannot produce a non-degenerate diversity panel for paper §5.

---

## 8. Files

- `molmetal/reports/wf_lambda_div_rotation/cisplatin/{report.json,summary.md}`
- `molmetal/reports/wf_lambda_div_rotation/ru_arene/{report.json,summary.md}`
- `molmetal/reports/wf_lambda_div_rotation/ir_cp_star/{report.json,summary.md}`
- `molmetal/reports/wf_lambda_div_rotation/final.md` (this file)

**Wall-clock budget used:** ~16 s of 30 min allowance.
**Lift achieved:** Δ diversity_tanimoto = **0.000** (vs target 0.10–0.20).
**Cross-seed pool size:** 3 distinct metal-organic scaffolds.
