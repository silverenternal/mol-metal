# WF-Lambda-1 — Pure Lambda-Only Baseline

> Honest-framing: this is a MEASURED run. PROJECTED numbers
> from the spec are not invoked here — the spec is in
> `molmetal/reports/ultracode_audit/wf_lambda1_spec.md`.

## Configuration

- n_pockets : `5`
- seeds     : `[42]`
- n_simulations per cell : `100`
- n_top_k    : `20`
- prior_enabled : `True`
- metal_seed    : `ir_cp_star`

## Aggregate metrics (mean across cells)

| metric | value |
|---|---|
| validity_rate | 1.0000 |
| uniqueness_rate | 1.0000 |
| diversity_tanimoto | 0.0000 |
| diversity_homotype | 0.0000 |
| novelty | 1.0000 |
| synthesizability_rate | 1.0000 |
| metal_compliance_rate | 0.0000 |
| reference_tanimoto | 0.0478 |

## WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns

| metric | value |
|---|---|
| logp_mean | 2.5164 |
| tpsa_mean | 52.0400 |
| rotb_mean | 1.0000 |
| coordination_number_mean | 4.0000 |
| monodentate_cl_count | 5 |
| gsh_evasion_score | 0.0000 |
| dna_kb_proxy | 0.5000 |
| anticancer_index | 0.2083 |
| oxidation_state_distribution | Ir_0=5 |

> `diversity_tanimoto` is the SE(3) / atom-symbol-histogram
> baseline (legacy `diversity_alpha`). `diversity_homotype`
> is the Lambda-native metric from WF-Lambda-2 (typed-variable
> cosine + β-reduction-depth + click-rule-fires Jaccard).

## Per-cell results

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | novel | syn | metal | ref_tan | logP | TPSA | RotB | coord | gsh | dna | ai | cl | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.064 | 2.52 | 52.0 | 1.00 | 4.00 | 0.000 | 0.500 | 0.208 | 1 | 3 |
| test_001 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.045 | 2.52 | 52.0 | 1.00 | 4.00 | 0.000 | 0.500 | 0.208 | 1 | 3 |
| test_002 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.045 | 2.52 | 52.0 | 1.00 | 4.00 | 0.000 | 0.500 | 0.208 | 1 | 3 |
| test_003 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.044 | 2.52 | 52.0 | 1.00 | 4.00 | 0.000 | 0.500 | 0.208 | 1 | 3 |
| test_004 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.040 | 2.52 | 52.0 | 1.00 | 4.00 | 0.000 | 0.500 | 0.208 | 1 | 3 |

Total elapsed: `5.92 s`

## Scorer (Lambda-only, NO docking / AdmetAI / PB)

- `alpha_equivalence_uniqueness_score` — distinct beta-NF count.
- `click_rule_match_bonus` — +1.0 if any of 5 click rules fires.
- `metal_geometry_prior_bonus` — +1.0 if Pt=4 / Ru=Ir=6 coord.
- `rdkit_validity_score` — 1.0 if RDKit can sanitize.
- `synthesizability_via_lambda_paths` — 1.0 if beta-NF + RDKit.

