# WF-Lambda-1 — Pure Lambda-Only Baseline

> Honest-framing: this is a MEASURED run. PROJECTED numbers
> from the spec are not invoked here — the spec is in
> `molmetal/reports/ultracode_audit/wf_lambda1_spec.md`.

## Configuration

- n_pockets : `5`
- seeds     : `[42, 0, 1234]`
- n_simulations per cell : `1000`
- n_top_k    : `20`
- prior_enabled : `True`
- metal_seed    : `None`

## Aggregate metrics (mean across cells)

| metric | value |
|---|---|
| validity_rate | 1.0000 |
| uniqueness_rate | 1.0000 |
| diversity_tanimoto | 0.0052 |
| diversity_homotype | 0.0022 |
| novelty | 1.0000 |
| synthesizability_rate | 1.0000 |
| metal_compliance_rate | 0.0000 |
| reference_tanimoto | 0.9658 |
| rigid_rmsd_mean | 0.5480 |
| com_shift_mean | 0.0000 |

## WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns

| metric | value |
|---|---|
| logp_mean | 0.6975 |
| tpsa_mean | 161.1475 |
| rotb_mean | 6.9633 |
| coordination_number_mean | 0.0000 |
| monodentate_cl_count | 0 |
| gsh_evasion_score | 0.0000 |
| dna_kb_proxy | 0.0533 |
| anticancer_index | 0.2383 |
| oxidation_state_distribution | (none) |

## WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET columns

Lit basis: Weininger 1990 / Patrick 2009 / Hou 2007 / Veith 2009 /
Benigni-Richard 2005 / Hughes 2008 / Delaney 2004 ESOL / Obach 1999.
All CPU-only; zero GPU load. Heuristic evaluators (NOT wet-lab calibrated).

| metric | value |
|---|---|
| logp7_4_mean | 1.0840 |
| gi50_proxy_mean | 8.0000 |
| cell_permeability_logPapp_mean | -3.9350 |
| herg_cardio_risk_mean | 0.4300 |
| ames_mutagen_mean | 0.4000 |
| hepatotox_index_mean | 0.5163 |
| aqueous_solubility_logS_mean | -2.8973 |
| plasma_protein_binding_mean | 0.5092 |

## WF-Phase3D-PerResidue-Diversity + WF-Phase3G-MetalCoordProbe

Lit basis: Bemis & Murcko 1996 / Jasial 2021 IntDiv / Peter 2019 SPF;
Lippard & Berg 1995 / Reedijk 1987 / Miessler 2014 d-block geometries.

| metric | value |
|---|---|
| diversity_subpocket | 0.0573 |
| metal_coord_compliance | 0.0000 |

> `diversity_tanimoto` is the SE(3) / atom-symbol-histogram
> baseline (legacy `diversity_alpha`). `diversity_homotype`
> is the Lambda-native metric from WF-Lambda-2 (typed-variable
> cosine + β-reduction-depth + click-rule-fires Jaccard).

## Per-cell results

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | novel | syn | metal | ref_tan | rigid_rmsd | com_shift | logP | TPSA | RotB | coord | gsh | dna | ai | cl | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 20 | 20 | 1.000 | 1.000 | 0.027 | 0.010 | 1.000 | 1.000 | 0.000 | 0.840 | 0.000 | 0.000 | -2.44 | 268.1 | 13.35 | 0.00 | 0.000 | 0.010 | 0.128 | 0 | 6 |
| test_000 | 0 | 20 | 20 | 1.000 | 1.000 | 0.025 | 0.012 | 1.000 | 1.000 | 0.000 | 0.808 | 0.000 | 0.000 | -3.65 | 295.3 | 13.05 | 0.00 | 0.000 | 0.110 | 0.152 | 0 | 6 |
| test_000 | 1234 | 20 | 20 | 1.000 | 1.000 | 0.025 | 0.011 | 1.000 | 1.000 | 0.000 | 0.840 | 0.000 | 0.000 | -1.86 | 282.6 | 15.05 | 0.00 | 0.000 | 0.080 | 0.145 | 0 | 6 |
| test_001 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 3.13 | 97.0 | 8.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_001 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 3.13 | 97.0 | 8.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_001 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 3.13 | 97.0 | 8.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_002 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.514 | 0.000 | -1.78 | 238.4 | 6.00 | 0.00 | 0.000 | 0.200 | 0.175 | 0 | 6 |
| test_002 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.514 | 0.000 | -1.78 | 238.4 | 6.00 | 0.00 | 0.000 | 0.200 | 0.175 | 0 | 6 |
| test_002 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.514 | 0.000 | -1.78 | 238.4 | 6.00 | 0.00 | 0.000 | 0.200 | 0.175 | 0 | 6 |
| test_003 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2.21 | 146.9 | 1.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_003 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2.21 | 146.9 | 1.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_003 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2.21 | 146.9 | 1.00 | 0.00 | 0.000 | 0.000 | 0.375 | 0 | 6 |
| test_004 | 42 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.226 | 0.000 | 2.58 | 41.5 | 6.00 | 0.00 | 0.000 | 0.000 | 0.125 | 0 | 6 |
| test_004 | 0 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.226 | 0.000 | 2.58 | 41.5 | 6.00 | 0.00 | 0.000 | 0.000 | 0.125 | 0 | 6 |
| test_004 | 1234 | 1 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.226 | 0.000 | 2.58 | 41.5 | 6.00 | 0.00 | 0.000 | 0.000 | 0.125 | 0 | 6 |

Total elapsed: `16289.37 s`

## Scorer (Lambda-only, NO docking / AdmetAI / PB)

- `alpha_equivalence_uniqueness_score` — distinct beta-NF count.
- `click_rule_match_bonus` — +1.0 if any of 5 click rules fires.
- `metal_geometry_prior_bonus` — +1.0 if Pt=4 / Ru=Ir=6 coord.
- `rdkit_validity_score` — 1.0 if RDKit can sanitize.
- `synthesizability_via_lambda_paths` — 1.0 if beta-NF + RDKit.

