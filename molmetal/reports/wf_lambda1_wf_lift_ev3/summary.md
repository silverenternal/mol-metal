# WF-Lambda-1 — Pure Lambda-Only Baseline

> Honest-framing: this is a MEASURED run. PROJECTED numbers
> from the spec are not invoked here — the spec is in
> `molmetal/reports/ultracode_audit/wf_lambda1_spec.md`.

## Configuration

- n_pockets : `1`
- seeds     : `[42, 0, 1234]`
- n_simulations per cell : `1000`
- n_top_k    : `20`
- prior_enabled : `True`
- metal_seed    : `cisplatin`

## Aggregate metrics (mean across cells)

| metric | value |
|---|---|
| validity_rate | 1.0000 |
| uniqueness_rate | 1.0000 |
| diversity_tanimoto | 0.1366 |
| diversity_homotype | 0.0450 |
| novelty | 1.0000 |
| synthesizability_rate | 1.0000 |
| metal_compliance_rate | 0.0000 |
| reference_tanimoto | 0.2286 |
| rigid_rmsd_mean | 0.0000 |
| com_shift_mean | 0.0000 |

## WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns

| metric | value |
|---|---|
| logp_mean | 0.3126 |
| tpsa_mean | 43.2625 |
| rotb_mean | 1.8000 |
| coordination_number_mean | 1.0000 |
| monodentate_cl_count | 0 |
| gsh_evasion_score | 0.0000 |
| dna_kb_proxy | 0.7200 |
| anticancer_index | 0.1800 |
| oxidation_state_distribution | Pt_0=60 |

## WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET columns

Lit basis: Weininger 1990 / Patrick 2009 / Hou 2007 / Veith 2009 /
Benigni-Richard 2005 / Hughes 2008 / Delaney 2004 ESOL / Obach 1999.
All CPU-only; zero GPU load. Heuristic evaluators (NOT wet-lab calibrated).

| metric | value |
|---|---|
| logp7_4_mean | 0.2526 |
| gi50_proxy_mean | 8.0000 |
| cell_permeability_logPapp_mean | -4.3392 |
| herg_cardio_risk_mean | 0.2450 |
| ames_mutagen_mean | 0.0000 |
| hepatotox_index_mean | 0.0300 |
| aqueous_solubility_logS_mean | -2.9115 |
| plasma_protein_binding_mean | 0.3007 |

## WF-Phase3D-PerResidue-Diversity + WF-Phase3G-MetalCoordProbe

Lit basis: Bemis & Murcko 1996 / Jasial 2021 IntDiv / Peter 2019 SPF;
Lippard & Berg 1995 / Reedijk 1987 / Miessler 2014 d-block geometries.

| metric | value |
|---|---|
| diversity_subpocket | 0.6790 |
| metal_coord_compliance | 0.0000 |

> `diversity_tanimoto` is the SE(3) / atom-symbol-histogram
> baseline (legacy `diversity_alpha`). `diversity_homotype`
> is the Lambda-native metric from WF-Lambda-2 (typed-variable
> cosine + β-reduction-depth + click-rule-fires Jaccard).

## Per-cell results

| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | novel | syn | metal | ref_tan | rigid_rmsd | com_shift | logP | TPSA | RotB | coord | gsh | dna | ai | cl | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_001 | 42 | 20 | 20 | 1.000 | 1.000 | 0.137 | 0.045 | 1.000 | 1.000 | 0.000 | 0.229 | 0.000 | 0.000 | 0.31 | 43.3 | 1.80 | 1.00 | 0.000 | 0.720 | 0.180 | 0 | 10 |
| test_001 | 0 | 20 | 20 | 1.000 | 1.000 | 0.137 | 0.045 | 1.000 | 1.000 | 0.000 | 0.229 | 0.000 | 0.000 | 0.31 | 43.3 | 1.80 | 1.00 | 0.000 | 0.720 | 0.180 | 0 | 10 |
| test_001 | 1234 | 20 | 20 | 1.000 | 1.000 | 0.137 | 0.045 | 1.000 | 1.000 | 0.000 | 0.229 | 0.000 | 0.000 | 0.31 | 43.3 | 1.80 | 1.00 | 0.000 | 0.720 | 0.180 | 0 | 10 |

Total elapsed: `361.82 s`

## Scorer (Lambda-only, NO docking / AdmetAI / PB)

- `alpha_equivalence_uniqueness_score` — distinct beta-NF count.
- `click_rule_match_bonus` — +1.0 if any of 5 click rules fires.
- `metal_geometry_prior_bonus` — +1.0 if Pt=4 / Ru=Ir=6 coord.
- `rdkit_validity_score` — 1.0 if RDKit can sanitize.
- `synthesizability_via_lambda_paths` — 1.0 if beta-NF + RDKit.

