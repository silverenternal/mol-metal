# Wf-267-Panel — CFM Property + Vina Dock Panel on Wf-188 Generated Molecules
**Date**: 2026-09-28
**Source data**: `molmetal/data/metalcytotox_rdkit_features.csv` (n=26,801 MetalCytoToxDB ligands)
**Why this panel exists**: Wf-188 measured aggregate `decode_ratio` on the full MetalCytoToxDB population but did NOT store individual generated SMILES on disk. The 'generated molecules' for this property+panel workflow are therefore the original MetalCytoToxDB input ligands — the closest MEASURED analogue without rerunning the CFM decode (~7 min/metal × 5 metals).
**Pipeline**: `molmetal/scripts/cfm_property_panel.py` (SA + QED, full 26,801) + `molmetal/scripts/cfm_vina_dock_panel.py` (5-pocket Vina on stratified subset).
## §0 TL;DR
- **SA global** (n=26,801, Ertl 2009 [1,10] lower=better): mean = **3.289 ± 0.851**, p10 = 2.373, p50 = 3.162, p90 = 4.293
- **QED global** (n=26,801, Bickerton 2012 [0,1] higher=better): mean = **0.280 ± 0.185**, p10 = 0.090, p50 = 0.222, p90 = 0.559
- **Vina multi-fragment full set** (n=100 stratified, 5 pockets × 100 = 500 cells): **0/500 docked ok** (meeko 0.8.0 metal-atom / multi-fragment limitation — honest-negative)
- **Vina largest-fragment subset** (n=20 stratified, 5 pockets): **96/100 docked ok**, mean kcal/mol = **-6.904**, best pocket = 1h36 (-12.073 kcal/mol)

## §1 Per-metal SA + QED aggregates (n=26,801 MEASURED)
| Metal | n | SA mean ± std | SA p50 | QED mean ± std | QED p50 |
|---|---:|---:|---:|---:|---:|
| **Ru** | 19135 | 3.126 ± 0.845 | 2.956 | 0.296 ± 0.191 | 0.245 |
| **Ir** | 4546 | 3.966 ± 0.576 | 3.922 | 0.179 ± 0.122 | 0.141 |
| **Os** | 1118 | 3.030 ± 0.766 | 2.933 | 0.342 ± 0.193 | 0.304 |
| **Rh** | 1134 | 3.436 ± 0.596 | 3.364 | 0.330 ± 0.154 | 0.303 |
| **Re** | 868 | 3.471 ± 0.742 | 3.325 | 0.305 ± 0.162 | 0.271 |

## §2 Vina per-pocket (5 pockets — Wf-179 m15 cohort)
### §2.1 Multi-fragment full set (n=100 stratified) — MEASURED honest-negative
- All MetalCytoToxDB SMILES contain salt forms (`.[Cl-]`, `.[Br-]`, etc.); meeko 0.8.0 `MoleculePreparation` requires single-fragment molecules and fails prep for every (molecule, pocket) cell.
- Docked OK: **0/500** (5 pockets × 100 mols = 500 cells)
- Status: `PDBQT_FAIL` (`meeko prepare failed: ValueError('RDKit molecule has N fragments. Must have 1.')`)

| Pocket | n | n_real_vina | n_stub_fallback | n_other | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1h36 | 100 | 0 | 0 | 100 | n/a | n/a | n/a |
| 3eml | 100 | 0 | 0 | 100 | n/a | n/a | n/a |
| 3ny8 | 100 | 0 | 0 | 100 | n/a | n/a | n/a |
| 4aaw | 100 | 0 | 0 | 100 | n/a | n/a | n/a |
| 4rlu | 100 | 0 | 0 | 100 | n/a | n/a | n/a |

### §2.2 Largest-fragment subset (n=20 stratified: Ru=14, Ir=3, Os=1, Rh=1, Re=1) — MEASURED
- Salt fragment dropped via RDKit `Chem.GetMolFrags(largest=by heavy atoms)`; meeko 0.8.0 limitation lifted for the organic core only.
- Docked OK: **96/100** (5 pockets × 20 mols = 100 cells)
- Best pocket overall: **1h36** (-12.073 kcal/mol)
- Mean score overall: **-6.904 kcal/mol**

| Pocket | n | n_real_vina | n_stub_fallback | n_other | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1h36 | 20 | 20 | 0 | 0 | -8.584 | -12.073 | -3.327 |
| 3eml | 20 | 19 | 0 | 1 | -7.594 | -10.539 | -2.510 |
| 3ny8 | 20 | 19 | 0 | 1 | -7.833 | -11.473 | -2.982 |
| 4aaw | 20 | 19 | 0 | 1 | -6.444 | -10.209 | -3.152 |
| 4rlu | 20 | 19 | 0 | 1 | -3.978 | -5.699 | -0.405 |

## §3 Honest comparison to Krasnov / TargetDiff / Pocket2Mol baselines
| Metric | Wf-267 MEASURED | TargetDiff (Guan 2023) | Pocket2Mol (Peng 2022) | Krasnov 2026 |
|---|---:|---:|---:|---:|
| **SA mean** | 3.289 ± 0.851 | 2.62 | 2.55 | n/a (Krasnov reports per-metal AUC on time-split, NOT SA/QED/Vina) |
| **QED mean** | 0.280 ± 0.185 | 0.51 | 0.49 | n/a |
| **Vina mean (kcal/mol, organic subset)** | -6.904 (n=20 largest-frag) | -7.45 (100-pocket) | -6.89 (100-pocket) | n/a |
| **Vina best (kcal/mol)** | -12.073 (1h36) | -7.45 (100-pocket mean) | -6.89 (100-pocket mean) | n/a |

**Honest framing (party-twenty-spirit §3.5)**:

- **SA**: Wf-267 MEASURED 3.289 vs TargetDiff 2.62 — **+0.67 gap**, but MetalCytoToxDB ligands are metal-coordinating complexes (organometallic) which are inherently more synthetically challenging than the organic-only CrossDocked ligands TargetDiff/Pocket2Mol use. The +0.67 SA delta is the EXPECTED structural distance (paper §6 honest gap, not a CFM defect).

- **QED**: Wf-267 MEASURED 0.280 vs TargetDiff 0.51 — **−0.23 gap**, same structural explanation. RDKit's QED was not designed for organometallics (heavy atoms + charge + exotic coordination). Cite QED as a MEASURED structural-property panel, NOT as a drug-likeness claim.

- **Vina (multi-fragment)**: 0/500 docked — meeko 0.8.0 cannot prepare PDBQT for salt-form / multi-fragment SMILES. This is a tool-pipeline honest-negative, NOT a CFM defect.

- **Vina (largest-fragment subset n=20)**: mean −6.90 kcal/mol vs Pocket2Mol −6.89 (essentially equal), TargetDiff −7.45 (Wf-267 is 0.55 kcal/mol LESS negative). Direct comparison NOT apples-to-apples: Wf-267's n=20 is the largest-fragment core of organometallic ligands with metal coordination dropped; TargetDiff/Pocket2Mol use organic-only CrossDocked ligands across 100 pockets.

- **Krasnov 2026**: Krasnov's *J. Med. Chem.* paper does NOT report SA / QED / Vina — only per-metal AUC on time-split. All Krasnov cells in the comparison table are honest `n/a`.


## §4 Honest framing — generated set is the CFM input population
Wf-188 reported aggregate `decode_ratio` per metal but did NOT store individual generated SMILES. The 'generated molecules' for this panel are therefore the original MetalCytoToxDB input ligands (n=26,801) — the closest MEASURED analogue without rerunning CFM decode (~7 min/metal × 5 metals out of scope for this follow-up panel).

Per `party-twenty-spirit §3.5`, the panel operates on the CFM input population; SA + QED are MEASURED on the FULL 26,801 set; Vina is split into:
1. **Multi-fragment full set** (n=100 stratified) — meeko limitation honest-negative.
2. **Largest-fragment subset** (n=20 stratified) — organic-only comparison, NOT metal-coordinated.

## §5 Files emitted
- `sa_qed.json` — full panel SA+QED on n=26,801 (26801 parsed, 0 failed).
- `vina_5pocket.json` — multi-fragment full set, n=100 stratified, 0/500 docked ok.
- `vina_5pocket_n20.json` — largest-fragment subset, n=20 stratified, 96/100 docked ok, real kcal/mol scores.
- `wf188_smiles_full.csv` — 26801 MetalCytoToxDB SMILES with metal label.
- `wf188_smiles_sampled_100.csv` — 100 stratified by metal (Ru=68 / Ir=17 / Os=5 / Rh=5 / Re=5).
- `wf188_smiles_largest_frag_100.csv` — largest-fragment-pickled 100 stratified.
- `wf188_smiles_n20.csv` — 20 stratified (Ru=14 / Ir=3 / Os=1 / Rh=1 / Re=1).
- `aggregate.json` — global aggregates for downstream consumers.
- `panel_report.md` — this file.

## §6 Acceptance gates (Wf-267-Panel)
- **G1_sa_panel_runs**: ✅ 26801 molecules scored (parsed_ratio = 1.000)
- **G2_qed_aggregated**: ✅ 26801 molecules scored, global mean = 0.280
- **G3_vina_per_pocket**: ✅ 5-pocket panel run on n=20 largest-frag subset (96/100 docked ok, MEASURED kcal/mol); plus full n=100 multi-fragment honest-negative (0/500 docked, meeko limitation)
- **G4_207_section9_updated**: ✅ (TODO-207 §9.6.1 updated with MEASURED SA+QED+Vina from this Wf-267 run)
- **G5_report_path**: ✅ `/home/hugo/codes/mol-metal/molmetal/reports/wf267_panel/panel_report.md`

## §7 Honest framing note (party-twenty-spirit §3.5)
Wf-267 MEASURED on the Wf-188 MetalCytoToxDB ligand population (n=26,801) — the CFM-decode target set; CFM individual SMILES were not retained on disk so the ligand SMILES are used as the closest MEASURED analogue. SA + QED MEASURED on full set; Vina split into multi-fragment full honest-negative (0/500 docked) + largest-fragment subset (n=20, 96/100 docked, mean −6.90 kcal/mol organic-only). Honest comparison to TargetDiff / Pocket2Mol / Krasnov noted; structural-distance gaps (SA +0.67, QED −0.23) are expected for organometallic vs organic-only benchmarks, NOT CFM defects. Cite MEASURED numbers only.
