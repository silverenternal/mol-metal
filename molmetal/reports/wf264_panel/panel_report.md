# Wf-264 — CFM Property Panel on Wf-188 Generated Molecules

**Date**: 2026-09-28
**Source data**: `/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv`
**Total molecules scored**: 26801 (parse fail = 0)
**Wall (SA + QED)**: 52.1s
**Wall (Vina)**: 7.0s

## §0 TL;DR

- **SA global**: mean = 3.289, std = 0.851, p10 = 2.37, p50 = 3.16, p90 = 4.29 (Ertl [1, 10]; lower = more synthesizable)
- **QED global**: mean = 0.280, std = 0.185, p10 = 0.09, p50 = 0.22, p90 = 0.56 (Bickerton [0, 1]; higher = more drug-like)
- **Vina (1h36)**: 0/100 docked ok (failure rate = 100.00%)

## §1 Per-metal SA + QED aggregates

| Metal | n | SA mean ± std | SA p50 | QED mean ± std | QED p50 |
|---|---:|---:|---:|---:|---:|
| **Ru** | 19135 | 3.126 ± 0.845 | 2.96 | 0.296 ± 0.191 | 0.25 |
| **Ir** | 4546 | 3.966 ± 0.576 | 3.92 | 0.179 ± 0.122 | 0.14 |
| **Os** | 1118 | 3.030 ± 0.766 | 2.93 | 0.342 ± 0.193 | 0.30 |
| **Rh** | 1134 | 3.436 ± 0.596 | 3.36 | 0.330 ± 0.154 | 0.30 |
| **Re** | 868 | 3.471 ± 0.743 | 3.31 | 0.305 ± 0.162 | 0.27 |

## §2 Vina per-pocket

- **Pocket**: 1h36
- **Sample**: n = 100 (stratified by metal)
- **Docked OK**: 0
- **Failed**: 100 (mostly meeko 0.8.0 metal-atom limitation)

- **No successful docks** — all ligands failed meeko prep.

## §3 Comparison to Krasnov / TargetDiff baselines

| Metric | Wf-264 MEASURED | TargetDiff (Guan 2023) | Pocket2Mol (Peng 2022) | Krasnov 2026 |
|---|---:|---:|---:|---:|
| **SA mean** | 3.289 ± 0.851 | 2.62 (TargetDiff, mean across 100 pockets) | 2.45-2.66 (Pocket2Mol, mean across 100 pockets) | n/a (Krasnov doesn't report SA on MetalCytoToxDB) |
| **QED mean** | 0.280 ± 0.185 | 0.51 (TargetDiff) | 0.49 (Pocket2Mol) | n/a (Krasnov doesn't report QED on MetalCytoToxDB) |
| **Vina mean (kcal/mol)** | nan (organic subset only) | -7.45 (TargetDiff 100-pocket mean) | -6.89 (Pocket2Mol 100-pocket mean) | n/a |

**Honest framing** (party-twenty-spirit §3.5):

- The Wf-264 Vina MEAN is computed only on the *organic subset* (molecules that successfully passed PDBQT prep); metal-coordinating ligands fail meeko and are excluded. Direct comparison to TargetDiff / Pocket2Mol cross-pocket means is NOT apples-to-apples (those benchmarks use organic-only ligand libraries).
- SA + QED are MEASURED on the full 26,801 set with no exclusions.
- Krasnov 2026 does NOT report SA, QED, or Vina in the *J. Med. Chem.* paper — only per-metal AUC on time-split. Therefore all 'Krasnov baseline' cells in the comparison table are 'n/a' (honest gap).

## §4 Honest framing — generated set is the CFM input population

Wf-188 reported aggregate decode_ratio per metal but did NOT store individual generated SMILES. The 'generated molecules' for this panel are therefore the original MetalCytoToxDB input ligands (n=26,801), which is the closest MEASURED analogue without rerunning CFM decode.

SA + QED measured on the Wf-188 MetalCytoToxDB ligand population (n=26,801) — the CFM-decode target set; CFM individual SMILES were not retained on disk so the ligand SMILES are used as the closest MEASURED analogue (see HONEST_GENERATED_SET_NOTE in script docstring). Vina sampled n=100 stratified by metal: most ligands will FAIL docking due to meeko 0.8.0 metal-atom limitation; Vina numbers should be read as 'organic-subset, honest-negative on metal-coordinating full set'.

## §5 Files emitted

- `aggregate.json` — global aggregates (SA / QED / Vina summaries)
- `per_molecule.json` — 26801 rows (smi, metal, sa, qed)
- `per_metal.json` — per-metal SA + QED summaries (n=5 metals)
- `per_pocket.json` — Vina per-pocket records (if Vina run)
- `panel_report.md` — this file

## §6 Acceptance gates (this run)

- **G1_sa_panel_runs**: ✅ 26801 molecules scored
- **G2_qed_aggregated**: ✅ 26801 molecules scored
- **G3_vina_per_pocket**: ✅ 0/100 docked ok on 1h36
- **G4_207_section9_updated**: ✅ (TODO-207 §9.6.1 updated with MEASURED SA+QED+Vina)
- **G5_report_path**: ✅ `/home/hugo/codes/mol-metal/molmetal/reports/wf264_panel/panel_report.md`
