# Wf-28 Phase 2 — NCI-60 9-cell precious-metal data prep

_Generated_: 2026-09-25 21:07:26 UTC
_Source_: `existing CSV (MetalCytoToxDB-derived, see train_nci60_10cell.py)`
_Rows_: 7823 long-form  |  7823 cross-link

## 1. Cell-line coverage

| Cell line | n_rows | mean pIC50 | n_active (IC50 < 10 µM) | active_rate |
|----------:|-------:|-----------:|-------------------------:|------------:|
| MCF7 | 1964 | 4.787 | 728 | 0.371 |
| A549 | 2647 | 4.697 | 865 | 0.327 |
| A2780 | 1095 | 4.894 | 466 | 0.426 |
| A2780cis | 564 | 4.849 | 235 | 0.417 |
| HCT116 | 701 | 4.873 | 310 | 0.442 |
| PC3 | 275 | 4.945 | 114 | 0.415 |
| HL60 | 225 | 5.077 | 121 | 0.538 |
| SKOV3 | 191 | 4.693 | 65 | 0.340 |
| DU145 | 161 | 5.055 | 83 | 0.516 |

## 2. PlatinAI 4-cell cross-link

- NCI-60 rows that map to a PlatinAI Pt complex: **0 / 7823** (0.0%)
- Unique SMILES appearing in BOTH datasets: **0**

## 3. Notes

- Source: real MetalCytoToxDB substrate (Krasnov 2026 / J. Med. Chem. 2026-04 protocol).
- Precious metals only: Ru / Ir / Rh / Os / Re (Pt is in platinai_4cell).
- Cross-link overlap is 0 by construction: the precious-metal filter excludes Pt, and PlatinAI 4-cell is Pt-only. This is the expected outcome — the two corpora are complementary, not overlapping.

