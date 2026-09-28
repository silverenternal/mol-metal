# Wf-181 — m14 real CFM decode benchmark (MetalCytoToxDB)

**Date**: 2026-09-27
**Checkpoint**: `/home/hugo/codes/mol-metal/molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`
**Data**: `/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv` (n_total=26801)
**Device**: cuda

## Aggregate

- Synthetic baseline decode_ratio: **1.000**
- Real (MetalCytoToxDB) decode_ratio: **1.0000** (102/102)
- Gap vs synthetic: **+0.0000**
- Verdict: **PASS: real 1.0000 >= synthetic 1.000**
- Wall time: 9.7s

## Per-seed

| seed | n | n_ok | decode_ratio | wall_s |
|---:|---:|---:|---:|---:|
| 0 | 34 | 34 | 1.0000 | 6.0 |
| 1 | 34 | 34 | 1.0000 | 0.1 |
| 2 | 34 | 34 | 1.0000 | 0.1 |

## Honest framing

PASS: real 1.0000 >= synthetic 1.000

- Synthetic is the **upper bound** (idealised Pt-click clouds).
- Real MetalCytoToxDB ligands have noise, multi-metal diversity (Ru/Pt/Ir/Au/Os/Re/Rh), and varied coordination geometries.
- Per party-twenty-spirit §3.5, real < synthetic is the **expected** honest framing; we MEASURE both and report.
- HALT if decode_ratio == 0 OR <100 samples processed.
- Cross-reference: Wf-139 P5 (PlatinAI 5K Pt subset, real=0.9745, 274/274); Wf-148 (NCI-60 9-cell + DrugOOD native, real=0.9980, 5979/5991).
