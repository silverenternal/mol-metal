# Wf-188 — m14 real CFM FULL MetalCytoToxDB benchmark (per-metal)

**Date**: 2026-09-27
**Checkpoint**: `/home/hugo/codes/mol-metal/molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`
**Data**: `/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv` (n_total=26801)
**Device**: cuda
**Embed workers**: 20

## Aggregate

- Synthetic baseline decode_ratio: **1.000**
- Real (MetalCytoToxDB FULL) decode_ratio: **0.9958** (26687/26799)
- Gap vs synthetic: **-0.0042**
- Verdict: **PARTIAL: real 0.9958 < synthetic 1.000 (expected — synthetic is upper bound; real < synthetic is the honest framing per party-twenty-spirit §3.5)**
- Wall time: 428.5s (target: <600s, hard cap: 1200s)

## Per-metal breakdown

| metal | n_total | n_embedded | n_decode_ok | decode_ratio | wall_s |
|---|---:|---:|---:|---:|---:|
| Ru | 19135 | 19133 | 19033 | 0.9948 | 328.9 |
| Ir | 4546 | 4546 | 4534 | 0.9974 | 54.8 |
| Os | 1118 | 1118 | 1118 | 1.0000 | 20.0 |
| Rh | 1134 | 1134 | 1134 | 1.0000 | 14.1 |
| Re | 868 | 868 | 868 | 1.0000 | 10.6 |

## Honest framing

PARTIAL: real 0.9958 < synthetic 1.000 (expected — synthetic is upper bound; real < synthetic is the honest framing per party-twenty-spirit §3.5)

- Synthetic is the **upper bound** (idealised Pt-click clouds).
- Real MetalCytoToxDB ligands have noise, multi-metal diversity (Ru/Pt/Ir/Au/Os/Re/Rh), and varied coordination geometries.
- Per party-twenty-spirit §3.5, real < synthetic is the **expected** honest framing; we MEASURE both and report.
- HALT if aggregate decode_ratio == 0 OR <10000 samples processed.
- Cross-reference: Wf-181 (102 samples, real=1.0 by lucky embedding; Wf-188 is the FULL 26,801 population).
