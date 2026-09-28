# Wf-148 — Real CFM decode benchmark synthesis

**Date**: 2026-09-27
**Checkpoint**: `/home/hugo/codes/mol-metal/molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`
**Data config**: `nci60+drugood`
**Data**: `nci60:/home/hugo/codes/mol-metal/molmetal/data/nci60_precious_9cell.csv + drugood:/home/hugo/codes/mol-metal/.data/molmetal/drugood/ic50/data/train.csv (7823+1000 rows -> 2000 subsampled)`
**Device**: cuda:0

## Aggregate

- Synthetic baseline decode_ratio: **1.000**
- Real (NCI-60 9-cell + DrugOOD native) decode_ratio: **0.9980** (5979/5991)
- Gap vs synthetic: **-0.0020**
- Verdict: **PARTIAL: real 0.9980 < synthetic 1.000 (expected — synthetic is upper bound; real < synthetic is the honest framing per party-twenty-spirit §3.5)**
- Wall time: 281.1s

## Per-seed

| seed | n | n_ok | decode_ratio | wall_s |
|---:|---:|---:|---:|---:|
| 0 | 1997 | 1993 | 0.9980 | 6.7 |
| 1 | 1997 | 1993 | 0.9980 | 6.7 |
| 2 | 1997 | 1993 | 0.9980 | 6.4 |

## Honest framing

PARTIAL: real 0.9980 < synthetic 1.000 (expected — synthetic is upper bound; real < synthetic is the honest framing per party-twenty-spirit §3.5)

- Synthetic is the **upper bound** (idealised Pt-click clouds).
- Real ligands have noise, non-Pt atoms, and varied coordination.
- Per party-twenty-spirit §3.5, real < synthetic is the **expected** honest framing; we MEASURE both and report.
- HALT if decode_ratio == 0 OR <100 samples processed.
