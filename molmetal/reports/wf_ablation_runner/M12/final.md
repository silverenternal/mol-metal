# Phase 3 Smoke Retrain — Stacked P0+P1 Fixes on 500-mol Metallodrug Pool

**Date**: 2026-09-16
**Script**: `molmetal/scripts/metallo_drug_smoke_retrain.py`
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (247 SMILES, 235 3D-embedded, 214 skipped)

## Configuration

| Knob | Value |
|---|---|
| steps (planned) | 200 |
| steps (executed) | 200 |
| batch_size | 8 |
| lr | 0.0001 |
| hidden_dim | 128 |
| n_layers | 3 |
| joint_train | True |
| bond_head | learned |
| atom vocab | 12 (metallodrug = 14) |
| bond_head in_dim | 265 (expected 265) |
| device | cuda |
| cuda_available | True |
| torch | 2.14.0+cu130 |
| wall-clock | 31.7s (0.16s/step) |

## P0 fixes active
all ON

| Fix | Status |
|---|---|
| F1 BondAwareDecoder.decode wired | True |
| F2 BondOrderHead in_dim correct (9 + 2*hidden_dim) | True |
| F3 vocab_mask in training CE loss | True |
| F4 hidden_dim >= 64 | True |
| F5 no bonds=zeros placeholder | True |

## P1 fixes active
| Fix | Status |
|---|---|
| P1.1 hidden_dim=128 | True |
| P1.2 learnable vel_scale | True |
| P1.3 PCGrad multi-task | False (off — bit-compat) |
| P1.4 ConnectivityAwareDecoder | True |

## TODO-21 Lambda-as-reward (Strategy 1)
| Knob | Value |
|---|---|
| enabled | False |
| samples requested | 64 |
| dataset size (kept) | 0 |
| after zero-weight filter | 0 |
| zero-weight dropped | 0 |
| mix ratio | 0.25 |
| reweight alpha | 1.0 |
| pocket_id | metallo_drug_smoke |
| loss is weighted | False |

## Loss curves (summary)
| Loss | first | last | min | delta |
|---|---|---|---|---|
| cfm   | 18.65391731262207 | 30.16060447692871 | 7.306643962860107 | 11.50668716430664 |
| atom  | 2.4849066734313965 | 7.42507791519165 | 1.0625675916671753 | 4.940171241760254 |
| bond  | 23.100807189941406 | 1.2570602893829346 | 1.2168750762939453 | -21.84374690055847 |
| total | 42.00321578979492 | 32.16017150878906 | 8.802492141723633 | -9.84304428100586 |

## Decode smoke (n_samples=16 per checkpoint)
| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|---|---|---|---|---|---|
| 200 | 1.000 | 16 | 8.0 | 0 | 1.9 |

**decode_ratio at step 200**: 1.0
**decode_ratio at step 500**: None
**decode_ratio at step 1000**: None
**max decode_ratio observed**: 1.000
**first non-zero step**: 200

## Verdict
**STILL 0 — coordinate-quality wall persists (bond head trains but decoder fails)**

References:
* Prior baseline (5000-step, h=32, single-fix): `wf_gpu_recovery_now/final.md` decode_ratio=0/192
* P0 fix inventory: `molmetal/reports/wf_cfm_p0_fixes/`
* P1 fix inventory: `molmetal/reports/wf_cfm_p1_fixes/`
* TODO/pending/24_cfm_architecture_redo_plan.md

JSON: `final.json`
Checkpoint: `molmetal/reports/wf_ablation_runner/M12/ckpt.pt`
