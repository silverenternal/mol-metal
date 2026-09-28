# WF-30 Phase 1 — 10 CrossDocked pocket receptor prep log

_Generated_: 2026-09-26T04:41:43+00:00

Per-pocket `mk_prepare_receptor.py` runs that materialise one static-receptor `receptor.pdbqt` per pocket under `molmetal/reports/wf30_pocket_prep/<pocket>/`. The script is staging-only: it does **not** dock and does **not** mutate the CrossDocked source PDBs.

## Status summary

- **READY**: 10

Counts above are computed against the WF-30 list of 10 pockets. READY means a non-empty `receptor.pdbqt` is on disk; SKIPPED means `receptor.pdbqt` already existed and `--skip-existing` was passed; FAILED is documented below.

## Per-pocket table

| pocket | status | pdbqt bytes | source bytes | source sha256 (first 12) | elapsed (s) |
|---|---|---:|---:|---|---:|
| `830c` | READY | 344560 | 407268 | `276bc4b8d73d` | 4.1 |
| `1h36` | READY | 55680 | 46095 | `317dda6f1812` | 1.9 |
| `3eml` | READY | 32880 | 27767 | `57ff2f5b34e7` | 1.9 |
| `3ny8` | READY | 44240 | 36931 | `41050469300f` | 1.9 |
| `4aaw` | READY | 38880 | 31085 | `647edad4f537` | 1.7 |
| `4rlu` | READY | 511280 | 461755 | `1ad1d2fa6275` | 3.8 |
| `1g9v` | READY | 33520 | 27214 | `0a758dbaeeab` | 1.8 |
| `2hnx` | READY | 42320 | 33534 | `969820ad3b0c` | 1.9 |
| `4rn0` | READY | 22880 | 19551 | `9c0c3872fc81` | 2.1 |
| `1fmc` | READY | 38160 | 30532 | `a84f24cec837` | 2.2 |

## Metric-measured pocket provenance

| pocket | metric file | metric status | mean kcal/mol | n samples |
|---|---|---|---:|---:|
| `830c` | `metrics/by_metric/vina_score.json` | MEASURED | -1.3546 | 100 |
| `1h36` | `metrics/by_metric/vina_score_1h36.json` | MEASURED | -4.4465 | 20 |
| `3eml` | `metrics/by_metric/vina_score_3eml.json` | MEASURED | -6.1357 | 20 |
| `3ny8` | `metrics/by_metric/vina_score_3ny8.json` | MEASURED | -6.8812 | 20 |
| `4aaw` | `metrics/by_metric/vina_score_4aaw.json` | MEASURED | -4.7048 | 20 |
| `4rlu` | `metrics/by_metric/vina_score_4rlu.json` | MEASURED | -6.093 | 20 |

## Per-pocket notes

### `830c` — READY

- **notes**: MMP-13 hydroxamate (sulphone hydroxamate RS1). Single-pocket anchor for WF-30. Already docked successfully in R10 phase-1.
- **source_pdb**: `/home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/830c/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/molmetal/data/mmp13_real/830c.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/830c/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `1h36` — READY

- **notes**: SQHC_ALIAD (squalene-hopene cyclase) — CrossDocked test_022.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/SQHC_ALIAD_1_631_0/1h36_A_rec_1h35_r01_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1h36/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/SQHC_ALIAD_1_631_0/1h36_A_rec_1h35_r01_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1h36/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `3eml` — READY

- **notes**: AA2AR_HUMAN (A2A adenosine receptor TM) — GPCR pocket.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/AA2AR_HUMAN_1_315_TM_0/3eml_A_rec_3eml_zma_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/3eml/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/AA2AR_HUMAN_1_315_TM_0/3eml_A_rec_3eml_zma_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/3eml/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `3ny8` — READY

- **notes**: ADRB2_HUMAN (β2 adrenergic receptor TM) — GPCR pocket.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/ADRB2_HUMAN_26_357_TM_0/3ny8_A_rec_3d4s_tim_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/3ny8/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/ADRB2_HUMAN_26_357_TM_0/3ny8_A_rec_3d4s_tim_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/3ny8/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `4aaw` — READY

- **notes**: GLMU_STRPN test_001 (glutamate mutarotase, R84 ligand). CrossDocked standard test entry.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4aaw/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/GLMU_STRPN_2_459_0/4aaw_A_rec_4ac3_r83_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4aaw/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `4rlu` — READY

- **notes**: Trypsin HCC inhibitor (HCC pocket). Sourced from pdbbindpp-2020 (full receptor protein, not the pocket cut). 12 histidines in the tetramer — all forced to HIE because Meeko cannot break the HIE/HID/HIP tie on its own.
- **source_pdb**: `/home/hugo/codes/pocket/data/pdbbindpp-2020/extracted/pbpp-2020/4rlu/4rlu_protein.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4rlu/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/pocket/data/pdbbindpp-2020/extracted/pbpp-2020/4rlu/4rlu_protein.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4rlu/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res --set_template A:11=HIE,A:17=HIE,A:74=HIE,A:116=HIE,B:41=HIE,B:58=HIE,C:11=HIE,C:17=HIE,C:74=HIE,C:116=HIE,D:41=HIE,D:58=HIE

### `1g9v` — READY

- **notes**: HBA_HUMAN (haemoglobin subunit α). CrossDocked pocket10 cut. Selected because 1g9v appears in the task brief candidate short-list and is staged.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HBA_HUMAN_2_142_0/1g9v_A_rec_4ni0_2p3_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1g9v/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HBA_HUMAN_2_142_0/1g9v_A_rec_4ni0_2p3_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1g9v/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `2hnx` — READY

- **notes**: FABP4_HUMAN (fatty-acid binding protein 4). CrossDocked pocket10 cut. Selected because 2hnx appears in the task brief candidate short-list and is staged.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/FABP4_HUMAN_2_132_0/2hnx_A_rec_1tou_b1v_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/2hnx/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/FABP4_HUMAN_2_132_0/2hnx_A_rec_1tou_b1v_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/2hnx/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `4rn0` — READY

- **notes**: HDAC8_HUMAN (histone deacetylase 8). CrossDocked test_005. Selected as a metalloprotein-relevant pocket — HDAC8 has a Zn2+ active site.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HDAC8_HUMAN_1_377_0/4rn0_B_rec_4rn1_l8g_lig_tt_min_0_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4rn0/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HDAC8_HUMAN_1_377_0/4rn0_B_rec_4rn1_l8g_lig_tt_min_0_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/4rn0/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

### `1fmc` — READY

- **notes**: HDHA_ECOLI (7-α-hydroxysteroid dehydrogenase). CrossDocked test_006. Already partly prepared as molmetal/reports/crossdocked_first10_resolved/test_006/receptor_source_names.pdbqt; this run materialises a fresh strict-mode pdbqt under the WF-30 dir.
- **source_pdb**: `/home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HDHA_ECOLI_1_255_0/1fmc_B_rec_1fmc_cho_lig_tt_docked_1_pocket10.pdb`
- **pdbqt_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1fmc/receptor.pdbqt`
- **returncode**: `0`
- **command**: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python3 /home/hugo/codes/CadButEaas/.venv-vlm/bin/mk_prepare_receptor.py --read_pdb /home/hugo/codes/mol-metal/.data/molmetal/crossdocked/HDHA_ECOLI_1_255_0/1fmc_B_rec_1fmc_cho_lig_tt_docked_1_pocket10.pdb -o /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep/1fmc/receptor -p -j --compute_charges --charge_model gasteiger --default_altloc A --allow_bad_res

## Reproducibility

```bash
source env.sh
uv run python -m molmetal.scripts.check_10_pockets --skip-existing
```
