# Wf-A198 -- the run ledger's VERSION PIN is a PAIR; only the sha half is validated

**MEASURED 2026-11-02, CPU-only, `.venv/bin/python`, 0 GPU.**  Every number below is read out of `report.json`, which is produced by `molmetal/scripts/wfa198_ledger_pin_consistency.py` (the auditor) on this working tree.  No numeral in this document is typed by hand.

## 0. WHAT THIS IS -- AND WHAT IT IS NOT

**READ-ONLY MEASUREMENT.**  No ledger row was written, rewritten or deleted.  `molmetal/reports/run_ledger.jsonl` is **APPEND-ONLY** and was READ, never touched: its sha256 is recorded in `report.json` and is unchanged before and after the audit.  The rows this report calls inconsistent are **REPORTED, not repaired** -- repairing them would mean rewriting an append-only log, which the repo's own contract forbids.

The auditor is **NOT** registered as a pre-flight gate and `pre_flight_check.EXPECTED_GATE_COUNTS` was **NOT** changed.  Registering a gate is a USER decision (honest-negative catalogue 6.266.9 / 6.267).

| | |
|---|---|
| ledger | `/home/hugo/codes/mol-metal/molmetal/reports/run_ledger.jsonl` |
| ledger sha256 | `3a5a8cd0690a0b854c5bfde6e6f73dd62f07e340be7699106b7e8098be135479` |
| ledger rows | 432 |
| rows carrying a complete pin (`git_sha` + `source_json_sha256`) | 432 |
| auditor wall time | 0.96 s |

## 1. THE DEFECT

A ledger row pins its metric value to a VERSION with a **PAIR** of columns: `git_sha` (the commit) and `source_json_sha256` (the content hash at that version).  The repository's own gate -- `p3_run_ledger_provenance` (`molmetal/scripts/pre_flight_check.py:9986`), delegating to `molmetal/scripts/run_ledger.py::uncommitted_provenance` (defined at `:469`) -- groups the rows by `(source_json, source_json_sha256)` and classifies each SHA as `on_disk` / `committed` / `uncommitted`.  **It never reads the `git_sha` column.**  Only the sha half of the pin is validated.

### 1.1  The pair is inconsistent

- **32 of 432 rows** carry an inconsistent pair.
- **27 rows**: `source_json` EXISTS at the row's `git_sha`, but its content there is NOT the row's `source_json_sha256`.
- **5 rows**: `source_json` does not exist AT ALL at the row's `git_sha`.
- **400 rows** are consistent: checking out the pinned commit DOES reproduce the version the row records.

=> **checking out the `git_sha` a row names does NOT reproduce the version that row records.**  With respect to verification, the `git_sha` column is decorative.

### 1.2  Why the repo's gate cannot see them

The gate's own classification of the 27 sha-mismatched rows (replicated here from the same primitives, in the gate's own order -- disk first, then committed history, cf. `run_ledger.uncommitted_provenance._classify` at `molmetal/scripts/run_ledger.py:523`):

| gate classification | sha-mismatch rows | all inconsistent rows |
|---|---|---|
| `on_disk` | 17 | 19 |
| `committed` | 2 | 2 |
| `uncommitted` | 8 | 11 |
| **total** | **27** | **32** |

**19 of the 27** sha-mismatched rows are classified by the gate as REPRODUCIBLE (their SHA matches disk, or matches some OTHER committed version of that path), so the gate cannot see them.  Across all 32 inconsistent rows, 21 are classified as reproducible.

> **ORDER MATTERS, and the gate's order is disk-first.**  A SHA that is BOTH on disk and present in some commit is reported by the gate as `on_disk`, never as `committed`.  The same 27 rows counted with the OPPOSITE precedence (`committed` checked first) split as 10 committed / 9 on_disk / 8 uncommitted -- the SAME 19 are reproducible either way, but the split moves.  This report uses the gate's own order, and the test file asserts the auditor's verdicts match the real gate's verdicts row for row.

### 1.3  The registry is blind to the ledger entirely

`registry.py check` validates the REGISTRY's own rows -- it compares the registry's `source_json_sha256` with the file on disk (`molmetal/scripts/registry.py:1666`).  It never reads the ledger: the string `ledger` does not appear in `molmetal/scripts/registry.py`.  So a ledger row can be inconsistent while the registry reports `current`.

## 2. EVERY OFFENDING ROW

32 rows over 31 distinct paths.  `sha AT THAT COMMIT` is the sha256 of the file's content at the row's own `git_sha`; `<PATH ABSENT AT THAT COMMIT>` means the path does not exist there at all.

### row 9 -- `l6_closure`

- `source_json`: `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json`
- `git_sha`: `8b9ceb52830f3203cf028ed12943fdcaff0e0ebb`
- recorded `source_json_sha256`: `49e13c420b74c5a64020cffc3207cb931877cefa010bfc3e952e109892209688`
- sha256 AT THAT COMMIT: `<PATH ABSENT AT THAT COMMIT>`
- `exists_at_commit`: False | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 0 | the SHA is reproducible elsewhere: False

### row 11 -- `m14_decode_ratio`

- `source_json`: `metrics/by_metric/decode_ratio.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `51d0ae2fd7b13a03122bfa96e63f1e1b36ad9935d67222ff3c236975a8538a98`
- sha256 AT THAT COMMIT: `bbb7c1fb9a3e559d5202333a9236fe2d3749d065a8c38e26d24c35e6ce0c2dde`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 6 | the SHA is reproducible elsewhere: True

### row 12 -- `cfm_bond_accuracy`

- `source_json`: `metrics/by_metric/cfm_bond_accuracy.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `15c839f7c5b84f6ec74d9475966680edbc67fcbe9132f808110c14d6407cf274`
- sha256 AT THAT COMMIT: `00eb109cc3b757d9c30aadddd0ea3231a5be76916611cd8167ce7a2c1403b179`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 2 | the SHA is reproducible elsewhere: True

### row 13 -- `m3_vina_dock`

- `source_json`: `metrics/by_metric/vina_kcal_per_mol.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `24a2653e38aaa484339291575278eb8c43d4fa662d19215d95f2f72e64c8da67`
- sha256 AT THAT COMMIT: `79ab0561f14b049d6897f761792149446f2e45b1b66e6656d29bece8a694d2ec`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 7 | the SHA is reproducible elsewhere: True

### row 14 -- `m7_diversity`

- `source_json`: `metrics/by_metric/m7_diversity.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `db84569d3f39bb40f1ce69b465b636499e77d45c16451d1326da2dca749f74ff`
- sha256 AT THAT COMMIT: `9ccae5e0867e812e69c5b0594ba809dc37c4d209600dc8b97d09a415e9272382`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `committed` | committed versions of that path: 7 | the SHA is reproducible elsewhere: True

### row 15 -- `m1_ru_auc`

- `source_json`: `metrics/by_metric/metalcytotox_ru_auc.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `d19be01d893a3d48a731ca63027b5ee44f7df67017eb70482c91bac96f920a90`
- sha256 AT THAT COMMIT: `da6f1a3f15b092486d54d04bfd84f7322d2dbd79d5a3c2caf11cfa19ab4090c1`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 5 | the SHA is reproducible elsewhere: True

### row 16 -- `m19_lomo`

- `source_json`: `metrics/by_metric/metal_lomo_auc_wasserstein.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `90a9c0c095605fbe20e49a2dfb830a6a890b961b9204330d6cc234bdb6d4aa74`
- sha256 AT THAT COMMIT: `0a3d3cae45b526679613ee64768d76becb9a309ed4fc37917d09d0a29fb6ceb9`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 10 | the SHA is reproducible elsewhere: True

### row 17 -- `m2_r2_per_cell`

- `source_json`: `metrics/by_metric/m2_r2_per_cohort.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `46455a168da47eda2d47de1b2dd1a80020ac6e20c94c73ee326a5bcb6d3a172f`
- sha256 AT THAT COMMIT: `77bc8b615379a4467190a1a8f552b77c41e9abadfe04106a8782ddeec62789fd`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `committed` | committed versions of that path: 7 | the SHA is reproducible elsewhere: True

### row 19 -- `l5_hilbert_r2`

- `source_json`: `metrics/by_metric/hilbert_4cell_r2.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `422a04b4321d2dcaa9ef9aee54b01d5b5963801adca2c0808c4c086f250f949e`
- sha256 AT THAT COMMIT: `14bef681b645f83bf9dfe7cf4050420542c7c868fc172a601c5e5bbeda3b8cd4`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 8 | the SHA is reproducible elsewhere: True

### row 20 -- `l6_closure`

- `source_json`: `metrics/by_metric/l6_closure.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `82305a3bd789b36fc0c304d4563dc48dd030814b8fe77a5bb909c442053bc30b`
- sha256 AT THAT COMMIT: `dfe9596b9a6a0a7ed82afdaf919676365eddb18bbb433554208a47a8bb831eb2`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 2 | the SHA is reproducible elsewhere: True

### row 21 -- `m6_kappa`

- `source_json`: `metrics/by_metric/m6_kappa.json`
- `git_sha`: `e0d473cc7baec3d454c9515694ca17fdadd631c3`
- recorded `source_json_sha256`: `079f802829f9aa10be2de4c9d5107912199e9393e32b5c931c08a8cc0d61cf08`
- sha256 AT THAT COMMIT: `c8294cee3c4700d78a903d1c5bc895a286dab6a29e3bf8e0d3d0529616690d95`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 5 | the SHA is reproducible elsewhere: True

### row 22 -- `l5_hilbert_r2`

- `source_json`: `metrics/by_metric/hilbert_4cell_r2.json`
- `git_sha`: `7e30d6ea509a8a42d953a77a166d69457ef7f037`
- recorded `source_json_sha256`: `d4079d7e0e1689b4577bebc0c451cb67d165a56ea5e792ae195d5c7509fc30d1`
- sha256 AT THAT COMMIT: `14bef681b645f83bf9dfe7cf4050420542c7c868fc172a601c5e5bbeda3b8cd4`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 8 | the SHA is reproducible elsewhere: False

### row 116 -- `dmpnn_ckpt_status`

- `source_json`: `metrics/by_metric/dmpnn_ckpt_status.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `a6db107ec57620bc3b04d367046740fc4a34fcc6476b2fba95ba98dbceaec932`
- sha256 AT THAT COMMIT: `b43f13849a3af614694c287008276c0da06223146abd8fc7ce281d59ccfccd09`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 3 | the SHA is reproducible elsewhere: False

### row 219 -- `metalcytotox_combined_r2`

- `source_json`: `metrics/by_metric/metalcytotox_combined_r2.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `425e68518470c3a5ceec3a83d3ee68136c222b35c2aa1ee6d7d2c9d21e8634b0`
- sha256 AT THAT COMMIT: `50248f3d363172e4832bc52c36d3b42fc208f87ed33776eeacfff344fde58984`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 2 | the SHA is reproducible elsewhere: False

### row 292 -- `ood_per_element_reachability`

- `source_json`: `metrics/by_metric/ood_per_element_reachability.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `8df5eceaabb397258133b3fe2b0f233b0a1c31db30fffbc75fcda7872fe3975e`
- sha256 AT THAT COMMIT: `41a70f724c935c58b8731341ccc2fd58e650dd9a848d9776682a38de581105d5`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 293 -- `ood_per_metal_reachability`

- `source_json`: `metrics/by_metric/ood_per_metal_reachability.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `ea59f36aec3de63c16f44c4bc5f912f235276277b939588abb4d9c9f58bd2703`
- sha256 AT THAT COMMIT: `31717c8a23ea6f3ed644f9dcde8865b96be40d293e65f491c8ec436c3808616e`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 2 | the SHA is reproducible elsewhere: False

### row 312 -- `platinai_combined_r2`

- `source_json`: `metrics/by_metric/platinai_combined_r2.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `21e1e71f8c64724ef0f90c697224443adb5dec2afe9fc55fc30230dd0bd73255`
- sha256 AT THAT COMMIT: `3b15a0a2459fc89c8bfed7a83a07e8a3d0121866672256c062362793f847c152`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 4 | the SHA is reproducible elsewhere: False

### row 339 -- `tmqm_combined_r2`

- `source_json`: `metrics/by_metric/tmqm_combined_r2.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `0c1411ef5957c373f9977372d09e2429a1823a1a8ce7307a415d9db54fa5b1e7`
- sha256 AT THAT COMMIT: `01a14d1c58649139bd2e762bfad189536c03628697b569e504975fb8c9d2d98a`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 2 | the SHA is reproducible elsewhere: False

### row 375 -- `vina_status`

- `source_json`: `metrics/by_metric/vina_status.json`
- `git_sha`: `d86c879a540e79ebefcda820e83ddff24f5477ad`
- recorded `source_json_sha256`: `09cee1ef037f17401c55d4080506607a27d93ea7dca68dc508296d7a441328d8`
- sha256 AT THAT COMMIT: `645ba15c9b1d1b7a3a59fbd41a274f9d9ac07a9f269d6cda738f041747f4b159`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 407 -- `chembl29_metal_scan`

- `source_json`: `metrics/by_metric/chembl29_metal_scan.json`
- `git_sha`: `975414bc6fb8ef20ca53429f287a4d3297184cad`
- recorded `source_json_sha256`: `bf5f88c2d3de845a2110b09478f93e6e0a065b50af56db15ba518ac5e6469bc7`
- sha256 AT THAT COMMIT: `<PATH ABSENT AT THAT COMMIT>`
- `exists_at_commit`: False | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 1 | the SHA is reproducible elsewhere: True

### row 418 -- `wfa155_ligand_connected_rate`

- `source_json`: `metrics/by_metric/wfa155_ligand_connected_rate.json`
- `git_sha`: `1b9470a0e2129e3c4d982da4453e034a6cc12e24`
- recorded `source_json_sha256`: `a60933e8281f808032ce5706a3e8ec174573e134918ac59952dc883d9a30d05e`
- sha256 AT THAT COMMIT: `<PATH ABSENT AT THAT COMMIT>`
- `exists_at_commit`: False | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 0 | the SHA is reproducible elsewhere: False

### row 419 -- `wfa155_metal_coordination_sphere`

- `source_json`: `metrics/by_metric/wfa155_metal_coordination_sphere.json`
- `git_sha`: `1b9470a0e2129e3c4d982da4453e034a6cc12e24`
- recorded `source_json_sha256`: `aa67ed49b645541032f99a14fa2f5414e52814c52f21383ae465f5c5083bd882`
- sha256 AT THAT COMMIT: `<PATH ABSENT AT THAT COMMIT>`
- `exists_at_commit`: False | `pin_consistent`: False
- gate classification of the SHA: `uncommitted` | committed versions of that path: 0 | the SHA is reproducible elsewhere: False

### row 420 -- `wfa155_posebusters_validity`

- `source_json`: `metrics/by_metric/wfa155_posebusters_validity.json`
- `git_sha`: `1b9470a0e2129e3c4d982da4453e034a6cc12e24`
- recorded `source_json_sha256`: `6464edc162523abd65dea101ddc2633f689cfaa8ba1b05aad73a3506382be832`
- sha256 AT THAT COMMIT: `<PATH ABSENT AT THAT COMMIT>`
- `exists_at_commit`: False | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 0 | the SHA is reproducible elsewhere: False

### row 421 -- `c7_fullgen_pool_quality`

- `source_json`: `metrics/by_metric/c7_fullgen_pool_quality.json`
- `git_sha`: `2b84f86676cd2325759a0b4eb29ca17a30928e7a`
- recorded `source_json_sha256`: `6a9f61f135dab90261f396a4ea77b305e4b93dabdad9e6ed3b8988e6a9b1d400`
- sha256 AT THAT COMMIT: `9f597a6b531a8ab9bae4d05615d5b7fe85a9e514a6f8e67ec479b52900718d89`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 4 | the SHA is reproducible elsewhere: False

### row 422 -- `c7_scale_matched_pools`

- `source_json`: `metrics/by_metric/c7_scale_matched_pools.json`
- `git_sha`: `e206cc7b9f5de235e760061ae130d966348e57c6`
- recorded `source_json_sha256`: `ea7583ad43b9c60fd0cd73872e7807008f9f36998e5e964cc88e543cd271b9e5`
- sha256 AT THAT COMMIT: `3c447b34791b7951c848258d3e6f87650d02d7aa5a30a9d6d99fcdf8097fd8ce`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 4 | the SHA is reproducible elsewhere: False

### row 423 -- `cfm_generation_metrics`

- `source_json`: `metrics/by_metric/cfm_generation_metrics.json`
- `git_sha`: `4458c5d57d3467ca5d3fb9d2cdb54e7e435db8d3`
- recorded `source_json_sha256`: `2487151810d6530e06ea16e6fa6cf8276983ca8a5aebd6861c2d9462c69d3eeb`
- sha256 AT THAT COMMIT: `435a50b1e21b282fe3cb235603522ce3adff891c80139657d76ffa781dd7bb29`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 424 -- `cfm_qed_fragment_aware`

- `source_json`: `metrics/by_metric/cfm_qed_fragment_aware.json`
- `git_sha`: `b2547f6f8f4f99d0f46160e57ed294aa0c39a0b4`
- recorded `source_json_sha256`: `5257b08641e83f3b45a99157cdc8eb348fd400c71fd462c6501878f115802712`
- sha256 AT THAT COMMIT: `672a02f0085af6843c0fd43ecf670bc67b4d0f4a052b53b7f53d72efae5787e3`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 2 | the SHA is reproducible elsewhere: False

### row 425 -- `cfm_run_census`

- `source_json`: `metrics/by_metric/cfm_run_census.json`
- `git_sha`: `ee7a987aba05460a0f096268a2326792944f15ec`
- recorded `source_json_sha256`: `40ef9c51ae93eeac35b16afd2301ed41d67f7226ac0cb984270ca90dafe49e5a`
- sha256 AT THAT COMMIT: `2268d0ef561be7ba681def8d29843955ccf7af1d9fae3d4de0cec94448bb306a`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 2 | the SHA is reproducible elsewhere: False

### row 426 -- `cfm_sa_guided`

- `source_json`: `metrics/by_metric/cfm_sa_guided.json`
- `git_sha`: `4f7dcb0a1c6750ae2fb5884bab92035b834d26d9`
- recorded `source_json_sha256`: `ba655c098d5dee7762a475caf3719eef9008601da2938a86ac8a1e3be7be4400`
- sha256 AT THAT COMMIT: `a61f62ded049b299e7b22e609d71770d3db15301dce0dcd94a36d8a69b216948`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 427 -- `generated_to_activity_cross_application`

- `source_json`: `metrics/by_metric/cfm_generated_to_nci60_cross_application.json`
- `git_sha`: `b2547f6f8f4f99d0f46160e57ed294aa0c39a0b4`
- recorded `source_json_sha256`: `04156e48256edb567a386b2b01e24d45c9cdedbd0bbbe580842ac423894b6250`
- sha256 AT THAT COMMIT: `a1fddf951bb4ad220bbbca5049b44cf05f34685dfb08d4cb5a447399e6d52d35`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 430 -- `msrs_data_card`

- `source_json`: `metrics/by_metric/msrs_data_card.json`
- `git_sha`: `c98c1da62231afd27841a4e2ec583ddb1e9af9db`
- recorded `source_json_sha256`: `8efc13c9bd45a4fd35256f08ec582d03ad319478fd759f192a247077e4005b31`
- sha256 AT THAT COMMIT: `e5ede80810c1c8c0823c8da3c6b7d72e63f25bd9d9d7d5d2f47a703b76169e19`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 1 | the SHA is reproducible elsewhere: False

### row 431 -- `wf32_pb_valid`

- `source_json`: `metrics/by_metric/wf32_pb_valid.json`
- `git_sha`: `c289aa42f12c04274055248cf304d5b3a4ff430d`
- recorded `source_json_sha256`: `d6eec79ee513e8427939b79096fd462b62b70d2aa648a04cf0e4338f951b6127`
- sha256 AT THAT COMMIT: `8d3fbd763de5f92d2a34a508fe2ca22d9eb7cc83434eb0a64eada08bd830b3a0`
- `exists_at_commit`: True | `pin_consistent`: False
- gate classification of the SHA: `on_disk` | committed versions of that path: 3 | the SHA is reproducible elsewhere: False

## 3. THE OFFENDING PATHS -- TRACKING AND GITIGNORE STATUS

| path | rows | tracked by git | gitignored | ignoring rule | committed versions |
|---|---|---|---|---|---|
| `metrics/by_metric/c7_fullgen_pool_quality.json` | 1 | True | False | - | 4 |
| `metrics/by_metric/c7_scale_matched_pools.json` | 1 | True | False | - | 4 |
| `metrics/by_metric/cfm_bond_accuracy.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/cfm_generated_to_nci60_cross_application.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/cfm_generation_metrics.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/cfm_qed_fragment_aware.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/cfm_run_census.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/cfm_sa_guided.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/chembl29_metal_scan.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/decode_ratio.json` | 1 | True | False | - | 6 |
| `metrics/by_metric/dmpnn_ckpt_status.json` | 1 | True | False | - | 3 |
| `metrics/by_metric/hilbert_4cell_r2.json` | 2 | True | False | - | 8 |
| `metrics/by_metric/l6_closure.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/m2_r2_per_cohort.json` | 1 | True | False | - | 7 |
| `metrics/by_metric/m6_kappa.json` | 1 | True | False | - | 5 |
| `metrics/by_metric/m7_diversity.json` | 1 | True | False | - | 7 |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | 1 | True | False | - | 10 |
| `metrics/by_metric/metalcytotox_combined_r2.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/metalcytotox_ru_auc.json` | 1 | True | False | - | 5 |
| `metrics/by_metric/msrs_data_card.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/ood_per_element_reachability.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/ood_per_metal_reachability.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/platinai_combined_r2.json` | 1 | True | False | - | 4 |
| `metrics/by_metric/tmqm_combined_r2.json` | 1 | True | False | - | 2 |
| `metrics/by_metric/vina_kcal_per_mol.json` | 1 | True | False | - | 7 |
| `metrics/by_metric/vina_status.json` | 1 | True | False | - | 1 |
| `metrics/by_metric/wf32_pb_valid.json` | 1 | True | False | - | 3 |
| `metrics/by_metric/wfa155_ligand_connected_rate.json` | 1 | False | False | - | 0 |
| `metrics/by_metric/wfa155_metal_coordination_sphere.json` | 1 | False | False | - | 0 |
| `metrics/by_metric/wfa155_posebusters_validity.json` | 1 | False | False | - | 0 |
| `molmetal/reports/wf67_p4_2_closure_r20/reachability_overall.json` | 1 | False | True | .gitignore:158:molmetal/reports/**/*.json | 0 |

## 4. THE CLI REPORTING DEFECT (a second, separate defect)

`uncommitted_provenance()` puts a metric into `detail` only when it has an OPEN finding, or when `include_superseded=True`.  The pre-flight gate calls the FUNCTION with `include_superseded=True` and formats the number itself, so **the gate is correct**.  The defect is in the CLI's DEFAULT mode only: it set `include_superseded = "--all" in args[1:]` (`molmetal/scripts/run_ledger.py:847`), so WITHOUT `--all` the superseded count is 0 **by construction** -- the rows were filtered out before the count was taken -- and the CLI printed that 0 as if it were a measurement.

**PRE-FIX, measured 2026-11-02 (preserved verbatim, AGENTS.md 5.5 rule #6):**

```
ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=0
    [open] wfa155_ligand_connected_rate  metrics/by_metric/wfa155_ligand_connected_rate.json
        on-disk sha256 9ad45a495751c2046f37c8377a6dd118e0f93efc6ef8da58d5e7e3dcc31764de
        committed versions of that path: 0
        ledger  sha256 a60933e8281f808032ce5706a3e8ec174573e134918ac59952dc883d9a30d05e
        ledger value: None
    [open] wfa155_metal_coordination_sphere  metrics/by_metric/wfa155_metal_coordination_sphere.json
        on-disk sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b
        committed versions of that path: 0
        ledger  sha256 aa67ed49b645541032f99a14fa2f5414e52814c52f21383ae465f5c5083bd882
        ledger value: None
LEDGER PROVENANCE: WARN
rc=1
```

```
ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=9
    [superseded] dmpnn_ckpt_status  metrics/by_metric/dmpnn_ckpt_status.json
        on-disk sha256 b43f13849a3af614694c287008276c0da06223146abd8fc7ce281d59ccfccd09
        committed versions of that path: 3
        NEVER-COMMITTED sha256 a6db107ec57620bc3b04d367046740fc4a34fcc6476b2fba95ba98dbceaec932
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] l5_hilbert_r2  metrics/by_metric/hilbert_4cell_r2.json
        on-disk sha256 422a04b4321d2dcaa9ef9aee54b01d5b5963801adca2c0808c4c086f250f949e
        committed versions of that path: 8
        NEVER-COMMITTED sha256 d4079d7e0e1689b4577bebc0c451cb67d165a56ea5e792ae195d5c7509fc30d1
            (row value -6.712634861296495, git_sha '7e30d6ea509a8a42d953a77a166d69457ef7f037')
        superseded uncommitted rows for this metric: 1
    [superseded] l6_closure  metrics/by_metric/l6_closure.json
        on-disk sha256 82305a3bd789b36fc0c304d4563dc48dd030814b8fe77a5bb909c442053bc30b
        committed versions of that path: 2
        NEVER-COMMITTED sha256 49e13c420b74c5a64020cffc3207cb931877cefa010bfc3e952e109892209688
            (row value 0.5962, git_sha '8b9ceb52830f3203cf028ed12943fdcaff0e0ebb')
        superseded uncommitted rows for this metric: 1
    [superseded] metalcytotox_combined_r2  metrics/by_metric/metalcytotox_combined_r2.json
        on-disk sha256 50248f3d363172e4832bc52c36d3b42fc208f87ed33776eeacfff344fde58984
        committed versions of that path: 2
        NEVER-COMMITTED sha256 425e68518470c3a5ceec3a83d3ee68136c222b35c2aa1ee6d7d2c9d21e8634b0
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] ood_per_element_reachability  metrics/by_metric/ood_per_element_reachability.json
        on-disk sha256 41a70f724c935c58b8731341ccc2fd58e650dd9a848d9776682a38de581105d5
        committed versions of that path: 1
        NEVER-COMMITTED sha256 8df5eceaabb397258133b3fe2b0f233b0a1c31db30fffbc75fcda7872fe3975e
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] ood_per_metal_reachability  metrics/by_metric/ood_per_metal_reachability.json
        on-disk sha256 31717c8a23ea6f3ed644f9dcde8865b96be40d293e65f491c8ec436c3808616e
        committed versions of that path: 2
        NEVER-COMMITTED sha256 ea59f36aec3de63c16f44c4bc5f912f235276277b939588abb4d9c9f58bd2703
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] platinai_combined_r2  metrics/by_metric/platinai_combined_r2.json
        on-disk sha256 3b15a0a2459fc89c8bfed7a83a07e8a3d0121866672256c062362793f847c152
        committed versions of that path: 4
        NEVER-COMMITTED sha256 21e1e71f8c64724ef0f90c697224443adb5dec2afe9fc55fc30230dd0bd73255
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] tmqm_combined_r2  metrics/by_metric/tmqm_combined_r2.json
        on-disk sha256 01a14d1c58649139bd2e762bfad189536c03628697b569e504975fb8c9d2d98a
        committed versions of that path: 2
        NEVER-COMMITTED sha256 0c1411ef5957c373f9977372d09e2429a1823a1a8ce7307a415d9db54fa5b1e7
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] vina_status  metrics/by_metric/vina_status.json
        on-disk sha256 645ba15c9b1d1b7a3a59fbd41a274f9d9ac07a9f269d6cda738f041747f4b159
        committed versions of that path: 1
        NEVER-COMMITTED sha256 09cee1ef037f17401c55d4080506607a27d93ea7dca68dc508296d7a441328d8
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [open] wfa155_ligand_connected_rate  metrics/by_metric/wfa155_ligand_connected_rate.json
        on-disk sha256 9ad45a495751c2046f37c8377a6dd118e0f93efc6ef8da58d5e7e3dcc31764de
        committed versions of that path: 0
        ledger  sha256 a60933e8281f808032ce5706a3e8ec174573e134918ac59952dc883d9a30d05e
        ledger value: None
    [open] wfa155_metal_coordination_sphere  metrics/by_metric/wfa155_metal_coordination_sphere.json
        on-disk sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b
        committed versions of that path: 0
        ledger  sha256 aa67ed49b645541032f99a14fa2f5414e52814c52f21383ae465f5c5083bd882
        ledger value: None
LEDGER PROVENANCE: WARN
rc=1
```

**POST-FIX, measured 2026-11-02 on this working tree:**

```
ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=NOT_MEASURED
    [open] wfa155_ligand_connected_rate  metrics/by_metric/wfa155_ligand_connected_rate.json
        on-disk sha256 9ad45a495751c2046f37c8377a6dd118e0f93efc6ef8da58d5e7e3dcc31764de
        committed versions of that path: 0
        ledger  sha256 a60933e8281f808032ce5706a3e8ec174573e134918ac59952dc883d9a30d05e
        ledger value: None
    [open] wfa155_metal_coordination_sphere  metrics/by_metric/wfa155_metal_coordination_sphere.json
        on-disk sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b
        committed versions of that path: 0
        ledger  sha256 aa67ed49b645541032f99a14fa2f5414e52814c52f21383ae465f5c5083bd882
        ledger value: None
LEDGER PROVENANCE: WARN
```

```
ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=9
    [superseded] dmpnn_ckpt_status  metrics/by_metric/dmpnn_ckpt_status.json
        on-disk sha256 b43f13849a3af614694c287008276c0da06223146abd8fc7ce281d59ccfccd09
        committed versions of that path: 3
        NEVER-COMMITTED sha256 a6db107ec57620bc3b04d367046740fc4a34fcc6476b2fba95ba98dbceaec932
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] l5_hilbert_r2  metrics/by_metric/hilbert_4cell_r2.json
        on-disk sha256 422a04b4321d2dcaa9ef9aee54b01d5b5963801adca2c0808c4c086f250f949e
        committed versions of that path: 8
        NEVER-COMMITTED sha256 d4079d7e0e1689b4577bebc0c451cb67d165a56ea5e792ae195d5c7509fc30d1
            (row value -6.712634861296495, git_sha '7e30d6ea509a8a42d953a77a166d69457ef7f037')
        superseded uncommitted rows for this metric: 1
    [superseded] l6_closure  metrics/by_metric/l6_closure.json
        on-disk sha256 82305a3bd789b36fc0c304d4563dc48dd030814b8fe77a5bb909c442053bc30b
        committed versions of that path: 2
        NEVER-COMMITTED sha256 49e13c420b74c5a64020cffc3207cb931877cefa010bfc3e952e109892209688
            (row value 0.5962, git_sha '8b9ceb52830f3203cf028ed12943fdcaff0e0ebb')
        superseded uncommitted rows for this metric: 1
    [superseded] metalcytotox_combined_r2  metrics/by_metric/metalcytotox_combined_r2.json
        on-disk sha256 50248f3d363172e4832bc52c36d3b42fc208f87ed33776eeacfff344fde58984
        committed versions of that path: 2
        NEVER-COMMITTED sha256 425e68518470c3a5ceec3a83d3ee68136c222b35c2aa1ee6d7d2c9d21e8634b0
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] ood_per_element_reachability  metrics/by_metric/ood_per_element_reachability.json
        on-disk sha256 41a70f724c935c58b8731341ccc2fd58e650dd9a848d9776682a38de581105d5
        committed versions of that path: 1
        NEVER-COMMITTED sha256 8df5eceaabb397258133b3fe2b0f233b0a1c31db30fffbc75fcda7872fe3975e
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] ood_per_metal_reachability  metrics/by_metric/ood_per_metal_reachability.json
        on-disk sha256 31717c8a23ea6f3ed644f9dcde8865b96be40d293e65f491c8ec436c3808616e
        committed versions of that path: 2
        NEVER-COMMITTED sha256 ea59f36aec3de63c16f44c4bc5f912f235276277b939588abb4d9c9f58bd2703
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] platinai_combined_r2  metrics/by_metric/platinai_combined_r2.json
        on-disk sha256 3b15a0a2459fc89c8bfed7a83a07e8a3d0121866672256c062362793f847c152
        committed versions of that path: 4
        NEVER-COMMITTED sha256 21e1e71f8c64724ef0f90c697224443adb5dec2afe9fc55fc30230dd0bd73255
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] tmqm_combined_r2  metrics/by_metric/tmqm_combined_r2.json
        on-disk sha256 01a14d1c58649139bd2e762bfad189536c03628697b569e504975fb8c9d2d98a
        committed versions of that path: 2
        NEVER-COMMITTED sha256 0c1411ef5957c373f9977372d09e2429a1823a1a8ce7307a415d9db54fa5b1e7
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [superseded] vina_status  metrics/by_metric/vina_status.json
        on-disk sha256 645ba15c9b1d1b7a3a59fbd41a274f9d9ac07a9f269d6cda738f041747f4b159
        committed versions of that path: 1
        NEVER-COMMITTED sha256 09cee1ef037f17401c55d4080506607a27d93ea7dca68dc508296d7a441328d8
            (row value None, git_sha 'd86c879a540e79ebefcda820e83ddff24f5477ad')
        superseded uncommitted rows for this metric: 1
    [open] wfa155_ligand_connected_rate  metrics/by_metric/wfa155_ligand_connected_rate.json
        on-disk sha256 9ad45a495751c2046f37c8377a6dd118e0f93efc6ef8da58d5e7e3dcc31764de
        committed versions of that path: 0
        ledger  sha256 a60933e8281f808032ce5706a3e8ec174573e134918ac59952dc883d9a30d05e
        ledger value: None
    [open] wfa155_metal_coordination_sphere  metrics/by_metric/wfa155_metal_coordination_sphere.json
        on-disk sha256 55034ba3e6376bb887c9543f5ea2550208f5294f332c3c76e1e4727ca5eeb94b
        committed versions of that path: 0
        ledger  sha256 aa67ed49b645541032f99a14fa2f5414e52814c52f21383ae465f5c5083bd882
        ledger value: None
LEDGER PROVENANCE: WARN
```

The default mode now prints the token `NOT_MEASURED` instead of a number it did not compute, and prints the hint on stderr.  The `--all` mode still prints the measured number.  **The pre-flight gate is untouched** -- it calls the function, not the CLI -- and still reports:

```
p3_run_ledger_provenance: ledger_rows=432 metrics=399 OPEN_never_committed=2 superseded_never_committed=9 OPEN wfa155_ligand_connected_rate sha a60933e8281f src metrics/by_metric/wfa155_ligand_connected_rate.json OPEN wfa155_metal_coordination_sphere sha aa67ed49b645 src metrics/by_metric/wfa155_metal_coordination_sphere.json
```

The gate's STATUS on this working tree is `FAIL` -- and it was ALREADY that before this wave.  The gate FAILS when the OPEN count is non-zero, and the OPEN rows are the two `wfa155_*` metric files (both UNTRACKED, see the table in section 3), whose recorded SHA matches neither the file on disk nor any committed version.  That FAIL is **PRE-EXISTING and NOT caused by this wave**: the Wf-A198 change touches the CLI's print statement only, and the gate calls the FUNCTION.  The superseded count the gate reports -- the very quantity the CLI used to fake -- is unchanged.

## 5. WHAT WAS NOT DONE (explicit)

- The 32 inconsistent rows were **NOT** rewritten.  The ledger is append-only; rewriting a row is not a repair, it is a destruction of history.
- **NO** pre-flight gate was added.
- `pre_flight_check.EXPECTED_GATE_COUNTS` was **NOT** changed.
- `molmetal/reports/run_ledger.jsonl` was **NOT** modified (sha256 unchanged).
- `registry/**`, `metrics/**`, `paper/**`, `docs/**`, `TODO/**` and `AGENTS.md` were **NOT** touched.
- **0 GPU**: every python invocation ran with `CUDA_VISIBLE_DEVICES=""` on CPU; no training, sampling, sweep or GPU-allocating gate was run.

## 6. HOW TO REPRODUCE

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python \
    molmetal/scripts/wfa198_ledger_pin_consistency.py --json
```

The auditor is read-only: it opens nothing for writing, and it shells out only to read-only git plumbing (`cat-file`, `log`, `ls-files`, `check-ignore`).  It is not registered as a gate and does not change `EXPECTED_GATE_COUNTS`.

## 7. FILES CHANGED BY THIS WAVE

- `molmetal/scripts/wfa198_ledger_pin_consistency.py` (NEW -- the auditor)
- `molmetal/tests/test_wfa198_ledger_pin_consistency.py` (NEW)
- `molmetal/reports/wfa198_ledger_pin_consistency/report.json` (NEW)
- `molmetal/reports/wfa198_ledger_pin_consistency/report.md` (NEW)
- `molmetal/scripts/run_ledger.py` (the CLI's DEFAULT-mode reporting ONLY)

## 8. VERIFICATION (measured at publish time)

**Test run (CPU-only, `CUDA_VISIBLE_DEVICES=""`, 0 GPU):**

```
$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
      molmetal/tests/test_wfa198_ledger_pin_consistency.py -q
16 passed, 1 warning in 20.78s
```

**The repo's own ledger test files (unchanged by this wave):**

```
$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
      molmetal/tests/test_run_ledger.py molmetal/tests/test_claim_ledger.py \
      molmetal/tests/test_curriculum_sample_ledger.py -q
2 failed, 48 passed, 1 warning in 29.82s
```

**PRE-EXISTING failures, NOT caused by this wave:** 2 in `molmetal/tests/test_claim_ledger.py` -- a JSON-pointer resolution failure in `molmetal/scripts/paper_metric_tables.py::resolve` (`KeyError: 'pools[0]'`, `KeyError: 'runs[3]'`).  That test file contains **zero** references to `run_ledger`, and this wave touched only the ledger CLI's DEFAULT-mode print string plus four NEW files. It is reported here as an honest negative, not as a pass.

```
FAILED molmetal/tests/test_claim_ledger.py::test_every_citation_resolves[c7_scale_matched_pools.json-/pools[0]/sa_counts_at_or_below/4.5]
FAILED molmetal/tests/test_claim_ledger.py::test_every_citation_resolves[cfm_fragmentation_audit.json-/runs[3]/pool_population_composition]
```

**The tests BITE.**  Each defect was put back and the new test file was re-run against it (the mutations were reverted immediately afterwards, and the files' sha256s were verified back to their pre-mutation values):

| mutation | result |
|---|---|
| the CLI prints the numeric count again (pre-fix behaviour) | `test_cli_default_mode_does_not_print_a_numeric_superseded_count` and `test_cli_guard_bites_on_the_preserved_pre_fix_rendering` FAIL with *"the default mode must print the token NOT_MEASURED for a quantity it did not measure"* and *"the post-fix default rendering must NOT parse as a number"* |
| the auditor never flags an inconsistency (`pin_consistent` forced True) | 5 tests FAIL, including *"the auditor did NOT flag the re-pointed row 0 (m14_decode_ratio) -- the checker does not discriminate"* |
| the auditor flags EVERY row (`pin_consistent` forced False) | 3 tests FAIL, including the dated-snapshot guard `pin_consistent == 400` |

**Baseline (AGENTS.md / rule 11):** `git status --porcelain metrics/ | md5sum` = `a022e54fd2938b9f333fd8438f6d3d27` -- identical before and after this wave.

**Ledger untouched:** `molmetal/reports/run_ledger.jsonl` sha256 = `3a5a8cd0690a0b854c5bfde6e6f73dd62f07e340be7699106b7e8098be135479` -- identical before and after.

**sha256 of every file this wave wrote (the two report files are excluded -- their own hash changes when they are written):**

```
fa82d0abeae20c94fed533f529bab94577ea8e5a7fb61fa8f86282eb5a9d5901  molmetal/scripts/wfa198_ledger_pin_consistency.py
07d5bb274240621ee5ce62406450fb1f8ecf62229aea6f56e3156feb0721df5c  molmetal/tests/test_wfa198_ledger_pin_consistency.py
42ee9db30832b4f996ae64e962081f46ae422ec53fb302f88b32f2f02e88b2ba  molmetal/scripts/run_ledger.py
```

